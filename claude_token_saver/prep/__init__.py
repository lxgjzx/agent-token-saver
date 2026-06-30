"""
Agent Token Saver - 预处理模块

负责在发送给 Claude 前精简输入内容。提供多级压缩策略：
  - skeleton: 仅符号索引（类/函数签名），~5-10% token（仅 Python）
  - stripped: 去除注释 + docstring，~30-50% token
  - full: 完整内容（默认，高信息密度）
  - block: 阻止读取（文件过大时）
"""
from __future__ import annotations

import ast
import hashlib
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock
from typing import Any

from claude_token_saver.compressor import extract_skeleton, extract_symbol_index, format_symbol_index
from claude_token_saver.utils import (
    count_tokens,
    get_file_hash,
    get_file_size,
    is_binary_file,
    should_ignore,
)


# ── 代码注释去除 ──

# 需要保留的重要注释关键词（去除其他注释时保留这些）
_IMPORTANT_COMMENT_PATTERN = re.compile(
    r'#.*?\b(?:TODO|FIXME|BUG|HACK|XXX|NOTE|IMPORTANT|WARNING|OPTIMIZE|DEPRECATED|NOQA)\b',
    re.IGNORECASE,
)

# 共享注释正则（减少 pattern 对象创建）
_C_COMMENT = re.compile(r'//.*?$|/\*.*?\*/', re.MULTILINE | re.DOTALL)
_SHARP_COMMENT = re.compile(r'#.*?$', re.MULTILINE)
_HTML_COMMENT = re.compile(r'<!--.*?-->', re.DOTALL)

_COMMENT_PATTERNS: dict[str, re.Pattern] = {
    ".py":   re.compile(r'#.*?$|""".*?"""|\'\'\'.*?\'\'\'', re.MULTILINE | re.DOTALL),
    ".js":   _C_COMMENT,
    ".ts":   _C_COMMENT,
    ".java": _C_COMMENT,
    ".c":    _C_COMMENT,
    ".cpp":  _C_COMMENT,
    ".h":    _C_COMMENT,
    ".go":   _C_COMMENT,
    ".rs":   _C_COMMENT,
    ".rb":   re.compile(r'#.*?$|=begin.*?=end', re.MULTILINE | re.DOTALL),
    ".php":  re.compile(r'//.*?$|#.*?$|/\*.*?\*/', re.MULTILINE | re.DOTALL),
    ".sh":   _SHARP_COMMENT,
    ".bash": _SHARP_COMMENT,
    ".yaml": _SHARP_COMMENT,
    ".yml":  _SHARP_COMMENT,
    ".toml": _SHARP_COMMENT,
    ".json": re.compile(r''),  # JSON 无注释
    ".xml":  _HTML_COMMENT,
    ".html": _HTML_COMMENT,
    ".css":  re.compile(r'/\*.*?\*/', re.DOTALL),
    ".sql":  re.compile(r'--.*?$|/\*.*?\*/', re.MULTILINE | re.DOTALL),
}


# ── 缓存 ────────────────────────────────────────────────────────────────

# 全局缓存：key = (file_path, mtime, size) → (content_hash, tokens)
_content_cache: dict[tuple, tuple[str, int]] = {}
# 全局缓存：key = content_hash → token_count
_token_cache: dict[str, int] = {}


def _cache_key(path: Path, content: str, *, stat_result=None) -> tuple:
    """生成缓存键：(路径, mtime, 大小)。

    Args:
        path: 文件路径
        content: 文件内容（仅 stat 失败时用于回退）
        stat_result: 可选的已获取 stat 结果，避免重复 syscall
    """
    try:
        if stat_result is None:
            stat_result = path.stat()
        return (str(path.resolve()), stat_result.st_mtime, stat_result.st_size)
    except OSError:
        return (str(path.resolve()), 0, len(content))


def is_binary_file_from_content(content: str) -> bool:
    """基于已读取的内容判断是否为二进制文件（检查空字节）。"""
    return "\x00" in content


def estimate_tokens_from_size(size_bytes: int, ext: str) -> int:
    """根据文件大小和扩展名快速估算 token 数量（不读取内容）。"""
    ext = ext.lower()
    if ext in {".json", ".yaml", ".yml", ".toml"}:
        ratio = 0.15
    elif ext in {".md", ".txt", ".rst"}:
        ratio = 0.12
    else:
        ratio = 0.125
    return max(1, int(size_bytes * ratio))


def _is_gitignored(path: Path) -> bool:
    """检查文件是否被 .gitignore 规则忽略。"""
    try:
        from claude_token_saver.gitignore import is_gitignored
        return is_gitignored(path)
    except Exception:
        return False


def _clear_caches() -> None:
    """清空所有缓存（用于测试或内存压力时）。"""
    _content_cache.clear()
    _token_cache.clear()
    # 同步清空压缩器缓存（AST、骨架）
    from claude_token_saver.compressor import clear_compressor_caches
    clear_compressor_caches()


def save_content_cache(cache_path: str | Path | None = None) -> None:
    """将内容哈希缓存持久化到磁盘（跨会话复用）。

    Args:
        cache_path: 缓存文件路径（默认使用平台缓存目录）
    """
    import json
    from pathlib import Path as _P

    if cache_path is None:
        cache_dir = _P(__import__("tempfile").gettempdir()) / "agent-token-saver"
        cache_dir.mkdir(exist_ok=True)
        cache_path = cache_dir / "content_cache.json"

    # 只保存可序列化的部分：(str(path), int(mtime), int(size)) → (content_hash, token_count)
    serializable = {}
    for key, value in _content_cache.items():
        try:
            serializable[str(key)] = value
        except Exception:
            continue

    try:
        _P(cache_path).write_text(
            json.dumps(serializable, ensure_ascii=False),
            encoding="utf-8",
        )
    except Exception:
        pass


def load_content_cache(cache_path: str | Path | None = None) -> int:
    """从磁盘加载内容哈希缓存。

    Returns:
        加载的缓存条目数
    """
    import json
    from pathlib import Path as _P

    if cache_path is None:
        cache_dir = _P(__import__("tempfile").gettempdir()) / "agent-token-saver"
        cache_path = cache_dir / "content_cache.json"

    try:
        data = json.loads(_P(cache_path).read_text(encoding="utf-8"))
        count = 0
        for key_str, value in data.items():
            try:
                # 反序列化 key: "(path, mtime, size)"
                key = eval(key_str)  # noqa: S307 - controlled cache key format
                _content_cache[key] = value
                count += 1
            except Exception:
                continue
        return count
    except Exception:
        return 0


def iter_file_content(file_paths: list[str | Path]) -> "GeneratorIterator":
    """流式迭代文件内容，减少内存占用。

    对于大型文件列表，逐文件读取内容而非一次性加载全部。

    Yields:
        (path, content, tokens_before)
    """
    for fp in file_paths:
        fp_path = Path(fp)
        if not fp_path.is_file():
            continue
        try:
            content = fp_path.read_text(encoding="utf-8", errors="replace")
            yield str(fp_path), content, count_tokens(content)
        except Exception:
            continue


# 为类型提示注册 GeneratorIterator
try:
    from typing import Generator
    GeneratorIterator = Generator[tuple[str, str, int], None, None]
except ImportError:
    GeneratorIterator = Any  # type: ignore[misc,assignment]


# ── 代码注释去除 ──

def strip_comments(content: str, ext: str) -> str:
    """去除文件中的注释内容，保留重要标记（TODO/FIXME/BUG/NOTE 等）。

    注意：非 Python 文件使用正则表达式，可能误伤字符串内的注释符号。
    Python 文件使用 AST 保护字符串区域。
    """
    if ext.lower() == ".py":
        return _strip_python_comments_safe(content)

    pattern = _COMMENT_PATTERNS.get(ext.lower())
    if not pattern:
        return content

    # 提取并保留重要注释（TODO/FIXME 等）
    important_comments = _IMPORTANT_COMMENT_PATTERN.findall(content)

    result = pattern.sub("", content)

    # 重新插入重要注释（在结果末尾追加，避免位置错误）
    if important_comments:
        result = _reinsert_important_comments(result, important_comments)

    return result


def _strip_python_comments_safe(content: str) -> str:
    """安全去除 Python 注释：使用 AST 识别字符串区域，避免误伤。

    同时保留重要注释（TODO/FIXME/BUG 等）。
    """
    # 提取并保留重要注释
    important_comments = _IMPORTANT_COMMENT_PATTERN.findall(content)

    try:
        import ast as _ast
        tree = _ast.parse(content)
    except SyntaxError:
        # 回退到正则
        pattern = _COMMENT_PATTERNS.get(".py")
        result = pattern.sub("", content) if pattern else content
        # 重新插入重要注释
        if important_comments:
            result = _reinsert_important_comments(result, important_comments)
        return result

    lines = content.split("\n")
    string_ranges: list[tuple[int, int]] = []

    for node in _ast.walk(tree):
        if isinstance(node, _ast.Constant) and isinstance(node.value, str):
            if hasattr(node, "lineno") and hasattr(node, "end_lineno"):
                start = node.lineno - 1
                end = node.end_lineno
                string_ranges.append((start, end))

    # 标记受保护的字符范围（仅完全在多行字符串内的行需要跳过）
    protected: set[int] = set()
    for start, end in string_ranges:
        # 仅跳过完全在多行字符串内部的中间行
        # （首行和末行由字符级解析器安全处理）
        for i in range(start + 1, min(end, len(lines))):
            protected.add(i)

    # 逐行处理，跳过完全在多行字符串内的行
    result_lines = []
    for line_idx, line in enumerate(lines):
        if line_idx in protected:
            result_lines.append(line)
            continue
        # 去除行内 # 注释（忽略字符串内的 #）
        in_str = False
        str_char = ""
        result = []
        i = 0
        while i < len(line):
            ch = line[i]
            if not in_str and ch in ('"', "'"):
                # 检查转义
                if i > 0 and line[i - 1] == "\\":
                    result.append(ch)
                    i += 1
                    continue
                in_str = True
                str_char = ch
                result.append(ch)
            elif in_str and ch == str_char:
                if i > 0 and line[i - 1] == "\\":
                    result.append(ch)
                    i += 1
                    continue
                in_str = False
                str_char = ""
                result.append(ch)
            elif not in_str and ch == "#":
                break
            else:
                result.append(ch)
            i += 1
        result_lines.append("".join(result))

    result = "\n".join(result_lines)

    # 重新插入重要注释
    if important_comments:
        result = _reinsert_important_comments(result, important_comments)

    return result


def _reinsert_important_comments(result: str, important_comments: list[str]) -> str:
    """将重要注释重新插入到结果末尾。"""
    seen = set()
    unique_important = []
    for c in important_comments:
        stripped = c.strip()
        if stripped and stripped not in seen:
            seen.add(stripped)
            unique_important.append(stripped)
    if unique_important:
        result = result.rstrip() + "\n\n# " + "\n# ".join(unique_important) + "\n"
    return result


def _remove_future_imports(content: str) -> str:
    """去除 __future__ import 声明（标准库特性，无项目特定信息）。"""
    return re.sub(
        r'^(?:from __future__ import.*|import __future__)(?:\n\n?)?',
        '',
        content,
        count=1,
        flags=re.MULTILINE,
    ).lstrip('\n')


def strip_python_docstrings(content: str) -> str:
    """去除 Python 文件的文档字符串（保留代码逻辑注释）。"""
    try:
        import ast as _ast
        tree = _ast.parse(content)
        lines = content.split("\n")
        docstring_ranges: list[tuple[int, int]] = []

        for node in _ast.walk(tree):
            if not isinstance(node, (_ast.FunctionDef, _ast.AsyncFunctionDef, _ast.ClassDef, _ast.Module)):
                continue
            if not hasattr(node, "body") or not node.body:
                continue
            first_stmt = node.body[0]
            if not isinstance(first_stmt, _ast.Expr):
                continue
            # 检查是否为字符串字面量（docstring）
            if not (hasattr(first_stmt, "value") and isinstance(first_stmt.value, _ast.Constant)
                    and isinstance(first_stmt.value.value, str)):
                continue
            if hasattr(first_stmt, "end_lineno"):
                start = first_stmt.lineno - 1
                end = first_stmt.end_lineno
                docstring_ranges.append((start, end))

        # 从后往前替换，避免行号偏移
        for start, end in sorted(docstring_ranges, reverse=True):
            lines[start:end] = [""] * (end - start)

        return "\n".join(lines)
    except SyntaxError:
        return content


# ── 智能截断 ──

def smart_truncate(content: str, max_tokens: int) -> str:
    """基于 token 预算的智能截断，优先保留头和尾。

    使用二分查找找到最大的前 keep_tokens + 后 keep_tokens 的组合，
    确保总 token 数不超过 max_tokens。
    """
    total = count_tokens(content)
    if not max_tokens or total <= max_tokens:
        return content

    head_budget = int(max_tokens * 0.65)
    tail_budget = int(max_tokens * 0.30)
    # 为省略标记预留 tokens
    marker_tokens = count_tokens("\n\n... [...] ...\n\n")
    head_budget = max(head_budget, max_tokens - tail_budget - marker_tokens)
    tail_budget = max_tokens - head_budget - marker_tokens

    lines = content.split("\n")

    # 二分查找：找到满足 token 预算的最大行数
    head_lines = _find_lines_for_budget(lines, head_budget, from_start=True)
    tail_lines = _find_lines_for_budget(lines, tail_budget, from_start=False)

    # 确保头和尾不重叠
    if head_lines + tail_lines >= len(lines):
        keep = max(1, int(len(lines) * 0.7))
        result = "\n".join(lines[:keep])
        result += f"\n\n... [已省略 {len(lines) - keep} 行，共 {len(lines)} 行] ...\n"
        return result

    head = lines[:head_lines]
    tail = lines[-tail_lines:]
    omitted = len(lines) - head_lines - tail_lines

    result = "\n".join(head)
    result += f"\n\n... [已省略 {omitted} 行（第 {head_lines + 1} - 第 {len(lines) - tail_lines} 行），共 {len(lines)} 行] ...\n\n"
    result += "\n".join(tail)
    return result


def _find_lines_for_budget(lines: list[str], token_budget: int, from_start: bool) -> int:
    """二分查找：找到满足 token 预算的最大行数（使用前缀和优化）。"""
    if not lines:
        return 0
    subset = lines if from_start else list(reversed(lines))
    # 预计算每行 token 数 + 前缀和（含行间 \n 的 1 token）
    line_tokens = [count_tokens(line) + 1 for line in subset]  # +1 for "\n" join
    prefix = [0]
    for t in line_tokens:
        prefix.append(prefix[-1] + t)

    lo, hi = 0, len(subset)
    best = 0
    while lo < hi:
        mid = (lo + hi) // 2
        if prefix[mid] <= token_budget:
            best = mid
            lo = mid + 1
        else:
            hi = mid

    return best


# ── 去重 ──

def deduplicate_files(file_paths: list[str | Path]) -> list[str | Path]:
    """去除内容完全重复的文件，只保留第一个。"""
    seen_hashes: set[str] = set()
    unique: list[str | Path] = []

    for fp in file_paths:
        try:
            h = get_file_hash(fp)
            if h not in seen_hashes:
                seen_hashes.add(h)
                unique.append(fp)
        except Exception:
            unique.append(fp)  # 读不了的文件保留

    return unique


def group_similar_files(file_paths: list[str | Path], threshold: float = 0.8) -> list[list[str | Path]]:
    """按内容相似度分组文件（简易实现：按 hash 分桶）。"""
    hash_buckets: dict[str, list[str | Path]] = {}
    for fp in file_paths:
        try:
            h = get_file_hash(fp)
            hash_buckets.setdefault(h, []).append(fp)
        except Exception:
            hash_buckets.setdefault("_error", []).append(fp)
    return list(hash_buckets.values())


# ── Prompt 压缩 ──

def compress_prompt(text: str, max_tokens: int = 10_000) -> str:
    """
    压缩 prompt 文本：
    - 去除多余空白行
    - 合并连续空白字符
    - 去除注释（如果检测到代码块）
    - 截断超长部分
    """
    # 空白标准化
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = text.strip()

    # 尝试使用管线进行更深层压缩（代码块 → 去注释 + 截断）
    try:
        from claude_token_saver.compression_pipeline import CompressionPipeline
        if "```" in text or "def " in text or "class " in text:
            pipeline = CompressionPipeline(ext=".py", detail_level="stripped", max_tokens=max_tokens)
            compressed, meta = pipeline.run(text)
            # 只在管线确实压缩了内容且不超过预算时使用
            if meta["total_savings"] > 0 and count_tokens(compressed) <= max_tokens * 1.2:
                return compressed
    except Exception:
        pass

    if count_tokens(text) <= max_tokens:
        return text

    # fallback：截断到 max_tokens
    try:
        from claude_token_saver.prep import smart_truncate
        return smart_truncate(text, max_tokens)
    except Exception:
        lines = text.split("\n")
        keep = int(len(lines) * max_tokens / max(count_tokens(text), 1))
        keep = max(keep, 1)
        result = "\n".join(lines[:keep])
        result += f"\n\n... [内容已压缩，原始 {len(lines)} 行] ...\n"
        return result


# ── 主入口 ────────────────────────────────────────────────────────────

def process_files(
    file_paths: list[str | Path],
    do_strip_comments: bool = True,
    do_strip_docstrings: bool = False,
    max_file_tokens: int = 50_000,
    dedup: bool = True,
    include_binary: bool = False,
    detail_level: str = "full",
    token_cache_enabled: bool = True,
    auto_detail: bool = False,
    token_budget: int | None = None,
    structural_dedup: bool = False,
    common_dedup: bool = False,
    near_dedup: bool = False,
    parallel_workers: int = 0,  # 0 = 自动（CPU 核心数），1 = 禁用并行
) -> dict:
    """
    处理文件列表，返回精简后的结果。

    detail_level 控制输出粒度：
      - "skeleton": 仅符号索引（类/函数签名），~5-10% token（仅 Python）
      - "stripped": 去除注释和 docstring，~30-50% token
      - "full": 完整内容（默认）
      - "block": 对超大文件返回阻止标记，不读取内容

    auto_detail: 根据 token_budget 自动为每个文件分配最优 detail_level

    structural_dedup: 基于代码结构相似度去重（超越 MD5）

    Returns:
        {
            "files": [...],
            "total_tokens_before": int,
            "total_tokens_after": int,
            "savings_pct": float,
            "skipped": [...],
            "duplicates_removed": int,
            "cache_hits": int,
        }
    """
    # ── 结构去重 ────────────────────────────────────────────────────────
    if structural_dedup:
        file_paths = list(file_paths)
        from claude_token_saver.compressor import structural_dedup as _structural_dedup
        file_paths = _structural_dedup(file_paths)
        # 重置去重计数器（MD5 去重不再需要）
        md5_dedup = dedup
        dedup = False
    else:
        md5_dedup = dedup

    if md5_dedup:
        file_paths = deduplicate_files(list(file_paths))

    # ── 常见文件组去重 ───────────────────────────────────────────────────
    if common_dedup:
        from claude_token_saver.common_dedup import filter_common_duplicates
        file_paths, _ = filter_common_duplicates(list(file_paths))

    # ── 智能 conftest 去重（始终开启，Python 项目）────────────────────────
    file_paths_list = list(file_paths)
    from claude_token_saver.common_dedup import dedup_conftest_always
    file_paths_list, _conftest_skipped = dedup_conftest_always(file_paths_list)
    file_paths = file_paths_list

    # ── .pyi 与 .py 去重（有对应 .py 的 .pyi 跳过）────────────────────────
    file_paths_list = list(file_paths)
    _py_files = {str(Path(fp).resolve()) for fp in file_paths_list if Path(fp).suffix == ".py"}
    _pyi_skipped = 0
    if _py_files:
        filtered = []
        for fp in file_paths_list:
            fp_path = Path(fp)
            if fp_path.suffix == ".pyi":
                corresponding_py = str(fp_path.with_suffix(".py").resolve())
                if corresponding_py in _py_files:
                    _pyi_skipped += 1
                    continue
            filtered.append(fp)
        file_paths_list = filtered
    file_paths = file_paths_list

    # ── 近似重复检测 ─────────────────────────────────────────────────────
    dup_removed = 0
    if near_dedup:
        from claude_token_saver.simhash_dedup import find_near_duplicates
        groups = find_near_duplicates(list(file_paths), threshold=3)
        to_remove = set()
        for g in groups:
            for dup in g.duplicates:
                to_remove.add(dup)
        file_paths = [fp for fp in file_paths if str(fp) not in to_remove]
        dup_removed += len(to_remove)

    # ── 自适应 detail_level ─────────────────────────────────────────────
    per_file_levels: dict[str, str] = {}
    if auto_detail:
        budget = token_budget or 50_000
        file_token_list: list[tuple[str, int]] = []
        for fp in file_paths:
            fp_path = Path(fp)
            if fp_path.is_file() and not should_ignore(fp_path, include_binary=include_binary):
                try:
                    size = fp_path.stat().st_size
                    tokens = estimate_tokens_from_size(size, fp_path.suffix)
                    file_token_list.append((str(fp), tokens))
                except OSError:
                    pass

        from claude_token_saver.budget import auto_detail_level
        per_file_levels = auto_detail_level(file_token_list, budget)

    results = []
    total_before = 0
    total_after = 0
    skipped = []
    cache_hits = 0
    seen_hashes: set[str] = set()
    _seen_lock = Lock()
    _cache_lock = Lock()
    _result_lock = Lock()

    # 并行处理：收集通过过滤的文件，批量处理
    if parallel_workers != 1:
        return _process_files_parallel(
            file_paths, file_paths,  # original list preserved
            do_strip_comments, do_strip_docstrings, max_file_tokens,
            dedup, include_binary, detail_level, token_cache_enabled,
            auto_detail, token_budget, structural_dedup, common_dedup, near_dedup,
            per_file_levels, seen_hashes, _seen_lock, _cache_lock, _result_lock,
            parallel_workers,
        )

    for fp in file_paths:
        fp_path = Path(fp)
        if should_ignore(fp_path, include_binary=include_binary):
            skipped.append(str(fp_path))
            continue

        # .gitignore 过滤
        try:
            if _is_gitignored(fp_path):
                skipped.append(f"{fp_path} (.gitignore)")
                continue
        except Exception:
            pass

        # 单次 I/O：读取内容（同时用于缓存检查、去重、处理）
        try:
            content = fp_path.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            skipped.append(f"{fp} ({e})")
            continue

        # 二进制检查（基于已读取的内容）
        if not include_binary and is_binary_file_from_content(content):
            skipped.append(f"{fp_path} (binary)")
            continue

        # 自适应 detail_level 决定
        if auto_detail and str(fp) in per_file_levels:
            detail_level = per_file_levels[str(fp)]
            if detail_level == "block":
                skipped.append(f"{fp_path} (超出预算，建议 --offset/--limit)")
                continue

        tokens_before = count_tokens(content)
        content_hash = hashlib.md5(content.encode()).hexdigest()

        # 缓存检查（复用已读取的内容，避免重复 stat）
        cache_key = None
        if token_cache_enabled:
            try:
                _stat = fp_path.stat()
            except OSError:
                _stat = None
            cache_key = _cache_key(fp_path, content, stat_result=_stat)
            if cache_key in _content_cache:
                cached_hash, cached_tokens = _content_cache[cache_key]
                if content_hash == cached_hash and detail_level == "full":
                    total_before += cached_tokens
                    total_after += cached_tokens
                    results.append({
                        "path": str(fp_path),
                        "tokens_before": cached_tokens,
                        "tokens_after": cached_tokens,
                        "savings": 0,
                        "content": content,
                        "detail_level": "full (cached)",
                    })
                    cache_hits += 1
                    continue

        # 去重
        if content_hash in seen_hashes:
            dup_removed += 1
            continue
        seen_hashes.add(content_hash)

        # 更新缓存（仅 full 级别）
        if token_cache_enabled and cache_key and detail_level == "full":
            _content_cache[cache_key] = (content_hash, tokens_before)

        total_before += tokens_before

        # 处理阶段（含异常捕获）
        try:
            # detail_level 路由
            if detail_level == "block" and tokens_before > max_file_tokens:
                skipped.append(f"{fp_path} (too large: {tokens_before} tokens, use --offset/--limit)")
                continue

            processed_content = _process_content(
                content, fp_path, detail_level, do_strip_comments, do_strip_docstrings, max_file_tokens
            )

            tokens_after = count_tokens(processed_content)
            total_after += tokens_after

            results.append({
                "path": str(fp_path),
                "tokens_before": tokens_before,
                "tokens_after": tokens_after,
                "savings": tokens_before - tokens_after,
                "content": processed_content,
                "detail_level": detail_level,
            })
        except Exception as e:
            import logging
            logging.getLogger("claude_token_saver.prep").warning("处理文件失败 %s: %s", fp, e)
            skipped.append(f"{fp} ({e})")

    savings_pct = ((total_before - total_after) / total_before * 100) if total_before else 0

    return {
        "files": results,
        "total_tokens_before": total_before,
        "total_tokens_after": total_after,
        "savings_pct": round(savings_pct, 1),
        "skipped": skipped,
        "duplicates_removed": dup_removed,
        "cache_hits": cache_hits,
    }


def _process_files_parallel(
    file_paths: list,
    original_paths: list,
    do_strip_comments: bool,
    do_strip_docstrings: bool,
    max_file_tokens: int,
    dedup: bool,
    include_binary: bool,
    detail_level: str,
    token_cache_enabled: bool,
    auto_detail: bool,
    token_budget: int | None,
    structural_dedup: bool,
    common_dedup: bool,
    near_dedup: bool,
    per_file_levels: dict,
    seen_hashes: set,
    seen_lock: Lock,
    cache_lock: Lock,
    result_lock: Lock,
    max_workers: int,
) -> dict:
    """并行版本的文件处理。"""
    import os
    from concurrent.futures import ThreadPoolExecutor, as_completed

    results = []
    skipped = []
    dup_removed = 0
    cache_hits = 0
    total_before = 0
    total_after = 0

    # 智能 conftest 去重（始终开启）
    from claude_token_saver.common_dedup import dedup_conftest_always
    file_paths, _ = dedup_conftest_always(list(file_paths))

    # 第一阶段：过滤 + 读取（并行）
    tasks = []
    for fp in file_paths:
        fp_path = Path(fp)
        if should_ignore(fp_path, include_binary=include_binary):
            skipped.append(str(fp_path))
            continue

        try:
            if _is_gitignored(fp_path):
                skipped.append(f"{fp_path} (.gitignore)")
                continue
        except Exception:
            pass

        tasks.append(fp)

    # 并行读取和处理
    if max_workers <= 0:
        max_workers = min(32, (os.cpu_count() or 1) + 4)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_fp = {}
        for fp in tasks:
            future = executor.submit(_read_and_process_file, fp,
                include_binary, auto_detail, per_file_levels, detail_level)
            future_to_fp[future] = fp

        for future in as_completed(future_to_fp):
            fp = future_to_fp[future]
            try:
                result = future.result()
                if result is None:
                    continue

                if result.get("skip"):
                    skipped.append(result["skip"])
                    continue

                # 线程安全地更新共享状态
                with seen_lock:
                    content_hash = result["content_hash"]
                    if content_hash in seen_hashes:
                        dup_removed += 1
                        continue
                    seen_hashes.add(content_hash)

                with cache_lock:
                    if token_cache_enabled and result.get("cache_key"):
                        cache_key = result["cache_key"]
                        if cache_key in _content_cache:
                            cached_hash, cached_tokens = _content_cache[cache_key]
                            if content_hash == cached_hash and result.get("detail_level") == "full":
                                total_before += cached_tokens
                                total_after += cached_tokens
                                with result_lock:
                                    results.append({
                                        "path": result["path"],
                                        "tokens_before": cached_tokens,
                                        "tokens_after": cached_tokens,
                                        "savings": 0,
                                        "content": result["content"],
                                        "detail_level": "full (cached)",
                                    })
                                cache_hits += 1
                                continue

                # 处理内容
                try:
                    processed_content = _process_content(
                        result["content"], Path(fp),
                        result.get("detail_level", detail_level),
                        do_strip_comments, do_strip_docstrings, max_file_tokens
                    )
                    tokens_after = count_tokens(processed_content)
                    total_before += result["tokens_before"]
                    total_after += tokens_after
                    with result_lock:
                        results.append({
                            "path": result["path"],
                            "tokens_before": result["tokens_before"],
                            "tokens_after": tokens_after,
                            "savings": result["tokens_before"] - tokens_after,
                            "content": processed_content,
                            "detail_level": result.get("detail_level", detail_level),
                        })
                    # 更新缓存（仅 full 级别）
                    if token_cache_enabled and result.get("cache_key") and result.get("detail_level") == "full":
                        with cache_lock:
                            _content_cache[result["cache_key"]] = (content_hash, result["tokens_before"])
                except Exception as e:
                    skipped.append(f"{fp} ({e})")

            except Exception as e:
                skipped.append(f"{fp} ({e})")

    # 恢复原始顺序
    path_order = {str(p): i for i, p in enumerate(original_paths)}
    results.sort(key=lambda r: path_order.get(r["path"], 999))

    savings_pct = ((total_before - total_after) / total_before * 100) if total_before else 0
    return {
        "files": results,
        "total_tokens_before": total_before,
        "total_tokens_after": total_after,
        "savings_pct": round(savings_pct, 1),
        "skipped": skipped,
        "duplicates_removed": dup_removed,
        "cache_hits": cache_hits,
    }


def _read_and_process_file(fp, include_binary, auto_detail, per_file_levels, default_detail_level, max_file_tokens=50_000):
    """读取单个文件并返回初步处理结果（用于并行）。"""
    fp_path = Path(fp)

    # 读取内容
    try:
        content = fp_path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return {"skip": f"{fp} ({e})"}

    # 二进制检查
    if not include_binary and is_binary_file_from_content(content):
        return {"skip": f"{fp_path} (binary)"}

    # detail_level 决定
    detail_level = default_detail_level
    if auto_detail and str(fp) in per_file_levels:
        detail_level = per_file_levels[str(fp)]

    tokens_before = count_tokens(content)

    # block 级别：跳过超大文件
    if detail_level == "block" and tokens_before > max_file_tokens:
        return {"skip": f"{fp_path} (too large: {tokens_before} tokens, use --offset/--limit)"}

    content_hash = hashlib.md5(content.encode()).hexdigest()

    # 缓存 key（复用已读取的内容）
    cache_key = None
    try:
        try:
            _stat = fp_path.stat()
        except OSError:
            _stat = None
        cache_key = _cache_key(fp_path, content, stat_result=_stat)
    except Exception:
        pass

    return {
        "path": str(fp_path),
        "content": content,
        "tokens_before": tokens_before,
        "content_hash": content_hash,
        "cache_key": cache_key,
        "detail_level": detail_level,
    }


def _apply_python_line_transforms(content: str, detail_level: str) -> str:
    """一次性应用所有 Python 行级变换（减少 split/join 次数）。

    仅处理行级操作：冗余 pass、空类体、空块、assert 压缩、raise from None。
    AST 级操作（死代码、main guard、self 赋值、函数内联）单独处理。
    """
    if detail_level not in ("skeleton", "stripped"):
        return content

    lines = content.split("\n")
    result = []
    i = 0
    add_marker = detail_level == "skeleton"  # stripped 级别不添加标记（节省 token）

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # 压缩空类体：class Foo:\n    pass → class Foo: ... [# empty]
        if stripped.startswith("class ") and stripped.endswith(":") and i + 1 < len(lines):
            next_stripped = lines[i + 1].strip()
            if next_stripped in ("pass", "..."):
                if add_marker:
                    result.append(line + " ...  # empty")
                else:
                    result.append(line + " ...")
                i += 2
                continue

        # 压缩空块体：def/if/for/while/try: + pass/... → def foo(): ... [# empty]
        if stripped.endswith(":") and i + 1 < len(lines):
            next_stripped = lines[i + 1].strip()
            if next_stripped in ("pass", "..."):
                if add_marker:
                    result.append(line + " " + next_stripped + "  # empty")
                else:
                    result.append(line + " " + next_stripped)
                i += 2
                continue

        # 去除冗余 pass
        if stripped == "pass" and _is_redundant_pass(lines, i):
            i += 1
            continue

        # 压缩 assert 语句（仅 stripped 级别）
        if detail_level == "stripped" and stripped.startswith("assert "):
            line = _compress_single_assert(line)

        # 去除 raise ... from None
        if stripped.startswith("raise ") and " from None" in stripped:
            line = line.replace(" from None", "")

        result.append(line)
        i += 1

    return "\n".join(result)


def _remove_type_annotations(content: str) -> str:
    """去除 Python 类型注解（stripped 模式）。

    使用 AST 定位注解位置，行级替换确保签名完整性。
    去除：
    - 函数返回类型：def foo() -> int → def foo()
    - 函数参数类型：def foo(x: int) → def foo(x)
    - 变量注解：x: int = 5 → x = 5
    - 独立变量注解：x: int → （整行删除）

    策略：单行函数签名用 AST 重建；AnnAssign 用行级正则。
    """
    try:
        import ast as _ast
        tree = _ast.parse(content)
    except SyntaxError:
        return content

    lines = content.split("\n")
    if not lines:
        return content

    # 收集行级编辑
    edits: dict[int, str] = {}  # line_idx -> replaced_line

    # ── 函数/方法签名 ──
    for node in _ast.walk(tree):
        if not isinstance(node, (_ast.FunctionDef, _ast.AsyncFunctionDef)):
            continue
        line_idx = node.lineno - 1
        if line_idx >= len(lines):
            continue
        orig_line = lines[line_idx]
        indent = orig_line[:len(orig_line) - len(orig_line.lstrip())]

        # 收集参数名（跳过所有注解），保留 * 和 ** 前缀
        args = []
        for arg in node.args.posonlyargs:
            args.append(arg.arg)
        for arg in node.args.args:
            args.append(arg.arg)
        # *args
        if node.args.vararg:
            args.append("*" + node.args.vararg.arg)
        # 关键字参数
        for arg in node.args.kwonlyargs:
            args.append(arg.arg)
        # **kwargs
        if node.args.kwarg:
            args.append("**" + node.args.kwarg.arg)

        # kwonly 分隔符：如果没有 *args 但有 kwonly args，需要 *, 分隔符
        has_kwonly = bool(node.args.kwonlyargs)
        has_vararg = bool(node.args.vararg)
        has_positional = bool(node.args.posonlyargs or node.args.args)
        # / 分隔符（posonly 和 regular 之间）
        has_slash = bool(node.args.posonlyargs) and has_positional
        if has_kwonly and not has_vararg:
            first_kwonly_idx = (
                len(node.args.posonlyargs) + len(node.args.args) +
                (1 if has_slash else 0)
            )
            if first_kwonly_idx < len(args):
                args[first_kwonly_idx] = "*, " + args[first_kwonly_idx]
            elif not args:
                args = ["*,"]

        # 确定签名是否跨越多行
        sig_end_line = line_idx
        if hasattr(node, 'body') and node.body:
            sig_end_line = node.body[0].lineno - 2
            if sig_end_line < line_idx:
                sig_end_line = line_idx

        if sig_end_line == line_idx:
            # 单行签名：用 AST 重建（去除类型注解和返回类型）
            prefix = "async " if isinstance(node, _ast.AsyncFunctionDef) else ""
            parts = []
            pi = 0
            if node.args.posonlyargs:
                parts.append(", ".join(args[pi:pi + len(node.args.posonlyargs)]))
                pi += len(node.args.posonlyargs)
            if has_slash:
                parts.append("/")
            if node.args.args:
                parts.append(", ".join(args[pi:pi + len(node.args.args)]))
                pi += len(node.args.args)
            remaining = args[pi:]
            if remaining:
                parts.append(", ".join(remaining))
            sig = f"{prefix}def {node.name}({', '.join(parts)})"
            sig += ":"
            edits[line_idx] = indent + sig
        else:
            # 多行签名：逐行用正则清理（保留缩进）
            for li in range(line_idx, sig_end_line + 1):
                if li >= len(lines):
                    continue
                orig = lines[li]
                li_indent = orig[:len(orig) - len(orig.lstrip())]
                l = orig
                l = re.sub(r'\s*->\s*[^:]+', '', l)
                l = re.sub(r'(\w+)\s*:\s*[^,)]+,?', lambda m: m.group(1) + (',' if m.group(0).rstrip().endswith(',') else '') , l)
                l = re.sub(r',\s*\w+\s*:\s*[^,)]+,?', '', l)
                l = re.sub(r'  +', ' ', l)
                if l != orig:
                    edits[li] = li_indent + l.lstrip()

    # ── 变量注解 ──
    for node in _ast.walk(tree):
        if not isinstance(node, _ast.AnnAssign):
            continue
        line_idx = node.lineno - 1
        if line_idx >= len(lines):
            continue
        line = lines[line_idx]

        if node.value:
            # x: int = 5 → x = 5
            new_line = re.sub(r'\s*:\s*[^=]+(?=\s*=)', ' ', line)
            edits[line_idx] = new_line.rstrip()
        else:
            # x: int → 删除整行
            edits[line_idx] = ""

    if not edits:
        return content

    # 从后往前应用编辑（避免行号偏移）
    modified = list(lines)
    for line_idx in sorted(edits.keys(), reverse=True):
        if line_idx < len(modified):
            modified[line_idx] = edits[line_idx]

    return "\n".join(modified)

def _compress_single_assert(line: str) -> str:
    """压缩单条 assert 语句。"""
    stripped = line.strip()
    if " is not None" in stripped:
        return line.replace(" is not None", "")
    if " is None" in stripped:
        return line.replace(" is None", " is False")
    if "len(" in stripped and "> 0" in stripped:
        return re.sub(r'assert len\(([^)]+)\) > 0', r'assert \1', line)
    if "len(" in stripped and "!= 0" in stripped:
        return re.sub(r'assert len\(([^)]+)\) != 0', r'assert \1', line)
    if "bool(" in stripped:
        return re.sub(r'assert bool\(([^)]+)\)', r'assert \1', line)
    return line


def _process_content(
    content: str,
    fp: Path,
    detail_level: str,
    do_strip_comments: bool,
    do_strip_docstrings: bool,
    max_tokens: int,
) -> str:
    """根据 detail_level 处理单个文件内容。"""
    ext = fp.suffix.lower()

    # 空白标准化（在去除注释之前执行，使后续处理更可靠）
    # 对所有文本文件应用基础空白标准化（不仅是 .py/.yaml/.yml/.toml）
    if ext not in {".json", ".csv", ".bin"}:
        content = normalize_whitespace(content, ext)

    if detail_level == "skeleton" and ext in {".py", ".js", ".ts", ".java", ".c", ".cpp", ".h", ".go", ".rs"}:
        # 快速跳过：如果注释密度过高，AST 解析节省有限
        if ext == ".py" and _comment_density(content, ext) > 0.7:
            detail_level = "stripped"
        else:
            # 最高压缩：符号索引 + 骨架
            symbols = extract_symbol_index(content, ext)
            content_tokens = count_tokens(content)  # 共享 token 计数
            if symbols:
                skeleton = extract_skeleton(content, ext)
                skeleton_tokens = count_tokens(skeleton)
                # 如果骨架本身不节省 token，回退到 stripped
                if skeleton_tokens >= content_tokens:
                    detail_level = "stripped"
                else:
                    header = format_symbol_index(symbols, str(fp))
                    combined = header + "\n\n" + skeleton if header else skeleton
                    combined_tokens = count_tokens(combined)
                    # 只在 combined 更优时才带上 header
                    if combined_tokens < skeleton_tokens:
                        return _compress_empty_bodies(combined)
                    return _compress_empty_bodies(skeleton)
            else:
                # 无符号索引时直接提取骨架（JS/TS/C-like 等）
                skeleton = extract_skeleton(content, ext)
                if count_tokens(skeleton) < content_tokens:
                    return _compress_empty_bodies(skeleton)
                detail_level = "stripped"

    if detail_level in ("skeleton", "stripped"):
        # 去除注释
        if do_strip_comments:
            content = strip_comments(content, ext)
        # 去除 docstring
        if do_strip_docstrings and ext == ".py":
            content = strip_python_docstrings(content)

    # Python: 一次性应用所有行级变换（减少 split/join 次数）
    if ext == ".py" and detail_level in ("skeleton", "stripped"):
        content = _apply_python_line_transforms(content, detail_level)

    # Python: AST 级变换（保持独立调用，避免复杂合并引入 bug）
    if ext == ".py" and detail_level in ("skeleton", "stripped"):
        content = _remove_dead_code_after_control_flow(content)
        content = _remove_main_guard(content)
        content = _compress_self_assignments(content)
    if ext == ".py" and detail_level == "stripped":
        content = _inline_single_expr_functions(content)
        # 合并相邻字符串字面量（修复语法错误，使后续 AST 解析可靠）
        content = _merge_adjacent_string_literals(content)
        # 简化字典/集合推导式（dict/set comprehension）
        content = _simplify_dict_comp(content)
        content = _simplify_set_comp(content)
        # 去除类型注解（节省 ~5-15% tokens for annotated code）
        content = _remove_type_annotations(content)
        # 移除 if TYPE_CHECKING: 块（类型检查专用代码）
        content = _remove_type_checking_blocks(content)
        # 简化 @dataclass 装饰器（移除装饰器、field() 调用）
        content = _simplify_dataclass_decorator(content)
        # 常量折叠
        content = _fold_constants(content)
        # 移除未使用的局部变量
        content = _remove_unused_locals(content)
        # 去除 encoding 声明和 shebang
        content = _remove_encoding_and_shebang(content)
        # 移除模块级元数据（__version__、__author__ 等）
        content = _remove_module_metadata(content)
        # 移除 __all__ 定义
        content = _remove_all_definition(content)
        # 移除调试断点（import pdb; pdb.set_trace()）
        content = _remove_pdb_breakpoints(content)
        # 移除 breakpoint() 调用
        content = _remove_breakpoint(content)
        # 简化 kwargs setattr 循环
        content = _simplify_kwargs_setattr_loop(content)
        # 简化布尔检查
        content = _simplify_boolean_checks(content)
        # 移除冗余的 return None
        content = _remove_redundant_return_none(content)
        # 移除 return/yield 中的冗余括号
        content = _remove_redundant_return_parens(content)
        # 移除 return() 中的空括号
        content = _remove_return_empty_parens(content)
        # 移除函数末尾的 bare return
        content = _remove_bare_return_at_end(content)
        # 简化 isinstance 检查
        content = _simplify_isinstance_checks(content)
        # isinstance(x, (A,)) → isinstance(x, A)
        content = _simplify_isinstance_single_type(content)
        # d.get(k, None) is (not) None → "k" in/not in d
        content = _simplify_dict_get_none(content)
        # getattr(obj, 'a', None) → obj.a
        content = _simplify_getattr_none_default(content)
        # 移除空特殊方法
        content = _remove_empty_special_methods(content)
        # 简化常见设计模式样板
        content = _compress_common_patterns(content)
        # 简化 vars(self) → self.__dict__
        content = _simplify_vars_call(content)
        # 去除无插值的 f-string
        content = _remove_fstring_no_interpolation(content)
        # 简化单变量 f-string（f"{x}" → x）
        content = _simplify_fstring_single_var(content)
        # 三引号单行字符串 → 普通字符串
        content = _simplify_triple_quote_strings(content)
        # "{}".format(x) → x
        content = _simplify_format_call(content)
        # 移除不必要的 str() 调用（str("hello") → "hello"）
        content = _remove_unnecessary_str_calls(content)
        # 扁平化 else-after-return/raise 控制流
        content = _remove_else_after_flow_control(content)
        # 移除只有 pass 的 try-except 块
        content = _remove_try_except_pass(content)
        # 简化冗余真值检查（== True/False, or True/False）
        content = _compress_truthiness_checks(content)
        # BoolOp chain → in/not in（x==a or x==b → x in (a,b)）
        content = _simplify_boolop_to_in(content)
        # 简化 len() 真值检查（len(x) > 0 → x）
        content = _simplify_len_truthiness(content)
        # 空集合初始化简化为字面量
        content = _compress_empty_collections(content)
        # tuple([...]) / list((...)) / set((...)) → 字面量
        content = _simplify_wrapper_to_literal(content)
        # dict.fromkeys(keys, None) → dict.fromkeys(keys)
        content = _simplify_fromkeys_none_default(content)
        # 合并嵌套 if 条件
        content = _merge_nested_ifs(content)
        # 简化三元表达式
        content = _simplify_ternary(content)
        # 简化旧式三元（x and y or z → y if x else z）
        content = _simplify_old_style_ternary(content)
        # 移除单元素 tuple unpack
        content = _remove_tuple_wrap_single(content)
        # any/all/sum/join/max/min: list comp → generator
        content = _list_to_generator(content)
        # not not x → x
        content = _remove_not_not(content)
        # 移除冗余括号
        content = _remove_redundant_parens(content)
        # list(d.keys()) → d.keys()，set([...]) → {...}
        content = _remove_list_wrap(content)
        # 扁平化嵌套三元表达式
        content = _flatten_nested_ternary(content)
        # 移除未使用的函数定义
        content = _remove_unused_functions(content)
        # 移除未使用的类定义
        content = _remove_unused_classes(content)
        # 去除不再使用的 typing import（在所有代码移除后执行）
        content = _remove_unused_typing_imports(content)
        # 去除所有未使用的 import（在所有代码移除后执行）
        content = _remove_unused_imports(content)
        # 反演死 if（if x: pass else: body → if not x: body）
        content = _invert_dead_if(content)
        # 合并重复条件的 elif 分支
        content = _merge_duplicate_conditions(content)
        # if x is None: return None → if x is None: return
        content = _remove_return_none_after_none_check(content)
        # 删除 return/raise/break/continue 后的不可达代码
        content = _remove_unreachable_code(content)
        # enumerate(seq, 0) → enumerate(seq)
        content = _simplify_enumerate_start_zero(content)
        # 移除捕获未抛出异常类型的空 except 块
        content = _remove_unused_except_blocks(content)
        # super(ClassName, self) → super()
        content = _simplify_super_calls(content)
        # 合并连续重复的语句
        content = _collapse_duplicate_lines(content)
        # 移除常量断言（assert True, assert False, assert 1 == 1）
        content = _compress_asserts(content)
        # 移除断言中的消息文本（assert x, "msg" → assert x）
        content = _remove_assert_message(content)
        # 移除断言中的重复条件（assert x, x → assert x）
        content = _remove_assert_duplicate_condition(content)
        # 合并同体 if/elif 分支（相同 body 的条件合并）
        content = _merge_same_body_conditions(content)
        # 移除循环后不可达代码（无 break 的循环后的代码可达性分析）
        content = _remove_dead_after_loop(content)
        # range(len) → enumerate / for item in seq
        content = _range_len_to_enumerate(content)
        # 移除未使用的 enumerate 索引（for i, item in enumerate(x) → for item in x）
        content = _remove_unused_enumerate_index(content)
        # 简化布尔常量表达式（A and True → A, A or False → A）
        content = _simplify_bool_expr_with_const(content)
        # 简化 None 检查 return（if x is not None: return x → if x is not None: return）
        content = _simplify_none_check_return(content)
        # 简化 isinstance + not in（isinstance(x, dict) and "k" not in x → "k" not in x）
        content = _simplify_isinstance_and_not_in(content)
        # 移除只有 re-raise 的 try/except 块
        content = _remove_try_except_reraise(content)
        # 合并连续相同目标属性赋值（只保留最后一个）
        content = _merge_consecutive_attr_assignments(content)
        # 移除 await asyncio.sleep(0) 等无操作 await
        content = _remove_await_noop(content)
        # 简化同一性检查（x is x → True, x is not x → False）
        content = _simplify_identity_checks(content)
        # 简化 not (a is/a==/a in b) → a is not/a!=/a not in b
        content = _simplify_not_comparison(content)
        # 简化 bool(x) is True/False → bool(x) / not bool(x)
        content = _simplify_bool_is_true_false(content)
        # 移除只有 pass 的 with 块
        content = _remove_empty_with(content)
        # 简化切片表达式（x[0:len(x)] → x[:]）
        content = _simplify_slice_patterns(content)
        # 移除 if True/if False 死代码块
        content = _remove_dead_if_const(content)
        # 内联单次使用的局部变量（在 SyntaxError 修复后执行，支持级联内联）
        content = _inline_single_use_vars(content)
        # 清理移除代码块后残留的多余空行
        content = _remove_excess_blank_lines(content)

    # JSON/YAML 压缩：移除空白（始终执行，与 detail_level 无关）
    if ext in {".json", ".yaml", ".yml", ".toml"}:
        content = _minify_structured(content, ext)

    # 智能截断（token 感知）
    content = smart_truncate(content, max_tokens)
    return content


def _simplify_dataclass_decorator(content: str) -> str:
    """简化 @dataclass 装饰器。

    移除 @dataclass 装饰器，将 field() 调用替换为直接默认值，
    移除不再使用的 dataclasses import。
    """
    try:
        import ast as _ast
        tree = _ast.parse(content)
    except SyntaxError:
        return content

    if not isinstance(tree, _ast.Module) or not tree.body:
        return content

    edits: dict[int, str] = {}
    changed = False
    content_lines = content.split("\n")

    # 检查是否有 dataclass 导入
    has_dataclass_import = False
    for node in _ast.walk(tree):
        if isinstance(node, _ast.ImportFrom) and node.module == 'dataclasses':
            has_dataclass_import = True
            break

    if not has_dataclass_import:
        return content

    # 处理每个有 @dataclass 装饰器的类
    for node in _ast.walk(tree):
        if not isinstance(node, _ast.ClassDef):
            continue

        # 检查是否有 @dataclass 装饰器（支持 @dataclass 和 @dataclass(...)）
        has_dataclass = False
        for dec in node.decorator_list:
            if isinstance(dec, _ast.Name) and dec.id == 'dataclass':
                has_dataclass = True
                break
            elif isinstance(dec, _ast.Call) and isinstance(dec.func, _ast.Name) and dec.func.id == 'dataclass':
                has_dataclass = True
                break

        if not has_dataclass:
            continue

        # 移除 @dataclass 装饰器行（包括 @dataclass(...)）
        for dec in node.decorator_list:
            is_dataclass = False
            if isinstance(dec, _ast.Name) and dec.id == 'dataclass':
                is_dataclass = True
            elif isinstance(dec, _ast.Call) and isinstance(dec.func, _ast.Name) and dec.func.id == 'dataclass':
                is_dataclass = True

            if is_dataclass:
                dec_line = dec.lineno - 1
                if dec_line < len(content_lines):
                    edits[dec_line] = ""
                    changed = True

        # 处理类体中的属性
        for stmt in node.body:
            if not isinstance(stmt, _ast.AnnAssign):
                continue
            if not stmt.value or not isinstance(stmt.value, _ast.Call):
                continue

            func = stmt.value.func
            if not (isinstance(func, _ast.Name) and func.id == 'field'):
                continue

            line_idx = stmt.lineno - 1
            if line_idx >= len(content_lines):
                continue
            line = content_lines[line_idx]

            # 检查 init=False
            init_false = False
            for kw in stmt.value.keywords:
                if kw.arg == 'init' and isinstance(kw.value, _ast.Constant) and kw.value.value is False:
                    init_false = True
                    break

            if init_false:
                edits[line_idx] = ""
                changed = True
                continue

            # 获取默认值
            default_val = None
            # 先检查关键字参数
            for kw in stmt.value.keywords:
                if kw.arg == 'default':
                    try:
                        default_val = _ast.unparse(kw.value)
                    except Exception:
                        default_val = 'None'
                    break
                elif kw.arg == 'default_factory':
                    try:
                        factory_expr = _ast.unparse(kw.value)
                        # 如果是 lambda，保留原样不添加 ()
                        if factory_expr.startswith('lambda'):
                            default_val = factory_expr
                        else:
                            default_val = factory_expr + '()'
                    except Exception:
                        default_val = 'None'
                    break
            # 再检查位置参数（第一个位置参数是默认值）
            if default_val is None and stmt.value.args:
                try:
                    default_val = _ast.unparse(stmt.value.args[0])
                except Exception:
                    default_val = 'None'

            if default_val is not None:
                # 替换 field(default=...) 为直接值
                new_line = re.sub(r'=\s*field\([^)]*\)', f'= {default_val}', line)
                edits[line_idx] = new_line.rstrip()
                changed = True

    if changed:
        modified = list(content_lines)
        for idx in sorted(edits.keys(), reverse=True):
            if idx < len(modified):
                modified[idx] = edits[idx]
        content = "\n".join(modified)

    return content


def _remove_unused_enumerate_index(content: str) -> str:
    """移除未使用的 enumerate 索引变量。

    for i, item in enumerate(items):
        print(item)
    → for item in items:
        print(item)

    当 enumerate 的索引变量在循环体中未被引用时，
    直接迭代可节省 tokens。
    """
    try:
        import ast as _ast
        tree = _ast.parse(content)
    except SyntaxError:
        return content

    if not isinstance(tree, _ast.Module) or not tree.body:
        return content

    edits: dict[int, str] = {}
    changed = False
    content_lines = content.split("\n")

    for node in _ast.walk(tree):
        if not isinstance(node, _ast.For):
            continue

        # 检查目标是否是 (idx, item) 元组
        if not (isinstance(node.target, _ast.Tuple) and len(node.target.elts) == 2):
            continue

        idx_node = node.target.elts[0]
        item_node = node.target.elts[1]
        if not (isinstance(idx_node, _ast.Name) and isinstance(item_node, _ast.Name)):
            continue

        # 检查迭代器是否是 enumerate(...)
        if not (isinstance(node.iter, _ast.Call) and
                isinstance(node.iter.func, _ast.Name) and
                node.iter.func.id == 'enumerate'):
            continue

        # 检查索引变量是否在循环体中使用
        idx_name = idx_node.id
        idx_used = False
        for stmt in node.body:
            for child in _ast.walk(stmt):
                if isinstance(child, _ast.Name) and child.id == idx_name:
                    if isinstance(child.ctx, _ast.Store):
                        continue
                    idx_used = True
                    break
            if idx_used:
                break

        if idx_used:
            continue

        # 提取可迭代对象
        if not node.iter.args:
            continue
        try:
            iter_source = _ast.unparse(node.iter.args[0])
        except Exception:
            continue

        # 替换循环头
        line_idx = node.lineno - 1
        if line_idx >= len(content_lines):
            continue
        line = content_lines[line_idx]
        indent = line[:len(line) - len(line.lstrip())]
        new_line = f"{indent}for {item_node.id} in {iter_source}:"

        if new_line != line.rstrip():
            edits[line_idx] = new_line
            changed = True

    if changed:
        modified = list(content_lines)
        for idx in sorted(edits.keys(), reverse=True):
            if idx < len(modified):
                modified[idx] = edits[idx]
        content = "\n".join(modified)

    return content


def _remove_assert_message(content: str) -> str:
    """移除断言中的消息文本（assert x, "msg" → assert x）。

    断言消息主要用于调试，在 stripped 模式下可移除以节省 tokens。
    """
    lines = content.split("\n")
    modified = []
    for line in lines:
        stripped = line.strip()
        if not stripped.startswith("assert "):
            modified.append(line)
            continue

        # 找到第一个不在括号内的逗号
        depth = 0
        comma_idx = -1
        for i, c in enumerate(stripped):
            if c in '([':
                depth += 1
            elif c in ')]':
                depth -= 1
            elif c == ',' and depth == 0:
                comma_idx = i
                break

        if comma_idx > 0:
            indent = line[:len(line) - len(line.lstrip())]
            new_stripped = stripped[:comma_idx].rstrip()
            modified.append(indent + new_stripped)
        else:
            modified.append(line)

    return "\n".join(modified)


def _remove_type_checking_blocks(content: str) -> str:
    """移除 if TYPE_CHECKING: 块及其相关 import。

    这些块中的代码只在类型检查时执行，在 stripped 模式下可直接移除。
    同时移除 from typing import TYPE_CHECKING。
    """
    try:
        import ast as _ast
        tree = _ast.parse(content)
    except SyntaxError:
        return content

    if not isinstance(tree, _ast.Module) or not tree.body:
        return content

    edits: dict[int, str] = {}
    changed = False
    content_lines = content.split("\n")

    # 收集 if TYPE_CHECKING 块的范围
    if_blocks: list[tuple[int, int]] = []
    for node in _ast.walk(tree):
        if not isinstance(node, _ast.If):
            continue
        if isinstance(node.test, _ast.Name) and node.test.id == 'TYPE_CHECKING':
            start_line = node.lineno - 1
            end_line = getattr(node, 'end_lineno', start_line + 10) - 1
            if_blocks.append((start_line, end_line))

    if not if_blocks:
        return content

    # 移除 if TYPE_CHECKING 块
    for start_line, end_line in if_blocks:
        for li in range(start_line, min(end_line + 1, len(content_lines))):
            edits[li] = ""
        changed = True

    # 移除 from typing import TYPE_CHECKING
    for i, line in enumerate(content_lines):
        stripped = line.strip()
        if re.match(r'^from typing import .*TYPE_CHECKING', stripped):
            edits[i] = ""
            changed = True

    if changed:
        modified = list(content_lines)
        for idx in sorted(edits.keys(), reverse=True):
            if idx < len(modified):
                modified[idx] = edits[idx]
        content = "\n".join(modified)

    return content


def _remove_module_metadata(content: str) -> str:
    """移除模块级元数据赋值（__version__、__author__ 等）。

    这些赋值在 stripped 模式下通常不需要。
    """
    lines = content.split("\n")
    modified = []
    for line in lines:
        stripped = line.strip()
        if re.match(r'^__(?:version|author|email|license|copyright|description|url|doc|name|status|date|year|maintainer|credits)__\s*=', stripped):
            continue
        modified.append(line)
    return "\n".join(modified)


def _remove_pdb_breakpoints(content: str) -> str:
    """移除调试断点代码（import pdb; pdb.set_trace() 等）。"""
    lines = content.split("\n")
    modified = []
    for line in lines:
        stripped = line.strip()
        # 移除 import pdb 行
        if stripped == 'import pdb':
            continue
        # 移除包含 pdb.set_trace() 的行
        if 'pdb.set_trace()' in stripped:
            continue
        # 移除 import pdb 作为多 import 的一部分
        if stripped.startswith('import ') and 'pdb' in stripped:
            parts = stripped[7:].split(',')
            parts = [p.strip() for p in parts if 'pdb' not in p]
            if parts:
                indent = line[:len(line) - len(line.lstrip())]
                modified.append(indent + 'import ' + ', '.join(parts))
            continue
        modified.append(line)
    return "\n".join(modified)


def _simplify_kwargs_setattr_loop(content: str) -> str:
    """简化 kwargs setattr 循环。

    for k, v in kwargs.items():
        setattr(self, k, v)
    → self.__dict__.update(kwargs)
    """
    lines = content.split("\n")
    modified = list(lines)
    changed = False
    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        # 检测 for k, v in X.items():
        m = re.match(r'^for\s+(\w+)\s*,\s*(\w+)\s+in\s+(\w+)\.items\(\):', stripped)
        if m:
            k, v, obj = m.group(1), m.group(2), m.group(3)
            body_start = i + 1
            base_indent = len(lines[i]) - len(lines[i].lstrip())

            # 检查循环体是否全是 setattr(self, k, v)
            j = body_start
            all_setattr = True
            while j < len(lines):
                s = lines[j].strip()
                if not s or s.startswith('#'):
                    j += 1
                    continue
                ci = len(lines[j]) - len(lines[j].lstrip())
                if ci <= base_indent:
                    break
                if not re.match(rf'setattr\(\s*self\s*,\s*{k}\s*,\s*{v}\s*\)(?:\s*#.*)?$', s):
                    all_setattr = False
                    break
                j += 1

            if all_setattr and j > body_start:
                indent = lines[i][:base_indent]
                replacement = f"{indent}self.__dict__.update({obj})"
                modified[i] = replacement
                for k_idx in range(body_start, j):
                    modified[k_idx] = ""
                changed = True
                i = j
                continue

        i += 1

    if changed:
        return "\n".join(modified)
    return content


def _remove_all_definition(content: str) -> str:
    """移除 __all__ 定义（stripped 模式下不需要）。"""
    lines = content.split("\n")
    modified = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith('__all__'):
            continue
        modified.append(line)
    return "\n".join(modified)


def _remove_unnecessary_str_calls(content: str) -> str:
    """移除不必要的 str() 调用。

    str("hello") → "hello"
    str('world') → 'world'
    """
    lines = content.split("\n")
    modified = []
    for line in lines:
        # 匹配 str("...") 或 str('...')
        new_line = re.sub(r'str\(("(?:[^"\\]|\\.)*")\)', r'\1', line)
        new_line = re.sub(r"str\(('(?:[^'\\]|\\.)*')\)", r'\1', new_line)
        modified.append(new_line)
    return "\n".join(modified)


def _simplify_vars_call(content: str) -> str:
    """简化 vars() 调用。

    vars(self) → self.__dict__
    """
    lines = content.split("\n")
    modified = []
    for line in lines:
        new_line = re.sub(r'vars\(\s*self\s*\)', 'self.__dict__', line)
        modified.append(new_line)
    return "\n".join(modified)


def _remove_unused_typing_imports(content: str) -> str:
    """去除不再使用的 typing import（在去除类型注解后执行）。

    处理：
    - `from typing import X, Y, Z`：移除不再被引用的名称
    - `import typing`：如果没有任何 typing.X 引用则删除整行
    - `import typing as t`：如果没有任何 t.X 引用则删除整行
    """
    lines = content.split("\n")
    from_typing_names: dict[int, list[str]] = {}  # line_idx -> [names]
    import_typing_lines: list[tuple[int, str | None]] = []  # [(line_idx, alias)]

    # 收集 typing import 行
    for i, line in enumerate(lines):
        stripped = line.strip()
        # from typing import X, Y, Z
        m = re.match(r'^from typing import (.+)$', stripped)
        if m:
            names_str = m.group(1).strip('()').strip()
            names = [n.strip().split(' as ')[0].strip() for n in names_str.split(',')]
            names = [n for n in names if n and not n.startswith('#')]
            if names:
                from_typing_names[i] = names
                continue
        # import typing
        m2 = re.match(r'^import typing(?:\s+as\s+(\w+))?$', stripped)
        if m2:
            alias = m2.group(1)  # None for `import typing`, alias name for `import typing as t`
            import_typing_lines.append((i, alias))

    if not from_typing_names and not import_typing_lines:
        return content

    # 构建不包含 typing import 行的代码用于检查引用
    skip_indices = set(from_typing_names.keys()) | {idx for idx, _ in import_typing_lines}
    non_import_text = '\n'.join(
        lines[i] for i in range(len(lines)) if i not in skip_indices
    )

    modified = list(lines)
    changed = False

    # 处理 from typing import ...
    for idx, names in from_typing_names.items():
        used = {n for n in names if re.search(r'\b' + re.escape(n) + r'\b', non_import_text)}
        if used == set(names):
            continue  # 全部使用中
        if not used:
            modified[idx] = ""
            changed = True
            continue
        # 部分使用：保留使用的
        line = lines[idx]
        m = re.match(r'^(\s*from typing import )(.+)$', line)
        if m:
            prefix = m.group(1)
            modified[idx] = f"{prefix}{', '.join(sorted(used))}"
            changed = True

    # 处理 import typing / import typing as t
    for idx, alias in import_typing_lines:
        if alias:
            # import typing as t：检查 t.X 是否被使用
            pattern = r'\b' + re.escape(alias) + r'\.\w+'
        else:
            # import typing：检查 typing.X 是否被使用
            pattern = r'\btyping\.\w+'
        if re.search(pattern, non_import_text):
            continue  # 被使用
        modified[idx] = ""
        changed = True

    return "\n".join(modified) if changed else content


def _fold_constants(content: str) -> str:
    """常量折叠：在预处理时计算简单常量表达式。

    处理：
    - 算术运算：1 + 2 → 3, 2 * 3 → 6
    - 字符串拼接：'a' + 'b' → 'ab'
    - len() 调用：len([1, 2, 3]) → 3
    """
    try:
        import ast as _ast
        tree = _ast.parse(content)
    except SyntaxError:
        return content

    lines = content.split("\n")
    if not lines:
        return content

    # 手动计算 BinOp 结果
    def _eval_binop(node):
        try:
            left = _ast.literal_eval(node.left)
            right = _ast.literal_eval(node.right)
        except (TypeError, ValueError):
            return None
        op = node.op
        if isinstance(op, _ast.Add):
            return left + right
        elif isinstance(op, _ast.Mult):
            return left * right
        elif isinstance(op, _ast.Sub):
            return left - right
        elif isinstance(op, _ast.Div):
            return left / right if right != 0 else None
        elif isinstance(op, _ast.FloorDiv):
            return left // right if right != 0 else None
        elif isinstance(op, _ast.Mod):
            return left % right if right != 0 else None
        elif isinstance(op, _ast.Pow):
            return left ** right
        return None

    # 迭代折叠（先折叠内层，再折叠外层）
    for _ in range(10):  # 最多 10 轮，防止无限循环
        try:
            tree = _ast.parse("\n".join(lines))
        except SyntaxError:
            break

        edits: dict[int, str] = {}
        found = False

        for node in _ast.walk(tree):
            # BinOp 折叠
            if isinstance(node, _ast.BinOp):
                result = _eval_binop(node)
                if result is not None and isinstance(result, (int, float, str)):
                    line_idx = node.lineno - 1
                    if line_idx < len(lines):
                        line = lines[line_idx]
                        start = node.col_offset
                        end = getattr(node, 'end_col_offset', start + 1)
                        new_line = line[:start] + repr(result) + line[end:]
                        if new_line != line:
                            edits[line_idx] = new_line
                            found = True

            # len() 折叠
            elif isinstance(node, _ast.Call):
                if isinstance(node.func, _ast.Name) and node.func.id == 'len':
                    if len(node.args) == 1:
                        try:
                            arg_val = _ast.literal_eval(node.args[0])
                            if isinstance(arg_val, (list, tuple, set, dict, str)):
                                result = len(arg_val)
                                line_idx = node.lineno - 1
                                if line_idx < len(lines):
                                    line = lines[line_idx]
                                    start = node.col_offset
                                    end = getattr(node, 'end_col_offset', start + 1)
                                    new_line = line[:start] + repr(result) + line[end:]
                                    if new_line != line:
                                        edits[line_idx] = new_line
                                        found = True
                        except (TypeError, ValueError):
                            pass

        if not found:
            break

        # 应用编辑（从后往前）
        for line_idx in sorted(edits.keys(), reverse=True):
            if line_idx < len(lines):
                lines[line_idx] = edits[line_idx]

    return "\n".join(lines)


def _remove_unused_locals(content: str) -> str:
    """移除未被使用的局部变量赋值（仅 stripped 模式）。

    安全规则：
    - 只移除变量赋值（`x = ...`），不修改属性（`self.x = ...`）
    - 使用活性分析：仅删除在赋值后从未被读取的变量
    - 保留参数、self、cls 等
    """
    try:
        import ast as _ast
        tree = _ast.parse(content)
    except SyntaxError:
        return content

    lines = content.split("\n")
    if not lines:
        return content

    removals: list[int] = []  # line indices to blank out

    # 需要忽略的名称集合
    _SKIP_NAMES = frozenset({'self', 'cls', 'super', '__name__', '__doc__', '__module__',
                              '__qualname__', '__annotations__', '__dict__',
                              '__class__', '__bases__', '__mro_entries__'})

    for node in _ast.walk(tree):
        if not isinstance(node, (_ast.FunctionDef, _ast.AsyncFunctionDef)):
            continue

        body = getattr(node, 'body', [])
        if not body:
            continue

        # 收集所有赋值和读取的位置
        # assignments: name -> sorted list of (line_idx, is_annassign)
        assignments: dict[str, list[tuple[int, bool]]] = {}
        reads: dict[str, list[int]] = {}  # name -> sorted list of line indices

        def _add_write(name: str, line_idx: int, is_ann: bool = False):
            assignments.setdefault(name, []).append((line_idx, is_ann))

        def _add_read(name: str, line_idx: int):
            reads.setdefault(name, []).append(line_idx)

        for child in _ast.walk(node):
            if isinstance(child, _ast.Name):
                if isinstance(child.ctx, _ast.Load):
                    _add_read(child.id, child.lineno - 1)
                elif isinstance(child.ctx, _ast.Store):
                    _add_write(child.id, child.lineno - 1)
            elif isinstance(child, _ast.AnnAssign):
                if child.target and isinstance(child.target, _ast.Name):
                    _add_write(child.target.id, child.lineno - 1, is_ann=True)

        # 活性分析：对于每个赋值，检查赋值后是否有读取
        for name, writes in assignments.items():
            if name in _SKIP_NAMES:
                continue
            # 获取该变量的所有读取位置
            var_reads = sorted(reads.get(name, []))
            if not var_reads:
                # 从未被读取：所有赋值都是死代码
                for line_idx, _ in writes:
                    if line_idx not in removals:
                        removals.append(line_idx)
                continue

            # 有读取：检查每个赋值是否在其后的读取之前被覆盖
            writes_sorted = sorted(writes, key=lambda x: x[0])
            for i, (write_line, is_ann) in enumerate(writes_sorted):
                # 找到此赋值后的第一个读取
                next_write = writes_sorted[i + 1][0] if i + 1 < len(writes_sorted) else float('inf')
                # 检查是否有读取在 [write_line, next_write] 区间内（含 write_line 自身）
                has_read = any(write_line <= r < next_write for r in var_reads)
                if not has_read and not is_ann:
                    # 此赋值后直到下次赋值（或函数结束）之间没有读取
                    # 但如果这是最后一次赋值且有读取在之后，保留
                    if i < len(writes_sorted) - 1:
                        # 不是最后一次赋值，且后面有赋值覆盖 → 死代码
                        removals.append(write_line)
                    elif var_reads and all(r <= write_line for r in var_reads):
                        # 所有读取都在此赋值之前 → 死代码
                        removals.append(write_line)

    if not removals:
        return content

    # 从后往前删除（避免行号偏移）
    modified = list(lines)
    for line_idx in sorted(set(removals), reverse=True):
        if line_idx < len(modified):
            line = modified[line_idx]
            stripped = line.strip()
            # 只删除简单的赋值行（防止误删控制流等）
            if stripped and not stripped.startswith(('#', 'if', 'elif', 'else',
                                                       'for', 'while', 'with', 'try',
                                                       'except', 'finally', 'return',
                                                       'raise', 'yield', 'class ', 'def ',
                                                       'import ', 'from ', '@')):
                modified[line_idx] = ""

    return "\n".join(modified)


def _remove_unused_imports(content: str) -> str:
    """移除未使用的 import 语句（stripped 模式）。

    处理：
    - `import X`：如果没有任何 `X.attr` 引用则删除
    - `import X as Y`：如果没有任何 `Y` 引用则删除
    - `from X import A, B`：仅保留被引用的名称
    """
    lines = content.split("\n")
    if not lines:
        return content

    import_entries: list[dict] = []

    for i, line in enumerate(lines):
        stripped = line.strip()
        # import X
        m = re.match(r'^import\s+(\w+(?:\s*,\s*\w+)*)(?:\s+as\s+(\w+))?\s*$', stripped)
        if m:
            modules_str = m.group(1)
            alias = m.group(2)
            modules = [x.strip() for x in modules_str.split(',')]
            for mod in modules:
                import_entries.append({
                    'line_idx': i,
                    'type': 'import',
                    'module': mod,
                    'alias': alias,
                    'original': line,
                })
            continue

        # from X import A, B
        m2 = re.match(r'^from\s+([\w.]+)\s+import\s+(.+)$', stripped)
        if m2:
            module = m2.group(1)
            names_str = m2.group(2).strip().strip('()')
            names = []
            alias_map = {}
            for part in names_str.split(','):
                part = part.strip()
                if not part or part.startswith('#'):
                    continue
                if ' as ' in part:
                    name, alias = part.rsplit(' as ', 1)
                    names.append(name.strip())
                    alias_map[name.strip()] = alias.strip()
                else:
                    names.append(part)
            if names:
                import_entries.append({
                    'line_idx': i,
                    'type': 'from_import',
                    'module': module,
                    'names': names,
                    'alias_map': alias_map,
                    'original': line,
                })

    if not import_entries:
        return content

    import_lines = {e['line_idx'] for e in import_entries}
    code_without_imports = '\n'.join(
        lines[i] for i in range(len(lines)) if i not in import_lines
    )

    modified = list(lines)
    changed = False

    # 按行分组处理（同一行可能有多个 import）
    by_line: dict[int, list[dict]] = {}
    for entry in import_entries:
        by_line.setdefault(entry['line_idx'], []).append(entry)

    for line_idx, entries in sorted(by_line.items()):
        if all(e['type'] == 'from_import' for e in entries):
            # from import：逐个处理
            for entry in entries:
                names = entry['names']
                alias_map = entry['alias_map']
                used = []
                for name in names:
                    alias = alias_map.get(name, name)
                    if re.search(r'\b' + re.escape(alias) + r'\b', code_without_imports):
                        used.append(name)
                if len(used) == len(names):
                    continue
                if not used:
                    modified[line_idx] = ""
                    changed = True
                    continue
                line = entry['original']
                m = re.match(r'^(\s*from\s+[\w.]+\s+import\s+)', line)
                if m:
                    prefix = m.group(1)
                    kept_names = []
                    for name in used:
                        if name in alias_map:
                            kept_names.append(f"{name} as {alias_map[name]}")
                        else:
                            kept_names.append(name)
                    modified[line_idx] = f"{prefix}{', '.join(kept_names)}"
                    changed = True
        else:
            # import X 或混合：检查是否整行可以保留
            all_used = True
            any_used = False
            for entry in entries:
                if entry['type'] == 'import':
                    module = entry['module']
                    alias = entry['alias']
                    if alias:
                        if re.search(r'\b' + re.escape(alias) + r'\b', code_without_imports):
                            any_used = True
                        else:
                            all_used = False
                    else:
                        if re.search(r'\b' + re.escape(module) + r'\.\w+', code_without_imports):
                            any_used = True
                        else:
                            all_used = False
            if all_used:
                continue
            if not any_used:
                modified[line_idx] = ""
                changed = True
                continue
            # 部分使用：重建行
            kept = []
            for entry in entries:
                if entry['type'] == 'import':
                    module = entry['module']
                    alias = entry['alias']
                    if alias:
                        if re.search(r'\b' + re.escape(alias) + r'\b', code_without_imports):
                            kept.append(f"{module} as {alias}" if module != alias else module)
                    else:
                        if re.search(r'\b' + re.escape(module) + r'\.\w+', code_without_imports):
                            kept.append(module)
            if kept:
                modified[line_idx] = f"import {', '.join(kept)}"
                changed = True
            else:
                modified[line_idx] = ""
                changed = True

    return "\n".join(modified) if changed else content



def _remove_encoding_and_shebang(content: str) -> str:
    """移除 Python 文件的 encoding 声明、shebang 和 __future__ 导入。

    Python 3 默认 utf-8 编码，不需要 encoding 声明或 shebang。
    __future__ 导入在目标 Python 版本中通常已默认启用。
    """
    lines = content.split("\n")
    result = []
    removed = 0
    for line in lines:
        stripped = line.strip()
        # shebang
        if stripped.startswith("#!"):
            removed += 1
            continue
        # coding 声明（多种格式）
        if stripped.startswith("#") and (
            "coding" in stripped or "utf-8" in stripped.lower()
            or "utf_8" in stripped.lower()
        ):
            removed += 1
            continue
        # __future__ 导入
        if stripped.startswith("from __future__") or stripped.startswith("import __future__"):
            removed += 1
            continue
        result.append(line)
    if removed:
        return "\n".join(result)
    return content


def _simplify_boolean_checks(content: str) -> str:
    """简化冗余的布尔检查表达式。

    - `if x is True:` → `if x:`
    - `if x is False:` → `if not x:`
    - `if not x is False:` → `if x:`
    - `if x is not False:` → `if x:`
    - `if not x is True:` → `if not x:`
    - `if x is not True:` → `if not x:`
    - `if isinstance(x, bool):` + bool 字面量 → 直接检查
    仅作用于独立的 if/elif/while 条件行，避免误改表达式。
    """
    lines = content.split("\n")
    modified = []
    changed = 0
    for line in lines:
        stripped = line.strip()
        # 仅处理 if/elif/while 条件行
        if not re.match(r'^(if|elif|while)\s+', stripped):
            modified.append(line)
            continue

        new_line = line
        # 去掉末尾冒号，处理主体，再恢复
        has_colon = stripped.endswith(":")
        body = stripped[:-1] if has_colon else stripped

        # 提取关键字和条件
        m = re.match(r'^(if|elif|while)\s+(.+)$', body)
        if not m:
            modified.append(line)
            continue
        keyword = m.group(1)
        cond = m.group(2)

        original_cond = cond

        # 顺序很重要：先处理带 not 的，再处理简单的
        # if not x is False: → if x:
        cond = re.sub(r'not\s+(\w+)\s+is\s+False\b', r'\1', cond)
        # if x is not False: → if x:
        cond = re.sub(r'(\w+)\s+is\s+not\s+False\b', r'\1', cond)
        # if not x is True: → if not x:
        cond = re.sub(r'not\s+(\w+)\s+is\s+True\b', r'not \1', cond)
        # if x is not True: → if not x:
        cond = re.sub(r'(\w+)\s+is\s+not\s+True\b', r'not \1', cond)
        # if x is False: → if not x:
        cond = re.sub(r'(\w+)\s+is\s+False\b', r'not \1', cond)
        # if x is True: → if x:
        cond = re.sub(r'(\w+)\s+is\s+True\b', r'\1', cond)
        # if x == True: → if x:
        cond = re.sub(r'(\w+)\s*==\s*True\b', r'\1', cond)
        # if x == False: → if not x:
        cond = re.sub(r'(\w+)\s*==\s*False\b', r'not \1', cond)

        if cond != original_cond:
            new_line = line.replace(original_cond, cond, 1)
            changed += 1

        modified.append(new_line)

    if changed:
        return "\n".join(modified)
    return content


def _remove_redundant_return_none(content: str) -> str:
    """移除函数末尾或分支中的冗余 `return None`。

    Python 函数默认返回 None，`return None` 是冗余的。
    但保留可能影响控制流的 return（如在 if/else 分支中，
    只移除没有 else 分支且后面没有代码的分支末尾的 return None）。
    安全策略：仅移除缩进级别匹配且后面是空行/注释/函数结束的 return None。
    """
    lines = content.split("\n")
    modified = list(lines)
    removed = 0
    func_indent = -1
    in_func = False

    for i, line in enumerate(lines):
        stripped = line.strip()
        indent = len(line) - len(line.lstrip()) if line.strip() else -1

        # 跟踪函数定义
        if stripped.startswith("def ") and stripped.endswith(":"):
            func_indent = indent
            in_func = True
            continue
        # 跟踪类定义
        if stripped.startswith("class ") and stripped.endswith(":"):
            in_func = False
            func_indent = -1
            continue

        if not in_func or func_indent < 0:
            continue

        # 检查是否是 return None 且缩进在函数体内
        if (indent > func_indent
                and re.match(r'^(\s*)return\s+None\s*(?:#.*)?$', line)):
            # 检查是否是函数体级别（可直接移除）vs 嵌套块内（需谨慎）
            is_at_func_level = (indent == func_indent + 4)

            if not is_at_func_level:
                # 检查是否是块中唯一的语句（如果是，不能移除，否则会导致空块 SyntaxError）
                block_header_idx = None
                for j in range(i - 1, -1, -1):
                    prev_line = lines[j]
                    prev_stripped = prev_line.strip()
                    if not prev_stripped:
                        continue
                    prev_indent = len(prev_line) - len(prev_line.lstrip())
                    if prev_indent < indent:
                        if prev_stripped.endswith(':'):
                            block_header_idx = j
                        break
                if block_header_idx is not None:
                    # 检查块内是否有其他同缩进语句
                    for j in range(block_header_idx + 1, i):
                        stmt_line = lines[j]
                        stmt_stripped = stmt_line.strip()
                        if not stmt_stripped or stmt_stripped.startswith('#'):
                            continue
                        stmt_indent = len(stmt_line) - len(stmt_line.lstrip())
                        if stmt_indent == indent:
                            break  # there's another statement at the same level
                    else:
                        # return None 是块中唯一语句，不能移除
                        continue

            # 检查后面是否还有有效代码（非空行、非注释、非 return）
            has_code_after = False
            for j in range(i + 1, len(lines)):
                next_line = lines[j]
                next_stripped = next_line.strip()
                if not next_stripped:
                    continue
                if next_stripped.startswith("#"):
                    continue
                next_indent = len(next_line) - len(next_line.lstrip())
                # 同级或更浅缩进 → 函数体结束
                if next_indent <= func_indent:
                    break
                # 同缩进且有实质代码
                if next_indent == indent and not next_stripped.startswith(
                    ("return", "raise", "yield", "#")
                ):
                    has_code_after = True
                    break
                # 深层缩进有代码
                if next_indent > indent and not next_stripped.startswith("#"):
                    has_code_after = True
                    break

            if not has_code_after:
                modified[i] = ""
                removed += 1

    if removed:
        return "\n".join(modified)
    return content


def _simplify_isinstance_checks(content: str) -> str:
    """简化 isinstance 检查为更紧凑的形式。

    - `isinstance(x, type(None))` → `x is None`
    - `isinstance(x, (int, float, str, list, dict, set, tuple, bool))`
      → 保留（无法简化，但多参数时保留原样）
    仅作用于 if/elif/while 条件和赋值条件中。
    """
    lines = content.split("\n")
    modified = []
    changed = 0

    for line in lines:
        stripped = line.strip()
        new_line = line

        # isinstance(x, type(None)) → x is None
        m = re.search(r'isinstance\s*\(\s*(\w+)\s*,\s*type\s*\(\s*None\s*\)\s*\)', line)
        if m:
            var = m.group(1)
            replacement = f"{var} is None"
            # 确定比较方向
            # Check if it's part of a larger expression
            start_pos = m.start()
            end_pos = m.end()
            before = line[:start_pos]
            after = line[end_pos:]
            # Look for == or != before/after
            if re.search(r'==\s*$', before.strip()):
                new_line = before.rstrip() + " " + replacement + after
            elif re.search(r'^\s*!=\s*', after):
                new_line = before + " " + replacement + after
            else:
                new_line = line[:start_pos] + replacement + line[end_pos:]
            changed += 1

        modified.append(new_line)

    if changed:
        return "\n".join(modified)
    return content


def _remove_empty_special_methods(content: str) -> str:
    """移除只有 `pass` 或 docstring + pass 的特殊方法。

    例如：
        def __repr__(self):
            \"\"\"Return repr.\"\"\"
            pass
    → 整个方法（含 def 行）被移除
    """
    lines = content.split("\n")
    modified = list(lines)
    removed_lines = 0

    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        # 匹配特殊方法定义
        m = re.match(r'^(def\s+(?:__\w+__|__init__|__str__|__repr__|__hash__|__eq__|__ne__|__lt__|__le__|__gt__|__ge__|__bool__|__len__|__getitem__|__setitem__|__delitem__|__iter__|__next__|__contains__|__call__|__enter__|__exit__|__aenter__|__aexit__|__await__|__aiter__|__anext__|__del__|__format__|__getattr__|__getattribute__|__setattr__|__delattr__|__dir__|__get__|__set__|__delete__|__set_name__|__init_subclass__|__class_getitem__|__instancecheck__|__subclasscheck__|__mro_entries__))\s*\(', stripped)
        if m:
            func_name = m.group(1)
            func_indent = len(lines[i]) - len(lines[i].lstrip())
            def_line_idx = i
            # 找到方法体结束
            j = i + 1
            body_lines = []
            while j < len(lines):
                next_line = lines[j]
                if not next_line.strip():
                    body_lines.append(j)
                    j += 1
                    continue
                next_indent = len(next_line) - len(next_line.lstrip())
                if next_indent <= func_indent and next_line.strip():
                    break
                body_lines.append(j)
                j += 1

            # 检查方法体是否只包含 docstring + pass
            if body_lines:
                code_lines = [lines[k].strip() for k in body_lines if lines[k].strip()]
                # 如果只有 pass（可能带 docstring），标记为可删除
                if all(l in ('pass',) or l.startswith(('"""', "'''")) for l in code_lines):
                    # 清除 def 行和所有 body 行
                    for k in [def_line_idx] + body_lines:
                        modified[k] = ""
                        removed_lines += 1
                    i = j
                    continue

        i += 1

    if removed_lines:
        return "\n".join(modified)
    return content


def _compress_common_patterns(content: str) -> str:
    """压缩常见的 Python 代码样板模式。

    当前处理的模式：
    1. `from __future__ import ...` — 已通过 normalize_whitespace 处理
    2. Markdown YAML frontmatter 已在入口处处理
    3. `if TYPE_CHECKING:` 块中的只用于类型注解的代码
       → 在 stripped 模式下这些通常已被移除
    4. 去除类中只包含 pass 的嵌套类（极罕见但存在）
    """
    lines = content.split("\n")
    modified = list(lines)
    changed = False

    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        # 检测类定义
        m = re.match(r'^(class\s+\w+[^:]*):', stripped)
        if m:
            class_indent = len(lines[i]) - len(lines[i].lstrip())
            # 找到类体结束
            j = i + 1
            body_indices = []
            while j < len(lines):
                if not lines[j].strip():
                    body_indices.append(j)
                    j += 1
                    continue
                ci = len(lines[j]) - len(lines[j].lstrip())
                if ci <= class_indent and lines[j].strip():
                    break
                body_indices.append(j)
                j += 1

            if body_indices:
                code_in_body = [lines[k].strip() for k in body_indices if lines[k].strip()]
                # 如果类体只有 pass，移除整个类
                non_pass = [l for l in code_in_body if l != 'pass'
                            and not l.startswith(('"""', "'''", '#',
                                                   'def ', 'class '))]
                if not non_pass and 'pass' in code_in_body:
                    # 清除 class 行和 body 行
                    modified[i] = ""
                    for k in body_indices:
                        modified[k] = ""
                    changed = True
                    i = j
                    continue

        i += 1

    if changed:
        return "\n".join(modified)
    return content


def _remove_fstring_no_interpolation(content: str) -> str:
    """将没有插值的 f-string 转换为普通字符串字面量。

    f"hello" → "hello"
    仅处理没有任何 {expr} 的 f-string，有插值的保持不变。
    """
    lines = content.split("\n")
    modified = []
    changed = 0
    for line in lines:
        # 匹配 f"..." 或 f'...' 或 f"""...""" 或 f'''...'''
        new_line = re.sub(
            r'f("(?:[^"\\]|\\.)*")'
            r"|f('(?:[^'\\]|\\.)*')"
            r'|f("""(?:[^"\\]|\\.)*""")'
            r"|f('''(?:[^'\\]|\\.)*''')",
            lambda m: (m.group(0) if "{" in m.group(0)
                       else (m.group(1) or m.group(2) or m.group(3) or m.group(4))),
            line,
        )
        if new_line != line:
            changed += 1
        modified.append(new_line)

    if changed:
        return "\n".join(modified)
    return content


def _remove_else_after_flow_control(content: str) -> str:
    """扁平化控制流：移除 return/raise/continue/break 后的 else/elif。

    模式：
        if cond:
            return x
        else:
            y = 1
    →  if cond:
           return x
       y = 1

    安全策略：仅当 if 分支体只包含单个 return/raise/continue/break
    时才扁平化 else 块。
    """
    lines = content.split("\n")
    if len(lines) < 4:
        return content

    modified = list(lines)
    changed = False
    i = 0

    while i < len(lines):
        stripped = lines[i].strip()
        m = re.match(r'^(if|elif)\s+(.+):\s*$', stripped)
        if not m:
            i += 1
            continue

        header_indent = len(lines[i]) - len(lines[i].lstrip())

        # 找到 if 分支体的结束（遇到 else:/elif: 或同级代码时停止）
        body_end = i + 1
        while body_end < len(lines):
            if lines[body_end].strip():
                ci = len(lines[body_end]) - len(lines[body_end].lstrip())
                if ci <= header_indent and lines[body_end].strip().startswith(("else:", "elif ")):
                    break
                if ci <= header_indent:
                    body_end += 1
                    break
            body_end += 1

        # if 分支体中的非空行
        if_body_lines = [k for k in range(i + 1, body_end) if lines[k].strip()]

        # 分支体只包含单个 return/raise/continue/break
        if (len(if_body_lines) == 1
                and re.match(r'^(return|raise|continue|break)\b',
                           lines[if_body_lines[0]].strip())):

            # 检查 body_end 处是否是 else:
            if body_end < len(lines) and lines[body_end].strip() == "else:":
                # 收集 else 体
                else_body_start = body_end + 1
                else_body_end = else_body_start
                while else_body_end < len(lines):
                    if lines[else_body_end].strip():
                        ci2 = len(lines[else_body_end]) - len(lines[else_body_end].lstrip())
                        if ci2 <= header_indent:
                            break
                    else_body_end += 1

                # 清除 else 行
                modified[body_end] = ""
                # 调整 else 体缩进（降低 4 个空格，与 if 同级）
                for k in range(else_body_start, else_body_end):
                    if lines[k].strip():
                        current_indent = len(lines[k]) - len(lines[k].lstrip())
                        if current_indent > header_indent:
                            modified[k] = lines[k][header_indent:]  # 去掉 4 个额外缩进
                changed = True
                i = else_body_end
                continue

        i = body_end

    if changed:
        return "\n".join(modified)
    return content


def _remove_try_except_pass(content: str) -> str:
    """移除只有 pass 的 try-except 块。

    try:
        x = 1
    except:
        pass
    →  x = 1

    只处理 except 块只有 pass 的情况（保留有实际错误处理的块）。
    """
    lines = content.split("\n")
    if len(lines) < 4:
        return content

    modified = list(lines)
    changed = False
    i = 0

    while i < len(lines):
        stripped = lines[i].strip()
        if not stripped.startswith("try:"):
            i += 1
            continue

        try_indent = len(lines[i]) - len(lines[i].lstrip())
        j = i + 1

        # 收集 try 体
        try_body_start = j
        try_body_end = j
        while try_body_end < len(lines):
            if lines[try_body_end].strip():
                ci = len(lines[try_body_end]) - len(lines[try_body_end].lstrip())
                if ci <= try_indent and lines[try_body_end].strip().startswith(("except", "finally")):
                    break
            try_body_end += 1

        # 检查是否有 except/finally
        if try_body_end >= len(lines):
            i = try_body_end
            continue

        except_line = lines[try_body_end].strip()

        # 处理 except 块
        if except_line.startswith("except"):
            except_indent = len(lines[try_body_end]) - len(lines[try_body_end].lstrip())
            k = try_body_end + 1

            # 收集 except 体
            except_body_start = k
            except_body_end = k
            while except_body_end < len(lines):
                if lines[except_body_end].strip():
                    ci = len(lines[except_body_end]) - len(lines[except_body_end].lstrip())
                    if ci <= except_indent and lines[except_body_end].strip().startswith(("except", "else", "finally")):
                        break
                except_body_end += 1

            # 检查 except 体是否只有 pass
            except_code = [lines[x].strip() for x in range(except_body_start, except_body_end)
                          if lines[x].strip()]
            if except_code == ['pass']:
                # 清除 try 行、except 行和 except body，并对 try body 去缩进
                modified[i] = ""
                for x in range(try_body_end, except_body_end):
                    modified[x] = ""
                # Dedent try body lines by try_indent spaces
                for x in range(try_body_start, try_body_end):
                    line = modified[x]
                    stripped = line.lstrip()
                    if stripped:
                        current_indent = len(line) - len(stripped)
                        if current_indent >= try_indent:
                            modified[x] = line[try_indent:]
                        else:
                            modified[x] = stripped
                changed = True
                i = except_body_end
                continue

        i = try_body_end + 1

    if changed:
        return "\n".join(modified)
    return content


def _compress_truthiness_checks(content: str) -> str:
    """简化冗余的真值检查表达式。

    模式（仅作用于 if/elif/while 条件行和 assert 行）：
      - `if x == True:` → `if x:`
      - `if x == False:` → `if not x:`
      - `if not x == True:` → `if not x:`
      - `if x != False:` → `if x:`
      - `if not x != True:` → `if not x:`
      - `if x or True:` → `if x:`
      - `if x and False:` → `if False:`  → `if False:` → `pass`（移除以减少 token）
    """
    lines = content.split("\n")
    modified = []
    changed = 0
    for line in lines:
        stripped = line.strip()
        # 仅处理 if/elif/while/assert 行
        if not re.match(r'^(if|elif|while|assert)\s+', stripped):
            modified.append(line)
            continue

        original = line
        # 去掉末尾冒号（条件部分）
        has_colon = stripped.endswith(":")
        body = stripped[:-1] if has_colon else stripped
        m = re.match(r'^(if|elif|while|assert)\s+(.+)$', body)
        if not m:
            modified.append(line)
            continue
        keyword = m.group(1)
        cond = m.group(2)

        original_cond = cond

        # 应用简化规则（顺序重要）
        # not x == True → not x:
        cond = re.sub(r'not\s+(\w+)\s*==\s*True\b', r'not \1', cond)
        # x != False → x:
        cond = re.sub(r'(\w+)\s*!=\s*False\b', r'\1', cond)
        # not x != True → not x:
        cond = re.sub(r'not\s+(\w+)\s*!=\s*True\b', r'not \1', cond)
        # x == False → not x:
        cond = re.sub(r'(\w+)\s*==\s*False\b', r'not \1', cond)
        # x == True → x:
        cond = re.sub(r'(\w+)\s*==\s*True\b', r'\1', cond)
        # x or True → x:
        cond = re.sub(r'(\w+)\s+or\s+True\b', r'\1', cond)
        # x and True → x:
        cond = re.sub(r'(\w+)\s+and\s+True\b', r'\1', cond)
        # x and False → False:（整个条件为假）
        cond = re.sub(r'(\w+)\s+and\s+False\b', r'False', cond)
        # False and x → False:
        cond = re.sub(r'False\s+and\s+(\w+)', r'False', cond)
        # True or x → True:
        cond = re.sub(r'True\s+or\s+(\w+)', r'True', cond)
        # x or False → x:
        cond = re.sub(r'(\w+)\s+or\s+False\b', r'\1', cond)

        if cond != original_cond:
            new_line = line.replace(original_cond, cond, 1)
            changed += 1
            modified.append(new_line)
        else:
            modified.append(line)

    if changed:
        return "\n".join(modified)
    return content


def _compress_empty_collections(content: str) -> str:
    """将冗长的空集合初始化替换为字面量。

    list() → []  （节省 1 token: 4→3）
    dict() → {}  （节省 1 token: 4→2）
    set()  → set()  （无更短形式，跳过）
    仅在赋值右侧或函数参数中匹配，不修改表达式内部。
    """
    lines = content.split("\n")
    modified = []
    changed = 0
    for line in lines:
        new_line = line
        # list() → []（避免替换 list(x) 等带参数的调用）
        new_line = re.sub(r'\blist\(\s*\)', '[]', new_line)
        # dict() → {}（避免替换 dict(x) 等）
        new_line = re.sub(r'\bdict\(\s*\)', '{}', new_line)
        if new_line != line:
            changed += 1
        modified.append(new_line)

    if changed:
        return "\n".join(modified)
    return content


def _merge_nested_ifs(content: str) -> str:
    """合并内层 if 与外层 if 的条件（当外层 if 体只有单个 if 时）。

    if x:
        if y:
            pass
    →  if x and y:
            pass

    仅处理外层 if 体只包含一个 if/elif/while 的情况。
    跳过有 else 分支的外层 if。
    """
    lines = content.split("\n")
    if len(lines) < 3:
        return content

    modified = list(lines)
    changed = False
    i = 0

    while i < len(lines):
        stripped = lines[i].strip()
        m = re.match(r'^(if|elif)\s+(.+):\s*$', stripped)
        if not m:
            i += 1
            continue

        outer_indent = len(lines[i]) - len(lines[i].lstrip())

        # 找到外层 if 体的第一个非空行（跳过空行）
        first_body_idx = None
        j = i + 1
        while j < len(lines):
            if lines[j].strip():
                ci = len(lines[j]) - len(lines[j].lstrip())
                if ci <= outer_indent:
                    break
                first_body_idx = j
                break
            j += 1

        if first_body_idx is None:
            i = j
            continue

        # 检查第一个 body 行是否是 if/elif
        inner_stripped = lines[first_body_idx].strip()
        inner_m = re.match(r'^(if|elif)\s+(.+):\s*$', inner_stripped)
        if not inner_m:
            i = first_body_idx + 1
            continue

        # 跳过内层 if 的 body，找到内层 if 块结束
        inner_indent = len(lines[first_body_idx]) - len(lines[first_body_idx].lstrip())
        inner_block_end = first_body_idx + 1
        while inner_block_end < len(lines):
            if lines[inner_block_end].strip():
                ci = len(lines[inner_block_end]) - len(lines[inner_block_end].lstrip())
                if ci <= inner_indent:
                    break
            inner_block_end += 1

        # 检查内层 if 是否有 else:
        has_inner_else = (inner_block_end < len(lines)
                          and lines[inner_block_end].strip().startswith("else:"))

        if has_inner_else:
            i = inner_block_end
            continue

        # 合并条件: if x: if y: → if x and y:
        outer_cond = m.group(2)
        inner_cond = inner_m.group(2)
        merged_cond = f"{outer_cond} and {inner_cond}"

        # 替换外层 if 行
        indent_str = lines[i][:outer_indent]
        modified[i] = f"{indent_str}if {merged_cond}:"
        # 内层 if 行缩进减少（去掉 if/elif 关键字和后续空白）
        inner_indent = len(lines[first_body_idx]) - len(lines[first_body_idx].lstrip())
        inner_body_indent = inner_indent + 4
        new_indent = inner_body_indent
        # 使用 regex match 精确定位条件起始位置
        inner_stripped = lines[first_body_idx].strip()
        inner_m = re.match(r'^(if|elif)\s+(.+):\s*$', inner_stripped)
        if inner_m:
            cond_start = inner_m.start(2)  # 条件在 stripped line 中的起始位置
            modified[first_body_idx] = lines[first_body_idx][:inner_indent] + inner_stripped[cond_start:]
        else:
            modified[first_body_idx] = lines[first_body_idx][:inner_indent] + lines[first_body_idx][inner_indent + 4:]
        changed = True
        i = j

    if changed:
        return "\n".join(modified)
    return content


def _simplify_ternary(content: str) -> str:
    """简化三元表达式的冗余情况。

    x if True else y → x
    x if False else y → y
    x if a else x → x（无论条件如何结果相同）
    """
    lines = content.split("\n")
    modified = []
    changed = 0

    for line in lines:
        new_line = line
        # x if True else y → x
        new_line = re.sub(
            r'(\S+)\s+if\s+True\s+else\s+(\S+)',
            r'\1',
            new_line
        )
        # x if False else y → y
        new_line = re.sub(
            r'(\S+)\s+if\s+False\s+else\s+(\S+)',
            r'\2',
            new_line
        )
        # x if cond else x → x
        new_line = re.sub(
            r'([A-Za-z_]\w*)\s+if\s+.+\s+else\s+\1\b',
            r'\1',
            new_line
        )
        if new_line != line:
            changed += 1
        modified.append(new_line)

    if changed:
        return "\n".join(modified)
    return content


def _remove_tuple_wrap_single(content: str) -> str:
    """移除单元素 tuple 包装（仅特定模式）。

    x, = [1] → x = 1      （tuple unpack 的单元素）
    (x,) → 此模式较复杂，暂不处理
    """
    lines = content.split("\n")
    modified = []
    changed = 0

    for line in lines:
        stripped = line.strip()
        # 匹配 x, = expr 模式（单元素 tuple unpack）
        m = re.match(r'^(\w+)\s*,\s*=\s*(.+)$', stripped)
        if m:
            var = m.group(1)
            expr = m.group(2).strip()
            indent = line[:len(line) - len(line.lstrip())]
            new_line = f"{indent}{var} = {expr}"
            modified.append(new_line)
            changed += 1
            continue

        modified.append(line)

    if changed:
        return "\n".join(modified)
    return content


def _list_to_generator(content: str) -> str:
    """将 list comprehension 替换为 generator expression（在 any/all/sum/join/max/min/sorted 中）。

    any([x for x in items]) → any(x for x in items)    # 节省 2 tokens
    all([x > 0 for x in items]) → all(x > 0 for x in items)
    sum([x for x in items]) → sum(x for x in items)
    "".join([a, b, c]) → "".join(a for a in [a, b, c])  # 仅列表字面量
    max([1, 2, 3]) → max(x for x in [1, 2, 3])
    """
    lines = content.split("\n")
    modified = []
    changed = 0

    for line in lines:
        new_line = line
        # any/all/sum: [expr for x in iter] → expr for x in iter
        for func in ("any", "all", "sum"):
            new_line = re.sub(
                rf'\b{func}\(\s*\[(.+?)\s+for\s+(\w+)\s+in\s+(.+?)\s*\]\s*\)',
                rf'{func}(\1 for \2 in \3)',
                new_line
            )
        # join: "sep".join([a, b, c]) → "sep".join(a for a in [a, b, c])
        new_line = re.sub(
            r'("(?:[^"\\]|\\.)*"|\'(?:[^\'\\]|\\.)*\')\s*\.\s*join\(\s*\[(.+?)\]\s*\)',
            r'\1.join(\2 for \2 in [\2])',
            new_line
        )
        # max/min/sorted: max([1, 2, 3]) → max(x for x in [1, 2, 3])
        for func in ("max", "min", "sorted"):
            new_line = re.sub(
                rf'\b{func}\(\s*\[(.+?)\]\s*\)',
                rf'{func}(x for x in [\1])',
                new_line
            )

        if new_line != line:
            changed += 1
        modified.append(new_line)

    if changed:
        return "\n".join(modified)
    return content


def _remove_not_not(content: str) -> str:
    """移除冗余的双重否定 `not not x` → `x`。

    仅在条件行（if/elif/while/assert/return/赋值）中处理。
    跳过字符串和注释中的内容。
    """
    lines = content.split("\n")
    modified = []
    changed = 0
    for line in lines:
        new_line = re.sub(r'\bnot\s+not\s+(\w+)', r'\1', line)
        if new_line != line:
            changed += 1
        modified.append(new_line)

    if changed:
        return "\n".join(modified)
    return content


def _remove_redundant_parens(content: str) -> str:
    """移除冗余的括号（仅安全的情况）。

    - `if (x > 0):` → `if x > 0:`
    - `if not (x):` → `if not x:`
    - 不处理可能改变运算优先级的情况（如 `(a or b) and c`）
    """
    lines = content.split("\n")
    modified = []
    changed = 0
    for line in lines:
        stripped = line.strip()
        # 仅处理 if/elif/while/assert 条件行
        if not re.match(r'^(if|elif|while|assert)\s+', stripped):
            modified.append(line)
            continue

        new_line = line
        # 情况 1: keyword (expr) → keyword expr
        new_line = re.sub(
            r'^(\s*(?:if|elif|while|assert)\s+)\(\s*([^()]+?)\s*\)(\s*:?)$',
            r'\1\2\3',
            new_line
        )
        # 情况 2: keyword not (expr) → keyword not expr
        new_line = re.sub(
            r'^(\s*(?:if|elif|while|assert)\s+not\s+)\(\s*([^()]+?)\s*\)(\s*:?)$',
            r'\1\2\3',
            new_line
        )
        if new_line != line:
            changed += 1
        modified.append(new_line)

    if changed:
        return "\n".join(modified)
    return content


def _remove_list_wrap(content: str) -> str:
    """移除冗余的 list()/tuple()/set() 包装。

    - `list(d.keys())` → `d.keys()`     # 节省 4 tokens
    - `list(d.values())` → `d.values()` # 节省 4 tokens
    - `list(d.items())` → `d.items()`   # 节省 4 tokens
    - `tuple()` → `()`                   # 节省 2 tokens
    - `set([1, 2, 3])` → `{1, 2, 3}`   # 节省 2-4 tokens
    - `list([1, 2, 3])` → `[1, 2, 3]`  # 节省 2 tokens
    - `frozenset([...])` → `frozenset(...)`  → keep (same length)
    """
    lines = content.split("\n")
    modified = []
    changed = 0

    for line in lines:
        new_line = line
        # list(d.keys()/values()/items()) → d.keys()/values()/items()
        new_line = re.sub(r'\blist\(\s*(d\.(keys|values|items)\(\))\s*\)', r'\1', new_line)
        # tuple() → ()
        new_line = re.sub(r'\btuple\(\s*\)', '()', new_line)
        # set([...]) → {...}（仅元素列表）
        new_line = re.sub(
            r'\bset\(\s*\[([^\]]*)\]\s*\)',
            r'{\1}',
            new_line
        )
        # list([...]) → [...]
        new_line = re.sub(
            r'\blist\(\s*\[([^\]]*)\]\s*\)',
            r'[\1]',
            new_line
        )
        # frozenset([...]) → frozenset(...) — keep, no savings

        if new_line != line:
            changed += 1
        modified.append(new_line)

    if changed:
        return "\n".join(modified)
    return content


def _flatten_nested_ternary(content: str) -> str:
    """扁平化嵌套三元表达式。

    仅处理安全的模式：
      `a if b else c if d else e` → `a if b and d else (c if not b else e)`
      （当 c == e 时，合并条件为 `a if b and d else c`）

    使用 AST 确保正确处理优先级。
    """
    try:
        import ast as _ast
        tree = _ast.parse(content)
    except SyntaxError:
        return content

    if not isinstance(tree, _ast.Module) or not tree.body:
        return content

    edits: dict[int, str] = {}
    found = False

    for node in _ast.walk(tree):
        if not isinstance(node, _ast.IfExp):
            continue
        # 检查 else 分支是否是另一个 IfExp（嵌套三元）
        if not isinstance(node.orelse, _ast.IfExp):
            continue

        inner = node.orelse
        # 尝试扁平化: a if b else (c if d else e)
        try:
            outer_true = _ast.unparse(node.body)
            outer_cond = _ast.unparse(node.test)
            inner_true = _ast.unparse(inner.body)
            inner_cond = _ast.unparse(inner.test)
            inner_false = _ast.unparse(inner.orelse)

            # 情况 1: a if b else a if d else e → a if b and d else e
            if outer_true == inner_true:
                flat = f"({outer_true}) if ({outer_cond}) and ({inner_cond}) else ({inner_false})"
            # 情况 2: a if b else c if d else c → a if b and d else c
            elif inner_false == outer_true:
                flat = f"({outer_true}) if ({outer_cond}) and ({inner_cond}) else ({inner_false})"
            # 情况 3: 一般扁平化（不合并条件，仅减少一层嵌套）
            else:
                flat = f"({outer_true}) if ({outer_cond}) else ({inner_true}) if ({inner_cond}) else ({inner_false})"
                # 检查是否真的更短
                original = _ast.unparse(node)
                if len(flat) >= len(original):
                    continue

            line_idx = node.lineno - 1
            if line_idx < len(content.split("\n")):
                edits[line_idx] = flat
                found = True
        except Exception:
            continue

    if not found:
        return content

    lines = content.split("\n")
    for line_idx in sorted(edits.keys(), reverse=True):
        if line_idx < len(lines):
            lines[line_idx] = edits[line_idx]
    return "\n".join(lines)


def _remove_unused_functions(content: str) -> str:
    """移除从未被调用的函数定义（AST 级分析）。

    扫描所有顶层函数定义，检查名称是否在模块其他位置被引用。
    保留：被调用的函数、作为参数传递的函数、被类引用的方法。
    跳过：main 函数、test 函数、以 _ 开头的私有函数。
    """
    try:
        import ast as _ast
        tree = _ast.parse(content)
    except SyntaxError:
        return content

    if not isinstance(tree, _ast.Module) or not tree.body:
        return content

    lines = content.split("\n")
    if not lines:
        return content

    # 收集所有函数定义及其调用位置
    func_defs: dict[str, list[int]] = {}  # name -> [line_indices of def]
    for node in _ast.walk(tree):
        if isinstance(node, (_ast.FunctionDef, _ast.AsyncFunctionDef)):
            name = node.name
            line_idx = node.lineno - 1
            func_defs.setdefault(name, []).append(line_idx)

    # 收集所有名称引用位置
    all_refs: set[str] = set()
    for node in _ast.walk(tree):
        if isinstance(node, _ast.Name):
            all_refs.add(node.id)
        elif isinstance(node, _ast.Attribute):
            # 收集 attr 名称（用于检测 self.method() 调用）
            all_refs.add(node.attr)

    # 检查每个函数是否被引用
    removals: dict[int, tuple[list[int], int]] = {}  # def_line -> (decorator_lines, end_line)
    for name, def_lines in func_defs.items():
        # 跳过 main/test/_ 开头的函数
        if name in ('main', 'test') or name.startswith('_'):
            continue
        # 检查函数名是否在引用中（排除自身定义位置）
        if name not in all_refs:
            # 检查是否是类的方法（通过 self.xxx 检测）
            is_method = any(
                node for node in _ast.walk(tree)
                if isinstance(node, _ast.FunctionDef)
                and any(
                    isinstance(c, _ast.Attribute) and c.attr == name
                    for c in _ast.walk(node)
                )
            )
            if not is_method:
                for line_idx in def_lines:
                    # 找到装饰器行和函数体范围
                    for node in _ast.walk(tree):
                        if isinstance(node, (_ast.FunctionDef, _ast.AsyncFunctionDef)):
                            if node.lineno - 1 == line_idx:
                                decorator_lines = [dec.lineno - 1 for dec in node.decorator_list]
                                end_line = getattr(node, 'end_lineno', line_idx + 20) - 1
                                removals[line_idx] = (sorted(decorator_lines), end_line)
                                break

    if not removals:
        return content

    # 从后往前删除（避免行号偏移）
    modified = list(lines)
    for line_idx, (dec_lines, end_line) in sorted(removals.items(), reverse=True):
        all_lines_to_remove = set(dec_lines)
        all_lines_to_remove.update(range(line_idx, min(end_line + 1, len(modified))))
        for k in sorted(all_lines_to_remove, reverse=True):
            if k < len(modified):
                modified[k] = ""

    return "\n".join(modified)


def _inline_single_use_vars(content: str) -> str:
    """内联仅使用一次的局部变量（AST 级分析）。

    模式：x = expr\n...use x once... → 直接用 expr 替换 x
    仅内联简单的单次赋值-使用对，跳过：
    - self/cls 属性
    - 复杂表达式（函数调用、列表推导等）
    - 跨函数边界的变量
    - 多目标赋值 (a, b = ...)
    """
    try:
        import ast as _ast
        tree = _ast.parse(content)
    except SyntaxError:
        return content

    lines = content.split("\n")
    if not lines:
        return content

    edits: dict[int, str] = {}
    inline_candidates: dict[str, tuple[int, str, int]] = {}  # name -> (assign_line, expr_str, use_line)

    for node in _ast.walk(tree):
        if not isinstance(node, (_ast.FunctionDef, _ast.AsyncFunctionDef)):
            continue

        func_name = node.name
        func_start = node.lineno - 1
        func_end = getattr(node, 'end_lineno', len(lines)) - 1

        # 收集函数内所有赋值和使用
        assignments: dict[str, list[tuple[int, str]]] = {}  # name -> [(line_idx, expr_str)]
        uses: dict[str, list[int]] = {}  # name -> [line_indices]

        for child in _ast.walk(node):
            if isinstance(child, _ast.Assign):
                if len(child.targets) == 1 and isinstance(child.targets[0], _ast.Name):
                    name = child.targets[0].id
                    if name in ('self', 'cls', 'super'):
                        continue
                    try:
                        expr_str = _ast.unparse(child.value)
                    except Exception:
                        expr_str = None
                    if expr_str and len(expr_str) < 50:  # 只内联短表达式
                        assignments.setdefault(name, []).append((child.lineno - 1, expr_str))
            elif isinstance(child, _ast.AnnAssign):
                if isinstance(child.target, _ast.Name) and child.value:
                    name = child.target.id
                    if name in ('self', 'cls', 'super'):
                        continue
                    try:
                        expr_str = _ast.unparse(child.value)
                    except Exception:
                        expr_str = None
                    if expr_str and len(expr_str) < 50:
                        assignments.setdefault(name, []).append((child.lineno - 1, expr_str))
            elif isinstance(child, _ast.Name) and isinstance(child.ctx, _ast.Load):
                uses.setdefault(child.id, []).append(child.lineno - 1)

        # 找出可内联的变量（恰好一次赋值，恰好一次使用）
        for name, assigns in assignments.items():
            var_uses = uses.get(name, [])
            if len(assigns) == 1 and len(var_uses) == 1:
                assign_line, expr = assigns[0]
                use_line = var_uses[0]
                if assign_line != use_line:  # 不同行才内联
                    inline_candidates[name] = (assign_line, expr, use_line)

    # 应用内联（从上往下处理，并更新依赖表达式的值）
    sorted_candidates = sorted(inline_candidates.items(), key=lambda x: x[1][0])
    for idx, (name, (assign_line, expr, use_line)) in enumerate(sorted_candidates):
        if use_line < len(lines):
            # 替换使用位置：用 expr 替换 name
            line = lines[use_line]
            new_line = re.sub(rf'\b{re.escape(name)}\b', expr, line)
            if new_line != line:
                edits[use_line] = new_line
                # 清除赋值行
                edits[assign_line] = ""
            # 更新后续候选变量的表达式（将已内联变量替换为其表达式）
            for later_idx in range(idx + 1, len(sorted_candidates)):
                later_name, (later_assign, later_expr, later_use) = sorted_candidates[later_idx]
                updated_expr = re.sub(rf'\b{re.escape(name)}\b', expr, later_expr)
                if updated_expr != later_expr:
                    sorted_candidates[later_idx] = (later_name, (later_assign, updated_expr, later_use))

    if not edits:
        return content

    modified = list(lines)
    for line_idx in sorted(edits.keys(), reverse=True):
        if line_idx < len(modified):
            modified[line_idx] = edits[line_idx]

    return "\n".join(modified)


def _remove_unused_classes(content: str) -> str:
    """移除从未被实例化或引用的类定义。

    检查类名是否在其他地方被引用（实例化、继承、类型注解等）。
    保留有基类的类（可能是框架需要的）。
    跳过以 _ 开头的私有类。
    """
    try:
        import ast as _ast
        tree = _ast.parse(content)
    except SyntaxError:
        return content

    if not isinstance(tree, _ast.Module) or not tree.body:
        return content

    lines = content.split("\n")
    if not lines:
        return content

    # 收集类定义
    class_defs: dict[str, int] = {}  # name -> def_line_idx
    for node in _ast.walk(tree):
        if isinstance(node, _ast.ClassDef):
            name = node.name
            if name.startswith('_'):
                continue
            # 跳过有基类的类（可能是框架需要的）
            if node.bases:
                continue
            class_defs[name] = node.lineno - 1

    if not class_defs:
        return content

    # 收集所有名称引用
    all_refs: set[str] = set()
    for node in _ast.walk(tree):
        if isinstance(node, _ast.Name):
            all_refs.add(node.id)
        elif isinstance(node, _ast.Attribute):
            all_refs.add(node.attr)

    # 找出未使用的类
    removals: dict[int, tuple[list[int], int]] = {}  # def_line -> (decorator_lines, end_line)
    for name, def_line in class_defs.items():
        if name not in all_refs:
            # 找到类体范围和装饰器行
            for node in _ast.walk(tree):
                if isinstance(node, _ast.ClassDef) and node.name == name and node.lineno - 1 == def_line:
                    decorator_lines = [dec.lineno - 1 for dec in node.decorator_list]
                    end_line = getattr(node, 'end_lineno', def_line + 20) - 1
                    removals[def_line] = (sorted(decorator_lines), end_line)
                    break

    if not removals:
        return content

    modified = list(lines)
    for start_line, (dec_lines, end_line) in sorted(removals.items(), reverse=True):
        all_lines_to_remove = set(dec_lines)
        all_lines_to_remove.update(range(start_line, min(end_line + 1, len(modified))))
        for k in sorted(all_lines_to_remove, reverse=True):
            if k < len(modified):
                modified[k] = ""

    return "\n".join(modified)


def _invert_dead_if(content: str) -> str:
    """反演死 if 条件：将 `if x: pass else: body` 反转为 `if not x: body`。

    当 if 分支只包含 pass/空语句，else 分支有实质代码时，
    反演条件可以移除整个 else 块，节省 4-6 tokens。
    """
    lines = content.split("\n")
    if len(lines) < 3:
        return content

    modified = list(lines)
    changed = False
    i = 0

    while i < len(lines):
        stripped = lines[i].strip()
        m = re.match(r'^(if|elif)\s+(.+):\s*$', stripped)
        if not m:
            i += 1
            continue

        header_indent = len(lines[i]) - len(lines[i].lstrip())
        keyword = m.group(1)
        cond = m.group(2)

        # 检查 if 体是否只有 pass/注释（跳过空行和注释）
        body_start = i + 1
        body_end = body_start
        body_only_pass = True
        while body_end < len(lines):
            if not lines[body_end].strip():
                body_end += 1
                continue
            ci = len(lines[body_end]) - len(lines[body_end].lstrip())
            if ci <= header_indent:
                break
            line_content = lines[body_end].strip()
            if line_content.startswith("#"):
                body_end += 1
                continue
            if line_content != "pass":
                body_only_pass = False
            body_end += 1

        if not body_only_pass or body_end >= len(lines):
            i = body_end
            continue

        # 检查下一个同级语句是否是 else:
        next_stripped = lines[body_end].strip()
        if not next_stripped.startswith("else:"):
            i = body_end
            continue

        # 收集 else 体
        else_body_start = body_end + 1
        else_body_end = else_body_start
        while else_body_end < len(lines):
            if not lines[else_body_end].strip():
                else_body_end += 1
                continue
            ci = len(lines[else_body_end]) - len(lines[else_body_end].lstrip())
            if ci <= header_indent:
                break
            else_body_end += 1

        # 反演: if x: pass else: body → if not x: body
        inverted = f"if not {cond}:"
        indent_str = lines[i][:header_indent]
        modified[i] = f"{indent_str}{inverted}"
        # 清除 if 体（只有 pass）
        for k in range(body_start, body_end):
            modified[k] = ""
        # 清除 else 行
        modified[body_end] = ""
        # 降低 else 体缩进（去掉 4 个空格）
        for k in range(else_body_start, else_body_end):
            if lines[k].strip():
                current_indent = len(lines[k]) - len(lines[k].lstrip())
                if current_indent > header_indent:
                    modified[k] = lines[k][header_indent:]
        changed = True
        i = else_body_end

    if changed:
        return "\n".join(modified)
    return content


def _merge_duplicate_conditions(content: str) -> str:
    """合并 if/elif 链中相同条件的连续分支。

    模式：
        if x:
            return 1
        elif x:
            return 1
    →  if x:
            return 1
    """
    try:
        import ast as _ast
        tree = _ast.parse(content)
    except SyntaxError:
        return content

    if not isinstance(tree, _ast.Module) or not tree.body:
        return content

    # 只处理顶层 if 语句的 elif 链
    for node in tree.body:
        if not isinstance(node, _ast.If):
            continue

        conditions_seen: dict[str, int] = {}  # unparsed_cond -> elif_line_idx
        removals: list[int] = []

        # 记录第一个 if 的条件
        try:
            first_cond = _ast.unparse(node.test)
            conditions_seen[first_cond] = node.lineno - 1
        except Exception:
            continue

        # 遍历 elif 链
        current = node
        while isinstance(current, _ast.If):
            try:
                cond_str = _ast.unparse(current.test)
            except Exception:
                break

            # 检查 elif 条件是否重复
            if current is not node:
                if cond_str in conditions_seen:
                    removals.append(current.lineno - 1)
                else:
                    conditions_seen[cond_str] = current.lineno - 1

            # 找到下一个 elif/else
            if current.orelse and len(current.orelse) == 1 and isinstance(current.orelse[0], _ast.If):
                current = current.orelse[0]
            else:
                break

        if not removals:
            continue

        # 从后往前删除重复的 elif 行
        lines = content.split("\n")
        modified = list(lines)
        for line_idx in sorted(removals, reverse=True):
            if line_idx < len(modified):
                # 清除 elif 行及其整个分支体
                indent = len(modified[line_idx]) - len(modified[line_idx].lstrip())
                modified[line_idx] = ""
                j = line_idx + 1
                while j < len(lines):
                    if not modified[j].strip():
                        j += 1
                        continue
                    ci = len(modified[j]) - len(modified[j].lstrip())
                    if ci <= indent:
                        break
                    modified[j] = ""
                    j += 1
        content = "\n".join(modified)

    return content


def _range_len_to_enumerate(content: str) -> str:
    """将 `for i in range(len(items))` 模式转换为更紧凑的形式。

    两种转换：
    1. 循环变量仅用于索引 seq[i] → `for item in seq:`
    2. 循环变量还有其他用途 → `for i, item in enumerate(seq):`

    同时替换循环体中的 seq[i] 为 item。
    """
    try:
        import ast as _ast
        tree = _ast.parse(content)
    except SyntaxError:
        return content

    if not isinstance(tree, _ast.Module) or not tree.body:
        return content

    edits: dict[int, str] = {}
    found = False
    content_lines = content.split("\n")

    for node in _ast.walk(tree):
        if not isinstance(node, _ast.For):
            continue

        # 检查迭代器: for i in range(len(seq))
        if not isinstance(node.iter, _ast.Call):
            continue
        if not isinstance(node.iter.func, _ast.Name) or node.iter.func.id != 'range':
            continue
        if len(node.iter.args) != 1:
            continue

        range_arg = node.iter.args[0]
        # range(len(seq))
        if not isinstance(range_arg, _ast.Call):
            continue
        if not isinstance(range_arg.func, _ast.Name) or range_arg.func.id != 'len':
            continue
        if len(range_arg.args) != 1:
            continue

        seq_node = range_arg.args[0]
        if not isinstance(seq_node, _ast.Name):
            continue
        seq_name = seq_node.id

        # 循环变量
        if not isinstance(node.target, _ast.Name):
            continue
        loop_var = node.target.id

        # 收集循环体中所有对 seq[loop_var] 的索引访问
        # 以及 loop_var 的所有使用
        seq_subscript_uses = 0  # seq[loop_var] 的使用次数
        loop_var_total_uses = 0  # loop_var 的总使用次数

        for child in _ast.walk(node):
            if isinstance(child, _ast.Subscript):
                sub = child
                if (isinstance(sub.value, _ast.Name) and sub.value.id == seq_name
                        and isinstance(sub.slice, _ast.Name) and sub.slice.id == loop_var):
                    seq_subscript_uses += 1
            if isinstance(child, _ast.Name) and child.id == loop_var:
                loop_var_total_uses += 1

        if seq_subscript_uses == 0:
            continue  # loop_var 没有用于 seq[i]，无法简化

        # 决定转换方式
        if seq_subscript_uses == loop_var_total_uses:
            # loop_var 仅用于 seq[i]，可以去掉循环变量
            item_name = seq_name[:-1] if len(seq_name) > 1 and seq_name.endswith('s') else seq_name + '_item'
            new_for_header = f"{content_lines[node.lineno - 1][:node.col_offset]}for {item_name} in {seq_name}:"
            # 替换循环体中的 seq[i] 为 item
            body_edits = {}
            for child in _ast.walk(node):
                if isinstance(child, _ast.Subscript):
                    sub = child
                    if (isinstance(sub.value, _ast.Name) and sub.value.id == seq_name
                            and isinstance(sub.slice, _ast.Name) and sub.slice.id == loop_var):
                        s = getattr(sub, 'lineno', None)
                        sc = getattr(sub, 'col_offset', None)
                        ec = getattr(sub, 'end_col_offset', None)
                        if s is not None and sc is not None and ec is not None:
                            line = content_lines[s - 1]
                            new_line = line[:sc] + item_name + line[ec:]
                            if new_line != line:
                                body_edits[s - 1] = new_line
            # 合并 edits
            edits[node.lineno - 1] = new_for_header
            for li, new_line in body_edits.items():
                if li not in edits:
                    edits[li] = new_line
            found = True
        else:
            # loop_var 有其他用途，使用 enumerate
            item_name = seq_name[:-1] if len(seq_name) > 1 and seq_name.endswith('s') else seq_name + '_item'
            new_for_header = f"{content_lines[node.lineno - 1][:node.col_offset]}for {loop_var}, {item_name} in enumerate({seq_name}):"
            # 替换循环体中的 seq[i] 为 item
            body_edits = {}
            for child in _ast.walk(node):
                if isinstance(child, _ast.Subscript):
                    sub = child
                    if (isinstance(sub.value, _ast.Name) and sub.value.id == seq_name
                            and isinstance(sub.slice, _ast.Name) and sub.slice.id == loop_var):
                        s = getattr(sub, 'lineno', None)
                        sc = getattr(sub, 'col_offset', None)
                        ec = getattr(sub, 'end_col_offset', None)
                        if s is not None and sc is not None and ec is not None:
                            line = content_lines[s - 1]
                            new_line = line[:sc] + item_name + line[ec:]
                            if new_line != line:
                                body_edits[s - 1] = new_line
            edits[node.lineno - 1] = new_for_header
            for li, new_line in body_edits.items():
                if li not in edits:
                    edits[li] = new_line
            found = True

    if not found:
        return content

    lines = list(content_lines)
    for line_idx in sorted(edits.keys(), reverse=True):
        if line_idx < len(lines):
            lines[line_idx] = edits[line_idx]
    return "\n".join(lines)


def _remove_return_none_after_none_check(content: str) -> str:
    """简化 `if x is None: return None` → `if x is None: return`。

    在 stripped 模式下，return None 末尾的 None 是冗余的。
    """
    lines = content.split("\n")
    if len(lines) < 2:
        return content

    modified = list(lines)
    changed = 0

    i = 0
    while i < len(lines):
        stripped = lines[i].strip()
        m = re.match(r'^if\s+(\w+)\s+is\s+None:\s*$', stripped)
        if m and i + 1 < len(lines):
            next_stripped = lines[i + 1].strip()
            if next_stripped == "return None":
                indent = len(lines[i + 1]) - len(lines[i + 1].lstrip())
                indent_str = lines[i + 1][:indent]
                modified[i + 1] = f"{indent_str}return"
                changed += 1
        i += 1

    if changed:
        return "\n".join(modified)
    return content


# ── 新深度优化函数 ────────────────────────────────────────────────────────

def _remove_unreachable_code(content: str) -> str:
    """删除 return/raise/break/continue 之后的不可达代码。

    例如：
        if x:
            return
            y = 1  # 不可达
        → if x: return

    逐块处理每个函数/类方法/模块顶层代码块。
    """
    try:
        import ast as _ast
        tree = _ast.parse(content)
    except SyntaxError:
        return content

    lines = content.split("\n")
    edits: dict[int, str] = {}
    _JUMP_TYPES = (_ast.Return, _ast.Raise, _ast.Break, _ast.Continue)

    def _clear_node_lines(node) -> None:
        """清除单个语句的所有源代码行。"""
        s = getattr(node, 'lineno', None)
        e = getattr(node, 'end_lineno', None)
        if s is None:
            return
        for li in range(s - 1, (e if e else s)):
            if 0 <= li < len(lines):
                edits[li] = ""

    def _process_block(block: list, depth: int = 0) -> bool:
        """处理单个语句块。返回 True 表示块以不可达状态结束。"""
        reachable = True
        for node in block:
            if not reachable:
                _clear_node_lines(node)
                continue

            if isinstance(node, _JUMP_TYPES):
                reachable = False
                _clear_node_lines(node)
                continue

            if isinstance(node, _ast.If):
                _process_block(node.body, depth + 1)
                if node.orelse:
                    _process_block(node.orelse, depth + 1)
            elif isinstance(node, (_ast.For, _ast.While)):
                _process_block(node.body, depth + 1)
                if hasattr(node, 'orelse') and node.orelse:
                    _process_block(node.orelse, depth + 1)
            elif isinstance(node, _ast.Try):
                _process_block(node.body, depth + 1)
                for handler in node.handlers:
                    _process_block(handler.body, depth + 1)
                    _clear_node_lines(handler)
                if node.orelse:
                    _process_block(node.orelse, depth + 1)
                if node.finalbody:
                    _process_block(node.finalbody, depth + 1)
            elif isinstance(node, (_ast.FunctionDef, _ast.AsyncFunctionDef)):
                _process_block(node.body, depth + 1)
            elif isinstance(node, _ast.ClassDef):
                _process_block(node.body, depth + 1)
            elif isinstance(node, _ast.With):
                _process_block(node.body, depth + 1)

        return reachable

    if isinstance(tree, _ast.Module):
        _process_block(tree.body)

    if edits:
        modified = list(lines)
        for idx in sorted(edits.keys(), reverse=True):
            modified[idx] = ""
        content = "\n".join(modified)

    return content


def _simplify_enumerate_start_zero(content: str) -> str:
    """enumerate(seq, 0) → enumerate(seq)。

    节省 ~6 tokens/调用（Python 3 中 start 默认为 0）。
    使用精确的 regex 避免误匹配。
    """
    # 匹配 enumerate(... , 0) 或 enumerate(... ,0) 或 enumerate(x, 0)
    content = re.sub(r'enumerate\(\s*([^,]+?)\s*,\s*0\s*\)', r'enumerate(\1)', content)
    return content


def _remove_unused_except_blocks(content: str) -> str:
    """移除捕获异常类型从未在 try 块中抛出的 except 块。

    例如：
        try:
            int(x)
        except KeyError:
            pass
        → try: int(x)

    基于 AST 分析：检查 try 块中是否出现了 except 捕获的类型。
    仅移除 body 为 pass 或仅含注释的 except 块。
    """
    try:
        import ast as _ast
        tree = _ast.parse(content)
    except SyntaxError:
        return content

    if not isinstance(tree, _ast.Module) or not tree.body:
        return content

    # 收集 try 块中实际抛出的异常类型（通过 Raise 语句）
    def _collect_raised_types(node) -> set[str]:
        raised = set()
        for child in _ast.walk(node):
            if isinstance(child, _ast.Raise) and child.exc:
                if isinstance(child.exc, _ast.Call) and isinstance(child.exc.func, _ast.Name):
                    raised.add(child.exc.func.id)
                elif isinstance(child.exc, _ast.Name):
                    raised.add(child.exc.id)
                elif isinstance(child.exc, _ast.Attribute):
                    raised.add(child.exc.attr)
        return raised

    edits: dict[int, str] = {}
    changed = False

    def _process_try(node: _ast.Try) -> bool:
        """处理单个 Try 节点，移除无效 except 块。返回是否修改。"""
        nonlocal changed
        if not node.handlers:
            return False

        raised_types = _collect_raised_types(node)

        kept_handlers = []
        for handler in node.handlers:
            caught_types = set()
            if handler.type:
                if isinstance(handler.type, _ast.Name):
                    caught_types.add(handler.type.id)
                elif isinstance(handler.type, _ast.Tuple):
                    for elt in handler.type.elts:
                        if isinstance(elt, _ast.Name):
                            caught_types.add(elt.id)

            # 检查是否有任何捕获的类型在 try 块中被抛出
            if not caught_types.intersection(raised_types):
                # 检查 handler body 是否只有 pass（或空）
                body = handler.body
                is_passthrough = (
                    len(body) == 1
                    and isinstance(body[0], _ast.Pass)
                )
                if is_passthrough:
                    # 标记 except 块的所有行为删除
                    start = handler.lineno - 1
                    end = getattr(handler, 'end_lineno', None)
                    if end:
                        for line_idx in range(start, end):
                            if line_idx < len(content.split("\n")):
                                edits[line_idx] = ""
                                changed = True
                    continue
            kept_handlers.append(handler)

        return changed

    for node in tree.body:
        if isinstance(node, _ast.Try):
            _process_try(node)
        elif isinstance(node, (_ast.FunctionDef, _ast.AsyncFunctionDef, _ast.ClassDef)):
            for child in _ast.walk(node):
                if isinstance(child, _ast.Try):
                    _process_try(child)

    if changed:
        lines = content.split("\n")
        modified = list(lines)
        for idx in sorted(edits.keys(), reverse=True):
            modified[idx] = ""
        content = "\n".join(modified)

    return content


def _simplify_super_calls(content: str) -> str:
    """super(ClassName, self) → super()  （Python 3 中简写）。

    节省 ~12-18 tokens/调用。仅匹配类方法中的模式（self/cls 作为第二参数）。
    """
    # super(ClassName, self) 或 super(ClassName, cls)
    content = re.sub(
        r'super\(\s*(\w+)\s*,\s*(?:self|cls)\s*\)',
        'super()',
        content
    )
    return content


def _collapse_duplicate_lines(content: str) -> str:
    """合并且块内连续重复的语句为单个 + 标记。

    例如：
        x = 1
        x = 1
        -> x = 1  # *2

    在 Python 函数/方法/模块顶层代码块级别进行。
    仅合并非空、非注释的行。
    """
    try:
        import ast as _ast
        tree = _ast.parse(content)
    except SyntaxError:
        return content

    if not isinstance(tree, _ast.Module) or not tree.body:
        return content

    lines = content.split("\n")
    edits = {}
    changed = False

    KW = ('def ', 'class ', 'if ', 'for ', 'while ', 'with ', 'try:', 'except', 'finally:', 'elif', 'else:')

    def _collapse_in_block(block):
        nonlocal changed
        if len(block) < 2:
            return

        stmt_lines = []
        for node in block:
            s = getattr(node, 'lineno', None)
            e = getattr(node, 'end_lineno', None)
            if s and e:
                stmt_lines.append((s - 1, e - 1))

        i = 0
        while i < len(stmt_lines) - 1:
            s1, e1 = stmt_lines[i]
            s2, e2 = stmt_lines[i + 1]
            if e1 - s1 == 0 and e2 - s2 == 0:
                l1 = lines[s1].strip()
                l2 = lines[s2].strip()
                if l1 and not l1.startswith("#") and l1 == l2:
                    if not any(l1.startswith(k) for k in KW):
                        edits[s2] = lines[s2] + "  # *2"
                        changed = True
                        j = i + 2
                        count = 2
                        while j < len(stmt_lines):
                            sj, ej = stmt_lines[j]
                            if ej - sj == 0 and lines[sj].strip() == l1:
                                edits[sj] = lines[sj] + "  # *" + str(count + 1)
                                count += 1
                                j += 1
                            else:
                                break
                        i = j
                        continue
            i += 1

    def _process_node(node):
        if isinstance(node, (_ast.FunctionDef, _ast.AsyncFunctionDef)):
            _collapse_in_block(node.body)
        elif isinstance(node, _ast.ClassDef):
            _collapse_in_block(node.body)
        elif isinstance(node, _ast.Try):
            _collapse_in_block(node.body)
            for h in node.handlers:
                _collapse_in_block(h.body)
            if node.orelse:
                _collapse_in_block(node.orelse)

    for node in tree.body:
        _process_node(node)

    if changed:
        modified = list(lines)
        for idx in sorted(edits.keys(), reverse=True):
            modified[idx] = edits[idx]
        content = "\n".join(modified)

    return content


def _compress_asserts(content: str) -> str:
    """删除恒真/恒假断言（assert True, assert False, assert 1 == 1 等）。

    这些断言不影响程序逻辑（恒真=无意义，恒假=必停），且浪费 tokens。
    使用 AST 精确分析断言条件，扫描模块顶层和所有嵌套函数/类。
    """
    try:
        import ast as _ast
        tree = _ast.parse(content)
    except SyntaxError:
        return content

    if not isinstance(tree, _ast.Module) or not tree.body:
        return content

    edits: dict[int, str] = {}
    changed = False
    lines = content.split("\n")

    def _eval_assert_condition(test) -> bool | None:
        """求值断言条件。返回 True 表示恒真, False 表示恒假, None 表示不确定。"""

        OPS = {
            _ast.Eq: lambda a, b: a == b,
            _ast.NotEq: lambda a, b: a != b,
            _ast.Lt: lambda a, b: a < b,
            _ast.LtE: lambda a, b: a <= b,
            _ast.Gt: lambda a, b: a > b,
            _ast.GtE: lambda a, b: a >= b,
            _ast.Is: lambda a, b: a is b,
            _ast.IsNot: lambda a, b: a is not b,
            _ast.In: lambda a, b: a in b,
            _ast.NotIn: lambda a, b: a not in b,
        }

        def _const_value(node):
            """提取字面值。"""
            if isinstance(node, _ast.Constant):
                return node.value
            if isinstance(node, _ast.UnaryOp) and isinstance(node.op, _ast.USub):
                val = _const_value(node.operand)
                if val is not None:
                    return -val
            if isinstance(node, _ast.UnaryOp) and isinstance(node.op, _ast.UAdd):
                val = _const_value(node.operand)
                if val is not None:
                    return +val
            return None

        def _eval_compare(node):
            """手动求值比较表达式。"""
            if not isinstance(node, _ast.Compare) or len(node.ops) != len(node.comparators):
                return None
            left = _const_value(node.left)
            if left is None:
                return None
            for op, comp in zip(node.ops, node.comparators):
                right = _const_value(comp)
                if right is None:
                    return None
                op_fn = OPS.get(type(op))
                if op_fn is None:
                    return None
                if not op_fn(left, right):
                    return False
                left = right
            return True

        if isinstance(test, _ast.Constant):
            return bool(test.value)
        elif isinstance(test, _ast.Compare):
            result = _eval_compare(test)
            return bool(result) if result is not None else None
        elif isinstance(test, _ast.BoolOp):
            # 手动求值 BoolOp (and/or)
            if isinstance(test.op, _ast.And):
                result = True
                for v in test.values:
                    r = _eval_assert_condition(v)
                    if r is False:
                        return False
                    if r is None:
                        return None
                return True
            elif isinstance(test.op, _ast.Or):
                result = False
                for v in test.values:
                    r = _eval_assert_condition(v)
                    if r is True:
                        return True
                    if r is None:
                        return None
                return False
            return None
        elif isinstance(test, _ast.UnaryOp) and isinstance(test.op, _ast.Not):
            inner = _eval_assert_condition(test.operand)
            if inner is not None:
                return not inner
        elif isinstance(test, _ast.Call):
            # 处理 len() 调用等简单情况
            pass
        return None

    def _process_assert(node: _ast.Assert) -> None:
        nonlocal changed
        result = _eval_assert_condition(node.test)
        if result is not None:  # 恒真或恒假
            start = node.lineno - 1
            end = getattr(node, 'end_lineno', node.lineno)
            for line_idx in range(start, end):
                if line_idx < len(lines):
                    edits[line_idx] = ""
                    changed = True

    def _walk(node) -> None:
        """递归遍历 AST 节点，查找断言。"""
        if isinstance(node, _ast.Assert):
            _process_assert(node)
        for child in _ast.iter_child_nodes(node):
            _walk(child)

    _walk(tree)

    if changed:
        modified = list(lines)
        for idx in sorted(edits.keys(), reverse=True):
            modified[idx] = ""
        content = "\n".join(modified)

    return content


def _merge_same_body_conditions(content: str) -> str:
    """合并具有相同 body 的连续 if/elif 分支的条件。

    例如：
        if x == 1:
            handle()
        elif x == 2:
            handle()
        → if x == 1 or x == 2:
            handle()

    仅合并连续的、body 完全相同的 if/elif 分支。
    使用 AST 精确匹配 body 内容。
    """
    try:
        import ast as _ast
        tree = _ast.parse(content)
    except SyntaxError:
        return content

    if not isinstance(tree, _ast.Module) or not tree.body:
        return content

    edits: dict[int, str] = {}
    changed = False

    def _body_source(node) -> str:
        """获取语句块的源代码（仅 body 中的实际语句，不含 elif/else/finally 等后续部分）。"""
        if not hasattr(node, 'body') or not node.body:
            return ""
        lines = content.split("\n")
        parts = []
        for stmt in node.body:
            s = getattr(stmt, 'lineno', None)
            e = getattr(stmt, 'end_lineno', None)
            if s is not None and e is not None:
                parts.append("\n".join(lines[s - 1:e]))
        return "\n".join(parts)

    def _body_hash(node) -> str:
        """生成语句块的唯一标识（用于比较是否相同）。"""
        import hashlib
        src = _body_source(node)
        return hashlib.md5(src.encode()).hexdigest()

    def _try_unparse(node) -> str:
        """尝试反编译节点为条件表达式。"""
        try:
            return _ast.unparse(node.test)
        except Exception:
            return None

    def _process_chain(chain: list) -> None:
        """处理一个 if/elif 链，合并相同 body 的连续分支。"""
        nonlocal changed
        if len(chain) < 2:
            return

        i = 0
        while i < len(chain) - 1:
            current = chain[i]
            current_hash = _body_hash(current)

            # 收集所有与 current 有相同 body 的后续 elif
            merged_conditions = []
            j = i + 1
            while j < len(chain):
                next_node = chain[j]
                if _body_hash(next_node) == current_hash:
                    cond = _try_unparse(next_node)
                    if cond:
                        merged_conditions.append(cond)
                    j += 1
                else:
                    break

            if merged_conditions:
                # 合并条件: if A: → if A or B or C:
                first_cond = _try_unparse(current)
                if first_cond:
                    combined = " or ".join([first_cond] + merged_conditions)
                    # 修改第一个 if 的条件行
                    start_line = current.lineno - 1
                    lines = content.split("\n")
                    indent = len(lines[start_line]) - len(lines[start_line].lstrip())
                    indent_str = lines[start_line][:indent]
                    new_line = f"{indent_str}if {combined}:"
                    edits[start_line] = new_line
                    changed = True

                    # 删除已合并的 elif 行
                    for k in range(i + 1, i + 1 + len(merged_conditions)):
                        elif_node = chain[k]
                        elif_start = elif_node.lineno - 1
                        elif_end = getattr(elif_node, 'end_lineno', elif_node.lineno)
                        for line_idx in range(elif_start, elif_end):
                            if line_idx < len(lines):
                                edits[line_idx] = ""

                    i = j
                    continue

            i += 1

    # 遍历所有 if 链
    def _find_if_chains(node) -> list:
        """从节点中提取所有 if/elif 链（递归遍历所有嵌套块）。"""
        chains = []
        if isinstance(node, _ast.If):
            chain = [node]
            current = node
            while current.orelse and len(current.orelse) == 1 and isinstance(current.orelse[0], _ast.If):
                chain.append(current.orelse[0])
                current = current.orelse[0]
            chains.append(chain)

        # 递归遍历所有子节点中的块
        for child in _ast.iter_child_nodes(node):
            if isinstance(child, _ast.If):
                chains.extend(_find_if_chains(child))
            elif hasattr(child, 'body') and isinstance(child.body, list):
                bodies = list(child.body)
                orelse = getattr(child, 'orelse', [])
                if orelse and not isinstance(child, _ast.If) and isinstance(orelse, list):
                    bodies = bodies + orelse
                finalbody = getattr(child, 'finalbody', [])
                if finalbody and isinstance(finalbody, list):
                    bodies = bodies + finalbody
                for sub in bodies:
                    if isinstance(sub, _ast.If):
                        chains.extend(_find_if_chains(sub))
        return chains

    for node in tree.body:
        chains = _find_if_chains(node)
        for chain in chains:
            _process_chain(chain)

    if changed:
        lines = content.split("\n")
        modified = list(lines)
        for idx in sorted(edits.keys(), reverse=True):
            modified[idx] = edits[idx]
        content = "\n".join(modified)

    return content


def _remove_dead_after_loop(content: str) -> str:
    """移除 for/while 循环后的不可达代码（当循环无条件终止时）。

    例如：
        for i in range(10):
            process(i)
        print("done")  # 可达（循环正常结束）
        → 保留

        while True:
            do_something()
        print("never")  # 不可达
        → 删除

    仅处理确定性的无限循环（while True, while 1 等）。
    """
    try:
        import ast as _ast
        tree = _ast.parse(content)
    except SyntaxError:
        return content

    if not isinstance(tree, _ast.Module) or not tree.body:
        return content

    edits: dict[int, str] = {}
    changed = False

    def _is_infinite_loop(node) -> bool:
        """检查是否为确定性无限循环。"""
        if not isinstance(node, (_ast.For, _ast.While)):
            return False
        if isinstance(node, _ast.While):
            # while True, while 1
            try:
                val = _ast.literal_eval(node.test)
                return bool(val)
            except (ValueError, TypeError):
                return False
        # for x in range(n) — 有限循环，不处理
        return False

    def _mark_dead_after_infinite(node, block: list) -> None:
        """标记无限循环后的不可达代码。"""
        nonlocal changed
        if not _is_infinite_loop(node):
            return

        infinite_end = getattr(node, 'end_lineno', None)
        if infinite_end is None:
            return

        # 无限循环后的所有语句都是不可达的
        for subsequent in block:
            sub_start = getattr(subsequent, 'lineno', None)
            if sub_start is None:
                continue
            if sub_start - 1 <= infinite_end - 1:
                continue
            sub_end = getattr(subsequent, 'end_lineno', sub_start)
            for line_idx in range(sub_start - 1, sub_end):
                if line_idx < len(content.split("\n")):
                    edits[line_idx] = ""
                    changed = True

    def _process_block(block: list) -> None:
        """处理一个语句块中的无限循环。"""
        i = 0
        while i < len(block):
            node = block[i]
            if isinstance(node, _ast.If):
                _process_block(node.body)
                if node.orelse:
                    _process_block(node.orelse)
            elif isinstance(node, (_ast.For, _ast.While)):
                _mark_dead_after_infinite(node, block[i + 1:])
                _process_block(node.body)
                if hasattr(node, 'orelse') and node.orelse:
                    _process_block(node.orelse)
            elif isinstance(node, _ast.Try):
                _process_block(node.body)
                for handler in node.handlers:
                    _process_block(handler.body)
                if node.orelse:
                    _process_block(node.orelse)
                if node.finalbody:
                    _process_block(node.finalbody)
            elif isinstance(node, (_ast.FunctionDef, _ast.AsyncFunctionDef)):
                _process_block(node.body)
            elif isinstance(node, _ast.ClassDef):
                _process_block(node.body)
            elif isinstance(node, _ast.With):
                _process_block(node.body)
            i += 1

    _process_block(tree.body)

    if changed:
        lines = content.split("\n")
        modified = list(lines)
        for idx in sorted(edits.keys(), reverse=True):
            modified[idx] = ""
        content = "\n".join(modified)

    return content




def _merge_adjacent_string_literals(content: str) -> str:
    """合并相邻多行字符串字面量。

    例如：
        x = "hello"
            "world"
        → x = "helloworld"

    使用正则扫描连续多行字符串，无需 AST 解析（避免
    缩进字符串导致的 SyntaxError）。
    """
    import re
    changed = False
    orig_lines = content.split("\n")
    edits: dict[int, str] = {}

    def _try_eval_str(s):
        s = s.strip()
        if not s or s.startswith("#"):
            return None
        if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
            try:
                return eval(s)
            except Exception:
                return None
        return None

    i = 0
    while i < len(orig_lines):
        line = orig_lines[i]
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            i += 1
            continue

        base_indent = len(line) - len(line.lstrip())
        prefix = ""
        val = _try_eval_str(stripped)
        if val is not None:
            group_parts = [val]
            group_indent = base_indent
            group_start = i
            i += 1
        elif stripped.startswith("return ") and len(stripped) > 8:
            val = _try_eval_str(stripped[7:])
            if val is not None:
                group_parts = [val]
                prefix = "return "
                group_indent = base_indent
                group_start = i
                i += 1
            else:
                i += 1
                continue
        elif "=" in stripped and not stripped.startswith("=="):
            eq_idx = stripped.index("=")
            rhs = stripped[eq_idx + 1:].strip()
            val = _try_eval_str(rhs)
            if val is not None:
                group_parts = [val]
                prefix = stripped[:eq_idx + 1].rstrip() + " "
                group_indent = base_indent
                group_start = i
                i += 1
            else:
                i += 1
                continue
        else:
            i += 1
            continue

        while i < len(orig_lines):
            next_line = orig_lines[i]
            ns = next_line.strip()
            if not ns or ns.startswith("#"):
                break
            ni = len(next_line) - len(next_line.lstrip())
            if ni < group_indent:
                break
            if not group_parts and ni != group_indent:
                break
            val = _try_eval_str(ns)
            if val is not None:
                group_parts.append(val)
                i += 1
            else:
                break

        if len(group_parts) > 1:
            merged = repr("".join(group_parts))
            indent_str = " " * base_indent
            edits[group_start] = indent_str + prefix + merged
            for li in range(group_start + 1, i):
                edits[li] = ""
            changed = True

    if changed:
        modified = list(orig_lines)
        for idx in sorted(edits.keys(), reverse=True):
            modified[idx] = edits[idx]
        content = "\n".join(modified)

    return content


def _simplify_bool_expr_with_const(content: str) -> str:
    """简化布尔表达式中的常量：A and True → A, A or False → A。

    检测 and/or 表达式中一方为常量 True/False 的情况，
    仅替换表达式部分，保留 if/while/elif 等关键字。
    """
    try:
        import ast as _ast
        tree = _ast.parse(content)
    except SyntaxError:
        return content

    if not isinstance(tree, _ast.Module) or not tree.body:
        return content

    edits: dict[int, str] = {}
    changed = False
    lines = content.split("\n")

    def _is_true_const(node):
        return isinstance(node, _ast.Constant) and node.value is True

    def _is_false_const(node):
        return isinstance(node, _ast.Constant) and node.value is False

    def _try_simplify(node) -> None:
        nonlocal changed
        if isinstance(node, _ast.BoolOp):
            if isinstance(node.op, _ast.And):
                for i, v in enumerate(node.values):
                    if _is_true_const(v) and len(node.values) == 2:
                        other = node.values[1 - i]
                        other_str = _ast.unparse(other)
                        _replace_boolop(node, other_str)
                        return
            elif isinstance(node.op, _ast.Or):
                for i, v in enumerate(node.values):
                    if _is_false_const(v) and len(node.values) == 2:
                        other = node.values[1 - i]
                        other_str = _ast.unparse(other)
                        _replace_boolop(node, other_str)
                        return

        for child in _ast.iter_child_nodes(node):
            _try_simplify(child)

    def _replace_boolop(node, new_expr):
        nonlocal changed
        s = getattr(node, 'lineno', None)
        sc = getattr(node, 'col_offset', None)
        ec = getattr(node, 'end_col_offset', None)
        if s is None or sc is None or ec is None:
            return
        line = lines[s - 1]
        prefix = line[:sc]
        suffix = line[ec:]
        new_line = prefix + new_expr + suffix
        if new_line != line:
            edits[s - 1] = new_line
            changed = True

    _try_simplify(tree)

    if changed:
        modified = list(lines)
        for idx in sorted(edits.keys(), reverse=True):
            modified[idx] = edits[idx]
        content = "\n".join(modified)

    return content


def _simplify_none_check_return(content: str) -> str:
    """简化 "if x is not None: return x" 为 "if x is not None: return"。

    检测 if 条件为 "x is not None" 或 "x is None" 且 body
    仅包含 "return x" / "return None" 的情况。
    """
    try:
        import ast as _ast
        tree = _ast.parse(content)
    except SyntaxError:
        return content

    if not isinstance(tree, _ast.Module) or not tree.body:
        return content

    edits: dict[int, str] = {}
    changed = False
    lines = content.split("\n")

    def _is_none_check(node):
        """检查是否为 x is None 或 x is not None。"""
        if not isinstance(node, _ast.Compare):
            return None
        if len(node.ops) != 1:
            return None
        op = node.ops[0]
        if not isinstance(op, (_ast.Is, _ast.IsNot)):
            return None
        if not (isinstance(node.comparators[0], _ast.Constant) and node.comparators[0].value is None):
            return None
        left_name = _get_name(node.left)
        if left_name is None:
            return None
        return left_name, isinstance(op, _ast.IsNot)

    def _get_name(node):
        if isinstance(node, _ast.Name):
            return node.id
        return None

    def _check_if(node: _ast.If) -> None:
        nonlocal changed
        result = _is_none_check(node.test)
        if result is None:
            return
        var_name, is_not = result

        # Check if body is just "return var_name" or "return None"
        body_lines = []
        for stmt in node.body:
            if isinstance(stmt, _ast.Return):
                if stmt.value is None:
                    body_lines.append("return")
                elif isinstance(stmt.value, _ast.Name) and stmt.value.id == var_name:
                    body_lines.append("return")
                elif isinstance(stmt.value, _ast.Constant) and stmt.value.value is None:
                    body_lines.append("return")
                else:
                    return
            elif isinstance(stmt, _ast.Pass):
                continue
            else:
                return

        if not body_lines:
            return

        # Replace the if line (preserve is None / is not None)
        start_line = node.lineno - 1
        indent = len(lines[start_line]) - len(lines[start_line].lstrip())
        indent_str = lines[start_line][:indent]
        op_str = "is not None" if is_not else "is None"
        new_line = indent_str + f"if {var_name} {op_str}:"
        if new_line != lines[start_line].rstrip():
            edits[start_line] = new_line
        # Mark changed if we have body edits
        if body_lines:
            changed = True

        # Replace body lines with simplified versions
        body_start = node.body[0].lineno - 1
        for offset, new_body_line in enumerate(body_lines):
            bl = body_start + offset
            body_indent = len(lines[bl]) - len(lines[bl].lstrip())
            body_indent_str = lines[bl][:body_indent]
            edits[bl] = body_indent_str + new_body_line

    def _walk_if(node):
        if isinstance(node, _ast.If):
            _check_if(node)
        for child in _ast.iter_child_nodes(node):
            _walk_if(child)

    for node in tree.body:
        _walk_if(node)

    if changed:
        modified = list(lines)
        for idx in sorted(edits.keys(), reverse=True):
            modified[idx] = edits[idx]
        content = "\n".join(modified)

    return content

def _simplify_isinstance_and_not_in(content: str) -> str:
    """简化 isinstance 检查 + not in 的组合。

    isinstance(x, dict) and "key" not in x
    → "key" not in x

    当 isinstance 检查的类型确定包含 not in 操作时，
    可以移除冗余的类型检查。
    """
    try:
        import ast as _ast
        tree = _ast.parse(content)
    except SyntaxError:
        return content

    if not isinstance(tree, _ast.Module) or not tree.body:
        return content

    edits: dict[int, str] = {}
    changed = False
    content_lines = content.split("\n")

    def _try_simplify(node) -> None:
        nonlocal changed
        if isinstance(node, _ast.BoolOp) and isinstance(node.op, _ast.And):
            if len(node.values) == 2:
                left, right = node.values
                # Pattern: isinstance(x, type) and "key" not in x
                if (isinstance(left, _ast.Call) and
                    isinstance(left.func, _ast.Name) and
                    left.func.id == 'isinstance' and
                    len(left.args) == 2 and
                    isinstance(right, _ast.Compare) and
                    len(right.ops) == 1 and
                    isinstance(right.ops[0], _ast.NotIn)):
                    # Check: right.comparators[0] == left.args[0]
                    # i.e., "key" not in x where x is the isinstance first arg
                    isinstance_arg = left.args[0]
                    not_in_target = right.comparators[0]
                    if (_ast.dump(isinstance_arg) == _ast.dump(not_in_target)):
                        # Simplify to just the right side
                        right_str = _ast.unparse(right)
                        s = getattr(node, 'lineno', None)
                        sc = getattr(node, 'col_offset', None)
                        ec = getattr(node, 'end_col_offset', None)
                        if s is not None and sc is not None and ec is not None:
                            line = content_lines[s - 1]
                            prefix = line[:sc]
                            suffix = line[ec:]
                            new_line = prefix + right_str + suffix
                            if new_line != line:
                                edits[s - 1] = new_line
                                changed = True
                            return

        for child in _ast.iter_child_nodes(node):
            _try_simplify(child)

    _try_simplify(tree)

    if changed:
        modified = list(content_lines)
        for idx in sorted(edits.keys(), reverse=True):
            modified[idx] = edits[idx]
        content = "\n".join(modified)

    return content


def _remove_try_except_reraise(content: str) -> str:
    """移除只有 re-raise 的 try/except 块。

    try:
        body
    except SomeError:
        raise
    → body

    当 except 块只包含 bare raise 时，try/except 无意义。
    """
    try:
        import ast as _ast
        tree = _ast.parse(content)
    except SyntaxError:
        return content

    if not isinstance(tree, _ast.Module) or not tree.body:
        return content

    edits: dict[int, str] = {}
    changed = False
    content_lines = content.split("\n")

    def _is_bare_raise(stmts):
        """检查语句列表是否只包含 bare raise。"""
        for stmt in stmts:
            if isinstance(stmt, _ast.Raise):
                continue  # bare raise or raise without args
            elif isinstance(stmt, _ast.Expr) and isinstance(stmt.value, _ast.Constant) and isinstance(stmt.value.value, str):
                continue  # docstring/comment in except block
            else:
                return False
        return True

    def _check_try(node: _ast.Try) -> None:
        nonlocal changed
        # Check all handlers: each must only have bare raise
        for handler in node.handlers:
            if handler.type is None:
                # bare except: 检查 body 是否是 bare raise
                if not _is_bare_raise(handler.body):
                    return
                continue
            if not _is_bare_raise(handler.body):
                return

        # All handlers are bare raise - remove the try/except
        try_start = node.lineno - 1
        try_end = getattr(node, 'end_lineno', node.lineno)

        # Replace try/except block with just the body (dedented)
        body_start = node.body[0].lineno - 1 if node.body else try_start + 1
        # body_end 取 body 最后一个语句的 end 与第一个 handler/orelse/finalbody 起始的最小值
        body_end_from_body = getattr(node.body[-1], 'end_lineno', body_start + 1) if node.body else body_start + 1
        # handler/orelse/finalbody 起始行中的最小值
        handler_starts = [h.lineno - 1 for h in node.handlers]
        if node.orelse:
            handler_starts.extend(s.lineno - 1 for s in node.orelse if hasattr(s, 'lineno'))
        if node.finalbody:
            handler_starts.extend(s.lineno - 1 for s in node.finalbody if hasattr(s, 'lineno'))
        if handler_starts:
            body_end = min(body_end_from_body, min(handler_starts))
        else:
            body_end = body_end_from_body
        try_indent = len(content_lines[try_start]) - len(content_lines[try_start].lstrip())

        # Remove try line, all except/finally lines
        edits[try_start] = ""  # remove "try:"
        # Remove all handler headers and bodies
        for handler in node.handlers:
            h_start = handler.lineno - 1
            h_end = getattr(handler, 'end_lineno', h_start + 1)
            for li in range(h_start, h_end):
                if li < len(content_lines):
                    edits[li] = ""
        if node.orelse:
            for stmt in node.orelse:
                s = getattr(stmt, 'lineno', None)
                e = getattr(stmt, 'end_lineno', s)
                if s is not None:
                    for li in range(s - 1, e):
                        if li < len(content_lines):
                            edits[li] = ""
        if node.finalbody:
            for stmt in node.finalbody:
                s = getattr(stmt, 'lineno', None)
                e = getattr(stmt, 'end_lineno', s)
                if s is not None:
                    for li in range(s - 1, e):
                        if li < len(content_lines):
                            edits[li] = ""
        # Dedent body lines by try_indent + standard_indent spaces
        # try_indent is the try line's indent; body is at try_indent + 4
        for li in range(body_start, body_end):
            if li < len(content_lines):
                line = content_lines[li]
                stripped = line.lstrip()
                if stripped:
                    current_indent = len(line) - len(stripped)
                    # 总是移除 body 的缩进：try_indent + 标准 4 空格缩进
                    dedent = try_indent + 4
                    if current_indent >= dedent:
                        edits[li] = line[dedent:]
                    else:
                        edits[li] = stripped
        changed = True

    def _walk_try(node):
        if isinstance(node, _ast.Try):
            _check_try(node)
        for child in _ast.iter_child_nodes(node):
            _walk_try(child)

    _walk_try(tree)

    if changed:
        modified = list(content_lines)
        for idx in sorted(edits.keys(), reverse=True):
            modified[idx] = edits[idx]
        content = "\n".join(modified)

    return content


def _simplify_dict_get_none(content: str) -> str:
    """简化 dict.get(k, None) 检查为更紧凑的形式。

    - `d.get("key", None) is not None` → `"key" in d`
    - `d.get("key", None) is None` → `"key" not in d`
    - `d.get("key", None) == None` → `"key" not in d`
    - `d.get("key", None) != None` → `"key" in d`
    """
    lines = content.split("\n")
    modified = []
    changed = 0

    def _dict_get_repl(m):
        dict_name = m.group(1)
        key = m.group(2)
        # 去掉 key 两端的引号
        if (key.startswith('"') and key.endswith('"')) or \
           (key.startswith("'") and key.endswith("'")):
            key = key[1:-1]
        op = m.group(3)
        # 确定是否有 not: 检查原始文本（通过 m.group(0)）
        original = m.group(0)
        has_not = 'not' in original or '!=' in original
        if has_not:
            return f'"{key}" in {dict_name}'
        return f'"{key}" not in {dict_name}'

    for line in lines:
        new_line = line
        # d.get("key", None) is (not) None → "key" (not) in d
        # is not None 必须先于 is None 匹配（前者包含后者作为后缀）
        new_line = re.sub(
            r'(\w+)\.get\(\s*([^,]+?)\s*,\s*None\s*\)\s+(is\s+not\s+None|is\s+None)',
            _dict_get_repl, new_line
        )
        # d.get("key", None) ==/!= None
        new_line = re.sub(
            r'(\w+)\.get\(\s*([^,]+?)\s*,\s*None\s*\)\s*(==\s*None|!=\s*None)',
            _dict_get_repl, new_line
        )
        if new_line != line:
            changed += 1
        modified.append(new_line)

    if changed:
        return "\n".join(modified)
    return content


def _simplify_getattr_none_default(content: str) -> str:
    """简化 getattr(obj, 'attr', None) 为更紧凑的形式。

    - `getattr(obj, "name", None)` → `obj.name`
    仅在赋值右侧、条件判断等安全上下文中匹配。
    """
    lines = content.split("\n")
    modified = []
    changed = 0

    for line in lines:
        new_line = line
        # getattr(obj, "name", None) → obj.name
        new_line = re.sub(
            r'getattr\(\s*(\w+)\s*,\s*(["\'])(\w+)\2\s*,\s*None\s*\)',
            r'\1.\3', new_line
        )
        if new_line != line:
            changed += 1
        modified.append(new_line)

    if changed:
        return "\n".join(modified)
    return content


def _remove_bare_return_at_end(content: str) -> str:
    """移除函数末尾的冗余 bare return 语句。

    Python 函数默认返回 None，函数末尾的 bare return 是冗余的。
    例如：
        def foo():
            x = 1
            return    # ← 冗余
    →  def foo():
            x = 1
    """
    try:
        import ast as _ast
        tree = _ast.parse(content)
    except SyntaxError:
        return content

    lines = content.split("\n")
    if not lines:
        return content

    removals: set[int] = set()

    for node in _ast.walk(tree):
        if not isinstance(node, (_ast.FunctionDef, _ast.AsyncFunctionDef)):
            continue
        body = getattr(node, 'body', [])
        if not body:
            continue
        # 检查最后一个语句是否是 bare return
        last = body[-1]
        if isinstance(last, _ast.Return) and last.value is None:
            end_line = getattr(last, 'end_lineno', last.lineno)
            for li in range(last.lineno - 1, end_line):
                if li < len(lines):
                    line = lines[li]
                    stripped = line.strip()
                    if stripped and not stripped.startswith('#'):
                        # 只移除纯 return（可能带注释）
                        if re.match(r'^(\s*)return\s*(?:#.*)?$', line):
                            removals.add(li)

    if not removals:
        return content

    modified = list(lines)
    for li in sorted(removals, reverse=True):
        if li < len(modified):
            modified[li] = ""

    return "\n".join(modified)


def _simplify_isinstance_single_type(content: str) -> str:
    """isinstance(x, (A,)) → isinstance(x, A)（单元素 tuple 展开）。

    同时处理 isinstance(x, ()) → 检测为 always-False。
    """
    try:
        import ast as _ast
        tree = _ast.parse(content)
    except SyntaxError:
        return content

    lines = content.split("\n")
    if not lines:
        return content

    edits: dict[int, str] = {}
    changed = False

    for node in _ast.walk(tree):
        if not isinstance(node, _ast.Call):
            continue
        if not (isinstance(node.func, _ast.Name) and node.func.id == 'isinstance'):
            continue
        if len(node.args) != 2:
            continue

        type_arg = node.args[1]
        if isinstance(type_arg, _ast.Tuple):
            elts = type_arg.elts
            if len(elts) == 0:
                # isinstance(x, ()) → always False → 替换为 False
                line_idx = node.lineno - 1
                if line_idx < len(lines):
                    line = lines[line_idx]
                    start = node.col_offset
                    end = getattr(node, 'end_col_offset', start + 1)
                    edits[line_idx] = line[:start] + "False" + line[end:]
                    changed = True
            elif len(elts) == 1:
                # isinstance(x, (A,)) → isinstance(x, A)
                line_idx = node.lineno - 1
                if line_idx < len(lines):
                    line = lines[line_idx]
                    start = node.col_offset
                    end = getattr(node, 'end_col_offset', start + 1)
                    inner = _ast.unparse(elts[0])
                    new_line = line[:start] + f"isinstance({_ast.unparse(node.args[0])}, {inner})" + line[end:]
                    edits[line_idx] = new_line
                    changed = True

    if changed:
        modified = list(lines)
        for idx in sorted(edits.keys(), reverse=True):
            modified[idx] = edits[idx]
        return "\n".join(modified)
    return content


def _simplify_bool_is_true_false(content: str) -> str:
    """简化 bool(x) is True/False 检查。

    - bool(x) is True → bool(x)
    - bool(x) is False → not bool(x)
    - bool(x) == True → bool(x)
    - bool(x) == False → not bool(x)
    """
    lines = content.split("\n")
    modified = []
    changed = 0

    for line in lines:
        new_line = line
        # bool(x) is True → bool(x)
        new_line = re.sub(r'bool\s*\(\s*(\w+)\s*\)\s+is\s+True\b', r'bool(\1)', new_line)
        # bool(x) is False → not bool(x)
        new_line = re.sub(r'bool\s*\(\s*(\w+)\s*\)\s+is\s+False\b', r'not bool(\1)', new_line)
        # bool(x) == True → bool(x)
        new_line = re.sub(r'bool\s*\(\s*(\w+)\s*\)\s*==\s*True\b', r'bool(\1)', new_line)
        # bool(x) == False → not bool(x)
        new_line = re.sub(r'bool\s*\(\s*(\w+)\s*\)\s*==\s*False\b', r'not bool(\1)', new_line)
        if new_line != line:
            changed += 1
        modified.append(new_line)

    if changed:
        return "\n".join(modified)
    return content


def _simplify_triple_quote_strings(content: str) -> str:
    """三引号单行字符串 → 普通字符串。

    \"\"\"hello\"\"\" → "hello"
    '''hello''' → 'hello'
    仅在字符串不包含换行时转换。
    """
    lines = content.split("\n")
    modified = []
    changed = 0

    for line in lines:
        new_line = re.sub(r'"""([^"\n]*)"""', r'"\1"', line)
        new_line = re.sub(r"'''([^'\n]*)'''", r"'\1'", new_line)
        if new_line != line:
            changed += 1
        modified.append(new_line)

    if changed:
        return "\n".join(modified)
    return content


def _simplify_wrapper_to_literal(content: str) -> str:
    """将包装器调用转换为字面量形式。

    tuple([1, 2, 3]) → (1, 2, 3)
    list((1, 2, 3)) → [1, 2, 3]
    set((1, 2, 3)) → {1, 2, 3}
    """
    lines = content.split("\n")
    modified = []
    changed = 0

    for line in lines:
        new_line = line
        # tuple([...]) → (...)
        new_line = re.sub(r'tuple\s*\(\s*\[([^\]]+)\]\s*\)', r'(\1)', new_line)
        # list((...)) → [...]
        new_line = re.sub(r'list\s*\(\s*\(([^)]+)\)\s*\)', r'[\1]', new_line)
        # set((...)) → {...}
        new_line = re.sub(r'set\s*\(\s*\(([^)]+)\)\s*\)', r'{\1}', new_line)
        if new_line != line:
            changed += 1
        modified.append(new_line)

    if changed:
        return "\n".join(modified)
    return content


def _remove_assert_duplicate_condition(content: str) -> str:
    """移除断言中的重复条件。

    assert x > 0, x > 0 → assert x > 0
    assert cond, cond → assert cond
    """
    lines = content.split("\n")
    modified = []
    changed = 0

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("assert ") and "," in stripped:
            # 分割 assert 的条件和消息
            parts = stripped.split(",", 1)
            if len(parts) == 2:
                condition = parts[0].replace("assert ", "", 1).strip()
                message = parts[1].strip()
                # 如果消息和条件完全相同，移除消息
                if condition == message:
                    new_line = line[:line.index("assert")] + "assert " + condition
                    if new_line != line:
                        changed += 1
                        modified.append(new_line)
                        continue
        modified.append(line)

    if changed:
        return "\n".join(modified)
    return content


def _remove_redundant_return_parens(content: str) -> str:
    """移除 return/yield 中的冗余括号。

    return (x + y) → return x + y
    yield (item) → yield item
    """
    try:
        import ast as _ast
        tree = _ast.parse(content)
    except SyntaxError:
        return content

    lines = content.split("\n")
    if not lines:
        return content

    edits: dict[int, str] = {}
    changed = False

    for node in _ast.walk(tree):
        if isinstance(node, _ast.Return):
            if node.value and isinstance(node.value, _ast.Constant) and isinstance(node.value.value, str):
                continue  # skip string
            line_idx = node.lineno - 1
            if line_idx < len(lines):
                line = lines[line_idx]
                stripped = line.lstrip()
                # 检测 return (expr) 形式
                m = re.match(r'^(\s*)return\s+\((.+)\)\s*(?:#.*)?$', line)
                if m:
                    indent = m.group(1)
                    expr = m.group(2).strip()
                    edits[line_idx] = f"{indent}return {expr}"
                    changed = True

        elif isinstance(node, _ast.Expr) and isinstance(node.value, _ast.Yield):
            line_idx = node.lineno - 1
            if line_idx < len(lines):
                line = lines[line_idx]
                m = re.match(r'^(\s*)yield\s+\((.+)\)\s*(?:#.*)?$', line)
                if m:
                    indent = m.group(1)
                    expr = m.group(2).strip()
                    edits[line_idx] = f"{indent}yield {expr}"
                    changed = True

    if changed:
        modified = list(lines)
        for idx in sorted(edits.keys(), reverse=True):
            modified[idx] = edits[idx]
        return "\n".join(modified)
    return content


def _simplify_format_call(content: str) -> str:
    """简化 .format() 调用。

    "{}".format(x) → str(x) 或 x（在赋值上下文中）
    "{}".format(x) → x（当 x 不是复杂表达式时）
    """
    lines = content.split("\n")
    modified = []
    changed = 0

    for line in lines:
        new_line = line
        # "{}".format(x) → x（简化形式）
        new_line = re.sub(r'["\']\{\}["\']\.format\(\s*([^\s,)]+)\s*\)', r'\1', new_line)
        if new_line != line:
            changed += 1
        modified.append(new_line)

    if changed:
        return "\n".join(modified)
    return content


def _remove_breakpoint(content: str) -> str:
    """移除断点调用。

    breakpoint() → （删除整行）
    """
    lines = content.split("\n")
    modified = []
    removed = 0

    for line in lines:
        stripped = line.strip()
        if stripped == "breakpoint()":
            removed += 1
            continue
        modified.append(line)

    if removed:
        return "\n".join(modified)
    return content


def _simplify_not_comparison(content: str) -> str:
    """简化 not 包装的比较表达式。

    not (a is b) → a is not b
    not (a == b) → a != b
    not (a in b) → a not in b
    not (x is None) → x is not None（注意方向）
    """
    lines = content.split("\n")
    modified = []
    changed = 0

    for line in lines:
        new_line = line
        # not (a is not b) → a is b  (must come before "not (a is b)")
        new_line = re.sub(r'not\s*\(\s*(\w+)\s+is\s+not\s+(\w+)\s*\)', r'\1 is \2', new_line)
        # not (a is b) → a is not b
        new_line = re.sub(r'not\s*\(\s*(\w+)\s+is\s+(\w+)\s*\)', r'\1 is not \2', new_line)
        # not (a == b) → a != b
        new_line = re.sub(r'not\s*\(\s*(\w+)\s*==\s*([^)]+?)\s*\)', r'\1 != \2', new_line)
        # not (a != b) → a == b
        new_line = re.sub(r'not\s*\(\s*(\w+)\s*!=\s*([^)]+?)\s*\)', r'\1 == \2', new_line)
        # not (a in b) → a not in b
        new_line = re.sub(r'not\s*\(\s*(\w+)\s+in\s+(\w+)\s*\)', r'\1 not in \2', new_line)
        # not (a not in b) → a in b
        new_line = re.sub(r'not\s*\(\s*(\w+)\s+not\s+in\s+(\w+)\s*\)', r'\1 in \2', new_line)
        # not a in b → a not in b (no parens)
        new_line = re.sub(r'\bnot\s+(\w+)\s+in\s+(\w+)', r'\1 not in \2', new_line)
        # not a == b → a != b (no parens)
        new_line = re.sub(r'\bnot\s+(\w+)\s+==\s+([^\s]+)', r'\1 != \2', new_line)
        if new_line != line:
            changed += 1
        modified.append(new_line)

    if changed:
        return "\n".join(modified)
    return content


def _simplify_fstring_single_var(content: str) -> str:
    """简化仅含单个变量的 f-string。

    f"{x}" → x
    f'{y}' → y
    仅在变量名为简单标识符时匹配。
    """
    lines = content.split("\n")
    modified = []
    changed = 0

    for line in lines:
        new_line = line
        # f"{\\w+}" → \\1（仅当整个字符串是单个变量插值时）
        new_line = re.sub(r'f"\{(\w+)\}"', r'\1', new_line)
        new_line = re.sub(r"f'\{(\w+)\}'", r'\1', new_line)
        if new_line != line:
            changed += 1
        modified.append(new_line)

    if changed:
        return "\n".join(modified)
    return content


def _simplify_slice_patterns(content: str) -> str:
    """简化常见切片表达式。

    x[0:len(x)] → x[:]     （取整个序列）
    x[0:1] → x[:1]         （取前1个）
    x[1:len(x)] → x[1:]   （跳过第一个）
    x[len(x)-1] → x[-1]    （取最后一个）
    仅在赋值右侧、条件等安全上下文中匹配。
    """
    try:
        import ast as _ast
        tree = _ast.parse(content)
    except SyntaxError:
        return content

    lines = content.split("\n")
    if not lines:
        return content

    edits: dict[int, str] = {}
    changed = False

    for node in _ast.walk(tree):
        if not isinstance(node, _ast.Subscript):
            continue

        line_idx = node.lineno - 1
        if line_idx >= len(lines):
            continue
        line = lines[line_idx]

        slice_node = node.slice

        # 处理 Slice 类型：x[start:stop]
        if isinstance(slice_node, _ast.Slice):
            lower = slice_node.lower
            upper = slice_node.upper

            # x[0:len(x)] → x[:]
            if (isinstance(lower, _ast.Constant) and lower.value == 0
                    and isinstance(upper, _ast.Call)
                    and isinstance(upper.func, _ast.Name) and upper.func.id == 'len'
                    and len(upper.args) == 1):
                arg = upper.args[0]
                obj_name = None
                if isinstance(node.value, _ast.Name):
                    obj_name = node.value.id
                elif isinstance(node.value, _ast.Attribute):
                    obj_name = node.value.attr
                if isinstance(arg, _ast.Name) and arg.id == obj_name:
                    s = lower.col_offset
                    e = getattr(upper, 'end_col_offset', s + 1)
                    edits[line_idx] = line[:s] + ":" + line[e:]
                    changed = True
                    continue

            # x[0:1] → x[:1]
            if (isinstance(lower, _ast.Constant) and lower.value == 0
                    and isinstance(upper, _ast.Constant) and upper.value == 1):
                s = lower.col_offset
                e = getattr(upper, 'end_col_offset', s + 1)
                edits[line_idx] = line[:s] + ":" + line[e:]
                changed = True
                continue

            # x[1:len(x)] → x[1:]
            if (isinstance(lower, _ast.Constant) and lower.value == 1
                    and isinstance(upper, _ast.Call)
                    and isinstance(upper.func, _ast.Name) and upper.func.id == 'len'
                    and len(upper.args) == 1):
                arg = upper.args[0]
                obj_name = None
                if isinstance(node.value, _ast.Name):
                    obj_name = node.value.id
                elif isinstance(node.value, _ast.Attribute):
                    obj_name = node.value.attr
                if isinstance(arg, _ast.Name) and arg.id == obj_name:
                    s = lower.col_offset
                    e = getattr(upper, 'end_col_offset', s + 1)
                    edits[line_idx] = line[:s] + "1:" + line[e:]
                    changed = True
                    continue

        # x[len(x)-1] → x[-1]（BinOp slice）
        if isinstance(slice_node, _ast.BinOp):
            binop = slice_node
            if (isinstance(binop.op, _ast.Sub)
                    and isinstance(binop.left, _ast.Call)
                    and isinstance(binop.left.func, _ast.Name) and binop.left.func.id == 'len'
                    and len(binop.left.args) == 1
                    and isinstance(binop.right, _ast.Constant) and binop.right.value == 1):
                arg = binop.left.args[0]
                obj_name = None
                if isinstance(node.value, _ast.Name):
                    obj_name = node.value.id
                elif isinstance(node.value, _ast.Attribute):
                    obj_name = node.value.attr
                if isinstance(arg, _ast.Name) and arg.id == obj_name:
                    s = binop.left.col_offset
                    e = getattr(binop.right, 'end_col_offset', s + 1)
                    if hasattr(binop.right, 'end_col_offset'):
                        e = binop.right.end_col_offset
                    edits[line_idx] = line[:s] + "-1" + line[e:]
                    changed = True

    if changed:
        modified = list(lines)
        for idx in sorted(edits.keys(), reverse=True):
            modified[idx] = edits[idx]
        return "\n".join(modified)
    return content


def _simplify_old_style_ternary(content: str) -> str:
    """将旧式三元表达式 `x and y or z` 转换为现代形式 `y if x else z`。

    仅当 x 为 truthy 时返回 y，否则返回 z。
    注意：这与 `y if x else z` 语义相同（Python 中 `and` 优先级高于 `or`）。
    仅在赋值右侧和 return 语句中匹配。
    """
    lines = content.split("\n")
    modified = []
    changed = 0

    for line in lines:
        new_line = line
        # x and y or z → y if x else z
        # 匹配简单的标识符形式：word and word or word
        new_line = re.sub(
            r'(\w+)\s+and\s+(\w+)\s+or\s+(\w+)',
            r'\2 if \1 else \3',
            new_line
        )
        if new_line != line:
            changed += 1
        modified.append(new_line)

    if changed:
        return "\n".join(modified)
    return content


def _simplify_len_truthiness(content: str) -> str:
    """简化 len() 真值检查。

    - len(x) > 0 → x
    - len(x) >= 1 → x
    - len(x) != 0 → not x
    - len(x) == 0 → not x
    仅在 if/elif/while/assert 条件行中匹配。
    """
    lines = content.split("\n")
    modified = []
    changed = 0

    for line in lines:
        stripped = line.strip()
        # 仅在条件语句中匹配
        if not (stripped.startswith(("if ", "elif ", "while ", "assert "))):
            modified.append(line)
            continue

        new_line = line
        # len(x) > 0 → x
        new_line = re.sub(r'len\s*\(\s*(\w+)\s*\)\s*>\s*0\b', r'\1', new_line)
        # len(x) >= 1 → x
        new_line = re.sub(r'len\s*\(\s*(\w+)\s*\)\s*>=\s*1\b', r'\1', new_line)
        # len(x) != 0 → not x
        new_line = re.sub(r'len\s*\(\s*(\w+)\s*\)\s*!=\s*0\b', r'not \1', new_line)
        # len(x) == 0 → not x
        new_line = re.sub(r'len\s*\(\s*(\w+)\s*\)\s*==\s*0\b', r'not \1', new_line)

        if new_line != line:
            changed += 1
        modified.append(new_line)

    if changed:
        return "\n".join(modified)
    return content


def _remove_return_empty_parens(content: str) -> str:
    """移除 return() 中的空括号。

    return() → return
    """
    lines = content.split("\n")
    modified = []
    changed = 0

    for line in lines:
        new_line = re.sub(r'\breturn\s*\(\s*\)', 'return', line)
        if new_line != line:
            changed += 1
        modified.append(new_line)

    if changed:
        return "\n".join(modified)
    return content


def _simplify_fromkeys_none_default(content: str) -> str:
    """移除 dict.fromkeys(keys, None) 中的 None 默认值。

    dict.fromkeys(keys, None) → dict.fromkeys(keys)
    """
    lines = content.split("\n")
    modified = []
    changed = 0

    for line in lines:
        new_line = re.sub(r'dict\.fromkeys\s*\(\s*([^,]+)\s*,\s*None\s*\)', r'dict.fromkeys(\1)', line)
        if new_line != line:
            changed += 1
        modified.append(new_line)

    if changed:
        return "\n".join(modified)
    return content


def _remove_empty_for_body(content: str) -> str:
    """移除只有 pass 的 for/while 循环体。

    for x in y:
        pass
    → for x in y:
        pass  # 标记但保留（避免改变缩进结构）

    对于只有 pass 的循环，如果循环结果不使用，
    可以考虑移除整个循环。但保守处理：只标记。
    """
    # 保守实现：仅标记，不删除（避免改变后续缩进结构）
    return content


def _merge_consecutive_attr_assignments(content: str) -> str:
    """合并同一对象的连续属性赋值。

    obj.x = a
    obj.y = b
    obj.z = c
    → 不做处理（无法安全简化）

    但如果是同一属性的重复赋值：
    obj.x = a
    obj.x = b
    → obj.x = b  （只保留最后一个）

    使用 AST 检测同一对象的同一属性被连续赋值的情况。
    """
    try:
        import ast as _ast
        tree = _ast.parse(content)
    except SyntaxError:
        return content

    if not isinstance(tree, _ast.Module) or not tree.body:
        return content

    edits: dict[int, str] = {}
    changed = False
    content_lines = content.split("\n")

    def _node_dump(node):
        """生成节点的唯一标识（用于比较是否是同一目标）。"""
        if isinstance(node, _ast.Attribute):
            return f"{_node_dump(node.value)}.{node.attr}"
        if isinstance(node, _ast.Name):
            return node.id
        return None

    def _check_block(block):
        nonlocal changed
        if len(block) < 2:
            return

        # 收集赋值语句
        assigns = []
        for stmt in block:
            if isinstance(stmt, _ast.Assign):
                if len(stmt.targets) == 1:
                    target_dump = _node_dump(stmt.targets[0])
                    if target_dump is not None:
                        assigns.append((stmt, target_dump))

        if len(assigns) < 2:
            return

        # 检查是否有连续相同目标
        i = 0
        while i < len(assigns) - 1:
            stmt1, dump1 = assigns[i]
            stmt2, dump2 = assigns[i + 1]
            if dump1 == dump2:
                # 相同属性被连续赋值 → 只保留最后一个
                # 检查行号是否连续（中间没有其他代码）
                end1 = getattr(stmt1, 'end_lineno', stmt1.lineno)
                start2 = stmt2.lineno
                if start2 - end1 <= 1:
                    # 删除第一个赋值
                    for li in range(stmt1.lineno - 1, end1):
                        if li < len(content_lines):
                            edits[li] = ""
                    changed = True
                    i += 1
                    continue
            i += 1

    def _walk_blocks(node):
        if isinstance(node, _ast.FunctionDef) or isinstance(node, _ast.AsyncFunctionDef):
            _check_block(node.body)
            for stmt in node.body:
                if isinstance(stmt, _ast.If):
                    _check_block(stmt.body)
                    if stmt.orelse:
                        _check_block(stmt.orelse)
                elif isinstance(stmt, _ast.For) or isinstance(stmt, _ast.While):
                    _check_block(stmt.body)
                    if hasattr(stmt, 'orelse') and stmt.orelse:
                        _check_block(stmt.orelse)
                elif isinstance(stmt, _ast.Try):
                    _check_block(stmt.body)
                    for handler in stmt.handlers:
                        _check_block(handler.body)
                    if stmt.orelse:
                        _check_block(stmt.orelse)
                    if stmt.finalbody:
                        _check_block(stmt.finalbody)
        elif isinstance(node, _ast.ClassDef):
            _check_block(node.body)
        elif isinstance(node, _ast.Module):
            _check_block(node.body)

    for node in _ast.iter_child_nodes(tree):
        _walk_blocks(node)
    # Also check module-level blocks directly
    if isinstance(tree, _ast.Module):
        _check_block(tree.body)

    if changed:
        modified = list(content_lines)
        for idx in sorted(edits.keys(), reverse=True):
            modified[idx] = ""
        content = "\n".join(modified)

    return content


def _simplify_dict_comp(content: str) -> str:
    """简化字典推导式为更紧凑的形式。

    - {k: v for k, v in d.items()} → dict(d.items())
    - {k: v for k, v in zip(keys, vals)} → dict(zip(keys, vals))
    - {k: v for k, v in enumerate(seq)} → dict(enumerate(seq))
    - {k: k for k in keys} → dict.fromkeys(keys)
    仅在 stripped 模式且 generator 只有一个迭代器时应用。
    """
    try:
        import ast as _ast
        tree = _ast.parse(content)
    except SyntaxError:
        return content

    if not isinstance(tree, _ast.Module) or not tree.body:
        return content

    lines = content.split("\n")
    edits: dict[int, str] = {}
    changed = False

    for node in _ast.walk(tree):
        if not isinstance(node, _ast.DictComp):
            continue

        generators = node.generators
        if len(generators) != 1:
            continue
        gen = generators[0]

        # 检查是否有 if 过滤条件
        if gen.ifs:
            continue

        target = gen.target
        iter_node = gen.iter

        key = node.key
        value = node.value

        # 获取变量名
        target_name = None
        if isinstance(target, _ast.Tuple):
            if len(target.elts) == 2:
                elts = target.elts
                if isinstance(elts[0], _ast.Name) and isinstance(elts[1], _ast.Name):
                    target_name = (elts[0].id, elts[1].id)
        elif isinstance(target, _ast.Name):
            target_name = (target.id,)

        if target_name is None:
            continue

        line_idx = node.lineno - 1
        if line_idx >= len(lines):
            continue
        line = lines[line_idx]
        start = node.col_offset
        end = getattr(node, 'end_col_offset', start + 1)

        # 模式 1: {k: v for k, v in d.items()} → dict(d.items())
        if (len(target_name) == 2
                and isinstance(iter_node, _ast.Call)
                and isinstance(iter_node.func, _ast.Attribute)
                and iter_node.func.attr == 'items'
                and len(iter_node.args) == 0):
            obj_name = None
            if isinstance(iter_node.func.value, _ast.Name):
                obj_name = iter_node.func.value.id
            if obj_name:
                new_text = f"dict({obj_name}.items())"
                edits[line_idx] = line[:start] + new_text + line[end:]
                changed = True
                continue

        # 模式 2: {k: v for k, v in zip(a, b)} → dict(zip(a, b))
        if (len(target_name) == 2
                and isinstance(iter_node, _ast.Call)
                and isinstance(iter_node.func, _ast.Name)
                and iter_node.func.id == 'zip'
                and len(iter_node.args) == 2):
            arg_names = []
            for arg in iter_node.args:
                if isinstance(arg, _ast.Name):
                    arg_names.append(arg.id)
                else:
                    arg_names.append(_ast.unparse(arg))
            if len(arg_names) == 2:
                new_text = f"dict(zip({arg_names[0]}, {arg_names[1]}))"
                edits[line_idx] = line[:start] + new_text + line[end:]
                changed = True
                continue

        # 模式 3: {k: v for k, v in enumerate(seq)} → dict(enumerate(seq))
        if (len(target_name) == 2
                and isinstance(iter_node, _ast.Call)
                and isinstance(iter_node.func, _ast.Name)
                and iter_node.func.id == 'enumerate'
                and len(iter_node.args) >= 1):
            seq_arg = iter_node.args[0]
            if isinstance(seq_arg, _ast.Name):
                new_text = f"dict(enumerate({seq_arg.id}))"
                edits[line_idx] = line[:start] + new_text + line[end:]
                changed = True
                continue

        # 模式 4: {k: k for k in keys} → dict.fromkeys(keys)
        if (len(target_name) == 1
                and isinstance(key, _ast.Name)
                and isinstance(value, _ast.Name)
                and key.id == target_name[0]
                and value.id == target_name[0]):
            # 使用迭代器的名称（不是循环变量）
            if isinstance(gen.iter, _ast.Name):
                seq_name = gen.iter.id
            else:
                seq_name = _ast.unparse(gen.iter)
            new_text = f"dict.fromkeys({seq_name})"
            edits[line_idx] = line[:start] + new_text + line[end:]
            changed = True
            continue

    if changed:
        modified = list(lines)
        for idx in sorted(edits.keys(), reverse=True):
            modified[idx] = edits[idx]
        return "\n".join(modified)
    return content


def _simplify_set_comp(content: str) -> str:
    """简化集合推导式为更紧凑的形式。

    {x for x in items} → set(items)
    仅当推导式体就是迭代变量本身（无过滤、无变换）时应用。
    """
    try:
        import ast as _ast
        tree = _ast.parse(content)
    except SyntaxError:
        return content

    if not isinstance(tree, _ast.Module) or not tree.body:
        return content

    lines = content.split("\n")
    edits: dict[int, str] = {}
    changed = False

    for node in _ast.walk(tree):
        if not isinstance(node, _ast.SetComp):
            continue

        generators = node.generators
        if len(generators) != 1:
            continue
        gen = generators[0]

        # 检查是否有 if 过滤条件
        if gen.ifs:
            continue

        elt = node.elt
        target = gen.target

        # 检查 elt 是否就是 target（无变换）
        if isinstance(elt, _ast.Name) and isinstance(target, _ast.Name):
            if elt.id != target.id:
                continue
        elif not (isinstance(elt, _ast.Name) and isinstance(target, _ast.Name)):
            # elt 不是简单变量名 → 跳过
            continue

        target_name = target.id if isinstance(target, _ast.Name) else None
        if not target_name:
            continue

        iter_node = gen.iter
        iter_text = None
        if isinstance(iter_node, _ast.Name):
            iter_text = iter_node.id
        else:
            try:
                iter_text = _ast.unparse(iter_node)
            except Exception:
                iter_text = None
        if not iter_text:
            continue

        line_idx = node.lineno - 1
        if line_idx >= len(lines):
            continue
        line = lines[line_idx]
        start = node.col_offset
        end = getattr(node, 'end_col_offset', start + 1)
        new_text = f"set({iter_text})"
        edits[line_idx] = line[:start] + new_text + line[end:]
        changed = True

    if changed:
        modified = list(lines)
        for idx in sorted(edits.keys(), reverse=True):
            modified[idx] = edits[idx]
        return "\n".join(modified)
    return content


def _simplify_boolop_to_in(content: str) -> str:
    """将 BoolOp chain 转换为 membership test。

    - x == a or x == b or x == c → x in (a, b, c)
    - x is a or x is b → x in (a, b)
    - x != a and x != b → x not in (a, b)
    - x is not a and x is not b → x not in (a, b)
    使用 AST 遍历所有 BoolOp 节点，检测 3+ 个同变量比较链。
    """
    try:
        import ast as _ast
        tree = _ast.parse(content)
    except SyntaxError:
        return content

    if not isinstance(tree, _ast.Module) or not tree.body:
        return content

    lines = content.split("\n")
    edits: dict[int, str] = {}
    changed = False
    _NO_VAL = object()  # sentinel for "could not extract value"

    def _get_name(node):
        if isinstance(node, _ast.Name):
            return node.id
        return None

    def _get_value(node):
        if isinstance(node, _ast.Constant):
            return node.value
        return _NO_VAL

    def _extract_comparison(comp):
        """提取比较操作的 (var_name, op_str, value)。"""
        if not isinstance(comp, _ast.Compare):
            return None
        if len(comp.ops) != 1 or len(comp.comparators) != 1:
            return None
        left = _get_name(comp.left)
        if not left:
            return None
        op = comp.ops[0]
        val = _get_value(comp.comparators[0])
        if val is _NO_VAL:
            return None
        if isinstance(op, _ast.Eq):
            return (left, '==', val)
        elif isinstance(op, _ast.Is):
            return (left, 'is', val)
        elif isinstance(op, _ast.IsNot):
            return (left, 'is not', val)
        elif isinstance(op, _ast.NotEq):
            return (left, '!=', val)
        return None

    def _try_convert_boolop(node):
        """尝试将 BoolOp 链转换为 in/not in。"""
        nonlocal changed
        if not isinstance(node, _ast.BoolOp):
            return

        op = node.op
        values = node.values

        if not isinstance(op, (_ast.Or, _ast.And)):
            return
        if len(values) < 2:
            return

        # 提取所有比较
        comparisons = []
        for v in values:
            comp = _extract_comparison(v)
            if not comp:
                return
            comparisons.append(comp)

        # 检查所有比较是否针对同一变量
        var_name = comparisons[0][0]
        comp_type = comparisons[0][1]  # '==', 'is', 'is not', '!='
        for c in comparisons[1:]:
            if c[0] != var_name or c[1] != comp_type:
                return

        # 提取所有值
        vals = [c[2] for c in comparisons]
        vals_str = ", ".join(repr(v) for v in vals)

        # 确定使用 in 还是 not in
        if comp_type in ('==', 'is'):
            new_text = f"{var_name} in ({vals_str})"
        elif comp_type in ('!=', 'is not'):
            new_text = f"{var_name} not in ({vals_str})"
        else:
            return

        line_idx = node.lineno - 1
        if line_idx >= len(lines):
            return
        line = lines[line_idx]
        start = node.col_offset
        end = getattr(node, 'end_col_offset', start + 1)
        edits[line_idx] = line[:start] + new_text + line[end:]
        changed = True

    for node in _ast.walk(tree):
        _try_convert_boolop(node)

    if changed:
        modified = list(lines)
        for idx in sorted(edits.keys(), reverse=True):
            modified[idx] = edits[idx]
        return "\n".join(modified)
    return content


def _remove_await_noop(content: str) -> str:
    """移除 await 无操作调用。

    await asyncio.sleep(0) → pass（在 async 函数中）
    await asyncio.sleep(0) 是常用的 yield 点，但 stripped 模式下可简化。
    """
    try:
        import ast as _ast
        tree = _ast.parse(content)
    except SyntaxError:
        return content

    if not isinstance(tree, _ast.Module) or not tree.body:
        return content

    edits: dict[int, str] = {}
    changed = False
    content_lines = content.split("\n")

    def _is_await_sleep_zero(node):
        """检查是否为 await asyncio.sleep(0)。"""
        if not isinstance(node, _ast.Await):
            return False
        call = node.value
        if not isinstance(call, _ast.Call):
            return False
        # Check asyncio.sleep(0) or sleep(0)
        func = call.func
        if isinstance(func, _ast.Attribute):
            if isinstance(func.value, _ast.Name) and func.value.id == 'asyncio' and func.attr == 'sleep':
                pass
            else:
                return False
        elif isinstance(func, _ast.Name) and func.id == 'sleep':
            pass
        else:
            return False
        if len(call.args) == 1 and isinstance(call.args[0], _ast.Constant) and call.args[0].value == 0:
            return True
        return False

    def _check_block(block):
        nonlocal changed
        for stmt in block:
            if isinstance(stmt, _ast.Expr) and _is_await_sleep_zero(stmt.value):
                s = getattr(stmt, 'lineno', None)
                e = getattr(stmt, 'end_lineno', s)
                if s is not None:
                    for li in range(s - 1, e):
                        if li < len(content_lines):
                            edits[li] = ""
                    changed = True

    def _walk(node):
        if isinstance(node, (_ast.FunctionDef, _ast.AsyncFunctionDef)):
            _check_block(node.body)
        for child in _ast.iter_child_nodes(node):
            _walk(child)

    _walk(tree)

    if changed:
        modified = list(content_lines)
        for idx in sorted(edits.keys(), reverse=True):
            modified[idx] = ""
        content = "\n".join(modified)

    return content


def _inline_trivial_getter(content: str) -> str:
    """内联单语句 getter 方法的调用。

    对于只有 `return self.x` 的 getter 方法，
    如果调用次数少，保留方法更短。此函数暂不做内联（保守）。

    替代方案：移除只有 `pass` 的 getter/setter stub。
    """
    return content


def _simplify_identity_checks(content: str) -> str:
    """简化同一性检查：x is x → True, x is not x → False。

    检测比较同一对象的同一性检查并简化为常量。
    """
    try:
        import ast as _ast
        tree = _ast.parse(content)
    except SyntaxError:
        return content

    if not isinstance(tree, _ast.Module) or not tree.body:
        return content

    edits: dict[int, str] = {}
    changed = False
    content_lines = content.split("\n")

    def _try_simplify(node) -> None:
        nonlocal changed
        if isinstance(node, _ast.Compare) and len(node.ops) == 1:
            op = node.ops[0]
            left = node.left
            right = node.comparators[0]

            if isinstance(op, _ast.Is):
                # x is x → True, None is None → True
                if _ast.dump(left) == _ast.dump(right):
                    s = getattr(node, 'lineno', None)
                    sc = getattr(node, 'col_offset', None)
                    ec = getattr(node, 'end_col_offset', None)
                    if s is not None and sc is not None and ec is not None:
                        line = content_lines[s - 1]
                        prefix = line[:sc]
                        suffix = line[ec:]
                        new_line = prefix + "True" + suffix
                        if new_line != line:
                            edits[s - 1] = new_line
                            changed = True
                        return

            elif isinstance(op, _ast.IsNot):
                # x is not x → False
                if _ast.dump(left) == _ast.dump(right):
                    s = getattr(node, 'lineno', None)
                    sc = getattr(node, 'col_offset', None)
                    ec = getattr(node, 'end_col_offset', None)
                    if s is not None and sc is not None and ec is not None:
                        line = content_lines[s - 1]
                        prefix = line[:sc]
                        suffix = line[ec:]
                        new_line = prefix + "False" + suffix
                        if new_line != line:
                            edits[s - 1] = new_line
                            changed = True
                        return

        for child in _ast.iter_child_nodes(node):
            _try_simplify(child)

    _try_simplify(tree)

    if changed:
        modified = list(content_lines)
        for idx in sorted(edits.keys(), reverse=True):
            modified[idx] = edits[idx]
        content = "\n".join(modified)

    return content


def _remove_empty_with(content: str) -> str:
    """移除只有 pass/空语句的 with 块。

    with open(f) as fh:
        pass
    → （移除整个 with）

    保守处理：只移除 body 完全为空的 with 语句。
    """
    try:
        import ast as _ast
        tree = _ast.parse(content)
    except SyntaxError:
        return content

    if not isinstance(tree, _ast.Module) or not tree.body:
        return content

    edits: dict[int, str] = {}
    changed = False
    content_lines = content.split("\n")

    def _is_empty_body(stmts):
        """检查语句列表是否只有 pass/字符串表达式。"""
        for stmt in stmts:
            if isinstance(stmt, _ast.Pass):
                continue
            elif isinstance(stmt, _ast.Expr) and isinstance(stmt.value, _ast.Constant) and isinstance(stmt.value.value, str):
                continue
            else:
                return False
        return True

    def _check_with(node: _ast.With) -> None:
        nonlocal changed
        if not _is_empty_body(node.body):
            return
        # Remove the with statement
        start = node.lineno - 1
        end = getattr(node, 'end_lineno', node.lineno)
        for li in range(start, end):
            if li < len(content_lines):
                edits[li] = ""
        changed = True

    def _walk(node):
        if isinstance(node, _ast.With):
            _check_with(node)
        for child in _ast.iter_child_nodes(node):
            _walk(child)

    _walk(tree)

    if changed:
        modified = list(content_lines)
        for idx in sorted(edits.keys(), reverse=True):
            modified[idx] = ""
        content = "\n".join(modified)

    return content


def _remove_dead_if_const(content: str) -> str:
    """移除 if True/if False 死代码块。

    if True:
        body
    → body（去缩进）

    if False:
        body
    → 移除（死代码）

    同时处理 else 分支：
    if True:\n        body1\n    else:\n        body2\n    → body1
    if False:\n        body1\n    else:\n        body2\n    → body2
    """
    try:
        import ast as _ast
        tree = _ast.parse(content)
    except SyntaxError:
        return content

    if not isinstance(tree, _ast.Module) or not tree.body:
        return content

    edits: dict[int, str] = {}
    changed = False
    content_lines = content.split("\n")

    def _check_if(node: _ast.If) -> None:
        nonlocal changed
        # Check if condition is True or False constant
        if not isinstance(node.test, _ast.Constant):
            return
        if node.test.value not in (True, False):
            return
        is_true = node.test.value is True

        start_line = node.lineno - 1
        header_indent = len(content_lines[start_line]) - len(content_lines[start_line].lstrip())
        body_start = node.body[0].lineno - 1 if node.body else start_line + 1
        body_end = getattr(node.body[-1], 'end_lineno', body_start + 1) if node.body else body_start + 1

        if is_true:
            # if True: body [else: else_body] → body (dedented)
            # Remove the if line
            edits[start_line] = ""
            # Dedent body lines by 4 spaces (one indentation level)
            for li in range(body_start, body_end):
                if li < len(content_lines):
                    line = content_lines[li]
                    stripped = line.lstrip()
                    if stripped:
                        current_indent = len(line) - len(stripped)
                        if current_indent >= 4:
                            edits[li] = line[4:]
                        else:
                            edits[li] = stripped
            # Remove else block if present
            if node.orelse:
                else_start = node.orelse[0].lineno - 1 if node.orelse else None
                else_end = getattr(node.orelse[-1], 'end_lineno', else_start + 1) if node.orelse else else_start + 1
                if else_start is not None:
                    for li in range(else_start, else_end):
                        if li < len(content_lines):
                            edits[li] = ""
            changed = True
        else:
            # if False: body [else: else_body] → else_body (dedented) or nothing
            # Remove the if line, body, and else header
            edits[start_line] = ""
            for li in range(body_start, body_end):
                if li < len(content_lines):
                    edits[li] = ""
            if node.orelse:
                else_start = node.orelse[0].lineno - 1 if node.orelse else None
                else_end = getattr(node.orelse[-1], 'end_lineno', else_start + 1) if node.orelse else else_start + 1
                else_header = else_start
                if else_header is not None:
                    edits[else_header] = ""  # remove "else:"
                    # Dedent else body by 4 spaces
                    for li in range(else_header + 1, else_end):
                        if li < len(content_lines):
                            line = content_lines[li]
                            stripped = line.lstrip()
                            if stripped:
                                current_indent = len(line) - len(stripped)
                                if current_indent >= 4:
                                    edits[li] = line[4:]
                                else:
                                    edits[li] = stripped
            changed = True

    def _walk(node):
        if isinstance(node, _ast.If):
            _check_if(node)
        for child in _ast.iter_child_nodes(node):
            _walk(child)

    _walk(tree)

    if changed:
        modified = list(content_lines)
        for idx in sorted(edits.keys(), reverse=True):
            modified[idx] = edits[idx]
        content = "\n".join(modified)

    return content


def _remove_excess_blank_lines(content: str) -> str:
    """移除所有空行，最大化 token 压缩。

    在 stripped 模式下，可读性由 detail_level 控制，
    此处追求极致压缩：移除所有空行。
    """
    lines = content.split("\n")
    result = [line for line in lines if line.strip()]
    return "\n".join(result)


def normalize_whitespace(content: str, ext: str) -> str:
    """标准化文件空白字符，减少 ~5-10% token。

    - 先做 Unicode NFC 规范化 + BOM 去除
    - Python: 去除行尾空白，折叠连续空行（最多保留 2 个），
              去除首尾空行，tab 转 4 空格
    - YAML/TOML: 去除行尾空白，折叠连续空行（最多保留 1 个）
    - 其他: 仅去除行尾空白
    """
    ext = ext.lower()
    # Unicode 规范化（幂等：对已规范化的内容无副作用）
    content = unicode_normalize(content)

    if ext == ".py":
        content = _remove_future_imports(content)
        content = _collapse_python_imports(content)

    # Markdown: 去除 YAML frontmatter
    if ext in {".md", ".markdown"}:
        content = _strip_markdown_frontmatter(content)

    lines = content.split("\n")

    if ext == ".py":
        # tab → 4 空格
        lines = [line.expandtabs(4) for line in lines]
        # 去除行尾空白
        lines = [line.rstrip() for line in lines]
        # 折叠连续空行：最多保留 2 个连续空行（保留段落间距）
        lines = _collapse_empty_lines(lines, 2)
    elif ext in {".yaml", ".yml", ".toml"}:
        # 去除行尾空白
        lines = [line.rstrip() for line in lines]
        # 折叠连续空行：最多保留 1 个
        lines = _collapse_empty_lines(lines, 1)
    else:
        lines = [line.rstrip() for line in lines]

    # 去除首尾空行
    while lines and lines[0] == "":
        lines.pop(0)
    while lines and lines[-1] == "":
        lines.pop()

    # 去除连续重复行（如连续的 import/空行/注释）
    lines = _deduplicate_consecutive_lines(lines)

    return "\n".join(lines)


def _collapse_python_imports(content: str) -> str:
    """将多行 import 压缩为单行（Python）。"""
    try:
        import ast as _ast
        tree = _ast.parse(content)
    except SyntaxError:
        return content

    lines = content.split("\n")
    modified = list(lines)
    replacements = []

    for node in _ast.walk(tree):
        if not isinstance(node, _ast.ImportFrom) or node.module is None:
            continue
        start = getattr(node, 'lineno', None)
        end = getattr(node, 'end_lineno', None)
        if not start or not end or start == end:
            continue

        start_idx = start - 1
        end_idx = end - 1

        # 检查是否是多行 import（有括号）
        orig = modified[start_idx]
        if '(' not in orig:
            continue

        # 重建为单行（过滤冗余别名：y as y → y）
        names = []
        for alias in node.names:
            if alias.asname and alias.asname != alias.name:
                names.append(f"{alias.name} as {alias.asname}")
            else:
                names.append(alias.name)

        level = '.' * node.level if node.level else ''
        module = node.module or ''
        new_line = f"from {level}{module} import {', '.join(names)}"
        replacements.append((start_idx, end_idx, new_line))

    # 从后往前替换（避免行号偏移）
    for start_idx, end_idx, new_line in sorted(replacements, reverse=True):
        modified[start_idx:end_idx + 1] = [new_line]

    return "\n".join(modified)


def _comment_density(content: str, ext: str) -> float:
    """估算文件的注释密度（0.0 - 1.0）。

    用于快速判断：如果文件大部分是注释， stripping 不会节省很多 token，
    可以跳过昂贵的 AST 解析。
    """
    if not content.strip():
        return 0.0

    lines = content.split("\n")
    if not lines:
        return 0.0

    comment_lines = 0
    if ext.lower() == ".py":
        # 统计以 # 开头的行（粗略估算）
        for line in lines:
            stripped = line.strip()
            if stripped.startswith("#"):
                comment_lines += 1
            elif stripped.startswith(('"""', "'''")):
                # docstring 也算注释
                comment_lines += 1
    else:
        pattern = _COMMENT_PATTERNS.get(ext.lower())
        if pattern:
            for line in lines:
                stripped = line.strip()
                if not stripped:
                    continue
                # 简单启发式：行首有注释标记
                if ext.lower() in {".js", ".ts", ".java", ".c", ".cpp", ".h", ".go", ".rs"}:
                    if stripped.startswith("//") or stripped.startswith("/*"):
                        comment_lines += 1
                elif ext.lower() in {".yaml", ".yml", ".toml", ".sh", ".bash"}:
                    if stripped.startswith("#"):
                        comment_lines += 1

    return comment_lines / len(lines)


def _deduplicate_consecutive_lines(lines: list[str]) -> list[str]:
    """去除连续重复的行（保留首个，标记后续为重复）。

    例如：
        import os
        import os        ← 移除
        import sys
        import sys       ← 移除
    变为：
        import os
        import sys
    """
    if not lines:
        return lines

    result = []
    prev_line = None
    for line in lines:
        stripped = line.strip()
        # 空行和纯注释行不参与去重（保留结构）
        if not stripped or stripped.startswith("#"):
            result.append(line)
            prev_line = None
            continue
        if stripped == prev_line:
            # 跳过重复的非空行
            continue
        result.append(line)
        prev_line = stripped
    return result


def _is_already_minified(content: str, ext: str) -> bool:
    """检查内容是否已经是压缩格式（无需再处理）。

    快速跳过已经高度压缩的文件，节省处理时间。
    """
    if not content:
        return True

    lines = content.split("\n")
    if not lines:
        return True

    # 检查是否有行尾空白
    if any(line != line.rstrip() for line in lines):
        return False

    # 检查是否有连续空行（超过阈值视为未压缩）
    empty_count = 0
    max_empty = 3 if ext.lower() == ".py" else 2
    for line in lines:
        if not line.strip():
            empty_count += 1
            if empty_count >= max_empty:
                return False
        else:
            empty_count = 0

    return True


def _strip_markdown_frontmatter(content: str) -> str:
    """去除 Markdown 文件中的 YAML frontmatter（静态站点生成器元数据）。"""
    if not content.startswith("---"):
        return content
    # 找到第二个 ---
    end = content.find("\n---", 3)
    if end == -1:
        return content
    # 跳过 frontmatter 后的首尾空行
    rest = content[end + 4:].lstrip("\n")
    return rest


def _strip_empty_lines(content: str) -> str:
    """去除内容中的空行（保留非空行和行尾空白清理）。"""
    lines = content.split("\n")
    return "\n".join(line.rstrip() for line in lines if line.strip())


def remove_redundant_pass(content: str) -> str:
    """去除 Python 文件中冗余的 pass 语句。

    规则：
    1. 单独一行的 pass（前后都是空行）可安全去除
    2. 紧跟在 docstring/注释后的 pass 可去除（如果是空函数/类的唯一语句）
    3. 保留：pass 前有其他代码、pass 与代码同行、try/except 中的 pass
    """
    if "pass" not in content:
        return content

    lines = content.split("\n")
    result = []
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if stripped == "pass":
            if _is_redundant_pass(lines, i):
                i += 1
                continue

        result.append(line)
        i += 1

    return "\n".join(result)


def _is_redundant_pass(lines: list[str], pass_idx: int) -> bool:
    """判断某行的 pass 是否可以安全去除。

    可去除条件：
    - pass 所在块的头部不是控制流语句（except/else/finally 等必须保留 pass）
    - pass 之前没有其他可执行语句（同缩进）
    - pass 之后没有其他可执行语句（同缩进）
    """
    pass_line = lines[pass_idx]
    if not pass_line.strip():
        return False
    pass_indent = len(pass_line) - len(pass_line.lstrip())

    # 向后扫描：找块头部（缩进更小的非空行）
    block_header = None
    for j in range(pass_idx - 1, -1, -1):
        prev = lines[j]
        if not prev.strip():
            continue
        prev_indent = len(prev) - len(prev.lstrip())
        if prev_indent < pass_indent:
            block_header = prev.strip()
            break
        if prev_indent == pass_indent:
            # 同缩进行 = 块内其他语句，pass 不是唯一语句
            return False

    # 控制流块（except/else/finally/for/while/if/with/try）中的 pass 不可去除
    if block_header and _is_control_flow_header(block_header):
        return False

    # 向前扫描：检查 pass 之后是否有其他可执行语句（同缩进）
    for j in range(pass_idx + 1, len(lines)):
        nxt = lines[j]
        stripped = nxt.strip()
        if not stripped:
            continue
        nxt_indent = len(nxt) - len(nxt.lstrip())
        if nxt_indent == pass_indent:
            return False  # 同缩进有其他语句
        if nxt_indent < pass_indent:
            break  # 出了当前块

    return True


def _collapse_empty_lines(lines: list[str], max_consecutive: int) -> list[str]:
    """折叠连续空行，最多保留 max_consecutive 个。"""
    if max_consecutive <= 0:
        return [line for line in lines if line.strip() or line != ""]
    collapsed = []
    empty_count = 0
    for line in lines:
        if not line.strip():
            empty_count += 1
            if empty_count <= max_consecutive:
                collapsed.append(line)
        else:
            empty_count = 0
            collapsed.append(line)
    return collapsed


def _is_control_flow_header(line: str) -> bool:
    """判断一行是否是控制流块头部。"""
    _CTRL = ("except:", "finally:", "else:", "elif:", "if:", "for:", "while:", "with:", "try:", "async for:", "async with:")
    for ctrl in _CTRL:
        if line == ctrl or line.startswith(ctrl + " "):
            return True
    return False


def _compress_empty_class_bodies(content: str) -> str:
    """将只有 pass 的空类体压缩为单行标记（stripped/skeleton 模式）。"""
    lines = content.split("\n")
    result = []
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if stripped.startswith("class ") and stripped.endswith(":"):
            if i + 1 < len(lines):
                next_stripped = lines[i + 1].strip()
                if next_stripped in ("pass", "..."):
                    result.append(line + " ...  # empty")
                    i += 2
                    continue
        result.append(line)
        i += 1
    return "\n".join(result)


def _compress_empty_bodies(content: str) -> str:
    """压缩所有空代码块（函数、if/for/while/try/with）为单行标记。

    例如：
        def foo():        →  def foo(): ...  # empty
            pass
        if condition:     →  if condition: ...  # empty
            pass
    """
    lines = content.split("\n")
    result = []
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # 检测函数定义（已有 def foo(): ... 形式的不再处理）
        if stripped.startswith("def ") or stripped.startswith("async def "):
            if stripped.endswith("..."):
                result.append(line)
                i += 1
                continue
            if stripped.endswith(":"):
                if i + 1 < len(lines):
                    next_stripped = lines[i + 1].strip()
                    if next_stripped in ("pass", "..."):
                        result.append(line + " ...  # empty")
                        i += 2
                        continue
                    # 检查是否只有 return None
                    if next_stripped == "return None":
                        result.append(line + " ...  # empty")
                        i += 2
                        continue
                    # 检查是否只有 return
                    if next_stripped == "return":
                        result.append(line + " ...  # empty")
                        i += 2
                        continue

        # 检测 if/for/while/with/try 块
        block_prefixes = ("if ", "elif ", "else:", "for ", "while ", "try:", "with ", "except", "finally:")
        if any(stripped.startswith(p) for p in block_prefixes) and stripped.endswith(":"):
            if i + 1 < len(lines):
                next_stripped = lines[i + 1].strip()
                if next_stripped in ("pass", "..."):
                    result.append(line + " ...  # empty")
                    i += 2
                    continue

        result.append(line)
        i += 1
    return "\n".join(result)


def _remove_main_guard(content: str) -> str:
    """去除 if __name__ == "__main__" 块（stripped/skeleton 模式）。"""
    try:
        import ast as _ast
        tree = _ast.parse(content)
    except SyntaxError:
        return content

    lines = content.split("\n")
    to_remove: list[tuple[int, int]] = []

    for node in getattr(tree, "body", []):
        if not isinstance(node, _ast.If):
            continue
        # 检查是否为 if __name__ == "__main__"（使用 == 或 is）
        test = node.test
        if isinstance(test, _ast.Compare):
            is_name_main = (
                isinstance(test.left, _ast.Name)
                and test.left.id == "__name__"
                and len(test.ops) == 1
                and len(test.comparators) == 1
                and isinstance(test.comparators[0], _ast.Constant)
                and test.comparators[0].value == "__main__"
                and isinstance(test.ops[0], (_ast.Eq, _ast.Is))
            )
            if is_name_main:
                start = getattr(node, "lineno", None)
                end = getattr(node, "end_lineno", None)
                if start and end:
                    to_remove.append((start - 1, end))

    if not to_remove:
        return content

    # 从后往前删除
    modified = list(lines)
    for start, end in sorted(to_remove, reverse=True):
        # 检查前一行是否为空行（通常 __main__ 块前有空行分隔）
        if start > 0 and not modified[start - 1].strip():
            start -= 1
        modified[start:end] = []

    return "\n".join(modified)


def _remove_dead_code_after_control_flow(content: str) -> str:
    """去除 return/raise/break/continue 之后的不可达代码（stripped/skeleton 模式）。

    例如：
        if not x:
            return None
            logger.warning("invalid")   # 不可达
            raise ValueError("bad")      # 不可达
        →
        if not x:
            return None
    """
    try:
        import ast as _ast
        tree = _ast.parse(content)
    except SyntaxError:
        return content

    lines = content.split("\n")
    to_remove: list[tuple[int, int]] = []

    def _scan_body(body_nodes):
        """扫描函数/模块体中的不可达代码。"""
        for i, node in enumerate(body_nodes):
            if isinstance(node, (_ast.Return, _ast.Raise, _ast.Break, _ast.Continue)):
                node_end = getattr(node, "end_lineno", None)
                if node_end is None:
                    continue
                # 查找同一层级中此节点之后的语句
                node_end_idx = node_end - 1  # 1-based → 0-based
                next_idx = node_end_idx + 1
                while next_idx < len(lines):
                    line = lines[next_idx]
                    stripped = line.strip()
                    if not stripped:
                        next_idx += 1
                        continue
                    # 遇到同级或更浅缩进的语句停止
                    current_indent = len(lines[node_end_idx]) - len(lines[node_end_idx].lstrip())
                    next_indent = len(line) - len(line.lstrip())
                    if next_indent < current_indent:
                        break
                    # 不可达代码：从 next_idx 到下一个同级语句
                    dead_end = next_idx + 1
                    while dead_end < len(lines):
                        dead_line = lines[dead_end]
                        if not dead_line.strip():
                            dead_end += 1
                            continue
                        dead_indent = len(dead_line) - len(dead_line.lstrip())
                        if dead_indent < current_indent:
                            break
                        dead_end += 1
                    to_remove.append((next_idx, dead_end))
                    next_idx = dead_end


    # 递归扫描所有含 body 的节点（函数、类、if、for、while、try 等）
    def _walk_all_bodies(nodes):
        for node in nodes:
            if not hasattr(node, "body"):
                continue
            body = getattr(node, "body", [])
            if isinstance(body, list) and body:
                _scan_body(body)
            # 递归子节点
            for child in ast.iter_child_nodes(node):
                if child is not node:
                    _walk_all_bodies([child])

    _walk_all_bodies(getattr(tree, "body", []))

    if not to_remove:
        return content

    # 合并重叠区间
    to_remove.sort()
    merged = [to_remove[0]]
    for s, e in to_remove[1:]:
        if s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))

    # 从后往前删除
    modified = list(lines)
    for start, end in sorted(merged, reverse=True):
        modified[start:end] = []

    return "\n".join(modified)


def _compress_assert_statements(content: str) -> str:
    """压缩冗长的 assert 语句（stripped 模式专用）。

    例如：
        assert x is not None  →  assert x
        assert x is None      →  assert not x
        assert x is True      →  assert x
        assert x is False     →  assert not x
        assert len(x) > 0    →  assert x
        assert x != []        →  assert x
    """
    lines = content.split("\n")
    result = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("assert ") and (" is " in stripped or "len(" in stripped or " != " in stripped or "bool(" in stripped):
            # 简化 is not None / is None / is True / is False
            new = stripped
            new = re.sub(r'(\w+)\s+is\s+not\s+None', r'\1', new)
            new = re.sub(r'(\w+)\s+is\s+None', r'not \1', new)
            new = re.sub(r'(\w+)\s+is\s+True\b', r'\1', new)
            new = re.sub(r'(\w+)\s+is\s+False\b', r'not \1', new)
            new = re.sub(r'len\((\w+)\)\s*>\s*0', r'\1', new)
            new = re.sub(r'(\w+)\s*!=\s*\[\]', r'\1', new)
            new = re.sub(r'(\w+)\s*!=\s*""', r'\1', new)
            new = re.sub(r'bool\((\w+)\)', r'\1', new)
            # 缩进对齐
            indent = line[:len(line) - len(line.lstrip())]
            result.append(indent + new)
        else:
            result.append(line)
    return "\n".join(result)


def _remove_raise_from_none(content: str) -> str:
    """去除 raise 语句中的 from None 子句（stripped/skeleton 模式）。

    例如：
        raise ValueError("bad") from None  →  raise ValueError("bad")
    """
    lines = content.split("\n")
    result = []
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("raise ") and " from None" in stripped:
            line = line.replace(" from None", "")
        result.append(line)
    return "\n".join(result)


def _compress_self_assignments(content: str) -> str:
    """将连续的 self.x = x 赋值压缩为 self.__dict__.update(x=x, y=y)。

    例如：
        self.x = x
        self.y = y
        self.z = z
    →
        self.__dict__.update(x=x, y=y, z=z)

    仅在 stripped 模式且连续 2+ 个匹配时应用。
    安全规则：所有值必须是函数参数或局部变量（不在 SkipNames 中）。
    """
    try:
        import ast as _ast
        tree = _ast.parse(content)
    except SyntaxError:
        return content

    lines = content.split("\n")
    replacements = []

    for node in _ast.walk(tree):
        if not isinstance(node, (_ast.FunctionDef, _ast.AsyncFunctionDef)):
            continue
        if node.name != "__init__":
            continue

        # 收集函数中定义的名称（参数 + 局部变量）
        defined_names: set[str] = set()
        for arg in node.args.args:
            defined_names.add(arg.arg)
        for arg in getattr(node.args, 'posonlyargs', []):
            defined_names.add(arg.arg)
        if node.args.vararg:
            defined_names.add(node.args.vararg.arg)
        if node.args.kwarg:
            defined_names.add(node.args.kwarg.arg)
        for arg in node.args.kwonlyargs:
            defined_names.add(arg.arg)
        for arg in node.args.kw_defaults:
            if arg and isinstance(arg, _ast.Name):
                defined_names.add(arg.id)

        body = getattr(node, "body", [])
        # 查找连续的 self.x = x 赋值
        i = 0
        while i < len(body):
            if not isinstance(body[i], _ast.Assign):
                i += 1
                continue
            assign = body[i]
            # 检查目标是否是 self.name
            if (len(assign.targets) != 1 or
                    not isinstance(assign.targets[0], _ast.Attribute) or
                    not isinstance(assign.targets[0].value, _ast.Name) or
                    assign.targets[0].value.id != "self"):
                i += 1
                continue
            # 检查值是否是简单的 Name（同名的局部变量）
            if not isinstance(assign.value, _ast.Name):
                i += 1
                continue
            attr_name = assign.targets[0].attr
            if assign.value.id != attr_name:
                i += 1
                continue
            # 检查值是否在已定义名称中（安全规则）
            if assign.value.id not in defined_names:
                i += 1
                continue
            # 收集连续的 self.x = x
            batch_start = i
            batch = []
            while i < len(body):
                if not isinstance(body[i], _ast.Assign):
                    break
                a = body[i]
                if (len(a.targets) != 1 or
                        not isinstance(a.targets[0], _ast.Attribute) or
                        not isinstance(a.targets[0].value, _ast.Name) or
                        a.targets[0].value.id != "self"):
                    break
                if not isinstance(a.value, _ast.Name):
                    break
                if a.targets[0].attr != a.value.id:
                    break
                # 安全检查：值必须在已定义名称中
                if a.value.id not in defined_names:
                    break
                batch.append((a.targets[0].attr, a.value.id))
                i += 1
            # 需要至少 2 个才值得压缩
            if len(batch) >= 2:
                start_line = getattr(body[batch_start], "lineno", None)
                end_line = getattr(body[i - 1], "end_lineno", None)
                if start_line and end_line:
                    start_idx = start_line - 1
                    end_idx = end_line
                    pairs = ", ".join(f"{k}={v}" for k, v in batch)
                    update_call = f"self.__dict__.update({pairs})"
                    # 清除所有赋值行，替换为 update 调用
                    for li in range(start_idx, end_idx):
                        if li < len(lines):
                            lines[li] = ""
                    lines[start_idx] = update_call
            continue

    # 检查是否有变更
    original = content
    modified = list(lines)
    result = "\n".join(modified)
    if result != original:
        return result
    return content

    return "\n".join(lines)


def _inline_single_expr_functions(content: str) -> str:
    """将只有单个表达式的函数体压缩为单行（stripped 模式）。

    例如：
        def get_name(self):
            return self.name
    →
        def get_name(self): return self.name
    """
    try:
        import ast as _ast
        tree = _ast.parse(content)
    except SyntaxError:
        return content

    lines = content.split("\n")
    replacements = []

    for node in _ast.walk(tree):
        if not isinstance(node, (_ast.FunctionDef, _ast.AsyncFunctionDef)):
            continue
        body = getattr(node, "body", [])
        # 过滤掉 docstring 和 pass
        meaningful = [s for s in body if not (
            isinstance(s, _ast.Expr) and isinstance(getattr(s, "value", None), _ast.Constant) and isinstance(s.value.value, str)
        ) and not isinstance(s, _ast.Pass)]
        if len(meaningful) != 1:
            continue
        stmt = meaningful[0]
        if not isinstance(stmt, (_ast.Return, _ast.Raise, _ast.Assign, _ast.AnnAssign, _ast.Expr)):
            continue
        # 获取签名行和语句行
        sig_line = getattr(node, "lineno", None)
        stmt_line = getattr(stmt, "lineno", None)
        stmt_end = getattr(stmt, "end_lineno", None)
        if not sig_line or not stmt_line or not stmt_end:
            continue
        if sig_line != stmt_line:
            # 合并为单行
            sig_text = lines[sig_line - 1].rstrip()
            stmt_text = lines[stmt_line - 1].strip()
            # sig_text 已含尾冒号（def foo():），不再添加
            new_line = f"{sig_text} {stmt_text}"
            # 删除签名和语句之间的行，以及语句行
            replacements.append((sig_line, stmt_end, new_line))

    # 应用替换（从后往前）
    for start_1based, end_1based, new_line in sorted(replacements, reverse=True):
        start_idx = start_1based - 1
        end_idx = end_1based
        lines[start_idx:end_idx] = [new_line]

    return "\n".join(lines)


def unicode_normalize(content: str) -> str:
    """Unicode NFC 规范化 + 去除 BOM，确保内容一致性。"""
    import unicodedata
    # NFC 规范化：兼容分解字符（如 é → e + ́ → é）
    content = unicodedata.normalize("NFC", content)
    # 去除 UTF-8 BOM
    if content.startswith("﻿"):
        content = content[1:]
    # 去除零宽字符（U+200B 零宽空格、U+200C 零宽非连接符、U+200D 零宽连接符、U+200E 左到右标记、U+200F 右到左标记、U+FEFF BOM）
    content = re.sub(r"[​‌‍‎‏﻿]", "", content)
    return content


def _minify_structured(content: str, ext: str) -> str:
    """压缩 JSON/YAML/TOML 等结构化文件。

    - JSON: 解析后重新序列化为紧凑格式（~40-60% 节省）
    - YAML: 简单去空行和多余缩进
    - TOML: 去空行
    """
    ext = ext.lower()
    if ext == ".json":
        try:
            import json as _json
            data = _json.loads(content)
            return _json.dumps(data, ensure_ascii=False, separators=(",", ":"))
        except (json.JSONDecodeError, Exception):
            # 回退：简单去空白
            return re.sub(r'\s+', ' ', content).strip()
    elif ext in {".yaml", ".yml"}:
        content = _strip_markdown_frontmatter(content)  # 兼容 .md 格式的 YAML
        return _strip_empty_lines(content)
    elif ext in {".md", ".markdown", ".toml"}:
        # 去除 YAML frontmatter（仅 .md/.markdown）+ 多余空行
        if ext in {".md", ".markdown"}:
            content = _strip_markdown_frontmatter(content)
        return _strip_empty_lines(content)
    return content


# ── 输出格式化 ──

def format_processed_output(result: dict, format: str = "markdown") -> str:
    """将处理结果格式化为可发送给 Claude 的文本。

    format 选项:
      - "markdown": 标准 Markdown，每个文件带标题
      - "markdown-compact": 紧凑 Markdown，省略文件标题和统计
      - "plain": 分隔符格式
      - "raw": 最简格式，仅路径+内容，最小开销
      - "super-compact": 超紧凑格式，使用 --- 分隔符，~60% token 比 raw
      - "json": JSON 格式
    """
    # 构建路径缩写映射（节省路径相关 token）
    all_paths = [f["path"] for f in result.get("files", [])]
    path_map = _build_path_abbreviation_map(all_paths)

    if format == "super-compact":
        # 超紧凑：path\n---\ncontent\n---\n，省略所有 markdown 标记
        parts = []
        for f in result["files"]:
            abbr = path_map.get(f["path"], f["path"])
            content = f["content"].rstrip()
            parts.append(f"{abbr}\n---\n{content}\n---")
        return "\n\n".join(parts)

    if format == "raw":
        # 最简格式：每个文件仅一行路径标记 + 内容，~40% token 比 markdown-compact
        parts = []
        for f in result["files"]:
            abbr = path_map.get(f["path"], f["path"])
            parts.append(f"=== {abbr} ===\n{f['content']}\n")
        return "\n".join(parts)

    if format == "markdown-compact":
        # 紧凑模式：省略文件标题和统计，只保留文件内容块
        parts = [f"# 文件（{len(result['files'])} 个）\n"]
        for f in result["files"]:
            abbr_path = path_map.get(f["path"], f["path"])
            ext = Path(f["path"]).suffix
            parts.append(f"### {abbr_path}\n```{ext.lstrip('.')}\n{f['content']}\n```")
        return "\n".join(parts)

    if format == "markdown":
        parts = []
        parts.append(f"# 文件内容（精简后，共 {len(result['files'])} 个文件）\n")
        for f in result["files"]:
            abbr_path = path_map.get(f["path"], f["path"])
            ext = Path(f["path"]).suffix
            label = f"tokens={f['tokens_after']}"
            if f.get("savings", 0) > 0:
                pct = round(f["savings"] / f["tokens_before"] * 100) if f["tokens_before"] else 0
                label += f", saved {pct}%"
            parts.append(f"## `{abbr_path}` ({label})\n```{ext.lstrip('.')}\n{f['content']}\n```\n")
        return "\n".join(parts)
    elif format == "plain":
        parts = []
        for f in result["files"]:
            abbr_path = path_map.get(f["path"], f["path"])
            parts.append(f"=== {abbr_path} ===\n{f['content']}\n")
        return "\n".join(parts)
    elif format == "json":
        return json.dumps(result, ensure_ascii=False, indent=2)
    else:
        raise ValueError(f"Unknown format: {format}")


def _build_path_abbreviation_map(paths: list[str]) -> dict[str, str]:
    """为输出构建路径缩写映射。"""
    try:
        from claude_token_saver.path_optimizer import build_path_abbreviation_map
        return build_path_abbreviation_map(paths)
    except Exception:
        return {}


# ── 渐进式披露 ────────────────────────────────────────────────────────────

def build_directory_index(
    paths: list[str | Path],
    max_files: int = 200,
    include_binary: bool = False,
    common_dedup: bool = False,
) -> dict:
    """构建目录索引（不读取文件内容，只返回大小和路径）。

    用于渐进式披露：先给 Claude 一个目录骨架，按需读取具体文件。
    适合大项目（>30 个文件）的场景。

    Returns:
        {
            "root": str,
            "total_files": int,
            "total_estimated_tokens": int,
            "files": [{"path", "relative", "size_kb", "tokens", "ext"}, ...],
            "by_directory": {"dir": [...], ...},
            "largest_files": [...],
            "skipped_common": [str, ...],  # common_dedup 跳过的文件
        }
    """
    from claude_token_saver.progressive import build_directory_index as _build_index
    index = _build_index(paths, max_files=max_files, include_binary=include_binary)

    # 常见文件组去重
    skipped_common = []
    if common_dedup:
        from claude_token_saver.common_dedup import filter_common_duplicates, get_common_pattern_summary
        suggestions = get_common_pattern_summary([e.path for e in index.files])
        # 过滤重复
        file_paths = [Path(e.path) for e in index.files]
        kept_paths, skip_reasons = filter_common_duplicates(file_paths)
        kept_set = {str(p) for p in kept_paths}
        index.files = [e for e in index.files if e.path in kept_set]
        skipped_common = skip_reasons

    return {
        "root": index.root,
        "total_files": index.total_files,
        "total_estimated_tokens": index.total_tokens,
        "files": [e.to_dict() for e in index.files],
        "by_directory": {
            d: [e.to_dict() for e in entries]
            for d, entries in index.by_directory.items()
        },
        "largest_files": [e.to_dict() for e in index.largest_files],
        "skipped_common": skipped_common,
    }


def format_index_for_prompt(index_data: dict, format: str = "markdown", compact: bool = False) -> str:
    """将目录索引格式化为可注入 Claude prompt 的文本。"""
    from claude_token_saver.progressive import format_index_markdown, format_index_json

    if format == "json":
        from claude_token_saver.progressive import DirectoryIndex
        idx = DirectoryIndex(
            root=index_data.get("root", ""),
            total_files=index_data.get("total_files", 0),
            total_tokens=index_data.get("total_estimated_tokens", 0),
            files=[],
            by_directory={},
            largest_files=[],
        )
        return format_index_json(idx)

    # 从 dict 数据重建 DirectoryIndex 对象
    from claude_token_saver.progressive import DirectoryIndex, FileIndexEntry

    def _entry_from_dict(e: dict) -> FileIndexEntry:
        size_kb = e.get("size_kb", 0)
        size_bytes = int(size_kb * 1024) if size_kb else e.get("size_bytes", 0)
        return FileIndexEntry(
            path=e["path"],
            size_bytes=size_bytes,
            estimated_tokens=e["tokens"],
            ext=e.get("ext", ""),
            relative_path=e.get("relative", ""),
        )

    files = [_entry_from_dict(e) for e in index_data.get("files", [])]
    by_dir = {}
    for d, entries in index_data.get("by_directory", {}).items():
        by_dir[d] = [_entry_from_dict(e) for e in entries]

    idx = DirectoryIndex(
        root=index_data.get("root", ""),
        total_files=index_data.get("total_files", 0),
        total_tokens=index_data.get("total_estimated_tokens", 0),
        files=files,
        by_directory=by_dir,
        largest_files=[],
    )
    return format_index_markdown(idx, compact=compact)
