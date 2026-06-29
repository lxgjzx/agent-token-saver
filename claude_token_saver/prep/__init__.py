"""
Claude Code Token Saver - 预处理模块

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
from pathlib import Path
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

_COMMENT_PATTERNS: dict[str, re.Pattern] = {
    ".py":   re.compile(r'#.*?$|""".*?"""|\'\'\'.*?\'\'\'', re.MULTILINE | re.DOTALL),
    ".js":   re.compile(r'//.*?$|/\*.*?\*/', re.MULTILINE | re.DOTALL),
    ".ts":   re.compile(r'//.*?$|/\*.*?\*/', re.MULTILINE | re.DOTALL),
    ".java": re.compile(r'//.*?$|/\*.*?\*/', re.MULTILINE | re.DOTALL),
    ".c":    re.compile(r'//.*?$|/\*.*?\*/', re.MULTILINE | re.DOTALL),
    ".cpp":  re.compile(r'//.*?$|/\*.*?\*/', re.MULTILINE | re.DOTALL),
    ".h":    re.compile(r'//.*?$|/\*.*?\*/', re.MULTILINE | re.DOTALL),
    ".go":   re.compile(r'//.*?$|/\*.*?\*/', re.MULTILINE | re.DOTALL),
    ".rs":   re.compile(r'//.*?$|/\*.*?\*/', re.MULTILINE | re.DOTALL),
    ".rb":   re.compile(r'#.*?$|=begin.*?=end', re.MULTILINE | re.DOTALL),
    ".php":  re.compile(r'//.*?$|#.*?$|/\*.*?\*/', re.MULTILINE | re.DOTALL),
    ".sh":   re.compile(r'#.*?$', re.MULTILINE),
    ".bash": re.compile(r'#.*?$', re.MULTILINE),
    ".yaml": re.compile(r'#.*?$', re.MULTILINE),
    ".yml":  re.compile(r'#.*?$', re.MULTILINE),
    ".toml": re.compile(r'#.*?$', re.MULTILINE),
    ".json": re.compile(r''),  # JSON 无注释
    ".xml":  re.compile(r'<!--.*?-->', re.DOTALL),
    ".html": re.compile(r'<!--.*?-->', re.DOTALL),
    ".css":  re.compile(r'/\*.*?\*/', re.DOTALL),
    ".sql":  re.compile(r'--.*?$|/\*.*?\*/', re.MULTILINE | re.DOTALL),
}


# ── 缓存 ────────────────────────────────────────────────────────────────

# 全局缓存：key = (file_path, mtime, size) → (content_hash, tokens)
_content_cache: dict[tuple, tuple[str, int]] = {}
# 全局缓存：key = content_hash → token_count
_token_cache: dict[str, int] = {}


def _cache_key(path: Path, content: str) -> tuple:
    """生成缓存键：(路径, mtime, 大小)。"""
    try:
        stat = path.stat()
        return (str(path.resolve()), stat.st_mtime, stat.st_size)
    except OSError:
        return (str(path.resolve()), 0, len(content))


def _clear_caches() -> None:
    """清空所有缓存（用于测试或内存压力时）。"""
    _content_cache.clear()
    _token_cache.clear()


# ── 代码注释去除 ──

def strip_comments(content: str, ext: str) -> str:
    """去除文件中的注释内容。

    注意：非 Python 文件使用正则表达式，可能误伤字符串内的注释符号。
    Python 文件使用 AST 保护字符串区域。
    """
    if ext.lower() == ".py":
        return _strip_python_comments_safe(content)
    pattern = _COMMENT_PATTERNS.get(ext.lower())
    if pattern:
        return pattern.sub("", content)
    return content


def _strip_python_comments_safe(content: str) -> str:
    """安全去除 Python 注释：使用 AST 识别字符串区域，避免误伤。"""
    try:
        import ast as _ast
        tree = _ast.parse(content)
    except SyntaxError:
        # 回退到正则
        pattern = _COMMENT_PATTERNS.get(".py")
        return pattern.sub("", content) if pattern else content

    lines = content.split("\n")
    string_ranges: list[tuple[int, int]] = []

    for node in _ast.walk(tree):
        if isinstance(node, _ast.Constant) and isinstance(node.value, str):
            if hasattr(node, "lineno") and hasattr(node, "end_lineno"):
                start = node.lineno - 1
                end = node.end_lineno
                string_ranges.append((start, end))

    # 标记受保护的字符范围
    protected = [False] * len(lines)
    for start, end in string_ranges:
        for i in range(start, min(end, len(lines))):
            protected[i] = True

    # 逐行处理，跳过字符串内的行
    result_lines = []
    for line_idx, line in enumerate(lines):
        if protected[line_idx]:
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

    return "\n".join(result_lines)


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
    if total <= max_tokens:
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
    """二分查找：找到满足 token 预算的最大行数。"""
    if not lines:
        return 0
    subset = lines if from_start else list(reversed(lines))

    lo, hi = 0, len(subset)
    best = 0
    while lo < hi:
        mid = (lo + hi) // 2
        tokens = count_tokens("\n".join(subset[:mid]))
        if tokens <= token_budget:
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
    - 截断超长部分
    """
    # 去除连续空行（最多保留 1 个）
    text = re.sub(r"\n{3,}", "\n\n", text)
    # 合并行内多余空格
    text = re.sub(r"[ \t]+", " ", text)
    # 去除首尾空白
    text = text.strip()

    if count_tokens(text) <= max_tokens:
        return text

    lines = text.split("\n")
    # 保留前 70%
    keep = int(len(lines) * 0.7)
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
) -> dict:
    """
    处理文件列表，返回精简后的结果。

    detail_level 控制输出粒度：
      - "skeleton": 仅符号索引（类/函数签名），~5-10% token（仅 Python）
      - "stripped": 去除注释和 docstring，~30-50% token
      - "full": 完整内容（默认）
      - "block": 对超大文件返回阻止标记，不读取内容

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
    if dedup:
        file_paths = deduplicate_files(list(file_paths))

    results = []
    total_before = 0
    total_after = 0
    skipped = []
    dup_removed = 0
    cache_hits = 0
    seen_hashes: set[str] = set()

    for fp in file_paths:
        fp = Path(fp)
        if should_ignore(fp, include_binary=include_binary):
            skipped.append(str(fp))
            continue

        try:
            if is_binary_file(fp):
                skipped.append(f"{fp} (binary)")
                continue

            # 缓存检查：跳过未更改的文件
            cache_key = None
            if token_cache_enabled:
                cache_key = _cache_key(fp, "")
                if cache_key in _content_cache:
                    cached_hash, cached_tokens = _content_cache[cache_key]
                    if detail_level == "full":
                        content = fp.read_text(encoding="utf-8", errors="replace")
                        content_hash = hashlib.md5(content.encode()).hexdigest()
                        if content_hash == cached_hash:
                            total_before += cached_tokens
                            total_after += cached_tokens
                            results.append({
                                "path": str(fp),
                                "tokens_before": cached_tokens,
                                "tokens_after": cached_tokens,
                                "savings": 0,
                                "content": content,
                            })
                            cache_hits += 1
                            continue

            content = fp.read_text(encoding="utf-8", errors="replace")
            tokens_before = count_tokens(content)
            total_before += tokens_before

            # 去重
            content_hash = hashlib.md5(content.encode()).hexdigest()
            if content_hash in seen_hashes:
                dup_removed += 1
                continue
            seen_hashes.add(content_hash)

            # 更新缓存
            if token_cache_enabled and cache_key:
                _content_cache[cache_key] = (content_hash, tokens_before)

            # detail_level 路由
            if detail_level == "block" and tokens_before > max_file_tokens:
                skipped.append(f"{fp} (too large: {tokens_before} tokens, use --offset/--limit)")
                continue

            processed_content = _process_content(
                content, fp, detail_level, do_strip_comments, do_strip_docstrings, max_file_tokens
            )

            tokens_after = count_tokens(processed_content)
            total_after += tokens_after

            results.append({
                "path": str(fp),
                "tokens_before": tokens_before,
                "tokens_after": tokens_after,
                "savings": tokens_before - tokens_after,
                "content": processed_content,
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

    if detail_level == "skeleton" and ext in {".py", ".js", ".ts", ".java", ".c", ".cpp", ".h", ".go", ".rs"}:
        # 最高压缩：符号索引 + 骨架
        symbols = extract_symbol_index(content, ext)
        if symbols:
            skeleton = extract_skeleton(content, ext)
            # 如果骨架本身不节省 token，回退到 stripped
            if count_tokens(skeleton) >= count_tokens(content):
                detail_level = "stripped"
            else:
                header = format_symbol_index(symbols, str(fp))
                combined = header + "\n\n" + skeleton if header else skeleton
                # 只在 combined 更优时才带上 header
                if count_tokens(combined) < count_tokens(skeleton):
                    return combined
                return skeleton
        else:
            detail_level = "stripped"

    if detail_level in ("skeleton", "stripped"):
        # 去除注释
        if do_strip_comments:
            content = strip_comments(content, ext)
        # 去除 docstring
        if do_strip_docstrings and ext == ".py":
            content = strip_python_docstrings(content)

    # 智能截断（token 感知）
    content = smart_truncate(content, max_tokens)
    return content


# ── 输出格式化 ──

def format_processed_output(result: dict, format: str = "markdown") -> str:
    """将处理结果格式化为可发送给 Claude 的文本。"""
    if format == "markdown":
        parts = []
        parts.append(f"# 文件内容（精简后，共 {len(result['files'])} 个文件）\n")
        for f in result["files"]:
            ext = Path(f["path"]).suffix
            label = f"tokens={f['tokens_after']}"
            if f.get("savings", 0) > 0:
                pct = round(f["savings"] / f["tokens_before"] * 100) if f["tokens_before"] else 0
                label += f", saved {pct}%"
            parts.append(f"## `{f['path']}` ({label})\n```{ext.lstrip('.')}\n{f['content']}\n```\n")
        return "\n".join(parts)
    elif format == "plain":
        parts = []
        for f in result["files"]:
            parts.append(f"=== {f['path']} ===\n{f['content']}\n")
        return "\n".join(parts)
    elif format == "json":
        return json.dumps(result, ensure_ascii=False, indent=2)
    else:
        raise ValueError(f"Unknown format: {format}")
