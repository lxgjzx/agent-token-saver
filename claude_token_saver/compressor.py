"""
Agent Token Saver - 压缩器模块

提供多级文件表示和代码骨架提取，大幅减少发送给 Claude 的 token 数量。

压缩层级：
  - skeleton: 仅提取函数/类签名、导入、类型定义（~5-10% 原始大小）
  - stripped: 去除注释和 docstring（~30-50% 原始大小）
  - full: 完整内容
  - block: 阻止读取（用于过大的文件）
"""
from __future__ import annotations

import ast
import hashlib
import re
from pathlib import Path
from typing import Any

from claude_token_saver.utils import count_tokens


# ── 语言支持 ────────────────────────────────────────────────────────────

_SKELETON_LANGUAGES: set[str] = {".py", ".js", ".ts", ".java", ".c", ".cpp", ".h", ".go", ".rs"}
_STRIP_LANGUAGES: set[str] = {".rb", ".php", ".sh", ".bash", ".yaml", ".yml", ".toml", ".xml", ".html", ".css", ".sql"}

# 骨架缓存：content_hash -> skeleton_string
_SKELETON_CACHE: dict[str, str] = {}


def clear_compressor_caches() -> None:
    """清空压缩器缓存（测试或内存压力时调用）。"""
    _SKELETON_CACHE.clear()


# ── 骨架提取 ────────────────────────────────────────────────────────────

def extract_skeleton(content: str, ext: str) -> str:
    """提取代码文件的骨架（签名 + 结构，去除函数体）。

    对于 Python 使用 AST 提取，其他语言使用正则近似。
    保留：import、class 定义、函数/方法签名、类型定义、常量声明。
    去除：函数体实现、详细注释、docstring。

    Args:
        content: 源代码内容
        ext: 文件扩展名（含点号）

    Returns:
        骨架代码字符串
    """
    if ext.lower() == ".py":
        return _extract_python_skeleton(content)
    if ext.lower() in {".js", ".ts"}:
        return _extract_js_ts_skeleton(content)
    if ext.lower() in {".java", ".c", ".cpp", ".h", ".go", ".rs"}:
        return _extract_c_like_skeleton(content, ext.lower())
    return content


def _is_decorator_assignment(lines: list[str], assign_end_idx: int, def_lines: set[int]) -> bool:
    """检查一个赋值语句是否是装饰器赋值（如 @property.setter 对应的 value = value.setter(...)）。

    判断依据：赋值语句之后的第一个非空行是 def/class 行。
    """
    next_idx = assign_end_idx + 1
    while next_idx < len(lines):
        stripped = lines[next_idx].strip()
        if stripped:
            # 检查是否是以 def/class 开头的行
            if stripped.startswith("def ") or stripped.startswith("class ") or stripped.startswith("async def "):
                return True
            return False
        next_idx += 1
    return False


def _extract_python_skeleton(content: str) -> str:
    """使用 AST 提取 Python 文件骨架（带 docstring 摘要注入）。"""
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return content

    lines = content.split("\n")

    # Pass 1: 收集 docstring 摘要（key = AST 节点起始行号 1-based）
    docstring_summaries: dict[int, str] = {}
    for node in ast.walk(tree):
        if not isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if not hasattr(node, "body") or not node.body:
            continue
        first = node.body[0]
        if not isinstance(first, ast.Expr):
            continue
        val = getattr(first, "value", None)
        if not isinstance(val, ast.Constant) or not isinstance(val.value, str):
            continue
        first_line = val.value.strip().split("\n")[0].strip()
        if first_line:
            docstring_summaries[node.lineno] = first_line[:80]

    # Pass 2: 构建保留区间
    keep_ranges: list[tuple[int, int]] = []
    # 收集所有 class/function 定义的行号（用于过滤装饰器赋值）
    _def_lines: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            _def_lines.add(node.lineno)

    # 顶层节点：仅保留模块级的 import 和赋值
    for node in getattr(tree, "body", []):
        start = getattr(node, "lineno", None)
        end = getattr(node, "end_lineno", None)
        if start is None or end is None:
            continue
        start_idx = start - 1
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            keep_ranges.append((start_idx, end))
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            keep_ranges.append((start_idx, end))

    # 递归处理 class/function（含装饰器、签名、类级常量）
    def _walk_code_nodes(node, is_class: bool = False):
        """处理 class/function 节点及其子节点。"""
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            start = getattr(node, "lineno", None)
            end = getattr(node, "end_lineno", None)
            if start is None or end is None:
                return
            start_idx = start - 1

            decorator_lines = []
            for deco in getattr(node, "decorator_list", []):
                deco_start = getattr(deco, "lineno", None)
                deco_end = getattr(deco, "end_lineno", None)
                if deco_start and deco_end:
                    decorator_lines.append((deco_start - 1, deco_end))

            sig_start = start_idx
            sig_end = start_idx
            if hasattr(node, "body") and node.body:
                first_body = node.body[0]
                body_start = getattr(first_body, "lineno", None)
                if body_start:
                    sig_end = body_start - 1

            for ds, de in decorator_lines:
                keep_ranges.append((ds, de))
            keep_ranges.append((sig_start, sig_end))

            # 类级常量赋值（跳过装饰器赋值）
            if isinstance(node, ast.ClassDef):
                for child in node.body:
                    if isinstance(child, (ast.Assign, ast.AnnAssign)):
                        cstart = getattr(child, "lineno", None)
                        cend = getattr(child, "end_lineno", None)
                        if cstart and cend:
                            cstart_idx = cstart - 1
                            if not _is_decorator_assignment(lines, cstart_idx, _def_lines):
                                keep_ranges.append((cstart_idx, cend))

        # 递归子节点
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
                _walk_code_nodes(child, is_class=isinstance(child, ast.ClassDef))

    for node in getattr(tree, "body", []):
        _walk_code_nodes(node, is_class=False)

    # 合并重叠区间并构建结果
    if not keep_ranges:
        return content

    keep_ranges.sort()
    merged = [keep_ranges[0]]
    for s, e in keep_ranges[1:]:
        if s <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))

    # 构建保留行列表（附带行号，用于 docstring 摘要注入）
    kept_lines: list[tuple[str, int]] = []  # (content, 1-based lineno)
    prev_end = 0
    for s, e in merged:
        if s > prev_end:
            omitted = s - prev_end
            if omitted >= 3:
                kept_lines.append((f"# ... ({omitted} lines) ...", 0))
        for line_idx in range(s, e):
            kept_lines.append((lines[line_idx], line_idx + 1))
        prev_end = e
    if prev_end < len(lines):
        omitted = len(lines) - prev_end
        if omitted >= 3:
            kept_lines.append((f"# ... ({omitted} lines) ...", 0))

    # 注入 docstring 摘要到签名行
    if docstring_summaries:
        for i, (line, lineno) in enumerate(kept_lines):
            stripped = line.strip()
            if stripped.startswith(("class ", "def ", "async def ")):
                summary = docstring_summaries.get(lineno)
                if summary:
                    kept_lines[i] = (line + f"  # {summary}", lineno)

    skeleton_no_summary = "\n".join(line for line, _ in kept_lines)
    # 压缩类型注解（缩短 verbose typing 名称）
    skeleton_no_summary = _compress_skeleton_types(skeleton_no_summary)
    skel_tokens_no_summary = count_tokens(skeleton_no_summary)
    orig_tokens = count_tokens(content)

    # 只在骨架确实更短时才注入 docstring 摘要
    if skel_tokens_no_summary < orig_tokens and docstring_summaries:
        skeleton_with_summary = _inject_summaries(kept_lines, docstring_summaries)
        skeleton_with_summary = _compress_skeleton_types("\n".join(line for line, _ in skeleton_with_summary))
        # 仅在注入摘要后仍然更短时才使用摘要版本
        if count_tokens(skeleton_with_summary) < orig_tokens:
            skeleton = skeleton_with_summary
        else:
            skeleton = skeleton_no_summary
    else:
        skeleton = skeleton_no_summary

    # 如果骨架比原文还长，返回原文（不产生负收益）
    if count_tokens(skeleton) >= orig_tokens:
        return content
    return skeleton


def _inject_summaries(kept_lines, docstring_summaries):
    """将 docstring 首行摘要注入到签名行末尾。"""
    result = list(kept_lines)
    for i, (line, lineno) in enumerate(result):
        stripped = line.strip()
        if stripped.startswith(("class ", "def ", "async def ")):
            summary = docstring_summaries.get(lineno)
            if summary:
                result[i] = (line + f"  # {summary}", lineno)
    return result


# 预编译的类型替换模式（避免每次调用都重新编译）
_TYPE_REPLACEMENTS: dict[str, str] = {
    r'\btyping\.Dict\b': 'dict',
    r'\btyping\.List\b': 'list',
    r'\btyping\.Set\b': 'set',
    r'\btyping\.Tuple\b': 'tuple',
    r'\btyping\.FrozenSet\b': 'frozenset',
}
_COMPILED_TYPE_REPLACEMENTS: list[tuple] = [
    (re.compile(pat), repl) for pat, repl in _TYPE_REPLACEMENTS.items()
]


def _compress_skeleton_types(skeleton: str) -> str:
    """压缩骨架中的类型注解，缩短 verbose typing 名称。"""
    for compiled, replacement in _COMPILED_TYPE_REPLACEMENTS:
        skeleton = compiled.sub(replacement, skeleton)
    # 去除 -> None 返回注解
    skeleton = re.sub(r'\s*-> None\b', '', skeleton)
    return skeleton


def _extract_js_ts_skeleton(content: str) -> str:
    """使用正则提取 JS/TS 文件骨架（仅保留结构签名）。"""
    # 只保留结构签名行，其他全部压缩
    patterns = [
        r'^import\s+',           # import 语句
        r'^export\s+',           # export 语句
        r'^class\s+\w+',         # class 定义
        r'^interface\s+\w+',     # interface 定义
        r'^type\s+\w+\s*=',      # type 定义
        r'^(?:async\s+)?(?:function|const|let|var)\s+\w+\s*\(',  # 函数/箭头函数
        r'^@\w+',                # 装饰器（@decorator）
    ]
    result_lines = []
    for line in content.split("\n"):
        stripped = line.strip()
        if not stripped:
            result_lines.append("")
            continue
        for pat in patterns:
            if re.match(pat, stripped):
                # 如果是 { 结尾，省略函数体
                if stripped.endswith("{"):
                    result_lines.append(stripped + " /* ... */ }")
                else:
                    result_lines.append(stripped)
                break
        else:
            # 非结构行：只保留每 N 行中的一个作为占位
            result_lines.append("    // ...")
    # 压缩连续 // ... 行
    compressed = []
    prev_omitted = False
    for line in result_lines:
        if line == "    // ...":
            if not prev_omitted:
                compressed.append(line)
            prev_omitted = True
        else:
            prev_omitted = False
            compressed.append(line)
    return "\n".join(compressed)


def _extract_c_like_skeleton(content: str, ext: str) -> str:
    """使用正则提取 C/Java/Go/Rust 风格文件的骨架。"""
    patterns = [
        r'^(#include\s+.*)',
        r'^(package\s+.*)',
        r'^(import\s+.*)',
        r'^(using\s+.*)',
        r'^((?:public\s+|private\s+|protected\s+|static\s+)*'
        r'(?:class|struct|enum|interface|func|fn)\s+\w+[^{]*\{)',
        r'^((?:(?:public|private|protected)\s+)*(?:static\s+)?'
        r'(?:\w+(?:<[^>]*>)?)\s+\w+\s*\([^)]*\)\s*\{)',
    ]
    result_lines = []
    for line in content.split("\n"):
        stripped = line.strip()
        if not stripped:
            result_lines.append("")
            continue
        for pat in patterns:
            if re.match(pat, stripped):
                if stripped.endswith("{"):
                    result_lines.append(stripped + " /* ... */ }")
                else:
                    result_lines.append(stripped)
                break
        else:
            result_lines.append(f"    // {stripped}")
    return "\n".join(result_lines)


# ── 符号索引 ────────────────────────────────────────────────────────────

def extract_symbol_index(content: str, ext: str) -> list[dict[str, Any]]:
    """提取代码文件的符号索引（函数名、类名、行号）。

    Returns:
        [{"name": str, "kind": str, "line": int, "signature": str}, ...]
    """
    if ext.lower() != ".py":
        return []

    try:
        tree = ast.parse(content)
    except SyntaxError:
        return []

    symbols = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            sig = f"class {node.name}"
            bases = []
            for b in node.bases:
                if isinstance(b, ast.Name):
                    bases.append(b.id)
            if bases:
                sig += f"({', '.join(bases)})"
            symbols.append({
                "name": node.name,
                "kind": "class",
                "line": node.lineno,
                "signature": sig,
            })
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            prefix = "async " if isinstance(node, ast.AsyncFunctionDef) else ""
            args = []
            for arg in node.args.args:
                args.append(arg.arg)
            sig = f"{prefix}def {node.name}({', '.join(args)})"
            if node.returns:
                sig += f" -> {ast.unparse(node.returns)}"
            symbols.append({
                "name": node.name,
                "kind": "function",
                "line": node.lineno,
                "signature": sig,
            })

    symbols.sort(key=lambda s: s["line"])
    return symbols


def format_symbol_index(symbols: list[dict[str, Any]], file_path: str) -> str:
    """将符号索引格式化为紧凑的字符串表示。"""
    if not symbols:
        return ""
    lines = [f"# {file_path} — 符号索引 ({len(symbols)} 个)"]
    for s in symbols:
        lines.append(f"  {s['kind'][0].upper()} L{s['line']}: {s['signature']}")
    return "\n".join(lines)


# ── 结构感知去重 ────────────────────────────────────────────────────────

def structural_dedup(file_paths: list[str | Path], threshold: float = 0.85) -> list[str | Path]:
    """基于代码结构相似度去重（超越 MD5 精确匹配）。

    对每个文件提取骨架签名，将骨架完全相同的文件视为重复，
    只保留第一个。这能捕获"同一个类模板生成的不同文件"这类 MD5 无法检测的重复。

    Args:
        file_paths: 文件路径列表
        threshold: 骨架相似度阈值（0-1），>= 此值视为重复

    Returns:
        去重后的文件路径列表
    """
    from claude_token_saver.prep import _clear_caches

    _clear_caches()

    # 骨架签名 → 第一个文件路径
    signature_map: dict[str, str | Path] = {}
    unique: list[str | Path] = []
    structural_dups = 0

    for fp in file_paths:
        fp = Path(fp)
        if not fp.is_file():
            unique.append(fp)
            continue

        try:
            content = fp.read_text(encoding="utf-8", errors="replace")
        except Exception:
            unique.append(fp)
            continue

        ext = fp.suffix.lower()
        skeleton = extract_skeleton(content, ext)

        # 使用骨架的 MD5 作为结构签名
        import hashlib
        sig = hashlib.md5(skeleton.encode()).hexdigest()

        if sig in signature_map:
            # 结构重复，跳过
            structural_dups += 1
            continue

        signature_map[sig] = fp
        unique.append(fp)

    if structural_dups > 0:
        import logging
        logging.getLogger("claude_token_saver.compressor").info(
            "结构去重: 移除了 %d 个结构重复文件", structural_dups
        )

    return unique


def group_by_structure(file_paths: list[str | Path]) -> list[list[str | Path]]:
    """按代码结构分组文件，返回结构相同的组。"""
    import hashlib
    from collections import defaultdict

    groups: dict[str, list[str | Path]] = defaultdict(list)

    for fp in file_paths:
        fp = Path(fp)
        if not fp.is_file():
            continue
        try:
            content = fp.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue

        ext = fp.suffix.lower()
        skeleton = extract_skeleton(content, ext)
        sig = hashlib.md5(skeleton.encode()).hexdigest()
        groups[sig].append(fp)

    return list(groups.values())

