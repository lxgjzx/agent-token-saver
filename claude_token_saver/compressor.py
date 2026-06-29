"""
Claude Code Token Saver - 压缩器模块

提供多级文件表示和代码骨架提取，大幅减少发送给 Claude 的 token 数量。

压缩层级：
  - skeleton: 仅提取函数/类签名、导入、类型定义（~5-10% 原始大小）
  - stripped: 去除注释和 docstring（~30-50% 原始大小）
  - full: 完整内容
  - block: 阻止读取（用于过大的文件）
"""
from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any


# ── 语言支持 ────────────────────────────────────────────────────────────

_SKELETON_LANGUAGES: set[str] = {".py", ".js", ".ts", ".java", ".c", ".cpp", ".h", ".go", ".rs"}
_STRIP_LANGUAGES: set[str] = {".rb", ".php", ".sh", ".bash", ".yaml", ".yml", ".toml", ".xml", ".html", ".css", ".sql"}


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


def _extract_python_skeleton(content: str) -> str:
    """使用 AST 提取 Python 文件骨架。"""
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return content

    lines = content.split("\n")
    keep_ranges: list[tuple[int, int]] = []

    for node in ast.walk(tree):
        if not isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                                 ast.AsyncFunctionDef, ast.Import, ast.ImportFrom,
                                 ast.Assign, ast.AnnAssign)):
            continue

        start = getattr(node, "lineno", None)
        end = getattr(node, "end_lineno", None)
        if start is None or end is None:
            continue

        start_idx = start - 1  # 转换为 0-based

        if isinstance(node, (ast.Import, ast.ImportFrom, ast.Assign, ast.AnnAssign)):
            # 顶级的 import 和常量，保留完整行
            keep_ranges.append((start_idx, end))
        elif isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            # 类/函数：只保留签名行（第一行）和可能的装饰器
            decorator_lines = []
            for deco in getattr(node, "decorator_list", []):
                deco_start = getattr(deco, "lineno", None)
                deco_end = getattr(deco, "end_lineno", None)
                if deco_start and deco_end:
                    decorator_lines.append((deco_start - 1, deco_end))

            # 提取签名行（可能跨多行）
            sig_start = start_idx
            sig_end = start_idx
            if hasattr(node, "body") and node.body:
                first_body = node.body[0]
                body_start = getattr(first_body, "lineno", None)
                if body_start:
                    sig_end = body_start - 1  # 签名到 body 开始前

            for ds, de in decorator_lines:
                keep_ranges.append((ds, de))
            keep_ranges.append((sig_start, sig_end))

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

    result_lines = list(lines)
    # 将不在保留区间内的行替换为省略标记
    for s, e in reversed(merged):
        # 保留区间内的内容不变，其余替换
        pass

    # 更简单的方法：只保留需要保留的行
    kept_lines = []
    prev_end = 0
    for s, e in merged:
        if s > prev_end:
            omitted = s - prev_end
            kept_lines.append(f"    # ... [省略 {omitted} 行] ...")
        kept_lines.extend(lines[s:e])
        prev_end = e
    if prev_end < len(lines):
        omitted = len(lines) - prev_end
        kept_lines.append(f"    # ... [省略 {omitted} 行] ...")

    return "\n".join(kept_lines)


def _extract_js_ts_skeleton(content: str) -> str:
    """使用正则提取 JS/TS 文件骨架。"""
    # 匹配 import/export、class/function/const 声明（第一行）
    patterns = [
        r'^(import\s+.*?;)\s*$',
        r'^(export\s+.*?;)\s*$',
        r'^(class\s+\w+[^{]*\{)',
        r'^((?:async\s+)?(?:function|const|let|var)\s+\w+\s*\([^)]*\)\s*(?::\s*\S+)?\s*\{)',
        r'^(interface\s+\w+[^{]*\{)',
        r'^(type\s+\w+\s*=)',
        r'^(export\s+(?:default\s+)?(?:class|function|const|let|var)\s+\w+)',
    ]
    result_lines = []
    for line in content.split("\n"):
        stripped = line.strip()
        if not stripped:
            result_lines.append("")
            continue
        for pat in patterns:
            if re.match(pat, stripped):
                # 如果是 { 结尾但不是 }，替换函数体
                if stripped.endswith("{"):
                    result_lines.append(stripped + " /* ... */ }")
                else:
                    result_lines.append(stripped)
                break
        else:
            result_lines.append(f"    // {stripped}")
    return "\n".join(result_lines)


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
