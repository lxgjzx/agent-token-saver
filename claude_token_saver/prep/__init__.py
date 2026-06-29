"""
Claude Code Token Saver - 预处理模块
负责在发送给 Claude 前精简输入内容。
"""
from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Optional

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


def strip_comments(content: str, ext: str) -> str:
    """去除文件中的注释内容。"""
    pattern = _COMMENT_PATTERNS.get(ext.lower())
    if pattern:
        return pattern.sub("", content)
    return content


def strip_python_docstrings(content: str) -> str:
    """去除 Python 文件的文档字符串（保留代码逻辑注释）。"""
    try:
        tree = ast.parse(content)
        lines = content.split("\n")
        docstring_ranges: list[tuple[int, int]] = []

        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef, ast.Module)):
                continue
            if not hasattr(node, "body") or not node.body:
                continue
            first_stmt = node.body[0]
            if not isinstance(first_stmt, ast.Expr):
                continue
            # 检查是否为字符串字面量（docstring）
            if not (hasattr(first_stmt, "value") and isinstance(first_stmt.value, ast.Constant)
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


# ── 文件精简 ──

def smart_truncate(content: str, max_tokens: int) -> str:
    """智能截断文件内容，优先保留头尾。"""
    lines = content.split("\n")
    if count_tokens(content) <= max_tokens:
        return content

    # 保留前 60% + 后 30%，中间省略
    head_count = int(len(lines) * 0.6)
    tail_count = int(len(lines) * 0.3)
    head = lines[:head_count]
    tail = lines[-tail_count:]
    omitted = len(lines) - head_count - tail_count

    result = "\n".join(head)
    result += f"\n\n... [已省略 {omitted} 行，共 {len(lines)} 行] ...\n\n"
    result += "\n".join(tail)
    return result


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


# ── 主入口 ──

def process_files(
    file_paths: list[str | Path],
    do_strip_comments: bool = True,
    do_strip_docstrings: bool = False,
    max_file_tokens: int = 50_000,
    dedup: bool = True,
    include_binary: bool = False,
) -> dict:
    """
    处理文件列表，返回精简后的结果。

    Returns:
        {
            "files": [...],
            "total_tokens_before": int,
            "total_tokens_after": int,
            "savings_pct": float,
            "skipped": [...],
            "duplicates_removed": int,
        }
    """
    if dedup:
        file_paths = deduplicate_files(list(file_paths))
        dup_count = len(file_paths)  # will be updated below

    results = []
    total_before = 0
    total_after = 0
    skipped = []
    dup_removed = 0
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

            content = fp.read_text(encoding="utf-8", errors="replace")
            tokens_before = count_tokens(content)
            total_before += tokens_before

            # 去重
            content_hash = get_file_hash(fp)
            if content_hash in seen_hashes:
                dup_removed += 1
                continue
            seen_hashes.add(content_hash)

            # 去除注释
            if do_strip_comments and fp.suffix.lower() in _COMMENT_PATTERNS:
                content = strip_comments(content, fp.suffix.lower())
            elif do_strip_comments:
                content = strip_comments(content, fp.suffix.lower())

            # 去除 Python docstring
            if do_strip_docstrings and fp.suffix == ".py":
                content = strip_python_docstrings(content)

            # 智能截断
            content = smart_truncate(content, max_file_tokens)

            tokens_after = count_tokens(content)
            total_after += tokens_after

            results.append({
                "path": str(fp),
                "tokens_before": tokens_before,
                "tokens_after": tokens_after,
                "savings": tokens_before - tokens_after,
                "content": content,
            })
        except Exception as e:
            skipped.append(f"{fp} ({e})")

    savings_pct = ((total_before - total_after) / total_before * 100) if total_before else 0

    return {
        "files": results,
        "total_tokens_before": total_before,
        "total_tokens_after": total_after,
        "savings_pct": round(savings_pct, 1),
        "skipped": skipped,
        "duplicates_removed": dup_removed,
    }


def format_processed_output(result: dict, format: str = "markdown") -> str:
    """将处理结果格式化为可发送给 Claude 的文本。"""
    if format == "markdown":
        parts = []
        parts.append(f"# 文件内容（精简后，共 {len(result['files'])} 个文件）\n")
        for f in result["files"]:
            ext = Path(f["path"]).suffix
            parts.append(f"## `{f['path']}` ({f['tokens_after']} tokens)\n```{ext.lstrip('.')}\n{f['content']}\n```\n")
        return "\n".join(parts)
    elif format == "plain":
        parts = []
        for f in result["files"]:
            parts.append(f"=== {f['path']} ===\n{f['content']}\n")
        return "\n".join(parts)
    elif format == "json":
        import json
        return json.dumps(result, ensure_ascii=False, indent=2)
    else:
        raise ValueError(f"Unknown format: {format}")
