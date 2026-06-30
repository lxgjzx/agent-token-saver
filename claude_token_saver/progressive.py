"""
Agent Token Saver - 渐进式披露（Progressive Disclosure）

核心思路：不一次性返回所有文件内容，而是先返回目录索引，
Claude 按需请求具体文件。

使用场景：
  - 大项目（>30 个文件）的文件列表
  - 用户说 "看看这个项目" 时
  - 作为 prep files --mode index 的子命令

流程：
  1. 扫描目录，收集所有文件
  2. 计算每个文件的 token 大小（不读取内容）
  3. 返回紧凑的目录索引
  4. Claude 按需调用 Read 读取具体文件
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from claude_token_saver.utils import get_file_size, should_ignore
from claude_token_saver.prep import estimate_tokens_from_size


@dataclass
class FileIndexEntry:
    """目录索引中的文件条目。"""
    path: str
    size_bytes: int
    estimated_tokens: int
    ext: str
    is_binary: bool = False
    relative_path: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "relative": self.relative_path,
            "size_kb": round(self.size_bytes / 1024, 1),
            "tokens": self.estimated_tokens,
            "ext": self.ext,
            "is_binary": self.is_binary,
        }


@dataclass
class DirectoryIndex:
    """目录索引结果。"""
    root: str
    total_files: int
    total_tokens: int
    files: list[FileIndexEntry]
    by_directory: dict[str, list[FileIndexEntry]]
    largest_files: list[FileIndexEntry]


def build_directory_index(
    paths: list[str | Path],
    max_files: int = 200,
    include_binary: bool = False,
) -> DirectoryIndex:
    """构建目录索引（不读取文件内容）。

    Args:
        paths: 文件或目录路径列表
        max_files: 最大索引文件数
        include_binary: 是否包含二进制文件

    Returns:
        DirectoryIndex
    """
    all_files: list[FileIndexEntry] = []
    seen: set[str] = set()

    for p in paths:
        p = Path(p)
        if p.is_dir():
            for f in sorted(p.rglob("*")):
                if not f.is_file():
                    continue
                if should_ignore(f, include_binary=include_binary):
                    continue
                fp_str = str(f.resolve())
                if fp_str in seen:
                    continue
                seen.add(fp_str)
                entry = _make_entry(f, p)
                if entry:
                    all_files.append(entry)
        elif p.is_file() and not should_ignore(p, include_binary=include_binary):
            fp_str = str(p.resolve())
            if fp_str not in seen:
                seen.add(fp_str)
                entry = _make_entry(p, p.parent)
                if entry:
                    all_files.append(entry)

    # 限制文件数
    if len(all_files) > max_files:
        all_files = sorted(all_files, key=lambda e: e.estimated_tokens, reverse=True)[:max_files]

    # 按目录分组
    by_directory: dict[str, list[FileIndexEntry]] = {}
    for entry in all_files:
        dir_path = str(Path(entry.relative_path).parent)
        by_directory.setdefault(dir_path, []).append(entry)

    # 最大的文件（top 10）
    largest = sorted(all_files, key=lambda e: e.estimated_tokens, reverse=True)[:10]

    total_tokens = sum(e.estimated_tokens for e in all_files)

    return DirectoryIndex(
        root=str(Path(paths[0]).resolve()) if paths else "",
        total_files=len(all_files),
        total_tokens=total_tokens,
        files=all_files,
        by_directory=by_directory,
        largest_files=largest,
    )


def format_index_markdown(index: DirectoryIndex, compact: bool = False) -> str:
    """将目录索引格式化为 Markdown（供 Claude 消费）。

    Args:
        compact: 如果 True，使用最紧凑格式（仅文件列表，无目录分组）
    """
    if compact:
        # 最紧凑模式：仅文件列表 + token 估算，~60% token
        parts = [f"# 项目索引（{index.total_files} 文件，~{index.total_tokens:,} tok）\n"]
        for e in sorted(index.files, key=lambda e: -e.estimated_tokens):
            parts.append(f"`{e.relative_path}` ~{e.estimated_tokens:,}tok")
        return "\n".join(parts)

    parts = []
    parts.append(f"# 项目目录索引（{index.total_files} 个文件，约 {index.total_tokens:,} tokens）\n")
    parts.append(f"根目录: `{index.root}`\n")

    # 按目录分组展示
    for dir_path in sorted(index.by_directory.keys()):
        entries = index.by_directory[dir_path]
        dir_label = dir_path if dir_path else "(root)"
        parts.append(f"## {dir_label}/ ({len(entries)} 个文件)\n")

        # 按大小排序
        sorted_entries = sorted(entries, key=lambda e: e.estimated_tokens, reverse=True)
        for e in sorted_entries:
            marker = "⚠️ " if e.estimated_tokens > 50_000 else ""
            size_kb = round(e.size_bytes / 1024, 1)
            parts.append(
                f"- {marker}`{e.relative_path}` "
                f"({size_kb} KB, ~{e.estimated_tokens:,} tokens)"
            )
        parts.append("")

    # 大文件提醒
    if index.largest_files:
        parts.append("## 最大的文件（建议使用 --offset/--limit）\n")
        for e in index.largest_files:
            size_kb = round(e.size_bytes / 1024, 1)
            parts.append(f"- `{e.relative_path}`: {size_kb} KB, ~{e.estimated_tokens:,} tokens")
        parts.append("")

    # 按扩展名统计
    ext_stats: dict[str, int] = {}
    for e in index.files:
        ext = e.ext or "(none)"
        ext_stats[ext] = ext_stats.get(ext, 0) + 1

    parts.append("## 文件类型分布\n")
    for ext, count in sorted(ext_stats.items(), key=lambda x: x[1], reverse=True):
        parts.append(f"- {ext}: {count} 个文件")
    parts.append("")

    return "\n".join(parts)


def format_index_json(index: DirectoryIndex) -> str:
    """将目录索引序列化为 JSON。"""
    import json
    return json.dumps({
        "root": index.root,
        "total_files": index.total_files,
        "total_tokens": index.total_tokens,
        "files": [e.to_dict() for e in index.files],
        "by_directory": {
            d: [e.to_dict() for e in entries]
            for d, entries in index.by_directory.items()
        },
    }, ensure_ascii=False, indent=2)


def _make_entry(file_path: Path, root: Path) -> FileIndexEntry | None:
    """从文件路径创建索引条目。"""
    try:
        size = file_path.stat().st_size
    except OSError:
        return None

    ext = file_path.suffix.lower()
    tokens = estimate_tokens_from_size(size, ext)

    try:
        rel = str(file_path.relative_to(root))
    except ValueError:
        rel = file_path.name

    return FileIndexEntry(
        path=str(file_path),
        size_bytes=size,
        estimated_tokens=tokens,
        ext=ext,
        relative_path=rel,
    )
