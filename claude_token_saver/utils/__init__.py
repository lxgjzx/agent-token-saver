"""
Claude Code Token Saver - 工具函数库
"""
from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path


def get_file_hash(path: str | Path) -> str:
    """计算文件内容的 MD5 hash，用于去重判断。"""
    content = Path(path).read_bytes()
    return hashlib.md5(content).hexdigest()


def count_tokens(text: str, model: str = "claude-sonnet-4-20250514") -> int:
    """估算文本的 token 数量。"""
    try:
        import tiktoken
        enc = tiktoken.encoding_for_model("gpt-4o")  # Claude 兼容的 encoding
        return len(enc.encode(text))
    except Exception:
        # 回退：粗略估算（中英混合约 1.5 token/字符）
        return int(len(text) * 1.5)


def is_binary_file(path: str | Path) -> bool:
    """判断是否为二进制文件。"""
    path = Path(path)
    if not path.exists():
        return False
    try:
        with open(path, "rb") as f:
            chunk = f.read(8192)
            return b"\x00" in chunk
    except Exception:
        return True


def get_file_size(path: str | Path) -> int:
    """获取文件大小（字节）。"""
    return Path(path).stat().st_size


# ── 常见忽略目录/文件 ──
DEFAULT_IGNORE_DIRS = {
    ".git", ".svn", ".hg", "__pycache__", "node_modules",
    ".venv", "venv", "dist", "build", ".idea", ".vscode",
    ".gradle", "target", "bin", "obj",
}

DEFAULT_IGNORE_FILES = {
    ".DS_Store", "Thumbs.db", "*.pyc", "*.pyo",
    "*.class", "*.jar", "*.zip", "*.tar.gz",
}

BINARY_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".gif", ".ico", ".pdf",
    ".zip", ".tar", ".gz", ".bz2", ".7z", ".exe",
    ".dll", ".so", ".dylib", ".wasm", ".pyc", ".class",
    ".jar", ".apk", ".ipa", ".mp3", ".mp4", ".avi",
}


def should_ignore(path: str | Path, ignore_dirs: set[str] | None = None,
                  ignore_files: set[str] | None = None,
                  include_binary: bool = False) -> bool:
    """判断文件/目录是否应被忽略。"""
    path = Path(path)
    ignore_dirs = ignore_dirs or DEFAULT_IGNORE_DIRS
    ignore_files = ignore_files or DEFAULT_IGNORE_FILES

    if path.name in ignore_dirs:
        return True
    if any(path.match(f) for f in ignore_files):
        return True
    if path.suffix.lower() in BINARY_EXTENSIONS and not include_binary:
        return True
    if is_binary_file(path) and not include_binary:
        return True
    return False
