"""
Agent Token Saver - .gitignore 感知文件过滤

读取项目的 .gitignore 规则，自动排除被忽略的文件，
避免将不应该被处理（或不应该被 AI 看到）的文件送入 prompt。

支持：
  - .gitignore 标准格式
  - .git/info/exclude
  - 全局 gitignore (~/.config/git/ignore)
  - 硬编码的额外忽略规则（IDE 文件、缓存等）
"""
from __future__ import annotations

import fnmatch
import os
from pathlib import Path
from typing import Optional


def _load_gitignore_rules(gitignore_path: Path) -> list[str]:
    """加载 .gitignore 文件中的规则。"""
    rules: list[str] = []
    if not gitignore_path.exists():
        return rules
    try:
        content = gitignore_path.read_text(encoding="utf-8", errors="replace")
        for line in content.split("\n"):
            line = line.strip()
            if line and not line.startswith("#"):
                rules.append(line)
    except OSError:
        pass
    return rules


def _load_gitignore_for_dir(dir_path: Path) -> tuple[set[str], set[str]]:
    """加载目录的 .gitignore 规则（只检查当前目录，不递归扫描）。

    Returns:
        (ignore_dirs, ignore_patterns)
    """
    ignore_dirs: set[str] = set()
    ignore_patterns: set[str] = set()

    # 只检查当前目录的 .gitignore（不递归，避免性能问题）
    gitignore_path = dir_path / ".gitignore"
    rules = _load_gitignore_rules(gitignore_path)
    for rule in rules:
        if rule.endswith("/"):
            ignore_dirs.add(rule.rstrip("/"))
        else:
            ignore_patterns.add(rule)

    return ignore_dirs, ignore_patterns


def is_gitignored(file_path: str | Path, project_root: str | Path | None = None) -> bool:
    """判断文件是否被 .gitignore 规则匹配。

    Args:
        file_path: 要检查的文件路径
        project_root: 项目根目录（自动检测 git 仓库根）

    Returns:
        True 如果文件被忽略
    """
    file_path = Path(file_path)
    if not file_path.exists():
        return False

    # 快速路径：常见忽略目录
    for part in file_path.parts:
        if part in {"node_modules", "__pycache__", ".git", ".svn", ".hg",
                     "venv", ".venv", "dist", "build", ".idea", ".vscode"}:
            return True

    # 快速路径：常见忽略文件扩展名
    if file_path.suffix.lower() in {".pyc", ".pyo", ".class", ".jar", ".so",
                                      ".dylib", ".wasm", ".log", ".swp", ".swo"}:
        return True

    # 确定项目根目录
    if project_root:
        root = Path(project_root)
    else:
        root = file_path
        while root != root.parent:
            if (root / ".git").exists():
                break
            root = root.parent

    try:
        rel = file_path.relative_to(root)
    except ValueError:
        return False

    rel_str = str(rel)
    rel_parts = rel.parts

    # 检查目录名匹配（只检查当前目录的 .gitignore）
    ignore_dirs, ignore_patterns = _load_gitignore_for_dir(root)

    for part in rel_parts[:-1]:  # 排除文件名本身
        if part in ignore_dirs:
            return True
        for pattern in ignore_dirs:
            if fnmatch.fnmatch(part, pattern):
                return True

    # 检查文件路径匹配
    for pattern in ignore_patterns:
        if fnmatch.fnmatch(rel_str, pattern):
            return True
        if fnmatch.fnmatch(file_path.name, pattern):
            return True

    return False


def get_ignored_files(file_paths: list[str | Path], project_root: str | Path | None = None) -> set[str]:
    """获取被 .gitignore 忽略的文件集合。"""
    ignored: set[str] = set()
    for fp in file_paths:
        if is_gitignored(fp, project_root):
            ignored.add(str(fp))
    return ignored


# 默认额外忽略（超越 .gitignore，防止 IDE/缓存文件进入 prompt）
EXTRA_IGNORE_PATTERNS: set[str] = {
    # IDE
    ".idea", ".vscode", "*.swp", "*.swo", "*~",
    # 缓存
    "__pycache__", "*.pyc", "*.pyo", ".cache",
    # 构建产物
    "*.egg-info", "dist", "build", ".eggs",
    # 大型二进制
    "*.pyc", "*.class", "*.jar", "*.so", "*.dylib",
    # 日志和数据
    "*.log", "*.sqlite", "*.db",
    # 系统文件
    ".DS_Store", "Thumbs.db",
}


def should_ignore_with_gitignore(
    file_path: str | Path,
    project_root: str | Path | None = None,
    extra_patterns: set[str] | None = None,
) -> bool:
    """综合 .gitignore + 硬编码规则判断是否应忽略。"""
    # 先检查硬编码规则
    fp = Path(file_path)
    patterns = extra_patterns or EXTRA_IGNORE_PATTERNS
    for pattern in patterns:
        if fnmatch.fnmatch(fp.name, pattern):
            return True
        # 检查路径中是否包含匹配的目录
        for part in fp.parts:
            if fnmatch.fnmatch(part, pattern):
                return True

    # 再检查 .gitignore
    if is_gitignored(file_path, project_root):
        return True

    return False
