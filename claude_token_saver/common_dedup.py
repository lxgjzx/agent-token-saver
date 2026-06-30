"""
Agent Token Saver - 常见文件组去重

核心思路：项目中某些常见文件（如 __init__.py, conftest.py, setup.py 等）
往往具有高度相似的结构。检测这些模式并在处理时只保留代表性文件，
避免向 Claude 发送大量结构相同的内容。

策略：
  - 空 __init__.py 只保留一个代表
  - conftest.py 通常共享大量样板代码
  - 配置文件（pyproject.toml, setup.py 等）通常只展示一个
  - 类型存根文件（.pyi）通常与 .py 文件对应
"""
from __future__ import annotations

import hashlib
from pathlib import Path


# 常见文件模式及其处理策略
# strategy: "keep_first" = 每组只保留第一个, "keep_unique" = 只保留内容不同的,
#            "reference" = 只输出引用不输出内容
COMMON_FILE_PATTERNS: dict[str, dict] = {
    "__init__.py": {
        "description": "Python 包初始化文件，通常为空或仅含导入",
        "strategy": "keep_first_per_dir",
        "min_size_for_keep": 500,  # 小于此字节的文件视为简单初始化文件
    },
    "conftest.py": {
        "description": "pytest 配置文件，常共享样板代码",
        "strategy": "keep_first_per_dir",
        "min_size_for_keep": 2000,
    },
    "setup.py": {
        "description": "包安装脚本",
        "strategy": "keep_unique",
    },
    "pyproject.toml": {
        "description": "项目配置文件",
        "strategy": "keep_unique",
    },
    "setup.cfg": {
        "description": "setup 配置",
        "strategy": "keep_unique",
    },
    "Makefile": {
        "description": "构建脚本",
        "strategy": "keep_unique",
    },
    "Dockerfile": {
        "description": "Docker 配置",
        "strategy": "keep_unique",
    },
    ".gitignore": {
        "description": "Git 忽略规则",
        "strategy": "keep_unique",
    },
    "README.md": {
        "description": "项目说明文档",
        "strategy": "keep_unique",
    },
    "README.rst": {
        "description": "项目说明文档",
        "strategy": "keep_unique",
    },
    "MANIFEST.in": {
        "description": "打包清单",
        "strategy": "keep_unique",
    },
    "tox.ini": {
        "description": "tox 测试配置",
        "strategy": "keep_unique",
    },
    ".flake8": {
        "description": "flake8 配置",
        "strategy": "keep_unique",
    },
    "mypy.ini": {
        "description": "mypy 配置",
        "strategy": "keep_unique",
    },
    ".pre-commit-config.yaml": {
        "description": "pre-commit 配置",
        "strategy": "keep_unique",
    },
}


def detect_common_pattern(path: Path) -> dict | None:
    """检测文件是否匹配常见模式。

    Returns:
        模式配置字典，如果不匹配则返回 None
    """
    name = path.name
    if name in COMMON_FILE_PATTERNS:
        return COMMON_FILE_PATTERNS[name]
    return None


def get_common_dedup_suggestion(
    pattern_name: str,
    files: list[Path],
) -> str:
    """获取常见文件组的去重建议。

    Args:
        pattern_name: 模式名称（文件名）
        files: 匹配该模式的文件列表

    Returns:
        去重建议文本
    """
    pattern = COMMON_FILE_PATTERNS.get(pattern_name)
    if not pattern:
        return ""

    count = len(files)
    if count <= 1:
        return ""

    desc = pattern.get("description", pattern_name)
    strategy = pattern.get("strategy", "keep_unique")

    if strategy == "keep_first_per_dir":
        dirs = {str(f.parent) for f in files}
        return (
            f"发现 {count} 个 {pattern_name}（{desc}），"
            f"分布在 {len(dirs)} 个目录中。"
            f"建议每个目录只保留一个，其余可跳过。"
        )
    elif strategy == "keep_unique":
        # 检查内容是否相同
        try:
            hashes = set()
            for f in files:
                if f.is_file():
                    content = f.read_bytes()
                    hashes.add(hashlib.md5(content).hexdigest())
            unique_count = len(hashes)
            if unique_count < count:
                return (
                    f"发现 {count} 个 {pattern_name}，"
                    f"其中 {count - unique_count} 个内容相同或高度相似。"
                    f"建议只保留 {unique_count} 个独特的版本。"
                )
        except Exception:
            pass
        return f"发现 {count} 个 {pattern_name}（{desc}），建议只保留一个。"

    return ""


def filter_common_duplicates(
    files: list[str | Path],
) -> tuple[list[str | Path], list[str]]:
    """过滤常见结构重复文件。

    - __init__.py: 每个目录只保留第一个（且只保留有实际内容的）
    - 其他 keep_unique 模式：只保留内容不同的

    Args:
        files: 文件路径列表

    Returns:
        (去重后的文件列表, 跳过原因列表)
    """
    result: list[str | Path] = []
    skipped: list[str] = []

    # 按文件名分组
    by_name: dict[str, list[Path]] = {}
    for fp in files:
        fp_path = Path(fp)
        name = fp_path.name
        by_name.setdefault(name, []).append(fp_path)

    for name, group in by_name.items():
        pattern = COMMON_FILE_PATTERNS.get(name)
        if not pattern:
            # 不匹配常见模式，全部保留
            result.extend(group)
            continue

        strategy = pattern.get("strategy", "keep_unique")

        if strategy == "keep_first_per_dir":
            # 每个目录只保留第一个
            seen_dirs: set[str] = set()
            for fp in sorted(group, key=lambda p: str(p.parent)):
                dir_key = str(fp.parent.resolve())
                if dir_key in seen_dirs:
                    skipped.append(
                        f"{fp}（同目录已有 {name}，跳过结构重复）"
                    )
                    continue
                seen_dirs.add(dir_key)
                result.append(fp)

        elif strategy == "keep_unique":
            # 只保留内容不同的
            seen_hashes: set[str] = set()
            for fp in sorted(group, key=lambda p: str(p)):
                try:
                    content = fp.read_bytes()
                    h = hashlib.md5(content).hexdigest()
                    if h in seen_hashes:
                        skipped.append(
                            f"{fp}（内容与其他 {name} 相同，跳过）"
                        )
                        continue
                    seen_hashes.add(h)
                    result.append(fp)
                except Exception:
                    result.append(fp)
        else:
            result.extend(group)

    return result, skipped


def dedup_conftest_always(files: list[str | Path]) -> tuple[list[str | Path], list[str]]:
    """智能去重 conftest.py：仅保留顶层 conftest，子目录相同文件跳过。

    检测子目录中的 conftest.py 是否与父目录内容相同，相同则跳过。
    这对 pytest 项目特别有效，通常子目录 conftest.py 是样板代码复制。

    Args:
        files: 文件路径列表

    Returns:
        (去重后的文件列表, 跳过原因列表)
    """
    from collections import defaultdict

    files_by_name: dict[str, list[Path]] = defaultdict(list)
    for fp in files:
        files_by_name[Path(fp).name].append(Path(fp))

    to_remove: set[str] = set()
    for name, group in files_by_name.items():
        if name != "conftest.py":
            continue
        # 按目录深度排序（浅的优先保留）
        sorted_group = sorted(group, key=lambda p: len(p.parts))
        if len(sorted_group) <= 1:
            continue
        # 读取内容比较
        contents: dict[Path, str] = {}
        for fp in sorted_group:
            try:
                contents[fp] = fp.read_text(encoding="utf-8", errors="replace")
            except Exception:
                pass
        # 保留第一个，其他如果内容相同则标记为重复
        keeper = sorted_group[0]
        keeper_content = contents.get(keeper, "")
        for fp in sorted_group[1:]:
            if fp in contents and contents[fp] == keeper_content:
                to_remove.add(str(fp))

    remaining = [f for f in files if str(f) not in to_remove]
    skipped = [f for f in files if str(f) in to_remove]
    return remaining, skipped


def get_common_pattern_summary(files: list[str | Path]) -> list[str]:
    """获取常见文件模式的摘要。

    Returns:
        [建议文本, ...]
    """
    from collections import defaultdict

    by_name: dict[str, list[Path]] = defaultdict(list)
    for fp in files:
        fp_path = Path(fp)
        name = fp_path.name
        if name in COMMON_FILE_PATTERNS:
            by_name[name].append(fp_path)

    suggestions = []
    for name, group in sorted(by_name.items()):
        suggestion = get_common_dedup_suggestion(name, group)
        if suggestion:
            suggestions.append(suggestion)

    return suggestions
