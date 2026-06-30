"""
Agent Token Saver - 路径优化器

在输出中缩短文件路径，减少 token 消耗。

策略：
  - 使用相对于项目根目录的路径（去掉绝对路径前缀）
  - 对常见深层路径使用缩写映射
  - 在输出中保留完整路径的映射表，可逆
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


# 常见路径缩写映射（统一用 / 分隔符，匹配前路径会被统一为此格式）
PATH_ABBREVIATIONS: dict[str, str] = {
    # Windows 常见（/ 分隔符形式）
    "C:/Users/": "~u/",
    "D:/Projects/": "~p/",
    # Unix 常见
    "/home/user/": "~u/",
    "/Users/": "~u/",
    "/home/": "~h/",
    "/workspace/": "~w/",
    "/app/": "~a/",
    "/src/": "~s/",
    # 项目内
    "agent-token-saver/": "ats/",
}


def abbreviate_path(path: str, project_root: str | None = None) -> str:
    """缩短文件路径用于输出。

    优先使用相对于项目根目录的路径，
    然后应用常见缩写。

    Args:
        path: 完整文件路径
        project_root: 项目根目录（用于计算相对路径）

    Returns:
        缩短后的路径字符串
    """
    p = Path(path)

    # 如果指定了项目根目录，使用相对路径
    if project_root:
        try:
            rel = p.relative_to(Path(project_root))
            result = str(rel).replace("\\", "/")
            for full, abbr in sorted(PATH_ABBREVIATIONS.items(), key=lambda x: -len(x[0])):
                if result.startswith(full):
                    result = abbr + result[len(full):]
                    break
            return result
        except ValueError:
            pass

    # 应用缩写规则（先统一为 / 分隔符，确保缩写匹配）
    result = str(p).replace("\\", "/")
    for full, abbr in sorted(PATH_ABBREVIATIONS.items(), key=lambda x: -len(x[0])):
        if result.startswith(full):
            result = abbr + result[len(full):]
            break

    return result


def abbreviate_paths_in_text(text: str, project_root: str | None = None) -> str:
    """批量替换文本中的所有路径。"""
    import re

    # 匹配常见的绝对路径模式
    patterns = [
        r'[A-Z]:\\[^\s"\']+',  # Windows C:\...
        r'/[a-zA-Z][^\s"\']+',  # Unix /path/...
    ]

    result = text
    for pattern in patterns:
        matches = re.findall(pattern, result)
        for match in matches:
            abbreviated = abbreviate_path(match, project_root)
            if abbreviated != match:
                result = result.replace(match, abbreviated)

    return result


def build_path_abbreviation_map(paths: list[str], project_root: str | None = None) -> dict[str, str]:
    """构建路径缩写映射表。

    Returns:
        {完整路径: 缩写路径}
    """
    mapping: dict[str, str] = {}
    for p in paths:
        abbreviated = abbreviate_path(p, project_root)
        if abbreviated != p:
            mapping[p] = abbreviated
    return mapping


def format_path_mapping(mapping: dict[str, str]) -> str:
    """将路径映射表格式化为文本（供 Claude 参考）。"""
    if not mapping:
        return ""
    lines = ["# 路径缩写映射"]
    for full, abbr in mapping.items():
        lines.append(f"  {abbr} → {full}")
    return "\n".join(lines)
