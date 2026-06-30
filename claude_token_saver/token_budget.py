"""
Agent Token Saver - Token 预算优化器

核心思路：给定一个文件/内容列表和 token 预算，
自动优化内容分配，确保总输出不超过预算，
同时最大化信息密度。

策略：
  - 按优先级和大小排序，优先保留高价值内容
  - 使用二分查找精确截断，避免超出预算
  - 文件排序：小/重要文件优先，大文件最后
"""
from __future__ import annotations

from typing import Any


class TokenBudgetExceeded(Exception):
    """Token 预算不足。"""
    pass


def fit_to_budget(
    items: list[tuple[str, str, int]],
    budget: int,
) -> list[dict[str, Any]]:
    """将内容列表适配到 token 预算。

    Args:
        items: [(name, content, priority), ...] 列表
                priority 越高越优先保留（0-100）
        budget: 总 token 预算

    Returns:
        [{"name": str, "content": str, "tokens": int, "truncated": bool, "priority": int}, ...]
    """
    if budget <= 0:
        return []

    # 按 priority 降序、token 升序排序（高优先级 + 小文件优先）
    sorted_items = sorted(
        enumerate(items),
        key=lambda x: (-x[1][2], x[1][1] if x[1][1] else 0),
    )

    result: list[dict[str, Any]] = []
    remaining = budget

    # 预留 5% 给格式标记
    reserved = max(50, int(budget * 0.05))
    usable = budget - reserved

    for orig_idx, (name, content, priority) in sorted_items:
        content_tokens = _estimate_tokens(content)

        if content_tokens <= remaining - reserved:
            # 完整放入
            result.append({
                "name": name,
                "content": content,
                "tokens": content_tokens,
                "truncated": False,
                "priority": priority,
            })
            remaining -= content_tokens
        elif remaining > reserved:
            # 尝试截断放入
            truncated = _truncate_to_fit(content, remaining - reserved)
            trunc_tokens = _estimate_tokens(truncated)
            if trunc_tokens > 0:
                result.append({
                    "name": name,
                    "content": truncated,
                    "tokens": trunc_tokens,
                    "truncated": True,
                    "priority": priority,
                })
                remaining -= trunc_tokens
        # else: 预算耗尽，跳过

    # 恢复原始顺序
    order = {id(r): i for i, r in enumerate(result)}
    result.sort(key=lambda r: order.get(id(r), 0))

    return result


def prioritize_files(
    file_tokens: list[tuple[str, int]],
    strategy: str = "small_first",
) -> list[str]:
    """对文件进行优先级排序。

    Args:
        file_tokens: [(file_path, token_count), ...]
        strategy: "small_first" | "large_first" | "important_first"

    Returns:
        排序后的文件路径列表
    """
    if strategy == "small_first":
        return [fp for fp, _ in sorted(file_tokens, key=lambda x: x[1])]
    elif strategy == "large_first":
        return [fp for fp, _ in sorted(file_tokens, key=lambda x: -x[1])]
    else:
        return [fp for fp, _ in file_tokens]


def estimate_combined_tokens(contents: list[str]) -> int:
    """估算多个内容的总 token 数。"""
    return sum(_estimate_tokens(c) for c in contents)


def optimize_file_order(
    file_paths: list[str],
    file_sizes: dict[str, int],
    strategy: str = "mixed",
) -> list[str]:
    """优化文件输出顺序，减少上下文碎片。

    Args:
        file_paths: 文件路径列表
        file_sizes: {path: size_in_bytes} 映射
        strategy: "mixed" | "small_first" | "by_type"

    Returns:
        优化后的文件路径列表
    """
    if strategy == "small_first":
        return sorted(file_paths, key=lambda p: file_sizes.get(p, 0))
    elif strategy == "by_type":
        # 同类型文件放在一起
        from pathlib import Path
        by_ext: dict[str, list[str]] = {}
        for fp in file_paths:
            ext = Path(fp).suffix.lower()
            by_ext.setdefault(ext, []).append(fp)
        result = []
        for ext in sorted(by_ext.keys()):
            result.extend(sorted(by_ext[ext]))
        return result
    else:
        # mixed: 小文件和大文件交替，避免大文件堆在一起
        small = sorted(file_paths, key=lambda p: file_sizes.get(p, 0))
        large = list(reversed(small))
        mixed = []
        for i in range(max(len(small), len(large))):
            if i < len(small):
                mixed.append(small[i])
            if i < len(large):
                mixed.append(large[i])
        return mixed


# ── 内部辅助 ────────────────────────────────────────────────────────────

def _estimate_tokens(text: str) -> int:
    """估算文本的 token 数（复用 count_tokens 的缓存 encoding）。"""
    try:
        from claude_token_saver.utils import count_tokens
        return count_tokens(text)
    except Exception:
        return max(1, int(len(text) * 1.5))


def _truncate_to_fit(content: str, max_tokens: int) -> str:
    """将内容截断到适合 max_tokens。

    使用二分查找找到最大的前 keep_ratio + 后 keep_ratio 组合。
    """
    total = _estimate_tokens(content)
    if total <= max_tokens:
        return content

    lines = content.split("\n")
    if len(lines) <= 4:
        return content[:max_tokens * 4]

    head_ratio = 0.6
    tail_ratio = 0.3
    marker = "\n\n... [已截断以适配 token 预算] ...\n"
    marker_tokens = _estimate_tokens(marker)

    head_budget = int(max_tokens * head_ratio)
    tail_budget = max_tokens - head_budget - marker_tokens
    if tail_budget < 0:
        tail_budget = 0
        head_budget = max_tokens - marker_tokens

    head_lines = _find_lines_for_budget(lines, head_budget)
    tail_lines = _find_lines_for_budget(lines, tail_budget) if tail_budget > 0 else 0

    if head_lines + tail_lines >= len(lines):
        keep = max(1, int(len(lines) * 0.7))
        result = "\n".join(lines[:keep])
        result += f"\n\n... [已省略 {len(lines) - keep} 行，共 {len(lines)} 行] ...\n"
        return result

    head = lines[:head_lines]
    tail = lines[-tail_lines:] if tail_lines > 0 else []
    omitted = len(lines) - head_lines - tail_lines

    result = "\n".join(head)
    result += f"\n\n... [已省略 {omitted} 行（第 {head_lines + 1} - 第 {len(lines) - tail_lines} 行），共 {len(lines)} 行] ...\n\n"
    result += "\n".join(tail)
    return result


def _find_lines_for_budget(lines: list[str], token_budget: int) -> int:
    """二分查找：找到满足 token 预算的最大行数。"""
    if not lines or token_budget <= 0:
        return 0

    lo, hi = 0, len(lines)
    best = 0
    while lo < hi:
        mid = (lo + hi) // 2
        tokens = _estimate_tokens("\n".join(lines[:mid]))
        if tokens <= token_budget:
            best = mid
            lo = mid + 1
        else:
            hi = mid
    return best
