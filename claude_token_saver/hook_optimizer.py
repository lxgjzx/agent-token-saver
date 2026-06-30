"""
Agent Token Saver - Hook 输出优化器

最小化 hook 输出到 Claude Code 的 token 开销：
  - 使用短 JSON key 替代长描述
  - 压缩 reason 文本
  - 最小化 modified_input 的体积
  - 避免返回不必要的字段
"""
from __future__ import annotations

import json
from typing import Any


# 短 key 映射（输出时使用）
SHORT_KEYS: dict[str, str] = {
    "file_path": "p",
    "file_size": "sz",
    "max_file_size_bytes": "max",
    "modified_input": "in",
    "modified_output": "out",
    "decision": "d",
    "reason": "r",
    "tool_name": "t",
    "tool_input": "i",
    "tool_output": "o",
    "session_id": "sid",
    "matches": "m",
    "results": "r",
    "context": "ctx",
    "total_results": "tr",
}


def minimize_hook_output(result: dict[str, Any]) -> dict[str, Any]:
    """最小化 hook 输出字典的体积。

    策略：
    1. 移除空值和默认值
    2. 缩短字符串
    3. 使用紧凑表示
    """
    if not result:
        return result

    minimized: dict[str, Any] = {}

    # decision 总是保留
    decision = result.get("decision", "approve")
    minimized["d"] = decision

    # reason：只保留非空且非默认的
    reason = result.get("reason", "")
    if reason and reason != "":
        # 截断过长的 reason
        minimized["r"] = reason[:200] if len(reason) > 200 else reason

    # modified_input：只保留变更的字段
    modified_input = result.get("modified_input")
    if modified_input:
        minimized["in"] = _minimize_tool_input(modified_input)

    # modified_output：只保留变更的字段
    modified_output = result.get("modified_output")
    if modified_output:
        minimized["out"] = _minimize_tool_output(modified_output)

    return minimized


def _minimize_tool_input(tool_input: dict[str, Any]) -> dict[str, Any]:
    """最小化工具输入变更的体积。"""
    minimized: dict[str, Any] = {}

    # 只保留变更的字段（排除原始字段）
    original_keys = {"file_path", "pattern", "path", "content"}
    for key, value in tool_input.items():
        if key in original_keys:
            # 路径/模式字段可能需要缩短
            minimized[key[:3]] = value
        elif isinstance(value, list) and len(value) > 10:
            # 长列表截断
            minimized[key] = value[:10]
            minimized[key + "_t"] = len(value)  # 记录原始长度
        else:
            minimized[key] = value

    return minimized


def _minimize_tool_output(tool_output: dict[str, Any]) -> dict[str, Any]:
    """最小化工具输出变更的体积。"""
    minimized: dict[str, Any] = {}

    if "matches" in tool_output:
        matches = tool_output["matches"]
        many = isinstance(matches, list) and len(matches) > 20
        minimized["m"] = [
            _minimize_match(m, many=many) for m in (matches[:20] if isinstance(matches, list) else matches)
        ]
        if isinstance(matches, list) and len(matches) > 20:
            minimized["m_t"] = len(matches)

    if "results" in tool_output:
        results = tool_output["results"]
        minimized["r"] = [
            _minimize_result(r) for r in (results[:20] if isinstance(results, list) else results)
        ]
        if isinstance(results, list) and len(results) > 20:
            minimized["r_t"] = len(results)

    return minimized


def _minimize_match(match: dict[str, Any], many: bool = False) -> dict[str, Any]:
    """最小化单个匹配的体积。"""
    m: dict[str, Any] = {}
    if "file" in match:
        import os
        m["f"] = os.path.basename(str(match["file"]))
    if "line" in match:
        m["ln"] = match["line"]
    # 多匹配时跳过 content（仅保留路径+行号）
    if not many and "content" in match:
        m["c"] = match["content"][:100]
    if "context" in match and not many:
        ctx = match["context"]
        m["ctx"] = ctx[:5] if isinstance(ctx, list) else ctx
    return m


def _minimize_result(result: dict[str, Any]) -> dict[str, Any]:
    """最小化单个结果的体积。"""
    r: dict[str, Any] = {}
    for key in ("path", "is_dir"):
        if key in result:
            r[key[:3]] = result[key]
    return r


def compress_reason_text(reason: str, max_length: int = 150) -> str:
    """压缩 reason 文本，保留核心信息。"""
    if len(reason) <= max_length:
        return reason

    # 提取核心信息：文件大小 + 建议
    import re
    # 查找大小信息
    size_match = re.search(r'(\d+(?:\.\d+)?)\s*(KB|MB|GB)', reason)
    size_info = ""
    if size_match:
        size_info = f"{size_match.group(1)}{size_match.group(2)}"

    # 查找建议操作
    suggestion = ""
    if "--offset" in reason:
        suggestion = "建议使用 --offset/--limit"
    elif "过大" in reason:
        suggestion = "文件过大"

    if size_info and suggestion:
        return f"{suggestion} ({size_info})"
    elif size_info:
        return f"文件大小: {size_info}"
    elif suggestion:
        return suggestion

    # fallback：截断
    return reason[:max_length].rstrip() + "..."


def compute_hook_output_size(result: dict[str, Any]) -> int:
    """估算 hook 输出的 token 大小。"""
    try:
        from claude_token_saver.utils import count_tokens
        return count_tokens(json.dumps(result, ensure_ascii=False))
    except Exception:
        return len(json.dumps(result, ensure_ascii=False))
