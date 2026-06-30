"""
Agent Token Saver - 对话 Diff 压缩

核心思路：多轮对话中，每轮的完整上下文往往高度重叠。
只发送与上一轮相比发生变化的增量信息，大幅减少 token。

压缩策略：
  - 文件变更 diff（工具调用结果）
  - 决策变更（用户新指令 vs 旧上下文）
  - 错误/异常高亮（失败的工具调用优先展示）
  - 摘要级上下文（超过 N 轮后只保留摘要）
"""
from __future__ import annotations

import difflib
import hashlib
import os
from dataclasses import dataclass, field
from typing import Any


@dataclass
class TurnDiff:
    """两轮对话之间的差异。"""
    turn_index: int
    # 新增/变更的文件
    changed_files: list[str] = field(default_factory=list)
    # 新增的命令/指令
    new_commands: list[str] = field(default_factory=list)
    # 错误/失败的工具调用
    errors: list[str] = field(default_factory=list)
    # 成功的关键结果
    key_results: list[str] = field(default_factory=list)
    # 完整内容（fallback，当 diff 太大时使用）
    full_content: str = ""
    # 标记：是否是摘要模式
    is_summary: bool = False
    # 原始 token 估算
    tokens_before: int = 0
    tokens_after: int = 0


@dataclass
class CompressedConversation:
    """压缩后的完整对话。"""
    session_id: str
    turns: list[TurnDiff]
    total_tokens_before: int = 0
    total_tokens_after: int = 0
    compression_ratio: float = 0.0


def compress_conversation_diff(
    turns: list[dict[str, Any]],
    max_tokens_per_turn: int = 3000,
    summary_after: int = 10,
) -> CompressedConversation:
    """压缩对话历史：生成每轮的增量 diff。

    Args:
        turns: 对话轮次列表 [{turn_index, type, content, tool_uses, ...}, ...]
        max_tokens_per_turn: 每轮最大 token 数
        summary_after: 超过此轮数后，旧轮次只保留摘要

    Returns:
        CompressedConversation
    """
    from claude_token_saver.utils import count_tokens

    total_before = sum(t.get("tokens", count_tokens(t.get("content", ""))) for t in turns)
    diffs: list[TurnDiff] = []

    for i, turn in enumerate(turns):
        turn_idx = turn.get("turn_index", i + 1)
        content = turn.get("content", "")
        tool_uses = turn.get("tool_uses", [])
        turn_type = turn.get("type", "user")

        # 提取工具调用涉及的文件
        changed_files = _extract_changed_files(tool_uses)

        # 提取错误
        errors = _extract_errors(tool_uses)

        # 提取关键结果
        key_results = _extract_key_results(tool_uses, max_results=3)

        # 提取新指令/命令
        new_commands = _extract_commands(turn, content)

        # 决定是否使用摘要模式
        is_summary = i < len(turns) - summary_after

        if is_summary and len(content) > 200:
            # 旧轮次：只保留摘要
            summary_content = _summarize_content(content, max_length=150)
            diff = TurnDiff(
                turn_index=turn_idx,
                changed_files=changed_files[:3],
                new_commands=new_commands[:2],
                errors=errors,
                key_results=key_results[:2],
                full_content=summary_content,
                is_summary=True,
                tokens_before=turn.get("tokens", count_tokens(content)),
            )
        else:
            # 最近几轮：保留完整内容
            diff = TurnDiff(
                turn_index=turn_idx,
                changed_files=changed_files,
                new_commands=new_commands,
                errors=errors,
                key_results=key_results,
                full_content=content[:max_tokens_per_turn * 4],  # 粗略截断
                is_summary=False,
                tokens_before=turn.get("tokens", count_tokens(content)),
            )

        diff.tokens_after = count_tokens(_format_turn_diff(diff))
        diffs.append(diff)

    total_after = sum(d.tokens_after for d in diffs)
    ratio = (1 - total_after / total_before) if total_before else 0.0

    return CompressedConversation(
        session_id=turns[0].get("session_id", "") if turns else "",
        turns=diffs,
        total_tokens_before=total_before,
        total_tokens_after=total_after,
        compression_ratio=round(ratio, 3),
    )


def format_compressed_conversation(cc: CompressedConversation) -> str:
    """将压缩后的对话格式化为可注入 prompt 的文本（紧凑格式）。"""
    parts: list[str] = []
    total_turns = len(cc.turns)
    saved = cc.total_tokens_before - cc.total_tokens_after
    parts.append(f"[对话 {total_turns}轮 节省{saved}tok {cc.compression_ratio:.0%}]\n")

    for diff in cc.turns:
        tag = "S" if diff.is_summary else "F"
        parts.append(f"T{diff.turn_index}{tag}")

        if diff.errors:
            parts.append(f"E:{';'.join(diff.errors[:1])}")

        if diff.changed_files:
            parts.append(f"C:{','.join(diff.changed_files[:2])}")

        if diff.new_commands:
            parts.append(f"Q:{'|'.join(diff.new_commands[:1])}")

        if diff.key_results:
            parts.append(f"R:{';'.join(diff.key_results[:1])}")

        if diff.full_content and not diff.is_summary:
            parts.append(diff.full_content[:300])

    return "\n".join(parts)


# ── 内部辅助 ────────────────────────────────────────────────────────────

def _extract_changed_files(tool_uses: list[dict]) -> list[str]:
    """从工具调用中提取涉及的文件路径。"""
    files: list[str] = []
    for t in tool_uses:
        inp = t.get("input", {})
        if isinstance(inp, dict):
            fp = inp.get("file_path", "")
            if fp:
                # 只取 basename，减少 token
                files.append(os.path.basename(str(fp)))
    return files


def _extract_errors(tool_uses: list[dict]) -> list[str]:
    """提取失败的工具调用。"""
    errors: list[str] = []
    for t in tool_uses:
        if t.get("is_error"):
            result = t.get("result_content", "")[:100]
            errors.append(f"{t.get('name', '?')}: {result}")
    return errors


def _extract_key_results(tool_uses: list[dict], max_results: int = 3) -> list[str]:
    """提取成功工具调用的关键结果。"""
    results: list[str] = []
    for t in tool_uses:
        if not t.get("is_error"):
            rc = t.get("result_content", "")[:80]
            if rc:
                results.append(f"{t.get('name', '?')}: {rc}")
    return results[:max_results]


def _extract_commands(turn: dict, content: str) -> list[str]:
    """提取用户指令中的关键命令。"""
    commands: list[str] = []
    lines = content.strip().split("\n")
    for line in lines[:5]:  # 只取前 5 行
        stripped = line.strip()
        if stripped and len(stripped) < 100:
            commands.append(stripped)
    return commands


def _summarize_content(content: str, max_length: int = 150) -> str:
    """将内容压缩为摘要（在句子边界截断）。"""
    if len(content) <= max_length:
        return content

    # 优先在句子边界截断（句号、问号、感叹号后）
    import re
    sentence_end = re.search(r'[。！？.!?]\s', content[:max_length])
    if sentence_end:
        cut = sentence_end.end()
        return content[:cut].rstrip() + "..."

    # 次优：在换行处截断
    last_newline = content[:max_length].rfind('\n')
    if last_newline > max_length // 2:
        return content[:last_newline].rstrip() + "\n..."

    # fallback：直接截断
    return content[:max_length].rstrip() + "..."


def _format_turn_diff(diff: TurnDiff) -> str:
    """将 TurnDiff 格式化为文本（用于 token 估算）。

    注意：summary 模式不包含 full_content，因为它会由 format_compressed_conversation
    单独处理，避免重复计算。
    """
    parts: list[str] = []

    if diff.errors:
        parts.append(f"错误: {'; '.join(diff.errors)}")

    if diff.changed_files:
        parts.append(f"文件: {', '.join(diff.changed_files)}")

    if diff.key_results:
        parts.append(f"结果: {'; '.join(diff.key_results)}")

    # 摘要模式：full_content 在 format_compressed_conversation 中单独处理
    # 非摘要模式：包含在格式化输出中
    if diff.full_content and not diff.is_summary:
        parts.append(diff.full_content)

    return "\n".join(parts)


def compute_diff_summary(
    old_text: str,
    new_text: str,
    context_lines: int = 3,
) -> str:
    """计算两个文本的 diff 摘要（用于文件内容变化）。

    Returns:
        diff 格式的字符串，只显示变更行及上下文
    """
    old_lines = old_text.split("\n")
    new_lines = new_text.split("\n")

    diff = list(difflib.unified_diff(
        old_lines, new_lines,
        lineterm="",
        n=context_lines,
    ))

    if not diff:
        return ""

    return "\n".join(diff[:50])  # 最多 50 行 diff
