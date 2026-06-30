"""
Agent 适配器注册表与工厂。

提供:
  - Agent 自动检测
  - 适配器注册/获取
  - 事件归一化/反归一化
  - CLI 命令支持
"""
from __future__ import annotations

from typing import Optional

from claude_token_saver.agents.base import (
    AgentID, HookDecision, HookEvent,
)

# ── 导入所有适配器（触发 @_register 装饰器 → 注册到 _ADAPTER_REGISTRY） ─
import claude_token_saver.agents.claude_code  # noqa: F401
import claude_token_saver.agents.codex        # noqa: F401
import claude_token_saver.agents.openclaw     # noqa: F401
import claude_token_saver.agents.cursor       # noqa: F401
import claude_token_saver.agents.aider        # noqa: F401
import claude_token_saver.agents.windsurf     # noqa: F401
# continue 是关键字，用 importlib
import importlib
importlib.import_module("claude_token_saver.agents.continue")  # noqa: F401


# ═══════════════════════════════════════════════════════════════════════════════
# CLI 共享常量
# ═══════════════════════════════════════════════════════════════════════════════

_AGENT_CHOICES = [a.value for a in AgentID]


def resolve_adapter(agent_name: str | None = None):
    """根据名称或自动检测返回适配器实例。找不到时回退到 Claude Code。

    供 cli.py / cli_agents.py / cli_hooks.py 共享。
    注意：此函数不依赖 Click，raise ValueError 供调用方处理。
    """
    if agent_name:
        aid = AgentID(agent_name.lower())
        adapter = get_adapter(aid)
        if not adapter:
            raise ValueError(f"Agent {agent_name} 未注册")
        return adapter
    return detect_agent() or get_adapter(AgentID.CLAUDE_CODE)


# ═══════════════════════════════════════════════════════════════════════════════
# 公共 API（读取 base.py 中的全局注册表）
# ═══════════════════════════════════════════════════════════════════════════════

def get_registered_ids() -> list[AgentID]:
    """返回所有已注册的 Agent 标识符。"""
    from claude_token_saver.agents.base import _ADAPTER_REGISTRY
    return list(_ADAPTER_REGISTRY.keys())


def get_adapter(agent_id: AgentID) -> Optional[AgentAdapter]:
    """根据 AgentID 获取适配器实例。"""
    from claude_token_saver.agents.base import _ADAPTER_REGISTRY
    cls = _ADAPTER_REGISTRY.get(agent_id)
    return cls() if cls else None


def detect_agent() -> Optional[AgentAdapter]:
    """按注册顺序检测当前运行环境，返回首个匹配的适配器。"""
    from claude_token_saver.agents.base import _ADAPTER_REGISTRY
    for cls in _ADAPTER_REGISTRY.values():
        try:
            adapter = cls()
            if adapter.detect():
                return adapter
        except Exception:
            continue
    return None


def get_all_adapters() -> list[AgentAdapter]:
    """返回所有已注册的适配器实例。"""
    from claude_token_saver.agents.base import _ADAPTER_REGISTRY
    return [cls() for cls in _ADAPTER_REGISTRY.values()]


# ── 事件归一化/反归一化 ───────────────────────────────────────────────

def normalize_event(raw: dict, adapter: AgentAdapter) -> HookEvent:
    """使用指定适配器将原始事件转换为统一格式。"""
    return adapter.parse_inbound_event(raw)


def denormalize_decision(
    decision: HookDecision,
    adapter: AgentAdapter,
    raw_event: Optional[dict] = None,
) -> dict:
    """使用指定适配器将统一决策转换为 Agent 期望格式。"""
    return adapter.format_outbound_decision(decision, raw_event)


def process_event(raw: dict, adapter: AgentAdapter) -> dict:
    """完整的事件处理流程：归一化 → 处理 → 反归一化。

    Args:
        raw: Agent 原始事件 JSON dict
        adapter: 目标 Agent 适配器

    Returns:
        Agent 期望格式的决策 dict
    """
    from claude_token_saver.hooks.handler import (
        handle_pre_tool,
        handle_post_tool,
    )

    event = adapter.parse_inbound_event(raw)
    tool_name = event.tool_name
    tool_input = event.tool_input
    tool_output = event.tool_output or {}
    session_id = event.session_id

    if event.event_type == "PreToolUse":
        result = handle_pre_tool(tool_name, tool_input, session_id)
    elif event.event_type == "PostToolUse":
        result = handle_post_tool(tool_name, tool_input, tool_output, session_id)
    else:
        result = {"decision": "approve", "reason": f"unknown event type: {event.event_type}"}

    decision = HookDecision(
        allow=result.get("decision") != "block",
        message=result.get("reason", ""),
        modified_input=result.get("modified_input"),
        modified_output=result.get("modified_output"),
    )
    return adapter.format_outbound_decision(decision, raw)


def process_event_direct(
    tool_name: str,
    tool_input: dict,
    tool_output: dict | None,
    session_id: str | None,
    event_type: str,
) -> dict:
    """直接处理 Claude Code 格式事件（无需适配器）。

    供 handler.py 在无适配器匹配时作为回退使用。
    """
    from claude_token_saver.hooks.handler import (
        handle_pre_tool,
        handle_post_tool,
    )

    tool_output = tool_output or {}

    if event_type == "PreToolUse":
        result = handle_pre_tool(tool_name, tool_input, session_id)
    elif event_type == "PostToolUse":
        result = handle_post_tool(tool_name, tool_input, tool_output, session_id)
    else:
        result = {"decision": "approve", "reason": f"unknown event type: {event_type}"}

    return {
        "allow": result.get("decision") != "block",
        "message": result.get("reason", ""),
        "updated_input": result.get("modified_input"),
        "modified_output": result.get("modified_output"),
    }


def build_test_event(adapter, display_tool: str, raw_tool: str) -> dict:
    """根据适配器类型构造测试事件（供 CLI 共享）。"""
    aid = adapter.agent_id.value
    if aid == "claude_code":
        return {
            "tool_name": display_tool,
            "tool_input": {"file_path": "test.py"},
            "hook_event_name": "PreToolUse",
        }
    return {
        "type": "pre_tool",
        "tool": raw_tool,
        "input": {"file_path": "test.py"},
    }
