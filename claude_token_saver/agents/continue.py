"""
Continue (VS Code/JetBrains) Agent Adapter。

配置:
- ~/.continue/config.json (JSON)
- matcher + handler 格式
"""
from __future__ import annotations

from pathlib import Path

from claude_token_saver.agents.base import (
    AgentID, GenericJsonAdapter, _register,
)


@_register
class _Continue(GenericJsonAdapter):
    agent_id = AgentID.CONTINUE
    name = "Continue"
    settings_path = Path.home() / ".continue" / "config.json"
    tool_map = {"Read": "Read", "Write": "Write", "Edit": "Edit",
                "Glob": "Glob", "Grep": "Grep", "Bash": "Bash"}
    env_vars = {"CONTINUE": "1"}
    event_type_map = {"preToolUse": "PreToolUse", "postToolUse": "PostToolUse",
                      "pre_tool": "PreToolUse", "post_tool": "PostToolUse"}
