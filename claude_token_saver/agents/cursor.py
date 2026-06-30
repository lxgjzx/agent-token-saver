"""
Cursor IDE Agent Adapter。

配置:
- ~/.cursor/settings.json (JSON)
- matcher + handler 格式
"""
from __future__ import annotations

from pathlib import Path

from claude_token_saver.agents.base import (
    AgentID, GenericJsonAdapter, _register,
)


@_register
class _Cursor(GenericJsonAdapter):
    agent_id = AgentID.CURSOR
    name = "Cursor"
    settings_path = Path.home() / ".cursor" / "settings.json"
    tool_map = {"read_file": "Read", "write_file": "Write", "edit": "Edit",
                "search": "Grep", "glob": "Glob", "bash": "Bash"}
    env_vars = {"CURSOR": "1"}
    event_type_map = {"pre_tool": "PreToolUse", "post_tool": "PostToolUse"}
    outbound_keys = {"allow": "allow", "message": "message", "modified_input": "updated_input"}
