"""
Windsurf (Codeium) Agent Adapter。

配置:
- ~/.windsurf/settings.json (JSON)
- matcher + handler 格式
"""
from __future__ import annotations

from pathlib import Path

from claude_token_saver.agents.base import (
    AgentID, GenericJsonAdapter, _register,
)


@_register
class _Windsurf(GenericJsonAdapter):
    agent_id = AgentID.WINDSURF
    name = "Windsurf"
    settings_path = Path.home() / ".windsurf" / "settings.json"
    tool_map = {"read_file": "Read", "write_file": "Write", "edit": "Edit",
                "search": "Grep", "glob": "Glob", "bash": "Bash"}
    env_vars = {"WINDSURF": "1"}
    outbound_keys = {"allow": "allow", "message": "message", "modified_input": "updated_input"}
