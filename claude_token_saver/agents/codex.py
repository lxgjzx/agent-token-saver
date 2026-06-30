"""
Codex CLI (OpenAI) Agent Adapter。

处理 Codex 的 hook 事件格式和配置路径。

Codex CLI 配置:
- 配置文件: ~/.codex/config.json
- Hook 格式: PreToolUse/PostToolUse → matcher + handler
- 工具名: 与 OpenAI API 一致 (read_file, write_file, grep, glob 等)
"""
from __future__ import annotations

from pathlib import Path

from claude_token_saver.agents.base import (
    AgentID, GenericJsonAdapter, _register,
)


@_register
class _Codex(GenericJsonAdapter):
    agent_id = AgentID.CODEX
    name = "OpenAI Codex"
    settings_path = Path.home() / ".codex" / "config.json"
    tool_map = {"read_file": "Read", "write_file": "Write", "edit_file": "Edit",
                "grep": "Grep", "glob": "Glob", "run_command": "Bash"}
    env_vars = {"OPENAI_CODEX": "1", "CODEX_CLI": "1"}
    event_type_map = {"pre_tool": "PreToolUse", "post_tool": "PostToolUse"}
    outbound_keys = {"allow": "allow", "message": "reason", "modified_input": "modified_input"}
