"""
OpenClaw Agent Adapter。

处理 OpenClaw 的 hook 事件格式和配置路径。

OpenClaw 是一个开源的 AI coding agent 框架。

配置:
- ~/.openclaw/settings.json (JSON)
- tools 列表格式（非 matcher）
"""
from __future__ import annotations

from pathlib import Path

from claude_token_saver.agents.base import (
    AgentID, GenericJsonAdapter, _register, _load_json, _save_json,
)


@_register
class _OpenClaw(GenericJsonAdapter):
    agent_id = AgentID.OPENCLAW
    name = "OpenClaw"
    settings_path = Path.home() / ".openclaw" / "settings.json"
    tool_map = {"read": "Read", "write": "Write", "edit": "Edit",
                "glob": "Glob", "grep": "Grep", "bash": "Bash"}
    env_vars = {"OPENCLAW": "1"}
    event_type_map = {"pre_tool": "PreToolUse", "post_tool": "PostToolUse"}
    hook_key = "tools"  # OpenClaw 用 tools 列表而非 matcher 字符串

    # OpenClaw 用原始工具名（非统一名）作为 tools 列表值
    def get_tool_name(self, raw: str) -> str:
        return raw

    def install_config(self, dry_run: bool = False) -> Path:
        import sys
        p = self.settings_path
        p.parent.mkdir(parents=True, exist_ok=True)
        cmd = f'"{sys.executable}" -m claude_token_saver.hooks.handler'
        raw_tools = sorted(set(self.tool_map.keys()))
        entry = {
            "hooks": {
                "PreToolUse": [{"tools": raw_tools, "handler": cmd}],
                "PostToolUse": [{"tools": raw_tools, "handler": cmd}],
            }
        }
        if dry_run:
            import json
            print(json.dumps(entry, indent=2, ensure_ascii=False))
            return p
        existing = _load_json(p)
        existing.setdefault("hooks", {})
        for evt, entries in entry.get("hooks", {}).items():
            existing["hooks"].setdefault(evt, [])
            for e in entries:
                if e not in existing["hooks"][evt]:
                    existing["hooks"][evt].append(e)
        _save_json(p, existing)
        return p
