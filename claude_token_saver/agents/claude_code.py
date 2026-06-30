"""
Claude Code (Anthropic) Agent Adapter。

负责将 Claude Code 的 hook 事件格式转换为内部统一格式，
并管理 ~/.claude/settings.local.json 的读写。
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from claude_token_saver.agents.base import (
    AgentAdapter, AgentConfig, AgentID, HookDecision, HookEvent,
)
from claude_token_saver.hooks import generate_hook_config, install_hooks, uninstall_hooks


@AgentAdapter.register
class ClaudeCodeAdapter(AgentAdapter):
    """Claude Code 适配器。"""

    name = "Claude Code"
    agent_id: AgentID = AgentID.CLAUDE_CODE

    # Claude Code 工具名 → 统一工具名
    TOOL_MAP: dict[str, str] = {
        "Read": "Read", "Write": "Write", "Edit": "Edit",
        "Glob": "Glob", "Grep": "Grep", "Bash": "Bash",
        "Task": "Task", "AskUserQuestion": "AskUserQuestion",
        "Skill": "Skill", "WebFetch": "WebFetch", "WebSearch": "WebSearch",
    }

    def __init__(self) -> None:
        home = Path.home()
        self._config = AgentConfig(
            id=AgentID.CLAUDE_CODE,
            name="Claude Code",
            settings_path=home / ".claude" / "settings.local.json",
            settings_format="json",
            hook_format="claude",
            transcript_dir=home / ".claude" / "projects",
            project_dir=home / ".claude" / "projects",
            tool_name_map=self.TOOL_MAP,
            env_vars={"CLAUDE_CODE": "1"},
        )

    def detect(self) -> bool:
        """检测是否运行在 Claude Code 环境中。"""
        # 方法1: 环境变量
        if os.environ.get("CLAUDE_CODE"):
            return True
        # 方法2: 检查 .claude/settings.json 是否存在
        settings = Path.home() / ".claude" / "settings.json"
        if settings.exists():
            return True
        # 方法3: 检查是否有活跃的 Claude Code 会话
        projects_dir = Path.home() / ".claude" / "projects"
        if projects_dir.exists() and any(projects_dir.glob("*.jsonl")):
            return True
        return False

    def get_tool_name(self, raw_tool_name: str) -> str:
        """Claude Code 工具名直接映射。"""
        return self.TOOL_MAP.get(raw_tool_name, raw_tool_name)

    def parse_inbound_event(self, raw: dict) -> HookEvent:
        """解析 Claude Code 的 hook 事件。

        Claude Code 事件格式:
        {
          "hook_event_name": "PreToolUse" | "PostToolUse",
          "tool_name": "Read",
          "tool_input": {...},
          "tool_output": {...},  // PostToolUse only
          "session_id": "...",
          ...
        }
        """
        event_type = raw.get("hook_event_name", "")
        raw_tool = raw.get("tool_name", "")

        return HookEvent(
            event_type=event_type,
            tool_name=self.get_tool_name(raw_tool),
            tool_input=raw.get("tool_input", {}),
            tool_output=raw.get("tool_output"),
            session_id=raw.get("session_id"),
        )

    def format_outbound_decision(self, decision: HookDecision, raw_event=None) -> dict:
        """将统一决策转换为 Claude Code 期望的格式。

        Claude Code 期望:
        {
          "allow": true/false,
          "message": "...",
          "updated_input": {...}  // optional
        }
        """
        result = {"allow": decision.allow}
        if decision.message:
            result["message"] = decision.message
        if decision.modified_input is not None:
            result["updated_input"] = decision.modified_input
        return result

    def install_config(self, dry_run: bool = False) -> Path:
        """安装 hook 配置到 ~/.claude/settings.local.json。"""
        return install_hooks(dry_run=dry_run)

    def uninstall_config(self) -> bool:
        """从 ~/.claude/settings.local.json 移除 hook 配置。"""
        try:
            uninstall_hooks()
            return True
        except Exception:
            return False

    def is_installed(self) -> bool:
        """检查 ~/.claude/settings.local.json 中是否已有 token-saver hooks。"""
        settings_path = self._config.settings_path
        if not settings_path.exists():
            return False
        try:
            with open(settings_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            hooks = config.get("hooks", {})
            for event_hooks in hooks.values():
                if isinstance(event_hooks, list):
                    for entry in event_hooks:
                        if isinstance(entry, dict):
                            for hook in entry.get("hooks", []):
                                if isinstance(hook, dict) and "claude_token_saver" in hook.get("command", ""):
                                    return True
                                if isinstance(hook, dict) and "handler.py" in hook.get("command", ""):
                                    return True
        except (json.JSONDecodeError, IOError):
            pass
        return False
