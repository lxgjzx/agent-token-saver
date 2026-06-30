"""
Agent 适配器抽象层 — 支持多 AI Coding Agent 统一接入。
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional


# ═══════════════════════════════════════════════════════════════════════════════
# 全局注册表
# ═══════════════════════════════════════════════════════════════════════════════

_ADAPTER_REGISTRY: dict[str, type] = {}


class AgentID(str, Enum):
    CLAUDE_CODE = "claude_code"
    CODEX = "codex"
    OPENCLAW = "openclaw"
    CURSOR = "cursor"
    AIDER = "aider"
    CONTINUE = "continue"
    WINDSURF = "windsurf"
    GENERIC = "generic"


# ═══════════════════════════════════════════════════════════════════════════════
# 数据类
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class AgentConfig:
    id: AgentID
    name: str
    settings_path: Path
    settings_format: str
    hook_format: str
    transcript_dir: Optional[Path] = None
    project_dir: Optional[Path] = None
    tool_name_map: dict[str, str] = field(default_factory=dict)
    env_vars: dict[str, str] = field(default_factory=dict)


@dataclass
class HookEvent:
    event_type: str
    tool_name: str
    tool_input: dict
    tool_output: Optional[dict] = None
    session_id: Optional[str] = None


@dataclass
class HookDecision:
    allow: bool
    message: str = ""
    modified_input: Optional[dict] = None
    modified_output: Optional[dict] = None


# ═══════════════════════════════════════════════════════════════════════════════
# 通用 JSON 适配器（覆盖 Codex/OpenClaw/Cursor/Windsurf/Continue）
# ═══════════════════════════════════════════════════════════════════════════════

class GenericJsonAdapter:
    """基于 JSON/YAML 配置文件的通用 Agent 适配器。

    子类只需定义类属性即可，无需重写任何方法（特殊 Agent 除外）。
    """

    agent_id: AgentID = AgentID.GENERIC
    name: str = ""
    settings_path: Path = Path("")
    tool_map: dict[str, str] = {}
    env_vars: dict[str, str] = {}
    event_type_map: dict[str, str] = {}
    outbound_keys: dict[str, str] = {
        "allow": "allow", "message": "message",
        "modified_input": "modified_input", "modified_output": "modified_output",
    }
    hook_key: str = "matcher"

    @classmethod
    def register(cls, subcls: type) -> type:
        _ADAPTER_REGISTRY[subcls.agent_id] = subcls
        return subcls

    @property
    def agent_name(self) -> str:
        return self.name

    def __init__(self):
        self._config = AgentConfig(
            id=self.agent_id,
            name=self.name,
            settings_path=self.settings_path,
            settings_format="json",
            hook_format="json",
            transcript_dir=self.settings_path.parent / "sessions",
            project_dir=self.settings_path.parent / "projects",
            tool_name_map=self.tool_map,
            env_vars=self.env_vars,
        )

    @property
    def config(self) -> AgentConfig:
        return self._config

    def detect(self) -> bool:
        for var in self.env_vars:
            if os.environ.get(var):
                return True
        # 需要 settings 文件实际存在（目录存在不足以证明已安装）
        return self.settings_path.exists()

    def get_tool_name(self, raw: str) -> str:
        return self.tool_map.get(raw, raw)

    def parse_inbound_event(self, raw: dict) -> HookEvent:
        et = self.event_type_map.get(raw.get("type", ""), raw.get("type", ""))
        tool = raw.get("tool", raw.get("tool_name", ""))
        return HookEvent(
            event_type=et,
            tool_name=self.get_tool_name(tool),
            tool_input=raw.get("input", raw.get("tool_input", {})),
            tool_output=raw.get("output", raw.get("tool_output")),
            session_id=raw.get("session_id"),
        )

    def format_outbound_decision(self, decision: HookDecision, raw_event=None) -> dict:
        keys = self.outbound_keys
        result = {keys["allow"]: decision.allow}
        if decision.message:
            result[keys["message"]] = decision.message
        if decision.modified_input is not None:
            result[keys["modified_input"]] = decision.modified_input
        if decision.modified_output is not None:
            result[keys["modified_output"]] = decision.modified_output
        return result

    def install_config(self, dry_run: bool = False) -> Path:
        p = self.settings_path
        p.parent.mkdir(parents=True, exist_ok=True)
        k = self.hook_key
        cmd = f'"{sys.executable}" -m claude_token_saver.hooks.handler'
        entry = {
            "hooks": {
                "PreToolUse": [{k: "Read|Glob|Grep", "handler": cmd}],
                "PostToolUse": [{k: "Read|Glob|Grep", "handler": cmd}],
            }
        }
        if dry_run:
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

    def uninstall_config(self) -> bool:
        p = self.settings_path
        if not p.exists():
            return True
        try:
            config = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return False
        hooks = config.get("hooks", {})
        marker = "claude_token_saver"
        changed = False
        for key in list(hooks):
            hooks[key] = [e for e in hooks[key] if not _has_marker(e, marker)]
            if not hooks[key]:
                del hooks[key]
            changed = True
        if changed:
            _save_json(p, config)
        return True

    def is_installed(self) -> bool:
        p = self.settings_path
        if not p.exists():
            return False
        try:
            config = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return False
        return any(
            _has_marker(e, "claude_token_saver")
            for hooks in config.get("hooks", {}).values()
            for e in hooks
        )


# 向后兼容别名
AgentAdapter = GenericJsonAdapter


def _has_marker(entry: Any, marker: str) -> bool:
    if not isinstance(entry, dict):
        return False
    for v in entry.values():
        if isinstance(v, str) and marker in v:
            return True
        if isinstance(v, list):
            for item in v:
                if isinstance(item, dict) and _has_marker(item, marker):
                    return True
    return False


def _load_json(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8")) or {}
        except Exception:
            return {}
    return {}


def _save_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


# ═══════════════════════════════════════════════════════════════════════════════
# Agent 注册宏
# ═══════════════════════════════════════════════════════════════════════════════

def _register(cls: type) -> type:
    _ADAPTER_REGISTRY[cls.agent_id] = cls
    return cls


# ═══════════════════════════════════════════════════════════════════════════════
# Claude Code（特殊：使用 hooks/__init__.py 的配置系统）
# ═══════════════════════════════════════════════════════════════════════════════

@_register
class ClaudeCodeAdapter(GenericJsonAdapter):
    agent_id = AgentID.CLAUDE_CODE
    name = "Claude Code"
    settings_path = Path.home() / ".claude" / "settings.local.json"
    tool_map = {"Read": "Read", "Write": "Write", "Edit": "Edit",
                "Glob": "Glob", "Grep": "Grep", "Bash": "Bash"}
    env_vars = {"CLAUDE_CODE": "1"}
    event_type_map = {}  # Claude Code uses hook_event_name, not type

    def detect(self) -> bool:
        if os.environ.get("CLAUDE_CODE"):
            return True
        return (Path.home() / ".claude" / "settings.json").exists()

    def parse_inbound_event(self, raw: dict) -> HookEvent:
        return HookEvent(
            event_type=raw.get("hook_event_name", ""),
            tool_name=self.get_tool_name(raw.get("tool_name", "")),
            tool_input=raw.get("tool_input", {}),
            tool_output=raw.get("tool_output"),
            session_id=raw.get("session_id"),
        )

    def install_config(self, dry_run: bool = False) -> Path:
        from claude_token_saver.hooks import install_hooks
        return Path(install_hooks(dry_run=dry_run))

    def uninstall_config(self) -> bool:
        from claude_token_saver.hooks import uninstall_hooks
        try:
            uninstall_hooks()
            return True
        except Exception:
            return False

    def is_installed(self) -> bool:
        p = self.settings_path
        if not p.exists():
            return False
        try:
            config = json.loads(p.read_text(encoding="utf-8"))
            hooks = config.get("hooks", {})
            for entries in hooks.values():
                for e in entries:
                    if isinstance(e, dict):
                        for hook in e.get("hooks", []):
                            if isinstance(hook, dict) and "token-saver" in hook.get("command", ""):
                                return True
                            if isinstance(hook, dict) and "handler.py" in hook.get("command", ""):
                                return True
        except Exception:
            pass
        return False


# ═══════════════════════════════════════════════════════════════════════════════
# Aider（特殊：YAML 配置）
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_yaml(text: str) -> dict:
    result = {}
    for line in text.split("\n"):
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        if ":" in s:
            k, v = s.split(":", 1)
            v = v.strip()
            if not v:
                result[k.strip()] = {}
            else:
                result[k.strip()] = _yaml_val(v)
    return result


def _yaml_val(s: str) -> Any:
    s = s.strip()
    if s.lower() == "true": return True
    if s.lower() == "false": return False
    if s.lower() in ("null", "~"): return None
    try: return int(s)
    except: pass
    try: return float(s)
    except: pass
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        return s[1:-1]
    return s


def _dump_yaml(data: dict) -> str:
    lines = []
    for k, v in data.items():
        if isinstance(v, dict):
            lines.append(f"{k}:")
            for k2, v2 in v.items():
                lines.append(f"  {k2}: {_yaml_scalar(v2)}")
        elif isinstance(v, list):
            lines.append(f"{k}:")
            for item in v:
                lines.append(f"  - {_yaml_scalar(item)}")
        else:
            lines.append(f"{k}: {_yaml_scalar(v)}")
    return "\n".join(lines) + "\n"


def _yaml_scalar(v: Any) -> str:
    if isinstance(v, bool): return "true" if v else "false"
    if v is None: return "null"
    return str(v)


@_register
class AiderAdapter(GenericJsonAdapter):
    agent_id = AgentID.AIDER
    name = "Aider"
    settings_path = Path.home() / ".aider" / "configuration.yml"
    tool_map = {"run": "Bash", "write": "Write", "read": "Read", "search": "Grep"}
    env_vars = {"AIDER": "1"}
    event_type_map = {"pre_tool": "PreToolUse", "post_tool": "PostToolUse"}
    outbound_keys = {"allow": "allow", "message": "message", "modified_input": "modified_input"}

    def detect(self) -> bool:
        if os.environ.get("AIDER"):
            return True
        return self.settings_path.parent.exists()

    def install_config(self, dry_run: bool = False) -> Path:
        p = self.settings_path
        p.parent.mkdir(parents=True, exist_ok=True)
        entry = {"hooks": [
            {"event": "pre_tool", "command": f'"{sys.executable}" -m claude_token_saver.hooks.handler'},
            {"event": "post_tool", "command": f'"{sys.executable}" -m claude_token_saver.hooks.handler'},
        ]}
        if dry_run:
            print(_dump_yaml(entry))
            return p
        existing = {}
        if p.exists():
            try:
                existing = _parse_yaml(p.read_text(encoding="utf-8"))
            except Exception:
                pass
        hooks = existing.get("hooks", [])
        if not isinstance(hooks, list):
            hooks = []
        for nh in entry["hooks"]:
            if not any(isinstance(e, dict) and e.get("event") == nh["event"] for e in hooks):
                hooks.append(nh)
        existing["hooks"] = hooks
        p.write_text(_dump_yaml(existing), encoding="utf-8")
        return p

    def uninstall_config(self) -> bool:
        p = self.settings_path
        if not p.exists():
            return True
        try:
            existing = _parse_yaml(p.read_text(encoding="utf-8"))
            hooks = existing.get("hooks", [])
            if isinstance(hooks, list):
                existing["hooks"] = [h for h in hooks
                    if not (isinstance(h, dict) and "claude_token_saver" in str(h.get("command", "")))]
                if not existing["hooks"]:
                    del existing["hooks"]
            p.write_text(_dump_yaml(existing), encoding="utf-8")
            return True
        except Exception:
            return False

    def is_installed(self) -> bool:
        p = self.settings_path
        if not p.exists():
            return False
        try:
            config = _parse_yaml(p.read_text(encoding="utf-8"))
            hooks = config.get("hooks", [])
            return any(isinstance(h, dict) and "claude_token_saver" in str(h.get("command", "")) for h in hooks)
        except Exception:
            return False
