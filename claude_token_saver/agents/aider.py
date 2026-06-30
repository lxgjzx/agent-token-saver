"""
Aider Agent Adapter。

处理 Aider 的 hook 事件格式和配置路径。

Aider 配置:
- 配置文件: ~/.aider/configuration.yml (YAML)
- Hook 格式: 基于 stdin/stdin JSON 的 command hook
- 工具名: add_to_chat, run, write, read 等
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from claude_token_saver.agents.base import (
    AgentAdapter, AgentConfig, AgentID, HookDecision, HookEvent,
)


def _parse_yaml_simple(text: str) -> dict:
    """极简 YAML 解析器（仅处理 aider 配置所需的子集）。

    支持: key: value, list items (- ...), nested dicts。
    不追求完整 YAML 兼容，仅满足 aider configuration.yml 的解析需求。
    """
    result: dict = {}
    if not text.strip():
        return result

    lines = text.split("\n")
    _parse_block(lines, 0, result)
    return result


def _parse_block(lines: list[str], start: int, target: dict) -> int:
    """解析 YAML 块，返回下一行索引。"""
    i = start
    while i < len(lines):
        line = lines[i]
        stripped = line.lstrip()
        if not stripped or stripped.startswith("#"):
            i += 1
            continue

        indent = len(line) - len(stripped)

        # 检测列表项
        if stripped.startswith("- "):
            # 列表需要在父级已设置 list 的情况下处理
            i += 1
            continue

        # key: value 对
        if ":" in stripped:
            key = stripped.split(":", 1)[0].strip()
            rest = stripped.split(":", 1)[1].strip()

            if not rest:
                # 可能是嵌套块
                i += 1
                sub = {}
                while i < len(lines):
                    sline = lines[i]
                    sstripped = sline.lstrip()
                    if not sstripped or sstripped.startswith("#"):
                        i += 1
                        continue
                    sindent = len(sline) - len(sstripped)
                    if sindent <= indent and not sstripped.startswith("-"):
                        break
                    if sstripped.startswith("- "):
                        # 列表值
                        if "__list__" not in sub:
                            sub["__list__"] = []
                        item_val = sstripped[2:].strip()
                        if ":" in item_val:
                            item_dict: dict = {}
                            _kv_from_string(item_val, item_dict)
                            sub["__list__"].append(item_dict)
                        else:
                            sub["__list__"].append(item_val)
                        i += 1
                    elif ":" in sstripped:
                        sub_key = sstripped.split(":", 1)[0].strip()
                        sub_rest = sstripped.split(":", 1)[1].strip()
                        if sub_rest:
                            sub[sub_key] = _yaml_value(sub_rest)
                            i += 1
                        else:
                            inner: dict = {}
                            i = _parse_block(lines, i + 1, inner)
                            sub[sub_key] = inner
                    else:
                        i += 1

                if "__list__" in sub:
                    target[key] = sub.pop("__list__")
                else:
                    target[key] = sub
            else:
                target[key] = _yaml_value(rest)
                i += 1
        else:
            i += 1

    return i


def _kv_from_string(s: str, target: dict) -> None:
    """从 'k: v' 格式字符串填充 target dict。"""
    if ":" in s:
        k, v = s.split(":", 1)
        target[k.strip()] = _yaml_value(v.strip())


def _yaml_value(s: str) -> Any:
    """将 YAML 标量值转换为 Python 类型。"""
    s = s.strip()
    if s.lower() == "true":
        return True
    if s.lower() == "false":
        return False
    if s.lower() in ("null", "~", ""):
        return None
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    # 去掉引号
    if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
        return s[1:-1]
    return s


def _write_yaml_simple(data: dict, path: Path) -> None:
    """将字典写入 YAML 文件。"""
    path.write_text(_yaml_dumps(data), encoding="utf-8")


def _write_yaml_simple_to(data: dict, buf: Any) -> None:
    """将字典写入 StringIO（用于 dry-run）。"""
    buf.write(_yaml_dumps(data))


def _yaml_dumps(data: dict) -> str:
    """将字典序列化为 YAML 字符串。"""
    lines = []
    for key, value in data.items():
        if isinstance(value, dict):
            lines.append(f"{key}:")
            for k2, v2 in value.items():
                if isinstance(v2, list):
                    lines.append(f"  {k2}:")
                    for item in v2:
                        if isinstance(item, dict):
                            parts = ", ".join(f"{k}: {_yaml_scalar(v)}" for k, v in item.items())
                            lines.append(f"    - {{{parts}}}")
                        else:
                            lines.append(f"    - {_yaml_scalar(item)}")
                else:
                    lines.append(f"  {k2}: {_yaml_scalar(v2)}")
        elif isinstance(value, list):
            lines.append(f"{key}:")
            for item in value:
                if isinstance(item, dict):
                    parts = ", ".join(f"{k}: {_yaml_scalar(v)}" for k, v in item.items())
                    lines.append(f"  - {{{parts}}}")
                else:
                    lines.append(f"  - {_yaml_scalar(item)}")
        else:
            lines.append(f"{key}: {_yaml_scalar(value)}")
    return "\n".join(lines) + "\n"


def _yaml_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return "null"
    if isinstance(value, (int, float)):
        return str(value)
    return str(value)


@AgentAdapter.register
class AiderAdapter(AgentAdapter):
    """Aider 适配器。

    处理:
    - Hook 事件: pre_tool / post_tool (stdin JSON → HookEvent)
    - 配置: ~/.aider/configuration.yml (YAML)
    - 工具名: add_to_chat, run, write, read, search 等
    """

    name = "Aider"
    agent_id: AgentID = AgentID.AIDER

    # Aider 工具名 → 统一工具名
    TOOL_MAP: dict[str, str] = {
        "add_to_chat": "Write",
        "run": "Bash",
        "write": "Write",
        "read": "Read",
        "search": "Grep",
        "glob": "Glob",
        "edit": "Edit",
        "ask": "AskUserQuestion",
    }

    def __init__(self) -> None:
        home = Path.home()
        self._config = AgentConfig(
            id=AgentID.AIDER,
            name="Aider",
            settings_path=home / ".aider" / "configuration.yml",
            settings_format="yaml",
            hook_format="aider",
            transcript_dir=home / ".aider" / "sessions",
            project_dir=home / ".aider" / "projects",
            tool_name_map=self.TOOL_MAP,
            env_vars={"AIDER": "1"},
        )


    def detect(self) -> bool:
        if os.environ.get("AIDER"):
            return True
        aider_dir = Path.home() / ".aider"
        if aider_dir.exists():
            return True
        return False

    def get_tool_name(self, raw_tool_name: str) -> str:
        return self.TOOL_MAP.get(raw_tool_name, raw_tool_name)

    def parse_inbound_event(self, raw: dict) -> HookEvent:
        event_type_raw = raw.get("event", "") or raw.get("type", "")
        raw_tool = raw.get("tool", "") or raw.get("tool_name", "")

        if event_type_raw == "pre_tool":
            event_type = "PreToolUse"
        elif event_type_raw == "post_tool":
            event_type = "PostToolUse"
        else:
            event_type = event_type_raw

        return HookEvent(
            event_type=event_type,
            tool_name=self.get_tool_name(raw_tool),
            tool_input=raw.get("input", raw.get("tool_input", {})),
            tool_output=raw.get("output", raw.get("tool_output")),
            session_id=raw.get("session_id"),
        )

    def format_outbound_decision(self, decision: HookDecision, raw_event=None) -> dict:
        result = {
            "allow": decision.allow,
            "message": decision.message,
            "modified_input": decision.modified_input,
        }
        if decision.modified_output is not None:
            result["modified_output"] = decision.modified_output
        return result

    def install_config(self, dry_run: bool = False) -> Path:
        config_dir = Path.home() / ".aider"
        config_dir.mkdir(parents=True, exist_ok=True)
        config_path = config_dir / "configuration.yml"

        hook_entry = {
            "hooks": [
                {"event": "pre_tool", "command": f'"{sys.executable}" -m claude_token_saver.hooks.handler'},
                {"event": "post_tool", "command": f'"{sys.executable}" -m claude_token_saver.hooks.handler'},
            ]
        }

        if dry_run:
            import io as _io
            buf = _io.StringIO()
            _write_yaml_simple_to(hook_entry, buf)
            print(buf.getvalue())
            return config_path

        existing = {}
        if config_path.exists():
            try:
                text = config_path.read_text(encoding="utf-8")
                existing = _parse_yaml_simple(text)
            except Exception:
                existing = {}

        existing_hooks = existing.get("hooks", [])
        if not isinstance(existing_hooks, list):
            existing_hooks = []
        new_hooks = hook_entry["hooks"]
        for nh in new_hooks:
            if not any(
                isinstance(eh, dict) and eh.get("event") == nh.get("event")
                for eh in existing_hooks
            ):
                existing_hooks.append(nh)
        existing["hooks"] = existing_hooks

        _write_yaml_simple(existing, config_path)
        return config_path

    def uninstall_config(self) -> bool:
        config_path = self._config.settings_path
        if not config_path.exists():
            return True
        try:
            text = config_path.read_text(encoding="utf-8")
            config = _parse_yaml_simple(text)
            hooks = config.get("hooks", [])
            marker = "claude_token_saver"
            if isinstance(hooks, list):
                config["hooks"] = [
                    h for h in hooks
                    if not (isinstance(h, dict) and marker in str(h.get("command", "")))
                ]
                if not config["hooks"]:
                    del config["hooks"]
            _write_yaml_simple(config, config_path)
            return True
        except Exception:
            return False

    def is_installed(self) -> bool:
        config_path = self._config.settings_path
        if not config_path.exists():
            return False
        try:
            text = config_path.read_text(encoding="utf-8")
            config = _parse_yaml_simple(text)
            hooks = config.get("hooks", [])
            if isinstance(hooks, list):
                for h in hooks:
                    if isinstance(h, dict) and "claude_token_saver" in str(h.get("command", "")):
                        return True
        except Exception:
            pass
        return False
