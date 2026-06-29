"""
hooks 子命令 - Claude Code Hook 安装、卸载、状态查询、测试
"""
from __future__ import annotations

import json
import subprocess
import sys

import click

from claude_token_saver.hooks import (
    generate_hook_config,
    install_hooks,
    uninstall_hooks,
)
from claude_token_saver.hooks.handler import handle_pre_tool, handle_post_tool


def _call_handler(event_data: dict) -> dict:
    """调用 hook handler，根据 hook_event_name 分发到对应处理函数。"""
    tool_name = event_data.get("tool_name", "")
    tool_input = event_data.get("tool_input", {})
    tool_output = event_data.get("tool_output") or {}
    session_id = event_data.get("session_id")
    event_type = event_data.get("hook_event_name", "")

    if not tool_name:
        return {"allow": True, "message": "no tool_name"}

    try:
        if event_type == "PreToolUse":
            result = handle_pre_tool(tool_name, tool_input, session_id)
        elif event_type == "PostToolUse":
            result = handle_post_tool(tool_name, tool_input, tool_output, session_id)
        else:
            return {"allow": True, "message": f"unknown event type: {event_type}"}
    except Exception as e:
        return {"allow": True, "message": f"handler error: {e}"}

    # hook handler 返回 {"decision": "approve", ...} 或 {"decision": "block", ...}
    decision = result.get("decision", "approve")
    return {
        "allow": decision == "approve",
        "message": result.get("reason", ""),
        "updated_input": result.get("modified_input"),
    }


@click.group()
def hooks() -> None:
    """管理 Claude Code Hooks（PreToolUse / PostToolUse）。"""
    pass


@hooks.command("install")
@click.option("--dry-run", is_flag=True, help="只显示将要写入的配置，不实际写入")
def hooks_install(dry_run: bool) -> None:
    """安装 token-saver hooks 到 .claude/settings.local.json。"""
    settings_local_path = __import__("pathlib").Path.home() / ".claude" / "settings.local.json"

    if dry_run:
        config = generate_hook_config()
        click.echo(click.style("📋 将写入以下配置到 " + str(settings_local_path) + ":\n", fg="cyan"))
        click.echo(json.dumps(config, indent=2, ensure_ascii=False))
        return

    try:
        path = install_hooks()
        click.echo(click.style(f"✅ Hooks 已安装到: {path}", fg="green"))
        click.echo("")
        click.echo("安装的 hooks:")
        hook_config = generate_hook_config()
        for event_type, matchers in hook_config.get("hooks", {}).items():
            click.echo(f"  {event_type}:")
            for m in matchers:
                click.echo(f"    matcher={m.get('matcher', '')}")
    except (OSError, IOError) as e:
        click.echo(f"❌ 安装失败: {e}", err=True)
        raise SystemExit(1)


@hooks.command("uninstall")
@click.option("--dry-run", is_flag=True, help="只显示将要移除的配置，不实际写入")
def hooks_uninstall(dry_run: bool) -> None:
    """从 .claude/settings.local.json 中移除 token-saver hooks。"""
    settings_local_path = __import__("pathlib").Path.home() / ".claude" / "settings.local.json"

    if dry_run:
        if settings_local_path.exists():
            try:
                with open(settings_local_path, "r", encoding="utf-8") as f:
                    existing = json.load(f)
                if "hooks" in existing:
                    click.echo(click.style("📋 将移除以下配置:\n", fg="yellow"))
                    click.echo(json.dumps(existing["hooks"], indent=2, ensure_ascii=False))
                else:
                    click.echo("📭 当前没有 hooks 配置需要移除")
            except (json.JSONDecodeError, IOError):
                click.echo("📭 配置文件无法解析")
        else:
            click.echo("📭 settings.local.json 不存在，无需卸载")
        return

    try:
        path = uninstall_hooks()
        click.echo(click.style(f"✅ Hooks 已从: {path} 中移除", fg="green"))
    except (OSError, IOError) as e:
        click.echo(f"❌ 卸载失败: {e}", err=True)
        raise SystemExit(1)


@hooks.command("status")
@click.option("--json", "json_output", is_flag=True, help="JSON 格式输出")
def hooks_status(json_output: bool) -> None:
    """检查 hooks 安装状态。"""
    settings_local_path = __import__("pathlib").Path.home() / ".claude" / "settings.local.json"
    from pathlib import Path

    hook_config = generate_hook_config()
    expected_hooks = hook_config.get("hooks", {})

    result = {
        "settings_local_exists": settings_local_path.exists(),
        "installed": False,
        "events": {},
        "handler_exists": Path(__file__).parent.parent / "hooks" / "handler.py",
    }

    if settings_local_path.exists():
        try:
            with open(settings_local_path, "r", encoding="utf-8") as f:
                existing = json.load(f) or {}
            current_hooks = existing.get("hooks", {})

            for event_type, matchers in expected_hooks.items():
                current_matchers = current_hooks.get(event_type, [])
                installed_matchers = []
                for cm in current_matchers:
                    if isinstance(cm, dict):
                        matcher_name = cm.get("matcher", "")
                        if any(
                            isinstance(em, dict) and em.get("matcher") == matcher_name
                            for em in matchers
                        ):
                            installed_matchers.append(matcher_name)
                result["events"][event_type] = installed_matchers

            result["installed"] = any(result["events"].values())
        except (json.JSONDecodeError, IOError):
            pass

    if json_output:
        result["handler_exists"] = str(result["handler_exists"])
        click.echo(click.style(json.dumps(result, indent=2, ensure_ascii=False), fg="cyan"))
        return

    # 文本输出
    if result["installed"]:
        click.echo(click.style("✅ Hooks 已安装\n", bold=True, fg="green"))
    else:
        click.echo(click.style("📭 Hooks 未安装\n", bold=True, fg="yellow"))

    if result["events"]:
        click.echo("已注册事件:")
        for event_type, matchers in result["events"].items():
            if matchers:
                click.echo(f"  {event_type}: {', '.join(matchers)}")

    click.echo(f"\n配置路径: {settings_local_path}")
    click.echo(f"Handler:   {result['handler_exists']}")


@hooks.command("test")
@click.argument("event_file", required=False)
@click.option("--tool", "-t", help="模拟工具名称（如 Read, Glob, Grep）")
@click.option("--stdin", is_flag=True, help="从标准输入读取事件 JSON")
def hooks_test(event_file: str | None, tool: str | None, stdin: bool) -> None:
    """测试 hook handler 是否正常工作。

    \b
    示例:
      cts hooks test --tool Read --stdin <<< '{"tool_name": "Read", "tool_input": {"file_path": "test.py"}}'
      cts hooks test event.json
    """
    if stdin:
        click.echo("⏳ 从 stdin 读取事件...")
        event_json = sys.stdin.read()
    elif event_file:
        try:
            with open(event_file, "r", encoding="utf-8") as f:
                event_json = f.read()
        except (OSError, IOError) as e:
            click.echo(f"❌ 无法读取文件: {e}", err=True)
            raise SystemExit(1)
    elif tool:
        # 构造一个模拟事件
        event = {
            "tool_name": tool,
            "tool_input": {"file_path": "test.py"},
            "cwd": ".",
        }
        event_json = json.dumps(event, ensure_ascii=False)
    else:
        click.echo("❌ 请提供事件来源：--tool, --stdin 或 event_file")
        raise SystemExit(1)

    click.echo(click.style("🧪 测试 hook handler...\n", bold=True, fg="cyan"))
    click.echo(f"事件数据: {event_json[:200]}...")

    try:
        event_data = json.loads(event_json)
    except json.JSONDecodeError as e:
        click.echo(f"❌ JSON 解析失败: {e}", err=True)
        raise SystemExit(1)

    result = _call_handler(event_data)

    click.echo("")
    if result.get("allow"):
        click.echo(click.style("✅ Hook 返回: allow", fg="green"))
    else:
        click.echo(click.style("🚫 Hook 返回: block", fg="red"))

    if result.get("message"):
        click.echo(f"   消息: {result['message']}")

    if result.get("updated_input"):
        click.echo(f"   修改后的输入: {json.dumps(result['updated_input'], ensure_ascii=False)[:200]}")
