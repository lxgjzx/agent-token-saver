"""
hooks 子命令 - 多 Agent Hook 安装、卸载、状态查询、测试

便捷特性:
  - 默认自动检测 Agent，无需手动指定 --agent
  - test 命令未安装时自动提示安装
"""
from __future__ import annotations

import json
import sys

import click

from claude_token_saver.agents import (
    build_test_event,
    get_all_adapters,
    resolve_adapter,
)
from claude_token_saver.agents.base import AgentID

_AGENT_CHOICES = [a.value for a in AgentID]


def _resolve(agent_name: str | None):
    """CLI 专用包装：将 ValueError 转为 click.BadParameter。"""
    try:
        return resolve_adapter(agent_name)
    except ValueError as e:
        raise click.BadParameter(str(e))


@click.group()
def hooks() -> None:
    """管理多 Agent Hooks（PreToolUse / PostToolUse）。"""
    pass


@hooks.command("install")
@click.option("--agent", "-a",
    type=click.Choice(_AGENT_CHOICES, case_sensitive=False),
    default=None, help="指定 Agent（默认: 自动检测）")
@click.option("--dry-run", is_flag=True)
def hooks_install(agent: str | None, dry_run: bool) -> None:
    """安装 token-saver hooks。"""
    adapter = _resolve(agent)

    if dry_run:
        click.echo(click.style(f"📋 将写入: {adapter.config.settings_path}\n", fg="cyan"))
        adapter.install_config(dry_run=True)
        if adapter.config.settings_path.exists():
            with open(adapter.config.settings_path, "r", encoding="utf-8") as f:
                existing = json.load(f) or {}
            click.echo(json.dumps(existing.get("hooks", {}), indent=2, ensure_ascii=False))
        return

    try:
        path = adapter.install_config()
        click.echo(click.style(f"✅ 已安装到: {path}", fg="green"))
    except (OSError, IOError) as e:
        click.echo(f"❌ 安装失败: {e}", err=True)
        raise SystemExit(1)


@hooks.command("uninstall")
@click.option("--agent", "-a",
    type=click.Choice(_AGENT_CHOICES, case_sensitive=False),
    default=None)
@click.option("--dry-run", is_flag=True)
def hooks_uninstall(agent: str | None, dry_run: bool) -> None:
    """移除 token-saver hooks。"""
    adapter = _resolve(agent)

    if dry_run:
        path = adapter.config.settings_path
        if path.exists():
            try:
                with open(path, "r", encoding="utf-8") as f:
                    existing = json.load(f)
                if "hooks" in existing:
                    click.echo(click.style("📋 将移除:\n", fg="yellow"))
                    click.echo(json.dumps(existing["hooks"], indent=2, ensure_ascii=False))
                else:
                    click.echo("📭 没有 hooks 配置")
            except Exception:
                click.echo("📭 无法解析配置文件")
        else:
            click.echo(f"📭 {path} 不存在")
        return

    try:
        ok = adapter.uninstall_config()
        if ok:
            click.echo(click.style(f"✅ 已从 {adapter.agent_name} 移除", fg="green"))
        else:
            click.echo("⚠️  移除失败")
    except (OSError, IOError) as e:
        click.echo(f"❌ 卸载失败: {e}", err=True)
        raise SystemExit(1)


@hooks.command("status")
@click.option("--json", "json_output", is_flag=True)
@click.option("--agent", "-a",
    type=click.Choice(_AGENT_CHOICES, case_sensitive=False),
    default=None)
def hooks_status(json_output: bool, agent: str | None) -> None:
    """检查 hooks 安装状态。"""
    adapters = [adapter for adapter in get_all_adapters()]

    if agent:
        adapter = _resolve(agent)
        adapters = [adapter]

    results = {}
    for a in adapters:
        results[a.agent_id.value] = {
            "name": a.agent_name,
            "installed": a.is_installed(),
            "detected": a.detect(),
            "config": str(a.config.settings_path),
        }

    if json_output:
        click.echo(json.dumps(results, indent=2, ensure_ascii=False))
        return

    any_installed = any(r["installed"] for r in results.values())
    if any_installed:
        click.echo(click.style("✅ Hooks 已安装\n", bold=True, fg="green"))
    else:
        click.echo(click.style("📭 Hooks 未安装\n", bold=True, fg="yellow"))
        click.echo("  运行 cts agents setup 一键配置\n")

    for aid_val, info in results.items():
        icon = "✅" if info["installed"] else "❌"
        click.echo(f"  {icon} {info['name']} ({aid_val})")


@hooks.command("test")
@click.argument("event_file", required=False)
@click.option("--tool", "-t", help="模拟工具名称")
@click.option("--stdin", is_flag=True, help="从 stdin 读取事件")
@click.option("--agent", "-a",
    type=click.Choice(_AGENT_CHOICES, case_sensitive=False),
    default=None)
def hooks_test(
    agent: str | None,
    event_file: str | None,
    tool: str | None,
    stdin: bool,
) -> None:
    """测试 hook handler。

    \b
    示例:
      cts hooks test -t Read
      cts hooks test --agent codex -t read_file --stdin
    """
    adapter = _resolve(agent)

    # 未安装时自动提示
    if not adapter.is_installed() and not event_file and not stdin and not tool:
        click.echo(click.style(
            "⚠️  Hooks 未安装 → cts agents install",
            fg="yellow",
        ))
        if click.confirm("是否现在安装并测试?"):
            adapter.install_config()
            tool = "Read" if adapter.agent_id == AgentID.CLAUDE_CODE else "read_file"
        else:
            raise SystemExit(0)

    if stdin:
        event_json = sys.stdin.read()
    elif event_file:
        try:
            with open(event_file, "r", encoding="utf-8") as f:
                event_json = f.read()
        except (OSError, IOError) as e:
            click.echo(f"❌ 无法读取: {e}", err=True)
            raise SystemExit(1)
    elif tool:
        reverse_map = {v: k for k, v in adapter.config.tool_name_map.items()}
        raw_tool = reverse_map.get(tool, tool)
        raw_event = build_test_event(adapter, tool, raw_tool)
        event_json = json.dumps(raw_event, ensure_ascii=False)
    else:
        click.echo("❌ 请提供事件来源: --tool, --stdin, 或 event_file")
        raise SystemExit(1)

    click.echo(click.style("🧪 测试...\n", bold=True, fg="cyan"))

    try:
        event_data = json.loads(event_json)
    except json.JSONDecodeError as e:
        click.echo(f"❌ JSON 解析失败: {e}", err=True)
        raise SystemExit(1)

    from claude_token_saver.agents import process_event
    result = process_event(event_data, adapter)

    if result.get("allow", True):
        click.echo(click.style("✅ allow", fg="green"))
    else:
        click.echo(click.style("🚫 block", fg="red"))

    for key in ("message", "modified_input", "updated_input"):
        val = result.get(key)
        if val:
            click.echo(f"  {key}: {json.dumps(val, ensure_ascii=False)[:200]}")
