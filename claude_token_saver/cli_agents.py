"""
agents 子命令 - 多 Agent 适配器管理

便捷特性:
  - `agents setup` 一键完成检测 + 安装 + daemon
  - `agents test` 未安装时自动提示安装
  - `agents check` 快速健康检查
"""
from __future__ import annotations

import json
import sys

import click

from claude_token_saver.agents import (
    build_test_event,
    detect_agent,
    get_adapter,
    get_all_adapters,
    resolve_adapter,
)
from claude_token_saver.agents.base import AgentID


def _resolve_adapter(agent_name: str | None = None):
    """CLI 专用包装：将 ValueError 转为 click.BadParameter。"""
    try:
        return resolve_adapter(agent_name)
    except ValueError as e:
        raise click.BadParameter(str(e))


_AGENT_CHOICES = [a.value for a in AgentID]


@click.group()
def agents() -> None:
    """多 Agent 适配器管理。"""
    pass


@agents.command("list")
@click.option("--json", "json_output", is_flag=True)
def agents_list(json_output: bool) -> None:
    """列出所有支持的 Agent。"""
    adapters = get_all_adapters()
    if json_output:
        data = []
        for a in adapters:
            data.append({
                "id": a.agent_id.value,
                "name": a.agent_name,
                "installed": a.is_installed(),
                "detected": a.detect(),
                "config": str(a.config.settings_path),
            })
        click.echo(json.dumps(data, indent=2, ensure_ascii=False))
        return

    click.echo(click.style("  Agent            状态      ", bold=True))
    click.echo("  " + "─" * 40)
    for a in adapters:
        icon = "✅" if a.is_installed() else "❌"
        det = " 🔍" if a.detect() else ""
        click.echo(f"  {a.agent_name:12s} {icon} 已安装{det}")
        click.echo(f"    ID:   {a.agent_id.value}")
        click.echo(f"    配置: {a.config.settings_path}")
        click.echo("")


@agents.command("detect")
def agents_detect() -> None:
    """自动检测当前运行的 Agent。"""
    adapter = detect_agent()
    if adapter:
        click.echo(click.style(
            f"🔍 检测到: {adapter.agent_name} ({adapter.agent_id.value})",
            fg="green",
        ))
        click.echo(f"   配置:  {adapter.config.settings_path}")
        click.echo(f"   格式:  {adapter.config.hook_format}")
        click.echo(f"   已安装: {'是' if adapter.is_installed() else '否'}")
    else:
        click.echo(click.style("❌ 未检测到任何支持的 Agent", fg="yellow"))
        click.echo(f"   可用选项: {', '.join(_AGENT_CHOICES)}")


@agents.command("check")
def agents_check() -> None:
    """快速健康检查：检测环境、hooks 状态、daemon 状态。"""
    click.echo(click.style("🏥 健康检查\n", bold=True))

    # Agent 检测
    adapter = detect_agent()
    if adapter:
        click.echo(click.style(f"  ✅ Agent: {adapter.agent_name}", fg="green"))
    else:
        click.echo(click.style("  ⚠️  Agent: 未检测到（将使用 Claude Code 默认值）", fg="yellow"))
        adapter = get_adapter(AgentID.CLAUDE_CODE)

    # Hooks 状态
    installed = adapter.is_installed()
    if installed:
        click.echo(click.style(f"  ✅ Hooks: 已安装 ({adapter.config.settings_path})", fg="green"))
    else:
        click.echo(click.style(f"  ❌ Hooks: 未安装 → 运行 cts agents install", fg="red"))

    # Daemon 状态
    try:
        from claude_token_saver.daemon import get_daemon_status
        status = get_daemon_status()
        if status.get("running"):
            click.echo(click.style(f"  ✅ Daemon: 运行中 (PID {status.get('pid')})", fg="green"))
        else:
            click.echo(click.style("  ❌ Daemon: 未运行 → 运行 cts daemon start", fg="red"))
    except Exception:
        click.echo(click.style("  ⚠️  Daemon: 无法检查", fg="yellow"))

    click.echo("")


@agents.command("setup")
@click.option("--agent", "-a",
    type=click.Choice(_AGENT_CHOICES, case_sensitive=False),
    default=None, help="指定 Agent（默认: 自动检测）")
@click.option("--dry-run", is_flag=True, help="只显示操作，不实际执行")
@click.option("--skip-daemon", is_flag=True, help="不启动 daemon")
@click.option("--skip-hooks", is_flag=True, help="不安装 hooks")
def agents_setup(agent: str | None, dry_run: bool, skip_daemon: bool, skip_hooks: bool) -> None:
    """一键完成初始化：检测 Agent → 安装 hooks → 启动 daemon。"""
    from claude_token_saver.daemon import start_daemon

    # 1. 检测
    click.echo(click.style("🔍 检测环境...", bold=True))
    adapter = _resolve_adapter(agent)
    click.echo(click.style(f"  → {adapter.agent_name} ({adapter.agent_id.value})", fg="cyan"))

    # 2. 安装 hooks
    if not skip_hooks:
        click.echo("")
        click.echo(click.style("🔧 安装 hooks...", bold=True))
        if dry_run:
            click.echo(f"  目标: {adapter.config.settings_path}")
            click.echo("  (dry-run 模式)")
        else:
            try:
                path = adapter.install_config()
                click.echo(click.style(f"  ✅ {path}", fg="green"))
            except Exception as e:
                click.echo(f"  ❌ 安装失败: {e}", err=True)

    # 3. 启动 daemon
    if not skip_daemon and not dry_run:
        click.echo("")
        click.echo(click.style("📡 启动 daemon...", bold=True))
        try:
            if start_daemon():
                click.echo(click.style("  ✅ 已启动", fg="green"))
            else:
                click.echo("  ℹ️  已在运行中")
        except Exception as e:
            click.echo(f"  ⚠️  {e}", fg="yellow")

    click.echo("")
    click.echo(click.style("🎉 完成!", bold=True, fg="green"))


@agents.command("install")
@click.option("--agent", "-a",
    type=click.Choice(_AGENT_CHOICES, case_sensitive=False),
    default=None)
@click.option("--dry-run", is_flag=True)
def agents_install(agent: str | None, dry_run: bool) -> None:
    """安装 token-saver hooks。"""
    adapter = _resolve_adapter(agent)

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
        click.echo(click.style(f"✅ 已安装: {path}", fg="green"))
    except (OSError, IOError) as e:
        click.echo(f"❌ 安装失败: {e}", err=True)
        raise SystemExit(1)


@agents.command("uninstall")
@click.option("--agent", "-a",
    type=click.Choice(_AGENT_CHOICES, case_sensitive=False),
    default=None)
@click.option("--dry-run", is_flag=True)
def agents_uninstall(agent: str | None, dry_run: bool) -> None:
    """移除 token-saver hooks。"""
    adapter = _resolve_adapter(agent)

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


@agents.command("test")
@click.argument("event_file", required=False)
@click.option("--tool", "-t", help="模拟工具名称（如 Read, read_file）")
@click.option("--stdin", is_flag=True, help="从 stdin 读取事件")
@click.option("--agent", "-a",
    type=click.Choice(_AGENT_CHOICES, case_sensitive=False),
    default=None)
def agents_test(
    agent: str | None,
    event_file: str | None,
    tool: str | None,
    stdin: bool,
) -> None:
    """测试 hook handler。

    \b
    示例:
      cts agents test -t Read
      cts agents test --agent codex -t read_file
      echo '{"tool": "read", "type": "pre_tool"}' | cts agents test --stdin
    """
    adapter = _resolve_adapter(agent)

    # 未安装时提示
    if not adapter.is_installed() and not event_file and not stdin:
        click.echo(click.style(
            "⚠️  Hooks 未安装，请先运行: cts agents install",
            fg="yellow",
        ))
        if click.confirm("是否现在安装?"):
            adapter.install_config()
            click.echo(click.style("✅ 已安装，继续测试...\n", fg="green"))
        else:
            raise SystemExit(0)

    # 读取事件
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
    click.echo(f"Agent: {adapter.agent_name} ({adapter.agent_id.value})")

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
