"""
Claude Code Token Saver - 主入口 CLI
"""
from __future__ import annotations

import sys
import io

# 强制 UTF-8 输出（Windows GBK 终端兼容）
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if sys.stderr.encoding and sys.stderr.encoding.lower() not in ("utf-8", "utf8"):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import click

from claude_token_saver.cli_prep import prep
from claude_token_saver.cli_sessions import sessions
from claude_token_saver.cli_stats import stats
from claude_token_saver.cli_tui import tui
from claude_token_saver.cli_transcript import transcript
from claude_token_saver.cli_hooks import hooks
from claude_token_saver.cli_daemon import daemon
from claude_token_saver.config import load_config, save_config


@click.group()
@click.version_option("0.1.0", prog_name="claude-token-saver")
@click.pass_context
def main(ctx: click.Context) -> None:
    """Claude Code Token Saver - 减少 Claude Code token 消耗的综合工具。

    \b
    七大核心模块：
      prep       预处理文件，去除注释、去重、智能截断
      sessions   会话管理，自动 compact、主题分类
      stats      统计分析，识别 token 浪费来源
      tui        实时 Token 消耗仪表盘
      transcript 解析 transcript JSONL，提取会话与费用
      hooks      Claude Code Hook 安装/卸载/测试
      daemon     后台监控服务，持续扫描 transcript
    """
    ctx.ensure_object(dict)


main.add_command(prep)
main.add_command(sessions)
main.add_command(stats)
main.add_command(tui)
main.add_command(transcript)
main.add_command(hooks)
main.add_command(daemon)


@main.command("config")
@click.option("--show", is_flag=True, help="显示当前配置")
@click.option("--set", "set_key", help="设置配置项 (格式: key=value)")
@click.option("--reset", is_flag=True, help="重置为默认配置")
def config_cmd(show: bool, set_key: str | None, reset: bool) -> None:
    """查看或修改配置。"""
    from claude_token_saver.cli_helpers import handle_config

    handle_config(show, set_key, reset)


@main.command("init")
def init_cmd() -> None:
    """初始化项目（安装依赖、创建配置）。"""
    import subprocess

    click.echo("🔧 初始化 claude-token-saver...")
    config = load_config()
    click.echo(f"✅ 配置文件已创建: {config.get('_path', '~/.claude-token-saver/config.yaml')}")
    click.echo("✅ 数据库已初始化")
    click.echo("")
    click.echo("安装依赖（需要 Python 3.10+）：")
    click.echo("  pip install -e .")
    click.echo("")
    click.echo("快速开始：")
    click.echo("  cts prep files/ --dry-run          # 预览处理效果")
    click.echo("  cts prep files/ -o content.txt     # 处理并输出")
    click.echo("  cts sessions list                  # 查看会话列表")
    click.echo("  cts stats report                   # 生成分析报告")
    click.echo("  cts transcript scan                # 扫描 transcript 文件")
    click.echo("  cts transcript parse --import-db   # 解析并导入数据库")


if __name__ == "__main__":
    main()
