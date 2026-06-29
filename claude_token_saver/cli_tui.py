"""
tui 子命令 - 实时 Token 消耗仪表盘
"""
from __future__ import annotations

import sys
import io

import click

from claude_token_saver.tui import TokenDashboard


@click.command()
@click.option(
    "--interval", "-i",
    type=int,
    default=5,
    help="自动刷新间隔（秒），默认 5",
)
def tui(interval: int) -> None:
    """启动实时 Token 消耗仪表盘（TUI）。

    \b
    按键说明:
      q  退出仪表盘
      r  立即刷新数据
    """
    # 强制 UTF-8 输出（Windows GBK 终端兼容）
    if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    if sys.stderr.encoding and sys.stderr.encoding.lower() not in ("utf-8", "utf8"):
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

    if interval < 1:
        click.echo("❌ 刷新间隔不能小于 1 秒")
        raise SystemExit(1)

    dashboard = TokenDashboard(interval=interval)
    try:
        dashboard.run()
    except KeyboardInterrupt:
        pass
