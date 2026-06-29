"""
TUI Dashboard - 实时 Token 消耗仪表盘

布局:
  左侧: 会话列表 + Token 进度条
  中间: Token 消耗趋势图
  右侧: 浪费 Top 5 + 节省建议
  底部: 状态栏

按键:
  q  退出
  r  立即刷新
"""
from __future__ import annotations

import sys
import time
import threading
import datetime
from typing import Any

from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.align import Align
from rich.rule import Rule
from rich.live import Live

from claude_token_saver.stats import AnalyticsEngine
from claude_token_saver.sessions import SessionManager


class TokenDashboard:
    """实时 Token 消耗仪表盘。"""

    # 进度条上限（用于计算百分比）
    TOKEN_LIMIT = 200_000

    def __init__(self, interval: int = 5):
        self.interval = interval
        self.console = Console()
        self.engine = AnalyticsEngine()
        self.session_mgr = SessionManager()
        self._stop_event = threading.Event()
        self._refresh_event = threading.Event()
        self._refresh_count = 0

    # ── 数据采集 ──────────────────────────────────────────────────────

    def _collect_data(self) -> dict[str, Any]:
        """采集所有面板所需数据。"""
        try:
            return {
                "sessions": self.session_mgr.list_sessions(),
                "top_sessions": self.engine.get_top_sessions(10),
                "trend": self.engine.get_token_trend(7),
                "waste": self.engine.get_waste_summary(),
                "opportunities": self.engine.get_savings_opportunities(),
                "stats": self.session_mgr.get_stats(),
            }
        except Exception as e:
            return {
                "error": str(e),
                "sessions": [],
                "top_sessions": [],
                "trend": [],
                "waste": [],
                "opportunities": [],
                "stats": {},
            }

    # ── 辅助方法 ──────────────────────────────────────────────────────

    def _progress_bar(self, pct: float, width: int = 10) -> str:
        """生成简易进度条字符串。"""
        filled = int(pct * width)
        empty = width - filled
        color = "green" if pct < 0.6 else ("yellow" if pct < 0.85 else "red")
        return f"[{color}]{'█' * filled}{'░' * empty}[/{color}]"

    # ── 面板构建 ──────────────────────────────────────────────────────

    def _build_session_panel(self, data: dict) -> Panel:
        """左侧：会话列表 + Token 进度条。"""
        if "error" in data:
            return Panel(
                Text(f"数据加载失败: {data['error']}", style="red"),
                title="[bold blue]💬 会话列表[/]",
                border_style="red",
            )

        table = Table(show_header=True, header_style="bold cyan", expand=True)
        table.add_column("会话", no_wrap=False, ratio=3)
        table.add_column("Token", justify="right", ratio=2)
        table.add_column("进度", ratio=3)

        sessions = data["sessions"]
        for s in sessions[:15]:
            pct = min(s.tokens_used / self.TOKEN_LIMIT, 1.0)
            bar = self._progress_bar(pct)
            token_style = "red" if s.tokens_used > 150_000 else "yellow"
            table.add_row(
                Text(s.title[:18], style="white"),
                Text(f"{s.tokens_used:,}", style=token_style),
                Text(bar),
            )

        if not sessions:
            table.add_row(Text("暂无会话", style="dim"), "", "")

        return Panel(table, title="[bold blue]💬 会话列表[/]", border_style="blue")

    def _build_trend_panel(self, data: dict) -> Panel:
        """中间：Token 消耗趋势图（简易柱状图）。"""
        if "error" in data:
            return Panel(
                Text(f"数据加载失败: {data['error']}", style="red"),
                title="[bold green]📈 消耗趋势[/]",
                border_style="red",
            )

        trend = data["trend"]
        if not trend:
            return Panel(
                Align.center(Text("暂无趋势数据", style="dim")),
                title="[bold green]📈 消耗趋势[/]",
                border_style="green",
            )

        max_tokens = max((t["total_tokens"] for t in trend), default=1) or 1
        bar_width = 18

        lines: list[str] = []
        for t in trend:
            day = t["day"][5:] if t["day"] else "??"
            tokens = t["total_tokens"] or 0
            bar_len = int(tokens / max_tokens * bar_width)
            bar = "█" * bar_len
            lines.append(f"{day}  {bar:<{bar_width}}  {tokens:>10,}")

        return Panel(
            "\n".join(lines),
            title="[bold green]📈 消耗趋势（最近 7 天）[/]",
            border_style="green",
            padding=(0, 1),
        )

    def _build_waste_panel(self, data: dict) -> Panel:
        """右侧：浪费 Top 5 + 节省建议。"""
        if "error" in data:
            return Panel(
                Text(f"数据加载失败: {data['error']}", style="red"),
                title="[bold red]⚠️  浪费 & 建议[/]",
                border_style="red",
            )

        # 浪费表格
        waste_table = Table(show_header=True, header_style="bold red", expand=True)
        waste_table.add_column("类别", no_wrap=False, ratio=2)
        waste_table.add_column("Tokens", justify="right", ratio=1)

        waste = data["waste"]
        for w in waste[:5]:
            cat = w["category"][:20]
            tw = w.get("total_wasted") or 0
            waste_table.add_row(Text(cat, style="white"), Text(f"{tw:,}", style="red"))

        if not waste:
            waste_table.add_row(Text("暂无浪费记录", style="dim"), "")

        # 节省建议
        opportunities = data["opportunities"][:3]
        sugg_table = Table(show_header=False, box=None, expand=True, padding=(0, 0))
        sugg_table.add_column("建议", ratio=1)

        if opportunities:
            for i, opp in enumerate(opportunities, 1):
                cat = opp.get("category", "")
                sugg = opp.get("suggestion", "")[:55]
                sugg_table.add_row(
                    f"[yellow]{i}.[/] [cyan]{cat}[/]\n   [white]{sugg}[/]"
                )
        else:
            sugg_table.add_row("[dim]暂无明显建议，继续保持！[/]")

        content = Table.grid(expand=True)
        content.add_row(waste_table)
        content.add_row(Rule(style="dim"))
        content.add_row(sugg_table)

        return Panel(
            content,
            title="[bold red]⚠️  浪费 & 建议[/]",
            border_style="red",
        )

    def _build_status_bar(self, data: dict) -> Panel:
        """底部状态栏。"""
        now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        stats = data.get("stats", {})
        total_sessions = stats.get("total_sessions", 0)
        total_tokens = stats.get("total_tokens", 0)

        text = (
            f"[cyan]⏰ {now}[/]  "
            f"[yellow]🔄 刷新 {self._refresh_count} 次[/]  "
            f"[green]💬 {total_sessions} 会话[/]  "
            f"[magenta]📊 {total_tokens:,} tokens[/]  "
            f"[white]间隔 {self.interval}s | q=退出 r=刷新[/]"
        )
        return Panel(Align.center(Text(text)), style="white on blue", height=3)

    # ── 布局组装 ──────────────────────────────────────────────────────

    def _build_layout(self, data: dict) -> Layout:
        """构建整体三栏布局。"""
        layout = Layout()
        layout.split_column(
            Layout(name="main", ratio=1),
            Layout(name="status", size=3),
        )
        layout["main"].split_row(
            Layout(name="left", ratio=3),
            Layout(name="center", ratio=4),
            Layout(name="right", ratio=3),
        )
        layout["left"].update(self._build_session_panel(data))
        layout["center"].update(self._build_trend_panel(data))
        layout["right"].update(self._build_waste_panel(data))
        layout["status"].update(self._build_status_bar(data))
        return layout

    # ── 键盘监听 ──────────────────────────────────────────────────────

    def _keyboard_listener(self) -> None:
        """后台线程：监听键盘输入（跨平台）。"""
        while not self._stop_event.is_set():
            key = None
            try:
                if sys.platform == "win32":
                    import msvcrt
                    if msvcrt.kbhit():
                        ch = msvcrt.getch()
                        try:
                            key = ch.decode("utf-8", errors="ignore").lower()
                        except Exception:
                            pass
                else:
                    import select
                    if sys.stdin in select.select([sys.stdin], [], [], 0.1)[0]:
                        key = sys.stdin.read(1).lower()
            except Exception:
                pass

            if key == "q":
                self._stop_event.set()
                break
            elif key == "r":
                self._refresh_event.set()

            time.sleep(0.05)

    # ── 主循环 ──────────────────────────────────────────────────────

    def run(self) -> None:
        """启动仪表盘主循环。"""
        self.console.show_cursor(False)

        listener = threading.Thread(target=self._keyboard_listener, daemon=True)
        listener.start()

        try:
            with Live(
                self._build_layout(self._collect_data()),
                console=self.console,
                screen=True,
                refresh_per_second=4,
            ) as live:
                last_refresh = time.monotonic()
                while not self._stop_event.is_set():
                    now = time.monotonic()
                    # 满足以下任一条件时刷新：
                    # 1. 到达自动刷新间隔
                    # 2. 用户按下 'r' 键
                    if self._refresh_event.is_set() or (now - last_refresh) >= self.interval:
                        if self._refresh_event.is_set():
                            self._refresh_event.clear()
                            self._refresh_count += 1
                        last_refresh = now
                        live.update(self._build_layout(self._collect_data()))
                    time.sleep(0.1)
        finally:
            self._stop_event.set()
            self.console.show_cursor(True)

    def stop(self) -> None:
        """外部请求停止仪表盘。"""
        self._stop_event.set()


__all__ = ["TokenDashboard"]
