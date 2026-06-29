"""
stats 子命令 - 统计分析
"""
from __future__ import annotations

import sys

import click

from claude_token_saver.stats import AnalyticsEngine
from claude_token_saver.sessions import SessionManager


@click.group()
def stats() -> None:
    """分析 token 消耗，识别浪费，提供优化建议。"""
    pass


@stats.command("report")
@click.option("--days", type=int, default=7, help="统计最近 N 天")
@click.option("--json", "json_output", is_flag=True, help="JSON 格式输出")
def stats_report(days: int, json_output: bool) -> None:
    """生成综合报告。"""
    engine = AnalyticsEngine()
    mgr = SessionManager()
    session_stats = mgr.get_stats()

    heatmap = engine.get_file_read_heatmap(limit=15)
    trend = engine.get_token_trend(days=days)
    top_sessions = engine.get_top_sessions(limit=10)
    waste = engine.get_waste_summary()
    opportunities = engine.get_savings_opportunities()

    if json_output:
        report = {
            "sessions": session_stats,
            "file_heatmap": heatmap,
            "token_trend": trend,
            "top_sessions": top_sessions,
            "waste_summary": waste,
            "savings_opportunities": opportunities,
        }
        click.echo(click.style(str(report), fg="cyan"))
        return

    # 文本报告
    click.echo(click.style("=" * 60, fg="cyan"))
    click.echo(click.style("  📊 Claude Code Token 分析报告", bold=True, fg="cyan"))
    click.echo(click.style("=" * 60, fg="cyan"))

    # 概览
    click.echo(f"\n{click.style('📋 概览', bold=True, fg='yellow')}")
    click.echo(f"   总会话数:  {session_stats['total_sessions']}")
    click.echo(f"   总 Token:   {session_stats['total_tokens']:,}")
    click.echo(f"   总轮次:     {session_stats['total_turns']}")
    click.echo(f"   Compact 数: {session_stats['total_compacts']}")

    # 趋势
    if trend:
        click.echo(f"\n{click.style('📈 消耗趋势（最近 {days} 天）', bold=True, fg='yellow')}")
        for t in trend:
            bar = "█" * min(int(t["total_tokens"] / 10000), 30) if t["total_tokens"] else ""
            click.echo(f"   {t['day']}  {t['total_tokens']:>10,} tokens  {bar}")

    # 文件读取热力图
    if heatmap:
        click.echo(f"\n{click.style('🔥 文件读取热力图（Top 15）', bold=True, fg='yellow')}")
        click.echo(f"   {'文件':<45} {'Token':>10} {'次数':>6}")
        click.echo(f"   {'─' * 45} {'─' * 10} {'─' * 6}")
        for f in heatmap:
            bar = "▓" * min(int(f["total_tokens"] / 5000), 20)
            click.echo(f"   {f['file_path'][:43]:<45} {f['total_tokens']:>10,} {f['read_count']:>6}  {bar}")

    # Top 会话
    if top_sessions:
        click.echo(f"\n{click.style('🏆 高消耗会话（Top 10）', bold=True, fg='yellow')}")
        click.echo(f"   {'会话 ID':<12} {'Token':>12} {'轮次':>6}")
        click.echo(f"   {'─' * 12} {'─' * 12} {'─' * 6}")
        for s in top_sessions:
            click.echo(f"   {s['session_id']:<12} {s['total_tokens']:>12,} {s['turns']:>6}")

    # 浪费摘要
    if waste:
        click.echo(f"\n{click.style('⚠️  浪费摘要', bold=True, fg='red')}")
        for w in waste:
            click.echo(f"   {w['category']:<20} {w['count']:>5} 次  {w['total_wasted'] or 0:>10,} tokens")

    # 节省建议
    if opportunities:
        click.echo(f"\n{click.style('💡 节省建议', bold=True, fg='green')}")
        for i, opp in enumerate(opportunities, 1):
            click.echo(f"   {i}. [{opp['category']}]")
            if "file" in opp:
                click.echo(f"      📄 {opp['file']}")
            if "session" in opp:
                click.echo(f"      💬 会话 {opp['session']}")
            click.echo(f"      → {opp['suggestion']}")
            if "tokens" in opp:
                click.echo(f"        影响: {opp['tokens']:,} tokens\n")
            else:
                click.echo()

    # 估算费用
    total_tokens = session_stats["total_tokens"]
    cost_claude_sonnet = total_tokens * 3 / 1_000_000  # $3/MTok
    cost_claude_opus = total_tokens * 15 / 1_000_000  # $15/MTok
    click.echo(f"{click.style('💰 费用估算', bold=True, fg='magenta')}")
    click.echo(f"   Claude Sonnet: ~${cost_claude_sonnet:.2f} (${cost_claude_sonnet * 7.2:.2f} CNY)")
    click.echo(f"   Claude Opus:    ~${cost_claude_opus:.2f} (${cost_claude_opus * 7.2:.2f} CNY)")
    click.echo()


@stats.command("files")
@click.option("--limit", "-n", type=int, default=20, help="显示条数")
@click.option("--min-tokens", type=int, default=0, help="最小 token 阈值")
def stats_files(limit: int, min_tokens: int) -> None:
    """文件读取分析。"""
    engine = AnalyticsEngine()
    heatmap = engine.get_file_read_heatmap(limit=limit)
    heatmap = [f for f in heatmap if f["total_tokens"] >= min_tokens]

    if not heatmap:
        click.echo("📭 暂无数据，需要先在 Claude Code 中使用并追踪文件读取")
        return

    click.echo(click.style(f"🔥 文件读取热力图（Top {len(heatmap)}）\n", bold=True, fg="cyan"))
    click.echo(f"   {'文件':<50} {'Token':>10} {'次数':>6} {'大小':>10}")
    click.echo(f"   {'─' * 50} {'─' * 10} {'─' * 6} {'─' * 10}")

    total_tokens = 0
    for f in heatmap:
        size_kb = f["total_size"] / 1024 if f["total_size"] else 0
        click.echo(f"   {f['file_path'][:48]:<50} {f['total_tokens']:>10,} {f['read_count']:>6} "
                   f"{size_kb:>9.1f}K")
        total_tokens += f["total_tokens"]

    click.echo(f"\n   总计: {total_tokens:,} tokens")


@stats.command("trend")
@click.option("--days", type=int, default=14, help="天数")
def stats_trend(days: int) -> None:
    """Token 消耗趋势。"""
    engine = AnalyticsEngine()
    trend = engine.get_token_trend(days=days)

    if not trend:
        click.echo("📭 暂无数据")
        return

    click.echo(click.style(f"📈 Token 趋势（最近 {days} 天）\n", bold=True, fg="cyan"))
    click.echo(f"   {'日期':<12} {'Token':>12}  可视化")
    click.echo(f"   {'─' * 12} {'─' * 12}  {'─' * 30}")

    max_tokens = max((t["total_tokens"] for t in trend), default=1)
    for t in trend:
        bar_len = int(t["total_tokens"] / max_tokens * 30) if max_tokens else 0
        bar = "█" * bar_len
        click.echo(f"   {t['day']:<12} {t['total_tokens']:>12,}  {bar}")


@stats.command("suggest")
def stats_suggest() -> None:
    """获取节省建议。"""
    engine = AnalyticsEngine()
    opportunities = engine.get_savings_opportunities()

    if not opportunities:
        click.echo("✅ 暂未发现明显的 token 浪费，继续保持！")
        return

    click.echo(click.style(f"💡 发现 {len(opportunities)} 个节省机会\n", bold=True, fg="green"))

    for i, opp in enumerate(opportunities, 1):
        tokens = opp.get("tokens", 0)
        click.echo(f"   {click.style(f'#{i}', fg='cyan')} [{click.style(opp['category'], bold=True)}]")
        if "file" in opp:
            click.echo(f"      📄 {opp['file']}")
        if "session" in opp:
            click.echo(f"      💬 {opp['session']}")
        click.echo(f"      → {opp['suggestion']}")
        if tokens:
            click.echo(f"      💰 潜在节省: ~{tokens:,} tokens")
        click.echo()


@stats.command("cost")
@click.option("--days", type=int, default=30, help="统计天数")
@click.option("--model", type=click.Choice(["sonnet", "opus", "haiku"]), default="sonnet", help="模型选择")
def stats_cost(days: int, model: str) -> None:
    """费用估算。"""
    engine = AnalyticsEngine()
    mgr = SessionManager()
    session_stats = mgr.get_stats()
    trend = engine.get_token_trend(days=days)
    total = sum(t["total_tokens"] for t in trend) if trend else session_stats["total_tokens"]

    prices = {
        "sonnet": 3.0,
        "opus": 15.0,
        "haiku": 0.8,
    }
    price_per_mtok = prices[model]

    input_cost = total * price_per_mtok / 1_000_000
    output_estimate = total * 0.3  # 假设输出是输入的 30%
    output_cost = output_estimate * price_per_mtok / 1_000_000
    total_cost = input_cost + output_cost

    click.echo(click.style(f"💰 {model.upper()} 费用估算（最近 {days} 天）\n", bold=True, fg="magenta"))
    click.echo(f"   总 Token:        {total:,}")
    click.echo(f"   输入 Token:      {total:,}")
    click.echo(f"   输出 Token:      ~{int(output_estimate):,}")
    click.echo(f"   输入费用:        ${input_cost:.4f} (¥{input_cost * 7.2:.2f})")
    click.echo(f"   输出费用:        ${output_cost:.4f} (¥{output_cost * 7.2:.2f})")
    click.echo(f"   总费用:          ${total_cost:.4f} (¥{total_cost * 7.2:.2f})")
    click.echo(f"   日均费用:        ${total_cost / days:.4f} (¥{total_cost / days * 7.2:.2f})")
    click.echo(f"\n   💡 启用 prep 工具可节省 30-60% token，对应费用减半")
