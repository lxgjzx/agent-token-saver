"""
transcript 子命令 - 解析 Claude Code transcript JSONL 文件
仅使用标准库：json, pathlib, sqlite3, datetime。
"""
from __future__ import annotations

import json
import sys

import click

from claude_token_saver.transcript import TranscriptParser


@click.group()
def transcript() -> None:
    """解析 Claude Code transcript，提取会话、轮次、工具调用、Token 消耗和费用。"""
    pass


@transcript.command("scan")
@click.option("--projects-dir", help="Claude projects 目录（默认 ~/.claude/projects/）")
@click.option("--json", "json_output", is_flag=True, help="JSON 格式输出")
def transcript_scan(projects_dir: str | None, json_output: bool) -> None:
    """扫描 projects 目录，列出所有 transcript 文件。"""
    parser = TranscriptParser(
        projects_dir=Path(projects_dir) if projects_dir else None
    )
    if not parser.projects_dir.exists():
        click.echo("📭 projects 目录不存在")
        return

    files = sorted(parser.projects_dir.rglob("*.jsonl"))
    files = [f for f in files if "subagents" not in f.parts]

    if not files:
        click.echo("📭 未找到 transcript 文件")
        return

    if json_output:
        output = []
        for f in files:
            output.append({
                "path": str(f),
                "session_id": f.stem,
                "size_kb": round(f.stat().st_size / 1024, 1),
            })
        click.echo(click.style(json.dumps(output, ensure_ascii=False, indent=2), fg="cyan"))
        return

    click.echo(click.style(f"📂 找到 {len(files)} 个 transcript 文件\n", bold=True, fg="cyan"))
    click.echo(f"   {'Session ID':<36} {'大小':>10} 路径")
    click.echo(f"   {'─' * 36} {'─' * 10} {'─' * 40}")
    total_size = 0
    for f in files:
        size_kb = f.stat().st_size / 1024
        total_size += size_kb
        size_str = f"{size_kb:.1f}K" if size_kb < 1024 else f"{size_kb / 1024:.1f}M"
        rel = f.relative_to(parser.projects_dir)
        click.echo(f"   {f.stem:<36} {size_str:>10} {rel}")
    click.echo(f"\n   总计: {len(files)} 个文件, {total_size / 1024:.1f}M")


@transcript.command("parse")
@click.argument("path", required=False)
@click.option("--projects-dir", help="Claude projects 目录")
@click.option("--json", "json_output", is_flag=True, help="JSON 格式输出")
@click.option("--import-db", "import_db", is_flag=True, help="导入到 analytics DB")
def transcript_parse(
    path: str | None,
    projects_dir: str | None,
    json_output: bool,
    import_db: bool,
) -> None:
    """解析 transcript 文件并显示会话/轮次/工具调用摘要。"""
    from pathlib import Path

    parser = TranscriptParser(
        projects_dir=Path(projects_dir) if projects_dir else None
    )

    # 确定要解析的文件列表
    if path:
        target = Path(path)
        if not target.exists():
            click.echo(f"❌ 路径不存在: {path}", err=True)
            sys.exit(1)
        candidates = [target] if target.is_file() else sorted(target.rglob("*.jsonl"))
        files = [f for f in candidates if f.is_file() and "subagents" not in f.parts]
    else:
        if not parser.projects_dir.exists():
            click.echo("📭 projects 目录不存在")
            return
        files = sorted(parser.projects_dir.rglob("*.jsonl"))
        files = [f for f in files if "subagents" not in f.parts]

    if not files:
        click.echo("📭 未找到 transcript 文件")
        return

    parsed_results: list[tuple] = []
    for f in files:
        result = parser.parse_file(f)
        if result:
            parsed_results.append(result)

    if not parsed_results:
        click.echo("📭 未能解析任何文件")
        return

    if import_db:
        count = parser.import_to_db(parsed_results)
        click.echo(f"✅ 已导入 {len(parsed_results)} 个会话到 analytics DB")
        return

    if json_output:
        output = []
        for session, turns in parsed_results:
            output.append({
                "session": session.to_dict(),
                "turns": [t.to_dict() for t in turns],
            })
        click.echo(click.style(json.dumps(output, ensure_ascii=False, indent=2), fg="cyan"))
        return

    # 文本输出
    for session, turns in parsed_results:
        click.echo("")
        click.echo(click.style(f"📋 会话: {session.id[:12]}...", bold=True, fg="cyan"))
        if session.title:
            click.echo(f"   标题: {session.title}")
        click.echo(f"   目录: {session.work_dir or '(unknown)'}")
        if session.model:
            click.echo(f"   模型: {session.model}")
        click.echo(f"   轮次: {session.turn_count}")
        click.echo(
            f"   Token: 输入 {session.total_input_tokens:,} | 输出 {session.total_output_tokens:,}"
        )

        for t in turns[:10]:
            preview = t.content[:60].replace("\n", " ")
            parts = [f"[{t.turn_index}] {preview}"]
            if t.tool_uses:
                parts.append(f"工具:{len(t.tool_uses)}")
            if t.total_tokens:
                parts.append(f"{t.total_tokens:,}t")
            click.echo(f"   {' | '.join(parts)}")

        if len(turns) > 10:
            click.echo(f"   ... 还有 {len(turns) - 10} 个轮次")

    click.echo("")
    total_turns = sum(len(turns) for _, turns in parsed_results)
    total_tools = sum(
        len(t.tool_uses)
        for _, turns in parsed_results
        for t in turns
    )
    click.echo(
        f"📊 汇总: {len(parsed_results)} 个会话, {total_turns} 轮, {total_tools} 次工具调用"
    )


@transcript.command("history")
@click.option("--session-id", help="过滤特定会话 ID")
@click.option("--days", type=int, default=30, help="最近 N 天")
@click.option("--json", "json_output", is_flag=True, help="JSON 格式输出")
@click.option("--limit", "-n", type=int, default=20, help="显示条数")
def transcript_history(
    session_id: str | None,
    days: int,
    json_output: bool,
    limit: int,
) -> None:
    """查看 transcript 解析后的 token 消耗历史。"""
    import sqlite3
    from pathlib import Path

    from claude_token_saver.stats import AnalyticsEngine

    engine = AnalyticsEngine()

    try:
        conn = sqlite3.connect(engine.db_path)
        conn.row_factory = sqlite3.Row

        query = """
            SELECT session_id, turn_id, input_tokens, output_tokens,
                   cache_creation_input_tokens, cache_read_input_tokens,
                   total_tokens, model, timestamp, service_tier
            FROM transcript_usage_records
            WHERE timestamp >= datetime('now', ?)
        """
        params: list = [f"-{days} days"]

        if session_id:
            query += " AND session_id = ?"
            params.append(session_id)

        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        rows = conn.execute(query, params).fetchall()
        conn.close()
    except sqlite3.OperationalError:
        rows = []

    if not rows:
        click.echo("📭 暂无 transcript 数据，请先运行 `cts transcript parse --import-db`")
        return

    if json_output:
        output = [dict(row) for row in rows]
        click.echo(click.style(json.dumps(output, ensure_ascii=False, indent=2), fg="cyan"))
        return

    click.echo(
        click.style(f"📈 Token 消耗历史（最近 {days} 天，Top {len(rows)}）\n", bold=True, fg="cyan")
    )
    click.echo(
        f"   {'会话':<12} {'时间':<20} {'输入':>10} {'输出':>10} {'总计':>12} {'模型'}"
    )
    click.echo(
        f"   {'─' * 12} {'─' * 20} {'─' * 10} {'─' * 10} {'─' * 12} {'─' * 15}"
    )

    for row in rows:
        sid = (row["session_id"] or "")[:10]
        ts = (row["timestamp"] or "")[:19].replace("T", " ")
        model = row["model"] or "unknown"
        click.echo(
            f"   {sid:<12} {ts:<20} {row['input_tokens']:>10,} {row['output_tokens']:>10,} "
            f"{row['total_tokens']:>12,} {model}"
        )

    total_in = sum(r["input_tokens"] for r in rows)
    total_out = sum(r["output_tokens"] for r in rows)
    total = sum(r["total_tokens"] for r in rows)
    click.echo(f"\n   合计: {total_in:,} 输入 + {total_out:,} 输出 = {total:,} tokens")


@transcript.command("tools")
@click.option("--session-id", help="过滤特定会话 ID")
@click.option("--limit", "-n", type=int, default=20, help="显示条数")
@click.option("--json", "json_output", is_flag=True, help="JSON 格式输出")
def transcript_tools(session_id: str | None, limit: int, json_output: bool) -> None:
    """查看工具调用统计。"""
    import sqlite3

    from claude_token_saver.stats import AnalyticsEngine

    engine = AnalyticsEngine()

    try:
        conn = sqlite3.connect(engine.db_path)
        conn.row_factory = sqlite3.Row

        query = """
            SELECT name, COUNT(*) as call_count,
                   SUM(CASE WHEN is_error THEN 1 ELSE 0 END) as error_count
            FROM transcript_tool_uses
        """
        params: list = []

        if session_id:
            query += """
                WHERE turn_id IN (
                    SELECT id FROM transcript_turns WHERE session_id = ?
                )
            """
            params.append(session_id)

        query += " GROUP BY name ORDER BY call_count DESC LIMIT ?"
        params.append(limit)

        rows = conn.execute(query, params).fetchall()
        conn.close()
    except sqlite3.OperationalError:
        rows = []

    if not rows:
        click.echo("📭 暂无工具调用数据，请先运行 `cts transcript parse --import-db`")
        return

    if json_output:
        output = [dict(row) for row in rows]
        click.echo(click.style(json.dumps(output, ensure_ascii=False, indent=2), fg="cyan"))
        return

    click.echo(click.style(f"🔧 工具调用统计（Top {len(rows)}）\n", bold=True, fg="cyan"))
    click.echo(f"   {'工具名':<20} {'调用':>8} {'错误':>6}")
    click.echo(f"   {'─' * 20} {'─' * 8} {'─' * 6}")
    for row in rows:
        err = f"⚠️  {row['error_count']}" if row["error_count"] else "✅"
        click.echo(f"   {row['name']:<20} {row['call_count']:>8} {err:>6}")


@transcript.command("cost")
@click.option("--session-id", help="过滤特定会话 ID")
@click.option("--days", type=int, default=30, help="统计天数")
@click.option("--model", type=click.Choice(["sonnet", "opus", "haiku"]), help="按模型定价计算")
@click.option("--json", "json_output", is_flag=True, help="JSON 格式输出")
def transcript_cost(
    session_id: str | None,
    days: int,
    model: str | None,
    json_output: bool,
) -> None:
    """费用估算。"""
    import sqlite3

    from claude_token_saver.stats import AnalyticsEngine

    engine = AnalyticsEngine()

    try:
        conn = sqlite3.connect(engine.db_path)
        conn.row_factory = sqlite3.Row

        query = """
            SELECT session_id, SUM(input_tokens) as input_tokens,
                   SUM(output_tokens) as output_tokens,
                   SUM(total_tokens) as total_tokens,
                   model
            FROM transcript_usage_records
            WHERE timestamp >= datetime('now', ?)
        """
        params: list = [f"-{days} days"]

        if session_id:
            query += " AND session_id = ?"
            params.append(session_id)

        query += " GROUP BY session_id, model ORDER BY total_tokens DESC"
        rows = conn.execute(query, params).fetchall()
        conn.close()
    except sqlite3.OperationalError:
        rows = []

    if not rows:
        click.echo("📭 暂无 transcript 数据，请先运行 `cts transcript parse --import-db`")
        return

    prices = {
        "sonnet": {"input": 3.0, "output": 15.0},
        "opus":   {"input": 15.0, "output": 75.0},
        "haiku":  {"input": 0.8,  "output": 4.0},
    }

    if json_output:
        output = []
        total_usd = 0.0
        for row in rows:
            m = row["model"] or "sonnet"
            p = prices.get(model or m.lower().split("-")[0] if m else "sonnet", prices["sonnet"])
            inp = row["input_tokens"] or 0
            out = row["output_tokens"] or 0
            ic = inp * p["input"] / 1_000_000
            oc = out * p["output"] / 1_000_000
            tc = ic + oc
            total_usd += tc
            output.append({
                "session_id": row["session_id"],
                "model": m,
                "input_tokens": inp,
                "output_tokens": out,
                "total_tokens": row["total_tokens"],
                "input_cost_usd": round(ic, 4),
                "output_cost_usd": round(oc, 4),
                "total_cost_usd": round(tc, 4),
            })
        click.echo(click.style(json.dumps(output, ensure_ascii=False, indent=2), fg="cyan"))
        return

    click.echo(
        click.style(f"💰 Transcript 费用估算（最近 {days} 天）\n", bold=True, fg="magenta")
    )

    total_usd = 0.0
    for row in rows:
        m = row["model"] or "sonnet"
        p = prices.get(model or m.lower().split("-")[0] if m else "sonnet", prices["sonnet"])
        inp = row["input_tokens"] or 0
        out = row["output_tokens"] or 0
        ic = inp * p["input"] / 1_000_000
        oc = out * p["output"] / 1_000_000
        tc = ic + oc
        total_usd += tc
        click.echo(f"   会话 {row['session_id'][:10]}... ({m})")
        click.echo(f"      输入: {inp:,} → ${ic:.4f}  |  输出: {out:,} → ${oc:.4f}  |  合计: ${tc:.4f}")

    click.echo(f"\n   总费用: ${total_usd:.4f} (¥{total_usd * 7.2:.2f})")
