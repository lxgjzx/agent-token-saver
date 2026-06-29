"""
sessions 子命令 - 会话管理
"""
from __future__ import annotations

import sys
from pathlib import Path

import click

from claude_token_saver.sessions import SessionManager


@click.group()
def sessions() -> None:
    """管理 Claude Code 会话，支持主题分类和 compact 追踪。"""
    pass


@sessions.command("list")
@click.option("--topic", "-t", help="按主题过滤")
@click.option("--json", "json_output", is_flag=True, help="JSON 格式输出")
def sessions_list(topic: str | None, json_output: bool) -> None:
    """列出所有会话。"""
    mgr = SessionManager()
    items = mgr.list_sessions(topic=topic)

    if not items:
        click.echo("📭 暂无会话记录")
        return

    if json_output:
        click.echo(click.style(json.dumps([s.to_dict() for s in items], ensure_ascii=False, indent=2), fg="cyan"))
        return

    click.echo(click.style(f"📋 会话列表 ({len(items)} 个)\n", bold=True, fg="cyan"))
    click.echo(f"   {'ID':<10} {'主题':<15} {'标题':<30} {'Token':>12} {'轮次':>6} {'状态'}")
    click.echo(f"   {'─' * 10} {'─' * 15} {'─' * 30} {'─' * 12} {'─' * 6} {'─' * 10}")

    for s in items:
        status = "🔁" if s.compacted else "🟢"
        click.echo(f"   {s.id:<10} {s.topic:<15} {s.title[:28]:<30} {s.tokens_used:>12,} {s.turns:>6} {status}")

    # 主题统计
    topics = mgr.list_topics()
    if topics:
        click.echo(f"\n   📂 主题: {', '.join(topics)}")


@sessions.command("create")
@click.argument("title")
@click.option("--topic", "-t", default="default", help="主题标签")
def sessions_create(title: str, topic: str) -> None:
    """创建新会话。"""
    mgr = SessionManager()
    s = mgr.create_session(title=title, topic=topic)
    click.echo(f"✅ 会话已创建: {s.id} ({topic})")


@sessions.command("info")
@click.argument("session_id")
def sessions_info(session_id: str) -> None:
    """查看会话详情。"""
    mgr = SessionManager()
    s = mgr.get_session(session_id)
    if not s:
        click.echo(f"❌ 未找到会话: {session_id}", err=True)
        sys.exit(1)

    click.echo(click.style(f"📋 会话详情: {s.id}\n", bold=True, fg="cyan"))
    click.echo(f"   标题:   {s.title}")
    click.echo(f"   主题:   {s.topic}")
    click.echo(f"   Token:  {s.tokens_used:,}")
    click.echo(f"   轮次:   {s.turns}")
    click.echo(f"   状态:   {'已 compact' if s.compacted else '活跃'}")
    click.echo(f"   创建:   {s.created_at}")
    click.echo(f"   更新:   {s.updated_at}")

    history = mgr.get_compact_history(session_id)
    if history:
        click.echo(f"\n   📜 Compact 历史 ({len(history)} 次):")
        for h in history:
            saved = h["tokens_before"] - h["tokens_after"]
            click.echo(f"      {h['timestamp']}  {h['tokens_before']:,} → {h['tokens_after']:,} "
                       f"(节省 {saved:,})")


@sessions.command("compact-log")
@click.argument("session_id")
@click.option("--tokens-before", type=int, required=True, help="compact 前的 token 数")
@click.option("--tokens-after", type=int, required=True, help="compact 后的 token 数")
def sessions_compact_log(session_id: str, tokens_before: int, tokens_after: int) -> None:
    """记录一次 compact 操作。"""
    mgr = SessionManager()
    s = mgr.get_session(session_id)
    if not s:
        click.echo(f"❌ 未找到会话: {session_id}", err=True)
        sys.exit(1)

    saved = tokens_before - tokens_after
    mgr.log_compact(session_id, tokens_before, tokens_after)
    mgr.update_session(session_id, compacted=True)
    click.echo(f"✅ 已记录 compact: {tokens_before:,} → {tokens_after:,} (节省 {saved:,} tokens, "
               f"{saved / tokens_before * 100:.1f}%)")


@sessions.command("auto-compact")
@click.option("--threshold", type=int, default=None, help="compact 阈值（默认使用配置文件值）")
@click.option("--keep", type=int, default=5, help="保留最近几轮完整内容")
@click.option("--dry-run", is_flag=True, help="仅显示建议，不执行 compact")
def sessions_auto_compact(threshold: int | None, keep: int, dry_run: bool) -> None:
    """自动检测并 compact 过大的会话。"""
    from claude_token_saver.config import load_config
    from claude_token_saver.compactor import ConversationCompactor

    config = load_config()
    threshold = threshold or config.get("auto_compact_threshold", 100_000)

    mgr = SessionManager()
    compactor = ConversationCompactor()

    sessions = mgr.list_sessions()
    candidates = [s for s in sessions if compactor.should_compact(s.id, s.tokens_used, threshold)]

    if not candidates:
        click.echo("✅ 没有需要 compact 的会话")
        return

    click.echo(click.style(f"🔍 发现 {len(candidates)} 个需要 compact 的会话（阈值: {threshold:,} tokens）\n", fg="yellow"))

    for s in candidates:
        saved = s.tokens_used - int(s.tokens_used * 0.3)
        click.echo(f"   {s.id:<10} {s.title[:28]:<30} {s.tokens_used:>12,} tokens")
        click.echo(f"              预估节省: ~{saved:,} tokens ({saved / s.tokens_used * 100:.0f}%)")

    if dry_run:
        click.echo("\n（dry-run 模式，未执行 compact）")
        return

    if not click.confirm(f"\n确定 compact 这 {len(candidates)} 个会话？"):
        return

    for s in candidates:
        # 构造模拟 turn 数据（从 transcript 恢复）
        from claude_token_saver.transcript import TranscriptParser
        parser = TranscriptParser()
        session_file = Path.home() / ".claude" / "projects" / f"{s.id}.jsonl"
        turns: list[dict] = []
        if session_file.exists():
            try:
                _, parsed_turns = parser.parse_file(session_file)
                turns = [
                    {
                        "turn_index": t.turn_index,
                        "type": t.type,
                        "content": t.content,
                        "tokens": t.total_tokens,
                        "timestamp": t.timestamp,
                        "tool_uses": [
                            {"name": tu.name, "result_content": tu.result_content}
                            for tu in t.tool_uses
                        ],
                    }
                    for t in parsed_turns
                ]
            except Exception:
                pass

        if not turns:
            click.echo(f"   ⚠️  会话 {s.id} 无法恢复对话历史，跳过")
            continue

        result = compactor.compact(s.id, turns, keep_recent=keep)
        formatted = compactor.format_for_prompt(result)
        click.echo(f"   ✅ {s.id}: {result.total_tokens_before:,} → {result.total_tokens_after:,} "
                   f"({result.total_tokens_before - result.total_tokens_after:,} tokens 已压缩)")


@sessions.command("delete")
@click.argument("session_id")
@click.option("--force", is_flag=True, help="强制删除，不确认")
def sessions_delete(session_id: str, force: bool) -> None:
    """删除会话。"""
    if not force:
        if not click.confirm(f"确定删除会话 {session_id}？"):
            return

    mgr = SessionManager()
    if mgr.delete_session(session_id):
        click.echo(f"✅ 会话已删除: {session_id}")
    else:
        click.echo(f"❌ 未找到会话: {session_id}", err=True)


@sessions.command("topics")
def sessions_topics() -> None:
    """列出所有主题及其会话数。"""
    mgr = SessionManager()
    stats = mgr.get_stats()

    click.echo(click.style("📂 主题分布\n", bold=True, fg="cyan"))
    click.echo(f"   {'主题':<20} {'会话数':>8}")
    click.echo(f"   {'─' * 20} {'─' * 8}")
    for t in stats["topics"]:
        click.echo(f"   {t['topic']:<20} {t['count']:>8}")


@sessions.command("stats")
def sessions_stats() -> None:
    """会话统计概览。"""
    mgr = SessionManager()
    stats = mgr.get_stats()

    click.echo(click.style("📊 会话统计\n", bold=True, fg="cyan"))
    click.echo(f"   总会话数:  {stats['total_sessions']}")
    click.echo(f"   总 Token:   {stats['total_tokens']:,}")
    click.echo(f"   总轮次:     {stats['total_turns']}")
    click.echo(f"   Compact 数: {stats['total_compacts']}")
