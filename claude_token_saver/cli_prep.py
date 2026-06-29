"""
prep 子命令 - 文件预处理
"""
from __future__ import annotations

import sys

import click

from claude_token_saver.prep import process_files, format_processed_output
from claude_token_saver.utils import should_ignore
from claude_token_saver.config import load_config


@click.group()
def prep() -> None:
    """预处理文件，减少发送给 Claude 的 token 量。"""
    pass


@prep.command("files")
@click.argument("paths", nargs=-1, required=True, type=click.Path(exists=True))
@click.option("-o", "--output", help="输出文件路径（默认打印到 stdout）")
@click.option("--format", type=click.Choice(["markdown", "plain", "json"]), default="markdown", help="输出格式")
@click.option("--no-strip-comments", is_flag=True, help="不去除注释")
@click.option("--strip-docstrings", is_flag=True, help="去除 Python 文档字符串")
@click.option("--no-dedup", is_flag=True, help="不去重")
@click.option("--max-tokens", type=int, default=50_000, help="单文件最大 token 数")
@click.option("--detail-level", type=click.Choice(["skeleton", "stripped", "full", "block"]), default="full", help="压缩级别")
@click.option("--no-cache", is_flag=True, help="禁用文件缓存（强制重新处理）")
@click.option("--dry-run", is_flag=True, help="仅显示统计，不输出内容")
@click.option("--include-binary", is_flag=True, help="包含二进制文件")
@click.option("-v", "--verbose", is_flag=True, help="显示详细信息")
# 新增：自适应 detail_level
@click.option("--auto-detail", is_flag=True, help="根据 token 预算自动分配压缩级别")
@click.option("--token-budget", type=int, default=50_000, help="token 预算（--auto-detail 时使用）")
# 新增：结构去重
@click.option("--structural-dedup", is_flag=True, help="基于代码结构相似度去重（超越 MD5）")
# 新增：渐进式披露（目录索引模式）
@click.option("--index", is_flag=True, help="仅输出目录索引（不读取文件内容）")
def prep_files(
    paths: tuple[str],
    output: str | None,
    format: str,
    no_strip_comments: bool,
    strip_docstrings: bool,
    no_dedup: bool,
    max_tokens: int,
    detail_level: str,
    no_cache: bool,
    dry_run: bool,
    include_binary: bool,
    verbose: bool,
    auto_detail: bool,
    token_budget: int,
    structural_dedup: bool,
    index: bool,
) -> None:
    """处理文件列表，输出精简后的内容。

    detail-level 选项:
      skeleton - 仅提取类/函数签名（Python 文件 ~5-10%% token）
      stripped - 去除注释和 docstring（~30-50%% token）
      full     - 完整内容（默认）
      block    - 阻止读取超大文件（不读取内容）

    新增模式:
      --auto-detail --token-budget N  根据预算自动分配压缩级别
      --structural-dedup              基于代码结构去重
      --index                         仅输出目录索引（渐进式披露）
    """
    config = load_config()
    file_paths = list(paths)

    # 收集目录中的所有文件
    expanded: list[str] = []
    for p in file_paths:
        p = __import__("pathlib").Path(p)
        if p.is_dir():
            for f in sorted(p.rglob("*")):
                if f.is_file() and not should_ignore(f, include_binary=include_binary):
                    expanded.append(str(f))
        else:
            expanded.append(str(p))

    if not expanded:
        click.echo("⚠️  没有找到可处理的文件", err=True)
        sys.exit(1)

    # ── 渐进式披露：目录索引模式 ─────────────────────────────────────────
    if index:
        click.echo(f"📂 构建目录索引（{len(expanded)} 个路径）...")
        from claude_token_saver.prep import build_directory_index, format_index_for_prompt
        idx = build_directory_index(expanded, include_binary=include_binary)
        click.echo(f"   索引文件: {idx['total_files']} 个")
        click.echo(f"   估计 token: {idx['total_estimated_tokens']:,}")
        output_text = format_index_for_prompt(idx, format=format)
        if output:
            __import__("pathlib").Path(output).write_text(output_text, encoding="utf-8")
            click.echo(f"\n✅ 索引已保存到: {output}")
        else:
            click.echo(f"\n{'=' * 60}")
            click.echo(output_text)
        return

    # ── 自动 compact 检查 ────────────────────────────────────────────────
    if auto_detail:
        click.echo(f"💰 自适应模式（预算: {token_budget:,} tokens）")

    # ── 结构去重 ─────────────────────────────────────────────────────────
    if structural_dedup:
        click.echo("🔍 结构去重中...")
        from claude_token_saver.compressor import structural_dedup as _sd
        before = len(expanded)
        expanded = [str(p) for p in _sd([__import__("pathlib").Path(p) for p in expanded])]
        dup_count = before - len(expanded)
        if dup_count:
            click.echo(f"   结构重复: {dup_count} 个文件已去重")

    result = process_files(
        expanded,
        do_strip_comments=not no_strip_comments,
        do_strip_docstrings=strip_docstrings,
        max_file_tokens=max_tokens,
        dedup=not no_dedup,
        include_binary=include_binary,
        detail_level=detail_level,
        token_cache_enabled=not no_cache,
        auto_detail=auto_detail,
        token_budget=token_budget,
        structural_dedup=False,  # 已在上方处理
    )

    # 输出统计
    effective_level = "auto" if auto_detail else detail_level
    click.echo(f"📊 处理结果（级别: {effective_level}）：")
    click.echo(f"   文件数: {len(result['files'])}")
    if result.get("cache_hits"):
        click.echo(f"   缓存命中: {result['cache_hits']} 个文件（跳过重新处理）")
    if result["duplicates_removed"]:
        click.echo(f"   去重: {result['duplicates_removed']} 个重复文件已移除")
    if result["skipped"]:
        click.echo(f"   跳过: {len(result['skipped'])} 个文件")
    click.echo(f"   Token 压缩: {result['total_tokens_before']:,} → {result['total_tokens_after']:,} "
               f"({result['savings_pct']}% 节省)")

    if verbose and result["files"]:
        click.echo(f"\n📁 各文件节省情况：")
        click.echo(f"   {'文件':<50} {'级别':<10} {'压缩前':>10} {'压缩后':>10} {'节省':>10}")
        click.echo(f"   {'─' * 50} {'─' * 10} {'─' * 10} {'─' * 10}")
        for f in result["files"]:
            lvl = f.get("detail_level", detail_level)
            savings_pct = (f["savings"] / f["tokens_before"] * 100) if f["tokens_before"] else 0
            click.echo(f"   {str(f['path']):<50} {lvl:<10} {f['tokens_before']:>10,} {f['tokens_after']:>10,} "
                       f"{savings_pct:>9.0f}%")

    if dry_run:
        return

    output_text = format_processed_output(result, format=format)

    if output:
        __import__("pathlib").Path(output).write_text(output_text, encoding="utf-8")
        click.echo(f"\n✅ 输出已保存到: {output}")
    else:
        click.echo(f"\n{'=' * 60}")
        click.echo(output_text)


@prep.command("prompt")
@click.argument("text", required=False)
@click.option("-f", "--file", type=click.Path(exists=True), help="从文件读取 prompt")
@click.option("--max-tokens", type=int, default=10_000, help="最大 token 数")
@click.option("-o", "--output", help="输出文件路径")
def prep_prompt(text: str | None, file: str | None, max_tokens: int, output: str | None) -> None:
    """压缩 prompt 文本。"""
    if file:
        content = __import__("pathlib").Path(file).read_text(encoding="utf-8")
    elif text:
        content = text
    else:
        content = sys.stdin.read()

    from claude_token_saver.prep import compress_prompt
    from claude_token_saver.utils import count_tokens

    original_tokens = count_tokens(content)
    compressed = compress_prompt(content, max_tokens=max_tokens)
    new_tokens = count_tokens(compressed)

    click.echo(f"📊 Prompt 压缩: {original_tokens:,} → {new_tokens:,} tokens "
               f"({(original_tokens - new_tokens) / original_tokens * 100:.1f}% 节省)")
    click.echo()

    if output:
        __import__("pathlib").Path(output).write_text(compressed, encoding="utf-8")
        click.echo(f"✅ 已保存到: {output}")
    else:
        click.echo(compressed)


@prep.command("diff")
@click.argument("path_a", type=click.Path(exists=True))
@click.argument("path_b", type=click.Path(exists=True))
def prep_diff(path_a: str, path_b: str) -> None:
    """比较两个文件/目录的处理效果。"""
    from claude_token_saver.prep import process_files
    from claude_token_saver.utils import count_tokens

    def collect(p: str) -> list[str]:
        p = __import__("pathlib").Path(p)
        if p.is_dir():
            return [str(f) for f in sorted(p.rglob("*")) if f.is_file() and not should_ignore(f)]
        return [str(p)]

    files_a = collect(path_a)
    files_b = collect(path_b)

    result_a = process_files(files_a)
    result_b = process_files(files_b)

    click.echo("📊 对比结果：")
    click.echo(f"   {'':30} {'A':>12} {'B':>12}")
    click.echo(f"   {'─' * 30} {'─' * 12} {'─' * 12}")
    click.echo(f"   {'文件数':30} {len(result_a['files']):>12,} {len(result_b['files']):>12,}")
    click.echo(f"   {'Token (压缩前)':30} {result_a['total_tokens_before']:>12,} {result_b['total_tokens_before']:>12,}")
    click.echo(f"   {'Token (压缩后)':30} {result_a['total_tokens_after']:>12,} {result_b['total_tokens_after']:>12,}")
    click.echo(f"   {'节省比例':30} {result_a['savings_pct']:>11}% {result_b['savings_pct']:>11}%")


@prep.command("pipe")
@click.option("--no-strip-comments", is_flag=True)
@click.option("--max-tokens", type=int, default=50_000)
def prep_pipe(no_strip_comments: bool, max_tokens: int) -> None:
    """从 stdin 读取，处理后输出到 stdout。适合管道使用。"""
    content = sys.stdin.read()
    from claude_token_saver.utils import count_tokens
    from claude_token_saver.prep import compress_prompt

    original = count_tokens(content)
    result = compress_prompt(content, max_tokens=max_tokens)
    new = count_tokens(result)

    click.echo(f"<!-- tokens: {original:,} → {new:,} ({(original-new)/original*100:.1f}% saved) -->")
    click.echo(result)


@prep.command("watch")
@click.option("--interval", "-i", type=int, default=60, help="检查间隔（秒）")
def prep_watch(interval: int) -> None:
    """监控 Claude Code 会话，接近 token 阈值时提醒 compact。"""
    import time

    mgr = SessionManager()
    config = load_config()
    threshold = config.get("auto_compact_threshold", 100_000)

    click.echo(f"👁️  监控中... (阈值: {threshold:,} tokens, 间隔: {interval}s)")
    click.echo("   按 Ctrl+C 退出\n")

    try:
        while True:
            sessions = mgr.list_sessions()
            for s in sessions:
                if not s.compacted and s.tokens_used > threshold * 0.8:
                    click.echo(
                        click.style(
                            f"⚠️  会话 [{s.id}] {s.title} 已使用 {s.tokens_used:,} tokens "
                            f"(阈值 {threshold:,})，建议 compact!",
                            fg="yellow",
                        )
                    )
            time.sleep(interval)
    except KeyboardInterrupt:
        click.echo("\n👋 监控已停止")
