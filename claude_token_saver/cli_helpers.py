"""
cli_helpers - 共享的 CLI 辅助函数
"""
from __future__ import annotations

import json

from claude_token_saver.config import DEFAULT_CONFIG, load_config, save_config


def handle_config(show: bool, set_key: str | None, reset: bool) -> None:
    """处理 config 子命令。"""
    import click

    if reset:
        save_config(DEFAULT_CONFIG)
        click.echo("✅ 配置已重置为默认值")
        return

    config = load_config()

    if set_key:
        key, _, value = set_key.partition("=")
        key = key.strip()
        value = value.strip()

        # 尝试解析为数字
        try:
            value = int(value)
        except ValueError:
            try:
                value = float(value)
            except ValueError:
                if value.lower() == "true":
                    value = True
                elif value.lower() == "false":
                    value = False

        config[key] = value
        save_config(config)
        click.echo(f"✅ 已设置 {key} = {value}")
        return

    if show:
        click.echo(click.style(json.dumps(config, indent=2, ensure_ascii=False), fg="cyan"))
    else:
        click.echo("当前配置 (使用 --show 查看详情，--set key=value 修改，--reset 重置):\n")
        for key, val in config.items():
            click.echo(f"  {key:<30} = {val}")
