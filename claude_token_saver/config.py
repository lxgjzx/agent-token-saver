"""
Claude Code Token Saver - 配置管理
"""
from __future__ import annotations

import os
from pathlib import Path

import yaml

CONFIG_DIR = Path.home() / ".claude-token-saver"
CONFIG_FILE = CONFIG_DIR / "config.yaml"

DEFAULT_CONFIG: dict = {
    "model": "claude-sonnet-4-20250514",
    "auto_compact_threshold": 100_000,  # token 数
    "compact_keep_ratio": 0.3,  # 保留最近 30%
    "strip_comments": True,
    "strip_docstrings": False,
    "max_file_tokens": 50_000,
    "max_total_tokens": 200_000,
    "ignore_dirs": sorted({
        ".git", ".svn", ".hg", "__pycache__", "node_modules",
        ".venv", "venv", "dist", "build", ".idea", ".vscode",
        ".gradle", "target", "bin", "obj",
    }),
    "ignore_files": sorted({
        ".DS_Store", "Thumbs.db",
    }),
    "include_binary": False,
}


def load_config() -> dict:
    """加载配置文件，不存在则创建默认配置。"""
    if not CONFIG_FILE.exists():
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        save_config(DEFAULT_CONFIG)
        return DEFAULT_CONFIG.copy()

    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    return {**DEFAULT_CONFIG, **config}


def save_config(config: dict) -> None:
    """保存配置到文件。"""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False)


def get_config_path() -> Path:
    return CONFIG_FILE
