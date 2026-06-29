"""
Claude Code Token Saver - 统计分析模块
追踪 token 消耗、识别浪费、提供优化建议。
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from claude_token_saver.utils import count_tokens, get_file_size
from claude_token_saver.sessions import SessionManager

DB_PATH = Path.home() / ".claude-token-saver" / "analytics.db"


@dataclass
class FileReadEvent:
    file_path: str
    tokens: int
    file_size: int
    timestamp: str
    session_id: str | None = None


@dataclass
class TokenRecord:
    session_id: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    timestamp: str
    prompt_preview: str = ""


class AnalyticsEngine:
    """统计分析引擎。"""

    def __init__(self, db_path: Path | None = None):
        self.db_path = db_path or DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
        self.session_mgr = SessionManager()

    def _init_db(self) -> None:
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS file_reads (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path TEXT NOT NULL,
                tokens INTEGER DEFAULT 0,
                file_size INTEGER DEFAULT 0,
                timestamp TEXT NOT NULL,
                session_id TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS token_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                input_tokens INTEGER DEFAULT 0,
                output_tokens INTEGER DEFAULT 0,
                total_tokens INTEGER DEFAULT 0,
                timestamp TEXT NOT NULL,
                prompt_preview TEXT DEFAULT ''
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS waste_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                category TEXT NOT NULL,
                description TEXT NOT NULL,
                tokens_wasted INTEGER DEFAULT 0,
                suggestion TEXT DEFAULT '',
                timestamp TEXT NOT NULL
            )
        """)
        conn.commit()
        conn.close()

    def record_file_read(self, event: FileReadEvent) -> None:
        """记录一次文件读取事件。"""
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "INSERT INTO file_reads (file_path, tokens, file_size, timestamp, session_id) VALUES (?, ?, ?, ?, ?)",
            (event.file_path, event.tokens, event.file_size, event.timestamp, event.session_id),
        )
        conn.commit()
        conn.close()

    def record_token_usage(self, record: TokenRecord) -> None:
        """记录一次 token 消耗。"""
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "INSERT INTO token_records (session_id, input_tokens, output_tokens, total_tokens, timestamp, prompt_preview) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (record.session_id, record.input_tokens, record.output_tokens,
             record.total_tokens, record.timestamp, record.prompt_preview[:500]),
        )
        # 同步更新 session 统计
        self.session_mgr.update_session(
            record.session_id,
            tokens_used=self.session_mgr.get_session(record.session_id).tokens_used + record.total_tokens,
            turns=self.session_mgr.get_session(record.session_id).turns + 1,
        )
        conn.commit()
        conn.close()

    def record_waste(self, category: str, description: str, tokens_wasted: int = 0, suggestion: str = "") -> None:
        """记录一次 token 浪费。"""
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "INSERT INTO waste_log (category, description, tokens_wasted, suggestion, timestamp) VALUES (?, ?, ?, ?, ?)",
            (category, description, tokens_wasted, suggestion, datetime.now().isoformat()),
        )
        conn.commit()
        conn.close()

    # ── 查询方法 ──

    def get_file_read_heatmap(self, limit: int = 20) -> list[dict[str, Any]]:
        """获取文件读取热力图（按总 token 消耗排序）。"""
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute(
            "SELECT file_path, SUM(tokens) as total_tokens, COUNT(*) as read_count, SUM(file_size) as total_size "
            "FROM file_reads GROUP BY file_path ORDER BY total_tokens DESC LIMIT ?",
            (limit,)
        ).fetchall()
        conn.close()
        return [
            {"file_path": r[0], "total_tokens": r[1], "read_count": r[2], "total_size": r[3]}
            for r in rows
        ]

    def get_token_trend(self, days: int = 7) -> list[dict]:
        """获取 token 消耗趋势（按天）。"""
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute(
            "SELECT DATE(timestamp) as day, SUM(total_tokens) as total "
            "FROM token_records WHERE timestamp >= DATE('now', ?) "
            "GROUP BY day ORDER BY day",
            (f"-{days} days",)
        ).fetchall()
        conn.close()
        return [{"day": r[0], "total_tokens": r[1]} for r in rows]

    def get_top_sessions(self, limit: int = 10) -> list[dict]:
        """获取 token 消耗最高的会话。"""
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute(
            "SELECT session_id, SUM(total_tokens) as total, COUNT(*) as turns "
            "FROM token_records GROUP BY session_id ORDER BY total DESC LIMIT ?",
            (limit,)
        ).fetchall()
        conn.close()
        return [{"session_id": r[0], "total_tokens": r[1], "turns": r[2]} for r in rows]

    def get_waste_summary(self) -> list[dict]:
        """获取浪费摘要。"""
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute(
            "SELECT category, COUNT(*) as count, SUM(tokens_wasted) as total_wasted "
            "FROM waste_log GROUP BY category ORDER BY total_wasted DESC"
        ).fetchall()
        conn.close()
        return [
            {"category": r[0], "count": r[1], "total_wasted": r[2]}
            for r in rows
        ]

    def get_savings_opportunities(self) -> list[dict]:
        """分析并提供节省建议。"""
        suggestions: list[dict] = []

        # 1. 大文件读取
        heatmap = self.get_file_read_heatmap(limit=10)
        large_files = [f for f in heatmap if f["total_tokens"] > 50_000]
        for f in large_files:
            suggestions.append({
                "category": "大文件读取",
                "file": f["file_path"],
                "tokens": f["total_tokens"],
                "suggestion": f"文件被读取了 {f['read_count']} 次，共消耗 {f['total_tokens']:,} tokens。建议使用 /read 命令的 --offset 和 --limit 参数只读取需要的部分。",
            })

        # 2. 会话膨胀
        sessions = self.session_mgr.list_sessions()
        for s in sessions:
            if s.tokens_used > 150_000:
                suggestions.append({
                    "category": "会话膨胀",
                    "session": s.id,
                    "tokens": s.tokens_used,
                    "suggestion": f"会话 '{s.title}' 已使用 {s.tokens_used:,} tokens。建议使用 /compact 压缩，或开启新会话。",
                })

        # 3. 浪费记录
        waste = self.get_waste_summary()
        for w in waste:
            suggestions.append({
                "category": f"浪费: {w['category']}",
                "tokens": w["total_wasted"],
                "suggestion": f"累计 {w['count']} 次此类浪费，共 {w['total_wasted']:,} tokens。",
            })

        return suggestions
