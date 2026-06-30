"""
Agent Token Saver - 会话管理模块
提供自动 compact、主题会话、上下文恢复等功能。
"""
from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

from claude_token_saver.utils import count_tokens

DB_PATH = Path.home() / ".agent-token-saver" / "sessions.db"


@dataclass
class Session:
    id: str
    title: str
    topic: str = "default"
    tokens_used: int = 0
    turns: int = 0
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    compacted: bool = False
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "title": self.title,
            "topic": self.topic,
            "tokens_used": self.tokens_used,
            "turns": self.turns,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "compacted": self.compacted,
            "metadata": self.metadata,
        }


class SessionManager:
    """会话管理器，使用 SQLite 持久化。"""

    def __init__(self, db_path: Path | None = None):
        self.db_path = db_path or DB_PATH
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        """初始化数据库表。"""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                topic TEXT DEFAULT 'default',
                tokens_used INTEGER DEFAULT 0,
                turns INTEGER DEFAULT 0,
                created_at TEXT,
                updated_at TEXT,
                compacted INTEGER DEFAULT 0,
                metadata TEXT DEFAULT '{}'
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS compact_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                tokens_before INTEGER,
                tokens_after INTEGER,
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            )
        """)
        conn.commit()
        conn.close()

    def create_session(self, title: str, topic: str = "default", session_id: str | None = None) -> Session:
        """创建新会话。"""
        import uuid
        sid = session_id or str(uuid.uuid4())[:8]
        now = datetime.now().isoformat()
        session = Session(
            id=sid,
            title=title,
            topic=topic,
            created_at=now,
            updated_at=now,
        )
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "INSERT INTO sessions (id, title, topic, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
            (session.id, session.title, session.topic, session.created_at, session.updated_at),
        )
        conn.commit()
        conn.close()
        return session

    def get_session(self, session_id: str) -> Optional[Session]:
        """获取会话信息。"""
        conn = sqlite3.connect(self.db_path)
        row = conn.execute(
            "SELECT id, title, topic, tokens_used, turns, created_at, updated_at, compacted, metadata "
            "FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()
        conn.close()
        if not row:
            return None
        return Session(
            id=row[0], title=row[1], topic=row[2], tokens_used=row[3],
            turns=row[4], created_at=row[5], updated_at=row[6],
            compacted=bool(row[7]), metadata=json.loads(row[8]),
        )

    def update_session(self, session_id: str, **kwargs) -> None:
        """更新会话字段。"""
        fields = []
        values = []
        for key, val in kwargs.items():
            fields.append(f"{key} = ?")
            values.append(val)
        fields.append("updated_at = ?")
        values.append(datetime.now().isoformat())
        values.append(session_id)

        conn = sqlite3.connect(self.db_path)
        conn.execute(f"UPDATE sessions SET {', '.join(fields)} WHERE id = ?", values)
        conn.commit()
        conn.close()

    def list_sessions(self, topic: str | None = None) -> list[Session]:
        """列出所有会话，可按主题过滤。"""
        conn = sqlite3.connect(self.db_path)
        if topic:
            rows = conn.execute(
                "SELECT id, title, topic, tokens_used, turns, created_at, updated_at, compacted, metadata "
                "FROM sessions WHERE topic = ? ORDER BY updated_at DESC", (topic,)
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT id, title, topic, tokens_used, turns, created_at, updated_at, compacted, metadata "
                "FROM sessions ORDER BY updated_at DESC"
            ).fetchall()
        conn.close()

        sessions = []
        for row in rows:
            sessions.append(Session(
                id=row[0], title=row[1], topic=row[2], tokens_used=row[3],
                turns=row[4], created_at=row[5], updated_at=row[6],
                compacted=bool(row[7]), metadata=json.loads(row[8]),
            ))
        return sessions

    def list_topics(self) -> list[str]:
        """获取所有主题列表。"""
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute(
            "SELECT DISTINCT topic FROM sessions ORDER BY topic"
        ).fetchall()
        conn.close()
        return [r[0] for r in rows]

    def delete_session(self, session_id: str) -> bool:
        """删除会话。"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        conn.execute("DELETE FROM compact_log WHERE session_id = ?", (session_id,))
        conn.commit()
        conn.close()
        return cursor.rowcount > 0

    def log_compact(self, session_id: str, tokens_before: int, tokens_after: int) -> None:
        """记录一次 compact 操作。"""
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "INSERT INTO compact_log (session_id, timestamp, tokens_before, tokens_after) VALUES (?, ?, ?, ?)",
            (session_id, datetime.now().isoformat(), tokens_before, tokens_after),
        )
        conn.commit()
        conn.close()

    def get_compact_history(self, session_id: str) -> list[dict]:
        """获取会话的 compact 历史。"""
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute(
            "SELECT timestamp, tokens_before, tokens_after FROM compact_log "
            "WHERE session_id = ? ORDER BY timestamp", (session_id,)
        ).fetchall()
        conn.close()
        return [
            {"timestamp": r[0], "tokens_before": r[1], "tokens_after": r[2]}
            for r in rows
        ]

    def get_stats(self) -> dict:
        """获取全局会话统计。"""
        conn = sqlite3.connect(self.db_path)
        total_sessions = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
        total_tokens = conn.execute("SELECT SUM(tokens_used) FROM sessions").fetchone()[0] or 0
        total_turns = conn.execute("SELECT SUM(turns) FROM sessions").fetchone()[0] or 0
        total_compacts = conn.execute("SELECT COUNT(*) FROM compact_log").fetchone()[0]
        topic_counts = conn.execute(
            "SELECT topic, COUNT(*) FROM sessions GROUP BY topic ORDER BY COUNT(*) DESC"
        ).fetchall()
        conn.close()

        return {
            "total_sessions": total_sessions,
            "total_tokens": total_tokens,
            "total_turns": total_turns,
            "total_compacts": total_compacts,
            "topics": [{"topic": t, "count": c} for t, c in topic_counts],
        }
