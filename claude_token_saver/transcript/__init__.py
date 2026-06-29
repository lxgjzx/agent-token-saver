"""
Claude Code Token Saver - Transcript 解析模块
解析 ~/.claude/projects/ 下的 JSONL transcript 文件。
仅使用标准库：json, pathlib, sqlite3, datetime。
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional


# ── 模型定价（$/MTok）─────────────────────────────────────────────

MODEL_PRICING: dict[str, dict[str, float]] = {
    "opus":   {"input": 15.0, "output": 75.0},
    "sonnet": {"input": 3.0,  "output": 15.0},
    "haiku":  {"input": 0.8,  "output": 4.0},
}


def _pricing_tier(model: str) -> tuple[str, dict[str, float]]:
    """根据模型名推断定价档位。"""
    lower = model.lower()
    for tier, prices in MODEL_PRICING.items():
        if tier in lower:
            return tier, prices
    return "sonnet", MODEL_PRICING["sonnet"]


def _compute_cost(usage: dict, model: str) -> "CostRecord":
    """根据 usage 和模型名估算费用。"""
    tier, prices = _pricing_tier(model)
    input_tokens = usage.get("input_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)
    input_cost = input_tokens * prices["input"] / 1_000_000
    output_cost = output_tokens * prices["output"] / 1_000_000
    return CostRecord(
        input_cost=input_cost,
        output_cost=output_cost,
        total_cost=input_cost + output_cost,
        model=model,
        pricing_tier=tier,
    )


# ── 数据类 ──────────────────────────────────────────────────────────

@dataclass
class ToolUse:
    """单次工具调用记录。"""
    id: str
    turn_id: str
    name: str
    input: dict = field(default_factory=dict)
    result_content: str = ""
    is_error: bool = False
    timestamp: str = ""

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "turn_id": self.turn_id,
            "name": self.name,
            "input": self.input,
            "result_content": self.result_content[:500],
            "is_error": self.is_error,
            "timestamp": self.timestamp,
        }


@dataclass
class Turn:
    """对话轮次（用户输入 + 助手响应 + 工具调用链）。"""
    id: str
    session_id: str
    turn_index: int
    type: str  # "user"
    parent_uuid: Optional[str]
    content: str  # 用户输入文本
    timestamp: str
    cwd: str
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    total_tokens: int = 0
    service_tier: str = ""
    tool_uses: list[ToolUse] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "session_id": self.session_id,
            "turn_index": self.turn_index,
            "type": self.type,
            "content_preview": self.content[:300],
            "timestamp": self.timestamp,
            "cwd": self.cwd,
            "model": self.model,
            "total_tokens": self.total_tokens,
            "tool_count": len(self.tool_uses),
            "tool_uses": [t.to_dict() for t in self.tool_uses],
        }


@dataclass
class Session:
    """会话元数据。"""
    id: str
    file_path: str
    work_dir: str = ""
    permission_mode: str = ""
    title: str = ""
    model: str = ""
    created_at: str = ""
    updated_at: str = ""
    turn_count: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "file_path": self.file_path,
            "title": self.title,
            "work_dir": self.work_dir,
            "model": self.model,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "turn_count": self.turn_count,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
        }


@dataclass
class UsageRecord:
    """单次 API 调用的 Token 消耗记录。"""
    session_id: str
    turn_id: str
    input_tokens: int = 0
    output_tokens: int = 0
    cache_creation_input_tokens: int = 0
    cache_read_input_tokens: int = 0
    total_tokens: int = 0
    model: str = ""
    timestamp: str = ""
    service_tier: str = ""

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "turn_id": self.turn_id,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_creation_input_tokens": self.cache_creation_input_tokens,
            "cache_read_input_tokens": self.cache_read_input_tokens,
            "total_tokens": self.total_tokens,
            "model": self.model,
            "timestamp": self.timestamp,
            "service_tier": self.service_tier,
        }


@dataclass
class CostRecord:
    """费用估算记录。"""
    session_id: str
    turn_id: str
    input_cost: float = 0.0
    output_cost: float = 0.0
    total_cost: float = 0.0
    model: str = ""
    pricing_tier: str = "sonnet"
    currency: str = "USD"
    timestamp: str = ""

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "turn_id": self.turn_id,
            "input_cost": round(self.input_cost, 6),
            "output_cost": round(self.output_cost, 6),
            "total_cost": round(self.total_cost, 6),
            "model": self.model,
            "pricing_tier": self.pricing_tier,
            "currency": self.currency,
            "timestamp": self.timestamp,
        }


# ── 解析器 ──────────────────────────────────────────────────────────

class TranscriptParser:
    """Claude Code transcript JSONL 解析器。

    使用方式：
        parser = TranscriptParser()
        results = parser.parse_directory()           # 扫描整个 projects 目录
        session, turns = parser.parse_file(path)     # 解析单个文件
        event = parser.parse_line(line)              # 解析单行 JSON
        count = parser.import_to_db(results)         # 写入 analytics DB
    """

    def __init__(self, projects_dir: Path | None = None):
        self.projects_dir = projects_dir or (Path.home() / ".claude" / "projects")

    # ── 目录扫描 ──

    def parse_directory(self) -> list[tuple[Session, list[Turn]]]:
        """扫描 ~/.claude/projects/ 下所有 .jsonl 文件（排除 subagents），返回解析结果列表。"""
        results: list[tuple[Session, list[Turn]]] = []
        if not self.projects_dir.exists():
            return results
        for jsonl_file in sorted(self.projects_dir.rglob("*.jsonl")):
            if "subagents" in jsonl_file.parts:
                continue
            parsed = self.parse_file(jsonl_file)
            if parsed:
                results.append(parsed)
        return results

    # ── 文件解析 ──

    def parse_file(self, file_path: Path) -> tuple[Session, list[Turn]] | None:
        """解析单个 JSONL 文件，返回 (Session, list[Turn])。

        解析策略：
        1. 第一遍扫描：收集所有 tool_result 块（按 tool_use_id 索引）
        2. 第二遍扫描：构建 Session 和 Turn，关联 tool_use → tool_result
        3. 以用户 text 消息为轮次边界，中间所有 assistant 消息归入同一轮次
        """
        file_path = Path(file_path)
        if not file_path.exists():
            return None

        session_id = file_path.stem
        session: Session | None = None
        turns: list[Turn] = []
        current_turn: Turn | None = None
        turn_index = 0

        # 第一遍：索引 tool_result（跨 user 类型的 tool_result 消息）
        tool_results: dict[str, dict] = {}
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if data.get("type") == "user":
                    msg = data.get("message", {})
                    content = msg.get("content", "")
                    if isinstance(content, list):
                        for block in content:
                            if block.get("type") == "tool_result":
                                tool_use_id = block.get("tool_use_id", "")
                                if tool_use_id:
                                    tool_results[tool_use_id] = block

        # 第二遍：构建 session 和 turns
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue

                event_type = data.get("type")

                # ── session-meta：初始化会话 ──
                if event_type == "session-meta":
                    if session is None:
                        session = Session(
                            id=session_id,
                            file_path=str(file_path),
                            work_dir=data.get("workDir", ""),
                            permission_mode=data.get("permissionMode", ""),
                        )
                    elif not session.work_dir:
                        session.work_dir = data.get("workDir", "") or session.work_dir
                    continue

                # ── ai-title：会话标题 ──
                if event_type == "ai-title":
                    if session:
                        session.title = data.get("aiTitle", "")
                    continue

                # ── file-history-snapshot：用于确定会话创建时间 ──
                if event_type == "file-history-snapshot":
                    if session and not session.created_at:
                        ts = data.get("snapshot", {}).get("timestamp", "")
                        if ts:
                            session.created_at = ts
                    continue

                # ── user 消息（真实输入，非 tool_result）→ 新轮次 ──
                if event_type == "user":
                    msg = data.get("message", {})
                    content = msg.get("content", "")
                    # 过滤 tool_result 类型的 user 消息（它们属于前一轮的工具结果）
                    if isinstance(content, list):
                        has_text = any(b.get("type") == "text" for b in content)
                        if not has_text:
                            continue

                    turn_index += 1
                    text_content = ""
                    if isinstance(content, list):
                        for block in content:
                            if block.get("type") == "text":
                                text_content += block.get("text", "")
                    else:
                        text_content = str(content)

                    current_turn = Turn(
                        id=data.get("uuid", f"{session_id}-{turn_index}"),
                        session_id=session_id,
                        turn_index=turn_index,
                        type="user",
                        parent_uuid=data.get("parentUuid"),
                        content=text_content[:5000],
                        timestamp=data.get("timestamp", ""),
                        cwd=data.get("cwd", ""),
                    )
                    turns.append(current_turn)
                    continue

                # ── assistant 消息 ──
                if event_type == "assistant":
                    if current_turn is None:
                        # 孤儿 assistant 消息（无前置 user）→ 创建虚拟轮次
                        turn_index += 1
                        current_turn = Turn(
                            id=f"{session_id}-{turn_index}-orphan",
                            session_id=session_id,
                            turn_index=turn_index,
                            type="assistant",
                            parent_uuid=None,
                            content="",
                            timestamp=data.get("timestamp", ""),
                            cwd=data.get("cwd", ""),
                        )
                        turns.append(current_turn)

                    # 更新模型
                    model = data.get("model", "")
                    if model:
                        if session and not session.model:
                            session.model = model
                        if not current_turn.model:
                            current_turn.model = model

                    # 提取 usage
                    usage = data.get("usage") or {}
                    if usage:
                        current_turn.input_tokens += usage.get("input_tokens", 0)
                        current_turn.output_tokens += usage.get("output_tokens", 0)
                        current_turn.cache_creation_input_tokens += usage.get("cache_creation_input_tokens", 0)
                        current_turn.cache_read_input_tokens += usage.get("cache_read_input_tokens", 0)
                        current_turn.total_tokens += (
                            usage.get("input_tokens", 0) + usage.get("output_tokens", 0)
                        )
                        current_turn.service_tier = usage.get("service_tier", current_turn.service_tier)

                        if session:
                            session.total_input_tokens += usage.get("input_tokens", 0)
                            session.total_output_tokens += usage.get("output_tokens", 0)

                    # 提取 tool_use 块
                    msg = data.get("message", {})
                    msg_content = msg.get("content", "")
                    if isinstance(msg_content, list):
                        for block in msg_content:
                            if block.get("type") == "tool_use":
                                tool_use_id = block.get("id", "")
                                result_block = tool_results.get(tool_use_id, {})
                                result_content = ""
                                is_error = False
                                if result_block:
                                    rc = result_block.get("content", "")
                                    if isinstance(rc, list):
                                        result_parts = []
                                        for part in rc:
                                            if part.get("type") == "text":
                                                result_parts.append(part.get("text", ""))
                                        result_content = " ".join(result_parts)
                                    else:
                                        result_content = str(rc)
                                    is_error = result_block.get("is_error", False)

                                current_turn.tool_uses.append(ToolUse(
                                    id=tool_use_id,
                                    turn_id=current_turn.id,
                                    name=block.get("name", ""),
                                    input=block.get("input", {}) if isinstance(block.get("input"), dict) else {},
                                    result_content=result_content[:2000],
                                    is_error=is_error,
                                    timestamp=data.get("timestamp", ""),
                                ))
                    continue

        # 收尾：填充 session 时间戳
        if session:
            if turns:
                if not session.created_at:
                    session.created_at = turns[0].timestamp or datetime.now().isoformat()
                session.updated_at = turns[-1].timestamp or session.updated_at or datetime.now().isoformat()
            else:
                if not session.created_at:
                    session.created_at = datetime.now().isoformat()
                if not session.updated_at:
                    session.updated_at = datetime.now().isoformat()
            session.turn_count = len(turns)

        if session:
            return session, turns
        return None

    # ── 单行解析 ──

    def parse_line(self, line: str) -> dict | None:
        """解析单行 JSON，返回结构化事件字典。

        支持的事件类型：session-meta, user, assistant, ai-title
        """
        line = line.strip()
        if not line:
            return None
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            return None

        event_type = data.get("type")
        result: dict = {"type": event_type}

        if event_type == "session-meta":
            result["workDir"] = data.get("workDir", "")
            result["permissionMode"] = data.get("permissionMode", "")
            result["timestamp"] = data.get("timestamp", "")

        elif event_type == "user":
            msg = data.get("message", {})
            content = msg.get("content", "")
            text = ""
            if isinstance(content, list):
                for block in content:
                    if block.get("type") == "text":
                        text += block.get("text", "")
            elif isinstance(content, str):
                text = content
            result["text"] = text[:200]
            result["uuid"] = data.get("uuid", "")
            result["timestamp"] = data.get("timestamp", "")
            result["cwd"] = data.get("cwd", "")

        elif event_type == "assistant":
            msg = data.get("message", {})
            result["model"] = data.get("model", "")
            result["usage"] = data.get("usage", {})
            result["uuid"] = data.get("uuid", "")
            result["timestamp"] = data.get("timestamp", "")
            content = msg.get("content", "")
            blocks: list[dict] = []
            if isinstance(content, list):
                for block in content:
                    entry: dict = {"type": block.get("type")}
                    if block.get("type") == "text":
                        entry["text"] = block.get("text", "")[:100]
                    elif block.get("type") == "tool_use":
                        entry["tool_name"] = block.get("name", "")
                        entry["tool_id"] = block.get("id", "")
                    elif block.get("type") == "thinking":
                        entry["thinking"] = (block.get("thinking", "") or "")[:100]
                    blocks.append(entry)
            result["content_blocks"] = blocks

        elif event_type == "ai-title":
            result["title"] = data.get("aiTitle", "")

        else:
            result["raw_keys"] = list(data.keys())

        return result

    # ── 写入数据库 ──

    def import_to_db(
        self,
        parsed: list[tuple[Session, list[Turn]]],
        db_path: Path | None = None,
    ) -> int:
        """将解析结果写入 analytics DB。

        创建/更新以下表：
          - transcript_sessions
          - transcript_turns
          - transcript_tool_uses
          - transcript_usage_records
          - transcript_cost_records

        返回写入的 turn 数量。
        """
        if not parsed:
            return 0

        analytics_path = db_path or (Path.home() / ".claude-token-saver" / "analytics.db")
        analytics_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(analytics_path)

        try:
            self._ensure_tables(conn)

            for session, turns in parsed:
                # 写入 session
                conn.execute(
                    """INSERT OR REPLACE INTO transcript_sessions
                       (id, file_path, work_dir, permission_mode, title, model,
                        created_at, updated_at, turn_count,
                        total_input_tokens, total_output_tokens)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        session.id, session.file_path, session.work_dir,
                        session.permission_mode, session.title, session.model,
                        session.created_at, session.updated_at,
                        session.turn_count,
                        session.total_input_tokens, session.total_output_tokens,
                    ),
                )

                for turn in turns:
                    # 写入 turn
                    conn.execute(
                        """INSERT OR REPLACE INTO transcript_turns
                           (id, session_id, turn_index, type, parent_uuid,
                            content_summary, timestamp, cwd, model,
                            input_tokens, output_tokens,
                            cache_creation_input_tokens, cache_read_input_tokens,
                            total_tokens, service_tier)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            turn.id, turn.session_id, turn.turn_index, turn.type,
                            turn.parent_uuid, turn.content[:5000], turn.timestamp,
                            turn.cwd, turn.model,
                            turn.input_tokens, turn.output_tokens,
                            turn.cache_creation_input_tokens, turn.cache_read_input_tokens,
                            turn.total_tokens, turn.service_tier,
                        ),
                    )

                    # 写入 tool_uses
                    for tool_use in turn.tool_uses:
                        conn.execute(
                            """INSERT OR REPLACE INTO transcript_tool_uses
                               (id, turn_id, name, input, result_content, is_error, timestamp)
                               VALUES (?, ?, ?, ?, ?, ?, ?)""",
                            (
                                tool_use.id, tool_use.turn_id, tool_use.name,
                                json.dumps(tool_use.input, ensure_ascii=False),
                                tool_use.result_content,
                                1 if tool_use.is_error else 0,
                                tool_use.timestamp,
                            ),
                        )

                    # 写入 usage & cost（仅当有 token 消耗时）
                    if turn.total_tokens > 0:
                        effective_model = turn.model or session.model or "sonnet"
                        conn.execute(
                            """INSERT INTO transcript_usage_records
                               (session_id, turn_id, input_tokens, output_tokens,
                                cache_creation_input_tokens, cache_read_input_tokens,
                                total_tokens, model, timestamp, service_tier)
                               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                            (
                                session.id, turn.id,
                                turn.input_tokens, turn.output_tokens,
                                turn.cache_creation_input_tokens,
                                turn.cache_read_input_tokens,
                                turn.total_tokens, effective_model,
                                turn.timestamp, turn.service_tier,
                            ),
                        )

                        cost = _compute_cost(
                            {
                                "input_tokens": turn.input_tokens,
                                "output_tokens": turn.output_tokens,
                            },
                            effective_model,
                        )
                        cost.session_id = session.id
                        cost.turn_id = turn.id
                        cost.timestamp = turn.timestamp
                        conn.execute(
                            """INSERT INTO transcript_cost_records
                               (session_id, turn_id, input_cost, output_cost,
                                total_cost, model, pricing_tier, currency, timestamp)
                               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                            (
                                cost.session_id, cost.turn_id,
                                cost.input_cost, cost.output_cost,
                                cost.total_cost, cost.model,
                                cost.pricing_tier, cost.currency,
                                cost.timestamp,
                            ),
                        )

            conn.commit()
        finally:
            conn.close()

        return len(parsed)

    @staticmethod
    def _ensure_tables(conn: sqlite3.Connection) -> None:
        """创建 transcript 相关表（如不存在）。"""
        conn.execute("""
            CREATE TABLE IF NOT EXISTS transcript_sessions (
                id                    TEXT    PRIMARY KEY,
                file_path             TEXT    NOT NULL,
                work_dir              TEXT,
                permission_mode       TEXT,
                title                 TEXT,
                model                 TEXT,
                created_at            TEXT,
                updated_at            TEXT,
                turn_count            INTEGER DEFAULT 0,
                total_input_tokens    INTEGER DEFAULT 0,
                total_output_tokens   INTEGER DEFAULT 0
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS transcript_turns (
                id                            TEXT    PRIMARY KEY,
                session_id                    TEXT    NOT NULL,
                turn_index                    INTEGER NOT NULL,
                type                          TEXT    NOT NULL,
                parent_uuid                   TEXT,
                content_summary               TEXT,
                timestamp                     TEXT,
                cwd                           TEXT,
                model                         TEXT,
                input_tokens                  INTEGER DEFAULT 0,
                output_tokens                 INTEGER DEFAULT 0,
                cache_creation_input_tokens   INTEGER DEFAULT 0,
                cache_read_input_tokens       INTEGER DEFAULT 0,
                total_tokens                  INTEGER DEFAULT 0,
                service_tier                  TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS transcript_tool_uses (
                id             TEXT    PRIMARY KEY,
                turn_id        TEXT    NOT NULL,
                name           TEXT    NOT NULL,
                input          TEXT,
                result_content TEXT,
                is_error       INTEGER DEFAULT 0,
                timestamp      TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS transcript_usage_records (
                id                            INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id                    TEXT    NOT NULL,
                turn_id                       TEXT,
                input_tokens                  INTEGER DEFAULT 0,
                output_tokens                 INTEGER DEFAULT 0,
                cache_creation_input_tokens   INTEGER DEFAULT 0,
                cache_read_input_tokens       INTEGER DEFAULT 0,
                total_tokens                  INTEGER DEFAULT 0,
                model                         TEXT,
                timestamp                     TEXT    NOT NULL,
                service_tier                  TEXT
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS transcript_cost_records (
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id   TEXT    NOT NULL,
                turn_id      TEXT,
                input_cost   REAL    DEFAULT 0,
                output_cost  REAL    DEFAULT 0,
                total_cost   REAL    DEFAULT 0,
                model        TEXT,
                pricing_tier TEXT,
                currency     TEXT    DEFAULT 'USD',
                timestamp    TEXT    NOT NULL
            )
        """)
        conn.commit()
