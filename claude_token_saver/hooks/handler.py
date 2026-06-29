"""
Claude Code Token Saver - Hook Handler
处理 Claude Code 的 PreToolUse 和 PostToolUse hooks。

从 stdin 读取 JSON 事件，输出 JSON 决策到 stdout，警告输出到 stderr。
仅使用标准库（json, sys, pathlib, sqlite3）及同包 utils 模块。

Token 节省策略：
  - 大文件读取询问（>500KB 询问用户是否继续，建议使用 --offset/--limit）
  - Glob 结果数量限制（注入 maxResults=100）
  - Grep 结果数量限制（注入 maxMatches=50）
  - 目录排除注入（自动排除 node_modules/.venv/.git 等）
"""
from __future__ import annotations

import json
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

from claude_token_saver.utils import get_file_size

# ── 常量 ──────────────────────────────────────────────────────────────

# 默认大文件阈值（字节），约 200KB
DEFAULT_MAX_FILE_SIZE_BYTES: int = 200_000

# 阻止读取的阈值（字节），约 500KB — 超过此大小的文件直接阻止
BLOCK_MAX_FILE_SIZE_BYTES: int = 500_000

# Glob 最大结果数
DEFAULT_GLOB_MAX_RESULTS: int = 100

# Grep 最大匹配数
DEFAULT_GREP_MAX_MATCHES: int = 50

# 分析数据库路径
ANALYTICS_DB: Path = Path.home() / ".claude-token-saver" / "analytics.db"

# 默认排除目录
DEFAULT_EXCLUDE_DIRS: set[str] = {
    ".git", ".svn", ".hg", "__pycache__", "node_modules",
    ".venv", "venv", "dist", "build", ".idea", ".vscode",
    ".gradle", "target", "bin", "obj",
}

# 默认排除文件
DEFAULT_EXCLUDE_FILES: set[str] = {
    ".DS_Store", "Thumbs.db",
}


# ── 数据库操作 ────────────────────────────────────────────────────────

def _init_db(db_path: Path) -> None:
    """初始化 tool_usage 和 waste_log 表（如不存在）。"""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS tool_usage (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            tool_name   TEXT    NOT NULL,
            tool_input  TEXT    DEFAULT '{}',
            tool_output TEXT    DEFAULT '{}',
            file_path   TEXT,
            file_size   INTEGER DEFAULT 0,
            decision    TEXT    DEFAULT 'approve',
            session_id  TEXT,
            timestamp   TEXT    NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS waste_log (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            category      TEXT    NOT NULL,
            description   TEXT    NOT NULL,
            tokens_wasted INTEGER DEFAULT 0,
            suggestion    TEXT    DEFAULT '',
            timestamp     TEXT    NOT NULL
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS security_log (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            event_type  TEXT    NOT NULL,
            description TEXT    NOT NULL,
            details     TEXT    DEFAULT '',
            timestamp   TEXT    NOT NULL
        )
    """)
    conn.commit()
    conn.close()


def _record_security_event(event_type: str, description: str, details: str = "") -> None:
    """记录安全相关事件到 analytics DB。"""
    _init_db(ANALYTICS_DB)
    conn = sqlite3.connect(ANALYTICS_DB)
    conn.execute(
        "INSERT INTO security_log (event_type, description, details, timestamp) VALUES (?, ?, ?, ?)",
        (event_type, description, details, datetime.now().isoformat()),
    )
    conn.commit()
    conn.close()


def _record_tool_usage(
    tool_name: str,
    tool_input: dict,
    tool_output: dict | None,
    file_path: str | None,
    file_size: int,
    decision: str,
    session_id: str | None,
) -> None:
    """记录 tool 使用到 analytics DB。"""
    _init_db(ANALYTICS_DB)
    conn = sqlite3.connect(ANALYTICS_DB)
    conn.execute(
        "INSERT INTO tool_usage "
        "(tool_name, tool_input, tool_output, file_path, file_size, decision, session_id, timestamp) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (
            tool_name,
            json.dumps(tool_input, ensure_ascii=False),
            json.dumps(tool_output or {}, ensure_ascii=False),
            file_path,
            file_size,
            decision,
            session_id,
            datetime.now().isoformat(),
        ),
    )
    conn.commit()
    conn.close()


# ── 工具函数 ──────────────────────────────────────────────────────────

def _warn_large_file(file_path: str, file_size: int) -> str:
    """生成大文件警告文本。"""
    size_kb = file_size / 1024
    return (
        f"[claude-token-saver] 文件过大 ({size_kb:.0f} KB)，"
        f"建议使用 --offset 和 --limit 参数只读取需要的部分。"
    )


def _add_exclude_paths_to_input(tool_input: dict, tool_name: str) -> dict:
    """为 Glob/Grep 添加排除路径和结果数量限制。"""
    modified = dict(tool_input)

    if tool_name == "Glob":
        # Glob 工具支持 excludeDirs 参数
        existing_excludes = modified.get("excludeDirs", [])
        if isinstance(existing_excludes, list):
            new_excludes = list(existing_excludes)
            for d in DEFAULT_EXCLUDE_DIRS:
                if d not in new_excludes:
                    new_excludes.append(d)
            modified["excludeDirs"] = new_excludes

        # 限制结果数量（如果未设置）
        if "maxResults" not in modified:
            modified["maxResults"] = DEFAULT_GLOB_MAX_RESULTS

    elif tool_name == "Grep":
        # Grep 工具支持 exclude 参数（管道分隔的 glob 模式）
        exclude_patterns = [f"*/{d}/**" for d in sorted(DEFAULT_EXCLUDE_DIRS)]
        for f in sorted(DEFAULT_EXCLUDE_FILES):
            exclude_patterns.append(f"**/{f}")

        existing_exclude = modified.get("exclude", "")
        if existing_exclude:
            modified["exclude"] = existing_exclude + "|" + "|".join(exclude_patterns)
        else:
            modified["exclude"] = "|".join(exclude_patterns)

        # 限制匹配数量（如果未设置）
        if "maxMatches" not in modified:
            modified["maxMatches"] = DEFAULT_GREP_MAX_MATCHES

    return modified


# ── 事件处理 ──────────────────────────────────────────────────────────

def _safe_resolve_path(file_path: str) -> str | None:
    """安全规范化路径：防止路径遍历，只允许在工作目录范围内。"""
    if not file_path:
        return None
    # 拒绝包含空字节的路径（Windows 上 Path.resolve() 不对此报错）
    if "\x00" in file_path:
        _record_security_event(
            "path_traversal_attempt",
            f"路径遍历尝试被阻止（空字节）: {file_path[:50]}",
            f"cwd={Path.cwd()}",
        )
        return None
    resolved = None
    try:
        resolved = Path(file_path).resolve()
        cwd = Path.cwd().resolve()
        resolved.relative_to(cwd)
    except (OSError, ValueError):
        _record_security_event(
            "path_traversal_attempt",
            f"路径遍历尝试被阻止: {file_path}",
            f"cwd={Path.cwd()}, resolved={resolved}",
        )
        return None
    return str(resolved)


def handle_pre_tool(tool_name: str, tool_input: dict, session_id: str | None = None) -> dict:
    """处理 PreToolUse 事件。

    - Read: 检查文件大小，过大则询问用户是否继续
    - Glob/Grep: 自动添加排除路径和结果数量限制
    """
    modified_input = dict(tool_input)
    decision = "approve"
    file_path: str | None = None
    safe_path: str | None = None
    file_size: int = 0
    reason = ""

    if tool_name == "Read":
        raw_path = tool_input.get("file_path", "")
        safe_path = _safe_resolve_path(raw_path)
        if safe_path:
            path = Path(safe_path)
            if path.exists():
                try:
                    file_size = get_file_size(path)
                    if file_size > BLOCK_MAX_FILE_SIZE_BYTES:
                        # 超大文件：询问用户是否继续
                        decision = "ask"
                        reason = (
                            f"[claude-token-saver] 文件过大 ({file_size / 1024:.0f} KB)，"
                            f"建议使用 --offset 和 --limit 参数只读取需要的部分。是否继续？"
                        )
                    elif file_size > DEFAULT_MAX_FILE_SIZE_BYTES:
                        # 大文件：警告但允许
                        reason = _warn_large_file(safe_path, file_size)
                except OSError:
                    pass
        file_path = safe_path or ""

    elif tool_name in ("Glob", "Grep"):
        raw_path = tool_input.get("file_path", "")
        safe_path = _safe_resolve_path(raw_path) if raw_path else None
        modified_input = _add_exclude_paths_to_input(tool_input, tool_name)

    result: dict = {"decision": decision, "reason": reason}

    if modified_input != tool_input:
        result["modified_input"] = modified_input

    _record_tool_usage(tool_name, tool_input, None, file_path, file_size, decision, session_id)

    return result


def handle_post_tool(
    tool_name: str,
    tool_input: dict,
    tool_output: dict,
    session_id: str | None = None,
) -> dict:
    """处理 PostToolUse 事件。

    - 记录所有 tool 使用到 analytics DB
    - 大文件读取记录浪费
    """
    file_path: str | None = None
    safe_path: str | None = None
    file_size: int = 0

    if tool_name == "Read":
        raw_path = tool_input.get("file_path", "")
        safe_path = _safe_resolve_path(raw_path)
        if safe_path:
            path = Path(safe_path)
            if path.exists():
                try:
                    file_size = get_file_size(path)
                except OSError:
                    pass

    # 记录大文件读取浪费
    if tool_name == "Read" and safe_path and file_size > DEFAULT_MAX_FILE_SIZE_BYTES:
        _init_db(ANALYTICS_DB)
        conn = sqlite3.connect(ANALYTICS_DB)
        tokens_wasted = file_size // 4
        conn.execute(
            "INSERT INTO waste_log "
            "(category, description, tokens_wasted, suggestion, timestamp) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                "大文件读取",
                f"读取大文件: {safe_path} ({file_size / 1024:.1f} KB)",
                tokens_wasted,
                "使用 --offset 和 --limit 参数只读取需要的部分",
                datetime.now().isoformat(),
            ),
        )
        conn.commit()
        conn.close()

    _record_tool_usage(tool_name, tool_input, tool_output, safe_path, file_size, "approve", session_id)

    # ── 工具输出压缩 ─────────────────────────────────────────────────────
    compressed_output = _compress_tool_output(tool_name, tool_input, tool_output)
    if compressed_output is not tool_output:
        return {"decision": "approve", "reason": "", "modified_output": compressed_output}

    return {"decision": "approve", "reason": ""}


# ── 工具输出压缩 ────────────────────────────────────────────────────────

# Grep 单次匹配最大上下文行数
DEFAULT_GREP_CONTEXT_LINES: int = 2

# Glob 结果最大展示数（超出后只列路径）
DEFAULT_GLOB_DISPLAY_LIMIT: int = 30


def _compress_tool_output(
    tool_name: str,
    tool_input: dict,
    tool_output: dict,
) -> dict:
    """压缩工具输出，减少返回给 Claude 的 token。

    - Grep: 限制每匹配的上下文行数（默认 2 行）
    - Glob: 结果过多时只返回路径列表，省略文件预览
    """
    if tool_name == "Grep":
        return _compress_grep_output(tool_output)
    elif tool_name == "Glob":
        return _compress_glob_output(tool_output)
    return tool_output


def _compress_grep_output(output: dict) -> dict:
    """压缩 Grep 输出：限制上下文行数。"""
    if not isinstance(output, dict):
        return output

    matches = output.get("matches", [])
    if not isinstance(matches, list):
        return output

    max_context = DEFAULT_GREP_CONTEXT_LINES
    compressed = []
    for match in matches:
        if not isinstance(match, dict):
            compressed.append(match)
            continue
        compressed_match = dict(match)
        context = match.get("context", [])
        if isinstance(context, list) and len(context) > max_context * 2 + 1:
            # 保留匹配行 + 前后各 N 行
            center = len(context) // 2
            start = max(0, center - max_context)
            end = min(len(context), center + max_context + 1)
            compressed_match["context"] = context[start:end]
            compressed_match["context_truncated"] = True
        compressed.append(compressed_match)

    result = dict(output)
    result["matches"] = compressed
    return result


def _compress_glob_output(output: dict) -> dict:
    """压缩 Glob 输出：结果过多时省略内容预览。"""
    if not isinstance(output, dict):
        return output

    results = output.get("results", [])
    if not isinstance(results, list):
        return output

    if len(results) > DEFAULT_GLOB_DISPLAY_LIMIT:
        # 只保留路径，移除大文件的内容预览
        compressed = []
        for r in results:
            if not isinstance(r, dict):
                compressed.append(r)
                continue
            cr = {k: v for k, v in r.items() if k in ("path", "is_dir", "size")}
            compressed.append(cr)
        result = dict(output)
        result["results"] = compressed
        result["truncated"] = True
        result["total_results"] = len(results)
        return result

    return output


# ── 主入口 ────────────────────────────────────────────────────────────

def main() -> None:
    """从 stdin 读取 JSON 事件，处理并输出 JSON 结果到 stdout。"""
    try:
        raw = sys.stdin.read()
        if not raw.strip():
            print(json.dumps({"decision": "approve", "reason": ""}), file=sys.stdout)
            sys.exit(0)

        event = json.loads(raw)
    except (json.JSONDecodeError, IOError, UnicodeDecodeError):
        print(json.dumps({"decision": "approve", "reason": ""}), file=sys.stdout)
        sys.exit(0)

    tool_name: str = event.get("tool_name", "")
    tool_input: dict = event.get("tool_input", {})
    tool_output: dict | None = event.get("tool_output")
    session_id: str | None = event.get("session_id")
    event_type: str = event.get("hook_event_name", "")

    if not tool_name:
        print(json.dumps({"decision": "approve", "reason": "no tool_name"}), file=sys.stdout)
        sys.exit(0)

    try:
        if event_type == "PreToolUse":
            result = handle_pre_tool(tool_name, tool_input, session_id)
        elif event_type == "PostToolUse":
            result = handle_post_tool(tool_name, tool_input, tool_output or {}, session_id)
        else:
            result = {"decision": "approve", "reason": f"unknown event type: {event_type}"}
    except Exception as e:
        _record_security_event("handler_error", f"Hook handler 异常: {type(e).__name__}", str(e)[:500])
        result = {"decision": "approve", "reason": ""}

    print(json.dumps(result), file=sys.stdout)


if __name__ == "__main__":
    main()
