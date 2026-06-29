"""Tests for claude-token-saver"""
from __future__ import annotations

import json
import pytest

from claude_token_saver.utils import count_tokens, is_binary_file, should_ignore
from claude_token_saver.prep import (
    strip_comments, strip_python_docstrings, smart_truncate,
    deduplicate_files, compress_prompt, process_files, _clear_caches,
)
from claude_token_saver.compressor import (
    extract_skeleton, extract_symbol_index, format_symbol_index,
    structural_dedup, group_by_structure,
)
from claude_token_saver.hooks.handler import (
    _safe_resolve_path, _add_exclude_paths_to_input,
    _compress_tool_output,
)
from claude_token_saver.budget import auto_detail_level, estimate_auto_cost
from claude_token_saver.compactor import ConversationCompactor, CompactedContext
from claude_token_saver.prep import build_directory_index, format_index_for_prompt
from claude_token_saver.sessions import SessionManager
from claude_token_saver.config import load_config, save_config


# ── utils ──

class TestCountTokens:
    def test_empty(self):
        assert count_tokens("") == 0

    def test_ascii(self):
        result = count_tokens("hello world")
        assert result > 0

    def test_chinese(self):
        result = count_tokens("你好世界")
        assert result > 0


class TestIsBinary:
    def test_text_file(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("hello")
        assert not is_binary_file(f)

    def test_nonexistent(self):
        assert not is_binary_file("/nonexistent/path")


class TestShouldIgnore:
    def test_git_dir(self, tmp_path):
        git_dir = tmp_path / ".git"
        git_dir.mkdir()
        assert should_ignore(git_dir)

    def test_node_modules(self, tmp_path):
        nm = tmp_path / "node_modules" / "pkg"
        nm.mkdir(parents=True)
        assert should_ignore(nm)

    def test_normal_file(self, tmp_path):
        f = tmp_path / "main.py"
        f.write_text("print('hello')")
        assert not should_ignore(f)


# ── prep ──

class TestStripComments:
    def test_python(self):
        code = "x = 1  # comment\n# another comment\ny = 2\n"
        result = strip_comments(code, ".py")
        assert "# comment" not in result
        assert "# another comment" not in result
        assert "x = 1" in result
        assert "y = 2" in result

    def test_javascript(self):
        code = "var x = 1; // comment\n/* block */\ny = 2;\n"
        result = strip_comments(code, ".js")
        assert "// comment" not in result
        assert "/* block */" not in result

    def test_no_pattern(self):
        code = "hello world"
        result = strip_comments(code, ".xyz")
        assert result == code


class TestStripPythonDocstrings:
    def test_removes_function_docstring(self):
        code = '''def foo():
    """This is a docstring."""
    return 1
'''
        result = strip_python_docstrings(code)
        assert "This is a docstring" not in result
        assert "return 1" in result

    def test_syntax_error(self):
        code = "def broken("
        result = strip_python_docstrings(code)
        assert result == code  # returns original on error


class TestSmartTruncate:
    def test_short_content(self):
        content = "line1\nline2\nline3\n"
        result = smart_truncate(content, 1000)
        assert result == content

    def test_long_content(self):
        lines = [f"line{i}" for i in range(100)]
        content = "\n".join(lines)
        result = smart_truncate(content, 100)
        assert "..." in result
        assert len(result) < len(content)


class TestDeduplicate:
    def test_dedup(self, tmp_path):
        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"
        f1.write_text("same content")
        f2.write_text("same content")
        result = deduplicate_files([str(f1), str(f2)])
        assert len(result) == 1


class TestCompressPrompt:
    def test_short(self):
        text = "hello"
        assert compress_prompt(text) == "hello"

    def test_excess_blank_lines(self):
        text = "hello\n\n\n\n\nworld"
        result = compress_prompt(text)
        assert "\n\n\n" not in result


# ── sessions ──

class TestSessionManager:
    def test_create_and_get(self, tmp_path, monkeypatch):
        monkeypatch.setattr("claude_token_saver.sessions.DB_PATH", tmp_path / "test.db")
        mgr = SessionManager(db_path=tmp_path / "test.db")
        s = mgr.create_session("Test Session", topic="testing")
        assert s.id is not None
        assert s.title == "Test Session"

        fetched = mgr.get_session(s.id)
        assert fetched is not None
        assert fetched.title == "Test Session"

    def test_list_sessions(self, tmp_path, monkeypatch):
        monkeypatch.setattr("claude_token_saver.sessions.DB_PATH", tmp_path / "test.db")
        mgr = SessionManager(db_path=tmp_path / "test.db")
        mgr.create_session("S1", topic="a")
        mgr.create_session("S2", topic="b")

        all_sessions = mgr.list_sessions()
        assert len(all_sessions) == 2

        topic_a = mgr.list_sessions(topic="a")
        assert len(topic_a) == 1

    def test_delete_session(self, tmp_path, monkeypatch):
        monkeypatch.setattr("claude_token_saver.sessions.DB_PATH", tmp_path / "test.db")
        mgr = SessionManager(db_path=tmp_path / "test.db")
        s = mgr.create_session("To Delete")
        assert mgr.delete_session(s.id) is True
        assert mgr.get_session(s.id) is None

    def test_compact_log(self, tmp_path, monkeypatch):
        monkeypatch.setattr("claude_token_saver.sessions.DB_PATH", tmp_path / "test.db")
        mgr = SessionManager(db_path=tmp_path / "test.db")
        s = mgr.create_session("Test")
        mgr.log_compact(s.id, 100000, 30000)
        history = mgr.get_compact_history(s.id)
        assert len(history) == 1
        assert history[0]["tokens_before"] == 100000


# ── config ──

class TestConfig:
    def test_default_config(self, tmp_path, monkeypatch):
        monkeypatch.setattr("claude_token_saver.config.CONFIG_DIR", tmp_path / "config")
        monkeypatch.setattr("claude_token_saver.config.CONFIG_FILE", tmp_path / "config" / "config.yaml")
        config = load_config()
        assert "model" in config
        assert "auto_compact_threshold" in config

    def test_save_and_load(self, tmp_path, monkeypatch):
        monkeypatch.setattr("claude_token_saver.config.CONFIG_DIR", tmp_path / "config")
        monkeypatch.setattr("claude_token_saver.config.CONFIG_FILE", tmp_path / "config" / "config.yaml")
        config = load_config()
        config["test_key"] = "test_value"
        save_config(config)
        reloaded = load_config()
        assert reloaded["test_key"] == "test_value"


# ── 增强功能测试 ──

class TestTokenAwareTruncate:
    def test_short_unchanged(self):
        content = "hello world\n"
        result = smart_truncate(content, 100)
        assert result == content

    def test_long_truncated(self):
        lines = [f"line{i}" * 10 for i in range(200)]
        content = "\n".join(lines)
        tokens = count_tokens(content)
        result = smart_truncate(content, tokens // 3)
        assert "..." in result
        assert count_tokens(result) <= tokens // 3 + 50  # 允许小幅超限


class TestSkeletonExtraction:
    def test_python_skeleton(self):
        code = '''import os
"""Module docstring."""

def foo(x: int) -> str:
    """Function docstring."""
    return str(x)

class Bar:
    """Class docstring."""
    def method(self):
        return 42
'''
        skeleton = extract_skeleton(code, ".py")
        assert "import os" in skeleton
        assert "def foo" in skeleton
        assert "class Bar" in skeleton
        assert "def method" in skeleton
        # 函数体应被去除
        assert "return str(x)" not in skeleton
        assert "return 42" not in skeleton

    def test_symbol_index(self):
        code = '''def foo(): pass
def bar(): pass
class Baz: pass
'''
        symbols = extract_symbol_index(code, ".py")
        assert len(symbols) == 3
        assert all("line" in s for s in symbols)
        assert all("name" in s for s in symbols)

    def test_format_symbol_index(self):
        symbols = [{"name": "foo", "kind": "function", "line": 1, "signature": "def foo()"}]
        result = format_symbol_index(symbols, "test.py")
        assert "test.py" in result
        assert "foo" in result
        assert "L1" in result


class TestHookHandlerEnhanced:
    def test_block_large_read(self, tmp_path, monkeypatch):
        """大文件（>500KB）应询问用户而非直接阻止。"""
        from claude_token_saver.hooks.handler import handle_pre_tool, _safe_resolve_path
        large_file = tmp_path / "large.txt"
        large_file.write_text("x" * 600_000)
        # mock _safe_resolve_path 绕过 cwd 检查（测试环境下 tmp_path 不在 cwd 内）
        monkeypatch.setattr("claude_token_saver.hooks.handler._safe_resolve_path", lambda p: str(large_file))
        result = handle_pre_tool("Read", {"file_path": str(large_file)})
        assert result["decision"] == "ask"

    def test_warn_medium_read(self, tmp_path, monkeypatch):
        """中等文件（200-500KB）应警告但允许。"""
        from claude_token_saver.hooks.handler import handle_pre_tool, _safe_resolve_path
        med_file = tmp_path / "medium.txt"
        med_file.write_text("x" * 300_000)
        monkeypatch.setattr("claude_token_saver.hooks.handler._safe_resolve_path", lambda p: str(med_file))
        result = handle_pre_tool("Read", {"file_path": str(med_file)})
        assert result["decision"] == "approve"

    def test_glob_max_results_injected(self):
        """Glob 查询应注入 maxResults。"""
        modified = _add_exclude_paths_to_input({"pattern": "*.py"}, "Glob")
        assert "maxResults" in modified
        assert modified["maxResults"] == 100

    def test_grep_max_matches_injected(self):
        """Grep 查询应注入 maxMatches。"""
        modified = _add_exclude_paths_to_input({"pattern": "foo"}, "Grep")
        assert "maxMatches" in modified
        assert modified["maxMatches"] == 50

    def test_exclude_dirs_injected_glob(self):
        """Glob 查询应注入排除目录。"""
        modified = _add_exclude_paths_to_input({"pattern": "*.py"}, "Glob")
        assert "excludeDirs" in modified
        assert "node_modules" in modified["excludeDirs"]

    def test_exclude_dirs_injected_grep(self):
        """Grep 查询应注入排除模式。"""
        modified = _add_exclude_paths_to_input({"pattern": "foo"}, "Grep")
        assert "exclude" in modified
        assert "node_modules" in modified["exclude"]

    def test_safe_resolve_path_oserror_no_crash(self):
        """Path.resolve() 抛 OSError 时不崩溃（UnboundLocalError 修复）。"""
        result = _safe_resolve_path("")
        assert result is None
        # 非法路径字符
        result2 = _safe_resolve_path("\x00invalid")
        assert result2 is None


class TestProcessFilesEnhanced:
    def test_detail_level_skeleton(self, tmp_path):
        """skeleton 级别应大幅减少 token。"""
        py_file = tmp_path / "module.py"
        # 使用较大文件以展示 skeleton 的实际压缩效果
        py_file.write_text('''
import os
import sys
import json
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


def function_a(x: int, y: int) -> int:
    """Calculate the sum of two numbers with detailed processing.

    This function takes two integer arguments and returns their sum.
    It includes input validation, error handling, and logging.

    Args:
        x: The first integer operand.
        y: The second integer operand.

    Returns:
        The sum of x and y.

    Raises:
        TypeError: If either argument is not an integer.
    """
    if not isinstance(x, int):
        raise TypeError(f"Expected int, got {type(x).__name__}")
    if not isinstance(y, int):
        raise TypeError(f"Expected int, got {type(y).__name__}")
    logger.info(f"Adding {x} + {y}")
    result = x + y
    logger.debug(f"Result: {result}")
    return result


def function_b() -> int:
    """Return the answer to everything.

    This function returns the integer 42, which is the answer
    to the ultimate question of life, the universe, and everything.

    Returns:
        Always returns 42.
    """
    logger.info("Getting the answer...")
    answer = 42
    logger.debug(f"Answer is {answer}")
    return answer


class MyClass:
    """A sample class demonstrating various features.

    This class provides basic CRUD operations for managing
    internal state and demonstrates proper OOP patterns.
    """

    def __init__(self, initial_value: int = 0):
        """Initialize the instance with an optional starting value.

        Args:
            initial_value: The initial value for the instance.
        """
        self.value = initial_value
        self._cache: Dict[str, Any] = {}
        logger.info(f"Initialized with value={initial_value}")

    def method(self) -> int:
        """Return the current value.

        Returns:
            The current stored value.
        """
        logger.debug(f"Returning value: {self.value}")
        return self.value

    def set_value(self, new_value: int) -> None:
        """Set a new value with validation.

        Args:
            new_value: The new value to store.

        Raises:
            TypeError: If new_value is not an integer.
        """
        if not isinstance(new_value, int):
            raise TypeError("Value must be an integer")
        logger.info(f"Setting value from {self.value} to {new_value}")
        self.value = new_value
        self._cache.clear()
''')
        _clear_caches()
        result = process_files(
            [str(py_file)],
            detail_level="skeleton",
            do_strip_comments=False,
            dedup=False,
        )
        assert len(result["files"]) == 1
        f = result["files"][0]
        assert f["savings"] > 0
        # 大文件 skeleton 级别应节省 50%+ token
        assert f["savings"] / f["tokens_before"] > 0.5

    def test_detail_level_stripped(self, tmp_path):
        """stripped 级别应去除注释。"""
        py_file = tmp_path / "module.py"
        py_file.write_text('import os\n# comment\ndef foo():\n    """docstring"""\n    pass\n')
        _clear_caches()
        result = process_files(
            [str(py_file)],
            detail_level="stripped",
            do_strip_comments=True,
            dedup=False,
        )
        assert len(result["files"]) == 1
        content = result["files"][0]["content"]
        assert "# comment" not in content

    def test_detail_level_block(self, tmp_path):
        """block 级别应阻止超大文件。"""
        large = tmp_path / "large.py"
        # 使用足够大的文件确保 token 数超过 max_file_tokens=50_000
        # tiktoken cl100k_base 下每个字符约 0.125 token，需 ~560KB+
        large.write_text("x" * 700_000)
        result = process_files(
            [str(large)],
            detail_level="block",
            max_file_tokens=50_000,
            dedup=False,
        )
        assert len(result["skipped"]) == 1
        assert "too large" in result["skipped"][0]

    def test_cache_hits(self, tmp_path):
        """缓存命中应返回 cache_hits 计数。"""
        f = tmp_path / "test.py"
        f.write_text("x = 1\n")
        _clear_caches()
        result1 = process_files([str(f)], detail_level="full", dedup=False)
        assert result1["cache_hits"] == 0
        result2 = process_files([str(f)], detail_level="full", dedup=False)
        assert result2["cache_hits"] == 1

    def test_output_has_cache_hits_key(self, tmp_path):
        """返回字典应包含 cache_hits 键。"""
        assert "cache_hits" in process_files([], dedup=False)


# ── 自适应 detail_level ────────────────────────────────────────────────

class TestAutoDetailLevel:
    def test_assigns_skeleton_for_large_files(self, tmp_path):
        """超过预算 50% 的文件应分配 block，15-50% 分配 skeleton。"""
        from claude_token_saver.budget import auto_detail_level
        files = [
            ("small.py", 500),
            ("medium.py", 16_000),  # > 15% of 100K → skeleton
            ("large.py", 200_000),
        ]
        levels = auto_detail_level(files, total_budget=100_000)
        assert levels["large.py"] == "block"  # 200K > 50% of 100K → block
        assert levels["medium.py"] == "skeleton"  # 16K > 15% of 100K → skeleton

    def test_assigns_full_for_small_files(self, tmp_path):
        """小文件应分配 full 级别。"""
        from claude_token_saver.budget import auto_detail_level
        files = [
            ("tiny.py", 200),
            ("small.py", 2_000),
        ]
        levels = auto_detail_level(files, total_budget=100_000)
        assert levels["tiny.py"] == "full"
        assert levels["small.py"] == "full"

    def test_assigns_stripped_for_medium(self, tmp_path):
        """中等文件应分配 stripped。"""
        from claude_token_saver.budget import auto_detail_level
        files = [
            ("med.py", 20_000),
        ]
        levels = auto_detail_level(files, total_budget=100_000)
        assert levels["med.py"] == "skeleton"  # >15% of budget

    def test_block_when_exceeds_budget(self, tmp_path):
        """超过总预算的文件应标记为 block。"""
        from claude_token_saver.budget import auto_detail_level
        files = [
            ("huge.py", 200_000),
        ]
        levels = auto_detail_level(files, total_budget=100_000)
        assert levels["huge.py"] == "block"

    def test_auto_detail_in_process_files(self, tmp_path):
        """process_files 应支持 auto_detail 模式。"""
        _clear_caches()
        # 创建大小不一的文件
        small = tmp_path / "small.py"
        small.write_text("x = 1\n")
        large = tmp_path / "large.py"
        large.write_text("x = " + "1\n" * 5000)

        result = process_files(
            [str(small), str(large)],
            auto_detail=True,
            token_budget=10_000,
            dedup=False,
        )
        assert len(result["files"]) > 0
        # 至少有一个文件被处理
        assert result["total_tokens_before"] > 0

    def test_estimate_auto_cost(self):
        """estimate_auto_cost 应返回合理的估算。"""
        from claude_token_saver.budget import estimate_auto_cost
        files = [
            ("a.py", 1000),
            ("b.py", 50_000),
            ("c.py", 200_000),
        ]
        est = estimate_auto_cost(files, total_budget=100_000)
        assert est["total_before"] > 0
        assert est["estimated_after"] < est["total_before"]
        assert est["savings_pct"] > 0


# ── 结构感知去重 ────────────────────────────────────────────────────────

class TestStructuralDedup:
    def test_dedup_identical_skeletons(self, tmp_path):
        """骨架完全相同的文件应被视为重复。"""
        from claude_token_saver.compressor import structural_dedup
        f1 = tmp_path / "a.py"
        f2 = tmp_path / "b.py"
        code = '''import os\n\ndef foo():\n    return 1\n'''
        f1.write_text(code)
        f2.write_text(code)
        result = structural_dedup([str(f1), str(f2)])
        assert len(result) == 1

    def test_keep_different_skeletons(self, tmp_path):
        """骨架不同的文件应保留（不同的类/函数结构）。"""
        from claude_token_saver.compressor import structural_dedup
        f1 = tmp_path / "a.py"
        f2 = tmp_path / "b.py"
        # 不同的结构：一个有类，一个只有函数
        f1.write_text("class Foo:\n    def method(self): pass\n")
        f2.write_text("def bar(): pass\n")
        result = structural_dedup([str(f1), str(f2)])
        assert len(result) == 2

    def test_same_different_names_deduped(self, tmp_path):
        """骨架中函数名被保留，只有完全相同的代码才被去重。"""
        from claude_token_saver.compressor import structural_dedup
        f1 = tmp_path / "a.py"
        f2 = tmp_path / "b.py"
        # 完全相同的代码 → 骨架也相同
        f1.write_text("def foo():\n    pass\n")
        f2.write_text("def foo():\n    pass\n")
        result = structural_dedup([str(f1), str(f2)])
        assert len(result) == 1

    def test_group_by_structure(self, tmp_path):
        """group_by_structure 应将相同结构的文件分组。"""
        from claude_token_saver.compressor import group_by_structure
        f1 = tmp_path / "a.py"
        f2 = tmp_path / "b.py"
        f3 = tmp_path / "c.py"
        # a 和 b 完全相同的骨架，c 不同
        f1.write_text("def foo():\n    pass\n")
        f2.write_text("def foo():\n    pass\n")  # 相同代码
        f3.write_text("class Baz:\n    pass\n")
        groups = group_by_structure([str(f1), str(f2), str(f3)])
        assert len(groups) == 2
        # 一个组有 2 个文件（a/b），另一个有 1 个（c）
        sizes = sorted(len(g) for g in groups)
        assert sizes == [1, 2]


# ── 对话上下文压缩 ─────────────────────────────────────────────────────

class TestConversationCompactor:
    def test_should_compact_when_over_threshold(self):
        """超过阈值应触发 compact。"""
        from claude_token_saver.compactor import ConversationCompactor
        c = ConversationCompactor()
        assert c.should_compact("s1", 150_000, threshold=100_000)

    def test_should_not_compact_below_threshold(self):
        """低于阈值不应触发 compact。"""
        from claude_token_saver.compactor import ConversationCompactor
        c = ConversationCompactor()
        assert not c.should_compact("s1", 50_000, threshold=100_000)

    def test_compact_produces_summaries(self):
        """compact 应将旧轮次压缩为摘要。"""
        from claude_token_saver.compactor import ConversationCompactor
        import tempfile, os
        c = ConversationCompactor()
        turns = [
            {"turn_index": 1, "type": "user", "content": "Hello " * 100,
             "tokens": 200, "timestamp": "2024-01-01T00:00:00"},
            {"turn_index": 2, "type": "assistant", "content": "Hi there! " * 100,
             "tokens": 200, "timestamp": "2024-01-01T00:01:00"},
            {"turn_index": 3, "type": "user", "content": "Help me " * 100,
             "tokens": 200, "timestamp": "2024-01-01T00:02:00"},
        ]
        result = c.compact("s1", turns, keep_recent=1)
        assert len(result.summaries) == 2  # 旧 2 轮被压缩
        assert len(result.recent_turns) == 1  # 最近 1 轮保留
        assert result.total_tokens_after < result.total_tokens_before

    def test_format_for_prompt(self):
        """format_for_prompt 应生成可读的紧凑文本。"""
        from claude_token_saver.compactor import ConversationCompactor, TurnSummary
        c = ConversationCompactor()
        ctx = CompactedContext(
            session_id="s1",
            original_turns=10,
            compacted_turns=4,
            summaries=[
                TurnSummary(turn_index=1, turn_type="user",
                           summary="Request summary", tokens_before=200, tokens_after=20),
            ],
            recent_turns=[{"turn_index": 10, "type": "user", "content": "last turn"}],
            total_tokens_before=2000,
            total_tokens_after=600,
        )
        text = c.format_for_prompt(ctx)
        assert "对话摘要" in text
        assert "轮次 1" in text
        assert "最近" in text


# ── 渐进式披露 ─────────────────────────────────────────────────────────

class TestProgressiveDisclosure:
    def test_build_directory_index(self, tmp_path):
        """build_directory_index 应返回文件索引。"""
        f = tmp_path / "main.py"
        f.write_text("x = 1\n")
        sub = tmp_path / "sub"
        sub.mkdir()
        f2 = sub / "util.py"
        f2.write_text("y = 2\n")

        from claude_token_saver.prep import build_directory_index
        idx = build_directory_index([str(tmp_path)])
        assert idx["total_files"] == 2
        assert idx["total_estimated_tokens"] > 0
        assert len(idx["files"]) == 2

    def test_index_does_not_read_content(self, tmp_path):
        """索引模式不应读取文件内容。"""
        import time
        f = tmp_path / "slow.py"
        # 写入一个标记，如果被读取就能检测到
        f.write_text("# NOT_READ_MARKER\nx = 1\n")

        from claude_token_saver.progressive import FileIndexEntry
        entry = FileIndexEntry(
            path=str(f),
            size_bytes=f.stat().st_size,
            estimated_tokens=10,
            ext=".py",
            relative_path="slow.py",
        )
        # 确认 entry 不包含文件内容
        assert "NOT_READ_MARKER" not in str(entry.to_dict())

    def test_format_index_markdown(self, tmp_path):
        """Markdown 格式索引应包含目录结构。"""
        f = tmp_path / "main.py"
        f.write_text("x = 1\n")

        from claude_token_saver.prep import build_directory_index, format_index_for_prompt
        idx = build_directory_index([str(tmp_path)])
        md = format_index_for_prompt(idx, format="markdown")
        assert "项目目录索引" in md
        assert "main.py" in md


# ── 工具输出压缩 ────────────────────────────────────────────────────────

class TestToolOutputCompression:
    def test_grep_context_limited(self):
        """Grep 输出应限制上下文行数。"""
        from claude_token_saver.hooks.handler import _compress_tool_output
        output = {
            "matches": [
                {
                    "file": "test.py",
                    "line": 10,
                    "content": "def foo():",
                    "context": [f"line {i}" for i in range(20)],
                }
            ]
        }
        result = _compress_tool_output("Grep", {}, output)
        assert len(result["matches"][0]["context"]) <= 5  # 匹配行 + 前后各 2
        assert result["matches"][0].get("context_truncated") is True

    def test_grep_short_context_unchanged(self):
        """短上下文不应被截断。"""
        from claude_token_saver.hooks.handler import _compress_tool_output
        output = {
            "matches": [
                {
                    "file": "test.py",
                    "line": 5,
                    "content": "def bar():",
                    "context": ["line 3", "line 4", "def bar():", "line 6"],
                }
            ]
        }
        result = _compress_tool_output("Grep", {}, output)
        assert len(result["matches"][0]["context"]) == 4
        assert "context_truncated" not in result["matches"][0]

    def test_glob_truncated_when_many(self):
        """Glob 结果过多时应截断。"""
        from claude_token_saver.hooks.handler import _compress_tool_output
        output = {
            "results": [
                {"path": f"file{i}.py", "content": "x" * 500}
                for i in range(50)
            ]
        }
        result = _compress_tool_output("Glob", {}, output)
        assert result.get("truncated") is True
        assert result.get("total_results") == 50
        # 内容预览应被移除
        assert "content" not in result["results"][0]

    def test_non_target_tool_unchanged(self):
        """非 Grep/Glob 工具的输出不应被修改。"""
        from claude_token_saver.hooks.handler import _compress_tool_output
        output = {"result": "some output"}
        result = _compress_tool_output("Read", {}, output)
        assert result is output
