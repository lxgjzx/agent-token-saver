"""Tests for claude-token-saver"""
from __future__ import annotations

import json
import pytest

from claude_token_saver.utils import count_tokens, is_binary_file, should_ignore
from claude_token_saver.prep import (
    strip_comments, strip_python_docstrings, smart_truncate,
    deduplicate_files, compress_prompt, process_files, _clear_caches,
)
from claude_token_saver.compressor import extract_skeleton, extract_symbol_index, format_symbol_index
from claude_token_saver.hooks.handler import _safe_resolve_path, _add_exclude_paths_to_input
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
        """大文件（>500KB）应被阻止。"""
        from claude_token_saver.hooks.handler import handle_pre_tool, _safe_resolve_path
        large_file = tmp_path / "large.txt"
        large_file.write_text("x" * 600_000)
        # mock _safe_resolve_path 绕过 cwd 检查（测试环境下 tmp_path 不在 cwd 内）
        monkeypatch.setattr("claude_token_saver.hooks.handler._safe_resolve_path", lambda p: str(large_file))
        result = handle_pre_tool("Read", {"file_path": str(large_file)})
        assert result["decision"] == "block"

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
