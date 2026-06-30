"""Tests for claude-token-saver"""
from __future__ import annotations

import json
import pytest

from claude_token_saver.utils import count_tokens, is_binary_file, should_ignore
from claude_token_saver.prep import (
    strip_comments, strip_python_docstrings, smart_truncate,
    deduplicate_files, compress_prompt, process_files, _clear_caches,
    _remove_encoding_and_shebang, _simplify_boolean_checks,
    _remove_redundant_return_none, _simplify_isinstance_checks,
    _remove_empty_special_methods, _compress_common_patterns,
    _remove_fstring_no_interpolation, _remove_else_after_flow_control,
    _remove_try_except_pass, _compress_truthiness_checks,
    _compress_empty_collections, _merge_nested_ifs,
    _simplify_ternary, _remove_tuple_wrap_single,
    _list_to_generator, _remove_not_not, _remove_redundant_parens,
    _remove_list_wrap, _flatten_nested_ternary,
    _remove_unused_functions, _inline_single_use_vars, _remove_unused_classes,
    _invert_dead_if, _merge_duplicate_conditions,
    _range_len_to_enumerate, _remove_return_none_after_none_check,
    _remove_type_annotations, _remove_unused_typing_imports,
    _remove_unused_imports, _fold_constants, _remove_unused_locals,
    _remove_unreachable_code, _simplify_enumerate_start_zero,
    _remove_unused_except_blocks, _simplify_super_calls,
    _collapse_duplicate_lines, _compress_asserts,
    _merge_same_body_conditions, _remove_dead_after_loop,
    _merge_adjacent_string_literals, _simplify_bool_expr_with_const,
    _simplify_none_check_return, _simplify_isinstance_and_not_in,
    _remove_try_except_reraise, _merge_consecutive_attr_assignments,
    _remove_await_noop, _simplify_identity_checks, _remove_empty_with,
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
        """format_for_prompt 应生成紧凑的可读文本。"""
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
        assert "摘要" in text
        assert "T1U" in text
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


# ── .gitignore 感知 ────────────────────────────────────────────────────

class TestGitignoreAwareness:
    def test_ignores_node_modules(self, tmp_path):
        """node_modules 应被识别为忽略。"""
        nm = tmp_path / "node_modules" / "pkg"
        nm.mkdir(parents=True)
        from claude_token_saver.gitignore import is_gitignored
        assert is_gitignored(nm)

    def test_ignores_pycache(self, tmp_path):
        """__pycache__ 应被识别为忽略。"""
        pc = tmp_path / "__pycache__"
        pc.mkdir()
        from claude_token_saver.gitignore import is_gitignored
        assert is_gitignored(pc)

    def test_normal_file_not_ignored(self, tmp_path):
        """普通文件不应被忽略。"""
        f = tmp_path / "main.py"
        f.write_text("x = 1\n")
        from claude_token_saver.gitignore import is_gitignored
        assert not is_gitignored(f)

    def test_gitignore_file_rules(self, tmp_path):
        """.gitignore 中声明的文件应被识别。"""
        gitignore = tmp_path / ".gitignore"
        gitignore.write_text("*.log\nbuild/\n")
        f = tmp_path / "debug.log"
        f.write_text("log")
        from claude_token_saver.gitignore import is_gitignored
        assert is_gitignored(f)

    def test_extra_patterns(self, tmp_path):
        """额外硬编码规则应生效。"""
        from claude_token_saver.gitignore import should_ignore_with_gitignore
        f = tmp_path / ".DS_Store"
        f.write_text("x")
        assert should_ignore_with_gitignore(f)


# ── 路径缩写 ────────────────────────────────────────────────────────────

class TestPathOptimizer:
    def test_abbreviate_windows_path(self):
        """Windows 路径应被缩写。"""
        from claude_token_saver.path_optimizer import abbreviate_path
        result = abbreviate_path("C:\\Users\\smy\\project\\main.py")
        assert "project" in result
        assert "Users" not in result or len(result) < len("C:\\Users\\smy\\project\\main.py")

    def test_abbreviate_unix_path(self):
        """Unix 路径应被缩写。"""
        from claude_token_saver.path_optimizer import abbreviate_path
        result = abbreviate_path("/home/user/project/main.py")
        assert "~" in result
        assert len(result) < len("/home/user/project/main.py")

    def test_relative_path_preferred(self):
        """应优先使用相对路径。"""
        from claude_token_saver.path_optimizer import abbreviate_path
        import tempfile, os
        with tempfile.TemporaryDirectory() as tmp:
            f = os.path.join(tmp, "main.py")
            result = abbreviate_path(f, project_root=tmp)
            assert result == "main.py"

    def test_build_abbreviation_map(self):
        """build_path_abbreviation_map 应生成正确映射。"""
        from claude_token_saver.path_optimizer import build_path_abbreviation_map
        paths = [
            "C:\\Users\\smy\\project\\src\\main.py",
            "C:\\Users\\smy\\project\\tests\\test.py",
        ]
        mapping = build_path_abbreviation_map(paths)
        assert len(mapping) > 0


# ── 对话 Diff 压缩 ─────────────────────────────────────────────────────

class TestConversationDiff:
    def test_extract_changed_files(self):
        """应提取工具调用中的文件路径。"""
        from claude_token_saver.conversation_diff import _extract_changed_files
        tool_uses = [
            {"name": "Read", "input": {"file_path": "/path/to/main.py"}},
            {"name": "Write", "input": {"file_path": "/path/to/util.py"}},
        ]
        files = _extract_changed_files(tool_uses)
        assert len(files) == 2
        assert "main.py" in files

    def test_extract_errors(self):
        """应提取失败的工具调用。"""
        from claude_token_saver.conversation_diff import _extract_errors
        tool_uses = [
            {"name": "Read", "is_error": True, "result_content": "Permission denied"},
            {"name": "Write", "is_error": False, "result_content": "OK"},
        ]
        errors = _extract_errors(tool_uses)
        assert len(errors) == 1

    def test_compress_conversation_produces_summaries(self):
        """压缩后旧轮次应为摘要模式。"""
        from claude_token_saver.conversation_diff import compress_conversation_diff
        turns = [
            {"turn_index": i, "type": "user" if i % 2 == 0 else "assistant",
             "content": f"Turn {i} content " * 20, "tokens": 200}
            for i in range(1, 15)
        ]
        result = compress_conversation_diff(turns, summary_after=5)
        assert result.compression_ratio > 0
        # 旧轮次应为摘要
        summary_turns = [t for t in result.turns if t.is_summary]
        assert len(summary_turns) > 0

    def test_recent_turns_not_summarized(self):
        """最近 N 轮不应被摘要。"""
        from claude_token_saver.conversation_diff import compress_conversation_diff
        turns = [
            {"turn_index": i, "type": "user",
             "content": f"Turn {i}", "tokens": 50}
            for i in range(1, 8)
        ]
        result = compress_conversation_diff(turns, summary_after=5)
        # 最后 2 轮不应被摘要（7 - 5 = 2）
        non_summary = [t for t in result.turns if not t.is_summary]
        assert len(non_summary) >= 2


# ── Hook 输出最小化 ────────────────────────────────────────────────────

class TestHookOptimizer:
    def test_minimize_removes_empty_fields(self):
        """应移除空值字段。"""
        from claude_token_saver.hook_optimizer import minimize_hook_output
        result = minimize_hook_output({"decision": "approve", "reason": "", "extra": None})
        assert "reason" not in result or result.get("reason") == ""
        assert "extra" not in result

    def test_compresses_long_reason(self):
        """应截断过长的 reason。"""
        from claude_token_saver.hook_optimizer import compress_reason_text
        long_reason = "文件过大 " + "x" * 300 + " 建议使用 --offset/--limit"
        compressed = compress_reason_text(long_reason)
        assert len(compressed) <= 200

    def test_preserves_block_decision(self):
        """应保留 ask 决策。"""
        from claude_token_saver.hook_optimizer import minimize_hook_output
        result = minimize_hook_output({"decision": "ask", "reason": "File too large"})
        assert result["d"] == "ask"


# ── Read 结果去重 ────────────────────────────────────────────────────────

class TestReadDedup:
    def test_cache_hit_unchanged_file(self, tmp_path, monkeypatch):
        """未变更的文件应命中缓存。"""
        from claude_token_saver.hooks.handler import _check_read_cache, _update_read_cache
        f = tmp_path / "test.py"
        f.write_text("x = 1\n")
        _update_read_cache(str(f), "x = 1\n")
        hit, tokens = _check_read_cache(str(f))
        assert hit is True
        assert tokens > 0

    def test_cache_miss_changed_file(self, tmp_path):
        """内容变更的文件应缓存未命中。"""
        from claude_token_saver.hooks.handler import _check_read_cache, _update_read_cache
        f = tmp_path / "test.py"
        f.write_text("x = 1\n")
        _update_read_cache(str(f), "x = 1\n")
        hit, _ = _check_read_cache(str(f))
        assert hit is True
        # After changing content on disk and updating cache
        f.write_text("y = 2\n")
        _update_read_cache(str(f), "y = 2\n")
        # 已更新缓存 → 应命中
        hit2, _ = _check_read_cache(str(f))
        assert hit2 is True
        # 清空缓存后应 miss
        from claude_token_saver.hooks.handler import clear_read_cache
        clear_read_cache()
        hit3, _ = _check_read_cache(str(f))
        assert hit3 is False

    def test_clear_cache(self, tmp_path):
        """清空缓存后应未命中。"""
        from claude_token_saver.hooks.handler import _check_read_cache, _update_read_cache, clear_read_cache
        f = tmp_path / "test.py"
        f.write_text("x = 1\n")
        _update_read_cache(str(f), "x = 1\n")
        clear_read_cache()
        hit, _ = _check_read_cache(str(f))
        assert hit is False


# ── 空白标准化 ──────────────────────────────────────────────────────────

class TestNormalizeWhitespace:
    def test_removes_trailing_whitespace(self):
        """应去除行尾空白。"""
        from claude_token_saver.prep import normalize_whitespace
        result = normalize_whitespace("x = 1  \ny = 2  \n", ".py")
        assert result == "x = 1\ny = 2"

    def test_collapses_excessive_blank_lines_py(self):
        """Python 文件应折叠连续空行为最多 2 个。"""
        from claude_token_saver.prep import normalize_whitespace
        result = normalize_whitespace("a\n\n\n\n\nb\n", ".py")
        # Should have at most 2 consecutive blank lines
        assert "\n\n\n\n" not in result

    def test_collapses_excessive_blank_lines_yaml(self):
        """YAML 文件应折叠连续空行为最多 1 个。"""
        from claude_token_saver.prep import normalize_whitespace
        result = normalize_whitespace("a\n\n\n\n\nb\n", ".yaml")
        assert "\n\n\n" not in result

    def test_strips_leading_trailing_blanks(self):
        """应去除首尾空行。"""
        from claude_token_saver.prep import normalize_whitespace
        result = normalize_whitespace("\n\nhello\n\n", ".py")
        assert not result.startswith("\n")
        assert not result.endswith("\n")


# ── 增量上下文 ──────────────────────────────────────────────────────────

class TestIncrementalContext:
    def test_detect_new_files(self):
        """应检测新文件。"""
        from claude_token_saver.incremental_context import IncrementalContextManager
        mgr = IncrementalContextManager("s1")
        changes = mgr.detect_changes(["/tmp/a.py", "/tmp/b.py"],
                                     read_content_fn=lambda p: "content")
        assert len(changes) == 2
        assert all(c.change_type == "new" for c in changes)

    def test_detect_unchanged_files(self):
        """应检测未变更文件。"""
        from claude_token_saver.incremental_context import IncrementalContextManager
        mgr = IncrementalContextManager("s1")
        mgr.detect_changes(["/tmp/a.py"], read_content_fn=lambda p: "content")
        changes = mgr.detect_changes(["/tmp/a.py"],
                                     read_content_fn=lambda p: "content")
        assert len(changes) == 1
        assert changes[0].change_type == "unchanged"

    def test_detect_modified_files(self):
        """应检测已修改文件。"""
        from claude_token_saver.incremental_context import IncrementalContextManager
        mgr = IncrementalContextManager("s1")
        mgr.detect_changes(["/tmp/a.py"], read_content_fn=lambda p: "v1")
        changes = mgr.detect_changes(["/tmp/a.py"],
                                     read_content_fn=lambda p: "v2")
        assert len(changes) == 1
        assert changes[0].change_type == "modified"

    def test_detect_deleted_files(self):
        """应检测已删除文件。"""
        from claude_token_saver.incremental_context import IncrementalContextManager
        mgr = IncrementalContextManager("s1")
        mgr.detect_changes(["/tmp/a.py", "/tmp/b.py"],
                           read_content_fn=lambda p: "content")
        changes = mgr.detect_changes(["/tmp/a.py"],
                                     read_content_fn=lambda p: "content")
        deleted = [c for c in changes if c.change_type == "deleted"]
        assert len(deleted) == 1
        assert deleted[0].path == "/tmp/b.py"

    def test_format_incremental_context(self):
        """格式化增量上下文应包含标记。"""
        from claude_token_saver.incremental_context import (
            IncrementalContextManager, FileChange,
        )
        mgr = IncrementalContextManager("s1")
        mgr.detect_changes(["/tmp/a.py"], read_content_fn=lambda p: "content")
        changes = [
            FileChange(path="/tmp/b.py", change_type="new"),
            FileChange(path="/tmp/a.py", change_type="unchanged"),
        ]
        text = mgr.format_incremental(changes, lambda p: "content")
        assert "新增文件" in text
        assert "未变更" in text


# ── Token 预算优化器 ────────────────────────────────────────────────────

class TestTokenBudget:
    def test_fit_to_budget(self):
        """应能将内容适配到预算内。"""
        from claude_token_saver.token_budget import fit_to_budget
        items = [
            ("a.py", "x = 1\n", 10),
            ("b.py", "y = 2\n" * 100, 10),
        ]
        result = fit_to_budget(items, budget=500)
        total = sum(r["tokens"] for r in result)
        assert total <= 500

    def test_prioritize_small_first(self):
        """应按大小升序排列。"""
        from claude_token_saver.token_budget import prioritize_files
        ordered = prioritize_files([("a.py", 1000), ("b.py", 100), ("c.py", 500)], "small_first")
        assert ordered == ["b.py", "c.py", "a.py"]

    def test_empty_budget(self):
        """零预算应返回空列表。"""
        from claude_token_saver.token_budget import fit_to_budget
        result = fit_to_budget([("a.py", "x=1", 10)], budget=0)
        assert len(result) == 0


# ── 常见文件组去重 ──────────────────────────────────────────────────────

class TestCommonDedup:
    def test_detect_init_pattern(self):
        """应识别 __init__.py 模式。"""
        from claude_token_saver.common_dedup import detect_common_pattern
        result = detect_common_pattern(__import__("pathlib").Path("__init__.py"))
        assert result is not None
        assert result["strategy"] == "keep_first_per_dir"

    def test_no_match_normal_file(self):
        """普通文件不应匹配常见模式。"""
        from claude_token_saver.common_dedup import detect_common_pattern
        result = detect_common_pattern(__import__("pathlib").Path("main.py"))
        assert result is None

    def test_filter_different_dirs_kept(self, tmp_path):
        """不同目录的 __init__.py 应都被保留。"""
        from claude_token_saver.common_dedup import filter_common_duplicates
        f1 = tmp_path / "pkg1" / "__init__.py"
        f2 = tmp_path / "pkg2" / "__init__.py"
        f1.parent.mkdir()
        f2.parent.mkdir()
        f1.write_text("")
        f2.write_text("")
        result, skipped = filter_common_duplicates([str(f1), str(f2)])
        assert len(result) == 2

    def test_content_dedup_identical(self, tmp_path):
        """内容相同的文件应去重。"""
        from claude_token_saver.common_dedup import filter_common_duplicates
        f1 = tmp_path / "setup.py"
        f2 = tmp_path / "sub" / "setup.py"
        f2.parent.mkdir()
        f1.write_text("from setuptools import setup\n")
        f2.write_text("from setuptools import setup\n")
        result, skipped = filter_common_duplicates([str(f1), str(f2)])
        assert len(result) == 1

    def test_suggestion_generated(self):
        """应生成去重建议。"""
        from claude_token_saver.common_dedup import get_common_pattern_summary
        suggestions = get_common_pattern_summary([
            "/tmp/a/__init__.py", "/tmp/b/__init__.py"
        ])
        assert len(suggestions) > 0
        assert "__init__.py" in suggestions[0]


# ── SimHash 近似重复检测 ────────────────────────────────────────────────

class TestSimHashDedup:
    def test_same_content_same_hash(self):
        """相同内容应产生相同的 simhash。"""
        from claude_token_saver.simhash_dedup import compute_simhash
        h1 = compute_simhash("def foo(): pass\ndef bar(): pass\n")
        h2 = compute_simhash("def foo(): pass\ndef bar(): pass\n")
        assert h1 == h2

    def test_different_content_different_hash(self):
        """不同内容应产生不同的 simhash。"""
        from claude_token_saver.simhash_dedup import compute_simhash
        h1 = compute_simhash("def foo(): pass\n")
        h2 = compute_simhash("def bar(): pass\n")
        assert h1 != h2

    def test_returns_int(self):
        """simhash 应返回 int 类型。"""
        from claude_token_saver.simhash_dedup import compute_simhash
        result = compute_simhash("hello world")
        assert isinstance(result, int)

    def test_short_content_returns_int(self):
        """短内容也应返回 int（非 str）。"""
        from claude_token_saver.simhash_dedup import compute_simhash
        result = compute_simhash("hi")
        assert isinstance(result, int)

    def test_hamming_distance_identical(self):
        """相同 hash 的汉明距离应为 0。"""
        from claude_token_saver.simhash_dedup import compute_simhash, hamming_distance
        h = compute_simhash("def foo(): pass\n")
        assert hamming_distance(h, h) == 0

    def test_hamming_distance_different(self):
        """不同 hash 的汉明距离应 > 0。"""
        from claude_token_saver.simhash_dedup import compute_simhash, hamming_distance
        h1 = compute_simhash("def foo(): pass\n" * 10)
        h2 = compute_simhash("def bar(): pass\n" * 10)
        assert hamming_distance(h1, h2) > 0

    def test_similarity_identical(self):
        """相同内容相似度应为 1.0。"""
        from claude_token_saver.simhash_dedup import compute_simhash, similarity_score
        h = compute_simhash("def foo(): pass\n" * 10)
        assert similarity_score(h, h) == 1.0

    def test_similarity_different(self):
        """不同内容相似度应 < 1.0。"""
        from claude_token_saver.simhash_dedup import compute_simhash, similarity_score
        h1 = compute_simhash("def foo(): pass\n" * 20)
        h2 = compute_simhash("def bar(): pass\n" * 20)
        assert 0 <= similarity_score(h1, h2) < 1.0

    def test_find_near_duplicates_none(self, tmp_path):
        """无重复文件时应返回空列表。"""
        from claude_token_saver.simhash_dedup import find_near_duplicates
        f1 = tmp_path / "a.py"
        f2 = tmp_path / "b.py"
        f1.write_text("def foo(): pass\n")
        f2.write_text("class Foo:\n    pass\n")
        groups = find_near_duplicates([str(f1), str(f2)], threshold=3)
        assert len(groups) == 0

    def test_find_near_duplicates_identical(self, tmp_path):
        """相同文件应被识别为重复。"""
        from claude_token_saver.simhash_dedup import find_near_duplicates
        f1 = tmp_path / "a.py"
        f2 = tmp_path / "b.py"
        content = "def foo(): pass\ndef bar(): pass\n" * 5
        f1.write_text(content)
        f2.write_text(content)
        groups = find_near_duplicates([str(f1), str(f2)], threshold=0)
        assert len(groups) == 1
        assert len(groups[0].duplicates) == 1

    def test_find_near_duplicates_empty_list(self):
        """空文件列表应返回空结果。"""
        from claude_token_saver.simhash_dedup import find_near_duplicates
        assert find_near_duplicates([]) == []

    def test_get_near_dup_suggestions(self):
        """应生成可读建议文本。"""
        from claude_token_saver.simhash_dedup import (
            find_near_duplicates, get_near_dup_suggestions, NearDuplicateGroup,
        )
        groups = [NearDuplicateGroup(
            representative="/tmp/a.py",
            duplicates=["/tmp/b.py"],
            similarity=0.95,
            fingerprint=12345,
        )]
        suggestions = get_near_dup_suggestions(groups)
        assert len(suggestions) == 1
        assert "近似重复" in suggestions[0]
        assert "95%" in suggestions[0]


# ── 压缩管线 ─────────────────────────────────────────────────────────────

class TestCompressionPipeline:
    def test_run_returns_tuple(self):
        """run() 应返回 (str, dict) 元组。"""
        from claude_token_saver.compression_pipeline import CompressionPipeline
        p = CompressionPipeline(ext=".py", detail_level="full")
        result = p.run("def foo(): pass\n")
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], str)
        assert isinstance(result[1], dict)

    def test_metadata_fields(self):
        """metadata 应包含所有必要字段。"""
        from claude_token_saver.compression_pipeline import CompressionPipeline
        p = CompressionPipeline(ext=".py", detail_level="full")
        _, meta = p.run("def foo(): pass\n")
        assert "original_tokens" in meta
        assert "final_tokens" in meta
        assert "stages_run" in meta
        assert "savings_per_stage" in meta
        assert "total_savings" in meta

    def test_stages_run_tracking(self):
        """应正确跟踪执行的阶段。"""
        from claude_token_saver.compression_pipeline import CompressionPipeline
        # 使用有多余空行的内容以触发 whitespace normalization
        p = CompressionPipeline(ext=".py", detail_level="stripped")
        _, meta = p.run("# comment\n\n\ndef foo(): pass\n\n\n")
        assert "normalize_whitespace" in meta["stages_run"]
        assert "strip_comments" in meta["stages_run"]

    def test_no_negative_savings(self):
        """总节省量应 >= 0（不会让内容变大）。"""
        from claude_token_saver.compression_pipeline import CompressionPipeline
        p = CompressionPipeline(ext=".py", detail_level="full")
        _, meta = p.run("def foo(): pass\n")
        assert meta["total_savings"] >= 0

    def test_skeleton_reduces_tokens(self):
        """skeleton 级别应显著减少 token 数。"""
        from claude_token_saver.compression_pipeline import CompressionPipeline
        content = "def foo():\n    x = 1\n    return x\n" * 10
        p = CompressionPipeline(ext=".py", detail_level="skeleton")
        _, meta = p.run(content)
        assert meta["total_savings"] > 0
        assert meta["final_tokens"] < meta["original_tokens"]

    def test_smart_truncate_respects_budget(self):
        """smart_truncate 应将内容截断到预算附近。"""
        from claude_token_saver.compression_pipeline import CompressionPipeline
        long_content = "def foo():\n    return 1\n" * 200
        p = CompressionPipeline(ext=".py", detail_level="full", max_tokens=200)
        result, meta = p.run(long_content)
        assert "smart_truncate" in meta["stages_run"]
        # 截断后应显著减少 tokens，且不超过预算的 1.5 倍（允许估算误差）
        assert meta["final_tokens"] <= 300
        assert meta["total_savings"] > 0

    def test_no_truncate_when_under_budget(self):
        """内容在预算内时不应截断。"""
        from claude_token_saver.compression_pipeline import CompressionPipeline
        short_content = "def foo(): pass\n"
        p = CompressionPipeline(ext=".py", detail_level="full", max_tokens=1000)
        _, meta = p.run(short_content)
        assert "smart_truncate" not in meta["stages_run"]

    def test_whitespace_normalization(self):
        """空白标准化应能去除多余空白。"""
        from claude_token_saver.compression_pipeline import CompressionPipeline
        content = "def foo():\n\n\n    x = 1\n\n\n    return x\n\n\n"
        p = CompressionPipeline(ext=".py", detail_level="full")
        result, _ = p.run(content)
        # 不应有连续 3 个以上空行
        assert "\n\n\n\n" not in result

    def test_full_level_keeps_comments(self):
        """full 级别应保留注释。"""
        from claude_token_saver.compression_pipeline import CompressionPipeline
        content = "# this is a comment\ndef foo(): pass\n"
        p = CompressionPipeline(ext=".py", detail_level="full")
        result, _ = p.run(content)
        assert "this is a comment" in result

    def test_empty_content(self):
        """空内容不应报错。"""
        from claude_token_saver.compression_pipeline import CompressionPipeline
        p = CompressionPipeline(ext=".py", detail_level="full")
        result, meta = p.run("")
        assert isinstance(result, str)
        assert meta["original_tokens"] == 0

    def test_structured_compression_json(self):
        """JSON 文件应被压缩。"""
        from claude_token_saver.compression_pipeline import CompressionPipeline
        content = '{\n  "key": "value",\n  "num": 42\n}\n'
        p = CompressionPipeline(ext=".json", detail_level="full")
        result, _ = p.run(content)
        # minify 应去除多余空白
        assert result.count("\n") <= 1


# ── Raw 输出格式 ──────────────────────────────────────────────────────────

class TestRawFormat:
    def test_raw_format_no_framing(self):
        """raw 格式不应包含 markdown 代码块标记。"""
        from claude_token_saver.prep import format_processed_output
        result = format_processed_output({
            "files": [{"path": "/tmp/test.py", "content": "x = 1", "tokens_after": 2, "tokens_before": 10, "savings": 8}],
        }, format="raw")
        assert "```" not in result
        assert "###" not in result

    def test_raw_format_has_path_marker(self):
        """raw 格式应包含路径分隔符。"""
        from claude_token_saver.prep import format_processed_output
        result = format_processed_output({
            "files": [{"path": "/tmp/test.py", "content": "x = 1", "tokens_after": 2, "tokens_before": 10, "savings": 8}],
        }, format="raw")
        assert "===" in result
        assert "test.py" in result

    def test_raw_format_more_compact_than_markdown(self):
        """raw 格式应比 markdown 格式更紧凑。"""
        from claude_token_saver.prep import format_processed_output
        data = {
            "files": [
                {"path": f"/tmp/f{i}.py", "content": "x = 1\n", "tokens_after": 2, "tokens_before": 10, "savings": 8}
                for i in range(5)
            ],
        }
        raw = format_processed_output(data, format="raw")
        md = format_processed_output(data, format="markdown")
        assert len(raw) < len(md)


# ── 增强 compress_prompt ──────────────────────────────────────────────────

class TestCompressPrompt:
    def test_basic_compression(self):
        """应去除多余空白。"""
        from claude_token_saver.prep import compress_prompt
        result = compress_prompt("hello\n\n\n\n\nworld\n\n\n\n\n")
        assert "\n\n\n" not in result

    def test_respects_max_tokens(self):
        """超长内容应被显著截断。"""
        from claude_token_saver.prep import compress_prompt
        long_text = "line\n" * 1000
        result = compress_prompt(long_text, max_tokens=50)
        from claude_token_saver.utils import count_tokens
        # 应显著小于原文（原文约 5000 tokens）
        assert count_tokens(result) < count_tokens(long_text) * 0.5

    def test_short_content_unchanged(self):
        """短内容不应被修改。"""
        from claude_token_saver.prep import compress_prompt
        text = "hello world"
        result = compress_prompt(text, max_tokens=1000)
        assert result == text


# ── 对话摘要改进 ──────────────────────────────────────────────────────────

class TestSummarizeContent:
    def test_sentence_boundary_truncation(self):
        """应在句子边界截断。"""
        from claude_token_saver.conversation_diff import _summarize_content
        text = "First sentence. Second sentence. Third sentence."
        result = _summarize_content(text, max_length=20)
        # 应该在句号后截断
        assert result.endswith("...")

    def test_short_content_unchanged(self):
        """短内容不应被截断。"""
        from claude_token_saver.conversation_diff import _summarize_content
        assert _summarize_content("hello", max_length=100) == "hello"

    def test_cjk_sentence_boundary(self):
        """中文句子边界也应被识别。"""
        from claude_token_saver.conversation_diff import _summarize_content
        text = "第一句话。第二句话。第三句话。"
        result = _summarize_content(text, max_length=8)
        assert "。" in result


# ── 目录索引紧凑模式 ──────────────────────────────────────────────────────

class TestCompactIndex:
    def test_compact_format_minimal(self):
        """紧凑索引应最简化。"""
        from claude_token_saver.progressive import DirectoryIndex, FileIndexEntry, format_index_markdown
        entries = [
            FileIndexEntry(path="/tmp/a.py", size_bytes=100, estimated_tokens=25, ext=".py", relative_path="a.py"),
            FileIndexEntry(path="/tmp/b.py", size_bytes=200, estimated_tokens=50, ext=".py", relative_path="b.py"),
        ]
        idx = DirectoryIndex(root="/tmp", total_files=2, total_tokens=75, files=entries, by_directory={}, largest_files=[])
        result = format_index_markdown(idx, compact=True)
        assert "## " not in result  # 无目录分组标题
        assert "文件类型分布" not in result  # 无统计区块
        assert "a.py" in result
        assert "b.py" in result

    def test_compact_shorter_than_full(self):
        """紧凑索引应比完整索引短。"""
        from claude_token_saver.progressive import DirectoryIndex, FileIndexEntry, format_index_markdown
        # 多个目录使完整格式产生更多分组标题
        entries = []
        by_dir = {}
        for i in range(10):
            d = f"dir{i % 3}"
            rel = f"{d}/f{i}.py"
            e = FileIndexEntry(path=f"/tmp/{rel}", size_bytes=100, estimated_tokens=25, ext=".py", relative_path=rel)
            entries.append(e)
            by_dir.setdefault(d, []).append(e)

        idx = DirectoryIndex(root="/tmp", total_files=10, total_tokens=250, files=entries, by_directory=by_dir, largest_files=[])
        full = format_index_markdown(idx, compact=False)
        compact = format_index_markdown(idx, compact=True)
        assert len(compact) < len(full)


# ── 重要注释保留 ──────────────────────────────────────────────────────────

class TestImportantCommentPreservation:
    def test_strip_removes_normal_comments(self):
        """普通注释应被去除。"""
        from claude_token_saver.prep import strip_comments
        result = strip_comments("x = 1  # regular comment\ny = 2\n", ".py")
        assert "regular comment" not in result

    def test_preserves_todo(self):
        """TODO 注释应被保留。"""
        from claude_token_saver.prep import strip_comments
        result = strip_comments("x = 1  # TODO: fix this\ny = 2\n", ".py")
        assert "TODO" in result

    def test_preserves_fixme(self):
        """FIXME 注释应被保留。"""
        from claude_token_saver.prep import strip_comments
        result = strip_comments("x = 1  # FIXME: bug here\ny = 2\n", ".py")
        assert "FIXME" in result

    def test_preserves_multiple_keywords(self):
        """多个重要注释都应被保留。"""
        from claude_token_saver.prep import strip_comments
        content = "x = 1  # TODO: a\n# NOTE: b\n# HACK: c\ny = 2\n"
        result = strip_comments(content, ".py")
        assert "TODO" in result
        assert "NOTE" in result
        assert "HACK" in result

    def test_dedup_important_comments(self):
        """重复的重要注释应去重。"""
        from claude_token_saver.prep import strip_comments
        content = "x = 1  # TODO: fix\n# TODO: fix\ny = 2\n"
        result = strip_comments(content, ".py")
        # TODO 应只出现一次
        assert result.count("TODO") == 1


# ── 连续行去重 ────────────────────────────────────────────────────────────

class TestDedupConsecutiveLines:
    def test_removes_duplicate_lines(self):
        """连续重复行应被去除。"""
        from claude_token_saver.prep import _deduplicate_consecutive_lines
        lines = ["import os", "import os", "import sys"]
        result = _deduplicate_consecutive_lines(lines)
        assert result == ["import os", "import sys"]

    def test_keeps_non_consecutive_duplicates(self):
        """非连续重复行应被保留。"""
        from claude_token_saver.prep import _deduplicate_consecutive_lines
        lines = ["import os", "import sys", "import os"]
        result = _deduplicate_consecutive_lines(lines)
        assert result == ["import os", "import sys", "import os"]

    def test_ignores_blank_lines(self):
        """空行不应参与去重。"""
        from claude_token_saver.prep import _deduplicate_consecutive_lines
        lines = ["import os", "", "", "import sys"]
        result = _deduplicate_consecutive_lines(lines)
        assert result == ["import os", "", "", "import sys"]

    def test_ignores_comments(self):
        """注释行不应参与去重。"""
        from claude_token_saver.prep import _deduplicate_consecutive_lines
        lines = ["# comment", "# comment", "x = 1"]
        result = _deduplicate_consecutive_lines(lines)
        assert result == ["# comment", "# comment", "x = 1"]

    def test_empty_input(self):
        """空列表应返回空列表。"""
        from claude_token_saver.prep import _deduplicate_consecutive_lines
        assert _deduplicate_consecutive_lines([]) == []


# ── Unicode 规范化 ────────────────────────────────────────────────────────

class TestUnicodeNormalize:
    def test_removes_bom(self):
        """应去除 UTF-8 BOM。"""
        from claude_token_saver.prep import unicode_normalize
        result = unicode_normalize("﻿hello")
        assert not result.startswith("﻿")
        assert "hello" in result

    def test_nfc_normalization(self):
        """应进行 NFC 规范化。"""
        from claude_token_saver.prep import unicode_normalize
        # é 的分解形式 (e + ́) 应被规范化为 NFC (é)
        decomposed = "é"  # e + combining acute accent
        result = unicode_normalize(decomposed)
        assert result == "é"  # é

    def test_removes_zero_width_chars(self):
        """应去除零宽字符。"""
        from claude_token_saver.prep import unicode_normalize
        # 零宽空格 (U+200B) + 零宽非连接符 (U+200C)
        result = unicode_normalize("hello​‌world")
        assert "​" not in result
        assert "‌" not in result

    def test_normal_content_unchanged(self):
        """正常内容不应被修改。"""
        from claude_token_saver.prep import unicode_normalize
        assert unicode_normalize("hello world") == "hello world"

    def test_normalize_whitespace_applies_unicode(self):
        """normalize_whitespace 应应用 Unicode 规范化。"""
        from claude_token_saver.prep import normalize_whitespace
        # BOM 应被去除
        result = normalize_whitespace("﻿x = 1\n", ".py")
        assert "﻿" not in result


# ── 完整流程集成 ──────────────────────────────────────────────────────────

class TestIntegrationOptimizations:
    def test_pipeline_with_important_comments(self):
        """管线应保留重要注释。"""
        from claude_token_saver.compression_pipeline import CompressionPipeline
        content = "# TODO: fix this\ndef foo():\n    pass\n"
        p = CompressionPipeline(ext=".py", detail_level="stripped")
        result, _ = p.run(content)
        assert "TODO" in result

    def test_process_files_unicode_normalize(self):
        """处理文件时应规范化 Unicode。"""
        from claude_token_saver.prep import process_files
        import tempfile, os
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False, encoding='utf-8') as f:
            f.write("x = 1  # TODO: fix\n")
            fname = f.name
        try:
            result = process_files([fname], dedup=False, token_cache_enabled=False)
            assert len(result["files"]) == 1
        finally:
            os.unlink(fname)

    def test_dedup_consecutive_in_whitespace(self):
        """normalize_whitespace 应去除连续重复行。"""
        from claude_token_saver.prep import normalize_whitespace
        content = "import os\nimport os\nimport sys\nx = 1\n"
        result = normalize_whitespace(content, ".py")
        assert result.count("import os") == 1
        assert "import sys" in result


# ── 注释密度检测 ──────────────────────────────────────────────────────────

class TestCommentDensity:
    def test_empty_content(self):
        """空内容密度应为 0。"""
        from claude_token_saver.prep import _comment_density
        assert _comment_density("", ".py") == 0.0

    def test_no_comments(self):
        """无注释内容密度应为 0。"""
        from claude_token_saver.prep import _comment_density
        assert _comment_density("x = 1\ny = 2\n", ".py") == 0.0

    def test_all_comments(self):
        """全是注释的内容密度应为 1.0。"""
        from claude_token_saver.prep import _comment_density
        assert _comment_density("# comment 1\n# comment 2", ".py") == 1.0

    def test_mixed_content(self):
        """混合内容应有中间密度。"""
        from claude_token_saver.prep import _comment_density
        content = "# comment\nx = 1\ny = 2\n"
        assert 0.0 < _comment_density(content, ".py") < 1.0

    def test_docstring_counted_as_comment(self):
        """docstring 应被计入注释。"""
        from claude_token_saver.prep import _comment_density
        content = '"""docstring"""\nx = 1\n'
        density = _comment_density(content, ".py")
        assert density > 0


# ── 已压缩内容检测 ────────────────────────────────────────────────────────

class TestAlreadyMinified:
    def test_detects_minified(self):
        """应识别已压缩内容。"""
        from claude_token_saver.prep import _is_already_minified
        assert _is_already_minified("x=1\ny=2\n", ".py")

    def test_detects_not_minified(self):
        """应识别未压缩内容。"""
        from claude_token_saver.prep import _is_already_minified
        assert not _is_already_minified("x = 1  \ny = 2  \n", ".py")

    def test_empty_content(self):
        """空内容应视为已压缩。"""
        from claude_token_saver.prep import _is_already_minified
        assert _is_already_minified("", ".py")

    def test_extra_blank_lines(self):
        """有多余空行的内容应视为未压缩。"""
        from claude_token_saver.prep import _is_already_minified
        assert not _is_already_minified("x = 1\n\n\n\ny = 2\n", ".py")


# ── 超紧凑输出格式 ────────────────────────────────────────────────────────

class TestSuperCompactFormat:
    def test_super_compact_uses_separators(self):
        """super-compact 应使用 --- 分隔符。"""
        from claude_token_saver.prep import format_processed_output
        result = format_processed_output({
            "files": [{"path": "/tmp/test.py", "content": "x = 1", "tokens_after": 2, "tokens_before": 10, "savings": 8}],
        }, format="super-compact")
        assert "---" in result

    def test_super_compact_no_markdown_headers(self):
        """super-compact 不应包含 markdown 标题。"""
        from claude_token_saver.prep import format_processed_output
        result = format_processed_output({
            "files": [{"path": "/tmp/test.py", "content": "x = 1", "tokens_after": 2, "tokens_before": 10, "savings": 8}],
        }, format="super-compact")
        assert "##" not in result
        assert "```" not in result

    def test_super_compact_shorter_than_raw(self):
        """super-compact 应比 raw 更短。"""
        from claude_token_saver.prep import format_processed_output
        data = {
            "files": [
                {"path": f"/tmp/f{i}.py", "content": "x = 1\n", "tokens_after": 2, "tokens_before": 10, "savings": 8}
                for i in range(5)
            ],
        }
        super_compact = format_processed_output(data, format="super-compact")
        raw = format_processed_output(data, format="raw")
        assert len(super_compact) <= len(raw)


# ── 冗余 pass 去除 ────────────────────────────────────────────────────────

class TestRemoveRedundantPass:
    def test_removes_standalone_pass(self):
        """单独的 pass 应被去除。"""
        from claude_token_saver.prep import remove_redundant_pass
        result = remove_redundant_pass("class Foo:\n    pass\n")
        assert "pass" not in result

    def test_keeps_pass_with_code(self):
        """与其他代码在一起的 pass 应被保留。"""
        from claude_token_saver.prep import remove_redundant_pass
        result = remove_redundant_pass("def foo():\n    x = 1\n    pass\n    return x\n")
        assert "pass" in result

    def test_keeps_pass_in_try_except(self):
        """try/except 中的 pass 应被保留。"""
        from claude_token_saver.prep import remove_redundant_pass
        result = remove_redundant_pass("try:\n    x = 1\nexcept:\n    pass\n")
        assert "pass" in result

    def test_no_pass_unchanged(self):
        """没有 pass 的内容不应被修改。"""
        from claude_token_saver.prep import remove_redundant_pass
        assert remove_redundant_pass("x = 1\ny = 2\n") == "x = 1\ny = 2\n"

    def test_empty_function_pass(self):
        """空函数的 pass 应被去除。"""
        from claude_token_saver.prep import remove_redundant_pass
        result = remove_redundant_pass("def foo():\n    pass\n")
        assert "pass" not in result
        assert "def foo():" in result


# ── 跨会话缓存持久化 ──────────────────────────────────────────────────────

class TestCachePersistence:
    def test_save_and_load_cache(self, tmp_path):
        """应能保存和加载缓存。"""
        from claude_token_saver.prep import save_content_cache, load_content_cache, _content_cache
        cache_file = tmp_path / "cache.json"
        _content_cache[("/tmp/test.py", 1234, 100)] = ("abc123", 50)
        save_content_cache(cache_file)
        # 清空缓存
        _content_cache.clear()
        assert len(_content_cache) == 0
        # 加载
        loaded = load_content_cache(cache_file)
        assert loaded == 1
        assert ("/tmp/test.py", 1234, 100) in _content_cache

    def test_load_nonexistent_cache(self, tmp_path):
        """加载不存在的缓存应返回 0。"""
        from claude_token_saver.prep import load_content_cache
        cache_file = tmp_path / "nonexistent.json"
        assert load_content_cache(cache_file) == 0


# ── __future__ import 去除 ────────────────────────────────────────────────

class TestRemoveFutureImports:
    def test_removes_future_import(self):
        """应去除 __future__ import 行。"""
        from claude_token_saver.prep import normalize_whitespace
        result = normalize_whitespace("from __future__ import annotations\nx = 1\n", ".py")
        assert "__future__" not in result
        assert "x = 1" in result

    def test_removes_following_blank_line(self):
        """应去除 __future__ import 后的空行。"""
        from claude_token_saver.prep import normalize_whitespace
        result = normalize_whitespace("from __future__ import annotations\n\nx = 1\n", ".py")
        assert "__future__" not in result
        assert result.startswith("x = 1")

    def test_keeps_other_imports(self):
        """其他 import 应保留。"""
        from claude_token_saver.prep import normalize_whitespace
        result = normalize_whitespace("import os\nfrom __future__ import annotations\nimport sys\n", ".py")
        assert "import os" in result
        assert "import sys" in result
        assert "__future__" not in result


# ── Markdown YAML frontmatter 去除 ────────────────────────────────────────

class TestStripMarkdownFrontmatter:
    def test_removes_frontmatter(self):
        """应去除 YAML frontmatter。"""
        from claude_token_saver.prep import _strip_markdown_frontmatter
        content = "---\ntitle: Test\ndescription: desc\n---\n# Hello\n"
        result = _strip_markdown_frontmatter(content)
        assert "title: Test" not in result
        assert "# Hello" in result

    def test_no_frontmatter_unchanged(self):
        """没有 frontmatter 的内容不应被修改。"""
        from claude_token_saver.prep import _strip_markdown_frontmatter
        content = "# Hello World\n"
        result = _strip_markdown_frontmatter(content)
        assert result == content

    def test_only_frontmatter(self):
        """只有 frontmatter 的内容应返回空。"""
        from claude_token_saver.prep import _strip_markdown_frontmatter
        content = "---\ntitle: Test\n---\n"
        result = _strip_markdown_frontmatter(content)
        assert "title: Test" not in result


# ── Python 多行 import 压缩 ──────────────────────────────────────────────

class TestCollapsePythonImports:
    def test_collapses_multiline_import(self):
        """应将多行 import 压缩为单行。"""
        from claude_token_saver.prep import _collapse_python_imports
        content = "from mypackage import (\n    ClassA,\n    ClassB,\n    ClassC,\n)\n"
        result = _collapse_python_imports(content)
        assert "ClassA, ClassB, ClassC" in result
        assert "(\n" not in result

    def test_single_line_import_unchanged(self):
        """单行 import 不应被修改。"""
        from claude_token_saver.prep import _collapse_python_imports
        content = "from mypackage import ClassA\n"
        result = _collapse_python_imports(content)
        assert result == content

    def test_import_with_aliases(self):
        """带 as 别名的 import 应正确处理。"""
        from claude_token_saver.prep import _collapse_python_imports
        content = "from mypackage import (\n    ClassA as A,\n    ClassB as B,\n)\n"
        result = _collapse_python_imports(content)
        assert "ClassA as A" in result
        assert "ClassB as B" in result

    def test_invalid_syntax_unchanged(self):
        """语法错误的内容不应被修改。"""
        from claude_token_saver.prep import _collapse_python_imports
        content = "from mypackage import (\n    ClassA,\n"  # missing closing paren
        result = _collapse_python_imports(content)
        assert result == content


# ── 类型注解压缩 ──────────────────────────────────────────────────────────

class TestCompressSkeletonTypes:
    def test_compresses_dict_type(self):
        """应压缩 typing.Dict 为 dict。"""
        from claude_token_saver.compressor import _compress_skeleton_types
        result = _compress_skeleton_types("def foo() -> typing.Dict[str, int]: pass")
        assert "dict[str, int]" in result
        assert "typing.Dict" not in result

    def test_compresses_list_type(self):
        """应压缩 typing.List 为 list。"""
        from claude_token_saver.compressor import _compress_skeleton_types
        result = _compress_skeleton_types("def foo() -> typing.List[int]: pass")
        assert "list[int]" in result
        assert "typing.List" not in result

    def test_strips_none_return(self):
        """应去除 -> None 返回注解。"""
        from claude_token_saver.compressor import _compress_skeleton_types
        result = _compress_skeleton_types("def foo() -> None: pass")
        assert "-> None" not in result


# ── 缓存清理整合 ──────────────────────────────────────────────────────────

class TestCacheClearingIntegration:
    def test_clear_caches_clears_compressor(self):
        """_clear_caches 应同步清空压缩器缓存。"""
        from claude_token_saver.prep import _clear_caches
        from claude_token_saver.compressor import _SKELETON_CACHE
        _SKELETON_CACHE["test"] = "value"
        _clear_caches()
        assert len(_SKELETON_CACHE) == 0


# ── 综合优化效果测试 ──────────────────────────────────────────────────────

class TestCombinedOptimizations:
    def test_python_file_full_pipeline(self, tmp_path):
        """Python 文件应受益于所有新优化。"""
        from claude_token_saver.prep import process_files
        f = tmp_path / "test.py"
        f.write_text("""from __future__ import annotations

from typing import (
    Dict,
    List,
    Optional,
)

def foo(x: Dict[str, List[int]]) -> Optional[str]:
    \"\"\"Docstring here.\"\"\"
    return None

class Bar:
    pass
""")
        _clear_caches()
        result = process_files([str(f)], detail_level="full", dedup=False)
        assert len(result["files"]) == 1
        content = result["files"][0]["content"]
        # __future__ import 应被去除
        assert "__future__" not in content
        # 多行 typing import 应被压缩为单行
        assert "from typing import Dict, List, Optional" in content
        # 不应有多行格式
        assert "(\n" not in content
        # 应有一定程度的 token 节省
        assert result["files"][0]["savings"] > 0

    def test_markdown_frontmatter_removed(self, tmp_path):
        """Markdown 文件的 frontmatter 应被去除。"""
        from claude_token_saver.prep import process_files
        f = tmp_path / "README.md"
        f.write_text("""---
title: "My Project"
description: "A test project"
---

# Hello World

This is the content.
""")
        result = process_files([str(f)], detail_level="full", dedup=False)
        assert len(result["files"]) == 1
        content = result["files"][0]["content"]
        assert "title:" not in content
        assert "# Hello World" in content


# ── 空类体压缩 ────────────────────────────────────────────────────────────

class TestCompressEmptyClassBodies:
    def test_compresses_empty_class(self):
        """只有 pass 的类应被压缩为单行。"""
        from claude_token_saver.prep import _compress_empty_class_bodies
        result = _compress_empty_class_bodies("class Foo:\n    pass\n")
        assert "pass" not in result
        assert "class Foo:" in result

    def test_keeps_class_with_body(self):
        """有实际内容的类不应被压缩。"""
        from claude_token_saver.prep import _compress_empty_class_bodies
        content = "class Foo:\n    x = 1\n"
        result = _compress_empty_class_bodies(content)
        assert result == content

    def test_multiple_empty_classes(self):
        """多个空类都应被压缩。"""
        from claude_token_saver.prep import _compress_empty_class_bodies
        content = "class A:\n    pass\nclass B:\n    pass\n"
        result = _compress_empty_class_bodies(content)
        assert result.count("pass") == 0
        assert "class A:" in result
        assert "class B:" in result


# ── JS/TS 类型注解去除 ────────────────────────────────────────────────────

class TestJsTsSkeleton:
    def test_keeps_imports(self):
        """应保留 import 语句。"""
        from claude_token_saver.compressor import _extract_js_ts_skeleton
        result = _extract_js_ts_skeleton("import { foo } from 'bar';\nconst x = 1;\n")
        assert "import { foo } from 'bar';" in result

    def test_comments_out_non_sig_lines(self):
        """非签名行应被压缩为省略标记。"""
        from claude_token_saver.compressor import _extract_js_ts_skeleton
        result = _extract_js_ts_skeleton("const x = 1;\n")
        assert "// ..." in result
        assert "const x = 1" not in result

    def test_keeps_class_signature(self):
        """应保留类签名并省略体。"""
        from claude_token_saver.compressor import _extract_js_ts_skeleton
        result = _extract_js_ts_skeleton("class Foo {\n    x = 1;\n}\n")
        assert "class Foo {" in result
        assert "/* ... */" in result

    def test_keeps_function_signature(self):
        """应保留函数签名并省略体。"""
        from claude_token_saver.compressor import _extract_js_ts_skeleton
        result = _extract_js_ts_skeleton("function foo() {\n    return 1;\n}\n")
        assert "function foo() {" in result
        assert "/* ... */" in result

    def test_keeps_interface(self):
        """应保留 interface 签名。"""
        from claude_token_saver.compressor import _extract_js_ts_skeleton
        result = _extract_js_ts_skeleton("interface User {\n    id: number;\n}\n")
        assert "interface User {" in result


# ── 智能 conftest 去重 ────────────────────────────────────────────────────

class TestConftestDedup:
    def test_dedups_identical_conftest(self, tmp_path):
        """相同的 conftest.py 在子目录中应被去重。"""
        from claude_token_saver.common_dedup import dedup_conftest_always
        # 创建顶层 conftest
        top = tmp_path / "conftest.py"
        top.write_text("import pytest\n@pytest.fixture\ndef foo(): pass\n")
        # 创建子目录中相同的 conftest
        sub = tmp_path / "tests" / "unit"
        sub.mkdir(parents=True)
        (sub / "conftest.py").write_text("import pytest\n@pytest.fixture\ndef foo(): pass\n")
        files = [str(top), str(sub / "conftest.py")]
        remaining, removed = dedup_conftest_always(files)
        assert len(remaining) == 1
        assert str(top) in remaining

    def test_keeps_different_conftest(self, tmp_path):
        """不同的 conftest.py 应都被保留。"""
        from claude_token_saver.common_dedup import dedup_conftest_always
        top = tmp_path / "conftest.py"
        top.write_text("# top\n")
        sub = tmp_path / "tests" / "unit"
        sub.mkdir(parents=True)
        (sub / "conftest.py").write_text("# unit\n")
        files = [str(top), str(sub / "conftest.py")]
        remaining, removed = dedup_conftest_always(files)
        assert len(remaining) == 2


# ── 骨架 docstring 摘要 ────────────────────────────────────────────────────

class TestSkeletonDocstringSummary:
    def test_injects_docstring_summary(self):
        """骨架应包含 docstring 首行摘要。"""
        from claude_token_saver.compressor import extract_skeleton
        content = '''class Foo:
    """This is a test class."""
    x = 1

def bar():
    """Calculate the sum."""
    return 1 + 1
'''
        result = extract_skeleton(content, ".py")
        assert "test class" in result or "Calculate" in result

    def test_no_docstring_unchanged(self):
        """没有 docstring 的骨架不应改变。"""
        from claude_token_saver.compressor import extract_skeleton
        content = "class Foo:\n    x = 1\ndef bar():\n    return 1\n"
        result = extract_skeleton(content, ".py")
        assert "class Foo:" in result
        assert "def bar():" in result


# ── 综合 benchmark ────────────────────────────────────────────────────────

class TestOptimizationBenchmark:
    def test_python_oop_file_savings(self, tmp_path):
        """OOP 代码应通过空类压缩和骨架提取显著节省 token。"""
        from claude_token_saver.prep import process_files
        f = tmp_path / "models.py"
        f.write_text("""
from typing import Dict, List, Optional
from abc import ABC, abstractmethod

class BaseModel(ABC):
    \"\"\"Base model class.\"\"\"
    pass

class User(BaseModel):
    \"\"\"User model.\"\"\"
    pass

class Order(BaseModel):
    \"\"\"Order model.\"\"\"
    pass

class Service:
    \"\"\"Service class.\"\"\"
    def __init__(self):
        pass

    def get_user(self, uid: int) -> Optional[User]:
        pass

    def get_orders(self, uid: int) -> List[Order]:
        pass

    def create_order(self, order: Dict[str, any]) -> Order:
        \"\"\"Create a new order.\"\"\"
        pass

    def delete_order(self, oid: int) -> bool:
        \"\"\"Delete an order.\"\"\"
        pass

    def update_order(self, oid: int, data: Dict[str, any]) -> Order:
        \"\"\"Update an order.\"\"\"
        pass
""")
        _clear_caches()
        result = process_files([str(f)], detail_level="skeleton", dedup=False)
        assert len(result["files"]) == 1
        assert result["files"][0]["savings"] > 10  # 至少节省 10 tokens


# ── 扩展空体压缩（函数/if/for/while/try） ─────────────────────────────────

class TestCompressEmptyBodies:
    def test_compresses_empty_function(self):
        """只有 pass 的函数应被压缩。"""
        from claude_token_saver.prep import _compress_empty_bodies
        result = _compress_empty_bodies("def foo():\n    pass\n")
        assert "pass" not in result
        assert "def foo():" in result

    def test_compresses_empty_if_block(self):
        """只有 pass 的 if 块应被压缩。"""
        from claude_token_saver.prep import _compress_empty_bodies
        result = _compress_empty_bodies("if True:\n    pass\n")
        assert "pass" not in result
        assert "if True:" in result

    def test_compresses_empty_for_block(self):
        """只有 pass 的 for 块应被压缩。"""
        from claude_token_saver.prep import _compress_empty_bodies
        result = _compress_empty_bodies("for x in range(10):\n    pass\n")
        assert "pass" not in result

    def test_compresses_empty_while_block(self):
        """只有 pass 的 while 块应被压缩。"""
        from claude_token_saver.prep import _compress_empty_bodies
        result = _compress_empty_bodies("while True:\n    pass\n")
        assert "pass" not in result

    def test_keeps_function_with_body(self):
        """有实际内容的函数不应被压缩。"""
        from claude_token_saver.prep import _compress_empty_bodies
        content = "def foo():\n    x = 1\n    return x\n"
        result = _compress_empty_bodies(content)
        assert result == content

    def test_compresses_empty_return_none(self):
        """只有 return None 的函数应被压缩。"""
        from claude_token_saver.prep import _compress_empty_bodies
        result = _compress_empty_bodies("def foo():\n    return None\n")
        assert "return None" not in result


# ── 去除 __main__ 块 ──────────────────────────────────────────────────────

class TestRemoveMainGuard:
    def test_removes_main_guard(self):
        """应去除 if __name__ == '__main__' 块。"""
        from claude_token_saver.prep import _remove_main_guard
        content = '''def helper():
    pass

if __name__ == "__main__":
    helper()
'''
        result = _remove_main_guard(content)
        assert '__name__' not in result
        assert "def helper():" in result

    def test_no_main_guard_unchanged(self):
        """没有 main guard 的内容不应被修改。"""
        from claude_token_saver.prep import _remove_main_guard
        content = "x = 1\ny = 2\n"
        result = _remove_main_guard(content)
        assert result == content

    def test_removes_preceding_blank_line(self):
        """应去除 main guard 前的空行。"""
        from claude_token_saver.prep import _remove_main_guard
        content = "x = 1\n\nif __name__ == '__main__':\n    pass\n"
        result = _remove_main_guard(content)
        assert "__name__" not in result


# ── 不可达代码去除 ────────────────────────────────────────────────────────

class TestRemoveDeadCode:
    def test_removes_after_return(self):
        """return 之后的代码应被去除。"""
        from claude_token_saver.prep import _remove_dead_code_after_control_flow
        content = '''def foo(x):
    if not x:
        return None
        logger.warning("bad")
        raise ValueError("bad")
    return x
'''
        result = _remove_dead_code_after_control_flow(content)
        assert "logger.warning" not in result
        assert "raise ValueError" not in result
        assert "return None" in result

    def test_removes_after_raise(self):
        """raise 之后的代码应被去除。"""
        from claude_token_saver.prep import _remove_dead_code_after_control_flow
        content = '''def foo():
    raise RuntimeError("fail")
    x = 1
    return x
'''
        result = _remove_dead_code_after_control_flow(content)
        assert "x = 1" not in result
        assert "return x" not in result

    def test_keeps_reachable_code(self):
        """可达代码应被保留。"""
        from claude_token_saver.prep import _remove_dead_code_after_control_flow
        content = "x = 1\nreturn x\ny = 2\n"
        result = _remove_dead_code_after_control_flow(content)
        assert "x = 1" in result
        assert "y = 2" in result


# ── Assert 语句压缩 ───────────────────────────────────────────────────────

class TestCompressAssertStatements:
    def test_compresses_is_not_none(self):
        """assert x is not None → assert x"""
        from claude_token_saver.prep import _compress_assert_statements
        result = _compress_assert_statements("assert x is not None\n")
        assert result == "assert x\n"

    def test_compresses_is_none(self):
        """assert x is None → assert not x"""
        from claude_token_saver.prep import _compress_assert_statements
        result = _compress_assert_statements("assert x is None\n")
        assert result == "assert not x\n"

    def test_compresses_len_gt_zero(self):
        """assert len(x) > 0 → assert x"""
        from claude_token_saver.prep import _compress_assert_statements
        result = _compress_assert_statements("assert len(x) > 0\n")
        assert result == "assert x\n"

    def test_no_assert_unchanged(self):
        """没有 assert 的内容不应被修改。"""
        from claude_token_saver.prep import _compress_assert_statements
        content = "x = 1\ny = 2\n"
        result = _compress_assert_statements(content)
        assert result == content


# ── raise from None 去除 ─────────────────────────────────────────────────────

class TestRemoveRaiseFromNone:
    def test_removes_from_none(self):
        """raise ... from None 应去除 from None。"""
        from claude_token_saver.prep import _remove_raise_from_none
        result = _remove_raise_from_none("raise ValueError('bad') from None\n")
        assert result == "raise ValueError('bad')\n"

    def test_keeps_raise_with_cause(self):
        """raise ... from e 应保留。"""
        from claude_token_saver.prep import _remove_raise_from_none
        result = _remove_raise_from_none("raise ValueError('bad') from e\n")
        assert "from e" in result

    def test_no_raise_unchanged(self):
        """没有 raise from None 的内容不应被修改。"""
        from claude_token_saver.prep import _remove_raise_from_none
        content = "x = 1\ny = 2\n"
        result = _remove_raise_from_none(content)
        assert result == content


# ── self 赋值压缩 ────────────────────────────────────────────────────────────

class TestCompressSelfAssignments:
    def test_compresses_three_self_assignments(self):
        """3+ 个连续的 self.x = x 应压缩为 dict 形式。"""
        from claude_token_saver.prep import _compress_self_assignments
        content = """class Foo:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
"""
        result = _compress_self_assignments(content)
        assert "self.__dict__.update" in result
        # 独立的赋值行应被移除（注释行中仍含 self.x = x 文本）
        assert "self.y = y\n" not in result
        assert "self.z = z\n" not in result

    def test_keeps_less_than_three(self):
        """少于 3 个的 self 赋值不应压缩。"""
        from claude_token_saver.prep import _compress_self_assignments
        content = """class Foo:
    def __init__(self, x):
        self.x = x
        self.y = y
"""
        result = _compress_self_assignments(content)
        assert "self.x = x" in result
        assert "self.__dict__.update" not in result

    def test_only_init_method(self):
        """仅在 __init__ 中应用压缩。"""
        from claude_token_saver.prep import _compress_self_assignments
        content = """class Foo:
    def setup(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
"""
        result = _compress_self_assignments(content)
        assert "self.__dict__.update" not in result


# ── 单表达式函数内联 ─────────────────────────────────────────────────────────

class TestInlineSingleExprFunctions:
    def test_inlines_return_function(self):
        """只有 return 的函数应压缩为单行。"""
        from claude_token_saver.prep import _inline_single_expr_functions
        content = """def get_name(self):
    return self.name
"""
        result = _inline_single_expr_functions(content)
        assert "def get_name(self): return self.name" in result

    def test_keeps_multi_stmt_function(self):
        """有多条语句的函数不应压缩。"""
        from claude_token_saver.prep import _inline_single_expr_functions
        content = """def foo(x):
    if x:
        return 1
    return 0
"""
        result = _inline_single_expr_functions(content)
        assert "def foo(x):" in result
        assert "return 1" in result

    def test_keeps_function_with_docstring_and_code(self):
        """有 docstring 和代码的函数不应压缩。"""
        from claude_token_saver.prep import _inline_single_expr_functions
        content = '''def foo(x):
    """Get value."""
    return x
'''
        result = _inline_single_expr_functions(content)
        assert "def foo(x):" in result


# ── 冗余导入别名过滤 ─────────────────────────────────────────────────────────

class TestCollapsePythonImportsAlias:
    def test_filters_redundant_alias(self):
        """冗余别名（y as y）应在多行 import 压缩时被过滤。"""
        from claude_token_saver.prep import _collapse_python_imports
        content = "from typing import (\n    Dict as Dict,\n    List as List,\n    Optional as Optional,\n)\n"
        result = _collapse_python_imports(content)
        assert "as Dict" not in result
        assert "as List" not in result
        assert "from typing import Dict, List, Optional" in result

    def test_keeps_meaningful_alias(self):
        """有意义的别名应保留。"""
        from claude_token_saver.prep import _collapse_python_imports
        content = "from typing import (\n    Dict as D,\n    List as L,\n)\n"
        result = _collapse_python_imports(content)
        assert "as D" in result
        assert "as L" in result


# ── .pyi 去重 ────────────────────────────────────────────────────────────────

class TestPyiDedup:
    def test_pyi_with_corresponding_py(self, tmp_path):
        """有对应 .py 的 .pyi 应被标记跳过。"""
        from claude_token_saver.prep import process_files
        py_file = tmp_path / "model.py"
        pyi_file = tmp_path / "model.pyi"
        py_file.write_text("class Model:\n    pass\n")
        pyi_file.write_text("class Model:\n    ...\n")
        result = process_files(
            [str(py_file), str(pyi_file)],
            detail_level="full",
        )
        paths = [r["path"] for r in result["files"]]
        # .pyi 应被跳过（不读取内容）
        pyi_in_result = any(str(pyi_file) == p for p in paths)
        assert not pyi_in_result

    def test_pyi_without_py(self, tmp_path):
        """没有对应 .py 的 .pyi 应正常处理。"""
        from claude_token_saver.prep import process_files
        pyi_file = tmp_path / "standalone.pyi"
        pyi_file.write_text("class Standalone:\n    ...\n")
        result = process_files(
            [str(pyi_file)],
            detail_level="full",
        )
        paths = [r["path"] for r in result["files"]]
        assert any(str(pyi_file) == p for p in paths)


class TestRemoveTypeAnnotations:
    """测试 _remove_type_annotations 函数。"""

    def _call(self, content: str) -> str:
        from claude_token_saver.prep import _remove_type_annotations
        return _remove_type_annotations(content)

    # ── 函数签名（返回类型 + 参数类型均被去除）──

    def test_return_type_simple(self):
        result = self._call("def foo() -> bool:\n    return True\n")
        assert result == "def foo():\n    return True\n"

    def test_param_type_simple(self):
        result = self._call("def foo(x: int):\n    return True\n")
        assert result == "def foo(x):\n    return True\n"

    def test_param_and_return_types(self):
        result = self._call("def foo(x: int, y: str):\n    return True\n")
        assert result == "def foo(x, y):\n    return True\n"

    def test_complex_param_type(self):
        result = self._call("def foo(x: Dict[str, int]):\n    pass\n")
        assert result == "def foo(x):\n    pass\n"

    def test_no_annotations(self):
        result = self._call("def foo(x, y):\n    return True\n")
        assert result == "def foo(x, y):\n    return True\n"

    # ── 变量注解 ──

    def test_annotated_assignment_with_value(self):
        result = self._call("x: int = 5\n")
        assert result == "x = 5\n"

    def test_annotated_assignment_no_value(self):
        result = self._call("x: int\n")
        assert result == "\n"

    def test_class_with_annotations(self):
        result = self._call('class Foo:\n    x: int = 5\n    y: str = "hello"\n')
        assert result == 'class Foo:\n    x = 5\n    y = "hello"\n'

    # ── 异步函数 ──

    def test_async_return_type(self):
        result = self._call("async def foo():\n    return 1\n")
        assert result == "async def foo():\n    return 1\n"

    # ── 边缘情况 ──

    def test_syntax_error_returns_original(self):
        result = self._call("def foo(:  # broken\n")
        assert result == "def foo(:  # broken\n"

    def test_empty_content(self):
        result = self._call("")
        assert result == ""

    def test_mixed_annotated_unannotated(self):
        result = self._call("def foo(x, y: int):\n    pass\n")
        assert result == "def foo(x, y):\n    pass\n"

    def test_multiple_annotated_params(self):
        result = self._call("def foo(a: int, b: str, c: float):\n    pass\n")
        assert result == "def foo(a, b, c):\n    pass\n"

    def test_varargs_kwargs(self):
        result = self._call("def foo(*args, **kwargs):\n    pass\n")
        assert result == "def foo(*args, **kwargs):\n    pass\n"

    def test_kwonly_separator(self):
        result = self._call("def foo(*, x: int):\n    pass\n")
        assert result == "def foo(*, x):\n    pass\n"

    def test_self_param(self):
        result = self._call("def foo(self, x: int):\n    pass\n")
        assert result == "def foo(self, x):\n    pass\n"

    def test_class_method_preserves_indent(self):
        result = self._call('class Foo:\n    def bar(self, x: int):\n        return ""\n')
        assert result == 'class Foo:\n    def bar(self, x):\n        return ""\n'

    def test_decorated_function(self):
        result = self._call("@decorator\ndef foo(x: int):\n    return True\n")
        assert result == "@decorator\ndef foo(x):\n    return True\n"


class TestRemoveUnusedTypingImports:
    """测试 _remove_unused_typing_imports 函数。"""

    def test_removes_all_unused(self):
        from claude_token_saver.prep import _remove_unused_typing_imports
        inp = "from typing import Dict, List, Optional\ndef foo(x):\n    return x\n"
        result = _remove_unused_typing_imports(inp)
        assert "Dict" not in result
        assert "List" not in result
        assert "Optional" not in result
        assert "def foo" in result

    def test_keeps_when_all_used(self):
        from claude_token_saver.prep import _remove_unused_typing_imports
        inp = "from typing import List\ndef foo(x: List[int]):\n    pass\n"
        result = _remove_unused_typing_imports(inp)
        assert "from typing import List" in result

    def test_filters_partial_usage(self):
        from claude_token_saver.prep import _remove_unused_typing_imports
        inp = "from typing import Dict, List, Optional\ndef foo(x: Dict[str, int]) -> None:\n    pass\n"
        result = _remove_unused_typing_imports(inp)
        assert "from typing import Dict" in result
        assert "List" not in result
        assert "Optional" not in result

    def test_no_typing_imports(self):
        from claude_token_saver.prep import _remove_unused_typing_imports
        inp = "import os\ndef foo(x):\n    return x\n"
        result = _remove_unused_typing_imports(inp)
        assert result == inp

    def test_typing_used_as_value(self):
        from claude_token_saver.prep import _remove_unused_typing_imports
        inp = "from typing import Dict, List\nresult = isinstance({}, dict)\n"
        result = _remove_unused_typing_imports(inp)
        # Neither Dict nor List is used as a value
        assert "Dict" not in result
        assert "List" not in result

    def test_import_typing_unused(self):
        from claude_token_saver.prep import _remove_unused_typing_imports
        inp = "import typing\ndef foo(x):\n    return x\n"
        result = _remove_unused_typing_imports(inp)
        assert "import typing" not in result
        assert "def foo" in result

    def test_import_typing_used(self):
        from claude_token_saver.prep import _remove_unused_typing_imports
        inp = "import typing\ndef foo(x):\n    return typing.cast(int, x)\n"
        result = _remove_unused_typing_imports(inp)
        assert "import typing" in result

    def test_import_typing_as_unused(self):
        from claude_token_saver.prep import _remove_unused_typing_imports
        inp = "import typing as t\ndef foo(x):\n    return x\n"
        result = _remove_unused_typing_imports(inp)
        assert "import typing" not in result

    def test_import_typing_as_used(self):
        from claude_token_saver.prep import _remove_unused_typing_imports
        inp = "import typing as t\ndef foo(x):\n    return t.cast(int, x)\n"
        result = _remove_unused_typing_imports(inp)
        assert "import typing as t" in result
        assert "List" not in result


# ── 新增 stripped 模式优化测试 ──────────────────────────────────────────────

class TestRemoveEncodingAndShebang:
    """测试 _remove_encoding_and_shebang。"""

    def test_removes_shebang(self):
        inp = "#!/usr/bin/env python3\nprint('hello')\n"
        result = _remove_encoding_and_shebang(inp)
        assert "#!/" not in result
        assert "print" in result

    def test_removes_coding_declaration(self):
        inp = "# -*- coding: utf-8 -*-\nprint('hello')\n"
        result = _remove_encoding_and_shebang(inp)
        assert "coding" not in result
        assert "print" in result

    def test_removes_utf8_comment(self):
        inp = "# utf-8\nprint('hello')\n"
        result = _remove_encoding_and_shebang(inp)
        assert "utf-8" not in result
        assert "print" in result

    def test_no_shebang_unchanged(self):
        inp = "print('hello')\n"
        result = _remove_encoding_and_shebang(inp)
        assert result == inp

    def test_both_shebang_and_coding(self):
        inp = "#!/usr/bin/env python3\n# coding: utf-8\nprint('hello')\n"
        result = _remove_encoding_and_shebang(inp)
        assert "#!/" not in result
        assert "coding" not in result
        assert "print" in result


class TestSimplifyBooleanChecks:
    """测试 _simplify_boolean_checks。"""

    def test_if_is_true(self):
        inp = "if x is True:\n    pass\n"
        result = _simplify_boolean_checks(inp)
        assert "if x:" in result
        assert "is True" not in result

    def test_if_is_false(self):
        inp = "if x is False:\n    pass\n"
        result = _simplify_boolean_checks(inp)
        assert "if not x:" in result
        assert "is False" not in result

    def test_elif_is_true(self):
        inp = "if x:\n    pass\nelif y is True:\n    pass\n"
        result = _simplify_boolean_checks(inp)
        assert "elif y:" in result
        assert "is True" not in result

    def test_while_is_true(self):
        inp = "while flag is True:\n    pass\n"
        result = _simplify_boolean_checks(inp)
        assert "while flag:" in result

    def test_if_not_x_is_false(self):
        inp = "if not x is False:\n    pass\n"
        result = _simplify_boolean_checks(inp)
        assert "if x:" in result

    def test_if_x_is_not_false(self):
        inp = "if x is not False:\n    pass\n"
        result = _simplify_boolean_checks(inp)
        assert "if x:" in result

    def test_if_not_x_is_true(self):
        inp = "if not x is True:\n    pass\n"
        result = _simplify_boolean_checks(inp)
        assert "if not x:" in result

    def test_if_x_is_not_true(self):
        inp = "if x is not True:\n    pass\n"
        result = _simplify_boolean_checks(inp)
        assert "if not x:" in result

    def test_if_eq_true(self):
        inp = "if x == True:\n    pass\n"
        result = _simplify_boolean_checks(inp)
        assert "if x:" in result

    def test_if_eq_false(self):
        inp = "if x == False:\n    pass\n"
        result = _simplify_boolean_checks(inp)
        assert "if not x:" in result

    def test_no_boolean_check_unchanged(self):
        inp = "if x > 0:\n    pass\n"
        result = _simplify_boolean_checks(inp)
        assert result == inp

    def test_does_not_modify_string_content(self):
        inp = "s = 'if x is True:'\n"
        result = _simplify_boolean_checks(inp)
        assert result == inp


class TestRemoveRedundantReturnNone:
    """测试 _remove_redundant_return_none。"""

    def test_trailing_return_none_removed(self):
        inp = "def foo():\n    return None\n"
        result = _remove_redundant_return_none(inp)
        assert "return None" not in result

    def test_return_none_with_code_before(self):
        inp = "def foo():\n    x = 1\n    return None\n"
        result = _remove_redundant_return_none(inp)
        assert "return None" not in result

    def test_return_with_value_kept(self):
        inp = "def foo():\n    return 42\n"
        result = _remove_redundant_return_none(inp)
        assert "return 42" in result

    def test_return_none_in_branch_kept(self):
        """return None 作为分支唯一语句时不应移除（否则会导致空块 SyntaxError）。"""
        inp = "def foo(x):\n    if x:\n        return None\n    return 0\n"
        result = _remove_redundant_return_none(inp)
        # return None 是 if 块唯一语句，不能移除（否则 if x: 变成空块）
        assert "return None" in result
        assert "return 0" in result

    def test_no_return_unchanged(self):
        inp = "def foo():\n    x = 1\n"
        result = _remove_redundant_return_none(inp)
        assert result == inp


class TestSimplifyIsinstanceChecks:
    """测试 _simplify_isinstance_checks。"""

    def test_isinstance_none_to_is_none(self):
        inp = "if isinstance(x, type(None)):\n    pass\n"
        result = _simplify_isinstance_checks(inp)
        assert "x is None" in result
        assert "isinstance" not in result

    def test_isinstance_not_none(self):
        inp = "if not isinstance(x, type(None)):\n    pass\n"
        result = _simplify_isinstance_checks(inp)
        assert "x is None" in result
        assert "isinstance" not in result

    def test_isinstance_bool_kept(self):
        inp = "if isinstance(x, bool):\n    pass\n"
        result = _simplify_isinstance_checks(inp)
        assert "isinstance(x, bool)" in result

    def test_isinstance_int_kept(self):
        inp = "if isinstance(x, int):\n    pass\n"
        result = _simplify_isinstance_checks(inp)
        assert result == inp


class TestRemoveEmptySpecialMethods:
    """测试 _remove_empty_special_methods。"""

    def test_removes_empty_repr(self):
        inp = "class Foo:\n    def __repr__(self):\n        pass\n"
        result = _remove_empty_special_methods(inp)
        assert "__repr__" not in result

    def test_removes_empty_str(self):
        inp = "class Foo:\n    def __str__(self):\n        pass\n"
        result = _remove_empty_special_methods(inp)
        assert "__str__" not in result

    def test_removes_with_docstring_and_pass(self):
        inp = 'class Foo:\n    def __repr__(self):\n        """Return repr."""\n        pass\n'
        result = _remove_empty_special_methods(inp)
        assert "__repr__" not in result

    def test_keeps_method_with_body(self):
        inp = "class Foo:\n    def __repr__(self):\n        return self.name\n"
        result = _remove_empty_special_methods(inp)
        assert "__repr__" in result

    def test_keeps_non_special_method(self):
        inp = "class Foo:\n    def regular_method(self):\n        pass\n"
        result = _remove_empty_special_methods(inp)
        assert "regular_method" in result


class TestCompressCommonPatterns:
    """测试 _compress_common_patterns。"""

    def test_removes_empty_class(self):
        inp = "class Foo:\n    pass\n"
        result = _compress_common_patterns(inp)
        assert "class Foo" not in result

    def test_keeps_class_with_body(self):
        inp = "class Foo:\n    x = 1\n    def bar(self):\n        pass\n"
        result = _compress_common_patterns(inp)
        assert "class Foo" in result

    def test_no_change_on_normal_code(self):
        inp = "x = 1\nprint(x)\n"
        result = _compress_common_patterns(inp)
        assert result == inp


class TestCompressTruthinessChecks:
    """测试 _compress_truthiness_checks。"""

    def test_eq_true_to_truthy(self):
        inp = "if x == True:\n    pass\n"
        result = _compress_truthiness_checks(inp)
        assert "if x:" in result
        assert "== True" not in result

    def test_eq_false_to_not(self):
        inp = "if x == False:\n    pass\n"
        result = _compress_truthiness_checks(inp)
        assert "if not x:" in result

    def test_not_eq_true(self):
        inp = "if not x == True:\n    pass\n"
        result = _compress_truthiness_checks(inp)
        assert "if not x:" in result

    def test_ne_false(self):
        inp = "if x != False:\n    pass\n"
        result = _compress_truthiness_checks(inp)
        assert "if x:" in result

    def test_or_true(self):
        inp = "if a or True:\n    pass\n"
        result = _compress_truthiness_checks(inp)
        assert "if a:" in result
        assert "or True" not in result

    def test_and_false(self):
        inp = "if a and False:\n    pass\n"
        result = _compress_truthiness_checks(inp)
        assert "False" in result

    def test_or_false(self):
        inp = "if a or False:\n    pass\n"
        result = _compress_truthiness_checks(inp)
        assert "if a:" in result

    def test_and_true(self):
        inp = "if a and True:\n    pass\n"
        result = _compress_truthiness_checks(inp)
        assert "if a:" in result

    def test_while_truthy(self):
        inp = "while flag == True:\n    pass\n"
        result = _compress_truthiness_checks(inp)
        assert "while flag:" in result

    def test_assert_truthy(self):
        inp = "assert x == True\n"
        result = _compress_truthiness_checks(inp)
        assert "assert x" in result

    def test_no_truthy_unchanged(self):
        inp = "if x > 0:\n    pass\n"
        result = _compress_truthiness_checks(inp)
        assert result == inp


class TestCompressEmptyCollections:
    """测试 _compress_empty_collections。"""

    def test_list_to_bracket(self):
        inp = "x = list()\n"
        result = _compress_empty_collections(inp)
        assert "x = []" in result
        assert "list()" not in result

    def test_dict_to_brace(self):
        inp = "x = dict()\n"
        result = _compress_empty_collections(inp)
        assert "x = {}" in result
        assert "dict()" not in result

    def test_set_unchanged(self):
        inp = "x = set()\n"
        result = _compress_empty_collections(inp)
        assert "set()" in result  # no shorter form

    def test_list_with_arg_kept(self):
        inp = "x = list(range(10))\n"
        result = _compress_empty_collections(inp)
        assert "list(range(10))" in result

    def test_no_empty_init_unchanged(self):
        inp = "x = [1, 2, 3]\n"
        result = _compress_empty_collections(inp)
        assert result == inp


class TestRemoveFstringNoInterpolation:
    """测试 _remove_fstring_no_interpolation。"""

    def test_fstring_double_quotes(self):
        inp = 's = f"hello world"\n'
        result = _remove_fstring_no_interpolation(inp)
        assert 's = "hello world"' in result
        assert 'f"' not in result

    def test_fstring_single_quotes(self):
        inp = "s = f'hello world'\n"
        result = _remove_fstring_no_interpolation(inp)
        assert "s = 'hello world'" in result
        assert "f'" not in result

    def test_fstring_with_interpolation_kept(self):
        inp = 's = f"hello {name}"\n'
        result = _remove_fstring_no_interpolation(inp)
        assert 'f"hello {name}"' in result

    def test_fstring_triple_quotes(self):
        inp = 's = f"""hello"""\n'
        result = _remove_fstring_no_interpolation(inp)
        assert 's = """hello"""' in result

    def test_no_fstring_unchanged(self):
        inp = 's = "hello world"\n'
        result = _remove_fstring_no_interpolation(inp)
        assert result == inp


class TestRemoveElseAfterFlowControl:
    """测试 _remove_else_after_flow_control。"""

    def test_else_after_return(self):
        inp = "def foo(x):\n    if x:\n        return 1\n    else:\n        return 0\n"
        result = _remove_else_after_flow_control(inp)
        assert "else:" not in result
        assert "return 1" in result
        assert "return 0" in result

    def test_elif_after_return(self):
        inp = "def foo(x):\n    if x:\n        return 1\n    elif y:\n        return 2\n    else:\n        return 0\n"
        result = _remove_else_after_flow_control(inp)
        # The outer if only has 1 statement (return 1), so else/elif should be flattened
        lines = result.split("\n")
        # After processing if x: return 1, the elif block (return 2) gets de-indented
        # The else: gets removed too
        else_count = sum(1 for l in lines if l.strip() == "else:")
        assert else_count == 0

    def test_else_with_multiple_stmts_kept(self):
        inp = "def foo(x):\n    if x:\n        return 1\n    else:\n        y = 2\n        return y\n"
        result = _remove_else_after_flow_control(inp)
        # if body is single return → else: removed, body de-indented
        assert "else:" not in result
        assert "y = 2" in result
        assert "return y" in result

    def test_no_else_unchanged(self):
        inp = "def foo(x):\n    if x:\n        return 1\n    return 0\n"
        result = _remove_else_after_flow_control(inp)
        assert result == inp


class TestRemoveTryExceptPass:
    """测试 _remove_try_except_pass。"""

    def test_removes_try_except_pass(self):
        inp = "try:\n    x = 1\nexcept:\n    pass\n"
        result = _remove_try_except_pass(inp)
        assert "try:" not in result
        assert "except:" not in result
        assert "pass" not in result
        assert "x = 1" in result

    def test_keeps_try_except_with_body(self):
        inp = "try:\n    x = 1\nexcept ValueError:\n    x = 0\n"
        result = _remove_try_except_pass(inp)
        assert "try:" in result
        assert "except" in result

    def test_keeps_try_except_pass_with_finally(self):
        inp = "try:\n    x = 1\nexcept:\n    pass\nfinally:\n    cleanup()\n"
        result = _remove_try_except_pass(inp)
        # try and except removed, finally kept
        assert "try:" not in result
        assert "except:" not in result
        assert "finally:" in result

    def test_no_try_unchanged(self):
        inp = "x = 1\n"
        result = _remove_try_except_pass(inp)
        assert result == inp


class TestMergeNestedIfs:
    """测试 _merge_nested_ifs。"""

    def test_merges_nested_if(self):
        inp = "if x:\n    if y:\n        pass\n"
        result = _merge_nested_ifs(inp)
        assert "if x and y:" in result
        assert "    if y:" not in result

    def test_keeps_if_with_else(self):
        inp = "if x:\n    if y:\n        pass\n    else:\n        pass\n"
        result = _merge_nested_ifs(inp)
        assert "if x and y:" not in result

    def test_no_nested_unchanged(self):
        inp = "if x:\n    y = 1\n    if y:\n        pass\n"
        result = _merge_nested_ifs(inp)
        # Outer body has 2 lines, no merge
        assert "if x and y:" not in result

    def test_single_nested_line(self):
        inp = "if x:\n    if y:\n        pass\n\nprint('done')\n"
        result = _merge_nested_ifs(inp)
        assert "if x and y:" in result


class TestSimplifyTernary:
    """测试 _simplify_ternary。"""

    def test_true_condition(self):
        inp = "x = a if True else b\n"
        result = _simplify_ternary(inp)
        assert "x = a" in result
        assert "if True else" not in result

    def test_false_condition(self):
        inp = "x = a if False else b\n"
        result = _simplify_ternary(inp)
        assert "x = b" in result

    def test_same_branches(self):
        inp = "x = y if cond else y\n"
        result = _simplify_ternary(inp)
        assert "x = y" in result
        assert "if cond" not in result

    def test_different_branches_kept(self):
        inp = "x = a if cond else b\n"
        result = _simplify_ternary(inp)
        assert "x = a if cond else b" in result


class TestRemoveTupleWrapSingle:
    """测试 _remove_tuple_wrap_single。"""

    def test_single_unpack(self):
        inp = "x, = [1, 2, 3]\n"
        result = _remove_tuple_wrap_single(inp)
        assert "x = [1, 2, 3]" in result
        assert "," not in result.split("\n")[0].split("=")[0]

    def test_normal_assignment_unchanged(self):
        inp = "x, y = 1, 2\n"
        result = _remove_tuple_wrap_single(inp)
        assert result == inp

    def test_no_tuple_unchanged(self):
        inp = "x = [1, 2, 3]\n"
        result = _remove_tuple_wrap_single(inp)
        assert result == inp


class TestListToGenerator:
    """测试 _list_to_generator。"""

    def test_any_list_comp(self):
        inp = "any([x > 0 for x in items])\n"
        result = _list_to_generator(inp)
        assert "any(x > 0 for x in items)" in result

    def test_all_list_comp(self):
        inp = "all([x > 0 for x in items])\n"
        result = _list_to_generator(inp)
        assert "all(x > 0 for x in items)" in result

    def test_sum_list_comp(self):
        inp = "sum([x for x in items])\n"
        result = _list_to_generator(inp)
        assert "sum(x for x in items)" in result

    def test_join_list(self):
        inp = '",".join([a, b, c])\n'
        result = _list_to_generator(inp)
        assert '",".join' in result

    def test_max_list(self):
        inp = "max([1, 2, 3])\n"
        result = _list_to_generator(inp)
        assert "max(" in result

    def test_keeps_list_with_args(self):
        inp = "list(range(10))\n"
        result = _list_to_generator(inp)
        assert "list(range(10))" in result

    def test_no_transform_unchanged(self):
        inp = "x = [1, 2, 3]\n"
        result = _list_to_generator(inp)
        assert result == inp


class TestRemoveNotNot:
    """测试 _remove_not_not。"""

    def test_not_not_removed(self):
        inp = "if not not x:\n    pass\n"
        result = _remove_not_not(inp)
        assert "if x:" in result
        assert "not not" not in result

    def test_not_not_in_assignment(self):
        inp = "y = not not x\n"
        result = _remove_not_not(inp)
        assert "y = x" in result

    def test_not_not_in_return(self):
        inp = "return not not x\n"
        result = _remove_not_not(inp)
        assert "return x" in result

    def test_single_not_kept(self):
        inp = "if not x:\n    pass\n"
        result = _remove_not_not(inp)
        assert result == inp


class TestRemoveRedundantParens:
    """测试 _remove_redundant_parens。"""

    def test_removes_single_cond_parens(self):
        inp = "if (x > 0):\n    pass\n"
        result = _remove_redundant_parens(inp)
        assert "if x > 0:" in result

    def test_removes_not_parens(self):
        inp = "if not (x):\n    pass\n"
        result = _remove_redundant_parens(inp)
        assert "if not x:" in result

    def test_while_parens(self):
        inp = "while (flag):\n    pass\n"
        result = _remove_redundant_parens(inp)
        assert "while flag:" in result

    def test_no_parens_unchanged(self):
        inp = "if x > 0:\n    pass\n"
        result = _remove_redundant_parens(inp)
        assert result == inp

    def test_keeps_compound_parens(self):
        inp = "if (a or b) and c:\n    pass\n"
        result = _remove_redundant_parens(inp)
        assert "(a or b)" in result


class TestRemoveListWrap:
    """测试 _remove_list_wrap。"""

    def test_list_dict_keys(self):
        inp = "keys = list(d.keys())\n"
        result = _remove_list_wrap(inp)
        assert "d.keys()" in result
        assert "list(" not in result

    def test_list_dict_values(self):
        inp = "vals = list(d.values())\n"
        result = _remove_list_wrap(inp)
        assert "d.values()" in result

    def test_list_dict_items(self):
        inp = "items = list(d.items())\n"
        result = _remove_list_wrap(inp)
        assert "d.items()" in result

    def test_tuple_empty(self):
        inp = "x = tuple()\n"
        result = _remove_list_wrap(inp)
        assert "x = ()" in result

    def test_set_list(self):
        inp = "s = set([1, 2, 3])\n"
        result = _remove_list_wrap(inp)
        assert "{1, 2, 3}" in result

    def test_list_list(self):
        inp = "x = list([1, 2, 3])\n"
        result = _remove_list_wrap(inp)
        assert "[1, 2, 3]" in result

    def test_keeps_list_with_args(self):
        inp = "x = list(range(10))\n"
        result = _remove_list_wrap(inp)
        assert "list(range(10))" in result


class TestFlattenNestedTernary:
    """测试 _flatten_nested_ternary。"""

    def test_flattens_same_true(self):
        inp = "x = a if b else a if c else d\n"
        result = _flatten_nested_ternary(inp)
        assert "and" in result

    def test_no_nested_unchanged(self):
        inp = "x = a if b else c\n"
        result = _flatten_nested_ternary(inp)
        assert result == inp


class TestRemoveUnusedFunctions:
    """测试 _remove_unused_functions。"""

    def test_removes_unused_function(self):
        inp = "def unused_helper():\n    x = 1\n    return x\ndef main():\n    pass\n"
        result = _remove_unused_functions(inp)
        assert "unused_helper" not in result
        assert "main" in result

    def test_keeps_used_function(self):
        inp = "def helper():\n    x = 1\n    return x\nresult = helper()\n"
        result = _remove_unused_functions(inp)
        assert "helper" in result

    def test_keeps_main(self):
        inp = "def main():\n    x = 1\n    return x\n"
        result = _remove_unused_functions(inp)
        assert "main" in result

    def test_removes_multiple_unused(self):
        inp = "def a():\n    pass\ndef b():\n    pass\ndef c():\n    pass\nresult = c()\n"
        result = _remove_unused_functions(inp)
        assert "def a():" not in result
        assert "def b():" not in result
        assert "def c():" in result

    def test_keeps_private_functions(self):
        inp = "def _private():\n    pass\ndef main():\n    pass\n"
        result = _remove_unused_functions(inp)
        assert "_private" in result


class TestInlineSingleUseVars:
    """测试 _inline_single_use_vars。"""

    def test_inlines_single_use(self):
        inp = "def f():\n    x = 5 + 3\n    return x\n"
        result = _inline_single_use_vars(inp)
        assert "5 + 3" in result
        assert "x =" not in result

    def test_no_inline_multiple_use(self):
        inp = "def f():\n    x = 5\n    y = x + 1\n    z = x + 2\n    return z\n"
        result = _inline_single_use_vars(inp)
        # x is used twice, should not be inlined
        assert "x = 5" in result

    def test_no_inline_self(self):
        inp = "def f(self):\n    self.x = 5\n    return self.x\n"
        result = _inline_single_use_vars(inp)
        assert "self.x = 5" in result

    def test_inlines_complex_expr(self):
        inp = "def f():\n    x = some_func(a, b, c, d, e, f)\n    return x\n"
        result = _inline_single_use_vars(inp)
        # Even complex expressions get inlined if used only once
        assert "return some_func(a, b, c, d, e, f)" in result
        assert "x =" not in result


class TestRemoveUnusedClasses:
    """测试 _remove_unused_classes。"""

    def test_removes_unused_class(self):
        inp = "class Unused:\n    def method(self):\n        pass\nclass Used:\n    pass\nobj = Used()\n"
        result = _remove_unused_classes(inp)
        assert "Unused" not in result
        assert "Used" in result

    def test_keeps_class_with_bases(self):
        inp = "class MyError(Exception):\n    pass\n"
        result = _remove_unused_classes(inp)
        assert "MyError" in result

    def test_keeps_private_class(self):
        inp = "class _Private:\n    pass\n"
        result = _remove_unused_classes(inp)
        assert "_Private" in result

    def test_removes_multiple_unused(self):
        inp = "class A:\n    pass\nclass B:\n    pass\nclass C:\n    pass\nobj = C()\n"
        result = _remove_unused_classes(inp)
        assert "A" not in result
        assert "B" not in result
        assert "C" in result


class TestInvertDeadIf:
    """测试 _invert_dead_if。"""

    def test_inverts_if_pass_else(self):
        inp = "if x:\n    pass\nelse:\n    return 1\n"
        result = _invert_dead_if(inp)
        assert "if not x:" in result
        assert "pass" not in result
        assert "return 1" in result

    def test_keeps_if_with_body(self):
        inp = "if x:\n    return 1\nelse:\n    return 0\n"
        result = _invert_dead_if(inp)
        # if body is not just pass, no inversion
        assert "if x:" in result

    def test_no_else_unchanged(self):
        inp = "if x:\n    pass\nreturn 1\n"
        result = _invert_dead_if(inp)
        assert result == inp


class TestMergeDuplicateConditions:
    """测试 _merge_duplicate_conditions。"""

    def test_merges_elif_same_cond(self):
        inp = "if x:\n    return 1\nelif x:\n    return 1\nelse:\n    return 0\n"
        result = _merge_duplicate_conditions(inp)
        assert "elif x:" not in result

    def test_keeps_different_conditions(self):
        inp = "if x:\n    return 1\nelif y:\n    return 2\nelse:\n    return 0\n"
        result = _merge_duplicate_conditions(inp)
        assert "elif y:" in result

    def test_merges_multiple_dupes(self):
        inp = "if x:\n    return 1\nelif x:\n    return 1\nelif x:\n    return 1\n"
        result = _merge_duplicate_conditions(inp)
        assert result.count("elif x:") == 0


class TestRangeLenToEnumerate:
    """测试 _range_len_to_enumerate。"""

    def test_converts_range_len(self):
        inp = "for i in range(len(items)):\n    x = items[i]\n"
        result = _range_len_to_enumerate(inp)
        assert "for i, item in enumerate(items):" in result
        assert "items[i]" not in result

    def test_keeps_range_without_index(self):
        inp = "for i in range(len(items)):\n    print(i)\n"
        result = _range_len_to_enumerate(inp)
        assert "range(len(items))" in result

    def test_keeps_regular_for(self):
        inp = "for x in items:\n    print(x)\n"
        result = _range_len_to_enumerate(inp)
        assert result == inp


class TestRemoveReturnNoneAfterNoneCheck:
    """测试 _remove_return_none_after_none_check。"""

    def test_removes_none_in_return(self):
        inp = "if x is None:\n    return None\n"
        result = _remove_return_none_after_none_check(inp)
        assert "return None" not in result
        assert "return" in result

    def test_keeps_other_returns(self):
        inp = "if x is None:\n    return 0\n"
        result = _remove_return_none_after_none_check(inp)
        assert "return 0" in result

    def test_no_none_check_unchanged(self):
        inp = "if x:\n    return None\n"
        result = _remove_return_none_after_none_check(inp)
        assert result == inp


# ── 新深度优化函数测试 ──────────────────────────────────────────────────

class TestRemoveUnreachableCode:
    """测试 _remove_unreachable_code。"""

    def test_removes_dead_code_after_return(self):
        inp = "def f():\n    if x:\n        return\n        y = 1\n"
        result = _remove_unreachable_code(inp)
        assert "y = 1" not in result

    def test_removes_dead_code_after_raise(self):
        inp = "def f():\n    raise ValueError()\n    x = 1\n"
        result = _remove_unreachable_code(inp)
        assert "x = 1" not in result

    def test_removes_dead_code_after_break(self):
        inp = "def f():\n    for i in range(10):\n        if x:\n            break\n            y = 1\n"
        result = _remove_unreachable_code(inp)
        assert "y = 1" not in result

    def test_removes_dead_code_after_continue(self):
        inp = "def f():\n    for i in range(10):\n        if x:\n            continue\n            y = 1\n"
        result = _remove_unreachable_code(inp)
        assert "y = 1" not in result

    def test_keeps_code_after_if_true_branch(self):
        inp = "def f():\n    if x:\n        return\n    y = 1\n"
        result = _remove_unreachable_code(inp)
        assert "y = 1" in result

    def test_module_level_dead_code(self):
        inp = "x = 1\nreturn\n"
        result = _remove_unreachable_code(inp)
        assert "return" not in result

    def test_no_jump_unchanged(self):
        inp = "def f():\n    x = 1\n    y = 2\n"
        result = _remove_unreachable_code(inp)
        assert "x = 1" in result
        assert "y = 2" in result


class TestSimplifyEnumerateStartZero:
    """测试 _simplify_enumerate_start_zero。"""

    def test_removes_start_zero(self):
        inp = "for i, x in enumerate(items, 0):\n    pass\n"
        result = _simplify_enumerate_start_zero(inp)
        assert "enumerate(items)" in result
        assert ", 0" not in result

    def test_keeps_nonzero_start(self):
        inp = "for i, x in enumerate(items, 1):\n    pass\n"
        result = _simplify_enumerate_start_zero(inp)
        assert "enumerate(items, 1)" in result

    def test_keeps_no_start(self):
        inp = "for i, x in enumerate(items):\n    pass\n"
        result = _simplify_enumerate_start_zero(inp)
        assert "enumerate(items)" in result

    def test_whitespace_variants(self):
        inp = "for i, x in enumerate(items,0):\n    pass\n"
        result = _simplify_enumerate_start_zero(inp)
        assert "enumerate(items)" in result


class TestRemoveUnusedExceptBlocks:
    """测试 _remove_unused_except_blocks。"""

    def test_removes_caught_but_unraised_exception(self):
        inp = "try:\n    int(x)\nexcept KeyError:\n    pass\n"
        result = _remove_unused_except_blocks(inp)
        assert "KeyError" not in result
        assert "except" not in result

    def test_keeps_raised_exception(self):
        inp = "try:\n    raise KeyError()\nexcept KeyError:\n    pass\n"
        result = _remove_unused_except_blocks(inp)
        assert "KeyError" in result

    def test_keeps_non_passthrough_block(self):
        inp = "try:\n    int(x)\nexcept KeyError:\n    print('error')\n"
        result = _remove_unused_except_blocks(inp)
        assert "KeyError" in result  # body does real work, keep it

    def test_no_except_unchanged(self):
        inp = "x = 1\n"
        result = _remove_unused_except_blocks(inp)
        assert result == inp


class TestSimplifySuperCalls:
    """测试 _simplify_super_calls。"""

    def test_super_class_self(self):
        inp = "class Foo:\n    def bar(self):\n        return super(Foo, self)\n"
        result = _simplify_super_calls(inp)
        assert "super()" in result
        assert "super(Foo, self)" not in result

    def test_super_class_cls(self):
        inp = "class Foo:\n    @classmethod\n    def bar(cls):\n        return super(Foo, cls)\n"
        result = _simplify_super_calls(inp)
        assert "super()" in result

    def test_keeps_simple_super(self):
        inp = "class Foo:\n    def bar(self):\n        return super()\n"
        result = _simplify_super_calls(inp)
        assert "super()" in result


class TestCollapseDuplicateLines:
    """测试 _collapse_duplicate_lines。"""

    def test_collapses_two_identical_lines(self):
        inp = "def f():\n    x = 1\n    x = 1\n    y = 2\n"
        result = _collapse_duplicate_lines(inp)
        assert "*2" in result

    def test_keeps_different_lines(self):
        inp = "def f():\n    x = 1\n    y = 2\n"
        result = _collapse_duplicate_lines(inp)
        assert "*" not in result

    def test_skips_def_class_lines(self):
        inp = "def f():\n    pass\ndef g():\n    pass\n"
        result = _collapse_duplicate_lines(inp)
        # def lines should not be collapsed
        assert result.count("def ") == 2

    def test_skips_comments(self):
        inp = "def f():\n    # hello\n    # hello\n"
        result = _collapse_duplicate_lines(inp)
        assert "*" not in result


class TestCompressAsserts:
    """测试 _compress_asserts。"""

    def test_removes_assert_true(self):
        inp = "assert True\n"
        result = _compress_asserts(inp)
        assert "assert" not in result

    def test_removes_assert_false_body(self):
        inp = "def f():\n    assert False\n    x = 1\n"
        result = _compress_asserts(inp)
        assert "assert False" not in result

    def test_removes_assert_constant_compare(self):
        inp = "assert 1 == 1\n"
        result = _compress_asserts(inp)
        assert "assert" not in result

    def test_keeps_variable_assert(self):
        inp = "def f():\n    assert x > 0\n"
        result = _compress_asserts(inp)
        assert "assert x > 0" in result

    def test_no_assert_unchanged(self):
        inp = "def f():\n    x = 1\n"
        result = _compress_asserts(inp)
        assert result == inp


class TestMergeSameBodyConditions:
    """测试 _merge_same_body_conditions。"""

    def test_merges_if_elif_same_body(self):
        inp = "if x == 1:\n    handle()\nelif x == 2:\n    handle()\n"
        result = _merge_same_body_conditions(inp)
        assert "or" in result
        assert "elif" not in result

    def test_keeps_different_bodies(self):
        inp = "if x == 1:\n    handle_a()\nelif x == 2:\n    handle_b()\n"
        result = _merge_same_body_conditions(inp)
        assert result == inp

    def test_single_if_unchanged(self):
        inp = "if x == 1:\n    handle()\n"
        result = _merge_same_body_conditions(inp)
        assert result == inp

    def test_merges_three_conditions(self):
        inp = "if x == 1:\n    f()\nelif x == 2:\n    f()\nelif x == 3:\n    f()\n"
        result = _merge_same_body_conditions(inp)
        assert result.count("or") >= 2
        assert result.count("elif") == 0


class TestRemoveDeadAfterLoop:
    """测试 _remove_dead_after_loop。"""

    def test_removes_dead_after_while_true(self):
        inp = "while True:\n    do_something()\nprint('never')\n"
        result = _remove_dead_after_loop(inp)
        assert "never" not in result

    def test_removes_dead_after_while_1(self):
        inp = "while 1:\n    do_something()\nprint('never')\n"
        result = _remove_dead_after_loop(inp)
        assert "never" not in result

    def test_keeps_code_after_normal_for(self):
        inp = "for i in range(10):\n    pass\nprint('done')\n"
        result = _remove_dead_after_loop(inp)
        assert "done" in result

    def test_no_loop_unchanged(self):
        inp = "x = 1\nprint(x)\n"
        result = _remove_dead_after_loop(inp)
        assert result == inp


class TestMergeAdjacentStringLiterals:
    """测试 _merge_adjacent_string_literals。"""

    def test_merges_two_adjacent_literals(self):
        inp = 'x = "hello"\n    "world"\n'
        result = _merge_adjacent_string_literals(inp)
        assert "helloworld" in result
        assert "hello" not in result.split("helloworld")[0]

    def test_no_merge_with_variable(self):
        inp = 'x = "hello"\n    y\n'
        result = _merge_adjacent_string_literals(inp)
        assert result == inp

    def test_no_merge_single_string(self):
        inp = 'x = "hello"\n'
        result = _merge_adjacent_string_literals(inp)
        assert result == inp

    def test_merges_in_function(self):
        inp = 'def f():\n    return "a"\n        "b"\n        "c"\n'
        result = _merge_adjacent_string_literals(inp)
        assert "abc" in result


class TestSimplifyBoolExprWithConst:
    """测试 _simplify_bool_expr_with_const。"""

    def test_and_true_removed(self):
        inp = "if x and True:\n    pass\n"
        result = _simplify_bool_expr_with_const(inp)
        assert "True" not in result

    def test_or_false_removed(self):
        inp = "if x or False:\n    pass\n"
        result = _simplify_bool_expr_with_const(inp)
        assert "False" not in result

    def test_true_and_keeps_other(self):
        inp = "if True and x:\n    pass\n"
        result = _simplify_bool_expr_with_const(inp)
        assert "x" in result
        assert "True" not in result

    def test_no_simplify_bool_vars(self):
        inp = "if x and y:\n    pass\n"
        result = _simplify_bool_expr_with_const(inp)
        assert result == inp


class TestSimplifyNoneCheckReturn:
    """测试 _simplify_none_check_return。"""

    def test_return_none_after_none_check(self):
        inp = "if x is not None:\n    return x\n"
        result = _simplify_none_check_return(inp)
        assert "return" in result
        assert "return x" not in result
        assert "is not None" in result

    def test_return_none_after_is_none_check(self):
        inp = "if x is None:\n    return None\n"
        result = _simplify_none_check_return(inp)
        assert "return" in result
        assert "return None" not in result
        assert "is None" in result

    def test_keeps_is_none_check(self):
        inp = "if x is not None:\n    return None\n"
        result = _simplify_none_check_return(inp)
        assert "return" in result

    def test_no_change_when_not_return_var(self):
        inp = "if x is not None:\n    print(x)\n"
        result = _simplify_none_check_return(inp)
        assert result == inp

    def test_no_change_when_check_is_not_none(self):
        inp = "if x > 0:\n    return x\n"
        result = _simplify_none_check_return(inp)
        assert result == inp


class TestSimplifyIsinstanceAndNotIn:
    """测试 _simplify_isinstance_and_not_in。"""

    def test_simplifies_isinstance_and_not_in(self):
        inp = 'if isinstance(x, dict) and "key" not in x:\n    pass\n'
        result = _simplify_isinstance_and_not_in(inp)
        assert "isinstance" not in result
        assert "not in x" in result

    def test_no_change_without_isinstance(self):
        inp = 'if x and "key" not in y:\n    pass\n'
        result = _simplify_isinstance_and_not_in(inp)
        assert result == inp

    def test_no_change_when_not_same_var(self):
        inp = 'if isinstance(x, dict) and "key" not in y:\n    pass\n'
        result = _simplify_isinstance_and_not_in(inp)
        assert result == inp


class TestRemoveTryExceptReraise:
    """测试 _remove_try_except_reraise。"""

    def test_removes_try_except_reraise(self):
        inp = "try:\n    x = 1\nexcept ValueError:\n    raise\n"
        result = _remove_try_except_reraise(inp)
        assert "try" not in result
        assert "x = 1" in result

    def test_keeps_try_with_real_handler(self):
        inp = "try:\n    x = 1\nexcept ValueError:\n    x = 2\n"
        result = _remove_try_except_reraise(inp)
        assert "try" in result

    def test_removes_multiple_handlers_all_reraise(self):
        inp = "try:\n    x = 1\nexcept ValueError:\n    raise\nexcept TypeError:\n    raise\n"
        result = _remove_try_except_reraise(inp)
        assert "try" not in result
        assert "x = 1" in result


class TestMergeConsecutiveAttrAssignments:
    """测试 _merge_consecutive_attr_assignments。"""

    def test_removes_redundant_attr_assign(self):
        inp = "obj.x = 1\nobj.x = 2\n"
        result = _merge_consecutive_attr_assignments(inp)
        assert "obj.x = 1" not in result
        assert "obj.x = 2" in result

    def test_keeps_different_attrs(self):
        inp = "obj.x = 1\nobj.y = 2\n"
        result = _merge_consecutive_attr_assignments(inp)
        assert "obj.x = 1" in result
        assert "obj.y = 2" in result

    def test_keeps_non_consecutive(self):
        inp = "obj.x = 1\nprint('hi')\nobj.x = 2\n"
        result = _merge_consecutive_attr_assignments(inp)
        assert "obj.x = 1" in result
        assert "obj.x = 2" in result


class TestRemoveAwaitNoop:
    """测试 _remove_await_noop。"""

    def test_removes_await_sleep_zero(self):
        inp = "import asyncio\nasync def f():\n    await asyncio.sleep(0)\n    x = 1\n"
        result = _remove_await_noop(inp)
        assert "sleep" not in result

    def test_keeps_await_non_zero(self):
        inp = "import asyncio\nasync def f():\n    await asyncio.sleep(1)\n"
        result = _remove_await_noop(inp)
        assert "sleep" in result


class TestSimplifyIdentityChecks:
    """测试 _simplify_identity_checks。"""

    def test_x_is_x_to_true(self):
        inp = "if x is x:\n    pass\n"
        result = _simplify_identity_checks(inp)
        assert "True" in result

    def test_x_is_not_x_to_false(self):
        inp = "if x is not x:\n    pass\n"
        result = _simplify_identity_checks(inp)
        assert "False" in result

    def test_none_is_none_to_true(self):
        inp = "if None is None:\n    pass\n"
        result = _simplify_identity_checks(inp)
        assert "True" in result

    def test_no_change_different_vars(self):
        inp = "if x is y:\n    pass\n"
        result = _simplify_identity_checks(inp)
        assert result == inp


class TestRemoveEmptyWith:
    """测试 _remove_empty_with。"""

    def test_removes_empty_with(self):
        inp = "with open(f) as fh:\n    pass\n"
        result = _remove_empty_with(inp)
        assert "with" not in result

    def test_keeps_with_body(self):
        inp = "with open(f) as fh:\n    data = fh.read()\n"
        result = _remove_empty_with(inp)
        assert "with" in result