"""Tests for claude-token-saver"""
from __future__ import annotations

import pytest

from claude_token_saver.utils import count_tokens, is_binary_file, should_ignore
from claude_token_saver.prep import (
    strip_comments, strip_python_docstrings, smart_truncate,
    deduplicate_files, compress_prompt, process_files,
)
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
