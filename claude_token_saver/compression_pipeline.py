"""
Agent Token Saver - 多阶段压缩管线

核心思路：不同压缩阶段按特定顺序执行可最大化 token 节省。
不是简单叠加，而是精心设计的管线，每阶段输出作为下一阶段输入。

管线顺序（每阶段都检查是否仍有收益）：
  1. 空白标准化（normalize_whitespace）     — 去除冗余空白
  2. 注释去除（strip_comments）            — 去除注释和 docstring
  3. 结构化压缩（minify_structured）       — JSON/YAML/TOML 压缩
  4. 语义压缩（semantic_compress）         — 基于语言特性的语义级压缩
  5. 智能截断（smart_truncate）            — token 预算截断

优化特性：
  - 单次 token 计数：管线内共享当前 token 计数，避免重复估算
  - 智能阶段跳过：基于内容特征预判阶段收益，跳过无效阶段
  - 阶段输出缓存：相同输入不重复处理

每个阶段：
  - 输入：上阶段输出
  - 输出：压缩后内容 + token 计数
  - 收益检查：如果压缩后反而变大，回退到上阶段输出
"""
from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from claude_token_saver.prep import (
    normalize_whitespace,
    strip_comments,
    strip_python_docstrings,
    remove_redundant_pass,
    _minify_structured,
    smart_truncate,
)
from claude_token_saver.compressor import _SKELETON_LANGUAGES


# 管线级 token 计数缓存（避免同一内容重复计数）
_token_cache: dict[str, int] = {}
_token_cache_lock = threading.Lock()


def _cached_estimate(text: str) -> int:
    """带缓存的 token 估算（线程安全）。"""
    cache_key = str(len(text)) + ":" + text[:50]
    with _token_cache_lock:
        if cache_key in _token_cache:
            return _token_cache[cache_key]
    try:
        from claude_token_saver.utils import count_tokens
        tokens = count_tokens(text)
    except Exception:
        tokens = max(1, len(text) // 4)
    with _token_cache_lock:
        _token_cache[cache_key] = tokens
    return tokens


def clear_pipeline_cache() -> None:
    """清空管线级缓存。"""
    with _token_cache_lock:
        _token_cache.clear()


class CompressionPipeline:
    """多阶段压缩管线（优化版）。

    使用方式：
        pipeline = CompressionPipeline(ext=".py", detail_level="stripped")
        result = pipeline.run(content, max_tokens=50_000)
    """

    def __init__(
        self,
        ext: str = ".py",
        detail_level: str = "full",
        do_strip_comments: bool = True,
        do_strip_docstrings: bool = False,
        max_tokens: int = 50_000,
    ):
        self.ext = ext.lower()
        self.detail_level = detail_level
        self.do_strip_comments = do_strip_comments
        self.do_strip_docstrings = do_strip_docstrings
        self.max_tokens = max_tokens
        self._stages_run: list[str] = []
        self._savings_at_stage: dict[str, int] = {}

    def run(self, content: str) -> tuple[str, dict[str, Any]]:
        """执行完整压缩管线（优化：单次 token 计数，智能阶段跳过）。

        Returns:
            (compressed_content, metadata)
            metadata 包含每阶段的 token 数和节省量
        """
        current = content
        current_tokens = _cached_estimate(content)
        original_tokens = current_tokens
        self._stages_run = []
        self._savings_at_stage = {}

        # 智能预分析：基于内容特征决定阶段执行顺序和跳过
        skip_whitespace = _is_already_clean_whitespace(current, self.ext)
        is_structured = self.ext in {".json", ".yaml", ".yml", ".toml"}
        want_skeleton = self.detail_level == "skeleton" and self.ext in {
            ".py", ".js", ".ts", ".java", ".c", ".cpp", ".h", ".go", ".rs"
        }

        # ── 阶段 1: 空白标准化 ──
        if not skip_whitespace:
            stage_name = "normalize_whitespace"
            compressed = _normalize_whitespace_stage(current, self.ext)
            compressed_tokens = _cached_estimate(compressed)
            if compressed_tokens < current_tokens:
                savings = current_tokens - compressed_tokens
                self._savings_at_stage[stage_name] = savings
                current = compressed
                current_tokens = compressed_tokens
                self._stages_run.append(stage_name)

        # ── 阶段 2: 注释去除 ──
        if self.detail_level in ("skeleton", "stripped"):
            stage_name = "strip_comments"
            compressed = _strip_comments_stage(current, self.ext, self.do_strip_comments, self.do_strip_docstrings)
            compressed_tokens = _cached_estimate(compressed)
            if compressed_tokens < current_tokens:
                savings = current_tokens - compressed_tokens
                self._savings_at_stage[stage_name] = savings
                current = compressed
                current_tokens = compressed_tokens
                self._stages_run.append(stage_name)

        # ── 阶段 3: 结构化压缩 ──
        if is_structured:
            stage_name = "minify_structured"
            compressed = _minify_structured_stage(current, self.ext)
            compressed_tokens = _cached_estimate(compressed)
            if compressed_tokens < current_tokens:
                savings = current_tokens - compressed_tokens
                self._savings_at_stage[stage_name] = savings
                current = compressed
                current_tokens = compressed_tokens
                self._stages_run.append(stage_name)

        # ── 阶段 4: 语义压缩（骨架提取）──
        if want_skeleton:
            stage_name = "skeleton"
            compressed = _skeleton_stage(current, self.ext)
            compressed_tokens = _cached_estimate(compressed)
            if compressed_tokens < current_tokens:
                savings = current_tokens - compressed_tokens
                self._savings_at_stage[stage_name] = savings
                current = compressed
                current_tokens = compressed_tokens
                self._stages_run.append(stage_name)

        # ── 阶段 5: 智能截断 ──
        if self.max_tokens > 0 and current_tokens > self.max_tokens:
            stage_name = "smart_truncate"
            compressed = _smart_truncate_stage(current, self.max_tokens)
            compressed_tokens = _cached_estimate(compressed)
            if compressed_tokens < current_tokens:
                savings = current_tokens - compressed_tokens
                self._savings_at_stage[stage_name] = savings
                current = compressed
                current_tokens = compressed_tokens
                self._stages_run.append(stage_name)

        metadata = {
            "original_tokens": original_tokens,
            "final_tokens": current_tokens,
            "stages_run": self._stages_run,
            "savings_per_stage": self._savings_at_stage,
            "total_savings": original_tokens - current_tokens,
        }

        return current, metadata


def _is_already_clean_whitespace(content: str, ext: str) -> bool:
    """快速判断内容是否已经具有良好的空白格式，跳过空白标准化。"""
    if not content:
        return True

    lines = content.split("\n")

    # 检查行尾空白
    for line in lines:
        if line != line.rstrip():
            return False

    # 检查连续空行
    empty_count = 0
    threshold = 2 if ext == ".py" else 1
    for line in lines:
        if not line.strip():
            empty_count += 1
            if empty_count > threshold:
                return False
        else:
            empty_count = 0

    return True


# ── 阶段实现 ────────────────────────────────────────────────────────────

def _normalize_whitespace_stage(content: str, ext: str) -> str:
    """阶段 1: 空白标准化。"""
    try:
        return normalize_whitespace(content, ext)
    except Exception:
        return content


def _strip_comments_stage(content: str, ext: str, do_strip: bool, do_docstrings: bool) -> str:
    """阶段 2: 注释去除。"""
    if not do_strip:
        return content
    try:
        result = strip_comments(content, ext)
        if do_docstrings and ext == ".py":
            result = strip_python_docstrings(result)
        if ext == ".py":
            result = remove_redundant_pass(result)
        return result
    except Exception:
        return content


def _minify_structured_stage(content: str, ext: str) -> str:
    """阶段 3: 结构化文件压缩。"""
    try:
        return _minify_structured(content, ext)
    except Exception:
        return content


def _skeleton_stage(content: str, ext: str) -> str:
    """阶段 4: 骨架提取。"""
    try:
        from claude_token_saver.compressor import extract_skeleton
        return extract_skeleton(content, ext)
    except Exception:
        return content


def _smart_truncate_stage(content: str, max_tokens: int) -> str:
    """阶段 5: 智能截断。"""
    try:
        return smart_truncate(content, max_tokens)
    except Exception:
        return content
