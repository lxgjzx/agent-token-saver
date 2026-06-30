"""
Agent Token Saver - 近似重复检测

使用 SimHash 的轻量实现检测近似重复文件，
超越 MD5 精确匹配，能识别"同一模板生成的不同文件"。

策略：
  - 对每个文件提取 shingle（连续 k-gram 词条）
  - 计算加权 hash 得到 simhash 指纹
  - 汉明距离 <= threshold 的文件视为近似重复

适用场景：
  - 自动生成的代码（ORM model、API endpoint 模板）
  - 复制粘贴的代码片段
  - 重构前后的相似文件
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class NearDuplicateGroup:
    """近似重复文件组。"""
    representative: str  # 代表文件路径
    duplicates: list[str]  # 近似重复的文件路径
    similarity: float  # 平均相似度 (0-1)
    fingerprint: int  # simhash 指纹


def compute_simhash(content: str, k: int = 4) -> int:
    """计算文本的 SimHash 指纹。

    SimHash 算法：
      1. 提取 k-shingle（连续 k 个 token）
      2. 对每个 shingle 计算 hash
      3. 按 bit 位加权累加（hash 的每个 bit 决定加减）
      4. 取符号得到 64 位指纹

    Args:
        content: 文本内容
        k: shingle 长度（默认 4）

    Returns:
        64 位整数指纹
    """
    # 提取 token：字母数字序列 + 保留部分运算符
    tokens = _tokenize(content)
    if len(tokens) < k:
        # 内容太短，使用全部内容的 hash（转为整数）
        return int(hashlib.md5(content.encode()).hexdigest(), 16) >> 4

    # 生成 shingle
    shingles = set()
    for i in range(len(tokens) - k + 1):
        shingle = " ".join(tokens[i:i + k])
        shingles.add(shingle)

    if not shingles:
        return 0

    # SimHash 计算（64 bit）
    bit_counts = [0] * 64
    for shingle in shingles:
        shingle_hash = _hash_shingle(shingle)
        for i in range(64):
            if shingle_hash & (1 << i):
                bit_counts[i] += 1
            else:
                bit_counts[i] -= 1

    # 构建指纹
    fingerprint = 0
    for i in range(64):
        if bit_counts[i] > 0:
            fingerprint |= (1 << i)

    return fingerprint


def hamming_distance(hash1: int, hash2: int) -> int:
    """计算两个整数的汉明距离（不同的 bit 位数）。"""
    x = hash1 ^ hash2
    count = 0
    while x:
        count += 1
        x &= x - 1
    return count


def similarity_score(hash1: int, hash2: int, bits: int = 64) -> float:
    """计算两个 simhash 指纹的相似度 (0-1)。"""
    dist = hamming_distance(hash1, hash2)
    return max(0.0, 1.0 - dist / bits)


def find_near_duplicates(
    file_paths: list[str | Path],
    threshold: int = 3,  # 汉明距离阈值
) -> list[NearDuplicateGroup]:
    """查找近似重复文件组。

    Args:
        file_paths: 文件路径列表
        threshold: 汉明距离阈值（越小越严格，0=完全相同）

    Returns:
        NearDuplicateGroup 列表
    """
    files = [Path(p) for p in file_paths if Path(p).is_file()]
    if not files:
        return []

    # 计算每个文件的 simhash
    fingerprints: dict[int, list[tuple[str, int]]] = {}
    for fp in files:
        try:
            content = fp.read_text(encoding="utf-8", errors="replace")
            fp_hash = compute_simhash(content)
            fingerprints.setdefault(fp_hash, []).append((str(fp), len(content)))
        except Exception:
            continue

    # 按汉明距离分组
    hash_list = list(fingerprints.keys())
    visited = set()
    groups: list[NearDuplicateGroup] = []

    for i, h1 in enumerate(hash_list):
        if h1 in visited:
            continue

        group_files = []
        for h2 in hash_list:
            if h2 in visited:
                continue
            dist = hamming_distance(h1, h2)
            if dist <= threshold:
                visited.add(h2)
                group_files.extend(fingerprints[h2])

        if len(group_files) > 1:
            # 选择最小的文件作为代表（通常是最原始的版本）
            representative = min(group_files, key=lambda x: x[1])[0]
            duplicates = [f for f, _ in group_files if f != representative]
            avg_sim = sum(
                similarity_score(h1, h2)
                for h2 in hash_list if h2 in visited and h2 != h1
            ) / max(1, len(duplicates))

            groups.append(NearDuplicateGroup(
                representative=representative,
                duplicates=duplicates,
                similarity=round(avg_sim, 3),
                fingerprint=h1,
            ))

    return groups


def get_near_dup_suggestions(groups: list[NearDuplicateGroup]) -> list[str]:
    """将近似重复组转换为可读建议。"""
    suggestions = []
    for g in groups:
        pct = int(g.similarity * 100)
        suggestions.append(
            f"发现 {len(g.duplicates) + 1} 个近似重复文件（相似度 {pct}%），"
            f"代表: {g.representative}，"
            f"建议仅保留代表文件，其余 {len(g.duplicates)} 个可跳过。"
        )
    return suggestions


# ── 内部辅助 ────────────────────────────────────────────────────────────

def _tokenize(content: str) -> list[str]:
    """将文本分词为 token 列表。"""
    # 匹配：标识符、数字、字符串字面量、运算符
    pattern = r'[a-zA-Z_]\w*|\d+(?:\.\d+)?|"[^"]*"|\'[^\']*\'|[+\-*/%=<>!&|^~]+'
    return re.findall(pattern, content)


def _hash_shingle(shingle: str) -> int:
    """对 shingle 计算 64 bit hash。"""
    return int(hashlib.md5(shingle.encode()).hexdigest(), 16) >> 4  # 取高 64 bit
