# -*- coding: utf-8 -*-
"""pytest：清洗（规范化 / 去噪 / page 透传 / 尾部署名之后的推荐噪声截断）。"""
import pytest

from preprocessing.cleaner import clean_block, clean_blocks, _normalize_ws, _is_noise
from preprocessing.extractors import Block


def test_normalize_ws_cjk_space():
    # 数字后空格去除（"2026 年"→"2026年"）；汉字后空格保留（"年 4月"）
    assert _normalize_ws("2026 年 4 月 15 日") == "2026年 4月 15日"
    assert _normalize_ws("晶珀*100 ，将通过") == "晶珀*100，将通过"
    assert _normalize_ws("  多  空格  ") == "多 空格"


def test_meta_line_is_noise():
    assert _is_noise("2026年4月13日 18:32上海晶核CoA听全文")
    assert not _is_noise("这是一段较长的正文内容，用于验证不会误判。")


@pytest.mark.parametrize("text,expected", [
    ("第 3 页 / 共 20 页", None),
    ("45", None),
    ("正常段落文字。", "para"),
])
def test_clean_block_drop_or_keep(text, expected):
    b = Block("para", text=text)
    r = clean_block(b)
    if expected is None:
        assert r is None
    else:
        assert r is not None and r.kind == expected


def test_clean_block_preserves_page():
    b = Block("para", text="正文。", page=2)
    r = clean_block(b)
    assert r is not None and r.page == 2


# ---------- 尾部噪声截断（署名之后是往期推荐 / PV 预告） ----------

def _texts(blocks):
    return [b.text for b in blocks]


def test_tail_truncation_drops_blocks_after_signature():
    """标记之后的推荐块整段丢弃，标记及之前的内容保留。"""
    blocks = [
        Block("heading", text="问题修复"),
        Block("para", text="1、修复了某个问题。"),
        Block("para", text="冒险者协会"),
        Block("para", text="往期推荐丨全新时装实机展示"),
        Block("heading", text="更多官方信息"),
    ]
    out = clean_blocks(blocks)
    assert _texts(out) == ["问题修复", "1、修复了某个问题。", "冒险者协会"]


def test_tail_truncation_accepts_fused_signature():
    """署名被拼进正文长句（块以标记结尾但不等于标记）时同样触发。"""
    blocks = [
        Block("para", text="1、修复了某个问题。冒险者协会"),
        Block("para", text="往期推荐丨全新载具实机演示"),
    ]
    out = clean_blocks(blocks)
    assert _texts(out) == ["1、修复了某个问题。冒险者协会"]


def test_tail_truncation_ignores_mid_text_mention():
    """反例：正文里**提到**这五个字（不在块尾）不算标记，不触发截断。"""
    blocks = [
        Block("para", text="冒险者协会将在本周维护后开放新玩法。"),
        Block("para", text="后续更新内容以公告为准。"),
    ]
    out = clean_blocks(blocks)
    assert _texts(out) == ["冒险者协会将在本周维护后开放新玩法。", "后续更新内容以公告为准。"]


def test_tail_truncation_uses_last_occurrence():
    """反例：前面出现过署名（非最后一次）时，截断点是**最后一次**，中间正文保留。"""
    blocks = [
        Block("para", text="冒险者协会"),
        Block("para", text="这段是署名之后的正文，必须保留。"),
        Block("para", text="冒险者协会"),
        Block("para", text="往期推荐丨PV 预告"),
    ]
    out = clean_blocks(blocks)
    assert _texts(out) == ["冒险者协会", "这段是署名之后的正文，必须保留。", "冒险者协会"]


def test_tail_truncation_no_marker_keeps_all():
    """无标记：原样返回（不截断）。"""
    blocks = [Block("para", text="第一段。"), Block("para", text="第二段。")]
    out = clean_blocks(blocks)
    assert _texts(out) == ["第一段。", "第二段。"]


def test_tail_truncation_marker_is_last_block_keeps_all():
    """标记即最后一块：其后没有内容可丢，结果与输入一致。"""
    blocks = [Block("para", text="正文内容。"), Block("para", text="冒险者协会")]
    out = clean_blocks(blocks)
    assert _texts(out) == ["正文内容。", "冒险者协会"]

