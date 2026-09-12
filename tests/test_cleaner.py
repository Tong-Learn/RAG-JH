# -*- coding: utf-8 -*-
"""pytest：清洗（规范化 / 去噪 / page 透传）。"""
import pytest

from preprocessing.cleaner import clean_block, _normalize_ws, _is_noise
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
