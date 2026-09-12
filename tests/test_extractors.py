# -*- coding: utf-8 -*-
"""pytest：PDF 提取核心逻辑测试（extractors 模块）。

覆盖：
  1) 视觉行聚类 `_cluster_visual_row`（「常规服/赛季服」被拆成散落字形的复原，及水印/整词/超长不误并）；
  2) 标题判定 `_heading_level`（大字号带标点仍是标题、小字号带标点不是、加粗短行、超长）；
  3) 段落合并 `_merge_para_lines` 与行拼接 `_join_para_line_parts`；
  4) 假表格判定 `_is_fake_table`；
  5) 用仓库内样例 PDF 的集成回归（标题不再被割裂）。

运行：.venv_rag311\\Scripts\\python.exe -m pytest tests/ -q
"""
from pathlib import Path

from preprocessing.extractors import (
    extract_pdf, _cluster_visual_row, _heading_level, _merge_para_lines,
    _join_para_line_parts, _is_fake_table,
)
from preprocessing.cleaner import clean_blocks

ROOT = Path(__file__).resolve().parents[1]
BODY = 9.08  # 正文字号（与真实公告一致）


def _ln(text, bbox, size=9.73, bold=True):
    """构造一条 line info（extractors 内部结构）。"""
    return {"text": text, "size": size, "bold": bold, "bbox": bbox}


# ---------- 1. 视觉行聚类 ----------

def test_cluster_glyphs_restores_split_title():
    # 「常规服」被排成 规/常/服 三个独立字形：规 baseline 抬高 3.9pt、x 相邻不重叠
    lines = [
        _ln("规", (278.89, 263.90, 288.62, 276.27)),
        _ln("常", (256.93, 267.80, 266.66, 280.16)),
        _ln("服", (300.85, 267.80, 310.59, 280.16)),
    ]
    merged = _cluster_visual_row(lines, BODY)
    assert len(merged) == 1                      # 三个字形并成一条
    assert merged[0]["text"] == "常规服"          # 且按 x0 重排（不是被抬高者排最前）


def test_cluster_keeps_x_overlapping_watermark_separate():
    # 装饰水印：两条《晶核》x 区间高度重叠，属重复非阅读序列 -> 不合并
    lines = [
        _ln("《晶核》", (247.83, 280.12, 297.15, 293.30), size=10.38),
        _ln("《晶核》", (248.48, 280.12, 297.80, 293.30), size=10.38),
    ]
    merged = _cluster_visual_row(lines, BODY)
    assert len(merged) == 2


def test_cluster_does_not_weld_multi_char_words():
    # 整词（>2 字）不参与聚类，即使 x 不重叠
    lines = [
        _ln("《晶核》", (247.83, 280.12, 297.15, 293.30), size=10.38),
        _ln("三周年", (307.53, 280.12, 343.87, 293.30), size=10.38),
    ]
    merged = _cluster_visual_row(lines, BODY)
    assert len(merged) == 2


def test_cluster_does_not_weld_overlong_result():
    # 11 个两字字形可拼成 22 字 > VISUAL_ROW_MAX_TITLE_LEN(20) -> 不焊接，原样返回
    lines = [_ln("测试", (100 + i * 12, 263.90, 110 + i * 12, 276.27)) for i in range(11)]
    merged = _cluster_visual_row(lines, BODY)
    assert len(merged) == 11


def test_cluster_ignores_non_heading_lines():
    # 两行正文（非标题候选）即使同基线也不合并
    lines = [
        _ln("这是一段较长的正文内容，用于确保长度超过五十个字符从而不会被判定为标题行。" * 2,
            (100, 200, 400, 212), size=BODY, bold=False),
        _ln("另一段正文", (410, 200, 460, 212), size=BODY, bold=False),
    ]
    merged = _cluster_visual_row(lines, BODY)
    assert len(merged) == 2


# ---------- 2. 标题判定 ----------

def test_heading_level_large_font_with_punct_is_heading():
    # 大字号(>=1.35x)即使以「！」结尾也判 level 1（标点判断在其后）
    assert _heading_level(_ln("新版本上线！", (0, 0, 100, 20), size=14.28), BODY) == 1


def test_heading_level_small_font_with_punct_is_body():
    # 正文字号 + 句末标点 -> 非标题
    assert _heading_level(_ln("这是正文短句。", (0, 0, 100, 12), size=BODY, bold=False), BODY) == 0


def test_heading_level_bold_short_is_level2():
    assert _heading_level(_ln("维护更新内容", (0, 0, 100, 12), size=BODY, bold=True), BODY) == 2


def test_heading_level_overlong_is_zero():
    assert _heading_level(_ln("长" * 51, (0, 0, 100, 20), size=14.28), BODY) == 0


# ---------- 3. 段落合并 / 行拼接 ----------

def test_join_para_line_parts_cjk_no_space_ascii_space():
    assert _join_para_line_parts(["你好", "世界"]) == "你好世界"
    assert _join_para_line_parts(["ABC", "def"]) == "ABC def"
    assert _join_para_line_parts(["你好", "abc"]) == "你好abc"   # 前字为 CJK -> 不插空格


def test_merge_para_lines_splits_by_y_gap():
    lines = [
        {"text": "第一行", "bbox": (0, 100, 100, 110)},
        {"text": "第二行", "bbox": (0, 120, 100, 130)},   # 间距 20 <= 20*1.6
        {"text": "隔段行", "bbox": (0, 200, 100, 210)},   # 间距 80 > 阈值 -> 分段
    ]
    assert _merge_para_lines(lines, line_h=20) == ["第一行第二行", "隔段行"]


# ---------- 4. 假表格 ----------

def test_is_fake_table():
    assert _is_fake_table([["a"]]) is True                 # 单行
    assert _is_fake_table([["标题", ""]]) is True           # 单列
    assert _is_fake_table([["车型", "核载"], ["商务车", "7座"]]) is False


# ---------- 5. 样例 PDF 集成回归（标题不再割裂） ----------

def _example_pdf(keyword):
    hits = sorted((ROOT / "examples" / "pdf").glob(f"*{keyword}*.pdf"))
    assert hits, f"样例 PDF 缺失：examples/pdf/*{keyword}*.pdf"
    return hits[0]


def test_example_pdf_titles_are_complete():
    blocks = clean_blocks(extract_pdf(_example_pdf("9月2日")))
    heads = [b.text for b in blocks if b.kind == "heading"]
    assert "常规服" in heads and "赛季服" in heads          # 完整标题
    assert not any(h in ("规", "常服", "季", "赛服") for h in heads)  # 无割裂残留
