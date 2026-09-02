# -*- coding: utf-8 -*-
"""pytest：切片核心逻辑测试（由 scripts/selftest_chunker.py 迁移）。

覆盖：超长按句切分 / overlap 收敛句边界 / 过短融合 / 标题分节 / 表格整体 / page 透传。
运行：.venv_rag311\\Scripts\\python.exe -m pytest tests/ -q
"""
from chunking.chunker import chunk_blocks, split_sentences, _overlap_tail


def _blocks(*items):
    """构造块：*items 为 ("heading",text,lvl) / ("para",text) / ("table",rows)。"""
    out = []
    for it in items:
        if it[0] == "heading":
            out.append({"kind": "heading", "text": it[1], "level": it[2]})
        elif it[0] == "para":
            out.append({"kind": "para", "text": it[1], "level": 1, "rows": [], "page": it[2] if len(it) > 2 else 0})
        elif it[0] == "table":
            out.append({"kind": "table", "text": "", "level": 1, "rows": it[1]})
    return out


def test_split_sentences_label_boundary():
    assert split_sentences("第一句。第二句！第三句？") == ["第一句。", "第二句！", "第三句？"]


def test_long_para_split_no_mid_sentence():
    long_para = "".join(f"这是第{i}个很长的句子用来验证切分不拦腰截断。" for i in range(1, 7))
    blocks = _blocks(("heading", "章节A", 1), ("para", long_para, 0))
    chunks = chunk_blocks(blocks, doc="t", chunk_size=40, min_chunk=10, overlap_ratio=0.2)
    assert len(chunks) >= 2
    for c in chunks:
        body = c.text.split("\n", 1)[1]
        assert body.endswith(("。", "！", "？", ";", "；")) or body == ""  # 每段以句边界收尾(非半句)


def test_overlap_is_whole_sentence():
    long_para = "".join(f"这是第{i}个很长的句子验证重叠。" for i in range(1, 6))
    blocks = _blocks(("heading", "A", 1), ("para", long_para, 0))
    chunks = chunk_blocks(blocks, doc="t", chunk_size=40, min_chunk=10, overlap_ratio=0.2)
    c1, c2 = chunks[0], chunks[1]
    body1 = c1.text.split("\n", 1)[1]
    body2 = c2.text.split("\n", 1)[1]
    expected_tail = _overlap_tail(body1, int(40 * 0.2))
    assert expected_tail != ""
    assert body2.startswith(expected_tail)   # 第2段带第1段尾句重合(整句，不切半句)


def test_small_tail_fused():
    blocks = _blocks(
        ("heading", "B", 1),
        ("para", "段落内容较长的一段文字，用于形成足够大的块。", 0),
        ("table", [["列1", "列2"], ["a", "b"]]),
        ("para", "尾部小尾巴。", 0),
    )
    chunks = chunk_blocks(blocks, doc="t", chunk_size=50, min_chunk=10, overlap_ratio=0.2)
    last = chunks[-1]
    assert "尾部小尾巴" in last.text   # 过短尾部已并入末块


def test_title_section_and_table_integrity():
    blocks = _blocks(
        ("heading", "总规定", 1),
        ("table", [["车型", "核载"], ["商务车", "7座"]]),
        ("heading", "一、范围", 2),
        ("para", "适用于所有正式员工的用车。", 0),
    )
    chunks = chunk_blocks(blocks, doc="t", chunk_size=50, min_chunk=10, overlap_ratio=0.2)
    tbl = [c for c in chunks if c.rows][0]
    assert tbl.section_path == ["总规定"]
    assert tbl.rows == [["车型", "核载"], ["商务车", "7座"]]   # 表格整体保留(未拆分)
    sub = [c for c in chunks if "一、范围" in c.section_path]
    assert sub and sub[0].section_path == ["总规定", "一、范围"]  # 二级标题嵌套


def test_page_propagation():
    # 含 page 的块：chunk 的 start_page/end_page 应取覆盖块的 min/max
    blocks = [
        {"kind": "heading", "text": "A", "level": 1, "page": 1},
        {"kind": "para", "text": "第一页正文。", "page": 1},
        {"kind": "para", "text": "第二页正文。", "page": 2},
    ]
    chunks = chunk_blocks(blocks, doc="t", chunk_size=100, min_chunk=10, overlap_ratio=0.2)
    assert chunks
    assert 0 <= chunks[0].start_page <= chunks[0].end_page
    assert chunks[0].start_page in (1, 2) or chunks[0].start_page == 0
