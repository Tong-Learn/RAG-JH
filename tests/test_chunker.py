# -*- coding: utf-8 -*-
"""pytest：切片核心逻辑测试。

覆盖：超长按句切分 / overlap 收敛句边界 / 过短融合 / 标题分节 / 表格整体 / page 透传。
运行：.venv_rag311\\Scripts\\python.exe -m pytest tests/ -q
"""
from chunking.chunker import chunk_blocks, split_sentences


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
    # 硬编码期望尾串（**不用被测内部函数 _overlap_tail 反算**，避免半自证）：
    # 第 2 段的 overlap 应是第 1 段的**整句**尾句，而不是按字符数切出的半句。
    expected_tail = "这是第2个很长的句子验证重叠。"
    assert body1.endswith(expected_tail)
    assert body2.startswith(expected_tail)


def test_small_tail_fused():
    blocks = _blocks(
        ("heading", "B", 1),
        ("para", "段落内容较长的一段文字，用于形成足够大的块。", 0),
        ("table", [["列1", "列2"], ["a", "b"]]),
        ("para", "尾部小尾巴。", 0),
    )
    chunks = chunk_blocks(blocks, doc="t", chunk_size=50, min_chunk=10, overlap_ratio=0.2)
    # 过短尾部必须**并入前一块**（共 2 块）；若未融合会各自成块而变成 3 块。
    assert len(chunks) == 2
    tbl = [c for c in chunks if c.rows][0]
    assert "尾部小尾巴" in tbl.text          # 并入的是表格块（不是自成一块）


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
    # 跨页 chunk：start_page / end_page 取覆盖块的 min / max（这里是第 1、2 页两块）
    blocks = [
        {"kind": "heading", "text": "A", "level": 1, "page": 1},
        {"kind": "para", "text": "第一页正文。", "page": 1},
        {"kind": "para", "text": "第二页正文。", "page": 2},
    ]
    chunks = chunk_blocks(blocks, doc="t", chunk_size=100, min_chunk=10, overlap_ratio=0.2)
    assert len(chunks) == 1
    assert (chunks[0].start_page, chunks[0].end_page) == (1, 2)
