# -*- coding: utf-8 -*-
"""pytest：端到端 smoke（离线，不调任何 API）。

用仓库内样例 PDF 走「提取 → 清洗 → 切片」，验证主要流程能串通且产物自洽：
  - 能提取出块、chunk 非空；
  - chunk id 连续且唯一（从 chunk_000001 起，0 不使用）；
  - 每个 chunk 有 text、有 section_path；
  - 散落字形的标题（常规服/赛季服）能一路传导到 section_path。

运行：.venv_rag311\\Scripts\\python.exe -m pytest tests/ -q
"""
import dataclasses
from pathlib import Path

from preprocessing.extractors import extract_pdf
from preprocessing.cleaner import clean_blocks
from chunking.chunker import chunk_blocks

ROOT = Path(__file__).resolve().parents[1]


def _example_pdf(keyword):
    hits = sorted((ROOT / "examples" / "pdf").glob(f"*{keyword}*.pdf"))
    assert hits, f"样例 PDF 缺失：examples/pdf/*{keyword}*.pdf"
    return hits[0]


def _chunks(pdf_path, doc="smoke"):
    blocks = clean_blocks(extract_pdf(pdf_path))
    dicts = [dataclasses.asdict(b) for b in blocks]
    return blocks, chunk_blocks(dicts, doc=doc, chunk_size=300, min_chunk=100,
                                overlap_ratio=0.15, id_start=0)


def test_e2e_extract_to_chunks_smoke():
    blocks, chunks = _chunks(_example_pdf("9月2日"))

    assert blocks, "未提取到任何块"
    assert chunks, "未切出任何 chunk"

    # id 连续唯一。注意：chunker 的 chunk_counter 是「先自增再用」，
    # 故首个 id = id_start + 1 —— 全局 id 从 chunk_000001 开始，0 不使用。
    ids = [c.id for c in chunks]
    assert ids[0] == "chunk_000001"
    assert len(ids) == len(set(ids))
    assert ids == [f"chunk_{i:06d}" for i in range(1, len(ids) + 1)]

    # 每个 chunk 自洽
    for c in chunks:
        assert c.text.strip()
        assert isinstance(c.section_path, list) and c.section_path
        assert c.start_page <= c.end_page or c.end_page == 0


def test_e2e_fixed_title_propagates_to_section_path():
    _, chunks = _chunks(_example_pdf("9月2日"))
    paths = [" / ".join(c.section_path) for c in chunks]
    assert any("常规服" in p for p in paths)
    assert any("赛季服" in p for p in paths)
    assert not any(p.endswith("常服") or p.endswith("赛服") for p in paths)
