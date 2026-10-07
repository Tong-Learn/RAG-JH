# -*- coding: utf-8 -*-
"""pytest：落盘记录的**字段集合**守卫（防字段悄悄增删导致下游/文档漂移）。

覆盖：
  1) run_vectorize.chunk_to_record() 的 chroma metadata 键集合**精确等于**期望；
  2) gen_answers.build_topk() 的 topk 元素**不含 score**（分数只留在检索快照里）。

运行：.venv_rag311\\Scripts\\python.exe -m pytest tests/ -q
"""


def test_chunk_to_record_meta_keys_exact():
    from scripts.run_vectorize import chunk_to_record
    c = {"id": "chunk_000123", "text": "片段正文", "doc": "某公告",
         "section_path": ["某公告", "问题修复"], "start_page": 2, "end_page": 3}
    cid, document, meta = chunk_to_record(c)
    assert cid == "chunk_000123"
    assert document == "片段正文"                     # document 就是 chunk.text，供展示/拼上下文
    assert meta == {"doc": "某公告", "section_path": "某公告 / 问题修复",
                    "start_page": 2, "end_page": 3}
    assert set(meta) == {"doc", "section_path", "start_page", "end_page"}


def test_build_topk_has_no_score():
    from scripts.gen_answers import build_topk
    hits = [{"id": "chunk_000001", "score": 0.912, "text": "正文",
             "meta": {"doc": "D", "section_path": "D / 活动", "start_page": 1, "end_page": 1}},
            {"id": "chunk_000002", "score": 0.5, "text": "正文2",
             "meta": {"doc": "D", "section_path": "", "start_page": 0, "end_page": 0}}]
    topk = build_topk(hits)
    assert len(topk) == 2
    for item in topk:
        assert set(item) == {"id", "doc", "section", "page"}   # 无 score
    assert topk[0] == {"id": "chunk_000001", "doc": "D", "section": "D / 活动", "page": [1, 1]}
    assert topk[1]["page"] is None                            # 无页码时不写页码
