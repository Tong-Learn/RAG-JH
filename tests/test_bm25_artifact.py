# -*- coding: utf-8 -*-
"""pytest：BM25 语料侧产物（构建→读回一致 / 缺失回退现场构建）。

覆盖：
  1) build_artifact -> save -> load 往返一致（ids / texts / meta / BM25 统计量与打分）；
  2) build_hybrid_search 优先加载产物：**切片目录为空也能构造**（证明不再读 data/chunks）；
  3) 产物缺失时回退：现场读切片产物 + 分词 + 建索引。

运行：.venv_rag311\\Scripts\\python.exe -m pytest tests/ -q
"""
import json
import shutil
from pathlib import Path
from uuid import uuid4

import pytest

ROOT = Path(__file__).resolve().parents[1]
WORK_DIR = ROOT / ".pytest_work"


@pytest.fixture
def workdir():
    d = WORK_DIR / uuid4().hex[:8]
    d.mkdir(parents=True, exist_ok=True)
    yield d
    shutil.rmtree(WORK_DIR, ignore_errors=True)


def _corpus():
    return [
        {"id": "chunk_000001", "text": "晶珀*100 维护补给，通过邮件发放。", "doc": "公告A",
         "section_path": ["公告A", "维护"], "start_page": 1, "end_page": 1},
        {"id": "chunk_000002", "text": "逐猎狂途为3v3娱乐竞技玩法，每周二开放。", "doc": "公告B",
         "section_path": ["公告B"], "start_page": 2, "end_page": 3},
        {"id": "chunk_000003", "text": "武器强化上限提升至+22，增幅进度5颗星。", "doc": "公告C",
         "section_path": [], "start_page": 0, "end_page": 0},
    ]


def test_build_save_load_roundtrip(workdir):
    from scripts.run_bm25_build import build_artifact, save_artifact
    from retrieval.hybrid import load_artifact, tokenize
    from rank_bm25 import BM25Okapi

    artifact = build_artifact(_corpus())
    path = save_artifact(artifact, workdir / "bm25_corpus.pkl")
    loaded = load_artifact(path)

    assert loaded["ids"] == ["chunk_000001", "chunk_000002", "chunk_000003"]
    assert loaded["texts"]["chunk_000002"].startswith("逐猎狂途")
    assert loaded["meta"]["chunk_000002"] == {"doc": "公告B", "section_path": "公告B",
                                              "start_page": 2, "end_page": 3}
    assert loaded["meta"]["chunk_000003"]["section_path"] == ""      # 无标题栈 -> 空串

    # 读回后的 BM25 与现场重建完全一致（统计量 + 打分）
    rebuilt = BM25Okapi([tokenize(c["text"]) for c in _corpus()])
    assert loaded["bm25"].doc_len == rebuilt.doc_len
    assert loaded["bm25"].avgdl == rebuilt.avgdl
    assert loaded["bm25"].idf == rebuilt.idf
    assert list(loaded["bm25"].get_scores(tokenize("逐猎狂途 3v3"))) == \
        list(rebuilt.get_scores(tokenize("逐猎狂途 3v3")))


def test_build_hybrid_search_prefers_artifact(workdir, monkeypatch):
    """产物存在时：即使切片目录为空，也能构造出可用的混合检索。"""
    import retrieval.hybrid as hy
    from scripts.run_bm25_build import build_artifact, save_artifact

    path = save_artifact(build_artifact(_corpus()), workdir / "bm25_corpus.pkl")
    monkeypatch.setattr(hy, "BM25_ARTIFACT", path)
    empty_dir = workdir / "no_chunks"                        # 故意不存在，证明不读 data/chunks

    hs = hy.build_hybrid_search(col=None, embedder=None, chunks_dir=empty_dir)
    assert hs._ids == ["chunk_000001", "chunk_000002", "chunk_000003"]
    assert hs._meta["chunk_000002"]["text"].startswith("逐猎狂途")     # text 来自产物
    assert hs._meta["chunk_000002"]["doc"] == "公告B"                 # 溯源字段来自产物
    assert hs._bm25_top("逐猎狂途", 3) == ["chunk_000002"]            # 查询期只分 query


def test_build_hybrid_search_falls_back_without_artifact(workdir, monkeypatch):
    """产物缺失时：回退到现场读切片产物 + 分词建索引。"""
    import retrieval.hybrid as hy
    monkeypatch.setattr(hy, "BM25_ARTIFACT", workdir / "not_built.pkl")

    chunks = workdir / "chunks"
    chunks.mkdir()
    with open(chunks / "公告A.chunks.jsonl", "w", encoding="utf-8") as f:
        for c in _corpus():
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    hs = hy.build_hybrid_search(col=None, embedder=None, chunks_dir=chunks)
    assert hs._ids == ["chunk_000001", "chunk_000002", "chunk_000003"]
    assert hs._bm25_top("逐猎狂途", 3) == ["chunk_000002"]
    assert hs._meta["chunk_000003"]["text"].startswith("武器强化")
