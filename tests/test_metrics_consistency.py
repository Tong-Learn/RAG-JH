# -*- coding: utf-8 -*-
"""pytest：`metrics.json` ↔ 代码常量/产物的**一致性守卫**。

目的：`metrics.json` 是「权威数字的单一来源」，但代码不读它（避免运行时耦合），
一致性完全靠手工维护 —— 极易漂移。本文件把它变成**可强制**的：
只要代码常量或产物与 metrics.json 不一致，测试立刻变红。

覆盖：
  1) 代码常量：评测 top-k、两条路线的拒答阈值、默认路线、各模型名、embedding 回退默认值；
  2) 评测集：题目数/可答数/不可答数（读 data/eval/*.json）；
  3) 语料产物：块数/heading/para、chunk 数（若 data/ 产物不存在则跳过，第三方环境不误报）。

运行：.venv_rag311\\Scripts\\python.exe -m pytest tests/ -q
"""
import inspect
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
METRICS = json.loads((ROOT / "metrics.json").read_text(encoding="utf-8"))


# ---------- 1. 代码常量 ----------

def test_qa_k_matches_metrics():
    from scripts._common import QA_K
    assert QA_K == METRICS["eval_set"]["k"], "评测 top-k 与 metrics.json 不一致"


def test_threshold_constants_match_metrics():
    from scripts.run_rag import MIN_SCORE_DEFAULT
    for route in ("vector_only", "hybrid_rerank"):
        assert MIN_SCORE_DEFAULT[route] == METRICS["threshold"][route]["chosen"], \
            f"{route} 拒答阈值与 metrics.json 不一致"


def test_default_route_and_models_match_metrics():
    from scripts.run_rag import DEFAULT_MODE, DEFAULT_CHAT_MODEL
    from retrieval.reranker import DEFAULT_RERANK_MODEL
    assert DEFAULT_MODE == "hybrid_rerank"
    assert DEFAULT_CHAT_MODEL == METRICS["models"]["chat"]
    assert DEFAULT_RERANK_MODEL == METRICS["models"]["rerank"]


def test_embedder_default_model_matches_metrics():
    """embedding 模型名有三处：.env、类内回退默认值、metrics.json —— 必须一致。

    回退默认值最容易漂移（.env 缺失时会静默用错模型），故单独守卫。
    """
    from embedding.dashscope_embedder import DashScopeEmbedder
    src = inspect.getsource(DashScopeEmbedder.__init__)
    import re
    m = re.search(r'_get_env\("DASHSCOPE_EMBEDDING_MODEL",\s*"([^"]+)"\)', src)
    assert m, "未能解析 DashScopeEmbedder 的模型回退默认值"
    assert m.group(1) == METRICS["models"]["embedding"], \
        "DashScopeEmbedder 回退默认模型与 metrics.json 不一致"
    # .env 存在时也要一致（本地环境）
    env = ROOT / ".env"
    if env.exists():
        import re as _re
        mm = _re.search(r"DASHSCOPE_EMBEDDING_MODEL=(\S+)", env.read_text(encoding="utf-8"))
        if mm:
            assert mm.group(1) == METRICS["models"]["embedding"], ".env 的模型名与 metrics.json 不一致"


# ---------- 2. 评测集 ----------

@pytest.mark.parametrize("setname,fname", [("simple", "simple_qa_30.json"), ("complex", "complex_qa_20.json")])
def test_eval_set_counts_match_metrics(setname, fname):
    p = ROOT / "data" / "eval" / fname
    if not p.exists():
        pytest.skip(f"{p} 不存在")
    data = json.loads(p.read_text(encoding="utf-8"))
    spec = METRICS["eval_set"][setname]
    assert len(data) == spec["n"]
    assert sum(1 for q in data if q["answerable"]) == spec["answerable"]
    assert sum(1 for q in data if not q["answerable"]) == spec["unanswerable"]
    total = METRICS["eval_set"]["simple"]["n"] + METRICS["eval_set"]["complex"]["n"]
    assert total == METRICS["eval_set"]["total_questions"]


# ---------- 3. 语料产物（无产物则跳过，便于第三方环境） ----------

def test_corpus_counts_match_metrics():
    man = ROOT / "data" / "processed" / "manifest.jsonl"
    if not man.exists():
        pytest.skip("data/processed 产物不存在（未跑数据准备）")
    recs = [json.loads(l) for l in man.read_text(encoding="utf-8").splitlines() if l.strip()]
    blk = sum(sum(r.get("blocks", {}).values()) for r in recs)
    head = sum(r.get("blocks", {}).get("heading", 0) for r in recs)
    para = sum(r.get("blocks", {}).get("para", 0) for r in recs)
    c = METRICS["corpus"]
    assert len(recs) == c["source_count"]
    assert blk == c["block_count"]
    assert head == c["heading_blocks"]
    assert para == c["para_blocks"]


def test_chunk_count_matches_metrics():
    cdir = ROOT / "data" / "chunks"
    if not cdir.exists():
        pytest.skip("data/chunks 产物不存在（未跑切片）")
    n = sum(1 for f in cdir.glob("*.chunks.jsonl")
            for l in f.read_text(encoding="utf-8").splitlines() if l.strip())
    assert n == METRICS["corpus"]["chunk_count"]
