# -*- coding: utf-8 -*-
"""pytest：检索侧核心链路测试（原为"三条核心链路零测试"的一部分）。

覆盖：
  1) RRF 融合（retrieval/hybrid.rrf_fuse）：融合后按 RRF 分排序、保留正确顺序；
  2) 阈值拒答（scripts.run_rag / gen_answers 的 min-score 分支）：低于阈值拒答、不调生成；
  3) 生成带引用（scripts.run_rag.build_context）：把 top-k 片段编号，附来源附录；
  4) 重排 top_n 越界（retrieval/reranker.DashScopeReranker 的 min(len(docs), top_n) 修复）——用假响应验证。

运行：.venv_rag311\\Scripts\\python.exe -m pytest tests/ -q
"""
from retrieval.hybrid import rrf_fuse


# ---------- 1. RRF 融合 ----------

def test_rrf_fuse_merges_and_orders():
    # 两个列表都命中的 pid 应排最前；两列表 RRF 分之和最高
    ranked = rrf_fuse([["a", "b", "c"], ["b", "a", "d"]], k=60)
    ids = [pid for pid, _ in ranked]
    # "a""b" 在两列表都有 -> RRF 分高于只单列表出现的 "c""d"
    assert ids[0] in ("a", "b")
    assert ids[1] in ("a", "b")
    assert ids[-1] in ("c", "d")          # 只在一侧出现，RRF 分更低
    # 分数降序
    scores = [s for _, s in ranked]
    assert all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))


def test_rrf_fuse_empty_and_single():
    assert rrf_fuse([]) == []
    assert rrf_fuse([["x", "y"]])[0][0] == "x"   # 单列表也返回 (id, score)
    assert rrf_fuse([[]]) == []


def test_rrf_fuse_rank_denominator_sanity():
    # 同一 pid 只在一侧第 1 名：RRF = 1/(k+1+1) + 1/(k+2+1)（若另一侧也有），此处它只有一侧 -> 单一贡献
    fused = rrf_fuse([["only"], []], k=60)
    assert fused == [("only", 1.0 / 61)]


# ---------- 2. 阈值拒答 ----------

def test_threshold_reject_does_not_call_generation():
    from scripts.gen_answers import gen_answer
    from scripts.run_rag import REJECT_ANSWER
    # 构造 hits top-1 低于阈值；gen_answer 的拒答分支不调 chat，故无需 mock 也不会联网。
    hits = [{"id": "chunk_000001", "score": 0.50, "text": "片段", "meta": {"doc": "D", "section_path": "S", "start_page": 1}}]
    res = gen_answer("问题", hits, model="m", min_score=0.70)
    assert res["rejected"] is True
    assert res["answer"] == REJECT_ANSWER          # 拒答时给出固定文案（不再是空串）
    assert res["answer"] == "当前资料相关性不足，无法回答"
    assert res["top1"] == 0.50
    assert res["citations"] is None                # 拒答未生成，无需引用校验


def test_threshold_pass_calls_generation(monkeypatch):
    import scripts.gen_answers as ga
    # top-1 >= 阈值 -> 应调用生成；这里 monkeypatch chat 返回固定回答，验证走的是生成分支
    called = []
    def fake_chat(*a, **kw):
        called.append(True)
        return "虚构回答 [chunk_000001]"
    monkeypatch.setattr(ga, "chat", fake_chat)
    hits = [{"id": "chunk_000001", "score": 0.90, "text": "片段",
             "meta": {"doc": "D", "section_path": "S", "start_page": 1}}]
    res = ga.gen_answer("问题", hits, model="m", min_score=0.70)
    assert res["rejected"] is False
    assert called
    assert "[chunk_000001]" in res["answer"]


# ---------- 3. 生成带引用（build_context / build_messages / extract_citations） ----------

def test_build_context_uses_chunk_id_and_appendix():
    from scripts.run_rag import build_context
    hits = [
        {"id": "chunk_000023", "text": "第一段资料", "meta": {"doc": "公告A", "section_path": "维护更新内容", "start_page": 1, "end_page": 1}},
        {"id": "chunk_000024", "text": "第二段资料", "meta": {"doc": "公告B", "section_path": "问题修复", "start_page": 2, "end_page": 2}},
    ]
    ctx, appendix = build_context(hits)
    assert "[chunk_000023]" in ctx and "[chunk_000024]" in ctx   # 以 chunk id 为编号（非 [1][2] 位置号）
    assert "第一段资料" in ctx and "第二段资料" in ctx
    assert "引用来源" in appendix
    assert "公告A" in appendix and "公告B" in appendix
    assert "p.1~1" in appendix                       # 页码附录


def test_build_context_no_page_ok():
    from scripts.run_rag import build_context
    hits = [{"id": "chunk_000001", "text": "正文", "meta": {"doc": "D", "section_path": "", "start_page": None}}]
    ctx, appendix = build_context(hits)
    assert "正文" in ctx
    assert "p." not in appendix                       # 无页码时不显示页码


def test_build_messages_puts_rules_in_system_data_in_user():
    """规则集中在 system（单一来源），user 只承载【资料】+【问题】。"""
    from scripts.run_rag import build_messages, SYSTEM_PROMPT
    hits = [{"id": "chunk_000007", "text": "资料正文", "meta": {"doc": "D", "section_path": "S", "start_page": 1, "end_page": 1}}]
    msgs = build_messages("这是问题？", hits)
    assert msgs[0]["role"] == "system" and msgs[0]["content"] == SYSTEM_PROMPT
    assert msgs[1]["role"] == "user"
    assert "【资料】" in msgs[1]["content"] and "【问题】这是问题？" in msgs[1]["content"]
    # 规则只出现在 system：user 里不应再重复"依据资料作答/资料中没有相关信息"这类规则句
    assert "请仅依据这些资料作答" not in msgs[1]["content"]
    assert "资料中没有相关信息" not in msgs[1]["content"]
    assert "资料中没有相关信息" in SYSTEM_PROMPT


def test_extract_citations_valid_invalid_none():
    from scripts.run_rag import extract_citations
    hits = [{"id": "chunk_000001"}, {"id": "chunk_000002"}]
    # 全部有效
    v, iv, none = extract_citations("结论A[chunk_000001]，结论B[chunk_000002]", hits)
    assert v == ["chunk_000001", "chunk_000002"] and not iv and not none
    # 含越界/编造编号 -> 记入 invalid
    v, iv, none = extract_citations("结论[chunk_000001]，另一结论[chunk_999999]", hits)
    assert v == ["chunk_000001"] and iv == ["chunk_999999"]
    # 完全没标注
    v, iv, none = extract_citations("没有任何引用的回答", hits)
    assert v == [] and iv == [] and none is True


# ---------- 4. 重排 top_n 越界（reranker 修复） ----------

def test_reranker_top_n_no_overflow(monkeypatch):
    # 构造候选数 3 < top_n 默认 20；request 里 parameters.top_n 应等于 3（min(len(docs), self.top_n)）
    from retrieval.reranker import DashScopeReranker
    captured = {}
    def fake_post(url, headers, json, timeout):
        captured["body"] = json
        class R:
            status_code = 200
            def json(self):
                return {"output": {"results": [
                    {"index": 1, "relevance_score": 0.9},
                    {"index": 0, "relevance_score": 0.8},
                    {"index": 2, "relevance_score": 0.7},
                ]}}
        return R()
    monkeypatch.setattr("retrieval.reranker.requests.post", fake_post)
    rr = DashScopeReranker(top_n=20)
    cands = [{"id": "a", "text": "t1", "meta": {}}, {"id": "b", "text": "t2", "meta": {}}, {"id": "c", "text": "t3", "meta": {}}]
    out = rr.rerank("q", cands, k=2)
    assert captured["body"]["parameters"]["top_n"] == 3   # 越界修复：top_n=候选数
    assert len(out) == 2
    assert out[0]["id"] == "b"                            # 按 relevance 降序
    assert out[0]["score"] == 0.9


# ---------- 5. 多来源检索指标（scripts/eval_metrics） ----------

def test_metrics_single_source_degenerates():
    from scripts.eval_metrics import score_question
    # 单来源：命中=1，未命中=0（退化为原 recall 定义）
    s1 = score_question(["D1", "X", "Y"], ["D1"], k=3)
    assert s1["recall"] == 1.0 and s1["p1"] == 1.0 and s1["mrr"] == 1.0
    s2 = score_question(["X", "D1", "Y"], ["D1"], k=3)
    assert s2["recall"] == 1.0 and s2["p1"] == 0.0 and s2["mrr"] == 0.5


def test_metrics_multi_source_coverage():
    from scripts.eval_metrics import score_question
    # 4 个来源，top-5 命中 3 个 -> recall=0.75, precision=3/5
    s = score_question(["D1", "D2", "X", "D3", "Y"], ["D1", "D2", "D3", "D4"], k=5)
    assert abs(s["recall"] - 0.75) < 1e-9
    assert abs(s["precision"] - 0.6) < 1e-9
    assert s["p1"] == 1.0
    # 同一来源的多个 chunk 只算一次命中
    s2 = score_question(["D1", "D1", "D1"], ["D1", "D2"], k=3)
    assert s2["recall"] == 0.5


def test_metrics_unanswerable_returns_none():
    from scripts.eval_metrics import score_question
    assert score_question(["X"], [], k=3) is None


def test_metrics_aggregate_and_hit_rates():
    from scripts.eval_metrics import score_question, aggregate, any_hit_rate, all_hit_rate
    a = score_question(["D1", "D2"], ["D1", "D2"], k=2)      # 全命中
    b = score_question(["X", "D1"], ["D1", "D2"], k=2)        # 半命中
    agg = aggregate([a, b], ["recall", "precision"])
    assert abs(agg["recall"] - 0.75) < 1e-9                   # (1.0 + 0.5)/2
    assert any_hit_rate([a, b]) == 1.0
    assert all_hit_rate([a, b]) == 0.5


# ---------- 6. 生成调用重试（防瞬态网络抖动污染产物） ----------

def _patch_chat_env(monkeypatch):
    import requests
    import scripts.run_rag as rr
    monkeypatch.setattr(rr, "_get_env", lambda name, default="": "sk-test")
    monkeypatch.setattr(rr.time, "sleep", lambda s: None)      # 不真等
    return rr, requests


def test_chat_retries_on_transient_error(monkeypatch):
    rr, requests = _patch_chat_env(monkeypatch)
    calls = {"n": 0}

    def fake_post(url, headers, json, timeout):
        calls["n"] += 1
        if calls["n"] == 1:
            raise requests.exceptions.ReadTimeout("boom")     # 首次超时
        class R:
            status_code = 200
            def json(self):
                return {"choices": [{"message": {"content": "ok"}}]}
        return R()

    monkeypatch.setattr(rr.requests, "post", fake_post)
    assert rr.chat([{"role": "user", "content": "hi"}]) == "ok"
    assert calls["n"] == 2                                     # 重试后成功


def test_chat_gives_up_after_retries(monkeypatch):
    import pytest
    rr, requests = _patch_chat_env(monkeypatch)
    calls = {"n": 0}

    def always_timeout(url, headers, json, timeout):
        calls["n"] += 1
        raise requests.exceptions.ReadTimeout("still down")

    monkeypatch.setattr(rr.requests, "post", always_timeout)
    with pytest.raises(RuntimeError, match="生成失败"):
        rr.chat([{"role": "user", "content": "hi"}], max_retries=2)
    assert calls["n"] == 3                                     # 1 次 + 2 次重试后放弃


def test_chat_does_not_retry_on_4xx(monkeypatch):
    import pytest
    rr, _ = _patch_chat_env(monkeypatch)
    calls = {"n": 0}

    def bad_request(url, headers, json, timeout):
        calls["n"] += 1
        class R:
            status_code = 400
            text = "bad request"
        return R()

    monkeypatch.setattr(rr.requests, "post", bad_request)
    with pytest.raises(RuntimeError):
        rr.chat([{"role": "user", "content": "hi"}])
    assert calls["n"] == 1                                     # 4xx 不重试


# ---------- 7. 运行前提检查（防"新克隆直接跑"抛底层异常） ----------

def test_hybrid_search_empty_corpus_raises_clear_error():
    """空语料不应抛 ZeroDivisionError，而应给出可执行的指引。"""
    import pytest
    from retrieval.hybrid import HybridSearch
    with pytest.raises(RuntimeError, match="语料为空"):
        HybridSearch(col=None, embedder=None, corpus=[])


def test_preflight_reports_missing_artifacts(monkeypatch):
    import scripts._common as c
    ghost = c.PROJECT_ROOT / "__no_such_dir__"          # 不存在的路径（无需真实创建）
    monkeypatch.setattr(c, "CHROMA_DIR", ghost / "chroma")
    monkeypatch.setattr(c, "CHUNKS_DIR", ghost / "chunks")
    problems = c.preflight("hybrid_rerank")
    assert any("向量库" in p for p in problems)
    assert any("切片产物" in p for p in problems)
    assert c.print_preflight(problems) is False               # 不就绪


def test_preflight_vector_mode_does_not_require_chunks(monkeypatch):
    import scripts._common as c
    ghost = c.PROJECT_ROOT / "__no_such_dir__"
    monkeypatch.setattr(c, "CHROMA_DIR", ghost / "chroma")
    monkeypatch.setattr(c, "CHUNKS_DIR", ghost / "chunks")
    problems = c.preflight("vector_only")
    assert not any("切片" in p for p in problems)              # 纯向量不需要 BM25 语料


# ---------- 8. CLI 输出路径（交互式入口全流程，全部依赖被替换，不联网） ----------

def test_run_rag_main_prints_answer_appendix_and_citations(monkeypatch, capsys):
    """main() 走完「检索 → 生成 → 打印来源附录 → 引用校验」。

    回归点：main() 曾引用未定义的 appendix 变量，导致答案都生成完了却在最后一行 NameError。
    """
    import sys
    import scripts.run_rag as r

    hits = [{"id": "chunk_000001", "score": 0.90, "text": "装甲矩阵活动",
             "meta": {"doc": "D", "section_path": "D / 活动", "start_page": 1, "end_page": 1}}]
    monkeypatch.setattr(r, "preflight", lambda mode: [])
    monkeypatch.setattr(r, "print_preflight", lambda problems: True)
    monkeypatch.setattr(r, "DashScopeEmbedder", lambda *a, **kw: object())
    monkeypatch.setattr(r, "get_collection", lambda: object())
    monkeypatch.setattr(r, "make_search", lambda *a, **kw: (lambda q, k: hits))
    monkeypatch.setattr(r, "chat", lambda messages, **kw: "答案 [chunk_000001]")
    monkeypatch.setattr(sys, "argv", ["run_rag.py", "问题？"])

    r.main()

    out = capsys.readouterr().out
    assert "生成回答" in out and "答案 [chunk_000001]" in out
    assert "引用来源: [chunk_000001]" in out        # 来源附录（含页码）被打印
    assert "引用校验: 有效 1 处" in out


def test_run_rag_main_rejects_below_threshold_without_generation(monkeypatch, capsys):
    """低于阈值：打印拒答文案，且不调用生成。"""
    import sys
    import scripts.run_rag as r

    hits = [{"id": "chunk_000001", "score": 0.10, "text": "无关片段",
             "meta": {"doc": "D", "section_path": "D / 其他", "start_page": 1}}]
    monkeypatch.setattr(r, "preflight", lambda mode: [])
    monkeypatch.setattr(r, "print_preflight", lambda problems: True)
    monkeypatch.setattr(r, "DashScopeEmbedder", lambda *a, **kw: object())
    monkeypatch.setattr(r, "get_collection", lambda: object())
    monkeypatch.setattr(r, "make_search", lambda *a, **kw: (lambda q, k: hits))
    monkeypatch.setattr(r, "chat", lambda *a, **kw: (_ for _ in ()).throw(AssertionError("不应调用生成")))
    monkeypatch.setattr(sys, "argv", ["run_rag.py", "问题？"])

    r.main()

    out = capsys.readouterr().out
    assert r.REJECT_ANSWER in out
    assert "未调用生成" in out

