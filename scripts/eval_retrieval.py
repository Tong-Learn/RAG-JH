# -*- coding: utf-8 -*-
"""
检索质量评估（流程四）：读取 data/eval/qa_basic.jsonl + qa_complex.jsonl，做 top-k 检索，
按「标准答案所在文档」计算 doc 级指标；并报告 top-1 相似度分布（用于阈值拒答校准）。

指标（对每个 set 分别输出）：
  - recall@k / MRR / P@1 / precision@k / nDCG@k
拒答相关：对 answerable 题统计 top-1 相似度（正类）；对 unanswerable/irrelevant 题统计（负类）。
两条检索路线：--mode vector_only（默认）| hybrid_rerank（BM25+向量 RRF 候选 -> 重排）。

运行：.venv_rag311\\Scripts\\python.exe -m scripts.eval_retrieval [-k N] [--mode vector_only|hybrid_rerank] [--sets all|basic|complex] [--min-score 0.70]
"""
import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from embedding.dashscope_embedder import DashScopeEmbedder
from scripts._common import get_collection, load_queries, PROJECT_ROOT
from retrieval.search import make_search


def aggregate(recs, k):
    if not recs:
        return None
    n = len(recs)
    recall = sum(1 for _, _, _, r in recs if r > 0) / n
    mrr = sum(1.0 / r for _, _, _, r in recs if r > 0) / n
    p1 = sum(1 for _, _, _, r in recs if r == 1) / n
    prec = 0.0
    for _, doc_ids, expected, _ in recs:
        prec += sum(1 for d in doc_ids if d == expected) / k
    prec /= n
    ndcg = sum((1.0 / math.log2(r + 1)) if r > 0 else 0.0 for _, _, _, r in recs) / n
    return {"recall": recall, "mrr": mrr, "p1": p1, "prec": prec, "ndcg": ndcg}


def pct(sorted_vals, p):
    if not sorted_vals:
        return 0.0
    idx = min(len(sorted_vals) - 1, int(round(p / 100.0 * (len(sorted_vals) - 1))))
    return sorted_vals[idx]


def main():
    ap = argparse.ArgumentParser(description="检索质量评估（双集 + 阈值校准）")
    ap.add_argument("-k", type=int, default=3, help="top-k（默认 3）")
    ap.add_argument("--mode", default="vector_only", choices=["vector_only", "hybrid_rerank"])
    ap.add_argument("--sets", default="all", choices=["all", "basic", "complex"])
    ap.add_argument("--min-score", type=float, default=None, help="可选：给定阈值下输出拒答情况")
    args = ap.parse_args()
    k = args.k

    queries = load_queries(args.sets)
    embedder = DashScopeEmbedder()
    col = get_collection()
    search = make_search(args.mode, embedder, col, PROJECT_ROOT / "data" / "chunks")

    recs = []
    for q in queries:
        res = search(q["query"], k)
        doc_ids = [h["meta"].get("doc") for h in res]
        top1 = res[0]["score"] if res else None
        expected = q.get("expected_doc", "")
        rank = (doc_ids.index(expected) + 1) if (expected and expected in doc_ids) else 0
        recs.append({"set": q.get("set", "basic"), "category": q.get("category", "answerable"),
                     "query": q["query"], "top1": top1, "doc_ids": doc_ids,
                     "expected": expected, "rank": rank, "hard": q.get("hard", False)})

    print(f"模式={args.mode}  评测集≈{len(recs)} 题，top-{k}")
    for setname in ("basic", "complex"):
        sub = [r for r in recs if r["set"] == setname]
        agg = aggregate([(r["top1"], r["doc_ids"], r["expected"], r["rank"]) for r in sub], k)
        if not agg:
            continue
        print(f"\n【{setname}】 {len(sub)} 题   (answerable={sum(1 for r in sub if r['category']=='answerable')})")
        print(f"  recall@{k}    = {agg['recall']:.3f}   （命中：预期文档出现在 top-{k}）")
        print(f"  MRR           = {agg['mrr']:.3f}")
        print(f"  P@1           = {agg['p1']:.3f}")
        print(f"  precision@{k} = {agg['prec']:.3f}   （top-{k} 中来自预期文档的 chunk 占比）")
        print(f"  nDCG@{k}      = {agg['ndcg']:.3f}")

    # 阈值校准：正类=answerable，负类=unanswerable/irrelevant，看 top-1 分数分布
    pos = sorted([r["top1"] for r in recs if r["category"] == "answerable" and r["top1"] is not None])
    neg = sorted([r["top1"] for r in recs if r["category"] in ("unanswerable", "irrelevant") and r["top1"] is not None])
    print("\n" + "-" * 78)
    print("阈值校准（top-1 相似度分布）")
    if pos:
        print(f"  正类(可答)  n={len(pos)}  min={pos[0]:.3f} p25={pct(pos,25):.3f} 中位={pct(pos,50):.3f} "
              f"p75={pct(pos,75):.3f} max={pos[-1]:.3f}")
    else:
        print("  正类: 无")
    if neg:
        print(f"  负类(无答/无关) n={len(neg)}  min={neg[0]:.3f} p25={pct(neg,25):.3f} 中位={pct(neg,50):.3f} "
              f"p75={pct(neg,75):.3f} max={neg[-1]:.3f}")
    else:
        print("  负类: 无")
    if pos and neg:
        print(f"  建议 min-score 落在两分布交界（可答下沿 vs 无答上沿）附近")
        print(f"  可答最低分={pos[0]:.3f}  无答最高分={neg[-1]:.3f}")

    if args.min_score is not None:
        ms = args.min_score
        pos_rej = sum(1 for r in recs if r["category"] == "answerable" and
                      (r["top1"] is None or r["top1"] < ms))
        neg_rej_ok = sum(1 for r in recs if r["category"] in ("unanswerable", "irrelevant") and
                         (r["top1"] is None or r["top1"] < ms))
        neg_total = sum(1 for r in recs if r["category"] in ("unanswerable", "irrelevant"))
        if pos_rej:
            print(f"\n[min-score={ms}] 正类(可答)被误拒: {pos_rej} 题（应尽量避免）")
        if neg_total:
            print(f"[min-score={ms}] 负类被正确拒答: {neg_rej_ok}/{neg_total} 题，拒答率={neg_rej_ok/neg_total:.2%}")

    # 逐题明细
    print("\n" + "-" * 78)
    for r in recs:
        mark = "√" if r["rank"] else "×"
        print(f"[{r['set'][:1]}{mark}] {r['query'][:36]}  (top1={r['top1']:.3f})  预期:{r['expected'][:16]}")


if __name__ == "__main__":
    main()
