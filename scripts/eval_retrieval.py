# -*- coding: utf-8 -*-
"""
检索质量评估（新评测集）：读 test/qa_dataset/（simple_qa_30 / complex_qa_20），做 top-k 检索并按
「来源文档集合」计算指标；不可答题单独统计（不计入检索指标）。

指标（已按决策缩减）：
  - simple 集（单来源，k=3）：recall@k / MRR / P@1
  - complex 集（多来源，k=5）：recall@k（来源覆盖率宏平均）/ precision@k
  - 不可答题（3 题）：拒答率（需 --min-score；否则只报数量）
precision@k/nDCG 等已移除；any-hit/all-hit 仅 --detail 时作为参照输出。

两条检索路线：--mode vector_only（默认）| hybrid_rerank（BM25+向量 RRF 候选 -> 重排）。

运行：.venv_rag311\\Scripts\\python.exe -m scripts.eval_retrieval [--mode vector_only|hybrid_rerank] [--sets all|simple|complex] [--min-score 0.70] [--detail]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from embedding.dashscope_embedder import DashScopeEmbedder
from scripts._common import get_collection, load_qa_dataset, PROJECT_ROOT, QA_K
from scripts.eval_metrics import aggregate, score_question, any_hit_rate, all_hit_rate
from retrieval.search import make_search

# 各集要报告的指标
METRIC_KEYS = {"simple": ["recall", "mrr", "p1"], "complex": ["recall", "precision"]}
METRIC_LABEL = {"recall": "recall@k", "mrr": "MRR", "p1": "P@1", "precision": "precision@k"}


def run_set(setname, queries, search, min_score, detail):
    k = QA_K[setname]
    scores, rejected_neg, answered_neg = [], 0, 0
    detail_rows = []
    for q in queries:
        res = search(q["question"], k)
        docs = [h["meta"].get("doc") for h in res]
        top1 = res[0]["score"] if res else None
        if q["answerable"]:
            s = score_question(docs, q["sources"], k)
            scores.append(s)
            detail_rows.append((q["id"], q["question"], docs, q["sources"], top1, s))
        else:
            # 不可答：理想情况是拒答（top1 < 阈值）
            rej = (top1 is None) or (min_score is not None and top1 < min_score)
            if rej:
                rejected_neg += 1
            else:
                answered_neg += 1
            detail_rows.append((q["id"], q["question"], docs, [], top1, None))

    keys = METRIC_KEYS[setname]
    agg = aggregate(scores, keys)
    n_neg = rejected_neg + answered_neg
    print(f"\n【{setname}】 可答 {agg['n']} 题，k={k}" + (f"，另有不可答 {n_neg} 题" if n_neg else ""))
    for key in keys:
        v = agg.get(key)
        print(f"  {METRIC_LABEL[key]:<12}= {'None' if v is None else f'{v:.3f}'}")
    if detail:
        print(f"  [参照] any-hit={any_hit_rate(scores)}  all-hit={all_hit_rate(scores)}")
    if n_neg:
        if min_score is None:
            print(f"  [不可答] {n_neg} 题（未给 --min-score，不判拒答）")
        else:
            print(f"  [不可答] 正确拒答 {rejected_neg}/{n_neg} = {rejected_neg/n_neg:.0%}  "
                  f"漏答(未拒) {answered_neg} 题  (阈值 {min_score})")
    if detail:
        print("  --- 逐题 ---")
        for qid, ques, docs, src, top1, s in detail_rows:
            mark = "-" if s is None else ("√" if s["recall"] > 0 else "×")
            t1 = f"{top1:.3f}" if top1 is not None else "None"
            print(f"  [{qid} {mark}] top1={t1}  {ques[:34]}")
    return agg, scores


def main():
    ap = argparse.ArgumentParser(description="检索质量评估（新评测集）")
    ap.add_argument("--mode", default="vector_only", choices=["vector_only", "hybrid_rerank"])
    ap.add_argument("--sets", default="all", choices=["all", "simple", "complex"])
    ap.add_argument("--min-score", type=float, default=None, help="给定时输出不可答题的拒答率")
    ap.add_argument("--detail", action="store_true", help="输出逐题明细与 any/all-hit 参照")
    args = ap.parse_args()

    queries = load_qa_dataset(args.sets)
    if not queries:
        print("未找到评测集（test/qa_dataset/ 或 data/eval/）。")
        return
    embedder = DashScopeEmbedder()
    col = get_collection()
    search = make_search(args.mode, embedder, col, PROJECT_ROOT / "data" / "chunks")

    print(f"模式={args.mode}  共 {len(queries)} 题")
    for setname in ("simple", "complex"):
        sub = [q for q in queries if q["set"] == setname]
        if sub:
            run_set(setname, sub, search, args.min_score, args.detail)


if __name__ == "__main__":
    main()
