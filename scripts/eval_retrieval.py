# -*- coding: utf-8 -*-
"""
检索质量评估（**离线**）：读检索快照 data/eval/retrieval_{route}.jsonl，配合同目录的评测集算指标。

**本脚本不调任何 API、不自己检索**——快照由 scripts/retrieval_pass.py 产出；要重算先重跑那一侧。

口径（**文档级 / 来源覆盖率**）：设 G = 该题 `sources`（文档集合，可答题非空），
R_k = top-k 命中片段**所属文档**的集合（同一文档多个片段只算一次）：
  - recall@k    = |R_k ∩ G| / |G|
  - precision@k = |{命中片段 : 其 doc ∈ G}| / k
  - P@1         = 1 if top-1 片段所属 doc ∈ G else 0
  - MRR         = 1 / (第一个所属 doc ∈ G 的片段排名)，未命中为 0
不可答题（G 为空）**不参与上述指标**（跳过、不补 0），只统计题数；拒答率看 threshold_{route}.json。

报告：simple → recall@k / MRR / P@1；complex → recall@k / precision@k（均为宏平均）。

运行：
  .venv_rag311\\Scripts\\python.exe -m scripts.eval_retrieval                    # 两条路线
  .venv_rag311\\Scripts\\python.exe -m scripts.eval_retrieval --mode hybrid_rerank --detail
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts._common import EVAL_DIR, load_qa_dataset
from scripts.eval_metrics import aggregate, all_hit_rate, any_hit_rate, score_question

# 各集要报告的指标（沿用原口径：simple 报 P@1/MRR，complex 多来源不报 P@1/MRR）
METRIC_KEYS = {"simple": ["recall", "mrr", "p1"], "complex": ["recall", "precision"]}
METRIC_LABEL = {"recall": "recall@k", "mrr": "MRR", "p1": "P@1", "precision": "precision@k"}


def load_snapshot(route):
    """读一条路线的快照，返回 {qid: row}；文件不存在返回 None。"""
    path = EVAL_DIR / f"retrieval_{route}.jsonl"
    if not path.exists():
        print(f"[错误] 找不到快照 {path}，请先运行 scripts.retrieval_pass --mode {route}")
        return None
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    return {r["qid"]: r for r in rows}


def run_route(route, qa_by_id, detail):
    """对一条路线的快照算指标，返回 {setname: agg}。"""
    snap = load_snapshot(route)
    if snap is None:
        return None
    missing = [qid for qid in qa_by_id if qid not in snap]
    unknown = [qid for qid in snap if qid not in qa_by_id]
    if missing:
        print(f"  [警告] 快照缺 {len(missing)} 题（{', '.join(missing[:6])}…）"
              f"——快照与评测集不同版本，请重跑 scripts.retrieval_pass")
    if unknown:
        print(f"  [警告] 快照含 {len(unknown)} 题不在当前评测集里（{', '.join(unknown[:6])}…）")

    per_set = {}
    for setname in ("simple", "complex"):
        ids = [qid for qid, q in qa_by_id.items() if q["set"] == setname]
        if not ids:
            continue
        scores, rows, n_neg = [], [], 0
        k = None
        for qid in ids:
            q = qa_by_id[qid]
            row = snap.get(qid)
            if row is None:
                continue
            k = row["k"]
            docs = [h["doc"] for h in row["hits"]]
            top1 = row["hits"][0]["score"] if row["hits"] else None
            if q["answerable"]:
                s = score_question(docs, q["sources"], row["k"])
                scores.append(s)
                rows.append((qid, q["question"], docs, q["sources"], top1, s))
            else:                       # 不可答：不计指标，只计数
                n_neg += 1
                rows.append((qid, q["question"], docs, [], top1, None))
        agg = aggregate(scores, METRIC_KEYS[setname])
        per_set[setname] = agg
        print(f"\n【{route} / {setname}】 可答 {agg['n']} 题，k={k}"
              + (f"，另有不可答 {n_neg} 题（不参与检索指标）" if n_neg else ""))
        for key in METRIC_KEYS[setname]:
            v = agg.get(key)
            print(f"  {METRIC_LABEL[key]:<12}= {'None' if v is None else f'{v:.3f}'}")
        if detail:
            print(f"  [参照] any-hit={any_hit_rate(scores)}  all-hit={all_hit_rate(scores)}")
            print("  --- 逐题 ---")
            for qid, ques, docs, src, top1, s in rows:
                mark = "-" if s is None else ("√" if s["recall"] > 0 else "×")
                t1 = f"{top1:.4f}" if top1 is not None else "None"
                print(f"  [{qid} {mark}] top1={t1}  命中doc={docs}  {ques[:34]}")
    return per_set


def main():
    ap = argparse.ArgumentParser(description="检索质量评估（离线读快照）")
    ap.add_argument("--mode", default="all", choices=["all", "vector_only", "hybrid_rerank"],
                    help="读哪条路线的快照（默认 all = 两条都算，离线代价为零）")
    ap.add_argument("--sets", default="all", choices=["all", "simple", "complex"])
    ap.add_argument("--detail", action="store_true", help="输出逐题明细与 any/all-hit 参照")
    args = ap.parse_args()

    queries = load_qa_dataset(args.sets)
    if not queries:
        print("未找到评测集（见 scripts/_common.py 的 QA_FILES，源文件在 test/qa_dataset/）。")
        return
    qa_by_id = {q["id"]: q for q in queries}
    modes = ["vector_only", "hybrid_rerank"] if args.mode == "all" else [args.mode]

    print(f"离线评估：共 {len(queries)} 题，路线 {modes}")
    summary = {}
    for route in modes:
        print(f"\n===== {route} =====")
        per_set = run_route(route, qa_by_id, args.detail)
        if per_set:
            summary[route] = per_set

    if len(summary) > 1:                # 两条路线并排对比（便于判读，不再逐个上翻）
        print("\n===== 汇总（可答题，宏平均）=====")
        for setname in ("simple", "complex"):
            if not all(setname in per_set for per_set in summary.values()):
                continue
            keys = METRIC_KEYS[setname]
            print(f"【{setname}】 " + " | ".join(
                f"{route}: " + " ".join(f"{METRIC_LABEL[k]}={summary[route][setname].get(k)}" for k in keys)
                for route in modes))
    print("\n注：不可答题不计入检索指标（G 为空则跳过）；拒答率与阈值权衡见 data/eval/threshold_*.json")


if __name__ == "__main__":
    main()
