# -*- coding: utf-8 -*-
"""
拒答阈值（min-score）标定（**离线**）：读检索快照 data/eval/retrieval_{route}.jsonl，
取每题 top-1 分数，按可答性分成正类（可答）/ 负类（不可答），给出：

  - 两类的分布（n / min / p25 / 中位 / p75 / max）；
  - 是否可分离：正类下沿 > 负类上沿 => 给出可行区间与建议中点；
  - 「正类误杀 vs 负类漏拦」权衡表：候选阈值**由实测分数生成**（不再写死网格），
    同一 (误杀, 漏拦) 组合只保留最小阈值，表本身就是决策面。

阈值语义：`top1 < min_score` 即拒答（见 run_rag）。故"不误杀"要求阈值 ≤ 正类最小值，
"不漏拦"要求阈值 > 负类最大值。

**本脚本不调任何 API、不自己检索**（快照由 scripts/retrieval_pass.py 产出）。
结论由用户确认后回填 run_rag.MIN_SCORE_DEFAULT / metrics.json / 文档。

产物：data/eval/threshold_{route}.json

运行：.venv_rag311\\Scripts\\python.exe -m scripts.calibrate_threshold --mode vector_only
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts._common import EVAL_DIR, load_qa_dataset


def pct(sorted_vals, p):
    """分位估计（下标法）：用于 p25/p75 这类描述量，不用于"中位数"。"""
    if not sorted_vals:
        return None
    idx = min(len(sorted_vals) - 1, int(round(p / 100.0 * (len(sorted_vals) - 1))))
    return round(sorted_vals[idx], 4)


def median(sorted_vals):
    """真中位数：偶数个取中间两个的均值（与 pct(s, 50) 的下标估计不同，别混用）。"""
    n = len(sorted_vals)
    if n == 0:
        return None
    mid = n // 2
    v = sorted_vals[mid] if n % 2 else (sorted_vals[mid - 1] + sorted_vals[mid]) / 2
    return round(v, 4)


def describe(vals):
    s = sorted(vals)
    return {"n": len(s), "min": pct(s, 0), "p25": pct(s, 25), "median": median(s),
            "p75": pct(s, 75), "max": pct(s, 100)}


def tradeoff(pos, neg):
    """候选阈值权衡表：每个「负类漏拦」水平只保留**正类误杀最少**的那个阈值。

    同一漏拦水平下误杀更多的阈值是被支配的（更严却没多拦住负类），列出来只会把表摊成几十行。
    按阈值升序扫描时误杀单调不减、漏拦单调不增，故每个漏拦水平首次出现的那一行即该水平的最优解。
    同时给出该行对应的**阈值区间**（上一档实测分数, 本行阈值]：区间内任意取值效果完全相同。
    表长 ≤ 负类样本数 + 1，直接就是决策面。
    """
    vals = sorted(set(pos + neg))
    best, prev = {}, None
    for t in vals:
        leak = sum(1 for x in neg if x >= t)
        if leak not in best:
            best[leak] = {"threshold": round(t, 4),
                          "interval": [None if prev is None else round(prev, 4), round(t, 4)],
                          "false_reject": sum(1 for x in pos if x < t), "leak": leak}
        prev = t
    return [best[leak] for leak in sorted(best, reverse=True)]


def build(route, qa_by_id):
    """读快照 -> 分布 / 可分离区间 / 权衡表 / 建议值。快照缺失返回 None。"""
    path = EVAL_DIR / f"retrieval_{route}.jsonl"
    if not path.exists():
        print(f"[错误] 找不到快照 {path}，请先运行 scripts.retrieval_pass --mode {route}")
        return None
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]

    pos, neg, per_q = [], [], []
    for r in rows:
        top1 = r["hits"][0]["score"] if r["hits"] else None
        per_q.append({"id": r["qid"], "set": r["set"], "answerable": r["answerable"], "top1": top1})
        if top1 is None:
            continue
        (pos if r["answerable"] else neg).append(top1)

    pos_d, neg_d = describe(pos), describe(neg)
    print(f"\n=== {route} ===  正类(可答) {pos_d['n']} 题 / 负类(不可答) {neg_d['n']} 题")
    for name, d in (("正类", pos_d), ("负类", neg_d)):
        if d["n"]:
            print(f"  {name}: min={d['min']} p25={d['p25']} 中位={d['median']} "
                  f"p75={d['p75']} max={d['max']}")
        else:
            print(f"  {name}: 无样本")

    result = {"route": route, "n_questions": len(rows), "pos": pos_d, "neg": neg_d,
              "separable": None, "feasible_window": None, "suggested_min_score": None,
              "candidates": [], "note": "", "per_question": per_q}
    if not pos or not neg:
        result["note"] = "缺少正类或负类样本，无法标定"
        print(f"  {result['note']}")
        return result

    p_min, n_max = min(pos), max(neg)
    result["separable"] = p_min > n_max
    if result["separable"]:
        lo, hi = n_max, p_min
        result["feasible_window"] = [round(lo, 4), round(hi, 4)]
        result["suggested_min_score"] = round((lo + hi) / 2, 3)
        result["note"] = (f"可分离：正类下沿 {p_min:.4f} > 负类上沿 {n_max:.4f}；"
                          f"阈值取 ({n_max:.4f}, {p_min:.4f}] 内任意值均 0 误杀 0 漏拦，"
                          f"建议中点 {result['suggested_min_score']}")
    else:
        result["note"] = (f"两分布重叠：正类下沿 {p_min:.4f} <= 负类上沿 {n_max:.4f}，"
                          f"无法干净分割，按权衡表取舍")
    print(f"  {result['note']}")

    result["candidates"] = tradeoff(pos, neg)
    print("  候选阈值权衡（阈值 = 拒答下界；top1 < 阈值 即拒答）：")
    print(f"    {'阈值区间':>20} | {'正类误杀':>8} | {'负类漏拦':>8}")
    for c in result["candidates"]:
        lo, hi = c["interval"]
        span = f"(-∞, {hi:.4f}]" if lo is None else f"({lo:.4f}, {hi:.4f}]"
        print(f"    {span:>20} | {c['false_reject']:>8} | {c['leak']:>8}")
    print("    （同一区间内任意取值效果完全相同；取区间上界最稳，取上界也不影响判据）")
    return result


def main():
    ap = argparse.ArgumentParser(description="拒答阈值标定（离线读快照）")
    ap.add_argument("--mode", default="all", choices=["all", "vector_only", "hybrid_rerank"],
                    help="读哪条路线的快照（默认 all = 两条都标定）")
    ap.add_argument("--sets", default="all", choices=["all", "simple", "complex"])
    ap.add_argument("--out", default=None, help="输出路径（默认 data/eval/threshold_{route}.json）")
    args = ap.parse_args()

    queries = load_qa_dataset(args.sets)
    if not queries:
        print("未找到评测集（见 scripts/_common.py 的 QA_FILES，源文件在 test/qa_dataset/）。")
        return
    qa_by_id = {q["id"]: q for q in queries}
    modes = ["vector_only", "hybrid_rerank"] if args.mode == "all" else [args.mode]

    for route in modes:
        result = build(route, qa_by_id)
        if result is None:
            continue
        out = Path(args.out) if args.out else EVAL_DIR / f"threshold_{route}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"  -> {out}（建议阈值 {result['suggested_min_score']}，待确认后回填）")

    print("\n注：阈值确认后需回填 run_rag.MIN_SCORE_DEFAULT、metrics.json、README/docs。")


if __name__ == "__main__":
    main()
