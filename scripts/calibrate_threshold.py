# -*- coding: utf-8 -*-
"""
阈值（拒答值）重校准：用**当前库 + 新评测集**重新标定 min-score。

做法：对 test/qa_dataset/ 的全部 50 题做 top-k 检索（--mode vector_only | hybrid_rerank），
取每题 top-1 相似度，按可答性分成两类：
  - 正类（answerable=True，47 题）：top-1 分布
  - 负类（answerable=False，3 题） ：top-1 分布（理想情况应整体偏低）
然后给出建议阈值：两分布有间隙 -> 取中点；重叠 -> 输出"误杀 vs 漏拦"权衡表供选择。
**只做检索，不重嵌入、不改任何值**；结论由用户确认后回填 run_rag 默认值 / README / metrics.json。

指标口径：simple k=3、complex k=5（与 eval_retrieval 一致）。

运行：.venv_rag311\\Scripts\\python.exe -m scripts.calibrate_threshold --mode vector_only [--out data/eval/threshold_vector.json]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from embedding.dashscope_embedder import DashScopeEmbedder
from scripts._common import get_collection, load_qa_dataset, PROJECT_ROOT, QA_K, EVAL_DIR
from retrieval.search import make_search


def pct(sorted_vals, p):
    if not sorted_vals:
        return None
    idx = min(len(sorted_vals) - 1, int(round(p / 100.0 * (len(sorted_vals) - 1))))
    return round(sorted_vals[idx], 3)


def collect(mode):
    queries = load_qa_dataset("all")
    embedder = DashScopeEmbedder()
    col = get_collection()
    search = make_search(mode, embedder, col, PROJECT_ROOT / "data" / "chunks")
    pos, neg, rows = [], [], []
    for q in queries:
        k = QA_K[q["set"]]
        res = search(q["question"], k)
        top1 = res[0]["score"] if res else None
        rows.append({"id": q["id"], "set": q["set"], "answerable": q["answerable"], "top1": top1})
        if top1 is None:
            continue
        (pos if q["answerable"] else neg).append(top1)
    return sorted(pos), sorted(neg), rows


def suggest(pos, neg):
    if not pos or not neg:
        return None, "缺少正类或负类样本，无法标定"
    p_min, n_max = pos[0], neg[-1]
    if p_min > n_max:
        return round((p_min + n_max) / 2, 3), f"分离开：可答下沿={p_min:.3f} > 无答上沿={n_max:.3f}，取中点"
    return None, f"重叠：可答下沿={p_min:.3f} <= 无答上沿={n_max:.3f}，见权衡表"


def main():
    ap = argparse.ArgumentParser(description="阈值重校准（新评测集）")
    ap.add_argument("--mode", default="vector_only", choices=["vector_only", "hybrid_rerank"])
    ap.add_argument("--out", default=None, help="校准 json 输出路径")
    args = ap.parse_args()

    pos, neg, rows = collect(args.mode)
    print(f"模式={args.mode}  正类(可答) {len(pos)} 题 / 负类(不可答) {len(neg)} 题")
    print(f"  正类: min={pos[0]:.3f} p25={pct(pos,25)} 中位={pct(pos,50)} p75={pct(pos,75)} max={pos[-1]:.3f}"
          if pos else "  正类: 无")
    print(f"  负类: min={neg[0]:.3f} p25={pct(neg,25)} 中位={pct(neg,50)} p75={pct(neg,75)} max={neg[-1]:.3f}"
          if neg else "  负类: 无")

    rec, msg = suggest(pos, neg)
    print(f"\n建议阈值: {rec}  ({msg})")

    if pos and neg:
        print("\n候选阈值权衡（正类误杀 / 负类漏拦）：")
        print(f"  {'阈值':>6} | {'正类误杀':>8} | {'负类漏拦':>8}")
        for ms in (0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85):
            pr = sum(1 for x in pos if x < ms)
            nl = sum(1 for x in neg if x >= ms)
            print(f"  {ms:>6.2f} | {pr:>8} | {nl:>8}")

    result = {
        "mode": args.mode,
        "pos": {"n": len(pos), "min": pos[0] if pos else None, "median": pct(pos, 50),
                "max": pos[-1] if pos else None},
        "neg": {"n": len(neg), "min": neg[0] if neg else None, "median": pct(neg, 50),
                "max": neg[-1] if neg else None},
        "suggested_min_score": rec,
        "note": msg,
        "per_question": rows,
    }
    out = Path(args.out) if args.out else EVAL_DIR / f"threshold_{args.mode}.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\n校准结果写至 {out}（建议阈值 {rec}，回填待确认）")


if __name__ == "__main__":
    main()
