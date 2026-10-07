# -*- coding: utf-8 -*-
"""
检索 pass：对评测集全部题目跑指定路线的 top-k 检索，把**命中快照**落盘。

**本脚本是唯一调检索 API 的地方**（embedding + 重排）。指标与阈值都离线读它的产物计算：
  - scripts/eval_retrieval.py     读快照算检索指标
  - scripts/calibrate_threshold.py 读快照算拒答阈值分布
要重算指标/阈值就先重跑本脚本，避免"两套口径"。

产物 data/eval/retrieval_{route}.jsonl，每题一行：
  {qid, set, answerable, route, k, hits: [{chunk_id, doc, section_path, page, score}]}
  —— **不存片段正文**：需要正文时按 chunk_id 回 data/chunks 取（judge 就是这么做的）。

k 两集统一取 _common.QA_K（当前都 = 5）。

运行：
  .venv_rag311\\Scripts\\python.exe -m scripts.retrieval_pass                      # 两条路线
  .venv_rag311\\Scripts\\python.exe -m scripts.retrieval_pass --mode vector_only
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from embedding.dashscope_embedder import DashScopeEmbedder
from scripts._common import (EVAL_DIR, PROJECT_ROOT, QA_K, get_collection,
                             load_qa_dataset, preflight, print_preflight)
from retrieval.search import make_search


def hit_record(h):
    """hit -> 落盘用的一条命中（只留 chunk id 与溯源坐标，不存正文）。"""
    meta = h["meta"]
    pg = [meta.get("start_page"), meta.get("end_page")]
    return {"chunk_id": h["id"], "doc": meta.get("doc"),
            "section_path": meta.get("section_path"),
            "page": pg if pg[0] else None,
            "score": h["score"]}


def write_snapshot(path, rows):
    """把快照行写成 JSONL（每题一行）。"""
    Path(path).write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
                          encoding="utf-8")


def run_route(mode, queries, embedder, col):
    """跑一条路线，落一份快照，返回产物路径。"""
    search = make_search(mode, embedder, col, PROJECT_ROOT / "data" / "chunks")
    rows = []
    for q in queries:
        k = QA_K[q["set"]]
        hits = search(q["question"], k)
        top1 = hits[0]["score"] if hits else None
        rows.append({"qid": q["id"], "set": q["set"], "answerable": q["answerable"],
                     "route": mode, "k": k, "hits": [hit_record(h) for h in hits]})
        print(f"  [{mode}][{q['id']}] top1={'无命中' if top1 is None else f'{top1:.4f}'}"
              f"  命中 {len(hits)} 条")
    path = EVAL_DIR / f"retrieval_{mode}.jsonl"
    write_snapshot(path, rows)
    print(f"  -> {path}（{len(rows)} 题）")
    return path


def main():
    ap = argparse.ArgumentParser(description="检索 pass：落盘命中快照（唯一调检索 API 的入口）")
    ap.add_argument("--mode", default="all", choices=["all", "vector_only", "hybrid_rerank"],
                    help="跑哪条路线（默认 all = 两条都跑、各落一份快照）")
    ap.add_argument("--sets", default="all", choices=["all", "simple", "complex"])
    args = ap.parse_args()

    queries = load_qa_dataset(args.sets)
    if not queries:
        print("未找到评测集（见 scripts/_common.py 的 QA_FILES，源文件在 test/qa_dataset/）。")
        return
    modes = ["vector_only", "hybrid_rerank"] if args.mode == "all" else [args.mode]
    for mode in modes:                       # 前置检查：缺索引/切片产物就早退，别跑到一半才炸
        if not print_preflight(preflight(mode)):
            return

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    embedder = DashScopeEmbedder()
    col = get_collection()
    print(f"共 {len(queries)} 题；路线 {modes}；k 按集 {QA_K}")
    for mode in modes:
        print(f"\n=== {mode} ===")
        run_route(mode, queries, embedder, col)


if __name__ == "__main__":
    main()
