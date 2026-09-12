# -*- coding: utf-8 -*-
"""
检索策略消融：四条路线同集对比，定位每一层的增量收益。

路线（按决策确认）：
  A. 混合候选 + 重排   hybrid_rerank ：BM25+向量 RRF 生成候选 -> DashScope rerank
  B. 混合候选（不重排）hybrid        ：BM25+向量 RRF 生成候选，直接取 top-k
  C. 纯向量候选 + 重排 vector_rerank ：同源向量 top-N 候选 -> DashScope rerank
  D. 纯向量（不重排）  vector_only   ：同源向量 top-k（基线）

对比收益来源：
  A vs C -> 「混合候选」相对「纯向量候选」在重排加持下的增量（BM25 是否有用）
  C vs D -> 「重排」的增量（在向量候选上）
  A vs D -> 全套优化的总增量
指标：simple 集 recall@3/MRR/P@1；complex 集 recall@5/precision@5（多来源宏平均）。

运行：.venv_rag311\\Scripts\\python.exe -m scripts.ablate_search [--sets all|simple|complex]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from embedding.dashscope_embedder import DashScopeEmbedder
from scripts._common import get_collection, load_qa_dataset, PROJECT_ROOT, QA_K
from scripts.eval_metrics import aggregate, score_question
from retrieval.search import vector_search
from retrieval.hybrid import build_hybrid_search
from retrieval.reranker import DashScopeReranker

METRIC_KEYS = {"simple": ["recall", "mrr", "p1"], "complex": ["recall", "precision"]}


def build_routes(embedder, col, chunks_dir):
    """返回 {路线名: callable(query, k) -> [hits]}。"""
    hy = build_hybrid_search(col, embedder, Path(chunks_dir))
    rr = DashScopeReranker()

    def A(q, k):   # 混合候选 + 重排
        cand = hy.search(q, k=max(20, k + 6))
        return rr.rerank(q, cand, k)

    def B(q, k):   # 混合候选（不重排）
        return hy.search(q, k)

    def C(q, k):   # 纯向量候选 + 重排
        cand = vector_search(embedder, col, q, max(20, k + 6))
        return rr.rerank(q, cand, k)

    def D(q, k):   # 纯向量（不重排）
        return vector_search(embedder, col, q, k)

    return {"A_混合候选+重排": A, "B_混合候选": B, "C_纯向量候选+重排": C, "D_纯向量": D}


def main():
    ap = argparse.ArgumentParser(description="检索四路消融")
    ap.add_argument("--sets", default="all", choices=["all", "simple", "complex"])
    args = ap.parse_args()

    queries = load_qa_dataset(args.sets)
    if not queries:
        print("未找到评测集。")
        return
    embedder = DashScopeEmbedder()
    col = get_collection()
    routes = build_routes(embedder, col, PROJECT_ROOT / "data" / "chunks")

    print(f"四路消融，共 {len(queries)} 题")
    for setname in ("simple", "complex"):
        sub = [q for q in queries if q["set"] == setname and q["answerable"]]
        if not sub:
            continue
        k = QA_K[setname]
        print(f"\n===== {setname}（可答 {len(sub)} 题，k={k}）=====")
        print(f"{'路线':<22} | " + " | ".join(f"{m:>11}" for m in METRIC_KEYS[setname]))
        for rname, fn in routes.items():
            scores = []
            for q in sub:
                res = fn(q["question"], k)
                docs = [h["meta"].get("doc") for h in res]
                scores.append(score_question(docs, q["sources"], k))
            agg = aggregate(scores, METRIC_KEYS[setname])
            cells = " | ".join(f"{agg.get(m):>11.3f}" if agg.get(m) is not None else f"{'-':>11}"
                               for m in METRIC_KEYS[setname])
            print(f"{rname:<22} | {cells}")
        print("收益来源判读：A vs C = 混合候选增量；C vs D = 重排增量；A vs D = 全套总增量。")


if __name__ == "__main__":
    main()
