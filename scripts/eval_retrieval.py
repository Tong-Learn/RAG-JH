# -*- coding: utf-8 -*-
"""
检索质量评估（流程四）：用评测集 data/eval/queries.json 对 chromadb 做 top-k 检索，
按「标准答案所在文档」计算指标（doc 级，粗粒度）。

指标：
  - recall@k   命中率：top-k 内是否出现预期文档(命中=1/未命中=0)，平均即 recall@k
  - MRR        平均倒数排名：预期文档第一次出现的位次取倒数，平均(未命中记0)
  - P@1        top-1 命中率：top-1 是否为预期文档（更严苛的「首要命中」）
  - precision@k top-k 中来自预期文档的 chunk 占比（chunk 级「答案密度」，反映答案是否集中）
  - nDCG@k     位置折扣(单个相关文档约定)：预期文档首现位次 r 时取 1/log2(r+1)，未命中记0

默认 k=3（语料较小，top-3 比 top-5 更接近真实可用的截断）；可 -k 5 复现 top-5。

运行：.venv_rag311\\Scripts\\python.exe -m scripts.eval_retrieval [-k 3]
"""
import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import chromadb
from chromadb.config import Settings

from embedding.dashscope_embedder import DashScopeEmbedder

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHROMA_DIR = PROJECT_ROOT / "data" / "chroma"
EVAL_FILE = PROJECT_ROOT / "data" / "eval" / "queries.json"
COLLECTION = "rag_chunks"


def load_queries():
    with open(EVAL_FILE, encoding="utf-8") as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser(description="检索质量评估")
    ap.add_argument("-k", type=int, default=3, help="top-k（默认 3，小语料更紧）")
    args = ap.parse_args()
    k = args.k

    queries = load_queries()
    embedder = DashScopeEmbedder()
    client = chromadb.PersistentClient(path=str(CHROMA_DIR), settings=Settings(anonymized_telemetry=False))
    col = client.get_or_create_collection(COLLECTION)

    # 逐题检索，记录 (预期, 命中doc列表, 首现位次)
    recs = []
    for q in queries:
        qvec = embedder.embed_one(q["query"])
        res = col.query(query_embeddings=[qvec], n_results=k)
        doc_ids = [m.get("doc") for m in res["metadatas"][0]]
        expected = q["expected_doc"]
        rank = (doc_ids.index(expected) + 1) if expected in doc_ids else 0
        recs.append((q["query"], expected, doc_ids, rank))

    # 指标聚合
    n = len(recs)
    recall = sum(1 for _, _, _, r in recs if r > 0) / n
    mrr = sum(1.0 / r for _, _, _, r in recs if r > 0) / n
    p1 = sum(1 for _, _, _, r in recs if r == 1) / n
    # chunk 级 precision@k：top-k 里来自预期文档的 chunk 占比
    prec = 0.0
    for _, expected, doc_ids, _ in recs:
        prec += sum(1 for d in doc_ids if d == expected) / k
    prec /= n
    # nDCG@k（单相关文档约定：IDCG=1，DCG=1/log2(r+1)）
    ndcg = sum((1.0 / math.log2(r + 1)) if r > 0 else 0.0 for _, _, _, r in recs) / n

    print(f"评测集 {n} 题，top-{k}")
    print(f"  recall@{k}    = {recall:.3f}   （命中率：预期文档出现在 top-{k}）")
    print(f"  MRR           = {mrr:.3f}   （预期文档首现位次的倒数均值）")
    print(f"  P@1           = {p1:.3f}   （top-1 即命中预期文档）")
    print(f"  precision@{k} = {prec:.3f}   （top-{k} 中来自预期文档的 chunk 占比，答案密度）")
    print(f"  nDCG@{k}      = {ndcg:.3f}   （位置折扣：越靠前得分越高）")
    print("-" * 78)

    for query, expected, doc_ids, rank in recs:
        mark = "√" if rank else "×"
        show_docs = [d[:16] for d in doc_ids]
        if rank:
            print(f"[{mark}] rank={rank:<2} 预期: {expected[:18]}")
        else:
            print(f"[{mark}] rank=-  预期: {expected[:18]}")
        print(f"      命: {show_docs}")


if __name__ == "__main__":
    main()
