# -*- coding: utf-8 -*-
"""检索封装：同源向量 / 混合+重排。供 scripts 与检索评估复用，避免各脚本重复实现检索。

mode 仅两种检索路线（与本项目决策一致）：
  - 'vector_only'     同源向量余弦 top-k（默认生成/基线）
  - 'hybrid_rerank'   BM25+向量 RRF 生成候选 -> DashScope 重排 -> top-k（检索侧优化）
（不再提供单独的 'hybrid'（BM25+向量 RRF 不重排）路线。）
"""
from pathlib import Path


def vector_search(embedder, col, query, k):
    """同源 embedding 向量化 query，在 chroma 里做余弦检索，返回 hit dict 列表。"""
    qvec = embedder.embed_one(query)
    res = col.query(query_embeddings=[qvec], n_results=k)
    out = []
    for dist, cid, doc, meta in zip(res["distances"][0], res["ids"][0],
                                    res["documents"][0], res["metadatas"][0]):
        out.append({"id": cid, "score": 1 - dist, "text": doc, "meta": meta})
    return out


def make_search(mode, embedder, col, chunks_dir):
    """按 mode 返回 search(query, k) -> [hit dict]。mode: 'vector_only' | 'hybrid_rerank'。"""
    if mode == "vector_only":
        return lambda q, k: vector_search(embedder, col, q, k)

    from retrieval.hybrid import build_hybrid_search
    from retrieval.reranker import DashScopeReranker
    hs = build_hybrid_search(col, embedder, Path(chunks_dir))
    rr = DashScopeReranker()

    def search(q, k):
        cand = hs.search(q, k=max(20, k + 6))   # 混合(BM25+向量 RRF) 生成候选
        return rr.rerank(q, cand, k)            # 重排取 top-k（已附 relevance 分）
    return search
