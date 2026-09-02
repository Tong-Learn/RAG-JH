# -*- coding: utf-8 -*-
"""
混合检索：BM25(rank_bm25) + 向量(chroma cosine) 的 RRF(Reciprocal Rank Fusion) 融合。

设计（见 docs/优化规划方案.md §4）：
- BM25 对「精确词 / 数字 / 专名」敏感，补向量检索的短板（游戏公告满是专名+数字+价格）；
- 两个排序用 RRF 融合，避免分数尺度不可比；k 取 60（常用经验值）；
- tokenize 优先 jieba 分词，否则回退「中文逐字 + 英文数字整词」（不依赖 jieba 也能跑，精度略低）。

search(query, k, rerank=False)：默认融合后取 top-k（rerank 由 el_retrieval 里再套 DashScope rerank）。
"""
import json
import re
from pathlib import Path

try:
    import jieba
    _V = "jieba"
except Exception:  # noqa: BLE001
    _V = "char"


def _tokenize(text: str) -> list:
    text = text.lower()
    if _V == "jieba":
        try:
            return [t for t in jieba.cut(text) if t.strip()]
        except Exception:  # noqa: BLE001
            pass
    return re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]", text)


def rrf_fuse(ranked_lists, k=60):
    """ranked_lists: [[id, ...], ...]。返回按 RRF 得分降序的 (id, score) 列表。"""
    scores = {}
    for lst in ranked_lists:
        for rank, pid in enumerate(lst):
            scores[pid] = scores.get(pid, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: -x[1])


class HybridSearch:
    def __init__(self, col, embedder, corpus):
        """corpus: list of chunk dict {id,text,doc,section_path,start_page,end_page,...}。"""
        from rank_bm25 import BM25Okapi
        self._col = col
        self._embedder = embedder
        self._ids = [c["id"] for c in corpus]
        self._meta = {c["id"]: c for c in corpus}
        self._bm25 = BM25Okapi([_tokenize(t["text"]) for t in corpus])

    def _vector(self, query, k):
        qvec = self._embedder.embed_one(query)
        res = self._col.query(query_embeddings=[qvec], n_results=k)
        out = []
        for dist, cid, doc, meta in zip(res["distances"][0], res["ids"][0],
                                        res["documents"][0], res["metadatas"][0]):
            out.append({"id": cid, "score": 1 - dist, "text": doc, "meta": meta})
        return out

    def _bm25_top(self, query, k):
        scores = self._bm25.get_scores(_tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
        return [self._ids[i] for i in order if scores[i] > 0]

    def search(self, query, k=4, rerank=False):
        n = max(20, k + 6)
        vec = self._vector(query, n)
        bm = self._bm25_top(query, n)
        vec_ids = [h["id"] for h in vec]
        fused = rrf_fuse([vec_ids, bm])
        chosen = [pid for pid, _ in fused if pid in self._meta][:k]
        vec_by_id = {h["id"]: h for h in vec}
        out = []
        for pid in chosen:
            h = vec_by_id.get(pid)
            if h is None:   # 仅 BM25 命中，从 corpus 取 text/meta
                c = self._meta[pid]
                h = {"id": pid, "score": 0.0, "text": c["text"],
                     "meta": {"doc": c.get("doc"),
                              "section_path": " / ".join(c.get("section_path") or []),
                              "start_block": c.get("start_block", 0), "end_block": c.get("end_block", 0),
                              "start_page": c.get("start_page", 0), "end_page": c.get("end_page", 0)}}
            out.append(h)
        return out


def build_hybrid_search(col, embedder, chunks_dir):
    corpus = []
    for f in sorted(Path(chunks_dir).glob("*.chunks.jsonl")):
        for line in open(f, encoding="utf-8"):
            if line.strip():
                corpus.append(json.loads(line))
    return HybridSearch(col, embedder, corpus)
