# -*- coding: utf-8 -*-
"""
混合检索：BM25(rank_bm25) + 向量(chroma cosine) 的 RRF(Reciprocal Rank Fusion) 融合。

设计（完整方案见 docs/检索方案.md）：
- BM25 对「精确词 / 数字 / 专名」敏感，补向量检索的短板（游戏公告满是专名+数字+价格）；
- 两个排序用 RRF 融合，避免分数尺度不可比；k 取 60（常用经验值）；
- tokenize 优先 jieba 分词，否则回退「中文逐字 + 英文数字整词」（不依赖 jieba 也能跑，精度略低）。

语料侧产物（与 data/chroma 对称，由 scripts/run_bm25_build.py 构建）：
- 逐块分词 + BM25Okapi 建索引是语料侧的固定开销（605 块约 0.38s），落盘后每个进程直接加载，
  查询期只对 query 分词；**不做指纹校验/失效逻辑**——语料或切片变了就重跑构建脚本。
- 产物缺失时本模块回退「现场读 data/chunks + 分词 + 建索引」，流程不受影响。

search(query, k)：融合后返回 top-k 候选（供重排；本项目不再单用作独立检索路线）。
"""
import json
import pickle
import re
from pathlib import Path

try:
    import jieba
    _V = "jieba"
except Exception:  # noqa: BLE001
    _V = "char"

# BM25 语料侧产物路径（构建脚本与查询期共用，避免两处各写一份路径）
BM25_DIR = Path(__file__).resolve().parents[1] / "data" / "bm25"
BM25_ARTIFACT = BM25_DIR / "bm25_corpus.pkl"


def tokenize(text: str) -> list:
    """分词（**构建产物与查询期必须共用本函数**）。

    语料侧产物里的 idf/doc_freqs 是按本函数切出的词表算的；查询期若换一套分词，
    分出的词对不上词表，BM25 分数会静默变差。故构建脚本照此导入，不另写一份。
    """
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
        if not corpus:
            # 空语料下 BM25Okapi([]) 会抛 ZeroDivisionError（avgdl 除零），信息量极差。
            # 这里显式给出可执行的指引，便于"新克隆/未建索引"时快速定位。
            raise RuntimeError(
                "混合检索语料为空：未找到任何 chunk。请先构建数据与索引：\n"
                "  1) preprocessing.pipeline   （PDF → data/processed）\n"
                "  2) scripts.run_chunking     （→ data/chunks）\n"
                "  3) scripts.run_vectorize    （→ data/chroma）"
            )
        self._col = col
        self._embedder = embedder
        self._ids = [c["id"] for c in corpus]
        self._meta = {c["id"]: c for c in corpus}
        self._bm25 = BM25Okapi([tokenize(t["text"]) for t in corpus])

    @classmethod
    def from_artifact(cls, col, embedder, artifact):
        """用已落盘的 BM25 产物构造（**不读 data/chunks、不重新分词**）。

        产物的 doc 下标顺序与 `ids` 一致，故 `_bm25_top` 用 `self._ids[i]` 回映射；
        `_meta` 由 `texts` + `meta` 合并成与 `__init__` 相同的形状（text + 溯源字段），
        这样"仅 BM25 命中"的分支不必区分来源。
        """
        self = cls.__new__(cls)                 # 绕过 __init__ 的空语料检查（产物自带语料）
        self._col = col
        self._embedder = embedder
        self._ids = list(artifact["ids"])
        texts, meta = artifact["texts"], artifact["meta"]
        self._meta = {cid: {**(meta.get(cid) or {}), "text": texts.get(cid, "")}
                      for cid in self._ids}
        self._bm25 = artifact["bm25"]
        return self

    def _vector(self, query, k):
        qvec = self._embedder.embed_one(query)
        res = self._col.query(query_embeddings=[qvec], n_results=k)
        out = []
        for dist, cid, doc, meta in zip(res["distances"][0], res["ids"][0],
                                        res["documents"][0], res["metadatas"][0]):
            out.append({"id": cid, "score": 1 - dist, "text": doc, "meta": meta})
        return out

    def _bm25_top(self, query, k):
        scores = self._bm25.get_scores(tokenize(query))
        order = sorted(range(len(scores)), key=lambda i: -scores[i])[:k]
        return [self._ids[i] for i in order if scores[i] > 0]

    def search(self, query, k=4):
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
                              "start_page": c.get("start_page", 0), "end_page": c.get("end_page", 0)}}
            out.append(h)
        return out


def load_artifact(path=None):
    """读 BM25 语料侧产物；不存在则返回 None（调用方回退现场构建）。"""
    p = Path(path) if path else BM25_ARTIFACT
    if not p.exists():
        return None
    with open(p, "rb") as f:
        return pickle.load(f)


def build_hybrid_search(col, embedder, chunks_dir):
    """构造混合检索：**优先加载语料侧产物**，缺省才现场读切片产物并分词建索引。"""
    artifact = load_artifact()
    if artifact is not None:
        return HybridSearch.from_artifact(col, embedder, artifact)
    corpus = []
    for f in sorted(Path(chunks_dir).glob("*.chunks.jsonl")):
        for line in open(f, encoding="utf-8"):
            if line.strip():
                corpus.append(json.loads(line))
    return HybridSearch(col, embedder, corpus)
