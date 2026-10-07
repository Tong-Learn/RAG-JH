# -*- coding: utf-8 -*-
"""
BM25 语料侧产物构建（与 run_vectorize 并列的一步，切片之后即可跑）。

为什么要有这一步：混合检索的语料侧要做「逐块分词 + 建 BM25Okapi」，实测 605 块约 0.4s；
把它像向量库一样落盘，此后每个进程直接加载，查询期只对 query 分词。
**不做指纹校验、不做失效逻辑**——语料或切片变了就重跑本脚本（与 data/chroma 同等对待）。

产物 data/bm25/bm25_corpus.pkl（dict）：
  ids    [chunk_id, ...]                     顺序与 BM25Okapi 的 doc 下标一一对应
  bm25   BM25Okapi                           内含 doc_freqs / doc_len / avgdl / idf
  texts  {chunk_id: chunk.text}              仅被 BM25 命中的块要用它拼 hit（交重排/生成）
  meta   {chunk_id: {doc, section_path, start_page, end_page}}   溯源字段
**不存分词结果**：查询期 get_scores() 只用 BM25Okapi 内部统计量，存了没人读（白增 0.4MB）。

运行：.venv_rag311\\Scripts\\python.exe -m scripts.run_bm25_build
"""
import json
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from retrieval.hybrid import BM25_ARTIFACT, tokenize

CHUNKS_DIR = Path(__file__).resolve().parents[1] / "data" / "chunks"


def load_corpus(chunks_dir=CHUNKS_DIR):
    """读全部切片产物，返回 chunk dict 列表（与 run_vectorize 同一批、同一顺序规则）。"""
    corpus = []
    for f in sorted(Path(chunks_dir).glob("*.chunks.jsonl")):
        for line in open(f, encoding="utf-8"):
            if line.strip():
                corpus.append(json.loads(line))
    return corpus


def build_artifact(corpus):
    """语料 -> 产物 dict。分词复用 retrieval.hybrid.tokenize（与查询期同一函数）。"""
    from rank_bm25 import BM25Okapi
    ids = [c["id"] for c in corpus]
    texts = {c["id"]: c["text"] for c in corpus}
    meta = {c["id"]: {"doc": c.get("doc"),
                      "section_path": " / ".join(c.get("section_path") or []),
                      "start_page": c.get("start_page", 0),
                      "end_page": c.get("end_page", 0)} for c in corpus}
    bm25 = BM25Okapi([tokenize(c["text"]) for c in corpus])
    return {"ids": ids, "bm25": bm25, "texts": texts, "meta": meta}


def save_artifact(artifact, path=None):
    """写产物文件，返回路径（目录不存在则建）。"""
    p = Path(path) if path else BM25_ARTIFACT
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "wb") as f:
        pickle.dump(artifact, f, protocol=4)
    return p


def main():
    corpus = load_corpus()
    if not corpus:
        print("无切片产物可构建：data/chunks 下没有 *.chunks.jsonl。\n"
              "请先运行：python -m preprocessing.pipeline --src test --out data\\processed\n"
              "          python -m scripts.run_chunking")
        return

    print(f"[1/2] 读入 {len(corpus)} 个 chunk，分词并建 BM25 索引 ...")
    t0 = time.time()
    artifact = build_artifact(corpus)
    cost = (time.time() - t0) * 1000
    bm25 = artifact["bm25"]
    total_tokens = sum(bm25.doc_len)

    path = save_artifact(artifact)
    size_mb = path.stat().st_size / 1024 / 1024
    print(f"[2/2] 完成：分词+建索引 {cost:.0f} ms；token 总数 {total_tokens}"
          f"（平均 {bm25.avgdl:.1f}/块，词表 {len(bm25.idf)} 个）")
    print(f"      -> {path}（{size_mb:.2f} MB）")


if __name__ == "__main__":
    main()
