# -*- coding: utf-8 -*-
"""
检索冒烟测试（流程四前置）：对给定问题用同一 embedding 模型向量化，在 chromadb 中做余弦检索，
并按命中 id 回映射溯源坐标（doc/section_path/start_block/end_block）。

运行：.venv_rag311\\Scripts\\python.exe -m scripts.query_retrieval "问题" [-k 5] [--reindex]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import chromadb
from chromadb.config import Settings

from embedding.dashscope_embedder import DashScopeEmbedder


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHROMA_DIR = PROJECT_ROOT / "data" / "chroma"
COLLECTION = "rag_chunks"

# 内置几条可运行的默认问题（按文档类型覆盖）
DEFAULT_QUERIES = [
    "仓库堆垛机验收有哪些技术要求？",
    "公务用车管理的适用范围是什么？",
    "充电桩报装需要说明哪些内容？",
    "燃气轮机运行注意事项有哪些？",
]


def query_collection(query_text, embedder, col, k=5):
    qvec = embedder.embed_one(query_text)
    return col.query(query_embeddings=[qvec], n_results=k)


def show(rows, query_text):
    print(f"\n查询: {query_text}")
    print("-" * 70)
    for i, (dist, cid, doc, meta) in enumerate(rows, 1):
        score = 1 - dist if dist is not None else None  # cosine：1-dist 即相似度
        simp = f"{score:.4f}" if score is not None else "?"
        print(f"[{i}] id={cid}  相似度={simp}")
        print(f"    doc={meta.get('doc')}  section={meta.get('section_path')}  "
              f"blocks=[{meta.get('start_block')},{meta.get('end_block')}]  "
              f"page=[{meta.get('start_page', 0)}~{meta.get('end_page', 0)}]")
        snippet = (doc or "")[:80].replace("\n", " ")
        print(f"    text: {snippet}...")


def main():
    ap = argparse.ArgumentParser(description="chromadb 检索冒烟测试")
    ap.add_argument("queries", nargs="*", help="要检索的问题；缺省用内置默认问题")
    ap.add_argument("-k", type=int, default=5, help="top-k（默认 5）")
    args = ap.parse_args()
    queries = args.queries or DEFAULT_QUERIES

    embedder = DashScopeEmbedder()
    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False),
    )
    col = client.get_or_create_collection(COLLECTION)
    print(f"集合 '{COLLECTION}' 当前共 {col.count()} 行，模型 {embedder.model}")

    for q in queries:
        rows = query_collection(q, embedder, col, args.k)
        # 组装成 (distance, id, document, metadata)
        zipped = list(zip(rows["distances"][0], rows["ids"][0],
                          rows["documents"][0], rows["metadatas"][0]))
        show(zipped, q)


if __name__ == "__main__":
    main()
