# -*- coding: utf-8 -*-
"""
检索冒烟测试（流程四前置）：对给定问题用同一 embedding 模型向量化，在 chromadb 中做余弦检索，
并按命中 id 回映射溯源坐标（doc/section_path/start_block/end_block/start/end_page）。

运行：.venv_rag311\\Scripts\\python.exe -m scripts.query_retrieval "问题" [-k 5]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from embedding.dashscope_embedder import DashScopeEmbedder
from scripts._common import get_collection, COLLECTION
from retrieval.search import vector_search

# 内置几条可运行的默认问题（按文档类型覆盖）
DEFAULT_QUERIES = [
    "「逐猎狂途」是几人制的娱乐竞技玩法？",
    "「征战之塔」共有几档难度？",
    "「辉月织梦」礼包售价多少欧珀/晶珀？",
    "「铸骨迷巢」团本的奖励刷新时间是什么时候？",
]


def show(hits, query_text):
    print(f"\n查询: {query_text}")
    print("-" * 70)
    for i, h in enumerate(hits, 1):
        meta = h["meta"]
        print(f"[{i}] id={h['id']}  相似度={h['score']:.4f}")
        print(f"    doc={meta.get('doc')}  section={meta.get('section_path')}  "
              f"blocks=[{meta.get('start_block')},{meta.get('end_block')}]  "
              f"page=[{meta.get('start_page', 0)}~{meta.get('end_page', 0)}]")
        snippet = (h["text"] or "")[:80].replace("\n", " ")
        print(f"    text: {snippet}...")


def main():
    ap = argparse.ArgumentParser(description="chromadb 检索冒烟测试")
    ap.add_argument("queries", nargs="*", help="要检索的问题；缺省用内置默认问题")
    ap.add_argument("-k", type=int, default=5, help="top-k（默认 5）")
    args = ap.parse_args()
    queries = args.queries or DEFAULT_QUERIES

    embedder = DashScopeEmbedder()
    col = get_collection()
    print(f"集合 '{COLLECTION}' 当前共 {col.count()} 行，模型 {embedder.model}")

    for q in queries:
        show(vector_search(embedder, col, q, args.k), q)


if __name__ == "__main__":
    main()
