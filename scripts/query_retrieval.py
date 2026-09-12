# -*- coding: utf-8 -*-
"""
检索冒烟（人工排查用）：对给定问题做检索，打印 top-k 命中的**分数与完整溯源坐标**
（doc / section_path / start_block~end_block / page），便于人眼核对"命中是否合理、坐标能否回溯原文"。

⚠️ 注意：本脚本是**人工冒烟**，无断言、不产指标；指标请用 scripts.eval_retrieval.py。
   默认路线与 run_rag 一致（hybrid_rerank = 混合候选 + 重排）；用 --mode vector_only 可切纯向量。

运行：.venv_rag311\\Scripts\\python.exe -m scripts.query_retrieval "问题" [-k 5] [--mode hybrid_rerank|vector_only]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from embedding.dashscope_embedder import DashScopeEmbedder
from scripts._common import get_collection, COLLECTION, PROJECT_ROOT, preflight, print_preflight
from scripts.run_rag import DEFAULT_MODE
from retrieval.search import make_search

# 内置几条可运行的默认问题（按文档类型覆盖）
DEFAULT_QUERIES = [
    "「逐猎狂途」是几人制的娱乐竞技玩法？",
    "「征战之塔」共有几档难度？",
    "「辉月织梦」礼包售价多少欧珀/晶珀？",
    "「铸骨迷巢」团本的奖励刷新时间是什么时候？",
]


def show(hits, query_text, mode):
    print(f"\n查询: {query_text}   [路线 {mode}]")
    print("-" * 70)
    if not hits:
        print("  （无命中）")
        return
    for i, h in enumerate(hits, 1):
        meta = h["meta"]
        print(f"[{i}] id={h['id']}  分数={h['score']:.4f}")
        print(f"    doc={meta.get('doc')}  section={meta.get('section_path')}  "
              f"blocks=[{meta.get('start_block')},{meta.get('end_block')}]  "
              f"page=[{meta.get('start_page', 0)}~{meta.get('end_page', 0)}]")
        snippet = (h["text"] or "")[:80].replace("\n", " ")
        print(f"    text: {snippet}...")


def main():
    ap = argparse.ArgumentParser(description="检索冒烟（人工核对命中与溯源坐标）")
    ap.add_argument("queries", nargs="*", help="要检索的问题；缺省用内置默认问题")
    ap.add_argument("-k", type=int, default=5, help="top-k（默认 5）")
    ap.add_argument("--mode", default=DEFAULT_MODE, choices=["vector_only", "hybrid_rerank"],
                    help=f"检索路线（默认 {DEFAULT_MODE}，与 run_rag 一致）")
    args = ap.parse_args()
    queries = args.queries or DEFAULT_QUERIES

    # 前置检查：语料/索引不入库，新克隆下先给出构建指引（与 run_rag 行为一致）
    if not print_preflight(preflight(args.mode)):
        return

    embedder = DashScopeEmbedder()
    col = get_collection()
    print(f"集合 '{COLLECTION}' 当前共 {col.count()} 行，模型 {embedder.model}，路线 {args.mode}")

    search = make_search(args.mode, embedder, col, PROJECT_ROOT / "data" / "chunks")
    for q in queries:
        show(search(q, args.k), q, args.mode)


if __name__ == "__main__":
    main()
