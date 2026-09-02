# -*- coding: utf-8 -*-
"""
端到端 RAG 生成（流程五）：问题 -> 检索 top-k chunk -> 拼上下文 -> 调用千问生成带引用的回答。

流程：
  1) 用与入库相同的 embedding 模型向量化问题，在 chromadb 里余弦检索 top-k chunk；
  2) 把每个 chunk 编号 [1][2]...，连同 doc/section 附录拼进上下文；
  3) 让生成模型(默认 qwen-plus)仅依据上下文作答，并标注引用来源。

运行：.venv_rag311\\Scripts\\python.exe -m scripts.run_rag "问题" [-k 4] [--model qwen-plus]
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import chromadb
import requests
from chromadb.config import Settings

from embedding.dashscope_embedder import DashScopeEmbedder, load_env, _get_env

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHROMA_DIR = PROJECT_ROOT / "data" / "chroma"
COLLECTION = "rag_chunks"

# 默认生成模型（可 --model 覆盖）
DEFAULT_CHAT_MODEL = "qwen3.8-max"
CHAT_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"

# 内置默认问题（-k 之外可缺省）
DEFAULT_QUESTION = "8月12日维护后开启了哪些新活动？"

# 相似度阈值（item① 拒答）：top-1 相似度低于此值 => 拒答「资料中没有相关信息」，不调生成。
# 缺省 None 时不启用(兼容)；校准值 0.70：正类(可答)min=0.673/中位=0.804，负类(无答/无关)max=0.719/中位=0.591，
# 两分布在 0.673~0.719 有重叠，故取 0.70 平衡（拒掉 7/8 负样本，牺牲 1 道 0.673 边界正样本）。
MIN_SCORE_DEFAULT = 0.70


def chat(messages, model=DEFAULT_CHAT_MODEL, max_tokens=1024, temperature=0.3, timeout=90):
    """调用 DashScope 聊天模型。messages 形如 [{'role':..,'content':..}]。"""
    load_env()
    key = _get_env("DASHSCOPE_API_KEY")
    if not key:
        raise RuntimeError("未配置 DASHSCOPE_API_KEY")
    body = {"model": model, "messages": messages, "max_tokens": max_tokens,
            "temperature": temperature}
    resp = requests.post(CHAT_URL, headers={"Authorization": f"Bearer {key}",
                                            "Content-Type": "application/json"},
                         json=body, timeout=timeout)
    if resp.status_code != 200:
        raise RuntimeError(f"DashScope 生成失败: HTTP {resp.status_code}: {resp.text[:200]}")
    return resp.json()["choices"][0]["message"]["content"]


def retrieve(embedder, col, query, k):
    qvec = embedder.embed_one(query)
    res = col.query(query_embeddings=[qvec], n_results=k)
    out = []
    for dist, cid, doc, meta in zip(res["distances"][0], res["ids"][0],
                                    res["documents"][0], res["metadatas"][0]):
        score = 1 - dist
        out.append({"id": cid, "score": score, "text": doc, "meta": meta})
    return out


def build_context(hits):
    """把 top-k chunk 拼成编号上下文，附一段来源附录（含页码，可用于回溯原文）。"""
    lines = ["以下是检索到的相关资料，请仅依据这些资料作答："]
    for i, h in enumerate(hits, 1):
        meta = h["meta"]
        pg = f" p.{meta.get('start_page')}~{meta.get('end_page')}" if meta.get("start_page") else ""
        lines.append(f"[{i}] (片段 {h['id']}, 来源:{meta.get('doc')}"
                     f" §{meta.get('section_path') or '全文'}{pg})")
        lines.append(h["text"])
        lines.append("")
    appendix = "引用来源: " + "; ".join(
        f"[{i}] {h['meta'].get('doc')} §{h['meta'].get('section_path') or '全文'}"
        + (f" (p.{h['meta'].get('start_page')}~{h['meta'].get('end_page')})"
           if h['meta'].get('start_page') else "")
        for i, h in enumerate(hits, 1))
    return "\n".join(lines), appendix


def main():
    ap = argparse.ArgumentParser(description="端到端 RAG 生成")
    ap.add_argument("question", nargs="*", help="要回答的问题；缺省用内置示例")
    ap.add_argument("-k", type=int, default=4, help="检索 top-k（默认 4）")
    ap.add_argument("--model", default=DEFAULT_CHAT_MODEL, help=f"生成模型（默认 {DEFAULT_CHAT_MODEL}）")
    ap.add_argument("--min-score", type=float, default=MIN_SCORE_DEFAULT,
                    help=f"相似度阈值（默认 {MIN_SCORE_DEFAULT}，不填则不启用拒答）")
    args = ap.parse_args()
    question = " ".join(args.question) or DEFAULT_QUESTION

    embedder = DashScopeEmbedder()
    client = chromadb.PersistentClient(path=str(CHROMA_DIR), settings=Settings(anonymized_telemetry=False))
    col = client.get_or_create_collection(COLLECTION)

    hits = retrieve(embedder, col, question, args.k)
    print(f"问题: {question}\n检索 top-{args.k}:")
    for h in hits:
        print(f"  - [{h['id']}] 相似度={h['score']:.3f}  {h['text'][:46]}...")
    print("-" * 78)

    # item① 阈值拒答：top-1 相似度低于阈值 => 拒答，不调生成 API
    top = hits[0]["score"] if hits else None
    if args.min_score is not None and (top is None or top < args.min_score):
        print(f"【拒答】资料中没有相关信息（top-1 相似度="
              f"{'无命中' if top is None else f'{top:.3f}'} < 阈值 {args.min_score}），未调用生成。")
        if hits:
            print(f"  最接近片段: {hits[0]['text'][:90]}...")
        return

    context, appendix = build_context(hits)
    prompt = (
        f"你是一个知识问答助手。请严格依据下方【资料】回答用户问题。\n"
        f"若资料不足以回答，请如实说明「资料中没有相关信息」。\n"
        f"回答时用 {{1}}、{{2}} 标注引用到的资料编号。\n\n"
        f"【资料】\n{context}\n\n【问题】{question}\n"
    )
    messages = [
        {"role": "system", "content": "你是一个严谨的知识问答助手，回答需忠实于所给资料并标注引用。"},
        {"role": "user", "content": prompt},
    ]
    answer = chat(messages, model=args.model)
    print("\n生成回答:\n" + answer)
    print("-" * 78)
    print(appendix)


if __name__ == "__main__":
    main()
