# -*- coding: utf-8 -*-
"""
生成 RAG 回答并落盘（用于复盘文档）。对 data/eval/qa_*.jsonl 逐题：
  检索 top-k -> 按 min_score 阈值拒答 -> 通过则用 qwen3.8-max 生成带引用回答。
--mode vector_only | hybrid_rerank 控制检索方式。
输出 JSONL：{id,set,category,type,query,expected_doc,mode,top1,topk:[{id,doc,section,page,score}],rejected,answer}

运行：.venv_rag311\Scripts\python.exe scripts\gen_answers.py --mode vector_only --sets all --min-score 0.70 --out data/eval/answers_vector.jsonl
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import chromadb
from chromadb.config import Settings

from embedding.dashscope_embedder import DashScopeEmbedder
from scripts.run_rag import chat, build_context, DEFAULT_CHAT_MODEL

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHROMA_DIR = PROJECT_ROOT / "data" / "chroma"
EVAL_DIR = PROJECT_ROOT / "data" / "eval"
COLLECTION = "rag_chunks"


def load_queries(sets):
    qs = []
    for fn in ("qa_basic.jsonl", "qa_complex.jsonl"):
        p = EVAL_DIR / fn
        if not p.exists():
            continue
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                q = json.loads(line)
                if sets == "all" or q.get("set") == sets:
                    qs.append(q)
    return qs


def make_search(mode, embedder, col):
    if mode == "vector_only":
        def search(q, k):
            qvec = embedder.embed_one(q)
            res = col.query(query_embeddings=[qvec], n_results=k)
            out = []
            for dist, cid, doc, meta in zip(res["distances"][0], res["ids"][0],
                                            res["documents"][0], res["metadatas"][0]):
                out.append({"id": cid, "score": 1 - dist, "text": doc, "meta": meta})
            return out
        return search
    from retrieval.hybrid import build_hybrid_search
    from retrieval.reranker import DashScopeReranker
    hs = build_hybrid_search(col, embedder, PROJECT_ROOT / "data" / "chunks")
    rr = DashScopeReranker()

    def search(q, k):
        cand = hs.search(q, k=max(20, k + 6), rerank=False)
        return rr.rerank(q, cand, k)
    return search


def gen_answer(query, hits, model, min_score):
    top1 = hits[0]["score"] if hits else None
    if top1 is None or top1 < min_score:
        return {"top1": top1, "rejected": True, "answer": "", "hits": hits}
    context, appendix = build_context(hits)
    prompt = (
        f"你是一个知识问答助手。请严格依据下方【资料】回答用户问题。\n"
        f"若资料不足以回答，请如实说明「资料中没有相关信息」。\n"
        f"回答时用 {{1}}、{{2}} 标注引用到的资料编号。\n\n"
        f"【资料】\n{context}\n\n【问题】{query}\n")
    messages = [{"role": "system",
                 "content": "你是一个严谨的知识问答助手，回答需忠实于所给资料并标注引用。"},
                {"role": "user", "content": prompt}]
    answer = chat(messages, model=model)
    return {"top1": top1, "rejected": False, "answer": answer, "hits": hits}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="vector_only", choices=["vector_only", "hybrid_rerank"])
    ap.add_argument("--sets", default="all", choices=["all", "basic", "complex"])
    ap.add_argument("--min-score", type=float, default=0.70)
    ap.add_argument("--model", default=DEFAULT_CHAT_MODEL)
    ap.add_argument("--out", required=True)
    ap.add_argument("-k", type=int, default=3)
    args = ap.parse_args()

    queries = load_queries(args.sets)
    embedder = DashScopeEmbedder()
    client = chromadb.PersistentClient(path=str(CHROMA_DIR), settings=Settings(anonymized_telemetry=False))
    col = client.get_or_create_collection(COLLECTION)
    search = make_search(args.mode, embedder, col)

    results = []
    for q in queries:
        try:
            hits = search(q["query"], args.k)
            r = gen_answer(q["query"], hits, args.model, args.min_score)
        except Exception as exc:  # noqa: BLE001
            r = {"top1": None, "rejected": True, "answer": f"[生成失败] {exc}", "hits": []}
        topk = []
        for h in r.get("hits", []):
            meta = h["meta"]
            topk.append({"id": h["id"], "doc": meta.get("doc"),
                         "section": meta.get("section_path"), "page": [meta.get("start_page"), meta.get("end_page")],
                         "score": round(h["score"], 3)})
        results.append({
            "id": q["id"], "set": q["set"], "category": q["category"], "type": q["type"],
            "query": q["query"], "expected_doc": q.get("expected_doc", ""), "mode": args.mode,
            "top1": r["top1"], "topk": topk, "rejected": r["rejected"], "answer": r["answer"],
        })
        print(f"[{q['id']}] {'拒答' if r['rejected'] else '回答'}  top1={r['top1']}")

    Path(args.out).write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in results) + "\n", encoding="utf-8")
    print(f"\n完成，共 {len(results)} 题，见 {args.out}")


if __name__ == "__main__":
    main()
