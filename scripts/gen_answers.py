# -*- coding: utf-8 -*-
"""
生成 RAG 回答并落盘（供生成侧评测使用）。对新评测集 test/qa_dataset/（simple 30 + complex 20）逐题：
  检索 top-k -> 按 min_score 阈值拒答 -> 通过则用 qwen3.8-max 生成带引用回答。
两条检索路线：--mode vector_only | hybrid_rerank。k 按集取（simple 3 / complex 5）。

输出 JSONL：{id,set,question,gold_answer,answerable,sources,mode,top1,topk:[{id,doc,section,page,score}],rejected,
              answer,citations,error}（citations 为 chunk id 引用的机械校验结果；error 非空表示该题生成失败、评测时跳过）

运行：.venv_rag311\\Scripts\\python.exe scripts\\gen_answers.py --mode hybrid_rerank --sets all --out data\\eval\\answers_v2_rerank.jsonl
（阈值缺省按路线取校准值：vector_only=0.55 / hybrid_rerank=0.58）
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from embedding.dashscope_embedder import DashScopeEmbedder
from scripts._common import get_collection, load_qa_dataset, PROJECT_ROOT, QA_K
from scripts.run_rag import (chat, build_messages, extract_citations,
                             DEFAULT_CHAT_MODEL, MIN_SCORE_DEFAULT, REJECT_ANSWER)
from retrieval.search import make_search


def gen_answer(query, hits, model, min_score):
    """阈值拒答 -> 通过则生成带引用回答。规则与 prompt 结构统一来自 run_rag（单一来源）。"""
    top1 = hits[0]["score"] if hits else None
    if top1 is None or top1 < min_score:
        # 拒答：不调生成，回答固定为 REJECT_ANSWER（明确告知相关性不足，而非留空）
        return {"top1": top1, "rejected": True, "answer": REJECT_ANSWER, "hits": hits, "citations": None}
    messages = build_messages(query, hits)
    answer = chat(messages, model=model)
    valid, invalid, no_cite = extract_citations(answer, hits)
    return {"top1": top1, "rejected": False, "answer": answer, "hits": hits,
            "citations": {"valid": len(valid), "invalid": len(invalid), "none": no_cite}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="vector_only", choices=["vector_only", "hybrid_rerank"])
    ap.add_argument("--sets", default="all", choices=["all", "simple", "complex"])
    ap.add_argument("--min-score", type=float, default=None,
                    help="拒答阈值；缺省按路线取校准值（见 run_rag.MIN_SCORE_DEFAULT）")
    ap.add_argument("--model", default=DEFAULT_CHAT_MODEL)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 题（0=全部，用于冒烟）")
    args = ap.parse_args()
    min_score = args.min_score if args.min_score is not None else MIN_SCORE_DEFAULT[args.mode]

    queries = load_qa_dataset(args.sets)
    if args.limit:
        queries = queries[:args.limit]
    embedder = DashScopeEmbedder()
    col = get_collection()
    search = make_search(args.mode, embedder, col, PROJECT_ROOT / "data" / "chunks")

    results = []
    for q in queries:
        k = QA_K[q["set"]]
        error = None
        hits = []
        try:
            hits = search(q["question"], k)
            r = gen_answer(q["question"], hits, args.model, min_score)
        except Exception as exc:  # noqa: BLE001
            # 调用侧故障（重试后仍失败）：**保留已检索到的 hits**，并显式记 error。
            # 不可伪装成「阈值拒答」——否则评测会把基础设施故障当成 0 分计入指标。
            error = f"{type(exc).__name__}: {exc}"
            r = {"top1": (hits[0]["score"] if hits else None),
                 "rejected": False, "answer": "", "hits": hits, "citations": None}
        topk = []
        for h in r.get("hits", []):
            meta = h["meta"]
            pg = [meta.get("start_page"), meta.get("end_page")]
            topk.append({"id": h["id"], "doc": meta.get("doc"),
                         "section": meta.get("section_path"),
                         "page": pg if pg[0] else None,
                         "score": round(h["score"], 3)})
        results.append({
            "id": q["id"], "set": q["set"], "question": q["question"],
            "gold_answer": q["answer"], "answerable": q["answerable"],
            "sources": q["sources"], "mode": args.mode,
            "top1": r["top1"], "topk": topk, "rejected": r["rejected"], "answer": r["answer"],
            "citations": r.get("citations"),   # chunk id 引用的机械校验结果
            "error": error,
        })
        if error:
            print(f"[{q['id']}] 生成失败(已标记 error，不计入指标)  {error[:80]}")
        else:
            flag = "拒答" if r["rejected"] else "回答"
            print(f"[{q['id']}] {flag}  top1={r['top1']}")

    Path(args.out).write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in results) + "\n", encoding="utf-8")
    n_err = sum(1 for x in results if x.get("error"))
    print(f"\n完成，共 {len(results)} 题（其中生成失败 {n_err} 题），见 {args.out}")


if __name__ == "__main__":
    main()
