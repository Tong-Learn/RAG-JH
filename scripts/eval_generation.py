# -*- coding: utf-8 -*-
"""
生成侧评测：对 gen_answers.py 的产出逐题打分（LLM-as-judge，默认判分模型 qwen3.7-flash，与生成模型分开）。

四个指标（0 / 0.5 / 1，各附理由）：
  1) faithfulness 忠实度 ：生成答案是否严格基于**检索到的资料**、无编造。
  2) citation 引用准确率 ：答案里的 [chunk_xxxxxx] 引用是否指向能支撑该句的片段。
  3) answer_accuracy 答案正确率 ：生成答案与 **gold answer** 是否语义一致（以 gold 为基准）。
  4) rejection 拒答正确性 ：不可答题是否拒答（且不编造）；可答题是否未误拒。
（按决策：不用 evidence 打分，只用生成的答案 + gold answer + 检索上下文。）

另可 `--no-judge` 只产出人工标注模板（不调 API）。

运行：
  .venv_rag311\\Scripts\\python.exe scripts\\eval_generation.py --mode vector_only --answers data\\eval\\answers_vector_only.jsonl --out data\\eval\\gen_scores_vector_only.jsonl
（--answers 缺省即 data/eval/answers_{mode}.jsonl，也可省略）
"""
import argparse
import json
import re
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts._common import PROJECT_ROOT, EVAL_DIR
from scripts.run_rag import chat

# 默认判分模型（可 --model 覆盖）：与生成模型**分开配置**，避免"同模型自评"偏差；
# 生成默认 qwen-flash（见 run_rag.DEFAULT_CHAT_MODEL）。
DEFAULT_JUDGE_MODEL = "qwen3.7-flash"

SCORE_KEYS = ["faithfulness", "citation", "answer_accuracy", "rejection"]
SCORE_LABEL = {"faithfulness": "忠实度", "citation": "引用准确率",
               "answer_accuracy": "答案正确率", "rejection": "拒答正确性"}

# chunk id -> text 缓存（judge 需看检索片段正文；answers 的 topk 只存 id 不存 text）
_CHUNK_TEXT = {}

_JUDGE_PROMPT = """你是严谨的 RAG 生成质量评审。请依据下方信息，对四个维度各打 0 / 0.5 / 1 分并各给一句理由。

【问题】{query}
【标准答案 gold】{gold}
【该题是否可答】{answerable}   （不可答时，标准答案是拒答，见下）
【检索到的资料】
{context}
【系统生成的回答】{answer}

说明：系统有两类拒答措辞，均视为"正确拒答"，不要因为措辞不同扣分：
  · 阈值拒答（检索相关性不足，未调生成）："当前资料相关性不足，无法回答"
  · 模型级拒答（资料存在但未提及）："资料中没有相关信息"

评分标准：
- faithfulness 忠实度：回答是否忠于【检索到的资料】、无编造？资料未提却断言 = 0；基本忠于但有轻微扩写 = 0.5；完全有据 = 1。
- citation 引用准确率：回答中以 [chunk_xxxxxx] 形式标注的引用，是否指向能支撑对应句子的片段？未标注或错位 = 0；部分对应 = 0.5；准确 = 1。
  （另有机械校验：引用的编号必须出现在【检索到的资料】里，越界/编造编号由脚本单独统计为"无效引用"。）
- answer_accuracy 答案正确率：与【标准答案】语义是否一致？明显错误/缺失关键信息 = 0；部分正确 = 0.5；一致 = 1。
  （不可答题：表达了上述任一拒答措辞 = 1；若强行编造 = 0。）
- rejection 拒答正确性：不可答题正确拒答 = 1，未拒而编造 = 0；可答题正常作答 = 1，误拒 = 0。

只输出 JSON，不要额外文字：
{{"faithfulness": <分>, "faithfulness_reason": "…",
  "citation": <分>, "citation_reason": "…",
  "answer_accuracy": <分>, "answer_accuracy_reason": "…",
  "rejection": <分>, "rejection_reason": "…",
  "notes": "…"}}"""


def _load_chunk_text():
    m = {}
    cd = PROJECT_ROOT / "data" / "chunks"
    if cd.exists():
        for f in cd.glob("*.chunks.jsonl"):
            for line in f.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    o = json.loads(line)
                    m[o["id"]] = o.get("text", "")
    return m


def _build_ctx(topk):
    """重建 judge 看的检索资料。**以 chunk id 为编号**，与回答中的引用格式一致，
    便于 judge 核对 [chunk_xxxxxx] 是否指向能支撑该句的片段。"""
    lines = []
    for h in topk:
        txt = _CHUNK_TEXT.get(h.get("id"), "")
        pg = h.get("page")
        pgs = f"~p.{pg}" if pg else ""
        lines.append(f"[{h.get('id')}] {h.get('doc')} §{h.get('section')} {pgs}\n{txt}")
    return "\n\n".join(lines) or "（无检索资料）"


def judge_one(query, gold, answerable, answer, topk, model):
    prompt = _JUDGE_PROMPT.format(query=query, gold=gold or "（无）",
                                  answerable="可答" if answerable else "不可答",
                                  context=_build_ctx(topk),
                                  answer=answer or "（系统拒答，无回答）")
    msgs = [{"role": "system", "content": "你是 RAG 生成质量评审，只输出 JSON。"},
            {"role": "user", "content": prompt}]
    try:
        raw = chat(msgs, model=model, temperature=0.0)
        m = re.search(r"\{.*\}", raw, re.S)
        return json.loads(m.group(0)) if m else None
    except Exception:  # noqa: BLE001
        return None


def _avg(vals):
    nums = [v for v in vals if isinstance(v, (int, float))]
    return round(statistics.mean(nums), 3) if nums else None


def main():
    ap = argparse.ArgumentParser(description="生成侧评测（4 指标 LLM-judge）")
    ap.add_argument("--answers", default=None,
                    help="回答文件；缺省 data/eval/answers_{mode}.jsonl"
                         "（即 answers_vector_only.jsonl / answers_hybrid_rerank.jsonl）")
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", default="vector_only", choices=["vector_only", "hybrid_rerank"])
    ap.add_argument("--model", default=DEFAULT_JUDGE_MODEL,
                    help=f"判分模型（默认 {DEFAULT_JUDGE_MODEL}）")
    ap.add_argument("--no-judge", action="store_true", help="只产出人工标注模板，不调 API")
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 题（0=全部）")
    args = ap.parse_args()

    ans_path = Path(args.answers) if args.answers else EVAL_DIR / f"answers_{args.mode}.jsonl"
    if not ans_path.exists():
        print(f"[错误] 找不到 {ans_path}，请先用 gen_answers.py 生成回答。")
        return

    records = [json.loads(l) for l in ans_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    if args.limit:
        records = records[:args.limit]
    global _CHUNK_TEXT
    _CHUNK_TEXT = _load_chunk_text()
    print(f"读入 {len(records)} 条回答：{ans_path.name}；chunk 文本 {len(_CHUNK_TEXT)} 条")

    out_records = []
    n_infra = 0
    for r in records:
        e = {k: r.get(k) for k in ("id", "set", "question", "gold_answer", "answerable",
                                   "sources", "top1", "rejected", "answer")}
        e["mode"] = args.mode
        e["topk"] = r.get("topk", [])
        e["error"] = r.get("error")
        if r.get("error"):
            # 调用侧故障（非模型行为）：不打分、不计入指标，单独计数并如实报告。
            e["judge"] = None
            e["_infra_failed"] = True
            n_infra += 1
            print(f"[{r['id']}] 跳过打分（调用侧故障，不计入指标）：{str(r['error'])[:70]}")
        elif args.no_judge:
            e["judge"] = None
            e["_template"] = True
        else:
            j = judge_one(r.get("question"), r.get("gold_answer"), r.get("answerable", True),
                          r.get("answer"), e["topk"], args.model)
            e["judge"] = j or {"error": "judge 失败"}
            if j:
                print(f"[{r['id']}] 忠实={j.get('faithfulness')} 引用={j.get('citation')} "
                      f"正确={j.get('answer_accuracy')} 拒答={j.get('rejection')}")
            else:
                print(f"[{r['id']}] judge 失败")
        out_records.append(e)

    Path(args.out).write_text("\n".join(json.dumps(x, ensure_ascii=False) for x in out_records) + "\n",
                              encoding="utf-8")

    if n_infra:
        print(f"\n注意：{n_infra}/{len(out_records)} 题因调用侧故障被排除（不计入指标）；"
              f"正式指标应报为 n={len(out_records)-n_infra}。")

    if not args.no_judge:
        judges = [r["judge"] for r in out_records if isinstance(r.get("judge"), dict) and "error" not in r["judge"]]
        print(f"\n===== 生成侧评测（{len(judges)}/{len(out_records)} 条有效 judge）=====")
        for key in SCORE_KEYS:
            print(f"  {SCORE_LABEL[key]:<10}({key}) = {_avg([j.get(key) for j in judges])}")
        # 分集
        for setname in ("simple", "complex"):
            sub = [r["judge"] for r in out_records
                   if r.get("set") == setname and isinstance(r.get("judge"), dict) and "error" not in r["judge"]]
            if sub:
                print(f"  [{setname}] " + "  ".join(
                    f"{SCORE_LABEL[k]}={_avg([j.get(k) for j in sub])}" for k in SCORE_KEYS))
    else:
        print("\n已产出人工标注模板（未调 API）。")
    print(f"\n完成，见 {args.out}")


if __name__ == "__main__":
    main()
