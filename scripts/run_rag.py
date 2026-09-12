# -*- coding: utf-8 -*-
"""
端到端 RAG 生成（流程五）：问题 -> 检索 top-k chunk -> 阈值拒答 -> 拼上下文 -> 调用千问生成带引用的回答。

流程：
  1) 向量化问题并检索 top-k chunk（路线：hybrid_rerank 混合候选+重排 / vector_only 纯向量）；
  2) top-1 分数低于该路线阈值 -> 拒答（不调生成）；否则拼接编号上下文；
  3) 让生成模型(默认 qwen3.8-max)仅依据上下文作答，并标注引用来源。

运行：.venv_rag311\\Scripts\\python.exe -m scripts.run_rag "问题" [--mode hybrid_rerank] [-k 5] [--min-score 0.58]
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import requests

from embedding.dashscope_embedder import DashScopeEmbedder, load_env, _get_env
from scripts._common import get_collection, PROJECT_ROOT, preflight, print_preflight
from retrieval.search import make_search

# 默认生成模型（可 --model 覆盖）
DEFAULT_CHAT_MODEL = "qwen3.8-max"
CHAT_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"

# 内置默认问题（-k 之外可缺省）
DEFAULT_QUESTION = "8月12日维护后开启了哪些新活动？"

# 默认检索路线：混合候选 + 重排（拒答更准，见下阈值）
DEFAULT_MODE = "hybrid_rerank"

# 拒答阈值（top-1 分数低于此值 => 拒答，不调生成）。**两路线分数尺度不同，各配一个**，
# 均由 scripts/calibrate_threshold.py 在当前库(605 chunk, qwen3.7-text-embedding-flash)+新评测集(50题)上校准：
#   vector_only  ：可答 min=0.568 / 不可答 max=0.676，两分布重叠，无法干净分割；
#                  取 0.55 -> 误杀 0 题、漏拦 3 题（不可答题交由模型自保）。
#   hybrid_rerank：可答 min=0.621 / 不可答 max=0.535，**分离开**；取 0.58（区间中点）-> 误杀 0、漏拦 0。
MIN_SCORE_DEFAULT = {"vector_only": 0.55, "hybrid_rerank": 0.58}

# 阈值拒答时的固定回答（此时**不调用生成模型**）：检索未找到相关内容，直接返回该文案，
# 而非留空或抛错。与模型级拒答区分——后者是检索到了资料但资料未写明，措辞由 SYSTEM_PROMPT 规定为
# 「资料中没有相关信息」。
REJECT_ANSWER = "当前资料相关性不足，无法回答"

# 生成调用重试（防止瞬态网络抖动污染产物）：
# 单次读超时/连接错误若直接失败，会被上层记成「拒答 + 空检索」，使本可答对的题被计 0 分、
# 拉低整批指标。此处对网络类错误做指数退避重试。
CHAT_MAX_RETRIES = 3          # 首次之外再重试 3 次（共 4 次尝试）
CHAT_BACKOFF_BASE = 2.0       # 退避基数(秒)：2、4、8


def chat(messages, model=DEFAULT_CHAT_MODEL, max_tokens=1024, temperature=0.3, timeout=90,
         max_retries=CHAT_MAX_RETRIES):
    """调用 DashScope 聊天模型。messages 形如 [{'role':..,'content':..}]。

    网络类错误（超时/连接错误）与 HTTP 429/5xx 自动重试（指数退避）；
    4xx（除 429）属请求本身问题，不重试。重试耗尽抛 RuntimeError。
    """
    load_env()
    key = _get_env("DASHSCOPE_API_KEY")
    if not key:
        raise RuntimeError("未配置 DASHSCOPE_API_KEY")
    body = {"model": model, "messages": messages, "max_tokens": max_tokens,
            "temperature": temperature}
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    last_err = None
    for attempt in range(max_retries + 1):
        try:
            resp = requests.post(CHAT_URL, headers=headers, json=body, timeout=timeout)
            if resp.status_code == 200:
                return resp.json()["choices"][0]["message"]["content"]
            last_err = f"HTTP {resp.status_code}: {resp.text[:200]}"
            if 400 <= resp.status_code < 500 and resp.status_code != 429:
                break                      # 请求本身有问题，重试无意义
        except requests.RequestException as exc:   # 超时/连接错误等
            last_err = f"{type(exc).__name__}: {exc}"
        if attempt < max_retries:
            time.sleep(CHAT_BACKOFF_BASE * (2 ** attempt))
    raise RuntimeError(f"DashScope 生成失败（含重试 {max_retries} 次）: {last_err}")


# 系统提示：**规则集中定义于此**，run_rag / gen_answers（以及未来前端）共用，
# 避免同一份指令在多处各写一份导致措辞漂移。user 消息只承载「资料 + 问题」。
SYSTEM_PROMPT = (
    "你是一个严谨的知识问答助手。请遵守以下规则：\n"
    "1) 只依据【资料】中的内容作答，不得引入资料之外的知识；\n"
    "2) 资料不足以回答时，如实说明「资料中没有相关信息」，不要猜测或编造；\n"
    "3) 每个结论后用所依据资料片段的编号标注，格式如 [chunk_000023]；"
    "只允许使用【资料】中出现过的编号。"
)

# 引用解析：chunk id 形如 chunk_000023（全局唯一，可脱离本次检索结果回溯）
CITATION_RE = re.compile(r"\[(chunk_\d{6})\]")


def build_context(hits):
    """把 top-k chunk 拼成上下文（**以 chunk id 为编号**），附来源附录（含页码，可回溯原文）。

    以 chunk id 编号而非检索结果内的位置编号：位置编号只在本次检索结果内有效，
    换问题/换路线/换库即失效，回答文本不自包含；chunk id 全局唯一，脱离检索结果也能定位原文。
    """
    lines = []
    for h in hits:
        meta = h["meta"]
        pg = f" p.{meta.get('start_page')}~{meta.get('end_page')}" if meta.get("start_page") else ""
        lines.append(f"[{h['id']}] 来源:{meta.get('doc')}"
                     f" §{meta.get('section_path') or '全文'}{pg}")
        lines.append(h["text"])
        lines.append("")
    appendix = "引用来源: " + "; ".join(
        f"[{h['id']}] {h['meta'].get('doc')} §{h['meta'].get('section_path') or '全文'}"
        + (f" (p.{h['meta'].get('start_page')}~{h['meta'].get('end_page')})"
           if h['meta'].get('start_page') else "")
        for h in hits)
    return "\n".join(lines), appendix


def build_messages(question, hits):
    """组装生成用 messages：规则放 system，数据+任务放 user（单一提示词来源）。"""
    context, _ = build_context(hits)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"【资料】\n{context}\n\n【问题】{question}"},
    ]


def extract_citations(answer, hits):
    """机械校验回答里的 chunk id 引用。

    返回 (有效引用列表, 无效引用列表, 是否完全没有引用)。
    「无效」= 引用了不在本次【资料】里的编号（模型幻觉/越界）——LLM judge 未必能发现，故做硬校验。
    """
    allowed = {h["id"] for h in hits}
    cited = CITATION_RE.findall(answer or "")
    valid = [c for c in cited if c in allowed]
    invalid = [c for c in cited if c not in allowed]
    return valid, invalid, (len(cited) == 0)


def main():
    ap = argparse.ArgumentParser(description="端到端 RAG 生成")
    ap.add_argument("question", nargs="*", help="要回答的问题；缺省用内置示例")
    ap.add_argument("--mode", default=DEFAULT_MODE, choices=["vector_only", "hybrid_rerank"],
                    help=f"检索路线（默认 {DEFAULT_MODE}）")
    ap.add_argument("-k", type=int, default=5, help="检索 top-k（默认 5，覆盖 complex 集多来源上限）")
    ap.add_argument("--model", default=DEFAULT_CHAT_MODEL, help=f"生成模型（默认 {DEFAULT_CHAT_MODEL}）")
    ap.add_argument("--min-score", type=float, default=None,
                    help="拒答阈值；缺省按路线取校准值 "
                         f"(vector_only={MIN_SCORE_DEFAULT['vector_only']}, "
                         f"hybrid_rerank={MIN_SCORE_DEFAULT['hybrid_rerank']})")
    args = ap.parse_args()
    question = " ".join(args.question) or DEFAULT_QUESTION
    min_score = args.min_score if args.min_score is not None else MIN_SCORE_DEFAULT[args.mode]

    # 前置检查：语料/索引/切片产物不入库（.gitignore），新克隆下直接跑会抛难懂的底层异常
    if not print_preflight(preflight(args.mode)):
        return

    embedder = DashScopeEmbedder()
    col = get_collection()
    search = make_search(args.mode, embedder, col, PROJECT_ROOT / "data" / "chunks")

    hits = search(question, args.k)
    print(f"问题: {question}\n路线: {args.mode}  检索 top-{args.k}:")
    for h in hits:
        print(f"  - [{h['id']}] 分数={h['score']:.3f}  {h['text'][:46]}...")
    print("-" * 78)

    # 阈值拒答：top-1 分数低于阈值 => 拒答，不调生成 API
    top = hits[0]["score"] if hits else None
    if top is None or top < min_score:
        print(f"\n{REJECT_ANSWER}")
        print(f"（拒答原因：top-1 分数="
              f"{'无命中' if top is None else f'{top:.3f}'} < 阈值 {min_score}，未调用生成。）")
        if hits:
            print(f"  最接近片段: {hits[0]['text'][:90]}...")
        return

    # 规则集中在 SYSTEM_PROMPT；user 只承载「资料 + 问题」
    messages = build_messages(question, hits)
    try:
        answer = chat(messages, model=args.model)
    except Exception as exc:  # noqa: BLE001 交互式：失败给明确提示，不抛栈
        print(f"\n【生成失败】{exc}")
        print("（检索结果如上；这属调用侧故障，可稍后重试同一问题）")
        return
    print("\n生成回答:\n" + answer)
    print("-" * 78)
    print(appendix)
    # 引用机械校验：回答里的 chunk id 是否都来自本次【资料】
    valid, invalid, no_cite = extract_citations(answer, hits)
    print(f"引用校验: 有效 {len(valid)} 处"
          + (f"，**无效 {len(invalid)} 处**（{', '.join(sorted(set(invalid)))}）" if invalid else "")
          + ("，**未标注任何引用**" if no_cite else ""))


if __name__ == "__main__":
    main()
