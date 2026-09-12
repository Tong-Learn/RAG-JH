# -*- coding: utf-8 -*-
"""
检索评测指标（纯函数，无 API、无 IO，便于单测）。

背景：新的评测集 test/qa_dataset 里，complex 集的**每题有 1~4 个来源文档**（sources），
所以「命中」不再是"唯一正确答案"，而是"命中来源集合"。本模块把指标定义成对多来源也成立的形式：

对一题（来源集合 S，检索返回文档序列 D，截取前 k 个 D_k）：
  - recall@k    = |unique(命中来源)| / |S|        （来源覆盖率；同来源多 chunk 只算一次）
  - precision@k = |{d in D_k : d in S}| / k       （top-k 中属于来源集的 chunk 占比）
  - P@1         = 1 if D_1 in S else 0            （多来源下仍成立：top-1 是否相关）
  - MRR         = 1 / (第一个属于 S 的文档的排名)  （没有则 0）

单来源题（|S| = 1）时，recall@k 退化为原来的 0/1 命中，因此两集可用同一套函数。
不可答题（S 为空）不参与这些指标，只在拒答统计里用。
"""
from typing import Optional, Sequence


def score_question(retrieved_docs: Sequence[str], sources: Sequence[str], k: int) -> Optional[dict]:
    """对单题打分。sources 为空（不可答）时返回 None。"""
    s = {x for x in sources if x}
    if not s:
        return None
    docs = list(retrieved_docs)[:k]
    hit_docs = [d for d in docs if d in s]
    recall = len(set(hit_docs)) / len(s)
    precision = len(hit_docs) / k if k else 0.0
    p1 = 1.0 if (docs and docs[0] in s) else 0.0
    mrr = 0.0
    for i, d in enumerate(docs):
        if d in s:
            mrr = 1.0 / (i + 1)
            break
    return {"recall": recall, "precision": precision, "p1": p1, "mrr": mrr,
            "n_sources": len(s), "n_hit": len(set(hit_docs))}


def aggregate(scores: Sequence[dict], keys: Sequence[str]) -> dict:
    """对多题打分做宏平均。scores 里可含 None（不可答），自动跳过。"""
    vals = [s for s in scores if s]
    out = {"n": len(vals)}
    for key in keys:
        nums = [s[key] for s in vals if s.get(key) is not None]
        out[key] = round(sum(nums) / len(nums), 3) if nums else None
    return out


def any_hit_rate(scores: Sequence[dict]) -> Optional[float]:
    """宽松下界：至少命中 1 个来源的题目占比。"""
    vals = [s for s in scores if s]
    if not vals:
        return None
    return round(sum(1 for s in vals if s["n_hit"] > 0) / len(vals), 3)


def all_hit_rate(scores: Sequence[dict]) -> Optional[float]:
    """严格上界：来源集被完全覆盖的题目占比（k 小于来源数时必然为 0）。"""
    vals = [s for s in scores if s]
    if not vals:
        return None
    return round(sum(1 for s in vals if s["n_hit"] == s["n_sources"]) / len(vals), 3)
