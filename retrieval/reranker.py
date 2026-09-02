# -*- coding: utf-8 -*-
"""
DashScope 文本重排（rerank）。用 qwen3.7-text-rerank 对候选 chunk 重排，返回 relevance 最高的 top-k。

背景（见 docs/故障报告.md）：本机 torch/onnxruntime 有崩溃前科，故重排走 DashScope API，而非本地 bge-reranker。
"""
import requests

from embedding.dashscope_embedder import load_env, _get_env

# DashScope 文本重排端点（native Rerank API）
RERANK_URL = "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank"
DEFAULT_RERANK_MODEL = "qwen3.7-text-rerank"


class DashScopeReranker:
    def __init__(self, model=DEFAULT_RERANK_MODEL, top_n=20, timeout=60):
        load_env()
        self.model = model
        self.api_key = _get_env("DASHSCOPE_API_KEY")
        self.top_n = top_n
        self.timeout = timeout
        if not self.api_key:
            raise RuntimeError("未配置 DASHSCOPE_API_KEY")

    def rerank(self, query, candidates, k=4):
        if not candidates:
            return []
        docs = [{"text": c["text"]} for c in candidates] if isinstance(candidates[0], dict) else candidates
        body = {
            "model": self.model,
            "input": {"query": query, "documents": docs},
            "parameters": {"top_n": max(k, min(len(docs), self.top_n))},
        }
        resp = requests.post(RERANK_URL, headers={"Authorization": f"Bearer {self.api_key}",
                                                 "Content-Type": "application/json"},
                             json=body, timeout=self.timeout)
        if resp.status_code != 200:
            raise RuntimeError(f"rerank 失败: HTTP {resp.status_code}: {resp.text[:300]}")
        results = resp.json()["output"]["results"]
        ranked = sorted(results, key=lambda r: -r["relevance_score"])
        out = []
        for r in ranked[:k]:
            idx = r["index"]
            if 0 <= idx < len(candidates):
                out.append(candidates[idx])
        return out
