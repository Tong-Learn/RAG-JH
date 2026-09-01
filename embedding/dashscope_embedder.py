# -*- coding: utf-8 -*-
"""
DashScope(阿里云千问) 文本向量化封装。

设计要点：
- 走 OpenAI 兼容接口 POST /compatible-mode/v1/embeddings；
- 模型名可配置（DASHSCOPE_EMBEDDING_MODEL，默认 qwen3.7-text-embedding，1024 维）；
- API Key 从环境变量 DASHSCOPE_API_KEY 读取，优先从项目 .env 载入（手动解析，避免额外依赖）；
- 批量嵌入（单次 ≤10 行），失败自动重试（带退避），保证整批不因偶发网络错误中断；
- 始终显式传入 embeddings 给 chromadb，不依赖其默认 ONNX 嵌入函数。

运行环境：.venv_rag311（Python 3.11 + chromadb 0.6.3 + requests）。
"""
import json
import os
import time
from pathlib import Path

import requests

# DashScope OpenAI 兼容 embeddings 端点
DASHSCOPE_EMBED_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings"
# 单次批量上限：DashScope 官方限制单请求 ≤10 行
MAX_BATCH = 10

# 项目根 .env
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = PROJECT_ROOT / ".env"


def load_env(path: Path = ENV_PATH):
    """手动解析 .env（KEY=VALUE，# 注释），写入 os.environ（已存在则不覆盖）。"""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v


def _get_env(name: str, default=""):
    return os.environ.get(name, default).strip()


class DashScopeEmbedder:
    """调用 DashScope 文本向量模型的嵌入器。

    参数：
        model             模型名，默认取 DASHSCOPE_EMBEDDING_MODEL / qwen3.7-text-embedding。
        api_key          经 DASHSCOPE_API_KEY / .env 提供；也可显式传入。
        url               端点地址。
        batch_size        批量大小(≤10)。
        timeout           单请求超时(秒)。
        max_retries       失败重试次数。
    """
    def __init__(self, model=None, api_key=None, url=DASHSCOPE_EMBED_URL,
                 batch_size=8, timeout=60, max_retries=3):
        load_env()
        self.model = model or _get_env("DASHSCOPE_EMBEDDING_MODEL", "qwen3.7-text-embedding")
        self.api_key = api_key or _get_env("DASHSCOPE_API_KEY")
        if not self.api_key:
            raise RuntimeError("未配置 DASHSCOPE_API_KEY：请在环境变量或项目 .env 中提供。")
        self.url = url
        self.batch_size = max(1, min(batch_size, MAX_BATCH))
        self.timeout = timeout
        self.max_retries = max_retries

    def _headers(self):
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def _one_batch(self, texts):
        """对一批文本（≤MAX_BATCH）调用一次 API，返回 [[float,...], ...]，失败抛异常。"""
        payload = {
            "model": self.model,
            "input": list(texts),   # 原样发送；超长由 DashScope 按自身 token 上限处理/报错，不做静默截断
            "encoding_format": "float",
        }
        last_err = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = requests.post(self.url, headers=self._headers(),
                                     json=payload, timeout=self.timeout)
                if resp.status_code == 200:
                    data = resp.json()["data"]
                    # 按底层给出的 index 排序，保证与输入顺序一致
                    data = sorted(data, key=lambda d: d.get("index", 0))
                    return [d["embedding"] for d in data]
                last_err = f"HTTP {resp.status_code}: {resp.text[:200]}"
            except Exception as exc:  # noqa: BLE001
                last_err = f"{type(exc).__name__}: {exc}"
            if attempt < self.max_retries:
                time.sleep(1.5 * (attempt + 1))
        raise RuntimeError(f"DashScope 嵌入失败({self.model}/{len(texts)}条): {last_err}")

    def embed(self, texts):
        """嵌入多条文本，透明分批。返回 list[list[float]]。"""
        texts = list(texts)
        vectors = []
        for i in range(0, len(texts), self.batch_size):
            vectors.extend(self._one_batch(texts[i:i + self.batch_size]))
        if not vectors:
            raise ValueError("输入文本为空。")
        return vectors

    def embed_one(self, text):
        return self.embed([text])[0]

    @property
    def dim(self):
        """用一条短文本探测输出维度。"""
        return len(self.embed_one("维度探测"))


if __name__ == "__main__":
    emb = DashScopeEmbedder()
    print("model:", emb.model, "| dim:", emb.dim)
    vs = emb.embed(["你好", "世界"])
    print("returned", len(vs), "vectors")
