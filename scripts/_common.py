# -*- coding: utf-8 -*-
"""scripts 公共：路径/常量、chroma 客户端、评测集读取。供各脚本复用，避免重复。"""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import chromadb
from chromadb.config import Settings

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHROMA_DIR = PROJECT_ROOT / "data" / "chroma"
CHUNKS_DIR = PROJECT_ROOT / "data" / "chunks"
EVAL_DIR = PROJECT_ROOT / "data" / "eval"
COLLECTION = "rag_chunks"

# 评测集：源文件在 test/qa_dataset/（本地，git 不入库），仓库内保留一份可提交副本 data/eval/
QA_DATASET_DIR = PROJECT_ROOT / "test" / "qa_dataset"
QA_FILES = {"simple": "simple_qa.json", "complex": "complex_qa.json"}
# 两集统一 top-k = 5（complex 每题最多 4 个来源文档，k=5 足以覆盖）
QA_K = {"simple": 5, "complex": 5}


def preflight(mode="hybrid_rerank"):
    """检查「能否运行检索/生成」的前提，返回问题清单（空列表 = 就绪）。

    为什么需要：data/chroma、data/chunks 均不入库（.gitignore），新克隆后直接跑 run_rag
    会在底层抛出难懂的异常（如空语料下 BM25Okapi 的 ZeroDivisionError）。
    入口脚本先调本函数，把缺什么、该跑哪条命令直接讲清楚。
    检查项：API Key / 向量库 / 切片产物（仅混合路线需要 BM25 语料）。
    """
    import os
    problems = []
    if not os.environ.get("DASHSCOPE_API_KEY") and not (PROJECT_ROOT / ".env").exists():
        problems.append("未配置 API Key：复制 .env.example 为 .env 并填入 DASHSCOPE_API_KEY")
    if not CHROMA_DIR.exists() or not any(CHROMA_DIR.iterdir()):
        problems.append("向量库为空：data/chroma 不存在或为空（先运行 scripts.run_vectorize）")
    if mode == "hybrid_rerank":
        if not CHUNKS_DIR.exists() or not any(CHUNKS_DIR.glob("*.chunks.jsonl")):
            problems.append("缺少切片产物：data/chunks 下无 *.chunks.jsonl（先运行 scripts.run_chunking）")
    return problems


def print_preflight(problems):
    """把 preflight 结果打印成可执行的指引；返回是否就绪。"""
    if not problems:
        return True
    print("【无法运行】运行前提未满足：")
    for p in problems:
        print(f"  - {p}")
    print("\n完整构建步骤（本仓库为源码仓库，语料与索引需本地构建）：")
    print("  0) 把公告 PDF 放入 test/")
    print("  1) python -m preprocessing.pipeline --src test --out data\\processed")
    print("  2) python -m scripts.run_chunking")
    print("  3) python -m scripts.run_vectorize")
    print("详见 README「运行指南」。")
    return False


def get_client(allow_reset=False):
    return chromadb.PersistentClient(path=str(CHROMA_DIR),
                                     settings=Settings(anonymized_telemetry=False, allow_reset=allow_reset))


def get_collection():
    return get_client().get_or_create_collection(COLLECTION)


def _find_qa_file(fname):
    """优先 test/qa_dataset/（源），回退 data/eval/（仓库内可提交副本）。"""
    for d in (QA_DATASET_DIR, EVAL_DIR):
        p = d / fname
        if p.exists():
            return p
    return None


def load_qa_dataset(sets="all"):
    """读评测集（simple_qa.json / complex_qa.json，各 30 题），返回统一记录列表。

    统一字段：{id, set, question, answer, sources(list[str]), evidence(list[str]), answerable}
      - question <- 原 question
      - answer   <- gold answer（生成侧"答案正确率"的基准）
      - sources  <- 原 sources，**去掉 .pdf 后缀**（与 chunk 的 doc=stem 对齐）
      - 不可答题 sources 为空、answerable=False
    sets: all | simple | complex
    """
    out = []
    for setname, fname in QA_FILES.items():
        if sets != "all" and sets != setname:
            continue
        p = _find_qa_file(fname)
        if p is None:
            continue
        data = json.loads(p.read_text(encoding="utf-8"))
        for q in data:
            out.append({
                "id": q["id"],
                "set": setname,
                "question": q["question"],
                "answer": q.get("answer", ""),
                "sources": [s[:-4] if s.lower().endswith(".pdf") else s for s in q.get("sources", [])],
                "evidence": q.get("evidence", []),
                "answerable": bool(q.get("answerable", True)),
            })
    return out
