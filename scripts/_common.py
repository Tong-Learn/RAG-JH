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
EVAL_DIR = PROJECT_ROOT / "data" / "eval"
COLLECTION = "rag_chunks"


def get_client(allow_reset=False):
    return chromadb.PersistentClient(path=str(CHROMA_DIR),
                                     settings=Settings(anonymized_telemetry=False, allow_reset=allow_reset))


def get_collection():
    return get_client().get_or_create_collection(COLLECTION)


def load_queries(sets="all"):
    """读取 data/eval/qa_basic.jsonl + qa_complex.jsonl。sets: all|basic|complex。"""
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
