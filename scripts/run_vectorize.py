# -*- coding: utf-8 -*-
"""
向量化管线流程三：读取切片产物 data/chunks/*.chunks.jsonl，用 DashScope 向量化后写入 chromadb。

映射策略（回答「如何与原切片对应」）：
  向量库一行 = {id, embedding, document, metadata}，由现有 chunk 直接映射：
    id        <- chunk.id           (如 "示例1#3")，检索后靠它回映射到原 chunk
    embedding <- embed(chunk.text)
    document  <- chunk.text         (正文/检索展示)
    metadata  <- doc/section_path/kind_comp/start_block/end_block/char_len/overlap
  metadata 里保留全部溯源坐标，命中后再靠 doc+section_path+start/end_block 回溯回原文档。

运行：.venv_rag311\\Scripts\\python.exe -m scripts.run_vectorize
依赖：.venv_rag311 里的 chromadb 0.6.3、numpy 1.26.4、requests；DASHSCOPE_API_KEY(项目 .env)
"""
import json
import sys
from pathlib import Path

# 本项目允许以脚本方式直接运行
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import chromadb
from chromadb.config import Settings

from embedding.dashscope_embedder import DashScopeEmbedder


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHUNKS_DIR = PROJECT_ROOT / "data" / "chunks"
CHROMA_DIR = PROJECT_ROOT / "data" / "chroma"

# 集合名（chromadb 0.6.3 要求 ≥3 字符、[a-zA-Z0-9._-]）
COLLECTION = "rag_chunks"
# 距离度量：余弦相似度（chunk 检索常用），chroma 中 distance 越小越好，0=完全一致、2=完全不相关
SPACE = "cosine"


def load_chunks():
    """读取所有 *.chunks.jsonl，返回 (chunk dict 列表, 总块数)。"""
    chunks = []
    for path in sorted(CHUNKS_DIR.glob("*.chunks.jsonl")):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    chunks.append(json.loads(line))
    return chunks


def chunk_to_record(c):
    """chunk dict -> chromadb 行的 (id, document, metadata)。

    metadata 只允许 str/int/float/bool 标量；列表(section_path)需拍平成字符串。
    只保留真正被检索/引用/评估消费的溯源字段(doc/section_path/start_block/end_block)；
    kind_comp/char_len/overlap 已从 chunk 结构移除(纯展示/统计/调试，无用且增转换)。
    """
    meta = {
        "doc": str(c["doc"]),
        "section_path": " / ".join(c.get("section_path") or []),  # list -> string
        "start_block": int(c.get("start_block", 0)),
        "end_block": int(c.get("end_block", 0)),
    }
    return c["id"], c["text"], meta


def main():
    chunks = load_chunks()
    if not chunks:
        print("无 chunk 可向量化，请先运行切片。")
        return

    # 1. 嵌入（DashScope，批处理）
    print(f"[1/3] 加载 {len(chunks)} 个 chunk 并向量化 ...")
    embedder = DashScopeEmbedder()
    vectors = embedder.embed([c["text"] for c in chunks])
    dim = len(vectors[0])
    print(f"      模型: {embedder.model}  维度: {dim}")

    # 2. 入 chromadb（persistent）
    print(f"[2/3] 写入 chromadb collection '{COLLECTION}' ({SPACE}) ...")
    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False, allow_reset=True),
    )
    # 幂等重建：先删除旧集合再新建（delete(where={}) 在 0.6.3 不会清空，只会留下 upsert 警告）
    try:
        client.delete_collection(COLLECTION)
    except Exception:  # noqa: BLE001 首次运行集合不存在则是预期分支
        pass
    # 坑：chroma-hnswlib 在本机 Windows 下，当索引跨过默认 hnsw:batch_size=100 的
    # 「暴力检索缓存→hnsw 落盘」边界时，add() 会 0xC0000005 空指针崩溃（Windows 弹窗，无 traceback）。
    # 规避：把 batch_size 调成 > 本次入库总条数，让单次入库不触发那次边界刷新。
    batch_size = max(1000, len(chunks) + 10)
    col = client.create_collection(
        name=COLLECTION,
        metadata={"hnsw:space": SPACE, "hnsw:batch_size": batch_size},
    )
    ids, documents, metadatas = [], [], []
    for c in chunks:
        cid, doc, meta = chunk_to_record(c)
        ids.append(cid)
        documents.append(doc)
        metadatas.append(meta)
    col.add(ids=ids, embeddings=vectors, documents=documents, metadatas=metadatas)

    # 3. 统计
    print(f"[3/3] 完成：库内共 {col.count()} 行（本次写入 {len(ids)}），维度 {dim}。")
    print(f"      落库路径: {CHROMA_DIR}")


if __name__ == "__main__":
    main()
