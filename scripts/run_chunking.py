# -*- coding: utf-8 -*-
"""
切片管线：读取数据准备产物 data/processed/*.jsonl(每行一个 Block)，
经 structure-aware 切片后，把每份文档的 chunk 写到 data/chunks/<stem>.chunks.jsonl，
并在控制台打印统计(块数、平均/最大/最小长度)。

运行：.venv_rag311\Scripts\python.exe -m scripts.run_chunking
（切片仅依赖标准库，在 .venv_rag311 下运行，与向量化同用一个解释器）
"""
import json
import statistics
from pathlib import Path

from chunking.chunker import chunk_blocks

PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"
CHUNKS_DIR = Path(__file__).resolve().parents[1] / "data" / "chunks"

# 切片参数(默认值，接真实数据后再调)
CHUNK_SIZE = 300
MIN_CHUNK = 100
OVERLAP_RATIO = 0.15


def load_blocks(path: Path):
    """从 jsonl 读取 Block 列表(普通 dict)。"""
    blocks = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                blocks.append(json.loads(line))
    return blocks


def write_chunks_jsonl(chunks, out_path: Path):
    with open(out_path, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c.__dict__, ensure_ascii=False) + "\n")


def describe(chunks):
    lens = [len(c.text) for c in chunks]   # char_len 已移除，改为现算 len(text)
    if not lens:
        return {"chunks": 0, "avg": 0, "max": 0, "min": 0}
    return {
        "chunks": len(chunks),
        "avg": round(statistics.mean(lens), 1),
        "max": max(lens),
        "min": min(lens),
    }


def run():
    CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    total = 0
    gidx = 0                     # 全局 chunk id 递增序号(方案A：id 用短唯一键)
    for path in sorted(PROCESSED_DIR.glob("*.jsonl")):
        if path.name == "manifest.jsonl":
            continue
        blocks = load_blocks(path)
        chunks = chunk_blocks(blocks, doc=path.stem,
                              chunk_size=CHUNK_SIZE, min_chunk=MIN_CHUNK,
                              overlap_ratio=OVERLAP_RATIO, id_start=gidx)
        gidx += len(chunks)
        out_path = CHUNKS_DIR / (path.stem + ".chunks.jsonl")
        write_chunks_jsonl(chunks, out_path)
        total += len(chunks)
        print(f"[OK] {path.name} -> {out_path.name}  {describe(chunks)}")
    print(f"\n完成，共切出 {total} 个 chunk，见 {CHUNKS_DIR}")


if __name__ == "__main__":
    run()
