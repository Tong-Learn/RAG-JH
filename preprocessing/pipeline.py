# -*- coding: utf-8 -*-
"""
管线编排：source_dir 里的每个源文档 -> 提取 -> 清洗 -> 分别输出：
  - *.jsonl  结构化语料(每行一个 Block)，是数据准备的真输出，供下一步切片直接读取
  - *.md     渲染后的文本，仅供人眼核对提取效果
同时生成 manifest.jsonl，记录每份文档的来源、格式、块数、字符数。
运行：.venv_rag311\Scripts\python.exe -m preprocessing.pipeline [--src DIR] [--out DIR]
  --src 缺省 test(当前语料)；--out 缺省 data/processed
  （本模块依赖 fitz(PyMuPDF)，用于 PDF 提取；与向量化同用一个解释器）
"""
import dataclasses
import json
from pathlib import Path

from preprocessing.extractors import EXTRACTORS, Block
from preprocessing.cleaner import clean_blocks

SRC_DIR = Path(__file__).resolve().parents[1] / "test"
OUT_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"


def render_md(blocks) -> str:
    lines = []
    for b in blocks:
        if b.kind == "heading":
            lines.append("#" * b.level + " " + b.text)
            lines.append("")
        elif b.kind == "para":
            lines.append(b.text)
            lines.append("")
        elif b.kind == "table":
            rows = b.rows
            if not rows:
                continue
            n_cols = max(len(r) for r in rows)
            pad = lambda r: r + [""] * (n_cols - len(r))
            lines.append("| " + " | ".join(pad(rows[0])) + " |")
            lines.append("| " + " | ".join(["---"] * n_cols) + " |")
            for r in rows[1:]:
                lines.append("| " + " | ".join(pad(r)) + " |")
            lines.append("")
    return "\n".join(lines).strip() + "\n"


def describe(blocks):
    from collections import Counter
    counter = Counter(b.kind for b in blocks)
    chars = sum(len(b.text) for b in blocks if b.kind != "table")
    chars += sum(len(c) for b in blocks if b.kind == "table" for r in b.rows for c in r)
    return {"blocks": dict(counter), "chars": chars}


def write_blocks_jsonl(blocks, out_path):
    """把 Block 列表序列化成 JSONL(每行一个块)，作为数据准备的结构化语料产出。"""
    with open(out_path, "w", encoding="utf-8") as f:
        for b in blocks:
            f.write(json.dumps(dataclasses.asdict(b), ensure_ascii=False) + "\n")


def run(src_dir=SRC_DIR, out_dir=OUT_DIR):
    src_dir = Path(src_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    for path in sorted(src_dir.iterdir()):
        ext = path.suffix.lower()
        if ext not in EXTRACTORS:
            continue
        raw = EXTRACTORS[ext](path)
        cleaned = clean_blocks(raw)
        stem = path.stem
        corpus_path = out_dir / (stem + ".jsonl")   # 结构化语料(切片用)
        md_path = out_dir / (stem + ".md")          # 人眼核对用
        write_blocks_jsonl(cleaned, corpus_path)
        md_path.write_text(render_md(cleaned), encoding="utf-8")
        rec = {
            "source": str(path),
            "format": ext.lstrip("."),
            "corpus": str(corpus_path),
            "human_readable": str(md_path),
            **describe(cleaned),
        }
        manifest.append(rec)
        print(f"[OK] {path.name} -> {corpus_path.name} + {md_path.name}  {describe(cleaned)}")
    (out_dir / "manifest.jsonl").write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in manifest) + "\n", encoding="utf-8"
    )
    print(f"\n完成，共处理 {len(manifest)} 份文档，清单见 manifest.jsonl")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="数据准备管线")
    ap.add_argument("--src", default=str(SRC_DIR), help="源文档目录，缺省 data/samples")
    ap.add_argument("--out", default=str(OUT_DIR), help="结构化输出目录，缺省 data/processed")
    args = ap.parse_args()
    run(src_dir=args.src, out_dir=args.out)
