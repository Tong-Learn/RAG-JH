# -*- coding: utf-8 -*-
"""
切片模块：把数据准备产出的 Block 列表(每行一个块)组装成适合「向量化 + 检索」的 chunk。

设计要点(详见 docs/切片方案.md)：
- 结构优先：heading 是一级边界(维护标题栈，得到 section path)，table 作为一个整体、不可拆分；
- 段落按目标大小聚合；超长段落按句切分(绝不在句中拦腰截断)；
- 过短 chunk 触发向后融合；
- 大小驱动的切点在边界处附加「重合(overlap)」，避免关键语义/问句被切在两块之间。

约定：本模块**只操作普通 dict 块**(keys: kind/text/level/rows)，不依赖 extractors.py，
避免为一个切片任务引入 PyMuPDF/python-docx 等重量级依赖。
"""
import re
from dataclasses import dataclass, field, asdict

# ---- 句边界：切分优先在这些位置(中文+英文标点、分号、换行) ----
SENT_RE = re.compile(r"[^。！？；;!?\n]*[。！？；;!?\n]+|[^。！？；;!?\n]+$")


def split_sentences(text: str) -> list:
    """按句边界把一段文本切成句子列表(保留下标点)。"""
    return [c.strip() for c in SENT_RE.findall(text) if c.strip()]


@dataclass
class Chunk:
    id: str                       # doc + '#' + 序号
    doc: str                      # 文档标识(sampleN_xxx)
    section_path: list            # 标题栈，如 ["公务用车管理规定","一、适用范围"]
    text: str                     # 真正拿去向量化+检索的字符串(含标题前缀、含 overlap 前缀)
    rows: list = field(default_factory=list)   # table 保留原始行列；para 为空
    start_block: int = 0          # 覆盖的源块起始下标
    end_block: int = 0            # 覆盖的源块结束下标
    start_page: int = 0           # 覆盖的源块起始页码(1-based；无页码为 0)
    end_page: int = 0             # 覆盖的源块结束页码


# ---------- 段落打包 ----------
def _overlap_tail(text: str, overlap_chars: int) -> str:
    """取 text 末尾约 overlap_chars 个字符作为 overlap，但**收敛到句边界**，避免把句子切一半。"""
    if overlap_chars <= 0 or not text:
        return ""
    sents = split_sentences(text)
    if not sents:
        return ""
    tail, total = [], 0
    for s in reversed(sents):
        tail.insert(0, s)
        total += len(s)
        if total >= overlap_chars:
            break
    return "".join(tail)


def _split_long_para(text: str, block_idx: int, chunk_size: int):
    """单个超长段落：按句切分成长度约 chunk_size 的若干组。每组返回 (block_idx, 子文本)。"""
    groups, cur, cur_len = [], [], 0
    for s in split_sentences(text):
        if len(s) > chunk_size:                      # 单句本身就超长 → 只能硬放(少见)
            if cur:
                groups.append((block_idx, "".join(cur)))
                cur, cur_len = [], 0
            groups.append((block_idx, s))
        elif cur_len + len(s) > chunk_size and cur:
            groups.append((block_idx, "".join(cur)))
            cur, cur_len = [s], len(s)
        else:
            cur.append(s)
            cur_len += len(s)
    if cur:
        groups.append((block_idx, "".join(cur)))
    return groups


def _join_paras(paras) -> str:
    """用换行连接段落，保留段落边界(避免多个段落被无分隔粘连)。"""
    return "\n".join(paras)


def _fuse_small(segs, min_chunk):
    """把长度 < min_chunk 的段并入前一段(向后融合)，避免碎块。"""
    out = []
    for s in segs:
        if out and s["chars"] < min_chunk:
            out[-1]["paras"] += s["paras"]
            out[-1]["blocks"] += s["blocks"]
            out[-1]["para_count"] += s["para_count"]
            out[-1]["chars"] += s["chars"]
        else:
            out.append(s)
    return out


def pack_paras(paras, chunk_size=300, min_chunk=100, overlap_chars=45):
    """把连续的段落[(text, block_idx)]打包成若干段(segment)，兼顾大小/句边界/重合/融合。

    返回段列表，每段 dict：
      {"paras":[段落文本...], "blocks":[源块下标...], "para_count":int,
       "chars":int(正文字符数), "overlap":str(来自上一段的尾部，用于防止语义被切)}
    """
    segs = []
    cur_paras, cur_blocks, cur_chars, cur_n = [], [], 0, 0

    def emit():
        nonlocal cur_paras, cur_blocks, cur_chars, cur_n
        if cur_paras:
            segs.append({"paras": cur_paras, "blocks": cur_blocks,
                         "para_count": cur_n, "chars": cur_chars, "overlap": ""})
            cur_paras, cur_blocks, cur_chars, cur_n = [], [], 0, 0

    for text, bi in paras:
        n = len(text)
        if n > chunk_size:                       # 超长段：先结算当前，再按句切
            emit()
            for b, sub in _split_long_para(text, bi, chunk_size):
                segs.append({"paras": [sub], "blocks": [b], "para_count": 1,
                             "chars": len(sub), "overlap": ""})
            continue
        # 普通段：装了会超 size、且当前段已达到最小长度 → 切段(段边界即语义边界)
        if cur_chars and cur_chars + n > chunk_size and cur_chars >= min_chunk:
            emit()
        cur_paras.append(text)
        cur_blocks.append(bi)
        cur_chars += n
        cur_n += 1
    emit()

    segs = _fuse_small(segs, min_chunk)

    # 往后的每一个段，单独记录上一段的尾部(overlap)，保持语义连续性；不混入正文，拼时再接。
    for k in range(1, len(segs)):
        tail = _overlap_tail(_join_paras(segs[k - 1]["paras"]), overlap_chars)
        if tail:
            segs[k]["overlap"] = tail
    return segs


# ---------- 主入口 ----------
def chunk_blocks(blocks, doc="", chunk_size=300, min_chunk=100,
                 overlap_ratio=0.15, attach_heading=True, id_start=0):
    """Block 列表 -> Chunk 列表。blocks 是普通 dict(kind/text/level/rows)。

    id 采用全局递增短键 `chunk_{序号:06d}`(id_start + 本文档内自增)，把完整溯源
    (doc/section_path/start_block/end_block) 留在 metadata，id 只保证唯一 + 短。
    """
    overlap_chars = max(1, int(chunk_size * overlap_ratio))
    doc = doc or ""

    # 1. 切分 section：以 heading 为边界，维护标题栈
    sections, title_stack = [], []
    cur = {"title_stack": [], "items": []}
    for bi, b in enumerate(blocks):
        kind = b.get("kind")
        if kind == "heading":
            if cur["items"]:
                sections.append(cur)
            level = max(1, int(b.get("level", 1)))
            title_stack = title_stack[: level - 1] + [b.get("text", "").strip()]
            cur = {"title_stack": list(title_stack), "items": []}
        elif kind == "table":
            cur["items"].append(("table", b.get("rows") or [], bi))
        else:                                     # para
            cur["items"].append(("para", b.get("text", ""), bi))
    if cur["items"]:
        sections.append(cur)

    # 2. 把每个 section 组装成 chunk
    chunks, chunk_counter = [], 0

    def make_chunk(sec, text, block_idxs, rows=None):
        nonlocal chunk_counter
        chunk_counter += 1
        lo, hi = min(block_idxs), max(block_idxs)
        pages = [blocks[i].get("page", 0) for i in block_idxs]
        sp = min(pages) if pages else 0
        ep = max(pages) if pages else 0
        return Chunk(
            id=f"chunk_{id_start + chunk_counter:06d}", doc=doc,
            section_path=sec["title_stack"],
            text=text, rows=rows or [], start_block=lo, end_block=hi,
            start_page=sp, end_page=ep,
        )

    def merge_tail(sec_chunks):
        """把 section 末尾**一个**过短(< min_chunk)的片段回并到前一块，避免出现 2 个字的废块。

        只做一次、不级联，避免把语义不同的内容连锁合并；表格保持整体性，绝不并入别块。
        """
        if len(sec_chunks) >= 2:
            tail = sec_chunks[-1]
            if len(tail.text) < min_chunk and not tail.rows:
                prev = sec_chunks[-2]
                prev.text = prev.text.rstrip() + "\n" + tail.text.strip()
                prev.end_block = max(prev.end_block, tail.end_block)
                prev.start_page = min(prev.start_page, tail.start_page)
                prev.end_page = max(prev.end_page, tail.end_page)
                sec_chunks.pop()

    for sec in sections:
        path = sec["title_stack"]
        prefix = (" > ".join(path) + "\n") if (attach_heading and path) else ""
        sec_chunks = []

        para_buf = []                              # 暂存连续段落，遇 table 中断
        def flush_paras():
            if not para_buf:
                return
            for seg in pack_paras(para_buf, chunk_size, min_chunk, overlap_chars):
                parts = ([seg["overlap"]] if seg["overlap"] else []) + seg["paras"]
                text = prefix + "\n".join(parts)
                sec_chunks.append(make_chunk(sec, text, seg["blocks"]))
            para_buf.clear()

        for item in sec["items"]:
            if item[0] == "para":
                para_buf.append((item[1], item[2]))
            else:                                   # table：先结算段落，再整体成块
                flush_paras()
                rows = item[1]
                flat = "\n".join(" | ".join(c for c in r) for r in rows)
                if attach_heading:
                    flat = prefix + flat
                sec_chunks.append(make_chunk(sec, flat, [item[2]], rows=rows))
        flush_paras()
        merge_tail(sec_chunks)
        chunks.extend(sec_chunks)

    return chunks
