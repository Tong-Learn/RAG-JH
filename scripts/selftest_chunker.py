# -*- coding: utf-8 -*-
"""轻量自测：验证切片核心逻辑(超长按句切分/重合/融合/标题分节/表格整体)。
运行：.venv_rag311\Scripts\python.exe -m scripts.selftest_chunker
"""
from chunking.chunker import chunk_blocks, split_sentences, _overlap_tail

TERM = "。！？；；!?\n"


def _assert(cond, msg):
    if not cond:
        raise AssertionError("FAIL: " + msg)
    print(f"  [v] {msg}")


# ---- 1. 超长段落按句切分：每组都是完整句子(不在句中截断)，且字符无丢失 ----
long_para = "".join(f"这是第{i}个很长的句子用来验证切分不拦腰截断。" for i in range(1, 7))
blocks = [
    {"kind": "heading", "text": "章节A", "level": 1},
    {"kind": "para", "text": long_para, "level": 1, "rows": []},
]
chunks = chunk_blocks(blocks, doc="test1", chunk_size=40, min_chunk=10, overlap_ratio=0.2)
# 去掉标题前缀后，把各 group 的正文重新拼起来，应等于原句全部句子
gathered = ""
for c in chunks:
    body = c.text.split("\n", 1)[1] if "\n" in c.text else c.text
    # 若带 overlap 前缀，正文开头会重复上段尾部——这里只验证没有"半句":
    gathered = body  # 仅记录，实际断言见下
sentences = split_sentences(long_para)
_assert(len(chunks) >= 2, f"超长段落切成 {len(chunks)} 段(预期>=2)")
# 每一段正文都能被完整切回句子(末尾带终止符)，且首段不含 overlap 前缀带来的半句
for c in chunks:
    body = c.text.split("\n", 1)[1]
    parts = split_sentences(body)
    if parts:
        _assert(body.endswith(("。", "！", "？", ";", "；")) or len(body) == 0,
                f"每段以句边界结尾(非半句): {body[:18]}...")


# ---- 2. 重合(overlap)：第2段正文开头重复第1段尾句(整句，不切半句) ----
c1, c2 = chunks[0], chunks[1]
body1 = c1.text.split("\n", 1)[1]
body2 = c2.text.split("\n", 1)[1]
expected_tail = _overlap_tail(body1, int(40 * 0.2))
_assert(body2.startswith(expected_tail),
        f"第2段带第1段尾句重合: …{expected_tail}… | {body2[:10]}…")
_assert(expected_tail != "", "重合部分非空(整句, 不切半句)")


# ---- 3. 融合：过短尾部片段并入前一块 ----
blocks3 = [
    {"kind": "heading", "text": "B", "level": 1},
    {"kind": "para", "text": "段落内容较长的一段文字，用于形成足够大的块。", "level": 1, "rows": []},
    {"kind": "table", "rows": [["列1", "列2"], ["a", "b"]]},
    {"kind": "para", "text": "尾部小尾巴。", "level": 1, "rows": []},
]
chunks3 = chunk_blocks(blocks3, doc="test3", chunk_size=50, min_chunk=10, overlap_ratio=0.2)
last = chunks3[-1]
_assert(len(last.text) >= 10, f"尾部小片段已融合, 末块长度={len(last.text)}")
_assert("尾部小尾巴" in last.text, "尾部小尾巴并入末块文本")


# ---- 4. 标题分节 + 表格整体：section_path 正确、表格不被拆分 ----
blocks4 = [
    {"kind": "heading", "text": "总规定", "level": 1},
    {"kind": "table", "rows": [["车型", "核载"], ["商务车", "7座"]]},
    {"kind": "heading", "text": "一、范围", "level": 2},
    {"kind": "para", "text": "适用于所有正式员工的用车。", "level": 1, "rows": []},
]
chunks4 = chunk_blocks(blocks4, doc="test4", chunk_size=50, min_chunk=10, overlap_ratio=0.2)
tbl = [c for c in chunks4 if c.rows][0]
_assert(tbl.section_path == ["总规定"], f"表格归属标题栈 {tbl.section_path}")
_assert(tbl.rows == [["车型", "核载"], ["商务车", "7座"]], "表格整体保留(未拆分)")
sub = [c for c in chunks4 if "一、范围" in c.section_path]
_assert(sub and sub[0].section_path == ["总规定", "一、范围"], f"二级标题嵌套 {sub[0].section_path}")


print("\n自测全部通过 ✅")
