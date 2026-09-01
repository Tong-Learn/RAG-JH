# -*- coding: utf-8 -*-
"""
清洗模块：对提取出的 Block 做规范化与去噪。
- 规范化：全角空格->半角、连续空白->单个空格、去行尾空白
- 去噪：页码/总页数("第X页/共Y页"、独立数字行)等版式噪音
- 丢弃空块
"""
import re
from preprocessing.extractors import Block

# 页脚/页码类噪音
PAGE_NOISE = [
    re.compile(r"第\s*\d+\s*页\s*/\s*共\s*\d+\s*页"),
    re.compile(r"第\s*\d+\s*页"),
    re.compile(r"^\s*\d+\s*$"),          # 独立数字行(页码)
]
# HTML 注释
HTML_COMMENT = re.compile(r"<!--.*?-->", flags=re.S)
# 文章元数据行：以「日期 时间」开头、且很短(如「2026年4月13日 18:32上海晶核CoA听全文」)。
# 用「日期+时间」的通用形态而非写死来源名(避免过拟合)，靠「很短 + 无句末标点」来限定。
META_LINE_RE = re.compile(r"^\s*\d{4}\s*年\s*\d{1,2}\s*月\s*\d{1,2}\s*日\s+\d{1,2}:\d{2}")
META_LINE_MAX_LEN = 35


def _normalize_ws(text: str) -> str:
    text = HTML_COMMENT.sub(" ", text)   # 去掉 HTML 注释
    text = text.replace("\u3000", " ")   # 全角空格 -> 半角
    # 去「数字/字母 + 空格 + 汉字」之间的多余空格：如「2026 年」「4 月」「15 日」-> 紧贴
    text = re.sub(r"(?<=[0-9A-Za-z])\s+(?=[\u4e00-\u9fff])", "", text)
    # 去全角标点前后的空格：如「1 、」「晶珀*100 ，」「，将通过」-> 紧贴
    text = re.sub(r"\s+(?=[，。、；：？！])", "", text)
    text = re.sub(r"(?<=[，。、；：？！])\s+", "", text)
    text = re.sub(r"[ \t]+", " ", text)  # 连续空格 -> 单个
    return text.strip()


def _is_noise(text: str) -> bool:
    for pat in PAGE_NOISE:
        if pat.search(text):
            return True
    # 文章元数据行：日期+时间开头且很短(排正文；正文长或带句末标点不会命中)
    if len(text) <= META_LINE_MAX_LEN and META_LINE_RE.match(text):
        return True
    return False


def clean_block(block: Block):
    if block.kind in ("heading", "para"):
        text = _normalize_ws(block.text)
        if not text or _is_noise(text):
            return None
        if block.kind == "heading":
            return Block("heading", text=text, level=block.level)
        return Block("para", text=text)
    if block.kind == "table":
        rows = [[_normalize_ws(c) for c in row] for row in block.rows]
        # 移除整行都为空的行；若整个表格空了则丢弃
        rows = [r for r in rows if any(c for c in r)]
        if not rows:
            return None
        return Block("table", rows=rows)
    return None


def clean_blocks(blocks):
    cleaned = []
    for b in blocks:
        cb = clean_block(b)
        if cb is not None:
            cleaned.append(cb)
    return cleaned
