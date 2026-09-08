# -*- coding: utf-8 -*-
"""
PDF 提取模块：把游戏公告类 PDF（文本层）解析成结构化 "块"(Block)。

Block 只有三种：heading(标题/带级别) / para(段落) / table(表格)。
本模块只处理 PDF（项目定位为《晶核》游戏公告，源语料全部为 PDF）。
"""
from dataclasses import dataclass, field
from pathlib import Path
import re

import fitz


@dataclass
class Block:
    kind: str                      # "heading" | "para" | "table"
    text: str = ""                 # heading / para 的正文
    level: int = 1                 # heading 级别
    rows: list = field(default_factory=list)  # table 的行(每行是字符串列表)
    page: int = 0                  # 页码(1-based)，无页码为 0


# 公告类 PDF 的装饰性分区号(「01」「02」「1.」等独立短行)，不产生任何 Block
_PDF_ORDER_NO_RE = re.compile(r"^\d{1,3}[.、．]?$")

# PyMuPDF span flags 的 bit4 = bold
_PDF_BOLD_FLAG = 16

# 段落合并：相邻正文行的 y 间距 <= line_h * 此系数 视为同一段，否则视为新段落。
# 实测公告类 PDF：正文行距(y0-y0)≈20pt，段落间距≈40pt(≈2×行距)，取 1.6× 可区分。
PARA_GAP_FACTOR = 1.6

# 标题判定的字号阈值(相对正文基准)。标号统一提为常量，便于按语料校准。
HEAD_RATIO_L1 = 1.35   # >=此比值 -> level 1(大标题)
HEAD_RATIO_L2 = 1.15   # >=此比值(且短) -> level 2(章节小标题)
# 标题跨行合并的字号带(±pt)：同级别且字号差在此范围内才视为"同一标题换行"，避免误并下一节标题。
HEAD_SIZE_BAND = 1.0
# 标题"满行"判定：一行 x1 距右边界不足此值视为"换行后溢出"，允许把下一行并入标题。
# 公告类标题为左对齐，换行的首行往往到不了绝对右边界(如 2月11日/4月22日)，故用「绝对容差」与「右缘比例」二选一(取更宽松)。
FULL_WIDTH_TOL = 6.0
FULL_WIDTH_RATIO = 0.90  # 行 x1 >= 右缘*此比例，视为"近满行/换行溢出"

# 同基线合并容差(pt)：两条标题行的 y0 差在此范围内，视为同一行被拆成两半(如「全新SS-Ⅱ级时装「逝灭魔权」上线」被拆成两行文本)。
SAME_BASELINE_TOL = 2.0


def _estimate_line_h(all_lines):
    """估算正文行高：取相邻行 y0 差值的正数众数(0.5pt 分桶，抗浮点噪声)。"""
    from collections import Counter
    ys = sorted(round(ln["bbox"][1], 2) for ln in all_lines)
    gaps = Counter()
    for a, b in zip(ys, ys[1:]):
        g = b - a
        if 0.5 <= g <= 60:
            gaps[round(g * 2) / 2] += 1
    return gaps.most_common(1)[0][0] if gaps else 1.0


def _join_para_line_parts(parts):
    """把若干行拼接成一个段落。CJK 直连不插空格；仅当相邻两字符均为 ASCII 时补空格(防英文断词粘连)。"""
    s = ""
    for t in parts:
        if s and s[-1].isascii() and s[-1] != " " and t and t[0].isascii():
            s += " "
        s += t
    return s


def _merge_para_lines(lines, line_h, gap_factor=PARA_GAP_FACTOR):
    """把按阅读顺序排列的正文行，按 y 间距合并成一个或多个完整段落。

    lines: 连续正文行的 info 列表(每个含 'text' 与 'bbox'，均非标题/表格行)。
    行间距(y0-y0) <= line_h*gap_factor 视为同段；明显大于则作为段落边界。
    返回：段落文本列表。
    """
    if not lines:
        return []
    gap_thr = max(1.0, line_h * gap_factor)
    paras, cur, prev_y0 = [], [], None
    for ln in lines:
        y0 = ln["bbox"][1]
        if prev_y0 is not None and (y0 - prev_y0) > gap_thr:
            paras.append(_join_para_line_parts(cur))
            cur = []
        cur.append(ln["text"])
        prev_y0 = y0
    if cur:
        paras.append(_join_para_line_parts(cur))
    return [p.strip() for p in paras if p.strip()]


def _line_info(line):
    """把 get_text("dict") 的一个 line 聚合成 {text, size, bold, bbox}。

    行内拼接：ASCII 词间补空格、CJK 直连(不经过 split/join)；空行返回 None。
    注：管线不考察页眉页脚，故不按坐标边距剔除顶部/底部行(领域无关的页码噪音由 cleaner 过滤)。
    """
    spans = [s for s in line.get("spans", []) if s.get("text", "").strip()]
    if not spans:
        return None
    x0, y0, x1, y1 = line["bbox"]
    text, size, bold = "", 0.0, False
    for s in spans:
        t = s["text"]
        if text and text[-1] != " " and t[0] != " " and text[-1].isascii() and t[0].isascii():
            text += " "
        text += t
        size = max(size, s.get("size", 0.0))
        bold = bold or bool(s.get("flags", 0) & _PDF_BOLD_FLAG)
    text = text.strip()
    if not text:
        return None
    return {"text": text, "size": size, "bold": bold, "bbox": (x0, y0, x1, y1)}


def _mode_body_size(lines):
    """正文字号：按行字符数加权的字号众数(0.5pt 分桶，抗浮点噪声)。"""
    from collections import Counter
    bucket = Counter()
    for ln in lines:
        bucket[round(ln["size"] * 2) / 2] += len(ln["text"])
    return bucket.most_common(1)[0][0] if bucket else 0.0


def _heading_level(ln, body_size):
    """字号/加粗启发式判标题：返回 heading 级别(1/2)，非标题返回 0。

    实测公告类 PDF 字号梯度：标题栏 14.3 / 章节名 11.7 / 正文 9.1，
    以正文为基准：>=1.35 倍判 level 1，>=1.15 倍或加粗短行判 level 2。
    大字号(近似标题)即使以「！/。」等句末标点结尾也判为标题；仅对小字号行保留「句末标点=非标题」约束(防正文短句误判)。
    超长行、纯序号行不视为标题。
    """
    text = ln["text"]
    if len(text) > 50:
        return 0
    ratio = ln["size"] / body_size if body_size else 0.0
    if ratio >= HEAD_RATIO_L1 and len(text) <= 50:
        return 1
    if text[-1] in "。！？；，,.;":
        return 0
    if ratio >= HEAD_RATIO_L2 and len(text) <= 30:
        return 2
    if ln["bold"] and len(text) <= 25:
        return 2
    return 0


def _is_fake_table(rows):
    """find_tables() 的单行/单列结果判为误判(真表应有表头+数据行或多列多行)。

    实测公告类 PDF 的板块标题(如「团战挑战规则调整」)会被误报成 1x1 表格。
    """
    if len(rows) <= 1:
        return True
    return max(sum(1 for c in r if c) for r in rows) <= 1


def _rect_in_any(rect, rects):
    """判断 rect(x0,y0,x1,y1) 是否完全落在任意一张表格 bbox 内(避免重复抽取)。"""
    rx0, ry0, rx1, ry1 = rect
    for tx0, ty0, tx1, ty1 in rects:
        if rx0 >= tx0 and ry0 >= ty0 and rx1 <= tx1 and ry1 <= ty1:
            return True
    return False


def extract_pdf(path: Path):
    """
    PDF（文本层）提取：
    1) 行级提取：get_text("dict") 拿字号/加粗；正文行按 y 间距合并成完整段落(见 _merge_para_lines)；
       字号显著大于正文或加粗的短行识别为 heading(文档级去重，防每页标题栏重复)；
       标题跨行(换行拆成两半)会按「同 level + 小 y 间距 + 近满行/同基线」并成一条标题。
    2) 表格识别：page.find_tables()，真表格保留为 table 块；单行/单列的误判"表格"不输出 table，其区域短文本转为 heading 候选。
    """
    doc = fitz.open(str(path))
    blocks = []
    seen_headings = set()
    for pno, page in enumerate(doc):
        pg = pno + 1                # 页码 1-based，供溯源(chunk 元数据)与引用展示
        raw = page.get_text("dict")
        # 1) 全页行级信息(先全量收集，用于统计正文字号)
        all_lines = []
        for b in raw["blocks"]:
            if b.get("type") != 0:
                continue
            for ln in b.get("lines", []):
                info = _line_info(ln)
                if info:
                    all_lines.append(info)
        body_size = _mode_body_size(all_lines)
        line_h = _estimate_line_h(all_lines)   # 正文行高，用于判断段落边界(见 _merge_para_lines)
        right_edge = max((i["bbox"][2] for i in all_lines), default=0.0)  # 右边界：判标题是否"满行"
        # 2) 表格：真表格保留，假表格区域记录下来转标题候选
        table_rects, fake_rects = [], []
        try:
            for t in page.find_tables():
                rows = t.extract()
                rows = [[(c or "").strip() for c in r] if r else [] for r in rows]
                rows = [r for r in rows if any(c for c in r)]
                if not rows:
                    continue
                if _is_fake_table(rows):
                    fake_rects.append(t.bbox)
                else:
                    blocks.append(Block("table", rows=rows, page=pg))
                    table_rects.append(t.bbox)
        except Exception:
            pass  # 某页表格检测失败不影响整份文档
        # 3) 按「y 排序后的阅读顺序」输出：标题/表格打断段落；正文行按 y 间距合并成完整段落。
        body_buf = []                                  # 当前段落缓冲：连续正文行 info
        last_heading = None                            # 上一条标题(用于标题跨行合并)
        last_heading_size = None                      # 其字号
        last_heading_y1 = None                        # 其最后一行 y 底
        last_heading_y0 = None                        # 其最后一行 y 顶(用于同基线"一行拆两半"判定)
        last_heading_fullwidth = None                 # 上一标题行是否"近满行/换行溢出"(允许标题续行)
        gap_thr = max(1.0, line_h * PARA_GAP_FACTOR)
        def flush_para():
            nonlocal body_buf
            if body_buf:
                for para in _merge_para_lines(body_buf, line_h):
                    blocks.append(Block("para", text=para, page=pg))
                body_buf = []

        # PyMuPDF get_text("dict") 返回的 text block 顺序并非版式阅读顺序(标题/正文常被拆到块尾)。
        # 统一把本页所有行按 (y0, x0) 重排成真正的阅读顺序，再逐行判标题/表格/正文。
        ordered = sorted(all_lines, key=lambda i: (round(i["bbox"][1], 2), i["bbox"][0]))
        for info in ordered:
            if _rect_in_any(info["bbox"], table_rects):
                flush_para()                       # 表格边界：先结算当前段落
                last_heading = None                # 表格打断标题链
                last_heading_fullwidth = None
                continue
            # 假表格区域：本是被误判的板块标题，短行直接按 level 2 标题收
            if _rect_in_any(info["bbox"], fake_rects):
                level = 2 if len(info["text"]) <= 30 and info["text"][-1] not in "。！？；，,.;;" else 0
            else:
                level = _heading_level(info, body_size)
            if level:
                flush_para()                       # 标题边界：先结算当前段落(标题不入段)
                x0, y0, _, y1 = info["bbox"]       # bbox 是 (x0,y0,x1,y1)
                # "近满行/换行溢出"：绝对右缘容差 与 右缘比例 取更宽松者(左对齐标题的换行首行常到不了绝对右缘)。
                fullwidth = (info["bbox"][2] >= right_edge - FULL_WIDTH_TOL
                             or info["bbox"][2] >= right_edge * FULL_WIDTH_RATIO)
                # 同一基线(y0 几乎相等) = 同一行被 PyMuPDF 拆成两个 line 对象，可安全合并。
                same_base = last_heading_y0 is not None and abs(y0 - last_heading_y0) <= SAME_BASELINE_TOL
                # 标题跨行合并：同级别 + 字号带内 + y 间距小 + (近满行 或 同基线) -> 并入
                if (last_heading is not None and last_heading.level == level
                        and last_heading_size is not None
                        and abs(last_heading_size - info["size"]) <= HEAD_SIZE_BAND
                        and last_heading_y1 is not None and (y0 - last_heading_y1) <= gap_thr
                        and (last_heading_fullwidth or same_base)):
                    last_heading.text = _join_para_line_parts([last_heading.text, info["text"]])
                    last_heading_size = info["size"]
                    last_heading_y1 = y1
                    last_heading_y0 = y0
                    last_heading_fullwidth = fullwidth
                else:
                    key = (info["text"], level)
                    if key not in seen_headings:   # 文档级去重：防每页重复标题栏
                        seen_headings.add(key)
                        hb = Block("heading", text=info["text"], level=level, page=pg)
                        blocks.append(hb)
                        last_heading, last_heading_size = hb, info["size"]
                        last_heading_y1 = y1
                        last_heading_y0 = y0
                        last_heading_fullwidth = fullwidth
                    else:
                        last_heading = None        # 去重(重复标题栏)：打断了链，避免误续
                        last_heading_fullwidth = None
            elif _PDF_ORDER_NO_RE.match(info["text"]):
                continue                           # 装饰性序号行丢弃
            else:
                last_heading = None                # 正文行打断标题链
                last_heading_fullwidth = None
                body_buf.append(info)              # 正文行：收集，待按 y 间距合并
        flush_para()                                   # 页末必 flush
    doc.close()
    return blocks


# 后缀 -> 提取函数（本模块只处理 PDF）
EXTRACTORS = {
    ".pdf": extract_pdf,
}
