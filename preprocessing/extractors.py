# -*- coding: utf-8 -*-
"""
格式提取模块：把各种源文档解析成结构化 "块"(Block)。
Block 只有三种：heading(标题/带级别) / para(段落) / table(表格)。
这样后续的清洗和切片都能拿到语义结构，而不是一串裸文本。
"""
from dataclasses import dataclass, field
from pathlib import Path
import re

import fitz
import docx
import openpyxl
from pptx import Presentation
from docx.text.paragraph import Paragraph
from docx.table import Table


@dataclass
class Block:
    kind: str                      # "heading" | "para" | "table"
    text: str = ""                 # heading / para 的正文
    level: int = 1                 # heading 级别
    rows: list = field(default_factory=list)  # table 的行(每行是字符串列表)
    page: int = 0                  # 页码(1-based)，非 PDF 或无页码为 0


# ---------- txt ----------
def extract_txt(path: Path):
    raw = _read_text_with_fallback(path)
    blocks = []
    for chunk in re.split(r"\n\s*\n", raw):   # 按空行切段落
        chunk = chunk.strip()
        if chunk:
            blocks.append(Block("para", text=" ".join(chunk.split())))
    return blocks


# ---------- md ----------
def extract_md(path: Path):
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    blocks = []
    i = 0
    while i < len(lines):
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        m = re.match(r"^(#{1,6})\s+(.*)$", line)
        if m:
            blocks.append(Block("heading", text=m.group(2).strip(), level=len(m.group(1))))
            i += 1
            continue
        # Markdown 表格：表头行 + 分隔行(|---|)
        if line.startswith("|") and i + 1 < len(lines) and re.match(r"^\s*\|[\s:\-|]+\|\s*$", lines[i + 1]):
            header = _parse_md_row(line)
            i += 2
            table_rows = [header]
            while i < len(lines) and lines[i].strip().startswith("|"):
                table_rows.append(_parse_md_row(lines[i].strip()))
                i += 1
            blocks.append(Block("table", rows=table_rows))
            continue
        # 其它视为段落（连续行合并成一个段落）
        buf = [line]
        i += 1
        while i < len(lines) and lines[i].strip() and not lines[i].strip().startswith("#"):
            buf.append(lines[i].strip())
            i += 1
        blocks.append(Block("para", text=" ".join(" ".join(buf).split())))
    return blocks


# ---------- docx ----------
def extract_docx(path: Path):
    d = docx.Document(str(path))
    blocks = []
    for item in d.iter_inner_content():
        if isinstance(item, Paragraph):
            text = item.text.strip()
            if not text:
                continue
            style = (item.style.name or "").lower()
            m = re.match(r"heading\s*(\d)", style)
            if m:
                blocks.append(Block("heading", text=text, level=int(m.group(1))))
            else:
                blocks.append(Block("para", text=text))
        elif isinstance(item, Table):
            rows = [[cell.text.strip() for cell in row.cells] for row in item.rows]
            blocks.append(Block("table", rows=rows))
    return blocks


# ---------- pdf ----------
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
# 标题"满行"判定容差(pt)：一行 x1 距右边界不足此值视为"换行后溢出"，才允许把下一行并入标题。
FULL_WIDTH_TOL = 6.0


def _estimate_line_h(all_lines):
    """估算正文行高：取相邻行 y0 差值的正数众数(0.5pt 分桶，抗浮点噪声)。

    只统计合理行距(0.5~60pt)，排除页内大幅跳变；单栏版面下众数即正文行距。
    """
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
    【说明】页眉/页脚处理已放弃：本项目不考察页眉页脚，不再按 y 坐标边距剔除顶部/底部行
    （旧方案会误删文档标题、又漏掉文章元数据行，且换成正则就过拟合到当前语料）。空行仍返回 None。
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
    【改进】大字号(近似标题)即使以「！/。」等句末标点结尾也判为标题(如「…上线！」
    「…放开打！」)；仅对小字号行保留「句末标点=非标题」约束(防正文短句误判)。
    超长行、纯序号行不视为标题。
    """
    text = ln["text"]
    if len(text) > 50:
        return 0
    ratio = ln["size"] / body_size if body_size else 0.0
    # 大字号：明显标题，忽略句末标点
    if ratio >= HEAD_RATIO_L1 and len(text) <= 50:
        return 1
    # 小字号：仍避开句末标点结尾的正文行
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


def extract_pdf(path: Path):
    """
    文本型 PDF：
    1) 行级提取：get_text("dict") 拿字号/加粗；正文行按 y 间距合并成完整段落(见 _merge_para_lines)；
       字号显著大于正文或加粗的短行识别为 heading(文档级去重，防每页标题栏重复)；
       标题跨行(换行拆成两半)会按「同 level + 小 y 间距」并成一条标题。
    2) 表格识别：page.find_tables()，真表格保留为 table 块；
       单行/单列的误判"表格"不输出 table，其区域短文本转为 heading 候选。
    【页眉/页脚处理已放弃】不再按 y 坐标边距剔除顶部/底部行——本管线不考察页眉页脚；旧方案
    会误删文档标题、又漏掉文章元数据行，改成正则又会过拟合到当前语料，故整体放弃（只保留
    cleaner 里领域无关的页码/数字噪音过滤）。
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
        # 3) 按 text block 顺序输出：标题/表格打断段落；正文行按 y 间距合并成完整段落。
        #    (先判标题/表格，再合并正文——标题/表格只作为边界，绝不并入段落，避免误判)
        body_buf = []                                  # 当前段落缓冲：连续正文行 info
        last_heading = None                            # 上一条标题(用于标题跨行合并)
        last_heading_size = None                      # 其字号
        last_heading_y1 = None                        # 其最后一行 y 底
        last_heading_fullwidth = None                 # 上一标题行是否"满行"(换行溢出才算标题续行)
        gap_thr = max(1.0, line_h * PARA_GAP_FACTOR)
        def flush_para():
            nonlocal body_buf
            if body_buf:
                for para in _merge_para_lines(body_buf, line_h):
                    blocks.append(Block("para", text=para, page=pg))
                body_buf = []

        for b in raw["blocks"]:
            if b.get("type") != 0:
                continue
            ls = [info for info in (_line_info(ln) for ln in b.get("lines", [])) if info]
            if not ls:
                continue
            for info in ls:
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
                    y0, _, _, y1 = info["bbox"]
                    fullwidth = info["bbox"][2] >= right_edge - FULL_WIDTH_TOL
                    # 标题跨行合并：同级别 + 字号带内 + y 间距小 + 上一行满行(换行溢出) -> 并入
                    if (last_heading is not None and last_heading.level == level
                            and last_heading_size is not None
                            and abs(last_heading_size - info["size"]) <= HEAD_SIZE_BAND
                            and last_heading_y1 is not None and (y0 - last_heading_y1) <= gap_thr
                            and last_heading_fullwidth):
                        last_heading.text = _join_para_line_parts([last_heading.text, info["text"]])
                        last_heading_size = info["size"]
                        last_heading_y1 = y1
                        last_heading_fullwidth = fullwidth
                    else:
                        key = (info["text"], level)
                        if key not in seen_headings:   # 文档级去重：防每页重复标题栏
                            seen_headings.add(key)
                            hb = Block("heading", text=info["text"], level=level, page=pg)
                            blocks.append(hb)
                            last_heading, last_heading_size = hb, info["size"]
                            last_heading_y1 = y1
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


# ---------- xlsx ----------
def extract_xlsx(path: Path):
    wb = openpyxl.load_workbook(str(path), data_only=True)
    blocks = []
    for ws in wb.worksheets:
        grid = [[_clean_cell(c.value) for c in row] for row in ws.iter_rows()]
        # 找到一个数据表格：行内"非空单元格数 >=2"且下一行也满足
        table = []
        i = 0
        while i < len(grid):
            nonempty = [v for v in grid[i] if v]
            if len(nonempty) >= 2 and i + 1 < len(grid) and len([v for v in grid[i + 1] if v]) >= 2:
                # 作为表头 + 数据行，进入表格
                table.append(grid[i])
                i += 1
                while i < len(grid) and len([v for v in grid[i] if v]) >= 2:
                    table.append(grid[i])
                    i += 1
                blocks.append(Block("table", rows=table))
                table = []
                continue
            if len(nonempty) == 1:
                blocks.append(Block("para", text=nonempty[0].strip()))
            i += 1
    wb.close()
    return blocks


# ---------- pptx ----------
def extract_pptx(path: Path):
    prs = Presentation(str(path))
    blocks = []
    for idx, slide in enumerate(prs.slides):
        # 标题
        if slide.shapes.title is not None and slide.shapes.title.text.strip():
            blocks.append(Block("heading", text=slide.shapes.title.text.strip(), level=idx + 1))
        for shape in slide.shapes:
            if shape == slide.shapes.title:
                continue
            if shape.has_table:
                rows = [[cell.text.strip() for cell in row.cells] for row in shape.table.rows]
                blocks.append(Block("table", rows=rows))
            elif shape.has_text_frame:
                for para in shape.text_frame.paragraphs:
                    t = "".join(run.text for run in para.runs).strip()
                    if t:
                        blocks.append(Block("para", text=t))
    prs = None
    return blocks


# ---------- 工具函数 ----------
def _rect_in_any(rect, rects):
    """判断 rect(x0,y0,x1,y1) 是否完全落在任意一张表格 bbox 内(避免重复抽取)。"""
    rx0, ry0, rx1, ry1 = rect
    for tx0, ty0, tx1, ty1 in rects:
        if rx0 >= tx0 and ry0 >= ty0 and rx1 <= tx1 and ry1 <= ty1:
            return True
    return False


def _clean_cell(v):
    if v is None:
        return ""
    return str(v).strip()


def _parse_md_row(line):
    cells = [c.strip() for c in line.strip().strip("|").split("|")]
    return cells


def _read_text_with_fallback(path: Path):
    data = Path(path).read_bytes()
    for enc in ("utf-8", "gbk", "gb18030"):
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


# 后缀 -> 提取函数
EXTRACTORS = {
    ".txt": extract_txt,
    ".md": extract_md,
    ".docx": extract_docx,
    ".pdf": extract_pdf,
    ".xlsx": extract_xlsx,
    ".pptx": extract_pptx,
}
