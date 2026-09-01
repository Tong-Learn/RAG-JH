# -*- coding: utf-8 -*-
"""
生成 RAG 数据准备阶段所需的样本文档。
每个格式生成一个，内容为模拟的中文文档，且故意包含一些"噪音"（页眉页脚、
多余空行、全角空格、空表格行等），以便验证后续"提取 + 清洗"的效果。
运行：.venv_rag311\Scripts\python.exe scripts/generate_samples.py
"""
import os
from pathlib import Path

# 用完整路径指向 Python 解释器环境里可用的库
import fitz          # PyMuPDF，处理 PDF
import docx          # python-docx，处理 Word
import openpyxl      # 处理 Excel
from pptx import Presentation
from pptx.util import Pt

OUT = Path(__file__).resolve().parents[1] / "data" / "samples"
OUT.mkdir(parents=True, exist_ok=True)


def make_txt():
    # 故意混入：全角空格、行尾空格、多处连续空行、乱序换行
    text = (
        "智能仓储管理系统产品手册　v1.2\n"
        "\n"
        "一、系统简介   \n"
        "本系统用于仓库货物的入库、出库、盘点与库存预警。\n"
        "  主要功能包括：批次管理、库位分配、效期监控。\n"
        "\n\n\n"
        "二、核心能力\n"
        "1. 多仓协同：支持总部与多个分仓的数据同步。\n"
        "2. 批次追溯：按批次记录入库与出库明细，可回溯到供应商。\n"
        "3. 智能预警：低于安全库存自动生成补货建议。\n"
        "\n"
        "三、部署要求\n"
        "服务器需 8C16G 以上，数据库使用 MySQL 8.0。\n"
    )
    (OUT / "sample1_warehouse.txt").write_text(text, encoding="utf-8")


def make_md():
    md = (
        "<!-- 厂商备注：本说明由市场部整理 -->\n"
        "# 支付网关接入指引\n\n"
        "本文档指导商户接入支付网关，覆盖基础对接与异常处理。\n\n"
        "## 1. 接入流程\n\n"
        "- 申请商户号\n"
        "- 配置密钥\n"
        "- 联调测试\n\n"
        "## 2. 接口说明\n\n"
        "| 接口 | 方法 | 说明 |\n"
        "| --- | --- | --- |\n"
        "| /pay | POST | 发起支付 |\n"
        "| /refund | POST | 申请退款 |\n\n"
        "## 3. 常见问题\n\n"
        "1. 签名失败：检查密钥是否一致。\n"
        "2. 超时：建议设置重试与幂等。\n"
    )
    (OUT / "sample2_payment.md").write_text(md, encoding="utf-8")


def make_docx():
    d = docx.Document()
    # 页眉 / 页脚（噪音，应被识别并剔除）
    for sec in d.sections:
        sec.header.paragraphs[0].text = "内部资料 · 禁止外传"
        sec.footer.paragraphs[0].text = "第 1 页 / 共 1 页"
    d.add_heading("公务用车管理规定", level=1)
    d.add_paragraph("为了规范公司公务用车使用，特制定本规定。")
    d.add_heading("一、适用范围", level=2)
    d.add_paragraph("本规定适用于公司所有正式员工作业用车的申请与审批。")
    d.add_heading("二、申请流程", level=2)
    d.add_paragraph("用车人需提前一个工作日提交申请，经部门负责人审批后派车。")
    # 故意加一个空段落（噪音）
    d.add_paragraph("")
    # 表格
    table = d.add_table(rows=3, cols=3)
    table.style = "Table Grid"
    headers = ["车型", "核载", "用途"]
    vals = [["商务车", "7 座", "接待"], ["货车", "2 吨", "运输"]]
    for j, h in enumerate(headers):
        table.rows[0].cells[j].text = h
    for i, row in enumerate(vals):
        for j, v in enumerate(row):
            table.rows[i + 1].cells[j].text = v
    d.save(str(OUT / "sample3_vehicle.docx"))


def make_pdf():
    doc = fitz.open()
    page = doc.new_page()
    # 页眉 / 页脚噪音(用中文字体，确保可被文本层提取)
    page.insert_text((72, 40), "XX 电力集团 · 操作规程", fontname="china-s", fontsize=10)
    body = [
        "高压配电柜巡检操作规程",
        "",
        "1. 巡检前须确认设备已停电，并挂好警示牌。",
        "2. 使用绝缘工具，佩戴绝缘手套与护目镜。",
        "3. 每 4 小时记录一次柜内温湿度与运行电流。",
        "4. 发现异常响声或焦味应立即上报并隔离。",
    ]
    y = 100
    for line in body:
        page.insert_text((72, y), line, fontname="china-s", fontsize=11)
        y += 24
    # 页脚噪音
    page.insert_text((72, 700), "第 12 页 / 共 20 页", fontname="china-s", fontsize=9)

    # 一个 2x3 表格(演示 PDF 表格识别)
    tx0, ty0, cw, rh = 72, 440, 150, 28
    cells = [["设备", "数量", "状态"], ["变压器", "2", "运行"]]
    for r in range(len(cells) + 1):
        yy = ty0 + r * rh
        page.draw_line((tx0, yy), (tx0 + len(cells[0]) * cw, yy), width=0.7)
    for c in range(len(cells[0]) + 1):
        xx = tx0 + c * cw
        page.draw_line((xx, ty0), (xx, ty0 + len(cells) * rh), width=0.7)
    for r, row in enumerate(cells):
        for c, val in enumerate(row):
            page.insert_text((tx0 + c * cw + 6, ty0 + r * rh + 20), val, fontname="china-s", fontsize=10)

    doc.save(str(OUT / "sample4_power.pdf"))
    doc.close()


def make_xlsx():
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "设备参数表"
    # 标题行 + 空行（噪音）
    ws["A1"] = "某型号冷水机组技术参数"
    ws["A3"] = "参数名称"
    ws["B3"] = "数值"
    ws["C3"] = "单位"
    rows = [
        ["制冷量", "1250", "kW"],
        ["输入功率", "198", "kW"],
        ["COP", "6.3", "-"],
        ["冷却水流量", "215", "m3/h"],
        ["噪声", "75", "dB(A)"],
    ]
    for i, row in enumerate(rows, start=4):
        for j, v in enumerate(row, start=1):
            ws.cell(row=i, column=j, value=v)
    # 末尾合计行 + 空行
    ws.cell(row=10, column=1, value="合计")
    wb.save(str(OUT / "sample5_chiller.xlsx"))


def make_pptx():
    prs = Presentation()
    slide1 = prs.slides.add_slide(prs.slide_layouts[1])  # 标题+内容
    slide1.shapes.title.text = "项目周报 · 第 12 周"
    body = slide1.placeholders[1].text_frame
    body.text = "本周完成了检索模块的联调"
    p = body.add_paragraph()
    p.text = "下周计划：数据准备管线上线测试"
    slide2 = prs.slides.add_slide(prs.slide_layouts[1])
    slide2.shapes.title.text = "风险与排期"
    body2 = slide2.placeholders[1].text_frame
    body2.text = "风险1：样本数据不足"
    p2 = body2.add_paragraph()
    p2.text = "风险2：PDF 解析待加固"
    prs.save(str(OUT / "sample6_report.pptx"))


if __name__ == "__main__":
    make_txt()
    make_md()
    make_docx()
    make_pdf()
    make_xlsx()
    make_pptx()
    print("样例文档已生成于:", OUT)
    for f in sorted(OUT.iterdir()):
        print(" -", f.name, f.stat().st_size, "bytes")
