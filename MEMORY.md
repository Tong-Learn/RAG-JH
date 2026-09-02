# 项目记忆 — RAG 学习实践项目

> 定位：学习 + 实践 RAG，集成简历项目；全程「结构感知 + 自研」。
> **处理对象是《晶核》游戏官方更新公告（27 份 PDF），不是通用 RAG 框架**；面向"单份公告型文档 + 公告集合问答"，泛化到任意文档需调整解析/切片策略。
> 本文档只留**当前状态与要点**；环境/版本故障、设计取舍、问题复盘分别见文档索引。各阶段「我的思考 / 解决思路 / 延伸思考」见 `docs/疑问解答.md`。

---

## 技术方案概览（5 步）

```
① 数据准备   各格式原生解析 → Block(kind=heading/para/table) → 清洗(规范化/去噪)
② 切片       结构感知：标题栈→section_path / 表格整体 / 目标大小聚合 / 按句切 / 句边界 overlap / 过短回并
③ 向量化     DashScope qwen3.7-text-embedding(1024) → chromadb 0.6.3(hnsw:space=cosine)
④ 检索       同源 embedding 向量化 → 余弦 top-k → 带溯源坐标(doc+section_path+start/end_block)
⑤ 生成       检索上下文 → DashScope chat(默认 qwen3.8-max) → 带[1][2]引用回答
```

- **核心中间表示**：`Block(kind=...)` 语义标签承载结构，md 仅作人工核对。
- **溯源**：chunk 进库即带坐标，命中后靠 `doc + section_path + start/end_block` 回溯回原文。
- **chunk 结构**：`id / doc / section_path / text / rows / start_block / end_block`；`id` 用全局递增短键 `chunk_{序号:06d}`。

---

## 当前进展（2026-08，真实语料 27 份游戏公告）

- **全链路跑通**：数据准备 → 切片 → 向量化 → 检索 → 生成，均在 `.venv_rag311`（Python 3.11）。
- **规模（2026-08 扩充后）**：`test` **54 份公告** → **578 chunk** → chroma **578 行**（集合 `rag_chunks`，cosine）。
- **新增能力**：溯源到页（`Block/Chunk` 带 `page`，chroma metadata `start_page/end_page`）；相似度阈值拒答（`run_rag --min-score`，校准值 0.70）；两套评测集 `qa_basic.jsonl` + `qa_complex.jsonl`；`retrieval/` 混合检索（BM25+向量 RRF）+ DashScope rerank。
- **检索评估（新双集，top-3；详见 docs/评测复盘.md；c11 地面真值已修正后为最终值）**：
  - 复杂集(13题，含跨文档/语义近似/精确词)：**纯向量** recall@3=1.0 / MRR=0.923 / P@1=0.846 / prec@3=0.538 / nDCG=0.943；**混合(BM25+向量RRF)**→0.910/0.846/0.513/0.933；**混合+重排**→1.0/**1.000**/**1.000**/**0.641**/**1.000**。
  - **诚实结论**：BM25+RRF 单独几乎中性（甚至 MRR 略降），**真正质变来自 DashScope 重排**（P@1/MRR/nDCG 拉满）。不夸大混合检索，突出 rerank 是杠杆。
  - 基础集(18题，含 8 道无答/无关负样本)：recall@3=0.556（负样本无 expected_doc 拉低分母）；阈值校准：正类 top-1 min=0.673/中位0.804，负类 max=0.719/中位0.591（0.673~0.719 有重叠区，精确价位题 c08 top1=0.589 被拒为边界情况）。
- **RAG 生成**：`run_rag.py` 用 `qwen3.8-max`，回答带 `[1][2]` 引用 + 页码；符合 min-score 才生成，否则**拒答**「资料中没有相关信息」。
- **工程收尾**：`requirements.txt`（锁版本+说明）、`tests/test_chunker.py` + `tests/test_cleaner.py`（pytest 12 通过）。
- **已开源**：仓库 <https://github.com/Tong-Learn/RAG-JH>（public，`main`；`test/`、`data/` 产物、`.env`、`.venv_rag311` 未纳入）。

---

## 待办清单（距离「简历级」还差什么）

| 优先级 | 事项 |
|---|---|
| **P0 可信度** | P0-2 数据准备·去重；P0-3 溯源坐标生产化 ~~（改为逐层 page，`start/end_page` 实现「附原文链接」，✅ 2026-08）~~ |
| **P1 指标** | P1-1 ~~扩真实语料（27→54 份，578 chunk）~~ ✅；P1-2 ~~造更硬问题（qa_basic/qa_complex 两套）~~ ✅；P1-3 指标细到 chunk/句级（部分：doc 级 + precision@k） |
| **P2 效果** | P2-1 ~~rerank（DashScope qwen3.7-text-rerank）~~ ✅；P2-2 ~~混合检索（BM25+向量 RRF）~~ ✅；P2-3 切片参数调优；P2-4 生成侧评估（已做检索侧对比 + 生成回答复盘，RAGAS 式指标未做） |
| **P3 收尾** | P3-1 端到端演示脚本（gen_answers 为批量生成，交互式 demo 未做）；P3-2 简历技术点凝练；P3-3 ~~清理临时目录/复核 `.gitignore`~~ ✅ 已上传 GitHub |

---

## 文档索引

| 文档 | 内容 |
|---|---|
| `examples/README.md` | 真实公告**解析产物样例**（提取 Blocks / 切片 chunks）+ 局限说明 |
| `docs/评测复盘.md` | 基础/复杂/优化后问答 + 检索指标对比（含阈值拒答、诚实边界） |
| `docs/技术栈.md` | 技术栈与刻意排除项 |
| `docs/面试要点.md` | RAG 标准化流程讲解 + 各步「我的思考/解决思路/延伸思考」+ 面试级回答口径 |
| `docs/数据准备方案.md` | 数据准备（提取/清洗/Block 中间表示）方案 |
| `docs/切片方案.md` | 结构感知切片方案与设计取舍 |
| `docs/向量化方案.md` | 向量化与 chromadb 入库方案 |
| `docs/疑问解答.md` | 各阶段「我的思考 / 解决思路 / 与生产对比」（面试深挖核心） |
| `docs/问题与处理记录.md` | 各问题「原问题→影响→方案→效果→状态」复盘 + 残留清单 + 历史基线 |
| `docs/故障报告.md` | 环境搭建/运行故障、版本锁定、排障过程 |
