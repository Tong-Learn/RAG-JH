# RAG 学习实践项目（简历项目）

一个从零实现的 **RAG（检索增强生成）** 管线：**数据准备 → 结构感知切片 → 向量化 → 检索 → 生成**。
定位是**学习 + 实践 RAG，并集成为简历项目**；设计上刻意走「**结构感知 + 自研**」路线，与主流框架 / LangChain 默认方案形成对照（见「简历亮点」）。

> 详细设计在 `docs/`；开发记忆与当前进展见 `MEMORY.md`；「原问题→策略→效果→状态」复盘见 `docs/问题与处理记录.md`；**环境搭建/运行故障**见 `docs/故障报告.md`。

---

## 处理对象 / 适用范围

本项目的处理对象是 **《晶核》游戏官方更新公告**（27 份 PDF）—— 头尾规整、标题化、含列表/表格的**高结构化公告**。它是**一套针对"单份公告型文档 + 公告集合问答"的自研 RAG 链路**，**不是**一个可泛化到任意文档的通用 RAG 框架：

- 管线：提取 → 结构感知切片 → 向量化 → 检索 → 生成（见下文「技术链路」）。
- 对**无结构 / 复杂排版 / 多栏混排**的文档，解析与切片策略需单独调整（见 `docs/数据准备方案.md`、`docs/切片方案.md`）。
- **样例**：仓库附带 `examples/`（**真实公告**的解析产物 + 一次端到端问答演示），见「样例」。完整源语料 `test/`（68.6MB PDF）**不纳入版本控制**，仅本地运行用。

---

## 技术栈（Tech Stack）

| 层 | 选型 |
|---|---|
| 语言 / 运行环境 | Python 3.11（`.venv_rag311`） |
| 文档解析 | PyMuPDF (fitz) / python-docx / openpyxl / python-pptx / 标准库 |
| 向量化 | DashScope `qwen3.7-text-embedding`（1024 维） |
| 向量库 | chromadb 0.6.3 + chroma-hnswlib 0.7.6（`hnsw:space=cosine`） |
| 生成 | DashScope chat（默认 `qwen-plus`） |
| 依赖版本 | onnxruntime 1.19.2 · posthog 3.x · numpy `<2` |

> 版本锁定原因、故障与排障见 `docs/故障报告.md`；更细的选型与排除项见 `docs/技术栈.md`。

---

## 技术链路（5 步）

```
源文档(word/pdf/excel/ppt/txt/md)
   │  ① 提取(原生格式解析,保结构)  → Block(kind=heading/para/table)
   ▼  ② 清洗(规范化+去噪+丢空块)
【数据准备】 data/processed/*.jsonl（每行一个 Block）+ *.md（人核对）

   │  ③ 结构感知切片(标题栈→section_path / 表格整体 / 按句切 / 句边界 overlap / 过短回并)
   ▼  ④ 向量化(DashScope qwen3.7) → chromadb 入库
【切片+向量化】 data/chunks/*.chunks.jsonl → data/chroma（集合 rag_chunks, cosine）

   │  ⑤ 检索(同源 embedding) → 拼上下文 → 千问生成带引用回答
【检索+生成】 scripts/query_retrieval.py / run_rag.py
```

## 核心设计（为什么这样设计）

- **核心中间表示**：`Block(kind=heading/para/table)` —— 用**语义标签**承载结构（标题/段落/表格），而非裸文本。不是把文档转成 Markdown（有损、丢失表格），而是按原格式直接解析，`md` 仅作人工核对。
- **结构感知切片**（而非固定大小）：直接读 `Block.kind` 判边界——标题为一节边界（维护标题栈得 `section_path`）、表格整体不可拆、段落按目标大小聚合、超长**按句切（不切半句）**、过短回并、切点附「句边界 overlap」。
- **溯源坐标**：每个 chunk 带 `id/doc/section_path/text/rows/start_block/end_block`，**进库即带坐标**；检索命中后靠 `doc + section_path + start/end_block` 回溯回原文，满足「附原文链接」。
- `id` 用**全局递增短键** `chunk_{序号:06d}`（不再含标题），完整标题/溯源放在 `doc`/`section_path` 等 metadata。

## 简历亮点（个性化 vs 生产）

| 本项目做法 | 生产对照（Unstructured / LangChain 主流） |
|---|---|
| 原生格式直接解析 + `Block(kind=...)` 保结构，md 仅作核对 | 复杂文档用 Unstructured/Docling/Textract 返回**类型化元素** |
| 结构感知切片（标题栈、表格整体、按句切、句边界 overlap、过短回并） | 常为固定大小 / `RecursiveCharacterTextSplitter`（会切碎语义） |
| 一套自研可审计产物（jsonl + md + manifest + chunks + chroma 行） | 通常直接 Loader → Document → 切片 → 入库，无中间层 |
| 评估诚实性：小语料下高指标仅视为**合理基线** | 区分「真实指标」与「小语料自嗨」 |
| 完整踩坑溯源（chromadb 段错误 → 锁 0.6.3、batch_size 边界崩溃等） | 体现工程排障能力（见 `docs/故障报告.md`） |

> 更多「疑问 vs 生产对比」见 `docs/疑问解答.md`、各方案文档附录。

---

## 快速开始

**统一在 `.venv_rag311`（Python 3.11）下运行**。

```powershell
# 1. 数据准备：处理源文档 → processed/*.jsonl + *.md + manifest.jsonl
.venv_rag311\Scripts\python.exe -m preprocessing.pipeline --src <源文档目录> --out data\processed

# 2. 切片：processed/*.jsonl → data/chunks/*.chunks.jsonl
.venv_rag311\Scripts\python.exe -m scripts.run_chunking

# 3. 向量化：chunks → DashScope embedding → chromadb（幂等重建）
.venv_rag311\Scripts\python.exe -m scripts.run_vectorize

# 4. 检索冒烟
.venv_rag311\Scripts\python.exe -m scripts.query_retrieval "问题" [-k N]

# 5. 端到端生成（带引用回答）
.venv_rag311\Scripts\python.exe -m scripts.run_rag "问题" [-k 4] [--model qwen-plus]
```

- 配置在 `.env`：`DASHSCOPE_API_KEY` + `DASHSCOPE_EMBEDDING_MODEL`（默认 `qwen3.7-text-embedding`）。**不要提交 `.env`**（已 `.gitignore`）；可复制 `.env.example` 改名 `_env` 查看格式。
- 源语料 `test/`（27 份真实游戏公告 PDF，约 68.6MB）**不纳入版本控制**（已 `.gitignore`），仅本地运行用。复现方式见下方「源语料说明」。

> **源语料说明**：仓库不携带 `test/` 原件（大体积 + 版权考虑）。复现链路时，将任意 PDF/Word/Excel/PPT/txt/md 放入本地目录，用 `preprocessing.pipeline --src <源文档目录>` 处理即可；
> 或运行 `python scripts/generate_samples.py` 生成含噪音的多格式样例（入库前提是已配置 `DASHSCOPE_API_KEY`）。

## 目录结构

```
RAG/
├─ preprocessing/        # ① 数据准备：extractors / cleaner / pipeline
├─ chunking/             # ③ 切片：chunker.py
├─ embedding/            # ④ 向量化：dashscope_embedder.py
├─ retrieval/            # 检索增强：hybrid.py(BM25+向量 RRF) / reranker.py(DashScope 重排)
├─ scripts/              # generate_samples / run_chunking / run_vectorize
│                        #   query_retrieval / eval_retrieval / run_rag / gen_answers
├─ tests/                # pytest：test_chunker / test_cleaner（切片+清洗逻辑）
├─ docs/                 # 方案 + 技术栈 + 问题与处理记录 + 故障报告 + 疑问解答 + 评测复盘
├─ examples/             # 真实公告的解析产物样例（提取 Blocks / 切片 chunks，受版本控制）
├─ data/                 # processed/chunks/chroma/eval（运行后生成，不入库）
├─ requirements.txt      # 直接依赖 + 版本说明（锁版原因见 docs/故障报告.md）
└─ MEMORY.md             # 开发记忆 / 当前进展 / 待办
```

## 文档索引

| 文档 | 内容 |
|---|---|
| `examples/README.md` | 真实公告**解析产物样例**：提取 Blocks + 切片 chunks，含局限说明 |
| `docs/评测复盘.md` | 基础/复杂/优化后问答 + 检索指标对比（含阈值拒答、诚实边界） |
| `docs/技术栈.md` | 技术栈与排除项明细 |
| `docs/面试要点.md` | RAG 标准化流程讲解 + 各步「我的思考/解决思路/延伸思考」+ 面试级回答 |
| `docs/数据准备方案.md` | 数据准备（提取/清洗/中间表示）方案 |
| `docs/切片方案.md` | 结构感知切片方案与设计取舍 |
| `docs/向量化方案.md` | 向量化与 chromadb 入库方案 |
| `docs/疑问解答.md` | 各阶段「我的思考 / 解决思路 / 与生产对比」 |
| `docs/问题与处理记录.md` | 各问题「原问题→影响→方案→效果→状态」复盘与残留清单 |
| `docs/故障报告.md` | 环境搭建/运行故障、版本锁定、排障过程 |
