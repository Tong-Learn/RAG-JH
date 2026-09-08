# 游戏公告 RAG 检索增强问答系统

一个从零实现、**结构感知**的 RAG（检索增强生成）管线：面向《晶核》游戏官方更新公告 PDF，把公告解析成带语义结构的中间表示，做**结构感知切片 → 向量化 → 混合检索 + 重排 → 带引用生成**；回答忠实于原文、附**页码溯源**，检索置信度不足时**拒答**。

设计上刻意走「结构感知 + 自研」路线，与「转 Markdown + 固定大小切片」的主流默认方案形成对照。

---

## 简介

- **输入**：游戏官方更新公告 PDF（源语料 54 份）。
- **输出**：自然语言问答，回答带 `[1][2]` 引用与来源（文档名 + 章节路径 + 页码）。
- **核心中间表示**：`Block(kind=heading/para/table, page)` —— 用语义标签承载结构，而非裸文本。
- **两条检索路线**：纯向量（基线）/ 混合检索 + 重排（优化）。

---

## 配置

**运行环境**：Python 3.11（推荐 venv；本项目使用 `.venv_rag311`）。

```powershell
# 创建虚拟环境并安装依赖
uv venv --python 3.11 .venv_rag311
.venv_rag311\Scripts\python.exe -m pip install -r requirements.txt
```

**密钥配置**：复制 `.env.example` 为 `.env`，填入 DashScope API Key。

```
DASHSCOPE_API_KEY=sk-xxxxxxxx
DASHSCOPE_EMBEDDING_MODEL=qwen3.7-text-embedding
```

> `.env` 已加入 `.gitignore`，不会提交。

**源语料**：把公告 PDF 放入 `test/`（仓库不携带源 PDF，需自行放置；默认源目录即 `test/`）。

---

## 技术栈

| 层 | 选型 |
|---|---|
| 语言 / 运行环境 | Python 3.11 |
| 文档解析 | PyMuPDF（文本层 + 字号/加粗 + `find_tables()`） |
| 向量化 | DashScope `qwen3.7-text-embedding`（1024 维） |
| 向量库 | chromadb 0.6.3 + chroma-hnswlib 0.7.6（`hnsw:space=cosine`） |
| 检索 | 同源向量余弦 top-k + 混合检索（`rank_bm25` + `jieba` 与向量 RRF 融合） |
| 重排序 | DashScope `qwen3.7-text-rerank` |
| 生成 | DashScope chat `qwen3.8-max` |
| 测试 | pytest |

依赖与版本锁定说明见 `requirements.txt`。

---

## 功能设计

### 管线（5 步）

```
源 PDF
  │ ① 提取：get_text("dict") 逐行取字号/加粗 → y-gap 段落合并 → 标题字号启发式 + 跨行合并
  ▼    → 表格 find_tables() + _is_fake_table() 真伪判别 → 元数据/页码正则过滤
【数据准备】 data/processed/*.jsonl（每行一个 Block）+ *.md（人工核对）

  │ ② 清洗：空白规范化、页码/元数据去噪、丢空块
  │ ③ 结构感知切片：标题栈 → section_path / 表格整体 / 按句切 / 句边界 overlap / 过短回并
  ▼    → data/chunks/*.chunks.jsonl（每 chunk 带溯源坐标）

  │ ④ 向量化：DashScope 嵌入 → chromadb 入库（cosine）
  │ ⑤ 检索：纯向量 或 混合+重排 → 拼上下文 → 阈值拒答 → 生成带引用回答
【检索+生成】 scripts/query_retrieval.py / eval_retrieval.py / run_rag.py / gen_answers.py
```

### 关键设计

- **语义中间表示 `Block(kind=heading/para/table, page)`**：按原格式直接解析，不转 Markdown（有损、拉平表格），`md` 仅作人工核对。
- **结构感知切片**（非固定大小）：标题为一节边界（维护标题栈得 `section_path` 全路径）、表格整体不拆、段落按目标大小聚合、超长**按句切（不切半句）**、切点附**句边界 overlap**、过短回并。
- **溯源坐标**：每个 chunk 带 `id / doc / section_path / text / rows / start_block / end_block / start_page / end_page`，命中后凭 `doc + section_path + start/end_block(+page)` **回溯回原文 + 页码**。
- **两条检索路线**：纯向量（基线）/ 混合检索（BM25 + 向量 RRF）叠加重排。
- **阈值拒答**：top-1 相似度低于阈值 → 拒答「资料中没有相关信息」，**不调生成 API**，防幻觉。

---

## 运行指南

> 统一在 `.venv_rag311` 下运行。**数据准备（①~③）两条路线共享**，之后按路线分叉。

### 通用前缀（数据准备 → 向量化）

```powershell
# 1. 提取 + 清洗：test/*.pdf → data/processed/*.jsonl + *.md + manifest.jsonl
.venv_rag311\Scripts\python.exe -m preprocessing.pipeline --src test --out data\processed

# 2. 结构感知切片：processed → data/chunks/*.chunks.jsonl
.venv_rag311\Scripts\python.exe -m scripts.run_chunking

# 3. 向量化：chunks → DashScope 嵌入 → chromadb（幂等重建）
.venv_rag311\Scripts\python.exe -m scripts.run_vectorize
```

### 路线 A · 纯向量

```powershell
# 检索冒烟
.venv_rag311\Scripts\python.exe -m scripts.query_retrieval "问题" -k 3

# 检索评估（basic + complex 两套评测集）
.venv_rag311\Scripts\python.exe -m scripts.eval_retrieval --mode vector_only -k 3 --sets all

# 端到端生成（带引用 + 页码；--min-score 拒答）
.venv_rag311\Scripts\python.exe -m scripts.run_rag "问题" -k 4 --min-score 0.70

# 批量生成回答
.venv_rag311\Scripts\python.exe scripts\gen_answers.py --mode vector_only --sets all --min-score 0.70 --out data\eval\answers_vector.jsonl
```

### 路线 B · 混合检索 + 重排

```powershell
# 检索评估（BM25+向量 RRF 生成候选 → DashScope 重排）
.venv_rag311\Scripts\python.exe -m scripts.eval_retrieval --mode hybrid_rerank -k 3 --sets all

# 批量生成回答（重排 relevance 尺度，阈值 ~0.80）
.venv_rag311\Scripts\python.exe scripts\gen_answers.py --mode hybrid_rerank --sets all --min-score 0.80 --out data\eval\answers_rerank.jsonl
```

### 测试

```powershell
.venv_rag311\Scripts\python.exe -m pytest tests/ -q
```

---

## 验收成果（数据准备自验收）

对全量重跑后的产物做自验收，均达标：

| 项 | 结果 |
|---|---|
| 语料 / 块数 | `test/` 54 篇 → `data/processed` **1788 块**（heading 673 / para 1115） |
| 切片 | **605 chunk** → chroma **605 行** |
| 元数据残留 | 含「晶核CoA / 听全文」的块 = **0** |
| 顺序正确性 | 抽检「维护更新内容」位于正文**之前** |
| 标题合并 | 抽检长标题**完整合并**（未被拆行） |
| 单元测试 | pytest **12 通过**（chunker / cleaner） |

---

## 指标

评测集：`data/eval/qa_basic.jsonl`（24 题）+ `qa_complex.jsonl`（19 题），共 **43 题**；top-3。

| 集 | 路线 | recall@3 | MRR | P@1 | precision@3 | nDCG@3 |
|---|---|---|---|---|---|---|
| basic(24) | 纯向量 | 0.667 | 0.639 | 0.625 | 0.417 | 0.646 |
| basic(24) | 混合+重排 | 0.667 | 0.646 | 0.625 | 0.417 | 0.651 |
| complex(19) | 纯向量 | 1.000 | 0.947 | 0.895 | 0.526 | 0.961 |
| complex(19) | **混合+重排** | 1.000 | **1.000** | **1.000** | **0.649** | **1.000** |

---

## 结果分析

- **basic 的 recall@3 = 0.667 是「结构所致」，不是检索差**：basic 24 题中含 **8 道无答 / 领域无关负样本**（`expected_doc` 为空，本就不可能命中），它们被计入 recall 分母，把指标拉低；**16 道可答题 + complex 19 题共 35 道可答题，全部 top-3 命中**。
- **complex 上混合+重排全面更优**：P@1 0.895→**1.000**、MRR 0.947→**1.000**、precision@3 0.526→**0.649**、nDCG 0.961→**1.000**。
- **重排的额外价值**：重排 relevance 分把可答/不可答**清晰分开**（可答 min=0.907、无答/无关 max=0.741），而纯向量是重叠的（0.655 / 0.719）——重排后阈值可干净取 ~0.80。
- **复杂评测集**覆盖跨文档辨析、语义近似但答案不同、精确词/数值三类问题（如「征战之塔第一期 vs 第二期」「机械要塞 24 名 vs 铸骨迷巢 24 名」），用于检验检索的区分度。
- **已知局限**：源语料为高结构化游戏公告（PDF 文本层），对强装饰版式 / 歌词类文档（如三周年庆典、音爆EP）的标题识别属启发式边界；对无文本层扫描件需另加 OCR。

---

## 目录结构

```
RAG/
├─ preprocessing/        # ① 数据准备：extractors（PDF）/ cleaner / pipeline
├─ chunking/             # ③ 切片：chunker.py
├─ embedding/            # ④ 向量化：dashscope_embedder.py
├─ retrieval/            # 检索：hybrid.py（BM25+向量 RRF）/ reranker.py（重排）/ search.py
├─ scripts/              # run_chunking / run_vectorize / query_retrieval
│                        #   eval_retrieval / run_rag / gen_answers / _common
├─ tests/                # pytest：test_chunker / test_cleaner
├─ examples/             # 真实公告的解析产物样例（提取 Blocks / 切片 chunks）
├─ data/                 # processed/chunks/chroma/eval（运行后生成，不入库）
├─ requirements.txt      # 依赖 + 版本说明
└─ README.md
```

---

## 样例

`examples/` 下附带 2 份真实公告经管线加工后的产物（`_extracted.md` 人工核对 / `_blocks.jsonl` 语义块 / `_chunks.jsonl` 切片），便于直观查看数据形态。完整源语料 `test/` 与 `data/` 产物不入库。

---

## 说明

- **源语料不入库**：`test/`（PDF）与 `data/` 产物均在 `.gitignore` 中，需本地放置源 PDF 后自行运行。
- **复现**：按「运行指南」先跑数据准备三步，再选一条检索路线评估/生成。
