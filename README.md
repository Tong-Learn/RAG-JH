# 游戏公告 RAG 检索增强问答系统

一个从零实现、**结构感知**的 RAG（检索增强生成）管线：面向《晶核》游戏官方更新公告 PDF，把公告解析成带语义结构的中间表示，做**结构感知切片 → 向量化 → 混合检索 + 重排 → 带引用生成**；回答忠实于原文、附**页码溯源**，检索置信度不足时**拒答**。

> **项目定位（重要）**：本仓库是**源码 / 开发维护仓库**，不是可直接分发的应用。
> 语料（`test/`）与全部产物（`data/processed|chunks|chroma`）**均不入库**，因此**克隆后不能直接问答** ——
> 必须先本地构建索引（放 PDF → 跑数据准备三步，见「运行指南」）。`run_rag` 是**本地验证/演示入口**；
> 若要做成面向终端用户的产品，应由**部署方预先构建好索引**再以服务形式提供。

---

## 文档

设计文档在 `docs/`（随仓库提供），按管线阶段组织：

| 文档 | 内容 |
|---|---|
| `docs/数据准备方案.md` | ① 数据准备：PDF 提取（标题判定 / 视觉行聚类 / 段落合并 / 表格）、清洗、产物 |
| `docs/切片方案.md` | ② 切片：结构感知规则、关键参数、chunk 结构 |
| `docs/向量化方案.md` | ③ 向量化：嵌入模型、chromadb 入库映射、运行约束 |
| `docs/检索方案.md` | ④ 检索：两条路线、RRF 与重排、指标口径 |
| `docs/生成方案.md` | ⑤ 生成：提示词结构、chunk id 引用、阈值拒答、调用重试 |
| `docs/技术栈.md` | 技术栈与刻意排除项 |
| `docs/验收复盘_混合重排.md` | ⑥ 评测结果：检索指标、四路消融、阈值校准、生成侧指标与错例 |

权威数字（块数 / chunk 数 / 各项指标）以仓库根目录的 `metrics.json` 为准。

---

## 简介

- **输入**：游戏官方更新公告 PDF（源语料 54 份）。
- **输出**：自然语言问答，回答带 `[chunk_xxxxxx]` 引用与来源（文档名 + 章节路径 + 页码）。
- **核心中间表示**：`Block(kind=heading/para/table, page)` —— 用语义标签承载结构，而非裸文本。
- **两条检索路线**：纯向量（基线）/ 混合检索 + 重排（默认）。
- **使用前提**：`.env`（API Key）+ `data/chroma`（向量库）+ `data/chunks`（混合路线的 BM25 语料）——后两者需本地构建。

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
DASHSCOPE_EMBEDDING_MODEL=qwen3.7-text-embedding-flash
```

> `.env` 已加入 `.gitignore`，不会提交。

**源语料**：完整语料在 `test/`（54 份公告 PDF，本地运行用，未入库）；`examples/pdf/` 另附 **3 份抽样公告 PDF** 作可直接运行的样例（见「样例」）。仓库不携带全部 54 份源 PDF。

---

## 技术栈

| 层 | 选型 |
|---|---|
| 语言 / 运行环境 | Python 3.11 |
| 文档解析 | PyMuPDF（文本层 + 字号/加粗 + `find_tables()`） |
| 向量化 | DashScope `qwen3.7-text-embedding-flash`（1024 维） |
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
- **阈值拒答**：top-1 分数低于阈值 → 固定回答「**当前资料相关性不足，无法回答**」（`run_rag.REJECT_ANSWER`），**不调生成 API**，防幻觉。两条路线 score 尺度不同，**各配校准阈值**（纯向量 0.55 / 重排 0.58，见 `scripts/calibrate_threshold.py`）。
  与**模型级拒答**（资料存在但未提及，模型答「资料中没有相关信息」）是两种不同情形，措辞刻意区分：「检索没找到相关内容」vs「找到了但资料里没写」。

---

## 运行指南

> 统一在 `.venv_rag311` 下运行。**数据准备（①~③）两条路线共享**，之后按路线分叉。

### 通用前缀（数据准备 → 向量化）

> **首次使用必须先跑这一步**：语料与索引均不入库，跳过它直接跑 `run_rag` 会因缺少向量库/切片产物而无法运行
> （脚本会给出明确的前置检查提示，而不是抛底层异常）。

```powershell
# 0. 把公告 PDF 放入 test/（仓库不携带全量语料）
# 1. 提取 + 清洗：test/*.pdf → data/processed/*.jsonl + *.md + manifest.jsonl
.venv_rag311\Scripts\python.exe -m preprocessing.pipeline --src test --out data\processed

# 2. 结构感知切片：processed → data/chunks/*.chunks.jsonl
.venv_rag311\Scripts\python.exe -m scripts.run_chunking

# 3. 向量化：chunks → DashScope 嵌入 → chromadb（幂等重建）
.venv_rag311\Scripts\python.exe -m scripts.run_vectorize
```

### 检索冒烟（人工核对命中与溯源坐标，无断言）

```powershell
# 默认走混合+重排（与 run_rag 一致）；打印 doc/section/blocks/page 全坐标便于回溯核对
.venv_rag311\Scripts\python.exe -m scripts.query_retrieval "问题" -k 5

# 切纯向量
.venv_rag311\Scripts\python.exe -m scripts.query_retrieval "问题" --mode vector_only
```

### 路线 A · 纯向量

```powershell
# 检索评估（新评测集 simple k=3 / complex k=5）
.venv_rag311\Scripts\python.exe -m scripts.eval_retrieval --mode vector_only --sets all

# 端到端生成（带引用 + 页码；--min-score 缺省取该路线校准值 0.55）
.venv_rag311\Scripts\python.exe -m scripts.run_rag "问题" --mode vector_only -k 5

# 批量生成回答
.venv_rag311\Scripts\python.exe scripts\gen_answers.py --mode vector_only --sets all --out data\eval\answers_v2_vector.jsonl
```

### 路线 B · 混合检索 + 重排（默认路线）

```powershell
# 检索评估（BM25+向量 RRF 生成候选 → DashScope 重排）
.venv_rag311\Scripts\python.exe -m scripts.eval_retrieval --mode hybrid_rerank --sets all

# 端到端生成（默认路线；--min-score 缺省取校准值 0.58）
.venv_rag311\Scripts\python.exe -m scripts.run_rag "问题" -k 5

# 批量生成回答
.venv_rag311\Scripts\python.exe scripts\gen_answers.py --mode hybrid_rerank --sets all --out data\eval\answers_v2_rerank.jsonl
```

### 消融 / 阈值校准 / 生成侧评测

```powershell
# 四路消融：混合候选+重排 / 混合候选 / 纯向量候选+重排 / 纯向量
.venv_rag311\Scripts\python.exe -m scripts.ablate_search

# 阈值重校准（输出正/负类 top-1 分布与权衡表）
.venv_rag311\Scripts\python.exe -m scripts.calibrate_threshold --mode hybrid_rerank

# 生成侧评测：忠实度 / 引用准确率 / 答案正确率 / 拒答正确性（LLM-as-judge）
.venv_rag311\Scripts\python.exe scripts\eval_generation.py --mode hybrid_rerank --answers data\eval\answers_v2_rerank.jsonl --out data\eval\gen_scores_v2_rerank.jsonl
```

### 测试

```powershell
.venv_rag311\Scripts\python.exe -m pytest tests/ -q
```

---

## 产物自检

对数据准备与切片产物做的一致性自检项：

| 项 | 结果 |
|---|---|
| 语料 / 块数 | `test/` 54 篇 → `data/processed` **1782 块**（heading 667 / para 1115） |
| 切片 | **605 chunk** → 向量库 **605 行** |
| 元数据残留 | 含「晶核CoA / 听全文」的块 = **0** |
| 顺序正确性 | 抽检「维护更新内容」位于正文**之前** |
| 标题合并 | 抽检长标题**完整合并**（未被拆行）；「常规服 / 赛季服」散落字形已复原 |
| 单元测试 | pytest **57 通过**（chunker / cleaner / 检索与生成链路 / PDF 提取 / 端到端 smoke / metrics.json 一致性 / 生成调用重试 / 运行前提检查） |

---

## 指标

向量模型 `qwen3.7-text-embedding-flash`；评测集 `data/eval/`（源文件在 `test/qa_dataset/`）——`simple_qa_30.json`（30 题，27 可答 + **3 不可答**）+ `complex_qa_20.json`（20 题，全可答），共 **50 题**；simple k=3、complex k=5。

### 检索指标

| 集 | 路线 | recall@k | MRR | P@1 | precision@k |
|---|---|---|---|---|---|
| simple(27可答, k=3) | 纯向量 | 0.889 | 0.772 | 0.704 | — |
| simple(27可答, k=3) | **混合+重排** | **1.000** | **0.975** | **0.963** | — |
| complex(20可答, k=5) | 纯向量 | 0.512 | — | — | 0.430 |
| complex(20可答, k=5) | **混合+重排** | **0.808** | — | — | **0.720** |

> complex 每题有 **2~4 个来源文档**，故 recall = 来源覆盖率（宏平均）、precision = top-k 中属于来源集的占比，**不计算 P@1/MRR**（多来源下无唯一正确答案）。

### 四路消融（定位每层收益）

| 路线 | simple recall/MRR/P@1 | complex recall/precision |
|---|---|---|
| A 混合候选 + 重排 | **1.000 / 0.975 / 0.963** | **0.808 / 0.720** |
| B 混合候选（不重排） | 0.963 / 0.907 / 0.852 | 0.667 / 0.560 |
| C 纯向量候选 + 重排 | 0.963 / 0.938 / 0.926 | 0.721 / 0.640 |
| D 纯向量（基线） | 0.889 / 0.772 / 0.704 | 0.529 / 0.450 |

- **重排是主要杠杆**（C−D）：simple recall +0.074 / MRR +0.166；complex recall +0.192 / precision +0.190
- **混合候选（BM25）亦有稳定正增量**（A−C）：simple +0.037；complex recall +0.087 / precision +0.080

### 生成侧指标（LLM-as-judge，50 题全有效）

| 路线 | 忠实度 | 引用准确率 | 答案正确率 | 拒答正确性 |
|---|---|---|---|---|
| 纯向量 | **0.990** | **0.920** | 0.640 | 0.710 |
| **混合+重排** | 0.980 | 0.930 | **0.780** | **0.970** |

> ⚠️ **口径提示（重要）**：上表数字产自**旧版提示词**（规则分散在 system + user、引用用位置编号 `{1}{2}`）。
> 此后提示词已重构为**单一 `SYSTEM_PROMPT` + `chunk id` 引用**，**当前代码与这批数字不是同一版本**。
> 若要严格对齐需重跑生成 + 判分（两路线共 100 次生成 + 100 次判分）。
> 详见 `metrics.json` → `known_issues.generation_metrics_predate_prompt_refactor`。

### 阈值（拒答值，新库重新校准）

| 路线 | 取值 | 可答下沿 | 不可答上沿 | 是否分离 | 效果 |
|---|---|---|---|---|---|
| 纯向量 | **0.55** | 0.568 | 0.676 | ❌ 重叠 | 误杀 0 / 漏拦 3 |
| 混合+重排 | **0.58** | 0.621 | 0.535 | ✅ 分离 | 误杀 0 / 漏拦 0 |

---

## 结果分析

- **默认路线取「混合 + 重排」**：综合均值 0.890（纯向量 0.688），在检索两项、答案正确率、拒答正确性上全面领先；纯向量仅在忠实度 / 引用上微幅领先，属噪声级。
- **拒答阈值必须分路线配置**：两条路线的 score 尺度不同（纯向量 = 余弦相似度；重排 = relevance 分）。重排能把可答 / 不可答**干净分开**（0.621 vs 0.535），纯向量则**重叠**（0.568 vs 0.676），只能取折中值。
- **complex 集是主要难点**：多跳、多来源（每题 2~4 篇）。数据显示**检索覆盖度直接决定生成正确率**——来源全覆盖的题答案正确率 0.667，部分覆盖的仅 0.429；瓶颈在检索未提供完整证据，而非生成忠实度（忠实度 48/50 满分、0 题编造）。
- 详细分析与逐题明细见 `docs/验收复盘_混合重排.md`。

---

## 诚实边界（如实说明，不美化）

先把话说清楚，免得指标被高估：

- **这是「技术验证 / 自研实践」项目，不是有真实用户的产品**。处理对象是《晶核》游戏公告（54 份 PDF），属于高结构化、标题化的单一类型文档；**不是**一个可泛化到任意文档的通用 RAG 框架。对无结构 / 多栏混排 / 扫描件，需另调解析与切片策略。
- **评测集是自建自评**：50 题由作者设计、自行标注来源文档，无第三方标注、无盲评。因此指标反映的是「**系统能否回答这批既定问题**」，不直接等于「真实用户关心的问题被正确回答」。（`evidence` 字段由独立大模型抽取，**仅作参考、不参与打分**；打分基准是 gold answer + 检索上下文。）
- **样本量仍小**：50 题中**不可答题只有 3 道**，阈值校准的负类样本仅 3 个，统计上很弱——两条路线的阈值都只能说「在这 3 个负样本上如此」，不足以支撑生产结论。
- **生成侧评测是 LLM-as-judge**：四指标由 `qwen3.8-max` 打分，非人工盲评，评分本身带模型偏好；与生成用的是同一模型族，存在自评偏差。且 0 / 0.5 / 1 属**有序等级**，直接求算术平均隐含「按等距数值处理」的简化。
- **complex 生成正确率偏低（0.550）**：多跳多来源问题需跨 2~4 篇整合，单轮 top-k 检索 + 生成覆盖不足；这是当前最明确的短板。
- **测试覆盖仍有限**：现有 57 项单测，**离线端到端**（提取→清洗→切片）与**交互式入口**（`run_rag.py` 主流程，依赖全部替换、不联网）已覆盖；仍缺的是 ①**在线链路**（向量化→检索→生成，需 DashScope API 与向量库，未纳入 pytest）②**评测脚本编排**（`eval_retrieval` / `calibrate_threshold` / `eval_generation` 的串接逻辑，仅 `eval_metrics` 纯函数有测）。
- **指标可复现性受限**：完整源语料不入库，第三方可查看评测集与 `examples/` 样例产物，但无法在完整 50 题上复现本仓库指标；向量库为幂等重建，历史版本对应的向量不保留。
- **生成侧指标与提示词版本绑定**：提示词结构或引用格式变更后，「系统」即发生变化，历史生成侧指标不可直接比较。

> 结论：这是一个**结构感知的工程实践与个人学习项目**，价值在于「怎么把结构感知切片、混合检索、重排、阈值拒答、生成侧评测一步步做出来，并用消融决定组件去留」，而不在于「小语料上刷到高分」。

---

## 目录结构

```
RAG/
├─ preprocessing/        # ① 数据准备：extractors（PDF）/ cleaner / pipeline
├─ chunking/             # ② 切片：chunker.py
├─ embedding/            # ③ 向量化：dashscope_embedder.py
├─ retrieval/            # ④ 检索：search.py / hybrid.py（BM25+向量 RRF）/ reranker.py（重排）
├─ scripts/              # run_chunking / run_vectorize / query_retrieval
│                        #   eval_retrieval / ablate_search / calibrate_threshold
│                        #   run_rag / gen_answers / eval_generation / eval_metrics / _common
├─ tests/                # pytest：test_chunker / test_cleaner / test_retrieval / test_extractors
│                        #   test_pipeline_smoke（离线端到端） / test_metrics_consistency
├─ docs/                 # 设计文档（见「文档」章节）
├─ examples/             # 样例：pdf/（3 份源 PDF）+ 提取/切片产物（受版本控制）
├─ data/                 # processed/chunks/chroma/eval（运行后生成，不入库）
├─ metrics.json          # 权威数字单一来源
├─ requirements.txt      # 依赖 + 版本锁定说明
└─ README.md
```

---

## 样例

`examples/` 下附带 3 份公告的完整数据：
- `pdf/`：**3 份真实公告源 PDF**（可直接运行样例，非完整语料）；
- 每份公告的 `_extracted.md`（人工核对）/ `_blocks.jsonl`（语义块）/ `_chunks.jsonl`（切片），按**标题修复后**的管线刷新，便于直观查看数据形态与「常规服 / 赛季服」等标题的完整合并。

完整源语料 `test/` 与 `data/` 产物不入库。

---

## 说明

- **完整源语料不入库**：全部 54 份公告 `test/`（PDF）与 `data/` 产物均在 `.gitignore` 中；**`examples/pdf/` 的 3 份抽样公告例外，已入库**，可直接复现它与配套产物。
- **克隆后不能直接问答**：`data/chroma`（向量库）与 `data/chunks`（混合路线的 BM25 语料）均不入库，**必须先跑「通用前缀」三步构建**。`run_rag` 会先做前置检查并给出指引。
- **评测集随仓库提供**：`data/eval/` 内含 50 题新评测集（`simple_qa_30.json` / `complex_qa_20.json`，源文件在 `test/qa_dataset/`），第三方可查看题目与 gold answer；`metrics.json` 汇总全部权威数字。
- **文档分工**：`docs/` 为设计文档（随仓库提供）；过程记录与个人笔记放在 `docs/private/`，已在 `.gitignore` 中，不随仓库分发。
- **阶段划分**：`run_rag` 面向运行期（本地验证 / 演示）；`pipeline` / `run_chunking` / `run_vectorize` / `eval_*` / `calibrate_threshold` / `ablate_search` / `gen_answers` / `tests` / `metrics.json` 面向开发维护期。
- **复现**：按「运行指南」先跑数据准备三步，再选一条检索路线评估 / 生成。
