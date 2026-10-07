# 游戏公告 RAG 检索增强问答系统

一个从零实现、**结构感知**的 RAG（检索增强生成）管线：面向《晶核》游戏官方更新公告 PDF，把公告解析成带语义结构的中间表示，做**结构感知切片 → 向量化 → 混合检索 + 重排 → 带引用生成**；回答忠实于原文、附**页码溯源**，检索置信度不足时**拒答**。

> **项目定位（重要）**：本仓库是**源码 / 开发维护仓库**，不是可直接分发的应用。
> 语料（`test/`）与全部产物（`data/processed|chunks|chroma|bm25`）**均不入库**，因此**克隆后不能直接问答** ——
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

- **输入**：游戏官方更新公告 PDF（源语料 50 份）。
- **输出**：自然语言问答，回答带 `[chunk_xxxxxx]` 引用与来源（文档名 + 章节路径 + 页码）。
- **核心中间表示**：`Block(kind=heading/para/table, page)` —— 用语义标签承载结构，而非裸文本。
- **两条检索路线**：纯向量（基线）/ 混合检索 + 重排（默认）。
- **使用前提**：`.env`（API Key）+ `data/chroma`（向量库）+ `data/bm25`（BM25 语料侧产物）+ `data/chunks`（判分取片段正文、以及 BM25 产物缺失时的回退构建）——后三者需本地构建。

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

**源语料**：完整语料在 `test/`（50 份公告 PDF，本地运行用，未入库）；`examples/pdf/` 另附 **3 份抽样公告 PDF** 作可直接运行的样例（见「样例」）。仓库不携带全部 50 份源 PDF。

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
| 生成 | DashScope chat `qwen-flash` |
| 判分（生成侧评测） | DashScope chat `qwen3.7-flash`（与生成模型分开配置，减少同模型自评偏差） |
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

  │ ② 清洗：空白规范化、页码/元数据去噪、**尾部署名之后的推荐类噪声整段截断**、丢空块
  │ ③ 结构感知切片：标题栈 → section_path / 表格整体 / 按句切 / 句边界 overlap / 过短回并
  ▼    → data/chunks/*.chunks.jsonl（每 chunk 带溯源坐标）

  │ ④ 语料侧产物（两条检索路线各一份，均只读）：向量化 → data/chroma；BM25 分词+索引 → data/bm25
  │ ⑤ 检索：纯向量 或 混合+重排 → 检索 pass 落快照 → 拼上下文 → 阈值拒答 → 生成带引用回答
【检索+生成】 scripts/retrieval_pass.py / eval_retrieval.py / run_rag.py / gen_answers.py / eval_generation.py
```

### 关键设计

- **语义中间表示 `Block(kind=heading/para/table, page)`**：按原格式直接解析，不转 Markdown（有损、拉平表格），`md` 仅作人工核对。
- **结构感知切片**（非固定大小）：标题为一节边界（维护标题栈得 `section_path` 全路径）、表格整体不拆、段落按目标大小聚合、超长**按句切（不切半句）**、切点附**句边界 overlap**、过短回并。
- **尾部噪声截断**：公告正文末尾的署名（`冒险者协会`）之后全是"往期推荐 / PV 预告 / 更多官方信息"，按**块以署名结尾**为界整段丢弃（不用位置比例阈值）。
- **溯源坐标**：每个 chunk 带 `id / doc / section_path / text / rows / start_page / end_page`，命中后凭 `doc + section_path(+page)` **回溯回原文 + 页码**。
- **两条检索路线**：纯向量（基线）/ 混合检索（BM25 + 向量 RRF）叠加重排；BM25 语料侧产物与向量库对称落盘（`data/bm25/`），缺失时自动回退现场分词构建。
- **阈值拒答**：top-1 分数低于阈值 → 固定回答「**当前资料相关性不足，无法回答**」（`run_rag.REJECT_ANSWER`），**不调生成 API**，防幻觉。两条路线 score 尺度不同，**各配校准阈值**（纯向量 0.55 / 重排 0.58，由 `scripts/calibrate_threshold.py` **离线读检索快照**标定）。
  与**模型级拒答**（资料存在但不足以回答，模型答「资料中没有相关信息」）是两种不同情形，措辞刻意区分：「检索没找到相关内容」vs「找到了但资料里没写」；SYSTEM_PROMPT 强制后者必须拒答、不得用常识补全。

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

> 精准检索（BM25）的**语料侧产物**由 `scripts.run_bm25_build` 落盘到 `data/bm25/`（与向量库对称的构建产物，
> 省掉每次起进程的语料分词 ≈0.4s）。它**不是必须的步骤**：未构建时混合路线会自动回退为现场分词构建，功能不受影响。
> 语料或切片变了就重跑一次（与 `data/chroma` 同等对待，不做指纹校验）。

### 检索 pass（落快照：**唯一调检索 API 的一步**）

```powershell
# 两条路线各跑一次 top-k 检索，落 data/eval/retrieval_{route}.jsonl；k 两集统一 5
.venv_rag311\Scripts\python.exe -m scripts.retrieval_pass
```

> 指标与阈值此后都**离线读这份快照**计算，不重复调 API；要重算指标就先重跑这一步，避免两套口径。

### 路线 A · 纯向量

```powershell
# 检索指标（离线读快照；默认两条路线都算）
.venv_rag311\Scripts\python.exe -m scripts.eval_retrieval --mode vector_only --detail

# 端到端生成（带引用 + 页码；--min-score 缺省取该路线校准值 0.55）
.venv_rag311\Scripts\python.exe -m scripts.run_rag "问题" --mode vector_only -k 5

# 批量生成回答
.venv_rag311\Scripts\python.exe scripts\gen_answers.py --mode vector_only --sets all --out data\eval\answers_vector_only.jsonl
```

### 路线 B · 混合检索 + 重排（默认路线）

```powershell
# 检索指标（BM25+向量 RRF 生成候选 → DashScope 重排 后的快照）
.venv_rag311\Scripts\python.exe -m scripts.eval_retrieval --mode hybrid_rerank --detail

# 端到端生成（默认路线；--min-score 缺省取校准值 0.58）
.venv_rag311\Scripts\python.exe -m scripts.run_rag "问题" -k 5

# 批量生成回答
.venv_rag311\Scripts\python.exe scripts\gen_answers.py --mode hybrid_rerank --sets all --out data\eval\answers_hybrid_rerank.jsonl
```

### 阈值校准 / 生成侧评测 / 消融

```powershell
# 阈值标定（离线读快照：正/负类 top-1 分布 + 阈值区间权衡表）
.venv_rag311\Scripts\python.exe -m scripts.calibrate_threshold

# 生成侧评测：忠实度 / 引用准确率 / 答案正确率 / 拒答正确性（LLM-as-judge，默认 qwen3.7-flash）
.venv_rag311\Scripts\python.exe scripts\eval_generation.py --mode hybrid_rerank --out data\eval\gen_scores_hybrid_rerank.jsonl

# 四路消融（本轮未复验；会显著增加 API 调用量）
.venv_rag311\Scripts\python.exe -m scripts.ablate_search
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
| 语料 / 块数 | `test/` 50 篇 → `data/processed` **1766 块**（heading 638 / para 1128） |
| 尾部截断 | 41 份触发署名截断 → 21 份真正丢块、共 **34 块**（全部为往期推荐 / PV 预告类） |
| 切片 | **607 chunk** → 向量库 **607 行** |
| 元数据残留 | 含「晶核CoA / 听全文」的块 = **0** |
| 顺序正确性 | 抽检「维护更新内容」位于正文**之前** |
| 标题合并 | 抽检长标题**完整合并**（未被拆行）；「常规服 / 赛季服」散落字形已复原 |
| BM25 产物 | `data/bm25/bm25_corpus.pkl` **0.89 MB**（68363 token、词表 3908）；加载 10–16 ms vs 现场构建 391–409 ms |
| 单元测试 | pytest **74 通过**（chunker / cleaner / 落盘字段 / 检索与生成链路 / 离线评测链路 / BM25 产物 / PDF 提取 / 端到端 smoke / metrics.json 一致性 / 生成调用重试 / 运行前提检查） |

---

## 指标

向量模型 `qwen3.7-text-embedding-flash`；评测集 `data/eval/`（源文件在 `test/qa_dataset/`）——`simple_qa.json`（30 题，27 可答 + **3 不可答**）+ `complex_qa.json`（30 题，27 可答 + **3 不可答**），共 **60 题 / 6 道不可答**；**k 两集统一 5**。

### 检索指标

**口径：文档级**（判定"召回片段所属文档是否在该题 `sources` 里"）；分母是 **54 道可答题**（不可答题 G 为空，跳过、不补 0）。数字由 `scripts/eval_retrieval.py` 离线读快照计算。

| 集 | 路线 | recall@k | MRR | P@1 | precision@k |
|---|---|---|---|---|---|
| simple(27可答, k=5) | 纯向量 | 0.926 | 0.779 | 0.704 | — |
| simple(27可答, k=5) | **混合+重排** | **1.000** | **0.975** | **0.963** | — |
| complex(27可答, k=5) | 纯向量 | 0.627 | — | — | 0.452 |
| complex(27可答, k=5) | **混合+重排** | **0.809** | — | — | **0.667** |

> complex 每题有 **1~4 个来源文档**，故 recall = 来源覆盖率（宏平均，同文档多片段只算一次）、precision = 命中片段数 / k（**按片段计数**）；多来源下不报 P@1/MRR（其余路线同理）。
> 参照项：complex any-hit / all-hit = 0.815 / 0.407（纯向量）、0.963 / 0.593（混合+重排）。

### 四路消融（定位每层收益）

> ⚠️ **口径说明**：下表是 **54 份语料 / 50 题 / simple k=3** 那一轮的结果，**本次全流程重跑未复验**；口径同为文档级，但语料与题库不同，**数字不可与上表直接比较**，仅作"哪一层是杠杆"的方向性证据。

| 路线 | simple recall/MRR/P@1 | complex recall/precision |
|---|---|---|
| A 混合候选 + 重排 | **1.000 / 0.975 / 0.963** | **0.808 / 0.720** |
| B 混合候选（不重排） | 0.963 / 0.907 / 0.852 | 0.667 / 0.560 |
| C 纯向量候选 + 重排 | 0.963 / 0.938 / 0.926 | 0.721 / 0.640 |
| D 纯向量（基线） | 0.889 / 0.772 / 0.704 | 0.529 / 0.450 |

- **重排是主要杠杆**（C−D）：simple recall +0.074 / MRR +0.166；complex recall +0.192 / precision +0.190
- **混合候选（BM25）亦有稳定正增量**（A−C）：simple +0.037；complex recall +0.087 / precision +0.080

### 生成侧指标（LLM-as-judge `qwen3.7-flash`，60 题全有效）

| 路线 | 忠实度 | 引用准确率 | 答案正确率 | 拒答正确性 |
|---|---|---|---|---|
| 纯向量 | 0.875 | 0.658 | 0.692 | 0.817 |
| **混合+重排** | **0.917** | **0.800** | **0.817** | **0.933** |

> 分集（混合+重排）：simple `1.000 / 0.900 / 0.967 / 0.967`；complex `0.833 / 0.700 / 0.667 / 0.900`。
> **口径**：四指标在**全部 60 题**上平均（不可答题贡献"拒答正确性"），与检索侧只在 54 题上算**不对称**；
> 且与**提示词版本 + 判分模型**绑定——本轮提示词要求"资料不足必须输出『资料中没有相关信息』"，判分为 `qwen3.7-flash`，
> 故与历史数字（`qwen3.8-max` + 旧提示词 + 50 题）**不可直接比较**；忠实度/引用率的下降主要来自拒答类回答"没有引用"被计 0。

### 阈值（拒答值，本库重新离线标定）

| 路线 | 取值 | 可答下沿 | 不可答上沿 | 是否分离 | 效果 |
|---|---|---|---|---|---|
| 纯向量 | **0.55** | 0.576 | 0.6426 | ❌ 重叠 | 误杀 0 / 漏拦 4（6 道不可答拦下 2 道） |
| 混合+重排 | **0.58** | 0.6212 | 0.5168 | ✅ 分离 | 误杀 0 / 漏拦 0（6 道全拦） |

> 阈值语义：`top1 < 阈值` 即拒答。可行区间：纯向量 `(0.5130, 0.5760]`、混合+重排 `(0.5168, 0.6212]`（区间内任意取值效果相同）。
> 完整区间权衡表见 `data/eval/threshold_{route}.json` 与 `scripts/calibrate_threshold.py` 的打印。

---

## 结果分析

- **默认路线取「混合 + 重排」**：四项综合均值 0.890（纯向量 0.766），在检索两项、答案正确率、拒答正确性上全面领先；本轮连忠实度/引用也领先。
- **拒答阈值必须分路线配置**：两条路线 score 尺度不同（纯向量 = 余弦相似度；重排 = relevance 分）。重排能把可答 / 不可答**干净分开**（可答下沿 0.6212 > 不可答上沿 0.5168），纯向量则**重叠**（0.576 vs 0.6426），只能取折中值 0.55。
- **complex 集是主要难点**：多跳、多来源（每题 1~4 篇）。**检索覆盖度直接决定生成正确率**（混合+重排）：来源全覆盖 16 题正确率 0.781，部分覆盖 10 题仅 0.350；瓶颈在检索未提供完整证据，而非生成忠实度（complex 忠实度 0.833）。
- 详细分析与逐题明细见 `docs/验收复盘_混合重排.md`。

---

## 诚实边界（如实说明，不美化）

先把话说清楚，免得指标被高估：

- **这是「技术验证 / 自研实践」项目，不是有真实用户的产品**。处理对象是《晶核》游戏公告（50 份 PDF），属于高结构化、标题化的单一类型文档；**不是**一个可泛化到任意文档的通用 RAG 框架。对无结构 / 多栏混排 / 扫描件，需另调解析与切片策略。
- **评测集是自建自评**：60 题由作者设计、自行标注来源文档，无第三方标注、无盲评。因此指标反映的是「**系统能否回答这批既定问题**」，不直接等于「真实用户关心的问题被正确回答」。（`evidence` 字段**仅作参考、不参与打分**；打分基准是 gold answer + 检索上下文。）
- **样本量仍小**：60 题中**不可答题只有 6 道**，阈值校准的负类样本仅 6 个，统计上仍弱——两条路线的阈值都只能说「在这 6 个负样本上如此」，不足以支撑生产结论。（比上一轮的 3 个负样本好，但仍不是生产级结论。）
- **生成侧评测是 LLM-as-judge**：四指标由 `qwen3.7-flash` 打分，非人工盲评，评分本身带模型偏好；0 / 0.5 / 1 属**有序等级**，直接求算术平均隐含「按等距数值处理」的简化。
- **complex 生成正确率偏低（混合+重排 0.667 / 纯向量 0.450）**：多跳多来源问题需跨 1~4 篇整合，单轮 top-k 检索 + 生成覆盖不足；这是当前最明确的短板。
- **尾部噪声未 100% 清除**：3 份公告**完全没有署名**（`“冒险传承”系统优化` / `家园偷菜？温泉浴盐？…` / `还能抢装备？…`），按署名截断的规则对它们无效，共 5 个宣传块仍留在库里（占 607 块的 0.8%）。
- **测试覆盖仍有限**：现有 74 项单测，**离线端到端**（提取→清洗→切片）、**检索侧离线链路**（快照读写 / 阈值标定 / BM25 产物）与**交互式入口**（`run_rag.py` 主流程，依赖全部替换、不联网）已覆盖；仍缺的是 ①**在线链路**（向量化→检索→生成，需 DashScope API 与向量库，未纳入 pytest）②**LLM 判分与串接**（`eval_generation` 的 judge 提示词与解析、`gen_answers` 端到端串接）。
- **指标可复现性受限**：完整源语料不入库，第三方可查看评测集与 `examples/` 样例产物，但无法在完整 60 题上复现本仓库指标；向量库为幂等重建，历史版本对应的向量不保留。
- **生成侧指标与提示词/判分模型版本绑定**：提示词、引用格式或判分模型变更后，「系统」即发生变化，历史生成侧指标不可直接比较。

> 结论：这是一个**结构感知的工程实践与个人学习项目**，价值在于「怎么把结构感知切片、混合检索、重排、阈值拒答、生成侧评测一步步做出来，并用消融决定组件去留」，而不在于「小语料上刷到高分」。

---

## 目录结构

```
RAG/
├─ preprocessing/        # ① 数据准备：extractors（PDF）/ cleaner / pipeline
├─ chunking/             # ② 切片：chunker.py
├─ embedding/            # ③ 向量化：dashscope_embedder.py
├─ retrieval/            # ④ 检索：search.py / hybrid.py（BM25+向量 RRF，优先加载 data/bm25 产物）/ reranker.py（重排）
├─ scripts/              # run_chunking / run_vectorize / run_bm25_build / retrieval_pass
│                        #   eval_retrieval / calibrate_threshold / ablate_search
│                        #   run_rag / gen_answers / eval_generation / eval_metrics / _common
├─ tests/                # pytest：test_chunker / test_cleaner / test_records / test_retrieval
│                        #   test_offline_eval / test_bm25_artifact / test_extractors
│                        #   test_pipeline_smoke（离线端到端） / test_metrics_consistency
├─ docs/                 # 设计文档（见「文档」章节）
├─ examples/             # 样例：pdf/（3 份源 PDF）+ 提取/切片产物（受版本控制）
├─ data/                 # processed/chunks/chroma/bm25/eval（运行后生成，不入库）
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

- **完整源语料不入库**：全部 50 份公告 `test/`（PDF）与 `data/` 产物均在 `.gitignore` 中；**`examples/pdf/` 的 3 份抽样公告例外，已入库**，可直接复现它与配套产物。
- **克隆后不能直接问答**：`data/chroma`（向量库）、`data/bm25`（BM25 语料侧产物）与 `data/chunks`（切片产物）均不入库，**必须先跑「通用前缀」三步构建**（BM25 产物可选、缺失会自动回退）。`run_rag` 会先做前置检查并给出指引。
- **评测集随仓库提供**：`data/eval/` 内含 60 题评测集（`simple_qa.json` / `complex_qa.json`，源文件在 `test/qa_dataset/`），第三方可查看题目与 gold answer；`metrics.json` 汇总全部权威数字。
- **文档分工**：`docs/` 为设计文档（随仓库提供）；过程记录与个人笔记放在 `docs/private/`，已在 `.gitignore` 中，不随仓库分发。
- **阶段划分**：`run_rag` 面向运行期（本地验证 / 演示）；`pipeline` / `run_chunking` / `run_vectorize` / `run_bm25_build` / `retrieval_pass` / `eval_*` / `calibrate_threshold` / `ablate_search` / `gen_answers` / `tests` / `metrics.json` 面向开发维护期。
- **复现**：按「运行指南」先跑数据准备三步，再跑一次**检索 pass** 落快照，之后指标与阈值都离线复算；生成与判分按路线各跑一次。
