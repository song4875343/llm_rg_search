# Agentic 本地知识库搜索工具

基于 ripgrep、BM25 和大语言模型的工程规范智能检索系统。支持多轮 Agentic 深度检索、两轮式 BM25 快速检索、以及"快检 + 评估 + 深挖"的混合检索三种模式，并提供 Web 服务与前端界面。

## ✨ 核心特性

- 🚀 **Agentic 多轮检索**（`rg_search_v6a/v6b`）：全局目录注入上下文，大模型自主调度 grep / 读目录 / 读原文工具，多轮迭代直到证据充分
- 🔎 **补充扫描防遗漏**（v6b）：限定文件搜索命中后自动扫描其余文件，生成 `S001` 补充索引供模型批量精读
- ⚡ **BM25 快速检索**（`rg-fast-v2a/v2b/v2c`）：系统先做全局 BM25 Top-N 短预览召回，LLM 只需选文件/选编号，两次 LLM 调用出答案
- 🔀 **混合检索**（`hybrid_search(_v2)`）：先走快速检索，LLM 评估证据是否足够——够则直接作答，不够自动转深度 Agent
- 🗂️ **TOC 目录索引**（`extract_toc`）：自动生成全局目录和单文件详细章节目录，搜索结果自动注入章节出处
- 📚 **回答依据抽取**：最终答案后额外调用模型提取"文件名 + 行号 + 条文号"结构化依据
- 💬 **Web 服务**：FastAPI + WebSocket 流式输出（思考过程、工具调用、回答分事件推送），内置历史记录与相似问题匹配

---

## 📦 快速开始

### 1. 环境要求

- Python 3.11+（仓库提供 `.python-version` 与 `uv.lock`）
- ripgrep（Windows 用户使用项目自带的 `rg.exe`）

### 2. 安装依赖

```bash
pip install openai python-dotenv fastapi "uvicorn[standard]"
```

或使用 uv：

```bash
uv sync
```

### 3. 配置 API Key

复制 `.env.example` 为 `.env`，按 `model_config.py` 中所用模型的 `api_key` 字段填入对应环境变量：

```env
kimi_key=your-api-key
deepseek_key=your-api-key
DASHSCOPE_API_KEY=your-api-key
...
```

### 4. 准备规范文件

将 `.txt` / `.md` 文件放入 `specs/` 文件夹（默认资料库）。首次运行会通过 `extract_toc` 自动在 `specs/.index/` 生成目录索引。

### 5. 运行

```bash
# Web 服务 v2 栈（推荐）：agentic=v6b / fast=v2c / hybrid=混合检索 v2
python server_v2.py

# Web 服务 v1 栈：agentic=v6a / fast=v2a / hybrid=混合检索 v1
python server.py

# 命令行直接运行某个版本
python rg_search_v6b.py
python rg-fast-v2c.py
python hybrid_search_v2.py
```

浏览器访问：

| 地址 | 前端页面 |
|---|---|
| http://localhost:5000 | `index3.html`（默认版） |
| http://localhost:5000/v1 | `index.html` |
| http://localhost:5000/v2 | `index2.html` |

---

## 🏗️ 架构总览

项目当前由**两条并行技术栈**组成，每条栈包含一个 Agentic 检索器、一个 BM25 快速检索器和对应的 Web 服务：

| 组件 | v1 栈 | v2 栈（当前主力） |
|---|---|---|
| Web 服务 | `server.py` | `server_v2.py` |
| Agentic 深度检索 | `rg_search_v6a.py` | `rg_search_v6b.py`（增加补充扫描） |
| BM25 快速检索 | `rg-fast-v2a.py`（RG+chunk 双路） | `rg-fast-v2c.py`（全局预览+文件内 BM25，另存实验版 `rg-fast-v2b.py`） |
| 混合检索 | `hybrid_search.py` | `hybrid_search_v2.py` |

历史版本（V1–V5 等）已移入 `old_vsion/`。

---

## 🎯 检索模式详解

### Agentic 模式 —— rg_search_v6a / v6b

多轮 Agent 主循环（最多 15 轮）：system prompt 注入资料库全局目录，模型自主调用工具收集证据，不再请求工具时生成最终答案并抽取依据。

**工具集：**

| 工具 | 说明 |
|---|---|
| `execute_grep` | 调用 `rg -n -i -H -C{N} -m50` 搜索，结果带章节注解并按 `(文件, 行号)` 去重，只展开前 20 条新记录 |
| `get_document_toc` | 读取单文件详细章节目录（`.index/{stem}.index.json`） |
| `read_file_range` | 按行范围读取原文 |
| `fetch_supplemental`（仅 v6b） | 按 `S001,S003` 编号批量精读补充扫描命中的原文上下文 |

**v6b 的补充扫描机制**：当模型用 `include_files` 限定了高概率文件且其中有命中时，系统自动用轻量命令（无 `-C`、每文件 `-m12`、上限 30 条）扫描其余文件，返回关键词居中短预览的补充索引。模型可继续调用 `fetch_supplemental` 批量精读，避免"找到一处就早退"漏掉其他规范的相关条文。

详细运行逻辑见 [`优化逻辑.md`](优化逻辑.md) 及流程图 [`v6b_flow.svg`](v6b_flow.svg)。

### Fast 模式 —— rg-fast 系列

**共同设计**：BM25 预筛选 + 固定两次 LLM 调用。系统先把全库按句子边界切成 ~512 字 chunks 并做全局 BM25 召回 Top-N，只向 LLM 暴露"编号 + 文件 + 行号 + 前 50 字预览"；第 1 次 LLM 只负责选择挖掘方向并调用本地工具获取完整片段；第 2 次 LLM 基于完整证据生成终稿。

| 版本 | 初筛 | 第 1 次 LLM 的工具 | 特点 |
|---|---|---|---|
| `rg-fast-v2a.py` | 无（直接让 LLM 出关键词） | `search_documents`：RG 关键词搜索 + 目标文件切片 BM25 双路召回，关键词加分（精确命中 +1.0；宽泛命中 2 个 +0.5、≥3 个 +1.0）后 BM25/修正双榜融合排序去重 | 单工具单轮，经典双路方案 |
| `rg-fast-v2b.py` | 全局 BM25 Top-N 预览 | `search_high_probability_files`（文件内 BM25）+ `read_preview_items`（按 P001 编号精读预览条目） | 双工具互补，证据合并去重 |
| `rg-fast-v2c.py` | 全局 BM25 Top-N 预览 | 仅 `search_high_probability_files`（文件内 BM25，Top-K 完整片段） | 单工具最短路径，速度最快 |

实践结论（详见 [`优化逻辑.md`](优化逻辑.md) §19–22）：v2b 与 v2c 效果接近，"按编号精读"的工具边际收益有限；两者共同的局限是 BM25 预筛可能因术语不一致而漏召，无法像 Agentic 模式那样自主换词扩展。流程图见 [`v2a_bm25_two_step_flow.svg`](v2a_bm25_two_step_flow.svg)、[`v2c_bm25_file_only_flow.svg`](v2c_bm25_file_only_flow.svg)。

### Hybrid 模式 —— hybrid_search / hybrid_search_v2

三阶段流水线，兼顾速度与召回率：

1. **阶段 1 · 快速取证**：v1 执行 fast 第 1 轮工具搜索；v2 执行 v2c 全局预览 + 文件内 BM25 取证
2. **阶段 2 · 完整性评估**：将证据截断至 2000 字，让 LLM 只回答 YES/NO 判断是否足以作答（评估失败默认转深挖）
3. **阶段 3a · 直接作答**：评估通过则基于证据流式生成终稿（等价于 fast 第 2 轮）
3. **阶段 3b · 深度兜底**：不通过则转 Agentic 模式（v1→v6a，v2→v6b）多轮深挖

---

## 🌐 Web 服务说明（server.py / server_v2.py）

两个服务的 HTTP/WebSocket 接口一致，仅绑定的检索栈不同。

### WebSocket `/ws/query`

请求：

```json
{
  "question": "筏板的最小厚度",
  "folder_path": "specs",
  "model_num": 8,
  "context_lines": 10,
  "thinking_enabled": false,
  "mode": "agentic",          // agentic | fast | hybrid
  "extract_references": true
}
```

推送事件类型：`turn`（轮次）、`tool_call` / `tool_result`（工具调用与摘要）、`thinking_chunk` / `thinking_complete`（思考流）、`stream_chunk`（回答流）、`final_answer`、`references`（结构化依据）、`error`。

### HTTP API

| 方法 | 路径 | 说明 |
|---|---|---|
| POST | `/api/set-folder` | 设置工作资料库文件夹（切换后清理缓存） |
| POST | `/api/set-model` | 切换模型序号与思考开关（运行时生效） |
| GET | `/api/models` | 列出 `model_config.py` 全部模型及思考能力 |
| POST / GET | `/api/set-context-lines` `/api/context-lines` | 设置/读取搜索上下文行数（0–50） |
| GET | `/api/folders` | 浏览可选文件夹 |
| GET | `/api/index-status` | 查询某文件夹的索引状态 |
| POST | `/api/index-folder` | 触发 `extract_toc` 生成目录索引 |
| POST | `/api/read-file-range` | 按文件名（支持模糊）读取行范围原文 |
| GET / POST / DELETE | `/api/history` | 历史记录增删查（保留最近 100 条，存于 `history.json`） |
| DELETE | `/api/history/{item_id}` | 删除单条历史 |
| POST | `/api/history/match` | 相似问题匹配（difflib + 中文 2-gram Jaccard，阈值默认 0.78） |

---

## ⚙️ 核心技术要点

- **模型配置**（`model_config.py`）：以序号选择 OpenAI-compatible 端点；`build_chat_kwargs()` 按厂商适配思考模式开关（kimi：`extra_body={"thinking":{"type":"disabled"}}` 且关闭思考时 temperature 固定 0.6；qwen：`enable_thinking`；deepseek：`thinking.enabled/disabled`），并兼容部分模型流式末尾 `choices=[]` 的边界情况
- **BM25 模块**：`bm25_module` 为编译好的 `.pyd` 扩展，fast 系列依赖它完成召回排序
- **TOC 索引**：`extract_toc.scanner.scan_folder` 生成 `index.json`（全局）与 `{stem}.index.json`（单文件章节），用于目录注入与 `get_chapter_context` 章节出处推断
- **rg 解析容错**：从右往左定位纯数字行号段解析 `file:line:content`，兼容 Windows 盘符路径含冒号的情况
- **缓存策略**：目录详情、搜索去重键、全局 chunk、预览条目均有模块级缓存；Web 端切换文件夹时统一清理

## 🧭 版本选择建议

| 场景 | 推荐 |
|---|---|
| 追求答案全面性、允许较多轮次 | **Agentic v6b**（补充扫描防遗漏，容错最强） |
| 追求响应速度、问题术语明确 | **Fast v2c**（两次调用固定流程，最快） |
| 生产环境兼顾速度与质量 | **Hybrid v2**（够则秒答，不够自动转深挖） |

## 📝 注意事项

1. 各脚本内通过 `num = ...` / `MODEL_NUM = ...` 选择默认模型，Web 端可在运行时通过 `/api/set-model` 或请求参数覆盖
2. Agentic 与 Fast 模式均要求模型支持 Function Calling
3. Fast 系列需要 `bm25_module` 对应 Python 版本的 `.pyd`
4. 规范文件建议 UTF-8 编码，Windows 用户确保 `rg.exe` 在项目根目录或已加入 PATH
5. `server*.py` 默认监听 `0.0.0.0:5000`

---

## 📄 许可证

本项目采用 [Apache License 2.0](LICENSE) 开源协议。
