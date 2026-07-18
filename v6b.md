# rg_search_v6b.py Agent 运行逻辑

本文档记录 `rg_search_v6b.py` 当前版本的完整运行逻辑：从启动、索引加载、模型调用、工具循环、补充扫描，到最终答案和回答依据抽取。

## 1. 总体定位

`rg_search_v6b.py` 是一个面向本地规范资料库的检索问答 Agent。它使用 OpenAI-compatible Chat Completions 接口驱动大模型，让模型在多轮对话中自主调用本地工具：

- `get_document_toc`：读取指定文档的详细目录。
- `execute_grep`：调用 `rg` 搜索关键词，并返回命中行上下文。
- `fetch_supplemental`：按补充扫描索引编号读取原文上下文。
- `read_file_range`：读取指定文件的指定行范围。

当前默认资料库是 `specs/`，默认索引目录是 `specs/.index/`。Agent 会先把资料库全局目录放进 system prompt，再让模型根据问题逐轮选择检索工具。工具返回原文证据后，模型继续判断是否还要换关键词、查目录、读上下文或读取补充索引；直到它不再请求工具，程序再生成最终答案，并额外调用一次模型抽取回答依据。

## 2. 运行流程图

<p align="center">
  <img src="v6b_flow.svg" alt="rg_search_v6b.py Agent 运行流程图" width="100%">
</p>

流程图源文件：[`v6b_flow.svg`](v6b_flow.svg)

## 3. 启动与初始化

程序启动时会完成这些工作：

1. 导入标准库、`OpenAI`、`dotenv` 和 `extract_toc.scanner.scan_folder`。
2. 调用 `load_dotenv()` 读取 `.env`。
3. 将脚本目录加入 `sys.path`，保证本地模块可导入。
4. 通过 `num = 1` 从 `MODEL_DICT` 中选择模型，当前默认是 `kimi-k2.5`。
5. 设置 `TARGET`、`INDEX_DIR`、`MAIN_INDEX` 和 `RG_EXE`。
6. 扫描 `TARGET` 下的 `.txt` 和 `.md` 文件，生成 `FILE_MAP`。
7. 初始化缓存：`DETAIL_TOC_CACHE`、`SEARCH_RESULT_CACHE`、`SUPPLEMENT_CACHE`、`SUPPLEMENT_KEY_MAP`、`CLIENT`。

`MODEL_DICT` 中每个模型配置通常包含 `base_url`、`.env` 中的 `api_key` 环境变量名、`model_name`，以及可选的 `thinking` 类型。`get_client()` 会懒加载 OpenAI 客户端，并复用全局 `CLIENT`。

## 4. 模型调用封装

`build_chat_kwargs()` 负责组装 Chat Completions 请求参数，包括 `model`、`messages`、`temperature`、`stream`、`tools` 和 `tool_choice="auto"`。如果模型配置里声明了 `thinking` 类型，它还会适配不同厂商的思考模式参数：

| 类型 | 控制方式 |
|---|---|
| `kimi` | `extra_body={"thinking": {"type": "disabled"}}` |
| `qwen` | `extra_body={"enable_thinking": thinking_enabled}` |
| `deepseek` | `extra_body={"thinking": {"type": "enabled" / "disabled"}}` |

`_chat_stream()` 统一消费流式输出，聚合普通回答文本、reasoning 内容和工具调用。它还兼容部分模型最后一个 chunk 里 `choices=[]` 的情况，避免越界错误。

## 5. 索引与目录

`ensure_index_exists()` 检查 `specs/.index/index.json` 是否存在；如果不存在，就调用 `scan_folder(str(TARGET), recursive=True, output_dir=str(INDEX_DIR))` 重新生成索引。

`get_global_toc_summary()` 读取全局目录，并把 JSON 字符串注入 `run_agent()` 的 system prompt，让模型一开始就知道资料库结构。

`get_document_toc(filename)` 是暴露给模型的工具。它按文件名片段匹配 `FILE_MAP`，再读取 `specs/.index/{stem}.index.json`，用于查看某个文档的详细章节目录。

`get_chapter_context(filepath, line_num)` 根据文件和行号，从详细目录中推断命中行所在章节、小节，并生成类似 `[出自：第X章 -> 第Y节 | 本章小节: ...]` 的上下文。搜索结果和补充索引都会尽量注入这个章节信息。

## 6. run_agent 主循环

入口函数是：

```python
run_agent(user_question, show_reasoning=False, stream=False, extract_refs=True)
```

参数含义：

| 参数 | 默认值 | 作用 |
|---|---:|---|
| `user_question` | 必填 | 用户问题 |
| `show_reasoning` | `False` | 是否显示模型 reasoning 字段 |
| `stream` | `False` | 是否以 generator 方式返回运行过程 |
| `extract_refs` | `True` | 最终答案后是否抽取回答依据 |

每次运行先调用 `reset_search_cache()`，清空 `SEARCH_RESULT_CACHE`、`SUPPLEMENT_CACHE` 和 `SUPPLEMENT_KEY_MAP`。然后构造两条初始消息：system prompt 和用户问题。

system prompt 的关键纪律是：必须调用工具查阅资料；必须明确引用依据；信息不足时继续换关键词、查目录或读原文；必须全面收集相关规范，不能找到一处就停止；最终回答前最少进行一次批量精读补充索引内容。

主循环最多执行 15 轮。每轮先调用 `_chat_stream(messages, TOOLS_SCHEMA, ...)`，如果模型返回 `tool_calls`，程序就解析工具名和参数，调用本地函数，并把工具结果作为 `role="tool"` 的消息追加回 `messages`。如果模型没有返回工具调用，说明它准备回答，程序进入最终答案阶段。

如果 15 轮后仍未结束，程序会追加 `已达到最大搜索次数，请立即总结回答`，然后强制生成最终答案。

## 7. execute_grep 主搜索

入口函数：

```python
execute_grep(pattern, include_files=None, stream=False)
```

基础命令：

```bash
rg -n -i -H -C {CONTENT_LINES} -e {pattern} -m 50
```

| 参数 | 作用 |
|---|---|
| `-n` | 输出行号 |
| `-i` | 忽略大小写 |
| `-H` | 输出文件名 |
| `-C` | 返回命中行上下文 |
| `-e` | 指定搜索表达式 |
| `-m 50` | 每个文件最多 50 个匹配 |

当模型传入 `include_files` 时，代码会按文件名片段从 `FILE_MAP` 中挑选目标文件，只在这些文件中搜索。否则直接搜索整个 `TARGET` 目录。

主搜索会按 `(filepath, line_number)` 写入 `SEARCH_RESULT_CACHE` 去重，避免多轮搜索反复返回同一条命中。结果会经过 `annotate_grep_output()` 转换成更适合模型阅读的格式：

```text
[下面内容出自：文件名-->章节上下文]
行号123-->原文内容
  行号124-->上下文内容
--
```

主搜索可能命中很多条，但实际只把前 20 条新记录展开给模型，以控制 token。

## 8. 补充扫描机制

补充扫描是这个版本的重要增强点。它解决的问题是：模型有时会先限定一两本高概率规范搜索，如果这些文件已经命中，它可能过早停止，漏掉其他规范中的相关条文。

触发条件：`execute_grep()` 使用了 `include_files`，并且限定文件内有命中。

触发后，`_supplemental_grep_lines(pattern, excluded_names, limit=30, preview_chars=50)` 会排除主搜索已经限定的文件，只扫描剩余文件。它使用更轻量的命令：

```bash
rg -n -i -H -e {pattern} -m 12 {remaining_files}
```

补充扫描不带 `-C`，不返回大段上下文；每个剩余文件最多 12 条命中，全局最多返回 30 条短索引。每条补充命中会生成 `S001`、`S002` 这样的编号，并写入 `SUPPLEMENT_CACHE` 和 `SUPPLEMENT_KEY_MAP`。

返回给模型的补充索引类似：

```text
补充ID=S001 | 文件=xxx.txt | 原文行=2539 [出自：...]
索引预览-->...命中关键词周边内容...
```

如果模型判断某条补充索引可能相关，应该继续调用 `fetch_supplemental(ids="S001,S003")` 批量精读。

## 9. fetch_supplemental 批量精读

入口函数：

```python
fetch_supplemental(ids, context_lines=CONTENT_LINES, stream=False)
```

它根据补充编号读取原文上下文，并做几类保护：

- 标准化编号，例如 `S1` 转成 `S001`。
- 对重复编号去重。
- 如果模型把原文行号误当成补充编号，会尝试按行号纠正。
- 如果同一行号对应多个补充索引，会提示歧义。

读取成功后返回：

```text
--- [S001] 文件名 行100-120 命中行110 [出自：章节] ---
原文内容
--- [S001] 片段结束 ---
```

已经精读过的补充命中会写入 `SEARCH_RESULT_CACHE`，减少后续重复返回。

## 10. read_file_range 原文读取

入口函数：

```python
read_file_range(filepath, start_line, end_line, stream=False)
```

它用于在已知文件和行号后直接读取一段原文。路径解析规则是：如果 `filepath` 是真实路径就直接读取；否则用 basename 到 `FILE_MAP` 中查找；最后按 `start_line` 到 `end_line` 截取内容。

返回格式：

```text
--- 文件名 ---
原文内容
--- 片段结束 ---
```

## 11. 最终答案与依据抽取

当模型停止调用工具时，`run_agent()` 进入 `output_final()`。程序会基于完整对话历史再调用一次 `_chat_stream(messages, thinking_enabled_override=False)`，整理生成最终答案。

如果 `extract_refs=True`，随后调用：

```python
extract_references(messages, final_answer)
```

它使用单独的工具 schema `output_references`，要求模型从对话历史和最终答案中提取：

| 字段 | 含义 |
|---|---|
| `filename` | 文件名 |
| `line_number` | 起始行号 |
| `end_line` | 结束行号 |
| `article_number` | 条文号或条目号 |

最终会追加类似内容：

```text
============================================================
📚 [回答依据]:
  [1] xxx.txt 行123 第x.x.x条
============================================================
```

## 12. 缓存设计

| 缓存 | 作用 | 清空时机 |
|---|---|---|
| `DETAIL_TOC_CACHE` | 缓存每个文件的详细目录 | 不在每次问题中清空 |
| `SEARCH_RESULT_CACHE` | 记录主搜索和已精读补充索引的 `(文件, 行号)` | 每次 `run_agent()` 开始清空 |
| `SUPPLEMENT_CACHE` | 保存 `S001` 等补充索引对应的文件、行号、内容 | 每次 `run_agent()` 开始清空 |
| `SUPPLEMENT_KEY_MAP` | 防止同一补充命中重复编号 | 每次 `run_agent()` 开始清空 |
| `CLIENT` | 复用 OpenAI 客户端 | 进程级复用 |

## 13. 关键可调参数

| 参数 | 默认值 | 影响 |
|---|---:|---|
| `num` | `1` | 选择模型 |
| `CONTENT_LINES` | `10` | 主搜索和补充精读的上下文行数 |
| `THINKING_ENABLED` | `False` | 是否启用模型思考模式 |
| `TARGET` | `specs` | 检索资料库目录 |
| `execute_grep -m` | `50` | 主搜索每个文件最多命中数 |
| `new_records[:20]` | `20` | 每轮最多展开给模型的主搜索记录数 |
| `_supplemental_grep_lines limit` | `30` | 补充扫描最多返回索引数 |
| `_supplemental_grep_lines preview_chars` | `50` | 补充索引预览字符数 |
| `fetch_supplemental context_lines` | `CONTENT_LINES` | 读取补充索引时的上下文行数 |
| `range(15)` | `15` | Agent 最大工具循环轮数 |

## 14. 一次典型运行示例

以问题 `门刚何时应采用揽风绳` 为例，可能过程是：

1. 程序启动，加载模型配置和 `specs` 文件列表。
2. `run_agent()` 清空搜索缓存，读取全局目录，构造 system prompt。
3. 第 1 轮模型调用 `execute_grep(pattern="揽风绳")` 或搜索同义词 `缆风绳`。
4. 如果命中不足，模型继续换关键词，例如 `门式刚架`、`施工阶段稳定`。
5. 如果模型限定某些文件搜索，`execute_grep()` 会额外扫描其余文件，并返回 `S001` 等补充索引。
6. 模型根据补充索引调用 `fetch_supplemental(ids="S001,S002")` 批量精读。
7. 模型对关键命中行调用 `read_file_range()` 读取更完整的上下文。
8. 当证据足够后，模型停止调用工具。
9. `output_final()` 生成最终答案。
10. `extract_references()` 从对话历史中抽取文件名、行号和条文号，追加回答依据。

## 15. 核心设计原则

`v6b` 的核心不是替模型写死检索路径，而是在工具层给模型更好的信息反馈：

- 全局目录让模型先知道资料库结构。
- `rg` 主搜索提供上下文和章节定位。
- 搜索缓存减少重复命中。
- 限定文件搜索时自动补充扫描其余文件，降低漏检概率。
- 补充扫描只返回短索引，避免 token 爆炸。
- `fetch_supplemental` 让模型按需批量精读真正相关的补充命中。
- 最终再单独抽取回答依据，使答案和证据更容易核查。

整体运行逻辑可以概括为：

```text
全局目录定向
  -> 多轮工具检索
  -> 主搜索给证据
  -> 补充扫描防遗漏
  -> 原文精读确认
  -> 最终回答
  -> 依据抽取
```

