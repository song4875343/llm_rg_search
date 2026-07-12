# rg_search_v6b 运行逻辑说明

本文档描述 `rg_search_v6b.py` 当前版本的完整运行逻辑、主要参数、工具行为和可调优位置。

## 1. 整体定位

`rg_search_v6b.py` 是一个面向本地规范文本库的 LLM 检索问答脚本。

当前版本保留的是原有高性能主流程：

```text
用户问题
  -> LLM 读取全局目录并自主决定工具调用
  -> execute_grep / get_document_toc / read_file_range
  -> 多轮循环补充检索和读原文
  -> LLM 生成最终答案
  -> extract_references 提取回答依据
```

在原有流程基础上，`v6b` 主要新增了一个轻量增强：

```text
当 LLM 限定某些高概率文件搜索时，
execute_grep 会额外扫描其余文件，
只返回短索引线索，不返回长正文。
```

这样可以在不破坏原有多轮 agent 能力的前提下，提高检索广度，减少“只搜了部分文件就停”的漏召回问题。

## 2. 启动入口

文件底部入口：

```python
if __name__ == "__main__":
    # run_agent("高烈度区能否用砌体女儿墙")
    run_agent("门刚何时应采用揽风绳")
```

直接运行：

```powershell
& e:/desktop/llm_rg_search/.venv/Scripts/python.exe e:/desktop/llm_rg_search/rg_search_v6b.py
```

如果要换问题，修改 `run_agent(...)` 里的字符串即可。

也可以在 Python 中调用：

```python
import rg_search_v6b as m
m.run_agent("高烈度区能否用砌体女儿墙")
```

## 3. 全局配置参数

### 3.1 模型选择

```python
num = 1
MODEL_NAME = MODEL_DICT[num]["model_name"]
```

`num` 控制使用哪个模型。当前默认：

```python
1: kimi-k2.5
```

`MODEL_DICT` 中每个模型配置包含：

| 字段 | 作用 |
|---|---|
| `base_url` | OpenAI-compatible API 地址 |
| `api_key` | 从 `.env` 中读取的环境变量名 |
| `model_name` | 模型名称 |
| `thinking` | 是否支持思考模式控制，例如 `kimi`、`qwen`、`deepseek` |

调用客户端时：

```python
OpenAI(
    base_url=MODEL_DICT[num]["base_url"],
    api_key=os.getenv(MODEL_DICT[num]["api_key"])
)
```

### 3.2 搜索上下文行数

```python
CONTENT_LINES = 10
```

用于 `execute_grep()` 主搜索：

```bash
rg -C 10
```

含义是命中行上下各返回 10 行上下文。

调大：

- 优点：LLM 更容易看到完整条文附近内容
- 缺点：返回内容更长，token 成本更高，噪声更多

调小：

- 优点：更省 token，检索更轻
- 缺点：可能需要更多次 `read_file_range()` 深挖

### 3.3 思考模式

```python
THINKING_ENABLED = False
```

`build_chat_kwargs()` 会根据模型类型控制思考模式：

| 模型类型 | 控制方式 |
|---|---|
| `kimi` | `extra_body = {"thinking": {"type": "disabled"}}` |
| `qwen` | `extra_body = {"enable_thinking": thinking_enabled}` |
| `deepseek` | `extra_body = {"thinking": {"type": "enabled" / "disabled"}}` |

Kimi 关闭思考时，代码会自动把 `temperature` 调整为 `0.6`。

### 3.4 工作资料目录

```python
TARGET, INDEX_DIR = SCRIPT_DIR / "specs", SCRIPT_DIR / "specs" / ".index"
```

当前默认检索目录是：

```text
specs/
```

文件类型限制：

```python
f.endswith((".txt", ".md"))
```

所以只有 `.txt` 和 `.md` 文件会进入 `FILE_MAP`。

如需切换目录，可调用：

```python
set_target_folder(folder_path)
```

它会更新：

- `TARGET`
- `INDEX_DIR`
- `MAIN_INDEX`
- `FILE_MAP`
- 并清空搜索缓存

### 3.5 rg 可执行文件

```python
RG_EXE = SCRIPT_DIR / "rg.exe" if exists else "rg"
```

优先使用项目目录下的 `rg.exe`，否则使用系统 PATH 中的 `rg`。

## 4. 索引与目录

### 4.1 全局目录

```python
get_global_toc_summary()
```

会读取：

```text
specs/.index/index.json
```

如果不存在，则调用：

```python
scan_folder(str(TARGET), recursive=True, output_dir=str(INDEX_DIR))
```

全局目录会放进 `run_agent()` 的 system prompt 中，让 LLM 先看到资料库整体结构。

### 4.2 单文件详细目录

工具：

```python
get_document_toc(filename)
```

用途：

- 根据文件名片段匹配 `FILE_MAP`
- 读取对应的 `.index/{stem}.index.json`
- 返回该文档的章节目录

LLM 可以在信息不足时调用这个工具，先看某本规范的章节结构，再决定搜索或阅读范围。

### 4.3 章节上下文注入

```python
get_chapter_context(filepath, line_num)
```

根据文件名和命中行号，从详细目录中推断该行所在章节，返回类似：

```text
[出自：第X章 -> 第Y节 | 本章小节: ...]
```

这个上下文会被用于：

- 主搜索注解
- 补充扫描索引

## 5. 主 Agent 循环

入口：

```python
run_agent(user_question, show_reasoning=False, stream=False, extract_refs=True)
```

参数：

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `user_question` | 必填 | 用户问题 |
| `show_reasoning` | `False` | 是否显示模型 reasoning 字段 |
| `stream` | `False` | 是否返回 generator 流 |
| `extract_refs` | `True` | 最终答案后是否提取引用依据 |

### 5.1 初始化

每次运行会清空搜索缓存：

```python
SEARCH_RESULT_CACHE = {}
```

然后构造对话：

```python
messages = [
    {"role": "system", "content": "...全局目录 + 工具说明 + 纪律..."},
    {"role": "user", "content": user_question},
]
```

system prompt 中的核心纪律：

1. 必须调用工具查阅资料
2. 必须明确引用依据
3. 信息不足时继续换关键词、查目录或读原文
4. 防遗漏，禁止找到一处就停止检索

### 5.2 最多 15 轮工具循环

```python
for turn in range(15):
```

每轮：

1. 调用 `_chat_stream(messages, TOOLS_SCHEMA, ...)`
2. 如果模型返回 tool calls，则执行工具
3. 工具结果追加到 `messages`
4. 如果模型不再调用工具，则进入最终回答阶段

工具调用执行逻辑：

```python
func = {
    "execute_grep": execute_grep,
    "read_file_range": read_file_range,
    "get_document_toc": get_document_toc
}[tool_name]
```

### 5.3 达到最大轮次

如果 15 轮后仍未结束，会追加：

```text
已达到最大搜索次数，请立即总结回答
```

然后强制进入最终答案生成。

## 6. LLM 调用与流式处理

### 6.1 build_chat_kwargs

```python
build_chat_kwargs(messages, stream=False, tools=None, temperature=1, thinking_enabled_override=None)
```

职责：

- 填入模型名
- 填入 messages
- 填入 temperature
- 如果有 tools，则设置：

```python
tools=tools
tool_choice="auto"
```

- 根据不同模型设置 thinking extra body

### 6.2 _chat_stream

```python
_chat_stream(messages, tools=None, show_reasoning=False, thinking_enabled_override=None, stream_content=True)
```

职责：

- 流式调用模型
- 聚合普通文本输出
- 聚合 reasoning 内容
- 聚合 tool calls
- 兼容部分模型最后一个 chunk 的 `choices=[]`

返回：

```python
(
    {
        "role": "assistant",
        "content": "...",
        "tool_calls": [...],
        "reasoning_content": "..."
    },
    reasoning_text
)
```

## 7. 工具 Schema

### 7.1 get_document_toc

```json
{
  "name": "get_document_toc",
  "description": "获取指定文档的详细章节目录",
  "parameters": {
    "filename": "string"
  }
}
```

LLM 用它查看某个文件的章节目录。

### 7.2 execute_grep

```json
{
  "name": "execute_grep",
  "description": "搜索关键词，返回匹配行及上下文",
  "parameters": {
    "pattern": "string",
    "include_files": "string"
  }
}
```

参数：

| 参数 | 必填 | 说明 |
|---|---|---|
| `pattern` | 是 | rg 搜索关键词或正则 |
| `include_files` | 否 | 限定在哪些文件中搜索，填文件名片段，逗号分隔 |

### 7.3 read_file_range

```json
{
  "name": "read_file_range",
  "parameters": {
    "filepath": "string",
    "start_line": "integer",
    "end_line": "integer"
  }
}
```

LLM 用它根据文件和行号读取原文。

## 8. execute_grep 主搜索逻辑

函数：

```python
execute_grep(pattern, include_files=None, stream=False)
```

### 8.1 rg 命令

基础命令：

```bash
rg -n -i -H -C {CONTENT_LINES} -e {pattern} -m 50
```

参数含义：

| rg 参数 | 说明 |
|---|---|
| `-n` | 输出行号 |
| `-i` | 忽略大小写 |
| `-H` | 输出文件名 |
| `-C CONTENT_LINES` | 返回命中上下文 |
| `-e pattern` | 搜索表达式 |
| `-m 50` | 每个文件最多 50 个匹配 |

如果有 `include_files`：

```python
targets = [
    (FILE_MAP[k], k)
    for req in include_files.split(",")
    for k in FILE_MAP
    if req.strip() in k
]
```

即：按文件名片段模糊匹配。

如果没有 `include_files`，就在整个 `TARGET` 目录中搜索。

### 8.2 主搜索去重

主搜索使用全局缓存：

```python
SEARCH_RESULT_CACHE
```

去重 key：

```python
(filepath, line_number)
```

逻辑：

```python
if key and key not in SEARCH_RESULT_CACHE:
    SEARCH_RESULT_CACHE[key] = True
    new_records.append(record)
elif key:
    dup_count += 1
```

作用：

- 多轮搜索中避免同一命中行反复返回
- 让 LLM 更容易看到新增信息

### 8.3 返回条数限制

主搜索内部可能命中很多条，但最终只注解前 20 条新记录：

```python
new_records[:20]
```

返回里仍会告诉 LLM 总共有多少条新记录：

```text
系统反馈：{len(new_records)} 条新记录
```

### 8.4 注解输出

主搜索结果会经过：

```python
annotate_grep_output(...)
```

转换成：

```text
[下面内容出自：文件名-->章节上下文]
行号xxx-->正文
```

如果章节解析失败，会在终端输出调试信息：

```text
解析块数
找到章节
未找到
失败示例
```

## 9. 补充扫描索引

函数：

```python
_supplemental_grep_lines(pattern, excluded_names, limit=30, preview_chars=30)
```

这是 `v6b` 当前相对原版的主要增强点。

### 9.1 触发条件

只有当 `execute_grep()` 使用了 `include_files` 时才触发。

也就是说：

```text
LLM 限定高概率文件搜索
  -> 主搜索搜索限定文件
  -> 补充扫描搜索剩余文件
```

如果 LLM 本来就是全库搜索，则不触发补充扫描。

### 9.2 搜索范围

补充扫描排除主搜索已经限定的文件：

```python
remaining = [
    (name, path)
    for name, path in FILE_MAP.items()
    if name not in excluded_names
]
```

### 9.3 rg 命令

补充扫描使用更轻量的 rg：

```bash
rg -n -i -H -e {pattern} -m 12 {remaining_files}
```

与主搜索不同：

- 不带 `-C`
- 不返回上下文
- 每个文件最多 12 个匹配
- 最终全局最多返回 `limit=30` 条索引

### 9.4 补充扫描去重

补充扫描有两层去重：

1. 本次补充扫描内部去重：

```python
seen = set()
key = (fp, ln)
```

2. 跳过主搜索已经返回过的行：

```python
if key in seen or key in SEARCH_RESULT_CACHE:
    continue
```

这样补充索引不会重复提示主搜索已经给过的命中。

### 9.5 短索引预览

补充扫描不是正文召回，而是索引线索。

每条只保留：

- 编号
- 文件名
- 行号
- 章节上下文
- 命中行前 30 字

代码：

```python
preview = content.strip().replace("\\t", " ")
preview = preview[:preview_chars] + ("..." if len(preview) > preview_chars else "")
```

输出示例：

```text
【补充扫描索引：其余文件也有命中。这里只保留每行前30字；如可能相关，请继续调用 execute_grep 限定文件或 read_file_range 拉取原文。】
[L003] 建筑与市政工程抗震通用规范.txt 行43
索引预览-->5.1.13  建筑主体结构中，幕墙、围护墙、隔墙、女儿墙、...
```

这段提示的设计意图是：

```text
告诉 LLM：其他文件可能也有资料；
但不要直接把大量正文塞进上下文；
如果 LLM 判断有价值，应继续调用工具深挖。
```

### 9.6 补充扫描接入点

如果限定范围没有命中：

```python
return "系统反馈：限定范围未找到匹配项" + supplement
```

如果限定范围有命中：

```python
return f"系统反馈：{len(new_records)} 条新记录:\n{annotated}{supplement}"
```

终端会额外显示：

```text
🔎 [补充扫描] 发现其余文件单行命中
```

## 10. read_file_range 读取原文

函数：

```python
read_file_range(filepath, start_line, end_line, stream=False)
```

逻辑：

1. 如果 `filepath` 是真实路径，直接使用
2. 否则按 basename 到 `FILE_MAP` 中查找
3. 读取指定行范围
4. 返回：

```text
--- 文件名 ---
原文
--- 片段结束 ---
```

注意：

- 行号从 1 开始
- 会自动 clamp 到文件实际长度
- 使用 `encoding="utf-8", errors="ignore"`

## 11. 最终答案与引用提取

当 LLM 不再调用工具时，进入：

```python
output_final(msg_obj)
```

### 11.1 生成最终答案

代码会再调用一次：

```python
_chat_stream(messages, thinking_enabled_override=False)
```

目的是基于完整对话历史生成最终回答。

### 11.2 引用提取

如果：

```python
extract_refs=True
```

则调用：

```python
extract_references(messages, final_answer)
```

它使用单独的 tool：

```python
output_references
```

提取字段：

| 字段 | 说明 |
|---|---|
| `filename` | 文件名 |
| `line_number` | 起始行 |
| `end_line` | 结束行 |
| `article_number` | 条文号，例如 `6.5.2` |

最终输出类似：

```text
============================================================
📚 [回答依据]:
  [1] 6砌体结构设计规范[附条文说明].txt 行111 6.5.2
============================================================
```

## 12. 缓存机制

### 12.1 DETAIL_TOC_CACHE

```python
DETAIL_TOC_CACHE = {}
```

缓存每个文件的详细目录，避免重复读取 `.index/{stem}.index.json`。

### 12.2 SEARCH_RESULT_CACHE

```python
SEARCH_RESULT_CACHE = {}
```

每次 `run_agent()` 开始时清空。

主搜索写入缓存，补充扫描读取缓存但不写入缓存。

这样的设计含义：

- 主搜索返回过的行，不再重复返回
- 补充扫描只是索引提示，不污染主搜索去重状态

## 13. 当前关键可调参数

| 参数 | 位置 | 默认值 | 影响 |
|---|---|---:|---|
| `num` | 顶部配置 | `1` | 选择模型 |
| `CONTENT_LINES` | 顶部配置 | `10` | 主搜索上下文行数 |
| `THINKING_ENABLED` | 顶部配置 | `False` | 是否启用模型思考 |
| `TARGET` | 顶部配置 | `specs` | 检索资料目录 |
| `execute_grep -m` | `execute_grep()` | `50` | 主搜索每文件最大匹配数 |
| `new_records[:20]` | `execute_grep()` | `20` | 每次返回给 LLM 的主搜索记录数 |
| `limit` | `_supplemental_grep_lines()` | `30` | 补充索引最多返回条数 |
| `preview_chars` | `_supplemental_grep_lines()` | `30` | 补充索引每条预览字符数 |
| 补充扫描 `-m` | `_supplemental_grep_lines()` | `12` | 补充扫描每文件最大匹配数 |
| `range(15)` | `run_agent()` | `15` | 最大工具循环轮数 |

## 14. 典型运行过程示例

以问题：

```text
高烈度区能否用砌体女儿墙
```

可能流程：

```text
第 1 轮
  LLM 调用 execute_grep("女儿墙", include_files="砌体结构设计规范")
  主搜索返回砌体规范中的女儿墙条文
  补充扫描提示抗震通用规范、建筑抗震设计规范也有命中

第 2 轮
  LLM 根据补充索引调用 execute_grep 或 read_file_range 深挖抗震规范

第 3 轮
  LLM 读取关键行附近原文

最终
  LLM 综合回答
  extract_references 提取引用依据
```

## 15. 设计原则

当前 `v6b` 的核心设计原则是：

```text
不要替代原 agent 的判断能力；
只在工具层给它更多低成本线索。
```

因此补充扫描不是完整第二套流程，也不是替代全库搜索，而是：

- 当 LLM 已经限定文件时，自动补一点其他文件索引
- 只给短预览，避免资料集变大时 token 爆炸
- 让 LLM 自己决定是否继续深挖

这保留了原流程的搜索精度，同时改善了跨文件召回广度。

