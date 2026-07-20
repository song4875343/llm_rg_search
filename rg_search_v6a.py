# ==================== 导入与初始化 ====================
import subprocess, json, os, sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
from extract_toc.scanner import scan_folder
from model_config import MODEL_CONFIG as MODEL_DICT

load_dotenv()
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# ==================== 配置与全局变量 ====================
num = 12  # 选择的模型序号
MODEL_NAME = MODEL_DICT[num]["model_name"]
CONTENT_LINES = 10
THINKING_ENABLED = False
print(f"🤖 当前使用模型: {MODEL_NAME} (序号: {num})")
# TARGET, INDEX_DIR = SCRIPT_DIR / "texts", SCRIPT_DIR / "texts" / ".index"
TARGET, INDEX_DIR = SCRIPT_DIR / "specs", SCRIPT_DIR / "specs" / ".index"
MAIN_INDEX, RG_EXE = (
    INDEX_DIR / "index.json",
    (str(SCRIPT_DIR / "rg.exe") if (SCRIPT_DIR / "rg.exe").exists() else "rg"),
)
FILE_MAP = (
    {
        f: str(TARGET / f)
        for f in os.listdir(TARGET)
        if (TARGET / f).is_file() and f.endswith((".txt", ".md"))
    }
    if TARGET.exists()
    else {}
)
DETAIL_TOC_CACHE, SEARCH_RESULT_CACHE, CLIENT = {}, {}, None


def _stream(core, stream=False):
    g = core()
    if stream:
        return g
    try:
        while True:
            x = next(g)
            print(*x[:1], end=x[1], flush=True) if isinstance(x, tuple) else print(
                x, flush=True
            )
    except StopIteration as e:
        return e.value


def _thinking_caps(cfg=None):
    cfg = cfg or MODEL_DICT[num]
    kind = cfg.get("thinking")
    forced = "thinking" in cfg["model_name"].lower()
    return {
        "supported": bool(kind),
        "can_disable": bool(kind) and not forced,
        "forced": forced,
        "kind": kind,
    }


def build_chat_kwargs(
    messages, stream=False, tools=None, temperature=1, thinking_enabled_override=None
):
    caps = _thinking_caps()
    thinking_enabled = (
        THINKING_ENABLED
        if thinking_enabled_override is None
        else thinking_enabled_override
    )
    # Kimi 关闭思考模式时 temperature 必须为 0.6
    if caps["kind"] == "kimi" and caps["can_disable"] and not thinking_enabled:
        temperature = 0.6

    kw = dict(
        model=MODEL_NAME, messages=messages, temperature=temperature, stream=stream
    )
    if tools:
        kw.update(tools=tools, tool_choice="auto")

    if caps["kind"] == "kimi" and caps["can_disable"] and not thinking_enabled:
        kw["extra_body"] = {"thinking": {"type": "disabled"}}
    elif caps["kind"] == "qwen" and caps["can_disable"]:
        kw["extra_body"] = {"enable_thinking": thinking_enabled}
    elif caps["kind"] == "deepseek" and caps["can_disable"]:
        kw["extra_body"] = {
            "thinking": {"type": "enabled" if thinking_enabled else "disabled"}
        }
    return kw


def _chat_stream(
    messages,
    tools=None,
    show_reasoning=False,
    thinking_enabled_override=None,
    stream_content=True,
):
    """流式消费模型输出，聚合回答文本、思考内容和工具调用。"""
    kw = build_chat_kwargs(
        messages,
        stream=True,
        tools=tools,
        temperature=1,
        thinking_enabled_override=thinking_enabled_override,
    )
    rs, cs, tc_map, saw_r, saw_c = [], [], {}, False, False
    for chunk in get_client().chat.completions.create(**kw):
        if not chunk.choices:
            continue  # 阿里云的模型在流式返回的最后一个 chunk 里，choices 是空列表 []，代码还在按 chunk.choices[0] 去取，就会直接越界产生错误。
        delta = chunk.choices[0].delta
        if r := getattr(delta, "reasoning_content", None) or getattr(
            delta, "reasoning", None
        ):
            rs.append(r)
            if show_reasoning:
                if not saw_r:
                    saw_r = True
                    yield ("🧠 [思考]: ", "")
                yield (r, "")
        if c := getattr(delta, "content", None):
            cs.append(c)
            if stream_content and not saw_c:
                saw_c = True
                yield (("\n" if saw_r else "") + "\n✅ [回答]: ", "")
            if stream_content:
                yield (c, "")
        for tc in getattr(delta, "tool_calls", None) or []:
            cur = tc_map.setdefault(
                tc.index,
                {
                    "id": "",
                    "type": "function",
                    "function": {"name": "", "arguments": ""},
                },
            )
            if getattr(tc, "id", None):
                cur["id"] = tc.id
            if f := getattr(tc, "function", None):
                if getattr(f, "name", None):
                    cur["function"]["name"] += f.name
                if getattr(f, "arguments", None):
                    cur["function"]["arguments"] += f.arguments
    if saw_r or saw_c:
        yield ("", "\n")
    return {
        "role": "assistant",
        "content": "".join(cs) or None,
        "tool_calls": [tc_map[i] for i in sorted(tc_map)] or None,
        "reasoning_content": "".join(rs) or None,
    }, "".join(rs)


def reset_search_cache():
    global SEARCH_RESULT_CACHE
    SEARCH_RESULT_CACHE = {}


def set_target_folder(folder_path: str):
    global TARGET, INDEX_DIR, MAIN_INDEX, FILE_MAP
    TARGET = Path(folder_path)
    INDEX_DIR = TARGET / ".index"
    MAIN_INDEX = INDEX_DIR / "index.json"
    FILE_MAP = {
        f: str(TARGET / f)
        for f in os.listdir(TARGET)
        if (TARGET / f).is_file() and f.endswith((".txt", ".md"))
    }
    reset_search_cache()


# ==================== 基础工具函数 ====================
def get_client():
    global CLIENT
    if not CLIENT:
        CLIENT = OpenAI(
            base_url=MODEL_DICT[num]["base_url"],
            api_key=os.getenv(MODEL_DICT[num]["api_key"]),
        )
    return CLIENT


def ensure_index_exists():
    if not MAIN_INDEX.exists():
        scan_folder(str(TARGET), recursive=True, output_dir=str(INDEX_DIR))


def get_global_toc_summary():
    ensure_index_exists()
    return json.dumps(
        json.load(open(MAIN_INDEX, "r", encoding="utf-8")), ensure_ascii=False, indent=2
    )


def _load_detail_toc(stem):
    if stem not in DETAIL_TOC_CACHE:
        path = INDEX_DIR / f"{stem}.index.json"
        DETAIL_TOC_CACHE[stem] = (
            json.load(open(path, "r", encoding="utf-8")).get("chapters", [])
            if path.exists()
            else []
        )
    return DETAIL_TOC_CACHE[stem]


# ==================== 章节上下文注入 ====================
def get_chapter_context(filepath, line_num):
    """根据命中的文件和行号，补出所在章节和小节信息。"""
    for ch in _load_detail_toc(Path(filepath).stem):
        if ch.get("line", 0) <= line_num:
            best_sec = next(
                (
                    s
                    for s in reversed(ch.get("sections", []))
                    if s.get("line", 0) <= line_num
                ),
                None,
            )
            title = f"{ch['title']} -> {best_sec['title']}" if best_sec else ch["title"]
            secs = [
                f"{s.get('title', '未命名')}(行{s.get('line', 0)})"
                for s in ch.get("sections", [])[:5]
            ]
            return f"[出自：{title}{' | 本章小节: ' + ', '.join(secs) + ('...' if len(ch.get('sections', [])) > 5 else '')}]"
    return ""


def _parse_grep_line(line):
    parts = line.split(":")
    for i in range(1, len(parts) - 1):
        if parts[i].isdigit():
            return (":".join(parts[:i]), int(parts[i]), ":".join(parts[i + 1 :]))
    return None


def _parse_record_block(block):
    for line in block:
        if p := _parse_grep_line(line):
            return (
                p[0],
                p[1],
                p[2],
                os.path.basename(p[0]),
                get_chapter_context(p[0], p[1]),
            )
    return (None, None, None, None, None)


def annotate_grep_output(raw):
    """把 grep 原始输出转成更适合给模型和人阅读的注解格式。"""
    blocks, current = [], []
    for line in raw.split("\n"):
        (blocks.append(current), current := []) if line == "--" else current.append(
            line
        )
    if current:
        blocks.append(current)

    annotated, ctx_found, ctx_not_found, failed = [], 0, 0, []
    for block in [b for b in blocks if b]:
        fp, ln, ct, fname, ctx = _parse_record_block(block)
        if ctx:
            annotated.append(
                f"[下面内容出自：{fname}-->{ctx.replace('[出自：', '').replace(']', '')}]"
            )
            ctx_found += 1
        elif fp and ln:
            ctx_not_found += 1
            if len(failed) < 5:
                failed.append((fname, ln))

        for line in block:
            if p := _parse_grep_line(line):
                annotated.append(f"行号{p[1]}-->{p[2]}")
            else:
                matched = False
                for full_path in FILE_MAP.values():
                    for sep in ["-", ":"]:
                        if line.startswith(full_path + sep):
                            rest = line[len(full_path) + 1 :]
                            if (
                                (parts := rest.split(sep, 1))
                                and len(parts) == 2
                                and parts[0].isdigit()
                            ):
                                annotated.append(f"  行号{parts[0]}-->{parts[1]}")
                                matched = True
                                break
                    if matched:
                        break
                if not matched:
                    annotated.append(f"  {line}")
        annotated.append("--")

    debug = [
        f"      🔍 [注解] 解析块数: {len([b for b in blocks if b])}, 找到章节: {ctx_found}, 未找到: {ctx_not_found}"
    ]
    if failed:
        debug.append(
            f"      ❌ [失败示例前三条]: {', '.join(f'{f}:L{ln}' for f, ln in failed[:3])}"
        )
    return (
        "\n".join(annotated[:-1])
        if annotated and annotated[-1] == "--"
        else "\n".join(annotated)
    ), debug


# ==================== Agent 工具函数 ====================
def get_document_toc(filename, stream=False):
    def core():
        yield f"📑 [Tool: TOC] 获取详细目录: {filename}\n"
        ensure_index_exists()
        for k, v in FILE_MAP.items():
            if filename in k:
                return json.dumps(
                    json.load(
                        open(
                            INDEX_DIR / f"{Path(v).stem}.index.json",
                            "r",
                            encoding="utf-8",
                        )
                    ),
                    ensure_ascii=False,
                    indent=2,
                )
        return json.dumps({"error": f"未找到 '{filename}'"}, ensure_ascii=False)

    return _stream(core, stream)


def execute_grep(pattern, include_files=None, stream=False):
    """执行带上下文的 rg 搜索，并对结果做去重和章节注解。"""

    def core():
        cmd = [
            RG_EXE,
            "-n",
            "-i",
            "-H",
            "-C",
            str(CONTENT_LINES),
            "-e",
            pattern,
            "-m",
            "50",
        ]
        scope = "全库"
        if include_files:
            targets = [
                (FILE_MAP[k], k)
                for req in include_files.split(",")
                for k in FILE_MAP
                if req.strip() in k
            ]
            if not targets:
                return f"系统反馈：'{include_files}' 未匹配任何文件"
            cmd.extend([t[0] for t in targets])
            scope = f"限定 {len(targets)} 个文件"
        else:
            cmd.append(str(TARGET))
        yield f"🛠️ [Grep] '{pattern}' ({scope})"
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
            if not res.stdout:
                yield f"   📊 0命中"
                return "系统反馈：未找到匹配项"
            records, current = [], []
            for line in res.stdout.strip().split("\n"):
                (
                    records.append(current),
                    current := [],
                ) if line == "--" else current.append(line)
            if current:
                records.append(current)
            new_records, dup_count = [], 0
            for record in records:
                key = next((p[:2] for l in record if (p := _parse_grep_line(l))), None)
                if key and key not in SEARCH_RESULT_CACHE:
                    SEARCH_RESULT_CACHE[key] = True
                    new_records.append(record)
                elif key:
                    dup_count += 1
                else:
                    new_records.append(record)
            yield f"   📊 命中: {len(records)} 条, 去重: {dup_count} 条, 新记录: {len(new_records)} 条"
            if not new_records:
                return "系统反馈：所有结果已重复"
            output_lines = sum([list(r) + ["--"] for r in new_records[:20]], [])[:-1]
            annotated, debug = annotate_grep_output(chr(10).join(output_lines))
            yield from debug
            return f"系统反馈：{len(new_records)} 条新记录:\n{annotated}"
        except Exception as e:
            return f"系统反馈：搜索出错 {e}"

    return _stream(core, stream)


def read_file_range(filepath, start_line, end_line, stream=False):
    def core():
        yield f"📖 [Tool: Read] 阅读: {os.path.basename(filepath)} (行 {start_line}-{end_line})\n"
        try:
            path = (
                filepath
                if os.path.exists(filepath)
                else FILE_MAP.get(os.path.basename(filepath), filepath)
            )
            lines = open(path, "r", encoding="utf-8", errors="ignore").readlines()
            content = "".join(lines[max(0, start_line - 1) : min(len(lines), end_line)])
            return f"--- {os.path.basename(path)} ---\n{content}\n--- 片段结束 ---"
        except Exception as e:
            return f"读取失败: {e}"

    return _stream(core, stream)


# ==================== 工具 Schema 定义 ====================
TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "get_document_toc",
            "description": "获取指定文档的详细章节目录",
            "parameters": {
                "type": "object",
                "properties": {"filename": {"type": "string"}},
                "required": ["filename"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "execute_grep",
            "description": "搜索关键词，返回匹配行及上下文",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"},
                    "include_files": {
                        "type": "string",
                        "description": "指定要在哪些文件中搜索，填入文件名。为空则全库搜索。",
                    },
                },
                "required": ["pattern"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file_range",
            "description": "读取指定文件的特定行数范围",
            "parameters": {
                "type": "object",
                "properties": {
                    "filepath": {"type": "string"},
                    "start_line": {"type": "integer"},
                    "end_line": {"type": "integer"},
                },
                "required": ["filepath", "start_line", "end_line"],
            },
        },
    },
]
EXTRACT_REFERENCES_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "output_references",
            "description": "输出回答依据",
            "parameters": {
                "type": "object",
                "properties": {
                    "references": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "filename": {"type": "string"},
                                "line_number": {"type": "integer"},
                                "end_line": {"type": "integer"},
                                "article_number": {"type": "string"},
                            },
                            "required": ["filename", "line_number", "article_number"],
                        },
                    }
                },
                "required": ["references"],
            },
        },
    }
]


def extract_references(messages, final_answer):
    """调用大模型提取回答依据"""
    try:
        kw = build_chat_kwargs(
            [
                {
                    "role": "system",
                    "content": "你是一个JSON提取专家。从对话历史中提取回答依据：文件名、起始行号line_number、结束行号end_line、条目号（如'第3.2.1条'或'(条文解释)3.2.1'，绝大部分有条目号，实在无则为空字符串）。遇到'行2539 [RG]'时输出line_number=2539,end_line=2539；遇到'行10776-10790 [CHUNK]'时输出line_number=10776,end_line=10790。只调用output_references函数，不要输出其他内容。",
                },
                {
                    "role": "user",
                    "content": f"对话历史:\n{json.dumps(messages[-10:], ensure_ascii=False)}\n\n最终回答:\n{final_answer}\n\n请提取依据并调用output_references函数。",
                },
            ],
            tools=EXTRACT_REFERENCES_SCHEMA,
            stream=False,
        )
        resp = get_client().chat.completions.create(**kw)
        if resp.choices and (tc := resp.choices[0].message.tool_calls):
            args_str = tc[0].function.arguments.strip()
            # 尝试找到第一个完整的 JSON 对象
            decoder = json.JSONDecoder()
            obj, idx = decoder.raw_decode(args_str)
            return obj
    except json.JSONDecodeError as e:
        print(f"⚠️ JSON解析失败: {e}")
        if "args_str" in locals():
            print(f"   原始内容: {repr(args_str[:200])}")
    except Exception as e:
        print(f"⚠️ 依据提取失败: {e}")
    return None


# ==================== Agent 主循环 ====================
def run_agent(user_question, show_reasoning=False, stream=False, extract_refs=True):
    """主循环：多轮调用工具，直到得到最终答案或达到轮次上限。"""

    def core():
        def output_final(msg_obj):
            """生成并输出最终答案和依据"""
            try:
                final, _ = yield from _chat_stream(
                    messages, thinking_enabled_override=False
                )
                content = final.get("content") or msg_obj.get("content")
            except Exception as e:
                content = msg_obj.get("content") or f"API Error: {e}"

            yield content if content else "生成失败: 最终回答为空"
            if extract_refs and content:
                refs = extract_references(messages, content)
                if refs:
                    yield "\n" + "=" * 60 + "\n📚 [回答依据]:\n"
                    for i, r in enumerate(refs.get("references", []), 1):
                        yield f"  [{i}] {r['filename']} 行{r['line_number']}" + (
                            f" {r['article_number']}" if r.get("article_number") else ""
                        )
                    yield "\n" + "=" * 60

        global SEARCH_RESULT_CACHE
        SEARCH_RESULT_CACHE = {}
        yield f"🚀 V7 Agent ({len(FILE_MAP)} 文件) | 问题: {user_question}"
        messages = [
            {
                "role": "system",
                "content": f"""你是一个工程规范检索与解读专家。根据资料库内容回答，未提及的不要回答。

【资料库全局目录】
{get_global_toc_summary()}

【工具】: get_document_toc(获取目录), execute_grep(搜索), read_file_range(读取原文)
【纪律】:
1. 必须调用工具查阅资料。
2. 必须明确引用依据（如某规范第X条）。
3. 信息不足时继续换关键词、查目录或读原文深挖，直到获得确凿证据。
4. 交叉验证防遗漏：必须全面收集所有相关规范中的信息，严禁“找到一处关联条款就立刻停止检索”的早退行为。""",
            },
            {"role": "user", "content": user_question},
        ]

        for turn in range(15):
            yield f"\n[第 {turn + 1} 轮]"
            try:
                msg, _ = yield from _chat_stream(
                    messages, TOOLS_SCHEMA, show_reasoning, stream_content=False
                )
            except Exception as e:
                yield f"API Error: {e}"
                return
            messages.append(msg)
            if msg.get("tool_calls"):
                for tc in msg["tool_calls"]:
                    print(tc)
                    args = json.loads(tc["function"]["arguments"])
                    func = {
                        "execute_grep": execute_grep,
                        "read_file_range": read_file_range,
                        "get_document_toc": get_document_toc,
                    }[tc["function"]["name"]]
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc["id"],
                            "content": func(**args),
                        }
                    )
            else:
                yield "\n✅ [最终答案] 流式输出中..."
                yield from output_final(msg)
                return

        messages.append(
            {"role": "user", "content": "已达到最大搜索次数，请立即总结回答"}
        )
        yield from output_final({})

    return _stream(core, stream)


# ==================== 程序入口 ====================
if __name__ == "__main__":
    run_agent("高烈度区能否用砌体女儿墙")
    # run_agent("门刚何时应采用揽风绳")
