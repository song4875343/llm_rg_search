# ==================== 导入与初始化 ====================
import subprocess, json, os, sys, re

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
num = 11  # 选择的模型序号
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
DETAIL_TOC_CACHE, SEARCH_RESULT_CACHE, SUPPLEMENT_CACHE, SUPPLEMENT_KEY_MAP, CLIENT = (
    {},
    {},
    {},
    {},
    None,
)


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
    global SEARCH_RESULT_CACHE, SUPPLEMENT_CACHE, SUPPLEMENT_KEY_MAP
    SEARCH_RESULT_CACHE = {}
    SUPPLEMENT_CACHE = {}
    SUPPLEMENT_KEY_MAP = {}


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


def _find_match_span(pattern, content):
    try:
        m = re.search(pattern, content, flags=re.IGNORECASE)
        if m:
            return m.start(), m.end()
    except re.error:
        pass
    idx = content.lower().find(pattern.lower())
    return (idx, idx + len(pattern)) if idx >= 0 else (0, 0)


def _centered_preview(pattern, content, preview_chars=50):
    text = content.strip().replace("\t", " ")
    start, end = _find_match_span(pattern, text)
    if end > start:
        left = max(0, start - max(0, (preview_chars - (end - start)) // 2))
        right = min(len(text), left + preview_chars)
        left = max(0, right - preview_chars)
        preview = text[left:start] + "【" + text[start:end] + "】" + text[end:right]
    else:
        left, right = 0, min(len(text), preview_chars)
        preview = text[left:right]
    return ("..." if left > 0 else "") + preview + ("..." if right < len(text) else "")


def _supplemental_grep_lines(pattern, excluded_names, limit=30, preview_chars=50):
    """Search remaining files cheaply, cache hits, and return short index hints only."""
    global SUPPLEMENT_CACHE, SUPPLEMENT_KEY_MAP
    excluded_names = set(excluded_names or [])
    remaining = [
        (name, path) for name, path in FILE_MAP.items() if name not in excluded_names
    ]
    if not remaining:
        return ""
    cmd = [RG_EXE, "-n", "-i", "-H", "-e", pattern, "-m", "12"] + [
        path for _, path in remaining
    ]
    try:
        res = subprocess.run(
            cmd, capture_output=True, text=True, encoding="utf-8", errors="ignore"
        )
    except Exception:
        return ""
    if not res.stdout:
        return ""
    lines, seen, hit_count, dup_count = [], set(), 0, 0
    for raw in res.stdout.strip().splitlines():
        parsed = _parse_grep_line(raw)
        if not parsed:
            continue
        fp, ln, content = parsed
        key = (fp, ln)
        hit_count += 1
        if key in seen or key in SEARCH_RESULT_CACHE or key in SUPPLEMENT_KEY_MAP:
            dup_count += 1
            continue
        seen.add(key)
        fname = os.path.basename(fp)
        ctx = get_chapter_context(fp, ln)
        sid = f"S{len(SUPPLEMENT_CACHE) + 1:03d}"
        SUPPLEMENT_KEY_MAP[key] = sid
        SUPPLEMENT_CACHE[sid] = {
            "file": fp,
            "filename": fname,
            "line": ln,
            "content": content,
            "pattern": pattern,
        }
        preview = _centered_preview(pattern, content, preview_chars)
        lines.append(
            f"补充ID={sid} | 文件={fname} | 原文行={ln} {ctx}\n索引预览-->{preview}"
        )
        if len(lines) >= limit:
            break
    header = f"\n\n【补充扫描索引：其余文件也有命中。补充命中: {hit_count} 条, 去重: {dup_count} 条, 新记录: {len(lines)} 条。这里只保留关键词居中预览；如可能相关，请继续调用 fetch_supplemental 或 read_file_range 深挖。】\n"
    if not lines:
        return header if hit_count else ""
    return header + "\n".join(lines)


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
                return (
                    "系统反馈：限定范围未找到匹配项"
                    if include_files
                    else "系统反馈：未找到匹配项"
                )
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
            supplement = ""
            if include_files and len(records) > 0:
                yield "      🔎 [补充扫描] 开始检索其余文件"
                supplement = _supplemental_grep_lines(
                    pattern, [name for _, name in targets]
                )
                if supplement:
                    if m := re.search(
                        r"补充命中: (\d+) 条, 去重: (\d+) 条, 新记录: (\d+) 条",
                        supplement,
                    ):
                        yield f"      🔎 [补充扫描] 命中: {m.group(1)} 条, 去重: {m.group(2)} 条, 新记录: {m.group(3)} 条"
                    else:
                        yield "      🔎 [补充扫描] 发现其余文件单行命中"
                else:
                    yield "      🔎 [补充扫描] 其余文件无新增命中"
            if not new_records:
                return (
                    "系统反馈：所有结果已重复" + supplement
                    if supplement
                    else "系统反馈：所有结果已重复"
                )
            output_lines = sum([list(r) + ["--"] for r in new_records[:20]], [])[:-1]
            annotated, debug = annotate_grep_output(chr(10).join(output_lines))
            yield from debug
            return f"系统反馈：{len(new_records)} 条新记录:\n{annotated}{supplement}"
        except Exception as e:
            return f"系统反馈：搜索出错 {e}"

    return _stream(core, stream)


def _normalize_supplement_id(sid):
    m = re.fullmatch(r"S0*(\d+)", (sid or "").strip().upper())
    return f"S{int(m.group(1)):03d}" if m else (sid or "").strip().upper()


def _resolve_supplement_id(raw_id):
    """Resolve exact IDs first; tolerate the model confusing source line numbers for S IDs."""
    sid = _normalize_supplement_id(raw_id)
    if sid in SUPPLEMENT_CACHE:
        return sid, None
    m = re.fullmatch(r"S0*(\d+)", (raw_id or "").strip().upper())
    if not m:
        return sid, None
    line_num = int(m.group(1))
    matches = [
        cache_id
        for cache_id, rec in SUPPLEMENT_CACHE.items()
        if rec.get("line") == line_num
    ]
    if len(matches) == 1:
        return matches[0], f"{raw_id}->{matches[0]}(按原文行{line_num}纠正)"
    if len(matches) > 1:
        return (
            sid,
            f"{raw_id}(像原文行{line_num}，但匹配到多个补充索引: {', '.join(matches)})",
        )
    return sid, None


def fetch_supplemental(ids, context_lines=CONTENT_LINES, stream=False):
    def core():
        yield f"🔎 [补充扫描] 开始读取: {ids}\n"
        raw_requested = [
            x.strip().upper() for x in re.split(r"[,，\s]+", ids or "") if x.strip()
        ]
        if not raw_requested:
            return "系统反馈：未提供补充索引编号，请传入如 S001,S003"
        requested, seen, dup_count, corrections, ambiguous = [], set(), 0, [], []
        for raw_id in raw_requested:
            sid, note = _resolve_supplement_id(raw_id)
            if note:
                if "多个补充索引" in note:
                    ambiguous.append(note)
                else:
                    corrections.append(note)
            if sid in seen:
                dup_count += 1
                continue
            seen.add(sid)
            requested.append(sid)
        yield f"   📊 请求: {len(raw_requested)} 个, 去重: {dup_count} 个, 待读取: {len(requested)} 个\n"
        if corrections:
            yield "   🔧 [编号纠正] " + "; ".join(corrections) + "\n"
        if ambiguous:
            yield "   ⚠️ [编号疑似行号] " + "; ".join(ambiguous) + "\n"

        blocks, missing, read_count = [], [], 0
        for sid in requested:
            rec = SUPPLEMENT_CACHE.get(sid)
            if not rec:
                missing.append(sid)
                continue
            fp, ln = rec["file"], rec["line"]
            preview30 = rec.get("content", "").strip().replace("\t", " ")[:30]
            yield f"   ↳ [{sid}] {rec['filename']} 行{ln} | {preview30}\n"
            try:
                file_lines = open(
                    fp, "r", encoding="utf-8", errors="ignore"
                ).readlines()
                start = max(1, ln - int(context_lines))
                end = min(len(file_lines), ln + int(context_lines))
                SEARCH_RESULT_CACHE[(fp, ln)] = True
                body = "".join(file_lines[start - 1 : end])
                ctx = get_chapter_context(fp, ln)
                read_count += 1
                blocks.append(
                    f"--- [{sid}] {rec['filename']} 行{start}-{end} 命中行{ln} {ctx} ---\n{body}\n--- [{sid}] 片段结束 ---"
                )
            except Exception as e:
                blocks.append(f"--- [{sid}] 读取失败: {e} ---")
        if missing:
            msg = "未找到补充索引编号: " + ", ".join(missing)
            yield f"   ❌ {msg}\n"
            blocks.append(msg)
        yield f"   📊 读取完成: 成功 {read_count} 个, 缺失 {len(missing)} 个\n"
        return "\n\n".join(blocks) if blocks else "系统反馈：未找到可读取的补充索引"

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
            "name": "fetch_supplemental",
            "description": "按补充扫描索引编号批量读取原文上下文",
            "parameters": {
                "type": "object",
                "properties": {
                    "ids": {
                        "type": "string",
                        "description": "补充索引编号，多个用逗号分隔，如 S001,S003",
                    },
                    "context_lines": {
                        "type": "integer",
                        "description": "命中行上下文行数，默认使用系统 CONTENT_LINES",
                    },
                },
                "required": ["ids"],
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
                    "content": f"对话历史:\n{json.dumps(messages, ensure_ascii=False)}\n\n最终回答:\n{final_answer}\n\n请提取依据并调用output_references函数。",
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

        reset_search_cache()
        yield f"🚀 V7 Agent ({len(FILE_MAP)} 文件) | 问题: {user_question}"
        messages = [
            {
                "role": "system",
                "content": f"""你是一个工程规范检索与解读专家。根据资料库内容回答，未提及的不要回答。

【资料库全局目录】
{get_global_toc_summary()}

【工具】: get_document_toc(获取目录), execute_grep(搜索), fetch_supplemental(批量精读补充索引编号), read_file_range(读取原文)
【纪律】:
1. 必须调用工具查阅资料。
2. 必须明确引用依据（如某规范第X条）。
3. 信息不足时继续换关键词、查目录或读原文深挖，批量精读补充索引内容，直到获得确凿证据。
4. 交叉验证防遗漏：必须全面收集所有相关规范中的信息，严禁“找到一处关联条款就立刻停止检索”的早退行为，最终回答最少进行一次批量精读补充索引内容""",
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
                        "fetch_supplemental": fetch_supplemental,
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
    # run_agent("高烈度区能否用砌体女儿墙")
    run_agent("门刚何时应采用揽风绳")
