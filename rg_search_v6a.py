# ==================== 导入与初始化 ====================
import subprocess, json, os, sys
if hasattr(sys.stdout, "reconfigure"): sys.stdout.reconfigure(encoding="utf-8")
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv
from extract_toc.scanner import scan_folder

load_dotenv()
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path: sys.path.insert(0, str(SCRIPT_DIR))

# ==================== 配置与全局变量 ====================
num=4 #选择的模型序号
MODEL_DICT = {
    1: {'base_url': 'https://api.moonshot.cn/v1', 'api_key': 'kimi_key', 'model_name': 'kimi-k2.5', 'thinking': 'kimi'},
    2: {'base_url': 'https://integrate.api.nvidia.com/v1', 'api_key': 'nvidia_key', 'model_name': 'minimaxai/minimax-m2.7'},
    3: {'base_url': 'https://api-inference.modelscope.cn/v1', 'api_key': 'modelscope_key', 'model_name': 'Qwen/Qwen3-235B-A22B-Instruct-2507', 'thinking': 'qwen'},
    4: {'base_url': 'https://api-inference.modelscope.cn/v1', 'api_key': 'modelscope_key', 'model_name': 'Qwen/Qwen3.5-27B', 'thinking': 'qwen'},
    5: {'base_url': 'https://api-inference.modelscope.cn/v1', 'api_key': 'modelscope_key', 'model_name': 'Qwen/Qwen3-30B-A3B-Instruct-2507', 'thinking': 'qwen'},
    6: {'base_url': 'https://ollama.com/v1', 'api_key': 'ollama_key', 'model_name': 'gemma4:31b-cloud'},
    7: {'base_url': 'https://ollama.com/v1', 'api_key': 'ollama_key', 'model_name': 'qwen3.5:397b-cloud'},
    8: {'base_url': 'https://api.deepseek.com/v1', 'api_key': 'deepseek_key', 'model_name': 'deepseek-v4-flash', 'thinking': 'deepseek'},
}
MODEL_NAME = MODEL_DICT[num]["model_name"]
CONTENT_LINES=10
THINKING_ENABLED = True
print(f"🤖 当前使用模型: {MODEL_NAME} (序号: {num})")
TARGET, INDEX_DIR = SCRIPT_DIR / "texts", SCRIPT_DIR / "texts" / ".index"
MAIN_INDEX, RG_EXE = INDEX_DIR / "index.json", (str(SCRIPT_DIR / "rg.exe") if (SCRIPT_DIR / "rg.exe").exists() else "rg")
FILE_MAP = {f: str(TARGET / f) for f in os.listdir(TARGET) if (TARGET / f).is_file() and f.endswith((".txt", ".md"))} if TARGET.exists() else {}
DETAIL_TOC_CACHE, SEARCH_RESULT_CACHE, CLIENT = {}, {}, None

def _stream(core, stream=False):
    g = core()
    if stream: return g
    try:
        while True:
            x = next(g)
            print(*x[:1], end=x[1], flush=True) if isinstance(x, tuple) else print(x, flush=True)
    except StopIteration as e:
        return e.value

def _thinking_caps(cfg=None):
    cfg = cfg or MODEL_DICT[num]
    kind = cfg.get('thinking')
    forced = 'thinking' in cfg['model_name'].lower()
    return {'supported': bool(kind), 'can_disable': bool(kind) and not forced, 'forced': forced, 'kind': kind}

def build_chat_kwargs(messages, stream=False, tools=None, temperature=1):
    caps = _thinking_caps()
    # Kimi 关闭思考模式时 temperature 必须为 0.6
    if caps['kind'] == 'kimi' and caps['can_disable'] and not THINKING_ENABLED:
        temperature = 0.6
    
    kw = dict(model=MODEL_NAME, messages=messages, temperature=temperature, stream=stream)
    if tools: kw.update(tools=tools, tool_choice="auto")
    
    if caps['kind'] == 'kimi' and caps['can_disable'] and not THINKING_ENABLED:
        kw['extra_body'] = {'thinking': {'type': 'disabled'}}
    elif caps['kind'] == 'qwen' and caps['can_disable']:
        kw['extra_body'] = {'enable_thinking': THINKING_ENABLED}
    elif caps['kind'] == 'deepseek' and caps['can_disable']:
        kw['extra_body'] = {'thinking': {'type': 'enabled' if THINKING_ENABLED else 'disabled'}}
    return kw

def _chat_stream(messages, tools=None, show_reasoning=False):
    """流式消费模型输出，聚合回答文本、思考内容和工具调用。"""
    kw = build_chat_kwargs(messages, stream=True, tools=tools, temperature=1)
    rs, cs, tc_map, saw_r, saw_c = [], [], {}, False, False
    for chunk in get_client().chat.completions.create(**kw):
        delta = chunk.choices[0].delta
        if show_reasoning and (r := getattr(delta, "reasoning_content", None) or getattr(delta, "reasoning", None)):
            rs.append(r)
            if not saw_r: saw_r = True; yield ("🧠 [思考]: ", "")
            yield (r, "")
        if c := getattr(delta, "content", None):
            cs.append(c)
            if not saw_c:
                saw_c = True
                yield (("\n" if saw_r else "") + "\n✅ [回答]: ", "")
            yield (c, "")
        for tc in getattr(delta, "tool_calls", None) or []:
            cur = tc_map.setdefault(tc.index, {"id": "", "type": "function", "function": {"name": "", "arguments": ""}})
            if getattr(tc, "id", None): cur["id"] = tc.id
            if f := getattr(tc, "function", None):
                if getattr(f, "name", None): cur["function"]["name"] += f.name
                if getattr(f, "arguments", None): cur["function"]["arguments"] += f.arguments
    if saw_r or saw_c: yield ("", "\n")
    return {"role": "assistant", "content": "".join(cs) or None, "tool_calls": [tc_map[i] for i in sorted(tc_map)] or None}, "".join(rs)

def reset_search_cache():
    global SEARCH_RESULT_CACHE
    SEARCH_RESULT_CACHE = {}

def set_target_folder(folder_path: str):
    global TARGET, INDEX_DIR, MAIN_INDEX, FILE_MAP
    TARGET = Path(folder_path)
    INDEX_DIR = TARGET / ".index"
    MAIN_INDEX = INDEX_DIR / "index.json"
    FILE_MAP = {f: str(TARGET / f) for f in os.listdir(TARGET) 
                if (TARGET / f).is_file() and f.endswith((".txt", ".md"))}
    reset_search_cache()


# ==================== 基础工具函数 ====================
def get_client():
    global CLIENT
    if not CLIENT: CLIENT = OpenAI(base_url=MODEL_DICT[num]["base_url"], api_key=os.getenv(MODEL_DICT[num]["api_key"]))
    return CLIENT

def ensure_index_exists():
    if not MAIN_INDEX.exists(): scan_folder(str(TARGET), recursive=True, output_dir=str(INDEX_DIR))

def get_global_toc_summary():
    ensure_index_exists()
    return json.dumps(json.load(open(MAIN_INDEX, "r", encoding="utf-8")), ensure_ascii=False, indent=2)

def _load_detail_toc(stem):
    if stem not in DETAIL_TOC_CACHE:
        path = INDEX_DIR / f"{stem}.index.json"
        DETAIL_TOC_CACHE[stem] = json.load(open(path, "r", encoding="utf-8")).get("chapters", []) if path.exists() else []
    return DETAIL_TOC_CACHE[stem]

# ==================== 章节上下文注入 ====================
def get_chapter_context(filepath, line_num):
    """根据命中的文件和行号，补出所在章节和小节信息。"""
    for ch in _load_detail_toc(Path(filepath).stem):
        if ch.get("line", 0) <= line_num:
            best_sec = next((s for s in reversed(ch.get("sections", [])) if s.get("line", 0) <= line_num), None)
            title = f"{ch['title']} -> {best_sec['title']}" if best_sec else ch["title"]
            secs = [f"{s.get('title', '未命名')}(行{s.get('line', 0)})" for s in ch.get("sections", [])[:5]]
            return f"[出自：{title}{' | 本章小节: ' + ', '.join(secs) + ('...' if len(ch.get('sections', [])) > 5 else '')}]"
    return ""

def _parse_grep_line(line):
    parts = line.split(":")
    for i in range(1, len(parts) - 1):
        if parts[i].isdigit(): return (":".join(parts[:i]), int(parts[i]), ":".join(parts[i + 1:]))
    return None

def _parse_record_block(block):
    for line in block:
        if p := _parse_grep_line(line): return (p[0], p[1], p[2], os.path.basename(p[0]), get_chapter_context(p[0], p[1]))
    return (None, None, None, None, None)

def annotate_grep_output(raw):
    """把 grep 原始输出转成更适合给模型和人阅读的注解格式。"""
    blocks, current = [], []
    for line in raw.split("\n"):
        (blocks.append(current), current := []) if line == "--" else current.append(line)
    if current: blocks.append(current)
    
    annotated, ctx_found, ctx_not_found, failed = [], 0, 0, []
    for block in [b for b in blocks if b]:
        fp, ln, ct, fname, ctx = _parse_record_block(block)
        if ctx:
            annotated.append(f"[下面内容出自：{fname}-->{ctx.replace('[出自：', '').replace(']', '')}]")
            ctx_found += 1
        elif fp and ln:
            ctx_not_found += 1
            if len(failed) < 5: failed.append((fname, ln))
        
        for line in block:
            if p := _parse_grep_line(line): annotated.append(f"行号{p[1]}-->{p[2]}")
            else:
                matched = False
                for full_path in FILE_MAP.values():
                    for sep in ["-", ":"]:
                        if line.startswith(full_path + sep):
                            rest = line[len(full_path) + 1:]
                            if (parts := rest.split(sep, 1)) and len(parts) == 2 and parts[0].isdigit():
                                annotated.append(f"  行号{parts[0]}-->{parts[1]}")
                                matched = True
                                break
                    if matched: break
                if not matched: annotated.append(f"  {line}")
        annotated.append("--")
    
    debug = [f"      🔍 [注解] 解析块数: {len([b for b in blocks if b])}, 找到章节: {ctx_found}, 未找到: {ctx_not_found}"]
    if failed: debug.append(f"      ❌ [失败示例前三条]: {', '.join(f'{f}:L{ln}' for f, ln in failed[:3])}")
    return ("\n".join(annotated[:-1]) if annotated and annotated[-1] == "--" else "\n".join(annotated)), debug


# ==================== Agent 工具函数 ====================
def get_document_toc(filename, stream=False):
    def core():
        yield f"📑 [Tool: TOC] 获取详细目录: {filename}\n"
        ensure_index_exists()
        for k, v in FILE_MAP.items():
            if filename in k:
                return json.dumps(json.load(open(INDEX_DIR / f"{Path(v).stem}.index.json", "r", encoding="utf-8")), ensure_ascii=False, indent=2)
        return json.dumps({"error": f"未找到 '{filename}'"}, ensure_ascii=False)
    return _stream(core, stream)

def execute_grep(pattern, include_files=None, stream=False):
    """执行带上下文的 rg 搜索，并对结果做去重和章节注解。"""
    def core():
        cmd = [RG_EXE, "-n", "-i", "-H", "-C", str(CONTENT_LINES), "-e", pattern, "-m", "50"]
        scope = "全库"
        if include_files:
            targets = [(FILE_MAP[k], k) for req in include_files.split(",") for k in FILE_MAP if req.strip() in k]
            if not targets: return f"系统反馈：'{include_files}' 未匹配任何文件"
            cmd.extend([t[0] for t in targets]); scope = f"限定 {len(targets)} 个文件"
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
                (records.append(current), current := []) if line == "--" else current.append(line)
            if current: records.append(current)
            new_records, dup_count = [], 0
            for record in records:
                key = next((p[:2] for l in record if (p := _parse_grep_line(l))), None)
                if key and key not in SEARCH_RESULT_CACHE:
                    SEARCH_RESULT_CACHE[key] = True; new_records.append(record)
                elif key: dup_count += 1
                else: new_records.append(record)
            yield f"   📊 命中: {len(records)} 条, 去重: {dup_count} 条, 新记录: {len(new_records)} 条"
            if not new_records: return "系统反馈：所有结果已重复"
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
            path = filepath if os.path.exists(filepath) else FILE_MAP.get(os.path.basename(filepath), filepath)
            lines = open(path, "r", encoding="utf-8", errors="ignore").readlines()
            content = "".join(lines[max(0, start_line - 1):min(len(lines), end_line)])
            return f"--- {os.path.basename(path)} ---\n{content}\n--- 片段结束 ---"
        except Exception as e:
            return f"读取失败: {e}"
    return _stream(core, stream)


# ==================== 工具 Schema 定义 ====================
TOOLS_SCHEMA = [
    {"type": "function", "function": {"name": "get_document_toc", "description": "获取指定文档的详细章节目录", "parameters": {"type": "object", "properties": {"filename": {"type": "string"}}, "required": ["filename"]}}},
    {"type": "function", "function": {"name": "execute_grep", "description": "搜索关键词，返回匹配行及上下文", "parameters": {"type": "object", "properties": {"pattern": {"type": "string"}, "include_files": {"type": "string", "description": "指定要在哪些文件中搜索，填入文件名。为空则全库搜索。"}}, "required": ["pattern"]}}},
    {"type": "function", "function": {"name": "read_file_range", "description": "读取指定文件的特定行数范围", "parameters": {"type": "object", "properties": {"filepath": {"type": "string"}, "start_line": {"type": "integer"}, "end_line": {"type": "integer"}}, "required": ["filepath", "start_line", "end_line"]}}},
]

# ==================== Agent 主循环 ====================
def run_agent(user_question, show_reasoning=False, stream=False):
    """主循环：多轮调用工具，直到得到最终答案或达到轮次上限。"""
    def core():
        global SEARCH_RESULT_CACHE
        SEARCH_RESULT_CACHE = {}
        yield f"🚀 V7 Agent ({len(FILE_MAP)} 文件) | 问题: {user_question}"
        messages = [
            {"role": "system", "content": f"""你是一个工程规范检索与解读专家。根据资料库内容回答，未提及的不要回答。

【资料库全局目录】
{get_global_toc_summary()}

【工具】: get_document_toc(获取目录), execute_grep(搜索), read_file_range(读取原文)
【纪律】: 1.必须调用工具查阅资料 2.必须明确引用依据 """},
            {"role": "user", "content": user_question},
        ]
        for turn in range(15):
            yield f"\n[第 {turn + 1} 轮]"
            try:
                msg, _ = yield from _chat_stream(messages, TOOLS_SCHEMA, show_reasoning)
            except Exception as e:
                return f"API Error: {e}"
            messages.append(msg)
            if msg.get("tool_calls"):
                for tc in msg["tool_calls"]:
                    args = json.loads(tc["function"]["arguments"])
                    func = {"execute_grep": execute_grep, "read_file_range": read_file_range, "get_document_toc": get_document_toc}[tc["function"]["name"]]
                    messages.append({"role": "tool", "tool_call_id": tc["id"], "content": func(**args)})
            else:
                return msg.get("content")
        messages.append({"role": "user", "content": "已达到最大搜索次数，请立即总结回答"})
        try:
            final, _ = yield from _chat_stream(messages)
            return final.get("content")
        except Exception as e:
            return f"生成失败: {e}"
    return _stream(core, stream)


# ==================== 程序入口 ====================
if __name__ == "__main__":
    # run_agent("门刚什么时候应设置揽风绳")
    run_agent("门刚什么时候应设置揽风绳",show_reasoning=True)
