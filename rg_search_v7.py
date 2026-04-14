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
MODEL_DICT = {
    1: {'base_url': 'https://api.moonshot.cn/v1', 'api_key': 'kimi_key', 'model_name': 'kimi-k2.5'},
    2: {'base_url': 'https://integrate.api.nvidia.com/v1', 'api_key': 'nvidia_key', 'model_name': 'minimaxai/minimax-m2.5'},
    3: {'base_url': 'https://api-inference.modelscope.cn/v1', 'api_key': 'modelscope_key', 'model_name': 'Qwen/Qwen3-235B-A22B-Instruct-2507'},
    4: {'base_url': 'https://aigw-jnzs5.cucloud.cn:8443/v1', 'api_key': 'OPENAI_API_KEY', 'model_name': 'MiniMax-M2.5'},
}
MODEL_NAME = MODEL_DICT[4]["model_name"]
TARGET, INDEX_DIR = SCRIPT_DIR / "texts", SCRIPT_DIR / "texts" / ".index"
MAIN_INDEX, RG_EXE = INDEX_DIR / "index.json", (str(SCRIPT_DIR / "rg.exe") if (SCRIPT_DIR / "rg.exe").exists() else "rg")
FILE_MAP = {f: str(TARGET / f) for f in os.listdir(TARGET) if (TARGET / f).is_file() and f.endswith((".txt", ".md"))} if TARGET.exists() else {}
DETAIL_TOC_CACHE, SEARCH_RESULT_CACHE, CLIENT = {}, {}, None


# ==================== 基础工具函数 ====================
def get_client():
    global CLIENT
    if not CLIENT: CLIENT = OpenAI(base_url=MODEL_DICT[4]["base_url"], api_key=os.getenv(MODEL_DICT[4]["api_key"]))
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
    blocks, current = [], []
    for line in raw.split("\n"):
        (blocks.append(current), current := []) if line == "--" else current.append(line)
    if current: blocks.append(current)
    
    annotated = []
    for block in [b for b in blocks if b]:
        fp, ln, ct, fname, ctx = _parse_record_block(block)
        if ctx: annotated.append(f"[下面内容出自：{fname}-->{ctx.replace('[出自：', '').replace(']', '')}]")
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
    return "\n".join(annotated[:-1]) if annotated and annotated[-1] == "--" else "\n".join(annotated)


# ==================== Agent 工具函数 ====================
def get_document_toc(filename):
    print(f"📑 [Tool: TOC] 获取详细目录: {filename}\n")
    ensure_index_exists()
    for k, v in FILE_MAP.items():
        if filename in k:
            return json.dumps(json.load(open(INDEX_DIR / f"{Path(v).stem}.index.json", "r", encoding="utf-8")), ensure_ascii=False, indent=2)
    return json.dumps({"error": f"未找到 '{filename}'"}, ensure_ascii=False)

def execute_grep(pattern, include_files=None):
    cmd = [RG_EXE, "-n", "-i", "-H", "-C", "10", "-e", pattern, "-m", "50"]
    scope = "全库"
    if include_files:
        targets = [(FILE_MAP[k], k) for req in include_files.split(",") for k in FILE_MAP if req.strip() in k]
        if not targets: return f"系统反馈：'{include_files}' 未匹配任何文件"
        cmd.extend([t[0] for t in targets])
        scope = f"限定 {len(targets)} 个文件"
    else:
        cmd.append(str(TARGET))
    
    print(f"🛠️ [Grep] '{pattern}' ({scope})")
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8")
        if not res.stdout: return "系统反馈：未找到匹配项"
        
        records, current = [], []
        for line in res.stdout.strip().split("\n"):
            (records.append(current), current := []) if line == "--" else current.append(line)
        if current: records.append(current)
        
        total_match = sum(1 for r in records for l in r if _parse_grep_line(l))
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
        
        print(f"   📊 命中: {total_match}, 去重: {dup_count}")
        if not new_records: return "系统反馈：所有结果已重复"
        
        output_lines = sum([list(r) + ["--"] for r in new_records[:20]], [])[:-1]
        return f"系统反馈：{len(new_records)} 条新记录:\n{annotate_grep_output(chr(10).join(output_lines))}"
    except Exception as e:
        return f"系统反馈：搜索出错 {e}"

def read_file_range(filepath, start_line, end_line):
    print(f"📖 [Tool: Read] 阅读: {os.path.basename(filepath)} (行 {start_line}-{end_line})\n")
    try:
        path = filepath if os.path.exists(filepath) else FILE_MAP.get(os.path.basename(filepath), filepath)
        lines = open(path, "r", encoding="utf-8", errors="ignore").readlines()
        content = "".join(lines[max(0, start_line - 1):min(len(lines), end_line)])
        return f"--- {os.path.basename(path)} ---\n{content}\n--- 片段结束 ---"
    except Exception as e:
        return f"读取失败: {e}"


# ==================== 工具 Schema 定义 ====================
TOOLS_SCHEMA = [
    {"type": "function", "function": {"name": "get_document_toc", "description": "获取指定文档的详细章节目录", "parameters": {"type": "object", "properties": {"filename": {"type": "string"}}, "required": ["filename"]}}},
    {"type": "function", "function": {"name": "execute_grep", "description": "搜索关键词，返回匹配行及上下文", "parameters": {"type": "object", "properties": {"pattern": {"type": "string"}, "include_files": {"type": "string"}}, "required": ["pattern"]}}},
    {"type": "function", "function": {"name": "read_file_range", "description": "读取指定文件的特定行数范围", "parameters": {"type": "object", "properties": {"filepath": {"type": "string"}, "start_line": {"type": "integer"}, "end_line": {"type": "integer"}}, "required": ["filepath", "start_line", "end_line"]}}},
]

# ==================== Agent 主循环 ====================
def run_agent(user_question):
    global SEARCH_RESULT_CACHE
    SEARCH_RESULT_CACHE = {}
    print(f"🚀 V7 Agent ({len(FILE_MAP)} 文件) | 问题: {user_question}")
    
    messages = [
        {"role": "system", "content": f"""你是一个工程规范检索与解读专家。根据资料库内容回答，未提及的不要回答。

【资料库全局目录】
{get_global_toc_summary()}

【工具】: get_document_toc(获取目录), execute_grep(搜索), read_file_range(读取原文)
【纪律】: 1.必须调用工具查阅资料 2.必须明确引用依据 """},
        {"role": "user", "content": user_question},
    ]
    
    for turn in range(15):
        print(f"\n[第 {turn + 1} 轮]")
        try:
            response = get_client().chat.completions.create(model=MODEL_NAME, messages=messages, tools=TOOLS_SCHEMA, tool_choice="auto", temperature=1)
        except Exception as e:
            print(f"API Error: {e}")
            break
        
        msg = response.choices[0].message
        messages.append(msg)
        if r := getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", None): print(f"🧠 [思考]: {r[:200]}...")
        
        if msg.tool_calls:
            for tc in msg.tool_calls:
                args = json.loads(tc.function.arguments)
                func = {"execute_grep": execute_grep, "read_file_range": read_file_range, "get_document_toc": get_document_toc}[tc.function.name]
                messages.append({"role": "tool", "tool_call_id": tc.id, "content": func(**args)})
        else:
            print(f"\n✅ [回答]: {msg.content}")
            return
    
    messages.append({"role": "user", "content": "已达到最大搜索次数，请立即总结回答"})
    try:
        final = get_client().chat.completions.create(model=MODEL_NAME, messages=messages, temperature=1)
        print(f"\n✅ [最终回答]: {final.choices[0].message.content}")
    except Exception as e:
        print(f"生成失败: {e}")


# ==================== 程序入口 ====================
if __name__ == "__main__":
    run_agent("独立基础的宽高比")
