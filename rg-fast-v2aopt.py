"""智能检索系统：LLM Function Call + rg 召回 + BM25 排序。"""

import os, re, json, subprocess
from openai import OpenAI
from dotenv import load_dotenv
from bm25_module import BM25

load_dotenv()

# ===== 模型与客户端初始化 =====
MODEL_CONFIG = {
    1: {'base_url': 'https://api.moonshot.cn/v1', 'api_key': 'kimi_key', 'model_name': 'kimi-k2.5', 'thinking': 'kimi'},
    2: {'base_url': 'https://integrate.api.nvidia.com/v1', 'api_key': 'nvidia_key', 'model_name': 'minimaxai/minimax-m2.7'},
    3: {'base_url': 'https://api-inference.modelscope.cn/v1', 'api_key': 'modelscope_key', 'model_name': 'Qwen/Qwen3-235B-A22B-Instruct-2507', 'thinking': 'qwen'},
    4: {'base_url': 'https://api-inference.modelscope.cn/v1', 'api_key': 'modelscope_key', 'model_name': 'Qwen/Qwen3.5-27B', 'thinking': 'qwen'},
    5: {'base_url': 'https://api-inference.modelscope.cn/v1', 'api_key': 'modelscope_key', 'model_name': 'Qwen/Qwen3-30B-A3B-Instruct-2507', 'thinking': 'qwen'},
    6: {'base_url': 'https://ollama.com/v1', 'api_key': 'ollama_key', 'model_name': 'gemma4:31b-cloud'},
    7: {'base_url': 'https://ollama.com/v1', 'api_key': 'ollama_key', 'model_name': 'qwen3.5:397b-cloud'},
    8: {'base_url': 'https://api.deepseek.com/v1', 'api_key': 'deepseek_key', 'model_name': 'deepseek-v4-flash', 'thinking': 'deepseek'},
}

MODEL_NUM, THINKING_ENABLED = 5, True
SHOW_THINKING_STREAM = True
config = MODEL_CONFIG[MODEL_NUM]
client = OpenAI(base_url=config['base_url'], api_key=os.getenv(config['api_key']))
model_name = config['model_name']
print(f"🤖 使用模型: {model_name}，序号{MODEL_NUM}")

# ===== Function Call 工具定义 =====
TOOLS = [{
    "type": "function",
    "function": {
        "name": "search_documents",
        "description": "搜索文档并返回最相关的内容片段",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "用户的原始问题，用于 BM25 相关性计算"},
                "broad_keywords": {"type": "array", "items": {"type": "string"}, "description": "宽泛关键词列表，用于初步召回"},
                "exact_keywords": {"type": "array", "items": {"type": "string"}, "description": "1元素的精确关键词列表，用于评定返回内容的命中可能性"},
                "target_files": {"type": "array", "items": {"type": "string"}, "description": "最可能包含答案的文件名列表（从可用文件列表中选择）"},
            },
            "required": ["query", "broad_keywords", "target_files"],
        },
    },
}]

# 辅助函数：提示词构造、流式输出、切块、BM25 打分、rg 解析、去重融合、结果格式化
# ================= 辅助函数 =================
def _delta_piece(delta):
    extra = getattr(delta, 'model_extra', None) or {}
    fields = ('reasoning_content', 'reasoning', 'thinking_content', 'thinking') if SHOW_THINKING_STREAM else ()
    for name in fields:
        if text := (getattr(delta, name, None) or extra.get(name)):
            return 'think', text
    return ('answer', text) if (text := (getattr(delta, 'content', None) or extra.get('content'))) else ('', '')

def emit(stream, data, tool_buf=None, label_answer=True):
    if not stream:
        print(data)
        return
    if isinstance(data, str):
        yield data
        return
    shown = set()
    for chunk in data:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        for tc in getattr(delta, 'tool_calls', None) or []:
            item = tool_buf.setdefault(tc.index, {'id': '', 'function': {'name': '', 'arguments': ''}}) if tool_buf is not None else None
            if item is not None:
                item['id'] = getattr(tc, 'id', None) or item['id']
                fn = getattr(tc, 'function', None)
                item['function']['name'] += (getattr(fn, 'name', None) or '') if fn else ''
                item['function']['arguments'] += (getattr(fn, 'arguments', None) or '') if fn else ''
        kind, text = _delta_piece(delta)
        if text:
            if kind == 'think' and kind not in shown:
                shown.add(kind); yield "\n\n🧠 思考过程：\n"
            elif kind == 'answer' and label_answer and kind not in shown:
                shown.add(kind); yield "\n\n✅ 最终答案:\n"
            yield text

def read_text(path):
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        return f.read()

def available_files(search_dir):
    return sorted(f for _, _, files in os.walk(search_dir) for f in files if f.endswith(('.txt', '.md')))

def candidate_key(item):
    return ('rg', item['file'], item['line_num']) if item['type'] == 'rg' else ('chunk', item['file'], item['start_pos'])

def split_sentences(text):
    return [s.strip() for s in re.findall(r'[^。！？.!?]+[。！？.!?]?', text.strip()) if s.strip()]

def find_file(search_dir, filename):
    return next((os.path.join(root, filename) for root, _, files in os.walk(search_dir) if filename in files), None)

def thinking_caps(cfg=None):
    cfg = cfg or config
    kind = cfg.get('thinking')
    forced = 'thinking' in cfg['model_name'].lower()
    return {'kind': kind, 'supported': bool(kind), 'forced': forced, 'can_disable': bool(kind) and not forced}

def build_chat_kwargs(messages, stream=False, tools=None, temperature=1):
    caps = thinking_caps()
    if caps['kind'] == 'kimi' and caps['can_disable'] and not THINKING_ENABLED:
        temperature = 0.6
    kwargs = {'model': model_name, 'messages': messages, 'temperature': temperature, 'stream': stream}
    if tools:
        kwargs.update(tools=tools, tool_choice='auto')
    extra_body = (
        {'thinking': {'type': 'disabled'}} if caps['kind'] == 'kimi' and caps['can_disable'] and not THINKING_ENABLED else
        {'enable_thinking': THINKING_ENABLED} if caps['kind'] == 'qwen' and caps['can_disable'] else
        {'thinking': {'type': 'enabled' if THINKING_ENABLED else 'disabled'}} if caps['kind'] == 'deepseek' and caps['can_disable'] else
        None
    )
    if extra_body:
        kwargs['extra_body'] = extra_body
    return kwargs

def build_tool_messages(query, search_dir):
    return [{"role": "system", "content": f"""你是文档检索专家。

可用文件列表：
{', '.join(available_files(search_dir))}

工作流程：
1. 分析用户问题，提取关键词和目标文件
2. 调用 search_documents 工具搜索
3. 基于搜索结果生成答案

注意：
- broad_keywords: 1-2个核心关键词
- exact_keywords: 1个最特殊、最关键的元关键词（可以是 broad_keywords 中较特殊的一个）
- target_files: 从可用文件列表中选择1-3个最可能包含答案的文件
- 必须先调用工具再回答"""}, {"role": "user", "content": query}]

def build_answer_messages(query, tool_results):
    return [{"role": "system", "content": f"""你是根据文档总结回答问题的专家
根据文档已经检索到的信息为{tool_results}，根据信息极短思考，清晰简要回答问题，并给出明确依据,未提及的不要回答。
"""}, {"role": "user", "content": query}]

def chunk_file(filepath, chunk_size=512, overlap=50):
    try:
        sentences = split_sentences(read_text(filepath))
    except Exception as e:
        print(f"⚠️ 读取文件 {filepath} 失败: {e}")
        return []

    chunks, buf, buf_len, start_pos = [], [], 0, 0

    def flush():
        nonlocal buf, buf_len, start_pos
        text = ''.join(buf)
        chunks.append({'content': text, 'file': filepath, 'start_pos': start_pos, 'type': 'chunk'})
        keep, keep_len = [], 0
        for sent in reversed(buf):
            if keep_len + len(sent) > overlap:
                break
            keep.insert(0, sent)
            keep_len += len(sent)
        buf, buf_len = keep, keep_len
        start_pos += len(text) - keep_len

    for sent in sentences:
        if buf and buf_len + len(sent) > chunk_size:
            flush()
        buf.append(sent)
        buf_len += len(sent)
    if buf:
        chunks.append({'content': ''.join(buf), 'file': filepath, 'start_pos': start_pos, 'type': 'chunk'})
    return chunks

def parse_rg_line(line):
    parts = line.split(':')
    idx = next((i for i in range(1, len(parts)) if parts[i].isdigit()), None)
    if idx is None or idx + 1 >= len(parts):
        return None
    return {'file': ':'.join(parts[:idx]), 'line_num': int(parts[idx]), 'content': ':'.join(parts[idx + 1:]), 'type': 'rg'}

def run_rg(keyword, search_dir):
    result = subprocess.run(['rg', '-n', '-i', keyword, search_dir], capture_output=True, text=True, encoding='utf-8', errors='ignore')
    if result.returncode != 0:
        return []
    return [m for line in result.stdout.splitlines() if (m := parse_rg_line(line))]

def extract_context(filepath, line_num, context_lines=0):
    try:
        lines = read_text(filepath).splitlines(True)
        start = max(0, line_num - context_lines - 1)
        end = min(len(lines), line_num + context_lines)
        return ''.join(f"{i + 1}: {lines[i]}" for i in range(start, end))
    except Exception:
        return ''

def attach_bm25_scores(candidates, query):
    if not candidates:
        return candidates
    docs, scores = BM25([x['content'] for x in candidates]).get_top_n(query, len(candidates))
    score_map = {int(doc_idx): score for (doc_idx, _), score in zip(docs, scores)}
    for i, item in enumerate(candidates):
        item.update(bm25_score=score_map.get(i, 0.0), score=score_map.get(i, 0.0))
    return candidates

def keyword_bonus(content, broad_keywords, exact_keywords):
    text = content.lower()
    broad_hits = sum(kw.lower() in text for kw in broad_keywords)
    exact_hit = exact_keywords and any(kw.lower() in text for kw in exact_keywords)
    return (0.5 if broad_hits == 2 else 1.0 if broad_hits >= 3 else 0.0) + (1.0 if exact_hit else 0.0)

def dedupe(candidates):
    rg_texts = {x['content'].strip().lower() for x in candidates if x['type'] == 'rg'}
    seen, result = set(), []
    for item in candidates:
        key, content = candidate_key(item), item['content'].strip().lower()
        if item['type'] == 'rg':
            if key not in seen:
                seen.add(key)
                result.append(item)
        elif not any(rg in content or content in rg for rg in rg_texts):
            result.append(item)
    return result

def merge_ranked(top_k, *groups):
    merged, selected = [], set()
    for candidates, source, limit in groups:
        added = 0
        for item in candidates:
            key = candidate_key(item)
            if key in selected:
                item.setdefault('selected_by', [])
                if source not in item['selected_by']:
                    item['selected_by'].append(source)
                continue
            item['selected_by'] = [source]
            merged.append(item)
            selected.add(key)
            added += 1
            if limit and added >= limit:
                break
    return sorted(merged[:top_k], key=lambda x: ({'bm25', 'boosted'} <= set(x.get('selected_by', [])), x['boosted_score'], x['bm25_score']), reverse=True)

def format_result(item, context_lines):
    if item['type'] == 'rg':
        return f"--- {os.path.basename(item['file'])}:行{item['line_num']} [RG] ---\n{extract_context(item['file'], item['line_num'], context_lines)}\n"
    return f"--- {os.path.basename(item['file'])}:位置{item['start_pos']} [CHUNK] ---\n{item['content']}\n"

# ================= 搜索工具实现 =================
def search_documents(query, broad_keywords, exact_keywords=None, target_files=None, search_dir='./specs', top_k=15, context_lines=0, stream=False):
    def core():
        exact, targets = exact_keywords or [], target_files or []
        yield from emit(stream, f"\n🔍 原始问题: {query}\n🔍 关键词: 宽泛={broad_keywords}, 精确={exact}\n🔍 目标文件: {targets}")

        matches = []
        for kw in broad_keywords + exact:
            try:
                matches.extend(run_rg(kw, search_dir))
            except Exception as e:
                yield from emit(stream, f"⚠️ 搜索 '{kw}' 出错: {e}")
        yield from emit(stream, f"✅ RG 找到 {len(matches)} 个匹配行")

        chunks = []
        for name in targets:
            path = find_file(search_dir, name)
            if not path:
                continue
            file_chunks = chunk_file(path)
            chunks.extend(file_chunks)
            yield from emit(stream, f"✅ 文件 {name} 切分为 {len(file_chunks)} 个 chunks")
        yield from emit(stream, f"✅ 总共生成 {len(chunks)} 个文件 chunks")

        candidates = matches + chunks
        if not candidates:
            yield '未找到匹配内容'
            return
        yield from emit(stream, f"✅ 总候选内容: {len(candidates)} 条")

        attach_bm25_scores(candidates, query)
        for item in candidates:
            item['keyword_bonus'] = keyword_bonus(item['content'], broad_keywords, exact)
            item['boosted_score'] = item['bm25_score'] + item['keyword_bonus']

        bm25_sorted = sorted(candidates, key=lambda x: x['bm25_score'], reverse=True)
        boosted_sorted = sorted(candidates, key=lambda x: x['boosted_score'], reverse=True)
        keys = {candidate_key(x) for x in dedupe(bm25_sorted)}
        bm25_dedup = [x for x in bm25_sorted if candidate_key(x) in keys]
        boosted_dedup = [x for x in boosted_sorted if candidate_key(x) in keys]
        bm25_n, boost_n = min(10, top_k), min(5, top_k)
        top_items = merge_ranked(top_k, (bm25_dedup, 'bm25', bm25_n), (boosted_dedup, 'boosted', boost_n), (bm25_dedup, 'bm25_fill', None))

        yield from emit(stream, f"📊 去重后: {len(keys)} 条 (RG优先), BM25取前{bm25_n} + 修正取前{boost_n} -> Top-{len(top_items)}")
        result = '\n'.join(format_result(x, context_lines) for x in top_items)
        yield from emit(stream, f"\n✅ 返回内容总长度: {len(result)} 字符")
        yield result

    return core() if stream else ''.join(filter(None, list(core())))

# ================= Agent 主循环(实际为两步) =================
def run_search(query, search_dir='./texts', top_k=15, context_lines=10, stream=False):
    def call_dicts(resp):
        return [{'id': c.id, 'function': {'name': c.function.name, 'arguments': c.function.arguments}} for c in (resp.choices[0].message.tool_calls or [])]

    def core():
        yield from emit(stream, f"\n{'=' * 60}\n🚀 问题: {query}\n{'=' * 60}")
        yield from emit(stream, "\n[第1轮] LLM 分析问题并调用工具...")

        kw = build_chat_kwargs(build_tool_messages(query, search_dir), stream=stream, tools=TOOLS, temperature=1)
        if stream:
            buf = {}
            yield from emit(True, client.chat.completions.create(**kw), buf, label_answer=False)
            calls = [buf[i] for i in sorted(buf)]
        else:
            resp = client.chat.completions.create(**kw)
            if not resp or not resp.choices:
                yield from emit(stream, "⚠️ API 返回空响应")
                return
            calls = call_dicts(resp)

        if not calls:
            yield from emit(stream, "⚠️ LLM 未生成关键词，结束")
            return

        tool_msgs = []
        for call in (c for c in calls if c['function']['name'] == 'search_documents'):
            args = json.loads(call['function']['arguments'] or '{}')
            # yield from emit(stream, f"📝 工具参数: {args}")
            tool_msgs.append({
                'role': 'tool', 'tool_call_id': call['id'],
                'content': search_documents(
                    args.get('query', query), args.get('broad_keywords', []), args.get('exact_keywords', []),
                    args.get('target_files', []), search_dir=search_dir, top_k=top_k, context_lines=context_lines, stream=False
                ),
            })

        yield from emit(stream, "\n[第2轮] LLM 生成最终答案...")
        final = client.chat.completions.create(**build_chat_kwargs(build_answer_messages(query, tool_msgs), stream=stream, temperature=1))
        if stream:
            yield from emit(True, final)
        elif final and final.choices:
            answer = final.choices[0].message.content
            yield from emit(stream, "⚠️ LLM 返回空答案！" if not answer else f"\n{'=' * 60}\n✅ 最终答案:\n{'=' * 60}\n{answer}\n{'=' * 60}")
        else:
            yield from emit(stream, "⚠️ API 返回空响应")

    return core() if stream else list(core())

# ================= 主程序 =================
if __name__ == '__main__':
    for chunk in run_search(query='跟扩展基础的宽高比有关的规定', search_dir='./specs', top_k=15, context_lines=0, stream=True):
        print(chunk, end='', flush=True)
