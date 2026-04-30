"""
简化版智能检索系统 - 基于 LLM Function Call + BM25 排序
压缩版本，使用语法糖和函数式编程减少代码行数
"""

import os, json, subprocess, re
from openai import OpenAI
from dotenv import load_dotenv
from bm25_module import BM25

load_dotenv()

# ===== 模型与客户端初始化 =====
MODEL_CONFIG = {
    1: {'base_url': 'https://api.moonshot.cn/v1', 'api_key': 'kimi_key', 'model_name': 'kimi-k2.5', 'thinking': 'kimi'},
    2: {'base_url': 'https://integrate.api.nvidia.com/v1', 'api_key': 'nvidia_key', 'model_name': 'minimaxai/minimax-m2.5'},
    3: {'base_url': 'https://api-inference.modelscope.cn/v1', 'api_key': 'modelscope_key', 'model_name': 'Qwen/Qwen3-235B-A22B-Instruct-2507', 'thinking': 'qwen'},
    4: {'base_url': 'https://api-inference.modelscope.cn/v1', 'api_key': 'modelscope_key', 'model_name': 'Qwen/Qwen3.5-27B', 'thinking': 'qwen'},
    5: {'base_url': 'http://localhost:11434/v1', 'api_key': 'ollama', 'model_name': 'gemma4:e4b'},
    6: {'base_url': 'https://ollama.com/v1', 'api_key': 'ollama_key', 'model_name': 'kimi-k2.5:cloud'},
}

MODEL_NUM = 3
THINKING_ENABLED = True
config = MODEL_CONFIG[MODEL_NUM]
client = OpenAI(base_url=config['base_url'], api_key=os.getenv(config['api_key']))
model_name = config['model_name']

print(f"🤖 使用模型: {model_name}，序号{MODEL_NUM}")

def _thinking_caps(cfg=None):
    cfg = cfg or config
    kind, forced = cfg.get('thinking'), 'thinking' in cfg['model_name'].lower()
    return {'supported': bool(kind), 'can_disable': bool(kind) and not forced, 'forced': forced, 'kind': kind}

def build_chat_kwargs(messages, stream=False, tools=None, temperature=1):
    caps = _thinking_caps()
    # Kimi 关闭思考模式时 temperature 必须为 0.6
    if caps['kind'] == 'kimi' and caps['can_disable'] and not THINKING_ENABLED:
        temperature = 0.6
    
    kw = {'model': model_name, 'messages': messages, 'temperature': temperature, 'stream': stream}
    if tools:
        kw.update(tools=tools, tool_choice="auto")
    
    extra_body = (
        {'thinking': {'type': 'disabled'}} if caps['kind'] == 'kimi' and caps['can_disable'] and not THINKING_ENABLED else
        {'enable_thinking': THINKING_ENABLED} if caps['kind'] == 'qwen' and caps['can_disable'] else
        None
    )
    if extra_body:
        kw['extra_body'] = extra_body
    return kw

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
                "target_files": {"type": "array", "items": {"type": "string"}, "description": "最可能包含答案的文件名列表（从可用文件列表中选择）"}
            },
            "required": ["query", "broad_keywords", "target_files"]
        }
    }
}]

# 辅助函数：提示词构造、流式输出、切块、BM25 打分、rg 解析、去重融合、结果格式化
# ================= 辅助函数 =================
get_available_files = lambda search_dir: sorted([f for root, _, files in os.walk(search_dir) for f in files if f.endswith(('.txt', '.md'))])
cut_by_punctuation = lambda text: [s.strip() for s in re.findall(r'[^。！？.!?]+[。！？.!?]?', text.strip()) if s.strip()]
get_candidate_key = lambda item: ('rg', item['file'], item['line_num']) if item['type'] == 'rg' else ('chunk', item['file'], item['start_pos'])

def build_tool_messages(query: str, search_dir: str):
    return [{"role": "system", "content": f"""你是文档检索专家。

可用文件列表：
{', '.join(get_available_files(search_dir))}

工作流程：
1. 分析用户问题，提取关键词和目标文件
2. 调用 search_documents 工具搜索
3. 基于搜索结果生成答案

注意：
- broad_keywords: 1-2个核心关键词
- exact_keywords: 1个最特殊、最关键的元关键词（可以是 broad_keywords 中较特殊的一个）
- target_files: 从可用文件列表中选择1-3个最可能包含答案的文件
- 必须先调用工具再回答"""}, {"role": "user", "content": query}]

def build_answer_messages(query: str, tool_results: list):
    return [{"role": "system", "content": f"""你是根据文档总结回答问题的专家
根据文档已经检索到的信息为{tool_results}，根据信息回答问题，并给出明确依据,未提及的不要回答。
"""}, {"role": "user", "content": query}]

def _emit(stream, msg):
    if stream: yield msg
    else: print(msg)

def chunk_file(filepath: str, chunk_size: int = 512, overlap: int = 50) -> list:
    """将文件切分成 chunks（按句子边界切分）"""
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            sentences = cut_by_punctuation(f.read())
    except Exception as e:
        print(f"⚠️ 读取文件 {filepath} 失败: {e}")
        return []
    
    chunks, current_chunk, current_length, start_pos = [], [], 0, 0
    def flush_chunk():
        nonlocal current_chunk, current_length, start_pos
        chunk_text = ''.join(current_chunk)
        chunks.append({'content': chunk_text, 'file': filepath, 'start_pos': start_pos, 'type': 'chunk'})
        overlap_sents, overlap_length = [], 0
        for s in reversed(current_chunk):
            if overlap_length + len(s) > overlap:
                break
            overlap_sents.insert(0, s)
            overlap_length += len(s)
        current_chunk, current_length = overlap_sents, overlap_length
        start_pos += len(chunk_text) - overlap_length
    for sent in sentences:
        sent_len = len(sent)
        if current_length + sent_len > chunk_size and current_chunk:
            flush_chunk()
        current_chunk.append(sent)
        current_length += sent_len
    if current_chunk:
        chunks.append({'content': ''.join(current_chunk), 'file': filepath, 'start_pos': start_pos, 'type': 'chunk'})
    return chunks

def attach_bm25_scores(candidates: list, query: str) -> list:
    """对候选内容进行 BM25 打分"""
    if not candidates:
        return candidates
    
    corpus = [item['content'] for item in candidates]
    docs, scores = BM25(corpus).get_top_n(query, len(corpus))
    score_by_index = {int(doc_idx): score for (doc_idx, _), score in zip(docs, scores)}
    for idx, item in enumerate(candidates):
        item.update(bm25_score=score_by_index.get(idx, 0.0), score=score_by_index.get(idx, 0.0))
    return candidates

def _extract_context(filepath: str, line_num: int, context_lines: int = 0) -> str:
    """提取上下文"""
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        start, end = max(0, line_num - context_lines - 1), min(len(lines), line_num + context_lines)
        return "".join(f"{i+1}: {lines[i]}" for i in range(start, end))
    except:
        return ""

def _run_rg(keyword: str, search_dir: str) -> list:
    result = subprocess.run(['rg', '-n', '-i', keyword, search_dir], capture_output=True, text=True, encoding='utf-8', errors='ignore')
    return [
        {'file': parts[0], 'line_num': int(parts[1]), 'content': parts[2], 'type': 'rg'}
        for line in result.stdout.strip().split('\n') if ':' in line
        for parts in [line.split(':', 2)] if len(parts) >= 3
    ] if result.returncode == 0 else []

def _find_target_path(search_dir: str, filename: str):
    return next((os.path.join(root, filename) for root, _, files in os.walk(search_dir) if filename in files), None)

def _keyword_bonus(content: str, broad_keywords: list, exact_keywords: list) -> float:
    content_lower = content.lower()
    broad_hits = sum(kw.lower() in content_lower for kw in broad_keywords)
    return (0.5 if broad_hits == 2 else 1.0 if broad_hits >= 3 else 0.0) + (1.0 if exact_keywords and any(kw.lower() in content_lower for kw in exact_keywords) else 0.0)

def _dedupe_candidates(candidates: list) -> list:
    rg_contents = {item['content'].strip().lower() for item in candidates if item['type'] == 'rg'}
    seen_rg, deduplicated = set(), []
    for item in candidates:
        key = get_candidate_key(item)
        content = item['content'].strip().lower()
        if item['type'] == 'rg' and key not in seen_rg:
            seen_rg.add(key)
            deduplicated.append(item)
        elif item['type'] == 'chunk' and not any(rg in content or content in rg for rg in rg_contents):
            deduplicated.append(item)
    return deduplicated

def _merge_ranked_lists(top_k: int, *groups) -> list:
    merged, selected = [], set()
    for candidates, source, limit in groups:
        added = 0
        for item in candidates:
            key = get_candidate_key(item)
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
    return sorted(
        merged[:top_k],
        key=lambda item: (
            {'bm25', 'boosted'} <= set(item.get('selected_by', [])),
            item['boosted_score'],
            item['bm25_score'],
        ),
        reverse=True,
    )

def _format_result(item: dict, context_lines: int) -> str:
    return (
        f"--- {os.path.basename(item['file'])}:行{item['line_num']} [RG] ---\n{_extract_context(item['file'], item['line_num'], context_lines)}\n"
        if item['type'] == 'rg' else
        f"--- {os.path.basename(item['file'])}:位置{item['start_pos']} [CHUNK] ---\n{item['content']}\n"
    )

# ================= 搜索工具实现 =================
def search_documents(query: str, broad_keywords: list, exact_keywords: list = None, 
                     target_files: list = None, search_dir: str = "./specs", 
                     top_k: int = 15, context_lines: int = 0, stream: bool = False):
    """四步走：RG搜索 -> 文件切块 -> BM25排序 -> 添加上下文"""
    def _core():
        _exact_keywords, _target_files = exact_keywords or [], target_files or []
        yield from _emit(stream, f"\n🔍 原始问题: {query}\n🔍 关键词: 宽泛={broad_keywords}, 精确={_exact_keywords}\n🔍 目标文件: {_target_files}")
        matching_lines = []
        for kw in broad_keywords + _exact_keywords:
            try:
                matching_lines.extend(_run_rg(kw, search_dir))
            except Exception as e:
                yield from _emit(stream, f"⚠️ 搜索 '{kw}' 出错: {e}")
        yield from _emit(stream, f"✅ RG 找到 {len(matching_lines)} 个匹配行")
        file_chunks = []
        for filename in _target_files:
            if filepath := _find_target_path(search_dir, filename):
                chunks = chunk_file(filepath, chunk_size=512, overlap=50)
                file_chunks.extend(chunks)
                yield from _emit(stream, f"✅ 文件 {filename} 切分为 {len(chunks)} 个 chunks")
        yield from _emit(stream, f"✅ 总共生成 {len(file_chunks)} 个文件 chunks")
        all_candidates = matching_lines + file_chunks
        if not all_candidates:
            yield "未找到匹配内容"
            return
        yield from _emit(stream, f"✅ 总候选内容: {len(all_candidates)} 条")
        attach_bm25_scores(all_candidates, query)
        for item in all_candidates:
            item['keyword_bonus'] = _keyword_bonus(item['content'], broad_keywords, _exact_keywords)
            item['boosted_score'] = item['bm25_score'] + item['keyword_bonus']
        bm25_sorted = sorted(all_candidates, key=lambda x: x['bm25_score'], reverse=True)
        boosted_sorted = sorted(all_candidates, key=lambda x: x['boosted_score'], reverse=True)
        dedup_keys = {get_candidate_key(item) for item in _dedupe_candidates(bm25_sorted)}
        bm25_dedup = [item for item in bm25_sorted if get_candidate_key(item) in dedup_keys]
        boosted_dedup = [item for item in boosted_sorted if get_candidate_key(item) in dedup_keys]
        bm25_pick_count, boosted_pick_count = min(10, top_k), min(5, top_k)
        top_items = _merge_ranked_lists(
            top_k,
            (bm25_dedup, 'bm25', bm25_pick_count),
            (boosted_dedup, 'boosted', boosted_pick_count),
            (bm25_dedup, 'bm25_fill', None),
        )
        yield from _emit(stream, f"📊 去重后: {len(dedup_keys)} 条 (RG优先), BM25取前{bm25_pick_count} + 修正取前{boosted_pick_count} -> Top-{len(top_items)}")
        results = [_format_result(item, context_lines) for item in top_items]
        final_result = "\n".join(results)
        yield from _emit(stream, f"\n✅ 返回内容总长度: {len(final_result)} 字符")
        yield final_result
    return _core() if stream else ''.join(filter(None, list(_core())))

# ================= Agent 主循环 =================
def run_search(query: str, search_dir: str = "./texts", top_k: int = 10, context_lines: int = 10, stream: bool = False):
    """两步 Agent：调用工具 -> 生成答案"""
    def _core():
        yield from _emit(stream, f"\n{'='*60}\n🚀 问题: {query}\n{'='*60}")
        messages = build_tool_messages(query, search_dir)
        yield from _emit(stream, "\n[第1轮] LLM 分析问题并调用工具...")
        response = client.chat.completions.create(**build_chat_kwargs(messages, tools=TOOLS, temperature=1))
        if not response or not response.choices:
            yield from _emit(stream, "⚠️ API 返回空响应")
            return
        msg_obj = response.choices[0].message
        if not msg_obj.tool_calls:
            yield from _emit(stream, "⚠️ LLM 未生成关键词，结束")
            return
        msg2 = []
        for tool_call in (call for call in msg_obj.tool_calls if call.function.name == "search_documents"):
            args = json.loads(tool_call.function.arguments)
            yield from _emit(stream, f"📝 工具参数: {args}")
            msg2.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": search_documents(
                    args.get("query", query),
                    args.get("broad_keywords", []),
                    args.get("exact_keywords", []),
                    args.get("target_files", []),
                    search_dir=search_dir,
                    top_k=top_k,
                    context_lines=context_lines,
                    stream=False
                )
            })
        yield from _emit(stream, "\n[第2轮] LLM 生成最终答案...")
        messages_2 = build_answer_messages(query, msg2)
        final_response = client.chat.completions.create(**build_chat_kwargs(messages_2, stream=stream, temperature=1))
        if stream:
            for chunk in final_response:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        else:
            if final_response and final_response.choices:
                answer = final_response.choices[0].message.content
                yield from _emit(stream, "⚠️ LLM 返回空答案！" if not answer else f"\n{'='*60}\n✅ 最终答案:\n{'='*60}\n{answer}\n{'='*60}")
            else:
                yield from _emit(stream, "⚠️ API 返回空响应")
    return _core() if stream else list(_core())

# ================= 主程序 =================
if __name__ == "__main__":
    # 非流式模式（默认）
    # run_search(query="独立基础的高宽比", search_dir="./specs", top_k=10, context_lines=0)
    
    # 流式模式示例
    for chunk in run_search(query="独立基础的高宽比", search_dir="./specs", top_k=10, context_lines=0, stream=True):
        print(chunk, end='', flush=True)
