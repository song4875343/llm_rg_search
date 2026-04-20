"""
简化版智能检索系统 - 基于 LLM Function Call + BM25 排序
压缩版本，使用语法糖和函数式编程减少代码行数
"""

import os, json, subprocess, re
from openai import OpenAI
from dotenv import load_dotenv
from bm25_module import BM25

load_dotenv()

# ================= 配置 =================
MODEL_CONFIG = {
    1: {'base_url': 'https://api.moonshot.cn/v1', 'api_key': 'kimi_key', 'model_name': 'kimi-k2.5'},
    2: {'base_url': 'https://integrate.api.nvidia.com/v1', 'api_key': 'nvidia_key', 'model_name': 'minimaxai/minimax-m2.5'},
    3: {'base_url': 'https://api-inference.modelscope.cn/v1', 'api_key': 'modelscope_key', 'model_name': 'Qwen/Qwen3-235B-A22B-Instruct-2507'},
    4: {'base_url': 'https://api-inference.modelscope.cn/v1', 'api_key': 'modelscope_key', 'model_name': 'Qwen/Qwen3.5-27B'},
    5: {'base_url': 'http://localhost:11434/v1', 'api_key': 'ollama', 'model_name': 'gemma4:e4b'},
    6: {'base_url': 'https://ollama.com/v1', 'api_key': 'ollama_key', 'model_name': 'kimi-k2.5:cloud'},
}

MODEL_NUM = 3
config = MODEL_CONFIG[MODEL_NUM]
client = OpenAI(base_url=config['base_url'], api_key=os.getenv(config['api_key']))
model_name = config['model_name']

print(f"🤖 使用模型: {model_name}，序号{MODEL_NUM}")

# ================= 工具定义 =================
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

# ================= 辅助函数 =================
get_available_files = lambda search_dir: sorted([f for root, _, files in os.walk(search_dir) for f in files if f.endswith(('.txt', '.md'))])

cut_by_punctuation = lambda text: [s.strip() for s in re.findall(r'[^。！？.!?]+[。！？.!?]?', text.strip()) if s.strip()]

def chunk_file(filepath: str, chunk_size: int = 512, overlap: int = 50) -> list:
    """将文件切分成 chunks（按句子边界切分）"""
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            sentences = cut_by_punctuation(f.read())
    except Exception as e:
        print(f"⚠️ 读取文件 {filepath} 失败: {e}")
        return []
    
    chunks, current_chunk, current_length, start_pos = [], [], 0, 0
    
    for sent in sentences:
        sent_len = len(sent)
        if current_length + sent_len > chunk_size and current_chunk:
            chunk_text = ''.join(current_chunk)
            chunks.append({'content': chunk_text, 'file': filepath, 'start_pos': start_pos, 'type': 'chunk'})
            
            # 计算重叠部分
            overlap_sents, overlap_length = [], 0
            for s in reversed(current_chunk):
                if overlap_length + len(s) <= overlap:
                    overlap_sents.insert(0, s)
                    overlap_length += len(s)
                else:
                    break
            
            current_chunk, current_length = overlap_sents, overlap_length
            start_pos += len(chunk_text) - overlap_length
        
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
        item['bm25_score'] = item['score'] = score_by_index.get(idx, 0.0)
    
    return candidates

get_candidate_key = lambda item: ('rg', item['file'], item['line_num']) if item['type'] == 'rg' else ('chunk', item['file'], item['start_pos'])

def _extract_context(filepath: str, line_num: int, context_lines: int = 0) -> str:
    """提取上下文"""
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        start, end = max(0, line_num - context_lines - 1), min(len(lines), line_num + context_lines)
        return "".join(f"{i+1}: {lines[i]}" for i in range(start, end))
    except:
        return ""

# ================= 搜索工具实现 =================
def search_documents(query: str, broad_keywords: list, exact_keywords: list = None, 
                     target_files: list = None, search_dir: str = "./specs", 
                     top_k: int = 15, context_lines: int = 0) -> str:
    """四步走：RG搜索 -> 文件切块 -> BM25排序 -> 添加上下文"""
    exact_keywords, target_files = exact_keywords or [], target_files or []
    all_keywords = broad_keywords + exact_keywords
    
    print(f"\n🔍 原始问题: {query}\n🔍 关键词: 宽泛={broad_keywords}, 精确={exact_keywords}\n🔍 目标文件: {target_files}")
    
    # Step 1: RG 搜索收集匹配行
    matching_lines = []
    for kw in all_keywords:
        try:
            result = subprocess.run(['rg', '-n', '-i', kw, search_dir], capture_output=True, text=True, encoding='utf-8', errors='ignore')
            if result.returncode == 0:
                matching_lines.extend([
                    {'file': parts[0], 'line_num': int(parts[1]), 'content': parts[2], 'type': 'rg'}
                    for line in result.stdout.strip().split('\n') if ':' in line
                    for parts in [line.split(':', 2)] if len(parts) >= 3
                ])
        except Exception as e:
            print(f"⚠️ 搜索 '{kw}' 出错: {e}")
    
    print(f"✅ RG 找到 {len(matching_lines)} 个匹配行")
    
    # Step 2: 读取目标文件并切块
    file_chunks = []
    for filename in target_files:
        for root, _, filenames in os.walk(search_dir):
            if filename in filenames:
                chunks = chunk_file(os.path.join(root, filename), chunk_size=512, overlap=50)
                file_chunks.extend(chunks)
                print(f"✅ 文件 {filename} 切分为 {len(chunks)} 个 chunks")
                break
    
    print(f"✅ 总共生成 {len(file_chunks)} 个文件 chunks")
    
    all_candidates = matching_lines + file_chunks
    if not all_candidates:
        return "未找到匹配内容"
    
    print(f"✅ 总候选内容: {len(all_candidates)} 条")
    
    # Step 3: BM25 排序
    attach_bm25_scores(all_candidates, query)
    
    # 计算关键词加分
    for item in all_candidates:
        content_lower = item['content'].lower()
        broad_hit_count = sum(1 for kw in broad_keywords if kw.lower() in content_lower)
        keyword_bonus = 0.5 if broad_hit_count == 2 else (1.0 if broad_hit_count >= 3 else 0.0)
        
        if exact_keywords and any(kw.lower() in content_lower for kw in exact_keywords):
            keyword_bonus += 1.0
        
        item['keyword_bonus'] = keyword_bonus
        item['boosted_score'] = item['bm25_score'] + keyword_bonus
    
    bm25_sorted = sorted(all_candidates, key=lambda x: x['bm25_score'], reverse=True)
    boosted_sorted = sorted(all_candidates, key=lambda x: x['boosted_score'], reverse=True)
    
    # 去重策略：RG 行优先
    rg_contents = {item['content'].strip().lower() for item in bm25_sorted if item['type'] == 'rg'}
    seen_rg, deduplicated = set(), []
    
    for item in bm25_sorted:
        if item['type'] == 'rg':
            key = get_candidate_key(item)
            if key not in seen_rg:
                deduplicated.append(item)
                seen_rg.add(key)
        else:
            chunk_lower = item['content'].strip().lower()
            if not any(rg_content in chunk_lower or chunk_lower in rg_content for rg_content in rg_contents):
                deduplicated.append(item)
    
    dedup_map = {get_candidate_key(item): item for item in deduplicated}
    bm25_dedup = [item for item in bm25_sorted if get_candidate_key(item) in dedup_map]
    boosted_dedup = [item for item in boosted_sorted if get_candidate_key(item) in dedup_map]
    
    # 合并结果
    bm25_pick_count, boosted_pick_count = min(10, top_k), min(5, top_k)
    merged_items, selected_keys = [], set()
    
    def add_candidates(candidates: list, source_name: str, limit: int = None):
        added = 0
        for candidate in candidates:
            key = get_candidate_key(candidate)
            if key in selected_keys:
                if source_name not in candidate.get('selected_by', []):
                    candidate.setdefault('selected_by', []).append(source_name)
                continue
            candidate['selected_by'] = [source_name]
            merged_items.append(candidate)
            selected_keys.add(key)
            added += 1
            if limit is not None and added >= limit:
                break
    
    add_candidates(bm25_dedup, 'bm25', bm25_pick_count)
    add_candidates(boosted_dedup, 'boosted', boosted_pick_count)
    add_candidates(bm25_dedup, 'bm25_fill')
    
    top_items = merged_items[:top_k]
    top_items.sort(key=lambda item: (
        1 if ('bm25' in item.get('selected_by', []) and 'boosted' in item.get('selected_by', [])) else 0,
        item['boosted_score'],
        item['bm25_score']
    ), reverse=True)
    
    print(f"📊 去重后: {len(deduplicated)} 条 (RG优先), BM25取前{bm25_pick_count} + 修正取前{boosted_pick_count} -> Top-{len(top_items)}")
    
    # Step 4: 添加上下文或直接返回 chunk
    results = [
        f"--- {os.path.basename(item['file'])}:行{item['line_num']} [RG] ---\n{_extract_context(item['file'], item['line_num'], context_lines)}\n"
        if item['type'] == 'rg' else
        f"--- {os.path.basename(item['file'])}:位置{item['start_pos']} [CHUNK] ---\n{item['content']}\n"
        for item in top_items
    ]
    
    final_result = "\n".join(results)
    print(f"\n✅ 返回内容总长度: {len(final_result)} 字符")
    return final_result

# ================= Agent 主循环 =================
def run_search(query: str, search_dir: str = "./texts", top_k: int = 10, context_lines: int = 10):
    """两步 Agent：调用工具 -> 生成答案"""
    print(f"\n{'='*60}\n🚀 问题: {query}\n{'='*60}")
    
    available_files = get_available_files(search_dir)
    system_prompt = f"""你是文档检索专家。

可用文件列表：
{', '.join(available_files)}

工作流程：
1. 分析用户问题，提取关键词和目标文件
2. 调用 search_documents 工具搜索
3. 基于搜索结果生成答案

注意：
- broad_keywords: 1-2个核心关键词
- exact_keywords: 1个最特殊、最关键的元关键词（可以是 broad_keywords 中较特殊的一个）
- target_files: 从可用文件列表中选择1-3个最可能包含答案的文件
- 必须先调用工具再回答"""
    
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": query}]
    
    # 第一轮：LLM 调用工具
    print("\n[第1轮] LLM 分析问题并调用工具...")
    response = client.chat.completions.create(model=model_name, messages=messages, tools=TOOLS, tool_choice="auto", temperature=1)
    
    msg = response.choices[0].message
    if not msg.tool_calls:
        print("⚠️ LLM 未生成关键词，结束")
        return
    
    msg2 = []
    for tool_call in msg.tool_calls:
        if tool_call.function.name == "search_documents":
            args = json.loads(tool_call.function.arguments)
            print(f"📝 工具参数: {args}")
            
            result = search_documents(
                args.get("query", query),
                args.get("broad_keywords", []),
                args.get("exact_keywords", []),
                args.get("target_files", []),
                search_dir=search_dir,
                top_k=top_k,
                context_lines=context_lines
            )
            
            msg2.append({"role": "tool", "tool_call_id": tool_call.id, "content": result})
    
    # 第二轮：LLM 生成答案
    print("\n[第2轮] LLM 生成最终答案...")
    system_prompt_2 = f"""你是根据文档总结回答问题的专家
    根据文档已经检索到的信息为{msg2}，根据信息回答问题，并给出明确依据,未提及的不要回答。
    """
    
    messages_2 = [{"role": "system", "content": system_prompt_2}, {"role": "user", "content": query}]
    final_response = client.chat.completions.create(model=model_name, messages=messages_2, temperature=1)
    
    answer = final_response.choices[0].message.content
    if not answer or len(answer) == 0:
        print(f"⚠️ LLM 返回空答案！\n📝 Response 对象: {final_response}")
    
    print(f"\n{'='*60}\n✅ 最终答案:\n{'='*60}\n{answer}\n{'='*60}")

# ================= 主程序 =================
if __name__ == "__main__":
    run_search(query="独立基础的高宽比", search_dir="./specs", top_k=10, context_lines=0)
