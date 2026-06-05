"""
简化版智能检索系统 - 基于 LLM Function Call + BM25 排序

流程：
1. LLM 生成关键词 + RG 搜索（Function Call）
2. LLM 根据搜索结果生成最终答案
创新点：1.宽泛搜索词限定小范围bm25排序，提升了bm25的准确率和即时性
       2.精确关键词额外加分，进一步修正了bm25的准确率
       3.宽泛多关键词额外加分，也能一定程度修正bm25的准确率
难以克服的缺点：
    1.由于是fast版本不希望经过多轮关键词搜索，在没有预先了解知识库的情况如果用户有错别字，尤其是关键信息上
    就会导致搜索失败，但是在多轮agent-search上就能避免，这是形式上的。
v2 上面是v1的做法
    v2解决了错别字的问题，之前的问题实际上存在两个问题，
    一rg对错误的关键词确实没有办法，经常是0命中
    二量太少是bm25会有漂移情况，结果很不稳定也不真实
    三v2 让llm提供了最可能出现在哪写文件中，然后对这些文件切片，回答完美避免了简写错写关键词的问题，如果碰巧有些关键词再写对，准确率比原来高，真的没写对
    一般也能解决问题，也就是不用关键词，用判断文件再bm25大部分问题也能解决了。
    目前是双路返回纯bm25排序返回10个，关键词加分返回5个，后续我会考虑关键词加分到底有意义吗
    30b的模型总体很不错，但是当问题问的有瑕疵，比如宽高比写成了高宽比之类，它会很教条的认为没有这个东西，但是没写错的化它表现得就很好，其次速度又快，性能又好的就是120b的，不管有没写错agent总是最稳的。
"""

import os
import json
import subprocess
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

# ================= 初始化客户端 =================
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
                "query": {
                    "type": "string",
                    "description": "用户的原始问题，用于 BM25 相关性计算"
                },
                "broad_keywords": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "宽泛关键词列表，用于初步召回"
                },
                "exact_keywords": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "1元素的精确关键词列表，用于评定返回内容的命中可能性"
                },
                "target_files": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "最可能包含答案的文件名列表（从可用文件列表中选择）"
                }
            },
            "required": ["query", "broad_keywords", "target_files"]
        }
    }
}]
# ================= 辅助函数 =================
def get_available_files(search_dir: str) -> list:
    """获取目录下所有 .txt 和 .md 文件名（不含路径）"""
    files = []
    for root, _, filenames in os.walk(search_dir):
        for filename in filenames:
            if filename.endswith(('.txt', '.md')):
                files.append(filename)
    return sorted(files)

def cut_by_punctuation(paragraph: str) -> list[str]:
    """中文句子切分"""
    import re
    sents = re.findall(r'[^。！？.!?]+[。！？.!?]?', paragraph.strip())
    return [s.strip() for s in sents if s.strip()]

def chunk_file(filepath: str, chunk_size: int = 512, overlap: int = 50) -> list:
    """
    将文件切分成 chunks（按句子边界切分，保证语义完整）
    
    Args:
        filepath: 文件路径
        chunk_size: 每个 chunk 的目标字符数
        overlap: 重叠字符数
    
    Returns:
        list of dict: [{'content': str, 'file': str, 'start_pos': int}, ...]
    """
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
    except Exception as e:
        print(f"⚠️ 读取文件 {filepath} 失败: {e}")
        return []
    
    # 按句子切分
    sentences = cut_by_punctuation(content)
    
    chunks = []
    current_chunk = []
    current_length = 0
    start_pos = 0
    
    for sent in sentences:
        sent_len = len(sent)
        
        # 如果当前 chunk + 新句子超过 chunk_size，保存当前 chunk
        if current_length + sent_len > chunk_size and current_chunk:
            chunk_text = ''.join(current_chunk)
            chunks.append({
                'content': chunk_text,
                'file': filepath,
                'start_pos': start_pos,
                'type': 'chunk'
            })
            
            # 计算重叠部分：保留最后几个句子
            overlap_length = 0
            overlap_sents = []
            for s in reversed(current_chunk):
                if overlap_length + len(s) <= overlap:
                    overlap_sents.insert(0, s)
                    overlap_length += len(s)
                else:
                    break
            
            # 重置 chunk，保留重叠部分
            current_chunk = overlap_sents
            current_length = overlap_length
            start_pos += len(chunk_text) - overlap_length
        
        current_chunk.append(sent)
        current_length += sent_len
    
    # 保存最后一个 chunk
    if current_chunk:
        chunk_text = ''.join(current_chunk)
        chunks.append({
            'content': chunk_text,
            'file': filepath,
            'start_pos': start_pos,
            'type': 'chunk'
        })
    
    return chunks

def attach_bm25_scores(candidates: list, query: str) -> list:
    """
    对候选内容进行 BM25 打分，并把分数绑定回原候选项。

    bm25_module.get_top_n() 返回的 scores 顺序与返回 docs 的排序一致，
    不是原始 corpus 的顺序，因此需要通过返回的文档索引回填分数。
    """
    if not candidates:
        return candidates

    corpus = [item['content'] for item in candidates]
    bm25 = BM25(corpus)
    docs, scores = bm25.get_top_n(query, len(corpus))

    score_by_index = {int(doc_idx): score for (doc_idx, _), score in zip(docs, scores)}
    for idx, item in enumerate(candidates):
        bm25_score = score_by_index.get(idx, 0.0)
        item['bm25_score'] = bm25_score
        item['score'] = bm25_score

    return candidates


def get_candidate_key(item: dict):
    """为候选项生成稳定唯一键，便于融合多路排序结果。"""
    if item['type'] == 'rg':
        return ('rg', item['file'], item['line_num'])
    return ('chunk', item['file'], item['start_pos'])

# ================= 1. 搜索工具实现 =================
def search_documents(query: str, broad_keywords: list, exact_keywords: list = None, 
                     target_files: list = None, search_dir: str = "./specs", 
                     top_k: int = 15, context_lines: int = 0) -> str:
    """
    四步走：RG搜索 -> 文件切块 -> BM25排序 -> 添加上下文
    
    Args:
        query: 用户的原始问题，用于 BM25 相关性计算
        broad_keywords: 宽泛关键词列表
        exact_keywords: 精确关键词列表
        target_files: 目标文件名列表
        search_dir: 搜索目录
        top_k: 返回前 K 个结果
        context_lines: 每个匹配行前后显示的行数
    """
    exact_keywords = exact_keywords or []
    target_files = target_files or []
    all_keywords = broad_keywords + exact_keywords
    
    print(f"\n🔍 原始问题: {query}")
    print(f"🔍 关键词: 宽泛={broad_keywords}, 精确={exact_keywords}")
    print(f"🔍 目标文件: {target_files}")
     
    # Step 1: RG 搜索收集匹配行
    matching_lines = []
    for kw in all_keywords:
        try:
            result = subprocess.run(
                ['rg', '-n', '-i', kw, search_dir],
                capture_output=True, text=True, encoding='utf-8', errors='ignore'
            )
            if result.returncode == 0:
                for line in result.stdout.strip().split('\n'):
                    if ':' in line:
                        parts = line.split(':', 2)
                        if len(parts) >= 3:
                            matching_lines.append({
                                'file': parts[0],
                                'line_num': int(parts[1]),
                                'content': parts[2],
                                'type': 'rg'
                            })
        except Exception as e:
            print(f"⚠️ 搜索 '{kw}' 出错: {e}")
    
    print(f"✅ RG 找到 {len(matching_lines)} 个匹配行")
    
    # Step 2: 读取目标文件并切块
    file_chunks = []
    for filename in target_files:
        # 在 search_dir 中查找文件
        for root, _, filenames in os.walk(search_dir):
            if filename in filenames:
                filepath = os.path.join(root, filename)
                chunks = chunk_file(filepath, chunk_size=512, overlap=50)
                file_chunks.extend(chunks)
                print(f"✅ 文件 {filename} 切分为 {len(chunks)} 个 chunks")
                break
    
    print(f"✅ 总共生成 {len(file_chunks)} 个文件 chunks")
    
    # 合并 RG 结果和文件 chunks
    all_candidates = matching_lines + file_chunks
    
    if not all_candidates:
        return "未找到匹配内容"
    
    print(f"✅ 总候选内容: {len(all_candidates)} 条")
    
    # Step 3: BM25 排序
    attach_bm25_scores(all_candidates, query)
    
    # 计算关键词加分，但保留原始 BM25 分数
    for item in all_candidates:
        content_lower = item['content'].lower()
        keyword_bonus = 0.0
        
        # 统计宽泛关键词命中数量
        broad_hit_count = sum(1 for kw in broad_keywords if kw.lower() in content_lower)
        if broad_hit_count == 2:
            keyword_bonus += 0.5
        elif broad_hit_count >= 3:
            keyword_bonus += 1.0
        
        # 如果内容中包含精确关键词，额外加分
        if exact_keywords:
            for exact_kw in exact_keywords:
                if exact_kw.lower() in content_lower:
                    keyword_bonus += 1.0
                    break  # 只加一次分

        item['keyword_bonus'] = keyword_bonus
        item['boosted_score'] = item['bm25_score'] + keyword_bonus
    
    bm25_sorted = sorted(all_candidates, key=lambda x: x['bm25_score'], reverse=True)
    boosted_sorted = sorted(all_candidates, key=lambda x: x['boosted_score'], reverse=True)
    
    # 去重策略：RG 行优先，如果 chunk 与 RG 行重叠则跳过 chunk
    # 1. 先收集所有 RG 行的内容（用于检测重叠）
    rg_contents = set()
    for item in bm25_sorted:
        if item['type'] == 'rg':
            rg_contents.add(item['content'].strip().lower())
    
    # 2. 去重：RG 结果按文件+行号，chunk 检查是否与 RG 内容重叠
    seen_rg = set()  # 已见过的 RG 行（文件+行号）
    deduplicated = []
    
    for item in bm25_sorted:
        if item['type'] == 'rg':
            # RG 行去重：按文件+行号
            key = get_candidate_key(item)
            if key not in seen_rg:
                deduplicated.append(item)
                seen_rg.add(key)
        else:  # chunk
            # Chunk 去重：检查是否与任何 RG 行内容重叠
            chunk_lower = item['content'].strip().lower()
            is_duplicate = False
            
            # 如果 chunk 包含任何 RG 行的内容，视为重复
            for rg_content in rg_contents:
                if rg_content in chunk_lower or chunk_lower in rg_content:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                deduplicated.append(item)

    dedup_map = {get_candidate_key(item): item for item in deduplicated}
    bm25_dedup = [item for item in bm25_sorted if get_candidate_key(item) in dedup_map]
    boosted_dedup = [item for item in boosted_sorted if get_candidate_key(item) in dedup_map]

    bm25_pick_count = min(10, top_k)
    boosted_pick_count = min(5, top_k)
    merged_items = []
    selected_keys = set()

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

    def final_rank_key(item: dict):
        selected_by = item.get('selected_by', [])
        both_selected = ('bm25' in selected_by and 'boosted' in selected_by)
        return (1 if both_selected else 0, item['boosted_score'], item['bm25_score'])

    top_items.sort(key=final_rank_key, reverse=True)
    print(
        f"📊 去重后: {len(deduplicated)} 条 (RG优先), "
        f"BM25取前{bm25_pick_count} + 修正取前{boosted_pick_count} -> Top-{len(top_items)}"
    )
    
    # 打印 Top-K 详细信息
    # print(f"\n{'='*80}")
    # print(f"📋 Top-{len(top_items)} 结果详情:")
    # print(f"{'='*80}")
    # for i, item in enumerate(top_items, 1):
    #     print(f"\n[{i}] 类型: {item['type'].upper()} | 分数: {item['score']:.4f}")
    #     print(f"    文件: {os.path.basename(item['file'])}")
    #     if item['type'] == 'rg':
    #         print(f"    行号: {item['line_num']}")
    #     else:
    #         print(f"    位置: {item['start_pos']}")
    #     print(f"    内容预览: {item['content']}...")
    # print(f"{'='*80}\n")

    # Step 4: 添加上下文（仅对 RG 结果）或直接返回 chunk
    results = []
    for i, item in enumerate(top_items, 1):
        if item['type'] == 'rg':
            context = _extract_context(item['file'], item['line_num'], context_lines)
            results.append(f"--- {os.path.basename(item['file'])}:行{item['line_num']} [RG] ---\n{context}\n")
        else:  # chunk
            results.append(f"--- {os.path.basename(item['file'])}:位置{item['start_pos']} [CHUNK] ---\n{item['content']}\n")
    
    final_result = "\n".join(results)
    print(f"\n✅ 返回内容总长度: {len(final_result)} 字符")
    return final_result

def _extract_context(filepath: str, line_num: int, context_lines: int = 0) -> str:
    """提取上下文"""
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        start = max(0, line_num - context_lines - 1)
        end = min(len(lines), line_num + context_lines)
        return "".join(f"{i+1}: {lines[i]}" for i in range(start, end))
    except:
        return ""

# ================= 2. Agent 主循环 =================
def run_search(query: str, search_dir: str = "./texts", top_k: int = 10, context_lines: int = 10):
    """
    两步 Agent：调用工具 -> 生成答案
    
    Args:
        query: 用户问题
        search_dir: 搜索目录
        top_k: 返回前 K 个结果
        context_lines: 每个匹配行前后显示的行数
    """
    
    print(f"\n{'='*60}\n🚀 问题: {query}\n{'='*60}")
    
    # 获取可用文件列表
    available_files = get_available_files(search_dir)
    # print(f"\n📁 可用文件 ({len(available_files)} 个): {available_files[:5]}..." if len(available_files) > 5 else f"\n📁 可用文件: {available_files}")
    
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

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query}
    ]
    
    # 第一轮：LLM 调用工具
    print("\n[第1轮] LLM 分析问题并调用工具...")
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,
        tools=TOOLS,
        tool_choice="auto",
        temperature=1
    )
    
    msg = response.choices[0].message
    msg2=[]
    
    # 处理工具调用
    if not msg.tool_calls:
        print("⚠️ LLM 未生成关键词，结束")
        return
    
    for tool_call in msg.tool_calls:
        if tool_call.function.name == "search_documents":
            args = json.loads(tool_call.function.arguments)
            print(f"📝 工具参数: {args}")
            
            result = search_documents(
                args.get("query", query),  # 使用 LLM 提供的 query 或原始问题
                args.get("broad_keywords", []),
                args.get("exact_keywords", []),
                args.get("target_files", []),
                search_dir=search_dir,
                top_k=top_k,
                context_lines=context_lines
            )
            
            # print(f"\n📄 搜索结果预览 (前200字符):\n{result[:200]}...\n")
            
            msg2.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": result
            })
    


    # print(msg2)

    # 第二轮：LLM 生成答案
    print("\n[第2轮] LLM 生成最终答案...")

    system_prompt_2 = f"""你是根据文档总结回答问题的专家
    根据文档已经检索到的信息为{msg2}，根据信息回答问题，并给出明确依据,未提及的不要回答。
    """

    messages_2 = [
        {"role": "system", "content": system_prompt_2},
        {"role": "user", "content": query}
    ]
    

    final_response = client.chat.completions.create(
        model=model_name,
        messages=messages_2,
        temperature=1
    )
    
    answer = final_response.choices[0].message.content
       
    if not answer or len(answer) == 0:
        print("⚠️ LLM 返回空答案！")
        print(f"📝 Response 对象: {final_response}")
    
    print(f"\n{'='*60}\n✅ 最终答案:\n{'='*60}\n{answer}\n{'='*60}")


# ================= 主程序 =================
if __name__ == "__main__":
    run_search(
        query="独立基础的高宽比",        
        search_dir="./specs",
        top_k=10,
        context_lines=0

    )
