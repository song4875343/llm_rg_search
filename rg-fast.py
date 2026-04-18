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

MODEL_NUM = 4

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
                }
            },
            "required": ["query", "broad_keywords"]
        }
    }
}]
# ================= 1. 搜索工具实现 =================
def search_documents(query: str, broad_keywords: list, exact_keywords: list = None, 
                     search_dir: str = "./specs", top_k: int = 15, context_lines: int = 0) -> str:
    """
    三步走：RG搜索 -> BM25排序 -> 添加上下文
    
    Args:
        query: 用户的原始问题，用于 BM25 相关性计算
        broad_keywords: 宽泛关键词列表
        exact_keywords: 精确关键词列表
        search_dir: 搜索目录
        top_k: 返回前 K 个结果
        context_lines: 每个匹配行前后显示的行数
    """
    exact_keywords = exact_keywords or []
    all_keywords = broad_keywords + exact_keywords
    
    print(f"\n🔍 原始问题: {query}")
    print(f"🔍 关键词: 宽泛={broad_keywords}, 精确={exact_keywords}")
    exact_keywords = exact_keywords or []
    all_keywords = broad_keywords + exact_keywords
     
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
                                'content': parts[2]
                            })
        except Exception as e:
            print(f"⚠️ 搜索 '{kw}' 出错: {e}")
    
    if not matching_lines:
        return "未找到匹配内容"
    
    print(f"✅ 找到 {len(matching_lines)} 个匹配行")
    
    # Step 2: BM25 排序
    corpus = [line['content'] for line in matching_lines]
    bm25 = BM25(corpus)
    
    # get_top_n 需要字符串查询，返回 (文档列表, 分数列表)
    _, scores = bm25.get_top_n(query, len(corpus))
    
    # 按分数排序，并对包含精确关键词的行加分
    for i, line in enumerate(matching_lines):
        line['score'] = scores[i]
        line_lower = line['content'].lower()
        
        # 统计宽泛关键词命中数量
        broad_hit_count = sum(1 for kw in broad_keywords if kw.lower() in line_lower)
        if broad_hit_count == 2:
            line['score'] += 0.5
        elif broad_hit_count >= 3:
            line['score'] += 1.0
        
        # 如果行中包含精确关键词，额外加分
        if exact_keywords:
            for exact_kw in exact_keywords:
                if exact_kw.lower() in line_lower:
                    line['score'] += 1.0
                    break  # 只加一次分
    
    matching_lines.sort(key=lambda x: x['score'], reverse=True)
    
    # 去重：确保原始行不重复（按文件+行号）
    seen = set()
    deduplicated = []
    for line in matching_lines:
        key = (line['file'], line['line_num'])
        if key not in seen:
            deduplicated.append(line)
            seen.add(key)
    
    top_lines = deduplicated[:top_k]
    print(f"📊 去重后: {len(deduplicated)} 行, Top-{len(top_lines)} 匹配行")
    # [print(L,'\n') for L in top_lines]

    # Step 3: 添加上下文
    results = []
    for i, line in enumerate(top_lines, 1):
        context = _extract_context(line['file'], line['line_num'], context_lines)
        results.append(f"--- {os.path.basename(line['file'])}:行{line['line_num']}  ---\n{context}\n")
    
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
    
    system_prompt = """你是文档检索专家。
    
工作流程：
1. 分析用户问题，提取关键词
2. 调用 search_documents 工具搜索
3. 基于搜索结果生成答案

注意：
- broad_keywords: 1-2个核心关键词
- exact_keywords: 1,最特殊,最关键的元关键词可以为broad_keywords元关键词中较特殊的一个
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
        query="基础的宽高比",        
        search_dir="./specs",
        top_k=10,
        context_lines=0

    )
