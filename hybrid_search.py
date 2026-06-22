"""
混合检索系统 - 先快速检索，信息不足时再深度搜索
流程：fast第1轮(搜索) -> 评估 -> 够:生成答案 / 不够:转agent
"""

import sys
import json
from pathlib import Path

# 确保能导入同目录模块
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# 导入两个搜索模块
import importlib.util
spec_fast = importlib.util.spec_from_file_location("fast_search", SCRIPT_DIR / "rg-fast-v2a.py")
fast_search = importlib.util.module_from_spec(spec_fast)
spec_fast.loader.exec_module(fast_search)

spec_v6a = importlib.util.spec_from_file_location("deep_search", SCRIPT_DIR / "rg_search_v6a.py")
deep_search = importlib.util.module_from_spec(spec_v6a)
spec_v6a.loader.exec_module(deep_search)


def evaluate_completeness(query: str, fast_result: str, client, model_name: str) -> bool:
    """让 LLM 评估快速检索结果是否足够回答问题（快速版本）"""
    # 限制输入长度，避免评估太慢
    max_result_length = 2000  # 进一步限制长度
    truncated_result = fast_result[:max_result_length]
    if len(fast_result) > max_result_length:
        truncated_result += "\n...(已截断)..."
    
    # 使用极简提示词，加快评估速度
    messages = [
        {"role": "user", "content": f"""问题：{query}

检索到的内容：
{truncated_result}

这些内容是否足够回答问题？只回答YES或NO。"""}
    ]
    
    try:
        # 针对不同模型优化参数
        eval_kwargs = {
            "model": model_name,
            "messages": messages,
            "temperature": 0,
            "max_tokens": 3,  # 只需要 YES/NO
        }
        
        # 如果是 qwen 模型，禁用思考模式加速
        if 'qwen' in model_name.lower():
            eval_kwargs['extra_body'] = {'enable_thinking': False}
        
        response = client.chat.completions.create(**eval_kwargs)
        answer = response.choices[0].message.content.strip().upper()
        result = "YES" in answer
        print(f"   ⚡ 评估: {'✅ 足够' if result else '❌ 不足'} ({answer[:20]})")
        return result
    except Exception as e:
        print(f"   ⚠️ 评估失败: {e}，默认转深度搜索")
        return False


def hybrid_search(query: str, search_dir: str = "./specs", stream: bool = False):
    """混合搜索：fast第1轮搜索 -> 评估 -> 够:生成答案 / 不够:转agent"""
    def _core():
        yield f"\n{'='*70}\n🔍 混合检索模式\n{'='*70}\n"
        yield f"📝 问题: {query}\n"
        
        # ===== 阶段1：快速检索（只执行第1轮：工具调用） =====
        yield f"\n{'─'*70}\n⚡ [阶段1] 快速检索 - 调用搜索工具\n{'─'*70}\n"
        
        messages = fast_search.build_tool_messages(query, search_dir)
        first_round_kwargs = fast_search.build_chat_kwargs(
            messages, 
            tools=fast_search.TOOLS, 
            temperature=1, 
            thinking_enabled_override=False
        )
        first_round_kwargs["tool_choice"] = {"type": "function", "function": {"name": "search_documents"}}
        
        response = fast_search.client.chat.completions.create(**first_round_kwargs)
        
        if not response or not response.choices or not response.choices[0].message.tool_calls:
            yield "⚠️ 快速检索失败：无法生成搜索关键词\n"
            return
        
        # 执行搜索工具
        tool_call = response.choices[0].message.tool_calls[0]
        args = json.loads(tool_call.function.arguments)
        yield f"📝 搜索参数: {args}\n"
        
        # 调用 search_documents
        fast_result = fast_search.search_documents(
            args.get("query", query),
            args.get("broad_keywords", []),
            args.get("exact_keywords", []),
            args.get("target_files", []),
            search_dir=search_dir,
            top_k=10,
            context_lines=0,
            stream=False
        )
        
        yield f"✅ 搜索完成，结果长度: {len(fast_result)} 字符\n"
        
        # ===== 阶段2：评估完整性 =====
        yield f"\n{'─'*70}\n🤔 [阶段2] 评估信息完整性\n{'─'*70}\n"
        
        is_complete = evaluate_completeness(
            query, 
            fast_result, 
            fast_search.client, 
            fast_search.model_name
        )
        
        if is_complete:
            # 信息足够，生成答案（fast 的第2轮）
            yield "✅ 信息足够完整，生成最终答案\n"
            yield f"\n{'─'*70}\n💡 [生成答案]\n{'─'*70}\n"
            
            # 构建答案消息
            answer_messages = fast_search.build_answer_messages(query, [{
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": fast_result
            }])
            
            # 流式生成答案
            answer_kwargs = fast_search.build_chat_kwargs(
                answer_messages,
                stream=True,
                temperature=1
            )
            
            answer_stream = fast_search.client.chat.completions.create(**answer_kwargs)
            
            for chunk in answer_stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    content = chunk.choices[0].delta.content
                    if stream:
                        yield content
            
            yield f"\n{'='*70}\n"
            return
        
        # ===== 阶段3：深度搜索 =====
        yield "⚠️ 信息不足，启动深度搜索 (Agent + 章节上下文)\n"
        yield f"\n{'─'*70}\n🔬 [阶段3] 深度搜索 (Multi-turn Agent)\n{'─'*70}\n"
        
        deep_search.set_target_folder(search_dir)
        
        # 调用深度搜索，不传递上下文（完全独立搜索）
        for chunk in deep_search.run_agent(
            query, 
            show_reasoning=False, 
            stream=True, 
            extract_refs=True
        ):
            # 处理 _chat_stream 返回的元组格式 (content, reasoning)
            if isinstance(chunk, tuple):
                content, _ = chunk
                if stream and content:
                    yield content
            else:
                # 普通字符串直接输出
                if stream:
                    yield chunk
        
        yield f"\n{'='*70}\n"
    
    if stream:
        return _core()
    else:
        results = list(_core())
        for r in results:
            print(r, end='', flush=True)
        return results


if __name__ == "__main__":
    # 使用示例
    # query = "独立基础的高宽比"
    query ='门式刚架何时采用缆风绳'
    # query = "独立基础的性能化设计"
    # 流式输出（推荐）
    for chunk in hybrid_search(query, search_dir="./specs", stream=True):
        print(chunk, end='', flush=True)
    
    # 或非流式
    # hybrid_search(query, search_dir="./specs", stream=False)
