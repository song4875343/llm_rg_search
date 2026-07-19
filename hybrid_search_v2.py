"""
混合检索系统 v2 - 先用 fast v2c 获取证据，信息不足时再转 v6b 深度 Agent。
流程：全局 BM25 预览 -> 第1次 LLM 选文件并调用工具1 -> 评估 -> 够则终稿 / 不够则转 v6b。
"""

import sys
import json
from pathlib import Path
import importlib.util

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

spec_fast = importlib.util.spec_from_file_location("fast_search", SCRIPT_DIR / "rg-fast-v2c.py")
fast_search = importlib.util.module_from_spec(spec_fast)
spec_fast.loader.exec_module(fast_search)

spec_v6b = importlib.util.spec_from_file_location("deep_search", SCRIPT_DIR / "rg_search_v6b.py")
deep_search = importlib.util.module_from_spec(spec_v6b)
spec_v6b.loader.exec_module(deep_search)


def _safe_json_loads(raw: str) -> dict:
    try:
        return json.loads(raw or "{}")
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        obj, _ = decoder.raw_decode((raw or "{}").strip())
        return obj


def evaluate_completeness(query: str, fast_result: str, client, model_name: str) -> bool:
    """让 LLM 评估快速检索结果是否足够回答问题。"""
    max_result_length = 2000
    truncated_result = fast_result[:max_result_length]
    if len(fast_result) > max_result_length:
        truncated_result += "\n...(已截断)..."
    messages = [
        {"role": "user", "content": f"""问题：{query}

检索到的内容：
{truncated_result}

这些内容是否足够回答问题？只回答YES或NO。"""}
    ]

    try:
        # 针对不同模型优化参数
        model_lower = model_name.lower()
        eval_kwargs = {
            "model": model_name,
            "messages": messages,
            "temperature": 0,
            "max_tokens": 3,  # 只需要 YES/NO
        }
        if "kimi" in model_lower:
            eval_kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
            eval_kwargs["temperature"] = 0.6
        elif "qwen" in model_lower:
            eval_kwargs["extra_body"] = {"enable_thinking": False}
        elif "deepseek" in model_lower:
            eval_kwargs["extra_body"] = {"thinking": {"type": "disabled"}}


        response = client.chat.completions.create(**eval_kwargs)
        answer = response.choices[0].message.content.strip().upper()
        result = "YES" in answer
        print(f"   ⚡ 评估: {'✅ 足够' if result else '❌ 不足'} ({answer[:20]})")
        return result
    except Exception as e:
        print(f"   ⚠️ 评估失败: {e}，默认转深度搜索")
        return False


def collect_fast_v2c_evidence(query: str, search_dir: str = "./specs", preview_top_n: int | None = None, file_top_k: int | None = None, context_lines: int | None = None):
    """执行 fast v2c 的前两步，返回预览文本和完整证据文本。"""
    preview_top_n = preview_top_n or fast_search.GLOBAL_TOP_N
    file_top_k = fast_search.FILE_TOP_K if file_top_k is None else file_top_k
    context_lines = fast_search.CONTENT_LINES if context_lines is None else context_lines

    preview_items, preview_text = fast_search.global_bm25_preview(
        query,
        search_dir=search_dir,
        top_n=preview_top_n,
        preview_chars=fast_search.PREVIEW_CHARS,
    )

    selection_messages = fast_search.build_selection_messages(query, preview_text, search_dir, preview_top_n=preview_top_n)
    first_kwargs = fast_search.build_chat_kwargs(
        selection_messages,
        tools=fast_search.TOOLS,
        tool_choice={"type": "function", "function": {"name": "search_high_probability_files"}},
        temperature=1,
        thinking_enabled_override=False,
    )
    response = fast_search.client.chat.completions.create(**first_kwargs)

    evidence_items = None
    tool_logs = []
    if response and response.choices:
        for tool_call in response.choices[0].message.tool_calls or []:
            if tool_call.function.name != "search_high_probability_files":
                continue
            args = _safe_json_loads(tool_call.function.arguments)
            items, text = fast_search.search_high_probability_files(
                query,
                args.get("target_files", []),
                search_dir=search_dir,
                file_top_k=file_top_k,
                stream=False,
            )
            evidence_items = items
            tool_logs.append({"tool_call": tool_call, "args": args, "content": text})

    if evidence_items is None:
        evidence_items = fast_search._fallback_evidence(query, search_dir, context_lines, file_top_k)

    if file_top_k and len(evidence_items) > file_top_k:
        evidence_items = evidence_items[:file_top_k]
    evidence_text = fast_search._format_evidence_list(evidence_items)

    return {
        "preview_items": preview_items,
        "preview_text": preview_text,
        "evidence_items": evidence_items,
        "evidence_text": evidence_text,
        "tool_logs": tool_logs,
    }


def hybrid_search(query: str, search_dir: str = "./specs", stream: bool = False):
    """混合搜索：fast v2c 证据召回 -> 评估 -> 够则生成答案 / 不够则转 v6b Agent。"""
    def _core():
        yield f"\n{'=' * 70}\n🔍 混合检索模式 v2\n{'=' * 70}\n"
        yield f"📝 问题: {query}\n"

        yield f"\n{'─' * 70}\n⚡ [阶段1] fast v2c：全局 BM25 预览 + 文件内 BM25 证据\n{'─' * 70}\n"
        fast_pack = collect_fast_v2c_evidence(query, search_dir=search_dir, context_lines=0)
        evidence_text = fast_pack["evidence_text"]
        timings = fast_search.LAST_TIMINGS.get("global_preview", {})
        yield f"✅ 全局 Top-{len(fast_pack['preview_items'])} 预览完成，证据片段 {len(fast_pack['evidence_items'])} 条\n"
        if timings:
            yield f"⏱️ 全局预览: {timings.get('total', 0):.3f}s | chunks={timings.get('chunk_count', 0)}\n"
        for log in fast_pack["tool_logs"]:
            yield f"📝 工具1参数: {log['args']}\n"
        yield f"✅ fast 证据长度: {len(evidence_text)} 字符\n"

        yield f"\n{'─' * 70}\n🤔 [阶段2] 评估信息完整性\n{'─' * 70}\n"
        is_complete = evaluate_completeness(query, evidence_text, fast_search.client, fast_search.model_name)
        if is_complete:
            yield "✅ 信息足够完整，生成最终答案\n"
            yield f"\n{'─' * 70}\n💡 [生成答案]\n{'─' * 70}\n"
            final_messages = fast_search.build_final_messages(query, fast_pack["preview_text"], evidence_text)
            answer_kwargs = fast_search.build_chat_kwargs(final_messages, stream=True, temperature=1, thinking_enabled_override=False)
            answer_stream = fast_search.client.chat.completions.create(**answer_kwargs)
            for chunk in answer_stream:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
            yield f"\n{'=' * 70}\n"
            return

        yield "⚠️ 信息不足，启动深度搜索 (v6b Agent)\n"
        yield f"\n{'─' * 70}\n🔬 [阶段3] 深度搜索\n{'─' * 70}\n"
        deep_search.set_target_folder(search_dir)
        for chunk in deep_search.run_agent(query, show_reasoning=False, stream=True, extract_refs=True):
            if isinstance(chunk, tuple):
                content, _ = chunk
                if content:
                    yield content
            elif chunk:
                yield chunk
        yield f"\n{'=' * 70}\n"

    if stream:
        return _core()
    results = list(_core())
    for r in results:
        print(r, end="", flush=True)
    return results


if __name__ == "__main__":
    query = "筏板的最小厚度"
    for chunk in hybrid_search(query, search_dir="./specs", stream=True):
        print(chunk, end="", flush=True)
