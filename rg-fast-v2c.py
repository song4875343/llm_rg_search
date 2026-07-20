"""
BM25 两步检索 Agent，v2c 精简版。

流程：
1. 系统先召回全局 BM25 Top-N 切片，只暴露原文短预览。
2. 第 1 次 LLM 只选择高概率文件，并且只调用 search_high_probability_files。
3. 系统整理证据后，第 2 次 LLM 生成最终答案。
"""

import bisect
import json
import os
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from bm25_module import BM25
from model_config import MODEL_CONFIG

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()

# ===== 模型配置 =====

MODEL_NUM = 8
THINKING_ENABLED = False
CONTENT_LINES = 0
GLOBAL_TOP_N = 30
PREVIEW_CHARS = 50
FILE_TOP_K = 15

SCRIPT_DIR = Path(__file__).resolve().parent
config = MODEL_CONFIG[MODEL_NUM]
client = OpenAI(base_url=config["base_url"], api_key=os.getenv(config["api_key"]))
model_name = config["model_name"]
print(f"🤖 使用模型: {model_name}，序号{MODEL_NUM}")

GLOBAL_CHUNK_CACHE = {}
PREVIEW_ITEM_CACHE = {}
LAST_TIMINGS = {}


def _thinking_caps(cfg=None):
    cfg = cfg or config
    kind, forced = cfg.get("thinking"), "thinking" in cfg["model_name"].lower()
    return {
        "supported": bool(kind),
        "can_disable": bool(kind) and not forced,
        "forced": forced,
        "kind": kind,
    }


def build_chat_kwargs(
    messages,
    stream=False,
    tools=None,
    tool_choice=None,
    temperature=1,
    thinking_enabled_override=None,
):
    caps = _thinking_caps()
    thinking_enabled = (
        THINKING_ENABLED
        if thinking_enabled_override is None
        else thinking_enabled_override
    )
    if caps["kind"] == "kimi" and caps["can_disable"] and not thinking_enabled:
        temperature = 0.6

    kw = {
        "model": model_name,
        "messages": messages,
        "temperature": temperature,
        "stream": stream,
    }
    if tools:
        # 这些版本的第 1 轮固定要调用工具，不再让模型自行判断是否用工具。
        kw.update(tools=tools, tool_choice=tool_choice or "required")

    extra_body = (
        {"thinking": {"type": "disabled"}}
        if caps["kind"] == "kimi" and caps["can_disable"] and not thinking_enabled
        else {"enable_thinking": thinking_enabled}
        if caps["kind"] == "qwen" and caps["can_disable"]
        else {"thinking": {"type": "enabled" if thinking_enabled else "disabled"}}
        if caps["kind"] == "deepseek" and caps["can_disable"]
        else None
    )
    if extra_body:
        kw["extra_body"] = extra_body
    return kw


def _emit(stream, msg):
    if stream:
        yield msg
    else:
        print(msg)


def _consume_return(gen):
    try:
        while True:
            next(gen)
    except StopIteration as e:
        return e.value


def _as_search_dir(search_dir: str) -> str:
    path = Path(search_dir)
    return str(path if path.is_absolute() else SCRIPT_DIR / path)


def get_available_files(search_dir: str) -> list:
    root_dir = _as_search_dir(search_dir)
    return sorted(
        f
        for root, _, files in os.walk(root_dir)
        for f in files
        if f.endswith((".txt", ".md"))
    )


def _normalize_ws(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def _preview_text(text: str, chars: int = PREVIEW_CHARS) -> str:
    preview = _normalize_ws(text)[:chars]
    return preview + ("..." if len(_normalize_ws(text)) > chars else "")


def _read_line_range(filepath: str, start_line: int, end_line: int) -> str:
    try:
        lines = open(filepath, "r", encoding="utf-8", errors="ignore").readlines()
        start = max(1, int(start_line))
        end = min(len(lines), int(end_line))
        return "".join(lines[start - 1 : end])
    except Exception:
        return ""


def chunk_file(filepath: str, chunk_size: int = 512, overlap: int = 50) -> list:
    """按句子边界将文件切分成 chunks。"""
    try:
        text = open(filepath, "r", encoding="utf-8", errors="ignore").read()
        line_starts = [0] + [m.end() for m in re.finditer("\n", text)]
        sentences = []
        for match in re.finditer(r"[^。！？.!?]+[。！？.!?]?", text):
            raw = match.group()
            sentence = raw.strip()
            if not sentence:
                continue
            left_trim, right_trim = (
                len(raw) - len(raw.lstrip()),
                len(raw) - len(raw.rstrip()),
            )
            start_pos, end_pos = match.start() + left_trim, match.end() - right_trim
            start_line = bisect.bisect_right(line_starts, start_pos)
            end_line = bisect.bisect_right(line_starts, end_pos)
            sentences.append((sentence, start_line, start_pos, end_line, end_pos))
    except Exception as e:
        print(f"⚠️ 读取文件 {filepath} 失败: {e}")
        return []

    chunks, current_chunk, current_length = [], [], 0

    def flush_chunk():
        nonlocal current_chunk, current_length
        chunk_text = "".join(sent for sent, _, _, _, _ in current_chunk)
        chunks.append(
            {
                "content": chunk_text,
                "file": filepath,
                "filename": os.path.basename(filepath),
                "start_line": current_chunk[0][1],
                "end_line": current_chunk[-1][3],
                "start_pos": current_chunk[0][2],
                "type": "chunk",
            }
        )
        overlap_items, overlap_length = [], 0
        for sent, start_line, start_pos, end_line, end_pos in reversed(current_chunk):
            if overlap_length + len(sent) > overlap:
                break
            overlap_items.insert(0, (sent, start_line, start_pos, end_line, end_pos))
            overlap_length += len(sent)
        current_chunk, current_length = overlap_items, overlap_length

    for sent, start_line, start_pos, end_line, end_pos in sentences:
        sent_len = len(sent)
        if current_length + sent_len > chunk_size and current_chunk:
            flush_chunk()
        current_chunk.append((sent, start_line, start_pos, end_line, end_pos))
        current_length += sent_len
    if current_chunk:
        flush_chunk()
    return chunks


def _build_global_chunks(
    search_dir: str, chunk_size: int = 512, overlap: int = 50
) -> list:
    root_dir = _as_search_dir(search_dir)
    cache_key = (root_dir, chunk_size, overlap)
    if cache_key in GLOBAL_CHUNK_CACHE:
        return GLOBAL_CHUNK_CACHE[cache_key]

    chunks = []
    for root, _, files in os.walk(root_dir):
        for filename in sorted(f for f in files if f.endswith((".txt", ".md"))):
            chunks.extend(
                chunk_file(
                    os.path.join(root, filename), chunk_size=chunk_size, overlap=overlap
                )
            )
    for idx, item in enumerate(chunks):
        item["global_id"] = f"G{idx + 1:06d}"
    GLOBAL_CHUNK_CACHE[cache_key] = chunks
    return chunks


def attach_bm25_scores(candidates: list, query: str) -> list:
    if not candidates:
        return candidates
    corpus = [item["content"] for item in candidates]
    docs, scores = BM25(corpus).get_top_n(query, len(corpus))
    score_by_index = {int(doc_idx): score for (doc_idx, _), score in zip(docs, scores)}
    for idx, item in enumerate(candidates):
        item["bm25_score"] = score_by_index.get(idx, 0.0)
        item["score"] = item["bm25_score"]
    return candidates


def _rank_bm25(candidates: list, query: str, top_k: int) -> list:
    ranked = [dict(item) for item in candidates]
    attach_bm25_scores(ranked, query)
    return sorted(ranked, key=lambda x: x.get("bm25_score", 0.0), reverse=True)[:top_k]


def _format_preview_items(items: list) -> str:
    lines = []
    for item in items:
        lines.append(
            f"{item['preview_id']} | 文件={item['filename']} | 行{item['start_line']}-{item['end_line']} | "
            f"score={item.get('bm25_score', 0.0):.4f}\n预览-->{_preview_text(item['content'])}"
        )
    return "\n".join(lines)


def global_bm25_preview(
    query: str,
    search_dir: str = "./specs",
    top_n: int = GLOBAL_TOP_N,
    preview_chars: int = PREVIEW_CHARS,
) -> tuple:
    global PREVIEW_ITEM_CACHE, LAST_TIMINGS
    t0 = time.perf_counter()
    chunks = _build_global_chunks(search_dir)
    t_chunks = time.perf_counter()
    top_items = _rank_bm25(chunks, query, top_n)
    t_rank = time.perf_counter()
    PREVIEW_ITEM_CACHE = {}
    for idx, item in enumerate(top_items, 1):
        item["preview_id"] = f"P{idx:03d}"
        item["preview_chars"] = preview_chars
        PREVIEW_ITEM_CACHE[item["preview_id"]] = item
    preview_text = _format_preview_items(top_items)
    t_format = time.perf_counter()
    LAST_TIMINGS["global_preview"] = {
        "chunks": t_chunks - t0,
        "rank": t_rank - t_chunks,
        "format": t_format - t_rank,
        "total": t_format - t0,
        "chunk_count": len(chunks),
        "top_count": len(top_items),
    }
    return top_items, preview_text


def _match_target_files(search_dir: str, target_files) -> list:
    requested = (
        [target_files] if isinstance(target_files, str) else list(target_files or [])
    )
    root_dir = _as_search_dir(search_dir)
    all_paths = [
        os.path.join(root, f)
        for root, _, files in os.walk(root_dir)
        for f in files
        if f.endswith((".txt", ".md"))
    ]
    matches = []
    for req in requested:
        req = str(req).strip()
        if not req:
            continue
        for path in all_paths:
            if (
                req == os.path.basename(path)
                or req in os.path.basename(path)
                or req in path
            ):
                if path not in matches:
                    matches.append(path)
    return matches


def _format_evidence(item: dict, idx: int = None) -> str:
    eid = f"E{idx:03d}" if idx is not None else item.get("evidence_id", "E???")
    source = item.get("source", "BM25")
    score = item.get("bm25_score", item.get("score", 0.0))
    content = item.get("expanded_content") or item.get("content", "")
    return (
        f"--- [{eid}] {os.path.basename(item['file'])}:行{item['start_line']}-{item['end_line']} "
        f"[{source}] score={score:.4f} ---\n{content}\n--- [{eid}] 片段结束 ---"
    )


def _format_evidence_list(items: list) -> str:
    if not items:
        return "系统反馈：没有可用完整片段。"
    blocks = []
    for idx, item in enumerate(items, 1):
        item["evidence_id"] = f"E{idx:03d}"
        blocks.append(_format_evidence(item, idx))
    return "\n\n".join(blocks)


def search_high_probability_files(
    query: str,
    target_files: list,
    search_dir: str = "./specs",
    file_top_k: int = FILE_TOP_K,
    stream: bool = False,
):
    """工具 1：在第 1 次 LLM 选出的高概率文件内做 BM25 召回。"""

    def _core():
        paths = _match_target_files(search_dir, target_files)
        yield from _emit(
            stream,
            f"🛠️ [工具1] 高概率文件 BM25: 请求={target_files}, 匹配文件={len(paths)}",
        )
        if not paths:
            return [], "系统反馈：未匹配到高概率文件。"
        chunks = []
        for path in paths:
            chunks.extend(chunk_file(path, chunk_size=512, overlap=50))
        top_items = _rank_bm25(chunks, query, int(file_top_k or FILE_TOP_K))
        for item in top_items:
            item["source"] = "TOOL1_FILE_BM25"
        text = _format_evidence_list(top_items)
        yield from _emit(
            stream, f"✅ [工具1] 文件 chunks={len(chunks)}，返回 Top-{len(top_items)}"
        )
        return top_items, text

    if stream:
        return _core()
    return _consume_return(_core())


TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_high_probability_files",
            "description": "工具1：在高概率文件列表范围内进行 BM25 召回，返回完整片段。",
            "parameters": {
                "type": "object",
                "properties": {
                    "target_files": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "高概率文件名列表，可用文件名片段",
                    },
                },
                "required": ["target_files"],
            },
        },
    },
]


def build_selection_messages(
    query: str, preview_text: str, search_dir: str, preview_top_n: int = GLOBAL_TOP_N
):
    return [
        {
            "role": "system",
            "content": f"""你是规范资料证据筛选专家。你现在处于严格两次 LLM 流程的第 1 次调用。

可用文件列表：
{", ".join(get_available_files(search_dir))}

系统已经根据用户问题完成全局 BM25 Top-{preview_top_n} 召回。每条只含编号、文件、行号和原文前 {PREVIEW_CHARS} 字预览。

你的任务：
1. 根据用户问题和 Top-{preview_top_n} 预览，判断哪些文件最可能包含答案。
2. 只调用 search_high_probability_files，在高概率文件范围内继续 BM25 召回完整片段；该工具会由系统自动使用原始用户问题，不需要你传 query。工具参数只允许 target_files，不要携带 top_k、file_top_k 或任何额外字段。
3. 本轮不要直接回答用户问题，只负责调用这个工具获取证据。

注意：
- target_files 必须来自可用文件列表，可传文件名片段。
- 不要调用其它工具；本版本只允许通过高概率文件列表做文件内 BM25。""",
        },
        {
            "role": "user",
            "content": f"用户问题：{query}\n\nTop-{preview_top_n} 条目预览：\n{preview_text}",
        },
    ]


def build_final_messages(
    query: str, preview_text: str, evidence_text: str, preview_top_n: int = GLOBAL_TOP_N
):
    return [
        {
            "role": "system",
            "content": """你是根据规范原文回答问题的专家。只能依据提供的完整片段回答，未提及的不要扩展。
要求：
1. 直接回答结论。
2. 给出关键依据，尽量包含规范名、行号范围和条文号。
3. 如果证据不足，明确说明不足点。""",
        },
        {
            "role": "user",
            "content": f"用户问题：{query}\n\n全局 Top-{preview_top_n} 预览仅供定位参考：\n{preview_text}\n\n完整证据片段：\n{evidence_text}",
        },
    ]


def _safe_json_loads(raw: str) -> dict:
    try:
        return json.loads(raw or "{}")
    except json.JSONDecodeError:
        decoder = json.JSONDecoder()
        obj, _ = decoder.raw_decode(raw.strip())
        return obj


def _execute_tool_call(
    tool_call, query: str, search_dir: str, context_lines: int, file_top_k: int
):
    name = tool_call.function.name
    args = _safe_json_loads(tool_call.function.arguments)
    if name == "search_high_probability_files":
        items, text = search_high_probability_files(
            query,
            args.get("target_files", []),
            search_dir=search_dir,
            file_top_k=file_top_k,
            stream=False,
        )
    else:
        items, text = [], f"系统反馈：未知工具 {name}"
    return {"role": "tool", "tool_call_id": tool_call.id, "content": text}, items, args


def _fallback_evidence(
    query: str, search_dir: str, context_lines: int, file_top_k: int
):
    """兜底策略：从 Top-N 预览中选择靠前文件，再执行文件内 BM25。"""
    top_files = []
    for item in PREVIEW_ITEM_CACHE.values():
        if item["filename"] not in top_files:
            top_files.append(item["filename"])
        if len(top_files) >= 3:
            break
    file_items, _ = search_high_probability_files(
        query, top_files, search_dir=search_dir, file_top_k=file_top_k, stream=False
    )
    return file_items


def run_search(
    query: str,
    search_dir: str = "./specs",
    preview_top_n: int = GLOBAL_TOP_N,
    file_top_k: int = FILE_TOP_K,
    context_lines: int = CONTENT_LINES,
    stream: bool = False,
):
    """严格两次 LLM 调用的 BM25 检索流程。"""

    def _core():
        total_t0 = time.perf_counter()
        yield from _emit(stream, f"\n{'=' * 60}\n🚀 问题: {query}\n{'=' * 60}")

        step1_t0 = time.perf_counter()
        yield from _emit(
            stream,
            f"\n[系统步骤] 全局 BM25 召回 Top-{preview_top_n}，并返回前 {PREVIEW_CHARS} 字预览...",
        )
        preview_items, preview_text = global_bm25_preview(
            query,
            search_dir=search_dir,
            top_n=preview_top_n,
            preview_chars=PREVIEW_CHARS,
        )
        step1_elapsed = time.perf_counter() - step1_t0
        gt = LAST_TIMINGS.get("global_preview", {})
        yield from _emit(
            stream,
            f"✅ 全局切片数: {gt.get('chunk_count', len(_build_global_chunks(search_dir)))}，Top-{len(preview_items)} 预览已生成\n"
            f"⏱️ [第1步耗时] 总计 {step1_elapsed:.3f}s | 切片构建/缓存 {gt.get('chunks', 0):.3f}s | BM25排序 {gt.get('rank', 0):.3f}s | 预览格式化 {gt.get('format', 0):.3f}s",
        )

        step2_t0 = time.perf_counter()
        yield from _emit(
            stream,
            f"\n[第1次 LLM] 根据用户问题和 Top-{preview_top_n} 预览选择高概率文件并调用文件内 BM25...",
        )
        selection_messages = build_selection_messages(
            query, preview_text, search_dir, preview_top_n=preview_top_n
        )
        first_kwargs = build_chat_kwargs(
            selection_messages,
            tools=TOOLS,
            tool_choice={
                "type": "function",
                "function": {"name": "search_high_probability_files"},
            },
            temperature=1,
            thinking_enabled_override=False,
        )
        response = client.chat.completions.create(**first_kwargs)
        if not response or not response.choices:
            yield from _emit(stream, "⚠️ 第1次 LLM 返回空响应，启用本地兜底证据选择")
            evidence_items = _fallback_evidence(
                query, search_dir, context_lines, file_top_k
            )
        else:
            first_msg = response.choices[0].message
            evidence_items, tool_messages = None, []
            for tool_call in first_msg.tool_calls or []:
                if tool_call.function.name != "search_high_probability_files":
                    continue
                yield from _emit(
                    stream,
                    f"📝 工具调用: {tool_call.function.name} {tool_call.function.arguments}",
                )
                tool_msg, items, _ = _execute_tool_call(
                    tool_call, query, search_dir, context_lines, file_top_k
                )
                tool_messages.append(tool_msg)
                evidence_items = items
            if evidence_items is None:
                yield from _emit(
                    stream, "⚠️ 第1次 LLM 未调用文件 BM25 工具，启用本地兜底证据选择"
                )
                evidence_items = _fallback_evidence(
                    query, search_dir, context_lines, file_top_k
                )

        format_t0 = time.perf_counter()
        if file_top_k and len(evidence_items) > file_top_k:
            evidence_items = evidence_items[:file_top_k]
        evidence_text = _format_evidence_list(evidence_items)
        format_elapsed = time.perf_counter() - format_t0
        step2_elapsed = time.perf_counter() - step2_t0
        yield from _emit(stream, f"✅ 工具1返回完整证据: {len(evidence_items)} 条")
        yield from _emit(
            stream,
            f"⏱️ [第2步耗时] 总计 {step2_elapsed:.3f}s | 证据格式化/截断 {format_elapsed:.3f}s",
        )

        step3_t0 = time.perf_counter()
        yield from _emit(stream, "\n[第2次 LLM] 基于完整证据生成终稿...")
        final_messages = build_final_messages(
            query, preview_text, evidence_text, preview_top_n=preview_top_n
        )
        final_response = client.chat.completions.create(
            **build_chat_kwargs(
                final_messages,
                stream=stream,
                temperature=1,
                thinking_enabled_override=False,
            )
        )
        if stream:
            for chunk in final_response:
                if chunk.choices and chunk.choices[0].delta.content:
                    yield chunk.choices[0].delta.content
        else:
            if final_response and final_response.choices:
                answer = final_response.choices[0].message.content
                yield from _emit(
                    stream,
                    "⚠️ 第2次 LLM 返回空答案！"
                    if not answer
                    else f"\n{'=' * 60}\n✅ 最终答案:\n{'=' * 60}\n{answer}\n{'=' * 60}",
                )
            else:
                yield from _emit(stream, "⚠️ 第2次 LLM 返回空响应")
        step3_elapsed = time.perf_counter() - step3_t0
        total_elapsed = time.perf_counter() - total_t0
        yield from _emit(stream, f"\n⏱️ [第3步耗时] {step3_elapsed:.3f}s")
        yield from _emit(stream, f"⏱️ [总耗时] {total_elapsed:.3f}s")

    return _core() if stream else list(_core())


if __name__ == "__main__":
    run_search(
        query="扩展基础的宽高比",
        search_dir="./specs",
        preview_top_n=30,
        file_top_k=15,
        context_lines=0,
    )
    # run_search(query="筏板的最小厚度", search_dir="./specs", preview_top_n=30, file_top_k=10, context_lines=0)
    # run_search(query="门刚何时采用拦风绳", search_dir="./specs", preview_top_n=30, file_top_k=10, context_lines=0)
