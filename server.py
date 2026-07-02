import asyncio, json, re, os
from importlib import import_module
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from starlette.websockets import WebSocketState
import rg_search_v6a
from rg_search_v6a import FILE_MAP, MODEL_DICT, SCRIPT_DIR, TOOLS_SCHEMA, EXTRACT_REFERENCES_SCHEMA, execute_grep, extract_references, get_client, get_document_toc, get_global_toc_summary, read_file_range, reset_search_cache, set_target_folder

try:
    import sys
    sys.path.insert(0, str(SCRIPT_DIR))
    rg_fast = import_module("rg-fast-v2a")
    FAST_MODE_AVAILABLE = True
except Exception:
    FAST_MODE_AVAILABLE = False
    print("⚠️ Fast模式不可用")

try:
    import hybrid_search
    HYBRID_MODE_AVAILABLE = True
except Exception:
    HYBRID_MODE_AVAILABLE = False
    print("⚠️ Hybrid模式不可用")
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
ok = lambda data=None, **kw: JSONResponse((data or {}) | kw)
bad = lambda msg, code=400, **kw: JSONResponse({"success": False, "message": msg, **kw}, status_code=code)
_path = lambda p=".": SCRIPT_DIR if p == "." else SCRIPT_DIR / p
_tool_funcs = {"execute_grep": execute_grep, "read_file_range": read_file_range, "get_document_toc": get_document_toc}
UPLOAD_DIR = SCRIPT_DIR / "uploaded_folders"
UPLOAD_DIR.mkdir(exist_ok=True)


# ==================== 基础状态与配置 ====================
def _alive(ws: WebSocket):
    return ws.client_state == ws.application_state == WebSocketState.CONNECTED


# ==================== 配置读写 ====================
def _cfg(key, value=Ellipsis, **kw):
    """统一处理 folder / model / context_lines / models 的读写。"""
    if key == "folder":
        folder = _path(value)
        if value is not Ellipsis:
            if not folder.is_dir(): raise ValueError("文件夹不存在或不是目录")
            set_target_folder(str(folder))
            print(f"📁 工作文件夹: {rg_search_v6a.TARGET}, 文件数: {len(FILE_MAP)}")
        return folder
    if key == "model":
        if value is not Ellipsis:
            if value not in MODEL_DICT: raise ValueError(f"无效模型序号: {value}")
            rg_search_v6a.num = value
            rg_search_v6a.MODEL_NAME = MODEL_DICT[value]["model_name"]
            if kw.get("thinking_enabled") is not None: rg_search_v6a.THINKING_ENABLED = kw["thinking_enabled"]
            rg_search_v6a.CLIENT = None
            print(f"🤖 模型: {rg_search_v6a.MODEL_NAME} (序号{value})")
        return {"model_num": rg_search_v6a.num, "model_name": rg_search_v6a.MODEL_NAME, "thinking_enabled": rg_search_v6a.THINKING_ENABLED}
    if key == "context_lines":
        if value is not Ellipsis:
            if not 0 <= value <= 50: raise ValueError("上下文行数必须在 0-50 之间")
            rg_search_v6a.CONTENT_LINES = value
            print(f"📏 上下文行数: {value}")
        return rg_search_v6a.CONTENT_LINES
    if key == "models":
        return {"models": [{"id": k, "name": f"{v['model_name']} (序号{k})", "model_name": v["model_name"], **rg_search_v6a._thinking_caps(v)} for k, v in MODEL_DICT.items()], "current": rg_search_v6a.num, "thinking_enabled": rg_search_v6a.THINKING_ENABLED}


def _cfg_api(key, field=None, message=None, extra=None, **kw):
    """把 _cfg 包装成 HTTP 接口，统一处理入参、返回和错误。"""
    async def endpoint(request: dict | None = None):
        try:
            request = request or {}
            value = _cfg(key, request.get(field), **{k: request.get(v) for k, v in kw.items()}) if field else _cfg(key)
            return ok(success=True, **({"message": message(value)} if message else {}), **(extra(value) if extra else value if isinstance(value, dict) else {key: value}))
        except ValueError as e:
            return bad(str(e))
    return endpoint


# ==================== WebSocket 输出 ====================
async def _safe_send_json(ws: WebSocket, data):
    try:
        if _alive(ws): await ws.send_json(data); return True
    except RuntimeError:
        pass
    return False


async def _safe_close(ws: WebSocket):
    try:
        if _alive(ws): await ws.close()
    except RuntimeError:
        pass


async def _chat_stream(ws: WebSocket, messages, tools=None, thinking_id=None, stream_content=True, tool_choice=None):
    """流式消费模型输出，聚合回答、思考内容和工具调用。"""
    loop = asyncio.get_event_loop()
    kwargs = rg_search_v6a.build_chat_kwargs(
        messages,
        stream=True,
        tools=tools,
        temperature=1,
        thinking_enabled_override=False if tool_choice is not None else None,
    )
    if tool_choice is not None:
        kwargs["tool_choice"] = tool_choice
    gen = get_client().chat.completions.create(**(kwargs | {"stream_options": {"include_usage": True}}))
    full_reasoning = full_content = ""
    tool_calls = {}
    while _alive(ws):
        chunk = await loop.run_in_executor(None, lambda: next(gen, None))
        if chunk is None:
            break
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        if r := getattr(delta, "reasoning_content", None) or getattr(delta, "reasoning", None):
            full_reasoning += r
            if thinking_id:
                await _safe_send_json(ws, {"type": "thinking_chunk", "data": {"thinking_id": thinking_id, "content": r}})
        if c := getattr(delta, "content", None):
            full_content += c
            if stream_content:
                await _safe_send_json(ws, {"type": "stream_chunk", "data": {"content": c}})
        for tc in getattr(delta, "tool_calls", None) or []:
            cur = tool_calls.setdefault(tc.index, {"id": "", "type": "function", "function": {"name": "", "arguments": ""}})
            if getattr(tc, "id", None):
                cur["id"] = tc.id
            if f := getattr(tc, "function", None):
                if getattr(f, "name", None):
                    cur["function"]["name"] += f.name
                if getattr(f, "arguments", None):
                    cur["function"]["arguments"] += f.arguments
        await asyncio.sleep(0)
    if full_reasoning and thinking_id:
        await _safe_send_json(ws, {"type": "thinking_complete", "data": {"thinking_id": thinking_id}})
    return {"role": "assistant", "content": full_content or None, "tool_calls": [tool_calls[i] for i in sorted(tool_calls)] or None, "reasoning_content": full_reasoning or None}


async def _stream_final_answer(ws: WebSocket, messages, thinking_id="thinking-final", extract_refs=True):
    final = await _chat_stream(ws, messages, thinking_id=thinking_id)
    final_content = final.get("content")
    await _safe_send_json(ws, {"type": "final_answer", "data": {"content": final_content}})
    
    if extract_refs and final_content:
        loop = asyncio.get_event_loop()
        refs = await loop.run_in_executor(None, extract_references, messages, final_content)
        if refs:
            await _safe_send_json(ws, {"type": "references", "data": refs})
    
    await _safe_close(ws)


# ==================== 工具调用 ====================
def _tool_summary(name, args, result):
    return (
        ("❌ 未找到匹配" if "未找到匹配项" in result or "未匹配任何文件" in result else "⚠️ 结果已重复" if "所有结果已重复" in result else f"✅ 找到 {m.group(1)} 条新记录" if (m := re.search(r"(\d+)\s*条新记录", result)) else "✅ 搜索完成")
        if name == "execute_grep"
        else f"✅ 读取 {args['end_line'] - args['start_line'] + 1} 行"
        if name == "read_file_range"
        else ("❌ 文件未找到" if "error" in result else "✅ 目录获取成功")
        if name == "get_document_toc"
        else ("❌ 未找到" if "未找到" in result else "✅ 搜索完成")
    )


async def _exec_tools(ws: WebSocket, tool_calls, runner):
    """统一执行工具调用，并向前端推送调用与结果摘要。"""
    out = []
    for tc in tool_calls or []:
        if not _alive(ws):
            return
        name, args = tc["function"]["name"], json.loads(tc["function"]["arguments"])
        print(f"🔧 [工具] {name}: {args}")
        await _safe_send_json(ws, {"type": "tool_call", "data": {"tool_call_id": tc["id"], "tool_name": name, "arguments": args}})
        result = await runner(name, args)
        if result is None:
            continue
        await _safe_send_json(ws, {"type": "tool_result", "data": {"tool_call_id": tc["id"], "summary": _tool_summary(name, args, result)}})
        out.append({"role": "tool", "tool_call_id": tc["id"], "content": result})
    return out


# ==================== HTTP 路由 ====================
@app.get("/")
async def index():
    return FileResponse("index3.html")


@app.get("/v1")
async def index_v1():
    return FileResponse("index.html")


@app.get("/v2")
async def index_v2():
    return FileResponse("index2.html")

app.post("/api/set-folder")(_cfg_api("folder", "folder_path", lambda v: f"已设置文件夹: {v}", lambda _: {"file_count": len(FILE_MAP), "files": list(FILE_MAP)}))
app.post("/api/set-model")(_cfg_api("model", "model_num", lambda v: f"已设置模型: {v['model_name']}", thinking_enabled="thinking_enabled"))
app.get("/api/models")(_cfg_api("models"))
app.post("/api/set-context-lines")(_cfg_api("context_lines", "context_lines", lambda v: f"已设置上下文行数: {v}"))
app.get("/api/context-lines")(_cfg_api("context_lines"))


@app.get("/api/folders")
async def get_folders(path: str = "."):
    target = _path(path)
    if not target.is_dir(): return JSONResponse({"error": "Folder not found"}, status_code=404)
    items = list(target.iterdir())
    return ok(current="." if target == SCRIPT_DIR else str(target.relative_to(SCRIPT_DIR)), parent=None if target == SCRIPT_DIR else str(target.parent.relative_to(SCRIPT_DIR)), folders=[x.name for x in sorted(items) if x.is_dir() and not x.name.startswith(".")], files_count=sum(1 for x in items if x.is_file() and x.suffix in [".txt", ".md"]))


@app.get("/api/index-status")
async def get_index_status(folder: str = "texts"):
    index_dir = _path(folder) / ".index"
    main = index_dir / "index.json"
    return ok(indexed=main.exists(), file_count=len(list(index_dir.glob("*.index.json"))) if main.exists() else 0, folder=folder)


@app.post("/api/index-folder")
async def index_folder_endpoint(request: dict):
    try:
        from extract_toc.scanner import scan_folder
        folder = _path(request.get("folder_path"))
        if not folder.is_dir(): return bad("文件夹不存在")
        index_dir = folder / ".index"; index_dir.mkdir(exist_ok=True)
        print(f"🔨 生成索引: {folder}")
        await asyncio.to_thread(scan_folder, str(folder), recursive=True, output_dir=str(index_dir))
        index_files = list(index_dir.glob("*.index.json"))
        print(f"✅ 索引完成: {len(index_files)} 个文件")
        if str(folder) == str(rg_search_v6a.TARGET): set_target_folder(str(folder)); print("🔄 已刷新索引")
        return ok(success=True, message="索引生成成功", index_count=len(index_files), has_main_index=(index_dir / "index.json").exists())
    except Exception as e:
        return bad(str(e), 500)


@app.post("/api/read-file-range")
async def read_file_range_endpoint(request: dict):
    try:
        filepath = request.get("filepath")
        start_line = request.get("start_line", 1)
        end_line = request.get("end_line", 10)
        
        # 更新 FILE_MAP 确保使用最新的文件映射
        current_file_map = {f: str(rg_search_v6a.TARGET / f) for f in os.listdir(rg_search_v6a.TARGET) 
                           if (rg_search_v6a.TARGET / f).is_file() and f.endswith((".txt", ".md"))}
        
        # 查找文件完整路径
        full_path = None
        
        # 1. 先尝试直接匹配文件名
        if filepath in current_file_map:
            full_path = current_file_map[filepath]
        else:
            # 2. 尝试模糊匹配（部分匹配）
            for fname, fpath in current_file_map.items():
                if filepath in fname:
                    full_path = fpath
                    break
        
        # 3. 如果还找不到，尝试直接作为路径
        if not full_path:
            potential_path = rg_search_v6a.TARGET / filepath
            if potential_path.exists():
                full_path = str(potential_path)
        
        if not full_path:
            print(f"❌ 文件未找到: {filepath}")
            print(f"   当前文件夹: {rg_search_v6a.TARGET}")
            print(f"   可用文件: {list(current_file_map.keys())}")
            return bad(f"文件未找到: {filepath}")
        
        # 复用现有的 read_file_range 函数
        content = await asyncio.to_thread(read_file_range, full_path, start_line, end_line, stream=False)
        return ok(success=True, content=content)
    except Exception as e:
        print(f"❌ 读取文件失败: {e}")
        return bad(str(e), 500)


# ==================== History Management ====================
HISTORY_FILE = SCRIPT_DIR / "history.json"

def _load_history():
    """加载 history.json"""
    if not HISTORY_FILE.exists():
        return {"version": 1, "items": []}
    try:
        with open(HISTORY_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"⚠️ 加载历史记录失败: {e}")
        return {"version": 1, "items": []}

def _save_history(data):
    """保存 history.json"""
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return True
    except Exception as e:
        print(f"❌ 保存历史记录失败: {e}")
        return False

def _calculate_similarity(text1: str, text2: str) -> float:
    """计算两个文本的相似度（使用简单的字符级方法）"""
    import difflib
    # 归一化：小写、去空白
    t1 = "".join(text1.lower().split())
    t2 = "".join(text2.lower().split())
    # 使用 SequenceMatcher 计算相似度
    ratio = difflib.SequenceMatcher(None, t1, t2).ratio()
    
    # 对于中文，额外使用字符级 Jaccard 相似度作为补充
    if any('\u4e00' <= c <= '\u9fff' for c in text1):
        # 2-gram Jaccard
        def get_ngrams(text, n=2):
            return set(text[i:i+n] for i in range(len(text)-n+1))
        g1 = get_ngrams(t1)
        g2 = get_ngrams(t2)
        if g1 or g2:
            jaccard = len(g1 & g2) / len(g1 | g2) if (g1 | g2) else 0
            # 取两者较高值
            ratio = max(ratio, jaccard)
    
    return ratio

@app.get("/api/history")
async def get_history():
    """获取所有历史记录"""
    history = _load_history()
    return ok(success=True, items=history.get("items", []))

@app.post("/api/history")
async def add_or_update_history(request: dict):
    """新增或更新一条历史记录"""
    try:
        item = request.get("item")
        if not item or not item.get("id"):
            return bad("缺少必要字段")
        
        history = _load_history()
        items = history.get("items", [])
        
        # 查找是否存在
        existing_idx = next((i for i, x in enumerate(items) if x.get("id") == item["id"]), None)
        
        if existing_idx is not None:
            # 更新
            items[existing_idx] = item
        else:
            # 新增到开头
            items.insert(0, item)
        
        # 限制数量（保留最近100条）
        history["items"] = items[:100]
        
        if _save_history(history):
            return ok(success=True, message="保存成功")
        else:
            return bad("保存失败", 500)
    except Exception as e:
        return bad(str(e), 500)

@app.delete("/api/history/{item_id}")
async def delete_history_item(item_id: str):
    """删除指定历史记录"""
    try:
        history = _load_history()
        items = history.get("items", [])
        history["items"] = [x for x in items if x.get("id") != item_id]
        
        if _save_history(history):
            return ok(success=True, message="删除成功")
        else:
            return bad("删除失败", 500)
    except Exception as e:
        return bad(str(e), 500)

@app.delete("/api/history")
async def clear_history():
    """清空所有历史记录"""
    try:
        history = {"version": 1, "items": []}
        if _save_history(history):
            return ok(success=True, message="清空成功")
        else:
            return bad("清空失败", 500)
    except Exception as e:
        return bad(str(e), 500)

@app.post("/api/history/match")
async def match_history(request: dict):
    """匹配相似历史记录"""
    try:
        question = request.get("question", "")
        folder = request.get("folder", "")
        threshold = request.get("threshold", 0.78)  # 默认阈值调整为 0.78
        
        if not question:
            return bad("问题不能为空")
        
        history = _load_history()
        items = history.get("items", [])
        
        # 只匹配同一个文件夹的历史
        candidates = [x for x in items if x.get("folder") == folder] if folder else items
        
        best_match = None
        best_score = 0
        
        for item in candidates:
            if not item.get("question"):
                continue
            score = _calculate_similarity(question, item["question"])
            if score > best_score:
                best_score = score
                best_match = item
        
        if best_match and best_score >= threshold:
            return ok(success=True, matched=True, score=round(best_score, 3), item=best_match)
        else:
            return ok(success=True, matched=False, score=round(best_score, 3) if best_score > 0 else 0)
    
    except Exception as e:
        return bad(str(e), 500)


# ==================== WebSocket 查询入口 ====================
@app.websocket("/ws/query")
async def query(ws: WebSocket):
    """接收前端请求，同步配置后分发到对应查询模式。"""
    await ws.accept()
    print("✅ WebSocket 连接")
    try:
        data = await ws.receive_json()
        question = data.get("question", "")
        if not question: return await _safe_send_json(ws, {"type": "error", "data": {"message": "问题不能为空"}})
        for key, val in [("folder", data.get("folder_path", "texts")), ("model", data.get("model_num")), ("context_lines", data.get("context_lines"))]:
            if val is None: continue
            try:
                _cfg(key, val, thinking_enabled=data.get("thinking_enabled"))
            except ValueError:
                pass
        mode = data.get("mode", "agentic")
        extract_refs = data.get("extract_references", True)  # 默认开启依据提取
        print(f"🔧 模式: {mode}, 问题: {question}, 提取依据: {extract_refs}")
        
        # 路由到对应模式
        if mode == "hybrid" and HYBRID_MODE_AVAILABLE:
            await _handle_hybrid(ws, question, extract_refs)
        elif mode == "fast" and FAST_MODE_AVAILABLE:
            await _handle_mode(ws, question, "fast", extract_refs)
        else:
            await _handle_mode(ws, question, "agentic", extract_refs)
    except Exception as e:
        await _safe_send_json(ws, {"type": "error", "data": {"message": str(e)}})
        await _safe_close(ws)


# ==================== 查询主流程 ====================
async def _handle_hybrid(ws: WebSocket, question: str, extract_refs: bool = True):
    """混合检索模式：fast第1轮 -> 评估 -> 够:生成答案 / 不够:转agent（性能优化版）"""
    print(f"\n[Hybrid模式] 模型: {rg_search_v6a.MODEL_NAME}")
    
    try:
        # ===== 阶段1: 快速检索（只执行第1轮：工具调用） =====
        await _safe_send_json(ws, {"type": "turn", "data": {"turn": 1}})
        print("⚡ [Hybrid阶段1] 快速检索 - 调用搜索工具")
        
        messages = rg_fast.build_tool_messages(question, str(rg_search_v6a.TARGET))
        tool_choice = {"type": "function", "function": {"name": "search_documents"}}
        
        msg = await _chat_stream(ws, messages, rg_fast.TOOLS, "thinking-hybrid-1", stream_content=False, tool_choice=tool_choice)
        
        if not msg.get("tool_calls"):
            await _safe_send_json(ws, {"type": "final_answer", "data": {"content": "快速检索失败：无法生成搜索关键词"}})
            return await _safe_close(ws)
        
        # 执行搜索工具
        tool_msgs = await _exec_tools(
            ws,
            msg.get("tool_calls"),
            lambda n, a: asyncio.to_thread(
                rg_fast.search_documents,
                a.get("query", question),
                a.get("broad_keywords", []),
                a.get("exact_keywords", []),
                a.get("target_files", []),
                search_dir=str(rg_search_v6a.TARGET),
                top_k=15,
                context_lines=rg_search_v6a.CONTENT_LINES,
                stream=False
            ) if n == "search_documents" else None
        )
        
        if tool_msgs is None:
            return
        
        fast_result_text = ''.join([m['content'] for m in tool_msgs if m.get('content')])
        
        # 限制评估输入长度，与 hybrid_search.py 保持一致
        max_eval_length = 2000
        eval_text = fast_result_text[:max_eval_length]
        if len(fast_result_text) > max_eval_length:
            eval_text += "\n...(已截断)..."
        
        # ===== 阶段2: 评估完整性 =====
        await _safe_send_json(ws, {"type": "turn", "data": {"turn": 2}})
        print(f"🤔 [Hybrid阶段2] 评估信息完整性 (结果长度: {len(fast_result_text)}字符, 评估: {len(eval_text)}字符)")
        
        loop = asyncio.get_event_loop()
        is_complete = await loop.run_in_executor(
            None,
            hybrid_search.evaluate_completeness,
            question,
            eval_text,  # 使用截断后的文本
            rg_fast.client,
            rg_fast.model_name
        )
        
        if is_complete:
            # 信息足够，生成答案（这是 fast 的第2轮，性能优化版）
            print("✅ 信息足够完整，生成最终答案")
            return await _stream_final_answer(
                ws,
                rg_fast.build_answer_messages(question, tool_msgs),
                "thinking-hybrid-final",
                extract_refs
            )
        
        # ===== 阶段3: 深度搜索（信息不足时才走到这里） =====
        print("⚠️ 信息不足，启动深度搜索")
        await _safe_send_json(ws, {"type": "turn", "data": {"turn": 3}})
        
        # 调用 agentic 模式
        reset_search_cache()
        messages_deep = [
            {"role": "system", "content": f"你是一个工程规范检索与解读专家。根据资料库内容回答，未提及的不要回答。\n\n【资料库全局目录】\n{get_global_toc_summary()}\n\n【工具】: get_document_toc(获取目录), execute_grep(搜索), read_file_range(读取原文)\n【纪律】: 1.必须调用工具查阅资料 2.必须明确引用依据 "},
            {"role": "user", "content": question},
        ]
        
        for turn in range(15):
            if not _alive(ws): return
            await _safe_send_json(ws, {"type": "turn", "data": {"turn": 3 + turn + 1}})
            
            msg = await _chat_stream(ws, messages_deep, TOOLS_SCHEMA, f"thinking-hybrid-deep-{turn + 1}", stream_content=False)
            messages_deep.append(msg)
            
            if not msg.get("tool_calls"):
                print("✅ [深度搜索完成] 流式输出最终答案")
                return await _stream_final_answer(ws, messages_deep, extract_refs=extract_refs)
            
            tool_msgs_deep = await _exec_tools(
                ws,
                msg.get("tool_calls"),
                lambda n, a: asyncio.to_thread(_tool_funcs[n], **a, stream=False) if n in _tool_funcs else None
            )
            
            if tool_msgs_deep is None: return
            messages_deep += tool_msgs_deep
        
        # 达到最大轮次
        messages_deep.append({"role": "user", "content": "已达到最大搜索次数，请立即总结回答"})
        await _stream_final_answer(ws, messages_deep, extract_refs=extract_refs)
        
    except Exception as e:
        print(f"❌ Hybrid模式错误: {e}")
        import traceback
        traceback.print_exc()
        await _safe_send_json(ws, {"type": "error", "data": {"message": f'Hybrid模式错误: {e}'}})
        await _safe_close(ws)


async def _handle_mode(ws: WebSocket, question: str, mode: str, extract_refs: bool = True):
    """agentic / fast 共用一条执行管线，只在消息和工具策略上分支。"""
    print(f"\n[{mode.capitalize()}模式] 模型: {rg_search_v6a.MODEL_NAME}")
    agentic = mode == "agentic"
    if agentic: reset_search_cache()
    messages = [
        {"role": "system", "content": f"你是一个工程规范检索与解读专家。根据资料库内容回答，未提及的不要回答。\n\n【资料库全局目录】\n{get_global_toc_summary()}\n\n【工具】: get_document_toc(获取目录), execute_grep(搜索), read_file_range(读取原文)\n【纪律】: 1.必须调用工具查阅资料 2.必须明确引用依据 "},
        {"role": "user", "content": question},
    ] if agentic else rg_fast.build_tool_messages(question, str(rg_search_v6a.TARGET))
    tools = TOOLS_SCHEMA if agentic else rg_fast.TOOLS
    try:
        for turn in range(15 if agentic else 2):
            if not _alive(ws): return
            await _safe_send_json(ws, {"type": "turn", "data": {"turn": turn + 1}})
            tool_choice = None if agentic or turn > 0 else {"type": "function", "function": {"name": "search_documents"}}
            msg = await _chat_stream(ws, messages, tools, f"thinking-{'agentic-' if agentic else ''}{turn + 1}", stream_content=False, tool_choice=tool_choice)
            if not agentic and not msg.get("tool_calls"):
                await _safe_send_json(ws, {"type": "final_answer", "data": {"content": "无法生成搜索关键词"}})
                return await _safe_close(ws)
            messages.append(msg)
            if agentic and not msg.get("tool_calls"):
                print("✅ [最终答案] 流式输出中...")
                return await _stream_final_answer(ws, messages, extract_refs=extract_refs)
            tool_msgs = await _exec_tools(
                ws,
                msg.get("tool_calls"),
                (lambda n, a: asyncio.to_thread(_tool_funcs[n], **a, stream=False) if n in _tool_funcs else None) if agentic
                else lambda n, a: asyncio.to_thread(rg_fast.search_documents, a.get("query", question), a.get("broad_keywords", []), a.get("exact_keywords", []), a.get("target_files", []), search_dir=str(rg_search_v6a.TARGET), top_k=15, context_lines=rg_search_v6a.CONTENT_LINES, stream=False) if n == "search_documents" else None,
            )
            if tool_msgs is None: return
            if not agentic:
                print("✅ [回答]: 流式输出中...")
                return await _stream_final_answer(ws, rg_fast.build_answer_messages(question, tool_msgs), "thinking-2", extract_refs)
            messages += tool_msgs
        messages.append({"role": "user", "content": "已达到最大搜索次数，请立即总结回答"})
        await _stream_final_answer(ws, messages, extract_refs=extract_refs)
    except Exception as e:
        print(f"❌ {mode.capitalize()}模式错误: {e}")
        await _safe_send_json(ws, {"type": "error", "data": {"message": f'API错误: {e}' if mode == 'agentic' else str(e)}})
        await _safe_close(ws)


if __name__ == "__main__":
    import uvicorn
    print(f"\n🤖 模型: {rg_search_v6a.MODEL_NAME} (序号{rg_search_v6a.num})")
    print(f"📁 目标文件夹: {rg_search_v6a.TARGET}")
    print(f"📄 文档数量: {len(FILE_MAP)}")
    print(f"📏 上下文行数: {rg_search_v6a.CONTENT_LINES}")
    print("\n🌐 服务地址:")
    print("   默认版本: http://localhost:5000")
    print("   v1版本: http://localhost:5000/v1")
    print("   v2版本: http://localhost:5000/v2 或 http://localhost:5000/single\n")
    uvicorn.run(app, host="0.0.0.0", port=5000)
