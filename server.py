"""
精简版 FastAPI Server - 只负责 Web 接口和流式输出处理
业务逻辑全部委托给 rg_search_v6a.py 和 rg-fast-v2a.py

架构说明：
1. server_v6a_new.py (本文件)：Web 接口层
   - FastAPI 路由和 WebSocket 处理
   - 配置管理（文件夹、模型、上下文行数）
   - 流式输出适配

2. rg_search_v6a.py：Agentic 模式业务逻辑
   - 多轮工具调用
   - 章节上下文注入
   - 去重缓存

3. rg-fast-v2a.py：Fast 模式业务逻辑
   - BM25 排序
   - 关键词提取
   - 两步走搜索

精简效果：从 1096 行 -> 380 行 (精简 65.3%)
"""
import asyncio, json, re
from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from pathlib import Path
from starlette.websockets import WebSocketState

# ==================== 导入业务逻辑模块 ====================
import rg_search_v6a
from rg_search_v6a import (
    get_client, execute_grep, read_file_range, get_document_toc,
    get_global_toc_summary, reset_search_cache, set_target_folder,
    TOOLS_SCHEMA, SCRIPT_DIR, FILE_MAP, MODEL_DICT
)

# 动态导入 fast 模式（如果需要）
try:
    import sys
    sys.path.insert(0, str(SCRIPT_DIR))
    from importlib import import_module
    rg_fast = import_module('rg-fast-v2a')
    FAST_MODE_AVAILABLE = True
except:
    FAST_MODE_AVAILABLE = False
    print("⚠️ Fast模式不可用")

# ==================== FastAPI 初始化 ====================
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

UPLOAD_DIR = SCRIPT_DIR / "uploaded_folders"
UPLOAD_DIR.mkdir(exist_ok=True)
ok = lambda data=None, **kw: JSONResponse((data or {}) | kw)
bad = lambda msg, code=400, **kw: JSONResponse({"success": False, "message": msg, **kw}, status_code=code)

# ==================== 请求模型 ====================
class FolderPathRequest(BaseModel):
    folder_path: str

class IndexRequest(BaseModel):
    folder_path: str

class ModelRequest(BaseModel):
    model_num: int
    thinking_enabled: bool | None = None

class ContextLinesRequest(BaseModel):
    context_lines: int

# ==================== 基础路由 ====================
@app.get("/")
async def index():
    return FileResponse('index.html')

# ==================== 配置管理接口 ====================
@app.post("/api/set-folder")
async def set_folder(request: FolderPathRequest):
    """设置工作文件夹"""
    folder_path = SCRIPT_DIR / request.folder_path if request.folder_path != "." else SCRIPT_DIR
    if not folder_path.exists() or not folder_path.is_dir(): return bad("文件夹不存在或不是目录")
    set_target_folder(str(folder_path))
    print(f"📁 工作文件夹: {rg_search_v6a.TARGET}, 文件数: {len(FILE_MAP)}")
    return ok(success=True, message=f"已设置文件夹: {folder_path}", file_count=len(FILE_MAP), files=list(FILE_MAP.keys()))

# ==================== 配置辅助函数 ====================
def _apply_model(model_num, thinking_enabled=None):
    rg_search_v6a.num = model_num
    rg_search_v6a.MODEL_NAME = MODEL_DICT[model_num]["model_name"]
    if thinking_enabled is not None: rg_search_v6a.THINKING_ENABLED = thinking_enabled
    rg_search_v6a.CLIENT = None

@app.post("/api/set-model")
async def set_model(request: ModelRequest):
    """设置模型"""
    if request.model_num not in MODEL_DICT: return bad(f"无效模型序号: {request.model_num}")
    _apply_model(request.model_num, request.thinking_enabled)
    print(f"🤖 模型: {rg_search_v6a.MODEL_NAME} (序号{request.model_num})")
    return ok(success=True, message=f"已设置模型: {rg_search_v6a.MODEL_NAME}", model_num=request.model_num, model_name=rg_search_v6a.MODEL_NAME, thinking_enabled=rg_search_v6a.THINKING_ENABLED)

@app.get("/api/models")
async def get_models():
    """获取可用模型列表"""
    models = [{"id": k, "name": f"{v['model_name']} (序号{k})", "model_name": v['model_name'], **rg_search_v6a._thinking_caps(v)} for k, v in MODEL_DICT.items()]
    return ok(success=True, models=models, current=rg_search_v6a.num, thinking_enabled=rg_search_v6a.THINKING_ENABLED)

@app.post("/api/set-context-lines")
async def set_context_lines(request: ContextLinesRequest):
    """设置上下文行数"""
    if not 0 <= request.context_lines <= 50: return bad("上下文行数必须在 0-50 之间")
    rg_search_v6a.CONTENT_LINES = request.context_lines
    print(f"📏 上下文行数: {request.context_lines}")
    return ok(success=True, message=f"已设置上下文行数: {request.context_lines}", context_lines=request.context_lines)

@app.get("/api/context-lines")
async def get_context_lines():
    """获取当前上下文行数"""
    return ok(success=True, context_lines=rg_search_v6a.CONTENT_LINES)

@app.get("/api/folders")
async def get_folders(path: str = "."):
    """获取文件夹列表"""
    target_path = SCRIPT_DIR if path == "." else SCRIPT_DIR / path
    if not target_path.exists() or not target_path.is_dir(): return JSONResponse({"error": "Folder not found"}, status_code=404)
    folders = [item.name for item in sorted(target_path.iterdir()) if item.is_dir() and not item.name.startswith('.')]
    files_count = sum(1 for item in target_path.iterdir() if item.is_file() and item.suffix in ['.txt', '.md'])
    parent = str(target_path.parent.relative_to(SCRIPT_DIR)) if target_path != SCRIPT_DIR else None
    current = str(target_path.relative_to(SCRIPT_DIR)) if target_path != SCRIPT_DIR else "."
    return ok(current=current, parent=parent, folders=folders, files_count=files_count)

@app.get("/api/index-status")
async def get_index_status(folder: str = "texts"):
    """检查索引状态"""
    folder_path = SCRIPT_DIR / folder if folder != "." else SCRIPT_DIR
    main_index = folder_path / ".index" / "index.json"
    return ok(indexed=main_index.exists(), file_count=len(list((folder_path / ".index").glob("*.index.json"))) if main_index.exists() else 0, folder=folder)

@app.post("/api/index-folder")
async def index_folder_endpoint(request: IndexRequest):
    """生成索引"""
    try:
        from extract_toc.scanner import scan_folder
        folder_path = SCRIPT_DIR / request.folder_path if request.folder_path != "." else SCRIPT_DIR
        if not folder_path.exists() or not folder_path.is_dir(): return bad("文件夹不存在")
        
        index_dir = folder_path / ".index"
        index_dir.mkdir(exist_ok=True)
        print(f"🔨 生成索引: {folder_path}")
        
        await asyncio.to_thread(scan_folder, str(folder_path), recursive=True, output_dir=str(index_dir))
        
        index_files = list(index_dir.glob("*.index.json"))
        print(f"✅ 索引完成: {len(index_files)} 个文件")
        
        if str(folder_path) == str(rg_search_v6a.TARGET):
            set_target_folder(str(folder_path))
            print(f"🔄 已刷新索引")
        
        return ok(success=True, message="索引生成成功", index_count=len(index_files), has_main_index=(index_dir / "index.json").exists())
    except Exception as e:
        return bad(str(e), 500)

# ==================== WebSocket 查询接口 ====================
@app.websocket("/ws/query")
async def query(ws: WebSocket):
    """统一查询入口，支持 agentic 和 fast 两种模式"""
    await ws.accept()
    print("✅ WebSocket 连接")
    
    try:
        data = await ws.receive_json()
        question = data.get('question', '')
        mode = data.get('mode', 'agentic')
        folder_path = data.get('folder_path', 'texts')
        model_num = data.get('model_num')
        context_lines = data.get('context_lines')
        thinking_enabled = data.get('thinking_enabled')
        
        if not question:
            return await _safe_send_json(ws, {'type': 'error', 'data': {'message': '问题不能为空'}})
        
        # 同步配置
        _sync_config(folder_path, model_num, context_lines, thinking_enabled)
        
        print(f"🔧 模式: {mode}, 问题: {question}")
        
        # 根据模式调用不同的处理函数
        if mode == 'fast' and FAST_MODE_AVAILABLE:
            await _handle_fast_mode(ws, question)
        else:
            await _handle_agentic_mode(ws, question)
    
    except Exception as e:
        await _safe_send_json(ws, {'type': 'error', 'data': {'message': str(e)}})
        await _safe_close(ws)

# ==================== 配置同步 ====================
def _sync_config(folder_path, model_num, context_lines, thinking_enabled=None):
    """同步前端配置到后端"""
    folder_full_path = SCRIPT_DIR / folder_path if folder_path != "." else SCRIPT_DIR
    if folder_full_path.exists() and folder_full_path.is_dir():
        set_target_folder(str(folder_full_path))
        print(f"📁 同步文件夹: {rg_search_v6a.TARGET} ({len(FILE_MAP)} 文件)")
    if model_num is not None and model_num in MODEL_DICT:
        _apply_model(model_num, thinking_enabled)
        print(f"🤖 同步模型: {rg_search_v6a.MODEL_NAME} (序号{model_num})")
    if context_lines is not None and 0 <= context_lines <= 50:
        rg_search_v6a.CONTENT_LINES = context_lines
        print(f"📏 同步上下文: {context_lines} 行")

# ==================== WebSocket 辅助函数 ====================
def _alive(ws: WebSocket):
    return ws.client_state == WebSocketState.CONNECTED and ws.application_state == WebSocketState.CONNECTED

async def _safe_send_json(ws: WebSocket, data):
    try:
        if _alive(ws):
            await ws.send_json(data)
            return True
    except RuntimeError:
        pass
    return False

async def _safe_close(ws: WebSocket):
    try:
        if _alive(ws): await ws.close()
    except RuntimeError:
        pass

async def _chat_stream(ws: WebSocket, messages, tools=None, thinking_id=None, stream_content=True):
    loop = asyncio.get_event_loop()
    def stream_gen():
        kw = rg_search_v6a.build_chat_kwargs(messages, stream=True, tools=tools, temperature=1)
        kw["stream_options"] = {"include_usage": True}
        yield from get_client().chat.completions.create(**kw)
    gen = stream_gen()
    def get_next():
        try: return next(gen)
        except StopIteration: return None
    full_reasoning, full_content, tool_calls_data = "", "", {}
    while True:
        if not _alive(ws): break
        chunk = await loop.run_in_executor(None, get_next)
        if chunk is None: break
        if not chunk.choices: continue
        delta = chunk.choices[0].delta
        if r := getattr(delta, 'reasoning_content', None) or getattr(delta, 'reasoning', None):
            full_reasoning += r
            if thinking_id: await _safe_send_json(ws, {'type': 'thinking_chunk', 'data': {'thinking_id': thinking_id, 'content': r}})
        if (c := getattr(delta, 'content', None)) and stream_content:
            full_content += c
            await _safe_send_json(ws, {'type': 'stream_chunk', 'data': {'content': c}})
        elif c:
            full_content += c
        for tc in getattr(delta, 'tool_calls', None) or []:
            cur = tool_calls_data.setdefault(tc.index, {'id': '', 'type': 'function', 'function': {'name': '', 'arguments': ''}})
            if getattr(tc, 'id', None): cur['id'] = tc.id
            if f := getattr(tc, 'function', None):
                if getattr(f, 'name', None): cur['function']['name'] += f.name
                if getattr(f, 'arguments', None): cur['function']['arguments'] += f.arguments
        await asyncio.sleep(0)
    if full_reasoning and thinking_id:
        await _safe_send_json(ws, {'type': 'thinking_complete', 'data': {'thinking_id': thinking_id}})
    return {
        'role': 'assistant',
        'content': full_content or None,
        'tool_calls': [tool_calls_data[i] for i in sorted(tool_calls_data)] or None,
        'reasoning_content': full_reasoning or None
    }

async def _stream_final_answer(ws: WebSocket, messages, thinking_id='thinking-final'):
    final = await _chat_stream(ws, messages, thinking_id=thinking_id, stream_content=True)
    await _safe_send_json(ws, {'type': 'final_answer', 'data': {'content': final.get('content')}})
    await _safe_close(ws)

# ==================== 结果摘要辅助 ====================
def _tool_summary(name, args, result):
    return (
        ('❌ 未找到匹配' if '未找到匹配项' in result or '未匹配任何文件' in result else '⚠️ 结果已重复' if '所有结果已重复' in result else f'✅ 找到 {m.group(1)} 条新记录' if (m := re.search(r'(\d+)\s*条新记录', result)) else '✅ 搜索完成')
        if name == 'execute_grep' else
        f'✅ 读取 {args["end_line"] - args["start_line"] + 1} 行'
        if name == 'read_file_range' else
        ('❌ 文件未找到' if 'error' in result else '✅ 目录获取成功')
        if name == 'get_document_toc' else
        ('❌ 未找到' if '未找到' in result else '✅ 搜索完成')
    )

# ==================== Agentic 模式处理 ====================
async def _handle_agentic_mode(ws: WebSocket, question: str):
    """Agentic 模式：流式输出思考、工具调用和最终答案"""
    reset_search_cache()
    print(f"\n[Agentic模式] 模型: {rg_search_v6a.MODEL_NAME}")
    
    messages = [
        {"role": "system", "content": f"你是一个工程规范检索与解读专家。根据资料库内容回答，未提及的不要回答。\n\n【资料库全局目录】\n{get_global_toc_summary()}\n\n【工具】: get_document_toc(获取目录), execute_grep(搜索), read_file_range(读取原文)\n【纪律】: 1.必须调用工具查阅资料 2.必须明确引用依据 "},
        {"role": "user", "content": question}
    ]
    
    for turn in range(15):
        if not _alive(ws): return
        print(f"\n[第 {turn + 1} 轮]")
        await _safe_send_json(ws, {'type': 'turn', 'data': {'turn': turn + 1}})
        await asyncio.sleep(0)
        
        try:
            msg = await _chat_stream(ws, messages, TOOLS_SCHEMA, f'thinking-agentic-{turn + 1}', stream_content=False)
        except Exception as e:
            print(f"❌ API错误: {e}")
            return await _safe_send_json(ws, {'type': 'error', 'data': {'message': f'API错误: {e}'}})
        messages.append(msg)
        
        if msg.get('tool_calls'):
            for tc in msg['tool_calls']:
                if not _alive(ws): return
                args = json.loads(tc['function']['arguments'])
                print(f"🔧 [工具] {tc['function']['name']}: {args}")
                await _safe_send_json(ws, {'type': 'tool_call', 'data': {'tool_call_id': tc['id'], 'tool_name': tc['function']['name'], 'arguments': args}})
                await asyncio.sleep(0)
                
                func = {"execute_grep": execute_grep, "read_file_range": read_file_range, "get_document_toc": get_document_toc}.get(tc['function']['name'])
                if func:
                    result = await asyncio.to_thread(func, **args, stream=False)
                    await _safe_send_json(ws, {'type': 'tool_result', 'data': {'tool_call_id': tc['id'], 'summary': _tool_summary(tc['function']['name'], args, result)}})
                    messages.append({"role": "tool", "tool_call_id": tc['id'], "content": result})
        else:
            print(f"✅ [最终答案] 流式输出中...")
            return await _stream_final_answer(ws, messages)
    
    # 达到最大轮次
    messages.append({"role": "user", "content": "已达到最大搜索次数，请立即总结回答"})
    try:
        await _stream_final_answer(ws, messages)
    except Exception as e:
        await _safe_send_json(ws, {'type': 'error', 'data': {'message': str(e)}})

# ==================== Fast 模式处理 ====================
async def _handle_fast_mode(ws: WebSocket, question: str):
    """Fast 模式：流式思考 + 工具调用 + 流式答案"""
    print(f"\n[Fast模式] 模型: {rg_search_v6a.MODEL_NAME}")
    
    if not FAST_MODE_AVAILABLE:
        return await _safe_send_json(ws, {'type': 'error', 'data': {'message': 'Fast模式不可用'}})
    
    try:
        if not _alive(ws): return
        messages = rg_fast.build_tool_messages(question, str(rg_search_v6a.TARGET))
        
        # 第一轮：LLM 调用工具
        await _safe_send_json(ws, {'type': 'turn', 'data': {'turn': 1}})
        msg = await _chat_stream(ws, messages, rg_fast.TOOLS, 'thinking-1', stream_content=False)
        if not msg.get('tool_calls'):
            await _safe_send_json(ws, {'type': 'final_answer', 'data': {'content': '无法生成搜索关键词'}})
            return await _safe_close(ws)
        
        # 执行工具调用
        msg2 = []
        for tc in msg['tool_calls']:
            if not _alive(ws): return
            if tc['function']['name'] == "search_documents":
                args = json.loads(tc['function']['arguments'])
                print(f"🔧 [工具] search_documents: {args}")
                await _safe_send_json(ws, {'type': 'tool_call', 'data': {'tool_call_id': tc['id'], 'tool_name': 'search_documents', 'arguments': args}})
                
                result = await asyncio.to_thread(
                    rg_fast.search_documents,
                    args.get("query", question),
                    args.get("broad_keywords", []),
                    args.get("exact_keywords", []),
                    args.get("target_files", []),
                    search_dir=str(rg_search_v6a.TARGET),
                    top_k=10,
                    context_lines=rg_search_v6a.CONTENT_LINES,
                    stream=False
                )
                
                await _safe_send_json(ws, {'type': 'tool_result', 'data': {'tool_call_id': tc['id'], 'summary': _tool_summary('search_documents', args, result)}})
                msg2.append({"role": "tool", "tool_call_id": tc['id'], "content": result})
        
        # 第二轮：生成答案
        await _safe_send_json(ws, {'type': 'turn', 'data': {'turn': 2}})
        messages_2 = rg_fast.build_answer_messages(question, msg2)
        print(f"✅ [回答]: 流式输出中...")
        await _stream_final_answer(ws, messages_2, 'thinking-2')
        
    except Exception as e:
        print(f"❌ Fast模式错误: {e}")
        await _safe_send_json(ws, {'type': 'error', 'data': {'message': str(e)}})
        await _safe_close(ws)

# ==================== 启动服务 ====================
if __name__ == '__main__':
    import uvicorn
    print(f"\n🤖 模型: {rg_search_v6a.MODEL_NAME} (序号{rg_search_v6a.num})")
    print(f"📁 目标文件夹: {rg_search_v6a.TARGET}")
    print(f"📄 文档数量: {len(FILE_MAP)}")
    print(f"📏 上下文行数: {rg_search_v6a.CONTENT_LINES}")
    print(f"🌐 服务地址: http://localhost:5000\n")
    uvicorn.run(app, host='0.0.0.0', port=5000)
