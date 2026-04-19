import asyncio
from fastapi import FastAPI, WebSocket, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json, re, os, shutil, subprocess
from pathlib import Path
from typing import List
import rg_search_v6a
from rg_search_v6a import (
    get_client, execute_grep, read_file_range, get_document_toc,
    get_global_toc_summary, reset_search_cache, set_target_folder,
    TOOLS_SCHEMA, SCRIPT_DIR, FILE_MAP, MODEL_DICT
)
from extract_toc.scanner import scan_folder

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# 临时上传目录
UPLOAD_DIR = SCRIPT_DIR / "uploaded_folders"
UPLOAD_DIR.mkdir(exist_ok=True)

# ==================== 请求模型 ====================
class FolderPathRequest(BaseModel):
    folder_path: str

class IndexRequest(BaseModel):
    folder_path: str

class ModelRequest(BaseModel):
    model_name: str

# ==================== Fast模式搜索函数 ====================
def search_documents_fast(query: str, broad_keywords: list, exact_keywords: list = None, 
                          search_dir: str = None, top_k: int = 15, context_lines: int = 0) -> str:
    """
    Fast模式：RG搜索 -> BM25排序 -> 添加上下文
    改编自 rg-fast.py
    """
    if search_dir is None:
        search_dir = str(rg_search_v6a.TARGET)
    
    exact_keywords = exact_keywords or []
    all_keywords = broad_keywords + exact_keywords
    
    print(f"\n🔍 [Fast模式] 原始问题: {query}")
    print(f"🔍 搜索目录: {search_dir}")
    print(f"🔍 关键词: 宽泛={broad_keywords}, 精确={exact_keywords}")
    
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
                    if not line:
                        continue
                    # 使用正则表达式解析，处理Windows路径中的盘符
                    # 格式: 文件路径:行号:内容
                    # Windows绝对路径: C:\path\file.txt:123:content
                    # 相对路径: path\file.txt:123:content
                    match = re.match(r'^(.+?):(\d+):(.*)$', line)
                    if match:
                        matching_lines.append({
                            'file': match.group(1),
                            'line_num': int(match.group(2)),
                            'content': match.group(3)
                        })
        except Exception as e:
            print(f"⚠️ 搜索 '{kw}' 出错: {e}")
    
    if not matching_lines:
        return "未找到匹配内容"
    
    print(f"✅ 找到 {len(matching_lines)} 个匹配行")
    
    # Step 2: BM25 排序
    try:
        from bm25_module import BM25
        corpus = [line['content'] for line in matching_lines]
        bm25 = BM25(corpus)
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
                        break
        
        matching_lines.sort(key=lambda x: x['score'], reverse=True)
    except ImportError:
        print("⚠️ BM25模块未找到，跳过排序")
    
    # 去重
    seen = set()
    deduplicated = []
    for line in matching_lines:
        key = (line['file'], line['line_num'])
        if key not in seen:
            deduplicated.append(line)
            seen.add(key)
    
    top_lines = deduplicated[:top_k]
    print(f"📊 去重后: {len(deduplicated)} 行, Top-{len(top_lines)} 匹配行")
    
    # Step 3: 添加上下文
    results = []
    for i, line in enumerate(top_lines, 1):
        context = _extract_context_fast(line['file'], line['line_num'], context_lines)
        results.append(f"--- {os.path.basename(line['file'])}:行{line['line_num']}  ---\n{context}\n")
    
    final_result = "\n".join(results)
    print(f"✅ 返回内容总长度: {len(final_result)} 字符")
    return final_result

def _extract_context_fast(filepath: str, line_num: int, context_lines: int = 0) -> str:
    """提取上下文"""
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        start = max(0, line_num - context_lines - 1)
        end = min(len(lines), line_num + context_lines)
        return "".join(f"{i+1}: {lines[i]}" for i in range(start, end))
    except:
        return ""

# ==================== Fast模式工具定义 ====================
TOOLS_FAST = [{
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

async def create_chat_completion(**kwargs):
    return await asyncio.to_thread(get_client().chat.completions.create, **kwargs)

async def run_tool(func, **kwargs):
    return await asyncio.to_thread(func, **kwargs)

@app.get("/")
async def index(): return FileResponse('index_v6a.html')

class ModelRequest(BaseModel):
    model_num: int

@app.post("/api/set-folder")
async def set_folder(request: FolderPathRequest):
    """设置工作文件夹路径"""
    try:
        folder_path = SCRIPT_DIR / request.folder_path if request.folder_path != "." else SCRIPT_DIR
        
        if not folder_path.exists():
            return JSONResponse({"success": False, "message": "文件夹不存在"}, status_code=400)
        
        if not folder_path.is_dir():
            return JSONResponse({"success": False, "message": "路径不是文件夹"}, status_code=400)
        
        set_target_folder(str(folder_path))
        
        print(f"📁 工作文件夹已更新: {rg_search_v6a.TARGET}")
        print(f"📄 找到 {len(rg_search_v6a.FILE_MAP)} 个文件")
        
        return JSONResponse({
            "success": True, 
            "message": f"已设置文件夹: {folder_path}",
            "file_count": len(rg_search_v6a.FILE_MAP),
            "files": list(rg_search_v6a.FILE_MAP.keys())
        })
    
    except Exception as e:
        print(f"❌ 设置文件夹失败: {e}")
        return JSONResponse({"success": False, "message": f"设置文件夹失败: {str(e)}"}, status_code=500)

@app.post("/api/set-model")
async def set_model(request: ModelRequest):
    """设置模型"""
    try:
        from rg_search_v6a import MODEL_DICT
        
        if request.model_num not in MODEL_DICT:
            return JSONResponse({
                "success": False, 
                "message": f"无效的模型序号: {request.model_num}"
            }, status_code=400)
        
        rg_search_v6a.num = request.model_num
        model_name = MODEL_DICT[request.model_num]["model_name"]
        print(f"🤖 模型已更新: {model_name} (序号: {request.model_num})")
        
        # 重置 CLIENT 以使用新模型
        rg_search_v6a.CLIENT = None
        
        return JSONResponse({
            "success": True,
            "message": f"已设置模型: {model_name}",
            "model_num": request.model_num,
            "model_name": model_name
        })
    
    except Exception as e:
        print(f"❌ 设置模型失败: {e}")
        return JSONResponse({"success": False, "message": f"设置模型失败: {str(e)}"}, status_code=500)

@app.get("/api/models")
async def get_models():
    """获取可用模型列表"""
    try:
        from rg_search_v6a import MODEL_DICT, num as current_num
        
        models = [
            {
                "id": num_key,
                "name": f"{info['model_name']} (序号{num_key})",
                "model_name": info['model_name']
            }
            for num_key, info in MODEL_DICT.items()
        ]
        
        return JSONResponse({
            "success": True,
            "models": models,
            "current": current_num
        })
    
    except Exception as e:
        print(f"❌ 获取模型列表失败: {e}")
        return JSONResponse({"success": False, "message": f"获取模型列表失败: {str(e)}"}, status_code=500)

@app.get("/api/folders")
async def get_folders(path: str = "."):
    """获取文件夹列表"""
    try:
        # 解析路径
        if path == ".":
            target_path = SCRIPT_DIR
        else:
            target_path = SCRIPT_DIR / path
        
        if not target_path.exists() or not target_path.is_dir():
            return JSONResponse({"error": "Folder not found"}, status_code=404)
        
        # 获取子文件夹
        folders = []
        files_count = 0
        
        for item in sorted(target_path.iterdir()):
            if item.is_dir() and not item.name.startswith('.'):
                folders.append(item.name)
            elif item.is_file() and item.suffix in ['.txt', '.md']:
                files_count += 1
        
        # 获取父文件夹
        parent = None
        if target_path != SCRIPT_DIR:
            parent_path = target_path.parent
            parent = str(parent_path.relative_to(SCRIPT_DIR)) if parent_path != SCRIPT_DIR else "."
        
        # 当前路径（相对于SCRIPT_DIR）
        current = str(target_path.relative_to(SCRIPT_DIR)) if target_path != SCRIPT_DIR else "."
        
        return JSONResponse({
            "current": current,
            "parent": parent,
            "folders": folders,
            "files_count": files_count
        })
    
    except Exception as e:
        print(f"❌ 获取文件夹列表失败: {e}")
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/api/index-status")
async def get_index_status(folder: str = "texts"):
    """检查文件夹索引状态"""
    try:
        folder_path = SCRIPT_DIR / folder if folder != "." else SCRIPT_DIR
        index_dir = folder_path / ".index"
        main_index = index_dir / "index.json"
        
        if main_index.exists():
            # 统计索引文件数量
            index_files = list(index_dir.glob("*.index.json"))
            return JSONResponse({
                "indexed": True,
                "file_count": len(index_files),
                "folder": folder
            })
        else:
            return JSONResponse({
                "indexed": False,
                "folder": folder
            })
    
    except Exception as e:
        print(f"❌ 检查索引状态失败: {e}")
        return JSONResponse({"indexed": False, "folder": folder})

@app.post("/api/index-folder")
async def index_folder_endpoint(request: IndexRequest):
    """对指定文件夹生成索引"""
    try:
        folder_path = SCRIPT_DIR / request.folder_path if request.folder_path != "." else SCRIPT_DIR
        
        if not folder_path.exists():
            return JSONResponse({"success": False, "message": "文件夹不存在"}, status_code=400)
        
        if not folder_path.is_dir():
            return JSONResponse({"success": False, "message": "路径不是文件夹"}, status_code=400)
        
        # 创建索引目录
        index_dir = folder_path / ".index"
        index_dir.mkdir(exist_ok=True)
        
        print(f"🔨 开始为文件夹生成索引: {folder_path}")
        
        # 调用扫描函数生成索引
        await asyncio.to_thread(scan_folder, str(folder_path), recursive=True, output_dir=str(index_dir))
        
        # 统计生成的索引文件
        index_files = list(index_dir.glob("*.index.json"))
        main_index = index_dir / "index.json"
        
        print(f"✅ 索引生成完成: {len(index_files)} 个文件")
        
        # 如果是当前工作目录，重新加载索引
        if str(folder_path) == str(rg_search_v6a.TARGET):
            # 重新设置目标文件夹以刷新索引
            set_target_folder(str(folder_path))
            print(f"🔄 已刷新当前工作目录的索引")
        
        return JSONResponse({
            "success": True,
            "message": f"索引生成成功",
            "index_count": len(index_files),
            "has_main_index": main_index.exists()
        })
    
    except Exception as e:
        print(f"❌ 索引生成失败: {e}")
        import traceback
        traceback.print_exc()
        return JSONResponse({"success": False, "message": f"索引生成失败: {str(e)}"}, status_code=500)

@app.websocket("/ws/query")
async def query(ws: WebSocket):
    await ws.accept()
    print("✅ WebSocket连接已建立")
    try:
        data = await ws.receive_json()
        print(f"📩 收到消息: {data}")
        question = data.get('question', '')
        mode = data.get('mode', 'agentic')  # 默认agentic模式
        folder_path = data.get('folder_path', 'texts')
        model_num = data.get('model_num', None)
        
        if not question: 
            return await ws.send_json({'type': 'error', 'data': {'message': '问题不能为空'}})
        
        # 同步文件夹设置
        try:
            folder_full_path = SCRIPT_DIR / folder_path if folder_path != "." else SCRIPT_DIR
            if folder_full_path.exists() and folder_full_path.is_dir():
                set_target_folder(str(folder_full_path))
                print(f"📁 已同步工作文件夹: {rg_search_v6a.TARGET}")
                print(f"📄 文件数量: {len(rg_search_v6a.FILE_MAP)}")
        except Exception as e:
            print(f"⚠️ 同步文件夹失败: {e}")
        
        # 同步模型设置
        if model_num is not None:
            from rg_search_v6a import MODEL_DICT
            if model_num in MODEL_DICT:
                rg_search_v6a.num = model_num
                rg_search_v6a.CLIENT = None  # 重置 CLIENT
                print(f"🤖 已同步模型: {MODEL_DICT[model_num]['model_name']} (序号: {model_num})")
        
        print(f"🔧 使用模式: {mode}")
        
        if mode == 'fast':
            # Fast模式：两步走
            await query_fast_mode(ws, question)
        else:
            # Agentic模式：原有逻辑
            await query_agentic_mode(ws, question)
    
    except Exception as e:
        await ws.send_json({'type': 'error', 'data': {'message': str(e)}})
        await ws.close()

async def query_fast_mode(ws: WebSocket, question: str):
    """Fast模式查询"""
    current_model = MODEL_DICT[rg_search_v6a.num]["model_name"]
    print(f"\n[Fast模式] 开始查询")
    print(f"🤖 使用模型: {current_model} (序号{rg_search_v6a.num})")
    print(f"📁 搜索目录: {rg_search_v6a.TARGET}")
    
    system_prompt = """你是文档检索专家。

工作流程：
1. 分析用户问题，提取关键词
2. 调用 search_documents 工具搜索
3. 基于搜索结果生成答案

注意：
- broad_keywords: 1-2个核心关键词
- exact_keywords: 1个最特殊、最关键的元关键词
- 必须先调用工具再回答"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question}
    ]
    
    # 第一轮：LLM 调用工具
    await ws.send_json({'type': 'turn', 'data': {'turn': 1}})
    await asyncio.sleep(0)
    
    try:
        response = await create_chat_completion(
            model=MODEL_DICT[rg_search_v6a.num]["model_name"], 
            messages=messages, 
            tools=TOOLS_FAST, 
            tool_choice="auto", 
            temperature=1
        )
    except Exception as e:
        print(f"❌ API错误: {e}")
        return await ws.send_json({'type': 'error', 'data': {'message': f'API错误: {e}'}})
    
    msg = response.choices[0].message
    
    if hasattr(msg, 'reasoning_content') and msg.reasoning_content:
        print(f"🧠 [思考]: {msg.reasoning_content[:100]}...")
        await ws.send_json({'type': 'thinking', 'data': {'content': msg.reasoning_content}})
        await asyncio.sleep(0)
    
    if not msg.tool_calls:
        print("⚠️ LLM 未生成关键词")
        await ws.send_json({'type': 'final_answer', 'data': {'content': '无法生成搜索关键词'}})
        await ws.close()
        return
    
    tool_results = []
    for tc in msg.tool_calls:
        if tc.function.name == "search_documents":
            args = json.loads(tc.function.arguments)
            print(f"🔧 [工具调用] search_documents: {args}")
            await ws.send_json({'type': 'tool_call', 'data': {
                'tool_call_id': tc.id, 
                'tool_name': 'search_documents', 
                'arguments': args
            }})
            await asyncio.sleep(0)
            
            result = await run_tool(
                search_documents_fast,
                query=args.get("query", question),
                broad_keywords=args.get("broad_keywords", []),
                exact_keywords=args.get("exact_keywords", []),
                search_dir=str(rg_search_v6a.TARGET),
                top_k=10,
                context_lines=10
            )
            
            summary = '❌ 未找到匹配' if '未找到匹配内容' in result else f'✅ 搜索完成'
            print(f"📤 [发送结果] {summary}")
            await ws.send_json({'type': 'tool_result', 'data': {'tool_call_id': tc.id, 'summary': summary}})
            
            tool_results.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result
            })
    
    # 第二轮：LLM 生成答案
    await ws.send_json({'type': 'turn', 'data': {'turn': 2}})
    await asyncio.sleep(0)
    
    system_prompt_2 = f"""你是根据文档总结回答问题的专家
根据文档已经检索到的信息为{tool_results}，根据信息回答问题，并给出明确依据，未提及的不要回答。"""

    messages_2 = [
        {"role": "system", "content": system_prompt_2},
        {"role": "user", "content": question}
    ]
    
    # 流式输出 - 用线程边产生边发送
    loop = asyncio.get_event_loop()
    full_content = ""
    
    def generator():
        stream = get_client().chat.completions.create(
            model=MODEL_DICT[rg_search_v6a.num]["model_name"],
            messages=messages_2,
            temperature=1,
            stream=True
        )
        for chunk in stream:
            yield chunk
    
    gen = generator()
    
    def get_next():
        try:
            return next(gen)
        except StopIteration:
            return None
    
    while True:
        chunk = await loop.run_in_executor(None, get_next)
        if chunk is None:
            break
        if chunk.choices and chunk.choices[0].delta.content:
            content = chunk.choices[0].delta.content
            full_content += content
            await ws.send_json({'type': 'stream_chunk', 'data': {'content': content}})
            await asyncio.sleep(0)
    
    print(f"✅ [最终答案]: {full_content[:100]}...")
    await ws.send_json({'type': 'final_answer', 'data': {'content': full_content}})
    await ws.close()

async def query_agentic_mode(ws: WebSocket, question: str):
    """Agentic模式查询（流式输出思考过程和最终答案）"""
    reset_search_cache()  # 重置搜索缓存
    
    current_model = MODEL_DICT[rg_search_v6a.num]["model_name"]
    print(f"\n[Agentic模式] 开始查询")
    print(f"🤖 使用模型: {current_model} (序号{rg_search_v6a.num})")
    print(f"📁 搜索目录: {rg_search_v6a.TARGET}")
    
    messages = [
        {"role": "system", "content": f"你是一个工程规范检索与解读专家。根据资料库内容回答，未提及的不要回答。\n\n【资料库全局目录】\n{get_global_toc_summary()}\n\n【工具】: get_document_toc(获取目录), execute_grep(搜索), read_file_range(读取原文)\n【纪律】: 1.必须调用工具查阅资料 2.必须明确引用依据 "},
        {"role": "user", "content": question}
    ]
    
    for turn in range(15):
        print(f"\n[第 {turn + 1} 轮]")
        await ws.send_json({'type': 'turn', 'data': {'turn': turn + 1}})
        await asyncio.sleep(0)
        
        try:
            response = await create_chat_completion(model=MODEL_DICT[rg_search_v6a.num]["model_name"], messages=messages, tools=TOOLS_SCHEMA, tool_choice="auto", temperature=1)
        except Exception as e:
            print(f"❌ API错误: {e}")
            return await ws.send_json({'type': 'error', 'data': {'message': f'API错误: {e}'}})
        
        msg = response.choices[0].message
        messages.append(msg)
        
        # 流式发送思考过程
        if hasattr(msg, 'reasoning_content') and msg.reasoning_content:
            print(f"🧠 [思考]: {msg.reasoning_content[:100]}...")
            await ws.send_json({'type': 'thinking', 'data': {'content': msg.reasoning_content}})
            await asyncio.sleep(0)
        
        if msg.tool_calls:
            for tc in msg.tool_calls:
                args = json.loads(tc.function.arguments)
                print(f"🔧 [工具调用] {tc.function.name}: {args}")
                await ws.send_json({'type': 'tool_call', 'data': {'tool_call_id': tc.id, 'tool_name': tc.function.name, 'arguments': args}})
                await asyncio.sleep(0)
                
                func = {"execute_grep": execute_grep, "read_file_range": read_file_range, "get_document_toc": get_document_toc}.get(tc.function.name)
                if func:
                    result = await run_tool(func, **args)
                    summary = ('❌ 未找到匹配' if '未找到匹配项' in result or '未匹配任何文件' in result else 
                               '⚠️ 结果已重复' if '所有结果已重复' in result else 
                               f'✅ 找到 {m.group(1)} 条新记录' if (m := re.search(r'(\d+)\s*条新记录', result)) else '✅ 搜索完成') if tc.function.name == 'execute_grep' else \
                              f'✅ 读取 {args["end_line"] - args["start_line"] + 1} 行' if tc.function.name == 'read_file_range' else \
                              ('❌ 文件未找到' if 'error' in result else '✅ 目录获取成功')
                    
                    print(f"📤 [发送结果] {summary}")
                    await ws.send_json({'type': 'tool_result', 'data': {'tool_call_id': tc.id, 'summary': summary}})
                    messages.append({"role": "tool", "tool_call_id": tc.id, "content": result})
        else:
            # 最终答案流式输出
            print(f"✅ [最终答案] 流式输出中...")
            await stream_final_answer(ws, messages)
            return
    
    messages.append({"role": "user", "content": "已达到最大搜索次数，请立即总结回答"})
    # 最终答案流式输出
    await stream_final_answer(ws, messages)


async def stream_final_answer(ws: WebSocket, messages: list):
    """流��输出最终答案"""
    loop = asyncio.get_event_loop()
    full_content = ""
    
    def stream_gen():
        stream = get_client().chat.completions.create(
            model=MODEL_DICT[rg_search_v6a.num]["model_name"],
            messages=messages,
            temperature=1,
            stream=True
        )
        for chunk in stream:
            yield chunk
    
    gen = stream_gen()
    
    def get_next():
        try:
            return next(gen)
        except StopIteration:
            return None
    
    while True:
        chunk = await loop.run_in_executor(None, get_next)
        if chunk is None:
            break
        if chunk.choices and chunk.choices[0].delta.content:
            content = chunk.choices[0].delta.content
            full_content += content
            await ws.send_json({'type': 'stream_chunk', 'data': {'content': content}})
            await asyncio.sleep(0)
    
    print(f"✅ [最终答案]: {full_content[:100]}...")
    await ws.send_json({'type': 'final_answer', 'data': {'content': full_content}})
    await ws.close()

if __name__ == '__main__':
    import uvicorn
    model_name = MODEL_DICT[rg_search_v6a.num]["model_name"]
    print(f"\n🤖 模型: {model_name} (序号{rg_search_v6a.num})")
    print(f"📁 目标文件夹: {rg_search_v6a.TARGET}")
    print(f"� 文档数量: {len(rg_search_v6a.FILE_MAP)}")
    print(f"🌐 服务地址: http://localhost:5000\n")
    uvicorn.run(app, host='0.0.0.0', port=5000)
