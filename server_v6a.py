import asyncio
from fastapi import FastAPI, WebSocket, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import json, re, os, shutil
from pathlib import Path
from typing import List
from rg_search_v6a import *
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

async def create_chat_completion(**kwargs):
    return await asyncio.to_thread(get_client().chat.completions.create, **kwargs)

async def run_tool(func, **kwargs):
    return await asyncio.to_thread(func, **kwargs)

@app.get("/")
async def index(): return FileResponse('index_v6a.html')

@app.post("/api/set-folder")
async def set_folder(request: FolderPathRequest):
    """设置工作文件夹路径"""
    try:
        folder_path = SCRIPT_DIR / request.folder_path if request.folder_path != "." else SCRIPT_DIR
        
        if not folder_path.exists():
            return JSONResponse({"success": False, "message": "文件夹不存在"}, status_code=400)
        
        if not folder_path.is_dir():
            return JSONResponse({"success": False, "message": "路径不是文件夹"}, status_code=400)
        
        # 更新全局变量
        global TARGET, INDEX_DIR, MAIN_INDEX, FILE_MAP, DETAIL_TOC_CACHE, SEARCH_RESULT_CACHE
        TARGET = folder_path
        INDEX_DIR = TARGET / ".index"
        MAIN_INDEX = INDEX_DIR / "index.json"
        
        # 重新扫描文件
        FILE_MAP = {f: str(TARGET / f) for f in os.listdir(TARGET) 
                    if (TARGET / f).is_file() and f.endswith((".txt", ".md"))}
        
        # 清空缓存
        DETAIL_TOC_CACHE = {}
        SEARCH_RESULT_CACHE = {}
        
        print(f"📁 工作文件夹已更新: {TARGET}")
        print(f"📄 找到 {len(FILE_MAP)} 个文件")
        
        return JSONResponse({
            "success": True, 
            "message": f"已设置文件夹: {folder_path}",
            "file_count": len(FILE_MAP),
            "files": list(FILE_MAP.keys())
        })
    
    except Exception as e:
        print(f"❌ 设置文件夹失败: {e}")
        return JSONResponse({"success": False, "message": f"设置文件夹失败: {str(e)}"}, status_code=500)

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
        
        # 如果是当前工作目录，更新全局变量
        global TARGET, INDEX_DIR, MAIN_INDEX, FILE_MAP, DETAIL_TOC_CACHE, SEARCH_RESULT_CACHE
        if folder_path == TARGET or str(folder_path) == str(TARGET):
            INDEX_DIR = index_dir
            MAIN_INDEX = main_index
            DETAIL_TOC_CACHE = {}
            SEARCH_RESULT_CACHE = {}
        
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
        if not question: return await ws.send_json({'type': 'error', 'data': {'message': '问题不能为空'}})
        
        global SEARCH_RESULT_CACHE
        SEARCH_RESULT_CACHE = {}
        
        messages = [
            {"role": "system", "content": f"你是一个工程规范检索与解读专家。根据资料库内容回答，未提及的不要回答。\n\n【资料库全局目录】\n{get_global_toc_summary()}\n\n【工具】: get_document_toc(获取目录), execute_grep(搜索), read_file_range(读取原文)\n【纪律】: 1.必须调用工具查阅资料 2.必须明确引用依据 "},
            {"role": "user", "content": question}
        ]
        
        for turn in range(15):
            print(f"\n[第 {turn + 1} 轮]")
            await ws.send_json({'type': 'turn', 'data': {'turn': turn + 1}})
            await asyncio.sleep(0)
            
            try:
                response = await create_chat_completion(model=MODEL_NAME, messages=messages, tools=TOOLS_SCHEMA, tool_choice="auto", temperature=1)
            except Exception as e:
                print(f"❌ API错误: {e}")
                return await ws.send_json({'type': 'error', 'data': {'message': f'API错误: {e}'}})
            
            msg = response.choices[0].message
            messages.append(msg)
            
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
                print(f"✅ [最终答案]: {msg.content[:100]}...")
                await ws.send_json({'type': 'final_answer', 'data': {'content': msg.content}})
                await ws.close()
                return
        
        messages.append({"role": "user", "content": "已达到最大搜索次数，请立即总结回答"})
        final = await create_chat_completion(model=MODEL_NAME, messages=messages, temperature=1)
        await ws.send_json({'type': 'final_answer', 'data': {'content': final.choices[0].message.content}})
        await ws.close()
    except Exception as e:
        await ws.send_json({'type': 'error', 'data': {'message': str(e)}})
        await ws.close()

if __name__ == '__main__':
    import uvicorn
    print(f"\n🤖 模型: {MODEL_NAME} | 📁 文档: {len(FILE_MAP)} | 🌐 http://localhost:5000\n")
    uvicorn.run(app, host='0.0.0.0', port=5000)
