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
def get_available_files_fast(search_dir: str) -> list:
    """递归获取目录下所有 .txt 和 .md 文件名（不含路径）"""
    files = []
    for root, _, filenames in os.walk(search_dir):
        for filename in filenames:
            if filename.endswith(('.txt', '.md')):
                files.append(filename)
    return sorted(files)

def cut_by_punctuation_fast(paragraph: str) -> list[str]:
    """按中英文标点进行句子切分"""
    sents = re.findall(r'[^。！？.!?]+[。！？.!?]?', paragraph.strip())
    return [s.strip() for s in sents if s.strip()]

def chunk_file_fast(filepath: str, chunk_size: int = 512, overlap: int = 50) -> list:
    """按句子边界切分文件，返回语义较完整的 chunk"""
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
    except Exception as e:
        print(f"⚠️ 读取文件 {filepath} 失败: {e}")
        return []

    sentences = cut_by_punctuation_fast(content)
    chunks = []
    current_chunk = []
    current_length = 0
    start_pos = 0

    for sent in sentences:
        sent_len = len(sent)
        if current_length + sent_len > chunk_size and current_chunk:
            chunk_text = ''.join(current_chunk)
            chunks.append({
                'content': chunk_text,
                'file': filepath,
                'start_pos': start_pos,
                'type': 'chunk'
            })

            overlap_length = 0
            overlap_sents = []
            for s in reversed(current_chunk):
                if overlap_length + len(s) <= overlap:
                    overlap_sents.insert(0, s)
                    overlap_length += len(s)
                else:
                    break

            current_chunk = overlap_sents
            current_length = overlap_length
            start_pos += len(chunk_text) - overlap_length

        current_chunk.append(sent)
        current_length += sent_len

    if current_chunk:
        chunk_text = ''.join(current_chunk)
        chunks.append({
            'content': chunk_text,
            'file': filepath,
            'start_pos': start_pos,
            'type': 'chunk'
        })

    return chunks

def attach_bm25_scores_fast(candidates: list, query: str) -> list:
    """将 BM25 分数回填到原始候选项"""
    if not candidates:
        return candidates

    from bm25_module import BM25

    corpus = [item['content'] for item in candidates]
    bm25 = BM25(corpus)
    docs, scores = bm25.get_top_n(query, len(corpus))
    score_by_index = {int(doc_idx): score for (doc_idx, _), score in zip(docs, scores)}

    for idx, item in enumerate(candidates):
        bm25_score = score_by_index.get(idx, 0.0)
        item['bm25_score'] = bm25_score
        item['score'] = bm25_score

    return candidates


def get_candidate_key_fast(item: dict):
    """为候选项生成稳定唯一键，便于融合多路排序结果。"""
    if item['type'] == 'rg':
        return ('rg', item['file'], item['line_num'])
    return ('chunk', item['file'], item['start_pos'])

def search_documents_fast(query: str, broad_keywords: list, exact_keywords: list = None,
                          target_files: list = None, search_dir: str = None,
                          top_k: int = 15, context_lines: int = 0) -> str:
    """
    Fast模式：RG搜索 -> BM25排序 -> 添加上下文
    改编自 rg-fast.py
    """
    if search_dir is None:
        search_dir = str(rg_search_v6a.TARGET)
    
    exact_keywords = exact_keywords or []
    target_files = target_files or []
    all_keywords = broad_keywords + exact_keywords
    
    print(f"\n🔍 [Fast模式] 原始问题: {query}")
    print(f"🔍 搜索目录: {search_dir}")
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
                    if not line:
                        continue
                    match = re.match(r'^(.+?):(\d+):(.*)$', line)
                    if match:
                        matching_lines.append({
                            'file': match.group(1),
                            'line_num': int(match.group(2)),
                            'content': match.group(3),
                            'type': 'rg'
                        })
        except Exception as e:
            print(f"⚠️ 搜索 '{kw}' 出错: {e}")

    print(f"✅ RG 找到 {len(matching_lines)} 个匹配行")

    # Step 2: 读取目标文件并切块
    file_chunks = []
    for filename in target_files:
        found = False
        for root, _, filenames in os.walk(search_dir):
            if filename in filenames:
                filepath = os.path.join(root, filename)
                chunks = chunk_file_fast(filepath, chunk_size=512, overlap=50)
                file_chunks.extend(chunks)
                print(f"✅ 文件 {filename} 切分为 {len(chunks)} 个 chunks")
                found = True
                break
        if not found:
            print(f"⚠️ 目标文件未找到: {filename}")

    print(f"✅ 总共生成 {len(file_chunks)} 个文件 chunks")

    all_candidates = matching_lines + file_chunks
    if not all_candidates:
        return "未找到匹配内容"

    print(f"✅ 总候选内容: {len(all_candidates)} 条")

    # Step 3: BM25 排序
    try:
        attach_bm25_scores_fast(all_candidates, query)

        for item in all_candidates:
            content_lower = item['content'].lower()
            keyword_bonus = 0.0

            # 统计宽泛关键词命中数量
            broad_hit_count = sum(1 for kw in broad_keywords if kw.lower() in content_lower)
            if broad_hit_count == 2:
                keyword_bonus += 0.5
            elif broad_hit_count >= 3:
                keyword_bonus += 1.0

            if exact_keywords:
                for exact_kw in exact_keywords:
                    if exact_kw.lower() in content_lower:
                        keyword_bonus += 1.0
                        break

            item['keyword_bonus'] = keyword_bonus
            item['boosted_score'] = item['bm25_score'] + keyword_bonus

        bm25_sorted = sorted(all_candidates, key=lambda x: x['bm25_score'], reverse=True)
        boosted_sorted = sorted(all_candidates, key=lambda x: x['boosted_score'], reverse=True)
    except ImportError:
        print("⚠️ BM25模块未找到，跳过排序")
        bm25_sorted = all_candidates[:]
        boosted_sorted = all_candidates[:]
        for item in all_candidates:
            item.setdefault('bm25_score', item.get('score', 0.0))
            item.setdefault('keyword_bonus', 0.0)
            item.setdefault('boosted_score', item.get('score', 0.0))

    # RG优先去重：RG按文件+行号去重，chunk若与RG内容重叠则跳过
    rg_contents = set()
    for item in bm25_sorted:
        if item['type'] == 'rg':
            rg_contents.add(item['content'].strip().lower())

    seen_rg = set()
    deduplicated = []

    for item in bm25_sorted:
        if item['type'] == 'rg':
            key = get_candidate_key_fast(item)
            if key not in seen_rg:
                deduplicated.append(item)
                seen_rg.add(key)
        else:
            chunk_lower = item['content'].strip().lower()
            is_duplicate = False
            for rg_content in rg_contents:
                if rg_content in chunk_lower or chunk_lower in rg_content:
                    is_duplicate = True
                    break
            if not is_duplicate:
                deduplicated.append(item)

    dedup_map = {get_candidate_key_fast(item): item for item in deduplicated}
    bm25_dedup = [item for item in bm25_sorted if get_candidate_key_fast(item) in dedup_map]
    boosted_dedup = [item for item in boosted_sorted if get_candidate_key_fast(item) in dedup_map]

    bm25_pick_count = min(10, top_k)
    boosted_pick_count = min(5, top_k)
    merged_items = []
    selected_keys = set()

    def add_candidates(candidates: list, source_name: str, limit: int = None):
        added = 0
        for candidate in candidates:
            key = get_candidate_key_fast(candidate)
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

    # Step 4: 添加上下文（仅 RG）或直接返回 chunk
    results = []
    for item in top_items:
        if item['type'] == 'rg':
            context = _extract_context_fast(item['file'], item['line_num'], context_lines)
            results.append(f"--- {os.path.basename(item['file'])}:行{item['line_num']} [RG] ---\n{context}\n")
        else:
            results.append(f"--- {os.path.basename(item['file'])}:位置{item['start_pos']} [CHUNK] ---\n{item['content']}\n")
    
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

async def create_chat_completion(**kwargs):
    return await asyncio.to_thread(get_client().chat.completions.create, **kwargs)

async def run_tool(func, **kwargs):
    return await asyncio.to_thread(func, **kwargs)

@app.get("/")
async def index(): return FileResponse('index_v6a.html')

class ModelRequest(BaseModel):
    model_num: int

class ContextLinesRequest(BaseModel):
    context_lines: int

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

@app.post("/api/set-context-lines")
async def set_context_lines(request: ContextLinesRequest):
    """设置上下文行数"""
    try:
        if request.context_lines < 0 or request.context_lines > 50:
            return JSONResponse({
                "success": False,
                "message": "上下文行数必须在 0-50 之间"
            }, status_code=400)
        
        rg_search_v6a.CONTENT_LINES = request.context_lines
        print(f"📏 上下文行数已更新: {request.context_lines}")
        
        return JSONResponse({
            "success": True,
            "message": f"已设置上下文行数: {request.context_lines}",
            "context_lines": request.context_lines
        })
    
    except Exception as e:
        print(f"❌ 设置上下文行数失败: {e}")
        return JSONResponse({"success": False, "message": f"设置上下文行数失败: {str(e)}"}, status_code=500)

@app.get("/api/context-lines")
async def get_context_lines():
    """获取当前上下文行数"""
    try:
        return JSONResponse({
            "success": True,
            "context_lines": rg_search_v6a.CONTENT_LINES
        })
    
    except Exception as e:
        print(f"❌ 获取上下文行数失败: {e}")
        return JSONResponse({"success": False, "message": f"获取上下文行数失败: {str(e)}"}, status_code=500)

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
        context_lines = data.get('context_lines', None)
        
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
        
        # 同步上下文行数设置
        if context_lines is not None:
            if 0 <= context_lines <= 50:
                rg_search_v6a.CONTENT_LINES = context_lines
                print(f"📏 已同步上下文行数: {context_lines}")
        
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
    available_files = get_available_files_fast(str(rg_search_v6a.TARGET))
    print(f"\n[Fast模式] 开始查询")
    print(f"🤖 使用模型: {current_model} (序号{rg_search_v6a.num})")
    print(f"📁 搜索目录: {rg_search_v6a.TARGET}")
    
    system_prompt = f"""你是文档检索专家。

可用文件列表：
{', '.join(available_files)}

工作流程：
1. 分析用户问题，提取关键词和目标文件
2. 调用 search_documents 工具搜索
3. 基于搜索结果生成答案

注意：
- broad_keywords: 1-2个核心关键词
- exact_keywords: 1个最特殊、最关键的元关键词
- target_files: 从可用文件列表中选择1-3个最可能包含答案的文件
- 必须先调用工具再回答"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question}
    ]
    
    # 第一轮：LLM 调用工具（流式获取思考过程）
    await ws.send_json({'type': 'turn', 'data': {'turn': 1}})
    await asyncio.sleep(0)
    
    try:
        # 使用流式API获取思考过程
        loop = asyncio.get_event_loop()
        thinking_id = 'thinking-1'
        
        def stream_gen():
            stream = get_client().chat.completions.create(
                model=MODEL_DICT[rg_search_v6a.num]["model_name"],
                messages=messages,
                tools=TOOLS_FAST,
                tool_choice="auto",
                temperature=1,
                stream=True,
                stream_options={"include_usage": True}
            )
            for chunk in stream:
                yield chunk
        
        gen = stream_gen()
        
        def get_next():
            try:
                return next(gen)
            except StopIteration:
                return None
        
        # 收集完整响应
        full_reasoning = ""
        tool_calls_data = {}
        
        while True:
            chunk = await loop.run_in_executor(None, get_next)
            if chunk is None:
                break
            
            # 检查 choices 是否存在
            if not chunk.choices or len(chunk.choices) == 0:
                continue
            
            choice = chunk.choices[0]
            
            # 流式发送思考内容
            if hasattr(choice.delta, 'reasoning_content') and choice.delta.reasoning_content:
                reasoning_chunk = choice.delta.reasoning_content
                full_reasoning += reasoning_chunk
                await ws.send_json({
                    'type': 'thinking_chunk',
                    'data': {
                        'thinking_id': thinking_id,
                        'content': reasoning_chunk
                    }
                })
                await asyncio.sleep(0)
            
            # 收集工具调用
            if hasattr(choice.delta, 'tool_calls') and choice.delta.tool_calls:
                for tc_delta in choice.delta.tool_calls:
                    idx = tc_delta.index
                    if idx not in tool_calls_data:
                        tool_calls_data[idx] = {
                            'id': tc_delta.id or '',
                            'function_name': '',
                            'arguments': ''
                        }
                    if tc_delta.id:
                        tool_calls_data[idx]['id'] = tc_delta.id
                    if tc_delta.function and tc_delta.function.name:
                        tool_calls_data[idx]['function_name'] = tc_delta.function.name
                    if tc_delta.function and tc_delta.function.arguments:
                        tool_calls_data[idx]['arguments'] += tc_delta.function.arguments
        
        # 完成思考步骤
        if full_reasoning:
            print(f"🧠 [思考]: {full_reasoning[:100]}...")
            await ws.send_json({
                'type': 'thinking_complete',
                'data': {'thinking_id': thinking_id}
            })
            await asyncio.sleep(0)
        
        # 重建工具调用
        tool_calls = []
        for idx in sorted(tool_calls_data.keys()):
            tc_data = tool_calls_data[idx]
            tool_calls.append({
                'id': tc_data['id'],
                'function': {
                    'name': tc_data['function_name'],
                    'arguments': tc_data['arguments']
                }
            })
    
    except Exception as e:
        print(f"❌ API错误: {e}")
        return await ws.send_json({'type': 'error', 'data': {'message': f'API错误: {e}'}})
    
    msg_tool_calls = tool_calls
    
    if not msg_tool_calls:
        print("⚠️ LLM 未生成关键词")
        await ws.send_json({'type': 'final_answer', 'data': {'content': '无法生成搜索关键词'}})
        await ws.close()
        return
    
    tool_results = []
    for tc in msg_tool_calls:
        if tc['function']['name'] == "search_documents":
            args = json.loads(tc['function']['arguments'])
            print(f"🔧 [工具调用] search_documents: {args}")
            await ws.send_json({'type': 'tool_call', 'data': {
                'tool_call_id': tc['id'], 
                'tool_name': 'search_documents', 
                'arguments': args
            }})
            await asyncio.sleep(0)
            
            result = await run_tool(
                search_documents_fast,
                query=args.get("query", question),
                broad_keywords=args.get("broad_keywords", []),
                exact_keywords=args.get("exact_keywords", []),
                target_files=args.get("target_files", []),
                search_dir=str(rg_search_v6a.TARGET),
                top_k=10,
                context_lines=rg_search_v6a.CONTENT_LINES
            )
            
            summary = '❌ 未找到匹配' if '未找到匹配内容' in result else f'✅ 搜索完成'
            print(f"📤 [发送结果] {summary}")
            await ws.send_json({'type': 'tool_result', 'data': {'tool_call_id': tc['id'], 'summary': summary}})
            
            tool_results.append({
                "role": "tool",
                "tool_call_id": tc['id'],
                "content": result
            })
    
    # 第二轮：LLM 生成答案（流式输出思考+答案）
    await ws.send_json({'type': 'turn', 'data': {'turn': 2}})
    await asyncio.sleep(0)
    
    system_prompt_2 = f"""你是根据文档总结回答问题的专家
根据文档已经检索到的信息为{tool_results}，根据信息回答问题，并给出明确依据，未提及的不要回答。"""

    messages_2 = [
        {"role": "system", "content": system_prompt_2},
        {"role": "user", "content": question}
    ]
    
    # 流式输出思考过程和答案
    loop = asyncio.get_event_loop()
    full_content = ""
    full_reasoning_2 = ""
    thinking_id_2 = 'thinking-2'
    
    def generator():
        stream = get_client().chat.completions.create(
            model=MODEL_DICT[rg_search_v6a.num]["model_name"],
            messages=messages_2,
            temperature=1,
            stream=True,
            stream_options={"include_usage": True}
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
        
        # 检查 choices 是否存在
        if not chunk.choices or len(chunk.choices) == 0:
            continue
        
        choice = chunk.choices[0]
        
        # 流式发送思考内容
        if hasattr(choice.delta, 'reasoning_content') and choice.delta.reasoning_content:
            reasoning_chunk = choice.delta.reasoning_content
            full_reasoning_2 += reasoning_chunk
            await ws.send_json({
                'type': 'thinking_chunk',
                'data': {
                    'thinking_id': thinking_id_2,
                    'content': reasoning_chunk
                }
            })
            await asyncio.sleep(0)
        
        # 流式发送答案内容
        if hasattr(choice.delta, 'content') and choice.delta.content:
            content = choice.delta.content
            full_content += content
            await ws.send_json({'type': 'stream_chunk', 'data': {'content': content}})
            await asyncio.sleep(0)
    
    # 完成思考步骤
    if full_reasoning_2:
        await ws.send_json({
            'type': 'thinking_complete',
            'data': {'thinking_id': thinking_id_2}
        })
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
            # 使用流式API获取思考过程
            loop = asyncio.get_event_loop()
            thinking_id = f'thinking-agentic-{turn + 1}'
            
            def stream_gen():
                stream = get_client().chat.completions.create(
                    model=MODEL_DICT[rg_search_v6a.num]["model_name"],
                    messages=messages,
                    tools=TOOLS_SCHEMA,
                    tool_choice="auto",
                    temperature=1,
                    stream=True,
                    stream_options={"include_usage": True}
                )
                for chunk in stream:
                    yield chunk
            
            gen = stream_gen()
            
            def get_next():
                try:
                    return next(gen)
                except StopIteration:
                    return None
            
            # 收集完整响应
            full_reasoning = ""
            full_content = ""
            tool_calls_data = {}
            finish_reason = None
            
            while True:
                chunk = await loop.run_in_executor(None, get_next)
                if chunk is None:
                    break
                
                # 检查 choices 是否存在
                if not chunk.choices or len(chunk.choices) == 0:
                    continue
                
                choice = chunk.choices[0]
                
                # 流式发送思考内容
                if hasattr(choice.delta, 'reasoning_content') and choice.delta.reasoning_content:
                    reasoning_chunk = choice.delta.reasoning_content
                    full_reasoning += reasoning_chunk
                    await ws.send_json({
                        'type': 'thinking_chunk',
                        'data': {
                            'thinking_id': thinking_id,
                            'content': reasoning_chunk
                        }
                    })
                    await asyncio.sleep(0)
                
                # 收集工具调用
                if hasattr(choice.delta, 'tool_calls') and choice.delta.tool_calls:
                    for tc_delta in choice.delta.tool_calls:
                        idx = tc_delta.index
                        if idx not in tool_calls_data:
                            tool_calls_data[idx] = {
                                'id': tc_delta.id or '',
                                'function_name': '',
                                'arguments': ''
                            }
                        if tc_delta.id:
                            tool_calls_data[idx]['id'] = tc_delta.id
                        if tc_delta.function and tc_delta.function.name:
                            tool_calls_data[idx]['function_name'] = tc_delta.function.name
                        if tc_delta.function and tc_delta.function.arguments:
                            tool_calls_data[idx]['arguments'] += tc_delta.function.arguments
                
                # 收集内容（用于判断是否是最终答案）
                if hasattr(choice.delta, 'content') and choice.delta.content:
                    full_content += choice.delta.content
                
                # 获取结束原因
                if hasattr(choice, 'finish_reason') and choice.finish_reason:
                    finish_reason = choice.finish_reason
            
            # 完成思考步骤
            if full_reasoning:
                print(f"🧠 [思考]: {full_reasoning[:100]}...")
                await ws.send_json({
                    'type': 'thinking_complete',
                    'data': {'thinking_id': thinking_id}
                })
                await asyncio.sleep(0)
            
            # 重建工具调用
            tool_calls = []
            for idx in sorted(tool_calls_data.keys()):
                tc_data = tool_calls_data[idx]
                tool_calls.append({
                    'id': tc_data['id'],
                    'type': 'function',
                    'function': {
                        'name': tc_data['function_name'],
                        'arguments': tc_data['arguments']
                    }
                })
            
            # 构建消息对象（转换为字典格式）
            assistant_message = {"role": "assistant"}
            
            # 添加思考内容（如果有）
            if full_reasoning:
                assistant_message["reasoning_content"] = full_reasoning
            
            # 添加工具调用（如果有）
            if tool_calls:
                # 转换为 OpenAI API 格式
                formatted_tool_calls = []
                for tc in tool_calls:
                    formatted_tool_calls.append({
                        "id": tc['id'],
                        "type": "function",
                        "function": {
                            "name": tc['function']['name'],
                            "arguments": tc['function']['arguments']
                        }
                    })
                assistant_message["tool_calls"] = formatted_tool_calls
            
            # 添加内容（如果有）
            if full_content:
                assistant_message["content"] = full_content
            
            messages.append(assistant_message)
            
        except Exception as e:
            print(f"❌ API错误: {e}")
            import traceback
            traceback.print_exc()
            return await ws.send_json({'type': 'error', 'data': {'message': f'API错误: {e}'}})
        
        if tool_calls:
            for tc in tool_calls:
                args = json.loads(tc['function']['arguments'])
                print(f"🔧 [工具调用] {tc['function']['name']}: {args}")
                await ws.send_json({'type': 'tool_call', 'data': {'tool_call_id': tc['id'], 'tool_name': tc['function']['name'], 'arguments': args}})
                await asyncio.sleep(0)
                
                func = {"execute_grep": execute_grep, "read_file_range": read_file_range, "get_document_toc": get_document_toc}.get(tc['function']['name'])
                if func:
                    result = await run_tool(func, **args)
                    summary = ('❌ 未找到匹配' if '未找到匹配项' in result or '未匹配任何文件' in result else 
                               '⚠️ 结果已重复' if '所有结果已重复' in result else 
                               f'✅ 找到 {m.group(1)} 条新记录' if (m := re.search(r'(\d+)\s*条新记录', result)) else '✅ 搜索完成') if tc['function']['name'] == 'execute_grep' else \
                              f'✅ 读取 {args["end_line"] - args["start_line"] + 1} 行' if tc['function']['name'] == 'read_file_range' else \
                              ('❌ 文件未找到' if 'error' in result else '✅ 目录获取成功')
                    
                    print(f"📤 [发送结果] {summary}")
                    await ws.send_json({'type': 'tool_result', 'data': {'tool_call_id': tc['id'], 'summary': summary}})
                    messages.append({"role": "tool", "tool_call_id": tc['id'], "content": result})
        else:
            # 最终答案流式输出
            print(f"✅ [最终答案] 流式输出中...")
            await stream_final_answer(ws, messages)
            return
    
    messages.append({"role": "user", "content": "已达到最大搜索次数，请立即总结回答"})
    # 最终答案流式输出
    await stream_final_answer(ws, messages)


async def stream_final_answer(ws: WebSocket, messages: list):
    """流式输出最终答案（包括思考过程）"""
    loop = asyncio.get_event_loop()
    full_content = ""
    full_reasoning = ""
    thinking_id = 'thinking-final'
    
    def stream_gen():
        stream = get_client().chat.completions.create(
            model=MODEL_DICT[rg_search_v6a.num]["model_name"],
            messages=messages,
            temperature=1,
            stream=True,
            stream_options={"include_usage": True}
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
        
        # 检查 choices 是否存在
        if not chunk.choices or len(chunk.choices) == 0:
            continue
        
        choice = chunk.choices[0]
        
        # 流式发送思考内容
        if hasattr(choice.delta, 'reasoning_content') and choice.delta.reasoning_content:
            reasoning_chunk = choice.delta.reasoning_content
            full_reasoning += reasoning_chunk
            await ws.send_json({
                'type': 'thinking_chunk',
                'data': {
                    'thinking_id': thinking_id,
                    'content': reasoning_chunk
                }
            })
            await asyncio.sleep(0)
        
        # 流式发送答案内容
        if hasattr(choice.delta, 'content') and choice.delta.content:
            content = choice.delta.content
            full_content += content
            await ws.send_json({'type': 'stream_chunk', 'data': {'content': content}})
            await asyncio.sleep(0)
    
    # 完成思考步骤
    if full_reasoning:
        await ws.send_json({
            'type': 'thinking_complete',
            'data': {'thinking_id': thinking_id}
        })
        await asyncio.sleep(0)
    
    print(f"✅ [最终答案]: {full_content[:100]}...")
    await ws.send_json({'type': 'final_answer', 'data': {'content': full_content}})
    await ws.close()

if __name__ == '__main__':
    import uvicorn
    model_name = MODEL_DICT[rg_search_v6a.num]["model_name"]
    print(f"\n🤖 模型: {model_name} (序号{rg_search_v6a.num})")
    print(f"📁 目标文件夹: {rg_search_v6a.TARGET}")
    print(f"📄 文档数量: {len(rg_search_v6a.FILE_MAP)}")
    print(f"📏 上下文行数: {rg_search_v6a.CONTENT_LINES}")
    print(f"🌐 服务地址: http://localhost:5000\n")
    uvicorn.run(app, host='0.0.0.0', port=5000)
