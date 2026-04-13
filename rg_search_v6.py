import subprocess
import json
import os
import sys
from pathlib import Path
from openai import OpenAI
from dotenv import load_dotenv

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from extract_toc.scanner import scan_folder

load_dotenv()
'''
nvidia model 不如原生的好
qwen/qwen3.5-122b-a10b 调用工具经常没调用就退出
minimaxai/minimax-m2.5 可以比较慢，有时弄得关键词正则不太行导致返回0行，但是好像不太影响，还行，就是慢
moonshotai/kimi-k2.5 调用工具经常没调用就退出

kimi模型
kimi-k2-turbo-preview  价格太贵
kimi-k2.5

modelscope
Qwen/Qwen3-235B-A22B-Instruct-2507
Qwen/Qwen3.5-122B-A10B
moonshotai/Kimi-K2.5
总的来讲qwen的或者原生的都行，英伟达的不太好
'''
index=4
# ================= 配置区 =================
model_dict={1:{'factory_name':'kimi','base_url':'https://api.moonshot.cn/v1','api_key':'kimi_key','model_name':'kimi-k2.5'},
            2:{'factory_name':'nvidia','base_url':'https://integrate.api.nvidia.com/v1','api_key':'nvidia_key','model_name':'minimaxai/minimax-m2.5'},
            3:{'factory_name':'modelscope','base_url':'https://api-inference.modelscope.cn/v1','api_key':'modelscope_key','model_name':'Qwen/Qwen3-235B-A22B-Instruct-2507'},
            4:{'factory_name':'openai','base_url':'https://aigw-jnzs5.cucloud.cn:8443/v1','api_key':'OPENAI_API_KEY','model_name':'MiniMax-M2.5'},
            }
CLIENT = None

MODEL_NAME = model_dict[index]['model_name']

TARGET_FOLDER = SCRIPT_DIR / "texts"
INDEX_DIR = TARGET_FOLDER / ".index"
MAIN_INDEX_PATH = INDEX_DIR / "index.json"
RG_EXE = str(SCRIPT_DIR / "rg.exe") if (SCRIPT_DIR / "rg.exe").exists() else "rg"

# ================= 0. 全局预加载 =================
def build_file_map():
    """
    构建 {filename: full_path} 映射表。
    只收录原始资料文件，不收录 .index 下生成的 json。
    """
    file_map = {}
    if TARGET_FOLDER.exists():
        for root, dirs, filenames in os.walk(TARGET_FOLDER):
            root_path = Path(root)
            dirs[:] = [d for d in dirs if d != ".index"]

            for f in filenames:
                if f.endswith(('.txt', '.md')):
                    full_path = root_path / f
                    file_map[f] = str(full_path)
    return file_map

FILE_MAP = build_file_map()

def get_client():
    """懒加载 OpenAI Client，便于本地工具函数在无 API Key 时也能运行"""
    global CLIENT
    if CLIENT is None:
        api_key = os.getenv(model_dict[index]['api_key'])
        if not api_key:
            raise RuntimeError(
                f"缺少 API Key 环境变量: {model_dict[index]['api_key']}"
            )
        CLIENT = OpenAI(
            base_url=model_dict[index]['base_url'],
            api_key=api_key,
        )
    return CLIENT

def ensure_index_exists():
    """确保全局目录索引已生成"""
    if not MAIN_INDEX_PATH.exists():
        print("🛠️ [System] 未检测到目录缓存，正在初始化全局总览目录及详细索引 ...")
        scan_folder(str(TARGET_FOLDER), recursive=True, output_dir=str(INDEX_DIR))
    else:
        print('🛠️ [System] 目录已存在')

def get_global_toc_summary() -> str:
    """读取全局索引原始 JSON，供 Prompt 预加载使用"""
    ensure_index_exists()
    with open(MAIN_INDEX_PATH, 'r', encoding='utf-8') as f:
        index_data = json.load(f)
    return json.dumps(index_data, ensure_ascii=False, indent=2)

# ================= 1. 增强版工具 =================

def get_document_toc(filename: str) -> str:
    """
    获取指定文档的详细目录结构。
    总览目录已预加载到上下文，此工具只返回某个文件的详细目录 JSON。
    """
    print(f"\n📑 [Tool: TOC] 获取详细目录: {filename}")
    ensure_index_exists()

    real_path = None
    for key_name, full_path in FILE_MAP.items():
        if filename in key_name:
            real_path = Path(full_path)
            break

    if not real_path:
        return json.dumps({"error": f"未找到名为 '{filename}' 的文件，请检查文件名。"}, ensure_ascii=False, indent=2)

    detail_index_path = INDEX_DIR / f"{real_path.stem}.index.json"

    if not detail_index_path.exists():
        return json.dumps({"error": f"无法获取 '{filename}' 的详细目录，可能是文件格式不支持提取目录或未生成。"}, ensure_ascii=False, indent=2)

    with open(detail_index_path, 'r', encoding='utf-8') as f:
        toc_data = json.load(f)

    return json.dumps(toc_data, ensure_ascii=False, indent=2)

def execute_grep(pattern: str, include_files: str = None) -> str:
    """
    搜索关键词。直接传递文件路径，解决文件名含 [] 等特殊字符导致过滤失效的问题。
    """
    cmd = [RG_EXE, "-n", "-i", "-C", "2", "-e", pattern]

    scope_desc = "全库"

    if include_files:
        req_list = include_files.split(',')
        target_paths = []
        matched_names = []

        for req in req_list:
            req = req.strip()
            if not req: continue

            for filename, full_path in FILE_MAP.items():
                if req in filename:
                    target_paths.append(full_path)
                    matched_names.append(filename)

        target_paths = list(set(target_paths))
        [print(f'\n🛠️ [Tool: Grep] 调试： 文件过滤--- {file}') for file in target_paths]
        if not target_paths:
            return f"系统反馈：文件过滤失败。指定的 '{include_files}' 未匹配到任何文件，请检查文件名。"

        cmd.extend(target_paths)
        scope_desc = f"限定于 {len(target_paths)} 个文件: {str(matched_names)[:100]}..."
    else:
        cmd.append(str(TARGET_FOLDER))

    print(f"\n🛠️ [Tool: Grep] 搜索: '{pattern}' (范围: {scope_desc})")

    cmd.extend(["-m", "50"])

    try:
        res = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')

        raw_hits = 0
        if res.stdout:
            raw_hits = len([line for line in res.stdout.strip().split('\n') if line.strip()])
        print(f"      📊 [Grep 统计] 命中行数: {raw_hits}")

        if not res.stdout:
            return "系统反馈：未找到任何匹配项。请尝试更换同义词。"

        lines = res.stdout.strip().split('\n')
        if len(lines) > 100:
            preview = "\n".join(lines[:60])
            return f"系统反馈：找到大量匹配（超过100行）。建议增加关键词精度。\n前 60 行预览：\n{preview}"

        return f"系统反馈：搜索结果如下 (包含上下文)：\n{res.stdout}"
    except Exception as e:
        return f"系统反馈：搜索出错 {str(e)}"

def read_file_range(filepath: str, start_line: int, end_line: int) -> str:
    print(f"\n📖 [Tool: Read] 阅读: {os.path.basename(filepath)} (行 {start_line}-{end_line})")
    try:
        real_path = filepath
        if not os.path.exists(real_path):
             base = os.path.basename(filepath)
             if base in FILE_MAP:
                 real_path = FILE_MAP[base]

        with open(real_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()

        total_lines = len(lines)
        start = max(0, start_line - 1)
        end = min(total_lines, end_line)

        content = "".join(lines[start:end])
        return f"--- 文件片段: {os.path.basename(real_path)} ---\n{content}\n--- 片段结束 ---"
    except Exception as e:
        return f"读取失败: {str(e)}"

# ================= 3. 工具模式定义 (V6 新版 schema) =================

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "get_document_toc",
            "description": "获取指定文档的详细章节目录和精确的位置，一般读取它就能一步获取答案的精确位置。当你已经知道问题在那本书，可以调用它来获取具体章节，返回起始行号用于读取精准的内容。",
            "parameters": {
                "type": "object",
                "properties": {
                    "filename": {
                        "type": "string",
                        "description": "目标文件名，例如 'GB50017钢结构设计标准'"
                    }
                },
                "required": ["filename"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "execute_grep",
            "description": "在文件中搜索特定关键词，返回匹配行及上下文。",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "要搜索的关键词"},
                    "include_files": {"type": "string", "description": "指定要在哪些文件中搜索，填入文件名或片段。为空则全库搜索。"}
                },
                "required": ["pattern"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_file_range",
            "description": "读取指定文件的特定行数范围的完整内容。",
            "parameters": {
                "type": "object",
                "properties": {
                    "filepath": {"type": "string", "description": "文件路径或文件名"},
                    "start_line": {"type": "integer", "description": "起始行号"},
                    "end_line": {"type": "integer", "description": "结束行号"}
                },
                "required": ["filepath", "start_line", "end_line"]
            }
        }
    }
]

# ================= 4. Agent 主循环 (V6 新版预加载 Prompt) =================

def run_agent(user_question: str):
    global_toc_summary = get_global_toc_summary()

    print(f"🚀 启动 V6 Agent (已加载 {len(FILE_MAP)} 个文件) | 问题: {user_question}")
    print("=" * 60)

    system_prompt = f"""你是一个专业的工程规范检索与解读专家。请你根据指定资料库内容回答问题，资料库中没提到到不要回答。

【资料库全局总概况目录 index.json】
{global_toc_summary}

【你的任务】：
解答用户的工程问题。你拥有获取指定文档详细目录、全文关键词搜索、按行号读取原文三个工具。请自主组合使用这些工具收集信息。

【纪律要求】：
1. 严禁捏造数据，必须调用工具查阅资料后才能回答。
2. 回答时，必须明确引用依据（如：“根据《XXX规范》第X.X条”）。
3. 如果单次读取或搜索的信息不足以得出结论，请继续调用工具深挖，直到获得确凿证据。
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_question}
    ]

    MAX_TURNS = 15
    for turn in range(MAX_TURNS):
        print(f"\n[第 {turn+1} 轮]")
        try:
            response = get_client().chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                tools=TOOLS_SCHEMA,
                tool_choice="auto",
                temperature=0.1
            )
        except Exception as e:
            print(f"API Error: {e}")
            break

        msg = response.choices[0].message
        messages.append(msg)

        if msg.tool_calls:
            for tool_call in msg.tool_calls:
                func_name = tool_call.function.name
                args = json.loads(tool_call.function.arguments)

                res = ""
                if func_name == "execute_grep":
                    res = execute_grep(args.get("pattern"), args.get("include_files"))
                elif func_name == "read_file_range":
                    res = read_file_range(args.get("filepath"), args.get("start_line"), args.get("end_line"))
                elif func_name == "get_document_toc":
                    res = get_document_toc(args.get("filename"))

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": res
                })
        else:
            print(f"\n✅ [最终回答]:\n{msg.content}")
            return

    # ================= 超时降级处理 =================
    print("\n⚠️ 超过最大轮数，停止工具调用，强制生成回答...")

    messages.append({
        "role": "user",
        "content": "系统指令：已达到最大搜索尝试次数。请立即停止搜索，根据以上历史信息，对我的问题进行总结回答。如果信息不完整，请基于现有线索进行推断并说明。"
    })

    try:
        final_response = get_client().chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            # tools=TOOLS_SCHEMA, # 不传工具参数，阻止继续调用
            temperature=0.3
        )
        print(f"\n✅ [最终回答 (强制输出)]:\n{final_response.choices[0].message.content}")
    except Exception as e:
        print(f"强制回答生成失败: {e}")

if __name__ == "__main__":
    # run_agent("何时需要设置拦风绳")
    # run_agent("门刚的伸缩缝距离")
    # run_agent("筏板的最小厚度")
    # run_agent("基础的宽高比")
    # run_agent("各种结构何时不需要计算温度工况")
    # run_agent("钢柱的长细比要求")
    # run_agent("高层框架结构的一般要求中抗震缝的相关要求")
    run_agent("基础的宽高比")
