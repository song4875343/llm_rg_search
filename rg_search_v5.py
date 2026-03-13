import subprocess
import json
import os
from openai import OpenAI
from dotenv import load_dotenv

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
index=2
# ================= 配置区 =================
model_dict={1:{'factory_name':'kimi','base_url':'https://api.moonshot.cn/v1','api_key':'kimi_key','model_name':'kimi-k2.5'},
            2:{'factory_name':'nvidia','base_url':'https://integrate.api.nvidia.com/v1','api_key':'nvidia_key','model_name':'minimaxai/minimax-m2.5'},
            3:{'factory_name':'modelscope','base_url':'https://api-inference.modelscope.cn/v1','api_key':'modelscope_key','model_name':'Qwen/Qwen3-235B-A22B-Instruct-2507'}
            }
CLIENT = OpenAI(
    base_url = model_dict[index]['base_url'],
    api_key=os.getenv(model_dict[index]['api_key']),
)

MODEL_NAME = model_dict[index]['model_name']

TARGET_FOLDER = './specs/'
RG_EXE = "rg" # Windows下改为 rg.exe 的绝对路径

# ================= 0. 全局预加载 (建立 文件名->路径 映射) =================
def build_file_map():
    """
    构建 {filename: full_path} 映射表。
    这样无论文件在根目录还是子目录，或者包含特殊字符，都能找到准确路径。
    """
    file_map = {}
    if os.path.exists(TARGET_FOLDER):
        for root, _, filenames in os.walk(TARGET_FOLDER):
            for f in filenames:
                if f.endswith(('.txt', '.md', '.json')):
                    # 存储绝对路径或相对路径，确保 rg 能找到
                    full_path = os.path.join(root, f)
                    file_map[f] = full_path
    return file_map

# 全局常量：文件名到路径的映射
FILE_MAP = build_file_map()
# 仅提取文件名列表供 Prompt 使用
ALL_FILES_LIST = list(FILE_MAP.keys())

# ================= 1. 增强版工具 (路径直传模式) =================

def execute_grep(pattern: str, include_files: str = None) -> str:
    """
    搜索关键词。
    ✨ 修复：不再使用 -g 参数，而是直接传递文件路径，解决文件名含 [] 等特殊字符导致过滤失效的问题。
    """
    cmd = [RG_EXE, "-n", "-i", "-C", "2", "-e", pattern]
    
    scope_desc = "全库"
    
    # === 关键修改逻辑 ===
    if include_files:
        # 1. 模糊匹配找出目标文件
        req_list = include_files.split(',')
        target_paths = []
        matched_names = []
        
        for req in req_list:
            req = req.strip()
            if not req: continue
            
            # 在 FILE_MAP 中查找包含关键词的文件
            for filename, full_path in FILE_MAP.items():
                if req in filename:
                    target_paths.append(full_path)
                    matched_names.append(filename)
        
        # 去重
        target_paths = list(set(target_paths))
        [print(f'\n🛠️ [Tool: Grep] 调试： 文件过滤--- {file}') for file in target_paths]
        if not target_paths:
            return f"系统反馈：文件过滤失败。指定的 '{include_files}' 未匹配到任何文件，请检查文件名。"
        
        # 2. 将具体路径作为参数加到命令末尾
        # 形式：rg -e pattern path/to/file1 path/to/file2
        cmd.extend(target_paths)
        
        scope_desc = f"限定于 {len(target_paths)} 个文件: {str(matched_names)[:100]}..."
    else:
        # 如果没有过滤，搜索整个目录
        cmd.append(TARGET_FOLDER)

    print(f"\n🛠️ [Tool: Grep] 搜索: '{pattern}' (范围: {scope_desc})")
    
    cmd.extend(["-m", "50"]) # 限制匹配次数
    
    try:
        # 注意：target_paths 可能会很多，Windows 命令行有长度限制
        # 如果文件极多(>50个)，建议分批或回退到全库搜索。这里假设过滤后不会太多。
        res = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
        
        # 统计行数
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
        # 这里的 filepath 应该是 grep 返回的准确路径，但为了保险，还是做个校验
        real_path = filepath
        if not os.path.exists(real_path):
             # 尝试从 Map 里找
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

# ================= 3. 主循环保持不变，只需更新 Prompt =================

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "execute_grep",
            "description": "搜索关键词。如需查特定规范，请在 include_files 填入文件名片段。",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "关键词"},
                    "include_files": {"type": "string", "description": "文件名过滤，例如 'GB50007'，程序会自动匹配完整路径。"}
                },
                "required": ["pattern"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_file_range",
            "description": "详细阅读文件内容。",
            "parameters": {
                "type": "object",
                "properties": {
                    "filepath": {"type": "string", "description": "文件路径（直接使用grep结果中的路径）"},
                    "start_line": {"type": "integer", "description": "起始行"},
                    "end_line": {"type": "integer", "description": "结束行"}
                },
                "required": ["filepath", "start_line", "end_line"]
            }
        }
    }
]

def run_agent(user_question: str):
    # 使用文件名列表生成提示
    files_str = "\n".join([f"- {f}" for f in ALL_FILES_LIST])
    
    print(f"🚀 启动 V5 Agent (已加载 {len(ALL_FILES_LIST)} 个文件) | 问题: {user_question}")
    print("=" * 60)

    system_prompt = f"""你是一个工程规范检索专家。
    
    【当前资料库包含以下文件】：
    {files_str}
    
    【检索策略与红线纪律（必须严格遵守）】：
    1. **必须先搜索后回答**：遇到用户的问题，你【必须】优先调用 `execute_grep` 工具进行搜索。绝对禁止未调用工具就凭记忆直接输出答案！
    2. **标准工具调用机制**：你必须通过底层的 Function Calling 协议调用工具！【绝对禁止】在文本输出中使用 Markdown 代码块（如 ```execute_grep ... ```）来伪造工具调用。
    3. **文件过滤**：若问题针对特定规范，必须在 `execute_grep` 的 `include_files` 中填入文件名片段（如 'GB50007' 或 '钢结构'）。
    4. **关键词技巧**：grep 会返回上下文，若已包含答案直接回答；若发现关键线索（如“见表5.1”、“按第3条执行”），必须继续调用 `read_file_range` 或再次 `execute_grep` 进行深挖。
    5. **最终回答**：当且仅当工具返回了充足的原文证据后，你才能进行最终总结，且必须引用规范全名和条文号。
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_question}
    ]

    MAX_TURNS = 15
    for turn in range(MAX_TURNS):
        print(f"\n[第 {turn+1} 轮]")
        try:
            response = CLIENT.chat.completions.create(
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
                
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": res
                })
        else:
            print(f"\n✅ [最终回答]:\n{msg.content}")
            return

    # ================= 修改开始 =================
    print("\n⚠️ 超过最大轮数，停止工具调用，强制生成回答...")
    
    # 追加一条系统指令，要求模型立刻总结
    messages.append({
        "role": "user", 
        "content": "系统指令：已达到最大搜索尝试次数。请立即停止搜索，根据以上历史信息，对我的问题进行总结回答。如果信息不完整，请基于现有线索进行推断并说明。"
    })

    try:
        # 这次调用不传递 tools 参数，强制模型输出纯文本
        final_response = CLIENT.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            # tools=TOOLS_SCHEMA, # 注释掉工具，防止模型继续调用
            temperature=0.3
        )
        print(f"\n✅ [最终回答 (强制输出)]:\n{final_response.choices[0].message.content}")
    except Exception as e:
        print(f"强制回答生成失败: {e}")
    # ================= 修改结束 =================

if __name__ == "__main__":
    # run_agent("何时需要设置拦风绳")
    # run_agent("门刚的伸缩缝距离")
    # run_agent("筏板的最小厚度")
    # run_agent("基础的宽高比")
    run_agent("各种结构何时不需要计算温度工况")
    # run_agent("钢柱的长细比要求")