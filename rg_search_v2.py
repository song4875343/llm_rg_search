# --- START OF FILE rg_search.py ---

import subprocess
import json
import os
import re
from openai import OpenAI
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 1. 初始化 ModelScope 接口 (或你所使用的其他兼容 OpenAI 格式的接口)
client = OpenAI(
    base_url='https://api-inference.modelscope.cn/v1',
    api_key=os.getenv('MODELSCOPE_API_KEY'),
)

# ==========================================
# 核心优化：快慢模型分离设置
# ==========================================
# 用于基础任务（生成正则）：速度极快、成本极低
FAST_MODEL = 'Qwen/Qwen3-30B-A3B-Instruct-2507' 
# 用于复杂推理任务（智能初筛、阅读长文本并总结）：能力最强
# REASONING_MODEL = 'moonshotai/Kimi-K2.5'
REASONING_MODEL = 'Qwen/Qwen3-235B-A22B-Instruct-2507'
def extract_json_from_text(text: str) -> dict:
    """工具函数：防止 Qwen/Kimi 输出带有 ```json 的 Markdown 格式导致解析失败"""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
        if match:
            return json.loads(match.group(1))
        start = text.find('{')
        end = text.rfind('}') + 1
        if start != -1 and end != 0:
            return json.loads(text[start:end])
        return {}

# ==========================================
# 核心优化：正则生成与自我修复重试机制
# ==========================================
def generate_regex_with_retry(user_question: str, max_retries: int = 3) -> str:
    sys_prompt = f"""你是一个精通中国建筑/工程规范的顶级检索专家。
    请分析用户的提问，提取核心实体和属性，生成【一个】极高覆盖率的 ripgrep 组合正则表达式。

    【核心检索铁律】（必须遵守）：
    1. 规范黑话转换：用户常问“最小/最大/多少”，但规范原文通常写为“不应小于/不宜小于/不得大于/不宜超过/≥/mm/m”。你的正则必须包含这些规范术语。
    2. 舍弃废话：绝对不要在正则中包含“要求、规定、是多少、最小、最大”这些在原文中极少精确出现的形容词或疑问词。
    3. 同义词扩充：主体名词必须扩充，例如用户问“筏板”，你必须扩充为 `(筏板|筏基|筏形基础|底板|筏)`；用户问“厚度”，扩充为 `(厚度|板厚|厚)`。
    4. 结构：使用 `.*` 连接主体和属性，并用 `|` 并列，形成多种可能性。
    5. 语法检查：确保所有的圆括号 `()` 必须成对闭合！千万不要漏掉闭合括号导致语法错误。

    【反面教材（错误）】：
    用户问：筏板的最小厚度是多少？
    错误正则：筏板.*最小.*厚度 （太死板，完全搜不到“不应小于”的条文）

    【正面教材（正确）】：
    用户问：筏板的最小厚度是多少？
    正确正则：(筏板|筏基|筏形基础|底板).*(厚度|板厚).*(不应|不宜|不得|不小于|mm)|(平板式|梁板式)筏基.*(厚度|板厚)|筏.*厚

    用户问：门刚的温度区段
    正确正则：(门式刚架|刚架结构|轻型钢架).*(温度区段|伸缩缝|温度缝)

    务必只返回 JSON 格式，包含一个名为 "regex" 的字符串。
    """
    
    error_feedback = ""
    
    for attempt in range(1, max_retries + 1):
        if error_feedback:
            prompt_content = f"用户问题：{user_question}\n\n⚠️注意！你上一次生成的正则存在语法错误：\n{error_feedback}\n请仔细检查括号()是否成对匹配，修正语法问题，重新输出 JSON。"
            print(f"🔄 [正则重试 {attempt}/{max_retries}] 发现正则语法错误，正在让 Ai 修复...")
        else:
            prompt_content = f"用户问题：{user_question}"
            print(f"🧠 [阶段1] 正在用快速模型 {FAST_MODEL} 生成组合正则...")

        response = client.chat.completions.create(
            model=FAST_MODEL, # 使用快速便宜的模型写正则
            messages=[
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": prompt_content}
            ],
            temperature=0.3
        )
        
        combined_regex = extract_json_from_text(response.choices[0].message.content).get("regex", "")
        
        if not combined_regex:
            error_feedback = "未能成功解析出 JSON 中的 regex 字段，请确保仅返回纯 JSON。"
            continue
            
        # 语法校验：测试正则合法性
        try:
            re.compile(combined_regex)
            return combined_regex # 校验通过
        except re.error as e:
            error_feedback = f"正则表达式编译失败，错误详情：{e.msg}。这通常是因为括号没有成对闭合。"
            
    # 3次都失败
    return ""

def run_spec_query(
    user_question: str, 
    reg_num=3,    
    context_lines: int = 15,     
    max_matches_per_file: int = 20 
    ):
    
    # 确保当前目录下有 rg.exe 和 specs 文件夹
    if not os.path.exists("./rg.exe"):
        print("警告：未在当前目录找到 rg.exe，请先下载并放置在此处！")
    
    if not os.path.exists("./specs/"):
        os.makedirs("./specs/")
        print("提示：已自动创建 ./specs/ 文件夹，请把 Markdown 规范放入其中。")

    print(f"\n用户的问题：{user_question}")
    print("-" * 50)
    
    # ==========================================
    # 第一步：生成正则（带自动修复校验）
    # ==========================================
    combined_regex = generate_regex_with_retry(user_question)
    
    if not combined_regex:
        print("❌ [严重错误] 连续 3 次未能生成合法的正则表达式，检索程序退出。请尝试换个问法。")
        return
        
    print(f"⚡ 执行极速正则: {combined_regex}")

    # ==========================================
    # 第二步：执行搜索，抓取“带有命中行文本”的数据
    # ==========================================
    raw_hits = {} # 使用字典去重，key="filepath|line_num", value="该行文本"
    rg_executable = os.path.join(os.path.dirname(__file__), "rg.exe")
    spec_dir = "./specs/" 

    cmd = [rg_executable, "-n", "-i", "-m", str(max_matches_per_file), "-e", combined_regex, spec_dir]
    
    rg_result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
    
    if rg_result.stdout:
        for line in rg_result.stdout.strip().split('\n'):
            parts = line.split(':', 2) # 分割出文件、行号、文本
            if len(parts) >= 3:
                filepath = parts[0]
                try:
                    line_num = int(parts[1])
                    text_content = parts[2].strip()
                    hit_key = f"{filepath}|{line_num}"
                    if hit_key not in raw_hits:
                        raw_hits[hit_key] = text_content
                except ValueError:
                    continue

    if not raw_hits:
        print("\n❌ 未能从规范库中检索到相关条款。可能是没匹配上，可以尝试调整提问。")
        return

    hits_list = list(raw_hits.items())

    # ==========================================
    # 第三步：Agentic 初筛 —— 推理模型做智能裁判
    # ==========================================
    print(f"\n🎯 [阶段2] 原始搜到 {len(hits_list)} 处锚点，正在让推理模型 {REASONING_MODEL} 进行智能初筛...")
    
    # 构建给大模型看的浓缩摘要列表 (最多传50个，防爆token)
    hits_summary = []
    for i, (hit_key, text) in enumerate(hits_list):
        filepath, line_num = hit_key.split('|')
        filename = os.path.basename(filepath)
        # 截断过长的文本
        short_text = text[:80] + "..." if len(text) > 80 else text
        hits_summary.append(f"ID:{i} | 文件:{filename} | 行号:{line_num} | 文本:{short_text}")
        
    hits_summary_text = "\n".join(hits_summary[:50])
    
    # 挑选最相关的 5 到 8 个
    filter_prompt = f"""你是一个智能检索裁判。用户的问题是："{user_question}"
    以下是 ripgrep 搜索规范文件得到的初步命中行（仅展示当前单行）。请你判断哪些行所在的位置最有可能包含用户想找的【完整答案】。
    请挑选出最相关的 5 到 8 个条目的 ID。

    {hits_summary_text}

    请仅返回 JSON 格式，例如：{{"selected_ids": [0, 3, 5]}}"""

    # 使用推理模型做判断（初筛对准确度影响很大，必须用强模型）
    filter_response = client.chat.completions.create(
        model=REASONING_MODEL,
        messages=[{"role": "user", "content": filter_prompt}],
        temperature=0.1 # 温度调低，保证理性判断
    )
    
    selected_ids = extract_json_from_text(filter_response.choices[0].message.content).get("selected_ids", [])
    
    # 校验返回值，若大模型回答异常或未选中任何结果，走兜底策略（默认前5个）
    valid_ids = [i for i in selected_ids if isinstance(i, int) and 0 <= i < len(hits_list)]
    if not valid_ids:
        print("⚠️ 大模型未返回有效 ID，将默认选用前 5 条结果进行扩展。")
        valid_ids = list(range(min(5, len(hits_list))))

    # ==========================================
    # 第四步：根据精准筛选后的 ID 进行展卷
    # ==========================================
    context_snippets = []
    print(f"✨ [阶段3] 筛选完毕，最终精选出 {len(valid_ids)} 处核心锚点，正在精准向上下扩展 {context_lines} 行...")
    
    for idx in valid_ids:
        hit_key, _ = hits_list[idx]
        filepath, line_str = hit_key.split('|')
        line_num = int(line_str)
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                all_lines = f.readlines()
                start = max(0, line_num - 1 - context_lines) 
                end = min(len(all_lines), line_num + context_lines)
                snippet = "".join(all_lines[start:end])
                
                doc_name = os.path.basename(filepath)
                context_snippets.append(f"【来源: {doc_name}，第 {start+1} 至 {end} 行】\n{snippet}")
                print(f" -> 📄 展卷加载: {doc_name} (行 {start+1}~{end})")
        except Exception as e:
            continue

    final_context = "\n\n================\n\n".join(context_snippets)

    # ==========================================
    # 第五步：推理模型流式阅读并综合回答
    # ==========================================
    print(f"\n📖 [阶段4] 正在由推理模型 {REASONING_MODEL} 基于提纯后的上下文生成最终结论...\n")
    print("🤖 Ai 回答: ", end="")
    
    final_prompt = f"""你是一个严谨的工程审查专家。请基于以下我为你精准检索到的规范原文片段，回答用户问题。
    【用户问题】：{user_question}

    【检索到的规范原文】：
    {final_context}

    回答要求：
    1. 完全基于提供的原文，严禁瞎编。若原文无法回答，请诚实说明。
    2. 先引出高相关原文，再进行总结回答，总结回答要明确。
    3. 必须在回答中明确引用原文的规范名称及条文号及行号（如：根据《GB50016》第几条第XX行）。
    """
    
    # 使用推理模型做最后总结
    stream_response = client.chat.completions.create(
        model=REASONING_MODEL,
        messages=[{"role": "user", "content": final_prompt}],
        temperature=0.3, 
        stream=True
    )
    
    for chunk in stream_response:
        if chunk.choices and chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end='', flush=True)
    print("\n\n" + "-" * 50)

# ==========================================
# 调试与运行入口
# ==========================================
if __name__ == "__main__":
    run_spec_query(
        user_question="基础的宽高比", 
        reg_num=3,    
        context_lines=1,          # 保留你的修改
        max_matches_per_file=20   # 保留你的修改
    )