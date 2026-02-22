import subprocess
import json
import os
import re
from openai import OpenAI
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 1. 初始化 ModelScope 接口
client = OpenAI(
    base_url='https://api-inference.modelscope.cn/v1',
    api_key=os.getenv('MODELSCOPE_API_KEY'),
)

# 默认使用你指定的 Qwen 千亿模型
MODEL_NAME = os.getenv('MODEL_NAME', 'moonshotai/Kimi-K2.5')
# 可选模型: Qwen/Qwen3-235B-A22B-Instruct-2507
def extract_json_from_text(text: str) -> dict:
    """工具函数：防止 Qwen 输出带有 ```json 的 Markdown 格式导致解析失败"""
    try:
        # 如果是纯 json 字符串，直接解析
        return json.loads(text)
    except json.JSONDecodeError:
        # 如果带有 markdown 标记，用正则提取大括号里的内容
        match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
        if match:
            return json.loads(match.group(1))
        # 终极兜底方案
        start = text.find('{')
        end = text.rfind('}') + 1
        if start != -1 and end != 0:
            return json.loads(text[start:end])
        return {"regex_list": []}

def run_spec_query(
    user_question: str, 
    context_lines: int = 15,     # 参数2：命中行上下扩展的行数
    max_matches_per_file: int = 10 # 参数3：每个规范文件最多抓取多少处（解决内容不全）
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
    # 第一步：让 Qwen 生成一组搜索正则
    # ==========================================
    sys_prompt = """你是一个精通中国建筑/工程规范的顶级检索专家。
    请分析用户的提问，提取核心实体和属性，生成【一个】极高覆盖率的 ripgrep 组合正则表达式。

    【核心检索铁律】（必须遵守）：
    1. 规范黑话转换：用户常问“最小/最大/多少”，但规范原文通常写为“不应小于/不宜小于/不得大于/不宜超过/≥/mm/m”。你的正则必须包含这些规范术语，而不是死板地匹配“最小”二字。
    2. 舍弃废话：绝对不要在正则中包含“要求、规定、是多少、最小、最大”这些在原文中极少精确出现的形容词或疑问词。
    3. 同义词扩充：主体名词必须扩充，例如用户问“筏板”，你必须扩充为 `(筏板|筏基|筏形基础|底板)`；用户问“厚度”，扩充为 `(厚度|板厚)`。
    4. 结构：使用 `.*` 连接主体和属性，并用 `|` 并列多种可能性。

    【反面教材（错误）】：
    用户问：筏板的最小厚度是多少？
    错误正则：筏板.*最小.*厚度 （太死板，完全搜不到“不应小于”的条文）

    【正面教材（正确）】：
    用户问：筏板的最小厚度是多少？
    正确正则：(筏板|筏基|筏形基础|底板).*(厚度|板厚).*(不应|不宜|不得|不小于|mm)|(平板式|梁板式)筏基.*(厚度|板厚)

    用户问：门刚的温度区段
    正确正则：(门式刚架|刚架结构|轻型钢架).*(温度区段|伸缩缝|温度缝)

    务必只返回 JSON 格式，包含一个名为 "regex" 的字符串。
    """
    
    print("🧠 正在生成组合正则...")
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_question}
        ],
        temperature=0.3
    )
    
    combined_regex = extract_json_from_text(response.choices[0].message.content).get("regex", "")
    if not combined_regex:
        print("❌ 未能生成有效的正则表达式。")
        return
        
    print(f"⚡ 执行极速正则: {combined_regex}")

    # ==========================================
    # 优化2：只执行 1 次 rg.exe
    # ==========================================
    unique_hits = set()
    rg_executable = os.path.join(os.path.dirname(__file__), "rg.exe")
    spec_dir = "./specs/" 

    cmd = [rg_executable, "-n", "-i", "-m", str(max_matches_per_file), "-e", combined_regex, spec_dir]
    
    # 运行单次查询
    rg_result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
    
    if rg_result.stdout:
        for line in rg_result.stdout.strip().split('\n'):
            parts = line.split(':', 2)
            if len(parts) >= 2:
                filepath = parts[0]
                try:
                    line_num = int(parts[1])
                    unique_hits.add(f"{filepath}|{line_num}")
                except ValueError:
                    continue

    if not unique_hits:
        print("\n❌ 未能从规范库中检索到相关条款。")
        return

    # ==========================================
    # 第三步：自动展卷（上下文读取）
    # ==========================================
    context_snippets = []
    print(f"🎯 共命中 {len(unique_hits)} 个独立规范锚点，正在向上下各扩展 {context_lines} 行...")
    
    for hit in unique_hits:
        filepath, line_str = hit.split('|')
        line_num = int(line_str)
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                all_lines = f.readlines()
                # 使用你传入的 context_lines 参数
                start = max(0, line_num - 1 - context_lines) 
                end = min(len(all_lines), line_num + context_lines)
                snippet = "".join(all_lines[start:end])
                
                doc_name = os.path.basename(filepath)
                context_snippets.append(f"【来源: {doc_name}，第 {start+1} 至 {end} 行】\n{snippet}")
                # print(f"【来源: {doc_name}，第 {start+1} 至 {end} 行】\n{snippet}")
        except Exception as e:
            continue

    final_context = "\n\n================\n\n".join(context_snippets)[:15000] # Qwen 397B 上下文很长，可稍微放宽限制

    # ==========================================
    # 第四步：Qwen 流式阅读并综合回答
    # ==========================================
    print("📖 正在由 Qwen 综合规范原文生成最终结论...\n")
    print("🤖 Qwen 回答: ", end="")
    
    final_prompt = f"""你是一个严谨的工程审查专家。请基于以下我为你检索到的规范原文片段，回答用户问题。
    【用户问题】：{user_question}

    【检索到的规范原文】：
    {final_context}

    回答要求：
    1. 完全基于提供的原文，严禁瞎编。
    2. 先引出高相关原文，再进行总结回答，总结回答要明确
    3. 必须在回答中明确引用原文的规范名称及条文号及行号（如：根据《GB50016》第几条第XX行）。
    """
    
    # 使用 stream=True 实现打字机效果
    stream_response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": final_prompt}],
        temperature=0.3, 
        stream=True
    )
    
    for chunk in stream_response:
        if chunk.choices and chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end='', flush=True)
    print("\n" + "-" * 50)

# ==========================================
# 调试与运行入口
# ==========================================
if __name__ == "__main__":
    # 你可以通过修改后面的参数来测试不同设定的效果
    run_spec_query(
        user_question="什么时候需要验算竖向地震", 
        context_lines=1,    # 设为 5 废话少，设为 30 能看到更全的上下文，如果规范是清洗好的一行一条这里可以用0或1
        max_matches_per_file=20   # 允许每个文件最多抓取20~30处，保证信息全面
    )