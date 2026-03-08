import subprocess
import json
import os
import re
from typing import List, Dict, Any
from openai import OpenAI
from dotenv import load_dotenv
from smart_expander import SmartExpander

# 加载环境变量
load_dotenv()

# ==========================================
# 控制 HTTP 请求日志显示
# ==========================================
# 隐藏 HTTP 请求日志（推荐用于生产环境）
import logging
logging.getLogger("httpx").setLevel(logging.WARNING)

# 如果想看到详细的 HTTP 请求日志，改为：
# logging.getLogger("httpx").setLevel(logging.DEBUG)

# 初始化 ModelScope 或其他兼容 OpenAI 接口的客户端
client = OpenAI(
    # base_url='https://api-inference.modelscope.cn/v1',
    base_url='https://api.moonshot.cn/v1',
    api_key=os.getenv('MODELSCOPE_API_KEY'),
)

# ==========================================
# 核心配置：快慢模型分离
# ==========================================
# FAST_MODEL = 'Qwen/Qwen3-30B-A3B-Instruct-2507'     # 速度快，用于生成正则、智能初筛
# FAST_MODEL ='Qwen/Qwen3-235B-A22B-Instruct-2507'
# REASONING_MODEL = 'Qwen/Qwen3-235B-A22B-Instruct-2507' # 智商高，用于评估上下文、生成最终长答案
# FAST_MODEL ='moonshotai/Kimi-K2.5'
# REASONING_MODEL = 'moonshotai/Kimi-K2.5' # 若使用 Kimmoonshotai/Kimi-K2.5i，切换此行
FAST_MODEL = 'kimi-k2-turbo-preview'     # 速度快，用于生成正则、智能初筛
REASONING_MODEL = 'kimi-k2-turbo-preview' # 智商高，用于评估上下文、生成最终长答案

def extract_json_from_text(text: str) -> dict:
    """提取 JSON，兼容各种模型的乱七八糟输出格式"""
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
# 核心组件 1：全局上下文记忆窗口
# ==========================================
class ContextWindow:
    """用于在多轮迭代中累积、去重和管理规范片段"""
    def __init__(self):
        self.segments = {} # key: "filepath|start_line-end_line", value: text
        self.history_queries = set() # 记录搜过的正则，防死循环
        self.processed_anchors = set() # 记录已处理的锚点 (filepath, line_num)，防重复展卷
        
    def add(self, filepath: str, start_line: int, end_line: int, text: str):
        key = f"{filepath}|{start_line}-{end_line}"
        if key not in self.segments:
            self.segments[key] = f"【来源: {os.path.basename(filepath)}，第 {start_line} ~ {end_line} 行】\n{text}"
            
    def get_all_context(self) -> str:
        if not self.segments:
            return ""
        return "\n\n================================\n\n".join(self.segments.values())

# ==========================================
# 核心组件 2：Agentic Search 智能体
# ==========================================
class DocAgenticSearch:
    def __init__(self, target_folder='./specs/', max_iterations=3, context_lines=15, max_matches=20):
        self.target_folder = target_folder
        self.max_iterations = max_iterations
        self.context_lines = context_lines
        self.max_matches = max_matches
        self.rg_executable = os.path.join(os.path.dirname(__file__), "rg.exe")
        
        # 初始化智能展卷引擎，使用主模块的 client 和 FAST_MODEL
        self.expander = SmartExpander(
            llm_client=client,
            fast_model=FAST_MODEL,
            max_radius=context_lines
        )
        
        if not os.path.exists(self.target_folder):
            os.makedirs(self.target_folder)

    def _generate_regex(self, user_question: str, missing_info: str = "", history_queries: set = None, 
                        failed_regexes: list = None, is_fallback: bool = False, max_retries: int = 3) -> str:
        """生成 ripgrep 正则。支持基于 missing_info 或失败回退(is_fallback)进行策略调整

        Args:
            user_question: 用户原始问题
            missing_info: 缺失信息描述
            history_queries: 所有用过的正则（防完全重复）
            failed_regexes: 失败的正则列表（给 LLM 看）
            is_fallback: 失败回退标志
            max_retries: 最大重试次数
        """

        # 构建失败历史文本
        failed_text = "、".join(failed_regexes) if failed_regexes else "无"

        # 【失败回退模式】上一轮没搜到结果或全部重复，触发放宽条件
        if is_fallback:
            sys_prompt = f"""你是一个规范检索专家。

            【用户问题】：{user_question}
            【失败反馈】：{missing_info}
            【已失败的正则（禁止再用相似词汇！）】：{failed_text}

            策略调整要求：
            1. 彻底放宽条件，只保留1-2个核心名词
            2. 必须使用同义词或上位词（例如"揽风绳"搜不到，换成"缆风绳|拉索|临时支撑|稳定"）
            3. 去掉所有限定词（如"设置|要求|规定|数量|位置"等具体属性）
            4. 可以尝试搜索章节标题或通用术语（如"施工|安全|荷载"）

            生成【一个】极简、宽泛的 ripgrep 正则表达式。
            务必只返回 JSON：{{"regex": "正则表达式"}}"""

        # 【补充检索模式】有缺失信息，重点搜索缺失内容（查表/条款）
        elif missing_info:
            sys_prompt = f"""你是一个规范检索专家。当前正在进行【第N轮补充检索】。
            用户原始问题是：{user_question}
            当前我们需要补充查找的【缺失信息】是：{missing_info}
            请针对这部分【缺失信息】（例如特定的表格号、条文号或特定名词），生成【一个】精准的 ripgrep 组合正则表达式。
            注意：如果是查表或条款，直接匹配如 `(表\s*5\.2\.1|5\.2\.1条)` 等。
            务必只返回 JSON：{{"regex": "正则表达式"}}"""

        # 【首次检索模式】默认策略，生成高覆盖率正则
        else:
            sys_prompt = f"""你是一个精通中国建筑/工程/法律规范的顶级检索专家。
            请分析用户的提问，提取核心实体和属性，生成【一个】极高覆盖率的 ripgrep 组合正则表达式。
            1. 规范黑话转换：用户问"最小"，必须扩充为 `(不应小于|不宜小于|不得大于|≥|mm|m)`。
            2. 同义词扩充：主体必须扩充，如 `(筏板|筏基|底板)`。
            3. 结构：使用 `.*` 连接主体和属性，并用 `|` 并列。
            4. 语法检查：确保所有的圆括号 `()` 必须成对闭合！
            务必只返回 JSON：{{"regex": "正则表达式"}}"""

        error_feedback = ""
        for attempt in range(1, max_retries + 1):
            prompt_content = f"用户问题：{user_question}" if not error_feedback else f"语法错误反馈：\n{error_feedback}\n请修复括号等语法问题，重新输出。"

            response = client.chat.completions.create(
                model=FAST_MODEL, # 廉价快模型
                messages=[{"role": "system", "content": sys_prompt}, {"role": "user", "content": prompt_content}],
                temperature=0.3  # 提高温度给 LLM 更多发散空间找同义词
            )

            regex = extract_json_from_text(response.choices[0].message.content).get("regex", "")
            try:
                re.compile(regex) # 语法校验
                return regex
            except re.error as e:
                error_feedback = f"正则编译失败：{e.msg}。"
        return ""


    def _execute_rg(self, regex: str) -> List[Dict]:
        """执行 Ripgrep 物理检索"""
        cmd =[self.rg_executable, "-n", "-i", "-m", str(self.max_matches), "-e", regex, self.target_folder]
        rg_result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
        
        raw_hits =[]
        if rg_result.stdout:
            for line in rg_result.stdout.strip().split('\n'):
                parts = line.split(':', 2)
                if len(parts) >= 3:
                    try:
                        raw_hits.append({
                            "filepath": parts[0],
                            "line_num": int(parts[1]),
                            "text": parts[2].strip()
                        })
                    except ValueError:
                        pass
        return raw_hits

    def _filter_and_expand(self, user_question: str, raw_hits: List[Dict], context: ContextWindow) -> bool:
        """用快模型粗筛，并使用智能展卷录入全局记忆
        
        Returns:
            bool: True 表示有新内容加入上下文，False 表示无新内容
        """
        if not raw_hits:
            return False
        
        # 第一步：去重已处理过的锚点
        new_hits = []
        skipped_count = 0
        for hit in raw_hits:
            anchor_key = (hit['filepath'], hit['line_num'])
            if anchor_key not in context.processed_anchors:
                new_hits.append(hit)
                context.processed_anchors.add(anchor_key)  # 立即标记为已处理
            else:
                skipped_count += 1
        
        if skipped_count > 0:
            print(f"   -> ⏭️  过滤掉 {skipped_count} 个重复锚点")
        
        if not new_hits:
            print("   -> ⚠️  所有锚点均已在之前轮次处理过，跳过本轮")
            return False
        
        print(f"   -> 🆕 剩余 {len(new_hits)} 个新锚点待处理")
            
        # 第二步：构造摘要给大模型（使用过滤后的 new_hits）
        hits_summary =[f"ID:{i} | 文件:{os.path.basename(h['filepath'])} | 行号:{h['line_num']} | 文本:{h['text'][:200]}" for i, h in enumerate(new_hits[:50])]
        summary_text = "\n".join(hits_summary)
        
        prompt = f"""用户问题："{user_question}"\n以下是搜索命中行。请挑选出最相关的 3 到 5 个条目的 ID。\n{summary_text}\n请仅返回 JSON：{{"selected_ids": [0,2,4,20,35]}}"""
        
        response = client.chat.completions.create(
            model=FAST_MODEL, # 粗筛使用快模型即可
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )
        selected_ids = extract_json_from_text(response.choices[0].message.content).get("selected_ids", [])
        valid_ids =[i for i in selected_ids if isinstance(i, int) and 0 <= i < len(new_hits)]
        if not valid_ids: valid_ids = list(range(min(3, len(new_hits))))
        
        # 显示初筛结果
        print(f"   -> 📌 初筛选中 {len(valid_ids)} 条锚点：")
        for idx in valid_ids:
            hit = new_hits[idx]
            print(f"      ID:{idx} | {os.path.basename(hit['filepath'])}:{hit['line_num']} | {hit['text'][:80]}...")
        
        # 使用智能展卷并加入上下文
        for idx in valid_ids:
            hit = new_hits[idx]
            print(f"\n   🔧 展卷 ID:{idx} [{os.path.basename(hit['filepath'])}:{hit['line_num']}]")
            try:
                # 调用智能展卷引擎
                expand_result = self.expander.expand(hit['filepath'], hit['line_num'])
                
                if expand_result.get('source') == 'error':
                    print(f"      ❌ 展卷失败: {expand_result.get('reason', '未知错误')}")
                    continue
                
                snippet = expand_result['text']
                
                # 显示展卷方式和原因
                if expand_result['source'] == 'llm_extracted':
                    print(f"      ✨ 使用 LLM 语义提取")
                    # LLM 提取的结果没有精确行号，使用命中行作为标识
                    context.add(hit['filepath'], hit['line_num'], hit['line_num'], snippet)
                elif expand_result['source'] == 'rule_based_fallback':
                    print(f"      ⚡ 使用规则引擎（LLM 提取失败，已退回）")
                    context.add(
                        hit['filepath'], 
                        expand_result['start_line'], 
                        expand_result['end_line'], 
                        snippet
                    )
                else:
                    print(f"      ⚡ 使用规则引擎 (行 {expand_result['start_line']}-{expand_result['end_line']})")
                    context.add(
                        hit['filepath'], 
                        expand_result['start_line'], 
                        expand_result['end_line'], 
                        snippet
                    )
            except Exception as e:
                print(f"      ⚠️ 展卷异常: {e}")
                continue
        
        return True  # 有新内容加入上下文

    def _assess_sufficiency(self, user_question: str, context_text: str) -> dict:
        """核心大脑：判断当前收集的规范是否足够回答问题，是否陷入了‘见表格XX’的嵌套引用陷阱"""
        if not context_text:
            return {"is_sufficient": False, "missing_info": "没有任何有效文本，需要放宽搜索条件换词重搜。"}

        prompt = f"""你是一个严谨的工程审查总工。请评估以下提取的【规范上下文】是否足以完整回答【用户问题】。
        
        【用户问题】：{user_question}
        【现有规范上下文】：
        {context_text}
        
        请仔细检查：
        1. 现有的上下文是否【直接且完整】地回答了问题？
        2. 致命陷阱检查：上下文中是否出现了类似“按本规范表 5.2.1 采用”、“按第 6.1.2 条执行”、“符合某某规定的要求”等**指向其他条款或表格**的语句？如果是，并且那个引用的表格/条文不在现有上下文中，则信息【不足】！
        
        请以 JSON 格式返回判断结果：
        {{
            "is_sufficient": true 或 false,
            "missing_info": "如果为false，请简短说明缺失了什么（例如：需要查阅 表5.2.1，或者 需要知道筏板基础的具体规定）",
            "reason": "你的判断理由"
        }}
        """
        
        # 评估过程极其重要，必须使用最聪明的推理模型
        response = client.chat.completions.create(
            model=REASONING_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )
        return extract_json_from_text(response.choices[0].message.content)

    # ==========================================
    # 主工作流：Agentic Loop
    # ==========================================
    def run_query(self, user_question: str):
        print(f"\n🔍 启动 Agentic 文档检索引擎 | 目标任务：{user_question}")
        print("=" * 60)
        
        context_window = ContextWindow()
        missing_info = ""
        is_fallback = False  # 失败回退标志
        failed_regexes = []  # 记录失败的正则
        
        # 核心 Loop
        for iteration in range(1, self.max_iterations + 1):
            print(f"\n🔄[第 {iteration}/{self.max_iterations} 轮迭代] 思考节点启动...")
            
            # 1. 动态生成/调整策略（传入失败历史和回退标志）
            regex = self._generate_regex(
                user_question, 
                missing_info, 
                context_window.history_queries,
                failed_regexes,
                is_fallback
            )
            is_fallback = False  # 用完立刻重置标志位
            
            if not regex:
                print("❌ 无法生成有效的检索条件，中止流程。")
                break
            
            if regex in context_window.history_queries:
                print(f"⚠️ 生成了重复的检索条件：{regex}，跳出循环防止死胡同。")
                break
            context_window.history_queries.add(regex)
            print(f"⚡ 执行 Ripgrep: {regex}")
            
            # 2. 检索与展卷
            raw_hits = self._execute_rg(regex)
            if not raw_hits:
                failed_regexes.append(regex)  # 记录失败的正则
                is_fallback = True  # 标记下一轮进入失败回退模式
                missing_info = "上一轮搜索正则太苛刻或词汇不存在，请彻底改变关键词，去掉过于绝对的词汇，换用同义词。"
                print(f"   -> 📭 未搜到匹配项，已记录反馈，准备下一轮尝试扩搜。")
                continue
                
            print(f"   -> 🎯 初筛定位 {len(raw_hits)} 处锚点，展卷提取上下文...")
            has_new_content = self._filter_and_expand(user_question, raw_hits, context_window)
            
            # 如果本轮无新内容，跳过评估直接进入下一轮
            if not has_new_content:
                failed_regexes.append(regex)  # 全部重复也算失败
                is_fallback = True  # 标记下一轮进入失败回退模式
                missing_info = "上一轮搜索出的结果已被处理过，陷入死循环，请彻底使用不同的同义词或上位词。"
                print(f"   -> ⏭️  本轮无新内容，跳过评估，直接进入下一轮")
                continue
            
            # 3. 智能体反思 (Assessment)
            current_context = context_window.get_all_context()
            print(f"🧠 {REASONING_MODEL} 正在评估当前信息是否闭环...")
            assessment = self._assess_sufficiency(user_question, current_context)
            
            if assessment.get("is_sufficient", False):
                print(f"✅ 评估结论：信息已完整构建闭环！理由：{assessment.get('reason', '')}")
                break
            else:
                missing_info = assessment.get("missing_info", "未知缺失")
                print(f"⚠️ 评估结论：信息【碎片化或存在外部引用】。")
                print(f"   -> 缺口分析：{missing_info}")
                print(f"   -> 理由：{assessment.get('reason', '')}")
                
                if iteration == self.max_iterations:
                    print("🛑 已达到最大迭代次数，将基于现有碎片强行解答。")
        
        # ==========================================
        # 最终阶段：输出综合解答
        # ==========================================
        final_context = context_window.get_all_context()
        if not final_context:
            print("\n❌ 经过多次尝试，未能从文档库中检索到相关内容。")
            return
            
        print(f"\n📖 [最终阶段] 正在由 {REASONING_MODEL} 进行大统合生成...\n")
        final_prompt = f"""你是一个严谨的工程/法律审查专家。请基于以下我经过多轮检索为你提取的【文档片段集合】，回答用户问题。
        
        【用户问题】：{user_question}
        【检索到的文档片段集合】：
        {final_context}

        回答要求：
        1. 完全基于提供的原文，严禁利用模型自身知识瞎编。若原文无法回答，请诚实说明。
        2. 因为片段可能来源于多次迭代检索（包括了正文条文和被引用的表格），请先梳理逻辑，再给出最终结论。
        3. 必须在回答中明确引用来源及行号（如：根据《GB50016》第XX条，以及该条文引用的表5.2.1）。
        """
        
        print("🤖 专家解答: \n")
        stream_response = client.chat.completions.create(
            model=REASONING_MODEL,
            messages=[{"role": "user", "content": final_prompt}],
            temperature=0.3, 
            stream=True
        )
        
        for chunk in stream_response:
            if chunk.choices and chunk.choices[0].delta.content:
                print(chunk.choices[0].delta.content, end='', flush=True)
        print("\n\n" + "=" * 60)

# ==========================================
# 测试入口
# ==========================================
if __name__ == "__main__":
    # 模拟用户提问：假设用户问的问题，在其正文中只写了“见表5.2.1”，通过 V3 它可以自动去把表格搜出来
    agent = DocAgenticSearch(
        target_folder='./specs/', 
        max_iterations=3,   # 最多思考并重搜 3 次
        context_lines=30    # 智能展卷的最大搜索半径
    )
    agent.run_query("基础的宽高比如何要求")