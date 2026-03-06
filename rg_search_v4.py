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
# 核心组件 1：全局上下文记忆窗口（三池分流版 + 打分记录）
# ==========================================
class ContextWindow:
    """用于在多轮迭代中累积、去重和管理规范片段（三池分流策略 + 打分记录）"""
    def __init__(self):
        self.segments = {} # key: "filepath|start_line-end_line", value: text
        self.history_queries = set() # 记录搜过的正则，防死循环
        
        # 三个池子
        self.selected_pool = set()  # 已展卷的锚点 (filepath, line_num)
        self.rejected_pool = set()  # 不太相关的锚点 (filepath, line_num)
        self.candidate_pool = []    # 待处理的锚点 [(hit对象, score)]，包含分数信息
        
    def add(self, filepath: str, start_line: int, end_line: int, text: str):
        key = f"{filepath}|{start_line}-{end_line}"
        if key not in self.segments:
            self.segments[key] = f"【来源: {os.path.basename(filepath)}，第 {start_line} ~ {end_line} 行】\n{text}"
            
    def get_all_context(self) -> str:
        if not self.segments:
            return ""
        return "\n\n================================\n\n".join(self.segments.values())
    
    def has_high_score_pending(self, threshold: float = 5.0) -> bool:
        """检查候选池中是否有 ≥threshold 分的锚点"""
        return any(score >= threshold for _, score in self.candidate_pool)

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

    def _classify_scenario(self, missing_info: str, raw_hits_count: int, 
                          selected_pool_count: int, current_context: str) -> str:
        """分类当前应该使用哪种搜索策略
        
        Returns:
            "reference_jump": 跳转查表/查条文
            "broaden_search": 放宽搜索（锚点太少）
            "refine_search": 换词精搜（锚点足够但信息不完整）
        """
        # 规则 1：missing_info 中有明确的表格号或条文号 → 跳转查表
        if re.search(r'(表\s*\d+\.\d+\.\d+|第\s*\d+\.\d+\.\d+\s*条)', missing_info):
            return "reference_jump"
        
        # 规则 2：上下文中有引用语句 → 跳转查表
        if re.search(r'(详见表|按表|见表|符合.*?表|按.*?条|符合.*?条)', current_context):
            return "reference_jump"
        
        # 规则 3：锚点太少 → 放宽搜索
        if raw_hits_count < 5 or selected_pool_count < 3:
            return "broaden_search"
        
        # 规则 4：其他情况 → 换词精搜
        return "refine_search"
    
    def _generate_regex(self, user_question: str, missing_info: str = "", history_queries: set = None, 
                        failed_regexes: list = None, is_fallback: bool = False, 
                        raw_hits_count: int = 0, selected_pool_count: int = 0, 
                        current_context: str = "", max_retries: int = 3) -> str:
        """生成 ripgrep 正则。支持三种场景：跳转查表、放宽搜索、换词精搜

        Args:
            user_question: 用户原始问题
            missing_info: 缺失信息描述
            history_queries: 所有用过的正则（防完全重复）
            failed_regexes: 失败的正则列表（给 LLM 看）
            is_fallback: 失败回退标志
            raw_hits_count: 上一轮搜到的原始锚点数量
            selected_pool_count: 当前精选池的数量
            current_context: 当前已收集的上下文
            max_retries: 最大重试次数
        """

        # 构建失败历史文本
        failed_text = "、".join(failed_regexes) if failed_regexes else "无"


        # 【失败回退模式】上一轮没搜到结果或全部重复，触发极致放宽条件
        # =======================================================
        if is_fallback:
            sys_prompt = f"""你是一个规范检索专家。

            【用户问题】：{user_question}
            【失败反馈】：{missing_info}
            【已失败的正则（禁止再用相似词汇！）】：{failed_text}

            请生成一个宽泛的正则表达式：
            ⚠️ 核心规则【退回一段式】：
            1. 必须使用纯粹的【一段式】结构，即 `(词A|词B|词C)`。
            2. 严禁使用 `.*?` 组合多个括号！直接砍掉所有属性词和限定词（如"最小"、"最大"、"间距"、"厚度"）。
            3. 只保留核心主体词，并极力扩充该主体的同义词或上位词及简化，包括用户写的可能的错别字修正。
            
            例如：
            - 原问题："筏板最小厚度" → 放宽一段式正则：`(筏板|筏基|筏形|筏|底板|)`
            - 原问题："何时应设置揽风绳" → 放宽一段式正则：`(揽风绳|缆风绳|缆风|拉索|临时支撑)`
            - 原问题："伸缩缝间距" → 放宽一段式正则：`(伸缩缝|温度缝|变形缝|伸缩|温度|变形)`
            
            务必只返回 JSON：{{"regex": "正则表达式"}}"""

        # =======================================================
        # 【补充检索模式】有缺失信息，根据场景分类生成策略
        # =======================================================
        elif missing_info:
            # 场景分类
            scenario = self._classify_scenario(
                missing_info, 
                raw_hits_count, 
                selected_pool_count, 
                current_context
            )
            
            if scenario == "reference_jump":
                # 场景 1：跳转查表/查条文 (精确匹配)
                sys_prompt = f"""你是一个规范检索专家。当前需要【跳转查表或查条文】。
                
                【用户问题】：{user_question}
                【缺失信息】：{missing_info}
                
                请从缺失信息中提取表格号或条文号（如"表 5.2.1"、"第 6.1.2 条"），
                生成精准匹配的正则表达式。
                
                例如：
                - 缺失"需要查阅表 5.2.1" → 正则：`(表\\s*5\\.2\\.1|5\\.2\\.1)`
                - 缺失"需要查阅第 6.1.2 条" → 正则：`(第\\s*6\\.1\\.2\\s*条|6\\.1\\.2条)`
                
                务必只返回 JSON：{{"regex": "正则表达式"}}"""
                
            elif scenario == "broaden_search":
                # 场景 2：放宽搜索（锚点太少） -> 【退回一段式】
                sys_prompt = f"""你是一个规范检索专家。当前需要【放宽搜索条件】。
                
                【用户问题】：{user_question}
                【上一轮搜索结果】：只搜到 {raw_hits_count} 条锚点，太少了
                【缺失信息】：{missing_info}
                
                请生成一个宽泛的正则表达式：
                ⚠️ 核心规则【退回一段式】：
                1. 必须使用纯粹的【一段式】结构，即 `(词A|词B|词C)`。
                2. 严禁使用 `.*?` 组合多个括号！直接砍掉所有属性词和限定词（如"最小"、"最大"、"间距"、"厚度"）。
                3. 只保留核心主体词，并极力扩充该主体的同义词或上位词及简化，包括用户写的可能的错别字修正。
                
                例如：
                - 原问题："筏板最小厚度" → 放宽一段式正则：`(筏板|筏基|筏形|筏|底板|)`
                - 原问题："何时应设置揽风绳" → 放宽一段式正则：`(揽风绳|缆风绳|缆风|拉索|临时支撑)`
                - 原问题："伸缩缝间距" → 放宽一段式正则：`(伸缩缝|温度缝|变形缝|伸缩|温度|变形)`
                
                务必只返回 JSON：{{"regex": "正则表达式"}}"""
                
            else:  # refine_search
                # 场景 3：换词精搜（锚点足够但信息不完整） -> 【强制两段式】
                sys_prompt = f"""你是一个规范检索专家。当前需要【换词精搜】。
                
                【用户问题】：{user_question}
                【缺失信息】：{missing_info}
                【上一轮搜索结果】：搜到 {raw_hits_count} 条锚点，但信息不完整
                
                请从缺失信息中提取核心关键词，并进行泛化，生成精准的正则表达式。
                
                ⚠️ 核心检索策略【强制两段式宽泛匹配】：
                1. 语法结构：必须严格使用 `(主体词组).*?(属性及特征词组)` 的两段式结构！
                2. 严禁三段式：绝对禁止使用 `(A).*?(B).*?(C)`（三个及以上括号串联）！因为中文语序颠倒（如"最大间距"和"间距最大"）会导致三段式正则直接失效。
                3. 词汇大乱炖：将所有相关的属性词、动词、数值符号、量词全部扔进第二个括号里用 `|` 并列。
                
                例如：
                - 缺失"伸缩缝最大允许间距的具体数值"
                  → 提取关键词：伸缩缝、间距、最大、数值
                  → 优秀示范（两段式）：`(伸缩缝|温度缝|伸缩|温度).*?(间距|距离|长度|m)`
                  → 错误示范（三段式，严禁！）：`(伸缩缝|温度缝).*?(间距|距离).*?(最大|≤)`
                
                务必只返回 JSON：{{"regex": "正则表达式"}}"""

        # =======================================================
        # 【首次检索模式】默认策略，生成高覆盖率正则 -> 【强制两段式】
        # =======================================================
        else:
            sys_prompt = f"""你是一个精通中国建筑/工程/法律规范的顶级检索专家。
            请分析用户的提问，提取核心实体和属性，生成【一个】极高覆盖率的 ripgrep 组合正则表达式。
            
            ⚠️ 核心检索策略【两段式宽泛匹配】：
            1. 语法结构：必须严格采用 `(主体词组).*?(属性及特征词组)` 的两段式结构。
            2. 严禁三段式：绝对禁止使用 `(A).*?(B).*?(C)` 这种三个括号串联的形式！规范中文语序多变，串联过多会导致严重漏搜。
            3. 泛化与并列：
               - 主体词放第一个括号："筏板" → `(筏板|筏基|筏|底板)`
               - 属性词/符号全部放第二个括号："最小厚度" → `(厚度|板厚|厚)`
               - 组合结果：`(筏板|筏基|筏|底板).*?(厚度|板厚|厚)`
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
        """三池分流策略：打分制 + 动态截断 + 智能补充
        
        Returns:
            bool: True 表示有新内容加入上下文，False 表示无新内容
        """
        if not raw_hits:
            return False
        
        # 第一步：去重（排除精选池和垃圾池）
        new_hits = []
        skipped_count = 0
        for hit in raw_hits:
            anchor_key = (hit['filepath'], hit['line_num'])
            if anchor_key not in context.selected_pool and anchor_key not in context.rejected_pool:
                new_hits.append(hit)
            else:
                skipped_count += 1
        
        if skipped_count > 0:
            print(f"   -> ⏭️  过滤掉 {skipped_count} 个已处理锚点（精选池/垃圾池）")
        
        if not new_hits:
            print("   -> ⚠️  所有锚点均已在精选池或垃圾池中，跳过本轮")
            return False
        
        # 第二步：合并候选池（新锚点 + 上轮待定锚点）
        # 候选池中的元素是 (hit, score) 元组
        old_candidates_with_scores = context.candidate_pool
        old_candidates = [hit for hit, score in old_candidates_with_scores]
        
        all_candidates = old_candidates + new_hits
        print(f"   -> 🆕 本轮候选池: {len(old_candidates)} 条待定 + {len(new_hits)} 条新增 = {len(all_candidates)} 条")
            
        # 第三步：构造摘要给大模型（打分制）
        hits_summary = [
            f"ID:{i} | 文件:{os.path.basename(h['filepath'])} | 行号:{h['line_num']} | 文本:{h['text'][:500]}" 
            for i, h in enumerate(all_candidates[:50])
        ]
        summary_text = "\n".join(hits_summary)
        
        prompt = f"""用户问题："{user_question}"

        以下是搜索命中行。请为每个锚点打分（0-10 分），评分标准：

        【9-10 分】直接回答问题的核心条文
        - 明确包含问题的主体和属性
        - 提供具体的数值、表格、公式
        - 例如：问"筏板最小厚度"，条文写"筏板厚度不应小于 300mm"

        【7-8 分】相关且提供关键信息
        - 涉及问题的主体或属性，但不够直接
        - 提供相关的背景规定或引用其他条款
        - 例如：问"筏板最小厚度"，条文写"筏板厚度应符合表 5.2.1"

        【4-6 分】弱相关或背景信息
        - 提到问题的某个关键词，但主题不完全匹配
        - 提供间接的背景知识
        - 例如：问"筏板最小厚度"，条文写"筏板应进行承载力验算"

        【0-3 分】不相关或误匹配
        - 主题完全不搭边,
        - 范围明显有问题，问的是结构问题返回的是建筑的条目，问的是砌体问题返回的是高层的条目
        - 关键词误匹配（如问"最小厚度"，搜到"最小配筋率"）
        - 纯定义性条文，对回答问题无实质帮助

        {summary_text}

        请仅返回 JSON（必须为每个 ID 打分）：
        {{
          "scores": [
            {{"id": 0, "score": 9}},
            {{"id": 1, "score": 2}},
            {{"id": 2, "score": 7}}
          ]
        }}"""
        
        response = client.chat.completions.create(
            model=FAST_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1
        )
        result = extract_json_from_text(response.choices[0].message.content)
        scores_list = result.get("scores", [])
        
        # 第四步：分数分流（8-10分、7分、3-6分、<3分）
        score_8_10 = []   # 8-10 分
        score_7 = []      # 7 分
        score_3_6 = []    # 3-6 分
        rejected_count = 0
        
        for score_item in scores_list:
            if not isinstance(score_item, dict):
                continue
            anchor_id = score_item.get('id')
            score = score_item.get('score', 0)
            
            if not isinstance(anchor_id, int) or anchor_id < 0 or anchor_id >= len(all_candidates):
                continue
            
            hit = all_candidates[anchor_id]
            anchor_key = (hit['filepath'], hit['line_num'])
            
            if score >= 8:
                score_8_10.append((hit, score))
            elif score >= 7:
                score_7.append((hit, score))
            elif score >= 3:
                score_3_6.append((hit, score))
            else:
                context.rejected_pool.add(anchor_key)
                rejected_count += 1
        
        # 按分数降序排序
        score_8_10.sort(key=lambda x: x[1], reverse=True)
        score_7.sort(key=lambda x: x[1], reverse=True)
        score_3_6.sort(key=lambda x: x[1], reverse=True)
        
        A = len(score_8_10)
        B = len(score_7)
        C = len(score_3_6)
        
        # 显示分流结果
        print(f"   -> 📊 打分分流: {A} 条(8-10分), {B} 条(7分), {C} 条(3-6分), {rejected_count} 条(<3分)")
        
        if rejected_count > 0:
            print(f"   -> 🗑️  {rejected_count} 条锚点已加入垃圾池")
        
        # 第五步：动态截断（7-8分总数≤8时全展卷，>8时7分降级）
        to_expand = []
        pending_for_next_round = []
        
        if A + B <= 8:
            # 场景 A：7-8 分总数 ≤8，全部展卷
            to_expand = score_8_10 + score_7
            pending_for_next_round = score_3_6
            print(f"   -> ✅ 7-8分总数({A+B})≤8，全部纳入展卷候选")
        else:
            # 场景 B：7-8 分总数 >8，7 分降级
            to_expand = score_8_10
            pending_for_next_round = score_7 + score_3_6
            print(f"   -> ⚠️ 7-8分总数({A+B})>8，{B}条7分锚点降级到候选池")
        
        # 第六步：智能补充（不足 5 条时从待定池补充）
        if len(to_expand) < 5:
            need_more = 5 - len(to_expand)
            available = len(pending_for_next_round)
            
            补充数量 = min(need_more, available)
            if 补充数量 > 0:
                print(f"   -> 📌 展卷候选不足5条，从待定池补充{补充数量}条")
                to_expand += pending_for_next_round[:补充数量]
                pending_for_next_round = pending_for_next_round[补充数量:]
        
        if not to_expand:
            print("   -> ⚠️  没有可展卷的锚点，更新候选池")
            context.candidate_pool = pending_for_next_round
            return False
        
        print(f"   -> 🎯 最终展卷 {len(to_expand)} 条锚点")
        
        # 第七步：展卷并加入上下文
        for idx, (hit, score) in enumerate(to_expand):
            print(f"\n   🔧 展卷 #{idx+1} [分数:{score}] [{os.path.basename(hit['filepath'])}:{hit['line_num']}]")
            print(f"      文本: {hit['text'][:80]}...")
            
            try:
                # 调用智能展卷引擎
                expand_result = self.expander.expand(hit['filepath'], hit['line_num'])
                
                if expand_result.get('source') == 'error':
                    print(f"      ❌ 展卷失败: {expand_result.get('reason', '未知错误')}")
                    continue
                
                snippet = expand_result['text']
                
                # 显示展卷方式
                if expand_result['source'] == 'llm_extracted':
                    print(f"      ✨ 使用 LLM 语义提取")
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
                
                # 加入精选池
                anchor_key = (hit['filepath'], hit['line_num'])
                context.selected_pool.add(anchor_key)
                
            except Exception as e:
                print(f"      ⚠️ 展卷异常: {e}")
                continue
        
        # 第八步：更新候选池（保留待定锚点，包含分数信息）
        context.candidate_pool = pending_for_next_round
        
        if len(pending_for_next_round) > 0:
            high_score_count = sum(1 for _, score in pending_for_next_round if score >= 5)
            print(f"   -> 📦 候选池更新: {len(pending_for_next_round)} 条待定锚点（其中{high_score_count}条≥5分）")
        
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
    # 主工作流：Agentic Loop（打分制 + 三池分流 + 动态截断 + 二次循环）
    # ==========================================
    def run_query(self, user_question: str):
        print(f"\n🔍 启动 Agentic 文档检索引擎（打分制 + 动态截断 + 再次循环）| 目标任务：{user_question}")
        print("=" * 60)
        
        context_window = ContextWindow()
        missing_info = ""
        is_fallback = False  # 失败回退标志
        failed_regexes = []  # 记录失败的正则
        
        # 核心 Loop
        for iteration in range(1, self.max_iterations + 1):
            print(f"\n🔄[第 {iteration}/{self.max_iterations} 轮迭代] 思考节点启动...")
            print(f"   📊 当前状态: 精选池 {len(context_window.selected_pool)} 条 | 垃圾池 {len(context_window.rejected_pool)} 条 | 候选池 {len(context_window.candidate_pool)} 条")
            
            # 1. 动态生成/调整策略（传入更多上下文信息）
            current_context = context_window.get_all_context()
            regex = self._generate_regex(
                user_question, 
                missing_info, 
                context_window.history_queries,
                failed_regexes,
                is_fallback,
                raw_hits_count=getattr(self, '_last_raw_hits_count', 0),  # 上一轮的原始锚点数
                selected_pool_count=len(context_window.selected_pool),
                current_context=current_context
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
            
            # 2. 检索与展卷（打分制 + 动态截断 + 三池分流）
            raw_hits = self._execute_rg(regex)
            self._last_raw_hits_count = len(raw_hits)  # 记录本轮原始锚点数，供下一轮使用
            
            if not raw_hits:
                failed_regexes.append(regex)  # 记录失败的正则
                is_fallback = True  # 标记下一轮进入失败回退模式
                missing_info = "上一轮搜索正则太苛刻或词汇不存在，请彻底改变关键词，去掉过于绝对的词汇，换用同义词。"
                print(f"   -> 📭 未搜到匹配项，已记录反馈，准备下一轮尝试扩搜。")
                continue
                
            print(f"   -> 🎯 rg 搜索到 {len(raw_hits)} 处原始锚点，开始打分分流...")
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
                
                # 二次循环检查
                should_continue = False
                reasons = []
                
                # 条件 1：候选池中有 ≥5 分的待定锚点
                if context_window.has_high_score_pending(threshold=5.0):
                    should_continue = True
                    high_score_count = sum(1 for _, score in context_window.candidate_pool if score >= 5)
                    reasons.append(f"候选池中还有 {high_score_count} 条 ≥5分 的待定锚点")
                
                # 条件 2：精选池 < 2 条
                if len(context_window.selected_pool) < 2:
                    should_continue = True
                    reasons.append(f"精选池只有 {len(context_window.selected_pool)} 条，信息量不足")
                
                if should_continue and iteration < self.max_iterations:
                    print(f"⚠️ 虽然评估为'信息完整'，但触发二次循环条件：")
                    for reason in reasons:
                        print(f"   - {reason}")
                    print(f"   → 强制进入下一轮迭代")
                    # 不 break，继续循环
                else:
                    # 真正结束
                    break
            else:
                missing_info = assessment.get("missing_info", "未知缺失")
                print(f"⚠️ 评估结论：信息【碎片化或存在外部引用】。")
                print(f"   -> 缺口分析：{missing_info}")
                print(f"   -> 理由：{assessment.get('reason', '')}")
                
                if iteration == self.max_iterations:
                    print("🛑 已达到最大迭代次数，将基于现有碎片强行解答。")
                    if context_window.candidate_pool:
                        high_score_count = sum(1 for _, score in context_window.candidate_pool if score >= 5)
                        print(f"   ⚠️ 候选池还有 {len(context_window.candidate_pool)} 条待定锚点未处理（其中{high_score_count}条≥5分）")
        
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
    agent.run_query("基础的宽高比的要求")