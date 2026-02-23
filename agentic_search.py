# --- START OF FILE agentic_search.py ---

import json
import os
import re
import difflib  # 新增：用于模糊匹配文件名
from openai import OpenAI
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 初始化 API 客户端
client = OpenAI(
    base_url='https://api-inference.modelscope.cn/v1',
    api_key=os.getenv('MODELSCOPE_API_KEY'),
)

# 快慢模型分离
# FAST_MODEL = 'Qwen/Qwen3-30B-A3B-Instruct-2507'     # 用于极速查阅目录，做路由
FAST_MODEL ='Qwen/Qwen3-235B-A22B-Instruct-2507'
REASONING_MODEL = 'moonshotai/Kimi-K2.5'           # 用于最终阅读原文并综合推理
# REASONING_MODEL = 'Qwen/Qwen3-235B-A22B-Instruct-2507'
MAX_ROUTE_TARGETS = 8
MAX_EXPANDED_TARGETS = 10
MAX_SECTION_CHARS = 5000
MAX_TOTAL_CONTEXT_CHARS = 18000

def extract_json_from_text(text: str):
    """提取大模型返回的 JSON"""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', text, re.DOTALL)
        if match:
            code_block = match.group(1).strip()
            try:
                return json.loads(code_block)
            except json.JSONDecodeError:
                pass

        start_obj = text.find('{')
        end_obj = text.rfind('}')
        start_arr = text.find('[')
        end_arr = text.rfind(']')

        candidates = []
        if start_obj != -1 and end_obj != -1 and end_obj > start_obj:
            candidates.append(text[start_obj:end_obj + 1])
        if start_arr != -1 and end_arr != -1 and end_arr > start_arr:
            candidates.append(text[start_arr:end_arr + 1])

        for candidate in candidates:
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                continue
        return {}

DOT_VARIANTS = r"[\.．。·•]"
FULLWIDTH_DIGITS = str.maketrans("０１２３４５６７８９", "0123456789")

def normalize_for_match(text: str) -> str:
    """文本归一化：去除空格，全角数字转半角，将各类“点”转换为 '.'"""
    text = text.translate(FULLWIDTH_DIGITS)
    text = re.sub(r'\s+', '', text)
    text = re.sub(DOT_VARIANTS, '.', text)
    return text

def extract_section_number(title: str) -> str:
    """提取章节数字编号，如 '6.3' 或 '3'"""
    normalized = normalize_for_match(title)
    # 兼容前缀样式: (条文解释)6.3 ...
    normalized = re.sub(r'^[\(（]条文解释\d*[\)）]', '', normalized).strip()
    match = re.search(r'(\d+(?:\.\d+)*)', normalized)
    return match.group(1) if match else ""

def flatten_book_toc(book_toc: dict) -> list:
    """扁平化目录并去重，便于做章节纠偏"""
    flat_list = []
    for chapter, sections in book_toc.items():
        flat_list.append(chapter)
        flat_list.extend(sections)

    unique_list = []
    seen = set()
    for item in flat_list:
        if item not in seen:
            seen.add(item)
            unique_list.append(item)
    return unique_list

def find_best_match(target: str, candidates: list) -> str:
    """在候选列表里找最相近项，优先包含匹配，其次相似度匹配"""
    if not target or not candidates:
        return ""

    target_norm = normalize_for_match(target)
    if not target_norm:
        return ""

    best = ""
    best_ratio = 0.0

    for candidate in candidates:
        candidate_norm = normalize_for_match(candidate)
        if not candidate_norm:
            continue

        if target_norm in candidate_norm or candidate_norm in target_norm:
            return candidate

        ratio = difflib.SequenceMatcher(None, target_norm, candidate_norm).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best = candidate

    return best if best_ratio > 0.35 else ""

def normalize_route_targets(route_info, menu_toc: dict) -> list:
    """
    兼容多种路由输出格式，归一成:
    [{"book": "...", "section": "..."}, ...]
    """
    raw_targets = []

    if isinstance(route_info, dict):
        if isinstance(route_info.get("targets"), list):
            raw_targets = route_info["targets"]
        elif route_info.get("book") and route_info.get("section"):
            raw_targets = [route_info]
    elif isinstance(route_info, list):
        raw_targets = route_info

    if not raw_targets:
        return []

    all_books = list(menu_toc.keys())
    normalized_targets = []
    seen = set()

    for item in raw_targets[:MAX_ROUTE_TARGETS]:
        if not isinstance(item, dict):
            continue

        raw_book = str(item.get("book", "")).strip()
        raw_section = str(item.get("section", "")).strip()
        if not raw_book or not raw_section:
            continue

        resolved_book = raw_book if raw_book in menu_toc else find_best_match(raw_book, all_books)
        if not resolved_book:
            continue

        sections = flatten_book_toc(menu_toc.get(resolved_book, {}))
        resolved_section = raw_section if raw_section in sections else find_best_match(raw_section, sections)
        if not resolved_section:
            continue

        key = (resolved_book, resolved_section)
        if key in seen:
            continue
        seen.add(key)

        normalized_targets.append({
            "book": resolved_book,
            "section": resolved_section
        })

    return normalized_targets

def is_explanation_section(section: str) -> bool:
    """判断是否为条文解释章节（前缀样式）"""
    return bool(re.match(r'^\s*[\(（]条文解释\d*[\)）]', section.strip()))

def strip_explanation_prefix(section: str) -> str:
    """去除章节前缀的条文解释标记"""
    return re.sub(r'^\s*[\(（]条文解释\d*[\)）]\s*', '', section).strip()

def find_explanation_for_base(base_section: str, sections: list) -> str:
    """从同一本目录中找到 base_section 对应的条文解释章节"""
    exact_candidates = [f"(条文解释){base_section}"]
    for exact in exact_candidates:
        if exact in sections:
            return exact

    for section in sections:
        if not is_explanation_section(section):
            continue
        if strip_explanation_prefix(section) == base_section:
            return section

    return ""

def expand_targets_with_explanations(route_targets: list, menu_toc: dict) -> list:
    """
    为每个目标章节补齐配对：
    - 正文章节 -> 对应条文解释章节
    - 条文解释章节 -> 对应正文章节
    """
    expanded = []
    seen = set()

    def append_target(book: str, section: str):
        if not book or not section:
            return
        key = (book, section)
        if key in seen:
            return
        if len(expanded) >= MAX_EXPANDED_TARGETS:
            return
        seen.add(key)
        expanded.append({"book": book, "section": section})

    for target in route_targets:
        book = target.get("book", "")
        section = target.get("section", "")
        if not book or not section:
            continue

        append_target(book, section)
        sections = flatten_book_toc(menu_toc.get(book, {}))
        if not sections:
            continue

        if is_explanation_section(section):
            base = strip_explanation_prefix(section)
            if base in sections:
                append_target(book, base)
            else:
                resolved_base = find_best_match(base, sections)
                if resolved_base:
                    append_target(book, resolved_base)
        else:
            explanation = find_explanation_for_base(section, sections)
            if explanation:
                append_target(book, explanation)

    return expanded

def find_best_matching_file(target_book: str, directory: str = "./specs/") -> str:
    """
    黑科技：文件名模糊匹配
    在目录中寻找与 target_book 相似度最高的 txt 文件。
    即使 target_book 是 "建筑地基基础设计规范"，也能匹配到 "3建筑地基基础设计规范[附条文说明].txt"
    """
    if not os.path.exists(directory):
        return ""
        
    files = [f for f in os.listdir(directory) if f.endswith('.txt')]
    if not files:
        return ""
        
    max_ratio = 0
    best_file = ""
    
    # 清理掉特殊字符再进行比较，提高准确率
    clean_target = re.sub(r'[^\w\u4e00-\u9fa5]', '', target_book)
    
    for f in files:
        name_no_ext = os.path.splitext(f)[0]
        clean_file = re.sub(r'[^\w\u4e00-\u9fa5]', '', name_no_ext)
        
        # 计算相似度得分 (0 到 1 之间)
        ratio = difflib.SequenceMatcher(None, clean_target, clean_file).ratio()
        
        if ratio > max_ratio:
            max_ratio = ratio
            best_file = f
            
    # 只要相似度大于 0.2 (有一定汉字重合)，我们就认为匹配成功
    if max_ratio > 0.2:
        return os.path.join(directory, best_file)
        
    return ""

def get_next_section(toc: dict, book: str, current_section: str) -> str:
    """在目录树中，寻找 target_section 的【下一个标题】，作为切片的终止符"""
    book_toc = toc.get(book, {})
    unique_list = flatten_book_toc(book_toc)
            
    # 寻找当前节的下一个小节
    if current_section in unique_list:
        idx = unique_list.index(current_section)
        if idx + 1 < len(unique_list):
            return unique_list[idx + 1]
    return ""

def extract_block_from_file(filepath: str, start_title: str, end_title: str) -> str:
    """Python 智能切片：截取 start_title 和 end_title 之间的所有文本"""
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    start_norm = normalize_for_match(start_title)
    end_norm = normalize_for_match(end_title) if end_title else ""

    def scan_by_titles() -> str:
        # 严格边界：必须同时命中起始标题和结束标题
        if not start_norm or not end_norm:
            return ""

        start_idx = -1
        end_idx = -1

        for idx, line in enumerate(lines):
            line_norm = normalize_for_match(line)

            if start_idx == -1 and start_norm in line_norm:
                start_idx = idx
                continue

            if start_idx != -1 and idx > start_idx and end_norm in line_norm:
                end_idx = idx
                break

        if start_idx == -1 or end_idx == -1 or end_idx <= start_idx:
            return ""

        return "".join(lines[start_idx:end_idx])

    def scan_by_number_boundaries() -> str:
        # 严格边界：按章节编号找到起止边界，不允许截到文件尾
        start_num = extract_section_number(start_title)
        end_num = extract_section_number(end_title) if end_title else ""
        if not start_num or not end_num:
            return ""

        start_regex = re.compile(rf'^{re.escape(start_num)}(?:\.|[^0-9]|$)')
        end_regex = re.compile(rf'^{re.escape(end_num)}(?:\.|[^0-9]|$)')
        start_explain_regex = re.compile(rf'条文解释[^\d]*{re.escape(start_num)}(?:\.|[^0-9]|$)')
        end_explain_regex = re.compile(rf'条文解释[^\d]*{re.escape(end_num)}(?:\.|[^0-9]|$)')
        explanation_mode = is_explanation_section(start_title)

        start_idx = -1
        end_idx = -1

        for idx, line in enumerate(lines):
            line_norm = normalize_for_match(line)
            line_has_explain = "条文解释" in line_norm

            if explanation_mode:
                if start_idx == -1 and line_has_explain and start_explain_regex.search(line_norm):
                    start_idx = idx
                    continue

                if start_idx != -1 and idx > start_idx and line_has_explain and end_explain_regex.search(line_norm):
                    end_idx = idx
                    break
                continue

            if start_idx == -1 and start_regex.search(line_norm):
                start_idx = idx
                continue

            if start_idx != -1 and idx > start_idx and end_regex.search(line_norm):
                end_idx = idx
                break

        if start_idx == -1 or end_idx == -1 or end_idx <= start_idx:
            return ""

        return "".join(lines[start_idx:end_idx])

    # 1) 先用标题匹配
    extracted = scan_by_titles()
    if len(extracted.strip()) >= 10:
        return extracted

    # 2) 标题失败则用编号边界匹配（仍要求起止边界都存在）
    return scan_by_number_boundaries()

def run_agentic_search(user_question: str):
    print(f"\n🙋 用户问题：{user_question}")
    print("=" * 60)
    
    # ==========================================
    # 第一步：加载浓缩版目录
    # ==========================================
    if not os.path.exists("menu.json"):
        print("❌ 错误：当前目录下未找到 menu.json！")
        return
        
    with open("menu.json", "r", encoding="utf-8") as f:
        menu_toc = json.load(f)
        
    toc_str = json.dumps(menu_toc, ensure_ascii=False)
    
    # ==========================================
    # 第二步：大模型看目录（宏观路由）
    # ==========================================
    print(f"🧠 [阶段1] {FAST_MODEL} 正在翻阅全库总目录...")
    
    router_prompt = f"""你是一个顶级的工程规范图书管理员。
    用户的提问是："{user_question}"
    
    下面是我院所有规范的极简目录树：
    {toc_str}
    
    请你分析用户问题，从上述目录树中挑选出【最有可能包含答案的多个规范章节】。
    场景里可能是基础规范与专业规范共同给出约束，所以不要只选一个。
    重要：优先返回“正文章节 + 对应条文解释章节”的配对。
    如果你选了某个正文章节，且目录里存在“(条文解释)同名章节”，请一并返回该条文解释章节；
    如果你选了条文解释章节，也请把对应正文章节一并返回。
    必须原样抄写 JSON 目录里的 key。
    
    严格返回 JSON 格式，仅返回 JSON，不要解释文字。格式如下：
    {{
      "targets": [
        {{"book": "建筑与市政地基基础通用规范", "section": "6.3 筏形基础设计"}},
        {{"book": "建筑与市政地基基础通用规范", "section": "(条文解释)6.3 筏形基础设计"}},
        {{"book": "3建筑地基基础设计规范[附条文说明]", "section": "8.4高层建筑筏形基础"}},
        {{"book": "3建筑地基基础设计规范[附条文说明]", "section": "(条文解释)8.4高层建筑筏形基础"}}
      ]
    }}
    要求：
    1) 返回 2 到 {MAX_ROUTE_TARGETS} 个候选；
    2) 如果目录中确实只有 1 个强相关，也可返回 1 个；
    3) section 尽量具体到小节；
    4) 能成对返回时，优先返回成对章节，避免漏掉条文解释。
    """
    
    response = client.chat.completions.create(
        model=FAST_MODEL,
        messages=[{"role": "user", "content": router_prompt}],
        temperature=0.1
    )
    
    route_info = extract_json_from_text(response.choices[0].message.content)
    route_targets = normalize_route_targets(route_info, menu_toc)
    route_targets = expand_targets_with_explanations(route_targets, menu_toc)

    if not route_targets:
        print("❌ 路由失败，大模型未能从目录中锁定相关章节。")
        return

    print(f"🎯 锁定到 {len(route_targets)} 个候选章节：")
    for idx, target in enumerate(route_targets, 1):
        print(f"   {idx}. 📚《{target['book']}》 🔖 [{target['section']}]")
    
    # ==========================================
    # 第三步：黑科技模糊匹配文件名 + 精准文件切片
    # ==========================================
    collected_segments = []
    for target in route_targets:
        target_book = target["book"]
        target_section = target["section"]
        filepath = find_best_matching_file(target_book, "./specs/")

        if not filepath:
            print(f"⚠️ 找不到与《{target_book}》相似的文件，已跳过该目标。")
            continue

        print(f"📂 模糊匹配到实体文件: {os.path.basename(filepath)}")

        next_section = get_next_section(menu_toc, target_book, target_section)
        if not next_section:
            print(f"⚠️ 《{target_book}》[{target_section}] 未找到下一节边界，已跳过（避免截到文件尾）。")
            continue

        print(f"✂️  [阶段2] Python 正在截取 [{target_section}] 至 [{next_section}] 的原汁原味正文...")

        section_text = extract_block_from_file(filepath, target_section, next_section)
        if len(section_text.strip()) < 10:
            print(f"⚠️ 《{target_book}》[{target_section}] 截取失败或过短，已跳过。")
            continue

        if len(section_text) > MAX_SECTION_CHARS:
            section_text = section_text[:MAX_SECTION_CHARS] + "\n[...该章节内容较长，已截断...]"

        collected_segments.append({
            "book": target_book,
            "section": target_section,
            "text": section_text
        })

    if not collected_segments:
        print("⚠️ 所有候选章节都未成功截取，建议检查目录标题与 txt 正文标题的一致性。")
        return

    context_parts = []
    total_chars = 0
    for idx, seg in enumerate(collected_segments, 1):
        block = (
            f"【片段{idx}】\n"
            f"【规范名称】：《{seg['book']}》\n"
            f"【章节名称】：{seg['section']}\n"
            f"【规范原文】\n{seg['text']}\n"
        )
        if total_chars + len(block) > MAX_TOTAL_CONTEXT_CHARS:
            break
        context_parts.append(block)
        total_chars += len(block)

    if not context_parts:
        print("⚠️ 可用上下文为空，请调整截断阈值后重试。")
        return

    print(f"✅ 截取成功！共纳入 {len(context_parts)} 个章节片段。")
    
    # ==========================================
    # 第四步：大模型精读正文并作答
    # ==========================================
    print(f"\n📖 [阶段3] {REASONING_MODEL} 正在精读提取的片段，思考结论...\n")
    print("🤖 Ai 回答: ", end="")
    
    joined_context = "\n".join(context_parts)
    reader_prompt = f"""你是一个严谨的工程审查专家。以下是通过目录锁定并提取的规范正文片段。
    注意：这些片段来自多个规范章节，可能存在一般规定与专项规定并存的情况。
    
    {joined_context}
    
    【用户问题】：{user_question}
    
    回答要求：
    1. 完全基于上述提供的原汁原味片段进行回答，严禁瞎编。
    2. 优先引述原文中的关键规定，然后再作总结；若多条规定可能冲突，请解释适用关系。
    3. 对每个结论标注来源片段编号（如：片段1、片段2）。
    4. 如果所有片段都未明确提到用户问题，请明确回答“所给片段未提及”。
    """
    
    stream_response = client.chat.completions.create(
        model=REASONING_MODEL,
        messages=[{"role": "user", "content": reader_prompt}],
        temperature=0.3,
        stream=True
    )
    
    for chunk in stream_response:
        if chunk.choices and chunk.choices[0].delta.content:
            print(chunk.choices[0].delta.content, end='', flush=True)
    print("\n\n" + "=" * 60)

# ==========================================
# 运行入口
# ==========================================
if __name__ == "__main__":
    run_agentic_search("我想知道砌体规范(条文解释)4.1.6的内容")
