import re
import os
import logging
from typing import Tuple, List, Dict, Optional

# 配置日志，方便在控制台看到什么时候触发了大模型兜底
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SmartExpander:
    def __init__(self, llm_client=None, fast_model: str = "deepseek-chat", max_radius: int = 15):
        """
        混合智能展卷引擎
        :param llm_client: OpenAI 兼容的客户端实例 (如 client = OpenAI(api_key="..."))
        :param fast_model: 用于兜底的快速大模型名称 (推荐 gpt-4o-mini, deepseek-chat, qwen-turbo)
        :param max_radius: 规则引擎向上/向下展卷的最大搜索行数
        """
        self.llm_client = llm_client
        self.fast_model = fast_model
        self.max_radius = max_radius

    def _is_boundary(self, text: str) -> bool:
        """定义什么是语义边界"""
        patterns = [
            r'^#{1,6}\s+',                                  # Markdown 标题 (如 ### 5.2)
            r'^第[一二三四五六七八九十百千万\d]+[条文章节]', # 法律/规范条款 (如 第5.2.1条)
            r'^\d+\.\d+\.\d+'                               # 编号 (如 5.2.1)
        ]
        return any(re.match(p, text.strip()) for p in patterns)

    def expand(self, filepath: str, hit_line_num: int) -> Dict:
        """
        主入口：执行混合展卷逻辑
        返回包含提取文本及元数据的字典。
        """
        if not os.path.exists(filepath):
            return {"text": "", "source": "error", "reason": "File not found"}

        with open(filepath, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        hit_idx = hit_line_num - 1
        if hit_idx < 0 or hit_idx >= len(lines):
            return {"text": "", "source": "error", "reason": "Index out of bounds"}

        hit_text = lines[hit_idx].strip()

        # ==========================================
        # 阶段一：Fast Path (纯代码正则测边)
        # ==========================================
        start_idx, end_idx, hit_max_up, hit_max_down = self._rule_based_expand(lines, hit_idx)
        fast_snippet = "".join(lines[start_idx:end_idx + 1])

        # ==========================================
        # 阶段二：评估是否触发 Fallback (大模型兜底)
        # ==========================================
        needs_fallback = False
        fallback_reason = ""

        # 触发条件 1：向上或向下寻找边界时，拉到了最大极限依然没找到（说明是无规则长文本）
        if hit_max_up or hit_max_down:
            needs_fallback = True
            fallback_reason = "达到最大搜索半径仍未找到边界"
            
        # 触发条件 2：文本中包含强关联悬挂词，但没有捕捉到表格体（Markdown 表格符 |）
        elif re.search(r'(见下表|如下|规定：|见表[:：]?)$', fast_snippet.strip()) and '|' not in fast_snippet:
            needs_fallback = True
            fallback_reason = "检测到表格引用但未捕获表格内容"
            
        # 触发条件 3：截取出的文本太短
        elif len(fast_snippet.strip()) < 20:
            needs_fallback = True
            fallback_reason = "提取文本过短（可能是孤立标题）"
        # 触发条件 4：文本太长且没有明确的段落结构（需排除有编号的规范文档）
        elif len(fast_snippet) > 800:
            # 检查是否有编号结构（如 5.2.1、第X条等）
            has_numbering = bool(re.search(r'(\d+\.\d+\.\d+|第[一二三四五六七八九十百千万\d]+[条文章节])', fast_snippet))
            # 检查是否有段落分隔
            has_paragraphs = '\n\n' in fast_snippet or fast_snippet.count('\n') > 5
            
            if not has_numbering and not has_paragraphs:
                needs_fallback = True
                fallback_reason = "文本过长且缺乏结构化标记"

        # 如果不需要 Fallback，直接返回极速结果！
        if not needs_fallback or not self.llm_client:
            return {
                "text": fast_snippet,
                "source": "rule_based",
                "start_line": start_idx + 1,
                "end_line": end_idx + 1
            }

        # ==========================================
        # 阶段三：Slow Path (Fast LLM 纯语义萃取)
        # ==========================================
        logger.info(f"触发 LLM 语义提取 - 原因: {fallback_reason}")
        llm_snippet = self._llm_semantic_extract(lines, hit_idx, hit_text)

        # 如果大模型提取失败（由于网络或风控），安全退回到正则结果
        if not llm_snippet:
            logger.warning("LLM 提取失败，退回规则引擎结果。")
            return {
                "text": fast_snippet,
                "source": "rule_based_fallback",
                "start_line": start_idx + 1,
                "end_line": end_idx + 1
            }

        return {
            "text": llm_snippet,
            "source": "llm_extracted",
            "hit_line": hit_line_num  # 大模型破坏了行号，所以只保留命中锚点行用于上下文去重
        }

    def _rule_based_expand(self, lines: List[str], hit_idx: int) -> Tuple[int, int, bool, bool]:
        """执行正则上下展卷，并返回是否触碰到了最大半径警戒线"""
        hit_text = lines[hit_idx].strip()
        start_idx = hit_idx
        end_idx = hit_idx
        hit_max_up = False
        hit_max_down = False

        # 向上展卷
        if not self._is_boundary(hit_text):
            for i in range(hit_idx - 1, max(-1, hit_idx - self.max_radius - 1), -1):
                if self._is_boundary(lines[i]):
                    start_idx = i
                    break
            else:
                hit_max_up = True # 循环正常结束说明没 break，即没找到边界

        # 向下展卷
        needs_downward = (len(hit_text) < 80 or 
                          re.search(r'(:|：|如下|规定|见表)$', hit_text) or 
                          hit_text.startswith('|'))

        if needs_downward:
            for i in range(hit_idx + 1, min(len(lines), hit_idx + self.max_radius + 1)):
                line_str = lines[i].strip()
                if self._is_boundary(line_str):
                    break
                end_idx = i
                if line_str.startswith('|'):
                    continue
                if not line_str and not hit_text.endswith((':', '：')):
                    break
            else:
                hit_max_down = True

        return start_idx, end_idx, hit_max_up, hit_max_down

    def _llm_semantic_extract(self, lines: List[str], hit_idx: int, hit_text: str) -> Optional[str]:
        """调用大模型进行上下文文意提纯"""
        # 暴力截取更宽泛的上下文（比如上下 25 行，共 51 行）
        wider_radius = self.max_radius * 2
        start_idx = max(0, hit_idx - wider_radius)
        end_idx = min(len(lines), hit_idx + wider_radius + 1)
        raw_context = "".join(lines[start_idx:end_idx])

        prompt = f"""以下是规范/工程文档的局部片段，由于解析问题可能包含杂乱排版。
用户命中的核心内容（锚点）是："{hit_text}"

请你扮演文档清洗专家，理解语义并提取出与该行属于【同一个逻辑整体】的完整文本（例如：该条款的完整段落、前置条件，以及其附带的表格）。

严格约束：
1. 逐字照抄原文，严禁任何改写、润色！
2. 严禁包含其他无关的章节或条款内容。
3. 如果原文存在 Markdown 表格结构，必须完整保留 `|---|---|` 等符号。
4. 将提取出的最终内容放置在 <extract> 和 </extract> 标签之间。

原始杂乱片段：
<raw_document>
{raw_context}
</raw_document>"""

        try:
            response = self.llm_client.chat.completions.create(
                model=self.fast_model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0 # 强制 0.0，杜绝大模型的创造性，逼迫它复制粘贴
            )
            content = response.choices[0].message.content
            
            # 使用正则安全提取 XML 标签内的内容
            match = re.search(r'<extract>\s*(.*?)\s*</extract>', content, re.DOTALL)
            if match:
                return match.group(1).strip()
            return None
            
        except Exception as e:
            logger.error(f"LLM 兜底提炼异常: {str(e)}")
            return None