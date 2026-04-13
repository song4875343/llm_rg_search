import re
import sys
from pathlib import Path

from .core import extract_chapter_lines_from_lines


def extract_md_candidate_lines(md_path):
    """
    从 Markdown 文件中提取可能的章节行（带#标题或加粗文本的行）。

    参数:
        md_path: Markdown 文件路径

    返回:
        list[tuple]: [(line_text, line_number), ...] 候选章节行及其行号
    """
    candidate_lines = []
    
    # 匹配 # 标题的正则
    heading_pattern = re.compile(r"^#{1,6}\s+(.+)$")
    # 匹配加粗文本的正则（**text** 或 __text__）
    bold_pattern = re.compile(r"\*\*(.+?)\*\*|__(.+?)__")
    
    try:
        with open(md_path, 'r', encoding='utf-8') as f:
            for line_number, line in enumerate(f, start=1):
                line = line.rstrip('\n\r')
                
                # 检查是否是 # 标题
                heading_match = heading_pattern.match(line)
                if heading_match:
                    # 提取标题文本（去掉 # 符号）
                    title_text = heading_match.group(1).strip()
                    candidate_lines.append((title_text, line_number))
                    continue
                
                # 检查是否包含加粗文本
                bold_matches = bold_pattern.findall(line)
                if bold_matches:
                    # 提取所有加粗内容并合并
                    bold_texts = []
                    for match in bold_matches:
                        # match 是 tuple (group1, group2)，取非空的那个
                        text = match[0] if match[0] else match[1]
                        bold_texts.append(text)
                    
                    if bold_texts:
                        # 如果一行有多个加粗，用空格连接
                        combined_text = ' '.join(bold_texts)
                        candidate_lines.append((combined_text, line_number))
    
    except FileNotFoundError:
        print(f"错误: 文件未找到 '{md_path}'")
        return []
    except Exception as e:
        print(f"错误: 读取文件失败 - {e}")
        return []
    
    return candidate_lines


def extract_chapter_lines_from_md(md_path):
    """
    从 Markdown 文件中提取章节相关的文本行。

    参数:
        md_path: Markdown 文件路径

    返回:
        tuple: (content_lines, toc_lines)
        - content_lines: list of (line_text, line_number) tuples - 正文章节行
        - toc_lines: list of (line_text, line_number) tuples - 目录行
    """
    # 1. 提取候选行（带#或加粗的行）
    candidate_lines = extract_md_candidate_lines(md_path)
    
    if not candidate_lines:
        return [], []
    
    # 2. 将候选行转换为纯文本列表，保持行号映射
    # 创建一个临时的行号映射字典
    line_number_map = {}
    text_lines = []
    for idx, (text, original_line_number) in enumerate(candidate_lines):
        text_lines.append(text)
        line_number_map[idx] = original_line_number
    
    # 3. 调用核心过滤逻辑
    filtered_results = extract_chapter_lines_from_lines(text_lines)
    
    # filtered_results 返回的是 (content_lines, toc_lines)
    # 每个都是 [(text, temp_line_number), ...] 格式
    # temp_line_number 是在 text_lines 中的索引（从1开始）
    
    # 4. 将临时行号映射回原始 MD 文件的行号
    content_lines = []
    toc_lines = []
    
    for text, temp_line_number in filtered_results[0]:  # content_lines
        # temp_line_number 是从1开始的，需要转换为0-based索引
        original_line_number = line_number_map[temp_line_number - 1]
        content_lines.append((text, original_line_number))
    
    for text, temp_line_number in filtered_results[1]:  # toc_lines
        original_line_number = line_number_map[temp_line_number - 1]
        toc_lines.append((text, original_line_number))
    
    return content_lines, toc_lines


def main():
    """
    测试入口：处理指定目录下的 Markdown 文件，并打印提取结果。
    """
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    # 默认目录，可以修改
    md_dir = Path(r"./specs/")
    
    # 支持命令行参数指定目录
    if len(sys.argv) > 1:
        md_dir = Path(sys.argv[1])
    
    md_files = sorted(md_dir.glob("*.md"))

    if not md_files:
        print(f"未找到 Markdown 文件: {md_dir}")
        print("提示: 可以使用 python -m src.extract_toc.markdown <目录路径> 指定目录")
        return

    # 处理前3个文件
    for md_file in md_files[:3]:
        print(f"\n{'=' * 80}")
        print(f"文件: {md_file}")
        file_start = time.perf_counter()
        
        try:
            content_lines, toc_lines = extract_chapter_lines_from_md(str(md_file))
        except Exception as exc:
            print(f"处理失败: {exc}")
            import traceback
            traceback.print_exc()
            continue

        print(f"\n{'=' * 80}")
        print(f"正文章节行 (共 {len(content_lines)} 行):")
        print('=' * 80)
        for line_text, line_number in content_lines:
            print(f"[行 {line_number:4d}] {line_text}")

        print(f"\n{'=' * 80}")
        print(f"目录行 (共 {len(toc_lines)} 行):")
        print('=' * 80)
        for line_text, line_number in toc_lines:
            print(f"[行 {line_number:4d}] {line_text}")

        file_elapsed = time.perf_counter() - file_start
        print(f"\n[timing] file_total: {file_elapsed:.3f}s | {md_file}")


if __name__ == "__main__":
    main()
