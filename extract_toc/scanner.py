#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
文档扫描模块

扫描指定文件夹，为每个文档生成独立的索引文件，并创建主索引。
"""

import os
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional

from .core import extract_chapter_lines_from_lines
from .pdf import extract_pdf_lines
from .markdown import extract_md_candidate_lines


def scan_text_file(file_path: str) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]]]:
    """扫描纯文本文件"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        return extract_chapter_lines_from_lines(lines)
    except Exception as e:
        print(f"警告: 无法读取文件 {file_path}: {e}")
        return [], []


def scan_pdf_file(file_path: str) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]]]:
    """扫描 PDF 文件"""
    try:
        lines = extract_pdf_lines(file_path)
        return extract_chapter_lines_from_lines(lines)
    except Exception as e:
        print(f"警告: 无法读取 PDF 文件 {file_path}: {e}")
        return [], []


def scan_markdown_file(file_path: str) -> Tuple[List[Tuple[str, int]], List[Tuple[str, int]]]:
    """扫描 Markdown 文件"""
    try:
        candidate_lines = extract_md_candidate_lines(file_path)
        if not candidate_lines:
            return [], []

        line_number_map = {}
        text_lines = []
        for idx, (text, original_line_number) in enumerate(candidate_lines):
            text_lines.append(text)
            line_number_map[idx] = original_line_number

        filtered_results = extract_chapter_lines_from_lines(text_lines)

        content_lines = []
        toc_lines = []

        for text, temp_line_number in filtered_results[0]:
            original_line_number = line_number_map[temp_line_number - 1]
            content_lines.append((text, original_line_number))

        for text, temp_line_number in filtered_results[1]:
            original_line_number = line_number_map[temp_line_number - 1]
            toc_lines.append((text, original_line_number))

        return content_lines, toc_lines
    except Exception as e:
        print(f"警告: 无法读取 Markdown 文件 {file_path}: {e}")
        return [], []


def filter_false_chapters(content_lines: List[Tuple[str, int]]) -> List[Tuple[str, int]]:
    """
    过滤误识别的章节（如列表项）

    使用两种策略：
    1. 行号位置过滤：前几章应该在文档前部
    2. 序号连续性检查：一级章节号应该连续且不重复

    注意：只对一级章节（如 "1 总则"）进行检查，不影响二级章节（如 "1.1 术语"）
    """
    import re

    if not content_lines:
        return []

    # 提取章节号的辅助函数
    def get_chapter_info(title):
        """返回 (一级章节号, 是否为一级章节)"""
        # 匹配格式：数字 或 数字.数字 或 数字.数字.数字（可能有空格）
        match = re.match(r'^(\d+)(?:\s*\.\s*(\d+))?(?:\s*\.\s*\d+)*\s+', title)
        if match:
            level1 = int(match.group(1))
            level2 = match.group(2)
            is_top_level = (level2 is None)  # 没有二级编号，说明是一级章节
            return level1, is_top_level
        return None, False

    # 方案2: 行号位置过滤（只针对一级章节），这部分其它对于一些小说等会产生误判，经验性太强。
    # 如果"第N章"出现的行号超过阈值，可能是误识别
    # position_thresholds = {
    #     1: 150,   # 第一章不应超过150行
    #     2: 400,   # 第二章不应超过400行
    #     # 3: 1000,   # 第三章不应超过800行
    # }

    # 方案3: 序号连续性检查（只针对一级章节）
    # 收集所有一级章节号及其出现位置
    top_level_occurrences = {}  # {chapter_num: [(index, line_num, title), ...]}
    for idx, (title, line_num) in enumerate(content_lines):
        chapter_num, is_top_level = get_chapter_info(title)
        if chapter_num and is_top_level:
            if chapter_num not in top_level_occurrences:
                top_level_occurrences[chapter_num] = []
            top_level_occurrences[chapter_num].append((idx, line_num, title))

    # 标记要删除的索引
    indices_to_remove = set()

    # 检查一级章节的重复
    for chapter_num, occurrences in top_level_occurrences.items():
        if len(occurrences) <= 1:
            continue

        # 如果同一一级章节号出现多次，保留第一个，删除后续的
        # （假设第一个是真正的章节，后续是误识别）
        for idx, line_num, title in occurrences[1:]:
            indices_to_remove.add(idx)

    # 检查前几章的行号位置（只针对一级章节）
    # for chapter_num, threshold in position_thresholds.items():
    #     if chapter_num in top_level_occurrences:
    #         for idx, line_num, title in top_level_occurrences[chapter_num]:
    #             if line_num > threshold:
    #                 indices_to_remove.add(idx)

    # 序号连续性检查：检查一级章节的连续性
    all_top_level_nums = sorted(top_level_occurrences.keys())
    if len(all_top_level_nums) > 1:
        # 找到最长的连续序列
        max_consecutive_start = all_top_level_nums[0]
        max_consecutive_length = 1
        current_start = all_top_level_nums[0]
        current_length = 1

        for i in range(1, len(all_top_level_nums)):
            if all_top_level_nums[i] == all_top_level_nums[i-1] + 1:
                current_length += 1
            else:
                if current_length > max_consecutive_length:
                    max_consecutive_length = current_length
                    max_consecutive_start = current_start
                current_start = all_top_level_nums[i]
                current_length = 1

        if current_length > max_consecutive_length:
            max_consecutive_length = current_length
            max_consecutive_start = current_start

        # 如果有明显的连续序列（长度>=3），则认为不在这个序列中的一级章节号可疑
        if max_consecutive_length >= 3:
            valid_range = set(range(max_consecutive_start, max_consecutive_start + max_consecutive_length))
            for chapter_num in all_top_level_nums:
                if chapter_num not in valid_range:
                    # 这个一级章节号不在主序列中，标记为可疑
                    for idx, line_num, title in top_level_occurrences[chapter_num]:
                        indices_to_remove.add(idx)

    # 过滤掉标记的索引
    filtered_lines = [
        (title, line_num)
        for idx, (title, line_num) in enumerate(content_lines)
        if idx not in indices_to_remove
    ]

    return filtered_lines


def build_chapter_tree(content_lines: List[Tuple[str, int]]) -> List[Dict]:
    """
    构建章节树结构

    保持标题完整（包含编号），只提取层级关系
    只保留一级和二级标题，过滤三级及以下
    """
    import re

    # 先过滤误识别的章节
    content_lines = filter_false_chapters(content_lines)

    chapters = []
    current_level1 = None
    current_level1_num = None  # 记录当前一级章节的编号
    current_level2 = None
    appendix_chapter = None  # 条文解释章节
    appendix_detected = False  # 是否已检测到条文解释

    # 第一遍：收集所有章节信息，用于检测条文解释的开始位置
    all_level1_nums = []
    all_items = []  # [(level1_num, level2_num, title, line_num), ...]

    for title, line_num in content_lines:
        match = re.match(r'^(\d+)(?:\s*\.\s*(\d+))?(?:\s*\.\s*(\d+))?\s+', title)
        if match:
            level1 = int(match.group(1))
            level2 = int(match.group(2)) if match.group(2) else None
            level3 = match.group(3)

            if not level3:  # 跳过三级标题
                all_items.append((level1, level2, title, line_num))
                if level2 is None:  # 一级标题
                    all_level1_nums.append(level1)

    # 检测条文解释的开始位置
    appendix_start_idx = None
    if len(all_level1_nums) >= 2:
        max_level1 = max(all_level1_nums)

        # 查找章节号突然从大跳到小的位置
        for i, (level1, level2, title, line_num) in enumerate(all_items):
            if level2 is not None:  # 二级标题
                # 检查是否是章节号回退（如从10.x跳到2.x）
                if i > 0:
                    prev_level1 = all_items[i-1][0]
                    # 如果当前二级标题的一级编号比前一个小很多（差距>=3），且前一个是较大的章节号
                    if prev_level1 >= max_level1 - 2 and level1 < prev_level1 - 2:
                        # 检查后续是否没有新的一级标题
                        has_new_level1 = False
                        for j in range(i, len(all_items)):
                            if all_items[j][1] is None:  # 发现一级标题
                                has_new_level1 = True
                                break

                        if not has_new_level1:
                            appendix_start_idx = i
                            break

    # 第二遍：构建章节树
    for idx, (level1, level2, title, line_num) in enumerate(all_items):
        # 如果检测到条文解释开始，且当前位置在开始位置之后
        if appendix_start_idx is not None and idx >= appendix_start_idx:
            if not appendix_detected:
                # 创建"条文解释"章节
                appendix_chapter = {
                    "title": "条文解释",
                    "line": line_num,
                    "sections": []
                }
                chapters.append(appendix_chapter)
                current_level1 = appendix_chapter
                appendix_detected = True

            # 所有二级标题都归入条文解释章节
            if level2 is not None:
                appendix_chapter['sections'].append({
                    "title": title,
                    "line": line_num
                })
            # 忽略一级标题（理论上不应该有）
        else:
            # 正常处理
            if level2 is not None:  # 二级标题
                # 只有当二级标题的一级编号与当前一级章节匹配时，才归为子节
                if current_level1 and current_level1_num == level1:
                    if 'sections' not in current_level1:
                        current_level1['sections'] = []
                    current_level2 = {
                        "title": title,
                        "line": line_num
                    }
                    current_level1['sections'].append(current_level2)
            else:  # 一级标题
                current_level1 = {
                    "title": title,
                    "line": line_num
                }
                current_level1_num = level1
                current_level2 = None
                chapters.append(current_level1)

    return chapters


def get_chapter_summary(chapters: List[Dict], max_items: int = 10) -> List[str]:
    """
    获取章节摘要（只包含一级章节标题）

    参数:
        chapters: 章节列表
        max_items: 最多返回的章节数（默认10个）

    返回:
        一级章节标题列表（前10章，去重）
    """
    import re

    if not chapters:
        return []

    # 只提取真正的一级章节（编号格式为 "数字 " 开头，不包含小数点）
    top_level_chapters = []
    seen_titles = set()
    seen_numbers = set()

    for chapter in chapters:
        title = chapter['title']
        # 匹配 "数字 " 开头的标题（如 "1 总则"），不匹配 "1.1" 或 "(1)" 等
        match = re.match(r'^(\d+)\s+(.+)', title)
        if match:
            chapter_num = int(match.group(1))
            chapter_text = match.group(2).strip()

            # 过滤条件：
            # 1. 标题文本长度至少2个字符（避免 "1 当" 这种片段）
            # 2. 章节号在合理范围内（1-30，规范一般不超过20章）
            # 3. 去重：跳过完全相同的标题
            # 4. 同一章节号只保留第一次出现（避免正文和附录重复）
            if (len(chapter_text) >= 2 and
                1 <= chapter_num <= 30 and
                title not in seen_titles and
                chapter_num not in seen_numbers):
                top_level_chapters.append(title)
                seen_titles.add(title)
                seen_numbers.add(chapter_num)

    total = len(top_level_chapters)

    if total == 0:
        return []

    # 返回前10章
    if total <= max_items:
        return top_level_chapters
    else:
        result = top_level_chapters[:max_items]
        result.append(f"... 共 {total} 章")
        return result


def scan_file(file_path: Path, output_dir: Path) -> Optional[Dict]:
    """
    扫描单个文件，生成独立的详细索引文件

    返回：文件的基本信息（用于主索引）
    """
    suffix = file_path.suffix.lower()

    # 根据文件类型选择处理方法
    if suffix == '.pdf':
        content_lines, toc_lines = scan_pdf_file(str(file_path))
        file_type = 'pdf'
    elif suffix in ['.md', '.markdown']:
        content_lines, toc_lines = scan_markdown_file(str(file_path))
        file_type = 'markdown'
    elif suffix in ['.txt', '.text']:
        content_lines, toc_lines = scan_text_file(str(file_path))
        file_type = 'text'
    else:
        return None

    # 构建章节树
    chapters = build_chapter_tree(content_lines)

    # 生成详细索引文件
    detail_index = {
        "file_name": file_path.name,
        "file_type": file_type,
        "file_size": file_path.stat().st_size,
        "modified_time": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat(),
        "scan_time": datetime.now().isoformat(),
        "chapters": chapters,
        "toc": [{"title": title, "line": line_num} for title, line_num in toc_lines],
        "statistics": {
            "total_chapters": len(content_lines),
            "total_toc_items": len(toc_lines),
            "chapter_tree_depth": get_tree_depth(chapters)
        }
    }

    # 保存详细索引（使用自定义格式化）
    detail_file = output_dir / f"{file_path.stem}.index.json"
    with open(detail_file, 'w', encoding='utf-8') as f:
        write_compact_json(detail_index, f)

    # 返回基本信息（用于主索引）
    basic_info = {
        "name": file_path.name,
        "type": file_type,
        "size": file_path.stat().st_size,
        "modified": datetime.fromtimestamp(file_path.stat().st_mtime).isoformat(),
        "total_chapters": len(content_lines),
        "chapter_summary": get_chapter_summary(chapters, max_items=10),
        "detail_index": f"{file_path.stem}.index.json"
    }

    return basic_info


def get_tree_depth(chapters: List[Dict], current_depth: int = 1) -> int:
    """计算章节树的深度"""
    max_depth = current_depth
    for chapter in chapters:
        if 'sections' in chapter:
            depth = get_tree_depth(chapter['sections'], current_depth + 1)
            max_depth = max(max_depth, depth)
        if 'subsections' in chapter:
            depth = get_tree_depth(chapter['subsections'], current_depth + 1)
            max_depth = max(max_depth, depth)
    return max_depth


def write_compact_json(data: Dict, file_handle):
    """
    以紧凑格式写入JSON，章节对象保持在一行
    """
    def format_chapter(chapter: Dict, indent: int = 0) -> str:
        """格式化单个章节对象"""
        spaces = "  " * indent

        # 基本信息保持在一行
        result = f'{spaces}{{"title": {json.dumps(chapter["title"], ensure_ascii=False)}, "line": {chapter["line"]}'

        # 如果有sections，递归格式化
        if 'sections' in chapter and chapter['sections']:
            result += ',\n' + spaces + '  "sections": [\n'
            section_strs = [format_chapter(sec, indent + 2) for sec in chapter['sections']]
            result += ',\n'.join(section_strs)
            result += '\n' + spaces + '  ]'

        # 如果有subsections，递归格式化
        if 'subsections' in chapter and chapter['subsections']:
            result += ',\n' + spaces + '  "subsections": [\n'
            subsection_strs = [format_chapter(sub, indent + 2) for sub in chapter['subsections']]
            result += ',\n'.join(subsection_strs)
            result += '\n' + spaces + '  ]'

        result += '}'
        return result

    # 开始写入
    file_handle.write('{\n')

    # 写入基本字段
    file_handle.write(f'  "file_name": {json.dumps(data["file_name"], ensure_ascii=False)},\n')
    file_handle.write(f'  "file_type": {json.dumps(data["file_type"], ensure_ascii=False)},\n')
    file_handle.write(f'  "file_size": {data["file_size"]},\n')
    file_handle.write(f'  "modified_time": {json.dumps(data["modified_time"], ensure_ascii=False)},\n')
    file_handle.write(f'  "scan_time": {json.dumps(data["scan_time"], ensure_ascii=False)},\n')

    # 写入chapters数组（紧凑格式）
    file_handle.write('  "chapters": [\n')
    if data['chapters']:
        chapter_strs = [format_chapter(ch, 2) for ch in data['chapters']]
        file_handle.write(',\n'.join(chapter_strs))
        file_handle.write('\n')
    file_handle.write('  ],\n')

    # 写入toc数组（紧凑格式）
    file_handle.write('  "toc": [')
    if data['toc']:
        file_handle.write('\n')
        toc_strs = [f'    {{"title": {json.dumps(item["title"], ensure_ascii=False)}, "line": {item["line"]}}}'
                    for item in data['toc']]
        file_handle.write(',\n'.join(toc_strs))
        file_handle.write('\n  ')
    file_handle.write('],\n')

    # 写入statistics
    file_handle.write('  "statistics": {\n')
    file_handle.write(f'    "total_chapters": {data["statistics"]["total_chapters"]},\n')
    file_handle.write(f'    "total_toc_items": {data["statistics"]["total_toc_items"]},\n')
    file_handle.write(f'    "chapter_tree_depth": {data["statistics"]["chapter_tree_depth"]}\n')
    file_handle.write('  }\n')

    file_handle.write('}\n')


def scan_folder(folder_path: str, recursive: bool = True, output_dir: Optional[str] = None) -> Dict:
    """
    扫描文件夹，生成分层索引

    参数:
        folder_path: 文件夹路径
        recursive: 是否递归扫描子文件夹
        output_dir: 输出目录，默认为 <folder_path>/.index/

    返回:
        主索引字典
    """
    folder = Path(folder_path)

    if not folder.exists():
        raise FileNotFoundError(f"文件夹不存在: {folder_path}")

    if not folder.is_dir():
        raise NotADirectoryError(f"不是文件夹: {folder_path}")

    # 设置输出目录
    if output_dir is None:
        output_dir = folder / ".index"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"开始扫描文件夹: {folder_path}")
    print(f"递归模式: {'是' if recursive else '否'}")
    print(f"输出目录: {output_dir}")

    # 构建主索引结构
    main_index = {
        "scan_time": datetime.now().isoformat(),
        "root_folder": str(folder.absolute()),
        "recursive": recursive,
        "folders": {}
    }

    # 扫描文件夹
    if recursive:
        for root_str, dirs, files in os.walk(folder):
            root = Path(root_str)
            relative_path = root.relative_to(folder)
            folder_key = str(relative_path) if str(relative_path) != '.' else 'root'

            print(f"\n扫描: {folder_key}")
            folder_info = scan_folder_files(root, files, output_dir)

            if folder_info['files']:
                main_index['folders'][folder_key] = folder_info
    else:
        files = [f.name for f in folder.iterdir() if f.is_file()]
        folder_info = scan_folder_files(folder, files, output_dir)
        main_index['folders']['root'] = folder_info

    # 统计信息
    total_files = sum(len(info['files']) for info in main_index['folders'].values())
    total_chapters = sum(
        file_info['total_chapters']
        for folder_info in main_index['folders'].values()
        for file_info in folder_info['files'].values()
    )

    main_index['summary'] = {
        "total_folders": len(main_index['folders']),
        "total_files": total_files,
        "total_chapters": total_chapters,
        "index_directory": str(output_dir.absolute())
    }

    print(f"\n扫描完成!")
    print(f"  文件夹数: {main_index['summary']['total_folders']}")
    print(f"  文件数: {main_index['summary']['total_files']}")
    print(f"  章节数: {main_index['summary']['total_chapters']}")

    # 保存主索引
    main_index_file = output_dir / "index.json"
    with open(main_index_file, 'w', encoding='utf-8') as f:
        json.dump(main_index, f, ensure_ascii=False, indent=2)
    print(f"\n主索引已保存到: {main_index_file.absolute()}")
    print(f"详细索引保存在: {output_dir.absolute()}")

    return main_index


def scan_folder_files(folder_path: Path, file_names: List[str], output_dir: Path) -> Dict:
    """扫描文件夹中的文件"""
    folder_info = {
        "path": str(folder_path),
        "files": {}
    }

    for file_name in file_names:
        file_path = folder_path / file_name

        if not file_path.is_file():
            continue

        file_info = scan_file(file_path, output_dir)

        if file_info:
            folder_info['files'][file_name] = file_info
            print(f"  [OK] {file_name} ({file_info['total_chapters']} 章节)")
        else:
            print(f"  [--] {file_name} (不支持的格式)")

    return folder_info


def main():
    """命令行入口"""
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    if len(sys.argv) < 2:
        print("用法: python -m src.scan_documents <文件夹路径> [输出目录] [--no-recursive]")
        print("示例: python -m src.scan_documents ./docs")
        print("      python -m src.scan_documents ./docs ./output_index")
        print("      python -m src.scan_documents ./docs --no-recursive")
        return

    folder_path = sys.argv[1]
    output_dir = None
    recursive = '--no-recursive' not in sys.argv

    # 解析输出目录参数
    if len(sys.argv) > 2 and not sys.argv[2].startswith('--'):
        output_dir = sys.argv[2]

    try:
        scan_folder(folder_path, recursive=recursive, output_dir=output_dir)
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
