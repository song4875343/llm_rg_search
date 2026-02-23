#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
结构规范目录JS文件转JSON脚本
将包含var list_1 = [...] 格式的JS文件转换为简化的JSON格式
"""

import json
import ast
import sys
import os


def clean_title(title):
    """清理标题中的特殊字符"""
    return title.replace('\u2002\u2002', ' ').replace('．', '.')


def collect_nested_titles(nodes, prefix=""):
    """递归收集任意层级目录标题（保序去重），可附加前缀"""
    titles = []

    def walk(node):
        if not isinstance(node, dict):
            return

        title = node.get('title')
        if isinstance(title, str) and title.strip():
            t = clean_title(title)
            if prefix:
                t = f"{prefix}{t}"
            titles.append(t)

        children = node.get('content')
        if isinstance(children, list):
            for child in children:
                walk(child)

    if isinstance(nodes, list):
        for item in nodes:
            walk(item)

    seen = set()
    unique_titles = []
    for t in titles:
        if t in seen:
            continue
        seen.add(t)
        unique_titles.append(t)
    return unique_titles


def merge_unique(base_list, extra_list):
    """将 extra_list 合并进 base_list，保序去重"""
    seen = set(base_list)
    merged = list(base_list)
    for item in extra_list:
        if item in seen:
            continue
        seen.add(item)
        merged.append(item)
    return merged


def explanation_prefix(occur: int) -> str:
    """
    解释目录前缀:
    occur=2 -> (条文解释)
    occur>2 -> (条文解释2)、(条文解释3)...
    """
    if occur <= 1:
        return ""
    if occur == 2:
        return "(条文解释)"
    return f"(条文解释{occur - 1})"


def parse_js_file(input_path):
    """解析JS文件内容"""
    with open(input_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 提取JS数组内容
    js_content = content.replace("var list_1 =", "").strip()
    if js_content.endswith(';'):
        js_content = js_content[:-1]

    # 使用ast.literal_eval安全解析
    data = ast.literal_eval(js_content)
    return data


def simplify_structure(data):
    """简化目录结构"""
    result = {}

    for book in data:
        book_title = book['title']
        result[book_title] = {}
        chapter_seen_count = {}

        for chapter in book['content']:
            chapter_title = clean_title(chapter['title'])
            occur = chapter_seen_count.get(chapter_title, 0) + 1
            chapter_seen_count[chapter_title] = occur

            # 同名章节第2次及以后视为条文解释目录，避免覆盖
            chapter_prefix = explanation_prefix(occur)
            chapter_key = f"{chapter_prefix}{chapter_title}" if chapter_prefix else chapter_title

            # 递归获取小节列表（包含深层目录）
            sections = collect_nested_titles(chapter.get('content', []), chapter_prefix)

            if chapter_key in result[book_title]:
                result[book_title][chapter_key] = merge_unique(
                    result[book_title][chapter_key],
                    sections
                )
            else:
                result[book_title][chapter_key] = sections

    return result


def convert_js_to_json(input_path, output_path=None):
    """
    将JS文件转换为JSON文件

    参数:
        input_path: 输入的JS文件路径
        output_path: 输出的JSON文件路径，默认为None（自动命名）
    """
    # 解析JS文件
    data = parse_js_file(input_path)
    print(f"成功解析，共有 {len(data)} 本书")

    # 简化结构
    result = simplify_structure(data)

    # 确定输出路径
    if output_path is None:
        base_name = os.path.splitext(input_path)[0]
        output_path = base_name + '.json'

    # 保存JSON文件
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\nJSON文件已保存到: {output_path}")
    print(f"共包含 {len(result)} 本书的目录信息\n")

    # 列出所有书名
    print("书名列表:")
    for i, book_name in enumerate(result.keys(), 1):
        chapter_count = len(result[book_name])
        print(f"  {i}. {book_name} ({chapter_count} 章)")

    return output_path


def main():
    """主函数"""
    if len(sys.argv) < 2:
        print("用法: python convert_catalog.py <输入JS文件路径> [输出JSON文件路径]")
        print("示例: python convert_catalog.py gf_结构规范_1234_m.js")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None

    if not os.path.exists(input_path):
        print(f"错误: 文件不存在 - {input_path}")
        sys.exit(1)

    convert_js_to_json(input_path, output_path)
def tt():
    convert_js_to_json('gf_结构规范(笔记本).js')

if __name__ == '__main__':
    # main()
    tt()
