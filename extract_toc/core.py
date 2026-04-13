import re
import unicodedata


_CHINESE_NUMERALS = "一二三四五六七八九十百千万零〇两"
_TOC_TRAILING_PAGE_PATTERN = re.compile(r"(?:\.{2,}|…{1,}|·{2,}|\s{2,}|\t+)\s*(\d+)\s*$")
_PURE_TRAILING_PAGE_PATTERN = re.compile(r"\s+(\d+)\s*$")
_ARABIC_SECTION_VALUE_PATTERN = re.compile(r"^(\d+)(?:\.\d+)*(?:\s*[-.)）]\s*|\s+)")
_CHINESE_PREFIX_PATTERN = re.compile(
    rf"^(?:第[{_CHINESE_NUMERALS}\d]+[章节部分篇卷条]|"
    rf"[{_CHINESE_NUMERALS}]+(?:[、.)）]|\s+))"
)
_SECTION_PREFIX_STRIP_PATTERN = re.compile(r"^\d+(?:\.\d+)*(?:\s*[-.)）]\s*|\s+)?")
_CHINESE_SECTION_PREFIX_STRIP_PATTERN = re.compile(
    rf"^(?:第[{_CHINESE_NUMERALS}\d]+[章节部分篇卷条]\s*|"
    rf"[{_CHINESE_NUMERALS}]+(?:[、.)）]|\s+))"
)
_CHINESE_DISALLOWED_PUNCTUATION_PATTERN = re.compile(
    r"[，。！？；：】【『』「」《》〈〉“”‘’'\",,!?;:|\\/@#$%^&*_+=~`]"
)
_NON_CHINESE_DISALLOWED_PUNCTUATION_PATTERN = re.compile(
    r"[，。！？；】【『』「」《》〈〉“”‘’'\",,!?;|\\/@#$%^&*_+=~`]"
)


def normalize_line(line):
    """
    归一化单行文本。
    """
    if line is None:
        return ""
    line = unicodedata.normalize("NFKC", str(line))
    line = line.replace("\u00a0", " ").replace("\u3000", " ")
    line = line.replace("．", ".")
    line = line.replace("․", ".")
    line = re.sub(r"\s+", " ", line)
    return line.strip()


def starts_with_chapter_number(line):
    """
    判断一行是否以常见章节编号开头。
    """
    stripped = line.lstrip()
    if not stripped:
        return False
    
    # 排除以 "数字)" 开头的列表项（如 "1)"、"2)" 等）
    if re.match(r'^\d+\)', stripped):
        return False
    
    # 排除以 "数字.数字(" 开头的表格注释（如 "2.1(有相邻建筑影响)"）
    if re.match(r'^\d+\.\d+\(', stripped):
        return False
    
    # 排除以 "数字.数字mm" 或 "数字.数字cm" 等单位结尾的数据（如 "6.75mm"）
    if re.match(r'^\d+\.\d+(?:mm|cm|m|km|kg|g|t|°C|℃|kPa|MPa|GPa)\s*$', stripped):
        return False
    
    # 排除以 "数字." 开头但后面紧跟中文（没有空格）的列表项（如 "2.未风化-微风化..."）
    if re.match(r'^\d+\.[\u4e00-\u9fff]', stripped):
        return False
    
    # 排除中文数字后面跟顿号的列表项（如 "一、等效均布地面荷载"）
    if re.match(rf'^[{_CHINESE_NUMERALS}]+、', stripped):
        return False
    
    # 排除中文数字后面直接跟中文（没有空格、点号等分隔符）的情况
    # 如 "一级"、"二级"、"三级"、"一般的" 等
    if re.match(rf'^[{_CHINESE_NUMERALS}]+[\u4e00-\u9fff]', stripped):
        return False
    
    arabic_match = _ARABIC_SECTION_VALUE_PATTERN.match(stripped)
    if arabic_match:
        first_number = int(arabic_match.group(1))
        return 0 < first_number <= 50
    if stripped[0] in _CHINESE_NUMERALS:
        return True
    return bool(_CHINESE_PREFIX_PATTERN.match(stripped))


def extract_title_body(line):
    """
    去掉章节编号和目录页码，只保留标题正文部分。
    """
    candidate = _SECTION_PREFIX_STRIP_PATTERN.sub("", line)
    candidate = _CHINESE_SECTION_PREFIX_STRIP_PATTERN.sub("", candidate)
    candidate = _TOC_TRAILING_PAGE_PATTERN.sub("", candidate).strip()
    candidate = _PURE_TRAILING_PAGE_PATTERN.sub("", candidate).strip()
    return candidate


def has_disallowed_punctuation(line):
    """
    判断一行是否包含不允许的标点。
    """
    sanitized = re.sub(r"\d+(?:\.\d+)+", "", line)
    sanitized = re.sub(r"(?:\.{2,}|…{1,}|·{2,})\s*\d*\s*$", "", sanitized)
    body = extract_title_body(sanitized)

    if re.search(r"[\u4e00-\u9fff]", body):
        if ":" in body or "：" in body:
            return True
        return bool(_CHINESE_DISALLOWED_PUNCTUATION_PATTERN.search(sanitized))

    return bool(_NON_CHINESE_DISALLOWED_PUNCTUATION_PATTERN.search(sanitized))


def has_meaningful_title_content(line):
    """
    排除只有编号、没有实际标题内容的行。
    """
    candidate = extract_title_body(line)
    candidate = re.sub(r"[.\-\s]+", "", candidate)
    return bool(candidate)


def get_arabic_section_number(line):
    """
    提取章节编号中的第一个阿拉伯数字。
    """
    match = _ARABIC_SECTION_VALUE_PATTERN.match(line.lstrip())
    if not match:
        return None
    return int(match.group(1))


def is_simple_top_level_arabic_line(line):
    """
    判断是否为简单的一级阿拉伯数字标题。
    """
    return bool(re.match(r"^\d+(?:\s+|[-.)）]\s*)", line)) and not bool(re.match(r"^\d+\.\d+", line))


def tokenize_title_words(text):
    """
    提取标题中的中英文词，用于简单相似度比对。
    """
    words = re.findall(r"[A-Za-z]+|[\u4e00-\u9fff]+", text.lower())
    return {word for word in words if len(word) > 1}


def has_title_case_signal(text):
    """
    判断标题正文是否具有标题特征。
    """
    if re.search(r"[\u4e00-\u9fff]", text):
        return True
    return bool(re.search(r"\b[A-Z][A-Za-z-]*\b", text))


def looks_like_noise(line):
    """
    判断一行是否更像噪声而不是章节标题。
    """
    body = extract_title_body(line)
    if not body:
        return True

    compact_body = body.replace(" ", "")
    if len(compact_body) <= 1:
        return True

    tokens = re.findall(r"[A-Za-z]+|[\u4e00-\u9fff]+|[^\w\s]", body)
    if tokens:
        alpha_tokens = [token for token in tokens if re.fullmatch(r"[A-Za-z]+|[\u4e00-\u9fff]+", token)]
        english_tokens = [token for token in alpha_tokens if re.fullmatch(r"[A-Za-z]+", token)]
        symbol_tokens = [token for token in tokens if not re.fullmatch(r"[A-Za-z]+|[\u4e00-\u9fff]+", token)]
        # 只针对英文单字符的情况判定为噪声（避免误删被空格分开的中文标题如"场 地"）
        if english_tokens and all(len(token) == 1 for token in english_tokens) and len(english_tokens) + len(symbol_tokens) <= 6 and not re.search(r"[\u4e00-\u9fff]", body):
            return True

    # 对于中文标题，去掉数字后如果只剩一个字符则判定为噪声（如"三级"去掉数字后只剩"级"）
    if re.search(r"[\u4e00-\u9fff]", body):
        # 提取所有中文字符
        chinese_chars = re.findall(r"[\u4e00-\u9fff]", body)
        if len(chinese_chars) == 1:
            return True

    if ".pdf" in body.lower() or "http://" in body.lower() or "https://" in body.lower():
        return True
    if len(body) > 120:
        return True
    if body.startswith(("-", ".", ")", "）")):
        return True
    if re.match(r"^\d+-[A-Za-z]", line):
        return True
    if re.search(r"\.\d+$", line) and not _TOC_TRAILING_PAGE_PATTERN.search(line):
        return True

    letter_count = sum(char.isalpha() for char in body)
    digit_count = sum(char.isdigit() for char in body)
    numeric_tokens = re.findall(r"\b\d+(?:\.\d+)?\b", body)
    word_tokens = re.findall(r"[A-Za-z]+(?:-[A-Za-z]+)?|[\u4e00-\u9fff]+", body)

    if letter_count == 0 and not re.search(r"[\u4e00-\u9fff]", body):
        return True
    if digit_count > letter_count and len(numeric_tokens) >= 2:
        return True
    if len(numeric_tokens) >= 3 and len(word_tokens) <= 2:
        return True
    if len(numeric_tokens) >= 2 and len(word_tokens) >= 6:
        return True
    if re.match(r"^\d+\s+[a-z]", line):
        return True
    if re.match(r"^\d+(?:\.\d+)?\s+[a-z]", line):
        return True
    if re.match(r"^\d+\.\s+[a-z]", line):
        return True
    if re.match(r"^\d+[A-Za-z]", line):
        return True
    if re.match(r"^\d{4}(?:\D|$)", line):
        return True
    if body.count("-") >= 2 and digit_count >= 4:
        return True

    if re.match(r"^\d+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?$", line):
        first_number = get_arabic_section_number(line)
        if first_number is not None and first_number >= 10:
            return True

    if re.search(r"\b(?:figure|table|appendix)\s+\d", body.lower()):
        return True

    lowercase_words = re.findall(r"\b[a-z][a-z-]*\b", body)
    uppercase_words = re.findall(r"\b[A-Z][A-Za-z-]*\b", body)
    # 排除中英文对照标题（包含中文时允许多个小写词）
    if len(lowercase_words) >= 4 and len(uppercase_words) <= 1 and not re.search(r"[\u4e00-\u9fff]", body):
        return True
    if re.match(r"^\d+\s+", line) and len(lowercase_words) >= 3:
        return True
    if re.match(r"^\d+\s+", line) and len(word_tokens) >= 5 and len(lowercase_words) >= 2:
        return True
    if line.endswith(tuple(str(i) for i in range(10))) and not has_title_case_signal(body):
        return True

    return False


def is_chapter_candidate(line):
    """
    判断一行是否像章节标题或目录项。
    """
    normalized = normalize_line(line)
    if not normalized:
        return False
    if normalized.isdigit():
        return False
    if not starts_with_chapter_number(normalized):
        return False
    if not has_meaningful_title_content(normalized):
        return False
    if has_disallowed_punctuation(normalized):
        return False
    if looks_like_noise(normalized):
        return False
    return True


def is_toc_line(line):
    """
    判断一行是否更像目录行。
    """
    normalized = normalize_line(line)
    if _TOC_TRAILING_PAGE_PATTERN.search(normalized):
        return True

    if _PURE_TRAILING_PAGE_PATTERN.search(normalized):
        body = _PURE_TRAILING_PAGE_PATTERN.sub("", normalized).rstrip()
        if not body or body[-1].isdigit():
            return False
        title_body = extract_title_body(normalized)
        if not has_title_case_signal(title_body):
            return False
        if title_body.startswith(("(", "（")) and title_body[:2].lower() == title_body[:2]:
            return False
        return True

    return False


def split_chapter_lines(lines):
    """
    将章节候选行拆分为正文章节行和目录行。
    
    返回:
        tuple: (content_lines, toc_lines)
        - content_lines: list of (line_text, line_number) tuples
        - toc_lines: list of (line_text, line_number) tuples
    """
    content_lines = []
    toc_lines = []
    candidates = []

    for line_number, raw_line in enumerate(lines, start=1):
        normalized = normalize_line(raw_line)
        if not is_chapter_candidate(normalized):
            continue
        candidates.append((normalized, line_number))

    for normalized, line_number in candidates:
        if is_toc_line(normalized):
            toc_lines.append((normalized, line_number))
        else:
            content_lines.append((normalized, line_number))

    toc_top_numbers = {
        number for number in (get_arabic_section_number(line) for line, _ in toc_lines) if number is not None
    }
    if toc_top_numbers:
        toc_word_map = {
            number: tokenize_title_words(extract_title_body(line))
            for line, _ in toc_lines
            for number in [get_arabic_section_number(line)]
            if number is not None
        }
        content_lines = [
            (line, line_number) for line, line_number in content_lines
            if get_arabic_section_number(line) is None or get_arabic_section_number(line) in toc_top_numbers
        ]
        toc_lines = [
            (line, line_number) for line, line_number in toc_lines
            if get_arabic_section_number(line) is None or get_arabic_section_number(line) in toc_top_numbers
        ]
        content_lines = [
            (line, line_number) for line, line_number in content_lines
            if not is_simple_top_level_arabic_line(line)
            or get_arabic_section_number(line) not in toc_word_map
            or bool(tokenize_title_words(extract_title_body(line)) & toc_word_map[get_arabic_section_number(line)])
        ]

    # 去重：对于标题完全相同且行号相差3行以内的章节，只保留第一个
    def deduplicate_nearby_titles(lines_list):
        if not lines_list:
            return lines_list
        
        result = []
        prev_title = None
        prev_line_num = None
        
        for title, line_num in lines_list:
            # 如果标题与前一个相同，且行号相差3行以内，则跳过
            if prev_title == title and prev_line_num is not None and abs(line_num - prev_line_num) <= 3:
                continue
            
            result.append((title, line_num))
            prev_title = title
            prev_line_num = line_num
        
        return result
    
    content_lines = deduplicate_nearby_titles(content_lines)
    toc_lines = deduplicate_nearby_titles(toc_lines)

    return content_lines, toc_lines


def extract_chapter_lines_from_lines(lines):
    """
    从已经清洗好的文本行列表中提取章节和目录行。

    参数:
        lines: 纯净文本行列表

    返回:
        tuple: (content_lines, toc_lines)
        - content_lines: list of (line_text, line_number) tuples
        - toc_lines: list of (line_text, line_number) tuples
    """
    return split_chapter_lines(lines)


def main():
    """
    从文本文件读取内容并提取章节行。
    
    用法:
        python -m src.extract_toc.core <text_file_path>
        或直接运行使用默认文件: python -m src.extract_toc.core
    """
    import sys
    
    # 默认文件名，便于调试
    default_file = "./specs/钢结构设计标准[附条文说明].txt"
    
    if len(sys.argv) < 2:
        text_file_path = default_file
        print(f"未指定文件，使用默认文件: {text_file_path}")
        print(f"提示: 也可以使用 python -m src.extract_toc.core <text_file_path> 指定文件\n")
    else:
        text_file_path = sys.argv[1]
    
    try:
        with open(text_file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print(f"错误: 文件未找到 '{text_file_path}'")
        return
    except Exception as e:
        print(f"错误: 读取文件失败 - {e}")
        return
    
    # 提取章节行
    content_lines, toc_lines = extract_chapter_lines_from_lines(lines)
    
    # 输出结果
    print(f"\n处理文件: {text_file_path}")
    print(f"总行数: {len(lines)}")
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


if __name__ == "__main__":
    main()
