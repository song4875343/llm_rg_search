import sys
from pathlib import Path

# 优先使用 PyMuPDF (快速)，如果失败则使用 pdfplumber (准确)
try:
    import fitz  # PyMuPDF
    USE_PYMUPDF = True
    print("[info] 使用 PyMuPDF (fitz) - 高速模式")
except ImportError:
    USE_PYMUPDF = False
    try:
        import pdfplumber
        print("[info] 使用 pdfplumber - 标准模式")
        print("[提示] 安装 PyMuPDF 可获得更快速度: pip install PyMuPDF")
    except ImportError:
        print("错误: 未安装 PDF 处理库")
        print("请安装其中之一:")
        print("  pip install PyMuPDF  (推荐，速度快)")
        print("  pip install pdfplumber")
        sys.exit(1)

from .core import extract_chapter_lines_from_lines


def extract_pdf_lines_pymupdf(pdf_path):
    """
    使用 PyMuPDF (fitz) 提取 PDF 文本 - 速度快。
    使用 sort=True 以获得"从上到下、从左到右"的阅读顺序。
    """
    lines = []
    
    try:
        doc = fitz.open(pdf_path)

        for page_index in range(len(doc)):
            page = doc[page_index]
            # 使用 sort=True 以获得自然阅读顺序
            page_text = page.get_text("text", sort=True)
            page_lines = page_text.splitlines()
            lines.extend(page_lines)
        
        doc.close()
    except Exception as e:
        print(f"错误: 无法打开或处理 PDF 文件 - {e}")
        return []

    return lines


def extract_pdf_lines_pdfplumber(pdf_path):
    """
    使用 pdfplumber 提取 PDF 文本 - 功能全面。
    """
    lines = []
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text() or ""
                lines.extend(page_text.splitlines())
    except Exception as e:
        print(f"错误: 无法打开或处理 PDF 文件 - {e}")
        return []

    return lines


def extract_pdf_lines(pdf_path):
    """
    提取 PDF 全部文本，并按行展开为列表。
    自动选择最快的可用库。

    参数:
        pdf_path: PDF 文件路径

    返回:
        list[str]: PDF 中的所有文本行
    """
    if USE_PYMUPDF:
        return extract_pdf_lines_pymupdf(pdf_path)
    else:
        return extract_pdf_lines_pdfplumber(pdf_path)


def extract_chapter_lines(pdf_path):
    """
    从 PDF 中提取章节相关的文本行。

    参数:
        pdf_path: PDF 文件路径

    返回:
        tuple: (content_lines, toc_lines)
        - content_lines: list of (line_text, line_number) tuples - 最后一个字符不是数字的章节行
        - toc_lines: list of (line_text, line_number) tuples - 最后通常带页码数字的目录行
    """
    lines = extract_pdf_lines(pdf_path)
    content_lines, toc_lines = extract_chapter_lines_from_lines(lines)
    return content_lines, toc_lines


def main():
    """
    测试入口：遍历示例目录下的所有 PDF，并打印提取结果。
    """
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    # 默认目录
    pdf_dir = Path(r"./specs")
    
    # 支持命令行参数
    if len(sys.argv) > 1:
        pdf_dir = Path(sys.argv[1])
    
    pdf_files = sorted(pdf_dir.glob("*.pdf"))

    if not pdf_files:
        print(f"未找到 PDF 文件: {pdf_dir}")
        print("提示: 可以使用 python -m src.extract_toc.pdf <目录路径> 指定目录")
        return

    # 处理前3个文件
    for pdf_file in pdf_files[:3]:
        print(f"\n{'=' * 80}")
        print(f"文件: {pdf_file}")
        try:
            content_lines, toc_lines = extract_chapter_lines(str(pdf_file))
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


if __name__ == "__main__":
    main()
