"""
文档目录提取工具

帮助 LLM 像人类一样高效读取文档的智能目录提取系统。
"""

from .core import extract_chapter_lines_from_lines
from .pdf import extract_chapter_lines as extract_pdf_chapters
from .markdown import extract_chapter_lines_from_md as extract_md_chapters
from .scanner import scan_folder, scan_file

__version__ = "1.1.0"
__all__ = [
    "extract_chapter_lines_from_lines",
    "extract_pdf_chapters",
    "extract_md_chapters",
    "scan_folder",
    "scan_file",
]
