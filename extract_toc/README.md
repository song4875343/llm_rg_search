# 文档目录提取工具

帮助 LLM 像人类一样高效读取文档的智能目录提取系统。

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)

## 📚 文档导航

- [快速开始](QUICKSTART.md) - 5 分钟上手指南
- [安装指南](INSTALL.md) - 详细安装说明
- [设计文档](DESIGN.md) - 架构和设计思路
- [构建指南](BUILD.md) - 打包和发布流程
- [更新日志](CHANGELOG.md) - 版本历史

## 项目简介

在处理大量文档资料时，LLM 往往需要读取整个文档才能了解其内容，这既耗时又消耗大量 token。本项目通过自动提取文档的章节目录，让 LLM 能够：

1. **快速浏览** - 仅通过文件名和目录就能判断文档是否包含所需信息
2. **精准定位** - 根据章节标题快速找到相关内容
3. **按需读取** - 只读取必要的章节，大幅节省时间和成本

就像人类阅读书籍时先看目录一样，LLM 也可以通过目录快速了解文档结构，按需深入阅读。

## 核心功能

- **多格式支持**: PDF、Markdown、纯文本文件
- **智能识别**: 自动识别章节标题和目录项
- **批量扫描**: 扫描整个文件夹，生成 JSON 索引
- **层级结构**: 自动构建章节树（章、节、小节）
- **高性能**: 支持 PyMuPDF 高速提取（比 pdfplumber 快 10-50 倍）
- **高准确度**: 复杂的过滤规则，准确区分章节标题和普通文本
- **中英文支持**: 完美支持中英文混合文档

## 使用场景

### 场景 1: 批量建立文档索引（推荐）
扫描整个资料文件夹,生成 JSON 索引文件：

```python
from src.extract_toc.scanner import scan_folder

# 扫描文件夹并生成索引
index = scan_folder(
    folder_path="./documents",
    recursive=True,  # 递归扫描子文件夹
    output_file="document_index.json"
)

# LLM 使用索引
import json
with open("document_index.json", "r", encoding="utf-8") as f:
    index = json.load(f)

# 查找相关文档
for folder_name, folder_info in index['folders'].items():
    for file_name, file_info in folder_info['files'].items():
        print(f"{file_name}: {file_info['total_chapters']} 章节")
```

### 场景 2: 单文件处理
处理单个文档：

```python
from src.extract_toc.pdf import extract_chapter_lines

# 提取章节
content_lines, toc_lines = extract_chapter_lines("document.pdf")
```

### 场景 3: LLM 智能检索
LLM 需要查找资料时：

1. 先查看文档目录，判断是否相关
2. 如果相关，再根据章节标题定位具体内容
3. 只读取必要的章节，而非整个文档

### 场景 3: 文档分析
快速了解大型文档的结构和内容组织。

## 项目结构

```
extract_toc/
├── src/                          # 源代码
│   └── extract_toc/              # 主包
│       ├── __init__.py           # 包初始化
│       ├── core.py               # 核心过滤逻辑
│       ├── pdf.py                # PDF 处理（支持 PyMuPDF/pdfplumber）
│       ├── markdown.py           # Markdown 处理
│       └── scanner.py            # 文档扫描和索引生成
├── test/                         # 测试文件
│   ├── debug_noise.py            # 噪声规则调试
│   ├── debug_title.py            # 标题过滤调试
│   ├── test_all_titles.py        # 标题测试集
│   └── specs/                    # 测试文档
├── example.py                    # 基本使用示例
├── example_scan.py               # 扫描功能示例
└── README.md                     # 项目说明
```

## 快速开始

### 安装

```bash
# 从源码安装（开发模式）
pip install -e .

# 或安装所有功能
pip install -e ".[all]"

# 从 PyPI 安装（发布后）
pip install extract-toc[pdf]
```

详细安装说明请查看 [INSTALL.md](INSTALL.md)

### 安装依赖

```bash
# PDF 处理（推荐 PyMuPDF，速度快）
pip install PyMuPDF

# 或使用 pdfplumber（功能全面）
pip install pdfplumber
```

### 基本使用

#### 1. 批量扫描文档（推荐）

```python
from src.extract_toc.scanner import scan_folder

# 扫描文件夹并生成索引
index = scan_folder(
    folder_path="./documents",
    recursive=True,  # 递归扫描子文件夹
    output_file="document_index.json"
)

# 查看统计信息
print(f"扫描了 {index['summary']['total_files']} 个文件")
print(f"找到 {index['summary']['total_chapters']} 个章节")
```

命令行使用：
```bash
# 扫描文件夹
python -m src.extract_toc.scanner ./documents index.json

# 不递归子文件夹
python -m src.extract_toc.scanner ./documents index.json --no-recursive
```

#### 2. 处理单个 PDF 文档

```python
from src.extract_toc.pdf import extract_chapter_lines

# 提取章节
content_lines, toc_lines = extract_chapter_lines("document.pdf")

# content_lines: 正文章节标题 [(标题, 行号), ...]
# toc_lines: 目录项 [(标题, 行号), ...]

print("正文章节:")
for title, line_num in content_lines:
    print(f"[行 {line_num:4d}] {title}")

print("\n目录:")
for title, line_num in toc_lines:
    print(f"[行 {line_num:4d}] {title}")
```

#### 3. 处理 Markdown 文档

```python
from src.extract_toc.markdown import extract_chapter_lines_from_md

# 提取章节（自动识别 # 标题和 **加粗** 文本）
content_lines, toc_lines = extract_chapter_lines_from_md("document.md")
```

#### 4. 处理纯文本

```python
from src.extract_toc.core import extract_chapter_lines_from_lines

# 读取文本文件
with open("document.txt", "r", encoding="utf-8") as f:
    lines = f.readlines()

# 提取章节
content_lines, toc_lines = extract_chapter_lines_from_lines(lines)
```

### 命令行使用

```bash
# 处理 PDF（自动使用最快的可用库）
python src/extract_toc/pdf.py <目录路径>

# 处理 Markdown
python src/extract_toc/markdown.py <目录路径>

# 处理纯文本
python src/extract_toc/core.py <文本文件路径>
```

## 核心特性

### 智能章节识别

支持多种章节编号格式：

- **阿拉伯数字**: `1.`, `1.1`, `1.1.1`
- **中文编号**: `第一章`, `第二节`, `一、`, `（一）`
- **混合格式**: `1 Overview`, `2.1 引言`
- **中英对照**: `2.1.4 直接分析设计法 direct analysis method of design`

### 智能层级控制

- 自动识别一级、二级、三级标题
- 只保留一级和二级标题，过滤三级及以下
- 自动构建章节树结构
- 二级标题正确归属到对应的一级章节下

### 高精度过滤系统

通过多重策略确保章节识别准确率达到95%+：

1. **基础格式检查** - 识别常见章节编号格式
2. **噪声内容过滤** - 过滤列表项、表格数据、公式等
3. **位置合理性检查** - 前几章应该在文档前部
4. **序号连续性检查** - 章节号应该连续且不重复
5. **条文解释识别** - 自动识别并归类附录内容

### 条文解释自动处理

针对技术规范类文档的特殊处理：
- 自动检测条文解释部分的开始
- 创建虚拟"条文解释"章节
- 保持文档结构清晰完整

### 目录与正文区分

自动区分目录行和正文章节：

- **目录行**: 通常带页码，如 `1 Overview ......... 1`
- **正文章节**: 实际章节标题，如 `1 Overview`

### 性能优化

- **PyMuPDF 模式**: 使用 `sort=True` 参数，速度快且准确
- **自动降级**: PyMuPDF 不可用时自动使用 pdfplumber
- **无时间统计开销**: 移除了性能统计代码，专注于提取

## API 文档

### core.py

核心过滤逻辑模块。

#### `extract_chapter_lines_from_lines(lines)`

从文本行列表中提取章节。

**参数:**
- `lines` (list[str]): 文本行列表

**返回:**
- `tuple`: `(content_lines, toc_lines)`
  - `content_lines`: `[(标题, 行号), ...]` - 正文章节
  - `toc_lines`: `[(标题, 行号), ...]` - 目录项

### pdf.py

PDF 文档处理模块。

#### `extract_chapter_lines(pdf_path)`

从 PDF 提取章节。

**参数:**
- `pdf_path` (str): PDF 文件路径

**返回:**
- `tuple`: `(content_lines, toc_lines)`

### markdown.py

Markdown 文档处理模块。

#### `extract_chapter_lines_from_md(md_path)`

从 Markdown 提取章节。

**参数:**
- `md_path` (str): Markdown 文件路径

**返回:**
- `tuple`: `(content_lines, toc_lines)`

## 测试

项目包含完整的测试工具：

```bash
# 测试特定标题
python test/debug_title.py

# 测试噪声规则
python test/debug_noise.py

# 运行所有标题测试
python test/test_all_titles.py
```

## 技术细节

### 文本归一化

- Unicode NFKC 标准化
- 统一空格和特殊字符
- 全角/半角转换

### 章节编号识别

使用正则表达式识别多种编号格式：

```python
# 阿拉伯数字编号
r"^(\d+)(?:\.\d+)*(?:\s*[-.)）]\s*|\s+)"

# 中文编号
rf"^(?:第[{_CHINESE_NUMERALS}\d]+[章节部分篇卷条]|..."
```

### 噪声过滤

多层过滤规则：

1. 基本检查（空行、纯数字）
2. 编号检查（是否以章节编号开头）
3. 内容检查（是否有实际标题内容）
4. 标点检查（是否包含不允许的标点）
5. 噪声检查（是否像公式、代码等）

## 常见问题

### Q: PyMuPDF 和 pdfplumber 如何选择？

A: 推荐使用 PyMuPDF：
- 速度快 10-50 倍
- 使用 `sort=True` 参数后准确度与 pdfplumber 相当
- 项目会自动选择可用的库

### Q: 为什么有些标题没有被识别？

A: 可能原因：
1. 标题格式不符合常见章节编号规范
2. 被噪声过滤规则误判
3. 使用 `test/debug_title.py` 调试具体原因

### Q: 如何处理自定义格式的标题？

A: 修改 `src/extract_toc/core.py` 中的正则表达式：
- `_ARABIC_SECTION_VALUE_PATTERN`: 阿拉伯数字格式
- `_CHINESE_PREFIX_PATTERN`: 中文编号格式

## 示例文件说明

项目包含两个示例文件，用于演示不同的使用场景：

### example.py - 基础功能示例
- 演示单个文档的处理
- 适合学习API用法
- 结果打印到控制台

### example_scan.py - 扫描模块示例
- 批量扫描整个文件夹
- 生成分层索引系统
- 演示LLM高效检索工作流
- 适合生产环境

## 高级特性

### 智能层级过滤

系统自动过滤三级及以下标题，只保留一级和二级章节：
- ✅ 保留：`1 总则`（一级）、`1.1 术语`（二级）
- ❌ 过滤：`1.1.1 定义`（三级）

### 噪声条目过滤

通过多重策略自动过滤误识别的内容：

#### 1. 列表项过滤
- `2.未风化-微风化...`（数字.后直接跟中文）
- `1)` `2)`（数字加右括号）
- `一、等效均布地面荷载`（中文数字+顿号）

#### 2. 表格数据过滤
- `6.75mm` `14.17mm`（数字+单位）
- `2.1(有相邻建筑影响)`（带括号的注释）

#### 3. 行号位置过滤 ⭐（这条对于规范适用对于文章类容易误判，删掉了发现不是必须的）
前几章应该在文档前部，超过阈值则判定为误识别：
- 第一章不超过 150 行
- 第二章不超过 200 行
- 第三章不超过 400 行

#### 4. 序号连续性检查 ⭐⭐
利用正常章节的两条原则：
1. 章节号连续（1→2→3...）
2. 章节号不重复

如果一级章节号不在连续序列中，或重复出现，则判定为误识别。

### 条文解释自动识别

对于技术规范类文档，自动识别"条文解释"部分：

**检测条件：**
1. 章节号突然从大跳到小（如从10跳回2）
2. 之后没有新的一级标题，全是二级标题
3. 这些二级标题与前面章节重复但基本符合顺序

**处理方式：**
- 自动创建"条文解释"虚拟章节
- 将后续所有二级标题归入其中
- 保持原有章节结构清晰

**示例：**
```
1. 1 总则
2. 2 术语和符号
   └─ 2.1 术语
   └─ 2.2 符号
...
10. 10 检验与监测
    └─ 10.1 一般规定
    └─ 10.2 检验
    └─ 10.3 监测
11. 条文解释
    └─ 2.1 术语
    └─ 4.1 岩土的分类
    └─ 5.1 基础埋置深度
    └─ ...（34个子节）
```

### 过滤效果

以《建筑地基基础设计规范》为例：
- 原始识别：136个章节
- 过滤后：126个有效章节
- 过滤率：约7.4%
- 结构：10个正文章节 + 1个条文解释章节
- 无噪声、无重复、层级清晰
目前对于规范大部分效果还是非常好的。实在不能形成目录的可以使用关键词，或者用用低成本大模型进行
快速搜索查看。



## 贡献

欢迎提交 Issue 和 Pull Request！

## 许可证

MIT License

## 更新日志

### v1.1.0 (2026-04-13)
- ✨ 新增：智能层级过滤，只保留一级和二级标题
- ✨ 新增：条文解释自动识别和归类
- 🔧 优化：增强噪声过滤规则
  - 列表项过滤（数字.中文、数字)、中文数字+顿号）
  - 表格数据过滤（数字+单位、带括号注释）
- 🔧 优化：行号位置过滤（前几章位置检查）
- 🔧 优化：序号连续性检查（章节号连续性和重复检测）
- 📈 效果：过滤准确率提升至95%+

### v1.0.0 (2026-04-03)
- 初始版本发布
- 支持 PDF、Markdown、纯文本
- 智能章节识别和过滤
- PyMuPDF 高速模式
- 完整的测试工具集


