# 版权声明 - 本代码基于 Apache License 2.0 许可证 (见 LICENSE.txt)
# 来源于《从零开始构建大型语言模型》一书
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# 代码仓库: https://github.com/rasbt/LLMs-from-scratch

"""
处理古腾堡计划文件并将其合并为较少的大文件的脚本。
"""

# 导入所需的库
import argparse  # 用于解析命令行参数
import os  # 用于文件和目录操作
import re  # 用于正则表达式操作
from tqdm import tqdm  # 用于显示进度条
from gutenberg.src.cleanup import strip_headers  # 用于清理古腾堡项目文件头部


def is_english(text, threshold=0.9):
    """
    判断文本是否主要为英文
    参数:
        text: 要检查的文本
        threshold: ASCII字符占比阈值，默认0.9
    返回:
        布尔值，表示是否为英文文本
    """
    ascii_chars = sum(1 for c in text if ord(c) < 128)  # 计算ASCII字符数量
    return ascii_chars / len(text) > threshold  # 返回ASCII字符占比是否超过阈值


def combine_files(file_paths, target_dir, max_size_mb=500, separator="<|endoftext|>", fallback_encoding="latin1"):
    """
    合并多个文本文件为较大的文件
    参数:
        file_paths: 输入文件路径列表
        target_dir: 输出目录
        max_size_mb: 每个合并文件的最大大小(MB)
        separator: 文件间的分隔符
        fallback_encoding: 备用编码格式
    """
    # 如果目标目录不存在则创建
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    current_content = []  # 存储当前要合并的内容
    current_size = 0  # 当前内容的大小
    file_counter = 1  # 输出文件计数器

    # 遍历所有输入文件
    for file_path in tqdm(file_paths):
        try:
            # 尝试以UTF-8编码读取文件
            with open(file_path, "r", encoding="utf-8") as file:
                content = file.read()
        except UnicodeDecodeError:
            # UTF-8失败时使用备用编码
            tqdm.write(f"警告: 遇到Unicode解码错误。尝试使用备用编码读取 {file_path}")
            with open(file_path, "r", encoding=fallback_encoding) as file:
                content = file.read()

        # 跳过非英文文本
        if not is_english(content):
            tqdm.write(f"跳过 {file_path} 因为它不是主要由英文构成。")
            continue
        content = strip_headers(content)  # 移除文件头部信息

        # 使用正则表达式将多个空行替换为单个空行
        content = re.sub(r'\n\s*\n', '\n\n', content)
        estimated_size = len(content.encode("utf-8"))  # 估计内容大小

        # 如果当前内容加上新内容超过大小限制，则写入文件
        if current_size + estimated_size > max_size_mb * 1024 * 1024:
            target_file_path = os.path.join(target_dir, f"combined_{file_counter}.txt")
            with open(target_file_path, "w", encoding="utf-8") as target_file:
                target_file.write(separator.join(current_content))
            file_counter += 1
            current_content = [content]
            current_size = estimated_size
        else:
            current_content.append(content)
            current_size += estimated_size

    # 处理剩余内容
    if current_content:
        target_file_path = os.path.join(target_dir, f"combined_{file_counter}.txt")
        with open(target_file_path, "w", encoding="utf-8") as target_file:
            target_file.write(separator.join(current_content))
    return file_counter


if __name__ == "__main__":
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description="预处理并合并用于预训练的文本文件")

    # 添加命令行参数
    parser.add_argument("--data_dir", type=str, default="gutenberg/data/raw",
                        help="包含下载的原始训练数据的目录")
    parser.add_argument("--max_size_mb", type=int, default=500,
                        help="每个合并文件的最大大小(MB)")
    parser.add_argument("--output_dir", type=str, default="gutenberg_preprocessed",
                        help="预处理后的数据保存目录")

    # 解析命令行参数
    args = parser.parse_args()

    # 获取所有txt文件的路径
    all_files = [os.path.join(path, name) for path, subdirs, files in os.walk(args.data_dir)
                 for name in files if name.endswith((".txt", ".txt.utf8"))]

    # 打印处理信息
    print(f"{len(all_files)} 个文件待处理。")
    file_counter = combine_files(all_files, args.output_dir, max_size_mb=args.max_size_mb)
    print(f"{file_counter} 个文件已保存到 {os.path.abspath(args.output_dir)}")
