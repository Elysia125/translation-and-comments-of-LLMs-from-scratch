# 版权声明 - 本代码基于 Apache License 2.0 许可证 (见 LICENSE.txt)
# 来源于《从零开始构建大型语言模型》一书
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# 代码仓库: https://github.com/rasbt/LLMs-from-scratch

# 用于内部测试的文件

# 导入所需的库
from pathlib import Path  # 用于处理文件路径
import os  # 用于文件系统操作
import subprocess  # 用于执行外部命令


def test_pretraining():
    """测试预训练功能的函数"""

    # 定义一个简单的重复序列
    sequence = "a b c d"  # 基础序列
    repetitions = 1000  # 重复次数
    content = sequence * repetitions  # 生成重复内容

    # 设置保存文件的路径
    folder_path = Path("gutenberg") / "data"  # 定义文件夹路径
    file_name = "repeated_sequence.txt"  # 定义文件名

    # 创建文件夹(如果不存在)
    os.makedirs(folder_path, exist_ok=True)

    # 将重复内容写入文件
    with open(folder_path/file_name, "w") as file:
        file.write(content)

    # 运行预训练脚本并捕获输出
    result = subprocess.run(
        ["python", "pretraining_simple.py", "--debug", "true"],  # 执行命令及参数
        capture_output=True,  # 捕获输出
        text=True  # 以文本形式返回输出
    )
    print(result.stdout)  # 打印输出结果
    assert "Maximum GPU memory allocated" in result.stdout  # 验证输出中包含预期的内存使用信息
