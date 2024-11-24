# 版权声明 - Sebastian Raschka 基于 Apache License 2.0 (见 LICENSE.txt)
# 来源:"从零开始构建大型语言模型"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# 代码: https://github.com/rasbt/LLMs-from-scratch

# 内部使用文件(单元测试)


# 导入subprocess模块用于执行命令行命令
import subprocess


def test_gpt_class_finetune():
    # 定义命令行参数
    command = ["python", "ch06/01_main-chapter-code/gpt_class_finetune.py", "--test_mode"]

    # 运行命令并捕获输出
    result = subprocess.run(command, capture_output=True, text=True)
    # 检查返回码,确保脚本正常退出
    assert result.returncode == 0, f"脚本运行出错: {result.stderr}"
