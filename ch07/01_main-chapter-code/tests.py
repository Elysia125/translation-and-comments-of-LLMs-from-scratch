# 版权声明 - 作者Sebastian Raschka，使用Apache License 2.0许可
# 来源于"从零开始构建大型语言模型"一书
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# 代码仓库: https://github.com/rasbt/LLMs-from-scratch

# 用于内部使用的文件(单元测试)


# 导入subprocess模块用于执行外部命令
import subprocess


def test_gpt_class_finetune():
    # 定义要执行的命令,包含Python解释器路径、脚本路径和测试模式参数
    command = ["python", "ch06/01_main-chapter-code/gpt_class_finetune.py", "--test_mode"]

    # 执行命令并捕获输出
    result = subprocess.run(command, capture_output=True, text=True)
    # 检查返回码是否为0(表示成功执行),如果不是则抛出异常并显示错误信息
    assert result.returncode == 0, f"Script exited with errors: {result.stderr}"
