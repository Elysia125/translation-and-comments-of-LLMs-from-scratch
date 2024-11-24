# 版权声明 - 作者Sebastian Raschka,使用Apache License 2.0许可证
# 来源:"从零开始构建大型语言模型"
# - https://www.manning.com/books/build-a-large-language-model-from-scratch
# 代码: https://github.com/rasbt/LLMs-from-scratch

# 内部使用的单元测试文件

# 从gpt.py导入main函数
from gpt import main

# 定义期望的输出字符串
expected = """
==================================================
                      IN
==================================================

Input text: Hello, I am
Encoded input text: [15496, 11, 314, 716]
encoded_tensor.shape: torch.Size([1, 4])


==================================================
                      OUT
==================================================

Output: tensor([[15496,    11,   314,   716, 27018, 24086, 47843, 30961, 42348,  7267,
         49706, 43231, 47062, 34657]])
Output length: 14
Output text: Hello, I am Featureiman Byeswickattribute argue logger Normandy Compton analogous
"""


# 定义测试main函数的测试用例
def test_main(capsys):
    # 运行main函数
    main()
    # 捕获标准输出
    captured = capsys.readouterr()

    # 标准化行尾并去除每行末尾的空白字符
    normalized_expected = '\n'.join(line.rstrip() for line in expected.splitlines())
    normalized_output = '\n'.join(line.rstrip() for line in captured.out.splitlines())

    # 比较标准化后的字符串是否相等
    assert normalized_output == normalized_expected
