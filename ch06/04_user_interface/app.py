# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

# 导入路径处理模块
from pathlib import Path
# 导入系统模块
import sys

# 导入OpenAI的分词器
import tiktoken
# 导入PyTorch深度学习框架
import torch
# 导入Chainlit用户界面框架
import chainlit

# 从之前的章节导入必要的函数和类
from previous_chapters import (
    classify_review,  # 用于分类评论的函数
    GPTModel         # GPT模型类
)

# 设置设备(GPU如果可用,否则使用CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_model_and_tokenizer():
    """
    加载在第6章中微调的GPT-2模型的代码。
    这需要你先运行第6章的代码,生成必要的model.pth文件。
    """

    # GPT-124M模型的配置参数
    GPT_CONFIG_124M = {
        "vocab_size": 50257,     # Vocabulary size - 词汇表大小
        "context_length": 1024,  # Context length - 上下文长度
        "emb_dim": 768,          # Embedding dimension - 嵌入维度
        "n_heads": 12,           # Number of attention heads - 注意力头数量
        "n_layers": 12,          # Number of layers - 层数
        "drop_rate": 0.1,        # Dropout rate - Dropout比率
        "qkv_bias": True         # Query-key-value bias - 是否使用QKV偏置
    }

    # 初始化GPT-2分词器
    tokenizer = tiktoken.get_encoding("gpt2")

    # 设置模型路径
    model_path = Path("..") / "01_main-chapter-code" / "review_classifier.pth"
    # 检查模型文件是否存在
    if not model_path.exists():
        print(
            f"Could not find the {model_path} file. Please run the chapter 6 code"
            " (ch06.ipynb) to generate the review_classifier.pth file."
        )
        sys.exit()

    # 实例化模型
    model = GPTModel(GPT_CONFIG_124M)

    # 将模型转换为分类器,如ch06.ipynb中的6.5节所示
    num_classes = 2  # 设置类别数量
    model.out_head = torch.nn.Linear(in_features=GPT_CONFIG_124M["emb_dim"], out_features=num_classes)

    # 加载模型权重
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint)
    # 将模型移至指定设备
    model.to(device)
    # 设置为评估模式
    model.eval()

    return tokenizer, model


# 获取chainlit函数所需的分词器和模型文件
tokenizer, model = get_model_and_tokenizer()


@chainlit.on_message
async def main(message: chainlit.Message):
    """
    主要的Chainlit函数。
    """
    # 获取用户输入
    user_input = message.content

    # 使用模型对输入进行分类
    label = classify_review(user_input, model, tokenizer, device, max_length=120)

    # 发送模型响应到界面
    await chainlit.Message(
        content=f"{label}",  # 将模型响应返回到界面
    ).send()
