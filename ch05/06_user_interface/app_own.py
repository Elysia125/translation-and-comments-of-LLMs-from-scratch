# 版权声明 - 本代码基于 Apache License 2.0 许可证 (见 LICENSE.txt)
# 来源于《从零开始构建大型语言模型》一书
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# 代码仓库: https://github.com/rasbt/LLMs-from-scratch

# 导入所需的库
from pathlib import Path  # 用于处理文件路径
import sys  # 用于系统相关操作

import tiktoken  # OpenAI的分词器
import torch  # PyTorch深度学习框架
import chainlit  # 用于构建聊天界面的库

# 从之前章节导入所需的函数和类
from previous_chapters import (
    generate,  # 生成文本的函数
    GPTModel,  # GPT模型类
    text_to_token_ids,  # 将文本转换为token ID的函数
    token_ids_to_text,  # 将token ID转换回文本的函数
)

# 设置设备(GPU如果可用,否则使用CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_model_and_tokenizer():
    """
    加载在第5章中生成的预训练GPT-2模型权重的代码。
    需要先运行第5章的代码来生成必要的model.pth文件。
    """

    # GPT-2小型模型(124M参数)的配置
    GPT_CONFIG_124M = {
        "vocab_size": 50257,    # 词汇表大小
        "context_length": 256,  # 缩短的上下文长度(原始为1024)
        "emb_dim": 768,         # 嵌入维度
        "n_heads": 12,          # 注意力头数量
        "n_layers": 12,         # 层数
        "drop_rate": 0.1,       # Dropout比率
        "qkv_bias": False       # 是否使用Query-Key-Value偏置
    }

    # 初始化GPT-2分词器
    tokenizer = tiktoken.get_encoding("gpt2")

    # 设置模型权重文件路径
    model_path = Path("..") / "01_main-chapter-code" / "model.pth"
    if not model_path.exists():  # 检查模型文件是否存在
        print(f"找不到{model_path}文件。请先运行第5章代码(ch05.ipynb)来生成model.pth文件。")
        sys.exit()

    # 加载模型权重并初始化模型
    checkpoint = torch.load(model_path, weights_only=True)  # 只加载权重
    model = GPTModel(GPT_CONFIG_124M)  # 创建模型实例
    model.load_state_dict(checkpoint)  # 加载权重
    model.to(device)  # 将模型移到指定设备

    return tokenizer, model, GPT_CONFIG_124M


# 获取chainlit函数所需的分词器和模型文件
tokenizer, model, model_config = get_model_and_tokenizer()


@chainlit.on_message  # chainlit装饰器,用于处理接收到的消息
async def main(message: chainlit.Message):
    """
    Chainlit的主函数,处理用户输入并返回模型响应。
    """
    # 生成文本响应
    token_ids = generate(  # generate函数内部已使用torch.no_grad()
        model=model,
        idx=text_to_token_ids(message.content, tokenizer).to(device),  # 将用户输入文本转换为token ID
        max_new_tokens=50,  # 最大生成的新token数量
        context_size=model_config["context_length"],  # 上下文长度
        top_k=1,  # 只选择概率最高的token
        temperature=0.0  # 温度参数为0,使输出更确定性
    )

    # 将生成的token ID转换回文本
    text = token_ids_to_text(token_ids, tokenizer)

    # 发送响应消息到界面
    await chainlit.Message(
        content=f"{text}",  # 返回模型响应到界面
    ).send()
