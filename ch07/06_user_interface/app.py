# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

# 导入所需的Python标准库
from pathlib import Path  # 用于处理文件路径
import sys  # 用于系统相关操作

# 导入第三方库
import tiktoken  # OpenAI的分词器
import torch  # PyTorch深度学习框架
import chainlit  # 用于构建聊天界面的框架

# 从previous_chapters.py导入必要的函数和类
from previous_chapters import (
    generate,  # 文本生成函数
    GPTModel,  # GPT模型类
    text_to_token_ids,  # 文本转token ID的函数
    token_ids_to_text,  # token ID转文本的函数
)

# 设置设备(GPU如果可用,否则使用CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_model_and_tokenizer():
    """
    Code to load a GPT-2 model with finetuned weights generated in chapter 7.
    This requires that you run the code in chapter 7 first, which generates the necessary gpt2-medium355M-sft.pth file.
    """

    # 定义GPT-2 355M模型的配置参数
    GPT_CONFIG_355M = {
        "vocab_size": 50257,     # Vocabulary size
        "context_length": 1024,  # Shortened context length (orig: 1024)
        "emb_dim": 1024,         # Embedding dimension
        "n_heads": 16,           # Number of attention heads
        "n_layers": 24,          # Number of layers
        "drop_rate": 0.0,        # Dropout rate
        "qkv_bias": True         # Query-key-value bias
    }

    # 初始化GPT-2分词器
    tokenizer = tiktoken.get_encoding("gpt2")

    # 构建模型权重文件路径
    model_path = Path("..") / "01_main-chapter-code" / "gpt2-medium355M-sft.pth"
    if not model_path.exists():
        print(
            f"Could not find the {model_path} file. Please run the chapter 7 code "
            " (ch07.ipynb) to generate the gpt2-medium355M-sft.pt file."
        )
        sys.exit()

    # 加载模型权重并初始化模型
    checkpoint = torch.load(model_path, weights_only=True)
    model = GPTModel(GPT_CONFIG_355M)
    model.load_state_dict(checkpoint)
    model.to(device)  # 将模型移至指定设备

    return tokenizer, model, GPT_CONFIG_355M


def extract_response(response_text, input_text):
    # 从完整响应中提取模型的回答部分
    return response_text[len(input_text):].replace("### Response:", "").strip()


# 获取tokenizer和模型实例
tokenizer, model, model_config = get_model_and_tokenizer()


@chainlit.on_message  # chainlit装饰器,用于处理用户消息
async def main(message: chainlit.Message):
    """
    The main Chainlit function.
    """
    
    # 设置随机种子以确保可重复性
    torch.manual_seed(123)

    # 构建提示模板
    prompt = f"""Below is an instruction that describes a task. Write a response
    that appropriately completes the request.

    ### Instruction:
    {message.content}
    """

    # 生成回答
    token_ids = generate(  # function uses `with torch.no_grad()` internally already
        model=model,
        idx=text_to_token_ids(prompt, tokenizer).to(device),  # The user text is provided via as `message.content`
        max_new_tokens=35,
        context_size=model_config["context_length"],
        eos_id=50256
    )

    # 将token ID转换回文本
    text = token_ids_to_text(token_ids, tokenizer)
    response = extract_response(text, prompt)

    # 发送响应到界面
    await chainlit.Message(
        content=f"{response}",  # This returns the model response to the interface
    ).send()
