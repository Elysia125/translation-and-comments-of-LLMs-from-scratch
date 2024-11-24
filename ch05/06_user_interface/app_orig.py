# 版权声明 - 本代码基于 Apache License 2.0 许可证 (见 LICENSE.txt)
# 来源于《从零开始构建大型语言模型》一书
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# 代码仓库: https://github.com/rasbt/LLMs-from-scratch

# 导入所需的库
import tiktoken  # OpenAI的分词器
import torch  # PyTorch深度学习框架
import chainlit  # 用于构建聊天界面的库

# 从之前章节导入所需的函数和类
from previous_chapters import (
    download_and_load_gpt2,  # 下载和加载GPT-2模型的函数
    generate,  # 生成文本的函数
    GPTModel,  # GPT模型类
    load_weights_into_gpt,  # 加载预训练权重的函数
    text_to_token_ids,  # 将文本转换为token ID的函数
    token_ids_to_text,  # 将token ID转换回文本的函数
)

# 设置设备(GPU如果可用,否则使用CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_model_and_tokenizer():
    """
    加载带有OpenAI预训练权重的GPT-2模型的代码。
    代码与第5章类似。
    如果当前文件夹中还不存在模型,将自动下载。
    """

    # 选择要使用的模型大小
    CHOOSE_MODEL = "gpt2-small (124M)"  # 可以选择替换为下面model_configs中的其他模型

    # 基础配置参数
    BASE_CONFIG = {
        "vocab_size": 50257,     # 词汇表大小
        "context_length": 1024,  # 上下文长度
        "drop_rate": 0.0,        # Dropout比率
        "qkv_bias": True         # 是否使用Query-Key-Value偏置
    }

    # 不同规模GPT-2模型的配置参数
    model_configs = {
        "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},  # 小型模型配置
        "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},  # 中型模型配置
        "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},  # 大型模型配置
        "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},  # 超大型模型配置
    }

    # 从选择的模型名称中提取模型大小信息
    model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")

    # 更新基础配置
    BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

    # 下载并加载模型参数
    settings, params = download_and_load_gpt2(model_size=model_size, models_dir="gpt2")

    # 初始化GPT模型
    gpt = GPTModel(BASE_CONFIG)
    load_weights_into_gpt(gpt, params)  # 加载预训练权重
    gpt.to(device)  # 将模型移到指定设备
    gpt.eval()  # 设置为评估模式

    # 初始化分词器
    tokenizer = tiktoken.get_encoding("gpt2")

    return tokenizer, gpt, BASE_CONFIG


# 获取chainlit函数所需的分词器和模型文件
tokenizer, model, model_config = get_model_and_tokenizer()


@chainlit.on_message  # chainlit装饰器,用于处理接收到的消息
async def main(message: chainlit.Message):
    """
    主要的Chainlit函数。
    """
    # 生成文本
    token_ids = generate(  # generate函数内部已使用torch.no_grad()
        model=model,
        idx=text_to_token_ids(message.content, tokenizer).to(device),  # 将用户输入文本转换为token ID
        max_new_tokens=50,  # 最大生成的新token数
        context_size=model_config["context_length"],  # 上下文长度
        top_k=1,  # 只选择概率最高的token
        temperature=0.0  # 温度参数为0,使输出更确定性
    )

    # 将生成的token ID转换回文本
    text = token_ids_to_text(token_ids, tokenizer)

    # 发送响应消息到界面
    await chainlit.Message(
        content=f"{text}",  # 将模型响应返回到界面
    ).send()
