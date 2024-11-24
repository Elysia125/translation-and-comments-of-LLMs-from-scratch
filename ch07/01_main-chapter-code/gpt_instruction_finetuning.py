# 版权声明 - 作者Sebastian Raschka，使用Apache License 2.0许可
# 来源于"从零开始构建大型语言模型"一书
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# 代码仓库: https://github.com/rasbt/LLMs-from-scratch
#
# 基于第7章代码的最小指令微调文件

# 导入functools中的partial函数用于函数参数的部分应用
from functools import partial
# 导入importlib.metadata中的version函数用于获取包版本信息
from importlib.metadata import version
# 导入json模块用于处理JSON数据
import json
# 导入os模块用于操作系统相关功能
import os
# 导入re模块用于正则表达式操作
import re
# 导入time模块用于时间相关操作
import time
# 导入urllib模块用于URL操作
import urllib

# 导入matplotlib.pyplot用于绘图
import matplotlib.pyplot as plt
# 导入tiktoken用于tokenization
import tiktoken
# 导入PyTorch
import torch
# 从torch.utils.data导入Dataset和DataLoader类
from torch.utils.data import Dataset, DataLoader
# 导入tqdm用于进度条显示
from tqdm import tqdm

# 从本地文件导入相关函数
from gpt_download import download_and_load_gpt2
from previous_chapters import (
    calc_loss_loader,
    generate,
    GPTModel,
    load_weights_into_gpt,
    text_to_token_ids,
    train_model_simple,
    token_ids_to_text
)


# 定义指令数据集类
class InstructionDataset(Dataset):
    def __init__(self, data, tokenizer):
        # 存储数据
        self.data = data

        # 预先对文本进行tokenization
        self.encoded_texts = []
        for entry in data:
            # 格式化输入文本
            instruction_plus_input = format_input(entry)
            # 添加响应文本
            response_text = f"\n\n### Response:\n{entry['output']}"
            # 合并完整文本
            full_text = instruction_plus_input + response_text
            # 对文本进行编码并存储
            self.encoded_texts.append(
                tokenizer.encode(full_text)
            )

    # 实现获取数据项的方法
    def __getitem__(self, index):
        return self.encoded_texts[index]

    # 实现获取数据集长度的方法
    def __len__(self):
        return len(self.data)


# 定义自定义的数据批处理函数
def custom_collate_fn(
    batch,
    pad_token_id=50256,  # 填充token的ID
    ignore_index=-100,   # 忽略的索引值
    allowed_max_length=None,  # 允许的最大长度
    device="cpu"         # 设备类型
):
    # 找出批次中最长序列的长度
    batch_max_length = max(len(item)+1 for item in batch)

    # 初始化输入和目标列表
    inputs_lst, targets_lst = [], []

    for item in batch:
        # 复制item以避免修改原始数据
        new_item = item.copy()
        # 添加结束token
        new_item += [pad_token_id]
        # 使用填充token将序列填充到最大长度
        padded = new_item + [pad_token_id] * (batch_max_length - len(new_item))
        # 创建输入tensor(去掉最后一个token)
        inputs = torch.tensor(padded[:-1])
        # 创建目标tensor(向右移动一位)
        targets = torch.tensor(padded[1:])

        # 在目标中将除第一个以外的所有填充token替换为ignore_index
        mask = targets == pad_token_id
        indices = torch.nonzero(mask).squeeze()
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index

        # 如果指定了最大长度，则截断序列
        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]

        # 将处理后的序列添加到列表中
        inputs_lst.append(inputs)
        targets_lst.append(targets)

    # 将输入和目标列表转换为tensor并移至目标设备
    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)

    return inputs_tensor, targets_tensor


# 定义下载和加载文件的函数
def download_and_load_file(file_path, url):
    # 如果文件不存在则下载
    if not os.path.exists(file_path):
        # 打开URL并读取数据
        with urllib.request.urlopen(url) as response:
            text_data = response.read().decode("utf-8")
        # 将数据写入文件
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(text_data)
    else:
        # 如果文件存在则直接读取
        with open(file_path, "r", encoding="utf-8") as file:
            text_data = file.read()

    # 加载JSON数据
    with open(file_path, "r") as file:
        data = json.load(file)

    return data


# 定义格式化输入的函数
def format_input(entry):
    # 创建指令文本
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )

    # 如果有输入文本则添加，否则为空
    input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""

    # 返回完整的格式化文本
    return instruction_text + input_text


# 定义绘制损失曲线的函数
def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    # 创建图形和主坐标轴
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # 绘制训练和验证损失
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")

    # 创建第二个x轴用于显示处理的token数量
    ax2 = ax1.twiny()
    ax2.plot(tokens_seen, train_losses, alpha=0)
    ax2.set_xlabel("Tokens seen")

    # 调整布局并保存图形
    fig.tight_layout()
    plot_name = "loss-plot-standalone.pdf"
    print(f"Plot saved as {plot_name}")
    plt.savefig(plot_name)


# 主函数
def main(test_mode=False):
    #######################################
    # 打印包版本信息
    #######################################
    print()
    pkgs = [
        "matplotlib",  # 绘图库
        "tiktoken",    # 分词器
        "torch",       # 深度学习库
        "tqdm",        # 进度条
        "tensorflow",  # 用于OpenAI预训练权重
    ]
    for p in pkgs:
        print(f"{p} version: {version(p)}")
    print(50*"-")

    #######################################
    # 下载并准备数据集
    #######################################
    # 设置文件路径和URL
    file_path = "instruction-data.json"
    url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch07/01_main-chapter-code/instruction-data.json"
    # 下载并加载数据
    data = download_and_load_file(file_path, url)

    # 划分数据集
    train_portion = int(len(data) * 0.85)  # 85%用于训练
    test_portion = int(len(data) * 0.1)    # 10%用于测试

    # 分割数据集
    train_data = data[:train_portion]
    test_data = data[train_portion:train_portion + test_portion]
    val_data = data[train_portion + test_portion:]

    # 测试模式下使用较小的数据集
    if args.test_mode:
        train_data = train_data[:10]
        val_data = val_data[:10]
        test_data = test_data[:10]

    # 打印数据集大小信息
    print("Training set length:", len(train_data))
    print("Validation set length:", len(val_data))
    print("Test set length:", len(test_data))
    print(50*"-")

    # 初始化tokenizer和设备
    tokenizer = tiktoken.get_encoding("gpt2")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print(50*"-")

    # 创建自定义的collate函数
    customized_collate_fn = partial(custom_collate_fn, device=device, allowed_max_length=1024)

    # 设置数据加载器参数
    num_workers = 0
    batch_size = 8

    # 设置随机种子
    torch.manual_seed(123)

    # 创建训练数据加载器
    train_dataset = InstructionDataset(train_data, tokenizer)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=customized_collate_fn,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers
    )

    # 创建验证数据加载器
    val_dataset = InstructionDataset(val_data, tokenizer)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        collate_fn=customized_collate_fn,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers
    )

    #######################################
    # 加载预训练模型
    #######################################

    # 测试模式下使用小型GPT模型
    if args.test_mode:
        BASE_CONFIG = {
            "vocab_size": 50257,
            "context_length": 120,
            "drop_rate": 0.0,
            "qkv_bias": False,
            "emb_dim": 12,
            "n_layers": 1,
            "n_heads": 2
        }
        model = GPTModel(BASE_CONFIG)
        model.eval()
        device = "cpu"
        CHOOSE_MODEL = "Small test model"

    # 主章节中使用的代码
    else:
        # 基础配置
        BASE_CONFIG = {
            "vocab_size": 50257,     # 词汇表大小
            "context_length": 1024,  # 上下文长度
            "drop_rate": 0.0,        # dropout率
            "qkv_bias": True         # 查询-键-值偏置
        }

        # 不同模型配置
        model_configs = {
            "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
            "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
            "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
            "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
        }

        # 选择模型
        CHOOSE_MODEL = "gpt2-medium (355M)"

        # 更新配置
        BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

        # 获取模型大小并下载模型
        model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")
        settings, params = download_and_load_gpt2(model_size=model_size, models_dir="gpt2")

        # 创建模型并加载权重
        model = GPTModel(BASE_CONFIG)
        load_weights_into_gpt(model, params)
        model.eval()
        model.to(device)

    print("Loaded model:", CHOOSE_MODEL)
    print(50*"-")

    #######################################
    # 微调模型
    #######################################
    # 计算初始损失
    print("Initial losses")
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=5)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=5)

    print("   Training loss:", train_loss)
    print("   Validation loss:", val_loss)

    # 开始训练
    start_time = time.time()
    # 初始化优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.00005, weight_decay=0.1)

    # 设置训练轮数
    num_epochs = 2

    # 设置随机种子并开始训练
    torch.manual_seed(123)
    train_losses, val_losses, tokens_seen = train_model_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=num_epochs, eval_freq=5, eval_iter=5,
        start_context=format_input(val_data[0]), tokenizer=tokenizer
    )

    # 计算训练时间
    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60
    print(f"Training completed in {execution_time_minutes:.2f} minutes.")

    # 绘制损失曲线
    epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
    plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)
    print(50*"-")

    #######################################
    # 保存结果
    #######################################
    # 生成响应
    print("Generating responses")
    for i, entry in tqdm(enumerate(test_data), total=len(test_data)):
        # 格式化输入文本
        input_text = format_input(entry)

        # 生成响应
        token_ids = generate(
            model=model,
            idx=text_to_token_ids(input_text, tokenizer).to(device),
            max_new_tokens=256,
            context_size=BASE_CONFIG["context_length"],
            eos_id=50256
        )
        # 将token转换为文本
        generated_text = token_ids_to_text(token_ids, tokenizer)
        response_text = generated_text[len(input_text):].replace("### Response:", "").strip()

        # 保存响应
        test_data[i]["model_response"] = response_text

    # 保存测试数据和响应
    test_data_path = "instruction-data-with-response-standalone.json"
    with open(test_data_path, "w") as file:
        json.dump(test_data, file, indent=4)
    print(f"Responses saved as {test_data_path}")

    # 保存模型
    file_name = f"{re.sub(r'[ ()]', '', CHOOSE_MODEL) }-sft-standalone.pth"
    torch.save(model.state_dict(), file_name)
    print(f"Model saved as {file_name}")


# 主程序入口
if __name__ == "__main__":
    # 导入参数解析器
    import argparse

    # 创建参数解析器
    parser = argparse.ArgumentParser(
        description="Finetune a GPT model for classification"
    )
    # 添加测试模式参数
    parser.add_argument(
        "--test_mode",
        default=False,
        action="store_true",
        help=("This flag runs the model in test mode for internal testing purposes. "
              "Otherwise, it runs the model as it is used in the chapter (recommended).")
    )
    # 解析参数
    args = parser.parse_args()

    # 运行主函数
    main(args.test_mode)
