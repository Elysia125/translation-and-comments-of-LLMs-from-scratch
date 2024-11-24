# 版权声明 - 作者Sebastian Raschka，使用Apache License 2.0许可
# 来源于"从零开始构建大型语言模型"一书
# 书籍链接: https://www.manning.com/books/build-a-large-language-model-from-scratch
# 代码仓库: https://github.com/rasbt/LLMs-from-scratch
#
# 运行练习的代码;更多信息请参见exercise-solutions.ipynb

# 导入functools中的partial函数用于函数参数的部分应用
from functools import partial
# 导入importlib.metadata中的version函数用于获取包版本信息
from importlib.metadata import version
# 导入json模块用于处理JSON数据
import json
# 导入math模块用于数学计算
import math
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
# 导入MaxNLocator用于设置坐标轴刻度
from matplotlib.ticker import MaxNLocator
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


# 定义带有掩码的指令数据集类
class InstructionDatasetWithMasking(Dataset):
    def __init__(self, data, tokenizer):
        # 存储数据
        self.data = data

        # 新增：用于存储指令长度的列表
        self.instruction_lengths = []
        # 存储编码后的文本
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

            # 新增：计算并存储指令长度
            instruction_length = len(tokenizer.encode(instruction_plus_input))
            self.instruction_lengths.append(instruction_length)

    # 实现获取数据项的方法，返回指令长度和编码文本
    def __getitem__(self, index):
        return self.instruction_lengths[index], self.encoded_texts[index]

    # 实现获取数据集长度的方法
    def __len__(self):
        return len(self.data)


# 定义Phi模型的指令数据集类
class InstructionDatasetPhi(Dataset):
    def __init__(self, data, tokenizer):
        # 存储数据
        self.data = data

        # 预先对文本进行tokenization
        self.encoded_texts = []
        for entry in data:

            # 使用phi格式的输入模板和调整响应文本模板
            instruction_plus_input = format_input_phi(entry)
            response_text = f"\n<|assistant|>:\n{entry['output']}"
            
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


# 定义带有LoRA的线性层类
class LinearWithLoRA(torch.nn.Module):
    def __init__(self, linear, rank, alpha):
        super().__init__()
        # 存储原始线性层
        self.linear = linear
        # 创建LoRA层
        self.lora = LoRALayer(
            linear.in_features, linear.out_features, rank, alpha
        )

    # 前向传播，结合原始线性层和LoRA层的输出
    def forward(self, x):
        return self.linear(x) + self.lora(x)


# 定义LoRA层类
class LoRALayer(torch.nn.Module):
    def __init__(self, in_dim, out_dim, rank, alpha):
        super().__init__()
        # 创建并初始化A矩阵
        self.A = torch.nn.Parameter(torch.empty(in_dim, rank))
        torch.nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        # 创建并初始化B矩阵
        self.B = torch.nn.Parameter(torch.zeros(rank, out_dim))
        # 存储alpha缩放因子
        self.alpha = alpha

    # 前向传播，计算LoRA的输出
    def forward(self, x):
        x = self.alpha * (x @ self.A @ self.B)
        return x


# 替换模型中的线性层为LoRA层的函数
def replace_linear_with_lora(model, rank, alpha):
    for name, module in model.named_children():
        if isinstance(module, torch.nn.Linear):
            # 将线性层替换为带有LoRA的线性层
            setattr(model, name, LinearWithLoRA(module, rank, alpha))
        else:
            # 递归处理子模块
            replace_linear_with_lora(module, rank, alpha)


# 自定义数据批处理函数
def custom_collate_fn(
    batch,
    pad_token_id=50256,
    ignore_index=-100,
    allowed_max_length=None,
    device="cpu"
):
    # 找出批次中最长序列的长度
    batch_max_length = max(len(item)+1 for item in batch)

    # 准备输入和目标列表
    inputs_lst, targets_lst = [], []

    for item in batch:
        new_item = item.copy()
        # 添加结束标记
        new_item += [pad_token_id]
        # 填充序列至最大长度
        padded = new_item + [pad_token_id] * (batch_max_length - len(new_item))
        # 准备输入序列（去掉最后一个token）
        inputs = torch.tensor(padded[:-1])
        # 准备目标序列（向右移动一位）
        targets = torch.tensor(padded[1:])

        # 将目标中除第一个外的所有填充token替换为ignore_index
        mask = targets == pad_token_id
        indices = torch.nonzero(mask).squeeze()
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index

        # 可选：截断到最大允许长度
        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]

        inputs_lst.append(inputs)
        targets_lst.append(targets)

    # 将输入和目标列表转换为张量并移至目标设备
    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)

    return inputs_tensor, targets_tensor


# 带有掩码的自定义数据批处理函数
def custom_collate_with_masking_fn(
    batch,
    pad_token_id=50256,
    ignore_index=-100,
    allowed_max_length=None,
    device="cpu"
):
    # 找出批次中最长序列的长度
    batch_max_length = max(len(item)+1 for instruction_length, item in batch)

    # 准备输入和目标列表
    inputs_lst, targets_lst = [], []

    for instruction_length, item in batch:
        new_item = item.copy()
        # 添加结束标记
        new_item += [pad_token_id]
        # 填充序列至最大长度
        padded = new_item + [pad_token_id] * (batch_max_length - len(new_item))
        # 准备输入序列
        inputs = torch.tensor(padded[:-1])
        # 准备目标序列
        targets = torch.tensor(padded[1:])

        # 将目标中除第一个外的所有填充token替换为ignore_index
        mask = targets == pad_token_id
        indices = torch.nonzero(mask).squeeze()
        if indices.numel() > 1:
            targets[indices[1:]] = ignore_index

        # 在目标中掩码所有输入和指令token
        targets[:instruction_length-1] = -100

        # 可选：截断到最大允许长度
        if allowed_max_length is not None:
            inputs = inputs[:allowed_max_length]
            targets = targets[:allowed_max_length]

        inputs_lst.append(inputs)
        targets_lst.append(targets)

    # 将输入和目标列表转换为张量并移至目标设备
    inputs_tensor = torch.stack(inputs_lst).to(device)
    targets_tensor = torch.stack(targets_lst).to(device)

    return inputs_tensor, targets_tensor


# 下载并加载文件的函数
def download_and_load_file(file_path, url):

    # 如果文件不存在则下载
    if not os.path.exists(file_path):
        with urllib.request.urlopen(url) as response:
            text_data = response.read().decode("utf-8")
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


# 格式化Phi模型输入的函数
def format_input_phi(entry):
    # 格式化指令文本
    instruction_text = (
        f"<|user|>\n{entry['instruction']}"
    )

    # 格式化输入文本（如果存在）
    input_text = f"\n{entry['input']}" if entry["input"] else ""

    return instruction_text + input_text


# 格式化标准输入的函数
def format_input(entry):
    # 格式化指令文本
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )

    # 格式化输入文本（如果存在）
    input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""

    return instruction_text + input_text


# 绘制损失曲线的函数
def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses, plot_name):
    # 创建图形和主坐标轴
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # 绘制训练和验证损失
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

    # 创建第二个x轴显示已处理的token数量
    ax2 = ax1.twiny()
    ax2.plot(tokens_seen, train_losses, alpha=0)
    ax2.set_xlabel("Tokens seen")

    # 调整布局并保存图形
    fig.tight_layout()
    print(f"Plot saved as {plot_name}")
    plt.savefig(plot_name)


# 主函数
def main(mask_instructions=False, alpaca52k=False, phi3_prompt=False, lora=False):
    # 打印包版本信息
    print()
    pkgs = [
        "matplotlib",
        "tiktoken",
        "torch",
        "tqdm",
        "tensorflow",
    ]
    for p in pkgs:
        print(f"{p} version: {version(p)}")
    print(50*"-")

    # 下载并准备数据集
    file_path = "instruction-data.json"

    # 选择数据源URL
    if alpaca52k:
        url = "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json"
    else:
        url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch07/01_main-chapter-code/instruction-data.json"
    data = download_and_load_file(file_path, url)

    # 划分数据集
    train_portion = int(len(data) * 0.85)
    test_portion = int(len(data) * 0.1)

    train_data = data[:train_portion]
    test_data = data[train_portion:train_portion + test_portion]
    val_data = data[train_portion + test_portion:]

    # 打印数据集大小
    print("Training set length:", len(train_data))
    print("Validation set length:", len(val_data))
    print("Test set length:", len(test_data))
    print(50*"-")

    # 初始化tokenizer和设备
    tokenizer = tiktoken.get_encoding("gpt2")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    print(50*"-")

    # 设置序列最大长度
    if alpaca52k:
        allowed_max_length = 512
    else:
        allowed_max_length = 1024

    # 检查不兼容的选项
    if mask_instructions and phi3_prompt:
        raise ValueError("Simultaneous support for instruction masking and the Phi-3 prompt template has not been implemented, yet.")

    # 根据选项配置数据集和收集函数
    if mask_instructions:
        customized_collate_fn = partial(custom_collate_with_masking_fn, device=device, allowed_max_length=allowed_max_length)
        CustomDataset = InstructionDatasetWithMasking
    elif phi3_prompt:
        customized_collate_fn = partial(custom_collate_fn, device=device, allowed_max_length=allowed_max_length)
        CustomDataset = InstructionDatasetPhi
    else:
        customized_collate_fn = partial(custom_collate_fn, device=device, allowed_max_length=allowed_max_length)
        CustomDataset = InstructionDataset

    # 设置数据加载器参数
    num_workers = 0

    if alpaca52k:
        batch_size = 4
    else:
        batch_size = 8

    # 设置随机种子
    torch.manual_seed(123)

    # 创建训练数据加载器
    train_dataset = CustomDataset(train_data, tokenizer)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=customized_collate_fn,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers
    )

    # 创建验证数据加载器
    val_dataset = CustomDataset(val_data, tokenizer)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        collate_fn=customized_collate_fn,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers
    )

    # 加载预训练模型的配置
    BASE_CONFIG = {
        "vocab_size": 50257,
        "context_length": 1024,
        "drop_rate": 0.0,
        "qkv_bias": True
    }

    # 定义不同规模模型的配置
    model_configs = {
        "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
        "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
        "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
        "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
    }

    # 选择要使用的模型
    CHOOSE_MODEL = "gpt2-medium (355M)"

    # 更新基础配置
    BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

    # 获取模型大小并下载预训练权重
    model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")
    settings, params = download_and_load_gpt2(model_size=model_size, models_dir="gpt2")

    # 创建模型实例并加载权重
    model = GPTModel(BASE_CONFIG)
    load_weights_into_gpt(model, params)
    model.eval()
    model.to(device)

    print("Loaded model:", CHOOSE_MODEL)
    print(50*"-")

    # 如果使用LoRA，配置模型参数
    if lora:
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total trainable parameters before: {total_params:,}")

        for param in model.parameters():
            param.requires_grad = False

        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total trainable parameters after: {total_params:,}")
        replace_linear_with_lora(model, rank=16, alpha=16)

        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total trainable LoRA parameters: {total_params:,}")
        model.to(device)

    # 计算初始损失
    print("Initial losses")
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=5)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=5)

    print("   Training loss:", train_loss)
    print("   Validation loss:", val_loss)

    # 记录开始时间
    start_time = time.time()

    # 设置训练参数
    num_epochs = 2
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.00005, weight_decay=0.1)

    torch.manual_seed(123)

    # 准备起始上下文
    start_context = format_input_phi(val_data[0]) if phi3_prompt else format_input(val_data[0])

    # 训练模型
    train_losses, val_losses, tokens_seen = train_model_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=num_epochs, eval_freq=5, eval_iter=5,
        start_context=start_context, tokenizer=tokenizer
    )

    # 计算训练时间
    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60
    print(f"Training completed in {execution_time_minutes:.2f} minutes.")

    # 生成epoch数组
    epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))

    # 设置损失曲线图文件名
    plot_name = "loss-plot.pdf"
    if mask_instructions:
        plot_name = plot_name.replace(".pdf", "-mask-instructions.pdf")
    if alpaca52k:
        plot_name = plot_name.replace(".pdf", "-alpaca52k.pdf")
    if phi3_prompt:
        plot_name = plot_name.replace(".pdf", "-phi3-prompt.pdf")
    if lora:
        plot_name = plot_name.replace(".pdf", "-lora.pdf")
    if not any([mask_instructions, alpaca52k, phi3_prompt, lora]):
        plot_name = plot_name.replace(".pdf", "-baseline.pdf")

    # 绘制损失曲线
    plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses, plot_name)
    print(50*"-")

    # 生成测试集响应
    print("Generating responses")
    for i, entry in tqdm(enumerate(test_data), total=len(test_data)):

        # 格式化输入文本
        input_text = format_input_phi(entry) if phi3_prompt else format_input(entry)

        # 生成响应
        token_ids = generate(
            model=model,
            idx=text_to_token_ids(input_text, tokenizer).to(device),
            max_new_tokens=256,
            context_size=BASE_CONFIG["context_length"],
            eos_id=50256
        )
        generated_text = token_ids_to_text(token_ids, tokenizer)

        # 提取响应文本
        if phi3_prompt:
            response_text = generated_text[len(input_text):].replace("<|assistant|>:", "").strip()
        else:
            response_text = generated_text[len(input_text):].replace("### Response:", "").strip()

        # 保存响应
        test_data[i]["model_response"] = response_text

    # 设置输出文件名
    test_data_path = "instruction-data-with-response.json"
    file_name = f"{re.sub(r'[ ()]', '', CHOOSE_MODEL) }-sft.pth"

    # 根据不同选项修改文件名
    if mask_instructions:
        test_data_path = test_data_path.replace(".json", "-mask-instructions.json")
        file_name = file_name.replace(".pth", "-mask-instructions.pth")
    if alpaca52k:
        test_data_path = test_data_path.replace(".json", "-alpaca52k.json")
        file_name = file_name.replace(".pth", "-alpaca52k.pth")
    if phi3_prompt:
        test_data_path = test_data_path.replace(".json", "-phi3-prompt.json")
        file_name = file_name.replace(".pth", "-phi3-prompt.pth")
    if lora:
        test_data_path = test_data_path.replace(".json", "-lora.json")
        file_name = file_name.replace(".pth", "-lora.pth")
    if not any([mask_instructions, alpaca52k, phi3_prompt, lora]):
        test_data_path = test_data_path.replace(".json", "-baseline.json")
        file_name = file_name.replace(".pth", "-baseline.pth")

    # 保存测试数据和响应
    with open(test_data_path, "w") as file:
        json.dump(test_data, file, indent=4)
    print(f"Responses saved as {test_data_path}")

    # 保存模型
    torch.save(model.state_dict(), file_name)
    print(f"Model saved as {file_name}")


# 主程序入口
if __name__ == "__main__":

    # 导入参数解析器
    import argparse

    # 创建参数解析器
    parser = argparse.ArgumentParser(
        description="Instruction finetune a GPT model"
    )
    # 定义可用选项
    options = {"baseline", "mask_instructions", "alpaca_52k", "phi3_prompt", "lora"}
    # 添加命令行参数
    parser.add_argument(
        "--exercise_solution",
        type=str,
        default="last_block",
        help=(
            f"Which experiment to run. Options: {options}."
        )
    )
    # 解析命令行参数
    args = parser.parse_args()

    # 根据参数选择运行不同的实验
    if args.exercise_solution == "baseline":
        main()
    elif args.exercise_solution == "mask_instructions":
        main(mask_instructions=True)
    elif args.exercise_solution == "alpaca_52k":
        main(alpaca52k=True)
    elif args.exercise_solution == "phi3_prompt":
        main(phi3_prompt=True)
    elif args.exercise_solution == "lora":
        main(lora=True)
    else:
        raise ValueError(f"{args.exercise_solution} is not a valid --args.exercise_solution option. Options: {options}")
