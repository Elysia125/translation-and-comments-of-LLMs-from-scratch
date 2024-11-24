# 版权声明：由Sebastian Raschka根据Apache License 2.0许可发布(见LICENSE.txt)
# 来源："从零开始构建大型语言模型"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# 代码: https://github.com/rasbt/LLMs-from-scratch

# 导入所需的Python标准库
import argparse  # 用于解析命令行参数
import math     # 数学运算
import os       # 操作系统接口
from pathlib import Path  # 文件路径处理
import time    # 时间相关功能
import urllib.request  # 用于下载文件
import zipfile  # 处理zip文件

# 导入第三方库
import pandas as pd  # 数据处理
import tiktoken    # OpenAI的分词器
import torch       # PyTorch深度学习框架
from torch.utils.data import DataLoader  # 数据加载器
from torch.utils.data import Dataset     # 数据集基类

# 导入自定义模块
from gpt_download import download_and_load_gpt2  # 下载GPT-2模型
from previous_chapters import GPTModel, load_weights_into_gpt  # 从之前章节导入模型相关代码


class LoRALayer(torch.nn.Module):
    """LoRA层的实现"""
    def __init__(self, in_dim, out_dim, rank, alpha):
        super().__init__()
        # 初始化A矩阵(低秩分解的第一部分)
        self.A = torch.nn.Parameter(torch.empty(in_dim, rank))
        torch.nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        # 初始化B矩阵(低秩分解的第二部分)
        self.B = torch.nn.Parameter(torch.zeros(rank, out_dim))
        self.alpha = alpha  # 缩放因子

    def forward(self, x):
        """前向传播"""
        x = self.alpha * (x @ self.A @ self.B)  # 矩阵乘法实现低秩更新
        return x


class LinearWithLoRA(torch.nn.Module):
    """带LoRA的线性层"""
    def __init__(self, linear, rank, alpha):
        super().__init__()
        self.linear = linear  # 原始线性层
        # 创建LoRA层
        self.lora = LoRALayer(
            linear.in_features, linear.out_features, rank, alpha
        )

    def forward(self, x):
        """前向传播：原始输出加上LoRA的输出"""
        return self.linear(x) + self.lora(x)


class LinearWithLoRAMerged(torch.nn.Module):
    """带合并权重的LoRA线性层实现"""
    def __init__(self, linear, rank, alpha):
        super().__init__()
        self.linear = linear  # 原始线性层
        # 创建LoRA层
        self.lora = LoRALayer(
            linear.in_features, linear.out_features, rank, alpha
        )

    def forward(self, x):
        """前向传播：使用合并后的权重矩阵进行计算"""
        lora = self.lora.A @ self.lora.B  # 计算LoRA权重
        combined_weight = self.linear.weight + self.lora.alpha*lora.T  # 合并权重
        return torch.nn.functional.linear(x, combined_weight, self.linear.bias)


class SpamDataset(Dataset):
    """垃圾邮件数据集类"""
    def __init__(self, csv_file, tokenizer, max_length=None, pad_token_id=50256, no_padding=False):
        self.data = pd.read_csv(csv_file)  # 读取CSV文件
        # 设置最大长度
        self.max_length = max_length if max_length is not None else self._longest_encoded_length(tokenizer)

        # 预先对文本进行分词
        self.encoded_texts = [
            tokenizer.encode(text)[:self.max_length]
            for text in self.data["Text"]
        ]

        if not no_padding:
            # 对序列进行填充到最大长度
            self.encoded_texts = [
                et + [pad_token_id] * (self.max_length - len(et))
                for et in self.encoded_texts
            ]

    def __getitem__(self, index):
        """获取单个数据样本"""
        encoded = self.encoded_texts[index]
        label = self.data.iloc[index]["Label"]
        return torch.tensor(encoded, dtype=torch.long), torch.tensor(label, dtype=torch.long)

    def __len__(self):
        """返回数据集大小"""
        return len(self.data)

    def _longest_encoded_length(self, tokenizer):
        """计算数据集中最长序列的长度"""
        max_length = 0
        for text in self.data["Text"]:
            encoded_length = len(tokenizer.encode(text))
            if encoded_length > max_length:
                max_length = encoded_length
        return max_length


def download_and_unzip(url, zip_path, extract_to, new_file_path):
    """下载并解压数据集"""
    if new_file_path.exists():
        print(f"{new_file_path} already exists. Skipping download and extraction.")
        return

    # 下载文件
    with urllib.request.urlopen(url) as response:
        with open(zip_path, "wb") as out_file:
            out_file.write(response.read())

    # 解压文件
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)

    # 重命名文件以表明其格式
    original_file = Path(extract_to) / "SMSSpamCollection"
    os.rename(original_file, new_file_path)
    print(f"File downloaded and saved as {new_file_path}")


def random_split(df, train_frac, validation_frac):
    """随机分割数据集为训练集、验证集和测试集"""
    # 打乱整个DataFrame
    df = df.sample(frac=1, random_state=123).reset_index(drop=True)

    # 计算分割索引
    train_end = int(len(df) * train_frac)
    validation_end = train_end + int(len(df) * validation_frac)

    # 分割DataFrame
    train_df = df[:train_end]
    validation_df = df[train_end:validation_end]
    test_df = df[validation_end:]

    return train_df, validation_df, test_df


def create_dataset_csvs(new_file_path):
    """创建数据集的CSV文件"""
    # 读取原始数据
    df = pd.read_csv(new_file_path, sep="\t", header=None, names=["Label", "Text"])

    # 创建平衡数据集
    n_spam = df[df["Label"] == "spam"].shape[0]
    ham_sampled = df[df["Label"] == "ham"].sample(n_spam, random_state=123)
    balanced_df = pd.concat([ham_sampled, df[df["Label"] == "spam"]])
    balanced_df = balanced_df.sample(frac=1, random_state=123).reset_index(drop=True)
    balanced_df["Label"] = balanced_df["Label"].map({"ham": 0, "spam": 1})

    # 分割并保存CSV文件
    train_df, validation_df, test_df = random_split(balanced_df, 0.7, 0.1)
    train_df.to_csv("train.csv", index=None)
    validation_df.to_csv("validation.csv", index=None)
    test_df.to_csv("test.csv", index=None)


def instantiate_model(choose_model, load_weights):
    """实例化GPT模型"""

    # 基础配置
    BASE_CONFIG = {
        "vocab_size": 50257,     # 词汇表大小
        "context_length": 1024,  # 上下文长度
        "drop_rate": 0.0,        # Dropout率
        "qkv_bias": True         # 查询-键-值偏置
    }

    # 不同规模模型的配置
    model_configs = {
        "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
        "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
        "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
        "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
    }

    # 更新配置
    BASE_CONFIG.update(model_configs[choose_model])

    # 初始化模型
    if not load_weights:
        torch.manual_seed(123)
    model = GPTModel(BASE_CONFIG, disable_causal_mask=args.disable_causal_mask)

    # 加载预训练权重
    if load_weights:
        model_size = choose_model.split(" ")[-1].lstrip("(").rstrip(")")
        settings, params = download_and_load_gpt2(model_size=model_size, models_dir="gpt2")
        load_weights_into_gpt(model, params)

    model.eval()
    return model


def calc_loss_batch(input_batch, target_batch, model, device,
                    trainable_token_pos=-1, ignore_index=-100, average_embeddings=False):
    """计算单个批次的损失"""
    # 将数据移到指定设备
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)

    # 前向传播
    model_output = model(input_batch)
    if average_embeddings:
        # 在序列维度上平均(dim=1)
        logits = model_output.mean(dim=1)
    else:
        # 选择指定位置的嵌入
        logits = model_output[:, trainable_token_pos, :]

    # 计算交叉熵损失
    loss = torch.nn.functional.cross_entropy(logits, target_batch, ignore_index=ignore_index)
    return loss


def calc_loss_loader(data_loader, model, device,
                     num_batches=None, trainable_token_pos=-1,
                     ignore_index=-100, average_embeddings=False):
    """计算整个数据加载器的平均损失"""
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        # 如果指定的批次数超过数据加载器的批次数，则减少批次数
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(
                input_batch, target_batch, model, device,
                trainable_token_pos=trainable_token_pos, ignore_index=ignore_index,
                average_embeddings=average_embeddings
            )
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches


@torch.no_grad()  # 禁用梯度跟踪以提高效率
def calc_accuracy_loader(data_loader, model, device, num_batches=None,
                         trainable_token_pos=-1, average_embeddings=False):
    """计算模型在数据加载器上的准确率"""
    model.eval()
    correct_predictions, num_examples = 0, 0

    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            # 将数据移到设备并进行预测
            input_batch, target_batch = input_batch.to(device), target_batch.to(device)

            model_output = model(input_batch)
            if average_embeddings:
                # 在序列维度上平均(dim=1)
                logits = model_output.mean(dim=1)
            else:
                # 选择指定位置的嵌入
                logits = model_output[:, trainable_token_pos, :]

            # 获取预测标签
            predicted_labels = torch.argmax(logits, dim=-1)

            # 统计正确预测数
            num_examples += predicted_labels.shape[0]
            correct_predictions += (predicted_labels == target_batch).sum().item()
        else:
            break
    return correct_predictions / num_examples


def evaluate_model(model, train_loader, val_loader, device,
                   eval_iter, trainable_token_pos=-1,
                   ignore_index=-100, average_embeddings=False):
    """评估模型在训练集和验证集上的表现"""
    model.eval()
    with torch.no_grad():
        # 计算训练集和验证集的损失
        train_loss = calc_loss_loader(
            train_loader, model, device, num_batches=eval_iter,
            trainable_token_pos=trainable_token_pos, ignore_index=ignore_index,
            average_embeddings=average_embeddings
        )
        val_loss = calc_loss_loader(
            val_loader, model, device, num_batches=eval_iter,
            trainable_token_pos=trainable_token_pos, ignore_index=ignore_index,
            average_embeddings=average_embeddings
        )
    model.train()
    return train_loss, val_loss


def train_classifier_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
                            eval_freq, eval_iter, max_steps=None, trainable_token_pos=-1,
                            accumulation_steps=1, ignore_index=-100, average_embeddings=False):
    """简单的分类器训练函数"""
    # 初始化跟踪列表
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    examples_seen, global_step = 0, -1

    # 主训练循环
    for epoch in range(num_epochs):
        model.train()  # 设置模型为训练模式

        for batch_idx, (input_batch, target_batch) in enumerate(train_loader):
            # 计算损失
            loss = calc_loss_batch(
                input_batch, target_batch, model, device,
                trainable_token_pos=trainable_token_pos, ignore_index=ignore_index,
                average_embeddings=average_embeddings
            )

            # 如果使用梯度累积，则除以累积步数
            loss /= accumulation_steps

            loss.backward()  # 计算梯度

            # 判断是否需要更新参数
            is_update_step = ((batch_idx + 1) % accumulation_steps == 0) or ((batch_idx + 1) == len(train_loader))
            if is_update_step:
                optimizer.step()  # 更新模型参数
                optimizer.zero_grad()  # 重置梯度

            examples_seen += input_batch.shape[0]  # 更新已处理的样本数
            global_step += 1

            # 定期评估
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter,
                    trainable_token_pos=trainable_token_pos, ignore_index=ignore_index,
                    average_embeddings=average_embeddings
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

            if max_steps is not None and global_step > max_steps:
                break

        # 每个epoch结束后计算准确率
        train_accuracy = calc_accuracy_loader(
            train_loader, model, device, num_batches=eval_iter,
            trainable_token_pos=trainable_token_pos, average_embeddings=average_embeddings
        )
        val_accuracy = calc_accuracy_loader(
            val_loader, model, device, num_batches=eval_iter,
            trainable_token_pos=trainable_token_pos, average_embeddings=average_embeddings
        )
        print(f"Training accuracy: {train_accuracy*100:.2f}% | ", end="")
        print(f"Validation accuracy: {val_accuracy*100:.2f}%")
        train_accs.append(train_accuracy)
        val_accs.append(val_accuracy)

        if max_steps is not None and global_step > max_steps:
            break

    return train_losses, val_losses, train_accs, val_accs, examples_seen


def replace_linear_with_lora(model, rank, alpha, alternative=False):
    """将模型中的线性层替换为LoRA层"""
    for name, module in model.named_children():
        if isinstance(module, torch.nn.Linear):
            # 替换线性层
            if alternative:
                setattr(model, name, LinearWithLoRAMerged(module, rank, alpha))
            else:
                setattr(model, name, LinearWithLoRA(module, rank, alpha))
        else:
            # 递归处理子模块
            replace_linear_with_lora(module, rank, alpha)


if __name__ == "__main__":
    """主程序入口"""

    # 创建命令行参数解析器
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_size",
        type=str,
        default="gpt2-small (124M)",
        help=(
            "Which GPT model to use. Options: 'gpt2-small (124M)', 'gpt2-medium (355M)',"
            " 'gpt2-large (774M)', 'gpt2-xl (1558M)'."
        )
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="pretrained",
        help=(
            "Whether to use 'pretrained' or 'random' weights."
        )
    )
    parser.add_argument(
        "--trainable_layers",
        type=str,
        default="last_block",
        help=(
            "Which layers to train. Options: 'all', 'last_block', 'last_two_blocks', 'last_layer', 'lora', 'lora_alternative'."
        )
    )
    parser.add_argument(
        "--trainable_token_pos",
        type=str,
        default="last",
        help=(
            "Which token position to train. Options: 'first', 'last'."
        )
    )
    parser.add_argument(
        "--average_embeddings",
        action='store_true',
        default=False,
        help=(
            "Average the output embeddings from all tokens instead of using"
            " only the embedding at the token position specified by `--trainable_token_pos`."
        )
    )
    parser.add_argument(
        "--context_length",
        type=str,
        default="longest_training_example",
        help=(
            "The context length of the data inputs."
            " Options: 'longest_training_example', 'model_context_length' or integer value."
        )
    )
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=8,
        help=(
            "The LoRA rank when choosing `--trainable_layers lora`"
        )
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=8,
        help=(
            "The LoRA alpha value when choosing `--trainable_layers lora`"
        )
    )
    parser.add_argument(
        "--no_padding",
        action='store_true',
        default=False,
        help=(
            "Disable padding, which means each example may have a different length."
            " This requires setting `--batch_size 1`."
        )
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=5,
        help=(
            "Number of training epochs."
        )
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help=(
            "The batch size used for training."
        )
    )
    parser.add_argument(
        "--accumulation_steps",
        type=int,
        default=1,
        help=(
            "Accumulation steps to allow for gradient accumulation."
            " See https://sebastianraschka.com/blog/2023/llm-grad-accumulation.html for explanation."
            " For example, setting `batch_size=8` and `accumulation_steps=1` compute the exact same"
            " loss and weight updates as setting `batch_size=1` and `accumulation_steps=8`, however,"
            " the latter setting uses more iterations."
        )
    )
    parser.add_argument(
        "--disable_causal_mask",
        action='store_true',
        default=False,
        help=(
            "Disables the causal attention mask."
        )
    )
    parser.add_argument(
        "--ignore_index",
        type=int,
        default=-100,
        help=(
            "Sets the `ignore_index` in the cross-entropy loss."
        )
    )

    args = parser.parse_args()

    # 处理trainable_token_pos参数
    if args.trainable_token_pos == "first":
        args.trainable_token_pos = 0
    elif args.trainable_token_pos == "last":
        args.trainable_token_pos = -1
    else:
        raise ValueError("Invalid --trainable_token_pos argument")

    ###############################
    # 加载模型
    ###############################

    if args.weights == "pretrained":
        load_weights = True
    elif args.weights == "random":
        load_weights = False
    else:
        raise ValueError("Invalid --weights argument.")

    # 实例化模型
    model = instantiate_model(args.model_size, load_weights)
    for param in model.parameters():
        param.requires_grad = False

    # 根据模型大小设置输入特征维度
    if args.model_size == "gpt2-small (124M)":
        in_features = 768
    elif args.model_size == "gpt2-medium (355M)":
        in_features = 1024
    elif args.model_size == "gpt2-large (774M)":
        in_features = 1280
    elif args.model_size == "gpt2-xl (1558M)":
        in_features = 1600
    else:
        raise ValueError("Invalid --model_size argument")

    # 添加输出头
    torch.manual_seed(123)
    model.out_head = torch.nn.Linear(in_features=in_features, out_features=2)

    # 根据trainable_layers参数设置可训练层
    if args.trainable_layers == "last_layer":
        pass
    elif args.trainable_layers == "last_block" or args.trainable_layers == "last_two_blocks":
        for param in model.trf_blocks[-1].parameters():
            param.requires_grad = True
        for param in model.final_norm.parameters():
            param.requires_grad = True
        if args.trainable_layers == "last_two_blocks":
            for param in model.trf_blocks[-2].parameters():
                param.requires_grad = True
    elif args.trainable_layers == "all":
        for param in model.parameters():
            param.requires_grad = True
    elif args.trainable_layers in ("lora", "lora_alternative"):
        if args.trainable_layers == "lora_alternative":
            alternative = True
        else:
            alternative = False
        replace_linear_with_lora(model, rank=args.lora_rank, alpha=args.lora_alpha, alternative=alternative)
    else:
        raise ValueError("Invalid --trainable_layers argument.")

    # 将模型移到指定设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    ###############################
    # 实例化数据加载器
    ###############################

    # 数据集URL和路径设置
    url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
    zip_path = "sms_spam_collection.zip"
    extract_to = "sms_spam_collection"
    new_file_path = Path(extract_to) / "SMSSpamCollection.tsv"

    # 检查数据文件是否存在
    base_path = Path(".")
    file_names = ["train.csv", "validation.csv", "test.csv"]
    all_exist = all((base_path / file_name).exists() for file_name in file_names)

    # 如果数据文件不存在，下载并处理数据
    if not all_exist:
        download_and_unzip(url, zip_path, extract_to, new_file_path)
        create_dataset_csvs(new_file_path)

    # 初始化tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    train_dataset = None

    # 处理上下文长度设置
    if args.no_padding:
        max_length = None
    else:
        if args.context_length == "model_context_length":
            max_length = model.pos_emb.weight.shape[0]
        elif args.context_length == "longest_training_example":
            train_dataset = SpamDataset(base_path / "train.csv", max_length=None, tokenizer=tokenizer, no_padding=args.no_padding)
            max_length = train_dataset.max_length
        else:
            try:
                max_length = int(args.context_length)
            except ValueError:
                raise ValueError("Invalid --context_length argument")

    # 创建数据集
    if train_dataset is None:
        train_dataset = SpamDataset(base_path / "train.csv", max_length=max_length, tokenizer=tokenizer, no_padding=args.no_padding)
    val_dataset = SpamDataset(base_path / "validation.csv", max_length=max_length, tokenizer=tokenizer, no_padding=args.no_padding)
    test_dataset = SpamDataset(base_path / "test.csv", max_length=max_length, tokenizer=tokenizer, no_padding=args.no_padding)

    tokenizer = tiktoken.get_encoding("gpt2")

    num_workers = 0

    # 创建数据加载器
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=args.batch_size,
        num_workers=num_workers,
        drop_last=False,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=args.batch_size,
        num_workers=num_workers,
        drop_last=False,
    )

    # 检查数据集长度是否超过模型上下文长度
    assert train_dataset.max_length <= model.pos_emb.weight.shape[0], (
        f"Dataset length {train_dataset.max_length} exceeds model's context "
        f"length {model.pos_emb.weight.shape[0]}. Reinitialize data sets with "
        f"`max_length={model.pos_emb.weight.shape[0]}`"
    )

    ###############################
    # 训练模型
    ###############################

    # 记录开始时间
    start_time = time.time()
    torch.manual_seed(123)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)

    # 训练模型
    train_losses, val_losses, train_accs, val_accs, examples_seen = train_classifier_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=args.num_epochs, eval_freq=50, eval_iter=5,
        max_steps=None, trainable_token_pos=args.trainable_token_pos,
        accumulation_steps=args.accumulation_steps, average_embeddings=args.average_embeddings
    )

    # 计算训练时间
    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60
    print(f"Training completed in {execution_time_minutes:.2f} minutes.")

    ###############################
    # 评估模型
    ###############################

    # 计算各数据集上的准确率
    train_accuracy = calc_accuracy_loader(
        train_loader, model, device,
        trainable_token_pos=args.trainable_token_pos, average_embeddings=args.average_embeddings
    )
    val_accuracy = calc_accuracy_loader(
        val_loader, model, device,
        trainable_token_pos=args.trainable_token_pos, average_embeddings=args.average_embeddings
    )
    test_accuracy = calc_accuracy_loader(
        test_loader, model, device,
        trainable_token_pos=args.trainable_token_pos, average_embeddings=args.average_embeddings
    )

    print(f"Training accuracy: {train_accuracy*100:.2f}%")
    print(f"Validation accuracy: {val_accuracy*100:.2f}%")
    print(f"Test accuracy: {test_accuracy*100:.2f}%")
