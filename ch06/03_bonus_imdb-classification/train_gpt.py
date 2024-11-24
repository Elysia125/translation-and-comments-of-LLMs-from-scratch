# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

# 导入所需的Python库
import argparse  # 用于解析命令行参数
from pathlib import Path  # 用于处理文件路径
import time  # 用于计时

import pandas as pd  # 用于数据处理
import tiktoken  # OpenAI的分词器
import torch  # PyTorch深度学习框架
from torch.utils.data import DataLoader  # 用于批量加载数据
from torch.utils.data import Dataset  # 用于创建自定义数据集

# 导入自定义模块
from gpt_download import download_and_load_gpt2  # 用于下载和加载GPT-2模型
from previous_chapters import GPTModel, load_weights_into_gpt  # 导入GPT模型和权重加载函数


class IMDBDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=None, pad_token_id=50256):
        # 初始化IMDB数据集
        self.data = pd.read_csv(csv_file)  # 读取CSV文件
        self.max_length = max_length if max_length is not None else self._longest_encoded_length(tokenizer)  # 设置最大序列长度

        # 预处理文本数据
        self.encoded_texts = [
            tokenizer.encode(text)[:self.max_length]  # 对每个文本进行编码并截断
            for text in self.data["text"]
        ]
        # 对序列进行填充
        self.encoded_texts = [
            et + [pad_token_id] * (self.max_length - len(et))  # 使用pad_token_id填充到相同长度
            for et in self.encoded_texts
        ]

    def __getitem__(self, index):
        # 返回单个数据样本
        encoded = self.encoded_texts[index]  # 获取编码后的文本
        label = self.data.iloc[index]["label"]  # 获取对应的标签
        return torch.tensor(encoded, dtype=torch.long), torch.tensor(label, dtype=torch.long)

    def __len__(self):
        # 返回数据集大小
        return len(self.data)

    def _longest_encoded_length(self, tokenizer):
        # 计算数据集中最长序列的长度
        max_length = 0
        for text in self.data["text"]:
            encoded_length = len(tokenizer.encode(text))
            if encoded_length > max_length:
                max_length = encoded_length
        return max_length


def instantiate_model(choose_model, load_weights):
    # 实例化GPT模型

    # 基础配置参数
    BASE_CONFIG = {
        "vocab_size": 50257,     # 词汇表大小
        "context_length": 1024,  # 上下文长度
        "drop_rate": 0.0,        # Dropout率
        "qkv_bias": True         # 是否使用Query-Key-Value偏置
    }

    # 不同规模GPT-2模型的配置
    model_configs = {
        "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
        "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
        "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
        "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
    }

    BASE_CONFIG.update(model_configs[choose_model])  # 更新配置

    if not load_weights:
        torch.manual_seed(123)  # 设置随机种子
    model = GPTModel(BASE_CONFIG)  # 创建模型

    if load_weights:
        # 加载预训练权重
        model_size = choose_model.split(" ")[-1].lstrip("(").rstrip(")")
        settings, params = download_and_load_gpt2(model_size=model_size, models_dir="gpt2")
        load_weights_into_gpt(model, params)

    model.eval()  # 设置为评估模式
    return model


def calc_loss_batch(input_batch, target_batch, model, device,
                    trainable_token_pos=-1, average_embeddings=False):
    # 计算单个批次的损失
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)  # 将数据移到指定设备

    model_output = model(input_batch)  # 前向传播
    if average_embeddings:
        # 对序列维度取平均
        logits = model_output.mean(dim=1)
    else:
        # 选择指定位置的嵌入
        logits = model_output[:, trainable_token_pos, :]

    loss = torch.nn.functional.cross_entropy(logits, target_batch)  # 计算交叉熵损失
    return loss


def calc_loss_loader(data_loader, model, device,
                     num_batches=None, trainable_token_pos=-1,
                     average_embeddings=False):
    # 计算整个数据加载器的平均损失
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        # 如果指定的批次数超过数据加载器的批次数，则使用较小的值
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(
                input_batch, target_batch, model, device,
                trainable_token_pos=trainable_token_pos, average_embeddings=average_embeddings
            )
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches


@torch.no_grad()  # 禁用梯度计算以提高效率
def calc_accuracy_loader(data_loader, model, device,
                         num_batches=None, trainable_token_pos=-1,
                         average_embeddings=False):
    # 计算准确率
    model.eval()  # 设置为评估模式
    correct_predictions, num_examples = 0, 0

    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            input_batch, target_batch = input_batch.to(device), target_batch.to(device)

            model_output = model(input_batch)  # 前向传播
            if average_embeddings:
                # 对序列维度取平均
                logits = model_output.mean(dim=1)
            else:
                # 选择指定位置的嵌入
                logits = model_output[:, trainable_token_pos, :]

            predicted_labels = torch.argmax(logits, dim=-1)  # 获取预测标签

            num_examples += predicted_labels.shape[0]  # 更新样本数
            correct_predictions += (predicted_labels == target_batch).sum().item()  # 更新正确预测数
        else:
            break
    return correct_predictions / num_examples  # 返回准确率


def evaluate_model(model, train_loader, val_loader, device, eval_iter,
                   trainable_token_pos=-1, average_embeddings=False):
    # 评估模型在训练集和验证集上的性能
    model.eval()  # 设置为评估模式
    with torch.no_grad():
        train_loss = calc_loss_loader(
            train_loader, model, device, num_batches=eval_iter,
            trainable_token_pos=trainable_token_pos, average_embeddings=average_embeddings
        )
        val_loss = calc_loss_loader(
            val_loader, model, device, num_batches=eval_iter,
            trainable_token_pos=trainable_token_pos, average_embeddings=average_embeddings
        )
    model.train()  # 恢复为训练模式
    return train_loss, val_loss


def train_classifier_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
                            eval_freq, eval_iter, max_steps=None, trainable_token_pos=-1,
                            average_embeddings=False):
    # 训练分类器的主函数
    # 初始化跟踪列表
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    examples_seen, global_step = 0, -1

    # 主训练循环
    for epoch in range(num_epochs):
        model.train()  # 设置为训练模式

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()  # 清零梯度
            loss = calc_loss_batch(input_batch, target_batch, model, device,
                                   trainable_token_pos=trainable_token_pos, average_embeddings=average_embeddings)
            loss.backward()  # 反向传播
            optimizer.step()  # 更新模型参数
            examples_seen += input_batch.shape[0]  # 更新已处理的样本数
            global_step += 1

            # 定期评估
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter,
                    trainable_token_pos=trainable_token_pos, average_embeddings=average_embeddings
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


if __name__ == "__main__":
    # 主程序入口

    # 设置命令行参数解析器
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
            "Which layers to train. Options: 'all', 'last_block', 'last_layer'."
        )
    )
    parser.add_argument(
        "--trainable_token_pos",
        type=str,
        default="last",
        help=(
            "Which token to train. Options: 'first', 'last'."
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
        default="256",
        help=(
            "The context length of the data inputs."
            "Options: 'longest_training_example', 'model_context_length' or integer value."
        )
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=1,
        help=(
            "Number of epochs."
        )
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-5,
        help=(
            "Learning rate."
        )
    )
    args = parser.parse_args()  # 解析命令行参数

    # 处理trainable_token_pos参数
    if args.trainable_token_pos == "first":
        args.trainable_token_pos = 0
    elif args.trainable_token_pos == "last":
        args.trainable_token_pos = -1
    else:
        raise ValueError("Invalid --trainable_token_pos argument")

    ###############################
    # Load model
    ###############################

    # 确定是否使用预训练权重
    if args.weights == "pretrained":
        load_weights = True
    elif args.weights == "random":
        load_weights = False
    else:
        raise ValueError("Invalid --weights argument.")

    # 实例化模型
    model = instantiate_model(args.model_size, load_weights)
    for param in model.parameters():
        param.requires_grad = False  # 默认冻结所有参数

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

    # 添加分类头
    torch.manual_seed(123)
    model.out_head = torch.nn.Linear(in_features=in_features, out_features=2)

    # 根据trainable_layers参数设置可训练层
    if args.trainable_layers == "last_layer":
        pass
    elif args.trainable_layers == "last_block":
        for param in model.trf_blocks[-1].parameters():
            param.requires_grad = True
        for param in model.final_norm.parameters():
            param.requires_grad = True
    elif args.trainable_layers == "all":
        for param in model.parameters():
            param.requires_grad = True
    else:
        raise ValueError("Invalid --trainable_layers argument.")

    # 将模型移到指定设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    ###############################
    # Instantiate dataloaders
    ###############################

    # 设置数据路径和分词器
    base_path = Path(".")
    tokenizer = tiktoken.get_encoding("gpt2")

    # 处理上下文长度参数
    train_dataset = None
    if args.context_length == "model_context_length":
        max_length = model.pos_emb.weight.shape[0]
    elif args.context_length == "longest_training_example":
        train_dataset = IMDBDataset(base_path / "train.csv", max_length=None, tokenizer=tokenizer)
        max_length = train_dataset.max_length
    else:
        try:
            max_length = int(args.context_length)
        except ValueError:
            raise ValueError("Invalid --context_length argument")

    # 创建数据集
    if train_dataset is None:
        train_dataset = IMDBDataset(base_path / "train.csv", max_length=max_length, tokenizer=tokenizer)
    val_dataset = IMDBDataset(base_path / "validation.csv", max_length=max_length, tokenizer=tokenizer)
    test_dataset = IMDBDataset(base_path / "test.csv", max_length=max_length, tokenizer=tokenizer)

    # 设置数据加载器参数
    num_workers = 0
    batch_size = 8

    # 创建数据加载器
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=False,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=False,
    )

    ###############################
    # Train model
    ###############################

    # 开始训练
    start_time = time.time()
    torch.manual_seed(123)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.1)

    # 训练模型
    train_losses, val_losses, train_accs, val_accs, examples_seen = train_classifier_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=args.num_epochs, eval_freq=50, eval_iter=20,
        max_steps=None, trainable_token_pos=args.trainable_token_pos,
        average_embeddings=args.average_embeddings
    )

    # 计算训练时间
    end_time = time.time()
    execution_time_minutes = (end_time - start_time) / 60
    print(f"Training completed in {execution_time_minutes:.2f} minutes.")

    ###############################
    # Evaluate model
    ###############################

    print("\nEvaluating on the full datasets ...\n")

    # 在完整数据集上评估模型
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

    # 打印最终结果
    print(f"Training accuracy: {train_accuracy*100:.2f}%")
    print(f"Validation accuracy: {val_accuracy*100:.2f}%")
    print(f"Test accuracy: {test_accuracy*100:.2f}%")
