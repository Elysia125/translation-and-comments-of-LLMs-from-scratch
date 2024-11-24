# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

# 导入所需的库
import argparse  # 用于解析命令行参数
from pathlib import Path  # 用于处理文件路径
import time  # 用于计时

# 导入数据处理和机器学习相关的库
import pandas as pd  # 用于数据处理
import torch  # PyTorch深度学习框架
from torch.utils.data import DataLoader  # 用于数据加载
from torch.utils.data import Dataset  # 用于创建数据集类

# 导入Transformers相关组件
from transformers import AutoTokenizer, AutoModelForSequenceClassification  # 用于加载预训练模型和分词器


class IMDBDataset(Dataset):
    """IMDB数据集的自定义Dataset类"""
    def __init__(self, csv_file, tokenizer, max_length=None, pad_token_id=50256, use_attention_mask=False):
        self.data = pd.read_csv(csv_file)  # 读取CSV文件
        # 设置序列最大长度,如果未指定则计算数据集中最长序列的长度
        self.max_length = max_length if max_length is not None else self._longest_encoded_length(tokenizer)
        self.pad_token_id = pad_token_id  # 设置填充token的ID
        self.use_attention_mask = use_attention_mask  # 是否使用注意力掩码

        # 预处理文本:分词并填充
        self.encoded_texts = [
            tokenizer.encode(text, truncation=True, max_length=self.max_length)
            for text in self.data["text"]
        ]
        self.encoded_texts = [
            et + [pad_token_id] * (self.max_length - len(et))
            for et in self.encoded_texts
        ]

        # 如果需要,创建注意力掩码
        if self.use_attention_mask:
            self.attention_masks = [
                self._create_attention_mask(et)
                for et in self.encoded_texts
            ]
        else:
            self.attention_masks = None

    def _create_attention_mask(self, encoded_text):
        """创建注意力掩码:对实际token为1,填充token为0"""
        return [1 if token_id != self.pad_token_id else 0 for token_id in encoded_text]

    def __getitem__(self, index):
        """获取指定索引的数据项"""
        encoded = self.encoded_texts[index]  # 获取编码后的文本
        label = self.data.iloc[index]["label"]  # 获取标签

        # 获取或创建注意力掩码
        if self.use_attention_mask:
            attention_mask = self.attention_masks[index]
        else:
            attention_mask = torch.ones(self.max_length, dtype=torch.long)

        # 返回张量形式的数据
        return (
            torch.tensor(encoded, dtype=torch.long),
            torch.tensor(attention_mask, dtype=torch.long),
            torch.tensor(label, dtype=torch.long)
        )

    def __len__(self):
        """返回数据集的长度"""
        return len(self.data)

    def _longest_encoded_length(self, tokenizer):
        """计算数据集中最长序列的长度"""
        max_length = 0
        for text in self.data["text"]:
            encoded_length = len(tokenizer.encode(text))
            if encoded_length > max_length:
                max_length = encoded_length
        return max_length


def calc_loss_batch(input_batch, attention_mask_batch, target_batch, model, device):
    """计算单个批次的损失"""
    attention_mask_batch = attention_mask_batch.to(device)  # 将注意力掩码移到指定设备
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)  # 将输入和目标移到指定设备
    # logits = model(input_batch)[:, -1, :]  # Logits of last output token
    logits = model(input_batch, attention_mask=attention_mask_batch).logits  # 获取模型输出的logits
    loss = torch.nn.functional.cross_entropy(logits, target_batch)  # 计算交叉熵损失
    return loss


# Same as in chapter 5
def calc_loss_loader(data_loader, model, device, num_batches=None):
    """计算整个数据加载器的平均损失"""
    total_loss = 0.
    if num_batches is None:
        num_batches = len(data_loader)
    else:
        # Reduce the number of batches to match the total number of batches in the data loader
        # if num_batches exceeds the number of batches in the data loader
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, attention_mask_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, attention_mask_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches


@torch.no_grad()  # Disable gradient tracking for efficiency
def calc_accuracy_loader(data_loader, model, device, num_batches=None):
    """计算模型在数据加载器上的准确率"""
    model.eval()  # 设置模型为评估模式
    correct_predictions, num_examples = 0, 0  # 初始化正确预测数和样本总数

    if num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    for i, (input_batch, attention_mask_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            attention_mask_batch = attention_mask_batch.to(device)
            input_batch, target_batch = input_batch.to(device), target_batch.to(device)
            # logits = model(input_batch)[:, -1, :]  # Logits of last output token
            logits = model(input_batch, attention_mask=attention_mask_batch).logits
            predicted_labels = torch.argmax(logits, dim=1)  # 获取预测标签
            num_examples += predicted_labels.shape[0]  # 更新样本总数
            correct_predictions += (predicted_labels == target_batch).sum().item()  # 更新正确预测数
        else:
            break
    return correct_predictions / num_examples  # 返回准确率


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    """评估模型在训练集和验证集上的损失"""
    model.eval()  # 设置模型为评估模式
    with torch.no_grad():  # 禁用梯度计算
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()  # 将模型设回训练模式
    return train_loss, val_loss


def train_classifier_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
                            eval_freq, eval_iter, max_steps=None):
    """训练分类器的主函数"""
    # Initialize lists to track losses and tokens seen
    train_losses, val_losses, train_accs, val_accs = [], [], [], []  # 初始化记录列表
    examples_seen, global_step = 0, -1  # 初始化计数器

    # Main training loop
    for epoch in range(num_epochs):  # 遍历每个训练周期
        model.train()  # Set model to training mode

        for input_batch, attention_mask_batch, target_batch in train_loader:
            optimizer.zero_grad()  # Reset loss gradients from previous batch iteration
            loss = calc_loss_batch(input_batch, attention_mask_batch, target_batch, model, device)
            loss.backward()  # Calculate loss gradients
            optimizer.step()  # Update model weights using loss gradients
            examples_seen += input_batch.shape[0]  # New: track examples instead of tokens
            global_step += 1

            # Optional evaluation step
            if global_step % eval_freq == 0:  # 定期评估
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

            if max_steps is not None and global_step > max_steps:
                break

        # New: Calculate accuracy after each epoch
        train_accuracy = calc_accuracy_loader(train_loader, model, device, num_batches=eval_iter)
        val_accuracy = calc_accuracy_loader(val_loader, model, device, num_batches=eval_iter)
        print(f"Training accuracy: {train_accuracy*100:.2f}% | ", end="")
        print(f"Validation accuracy: {val_accuracy*100:.2f}%")
        train_accs.append(train_accuracy)
        val_accs.append(val_accuracy)

        if max_steps is not None and global_step > max_steps:
            break

    return train_losses, val_losses, train_accs, val_accs, examples_seen


if __name__ == "__main__":
    """主程序入口"""
    parser = argparse.ArgumentParser()  # 创建命令行参数解析器
    parser.add_argument(
        "--trainable_layers",
        type=str,
        default="all",
        help=(
            "Which layers to train. Options: 'all', 'last_block', 'last_layer'."
        )
    )
    parser.add_argument(
        "--use_attention_mask",
        type=str,
        default="true",
        help=(
            "Whether to use a attention mask for padding tokens. Options: 'true', 'false'."
        )
    )
    parser.add_argument(
        "--model",
        type=str,
        default="distilbert",
        help=(
            "Which model to train. Options: 'distilbert', 'bert', 'roberta'."
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
        default=5e-6,
        help=(
            "Learning rate."
        )
    )
    args = parser.parse_args()  # 解析命令行参数

    ###############################
    # Load model
    ###############################

    torch.manual_seed(123)  # 设置随机种子
    if args.model == "distilbert":  # 如果选择DistilBERT模型
        # 加载预训练的DistilBERT模型
        model = AutoModelForSequenceClassification.from_pretrained(
            "distilbert-base-uncased", num_labels=2
        )
        model.out_head = torch.nn.Linear(in_features=768, out_features=2)  # 添加输出层
        for param in model.parameters():  # 首先冻结所有参数
            param.requires_grad = False
        if args.trainable_layers == "last_layer":  # 如果只训练最后一层
            for param in model.out_head.parameters():
                param.requires_grad = True
        elif args.trainable_layers == "last_block":  # 如果训练最后一个块
            for param in model.pre_classifier.parameters():
                param.requires_grad = True
            for param in model.distilbert.transformer.layer[-1].parameters():
                param.requires_grad = True
        elif args.trainable_layers == "all":  # 如果训练所有层
            for param in model.parameters():
                param.requires_grad = True
        else:
            raise ValueError("Invalid --trainable_layers argument.")

        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")  # 加载对应的分词器

    elif args.model == "bert":  # 如果选择BERT模型
        # 加载预训练的BERT模型
        model = AutoModelForSequenceClassification.from_pretrained(
            "bert-base-uncased", num_labels=2
        )
        model.classifier = torch.nn.Linear(in_features=768, out_features=2)  # 添加分类器
        for param in model.parameters():  # 首先冻结所有参数
            param.requires_grad = False
        if args.trainable_layers == "last_layer":  # 如果只训练最后一层
            for param in model.classifier.parameters():
                param.requires_grad = True
        elif args.trainable_layers == "last_block":  # 如果训练最后一个块
            for param in model.classifier.parameters():
                param.requires_grad = True
            for param in model.bert.pooler.dense.parameters():
                param.requires_grad = True
            for param in model.bert.encoder.layer[-1].parameters():
                param.requires_grad = True
        elif args.trainable_layers == "all":  # 如果训练所有层
            for param in model.parameters():
                param.requires_grad = True
        else:
            raise ValueError("Invalid --trainable_layers argument.")

        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")  # 加载对应的分词器
    elif args.model == "roberta":  # 如果选择RoBERTa模型
        # 加载预训练的RoBERTa模型
        model = AutoModelForSequenceClassification.from_pretrained(
            "FacebookAI/roberta-large", num_labels=2
        )
        model.classifier.out_proj = torch.nn.Linear(in_features=1024, out_features=2)  # 添加输出投影层
        for param in model.parameters():  # 首先冻结所有参数
            param.requires_grad = False
        if args.trainable_layers == "last_layer":  # 如果只训练最后一层
            for param in model.classifier.parameters():
                param.requires_grad = True
        elif args.trainable_layers == "last_block":  # 如果训练最后一个块
            for param in model.classifier.parameters():
                param.requires_grad = True
            for param in model.roberta.encoder.layer[-1].parameters():
                param.requires_grad = True
        elif args.trainable_layers == "all":  # 如果训练所有层
            for param in model.parameters():
                param.requires_grad = True
        else:
            raise ValueError("Invalid --trainable_layers argument.")

        tokenizer = AutoTokenizer.from_pretrained("FacebookAI/roberta-large")  # 加载对应的分词器
    else:
        raise ValueError("Selected --model {args.model} not supported.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 设置设备(GPU/CPU)
    model.to(device)  # 将模型移到指定设备
    model.eval()  # 设置模型为评估模式

    ###############################
    # Instantiate dataloaders
    ###############################

    base_path = Path(".")  # 设置基础路径

    # 解析注意力掩码参数
    if args.use_attention_mask.lower() == "true":
        use_attention_mask = True
    elif args.use_attention_mask.lower() == "false":
        use_attention_mask = False
    else:
        raise ValueError("Invalid argument for `use_attention_mask`.")

    # 创建训练集
    train_dataset = IMDBDataset(
        base_path / "train.csv",
        max_length=256,
        tokenizer=tokenizer,
        pad_token_id=tokenizer.pad_token_id,
        use_attention_mask=use_attention_mask
    )
    # 创建验证集
    val_dataset = IMDBDataset(
        base_path / "validation.csv",
        max_length=256,
        tokenizer=tokenizer,
        pad_token_id=tokenizer.pad_token_id,
        use_attention_mask=use_attention_mask
    )
    # 创建测试集
    test_dataset = IMDBDataset(
        base_path / "test.csv",
        max_length=256,
        tokenizer=tokenizer,
        pad_token_id=tokenizer.pad_token_id,
        use_attention_mask=use_attention_mask
    )

    num_workers = 0  # 设置数据加载的工作进程数
    batch_size = 8  # 设置批次大小

    # 创建训练数据加载器
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )

    # 创建验证数据加载器
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=False,
    )

    # 创建测试数据加载器
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        drop_last=False,
    )

    ###############################
    # Train model
    ###############################

    start_time = time.time()  # 记录开始时间
    torch.manual_seed(123)  # 设置随机种子
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.1)  # 创建优化器

    # 训练模型
    train_losses, val_losses, train_accs, val_accs, examples_seen = train_classifier_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=args.num_epochs, eval_freq=50, eval_iter=20,
        max_steps=None
    )

    end_time = time.time()  # 记录结束时间
    execution_time_minutes = (end_time - start_time) / 60  # 计算执行时间(分钟)
    print(f"Training completed in {execution_time_minutes:.2f} minutes.")

    ###############################
    # Evaluate model
    ###############################

    print("\nEvaluating on the full datasets ...\n")

    # 在完整数据集上评估模型
    train_accuracy = calc_accuracy_loader(train_loader, model, device)
    val_accuracy = calc_accuracy_loader(val_loader, model, device)
    test_accuracy = calc_accuracy_loader(test_loader, model, device)

    # 打印最终结果
    print(f"Training accuracy: {train_accuracy*100:.2f}%")
    print(f"Validation accuracy: {val_accuracy*100:.2f}%")
    print(f"Test accuracy: {test_accuracy*100:.2f}%")
