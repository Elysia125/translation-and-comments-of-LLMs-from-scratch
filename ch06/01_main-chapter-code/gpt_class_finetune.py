# 版权所有 (c) Sebastian Raschka，依据 Apache License 2.0（见 LICENSE.txt）。
# 来源于《从头构建大型语言模型》
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# 代码: https://github.com/rasbt/LLMs-from-scratch

# 这是一个包含第六章主要要点的摘要文件。

import urllib.request  # 导入用于处理 URL 请求的库
import zipfile  # 导入用于处理 ZIP 文件的库
import os  # 导入用于与操作系统交互的库
from pathlib import Path  # 导入用于处理路径的库
import time  # 导入用于时间处理的库

import matplotlib.pyplot as plt  # 导入用于绘图的库
import pandas as pd  # 导入用于数据处理的库
import tiktoken  # 导入用于文本编码的库
import torch  # 导入 PyTorch 库
from torch.utils.data import Dataset, DataLoader  # 从 PyTorch 导入数据集和数据加载器

from gpt_download import download_and_load_gpt2  # 从 gpt_download 模块导入下载和加载 GPT2 的函数
from previous_chapters import GPTModel, load_weights_into_gpt  # 从 previous_chapters 模块导入 GPTModel 和加载权重的函数


def download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path, test_mode=False):
    # 检查数据文件是否已存在
    if data_file_path.exists():
        print(f"{data_file_path} 已存在，跳过下载和解压。")
        return

    if test_mode:  # 如果在测试模式下，尝试多次下载，因为 CI 有时会遇到连接问题
        max_retries = 5  # 最大重试次数
        delay = 5  # 重试之间的延迟（秒）
        for attempt in range(max_retries):
            try:
                # 下载文件
                with urllib.request.urlopen(url, timeout=10) as response:
                    with open(zip_path, "wb") as out_file:
                        out_file.write(response.read())  # 将下载的内容写入 ZIP 文件
                break  # 如果下载成功，跳出循环
            except urllib.error.URLError as e:
                print(f"尝试 {attempt + 1} 失败: {e}")
                if attempt < max_retries - 1:
                    time.sleep(delay)  # 等待后重试
                else:
                    print("经过多次尝试后下载文件失败。")
                    return  # 如果所有重试都失败，则退出

    else:  # 按照章节中的代码执行
        # 下载文件
        with urllib.request.urlopen(url) as response:
            with open(zip_path, "wb") as out_file:
                out_file.write(response.read())  # 将下载的内容写入 ZIP 文件

    # 解压 ZIP 文件
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extracted_path)  # 解压到指定路径

    # 添加 .tsv 文件扩展名
    original_file_path = Path(extracted_path) / "SMSSpamCollection"  # 原始文件路径
    os.rename(original_file_path, data_file_path)  # 重命名文件
    print(f"文件已下载并保存为 {data_file_path}")  # 打印文件保存信息


def create_balanced_dataset(df):
    # 计算 "spam" 的实例数量
    num_spam = df[df["Label"] == "spam"].shape[0]

    # 随机抽样 "ham" 实例以匹配 "spam" 实例的数量
    ham_subset = df[df["Label"] == "ham"].sample(num_spam, random_state=123)

    # 将 "ham" 子集与 "spam" 结合
    balanced_df = pd.concat([ham_subset, df[df["Label"] == "spam"]])

    return balanced_df  # 返回平衡后的数据集


def random_split(df, train_frac, validation_frac):
    # 打乱整个 DataFrame
    df = df.sample(frac=1, random_state=123).reset_index(drop=True)

    # 计算分割索引
    train_end = int(len(df) * train_frac)  # 训练集结束索引
    validation_end = train_end + int(len(df) * validation_frac)  # 验证集结束索引

    # 分割 DataFrame
    train_df = df[:train_end]  # 训练集
    validation_df = df[train_end:validation_end]  # 验证集
    test_df = df[validation_end:]  # 测试集

    return train_df, validation_df, test_df  # 返回训练集、验证集和测试集


class SpamDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=None, pad_token_id=50256):
        self.data = pd.read_csv(csv_file)  # 从 CSV 文件读取数据

        # 预先标记文本
        self.encoded_texts = [
            tokenizer.encode(text) for text in self.data["Text"]  # 使用 tokenizer 对文本进行编码
        ]

        if max_length is None:
            self.max_length = self._longest_encoded_length()  # 如果没有指定最大长度，则计算最长编码长度
        else:
            self.max_length = max_length
            # 如果序列长度超过最大长度，则截断序列
            self.encoded_texts = [
                encoded_text[:self.max_length]
                for encoded_text in self.encoded_texts
            ]

        # 将序列填充到最长序列
        self.encoded_texts = [
            encoded_text + [pad_token_id] * (self.max_length - len(encoded_text))  # 使用填充标记填充
            for encoded_text in self.encoded_texts
        ]

    def __getitem__(self, index):
        encoded = self.encoded_texts[index]  # 获取编码文本
        label = self.data.iloc[index]["Label"]  # 获取标签
        return (
            torch.tensor(encoded, dtype=torch.long),  # 返回编码文本的张量
            torch.tensor(label, dtype=torch.long)  # 返回标签的张量
        )

    def __len__(self):
        return len(self.data)  # 返回数据集的长度

    def _longest_encoded_length(self):
        max_length = 0  # 初始化最大长度
        for encoded_text in self.encoded_texts:
            encoded_length = len(encoded_text)  # 获取编码文本的长度
            if encoded_length > max_length:
                max_length = encoded_length  # 更新最大长度
        return max_length  # 返回最长编码长度


def calc_accuracy_loader(data_loader, model, device, num_batches=None):
    model.eval()  # 设置模型为评估模式
    correct_predictions, num_examples = 0, 0  # 初始化正确预测和示例数量

    if num_batches is None:
        num_batches = len(data_loader)  # 如果没有指定批次数，则使用数据加载器的长度
    else:
        num_batches = min(num_batches, len(data_loader))  # 取最小值

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            input_batch, target_batch = input_batch.to(device), target_batch.to(device)  # 将数据移动到设备

            with torch.no_grad():  # 不计算梯度
                logits = model(input_batch)[:, -1, :]  # 获取最后一个输出标记的 logits
            predicted_labels = torch.argmax(logits, dim=-1)  # 获取预测标签

            num_examples += predicted_labels.shape[0]  # 更新示例数量
            correct_predictions += (predicted_labels == target_batch).sum().item()  # 更新正确预测数量
        else:
            break
    return correct_predictions / num_examples  # 返回准确率


def calc_loss_batch(input_batch, target_batch, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)  # 将数据移动到设备
    logits = model(input_batch)[:, -1, :]  # 获取最后一个输出标记的 logits
    loss = torch.nn.functional.cross_entropy(logits, target_batch)  # 计算交叉熵损失
    return loss  # 返回损失


def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.  # 初始化总损失
    if len(data_loader) == 0:
        return float("nan")  # 如果数据加载器为空，返回 NaN
    elif num_batches is None:
        num_batches = len(data_loader)  # 如果没有指定批次数，则使用数据加载器的长度
    else:
        num_batches = min(num_batches, len(data_loader))  # 取最小值

    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)  # 计算损失
            total_loss += loss.item()  # 累加损失
        else:
            break
    return total_loss / num_batches  # 返回平均损失


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()  # 设置模型为评估模式
    with torch.no_grad():  # 不计算梯度
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)  # 计算训练损失
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)  # 计算验证损失
    model.train()  # 设置模型为训练模式
    return train_loss, val_loss  # 返回训练损失和验证损失


def train_classifier_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
                            eval_freq, eval_iter, tokenizer):
    # 初始化列表以跟踪损失和已见示例
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    examples_seen, global_step = 0, -1  # 初始化已见示例和全局步骤

    # 主训练循环
    for epoch in range(num_epochs):
        model.train()  # 设置模型为训练模式

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()  # 重置前一批次的损失梯度
            loss = calc_loss_batch(input_batch, target_batch, model, device)  # 计算损失
            loss.backward()  # 计算损失梯度
            optimizer.step()  # 使用损失梯度更新模型权重
            examples_seen += input_batch.shape[0]  # 跟踪已见示例数量
            global_step += 1  # 更新全局步骤

            # 可选评估步骤
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)  # 评估模型
                train_losses.append(train_loss)  # 记录训练损失
                val_losses.append(val_loss)  # 记录验证损失
                print(f"第 {epoch+1} 轮 (步骤 {global_step:06d}): "
                      f"训练损失 {train_loss:.3f}, 验证损失 {val_loss:.3f}")

        # 每个 epoch 结束后计算准确率
        train_accuracy = calc_accuracy_loader(train_loader, model, device, num_batches=eval_iter)  # 计算训练准确率
        val_accuracy = calc_accuracy_loader(val_loader, model, device, num_batches=eval_iter)  # 计算验证准确率
        print(f"训练准确率: {train_accuracy*100:.2f}% | ", end="")
        print(f"验证准确率: {val_accuracy*100:.2f}%")
        train_accs.append(train_accuracy)  # 记录训练准确率
        val_accs.append(val_accuracy)  # 记录验证准确率

    return train_losses, val_losses, train_accs, val_accs, examples_seen  # 返回损失和准确率


def plot_values(epochs_seen, examples_seen, train_values, val_values, label="loss"):
    fig, ax1 = plt.subplots(figsize=(5, 3))  # 创建绘图对象

    # 绘制训练和验证损失与 epochs 的关系
    ax1.plot(epochs_seen, train_values, label=f"训练 {label}")
    ax1.plot(epochs_seen, val_values, linestyle="-.", label=f"验证 {label}")
    ax1.set_xlabel("轮次")  # 设置 x 轴标签
    ax1.set_ylabel(label.capitalize())  # 设置 y 轴标签
    ax1.legend()  # 显示图例

    # 创建第二个 x 轴以显示已见示例
    ax2 = ax1.twiny()  # 创建共享 y 轴的第二个 x 轴
    ax2.plot(examples_seen, train_values, alpha=0)  # 不可见的绘图以对齐刻度
    ax2.set_xlabel("已见示例")  # 设置第二个 x 轴标签

    fig.tight_layout()  # 调整布局以留出空间
    plt.savefig(f"{label}-plot.pdf")  # 保存绘图为 PDF 文件
    # plt.show()  # 显示绘图（可选）


if __name__ == "__main__":  # 如果该文件是主程序

    import argparse  # 导入 argparse 库以处理命令行参数

    parser = argparse.ArgumentParser(
        description="对 GPT 模型进行微调以进行分类"
    )
    parser.add_argument(
        "--test_mode",
        default=False,
        action="store_true",
        help=("此标志在测试模式下运行模型以进行内部测试。 "
              "否则，它将按章节中使用的方式运行模型（推荐）。")
    )
    args = parser.parse_args()  # 解析命令行参数

    ########################################
    # 下载和准备数据集
    ########################################

    url = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"  # 数据集的 URL
    zip_path = "sms_spam_collection.zip"  # ZIP 文件路径
    extracted_path = "sms_spam_collection"  # 解压路径
    data_file_path = Path(extracted_path) / "SMSSpamCollection.tsv"  # 数据文件路径

    download_and_unzip_spam_data(url, zip_path, extracted_path, data_file_path, test_mode=args.test_mode)  # 下载并解压数据
    df = pd.read_csv(data_file_path, sep="\t", header=None, names=["Label", "Text"])  # 读取数据文件
    balanced_df = create_balanced_dataset(df)  # 创建平衡数据集
    balanced_df["Label"] = balanced_df["Label"].map({"ham": 0, "spam": 1})  # 将标签映射为数字

    train_df, validation_df, test_df = random_split(balanced_df, 0.7, 0.1)  # 随机分割数据集
    train_df.to_csv("train.csv", index=None)  # 保存训练集
    validation_df.to_csv("validation.csv", index=None)  # 保存验证集
    test_df.to_csv("test.csv", index=None)  # 保存测试集

    ########################################
    # 创建数据加载器
    ########################################
    tokenizer = tiktoken.get_encoding("gpt2")  # 获取 GPT2 的编码器

    train_dataset = SpamDataset(
        csv_file="train.csv",  # 训练集文件
        max_length=None,  # 最大长度
        tokenizer=tokenizer  # 编码器
    )

    val_dataset = SpamDataset(
        csv_file="validation.csv",  # 验证集文件
        max_length=train_dataset.max_length,  # 最大长度与训练集相同
        tokenizer=tokenizer  # 编码器
    )

    test_dataset = SpamDataset(
        csv_file="test.csv",  # 测试集文件
        max_length=train_dataset.max_length,  # 最大长度与训练集相同
        tokenizer=tokenizer  # 编码器
    )

    num_workers = 0  # 工作线程数
    batch_size = 8  # 批次大小

    torch.manual_seed(123)  # 设置随机种子

    train_loader = DataLoader(
        dataset=train_dataset,  # 训练集
        batch_size=batch_size,  # 批次大小
        shuffle=True,  # 打乱数据
        num_workers=num_workers,  # 工作线程数
        drop_last=True,  # 如果批次不完整则丢弃
    )

    val_loader = DataLoader(
        dataset=val_dataset,  # 验证集
        batch_size=batch_size,  # 批次大小
        num_workers=num_workers,  # 工作线程数
        drop_last=False,  # 不丢弃最后一个批次
    )

    test_loader = DataLoader(
        dataset=test_dataset,  # 测试集
        batch_size=batch_size,  # 批次大小
        num_workers=num_workers,  # 工作线程数
        drop_last=False,  # 不丢弃最后一个批次
    )

    ########################################
    # 加载预训练模型
    ########################################

    # 用于测试目的的小型 GPT 模型
    if args.test_mode:
        BASE_CONFIG = {
            "vocab_size": 50257,  # 词汇表大小
            "context_length": 120,  # 上下文长度
            "drop_rate": 0.0,  # 丢弃率
            "qkv_bias": False,  # 查询-键-值偏置
            "emb_dim": 12,  # 嵌入维度
            "n_layers": 1,  # 层数
            "n_heads": 2  # 头数
        }
        model = GPTModel(BASE_CONFIG)  # 创建模型
        model.eval()  # 设置模型为评估模式
        device = "cpu"  # 使用 CPU

    # 按章节中使用的代码
    else:
        CHOOSE_MODEL = "gpt2-small (124M)"  # 选择的模型
        INPUT_PROMPT = "Every effort moves"  # 输入提示

        BASE_CONFIG = {
            "vocab_size": 50257,  # 词汇表大小
            "context_length": 1024,  # 上下文长度
            "drop_rate": 0.0,  # 丢弃率
            "qkv_bias": True  # 查询-键-值偏置
        }

        model_configs = {
            "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},  # 小型模型配置
            "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},  # 中型模型配置
            "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},  # 大型模型配置
            "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},  # 超大型模型配置
        }

        BASE_CONFIG.update(model_configs[CHOOSE_MODEL])  # 更新基础配置

        assert train_dataset.max_length <= BASE_CONFIG["context_length"], (  # 确保数据集长度不超过模型上下文长度
            f"数据集长度 {train_dataset.max_length} 超过模型的上下文 "
            f"长度 {BASE_CONFIG['context_length']}。请使用 "
            f"`max_length={BASE_CONFIG['context_length']}` 重新初始化数据集"
        )

        model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")  # 获取模型大小
        settings, params = download_and_load_gpt2(model_size=model_size, models_dir="gpt2")  # 下载并加载 GPT2 模型

        model = GPTModel(BASE_CONFIG)  # 创建模型
        load_weights_into_gpt(model, params)  # 加载模型权重
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 使用 GPU 或 CPU

    ########################################
    # 修改和预训练模型
    ########################################

    for param in model.parameters():
        param.requires_grad = False  # 冻结所有参数

    torch.manual_seed(123)  # 设置随机种子

    num_classes = 2  # 类别数量
    model.out_head = torch.nn.Linear(in_features=BASE_CONFIG["emb_dim"], out_features=num_classes)  # 修改输出层
    model.to(device)  # 将模型移动到设备

    for param in model.trf_blocks[-1].parameters():
        param.requires_grad = True  # 解冻最后一层的参数

    for param in model.final_norm.parameters():
        param.requires_grad = True  # 解冻最终归一化层的参数

    ########################################
    # 微调修改后的模型
    ########################################

    start_time = time.time()  # 记录开始时间
    torch.manual_seed(123)  # 设置随机种子

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.1)  # 创建优化器

    num_epochs = 5  # 训练轮数
    train_losses, val_losses, train_accs, val_accs, examples_seen = train_classifier_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=num_epochs, eval_freq=50, eval_iter=5,
        tokenizer=tokenizer
    )

    end_time = time.time()  # 记录结束时间
    execution_time_minutes = (end_time - start_time) / 60  # 计算执行时间（分钟）
    print(f"训练完成，耗时 {execution_time_minutes:.2f} 分钟。")  # 打印训练完成信息

    ########################################
    # 绘制结果
    ########################################

    # 损失绘图
    epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))  # 创建轮次张量
    examples_seen_tensor = torch.linspace(0, examples_seen, len(train_losses))  # 创建已见示例张量
    plot_values(epochs_tensor, examples_seen_tensor, train_losses, val_losses)  # 绘制损失图

    # 准确率绘图
    epochs_tensor = torch.linspace(0, num_epochs, len(train_accs))  # 创建轮次张量
    examples_seen_tensor = torch.linspace(0, examples_seen, len(train_accs))  # 创建已见示例张量
    plot_values(epochs_tensor, examples_seen_tensor, train_accs, val_accs, label="accuracy")  # 绘制准确率图
