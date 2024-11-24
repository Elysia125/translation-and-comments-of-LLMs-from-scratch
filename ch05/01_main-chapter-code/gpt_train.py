# 版权声明：由Sebastian Raschka根据Apache License 2.0许可发布(详见LICENSE.txt)
# 来源："从零开始构建大型语言模型"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# 代码: https://github.com/rasbt/LLMs-from-scratch

# 导入所需的Python库
import matplotlib.pyplot as plt  # 用于绘图
import os  # 用于操作系统相关功能
import torch  # PyTorch深度学习框架
import urllib.request  # 用于下载文件
import tiktoken  # 用于GPT-2分词

# 从本地文件导入自定义模块
from previous_chapters import GPTModel, create_dataloader_v1, generate_text_simple  # 导入之前章节定义的模型和函数


def text_to_token_ids(text, tokenizer):
    """将文本转换为token ID"""
    encoded = tokenizer.encode(text)  # 使用tokenizer编码文本
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # 添加batch维度
    return encoded_tensor


def token_ids_to_text(token_ids, tokenizer):
    """将token ID转换回文本"""
    flat = token_ids.squeeze(0)  # 移除batch维度
    return tokenizer.decode(flat.tolist())  # 解码为文本


def calc_loss_batch(input_batch, target_batch, model, device):
    """计算单个批次的损失"""
    # 将输入和目标数据移到指定设备
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    # 通过模型前向传播获取logits
    logits = model(input_batch)
    # 计算交叉熵损失
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    """计算数据加载器中所有批次的平均损失"""
    total_loss = 0.
    # 处理空数据加载器的情况
    if len(data_loader) == 0:
        return float("nan")
    # 确定要处理的批次数
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    # 遍历批次并累计损失
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    """评估模型在训练集和验证集上的表现"""
    model.eval()  # 设置为评估模式
    with torch.no_grad():  # 禁用梯度计算
        # 计算训练集和验证集损失
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()  # 恢复训练模式
    return train_loss, val_loss


def generate_and_print_sample(model, tokenizer, device, start_context):
    """生成并打印示例文本"""
    model.eval()  # 设置为评估模式
    # 获取上下文大小
    context_size = model.pos_emb.weight.shape[0]
    # 将起始文本转换为token ID并移到设备
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():  # 禁用梯度计算
        # 生成新的token
        token_ids = generate_text_simple(
            model=model, idx=encoded,
            max_new_tokens=50, context_size=context_size
        )
        # 将token解码为文本并打印
        decoded_text = token_ids_to_text(token_ids, tokenizer)
        print(decoded_text.replace("\n", " "))  # 压缩打印格式
    model.train()  # 恢复训练模式


def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
                       eval_freq, eval_iter, start_context, tokenizer):
    """简单的模型训练函数"""
    # 初始化跟踪列表
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen = 0
    global_step = -1

    # 主训练循环
    for epoch in range(num_epochs):
        model.train()  # 设置模型为训练模式

        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()  # 重置梯度
            # 计算损失
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()  # 计算梯度
            optimizer.step()  # 更新模型权重
            tokens_seen += input_batch.numel()  # 更新已处理的token数量
            global_step += 1

            # 定期评估
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                # 记录损失和token数
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                # 打印训练状态
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

        # 每个epoch结束后生成示例文本
        generate_and_print_sample(
            model, tokenizer, device, start_context
        )

    return train_losses, val_losses, track_tokens_seen


def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    """绘制训练和验证损失曲线"""
    fig, ax1 = plt.subplots()

    # 绘制训练和验证损失随epoch变化的曲线
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")

    # 创建第二个x轴显示处理的token数量
    ax2 = ax1.twiny()  # 创建共享y轴的第二个x轴
    ax2.plot(tokens_seen, train_losses, alpha=0)  # 绘制不可见的曲线以对齐刻度
    ax2.set_xlabel("Tokens seen")

    fig.tight_layout()  # 调整布局


def main(gpt_config, settings):
    """主函数，包含完整的训练流程"""
    # 设置随机种子和设备
    torch.manual_seed(123)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 下载训练数据
    file_path = "the-verdict.txt"
    url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt"

    # 如果文件不存在则下载
    if not os.path.exists(file_path):
        with urllib.request.urlopen(url) as response:
            text_data = response.read().decode('utf-8')
        with open(file_path, "w", encoding="utf-8") as file:
            file.write(text_data)
    else:
        # 如果文件存在则直接读取
        with open(file_path, "r", encoding="utf-8") as file:
            text_data = file.read()

    # 初始化模型
    model = GPTModel(gpt_config)
    model.to(device)  # 将模型移到指定设备
    # 初始化优化器
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=settings["learning_rate"], weight_decay=settings["weight_decay"]
    )

    # 设置数据加载器
    train_ratio = 0.90  # 训练集比例
    split_idx = int(train_ratio * len(text_data))  # 计算分割点

    # 创建训练数据加载器
    train_loader = create_dataloader_v1(
        text_data[:split_idx],
        batch_size=settings["batch_size"],
        max_length=gpt_config["context_length"],
        stride=gpt_config["context_length"],
        drop_last=True,
        shuffle=True,
        num_workers=0
    )

    # 创建验证数据加载器
    val_loader = create_dataloader_v1(
        text_data[split_idx:],
        batch_size=settings["batch_size"],
        max_length=gpt_config["context_length"],
        stride=gpt_config["context_length"],
        drop_last=False,
        shuffle=False,
        num_workers=0
    )

    # 初始化tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")

    # 训练模型
    train_losses, val_losses, tokens_seen = train_model_simple(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=settings["num_epochs"], eval_freq=5, eval_iter=1,
        start_context="Every effort moves you", tokenizer=tokenizer
    )

    return train_losses, val_losses, tokens_seen, model


if __name__ == "__main__":
    # 定义GPT-124M模型配置
    GPT_CONFIG_124M = {
        "vocab_size": 50257,    # 词汇表大小
        "context_length": 256,  # 上下文长度(原始为1024)
        "emb_dim": 768,         # 嵌入维度
        "n_heads": 12,          # 注意力头数
        "n_layers": 12,         # 层数
        "drop_rate": 0.1,       # 丢弃率
        "qkv_bias": False       # 是否使用QKV偏置
    }

    # 定义其他训练设置
    OTHER_SETTINGS = {
        "learning_rate": 5e-4,  # 学习率
        "num_epochs": 10,       # 训练轮数
        "batch_size": 2,        # 批次大小
        "weight_decay": 0.1     # 权重衰减
    }

    # 开始训练
    train_losses, val_losses, tokens_seen, model = main(GPT_CONFIG_124M, OTHER_SETTINGS)

    # 训练后的操作
    # 绘制损失曲线
    epochs_tensor = torch.linspace(0, OTHER_SETTINGS["num_epochs"], len(train_losses))
    plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)
    plt.savefig("loss.pdf")

    # 保存和加载模型
    torch.save(model.state_dict(), "model.pth")  # 保存模型状态
    model = GPTModel(GPT_CONFIG_124M)  # 创建新模型实例
    model.load_state_dict(torch.load("model.pth"), weights_only=True)  # 加载保存的状态
