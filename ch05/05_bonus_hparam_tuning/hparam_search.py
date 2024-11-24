# 版权声明 - 本代码基于 Apache License 2.0 许可证 (见 LICENSE.txt)
# 来源于《从零开始构建大型语言模型》一书
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# 代码仓库: https://github.com/rasbt/LLMs-from-scratch

# 导入所需的库
import itertools  # 用于生成超参数组合
import math  # 用于数学计算
import os  # 用于文件和路径操作
import tiktoken  # OpenAI的分词器
import torch  # PyTorch深度学习框架
from previous_chapters import GPTModel, create_dataloader_v1  # 导入之前章节定义的模型和数据加载器


# 定义要搜索的超参数网格
HPARAM_GRID = {
    "batch_size": [2, 4, 8, 16],  # 批次大小选项
    "drop_rate": [0.0, 0.1, 0.2],  # Dropout率选项
    "warmup_iters": [10, 20, 30],  # 预热迭代次数选项
    "weight_decay": [0.1, 0.01, 0.0],  # 权重衰减选项
    "peak_lr": [0.0001, 0.0005, 0.001, 0.005],  # 峰值学习率选项
    "initial_lr": [0.00005, 0.0001],  # 初始学习率选项
    "min_lr": [0.00005, 0.00001, 0.0001],  # 最小学习率选项
    "n_epochs": [5, 10, 15, 20, 25],  # 训练轮数选项
}


def calc_loss_loader(data_loader, model, device, num_batches=None):
    """计算数据加载器中所有批次的平均损失"""
    total_loss = 0.  # 总损失初始化为0
    if len(data_loader) == 0:  # 如果数据加载器为空
        return float("nan")  # 返回NaN
    elif num_batches is None:  # 如果未指定批次数
        num_batches = len(data_loader)  # 使用所有批次
    else:
        num_batches = min(num_batches, len(data_loader))  # 取指定批次数和总批次数的较小值
    for i, (input_batch, target_batch) in enumerate(data_loader):  # 遍历批次
        if i < num_batches:  # 如果未达到指定批次数
            loss = calc_loss_batch(input_batch, target_batch, model, device)  # 计算当前批次损失
            total_loss += loss.item()  # 累加损失值
        else:
            break  # 达到指定批次数后退出
    return total_loss / num_batches  # 返回平均损失


def calc_loss_batch(input_batch, target_batch, model, device):
    """计算单个批次的损失"""
    # 将输入和目标数据移到指定设备
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)

    logits = model(input_batch)  # 前向传播得到预测结果
    logits = logits.view(-1, logits.size(-1))  # 重塑logits维度
    # 计算交叉熵损失
    loss = torch.nn.functional.cross_entropy(logits, target_batch.view(-1))
    return loss


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    """评估模型在训练集和验证集上的性能"""
    model.eval()  # 设置为评估模式
    with torch.no_grad():  # 不计算梯度
        # 计算训练集和验证集损失
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()  # 恢复训练模式
    return train_loss, val_loss


def train_model(model, train_loader, val_loader, optimizer, device,
                n_epochs, eval_freq, eval_iter,
                encoded_start_context, tokenizer, warmup_iters=10,
                initial_lr=3e-05, min_lr=1e-6):
    """训练模型的主函数"""
    global_step = 0  # 全局步数计数器

    max_lr = optimizer.param_groups[0]["lr"]  # 获取最大学习率

    # 计算总训练迭代次数
    total_training_iters = len(train_loader) * n_epochs

    # 计算预热阶段每步学习率增量
    lr_increment = (optimizer.param_groups[0]["lr"] - initial_lr) / warmup_iters

    # 开始训练循环
    for epoch in range(n_epochs):  # 遍历每个训练轮次
        model.train()  # 设置为训练模式
        for input_batch, target_batch in train_loader:  # 遍历数据批次
            optimizer.zero_grad()  # 清零梯度

            global_step += 1  # 更新全局步数

            # 预热阶段：线性增加学习率
            if global_step <= warmup_iters:
                lr = initial_lr + global_step * lr_increment
            # 余弦退火阶段
            else:
                progress = (global_step - warmup_iters) / (total_training_iters - warmup_iters)
                lr = min_lr + (max_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))

            # 更新优化器的学习率
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            # 计算损失并反向传播
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()

            # 在预热后应用梯度裁剪
            if global_step >= warmup_iters:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()  # 更新模型参数

    # 训练结束后评估模型
    train_loss, val_loss = evaluate_model(model, train_loader, val_loader, device, eval_iter)

    return train_loss, val_loss


if __name__ == "__main__":

    # 生成所有超参数组合
    hyperparameter_combinations = list(itertools.product(*HPARAM_GRID.values()))
    total_combinations = len(hyperparameter_combinations)
    print(f"Total hyperparameter configurations: {total_combinations}")

    # 初始化最佳损失和最佳超参数
    best_val_loss = float('inf')
    best_hparams = {}

    # 获取脚本路径并读取训练数据
    script_path = os.path.abspath(__file__)
    script_dir = os.path.dirname(script_path)
    with open(os.path.join(script_dir, "the-verdict.txt"), "r", encoding="utf-8") as file:
        text_data = file.read()

    # 初始化分词器和设备
    tokenizer = tiktoken.get_encoding("gpt2")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 设置训练集和验证集比例
    train_ratio = 0.95
    split_idx = int(train_ratio * len(text_data))

    # 设置随机种子
    torch.manual_seed(123)

    # 超参数搜索主循环
    interrupted = False
    current_config = 0
    for combination in hyperparameter_combinations:

        try:
            current_config += 1
            print(f"Evaluating configuration {current_config} of {total_combinations}")

            # 解包当前超参数组合
            HPARAM_CONFIG = dict(zip(HPARAM_GRID.keys(), combination))

            # 定义模型配置
            GPT_CONFIG_124M = {
                "vocab_size": 50257,    # 词汇表大小
                "context_length": 256,   # 上下文长度
                "emb_dim": 768,         # 嵌入维度
                "n_heads": 12,          # 注意力头数
                "n_layers": 12,         # 层数
                "drop_rate": HPARAM_CONFIG["drop_rate"],  # Dropout率
                "qkv_bias": False,      # 是否使用QKV偏置
            }

            # 设置随机种子
            torch.manual_seed(123)
            
            # 创建训练数据加载器
            train_loader = create_dataloader_v1(
                text_data[:split_idx],
                batch_size=HPARAM_CONFIG["batch_size"],
                max_length=GPT_CONFIG_124M["context_length"],
                stride=GPT_CONFIG_124M["context_length"],
                drop_last=True,
                shuffle=True,
                num_workers=0
            )

            # 创建验证数据加载器
            val_loader = create_dataloader_v1(
                text_data[split_idx:],
                batch_size=HPARAM_CONFIG["batch_size"],
                max_length=GPT_CONFIG_124M["context_length"],
                stride=GPT_CONFIG_124M["context_length"],
                drop_last=False,
                shuffle=False,
                num_workers=0
            )

            # 初始化模型
            model = GPTModel(GPT_CONFIG_124M)
            model.to(device)

            # 初始化优化器
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=HPARAM_CONFIG["peak_lr"],
                weight_decay=HPARAM_CONFIG["weight_decay"]
            )

            # 准备起始上下文
            encoded_start_context = tokenizer.encode("Nevertheless")
            encoded_tensor = torch.tensor(encoded_start_context).unsqueeze(0)

            # 训练模型
            train_loss, val_loss = train_model(
                model, train_loader, val_loader, optimizer, device,
                n_epochs=HPARAM_CONFIG["n_epochs"],
                eval_freq=5, eval_iter=1,
                encoded_start_context=encoded_tensor,
                tokenizer=tokenizer,
                warmup_iters=HPARAM_CONFIG["warmup_iters"],
                initial_lr=HPARAM_CONFIG["initial_lr"],
                min_lr=HPARAM_CONFIG["min_lr"]
            )

            # 更新最佳超参数
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_train_loss = train_loss
                best_hparams = HPARAM_CONFIG

        except KeyboardInterrupt:
            # 处理键盘中断
            print("Hyperparameter search completed.")
            print(f"Best hyperparameters: {best_hparams}")
            print(f"Best Val loss: {best_val_loss} | Training loss {train_loss}")
            interrupted = True
            break

    # 打印最终结果
    if not interrupted:
        print("Hyperparameter search completed.")
        print(f"Best hyperparameters: {best_hparams}")
        print(f"Best Val loss: {best_val_loss} | Training loss {train_loss}")
