# 版权声明 - 本代码基于 Apache License 2.0 许可证 (见 LICENSE.txt)
# 来源于《从零开始构建大型语言模型》一书
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# 代码仓库: https://github.com/rasbt/LLMs-from-scratch

"""
在古腾堡计划的书籍上预训练一个小型GPT-2模型(124M参数)的脚本。

在运行此脚本之前,请确保您已按照README.md中的说明
下载并处理了数据集。
"""

# 导入所需的库
import argparse  # 用于解析命令行参数
import os  # 用于文件和目录操作
from pathlib import Path  # 用于路径操作
import time  # 用于时间相关操作
import tiktoken  # OpenAI的分词器
import torch  # PyTorch深度学习框架

# 从previous_chapters.py导入所需的函数和类
from previous_chapters import (
    create_dataloader_v1,  # 创建数据加载器
    GPTModel,  # GPT模型类
    generate_and_print_sample,  # 生成并打印样本文本
    calc_loss_batch,  # 计算批次损失
    evaluate_model,  # 评估模型
    plot_losses  # 绘制损失曲线
)


def read_text_file(file_path):
    """读取文本文件的函数"""
    with open(file_path, "r", encoding="utf-8") as file:
        text_data = file.read()
    return text_data


def create_dataloaders(text_data, train_ratio, batch_size, max_length, stride, num_workers=0):
    """创建训练和验证数据加载器的函数"""
    # 根据train_ratio计算分割点
    split_idx = int(train_ratio * len(text_data))
    
    # 创建训练数据加载器
    train_loader = create_dataloader_v1(
        text_data[:split_idx],  # 训练数据
        batch_size=batch_size,  # 批次大小
        max_length=max_length,  # 最大序列长度
        stride=stride,  # 滑动窗口步长
        drop_last=True,  # 丢弃最后不完整的批次
        shuffle=True,  # 随机打乱数据
        num_workers=num_workers  # 数据加载的工作进程数
    )
    
    # 创建验证数据加载器
    val_loader = create_dataloader_v1(
        text_data[split_idx:],  # 验证数据
        batch_size=batch_size,
        max_length=max_length,
        stride=stride,
        drop_last=False,  # 保留最后不完整的批次
        shuffle=False,  # 不打乱数据
        num_workers=num_workers
    )
    return train_loader, val_loader


def convert_time(seconds):
    """将秒数转换为小时、分钟和秒"""
    hours, rem = divmod(seconds, 3600)  # 计算小时数和余数
    minutes, seconds = divmod(rem, 60)  # 计算分钟数和秒数
    return int(hours), int(minutes), int(seconds)


def print_eta(start_time, book_start_time, index, total_files):
    """打印预计完成时间的函数"""
    book_end_time = time.time()  # 处理当前书籍的结束时间
    elapsed_time = book_end_time - book_start_time  # 处理当前书籍所用时间
    total_elapsed_time = book_end_time - start_time  # 总耗时
    books_remaining = total_files - index  # 剩余书籍数量
    average_time_per_book = total_elapsed_time / index  # 每本书平均耗时
    eta = average_time_per_book * books_remaining  # 预计剩余时间

    # 转换各个时间为小时、分钟、秒
    book_h, book_m, book_s = convert_time(elapsed_time)
    total_h, total_m, total_s = convert_time(total_elapsed_time)
    eta_h, eta_m, eta_s = convert_time(eta)

    # 打印时间信息
    print(f"Book processed {book_h}h {book_m}m {book_s}s"
          f"\nTotal time elapsed {total_h}h {total_m}m {total_s}s"
          f"\nETA for remaining books: {eta_h}h {eta_m}m {eta_s}s")


def train_model_simple(model, optimizer, device, n_epochs,
                       eval_freq, eval_iter, print_sample_iter, start_context,
                       output_dir, save_ckpt_freq, tokenizer,
                       batch_size=1024, train_ratio=0.90):
    """训练模型的主函数"""
    
    # 初始化跟踪变量
    train_losses, val_losses, track_tokens_seen = [], [], []  # 存储训练损失、验证损失和处理的token数
    tokens_seen = 0  # 已处理的token总数
    global_step = -1  # 全局步数计数器
    start_time = time.time()  # 训练开始时间

    try:
        # 训练循环
        for epoch in range(n_epochs):
            # 遍历训练语料库中的所有书籍
            for index, file_path in enumerate(all_files, 1):
                book_start_time = time.time()  # 开始处理当前书籍的时间
                text_data = read_text_file(file_path) + " <|endoftext|> "  # 读取文本并添加结束标记
                print(f"Tokenizing file {index} of {total_files}: {file_path}")

                # 为每本书创建新的数据加载器
                train_loader, val_loader = create_dataloaders(
                    text_data,
                    train_ratio=train_ratio,
                    batch_size=batch_size,
                    max_length=GPT_CONFIG_124M["context_length"],
                    stride=GPT_CONFIG_124M["context_length"],
                    num_workers=0
                )
                
                print("Training ...")
                model.train()  # 设置模型为训练模式
                
                # 批次训练循环
                for input_batch, target_batch in train_loader:
                    optimizer.zero_grad()  # 清零梯度
                    loss = calc_loss_batch(input_batch, target_batch, model, device)  # 计算损失
                    loss.backward()  # 反向传播
                    optimizer.step()  # 更新参数
                    tokens_seen += input_batch.numel()  # 更新已处理的token数
                    global_step += 1  # 更新全局步数

                    # 定期评估模型
                    if global_step % eval_freq == 0:
                        train_loss, val_loss = evaluate_model(
                            model, train_loader, val_loader, device, eval_iter)
                        train_losses.append(train_loss)
                        val_losses.append(val_loss)
                        track_tokens_seen.append(tokens_seen)
                        print(f"Ep {epoch+1} (Step {global_step}): "
                              f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

                    # 定期生成文本样本
                    if global_step % print_sample_iter == 0:
                        generate_and_print_sample(
                            model, tokenizer, device, start_context
                        )

                # 定期保存模型检查点
                if global_step % save_ckpt_freq:
                    file_name = output_dir / f"model_pg_{global_step}.pth"
                    torch.save(model.state_dict(), file_name)
                    print(f"Saved {file_name}")

                # 打印进度信息
                print_eta(start_time, book_start_time, index, total_files)

    except KeyboardInterrupt:
        # 处理键盘中断，保存模型
        file_name = output_dir / f"model_pg_{global_step}_interrupted.pth"
        torch.save(model.state_dict(), file_name)
        print(f"Saved {file_name}")

    return train_losses, val_losses, track_tokens_seen


if __name__ == "__main__":
    # 创建命令行参数解析器
    parser = argparse.ArgumentParser(description='GPT Model Training Configuration')

    # 添加命令行参数
    parser.add_argument('--data_dir', type=str, default='gutenberg/data',
                        help='Directory containing the training data')
    parser.add_argument('--output_dir', type=str, default='model_checkpoints',
                        help='Directory where the model checkpoints will be saved')
    parser.add_argument('--n_epochs', type=int, default=1,
                        help='Number of epochs to train the model')
    parser.add_argument('--print_sample_iter', type=int, default=1000,
                        help='Iterations between printing sample outputs')
    parser.add_argument('--eval_freq', type=int, default=100,
                        help='Frequency of evaluations during training')
    parser.add_argument('--save_ckpt_freq', type=int, default=100_000,
                        help='Frequency of saving model checkpoints during training')
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='Learning rate for the optimizer')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for training')
    parser.add_argument('--debug', type=bool, default=False,
                        help='Uses a very small model for debugging purposes')

    args = parser.parse_args()

    # 根据是否为调试模式设置模型配置
    if args.debug:
        # 调试模式使用小型模型配置
        GPT_CONFIG_124M = {
            "vocab_size": 50257,     # 词汇表大小
            "context_length": 10,    # 上下文长度
            "emb_dim": 12,           # 嵌入维度
            "n_heads": 2,            # 注意力头数
            "n_layers": 2,           # 层数
            "drop_rate": 0.0,        # Dropout率(已禁用)
            "qkv_bias": False        # 是否使用QKV偏置
        }
    else:
        # 正常模式使用完整模型配置
        GPT_CONFIG_124M = {
            "vocab_size": 50257,     # 词汇表大小
            "context_length": 1024,  # 上下文长度
            "emb_dim": 768,          # 嵌入维度
            "n_heads": 12,           # 注意力头数
            "n_layers": 12,          # 层数
            "drop_rate": 0.1,        # Dropout率
            "qkv_bias": False        # 是否使用QKV偏置
        }

    # 设置设备(GPU/CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(123)  # 设置随机种子
    
    # 初始化模型、优化器和分词器
    model = GPTModel(GPT_CONFIG_124M)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1)
    tokenizer = tiktoken.get_encoding("gpt2")

    # 获取训练数据文件列表
    data_dir = args.data_dir
    all_files = [os.path.join(path, name) for path, subdirs, files
                 in os.walk(data_dir) for name in files if name.endswith((".txt"))]
    total_files = len(all_files)

    # 检查是否找到训练文件
    if total_files == 0:
        print("No training text files found. Make sure you "
              "selected the correct input directory")
        quit()
    print("Total files:", total_files)

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 训练模型
    train_losses, val_losses, tokens_seen = train_model_simple(
        model, optimizer, device,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        eval_freq=args.eval_freq,
        eval_iter=1,
        print_sample_iter=args.print_sample_iter,
        output_dir=output_dir,
        save_ckpt_freq=args.save_ckpt_freq,
        start_context="Every effort moves you",
        tokenizer=tokenizer
    )

    # 绘制损失曲线
    epochs_tensor = torch.linspace(0, args.n_epochs, len(train_losses))
    plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses, output_dir)

    # 保存最终模型
    torch.save(model.state_dict(), output_dir / "model_pg_final.pth")
    print(f"Maximum GPU memory allocated: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
