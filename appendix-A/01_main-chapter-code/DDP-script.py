# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://github.com/rasbt/LLMs-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

# Appendix A: Introduction to PyTorch (Part 3)

# 导入PyTorch基础包
import torch
# 导入PyTorch函数式接口
import torch.nn.functional as F
# 导入数据集和数据加载器相关类
from torch.utils.data import Dataset, DataLoader

# NEW imports:
# 导入操作系统相关功能
import os
# 导入平台信息相关功能
import platform
# 导入PyTorch多进程处理模块
import torch.multiprocessing as mp
# 导入分布式采样器
from torch.utils.data.distributed import DistributedSampler
# 导入分布式数据并行模块
from torch.nn.parallel import DistributedDataParallel as DDP
# 导入分布式进程组管理函数
from torch.distributed import init_process_group, destroy_process_group


# NEW: function to initialize a distributed process group (1 process / GPU)
# this allows communication among processes
def ddp_setup(rank, world_size):
    """
    Arguments:
        rank: a unique process ID
        world_size: total number of processes in the group
    """
    # rank of machine running rank:0 process
    # here, we assume all GPUs are on the same machine
    # 设置主进程地址为本地主机
    os.environ["MASTER_ADDR"] = "localhost"
    # any free port on the machine
    # 设置主进程端口号
    os.environ["MASTER_PORT"] = "12345"
    # 如果是Windows系统
    if platform.system() == "Windows":
        # Disable libuv because PyTorch for Windows isn't built with support
        # 禁用libuv
        os.environ["USE_LIBUV"] = "0"

    # initialize process group
    if platform.system() == "Windows":
        # Windows users may have to use "gloo" instead of "nccl" as backend
        # gloo: Facebook Collective Communication Library
        # Windows系统使用gloo后端初始化进程组
        init_process_group(backend="gloo", rank=rank, world_size=world_size)
    else:
        # nccl: NVIDIA Collective Communication Library
        # 其他系统使用nccl后端初始化进程组
        init_process_group(backend="nccl", rank=rank, world_size=world_size)

    # 设置当前进程使用的CUDA设备
    torch.cuda.set_device(rank)


# 定义玩具数据集类
class ToyDataset(Dataset):
    def __init__(self, X, y):
        # 初始化特征和标签
        self.features = X
        self.labels = y

    def __getitem__(self, index):
        # 获取单个样本的特征和标签
        one_x = self.features[index]
        one_y = self.labels[index]
        return one_x, one_y

    def __len__(self):
        # 返回数据集大小
        return self.labels.shape[0]


# 定义神经网络类
class NeuralNetwork(torch.nn.Module):
    def __init__(self, num_inputs, num_outputs):
        # 调用父类初始化
        super().__init__()

        # 定义网络层结构
        self.layers = torch.nn.Sequential(
            # 1st hidden layer
            # 第一个隐藏层，输入层到30个神经元
            torch.nn.Linear(num_inputs, 30),
            torch.nn.ReLU(),

            # 2nd hidden layer
            # 第二个隐藏层，30个神经元到20个神经元
            torch.nn.Linear(30, 20),
            torch.nn.ReLU(),

            # output layer
            # 输出层，20个神经元到输出维度
            torch.nn.Linear(20, num_outputs),
        )

    def forward(self, x):
        # 前向传播函数
        logits = self.layers(x)
        return logits


# 准备数据集函数
def prepare_dataset():
    # 创建训练数据
    X_train = torch.tensor([
        [-1.2, 3.1],
        [-0.9, 2.9],
        [-0.5, 2.6],
        [2.3, -1.1],
        [2.7, -1.5]
    ])
    y_train = torch.tensor([0, 0, 0, 1, 1])

    # 创建测试数据
    X_test = torch.tensor([
        [-0.8, 2.8],
        [2.6, -1.6],
    ])
    y_test = torch.tensor([0, 1])

    # 创建训练和测试数据集
    train_ds = ToyDataset(X_train, y_train)
    test_ds = ToyDataset(X_test, y_test)

    # 创建训练数据加载器
    train_loader = DataLoader(
        dataset=train_ds,
        batch_size=2,
        shuffle=False,  # NEW: False because of DistributedSampler below
        pin_memory=True,
        drop_last=True,
        # NEW: chunk batches across GPUs without overlapping samples:
        sampler=DistributedSampler(train_ds)  # NEW
    )
    # 创建测试数据加载器
    test_loader = DataLoader(
        dataset=test_ds,
        batch_size=2,
        shuffle=False,
    )
    return train_loader, test_loader


# NEW: wrapper
# 主训练函数
def main(rank, world_size, num_epochs):

    ddp_setup(rank, world_size)  # NEW: initialize process groups

    # 准备数据集
    train_loader, test_loader = prepare_dataset()
    # 创建模型
    model = NeuralNetwork(num_inputs=2, num_outputs=2)
    # 将模型移到对应GPU
    model.to(rank)
    # 创建优化器
    optimizer = torch.optim.SGD(model.parameters(), lr=0.5)

    # 将模型包装为DDP模型
    model = DDP(model, device_ids=[rank])  # NEW: wrap model with DDP
    # the core model is now accessible as model.module

    # 训练循环
    for epoch in range(num_epochs):
        # NEW: Set sampler to ensure each epoch has a different shuffle order
        # 设置采样器的epoch，确保每个epoch有不同的打乱顺序
        train_loader.sampler.set_epoch(epoch)

        # 设置为训练模式
        model.train()
        for features, labels in train_loader:

            # 将数据移到对应GPU
            features, labels = features.to(rank), labels.to(rank)  # New: use rank
            # 前向传播
            logits = model(features)
            # 计算损失
            loss = F.cross_entropy(logits, labels)  # Loss function

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # LOGGING
            # 打印训练信息
            print(f"[GPU{rank}] Epoch: {epoch+1:03d}/{num_epochs:03d}"
                  f" | Batchsize {labels.shape[0]:03d}"
                  f" | Train/Val Loss: {loss:.2f}")

    # 评估模式
    model.eval()
    # 计算训练集准确率
    train_acc = compute_accuracy(model, train_loader, device=rank)
    print(f"[GPU{rank}] Training accuracy", train_acc)
    # 计算测试集准确率
    test_acc = compute_accuracy(model, test_loader, device=rank)
    print(f"[GPU{rank}] Test accuracy", test_acc)

    # 清理进程组
    destroy_process_group()  # NEW: cleanly exit distributed mode


# 计算准确率函数
def compute_accuracy(model, dataloader, device):
    # 设置为评估模式
    model = model.eval()
    correct = 0.0
    total_examples = 0

    # 遍历数据集
    for idx, (features, labels) in enumerate(dataloader):
        # 将数据移到对应设备
        features, labels = features.to(device), labels.to(device)

        # 不计算梯度
        with torch.no_grad():
            logits = model(features)
        # 获取预测结果
        predictions = torch.argmax(logits, dim=1)
        # 比较预测值和真实值
        compare = labels == predictions
        # 累加正确预测数
        correct += torch.sum(compare)
        # 累加总样本数
        total_examples += len(compare)
    # 返回准确率
    return (correct / total_examples).item()


# 主程序入口
if __name__ == "__main__":
    # 打印PyTorch版本
    print("PyTorch version:", torch.__version__)
    # 打印CUDA是否可用
    print("CUDA available:", torch.cuda.is_available())
    # 打印可用GPU数量
    print("Number of GPUs available:", torch.cuda.device_count())

    # 设置随机种子
    torch.manual_seed(123)

    # NEW: spawn new processes
    # note that spawn will automatically pass the rank
    # 设置训练轮数
    num_epochs = 3
    # 获取GPU数量作为进程数
    world_size = torch.cuda.device_count()
    # 启动多进程训练
    mp.spawn(main, args=(world_size, num_epochs), nprocs=world_size)
    # nprocs=world_size spawns one process per GPU
