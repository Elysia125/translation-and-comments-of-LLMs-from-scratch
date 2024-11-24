# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

# 导入所需的标准库
import os  # 用于文件和目录操作
import sys  # 用于系统相关操作
import tarfile  # 用于处理tar压缩文件
import time  # 用于时间相关操作
import urllib.request  # 用于从网络下载文件
import pandas as pd  # 用于数据处理和分析


def reporthook(count, block_size, total_size):
    """下载进度回调函数"""
    global start_time  # 声明全局变量start_time
    if count == 0:  # 如果是开始下载
        start_time = time.time()  # 记录开始时间
    else:
        duration = time.time() - start_time  # 计算已经过的时间
        progress_size = int(count * block_size)  # 计算已下载的大小
        percent = count * block_size * 100 / total_size  # 计算下载百分比

        # 计算下载速度(MB/s)
        speed = int(progress_size / (1024 * duration)) if duration else 0
        sys.stdout.write(  # 输出下载进度信息
            f"\r{int(percent)}% | {progress_size / (1024**2):.2f} MB "
            f"| {speed:.2f} MB/s | {duration:.2f} sec elapsed"
        )
        sys.stdout.flush()  # 刷新输出缓冲区


def download_and_extract_dataset(dataset_url, target_file, directory):
    """下载并解压数据集"""
    if not os.path.exists(directory):  # 如果目标目录不存在
        if os.path.exists(target_file):  # 如果目标文件已存在
            os.remove(target_file)  # 删除已存在的文件
        urllib.request.urlretrieve(dataset_url, target_file, reporthook)  # 下载数据集
        print("\nExtracting dataset ...")  # 提示开始解压
        with tarfile.open(target_file, "r:gz") as tar:  # 打开tar.gz文件
            tar.extractall()  # 解压所有文件
    else:
        print(f"Directory `{directory}` already exists. Skipping download.")  # 如果目录已存在，跳过下载


def load_dataset_to_dataframe(basepath="aclImdb", labels={"pos": 1, "neg": 0}):
    """将数据集加载到DataFrame中"""
    data_frames = []  # 用于存储DataFrame片段的列表
    for subset in ("test", "train"):  # 遍历测试集和训练集
        for label in ("pos", "neg"):  # 遍历正面和负面评论
            path = os.path.join(basepath, subset, label)  # 构建文件路径
            for file in sorted(os.listdir(path)):  # 遍历目录中的所有文件
                with open(os.path.join(path, file), "r", encoding="utf-8") as infile:  # 打开并读取文件
                    # 为每个文件创建DataFrame并添加到列表中
                    data_frames.append(pd.DataFrame({"text": [infile.read()], "label": [labels[label]]}))
    # 将所有DataFrame片段连接在一起
    df = pd.concat(data_frames, ignore_index=True)
    df = df.sample(frac=1, random_state=123).reset_index(drop=True)  # 打乱DataFrame并重置索引
    return df


def partition_and_save(df, sizes=(35000, 5000, 10000)):
    """划分数据集并保存为CSV文件"""
    # 打乱DataFrame
    df_shuffled = df.sample(frac=1, random_state=123).reset_index(drop=True)

    # 获取划分数据的索引位置
    train_end = sizes[0]  # 训练集结束位置
    val_end = sizes[0] + sizes[1]  # 验证集结束位置

    # 划分DataFrame
    train = df_shuffled.iloc[:train_end]  # 获取训练集
    val = df_shuffled.iloc[train_end:val_end]  # 获取验证集
    test = df_shuffled.iloc[val_end:]  # 获取测试集

    # 保存为CSV文件
    train.to_csv("train.csv", index=False)  # 保存训练集
    val.to_csv("validation.csv", index=False)  # 保存验证集
    test.to_csv("test.csv", index=False)  # 保存测试集


if __name__ == "__main__":
    dataset_url = "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"  # 数据集URL
    print("Downloading dataset ...")  # 提示开始下载
    download_and_extract_dataset(dataset_url, "aclImdb_v1.tar.gz", "aclImdb")  # 下载并解压数据集
    print("Creating data frames ...")  # 提示开始创建DataFrame
    df = load_dataset_to_dataframe()  # 加载数据到DataFrame
    print("Partitioning and saving data frames ...")  # 提示开始划分和保存
    partition_and_save(df)  # 划分并保存数据集
