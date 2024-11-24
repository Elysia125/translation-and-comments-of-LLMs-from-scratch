# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch


# 导入所需的标准库
import os  # 用于文件和目录操作
import urllib.request  # 用于从网络下载文件

# 导入第三方库
# import requests  # 用于HTTP请求(这里被注释掉了)
import json  # 用于处理JSON数据
import numpy as np  # 用于数值计算
import tensorflow as tf  # 用于加载TensorFlow模型
from tqdm import tqdm  # 用于显示进度条


def download_and_load_gpt2(model_size, models_dir):
    """下载并加载GPT-2模型"""
    # 验证模型大小是否有效
    allowed_sizes = ("124M", "355M", "774M", "1558M")  # 定义允许的模型大小
    if model_size not in allowed_sizes:  # 检查输入的模型大小是否合法
        raise ValueError(f"Model size not in {allowed_sizes}")  # 如果不合法则抛出异常

    # 定义文件路径
    model_dir = os.path.join(models_dir, model_size)  # 构建模型保存目录路径
    base_url = "https://openaipublic.blob.core.windows.net/gpt-2/models"  # 定义基础URL
    filenames = [  # 需要下载的文件列表
        "checkpoint", "encoder.json", "hparams.json",
        "model.ckpt.data-00000-of-00001", "model.ckpt.index",
        "model.ckpt.meta", "vocab.bpe"
    ]

    # 下载文件
    os.makedirs(model_dir, exist_ok=True)  # 创建模型目录(如果不存在)
    for filename in filenames:  # 遍历需要下载的文件
        file_url = os.path.join(base_url, model_size, filename)  # 构建完整的文件URL
        file_path = os.path.join(model_dir, filename)  # 构建本地保存路径
        download_file(file_url, file_path)  # 下载文件

    # 加载设置和参数
    tf_ckpt_path = tf.train.latest_checkpoint(model_dir)  # 获取最新的检查点路径
    settings = json.load(open(os.path.join(model_dir, "hparams.json")))  # 加载模型超参数
    params = load_gpt2_params_from_tf_ckpt(tf_ckpt_path, settings)  # 加载模型参数

    return settings, params  # 返回设置和参数


def download_file(url, destination):
    """从指定URL下载文件到目标路径"""
    try:
        with urllib.request.urlopen(url) as response:  # 打开URL连接
            # 从响应头获取文件大小，如果不存在则默认为0
            file_size = int(response.headers.get("Content-Length", 0))

            # 检查文件是否已存在且大小相同
            if os.path.exists(destination):  # 如果目标文件已存在
                file_size_local = os.path.getsize(destination)  # 获取本地文件大小
                if file_size == file_size_local:  # 如果大小相同
                    print(f"File already exists and is up-to-date: {destination}")  # 打印提示信息
                    return  # 直接返回

            # 定义读取块大小
            block_size = 1024  # 1KB

            # 初始化进度条
            progress_bar_description = os.path.basename(url)  # 从URL中提取文件名
            with tqdm(total=file_size, unit="iB", unit_scale=True, desc=progress_bar_description) as progress_bar:
                # 以二进制写模式打开目标文件
                with open(destination, "wb") as file:
                    # 分块读取并写入文件
                    while True:  # 循环读取
                        chunk = response.read(block_size)  # 读取一个块
                        if not chunk:  # 如果没有更多数据
                            break  # 退出循环
                        file.write(chunk)  # 写入块数据
                        progress_bar.update(len(chunk))  # 更新进度条
    except urllib.error.HTTPError:  # 捕获HTTP错误
        s = (  # 构建错误信息
            f"The specified URL ({url}) is incorrect, the internet connection cannot be established,"
            "\nor the requested file is temporarily unavailable.\nPlease visit the following website"
            " for help: https://github.com/rasbt/LLMs-from-scratch/discussions/273")
        print(s)  # 打印错误信息


# 使用requests库的替代方案
"""
def download_file(url, destination):
    # Send a GET request to download the file in streaming mode
    response = requests.get(url, stream=True)

    # Get the total file size from headers, defaulting to 0 if not present
    file_size = int(response.headers.get("content-length", 0))

    # Check if file exists and has the same size
    if os.path.exists(destination):
        file_size_local = os.path.getsize(destination)
        if file_size == file_size_local:
            print(f"File already exists and is up-to-date: {destination}")
            return

    # Define the block size for reading the file
    block_size = 1024  # 1 Kilobyte

    # Initialize the progress bar with total file size
    progress_bar_description = url.split("/")[-1]  # Extract filename from URL
    with tqdm(total=file_size, unit="iB", unit_scale=True, desc=progress_bar_description) as progress_bar:
        # Open the destination file in binary write mode
        with open(destination, "wb") as file:
            # Iterate over the file data in chunks
            for chunk in response.iter_content(block_size):
                progress_bar.update(len(chunk))  # Update progress bar
                file.write(chunk)  # Write the chunk to the file
"""


def load_gpt2_params_from_tf_ckpt(ckpt_path, settings):
    """从TensorFlow检查点加载GPT-2参数"""
    # 初始化参数字典，为每一层创建空的块
    params = {"blocks": [{} for _ in range(settings["n_layer"])]}

    # 遍历检查点中的每个变量
    for name, _ in tf.train.list_variables(ckpt_path):  # 列出所有变量
        # 加载变量并移除单维度
        variable_array = np.squeeze(tf.train.load_variable(ckpt_path, name))

        # 处理变量名以提取相关部分
        variable_name_parts = name.split("/")[1:]  # 跳过'model/'前缀

        # 确定变量的目标字典
        target_dict = params  # 默认使用主参数字典
        if variable_name_parts[0].startswith("h"):  # 如果是层级变量
            layer_number = int(variable_name_parts[0][1:])  # 提取层号
            target_dict = params["blocks"][layer_number]  # 使用对应层的字典

        # 递归访问或创建嵌套字典
        for key in variable_name_parts[1:-1]:  # 遍历中间的键
            target_dict = target_dict.setdefault(key, {})  # 获取或创建嵌套字典

        # 将变量数组赋值给最后的键
        last_key = variable_name_parts[-1]  # 获取最后的键名
        target_dict[last_key] = variable_array  # 保存变量数组

    return params  # 返回加载的参数