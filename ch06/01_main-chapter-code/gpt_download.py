# 版权所有 (c) Sebastian Raschka，依据 Apache License 2.0（见 LICENSE.txt）。
# 来源于《从头构建大型语言模型》
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# 代码: https://github.com/rasbt/LLMs-from-scratch


import os  # 导入操作系统模块
import urllib.request  # 导入用于处理 URL 请求的库

# import requests  # 导入请求库（已注释）
import json  # 导入 JSON 处理库
import numpy as np  # 导入 NumPy 库
import tensorflow as tf  # 导入 TensorFlow 库
from tqdm import tqdm  # 导入进度条库


def download_and_load_gpt2(model_size, models_dir):  # 定义下载和加载 GPT2 的函数
    # 验证模型大小
    allowed_sizes = ("124M", "355M", "774M", "1558M")  # 允许的模型大小
    if model_size not in allowed_sizes:  # 如果模型大小不在允许的范围内
        raise ValueError(f"Model size not in {allowed_sizes}")  # 抛出错误

    # 定义路径
    model_dir = os.path.join(models_dir, model_size)  # 模型目录
    base_url = "https://openaipublic.blob.core.windows.net/gpt-2/models"  # 基础 URL
    filenames = [  # 文件名列表
        "checkpoint", "encoder.json", "hparams.json",
        "model.ckpt.data-00000-of-00001", "model.ckpt.index",
        "model.ckpt.meta", "vocab.bpe"
    ]

    # 下载文件
    os.makedirs(model_dir, exist_ok=True)  # 创建模型目录
    for filename in filenames:  # 遍历文件名
        file_url = os.path.join(base_url, model_size, filename)  # 文件 URL
        file_path = os.path.join(model_dir, filename)  # 文件路径
        download_file(file_url, file_path)  # 下载文件

    # 加载设置和参数
    tf_ckpt_path = tf.train.latest_checkpoint(model_dir)  # 获取最新的检查点路径
    settings = json.load(open(os.path.join(model_dir, "hparams.json")))  # 加载设置
    params = load_gpt2_params_from_tf_ckpt(tf_ckpt_path, settings)  # 加载参数

    return settings, params  # 返回设置和参数


def download_file(url, destination):  # 定义下载文件的函数
    # 发送 GET 请求以下载文件

    try:
        with urllib.request.urlopen(url) as response:  # 打开 URL
            # 从头部获取文件总大小，如果不存在则默认为 0
            file_size = int(response.headers.get("Content-Length", 0))

            # 检查文件是否存在且大小相同
            if os.path.exists(destination):  # 如果文件已存在
                file_size_local = os.path.getsize(destination)  # 获取本地文件大小
                if file_size == file_size_local:  # 如果文件大小相同
                    print(f"File already exists and is up-to-date: {destination}")  # 打印信息
                    return  # 返回

            # 定义读取文件的块大小
            block_size = 1024  # 1 Kilobyte

            # 使用总文件大小初始化进度条
            progress_bar_description = os.path.basename(url)  # 从 URL 提取文件名
            with tqdm(total=file_size, unit="iB", unit_scale=True, desc=progress_bar_description) as progress_bar:  # 初始化进度条
                # 以二进制写模式打开目标文件
                with open(destination, "wb") as file:
                    # 以块的方式读取文件并写入目标文件
                    while True:
                        chunk = response.read(block_size)  # 读取块
                        if not chunk:  # 如果没有更多数据
                            break  # 退出循环
                        file.write(chunk)  # 写入块
                        progress_bar.update(len(chunk))  # 更新进度条
    except urllib.error.HTTPError:  # 捕获 HTTP 错误
        s = (  # 错误信息
            f"The specified URL ({url}) is incorrect, the internet connection cannot be established,"
            "\nor the requested file is temporarily unavailable.\nPlease visit the following website"
            " for help: https://github.com/rasbt/LLMs-from-scratch/discussions/273")
        print(s)  # 打印错误信息


# 使用 `requests` 的替代方法
"""
def download_file(url, destination):  # 定义下载文件的函数
    # 发送 GET 请求以流式下载文件
    response = requests.get(url, stream=True)  # 发送请求

    # 从头部获取文件总大小，如果不存在则默认为 0
    file_size = int(response.headers.get("content-length", 0))

    # 检查文件是否存在且大小相同
    if os.path.exists(destination):  # 如果文件已存在
        file_size_local = os.path.getsize(destination)  # 获取本地文件大小
        if file_size == file_size_local:  # 如果文件大小相同
            print(f"File already exists and is up-to-date: {destination}")  # 打印信息
            return  # 返回

    # 定义读取文件的块大小
    block_size = 1024  # 1 Kilobyte

    # 使用总文件大小初始化进度条
    progress_bar_description = url.split("/")[-1]  # 从 URL 提取文件名
    with tqdm(total=file_size, unit="iB", unit_scale=True, desc=progress_bar_description) as progress_bar:  # 初始化进度条
        # 以二进制写模式打开目标文件
        with open(destination, "wb") as file:
            # 以块的方式读取文件数据
            for chunk in response.iter_content(block_size):  # 遍历块
                progress_bar.update(len(chunk))  # 更新进度条
                file.write(chunk)  # 写入块到文件
"""


def load_gpt2_params_from_tf_ckpt(ckpt_path, settings):  # 定义从 TensorFlow 检查点加载 GPT2 参数的函数
    # 初始化参数字典，为每一层创建空块
    params = {"blocks": [{} for _ in range(settings["n_layer"])]}  # 参数字典

    # 遍历检查点中的每个变量
    for name, _ in tf.train.list_variables(ckpt_path):  # 列出变量
        # 加载变量并去除单例维度
        variable_array = np.squeeze(tf.train.load_variable(ckpt_path, name))  # 加载变量

        # 处理变量名以提取相关部分
        variable_name_parts = name.split("/")[1:]  # 跳过 'model/' 前缀

        # 确定变量的目标字典
        target_dict = params  # 目标字典
        if variable_name_parts[0].startswith("h"):  # 如果变量名以 'h' 开头
            layer_number = int(variable_name_parts[0][1:])  # 获取层号
            target_dict = params["blocks"][layer_number]  # 更新目标字典

        # 递归访问或创建嵌套字典
        for key in variable_name_parts[1:-1]:  # 遍历变量名的中间部分
            target_dict = target_dict.setdefault(key, {})  # 设置默认值

        # 将变量数组分配给最后一个键
        last_key = variable_name_parts[-1]  # 获取最后一个键
        target_dict[last_key] = variable_array  # 赋值

    return params  # 返回参数
