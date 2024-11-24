# 版权声明 - 作者Sebastian Raschka，使用Apache License 2.0许可
# 来源于"从零开始构建大型语言模型"一书
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# 代码仓库: https://github.com/rasbt/LLMs-from-scratch

# 导入操作系统相关功能的模块
import os
# 导入用于URL请求的模块
import urllib.request

# 导入用于处理JSON数据的模块
import json
# 导入用于数值计算的NumPy库
import numpy as np
# 导入TensorFlow深度学习框架
import tensorflow as tf
# 导入进度条显示模块
from tqdm import tqdm


def download_and_load_gpt2(model_size, models_dir):
    # 验证模型大小是否合法
    allowed_sizes = ("124M", "355M", "774M", "1558M")
    if model_size not in allowed_sizes:
        raise ValueError(f"Model size not in {allowed_sizes}")

    # 定义文件路径
    model_dir = os.path.join(models_dir, model_size)  # 模型保存目录
    base_url = "https://openaipublic.blob.core.windows.net/gpt-2/models"  # GPT-2模型下载基础URL
    # 需要下载的文件列表
    filenames = [
        "checkpoint", "encoder.json", "hparams.json",
        "model.ckpt.data-00000-of-00001", "model.ckpt.index",
        "model.ckpt.meta", "vocab.bpe"
    ]

    # 下载所有必需文件
    os.makedirs(model_dir, exist_ok=True)  # 创建模型目录(如果不存在)
    for filename in filenames:
        file_url = os.path.join(base_url, model_size, filename)  # 构建完整的文件URL
        file_path = os.path.join(model_dir, filename)  # 构建本地保存路径
        download_file(file_url, file_path)  # 下载文件

    # 加载模型设置和参数
    tf_ckpt_path = tf.train.latest_checkpoint(model_dir)  # 获取最新的检查点文件路径
    settings = json.load(open(os.path.join(model_dir, "hparams.json")))  # 加载模型超参数
    params = load_gpt2_params_from_tf_ckpt(tf_ckpt_path, settings)  # 从检查点加载模型参数

    return settings, params  # 返回设置和参数


def download_file(url, destination):
    # 使用urllib下载文件

    try:
        with urllib.request.urlopen(url) as response:  # 打开URL连接
            # 从响应头获取文件大小，如果不存在则默认为0
            file_size = int(response.headers.get("Content-Length", 0))

            # 检查文件是否已存在且大小相同
            if os.path.exists(destination):
                file_size_local = os.path.getsize(destination)
                if file_size == file_size_local:
                    print(f"文件已存在且是最新的: {destination}")
                    return

            # 定义读取文件的块大小
            block_size = 1024  # 1KB

            # 使用文件名初始化进度条
            progress_bar_description = os.path.basename(url)  # 从URL提取文件名
            with tqdm(total=file_size, unit="iB", unit_scale=True, desc=progress_bar_description) as progress_bar:
                # 以二进制写模式打开目标文件
                with open(destination, "wb") as file:
                    # 分块读取并写入文件
                    while True:
                        chunk = response.read(block_size)  # 读取数据块
                        if not chunk:  # 如果没有更多数据则退出
                            break
                        file.write(chunk)  # 写入数据块
                        progress_bar.update(len(chunk))  # 更新进度条
    except urllib.error.HTTPError:
        s = (
            f"指定的URL ({url}) 不正确，无法建立网络连接，"
            "\n或请求的文件暂时不可用。\n请访问以下网站获取帮助："
            " https://github.com/rasbt/LLMs-from-scratch/discussions/273")
        print(s)


def load_gpt2_params_from_tf_ckpt(ckpt_path, settings):
    # 初始化参数字典，为每一层创建空的块
    params = {"blocks": [{} for _ in range(settings["n_layer"])]}

    # 遍历检查点中的每个变量
    for name, _ in tf.train.list_variables(ckpt_path):
        # 加载变量并移除单维度
        variable_array = np.squeeze(tf.train.load_variable(ckpt_path, name))

        # 处理变量名以提取相关部分
        variable_name_parts = name.split("/")[1:]  # 跳过'model/'前缀

        # 确定变量的目标字典
        target_dict = params
        if variable_name_parts[0].startswith("h"):
            layer_number = int(variable_name_parts[0][1:])  # 提取层号
            target_dict = params["blocks"][layer_number]  # 获取对应层的字典

        # 递归访问或创建嵌套字典
        for key in variable_name_parts[1:-1]:
            target_dict = target_dict.setdefault(key, {})

        # 将变量数组分配给最后一个键
        last_key = variable_name_parts[-1]
        target_dict[last_key] = variable_array

    return params  # 返回加载的参数
