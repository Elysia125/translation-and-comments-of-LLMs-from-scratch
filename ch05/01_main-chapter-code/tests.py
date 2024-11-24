# 版权声明：由Sebastian Raschka根据Apache License 2.0许可发布(详见LICENSE.txt)
# 来源："从零开始构建大型语言模型"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# 代码: https://github.com/rasbt/LLMs-from-scratch

# 内部使用的单元测试文件

# 导入所需的Python库
import pytest  # 用于单元测试
from gpt_train import main  # 导入训练主函数
import http.client  # 用于HTTP连接
from urllib.parse import urlparse  # 用于解析URL


@pytest.fixture  # pytest装饰器,用于定义可重用的测试数据
def gpt_config():
    # 返回GPT模型的配置参数
    return {
        "vocab_size": 50257,  # 词汇表大小
        "context_length": 12,  # 上下文长度(为了测试效率设置较小)
        "emb_dim": 32,        # 嵌入维度(为了测试效率设置较小)
        "n_heads": 4,         # 注意力头数(为了测试效率设置较小)
        "n_layers": 2,        # 层数(为了测试效率设置较小)
        "drop_rate": 0.1,     # dropout比率
        "qkv_bias": False     # 是否使用QKV偏置
    }


@pytest.fixture  # pytest装饰器,用于定义可重用的测试数据
def other_settings():
    # 返回其他训练相关的设置参数
    return {
        "learning_rate": 5e-4,  # 学习率
        "num_epochs": 1,        # 训练轮数(为了测试效率设置较小)
        "batch_size": 2,        # 批次大小
        "weight_decay": 0.1     # 权重衰减
    }


def test_main(gpt_config, other_settings):
    # 测试主训练函数
    train_losses, val_losses, tokens_seen, model = main(gpt_config, other_settings)

    # 验证训练过程中记录的数据点数量是否符合预期
    assert len(train_losses) == 39, "训练损失数量不符合预期"
    assert len(val_losses) == 39, "验证损失数量不符合预期"
    assert len(tokens_seen) == 39, "处理的token数量不符合预期"


def check_file_size(url, expected_size):
    # 检查给定URL的文件大小是否符合预期
    parsed_url = urlparse(url)  # 解析URL
    if parsed_url.scheme == "https":  # 根据协议选择连接类型
        conn = http.client.HTTPSConnection(parsed_url.netloc)
    else:
        conn = http.client.HTTPConnection(parsed_url.netloc)

    conn.request("HEAD", parsed_url.path)  # 发送HEAD请求
    response = conn.getresponse()  # 获取响应
    if response.status != 200:  # 检查响应状态
        return False, f"{url} 无法访问"
    size = response.getheader("Content-Length")  # 获取文件大小
    if size is None:
        return False, "Content-Length头部信息缺失"
    size = int(size)
    if size != expected_size:  # 验证文件大小
        return False, f"{url} 文件预期大小为 {expected_size}, 但实际为 {size}"
    return True, f"{url} 文件大小正确"


def test_model_files():
    # 测试模型文件的可用性和大小
    base_url = "https://openaipublic.blob.core.windows.net/gpt-2/models"

    # 测试124M模型的文件
    model_size = "124M"
    files = {
        "checkpoint": 77,
        "encoder.json": 1042301,
        "hparams.json": 90,
        "model.ckpt.data-00000-of-00001": 497759232,
        "model.ckpt.index": 5215,
        "model.ckpt.meta": 471155,
        "vocab.bpe": 456318
    }

    # 验证每个文件的大小
    for file_name, expected_size in files.items():
        url = f"{base_url}/{model_size}/{file_name}"
        valid, message = check_file_size(url, expected_size)
        assert valid, message

    # 测试355M模型的文件
    model_size = "355M"
    files = {
        "checkpoint": 77,
        "encoder.json": 1042301,
        "hparams.json": 91,
        "model.ckpt.data-00000-of-00001": 1419292672,
        "model.ckpt.index": 10399,
        "model.ckpt.meta": 926519,
        "vocab.bpe": 456318
    }

    # 验证每个文件的大小
    for file_name, expected_size in files.items():
        url = f"{base_url}/{model_size}/{file_name}"
        valid, message = check_file_size(url, expected_size)
        assert valid, message
