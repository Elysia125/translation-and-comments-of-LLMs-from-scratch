# 版权声明 - 本代码基于 Apache License 2.0 许可证 (见 LICENSE.txt)
# 来源于《从零开始构建大型语言模型》一书
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# 代码仓库: https://github.com/rasbt/LLMs-from-scratch

# 本文件用于内部使用(单元测试)

# 导入所需的Python标准库
import io  # 用于文件I/O操作
import os  # 用于操作系统相关功能
import sys  # 用于系统相关功能
import types  # 用于动态创建模块
import nbformat  # 用于处理Jupyter notebook文件
from typing import Optional, Tuple  # 用于类型提示
import torch  # PyTorch深度学习框架
import pytest  # 测试框架
from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding, apply_rotary_pos_emb  # 导入Hugging Face的Llama模型组件


# 以下代码来自LitGPT项目: https://github.com/Lightning-AI/litgpt/blob/main/litgpt/model.py
# LitGPT使用Apache v2许可证: https://github.com/Lightning-AI/litgpt/blob/main/LICENSE
def litgpt_build_rope_cache(
    seq_len: int,  # 序列长度
    n_elem: int,  # 元素数量(头部维度)
    device: Optional[torch.device] = None,  # 设备(CPU/GPU)
    base: int = 10000,  # 计算逆频率的基数
    condense_ratio: int = 1,  # 位置索引压缩比
    extra_config: Optional[dict] = None,  # Llama 3.1和3.2使用的额外配置参数
) -> Tuple[torch.Tensor, torch.Tensor]:  # 返回余弦和正弦缓存
    """
    增强型Transformer旋转位置编码(RoPE)。

    参数:
        seq_len (int): 序列长度
        n_elem (int): 元素数量(头部维度)
        device (torch.device, optional): 张量分配的设备
        base (int, optional): 计算逆频率的基数
        condense_ratio (int, optional): 位置索引压缩比
        extra_config (dict, optional): Llama 3.1和3.2使用的频率调整配置参数

    返回:
        Tuple[torch.Tensor, torch.Tensor]: RoPE的余弦和正弦缓存
    """

    # 计算逆频率theta
    theta = 1.0 / (base ** (torch.arange(0, n_elem, 2, device=device).float() / n_elem))

    # 如果提供了额外配置,进行频率调整
    if extra_config is not None:
        orig_context_len = extra_config["original_max_seq_len"]  # 原始最大序列长度
        factor = extra_config["factor"]  # 缩放因子
        low_freq_factor = extra_config["low_freq_factor"]  # 低频因子
        high_freq_factor = extra_config["high_freq_factor"]  # 高频因子

        # 计算波长和比率
        wavelen = 2 * torch.pi / theta
        ratio = orig_context_len / wavelen
        
        # 计算平滑因子
        smooth_factor = (ratio - low_freq_factor) / (high_freq_factor - low_freq_factor)
        smooth_factor = torch.clamp(smooth_factor, min=0.0, max=1.0)  # 限制在[0,1]范围内

        # 计算调整后的theta
        adjusted_theta = (1 - smooth_factor) * (theta / factor) + smooth_factor * theta
        theta = adjusted_theta

    # 创建位置索引[0, 1, ..., seq_len - 1]
    seq_idx = torch.arange(seq_len, device=device) / condense_ratio

    # 计算位置索引和theta_i的外积
    idx_theta = torch.outer(seq_idx, theta).repeat(1, 2)

    # 返回余弦和正弦值
    return torch.cos(idx_theta), torch.sin(idx_theta)


# LitGPT代码来自: https://github.com/Lightning-AI/litgpt/blob/main/litgpt/model.py
# LitGPT使用Apache v2许可证: https://github.com/Lightning-AI/litgpt/blob/main/LICENSE
def litgpt_apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    应用旋转位置编码(RoPE)到输入张量。
    
    参数:
        x: 输入张量
        cos: 余弦缓存
        sin: 正弦缓存
    返回:
        应用RoPE后的张量
    """
    head_size = x.size(-1)  # 获取头部大小
    x1 = x[..., : head_size // 2]  # 前半部分 (B, nh, T, hs/2)
    x2 = x[..., head_size // 2:]  # 后半部分 (B, nh, T, hs/2)
    rotated = torch.cat((-x2, x1), dim=-1)  # 旋转拼接 (B, nh, T, hs)
    
    # 如果cos维度>1,需要调整batch维度
    if cos.dim() > 1:
        # sin/cos是(B, T, hs),需要在-3维度上增加维度以匹配nh
        cos = cos.unsqueeze(-3)
        sin = sin.unsqueeze(-3)

    # 应用旋转
    roped = (x * cos) + (rotated * sin)
    return roped.to(dtype=x.dtype)


@pytest.fixture(scope="module")
def notebook():
    """
    用于从Jupyter notebook导入定义的fixture
    """
    def import_definitions_from_notebook(notebooks):
        """从notebook导入定义的辅助函数"""
        imported_modules = {}

        for fullname, names in notebooks.items():
            # 获取当前测试文件的目录
            current_dir = os.path.dirname(__file__)
            path = os.path.join(current_dir, "..", fullname + ".ipynb")
            path = os.path.normpath(path)

            # 加载notebook
            if not os.path.exists(path):
                raise FileNotFoundError(f"Notebook file not found at: {path}")

            # 以UTF-8编码打开并读取notebook
            with io.open(path, "r", encoding="utf-8") as f:
                nb = nbformat.read(f, as_version=4)

            # 创建模块来存储导入的函数和类
            mod = types.ModuleType(fullname)
            sys.modules[fullname] = mod

            # 遍历notebook单元格,只执行函数或类定义
            for cell in nb.cells:
                if cell.cell_type == "code":
                    cell_code = cell.source
                    for name in names:
                        # 检查函数或类定义
                        if f"def {name}" in cell_code or f"class {name}" in cell_code:
                            exec(cell_code, mod.__dict__)

            imported_modules[fullname] = mod

        return imported_modules

    # 定义需要导入的notebook和函数/类
    notebooks = {
        "converting-gpt-to-llama2": ["SiLU", "RMSNorm", "precompute_rope_params", "compute_rope"],
        "converting-llama2-to-llama3": ["precompute_rope_params"]
    }

    return import_definitions_from_notebook(notebooks)


@pytest.fixture(autouse=True)
def set_seed():
    """设置随机种子以确保测试结果可重现"""
    torch.manual_seed(123)


def test_rope_llama2(notebook):
    """测试Llama2的RoPE实现"""

    # 获取当前notebook模块
    this_nb = notebook["converting-gpt-to-llama2"]

    # 设置测试参数
    batch_size = 1  # 批次大小
    context_len = 4096  # 上下文长度
    num_heads = 4  # 注意力头数
    head_dim = 16  # 每个头的维度

    # 预计算RoPE参数
    cos, sin = this_nb.precompute_rope_params(head_dim=head_dim, context_length=context_len)

    # 创建随机查询和键张量
    queries = torch.randn(batch_size, num_heads, context_len, head_dim)
    keys = torch.randn(batch_size, num_heads, context_len, head_dim)

    # 应用旋转位置编码
    queries_rot = this_nb.compute_rope(queries, cos, sin)
    keys_rot = this_nb.compute_rope(keys, cos, sin)

    # 使用Hugging Face的实现生成参考RoPE
    rot_emb = LlamaRotaryEmbedding(
        dim=head_dim,
        max_position_embeddings=context_len,
        base=10_000
    )
    position_ids = torch.arange(context_len, dtype=torch.long).unsqueeze(0)
    ref_cos, ref_sin = rot_emb(queries, position_ids)
    ref_queries_rot, ref_keys_rot = apply_rotary_pos_emb(queries, keys, ref_cos, ref_sin)
    
    # 验证结果与参考实现一致
    torch.testing.assert_close(sin, ref_sin.squeeze(0))
    torch.testing.assert_close(cos, ref_cos.squeeze(0))
    torch.testing.assert_close(keys_rot, ref_keys_rot)
    torch.testing.assert_close(queries_rot, ref_queries_rot)

    # 使用LitGPT的实现生成参考RoPE
    litgpt_cos, litgpt_sin = litgpt_build_rope_cache(context_len, n_elem=head_dim, base=10_000)
    litgpt_queries_rot = litgpt_apply_rope(queries, litgpt_cos, litgpt_sin)
    litgpt_keys_rot = litgpt_apply_rope(keys, litgpt_cos, litgpt_sin)

    # 验证结果与LitGPT实现一致
    torch.testing.assert_close(sin, litgpt_sin)
    torch.testing.assert_close(cos, litgpt_cos)
    torch.testing.assert_close(keys_rot, litgpt_keys_rot)
    torch.testing.assert_close(queries_rot, litgpt_queries_rot)


def test_rope_llama3(notebook):
    """测试Llama3的RoPE实现"""

    # 获取notebook模块
    nb1 = notebook["converting-gpt-to-llama2"]
    nb2 = notebook["converting-llama2-to-llama3"]

    # 设置测试参数
    batch_size = 1
    context_len = 8192
    num_heads = 4
    head_dim = 16
    theta_base = 500_000

    # 预计算RoPE参数
    cos, sin = nb2.precompute_rope_params(
        head_dim=head_dim,
        context_length=context_len,
        theta_base=theta_base
    )

    # 创建随机查询和键张量
    torch.manual_seed(123)
    queries = torch.randn(batch_size, num_heads, context_len, head_dim)
    keys = torch.randn(batch_size, num_heads, context_len, head_dim)

    # 应用旋转位置编码
    queries_rot = nb1.compute_rope(queries, cos, sin)
    keys_rot = nb1.compute_rope(keys, cos, sin)

    # 使用Hugging Face的实现生成参考RoPE
    rot_emb = LlamaRotaryEmbedding(
        dim=head_dim,
        max_position_embeddings=context_len,
        base=theta_base
    )

    position_ids = torch.arange(context_len, dtype=torch.long).unsqueeze(0)
    ref_cos, ref_sin = rot_emb(queries, position_ids)
    ref_queries_rot, ref_keys_rot = apply_rotary_pos_emb(queries, keys, ref_cos, ref_sin)

    # 验证结果与参考实现一致
    torch.testing.assert_close(sin, ref_sin.squeeze(0))
    torch.testing.assert_close(cos, ref_cos.squeeze(0))
    torch.testing.assert_close(keys_rot, ref_keys_rot)
    torch.testing.assert_close(queries_rot, ref_queries_rot)

    # 使用LitGPT的实现生成参考RoPE
    litgpt_cos, litgpt_sin = litgpt_build_rope_cache(context_len, n_elem=head_dim, base=theta_base)
    litgpt_queries_rot = litgpt_apply_rope(queries, litgpt_cos, litgpt_sin)
    litgpt_keys_rot = litgpt_apply_rope(keys, litgpt_cos, litgpt_sin)

    # 验证结果与LitGPT实现一致
    torch.testing.assert_close(sin, litgpt_sin)
    torch.testing.assert_close(cos, litgpt_cos)
    torch.testing.assert_close(keys_rot, litgpt_keys_rot)
    torch.testing.assert_close(queries_rot, litgpt_queries_rot)


def test_rope_llama3_12(notebook):
    """测试Llama3-12的RoPE实现"""

    # 获取notebook模块
    nb1 = notebook["converting-gpt-to-llama2"]
    nb2 = notebook["converting-llama2-to-llama3"]

    # 设置测试参数
    batch_size = 1
    context_len = 8192
    num_heads = 4
    head_dim = 16
    rope_theta = 500_000

    # RoPE配置
    rope_config = {
        "factor": 8.0,
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
        "original_context_length": 8192,
    }

    # 预计算RoPE参数
    cos, sin = nb2.precompute_rope_params(
        head_dim=head_dim,
        theta_base=rope_theta,
        context_length=context_len,
        freq_config=rope_config,
    )

    # 创建随机查询和键张量
    torch.manual_seed(123)
    queries = torch.randn(batch_size, num_heads, context_len, head_dim)
    keys = torch.randn(batch_size, num_heads, context_len, head_dim)

    # 应用旋转位置编码
    queries_rot = nb1.compute_rope(queries, cos, sin)
    keys_rot = nb1.compute_rope(keys, cos, sin)

    # Hugging Face RoPE参数配置
    hf_rope_params = {
        "factor": 8.0,
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
        "original_max_position_embeddings": 8192,
        "rope_type": "llama3"
    }

    # 创建RoPE配置类
    class RoPEConfig:
        rope_type = "llama3"
        rope_scaling = hf_rope_params
        factor = 1.0
        dim: int = head_dim
        rope_theta = 500_000
        max_position_embeddings: int = 8192
        hidden_size = head_dim * num_heads
        num_attention_heads = num_heads

    config = RoPEConfig()

    # 使用Hugging Face的实现生成参考RoPE
    rot_emb = LlamaRotaryEmbedding(config=config)
    position_ids = torch.arange(context_len, dtype=torch.long).unsqueeze(0)
    ref_cos, ref_sin = rot_emb(queries, position_ids)
    ref_queries_rot, ref_keys_rot = apply_rotary_pos_emb(queries, keys, ref_cos, ref_sin)

    # 验证结果与参考实现一致
    torch.testing.assert_close(sin, ref_sin.squeeze(0))
    torch.testing.assert_close(cos, ref_cos.squeeze(0))
    torch.testing.assert_close(keys_rot, ref_keys_rot)
    torch.testing.assert_close(queries_rot, ref_queries_rot)

    # LitGPT RoPE配置
    litgpt_rope_config = {
        "factor": 8.0,
        "low_freq_factor": 1.0,
        "high_freq_factor": 4.0,
        "original_max_seq_len": 8192
    }

    # 使用LitGPT的实现生成参考RoPE
    litgpt_cos, litgpt_sin = litgpt_build_rope_cache(
        context_len,
        n_elem=head_dim,
        base=rope_theta,
        extra_config=litgpt_rope_config
    )
    litgpt_queries_rot = litgpt_apply_rope(queries, litgpt_cos, litgpt_sin)
    litgpt_keys_rot = litgpt_apply_rope(keys, litgpt_cos, litgpt_sin)

    # 验证结果与LitGPT实现一致
    torch.testing.assert_close(sin, litgpt_sin)
    torch.testing.assert_close(cos, litgpt_cos)
    torch.testing.assert_close(keys_rot, litgpt_keys_rot)
    torch.testing.assert_close(queries_rot, litgpt_queries_rot)


def test_silu(notebook):
    """测试SiLU激活函数"""
    example_batch = torch.randn(2, 3, 4)  # 创建示例批次
    silu = notebook["converting-gpt-to-llama2"].SiLU()  # 实例化SiLU
    # 验证结果与PyTorch内置实现一致
    assert torch.allclose(silu(example_batch), torch.nn.functional.silu(example_batch))


@pytest.mark.skipif(torch.__version__ < "2.4", reason="Requires PyTorch 2.4 or newer")
def test_rmsnorm(notebook):
    """测试RMSNorm层归一化"""
    example_batch = torch.randn(2, 3, 4)  # 创建示例批次
    # 实例化RMSNorm
    rms_norm = notebook["converting-gpt-to-llama2"].RMSNorm(emb_dim=example_batch.shape[-1], eps=1e-5)
    # 实例化PyTorch的RMSNorm
    rmsnorm_pytorch = torch.nn.RMSNorm(example_batch.shape[-1], eps=1e-5)

    # 验证结果与PyTorch实现一致
    assert torch.allclose(rms_norm(example_batch), rmsnorm_pytorch(example_batch))
