# 版权声明 - Sebastian Raschka 基于 Apache License 2.0 (见 LICENSE.txt)
# 来源:"从零开始构建大型语言模型"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# 代码: https://github.com/rasbt/LLMs-from-scratch
#
# 本文件收集了第2-5章中所涉及的所有相关代码
# 本文件可以作为独立脚本运行

# 导入必要的numpy库
import numpy as np
# 导入tiktoken分词器库
import tiktoken
# 导入PyTorch主库
import torch
# 导入PyTorch神经网络模块
import torch.nn as nn
# 导入PyTorch数据集和数据加载器
from torch.utils.data import Dataset, DataLoader

#####################################
# 第2章
#####################################


# 定义GPT数据集类,继承自PyTorch的Dataset
class GPTDatasetV1(Dataset):
    # 初始化函数,接收文本、分词器、最大长度和步长参数
    def __init__(self, txt, tokenizer, max_length, stride):
        # 初始化输入ID列表
        self.input_ids = []
        # 初始化目标ID列表
        self.target_ids = []

        # 使用分词器对整个文本进行编码
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        # 使用滑动窗口将文本分成重叠的序列
        for i in range(0, len(token_ids) - max_length, stride):
            # 获取输入块
            input_chunk = token_ids[i:i + max_length]
            # 获取目标块(向后偏移一位)
            target_chunk = token_ids[i + 1: i + max_length + 1]
            # 将输入块转换为张量并添加到列表
            self.input_ids.append(torch.tensor(input_chunk))
            # 将目标块转换为张量并添加到列表
            self.target_ids.append(torch.tensor(target_chunk))

    # 返回数据集长度
    def __len__(self):
        return len(self.input_ids)

    # 获取指定索引的数据项
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


# 创建数据加载器的函数
def create_dataloader_v1(txt, batch_size=4, max_length=256,
                         stride=128, shuffle=True, drop_last=True, num_workers=0):
    # 初始化GPT-2分词器
    tokenizer = tiktoken.get_encoding("gpt2")

    # 创建数据集实例
    dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

    # 创建并返回数据加载器
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, num_workers=num_workers)

    return dataloader


#####################################
# 第3章
#####################################
# 多头注意力机制类
class MultiHeadAttention(nn.Module):
    # 初始化函数
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        # 确保输出维度能被注意力头数整除
        assert d_out % num_heads == 0, "d_out must be divisible by n_heads"

        # 设置类属性
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads  # 计算每个注意力头的维度

        # 创建查询、键、值的线性变换层
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        # 创建输出投影层
        self.out_proj = nn.Linear(d_out, d_out)
        # 创建dropout层
        self.dropout = nn.Dropout(dropout)
        # 注册因果掩码
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))

    # 前向传播函数
    def forward(self, x):
        # 获取输入张量的维度
        b, num_tokens, d_in = x.shape

        # 计算键、查询、值矩阵
        keys = self.W_key(x)  # 形状: (b, num_tokens, d_out)
        queries = self.W_query(x)
        values = self.W_value(x)

        # 重塑张量以添加注意力头维度
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # 调整维度顺序
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # 计算注意力分数
        attn_scores = queries @ keys.transpose(2, 3)

        # 截取掩码并转换为布尔类型
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        # 应用掩码到注意力分数
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        # 计算注意力权重
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 计算上下文向量
        context_vec = (attn_weights @ values).transpose(1, 2)

        # 合并所有注意力头
        context_vec = context_vec.reshape(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)

        return context_vec


#####################################
# 第4章
#####################################
# 层归一化类
class LayerNorm(nn.Module):
    # 初始化函数
    def __init__(self, emb_dim):
        super().__init__()
        # 设置epsilon值防止除零
        self.eps = 1e-5
        # 创建可学习的缩放参数
        self.scale = nn.Parameter(torch.ones(emb_dim))
        # 创建可学习的偏移参数
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    # 前向传播函数
    def forward(self, x):
        # 计算均值
        mean = x.mean(dim=-1, keepdim=True)
        # 计算方差
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        # 归一化
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        # 应用缩放和偏移
        return self.scale * norm_x + self.shift


# GELU激活函数类
class GELU(nn.Module):
    # 初始化函数
    def __init__(self):
        super().__init__()

    # 前向传播函数
    def forward(self, x):
        # 实现GELU激活函数
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))


# 前馈神经网络类
class FeedForward(nn.Module):
    # 初始化函数
    def __init__(self, cfg):
        super().__init__()
        # 创建前馈网络层序列
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    # 前向传播函数
    def forward(self, x):
        return self.layers(x)


# Transformer块类
class TransformerBlock(nn.Module):
    # 初始化函数
    def __init__(self, cfg):
        super().__init__()
        # 创建多头注意力层
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"])
        # 创建前馈网络层
        self.ff = FeedForward(cfg)
        # 创建两个层归一化层
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        # 创建残差连接的dropout层
        self.drop_resid = nn.Dropout(cfg["drop_rate"])

    # 前向传播函数
    def forward(self, x):
        # 注意力块的残差连接
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_resid(x)
        x = x + shortcut

        # 前馈网络块的残差连接
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_resid(x)
        x = x + shortcut

        return x


# GPT模型类
class GPTModel(nn.Module):
    # 初始化函数
    def __init__(self, cfg):
        super().__init__()
        # 创建词嵌入层
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        # 创建位置嵌入层
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        # 创建嵌入dropout层
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        # 创建Transformer块序列
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])

        # 创建最终的层归一化
        self.final_norm = LayerNorm(cfg["emb_dim"])
        # 创建输出头
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    # 前向传播函数
    def forward(self, in_idx):
        # 获取批次大小和序列长度
        batch_size, seq_len = in_idx.shape
        # 获取词嵌入
        tok_embeds = self.tok_emb(in_idx)
        # 获取位置嵌入
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        # 组合词嵌入和位置嵌入
        x = tok_embeds + pos_embeds
        # 应用dropout
        x = self.drop_emb(x)
        # 通过Transformer块
        x = self.trf_blocks(x)
        # 应用最终的层归一化
        x = self.final_norm(x)
        # 计算logits
        logits = self.out_head(x)
        return logits


# 简单文本生成函数
def generate_text_simple(model, idx, max_new_tokens, context_size):
    # 循环生成新的token
    for _ in range(max_new_tokens):
        # 截取上下文窗口
        idx_cond = idx[:, -context_size:]

        # 获取预测结果
        with torch.no_grad():
            logits = model(idx_cond)

        # 只关注最后一个时间步
        logits = logits[:, -1, :]

        # 获取最高概率的token索引
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)

        # 将新生成的token添加到序列中
        idx = torch.cat((idx, idx_next), dim=1)

    return idx


#####################################
# 第5章
#####################################
# 参数赋值函数
def assign(left, right):
    # 检查形状是否匹配
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
    return torch.nn.Parameter(torch.tensor(right))


# 将预训练权重加载到GPT模型中的函数
def load_weights_into_gpt(gpt, params):
    # 加载位置嵌入权重
    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params['wpe'])
    # 加载词嵌入权重
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params['wte'])

    # 遍历所有块加载权重
    for b in range(len(params["blocks"])):
        # 分割注意力权重
        q_w, k_w, v_w = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1)
        # 加载查询、键、值的权重
        gpt.trf_blocks[b].att.W_query.weight = assign(
            gpt.trf_blocks[b].att.W_query.weight, q_w.T)
        gpt.trf_blocks[b].att.W_key.weight = assign(
            gpt.trf_blocks[b].att.W_key.weight, k_w.T)
        gpt.trf_blocks[b].att.W_value.weight = assign(
            gpt.trf_blocks[b].att.W_value.weight, v_w.T)

        # 分割注意力偏置
        q_b, k_b, v_b = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1)
        # 加载查询、键、值的偏置
        gpt.trf_blocks[b].att.W_query.bias = assign(
            gpt.trf_blocks[b].att.W_query.bias, q_b)
        gpt.trf_blocks[b].att.W_key.bias = assign(
            gpt.trf_blocks[b].att.W_key.bias, k_b)
        gpt.trf_blocks[b].att.W_value.bias = assign(
            gpt.trf_blocks[b].att.W_value.bias, v_b)

        # 加载输出投影层的权重和偏置
        gpt.trf_blocks[b].att.out_proj.weight = assign(
            gpt.trf_blocks[b].att.out_proj.weight,
            params["blocks"][b]["attn"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].att.out_proj.bias = assign(
            gpt.trf_blocks[b].att.out_proj.bias,
            params["blocks"][b]["attn"]["c_proj"]["b"])

        # 加载前馈网络的权重和偏置
        gpt.trf_blocks[b].ff.layers[0].weight = assign(
            gpt.trf_blocks[b].ff.layers[0].weight,
            params["blocks"][b]["mlp"]["c_fc"]["w"].T)
        gpt.trf_blocks[b].ff.layers[0].bias = assign(
            gpt.trf_blocks[b].ff.layers[0].bias,
            params["blocks"][b]["mlp"]["c_fc"]["b"])
        gpt.trf_blocks[b].ff.layers[2].weight = assign(
            gpt.trf_blocks[b].ff.layers[2].weight,
            params["blocks"][b]["mlp"]["c_proj"]["w"].T)
        gpt.trf_blocks[b].ff.layers[2].bias = assign(
            gpt.trf_blocks[b].ff.layers[2].bias,
            params["blocks"][b]["mlp"]["c_proj"]["b"])

        # 加载层归一化的参数
        gpt.trf_blocks[b].norm1.scale = assign(
            gpt.trf_blocks[b].norm1.scale,
            params["blocks"][b]["ln_1"]["g"])
        gpt.trf_blocks[b].norm1.shift = assign(
            gpt.trf_blocks[b].norm1.shift,
            params["blocks"][b]["ln_1"]["b"])
        gpt.trf_blocks[b].norm2.scale = assign(
            gpt.trf_blocks[b].norm2.scale,
            params["blocks"][b]["ln_2"]["g"])
        gpt.trf_blocks[b].norm2.shift = assign(
            gpt.trf_blocks[b].norm2.shift,
            params["blocks"][b]["ln_2"]["b"])

    # 加载最终层归一化的参数
    gpt.final_norm.scale = assign(gpt.final_norm.scale, params["g"])
    gpt.final_norm.shift = assign(gpt.final_norm.shift, params["b"])
    # 加载输出头的权重
    gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"])


# 文本转换为token ID的函数
def text_to_token_ids(text, tokenizer):
    # 编码文本
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    # 转换为张量并添加批次维度
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor


# token ID转换为文本的函数
def token_ids_to_text(token_ids, tokenizer):
    # 移除批次维度
    flat = token_ids.squeeze(0)
    # 解码为文本
    return tokenizer.decode(flat.tolist())
