# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

# This file collects all the relevant code that we covered thus far
# throughout Chapters 2-4.
# This file can be run as a standalone script.

# 导入必要的库
import tiktoken  # 用于文本分词
import torch    # PyTorch深度学习框架
import torch.nn as nn  # PyTorch神经网络模块
from torch.utils.data import Dataset, DataLoader  # 用于数据加载和处理
import matplotlib.pyplot as plt  # 用于绘图


#####################################
# Chapter 2
#####################################

class GPTDatasetV1(Dataset):
    # 初始化GPT数据集类,用于处理文本数据
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []  # 存储输入序列
        self.target_ids = []  # 存储目标序列

        # 使用tokenizer对整个文本进行编码
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        # 使用滑动窗口将文本分成重叠的序列
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]  # 获取输入序列
            target_chunk = token_ids[i + 1: i + max_length + 1]  # 获取目标序列(向后偏移一位)
            self.input_ids.append(torch.tensor(input_chunk))  # 转换为tensor并存储
            self.target_ids.append(torch.tensor(target_chunk))  # 转换为tensor并存储

    # 返回数据集长度
    def __len__(self):
        return len(self.input_ids)

    # 获取指定索引的数据项
    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


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
# Chapter 3
#####################################

class MultiHeadAttention(nn.Module):
    # 初始化多头注意力层
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by n_heads"

        self.d_out = d_out  # 输出维度
        self.num_heads = num_heads  # 注意力头数
        self.head_dim = d_out // num_heads  # 每个注意力头的维度

        # 定义查询、键、值的线性变换层
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)  # 输出投影层
        self.dropout = nn.Dropout(dropout)  # dropout层
        # 注册因果掩码
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        b, num_tokens, d_in = x.shape  # 获取输入维度

        # 计算查询、键、值矩阵
        keys = self.W_key(x)  # Shape: (b, num_tokens, d_out)
        queries = self.W_query(x)
        values = self.W_value(x)

        # 重塑张量以支持多头注意力
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # 调整维度顺序
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # 计算注意力分数
        attn_scores = queries @ keys.transpose(2, 3)  # 计算点积注意力

        # 准备掩码
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        # 应用掩码
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        # 计算注意力权重
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 计算上下文向量
        context_vec = (attn_weights @ values).transpose(1, 2)

        # 合并多个注意力头的结果
        context_vec = context_vec.reshape(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)  # 应用输出投影

        return context_vec


#####################################
# Chapter 4
#####################################

class LayerNorm(nn.Module):
    # 初始化层归一化
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5  # 防止除零的小值
        self.scale = nn.Parameter(torch.ones(emb_dim))  # 可学习的缩放参数
        self.shift = nn.Parameter(torch.zeros(emb_dim))  # 可学习的偏移参数

    def forward(self, x):
        # 计算均值和方差
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        # 归一化并应用缩放和偏移
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift


class GELU(nn.Module):
    # 高斯误差线性单元激活函数
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # 实现GELU激活函数
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))


class FeedForward(nn.Module):
    # 前馈神经网络层
    def __init__(self, cfg):
        super().__init__()
        # 定义前馈网络层序列
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),  # 扩展维度
            GELU(),  # GELU激活
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),  # 压缩回原维度
        )

    def forward(self, x):
        return self.layers(x)


class TransformerBlock(nn.Module):
    # Transformer块
    def __init__(self, cfg):
        super().__init__()
        # 初始化多头注意力层
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"])
        self.ff = FeedForward(cfg)  # 前馈网络
        self.norm1 = LayerNorm(cfg["emb_dim"])  # 第一个层归一化
        self.norm2 = LayerNorm(cfg["emb_dim"])  # 第二个层归一化
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])  # Dropout层

    def forward(self, x):
        # 注意力块的残差连接
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)   # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_shortcut(x)
        x = x + shortcut  # 添加残差连接

        # 前馈网络块的残差连接
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut  # 添加残差连接

        return x


class GPTModel(nn.Module):
    # GPT模型主体
    def __init__(self, cfg):
        super().__init__()
        # 词嵌入层
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        # 位置嵌入层
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        # 嵌入dropout
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        # 创建Transformer块序列
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])

        # 最终的层归一化
        self.final_norm = LayerNorm(cfg["emb_dim"])
        # 输出层
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        # 获取批次大小和序列长度
        batch_size, seq_len = in_idx.shape
        # 计算词嵌入
        tok_embeds = self.tok_emb(in_idx)
        # 计算位置嵌入
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        # 组合词嵌入和位置嵌入
        x = tok_embeds + pos_embeds  # Shape [batch_size, num_tokens, emb_size]
        x = self.drop_emb(x)
        # 通过Transformer块
        x = self.trf_blocks(x)
        # 最终归一化
        x = self.final_norm(x)
        # 计算logits
        logits = self.out_head(x)
        return logits


def generate_text_simple(model, idx, max_new_tokens, context_size):
    # 简单的文本生成函数
    # idx是当前上下文中的索引数组，形状为(B, T)
    for _ in range(max_new_tokens):

        # 如果超出支持的上下文大小，则裁剪当前上下文
        idx_cond = idx[:, -context_size:]

        # 获取预测结果
        with torch.no_grad():
            logits = model(idx_cond)

        # 只关注最后一个时间步
        logits = logits[:, -1, :]

        # 获取logits值最高的词汇表条目的索引
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch, 1)

        # 将采样的索引附加到运行序列中
        idx = torch.cat((idx, idx_next), dim=1)  # (batch, n_tokens+1)

    return idx


#####################################
# Chapter 5
####################################


def calc_loss_batch(input_batch, target_batch, model, device):
    # 计算单个批次的损失
    # 将数据移动到指定设备
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    # 获取模型预测
    logits = model(input_batch)
    # 计算交叉熵损失
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    # 计算整个数据加载器的平均损失
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))
    # 遍历批次
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    return total_loss / num_batches


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    # 评估模型在训练集和验证集上的表现
    model.eval()
    with torch.no_grad():
        # 计算训练损失和验证损失
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    model.train()
    return train_loss, val_loss


def generate_and_print_sample(model, tokenizer, device, start_context):
    # 生成并打印文本样本
    model.eval()
    # 获取上下文大小
    context_size = model.pos_emb.weight.shape[0]
    # 编码起始上下文
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        # 生成文本
        token_ids = generate_text_simple(
            model=model, idx=encoded,
            max_new_tokens=50, context_size=context_size)
        # 解码生成的文本
        decoded_text = token_ids_to_text(token_ids, tokenizer)
        print(decoded_text.replace("\n", " "))  # 紧凑打印格式
    model.train()


def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    # 绘制损失曲线
    fig, ax1 = plt.subplots(figsize=(5, 3))

    # 绘制训练和验证损失
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")

    # 创建第二个x轴显示已见token数
    ax2 = ax1.twiny()  # 创建共享y轴的第二个x轴
    ax2.plot(tokens_seen, train_losses, alpha=0)  # 用于对齐刻度的不可见图
    ax2.set_xlabel("Tokens seen")

    fig.tight_layout()  # 调整布局


def text_to_token_ids(text, tokenizer):
    # 将文本转换为token ID
    encoded = tokenizer.encode(text)
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # 添加批次维度
    return encoded_tensor


def token_ids_to_text(token_ids, tokenizer):
    # 将token ID转换回文本
    flat = token_ids.squeeze(0)  # 移除批次维度
    return tokenizer.decode(flat.tolist())
