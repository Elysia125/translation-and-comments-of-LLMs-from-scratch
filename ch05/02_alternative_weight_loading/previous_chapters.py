# 版权声明 - 本代码基于 Apache License 2.0 许可证 (见 LICENSE.txt)
# 来源于《从零开始构建大型语言模型》一书
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# 代码仓库: https://github.com/rasbt/LLMs-from-scratch
#
# 本文件收集了第2-4章中涉及的所有相关代码
# 本文件可以作为独立脚本运行

# 导入必要的库
import tiktoken  # OpenAI的分词器
import torch  # PyTorch深度学习框架
import torch.nn as nn  # PyTorch神经网络模块
from torch.utils.data import Dataset, DataLoader  # 用于数据加载的工具类

#####################################
# 第2章
#####################################


class GPTDatasetV1(Dataset):
    """GPT数据集的第一个版本实现"""
    def __init__(self, txt, tokenizer, max_length, stride):
        self.input_ids = []  # 存储输入序列
        self.target_ids = []  # 存储目标序列

        # 使用分词器对整个文本进行编码
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        # 使用滑动窗口将文本分成重叠的序列，每个序列长度为max_length
        for i in range(0, len(token_ids) - max_length, stride):
            input_chunk = token_ids[i:i + max_length]  # 输入序列
            target_chunk = token_ids[i + 1: i + max_length + 1]  # 目标序列(向后偏移一位)
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        """返回数据集的长度"""
        return len(self.input_ids)

    def __getitem__(self, idx):
        """返回指定索引的数据项"""
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(txt, batch_size=4, max_length=256,
                         stride=128, shuffle=True, drop_last=True, num_workers=0):
    """创建数据加载器的函数"""
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
class MultiHeadAttention(nn.Module):
    """多头注意力机制的实现"""
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out必须能被num_heads整除"

        self.d_out = d_out  # 输出维度
        self.num_heads = num_heads  # 注意力头数
        self.head_dim = d_out // num_heads  # 每个注意力头的维度

        # 定义查询、键、值的线性变换层
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)  # 输出投影层
        self.dropout = nn.Dropout(dropout)  # dropout层
        # 注册因果掩码(上三角矩阵)
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        """前向传播"""
        b, num_tokens, d_in = x.shape  # 获取输入张量的形状

        # 计算查询、键、值矩阵
        keys = self.W_key(x)  # 形状: (b, num_tokens, d_out)
        queries = self.W_query(x)
        values = self.W_value(x)

        # 重塑张量以支持多头注意力
        # (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # 调整维度顺序
        # (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # 计算注意力分数
        attn_scores = queries @ keys.transpose(2, 3)  # 矩阵乘法计算注意力分数

        # 将原始掩码截断到所需的token数量并转换为布尔类型
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        # 使用掩码填充注意力分数
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        # 计算注意力权重
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 计算上下文向量 (b, num_tokens, num_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2)

        # 合并所有注意力头
        context_vec = context_vec.reshape(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec)  # 可选的投影

        return context_vec


#####################################
# 第4章
#####################################
class LayerNorm(nn.Module):
    """层归一化的实现"""
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5  # 防止除零的小常数
        self.scale = nn.Parameter(torch.ones(emb_dim))  # 可学习的缩放参数
        self.shift = nn.Parameter(torch.zeros(emb_dim))  # 可学习的偏移参数

    def forward(self, x):
        """前向传播"""
        mean = x.mean(dim=-1, keepdim=True)  # 计算均值
        var = x.var(dim=-1, keepdim=True, unbiased=False)  # 计算方差
        norm_x = (x - mean) / torch.sqrt(var + self.eps)  # 标准化
        return self.scale * norm_x + self.shift  # 缩放和偏移


class GELU(nn.Module):
    """GELU激活函数的实现"""
    def __init__(self):
        super().__init__()

    def forward(self, x):
        """前向传播 - 使用GELU的近似实现"""
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))


class FeedForward(nn.Module):
    """前馈神经网络的实现"""
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),  # 第一个线性层
            GELU(),  # GELU激活函数
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),  # 第二个线性层
        )

    def forward(self, x):
        """前向传播"""
        return self.layers(x)


class TransformerBlock(nn.Module):
    """Transformer块的实现"""
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
        self.drop_shortcut = nn.Dropout(cfg["drop_rate"])  # dropout层

    def forward(self, x):
        """前向传播"""
        # 注意力块的残差连接
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)   # 形状 [batch_size, num_tokens, emb_size]
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
    """GPT模型的实现"""
    def __init__(self, cfg):
        super().__init__()
        # 初始化词嵌入层
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        # 初始化位置嵌入层
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        # 创建Transformer块的序列
        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])

        self.final_norm = LayerNorm(cfg["emb_dim"])  # 最终的层归一化
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)  # 输出层

    def forward(self, in_idx):
        """前向传播"""
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)  # 词嵌入
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))  # 位置嵌入
        x = tok_embeds + pos_embeds  # 形状 [batch_size, num_tokens, emb_size]
        x = self.drop_emb(x)  # 应用dropout
        x = self.trf_blocks(x)  # 通过Transformer块
        x = self.final_norm(x)  # 最终归一化
        logits = self.out_head(x)  # 计算logits
        return logits


def generate_text_simple(model, idx, max_new_tokens, context_size):
    """简单的文本生成函数"""
    # idx是当前上下文中的索引数组，形状为(B, T)
    for _ in range(max_new_tokens):

        # 如果超出支持的上下文大小，则裁剪当前上下文
        # 例如，如果LLM只支持5个token，而上下文大小是10
        # 则只使用最后5个token作为上下文
        idx_cond = idx[:, -context_size:]

        # 获取预测结果
        with torch.no_grad():
            logits = model(idx_cond)

        # 只关注最后一个时间步
        # (batch, n_token, vocab_size)变为(batch, vocab_size)
        logits = logits[:, -1, :]

        # 获取logits值最高的词汇表条目的索引
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch, 1)

        # 将采样的索引附加到运行序列中
        idx = torch.cat((idx, idx_next), dim=1)  # (batch, n_tokens+1)

    return idx


#####################################
# 第5章
#####################################


def text_to_token_ids(text, tokenizer):
    """将文本转换为token ID"""
    encoded = tokenizer.encode(text)  # 编码文本
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # 添加批次维度
    return encoded_tensor


def token_ids_to_text(token_ids, tokenizer):
    """将token ID转换回文本"""
    flat = token_ids.squeeze(0)  # 移除批次维度
    return tokenizer.decode(flat.tolist())  # 解码为文本


def generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):
    """增强版的文本生成函数"""

    # 与之前相同的循环：获取logits，只关注最后一个时间步
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]  # 获取上下文
        with torch.no_grad():
            logits = model(idx_cond)  # 获取模型预测
        logits = logits[:, -1, :]  # 只保留最后一个时间步

        # 新增：使用top_k采样过滤logits
        if top_k is not None:
            # 只保留top_k个值
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(logits < min_val, torch.tensor(float('-inf')).to(logits.device), logits)

        # 新增：应用温度缩放
        if temperature > 0.0:
            logits = logits / temperature  # 应用温度

            # 应用softmax获取概率
            probs = torch.softmax(logits, dim=-1)  # (batch_size, context_len)

            # 从分布中采样
            idx_next = torch.multinomial(probs, num_samples=1)  # (batch_size, 1)

        # 否则与之前相同：获取logits值最高的词汇表条目的索引
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch_size, 1)

        # 如果遇到结束符且指定了eos_id，则提前停止生成
        if idx_next == eos_id:
            break

        # 与之前相同：将采样的索引附加到运行序列中
        idx = torch.cat((idx, idx_next), dim=1)  # (batch_size, num_tokens+1)

    return idx