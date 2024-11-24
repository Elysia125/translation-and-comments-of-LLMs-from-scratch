# 版权声明: 本代码由Sebastian Raschka在Apache License 2.0下发布(见LICENSE.txt)
# 来源: "从零开始构建大型语言模型"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# 代码仓库: https://github.com/rasbt/LLMs-from-scratch
#
# 本文件收集了第2-6章中所涵盖的所有相关代码
# 本文件可以作为独立脚本运行

# 导入matplotlib库用于绘图
import matplotlib.pyplot as plt
# 导入MaxNLocator用于设置整数刻度
from matplotlib.ticker import MaxNLocator
# 导入numpy用于数值计算
import numpy as np
# 导入tiktoken用于分词
import tiktoken
# 导入PyTorch主库
import torch
# 导入PyTorch神经网络模块
import torch.nn as nn
# 导入Dataset和DataLoader用于数据加载
from torch.utils.data import Dataset, DataLoader


#####################################
# 第2章
#####################################


class GPTDatasetV1(Dataset):
    def __init__(self, txt, tokenizer, max_length, stride):
        # 初始化数据集类
        self.tokenizer = tokenizer
        # 存储输入和目标token ID的列表
        self.input_ids = []
        self.target_ids = []

        # 使用tokenizer对整个文本进行编码
        token_ids = tokenizer.encode(txt, allowed_special={"<|endoftext|>"})

        # 使用滑动窗口将文本分成重叠的序列
        for i in range(0, len(token_ids) - max_length, stride):
            # 获取输入序列
            input_chunk = token_ids[i:i + max_length]
            # 获取目标序列(向后偏移一位)
            target_chunk = token_ids[i + 1: i + max_length + 1]
            # 将序列转换为tensor并添加到列表中
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        # 返回数据集长度
        return len(self.input_ids)

    def __getitem__(self, idx):
        # 返回指定索引的输入和目标序列
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader_v1(txt, batch_size=4, max_length=256,
                         stride=128, shuffle=True, drop_last=True, num_workers=0):
    # 初始化GPT-2 tokenizer
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
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        # 初始化多头注意力层
        super().__init__()
        # 确保输出维度可以被注意力头数整除
        assert d_out % num_heads == 0, "d_out must be divisible by n_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        # 计算每个注意力头的维度
        self.head_dim = d_out // num_heads

        # 创建查询、键、值的线性变换层
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        # 创建输出投影层
        self.out_proj = nn.Linear(d_out, d_out)
        # Dropout层
        self.dropout = nn.Dropout(dropout)
        # 注册因果掩码(上三角矩阵)
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        # 获取输入张量的维度
        b, num_tokens, d_in = x.shape

        # 对输入进行线性变换得到键、查询、值
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        # 重塑张量以支持多头注意力
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # 调整维度顺序以便进行注意力计算
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # 计算注意力分数
        attn_scores = queries @ keys.transpose(2, 3)

        # 将掩码转换为布尔值并截断到当前序列长度
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        # 使用掩码将未来位置的注意力分数设为负无穷
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        # 计算注意力权重并应用dropout
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 计算注意力输出
        context_vec = (attn_weights @ values).transpose(1, 2)

        # 合并多个注意力头的输出
        context_vec = context_vec.reshape(b, num_tokens, self.d_out)
        # 通过输出投影层
        context_vec = self.out_proj(context_vec)

        return context_vec


#####################################
# 第4章
#####################################
class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        # 初始化层归一化
        super().__init__()
        # 设置数值稳定性的小值
        self.eps = 1e-5
        # 可学习的缩放参数
        self.scale = nn.Parameter(torch.ones(emb_dim))
        # 可学习的偏移参数
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    def forward(self, x):
        # 计算均值和方差
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        # 归一化
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        # 应用缩放和偏移
        return self.scale * norm_x + self.shift


class GELU(nn.Module):
    def __init__(self):
        # 初始化GELU激活函数
        super().__init__()

    def forward(self, x):
        # 实现GELU激活函数的近似计算
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) *
            (x + 0.044715 * torch.pow(x, 3))
        ))


class FeedForward(nn.Module):
    def __init__(self, cfg):
        # 初始化前馈神经网络
        super().__init__()
        # 创建包含两个线性层和GELU激活的序列
        self.layers = nn.Sequential(
            nn.Linear(cfg["emb_dim"], 4 * cfg["emb_dim"]),
            GELU(),
            nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]),
        )

    def forward(self, x):
        # 前向传播
        return self.layers(x)


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        # 初始化Transformer块
        super().__init__()
        # 创建多头注意力层
        self.att = MultiHeadAttention(
            d_in=cfg["emb_dim"],
            d_out=cfg["emb_dim"],
            context_length=cfg["context_length"],
            num_heads=cfg["n_heads"],
            dropout=cfg["drop_rate"],
            qkv_bias=cfg["qkv_bias"])
        # 创建前馈网络
        self.ff = FeedForward(cfg)
        # 创建两个层归一化层
        self.norm1 = LayerNorm(cfg["emb_dim"])
        self.norm2 = LayerNorm(cfg["emb_dim"])
        # 创建残差连接的dropout层
        self.drop_resid = nn.Dropout(cfg["drop_rate"])

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


class GPTModel(nn.Module):
    def __init__(self, cfg):
        # 初始化GPT模型
        super().__init__()
        # 创建token嵌入层
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
        # 创建输出层
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx):
        # 获取输入维度
        batch_size, seq_len = in_idx.shape
        # 获取token嵌入
        tok_embeds = self.tok_emb(in_idx)
        # 获取位置嵌入
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        # 组合token和位置嵌入
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


def generate_text_simple(model, idx, max_new_tokens, context_size):
    # 简单的文本生成函数
    # idx是当前上下文中的索引数组，形状为(B, T)
    for _ in range(max_new_tokens):

        # 如果当前上下文超过支持的上下文大小，则裁剪
        idx_cond = idx[:, -context_size:]

        # 获取预测结果
        with torch.no_grad():
            logits = model(idx_cond)

        # 只关注最后一个时间步
        logits = logits[:, -1, :]

        # 获取logits值最高的词汇表条目的索引
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)

        # 将采样的索引附加到运行序列中
        idx = torch.cat((idx, idx_next), dim=1)

    return idx


#####################################
# 第5章
#####################################
def generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):
    # 带有temperature和top-k采样的文本生成函数

    # 循环生成新的token
    for _ in range(max_new_tokens):
        # 获取条件上下文
        idx_cond = idx[:, -context_size:]
        # 获取模型预测
        with torch.no_grad():
            logits = model(idx_cond)
        # 只关注最后一个时间步
        logits = logits[:, -1, :]

        # 使用top-k采样过滤logits
        if top_k is not None:
            # 保留top-k个值
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(logits < min_val, torch.tensor(float('-inf')).to(logits.device), logits)

        # 应用temperature缩放
        if temperature > 0.0:
            logits = logits / temperature

            # 应用softmax获取概率
            probs = torch.softmax(logits, dim=-1)

            # 从分布中采样
            idx_next = torch.multinomial(probs, num_samples=1)

        else:
            # 否则选择logits最高的token
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)

        # 如果遇到结束标记且指定了eos_id，则提前停止生成
        if idx_next == eos_id:
            break

        # 将采样的索引附加到运行序列中
        idx = torch.cat((idx, idx_next), dim=1)

    return idx


def train_model_simple(model, train_loader, val_loader, optimizer, device, num_epochs,
                       eval_freq, eval_iter, start_context, tokenizer):
    # 简单的模型训练函数
    
    # 初始化列表以跟踪损失和已见token数
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    # 主训练循环
    for epoch in range(num_epochs):
        # 设置模型为训练模式
        model.train()

        # 遍历数据加载器中的批次
        for input_batch, target_batch in train_loader:
            # 重置优化器的梯度
            optimizer.zero_grad()
            # 计算批次损失
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            # 计算梯度
            loss.backward()
            # 使用梯度更新模型权重
            optimizer.step()
            # 更新已见token数
            tokens_seen += input_batch.numel()
            global_step += 1

            # 可选的评估步骤
            if global_step % eval_freq == 0:
                # 评估模型
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                # 记录损失和token数
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                # 打印训练状态
                print(f"Ep {epoch+1} (Step {global_step:06d}): "
                      f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}")

        # 每个epoch后生成样本文本
        generate_and_print_sample(
            model, tokenizer, device, start_context
        )

    return train_losses, val_losses, track_tokens_seen


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    # 评估模型函数
    
    # 设置模型为评估模式
    model.eval()
    # 禁用梯度计算
    with torch.no_grad():
        # 计算训练和验证损失
        train_loss = calc_loss_loader(train_loader, model, device, num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
    # 恢复训练模式
    model.train()
    return train_loss, val_loss


def generate_and_print_sample(model, tokenizer, device, start_context):
    # 生成并打印样本文本的函数
    
    # 设置模型为评估模式
    model.eval()
    # 获取上下文大小
    context_size = model.pos_emb.weight.shape[0]
    # 编码起始上下文
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    # 生成文本
    with torch.no_grad():
        token_ids = generate_text_simple(
            model=model, idx=encoded,
            max_new_tokens=50, context_size=context_size
        )
        # 解码生成的文本
        decoded_text = token_ids_to_text(token_ids, tokenizer)
        # 打印文本，将换行符替换为空格
        print(decoded_text.replace("\n", " "))
    # 恢复训练模式
    model.train()


def assign(left, right):
    # 参数赋值函数
    
    # 检查形状是否匹配
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
    # 创建新的参数
    return torch.nn.Parameter(torch.tensor(right))


def load_weights_into_gpt(gpt, params):
    # 将预训练权重加载到GPT模型中的函数
    
    # 加载位置嵌入和token嵌入
    gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params['wpe'])
    gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params['wte'])

    # 遍历所有Transformer块
    for b in range(len(params["blocks"])):
        # 分割注意力权重为查询、键、值
        q_w, k_w, v_w = np.split(
            (params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1)
        # 加载查询、键、值的权重
        gpt.trf_blocks[b].att.W_query.weight = assign(
            gpt.trf_blocks[b].att.W_query.weight, q_w.T)
        gpt.trf_blocks[b].att.W_key.weight = assign(
            gpt.trf_blocks[b].att.W_key.weight, k_w.T)
        gpt.trf_blocks[b].att.W_value.weight = assign(
            gpt.trf_blocks[b].att.W_value.weight, v_w.T)

        # 分割注意力偏置为查询、键、值
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
    # 加载输出层权重
    gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"])


def text_to_token_ids(text, tokenizer):
    # 将文本转换为token ID的函数
    
    # 编码文本
    encoded = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
    # 添加批次维度并返回tensor
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor


def token_ids_to_text(token_ids, tokenizer):
    # 将token ID转换回文本的函数
    
    # 移除批次维度
    flat = token_ids.squeeze(0)
    # 解码为文本
    return tokenizer.decode(flat.tolist())


def calc_loss_batch(input_batch, target_batch, model, device):
    # 计算单个批次损失的函数
    
    # 将输入和目标移到指定设备
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    # 获取模型预测
    logits = model(input_batch)
    # 计算交叉熵损失
    loss = torch.nn.functional.cross_entropy(logits.flatten(0, 1), target_batch.flatten())
    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    # 计算数据加载器中所有批次的平均损失
    
    # 初始化总损失
    total_loss = 0.
    # 处理空数据加载器的情况
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        # 如果指定的批次数超过数据加载器中的批次数，则使用较小的值
        num_batches = min(num_batches, len(data_loader))
    
    # 遍历批次并计算损失
    for i, (input_batch, target_batch) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            total_loss += loss.item()
        else:
            break
    # 返回平均损失
    return total_loss / num_batches


def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    # 绘制训练和验证损失的函数
    
    # 创建图形和主坐标轴
    fig, ax1 = plt.subplots(figsize=(5, 3))

    # 绘制训练和验证损失
    ax1.plot(epochs_seen, train_losses, label="Training loss")
    ax1.plot(epochs_seen, val_losses, linestyle="-.", label="Validation loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    # 设置x轴只显示整数标签
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

    # 创建第二个x轴显示已见token数
    ax2 = ax1.twiny()
    # 创建不可见的图以对齐刻度
    ax2.plot(tokens_seen, train_losses, alpha=0)
    ax2.set_xlabel("Tokens seen")

    # 调整布局
    fig.tight_layout()
    # 保存图形
    plt.savefig("loss-plot.pdf")
    # 显示图形
    plt.show()
