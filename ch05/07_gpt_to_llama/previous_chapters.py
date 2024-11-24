# 版权声明 - 本代码基于 Apache License 2.0 许可证 (见 LICENSE.txt)
# 来源于《从零开始构建大型语言模型》一书
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# 代码仓库: https://github.com/rasbt/LLMs-from-scratch
#
# 本文件收集了第2-4章中涉及的所有相关代码
# 本文件可以作为独立脚本运行

# 导入PyTorch深度学习框架
import torch


#####################################
# 第5章
#####################################

def text_to_token_ids(text, tokenizer):
    """将文本转换为token ID"""
    encoded = tokenizer.encode(text)  # 使用分词器将文本编码为token ID
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)  # 添加batch维度
    return encoded_tensor


def token_ids_to_text(token_ids, tokenizer):
    """将token ID转换回文本"""
    flat = token_ids.squeeze(0)  # 移除batch维度
    return tokenizer.decode(flat.tolist())  # 将token ID解码为文本


def generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):
    """生成文本的函数
    
    参数:
    - model: 语言模型
    - idx: 初始token序列
    - max_new_tokens: 最大生成token数量
    - context_size: 上下文窗口大小
    - temperature: 采样温度(0表示贪婪搜索)
    - top_k: top-k采样的k值
    - eos_id: 结束符token ID
    """

    # 循环生成新的token
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]  # 截取最后context_size个token作为条件
        with torch.no_grad():  # 不计算梯度
            logits = model(idx_cond)  # 获取模型预测的logits
        logits = logits[:, -1, :]  # 只保留最后一个时间步的logits

        # 如果指定了top_k,进行top-k过滤
        if top_k is not None:
            # 保留最高的k个值
            top_logits, _ = torch.topk(logits, top_k)  
            min_val = top_logits[:, -1]  # 获取第k个最大值
            # 将小于min_val的logits设为负无穷
            logits = torch.where(logits < min_val, torch.tensor(float('-inf')).to(logits.device), logits)

        # 如果temperature>0,使用温度缩放进行采样
        if temperature > 0.0:
            logits = logits / temperature  # 应用温度缩放

            # 使用softmax获取概率分布
            probs = torch.softmax(logits, dim=-1)  # (batch_size, context_len)

            # 从概率分布中采样
            idx_next = torch.multinomial(probs, num_samples=1)  # (batch_size, 1)

        # 否则使用贪婪搜索,选择logits最大值对应的token
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch_size, 1)

        # 如果生成了结束符且指定了eos_id,提前结束生成
        if idx_next == eos_id:
            break

        # 将新生成的token添加到序列中
        idx = torch.cat((idx, idx_next), dim=1)  # (batch_size, num_tokens+1)

    return idx
