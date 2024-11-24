# Source: https://github.com/openai/gpt-2/blob/master/src/encoder.py
# License:
# Modified MIT License

# Software Copyright (c) 2019 OpenAI

# We don't claim ownership of the content you create with GPT-2, so it is yours to do with as you please.
# 我们不对您使用GPT-2创建的内容主张所有权，因此您可以随意处置。
# We only ask that you use GPT-2 responsibly and clearly indicate your content was created using GPT-2.
# 我们只要求您负责任地使用GPT-2，并明确指出您的内容是使用GPT-2创建的。

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and
# associated documentation files (the "Software"), to deal in the Software without restriction,
# including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
# 特此免费授予任何获得本软件及相关文档文件（"软件"）副本的人无限制处理软件的权限，
# 包括但不限于使用、复制、修改、合并、发布、分发、再许可和/或出售软件副本的权利，
# 并允许向其提供软件的人这样做，但须符合以下条件：

# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software.
# 上述版权声明和本许可声明应包含在软件的所有副本或主要部分中。
# The above copyright notice and this permission notice need not be included
# with content created by the Software.
# 上述版权声明和本许可声明不需要包含在由软件创建的内容中。

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS
# BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE
# OR OTHER DEALINGS IN THE SOFTWARE.
# 本软件按"原样"提供，不提供任何形式的明示或暗示的保证，
# 包括但不限于对适销性、特定用途的适用性和非侵权性的保证。
# 在任何情况下，作者或版权持有人均不对任何索赔、损害或其他责任负责，
# 无论是在合同诉讼、侵权行为或其他方面，由软件或软件的使用或其他交易引起、产生或与之相关。

# 导入操作系统相关功能的模块
import os
# 导入JSON数据处理的模块
import json
# 导入正则表达式模块(使用regex替代re以支持更多功能)
import regex as re
# 导入HTTP请求处理模块
import requests
# 导入进度条显示模块
from tqdm import tqdm
# 导入LRU缓存装饰器
from functools import lru_cache


# @lru_cache()装饰器用于缓存函数的返回结果
# 当使用相同的参数多次调用函数时,会直接返回缓存的结果而不重新计算
# 这可以显著提高程序性能,特别是对于计算密集型的函数
# LRU(Least Recently Used)表示当缓存满时会删除最近最少使用的结果
@lru_cache()
def bytes_to_unicode():
    """
    Returns list of utf-8 byte and a corresponding list of unicode strings.
    返回utf-8字节列表和对应的unicode字符串列表。
    The reversible bpe codes work on unicode strings.
    可逆的BPE编码在unicode字符串上工作。
    This means you need a large # of unicode characters in your vocab if you want to avoid UNKs.
    这意味着如果要避免未知词符(UNKs)，词汇表中需要大量的unicode字符。
    When you're at something like a 10B token dataset you end up needing around 5K for decent coverage.
    当你处理类似100亿token的数据集时，你最终需要大约5K的词汇量才能获得decent覆盖率。
    This is a significant percentage of your normal, say, 32K bpe vocab.
    这在你通常使用的32K BPE词汇表中占据了相当大的比例。
    To avoid that, we want lookup tables between utf-8 bytes and unicode strings.
    为了避免这种情况，我们需要在utf-8字节和unicode字符串之间建立查找表。
    And avoids mapping to whitespace/control characters the bpe code barfs on.
    并避免映射到会导致BPE代码出错的空白字符/控制字符。
    """
    # 创建一个包含ASCII可打印字符和扩展ASCII字符的列表
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    # 复制bs列表
    cs = bs[:]
    n = 0
    # 遍历所有可能的字节值(0-255)
    for b in range(2**8):
        # 如果字节值不在bs中,添加到bs和cs中
        if b not in bs:
            # 将当前字节值b添加到bs列表中
            bs.append(b)
            # Unicode私有使用区(PUA)是Unicode字符集中专门预留给用户自定义字符的区域
            # 它包含三个区块:
            # - 基本平面(BMP)中的 U+E000 到 U+F8FF (6400个码位)
            # - 第15平面中的 U+F0000 到 U+FFFFD (65534个码位) 
            # - 第16平面中的 U+100000 到 U+10FFFD (65534个码位)
            # 这里我们使用256(2^8)作为基准,加上计数器n来生成PUA中的码点
            cs.append(2**8 + n)
            # 计数器加1,为下一个映射准备
            n += 1
    # 将cs中的数字转换为对应的Unicode字符
    cs = [chr(n) for n in cs]
    # 返回字节到Unicode字符的映射字典
    return dict(zip(bs, cs))


def get_pairs(word):
    """
    Return set of symbol pairs in a word.
    返回一个词中的符号对集合。
    Word is represented as tuple of symbols (symbols being variable-length strings).
    词被表示为符号元组(符号是可变长度的字符串)。
    """
    # 创建一个空集合用于存储字符对
    pairs = set()
    # 获取第一个字符作为前一个字符
    prev_char = word[0]
    # 遍历剩余字符
    for char in word[1:]:
        # 将前一个字符和当前字符组成的元组添加到集合中
        pairs.add((prev_char, char))
        # 更新前一个字符
        prev_char = char
    return pairs


class Encoder:
    def __init__(self, encoder, bpe_merges, errors='replace'):
        """
        初始化BPE编码器
        Args:
            encoder: 词汇表字典,将token映射到id
            bpe_merges: BPE合并规则列表,每个规则是一个元组(first, second),
                       表示将字符first和second合并成一个新的token。
                       这些规则是按优先级排序的,索引越小优先级越高。
            errors: 解码时的错误处理方式,默认为'replace'
        """
        # 初始化编码器字典
        self.encoder = encoder
        # 创建解码器字典(编码器的反向映射)
        self.decoder = {v: k for k, v in self.encoder.items()}
        # 设置解码错误处理方式
        self.errors = errors  # how to handle errors in decoding
        # 获取字节到Unicode的编码器
        self.byte_encoder = bytes_to_unicode()
        # 创建Unicode到字节的解码器
        self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}
        # 创建BPE合并规则的排名字典
        self.bpe_ranks = dict(zip(bpe_merges, range(len(bpe_merges))))
        # 初始化缓存字典
        self.cache = {}

        # 编译正则表达式模式用于分词
        # Should have added re.IGNORECASE so BPE merges can happen for capitalized versions of contractions
        self.pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")

    def bpe(self, token):
        # 如果token在缓存中,直接返回缓存的结果
        if token in self.cache:
            return self.cache[token]
        # 将token转换为元组
        word = tuple(token)
        # 获取所有相邻字符对
        pairs = get_pairs(word)

        # 如果没有字符对,直接返回token
        if not pairs:
            return token

        while True:
            # 找到排名最低的字符对
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float('inf')))
            # 如果该字符对不在合并规则中,退出循环
            if bigram not in self.bpe_ranks:
                break
            # 分解字符对为first和second
            first, second = bigram
            # 创建新的单词列表
            new_word = []
            i = 0
            while i < len(word):
                try:
                    # 查找first的下一个位置
                    j = word.index(first, i)
                    # 添加i到j之间的所有字符
                    new_word.extend(word[i:j])
                    i = j
                except ValueError:
                    # 如果找不到first,添加剩余所有字符
                    new_word.extend(word[i:])
                    break

                # 如果找到连续的first和second,合并它们
                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    # 否则只添加当前字符
                    new_word.append(word[i])
                    i += 1
            # 将新单词转换为元组
            new_word = tuple(new_word)
            word = new_word
            # 如果单词长度为1,退出循环
            if len(word) == 1:
                break
            else:
                # 否则重新获取字符对
                pairs = get_pairs(word)
        # 将单词转换为空格分隔的字符串
        word = ' '.join(word)
        # 将结果存入缓存
        self.cache[token] = word
        return word

    def encode(self, text):
        # 创建空列表存储BPE tokens
        bpe_tokens = []
        # 使用正则表达式分词
        for token in re.findall(self.pat, text):
            # 将token转换为UTF-8编码,然后转换为Unicode字符
            token = ''.join(self.byte_encoder[b] for b in token.encode('utf-8'))
            # 对token进行BPE编码,并添加到结果列表中
            bpe_tokens.extend(self.encoder[bpe_token] for bpe_token in self.bpe(token).split(' '))
        return bpe_tokens

    def decode(self, tokens):
        # 将token ID转换回原始文本
        text = ''.join([self.decoder[token] for token in tokens])
        # 将Unicode字符转换回UTF-8字节,然后解码为字符串
        text = bytearray([self.byte_decoder[c] for c in text]).decode('utf-8', errors=self.errors)
        return text


def get_encoder(model_name, models_dir):
    # 读取encoder.json文件
    with open(os.path.join(models_dir, model_name, 'encoder.json'), 'r') as f:
        encoder = json.load(f)
    # 读取vocab.bpe文件
    with open(os.path.join(models_dir, model_name, 'vocab.bpe'), 'r', encoding="utf-8") as f:
        bpe_data = f.read()
    # 解析BPE合并规则
    bpe_merges = [tuple(merge_str.split()) for merge_str in bpe_data.split('\n')[1:-1]]
    # 返回Encoder实例
    return Encoder(encoder=encoder, bpe_merges=bpe_merges)


def download_vocab():
    # Modified code from
    # 设置模型文件保存目录
    subdir = 'gpt2_model'
    # 如果目录不存在则创建
    if not os.path.exists(subdir):
        os.makedirs(subdir)
    # 处理Windows路径分隔符
    subdir = subdir.replace('\\', '/')  # needed for Windows

    # 下载encoder.json和vocab.bpe文件
    for filename in ['encoder.json', 'vocab.bpe']:
        # 发送GET请求下载文件
        r = requests.get("https://openaipublic.blob.core.windows.net/gpt-2/models/117M/" + filename, stream=True)

        # 打开文件准备写入
        with open(os.path.join(subdir, filename), 'wb') as f:
            # 获取文件大小
            file_size = int(r.headers["content-length"])
            # 设置块大小
            chunk_size = 1000
            # 使用tqdm显示下载进度
            with tqdm(ncols=100, desc="Fetching " + filename, total=file_size, unit_scale=True) as pbar:
                # 1k for chunk_size, since Ethernet packet size is around 1500 bytes
                # 分块下载并写入文件
                for chunk in r.iter_content(chunk_size=chunk_size):
                    f.write(chunk)
                    pbar.update(chunk_size)
