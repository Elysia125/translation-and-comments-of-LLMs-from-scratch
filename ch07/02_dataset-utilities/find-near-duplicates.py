
# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

# 导入所需的Python库
import argparse  # 用于解析命令行参数
import json  # 用于处理JSON数据
import re  # 用于正则表达式操作
from sklearn import __version__ as sklearn_version  # 导入scikit-learn版本号
from sklearn.feature_extraction.text import TfidfVectorizer  # 用于文本向量化
from sklearn.metrics.pairwise import cosine_similarity  # 用于计算余弦相似度


# Sample JSON dataset
# 示例JSON数据集,包含指令-输入-输出的格式
example_data = [
    {"instruction": "What is the capital of Italy?",
     "input": "", "output": "The capital of Italy is Rome."
     },
    {"instruction": "What's the capital city of Italy?",
     "input": "", "output": "The capital city is Rome."
     },
    {"instruction": "Identify the main verb in the sentence: 'The cat sleeps on the couch.'",
     "input": "", "output": "The verb is 'sleeps'."
     },
    {"instruction": "Identify the verb in the following sentence: The cat sleeps on the couch.",
     "input": "", "output": "The verb in the sentence is \"sleeps.\""
     },
    # ...
]


def preprocess_text(text):
    # Lowercase the text
    text = text.lower()  # 将文本转换为小写
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)  # 移除标点符号
    return text


def find_near_duplicates(json_data, threshold=0.75, key="instruction"):
    """The higher the threshold, the more similar the texts have to be to match"""
    # 阈值越高,文本需要越相似才会被认为是重复

    # Extract instructions
    text = [preprocess_text(item[key]) for item in json_data if item[key]]  # 提取并预处理指定key的文本
    near_duplicates = []  # 存储近似重复的文本对
    indices_to_remove = set()  # 存储需要移除的索引

    if not text:  # 如果没有文本,返回空结果
        return {}, near_duplicates

    # Vectorize the text data
    vectorizer = TfidfVectorizer(stop_words=None, analyzer='char', ngram_range=(1, 3))  # 创建TF-IDF向量化器
    tfidf_matrix = vectorizer.fit_transform(text)  # 将文本转换为TF-IDF矩阵

    # Compute cosine similarity between each pair of entries
    cos_sim_matrix = cosine_similarity(tfidf_matrix)  # 计算文本间的余弦相似度矩阵

    # Find pairs of near-duplicate instructions based on the threshold
    # 基于阈值寻找近似重复的文本对
    for i in range(len(cos_sim_matrix)):
        for j in range(i+1, len(cos_sim_matrix)):
            if cos_sim_matrix[i, j] > threshold:
                if len(json_data[i][key]) <= 1 or len(json_data[j][key]) <= 1:  # 跳过长度为1或0的文本
                    continue
                near_duplicates.append((json_data[i], json_data[j], cos_sim_matrix[i, j]))  # 添加重复对
                if key in ("input", "output"):  # Don't remove duplicates based on the instruction
                    indices_to_remove.add(j)  # 标记需要移除的索引

    # Remove the near-duplicate entries
    filtered_json_data = [item for index, item in enumerate(json_data) if index not in indices_to_remove]  # 移除重复项

    return filtered_json_data, near_duplicates


def find_print_and_remove_near_duplicates(json_data, remove_duplicates=False, threshold=0.75):
    """
    Searches each key in the first JSON object for duplicates across a list of JSON objects.
    Prints the duplicates if found.
    """
    # 遍历JSON对象中的每个键,搜索重复项并打印
    for key in json_data[0].keys():  # 遍历第一个JSON对象的所有键

        if remove_duplicates:  # 如果需要移除重复项
            json_data, near_duplicates = find_near_duplicates(json_data, key=key, threshold=threshold)
        else:  # 如果只需要查找重复项
            _, near_duplicates = find_near_duplicates(json_data, key=key, threshold=threshold)
        separator = 50 * '='  # 分隔符
        print(f"\n\n{separator}\nSearching '{key}' for duplicates ...\n{separator}")  # 打印搜索信息
        if not near_duplicates:  # 如果没有找到重复项
            print("No duplicates found")
        else:  # 如果找到重复项,打印每对重复项
            for dup in near_duplicates:
                print(
                    f"Duplicate pair found with similarity {dup[2]:.2f}:\n"
                    f"1. {dup[0][key]}\n2. {dup[1][key]}\n"
                )
    return json_data


if __name__ == "__main__":
    print("scikit-learn version:", sklearn_version)  # 打印scikit-learn版本

    # 创建命令行参数解析器
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--json_file",
        type=str,
        help=("Path to the dataset JSON file")  # JSON数据集文件路径
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.9,
        help=("A sensitivity threshold between 0 and 1 where 1 is strictest")  # 相似度阈值
    )
    parser.add_argument(
        "--remove_duplicates",
        action='store_true',
        default=False,
        help=(  # 是否移除重复项的标志
            "Removes duplicates based on the 'input' or 'output' keys "
            " (but not the 'instruction') and saves the cleaned JSON file as --json_output_file"
        )
    )
    parser.add_argument(
        "--json_output_file",
        type=str,
        help=("Path to the dataset JSON file")  # 输出JSON文件路径
    )

    args = parser.parse_args()  # 解析命令行参数

    # 验证参数
    if args.remove_duplicates and not args.json_output_file:
        raise ValueError(
            "Provide an output file via --json_output_file "
            "to save the cleaned JSON data."
        )

    # 加载JSON数据
    if not args.json_file:
        json_data = example_data  # 如果没有提供文件,使用示例数据
    else:
        with open(args.json_file, "r") as file:  # 从文件读取JSON数据
            json_data = json.load(file)

    # 处理数据
    json_data = find_print_and_remove_near_duplicates(
        json_data=json_data,
        remove_duplicates=args.remove_duplicates,
        threshold=args.threshold
    )

    # 如果需要移除重复项,将处理后的数据保存到文件
    if args.remove_duplicates:
        with open(args.json_output_file, "w") as file:
            json.dump(json_data, file, indent=4)
