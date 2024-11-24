# 版权声明 - 作者Sebastian Raschka，使用Apache License 2.0许可
# 来源于"从零开始构建大型语言模型"一书
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# 代码仓库: https://github.com/rasbt/LLMs-from-scratch
#
# 基于第7章代码的最小指令微调文件

# 导入json模块用于处理JSON数据
import json
# 导入psutil用于检查系统进程
import psutil
# 导入tqdm用于显示进度条
from tqdm import tqdm
# 导入urllib.request用于发送HTTP请求
import urllib.request


def query_model(prompt, model="llama3", url="http://localhost:11434/api/chat"):
    # 创建包含请求数据的字典
    data = {
        "model": model,  # 指定要使用的模型
        "messages": [
            {"role": "user", "content": prompt}  # 设置用户提示信息
        ],
        "options": {     # 设置生成参数以获得确定性的响应
            "seed": 123,  # 设置随机种子
            "temperature": 0,  # 设置温度为0使输出更确定
            "num_ctx": 2048  # 设置上下文窗口大小
        }
    }

    # 将字典转换为JSON字符串并编码为字节
    payload = json.dumps(data).encode("utf-8")

    # 创建POST请求对象并添加必要的头部信息
    request = urllib.request.Request(url, data=payload, method="POST")
    request.add_header("Content-Type", "application/json")

    # 发送请求并获取响应
    response_data = ""
    with urllib.request.urlopen(request) as response:
        # 读取并解码响应
        while True:
            line = response.readline().decode("utf-8")
            if not line:  # 如果没有更多数据则退出循环
                break
            response_json = json.loads(line)  # 解析JSON响应
            response_data += response_json["message"]["content"]  # 累加响应内容

    return response_data  # 返回完整的响应文本


def check_if_running(process_name):
    # 检查指定进程是否正在运行
    running = False
    for proc in psutil.process_iter(["name"]):  # 遍历所有运行中的进程
        if process_name in proc.info["name"]:  # 检查进程名是否匹配
            running = True
            break
    return running


def format_input(entry):
    # 格式化输入文本，添加指令和输入提示
    instruction_text = (
        f"Below is an instruction that describes a task. "
        f"Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )

    # 如果有输入文本则添加，否则为空
    input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""

    return instruction_text + input_text  # 返回完整的格式化文本


def main(file_path):
    # 检查ollama服务是否运行
    ollama_running = check_if_running("ollama")

    # 如果ollama未运行则抛出异常
    if not ollama_running:
        raise RuntimeError("Ollama not running. Launch ollama before proceeding.")
    print("Ollama running:", check_if_running("ollama"))

    # 读取测试数据文件
    with open(file_path, "r") as file:
        test_data = json.load(file)

    # 设置模型并生成评分
    model = "llama3"
    scores = generate_model_scores(test_data, "model_response", model)
    # 打印评分统计信息
    print(f"Number of scores: {len(scores)} of {len(test_data)}")
    print(f"Average score: {sum(scores)/len(scores):.2f}\n")


def generate_model_scores(json_data, json_key, model="llama3"):
    # 存储所有评分的列表
    scores = []
    # 遍历数据条目并生成评分
    for entry in tqdm(json_data, desc="Scoring entries"):
        if entry[json_key] == "":  # 如果模型响应为空，评分为0
            scores.append(0)
        else:
            # 构造评分提示
            prompt = (
                f"Given the input `{format_input(entry)}` "
                f"and correct output `{entry['output']}`, "
                f"score the model response `{entry[json_key]}`"
                f" on a scale from 0 to 100, where 100 is the best score. "
                f"Respond with the integer number only."
            )
            # 获取模型评分
            score = query_model(prompt, model)
            try:
                scores.append(int(score))  # 将评分转换为整数并添加到列表
            except ValueError:
                print(f"Could not convert score: {score}")  # 如果转换失败则打印错误
                continue

    return scores  # 返回所有评分列表


if __name__ == "__main__":
    # 导入参数解析模块
    import argparse

    # 创建参数解析器
    parser = argparse.ArgumentParser(
        description="Evaluate model responses with ollama"
    )
    # 添加文件路径参数
    parser.add_argument(
        "--file_path",
        required=True,
        help=(
            "The path to the test dataset `.json` file with the"
            " `'output'` and `'model_response'` keys"
        )
    )
    # 解析命令行参数
    args = parser.parse_args()

    # 调用主函数
    main(file_path=args.file_path)
