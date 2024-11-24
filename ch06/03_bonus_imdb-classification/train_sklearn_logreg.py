# Copyright (c) Sebastian Raschka under Apache License 2.0 (see LICENSE.txt).
# Source for "Build a Large Language Model From Scratch"
#   - https://www.manning.com/books/build-a-large-language-model-from-scratch
# Code: https://github.com/rasbt/LLMs-from-scratch

# 导入必要的库
import pandas as pd  # 用于数据处理和分析
from sklearn.feature_extraction.text import CountVectorizer  # 用于文本特征提取
from sklearn.linear_model import LogisticRegression  # 用于逻辑回归分类
from sklearn.metrics import accuracy_score  # 用于计算准确率
# from sklearn.metrics import balanced_accuracy_score
from sklearn.dummy import DummyClassifier  # 用于创建基准模型


def load_dataframes():
    # 从CSV文件加载训练集
    df_train = pd.read_csv("train.csv")
    # 从CSV文件加载验证集
    df_val = pd.read_csv("validation.csv")
    # 从CSV文件加载测试集
    df_test = pd.read_csv("test.csv")

    return df_train, df_val, df_test


def eval(model, X_train, y_train, X_val, y_val, X_test, y_test):
    # Making predictions
    # 对训练集进行预测
    y_pred_train = model.predict(X_train)
    # 对验证集进行预测
    y_pred_val = model.predict(X_val)
    # 对测试集进行预测
    y_pred_test = model.predict(X_test)

    # Calculating accuracy and balanced accuracy
    # 计算训练集准确率
    accuracy_train = accuracy_score(y_train, y_pred_train)
    # balanced_accuracy_train = balanced_accuracy_score(y_train, y_pred_train)

    # 计算验证集准确率
    accuracy_val = accuracy_score(y_val, y_pred_val)
    # balanced_accuracy_val = balanced_accuracy_score(y_val, y_pred_val)

    # 计算测试集准确率
    accuracy_test = accuracy_score(y_test, y_pred_test)
    # balanced_accuracy_test = balanced_accuracy_score(y_test, y_pred_test)

    # Printing the results
    # 打印各个数据集的准确率结果
    print(f"Training Accuracy: {accuracy_train*100:.2f}%")
    print(f"Validation Accuracy: {accuracy_val*100:.2f}%")
    print(f"Test Accuracy: {accuracy_test*100:.2f}%")

    # print(f"\nTraining Balanced Accuracy: {balanced_accuracy_train*100:.2f}%")
    # print(f"Validation Balanced Accuracy: {balanced_accuracy_val*100:.2f}%")
    # print(f"Test Balanced Accuracy: {balanced_accuracy_test*100:.2f}%")


if __name__ == "__main__":
    # 加载所有数据集
    df_train, df_val, df_test = load_dataframes()

    #########################################
    # Convert text into bag-of-words model
    # 初始化词袋模型向量化器
    vectorizer = CountVectorizer()
    #########################################

    # 对训练集文本进行特征提取和转换
    X_train = vectorizer.fit_transform(df_train["text"])
    # 使用训练好的向量化器转换验证集
    X_val = vectorizer.transform(df_val["text"])
    # 使用训练好的向量化器转换测试集
    X_test = vectorizer.transform(df_test["text"])
    # 提取所有数据集的标签
    y_train, y_val, y_test = df_train["label"], df_val["label"], df_test["label"]

    #####################################
    # Model training and evaluation
    #####################################

    # Create a dummy classifier with the strategy to predict the most frequent class
    # 创建一个基准分类器，使用最频繁类别作为预测策略
    dummy_clf = DummyClassifier(strategy="most_frequent")
    # 训练基准分类器
    dummy_clf.fit(X_train, y_train)

    # 打印基准分类器的评估结果
    print("Dummy classifier:")
    eval(dummy_clf, X_train, y_train, X_val, y_val, X_test, y_test)

    # 打印逻辑回归分类器的评估结果
    print("\n\nLogistic regression classifier:")
    # 创建逻辑回归模型，设置最大迭代次数为1000
    model = LogisticRegression(max_iter=1000)
    # 训练逻辑回归模型
    model.fit(X_train, y_train)
    # 评估逻辑回归模型
    eval(model, X_train, y_train, X_val, y_val, X_test, y_test)
