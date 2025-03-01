import json

def calculate_accuracy(json_file):
    # 读取 JSON 文件
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 初始化正确和错误的计数器
    correct_count = 0
    total_count = len(data)

    # 计算正确和错误的条目数量
    for entry in data:
        answer = entry["answer"]
        label = entry["label"]
        if ("Option A" in answer  and label == "0") or ("Option B" in answer and label == "1") or ("Option C" in answer and label == "2"):
            correct_count += 1

    # 计算正确率
    accuracy = correct_count / total_count
    return accuracy

# 计算第一个 JSON 文件的正确率
json_file1_path = "/home/xuxiaoan/Debias/Guide-Align-main/code/evaluate/result/answer_llama3_fassis_with_guidelines_whole.json"  # 替换为第一个 JSON 文件路径
accuracy1 = calculate_accuracy(json_file1_path)

# 计算第二个 JSON 文件的正确率
json_file2_path = "/home/xuxiaoan/Debias/Guide-Align-main/code/evaluate/result/answer_llama3_whole_1000_test.json"  # 替换为第二个 JSON 文件路径
accuracy2 = calculate_accuracy(json_file2_path)

# # 计算第三个 JSON 文件的正确率
# json_file3_path = "/home/xuxiaoan/Debias/Guide-Align-main/code/evaluate/answer_llama3_fassis_with_suggestions.json"  # 替换为第三个 JSON 文件路径
# accuracy3 = calculate_accuracy(json_file3_path)

# 将结果保存到 JSON 文件
result = {
    "accuracy1": accuracy1,
    "accuracy2": accuracy2
    # "accuracy3": accuracy3
}

output_file_path = "accuracy_results.json"  # 输出文件路径
with open(output_file_path, "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

print("Accuracy results saved to:", output_file_path)
