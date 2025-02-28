# 假设输入文件路径和输出文件路径
input_file_path = 'Insult_instructions.txt'
output_file_path = 'Insult_instructions_100s.txt'

# 打开输入文件并读取前100行
with open(input_file_path, 'r', encoding='utf-8') as file:
    lines = [next(file) for _ in range(100)]

# 打开输出文件并写入这100行
with open(output_file_path, 'w', encoding='utf-8') as file:
    file.writelines(lines)

print("前100行已成功保存到输出文件。")
