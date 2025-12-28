import os

# 设置文件夹路径
folder_path = '/dockerdata/yuuweizhang/projects/sd1/outputs/tome/coco30k/ratio_0.6/samples'  # 替换为你的文件夹路径

# 初始化计数器
png_count = 0

# 遍历文件夹中的文件
for filename in os.listdir(folder_path):
    if filename.endswith('.png'):
        png_count += 1

print(f'文件夹中共有 {png_count} 张 .png 图片。')

# 文件路径和保留的行数
file_path = 'coco30k_tome1.txt'  # 替换为你的文件路径
n = 30000-png_count  # 替换为你想要保留的行数

# 读取文件的所有行
with open(file_path, 'r', encoding='utf-8') as file:
    lines = file.readlines()

# 计算需要保留的行
lines_to_keep = lines[-n:]  # 获取文件的最后 n 行

# 将这些行写回到文件
with open(file_path, 'w', encoding='utf-8') as file:
    file.writelines(lines_to_keep)
