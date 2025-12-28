import os

prompt_file='/dockerdata/yuuweizhang/projects/sd1/coco30k.txt'

# 定义 x 的值列表
x_values = [0.3, 0.4, 0.5, 0.6, 0.7]#[1,2]#[0,2,3,4,5]#[5,7,10,15]# [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18] ##[4,5,6,7] # [0.3, 0.4, 0.5, 0.6, 0.7]

# 遍历每个 x 值，执行命令
for x in x_values:
    image_folder = f"/dockerdata/yuuweizhang/projects/sd2/outputs/sito/coco30k/ratio_{x}/samples"
    command = f"CUDA_VISIBLE_DEVICES=7 python clipscore.py --prompt_file {prompt_file} --image_folder {image_folder} --term 30000"
    print(f"Executing: {command}")
    os.system(command)