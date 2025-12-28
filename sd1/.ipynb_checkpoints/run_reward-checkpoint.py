
import os
x_values=[0.6]#[0.3,0.4,0.5,0.6,0.7]

for x in x_values:
    input_path=f"/dockerdata/yuuweizhang/projects/sd2/outputs/tome/coco30k/ratio_{x}/samples"
    input_file="/dockerdata/yuuweizhang/projects/sd2/coco30k.txt"
    command=f"CUDA_VISIBLE_DEVICES=0  python image_reward.py --image_folder {input_path} --batch_size 8 --prompt_source 'prompt-number' --prompt_txt {input_file} "
    os.system(command)
    print(input_path)
    print('*******')

'''
import os
import torch
import ImageReward as RM
import glob

if __name__ == "__main__":
    x_values= [0.4,0.5,0.6,0.7]# [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]
    file_path = '/dockerdata/yuuweizhang/projects/sd1/outputs/tome/coco30k/image_reward.txt'
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    for x in x_values:
        input_path=f"/dockerdata/yuuweizhang/projects/sd1/outputs/tome/coco30k/ratio_{x}/samples"
        input_file="/dockerdata/yuuweizhang/projects/sd1/coco30k.txt"
        
        img_list= glob.glob(os.path.join(input_path, '*.[pjg][np]*'))
        
        with open(input_file, 'r', encoding='utf-8') as file:
            prompts = file.readlines()
        prompt_list=prompts+prompts
        model = RM.load("ImageReward-v1.0")
        
        with torch.no_grad():
            image_rewards=0.0
            
            for index in range(len(img_list)):
                score = model.score(prompt_list[index], img_list[index])
                image_rewards+=score
            rewards=image_rewards/len(img_list)
            print('ratio ',x,': ',rewards)
            with open(file_path, 'a') as file:
                file.write(f'ratio {x}: {rewards}\n')  # 写入内容并换行
'''