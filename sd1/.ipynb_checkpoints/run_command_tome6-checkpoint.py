import os
pre_command="ps -ef | grep train.py | awk '{print $2}' | xargs kill -9"
os.system(pre_command)

import os
ratios=[0.6]
for x in ratios:
    # 定义命令
    command = f"CUDA_VISIBLE_DEVICES=4 "\
              f"python scripts/txt2img_tome.py --n_iter 1 --n_samples 4 --W 512 --H 512 --ddim_steps 50 --plms --skip_grid " \
              f"--from-file coco30k_tome1.txt --outdir outputs/tome/coco30k/ratio_{x} --prune_ratio {x}" 
    
    # 运行命令
    os.system(command)