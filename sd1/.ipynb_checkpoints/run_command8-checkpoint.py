import os
pre_command="ps -ef | grep train.py | awk '{print $2}' | xargs kill -9"
os.system(pre_command)

x_values=[
[1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 12, 1, 0, 0, 0, 1, 1, 0, 1, 0, 12, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 12, 0, 0, 0, 0, 0, 0, 1, 0, 1, 12, 0, 1, 0, 1]
]

cnt=0
for x in x_values:
    
    # x=list(x)
    print('x',x)
    command = f"CUDA_VISIBLE_DEVICES=7 "\
                  f"python scripts/eval_strategy.py --n_iter 1 --n_samples 8 --W 512 --H 512 --ddim_steps 50 --plms --skip_grid " \
                  f"--from-file coco30k.txt --outdir outputs/eval_depth/search_old/coco30k " \
                  f"--enable_deepcache --cache_strategy '{x}' " \
                  f"--enable_dato --prune_selfattn_flag --prune_strategy 0.4 --local_random"
    os.system(command)
