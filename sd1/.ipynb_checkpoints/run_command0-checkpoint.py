import os
pre_command="ps -ef | grep train.py | awk '{print $2}' | xargs kill -9"
os.system(pre_command)

ratios=[0.3,0.4,0.5,0.6,0.7]
for x in ratios:
    # 定义命令
    command = f"CUDA_VISIBLE_DEVICES=2 "\
              f"python scripts/txt2img.py --n_iter 2 --n_samples 4 --W 512 --H 512 --ddim_steps 50 --plms --skip_grid " \
              f"--from-file imagenet.txt --outdir outputs/sito/imagenet/ratio_{x}_noise_0.1_sim_1_aligncfg " \
              f"--enable_deepcache --deepcache_cache_interval 1 --cache_block_id_full 13 --cache_block_id_sub 13 " \
              f"--enable_sito --prune_selfattn_flag --align_cfg --prune_ratio {x} --diff_gema 0 --noise_alpha 0.1 --sim_beta 1"
    
    # 运行命令
    os.system(command)
    
'''
import os
pre_command="ps -ef | grep train.py | awk '{print $2}' | xargs kill -9"
os.system(pre_command)

x_values=[
    [1, 1, 1, 2, 0, 1, 1, 2, 1, 1, 2, 1, 2, 1, 1, 1, 1, 2, 0, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 0, 1, 2, 2, 2],
    [1, 1, 1, 1, 1, 1, 2, 0, 2, 0, 1, 1, 1, 1, 1, 1, 1, 2, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 2, 0, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 0, 1, 0, 1, 1, 1, 1, 2, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 1, 1, 1, 2, 0, 2, 0, 1, 1, 1, 1, 1, 1, 1, 2, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 0, 1, 1, 1, 2, 0, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 2, 0, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 2, 0, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 2, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 0, 1, 0, 0, 1, 1, 1, 2, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 1, 1, 2, 0, 1, 1, 2, 1, 1, 2, 1, 2, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 2, 0, 1, 2, 2, 2],
    [1, 1, 1, 2, 0, 1, 1, 1, 2, 1, 2, 1, 2, 1, 1, 1, 0, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1],
    [1, 1, 1, 2, 0, 1, 1, 2, 2, 1, 2, 1, 2, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 2, 2],
    [1, 1, 1, 2, 0, 1, 1, 1, 2, 1, 2, 1, 2, 1, 1, 1, 0, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 1, 1]
]
cnt=0
for x in x_values:
    if (cnt>=9):
        # x=list(x)
        print('x',x)
        # 定义命令
        command = f"CUDA_VISIBLE_DEVICES=0 "\
                  f"python scripts/eval_strategy.py --n_iter 2 --n_samples 8 --W 512 --H 512 --ddim_steps 50 --plms --skip_grid " \
                  f"--from-file imagenet.txt --outdir outputs/eval_depth/search_v3/strategy{cnt} " \
                  f"--enable_deepcache --cache_strategy '{x}' "
        
        # 运行命令
        os.system(command)
    cnt+=1
'''