import os
pre_command="ps -ef | grep train.py | awk '{print $2}' | xargs kill -9"
os.system(pre_command)

# x_values=[5,7,10,15]
x_values=[5,7,15]
cnt=0
for x in x_values:
    # x=list(x)
    print('x',x)
    '''
    command = f"CUDA_VISIBLE_DEVICES=0 "\
                  f"python scripts/txt2img.py --n_iter 2 --n_samples 8 --W 512 --H 512 --ddim_steps 50 --plms --skip_grid " \
                  f"--from-file imagenet.txt --outdir outputs/fora/delta_{x} " \
                  f"--enable_deepcache --deepcache_cache_interval {x} " 
    '''
    command = f"CUDA_VISIBLE_DEVICES=0 "\
                  f"python scripts/txt2img.py --n_iter 2 --n_samples 8 --W 512 --H 512 --ddim_steps 50 --plms --skip_grid " \
                  f"--from-file coco_tmp.txt --enable_deepcache --deepcache_cache_interval {x} " 
    os.system(command)
