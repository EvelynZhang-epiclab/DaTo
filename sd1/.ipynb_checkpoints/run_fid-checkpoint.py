import os

# 定义路径
fid_script = "pytorch_fid"
# path1 = "/dockerdata/yuuweizhang/datasets/imagenet50k.npz"
path1 = "/dockerdata/yuuweizhang/datasets/coco30k.npz"
# 定义 x 的值列表
x_values = [0.3, 0.4, 0.5, 0.6, 0.7]#[1,2]#[0,2,3,4,5]#[5,7,10,15]# [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18] ##[4,5,6,7] # [0.3, 0.4, 0.5, 0.6, 0.7]

# 遍历每个 x 值，执行命令
for x in x_values:
    # path2 = f"/dockerdata/yuuweizhang/projects/DiT/samples/eval_strategy/strategy_v{x}/DiT-XL-2-pretrained-size-256-vae-ema-cfg-1.5-seed-0"
    path2 = f"/dockerdata/yuuweizhang/projects/sd2/outputs/sito/coco30k/ratio_{x}_noise_0.1_sim_1_aligncfg/samples"
    command = f"CUDA_VISIBLE_DEVICES=6 python -m {fid_script} {path1} {path2}"
    print(f"Executing: {command}")
    os.system(command)
