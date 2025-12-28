import os
import subprocess
import numpy as np
from pytorch_fid import fid_score
import logging
from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.bitflip import BitflipMutation
from pymoo.operators.sampling.rnd import IntegerRandomSampling
from pymoo.optimize import minimize
from datetime import datetime
from txt2img_search_new import txt2img_gen
import re
import math
import shutil
from pymoo.operators.repair.rounding import RoundingRepair
from pymoo.operators.mutation.pm import PM
import random
from pymoo.core.sampling import Sampling
import numpy as np
import glob
import ImageReward as RM
import torch

# CUDA_VISIBLE_DEVICES=0 python scripts/search_new.py

log_dir = '/path/to/sd1/outputs/search_prune_log'
os.makedirs(log_dir, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(log_dir, f'{timestamp}.txt')

logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


counter = 0
def calculate_fid_and_cache_id(prune_strategy):
    global counter
    outdir = f"/path/to/sd1/outputs/search_prune/{counter}"
    os.makedirs(outdir, exist_ok=True)
    print("%%%%%%%%%%%%%%%%")
    print(prune_strategy)
    
    input_file="/dockerdata/yuuweizhang/projects/sd1/coco_tmp.txt"
    output_file=f"/path/to/sd1/outputs/search_prune/{counter}.txt"
    num_lines=16
    with open(input_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    # 随机挑选指定数量的行
    selected_prompts = random.sample(lines, min(num_lines, len(lines)))  # 确保不超过文件总行数
    # 将选中的行写入新的文件
    with open(output_file, 'w', encoding='utf-8') as file:
        file.writelines(selected_prompts)
    
    logging.info(f"Counter: {counter}, Strategy: {prune_strategy}")
    sum_time=txt2img_gen(
    outdir=outdir,
    ddim_steps=50,
    n_samples=4,
    n_iter=1,
    H=512,W=512,
    from_file=output_file,
    # cache
    enable_deepcache=True,
    cache_strategy=[0, 1, 1, 1, 1, 1, 11, 1, 1, 1, 1, 1, 1, 1, 0, 0, 11, 1, 1, 1, 0, 11, 1, 0, 1, 1, 1, 1, 1, 11, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 11, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    # Reduce Token
    enable_dato=True,
    prune_selfattn_flag=True,
    local_random=False,
    align_cfg=True,
    prune_ratio=prune_strategy,
    noise_alpha=0.1,
    sim_beta=1,
    diff_gema=0.1)
    
    real_images_path = '/path/to/datasets/imagenet50k.npz'
    generated_images_path = os.path.join(outdir, 'samples')
    image_paths = glob.glob(os.path.join(generated_images_path, '*.[pjg][np]*'))  # 匹配 jpg, jpeg, png
    image_rewards=0.0
    
    for index in range(len(image_paths)):
        score = model.score(selected_prompts[index], img_list[index])
        image_rewards+=score
    rewards=image_rewards/len(image_paths)
    # rewards = model.score(from_file, image_paths)
    '''
    fid_paths = [real_images_path, generated_images_path]
    
    try:
        fid = fid_score.get_fid_score(fid_paths)
    except Exception as e:
        print(f"Error during FID calculation: {e}")
        fid = np.inf
    # Check if fid > 26 and delete generated_images_path if true
    global max_fid
    max_fid = 26.5
    if fid > max_fid:
        print(f"FID score {fid} exceeded threshold. Deleting {generated_images_path} and its contents.")
        shutil.rmtree(generated_images_path, ignore_errors=True)
    if fid<max_fid:
        max_fid=fid
    '''
    print('============================================================')
    print('strategy', prune_strategy)
    print('counter', counter, 'ImageReward:', rewards, 'AvgTime:', sum_time)
    
    logging.info(f"ImageReward: {fid}, Avg Time: {sum_time:.5f} s")
    counter += 1
    return fid, sum_time


class DiffusionOptimizationProblem(Problem): 
    def __init__(self, steps):
        super().__init__(n_var=steps, n_obj=2, n_ieq_constr=1, xl=3, xu=7,type_var=int)
        self.steps = steps
        self.valid_ids = [0.3,0.4,0.5,0.6,0.7]  # 允许的策略值
        self.generated_strategies = set()  # 存储已生成的策略组合
        self.encoding_map = {3: 0.3, 4: 0.4, 5: 0.5, 6:0.6, 7:0.7}

    def _decode(self, X_encoded):
        return np.vectorize(self.encoding_map.get)(X_encoded)
    
    def _evaluate(self, X, out, *args, **kwargs):
        # 解码 X
        X_decoded = self._decode(X)
        population_size = X.shape[0]
        fid_values = np.zeros(population_size)
        total_times = np.zeros(population_size)
        
        for i in range(population_size):
            strategy = X_decoded[i]
            fid, sum_time = calculate_fid_and_cache_id(strategy)
            fid_values[i] = fid
            total_times[i] = sum_time

        # 定义目标函数
        out["F"] = np.column_stack([(fid_values - 25)*40, 2.24/total_times])
        # out["G"] = fid_values - 25.7
        out["G"] = -fid_values  

steps = 50
problem = DiffusionOptimizationProblem(steps=steps)

algorithm = NSGA2(
    pop_size=25,
    sampling=IntegerRandomSampling(),
    crossover=SBX(prob=0.7, eta=7, vtype=float, repair=RoundingRepair()),
    mutation=PM(prob=0.4, eta=15, vtype=float, repair=RoundingRepair()),
    eliminate_duplicates=True
)

termination = ('n_gen', 100)

res = minimize(problem,
               algorithm,
               termination,
               seed=1,
               verbose=True)
result = res.X.astype(int)

print("Best Strategy: \n", result)
print("FID & Time: \n", res.F)
logging.info(f"Best Strategy: {result}")
logging.info(f"FID and Time: {res.F}")
