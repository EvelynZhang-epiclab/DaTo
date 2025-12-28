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
from pymoo.core.sampling import Sampling
import numpy as np

# CUDA_VISIBLE_DEVICES=1 python scripts/search_depth_v1.py
root_dir= '/dockerdata/yuuweizhang/projects/sd1/outputs/search_depth/v1'
log_dir = os.path.join(root_dir, 'search_prune_log')
os.makedirs(log_dir, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(log_dir, f'{timestamp}.txt')

logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class ConstrainedSampling(Sampling):
    def __init__(self, steps, max1,max2,max3):
        super().__init__()
        self.steps = steps
        self.max1=max1
        self.max2=max2
        self.max3=max3
        self.allowed_values = [0, 1, 12]

    def _do(self, problem, n_samples, **kwargs):
        X = []
        for _ in range(n_samples):
            while True:
                strategy = self.generate_individual()
                if strategy is not None:
                    X.append(strategy)
                    break
        return np.array(X)
    
    def generate_individual(self):
        # 初始化策略
        strategy = []
        
        max1 = np.random.randint(1,self.max1)
        # 第一层：15个元素，最多5个12
        layer1 = [12]*max1 + [1]*(15 - max1)
        random.shuffle(layer1)
        for i in range(len(layer1)-1):
            if layer1[i] == 12:
                flag = random.randint(0,1)
                if flag:
                    layer1[i+1] = 0
        strategy.extend(layer1)
        
        # 第二层：20个元素，最多3个12，并将12后一个元素设为0
        max2 = np.random.randint(1,self.max2)
        layer2 = [12]*max2 + [1]*(20 - max2)
        random.shuffle(layer2)
        for i in range(len(layer2)-1):
            if layer2[i] == 12:
                layer2[i+1] = 0
        strategy.extend(layer2)
        
        # 第三层：15个元素，最多5个12
        max3 = random.randint(1,self.max3)
        layer3 = [12]*max3 + [1]*(15 - max3)
        random.shuffle(layer3)
        for i in range(len(layer3)-1):
            if layer3[i] == 12:
                flag = random.randint(0,1)
                if flag:
                    layer3[i+1] = 0
        strategy.extend(layer3)
        
        # 将策略映射到编码值
        encoding_map_reverse = {0: 0, 1: 1, 12: 2}
        encoded_strategy = [encoding_map_reverse[val] for val in strategy]
        
        return encoded_strategy

counter = 0
def calculate_fid_and_cache_id(cache_strategy):
    global counter
    outdir = os.path.join(root_dir, f'images/{counter}')
    os.makedirs(outdir, exist_ok=True)
    print("%%%%%%%%%%%%%%%%")
    print(cache_strategy)
    
    input_file="/dockerdata/yuuweizhang/projects/sd1/imagenet.txt"
    output_file=os.path.join(root_dir, f'prompts/{counter}.txt')
    os.makedirs(os.path.join(root_dir, 'prompts'), exist_ok=True)
    num_lines=32
    with open(input_file, 'r', encoding='utf-8') as file:
        lines = file.readlines()
    # 随机挑选指定数量的行
    selected_prompts = random.sample(lines, min(num_lines, len(lines)))  # 确保不超过文件总行数
    # 将选中的行写入新的文件
    with open(output_file, 'w', encoding='utf-8') as file:
        file.writelines(selected_prompts)
    
    logging.info(f"Counter: {counter}, Strategy: {cache_strategy}")
    sum_time=txt2img_gen(
    outdir=outdir,
    ddim_steps=50,
    n_samples=4,
    n_iter=1,
    H=512,W=512,
    from_file=output_file,
    # cache
    enable_deepcache=True,
    cache_strategy=cache_strategy,
    # Reduce Token
    enable_dato=False,
    prune_selfattn_flag=True,
    local_random=False,
    align_cfg=True,
    prune_ratio=[],
    noise_alpha=0.1,
    sim_beta=1,
    diff_gema=0.1)
    
    generated_images_path = os.path.join(outdir, 'samples')
    image_paths = glob.glob(os.path.join(generated_images_path, '*.[pjg][np]*'))  # 匹配 jpg, jpeg, png
    image_rewards=0.0
    model = RM.load("ImageReward-v1.0")
    with torch.no_grad():
        for index in range(len(image_paths)):
            score = model.score(selected_prompts[index], image_paths[index])
            image_rewards+=score
    rewards=image_rewards/len(image_paths)
    # rewards = model.score(from_file, image_paths)
    shutil.rmtree(generated_images_path, ignore_errors=True)
        
    print('============================================================')
    print('strategy', cache_strategy)
    print('counter', counter, 'ImageReward:', rewards, 'AvgTime:', sum_time)
    
    logging.info(f"ImageReward: {rewards}, Avg Time: {sum_time:.5f} s")
    counter += 1
    return rewards, sum_time

class DiffusionOptimizationProblem(Problem): 
    def __init__(self, steps,cal_counts):
        super().__init__(n_var=steps, n_obj=2, n_ieq_constr=1, xl=0, xu=2, type_var=int)
        self.steps = steps
        self.cal_counts=cal_counts
        self.valid_ids = [0, 1, 12]  # 允许的策略值
        self.generated_strategies = set()  # 存储已生成的策略组合
        self.encoding_map = {0: 0, 1: 1, 2: 12}

    def _decode(self, X_encoded):
        return np.vectorize(self.encoding_map.get)(X_encoded)
    
    def _evaluate(self, X, out, *args, **kwargs):
        # 解码 X
        X_decoded = self._decode(X)
        population_size = X.shape[0]
        rewards = np.zeros(population_size)
        total_times = np.zeros(population_size)
        
        for i in range(population_size):
            strategy = X_decoded[i]
            reward, sum_time = calculate_fid_and_cache_id(strategy)
            rewards[i] = reward
            total_times[i] = sum_time

        # 定义目标函数
        out["F"] = np.column_stack([-rewards, total_times])
        # 定义约束条件
        out["G"] = -rewards
        
steps = 50
cal_counts = 5
problem = DiffusionOptimizationProblem(steps=steps,cal_counts=cal_counts)

algorithm = NSGA2(
    pop_size=20,
    sampling=ConstrainedSampling(steps=50, max1=3,max2=4,max3=3),
    crossover=SBX(prob=0.8, eta=7, vtype=float, repair=RoundingRepair()),
    mutation=PM(prob=0.6, eta=15, vtype=float, repair=RoundingRepair()),
    eliminate_duplicates=True
)

termination = ('n_gen', 50)

res = minimize(problem,
               algorithm,
               termination,
               seed=1,
               verbose=True)
result = res.X.astype(int)

print("Best Strategy: \n", result)
print("Reward & Time: \n", res.F)
logging.info(f"Best Strategy: {result}")
logging.info(f"Reward and Time: {res.F}")
