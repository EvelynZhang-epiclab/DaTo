import time
import matplotlib.pyplot as plt
import numpy as np
import torch
from typing import Tuple, Callable, Optional, Dict, Any
import math
from .utils import init_generator
import random


def do_nothing(x: torch.Tensor, mode: str = None):
    return x

# 初始化 count_num 张量为全零，大小为 (B, N)
count_num = torch.zeros((2, 4096), dtype=torch.int32, device="cuda")

def do_nothing(x: torch.Tensor, mode: str = None):
    return x

def find_patch_max_indices(tensor, sx, sy):  # tensor: (B,N)
    b, N = tensor.size()
    n = int(math.sqrt(N))
    tensor = tensor.view(b, n, n)
    h_patches = n // sy
    w_patches = n // sx
    tensor = tensor[:, :h_patches * sy, :w_patches * sx]
    tensor_reshaped = tensor.view(b, h_patches, sy, w_patches, sx)
    tensor_reshaped = tensor_reshaped.permute(0, 1, 3, 2, 4).contiguous()
    tensor_reshaped = tensor_reshaped.view(b, h_patches, w_patches, sy * sx)
    _, max_indices = tensor_reshaped.max(dim=-1, keepdim=True)
    return max_indices

def duplicate_half_tensor(tensor):
    B, _, _,_ = tensor.shape
    # 检查 B 是否为偶数
    if B % 2 != 0:
        raise ValueError("B must be even for this operation")
    # 前 1/2B 的部分
    first_half = tensor[:B // 2]
    # 复制前 1/2B 的部分到后 1/2B
    tensor[B // 2:] = first_half
    return tensor

def prune_and_recover_tokens(metric: torch.Tensor,
                                 num_prune: int,
                                 w: int,
                                 h: int,
                                 sx: int,
                                 sy: int,
                                 sim_beta: float,
                                 noise_alpha: float,
                                 diff_gema:float,
                                 current_timestep: int,
                                 local_random: bool,
                                 align_cfg: bool,
                                 output_diff: torch.Tensor
                                 ) -> Tuple[Callable, Callable]:
    B, N, C = metric.shape
    if num_prune <= 0:  # 如果r<0, 什么也不做
        return do_nothing, do_nothing
    gather = mps_gather_workaround if metric.device.type == "mps" else torch.gather
    with torch.no_grad():
        my_norm = metric.norm(dim=-1, keepdim=True)
        my_norm = my_norm.to(torch.float16) # (B,N,1)
        metric = metric / my_norm # (B,N,C)
        consine_graph = metric @ metric.transpose(-1, -2) #(B,N,N)

        hsy, wsx = h // sy, w // sx
        if not local_random:
            dst_score = consine_graph.sum(-1)
            noise_score= torch.randn(B, N, dtype=metric.dtype, device=metric.device)# (B,N)
            # dst_score=sim_beta*dst_score+noise_alpha*noise_score
            if random.random() < 0.5:
                diff_gema = -diff_gema
            if output_diff is not None:
                dst_score=diff_gema*output_diff+noise_alpha*noise_score
            else:
                dst_score=noise_score
            # print('*****dst_score',dst_score.shape)
            rand_idx = find_patch_max_indices(dst_score, sx, sy)  # [B,hsy,wsx,1]
            if align_cfg:
                rand_idx = duplicate_half_tensor(rand_idx)
        else:
            generator = init_generator(metric.device)
            rand_idx = torch.randint(sy * sx, size=(hsy, wsx, 1), device=generator.device, generator=generator).to(metric.device)  # [hsy, wsx, 1] 取值都为0~3的随机整数
            #rand_idx = torch.randint(sy * sx, size=(hsy, wsx, 1), device=metric.device).to(metric.device)
            rand_idx = rand_idx.unsqueeze(0).expand(B, -1, -1, -1)

        idx_buffer_view = torch.zeros(B, hsy, wsx, sy * sx, device=metric.device, dtype=torch.int64)
        idx_buffer_view.scatter_(dim=-1, index=rand_idx, src=-torch.ones_like(rand_idx, dtype=rand_idx.dtype))
        idx_buffer_view = idx_buffer_view.view(B, hsy, wsx, sy, sx).transpose(2, 3).reshape(B, hsy * sy, wsx * sx)
        if (hsy * sy) < h or (wsx * sx) < w:
            idx_buffer = torch.zeros(B, h, w, device=metric.device, dtype=torch.int64)
            idx_buffer[:, :(hsy * sy), :(wsx * sx)] = idx_buffer_view  # (B,h,w)
        else:
            idx_buffer = idx_buffer_view  # (B,h,w)
        rand_idx = idx_buffer.reshape(B, -1).argsort(dim=1)  # (B,N)
        del idx_buffer, idx_buffer_view

        num_dst = hsy * wsx
        a_idx = rand_idx[:, num_dst:]  # src  (B,N-num_dst) 0~N
        b_idx = rand_idx[:, :num_dst]  # dst  (B,num_dst) 0~N

        score_sim_1 = gather(consine_graph, dim=1, index=a_idx.unsqueeze(-1).expand(B, -1, N))  # (B,N-num_dst,N)
        score_sim = gather(score_sim_1, dim=-1,
                           index=b_idx.unsqueeze(1).expand(B, score_sim_1.shape[1], -1))  # (B,N-num_dst,num_dst)
        score_sim_value, score_sim_index = torch.max(score_sim, dim=2)  # (B,N-num_dst) 0~num_dst
        total_score = - score_sim_value

        edge_idx = total_score.argsort(dim=-1)  # （0~N-num_dst）
        unm_idx = edge_idx[..., num_prune:]  # (B,N-num_dst-num_prune)
        src_idx = edge_idx[..., :num_prune]  # (B,num_prune)
        a_idx_tmp = a_idx.expand(B, N - num_dst)
        a_unm_idx = gather(a_idx_tmp, dim=1, index=unm_idx)  # (B,N-num_dst-unm_prop)
        a_src_idx = gather(a_idx_tmp, dim=1, index=src_idx)  # (B,num_prune)

        combined = torch.cat((a_unm_idx, b_idx), dim=1)  # (B,N-num_prune)
        weight = gather(consine_graph, dim=1, index=combined.unsqueeze(-1).expand(B, -1, N))  # (B,N-num_prune,N)
        weight_prop = gather(weight, dim=2,
                             index=a_src_idx.unsqueeze(1).expand(-1, weight.shape[1], -1))  # (B,N-prune,num_prune)
        _, max_indices = torch.max(weight_prop, dim=1)  # (B,num_prune) 0~N-num_prop

    def prune_tokens(x: torch.Tensor) -> torch.Tensor:  # x: (B,N,C)
        B, N, C = x.shape  #
        unm = gather(x, dim=-2,
                     index=a_unm_idx.unsqueeze(2).expand(B, N - num_dst - num_prune, C))  # (B, N-num_dst-num_prune, C)
        dst = gather(x, dim=-2, index=b_idx.unsqueeze(2).expand(B, num_dst, C))  # (B,num_dst,C)
        result = torch.cat([unm, dst], dim=1)
        return result  # (B,N-num_prune,c)

    def recover_tokens(x: torch.Tensor) -> torch.Tensor:  # (B,N-num_prune,c)
        unm_len = a_unm_idx.shape[1]  # N-num_dst-num_prune
        # unm: (B, N-num_dst-num_prune,C) dst: (B,num_dst,C)
        unm, dst = x[..., :unm_len, :], x[..., unm_len:, :]
        _, _, c = unm.shape
        # weight_prop: (B,num_prune,num_dst)
        # dst: (B,num_dst,C)
        out = torch.zeros(B, N, c, device=x.device, dtype=x.dtype)
        src = torch.gather(x, 1, max_indices.unsqueeze(-1).expand(-1, -1, c))  # (B,num_prune,c)
        out.scatter_(dim=-2, index=b_idx.unsqueeze(2).expand(B, num_dst, c), src=dst)
        out.scatter_(dim=-2, index=a_unm_idx.unsqueeze(2).expand(B, unm_len, c), src=unm)
        out.scatter_(dim=-2, index=a_src_idx.unsqueeze(2).expand(B, num_prune, c), src=src)
        return out  # (B,N,C)

    return prune_tokens, recover_tokens

