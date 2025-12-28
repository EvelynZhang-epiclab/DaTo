# DaTo
The offical repo for [Token Pruning for Caching Better: 9× Acceleration on Stable Diffusion for Free](https://arxiv.org/pdf/2501.00375).

This repository contains the core code for DaTo on Stable Diffusion 1.x, including:
- Dynamic token pruning and attention patching
- Multi-objective search for depth-wise cache strategies
- Multi-objective search for per-step pruning ratios

> **Note:** This is research code. Many paths in the scripts (e.g., datasets, output folders) are placeholders and should be modified to match your environment.

---

## 1. Environment Setup

### 1.1 Requirements

- Python 3.8+
- PyTorch with CUDA
- `pymoo` for multi-objective optimization
- `pytorch-fid` for FID computation (optional)
- `ImageReward` for image–text reward scoring
- Stable Diffusion 1.x checkpoint



---

## 2. Project Structure

Main DaTo-related components:

```text
sd1/
  scripts/
    datosd/
      __init__.py
      dato.py        # Core DaTo logic: token pruning, integration with SD
      patch.py       # Patching/hooking Stable Diffusion modules
      utils.py       # Helper utilities
    search_depth.py  # Search over cache strategies across timesteps
    search_ratio.py  # Search over per-step pruning ratios
```

- **Core implementation**  
  The directory [`sd1/scripts/datosd`](https://github.com/EvelynZhang-epiclab/DaTo/tree/main/sd1/scripts/datosd) contains the main DaTo implementation:
  - [`dato.py`](https://github.com/EvelynZhang-epiclab/DaTo/blob/main/sd1/scripts/datosd/dato.py): core DaTo logic (dynamic token pruning/selection and integration with the diffusion sampling process).
  - [`patch.py`](https://github.com/EvelynZhang-epiclab/DaTo/blob/main/sd1/scripts/datosd/patch.py): patches / hooks for Stable Diffusion modules, used to inject token policies and cache strategies.
  - [`utils.py`](https://github.com/EvelynZhang-epiclab/DaTo/blob/main/sd1/scripts/datosd/utils.py): helper functions (e.g., strategy processing, mask construction).

- **Search scripts**
  - [`search_depth.py`](https://github.com/EvelynZhang-epiclab/DaTo/blob/main/sd1/scripts/search_depth.py):  
    Uses **NSGA-II** to search over depth-wise **cache strategies** for each diffusion step. It jointly optimizes:
    - image quality (via **ImageReward**), and  
    - inference time (average generation time).
  - [`search_ratio.py`](https://github.com/EvelynZhang-epiclab/DaTo/blob/main/sd1/scripts/search_ratio.py):  
    Uses **NSGA-II** to search over per-step **pruning ratios**, balancing:
    - FID / ImageReward quality, and  
    - speed-up.

---

## 3. Core Method: DaTo

**DaTo (Dynamic Token Optimization)** reduces redundant tokens during the diffusion sampling process to accelerate inference while maintaining generation quality.

At a high level:
- DaTo applies dynamic token pruning on the relevant attention layers of Stable Diffusion.
- It can be combined with caching strategies to reuse intermediate features at certain timesteps.
- The actual token and cache strategies are obtained from the multi-objective search procedures implemented in the search scripts.

Core logic:
- [`sd1/scripts/datosd/dato.py`](https://github.com/EvelynZhang-epiclab/DaTo/blob/main/sd1/scripts/datosd/dato.py)
- [`sd1/scripts/datosd/patch.py`](https://github.com/EvelynZhang-epiclab/DaTo/blob/main/sd1/scripts/datosd/patch.py)

---

## 4. Search Scripts

### 4.1 Depth-wise Cache Strategy Search

Script: [`sd1/scripts/search_depth.py`](https://github.com/EvelynZhang-epiclab/DaTo/blob/main/sd1/scripts/search_depth.py)

This script uses `pymoo`’s **NSGA-II** algorithm to search over a length‑`steps` (default 50) cache strategy vector. Each dimension encodes a cache behavior for a specific diffusion step.

For each candidate strategy, the script:
1. Randomly samples a set of prompts from a text file.
2. Calls `txt2img_gen` to generate images with:
   - deep cache enabled,
   - a specific `cache_strategy`.
3. Computes the **ImageReward** score and average inference time.
4. Uses NSGA-II to evolve the population and obtain a Pareto front over:
   - **maximizing ImageReward**, and  
   - **minimizing inference time**.

Example usage (adapt `root_dir`, dataset paths, etc., inside the script):

```bash
CUDA_VISIBLE_DEVICES=0 python sd1/scripts/search_depth.py
```

The final result contains:
- The best (or Pareto) strategies `res.X`
- Corresponding objective values `res.F` (quality vs. time)

These strategies can then be used as the cache configuration in DaTo.

### 4.2 Pruning Ratio Search

Script: [`sd1/scripts/search_ratio.py`](https://github.com/EvelynZhang-epiclab/DaTo/blob/main/sd1/scripts/search_ratio.py)

This script searches for a length‑`steps` (default 50) **pruning ratio** vector. The ratios are encoded as discrete integers (e.g., 3–7) and then mapped to real-valued pruning rates (e.g., 0.3–0.7).

For each candidate pruning strategy, the script:
1. Randomly samples prompts from a text file.
2. Calls `txt2img_gen` with:
   - deep cache enabled,
   - `enable_dato=True`,
   - `prune_ratio` set to the candidate strategy.
3. Evaluates a combination of quality (e.g., FID or ImageReward) and speed.

The objective typically includes:
- a term related to FID / quality, and
- a term related to inverse runtime.

Example usage (again, please adapt the paths inside the script):

```bash
CUDA_VISIBLE_DEVICES=0 python sd1/scripts/search_ratio.py
```

The resulting `res.X` contains candidate pruning ratio schedules that can be plugged into DaTo for deployment.

---


## 5. Citation

If you find this repository useful, please consider citing:

```bibtex
@article{zhang2024token,
  title={Token pruning for caching better: 9 times acceleration on stable diffusion for free},
  author={Zhang, Evelyn and Xiao, Bang and Tang, Jiayi and Ma, Qianli and Zou, Chang and Ning, Xuefei and Hu, Xuming and Zhang, Linfeng},
  journal={arXiv preprint arXiv:2501.00375},
  year={2024}
}
```
