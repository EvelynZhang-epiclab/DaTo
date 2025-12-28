import argparse, os, sys, glob
import cv2
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from imwatermark import WatermarkEncoder
from itertools import islice
from einops import rearrange
from torchvision.utils import make_grid
import time
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import contextmanager, nullcontext
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler

import datosd

def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]

    return pil_images


def load_model_from_config(config, ckpt, verbose=False):
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        pass
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        pass
    if len(u) > 0 and verbose:
        pass
    model.cuda()
    model.eval()
    return model


def put_watermark(img, wm_encoder=None):
    if wm_encoder is not None:
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img = wm_encoder.encode(img, 'dwtDct')
        img = Image.fromarray(img[:, :, ::-1])
    return img


def load_replacement(x):
    try:
        hwc = x.shape
        y = Image.open("assets/rick.jpeg").convert("RGB").resize((hwc[1], hwc[0]))
        y = (np.array(y)/255.0).astype(x.dtype)
        assert y.shape == x.shape
        return y
    except Exception:
        return x


def check_safety(x_image):
    safety_checker_input = safety_feature_extractor(numpy_to_pil(x_image), return_tensors="pt")
    x_checked_image, has_nsfw_concept = safety_checker(images=x_image, clip_input=safety_checker_input.pixel_values)
    assert x_checked_image.shape[0] == len(has_nsfw_concept)
    for i in range(len(has_nsfw_concept)):
        if has_nsfw_concept[i]:
            x_checked_image[i] = load_replacement(x_checked_image[i])
    return x_checked_image, has_nsfw_concept


def txt2img_gen(
    prompt="a photo of a bird",
    outdir="outputs/search",
    ddim_steps=50,
    ddim_eta=0.0,
    n_iter=2,H=512,W=512,C=4,f=8,n_samples=8,n_rows=0,scale=7.5,seed=42,
    from_file="/dockerdata/yuuweizhang/projects/sd1/",
    config="/dockerdata/yuuweizhang/projects/sd1/configs/stable-diffusion/v1-inference.yaml",
    ckpt="/youtu-pangu_media_public_1511122_cq10/yuuweizhang/weight/sd1/v1-5-pruned-emaonly.ckpt",
    precision="autocast",
    enable_deepcache=True,
    cache_strategy=[],
    enable_dato=True,
    prune_selfattn_flag=True,
    local_random=False,
    align_cfg=True,
    prune_ratio=[],
    noise_alpha=0.1,
    sim_beta=1,
    diff_gema=0.1
):

    print("============INFO============")
    print("enable_deepcache", enable_deepcache)
    print("dato:", enable_dato)
    print("prune_strategy", prune_ratio)
    print("cache_strategy",cache_strategy)
    seed_everything(seed)
    config = OmegaConf.load(f"{config}")
    model = load_model_from_config(config, f"{ckpt}")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)#.half()

    datosd.apply_patch(model,ddim_steps=ddim_steps,
                       # Reduce Token
                       enable_dato=enable_dato,prune_selfattn_flag=prune_selfattn_flag,prune_strategy=prune_ratio,
                       local_random=local_random,align_cfg=align_cfg,
                       noise_alpha=noise_alpha, sim_beta=sim_beta,diff_gema=diff_gema,
                       # Deepcache
                       enable_deepcache=enable_deepcache, 
                       cache_strategy=cache_strategy)

    sampler = PLMSSampler(model)


    os.makedirs(outdir, exist_ok=True)
    outpath = outdir

    # print("Creating invisible watermark encoder (see https://github.com/ShieldMnt/invisible-watermark)...")
    wm = "StableDiffusionV1"
    wm_encoder = WatermarkEncoder()
    wm_encoder.set_watermark('bytes', wm.encode('utf-8'))

    batch_size = n_samples
    n_rows = n_rows if n_rows > 0 else batch_size
    if not from_file:
        prompt = prompt
        assert prompt is not None
        data = [batch_size * [prompt]]

    else:
        # print(f"reading prompts from {from_file}")
        with open(from_file, "r") as f_read:
            data = f_read.read().splitlines()
            data = list(chunk(data, batch_size))

    sample_path = os.path.join(outpath, "samples")
    os.makedirs(sample_path, exist_ok=True)
    base_count = len(os.listdir(sample_path))
    grid_count = len(os.listdir(outpath)) - 1

    start_code = None
    sum_time=0.0
    precision_scope = autocast if precision=="autocast" else nullcontext
    with torch.no_grad():
        with precision_scope("cuda"):
            with model.ema_scope():
                
                all_samples = list()
                
                for n in trange(n_iter, desc="Sampling"):
                    for prompts in tqdm(data, desc="data"):
                        time0= torch.cuda.Event(enable_timing=True)
                        time1= torch.cuda.Event(enable_timing=True)
                        time0.record()
                        uc = None
                        if scale != 1.0:
                            uc = model.get_learned_conditioning(batch_size * [""])
                        if isinstance(prompts, tuple):
                            prompts = list(prompts)
                        c = model.get_learned_conditioning(prompts)
                        # print(f"C: {C}, H: {H}, W: {W},f: {f}")
                        # print(f"Types: C: {type(C)}, H: {type(H)}, W: {type(W)},f: {type(f)}")

                        shape = [C, H // f, W // f]
                        samples_ddim, _ = sampler.sample(S=ddim_steps,
                                                         conditioning=c,
                                                         batch_size=n_samples,
                                                         shape=shape,
                                                         verbose=False,
                                                         unconditional_guidance_scale=scale,
                                                         unconditional_conditioning=uc,
                                                         eta=ddim_eta,
                                                         x_T=start_code)
                        #print(samples_ddim)
                        x_samples_ddim = model.decode_first_stage(samples_ddim)
                        
                        x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                        x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()

                        # x_checked_image, has_nsfw_concept = check_safety(x_samples_ddim)

                        x_checked_image_torch = torch.from_numpy(x_samples_ddim).permute(0, 3, 1, 2)
                        time1.record()
                        torch.cuda.synchronize()
                        # if(n>=1):
                        sum_time+=time0.elapsed_time(time1)                       
                        for x_sample in x_checked_image_torch:
                            x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                            img = Image.fromarray(x_sample.astype(np.uint8))
                            img = put_watermark(img, wm_encoder)
                            img.save(os.path.join(sample_path, f"{base_count:05}.png"))
                            base_count += 1

    sum_time=sum_time/(1000000*n_iter)
    # print("sum time is:",sum_time)
    # print(f"Your samples are ready and waiting for you here: \n{outpath} \n"f" \nEnjoy.")
    print('SumTime:',sum_time)
    return sum_time
