from ldm.modules.diffusionmodules.util import make_ddim_timesteps
import torch
from . import dato
from .utils import isinstance_str, init_generator
import numpy as np
import torch.nn.functional as F
import time
import math
from typing import Type, Dict, Any, Tuple, Callable
import copy
import numpy as np

my_cur_timestep=0
my_prune_strategy=[]
prev_timestep=None
cur_timestep_for_prune=0

model_timesteps=[ 1,  21,  41,  61,  81, 101, 121, 141, 161, 181, 201, 221, 241,
       261, 281, 301, 321, 341, 361, 381, 401, 421, 441, 461, 481, 501,
       521, 541, 561, 581, 601, 621, 641, 661, 681, 701, 721, 741, 761,
       781, 801, 821, 841, 861, 881, 901, 921, 941, 961, 981]

my_pre_output=None
my_output_diff=None
class DeepCacheSDHelper(object):
    def __init__(self, num_steps, pipe=None, cache_strategy=[]):
        if pipe is not None: self.pipe = pipe
        self.num_steps = num_steps 
        self.cache_strategy=cache_strategy

    def make_ddim_timesteps(self, verbose=False):
        pass
        steps_out=[]
        return steps_out

    def set_params(self, skip_mode='uniform'):
        self.params = {
            'cache_strategy': self.cache_strategy,
            'skip_mode': skip_mode
        }
        
    def is_skip_step(self, block_i,block_name, blocktype="down"): # Flase: Calculate  True:skip， Cache
        self.start_timestep = self.cur_timestep if self.start_timestep is None else self.start_timestep  
        cache_strategy, skip_mode =  self.params['cache_strategy'],  self.params['skip_mode']
        if self.cur_timestep==0: return False

        cache_index = self.cur_timestep-1
        cache_block_id= cache_strategy[cache_index]
        if cache_block_id>=2: return False
        if block_name == 'middle_block' or block_i >=cache_block_id:
            return True
        return False
        
    def wrap_unet_forward(self): 
        self.function_dict['unet_forward'] = self.pipe.forward

        def wrapped_forward(*args, **kwargs):
            global my_cur_timestep
            global my_pre_output
            global my_output_diff
            self.cur_timestep = my_cur_timestep 
            # print('**** cur_timestep',self.cur_timestep)
            my_cur_timestep+=1
            if my_cur_timestep==self.num_steps+1:
                my_cur_timestep=0
            result = self.function_dict['unet_forward'](*args, **kwargs)
            if self.cur_timestep>0: 
                my_output_diff=result-my_pre_output
                b,c,n,_=my_output_diff.shape
                my_output_diff=my_output_diff.abs()
                my_output_diff=my_output_diff.reshape(b,c,-1)
                my_output_diff=my_output_diff.mean(dim=1) # (B,N)

            else:
                my_output_diff=None
            my_pre_output=result# (B*2,4,64,64)
            
            return result

        self.pipe.forward = wrapped_forward

    def wrap_block_forward(self, block, block_name, block_i,blocktype="down"): 
        self.function_dict[
            (blocktype, block_name, block_i)
        ] = block.forward
        def wrapped_forward(*args, **kwargs):
            skip = self.is_skip_step(block_i,block_name, blocktype)
            result = self.cached_output[(blocktype, block_name, block_i)] if skip else self.function_dict[
                (blocktype, block_name, block_i)](*args, **kwargs)  
            if not skip: 
                self.cached_output[(blocktype, block_name, block_i)] = result
            return result
        block.forward = wrapped_forward
        
    def wrap_modules(self):
        # 1. wrap unet forward
        self.wrap_unet_forward()
        # 2. wrap downblock forward
        cnt=-1
        for block_i, block in enumerate(self.pipe.input_blocks):
            cnt+=1
            for _, layer in block.named_modules():
                if isinstance_str(layer, "ResBlock"):
                    self.wrap_block_forward(layer, "ResBlock", block_i)
                if isinstance_str(layer, "SpatialTransformer"):
                    self.wrap_block_forward(layer, "SpatialTransformer", block_i) 
                if isinstance_str(layer, "BasicTransformerBlock"):
                    self.wrap_block_forward(layer, "BasicTransformerBlock", block_i)
                if isinstance_str(layer, "Downsample"):
                    self.wrap_block_forward(layer, "Downsample", block_i)
                   
            self.wrap_block_forward(block, "input_blocks", block_i,  blocktype="down")
        # 3. wrap midblock forward
        self.wrap_block_forward(self.pipe.middle_block, "mid_block", 0,  blocktype="mid")
        # 4. wrap upblock forward
        block_num = len(self.pipe.output_blocks)
        for block_i, block in enumerate(self.pipe.output_blocks):
            for _, layer in block.named_modules():
                if isinstance_str(layer, "ResBlock"):
                    self.wrap_block_forward(layer, "ResBlock", cnt-block_i, blocktype="up")
                if isinstance_str(layer, "BasicTransformerBlock"):
                    self.wrap_block_forward(layer, "BasicTransformerBlock", cnt-block_i, blocktype="up")
                if isinstance_str(layer, "Upsample"):
                    self.wrap_block_forward(layer, "Upsample", cnt-block_i, blocktype="up")
                if isinstance_str(layer, "SpatialTransformer"):
                    self.wrap_block_forward(layer, "SpatialTransformer", cnt-block_i,blocktype="up")
            self.wrap_block_forward(block, "output_blocks", cnt-block_i, blocktype="up")
            
        return self.pipe

    def reset_states(self):
        self.cur_timestep = 0
        self.function_dict = {}
        self.cached_output = {}
        self.start_timestep = None
        
def isinstance_str(x: object, cls_name: str):
    """
    Checks whether x has any class *named* cls_name in its ancestry.
    Doesn't require access to the class's implementation.
    
    Useful for patching!
    """

    for _cls in x.__class__.__mro__:
        if _cls.__name__ == cls_name:
            return True
    return False


#prune_strategy: list, model_timestep: list
def select_dato_method(x: torch.Tensor, dato_info: Dict[str, Any], prune_strategy: list=[]) -> Tuple[Callable, ...]:
    args = dato_info["args"]
    current_timestep = dato_info["timestep"]
    global model_timesteps
    current_id = 49 - model_timesteps.index(current_timestep)
    
    
    global my_prune_strategy
    # !!!!!
    # current_prune_ratio = my_prune_strategy[current_id]
    current_prune_ratio=0.4
    original_h, original_w = dato_info["size"]  # 64,64
    original_tokens = original_h * original_w
    downsaple_ratio = int(math.ceil(math.sqrt(original_tokens // x.shape[1])))
    global my_output_diff
    if (downsaple_ratio <= args["max_downsample_ratio"]):  
        w = int(math.ceil(original_w / downsaple_ratio))
        h = int(math.ceil(original_h / downsaple_ratio))
       
        num_prune = int(x.shape[1] * current_prune_ratio) # new method 
        p, r = dato.prune_and_recover_tokens(x, num_prune, w=w, h=h, sx=args['sx'], sy=args['sy'],
                                             noise_alpha=args['noise_alpha'],sim_beta=args['sim_beta'],
                                             current_timestep=current_timestep,local_random=args['local_random'],
                                             align_cfg=args['align_cfg'],diff_gema=args['diff_gema'],
                                             output_diff=my_output_diff)

    else:
        p, r = (dato.do_nothing, dato.do_nothing)
    p_a, r_a = (p, r) if args["prune_selfattn_flag"] else (dato.do_nothing, dato.do_nothing)
    p_c, r_c = (p, r) if args["prune_crossattn_flag"] else (dato.do_nothing, dato.do_nothing)
    p_m, r_m = (p, r) if args["prune_mlp_flag"] else (dato.do_nothing, dato.do_nothing)
    return p_a, p_c, p_m, r_a, r_c, r_m


def make_dato_block(block_class: Type[torch.nn.Module]) -> Type[torch.nn.Module]:
    class datoBlock(block_class):
        _parent = block_class
        def _forward(self, x: torch.Tensor, context: torch.Tensor = None) -> torch.Tensor:
            p_a, p_c, p_m, r_a, r_c, r_m = select_dato_method(x, self._dato_info)  # 选择方法
            # self-attention
            prune_x = p_a(self.norm1(x))
            out1 = self.attn1(prune_x, context=context if self.disable_self_attn else None)
            x = r_a(out1) + x
            # cross-attention
            prop_x = p_c(self.norm2(x))
            out2= self.attn2(prop_x, context=context)
            x = r_c(out2) + x
            # MLP
            x = r_m(self.ff(p_m(self.norm3(x)))) + x
            return x
    return datoBlock


def make_diffusers_dato_block(block_class: Type[torch.nn.Module]) -> Type[torch.nn.Module]:
    """
    Make a patched class for a diffusers model.
    This patch applies ToMe to the forward function of the block.
    """

    class datoBlock(block_class):  # 继承自<class 'diffusers.models.attention.BasicTransformerBlock'>
        # Save for unpatching later
        _parent = block_class

        def forward(
                self,
                hidden_states,
                attention_mask=None,
                encoder_hidden_states=None,
                encoder_attention_mask=None,
                timestep=None,
                cross_attention_kwargs=None,
                class_labels=None,
        ) -> torch.Tensor:
            # (1) dato
            # 设置 dato tokens 和 recover tokens 对应的函数
            p_a, p_c, p_m, r_a, r_c, r_m = select_dato_method(hidden_states, self._dato_info)

            if self.use_ada_layer_norm:
                norm_hidden_states = self.norm1(hidden_states, timestep)
            elif self.use_ada_layer_norm_zero:
                norm_hidden_states, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.norm1(
                    hidden_states, timestep, class_labels, hidden_dtype=hidden_states.dtype
                )
            else:
                norm_hidden_states = self.norm1(hidden_states)

            # (2) dato p_a
            norm_hidden_states = p_a(norm_hidden_states)
            # 1. Self-Attention
            cross_attention_kwargs = cross_attention_kwargs if cross_attention_kwargs is not None else {}
            attn_output = self.attn1(
                norm_hidden_states,
                encoder_hidden_states=encoder_hidden_states if self.only_cross_attention else None,
                attention_mask=attention_mask,
                **cross_attention_kwargs,
            )
            if self.use_ada_layer_norm_zero:
                attn_output = gate_msa.unsqueeze(1) * attn_output

            # (3) dato r_a
            hidden_states = r_a(attn_output) + hidden_states

            if self.attn2 is not None:
                norm_hidden_states = (
                    self.norm2(hidden_states, timestep) if self.use_ada_layer_norm else self.norm2(hidden_states)
                )
                # (4) dato p_c
                norm_hidden_states = p_c(norm_hidden_states)

                # 2. Cross-Attention
                attn_output = self.attn2(
                    norm_hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    attention_mask=encoder_attention_mask,
                    **cross_attention_kwargs,
                )
                # (5) dato r_c
                hidden_states = r_c(attn_output) + hidden_states

            # 3. Feed-forward
            norm_hidden_states = self.norm3(hidden_states)

            if self.use_ada_layer_norm_zero:
                norm_hidden_states = norm_hidden_states * (1 + scale_mlp[:, None]) + shift_mlp[:, None]

            # (6) dato p_m
            norm_hidden_states = p_m(norm_hidden_states)

            ff_output = self.ff(norm_hidden_states)

            if self.use_ada_layer_norm_zero:
                ff_output = gate_mlp.unsqueeze(1) * ff_output

            # (7) dato r_m
            hidden_states = r_m(ff_output) + hidden_states

            return hidden_states

    return datoBlock


def hook_dato_model(model: torch.nn.Module):
    def hook(module, args):
        module._dato_info["size"] = (args[0].shape[2], args[0].shape[3])
        module._dato_info["timestep"] = args[1][0].cpu().item()
        return None
    model._dato_info["hooks"].append(model.register_forward_pre_hook(hook))



def apply_patch(
        model: torch.nn.Module,
        ddim_steps: int = 50, 
        ddpm_steps:int = 1000, 
        ddim_discr_method='uniform',
        enable_dato: bool=True,
        #prune_ratio: float = 0.7,
        max_downsample_ratio: int = 1,
        prune_selfattn_flag: bool = True,
        prune_crossattn_flag: bool = False,
        prune_mlp_flag: bool = False,
        sx: int = 2, sy: int = 2,
        local_random: bool=True,
        align_cfg: bool= True,
        noise_alpha:float = 0.1,
        sim_beta:float = 1, 
        diff_gema:float=0.1,
        prune_strategy: list=[],
        # cache
        enable_deepcache: bool=False,   
        cache_strategy=[]
):
    global my_prune_strategy
    my_prune_strategy=prune_strategy
    remove_patch(model)

    is_diffusers = isinstance_str(model, "DiffusionPipeline") or isinstance_str(model, "ModelMixin")
    if not is_diffusers:
        print("Not diffusers!")
        if not hasattr(model, "model") or not hasattr(model.model, "diffusion_model"):
            raise RuntimeError("Provided model was not a Stable Diffusion / Latent Diffusion model, as expected.")
        diffusion_model = model.model.diffusion_model
    else:
        diffusion_model = model.unet if hasattr(model, "unet") else model
    
    if enable_deepcache:
        helper = DeepCacheSDHelper(pipe=diffusion_model, num_steps=ddim_steps, cache_strategy=cache_strategy)
        
        helper.reset_states()
        helper.set_params()
        diffusion_model=helper.wrap_modules() 
    
    if enable_dato: 
        diffusion_model._dato_info = {
            "name": None,
            "size": None,
            "timestep": None,
            "hooks": [],
            "args": {
                "prune_selfattn_flag": prune_selfattn_flag,
                "prune_crossattn_flag": prune_crossattn_flag,
                "prune_mlp_flag": prune_mlp_flag,
                "max_downsample_ratio": max_downsample_ratio,
                "sx": sx, "sy": sy,
                "noise_alpha":noise_alpha,
                "sim_beta":sim_beta,
                "local_random":local_random,
                "align_cfg":align_cfg,
                "diff_gema":diff_gema
            }
        }
        hook_dato_model(diffusion_model)
        for x, module in diffusion_model.named_modules():
            if isinstance_str(module, "BasicTransformerBlock"):
                make_dato_block_fn = make_diffusers_dato_block if is_diffusers else make_dato_block
                module.__class__ = make_dato_block_fn(
                    module.__class__)
                module._dato_info = diffusion_model._dato_info
                module._myname = x
                if not hasattr(module, "disable_self_attn") and not is_diffusers:
                    module.disable_self_attn = False
                if not hasattr(module, "use_ada_layer_norm_zero") and is_diffusers:
                    module.use_ada_layer_norm = False
                    module.use_ada_layer_norm_zero = False
    return model

def remove_patch(model: torch.nn.Module):
    """ Removes a patch from a dato Diffusion module if it was already patched. """
    model = model.unet if hasattr(model, "unet") else model
    for _, module in model.named_modules():
        if hasattr(module, "_dato_info"):
            for hook in module._dato_info["hooks"]:
                hook.remove()
            module._dato_info["hooks"].clear()

        if module.__class__.__name__ == "datoBlock":
            module.__class__ = module._parent

    return model
