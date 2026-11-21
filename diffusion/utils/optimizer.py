# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
# Some Code adapted from mmcv https://github.com/open-mmlab/mmcv modified by @ly403 

import math
from typing import Optional, Dict, List, Tuple

from torch import nn
from omegaconf import OmegaConf

from termcolor import colored
from torch.nn import GroupNorm, LayerNorm
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.modules.instancenorm import _InstanceNorm
from .logger import get_root_logger
from diffusion.utils.config import instantiate_from_config


DEFAULT_OPTIMIZER_CONSTRUCT = "diffusion.utils.optimizer.OptimizerConstructor"

def split_params(named_params, nonhidden_names=[]):
    hidden_weights = []
    hidden_gains_biases = []
    nonhidden_params = []
    for n, p in named_params:
        nonhidden = False
        for nonhidden_name in nonhidden_names:
            if nonhidden_name in n:
                nonhidden = True
                break
        if nonhidden:
            nonhidden_params.append(p)
        else:
            if p.ndim >= 2:
                hidden_weights.append(p)
            elif p.ndim < 2:
                hidden_gains_biases.append(p)
    return hidden_weights, hidden_gains_biases + nonhidden_params

class OptimizerConstructor:
    
    def __init__(
        self,
        optimizer_cfg: Dict,
        paramwise_cfg: Optional[Dict] = None,
        nonhidden_names: List[str] = []
    ):
        self.optimizer_cfg = optimizer_cfg
        self.paramwise_cfg = paramwise_cfg
        if isinstance(optimizer_cfg, list):
            self.base_lr = [optimizer_i["params"].get('lr', None) for optimizer_i in optimizer_cfg]
            self.base_wd = [optimizer_i["params"].get('weight_decay', None) for optimizer_i in optimizer_cfg]
        else:
            self.base_lr = optimizer_cfg["params"].get('lr', None)
            self.base_wd = optimizer_cfg["params"].get('weight_decay', None)
        self.nonhidden_names = nonhidden_names
    
    def add_params(self, params, module, prefix="", is_dcn_module=None):
        """Add all parameters of module to the params list.

        The parameters of the given module will be added to the list of param
        groups, with specific rules defined by paramwise_cfg.

        Args:
            params (list[dict]): A list of param groups, it will be modified
                in place.
            module (nn.Module): The module to be added.
            prefix (str): The prefix of the module

        """
        # get param-wise options
        custom_keys = self.paramwise_cfg.get("custom_keys", {})
        # first sort with alphabet order and then sort with reversed len of str
        # sorted_keys = sorted(sorted(custom_keys.keys()), key=len, reverse=True)

        bias_lr_mult = self.paramwise_cfg.get("bias_lr_mult", 1.0)
        bias_decay_mult = self.paramwise_cfg.get("bias_decay_mult", 1.0)
        norm_decay_mult = self.paramwise_cfg.get("norm_decay_mult", 1.0)
        bypass_duplicate = self.paramwise_cfg.get("bypass_duplicate", False)

        # special rules for norm layers and depth-wise conv layers
        is_norm = isinstance(module, (_BatchNorm, _InstanceNorm, GroupNorm, LayerNorm))

        for name, param in module.named_parameters(recurse=False):
            base_lr = self.base_lr
            if name == "bias" and not (is_norm or is_dcn_module):
                base_lr *= bias_lr_mult

            # apply weight decay policies
            base_wd = self.base_wd
            if self.base_wd is not None:
                # norm decay
                if is_norm:
                    base_wd *= norm_decay_mult
                # bias lr and decay
                elif name == "bias" and not is_dcn_module:
                    # TODO: current bias_decay_mult will have affect on DCN
                    base_wd *= bias_decay_mult

            param_group = {"params": [param]}
            if not param.requires_grad:
                param_group["requires_grad"] = False
                params.append(param_group)
                continue
            if bypass_duplicate and self._is_in(param_group, params):
                logger = get_root_logger()
                logger.warn(f"{prefix} is duplicate. It is skipped since " f"bypass_duplicate={bypass_duplicate}")
                continue
            # if the parameter match one of the custom keys, ignore other rules
            is_custom = False
            for key in custom_keys:
                if isinstance(key, tuple):
                    scope, key_name = key
                else:
                    scope, key_name = None, key
                if scope is not None and scope not in f"{prefix}":
                    continue
                if key_name in f"{prefix}.{name}":
                    is_custom = True
                    if "lr_mult" in custom_keys[key]:
                        # if 'base_classes' in f'{prefix}.{name}' or 'attn_base' in f'{prefix}.{name}':
                        #     param_group['lr'] = self.base_lr
                        # else:
                        param_group["lr"] = self.base_lr * custom_keys[key]["lr_mult"]
                    elif "lr" not in param_group:
                        param_group["lr"] = base_lr
                    if self.base_wd is not None:
                        if "decay_mult" in custom_keys[key]:
                            param_group["weight_decay"] = self.base_wd * custom_keys[key]["decay_mult"]
                        elif "weight_decay" not in param_group:
                            param_group["weight_decay"] = base_wd

            if not is_custom:
                # bias_lr_mult affects all bias parameters
                # except for norm.bias dcn.conv_offset.bias
                if base_lr != self.base_lr:
                    param_group["lr"] = base_lr
                if base_wd != self.base_wd:
                    param_group["weight_decay"] = base_wd
            params.append(param_group)

        for child_name, child_mod in module.named_children():
            child_prefix = f"{prefix}.{child_name}" if prefix else child_name
            self.add_params(params, child_mod, prefix=child_prefix, is_dcn_module=is_dcn_module)

    def _is_in(self, param_group: Dict, param_group_list: List) -> bool:
        assert isinstance(param_group_list, dict)
        param = set(param_group['params'])
        param_set = set()
        for group in param_group_list:
            param_set.update(set(group['params']))

        return not param.isdisjoint(param_set)

    def __call__(self, model: nn.Module):
        if hasattr(model, 'module'):
            model = model.module

        optimizer_cfg = self.optimizer_cfg.copy()
        
        if isinstance(optimizer_cfg, list) and "muon" in optimizer_cfg[0]["target"].lower():
            param_groups = split_params(list(model.named_parameters()), self.nonhidden_names)
            optimizer_muon = instantiate_from_config({
                "target": optimizer_cfg[0]["target"],
                "params": {
                    "params": param_groups[0],
                    **optimizer_cfg[0]['params']
                }
            })
            optimizer_other = instantiate_from_config({
                "target": optimizer_cfg[1]["target"],
                "params": {
                    "params": param_groups[1],
                    **optimizer_cfg[1]['params']
                }
            })
            return [optimizer_muon, optimizer_other]
            
        elif "muon" in optimizer_cfg["target"].lower():    
            param_groups = split_params(list(model.named_parameters()), self.nonhidden_names)
            adam_groups = {
                "params": param_groups[1],
                **optimizer_cfg['params']['adam_groups']
            }
            muon_group = {
                "params": param_groups[0],
                **optimizer_cfg['params']['muon_group']
            }
            return instantiate_from_config({
                    "target": optimizer_cfg["target"],
                    "params": {
                        "param_groups": [adam_groups, muon_group]
                    }
                }
            )
        else:
            # if no paramwise option is specified, just use the global setting
            if not self.paramwise_cfg:
                return instantiate_from_config({
                        "target": optimizer_cfg["target"],
                        "params": {
                            "params": model.parameters(),
                            **optimizer_cfg['params']
                        }
                    }
                )

            # set param-wise lr and weight decay recursively
            params: List[Dict] = []
            self.add_params(params, model)

            return instantiate_from_config({
                    "target": optimizer_cfg["target"],
                    "params": {
                        "params": params,
                        **optimizer_cfg['params']
                    }
                }
            )

def auto_scale_lr(effective_bs, optimizer_cfg, rule="linear", base_batch_size=256):
    assert rule in ["linear", "sqrt"]
    logger = get_root_logger()
    # scale by world size
    if rule == "sqrt":
        scale_ratio = math.sqrt(effective_bs / base_batch_size)
    elif rule == "linear":
        scale_ratio = effective_bs / base_batch_size
    optimizer_cfg["params"]["lr"] *= scale_ratio
    logger.info(f'Automatically adapt lr to {optimizer_cfg["params"]["lr"]:.5f} (using {rule} scaling rule).')
    return scale_ratio

def build_optimizer(model, optimizer_cfg, constructor_cfg={}):
    # default parameter-wise config
    logger = get_root_logger()
    optimizer_cfg = OmegaConf.to_container(optimizer_cfg)
    if hasattr(model, "module"):
        model = model.module
    # set optimizer constructor
    
    if constructor_cfg is None or constructor_cfg == {}:
        constructor_cfg = dict(
            target=DEFAULT_OPTIMIZER_CONSTRUCT,
            params=dict(
                optimizer_cfg={},
                paramwise_cfg={}
            )
        )
    
    # parameter-wise setting: cancel weight decay for some specific modules
    custom_keys = dict()
    for name, module in model.named_modules():
        if hasattr(module, "zero_weight_decay"):
            custom_keys.update({(name, key): dict(decay_mult=0) for key in module.zero_weight_decay})
    
    given_cfg = constructor_cfg["params"].get("paramwise_cfg") if "params" in constructor_cfg["params"] else {}
    if custom_keys == {} and given_cfg == {}:
        paramwise_cfg = None
        constructor_cfg["params"]["paramwise_cfg"] = None
    else:
        paramwise_cfg = dict(cfg=dict(custom_keys=custom_keys))
        if given_cfg:
            paramwise_cfg.merge_from_dict(dict(cfg=given_cfg))
        constructor_cfg["params"]["paramwise_cfg"] = paramwise_cfg["cfg"]
    constructor_cfg["params"]["optimizer_cfg"] = dict(optimizer_cfg) if not isinstance(optimizer_cfg, list) else optimizer_cfg
    # build optimizer
    constructor = instantiate_from_config(constructor_cfg)
    optimizer = constructor(model)

    weight_decay_groups = dict()
    lr_groups = dict()
    if isinstance(optimizer, list):
        for optimizer_i in optimizer:
            for group in optimizer_i.param_groups:
                if not group.get("requires_grad", True):
                    continue
                lr_groups.setdefault(group["lr"], []).append(group)
                weight_decay_groups.setdefault(group["weight_decay"], []).append(group)
    else:
        for group in optimizer.param_groups:
            if not group.get("requires_grad", True):
                continue
            lr_groups.setdefault(group["lr"], []).append(group)
            weight_decay_groups.setdefault(group["weight_decay"], []).append(group)

    learnable_count, fix_count = 0, 0
    for p in model.parameters():
        if p.requires_grad:
            learnable_count += 1
        else:
            fix_count += 1
    fix_info = colored(f"{learnable_count} are learnable, {fix_count} are fix", "green")
    lr_info = "Lr group: " + ", ".join([f"{len(group)} params with lr {lr:.5f}" for lr, group in lr_groups.items()])
    wd_info = "Weight decay group: " + ", ".join(
        [f"{len(group)} params with weight decay {wd}" for wd, group in weight_decay_groups.items()]
    )
    
    if isinstance(optimizer, list):
        optimizer_info = [f"{optimizer_i.__class__.__name__} Optimizer: total {len(optimizer_i.param_groups)} param groups." \
            for optimizer_i in optimizer]
        opt_info = "[ " + ", ".join(optimizer_info) + " ], "
        opt_info += f" {fix_info}. {lr_info}; {wd_info}."
    else:
        opt_info = f"{optimizer.__class__.__name__} Optimizer: total {len(optimizer.param_groups)} param groups, {fix_info}. {lr_info}; {wd_info}."
    logger.info(opt_info)

    return optimizer

