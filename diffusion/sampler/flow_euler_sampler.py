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

import os
import torch
import torch.nn.functional as F
from typing import Optional, Dict, List, Any
from diffusion.scheduler.flow_euler_scheduler import FlowMatchEulerDiscreteScheduler
from diffusion.scheduler.identity_scheduler import IdentityScheduler
from diffusers.pipelines.stable_diffusion_3.pipeline_stable_diffusion_3 import retrieve_timesteps
from tqdm import tqdm
from diffusion.utils import rotation_conversions as rc


def compute_rot6d_manifold_stats(
    x: torch.Tensor, 
    rot6d_dim: int,
    compute_per_joint: bool = False,
) -> Dict[str, Any]:
    """
    计算 rot6d 部分与标准流形（Stiefel manifold的距离统计。
    
    Args:
        x: (B, T, D) 运动张量，其中前 rot6d_dim 维是 rot6d
        rot6d_dim: rot6d 部分的维度（通常是 joints * 6）
        compute_per_joint: 是否计算每个关节的统计
        
    Returns:
        包含各种统计指标的字典
    """
    # 提取 rot6d 部分
    rot6d = x[..., :rot6d_dim]  # (B, T, J*6)
    B, T = rot6d.shape[:2]
    n_joints = rot6d_dim // 6
    
    # reshape 为 (B, T, J, 6)
    rot6d_reshaped = rot6d.view(B, T, n_joints, 6)
    
    # 分离 a1, a2（rot6d 的两个 3D 向量）
    a1 = rot6d_reshaped[..., :3]  # (B, T, J, 3)
    a2 = rot6d_reshaped[..., 3:]  # (B, T, J, 3)
    
    # ============ 指标 1: 投影距离 ============
    # 将 rot6d 投影到流形上，计算距离
    rot6d_normalized = rc.NormalizeRot6d(rot6d)  # (B, T, J*6)
    proj_diff = rot6d - rot6d_normalized
    proj_l2 = torch.linalg.norm(proj_diff.view(B, T, n_joints, 6), dim=-1)  # (B, T, J)
    
    # ============ 指标 2: 单位性误差 ============
    # ||a1|| 应该等于 1
    a1_norm = torch.linalg.norm(a1, dim=-1)  # (B, T, J)
    a1_norm_err = (a1_norm - 1.0).abs()
    
    # ||a2|| 应该等于 1（归一化后）
    a2_norm = torch.linalg.norm(a2, dim=-1)  # (B, T, J)
    a2_norm_err = (a2_norm - 1.0).abs()
    
    # ============ 指标 3: 正交性误差 ============
    # a1 · a2 应该等于 0（归一化后）
    # 对于原始数据，我们计算 normalized 后的正交性
    b1 = F.normalize(a1, dim=-1)
    b2_raw = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
    b2 = F.normalize(b2_raw, dim=-1)
    orth_err = (b1 * a2).sum(dim=-1).abs()  # (B, T, J)
    
    # ============ 指标 4: Gram-Schmidt 修正量 ============
    # 归一化过程中 a2 被修正了多少
    gs_correction = torch.linalg.norm(a2 - b2_raw, dim=-1)  # (B, T, J)
    
    # 聚合统计
    stats = {
        # 投影距离
        "proj_l2_mean": proj_l2.mean().item(),
        "proj_l2_std": proj_l2.std().item(),
        "proj_l2_max": proj_l2.max().item(),
        "proj_l2_min": proj_l2.min().item(),
        # a1 单位性
        "a1_norm_mean": a1_norm.mean().item(),
        "a1_norm_err_mean": a1_norm_err.mean().item(),
        "a1_norm_err_max": a1_norm_err.max().item(),
        # a2 单位性
        "a2_norm_mean": a2_norm.mean().item(),
        "a2_norm_err_mean": a2_norm_err.mean().item(),
        "a2_norm_err_max": a2_norm_err.max().item(),
        # 正交性
        "orth_err_mean": orth_err.mean().item(),
        "orth_err_max": orth_err.max().item(),
        # Gram-Schmidt 修正量
        "gs_correction_mean": gs_correction.mean().item(),
        "gs_correction_max": gs_correction.max().item(),
    }
    
    if compute_per_joint:
        # 每个关节的平均统计
        stats["proj_l2_per_joint"] = proj_l2.mean(dim=(0, 1)).tolist()
        stats["a1_norm_err_per_joint"] = a1_norm_err.mean(dim=(0, 1)).tolist()
        stats["orth_err_per_joint"] = orth_err.mean(dim=(0, 1)).tolist()
    
    return stats


class OmniFlowEulerSampler:
    def __init__(
        self,  
        diffusion,
        shift=3.0,
        steps=28,
        task="text2all",
        prediction_type="velocity",
        use_dynamic_shifting=False,
        stochastic_sampling=False,
        use_karras_sigmas: Optional[bool] = False,
        use_exponential_sigmas: Optional[bool] = False,
        use_beta_sigmas: Optional[bool] = False,
    ):
        # 三个模态都用一样的scheduler，model放到后面去，便于我们在trainer的init函数里面就能把sampler实例化出来
        # 关于三个模态用不养的scheduler，后续再考虑
        self.diffusion = diffusion
        self.max_training_steps_index = diffusion.num_timesteps - 1
        self.prediction_type = prediction_type
        # 先支持下面这些
        if task.endswith("all"):
            self.motion_scheduler = FlowMatchEulerDiscreteScheduler(
                prediction_type=self.prediction_type,
                shift=shift,
                use_dynamic_shifting=use_dynamic_shifting,
                stochastic_sampling=stochastic_sampling,
                use_karras_sigmas=use_karras_sigmas,
                use_exponential_sigmas=use_exponential_sigmas,
                use_beta_sigmas=use_beta_sigmas,
            )
            self.expression_scheduler = FlowMatchEulerDiscreteScheduler(
                prediction_type=self.prediction_type,
                shift=shift,
                use_dynamic_shifting=use_dynamic_shifting,
                stochastic_sampling=stochastic_sampling,
                use_karras_sigmas=use_karras_sigmas,
                use_exponential_sigmas=use_exponential_sigmas,
                use_beta_sigmas=use_beta_sigmas,
            )
            self.audio_scheduler = FlowMatchEulerDiscreteScheduler(
                prediction_type=self.prediction_type,
                shift=shift,
                use_dynamic_shifting=use_dynamic_shifting,
                stochastic_sampling=stochastic_sampling,
                use_karras_sigmas=use_karras_sigmas,
                use_exponential_sigmas=use_exponential_sigmas,
                use_beta_sigmas=use_beta_sigmas,
            )
        # =========================================================================
        elif task.endswith("2motion"):
            self.motion_scheduler = FlowMatchEulerDiscreteScheduler(
                prediction_type=self.prediction_type,
                shift=shift,
                use_dynamic_shifting=use_dynamic_shifting,
                stochastic_sampling=stochastic_sampling,
                use_karras_sigmas=use_karras_sigmas,
                use_exponential_sigmas=use_exponential_sigmas,
                use_beta_sigmas=use_beta_sigmas,
            )
            cond_type = task.split("2")[0]
            if "expression" in cond_type:
                self.expression_scheduler = IdentityScheduler(
                    constant_timestep=0,
                )
            else:
                self.expression_scheduler = IdentityScheduler(
                    constant_timestep=self.max_training_steps_index,
                )
            if "audio" in cond_type:
                self.audio_scheduler = IdentityScheduler(
                    constant_timestep=0,
                )
            else:
                self.audio_scheduler = IdentityScheduler(
                    constant_timestep=self.max_training_steps_index,
                )
        elif task.endswith("2expression"):    
            self.expression_scheduler = FlowMatchEulerDiscreteScheduler(
                prediction_type=self.prediction_type,
                shift=shift,
                use_dynamic_shifting=use_dynamic_shifting,
                stochastic_sampling=stochastic_sampling,
                use_karras_sigmas=use_karras_sigmas,
                use_exponential_sigmas=use_exponential_sigmas,
                use_beta_sigmas=use_beta_sigmas,
            )
            cond_type = task.split("2")[0]
            if "motion" in cond_type:
                self.motion_scheduler = IdentityScheduler(
                    constant_timestep=0,
                )
            else:
                self.motion_scheduler = IdentityScheduler(
                    constant_timestep=self.max_training_steps_index,
                )
            if "audio" in cond_type:
                self.audio_scheduler = IdentityScheduler(
                    constant_timestep=0,
                )
            else:
                self.audio_scheduler = IdentityScheduler(
                    constant_timestep=self.max_training_steps_index,
                )
        elif task.endswith("2audio"):
            self.audio_scheduler = FlowMatchEulerDiscreteScheduler(
                prediction_type=self.prediction_type,
                shift=shift,
                use_dynamic_shifting=use_dynamic_shifting,
                stochastic_sampling=stochastic_sampling,
                use_karras_sigmas=use_karras_sigmas,
                use_exponential_sigmas=use_exponential_sigmas,
                use_beta_sigmas=use_beta_sigmas,
            )
            cond_type = task.split("2")[0]
            if "motion" in cond_type:
                self.motion_scheduler = IdentityScheduler(
                    constant_timestep=0,
                )
            else:
                self.motion_scheduler = IdentityScheduler(
                    constant_timestep=self.max_training_steps_index,
                )
            if "expression" in cond_type:
                self.expression_scheduler = IdentityScheduler(
                    constant_timestep=0,
                )
            else:
                self.expression_scheduler = IdentityScheduler(
                    constant_timestep=self.max_training_steps_index,
                )
        elif task.endswith("2motion+expression"):
            self.motion_scheduler = FlowMatchEulerDiscreteScheduler(
                prediction_type=self.prediction_type,
                shift=shift,
                use_dynamic_shifting=use_dynamic_shifting,
                stochastic_sampling=stochastic_sampling,
                use_karras_sigmas=use_karras_sigmas,
                use_exponential_sigmas=use_exponential_sigmas,
                use_beta_sigmas=use_beta_sigmas,
            )
            self.expression_scheduler = FlowMatchEulerDiscreteScheduler(
                prediction_type=self.prediction_type,
                shift=shift,
                use_dynamic_shifting=use_dynamic_shifting,
                stochastic_sampling=stochastic_sampling,
                use_karras_sigmas=use_karras_sigmas,
                use_exponential_sigmas=use_exponential_sigmas,
                use_beta_sigmas=use_beta_sigmas,
            )
            cond_type = task.split("2")[0]
            if "audio" in cond_type:
                self.audio_scheduler = IdentityScheduler(
                    constant_timestep=0,
                )
            else:
                self.audio_scheduler = IdentityScheduler(
                    constant_timestep=self.max_training_steps_index,
                )
        elif task.endswith("2motion+audio"):
            self.motion_scheduler = FlowMatchEulerDiscreteScheduler(
                prediction_type=self.prediction_type,
                shift=shift,
                use_dynamic_shifting=use_dynamic_shifting,
                stochastic_sampling=stochastic_sampling,
                use_karras_sigmas=use_karras_sigmas,
                use_exponential_sigmas=use_exponential_sigmas,
                use_beta_sigmas=use_beta_sigmas,
            )
            self.audio_scheduler = FlowMatchEulerDiscreteScheduler(
                prediction_type=self.prediction_type,
                shift=shift,
                use_dynamic_shifting=use_dynamic_shifting,
                stochastic_sampling=stochastic_sampling,
                use_karras_sigmas=use_karras_sigmas,
                use_exponential_sigmas=use_exponential_sigmas,
                use_beta_sigmas=use_beta_sigmas,
            )
            cond_type = task.split("2")[0]
            if "expression" in cond_type:
                self.expression_scheduler = IdentityScheduler(
                    constant_timestep=0,
                )
            else:
                self.expression_scheduler = IdentityScheduler(
                    constant_timestep=self.max_training_steps_index,
                )
        elif task.endswith("2expression+audio"):
            self.expression_scheduler = FlowMatchEulerDiscreteScheduler(
                prediction_type=self.prediction_type,
                shift=shift,
                use_dynamic_shifting=use_dynamic_shifting,
                stochastic_sampling=stochastic_sampling,
                use_karras_sigmas=use_karras_sigmas,
                use_exponential_sigmas=use_exponential_sigmas,
                use_beta_sigmas=use_beta_sigmas,
            )
            self.audio_scheduler = FlowMatchEulerDiscreteScheduler(
                prediction_type=self.prediction_type,
                shift=shift,
                use_dynamic_shifting=use_dynamic_shifting,
                stochastic_sampling=stochastic_sampling,
                use_karras_sigmas=use_karras_sigmas,
                use_exponential_sigmas=use_exponential_sigmas,
                use_beta_sigmas=use_beta_sigmas,
            )
            cond_type = task.split("2")[0]
            if "motion" in cond_type:
                self.motion_scheduler = IdentityScheduler(
                    constant_timestep=0,
                )
            else:
                self.motion_scheduler = IdentityScheduler(
                    constant_timestep=self.max_training_steps_index,
                )
        # =========================================================================
        else:
            raise ValueError(f"Unsupported task: {task}, has not been implemented yet.")   
        self.steps = steps

    def get_adding_noise_sequence(
        self, 
        motion_latents, 
        expression_latents, 
        audio_latents,
        steps=None,
    ):
        device = motion_latents.device
        if steps is None:
            steps = self.steps
        motion_timesteps, num_motion_inference_steps = \
            retrieve_timesteps(self.motion_scheduler, steps, device, None)
        expression_timesteps, num_expression_inference_steps = \
            retrieve_timesteps(self.expression_scheduler, steps, device, None)
        audio_timesteps, num_audio_inference_steps = \
            retrieve_timesteps(self.audio_scheduler, steps, device, None)
        
        # 给倒回来
        motion_timesteps = torch.flip(motion_timesteps, dims=[0]).long()
        expression_timesteps = torch.flip(expression_timesteps, dims=[0]).long()
        audio_timesteps = torch.flip(audio_timesteps, dims=[0]).long()
        
        motion_timesteps = motion_timesteps[:-1]
        expression_timesteps = expression_timesteps[:-1]
        audio_timesteps = audio_timesteps[:-1]
        
        motion_timesteps = torch.cat([torch.tensor([0], device=device, dtype=motion_timesteps.dtype),
                                    motion_timesteps])
        expression_timesteps = torch.cat([torch.tensor([0], device=device, dtype=motion_timesteps.dtype),
                                    expression_timesteps])
        audio_timesteps = torch.cat([torch.tensor([0], device=device, dtype=motion_timesteps.dtype),
                                    audio_timesteps])
        
        motion_timesteps = list(motion_timesteps)
        expression_timesteps = list(expression_timesteps)
        audio_timesteps = list(audio_timesteps)
        
        timesteps = zip(motion_timesteps, expression_timesteps, audio_timesteps)
        
        motion_ts = []
        expression_ts = []
        audio_ts = []
        motion_noise = torch.randn_like(motion_latents)
        expression_noise = torch.randn_like(expression_latents)
        audio_noise = torch.randn_like(audio_latents)
        for i, (motion_timestep, expression_timestep, audio_timestep) \
            in tqdm(enumerate(timesteps), desc="Getting adding noise sequence",\
                disable=os.getenv("DPM_TQDM", "True") == "True"):

                motion_t = self.diffusion.q_sample(
                    motion_latents, 
                    t=motion_timestep, 
                    noise=motion_noise
                )
                expression_t = self.diffusion.q_sample(
                    expression_latents, 
                    t=expression_timestep, 
                    noise=expression_noise
                )
                audio_t = self.diffusion.q_sample(
                    audio_latents, 
                    t=audio_timestep, 
                    noise=audio_noise
                )
                motion_ts.append(motion_t.detach())
                expression_ts.append(expression_t.detach())
                audio_ts.append(audio_t.detach())
                
        return motion_ts, expression_ts, audio_ts

    def sample(
        self, 
        model_fn,
        motion_latents, 
        expression_latents, 
        audio_latents,
        return_intermediate=False,
        return_model_output=False,
    ):
        # classifier-free guidance的逻辑放出去，这里只管采样
        device = motion_latents.device
        motion_timesteps, num_motion_inference_steps = \
            retrieve_timesteps(self.motion_scheduler, self.steps, device, None)
        expression_timesteps, num_expression_inference_steps = \
            retrieve_timesteps(self.expression_scheduler, self.steps, device, None)
        audio_timesteps, num_audio_inference_steps = \
            retrieve_timesteps(self.audio_scheduler, self.steps, device, None)
        
        motion_timesteps = list(motion_timesteps)
        expression_timesteps = list(expression_timesteps)
        audio_timesteps = list(audio_timesteps)
        
        timesteps = zip(motion_timesteps, expression_timesteps, audio_timesteps)
        
        if return_intermediate:
            motion_intermediates = []
            expression_intermediates = []
            audio_intermediates = []
        if return_model_output:
            motion_model_outputs = []
            expression_model_outputs = []
            audio_model_outputs = []
        
        for i, (motion_timestep, expression_timestep, audio_timestep) \
            in tqdm(enumerate(timesteps), desc="Sampling",\
                disable=os.getenv("DPM_TQDM", "True") == "True"):

            # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
            motion_timestep_bc = motion_timestep.expand(motion_latents.shape[0])
            expression_timestep_bc = expression_timestep.expand(expression_latents.shape[0])
            audio_timestep_bc = audio_timestep.expand(audio_latents.shape[0])

            pred_motion, pred_expression, pred_audio = model_fn(
                motion_latents,
                expression_latents,
                audio_latents,
                motion_timestep_bc,
                expression_timestep_bc,
                audio_timestep_bc,
            )
            if return_model_output:
                motion_model_outputs.append(pred_motion.detach())
                expression_model_outputs.append(pred_expression.detach())
                audio_model_outputs.append(pred_audio.detach())
            
            motion_latents_dtype = motion_latents.dtype
            expression_latents_dtype = expression_latents.dtype
            audio_latents_dtype = audio_latents.dtype

            motion_latents = self.motion_scheduler.step(
                pred_motion, motion_timestep, motion_latents, return_dict=False)[0]        
            expression_latents = self.expression_scheduler.step(
                pred_expression, expression_timestep, expression_latents, return_dict=False)[0]
            audio_latents = self.audio_scheduler.step(
                pred_audio, audio_timestep, audio_latents, return_dict=False)[0]
            

            if motion_latents.dtype != motion_latents_dtype:
                motion_latents = motion_latents.to(motion_latents_dtype)
            if expression_latents.dtype != expression_latents_dtype:
                expression_latents = expression_latents.to(expression_latents_dtype)
            if audio_latents.dtype != audio_latents_dtype:
                audio_latents = audio_latents.to(audio_latents_dtype)
            
            if return_intermediate:
                motion_intermediates.append(motion_latents.detach())
                expression_intermediates.append(expression_latents.detach())
                audio_intermediates.append(audio_latents.detach())
        
        extra = {}
        if return_intermediate:
            extra["motion_intermediates"] = motion_intermediates
            extra["expression_intermediates"] = expression_intermediates
            extra["audio_intermediates"] = audio_intermediates
        if return_model_output:
            extra["motion_model_outputs"] = motion_model_outputs
            extra["expression_model_outputs"] = expression_model_outputs
            extra["audio_model_outputs"] = audio_model_outputs
        return motion_latents, expression_latents, audio_latents, extra


class SimpleFlowEulerSampler:
    def __init__(
        self,  
        diffusion,
        shift=3.0,
        steps=28,
        task="text2all",
        prediction_type="velocity",
        use_dynamic_shifting=False,
        stochastic_sampling=False,
        use_karras_sigmas: Optional[bool] = False,
        use_exponential_sigmas: Optional[bool] = False,
        use_beta_sigmas: Optional[bool] = False,
    ):
        # 三个模态都用一样的scheduler，model放到后面去，便于我们在trainer的init函数里面就能把sampler实例化出来
        # 关于三个模态用不养的scheduler，后续再考虑
        self.diffusion = diffusion
        self.max_training_steps_index = diffusion.num_timesteps - 1
        self.prediction_type = prediction_type
        # 先支持下面这些
        if task.endswith("2motion"):
            self.motion_scheduler = FlowMatchEulerDiscreteScheduler(
                prediction_type=self.prediction_type,
                shift=shift,
                use_dynamic_shifting=use_dynamic_shifting,
                stochastic_sampling=stochastic_sampling,
                use_karras_sigmas=use_karras_sigmas,
                use_exponential_sigmas=use_exponential_sigmas,
                use_beta_sigmas=use_beta_sigmas,
            )
            self.expression_scheduler = IdentityScheduler(
                constant_timestep=self.max_training_steps_index,
            )
        elif task.endswith("2motion+expression"):
            self.motion_scheduler = FlowMatchEulerDiscreteScheduler(
                prediction_type=self.prediction_type,
                shift=shift,
                use_dynamic_shifting=use_dynamic_shifting,
                stochastic_sampling=stochastic_sampling,
                use_karras_sigmas=use_karras_sigmas,
                use_exponential_sigmas=use_exponential_sigmas,
                use_beta_sigmas=use_beta_sigmas,
            )
            self.expression_scheduler = FlowMatchEulerDiscreteScheduler(
                prediction_type=self.prediction_type,
                shift=shift,
                use_dynamic_shifting=use_dynamic_shifting,
                stochastic_sampling=stochastic_sampling,
                use_karras_sigmas=use_karras_sigmas,
                use_exponential_sigmas=use_exponential_sigmas,
                use_beta_sigmas=use_beta_sigmas,
            )
        else:
            raise ValueError(f"Unsupported task: {task}, has not been implemented yet.")   
        self.steps = steps

    def get_adding_noise_sequence(
        self, 
        motion_latents, 
        expression_latents, 
        steps=None,
    ):
        device = motion_latents.device
        if steps is None:
            steps = self.steps
        motion_timesteps, num_motion_inference_steps = \
            retrieve_timesteps(self.motion_scheduler, steps, device, None)
        expression_timesteps, num_expression_inference_steps = \
            retrieve_timesteps(self.expression_scheduler, steps, device, None)
        
        # 给倒回来
        motion_timesteps = torch.flip(motion_timesteps, dims=[0]).long()
        expression_timesteps = torch.flip(expression_timesteps, dims=[0]).long()
        
        motion_timesteps = motion_timesteps[:-1]
        expression_timesteps = expression_timesteps[:-1]
        
        motion_timesteps = torch.cat([torch.tensor([0], device=device, dtype=motion_timesteps.dtype),
                                    motion_timesteps])
        expression_timesteps = torch.cat([torch.tensor([0], device=device, dtype=motion_timesteps.dtype),
                                    expression_timesteps])
        
        motion_timesteps = list(motion_timesteps)
        expression_timesteps = list(expression_timesteps)
        
        timesteps = zip(motion_timesteps, expression_timesteps)
        
        motion_ts = []
        expression_ts = []
        motion_noise = torch.randn_like(motion_latents)
        expression_noise = torch.randn_like(expression_latents)
        for i, (motion_timestep, expression_timestep) \
            in tqdm(enumerate(timesteps), desc="Getting adding noise sequence",\
                disable=os.getenv("DPM_TQDM", "True") == "True"):
                motion_t = self.diffusion.q_sample(
                    motion_latents, 
                    t=motion_timestep, 
                    noise=motion_noise
                )
                expression_t = self.diffusion.q_sample(
                    expression_latents, 
                    t=expression_timestep, 
                    noise=expression_noise
                )
                motion_ts.append(motion_t.detach())
                expression_ts.append(expression_t.detach())
                
        return motion_ts, expression_ts

    def sample(
        self, 
        model_fn,
        motion_latents, 
        expression_latents, 
        return_intermediate=False,
        return_model_output=False,
    ):
        # classifier-free guidance的逻辑放出去，这里只管采样
        device = motion_latents.device
        motion_timesteps, num_motion_inference_steps = \
            retrieve_timesteps(self.motion_scheduler, self.steps, device, None)
        expression_timesteps, num_expression_inference_steps = \
            retrieve_timesteps(self.expression_scheduler, self.steps, device, None)
        
        motion_timesteps = list(motion_timesteps)
        expression_timesteps = list(expression_timesteps)
        
        timesteps = zip(motion_timesteps, expression_timesteps)
        
        if return_intermediate:
            motion_intermediates = []
            expression_intermediates = []
        if return_model_output:
            motion_model_outputs = []
            expression_model_outputs = []
        
        for i, (motion_timestep, expression_timestep) \
            in tqdm(enumerate(timesteps), desc="Sampling",\
                disable=os.getenv("DPM_TQDM", "True") == "True"):

            # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
            motion_timestep_bc = motion_timestep.expand(motion_latents.shape[0])
            expression_timestep_bc = expression_timestep.expand(expression_latents.shape[0])
            # assert torch.all(motion_timestep_bc == expression_timestep_bc)

            pred_motion, pred_expression = model_fn(
                motion_latents,
                expression_latents,
                motion_timestep_bc,
            )
            if return_model_output:
                motion_model_outputs.append(pred_motion.detach())
                expression_model_outputs.append(pred_expression.detach())
            
            motion_latents_dtype = motion_latents.dtype
            expression_latents_dtype = expression_latents.dtype

            motion_latents = self.motion_scheduler.step(
                pred_motion, motion_timestep, motion_latents, return_dict=False)[0]        
            expression_latents = self.expression_scheduler.step(
                pred_expression, expression_timestep, expression_latents, return_dict=False)[0]

            if motion_latents.dtype != motion_latents_dtype:
                motion_latents = motion_latents.to(motion_latents_dtype)
            if expression_latents.dtype != expression_latents_dtype:
                expression_latents = expression_latents.to(expression_latents_dtype)    
            
            if return_intermediate:
                motion_intermediates.append(motion_latents.detach())
                expression_intermediates.append(expression_latents.detach())
        
        extra = {}
        if return_intermediate:
            extra["motion_intermediates"] = motion_intermediates
            extra["expression_intermediates"] = expression_intermediates
        if return_model_output:
            extra["motion_model_outputs"] = motion_model_outputs
            extra["expression_model_outputs"] = expression_model_outputs
        return motion_latents, expression_latents, extra


class EmageFlowEulerSampler:
    def __init__(
        self,
        diffusion,
        shift=3.0,
        steps=28,
        task="text2all",
        prediction_type="velocity",
        use_dynamic_shifting=False,
        stochastic_sampling=False,
        use_karras_sigmas: Optional[bool] = False,
        use_exponential_sigmas: Optional[bool] = False,
        use_beta_sigmas: Optional[bool] = False,
    ):
        self.diffusion = diffusion
        self.max_training_steps_index = diffusion.num_timesteps - 1
        self.prediction_type = prediction_type

        self.motion_scheduler = FlowMatchEulerDiscreteScheduler(
            prediction_type=self.prediction_type,
            shift=shift,
            use_dynamic_shifting=use_dynamic_shifting,
            stochastic_sampling=stochastic_sampling,
            use_karras_sigmas=use_karras_sigmas,
            use_exponential_sigmas=use_exponential_sigmas,
            use_beta_sigmas=use_beta_sigmas,
        )
        self.steps = steps

    def get_adding_noise_sequence(
        self, 
        motion_latents,  
        steps=None,
    ):
        device = motion_latents.device
        if steps is None:
            steps = self.steps
        motion_timesteps, num_motion_inference_steps = \
            retrieve_timesteps(self.motion_scheduler, steps, device, None)
        
        # 给倒回来
        motion_timesteps = torch.flip(motion_timesteps, dims=[0]).long()
        motion_timesteps = motion_timesteps[:-1]
        motion_timesteps = torch.cat([
            torch.tensor([0], device=device, dtype=motion_timesteps.dtype), 
            motion_timesteps
        ])
        motion_timesteps = list(motion_timesteps)
        motion_ts = []
        motion_noise = torch.randn_like(motion_latents)
        for i, (motion_timestep) \
            in tqdm(enumerate(motion_timesteps), desc="Getting adding noise sequence",\
                disable=os.getenv("DPM_TQDM", "True") == "True"):
                motion_t = self.diffusion.q_sample(
                    motion_latents, 
                    t=motion_timestep, 
                    noise=motion_noise
                )
                motion_ts.append(motion_t.detach())
                
        return motion_ts

    @torch.autocast(device_type="cuda", dtype=torch.bfloat16, \
        enabled=os.getenv("USE_MIXED_PRECISION_FOR_SAMPLER", "False") == "True")
    def sample(
        self,
        model_fn,
        motion_latents,
        return_intermediate=False,
        return_model_output=False,
        post_process_func=None,
        return_rot6d_stats=False,
        rot6d_dim=None,
        compute_per_joint_stats=False,
        start_sigma: float | None = None,
        **kwargs,
    ):
        """
        采样方法。

        Args:
            model_fn: 模型前向函数
            motion_latents: 初始噪声 (B, T, D)
            return_intermediate: 是否返回中间结果
            return_model_output: 是否返回模型输出
            post_process_func: 后处理函数（如 NormalizeRot6d）
            return_rot6d_stats: 是否返回每一步的 rot6d 流形统计
            rot6d_dim: rot6d 的维度（joints * 6），用于 rot6d_stats
            compute_per_joint_stats: 是否计算每个关节的统计
            start_sigma: 从指定 sigma 开始采样（用于 self-sampling / Resampling Forcing）。
                         为 None 时使用完整时间表（默认行为）。
            **kwargs: 传递给 model_fn 的其他参数
        """
        # classifier-free guidance的逻辑放出去，这里只管采样
        device = motion_latents.device
        motion_timesteps, num_motion_inference_steps = \
            retrieve_timesteps(self.motion_scheduler, self.steps, device, None)
        motion_timesteps = list[int](motion_timesteps)

        num_train_timesteps = self.motion_scheduler.config.num_train_timesteps

        # start_sigma: 截断时间表，只保留 sigma <= start_sigma 的步骤
        # 用于 Resampling Forcing，ODE 从 t_s 积分到 0 而非从 1 到 0
        if start_sigma is not None:
            start_t = start_sigma * num_train_timesteps
            motion_timesteps = [t for t in motion_timesteps if float(t) <= start_t + 1e-3]
            if len(motion_timesteps) == 0:
                # t_s 太小，无需去噪，输入已接近 clean
                extra = {}
                if return_intermediate:
                    extra["motion_intermediates"] = []
                if return_model_output:
                    extra["motion_model_outputs"] = []
                return motion_latents, extra

        N = len(motion_timesteps)

        if return_intermediate:
            motion_intermediates = []
        if return_model_output:
            motion_model_outputs = []
        if return_rot6d_stats:
            rot6d_stats_list = []
            if rot6d_dim is None:
                rot6d_dim = getattr(self.diffusion, 'joints', 55) * 6

        # start_sigma 模式下使用 ODE 形式（noise=None），不需要 noise reference。
        # ODE 形式 x_{σ'} = (σ'/σ)*x_σ + (1-σ'/σ)*pred_x0 从任意 σ 出发数学正确。
        # 正常推理时保持原行为：存储初始噪声作为 noise reference。
        if start_sigma is not None:
            noise = None
        else:
            noise = motion_latents.clone().detach()
        
        if return_rot6d_stats:
            init_stats = compute_rot6d_manifold_stats(
                motion_latents, rot6d_dim, compute_per_joint_stats
            )
            init_stats["step"] = -1
            init_stats["timestep"] = 1000.0
            rot6d_stats_list.append(init_stats)
        
        for i, (motion_timestep) \
            in tqdm(enumerate(motion_timesteps), desc="Sampling",\
                disable=os.getenv("DPM_TQDM", "True") == "True"):

            motion_latents_dtype = motion_latents.dtype

            # --- Determine current_sigma_override ---
            # start_sigma first step: model sees the true continuous noise level,
            # scheduler steps from start_sigma directly to the next discrete node.
            if i == 0 and start_sigma is not None:
                sigma_override = start_sigma
            else:
                sigma_override = None

            # --- Model evaluation ---
            if sigma_override is not None:
                timestep_for_model = torch.tensor(
                    sigma_override * num_train_timesteps,
                    device=device, dtype=motion_latents.dtype
                ).clamp(max=num_train_timesteps)
            else:
                timestep_for_model = motion_timestep
            timestep_for_model_bc = timestep_for_model.expand(motion_latents.shape[0])

            pred_motion = model_fn(
                motion_latents,
                timestep_for_model_bc,
                **kwargs,
            )
            if post_process_func is not None:
                pred_motion = post_process_func(
                    pred_motion,
                    timestep=float(timestep_for_model),
                    step_idx=i,
                    total_steps=N,
                )
            if return_model_output:
                motion_model_outputs.append(pred_motion.clone().detach())

            # --- Euler step via scheduler ---
            motion_latents = self.motion_scheduler.step(
                pred_motion, motion_timestep, motion_latents,
                return_dict=False, noise=noise,
                current_sigma_override=sigma_override,
            )[0]

            if motion_latents.dtype != motion_latents_dtype:
                motion_latents = motion_latents.to(motion_latents_dtype)
            
            if return_intermediate:
                motion_intermediates.append(motion_latents.clone().detach())
            
            if return_rot6d_stats:
                step_stats = compute_rot6d_manifold_stats(
                    motion_latents, rot6d_dim, compute_per_joint_stats
                )
                step_stats["step"] = i
                step_stats["timestep"] = float(motion_timestep)
                
                pred_stats = compute_rot6d_manifold_stats(
                    pred_motion, rot6d_dim, False
                )
                step_stats["pred_proj_l2_mean"] = pred_stats["proj_l2_mean"]
                step_stats["pred_a1_norm_err_mean"] = pred_stats["a1_norm_err_mean"]
                step_stats["pred_orth_err_mean"] = pred_stats["orth_err_mean"]
                
                rot6d_stats_list.append(step_stats)
        
        extra = {}
        if return_intermediate:
            extra["motion_intermediates"] = motion_intermediates
        if return_model_output:
            extra["motion_model_outputs"] = motion_model_outputs
        if return_rot6d_stats:
            extra["rot6d_stats"] = rot6d_stats_list
        return motion_latents, extra