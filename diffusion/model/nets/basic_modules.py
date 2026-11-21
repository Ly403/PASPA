# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

# This file is modified from https://github.com/PixArt-alpha/PixArt-sigma
import math
import torch
import torch.nn as nn
import pickle
import os.path as osp

from diffusion.model.act import build_act
from diffusion.model.norms import build_norm
from positional_encodings.torch_encodings import PositionalEncoding1D, Summer
import numpy as np
import torch.nn.functional as F

# @torch.compile
def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

# @torch.compile
def t2i_modulate(x, shift, scale):
    return x * (1 + scale) + shift

class Mlp(nn.Module):
    """MLP as used in Vision Transformer, MLP-Mixer and related networks"""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_type="gelu", bias=True, drop=0.0):
        super(Mlp, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = build_act(act_type)
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class TripleStreamMLP(nn.Module):
    def __init__(
        self, 
        in_features, 
        hidden_features=None, 
        out_features=None, 
        act_type="gelu",
        proj_type="linear",
        bias=True, 
        dropout=0.0
    ):
        super(TripleStreamMLP, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.proj_type = proj_type
        if self.proj_type == "linear":
            self.motion_fc1 = nn.Linear(in_features, hidden_features, bias=bias)
            self.expression_fc1 = nn.Linear(in_features, hidden_features, bias=bias)
            self.audio_fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        elif self.proj_type == "conv":
            self.motion_fc1 = nn.Conv1d(in_features, hidden_features, stride=1, kernel_size=3, padding=1, bias=bias)
            self.expression_fc1 = nn.Conv1d(in_features, hidden_features, stride=1, kernel_size=3, padding=1, bias=bias)
            self.audio_fc1 = nn.Conv1d(in_features, hidden_features, stride=1, kernel_size=3, padding=1, bias=bias)
        else:
            raise NotImplementedError(f"proj_type {proj_type} not implemented")
        
        self.motion_act = build_act(act_type)
        self.expression_act = build_act(act_type)
        self.audio_act = build_act(act_type)
        
        self.motion_drop1 = nn.Dropout(dropout)
        self.expression_drop1 = nn.Dropout(dropout)
        self.audio_drop1 = nn.Dropout(dropout)
        
        self.proj_type = proj_type
        if self.proj_type == "linear":
            self.motion_fc2 = nn.Linear(hidden_features, in_features, bias=bias)
            self.expression_fc2 = nn.Linear(hidden_features, in_features, bias=bias)
            self.audio_fc2 = nn.Linear(hidden_features, in_features, bias=bias)
        elif self.proj_type == "conv":
            self.motion_fc2 = nn.Conv1d(hidden_features, in_features, stride=1, kernel_size=3, padding=1, bias=bias)
            self.expression_fc2 = nn.Conv1d(hidden_features, in_features, stride=1, kernel_size=3, padding=1, bias=bias)
            self.audio_fc2 = nn.Conv1d(hidden_features, in_features, stride=1, kernel_size=3, padding=1, bias=bias)
        else:
            raise NotImplementedError(f"proj_type {self.proj_type} not implemented")
        
        self.motion_drop2 = nn.Dropout(dropout)
        self.expression_drop2 = nn.Dropout(dropout)
        self.audio_drop2 = nn.Dropout(dropout)

    def forward(
        self,
        motion: torch.FloatTensor, 
        expression: torch.FloatTensor,
        audio: torch.FloatTensor,
        **kwargs
    ):
        if self.proj_type == "linear":
            motion = self.motion_fc1(motion)
            expression = self.expression_fc1(expression)
            audio = self.audio_fc1(audio)
        elif self.proj_type == "conv":
            motion = self.motion_fc1(motion.transpose(1, 2)).transpose(1, 2)
            expression = self.expression_fc1(expression.transpose(1, 2)).transpose(1, 2)
            audio = self.audio_fc1(audio.transpose(1, 2)).transpose(1, 2)
        
        motion = self.motion_act(motion)
        expression = self.expression_act(expression)
        audio = self.audio_act(audio)
        
        motion = self.motion_drop1(motion)
        expression = self.expression_drop1(expression)
        audio = self.audio_drop1(audio)
        
        if self.proj_type == "linear":
            motion = self.motion_fc2(motion)
            expression = self.expression_fc2(expression)
            audio = self.audio_fc2(audio)
        elif self.proj_type == "conv":
            motion = self.motion_fc2(motion.transpose(1, 2)).transpose(1, 2)
            expression = self.expression_fc2(expression.transpose(1, 2)).transpose(1, 2)
            audio = self.audio_fc2(audio.transpose(1, 2)).transpose(1, 2)
        else:
            raise NotImplementedError(f"proj_type {self.proj_type} not implemented")
        
        motion = self.motion_drop2(motion)
        expression = self.expression_drop2(expression)
        audio = self.audio_drop2(audio)
        
        return motion, expression, audio
    
class DoubleStreamMLP(nn.Module):
    def __init__(
        self, 
        in_features, 
        hidden_features=None, 
        out_features=None, 
        act_type="gelu",
        proj_type="linear",
        bias=True, 
        dropout=0.0
    ):
        super(DoubleStreamMLP, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.proj_type = proj_type
        if self.proj_type == "linear":
            self.motion_fc1 = nn.Linear(in_features, hidden_features, bias=bias)
            self.expression_fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        elif self.proj_type == "conv":
            self.motion_fc1 = nn.Conv1d(in_features, hidden_features, stride=1, kernel_size=3, padding=1, bias=bias)
            self.expression_fc1 = nn.Conv1d(in_features, hidden_features, stride=1, kernel_size=3, padding=1, bias=bias)
        else:
            raise NotImplementedError(f"proj_type {proj_type} not implemented")
        
        self.motion_act = build_act(act_type)
        self.expression_act = build_act(act_type)
        
        self.motion_drop1 = nn.Dropout(dropout)
        self.expression_drop1 = nn.Dropout(dropout)
        
        self.proj_type = proj_type
        if self.proj_type == "linear":
            self.motion_fc2 = nn.Linear(hidden_features, in_features, bias=bias)
            self.expression_fc2 = nn.Linear(hidden_features, in_features, bias=bias)
        elif self.proj_type == "conv":
            self.motion_fc2 = nn.Conv1d(hidden_features, in_features, stride=1, kernel_size=3, padding=1, bias=bias)
            self.expression_fc2 = nn.Conv1d(hidden_features, in_features, stride=1, kernel_size=3, padding=1, bias=bias)
        else:
            raise NotImplementedError(f"proj_type {self.proj_type} not implemented")
        
        self.motion_drop2 = nn.Dropout(dropout)
        self.expression_drop2 = nn.Dropout(dropout)

    def forward(
        self,
        motion: torch.FloatTensor, 
        expression: torch.FloatTensor,
        **kwargs
    ):
        if self.proj_type == "linear":
            motion = self.motion_fc1(motion)
            expression = self.expression_fc1(expression)
        elif self.proj_type == "conv":
            motion = self.motion_fc1(motion.transpose(1, 2)).transpose(1, 2)
            expression = self.expression_fc1(expression.transpose(1, 2)).transpose(1, 2)
        
        motion = self.motion_act(motion)
        expression = self.expression_act(expression)
        
        motion = self.motion_drop1(motion)
        expression = self.expression_drop1(expression)
        
        if self.proj_type == "linear":
            motion = self.motion_fc2(motion)
            expression = self.expression_fc2(expression)
        elif self.proj_type == "conv":
            motion = self.motion_fc2(motion.transpose(1, 2)).transpose(1, 2)
            expression = self.expression_fc2(expression.transpose(1, 2)).transpose(1, 2)
        else:
            raise NotImplementedError(f"proj_type {self.proj_type} not implemented")
        
        motion = self.motion_drop2(motion)
        expression = self.expression_drop2(expression)
        
        return motion, expression

class MLP(nn.Module):
    def __init__(
        self, 
        in_features, 
        hidden_features=None, 
        out_features=None, 
        act_type="gelu",
        proj_type="linear",
        bias=True, 
        dropout=0.0
    ):
        super(MLP, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.proj_type = proj_type
        if self.proj_type == "linear":
            self.motion_fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        elif self.proj_type == "conv":
            self.motion_fc1 = nn.Conv1d(in_features, hidden_features, stride=1, kernel_size=3, padding=1, bias=bias)
        else:
            raise NotImplementedError(f"proj_type {proj_type} not implemented")
        
        self.motion_act = build_act(act_type)
        self.motion_drop1 = nn.Dropout(dropout)
        
        self.proj_type = proj_type
        if self.proj_type == "linear":
            self.motion_fc2 = nn.Linear(hidden_features, in_features, bias=bias)
        elif self.proj_type == "conv":
            self.motion_fc2 = nn.Conv1d(hidden_features, in_features, stride=1, kernel_size=3, padding=1, bias=bias)
        else:
            raise NotImplementedError(f"proj_type {self.proj_type} not implemented")
        
        self.motion_drop2 = nn.Dropout(dropout)

    def forward(
        self,
        motion: torch.FloatTensor, 
        **kwargs
    ):
        if self.proj_type == "linear":
            motion = self.motion_fc1(motion)
        elif self.proj_type == "conv":
            motion = self.motion_fc1(motion.transpose(1, 2)).transpose(1, 2)
        
        motion = self.motion_act(motion)
        motion = self.motion_drop1(motion)
        
        if self.proj_type == "linear":
            motion = self.motion_fc2(motion)
        elif self.proj_type == "conv":
            motion = self.motion_fc2(motion.transpose(1, 2)).transpose(1, 2)
        else:
            raise NotImplementedError(f"proj_type {self.proj_type} not implemented")
        motion = self.motion_drop2(motion)
        return motion


class TripleStreamGatedMLP(nn.Module):
    def __init__(
        self, 
        in_features, 
        hidden_features=None, 
        out_features=None, 
        act_type="gelu",
        proj_type="linear",
        bias=True, 
        dropout=0.0
    ):
        super(TripleStreamGatedMLP, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.proj_type = proj_type
        if self.proj_type == "linear":
            self.motion_fc1 = nn.Linear(in_features, hidden_features * 2, bias=bias)
            self.expression_fc1 = nn.Linear(in_features, hidden_features * 2, bias=bias)
            self.audio_fc1 = nn.Linear(in_features, hidden_features * 2, bias=bias)
        elif self.proj_type == "conv":
            self.motion_fc1 = nn.Conv1d(in_features, hidden_features * 2, stride=1, kernel_size=3, padding=1, bias=bias)
            self.expression_fc1 = nn.Conv1d(in_features, hidden_features * 2, stride=1, kernel_size=3, padding=1, bias=bias)
            self.audio_fc1 = nn.Conv1d(in_features, hidden_features * 2, stride=1, kernel_size=3, padding=1, bias=bias)
        else:
            raise NotImplementedError(f"proj_type {self.proj_type} not implemented")
        
        self.motion_act = build_act(act_type)
        self.expression_act = build_act(act_type)
        self.audio_act = build_act(act_type)
        
        self.motion_drop1 = nn.Dropout(dropout)
        self.expression_drop1 = nn.Dropout(dropout)
        self.audio_drop1 = nn.Dropout(dropout)

        if self.proj_type == "linear":
            self.motion_fc2 = nn.Linear(hidden_features, in_features, bias=bias)
            self.expression_fc2 = nn.Linear(hidden_features, in_features, bias=bias)
            self.audio_fc2 = nn.Linear(hidden_features, in_features, bias=bias)
        elif self.proj_type == "conv":
            self.motion_fc2 = nn.Conv1d(hidden_features, in_features, stride=1, kernel_size=3, padding=1, bias=bias)
            self.expression_fc2 = nn.Conv1d(hidden_features, in_features, stride=1, kernel_size=3, padding=1, bias=bias)
            self.audio_fc2 = nn.Conv1d(hidden_features, in_features, stride=1, kernel_size=3, padding=1, bias=bias)
        else:
            raise NotImplementedError(f"proj_type {self.proj_type} not implemented")
        
        self.motion_drop2 = nn.Dropout(dropout)
        self.expression_drop2 = nn.Dropout(dropout)
        self.audio_drop2 = nn.Dropout(dropout)

    def forward(
        self,
        motion: torch.FloatTensor, 
        expression: torch.FloatTensor,
        audio: torch.FloatTensor,
        **kwargs
    ):
        if self.proj_type == "linear":
            motion = self.motion_fc1(motion)
            expression = self.expression_fc1(expression)
            audio = self.audio_fc1(audio)
        elif self.proj_type == "conv":
            motion = self.motion_fc1(motion.transpose(1, 2)).transpose(1, 2)
            expression = self.expression_fc1(expression.transpose(1, 2)).transpose(1, 2)
            audio = self.audio_fc1(audio.transpose(1, 2)).transpose(1, 2)
        else:
            raise NotImplementedError(f"proj_type {self.proj_type} not implemented")
        
        motion, motion_gate = torch.chunk(motion, 2, dim=-1)
        expression, expression_gate = torch.chunk(expression, 2, dim=-1)
        audio, audio_gate = torch.chunk(audio, 2, dim=-1)
        
        motion = self.motion_act(motion)
        expression = self.expression_act(expression)
        audio = self.audio_act(audio)
        
        motion = motion * motion_gate
        expression = expression * expression_gate
        audio = audio * audio_gate
        
        motion = self.motion_drop1(motion)
        expression = self.expression_drop1(expression)
        audio = self.audio_drop1(audio)
        
        if self.proj_type == "linear":
            motion = self.motion_fc2(motion)
            expression = self.expression_fc2(expression)
            audio = self.audio_fc2(audio)
        elif self.proj_type == "conv":
            motion = self.motion_fc2(motion.transpose(1, 2)).transpose(1, 2)
            expression = self.expression_fc2(expression.transpose(1, 2)).transpose(1, 2)
            audio = self.audio_fc2(audio.transpose(1, 2)).transpose(1, 2)
        else:
            raise NotImplementedError(f"proj_type {self.proj_type} not implemented")
        
        motion = self.motion_drop2(motion)
        expression = self.expression_drop2(expression)
        audio = self.audio_drop2(audio)
        
        return motion, expression, audio
    
class DoubleStreamGatedMLP(nn.Module):
    def __init__(
        self, 
        in_features, 
        hidden_features=None, 
        out_features=None, 
        act_type="gelu",
        proj_type="linear",
        bias=True, 
        dropout=0.0
    ):
        super(DoubleStreamGatedMLP, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.proj_type = proj_type
        if self.proj_type == "linear":
            self.motion_fc1 = nn.Linear(in_features, hidden_features * 2, bias=bias)
            self.expression_fc1 = nn.Linear(in_features, hidden_features * 2, bias=bias)
        elif self.proj_type == "conv":
            self.motion_fc1 = nn.Conv1d(in_features, hidden_features * 2, stride=1, kernel_size=3, padding=1, bias=bias)
            self.expression_fc1 = nn.Conv1d(in_features, hidden_features * 2, stride=1, kernel_size=3, padding=1, bias=bias)
        else:
            raise NotImplementedError(f"proj_type {self.proj_type} not implemented")
        
        self.motion_act = build_act(act_type)
        self.expression_act = build_act(act_type)
        
        self.motion_drop1 = nn.Dropout(dropout)
        self.expression_drop1 = nn.Dropout(dropout)

        if self.proj_type == "linear":
            self.motion_fc2 = nn.Linear(hidden_features, in_features, bias=bias)
            self.expression_fc2 = nn.Linear(hidden_features, in_features, bias=bias)
        elif self.proj_type == "conv":
            self.motion_fc2 = nn.Conv1d(hidden_features, in_features, stride=1, kernel_size=3, padding=1, bias=bias)
            self.expression_fc2 = nn.Conv1d(hidden_features, in_features, stride=1, kernel_size=3, padding=1, bias=bias)
        else:
            raise NotImplementedError(f"proj_type {self.proj_type} not implemented")
        
        self.motion_drop2 = nn.Dropout(dropout)
        self.expression_drop2 = nn.Dropout(dropout)

    def forward(
        self,
        motion: torch.FloatTensor, 
        expression: torch.FloatTensor,
        **kwargs
    ):
        if self.proj_type == "linear":
            motion = self.motion_fc1(motion)
            expression = self.expression_fc1(expression)
        elif self.proj_type == "conv":
            motion = self.motion_fc1(motion.transpose(1, 2)).transpose(1, 2)
            expression = self.expression_fc1(expression.transpose(1, 2)).transpose(1, 2)
        else:
            raise NotImplementedError(f"proj_type {self.proj_type} not implemented")
        
        motion, motion_gate = torch.chunk(motion, 2, dim=-1)
        expression, expression_gate = torch.chunk(expression, 2, dim=-1)
        
        motion = self.motion_act(motion)
        expression = self.expression_act(expression)
        
        motion = motion * motion_gate
        expression = expression * expression_gate
        
        motion = self.motion_drop1(motion)
        expression = self.expression_drop1(expression)
        
        if self.proj_type == "linear":
            motion = self.motion_fc2(motion)
            expression = self.expression_fc2(expression)
        elif self.proj_type == "conv":
            motion = self.motion_fc2(motion.transpose(1, 2)).transpose(1, 2)
            expression = self.expression_fc2(expression.transpose(1, 2)).transpose(1, 2)
        else:
            raise NotImplementedError(f"proj_type {self.proj_type} not implemented")
        
        motion = self.motion_drop2(motion)
        expression = self.expression_drop2(expression)
        
        return motion, expression
    
class GatedMLP(nn.Module):
    def __init__(
        self, 
        in_features, 
        hidden_features=None, 
        out_features=None, 
        act_type="gelu",
        proj_type="linear",
        bias=True, 
        dropout=0.0
    ):
        super(GatedMLP, self).__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.proj_type = proj_type
        if self.proj_type == "linear":
            self.motion_fc1 = nn.Linear(in_features, hidden_features * 2, bias=bias)
        elif self.proj_type == "conv":
            self.motion_fc1 = nn.Conv1d(in_features, hidden_features * 2, stride=1, kernel_size=3, padding=1, bias=bias)
        else:
            raise NotImplementedError(f"proj_type {self.proj_type} not implemented")
        
        self.motion_act = build_act(act_type)
        self.motion_drop1 = nn.Dropout(dropout)

        if self.proj_type == "linear":
            self.motion_fc2 = nn.Linear(hidden_features, in_features, bias=bias)
        elif self.proj_type == "conv":
            self.motion_fc2 = nn.Conv1d(hidden_features, in_features, stride=1, kernel_size=3, padding=1, bias=bias)
        else:
            raise NotImplementedError(f"proj_type {self.proj_type} not implemented")
        self.motion_drop2 = nn.Dropout(dropout)

    def forward(
        self,
        motion: torch.FloatTensor, 
        **kwargs
    ):
        if self.proj_type == "linear":
            motion = self.motion_fc1(motion)
        elif self.proj_type == "conv":
            motion = self.motion_fc1(motion.transpose(1, 2)).transpose(1, 2)
        else:
            raise NotImplementedError(f"proj_type {self.proj_type} not implemented")
        
        motion, motion_gate = torch.chunk(motion, 2, dim=-1)
        motion = self.motion_act(motion)
        motion = motion * motion_gate
        motion = self.motion_drop1(motion)
        
        if self.proj_type == "linear":
            motion = self.motion_fc2(motion)
        elif self.proj_type == "conv":
            motion = self.motion_fc2(motion.transpose(1, 2)).transpose(1, 2)
        else:
            raise NotImplementedError(f"proj_type {self.proj_type} not implemented")
        motion = self.motion_drop2(motion)
        return motion

class InputLayer(nn.Module):
    def __init__(
        self, 
        in_channels,
        hidden_size, 
        enable_t_modulation=False,
        mapping_size=768,
        bias=True,
        norm_type="rms",
        proj_type="linear",
        norm_eps=1e-6
    ):
        super().__init__()
        self.norm = build_norm(norm_type, num_features=hidden_size, eps=norm_eps, affine=not enable_t_modulation)
        self.proj_type = proj_type
        self.enable_t_modulation = enable_t_modulation
        if proj_type == "linear":
            self.proj = nn.Linear(in_channels, hidden_size, bias=bias)
        else:
            self.proj = nn.Conv1d(in_channels, hidden_size, stride=1, kernel_size=3, padding=1, bias=bias)
        if self.enable_t_modulation:
            self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(mapping_size, 2 * hidden_size, bias=bias))

    def initialize_weights(self):
        if self.enable_t_modulation:
            nn.init.constant_(self.modulation[1].weight, val=0)
            if self.modulation[1].bias is not None:
                nn.init.constant_(self.modulation[1].bias, val=0)

        if isinstance(self.proj, (nn.Linear, nn.Conv1d)):
            nn.init.xavier_uniform_(self.proj.weight)
            if self.proj.bias is not None:
                nn.init.constant_(self.proj.bias, 0)
                
    def forward(self, x, t=None):
        if self.proj_type == "linear":
            x = self.proj(x)
        else:
            x = self.proj(x.transpose(1, 2)).transpose(1, 2)
        x = self.norm(x) if self.norm is not None else x

        if self.enable_t_modulation:
            if t is None:
                raise ValueError("InputLayer with enable_t_modulation=True requires timestep embedding t.")
            if t.dim() == 2:
                t = t.unsqueeze(1)
            if t.dim() != 3:
                raise ValueError(f"t must have shape (B, T, D) or (B, D), but got {t.shape}")
            if t.shape[1] != x.shape[1]:
                if t.shape[1] == 1:
                    t = t.expand(-1, x.shape[1], -1)
                else:
                    raise ValueError(
                        f"InputLayer modulation length mismatch: x.shape[1]={x.shape[1]} vs t.shape[1]={t.shape[1]}"
                    )
            shift, scale = self.modulation(t).chunk(2, dim=-1)
            x = t2i_modulate(x, shift, scale)

        return x

class FinalLayer(nn.Module):
    """
    The final layer of Sana.
    """

    def __init__(
        self, 
        hidden_size, 
        out_channels,
        mapping_size=768,
        bias=True,
        norm_type="rms",
        proj_type="linear",
        simple_modulation=False,
        norm_eps=1e-6
    ):
        super().__init__()
        self.norm_final = build_norm(norm_type, num_features=hidden_size, eps=norm_eps)
        self.proj_type = proj_type
        if self.proj_type == "linear":
            self.proj = nn.Linear(hidden_size,  out_channels, bias=bias)
        else:
            self.proj = nn.Conv1d(hidden_size, out_channels, stride=1, kernel_size=3, padding=1, bias=bias)
        self.simple_modulation = simple_modulation
        if not simple_modulation:
            self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(mapping_size, 2 * hidden_size, bias=bias))
        else:
            self.scale_shift_table = nn.Parameter(torch.randn(2, hidden_size))

    def initialize_weights(self):
        if not self.simple_modulation:
            nn.init.constant_(self.modulation[1].weight, val=0)
            if self.modulation[1].bias is not None:
                nn.init.constant_(self.modulation[1].bias, val=0)
        else:
            nn.init.constant_(self.scale_shift_table, 0)
        nn.init.normal_(self.proj.weight, mean=0.0, std=0.02)
        if self.proj.bias is not None:
            nn.init.constant_(self.proj.bias, val=0)

    def forward(self, x, c):
        if not self.simple_modulation:
            shift, scale = self.modulation(c).chunk(2, dim=1)
        else:
            shift, scale = (self.scale_shift_table[None] + c[:, None]).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        if self.proj_type == "linear":
            x = self.proj(x)
        else:
            x = self.proj(x.transpose(1, 2)).transpose(1, 2)
        return x


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################
class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """

    def __init__(self, hidden_size, frequency_embedding_size=256, bias=True):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=bias),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=bias),
        )
        self.frequency_embedding_size = frequency_embedding_size
    
    
    def initialize_weights(self):
        nn.init.normal_(self.mlp[0].weight, std=0.02)
        nn.init.normal_(self.mlp[2].weight, std=0.02)

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32, device=t.device) / half
        )
        args = t.unsqueeze(-1).float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size).to(self.dtype)
        t_emb = self.mlp(t_freq)
        return t_emb

    @property
    def dtype(self):
        try:
            return next(self.parameters()).dtype
        except StopIteration:
            return torch.float32


class MPFourier(torch.nn.Module):
    def __init__(self, num_channels, bandwidth=1):
        super().__init__()
        self.register_buffer('freqs', 2 * np.pi * torch.randn(num_channels) * bandwidth)
        self.register_buffer('phases', 2 * np.pi * torch.rand(num_channels))

    def forward(self, x):
        if len(x.shape) == 2:
            bs, n = x.shape
            y = x.to(torch.float32).view(-1)
        else:
            y = x.to(torch.float32)
        y = y.ger(self.freqs.to(torch.float32))
        y = y + self.phases.to(torch.float32)
        y = y.cos() * np.sqrt(2)
        if len(x.shape) == 2:
            y = y.view(bs, n, -1)
        return y.to(x.dtype)

class MPFourierTimeEmbedding(nn.Module):
    def __init__(self, hidden_size, num_embedding_channels=768, bias=True, reduce_channels=64):
        super().__init__()
        self.time_embed = nn.Sequential(
            nn.Linear(num_embedding_channels, hidden_size, bias=bias),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size - reduce_channels, bias=bias),
        )
        self.time_embed_mp = MPFourier(num_embedding_channels)
    
    def forward(self, x):
        time_embed_mp = self.time_embed_mp(x.float())
        time_embed_mp = self.time_embed(time_embed_mp)
        return time_embed_mp

class PoolContext(nn.Module):
    def __init__(self, num_embedding_channels, hidden_size, bias=True):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Sequential(
            nn.Linear(num_embedding_channels, hidden_size, bias=bias),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=bias),
        )
    
    def forward(self, x):
        x = self.pool(x.transpose(1, 2)).squeeze(2)
        x = self.proj(x)
        return x

class CaptionEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Does not handle label dropout for classifier-free guidance.
    """

    def __init__(
        self,
        in_channels,
        hidden_size,
        y_proj_type="mlp",
        act_type="gelu",
        y_norm_type="rms",
        bias=True,
        dropout=0.0, 
        norm_eps=1e-6,
        drop_condition=False,
    ):
        super().__init__()
        self.in_channels = in_channels  # 保存输入维度，用于DDP兼容性
        self.y_proj_type = y_proj_type
        if y_proj_type == "mlp":
            self.y_proj = Mlp(
                in_features=in_channels,
                hidden_features=hidden_size,
                out_features=hidden_size,
                act_type=act_type,
                drop=dropout,
                bias=bias
            )
            
        elif y_proj_type == "linear":
            self.y_proj = nn.Sequential(
                nn.Linear(in_features=in_channels, out_features=hidden_size, bias=bias),
                nn.Dropout(dropout)
            )
        self.attention_y_norm = build_norm(y_norm_type, num_features=hidden_size, eps=norm_eps)
        self.drop_condition = drop_condition
        if drop_condition:
            self.null = nn.Parameter(torch.zeros((1, 1, hidden_size)))

    def forward(self, caption, seq_len=None):
        if not self.drop_condition: # 外面直接给null embedding的话，就不需要再投影了
            caption = self.y_proj(caption)
        else:
            # 分离None和非None的索引
            non_none_indices = [i for i, c in enumerate(caption) if c is not None]
            none_indices = [i for i, c in enumerate(caption) if c is None]
            
            # 批量处理非None的caption
            seq_len_inferred = None
            projected = None
            if non_none_indices:
                # 确保每个caption有batch维度 (1, seq_len, in_channels)，然后沿dim=0拼接
                non_none_captions = torch.cat([c.unsqueeze(0) if c.dim() == 2 else c for c in [caption[i] for i in non_none_indices]], dim=0)
                projected = self.y_proj(non_none_captions)
                seq_len_inferred = projected.shape[1]  # 获取序列长度用于广播self.null
            else:
                # 全是None的情况：为了确保DDP中所有参数都被使用，需要进行dummy forward
                # 使用传入的seq_len或默认值1来创建dummy输入
                dummy_seq_len = seq_len if seq_len is not None else 1
                # 使用保存的in_channels创建dummy输入
                # 创建dummy输入并调用y_proj以确保参数被使用（结果不用于最终输出）
                dummy_input = torch.zeros(1, dummy_seq_len, self.in_channels, device=self.null.device, dtype=self.null.dtype)
                _ = self.y_proj(dummy_input)  # Dummy forward，确保参数参与计算
            
            # 构建结果列表
            result = [None] * len(caption)
            for idx, i in enumerate(non_none_indices):
                result[i] = projected[idx:idx+1]  # 保持batch维度 (1, seq_len, hidden_size)
            
            # 处理None的部分，需要将self.null广播到正确的seq_len
            # 确定 seq_len 用于广播
                final_seq_len = seq_len_inferred if seq_len_inferred is not None else (seq_len if seq_len is not None else 1)
                if final_seq_len == 1:
                    null_expanded = self.null
                else:
                    null_expanded = self.null.expand(-1, final_seq_len, -1)  # (1, seq_len, hidden_size)
            
            if none_indices:
                for i in none_indices:
                    result[i] = null_expanded
            else:
                # DDP兼容性：即使没有None索引，也要确保self.null参与计算图
                # 通过加上一个乘以0的null来实现，不影响数值结果
                null_dummy = null_expanded * 0.0
                for idx, i in enumerate(non_none_indices):
                    result[i] = result[i] + null_dummy
            
            caption = torch.cat(result, dim=0)
        caption = self.attention_y_norm(caption)
        return caption
    
    def initialize_weights(self):
        if self.y_proj_type == "mlp":
            nn.init.normal_(self.y_proj.fc1.weight, std=0.02)
            nn.init.normal_(self.y_proj.fc2.weight, std=0.02)
        else:
            nn.init.normal_(self.y_proj[0].weight, std=0.02)


class EMAGECaptionEmbedder(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        lang_model_path: str,
        in_channels: int = 300,
        text_fix_pretrained: bool = False,
        y_proj_type: str = "mlp",
        act_type: str = "gelu",
        y_norm_type: str = "rms",
        bias: bool = True,
        dropout: float = 0.0,
        norm_eps: float = 1e-6,
        **kwargs,
    ):
        super().__init__()
        with open(
            osp.join(lang_model_path, "weights", "vocab.pkl"), 'rb') as f:
            self.lang_model = pickle.load(f)
            pre_trained_embedding = self.lang_model.word_embedding_weights
        self.text_pre_encoder = nn.Embedding.from_pretrained(
            torch.FloatTensor(pre_trained_embedding),
            freeze=text_fix_pretrained,
        )
        self.y_proj_type = y_proj_type
        if y_proj_type == "mlp":
            self.y_proj = Mlp(
                in_features=in_channels,
                hidden_features=hidden_size,
                out_features=hidden_size,
                act_type=act_type,
                drop=dropout,
                bias=bias
            )
            
        elif y_proj_type == "linear":
            self.y_proj = nn.Sequential(
                nn.Linear(in_features=in_channels, out_features=hidden_size, bias=bias),
                nn.Dropout(dropout)
            )
        self.attention_y_norm = build_norm(y_norm_type, num_features=hidden_size, eps=norm_eps)
        
    def forward(self, text):
        text = text.to(torch.long)
        text_emb = self.text_pre_encoder(text)
        text_emb = self.y_proj(text_emb)
        text_emb = self.attention_y_norm(text_emb)
        return text_emb

    def initialize_weights(self):
        if self.y_proj_type == "mlp":
            nn.init.normal_(self.y_proj.fc1.weight, std=0.02)
            nn.init.normal_(self.y_proj.fc2.weight, std=0.02)
        else:
            nn.init.normal_(self.y_proj[0].weight, std=0.02)
        nn.init.normal_(self.text_pre_encoder.weight, std=0.02)


class StyleEmbedding(nn.Module):
    def __init__(self, num_persons, dim, bias=True):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(num_persons, dim * 2, bias=bias),
            nn.SiLU(),
            nn.Linear(dim * 2, dim, bias=bias)
        )

    def forward(self, t):
        return self.mlp(t)

class SimpleSpeechModel(nn.Module):
    def __init__(self, 
        feature_dim=512, 
        output_feat_dim=100,
        num_person=31):
        super(SimpleSpeechModel, self).__init__()
        self.num_person = num_person
        self.pos_enc = Summer(PositionalEncoding1D(feature_dim))
        decoder_layer = nn.TransformerDecoderLayer(d_model=feature_dim, 
                                                   nhead=4,
                                                   dim_feedforward=2 * feature_dim, 
                                                   batch_first=True, 
                                                   norm_first=True)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=4)
        self.audio_feature_map = nn.Linear(1024 + 64, feature_dim)
        self.vertice_map_r = nn.Linear(feature_dim, output_feat_dim)
        self.output_feat_dim = output_feat_dim

        self.style_embed = StyleEmbedding(num_persons=num_person, dim=64)

        nn.init.constant_(self.vertice_map_r.weight, 0)
        nn.init.constant_(self.vertice_map_r.bias, 0)

    def forward(self, huberts, pid):
        one_hots = F.one_hot(pid, num_classes=self.num_person).float().to(huberts.device)
        style_embed = self.style_embed(one_hots).unsqueeze(1).expand(-1, huberts.shape[1], -1)
        
        vertice_input = self.audio_feature_map(torch.cat([huberts, style_embed], dim=-1))
        vertice_input = self.pos_enc(vertice_input)
        vertice_out = self.transformer_decoder(vertice_input, vertice_input)
        vertice_out = self.vertice_map_r(vertice_out)
        return vertice_out


class NoSpeechModel(nn.Module):
    def __init__(self, output_feat_dim=100, **kwargs):
        super(NoSpeechModel, self).__init__()
        self.output_feat_dim = output_feat_dim
        self.model = nn.Identity()
    
    def forward(self, huberts, pid):
        audio_len = huberts.shape[1]
        return torch.zeros(huberts.shape[0], audio_len, self.output_feat_dim).to(huberts.device)


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega  # (D/2,)
    if isinstance(pos, int):
        pos = np.array(pos)
    
    pos = pos.reshape(-1)  # (M,)
    out = np.einsum("m,d->md", pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb
