# 实际上就是DiT
import os

import numpy as np
import torch
import torch.nn as nn
from timm.layers import DropPath
import torch.nn.functional as F
from diffusion.model.nets.positional_embeding import RopePosEmbed
from positional_encodings.torch_encodings import PositionalEncoding1D, Summer
from diffusion.model.nets.basic_modules import (
    DoubleStreamGatedMLP,
    DoubleStreamMLP,
    t2i_modulate,
    StyleEmbedding,
)
from diffusion.model.nets.attention_modules import (
    Attention,
    MultiHeadCrossAttention,
) 
from diffusion.model.norms import build_norm
from diffusion.model.utils import auto_grad_checkpoint
from diffusion.utils.dist_utils import get_rank
from diffusion.utils.logger import get_root_logger
from diffusion.utils.config import instantiate_from_config
from omegaconf import DictConfig
from copy import deepcopy

DEFAULT_CAPTION_EMBEDDER_CONFIG = DictConfig(dict(
    target="diffusion.model.nets.basic_modules.CaptionEmbedder",
    params=dict(
        in_channels=None,
        hidden_size=None,
        y_proj_type="mlp",
        act_type="gelu",
        y_norm_type="rms",
        bias=True,
        dropout=0.0, 
        norm_eps=1e-6,
    )
))

DEFAULT_TIMESTEP_EMBEDDER_CONFIG = DictConfig(dict(
    target="diffusion.model.nets.basic_modules.TimestepEmbedder",
    params=dict(
        hidden_size=None,
        frequency_embedding_size=256,
        bias=True
    )
))

DEFAULT_BLOCK_CONFIG=DictConfig(dict(
    target="diffusion.model.nets.sana_emage_transconv.SanaEmageTransConvBlock",
    params=dict(
        hidden_size=None,
        trans_hidden_size=None,
        num_heads=8,
        head_dim=None,
        mapping_size=768,
        simple_modulation=False,
        bias=True,
        norm_type="rms",
        attn_type="flash",
        prompt_attn_type=None,
        qk_norm_type="identity",
        cross_attn_type=None,
        cross_qk_norm_type="identity",
        ffn_type="mlp",
        ffn_act_type="gelu",
        ffn_proj_type="linear", 
        ffn_mlp_ratio=4.0,
        attn_drop=0.0,
        attn_proj_drop=0.0,
        cross_attn_drop=0.0,
        cross_attn_proj_drop=0.0,
        ffn_drop=0.0,
        drop_path=0.0,
        norm_eps=1e-6,
    )
))

DEFAULT_INPUT_CONFIG=DictConfig(dict(
    target="diffusion.model.nets.basic_modules.InputLayer",
    params=dict(
        in_channels=None,
        hidden_size=None,         
        bias=True,
        norm_type="rms",
        proj_type="linear",
        norm_eps=1e-6
    )
))

DEFAULT_FINALLAYER_CONFIG=DictConfig(dict(
    target="diffusion.model.nets.basic_modules.FinalLayer",
    params=dict(
        hidden_size=None, 
        out_channels=None,
        mapping_size=768,
        bias=True,
        norm_type="rms",
        proj_type="linear",
        simple_modulation=False,
        norm_eps=1e-6,
    )
))


class SanaEmageTransConvBlock(nn.Module):
    """
    A Sana block with global shared adaptive layer norm (adaLN-single) conditioning.
    """

    def __init__(
        self,
        hidden_size,
        trans_hidden_size,
        num_heads=8,
        mapping_size=768,
        head_dim=None,
        simple_modulation=False,
        bias=True,
        norm_type="rms",
        attn_type="flash",
        prompt_attn_type="flash",
        qk_norm_type="identity",
        cross_attn_type="flash",
        cross_qk_norm_type="identity",
        ffn_type="mlp",
        ffn_act_type="gelu",
        ffn_proj_type="linear", 
        ffn_mlp_ratio=4.0,
        attn_drop=0.0,
        attn_proj_drop=0.0,
        cross_attn_drop=0.0,
        cross_attn_proj_drop=0.0,
        ffn_drop=0.0,
        drop_path=0.0,
        norm_eps=1e-6,
    ):
        super().__init__()
        self.hand_norm1 = build_norm(norm_type, num_features=hidden_size, affine=False, eps=norm_eps)
        self.body_norm1 = build_norm(norm_type, num_features=hidden_size, affine=False, eps=norm_eps)
        self.hand_norm2 = build_norm(norm_type, num_features=hidden_size, affine=False, eps=norm_eps)
        self.body_norm2 = build_norm(norm_type, num_features=hidden_size, affine=False, eps=norm_eps)        
        self.trans_norm1 = build_norm(norm_type, num_features=trans_hidden_size, affine=False, eps=norm_eps)
        self.trans_norm2 = build_norm(norm_type, num_features=trans_hidden_size, affine=False, eps=norm_eps)
        
        self.hand_drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.body_drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.trans_drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        
        self.attn = Attention(
            dim=hidden_size,
            num_heads=num_heads,
            head_dim=head_dim,
            bias=bias,
            attn_type=attn_type,
            qk_norm_type=qk_norm_type,
            attn_drop=attn_drop, 
            proj_drop=attn_proj_drop,
        )
        self.trans_conv1 = nn.Sequential(
            nn.SiLU(),
            nn.Conv1d(in_channels=trans_hidden_size, out_channels=trans_hidden_size, kernel_size=9, padding=4, bias=bias),
        )
        
        self.hand_cross_attn = MultiHeadCrossAttention(
            dim=hidden_size,
            num_heads=num_heads,
            head_dim=head_dim,
            bias=bias,
            attn_type=cross_attn_type,
            qk_norm_type=cross_qk_norm_type,
            attn_drop=cross_attn_drop, 
            proj_drop=cross_attn_proj_drop,
        )
        self.body_cross_attn = deepcopy(self.hand_cross_attn)
        
        if ffn_type == "mlp":
            self.mlp = DoubleStreamMLP(
                in_features=hidden_size, 
                hidden_features=int(hidden_size * ffn_mlp_ratio),  
                act_type=ffn_act_type,
                proj_type=ffn_proj_type,
                bias=bias, 
                dropout=ffn_drop
            )
        elif ffn_type == "gated_mlp":
            self.mlp = DoubleStreamGatedMLP(
                in_features=hidden_size, 
                hidden_features=int(hidden_size * ffn_mlp_ratio),  
                act_type=ffn_act_type,
                proj_type=ffn_proj_type,
                bias=bias, 
                dropout=ffn_drop
            )
        else:
            raise NotImplementedError(f"ffn_type {ffn_type} is not implemented")
        self.trans_conv2 = nn.Sequential(
            nn.SiLU(),
            nn.Conv1d(in_channels=trans_hidden_size, out_channels=trans_hidden_size, kernel_size=9, padding=4, bias=bias),
        )
              
        self.simple_modulation = simple_modulation
        if self.simple_modulation:
            # like pixart
            self.hand_scale_shift_table = nn.Parameter(torch.randn(6, hidden_size))
            self.body_scale_shift_table = nn.Parameter(torch.randn(6, hidden_size))
            self.trans_scale_shift_table = nn.Parameter(torch.randn(6, trans_hidden_size))
        else:
            self.hand_time_in = nn.Sequential(nn.SiLU(), nn.Linear(mapping_size, 6 * hidden_size, bias=bias))   
            self.body_time_in = nn.Sequential(nn.SiLU(), nn.Linear(mapping_size, 6 * hidden_size, bias=bias))   
            self.trans_time_in = nn.Sequential(nn.SiLU(), nn.Linear(mapping_size, 6 * trans_hidden_size, bias=bias))
        
    def initialize_weights(self):
        if self.simple_modulation:
            nn.init.constant_(self.hand_scale_shift_table, 0)
            nn.init.constant_(self.body_scale_shift_table, 0)
            nn.init.constant_(self.trans_scale_shift_table, 0)
        else:
            nn.init.constant_(self.hand_time_in[1].weight, val=0)
            if self.hand_time_in[1].bias is not None:
                nn.init.constant_(self.hand_time_in[1].bias, val=0)
                
            nn.init.constant_(self.body_time_in[1].weight, val=0)
            if self.body_time_in[1].bias is not None:
                nn.init.constant_(self.body_time_in[1].bias, val=0)
            
            nn.init.constant_(self.trans_time_in[1].weight, val=0)
            if self.trans_time_in[1].bias is not None:
                nn.init.constant_(self.trans_time_in[1].bias, val=0)

    def forward(
        self, 
        hand: torch.FloatTensor, 
        body: torch.FloatTensor,
        trans: torch.FloatTensor,
        t: torch.FloatTensor,
        rotary_emb: torch.FloatTensor = None,
        hand_prompt: torch.FloatTensor = None, 
        hand_prompt_mask: torch.FloatTensor = None,
        body_prompt: torch.FloatTensor = None,
        body_prompt_mask: torch.FloatTensor = None,
        **kwargs
    ):
        assert hand.shape[0] == body.shape[0] == trans.shape[0], "batch size must be the same"
        B, N, C = hand.shape

        if self.simple_modulation:
            hand_shift_msa, hand_scale_msa, hand_gate_msa, \
                hand_shift_mlp, hand_scale_mlp, hand_gate_mlp = (
                self.hand_scale_shift_table[None] + t.reshape(B, 6, -1)
            ).chunk(6, dim=1)
            body_shift_msa, body_scale_msa, body_gate_msa, \
                body_shift_mlp, body_scale_mlp, body_gate_mlp = (
                self.body_scale_shift_table[None] + t.reshape(B, 6, -1)
            ).chunk(6, dim=1)
            trans_shift1, trans_scale1, trans_gate1, trans_shift2, trans_scale2, trans_gate2 = (
                self.trans_scale_shift_table[None] + t.reshape(B, 6, -1)
            ).chunk(6, dim=1)
        else:
            hand_shift_msa, hand_scale_msa, hand_gate_msa, \
                hand_shift_mlp, hand_scale_mlp, hand_gate_mlp\
                = self.hand_time_in(t.unsqueeze(1)).chunk(6, dim=-1)
            body_shift_msa, body_scale_msa, body_gate_msa, \
                body_shift_mlp, body_scale_mlp, body_gate_mlp\
                = self.body_time_in(t.unsqueeze(1)).chunk(6, dim=-1)
            trans_shift1, trans_scale1, trans_gate1, trans_shift2, trans_scale2, trans_gate2\
                = self.trans_time_in(t.unsqueeze(1)).chunk(6, dim=-1)
        
        ##################################### The first layer #######################################33
        residual_hand = self.hand_norm1(hand)
        residual_body = self.body_norm1(body)
        residual_trans = self.trans_norm1(trans)
        
        residual_hand = t2i_modulate(residual_hand, hand_shift_msa, hand_scale_msa)
        residual_body = t2i_modulate(residual_body, body_shift_msa, body_scale_msa)
        residual_trans = t2i_modulate(residual_trans, trans_shift1, trans_scale1)

        residual = torch.cat([residual_hand, residual_body], dim=1)
        pos_embed = None
        if rotary_emb is not None:
            pos_embed = [ torch.cat([emb, emb], dim=0) for emb in rotary_emb ]
        residual = self.attn(residual, pos_embed)
        residual_hand, residual_body = residual.chunk(2, dim=1)
        residual_trans = self.trans_conv1(residual_trans.transpose(1, 2)).transpose(1, 2)
    
        residual_hand = hand_gate_msa * residual_hand
        residual_body = body_gate_msa * residual_body
        residual_trans = trans_gate1 * residual_trans
        
        hand = hand + self.hand_drop_path(residual_hand)
        body = body + self.body_drop_path(residual_body)
        trans = trans + self.trans_drop_path(residual_trans)
        
        ######################################## The second layer ########################################
        residual_hand = self.hand_cross_attn(hand, hand_prompt, None, hand_prompt_mask)
        residual_body = self.body_cross_attn(body, body_prompt, None, body_prompt_mask)

        hand = hand + self.hand_drop_path(residual_hand)
        body = body + self.body_drop_path(residual_body)
        
        ######################################## The third layer ########################################
        residual_hand = self.hand_norm2(hand)
        residual_body = self.body_norm2(body)
        residual_trans = self.trans_norm2(trans)
        
        residual_hand = t2i_modulate(residual_hand, hand_shift_mlp, hand_scale_mlp)
        residual_body = t2i_modulate(residual_body, body_shift_mlp, body_scale_mlp)
        residual_trans = t2i_modulate(residual_trans, trans_shift2, trans_scale2)
        
        residual_hand, residual_body = self.mlp(residual_hand, residual_body)
        residual_trans = self.trans_conv2(residual_trans.transpose(1, 2)).transpose(1, 2)
        
        residual_hand = hand_gate_mlp * residual_hand
        residual_body = body_gate_mlp * residual_body
        residual_trans = trans_gate2 * residual_trans
        
        hand = hand + self.hand_drop_path(residual_hand)
        body = body + self.body_drop_path(residual_body)
        trans = trans + self.trans_drop_path(residual_trans)
        
        return hand, body, trans
    
class SanaEmageTransConvSelfAttnBlock(nn.Module):
    """
    A Sana block with global shared adaptive layer norm (adaLN-single) conditioning.
    """

    def __init__(
        self,
        hidden_size,
        trans_hidden_size,
        num_heads=8,
        mapping_size=768,
        head_dim=None,
        simple_modulation=False,
        bias=True,
        norm_type="rms",
        attn_type="flash",
        prompt_attn_type="flash",
        qk_norm_type="identity",
        cross_attn_type="flash",
        cross_qk_norm_type="identity",
        ffn_type="mlp",
        ffn_act_type="gelu",
        ffn_proj_type="linear", 
        ffn_mlp_ratio=4.0,
        attn_drop=0.0,
        attn_proj_drop=0.0,
        cross_attn_drop=0.0,
        cross_attn_proj_drop=0.0,
        ffn_drop=0.0,
        drop_path=0.0,
        norm_eps=1e-6,
    ):
        super().__init__()
        self.hand_norm1 = build_norm(norm_type, num_features=hidden_size, affine=False, eps=norm_eps)
        self.body_norm1 = build_norm(norm_type, num_features=hidden_size, affine=False, eps=norm_eps)
        self.hand_norm2 = build_norm(norm_type, num_features=hidden_size, affine=False, eps=norm_eps)
        self.body_norm2 = build_norm(norm_type, num_features=hidden_size, affine=False, eps=norm_eps)        
        self.trans_norm1 = build_norm(norm_type, num_features=trans_hidden_size, affine=False, eps=norm_eps)
        self.trans_norm2 = build_norm(norm_type, num_features=trans_hidden_size, affine=False, eps=norm_eps)
        
        self.hand_drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.body_drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.trans_drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        
        self.attn = Attention(
            dim=hidden_size,
            num_heads=num_heads,
            head_dim=head_dim,
            bias=bias,
            attn_type=attn_type,
            qk_norm_type=qk_norm_type,
            attn_drop=attn_drop, 
            proj_drop=attn_proj_drop,
        )
        self.attn2 = deepcopy(self.attn)
        self.trans_conv1 = nn.Sequential(
            nn.SiLU(),
            nn.Conv1d(in_channels=trans_hidden_size, out_channels=trans_hidden_size, kernel_size=9, padding=4, bias=bias),
        )
        
        self.hand_cross_attn = MultiHeadCrossAttention(
            dim=hidden_size,
            num_heads=num_heads,
            head_dim=head_dim,
            bias=bias,
            attn_type=cross_attn_type,
            qk_norm_type=cross_qk_norm_type,
            attn_drop=cross_attn_drop, 
            proj_drop=cross_attn_proj_drop,
        )
        self.body_cross_attn = deepcopy(self.hand_cross_attn)
        
        if ffn_type == "mlp":
            self.mlp = DoubleStreamMLP(
                in_features=hidden_size, 
                hidden_features=int(hidden_size * ffn_mlp_ratio),  
                act_type=ffn_act_type,
                proj_type=ffn_proj_type,
                bias=bias, 
                dropout=ffn_drop
            )
        elif ffn_type == "gated_mlp":
            self.mlp = DoubleStreamGatedMLP(
                in_features=hidden_size, 
                hidden_features=int(hidden_size * ffn_mlp_ratio),  
                act_type=ffn_act_type,
                proj_type=ffn_proj_type,
                bias=bias, 
                dropout=ffn_drop
            )
        else:
            raise NotImplementedError(f"ffn_type {ffn_type} is not implemented")
        self.trans_conv2 = nn.Sequential(
            nn.SiLU(),
            nn.Conv1d(in_channels=trans_hidden_size, out_channels=trans_hidden_size, kernel_size=9, padding=4, bias=bias),
        )
              
        self.simple_modulation = simple_modulation
        if self.simple_modulation:
            # like pixart
            self.hand_scale_shift_table = nn.Parameter(torch.randn(6, hidden_size))
            self.body_scale_shift_table = nn.Parameter(torch.randn(6, hidden_size))
            self.trans_scale_shift_table = nn.Parameter(torch.randn(6, trans_hidden_size))
        else:
            self.hand_time_in = nn.Sequential(nn.SiLU(), nn.Linear(mapping_size, 6 * hidden_size, bias=bias))   
            self.body_time_in = nn.Sequential(nn.SiLU(), nn.Linear(mapping_size, 6 * hidden_size, bias=bias))   
            self.trans_time_in = nn.Sequential(nn.SiLU(), nn.Linear(mapping_size, 6 * trans_hidden_size, bias=bias))
        
    def initialize_weights(self):
        if self.simple_modulation:
            nn.init.constant_(self.hand_scale_shift_table, 0)
            nn.init.constant_(self.body_scale_shift_table, 0)
            nn.init.constant_(self.trans_scale_shift_table, 0)
        else:
            nn.init.constant_(self.hand_time_in[1].weight, val=0)
            if self.hand_time_in[1].bias is not None:
                nn.init.constant_(self.hand_time_in[1].bias, val=0)
                
            nn.init.constant_(self.body_time_in[1].weight, val=0)
            if self.body_time_in[1].bias is not None:
                nn.init.constant_(self.body_time_in[1].bias, val=0)
            
            nn.init.constant_(self.trans_time_in[1].weight, val=0)
            if self.trans_time_in[1].bias is not None:
                nn.init.constant_(self.trans_time_in[1].bias, val=0)

    def forward(
        self, 
        hand: torch.FloatTensor, 
        body: torch.FloatTensor,
        trans: torch.FloatTensor,
        t: torch.FloatTensor,
        rotary_emb: torch.FloatTensor = None,
        hand_prompt: torch.FloatTensor = None, 
        hand_prompt_mask: torch.FloatTensor = None,
        body_prompt: torch.FloatTensor = None,
        body_prompt_mask: torch.FloatTensor = None,
        **kwargs
    ):
        assert hand.shape[0] == body.shape[0] == trans.shape[0], "batch size must be the same"
        B, N, C = hand.shape

        if self.simple_modulation:
            hand_shift_msa, hand_scale_msa, hand_gate_msa, \
                hand_shift_mlp, hand_scale_mlp, hand_gate_mlp = (
                self.hand_scale_shift_table[None] + t.reshape(B, 6, -1)
            ).chunk(6, dim=1)
            body_shift_msa, body_scale_msa, body_gate_msa, \
                body_shift_mlp, body_scale_mlp, body_gate_mlp = (
                self.body_scale_shift_table[None] + t.reshape(B, 6, -1)
            ).chunk(6, dim=1)
            trans_shift1, trans_scale1, trans_gate1, trans_shift2, trans_scale2, trans_gate2 = (
                self.trans_scale_shift_table[None] + t.reshape(B, 6, -1)
            ).chunk(6, dim=1)
        else:
            hand_shift_msa, hand_scale_msa, hand_gate_msa, \
                hand_shift_mlp, hand_scale_mlp, hand_gate_mlp\
                = self.hand_time_in(t.unsqueeze(1)).chunk(6, dim=-1)
            body_shift_msa, body_scale_msa, body_gate_msa, \
                body_shift_mlp, body_scale_mlp, body_gate_mlp\
                = self.body_time_in(t.unsqueeze(1)).chunk(6, dim=-1)
            trans_shift1, trans_scale1, trans_gate1, trans_shift2, trans_scale2, trans_gate2\
                = self.trans_time_in(t.unsqueeze(1)).chunk(6, dim=-1)
        
        ##################################### The first layer #######################################33
        residual_hand = self.hand_norm1(hand)
        residual_body = self.body_norm1(body)
        residual_trans = self.trans_norm1(trans)
        
        residual_hand = t2i_modulate(residual_hand, hand_shift_msa, hand_scale_msa)
        residual_body = t2i_modulate(residual_body, body_shift_msa, body_scale_msa)
        residual_trans = t2i_modulate(residual_trans, trans_shift1, trans_scale1)

        residual_hand = self.attn(residual_hand, rotary_emb)
        residual_body = self.attn2(residual_body, rotary_emb)
        residual_trans = self.trans_conv1(residual_trans.transpose(1, 2)).transpose(1, 2)
    
        residual_hand = hand_gate_msa * residual_hand
        residual_body = body_gate_msa * residual_body
        residual_trans = trans_gate1 * residual_trans
        
        hand = hand + self.hand_drop_path(residual_hand)
        body = body + self.body_drop_path(residual_body)
        trans = trans + self.trans_drop_path(residual_trans)
        
        ######################################## The second layer ########################################
        residual_hand = self.hand_cross_attn(hand, hand_prompt, None, hand_prompt_mask)
        residual_body = self.body_cross_attn(body, body_prompt, None, body_prompt_mask)

        hand = hand + self.hand_drop_path(residual_hand)
        body = body + self.body_drop_path(residual_body)
        
        ######################################## The third layer ########################################
        residual_hand = self.hand_norm2(hand)
        residual_body = self.body_norm2(body)
        residual_trans = self.trans_norm2(trans)
        
        residual_hand = t2i_modulate(residual_hand, hand_shift_mlp, hand_scale_mlp)
        residual_body = t2i_modulate(residual_body, body_shift_mlp, body_scale_mlp)
        residual_trans = t2i_modulate(residual_trans, trans_shift2, trans_scale2)
        
        residual_hand, residual_body = self.mlp(residual_hand, residual_body)
        residual_trans = self.trans_conv2(residual_trans.transpose(1, 2)).transpose(1, 2)
        
        residual_hand = hand_gate_mlp * residual_hand
        residual_body = body_gate_mlp * residual_body
        residual_trans = trans_gate2 * residual_trans
        
        hand = hand + self.hand_drop_path(residual_hand)
        body = body + self.body_drop_path(residual_body)
        trans = trans + self.trans_drop_path(residual_trans)
        
        return hand, body, trans


class SanaEmageTransConv(nn.Module):
    def __init__(
        self,
        motion_in_channels=6,
        expression_in_channels=100,
        trans_in_channels=3,
        hand_joints = 30,
        body_joints = 13 + 9 + 3,
        hidden_size=384,
        trans_hidden_size=128,
        caption_channels=2304,
        mapping_size=768,
        num_person=31,
        depth=28,
        drop_path=0.0,
        inputlayer_config=DEFAULT_INPUT_CONFIG,
        timestep_embedder_config=DEFAULT_TIMESTEP_EMBEDDER_CONFIG,
        caption_embedder_config=DEFAULT_CAPTION_EMBEDDER_CONFIG,
        block_config=DEFAULT_BLOCK_CONFIG,
        finallayer_config=DEFAULT_FINALLAYER_CONFIG,
        work_dir=None,
        use_pe=True,
        pos_embed_type="sincos",
        pe_max_length=10000,
        training_step=1000,
        simple_modulation=False, # for pixart-like only one modulation
        **kwargs,
    ):
        super().__init__()
        self.num_person = num_person
        self.motion_in_channels = motion_in_channels
        self.trans_in_channels = trans_in_channels
        self.expression_in_channels = expression_in_channels
        
        self.hidden_size = hidden_size
        self.trans_hidden_size = trans_hidden_size
        self.depth = depth
        self.pos_embed_type = pos_embed_type
        self.fp32_attention = kwargs.get("use_fp32_attention", False)

        if work_dir is not None:
            logger = get_root_logger(os.path.join(work_dir, "train_log.log"))
            logger = logger.info
        else:
            logger = print
        
        self.logger = logger
        joints = hand_joints + body_joints
        ##  解决 motion 相关的初始化
        inputlayer_config["params"]["hidden_size"] = self.hidden_size
        inputlayer_config["params"]["in_channels"] = hand_joints * self.motion_in_channels
        self.hand_in = instantiate_from_config(inputlayer_config)
        self.seed_hand_in = instantiate_from_config(inputlayer_config)
        inputlayer_config["params"]["in_channels"] = body_joints * self.motion_in_channels
        self.body_in = instantiate_from_config(inputlayer_config)
        self.seed_body_in = instantiate_from_config(inputlayer_config)
        
        inputlayer_config["params"]["in_channels"] = joints * self.motion_in_channels + self.expression_in_channels + self.trans_in_channels
        self.xhand_in = instantiate_from_config(inputlayer_config)
        inputlayer_config["params"]["in_channels"] = joints * self.motion_in_channels + self.expression_in_channels + self.trans_in_channels
        self.xbody_in = instantiate_from_config(inputlayer_config)
        
        inputlayer_config["params"]["in_channels"] = self.hidden_size * 3
        self.hand_fuse = instantiate_from_config(inputlayer_config)
        inputlayer_config["params"]["in_channels"] = self.hidden_size * 3
        self.body_fuse = instantiate_from_config(inputlayer_config)
        
        ## 解决 trans 相关的初始化
        inputlayer_config["params"]["hidden_size"] = self.trans_hidden_size
        inputlayer_config["params"]["in_channels"] = self.trans_in_channels
        self.trans_in = instantiate_from_config(inputlayer_config)
        self.seed_trans_in = instantiate_from_config(inputlayer_config)
        inputlayer_config["params"]["in_channels"] = joints * self.motion_in_channels + self.expression_in_channels + self.trans_in_channels
        self.xtrans_in = instantiate_from_config(inputlayer_config)
        inputlayer_config["params"]["in_channels"] = self.trans_hidden_size * 2
        self.trans_fuse = instantiate_from_config(inputlayer_config)
        
        timestep_embedder_config["params"]["hidden_size"] = mapping_size
        self.motion_t_embedder = instantiate_from_config(timestep_embedder_config)
        self.simple_modulation = simple_modulation
        self.style_embed = StyleEmbedding(num_person, 64)
        
        self.use_pe = use_pe
        self.pe_max_length = pe_max_length
        # Will use fixed sin-cos embedding:
        self.register_buffer("pos_embed", torch.zeros(1, pe_max_length, hidden_size))
        if self.simple_modulation:
            assert hidden_size == trans_hidden_size,\
                "hidden_size must be equal to trans_hidden_size for simple modulation"
            self.t_block = nn.Sequential(nn.SiLU(), nn.Linear(mapping_size, 6 * hidden_size, bias=True))
        
        caption_embedder_config["params"]["in_channels"] = caption_channels
        caption_embedder_config["params"]["hidden_size"] = hidden_size
        self.txt_embedder = instantiate_from_config(caption_embedder_config)
        
        drop_path = [x.item() for x in torch.linspace(0, drop_path, depth)]  # stochastic depth decay rule
        block_config["params"]["hidden_size"] = hidden_size
        block_config["params"]["mapping_size"] = mapping_size
        block_config["params"]["trans_hidden_size"] = trans_hidden_size
        block_config["params"]["simple_modulation"] = simple_modulation
        self.blocks = nn.ModuleList()
        for i in range(depth):
            block_config["params"]["drop_path"] = drop_path[i]
            self.blocks.append(instantiate_from_config(block_config))
        
        finallayer_config["params"]["hidden_size"] = hidden_size
        finallayer_config["params"]["mapping_size"] = mapping_size
        finallayer_config["params"]["out_channels"] = hand_joints * self.motion_in_channels
        self.hand_final_layer = instantiate_from_config(finallayer_config)
        finallayer_config["params"]["out_channels"] = body_joints * self.motion_in_channels
        self.body_final_layer = instantiate_from_config(finallayer_config)
        finallayer_config["params"]["hidden_size"] = self.trans_hidden_size
        finallayer_config["params"]["out_channels"] = self.trans_in_channels
        self.trans_final_layer = instantiate_from_config(finallayer_config)
        
        self.training_step = training_step
        if self.use_pe and self.pos_embed_type in ["sincos", "rope"]:
            if self.pos_embed_type == "sincos":
                # Initialize (and freeze) pos_embed by sin-cos embedding:
                pos_embed = get_1d_sincos_pos_embed_from_grid(
                    self.pos_embed.shape[-1],
                    pe_max_length
                )
                self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
            elif self.pos_embed_type == "rope":
                self.pos_embed = RopePosEmbed(theta=10000, axes_dim=[hidden_size // self.blocks[0].attn.num_heads])
            elif self.pos_embed_type == "none":
                pass
            else:
                raise NotImplementedError(f"pos embed type {pos_embed_type} is not supported.")
        self.initialize_weights()

        if get_rank() == 0:
            logger(
                f"use pe: {use_pe}, pos embed type: {pos_embed_type}"
            )
            attn_type = block_config["params"]["attn_type"]
            ffn_type = block_config["params"]["ffn_type"]
            qk_norm_type = block_config["params"]["qk_norm_type"]
            cross_attn_type = block_config["params"]["cross_attn_type"]
            cross_qk_norm_type = block_config["params"]["cross_qk_norm_type"]
            logger(
                f"attention type: {attn_type}; ffn type: {ffn_type}; self-attn qk norm: {qk_norm_type}; "
                f"cross-attn type: {cross_attn_type};  cross-attn qk norm: {cross_qk_norm_type}; "
                f"autocast linear attn: {os.environ.get('AUTOCAST_LINEAR_ATTN', False)}"
            )
            
    def forward(
        self,  
        motion, 
        timestep,
        expression,
        audio = None,
        audio_mask = None,
        txt = None,
        txt_mask = None,
        **kwargs
    ):
        """
        Forward pass of Sana.
            motion: (N, T, D) tensor of motion
            expression: (N, T, D) tensor of expression
            timestep: (N, ) tensor of timestep
            cond: (N, L, D) tensor of condition
            cond_mask: (N, L) tensor of condition mask
        """
        motion = motion.to(self.dtype)
        timestep = timestep.to(self.dtype)
        expression = expression.to(self.dtype)
            
        if audio is not None:
            cond = audio.to(self.dtype)
            cond_mask = audio_mask.to(self.dtype) if audio_mask is not None else audio_mask
        
        if txt is not None:
            cond = txt.to(self.dtype)
            cond_mask = txt_mask.to(self.dtype) if txt_mask is not None else txt_mask
        
        id = kwargs["speaker_id"]
        hand_indices = kwargs["hand_indices"]
        upper_indices = kwargs["upper_indices"]
        lower_indices = kwargs["lower_indices"]
        face_joint_indices = kwargs["face_joint_indices"]
        trans_indices = kwargs['trans_indices']
        
        pose_prompt = kwargs['pose_prompt'].to(self.dtype)
        pose_prompt_mask = kwargs['pose_prompt_mask']
        trans_prompt = kwargs['trans_prompt'].to(self.dtype)
        trans_prompt_mask = kwargs['trans_prompt_mask']

        body_indices = np.concatenate([upper_indices, lower_indices, face_joint_indices], axis=0)
        #########################################################
        # for expression
        #########################################################
        hand = motion[..., hand_indices]
        body = motion[..., body_indices]
        trans = motion[..., trans_indices]
        
        trans_prompt_mask = trans_prompt_mask.to(device=trans.device, dtype=torch.bool)
        pose_prompt_mask  = pose_prompt_mask.to(device=hand.device, dtype=torch.bool)
        assert hand.shape[:2] == body.shape[:2] == trans.shape[:2], (hand.shape, body.shape, trans.shape)
        assert pose_prompt.shape[:2] == hand.shape[:2], (pose_prompt.shape, hand.shape)
        assert pose_prompt_mask.shape[:2] == (hand.shape[0], hand.shape[1]), pose_prompt_mask.shape
        assert trans_prompt_mask.shape[:2] == (trans.shape[0], trans.shape[1]), trans_prompt_mask.shape
        
        
        hand = self.hand_in(hand)
        body = self.body_in(body)
        trans = self.trans_in(trans)
    
        x = torch.cat([motion, expression], dim=-1)
        xhand = self.xhand_in(x)
        xbody = self.xbody_in(x)
        xtrans = self.xtrans_in(x)
        
        
        seed_hand = pose_prompt[..., hand_indices]
        seed_body = pose_prompt[..., body_indices]
        seed_trans = trans_prompt
        
        seed_hand = self.seed_hand_in(seed_hand)
        seed_body = self.seed_body_in(seed_body)
        seed_trans = self.seed_trans_in(seed_trans)
        
        cond = self.txt_embedder(cond)  # (N, L, D)
        
        hand = self.hand_fuse(torch.cat([hand, xhand, cond], dim=-1))
        body = self.body_fuse(torch.cat([body, xbody, cond], dim=-1))
        trans = self.trans_fuse(torch.cat([trans, xtrans], dim=-1))
        
        # t = timestep / self.training_step # 归一�?
        t = self.motion_t_embedder(timestep)  # (N, D)
        one_hots = F.one_hot(id, num_classes=self.num_person).float().to(x.device)
        t = torch.cat([t, self.style_embed(one_hots)], dim=-1)
        if self.simple_modulation:
            t0 = self.t_block(t)
        else:
            t0 = t
        
        pos_embed = None
        if self.use_pe:
            if self.pos_embed_type == "sincos":
                hand = hand + self.pos_embed[:,:hand.shape[1], :].to(
                    device=hand.device, dtype=hand.dtype)  # (N, T, D)
                body = body + self.pos_embed[:,:body.shape[1], :].to(
                    device=body.device, dtype=body.dtype)  # (N, T, D)
                trans = trans + self.pos_embed[:,:trans.shape[1], :]  # (N, T, D)
                seed_hand = seed_hand + self.pos_embed[:,:seed_hand.shape[1], :]  # (N, T, D)
                seed_body = seed_body + self.pos_embed[:,:seed_body.shape[1], :]  # (N, T, D)
                seed_trans = seed_trans + self.pos_embed[:,:seed_trans.shape[1], :]  # (N, T, D)
            elif self.pos_embed_type == "rope":
                pos_embed = self.pos_embed._prepare_latent_sequence_ids(
                        hand.shape[0], hand.shape[1], hand.device, self.dtype)
                pos_embed = self.pos_embed(pos_embed)
                pos_embed = tuple(pos_embed)
            elif self.pos_embed_type == "none":
                pass
            else:
                raise NotImplementedError(f"pos embed type {self.pos_embed_type} not implemented")
        
        trans = trans_prompt_mask.unsqueeze(-1) * trans + (~trans_prompt_mask.unsqueeze(-1)) * seed_trans
        for block in self.blocks:
            hand, body, trans = auto_grad_checkpoint(
                block, hand, body, trans, t0, pos_embed, seed_hand, pose_prompt_mask, seed_body, pose_prompt_mask
            )  # (N, T, D) #support grad checkpoint
        
        hand = self.hand_final_layer(hand, t)
        body = self.body_final_layer(body, t)  
        trans = self.trans_final_layer(trans, t)

        output_motion = torch.zeros_like(motion)
        output_motion[..., hand_indices] = hand
        output_motion[..., body_indices] = body
        output_motion[..., trans_indices] = trans
        
        return {
            "motion": output_motion, # 返回�?tensor
        }
    
    # def __call__(self, *args, **kwargs):
    #     """
    #     This method allows the object to be called like a function.
    #     It simply calls the forward method.
    #     """
    #     return self.forward(*args, **kwargs)

    def forward_with_dpmsolver(
        self,      
        motion,
        timestep,
        expression,
        cond = None,
        cond_mask = None,
        **kwargs
    ):
        """
        dpm solver donnot need variance prediction
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        model_out = self.forward(
            motion,
            timestep,
            expression,
            cond,
            cond_mask,
            **kwargs
        )
        return model_out

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if hasattr(module, "initialize_weights") and module is not self:
                module.initialize_weights()
                return
            if isinstance(module, nn.Linear) or isinstance(module, nn.Conv1d):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        self.apply(_basic_init)

        # # Initialize timestep embedding MLP:
        # nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        # nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        if self.simple_modulation:
            nn.init.constant_(self.t_block[1].weight, val=0)
            if self.t_block[1].bias is not None:
                nn.init.constant_(self.t_block[1].bias, val=0)
    
        # # # Initialize caption embedding MLP:
        # # nn.init.normal_(self.y_embedder.y_proj.fc1.weight, std=0.02)
        # # nn.init.normal_(self.y_embedder.y_proj.fc2.weight, std=0.02)

    @property
    def dtype(self):
        return next(self.parameters()).dtype


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


