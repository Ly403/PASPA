# 双分�?
# 实际上就是DiT
import os
import math
import numpy as np
import torch
import torch.nn as nn
from timm.layers import DropPath
from diffusion.model.nets.basic_modules import (
    DoubleStreamMLP,
    DoubleStreamGatedMLP   
)
from diffusion.model.nets.positional_embeding import RopePosEmbed
from diffusion.model.nets.basic_modules import t2i_modulate
from diffusion.model.nets.causal.causal_block import (
    MMCausalSelfAttention,
    CrossAttention,
    CrossAttentionMask,
)
from torch.nn.attention.flex_attention import BlockMask, create_block_mask, create_mask
from diffusion.model.norms import build_norm
from diffusion.model.utils import auto_grad_checkpoint
from diffusion.utils.dist_utils import get_rank
from diffusion.utils.logger import get_root_logger
from diffusion.utils.config import instantiate_from_config
from omegaconf import DictConfig
from copy import deepcopy
from diffusion.model.nets.causal.causal_dit import get_1d_sincos_pos_embed_from_grid

DEFAULT_CAPTION_EMBEDDER_CONFIG = DictConfig(dict(
    target="diffusion.model.nets.basic_modules.CaptionEmbedder",
    params=dict(
        in_channels=None,
        hidden_size=None,
        y_proj_type="mlp",
        act_type="gelu",
        y_norm_type="rms",
        bias=True,
        droput=0.0, 
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
    target="diffusion.model.nets.causal.causal_mmdit.CausalMMDITBlock",
    params=dict(
        hidden_size=None,
        num_heads=8,
        head_dim=None,
        simple_modulation=False,
        bias=True,
        norm_type="rms",
        attn_type="flash",
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
        is_teacher_forcing=False,
        local_attn_size=-1,
        padding_unit_length=128,
        max_attention_size=512,
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
    target="diffusion.model.nets.causal.causal_block.FinalLayer",
    params=dict(
        hidden_size=None, 
        out_channels=None,
        bias=True,
        norm_type="rms",
        proj_type="linear",
        norm_eps=1e-6,
    )
))


class StyleEmbedding(nn.Module):
    def __init__(self, num_persons, dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(num_persons, dim * 2),
            nn.SiLU(),
            nn.Linear(dim * 2, dim)
        )

    def forward(self, t):
        return self.mlp(t)

class CausalMMDITBlock(nn.Module):
    """
    A Causal MMDIT block with global shared adaptive layer norm (adaLN-single) conditioning.
    """

    def __init__(
        self,
        hidden_size,
        num_heads=8,
        head_dim=None,
        simple_modulation=False,
        bias=True,
        norm_type="rms",
        attn_type="flash",
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
        is_teacher_forcing=False,
        local_attn_size=-1,
        padding_unit_length=128,
        use_crossattn_mask=True,
        max_attention_size=512,
    ):
        super().__init__()
        self.hand_norm1 = build_norm(norm_type, num_features=hidden_size, affine=False, eps=norm_eps)
        self.body_norm1 = build_norm(norm_type, num_features=hidden_size, affine=False, eps=norm_eps)
        self.hand_norm2 = build_norm(norm_type, num_features=hidden_size, affine=False, eps=norm_eps)
        self.body_norm2 = build_norm(norm_type, num_features=hidden_size, affine=False, eps=norm_eps)
        self.hand_norm3 = build_norm(norm_type, num_features=hidden_size, affine=False, eps=norm_eps)
        self.body_norm3 = build_norm(norm_type, num_features=hidden_size, affine=False, eps=norm_eps)
        
        self.attn = MMCausalSelfAttention(
            dim=hidden_size,
            num_heads=num_heads,
            head_dim=head_dim,
            bias=bias,
            attn_type=attn_type,
            qk_norm_type=qk_norm_type,
            attn_drop=attn_drop, 
            proj_drop=attn_proj_drop,
            is_teacher_forcing=is_teacher_forcing,
            local_attn_size=local_attn_size,
            padding_unit_length=padding_unit_length,
            max_attention_size=max_attention_size,
        )
        
        if not use_crossattn_mask:
            self.hand_cross_attn = CrossAttention(
                dim=hidden_size,
                num_heads=num_heads,
                head_dim=head_dim,
                bias=bias,
                attn_type=cross_attn_type,
                qk_norm_type=cross_qk_norm_type,
                attn_drop=cross_attn_drop, 
                proj_drop=cross_attn_proj_drop,
            )
        else:
            self.hand_cross_attn = CrossAttentionMask(
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
                
        self.simple_modulation = simple_modulation
        if self.simple_modulation:
            # like pixart
            self.hand_scale_shift_table = nn.Parameter(torch.randn(hidden_size, 9))
            self.body_scale_shift_table = nn.Parameter(torch.randn(hidden_size, 9))
        else:
            self.hand_time_in = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 9 * hidden_size, bias=bias))   
            self.body_time_in = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 9 * hidden_size, bias=bias))   
        
        self.hand_drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.body_drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        
    def initialize_weights(self):
        if self.simple_modulation:
            nn.init.constant(self.hand_scale_shift_table, 0)
            nn.init.constant(self.body_scale_shift_table, 0)
        else:
            nn.init.constant_(self.hand_time_in[1].weight, val=0)
            if self.hand_time_in[1].bias is not None:
                nn.init.constant_(self.hand_time_in[1].bias, val=0)
                
            nn.init.constant_(self.body_time_in[1].weight, val=0)
            if self.body_time_in[1].bias is not None:
                nn.init.constant_(self.body_time_in[1].bias, val=0)
            

    def forward(
        self, 
        hand: torch.FloatTensor, 
        body: torch.FloatTensor,
        context: torch.FloatTensor,
        t: torch.FloatTensor,
        mask: torch.BoolTensor,
        crossattn_mask: torch.BoolTensor = None,
        rotary_emb: torch.FloatTensor = None,
        kv_cache=None,
        crossattn_cache=None,
        current_start=0,
        sampling_mode=False,
        **kwargs
    ):
        assert hand.shape[0] == body.shape[0], "batch size must be the same"
        B, N, C = hand.shape

        if self.simple_modulation:
            if len(t.shape) == 3:
                hand_shift_msa, hand_scale_msa, hand_gate_msa, \
                    hand_shift_cross, hand_scale_cross, hand_gate_cross, \
                    hand_shift_mlp, hand_scale_mlp, hand_gate_mlp = (
                    self.hand_scale_shift_table + t.reshape(B, t.shape[1], -1, 9)
                ).unbind(dim=-1)
                body_shift_msa, body_scale_msa, body_gate_msa, \
                    body_shift_cross, body_scale_cross, body_gate_cross, \
                    body_shift_mlp, body_scale_mlp, body_gate_mlp = (
                    self.body_scale_shift_table + t.reshape(B, t.shape[1], -1, 9)
                ).unbind(dim=-1)
            elif len(t.shape) == 2:
                hand_shift_msa, hand_scale_msa, hand_gate_msa, \
                    hand_shift_cross, hand_scale_cross, hand_gate_cross, \
                    hand_shift_mlp, hand_scale_mlp, hand_gate_mlp = (
                    self.hand_scale_shift_table + t.reshape(B, -1, 9)
                ).unbind(dim=-1)
                body_shift_msa, body_scale_msa, body_gate_msa, \
                    body_shift_cross, body_scale_cross, body_gate_cross, \
                    body_shift_mlp, body_scale_mlp, body_gate_mlp = (
                    self.body_scale_shift_table + t.reshape(B, -1, 9)
                ).unbind(dim=-1)
            else:
                raise ValueError(f"t.shape must be (B, T, D) or (B, D), but got {t.shape}")
        else:
            if len(t.shape) == 3:
                hand_shift_msa, hand_scale_msa, hand_gate_msa,\
                    hand_shift_cross, hand_scale_cross, hand_gate_cross, \
                    hand_shift_mlp, hand_scale_mlp, hand_gate_mlp = self.hand_time_in(t).chunk(9, dim=-1)
                body_shift_msa, body_scale_msa, body_gate_msa, \
                    body_shift_cross, body_scale_cross, body_gate_cross, \
                    body_shift_mlp, body_scale_mlp, body_gate_mlp = self.body_time_in(t).chunk(9, dim=-1)
            elif len(t.shape) == 2:
                hand_shift_msa, hand_scale_msa, hand_gate_msa, \
                    hand_shift_cross, hand_scale_cross, hand_gate_cross, \
                    hand_shift_mlp, hand_scale_mlp, hand_gate_mlp = self.hand_time_in(t.unsqueeze(1)).chunk(9, dim=-1)
                body_shift_msa, body_scale_msa, body_gate_msa, \
                    body_shift_cross, body_scale_cross, body_gate_cross, \
                    body_shift_mlp, body_scale_mlp, body_gate_mlp = self.body_time_in(t.unsqueeze(1)).chunk(9, dim=-1)
            else:
                raise ValueError(f"t.shape must be (B, T, D) or (B, D), but got {t.shape}")
        
        ##################################### The first layer #######################################33
        residual_hand = self.hand_norm1(hand)
        residual_body = self.body_norm1(body)
        
        residual_hand = t2i_modulate(residual_hand, hand_shift_msa, hand_scale_msa)
        residual_body = t2i_modulate(residual_body, body_shift_msa, body_scale_msa)

        residual_hand, residual_body = self.attn(
            residual_hand, residual_body, mask, rotary_emb, kv_cache, current_start
        )
    
        residual_hand = hand_gate_msa * residual_hand
        residual_body = body_gate_msa * residual_body
        
        hand = hand + self.hand_drop_path(residual_hand)
        body = body + self.body_drop_path(residual_body)
        
        ######################################## The forth layer ########################################
        residual_hand = self.hand_norm2(hand)
        residual_body = self.body_norm2(body)
        
        residual_hand = t2i_modulate(residual_hand, hand_shift_cross, hand_scale_cross)
        residual_body = t2i_modulate(residual_body, body_shift_cross, body_scale_cross)
        
        residual_hand = self.hand_cross_attn( # 对外虽有接口，暂不支持crossattn_cache
            residual_hand, context, crossattn_cache=None, mask=crossattn_mask, sampling_mode=sampling_mode)
        residual_body = self.body_cross_attn(
            residual_body, context, crossattn_cache=None, mask=crossattn_mask, sampling_mode=sampling_mode)

        residual_hand = hand_gate_cross * residual_hand
        residual_body = body_gate_cross * residual_body

        hand = hand + self.hand_drop_path(residual_hand)
        body = body + self.body_drop_path(residual_body)
        
        ######################################## The third layer ########################################
        residual_hand = self.hand_norm3(hand)
        residual_body = self.body_norm3(body)
        
        residual_hand = t2i_modulate(residual_hand, hand_shift_mlp, hand_scale_mlp)
        residual_body = t2i_modulate(residual_body, body_shift_mlp, body_scale_mlp)
        
        residual_hand, residual_body =\
            self.mlp(residual_hand, residual_body)
        
        residual_hand = hand_gate_mlp * residual_hand
        residual_body = body_gate_mlp * residual_body
        
        hand = hand + self.hand_drop_path(residual_hand)
        body = body + self.body_drop_path(residual_body)
        
        return hand, body

class CausalMMDITBlockWoCrossAttn(nn.Module):
    def __init__(
        self,
        hidden_size,
        num_heads=8,
        head_dim=None,
        simple_modulation=False,
        bias=True,
        norm_type="rms",
        attn_type="flash",
        qk_norm_type="identity",
        cross_attn_type="flash",
        cross_qk_norm_type="identity",
        ffn_type="mlp",
        ffn_act_type="gelu",
        ffn_proj_type="linear", 
        ffn_mlp_ratio=4.0,
        attn_drop=0.0,
        attn_proj_drop=0.0,
        ffn_drop=0.0,
        drop_path=0.0,
        norm_eps=1e-6,
        is_teacher_forcing=False,
        local_attn_size=-1,
        padding_unit_length=128,
        max_attention_size=512,
    ):
        super().__init__()
        self.hand_norm1 = build_norm(norm_type, num_features=hidden_size, affine=False, eps=norm_eps)
        self.body_norm1 = build_norm(norm_type, num_features=hidden_size, affine=False, eps=norm_eps)
        
        self.attn = MMCausalSelfAttention(
            dim=hidden_size,
            num_heads=num_heads,
            head_dim=head_dim,
            bias=bias,
            attn_type=attn_type,
            qk_norm_type=qk_norm_type,
            attn_drop=attn_drop, 
            proj_drop=attn_proj_drop,
            is_teacher_forcing=is_teacher_forcing,
            local_attn_size=local_attn_size,
            padding_unit_length=padding_unit_length,
            max_attention_size=max_attention_size,
        )
        
        
        self.hand_norm2 = build_norm(norm_type, num_features=hidden_size, affine=False, eps=norm_eps)
        self.body_norm2 = build_norm(norm_type, num_features=hidden_size, affine=False, eps=norm_eps)
        
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
                
        self.simple_modulation = simple_modulation
        if self.simple_modulation:
            # like pixart
            self.hand_scale_shift_table = nn.Parameter(torch.randn(hidden_size, 6))
            self.body_scale_shift_table = nn.Parameter(torch.randn(hidden_size, 6))
        else:
            self.hand_time_in = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=bias))   
            self.body_time_in = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=bias))   
        
        self.hand_drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.body_drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
    
    def initialize_weights(self):
        if self.simple_modulation:
            nn.init.constant(self.hand_scale_shift_table, 0)
            nn.init.constant(self.body_scale_shift_table, 0)
        else:
            nn.init.constant_(self.hand_time_in[1].weight, val=0)
            if self.hand_time_in[1].bias is not None:
                nn.init.constant_(self.hand_time_in[1].bias, val=0)
                
            nn.init.constant_(self.body_time_in[1].weight, val=0)
            if self.body_time_in[1].bias is not None:
                nn.init.constant_(self.body_time_in[1].bias, val=0)
                
    def forward(
        self, 
        hand: torch.FloatTensor, 
        body: torch.FloatTensor,
        t: torch.FloatTensor,
        mask: torch.BoolTensor,
        rotary_emb: torch.FloatTensor = None,
        kv_cache=None,
        current_start=0,
        **kwargs
    ):
        assert hand.shape[0] == body.shape[0], "batch size must be the same"
        B, N, C = hand.shape

        if self.simple_modulation:
            if len(t.shape) == 3:
                hand_shift_msa, hand_scale_msa, hand_gate_msa, hand_shift_mlp, hand_scale_mlp, hand_gate_mlp = (
                    self.hand_scale_shift_table + t.reshape(B, t.shape[1], -1, 6)
                ).unbind(dim=-1)
                body_shift_msa, body_scale_msa, body_gate_msa, body_shift_mlp, body_scale_mlp, body_gate_mlp = (
                    self.body_scale_shift_table + t.reshape(B, t.shape[1], -1, 6)
                ).unbind(dim=-1)
            elif len(t.shape) == 2:
                hand_shift_msa, hand_scale_msa, hand_gate_msa, hand_shift_mlp, hand_scale_mlp, hand_gate_mlp = (
                    self.hand_scale_shift_table + t.reshape(B, -1, 6)
                ).unbind(dim=-1)
                body_shift_msa, body_scale_msa, body_gate_msa, body_shift_mlp, body_scale_mlp, body_gate_mlp = (
                    self.body_scale_shift_table + t.reshape(B, -1, 6)
                ).unbind(dim=-1)
            else:
                raise ValueError(f"t.shape must be (B, T, D) or (B, D), but got {t.shape}")
        else:
            if len(t.shape) == 3:
                hand_shift_msa, hand_scale_msa, hand_gate_msa, hand_shift_mlp, hand_scale_mlp, hand_gate_mlp\
                    = self.hand_time_in(t).chunk(6, dim=-1)
                body_shift_msa, body_scale_msa, body_gate_msa, body_shift_mlp, body_scale_mlp, body_gate_mlp\
                    = self.body_time_in(t).chunk(6, dim=-1)
            elif len(t.shape) == 2:
                hand_shift_msa, hand_scale_msa, hand_gate_msa, hand_shift_mlp, hand_scale_mlp, hand_gate_mlp\
                    = self.hand_time_in(t.unsqueeze(1)).chunk(6, dim=-1)
                body_shift_msa, body_scale_msa, body_gate_msa, body_shift_mlp, body_scale_mlp, body_gate_mlp\
                    = self.body_time_in(t.unsqueeze(1)).chunk(6, dim=-1)
            else:
                raise ValueError(f"t.shape must be (B, T, D) or (B, D), but got {t.shape}")
        
        ##################################### The first layer #######################################33
        residual_hand = self.hand_norm1(hand)
        residual_body = self.body_norm1(body)
        
        residual_hand = t2i_modulate(residual_hand, hand_shift_msa, hand_scale_msa)
        residual_body = t2i_modulate(residual_body, body_shift_msa, body_scale_msa)

        residual_hand, residual_body = self.attn(
            residual_hand, residual_body, mask, rotary_emb, kv_cache, current_start
        )
    
        residual_hand = hand_gate_msa * residual_hand
        residual_body = body_gate_msa * residual_body
        
        hand = hand + self.hand_drop_path(residual_hand)
        body = body + self.body_drop_path(residual_body)
                
        ######################################## The third layer ########################################
        residual_hand = self.hand_norm2(hand)
        residual_body = self.body_norm2(body)
        
        residual_hand = t2i_modulate(residual_hand, hand_shift_mlp, hand_scale_mlp)
        residual_body = t2i_modulate(residual_body, body_shift_mlp, body_scale_mlp)
        
        residual_hand, residual_body =\
            self.mlp(residual_hand, residual_body)
        
        residual_hand = hand_gate_mlp * residual_hand
        residual_body = body_gate_mlp * residual_body
        
        hand = hand + self.hand_drop_path(residual_hand)
        body = body + self.body_drop_path(residual_body)
        
        return hand, body
    

class CausalMMDITEmage(nn.Module):
    def __init__(
        self,
        hand_in_channels=30 * 6,
        body_in_channels=(13 + 9 + 3) * 6,
        expression_in_channels=100,
        trans_in_channels=3,
        hidden_size=1152,
        caption_channels=2304,
        num_person=31,
        num_frame_per_block=30,
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
        simple_modulation=False, # for pixart-like only one modulation
        independent_first_frame=False,
        use_expression=True,
        use_crossattn_mask=True,
        **kwargs,
    ):
        super().__init__()
        self.num_person = num_person
        self.hand_in_channels = hand_in_channels
        self.body_in_channels = body_in_channels
        self.expression_in_channels = expression_in_channels
        self.trans_in_channels = trans_in_channels
        self.block_mask = None
        self.independent_first_frame = independent_first_frame
        self.use_expression = use_expression
        self.use_crossattn_mask = use_crossattn_mask
        if self.use_crossattn_mask:
            self.crossattn_mask = None
        self.local_attn_size = block_config["params"].get("local_attn_size", -1)
        self.num_frame_per_block = num_frame_per_block
        self.hidden_size = hidden_size
        self.depth = depth
        self.pos_embed_type = pos_embed_type

        if work_dir is not None:
            logger = get_root_logger(os.path.join(work_dir, "train_log.log"))
            logger = logger.info
        else:
            logger = print
        
        self.logger = logger
        inputlayer_config["params"]["hidden_size"] = hidden_size
        
        inputlayer_config["params"]["in_channels"] = hand_in_channels
        self.hand_in = instantiate_from_config(inputlayer_config)
        inputlayer_config["params"]["in_channels"] = body_in_channels + trans_in_channels
        self.body_in = instantiate_from_config(inputlayer_config)
        
        if self.use_expression:
            inputlayer_config["params"]["in_channels"] = expression_in_channels
            self.expression_in = instantiate_from_config(inputlayer_config)
   
        timestep_embedder_config["params"]["hidden_size"] = hidden_size
        self.motion_t_embedder = instantiate_from_config(timestep_embedder_config)
        self.simple_modulation = simple_modulation
        
        self.id_embedder = nn.Embedding(num_person, hidden_size)
        
        self.use_pe = use_pe
        self.pe_max_length = pe_max_length
        # Will use fixed sin-cos embedding:
        self.register_buffer("pos_embed", torch.zeros(1, pe_max_length, hidden_size))
        if self.simple_modulation:
            self.t_block = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True))
        
        caption_embedder_config["params"]["in_channels"] = caption_channels
        caption_embedder_config["params"]["hidden_size"] = hidden_size
        self.context_embedder = instantiate_from_config(caption_embedder_config)
        
        drop_path = [x.item() for x in torch.linspace(0, drop_path, depth)]  # stochastic depth decay rule
        block_config["params"]["simple_modulation"] = simple_modulation
        block_config["params"]["hidden_size"] = hidden_size
        self.blocks = nn.ModuleList()
        for i in range(depth):
            block_config["params"]["drop_path"] = drop_path[i]
            self.blocks.append(instantiate_from_config(block_config))
        
        finallayer_config["params"]["hidden_size"] = hidden_size
        finallayer_config["params"]["out_channels"] = hand_in_channels
        self.hand_final_layer = instantiate_from_config(finallayer_config)
        finallayer_config["params"]["out_channels"] = body_in_channels + trans_in_channels
        self.body_final_layer = instantiate_from_config(finallayer_config)
        

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
            )
    
    @staticmethod
    def _prepare_blockwise_causal_attn_mask(
        device: torch.device | str, num_frames: int = 21,
        frame_seqlen: int = 1560, num_frame_per_block=1, local_attn_size=-1,
        unit_length: int = 128, debug = False,
    ) -> BlockMask:
        """
        we will divide the token sequence into the following format
        [1 latent frame] [1 latent frame] ... [1 latent frame]
        We use flexattention to construct the attention mask
        """
        total_length = num_frames * frame_seqlen * 2

        # we do right padding to get to a multiple of 128
        padded_length = math.ceil(total_length / unit_length) * unit_length - total_length

        starts1 = torch.zeros(total_length + padded_length,
                           device=device, dtype=torch.long)
        ends1 = torch.zeros(total_length + padded_length,
                           device=device, dtype=torch.long)
        starts2 = torch.zeros(total_length + padded_length,
                           device=device, dtype=torch.long)
        ends2 = torch.zeros(total_length + padded_length,
                           device=device, dtype=torch.long)

        # Block-wise causal mask will attend to all elements that are before the end of the current chunk
        frame_indices = torch.arange(
            start=0,
            end=num_frames * frame_seqlen,
            step=frame_seqlen * num_frame_per_block,
            device=device
        )

        for tmp in frame_indices:
            starts1[tmp:tmp + frame_seqlen * num_frame_per_block] = 0
            ends1[tmp:tmp + frame_seqlen * num_frame_per_block] = tmp + \
                frame_seqlen * num_frame_per_block

            starts2[tmp: tmp + frame_seqlen * num_frame_per_block] = num_frames * frame_seqlen
            ends2[tmp: tmp + frame_seqlen * num_frame_per_block] = tmp + \
                frame_seqlen * num_frame_per_block + num_frames * frame_seqlen
    
        frame_indices = torch.arange(
            start=num_frames * frame_seqlen,
            end=num_frames * frame_seqlen * 2,
            step=frame_seqlen * num_frame_per_block,
            device=device
        )

        for tmp in frame_indices:
            starts1[tmp:tmp + frame_seqlen * num_frame_per_block] = 0
            ends1[tmp:tmp + frame_seqlen * num_frame_per_block] = tmp + \
                frame_seqlen * num_frame_per_block - num_frames * frame_seqlen

            starts2[tmp: tmp + frame_seqlen * num_frame_per_block] = num_frames * frame_seqlen
            ends2[tmp: tmp + frame_seqlen * num_frame_per_block] = tmp + \
                frame_seqlen * num_frame_per_block

        def attention_mask(b, h, q_idx, kv_idx):
            if local_attn_size == -1:
                C1 = (kv_idx < ends1[q_idx]) & (kv_idx >= starts1[q_idx])
                C2 = (kv_idx < ends2[q_idx]) & (kv_idx >= starts2[q_idx])
                return C1 | C2 | (q_idx == kv_idx)
            else:
                C1 = ((kv_idx < ends1[q_idx]) & (kv_idx >= starts1[q_idx]) & (kv_idx >= (ends1[q_idx] - local_attn_size * frame_seqlen)))
                C2 = ((kv_idx < ends2[q_idx]) & (kv_idx >= starts2[q_idx]) & (kv_idx >= (ends2[q_idx] - local_attn_size * frame_seqlen)))
                return C1 | C2 | (q_idx == kv_idx)
            # return ((kv_idx < total_length) & (q_idx < total_length))  | (q_idx == kv_idx) # bidirectional mask


        if not debug:
            block_mask = create_block_mask(attention_mask, B=None, H=None, Q_LEN=total_length + padded_length,
                                        KV_LEN=total_length + padded_length, _compile=False, device=device)
        else:
            block_mask = create_mask(attention_mask, B=None, H=None, Q_LEN=total_length + padded_length,
                                        KV_LEN=total_length + padded_length, device=device)

        return block_mask

    @staticmethod
    def _prepare_teacher_forcing_mask(
        device: torch.device | str, num_frames: int = 21,
        frame_seqlen: int = 1560, num_frame_per_block=1,
        unit_length: int = 128, debug=False,
    ) -> BlockMask:
        """
        we will divide the token sequence into the following format
        [1 latent frame] [1 latent frame] ... [1 latent frame]
        We use flexattention to construct the attention mask
        """
        total_length = num_frames * frame_seqlen * 2 * 2

        # we do right padding to get to a multiple of 128
        padded_length = math.ceil(total_length / unit_length) * unit_length - total_length

        clean1_starts = 0
        clean1_ends = num_frames * frame_seqlen
        
        clean2_starts = num_frames * frame_seqlen * 2
        clean2_ends = num_frames * frame_seqlen * 3
        
        # for clean context frames, we can construct their flex attention mask based on a [start, end] interval
        context_starts1 = 0 
        context_starts2 = num_frames * frame_seqlen * 2
        context_ends1 = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        context_ends2 = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)

        # for noisy frames, we need two intervals to construct the flex attention mask [context_start, context_end] [noisy_start, noisy_end]
        noise_context_starts1 = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        noise_context_ends1 = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        noise_noise_starts1 = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        noise_noise_ends1 = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        
        noise_context_starts2 = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        noise_context_ends2 = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        noise_noise_starts2 = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        noise_noise_ends2 = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)


        # Block-wise causal mask will attend to all elements that are before the end of the current chunk
        attention_block_size = frame_seqlen * num_frame_per_block
        frame_indices = torch.arange(
            start=0,
            end=num_frames * frame_seqlen,
            step=attention_block_size,
            device=device, dtype=torch.long
        )

        # attention for clean context frames
        for start in frame_indices:
            context_ends1[start:start + attention_block_size] = start + attention_block_size
            context_ends2[start:start + attention_block_size] = start + attention_block_size + num_frames * frame_seqlen * 2
        
        frame_indices = torch.arange(
            start=num_frames * frame_seqlen * 2,
            end=num_frames * frame_seqlen * 3,
            step=attention_block_size,
            device=device, dtype=torch.long
        )
        
        for start in frame_indices:
            context_ends2[start:start + attention_block_size] = start + attention_block_size
            context_ends1[start:start + attention_block_size] = start + attention_block_size - num_frames * frame_seqlen * 2

        noisy_image_start_list = torch.arange(
            start=num_frames * frame_seqlen, 
            end=num_frames * frame_seqlen * 2,
            step=attention_block_size,
            device=device, dtype=torch.long
        )
        noisy_image_end_list = noisy_image_start_list + attention_block_size

        # attention for noisy frames
        for block_index, (start, end) in enumerate(zip(noisy_image_start_list, noisy_image_end_list)):
            # attend to noisy tokens within the same block
            noise_noise_starts1[start:end] = start
            noise_noise_ends1[start:end] = end
            # attend to context tokens in previous blocks
            # noise_context_starts[start:end] = 0
            noise_context_ends1[start:end] = block_index * attention_block_size
            
            noise_noise_starts2[start:end] = start + num_frames * frame_seqlen * 2
            noise_noise_ends2[start:end] = end + num_frames * frame_seqlen * 2
            noise_context_starts2[start:end] = num_frames * frame_seqlen * 2
            noise_context_ends2[start:end] = block_index * attention_block_size + num_frames * frame_seqlen * 2

        noisy_image_start_list = torch.arange(
            start=num_frames * frame_seqlen * 3,
            end=num_frames * frame_seqlen * 4,
            step=attention_block_size,
            device=device, dtype=torch.long
        )
        noisy_image_end_list = noisy_image_start_list + attention_block_size
        
        
        # attention for noisy frames
        for block_index, (start, end) in enumerate(zip(noisy_image_start_list, noisy_image_end_list)):
            # attend to noisy tokens within the same block
            noise_noise_starts2[start:end] = start
            noise_noise_ends2[start:end] = end
            # attend to context tokens in previous blocks
            noise_context_ends2[start:end] = block_index * attention_block_size + num_frames * frame_seqlen * 2

            noise_noise_starts1[start:end] = start - num_frames * frame_seqlen * 2
            noise_noise_ends1[start:end] = end - num_frames * frame_seqlen * 2
            noise_context_starts2[start:end] = num_frames * frame_seqlen * 2
            noise_context_ends1[start:end] = block_index * attention_block_size

        def attention_mask(b, h, q_idx, kv_idx):
            # first design the mask for clean frames
            C1 = (q_idx < clean1_ends) & (q_idx >= clean1_starts)
            C2 = (q_idx < clean2_ends) & (q_idx >= clean2_starts)
            C3 = (kv_idx < context_ends1[q_idx]) & (kv_idx >= context_starts1)
            C4 = (kv_idx < context_ends2[q_idx]) & (kv_idx >= context_starts2)
            clean1_mask = (C1 | C2) & (C3 | C4)
            clean2_mask = (C1 | C2) & (C3 | C4)
            # then design the mask for noisy frames
            # noisy frames will attend to all clean preceeding clean frames + itself
            C1 = (kv_idx < noise_noise_ends1[q_idx]) & (kv_idx >= noise_noise_starts1[q_idx])
            C2 = (kv_idx < noise_context_ends1[q_idx]) & (kv_idx >= noise_context_starts1[q_idx])
            
            C3 = (kv_idx < noise_noise_ends2[q_idx]) & (kv_idx >= noise_noise_starts2[q_idx])
            C4 = (kv_idx < noise_context_ends2[q_idx]) & (kv_idx >= noise_context_starts2[q_idx])
            
            noise1_mask = (q_idx >= clean1_ends) & (q_idx < clean2_starts) & (C1 | C2 | C3 | C4)
            noise2_mask = (q_idx >= clean2_ends) & (C1 | C2 | C3 | C4)

            eye_mask = q_idx == kv_idx
            return clean1_mask | clean2_mask | noise1_mask | noise2_mask | eye_mask

        if not debug:
            block_mask = create_block_mask(attention_mask, B=None, H=None, Q_LEN=total_length + padded_length,
                                        KV_LEN=total_length + padded_length, _compile=False, device=device)
        else:
            block_mask = create_mask(attention_mask, B=None, H=None, Q_LEN=total_length + padded_length,
                                        KV_LEN=total_length + padded_length, device=device)

        return block_mask

    @staticmethod
    def _prepare_blockwise_causal_attn_mask_i2v(
        device: torch.device | str, num_frames: int = 21,
        frame_seqlen: int = 1560, num_frame_per_block=4, local_attn_size=-1,
        unit_length: int = 128, debug=False,
    ) -> BlockMask:
        """
        we will divide the token sequence into the following format
        [1 latent frame] [N latent frame] ... [N latent frame]
        The first frame is separated out to support I2V generation
        We use flexattention to construct the attention mask
        """
        total_length = num_frames * frame_seqlen * 2

        # we do right padding to get to a multiple of 128
        padded_length = math.ceil(total_length / unit_length) * unit_length - total_length

        starts1 = torch.zeros(total_length + padded_length,
                           device=device, dtype=torch.long)
        ends1 = torch.zeros(total_length + padded_length,
                           device=device, dtype=torch.long)
        starts2 = torch.zeros(total_length + padded_length,
                           device=device, dtype=torch.long)
        ends2 = torch.zeros(total_length + padded_length,
                           device=device, dtype=torch.long)

        # special handling for the first frame
        starts1[:frame_seqlen] = 0
        ends1[:frame_seqlen] = frame_seqlen
        starts2[:frame_seqlen + num_frames * frame_seqlen] = num_frames * frame_seqlen
        ends2[:frame_seqlen + num_frames * frame_seqlen] = frame_seqlen + num_frames * frame_seqlen

        # Block-wise causal mask will attend to all elements that are before the end of the current chunk
        frame_indices = torch.arange(
            start=frame_seqlen,
            end=num_frames * frame_seqlen,
            step=frame_seqlen * num_frame_per_block,
            device=device
        )

        for idx, tmp in enumerate(frame_indices):
            starts1[tmp:tmp + frame_seqlen * num_frame_per_block] = 0
            starts2[tmp:tmp + frame_seqlen * num_frame_per_block] = num_frames * frame_seqlen
            ends1[tmp:tmp + frame_seqlen * num_frame_per_block] = tmp + \
                frame_seqlen * num_frame_per_block
            ends2[tmp:tmp + frame_seqlen * num_frame_per_block] = tmp + \
                frame_seqlen * num_frame_per_block + num_frames * frame_seqlen
        
        
        # special handling for the first frame
        starts1[num_frames * frame_seqlen:num_frames * frame_seqlen + frame_seqlen] = 0
        ends1[num_frames * frame_seqlen:num_frames * frame_seqlen + frame_seqlen] = frame_seqlen
        starts2[num_frames * frame_seqlen:num_frames * frame_seqlen + frame_seqlen] = num_frames * frame_seqlen
        ends2[num_frames * frame_seqlen:num_frames * frame_seqlen + frame_seqlen] = frame_seqlen + num_frames * frame_seqlen
        # Block-wise causal mask will attend to all elements that are before the end of the current chunk
        frame_indices = torch.arange(
            start=frame_seqlen + num_frames * frame_seqlen,
            end=num_frames * frame_seqlen * 2,
            step=frame_seqlen * num_frame_per_block,
            device=device
        )

        for idx, tmp in enumerate(frame_indices):
            starts1[tmp:tmp + frame_seqlen * num_frame_per_block] = 0
            starts2[tmp:tmp + frame_seqlen * num_frame_per_block] = num_frames * frame_seqlen
            ends1[tmp:tmp + frame_seqlen * num_frame_per_block] = tmp + \
                frame_seqlen * num_frame_per_block - num_frames * frame_seqlen
            ends2[tmp:tmp + frame_seqlen * num_frame_per_block] = tmp + \
                frame_seqlen * num_frame_per_block 

        def attention_mask(b, h, q_idx, kv_idx):
            if local_attn_size == -1:
                C1 = (kv_idx < ends1[q_idx]) & (kv_idx >= starts1[q_idx])
                C2 = (kv_idx < ends2[q_idx]) & (kv_idx >= starts2[q_idx])
                return C1 | C2 | (q_idx == kv_idx)
            else:
                C1 = (kv_idx < ends1[q_idx]) & (kv_idx >= starts1[q_idx]) & (kv_idx >= (ends1[q_idx] - local_attn_size * frame_seqlen))
                C2 = (kv_idx < ends2[q_idx]) & (kv_idx >= starts2[q_idx]) & (kv_idx >= (ends2[q_idx] - local_attn_size * frame_seqlen))
                return C1 | C2 | (q_idx == kv_idx)

        if not debug:
            block_mask = create_block_mask(attention_mask, B=None, H=None, Q_LEN=total_length + padded_length,
                                        KV_LEN=total_length + padded_length, _compile=False, device=device)
        else:
            block_mask = create_mask(attention_mask, B=None, H=None, Q_LEN=total_length + padded_length,
                                        KV_LEN=total_length + padded_length, device=device)


        return block_mask
    
    
    @staticmethod
    def _prepare_blockwise_cross_attn_mask(
        device: torch.device | str, num_frames: int = 21,
        frame_seqlen: int = 1560, num_frame_per_block=1, local_attn_size=-1,
        context_length: int = 1560, 
        unit_length: int = 128, debug = False,
    ) -> BlockMask:
        """
        we will divide the token sequence into the following format
        [1 latent frame] [1 latent frame] ... [1 latent frame]
        We use flexattention to construct the attention mask
        """
        # 不大敢保证context_length != total_length时候的准确性，其实基本遇不到这种情�?
        total_length = num_frames * frame_seqlen * 2
        context_length *= 2

        coeff = context_length / total_length
        # we do right padding to get to a multiple of 128
        padded_length = math.ceil(total_length / unit_length) * unit_length - total_length
        padded_context_length = math.ceil(context_length / unit_length) * unit_length - context_length

        context_slice_length = math.ceil(frame_seqlen * num_frame_per_block * coeff)

        context_starts1 = torch.zeros(context_length + padded_context_length,
                           device=device, dtype=torch.long)
        context_ends1 = torch.zeros(context_length + padded_context_length,
                           device=device, dtype=torch.long)
        context_starts2 = torch.zeros(context_length + padded_context_length,
                           device=device, dtype=torch.long)
        context_ends2 = torch.zeros(context_length + padded_context_length,
                           device=device, dtype=torch.long)

        frame_indices = torch.arange(
            start=0,
            end=frame_seqlen * num_frames,
            step=frame_seqlen * num_frame_per_block,
            device=device
        )

        for tmp in frame_indices:
            context_starts1[tmp:tmp + frame_seqlen * num_frame_per_block] = math.ceil(tmp * coeff)
            context_ends1[tmp:tmp + frame_seqlen * num_frame_per_block] = math.ceil(tmp * coeff) + context_slice_length
            context_starts2[tmp:tmp + frame_seqlen * num_frame_per_block] = math.ceil(tmp * coeff \
                + frame_seqlen * num_frames * coeff) 
            context_ends2[tmp:tmp + frame_seqlen * num_frame_per_block] = math.ceil(tmp * coeff \
                + frame_seqlen * num_frames * coeff)  + context_slice_length
        
        
        frame_indices = torch.arange(
            start=frame_seqlen * num_frames,
            end=total_length,
            step=frame_seqlen * num_frame_per_block,
            device=device
        )

        for tmp in frame_indices:
            context_starts1[tmp:tmp + frame_seqlen * num_frame_per_block] = math.ceil(tmp * coeff \
                - frame_seqlen * num_frames * coeff) 
            context_ends1[tmp:tmp + frame_seqlen * num_frame_per_block] = math.ceil(tmp * coeff \
                - frame_seqlen * num_frames * coeff) + context_slice_length
            context_starts2[tmp:tmp + frame_seqlen * num_frame_per_block] = math.ceil(tmp * coeff) 
            context_ends2[tmp:tmp + frame_seqlen * num_frame_per_block] = math.ceil(tmp * coeff)  + context_slice_length

        def attention_mask(b, h, q_idx, kv_idx):
            C0 = q_idx * coeff == kv_idx
            if local_attn_size == -1:
                C1 = (kv_idx < context_ends1[q_idx]) & (kv_idx >= context_starts1[q_idx])
                C2 = (kv_idx < context_ends2[q_idx]) & (kv_idx >= context_starts2[q_idx])   
                return C1 | C2 | C0
            else:
                C1 = (kv_idx < context_ends1[q_idx]) & (kv_idx >= context_starts1[q_idx]) \
                    & (kv_idx >= (context_ends1[q_idx] - math.ceil(local_attn_size * frame_seqlen * coeff)))
                C2 = (kv_idx < context_ends2[q_idx]) & (kv_idx >= context_starts2[q_idx]) \
                    & (kv_idx >= (context_ends2[q_idx] - math.ceil(local_attn_size * frame_seqlen * coeff)))
                return C1 | C2 | C0

        if not debug:
            block_mask = create_block_mask(attention_mask, B=None, H=None, Q_LEN=total_length + padded_length,
                                        KV_LEN=context_length + padded_context_length, _compile=False, device=device)
        else:
            block_mask = create_mask(attention_mask, B=None, H=None, Q_LEN=total_length + padded_length,
                                        KV_LEN=context_length + padded_context_length, device=device)

        return block_mask
    
    def forward_train(
        self,  
        motion,
        context,
        timestep,
        expression=None,
        clean_motion=None,
        aug_timestep=None,
        **kwargs
    ):
        """
        Forward pass of Sana.
            motion: (N, T, D) tensor of motion
            expression: (N, T, D) tensor of expression
            timestep: (N, T) tensor of timestep
            cond: (N, L, D) tensor of condition
            cond_mask: (N, L) tensor of condition mask
        """        
        motion = motion.to(self.dtype)
        timestep = timestep.to(self.dtype)
        pos_embed = self.pos_embed.to(self.dtype)
        
        device = motion.device
        # Construct blockwise causal attn mask
        if self.block_mask is None:
            if clean_motion is not None:
                if self.independent_first_frame:
                    raise NotImplementedError()
                else:
                    self.block_mask = self._prepare_teacher_forcing_mask(
                        device, num_frames=motion.shape[1],
                        frame_seqlen=1,
                        num_frame_per_block=self.num_frame_per_block
                    )
            else:
                if self.independent_first_frame:
                    self.block_mask = self._prepare_blockwise_causal_attn_mask_i2v(
                        device, num_frames=motion.shape[1],
                        frame_seqlen=1,
                        num_frame_per_block=self.num_frame_per_block,
                        local_attn_size=self.local_attn_size
                    )
                else:
                    self.block_mask = self._prepare_blockwise_causal_attn_mask(
                        device, num_frames=motion.shape[1],
                        frame_seqlen=1,
                        num_frame_per_block=self.num_frame_per_block,
                        local_attn_size=self.local_attn_size
                    )

        if self.use_crossattn_mask and context is not None and self.crossattn_mask is None:
            self.crossattn_mask = self._prepare_blockwise_cross_attn_mask(
                device, num_frames=motion.shape[1],
                frame_seqlen=1,
                num_frame_per_block=self.num_frame_per_block,
                context_length=context.shape[1],
                local_attn_size=self.local_attn_size
            )

        id = kwargs["speaker_id"]
        hand_indices = kwargs["hand_indices"]
        upper_indices = kwargs["upper_indices"]
        lower_indices = kwargs["lower_indices"]
        face_joint_indices = kwargs["face_joint_indices"]
        trans_indices = kwargs['trans_indices']
         
        body_indices = np.concatenate([upper_indices, lower_indices, trans_indices, face_joint_indices], axis=0)
        #########################################################
        # for expression
        #########################################################
        
        hand = motion[..., hand_indices]
        body = motion[..., body_indices]
        
        hand = self.hand_in(hand)
        body = self.body_in(body)
    
        context = self.context_embedder(context)  # (N, L, D)
        if self.use_expression and expression is not None:
            expression = self.expression_in(expression)  # (N, L, D)
            context = torch.cat([context, expression], dim=1) 
        
        t = self.motion_t_embedder(timestep)  # (N, T, D)
        id_embed = self.id_embedder(id)
        if len(t.shape) == 2:
            t = t.unsqueeze(1)
        t = t + id_embed.unsqueeze(1)
        if self.simple_modulation:
            t0 = self.t_block(t)
        else:
            t0 = t
        
        if clean_motion is not None:
            clean_hand = clean_motion[..., hand_indices]
            clean_body = clean_motion[..., body_indices]
            
            clean_hand = self.hand_in(clean_hand)
            clean_body = self.body_in(clean_body)

            hand = torch.cat([clean_hand, hand], dim=1)
            body = torch.cat([clean_body, body], dim=1)
            if aug_timestep is None:
                aug_timestep = torch.zeros_like(timestep)
            t_clean = self.motion_t_embedder(aug_timestep)
            t_clean = t_clean + id_embed.unsqueeze(1)
            if self.simple_modulation:
                clean_t0 = self.t_block(t_clean)
            else:
                clean_t0 = t_clean
            t0 = torch.cat([clean_t0, t0], dim=1)
            context = torch.cat([context, context], dim=1)
        
        pos_embed = None
        if self.use_pe:
            if self.pos_embed_type == "sincos":
                hand = hand + self.pos_embed[:,:hand.shape[1], :]  # (N, T, D)
                body = body + self.pos_embed[:,:body.shape[1], :]  # (N, T, D)
                if clean_motion is not None:
                    clean_hand = clean_hand + self.pos_embed[:,:clean_hand.shape[1], :]  # (N, T, D)
                    clean_body = clean_body + self.pos_embed[:,:clean_body.shape[1], :]  # (N, T, D)
            elif self.pos_embed_type == "rope":
                if clean_motion is not None:
                    seq_len_clean = hand.shape[1] // 2 
                    
                    # 1. 生成单份长度 (T) �?Pos Embedding
                    rope_emb = self.pos_embed._prepare_latent_sequence_ids(
                        hand.shape[0], seq_len_clean, hand.device, self.dtype)
                    rope_emb = self.pos_embed(rope_emb) # (1, T, D)
                    
                    # 2. 拼接两份相同�?Embedding -> (1, 2T, D)
                    # 前半部分�?Clean，后半部分给 Noisy，它们的 Pos ID 都是 0 ~ T-1
                    pos_embed  = []
                    for emb in rope_emb:
                        pos_embed.append(torch.cat([emb, emb], dim=0))
                else:
                    pos_embed = self.pos_embed._prepare_latent_sequence_ids(
                            hand.shape[0], hand.shape[1], hand.device, self.dtype)
                    pos_embed = self.pos_embed(pos_embed)
            elif self.pos_embed_type == "none":
                pass
            else:
                raise NotImplementedError(f"pos embed type {self.pos_embed_type} not implemented")
        
        kwargs = dict(
            context = context,
            t = t0,
            mask = self.block_mask,
            crossattn_mask = self.crossattn_mask,
            rotary_emb = pos_embed,
            kv_cache = None,
            crossattn_cache = None,
            current_start = 0,
            sampling_mode=False,
        )
        
        for block in self.blocks:
            if torch.is_grad_enabled():
                hand, body = auto_grad_checkpoint(block, hand=hand, body=body, **kwargs)
            else:
                hand, body = block(hand=hand, body=body, **kwargs)
        
        if clean_motion is not None:
            hand = hand[:,hand.shape[1] // 2:]
            body = body[:,body.shape[1] // 2:]
            t0 = t0[:,t0.shape[1] // 2:]

        
        hand = self.hand_final_layer(hand, t)
        body = self.body_final_layer(body, t)
        
        output_motion = torch.zeros_like(motion)
        output_motion[..., hand_indices] = hand
        output_motion[..., body_indices] = body
        return {
            "motion": output_motion, 
        }
    
    def forward_inference(
        self,  
        motion,
        context,
        timestep,
        expression=None,
        kv_cache: dict = None,
        crossattn_cache: dict = None,
        seq_len: int = None,
        current_start: int = 0,
        **kwargs
    ):
        """
        Forward pass of Sana.
            motion: (N, T, D) tensor of motion
            expression: (N, T, D) tensor of expression
            timestep: (N, T) tensor of timestep
            cond: (N, L, D) tensor of condition
            cond_mask: (N, L) tensor of condition mask
        """        
        motion = motion.to(self.dtype)
        timestep = timestep.to(self.dtype)
        pos_embed = self.pos_embed.to(self.dtype)

        id = kwargs["speaker_id"]
        hand_indices = kwargs["hand_indices"]
        upper_indices = kwargs["upper_indices"]
        lower_indices = kwargs["lower_indices"]
        face_joint_indices = kwargs["face_joint_indices"]
        trans_indices = kwargs['trans_indices']
         
        body_indices = np.concatenate([upper_indices, lower_indices, trans_indices, face_joint_indices], axis=0)
        #########################################################
        # for expression
        #########################################################
        
        hand = motion[..., hand_indices]
        body = motion[..., body_indices]
        
        hand = self.hand_in(hand)
        body = self.body_in(body)
      
        context = self.context_embedder(context)  # (N, L, D)
        if self.use_expression and expression is not None:
            expression = self.expression_in(expression)  # (N, L, D)
            context = torch.cat([context, expression], dim=1)
        t = self.motion_t_embedder(timestep)  # (N, T, D)
        id_embed = self.id_embedder(id)
        if len(t.shape) == 2:
            t = t.unsqueeze(1)
        t = t + id_embed.unsqueeze(1)
        if self.simple_modulation:
            t0 = self.t_block(t)
        else:
            t0 = t
        
        pos_embed = None
        if self.use_pe:
            if self.pos_embed_type == "sincos":
                hand = hand + self.pos_embed[:,:hand.shape[1], :]  # (N, T, D)
                body = body + self.pos_embed[:,:body.shape[1], :]  # (N, T, D)
            elif self.pos_embed_type == "rope":
                seq_len = seq_len if seq_len is not None else hand.shape[1]
                pos_embed = self.pos_embed._prepare_latent_sequence_ids(
                        hand.shape[0], seq_len, hand.device, self.dtype)
                pos_embed = self.pos_embed(pos_embed)
            elif self.pos_embed_type == "none":
                pass
            else:
                raise NotImplementedError(f"pos embed type {self.pos_embed_type} not implemented")
        
        kwargs = dict(
            context = context,
            t = t0,
            mask = None,
            crossattn_mask = None,
            rotary_emb = pos_embed,
            sampling_mode=True,
        )
        
        for block_index, block in enumerate(self.blocks):
            kwargs.update(
                {
                    "kv_cache": kv_cache[block_index],
                    "crossattn_cache": crossattn_cache[block_index] if crossattn_cache is not None else None,
                    "current_start": current_start,
                }
            )
            if torch.is_grad_enabled():
                hand, body = auto_grad_checkpoint(block, hand=hand, body=body, **kwargs)
            else:
                hand, body = block(hand=hand, body=body, **kwargs)

        
        hand = self.hand_final_layer(hand, t)
        body = self.body_final_layer(body, t)
        
        output_motion = torch.zeros_like(motion)
        output_motion[..., hand_indices] = hand
        output_motion[..., body_indices] = body
        return {
            "motion": output_motion, 
        }

    def forward(
        self,
        *args,
        **kwargs
    ):
        if kwargs.get('kv_cache', None) is not None:
            return self.forward_inference(*args, **kwargs)
        else:
            return self.forward_train(*args, **kwargs)

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
    
        # # Initialize caption embedding MLP:
        # nn.init.normal_(self.y_embedder.y_proj.fc1.weight, std=0.02)
        # nn.init.normal_(self.y_embedder.y_proj.fc2.weight, std=0.02)

    @property
    def dtype(self):
        return next(self.parameters()).dtype


class CausalMMDiTEmageWoCrossAttn(CausalMMDITEmage):
    def __init__(self, inputlayer_config=DEFAULT_INPUT_CONFIG, **kwargs):
        super().__init__(**kwargs)
        if self.use_expression:
            del self.expression_in
        inputlayer_config["params"]["hidden_size"] = self.hidden_size
        if self.use_expression:
            inputlayer_config["params"]["in_channels"] = 55 * 6 + 100 + 3
            self.xhand_in = instantiate_from_config(inputlayer_config)
            inputlayer_config["params"]["in_channels"] = 55 * 6 + 100 + 3
            self.xbody_in = instantiate_from_config(inputlayer_config)
                 
            inputlayer_config["params"]["in_channels"] = self.hidden_size * 3
            self.hand_fuse = instantiate_from_config(inputlayer_config)
            inputlayer_config["params"]["in_channels"] = self.hidden_size * 3
            self.body_fuse = instantiate_from_config(inputlayer_config)
        else:
            inputlayer_config["params"]["in_channels"] = self.hidden_size * 2
            self.hand_fuse = instantiate_from_config(inputlayer_config)
            inputlayer_config["params"]["in_channels"] = self.hidden_size * 2
            self.body_fuse = instantiate_from_config(inputlayer_config)

    
    def forward_train(
        self,  
        motion,
        context,
        timestep,
        expression=None,
        clean_motion=None,
        aug_timestep=None,
        **kwargs
    ):
        """
        Forward pass of Sana.
            motion: (N, T, D) tensor of motion
            expression: (N, T, D) tensor of expression
            timestep: (N, T) tensor of timestep
            cond: (N, L, D) tensor of condition
            cond_mask: (N, L) tensor of condition mask
        """        
        motion = motion.to(self.dtype)
        timestep = timestep.to(self.dtype)
        pos_embed = self.pos_embed.to(self.dtype)
        
        device = motion.device
        # Construct blockwise causal attn mask
        if self.block_mask is None:
            if clean_motion is not None:
                if self.independent_first_frame:
                    raise NotImplementedError()
                else:
                    self.block_mask = self._prepare_teacher_forcing_mask(
                        device, num_frames=motion.shape[1],
                        frame_seqlen=1,
                        num_frame_per_block=self.num_frame_per_block
                    )
            else:
                if self.independent_first_frame:
                    self.block_mask = self._prepare_blockwise_causal_attn_mask_i2v(
                        device, num_frames=motion.shape[1],
                        frame_seqlen=1,
                        num_frame_per_block=self.num_frame_per_block,
                        local_attn_size=self.local_attn_size
                    )
                else:
                    self.block_mask = self._prepare_blockwise_causal_attn_mask(
                        device, num_frames=motion.shape[1],
                        frame_seqlen=1,
                        num_frame_per_block=self.num_frame_per_block,
                        local_attn_size=self.local_attn_size
                    )

        id = kwargs["speaker_id"]
        hand_indices = kwargs["hand_indices"]
        upper_indices = kwargs["upper_indices"]
        lower_indices = kwargs["lower_indices"]
        face_joint_indices = kwargs["face_joint_indices"]
        trans_indices = kwargs['trans_indices']
         
        body_indices = np.concatenate([upper_indices, lower_indices, trans_indices, face_joint_indices], axis=0)

        
        hand = motion[..., hand_indices]
        body = motion[..., body_indices]
        
        hand = self.hand_in(hand)
        body = self.body_in(body)
    
        context = self.context_embedder(context)  # (N, L, D)
        # if self.use_expression and expression is not None:
        #     expression = self.expression_in(expression)  # (N, L, D)
        
        if self.use_expression and expression is not None:
            x = torch.cat([motion, expression], dim=-1)
            xhand = self.xhand_in(x)
            xbody = self.xbody_in(x)
            hand = self.hand_fuse(torch.cat([hand, xhand, context], dim=-1))
            body = self.body_fuse(torch.cat([body, xbody, context], dim=-1))
        else:
            hand = self.hand_fuse(torch.cat([hand, context], dim=-1))
            body = self.body_fuse(torch.cat([body, context], dim=-1))
        
        t = self.motion_t_embedder(timestep)  # (N, T, D)
        id_embed = self.id_embedder(id)
        if len(t.shape) == 2:
            t = t.unsqueeze(1)
        t = t + id_embed.unsqueeze(1)
        if self.simple_modulation:
            t0 = self.t_block(t)
        else:
            t0 = t
        
        if clean_motion is not None:
            clean_hand = clean_motion[..., hand_indices]
            clean_body = clean_motion[..., body_indices]
            
            clean_hand = self.hand_in(clean_hand)
            clean_body = self.body_in(clean_body)
            
            if self.use_expression and expression is not None:
                x = torch.cat([clean_motion, expression], dim=-1)
                xhand = self.xhand_in(x)
                xbody = self.xbody_in(x)
                clean_hand = self.hand_fuse(torch.cat([clean_hand, xhand, context], dim=-1))
                clean_body = self.body_fuse(torch.cat([clean_body, xbody, context], dim=-1))
            else:
                clean_hand = self.hand_fuse(torch.cat([clean_hand, context], dim=-1))
                clean_body = self.body_fuse(torch.cat([clean_body, context], dim=-1))

            hand = torch.cat([clean_hand, hand], dim=1)
            body = torch.cat([clean_body, body], dim=1)
            if aug_timestep is None:
                aug_timestep = torch.zeros_like(timestep)
            t_clean = self.motion_t_embedder(aug_timestep)
            t_clean = t_clean + id_embed.unsqueeze(1)
            if self.simple_modulation:
                clean_t0 = self.t_block(t_clean)
            else:
                clean_t0 = t_clean
            t0 = torch.cat([clean_t0, t0], dim=1)
        
        pos_embed = None
        if self.use_pe:
            if self.pos_embed_type == "sincos":
                hand = hand + self.pos_embed[:,:hand.shape[1], :]  # (N, T, D)
                body = body + self.pos_embed[:,:body.shape[1], :]  # (N, T, D)
                if clean_motion is not None:
                    clean_hand = clean_hand + self.pos_embed[:,:clean_hand.shape[1], :]  # (N, T, D)
                    clean_body = clean_body + self.pos_embed[:,:clean_body.shape[1], :]  # (N, T, D)
            elif self.pos_embed_type == "rope":
                if clean_motion is not None:
                    seq_len_clean = hand.shape[1] // 2 
                    
                    # 1. 生成单份长度 (T) �?Pos Embedding
                    rope_emb = self.pos_embed._prepare_latent_sequence_ids(
                        hand.shape[0], seq_len_clean, hand.device, self.dtype)
                    rope_emb = self.pos_embed(rope_emb) # (1, T, D)
                    
                    # 2. 拼接两份相同�?Embedding -> (1, 2T, D)
                    # 前半部分�?Clean，后半部分给 Noisy，它们的 Pos ID 都是 0 ~ T-1
                    pos_embed  = []
                    for emb in rope_emb:
                        pos_embed.append(torch.cat([emb, emb], dim=0))
                else:
                    pos_embed = self.pos_embed._prepare_latent_sequence_ids(
                            hand.shape[0], hand.shape[1], hand.device, self.dtype)
                    pos_embed = self.pos_embed(pos_embed)
            elif self.pos_embed_type == "none":
                pass
            else:
                raise NotImplementedError(f"pos embed type {self.pos_embed_type} not implemented")
        
        kwargs = dict(
            t = t0,
            mask = self.block_mask,
            rotary_emb = pos_embed,
            kv_cache = None,
            crossattn_cache = None,
            current_start = 0,
        )
        
        for block in self.blocks:
            if torch.is_grad_enabled():
                hand, body = auto_grad_checkpoint(block, hand=hand, body=body, **kwargs)
            else:
                hand, body = block(hand=hand, body=body, **kwargs)
        
        if clean_motion is not None:
            hand = hand[:,hand.shape[1] // 2:]
            body = body[:,body.shape[1] // 2:]
            t0 = t0[:,t0.shape[1] // 2:]

        
        hand = self.hand_final_layer(hand, t)
        body = self.body_final_layer(body, t)
        
        output_motion = torch.zeros_like(motion)
        output_motion[..., hand_indices] = hand
        output_motion[..., body_indices] = body
        return {
            "motion": output_motion, 
        }
    
    def forward_inference(
        self,  
        motion,
        context,
        timestep,
        expression=None,
        kv_cache: dict = None,
        seq_len: int = None,
        current_start: int = 0,
        **kwargs
    ):
        """
        Forward pass of Sana.
            motion: (N, T, D) tensor of motion
            expression: (N, T, D) tensor of expression
            timestep: (N, T) tensor of timestep
            cond: (N, L, D) tensor of condition
            cond_mask: (N, L) tensor of condition mask
        """        
        motion = motion.to(self.dtype)
        timestep = timestep.to(self.dtype)
        pos_embed = self.pos_embed.to(self.dtype)

        id = kwargs["speaker_id"]
        hand_indices = kwargs["hand_indices"]
        upper_indices = kwargs["upper_indices"]
        lower_indices = kwargs["lower_indices"]
        face_joint_indices = kwargs["face_joint_indices"]
        trans_indices = kwargs['trans_indices']
         
        body_indices = np.concatenate([upper_indices, lower_indices, trans_indices, face_joint_indices], axis=0)
        #########################################################
        # for expression
        #########################################################
        
        hand = motion[..., hand_indices]
        body = motion[..., body_indices]
        
        hand = self.hand_in(hand)
        body = self.body_in(body)
      
        context = self.context_embedder(context)  # (N, L, D)
        # if self.use_expression and expression is not None:
        #     expression = self.expression_in(expression)  # (N, L, D)

        if self.use_expression and expression is not None:
            x = torch.cat([motion, expression], dim=-1)
            xhand = self.xhand_in(x)
            xbody = self.xbody_in(x)
            hand = self.hand_fuse(torch.cat([hand, xhand, context], dim=-1))
            body = self.body_fuse(torch.cat([body, xbody, context], dim=-1))
        else:
            hand = self.hand_fuse(torch.cat([hand, context], dim=-1))
            body = self.body_fuse(torch.cat([body, context], dim=-1))
      
        t = self.motion_t_embedder(timestep)  # (N, T, D)
        id_embed = self.id_embedder(id)
        if len(t.shape) == 2:
            t = t.unsqueeze(1)
        t = t + id_embed.unsqueeze(1)
        if self.simple_modulation:
            t0 = self.t_block(t)
        else:
            t0 = t
        
        pos_embed = None
        if self.use_pe:
            if self.pos_embed_type == "sincos":
                hand = hand + self.pos_embed[:,:hand.shape[1], :]  # (N, T, D)
                body = body + self.pos_embed[:,:body.shape[1], :]  # (N, T, D)
            elif self.pos_embed_type == "rope":
                seq_len = seq_len if seq_len is not None else hand.shape[1]
                pos_embed = self.pos_embed._prepare_latent_sequence_ids(
                        hand.shape[0], seq_len, hand.device, self.dtype)
                pos_embed = self.pos_embed(pos_embed)
            elif self.pos_embed_type == "none":
                pass
            else:
                raise NotImplementedError(f"pos embed type {self.pos_embed_type} not implemented")
        
        kwargs = dict(
            t = t0,
            mask = None,
            rotary_emb = pos_embed,
        )
        
        for block_index, block in enumerate(self.blocks):
            kwargs.update(
                {
                    "kv_cache": kv_cache[block_index],
                    "current_start": current_start,
                }
            )
            if torch.is_grad_enabled():
                hand, body = auto_grad_checkpoint(block, hand=hand, body=body, **kwargs)
            else:
                hand, body = block(hand=hand, body=body, **kwargs)

        
        hand = self.hand_final_layer(hand, t)
        body = self.body_final_layer(body, t)
        
        output_motion = torch.zeros_like(motion)
        output_motion[..., hand_indices] = hand
        output_motion[..., body_indices] = body
        return {
            "motion": output_motion, 
        }   
    

if __name__ == "__main__":
    import imageio
    import cv2
    ## for teacher forcing
    block_mask = CausalMMDITEmage._prepare_teacher_forcing_mask(
        num_frames = 256,
        frame_seqlen = 1,
        num_frame_per_block = 64,
        device=torch.device("cuda"), 
        debug=True
    )
    print(block_mask)    
    block_mask = cv2.resize(block_mask[0, 0].cpu().float().numpy(), (1024, 1024))
    imageio.imwrite("teacher_forcing_mask_%d.jpg" % (0), np.uint8(255. * block_mask))
    ## for blockwise causal mask
    block_mask = CausalMMDITEmage._prepare_blockwise_causal_attn_mask(
        device=torch.device("cuda"), 
        num_frames = 9,
        frame_seqlen = 256,
        debug=True
    )
    print(block_mask)    
    block_mask = cv2.resize(block_mask[0, 0].cpu().float().numpy(), (1024, 1024))
    imageio.imwrite("blockwise_causal_mask_%d.jpg" % (0), np.uint8(255. * block_mask))
    
    block_mask = CausalMMDITEmage._prepare_blockwise_cross_attn_mask(
        device=torch.device("cuda"), 
        num_frames = 256,
        frame_seqlen = 1,
        num_frame_per_block = 64,
        local_attn_size = 256,
        debug=True,
        context_length = 256
    )
    print(block_mask.shape)    
    block_mask = cv2.resize(block_mask[0, 0].cpu().float().numpy(), (1024, 1024))
    imageio.imwrite("blockwise_cross_attn_mask_%d.jpg" % (0), np.uint8(255. * block_mask))
    
    ## for blockwise causal mask i2v
    block_mask = CausalMMDITEmage._prepare_blockwise_causal_attn_mask_i2v(
        device=torch.device("cuda"), 
        num_frames = 20,
        frame_seqlen = 256,
        num_frame_per_block = 4,
        debug=True
    )
    print(block_mask)    
    block_mask = cv2.resize(block_mask[0, 0].cpu().float().numpy(), (1024, 1024))
    imageio.imwrite("blockwise_causal_mask_i2v_%d.jpg" % (0), np.uint8(255. * block_mask))
    

    from diffusion.model.nets.positional_embeding import RopePosEmbed
    rotary_emb = RopePosEmbed(theta=10000, axes_dim=[32])
    rotary_emb = rotary_emb(torch.arange(0, 1024, device=torch.device("cuda")).unsqueeze(-1))
    kv_cache = None
    crossattn_cache = None
    current_start = 0
    from diffusion.data.datasets.beat2.data_tools import GetBodyPartIndices
    hand_indices = GetBodyPartIndices("hand", "rot6d")
    upper_indices = GetBodyPartIndices("upper", "rot6d")
    lower_indices = GetBodyPartIndices("lower", "rot6d")
    face_joint_indices = GetBodyPartIndices("face", "rot6d")
    trans_indices = np.array([330, 331, 332])  # the last 3 dims are translation
    
    mm_block = CausalMMDITBlock(
        hidden_size=256,
        num_heads=8,
        head_dim=None,
        simple_modulation=False,
        bias=True,
        norm_type="rms",
        attn_type="flash",
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
        is_teacher_forcing=True,
        local_attn_size=-1,
        padding_unit_length=128,
        use_crossattn_mask=True,
    )
    block_mask = CausalMMDITEmage._prepare_teacher_forcing_mask(
        num_frames = 256,
        frame_seqlen = 1,
        num_frame_per_block = 64,
        device=torch.device("cuda"), 
        debug=False
    )
    crossattn_mask = CausalMMDITEmage._prepare_blockwise_cross_attn_mask(
        device=torch.device("cuda"), 
        num_frames = 256,
        frame_seqlen = 1,
        num_frame_per_block = 64,
        local_attn_size = 256,
        debug=True,
        context_length = 256
    )
    
    motion = torch.randn(4, 256, 256).cuda()
    motion_noise = torch.randn(4, 256, 256).cuda()
    motion2 = torch.randn(4, 256, 256).cuda()
    motion2_noise = torch.randn(4, 256, 256).cuda()
    motion = torch.cat([motion, motion_noise], dim=1)
    motion2 = torch.cat([motion2, motion2_noise], dim=1)
    
    context = torch.randn(4, 256, 256).cuda()
    t = torch.randn((4, 256)).cuda()
    mm_block.cuda()
    motion_out, motion2_out = mm_block(
        hand = motion, 
        body = motion2,
        context = context,
        t = t,
        mask = block_mask,
        crossattn_mask = crossattn_mask,
        rotary_emb = rotary_emb,
        kv_cache=kv_cache,
        crossattn_cache=crossattn_cache,
        current_start=current_start,
    )
    print(motion_out.shape, motion2_out.shape)
    
    motion = torch.randn(4, 256, 333).cuda()
    clean_motion = torch.randn(4, 256, 333).cuda()
    expression = None
    context = torch.randn(4, 256, 256).cuda()
    timestep = torch.randn(4, 256).cuda()
    id = torch.tensor([0, 1, 2, 3]).cuda()
    net = CausalMMDITEmage(
        use_expression=False,
        use_crossattn_mask=True,
        hidden_size=256,
        caption_channels=256,
        depth=4,
        drop_path=0.2,
        inputlayer_config=DEFAULT_INPUT_CONFIG,
        timestep_embedder_config=DEFAULT_TIMESTEP_EMBEDDER_CONFIG,
        caption_embedder_config=DEFAULT_CAPTION_EMBEDDER_CONFIG,
        block_config=DEFAULT_BLOCK_CONFIG,
        finallayer_config=DEFAULT_FINALLAYER_CONFIG,
        work_dir=None,
        use_pe=True,
        pos_embed_type="rope",
        simple_modulation=False,
    )
    net.cuda()
    motion = net(
        motion,
        expression=expression,
        context=context,
        timestep=timestep,
        clean_motion=clean_motion,
        aug_timestep=None,
        speaker_id=id,
        hand_indices=hand_indices,
        upper_indices=upper_indices,
        lower_indices=lower_indices,
        face_joint_indices=face_joint_indices,
        trans_indices=trans_indices,
    )
    for k, v in motion.items():
        print(k, v.shape)
