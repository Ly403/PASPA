import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import DropPath
from diffusion.model.nets.causal.causal_block import (
    CrossCausalSelfAttention,
    BlockWiseAttention,
    TimestepAdaptiveFusionGate,
)
from diffusion.model.nets.basic_modules import DoubleStreamMLP, DoubleStreamGatedMLP, StyleEmbedding
from diffusion.model.nets.basic_modules import get_1d_sincos_pos_embed_from_grid, t2i_modulate
from diffusion.model.nets.positional_embeding import RopePosEmbed
from torch.nn.attention.flex_attention import BlockMask, create_block_mask, create_mask
from diffusion.model.norms import build_norm
from copy import deepcopy
from diffusion.model.utils import auto_grad_checkpoint
from diffusion.utils.dist_utils import get_rank
from diffusion.utils.config import instantiate_from_config
from diffusion.utils.logger import get_root_logger
from omegaconf import DictConfig
from diffusion.model.nets.causal.gated_deltanet import GatedDeltaNetForBLD, ZeroGatedDeltaNetForBLD

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
    target="diffusion.model.nets.causal.causal_emage.CausalEmageBlock",
    params=dict(
        hidden_size=None,
        num_heads=8,
        head_dim=None,
        mapping_size=768,  # time embedding input dimension
        simple_modulation=False,
        bias=True,
        norm_type="rms",
        attn_type="flash",
        qk_norm_type="identity",
        ffn_type="mlp",
        ffn_act_type="gelu",
        ffn_proj_type="linear",
        ffn_mlp_ratio=4.0,
        attn_drop=0.0,
        attn_proj_drop=0.0,
        ffn_drop=0.0,
        drop_path=0.0,
        norm_eps=1e-6,
        num_frame_per_block=64,
        is_teacher_forcing=False,
        local_attn_size=-1,
        padding_unit_length=128,
        max_attention_size=512,
        sink_size=0,
        # Cross-attention type: "standard", "deltanet_block", or "deltanet_hierarchical"
        cross_attn_type="standard",
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

# Keep a dedicated fuse config entry (defaulting to InputLayer defaults)
# so fuse/input can be tuned independently in configs later.
DEFAULT_FUSE_INPUT_CONFIG = deepcopy(DEFAULT_INPUT_CONFIG)

DEFAULT_FINALLAYER_CONFIG=DictConfig(dict(
    target="diffusion.model.nets.causal.causal_block.FinalLayer",
    params=dict(
        hidden_size=None, 
        out_channels=None,
        mapping_size=768,  # 新增：time embedding 的输入维度
        bias=True,
        norm_type="rms",
        proj_type="linear",
        norm_eps=1e-6,
    )
))

class CausalEmageBlock(nn.Module):
    """
    CausalEmageBlock using standard CrossCausalSelfAttention for cross-attention.
    
    For DeltaNet-based cross-attention, use CausalEmageBlockDeltaNet instead.
    """
    def __init__(
        self,
        hidden_size,
        num_heads=8,
        head_dim=None,
        mapping_size=768,  # time embedding input dimension
        simple_modulation=False,
        bias=True,
        norm_type="rms",
        attn_type="flash",
        qk_norm_type="identity",
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
        num_frame_per_block=64,
        local_attn_size=-1,
        padding_unit_length=128,
        max_attention_size=512,
        sink_size=0,
        cross_attn_impl_type=None,
        add_cross_attn_prefix=False,
        # Timestep-adaptive fusion parameters
        use_timestep_fusion=False,
        timestep_dim=None,
        fusion_gate_hidden_dim=None,
        fusion_gate_activation="silu",
        **kwargs,  # Ignore unknown parameters for compatibility
    ):
        super().__init__()
        self.num_frame_per_block = num_frame_per_block
        self.use_timestep_fusion = use_timestep_fusion

        self.hand_norm1 = build_norm(norm_type, num_features=hidden_size, affine=False, eps=norm_eps)
        self.body_norm1 = build_norm(norm_type, num_features=hidden_size, affine=False, eps=norm_eps)
        self.hand_norm2 = build_norm(norm_type, num_features=hidden_size, affine=False, eps=norm_eps)
        self.body_norm2 = build_norm(norm_type, num_features=hidden_size, affine=False, eps=norm_eps)
        self.hand_norm3 = build_norm(norm_type, num_features=hidden_size, affine=False, eps=norm_eps)
        self.body_norm3 = build_norm(norm_type, num_features=hidden_size, affine=False, eps=norm_eps)
        cross_attn_impl_type = attn_type if cross_attn_impl_type is None else cross_attn_impl_type
        
        self.attn = BlockWiseAttention(
            dim=hidden_size,
            num_heads=num_heads,
            head_dim=head_dim,
            bias=bias,
            attn_type=attn_type,
            qk_norm_type=qk_norm_type,
            attn_drop=attn_drop,
            proj_drop=attn_proj_drop,
            block_size=num_frame_per_block,
        )
        self.attn2 = deepcopy(self.attn)
        
        # Determine timestep_dim for fusion gate
        fusion_timestep_dim = timestep_dim if timestep_dim is not None else mapping_size
        
        # Standard CrossCausalSelfAttention (softmax attention)
        self.cross_attn = CrossCausalSelfAttention(
            dim=hidden_size,
            num_heads=num_heads,
            head_dim=head_dim,
            bias=bias,
            attn_type=cross_attn_impl_type,
            qk_norm_type=qk_norm_type,
            attn_drop=attn_drop,
            proj_drop=attn_proj_drop,
            is_teacher_forcing=is_teacher_forcing,
            local_attn_size=local_attn_size,
            padding_unit_length=padding_unit_length,
            max_attention_size=max_attention_size,
            sink_size=sink_size,
            add_learnable_prefix=add_cross_attn_prefix,
        )
        self.cross_attn2 = deepcopy(self.cross_attn)
        
        # Timestep-adaptive fusion gates for parallel Self-Attn / Cross-Attn architecture
        # When enabled, Self-Attn and Cross-Attn are computed in parallel and fused via learned gate
        if use_timestep_fusion:
            self.hand_fusion_gate = TimestepAdaptiveFusionGate(
                hidden_dim=hidden_size,
                timestep_dim=fusion_timestep_dim,
                gate_hidden_dim=fusion_gate_hidden_dim,
                gate_activation=fusion_gate_activation,
            )
            self.body_fusion_gate = TimestepAdaptiveFusionGate(
                hidden_dim=hidden_size,
                timestep_dim=fusion_timestep_dim,
                gate_hidden_dim=fusion_gate_hidden_dim,
                gate_activation=fusion_gate_activation,
            )
 
        
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
        self.modulation_chunks = 9
        if self.simple_modulation:
            self.hand_scale_shift_table = nn.Parameter(torch.randn(self.modulation_chunks, hidden_size))
            self.body_scale_shift_table = nn.Parameter(torch.randn(self.modulation_chunks, hidden_size))
        else:
            self.hand_time_in = nn.Sequential(nn.SiLU(), nn.Linear(mapping_size, self.modulation_chunks * hidden_size, bias=bias))
            self.body_time_in = nn.Sequential(nn.SiLU(), nn.Linear(mapping_size, self.modulation_chunks * hidden_size, bias=bias))

        self.hand_drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.body_drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def initialize_weights(self):
        if self.simple_modulation:
            nn.init.constant_(self.hand_scale_shift_table, 0)
            nn.init.constant_(self.body_scale_shift_table, 0)
        else:
            nn.init.constant_(self.hand_time_in[1].weight, val=0)
            if self.hand_time_in[1].bias is not None:
                nn.init.constant_(self.hand_time_in[1].bias, val=0)

            nn.init.constant_(self.body_time_in[1].weight, val=0)
            if self.body_time_in[1].bias is not None:
                nn.init.constant_(self.body_time_in[1].bias, val=0)

    def _compute_modulation_params(self, t: torch.FloatTensor):
        if len(t.shape) not in (2, 3):
            raise ValueError(f"t.shape must be (B, T, D) or (B, D), but got {t.shape}")

        if self.simple_modulation:
            if len(t.shape) == 3:
                hand_chunks = (
                    self.hand_scale_shift_table[None, None, :, :] + t[:, :, None, :]
                ).unbind(dim=2)
                body_chunks = (
                    self.body_scale_shift_table[None, None, :, :] + t[:, :, None, :]
                ).unbind(dim=2)
            else:
                hand_chunks = (
                    self.hand_scale_shift_table[None, :, :] + t[:, None, :]
                ).unbind(dim=1)
                body_chunks = (
                    self.body_scale_shift_table[None, :, :] + t[:, None, :]
                ).unbind(dim=1)
        else:
            t_in = t if len(t.shape) == 3 else t.unsqueeze(1)
            hand_chunks = self.hand_time_in(t_in).chunk(self.modulation_chunks, dim=-1)
            body_chunks = self.body_time_in(t_in).chunk(self.modulation_chunks, dim=-1)

        hand_shift_msa, hand_scale_msa, hand_gate_msa, hand_shift_cross, hand_scale_cross, hand_gate_cross, hand_shift_mlp, hand_scale_mlp, hand_gate_mlp = hand_chunks
        body_shift_msa, body_scale_msa, body_gate_msa, body_shift_cross, body_scale_cross, body_gate_cross, body_shift_mlp, body_scale_mlp, body_gate_mlp = body_chunks

        return (
            hand_shift_msa, hand_scale_msa, hand_gate_msa,
            hand_shift_cross, hand_scale_cross, hand_gate_cross,
            hand_shift_mlp, hand_scale_mlp, hand_gate_mlp,
            body_shift_msa, body_scale_msa, body_gate_msa,
            body_shift_cross, body_scale_cross, body_gate_cross,
            body_shift_mlp, body_scale_mlp, body_gate_mlp,
        )
                
    def forward(
        self, 
        hand: torch.FloatTensor,
        body: torch.FloatTensor, 
        t: torch.FloatTensor,
        block_mask: torch.BoolTensor = None,
        clean_hand: torch.FloatTensor = None,
        clean_body: torch.FloatTensor = None,
        rotary_emb: torch.FloatTensor = None,
        use_liger_rope: bool = False,
        hand_kv_cache=None,
        body_kv_cache=None,
        current_start=0,
        update_kv_cache=False,
        is_inference=False,
        **kwargs
    ):
        assert hand.shape[0] == body.shape[0], "batch size must be the same"
        B, N, C = hand.shape

        (
            hand_shift_msa, hand_scale_msa, hand_gate_msa,
            hand_shift_cross, hand_scale_cross, hand_gate_cross,
            hand_shift_mlp, hand_scale_mlp, hand_gate_mlp,
            body_shift_msa, body_scale_msa, body_gate_msa,
            body_shift_cross, body_scale_cross, body_gate_cross,
            body_shift_mlp, body_scale_mlp, body_gate_mlp,
        ) = self._compute_modulation_params(t)
        
        ##################################### Parallel Self-Attn & Cross-Attn Layer #######################################
        if self.use_timestep_fusion:
            # === Parallel Architecture ===
            # Both Self-Attn and Cross-Attn use the original input, then fuse via learned gate
            
            # Branch 1: Self-Attention (block-internal attention)
            self_hand = self.hand_norm1(hand)
            self_body = self.body_norm1(body)
            self_hand = t2i_modulate(self_hand, hand_shift_msa, hand_scale_msa)
            self_body = t2i_modulate(self_body, body_shift_msa, body_scale_msa)
            self_hand = self.attn(self_hand, rotary_emb=rotary_emb, current_start=current_start, is_inference=is_inference)
            self_body = self.attn2(self_body, rotary_emb=rotary_emb, current_start=current_start, is_inference=is_inference)
            self_hand = hand_gate_msa * self_hand
            self_body = body_gate_msa * self_body
            
            # Branch 2: Cross-Attention (historical blocks attention)
            cross_hand = self.hand_norm2(hand)
            cross_body = self.body_norm2(body)
            cross_hand = t2i_modulate(cross_hand, hand_shift_cross, hand_scale_cross)
            cross_body = t2i_modulate(cross_body, body_shift_cross, body_scale_cross)
            # Pass t_emb for DeltaNet variants; standard CrossCausalSelfAttention ignores it via **kwargs
            cross_hand = self.cross_attn(
                cross_hand, clean_hand, block_mask, rotary_emb, use_liger_rope, hand_kv_cache, current_start, update_kv_cache)
            cross_body = self.cross_attn2(
                cross_body, clean_body, block_mask, rotary_emb, use_liger_rope, body_kv_cache, current_start, update_kv_cache)
            cross_hand = hand_gate_cross * cross_hand
            cross_body = body_gate_cross * cross_body
            
            # Compute fusion coefficients based on timestep and hidden state
            # alpha high (early timesteps) -> rely more on history (cross_attn)
            # alpha low (late timesteps) -> rely more on current block (self_attn)
            # Note: alpha is computed per-frame (B, T, 1) because different blocks
            # may have different timesteps in BLD training
            alpha_hand = self.hand_fusion_gate(t, hand)  # (B, T, 1)
            alpha_body = self.body_fusion_gate(t, body)  # (B, T, 1)
            
            # Fuse: alpha * cross_attn_out + (1 - alpha) * self_attn_out
            residual_hand = alpha_hand * cross_hand + (1 - alpha_hand) * self_hand
            residual_body = alpha_body * cross_body + (1 - alpha_body) * self_body
            
            # Single residual connection (replaces the two separate residual connections)
            hand = hand + self.hand_drop_path(residual_hand)
            body = body + self.body_drop_path(residual_body)
        else:
            # === Original Serial Architecture (backward compatible) ===
            ##################################### The first layer #######################################
            residual_hand = self.hand_norm1(hand)
            residual_body = self.body_norm1(body)
            
            residual_hand = t2i_modulate(residual_hand, hand_shift_msa, hand_scale_msa)
            residual_body = t2i_modulate(residual_body, body_shift_msa, body_scale_msa)

            # 块内双向注意力
            residual_hand = self.attn(residual_hand, rotary_emb=rotary_emb, current_start=current_start, is_inference=is_inference)
            residual_body = self.attn2(residual_body, rotary_emb=rotary_emb, current_start=current_start, is_inference=is_inference)

            residual_hand = hand_gate_msa * residual_hand
            residual_body = body_gate_msa * residual_body
            
            hand = hand + self.hand_drop_path(residual_hand)
            body = body + self.body_drop_path(residual_body)
                    
            ####################################### The second layer ############################################
            residual_hand = self.hand_norm2(hand)
            residual_body = self.body_norm2(body)

            residual_hand = t2i_modulate(residual_hand, hand_shift_cross, hand_scale_cross)
            residual_body = t2i_modulate(residual_body, body_shift_cross, body_scale_cross)

            # Autoregressive cross-attention with history
            # Pass t_emb for DeltaNet variants; standard CrossCausalSelfAttention ignores it via **kwargs
            residual_hand = self.cross_attn(
                residual_hand, clean_hand, block_mask, rotary_emb, use_liger_rope, hand_kv_cache, current_start, update_kv_cache)
            residual_body = self.cross_attn2(
                residual_body, clean_body, block_mask, rotary_emb, use_liger_rope, body_kv_cache, current_start, update_kv_cache)

            residual_hand = hand_gate_cross * residual_hand
            residual_body = body_gate_cross * residual_body
            
            hand = hand + self.hand_drop_path(residual_hand)
            body = body + self.body_drop_path(residual_body)
        
        ######################################## The third layer ########################################        
        residual_hand = self.hand_norm3(hand)
        residual_body = self.body_norm3(body)

        residual_hand = t2i_modulate(residual_hand, hand_shift_mlp, hand_scale_mlp)
        residual_body = t2i_modulate(residual_body, body_shift_mlp, body_scale_mlp)
        
        residual_hand, residual_body = self.mlp(residual_hand, residual_body)
        
        residual_hand = hand_gate_mlp * residual_hand
        residual_body = body_gate_mlp * residual_body
        
        hand = hand + self.hand_drop_path(residual_hand)
        body = body + self.body_drop_path(residual_body)
        
        return hand, body


class CausalEmageBlockCrossFirst(CausalEmageBlock):
    """
    Standard cross-attention block variant that keeps all modules and interfaces
    identical to [CausalEmageBlock](diffusion/model/nets/causal/causal_emage.py:110),
    but swaps the serial execution order to:

        cross-attn -> self-attn -> ffn

    The parallel timestep-fusion path is intentionally kept identical to the
    parent implementation to avoid changing existing semantics outside the
    serial ordering ablation.
    """

    def forward(
        self,
        hand: torch.FloatTensor,
        body: torch.FloatTensor,
        t: torch.FloatTensor,
        block_mask: torch.BoolTensor = None,
        clean_hand: torch.FloatTensor = None,
        clean_body: torch.FloatTensor = None,
        rotary_emb: torch.FloatTensor = None,
        use_liger_rope: bool = False,
        hand_kv_cache=None,
        body_kv_cache=None,
        current_start=0,
        update_kv_cache=False,
        is_inference=False,
        **kwargs,
    ):
        if self.use_timestep_fusion:
            return super().forward(
                hand=hand,
                body=body,
                t=t,
                block_mask=block_mask,
                clean_hand=clean_hand,
                clean_body=clean_body,
                rotary_emb=rotary_emb,
                use_liger_rope=use_liger_rope,
                hand_kv_cache=hand_kv_cache,
                body_kv_cache=body_kv_cache,
                current_start=current_start,
                update_kv_cache=update_kv_cache,
                is_inference=is_inference,
                **kwargs,
            )

        assert hand.shape[0] == body.shape[0], "batch size must be the same"

        (
            hand_shift_msa, hand_scale_msa, hand_gate_msa,
            hand_shift_cross, hand_scale_cross, hand_gate_cross,
            hand_shift_mlp, hand_scale_mlp, hand_gate_mlp,
            body_shift_msa, body_scale_msa, body_gate_msa,
            body_shift_cross, body_scale_cross, body_gate_cross,
            body_shift_mlp, body_scale_mlp, body_gate_mlp,
        ) = self._compute_modulation_params(t)

        ####################################### Layer 1: Cross-Attention ############################################
        residual_hand = self.hand_norm2(hand)
        residual_body = self.body_norm2(body)

        residual_hand = t2i_modulate(residual_hand, hand_shift_cross, hand_scale_cross)
        residual_body = t2i_modulate(residual_body, body_shift_cross, body_scale_cross)

        residual_hand = self.cross_attn(
            residual_hand,
            clean_hand,
            block_mask,
            rotary_emb,
            use_liger_rope,
            hand_kv_cache,
            current_start,
            update_kv_cache,
        )
        residual_body = self.cross_attn2(
            residual_body,
            clean_body,
            block_mask,
            rotary_emb,
            use_liger_rope,
            body_kv_cache,
            current_start,
            update_kv_cache,
        )

        residual_hand = hand_gate_cross * residual_hand
        residual_body = body_gate_cross * residual_body

        hand = hand + self.hand_drop_path(residual_hand)
        body = body + self.body_drop_path(residual_body)

        ##################################### Layer 2: Block-wise Self-Attention #######################################
        residual_hand = self.hand_norm1(hand)
        residual_body = self.body_norm1(body)

        residual_hand = t2i_modulate(residual_hand, hand_shift_msa, hand_scale_msa)
        residual_body = t2i_modulate(residual_body, body_shift_msa, body_scale_msa)

        residual_hand = self.attn(
            residual_hand,
            rotary_emb=rotary_emb,
            current_start=current_start,
            is_inference=is_inference,
        )
        residual_body = self.attn2(
            residual_body,
            rotary_emb=rotary_emb,
            current_start=current_start,
            is_inference=is_inference,
        )

        residual_hand = hand_gate_msa * residual_hand
        residual_body = body_gate_msa * residual_body

        hand = hand + self.hand_drop_path(residual_hand)
        body = body + self.body_drop_path(residual_body)

        ######################################## Layer 3: FFN ########################################
        residual_hand = self.hand_norm3(hand)
        residual_body = self.body_norm3(body)

        residual_hand = t2i_modulate(residual_hand, hand_shift_mlp, hand_scale_mlp)
        residual_body = t2i_modulate(residual_body, body_shift_mlp, body_scale_mlp)

        residual_hand, residual_body = self.mlp(residual_hand, residual_body)

        residual_hand = hand_gate_mlp * residual_hand
        residual_body = body_gate_mlp * residual_body

        hand = hand + self.hand_drop_path(residual_hand)
        body = body + self.body_drop_path(residual_body)

        return hand, body


class CausalEmageBlockDeltaNet(nn.Module):
    """
    CausalEmageBlock variant using Gated DeltaNet for cross-attention.
    
    This is a cleaner implementation that separates DeltaNet logic from the original
    CrossCausalSelfAttention-based block. Key differences:
    
    1. Uses GatedDeltaNetForBLD for cross-attention (linear complexity)
    2. Simpler interface - no block_mask needed (DeltaNet handles causality via state)
    3. Timestep embedding is passed to DeltaNet for gate modulation
    
    Architecture:
    - Layer 1: DeltaNet cross-attention (current block queries historical state)
    - Layer 2: Block-wise self-attention (within current block)
    - Layer 3: FFN
    """
    def __init__(
        self,
        hidden_size,
        num_heads=8,
        head_dim=None,
        mapping_size=768,
        simple_modulation=False,
        bias=True,
        norm_type="rms",
        attn_type="flash",
        qk_norm_type="identity",
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
        num_frame_per_block=64,
        local_attn_size=-1,
        padding_unit_length=128,
        max_attention_size=512,
        sink_size=0,
        add_cross_attn_prefix=False,
        # DeltaNet-specific parameters
        deltanet_gate_low_rank_dim=16,
        deltanet_gate_logit_normalizer=16.0,
        deltanet_allow_neg_eigval=False,
        deltanet_use_short_conv=True,
        deltanet_conv_size=4,
        deltanet_use_triton_kernel=False,
        deltanet_scale=None,
        deltanet_q_t_modulation_mode="none",
        deltanet_q_t_modulation_dim=768,
        deltanet_kv_t_modulation_mode="none",
        deltanet_kv_t_modulation_dim=768,
        deltanet_use_learnable_init_state: bool = False,
        history_cfg_enabled: bool = False,
        history_cfg_p_clean: float = 0.3,
        history_cfg_p_null: float = 0.2,
        history_cfg_null_state_mode: str = "learned",
        # Timestep-adaptive fusion parameters
        use_timestep_fusion=False,
        timestep_dim=None,
        fusion_gate_hidden_dim=None,
        fusion_gate_activation="silu",
        **kwargs,
    ):
        super().__init__()
        self.num_frame_per_block = num_frame_per_block
        self.use_timestep_fusion = use_timestep_fusion

        # Normalization layers
        self.hand_norm1 = build_norm(norm_type, num_features=hidden_size, affine=False, eps=norm_eps)
        self.body_norm1 = build_norm(norm_type, num_features=hidden_size, affine=False, eps=norm_eps)
        self.hand_norm2 = build_norm(norm_type, num_features=hidden_size, affine=False, eps=norm_eps)
        self.body_norm2 = build_norm(norm_type, num_features=hidden_size, affine=False, eps=norm_eps)
        self.hand_norm3 = build_norm(norm_type, num_features=hidden_size, affine=False, eps=norm_eps)
        self.body_norm3 = build_norm(norm_type, num_features=hidden_size, affine=False, eps=norm_eps)
        
        # Block-wise self-attention (within current block)
        self.attn = BlockWiseAttention(
            dim=hidden_size,
            num_heads=num_heads,
            head_dim=head_dim,
            bias=bias,
            attn_type=attn_type,
            qk_norm_type=qk_norm_type,
            attn_drop=attn_drop,
            proj_drop=attn_proj_drop,
            block_size=num_frame_per_block,
        )
        self.attn2 = deepcopy(self.attn)
        # DeltaNet cross-attention (current block queries historical state)
        self.cross_attn = GatedDeltaNetForBLD(
            dim=hidden_size,
            num_heads=num_heads,
            head_dim=head_dim,
            bias=bias,
            qk_norm=(qk_norm_type != "identity"),  # Convert qk_norm_type to bool
            proj_drop=attn_proj_drop,
            gate_low_rank_dim=deltanet_gate_low_rank_dim,
            gate_logit_normalizer=deltanet_gate_logit_normalizer,
            block_size=num_frame_per_block,
            allow_neg_eigval=deltanet_allow_neg_eigval,
            use_short_conv=deltanet_use_short_conv,
            conv_size=deltanet_conv_size,
            use_triton_kernel=deltanet_use_triton_kernel,
            scale=deltanet_scale,
            q_t_modulation_mode=deltanet_q_t_modulation_mode,
            q_t_modulation_dim=deltanet_q_t_modulation_dim,
            kv_t_modulation_mode=deltanet_kv_t_modulation_mode,
            kv_t_modulation_dim=deltanet_kv_t_modulation_dim,
            use_learnable_init_state=deltanet_use_learnable_init_state,
            history_cfg_enabled=history_cfg_enabled,
            history_cfg_p_clean=history_cfg_p_clean,
            history_cfg_p_null=history_cfg_p_null,
            history_cfg_null_state_mode=history_cfg_null_state_mode,
        )
        self.cross_attn2 = deepcopy(self.cross_attn)

        # Timestep-adaptive fusion gates for parallel Self-Attn / Cross-Attn architecture
        if use_timestep_fusion:
            fusion_timestep_dim = timestep_dim if timestep_dim is not None else mapping_size
            self.hand_fusion_gate = TimestepAdaptiveFusionGate(
                hidden_dim=hidden_size,
                timestep_dim=fusion_timestep_dim,
                gate_hidden_dim=fusion_gate_hidden_dim,
                gate_activation=fusion_gate_activation,
            )
            self.body_fusion_gate = TimestepAdaptiveFusionGate(
                hidden_dim=hidden_size,
                timestep_dim=fusion_timestep_dim,
                gate_hidden_dim=fusion_gate_hidden_dim,
                gate_activation=fusion_gate_activation,
            )

        # FFN
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
        
        # Modulation
        self.simple_modulation = simple_modulation
        self.modulation_chunks = 9
        if self.simple_modulation:
            self.hand_scale_shift_table = nn.Parameter(torch.randn(self.modulation_chunks, hidden_size))
            self.body_scale_shift_table = nn.Parameter(torch.randn(self.modulation_chunks, hidden_size))
        else:
            self.hand_time_in = nn.Sequential(nn.SiLU(), nn.Linear(mapping_size, self.modulation_chunks * hidden_size, bias=bias))
            self.body_time_in = nn.Sequential(nn.SiLU(), nn.Linear(mapping_size, self.modulation_chunks * hidden_size, bias=bias))

        # Drop path
        self.hand_drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.body_drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def initialize_weights(self):
        if self.simple_modulation:
            nn.init.constant_(self.hand_scale_shift_table, 0)
            nn.init.constant_(self.body_scale_shift_table, 0)
        else:
            nn.init.constant_(self.hand_time_in[1].weight, val=0)
            if self.hand_time_in[1].bias is not None:
                nn.init.constant_(self.hand_time_in[1].bias, val=0)
            nn.init.constant_(self.body_time_in[1].weight, val=0)
            if self.body_time_in[1].bias is not None:
                nn.init.constant_(self.body_time_in[1].bias, val=0)

    def _compute_modulation_params(self, t: torch.FloatTensor):
        if len(t.shape) not in (2, 3):
            raise ValueError(f"t.shape must be (B, T, D) or (B, D), but got {t.shape}")

        if self.simple_modulation:
            if len(t.shape) == 3:
                hand_chunks = (
                    self.hand_scale_shift_table[None, None, :, :] + t[:, :, None, :]
                ).unbind(dim=2)
                body_chunks = (
                    self.body_scale_shift_table[None, None, :, :] + t[:, :, None, :]
                ).unbind(dim=2)
            else:
                hand_chunks = (
                    self.hand_scale_shift_table[None, :, :] + t[:, None, :]
                ).unbind(dim=1)
                body_chunks = (
                    self.body_scale_shift_table[None, :, :] + t[:, None, :]
                ).unbind(dim=1)
        else:
            t_in = t if len(t.shape) == 3 else t.unsqueeze(1)
            hand_chunks = self.hand_time_in(t_in).chunk(self.modulation_chunks, dim=-1)
            body_chunks = self.body_time_in(t_in).chunk(self.modulation_chunks, dim=-1)

        hand_shift_msa, hand_scale_msa, hand_gate_msa, hand_shift_cross, hand_scale_cross, hand_gate_cross, hand_shift_mlp, hand_scale_mlp, hand_gate_mlp = hand_chunks
        body_shift_msa, body_scale_msa, body_gate_msa, body_shift_cross, body_scale_cross, body_gate_cross, body_shift_mlp, body_scale_mlp, body_gate_mlp = body_chunks

        return (
            hand_shift_msa, hand_scale_msa, hand_gate_msa,
            hand_shift_cross, hand_scale_cross, hand_gate_cross,
            hand_shift_mlp, hand_scale_mlp, hand_gate_mlp,
            body_shift_msa, body_scale_msa, body_gate_msa,
            body_shift_cross, body_scale_cross, body_gate_cross,
            body_shift_mlp, body_scale_mlp, body_gate_mlp,
        )
                
    def forward(
        self,
        hand: torch.FloatTensor,
        body: torch.FloatTensor,
        t: torch.FloatTensor,
        block_mask: torch.BoolTensor = None,  # Not used by DeltaNet, kept for interface compatibility
        clean_hand: torch.FloatTensor = None,
        clean_body: torch.FloatTensor = None,
        rotary_emb: torch.FloatTensor = None,
        use_liger_rope: bool = False,
        hand_kv_cache=None,
        body_kv_cache=None,
        current_start=0,
        update_kv_cache=False,
        is_inference=False,
        **kwargs
    ):
        """
        Forward pass of CausalEmageBlockDeltaNet.
        
        Args:
            hand: Hand features, shape (B, T, D)
            body: Body features, shape (B, T, D)
            t: Timestep embedding, shape (B, T, mapping_size) or (B, mapping_size)
            block_mask: Not used by DeltaNet (kept for interface compatibility)
            clean_hand: Historical clean hand features for cross-attention
            clean_body: Historical clean body features for cross-attention
            rotary_emb: Rotary position embeddings for self-attention
            use_liger_rope: Whether to use Liger RoPE optimization
            hand_kv_cache: DeltaNet state cache for hand
            body_kv_cache: DeltaNet state cache for body
            current_start: Starting position for RoPE
            update_kv_cache: Whether to update cache (for prefill)
            is_inference: Whether in inference mode
        """
        assert hand.shape[0] == body.shape[0], "batch size must be the same"
        B, N, C = hand.shape

        (
            hand_shift_msa, hand_scale_msa, hand_gate_msa,
            hand_shift_cross, hand_scale_cross, hand_gate_cross,
            hand_shift_mlp, hand_scale_mlp, hand_gate_mlp,
            body_shift_msa, body_scale_msa, body_gate_msa,
            body_shift_cross, body_scale_cross, body_gate_cross,
            body_shift_mlp, body_scale_mlp, body_gate_mlp,
        ) = self._compute_modulation_params(t)
        
        ##################################### Parallel or Serial Self-Attn & Cross-Attn #######################################
        if self.use_timestep_fusion:
            # === Parallel Architecture ===
            # Both Self-Attn and Cross-Attn use the original input, then fuse via learned gate

            # Branch 1: Self-Attention (block-internal attention)
            self_hand = self.hand_norm1(hand)
            self_body = self.body_norm1(body)
            self_hand = t2i_modulate(self_hand, hand_shift_msa, hand_scale_msa)
            self_body = t2i_modulate(self_body, body_shift_msa, body_scale_msa)
            self_hand = self.attn(self_hand, rotary_emb=rotary_emb, current_start=current_start, is_inference=is_inference)
            self_body = self.attn2(self_body, rotary_emb=rotary_emb, current_start=current_start, is_inference=is_inference)
            self_hand = hand_gate_msa * self_hand
            self_body = body_gate_msa * self_body

            # Branch 2: DeltaNet Cross-Attention (historical blocks attention)
            cross_hand = self.hand_norm2(hand)
            cross_body = self.body_norm2(body)
            cross_hand = t2i_modulate(cross_hand, hand_shift_cross, hand_scale_cross)
            cross_body = t2i_modulate(cross_body, body_shift_cross, body_scale_cross)

            if hand_kv_cache is not None:
                if update_kv_cache:
                    # Prefill: write clean frames to state, no cross-attn output
                    self.cross_attn(
                        motion=cross_hand, context=clean_hand, t_emb=t,
                        kv_t_emb=kwargs.get("kv_t_emb", None),
                        kv_cache=hand_kv_cache, current_start=current_start, prefill_only=True,
                    )
                    self.cross_attn2(
                        motion=cross_body, context=clean_body, t_emb=t,
                        kv_t_emb=kwargs.get("kv_t_emb", None),
                        kv_cache=body_kv_cache, current_start=current_start, prefill_only=True,
                    )
                    # Prefill: only self-attn contributes (alpha irrelevant, cross is zero)
                    hand = hand + self.hand_drop_path(self_hand)
                    body = body + self.body_drop_path(self_body)
                else:
                    # Decode: query state with current noisy block
                    cross_hand = self.cross_attn(
                        motion=cross_hand, t_emb=t, kv_cache=hand_kv_cache,
                        current_start=current_start, history_cfg_alpha=kwargs.get("history_cfg_alpha", None),
                    )
                    cross_body = self.cross_attn2(
                        motion=cross_body, t_emb=t, kv_cache=body_kv_cache,
                        current_start=current_start, history_cfg_alpha=kwargs.get("history_cfg_alpha", None),
                    )
                    cross_hand = hand_gate_cross * cross_hand
                    cross_body = body_gate_cross * cross_body

                    alpha_hand = self.hand_fusion_gate(t, hand)  # (B, T, 1)
                    alpha_body = self.body_fusion_gate(t, body)  # (B, T, 1)
                    residual_hand = alpha_hand * cross_hand + (1 - alpha_hand) * self_hand
                    residual_body = alpha_body * cross_body + (1 - alpha_body) * self_body
                    hand = hand + self.hand_drop_path(residual_hand)
                    body = body + self.body_drop_path(residual_body)
            else:
                # Training mode
                cross_hand = self.cross_attn(
                    motion=cross_hand, context=clean_hand, t_emb=t,
                    kv_t_emb=kwargs.get("kv_t_emb", None),
                )
                cross_body = self.cross_attn2(
                    motion=cross_body, context=clean_body, t_emb=t,
                    kv_t_emb=kwargs.get("kv_t_emb", None),
                )
                cross_hand = hand_gate_cross * cross_hand
                cross_body = body_gate_cross * cross_body

                alpha_hand = self.hand_fusion_gate(t, hand)  # (B, T, 1)
                alpha_body = self.body_fusion_gate(t, body)  # (B, T, 1)
                residual_hand = alpha_hand * cross_hand + (1 - alpha_hand) * self_hand
                residual_body = alpha_body * cross_body + (1 - alpha_body) * self_body
                hand = hand + self.hand_drop_path(residual_hand)
                body = body + self.body_drop_path(residual_body)

        else:
            # === Original Serial Architecture (backward compatible) ===
            ####################################### Layer 1: DeltaNet Cross-Attention ############################################
            residual_hand = self.hand_norm2(hand)
            residual_body = self.body_norm2(body)

            residual_hand = t2i_modulate(residual_hand, hand_shift_cross, hand_scale_cross)
            residual_body = t2i_modulate(residual_body, body_shift_cross, body_scale_cross)

            # DeltaNet cross-attention: current block queries historical state
            # Training: clean_hand/clean_body are historical clean frames
            # Inference: use kv_cache (DeltaNet state)
            if hand_kv_cache is not None:
                # Inference mode
                if update_kv_cache:
                    # Prefill: write clean frames to state
                    self.cross_attn(
                        motion=residual_hand, context=clean_hand, t_emb=t,
                        kv_t_emb=kwargs.get("kv_t_emb", None),
                        kv_cache=hand_kv_cache, current_start=current_start, prefill_only=True,
                    )
                    self.cross_attn2(
                        motion=residual_body, context=clean_body, t_emb=t,
                        kv_t_emb=kwargs.get("kv_t_emb", None),
                        kv_cache=body_kv_cache, current_start=current_start, prefill_only=True,
                    )
                    residual_hand = torch.zeros_like(residual_hand)
                    residual_body = torch.zeros_like(residual_body)
                else:
                    # Decode: query state with current noisy block
                    residual_hand = self.cross_attn(
                        motion=residual_hand, t_emb=t, kv_cache=hand_kv_cache,
                        current_start=current_start, history_cfg_alpha=kwargs.get("history_cfg_alpha", None),
                    )
                    residual_body = self.cross_attn2(
                        motion=residual_body, t_emb=t, kv_cache=body_kv_cache,
                        current_start=current_start, history_cfg_alpha=kwargs.get("history_cfg_alpha", None),
                    )
            else:
                # Training mode: use clean_hand/clean_body as context
                residual_hand = self.cross_attn(
                    motion=residual_hand, context=clean_hand, t_emb=t,
                    kv_t_emb=kwargs.get("kv_t_emb", None),
                    history_cfg_alpha=kwargs.get("history_cfg_alpha", None),
                )
                residual_body = self.cross_attn2(
                    motion=residual_body, context=clean_body, t_emb=t,
                    kv_t_emb=kwargs.get("kv_t_emb", None),
                    history_cfg_alpha=kwargs.get("history_cfg_alpha", None),
                )

            residual_hand = hand_gate_cross * residual_hand
            residual_body = body_gate_cross * residual_body

            hand = hand + self.hand_drop_path(residual_hand)
            body = body + self.body_drop_path(residual_body)

            ##################################### Layer 2: Block-wise Self-Attention #######################################
            residual_hand = self.hand_norm1(hand)
            residual_body = self.body_norm1(body)

            residual_hand = t2i_modulate(residual_hand, hand_shift_msa, hand_scale_msa)
            residual_body = t2i_modulate(residual_body, body_shift_msa, body_scale_msa)

            residual_hand = self.attn(residual_hand, rotary_emb=rotary_emb, current_start=current_start, is_inference=is_inference)
            residual_body = self.attn2(residual_body, rotary_emb=rotary_emb, current_start=current_start, is_inference=is_inference)

            residual_hand = hand_gate_msa * residual_hand
            residual_body = body_gate_msa * residual_body

            hand = hand + self.hand_drop_path(residual_hand)
            body = body + self.body_drop_path(residual_body)

        ######################################## Layer 3: FFN ########################################
        residual_hand = self.hand_norm3(hand)
        residual_body = self.body_norm3(body)

        residual_hand = t2i_modulate(residual_hand, hand_shift_mlp, hand_scale_mlp)
        residual_body = t2i_modulate(residual_body, body_shift_mlp, body_scale_mlp)
        
        residual_hand, residual_body = self.mlp(residual_hand, residual_body)
        
        residual_hand = hand_gate_mlp * residual_hand
        residual_body = body_gate_mlp * residual_body
        
        hand = hand + self.hand_drop_path(residual_hand)
        body = body + self.body_drop_path(residual_body)
        
        return hand, body


class CausalEmageBlockZeroDeltaNet(nn.Module):
    """
    CausalEmageBlock variant for ablation: keeps the same DeltaNet-facing interface,
    but uses a zero-output cross module in training/prefill/decode.

    Architecture is kept identical to CausalEmageBlockDeltaNet except Layer-2
    cross interaction always returns zeros.
    """

    def __init__(
        self,
        hidden_size,
        num_heads=8,
        head_dim=None,
        mapping_size=768,
        simple_modulation=False,
        bias=True,
        norm_type="rms",
        attn_type="flash",
        qk_norm_type="identity",
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
        num_frame_per_block=64,
        local_attn_size=-1,
        padding_unit_length=128,
        max_attention_size=512,
        sink_size=0,
        add_cross_attn_prefix=False,
        # Keep DeltaNet-compatible constructor parameters for drop-in compatibility
        deltanet_gate_low_rank_dim=16,
        deltanet_gate_logit_normalizer=16.0,
        deltanet_allow_neg_eigval=False,
        deltanet_use_short_conv=True,
        deltanet_conv_size=4,
        deltanet_use_triton_kernel=False,
        deltanet_scale=None,
        **kwargs,
    ):
        super().__init__()
        self.num_frame_per_block = num_frame_per_block

        # Normalization layers
        self.hand_norm1 = build_norm(norm_type, num_features=hidden_size, affine=False, eps=norm_eps)
        self.body_norm1 = build_norm(norm_type, num_features=hidden_size, affine=False, eps=norm_eps)
        self.hand_norm2 = build_norm(norm_type, num_features=hidden_size, affine=False, eps=norm_eps)
        self.body_norm2 = build_norm(norm_type, num_features=hidden_size, affine=False, eps=norm_eps)
        self.hand_norm3 = build_norm(norm_type, num_features=hidden_size, affine=False, eps=norm_eps)
        self.body_norm3 = build_norm(norm_type, num_features=hidden_size, affine=False, eps=norm_eps)

        # Layer-1: block-wise self-attention
        self.attn = BlockWiseAttention(
            dim=hidden_size,
            num_heads=num_heads,
            head_dim=head_dim,
            bias=bias,
            attn_type=attn_type,
            qk_norm_type=qk_norm_type,
            attn_drop=attn_drop,
            proj_drop=attn_proj_drop,
            block_size=num_frame_per_block,
        )
        self.attn2 = deepcopy(self.attn)

        # Layer-2: zero cross module (DeltaNet-compatible interface)
        self.cross_attn = ZeroGatedDeltaNetForBLD(
            dim=hidden_size,
            num_heads=num_heads,
            head_dim=head_dim,
            bias=bias,
            qk_norm=(qk_norm_type != "identity"),
            proj_drop=attn_proj_drop,
            gate_low_rank_dim=deltanet_gate_low_rank_dim,
            gate_logit_normalizer=deltanet_gate_logit_normalizer,
            block_size=num_frame_per_block,
            allow_neg_eigval=deltanet_allow_neg_eigval,
            use_short_conv=deltanet_use_short_conv,
            conv_size=deltanet_conv_size,
            use_triton_kernel=deltanet_use_triton_kernel,
            scale=deltanet_scale,
        )
        self.cross_attn2 = deepcopy(self.cross_attn)

        # Layer-3: FFN
        if ffn_type == "mlp":
            self.mlp = DoubleStreamMLP(
                in_features=hidden_size,
                hidden_features=int(hidden_size * ffn_mlp_ratio),
                act_type=ffn_act_type,
                proj_type=ffn_proj_type,
                bias=bias,
                dropout=ffn_drop,
            )
        elif ffn_type == "gated_mlp":
            self.mlp = DoubleStreamGatedMLP(
                in_features=hidden_size,
                hidden_features=int(hidden_size * ffn_mlp_ratio),
                act_type=ffn_act_type,
                proj_type=ffn_proj_type,
                bias=bias,
                dropout=ffn_drop,
            )
        else:
            raise NotImplementedError(f"ffn_type {ffn_type} is not implemented")

        self.simple_modulation = simple_modulation
        self.modulation_chunks = 9
        if self.simple_modulation:
            self.hand_scale_shift_table = nn.Parameter(torch.randn(self.modulation_chunks, hidden_size))
            self.body_scale_shift_table = nn.Parameter(torch.randn(self.modulation_chunks, hidden_size))
        else:
            self.hand_time_in = nn.Sequential(nn.SiLU(), nn.Linear(mapping_size, self.modulation_chunks * hidden_size, bias=bias))
            self.body_time_in = nn.Sequential(nn.SiLU(), nn.Linear(mapping_size, self.modulation_chunks * hidden_size, bias=bias))

        self.hand_drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.body_drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def initialize_weights(self):
        if self.simple_modulation:
            nn.init.constant_(self.hand_scale_shift_table, 0)
            nn.init.constant_(self.body_scale_shift_table, 0)
        else:
            nn.init.constant_(self.hand_time_in[1].weight, val=0)
            if self.hand_time_in[1].bias is not None:
                nn.init.constant_(self.hand_time_in[1].bias, val=0)
            nn.init.constant_(self.body_time_in[1].weight, val=0)
            if self.body_time_in[1].bias is not None:
                nn.init.constant_(self.body_time_in[1].bias, val=0)

    def _compute_modulation_params(self, t: torch.FloatTensor):
        if len(t.shape) not in (2, 3):
            raise ValueError(f"t.shape must be (B, T, D) or (B, D), but got {t.shape}")

        if self.simple_modulation:
            if len(t.shape) == 3:
                hand_chunks = (
                    self.hand_scale_shift_table[None, None, :, :] + t[:, :, None, :]
                ).unbind(dim=2)
                body_chunks = (
                    self.body_scale_shift_table[None, None, :, :] + t[:, :, None, :]
                ).unbind(dim=2)
            else:
                hand_chunks = (
                    self.hand_scale_shift_table[None, :, :] + t[:, None, :]
                ).unbind(dim=1)
                body_chunks = (
                    self.body_scale_shift_table[None, :, :] + t[:, None, :]
                ).unbind(dim=1)
        else:
            t_in = t if len(t.shape) == 3 else t.unsqueeze(1)
            hand_chunks = self.hand_time_in(t_in).chunk(self.modulation_chunks, dim=-1)
            body_chunks = self.body_time_in(t_in).chunk(self.modulation_chunks, dim=-1)

        hand_shift_msa, hand_scale_msa, hand_gate_msa, hand_shift_cross, hand_scale_cross, hand_gate_cross, hand_shift_mlp, hand_scale_mlp, hand_gate_mlp = hand_chunks
        body_shift_msa, body_scale_msa, body_gate_msa, body_shift_cross, body_scale_cross, body_gate_cross, body_shift_mlp, body_scale_mlp, body_gate_mlp = body_chunks

        return (
            hand_shift_msa, hand_scale_msa, hand_gate_msa,
            hand_shift_cross, hand_scale_cross, hand_gate_cross,
            hand_shift_mlp, hand_scale_mlp, hand_gate_mlp,
            body_shift_msa, body_scale_msa, body_gate_msa,
            body_shift_cross, body_scale_cross, body_gate_cross,
            body_shift_mlp, body_scale_mlp, body_gate_mlp,
        )

    def forward(
        self,
        hand: torch.FloatTensor,
        body: torch.FloatTensor,
        t: torch.FloatTensor,
        block_mask: torch.BoolTensor = None,
        clean_hand: torch.FloatTensor = None,
        clean_body: torch.FloatTensor = None,
        rotary_emb: torch.FloatTensor = None,
        use_liger_rope: bool = False,
        hand_kv_cache=None,
        body_kv_cache=None,
        current_start=0,
        update_kv_cache=False,
        is_inference=False,
        **kwargs,
    ):
        assert hand.shape[0] == body.shape[0], "batch size must be the same"

        (
            hand_shift_msa, hand_scale_msa, hand_gate_msa,
            hand_shift_cross, hand_scale_cross, hand_gate_cross,
            hand_shift_mlp, hand_scale_mlp, hand_gate_mlp,
            body_shift_msa, body_scale_msa, body_gate_msa,
            body_shift_cross, body_scale_cross, body_gate_cross,
            body_shift_mlp, body_scale_mlp, body_gate_mlp,
        ) = self._compute_modulation_params(t)

        # Layer 1: zero cross-attention module (train/prefill/decode all output zero)
        residual_hand = self.hand_norm2(hand)
        residual_body = self.body_norm2(body)

        residual_hand = t2i_modulate(residual_hand, hand_shift_cross, hand_scale_cross)
        residual_body = t2i_modulate(residual_body, body_shift_cross, body_scale_cross)

        if hand_kv_cache is not None:
            if update_kv_cache:
                self.cross_attn(
                    motion=residual_hand,
                    context=clean_hand,
                    kv_cache=hand_kv_cache,
                    current_start=current_start,
                    prefill_only=True,
                )
                self.cross_attn2(
                    motion=residual_body,
                    context=clean_body,
                    kv_cache=body_kv_cache,
                    current_start=current_start,
                    prefill_only=True,
                )
                residual_hand = torch.zeros_like(residual_hand)
                residual_body = torch.zeros_like(residual_body)
            else:
                residual_hand = self.cross_attn(
                    motion=residual_hand,
                    kv_cache=hand_kv_cache,
                    current_start=current_start,
                )
                residual_body = self.cross_attn2(
                    motion=residual_body,
                    kv_cache=body_kv_cache,
                    current_start=current_start,
                )
        else:
            residual_hand = self.cross_attn(
                motion=residual_hand,
                context=clean_hand,
            )
            residual_body = self.cross_attn2(
                motion=residual_body,
                context=clean_body,
            )

        residual_hand = hand_gate_cross * residual_hand
        residual_body = body_gate_cross * residual_body

        hand = hand + self.hand_drop_path(residual_hand)
        body = body + self.body_drop_path(residual_body)

        # Layer 2: block-wise self-attention
        residual_hand = self.hand_norm1(hand)
        residual_body = self.body_norm1(body)

        residual_hand = t2i_modulate(residual_hand, hand_shift_msa, hand_scale_msa)
        residual_body = t2i_modulate(residual_body, body_shift_msa, body_scale_msa)

        residual_hand = self.attn(residual_hand, rotary_emb=rotary_emb, current_start=current_start, is_inference=is_inference)
        residual_body = self.attn2(residual_body, rotary_emb=rotary_emb, current_start=current_start, is_inference=is_inference)

        residual_hand = hand_gate_msa * residual_hand
        residual_body = body_gate_msa * residual_body

        hand = hand + self.hand_drop_path(residual_hand)
        body = body + self.body_drop_path(residual_body)

        # Layer 3: FFN
        residual_hand = self.hand_norm3(hand)
        residual_body = self.body_norm3(body)

        residual_hand = t2i_modulate(residual_hand, hand_shift_mlp, hand_scale_mlp)
        residual_body = t2i_modulate(residual_body, body_shift_mlp, body_scale_mlp)

        residual_hand, residual_body = self.mlp(residual_hand, residual_body)

        residual_hand = hand_gate_mlp * residual_hand
        residual_body = body_gate_mlp * residual_body

        hand = hand + self.hand_drop_path(residual_hand)
        body = body + self.body_drop_path(residual_body)

        return hand, body


# Default config for DeltaNet block
DEFAULT_DELTANET_BLOCK_CONFIG = DictConfig(dict(
    target="diffusion.model.nets.causal.causal_emage.CausalEmageBlockDeltaNet",
    params=dict(
        hidden_size=None,
        num_heads=8,
        head_dim=None,
        mapping_size=768,
        simple_modulation=False,
        bias=True,
        norm_type="rms",
        attn_type="flash",
        qk_norm_type="identity",
        ffn_type="mlp",
        ffn_act_type="gelu",
        ffn_proj_type="linear",
        ffn_mlp_ratio=4.0,
        attn_drop=0.0,
        attn_proj_drop=0.0,
        ffn_drop=0.0,
        drop_path=0.0,
        norm_eps=1e-6,
        num_frame_per_block=64,
        is_teacher_forcing=False,
        local_attn_size=-1,
        padding_unit_length=128,
        max_attention_size=512,
        sink_size=0,
        deltanet_gate_low_rank_dim=16,
        deltanet_gate_logit_normalizer=16.0,
    )
))


class CausalEmage(nn.Module):
    def __init__(
        self,
        hand_in_channels=30 * 6,
        body_in_channels=(13 + 9 + 3) * 6,
        expression_in_channels=100,
        trans_in_channels=3,
        hidden_size=1152,
        caption_channels=2304,
        mapping_size=768,  # 新增：time embedding 的维度 (timestep_embed + style_embed)
        num_person=31,
        num_frame_per_block=30,
        depth=28,
        drop_path=0.0,
        inputlayer_config=DEFAULT_INPUT_CONFIG,
        fuse_inputlayer_config=DEFAULT_FUSE_INPUT_CONFIG,
        timestep_embedder_config=DEFAULT_TIMESTEP_EMBEDDER_CONFIG,
        caption_embedder_config=DEFAULT_CAPTION_EMBEDDER_CONFIG,
        block_config=DEFAULT_BLOCK_CONFIG,
        finallayer_config=DEFAULT_FINALLAYER_CONFIG,
        work_dir=None,
        use_pe=True,
        pos_embed_type="sincos",
        pe_max_length=10000,
        simple_modulation=False, # for pixart-like only one modulation
        time_condition_mode="raw",
        independent_first_frame=False,
        n_seed=0,
        add_cross_attn_prefix=False,
        zero_history_context_enabled: bool = False,
        zero_history_context_mode: str = "post_encoder",
        **kwargs,
    ):
        super().__init__()
        self.num_person = num_person
        self.hand_in_channels = hand_in_channels
        self.body_in_channels = body_in_channels
        self.trans_in_channels = trans_in_channels
        self.expression_in_channels = expression_in_channels
        self.block_mask = None
        self.independent_first_frame = independent_first_frame
        self.n_seed = n_seed
        self.local_attn_size = block_config["params"].get("local_attn_size", -1)
        self.num_frame_per_block = num_frame_per_block
        self.hidden_size = hidden_size
        self.depth = depth
        self.pos_embed_type = pos_embed_type
        self.time_condition_mode = time_condition_mode
        self.add_cross_attn_prefix = add_cross_attn_prefix
        self.padding_unit_length = block_config["params"].get("padding_unit_length", 128)
        self.zero_history_context_enabled = bool(zero_history_context_enabled)
        self.zero_history_context_mode = str(zero_history_context_mode).lower()
        if self.zero_history_context_mode not in ["post_encoder"]:
            raise ValueError(
                f"Unsupported zero_history_context_mode={self.zero_history_context_mode}. "
                "Expected one of ['post_encoder']."
            )

        if work_dir is not None:
            logger = get_root_logger(os.path.join(work_dir, "train_log.log"))
            logger = logger.info
        else:
            logger = print
        
        self.logger = logger
        inputlayer_config["params"]["hidden_size"] = hidden_size
        inputlayer_config["params"]["in_channels"] = hand_in_channels
        self.hand_in = instantiate_from_config(inputlayer_config)
        self.clean_hand_in = instantiate_from_config(inputlayer_config)
        inputlayer_config["params"]["in_channels"] = body_in_channels + trans_in_channels
        self.body_in = instantiate_from_config(inputlayer_config)
        self.clean_body_in = instantiate_from_config(inputlayer_config)

        inputlayer_config["params"]["in_channels"] = \
            self.hand_in_channels + self.body_in_channels + self.expression_in_channels + self.trans_in_channels
        self.xhand_in = instantiate_from_config(inputlayer_config)
        self.xbody_in = instantiate_from_config(inputlayer_config)
        
        fuse_inputlayer_config["params"]["hidden_size"] = hidden_size
        fuse_inputlayer_config["params"]["in_channels"] = self.hidden_size * 3
        # fuse t-modulation is controlled directly by fuse_inputlayer_config.params.enable_t_modulation
        # (InputLayer default is False). Only set mapping_size fallback when not explicitly provided.
        if "mapping_size" not in fuse_inputlayer_config["params"]:
            fuse_inputlayer_config["params"]["mapping_size"] = mapping_size
        self.hand_fuse = instantiate_from_config(fuse_inputlayer_config)
        self.body_fuse = instantiate_from_config(fuse_inputlayer_config)
        
        # timestep_embedder 输出维度 = mapping_size，与 SanaEmageNoTrans 保持一致
        self.mapping_size = mapping_size
        timestep_embedder_config["params"]["hidden_size"] = mapping_size
        self.motion_t_embedder = instantiate_from_config(timestep_embedder_config)
        self.simple_modulation = simple_modulation
        self.style_embed = StyleEmbedding(num_person, 64)
        
        self.use_pe = use_pe
        self.pe_max_length = pe_max_length
        # Will use fixed sin-cos embedding:
        if self.simple_modulation:
            # 使用 mapping_size 作为输入维度
            self.t_block = nn.Sequential(nn.SiLU(), nn.Linear(mapping_size, hidden_size, bias=True))

        caption_embedder_config["params"]["in_channels"] = caption_channels
        caption_embedder_config["params"]["hidden_size"] = hidden_size
        self.context_embedder = instantiate_from_config(caption_embedder_config)
        
        drop_path = [x.item() for x in torch.linspace(0, drop_path, depth)]  # stochastic depth decay rule
        block_config["params"]["simple_modulation"] = simple_modulation
        block_config["params"]["hidden_size"] = hidden_size
        block_config["params"]["mapping_size"] = mapping_size  # 传递 mapping_size 给 block
        block_config["params"]["num_frame_per_block"] = num_frame_per_block
        block_config["params"]["add_cross_attn_prefix"] = add_cross_attn_prefix
        self.blocks = nn.ModuleList()
        for i in range(depth):
            block_config["params"]["drop_path"] = drop_path[i]
            self.blocks.append(instantiate_from_config(block_config))
        
        finallayer_config["params"]["hidden_size"] = hidden_size
        finallayer_config["params"]["mapping_size"] = mapping_size  # 添加 mapping_size
        finallayer_config["params"]["out_channels"] = hand_in_channels
        self.hand_final_layer = instantiate_from_config(finallayer_config)
        finallayer_config["params"]["out_channels"] = body_in_channels + trans_in_channels
        self.body_final_layer = instantiate_from_config(finallayer_config)
        
        if self.use_pe:
            if self.pos_embed_type == "sincos":
                # Initialize (and freeze) pos_embed by sin-cos embedding:
                self.register_buffer("pos_embed", torch.zeros(1, pe_max_length, hidden_size))
                pos_grid = np.arange(pe_max_length, dtype=np.float64)
                pos_embed = get_1d_sincos_pos_embed_from_grid(
                    self.pos_embed.shape[-1],
                    pos_grid,
                )
                self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
            elif self.pos_embed_type in ["rope", "liger-rope", "block-local-rope", "block-local-liger-rope"]:
                self.pos_embed = RopePosEmbed(theta=10000, axes_dim=[hidden_size // self.blocks[0].attn.num_heads])
            elif self.pos_embed_type == "none":
                pass
            else:
                raise NotImplementedError(f"pos embed type {pos_embed_type} is not supported.")
        self.initialize_weights()

        # Determine if flex_attention is usable (requires head_dim to be a power of 2)
        _head_dim = hidden_size // self.blocks[0].attn.num_heads
        self._use_flex_crossattn = (_head_dim & (_head_dim - 1) == 0) and _head_dim >= 16

        # Lazy loading cache for indices (to avoid repeated numpy array access and conversion)
        self._indices_cache = {
            "hand_indices": None,
            "upper_indices": None,
            "lower_indices": None,
            "face_joint_indices": None,
            "trans_indices": None,
            "body_indices": None,  # concatenated indices
        }

        if get_rank() == 0:
            logger(
                f"use pe: {use_pe}, pos embed type: {pos_embed_type}"
            )
            attn_type = block_config["params"]["attn_type"]
            cross_attn_impl_type = block_config["params"].get("cross_attn_impl_type", attn_type)
            ffn_type = block_config["params"]["ffn_type"]
            qk_norm_type = block_config["params"]["qk_norm_type"]
            logger(
                f"attention type: {attn_type}; cross-attn impl type: {cross_attn_impl_type}; "
                f"ffn type: {ffn_type}; self-attn qk norm: {qk_norm_type}; "
            )
            if self.zero_history_context_enabled:
                logger(
                    f"zero_history_context enabled, mode={self.zero_history_context_mode}"
                )

    def _apply_time_condition_mode(self, t: torch.Tensor) -> torch.Tensor:
        """
        Controls whether timestep conditioning keeps temporal variation.

        - raw: keep original t
        - mean / temporal_mean: average on time dim and broadcast back to all frames
        """
        if self.time_condition_mode == "raw":
            return t
        if self.time_condition_mode in ["mean", "temporal_mean"]:
            if t.dim() == 3:
                t_mean = t.mean(dim=1, keepdim=True)
                return t_mean.expand(-1, t.shape[1], -1)
            if t.dim() == 2:
                return t
            raise ValueError(f"Unsupported t shape for time_condition_mode={self.time_condition_mode}: {t.shape}")
        raise ValueError(
            f"Unsupported time_condition_mode: {self.time_condition_mode}. "
            "Expected one of ['raw', 'mean', 'temporal_mean']."
        )

    def _is_rope_pos_embed(self) -> bool:
        return self.pos_embed_type in ["rope", "liger-rope", "block-local-rope", "block-local-liger-rope"]

    def _is_block_local_rope(self) -> bool:
        return self.pos_embed_type in ["block-local-rope", "block-local-liger-rope"]

    def _use_liger_rope_kernel(self) -> bool:
        return self.pos_embed_type in ["liger-rope", "block-local-liger-rope"]

    def _get_indices(self, kwargs, device):
        """
        Lazy loading and caching of indices from kwargs.
        Converts numpy arrays to torch tensors on first access and caches them.

        Args:
            kwargs: Dictionary containing the indices as numpy arrays
            device: Target device for the tensors

        Returns:
            tuple: (hand_indices, body_indices) as torch tensors
        """
        # Check if we need to initialize the cache
        if self._indices_cache["hand_indices"] is None:
            # First time: load from kwargs and convert to tensors
            hand_indices_np = kwargs["hand_indices"]
            upper_indices_np = kwargs["upper_indices"]
            lower_indices_np = kwargs["lower_indices"]
            face_joint_indices_np = kwargs["face_joint_indices"]
            trans_indices_np = kwargs["trans_indices"]

            # Convert to torch tensors and cache
            self._indices_cache["hand_indices"] = torch.from_numpy(hand_indices_np).to(device)
            self._indices_cache["upper_indices"] = torch.from_numpy(upper_indices_np).to(device)
            self._indices_cache["lower_indices"] = torch.from_numpy(lower_indices_np).to(device)
            self._indices_cache["face_joint_indices"] = torch.from_numpy(face_joint_indices_np).to(device)
            self._indices_cache["trans_indices"] = torch.from_numpy(trans_indices_np).to(device)

            # Concatenate body indices
            # IMPORTANT: keep consistent with SanaEmageNoTrans
            # Order: upper, lower, trans, face
            self._indices_cache["body_indices"] = torch.cat([
                self._indices_cache["upper_indices"],
                self._indices_cache["lower_indices"],
                self._indices_cache["trans_indices"],
                self._indices_cache["face_joint_indices"]
            ], dim=0)

        return self._indices_cache["hand_indices"], self._indices_cache["body_indices"]


    def forward_train(
        self,  
        motion,
        timestep,
        context,
        expression,
        clean_motion,
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
        clean_motion = clean_motion.to(self.dtype)
        expression = expression.to(self.dtype)
        
        device = motion.device
        # Construct blockwise causal attn mask (skip for DeltaNet cross-attn)
        if not isinstance(self.blocks[0], (CausalEmageBlockDeltaNet, CausalEmageBlockZeroDeltaNet)):
            if self.block_mask is None:
                self.block_mask = self._prepare_blockwise_causal_cross_attn_mask(
                    device, num_frames=motion.shape[1],
                    frame_seqlen=1,
                    num_frame_per_block=self.num_frame_per_block,
                    seed_num_frames=self.n_seed,
                    local_attn_size=self.local_attn_size,
                    unit_length=self.padding_unit_length,
                    add_learnable_prefix=self.add_cross_attn_prefix,
                    debug=not self._use_flex_crossattn,  # dense mask when flex unavailable
                )
            else:
                assert self.block_mask.shape[-1] == motion.shape[1], \
                    f"block_mask.shape: {self.block_mask.shape} != motion.shape: {motion.shape}, " \
                        "the seq_len of block_mask and motion must be the same"
        else:
            self.block_mask = None

        id = kwargs["speaker_id"]

        # Use lazy loading for indices (cached after first call)
        hand_indices, body_indices = self._get_indices(kwargs, device)

        #########################################################
        # for expression
        #########################################################

        hand = motion[..., hand_indices]
        body = motion[..., body_indices]
        
        hand = self.hand_in(hand)
        body = self.body_in(body)

        # t = timestep / self.training_step # 归一化
        t = self.motion_t_embedder(timestep)  # (N, L, D)
        one_hots = F.one_hot(id, num_classes=self.num_person).float().to(device=t.device, dtype=t.dtype)
        id_embed = self.style_embed(one_hots)
        if t.dim() == 3:
            id_embed = id_embed[:, None, :].expand(-1, t.size(1), -1)
        t = torch.cat([t, id_embed], dim=-1)
        t = self._apply_time_condition_mode(t)
        if self.simple_modulation:
            t0 = self.t_block(t)
        else:
            t0 = t
        
        # Generate kv_t_emb for QK-Alignment if KV modulation is enabled
        kv_t0 = None
        if hasattr(self.blocks[0].cross_attn, 'kv_t_modulation_mode') and \
           self.blocks[0].cross_attn.kv_t_modulation_mode == "film":
                kv_timestep = torch.zeros_like(timestep) + 1e-4  # Very small timestep for clean history
                kv_t = self.motion_t_embedder(kv_timestep)
                if kv_t.dim() == 3:
                    kv_id_embed = id_embed  # Already expanded above
                else:
                    kv_id_embed = id_embed
                kv_t = torch.cat([kv_t, kv_id_embed], dim=-1)
                kv_t = self._apply_time_condition_mode(kv_t)
                if self.simple_modulation:
                    kv_t0 = self.t_block(kv_t)
                else:
                    kv_t0 = kv_t

        x = torch.cat([motion, expression], dim=-1)
        xhand = self.xhand_in(x) 
        xbody = self.xbody_in(x)
        
        context = self.context_embedder(context)  # (N, L, D)
        hand = self.hand_fuse(torch.cat([hand, xhand, context], dim=-1), t=t)
        body = self.body_fuse(torch.cat([body, xbody, context], dim=-1), t=t)
        
        clean_hand, clean_body = self._encode_clean_tokens_for_kv(
            clean_motion=clean_motion,
            hand_indices=hand_indices,
            body_indices=body_indices,
        )
        clean_hand, clean_body = self._maybe_zero_history_context(clean_hand, clean_body)
        # noisy tokens：sincos 也统一
        hand = self._add_sincos_pos(hand, current_start=0)
        body = self._add_sincos_pos(body, current_start=0)
        clean_hand = self._add_sincos_pos(clean_hand, current_start=0)
        clean_body = self._add_sincos_pos(clean_body, current_start=0)

        # rope：统一构造 rotary_emb
        rotary_emb = None
        use_liger_rope = self._use_liger_rope_kernel()
        if self.use_pe and self._is_rope_pos_embed():
            rotary_emb = self._build_rope_emb(
                batch=hand.shape[0],
                rope_len=hand.shape[1],   # train 是从 0 开始
                device=hand.device,
            )

        # 统一生成全局历史 CFG alpha (保证身/首联合，且层间一致)
        history_cfg_alpha = None
        if hasattr(self.blocks[0], 'cross_attn') and hasattr(self.blocks[0].cross_attn, '_generate_history_cfg_alpha'):
            num_chunks = (hand.shape[1] + self.num_frame_per_block - 1) // self.num_frame_per_block
            history_cfg_alpha = self.blocks[0].cross_attn._generate_history_cfg_alpha(
                batch_size=hand.shape[0],
                num_chunks=num_chunks,
                device=hand.device,
            )

        kwargs = dict(
            t = t0,
            kv_t_emb = kv_t0,
            block_mask = self.block_mask,
            rotary_emb = rotary_emb,
            use_liger_rope = use_liger_rope,
            hand_kv_cache = None,
            body_kv_cache = None,
            current_start = 0,
            is_inference = False,
            history_cfg_alpha = history_cfg_alpha,
        )
        if torch.is_grad_enabled():
            for block in self.blocks:    
                hand, body = auto_grad_checkpoint(block, hand=hand, body=body, clean_hand=clean_hand, clean_body=clean_body, **kwargs)
        else:
            for block in self.blocks:  
                hand, body = block(hand=hand, body=body, clean_hand=clean_hand, clean_body=clean_body, **kwargs)
        
        hand = self.hand_final_layer(hand, t)
        body = self.body_final_layer(body, t)
        
        output_motion = torch.zeros_like(motion)
        output_motion[..., hand_indices] = hand.to(motion.dtype)
        output_motion[..., body_indices] = body.to(motion.dtype)
        return {
            "motion": output_motion, 
        }

    def _build_rope_emb(self, batch: int, rope_len: int, device: torch.device):
        if not (self.use_pe and self._is_rope_pos_embed()):
            return None
        assert rope_len > 0
        # Position ids must not use low-precision model dtype (fp16/bf16),
        # otherwise long-context RoPE may suffer from position aliasing.
        if self._is_block_local_rope():
            # Block-local relative positions: each block reuses [0, block_size-1].
            # This breaks absolute-position leakage while preserving in-block order.
            seq_ids = (torch.arange(rope_len, device=device, dtype=torch.float32) % self.num_frame_per_block).unsqueeze(-1)
        else:
            seq_ids = self.pos_embed._prepare_latent_sequence_ids(batch, rope_len, device, torch.float32)
        rotary_emb = self.pos_embed(seq_ids)
        return rotary_emb

    def _add_sincos_pos(self, x: torch.Tensor, current_start: int):
        if not (self.use_pe and self.pos_embed_type == "sincos"):
            return x
        start = 0 # 规避 current_start，永远从 0 开始，这是防止sincos下没见过过长的位置编码
        end = x.shape[1]
        assert end <= self.pos_embed.shape[1], f"pe_max_length too small: {end} > {self.pos_embed.shape[1]}"
        pe = self.pos_embed[:, start:end, :].to(device=x.device, dtype=x.dtype)
        return x + pe

    def _encode_clean_tokens_for_kv(
        self,
        clean_motion: torch.Tensor,
        hand_indices,
        body_indices,
    ):
        clean_motion = clean_motion.to(self.dtype)
        clean_hand = self.clean_hand_in(clean_motion[..., hand_indices])
        clean_body = self.clean_body_in(clean_motion[..., body_indices])
        return clean_hand, clean_body

    def _maybe_zero_history_context(
        self,
        clean_hand: torch.Tensor,
        clean_body: torch.Tensor,
    ):
        if not self.zero_history_context_enabled:
            return clean_hand, clean_body

        if self.zero_history_context_mode == "post_encoder":
            # Keep graph connectivity for clean_* encoders to avoid DDP unused-parameter
            # reduction errors while still forcing history context values to zero.
            return clean_hand * 0.0, clean_body * 0.0

        raise ValueError(
            f"Unsupported zero_history_context_mode={self.zero_history_context_mode}. "
            "Expected one of ['post_encoder']."
        )

    def forward_inference(
        self,
        motion,
        timestep,
        context,
        expression,
        kv_cache: dict = None,
        seq_len: int = None,
        current_start: int = 0,
        prefill_only: bool = False,
        update_kv_cache: bool = False,
        clean_motion: torch.Tensor = None,   # 仅 prefill_only=True 时需要
        **kwargs
    ):
        """
        两种模式:
        (1) prefill_only=True：把 clean_motion 对应 chunk 写入每层 kv_cache（只写不算输出）
        (2) prefill_only=False：正常 decode，用 kv_cache 读历史 K/V，算当前 noisy chunk 输出
        """
        assert kv_cache is not None

        # -------------------------
        # 0) speaker / indices - Use lazy loading
        # -------------------------
        id = kwargs["speaker_id"]
        device = motion.device if motion is not None else (clean_motion.device if clean_motion is not None else "cuda")
        hand_indices, body_indices = self._get_indices(kwargs, device)

        # -------------------------
        # 1) PREFILL / COMMIT: only write KV cache
        # -------------------------
        if prefill_only:
            assert update_kv_cache is True
            assert clean_motion is not None

            clean_hand, clean_body = self._encode_clean_tokens_for_kv(
                clean_motion=clean_motion,
                hand_indices=hand_indices,
                body_indices=body_indices,
            )
            clean_hand, clean_body = self._maybe_zero_history_context(clean_hand, clean_body)
            clean_hand = self._add_sincos_pos(clean_hand, current_start)
            clean_body = self._add_sincos_pos(clean_body, current_start)

            # Build kv_t_emb for QK-Alignment prefill only when KV FiLM is enabled
            kv_t0 = None
            if hasattr(self.blocks[0].cross_attn, 'kv_t_modulation_mode') and \
               self.blocks[0].cross_attn.kv_t_modulation_mode == "film":
                    kv_timestep = torch.zeros_like(timestep) + 1e-4
                    kv_t = self.motion_t_embedder(kv_timestep)
                    kv_one_hots = F.one_hot(id, num_classes=self.num_person).float().to(device=kv_t.device, dtype=kv_t.dtype)
                    kv_id_embed = self.style_embed(kv_one_hots)
                    if kv_t.dim() == 3:
                        kv_id_embed = kv_id_embed[:, None, :].expand(-1, kv_t.size(1), -1)
                    kv_t = torch.cat([kv_t, kv_id_embed], dim=-1)
                    kv_t = self._apply_time_condition_mode(kv_t)
                    kv_t0 = self.t_block(kv_t) if self.simple_modulation else kv_t

            rotary_emb = None
            use_liger_rope = self._use_liger_rope_kernel()
            if self.use_pe and self._is_rope_pos_embed():
                rope_len = seq_len if seq_len is not None else (current_start + clean_hand.shape[1])
                rotary_emb = self._build_rope_emb(
                    batch=clean_hand.shape[0],
                    rope_len=rope_len,
                    device=clean_hand.device,
                )

            qh = torch.zeros_like(clean_hand)  # (B, T, C) 确保 q_len == context_len
            qb = torch.zeros_like(clean_body)
            # 逐层写 cache（不走 block.forward，避免额外计算）
            for block_index, block in enumerate(self.blocks):
                if isinstance(block, (CausalEmageBlockDeltaNet, CausalEmageBlockZeroDeltaNet)):
                    # hand cache (DeltaNet)
                    block.cross_attn(
                        motion=qh,      # dummy，不会被用到
                        context=clean_hand,
                        kv_t_emb=kv_t0,
                        kv_cache=kv_cache["hand_kv_cache"][block_index],
                        current_start=current_start,
                        prefill_only=True,
                    )
                    # body cache (DeltaNet)
                    block.cross_attn2(
                        motion=qb,      # dummy
                        context=clean_body,
                        kv_t_emb=kv_t0,
                        kv_cache=kv_cache["body_kv_cache"][block_index],
                        current_start=current_start,
                        prefill_only=True,
                    )
                else:
                    # hand cache (standard cross-attn)
                    block.cross_attn(
                        motion=qh,      # dummy，不会被用到
                        context=clean_hand,
                        mask=None,
                        rotary_emb=rotary_emb,
                        use_liger_rope=use_liger_rope,
                        kv_cache=kv_cache["hand_kv_cache"][block_index],
                        current_start=current_start,
                        update_kv_cache=True,
                        prefill_only=True,
                    )
                    # body cache (standard cross-attn)
                    block.cross_attn2(
                        motion=qb,      # dummy
                        context=clean_body,
                        mask=None,
                        rotary_emb=rotary_emb,
                        use_liger_rope=use_liger_rope,
                        kv_cache=kv_cache["body_kv_cache"][block_index],
                        current_start=current_start,
                        update_kv_cache=True,
                        prefill_only=True,
                    )

            # prefill 不产生输出
            return {
                "motion": None,
            }
        # -------------------------
        # 2) DECODE: normal inference
        # -------------------------
        motion = motion.to(self.dtype)
        expression = expression.to(self.dtype)
        
        # (a) motion -> hand/body embed + fuse（与你原代码一致）
        hand = motion[..., hand_indices]
        body = motion[..., body_indices]
        hand = self.hand_in(hand)
        body = self.body_in(body)

        # (b) timestep + style
        t = self.motion_t_embedder(timestep)
        one_hots = F.one_hot(id, num_classes=self.num_person).float().to(device=t.device, dtype=t.dtype)
        id_embed = self.style_embed(one_hots)
        if t.dim() == 3:
            id_embed = id_embed[:, None, :].expand(-1, t.size(1), -1)
        t = torch.cat([t, id_embed], dim=-1)
        t = self._apply_time_condition_mode(t)
        t0 = self.t_block(t) if self.simple_modulation else t
        
        # Generate kv_t_emb for QK-Alignment if KV modulation is enabled
        kv_t0 = None
        if hasattr(self.blocks[0].cross_attn, 'kv_t_modulation_mode') and \
           self.blocks[0].cross_attn.kv_t_modulation_mode == "film":
                kv_timestep = torch.zeros_like(timestep) + 1e-4  # Very small timestep for clean history
                kv_t = self.motion_t_embedder(kv_timestep)
                if kv_t.dim() == 3:
                    kv_id_embed = id_embed  # Already expanded above
                else:
                    kv_id_embed = id_embed
                kv_t = torch.cat([kv_t, kv_id_embed], dim=-1)
                kv_t = self._apply_time_condition_mode(kv_t)
                if self.simple_modulation:
                    kv_t0 = self.t_block(kv_t)
                else:
                    kv_t0 = kv_t

        x = torch.cat([motion, expression], dim=-1)
        xhand = self.xhand_in(x)
        xbody = self.xbody_in(x)

        context = self.context_embedder(context)
        hand = self.hand_fuse(torch.cat([hand, xhand, context], dim=-1), t=t)
        body = self.body_fuse(torch.cat([body, xbody, context], dim=-1), t=t)

        hand = self._add_sincos_pos(hand, current_start)
        body = self._add_sincos_pos(body, current_start)
        rotary_emb = None
        use_liger_rope = self._use_liger_rope_kernel()
        if self.use_pe and self._is_rope_pos_embed():
            rope_len = seq_len if seq_len is not None else (current_start + hand.shape[1])
            rotary_emb = self._build_rope_emb(
                batch=hand.shape[0],
                rope_len=rope_len,   # RoPE 表长度必须覆盖到 current_start + chunk_len
                device=hand.device,
            )

        # (d) run blocks（decode：update_kv_cache=False）
        block_kwargs = dict(
            t=t0,
            kv_t_emb=kv_t0,
            block_mask=None,
            rotary_emb=rotary_emb,
            use_liger_rope=use_liger_rope,
            clean_hand=None,
            clean_body=None,
            update_kv_cache=False,
            current_start=current_start,
            is_inference=True,
            history_cfg_alpha=kwargs.get("history_cfg_alpha", None),
        )

        for block_index, block in enumerate(self.blocks):
            block_kwargs.update({
                "hand_kv_cache": kv_cache["hand_kv_cache"][block_index],
                "body_kv_cache": kv_cache["body_kv_cache"][block_index],
            })
            if torch.is_grad_enabled():
                hand, body = auto_grad_checkpoint(block, hand=hand, body=body, **block_kwargs)
            else:
                hand, body = block(hand=hand, body=body, **block_kwargs)

        # (e) final layer
        hand = self.hand_final_layer(hand, t)
        body = self.body_final_layer(body, t)

        output_motion = torch.zeros_like(motion)
        output_motion[..., hand_indices] = hand.to(motion.dtype)
        output_motion[..., body_indices] = body.to(motion.dtype)
        return {"motion": output_motion}


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

        if self.simple_modulation:
            nn.init.constant_(self.t_block[1].weight, val=0)
            if self.t_block[1].bias is not None:
                nn.init.constant_(self.t_block[1].bias, val=0)
    

    @property
    def dtype(self):
        return next(self.parameters()).dtype


    @staticmethod
    def _prepare_blockwise_causal_cross_attn_mask(
        device: torch.device | str, num_frames: int = 21,
        frame_seqlen: int = 1560, num_frame_per_block=4, local_attn_size=-1,
        unit_length: int = 128, seed_num_frames: int = 8, debug=False,
        add_learnable_prefix: bool = False,
    ) -> BlockMask:
        """
        we will divide the token sequence into the following format
        [1 latent frame] [N latent frame] ... [N latent frame]
        The first frame is separated out to support I2V generation
        We use flexattention to construct the attention mask
        If add_learnable_prefix is True, an extra KV token is inserted at position 0 and is visible to every query.
        """
        prefix_len = 1 if add_learnable_prefix else 0
        q_length = num_frames * frame_seqlen
        k_length = q_length + prefix_len

        # we do right padding to get to a multiple of unit_length
        q_padded = math.ceil(q_length / unit_length) * unit_length - q_length
        k_padded = math.ceil(k_length / unit_length) * unit_length - k_length

        ends = torch.zeros(q_length + q_padded,
                           device=device, dtype=torch.long)

        # special handling for the first frame
        first_block_len = frame_seqlen * num_frame_per_block
        ends[:first_block_len] = prefix_len + frame_seqlen * seed_num_frames

        # Block-wise causal mask will attend to all elements that are before the end of the current chunk
        frame_indices = torch.arange(
            start=frame_seqlen * num_frame_per_block,
            end=q_length + q_padded,
            step=frame_seqlen * num_frame_per_block,
            device=device
        )

        for idx, tmp in enumerate(frame_indices):
            ends[tmp:tmp + frame_seqlen * num_frame_per_block] = prefix_len + tmp

        def attention_mask(b, h, q_idx, kv_idx):
            if local_attn_size == -1:
                return (kv_idx < ends[q_idx])
            else:
                return ((kv_idx < ends[q_idx]) & (kv_idx >= (ends[q_idx] - local_attn_size * frame_seqlen)))

        if not debug:
            block_mask = create_block_mask(attention_mask, B=None, H=None, Q_LEN=q_length + q_padded,
                                        KV_LEN=k_length + k_padded, _compile=False, device=device)
        else:
            block_mask = create_mask(attention_mask, B=None, H=None, Q_LEN=q_length + q_padded,
                                        KV_LEN=k_length + k_padded, device=device)

        return block_mask

if __name__ == "__main__":
    import imageio
    import cv2
    from calflops import calculate_flops
    
    ## for blockwise causal cross attention mask
    block_mask = CausalEmage._prepare_blockwise_causal_cross_attn_mask(
        num_frames = 256,
        frame_seqlen = 1,
        num_frame_per_block = 64,
        seed_num_frames = 0,
        local_attn_size = 64,
        add_learnable_prefix = False,
        device=torch.device("cuda"), 
        debug=True
    )
    print(block_mask)    
    block_mask = cv2.resize(block_mask[0, 0].cpu().float().numpy(), (1024, 1024))
    imageio.imwrite("blockwise_causal_cross_attn_mask_%d.jpg" % (0), np.uint8(255. * block_mask))
    
    # ==================== 计算模型参数量、计算量、MACs ====================
    print("\n" + "="*60)
    print("计算 CausalEmage 模型的参数量、计算量、MACs")
    print("="*60)
    
    # 模型配置
    hand_in_channels = 30 * 6  # 180
    body_in_channels = (13 + 9 + 3) * 6  # 150
    expression_in_channels = 100
    trans_in_channels = 3
    hidden_size = 384
    caption_channels = 1024
    mapping_size = 768  # timestep_embed + style_embed 的总维度
    depth = 6
    num_frame_per_block = 64
    
    # 创建模型
    model = CausalEmage(
        expression_in_channels=100,
        hidden_size=384,
        caption_channels=1024,
        mapping_size=768,  # 新增
        depth=6,
        drop_path=0.0,
        is_teacher_forcing=True,
        inputlayer_config=DictConfig(dict(
            target="diffusion.model.nets.basic_modules.InputLayer",
            params=dict(
                bias=True,
                norm_type="rms",
                proj_type="linear",
                norm_eps=1e-6,
            )
        )),
        timestep_embedder_config=DictConfig(dict(
            target="diffusion.model.nets.basic_modules.MPFourierTimeEmbedding",
            params=dict(
                num_embedding_channels=768,
                bias=True,
            )
        )),
        caption_embedder_config=DictConfig(dict(
            target="diffusion.model.nets.basic_modules.CaptionEmbedder",
            params=dict(
                y_proj_type="mlp",
                act_type="gelu",
                y_norm_type="rms",
                bias=True,
                dropout=0.0,
                norm_eps=1e-6,
            )
        )),
        block_config=DictConfig(dict(
            target="diffusion.model.nets.causal.causal_emage.CausalEmageBlock",
            params=dict(
                num_heads=12,
                head_dim=None,
                mapping_size=768,  # 新增
                simple_modulation=False,
                bias=True,
                norm_type="rms",
                attn_type="flash",
                qk_norm_type="rms",
                ffn_type="mlp",
                ffn_act_type="gelu",
                ffn_proj_type="linear",
                ffn_mlp_ratio=4.0,
                attn_drop=0.0,
                attn_proj_drop=0.0,
                ffn_drop=0.0,
                drop_path=0.0,
                norm_eps=1e-6,
                is_teacher_forcing=True,
                local_attn_size=-1,
                max_attention_size=512,
                sink_size=0,
            )
        )),
        finallayer_config=DictConfig(dict(
            target="diffusion.model.nets.causal.causal_block.FinalLayer",
            params=dict(
                mapping_size=768,  # 新增
                bias=True,
                norm_type="rms",
                proj_type="linear",
                norm_eps=1e-6,
            )
        )),
        use_pe=True,
        pos_embed_type="rope",
        pe_max_length=10000,
        simple_modulation=True,
        num_frame_per_block=64,
        independent_first_frame=False,
    ).cuda()
    
    # 构造输入维度
    batch_size = 1
    seq_len = 256  # 序列长度
    motion_dim = hand_in_channels + body_in_channels + trans_in_channels  # 333
    
    # 定义索引
    hand_indices = np.arange(0, hand_in_channels)
    upper_indices = np.arange(hand_in_channels, hand_in_channels + 13 * 6)
    lower_indices = np.arange(hand_in_channels + 13 * 6, hand_in_channels + (13 + 9) * 6)
    face_joint_indices = np.arange(hand_in_channels + (13 + 9) * 6, hand_in_channels + (13 + 9 + 3) * 6)
    trans_indices = np.arange(hand_in_channels + body_in_channels, motion_dim)
    
    # 包装 forward 函数以适配 calflops
    class ModelWrapper(nn.Module):
        def __init__(self, model, kwargs):
            super().__init__()
            self.model = model
            self.kwargs = kwargs
            
        def forward(self, motion, timestep, context, expression, clean_motion):
            return self.model(
                motion=motion,
                timestep=timestep,
                context=context,
                expression=expression,
                clean_motion=clean_motion,
                **self.kwargs
            )
    
    kwargs = {
        "speaker_id": torch.zeros(batch_size, dtype=torch.long).cuda(),
        "hand_indices": hand_indices,
        "upper_indices": upper_indices,
        "lower_indices": lower_indices,
        "face_joint_indices": face_joint_indices,
        "trans_indices": trans_indices,
    }
    
    wrapped_model = ModelWrapper(model, kwargs).cuda()
    wrapped_model.eval()
    
    print(f"\n模型配置:")
    print(f"  - hidden_size: {hidden_size}")
    print(f"  - depth: {depth}")
    print(f"  - motion_dim: {motion_dim}")
    print(f"  - caption_channels: {caption_channels}")
    print(f"  - expression_in_channels: {expression_in_channels}")
    print(f"\n输入形状 (batch={batch_size}, seq_len={seq_len}):")
    print(f"  - motion: ({batch_size}, {seq_len}, {motion_dim})")
    print(f"  - timestep: ({batch_size}, {seq_len})")
    print(f"  - context: ({batch_size}, {seq_len}, {caption_channels})")
    print(f"  - expression: ({batch_size}, {seq_len}, {expression_in_channels})")
    print(f"  - clean_motion: ({batch_size}, {seq_len}, {motion_dim})")
    
    # 创建实际的输入张量
    dummy_motion = torch.randn(batch_size, seq_len, motion_dim).cuda()
    dummy_timestep = torch.randint(0, 1000, (batch_size, seq_len)).cuda()
    dummy_context = torch.randn(batch_size, seq_len, caption_channels).cuda()
    dummy_expression = torch.randn(batch_size, seq_len, expression_in_channels).cuda()
    dummy_clean_motion = torch.randn(batch_size, seq_len, motion_dim).cuda()
    
    # 使用 calflops 计算
    with torch.no_grad():
        flops, macs, params = calculate_flops(
            model=wrapped_model,
            args=[dummy_motion, dummy_timestep, dummy_context, dummy_expression, dummy_clean_motion],
            output_as_string=True,
            output_precision=4,
            print_results=True,
            print_detailed=True,
        )
    
    print("\n" + "="*60)
    print("计算结果汇总")
    print("="*60)
    print(f"  参数量(Params): {params}")
    print(f"  MACs: {macs}")
    print(f"  FLOPs: {flops}")
    print("="*60)


    # ==================== 测试 Liger Kernel RoPE vs 原始 RoPE ====================
    print("\n\n" + "="*80)
    print("测试 Liger Kernel RoPE vs 原始 RoPE 的输出一致性")
    print("="*80)

    from diffusion.model.nets.positional_embeding import LIGER_ROPE_AVAILABLE
    print(f"\nLIGER_ROPE_AVAILABLE: {LIGER_ROPE_AVAILABLE}")

    if not LIGER_ROPE_AVAILABLE:
        print("Liger kernel 不可用，跳过测试")
    else:
        # 测试配置
        test_batch_size = 2
        test_seq_len = 128
        test_hidden_size = 256
        test_depth = 2
        test_num_heads = 8

        print(f"\n测试配置:")
        print(f"  batch_size: {test_batch_size}")
        print(f"  seq_len: {test_seq_len}")
        print(f"  hidden_size: {test_hidden_size}")
        print(f"  depth: {test_depth}")
        print(f"  num_heads: {test_num_heads}")

        # 创建两个模型：一个使用原始 rope，一个使用 liger-rope
        def create_test_model(pos_embed_type):
            return CausalEmage(
                expression_in_channels=100,
                hidden_size=test_hidden_size,
                caption_channels=512,
                mapping_size=512,
                depth=test_depth,
                drop_path=0.0,
                is_teacher_forcing=True,
                inputlayer_config=DictConfig(dict(
                    target="diffusion.model.nets.basic_modules.InputLayer",
                    params=dict(
                        bias=True,
                        norm_type="rms",
                        proj_type="linear",
                        norm_eps=1e-6,
                    )
                )),
                timestep_embedder_config=DictConfig(dict(
                    target="diffusion.model.nets.basic_modules.MPFourierTimeEmbedding",
                    params=dict(
                        num_embedding_channels=512,
                        bias=True,
                    )
                )),
                caption_embedder_config=DictConfig(dict(
                    target="diffusion.model.nets.basic_modules.CaptionEmbedder",
                    params=dict(
                        y_proj_type="mlp",
                        act_type="gelu",
                        y_norm_type="rms",
                        bias=True,
                        dropout=0.0,
                        norm_eps=1e-6,
                    )
                )),
                block_config=DictConfig(dict(
                    target="diffusion.model.nets.causal.causal_emage.CausalEmageBlock",
                    params=dict(
                        num_heads=test_num_heads,
                        head_dim=None,
                        mapping_size=512,
                        simple_modulation=False,
                        bias=True,
                        norm_type="rms",
                        attn_type="flash",
                        qk_norm_type="rms",
                        ffn_type="mlp",
                        ffn_act_type="gelu",
                        ffn_proj_type="linear",
                        ffn_mlp_ratio=4.0,
                        attn_drop=0.0,
                        attn_proj_drop=0.0,
                        ffn_drop=0.0,
                        drop_path=0.0,
                        norm_eps=1e-6,
                        is_teacher_forcing=True,
                        local_attn_size=-1,
                        max_attention_size=512,
                        sink_size=0,
                    )
                )),
                finallayer_config=DictConfig(dict(
                    target="diffusion.model.nets.causal.causal_block.FinalLayer",
                    params=dict(
                        mapping_size=512,
                        bias=True,
                        norm_type="rms",
                        proj_type="linear",
                        norm_eps=1e-6,
                    )
                )),
                use_pe=True,
                pos_embed_type=pos_embed_type,
                pe_max_length=10000,
                simple_modulation=True,
                num_frame_per_block=64,
                independent_first_frame=False,
            ).cuda()

        print("\n创建模型...")
        model_rope = create_test_model("rope")
        model_liger = create_test_model("liger-rope")

        # 复制权重，确保两个模型完全相同
        print("复制权重，确保两个模型参数完全相同...")
        model_liger.load_state_dict(model_rope.state_dict())

        # 设置为评估模式
        model_rope.eval()
        model_liger.eval()

        # 准备测试数据
        torch.manual_seed(42)
        test_motion_dim = hand_in_channels + body_in_channels + trans_in_channels
        test_motion = torch.randn(test_batch_size, test_seq_len, test_motion_dim).cuda()
        test_timestep = torch.randint(0, 1000, (test_batch_size, test_seq_len)).cuda()
        test_context = torch.randn(test_batch_size, test_seq_len, 512).cuda()
        test_expression = torch.randn(test_batch_size, test_seq_len, 100).cuda()
        test_clean_motion = torch.randn(test_batch_size, test_seq_len, test_motion_dim).cuda()
        test_speaker_id = torch.zeros(test_batch_size, dtype=torch.long).cuda()

        test_kwargs = {
            "speaker_id": test_speaker_id,
            "hand_indices": hand_indices,
            "upper_indices": upper_indices,
            "lower_indices": lower_indices,
            "face_joint_indices": face_joint_indices,
            "trans_indices": trans_indices,
        }

        # ==================== 测试 1: forward_train ====================
        print("\n" + "-"*80)
        print("测试 1: forward_train 模式")
        print("-"*80)

        with torch.no_grad():
            output_rope = model_rope(
                motion=test_motion,
                timestep=test_timestep,
                context=test_context,
                expression=test_expression,
                clean_motion=test_clean_motion,
                **test_kwargs
            )

            output_liger = model_liger(
                motion=test_motion,
                timestep=test_timestep,
                context=test_context,
                expression=test_expression,
                clean_motion=test_clean_motion,
                **test_kwargs
            )

        # 比较输出
        motion_rope = output_rope["motion"]
        motion_liger = output_liger["motion"]

        diff = torch.abs(motion_rope - motion_liger)
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        allclose = torch.allclose(motion_rope, motion_liger, atol=1e-5, rtol=1e-4)

        print(f"\n输出比较:")
        print(f"  输出形状: {motion_rope.shape}")
        print(f"  最大差异: {max_diff:.2e}")
        print(f"  平均差异: {mean_diff:.2e}")
        print(f"  allclose(atol=1e-5, rtol=1e-4): {allclose}")

        if allclose:
            print("  结果: PASSED - 输出一致")
        else:
            print("  结果: FAILED - 输出不一致")
            print(f"  rope 输出范围: [{motion_rope.min().item():.4f}, {motion_rope.max().item():.4f}]")
            print(f"  liger 输出范围: [{motion_liger.min().item():.4f}, {motion_liger.max().item():.4f}]")

        # ==================== 测试 2: forward_inference (prefill) ====================
        print("\n" + "-"*80)
        print("测试 2: forward_inference 模式 (prefill)")
        print("-"*80)

        # 创建 KV cache - 参考 emage_inference.py 的实现
        def create_kv_cache(model, batch_size, dtype, device):
            """Initialize KV cache for inference."""
            kv_cache_size = 512  # 足够大的 cache
            num_heads = test_num_heads
            head_dim = test_hidden_size // num_heads

            hand_kv_cache = []
            body_kv_cache = []

            for _ in range(model.depth):
                hand_kv_cache.append({
                    "k": torch.zeros([batch_size, kv_cache_size, num_heads, head_dim], dtype=dtype, device=device),
                    "v": torch.zeros([batch_size, kv_cache_size, num_heads, head_dim], dtype=dtype, device=device),
                    "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                    "local_end_index": torch.tensor([0], dtype=torch.long, device=device)
                })
                body_kv_cache.append({
                    "k": torch.zeros([batch_size, kv_cache_size, num_heads, head_dim], dtype=dtype, device=device),
                    "v": torch.zeros([batch_size, kv_cache_size, num_heads, head_dim], dtype=dtype, device=device),
                    "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                    "local_end_index": torch.tensor([0], dtype=torch.long, device=device)
                })

            return {
                "hand_kv_cache": hand_kv_cache,
                "body_kv_cache": body_kv_cache,
            }

        kv_cache_rope = create_kv_cache(model_rope, test_batch_size, torch.float32, 'cuda')
        kv_cache_liger = create_kv_cache(model_liger, test_batch_size, torch.float32, 'cuda')

        # Prefill
        prefill_seq_len = 64
        test_clean_motion_prefill = test_clean_motion[:, :prefill_seq_len, :]

        with torch.no_grad():
            output_rope_prefill = model_rope.forward_inference(
                motion=None,
                timestep=None,
                context=test_context[:, :prefill_seq_len, :],
                expression=test_expression[:, :prefill_seq_len, :],
                kv_cache=kv_cache_rope,
                seq_len=prefill_seq_len,
                current_start=0,
                prefill_only=True,
                update_kv_cache=True,
                clean_motion=test_clean_motion_prefill,
                **test_kwargs
            )

            output_liger_prefill = model_liger.forward_inference(
                motion=None,
                timestep=None,
                context=test_context[:, :prefill_seq_len, :],
                expression=test_expression[:, :prefill_seq_len, :],
                kv_cache=kv_cache_liger,
                seq_len=prefill_seq_len,
                current_start=0,
                prefill_only=True,
                update_kv_cache=True,
                clean_motion=test_clean_motion_prefill,
                **test_kwargs
            )

        print(f"\nPrefill 完成 (seq_len={prefill_seq_len})")

        # 比较 KV cache
        cache_match = True
        max_k_diff = 0.0
        max_v_diff = 0.0

        for layer_idx in range(model_rope.depth):
            k_rope = kv_cache_rope["hand_kv_cache"][layer_idx]["k"]
            k_liger = kv_cache_liger["hand_kv_cache"][layer_idx]["k"]
            v_rope = kv_cache_rope["hand_kv_cache"][layer_idx]["v"]
            v_liger = kv_cache_liger["hand_kv_cache"][layer_idx]["v"]

            if k_rope is not None and k_liger is not None:
                k_diff = torch.abs(k_rope - k_liger).max().item()
                v_diff = torch.abs(v_rope - v_liger).max().item()
                max_k_diff = max(max_k_diff, k_diff)
                max_v_diff = max(max_v_diff, v_diff)

                k_match = torch.allclose(k_rope, k_liger, atol=1e-5, rtol=1e-4)
                v_match = torch.allclose(v_rope, v_liger, atol=1e-5, rtol=1e-4)

                if layer_idx == 0:
                    print(f"\nKV Cache 比较 (layer 0, hand):")
                    print(f"  K 最大差异: {k_diff:.2e}, allclose: {k_match}")
                    print(f"  V 最大差异: {v_diff:.2e}, allclose: {v_match}")

                if not (k_match and v_match):
                    cache_match = False

        # 检查 body cache
        for layer_idx in range(model_rope.depth):
            k_rope = kv_cache_rope["body_kv_cache"][layer_idx]["k"]
            k_liger = kv_cache_liger["body_kv_cache"][layer_idx]["k"]
            v_rope = kv_cache_rope["body_kv_cache"][layer_idx]["v"]
            v_liger = kv_cache_liger["body_kv_cache"][layer_idx]["v"]

            if k_rope is not None and k_liger is not None:
                k_diff = torch.abs(k_rope - k_liger).max().item()
                v_diff = torch.abs(v_rope - v_liger).max().item()
                max_k_diff = max(max_k_diff, k_diff)
                max_v_diff = max(max_v_diff, v_diff)

                k_match = torch.allclose(k_rope, k_liger, atol=1e-5, rtol=1e-4)
                v_match = torch.allclose(v_rope, v_liger, atol=1e-5, rtol=1e-4)

                if layer_idx == 0:
                    print(f"\nKV Cache 比较 (layer 0, body):")
                    print(f"  K 最大差异: {k_diff:.2e}, allclose: {k_match}")
                    print(f"  V 最大差异: {v_diff:.2e}, allclose: {v_match}")

                if not (k_match and v_match):
                    cache_match = False

        print(f"\n所有层 KV Cache:")
        print(f"  最大 K 差异: {max_k_diff:.2e}")
        print(f"  最大 V 差异: {max_v_diff:.2e}")

        if cache_match:
            print("  结果: PASSED - KV cache 一致")
        else:
            print("  结果: FAILED - KV cache 不一致")

        # ==================== 测试 3: forward_inference (decode) ====================
        print("\n" + "-"*80)
        print("测试 3: forward_inference 模式 (decode)")
        print("-"*80)

        # Decode 一个 token
        decode_seq_len = 1
        current_start = prefill_seq_len
        test_motion_decode = test_motion[:, current_start:current_start+decode_seq_len, :]
        test_timestep_decode = test_timestep[:, current_start:current_start+decode_seq_len]
        test_context_decode = test_context[:, current_start:current_start+decode_seq_len, :]
        test_expression_decode = test_expression[:, current_start:current_start+decode_seq_len, :]

        with torch.no_grad():
            output_rope_decode = model_rope.forward_inference(
                motion=test_motion_decode,
                timestep=test_timestep_decode,
                context=test_context_decode,
                expression=test_expression_decode,
                kv_cache=kv_cache_rope,
                seq_len=current_start + decode_seq_len,
                current_start=current_start,
                prefill_only=False,
                update_kv_cache=False,
                clean_motion=None,
                **test_kwargs
            )

            output_liger_decode = model_liger.forward_inference(
                motion=test_motion_decode,
                timestep=test_timestep_decode,
                context=test_context_decode,
                expression=test_expression_decode,
                kv_cache=kv_cache_liger,
                seq_len=current_start + decode_seq_len,
                current_start=current_start,
                prefill_only=False,
                update_kv_cache=False,
                clean_motion=None,
                **test_kwargs
            )

        # 比较输出
        motion_rope_decode = output_rope_decode["motion"]
        motion_liger_decode = output_liger_decode["motion"]

        diff_decode = torch.abs(motion_rope_decode - motion_liger_decode)
        max_diff_decode = diff_decode.max().item()
        mean_diff_decode = diff_decode.mean().item()
        allclose_decode = torch.allclose(motion_rope_decode, motion_liger_decode, atol=1e-5, rtol=1e-4)

        print(f"\n输出比较:")
        print(f"  输出形状: {motion_rope_decode.shape}")
        print(f"  最大差异: {max_diff_decode:.2e}")
        print(f"  平均差异: {mean_diff_decode:.2e}")
        print(f"  allclose(atol=1e-5, rtol=1e-4): {allclose_decode}")

        if allclose_decode:
            print("  结果: PASSED - 输出一致")
        else:
            print("  结果: FAILED - 输出不一致")

        # ==================== 总结 ====================
        print("\n" + "="*80)
        print("测试总结")
        print("="*80)

        all_passed = allclose and cache_match and allclose_decode

        print(f"\n测试 1 (forward_train): {'PASSED' if allclose else 'FAILED'}")
        print(f"测试 2 (forward_inference prefill): {'PASSED' if cache_match else 'FAILED'}")
        print(f"测试 3 (forward_inference decode): {'PASSED' if allclose_decode else 'FAILED'}")

        if all_passed:
            print("\n所有测试通过！Liger Kernel RoPE 与原始 RoPE 输出完全一致。")
        else:
            print("\n部分测试失败，请检查实现。")

        print("="*80)
