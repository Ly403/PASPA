# 双分�?
# 实际上就是DiT
import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import DropPath
from diffusion.model.nets.causal.causal_block import CausalSelfAttention
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
    target="diffusion.model.nets.causal.causal_ddit.CausalDDITBlock",
    params=dict(
        hidden_size=None,
        num_heads=8,
        head_dim=None,
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


class CausalDDITBlock(nn.Module):
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
        self.hand_norm2 = build_norm(norm_type, num_features=hidden_size, affine=False, eps=norm_eps)
        self.body_norm2 = build_norm(norm_type, num_features=hidden_size, affine=False, eps=norm_eps)
        
        self.attn = CausalSelfAttention(
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
        self.attn2 = deepcopy(self.attn)
        
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
            nn.init.constant_(self.hand_scale_shift_table, 0)
            nn.init.constant_(self.body_scale_shift_table, 0)
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
        hand_kv_cache=None,
        body_kv_cache=None,
        current_start=0,
        prefill_only: bool = False,
        update_kv_cache: bool = True,
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

        # P0-3: Use keyword arguments to avoid parameter misalignment
        residual_hand = self.attn(
            motion=residual_hand,
            mask=mask,
            rotary_emb=rotary_emb,
            kv_cache=hand_kv_cache,
            current_start=current_start,
            prefill_only=prefill_only,
            update_kv_cache=update_kv_cache,
        )
        residual_body = self.attn2(
            motion=residual_body,
            mask=mask,
            rotary_emb=rotary_emb,
            kv_cache=body_kv_cache,
            current_start=current_start,
            prefill_only=prefill_only,
            update_kv_cache=update_kv_cache,
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
    
class CausalDDITEmage(nn.Module):
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
        
        inputlayer_config["params"]["in_channels"] = \
            self.hand_in_channels + self.body_in_channels + self.expression_in_channels + self.trans_in_channels
        self.xhand_in = instantiate_from_config(inputlayer_config)
        self.xbody_in = instantiate_from_config(inputlayer_config)
        
        inputlayer_config["params"]["in_channels"] = self.hidden_size * 3
        self.hand_fuse = instantiate_from_config(inputlayer_config)
        self.body_fuse = instantiate_from_config(inputlayer_config)
        
        timestep_embedder_config["params"]["hidden_size"] = hidden_size
        self.motion_t_embedder = instantiate_from_config(timestep_embedder_config)
        self.simple_modulation = simple_modulation
        self.style_embed = StyleEmbedding(num_person, 64)
        
        self.use_pe = use_pe
        self.pe_max_length = pe_max_length
        # Will use fixed sin-cos embedding:
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
                self.register_buffer("pos_embed", torch.zeros(1, pe_max_length, hidden_size))
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
            logger(
                f"attention type: {attn_type}; ffn type: {ffn_type}; self-attn qk norm: {qk_norm_type}; "
            )
    
    
    def forward_train(
        self,  
        motion,
        timestep,
        context,
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
        #########################################################
        # for expression
        #########################################################
        
        hand = motion[..., hand_indices]
        body = motion[..., body_indices]
        
        hand = self.hand_in(hand)
        body = self.body_in(body)

        x = torch.cat([motion, expression], dim=-1)
        xhand = self.xhand_in(x) 
        xbody = self.xbody_in(x)
        
        context = self.context_embedder(context)  # (N, L, D)
        hand = self.hand_fuse(torch.cat([hand, xhand, context], dim=-1))
        body = self.body_fuse(torch.cat([body, xbody, context], dim=-1))
        
        # t = timestep / self.training_step # 归一�?
        t = self.motion_t_embedder(timestep)  # (N, L, D)
        one_hots = F.one_hot(id, num_classes=self.num_person).float().to(device=t.device, dtype=t.dtype)
        id_embed = self.style_embed(one_hots)
        if t.dim() == 3:
            id_embed = id_embed[:, None, :].expand(-1, t.size(1), -1)
        t = torch.cat([t, id_embed], dim=-1)
        if self.simple_modulation:
            t0 = self.t_block(t)
        else:
            t0 = t
        
        if clean_motion is not None:
            clean_hand = clean_motion[..., hand_indices]
            clean_body = clean_motion[..., body_indices]
            
            clean_hand = self.hand_in(clean_hand)
            clean_body = self.body_in(clean_body)
            x = torch.cat([clean_motion, expression], dim=-1)
            xhand = self.xhand_in(x) 
            xbody = self.xbody_in(x)
            clean_hand = self.hand_fuse(torch.cat([clean_hand, xhand, context], dim=-1))
            clean_body = self.body_fuse(torch.cat([clean_body, xbody, context], dim=-1))
            
            hand = torch.cat([clean_hand, hand], dim=1)
            body = torch.cat([clean_body, body], dim=1)
            if aug_timestep is None:
                aug_timestep = torch.zeros_like(timestep)
            t_clean = self.motion_t_embedder(aug_timestep)
            t_clean = torch.cat([t_clean, id_embed], dim=-1)
            if self.simple_modulation:
                clean_t0 = self.t_block(t_clean)
            else:
                clean_t0 = t_clean
            t0 = torch.cat([clean_t0, t0], dim=1)
             
        pos_embed = None
        if self.use_pe:
            if self.pos_embed_type == "sincos":
                hand = hand + self.pos_embed[:,:hand.shape[1], :].to(device=hand.device, dtype=hand.dtype)  # (N, T, D)
                body = body + self.pos_embed[:,:body.shape[1], :].to(device=body.device, dtype=body.dtype)  # (N, T, D)
                if clean_motion is not None:
                    clean_hand = clean_hand + \
                        self.pos_embed[:,:clean_hand.shape[1], :].to(device=clean_hand.device, dtype=clean_hand.dtype)  # (N, T, D)
                    clean_body = clean_body + \
                        self.pos_embed[:,:clean_body.shape[1], :].to(device=clean_body.device, dtype=clean_body.dtype)  # (N, T, D)
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
            hand_kv_cache = None,
            body_kv_cache = None,
            current_start = 0,
        )
        if torch.is_grad_enabled():
            for block in self.blocks:    
                hand, body = auto_grad_checkpoint(block, hand=hand, body=body, **kwargs)
        else:
            for block in self.blocks:  
                hand, body = block(hand=hand, body=body, **kwargs)
        
        if clean_motion is not None:
            hand = hand[:,hand.shape[1] // 2:]
            body = body[:,body.shape[1] // 2:]
            t0 = t0[:,t0.shape[1] // 2:]

        hand = self.hand_final_layer(hand, t)
        body = self.body_final_layer(body, t)
        
        output_motion = torch.zeros_like(motion)
        output_motion[..., hand_indices] = hand.to(motion.dtype)
        output_motion[..., body_indices] = body.to(motion.dtype)
        return {
            "motion": output_motion, 
        }
    
    def forward_inference(
        self,
        motion,
        timestep,
        context,
        expression=None,
        kv_cache: dict = None,
        seq_len:int = None,
        current_start: int = 0,
        prefill_only: bool = False,
        update_kv_cache: bool = True,
        **kwargs
    ):
        """
        Forward pass for inference with KV cache support.
            motion: (N, T, D) tensor of motion (noisy in sampling, clean in prefill)
            expression: (N, T, D) tensor of expression
            timestep: (N, T) tensor of timestep
            context: (N, L, D) tensor of condition
            prefill_only: stage marker only; does NOT change the main compute graph
            update_kv_cache: controls whether KV cache is written (True) or read-only (False)
            clean_motion: (optional) accepted for compatibility but ignored in inference

        Note:
            Prefill and decode share the same forward compute path.
            The only stage difference is KV cache read/write policy.
        """
        
        # P1-1: Cache structure consistency check
        if kv_cache is not None:
            # Verify cache has expected k/v structure for DDIT
            if "hand_kv_cache" in kv_cache and "body_kv_cache" in kv_cache:
                for block_idx, (hand_cache, body_cache) in enumerate(zip(
                    kv_cache["hand_kv_cache"], kv_cache["body_kv_cache"]
                )):
                    if not (isinstance(hand_cache, dict) and "k" in hand_cache and "v" in hand_cache):
                        raise ValueError(
                            f"DDIT expects k/v cache structure at block {block_idx}, "
                            f"but got hand_cache keys: {hand_cache.keys() if isinstance(hand_cache, dict) else type(hand_cache)}"
                        )
                    if not (isinstance(body_cache, dict) and "k" in body_cache and "v" in body_cache):
                        raise ValueError(
                            f"DDIT expects k/v cache structure at block {block_idx}, "
                            f"but got body_cache keys: {body_cache.keys() if isinstance(body_cache, dict) else type(body_cache)}"
                        )
        
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
        
        hand = motion[..., hand_indices]
        body = motion[..., body_indices]
        
        hand = self.hand_in(hand)
        body = self.body_in(body)

        x = torch.cat([motion, expression], dim=-1)
        xhand = self.xhand_in(x)
        xbody = self.xbody_in(x)

        context = self.context_embedder(context)  # (N, L, D)
        hand = self.hand_fuse(torch.cat([hand, xhand, context], dim=-1))
        body = self.body_fuse(torch.cat([body, xbody, context], dim=-1))
        
        t = self.motion_t_embedder(timestep)  # (N, L, D)
        one_hots = F.one_hot(id, num_classes=self.num_person).float().to(device=t.device, dtype=t.dtype)
        id_embed = self.style_embed(one_hots)
        if t.dim() == 3:
            id_embed = id_embed[:, None, :].expand(-1, t.size(1), -1)
        t = torch.cat([t, id_embed], dim=-1)
        if self.simple_modulation:
            t0 = self.t_block(t)
        else:
            t0 = t
        
        pos_embed = None
        if self.use_pe:
            if self.pos_embed_type == "sincos":
                hand = hand + self.pos_embed[:,:hand.shape[1], :].to(device=hand.device, dtype=hand.dtype)  # (N, T, D)
                body = body + self.pos_embed[:,:body.shape[1], :].to(device=body.device, dtype=body.dtype)  # (N, T, D)
            elif self.pos_embed_type == "rope":
                seq_len = seq_len if seq_len is not None else hand.shape[1]
                pos_embed = self.pos_embed._prepare_latent_sequence_ids(
                        hand.shape[0], seq_len, hand.device, self.dtype)
                pos_embed = self.pos_embed(pos_embed)
            elif self.pos_embed_type == "none":
                pass
            else:
                raise NotImplementedError(f"pos embed type {self.pos_embed_type} not implemented")
        
        block_kwargs = dict(
            context = context,
            t = t0,
            mask = None,
            rotary_emb = pos_embed,
            sampling_mode=True,
            prefill_only=prefill_only,
            update_kv_cache=update_kv_cache,
        )
        
        for block_index, block in enumerate(self.blocks):
            block_kwargs.update(
                {
                    "hand_kv_cache": kv_cache["hand_kv_cache"][block_index],
                    "body_kv_cache": kv_cache["body_kv_cache"][block_index],
                    "current_start": current_start,
                }
            )
            if torch.is_grad_enabled():
                hand, body = auto_grad_checkpoint(block, hand=hand, body=body, **block_kwargs)
            else:
                hand, body = block(hand=hand, body=body, **block_kwargs)

        hand = self.hand_final_layer(hand, t)
        body = self.body_final_layer(body, t)
        
        output_motion = torch.zeros_like(motion)
        output_motion[..., hand_indices] = hand.to(motion.dtype)
        output_motion[..., body_indices] = body.to(motion.dtype)

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
        total_length = num_frames * frame_seqlen

        # we do right padding to get to a multiple of 128
        padded_length = math.ceil(total_length / unit_length) * unit_length - total_length

        ends = torch.zeros(total_length + padded_length,
                           device=device, dtype=torch.long)

        # Block-wise causal mask will attend to all elements that are before the end of the current chunk
        frame_indices = torch.arange(
            start=0,
            end=total_length,
            step=frame_seqlen * num_frame_per_block,
            device=device
        )

        for tmp in frame_indices:
            ends[tmp:tmp + frame_seqlen * num_frame_per_block] = tmp + \
                frame_seqlen * num_frame_per_block

        def attention_mask(b, h, q_idx, kv_idx):
            if local_attn_size == -1:
                return (kv_idx < ends[q_idx]) | (q_idx == kv_idx)
            else:
                return ((kv_idx < ends[q_idx]) & (kv_idx >= (ends[q_idx] - local_attn_size * frame_seqlen))) | (q_idx == kv_idx)
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
        total_length = num_frames * frame_seqlen * 2

        # we do right padding to get to a multiple of 128
        padded_length = math.ceil(total_length / unit_length) * unit_length - total_length

        clean_ends = num_frames * frame_seqlen
        # for clean context frames, we can construct their flex attention mask based on a [start, end] interval
        context_ends = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        # for noisy frames, we need two intervals to construct the flex attention mask [context_start, context_end] [noisy_start, noisy_end]
        noise_context_starts = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        noise_context_ends = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        noise_noise_starts = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)
        noise_noise_ends = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)

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
            context_ends[start:start + attention_block_size] = start + attention_block_size

        noisy_image_start_list = torch.arange(
            num_frames * frame_seqlen, total_length,
            step=attention_block_size,
            device=device, dtype=torch.long
        )
        noisy_image_end_list = noisy_image_start_list + attention_block_size

        # attention for noisy frames
        for block_index, (start, end) in enumerate(zip(noisy_image_start_list, noisy_image_end_list)):
            # attend to noisy tokens within the same block
            noise_noise_starts[start:end] = start
            noise_noise_ends[start:end] = end
            # attend to context tokens in previous blocks
            # noise_context_starts[start:end] = 0
            noise_context_ends[start:end] = block_index * attention_block_size

        def attention_mask(b, h, q_idx, kv_idx):
            # first design the mask for clean frames
            clean_mask = (q_idx < clean_ends) & (kv_idx < context_ends[q_idx])
            # then design the mask for noisy frames
            # noisy frames will attend to all clean preceeding clean frames + itself
            C1 = (kv_idx < noise_noise_ends[q_idx]) & (kv_idx >= noise_noise_starts[q_idx])
            C2 = (kv_idx < noise_context_ends[q_idx]) & (kv_idx >= noise_context_starts[q_idx])
            noise_mask = (q_idx >= clean_ends) & (C1 | C2)

            eye_mask = q_idx == kv_idx
            return eye_mask | clean_mask | noise_mask

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
        total_length = num_frames * frame_seqlen

        # we do right padding to get to a multiple of 128
        padded_length = math.ceil(total_length / unit_length) * unit_length - total_length

        ends = torch.zeros(total_length + padded_length,
                           device=device, dtype=torch.long)

        # special handling for the first frame
        ends[:frame_seqlen] = frame_seqlen

        # Block-wise causal mask will attend to all elements that are before the end of the current chunk
        frame_indices = torch.arange(
            start=frame_seqlen,
            end=total_length,
            step=frame_seqlen * num_frame_per_block,
            device=device
        )

        for idx, tmp in enumerate(frame_indices):
            ends[tmp:tmp + frame_seqlen * num_frame_per_block] = tmp + \
                frame_seqlen * num_frame_per_block

        def attention_mask(b, h, q_idx, kv_idx):
            if local_attn_size == -1:
                return (kv_idx < ends[q_idx]) | (q_idx == kv_idx)
            else:
                return ((kv_idx < ends[q_idx]) & (kv_idx >= (ends[q_idx] - local_attn_size * frame_seqlen))) | \
                    (q_idx == kv_idx)

        if not debug:
            block_mask = create_block_mask(attention_mask, B=None, H=None, Q_LEN=total_length + padded_length,
                                        KV_LEN=total_length + padded_length, _compile=False, device=device)
        else:
            block_mask = create_mask(attention_mask, B=None, H=None, Q_LEN=total_length + padded_length,
                                        KV_LEN=total_length + padded_length, device=device)


        return block_mask
    
if __name__ == "__main__":
    import imageio
    import cv2
    ## for teacher forcing
    block_mask = CausalDDITEmage._prepare_teacher_forcing_mask(
        num_frames = 9,
        frame_seqlen = 256,
        device=torch.device("cuda"), 
        debug=True
    )
    print(block_mask)    
    block_mask = cv2.resize(block_mask[0, 0].cpu().float().numpy(), (1024, 1024))
    imageio.imwrite("teacher_forcing_mask_%d.jpg" % (0), np.uint8(255. * block_mask))
    ## for blockwise causal mask
    block_mask = CausalDDITEmage._prepare_blockwise_causal_attn_mask(
        device=torch.device("cuda"), 
        num_frames = 9,
        frame_seqlen = 256,
        debug=True
    )
    print(block_mask)    
    block_mask = cv2.resize(block_mask[0, 0].cpu().float().numpy(), (1024, 1024))
    imageio.imwrite("blockwise_causal_mask_%d.jpg" % (0), np.uint8(255. * block_mask))
    ## for blockwise causal mask i2v
    block_mask = CausalDDITEmage._prepare_blockwise_causal_attn_mask_i2v(
        device=torch.device("cuda"), 
        num_frames = 20,
        frame_seqlen = 256,
        num_frame_per_block = 4,
        debug=True
    )
    print(block_mask)    
    block_mask = cv2.resize(block_mask[0, 0].cpu().float().numpy(), (1024, 1024))
    imageio.imwrite("blockwise_causal_mask_i2v_%d.jpg" % (0), np.uint8(255. * block_mask))
    
    
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device =", device)

    # -------------------------
    # 1) teacher forcing mask sanity check
    # -------------------------
    num_frames = 256
    frame_seqlen = 1                 # 你训练时就是 1 token / frame
    num_frame_per_block = 64
    unit_length = 128                # 必须�?attention �?padding_unit_length 保持一致（默认 128�?

    # dense mask（debug=True）方便做语义断言
    tf_dense = CausalDDITEmage._prepare_teacher_forcing_mask(
        device=device,
        num_frames=num_frames,
        frame_seqlen=frame_seqlen,
        num_frame_per_block=num_frame_per_block,
        unit_length=unit_length,
        debug=True,
    )  # (1,1,Q,KV) bool

    M = tf_dense[0, 0]  # (Q,KV)
    clean_len = num_frames * frame_seqlen
    total_len = clean_len * 2
    padded_len = math.ceil(total_len / unit_length) * unit_length - total_len
    Q = total_len + padded_len

    assert M.shape == (Q, Q), f"mask shape mismatch: {M.shape} vs {(Q,Q)}"
    assert torch.all(torch.diag(M)), "mask diagonal should be True"

    block = frame_seqlen * num_frame_per_block

    # (A) clean 部分：每�?clean token 只能看到“当�?clean block 内”的 tokens（block-wise 全连接）
    #     并且不能看到后续 clean block �?tokens
    for q in range(clean_len):
        q_block_start = (q // block) * block
        q_block_end = q_block_start + block
        # block 内全可见
        assert bool(M[q, q_block_start:q_block_end].all()), f"clean q={q} should see its clean block"
        # block 外（后面）不可见（取一个点测即可）
        if q_block_end < clean_len:
            assert not bool(M[q, q_block_end].item()), f"clean q={q} should NOT see next clean block start"

    # (B) noisy 部分：noisy token 属于�?i �?noisy block�?
    #     1) 必须能看到“本 noisy block 内”的 tokens
    #     2) 必须能看到“前 i �?clean block”的 tokens（即 clean prefix，长�?i*block�?
    #     3) 不能看到未来 noisy block / 不能看到更靠后的 clean tokens
    for q in range(clean_len, total_len):
        i = (q - clean_len) // block
        noisy_block_start = clean_len + i * block
        noisy_block_end = noisy_block_start + block

        assert bool(M[q, noisy_block_start:noisy_block_end].all()), f"noisy q={q} should see its noisy block"
        if noisy_block_end < total_len:
            assert not bool(M[q, noisy_block_end].item()), f"noisy q={q} should NOT see next noisy block start"

        clean_vis_end = i * block
        if clean_vis_end > 0:
            assert bool(M[q, 0:clean_vis_end].all()), f"noisy q={q} should see clean prefix [0:{clean_vis_end})"
        if clean_vis_end < clean_len:
            assert not bool(M[q, clean_vis_end].item()), f"noisy q={q} should NOT see clean token at {clean_vis_end}"

    print(f"[OK] teacher forcing dense mask sanity passed. shape={tuple(tf_dense.shape)}")

    # -------------------------
    # 2) end-to-end: DDIT block forward with BlockMask (debug=False)
    #    这里用你�?DDIT：hand/body 分开做自注意力（两套 CausalSelfAttention�?
    # -------------------------
    if device.type != "cuda":
        print("[SKIP] attention forward test (flex_attention 通常需�?CUDA 环境).")
    else:
        tf_blockmask = CausalDDITEmage._prepare_teacher_forcing_mask(
            device=device,
            num_frames=num_frames,
            frame_seqlen=frame_seqlen,
            num_frame_per_block=num_frame_per_block,
            unit_length=unit_length,
            debug=False,   # 返回 BlockMask，供 flex_attention �?
        )

        B = 16
        hidden = 256
        heads = 8
        seq_len = total_len  # �?padding 的长度（attention 内部�?pad �?unit_length�?

        # 构造一�?DDIT block
        block = CausalDDITBlock(
            hidden_size=hidden,
            num_heads=heads,
            attn_type="flash",
            qk_norm_type="identity",
            ffn_type="mlp",
            ffn_mlp_ratio=4.0,
            padding_unit_length=unit_length,
            max_attention_size=256,
            local_attn_size=-1,
            is_teacher_forcing=True,     # �?teacher forcing 场景对齐（虽然这�?rotary_emb=None�?
        ).to(device)

        hand = torch.randn(B, seq_len, hidden, device=device, dtype=torch.float32)
        body = torch.randn(B, seq_len, hidden, device=device, dtype=torch.float32)
        t = torch.randn(B, seq_len, hidden, device=device, dtype=torch.float32)  # 注意：这�?t 的最后一维必须是 hidden

        with torch.no_grad():
            out_hand, out_body = block(
                hand=hand,
                body=body,
                t=t,
                mask=tf_blockmask,
                rotary_emb=None,
                hand_kv_cache=None,
                body_kv_cache=None,
                current_start=0,
            )

        assert out_hand.shape == hand.shape and out_body.shape == body.shape
        assert torch.isfinite(out_hand).all() and torch.isfinite(out_body).all()
        print(f"[OK] DDIT block forward passed. out_hand={tuple(out_hand.shape)}, out_body={tuple(out_body.shape)}")
