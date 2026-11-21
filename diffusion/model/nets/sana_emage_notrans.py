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
    MLP, GatedMLP,
    DoubleStreamGatedMLP,
    DoubleStreamMLP,
    t2i_modulate,
    StyleEmbedding,
    PoolContext,
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
    target="diffusion.model.nets.sana_emage_notrans.SanaEmageNoTransSelfAttnBlock",
    params=dict(
        hidden_size=None,
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


    
class SanaEmageNoTransSelfAttnBlock(nn.Module):
    """
    A Sana block with global shared adaptive layer norm (adaLN-single) conditioning.
    """

    def __init__(
        self,
        hidden_size,
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
        n_seed=8,
    ):
        super().__init__()
        self.n_seed = n_seed
        self.hand_norm1 = build_norm(norm_type, num_features=hidden_size, affine=False, eps=norm_eps)
        self.body_norm1 = build_norm(norm_type, num_features=hidden_size, affine=False, eps=norm_eps)
        self.hand_norm2 = build_norm(norm_type, num_features=hidden_size, affine=False, eps=norm_eps)
        self.body_norm2 = build_norm(norm_type, num_features=hidden_size, affine=False, eps=norm_eps)   
        self.hand_norm3 = build_norm(norm_type, num_features=hidden_size, affine=False, eps=norm_eps)
        self.body_norm3 = build_norm(norm_type, num_features=hidden_size, affine=False, eps=norm_eps)        
        
        self.hand_drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.body_drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        
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
        
          
        self.simple_modulation = simple_modulation
        if self.simple_modulation:
            # like pixart
            self.hand_scale_shift_table = nn.Parameter(torch.randn(9, hidden_size))
            self.body_scale_shift_table = nn.Parameter(torch.randn(9, hidden_size))
        else:
            self.hand_time_in = nn.Sequential(nn.SiLU(), nn.Linear(mapping_size, 9 * hidden_size, bias=bias))   
            self.body_time_in = nn.Sequential(nn.SiLU(), nn.Linear(mapping_size, 9 * hidden_size, bias=bias))   
        
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
        rotary_emb: torch.FloatTensor = None,
        hand_prompt: torch.FloatTensor = None, 
        body_prompt: torch.FloatTensor = None,
        **kwargs
    ):
        assert hand.shape[0] == body.shape[0], "batch size must be the same"
        B, N, C = hand.shape

        if self.simple_modulation:
            hand_shift_msa, hand_scale_msa, hand_gate_msa, \
                hand_shift_cross, hand_scale_cross, hand_gate_cross, \
                hand_shift_mlp, hand_scale_mlp, hand_gate_mlp = (
                self.hand_scale_shift_table[None] + t.reshape(B, 9, -1)
            ).chunk(9, dim=1)
            body_shift_msa, body_scale_msa, body_gate_msa, \
                body_shift_cross, body_scale_cross, body_gate_cross, \
                body_shift_mlp, body_scale_mlp, body_gate_mlp = (
                self.body_scale_shift_table[None] + t.reshape(B, 9, -1)
            ).chunk(9, dim=1)
        else:
            hand_shift_msa, hand_scale_msa, hand_gate_msa, \
                hand_shift_cross, hand_scale_cross, hand_gate_cross, \
                hand_shift_mlp, hand_scale_mlp, hand_gate_mlp\
                = self.hand_time_in(t.unsqueeze(1)).chunk(9, dim=-1)
            body_shift_msa, body_scale_msa, body_gate_msa, \
                body_shift_cross, body_scale_cross, body_gate_cross, \
                body_shift_mlp, body_scale_mlp, body_gate_mlp\
                = self.body_time_in(t.unsqueeze(1)).chunk(9, dim=-1)
        
        ######################################## The second layer ########################################
        # cross-attn (seed part only)
        hand_seed = hand[:, :self.n_seed, :]
        body_seed = body[:, :self.n_seed, :]

        hand_q = t2i_modulate(self.hand_norm1(hand_seed), hand_shift_cross, hand_scale_cross)
        body_q = t2i_modulate(self.body_norm1(body_seed), body_shift_cross, body_scale_cross)

        hand_delta = self.hand_cross_attn(hand_q, hand_prompt, None)
        body_delta = self.body_cross_attn(body_q, body_prompt, None)

        hand_seed = hand_seed + self.hand_drop_path(hand_gate_cross * hand_delta)
        body_seed = body_seed + self.body_drop_path(body_gate_cross * body_delta)

        hand = torch.cat([hand_seed, hand[:, self.n_seed:, :]], dim=1)
        body = torch.cat([body_seed, body[:, self.n_seed:, :]], dim=1)
        
        ##################################### The first layer #######################################33
        residual_hand = self.hand_norm2(hand)
        residual_body = self.body_norm2(body)
        
        residual_hand = t2i_modulate(residual_hand, hand_shift_msa, hand_scale_msa)
        residual_body = t2i_modulate(residual_body, body_shift_msa, body_scale_msa)

        residual_hand = self.attn(residual_hand, rotary_emb)
        residual_body = self.attn2(residual_body, rotary_emb)
    
        residual_hand = hand_gate_msa * residual_hand
        residual_body = body_gate_msa * residual_body
        
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


class SanaEmageNoTrans(nn.Module):
    def __init__(
        self,
        motion_in_channels=6,
        expression_in_channels=100,
        trans_in_channels=3,
        hand_joints = 30,
        body_joints = 13 + 9 + 3,
        hidden_size=384,
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
        n_seed=8,
        simple_modulation=False, # for pixart-like only one modulation
        bias_channels=False, # 如果这个打开了，bias全部设为False
        drop_condition=False,
        **kwargs,
    ):
        super().__init__()
        self.num_person = num_person
        self.motion_in_channels = motion_in_channels
        self.trans_in_channels = trans_in_channels
        self.expression_in_channels = expression_in_channels
        
        self.hidden_size = hidden_size
        self.depth = depth
        self.pos_embed_type = pos_embed_type
        self.fp32_attention = kwargs.get("use_fp32_attention", False)
        self.n_seed = n_seed
        self.drop_condition = drop_condition
        if self.drop_condition:
            self.null_embedding = nn.Parameter(torch.zeros((1, 1, expression_in_channels))) 
        if work_dir is not None:
            logger = get_root_logger(os.path.join(work_dir, "train_log.log"))
            logger = logger.info
        else:
            logger = print
        
        self.logger = logger
        self.bias_channels = bias_channels
        joints = hand_joints + body_joints
        ##  解决 motion 相关的初始化
        inputlayer_config["params"]["hidden_size"] = self.hidden_size
        inputlayer_config["params"]["in_channels"] = hand_joints * self.motion_in_channels + bias_channels
        self.hand_in = instantiate_from_config(inputlayer_config)
        self.seed_hand_in = instantiate_from_config(inputlayer_config)
        inputlayer_config["params"]["in_channels"] = body_joints * self.motion_in_channels + self.trans_in_channels + bias_channels
        self.body_in = instantiate_from_config(inputlayer_config)
        self.seed_body_in = instantiate_from_config(inputlayer_config)
        
        inputlayer_config["params"]["in_channels"] = joints * self.motion_in_channels + self.expression_in_channels + self.trans_in_channels
        self.xhand_in = instantiate_from_config(inputlayer_config)
        inputlayer_config["params"]["in_channels"] = joints * self.motion_in_channels + self.expression_in_channels + self.trans_in_channels
        self.xbody_in = instantiate_from_config(inputlayer_config)
        
        inputlayer_config["params"]["in_channels"] = self.hidden_size * 3
        self.hand_fuse = instantiate_from_config(inputlayer_config)
        self.body_fuse = instantiate_from_config(inputlayer_config)


        timestep_embedder_config["params"]["hidden_size"] = mapping_size
        self.motion_t_embedder = instantiate_from_config(timestep_embedder_config)
        self.simple_modulation = simple_modulation
        self.style_embed = StyleEmbedding(num_person, 64, bias=not bias_channels)
        
        self.use_pe = use_pe
        self.pe_max_length = pe_max_length
        # Will use fixed sin-cos embedding:
        self.register_buffer("pos_embed", torch.zeros(1, pe_max_length, hidden_size))
        if self.simple_modulation:
            self.t_block = nn.Sequential(nn.SiLU(), nn.Linear(mapping_size, 6 * hidden_size, bias=not bias_channels))
        
        caption_embedder_config["params"]["in_channels"] = caption_channels
        caption_embedder_config["params"]["hidden_size"] = hidden_size
        self.context_embedder = instantiate_from_config(caption_embedder_config)
        
        drop_path = [x.item() for x in torch.linspace(0, drop_path, depth)]  # stochastic depth decay rule
        block_config["params"]["hidden_size"] = hidden_size
        block_config["params"]["mapping_size"] = mapping_size
        block_config["params"]["simple_modulation"] = simple_modulation
        self.blocks = nn.ModuleList()
        for i in range(depth):
            block_config["params"]["drop_path"] = drop_path[i]
            self.blocks.append(instantiate_from_config(block_config))
        
        finallayer_config["params"]["hidden_size"] = hidden_size
        finallayer_config["params"]["mapping_size"] = mapping_size
        finallayer_config["params"]["out_channels"] = hand_joints * self.motion_in_channels
        self.hand_final_layer = instantiate_from_config(finallayer_config)
        finallayer_config["params"]["hidden_size"] = hidden_size
        finallayer_config["params"]["out_channels"] = body_joints * self.motion_in_channels + self.trans_in_channels
        self.body_final_layer = instantiate_from_config(finallayer_config)
        
        self.training_step = training_step
        if self.use_pe and self.pos_embed_type in ["sincos", "rope"]:
            if self.pos_embed_type == "sincos":
                # Initialize (and freeze) pos_embed by sin-cos embedding:
                pos_embed = get_1d_sincos_pos_embed_from_grid(
                    self.pos_embed.shape[-1],
                    np.arange(pe_max_length)
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
    
    def _process_expression_cfg(self, expression, seq_len):
        """
        处理 expression 的 CFG (Classifier-Free Guidance)。
        
        Args:
            expression: tensor (N, T, D) 或 list of tensors (some may be None)
            seq_len: 序列长度，用于广播 null embedding
            
        Returns:
            expression: (N, T, D) tensor
        """
        if not self.drop_condition or not isinstance(expression, (list, tuple)):
            return expression.to(self.dtype)
        
        # expression 是一个 list，其中某些元素可能是 None（表示 unconditional）
        non_none_indices = [i for i, e in enumerate(expression) if e is not None]
        none_indices = [i for i, e in enumerate(expression) if e is None]
        batch_size = len(expression)
        
        # 将 self.null 广播到正确的 seq_len
        null_expanded = self.null_embedding.expand(-1, seq_len, -1).to(self.dtype)  # (1, seq_len, expression_dim)
        
        # 全是 None 的情况
        if not non_none_indices:
            return null_expanded.expand(batch_size, -1, -1)
        
        # 批量处理非 None 的 expression
        non_none_expressions = torch.cat([
            e.unsqueeze(0) if e.dim() == 2 else e
            for e in [expression[i] for i in non_none_indices]
        ], dim=0).to(self.dtype)
        
        # 构建结果列表
        result = [None] * batch_size
        for idx, i in enumerate(non_none_indices):
            result[i] = non_none_expressions[idx:idx+1]
        
        if none_indices:
            for i in none_indices:
                result[i] = null_expanded
        else:
            # DDP 兼容性：确保 self.null 参与计算图
            null_dummy = null_expanded * 0.0
            for idx, i in enumerate(non_none_indices):
                result[i] = result[i] + null_dummy
        
        return torch.cat(result, dim=0)
            
    def forward(
        self,  
        motion, 
        timestep,
        expression,
        audio = None,
        txt = None,
        **kwargs
    ):
        """
        Forward pass of Sana.
            motion: (N, T, D) tensor of motion
            expression: (N, T, D) tensor of expression, or list of tensors (some may be None) when drop_condition=True
            timestep: (N, ) tensor of timestep
            cond: (N, L, D) tensor of condition
            cond_mask: (N, L) tensor of condition mask
        """
        motion = motion.to(self.dtype)
        timestep = timestep.to(self.dtype)
        expression = self._process_expression_cfg(expression, seq_len=motion.shape[1])
        
        cond = audio if audio is not None else txt
        
        id = kwargs["speaker_id"]
        hand_indices = kwargs["hand_indices"]
        upper_indices = kwargs["upper_indices"]
        lower_indices = kwargs["lower_indices"]
        face_joint_indices = kwargs["face_joint_indices"]
        trans_indices = kwargs['trans_indices']
        
        pose_prompt = kwargs['pose_prompt'].to(self.dtype)
        trans_prompt = kwargs['trans_prompt'].to(self.dtype)
        pose_prompt = torch.cat([pose_prompt, trans_prompt], dim=-1)

        body_indices = np.concatenate([upper_indices, lower_indices, trans_indices, face_joint_indices], axis=0)
        #########################################################
        # for expression
        #########################################################
        hand = motion[..., hand_indices]
        body = motion[..., body_indices]
        
        assert hand.shape[:2] == body.shape[:2], (hand.shape, body.shape)
        assert pose_prompt.shape[:2] == hand.shape[:2], (pose_prompt.shape, hand.shape)
        
        if self.bias_channels:
            hand = torch.cat([hand, torch.ones_like(hand[..., :1])], dim=-1) # 补充一个 bias channel
            body = torch.cat([body, torch.ones_like(body[..., :1])], dim=-1) # 补充一个 bias channel
        
        hand = self.hand_in(hand)
        body = self.body_in(body)
            
        x = torch.cat([motion, expression], dim=-1)
        xhand = self.xhand_in(x)
        xbody = self.xbody_in(x)
        
        seed_hand = pose_prompt[..., hand_indices][:,:self.n_seed, :]
        seed_body = pose_prompt[..., body_indices][:,:self.n_seed, :]
        if self.bias_channels:
            seed_hand = torch.cat([seed_hand, torch.ones_like(seed_hand[..., :1])], dim=-1) # 补充一个 bias channel
            seed_body = torch.cat([seed_body, torch.ones_like(seed_body[..., :1])], dim=-1) # 补充一个 bias channel
        
        seed_hand = self.seed_hand_in(seed_hand)
        seed_body = self.seed_body_in(seed_body)
        
        cond = self.context_embedder(cond, seq_len=hand.shape[1])  # (N, L, D)
        
        hand = self.hand_fuse(torch.cat([hand, xhand, cond], dim=-1))
        body = self.body_fuse(torch.cat([body, xbody, cond], dim=-1))
        
        # t = timestep / self.training_step # 归一化
        t = self.motion_t_embedder(timestep)  # (N, D)
        one_hots = F.one_hot(id, num_classes=self.num_person).float().to(device=t.device, dtype=t.dtype)
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
                seed_hand = seed_hand + self.pos_embed[:,:seed_hand.shape[1], :].to(
                    device=seed_hand.device, dtype=seed_hand.dtype)  # (N, T, D)
                seed_body = seed_body + self.pos_embed[:,:seed_body.shape[1], :].to(
                    device=seed_body.device, dtype=seed_body.dtype)  # (N, T, D)
            elif self.pos_embed_type == "rope":
                pos_embed = self.pos_embed._prepare_latent_sequence_ids(
                        hand.shape[0], hand.shape[1], hand.device, self.dtype)
                pos_embed = self.pos_embed(pos_embed)
                pos_embed = tuple(pos_embed)
            elif self.pos_embed_type == "none":
                pass
            else:
                raise NotImplementedError(f"pos embed type {self.pos_embed_type} not implemented")
        
        for block in self.blocks:
            hand, body = auto_grad_checkpoint(
                block, hand, body, t0, pos_embed, seed_hand, seed_body,
            )  # (N, T, D) #support grad checkpoint

        hand = self.hand_final_layer(hand, t)
        body = self.body_final_layer(body, t)

        output_motion = torch.zeros_like(motion)
        output_motion[..., hand_indices] = hand.to(motion.dtype)
        output_motion[..., body_indices] = body.to(motion.dtype)

        return {
            "motion": output_motion, # 返回的 tensor
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


if __name__ == "__main__":
    from calflops import calculate_flops
    
    print("="*60)
    print("计算 SanaEmageNoTrans 模型的参数量、计算量、MACs")
    print("="*60)
    
    # 模型配置
    motion_in_channels = 6
    expression_in_channels = 100
    trans_in_channels = 3
    hand_joints = 30
    body_joints = 13 + 9 + 3  # 25
    hidden_size = 384
    caption_channels = 1024
    mapping_size = 768
    depth = 6
    n_seed = 8
    
    # 创建模型
    model = SanaEmageNoTrans(
        motion_in_channels=motion_in_channels,
        expression_in_channels=expression_in_channels,
        trans_in_channels=trans_in_channels,
        hand_joints=hand_joints,
        body_joints=body_joints,
        hidden_size=hidden_size,
        caption_channels=caption_channels,
        mapping_size=mapping_size,
        depth=depth,
        drop_path=0.0,
        n_seed=n_seed,
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
                drop_condition=0.0,
            )
        )),
        block_config=DictConfig(dict(
            target="diffusion.model.nets.sana_emage_notrans.SanaEmageNoTransSelfAttnBlock",
            params=dict(
                n_seed=n_seed,
                num_heads=8,
                head_dim=None,
                simple_modulation=False,
                bias=True,
                norm_type="rms",
                attn_type="flash",
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
        )),
        finallayer_config=DictConfig(dict(
            target="diffusion.model.nets.basic_modules.FinalLayer",
            params=dict(
                bias=True,
                norm_type="rms",
                proj_type="linear",
                simple_modulation=False,
                norm_eps=1e-6,
            )
        )),
        work_dir="./",
        use_pe=True,
        pos_embed_type="rope",
        pe_max_length=10000,
        simple_modulation=False,
        bias_channels=0,
        drop_condition=0.0,
    ).cuda()
    
    # 构造输入维度
    batch_size = 1
    seq_len = 256  # 序列长度
    joints = hand_joints + body_joints  # 55
    motion_dim = joints * motion_in_channels + trans_in_channels  # 55*6+3=333
    
    # 定义索引
    hand_in_channels = hand_joints * motion_in_channels  # 180
    body_in_channels = body_joints * motion_in_channels  # 150
    
    hand_indices = np.arange(0, hand_in_channels)
    upper_indices = np.arange(hand_in_channels, hand_in_channels + 13 * motion_in_channels)
    lower_indices = np.arange(hand_in_channels + 13 * motion_in_channels, hand_in_channels + (13 + 9) * motion_in_channels)
    face_joint_indices = np.arange(hand_in_channels + (13 + 9) * motion_in_channels, hand_in_channels + body_in_channels)
    trans_indices = np.arange(hand_in_channels + body_in_channels, motion_dim)
    
    # 包装 forward 函数以适配 calflops
    class ModelWrapper(nn.Module):
        def __init__(self, model, kwargs):
            super().__init__()
            self.model = model
            self.kwargs = kwargs
            
        def forward(self, motion, timestep, expression, audio):
            return self.model(
                motion=motion,
                timestep=timestep,
                expression=expression,
                audio=audio,
                **self.kwargs
            )
    
    # 创建实际的输入张量
    dummy_motion = torch.randn(batch_size, seq_len, motion_dim).cuda()
    dummy_timestep = torch.rand(batch_size).cuda() * 1000  # (N,) float 类型
    dummy_expression = torch.randn(batch_size, seq_len, expression_in_channels).cuda()
    dummy_audio = torch.randn(batch_size, seq_len, caption_channels).cuda()
    dummy_pose_prompt = torch.randn(batch_size, seq_len, motion_dim).cuda()
    dummy_trans_prompt = torch.randn(batch_size, seq_len, trans_in_channels).cuda()
    
    kwargs = {
        "speaker_id": torch.zeros(batch_size, dtype=torch.long).cuda(),
        "hand_indices": hand_indices,
        "upper_indices": upper_indices,
        "lower_indices": lower_indices,
        "face_joint_indices": face_joint_indices,
        "trans_indices": trans_indices,
        "pose_prompt": dummy_pose_prompt,
        "trans_prompt": dummy_trans_prompt,
    }
    
    wrapped_model = ModelWrapper(model, kwargs).cuda()
    wrapped_model.eval()
    
    print(f"\n模型配置:")
    print(f"  - hidden_size: {hidden_size}")
    print(f"  - depth: {depth}")
    print(f"  - mapping_size: {mapping_size}")
    print(f"  - motion_dim: {motion_dim}")
    print(f"  - caption_channels: {caption_channels}")
    print(f"  - expression_in_channels: {expression_in_channels}")
    print(f"  - n_seed: {n_seed}")
    print(f"\n输入形状 (batch={batch_size}, seq_len={seq_len}):")
    print(f"  - motion: ({batch_size}, {seq_len}, {motion_dim})")
    print(f"  - timestep: ({batch_size},)")
    print(f"  - expression: ({batch_size}, {seq_len}, {expression_in_channels})")
    print(f"  - audio: ({batch_size}, {seq_len}, {caption_channels})")
    
    # 使用 calflops 计算
    with torch.no_grad():
        flops, macs, params = calculate_flops(
            model=wrapped_model,
            args=[dummy_motion, dummy_timestep, dummy_expression, dummy_audio],
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


