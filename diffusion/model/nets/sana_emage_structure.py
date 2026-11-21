# 实际上就是DiT
import os

import numpy as np
import torch
import torch.nn as nn
from timm.layers import DropPath
import torch.nn.functional as F
from diffusion.model.nets.basic_modules import MLP, GatedMLP, t2i_modulate, StyleEmbedding, SimpleSpeechModel, NoSpeechModel, modulate
from diffusion.model.nets.attention_modules import Attention, MultiHeadCrossAttention
from diffusion.model.nets.positional_embeding import RopePosEmbed
from diffusion.model.norms import build_norm
from diffusion.model.utils import auto_grad_checkpoint
from diffusion.utils.dist_utils import get_rank
from diffusion.utils.logger import get_root_logger
from diffusion.utils.config import instantiate_from_config
from omegaconf import DictConfig
from einops import rearrange

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
    target="diffusion.model.nets.sana_emage_structure.SanaEmageStructureBlock",
    params=dict(
        hidden_size=None,
        num_heads=8,
        mapping_size=768,
        head_dim=None,
        simple_modulation=False,
        bias=True,
        norm_type="rms",
        temporal_qk_norm_type="identity",
        structure_qk_norm_type="identity",
        cross_qk_norm_type="identity",
        temporal_attn_type="flash",
        structure_attn_type="flash",
        cross_attn_type="flash",
        temporal_attn_drop=0.0,
        structure_attn_drop=0.0,
        cross_attn_drop=0.0,
        temporal_attn_proj_drop=0.0,
        structure_attn_proj_drop=0.0,
        cross_attn_proj_drop=0.0,
        ffn_type="mlp",
        ffn_act_type="gelu",
        ffn_proj_type="linear", 
        ffn_mlp_ratio=4.0,
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
    target="diffusion.model.nets.sana_emage_structure.FinalLayer",
    params=dict(
        hidden_size=None, 
        out_channels=None,
        bias=True,
        norm_type="rms",
        proj_type="linear",
        simple_modulation=False,
        norm_eps=1e-6,
    )
))

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
        self.proj = nn.Linear(hidden_size,  out_channels, bias=bias)
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
            shift, scale = self.modulation(c.unsqueeze(1)).chunk(2, dim=-1)
        else:
            shift, scale = (self.scale_shift_table[None] + c[:, None]).chunk(2, dim=-1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.proj(x)
        return x


class SanaEmageStructureBlock(nn.Module):
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
        temporal_qk_norm_type="identity",
        structure_qk_norm_type="identity",
        cross_qk_norm_type="identity",
        temporal_attn_type="flash",
        structure_attn_type="flash",
        cross_attn_type="flash",
        temporal_attn_drop=0.0,
        structure_attn_drop=0.0,
        cross_attn_drop=0.0,
        temporal_attn_proj_drop=0.0,
        structure_attn_proj_drop=0.0,
        cross_attn_proj_drop=0.0,
        ffn_type="mlp",
        ffn_act_type="gelu",
        ffn_proj_type="linear", 
        ffn_mlp_ratio=4.0,
        ffn_drop=0.0,
        drop_path=0.0,
        norm_eps=1e-6,
    ):
        super().__init__()
        self.norm1 = build_norm(norm_type, num_features=hidden_size, affine=False, eps=norm_eps)
        self.norm2 = build_norm(norm_type, num_features=hidden_size, affine=False, eps=norm_eps)
        self.norm3 = build_norm(norm_type, num_features=hidden_size, affine=False, eps=norm_eps)
        
        self.temp_attn = Attention(
            dim=hidden_size,
            num_heads=num_heads,
            head_dim=head_dim,
            bias=bias,
            attn_type=temporal_attn_type,
            qk_norm_type=temporal_qk_norm_type,
            attn_drop=temporal_attn_drop, 
            proj_drop=temporal_attn_proj_drop,
        )
        self.stru_attn = Attention(
            dim=hidden_size,
            num_heads=num_heads,
            head_dim=head_dim,
            bias=bias,
            attn_type=structure_attn_type,
            qk_norm_type=structure_qk_norm_type,
            attn_drop=structure_attn_drop, 
            proj_drop=structure_attn_proj_drop,
        )

        self.cross_attn = MultiHeadCrossAttention(
            dim=hidden_size,
            num_heads=num_heads,
            head_dim=head_dim,
            bias=bias,
            attn_type=cross_attn_type,
            qk_norm_type=cross_qk_norm_type,
            attn_drop=cross_attn_drop, 
            proj_drop=cross_attn_proj_drop,
        )
        
        if ffn_type == "mlp":
            self.mlp = MLP(
                in_features=hidden_size, 
                hidden_features=int(hidden_size * ffn_mlp_ratio),  
                act_type=ffn_act_type,
                proj_type=ffn_proj_type,
                bias=bias, 
                dropout=ffn_drop
            )
        elif ffn_type == "gated_mlp":
            self.mlp = GatedMLP(
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
            self.scale_shift_table = nn.Parameter(torch.randn(12, hidden_size))
        else:
            self.time_in = nn.Sequential(nn.SiLU(), nn.Linear(mapping_size, 12 * hidden_size, bias=bias))   
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        
    def initialize_weights(self):
        if self.simple_modulation:
            nn.init.constant(self.scale_shift_table, 0)
        else:
            nn.init.constant_(self.time_in[1].weight, val=0)
            if self.time_in[1].bias is not None:
                nn.init.constant_(self.time_in[1].bias, val=0)


    def forward(
        self, 
        x: torch.FloatTensor,
        t: torch.FloatTensor,
        rotary_emb: torch.FloatTensor = None,
        seed_x: torch.FloatTensor = None, 
        seed_mask: torch.FloatTensor = None,
        **kwargs
    ):
        B, T, J, C = x.shape

        if self.simple_modulation:
            shift_temp_msa, scale_temp_msa, gate_temp_msa, \
                shift_stru_msa, scale_stru_msa, gate_stru_msa, \
                    shift_cross_msa, scale_cross_msa, gate_cross_msa, \
                    shift_mlp, scale_mlp, gate_mlp = (
                self.scale_shift_table[None] + t.reshape(B, 1, 12, -1)
            ).chunk(12, dim=1)
        else:
            shift_temp_msa, scale_temp_msa, gate_temp_msa, \
                shift_stru_msa, scale_stru_msa, gate_stru_msa, \
                    shift_cross_msa, scale_cross_msa, gate_cross_msa, \
                    shift_mlp, scale_mlp, gate_mlp\
                = self.time_in(t.unsqueeze(1).unsqueeze(1)).chunk(12, dim=-1)
        
        ##################################### The first layer #######################################33
        residual = self.norm1(x)
        residual = t2i_modulate(residual, shift_temp_msa, scale_temp_msa)
        residual = rearrange(residual, "b t j c -> (b j) t c")
        residual = self.temp_attn(residual, rotary_emb)
        residual = rearrange(residual, "(b j) t c -> b t j c", b=B, j=J)
        residual = gate_temp_msa * residual
        x = x + self.drop_path(residual)
        ######################################## The second layer ########################################
        residual = self.norm2(x)
        residual = t2i_modulate(residual, shift_stru_msa, scale_stru_msa)
        residual = rearrange(residual, "b t j c -> (b t) j c")
        residual = self.stru_attn(residual, None)
        residual = rearrange(residual, "(b t) j c -> b t j c", b=B, t=T)
        residual = gate_stru_msa * residual
        x = x + self.drop_path(residual)
        ######################################## The third layer ########################################
        residual = self.norm3(x)
        residual = t2i_modulate(residual, shift_cross_msa, scale_cross_msa)
        residual = rearrange(residual, "b t j c -> (b j) t c")
        seed_rx = rearrange(seed_x, "b t j c -> (b j) t c")
        seed_rmask = rearrange(seed_mask, "b t j -> (b j) t")
        residual = self.cross_attn(residual, seed_rx, None, seed_rmask)
        residual = rearrange(residual, "(b j) t c -> b t j c", b=B, j=J)
        residual = gate_cross_msa * residual
        x = x + self.drop_path(residual)
        
        ######################################## The fourth layer ########################################
        residual = self.norm3(x)
        residual = t2i_modulate(residual, shift_mlp, scale_mlp)
        residual = self.mlp(residual)
        residual = gate_mlp * residual
        x = x + self.drop_path(residual)
        return x


class SanaEmageStructure(nn.Module):
    def __init__(
        self,
        motion_in_channels=6,
        trans_in_channels=3,
        expression_in_channels=100,
        hidden_size=1152,
        mapping_size=768,
        caption_channels=2304,
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
        simple_modulation=False, # for pixart-like only one modulation
        training_step=1000,
        **kwargs,
    ):
        super().__init__()
        self.num_person = num_person
        self.motion_in_channels = motion_in_channels
        self.trans_in_channels = trans_in_channels
        self.motion_out_channels = motion_in_channels
        self.expression_in_channels = expression_in_channels
        
        self.hidden_size = hidden_size
        self.depth = depth
        self.pos_embed_type = pos_embed_type
        self.fp32_attention = kwargs.get("use_fp32_attention", False)
        self.training_step = training_step
        
        if work_dir is not None:
            logger = get_root_logger(os.path.join(work_dir, "train_log.log"))
            logger = logger.info
        else:
            logger = print
        
        self.logger = logger
        inputlayer_config["params"]["hidden_size"] = hidden_size
        
        inputlayer_config["params"]["in_channels"] = self.motion_in_channels
        self.poses_in = instantiate_from_config(inputlayer_config)
        self.seed_poses_in = instantiate_from_config(inputlayer_config)
        inputlayer_config["params"]["in_channels"] = self.trans_in_channels 
        self.trans_in = instantiate_from_config(inputlayer_config)
        self.seed_trans_in = instantiate_from_config(inputlayer_config)
        
        inputlayer_config["params"]["in_channels"] = self.expression_in_channels
        self.expression_in = instantiate_from_config(inputlayer_config)
        
        inputlayer_config["params"]["in_channels"] = hidden_size * 3
        self.poses_fuse = instantiate_from_config(inputlayer_config)
        inputlayer_config["params"]["in_channels"] = hidden_size * 2
        self.trans_fuse = instantiate_from_config(inputlayer_config)
        
        timestep_embedder_config["params"]["hidden_size"] = mapping_size - 64
        self.motion_t_embedder = instantiate_from_config(timestep_embedder_config)
        self.simple_modulation = simple_modulation
        
        self.use_pe = use_pe
        self.pe_max_length = pe_max_length
        # Will use fixed sin-cos embedding:
        self.register_buffer("pos_embed", torch.zeros(1, pe_max_length, hidden_size))
        if self.simple_modulation:
            self.t_block = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 12 * hidden_size, bias=True))
        
        caption_embedder_config["params"]["in_channels"] = caption_channels
        caption_embedder_config["params"]["hidden_size"] = hidden_size
        self.txt_embedder = instantiate_from_config(caption_embedder_config)
        self.style_embed = StyleEmbedding(num_person, 64)
         
        drop_path = [x.item() for x in torch.linspace(0, drop_path, depth)]  # stochastic depth decay rule
        block_config["params"]["hidden_size"] = hidden_size
        block_config["params"]["mapping_size"] = mapping_size
        self.blocks = nn.ModuleList()
        for i in range(depth):
            block_config["params"]["drop_path"] = drop_path[i]
            self.blocks.append(instantiate_from_config(block_config))
        
        finallayer_config["params"]["hidden_size"] = hidden_size
        finallayer_config["params"]["out_channels"] = self.motion_in_channels
        self.poses_final_layer = instantiate_from_config(finallayer_config)
        finallayer_config["params"]["out_channels"] = self.trans_in_channels
        self.trans_final_layer = instantiate_from_config(finallayer_config)

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
            temporal_attn_type = block_config["params"]["temporal_attn_type"]
            structure_attn_type = block_config["params"]["structure_attn_type"]
            cross_attn_type = block_config["params"]["cross_attn_type"]
            temporal_qk_norm_type = block_config["params"]["temporal_qk_norm_type"]
            structure_qk_norm_type = block_config["params"]["structure_qk_norm_type"]
            cross_qk_norm_type = block_config["params"]["cross_qk_norm_type"]
            ffn_type = block_config["params"]["ffn_type"]
            logger(
                f"temporal attention type: {temporal_attn_type};" 
                f"structure attention type: {structure_attn_type};"
                f"cross attention type: {cross_attn_type};"
                f"temporal qk norm type: {temporal_qk_norm_type};"
                f"structure qk norm type: {structure_qk_norm_type};"
                f"cross qk norm type: {cross_qk_norm_type};"
                f"ffn type: {ffn_type}; "
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
        bs, n = motion.shape[:2]
        x = motion.to(self.dtype)
        t = timestep.to(self.dtype)
        pos_embed = self.pos_embed.to(self.dtype)
        
        if audio is not None:
            cond = audio.to(self.dtype)
            cond_mask = audio_mask.to(self.dtype) if audio_mask is not None else audio_mask
        
        if txt is not None:
            cond = txt.to(self.dtype)
            cond_mask = txt_mask.to(self.dtype) if txt_mask is not None else txt_mask
        
        id = kwargs["speaker_id"]
        pose_prompt = kwargs['pose_prompt'].to(self.dtype)
        pose_prompt_mask = kwargs['pose_prompt_mask']
        trans_prompt = kwargs['trans_prompt'].to(self.dtype)
        trans_prompt_mask = kwargs['trans_prompt_mask']
        poses_indices = kwargs['poses_indices']
        trans_indices = kwargs['trans_indices']
        #########################################################
        # for expression
        #########################################################
        poses = x[..., poses_indices] # (N, T, J*6) rot6d
        trans = x[..., trans_indices] # (N, T, 3)
        
        poses = poses.view(bs, n, -1, 6) # (N, T, J, 6)
        poses = self.poses_in(poses) # (N, T, J, hidden_size)
        trans = self.trans_in(trans) # (N, T, hidden_size)
                
        seed_poses = self.seed_poses_in(pose_prompt.view(bs, n, -1, 6)) # (N, T, J, 6)
        seed_trans = self.seed_trans_in(trans_prompt) # (N, T, 3)
        
        cond = self.txt_embedder(cond)  # (N, L, D)
        cond_exp = self.expression_in(expression) # (N, T, hidden_size)
        poses = self.poses_fuse(torch.cat([
            poses, 
            cond_exp.unsqueeze(-2).repeat(1, 1, poses.shape[2], 1), 
            cond.unsqueeze(-2).repeat(1, 1, poses.shape[2], 1),
        ], dim=-1)) # (N, T, J, hidden_size)
        trans = self.trans_fuse(torch.cat([trans, cond_exp], dim=-1)) # (N, T, hidden_size)
        
        t = self.motion_t_embedder(t)  # (N, D)
        one_hots = F.one_hot(id, num_classes=self.num_person).float().to(x.device)
        t = torch.cat([t, self.style_embed(one_hots)], dim=-1)
        if self.simple_modulation:
            t0 = self.t_block(t)
        else:
            t0 = t
        
        pos_embed = None
        if self.use_pe:
            if self.pos_embed_type == "sincos":
                # 时间上仍需要位置编�?
                poses = poses + self.pos_embed[:,:poses.shape[1], :].unsqueeze(-2)  # (N, T, J, D)
                trans = trans + self.pos_embed[:,:trans.shape[1], :]  # (N, T, D)
                seed_poses = seed_poses + self.pos_embed[:,:seed_poses.shape[1], :].unsqueeze(-2)  # (N, T, J, D)
                seed_trans = seed_trans + self.pos_embed[:,:seed_trans.shape[1], :]  # (N, T, D)
            elif self.pos_embed_type == "rope":
                pos_embed = self.pos_embed._prepare_latent_sequence_ids(
                        poses.shape[0], poses.shape[1], poses.device, self.dtype)
                pos_embed = self.pos_embed(pos_embed)
            elif self.pos_embed_type == "none":
                pass
            else:
                raise NotImplementedError(f"pos embed type {self.pos_embed_type} not implemented")
        
        x = torch.cat([poses, trans.unsqueeze(-2)], dim=-2) # (N, T, J + 1, hidden_size)
        seed_x = torch.cat([seed_poses, seed_trans.unsqueeze(-2)], dim=-2) # (N, T, J + 1, hidden_size)
        seed_mask = torch.cat([
            pose_prompt_mask.unsqueeze(-1).repeat(1, 1, poses.shape[2]), 
            trans_prompt_mask.unsqueeze(-1)
        ], dim=-1) # (N, T, J + 1)
        for block in self.blocks:
            x = auto_grad_checkpoint(
                block, x, t0, pos_embed, seed_x, seed_mask
            )  # (N, T, D) #support grad checkpoint
        trans = x[:, :, -1, :].unsqueeze(-2) # (N, T, 1, hidden_size)
        poses = x[:, :, :-1, :] # (N, T, J, hidden_size)
        
        poses = self.poses_final_layer(poses, t)  
        trans = self.trans_final_layer(trans, t) 
        poses = rearrange(poses, "b t j c -> b t (j c)")
        trans = rearrange(trans, "b t 1 c -> b t c")

        output_motion = torch.zeros_like(motion)
        output_motion[..., poses_indices] = poses
        output_motion[..., trans_indices] = trans
        
        return {
            "motion": output_motion, # 返回�?tensor
        }
    
    def __call__(self, *args, **kwargs):
        """
        This method allows the object to be called like a function.
        It simply calls the forward method.
        """
        return self.forward(*args, **kwargs)

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
            # if isinstance(module, nn.Linear) or isinstance(module, nn.Conv1d):
            #     torch.nn.init.xavier_uniform_(module.weight)
            #     if module.bias is not None:
            #         nn.init.constant_(module.bias, 0)

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
