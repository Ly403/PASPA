import os

import numpy as np
import torch
import torch.nn as nn
from timm.layers import DropPath

from diffusion.model.nets.basic_modules import (
    TripleStreamMLP,
    TripleStreamGatedMLP,
    t2i_modulate 
)
from diffusion.model.nets.attention_modules import (
    TripleStreamFlashAttention,
    TripleStreamLiteLA,
    TripleStreamMultiHeadCrossAttention,
    TripleStreamCrossLiteLA,
)
from diffusion.model.nets.positional_embeding import RopePosEmbed
from diffusion.model.norms import build_norm
from diffusion.model.utils import auto_grad_checkpoint
from diffusion.utils.dist_utils import get_rank
from diffusion.utils.logger import get_root_logger
from diffusion.utils.config import instantiate_from_config
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
    target="diffusion.model.nets.sana_triple_stream.SanaTripleStreamBlock",
    params=dict(
        hidden_size=None,
        num_heads=8,
        head_dim=None,
        simple_modulation=False,
        bias=True,
        norm_type="rms",
        attn_type=None,
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
        bias=True,
        norm_type="rms",
        proj_type="linear",
        simple_modulation=False,
        norm_eps=1e-6,
    )
))


class SanaTripleStreamBlock(nn.Module):
    """
    A Sana block with global shared adaptive layer norm (adaLN-single) conditioning.
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
    ):
        super().__init__()
        self.motion_norm1 = build_norm(norm_type, num_features=hidden_size, affine=False, eps=norm_eps)
        self.expression_norm1 = build_norm(norm_type, num_features=hidden_size, affine=False, eps=norm_eps)
        self.audio_norm1 = build_norm(norm_type, num_features=hidden_size, affine=False, eps=norm_eps)
        if attn_type == "linear":
            self.attn = TripleStreamLiteLA(
                dim=hidden_size,
                num_heads=num_heads,
                head_dim=head_dim,
                bias=bias,
                attn_type=attn_type,
                qk_norm_type=qk_norm_type,
                attn_drop=attn_drop, 
                proj_drop=attn_proj_drop,
            )
        elif attn_type == "triton_linear":
            # linear self attention with triton kernel fusion
            # TODO: Here the num_heads set to 36 for tmp used
            # TODO: implement multi-head attention with triton kernel fusion
            raise NotImplementedError(f"{attn_type} type is not implemented.")
        else:
            # for attn_type = None, rolling back to the vanlilla attention 
            self.attn = TripleStreamFlashAttention(
                dim=hidden_size,
                num_heads=num_heads,
                head_dim=head_dim,
                bias=bias,
                attn_type=attn_type,
                qk_norm_type=qk_norm_type,
                attn_drop=attn_drop, 
                proj_drop=attn_proj_drop,
            )

        if cross_attn_type == "linear":
            self.cross_attn = TripleStreamCrossLiteLA(
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
            self.cross_attn = TripleStreamMultiHeadCrossAttention(
                dim=hidden_size,
                num_heads=num_heads,
                head_dim=head_dim,
                bias=bias,
                attn_type=cross_attn_type,
                qk_norm_type=cross_qk_norm_type,
                attn_drop=cross_attn_drop, 
                proj_drop=cross_attn_proj_drop,
            )
        
        self.motion_norm2 = build_norm(norm_type, num_features=hidden_size, affine=False, eps=norm_eps)
        self.expression_norm2 = build_norm(norm_type, num_features=hidden_size, affine=False, eps=norm_eps)
        self.audio_norm2 = build_norm(norm_type, num_features=hidden_size, affine=False, eps=norm_eps)
        
        if ffn_type == "mlp":
            self.mlp = TripleStreamMLP(
                in_features=hidden_size, 
                hidden_features=int(hidden_size * ffn_mlp_ratio),  
                act_type=ffn_act_type,
                proj_type=ffn_proj_type,
                bias=bias, 
                dropout=ffn_drop
            )
        elif ffn_type == "gated_mlp":
            self.mlp = TripleStreamGatedMLP(
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
            self.motion_scale_shift_table = nn.Parameter(torch.randn(6, hidden_size))
            self.expression_scale_shift_table = nn.Parameter(torch.randn(6, hidden_size))
            self.audio_scale_shift_table = nn.Parameter(torch.randn(6, hidden_size))
        else:
            self.motion_time_in = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=bias))   
            self.expression_time_in = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=bias))   
            self.audio_time_in = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=bias))
            
        
        self.motion_drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.expression_drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.audio_drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        
    def initialize_weights(self):
        if self.simple_modulation:
            nn.init.constant(self.motion_scale_shift_table, 0)
            nn.init.constant(self.expression_scale_shift_table, 0)
            nn.init.constant(self.audio_scale_shift_table, 0)
        else:
            nn.init.constant_(self.motion_time_in[1].weight, val=0)
            if self.motion_time_in[1].bias is not None:
                nn.init.constant_(self.motion_time_in[1].bias, val=0)
                
            nn.init.constant_(self.expression_time_in[1].weight, val=0)
            if self.expression_time_in[1].bias is not None:
                nn.init.constant_(self.expression_time_in[1].bias, val=0)
            
            nn.init.constant_(self.audio_time_in[1].weight, val=0)
            if self.audio_time_in[1].bias is not None:
                nn.init.constant_(self.audio_time_in[1].bias, val=0)


    def forward(
        self, 
        motion: torch.FloatTensor, 
        expression: torch.FloatTensor,
        audio: torch.FloatTensor,
        t: torch.FloatTensor,
        txt: torch.FloatTensor,
        rotary_emb: torch.FloatTensor = None,
        mask=None, 
        **kwargs
    ):
        assert motion.shape[0] == expression.shape[0] == audio.shape[0], "batch size must be the same"
        B, N, C = motion.shape

        if self.simple_modulation:
            motion_shift_msa, motion_scale_msa, motion_gate_msa, motion_shift_mlp, motion_scale_mlp, motion_gate_mlp = (
                self.scale_shift_table[None] + t.reshape(B, 6, -1)
            ).chunk(6, dim=1)
            expression_shift_msa, expression_scale_msa, expression_gate_msa, expression_shift_mlp, expression_scale_mlp, expression_gate_mlp = (
                self.scale_shift_table[None] + t.reshape(B, 6, -1)
            ).chunk(6, dim=1)
            audio_shift_msa, audio_scale_msa, audio_gate_msa, audio_shift_mlp, audio_scale_mlp, audio_gate_mlp = (
                self.scale_shift_table[None] + t.reshape(B, 6, -1)
            ).chunk(6, dim=1)
        else:
            motion_shift_msa, motion_scale_msa, motion_gate_msa, motion_shift_mlp, motion_scale_mlp, motion_gate_mlp\
                = self.motion_time_in(t.unsqueeze(1)).chunk(6, dim=-1)
            expression_shift_msa, expression_scale_msa, expression_gate_msa, expression_shift_mlp, expression_scale_mlp, expression_gate_mlp\
                = self.expression_time_in(t.unsqueeze(1)).chunk(6, dim=-1)
            audio_shift_msa, audio_scale_msa, audio_gate_msa, audio_shift_mlp, audio_scale_mlp, audio_gate_mlp\
                = self.audio_time_in(t.unsqueeze(1)).chunk(6, dim=-1)
        
        ##################################### The first layer #######################################33
        residual_motion = self.motion_norm1(motion)
        residual_expression = self.expression_norm1(expression)
        residual_audio = self.audio_norm1(audio)
        
        residual_motion = t2i_modulate(residual_motion, motion_shift_msa, motion_scale_msa)
        residual_expression = t2i_modulate(residual_expression, expression_shift_msa, expression_scale_msa)
        residual_audio = t2i_modulate(residual_audio, audio_shift_msa, audio_scale_msa)
        
        residual_motion, residual_expression, residual_audio =\
            self.attn(residual_motion, residual_expression, residual_audio, rotary_emb)
        
        residual_motion = motion_gate_msa * residual_motion
        residual_expression = expression_gate_msa * residual_expression
        residual_audio = audio_gate_msa * residual_audio
        
        motion = motion + self.motion_drop_path(residual_motion)
        expression = expression + self.expression_drop_path(residual_expression)
        audio = audio + self.audio_drop_path(residual_audio)
        
        ######################################## The second layer ########################################
        residual_motion, residual_expression, residual_audio =\
            self.cross_attn(residual_motion, residual_expression, residual_audio, txt, None, mask)

        motion = motion + self.motion_drop_path(residual_motion)
        expression = expression + self.expression_drop_path(residual_expression)
        audio = audio + self.audio_drop_path(residual_audio)
        
        ######################################## The third layer ########################################
        residual_motion = self.motion_norm2(residual_motion)
        residual_expression = self.expression_norm2(residual_expression)
        residual_audio = self.audio_norm2(residual_audio)
        
        residual_motion = t2i_modulate(residual_motion, motion_shift_mlp, motion_scale_mlp)
        residual_expression = t2i_modulate(residual_expression, expression_shift_mlp, expression_scale_mlp)
        residual_audio = t2i_modulate(residual_audio, audio_shift_mlp, audio_scale_mlp)
        
        residual_motion, residual_expression, residual_audio =\
            self.mlp(residual_motion, residual_expression, residual_audio)
        
        residual_motion = motion_gate_mlp * residual_motion
        residual_expression = expression_gate_mlp * residual_expression
        residual_audio = audio_gate_mlp * residual_audio
        
        motion = motion + self.motion_drop_path(residual_motion)
        expression = expression + self.expression_norm2(residual_expression)
        audio = audio + self.audio_drop_path(residual_audio)
        
        return motion, expression, audio


class SanaTripleStream(nn.Module):
    def __init__(
        self,
        motion_in_channels=4,
        expression_in_channels=4,
        audio_in_channels=4,
        hidden_size=1152,
        caption_channels=2304,
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
        **kwargs,
    ):
        super().__init__()
        self.motion_in_channels = motion_in_channels
        self.expression_in_channels = expression_in_channels
        self.audio_in_channels = audio_in_channels
        self.motion_out_channels = motion_in_channels
        self.expression_out_channels = expression_in_channels
        self.audio_out_channels = audio_in_channels
        
        self.hidden_size = hidden_size
        self.depth = depth
        self.pos_embed_type = pos_embed_type
        self.fp32_attention = kwargs.get("use_fp32_attention", False)

        if work_dir is not None:
            logger = get_root_logger(os.path.join(work_dir, "train_log.log"))
            logger = logger.info
        else:
            logger = print
        
        inputlayer_config["params"]["hidden_size"] = hidden_size
        inputlayer_config["params"]["in_channels"] = motion_in_channels
        self.motion_in = instantiate_from_config(inputlayer_config)
        inputlayer_config["params"]["in_channels"] = expression_in_channels
        self.expression_in = instantiate_from_config(inputlayer_config)
        inputlayer_config["params"]["in_channels"] = audio_in_channels
        self.audio_in = instantiate_from_config(inputlayer_config)
        
        timestep_embedder_config["params"]["hidden_size"] = hidden_size
        self.motion_t_embedder = instantiate_from_config(timestep_embedder_config)
        self.expression_t_embedder = instantiate_from_config(timestep_embedder_config)
        self.audio_t_embedder = instantiate_from_config(timestep_embedder_config)
        self.simple_modulation = simple_modulation
        
        self.use_pe = use_pe
        self.pe_max_length = pe_max_length
        # Will use fixed sin-cos embedding:
        self.register_buffer("pos_embed", torch.zeros(1, pe_max_length, hidden_size))
        if self.simple_modulation:
            self.t_block = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True))
        
        caption_embedder_config["params"]["in_channels"] = caption_channels
        caption_embedder_config["params"]["hidden_size"] = hidden_size
        self.txt_embedder = instantiate_from_config(caption_embedder_config)
        
        drop_path = [x.item() for x in torch.linspace(0, drop_path, depth)]  # stochastic depth decay rule
        block_config["params"]["hidden_size"] = hidden_size
        self.blocks = nn.ModuleList()
        for i in range(depth):
            block_config["params"]["drop_path"] = drop_path[i]
            self.blocks.append(instantiate_from_config(block_config))
        
        finallayer_config["params"]["hidden_size"] = hidden_size
        finallayer_config["params"]["out_channels"] = self.motion_out_channels
        self.motion_final_layer = instantiate_from_config(finallayer_config)
        finallayer_config["params"]["out_channels"] = self.expression_out_channels
        self.expression_final_layer = instantiate_from_config(finallayer_config)
        finallayer_config["params"]["out_channels"] = self.audio_out_channels
        self.audio_final_layer = instantiate_from_config(finallayer_config)

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
        expression, 
        audio,
        timestep_motion,
        timestep_expression,
        timestep_audio, 
        txt,
        txt_mask = None,
        **kwargs
    ):
        """
        Forward pass of Sana.
            motion: (N, T, D) tensor of motion
            expression: (N, T, D) tensor of expression
            audio: (N, T, D) tensor of audio
            txt: (N, T, D) tensor of text
        """
        motion = motion.to(self.dtype)
        expression = expression.to(self.dtype)
        audio = audio.to(self.dtype)
        timestep_motion = timestep_motion.to(self.dtype)
        timestep_expression = timestep_expression.to(self.dtype)
        timestep_audio = timestep_audio.to(self.dtype)
        pos_embed = self.pos_embed.to(self.dtype)
        txt = txt.to(self.dtype)
        txt_mask = txt_mask.to(self.dtype) if txt_mask is not None else txt_mask
        
        motion = self.motion_in(motion)
        expression = self.expression_in(expression)
        audio = self.audio_in(audio)
        
        pos_embed = None
        if self.use_pe:
            if self.pos_embed_type == "sincos":
                motion = motion + self.pos_embed[:,:motion.shape[1], :]  # (N, T, D)
                expression = expression + self.pos_embed[:,:expression.shape[1], :]
                audio = audio + self.pos_embed[:,:audio.shape[1], :]
            elif self.pos_embed_type == "rope":
                motion_pos_embed = self.pos_embed._prepare_latent_sequence_ids(motion.shape[0], motion.shape[1], motion.device, self.dtype)
                expression_pos_embed = self.pos_embed._prepare_latent_sequence_ids(
                    expression.shape[0], expression.shape[1], expression.device, self.dtype)
                audio_pos_embed = self.pos_embed._prepare_latent_sequence_ids(audio.shape[0], audio.shape[1], audio.device, self.dtype)
                pos_embed = torch.cat([motion_pos_embed, expression_pos_embed, audio_pos_embed], dim=0)
                pos_embed = self.pos_embed(pos_embed)
            elif self.pos_embed_type == "none":
                pass
            else:
                raise NotImplementedError(f"pos embed type {self.pos_embed_type} not implemented")
        motion_t = self.motion_t_embedder(timestep_motion)  # (N, D)
        expression_t = self.expression_t_embedder(timestep_expression)
        audio_t = self.audio_t_embedder(timestep_audio)
        t = motion_t + expression_t + audio_t
        if self.simple_modulation:
            t = self.t_block(t)
        
        txt = self.txt_embedder(txt)  # (N, L, D)
        if txt_mask is not None:
            txt_mask = txt_mask.to(torch.int16)
            txt_mask = txt_mask.repeat(txt.shape[0] // txt_mask.shape[0], *[1] * len(txt_mask.shape[1:])) \
                if txt_mask.shape[0] != txt.shape[0] else txt_mask
            txt_mask = txt_mask.squeeze(1).squeeze(1)
            y_lens = txt_mask
        else:
            y_lens = None
                
        for block in self.blocks:
            motion, expression, audio = auto_grad_checkpoint(block, motion, expression, audio, t, txt, pos_embed, y_lens)  # (N, T, D) #support grad checkpoint
        motion = self.motion_final_layer(motion, t)  # (N, T, patch_size ** 2 * out_channels)
        expression = self.expression_final_layer(expression, t)
        audio = self.audio_final_layer(audio, t)
        return {
            "motion": motion, 
            "expression": expression, 
            "audio": audio
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
        expression, 
        audio,
        timestep_motion,
        timestep_expression,
        timestep_audio, 
        txt,
        txt_mask,
        **kwargs
    ):
        """
        dpm solver donnot need variance prediction
        """
        # https://github.com/openai/glide-text2im/blob/main/notebooks/text2im.ipynb
        model_out = self.forward(
            motion,
            expression, 
            audio,
            timestep_motion,
            timestep_expression,
            timestep_audio, 
            txt,
            txt_mask,
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
    
        # # Initialize caption embedding MLP:
        # nn.init.normal_(self.y_embedder.y_proj.fc1.weight, std=0.02)
        # nn.init.normal_(self.y_embedder.y_proj.fc2.weight, std=0.02)

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
    model = SanaTripleStream(
        motion_in_channels=263,
        expression_in_channels=100,
        audio_in_channels=256,
        hidden_size=256,
        caption_channels=256,
        depth=12,
        drop_path = 0.2,
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
    )
    model.cuda()
    
    motion = torch.randn(4, 256, 263).cuda()
    expression = torch.randn(4, 256, 100).cuda()
    audio = torch.randn(4, 286, 256).cuda()
    timestep_motion = torch.randn(4).cuda()
    timestep_expression = torch.randn(4).cuda()
    timestep_audio = torch.randn(4).cuda()
    txt = torch.randn(4, 286, 256).cuda()
    results = model(
        motion,
        expression,
        audio,
        timestep_motion,
        timestep_expression,
        timestep_audio,
        txt
    )
    for key, val in results.items():
        print(key, val.shape)
    
    
    model = SanaTripleStream(
        motion_in_channels=263,
        expression_in_channels=100,
        audio_in_channels=256,
        hidden_size=256,
        caption_channels=256,
        depth=12,
        drop_path = 0.2,
        inputlayer_config=DEFAULT_INPUT_CONFIG,
        timestep_embedder_config=DEFAULT_TIMESTEP_EMBEDDER_CONFIG,
        caption_embedder_config=DEFAULT_CAPTION_EMBEDDER_CONFIG,
        block_config=DEFAULT_BLOCK_CONFIG,
        finallayer_config=DEFAULT_FINALLAYER_CONFIG,
        work_dir=None,
        use_pe=True,
        pos_embed_type="rope",
        pe_max_length=10000,
        simple_modulation=False, # for pixart-like only one modulation
    )
    model.cuda()
    
    results = model(
        motion,
        expression,
        audio,
        timestep_motion,
        timestep_expression,
        timestep_audio,
        txt
    )
    for key, val in results.items():
        print(key, val.shape)
    
    
    
    model = SanaTripleStream(
        motion_in_channels=263,
        expression_in_channels=100,
        audio_in_channels=256,
        hidden_size=256,
        caption_channels=256,
        depth=12,
        drop_path = 0.2,
        inputlayer_config=DEFAULT_INPUT_CONFIG,
        timestep_embedder_config=DEFAULT_TIMESTEP_EMBEDDER_CONFIG,
        caption_embedder_config=DEFAULT_CAPTION_EMBEDDER_CONFIG,
        block_config=dict(
            target="diffusion.model.nets.sana_triple_stream.SanaTripleStreamBlock",
            params=dict(
                hidden_size=None,
                num_heads=8,
                head_dim=None,
                simple_modulation=False,
                bias=True,
                norm_type="rms",
                attn_type="linear",
                qk_norm_type="identity",
                cross_attn_type="linear",
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
        ),
        finallayer_config=DEFAULT_FINALLAYER_CONFIG,
        work_dir=None,
        use_pe=True,
        pos_embed_type="rope",
        pe_max_length=10000,
        simple_modulation=False, # for pixart-like only one modulation
    )
    model.cuda()
    
    results = model(
        motion,
        expression,
        audio,
        timestep_motion,
        timestep_expression,
        timestep_audio,
        txt
    )
    for key, val in results.items():
        print(key, val.shape)
    
