# 实际上就是DiT
import os
import math
import numpy as np
import torch
import torch.nn as nn
from timm.layers import DropPath
import torch.nn.functional as F
from positional_encodings.torch_encodings import PositionalEncoding1D, Summer
from diffusion.model.nets.basic_modules import (
    TripleStreamMLP,
    TripleStreamGatedMLP,
    t2i_modulate
)
from diffusion.model.nets.attention_modules import (
    TripleStreamFlashAttention,
    TripleStreamLiteLA,
    FlashAttention,
    LiteLA,
    MultiHeadCrossAttention,
    CrossLiteLA,
)
from diffusion.model.nets.positional_embeding import RopePosEmbed
from torch.nn.attention.flex_attention import BlockMask, create_block_mask, create_mask
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
    target="diffusion.model.nets.causal.causal_dit.SanaEmageBlock",
    params=dict(
        hidden_size=None,
        num_heads=8,
        head_dim=None,
        simple_modulation=False,
        bias=True,
        norm_type="rms",
        attn_type=None,
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
        bias=True,
        norm_type="rms",
        proj_type="linear",
        simple_modulation=False,
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

        self.style_embed = StyleEmbedding(num_persons=31, dim=64)

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

class SanaEmageBlock(nn.Module):
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
        self.trans_norm1 = build_norm(norm_type, num_features=hidden_size, affine=False, eps=norm_eps)
        if attn_type == "linear":
            self.hand_attn = LiteLA(
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
            self.hand_attn = FlashAttention(
                dim=hidden_size,
                num_heads=num_heads,
                head_dim=head_dim,
                bias=bias,
                attn_type=attn_type,
                qk_norm_type=qk_norm_type,
                attn_drop=attn_drop, 
                proj_drop=attn_proj_drop,
            )
        
        self.body_attn = deepcopy(self.hand_attn)
        self.trans_attn = deepcopy(self.hand_attn)

        if prompt_attn_type == "linear":
            self.prompt_attn = TripleStreamLiteLA(
                dim=hidden_size,
                num_heads=num_heads,
                head_dim=head_dim,
                bias=bias,
                attn_type=attn_type,
                qk_norm_type=qk_norm_type,
                attn_drop=attn_drop, 
                proj_drop=attn_proj_drop,
            )
        elif prompt_attn_type == "triton_linear":
            # linear self attention with triton kernel fusion
            # TODO: Here the num_heads set to 36 for tmp used
            # TODO: implement multi-head attention with triton kernel fusion
            raise NotImplementedError(f"{attn_type} type is not implemented.")
        else:
            # for attn_type = None, rolling back to the vanlilla attention 
            self.prompt_attn = TripleStreamFlashAttention(
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
            self.hand_cross_attn = CrossLiteLA(
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
        self.trans_cross_attn = deepcopy(self.hand_cross_attn)
        
        self.hand_norm2 = build_norm(norm_type, num_features=hidden_size, affine=False, eps=norm_eps)
        self.body_norm2 = build_norm(norm_type, num_features=hidden_size, affine=False, eps=norm_eps)
        self.trans_norm2 = build_norm(norm_type, num_features=hidden_size, affine=False, eps=norm_eps)
        
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
            self.hand_scale_shift_table = nn.Parameter(torch.randn(6, hidden_size))
            self.body_scale_shift_table = nn.Parameter(torch.randn(6, hidden_size))
            self.trans_scale_shift_table = nn.Parameter(torch.randn(6, hidden_size))
        else:
            self.hand_time_in = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=bias))   
            self.body_time_in = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=bias))   
            self.trans_time_in = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=bias))
        
        self.hand_drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.body_drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.trans_drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        
    def initialize_weights(self):
        if self.simple_modulation:
            nn.init.constant(self.hand_scale_shift_table, 0)
            nn.init.constant(self.body_scale_shift_table, 0)
            nn.init.constant(self.trans_scale_shift_table, 0)
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
        trans_prompt: torch.FloatTensor = None,
        trans_prompt_mask: torch.FloatTensor = None,
        **kwargs
    ):
        assert hand.shape[0] == body.shape[0] == trans.shape[0], "batch size must be the same"
        B, N, C = hand.shape

        if self.simple_modulation:
            hand_shift_msa, hand_scale_msa, hand_gate_msa, hand_shift_mlp, hand_scale_mlp, hand_gate_mlp = (
                self.hand_scale_shift_table[None] + t.reshape(B, 6, -1)
            ).chunk(6, dim=1)
            body_shift_msa, body_scale_msa, body_gate_msa, body_shift_mlp, body_scale_mlp, body_gate_mlp = (
                self.body_scale_shift_table[None] + t.reshape(B, 6, -1)
            ).chunk(6, dim=1)
            trans_shift_msa, trans_scale_msa, trans_gate_msa, trans_shift_mlp, trans_scale_mlp, trans_gate_mlp = (
                self.trans_scale_shift_table[None] + t.reshape(B, 6, -1)
            ).chunk(6, dim=1)
        else:
            hand_shift_msa, hand_scale_msa, hand_gate_msa, hand_shift_mlp, hand_scale_mlp, hand_gate_mlp\
                = self.hand_time_in(t.unsqueeze(1)).chunk(6, dim=-1)
            body_shift_msa, body_scale_msa, body_gate_msa, body_shift_mlp, body_scale_mlp, body_gate_mlp\
                = self.body_time_in(t.unsqueeze(1)).chunk(6, dim=-1)
            trans_shift_msa, trans_scale_msa, trans_gate_msa, trans_shift_mlp, trans_scale_mlp, trans_gate_mlp\
                = self.trans_time_in(t.unsqueeze(1)).chunk(6, dim=-1)
        
        ##################################### The first layer #######################################33
        residual_hand = self.hand_norm1(hand)
        residual_body = self.body_norm1(body)
        residual_trans = self.trans_norm1(trans)
        
        residual_hand = t2i_modulate(residual_hand, hand_shift_msa, hand_scale_msa)
        residual_body = t2i_modulate(residual_body, body_shift_msa, body_scale_msa)
        residual_trans = t2i_modulate(residual_trans, trans_shift_msa, trans_scale_msa)

        residual_hand = self.hand_attn(residual_hand, rotary_emb)
        residual_body = self.body_attn(residual_body, rotary_emb)
        residual_trans = self.trans_attn(residual_trans, rotary_emb)
    
        residual_hand = hand_gate_msa * residual_hand
        residual_body = body_gate_msa * residual_body
        residual_trans = trans_gate_msa * residual_trans
        
        hand = hand + self.hand_drop_path(residual_hand)
        body = body + self.body_drop_path(residual_body)
        trans = trans + self.trans_drop_path(residual_trans)
        
        
        ######################################## The second layer ########################################
        residual_hand, residual_body, residual_trans \
            = self.prompt_attn(residual_hand, residual_body, residual_trans, None)
        
        hand = hand + self.hand_drop_path(residual_hand)
        body = body + self.body_drop_path(residual_body)
        trans = trans + self.trans_drop_path(residual_trans)
        
        ######################################## The forth layer ########################################
        residual_hand = self.hand_cross_attn(residual_hand, hand_prompt, None, hand_prompt_mask)
        residual_body = self.body_cross_attn(residual_body, body_prompt, None, body_prompt_mask)
        residual_trans = self.trans_cross_attn(residual_trans, trans_prompt, None, trans_prompt_mask)

        hand = hand + self.hand_drop_path(residual_hand)
        body = body + self.body_drop_path(residual_body)
        trans = trans + self.trans_drop_path(residual_trans)
        
        ######################################## The third layer ########################################
        residual_hand = self.hand_norm2(residual_hand)
        residual_body = self.body_norm2(residual_body)
        residual_trans = self.trans_norm2(residual_trans)
        
        residual_hand = t2i_modulate(residual_hand, hand_shift_mlp, hand_scale_mlp)
        residual_body = t2i_modulate(residual_body, body_shift_mlp, body_scale_mlp)
        residual_trans = t2i_modulate(residual_trans, trans_shift_mlp, trans_scale_mlp)
        
        residual_hand, residual_body, residual_trans =\
            self.mlp(residual_hand, residual_body, residual_trans)
        
        residual_hand = hand_gate_mlp * residual_hand
        residual_body = body_gate_mlp * residual_body
        residual_trans = trans_gate_mlp * residual_trans
        
        hand = hand + self.hand_drop_path(residual_hand)
        body = body + self.body_norm2(residual_body)
        trans = trans + self.trans_drop_path(residual_trans)
        
        return hand, body, trans


class CausalDITEmage(nn.Module):
    def __init__(
        self,
        motion_in_channels=4,
        expression_in_channels=100,
        hidden_size=1152,
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
        **kwargs,
    ):
        super().__init__()
        self.num_person = num_person
        self.motion_in_channels = motion_in_channels
        self.motion_out_channels = motion_in_channels
        
        self.hidden_size = hidden_size
        self.depth = depth
        self.pos_embed_type = pos_embed_type
        self.fp32_attention = kwargs.get("use_fp32_attention", False)

        if work_dir is not None:
            logger = get_root_logger(os.path.join(work_dir, "train_log.log"))
            logger = logger.info
        else:
            logger = print
        
        self.logger = logger
        inputlayer_config["params"]["hidden_size"] = hidden_size
        
        inputlayer_config["params"]["in_channels"] = 30 * 6
        self.hand_in = instantiate_from_config(inputlayer_config)
        inputlayer_config["params"]["in_channels"] = (13 + 9 + 3) * 6 + 3
        self.body_in = instantiate_from_config(inputlayer_config)
        
        
        inputlayer_config["params"]["in_channels"] = 30 * 6
        self.seed_hand_in = instantiate_from_config(inputlayer_config)
        inputlayer_config["params"]["in_channels"] = (13 + 9 + 3) * 6 + 3
        self.seed_body_in = instantiate_from_config(inputlayer_config)
        
        inputlayer_config["params"]["in_channels"] = 55 * 6 + 100 + 3
        self.xhand_in = instantiate_from_config(inputlayer_config)
        inputlayer_config["params"]["in_channels"] = 55 * 6 + 100 + 3
        self.xbody_in = instantiate_from_config(inputlayer_config)
        
        inputlayer_config["params"]["in_channels"] = hidden_size * 3
        self.hand_fuse = instantiate_from_config(inputlayer_config)
        inputlayer_config["params"]["in_channels"] = hidden_size * 3
        self.body_fuse = instantiate_from_config(inputlayer_config)
        
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
        self.txt_embedder = instantiate_from_config(caption_embedder_config)
        
        drop_path = [x.item() for x in torch.linspace(0, drop_path, depth)]  # stochastic depth decay rule
        block_config["params"]["simple_modulation"] = simple_modulation
        block_config["params"]["hidden_size"] = hidden_size
        self.blocks = nn.ModuleList()
        for i in range(depth):
            block_config["params"]["drop_path"] = drop_path[i]
            self.blocks.append(instantiate_from_config(block_config))
        
        finallayer_config["params"]["hidden_size"] = hidden_size
        finallayer_config["params"]["out_channels"] = 30 * 6
        self.hand_final_layer = instantiate_from_config(finallayer_config)
        finallayer_config["params"]["out_channels"] = (13 + 9 + 3) * 6
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
                f"autocast linear attn: {os.environ.get('AUTOCAST_LINEAR_ATTN', False)}"
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
    
    
    
    def forward_train(
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
        pos_embed = self.pos_embed.to(self.dtype)
        
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
        
        body_indices = np.concatenate([upper_indices, lower_indices, face_joint_indices], axis=0)
        #########################################################
        # for expression
        #########################################################
        
        hand = motion[..., hand_indices]
        body = motion[..., body_indices]
        trans = motion[..., trans_indices]
        body = torch.cat([body, trans], dim=-1)
        
        hand = self.hand_in(hand)
        body = self.body_in(body)
    
        x = torch.cat([motion, expression], dim=-1)
        xhand = self.xhand_in(x)
        xbody = self.xbody_in(x)
        
        seed_hand = pose_prompt[..., hand_indices]
        seed_body = pose_prompt[..., body_indices]
        
        seed_hand = self.seed_hand_in(seed_hand)
        seed_body = self.seed_body_in(seed_body)
        
        cond = self.txt_embedder(cond)  # (N, L, D)
        
        hand = self.hand_fuse(torch.cat([hand, xhand, cond], dim=-1))
        body = self.body_fuse(torch.cat([body, xbody, cond], dim=-1))
        
        t = self.motion_t_embedder(timestep)  # (N, D)
        id_embed = self.id_embedder(id)
        t = t + id_embed
        if self.simple_modulation:
            t0 = self.t_block(t)
        else:
            t0 = t
        
        pos_embed = None
        if self.use_pe:
            if self.pos_embed_type == "sincos":
                hand = hand + self.pos_embed[:,:hand.shape[1], :]  # (N, T, D)
                body = body + self.pos_embed[:,:body.shape[1], :]  # (N, T, D)
                seed_hand = seed_hand + self.pos_embed[:,:seed_hand.shape[1], :]  # (N, T, D)
                seed_body = seed_body + self.pos_embed[:,:seed_body.shape[1], :]  # (N, T, D)
            elif self.pos_embed_type == "rope":
                pos_embed = self.pos_embed._prepare_latent_sequence_ids(
                        hand.shape[0], hand.shape[1], hand.device, self.dtype)
                pos_embed = self.pos_embed(pos_embed)
            elif self.pos_embed_type == "none":
                pass
            else:
                raise NotImplementedError(f"pos embed type {self.pos_embed_type} not implemented")
        
                
        for block in self.blocks:
            hand, body = auto_grad_checkpoint(
                block, hand, body, t0, pos_embed, seed_hand, pose_prompt_mask, seed_body, pose_prompt_mask
            )  # (N, T, D) #support grad checkpoint
        
        hand = self.hand_final_layer(hand, t)
        body = self.body_final_layer(body, t)  

        trans = body[..., -3:]
        body = body[..., :-3]

        motion[..., hand_indices] = hand
        motion[..., body_indices] = body
        motion[..., trans_indices] = trans
        return {
            "motion": motion, 
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
    import imageio
    import cv2
    ## for teacher forcing
    block_mask = CausalDITEmage._prepare_teacher_forcing_mask(
        num_frames = 9,
        frame_seqlen = 256,
        device=torch.device("cuda"), 
        debug=True
    )
    print(block_mask)    
    block_mask = cv2.resize(block_mask[0, 0].cpu().float().numpy(), (1024, 1024))
    imageio.imwrite("teacher_forcing_mask_%d.jpg" % (0), np.uint8(255. * block_mask))
    ## for blockwise causal mask
    block_mask = CausalDITEmage._prepare_blockwise_causal_attn_mask(
        device=torch.device("cuda"), 
        num_frames = 9,
        frame_seqlen = 256,
        debug=True
    )
    print(block_mask)    
    block_mask = cv2.resize(block_mask[0, 0].cpu().float().numpy(), (1024, 1024))
    imageio.imwrite("blockwise_causal_mask_%d.jpg" % (0), np.uint8(255. * block_mask))
    ## for blockwise causal mask i2v
    block_mask = CausalDITEmage._prepare_blockwise_causal_attn_mask_i2v(
        device=torch.device("cuda"), 
        num_frames = 20,
        frame_seqlen = 256,
        num_frame_per_block = 4,
        debug=True
    )
    print(block_mask)    
    block_mask = cv2.resize(block_mask[0, 0].cpu().float().numpy(), (1024, 1024))
    imageio.imwrite("blockwise_causal_mask_i2v_%d.jpg" % (0), np.uint8(255. * block_mask))
    
