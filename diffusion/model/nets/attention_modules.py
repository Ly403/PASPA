import os
import torch
import numpy as np
from typing import Union, List, Tuple
import torch.nn as nn
import torch.nn.functional as F
from diffusion.model.norms import build_norm
from torch.nn.attention import SDPBackend, sdpa_kernel
from diffusion.model.act import build_act
from diffusion.model.nets.positional_embeding import apply_rotary_emb
from diffusion.model.nets.attention import attention, vanilla_attention

class TripleStreamMultiHeadCrossAttention(nn.Module):
    """Multi-head Flash Attention block with qk norm."""

    def __init__(
        self,
        dim,
        num_heads=8,
        head_dim=None,
        bias=True,
        attn_type=None,
        qk_norm_type="identity",
        attn_drop=0., 
        proj_drop=0.,
    ):
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads.
            qkv_bias (bool:  If True, add a learnable bias to query, key, value.
        """
        super().__init__()
        assert (num_heads is not None and head_dim is None) or (num_heads is None and head_dim is not None), \
            "only one of num_heads and head_dim should be specified"
        if num_heads is not None:
            assert dim % num_heads == 0, 'dim should be divisible by num_heads'
            self.num_heads = num_heads
        elif head_dim is not None:
            assert dim % head_dim == 0, 'head_dim should be divisible by dim'
            self.num_heads = dim // head_dim  
        
        assert attn_type in ["flash", "memory-efficient", "math", None]
        self.attn_type = attn_type
        
        self.m_q_linear = nn.Linear(dim, dim, bias=bias)
        self.e_q_linear = nn.Linear(dim, dim, bias=bias)
        self.a_q_linear = nn.Linear(dim, dim, bias=bias)
        self.kv_linear = nn.Linear(dim, 2 * dim, bias=bias)
        
        self.attn_drop = attn_drop
        
        self.motion_proj = nn.Linear(dim, dim, bias=bias)
        self.motion_proj_drop = nn.Dropout(proj_drop)
        
        self.expression_proj = nn.Linear(dim, dim, bias=bias)
        self.expression_proj_drop = nn.Dropout(proj_drop)
        
        self.audio_proj = nn.Linear(dim, dim, bias=bias)
        self.audio_proj_drop = nn.Dropout(proj_drop)
        
        self.mq_norm = build_norm(name=qk_norm_type, num_features=dim)
        self.eq_norm = build_norm(name=qk_norm_type, num_features=dim)
        self.aq_norm = build_norm(name=qk_norm_type, num_features=dim)
        self.k_norm = build_norm(name=qk_norm_type, num_features=dim)

    def forward(
        self, 
        motion: torch.FloatTensor,
        expression: torch.FloatTensor,
        audio: torch.FloatTensor,
        cond: torch.FloatTensor,
        rotary_emb: torch.FloatTensor = None,
        mask=None,
    ):
        mB, mN, mC = motion.shape
        eB, eN, eC = expression.shape
        aB, aN, aC = audio.shape
        cB, cN, cC = cond.shape
        assert mB == eB == aB == cB, "batch size must be equal"
        assert mC == eC == aC, "channel size must be equal"
        N = mN + eN + aN
        #####################################  for motion ######################################
        mq = self.m_q_linear(motion)
        dtype = mq.dtype
        mq = self.mq_norm(mq)
        mq = mq.reshape(mB, mN, self.num_heads, mC // self.num_heads).to(dtype)

        #################################### for expression #####################################
        eq = self.e_q_linear(expression)
        eq = self.eq_norm(eq)
        eq = eq.reshape(eB, eN, self.num_heads, eC // self.num_heads).to(dtype)

        #################################### for audio #####################################
        aq = self.a_q_linear(audio)
        aq = self.aq_norm(aq)
        aq = aq.reshape(aB, aN, self.num_heads, aC // self.num_heads).to(dtype)

        ################################### attention #########################################
        ################## for fusing ####################
        q = torch.cat([mq, eq, aq], dim=1)
        
        kv = self.kv_linear(cond).view(cB, cN, 2, cC)
        k, v = kv.unbind(2)
        k = self.k_norm(k).view(cB, -1, self.num_heads, cC // self.num_heads)
        v = v.view(cB, -1, self.num_heads, cC // self.num_heads)

        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        ##################################################
        if rotary_emb is not None:
            q = apply_rotary_emb(q, rotary_emb, use_real_unbind_dim=-1)
            k = apply_rotary_emb(k, rotary_emb, use_real_unbind_dim=-1)

        use_fp32_attention = getattr(self, "fp32_attention", False)  # necessary for NAN loss
        if use_fp32_attention:
            q, k, v = q.float(), k.float(), v.float()

        # attn_bias = None
        # if mask is not None:
        #     attn_bias = torch.zeros([mB * self.num_heads, q.shape[1], k.shape[1]], dtype=q.dtype, device=q.device)
        #     attn_bias.masked_fill_(mask.squeeze(1).repeat(self.num_heads, 1, 1) == 0, float("-inf"))

        if mask is not None and mask.ndim == 2:
            # mask = (1 - mask.to(dtype)) * -10000.0
            mask = mask.to(dtype) * -10000.0 # False -> -10000.0, True -> 0.0
            mask = mask[:, None, None].repeat(1, self.num_heads, 1, 1)
        
        if self.attn_type == "flash":
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=self.attn_drop, is_causal=False)
        elif self.attn_type == "memory-efficient":
            with sdpa_kernel(SDPBackend.MEMORY_EFFICIENT_ATTENTION):
                attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=self.attn_drop, is_causal=False)
        elif self.attn_type == "math":
            with sdpa_kernel(SDPBackend.MATH):
                attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=self.attn_drop, is_causal=False)
        else:
            attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=self.attn_drop, is_causal=False)
            
        attn_out = attn_out.transpose(1, 2)
        
        attn_out = attn_out.reshape(mB, N, mC)
        motion_out, expression_out, audio_out = \
            attn_out[:, 0:mN, ...], \
            attn_out[:, mN:mN + eN, ...], \
            attn_out[:, mN + eN:N, ...]
        
        motion_out = self.motion_proj(motion_out)
        motion_out = self.motion_proj_drop(motion_out)
        
        expression_out = self.expression_proj(expression_out)
        expression_out = self.expression_proj_drop(expression_out)
        
        audio_out = self.audio_proj(audio_out)
        audio_out = self.audio_proj_drop(audio_out)

        if torch.get_autocast_gpu_dtype() == torch.float16:
            motion_out = motion_out.clip(-65504, 65504)
            expression_out = expression_out.clip(-65504, 65504)
            audio_out = audio_out.clip(-65504, 65504)
        
        return motion_out, expression_out, audio_out


class DoubleStreamMultiHeadCrossAttention(nn.Module):
    """Multi-head Flash Attention block with qk norm."""

    def __init__(
        self,
        dim,
        num_heads=8,
        head_dim=None,
        bias=True,
        attn_type=None,
        qk_norm_type="identity",
        attn_drop=0., 
        proj_drop=0.,
        concat_on_sequence=True,
    ):
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads.
            qkv_bias (bool:  If True, add a learnable bias to query, key, value.
        """
        super(DoubleStreamMultiHeadCrossAttention, self).__init__()
        assert (num_heads is not None and head_dim is None) or (num_heads is None and head_dim is not None), \
            "only one of num_heads and head_dim should be specified"
        if num_heads is not None:
            assert dim % num_heads == 0, 'dim should be divisible by num_heads'
            self.num_heads = num_heads
        elif head_dim is not None:
            assert dim % head_dim == 0, 'head_dim should be divisible by dim'
            self.num_heads = dim // head_dim  
        
        assert attn_type in ["flash", "memory-efficient", "math", None]
        self.attn_type = attn_type
        self.concat_on_sequence = concat_on_sequence
        
        self.m_q_linear = nn.Linear(dim, dim, bias=bias)
        self.e_q_linear = nn.Linear(dim, dim, bias=bias)
        self.kv_linear = nn.Linear(dim, 2 * dim, bias=bias) if self.concat_on_sequence else nn.Linear(dim, 4 * dim, bias=bias)
        
        self.attn_drop = attn_drop
        
        self.motion_proj = nn.Linear(dim, dim, bias=bias)
        self.motion_proj_drop = nn.Dropout(proj_drop)
        
        self.expression_proj = nn.Linear(dim, dim, bias=bias)
        self.expression_proj_drop = nn.Dropout(proj_drop)
        
        self.mq_norm = build_norm(name=qk_norm_type, num_features=dim)
        self.eq_norm = build_norm(name=qk_norm_type, num_features=dim)
        self.k_norm = build_norm(name=qk_norm_type, num_features=dim)

    def forward(
        self, 
        motion: torch.FloatTensor,
        expression: torch.FloatTensor,
        cond: torch.FloatTensor,
        rotary_emb: torch.FloatTensor = None,
        mask=None,
    ):
        mB, mN, mC = motion.shape
        eB, eN, eC = expression.shape
        cB, cN, cC = cond.shape
        assert mB == eB == cB, "batch size must be equal"
        assert mC == eC, "channel size must be equal"
        if self.concat_on_sequence:
            N = mN + eN
            C = mC
        else:
            assert mN == eN, "motion and expression sequence length must be equal"
            N = mN
            C = mC + eC
            cC = cC * 2
        #####################################  for motion ######################################
        mq = self.m_q_linear(motion)
        dtype = mq.dtype
        mq = self.mq_norm(mq)
        mq = mq.reshape(mB, mN, self.num_heads, mC // self.num_heads).to(dtype)

        #################################### for expression #####################################
        eq = self.e_q_linear(expression)
        eq = self.eq_norm(eq)
        eq = eq.reshape(eB, eN, self.num_heads, eC // self.num_heads).to(dtype)

        ################################### attention #########################################
        ################## for fusing ####################
        if self.concat_on_sequence:
            q = torch.cat([mq, eq], dim=1)
        else:
            q = torch.cat([mq, eq], dim=-1)
        
        kv = self.kv_linear(cond).view(cB, cN, 2, cC)
        k, v = kv.unbind(2)
        k = self.k_norm(k).view(cB, -1, self.num_heads, cC // self.num_heads)
        v = v.view(cB, -1, self.num_heads, cC // self.num_heads)

        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        ##################################################
        if rotary_emb is not None:
            q = apply_rotary_emb(q, rotary_emb, use_real_unbind_dim=-1)
            k = apply_rotary_emb(k, rotary_emb, use_real_unbind_dim=-1)

        use_fp32_attention = getattr(self, "fp32_attention", False)  # necessary for NAN loss
        if use_fp32_attention:
            q, k, v = q.float(), k.float(), v.float()

        # attn_bias = None
        # if mask is not None:
        #     attn_bias = torch.zeros([mB * self.num_heads, q.shape[1], k.shape[1]], dtype=q.dtype, device=q.device)
        #     attn_bias.masked_fill_(mask.squeeze(1).repeat(self.num_heads, 1, 1) == 0, float("-inf"))

        if mask is not None and mask.ndim == 2:
            # mask = (1 - mask.to(dtype)) * -10000.0
            mask = mask.to(dtype) * -10000.0 # False -> -10000.0, True -> 0.0
            mask = mask[:, None, None].repeat(1, self.num_heads, 1, 1)
        
        if self.attn_type == "flash":
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=self.attn_drop, is_causal=False)
        elif self.attn_type == "memory-efficient":
            with sdpa_kernel(SDPBackend.MEMORY_EFFICIENT_ATTENTION):
                attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=self.attn_drop, is_causal=False)
        elif self.attn_type == "math":
            with sdpa_kernel(SDPBackend.MATH):
                attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=self.attn_drop, is_causal=False)
        else:
            attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=self.attn_drop, is_causal=False)
            
        attn_out = attn_out.transpose(1, 2)
        
        attn_out = attn_out.reshape(mB, N, C)
        if self.concat_on_sequence:
            
            motion_out, expression_out = \
                attn_out[:, 0:mN, ...], \
                attn_out[:, mN:N, ...]
        else:
            motion_out, expression_out = \
                attn_out[:, :, 0:mC], \
                attn_out[:, :, mC:]
        
        motion_out = self.motion_proj(motion_out)
        motion_out = self.motion_proj_drop(motion_out)
        
        expression_out = self.expression_proj(expression_out)
        expression_out = self.expression_proj_drop(expression_out)
        
        if torch.get_autocast_gpu_dtype() == torch.float16:
            motion_out = motion_out.clip(-65504, 65504)
            expression_out = expression_out.clip(-65504, 65504)
        
        return motion_out, expression_out


class MultiHeadCrossAttention(nn.Module):
    """Multi-head Flash Attention block with qk norm."""

    def __init__(
        self,
        dim,
        num_heads=8,
        head_dim=None,
        bias=True,
        attn_type=None,
        qk_norm_type="identity",
        attn_drop=0., 
        proj_drop=0.,
    ):
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads.
            qkv_bias (bool:  If True, add a learnable bias to query, key, value.
        """
        super(MultiHeadCrossAttention, self).__init__()
        assert (num_heads is not None and head_dim is None) or (num_heads is None and head_dim is not None), \
            "only one of num_heads and head_dim should be specified"
        if num_heads is not None:
            assert dim % num_heads == 0, 'dim should be divisible by num_heads'
            self.num_heads = num_heads
        elif head_dim is not None:
            assert dim % head_dim == 0, 'head_dim should be divisible by dim'
            self.num_heads = dim // head_dim  
        
        assert attn_type in ["flash", "memory-efficient", "math", None]
        self.attn_type = attn_type
        
        self.m_q_linear = nn.Linear(dim, dim, bias=bias)
        self.kv_linear = nn.Linear(dim, 2 * dim, bias=bias)
        
        self.attn_drop = attn_drop
        
        self.motion_proj = nn.Linear(dim, dim, bias=bias)
        self.motion_proj_drop = nn.Dropout(proj_drop)
        
        self.mq_norm = build_norm(name=qk_norm_type, num_features=dim)
        self.k_norm = build_norm(name=qk_norm_type, num_features=dim)

    def forward(
        self, 
        motion: torch.FloatTensor,
        cond: torch.FloatTensor,
        rotary_emb: torch.FloatTensor = None,
        mask=None,
    ):
        mB, mN, mC = motion.shape
        cB, cN, cC = cond.shape
        #####################################  for motion ######################################
        q = self.m_q_linear(motion)
        dtype = q.dtype
        q = self.mq_norm(q)
        q = q.reshape(mB, mN, self.num_heads, mC // self.num_heads).to(dtype)

        ################################### attention #########################################
        
        kv = self.kv_linear(cond).view(cB, cN, 2, cC)
        k, v = kv.unbind(2)
        k = self.k_norm(k).view(cB, -1, self.num_heads, cC // self.num_heads)
        v = v.view(cB, -1, self.num_heads, cC // self.num_heads)

        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        ##################################################
        if rotary_emb is not None:
            q = apply_rotary_emb(q, rotary_emb, use_real_unbind_dim=-1)
            k = apply_rotary_emb(k, rotary_emb, use_real_unbind_dim=-1)

        use_fp32_attention = getattr(self, "fp32_attention", False)  # necessary for NAN loss
        if use_fp32_attention:
            q, k, v = q.float(), k.float(), v.float()

        # attn_bias = None
        # if mask is not None:
        #     attn_bias = torch.zeros([mB * self.num_heads, q.shape[1], k.shape[1]], dtype=q.dtype, device=q.device)
        #     attn_bias.masked_fill_(mask.squeeze(1).repeat(self.num_heads, 1, 1) == 0, float("-inf"))

        if mask is not None and mask.ndim == 2:
            # mask = (1 - mask.to(dtype)) * -10000.0
            mask = mask.to(dtype) * -1e6 # True -> -1e6, False -> 0.0
            mask = mask[:, None, None].repeat(1, self.num_heads, 1, 1)
        
        if self.attn_type == "flash":
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=self.attn_drop, is_causal=False)
        elif self.attn_type == "memory-efficient":
            with sdpa_kernel(SDPBackend.MEMORY_EFFICIENT_ATTENTION):
                attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=self.attn_drop, is_causal=False)
        elif self.attn_type == "math":
            with sdpa_kernel(SDPBackend.MATH):
                attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=self.attn_drop, is_causal=False)
        else:
            attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=self.attn_drop, is_causal=False)
            
        attn_out = attn_out.transpose(1, 2)
        attn_out = attn_out.reshape(mB, mN, mC)
        
        motion_out = self.motion_proj(attn_out)
        motion_out = self.motion_proj_drop(motion_out)
        
        if torch.get_autocast_gpu_dtype() == torch.float16:
            motion_out = motion_out.clip(-65504, 65504)
        
        return motion_out

class TripleStreamCrossLiteLA(nn.Module):
    r"""Lightweight linear attention"""

    PAD_VAL = 1

    def __init__(
        self,
        dim,
        num_heads=8,
        head_dim=None,
        bias=True,
        attn_type=None,
        qk_norm_type="identity",
        kernel_act_type = "relu",
        attn_drop=0., 
        proj_drop=0.,
        eps=1e-6,
    ):        
        super().__init__()
        assert (num_heads is not None and head_dim is None) or (num_heads is None and head_dim is not None), \
            "only one of num_heads and head_dim should be specified"
        if num_heads is not None:
            assert dim % num_heads == 0, 'dim should be divisible by num_heads'
            self.num_heads = num_heads
        elif head_dim is not None:
            assert dim % head_dim == 0, 'head_dim should be divisible by dim'
            self.num_heads = dim // head_dim  
        
        self.attn_type = attn_type
        
        self.m_q_linear = nn.Linear(dim, dim, bias=bias)
        self.e_q_linear = nn.Linear(dim, dim, bias=bias)
        self.a_q_linear = nn.Linear(dim, dim, bias=bias)
        self.kv_linear = nn.Linear(dim, dim * 2, bias=bias)
        
        self.attn_drop = attn_drop
        
        self.motion_proj = nn.Linear(dim, dim, bias=bias)
        self.motion_proj_drop = nn.Dropout(proj_drop)
        
        self.expression_proj = nn.Linear(dim, dim, bias=bias)
        self.expression_proj_drop = nn.Dropout(proj_drop)
        
        self.audio_proj = nn.Linear(dim, dim, bias=bias)
        self.audio_proj_drop = nn.Dropout(proj_drop)
        
        self.mq_norm = build_norm(name=qk_norm_type, num_features=dim)    
        self.eq_norm = build_norm(name=qk_norm_type, num_features=dim)
        self.aq_norm = build_norm(name=qk_norm_type, num_features=dim)
        self.k_norm = build_norm(name=qk_norm_type, num_features=dim)

        self.eps = eps
        self.kernel_func = build_act(name=kernel_act_type, inplace=False)
    
    @torch.amp.autocast("cuda", enabled=os.environ.get("AUTOCAST_LINEAR_ATTN", False) == "true")
    def attn_matmul(self, q, k, v: torch.Tensor) -> torch.Tensor:
        # lightweight linear attention
        q = self.kernel_func(q)  # B, h, h_d, N
        k = self.kernel_func(k)

        use_fp32_attention = getattr(self, "fp32_attention", False)  # necessary for NAN loss
        if use_fp32_attention:
            q, k, v = q.float(), k.float(), v.float()

        v = F.pad(v, (0, 0, 0, 1), mode="constant", value=TripleStreamCrossLiteLA.PAD_VAL)
        vk = torch.matmul(v, k)
        out = torch.matmul(vk, q)

        if out.dtype in [torch.float16, torch.bfloat16]:
            out = out.float()
        out = out[:, :, :-1] / (out[:, :, -1:] + self.eps)

        return out

    def forward(
        self, 
        motion: torch.FloatTensor,
        expression: torch.FloatTensor,
        audio: torch.FloatTensor,
        cond: torch.FloatTensor,
        rotary_emb: torch.FloatTensor = None,
        mask=None, # TODO: support attention mask for linear attention
    ) -> torch.FloatTensor:
        mB, mN, mC = motion.shape
        eB, eN, eC = expression.shape
        aB, aN, aC = audio.shape
        cB, cN, cC = cond.shape
        assert mB == eB == aB == cB, "batch size must be equal"
        assert mC == eC == aC, "channel size must be equal"
        N = mN + eN + aN
        #####################################  for motion ######################################
        mq = self.m_q_linear(motion)
        dtype = mq.dtype
        mq = self.mq_norm(mq).transpose(-1, -2)  # (B, N, C) -> (B, C, N)
        mq = mq.reshape(mB, self.num_heads, mC // self.num_heads, mN).to(dtype)

        #################################### for expression #####################################
        eq = self.e_q_linear(expression)
        eq = self.eq_norm(eq)
        eq = eq.reshape(eB, self.num_heads, eC // self.num_heads, eN).to(dtype)

        #################################### for audio #####################################
        aq = self.a_q_linear(audio)
        aq = self.aq_norm(aq)
        aq = aq.reshape(aB, self.num_heads, aC // self.num_heads, aN).to(dtype)

        ################################### attention #########################################
        ################## for fusing ####################
        q = torch.cat([mq, eq, aq], dim=-1)
        
        kv = self.kv_linear(cond).view(cB, cN, 2, cC)
        k, v = kv.unbind(2)
        k = self.k_norm(k).transpose(-1, -2).reshape(cB, self.num_heads, cC // self.num_heads, -1)  # (B, N, C) -> (B, C, N)
        v = v.transpose(-1, -2).reshape(cB, self.num_heads, cC // self.num_heads, -1)  # (B, N, C) -> (B, C, N)
        ##################################################

        if rotary_emb is not None:
            q = apply_rotary_emb(q, rotary_emb, use_real_unbind_dim=-2)
            k = apply_rotary_emb(k, rotary_emb, use_real_unbind_dim=-2)

        attn_out = self.attn_matmul(q, k.transpose(-1, -2), v).to(dtype)

        attn_out = attn_out.view(mB, mC, N).permute(0, 2, 1)  # B, N, C
        
        motion_out, expression_out, audio_out = \
            attn_out[:, 0:mN, ...], \
            attn_out[:, mN:mN + eN, ...], \
            attn_out[:, mN + eN:N, ...]
            
        motion_out = self.motion_proj(motion_out)
        motion_out = self.motion_proj_drop(motion_out)
        
        expression_out = self.expression_proj(expression_out)
        expression_out = self.expression_proj_drop(expression_out)
        
        audio_out = self.audio_proj(audio_out)
        audio_out = self.audio_proj_drop(audio_out)

        if torch.get_autocast_gpu_dtype() == torch.float16:
            motion_out = motion_out.clip(-65504, 65504)
            expression_out = expression_out.clip(-65504, 65504)
            audio_out = audio_out.clip(-65504, 65504)

        return motion_out, expression_out, audio_out
    
    
class DoubleStreamCrossLiteLA(nn.Module):
    r"""Lightweight linear attention"""

    PAD_VAL = 1

    def __init__(
        self,
        dim,
        num_heads=8,
        head_dim=None,
        bias=True,
        attn_type=None,
        qk_norm_type="identity",
        kernel_act_type = "relu",
        attn_drop=0., 
        proj_drop=0.,
        eps=1e-6,
        concat_on_sequence=True,    
    ):        
        super(DoubleStreamCrossLiteLA, self).__init__()
        assert (num_heads is not None and head_dim is None) or (num_heads is None and head_dim is not None), \
            "only one of num_heads and head_dim should be specified"
        if num_heads is not None:
            assert dim % num_heads == 0, 'dim should be divisible by num_heads'
            self.num_heads = num_heads
        elif head_dim is not None:
            assert dim % head_dim == 0, 'head_dim should be divisible by dim'
            self.num_heads = dim // head_dim  
        
        self.attn_type = attn_type
        self.concat_on_sequence = concat_on_sequence
        self.m_q_linear = nn.Linear(dim, dim, bias=bias)
        self.e_q_linear = nn.Linear(dim, dim, bias=bias)
        self.kv_linear = nn.Linear(dim, dim * 2, bias=bias)
        
        self.attn_drop = attn_drop
        
        self.motion_proj = nn.Linear(dim, dim, bias=bias)
        self.motion_proj_drop = nn.Dropout(proj_drop)
        
        self.expression_proj = nn.Linear(dim, dim, bias=bias)
        self.expression_proj_drop = nn.Dropout(proj_drop)
        
        self.mq_norm = build_norm(name=qk_norm_type, num_features=dim)    
        self.eq_norm = build_norm(name=qk_norm_type, num_features=dim)
        self.k_norm = build_norm(name=qk_norm_type, num_features=dim)

        self.eps = eps
        self.kernel_func = build_act(name=kernel_act_type, inplace=False)
    
    @torch.amp.autocast("cuda", enabled=os.environ.get("AUTOCAST_LINEAR_ATTN", False) == "true")
    def attn_matmul(self, q, k, v: torch.Tensor) -> torch.Tensor:
        # lightweight linear attention
        q = self.kernel_func(q)  # B, h, h_d, N
        k = self.kernel_func(k)

        use_fp32_attention = getattr(self, "fp32_attention", False)  # necessary for NAN loss
        if use_fp32_attention:
            q, k, v = q.float(), k.float(), v.float()

        v = F.pad(v, (0, 0, 0, 1), mode="constant", value=TripleStreamCrossLiteLA.PAD_VAL)
        vk = torch.matmul(v, k)
        out = torch.matmul(vk, q)

        if out.dtype in [torch.float16, torch.bfloat16]:
            out = out.float()
        out = out[:, :, :-1] / (out[:, :, -1:] + self.eps)

        return out

    def forward(
        self, 
        motion: torch.FloatTensor,
        expression: torch.FloatTensor,
        cond: torch.FloatTensor,
        rotary_emb: torch.FloatTensor = None,
        mask=None, # TODO: support attention mask for linear attention
    ) -> torch.FloatTensor:
        mB, mN, mC = motion.shape
        eB, eN, eC = expression.shape
        cB, cN, cC = cond.shape
        assert mB == eB == cB, "batch size must be equal"
        assert mC == eC, "channel size must be equal"
        if self.concat_on_sequence:
            N = mN + eN
            C = mC
        else:
            assert mN == eN, "motion and expression sequence length must be equal"
            N = mN
            C = mC + eC
        #####################################  for motion ######################################
        mq = self.m_q_linear(motion)
        dtype = mq.dtype
        mq = self.mq_norm(mq).transpose(-1, -2)  # (B, N, C) -> (B, C, N)
        mq = mq.reshape(mB, self.num_heads, mC // self.num_heads, mN).to(dtype)

        #################################### for expression #####################################
        eq = self.e_q_linear(expression)
        eq = self.eq_norm(eq)
        eq = eq.reshape(eB, self.num_heads, eC // self.num_heads, eN).to(dtype)


        ################################### attention #########################################
        ################## for fusing ####################
        if self.concat_on_sequence:
            q = torch.cat([mq, eq], dim=-1)
        else:
            q = torch.cat([mq, eq], dim=1)
        
        kv = self.kv_linear(cond).view(cB, cN, 2, cC)
        k, v = kv.unbind(2)
        k = self.k_norm(k).transpose(-1, -2).reshape(cB, self.num_heads, cC // self.num_heads, -1)  # (B, N, C) -> (B, C, N)
        v = v.transpose(-1, -2).reshape(cB, self.num_heads, cC // self.num_heads, -1)  # (B, N, C) -> (B, C, N)
        ##################################################

        if rotary_emb is not None:
            q = apply_rotary_emb(q, rotary_emb, use_real_unbind_dim=-2)
            k = apply_rotary_emb(k, rotary_emb, use_real_unbind_dim=-2)

        attn_out = self.attn_matmul(q, k.transpose(-1, -2), v).to(dtype)
        attn_out = attn_out.view(mB, C, N).permute(0, 2, 1)  # B, N, C
        
        if self.concat_on_sequence:
            motion_out, expression_out = \
                attn_out[:, 0:mN, ...], \
                attn_out[:, mN:N, ...]
        else:
            motion_out, expression_out = \
                attn_out[:, :, 0:mC], \
                attn_out[:, :, mC:]
            
        motion_out = self.motion_proj(motion_out)
        motion_out = self.motion_proj_drop(motion_out)
        
        expression_out = self.expression_proj(expression_out)
        expression_out = self.expression_proj_drop(expression_out)
        
        if torch.get_autocast_gpu_dtype() == torch.float16:
            motion_out = motion_out.clip(-65504, 65504)
            expression_out = expression_out.clip(-65504, 65504)

        return motion_out, expression_out


class CrossLiteLA(nn.Module):
    r"""Lightweight linear attention"""

    PAD_VAL = 1

    def __init__(
        self,
        dim,
        num_heads=8,
        head_dim=None,
        bias=True,
        attn_type=None,
        qk_norm_type="identity",
        kernel_act_type = "relu",
        attn_drop=0., 
        proj_drop=0.,
        eps=1e-6,
    ):        
        super().__init__()
        assert (num_heads is not None and head_dim is None) or (num_heads is None and head_dim is not None), \
            "only one of num_heads and head_dim should be specified"
        if num_heads is not None:
            assert dim % num_heads == 0, 'dim should be divisible by num_heads'
            self.num_heads = num_heads
        elif head_dim is not None:
            assert dim % head_dim == 0, 'head_dim should be divisible by dim'
            self.num_heads = dim // head_dim  
        
        self.attn_type = attn_type
        
        self.m_q_linear = nn.Linear(dim, dim, bias=bias)
        self.kv_linear = nn.Linear(dim, dim * 2, bias=bias)
        
        self.attn_drop = attn_drop
        self.motion_proj = nn.Linear(dim, dim, bias=bias)
        self.motion_proj_drop = nn.Dropout(proj_drop)
        self.mq_norm = build_norm(name=qk_norm_type, num_features=dim)    
        self.k_norm = build_norm(name=qk_norm_type, num_features=dim)

        self.eps = eps
        self.kernel_func = build_act(name=kernel_act_type, inplace=False)
    
    @torch.amp.autocast("cuda", enabled=os.environ.get("AUTOCAST_LINEAR_ATTN", False) == "true")
    def attn_matmul(self, q, k, v: torch.Tensor) -> torch.Tensor:
        # lightweight linear attention
        q = self.kernel_func(q)  # B, h, h_d, N
        k = self.kernel_func(k)

        use_fp32_attention = getattr(self, "fp32_attention", False)  # necessary for NAN loss
        if use_fp32_attention:
            q, k, v = q.float(), k.float(), v.float()

        v = F.pad(v, (0, 0, 0, 1), mode="constant", value=TripleStreamCrossLiteLA.PAD_VAL)
        vk = torch.matmul(v, k)
        out = torch.matmul(vk, q)

        if out.dtype in [torch.float16, torch.bfloat16]:
            out = out.float()
        out = out[:, :, :-1] / (out[:, :, -1:] + self.eps)

        return out

    def forward(
        self, 
        motion: torch.FloatTensor,
        cond: torch.FloatTensor,
        rotary_emb: torch.FloatTensor = None,
        mask=None, # TODO: support attention mask for linear attention
    ) -> torch.FloatTensor:
        mB, mN, mC = motion.shape
        cB, cN, cC = cond.shape
        #####################################  for motion ######################################
        q = self.m_q_linear(motion)
        dtype = q.dtype
        q = self.mq_norm(q).transpose(-1, -2)  # (B, N, C) -> (B, C, N)
        q = q.reshape(mB, self.num_heads, mC // self.num_heads, mN).to(dtype)
        ################################### attention #########################################
        
        kv = self.kv_linear(cond).view(cB, cN, 2, cC)
        k, v = kv.unbind(2)
        k = self.k_norm(k).transpose(-1, -2).reshape(cB, self.num_heads, cC // self.num_heads, -1)  # (B, N, C) -> (B, C, N)
        v = v.transpose(-1, -2).reshape(cB, self.num_heads, cC // self.num_heads, -1)  # (B, N, C) -> (B, C, N)
        ##################################################

        if rotary_emb is not None:
            q = apply_rotary_emb(q, rotary_emb, use_real_unbind_dim=-2)
            k = apply_rotary_emb(k, rotary_emb, use_real_unbind_dim=-2)

        attn_out = self.attn_matmul(q, k.transpose(-1, -2), v).to(dtype)
        attn_out = attn_out.view(mB, mC, mN).permute(0, 2, 1)  # B, N, C
                    
        motion_out = self.motion_proj(attn_out)
        motion_out = self.motion_proj_drop(motion_out)
        if torch.get_autocast_gpu_dtype() == torch.float16:
            motion_out = motion_out.clip(-65504, 65504)

        return motion_out

class TripleStreamLiteLA(nn.Module):
    r"""Lightweight linear attention"""

    PAD_VAL = 1

    def __init__(
        self,
        dim,
        num_heads=8,
        head_dim=None,
        bias=True,
        attn_type=None,
        qk_norm_type="identity",
        kernel_act_type = "relu",
        attn_drop=0., 
        proj_drop=0.,
        eps: float = 1e-6,
    ):        
        super(TripleStreamLiteLA, self).__init__()
        assert (num_heads is not None and head_dim is None) or (num_heads is None and head_dim is not None), \
            "only one of num_heads and head_dim should be specified"
        if num_heads is not None:
            assert dim % num_heads == 0, 'dim should be divisible by num_heads'
            self.num_heads = num_heads
        elif head_dim is not None:
            assert dim % head_dim == 0, 'head_dim should be divisible by dim'
            self.num_heads = dim // head_dim  
        
        self.attn_type = attn_type
        
        self.m_qkv = nn.Linear(dim, dim * 3, bias=bias)
        self.e_qkv = nn.Linear(dim, dim * 3, bias=bias)
        self.a_qkv = nn.Linear(dim, dim * 3, bias=bias)
        
        self.attn_drop = attn_drop
        
        self.motion_proj = nn.Linear(dim, dim, bias=bias)
        self.motion_proj_drop = nn.Dropout(proj_drop)
        
        self.expression_proj = nn.Linear(dim, dim, bias=bias)
        self.expression_proj_drop = nn.Dropout(proj_drop)
        
        self.audio_proj = nn.Linear(dim, dim, bias=bias)
        self.audio_proj_drop = nn.Dropout(proj_drop)
        
        self.mq_norm = build_norm(name=qk_norm_type, num_features=dim)
        self.mk_norm = build_norm(name=qk_norm_type, num_features=dim)
        
        self.eq_norm = build_norm(name=qk_norm_type, num_features=dim)
        self.ek_norm = build_norm(name=qk_norm_type, num_features=dim)
        
        self.aq_norm = build_norm(name=qk_norm_type, num_features=dim)
        self.ak_norm = build_norm(name=qk_norm_type, num_features=dim)

        self.kernel_func = build_act(name=kernel_act_type, inplace=False)
        self.eps = eps
    
    @torch.amp.autocast("cuda", enabled=os.environ.get("AUTOCAST_LINEAR_ATTN", False) == "true")
    def attn_matmul(self, q, k, v: torch.Tensor) -> torch.Tensor:
        # lightweight linear attention
        q = self.kernel_func(q)  # B, h, h_d, N
        k = self.kernel_func(k)

        use_fp32_attention = getattr(self, "fp32_attention", False)  # necessary for NAN loss
        if use_fp32_attention:
            q, k, v = q.float(), k.float(), v.float()

        v = F.pad(v, (0, 0, 0, 1), mode="constant", value=TripleStreamLiteLA.PAD_VAL)
        vk = torch.matmul(v, k)
        out = torch.matmul(vk, q)

        if out.dtype in [torch.float16, torch.bfloat16]:
            out = out.float()
        out = out[:, :, :-1] / (out[:, :, -1:] + self.eps)

        return out

    def forward(
        self, 
        motion: torch.FloatTensor,
        expression: torch.FloatTensor,
        audio: torch.FloatTensor,
        rotary_emb: torch.FloatTensor = None,
        mask=None,
    ) -> torch.FloatTensor:
        mB, mN, mC = motion.shape
        eB, eN, eC = expression.shape
        aB, aN, aC = audio.shape
        assert mB == eB == aB, "batch size must be equal"
        assert mC == eC == aC, "channel size must be equal"
        N = mN + eN + aN
        #####################################  for motion ######################################
        m_qkv = self.m_qkv(motion).view(mB, mN, 3, mC)
        mq, mk, mv = m_qkv.unbind(2)
        dtype = mq.dtype

        mq = self.mq_norm(mq).transpose(-1, -2)  # (B, N, C) -> (B, C, N)
        mk = self.mk_norm(mk).transpose(-1, -2)  # (B, N, C) -> (B, C, N)
        mv = mv.transpose(-1, -2)  # (B, N, C) -> (B, C, N)
        
        mq = mq.reshape(mB, self.num_heads, mC // self.num_heads, mN).to(dtype)
        mk = mk.reshape(mB, self.num_heads, mC // self.num_heads, mN).to(dtype)
        mv = mv.reshape(mB, self.num_heads, mC // self.num_heads, mN).to(dtype)

        #################################### for expression #####################################
        e_qkv = self.e_qkv(expression).view(eB, eN, 3, eC)
        eq, ek, ev = e_qkv.unbind(2)

        eq = self.eq_norm(eq).transpose(-1, -2)  # (B, N, C) -> (B, C, N)
        ek = self.ek_norm(ek).transpose(-1, -2)  # (B, N, C) -> (B, C, N)
        ev = ev.transpose(-1, -2) # (B, N, C) -> (B, C, N)

        eq = eq.reshape(eB, self.num_heads, eC // self.num_heads, eN).to(dtype)
        ek = ek.reshape(eB, self.num_heads, eC // self.num_heads, eN).to(dtype)
        ev = ev.reshape(eB, self.num_heads, eC // self.num_heads, eN).to(dtype)

        #################################### for audio #####################################
        a_qkv = self.a_qkv(audio).view(aB, aN, 3, aC)
        aq, ak, av = a_qkv.unbind(2)

        aq = self.aq_norm(aq).transpose(-1, -2)  # (B, N, C) -> (B, C, N)
        ak = self.ak_norm(ak).transpose(-1, -2)  # (B, N, C) -> (B, C, N)
        av = av.transpose(-1, -2)  # (B, N, C) -> (B, C, N)

        aq = aq.reshape(aB, self.num_heads, aC // self.num_heads, aN).to(dtype)
        ak = ak.reshape(aB, self.num_heads, aC // self.num_heads, aN).to(dtype)
        av = av.reshape(aB, self.num_heads, aC // self.num_heads, aN).to(dtype)

        ################################### attention #########################################
        ################## for fusing ####################
        q = torch.cat([mq, eq, aq], dim=-1)
        k = torch.cat([mk, ek, ak], dim=-1)
        v = torch.cat([mv, ev, av], dim=-1)
        ##################################################

        if rotary_emb is not None:
            q = apply_rotary_emb(q, rotary_emb, use_real_unbind_dim=-2)
            k = apply_rotary_emb(k, rotary_emb, use_real_unbind_dim=-2)

        attn_out = self.attn_matmul(q, k.transpose(-1, -2), v).to(dtype)
        attn_out = attn_out.view(mB, mC, N).permute(0, 2, 1)  # B, N, C
        
        motion_out, expression_out, audio_out = \
            attn_out[:, 0:mN, ...], \
            attn_out[:, mN:mN + eN, ...], \
            attn_out[:, mN + eN:N, ...]
            
        motion_out = self.motion_proj(motion_out)
        motion_out = self.motion_proj_drop(motion_out)
        
        expression_out = self.expression_proj(expression_out)
        expression_out = self.expression_proj_drop(expression_out)
        
        audio_out = self.audio_proj(audio_out)
        audio_out = self.audio_proj_drop(audio_out)

        if torch.get_autocast_gpu_dtype() == torch.float16:
            motion_out = motion_out.clip(-65504, 65504)
            expression_out = expression_out.clip(-65504, 65504)
            audio_out = audio_out.clip(-65504, 65504)

        return motion_out, expression_out, audio_out

class DoubleStreamLiteLA(nn.Module):
    r"""Lightweight linear attention"""

    PAD_VAL = 1
    
    def __init__(
        self,
        dim,
        num_heads=8,
        head_dim=None,
        bias=True,
        attn_type=None,
        qk_norm_type="identity",
        kernel_act_type = "relu",
        attn_drop=0., 
        proj_drop=0.,
        eps: float = 1e-6,
        concat_on_sequence=True,
    ):        
        super(DoubleStreamLiteLA, self).__init__()
        assert (num_heads is not None and head_dim is None) or (num_heads is None and head_dim is not None), \
            "only one of num_heads and head_dim should be specified"
        if num_heads is not None:
            assert dim % num_heads == 0, 'dim should be divisible by num_heads'
            self.num_heads = num_heads
        elif head_dim is not None:
            assert dim % head_dim == 0, 'head_dim should be divisible by dim'
            self.num_heads = dim // head_dim  
        
        self.attn_type = attn_type
        self.concat_on_sequence = concat_on_sequence
        self.m_qkv = nn.Linear(dim, dim * 3, bias=bias)
        self.e_qkv = nn.Linear(dim, dim * 3, bias=bias)
        
        self.attn_drop = attn_drop
        
        self.motion_proj = nn.Linear(dim, dim, bias=bias)
        self.motion_proj_drop = nn.Dropout(proj_drop)
        
        self.expression_proj = nn.Linear(dim, dim, bias=bias)
        self.expression_proj_drop = nn.Dropout(proj_drop)
        
        self.mq_norm = build_norm(name=qk_norm_type, num_features=dim)
        self.mk_norm = build_norm(name=qk_norm_type, num_features=dim)
        
        self.eq_norm = build_norm(name=qk_norm_type, num_features=dim)
        self.ek_norm = build_norm(name=qk_norm_type, num_features=dim)
        
        self.kernel_func = build_act(name=kernel_act_type, inplace=False)
        self.eps = eps
    
    @torch.amp.autocast("cuda", enabled=os.environ.get("AUTOCAST_LINEAR_ATTN", False) == "true")
    def attn_matmul(self, q, k, v: torch.Tensor) -> torch.Tensor:
        # lightweight linear attention
        q = self.kernel_func(q)  # B, h, h_d, N
        k = self.kernel_func(k)

        use_fp32_attention = getattr(self, "fp32_attention", False)  # necessary for NAN loss
        if use_fp32_attention:
            q, k, v = q.float(), k.float(), v.float()

        v = F.pad(v, (0, 0, 0, 1), mode="constant", value=TripleStreamLiteLA.PAD_VAL)
        vk = torch.matmul(v, k)
        out = torch.matmul(vk, q)

        if out.dtype in [torch.float16, torch.bfloat16]:
            out = out.float()
        out = out[:, :, :-1] / (out[:, :, -1:] + self.eps)

        return out

    def forward(
        self, 
        motion: torch.FloatTensor,
        expression: torch.FloatTensor,
        rotary_emb: torch.FloatTensor = None,
        mask=None,
    ) -> torch.FloatTensor:
        mB, mN, mC = motion.shape
        eB, eN, eC = expression.shape
        assert mB == eB, "batch size must be equal"
        assert mC == eC, "channel size must be equal"
        if self.concat_on_sequence:
            N = mN + eN
            C = mC
        else:
            assert mN == eN, "motion and expression sequence length must be equal"
            N = mN
            C = mC + eC
        #####################################  for motion ######################################
        m_qkv = self.m_qkv(motion).view(mB, mN, 3, mC)
        mq, mk, mv = m_qkv.unbind(2)
        dtype = mq.dtype

        mq = self.mq_norm(mq).transpose(-1, -2)  # (B, N, C) -> (B, C, N)
        mk = self.mk_norm(mk).transpose(-1, -2)  # (B, N, C) -> (B, C, N)
        mv = mv.transpose(-1, -2)  # (B, N, C) -> (B, C, N)
        
        mq = mq.reshape(mB, self.num_heads, mC // self.num_heads, mN).to(dtype)
        mk = mk.reshape(mB, self.num_heads, mC // self.num_heads, mN).to(dtype)
        mv = mv.reshape(mB, self.num_heads, mC // self.num_heads, mN).to(dtype)

        #################################### for expression #####################################
        e_qkv = self.e_qkv(expression).view(eB, eN, 3, eC)
        eq, ek, ev = e_qkv.unbind(2)

        eq = self.eq_norm(eq).transpose(-1, -2)  # (B, N, C) -> (B, C, N)
        ek = self.ek_norm(ek).transpose(-1, -2)  # (B, N, C) -> (B, C, N)
        ev = ev.transpose(-1, -2) # (B, N, C) -> (B, C, N)

        eq = eq.reshape(eB, self.num_heads, eC // self.num_heads, eN).to(dtype)
        ek = ek.reshape(eB, self.num_heads, eC // self.num_heads, eN).to(dtype)
        ev = ev.reshape(eB, self.num_heads, eC // self.num_heads, eN).to(dtype)


        ################################### attention #########################################
        ################## for fusing ####################
        if self.concat_on_sequence:
            q = torch.cat([mq, eq], dim=-1)
            k = torch.cat([mk, ek], dim=-1)
            v = torch.cat([mv, ev], dim=-1)
        else:
            q = torch.cat([mq, eq], dim=1)
            k = torch.cat([mk, ek], dim=1)
            v = torch.cat([mv, ev], dim=1)
        ##################################################

        if rotary_emb is not None:
            q = apply_rotary_emb(q, rotary_emb, use_real_unbind_dim=-2)
            k = apply_rotary_emb(k, rotary_emb, use_real_unbind_dim=-2)

        attn_out = self.attn_matmul(q, k.transpose(-1, -2), v).to(dtype)
        attn_out = attn_out.view(mB, C, N).permute(0, 2, 1)  # B, N, C
        
        if self.concat_on_sequence:
            motion_out, expression_out = \
            attn_out[:, 0:mN, ...], \
            attn_out[:, mN:N, ...]
        else:
            motion_out, expression_out = \
                attn_out[:, :, 0:mC], \
                attn_out[:, :, mC:]
            
        motion_out = self.motion_proj(motion_out)
        motion_out = self.motion_proj_drop(motion_out)
        
        expression_out = self.expression_proj(expression_out)
        expression_out = self.expression_proj_drop(expression_out)
        
        if torch.get_autocast_gpu_dtype() == torch.float16:
            motion_out = motion_out.clip(-65504, 65504)
            expression_out = expression_out.clip(-65504, 65504)

        return motion_out, expression_out
    
class LiteLA(nn.Module):
    r"""Lightweight linear attention"""

    PAD_VAL = 1
    
    def __init__(
        self,
        dim,
        num_heads=8,
        head_dim=None,
        bias=True,
        attn_type=None,
        qk_norm_type="identity",
        kernel_act_type = "relu",
        attn_drop=0., 
        proj_drop=0.,
        eps: float = 1e-6,
    ):        
        super(LiteLA, self).__init__()
        assert (num_heads is not None and head_dim is None) or (num_heads is None and head_dim is not None), \
            "only one of num_heads and head_dim should be specified"
        if num_heads is not None:
            assert dim % num_heads == 0, 'dim should be divisible by num_heads'
            self.num_heads = num_heads
        elif head_dim is not None:
            assert dim % head_dim == 0, 'head_dim should be divisible by dim'
            self.num_heads = dim // head_dim  
        
        self.attn_type = attn_type
        
        self.m_qkv = nn.Linear(dim, dim * 3, bias=bias)
        self.attn_drop = attn_drop        
        self.motion_proj = nn.Linear(dim, dim, bias=bias)
        self.motion_proj_drop = nn.Dropout(proj_drop)
                
        self.mq_norm = build_norm(name=qk_norm_type, num_features=dim)
        self.mk_norm = build_norm(name=qk_norm_type, num_features=dim)
        
        self.kernel_func = build_act(name=kernel_act_type, inplace=False)
        self.eps = eps
    
    @torch.amp.autocast("cuda", enabled=os.environ.get("AUTOCAST_LINEAR_ATTN", False) == "true")
    def attn_matmul(self, q, k, v: torch.Tensor) -> torch.Tensor:
        # lightweight linear attention
        q = self.kernel_func(q)  # B, h, h_d, N
        k = self.kernel_func(k)

        use_fp32_attention = getattr(self, "fp32_attention", False)  # necessary for NAN loss
        if use_fp32_attention:
            q, k, v = q.float(), k.float(), v.float()

        v = F.pad(v, (0, 0, 0, 1), mode="constant", value=TripleStreamLiteLA.PAD_VAL)
        vk = torch.matmul(v, k)
        out = torch.matmul(vk, q)

        if out.dtype in [torch.float16, torch.bfloat16]:
            out = out.float()
        out = out[:, :, :-1] / (out[:, :, -1:] + self.eps)

        return out

    def forward(
        self, 
        motion: torch.FloatTensor,
        rotary_emb: torch.FloatTensor = None,
        mask=None,
    ) -> torch.FloatTensor:
        mB, mN, mC = motion.shape
        #####################################  for motion ######################################
        qkv = self.m_qkv(motion).view(mB, mN, 3, mC)
        q, k, v = qkv.unbind(2)
        dtype = q.dtype

        q = self.mq_norm(q).transpose(-1, -2)  # (B, N, C) -> (B, C, N)
        k = self.mk_norm(k).transpose(-1, -2)  # (B, N, C) -> (B, C, N)
        v = v.transpose(-1, -2)  # (B, N, C) -> (B, C, N)
        
        q = q.reshape(mB, self.num_heads, mC // self.num_heads, mN).to(dtype)
        k = k.reshape(mB, self.num_heads, mC // self.num_heads, mN).to(dtype)
        v = v.reshape(mB, self.num_heads, mC // self.num_heads, mN).to(dtype)

        ################################### attention #########################################
        ##################################################

        if rotary_emb is not None:
            q = apply_rotary_emb(q, rotary_emb, use_real_unbind_dim=-2)
            k = apply_rotary_emb(k, rotary_emb, use_real_unbind_dim=-2)

        attn_out = self.attn_matmul(q, k.transpose(-1, -2), v).to(dtype)
        attn_out = attn_out.view(mB, mC, mN).permute(0, 2, 1)  # B, N, C
        
        motion_out = self.motion_proj(attn_out)
        motion_out = self.motion_proj_drop(motion_out)
        
        if torch.get_autocast_gpu_dtype() == torch.float16:
            motion_out = motion_out.clip(-65504, 65504)

        return motion_out
    
class TripleStreamFlashAttention(nn.Module):
    """Multi-head Flash Attention block with qk norm."""

    def __init__(
        self,
        dim,
        num_heads=8,
        head_dim=None,
        bias=True,
        attn_type=None,
        qk_norm_type="identity",
        attn_drop=0., 
        proj_drop=0.,
    ):
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads.
            qkv_bias (bool:  If True, add a learnable bias to query, key, value.
        """
        super(TripleStreamFlashAttention, self).__init__()
        assert (num_heads is not None and head_dim is None) or (num_heads is None and head_dim is not None), \
            "only one of num_heads and head_dim should be specified"
        if num_heads is not None:
            assert dim % num_heads == 0, 'dim should be divisible by num_heads'
            self.num_heads = num_heads
        elif head_dim is not None:
            assert dim % head_dim == 0, 'head_dim should be divisible by dim'
            self.num_heads = dim // head_dim  
        
        assert attn_type in ["flash", "memory-efficient", "math", None]
        self.attn_type = attn_type
        
        self.m_qkv = nn.Linear(dim, dim * 3, bias=bias)
        self.e_qkv = nn.Linear(dim, dim * 3, bias=bias)
        self.a_qkv = nn.Linear(dim, dim * 3, bias=bias)
        
        self.attn_drop = attn_drop
        
        self.motion_proj = nn.Linear(dim, dim, bias=bias)
        self.motion_proj_drop = nn.Dropout(proj_drop)
        
        self.expression_proj = nn.Linear(dim, dim, bias=bias)
        self.expression_proj_drop = nn.Dropout(proj_drop)
        
        self.audio_proj = nn.Linear(dim, dim, bias=bias)
        self.audio_proj_drop = nn.Dropout(proj_drop)
        
        self.mq_norm = build_norm(name=qk_norm_type, num_features=dim)
        self.mk_norm = build_norm(name=qk_norm_type, num_features=dim)
        
        self.eq_norm = build_norm(name=qk_norm_type, num_features=dim)
        self.ek_norm = build_norm(name=qk_norm_type, num_features=dim)
        
        self.aq_norm = build_norm(name=qk_norm_type, num_features=dim)
        self.ak_norm = build_norm(name=qk_norm_type, num_features=dim)

    def forward(
        self, 
        motion: torch.FloatTensor,
        expression: torch.FloatTensor,
        audio: torch.FloatTensor,
        rotary_emb: torch.FloatTensor = None,
        mask=None,
    ):
        mB, mN, mC = motion.shape
        eB, eN, eC = expression.shape
        aB, aN, aC = audio.shape
        assert mB == eB == aB, "batch size must be equal"
        assert mC == eC == aC, "channel size must be equal"
        N = mN + eN + aN
        #####################################  for motion ######################################
        m_qkv = self.m_qkv(motion).view(mB, mN, 3, mC)
        mq, mk, mv = m_qkv.unbind(2)
        dtype = mq.dtype

        mq = self.mq_norm(mq)
        mk = self.mk_norm(mk)

        mq = mq.reshape(mB, mN, self.num_heads, mC // self.num_heads).to(dtype)
        mk = mk.reshape(mB, mN, self.num_heads, mC // self.num_heads).to(dtype)
        mv = mv.reshape(mB, mN, self.num_heads, mC // self.num_heads).to(dtype)

        #################################### for expression #####################################
        e_qkv = self.e_qkv(expression).view(eB, eN, 3, eC)
        eq, ek, ev = e_qkv.unbind(2)

        eq = self.eq_norm(eq)
        ek = self.ek_norm(ek)

        eq = eq.reshape(eB, eN, self.num_heads, eC // self.num_heads).to(dtype)
        ek = ek.reshape(eB, eN, self.num_heads, eC // self.num_heads).to(dtype)
        ev = ev.reshape(eB, eN, self.num_heads, eC // self.num_heads).to(dtype)

        #################################### for audio #####################################
        a_qkv = self.a_qkv(audio).view(aB, aN, 3, aC)
        aq, ak, av = a_qkv.unbind(2)

        aq = self.aq_norm(aq)
        ak = self.ak_norm(ak)

        aq = aq.reshape(aB, aN, self.num_heads, aC // self.num_heads).to(dtype)
        ak = ak.reshape(aB, aN, self.num_heads, aC // self.num_heads).to(dtype)
        av = av.reshape(aB, aN, self.num_heads, aC // self.num_heads).to(dtype)

        ################################### attention #########################################
        ################## for fusing ####################
        q = torch.cat([mq, eq, aq], dim=1)
        k = torch.cat([mk, ek, ak], dim=1)
        v = torch.cat([mv, ev, av], dim=1)
        ##################################################
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        if rotary_emb is not None:
            q = apply_rotary_emb(q, rotary_emb, use_real_unbind_dim=-1)
            k = apply_rotary_emb(k, rotary_emb, use_real_unbind_dim=-1)

        use_fp32_attention = getattr(self, "fp32_attention", False)  # necessary for NAN loss
        if use_fp32_attention:
            q, k, v = q.float(), k.float(), v.float()

        attn_bias = None
        if mask is not None:
            attn_bias = torch.zeros([mB * self.num_heads, q.shape[1], k.shape[1]], dtype=q.dtype, device=q.device)
            attn_bias.masked_fill_(mask.squeeze(1).repeat(self.num_heads, 1, 1) == 0, float("-inf"))

        if mask is not None and mask.ndim == 2:
            mask = mask.to(dtype) * -10000.0 # False -> -10000.0, True -> 0.0
            # mask = (1 - mask.to(dtype)) * -10000.0
            mask = mask[:, None, None].repeat(1, self.num_heads, 1, 1)
        
        if self.attn_type == "flash":
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=self.attn_drop, is_causal=False)
        elif self.attn_type == "memory-efficient":
            with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
                attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=self.attn_drop, is_causal=False)
        elif self.attn_type == "math":
            with sdpa_kernel(SDPBackend.MATH):
                attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=self.attn_drop, is_causal=False)
        else:
            attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=self.attn_drop, is_causal=False)
            
        attn_out = attn_out.transpose(1, 2)
        
        attn_out = attn_out.reshape(mB, N, mC)
        motion_out, expression_out, audio_out = \
            attn_out[:, 0:mN, ...], \
            attn_out[:, mN:mN + eN, ...], \
            attn_out[:, mN + eN:N, ...]
        
        motion_out = self.motion_proj(motion_out)
        motion_out = self.motion_proj_drop(motion_out)
        
        expression_out = self.expression_proj(expression_out)
        expression_out = self.expression_proj_drop(expression_out)
        
        audio_out = self.audio_proj(audio_out)
        audio_out = self.audio_proj_drop(audio_out)

        if torch.get_autocast_gpu_dtype() == torch.float16:
            motion_out = motion_out.clip(-65504, 65504)
            expression_out = expression_out.clip(-65504, 65504)
            audio_out = audio_out.clip(-65504, 65504)
        
        return motion_out, expression_out, audio_out

class DoubleStreamFlashAttention(nn.Module):
    """Multi-head Flash Attention block with qk norm."""

    def __init__(
        self,
        dim,
        num_heads=8,
        head_dim=None,
        bias=True,
        attn_type=None,
        qk_norm_type="identity",
        attn_drop=0., 
        proj_drop=0.,
        concat_on_sequence=True,
    ):
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads.
            qkv_bias (bool:  If True, add a learnable bias to query, key, value.
        """
        super(DoubleStreamFlashAttention, self).__init__()
        assert (num_heads is not None and head_dim is None) or (num_heads is None and head_dim is not None), \
            "only one of num_heads and head_dim should be specified"
        if num_heads is not None:
            assert dim % num_heads == 0, 'dim should be divisible by num_heads'
            self.num_heads = num_heads
        elif head_dim is not None:
            assert dim % head_dim == 0, 'head_dim should be divisible by dim'
            self.num_heads = dim // head_dim  
        
        assert attn_type in ["flash", "memory-efficient", "math", None]
        self.attn_type = attn_type
        self.concat_on_sequence = concat_on_sequence
        self.m_qkv = nn.Linear(dim, dim * 3, bias=bias)
        self.e_qkv = nn.Linear(dim, dim * 3, bias=bias)
        
        self.attn_drop = attn_drop
        
        self.motion_proj = nn.Linear(dim, dim, bias=bias)
        self.motion_proj_drop = nn.Dropout(proj_drop)
        
        self.expression_proj = nn.Linear(dim, dim, bias=bias)
        self.expression_proj_drop = nn.Dropout(proj_drop)
        
        
        self.mq_norm = build_norm(name=qk_norm_type, num_features=dim)
        self.mk_norm = build_norm(name=qk_norm_type, num_features=dim)
        
        self.eq_norm = build_norm(name=qk_norm_type, num_features=dim)
        self.ek_norm = build_norm(name=qk_norm_type, num_features=dim)

    def forward(
        self, 
        motion: torch.FloatTensor,
        expression: torch.FloatTensor,
        rotary_emb: torch.FloatTensor = None,
        mask=None,
    ):
        mB, mN, mC = motion.shape
        eB, eN, eC = expression.shape
        assert mB == eB, "batch size must be equal"
        assert mC == eC, "channel size must be equal"
        if self.concat_on_sequence:
            N = mN + eN
            C = mC
        else:
            assert mN == eN, "motion and expression sequence length must be equal"
            N = mN
            C = mC + eC
        #####################################  for motion ######################################
        m_qkv = self.m_qkv(motion).view(mB, mN, 3, mC)
        mq, mk, mv = m_qkv.unbind(2)
        dtype = mq.dtype

        mq = self.mq_norm(mq)
        mk = self.mk_norm(mk)

        mq = mq.reshape(mB, mN, self.num_heads, mC // self.num_heads).to(dtype)
        mk = mk.reshape(mB, mN, self.num_heads, mC // self.num_heads).to(dtype)
        mv = mv.reshape(mB, mN, self.num_heads, mC // self.num_heads).to(dtype)

        #################################### for expression #####################################
        e_qkv = self.e_qkv(expression).view(eB, eN, 3, eC)
        eq, ek, ev = e_qkv.unbind(2)

        eq = self.eq_norm(eq)
        ek = self.ek_norm(ek)

        eq = eq.reshape(eB, eN, self.num_heads, eC // self.num_heads).to(dtype)
        ek = ek.reshape(eB, eN, self.num_heads, eC // self.num_heads).to(dtype)
        ev = ev.reshape(eB, eN, self.num_heads, eC // self.num_heads).to(dtype)

        ################################### attention #########################################
        ################## for fusing ####################
        if self.concat_on_sequence:
            q = torch.cat([mq, eq], dim=1)
            k = torch.cat([mk, ek], dim=1)
            v = torch.cat([mv, ev], dim=1)
        else:
            q = torch.cat([mq, eq], dim=-1)
            k = torch.cat([mk, ek], dim=-1)
            v = torch.cat([mv, ev], dim=-1)
        ##################################################
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        if rotary_emb is not None:
            q = apply_rotary_emb(q, rotary_emb, use_real_unbind_dim=-1)
            k = apply_rotary_emb(k, rotary_emb, use_real_unbind_dim=-1)

        use_fp32_attention = getattr(self, "fp32_attention", False)  # necessary for NAN loss
        if use_fp32_attention:
            q, k, v = q.float(), k.float(), v.float()

        attn_bias = None
        if mask is not None:
            attn_bias = torch.zeros([mB * self.num_heads, q.shape[1], k.shape[1]], dtype=q.dtype, device=q.device)
            attn_bias.masked_fill_(mask.squeeze(1).repeat(self.num_heads, 1, 1) == 0, float("-inf"))

        if mask is not None and mask.ndim == 2:
            # mask = (1 - mask.to(dtype)) * -10000.0
            mask = mask.to(dtype) * -10000.0 # False -> -10000.0, True -> 0.0
            mask = mask[:, None, None].repeat(1, self.num_heads, 1, 1)
        
        if self.attn_type == "flash":
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=self.attn_drop, is_causal=False)
        elif self.attn_type == "memory-efficient":
            with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
                attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=self.attn_drop, is_causal=False)
        elif self.attn_type == "math":
            with sdpa_kernel(SDPBackend.MATH):
                attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=self.attn_drop, is_causal=False)
        else:
            attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=self.attn_drop, is_causal=False)
            
        attn_out = attn_out.transpose(1, 2)
        
        attn_out = attn_out.reshape(mB, N, C)
        if self.concat_on_sequence:
            motion_out, expression_out = \
                attn_out[:, 0:mN, ...], \
                attn_out[:, mN:N, ...]
        else:
            motion_out, expression_out = \
                attn_out[:, :, 0:mC], \
                attn_out[:, :, mC:]
        
        motion_out = self.motion_proj(motion_out)
        motion_out = self.motion_proj_drop(motion_out)
        
        expression_out = self.expression_proj(expression_out)
        expression_out = self.expression_proj_drop(expression_out)
        

        if torch.get_autocast_gpu_dtype() == torch.float16:
            motion_out = motion_out.clip(-65504, 65504)
            expression_out = expression_out.clip(-65504, 65504)
        
        return motion_out, expression_out


class FlashAttention(nn.Module):
    """Multi-head Flash Attention block with qk norm."""

    def __init__(
        self,
        dim,
        num_heads=8,
        head_dim=None,
        bias=True,
        attn_type=None,
        qk_norm_type="identity",
        attn_drop=0., 
        proj_drop=0.,
    ):
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads.
            qkv_bias (bool:  If True, add a learnable bias to query, key, value.
        """
        super(FlashAttention, self).__init__()
        assert (num_heads is not None and head_dim is None) or (num_heads is None and head_dim is not None), \
            "only one of num_heads and head_dim should be specified"
        if num_heads is not None:
            assert dim % num_heads == 0, 'dim should be divisible by num_heads'
            self.num_heads = num_heads
        elif head_dim is not None:
            assert dim % head_dim == 0, 'head_dim should be divisible by dim'
            self.num_heads = dim // head_dim  
        
        assert attn_type in ["flash", "memory-efficient", "math", None]
        self.attn_type = attn_type
        
        self.m_qkv = nn.Linear(dim, dim * 3, bias=bias)
        self.attn_drop = attn_drop
        self.motion_proj = nn.Linear(dim, dim, bias=bias)
        self.motion_proj_drop = nn.Dropout(proj_drop)
                
        self.mq_norm = build_norm(name=qk_norm_type, num_features=dim)
        self.mk_norm = build_norm(name=qk_norm_type, num_features=dim)


    def forward(
        self, 
        motion: torch.FloatTensor,
        rotary_emb: torch.FloatTensor = None,
        mask=None,
    ):
        mB, mN, mC = motion.shape
        #####################################  for motion ######################################
        qkv = self.m_qkv(motion).view(mB, mN, 3, mC)
        q, k, v = qkv.unbind(2)
        dtype = q.dtype

        q = self.mq_norm(q)
        k = self.mk_norm(k)

        q = q.reshape(mB, mN, self.num_heads, mC // self.num_heads).to(dtype)
        k = k.reshape(mB, mN, self.num_heads, mC // self.num_heads).to(dtype)
        v = v.reshape(mB, mN, self.num_heads, mC // self.num_heads).to(dtype)

        ################################### attention #########################################
        ##################################################
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        if rotary_emb is not None:
            q = apply_rotary_emb(q, rotary_emb, use_real_unbind_dim=-1)
            k = apply_rotary_emb(k, rotary_emb, use_real_unbind_dim=-1)

        use_fp32_attention = getattr(self, "fp32_attention", False)  # necessary for NAN loss
        if use_fp32_attention:
            q, k, v = q.float(), k.float(), v.float()

        attn_bias = None
        if mask is not None:
            attn_bias = torch.zeros([mB * self.num_heads, q.shape[1], k.shape[1]], dtype=q.dtype, device=q.device)
            attn_bias.masked_fill_(mask.squeeze(1).repeat(self.num_heads, 1, 1) == 0, float("-inf"))

        if mask is not None and mask.ndim == 2:
            # mask = (1 - mask.to(dtype)) * -10000.0
            mask = mask.to(dtype) * -10000.0 # False -> -10000.0, True -> 0.0
            mask = mask[:, None, None].repeat(1, self.num_heads, 1, 1)
        
        if self.attn_type == "flash":
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=self.attn_drop, is_causal=False)
        elif self.attn_type == "memory-efficient":
            with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
                attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=self.attn_drop, is_causal=False)
        elif self.attn_type == "math":
            with sdpa_kernel(SDPBackend.MATH):
                attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=self.attn_drop, is_causal=False)
        else:
            attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=self.attn_drop, is_causal=False)
            
        attn_out = attn_out.transpose(1, 2)
        attn_out = attn_out.reshape(mB, mN, mC)
        
        motion_out = self.motion_proj(attn_out)
        motion_out = self.motion_proj_drop(motion_out)

        if torch.get_autocast_gpu_dtype() == torch.float16:
            motion_out = motion_out.clip(-65504, 65504)
        
        return motion_out

class Attention(nn.Module):
    """Multi-head Flash Attention block with qk norm."""

    def __init__(
        self,
        dim,
        num_heads=8,
        head_dim=None,
        bias=True,
        attn_type=None,
        qk_norm_type="identity",
        attn_drop=0., 
        proj_drop=0.,
    ):
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads.
            qkv_bias (bool:  If True, add a learnable bias to query, key, value.
        """
        super(Attention, self).__init__()
        assert (num_heads is not None and head_dim is None) or (num_heads is None and head_dim is not None), \
            "only one of num_heads and head_dim should be specified"
        if num_heads is not None:
            assert dim % num_heads == 0, 'dim should be divisible by num_heads'
            self.num_heads = num_heads
        elif head_dim is not None:
            assert dim % head_dim == 0, 'head_dim should be divisible by dim'
            self.num_heads = dim // head_dim  
        
        assert attn_type in ["flash", "naive"]
        self.attn_type = attn_type
        
        self.m_qkv = nn.Linear(dim, dim * 3, bias=bias)
        self.attn_drop = attn_drop
        self.motion_proj = nn.Linear(dim, dim, bias=bias)
        self.motion_proj_drop = nn.Dropout(proj_drop)
                
        self.mq_norm = build_norm(name=qk_norm_type, num_features=dim)
        self.mk_norm = build_norm(name=qk_norm_type, num_features=dim)


    def forward(
        self, 
        motion: torch.FloatTensor,
        rotary_emb: torch.FloatTensor = None,
    ):
        mB, mN, mC = motion.shape
        #####################################  for motion ######################################
        qkv = self.m_qkv(motion).view(mB, mN, 3, mC)
        q, k, v = qkv.unbind(2)
        dtype = q.dtype

        q = self.mq_norm(q)
        k = self.mk_norm(k)

        q = q.reshape(mB, mN, self.num_heads, mC // self.num_heads).to(dtype)
        k = k.reshape(mB, mN, self.num_heads, mC // self.num_heads).to(dtype)
        v = v.reshape(mB, mN, self.num_heads, mC // self.num_heads).to(dtype)

        ################################### attention #########################################
        ##################################################
        if rotary_emb is not None:
            q = apply_rotary_emb(q, rotary_emb, use_real_unbind_dim=-1)
            k = apply_rotary_emb(k, rotary_emb, use_real_unbind_dim=-1)
        
        if self.attn_type == "flash":
            attn_out = attention(q, k, v, causal=False, dropout_p=self.attn_drop)
        elif self.attn_type == "naive":
            attn_out = vanilla_attention(q, k, v, causal=False, dropout_p=self.attn_drop)
        else:
            raise ValueError(f"Invalid attention type: {self.attn_type}")
        
        attn_out = attn_out.reshape(mB, mN, mC)
        
        motion_out = self.motion_proj(attn_out)
        motion_out = self.motion_proj_drop(motion_out)
        
        return motion_out