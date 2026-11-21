import torch
import torch.nn.functional as F
import math
import os
from typing import Optional, Dict, Any, Union
from torch import nn
from diffusion.model.norms import build_norm
from diffusion.model.nets.positional_embeding import (
    apply_rotary_emb, apply_causal_rotary_emb,
    apply_rotary_emb_qk, apply_causal_rotary_emb_qk
)
from diffusion.model.nets.attention import attention, vanilla_attention
from torch.nn.attention.flex_attention import flex_attention, BlockMask
from diffusion.model.nets.basic_modules import t2i_modulate
from einops import rearrange
from diffusion.model.act import build_act

# Import Gated DeltaNet for cross-attention replacement
from diffusion.model.nets.causal.gated_deltanet import GatedDeltaNetForBLD

# IMPORTANT: flex_attention uses higher-order ops that are incompatible with DDPOptimizer
# Must set optimize_ddp=False BEFORE any flex_attention calls to avoid compilation errors
# See: https://github.com/pytorch/pytorch/issues/104674
torch._dynamo.config.optimize_ddp = False

# Set DISABLE_FLEX_COMPILE=1 to disable our additional torch.compile wrapper for flex_attention
# Note: flex_attention internally still uses torch.compile, this only controls our wrapper
_disable_flex_compile = os.environ.get("DISABLE_FLEX_COMPILE", "0") == "1"
if not _disable_flex_compile:
    flex_attention = torch.compile(
        flex_attention, dynamic=False, mode="default")  # Ampere架构替换为default mode="max-autotune-no-cudagraphs")

class CausalSelfAttention(nn.Module):
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
        sink_size=0,
        local_attn_size=-1, # local window的注意力的兼容
        padding_unit_length=128,
        is_teacher_forcing=False,
        max_attention_size=512,
    ):
        """
        Args:
            dim (int): Number of input channels.
            num_heads (int): Number of attention heads.
            qkv_bias (bool:  If True, add a learnable bias to query, key, value.
        """
        super(CausalSelfAttention, self).__init__()
        assert (num_heads is not None and head_dim is None) or (num_heads is None and head_dim is not None), \
            "only one of num_heads and head_dim should be specified"
        if num_heads is not None:
            assert dim % num_heads == 0, 'dim should be divisible by num_heads'
            self.num_heads = num_heads
        elif head_dim is not None:
            assert dim % head_dim == 0, 'head_dim should be divisible by dim'
            self.num_heads = dim // head_dim  
        
        assert attn_type in ["flash", "vanilla", "naive"]
        # Keep "vanilla" as a backward-compatible alias for the non-flash path.
        self.attn_func = attention if attn_type == "flash" else vanilla_attention # attention里面有回滚的逻辑
        self.sink_size = sink_size # attention sink，一般留第一帧
        self.local_attn_size = local_attn_size
        self.max_attention_size = max_attention_size
        self.padding_unit_length = padding_unit_length
        self.m_qkv = nn.Linear(dim, dim * 3, bias=bias)
        self.attn_drop = attn_drop
        self.motion_proj = nn.Linear(dim, dim, bias=bias)
        self.motion_proj_drop = nn.Dropout(proj_drop)
                
        self.mq_norm = build_norm(name=qk_norm_type, num_features=dim)
        self.mk_norm = build_norm(name=qk_norm_type, num_features=dim)
        self.is_teacher_forcing = is_teacher_forcing

    def apply_rope(self, q, k, rotary_emb, is_causal=False, current_start=0, use_liger=False):
        # 方便外面覆写
        if rotary_emb is not None:
            if is_causal:
                frame_seqlen = 1
                current_start_frame = current_start // frame_seqlen
                if use_liger:
                    q, k = apply_causal_rotary_emb_qk(q, k, rotary_emb, use_real_unbind_dim=-1, start_frame=current_start_frame, use_liger=True)
                else:
                    q = apply_causal_rotary_emb(q, rotary_emb, use_real_unbind_dim=-1, start_frame=current_start_frame)
                    k = apply_causal_rotary_emb(k, rotary_emb, use_real_unbind_dim=-1, start_frame=current_start_frame)
            else:
                if self.is_teacher_forcing:
                    q_chunk = torch.chunk(q, 2, dim=1)
                    k_chunk = torch.chunk(k, 2, dim=1)
                    roped_query = []
                    roped_key = []
                    # rope should be same for clean and noisy parts
                    for ii in range(2):
                        if use_liger:
                            rq, rk = apply_rotary_emb_qk(q_chunk[ii], k_chunk[ii], rotary_emb, use_real_unbind_dim=-1, use_liger=True)
                        else:
                            rq = apply_rotary_emb(q_chunk[ii], rotary_emb, use_real_unbind_dim=-1)
                            rk = apply_rotary_emb(k_chunk[ii], rotary_emb, use_real_unbind_dim=-1)
                        roped_query.append(rq)
                        roped_key.append(rk)

                    q = torch.cat(roped_query, dim=1)
                    k = torch.cat(roped_key, dim=1)
                else:
                    if use_liger:
                        q, k = apply_rotary_emb_qk(q, k, rotary_emb, use_real_unbind_dim=-1, use_liger=True)
                    else:
                        q = apply_rotary_emb(q, rotary_emb, use_real_unbind_dim=-1)
                        k = apply_rotary_emb(k, rotary_emb, use_real_unbind_dim=-1)
        return q, k

    def attn_multihead(self, q, k, v, mask, kv_cache=None, current_start=0, prefill_only=False, update_kv_cache=True):
        ################################### attention #########################################
        ##################################################
        if kv_cache is None:
            padded_length = math.ceil(q.shape[1] / self.padding_unit_length) * self.padding_unit_length - q.shape[1]
            padded_q = torch.cat(
                [q, torch.zeros(
                    [q.shape[0], padded_length, q.shape[2], q.shape[3]],
                    device=q.device, dtype=q.dtype
                )],
                dim=1
            )

            padded_k = torch.cat(
                [k, torch.zeros(
                    [k.shape[0], padded_length, k.shape[2], k.shape[3]],
                    device=k.device, dtype=v.dtype
                )],
                dim=1
            )

            padded_v = torch.cat(
                [v, torch.zeros(
                    [v.shape[0], padded_length, v.shape[2], v.shape[3]],
                    device=v.device, dtype=v.dtype
                )],
                dim=1
            )

            if isinstance(mask, BlockMask):
                attn_out = flex_attention(
                    query=padded_q.transpose(2, 1),
                    key=padded_k.transpose(2, 1),
                    value=padded_v.transpose(2, 1),
                    block_mask=mask
                )
            else:
                # Fallback: dense mask + SDPA (when head_dim not power-of-2 for flex)
                # mask is (1, 1, Q_padded, KV_padded) bool, True=attend
                float_mask = torch.zeros(
                    1, 1, padded_q.shape[1], padded_k.shape[1],
                    device=padded_q.device, dtype=padded_q.dtype,
                )
                float_mask.masked_fill_(~mask, float('-inf'))
                attn_out = F.scaled_dot_product_attention(
                    padded_q.transpose(2, 1),
                    padded_k.transpose(2, 1),
                    padded_v.transpose(2, 1),
                    attn_mask=float_mask,
                )
            if padded_length > 0:
                attn_out = attn_out[:, :, :-padded_length].transpose(2, 1)
            else:
                attn_out = attn_out.transpose(2, 1)
        else:
            frame_seqlen = 1 # 一般来说对motion序列，每一帧就是一个token，但是在这里放出来，或许后面有改进
            current_end = current_start + q.shape[1]
            sink_tokens = self.sink_size * frame_seqlen
            kv_cache_size = kv_cache["k"].shape[1]
            num_new_tokens = q.shape[1]

            # prefill_only is only a stage marker / consistency constraint.
            # It must NOT alter the attention compute path.
            if prefill_only and not update_kv_cache:
                raise ValueError("prefill_only=True requires update_kv_cache=True for cache prefill consistency")
            
            # P0-4: Handle cache write logic based on update_kv_cache flag
            if update_kv_cache:
                if self.local_attn_size != -1 and (current_end > kv_cache["global_end_index"].item()) and (
                    num_new_tokens + kv_cache["local_end_index"].item() > kv_cache_size):
                    # Calculate the number of new tokens added in this step
                    # Shift existing cache content left to discard oldest tokens
                    # Clone the source slice to avoid overlapping memory error
                    # 下面这一段的逻辑就是驱逐旧的token，插入新的token
                    num_evicted_tokens = num_new_tokens + kv_cache["local_end_index"].item() - kv_cache_size
                    num_rolled_tokens = kv_cache["local_end_index"].item() - num_evicted_tokens - sink_tokens
                    kv_cache["k"][:, sink_tokens:sink_tokens + num_rolled_tokens] = \
                        kv_cache["k"][:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
                    kv_cache["v"][:, sink_tokens:sink_tokens + num_rolled_tokens] = \
                        kv_cache["v"][:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
                    # Insert the new keys/values at the end
                    local_end_index = kv_cache["local_end_index"].item() + current_end - \
                        kv_cache["global_end_index"].item() - num_evicted_tokens
                    local_start_index = local_end_index - num_new_tokens
                    kv_cache["k"][:, local_start_index:local_end_index] = k
                    kv_cache["v"][:, local_start_index:local_end_index] = v
                else: # 这边就简单很多了
                    local_end_index = kv_cache["local_end_index"].item() + current_end - kv_cache["global_end_index"].item()
                    local_start_index = local_end_index - num_new_tokens
                    # Defensive: with local_attn_size==-1 (full crossattn) the
                    # eviction branch above is skipped. If cache_size < total
                    # rollout, the next slice assignment silently truncates and
                    # the read window mismatches the query length downstream.
                    # Fail loudly instead.
                    if local_end_index > kv_cache_size:
                        raise RuntimeError(
                            f"KV cache write would overflow at current_end={current_end}: "
                            f"local_end_index={local_end_index} > kv_cache_size={kv_cache_size}. "
                            f"With local_attn_size={self.local_attn_size}, no eviction is "
                            f"performed (eviction is gated on local_attn_size != -1). "
                            f"Either raise model.max_attention_size to >= total rollout length, "
                            f"or use a windowed local_attn_size."
                        )
                    kv_cache["k"][:, local_start_index:local_end_index] = k
                    kv_cache["v"][:, local_start_index:local_end_index] = v
                
                kv_cache["global_end_index"].fill_(current_end)
                kv_cache["local_end_index"].fill_(local_end_index)
            else:
                # Read-only mode: use existing cache indices without updating
                local_end_index = kv_cache["local_end_index"].item()
            
            # Keep shared compute graph between prefill and decode:
            # both paths run real attention; only cache write policy differs.
            attn_out = self.attn_func(
                q,
                kv_cache["k"][:, max(0, local_end_index - self.max_attention_size):local_end_index],
                kv_cache["v"][:, max(0, local_end_index - self.max_attention_size):local_end_index]
            )
        
        
        return attn_out

    def forward(
        self,
        motion: torch.FloatTensor,
        mask,
        rotary_emb: torch.FloatTensor = None,
        use_liger_rope: bool = False,
        kv_cache=None,
        current_start=0,
        prefill_only: bool = False,
        update_kv_cache: bool = True,
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

        q, k = self.apply_rope(q, k, rotary_emb, is_causal=kv_cache is not None, current_start=current_start, use_liger=use_liger_rope)
        attn_out = self.attn_multihead(q, k, v, mask, kv_cache, current_start, prefill_only, update_kv_cache)
        attn_out = attn_out.reshape(mB, mN, mC)

        motion_out = self.motion_proj(attn_out)
        motion_out = self.motion_proj_drop(motion_out)

        if torch.get_autocast_dtype('cuda') == torch.float16:
            motion_out = motion_out.clip(-65504, 65504)

        return motion_out

class TimestepAdaptiveFusionGate(nn.Module):
    """
    Timestep-adaptive fusion gate for balancing historical memory vs current context.
    
    This module learns to output a fusion coefficient α ∈ (0,1) based on:
    - Timestep embedding (t_emb): diffusion timestep information, shape (B, T, timestep_dim)
    - Hidden state (h): current block's hidden representation, shape (B, T, hidden_dim)
    
    The output α is used to blend:
    - cross_attn_out: information from historical blocks
    - current_info: information from current block self-attention
    
    Final output = α * cross_attn_out + (1 - α) * current_info
    
    Intuition:
    - Early timesteps (high noise): α should be higher (rely more on history)
    - Late timesteps (low noise): α can be lower (current block has enough signal)
    
    Note: In BLD (Block-Level Diffusion), t_emb has shape (B, T, timestep_dim) where:
    - Within each block, frames share the same timestep value
    - Different blocks may have different timestep values (causal diffusion)
    - The fusion coefficient is computed per-frame to handle this correctly
    """
    def __init__(
        self,
        hidden_dim: int,
        timestep_dim: int,
        gate_hidden_dim: int = None,
        gate_activation: str = "silu",
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.timestep_dim = timestep_dim
        gate_hidden_dim = gate_hidden_dim or hidden_dim * 2
        
        # Timestep projection (if dimensions don't match)
        if timestep_dim != hidden_dim:
            self.t_proj = nn.Linear(timestep_dim, hidden_dim)
        else:
            self.t_proj = nn.Identity()
        
        # Activation function
        act_fn = build_act(gate_activation)
        
        # Gating network: (t_emb || h) -> α
        # Input: concatenation of timestep embedding and hidden state (per-frame)
        # This allows different blocks to have different fusion coefficients
        self.gate_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, gate_hidden_dim),
            act_fn,
            nn.Linear(gate_hidden_dim, gate_hidden_dim // 2),
            act_fn,
            nn.Linear(gate_hidden_dim // 2, 1),
            nn.Sigmoid(),  # Output α ∈ (0, 1)
        )
        
        # Initialize the final layer to output ~0.5 (balanced fusion)
        nn.init.zeros_(self.gate_net[-2].weight)
        nn.init.zeros_(self.gate_net[-2].bias)
    
    def forward(
        self,
        t_emb: torch.Tensor,         # (B, T, timestep_dim)
        hidden_state: torch.Tensor,  # (B, T, hidden_dim)
    ) -> torch.Tensor:
        """
        Compute fusion coefficient α per-frame.
        
        In BLD training, t_emb has shape (B, T, timestep_dim) where:
        - Frames within the same block share the same timestep value
        - Different blocks may have different timestep values
        
        We compute α per-frame so that:
        - Early blocks (high noise, early timestep) get higher α (rely more on history)
        - Late blocks (low noise, late timestep) get lower α (rely more on current block)
        
        Args:
            t_emb: Timestep embedding, shape (B, T, timestep_dim)
            hidden_state: Current hidden state, shape (B, T, hidden_dim)
            
        Returns:
            alpha: Fusion coefficient, shape (B, T, 1) for broadcasting
        """
        B, T, _ = hidden_state.shape
        
        # Handle different t_emb shapes
        if t_emb.dim() == 2:
            # (B, timestep_dim) -> (B, T, timestep_dim) by broadcasting
            t_emb = t_emb.unsqueeze(1).expand(-1, T, -1)
        elif t_emb.dim() == 3:
            # Already (B, T, timestep_dim), check shape compatibility
            if t_emb.shape[1] != T:
                # If t_emb has different sequence length, we need to align
                # This can happen if t_emb is (B, 1, timestep_dim)
                if t_emb.shape[1] == 1:
                    t_emb = t_emb.expand(-1, T, -1)
                else:
                    # Interpolate or use first frame's value
                    # For now, take the mean across time as a fallback
                    t_emb = t_emb.mean(dim=1, keepdim=True).expand(-1, T, -1)
        
        # Project timestep embedding: (B, T, timestep_dim) -> (B, T, hidden_dim)
        t_feat = self.t_proj(t_emb)
        
        # Concatenate t_emb and hidden_state per-frame
        # gate_input: (B, T, hidden_dim * 2)
        gate_input = torch.cat([t_feat, hidden_state], dim=-1)
        
        # Compute gate per-frame: (B, T, 1)
        alpha = self.gate_net(gate_input)
        
        return alpha


class CrossCausalSelfAttention(nn.Module):
    """
    Cross-attention from noisy current block (Query) to clean historical blocks (Key/Value).
    
    Note: Timestep-adaptive fusion is now handled at the CausalEmageBlock level,
    where Self-Attn and Cross-Attn are computed in parallel and fused via a learned gate.
    """
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
        sink_size=0,
        local_attn_size=-1, # local window的注意力的兼容
        padding_unit_length=128,
        is_teacher_forcing=False,
        max_attention_size=512,
        add_learnable_prefix=False,
    ):
        super(CrossCausalSelfAttention, self).__init__()
        assert (num_heads is not None and head_dim is None) or (num_heads is None and head_dim is not None), \
            "only one of num_heads and head_dim should be specified"
        if num_heads is not None:
            assert dim % num_heads == 0, 'dim should be divisible by num_heads'
            self.num_heads = num_heads
        elif head_dim is not None:
            assert dim % head_dim == 0, 'head_dim should be divisible by dim'
            self.num_heads = dim // head_dim
        
        assert attn_type in ["flash", "vanilla", "naive"]
        # Keep "vanilla" as a backward-compatible alias for the non-flash path.
        self.attn_func = attention if attn_type == "flash" else vanilla_attention # attention里面有回滚的逻辑
        self.sink_size = sink_size # attention sink，一般留第一帧
        self.local_attn_size = local_attn_size
        self.max_attention_size = max_attention_size
        self.padding_unit_length = padding_unit_length
        self.m_q_linear = nn.Linear(dim, dim, bias=bias)
        self.kv_linear = nn.Linear(dim, 2 * dim, bias=bias)
        self.attn_drop = attn_drop
        self.motion_proj = nn.Linear(dim, dim, bias=bias)
        self.motion_proj_drop = nn.Dropout(proj_drop)
                
        self.mq_norm = build_norm(name=qk_norm_type, num_features=dim)
        self.mk_norm = build_norm(name=qk_norm_type, num_features=dim)
        self.is_teacher_forcing = is_teacher_forcing
        self.add_learnable_prefix = add_learnable_prefix
        if self.add_learnable_prefix:
            self.prefix_token = nn.Parameter(torch.zeros(1, 1, dim))
    
    def _kv_cache_write(self, k, v, kv_cache, current_start: int):
        """
        k, v: (B, T, H, Hd)  只写入 cache，不做 attention
        kv_cache: dict with keys ['k','v','global_end_index','local_end_index']
        """
        frame_seqlen = 1
        num_new_tokens = k.shape[1]
        current_end = current_start + num_new_tokens

        sink_tokens = self.sink_size * frame_seqlen
        kv_cache_size = kv_cache["k"].shape[1]

        # 这里要求：commit/prefill 时 current_start 必须等于 cache 的 global_end_index
        # 否则你的“覆盖式写入”会变成“乱序写入”
        if kv_cache["global_end_index"].item() != current_start:
            raise RuntimeError(
                f"KV cache write expects global_end_index==current_start, "
                f"but got {kv_cache['global_end_index'].item()} vs {current_start}"
            )

        # 计算写入区间（cache 坐标）
        local_end_index = kv_cache["local_end_index"].item() + num_new_tokens
        local_start_index = local_end_index - num_new_tokens

        # 需要驱逐（滚动）时：只驱逐非 sink 区域
        if (self.local_attn_size != -1) and (local_end_index > kv_cache_size):
            num_evicted_tokens = local_end_index - kv_cache_size
            num_rolled_tokens = kv_cache_size - sink_tokens - num_new_tokens
            if num_rolled_tokens < 0:
                raise RuntimeError(
                    "kv_cache_size too small: must be >= sink_tokens + num_new_tokens"
                )

            # roll
            kv_cache["k"][:, sink_tokens:sink_tokens + num_rolled_tokens] = \
                kv_cache["k"][:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()
            kv_cache["v"][:, sink_tokens:sink_tokens + num_rolled_tokens] = \
                kv_cache["v"][:, sink_tokens + num_evicted_tokens:sink_tokens + num_evicted_tokens + num_rolled_tokens].clone()

            # 写到尾部
            local_end_index = kv_cache_size
            local_start_index = kv_cache_size - num_new_tokens

        # Defensive: with local_attn_size==-1 the eviction branch above is
        # skipped, so a too-small cache silently truncates the write here.
        # See sibling check in BlockWiseAttention._cross_block_attention.
        if local_end_index > kv_cache_size:
            raise RuntimeError(
                f"KV cache write would overflow at current_end={current_end}: "
                f"local_end_index={local_end_index} > kv_cache_size={kv_cache_size}. "
                f"With local_attn_size={self.local_attn_size}, no eviction is "
                f"performed (eviction is gated on local_attn_size != -1). "
                f"Either raise model.max_attention_size to >= total rollout length, "
                f"or use a windowed local_attn_size."
            )
        kv_cache["k"][:, local_start_index:local_end_index] = k
        kv_cache["v"][:, local_start_index:local_end_index] = v
        kv_cache["global_end_index"].fill_(current_end)
        kv_cache["local_end_index"].fill_(local_end_index)
        return local_end_index

    def _kv_cache_read_slice(self, kv_cache):
        local_end_index = kv_cache["local_end_index"].item()
        # 读窗口：最后 max_attention_size 个 token
        start = max(0, local_end_index - self.max_attention_size)
        k = kv_cache["k"][:, start:local_end_index]
        v = kv_cache["v"][:, start:local_end_index]
        return k, v

    def _maybe_prepend_prefix(self, context: torch.FloatTensor) -> torch.FloatTensor:
        if not self.add_learnable_prefix:
            return context
        prefix = self.prefix_token.to(device=context.device, dtype=context.dtype)
        prefix = prefix.expand(context.shape[0], -1, -1)
        return torch.cat([prefix, context], dim=1)

    def _build_prefix_kv(self, batch_size: int, dtype: torch.dtype, device: torch.device):
        prefix = self.prefix_token.to(device=device, dtype=dtype)
        prefix = prefix.expand(batch_size, -1, -1)
        kv = self.kv_linear(prefix)
        k, v = kv.chunk(2, dim=-1)
        k = self.mk_norm(k)
        k = k.reshape(batch_size, 1, self.num_heads, -1).to(dtype)
        v = v.reshape(batch_size, 1, self.num_heads, -1).to(dtype)
        return k, v

    def apply_rope_train(self, q, k, rotary_emb, use_liger=False):
        if rotary_emb is None:
            return q, k
        if use_liger:
            return apply_rotary_emb_qk(q, k, rotary_emb, use_real_unbind_dim=-1, use_liger=True)
        else:
            q = apply_rotary_emb(q, rotary_emb, use_real_unbind_dim=-1)
            k = apply_rotary_emb(k, rotary_emb, use_real_unbind_dim=-1)
            return q, k

    def apply_rope_causal(self, vec, rotary_emb, current_start=0, use_liger=False):
        if rotary_emb is None:
            return vec
        frame_seqlen = 1
        start_frame = current_start // frame_seqlen
        # Note: apply_causal_rotary_emb only handles single tensor, not qk pair
        # Liger kernel optimization is not applicable here since we only have one tensor
        return apply_causal_rotary_emb(vec, rotary_emb, use_real_unbind_dim=-1, start_frame=start_frame)

    def attn_multihead(
        self, q, k, v, mask,
        kv_cache=None,
        current_start=0,
        update_kv_cache: bool = True,
        prefill_only: bool = False,
    ):
        if kv_cache is None:
            q_len = q.shape[1]
            k_len = k.shape[1]
            padded_q_len = math.ceil(q_len / self.padding_unit_length) * self.padding_unit_length
            padded_k_len = math.ceil(k_len / self.padding_unit_length) * self.padding_unit_length
            q_padding = padded_q_len - q_len
            k_padding = padded_k_len - k_len
            padded_q = torch.cat(
                [q, torch.zeros(
                    [q.shape[0], q_padding, q.shape[2], q.shape[3]],
                    device=q.device, dtype=q.dtype
                )],
                dim=1
            )

            padded_k = torch.cat(
                [k, torch.zeros(
                    [k.shape[0], k_padding, k.shape[2], k.shape[3]],
                    device=k.device, dtype=v.dtype
                )],
                dim=1
            )

            padded_v = torch.cat(
                [v, torch.zeros(
                    [v.shape[0], k_padding, v.shape[2], v.shape[3]],
                    device=v.device, dtype=v.dtype
                )],
                dim=1
            )

            if isinstance(mask, BlockMask):
                attn_out = flex_attention(
                    query=padded_q.transpose(2, 1),
                    key=padded_k.transpose(2, 1),
                    value=padded_v.transpose(2, 1),
                    block_mask=mask
                )
            else:
                # Fallback: dense mask + SDPA (when head_dim not power-of-2 for flex)
                # mask is (1, 1, Q_padded, KV_padded) bool, True=attend
                float_mask = torch.zeros(
                    1, 1, padded_q.shape[1], padded_k.shape[1],
                    device=padded_q.device, dtype=padded_q.dtype,
                )
                float_mask.masked_fill_(~mask, float('-inf'))
                attn_out = F.scaled_dot_product_attention(
                    padded_q.transpose(2, 1),
                    padded_k.transpose(2, 1),
                    padded_v.transpose(2, 1),
                    attn_mask=float_mask,
                )
            if q_padding > 0:
                attn_out = attn_out[:, :, :-q_padding].transpose(2, 1)
            else:
                attn_out = attn_out.transpose(2, 1)
            return attn_out

        # ======= inference path: KV cache =======
        if update_kv_cache:
            self._kv_cache_write(k, v, kv_cache, current_start)
        if prefill_only: # 只写 cache，不计算 attn_out
            return None

        k_cache, v_cache = self._kv_cache_read_slice(kv_cache)
        if self.add_learnable_prefix and q is not None:
            prefix_k, prefix_v = self._build_prefix_kv(batch_size=q.shape[0], dtype=q.dtype, device=q.device)
            k_cache = torch.cat([prefix_k, k_cache], dim=1)
            v_cache = torch.cat([prefix_v, v_cache], dim=1)
        attn_out = self.attn_func(q, k_cache, v_cache)
        return attn_out
    
    def forward(
        self,
        motion: torch.FloatTensor,            # noisy block for decode; 任意 tensor (prefill 时不会用)
        context: torch.FloatTensor = None,    # clean tokens (prefill/commit 时必须有)
        mask: torch.BoolTensor = None,        # training 时必须有
        rotary_emb: torch.FloatTensor = None,
        use_liger_rope: bool = False,
        kv_cache=None,
        current_start=0,
        update_kv_cache: bool = False,
        prefill_only: bool = False,
    ):
        # -------------------------
        # (1) TRAINING: no kv_cache, use mask + flex_attention
        # -------------------------
        if kv_cache is None:
            assert (context is not None) and (mask is not None)
            mB, mN, mC = motion.shape
            context = self._maybe_prepend_prefix(context)
            cB, cN, cC = context.shape
            assert mB == cB and mC == cC

            kv = self.kv_linear(context).view(cB, cN, 2, cC)
            k, v = kv.unbind(2)
            q = self.m_q_linear(motion)

            dtype = q.dtype
            q = self.mq_norm(q)
            k = self.mk_norm(k)

            q = q.reshape(mB, mN, self.num_heads, mC // self.num_heads).to(dtype)
            k = k.reshape(cB, cN, self.num_heads, cC // self.num_heads).to(dtype)
            v = v.reshape(cB, cN, self.num_heads, cC // self.num_heads).to(dtype)

            # training: 你原来的 apply_rope（非 causal）
            if self.add_learnable_prefix and rotary_emb is not None:
                k_prefix, k_body = k[:, :1], k[:, 1:]
                if use_liger_rope:
                    q, k_body = apply_rotary_emb_qk(q, k_body, rotary_emb, use_real_unbind_dim=-1, use_liger=True)
                else:
                    q = apply_rotary_emb(q, rotary_emb, use_real_unbind_dim=-1)
                    k_body = apply_rotary_emb(k_body, rotary_emb, use_real_unbind_dim=-1)
                k = torch.cat([k_prefix, k_body], dim=1)
            else:
                q, k = self.apply_rope_train(q, k, rotary_emb, use_liger=use_liger_rope)
            attn_out = self.attn_multihead(q, k, v, mask, kv_cache=None, current_start=0)
            attn_out = attn_out.reshape(mB, mN, mC)

            out = self.motion_proj_drop(self.motion_proj(attn_out))
            if torch.get_autocast_dtype('cuda') == torch.float16:
                out = out.clip(-65504, 65504)
            return out

        # -------------------------
        # (2) PREFILL / COMMIT: only write KV cache from context
        # -------------------------
        if prefill_only:
            assert update_kv_cache is True
            assert context is not None

            cB, cN, cC = context.shape
            kv = self.kv_linear(context).view(cB, cN, 2, cC)
            k, v = kv.unbind(2)

            k = self.mk_norm(k)
            k = k.reshape(cB, cN, self.num_heads, cC // self.num_heads)
            v = v.reshape(cB, cN, self.num_heads, cC // self.num_heads)

            # cache 中的 K 必须用 causal RoPE（按 current_start 对齐绝对位置）
            if self.add_learnable_prefix and rotary_emb is not None:
                k_prefix, k_body = k[:, :1], k[:, 1:]
                k_body = self.apply_rope_causal(k_body, rotary_emb, current_start=current_start, use_liger=use_liger_rope)
                k = torch.cat([k_prefix, k_body], dim=1)
            else:
                k = self.apply_rope_causal(k, rotary_emb, current_start=current_start, use_liger=use_liger_rope)

            # 只写 cache，不算注意力
            self.attn_multihead(
                q=None, k=k, v=v, mask=None,
                kv_cache=kv_cache,
                current_start=current_start,
                update_kv_cache=True,
                prefill_only=True,
            )
            return None

        # -------------------------
        # (3) DECODE: read-only KV cache, compute attention for motion->context(cache)
        # -------------------------
        assert update_kv_cache is False
        mB, mN, mC = motion.shape

        q = self.m_q_linear(motion)
        dtype = q.dtype
        q = self.mq_norm(q)
        q = q.reshape(mB, mN, self.num_heads, mC // self.num_heads).to(dtype)

        # decode 时 Q 需要 causal RoPE（按 current_start 对齐绝对位置）
        q = self.apply_rope_causal(q, rotary_emb, current_start=current_start, use_liger=use_liger_rope)

        attn_out = self.attn_multihead(
            q=q, k=None, v=None, mask=None,
            kv_cache=kv_cache,
            current_start=current_start,
            update_kv_cache=False,
            prefill_only=False,
        )
        attn_out = attn_out.reshape(mB, mN, mC)

        out = self.motion_proj_drop(self.motion_proj(attn_out))
        if torch.get_autocast_dtype('cuda') == torch.float16:
            out = out.clip(-65504, 65504)
        return out

class MMCausalSelfAttention(CausalSelfAttention):
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
        sink_size=0,
        local_attn_size=-1, # local window的注意力的兼容
        padding_unit_length=128,
        is_teacher_forcing=False,
        max_attention_size=256,
    ):
        super(MMCausalSelfAttention, self).__init__(
            dim,
            num_heads,
            head_dim,
            bias,
            attn_type,
            qk_norm_type,
            attn_drop,
            proj_drop,
            sink_size,
            local_attn_size,
            padding_unit_length,
            is_teacher_forcing,
            max_attention_size,
        )
        
        self.m2_qkv = nn.Linear(dim, dim * 3, bias=bias)        
        self.motion2_proj = nn.Linear(dim, dim, bias=bias)
        self.motion2_proj_drop = nn.Dropout(proj_drop)

        self.mq2_norm = build_norm(name=qk_norm_type, num_features=dim)
        self.mk2_norm = build_norm(name=qk_norm_type, num_features=dim)
    
        
    def forward(
        self, 
        motion: torch.FloatTensor,
        motion2: torch.FloatTensor,
        mask,
        rotary_emb: torch.FloatTensor = None,
        kv_cache=None,
        current_start=0,
    ):
        mB, mN, mC = motion.shape
        eB, eN, eC = motion2.shape
        assert mB == eB, "batch size must be equal"
        assert mC == eC, "channel size must be equal"
        N = mN + eN
        C = mC
       
        #####################################  for motion ######################################
        m_qkv = self.m_qkv(motion).view(mB, mN, 3, mC)
        mq, mk, mv = m_qkv.unbind(2)
        dtype = mq.dtype

        mq = self.mq_norm(mq)
        mk = self.mk_norm(mk)

        mq = mq.reshape(mB, mN, self.num_heads, mC // self.num_heads).to(dtype)
        mk = mk.reshape(mB, mN, self.num_heads, mC // self.num_heads).to(dtype)
        mv = mv.reshape(mB, mN, self.num_heads, mC // self.num_heads).to(dtype)

        #################################### for another #####################################
        m2_qkv = self.m2_qkv(motion2).view(eB, eN, 3, eC)
        m2q, m2k, m2v = m2_qkv.unbind(2)

        m2q = self.mq2_norm(m2q)
        m2k = self.mk2_norm(m2k)

        m2q = m2q.reshape(eB, eN, self.num_heads, eC // self.num_heads).to(dtype)
        m2k = m2k.reshape(eB, eN, self.num_heads, eC // self.num_heads).to(dtype)
        m2v = m2v.reshape(eB, eN, self.num_heads, eC // self.num_heads).to(dtype)
        
        mq, mk = self.apply_rope(mq, mk, rotary_emb, is_causal=kv_cache is not None, current_start=current_start)
        m2q, m2k = self.apply_rope(m2q, m2k, rotary_emb, is_causal=kv_cache is not None, current_start=current_start)
        
        q = torch.cat([mq, m2q], dim=1)
        k = torch.cat([mk, m2k], dim=1)
        v = torch.cat([mv, m2v], dim=1)
        
        attn_out = self.attn_multihead(q, k, v, mask, kv_cache, current_start * 2)        
        attn_out = attn_out.reshape(mB, N, C)
    
        motion_out, motion2_out = \
            attn_out[:, 0:mN, ...], \
            attn_out[:, mN:N, ...]
        
        motion_out = self.motion_proj(motion_out)
        motion_out = self.motion_proj_drop(motion_out)
        
        motion2_out = self.motion2_proj(motion2_out)
        motion2_out = self.motion2_proj_drop(motion2_out)

        if torch.get_autocast_dtype('cuda') == torch.float16:
            motion_out = motion_out.clip(-65504, 65504)
            motion2_out = motion2_out.clip(-65504, 65504)
        
        return motion_out, motion2_out
    
######################################################################################################################
######################################## for cross attention #########################################################
######################################################################################################################
class CrossAttention(nn.Module):
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
        super(CrossAttention, self).__init__()
        assert (num_heads is not None and head_dim is None) or (num_heads is None and head_dim is not None), \
            "only one of num_heads and head_dim should be specified"
        if num_heads is not None:
            assert dim % num_heads == 0, 'dim should be divisible by num_heads'
            self.num_heads = num_heads
        elif head_dim is not None:
            assert dim % head_dim == 0, 'head_dim should be divisible by dim'
            self.num_heads = dim // head_dim  
        
        assert attn_type in ["flash", "vanilla"]
        self.attn_func = attention if attn_type == "flash" else vanilla_attention
        
        self.m_q_linear = nn.Linear(dim, dim, bias=bias)
        self.kv_linear = nn.Linear(dim, 2 * dim, bias=bias)
        
        self.attn_drop = attn_drop
        
        self.motion_proj = nn.Linear(dim, dim, bias=bias)
        self.motion_proj_drop = nn.Dropout(proj_drop)
        
        self.mq_norm = build_norm(name=qk_norm_type, num_features=dim)
        self.k_norm = build_norm(name=qk_norm_type, num_features=dim)

    # def apply_rope(self, q, k, rotary_emb, is_causal=False):
    #     # 方便外面覆写
    #     if rotary_emb is not None:
    #         if is_causal:
    #             q = apply_causal_rotary_emb(q, rotary_emb, use_real_unbind_dim=-1)
    #             k = apply_causal_rotary_emb(k, rotary_emb, use_real_unbind_dim=-1)
    #         else:
    #             q = apply_rotary_emb(q, rotary_emb, use_real_unbind_dim=-1)
    #             k = apply_rotary_emb(k, rotary_emb, use_real_unbind_dim=-1)
    #     return q, k
    
    def forward(
        self, 
        motion: torch.FloatTensor,
        context: torch.FloatTensor,
        crossattn_cache=None,
        **kwargs,
    ):
        mB, mN, mC = motion.shape
        cB, cN, cC = context.shape
        #####################################  for motion ######################################
        q = self.m_q_linear(motion)
        dtype = q.dtype
        q = self.mq_norm(q)
        q = q.reshape(mB, mN, self.num_heads, mC // self.num_heads).to(dtype)
 
        if crossattn_cache is None:
            ################################### attention #########################################
            kv = self.kv_linear(context).view(cB, cN, 2, cC)
            k, v = kv.unbind(2)
            k = self.k_norm(k).view(cB, -1, self.num_heads, cC // self.num_heads)
            v = v.view(cB, -1, self.num_heads, cC // self.num_heads)
            ##################################################
        else:
            if not crossattn_cache["is_init"]:
                crossattn_cache["is_init"] = True
                kv = self.kv_linear(context).view(cB, cN, 2, cC)
                k, v = kv.unbind(2)
                k = self.k_norm(k).view(cB, -1, self.num_heads, cC // self.num_heads)
                v = v.view(cB, -1, self.num_heads, cC // self.num_heads)
                crossattn_cache["k"] = k
                crossattn_cache["v"] = v
            else:
                k = crossattn_cache["k"]
                v = crossattn_cache["v"]
            
        attn_out = self.attn_func(
            q, k, v
        )
        attn_out = attn_out.transpose(1, 2)
        attn_out = attn_out.reshape(mB, mN, mC)
        
        motion_out = self.motion_proj(attn_out)
        motion_out = self.motion_proj_drop(motion_out)
        
        if torch.get_autocast_dtype('cuda') == torch.float16:
            motion_out = motion_out.clip(-65504, 65504)
        
        return motion_out

class CrossAttentionMask(CrossAttention):
    def __init__(self, pad_unit_length=128, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pad_unit_length = pad_unit_length
    
    def forward(
        self, 
        motion: torch.FloatTensor,
        context: torch.FloatTensor,
        mask=None,
        sampling_mode=False,
        **kwargs,
    ):
        # 实际上这种情况下kv_cache根本不管用，挂到上面起到一个判断的作用
        mB, mN, mC = motion.shape
        cB, cN, cC = context.shape
        #####################################  for motion ######################################
        q = self.m_q_linear(motion)
        dtype = q.dtype
        q = self.mq_norm(q)
        q = q.reshape(mB, mN, self.num_heads, mC // self.num_heads).to(dtype)
   
        ################################### attention #########################################
        kv = self.kv_linear(context).view(cB, cN, 2, cC)
        k, v = kv.unbind(2)
        k = self.k_norm(k).view(cB, -1, self.num_heads, cC // self.num_heads)
        v = v.view(cB, -1, self.num_heads, cC // self.num_heads)
        ##################################################
        if not sampling_mode:
            padded_length_q = math.ceil(q.shape[1] / self.pad_unit_length) * self.pad_unit_length - q.shape[1]
            padded_q = torch.cat(
                [q, torch.zeros(
                    [q.shape[0], padded_length_q, q.shape[2], q.shape[3]],
                    device=q.device, dtype=q.dtype
                )],
                dim=1
            )

            padded_length_k = math.ceil(k.shape[1] / self.pad_unit_length) * self.pad_unit_length - k.shape[1]
            padded_k = torch.cat(
                [k, torch.zeros(
                    [k.shape[0], padded_length_k, k.shape[2], k.shape[3]],
                    device=k.device, dtype=v.dtype
                )],
                dim=1
            )

            padded_v = torch.cat(
                [v, torch.zeros(
                    [v.shape[0], padded_length_k, v.shape[2], v.shape[3]],
                    device=v.device, dtype=v.dtype
                )],
                dim=1
            )

            attn_out = flex_attention(
                query=padded_q.transpose(2, 1),
                key=padded_k.transpose(2, 1),
                value=padded_v.transpose(2, 1),
                block_mask=mask
            )
            if padded_length_q > 0:
                attn_out = attn_out[:, :, :-padded_length_q].transpose(2, 1)
            else:
                attn_out = attn_out.transpose(2, 1) 
        else:
            attn_out = self.attn_func(
                q, k, v
            )
            attn_out = attn_out.transpose(1, 2)

        attn_out = attn_out.reshape(mB, mN, mC)
        
        motion_out = self.motion_proj(attn_out)
        motion_out = self.motion_proj_drop(motion_out)
        
        if torch.get_autocast_dtype('cuda') == torch.float16:
            motion_out = motion_out.clip(-65504, 65504)
        
        return motion_out

class FinalLayer(nn.Module):
    """
    The final layer of Sana.
    """

    def __init__(
        self, 
        hidden_size, 
        out_channels,
        mapping_size=768,  # 新增：time embedding 的输入维度
        disable_modulation=False,
        bias=True,
        norm_type="rms",
        proj_type="linear",
        norm_eps=1e-6
    ):
        super().__init__()
        self.proj_type = proj_type
        if self.proj_type == "linear":
            self.proj = nn.Linear(hidden_size, out_channels, bias=bias)
        else:
            self.proj = nn.Conv1d(hidden_size, out_channels, stride=1, kernel_size=3, padding=1, bias=bias)
        self.disable_modulation = disable_modulation
        if self.disable_modulation:
            self.norm_final = build_norm(norm_type, num_features=hidden_size, affine=True, eps=norm_eps)
        else:
            self.norm_final = build_norm(norm_type, num_features=hidden_size, affine=False, eps=norm_eps)
            self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(mapping_size, 2 * hidden_size, bias=bias))


    def initialize_weights(self):
        if not self.disable_modulation:
            nn.init.constant_(self.modulation[1].weight, val=0)
            if self.modulation[1].bias is not None:
                nn.init.constant_(self.modulation[1].bias, val=0)
       
        nn.init.normal_(self.proj.weight, mean=0.0, std=0.02)
        if self.proj.bias is not None:
            nn.init.constant_(self.proj.bias, val=0)

    def forward(self, x, c):
        if self.disable_modulation:
            x = self.norm_final(x)
        else:
            shift, scale = self.modulation(c).chunk(2, dim=-1)
            x = t2i_modulate(self.norm_final(x), shift, scale)
        if self.proj_type == "linear":
            x = self.proj(x)
        else:
            x = self.proj(x.transpose(1, 2)).transpose(1, 2)
        return x

import torch
import torch.nn as nn
from einops import rearrange

class BlockWiseAttention(nn.Module):
    """
    Block-wise multi-head attention with absolute RoPE (offset by current_start).

    - Train:
        input is a full training sequence (B, N, C) that is expected to be divisible by block_size;
        apply absolute RoPE (via apply_rotary_emb + slicing rotary_emb with offset),
        then rearrange into blocks and do bi-directional attention within each block.

    - Inference:
        input is the current chunk/block only (B, T, C), T can be any length (e.g., 56 after seed cut);
        apply absolute RoPE on this chunk and do bi-directional attention within this chunk.
        No kv-cache, no mask, no rearrange, no divisibility requirement.
    """

    def __init__(
        self,
        dim,
        num_heads=8,
        head_dim=None,
        bias=True,
        attn_type=None,              # "flash" or "naive"
        qk_norm_type="identity",
        attn_drop=0.,
        proj_drop=0.,
        block_size=64,               # num_frame_per_block (train only)
        frame_seqlen=1,              # 1 token per frame by default
    ):
        super().__init__()
        assert (num_heads is not None and head_dim is None) or (num_heads is None and head_dim is not None), \
            "only one of num_heads and head_dim should be specified"
        if num_heads is not None:
            assert dim % num_heads == 0
            self.num_heads = num_heads
        else:
            assert dim % head_dim == 0
            self.num_heads = dim // head_dim

        assert attn_type in ["flash", "naive"]
        self.attn_type = attn_type
        self.block_size = int(block_size)
        self.frame_seqlen = int(frame_seqlen)

        self.m_qkv = nn.Linear(dim, dim * 3, bias=bias)
        self.attn_drop = float(attn_drop)
        self.motion_proj = nn.Linear(dim, dim, bias=bias)
        self.motion_proj_drop = nn.Dropout(proj_drop)

        self.mq_norm = build_norm(name=qk_norm_type, num_features=dim)
        self.mk_norm = build_norm(name=qk_norm_type, num_features=dim)

    def _apply_abs_rope(self, q, k, rotary_emb, current_start: int, is_causal: bool = False):
        """
        q,k: (B, N, H, Hd)
        rotary_emb: prebuilt for [0..rope_len-1]
        current_start: absolute token offset (in tokens)
        """
        
        if rotary_emb is None:
            return q, k
        if is_causal:
            start_frame = current_start // self.frame_seqlen
            q = apply_causal_rotary_emb(q, rotary_emb, use_real_unbind_dim=-1, start_frame=start_frame)
            k = apply_causal_rotary_emb(k, rotary_emb, use_real_unbind_dim=-1, start_frame=start_frame)
        else:
            q = apply_rotary_emb(q, rotary_emb, use_real_unbind_dim=-1)
            k = apply_rotary_emb(k, rotary_emb, use_real_unbind_dim=-1)
        return q, k

    def _attn(self, q, k, v, dropout_p: float):
        if self.attn_type == "flash":
            return attention(q, k, v, causal=False, dropout_p=dropout_p)
        elif self.attn_type == "naive":
            return vanilla_attention(q, k, v, causal=False, dropout_p=dropout_p)
        else:
            raise ValueError(f"Invalid attention type: {self.attn_type}")

    def forward(
        self,
        motion: torch.FloatTensor,          # (B, N, C)
        rotary_emb: torch.FloatTensor = None,
        current_start: int = 0,             # absolute token offset for RoPE
        is_inference: bool = None,          # None -> auto from self.training
        **kwargs,
    ):
        """
        kwargs deliberately accepted to be drop-in compatible with older call sites.
        In inference mode, kv_cache/mask args are ignored by design.
        """
        if is_inference is None:
            is_inference = (not self.training)

        B, N, C = motion.shape
        s = self.block_size

        qkv = self.m_qkv(motion).view(B, N, 3, C)
        q, k, v = qkv.unbind(2)
        dtype = q.dtype

        q = self.mq_norm(q)
        k = self.mk_norm(k)

        q = q.view(B, N, self.num_heads, C // self.num_heads).to(dtype)
        k = k.view(B, N, self.num_heads, C // self.num_heads).to(dtype)
        v = v.view(B, N, self.num_heads, C // self.num_heads).to(dtype)

        # (1) absolute-position RoPE on the original order (non-causal)
        q, k = self._apply_abs_rope(q, k, rotary_emb, current_start=current_start, is_causal=is_inference)

        dropout_p = self.attn_drop if self.training else 0.0

        if is_inference:
            # --------------------------------------------------
            # Inference: ONLY within current chunk, no rearrange,
            #            no cache, no mask, no divisibility req.
            # --------------------------------------------------
            attn_out = self._attn(q, k, v, dropout_p=dropout_p)  # (B, N, H, Hd)
            attn_out = attn_out.reshape(B, N, C)
            out = self.motion_proj_drop(self.motion_proj(attn_out))
            return out

        # ------------------------------------------------------
        # Train: block-parallel within-block bi-directional attn
        # Expect N divisible by block_size (your training pipeline
        # normally guarantees this via num_frame_per_block).
        # ------------------------------------------------------
        assert N % s == 0, (
            f"Train mode requires N % block_size == 0, got N={N}, block_size={s}. "
            f"If you want partial blocks in train, pad in dataloader/trainer."
        )

        nblk = N // s
        q_blk = rearrange(q, "b (nb s) h d -> (b nb) s h d", nb=nblk, s=s)
        k_blk = rearrange(k, "b (nb s) h d -> (b nb) s h d", nb=nblk, s=s)
        v_blk = rearrange(v, "b (nb s) h d -> (b nb) s h d", nb=nblk, s=s)

        attn_blk = self._attn(q_blk, k_blk, v_blk, dropout_p=dropout_p)  # (B*nb, s, H, Hd)

        attn_out = rearrange(attn_blk, "(b nb) s h d -> b (nb s) h d", b=B, nb=nblk, s=s)
        attn_out = attn_out.reshape(B, N, C)

        out = self.motion_proj_drop(self.motion_proj(attn_out))
        return out


# ============================================================================
# Factory Function for Cross-Attention Modules
# ============================================================================

def build_cross_attention(
    cross_attn_type: str = "standard",
    dim: int = 256,
    num_heads: int = 8,
    head_dim: int = None,
    bias: bool = True,
    qk_norm_type: str = "identity",
    attn_drop: float = 0.,
    proj_drop: float = 0.,
    attn_type: str = "flash",
    sink_size: int = 0,
    local_attn_size: int = -1,
    padding_unit_length: int = 128,
    is_teacher_forcing: bool = False,
    max_attention_size: int = 512,
    add_learnable_prefix: bool = False,
    # DeltaNet specific parameters
    block_size: int = 64,
    **kwargs,
) -> nn.Module:
    """
    Factory function to build cross-attention modules.
    
    Supports:
    - "standard": CrossCausalSelfAttention (softmax attention)
    - "deltanet", "deltanet_block", "deltanet_hierarchical": GatedDeltaNetForBLD (linear attention)
    
    Note: For DeltaNet, it's recommended to use CausalEmageBlockDeltaNet directly
    instead of this factory function, as it provides a cleaner interface.
    
    Args:
        cross_attn_type: Type of cross-attention module
        dim: Hidden dimension
        num_heads: Number of attention heads
        head_dim: Dimension per head (alternative to num_heads)
        bias: Whether to use bias in linear layers
        qk_norm_type: Type of normalization for Q/K
        attn_drop: Dropout rate for attention
        proj_drop: Dropout rate for output projection
        attn_type: Attention implementation type ("flash" or "vanilla")
        sink_size: Attention sink size
        local_attn_size: Local attention window size
        padding_unit_length: Padding unit for flex_attention
        is_teacher_forcing: Whether in teacher forcing mode
        max_attention_size: Maximum attention size for KV cache
        add_learnable_prefix: Whether to add learnable prefix token
        block_size: Block size for DeltaNet
        **kwargs: Additional arguments (ignored for compatibility)
    
    Returns:
        Cross-attention module
    
    Example:
        # Standard attention
        cross_attn = build_cross_attention("standard", dim=256, num_heads=8)
        
        # DeltaNet (all variants use the same implementation)
        cross_attn = build_cross_attention("deltanet", dim=256, num_heads=8)
    """
    if cross_attn_type == "standard":
        return CrossCausalSelfAttention(
            dim=dim,
            num_heads=num_heads,
            head_dim=head_dim,
            bias=bias,
            attn_type=attn_type,
            qk_norm_type=qk_norm_type,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            sink_size=sink_size,
            local_attn_size=local_attn_size,
            padding_unit_length=padding_unit_length,
            is_teacher_forcing=is_teacher_forcing,
            max_attention_size=max_attention_size,
            add_learnable_prefix=add_learnable_prefix,
        )
    elif cross_attn_type in ["deltanet_block", "deltanet_hierarchical", "deltanet"]:
        # All DeltaNet variants now use the unified GatedDeltaNetForBLD
        return GatedDeltaNetForBLD(
            dim=dim,
            num_heads=num_heads,
            head_dim=head_dim,
            bias=bias,
            qk_norm=(qk_norm_type != "identity"),  # Convert qk_norm_type to bool
            proj_drop=proj_drop,
            block_size=block_size,
        )
    else:
        raise ValueError(
            f"Unknown cross_attn_type: {cross_attn_type}. "
            f"Choose from 'standard', 'deltanet_block', or 'deltanet_hierarchical'"
        )


if __name__ == "__main__":
    from diffusion.model.nets.causal.causal_dit import CausalDITEmage
    from diffusion.model.nets.causal.causal_mmdit import CausalMMDITEmage
    from diffusion.model.nets.positional_embeding import RopePosEmbed
    motion = torch.randn(4, 60, 256).cuda()
    block_mask = CausalDITEmage._prepare_blockwise_causal_attn_mask(
        device=torch.device("cuda"), 
        num_frames = 60,
        frame_seqlen = 1
    )
    rotary_emb = RopePosEmbed(theta=10000, axes_dim=[32])
    rotary_emb = rotary_emb(torch.arange(0, 1024, device=torch.device("cuda")).unsqueeze(-1))
    kv_cache = None
    current_start = 0
    causal_attention = CausalSelfAttention(
        attn_type="flash",
        dim=256,
        num_heads=8,
        bias=True,
    )
    causal_attention.cuda()
    motion_out = causal_attention(motion, block_mask, rotary_emb, kv_cache, current_start)
    print(motion_out.shape)
    
    
    crossattn = CrossAttention(
        attn_type="flash",
        dim=256,
        num_heads=8,
        bias=True,
    )
    crossattn.cuda()
    context = torch.randn(4, 60, 256).cuda()
    crossattn_cache = None
    motion_out = crossattn(motion, context, crossattn_cache)
    print(motion_out.shape)
    
    block_mask2 = CausalMMDITEmage._prepare_teacher_forcing_mask(
        device=torch.device("cuda"), 
        num_frames = 60,
        frame_seqlen = 1
    )
    mm_causal_attention = MMCausalSelfAttention(
        attn_type="flash",
        dim=256,
        num_heads=8,
        bias=True,
    )
    motion2 = torch.randn(4, 60, 256).cuda()
    mm_causal_attention.cuda()
    motion_out, motion2_out = mm_causal_attention(motion, motion2, block_mask2, rotary_emb, kv_cache, current_start)
    print(motion_out.shape, motion2_out.shape)
