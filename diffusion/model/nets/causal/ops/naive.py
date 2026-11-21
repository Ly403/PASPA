# -*- coding: utf-8 -*-
"""
Pure PyTorch reference implementations for block-causal gated DeltaNet.

These kernels are extracted from gated_deltanet.py for clarity.
They are mathematically equivalent to the Triton kernels but use native
PyTorch ops and autograd, making them easier to read and debug.

Reference: flash-linear-attention (https://github.com/fla-org/flash-linear-attention)
"""

import torch
import torch.nn.functional as F
from einops import rearrange
from typing import Optional, Tuple


def _maybe_normalize_qk(
    q: Optional[torch.Tensor],
    k: Optional[torch.Tensor],
    enabled: bool = True,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    L2 normalize Q and K if enabled.

    Note:
    - q can be None for prefill scenarios where only k needs normalization.
    - k can be None for decode scenarios where only q needs normalization.
    """
    if not enabled:
        return q, k
    # Use native dtype for normalization - F.normalize handles all dtypes
    # Only normalize each tensor if it's not None
    if q is not None:
        q = F.normalize(q, p=2, dim=-1, eps=1e-6)
    if k is not None:
        k = F.normalize(k, p=2, dim=-1, eps=1e-6)
    return q, k


def _upcast_state_if_needed(state: torch.Tensor) -> torch.Tensor:
    """
    Upcast state to float32 if it's in half precision for numerical stability.
    If state is already float32 or higher, leave it unchanged.
    """
    if state.dtype in (torch.float16, torch.bfloat16):
        return state.float()
    return state


def recurrent_write_states(
    initial_state: torch.Tensor,  # (B, H, K, V), will be upcast to float32 if half precision
    k: torch.Tensor,              # (B, T, H, K)
    v: torch.Tensor,              # (B, T, H, V)
    g: torch.Tensor,              # (B, T, H)
    beta: torch.Tensor,           # (B, T, H)
    qk_norm: bool = False,
) -> torch.Tensor:
    """
    Write-only recurrent update for DeltaNet state.

    dtype 策略：
    - state 始终 fp32
    - k/v/g/beta 在进入循环前 cast 到 fp32
    - 所有 einsum/exp 在 fp32 下计算
    - 使用 autocast disabled 包住循环主体（CUDA 时）
    - 返回 fp32 state

    Update formula for each token t:
    1. Decay:  state = alpha_t * state
    2. Predict: pred = einsum("bhkv,bhk->bhv", state, k_t)
    3. Delta:   u = (v_t - pred) * beta_t
    4. Write:   state = state + einsum("bhk,bhv->bhkv", k_t, u)

    Args:
        initial_state: Current state, shape (B, H, K, V). Will be upcast to float32 if half precision.
        k: Key vectors, shape (B, T, H, K)
        v: Value vectors, shape (B, T, H, V)
        g: Log decay factor, shape (B, T, H). alpha = exp(g)
        beta: Write strength, shape (B, T, H), in [0, 1] or [0, 2]
        scale: Kept for API consistency with block_causal_reference_naive.
        qk_norm: Whether to L2 normalize k (to match training semantics).

    Returns:
        Updated state, shape (B, H, K, V), dtype float32 (for numerical stability)
    """
    B, T, H, K = k.shape
    V = v.shape[-1]
    compute_dtype = torch.float32

    # Keep qk_norm behavior: normalize k only (no q in this write-only function)
    _, k = _maybe_normalize_qk(q=None, k=k, enabled=qk_norm)

    # Upcast state to float32 for numerical stability
    state = _upcast_state_if_needed(initial_state)

    # Cast k/v/g/beta to fp32 before entering the loop
    k_f = k.to(compute_dtype)
    v_f = v.to(compute_dtype)
    g_f = g.to(compute_dtype)
    beta_f = beta.to(compute_dtype)

    # Determine device type for autocast
    device_type = 'cuda' if k.is_cuda else 'cpu'

    # Disable autocast for the recurrent loop to ensure fp32 computation
    with torch.autocast(device_type=device_type, enabled=False):
        for t in range(T):
            k_t = k_f[:, t]  # (B, H, K), fp32
            v_t = v_f[:, t]  # (B, H, V), fp32
            alpha_t = torch.exp(g_f[:, t])  # (B, H), fp32
            beta_t = beta_f[:, t]  # (B, H), fp32

            # 1. Decay
            state = alpha_t.unsqueeze(-1).unsqueeze(-1) * state  # (B, H, K, V), fp32
            # 2. Predict
            pred = torch.einsum('bhkv,bhk->bhv', state, k_t)  # (B, H, V), fp32
            # 3. Delta
            u = (v_t - pred) * beta_t.unsqueeze(-1)  # (B, H, V), fp32
            # 4. Write
            state = state + torch.einsum('bhk,bhv->bhkv', k_t, u)  # (B, H, K, V), fp32

    return state


def block_causal_reference_naive(
    q: torch.Tensor,  # (B, T, H, K)
    k: torch.Tensor,  # (B, T, H, K)
    v: torch.Tensor,  # (B, T, H, V)
    g: torch.Tensor,  # (B, T, H) log alpha
    beta: torch.Tensor,  # (B, T, H)
    chunk_size: int = 64,
    scale: float = None,
    initial_state: Optional[torch.Tensor] = None,  # (B, H, K, V)
    output_final_state: bool = False,
    qk_norm: bool = True,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Reference naive implementation: token-level loop for state update,
    block-level read for output (block-causal autoregressive).

    dtype 策略：
    - state 始终 fp32（无论输入 dtype）
    - q/k/v/g/beta 在 normalize 后保持原 dtype
    - block read: q_block cast 到 fp32 与 H0 做 einsum，结果 cast 回 q.dtype
    - token update: k_t/v_t/g_t/beta_t cast 到 fp32 参与计算
    - 使用 autocast disabled 包住 block 循环主体（CUDA 时）
    - 输出 o: q.dtype, state: fp32
    """
    B, T, H, K = q.shape
    V = v.shape[-1]
    input_dtype = q.dtype
    compute_dtype = torch.float32

    # qk_norm: L2 normalize q and k (in original dtype, before any loop)
    if qk_norm:
        q = F.normalize(q, p=2, dim=-1, eps=1e-6)
        k = F.normalize(k, p=2, dim=-1, eps=1e-6)

    # scale: default 1/sqrt(K), applied to q in block-read (locally cast to fp32)
    if scale is None:
        scale = 1 / (K ** 0.5)

    # State always fp32 for numerical stability
    if initial_state is None:
        state = torch.zeros(B, H, K, V, device=q.device, dtype=compute_dtype)
    else:
        state = initial_state.to(compute_dtype)

    o = torch.zeros(B, T, H, V, device=q.device, dtype=input_dtype)
    num_blocks = (T + chunk_size - 1) // chunk_size

    # Determine device type for autocast
    device_type = 'cuda' if q.is_cuda else 'cpu'

    # Disable autocast for the block loop to ensure fp32 computation
    with torch.autocast(device_type=device_type, enabled=False):
        for block_idx in range(num_blocks):
            start = block_idx * chunk_size
            end = min(start + chunk_size, T)

            H0 = state  # fp32, block-start state

            # Block-causal read: output uses block-start state
            # Cast q_block to fp32 for einsum with fp32 state, then cast back
            q_block = q[:, start:end]  # (B, C, H, K), original dtype
            q_block_f = q_block.to(compute_dtype) * scale
            o_block_f = torch.einsum('bchk,bhkv->bchv', q_block_f, H0)  # fp32
            o[:, start:end] = o_block_f.to(input_dtype)

            # Token-level state update (all in fp32)
            for t in range(start, end):
                k_t = k[:, t].to(compute_dtype)  # (B, H, K), fp32
                v_t = v[:, t].to(compute_dtype)  # (B, H, V), fp32
                alpha_t = torch.exp(g[:, t].to(compute_dtype))  # (B, H), fp32
                beta_t = beta[:, t].to(compute_dtype)  # (B, H), fp32

                state = alpha_t.unsqueeze(-1).unsqueeze(-1) * state  # fp32
                pred  = torch.einsum('bhkv,bhk->bhv', state, k_t)  # fp32
                delta = (v_t - pred) * beta_t.unsqueeze(-1)  # fp32
                state = state + torch.einsum('bhk,bhv->bhkv', k_t, delta)  # fp32

    if not output_final_state:
        state = None

    return o, state


def block_causal_fast_chunk(
    q: torch.Tensor,  # (B, T, H, K)
    k: torch.Tensor,  # (B, T, H, K)
    v: torch.Tensor,  # (B, T, H, V)
    g: torch.Tensor,  # (B, T, H) log alpha
    beta: torch.Tensor,  # (B, T, H)
    chunk_size: int = 64,
    scale: float = None,
    initial_state: torch.Tensor = None,  # (B, H, K, V)
    output_final_state: bool = False,
    qk_norm: bool = True,
    history_cfg_alpha: Optional[torch.Tensor] = None,  # (B, num_chunks) float in [0,1]: 1=clean state, 0=null state
    null_state: Optional[torch.Tensor] = None,  # (H, K, V) or (B, H, K, V): learnable null state
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Block-causal chunkwise implementation using FLA chunk_gated_delta_rule_ref logic.

    Output: block-level read (chunk-start state), state update: FLA-style chunkwise.

    Supports read-side null-history interpolation for history-state CFG:
    - history_cfg_alpha: (B, num_chunks) float tensor in [0,1].
      alpha=1 means read real state, alpha=0 means read null_state.
    - null_state: (H, K, V) or (B, H, K, V) learnable null state parameter.
    When both are provided, h_read = alpha * h_real + (1-alpha) * null_state.
    State update chain is NOT affected.

    dtype 策略：
    - state 始终 fp32
    - q/k/v/g/beta 在 normalize 后保持原 dtype
    - block read: q_i cast 到 fp32 与 fp32 state 做 @，结果 cast 回 q.dtype
    - chunk update: k_i/v_i/g_i/beta_i 全部 cast 到 fp32
    - 所有中间量（decay/L_mask/attn/k_cumsum/k_cumdecay/v_prime/v_new 等）都在 fp32
    - 输出 o: q.dtype, state: fp32
    """
    BT = chunk_size
    input_dtype = q.dtype
    compute_dtype = torch.float32

    if scale is None:
        scale = 1 / (q.shape[-1] ** 0.5)

    # Step 2: qk_norm L2 normalize (in original dtype)
    if qk_norm:
        q = F.normalize(q, p=2, dim=-1, eps=1e-6)
        k = F.normalize(k, p=2, dim=-1, eps=1e-6)

    # Transpose to (B, H, T, D) layout (FLA convention), keep original dtype
    q_bhtk = q.transpose(1, 2).contiguous()  # (B, H, T, K)
    k_bhtk = k.transpose(1, 2).contiguous()  # (B, H, T, K)
    v_bhtv = v.transpose(1, 2).contiguous()  # (B, H, T, V)
    beta_bht = beta.transpose(1, 2).contiguous()  # (B, H, T) -- beta is (B,T,H)
    g_bht = g.transpose(1, 2).contiguous()  # (B, H, T) -- g is (B,T,H)

    B, H, T_orig, K = q_bhtk.shape
    V = v_bhtv.shape[-1]

    # Pad to multiple of BT
    pad_len = (BT - (T_orig % BT)) % BT
    if pad_len > 0:
        q_bhtk = F.pad(q_bhtk, (0, 0, 0, pad_len))
        k_bhtk = F.pad(k_bhtk, (0, 0, 0, pad_len))
        v_bhtv = F.pad(v_bhtv, (0, 0, 0, pad_len))
        beta_bht = F.pad(beta_bht, (0, pad_len))
        g_bht = F.pad(g_bht, (0, pad_len))

    T_padded = q_bhtk.shape[2]

    # ===== Cast to fp32 for chunk update computation =====
    q_f = q_bhtk.to(compute_dtype)
    k_f = k_bhtk.to(compute_dtype)
    v_f = v_bhtv.to(compute_dtype)
    beta_f = beta_bht.to(compute_dtype)
    decay_f = g_bht.to(compute_dtype)

    # Apply scale to q
    q_f = q_f * scale

    # v_scaled = v * beta
    v_scaled = v_f * beta_f[..., None]
    # k_beta = k * beta
    k_beta = k_f * beta_f[..., None]

    # Reshape into chunks: (B, H, num_chunks, chunk_size, D)
    num_chunks = T_padded // BT
    q_c, k_c, v_c, k_beta_c, decay_c = map(
        lambda x: rearrange(x, 'b h (n c) d -> b h n c d', c=BT),
        [q_f, k_f, v_scaled, k_beta, decay_f.unsqueeze(-1)],
    )
    decay_c = decay_c.squeeze(-1).cumsum(-1)  # (B, H, num_chunks, BT)

    # L_mask: lower-triangular decay matrix per chunk
    L_mask = ((decay_c.unsqueeze(-1) - decay_c.unsqueeze(-2)).tril().exp().float()).tril()

    # Upper-triangular mask (diagonal included) for masking attn
    mask_upper = torch.triu(torch.ones(BT, BT, dtype=torch.bool, device=q.device), diagonal=0)

    # attn = -(k_beta @ k^T * L_mask), masked upper triangle to 0
    attn = -((k_beta_c @ k_c.transpose(-1, -2)) * L_mask).masked_fill(mask_upper, 0)
    for i in range(1, BT):
        attn[..., i, :i] = attn[..., i, :i].clone() + (attn[..., i, :i, None].clone() * attn[..., :i, :i].clone()).sum(-2)
    attn = attn + torch.eye(BT, dtype=compute_dtype, device=q.device)

    # k_cumsum = attn @ v_scaled (the "new v" after WY-like transform)
    k_cumsum = attn @ v_c
    # k_cumdecay = attn @ (k_beta * decay_exp)
    decay_exp = decay_c.exp()[..., None]
    k_cumdecay = attn @ (k_beta_c * decay_exp)

    # v_new is the transformed v
    v_new = k_cumsum

    # Initialize state
    S = q.new_zeros(B, H, K, V, dtype=compute_dtype)
    if initial_state is not None:
        S = initial_state.to(compute_dtype)

    o_chunks = torch.zeros_like(v_new)  # (B, H, num_chunks, BT, V), fp32

    # Determine device type for autocast
    device_type = 'cuda' if q.is_cuda else 'cpu'

    with torch.autocast(device_type=device_type, enabled=False):
        # Prepare null_state for read-side history alpha interpolation if provided
        null_state_f = None
        if history_cfg_alpha is not None and null_state is not None:
            # null_state: (H, K, V) or (B, H, K, V) -> ensure (B, H, K, V) fp32
            ns = null_state.to(compute_dtype)
            if ns.dim() == 3:
                ns = ns.unsqueeze(0).expand(B, -1, -1, -1)
            null_state_f = ns

        for i in range(num_chunks):
            q_i = q_c[:, :, i]       # (B, H, BT, K), fp32
            k_i = k_c[:, :, i]       # (B, H, BT, K), fp32
            v_i = v_new[:, :, i]     # (B, H, BT, V), fp32

            # Block-causal read: output uses chunk-start state S0
            S0 = S  # fp32, before any update in this chunk

            # ===== Read-side null-history alpha interpolation =====
            # h_read = alpha * h_real + (1 - alpha) * null_state
            if null_state_f is not None and history_cfg_alpha is not None:
                alpha_i = history_cfg_alpha[:, i] if history_cfg_alpha.dim() > 1 else history_cfg_alpha
                # alpha_i: (B,) float -> (B, 1, 1, 1) for broadcasting with (B, H, K, V)
                alpha_bc = alpha_i.to(dtype=compute_dtype, device=S0.device).view(B, 1, 1, 1)
                S0_read = S0 * alpha_bc + null_state_f * (1.0 - alpha_bc)
            else:
                S0_read = S0

            o_i = q_i @ S0_read  # (B, H, BT, V), fp32

            # Write back output (fp32, will be cast later)
            o_chunks[:, :, i] = o_i

            # State update (FLA logic, all fp32)
            v_prime = k_cumdecay[:, :, i] @ S  # (B, H, BT, V), fp32
            v_delta = v_i - v_prime  # (B, H, BT, V), fp32

            S = S * decay_c[:, :, i, -1, None, None].exp() + \
                (k_i * (decay_c[:, :, i, -1, None] - decay_c[:, :, i]).exp()[..., None]).transpose(-1, -2) @ v_delta

    # Unpad and transpose back to (B, T, H, V)
    o_out = rearrange(o_chunks, 'b h n c d -> b h (n c) d')
    o_out = o_out[:, :, :T_orig]  # remove padding
    o_out = o_out.transpose(1, 2).to(input_dtype)  # (B, T, H, V), original dtype

    if not output_final_state:
        S = None

    return o_out, S
