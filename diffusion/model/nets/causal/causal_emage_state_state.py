"""
State-State ablation for PASPA A1.

This file provides the "State-State" variant that replaces the block-internal
BlockWiseAttention with a token-level Gated DeltaNet (GDN) while keeping the
cross-block DeltaNet cross-attention unchanged. The goal is to contrast against:

    - State-Attention (current best): DeltaNet cross-attn + BlockWiseAttention
                                      self-attn   -> `CausalEmageBlockDeltaNet`
    - State-State (THIS file):       DeltaNet cross-attn + GDN self-attn

Design notes (approved in plan):

1.  Block isolation.  To match BlockWiseAttention semantics (bidirectional
    within-block, no information leakage across blocks) we reshape the sequence
    into non-overlapping blocks of size `num_frame_per_block` during training
    and call `fla.ops.gated_delta_rule.chunk_gated_delta_rule` on each block
    independently with `initial_state=None`.  At inference time a single block
    is fed in per call, so we simply run the same kernel on the current chunk.

2.  Token-level GDN.  We deliberately use the *standard* FLA kernel (which
    includes the intra-chunk `q @ k^T @ v` term) rather than the project's
    custom block-causal variant in `ops/chunk.py`.  The standard kernel chunks
    internally at size 64 -- with our `block_size=64` every block becomes one
    internal chunk, so the within-block attention is exact.

3.  Fused QKV + SiLU, no RoPE.  Mirrors `BlockWiseAttention.m_qkv` for fair
    parameter comparison.  GDN encodes position implicitly via the recurrence,
    so we do not apply RoPE inside the block (adding RoPE on top of GDN tends
    to hurt).

4.  Self-loop-only.  Self-attention has no cross-block state to maintain, so
    training and inference share a single forward path (`initial_state=None`).

The block class (`CausalEmageBlockStateState`) subclasses
`CausalEmageBlockDeltaNet` and only swaps `self.attn`/`self.attn2` out for
`GatedDeltaNetSelfAttn`.  Everything else (cross-attn, FFN, modulation, forward
flow) is inherited unchanged, so the ablation differs from State-Attention by
exactly one module type.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from diffusion.model.nets.causal.causal_emage import CausalEmageBlockDeltaNet


class GatedDeltaNetSelfAttn(nn.Module):
    """Block-internal Gated DeltaNet self-attention.

    Drop-in replacement for ``BlockWiseAttention`` with identical ``forward``
    signature.  ``rotary_emb`` and ``current_start`` are accepted but ignored
    (GDN does not use RoPE).

    Parameters
    ----------
    dim : int
        Hidden size.
    num_heads, head_dim : int
        Either ``num_heads`` or ``head_dim`` must be provided (exclusive).
    bias : bool
        Bias flag for the fused QKV / output projections.
    qk_norm_type : str
        Matches ``BlockWiseAttention`` interface.  ``"identity"`` means no Q/K
        normalisation; any other value enables FLA's in-kernel L2 norm.
    proj_drop : float
        Dropout on the output projection.
    block_size : int
        Block granularity used to split the training sequence before calling
        the GDN kernel.
    gate_low_rank_dim, gate_logit_normalizer, allow_neg_eigval, scale
        GDN gate hyper-parameters, semantics identical to
        ``GatedDeltaNetForBLD`` in ``gated_deltanet.py``.
    """

    def __init__(
        self,
        dim: int,
        num_heads: Optional[int] = 8,
        head_dim: Optional[int] = None,
        bias: bool = True,
        attn_type: Optional[str] = None,   # accepted for interface compat, ignored
        qk_norm_type: str = "identity",
        attn_drop: float = 0.0,            # accepted for interface compat, ignored
        proj_drop: float = 0.0,
        block_size: int = 64,
        frame_seqlen: int = 1,             # accepted for interface compat, unused
        # GDN gate parameters
        gate_low_rank_dim: int = -1,
        gate_logit_normalizer: float = 1.0,
        allow_neg_eigval: bool = False,
        scale: Optional[float] = None,
        **kwargs,
    ):
        super().__init__()
        assert (num_heads is not None and head_dim is None) or (
            num_heads is None and head_dim is not None
        ), "only one of num_heads and head_dim should be specified"
        if num_heads is not None:
            assert dim % num_heads == 0, f"dim={dim} must be divisible by num_heads={num_heads}"
            self.num_heads = int(num_heads)
        else:
            assert dim % head_dim == 0, f"dim={dim} must be divisible by head_dim={head_dim}"
            self.num_heads = dim // int(head_dim)
        self.head_dim = dim // self.num_heads
        self.dim = int(dim)
        self.block_size = int(block_size)

        # Q/K norm: identity -> no norm; otherwise use FLA's built-in L2 norm.
        self.qk_l2norm = qk_norm_type != "identity"

        # Scale (default 1/sqrt(head_dim)); mirrors FLA behaviour.
        self.scale = float(scale) if scale is not None else (1.0 / math.sqrt(self.head_dim))

        # Gate hyper-parameters
        self.gate_low_rank_dim = int(gate_low_rank_dim)
        self.gate_logit_normalizer = float(gate_logit_normalizer)
        self.allow_neg_eigval = bool(allow_neg_eigval)

        # ==============================================================
        # Parameters
        # ==============================================================
        # Fused QKV projection (mirrors BlockWiseAttention for fair compare)
        self.m_qkv = nn.Linear(dim, dim * 3, bias=bias)
        self.motion_proj = nn.Linear(dim, dim, bias=bias)
        self.motion_proj_drop = nn.Dropout(proj_drop)

        # Gate projections (follow GatedDeltaNetForBLD.compute_gates)
        if self.gate_low_rank_dim > 0:
            self.a_proj = nn.Sequential(
                nn.Linear(dim, self.gate_low_rank_dim, bias=False),
                nn.Linear(self.gate_low_rank_dim, self.num_heads, bias=True),
            )
            self.b_proj = nn.Sequential(
                nn.Linear(dim, self.gate_low_rank_dim, bias=False),
                nn.Linear(self.gate_low_rank_dim, self.num_heads, bias=True),
            )
        else:
            self.a_proj = nn.Linear(dim, self.num_heads, bias=True)
            self.b_proj = nn.Linear(dim, self.num_heads, bias=True)

        # A_log: per-head base decay (log-space), U(0, 16) init.
        A = torch.empty(self.num_heads, dtype=torch.float32).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True

        # dt_bias: inverse-softplus of dt in [dt_min, dt_max].
        dt_min, dt_max, dt_init_floor = 0.001, 0.1, 1e-4
        dt = torch.exp(
            torch.rand(self.num_heads) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        self.dt_bias._no_weight_decay = True

        self._init_weights()

    def _init_weights(self):
        # Gate biases start at zero so initial alpha ~= -A * softplus(dt_bias)
        # and beta ~= 0.5 -- matching FLA conventions.
        if self.gate_low_rank_dim > 0:
            nn.init.zeros_(self.a_proj[1].bias)
            nn.init.zeros_(self.b_proj[1].bias)
        else:
            nn.init.zeros_(self.a_proj.bias)
            nn.init.zeros_(self.b_proj.bias)
        # Output projection small init (common for residual blocks)
        nn.init.normal_(self.motion_proj.weight, std=0.02)
        if self.motion_proj.bias is not None:
            nn.init.zeros_(self.motion_proj.bias)

    # ------------------------------------------------------------------
    # Gate computation (replicates gated_deltanet.py:502-542)
    # ------------------------------------------------------------------
    def _compute_gates(self, x: torch.Tensor):
        """Compute (alpha_log, beta) from the pre-projection features.

        x: (B, T, D) -> alpha: (B, T, H), beta: (B, T, H).
        """
        beta = self.b_proj(x).sigmoid()
        if self.allow_neg_eigval:
            beta = beta * 2.0

        A = self.A_log.float().exp().to(x.dtype)  # (H,)
        a = self.a_proj(x) / self.gate_logit_normalizer  # (B, T, H)
        alpha = -A.view(1, 1, -1) * F.softplus(a + self.dt_bias.to(x.dtype).view(1, 1, -1))
        return alpha, beta

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        motion: torch.Tensor,           # (B, N, C)
        rotary_emb: torch.Tensor = None,  # ignored -- GDN has no RoPE
        current_start: int = 0,           # ignored -- no cross-block state
        is_inference: Optional[bool] = None,
        **kwargs,
    ) -> torch.Tensor:
        # Import locally to avoid import cost at module-load time.
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule

        if is_inference is None:
            is_inference = not self.training

        B, N, C = motion.shape
        H, D = self.num_heads, self.head_dim
        s = self.block_size

        # Fused QKV + SiLU (SiLU matches FLA's default q/k/v activation when
        # short-conv is disabled, cf. gated_deltanet.py).
        qkv = self.m_qkv(motion).view(B, N, 3, C)
        q, k, v = qkv.unbind(2)
        q = F.silu(q)
        k = F.silu(k)
        v = F.silu(v)

        # Reshape to (B, N, H, D_head)
        q = q.view(B, N, H, D)
        k = k.view(B, N, H, D)
        v = v.view(B, N, H, D)

        # Gates computed from the original input (matches GatedDeltaNetForBLD)
        alpha, beta = self._compute_gates(motion)  # (B, N, H), (B, N, H)

        # --------------------------------------------------------------
        # Block isolation reshape: (B, N, ...) -> (B*nblk, s, ...)
        # --------------------------------------------------------------
        if is_inference:
            # Inference: a single block (possibly < block_size) -- no reshape.
            q_blk, k_blk, v_blk = q, k, v
            alpha_blk, beta_blk = alpha, beta
            nblk = 1
        else:
            assert N % s == 0, (
                f"Train mode requires N % block_size == 0, got N={N}, block_size={s}"
            )
            nblk = N // s
            q_blk = rearrange(q, "b (nb s) h d -> (b nb) s h d", nb=nblk, s=s)
            k_blk = rearrange(k, "b (nb s) h d -> (b nb) s h d", nb=nblk, s=s)
            v_blk = rearrange(v, "b (nb s) h d -> (b nb) s h d", nb=nblk, s=s)
            alpha_blk = rearrange(alpha, "b (nb s) h -> (b nb) s h", nb=nblk, s=s)
            beta_blk = rearrange(beta, "b (nb s) h -> (b nb) s h", nb=nblk, s=s)

        # FLA's standard token-level kernel. `g` is interpreted as log-space
        # when `use_gate_in_kernel=False` (default), which matches our
        # externally-computed alpha.
        o_blk, _ = chunk_gated_delta_rule(
            q=q_blk,
            k=k_blk,
            v=v_blk,
            g=alpha_blk,
            beta=beta_blk,
            scale=self.scale,
            initial_state=None,
            output_final_state=False,
            use_qk_l2norm_in_kernel=self.qk_l2norm,
        )

        if is_inference:
            o = o_blk
        else:
            o = rearrange(o_blk, "(b nb) s h d -> b (nb s) h d", b=B, nb=nblk, s=s)

        o = o.reshape(B, N, C)
        return self.motion_proj_drop(self.motion_proj(o))


class CausalEmageBlockStateState(CausalEmageBlockDeltaNet):
    """Block variant combining DeltaNet cross-attn with GDN self-attn.

    Inherits the entire forward logic from ``CausalEmageBlockDeltaNet`` and
    only replaces the block-internal self-attention modules.  Because
    ``GatedDeltaNetSelfAttn.forward`` mirrors the ``BlockWiseAttention``
    signature (same positional + keyword arguments), the parent forward runs
    unchanged.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 8,
        head_dim: Optional[int] = None,
        bias: bool = True,
        qk_norm_type: str = "identity",
        attn_drop: float = 0.0,
        attn_proj_drop: float = 0.0,
        num_frame_per_block: int = 64,
        # Self-attention GDN gate hyper-parameters (default match cross-attn)
        self_deltanet_gate_low_rank_dim: int = -1,
        self_deltanet_gate_logit_normalizer: float = 1.0,
        self_deltanet_allow_neg_eigval: bool = False,
        self_deltanet_scale: Optional[float] = None,
        **kwargs,
    ):
        # Delegate everything else (cross-attn, FFN, modulation, norms, ...)
        # to the parent, which also instantiates BlockWiseAttention into
        # self.attn/self.attn2.  We'll overwrite them right below.
        super().__init__(
            hidden_size=hidden_size,
            num_heads=num_heads,
            head_dim=head_dim,
            bias=bias,
            qk_norm_type=qk_norm_type,
            attn_drop=attn_drop,
            attn_proj_drop=attn_proj_drop,
            num_frame_per_block=num_frame_per_block,
            **kwargs,
        )

        if getattr(self, "use_timestep_fusion", False):
            raise NotImplementedError(
                "CausalEmageBlockStateState does not support use_timestep_fusion=True. "
                "The parallel Self/Cross-Attn fusion branch is out of scope for this "
                "ablation; please use the serial architecture."
            )

        # Swap BlockWiseAttention -> GatedDeltaNetSelfAttn.
        self.attn = GatedDeltaNetSelfAttn(
            dim=hidden_size,
            num_heads=num_heads,
            head_dim=head_dim,
            bias=bias,
            qk_norm_type=qk_norm_type,
            proj_drop=attn_proj_drop,
            block_size=num_frame_per_block,
            gate_low_rank_dim=self_deltanet_gate_low_rank_dim,
            gate_logit_normalizer=self_deltanet_gate_logit_normalizer,
            allow_neg_eigval=self_deltanet_allow_neg_eigval,
            scale=self_deltanet_scale,
        )
        self.attn2 = GatedDeltaNetSelfAttn(
            dim=hidden_size,
            num_heads=num_heads,
            head_dim=head_dim,
            bias=bias,
            qk_norm_type=qk_norm_type,
            proj_drop=attn_proj_drop,
            block_size=num_frame_per_block,
            gate_low_rank_dim=self_deltanet_gate_low_rank_dim,
            gate_logit_normalizer=self_deltanet_gate_logit_normalizer,
            allow_neg_eigval=self_deltanet_allow_neg_eigval,
            scale=self_deltanet_scale,
        )
