"""DeltaNet variant without Prefill-Decode separation, for inference latency benchmarking.

Production GatedDeltaNetForBLD splits inference into two specialized paths:
  - `_forward_prefill` calls `chunk_gated_delta_rule_prefill` — state-only
    kernel (no output computation; cheaper than the full kernel)
  - `_forward_decode` does a simple `einsum(q, cached_state)` — read-only
    (no state update, no k_proj/v_proj/gates)

The "no PD separation" baseline removes this specialization. There is only
ONE operation: `chunk_gated_delta_rule` (the same kernel used in training,
see `_forward_train` in gated_deltanet.py:708). It computes state update AND
output simultaneously. Both decode and prefill paths must use it; the prefill
caller discards its output.

Pipeline-level call account per block (block_size=64, sampler steps N=10):

    PASPA (production):
        10 × _forward_decode      (cheap einsum, no state update)
         1 × _forward_prefill     (state-only kernel)
        ≈ 10 × cheap + 1 × medium

    No-PD (this module):
        10 × _forward_decode      (full fused kernel: state + output)
         1 × _forward_prefill     (full fused kernel: state + output, output discarded)
        = 11 × full fused kernel

The end-of-block prefill is *required* — it writes the *denoised clean
prediction* into state. The 10 decode-step writes used noisy intermediate
predictions, leaving state in a corrupted regime; the final prefill is the
single "clean" state update of the block. Skipping it would desync every
subsequent block's state semantics.

Inference / bench artifact only — never used in training. Generated motion
quality during the bench is meaningless (random init + noisy intermediate
state writes); we measure wall-clock latency / peak memory only. CFG is
disabled throughout the efficiency experiment (note in figure caption).

References (production code, NOT modified):
    diffusion/model/nets/causal/gated_deltanet.py
        - class GatedDeltaNetForBLD (line 46)
        - _forward_train  (line 604)  — calls chunk_gated_delta_rule (line 708)
        - _forward_prefill (line 749) — calls chunk_gated_delta_rule_prefill (line 845)
        - _forward_decode  (line 875) — does Q @ state einsum only
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn.functional as F

from diffusion.model.nets.causal.causal_emage import (
    CausalEmage,
    CausalEmageBlockDeltaNet,
)
from diffusion.model.nets.causal.gated_deltanet import GatedDeltaNetForBLD
from diffusion.model.nets.causal.ops.chunk import chunk_gated_delta_rule
from diffusion.model.nets.causal.ops.naive import (
    _upcast_state_if_needed,
    block_causal_fast_chunk,
)


class _GatedDeltaNetForBLDNoPD(GatedDeltaNetForBLD):
    """Cross-attention without Prefill-Decode separation.

    Both ``_forward_decode`` and ``_forward_prefill`` route through the same
    ``_forward_fused`` helper, which calls the full ``chunk_gated_delta_rule``
    kernel (state update + output). Prefill discards its output.

    Only ``_forward_decode`` and ``_forward_prefill`` are overridden. Everything
    else (the public ``forward`` dispatcher, ``_forward_train``, ``compute_gates``,
    ``_apply_time_modulation``, ``_get_initial_state``, ...) is inherited from
    the parent class and unchanged.
    """

    # ------------------------------------------------------------------
    # Overrides of GatedDeltaNetForBLD's two specialized inference paths
    # ------------------------------------------------------------------

    def _forward_decode(
        self,
        motion: torch.Tensor,
        kv_cache: Dict,
        current_start: int,
        t_emb: Optional[torch.Tensor] = None,
        history_cfg_alpha: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # `history_cfg_alpha` ignored (CFG disabled for the efficiency
        # experiment; this is mentioned in the paper figure caption).
        return self._forward_fused(motion, kv_cache, current_start, t_emb=t_emb)

    def _forward_prefill(
        self,
        context: torch.Tensor,
        kv_cache: Dict,
        current_start: int,
        kv_t_emb: Optional[torch.Tensor] = None,
    ) -> None:
        # In a no-PD world there is no specialized cheap prefill kernel — the
        # end-of-block state update (with the *denoised clean prediction*) must
        # use the same fused kernel as decode. The output is discarded; only
        # the state update is meaningful.
        # `kv_t_emb` ignored: KV time modulation is unused throughout this codebase.
        if context is None:
            return None
        _ = self._forward_fused(context, kv_cache, current_start, t_emb=None)
        return None

    # ------------------------------------------------------------------
    # The single fused operation (state + output)
    # ------------------------------------------------------------------

    def _forward_fused(
        self,
        x: torch.Tensor,
        kv_cache: Dict,
        current_start: int,
        t_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Single fused operation: ``chunk_gated_delta_rule`` (state + output).

        Mirrors ``_forward_train``'s kernel call pattern, but K/V are derived
        from the same input tensor as Q (in this no-PD baseline there is no
        separate clean context; ``x`` plays both roles).

        Side effects on ``kv_cache`` (mirror ``_forward_prefill``):
            - ``deltanet_state``         : updated with new state
            - ``k_conv_cache`` / ``v_conv_cache`` : updated when shortconv on
            - ``global_end_index`` / ``local_end_index`` : advanced to (current_start + T)

        Returns:
            output: (B, T, D) — for decode caller, this is the per-frame
                    cross-attention output the sampler will consume; for
                    prefill caller, the value is discarded.
        """
        B, T, D = x.shape

        # ---- Q (mirrors production decode q-path: q_proj + SiLU + time mod) ----
        q = F.silu(self.q_proj(x))
        q = self._apply_time_modulation(
            x=q,
            t_emb=t_emb,
            mode=self.q_t_modulation_mode,
            expected_dim=self.q_t_modulation_dim,
            emb_name="t_emb (query)",
            additive_proj=getattr(self, "q_t_proj", None),
            gamma_proj=getattr(self, "q_t_gamma", None),
            beta_proj=getattr(self, "q_t_beta", None),
        )
        q = q.unflatten(-1, (self.num_heads, self.head_dim))  # (B, T, H, D_head)

        # ---- K, V with shortconv cache (mirrors _forward_prefill:765-791) ----
        # The cache is updated on every fused call (11 times/block in no-PD vs
        # 1 time/block in PASPA). This faithfully reflects "every step does the
        # full kv_proj + shortconv work that prefill does in production".
        if self.use_short_conv:
            k_conv_cache = kv_cache.get("k_conv_cache", None)
            v_conv_cache = kv_cache.get("v_conv_cache", None)
            if k_conv_cache is not None and k_conv_cache.shape[0] != B:
                k_conv_cache = None
            if v_conv_cache is not None and v_conv_cache.shape[0] != B:
                v_conv_cache = None
            k, k_conv_cache_new = self.k_conv1d(
                x=self.k_proj(x), cache=k_conv_cache, output_final_state=True,
            )
            v, v_conv_cache_new = self.v_conv1d(
                x=self.v_proj(x), cache=v_conv_cache, output_final_state=True,
            )
            kv_cache["k_conv_cache"] = k_conv_cache_new
            kv_cache["v_conv_cache"] = v_conv_cache_new
        else:
            k = F.silu(self.k_proj(x))
            v = F.silu(self.v_proj(x))
        k = k.unflatten(-1, (self.num_heads, self.head_dim))  # (B, T, H, D_head)
        v = v.unflatten(-1, (self.num_heads, self.head_dim))  # (B, T, H, D_head)

        # ---- Per-frame gates ----
        alpha, beta = self.compute_gates(x)  # (B, T, H), (B, T, H)

        # ---- State (init zeros / learnable_init if absent) ----
        state = kv_cache.get("deltanet_state", None)
        if state is None:
            init = self._get_initial_state(B, k.device)
            if init is not None:
                state = init
            else:
                state = torch.zeros(
                    B, self.num_heads, self.head_dim, self.head_dim,
                    device=k.device, dtype=torch.float32,
                )
        state = _upcast_state_if_needed(state)

        # ---- Single fused kernel call: state update + per-frame output ----
        # CFG is disabled for the efficiency experiment, so history_cfg_alpha
        # / null_state are intentionally not passed (defaults: None).
        if self.use_triton_kernel:
            o, new_state = chunk_gated_delta_rule(
                q=q, k=k, v=v, g=alpha, beta=beta,
                chunk_size=self.block_size,
                scale=self.scale,
                initial_state=state,
                output_final_state=True,
                use_qk_l2norm_in_kernel=self.qk_norm,
            )
        else:
            o, new_state = block_causal_fast_chunk(
                q=q, k=k, v=v, g=alpha, beta=beta,
                chunk_size=self.block_size,
                scale=self.scale,
                initial_state=state,
                output_final_state=True,
                qk_norm=self.qk_norm,
            )

        # ---- Persist state + housekeeping (mirrors _forward_prefill) ----
        kv_cache["deltanet_state"] = new_state
        if "global_end_index" in kv_cache:
            kv_cache["global_end_index"].fill_(current_start + T)
        if "local_end_index" in kv_cache:
            kv_cache["local_end_index"].fill_(current_start + T)

        # ---- Output projection (run unconditionally; prefill caller discards) ----
        o = o.flatten(-2)  # (B, T, H * D_head)
        o = self.o_proj(o)
        o = self.proj_drop(o)
        if o.dtype == torch.float16:
            o = o.clamp(-65504, 65504)
        return o


class CausalEmageNoPD(CausalEmage):
    """CausalEmage variant ablating Prefill-Decode separation at the kernel level.

    In each DeltaNet block, the ``cross_attn`` and ``cross_attn2`` modules'
    ``__class__`` is reassigned to :class:`_GatedDeltaNetForBLDNoPD`.
    ``__class__`` reassignment changes method dispatch only; all weights,
    sub-modules, buffers, and forward hooks are preserved.

    Functionally identical in *training* (``forward_train`` inherited from
    :class:`CausalEmage` is unchanged, and the cross-attention's training
    path ``_forward_train`` is also inherited from
    :class:`GatedDeltaNetForBLD`). Differs only at *inference*: every cross-
    attention call (both sampler decode steps and end-of-block prefill) routes
    through :meth:`_GatedDeltaNetForBLDNoPD._forward_fused`, the full kernel.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        for block in self.blocks:
            if not isinstance(block, CausalEmageBlockDeltaNet):
                continue
            if isinstance(block.cross_attn, GatedDeltaNetForBLD):
                block.cross_attn.__class__ = _GatedDeltaNetForBLDNoPD
            if isinstance(block.cross_attn2, GatedDeltaNetForBLD):
                block.cross_attn2.__class__ = _GatedDeltaNetForBLDNoPD
