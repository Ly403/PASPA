# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

import warnings

import torch

from fla.modules.l2norm import l2norm_bwd, l2norm_fwd
from fla.ops.common.chunk_scaled_dot_kkt import chunk_scaled_dot_kkt_fwd
from diffusion.model.nets.causal.ops.wy_fast import prepare_wy_repr_bwd, recompute_w_u_fwd
from fla.ops.utils import chunk_local_cumsum, solve_tril
from fla.utils import autocast_custom_bwd, autocast_custom_fwd, input_guard
from diffusion.model.nets.causal.ops.chunk_delta_h import chunk_gated_delta_rule_bwd_dhu, chunk_gated_delta_rule_fwd_h
from diffusion.model.nets.causal.ops.chunk_block_o import chunk_bwd_dqkwg, chunk_fwd_o_block_causal


def chunk_gated_delta_rule_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    output_final_state: bool,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    history_cfg_alpha: torch.Tensor | None = None,  # (B, num_chunks) float32
    null_state: torch.Tensor | None = None,  # (H, K, V)
):
    g = chunk_local_cumsum(g, chunk_size=chunk_size, cu_seqlens=cu_seqlens)
    # obtain WY representation. u is actually the new v.
    A = chunk_scaled_dot_kkt_fwd(
        k=k,
        g=g,
        beta=beta,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
        output_dtype=torch.float32,
    )
    A = solve_tril(
        A=A,
        cu_seqlens=cu_seqlens,
        output_dtype=k.dtype,
    )
    w, u = recompute_w_u_fwd(
        k=k,
        v=v,
        beta=beta,
        A=A,
        g=g,
        cu_seqlens=cu_seqlens,
    )

    h, v_new, final_state = chunk_gated_delta_rule_fwd_h(
        k=k,
        w=w,
        u=u,
        g=g,
        initial_state=initial_state,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
        save_new_value=False,
    )

    # Block-causal only: outputs read from block-start state, no block-internal terms.
    # Null-history read-side selection is fused inside the kernel.
    o = chunk_fwd_o_block_causal(
        q=q,
        h=h,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
        history_cfg_alpha=history_cfg_alpha,
        null_state=null_state,
    )
    return g, o, A, final_state, initial_state


def chunk_gated_delta_rule_bwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    A: torch.Tensor,
    scale: float,
    initial_state: torch.Tensor,
    do: torch.Tensor,
    dht: torch.Tensor,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    history_cfg_alpha: torch.Tensor | None = None,
    null_state: torch.Tensor | None = None,
):
    w, u = recompute_w_u_fwd(
        k=k,
        v=v,
        beta=beta,
        A=A,
        g=g,
        cu_seqlens=cu_seqlens,
    )

    h, v_new, _ = chunk_gated_delta_rule_fwd_h(
        k=k,
        w=w,
        u=u,
        g=g,
        initial_state=initial_state,
        output_final_state=False,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
    )

    # Block-causal: output has no chunk-internal attention, so dv is computed
    # purely from the state chain inside bwd_dhu (no external dv_local input).
    dh, dh0, dv, dnull = chunk_gated_delta_rule_bwd_dhu(
        q=q,
        k=k,
        w=w,
        g=g,
        h0=initial_state,
        dht=dht,
        do=do,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
        history_cfg_alpha=history_cfg_alpha,
        null_state=null_state,
    )
    dq, dk, dw, dg = chunk_bwd_dqkwg(
        q=q,
        k=k,
        v=v_new,
        w=w,
        g=g,
        h=h,
        dv=dv,
        do=do,
        dh=dh,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
        history_cfg_alpha=history_cfg_alpha,
        null_state=null_state,
    )
    dk2, dv, db, dg2 = prepare_wy_repr_bwd(
        k=k,
        v=v,
        beta=beta,
        g=g,
        A=A,
        dw=dw,
        du=dv,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
    )
    dk.add_(dk2)
    dg.add_(dg2)
    dg = chunk_local_cumsum(dg, chunk_size=chunk_size, reverse=True, cu_seqlens=cu_seqlens)
    return dq, dk, dv, db, dg, dh0, dnull


class ChunkGatedDeltaRuleFunction(torch.autograd.Function):

    @staticmethod
    @input_guard
    @autocast_custom_fwd
    def forward(
        ctx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        g: torch.Tensor,
        beta: torch.Tensor,
        scale: float,
        initial_state: torch.Tensor,
        output_final_state: bool,
        cu_seqlens: torch.LongTensor | None = None,
        use_qk_l2norm_in_kernel: bool = False,
        chunk_size: int = 64,
        history_cfg_alpha: torch.Tensor | None = None,
        null_state: torch.Tensor | None = None,
    ):
        q_rstd, k_rstd = None, None
        if use_qk_l2norm_in_kernel:
            q, q_rstd = l2norm_fwd(q)
            k, k_rstd = l2norm_fwd(k)

        g, o, A, final_state, initial_state = chunk_gated_delta_rule_fwd(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            scale=scale,
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
            chunk_size=chunk_size,
            history_cfg_alpha=history_cfg_alpha,
            null_state=null_state,
        )
        # save_for_backward only accepts Tensor. Keep optional metadata on ctx attrs.
        q_rstd_safe = q_rstd if q_rstd is not None else q.new_empty(0)
        k_rstd_safe = k_rstd if k_rstd is not None else k.new_empty(0)
        hdm_safe = history_cfg_alpha if history_cfg_alpha is not None else q.new_empty(0)
        ns_safe = null_state if null_state is not None else q.new_empty(0)
        ctx.save_for_backward(q, q_rstd_safe, k, k_rstd_safe, v, g, beta, A, initial_state, hdm_safe, ns_safe)
        ctx.cu_seqlens = cu_seqlens
        ctx.has_q_rstd = q_rstd is not None
        ctx.has_k_rstd = k_rstd is not None
        ctx.has_history_cfg_alpha = history_cfg_alpha is not None and null_state is not None
        ctx.scale = scale
        ctx.use_qk_l2norm_in_kernel = use_qk_l2norm_in_kernel
        ctx.chunk_size = chunk_size
        return o.to(q.dtype), final_state

    @staticmethod
    @input_guard
    @autocast_custom_bwd
    def backward(
        ctx,
        do: torch.Tensor,
        dht: torch.Tensor,
    ):
        q, q_rstd_saved, k, k_rstd_saved, v, g, beta, A, initial_state, hdm_saved, ns_saved = ctx.saved_tensors
        q_rstd = q_rstd_saved if ctx.has_q_rstd else None
        k_rstd = k_rstd_saved if ctx.has_k_rstd else None
        history_cfg_alpha = hdm_saved if ctx.has_history_cfg_alpha else None
        null_state = ns_saved if ctx.has_history_cfg_alpha else None
        cu_seqlens = ctx.cu_seqlens
        dq, dk, dv, db, dg, dh0, dnull = chunk_gated_delta_rule_bwd(
            q=q,
            k=k,
            v=v,
            g=g,
            beta=beta,
            A=A,
            scale=ctx.scale,
            initial_state=initial_state,
            do=do,
            dht=dht,
            cu_seqlens=cu_seqlens,
            chunk_size=ctx.chunk_size,
            history_cfg_alpha=history_cfg_alpha,
            null_state=null_state,
        )
        if ctx.use_qk_l2norm_in_kernel:
            dq = l2norm_bwd(q, q_rstd, dq)
            dk = l2norm_bwd(k, k_rstd, dk)
        return dq.to(q), dk.to(k), dv.to(v), dg.to(g), db.to(beta), None, dh0, None, None, None, None, None, dnull


@torch.compiler.disable
def chunk_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: float = None,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
    history_cfg_alpha: torch.Tensor | None = None,
    null_state: torch.Tensor | None = None,
    **kwargs,
):
    r"""
    Args:
        q (torch.Tensor):
            queries of shape `[B, T, H, K]`.
        k (torch.Tensor):
            keys of shape `[B, T, H, K]`.
        v (torch.Tensor):
            values of shape `[B, T, H, V]`.
        g (torch.Tensor):
            (forget) gating tensor (in log space!) of shape `[B, T, H]`.
        beta (torch.Tensor):
            betas of shape `[B, T, H]`.
        scale (Optional[float]):
            Scale factor for the RetNet attention scores.
            If not provided, it will default to `1 / sqrt(K)`. Default: `None`.
        initial_state (Optional[torch.Tensor]):
            Initial state of shape `[N, H, K, V]` for `N` input sequences.
            For equal-length input sequences, `N` equals the batch size `B`.
            Default: `None`.
        output_final_state (Optional[bool]):
            Whether to output the final state of shape `[N, H, K, V]`. Default: `False`.
        use_qk_l2norm_in_kernel (bool):
            Whether to apply L2norm to the q/k tensor internally. Default: `False`.
        cu_seqlens (torch.LongTensor):
            Cumulative sequence lengths of shape `[N+1]` used for variable-length training,
            consistent with the FlashAttention API.
        chunk_size (int):
            Chunk/block size used by forward and backward kernels. Default: `64`.
    Returns:
        o (torch.Tensor):
            Outputs of shape `[B, T, H, V]`.
        final_state (torch.Tensor):
            Final state of shape `[N, H, K, V]` if `output_final_state=True` else `None`.

    Examples::
        >>> import torch
        >>> import torch.nn.functional as F
        >>> from einops import rearrange
        >>> from fla.ops.gated_delta_rule import chunk_gated_delta_rule
        # inputs with equal lengths
        >>> B, T, H, K, V = 4, 2048, 4, 512, 512
        >>> q = torch.randn(B, T, H, K, dtype=torch.bfloat16, device='cuda')
        >>> k = F.normalize(torch.randn(B, T, H, K, dtype=torch.bfloat16, device='cuda'), p=2, dim=-1)
        >>> v = torch.randn(B, T, H, V, dtype=torch.bfloat16, device='cuda')
        >>> beta = torch.rand(B, T, H, dtype=torch.bfloat16, device='cuda').sigmoid()
        >>> g = F.logsigmoid(torch.rand(B, T, H, dtype=torch.bfloat16, device='cuda'))
        >>> h0 = torch.randn(B, H, K, V, dtype=torch.bfloat16, device='cuda')
        >>> o, ht = chunk_gated_delta_rule(
            q, k, v, g, beta,
            initial_state=h0,
            output_final_state=True
        )
        # for variable-length inputs, the batch size `B` is expected to be 1 and `cu_seqlens` is required
        >>> q, k, v, beta, g = map(lambda x: rearrange(x, 'b t ... -> 1 (b t) ...'), (q, k, v, beta, g))
        # for a batch with 4 sequences, `cu_seqlens` with 5 start/end positions are expected
        >>> cu_seqlens = q.new_tensor([0, 2048, 4096, 6144, 8192], dtype=torch.long)
        >>> o, ht = chunk_gated_delta_rule(
            q, k, v, g, beta,
            initial_state=h0,
            output_final_state=True,
            cu_seqlens=cu_seqlens
        )
    """
    if 'head_first' in kwargs:
        warnings.warn(
            "head_first is deprecated and will be removed in a future version. "
            "Please use head_first=False for now instead.",
        )

    if cu_seqlens is not None:
        if q.shape[0] != 1:
            raise ValueError(
                f"The batch size is expected to be 1 rather than {q.shape[0]} when using `cu_seqlens`."
                f"Please flatten variable-length inputs before processing.",
            )
        if initial_state is not None and initial_state.shape[0] != len(cu_seqlens) - 1:
            raise ValueError(
                f"The number of initial states is expected to be equal to the number of input sequences, "
                f"i.e., {len(cu_seqlens) - 1} rather than {initial_state.shape[0]}.",
            )
    if scale is None:
        scale = k.shape[-1] ** -0.5
    if chunk_size not in (16, 32, 64):
        raise ValueError(
            f"`chunk_size` must be one of [16, 32, 64], but got {chunk_size}."
        )
    o, final_state = ChunkGatedDeltaRuleFunction.apply(
        q,
        k,
        v,
        g,
        beta,
        scale,
        initial_state,
        output_final_state,
        cu_seqlens,
        use_qk_l2norm_in_kernel,
        chunk_size,
        history_cfg_alpha,
        null_state,
    )
    return o, final_state


# ---------------------------------------------------------------------------
# Prefill-only: state update without output computation
# ---------------------------------------------------------------------------


def chunk_gated_delta_rule_prefill_fwd(
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
) -> torch.Tensor:
    """
    Chunk-level prefill forward: computes WY representation and propagates
    hidden state through all chunks, but does NOT compute the output ``o``.

    This is equivalent to ``chunk_gated_delta_rule_fwd`` with the
    ``chunk_fwd_o_block_causal`` step removed.

    Returns:
        final_state (torch.Tensor): ``[N, H, K, V]`` hidden state after the
            last token.
    """
    g = chunk_local_cumsum(g, chunk_size=chunk_size, cu_seqlens=cu_seqlens)

    # obtain WY representation. u is actually the new v.
    A = chunk_scaled_dot_kkt_fwd(
        k=k,
        g=g,
        beta=beta,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
        output_dtype=torch.float32,
    )
    A = solve_tril(
        A=A,
        cu_seqlens=cu_seqlens,
        output_dtype=k.dtype,
    )
    w, u = recompute_w_u_fwd(
        k=k,
        v=v,
        beta=beta,
        A=A,
        g=g,
        cu_seqlens=cu_seqlens,
    )

    _h, _v_new, final_state = chunk_gated_delta_rule_fwd_h(
        k=k,
        w=w,
        u=u,
        g=g,
        initial_state=initial_state,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
        save_new_value=False,
    )

    return final_state


@torch.compiler.disable
def chunk_gated_delta_rule_prefill(
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor = None,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
) -> torch.Tensor:
    r"""
    Chunk-level prefill-only entry point: propagates hidden state through the
    input sequence using chunk-parallelised WY + state update kernels, without
    computing query-state products (no output ``o``).

    Args:
        k (torch.Tensor):
            Keys of shape ``[B, T, H, K]``.
        v (torch.Tensor):
            Values of shape ``[B, T, H, V]``.
        g (torch.Tensor):
            (forget) gating tensor (in log space!) of shape ``[B, T, H]``.
        beta (torch.Tensor):
            Betas of shape ``[B, T, H]``.
        initial_state (Optional[torch.Tensor]):
            Initial state of shape ``[N, H, K, V]`` for ``N`` input sequences.
            For equal-length input sequences, ``N`` equals the batch size ``B``.
            Default: ``None``.
        use_qk_l2norm_in_kernel (bool):
            Whether to L2-normalise ``k`` inside this function.
            Default: ``False``.
        cu_seqlens (torch.LongTensor):
            Cumulative sequence lengths of shape ``[N+1]`` used for
            variable-length training, consistent with the FlashAttention API.
        chunk_size (int):
            Chunk/block size used by the forward kernel. Default: ``64``.

    Returns:
        final_state (torch.Tensor):
            Final hidden state of shape ``[N, H, K, V]``.
    """
    if cu_seqlens is not None:
        if k.shape[0] != 1:
            raise ValueError(
                f"The batch size is expected to be 1 rather than {k.shape[0]} when using `cu_seqlens`."
                f"Please flatten variable-length inputs before processing.",
            )
        if initial_state is not None and initial_state.shape[0] != len(cu_seqlens) - 1:
            raise ValueError(
                f"The number of initial states is expected to be equal to the number of input sequences, "
                f"i.e., {len(cu_seqlens) - 1} rather than {initial_state.shape[0]}.",
            )
    if chunk_size not in (16, 32, 64):
        raise ValueError(
            f"`chunk_size` must be one of [16, 32, 64], but got {chunk_size}."
        )

    if use_qk_l2norm_in_kernel:
        k, _k_rstd = l2norm_fwd(k)

    final_state = chunk_gated_delta_rule_prefill_fwd(
        k=k,
        v=v,
        g=g,
        beta=beta,
        initial_state=initial_state,
        cu_seqlens=cu_seqlens,
        chunk_size=chunk_size,
    )
    return final_state


def main():
    """
    Forward-only numeric alignment tests.

    Keep only necessary checks:
    1) local block-causal output/state vs block_causal_fast_chunk reference.
    2) final-state equivalence: FLA token-causal vs local block-causal.

    Input dtypes: fp32 / fp16 / bf16.
    """
    import torch.nn.functional as F
    from diffusion.model.nets.causal.ops.naive import block_causal_fast_chunk
    from fla.ops.gated_delta_rule.chunk import chunk_gated_delta_rule as fla_chunk_gated_delta_rule
    
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")

    print("=" * 80)
    print("Block-Causal Gated DeltaNet Forward Validation Tests")
    print("=" * 80)
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Check for CUDA
    if not torch.cuda.is_available():
        print("CUDA not available, skipping tests")
        return
    
    device = torch.device('cuda')

    # Test control: choose from {"all", "forward", "backward"}
    TEST_TARGET = "backward"
    RUN_FORWARD_TESTS = TEST_TARGET in ("all", "forward")
    RUN_BACKWARD_TESTS = TEST_TARGET in ("all", "backward")
    print(f"Test target: {TEST_TARGET} (forward={RUN_FORWARD_TESTS}, backward={RUN_BACKWARD_TESTS})")
    
    # Test dimensions
    B = 2
    T = 256
    H = 4
    K = 64
    V = 64
    block_size = 64
    
    # Use scale=1.0 for easier comparison
    scale = 1.0
    
    print(f"\nTest Configuration:")
    print(f"  B={B}, T={T}, H={H}, K={K}, V={V}, block_size={block_size}")
    print(f"  scale={scale}")
    print()

    dtype_configs = [
        # (input_dtype, tol_output, tol_state_token_vs_block, tol_state_vs_ref)
        (torch.float32, 1e-3, 1e-3, 1e-3),
        (torch.float16, 8e-2, 8e-2, 8e-2),
        (torch.bfloat16, 1.2e-1, 1.2e-1, 1.2e-1),
    ]

    norm_test_modes = [
        {
            "name": "kernel_l2norm",
            "external_qk_l2norm": False,
            "ref_qk_norm": True,
            "kernel_qk_l2norm": True,
        },
        {
            "name": "external_qk_l2norm_kernel_disabled",
            "external_qk_l2norm": True,
            "ref_qk_norm": False,
            "kernel_qk_l2norm": False,
        },
    ]
    
    with torch.no_grad():
        # Shared base tensors in fp32, then cast to each input dtype.
        q_base = torch.randn(B, T, H, K, dtype=torch.float32, device=device)
        k_base = torch.randn(B, T, H, K, dtype=torch.float32, device=device)
        v_base = torch.randn(B, T, H, V, dtype=torch.float32, device=device)
        g_base = F.logsigmoid(torch.randn(B, T, H, dtype=torch.float32, device=device))
        beta_base = torch.sigmoid(torch.randn(B, T, H, dtype=torch.float32, device=device))
        initial_state_base = torch.randn(B, H, K, V, dtype=torch.float32, device=device)

        overall_pass = True

        for mode in (norm_test_modes if RUN_FORWARD_TESTS else []):
            mode_name = mode["name"]
            external_qk_l2norm = mode["external_qk_l2norm"]
            ref_qk_norm = mode["ref_qk_norm"]
            kernel_qk_l2norm = mode["kernel_qk_l2norm"]

            print("#" * 60)
            print(f"Norm Test Mode: {mode_name}")
            print(
                f"  external_qk_l2norm={external_qk_l2norm}, "
                f"ref_qk_norm={ref_qk_norm}, "
                f"kernel_qk_l2norm={kernel_qk_l2norm}"
            )
            print("#" * 60)

            mode_pass = True

            for in_dtype, tol_output, tol_state_token, tol_state_ref in dtype_configs:
                print("-" * 60)
                print(f"DType: {in_dtype}")
                print("-" * 60)

                q_in = q_base.to(in_dtype)
                k_in = k_base.to(in_dtype)
                v_in = v_base.to(in_dtype)
                g_in = g_base.to(in_dtype)
                beta_in = beta_base.to(in_dtype)
                h0_in = initial_state_base.to(in_dtype)

                if external_qk_l2norm:
                    # External Q/K L2 norm, then disable all in-kernel Q/K norm paths.
                    q_eval = F.normalize(q_in.float(), p=2, dim=-1).to(in_dtype)
                    k_eval = F.normalize(k_in.float(), p=2, dim=-1).to(in_dtype)
                else:
                    q_eval = q_in
                    k_eval = k_in

                # 1) Reference: block_causal_fast_chunk
                o_ref, s_ref = block_causal_fast_chunk(
                    q=q_eval,
                    k=k_eval,
                    v=v_in,
                    g=g_in,
                    beta=beta_in,
                    initial_state=h0_in.float(),
                    chunk_size=block_size,
                    output_final_state=True,
                    qk_norm=ref_qk_norm,
                )

                # 2) Baseline: original FLA token-causal
                _, s_token_causal = fla_chunk_gated_delta_rule(
                    q=q_eval,
                    k=k_eval,
                    v=v_in,
                    g=g_in,
                    beta=beta_in,
                    scale=scale,
                    initial_state=h0_in,
                    output_final_state=True,
                    use_qk_l2norm_in_kernel=kernel_qk_l2norm,
                )

                # 3) Local block-causal implementation
                o_block_causal, s_block_causal = chunk_gated_delta_rule(
                    q=q_eval,
                    k=k_eval,
                    v=v_in,
                    g=g_in,
                    beta=beta_in,
                    scale=scale,
                    initial_state=h0_in,
                    output_final_state=True,
                    use_qk_l2norm_in_kernel=kernel_qk_l2norm,
                )

                # Numeric alignment checks
                state_token_diff = (s_token_causal.float() - s_block_causal.float()).abs()
                state_token_max = state_token_diff.max().item()
                state_token_mean = state_token_diff.mean().item()

                output_diff = (o_block_causal.float() - o_ref.float()).abs()
                output_max = output_diff.max().item()
                output_mean = output_diff.mean().item()

                state_ref_diff = (s_block_causal.float() - s_ref.float()).abs()
                state_ref_max = state_ref_diff.max().item()
                state_ref_mean = state_ref_diff.mean().item()

                pass_state_token = state_token_max < tol_state_token
                pass_output = output_max < tol_output
                pass_state_ref = state_ref_max < tol_state_ref
                dtype_pass = pass_state_token and pass_output and pass_state_ref
                mode_pass = mode_pass and dtype_pass
                overall_pass = overall_pass and dtype_pass

                print(f"  [state token_vs_block] max={state_token_max:.6e} mean={state_token_mean:.6e} tol={tol_state_token:.3e} -> {'PASS' if pass_state_token else 'FAIL'}")
                print(f"  [output block_vs_ref]  max={output_max:.6e} mean={output_mean:.6e} tol={tol_output:.3e} -> {'PASS' if pass_output else 'FAIL'}")
                print(f"  [state block_vs_ref]   max={state_ref_max:.6e} mean={state_ref_mean:.6e} tol={tol_state_ref:.3e} -> {'PASS' if pass_state_ref else 'FAIL'}")
                print(f"  [dtype result] {'PASS' if dtype_pass else 'FAIL'}")
                print()

            print(f"[mode result] {mode_name}: {'PASS' if mode_pass else 'FAIL'}")
            print()

        if not RUN_FORWARD_TESTS:
            print("Forward tests are skipped by TEST_TARGET.")
            print()

        print("=" * 60)
        print("Forward-Only Summary")
        print("=" * 60)
        forward_status = 'SKIPPED' if not RUN_FORWARD_TESTS else ('ALL PASS' if overall_pass else 'HAS FAILURE')
        print(f"Forward Overall: {forward_status}")
        print()

        if RUN_FORWARD_TESTS:
            print("=" * 80)
            print("Forward Runtime Benchmark (Local Triton vs block_causal_fast_chunk)")
            print("=" * 80)
            import time

            bench_B, bench_H, bench_T, bench_K, bench_V = 1, 4, 256, 64, 64
            bench_dtype = torch.float32
            bench_scale = 1.0
            bench_iters = 50
            bench_warmup = 10

            q_b = torch.randn(bench_B, bench_T, bench_H, bench_K, dtype=bench_dtype, device=device)
            k_b = torch.randn(bench_B, bench_T, bench_H, bench_K, dtype=bench_dtype, device=device)
            v_b = torch.randn(bench_B, bench_T, bench_H, bench_V, dtype=bench_dtype, device=device)
            g_b = F.logsigmoid(torch.randn(bench_B, bench_T, bench_H, dtype=bench_dtype, device=device))
            beta_b = torch.sigmoid(torch.randn(bench_B, bench_T, bench_H, dtype=bench_dtype, device=device))
            h0_b = torch.randn(bench_B, bench_H, bench_K, bench_V, dtype=bench_dtype, device=device)

            def _time_forward(path: str, warmup: int, iters: int) -> float:
                torch.cuda.synchronize()
                for _ in range(warmup):
                    if path == "triton":
                        chunk_gated_delta_rule(
                            q=q_b,
                            k=k_b,
                            v=v_b,
                            g=g_b,
                            beta=beta_b,
                            scale=bench_scale,
                            initial_state=h0_b,
                            output_final_state=True,
                            use_qk_l2norm_in_kernel=True,
                        )
                    else:
                        block_causal_fast_chunk(
                            q=q_b,
                            k=k_b,
                            v=v_b,
                            g=g_b,
                            beta=beta_b,
                            initial_state=h0_b.float(),
                            chunk_size=64,
                            qk_norm=True,
                        )
                torch.cuda.synchronize()

                t0 = time.perf_counter()
                for _ in range(iters):
                    if path == "triton":
                        chunk_gated_delta_rule(
                            q=q_b,
                            k=k_b,
                            v=v_b,
                            g=g_b,
                            beta=beta_b,
                            scale=bench_scale,
                            initial_state=h0_b,
                            output_final_state=True,
                            use_qk_l2norm_in_kernel=True,
                        )
                    else:
                        block_causal_fast_chunk(
                            q=q_b,
                            k=k_b,
                            v=v_b,
                            g=g_b,
                            beta=beta_b,
                            initial_state=h0_b.float(),
                            chunk_size=64,
                            qk_norm=True,
                        )
                torch.cuda.synchronize()
                t1 = time.perf_counter()
                return (t1 - t0) * 1000.0 / iters

            triton_fwd_ms = _time_forward(path="triton", warmup=bench_warmup, iters=bench_iters)
            ref_fwd_ms = _time_forward(path="reference", warmup=bench_warmup, iters=bench_iters)
            speedup_fwd = ref_fwd_ms / triton_fwd_ms if triton_fwd_ms > 0 else float('inf')

            print(f"  Benchmark config: B={bench_B}, H={bench_H}, T={bench_T}, K={bench_K}, V={bench_V}, dtype={bench_dtype}")
            print(f"  Triton forward (ms/iter):    {triton_fwd_ms:.3f}")
            print(f"  Reference forward (ms/iter): {ref_fwd_ms:.3f}")
            print(f"  Speedup (Reference/Triton):  {speedup_fwd:.3f}x")
            print()

    if not RUN_BACKWARD_TESTS:
        print("=" * 60)
        print("Final Summary (Forward + Backward)")
        print("=" * 60)
        print(f"Forward:  {forward_status}")
        print("Backward: SKIPPED")
        print("Overall:  SKIPPED")
        print()
        return

    # ================================================================
    # Backward Smoke Test: Original FLA Kernel Only
    # ================================================================
    # print("=" * 80)
    # print("FLA Original Kernel Backward Smoke Test (No Numeric Comparison)")
    # print("=" * 80)
    # fla_backward_smoke_pass = True

    # B_fla, H_fla, T_fla, K_fla, V_fla = 1, 2, 128, 64, 32
    # scale_fla = K_fla ** -0.5
    # fla_dtypes = [torch.float32, torch.bfloat16]

    # for fla_dtype in fla_dtypes:
    #     print(f"  --- dtype={fla_dtype} ---")
    #     try:
    #         torch.manual_seed(321)
    #         q_fla = torch.randn(B_fla, T_fla, H_fla, K_fla, dtype=fla_dtype, device=device, requires_grad=True)
    #         k_fla = F.normalize(torch.randn(B_fla, T_fla, H_fla, K_fla, dtype=fla_dtype, device=device), p=2, dim=-1).detach().requires_grad_(True)
    #         v_fla = torch.randn(B_fla, T_fla, H_fla, V_fla, dtype=fla_dtype, device=device, requires_grad=True)
    #         g_fla = F.logsigmoid(torch.randn(B_fla, T_fla, H_fla, dtype=fla_dtype, device=device)).detach().requires_grad_(True)
    #         beta_fla = torch.sigmoid(torch.randn(B_fla, T_fla, H_fla, dtype=fla_dtype, device=device)).detach().requires_grad_(True)
    #         h0_fla = torch.randn(B_fla, H_fla, K_fla, V_fla, dtype=fla_dtype, device=device, requires_grad=True)

    #         o_fla, s_fla = fla_chunk_gated_delta_rule(
    #             q=q_fla,
    #             k=k_fla,
    #             v=v_fla,
    #             g=g_fla,
    #             beta=beta_fla,
    #             scale=scale_fla,
    #             initial_state=h0_fla,
    #             output_final_state=True,
    #             use_qk_l2norm_in_kernel=False,
    #         )
    #         do_fla = torch.randn_like(o_fla)
    #         dht_fla = torch.randn_like(s_fla)
    #         loss_fla = (o_fla * do_fla).sum() + (s_fla * dht_fla).sum()
    #         loss_fla.backward()

    #         print("    backward: PASS")
    #     except Exception as e:
    #         fla_backward_smoke_pass = False
    #         print(f"    backward: FAIL ({type(e).__name__}: {e})")
    #         traceback.print_exc()
    #     print()

    # print(f"[FLA backward smoke result] {'PASS' if fla_backward_smoke_pass else 'FAIL'}")
    # print()

    # ================================================================
    # Backward Validation Tests
    # ================================================================
    print("=" * 80)
    print("Block-Causal Gated DeltaNet Backward Validation Tests")
    print("=" * 80)

    backward_test_cases = [
        {
            "name": "case_small_norm",
            "B": 1,
            "H": 2,
            "T": 128,
            "K": 64,
            "V": 32,
            "BT": 64,
            "scale": 1.0,
            "kernel_qk_l2norm": True,
            "ref_qk_norm": True,
            "dtype_configs": [
                (torch.float32, 5e-2),
                (torch.bfloat16, 2e-1),
                (torch.float16, 8e-2),
            ],
        },
        {
            "name": "case_medium_norm",
            "B": 2,
            "H": 4,
            "T": 256,
            "K": 64,
            "V": 64,
            "BT": 64,
            "scale": 1.0,
            "kernel_qk_l2norm": True,
            "ref_qk_norm": True,
            "dtype_configs": [
                (torch.float32, 8e-2),
                (torch.bfloat16, 2.5e-1),
            ],
        },
        {
            "name": "case_small_no_norm",
            "B": 1,
            "H": 2,
            "T": 192,
            "K": 64,
            "V": 32,
            "BT": 64,
            "scale": 1.0,
            "kernel_qk_l2norm": False,
            "ref_qk_norm": False,
            "dtype_configs": [
                (torch.float32, 6e-2),
                (torch.bfloat16, 2e-1),
            ],
        },
    ]

    bwd_overall_pass = True
    torch.manual_seed(123)

    for case in backward_test_cases:
        B_bwd = case["B"]
        H_bwd = case["H"]
        T_bwd = case["T"]
        K_bwd = case["K"]
        V_bwd = case["V"]
        block_size_bwd = case["BT"]
        scale_bwd = case["scale"]
        kernel_qk_l2norm = case["kernel_qk_l2norm"]
        ref_qk_norm = case["ref_qk_norm"]

        print(f"\nBackward Test ({case['name']}):")
        print(f"  B={B_bwd}, H={H_bwd}, T={T_bwd}, K={K_bwd}, V={V_bwd}, BT={block_size_bwd}, qk_norm(kernel/ref)=({kernel_qk_l2norm}/{ref_qk_norm})")

        case_pass = True
        for bwd_dtype, bwd_tol in case["dtype_configs"]:
            print(f"  --- dtype={bwd_dtype}, tol={bwd_tol} ---")

            q_t = torch.randn(B_bwd, T_bwd, H_bwd, K_bwd, dtype=bwd_dtype, device=device, requires_grad=True)
            k_t = torch.randn(B_bwd, T_bwd, H_bwd, K_bwd, dtype=bwd_dtype, device=device, requires_grad=True)
            v_t = torch.randn(B_bwd, T_bwd, H_bwd, V_bwd, dtype=bwd_dtype, device=device, requires_grad=True)
            g_t = F.logsigmoid(torch.randn(B_bwd, T_bwd, H_bwd, dtype=bwd_dtype, device=device)).detach().requires_grad_(True)
            beta_t = torch.sigmoid(torch.randn(B_bwd, T_bwd, H_bwd, dtype=bwd_dtype, device=device)).detach().requires_grad_(True)
            h0_t = torch.randn(B_bwd, H_bwd, K_bwd, V_bwd, dtype=bwd_dtype, device=device, requires_grad=True)

            o_triton, s_triton = chunk_gated_delta_rule(
                q=q_t,
                k=k_t,
                v=v_t,
                g=g_t,
                beta=beta_t,
                scale=scale_bwd,
                initial_state=h0_t,
                output_final_state=True,
                use_qk_l2norm_in_kernel=kernel_qk_l2norm,
            )
            do = torch.randn_like(o_triton)
            dht = torch.randn_like(s_triton)
            loss_triton = (o_triton * do).sum() + (s_triton * dht).sum()
            loss_triton.backward()

            dq_triton = q_t.grad.clone()
            dk_triton = k_t.grad.clone()
            dv_triton = v_t.grad.clone()
            dg_triton = g_t.grad.clone()
            db_triton = beta_t.grad.clone()
            dh0_triton = h0_t.grad.clone()

            q_r = q_t.detach().clone().requires_grad_(True)
            k_r = k_t.detach().clone().requires_grad_(True)
            v_r = v_t.detach().clone().requires_grad_(True)
            g_r = g_t.detach().clone().requires_grad_(True)
            beta_r = beta_t.detach().clone().requires_grad_(True)
            h0_r = h0_t.detach().clone().float().requires_grad_(True)

            o_ref, s_ref = block_causal_fast_chunk(
                q=q_r,
                k=k_r,
                v=v_r,
                g=g_r,
                beta=beta_r,
                initial_state=h0_r,
                chunk_size=block_size_bwd,
                output_final_state=True,
                qk_norm=ref_qk_norm,
            )
            loss_ref = (o_ref.to(bwd_dtype) * do).sum() + (s_ref.to(dht.dtype) * dht).sum()
            loss_ref.backward()

            dq_ref = q_r.grad.clone()
            dk_ref = k_r.grad.clone()
            dv_ref = v_r.grad.clone()
            dg_ref = g_r.grad.clone()
            db_ref = beta_r.grad.clone()
            dh0_ref = h0_r.grad.clone()

            grad_names = ['dq', 'dk', 'dv', 'dg', 'db', 'dh0']
            grad_triton_list = [dq_triton, dk_triton, dv_triton, dg_triton, db_triton, dh0_triton]
            grad_ref_list = [dq_ref, dk_ref, dv_ref, dg_ref, db_ref, dh0_ref]

            dtype_bwd_pass = True
            for name, gt, gr in zip(grad_names, grad_triton_list, grad_ref_list):
                diff = (gt.float() - gr.float()).abs()
                max_diff = diff.max().item()
                mean_diff = diff.mean().item()
                passed = max_diff < bwd_tol
                dtype_bwd_pass = dtype_bwd_pass and passed
                print(f"    [{name}] max={max_diff:.6e} mean={mean_diff:.6e} tol={bwd_tol:.3e} -> {'PASS' if passed else 'FAIL'}")

            case_pass = case_pass and dtype_bwd_pass
            print(f"    [dtype result] {'PASS' if dtype_bwd_pass else 'FAIL'}")
            print()

        bwd_overall_pass = bwd_overall_pass and case_pass
        print(f"  [case result] {case['name']}: {'PASS' if case_pass else 'FAIL'}")
        print()

    # Test: Verify no local do->dv shortcut via output path
    print("Backward Extra Test: no chunk-internal do->dv shortcut (sanity)")
    torch.manual_seed(456)
    q2 = torch.randn(1, 32, 2, 64, dtype=torch.float32, device=device, requires_grad=True)
    k2 = torch.randn(1, 32, 2, 64, dtype=torch.float32, device=device, requires_grad=True)
    v2 = torch.randn(1, 32, 2, 32, dtype=torch.float32, device=device, requires_grad=True)
    g2 = F.logsigmoid(torch.randn(1, 32, 2, dtype=torch.float32, device=device)).detach().requires_grad_(True)
    beta2 = torch.sigmoid(torch.randn(1, 32, 2, dtype=torch.float32, device=device)).detach().requires_grad_(True)
    h02 = torch.randn(1, 2, 64, 32, dtype=torch.float32, device=device, requires_grad=True)
    o2a, s2a = chunk_gated_delta_rule(
        q=q2,
        k=k2,
        v=v2,
        g=g2,
        beta=beta2,
        scale=1,
        initial_state=h02,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
    )
    do2a = torch.randn_like(o2a)
    loss2a = (o2a * do2a).sum()
    loss2a.backward()
    print(f"  dq norm with do: {q2.grad.norm().item():.6e}")
    print()

    # ================================================================
    # Runtime Benchmark (forward + backward end-to-end)
    # ================================================================
    print("=" * 80)
    print("Backward Runtime Benchmark (Triton vs block_causal_fast_chunk)")
    print("=" * 80)

    bench_B, bench_H, bench_T, bench_K, bench_V = 1, 4, 256, 64, 64
    bench_dtype = torch.float32
    bench_scale = 1.0
    bench_iters = 20
    bench_warmup = 5

    q_base = torch.randn(bench_B, bench_T, bench_H, bench_K, dtype=bench_dtype, device=device)
    k_base = torch.randn(bench_B, bench_T, bench_H, bench_K, dtype=bench_dtype, device=device)
    v_base = torch.randn(bench_B, bench_T, bench_H, bench_V, dtype=bench_dtype, device=device)
    g_base = F.logsigmoid(torch.randn(bench_B, bench_T, bench_H, dtype=bench_dtype, device=device))
    beta_base = torch.sigmoid(torch.randn(bench_B, bench_T, bench_H, dtype=bench_dtype, device=device))
    h0_base = torch.randn(bench_B, bench_H, bench_K, bench_V, dtype=bench_dtype, device=device)

    def _time_backward(path: str, warmup: int, iters: int) -> float:
        torch.cuda.synchronize()
        for _ in range(warmup):
            q = q_base.detach().clone().requires_grad_(True)
            k = k_base.detach().clone().requires_grad_(True)
            v = v_base.detach().clone().requires_grad_(True)
            g = g_base.detach().clone().requires_grad_(True)
            beta = beta_base.detach().clone().requires_grad_(True)
            h0 = h0_base.detach().clone().requires_grad_(True)
            if path == "triton":
                o, s = chunk_gated_delta_rule(
                    q=q,
                    k=k,
                    v=v,
                    g=g,
                    beta=beta,
                    scale=bench_scale,
                    initial_state=h0,
                    output_final_state=True,
                    use_qk_l2norm_in_kernel=True,
                )
            else:
                o, s = block_causal_fast_chunk(
                    q=q,
                    k=k,
                    v=v,
                    g=g,
                    beta=beta,
                    initial_state=h0.float(),
                    chunk_size=64,
                    output_final_state=True,
                    qk_norm=True,
                )
            do = torch.randn_like(o)
            dht = torch.randn_like(s)
            loss = (o * do).sum() + (s.to(dht.dtype) * dht).sum()
            loss.backward()
        torch.cuda.synchronize()

        import time
        t0 = time.perf_counter()
        for _ in range(iters):
            q = q_base.detach().clone().requires_grad_(True)
            k = k_base.detach().clone().requires_grad_(True)
            v = v_base.detach().clone().requires_grad_(True)
            g = g_base.detach().clone().requires_grad_(True)
            beta = beta_base.detach().clone().requires_grad_(True)
            h0 = h0_base.detach().clone().requires_grad_(True)
            if path == "triton":
                o, s = chunk_gated_delta_rule(
                    q=q,
                    k=k,
                    v=v,
                    g=g,
                    beta=beta,
                    scale=bench_scale,
                    initial_state=h0,
                    output_final_state=True,
                    use_qk_l2norm_in_kernel=True,
                )
            else:
                o, s = block_causal_fast_chunk(
                    q=q,
                    k=k,
                    v=v,
                    g=g,
                    beta=beta,
                    initial_state=h0.float(),
                    chunk_size=64,
                    output_final_state=True,
                    qk_norm=True,
                )
            do = torch.randn_like(o)
            dht = torch.randn_like(s)
            loss = (o * do).sum() + (s.to(dht.dtype) * dht).sum()
            loss.backward()
        torch.cuda.synchronize()
        t1 = time.perf_counter()
        return (t1 - t0) * 1000.0 / iters

    triton_ms = _time_backward(path="triton", warmup=bench_warmup, iters=bench_iters)
    ref_ms = _time_backward(path="reference", warmup=bench_warmup, iters=bench_iters)
    speedup = ref_ms / triton_ms if triton_ms > 0 else float('inf')
    print(f"  Benchmark config: B={bench_B}, H={bench_H}, T={bench_T}, K={bench_K}, V={bench_V}, dtype={bench_dtype}")
    print(f"  Triton backward (ms/iter):   {triton_ms:.3f}")
    print(f"  Reference backward (ms/iter):{ref_ms:.3f}")
    print(f"  Speedup (Reference/Triton):  {speedup:.3f}x")

    print()
    print("=" * 60)
    print("Final Summary (Forward + Backward)")
    print("=" * 60)
    print(f"Forward:  {'SKIPPED' if not RUN_FORWARD_TESTS else ('ALL PASS' if overall_pass else 'HAS FAILURE')}")
    # print(f"FLA Bwd: {'ALL PASS' if fla_backward_smoke_pass else 'HAS FAILURE'}")
    print(f"Backward: {'ALL PASS' if bwd_overall_pass else 'HAS FAILURE'}")
    # all_pass = overall_pass and fla_backward_smoke_pass and bwd_overall_pass
    all_pass = overall_pass and bwd_overall_pass
    print(f"Overall:  {'ALL PASS' if all_pass else 'HAS FAILURE'}")
    print()


if __name__ == "__main__":
    main()
