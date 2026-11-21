import torch
import torch.nn as nn
from typing import List, Tuple, Union
import numpy as np

# Try to import liger kernel for optimized RoPE
try:
    from liger_kernel.ops import LigerRopeFunction
    LIGER_ROPE_AVAILABLE = True
except ImportError:
    LIGER_ROPE_AVAILABLE = False

class RopePosEmbed(nn.Module):
    # modified from https://github.com/black-forest-labs/flux/blob/c00d7c60b085fce8058b9df845e036090873f2ce/src/flux/modules/layers.py#L11
    def __init__(self, theta: int, axes_dim: List[int]):
        super().__init__()
        self.theta = theta
        self.axes_dim = axes_dim

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        n_axes = ids.shape[-1]
        cos_out = []
        sin_out = []
        pos = ids.float()
        is_mps = ids.device.type == "mps"
        is_npu = ids.device.type == "npu"
        freqs_dtype = torch.float32 if (is_mps or is_npu) else torch.float64
        for i in range(n_axes):
            cos, sin = get_1d_rotary_pos_embed(
                self.axes_dim[i],
                pos[:, i],
                theta=self.theta,
                repeat_interleave_real=True,
                use_real=True,
                freqs_dtype=freqs_dtype,
            )
            cos_out.append(cos)
            sin_out.append(sin)
        freqs_cos = torch.cat(cos_out, dim=-1).to(ids.device)
        freqs_sin = torch.cat(sin_out, dim=-1).to(ids.device)
        return freqs_cos, freqs_sin
    
    @staticmethod
    def _prepare_latent_sequence_ids(batch_size, seq_len, device, dtype):
        seq_ids = torch.arange(seq_len, device=device, dtype=dtype)
        return seq_ids.unsqueeze(-1)


def get_1d_rotary_pos_embed(
    dim: int,
    pos: Union[np.ndarray, int],
    theta: float = 10000.0,
    use_real=False,
    linear_factor=1.0,
    ntk_factor=1.0,
    repeat_interleave_real=True,
    freqs_dtype=torch.float32,  #  torch.float32, torch.float64 (flux)
):
    """
    Precompute the frequency tensor for complex exponentials (cis) with given dimensions.

    This function calculates a frequency tensor with complex exponentials using the given dimension 'dim' and the end
    index 'end'. The 'theta' parameter scales the frequencies. The returned tensor contains complex values in complex64
    data type.

    Args:
        dim (`int`): Dimension of the frequency tensor.
        pos (`np.ndarray` or `int`): Position indices for the frequency tensor. [S] or scalar
        theta (`float`, *optional*, defaults to 10000.0):
            Scaling factor for frequency computation. Defaults to 10000.0.
        use_real (`bool`, *optional*):
            If True, return real part and imaginary part separately. Otherwise, return complex numbers.
        linear_factor (`float`, *optional*, defaults to 1.0):
            Scaling factor for the context extrapolation. Defaults to 1.0.
        ntk_factor (`float`, *optional*, defaults to 1.0):
            Scaling factor for the NTK-Aware RoPE. Defaults to 1.0.
        repeat_interleave_real (`bool`, *optional*, defaults to `True`):
            If `True` and `use_real`, real part and imaginary part are each interleaved with themselves to reach `dim`.
            Otherwise, they are concateanted with themselves.
        freqs_dtype (`torch.float32` or `torch.float64`, *optional*, defaults to `torch.float32`):
            the dtype of the frequency tensor.
    Returns:
        `torch.Tensor`: Precomputed frequency tensor with complex exponentials. [S, D/2]
    """
    assert dim % 2 == 0

    if isinstance(pos, int):
        pos = torch.arange(pos)
    if isinstance(pos, np.ndarray):
        pos = torch.from_numpy(pos)  # type: ignore  # [S]

    theta = theta * ntk_factor
    freqs = (
        1.0
        / (theta ** (torch.arange(0, dim, 2, dtype=freqs_dtype, device=pos.device)[: (dim // 2)] / dim))
        / linear_factor
    )  # [D/2]
    freqs = torch.outer(pos, freqs)  # type: ignore   # [S, D/2]
    if use_real and repeat_interleave_real:
        # flux, hunyuan-dit, cogvideox
        freqs_cos = freqs.cos().repeat_interleave(2, dim=1).float()  # [S, D]
        freqs_sin = freqs.sin().repeat_interleave(2, dim=1).float()  # [S, D]
        return freqs_cos, freqs_sin
    elif use_real:
        # stable audio, allegro
        freqs_cos = torch.cat([freqs.cos(), freqs.cos()], dim=-1).float()  # [S, D]
        freqs_sin = torch.cat([freqs.sin(), freqs.sin()], dim=-1).float()  # [S, D]
        return freqs_cos, freqs_sin
    else:
        # lumina
        freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64     # [S, D/2]
        return freqs_cis


def apply_rotary_emb(
    x: torch.Tensor,
    freqs_cis: Union[torch.Tensor, Tuple[torch.Tensor]],
    use_real: bool = True,
    use_real_unbind_dim: int = -1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor. This function applies rotary embeddings
    to the given query or key 'x' tensors using the provided frequency tensor 'freqs_cis'. The input tensors are
    reshaped as complex numbers, and the frequency tensor is reshaped for broadcasting compatibility. The resulting
    tensors contain rotary embeddings and are returned as real tensors.

    Args:
        x (`torch.Tensor`):
            Query or key tensor to apply rotary embeddings. [B, S, H, D] xk (torch.Tensor): Key tensor to apply
        freqs_cis (`Tuple[torch.Tensor]`): Precomputed frequency tensor for complex exponentials. ([S, D], [S, D],)

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.
    """
    seq_len = x.shape[1] # B,S,H,D -> S    
    if use_real:
        new_freqs_cis = []
        for i in range(len(freqs_cis)):
            new_freqs_cis.append(freqs_cis[i][:seq_len])
        freqs_cis = tuple(new_freqs_cis)
        cos, sin = freqs_cis  # [S, D]        
        cos, sin = cos.to(x.device), sin.to(x.device)

        if use_real_unbind_dim == -1:
            # Used for flux, cogvideox, hunyuan-dit
            cos = cos.unsqueeze(1) # S, D-> S, 1, D
            sin = sin.unsqueeze(1) # S, D-> S, 1, D
            x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)  # [B, S, H, D//2, 2]
            x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(3) # [B, S, H, D]
        elif use_real_unbind_dim == -2:
            # Used for Sana
            cos = cos.transpose(-1, -2) # S, D-> D, S
            sin = sin.transpose(-1, -2) # S, D-> D, S
            x_real, x_imag = x.reshape(*x.shape[:-2], -1, 2, x.shape[-1]).unbind(-2)  # [B, H, D//2, 2, S]
            x_rotated = torch.stack([-x_imag, x_real], dim=-2).flatten(2, 3) # [B, H, D, S]
        else:
            raise ValueError(f"`use_real_unbind_dim={use_real_unbind_dim}` but should be -1 or -2.")

        out = (x.float() * cos + x_rotated.float() * sin).to(x.dtype)

        return out
    else:
        freqs_cis = freqs_cis[:seq_len]
        # used for lumina
        x_rotated = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2)) # B,S,H,D -> B,S,H,D/2,2
        freqs_cis = freqs_cis.unsqueeze(1) # S,D/2 -> S,1,D/2
        x_out = torch.view_as_real(x_rotated * freqs_cis).flatten(3)

        return x_out.type_as(x)


def apply_causal_rotary_emb(
    x: torch.Tensor,
    freqs_cis: Union[torch.Tensor, Tuple[torch.Tensor]],
    use_real: bool = True,
    use_real_unbind_dim: int = -1,
    start_frame: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to input tensors using the given frequency tensor. This function applies rotary embeddings
    to the given query or key 'x' tensors using the provided frequency tensor 'freqs_cis'. The input tensors are
    reshaped as complex numbers, and the frequency tensor is reshaped for broadcasting compatibility. The resulting
    tensors contain rotary embeddings and are returned as real tensors.

    Args:
        x (`torch.Tensor`):
            Query or key tensor to apply rotary embeddings. [B, H, S, D] xk (torch.Tensor): Key tensor to apply
        freqs_cis (`Tuple[torch.Tensor]`): Precomputed frequency tensor for complex exponentials. ([S, D], [S, D],)

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Tuple of modified query tensor and key tensor with rotary embeddings.
    """
    seq_len = x.shape[1]
    # start_frame = 0 # for debug
    if use_real:
        new_freqs_cis = []
        for i in range(len(freqs_cis)):
            new_freqs_cis.append(freqs_cis[i][start_frame:start_frame + seq_len])
        cos, sin = new_freqs_cis  # [S, D]        
        cos, sin = cos.to(x.device), sin.to(x.device)

        if use_real_unbind_dim == -1:
            # Used for flux, cogvideox, hunyuan-dit
            cos = cos.unsqueeze(1) # S, D-> S, 1, D
            sin = sin.unsqueeze(1) # S, D-> S, 1, D
            x_real, x_imag = x.reshape(*x.shape[:-1], -1, 2).unbind(-1)  # [B, S, H, D//2, 2]
            x_rotated = torch.stack([-x_imag, x_real], dim=-1).flatten(3) # [B, S, H, D]
        elif use_real_unbind_dim == -2:
            # Used for Sana
            cos = cos.transpose(-1, -2) # S, D-> D, S
            sin = sin.transpose(-1, -2) # S, D-> D, S
            x_real, x_imag = x.reshape(*x.shape[:-2], -1, 2, x.shape[-1]).unbind(-2)  # [B, H, D//2, 2, S]
            x_rotated = torch.stack([-x_imag, x_real], dim=-2).flatten(2, 3) # [B, H, D, S]
        else:
            raise ValueError(f"`use_real_unbind_dim={use_real_unbind_dim}` but should be -1 or -2.")

        out = (x.float() * cos + x_rotated.float() * sin).to(x.dtype)

        return out
    else:
        new_freqs_cis = freqs_cis[start_frame:start_frame + seq_len]
        # used for lumina
        x_rotated = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2)) # B,S,H,D -> B,S,H,D/2,2
        new_freqs_cis = new_freqs_cis.unsqueeze(1) # S,D/2 -> S,1,D/2
        x_out = torch.view_as_real(x_rotated * new_freqs_cis).flatten(3)

        return x_out.type_as(x)


def _apply_rotary_emb_liger(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings using liger kernel.

    Liger kernel uses half-split RoPE (pairs first half with second half of head_dim),
    while our implementation uses interleaved RoPE (pairs odd/even positions).
    We need to reorder the input before and after to get equivalent results.

    Args:
        q: Query tensor of shape [B, S, H, D] or [B, H, S, D]
        k: Key tensor of shape [B, S, H, D] or [B, H, S, D]
        cos: Cosine tensor of shape [S, D] (interleaved)
        sin: Sine tensor of shape [S, D] (interleaved)

    Returns:
        Tuple of rotated q and k tensors with same shape as input.
    """
    head_dim = q.shape[-1]
    seq_len = q.shape[1]

    # Reorder from interleaved to half-split: [q0,q1,q2,...] -> [q0,q2,q4,...,q1,q3,q5,...]
    q_even = q[..., 0::2]
    q_odd = q[..., 1::2]
    q_reordered = torch.cat([q_even, q_odd], dim=-1)

    k_even = k[..., 0::2]
    k_odd = k[..., 1::2]
    k_reordered = torch.cat([k_even, k_odd], dim=-1)

    # Convert cos/sin from interleaved [c0,c0,c1,c1,...] to half [c0,c1,c2,...]
    # Take every other element starting from 0
    cos_half = cos[:seq_len, 0::2].unsqueeze(0)  # [1, S, D/2]
    sin_half = sin[:seq_len, 0::2].unsqueeze(0)  # [1, S, D/2]

    # Transpose q, k from [B, S, H, D] to [B, H, S, D] for liger kernel
    q_transposed = q_reordered.transpose(1, 2)
    k_transposed = k_reordered.transpose(1, 2)

    # Apply liger RoPE
    q_rotated, k_rotated = LigerRopeFunction.apply(
        q_transposed, k_transposed, cos_half, sin_half, None, 1
    )

    # Transpose back from [B, H, S, D] to [B, S, H, D]
    q_rotated = q_rotated.transpose(1, 2)
    k_rotated = k_rotated.transpose(1, 2)

    # Reorder back from half-split to interleaved
    q_first = q_rotated[..., :head_dim//2]
    q_second = q_rotated[..., head_dim//2:]
    q_out = torch.stack([q_first, q_second], dim=-1).flatten(-2)

    k_first = k_rotated[..., :head_dim//2]
    k_second = k_rotated[..., head_dim//2:]
    k_out = torch.stack([k_first, k_second], dim=-1).flatten(-2)

    return q_out, k_out


def apply_rotary_emb_qk(
    q: torch.Tensor,
    k: torch.Tensor,
    freqs_cis: Union[torch.Tensor, Tuple[torch.Tensor]],
    use_real: bool = True,
    use_real_unbind_dim: int = -1,
    use_liger: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embeddings to query and key tensors.

    This function can use liger kernel for acceleration when available and applicable.
    Liger kernel is only used when:
    - use_real=True
    - use_real_unbind_dim=-1 (flux/cogvideox/hunyuan-dit style)
    - LIGER_ROPE_AVAILABLE is True
    - use_liger=True

    Args:
        q: Query tensor of shape [B, S, H, D]
        k: Key tensor of shape [B, S, H, D]
        freqs_cis: Precomputed frequency tensor (cos, sin) each of shape [S, D]
        use_real: Whether to use real-valued cos/sin (vs complex)
        use_real_unbind_dim: Dimension for unbinding (-1 for flux style, -2 for Sana style)
        use_liger: Whether to try using liger kernel

    Returns:
        Tuple of rotated q and k tensors.
    """
    seq_len = q.shape[1]

    if use_real:
        new_freqs_cis = []
        for i in range(len(freqs_cis)):
            new_freqs_cis.append(freqs_cis[i][:seq_len])
        cos, sin = new_freqs_cis
        cos, sin = cos.to(q.device), sin.to(q.device)

        # Try to use liger kernel for flux/cogvideox/hunyuan-dit style
        if (use_liger and LIGER_ROPE_AVAILABLE and use_real_unbind_dim == -1
            and q.is_cuda and q.dtype in [torch.float32, torch.float16, torch.bfloat16]):
            return _apply_rotary_emb_liger(q, k, cos, sin)

        # Fallback to original implementation
        if use_real_unbind_dim == -1:
            cos = cos.unsqueeze(1)
            sin = sin.unsqueeze(1)
            q_real, q_imag = q.reshape(*q.shape[:-1], -1, 2).unbind(-1)
            q_rotated = torch.stack([-q_imag, q_real], dim=-1).flatten(3)
            k_real, k_imag = k.reshape(*k.shape[:-1], -1, 2).unbind(-1)
            k_rotated = torch.stack([-k_imag, k_real], dim=-1).flatten(3)
        elif use_real_unbind_dim == -2:
            cos = cos.transpose(-1, -2)
            sin = sin.transpose(-1, -2)
            q_real, q_imag = q.reshape(*q.shape[:-2], -1, 2, q.shape[-1]).unbind(-2)
            q_rotated = torch.stack([-q_imag, q_real], dim=-2).flatten(2, 3)
            k_real, k_imag = k.reshape(*k.shape[:-2], -1, 2, k.shape[-1]).unbind(-2)
            k_rotated = torch.stack([-k_imag, k_real], dim=-2).flatten(2, 3)
        else:
            raise ValueError(f"`use_real_unbind_dim={use_real_unbind_dim}` but should be -1 or -2.")

        q_out = (q.float() * cos + q_rotated.float() * sin).to(q.dtype)
        k_out = (k.float() * cos + k_rotated.float() * sin).to(k.dtype)
        return q_out, k_out
    else:
        freqs_cis = freqs_cis[:seq_len]
        q_rotated = torch.view_as_complex(q.float().reshape(*q.shape[:-1], -1, 2))
        k_rotated = torch.view_as_complex(k.float().reshape(*k.shape[:-1], -1, 2))
        freqs_cis = freqs_cis.unsqueeze(1)
        q_out = torch.view_as_real(q_rotated * freqs_cis).flatten(3)
        k_out = torch.view_as_real(k_rotated * freqs_cis).flatten(3)
        return q_out.type_as(q), k_out.type_as(k)


def apply_causal_rotary_emb_qk(
    q: torch.Tensor,
    k: torch.Tensor,
    freqs_cis: Union[torch.Tensor, Tuple[torch.Tensor]],
    use_real: bool = True,
    use_real_unbind_dim: int = -1,
    start_frame: int = 0,
    use_liger: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply causal rotary embeddings to query and key tensors.

    This function can use liger kernel for acceleration when available and applicable.

    Args:
        q: Query tensor of shape [B, S, H, D]
        k: Key tensor of shape [B, S, H, D]
        freqs_cis: Precomputed frequency tensor (cos, sin) each of shape [max_S, D]
        use_real: Whether to use real-valued cos/sin (vs complex)
        use_real_unbind_dim: Dimension for unbinding (-1 for flux style, -2 for Sana style)
        start_frame: Starting frame index for causal generation
        use_liger: Whether to try using liger kernel

    Returns:
        Tuple of rotated q and k tensors.
    """
    seq_len = q.shape[1]
    # start_frame = 0 # for debug
    if use_real:
        new_freqs_cis = []
        for i in range(len(freqs_cis)):
            new_freqs_cis.append(freqs_cis[i][start_frame:start_frame + seq_len])
        cos, sin = new_freqs_cis
        cos, sin = cos.to(q.device), sin.to(q.device)

        # Try to use liger kernel for flux/cogvideox/hunyuan-dit style
        if (use_liger and LIGER_ROPE_AVAILABLE and use_real_unbind_dim == -1
            and q.is_cuda and q.dtype in [torch.float32, torch.float16, torch.bfloat16]):
            return _apply_rotary_emb_liger(q, k, cos, sin)

        # Fallback to original implementation
        if use_real_unbind_dim == -1:
            cos = cos.unsqueeze(1)
            sin = sin.unsqueeze(1)
            q_real, q_imag = q.reshape(*q.shape[:-1], -1, 2).unbind(-1)
            q_rotated = torch.stack([-q_imag, q_real], dim=-1).flatten(3)
            k_real, k_imag = k.reshape(*k.shape[:-1], -1, 2).unbind(-1)
            k_rotated = torch.stack([-k_imag, k_real], dim=-1).flatten(3)
        elif use_real_unbind_dim == -2:
            cos = cos.transpose(-1, -2)
            sin = sin.transpose(-1, -2)
            q_real, q_imag = q.reshape(*q.shape[:-2], -1, 2, q.shape[-1]).unbind(-2)
            q_rotated = torch.stack([-q_imag, q_real], dim=-2).flatten(2, 3)
            k_real, k_imag = k.reshape(*k.shape[:-2], -1, 2, k.shape[-1]).unbind(-2)
            k_rotated = torch.stack([-k_imag, k_real], dim=-2).flatten(2, 3)
        else:
            raise ValueError(f"`use_real_unbind_dim={use_real_unbind_dim}` but should be -1 or -2.")

        q_out = (q.float() * cos + q_rotated.float() * sin).to(q.dtype)
        k_out = (k.float() * cos + k_rotated.float() * sin).to(k.dtype)
        return q_out, k_out
    else:
        new_freqs_cis = freqs_cis[start_frame:start_frame + seq_len]
        q_rotated = torch.view_as_complex(q.float().reshape(*q.shape[:-1], -1, 2))
        k_rotated = torch.view_as_complex(k.float().reshape(*k.shape[:-1], -1, 2))
        new_freqs_cis = new_freqs_cis.unsqueeze(1)
        q_out = torch.view_as_real(q_rotated * new_freqs_cis).flatten(3)
        k_out = torch.view_as_real(k_rotated * new_freqs_cis).flatten(3)
        return q_out.type_as(q), k_out.type_as(k)


if __name__ == "__main__":
    """
    Test cases for liger kernel RoPE implementation vs original implementation.
    Run with: python diffusion/model/nets/positional_embeding.py
    """
    import time

    print("=" * 80)
    print("Testing Liger Kernel RoPE Implementation")
    print("=" * 80)
    print(f"\nLIGER_ROPE_AVAILABLE: {LIGER_ROPE_AVAILABLE}")

    if not torch.cuda.is_available():
        print("CUDA not available, skipping tests")
        exit(0)

    # Test parameters
    bsz, seq_len, n_head, head_dim = 2, 64, 8, 64
    max_seq_len = 256  # For causal test

    print(f"\nTest configuration:")
    print(f"  batch_size={bsz}, seq_len={seq_len}, n_head={n_head}, head_dim={head_dim}")

    # Create test tensors
    torch.manual_seed(42)
    q = torch.randn(bsz, seq_len, n_head, head_dim, device='cuda', dtype=torch.float32)
    k = torch.randn(bsz, seq_len, n_head, head_dim, device='cuda', dtype=torch.float32)

    # Get frequency tensors (longer for causal test)
    freqs_cos, freqs_sin = get_1d_rotary_pos_embed(
        head_dim, max_seq_len, theta=10000.0,
        use_real=True, repeat_interleave_real=True
    )
    freqs_cis = (freqs_cos.cuda(), freqs_sin.cuda())

    # Test 1: apply_rotary_emb vs apply_rotary_emb_qk (with liger)
    print("\n" + "=" * 80)
    print("Test 1: apply_rotary_emb_qk correctness")
    print("=" * 80)

    q_orig = apply_rotary_emb(q, freqs_cis, use_real=True, use_real_unbind_dim=-1)
    k_orig = apply_rotary_emb(k, freqs_cis, use_real=True, use_real_unbind_dim=-1)

    q_liger, k_liger = apply_rotary_emb_qk(q, k, freqs_cis, use_real=True, use_real_unbind_dim=-1, use_liger=True)
    q_no_liger, k_no_liger = apply_rotary_emb_qk(q, k, freqs_cis, use_real=True, use_real_unbind_dim=-1, use_liger=False)

    q_orig_vs_liger_diff = torch.abs(q_orig - q_liger).max().item()
    k_orig_vs_liger_diff = torch.abs(k_orig - k_liger).max().item()
    q_orig_vs_no_liger_diff = torch.abs(q_orig - q_no_liger).max().item()
    k_orig_vs_no_liger_diff = torch.abs(k_orig - k_no_liger).max().item()

    print(f"\nQ original vs liger:")
    print(f"  max_diff={q_orig_vs_liger_diff:.2e}")
    print(f"  allclose(atol=1e-5)={torch.allclose(q_orig, q_liger, atol=1e-5)}")

    print(f"\nK original vs liger:")
    print(f"  max_diff={k_orig_vs_liger_diff:.2e}")
    print(f"  allclose(atol=1e-5)={torch.allclose(k_orig, k_liger, atol=1e-5)}")

    print(f"\nQ original vs no_liger:")
    print(f"  max_diff={q_orig_vs_no_liger_diff:.2e}")
    print(f"  allclose(atol=1e-5)={torch.allclose(q_orig, q_no_liger, atol=1e-5)}")

    print(f"\nK original vs no_liger:")
    print(f"  max_diff={k_orig_vs_no_liger_diff:.2e}")
    print(f"  allclose(atol=1e-5)={torch.allclose(k_orig, k_no_liger, atol=1e-5)}")

    # Test 2: Causal version
    print("\n" + "=" * 80)
    print("Test 2: apply_causal_rotary_emb_qk correctness")
    print("=" * 80)

    start_frame = 10
    q_causal_orig = apply_causal_rotary_emb(q, freqs_cis, use_real=True, use_real_unbind_dim=-1, start_frame=start_frame)
    k_causal_orig = apply_causal_rotary_emb(k, freqs_cis, use_real=True, use_real_unbind_dim=-1, start_frame=start_frame)

    q_causal_liger, k_causal_liger = apply_causal_rotary_emb_qk(q, k, freqs_cis, use_real=True, use_real_unbind_dim=-1, start_frame=start_frame, use_liger=True)
    q_causal_no_liger, k_causal_no_liger = apply_causal_rotary_emb_qk(q, k, freqs_cis, use_real=True, use_real_unbind_dim=-1, start_frame=start_frame, use_liger=False)

    q_causal_orig_vs_liger_diff = torch.abs(q_causal_orig - q_causal_liger).max().item()
    k_causal_orig_vs_liger_diff = torch.abs(k_causal_orig - k_causal_liger).max().item()
    q_causal_orig_vs_no_liger_diff = torch.abs(q_causal_orig - q_causal_no_liger).max().item()
    k_causal_orig_vs_no_liger_diff = torch.abs(k_causal_orig - k_causal_no_liger).max().item()

    print(f"\nQ causal original vs liger:")
    print(f"  max_diff={q_causal_orig_vs_liger_diff:.2e}")
    print(f"  allclose(atol=1e-5)={torch.allclose(q_causal_orig, q_causal_liger, atol=1e-5)}")

    print(f"\nK causal original vs liger:")
    print(f"  max_diff={k_causal_orig_vs_liger_diff:.2e}")
    print(f"  allclose(atol=1e-5)={torch.allclose(k_causal_orig, k_causal_liger, atol=1e-5)}")

    print(f"\nQ causal original vs no_liger:")
    print(f"  max_diff={q_causal_orig_vs_no_liger_diff:.2e}")
    print(f"  allclose(atol=1e-5)={torch.allclose(q_causal_orig, q_causal_no_liger, atol=1e-5)}")

    print(f"\nK causal original vs no_liger:")
    print(f"  max_diff={k_causal_orig_vs_no_liger_diff:.2e}")
    print(f"  allclose(atol=1e-5)={torch.allclose(k_causal_orig, k_causal_no_liger, atol=1e-5)}")

    # Test 3: Different dtypes
    print("\n" + "=" * 80)
    print("Test 3: Different dtypes")
    print("=" * 80)

    for dtype in [torch.float16, torch.bfloat16]:
        q_dtype = q.to(dtype)
        k_dtype = k.to(dtype)
        q_out, k_out = apply_rotary_emb_qk(q_dtype, k_dtype, freqs_cis, use_real=True, use_real_unbind_dim=-1, use_liger=True)
        print(f"\n{dtype}:")
        print(f"  Q shape={q_out.shape}, dtype={q_out.dtype}")
        print(f"  K shape={k_out.shape}, dtype={k_out.dtype}")

    # Test 4: Performance comparison
    print("\n" + "=" * 80)
    print("Test 4: Performance comparison")
    print("=" * 80)

    # Larger tensors for performance testing
    bsz_perf, seq_len_perf, n_head_perf, head_dim_perf = 16, 256, 16, 64

    q_perf = torch.randn(bsz_perf, seq_len_perf, n_head_perf, head_dim_perf, device='cuda', dtype=torch.bfloat16)
    k_perf = torch.randn(bsz_perf, seq_len_perf, n_head_perf, head_dim_perf, device='cuda', dtype=torch.bfloat16)

    freqs_cos_perf, freqs_sin_perf = get_1d_rotary_pos_embed(
        head_dim_perf, seq_len_perf, theta=10000.0,
        use_real=True, repeat_interleave_real=True
    )
    freqs_cis_perf = (freqs_cos_perf.cuda(), freqs_sin_perf.cuda())

    print(f"\nPerformance test configuration:")
    print(f"  batch_size={bsz_perf}, seq_len={seq_len_perf}, n_head={n_head_perf}, head_dim={head_dim_perf}")

    # Warmup
    for _ in range(10):
        _ = apply_rotary_emb(q_perf, freqs_cis_perf, use_real=True, use_real_unbind_dim=-1)
        _ = apply_rotary_emb_qk(q_perf, k_perf, freqs_cis_perf, use_real=True, use_real_unbind_dim=-1, use_liger=True)
        _ = apply_rotary_emb_qk(q_perf, k_perf, freqs_cis_perf, use_real=True, use_real_unbind_dim=-1, use_liger=False)

    torch.cuda.synchronize()

    # Benchmark original (2x apply_rotary_emb)
    n_iter = 100
    start = time.time()
    for _ in range(n_iter):
        q_out = apply_rotary_emb(q_perf, freqs_cis_perf, use_real=True, use_real_unbind_dim=-1)
        k_out = apply_rotary_emb(k_perf, freqs_cis_perf, use_real=True, use_real_unbind_dim=-1)
    torch.cuda.synchronize()
    orig_time = (time.time() - start) / n_iter * 1000

    # Benchmark liger
    start = time.time()
    for _ in range(n_iter):
        q_out, k_out = apply_rotary_emb_qk(q_perf, k_perf, freqs_cis_perf, use_real=True, use_real_unbind_dim=-1, use_liger=True)
    torch.cuda.synchronize()
    liger_time = (time.time() - start) / n_iter * 1000

    # Benchmark no liger
    start = time.time()
    for _ in range(n_iter):
        q_out, k_out = apply_rotary_emb_qk(q_perf, k_perf, freqs_cis_perf, use_real=True, use_real_unbind_dim=-1, use_liger=False)
    torch.cuda.synchronize()
    no_liger_time = (time.time() - start) / n_iter * 1000

    print(f"\nTiming results (averaged over {n_iter} iterations):")
    print(f"  Original (2x apply_rotary_emb): {orig_time:.3f} ms")
    print(f"  Liger kernel:                   {liger_time:.3f} ms")
    print(f"  No liger (apply_rotary_emb_qk): {no_liger_time:.3f} ms")
    print(f"\nSpeedup:")
    print(f"  Liger vs original:  {orig_time/liger_time:.2f}x")
    print(f"  Liger vs no_liger:  {no_liger_time/liger_time:.2f}x")

    # Summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)

    all_tests_passed = (
        torch.allclose(q_orig, q_liger, atol=1e-5) and
        torch.allclose(k_orig, k_liger, atol=1e-5) and
        torch.allclose(q_orig, q_no_liger, atol=1e-5) and
        torch.allclose(k_orig, k_no_liger, atol=1e-5) and
        torch.allclose(q_causal_orig, q_causal_liger, atol=1e-5) and
        torch.allclose(k_causal_orig, k_causal_liger, atol=1e-5) and
        torch.allclose(q_causal_orig, q_causal_no_liger, atol=1e-5) and
        torch.allclose(k_causal_orig, k_causal_no_liger, atol=1e-5)
    )

    if all_tests_passed:
        print("\nAll tests PASSED")
        print(f"Liger kernel provides {orig_time/liger_time:.2f}x speedup over original implementation")
    else:
        print("\nSome tests FAILED")
        print("Please check the differences above")

    print("=" * 80)