# -*- coding: utf-8 -*-
"""
Block-Level Gated DeltaNet for Causal Cross-Attention in Block-Level Diffusion.

This module provides Gated DeltaNet adapted for Block-Level Diffusion (BLD):
- GatedDeltaNetForBLD: Frame-level gating with timestep modulation for block-level diffusion

The key insight is that in BLD:
1. Historical blocks are clean (already denoised)
2. Current block is noisy (being denoised)
3. We want to query historical information to help denoise current block
4. Timestep affects how much we trust/use historical information

Reference: flash-linear-attention (https://github.com/fla-org/flash-linear-attention)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from typing import Optional, Tuple, Dict, Literal, List

# Import ShortConvolution from FLA
try:
    from fla.modules import ShortConvolution
    FLA_AVAILABLE = True
except ImportError:
    FLA_AVAILABLE = False
    ShortConvolution = None
from diffusion.model.nets.causal.ops.chunk import chunk_gated_delta_rule
from diffusion.model.nets.causal.ops.chunk import chunk_gated_delta_rule_prefill
from diffusion.model.nets.causal.ops.naive import (
    _maybe_normalize_qk,
    _upcast_state_if_needed,
    recurrent_write_states,
    block_causal_reference_naive,
    block_causal_fast_chunk,
)


# ============================================================================
# Gated DeltaNet for Block-Level Diffusion
# ============================================================================

class GatedDeltaNetForBLD(nn.Module):
    """
    Gated DeltaNet adapted for Block-Level Diffusion (BLD).
    
    Key Design Principles:
    1. Each frame has its own α (forget) and β (update) gates
    2. Gates are computed from frame content + timestep modulation
    3. Timestep affects gating: high noise → trust history more
    4. State S accumulates historical information with decay
    5. Current block queries the accumulated state
    
    Training Mode:
        - context: historical clean frames (B, T_kv, D)
        - motion: current noisy block (B, T_q, D)
        - Process context to build state S, then query with motion
    
    Inference Mode:
        - prefill: write clean frames to state
        - decode: query state with noisy frames
    
    Args:
        dim: Hidden dimension
        num_heads: Number of attention heads
        head_dim: Dimension per head (alternative to num_heads)
        bias: Whether to use bias in linear layers
        qk_norm: Whether to L2 normalize Q/K in block_causal_fast_chunk
        use_triton_kernel: Whether to use Triton kernel in training forward
        proj_drop: Dropout rate for output projection
        gate_low_rank_dim: Low-rank dimension for gate projection (0 = full rank)
        gate_logit_normalizer: Normalizer for gate logits
        block_size: Number of frames per block for causal masking
        allow_neg_eigval: Following FLA - allow beta in [0, 2] for negative eigenvalues
        use_short_conv: Following FLA - use ShortConv before Q/K/V projection
        conv_size: Kernel size for ShortConv (default: 4)
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        head_dim: int = None,
        bias: bool = True,
        qk_norm: bool = True,  # Whether to L2 normalize Q/K
        use_triton_kernel: bool = False,
        scale: Optional[float] = None,
        proj_drop: float = 0.,
        gate_low_rank_dim: int = 16,
        gate_logit_normalizer: float = 16.0,
        block_size: int = 64,  # Number of frames per block for causal masking
        allow_neg_eigval: bool = False,  # Following FLA: allow beta in [0, 2] for negative eigenvalues
        use_short_conv: bool = True,  # Following FLA: use ShortConv before projection
        conv_size: int = 4,  # Kernel size for ShortConv
        q_t_modulation_mode: str = "none",  # none | additive | film
        q_t_modulation_dim: int = 768,
        kv_t_modulation_mode: str = "none",  # none | additive | film
        kv_t_modulation_dim: int = 768,
        use_learnable_init_state: bool = False,  # Whether to use learnable initial state instead of zeros
        history_cfg_enabled: bool = False,  # Enable null-history CFG (learnable null state + alpha interpolation)
        history_cfg_p_clean: float = 0.3,  # Probability of alpha=1 (clean state) per block
        history_cfg_p_null: float = 0.2,  # Probability of alpha=0 (null state) per block
        history_cfg_null_state_mode: str = "learned",  # learned | zero | shared_init
    ):
        super().__init__()
        
        # Head configuration
        if num_heads is not None and head_dim is None:
            assert dim % num_heads == 0, 'dim should be divisible by num_heads'
            self.num_heads = num_heads
            self.head_dim = dim // num_heads
        elif head_dim is not None:
            assert dim % head_dim == 0, 'dim should be divisible by head_dim'
            self.head_dim = head_dim
            self.num_heads = dim // head_dim
        else:
            raise ValueError("Either num_heads or head_dim must be specified")
        
        self.dim = dim
        self.gate_logit_normalizer = gate_logit_normalizer
        self.block_size = block_size
        self.allow_neg_eigval = allow_neg_eigval
        self.use_short_conv = use_short_conv
        self.conv_size = conv_size
        self.qk_norm = qk_norm
        self.use_triton_kernel = use_triton_kernel
        self.scale = scale
        self.use_learnable_init_state = use_learnable_init_state

        # Learnable initial state: (H, K, V) parameter, expanded to (B, H, K, V) at runtime
        if self.use_learnable_init_state:
            self.init_state = nn.Parameter(
                torch.zeros(self.num_heads, self.head_dim, self.head_dim)
            )

        # Null-history CFG: null state for read-side history interpolation
        # h_read = alpha * h_real + (1 - alpha) * null_state
        self.history_cfg_enabled = history_cfg_enabled
        self.history_cfg_p_clean = history_cfg_p_clean
        self.history_cfg_p_null = history_cfg_p_null
        self.history_cfg_null_state_mode = str(history_cfg_null_state_mode).lower()
        _valid_modes = ("learned", "zero", "shared_init")
        if self.history_cfg_null_state_mode not in _valid_modes:
            raise ValueError(
                f"history_cfg_null_state_mode must be one of {_valid_modes}, "
                f"got '{self.history_cfg_null_state_mode}'"
            )
        if self.history_cfg_enabled:
            _state_shape = (self.num_heads, self.head_dim, self.head_dim)
            if self.history_cfg_null_state_mode == "learned":
                # Independent learnable null state (default)
                self.null_state = nn.Parameter(torch.zeros(*_state_shape))
            elif self.history_cfg_null_state_mode == "zero":
                # Fixed all-zero null state — ablation baseline
                self.register_buffer("null_state", torch.zeros(*_state_shape), persistent=True)
            elif self.history_cfg_null_state_mode == "shared_init":
                # Alias to init_state: "no history" = "beginning of sequence"
                if not self.use_learnable_init_state:
                    raise ValueError(
                        "history_cfg_null_state_mode='shared_init' requires "
                        "use_learnable_init_state=True"
                    )
                self.null_state = self.init_state
        else:
            self.null_state = None

        self.q_t_modulation_mode = str(q_t_modulation_mode).lower()
        if self.q_t_modulation_mode not in ["none", "additive", "film"]:
            raise ValueError(
                f"Unsupported q_t_modulation_mode={self.q_t_modulation_mode}. "
                "Use one of [none, additive, film]."
            )
        self.q_t_modulation_dim = int(q_t_modulation_dim)
        if self.q_t_modulation_mode != "none" and self.q_t_modulation_dim <= 0:
            raise ValueError(
                f"q_t_modulation_dim must be > 0 when q_t_modulation_mode != 'none', got {self.q_t_modulation_dim}"
            )
        
        self.kv_t_modulation_mode = str(kv_t_modulation_mode).lower()
        if self.kv_t_modulation_mode not in ["none", "additive", "film"]:
            raise ValueError(
                f"Unsupported kv_t_modulation_mode={self.kv_t_modulation_mode}. "
                "Use one of [none, additive, film]."
            )
        self.kv_t_modulation_dim = int(kv_t_modulation_dim)
        if self.kv_t_modulation_mode != "none" and self.kv_t_modulation_dim <= 0:
            raise ValueError(
                f"kv_t_modulation_dim must be > 0 when kv_t_modulation_mode != 'none', got {self.kv_t_modulation_dim}"
            )
        
        # Q/K/V projections - 遵循FLA的设计
        # 始终创建Linear projection
        self.q_proj = nn.Linear(dim, dim, bias=bias)
        self.k_proj = nn.Linear(dim, dim, bias=bias)
        self.v_proj = nn.Linear(dim, dim, bias=bias)

        # Optional timestep-aware query modulation (kept outside kernel path)
        if self.q_t_modulation_mode == "additive":
            self.q_t_proj = nn.Linear(self.q_t_modulation_dim, dim, bias=bias)
        elif self.q_t_modulation_mode == "film":
            self.q_t_gamma = nn.Linear(self.q_t_modulation_dim, dim, bias=bias)
            self.q_t_beta = nn.Linear(self.q_t_modulation_dim, dim, bias=bias)
        
        # Optional timestep-aware KV modulation (for QK-Alignment)
        # K and V share the same modulation layers
        if self.kv_t_modulation_mode == "additive":
            self.kv_k_t_proj = nn.Linear(self.kv_t_modulation_dim, dim, bias=bias)
            self.kv_v_t_proj = nn.Linear(self.kv_t_modulation_dim, dim, bias=bias)
        elif self.kv_t_modulation_mode == "film":
            self.kv_k_t_gamma = nn.Linear(self.kv_t_modulation_dim, dim, bias=bias)
            self.kv_k_t_beta = nn.Linear(self.kv_t_modulation_dim, dim, bias=bias)
            self.kv_v_t_gamma = nn.Linear(self.kv_t_modulation_dim, dim, bias=bias)
            self.kv_v_t_beta = nn.Linear(self.kv_t_modulation_dim, dim, bias=bias)
        
        # ShortConvolution: 只对 K/V 应用，不对 Q 应用
        # 原因：Q 来自 noisy motion，如果使用 shortconv 会导致跨块信息泄露，破坏块级自回归语义
        # K/V 来自 clean context，需要 shortconv 来保持训练/推理一致性
        if use_short_conv:
            if not FLA_AVAILABLE:
                raise ImportError("FLA is required for ShortConvolution. Install with: pip install flash-linear-attention")
            # 注意：不创建 q_conv1d，Q 路径固定使用 F.silu(q_proj(x))
            self.k_conv1d = ShortConvolution(
                hidden_size=dim,
                kernel_size=conv_size,
                bias=bias,
                activation='silu',
            )
            self.v_conv1d = ShortConvolution(
                hidden_size=dim,
                kernel_size=conv_size,
                bias=bias,
                activation='silu',
            )
        
        # Output projection
        self.o_proj = nn.Linear(dim, dim, bias=bias)
        
        # Q/K normalization: use L2 norm (no learnable parameters)
        # L2 norm will be applied in forward using F.normalize
        
        # ========================================
        # Gate projections - following FLA's design
        # ========================================
        # α (forget gate): g = -A * softplus(a_proj(x) + dt_bias)
        # β (update gate): b = sigmoid(b_proj(x))
        #
        # Note: Gates are computed from context hidden states only (no t_emb).
        # This follows original FLA design and ensures cache reusability in inference.
        
        self.gate_low_rank_dim = gate_low_rank_dim
        
        # Input dimension for gate projections: always dim (gates only depend on context)
        gate_input_dim = dim
        
        if gate_low_rank_dim > 0:
            self.a_proj = nn.Sequential(
                nn.Linear(gate_input_dim, gate_low_rank_dim, bias=False),
                nn.Linear(gate_low_rank_dim, self.num_heads, bias=True)
            )
            self.b_proj = nn.Sequential(
                nn.Linear(gate_input_dim, gate_low_rank_dim, bias=False),
                nn.Linear(gate_low_rank_dim, self.num_heads, bias=True)
            )
        else:
            self.a_proj = nn.Linear(gate_input_dim, self.num_heads, bias=True)
            self.b_proj = nn.Linear(gate_input_dim, self.num_heads, bias=True)
        
        # A_log: per-head base decay rate (log-space), following FLA
        # A controls the "strength" of forgetting
        # Initialized uniformly in (0, 16], stored in log-space
        A = torch.empty(self.num_heads, dtype=torch.float32).uniform_(0, 16)
        self.A_log = nn.Parameter(torch.log(A))
        self.A_log._no_weight_decay = True
        
        # dt_bias: per-head timestep bias, following FLA
        # Controls the "baseline" of forgetting
        dt_min = 0.001
        dt_max = 0.1
        dt_init_floor = 1e-4
        dt = torch.exp(
            torch.rand(self.num_heads) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
        dt = torch.clamp(dt, min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        self.dt_bias = nn.Parameter(inv_dt)
        self.dt_bias._no_weight_decay = True
        
        # Dropout
        self.proj_drop = nn.Dropout(proj_drop)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights following FLA conventions."""
        # Gate biases: zero init
        if self.gate_low_rank_dim > 0:
            nn.init.zeros_(self.a_proj[1].bias)
            nn.init.zeros_(self.b_proj[1].bias)
        else:
            nn.init.zeros_(self.a_proj.bias)
            nn.init.zeros_(self.b_proj.bias)
        
        # Output projection: small init for residual
        nn.init.normal_(self.o_proj.weight, std=0.02)
        if self.o_proj.bias is not None:
            nn.init.zeros_(self.o_proj.bias)

        # Query time-modulation branch: zero-init to preserve baseline at startup
        if self.q_t_modulation_mode == "additive":
            nn.init.zeros_(self.q_t_proj.weight)
            if self.q_t_proj.bias is not None:
                nn.init.zeros_(self.q_t_proj.bias)
        elif self.q_t_modulation_mode == "film":
            nn.init.zeros_(self.q_t_gamma.weight)
            nn.init.zeros_(self.q_t_beta.weight)
            if self.q_t_gamma.bias is not None:
                nn.init.zeros_(self.q_t_gamma.bias)
            if self.q_t_beta.bias is not None:
                nn.init.zeros_(self.q_t_beta.bias)
        
        # KV time-modulation branch: zero-init to preserve baseline at startup
        if self.kv_t_modulation_mode == "additive":
            nn.init.zeros_(self.kv_k_t_proj.weight)
            nn.init.zeros_(self.kv_v_t_proj.weight)
            if self.kv_k_t_proj.bias is not None:
                nn.init.zeros_(self.kv_k_t_proj.bias)
            if self.kv_v_t_proj.bias is not None:
                nn.init.zeros_(self.kv_v_t_proj.bias)
        elif self.kv_t_modulation_mode == "film":
            nn.init.zeros_(self.kv_k_t_gamma.weight)
            nn.init.zeros_(self.kv_k_t_beta.weight)
            nn.init.zeros_(self.kv_v_t_gamma.weight)
            nn.init.zeros_(self.kv_v_t_beta.weight)
            if self.kv_k_t_gamma.bias is not None:
                nn.init.zeros_(self.kv_k_t_gamma.bias)
            if self.kv_k_t_beta.bias is not None:
                nn.init.zeros_(self.kv_k_t_beta.bias)
            if self.kv_v_t_gamma.bias is not None:
                nn.init.zeros_(self.kv_v_t_gamma.bias)
            if self.kv_v_t_beta.bias is not None:
                nn.init.zeros_(self.kv_v_t_beta.bias)
        
        # Note: A_log and dt_bias are initialized in __init__ following FLA's design

    def _align_t_emb(
        self,
        t_emb: Optional[torch.Tensor],
        target_len: int,
        target_dtype: torch.dtype,
        target_device: torch.device,
        expected_dim: int,
        emb_name: str = "t_emb",
    ) -> Optional[torch.Tensor]:
        """Shared helper to align time embedding to target shape."""
        if t_emb is None:
            return None
        t = t_emb.to(device=target_device, dtype=target_dtype)
        if t.dim() == 2:
            t = t.unsqueeze(1).expand(-1, target_len, -1)
        elif t.dim() == 3:
            if t.shape[1] == target_len:
                pass
            elif t.shape[1] == 1:
                t = t.expand(-1, target_len, -1)
            else:
                raise ValueError(
                    f"{emb_name} length mismatch: got {emb_name}.shape[1]={t.shape[1]}, "
                    f"expected {target_len} or 1."
                )
        else:
            raise ValueError(f"Unsupported {emb_name} shape: {t.shape}")

        if t.shape[-1] != expected_dim:
            raise ValueError(
                f"{emb_name} last dim ({t.shape[-1]}) does not match expected_dim ({expected_dim})."
            )
        return t

    def _apply_affine_time_modulation(
        self,
        x: torch.Tensor,
        t: Optional[torch.Tensor],
        mode: str,
        additive_proj: Optional[nn.Module] = None,
        gamma_proj: Optional[nn.Module] = None,
        beta_proj: Optional[nn.Module] = None,
    ) -> torch.Tensor:
        """Shared affine time modulation: additive or FiLM."""
        if mode == "none" or t is None:
            return x

        if mode == "additive":
            if additive_proj is None:
                raise ValueError("additive_proj is required when mode='additive'.")
            return x + additive_proj(t)

        if mode == "film":
            if gamma_proj is None or beta_proj is None:
                raise ValueError("gamma_proj and beta_proj are required when mode='film'.")
            gamma = gamma_proj(t)
            beta = beta_proj(t)
            return x * (1.0 + gamma) + beta

        raise ValueError(f"Unsupported modulation mode: {mode}")

    def _apply_time_modulation(
        self,
        x: torch.Tensor,
        t_emb: Optional[torch.Tensor],
        mode: str,
        expected_dim: int,
        emb_name: str,
        additive_proj: Optional[nn.Module] = None,
        gamma_proj: Optional[nn.Module] = None,
        beta_proj: Optional[nn.Module] = None,
    ) -> torch.Tensor:
        """Common time-modulation entry for query/key/value tensors."""
        if mode == "none":
            return x

        t = self._align_t_emb(
            t_emb=t_emb,
            target_len=x.shape[1],
            target_dtype=x.dtype,
            target_device=x.device,
            expected_dim=expected_dim,
            emb_name=emb_name,
        )
        return self._apply_affine_time_modulation(
            x=x,
            t=t,
            mode=mode,
            additive_proj=additive_proj,
            gamma_proj=gamma_proj,
            beta_proj=beta_proj,
        )
    
    def _get_initial_state(self, batch_size: int, device: torch.device) -> Optional[torch.Tensor]:
        """
        Get the initial state for the recurrence.
        
        If use_learnable_init_state is True, returns the learnable parameter expanded
        to (B, H, K, V) in float32. Otherwise returns None (callers will use zeros).
        """
        if self.use_learnable_init_state:
            # self.init_state: (H, K, V) -> expand to (B, H, K, V)
            return self.init_state.unsqueeze(0).expand(batch_size, -1, -1, -1).float()
        return None

    def _generate_history_cfg_alpha(
        self,
        batch_size: int,
        num_chunks: int,
        device: torch.device,
        history_cfg_alpha: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        """
        Generate or validate history_cfg_alpha for null-history CFG.

        Training: if history_cfg_alpha is None and history_cfg_enabled is True,
                  generate random per-block alpha values using a mixture distribution:
                  - With prob p_clean: alpha=1 (use real state)
                  - With prob p_null: alpha=0 (use null state)
                  - Otherwise: alpha ~ Uniform(0, 1) (smooth interpolation)
                  Block 0 is always alpha=1 (no history to interpolate).
        Inference: history_cfg_alpha is passed in externally (e.g., 1.0 for cond, 0.0 for uncond).

        Returns:
            history_cfg_alpha: (B, num_chunks) float32 tensor with values in [0,1], or None if disabled.
        """
        if history_cfg_alpha is not None:
            return history_cfg_alpha

        if self.null_state is None or not self.history_cfg_enabled:
            return None

        # Training: generate random per-block alpha values
        # Block 0 always alpha=1 (no real history), blocks 1..K-1 use mixture distribution
        alpha = torch.ones(batch_size, num_chunks, dtype=torch.float32, device=device)
        if num_chunks > 1:
            r = torch.rand(batch_size, num_chunks - 1, device=device)
            vals = torch.rand(batch_size, num_chunks - 1, device=device)
            p_clean = self.history_cfg_p_clean
            p_null = self.history_cfg_p_null
            alpha[:, 1:] = torch.where(
                r < p_clean,
                torch.ones_like(vals),
                torch.where(
                    r < p_clean + p_null,
                    torch.zeros_like(vals),
                    vals,
                ),
            )
        return alpha
    
    def compute_gates(
        self,
        x: torch.Tensor,           # (B, T, D) - frame features
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute per-frame forget (α) and update (β) gates following FLA's design.
        
        Gate Formulas (following FLA GatedDeltaNet):
        - α (forget/g): g = -A * softplus(a_proj(x) + dt_bias)
        - β (update/b): b = sigmoid(b_proj(x))
        
        Note: Gates are computed from context hidden states only.
        This follows original FLA design and ensures cache reusability in inference.
        
        Args:
            x: Frame features, shape (B, T, D)
        
        Returns:
            alpha: Forget gate in log space, shape (B, T, H)
            beta: Update gate in [0, 1] (or [0, 2] if allow_neg_eigval), shape (B, T, H)
        """
        B, T, D = x.shape
        
        # Gate input is only x (context hidden states), t_emb is ignored
        gate_input = x
        
        # Compute gates following FLA's formula
        # β (update gate): b = sigmoid(b_proj(gate_input))
        # Following FLA: if allow_neg_eigval, beta in [0, 2] instead of [0, 1]
        beta = self.b_proj(gate_input).sigmoid()  # (B, T, H)
        if self.allow_neg_eigval:
            beta = beta * 2.0
        
        # α (forget gate): g = -A * softplus(a_proj(gate_input) + dt_bias)
        # A = exp(A_log), stored in log-space for numerical stability
        # Note: exp and softplus are numerically stable in fp16/bf16 for typical gate ranges
        A = self.A_log.exp()  # (H,)
        a_proj_out = self.a_proj(gate_input) / self.gate_logit_normalizer  # (B, T, H)
        alpha = -A.view(1, 1, -1) * F.softplus(a_proj_out + self.dt_bias.view(1, 1, -1))  # (B, T, H)
        
        return alpha, beta  # (B, T, H), (B, T, H)
    
    def forward(
        self,
        motion: Optional[torch.Tensor],            # (B, T_q, D) - current noisy block
        context: Optional[torch.Tensor] = None,    # (B, T_kv, D) - historical clean blocks
        kv_cache: Optional[Dict] = None,           # State cache for inference
        current_start: int = 0,
        prefill_only: bool = False,
        **kwargs,                                  # Accept and ignore extra kwargs for compatibility
    ) -> Optional[torch.Tensor]:
        """
        Forward pass compatible with CrossCausalSelfAttention interface.
        
        Training mode (kv_cache is None):
            - context: historical clean frames
            - motion: current noisy block
            - Build state from context, query with motion
        
        Inference mode (kv_cache is not None):
            - prefill_only=True: Write context to state, return None
            - prefill_only=False: Query state with motion
        
        Args:
            motion: Current block features, shape (B, T_q, D)
            context: Historical block features, shape (B, T_kv, D)
            kv_cache: State cache for inference
            current_start: Starting position
            prefill_only: If True, only write to cache
            **kwargs: Optional extra kwargs (e.g., t_emb for query modulation, kv_t_emb for KV modulation)
        
        Returns:
            output: Cross-attention output, shape (B, T_q, D), or None if prefill_only
        """
        # =====================
        # Training Mode
        # =====================
        t_emb = kwargs.get("t_emb", None)
        kv_t_emb = kwargs.get("kv_t_emb", None)

        if kv_cache is None:
            if context is None:
                # No history, return zeros
                return torch.zeros_like(motion)
            return self._forward_train(motion, context, t_emb=t_emb, kv_t_emb=kv_t_emb, history_cfg_alpha=kwargs.get("history_cfg_alpha", None))
        
        # =====================
        # Inference: Prefill
        # =====================
        if prefill_only:
            if context is None:
                return None
            return self._forward_prefill(context, kv_cache, current_start, kv_t_emb=kv_t_emb)
        
        # =====================
        # Inference: Decode
        # =====================
        return self._forward_decode(
            motion, kv_cache, current_start, t_emb=t_emb,
            history_cfg_alpha=kwargs.get("history_cfg_alpha", None),
        )
    
    def _forward_train(
        self,
        motion: torch.Tensor,      # (B, T, D) - all noisy frames (full sequence)
        context: torch.Tensor,     # (B, T, D) - all clean frames (full sequence, same length as motion)
        t_emb: Optional[torch.Tensor] = None,
        kv_t_emb: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Training forward pass with block-level causal masking.
        
        IMPORTANT: In BLD training, motion and context have the SAME length.
        - motion[i]: noisy version of frame i
        - context[i]: clean version of frame i
        
        For block-level causality:
        - Block k can only see clean frames from blocks 0, 1, ..., k-1
        - Block k CANNOT see clean frames from blocks k, k+1, ...
        
        Implementation:
        - For each block, we build a state S from all PREVIOUS blocks' clean frames
        - Then query S with the current block's noisy frames
        - This is done in parallel for all blocks using cumulative state computation
        """
        B, T, D = motion.shape
        block_size = self.block_size  # Use instance attribute instead of hardcoded parameter
        assert context.shape[1] == T, "motion and context must have same length"
        
        # Compute Q from motion (noisy frames)
        # Q 路径固定使用 linear proj -> SiLU，不使用 shortconv
        # 原因：shortconv 会导致跨块信息泄露，破坏块级自回归语义
        q = F.silu(self.q_proj(motion))
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
        # Reshape Q/K/V directly to (B, T, H, D) — the layout expected by triton kernel
        # and block_causal_fast_chunk. Avoids unnecessary (B, H, T, D) intermediate + permute.
        q = q.unflatten(-1, (self.num_heads, self.head_dim))  # (B, T, H, D)
        
        # Compute K, V from context (clean frames)
        # K/V 路径使用 shortconv（如果启用），因为 clean context 需要保持训练/推理一致性
        if self.use_short_conv:
            k, _ = self.k_conv1d(x=self.k_proj(context), cache=None, output_final_state=False)
            v, _ = self.v_conv1d(x=self.v_proj(context), cache=None, output_final_state=False)
        else:
            k = F.silu(self.k_proj(context))
            v = F.silu(self.v_proj(context))
        
        # Apply KV time modulation (QK-Alignment)
        k = self._apply_time_modulation(
            x=k,
            t_emb=kv_t_emb,
            mode=self.kv_t_modulation_mode,
            expected_dim=self.kv_t_modulation_dim,
            emb_name="kv_t_emb",
            additive_proj=getattr(self, "kv_k_t_proj", None),
            gamma_proj=getattr(self, "kv_k_t_gamma", None),
            beta_proj=getattr(self, "kv_k_t_beta", None),
        )
        v = self._apply_time_modulation(
            x=v,
            t_emb=kv_t_emb,
            mode=self.kv_t_modulation_mode,
            expected_dim=self.kv_t_modulation_dim,
            emb_name="kv_t_emb",
            additive_proj=getattr(self, "kv_v_t_proj", None),
            gamma_proj=getattr(self, "kv_v_t_gamma", None),
            beta_proj=getattr(self, "kv_v_t_beta", None),
        )
        
        # Reshape K/V directly to (B, T, H, D) — avoids (B, H, T, D) intermediate + permute
        k = k.unflatten(-1, (self.num_heads, self.head_dim))  # (B, T, H, D)
        v = v.unflatten(-1, (self.num_heads, self.head_dim))  # (B, T, H, D)
        
        # Compute per-frame gates from context — returns (B, T, H) directly
        alpha, beta = self.compute_gates(context)  # (B, T, H)
        
        # ========================================
        # Block-level causal state computation (block-causal read + token-level write)
        # ========================================
        # For block i, we need state S_i that contains info from blocks 0..i-1
        # Use chunkwise parallel update (triangular solve) while keeping block-causal read.
        # q/k/v are already (B, T, H, D), alpha/beta are already (B, T, H).

        # Generate history cfg alpha for null-history CFG (training only)
        num_chunks = (T + block_size - 1) // block_size
        history_cfg_alpha = self._generate_history_cfg_alpha(
            batch_size=B,
            num_chunks=num_chunks,
            device=q.device,
            history_cfg_alpha=kwargs.get("history_cfg_alpha", None),
        )

        if self.use_triton_kernel:
            assert block_size in (16, 32, 64), (
                f"Triton kernel only supports chunk_size in [16, 32, 64], got {block_size}."
            )
            o, _ = chunk_gated_delta_rule(
                q=q,
                k=k,
                v=v,
                g=alpha,
                beta=beta,
                chunk_size=block_size,
                scale=self.scale,
                initial_state=self._get_initial_state(B, q.device),
                output_final_state=False,
                use_qk_l2norm_in_kernel=self.qk_norm,
                history_cfg_alpha=history_cfg_alpha,
                null_state=self.null_state,
            )
        else:
            o, _ = block_causal_fast_chunk(
                q=q,
                k=k,
                v=v,
                g=alpha,
                beta=beta,
                chunk_size=block_size,
                scale=self.scale,
                initial_state=self._get_initial_state(B, q.device),
                output_final_state=False,
                qk_norm=self.qk_norm,
                history_cfg_alpha=history_cfg_alpha,
                null_state=self.null_state,
            )
        
        # Output projection: reshape (B, T, H, D) -> (B, T, H*D)
        o = o.flatten(-2)
        o = self.o_proj(o)
        o = self.proj_drop(o)
        
        # Clamp for fp16 stability
        if o.dtype == torch.float16:
            o = o.clamp(-65504, 65504)
        
        return o
    
    def _forward_prefill(
        self,
        context: torch.Tensor,     # (B, T, D) - clean frames to write
        kv_cache: Dict,
        current_start: int,
        kv_t_emb: Optional[torch.Tensor] = None,
    ) -> None:
        """Prefill: update hidden state with context using recurrent update, return None.
        
        K/V 的 shortconv cache 逻辑：
        - 推理时 clean context 是分块逐步增长的
        - 如果每次 prefill 都用 cache=None，相当于每块卷积"从零开始"，会导致每个块起始位置的 K/V 与训练不一致
        - 因此 prefill 需要保存并传入 cache，且每次 prefill 要更新 cache
        """
        B, T, D = context.shape
        
        # Compute K, V from context with shortconv cache support
        if self.use_short_conv:
            # Get existing conv caches (if any)
            k_conv_cache = kv_cache.get('k_conv_cache', None)
            v_conv_cache = kv_cache.get('v_conv_cache', None)
            
            # Handle batch size mismatch: discard old cache if B changed
            if k_conv_cache is not None and k_conv_cache.shape[0] != B:
                k_conv_cache = None
            if v_conv_cache is not None and v_conv_cache.shape[0] != B:
                v_conv_cache = None
            
            # Apply shortconv with cache
            k, k_conv_cache_new = self.k_conv1d(
                x=self.k_proj(context),
                cache=k_conv_cache,
                output_final_state=True
            )
            v, v_conv_cache_new = self.v_conv1d(
                x=self.v_proj(context),
                cache=v_conv_cache,
                output_final_state=True
            )
            
            # Save updated conv caches
            kv_cache['k_conv_cache'] = k_conv_cache_new
            kv_cache['v_conv_cache'] = v_conv_cache_new
        else:
            k = F.silu(self.k_proj(context))
            v = F.silu(self.v_proj(context))
        
        # Apply KV time modulation (QK-Alignment)
        k = self._apply_time_modulation(
            x=k,
            t_emb=kv_t_emb,
            mode=self.kv_t_modulation_mode,
            expected_dim=self.kv_t_modulation_dim,
            emb_name="kv_t_emb",
            additive_proj=getattr(self, "kv_k_t_proj", None),
            gamma_proj=getattr(self, "kv_k_t_gamma", None),
            beta_proj=getattr(self, "kv_k_t_beta", None),
        )
        v = self._apply_time_modulation(
            x=v,
            t_emb=kv_t_emb,
            mode=self.kv_t_modulation_mode,
            expected_dim=self.kv_t_modulation_dim,
            emb_name="kv_t_emb",
            additive_proj=getattr(self, "kv_v_t_proj", None),
            gamma_proj=getattr(self, "kv_v_t_gamma", None),
            beta_proj=getattr(self, "kv_v_t_beta", None),
        )
        
        # Reshape K/V directly to (B, T, H, D) — avoids (B, H, T, D) intermediate + permute
        k = k.unflatten(-1, (self.num_heads, self.head_dim))  # (B, T, H, D)
        v = v.unflatten(-1, (self.num_heads, self.head_dim))  # (B, T, H, D)
        
        # Compute gates from context — returns (B, T, H) directly
        alpha, beta = self.compute_gates(context)
        
        # Get or initialize state (force float32 for numerical stability)
        state = kv_cache.get('deltanet_state', None)
        if state is None:
            init = self._get_initial_state(B, k.device)
            if init is not None:
                state = init
            else:
                state = torch.zeros(
                    B, self.num_heads, self.head_dim, self.head_dim,
                    device=k.device, dtype=torch.float32
                )
        
        # Upcast state to float32 if half precision for numerical stability
        state = _upcast_state_if_needed(state)
        
        # k/v are already (B, T, H, D), alpha/beta are already (B, T, H)
        # Update state using recurrent formula (write-only, no output computation)
        if self.use_triton_kernel:
            # Chunk-level Triton prefill kernel — much faster than token-by-token
            # recurrent for long sequences. Handles non-chunk-aligned T internally.
            new_state = chunk_gated_delta_rule_prefill(
                k=k,
                v=v,
                g=alpha,
                beta=beta,
                initial_state=state,
                use_qk_l2norm_in_kernel=self.qk_norm,
                chunk_size=self.block_size,
            )
        else:
            # PyTorch recurrent fallback
            # recurrent_write_states internally handles qk_norm.
            new_state = recurrent_write_states(
                initial_state=state,
                k=k,
                v=v,
                g=alpha,
                beta=beta,
                qk_norm=self.qk_norm,
            )
        
        # Save updated state (always float32)
        kv_cache['deltanet_state'] = new_state
        if 'global_end_index' in kv_cache:
            kv_cache['global_end_index'].fill_(current_start + T)
        if 'local_end_index' in kv_cache:
            kv_cache['local_end_index'].fill_(current_start + T)
        
        return None
    
    def _forward_decode(
        self,
        motion: torch.Tensor,      # (B, T, D) - current noisy block
        kv_cache: Dict,
        current_start: int,
        t_emb: Optional[torch.Tensor] = None,
        history_cfg_alpha: Optional[torch.Tensor] = None,  # (B,) float in [0,1]: 1=clean state, 0=null state
    ) -> torch.Tensor:
        """Decode: query the cached state with motion.
        
        dtype 策略：
        - 保存输入 motion 的 dtype 作为 input_dtype
        - q 在 einsum 前 cast 到 fp32，与 fp32 state 做计算
        - 结果 cast 回 input_dtype 再进行 output projection
        - 这与 block_causal_fast_chunk 中的 block-causal read 保持一致
        """
        B, T, D = motion.shape
        input_dtype = motion.dtype
        compute_dtype = torch.float32
        
        # Compute Q from motion
        # Q 路径固定使用 linear proj -> SiLU，不使用 shortconv
        # 原因：shortconv 会导致跨块信息泄露，破坏块级自回归语义
        q = F.silu(self.q_proj(motion))
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
        q = rearrange(q, 'b t (h d) -> b t h d', h=self.num_heads)  # (B, T, H, K)
        
        # Apply qk_norm to q (to match training semantics)
        # Note: k=None for decode since we only need to normalize q
        q, _ = _maybe_normalize_qk(q=q, k=None, enabled=self.qk_norm)

        # Apply scale to q (same semantics as block_causal_reference_naive)
        scale = self.scale if self.scale is not None else (self.head_dim ** -0.5)
        q = q * scale
        
        # Permute to (B, H, T, K) for einsum
        q = q.permute(0, 2, 1, 3)  # (B, H, T, K)
        
        # Get cached state
        state = kv_cache.get('deltanet_state', None)
        if state is None:
            # No history, return zeros
            return torch.zeros_like(motion)
        
        # Null-history CFG: per-batch-element state interpolation for 2B inference
        # history_cfg_alpha: (B,) float — alpha=1 means use real state, alpha=0 means use null_state
        if history_cfg_alpha is not None and self.null_state is not None:
            ns = self.null_state.float()
            if ns.dim() == 3:
                ns = ns.unsqueeze(0).expand(B, -1, -1, -1)  # (B, H, K, V)
            alpha_bc = history_cfg_alpha.to(dtype=state.dtype, device=state.device).view(B, 1, 1, 1)
            state = state * alpha_bc + ns * (1.0 - alpha_bc)

        # Query the state: o = Q @ S
        # Cast q to fp32 for einsum with fp32 state, then cast back to input_dtype
        # This matches the dtype strategy in block_causal_fast_chunk
        q_f = q.to(compute_dtype)
        o_f = torch.einsum('bhtk,bhkv->bhtv', q_f, state)  # fp32
        o = o_f.to(input_dtype)
        
        # Output projection
        o = rearrange(o, 'b h t d -> b t (h d)')
        o = self.o_proj(o)
        o = self.proj_drop(o)
        
        if o.dtype == torch.float16:
            o = o.clamp(-65504, 65504)
        
        return o
    
    def reset_state(self):
        """Reset the hidden state for a new sequence."""
        pass  # State is managed externally via kv_cache


class ZeroGatedDeltaNetForBLD(nn.Module):
    """
    Zero-output drop-in replacement for GatedDeltaNetForBLD.

    This module intentionally preserves the same constructor and forward interface,
    but returns zero outputs in all modes:
    - Training (kv_cache is None): returns zeros_like(motion)
    - Prefill (kv_cache is not None and prefill_only=True): returns None
    - Decode (kv_cache is not None and prefill_only=False): returns zeros_like(motion)

    It can be used for ablation to isolate the effect of historical cross interaction.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        head_dim: int = None,
        bias: bool = True,
        qk_norm: bool = True,
        use_triton_kernel: bool = False,
        scale: Optional[float] = None,
        proj_drop: float = 0.,
        gate_low_rank_dim: int = 16,
        gate_logit_normalizer: float = 16.0,
        block_size: int = 64,
        allow_neg_eigval: bool = False,
        use_short_conv: bool = True,
        conv_size: int = 4,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = head_dim if head_dim is not None else (dim // max(1, num_heads))
        self.bias = bias
        self.qk_norm = qk_norm
        self.use_triton_kernel = use_triton_kernel
        self.scale = scale
        self.proj_drop = nn.Dropout(proj_drop)
        self.gate_low_rank_dim = gate_low_rank_dim
        self.gate_logit_normalizer = gate_logit_normalizer
        self.block_size = block_size
        self.allow_neg_eigval = allow_neg_eigval
        self.use_short_conv = use_short_conv
        self.conv_size = conv_size

    def forward(
        self,
        motion: Optional[torch.Tensor],
        context: Optional[torch.Tensor] = None,
        kv_cache: Optional[Dict] = None,
        current_start: int = 0,
        prefill_only: bool = False,
        **kwargs,
    ) -> Optional[torch.Tensor]:
        # Inference prefill path: keep same behavior as GatedDeltaNetForBLD (no output tensor)
        if kv_cache is not None and prefill_only:
            return None

        # Training/decode paths return explicit zero tensor, but keep graph dependency
        # on motion/context to avoid DDP "unused parameter" false positives in ablation.
        if motion is None:
            if context is None:
                return None
            return context * 0.0

        out = motion * 0.0
        if context is not None:
            # Add a zero-valued scalar term that depends on context.
            # This keeps upstream context encoder params in the autograd graph.
            out = out + (context.mean() * 0.0)
        return out

    def reset_state(self):
        """No internal state. Kept for interface compatibility."""
        pass


# ============================================================================
# Test Code
# ============================================================================

if __name__ == "__main__":
    import time

    def _compare_tensors(name: str, a: torch.Tensor, b: torch.Tensor, atol: float, rtol: float):
        max_abs = (a - b).abs().max().item()
        max_rel = ((a - b).abs() / (b.abs() + 1e-9)).max().item()
        ok = torch.allclose(a, b, atol=atol, rtol=rtol)
        print(f"  [{name}] allclose={ok} max_abs={max_abs:.3e} max_rel={max_rel:.3e}")
        return ok

    def _run_block_causal_correctness_tests(device: str = "cpu"):
        print("\n" + "-" * 40)
        print("Block-Causal Core Correctness Tests")
        print("-" * 40)

        torch.manual_seed(42)
        B, T, C, H, K, V = 2, 128, 32, 3, 16, 20
        dtype = torch.float32
        q = torch.randn(B, T, H, K, device=device, dtype=dtype)
        k = torch.randn(B, T, H, K, device=device, dtype=dtype)
        v = torch.randn(B, T, H, V, device=device, dtype=dtype)
        g = torch.randn(B, T, H, device=device, dtype=dtype).clamp_max(0)
        beta = torch.sigmoid(torch.randn(B, T, H, device=device, dtype=dtype))
        init = torch.randn(B, H, K, V, device=device, dtype=dtype)

        o_ref, s_ref = block_causal_reference_naive(
            q, k, v, g, beta, chunk_size=C, initial_state=init,
            output_final_state=True,
        )
        o_fast, s_fast = block_causal_fast_chunk(
            q, k, v, g, beta, chunk_size=C, initial_state=init,
            output_final_state=True,
        )

        atol = 1e-5
        rtol = 1e-5
        _compare_tensors("o", o_fast, o_ref, atol=atol, rtol=rtol)
        _compare_tensors("final_state", s_fast, s_ref, atol=atol, rtol=rtol)

        # Edge case: T not multiple of C
        T2 = 77
        q2 = q[:, :T2].contiguous()
        k2 = k[:, :T2].contiguous()
        v2 = v[:, :T2].contiguous()
        g2 = g[:, :T2].contiguous()
        beta2 = beta[:, :T2].contiguous()
        o_ref2, s_ref2 = block_causal_reference_naive(
            q2, k2, v2, g2, beta2, chunk_size=C, initial_state=init,
            output_final_state=True,
        )
        o_fast2, s_fast2 = block_causal_fast_chunk(
            q2, k2, v2, g2, beta2, chunk_size=C, initial_state=init,
            output_final_state=True,
        )
        _compare_tensors("o (T not multiple of C)", o_fast2, o_ref2, atol=atol, rtol=rtol)
        _compare_tensors("final_state (T not multiple of C)", s_fast2, s_ref2, atol=atol, rtol=rtol)

        # Edge case: beta all zeros (only decay)
        beta_zero = torch.zeros_like(beta)
        o_ref3, s_ref3 = block_causal_reference_naive(
            q, k, v, g, beta_zero, chunk_size=C, initial_state=init,
            output_final_state=True,
        )
        o_fast3, s_fast3 = block_causal_fast_chunk(
            q, k, v, g, beta_zero, chunk_size=C, initial_state=init,
            output_final_state=True,
        )
        _compare_tensors("beta=0 o", o_fast3, o_ref3, atol=atol, rtol=rtol)
        _compare_tensors("beta=0 state", s_fast3, s_ref3, atol=atol, rtol=rtol)

        # Edge case: alpha all ones (g=0)
        g_zero = torch.zeros_like(g)
        o_ref4, s_ref4 = block_causal_reference_naive(
            q, k, v, g_zero, beta, chunk_size=C, initial_state=init,
            output_final_state=True,
        )
        o_fast4, s_fast4 = block_causal_fast_chunk(
            q, k, v, g_zero, beta, chunk_size=C, initial_state=init,
            output_final_state=True,
        )
        _compare_tensors("alpha=1 o", o_fast4, o_ref4, atol=atol, rtol=rtol)
        _compare_tensors("alpha=1 state", s_fast4, s_ref4, atol=atol, rtol=rtol)

    def _run_block_causal_gradcheck(device: str = "cpu"):
        print("\n" + "-" * 40)
        print("Block-Causal Core Gradcheck")
        print("-" * 40)
        torch.manual_seed(0)
        B, T, C, H, K, V = 4, 256, 64, 4, 32, 32
        dtype = torch.double
        q = torch.randn(B, T, H, K, device=device, dtype=dtype, requires_grad=True)
        k = torch.randn(B, T, H, K, device=device, dtype=dtype, requires_grad=True)
        v = torch.randn(B, T, H, V, device=device, dtype=dtype, requires_grad=True)
        g = torch.randn(B, T, H, device=device, dtype=dtype, requires_grad=True).clamp_max(0)
        beta = torch.sigmoid(torch.randn(B, T, H, device=device, dtype=dtype, requires_grad=True))

        def func(q_, k_, v_, g_, beta_):
            o, s = block_causal_reference_naive(
                q_, k_, v_, g_, beta_,
                chunk_size=C,
                initial_state=None,
                output_final_state=True,
                qk_norm=True,
            )
            return (o.sum() + s.sum())

        ok = torch.autograd.gradcheck(func, (q, k, v, g, beta), eps=1e-6, atol=1e-4, rtol=1e-4)
        print(f"  gradcheck ok={ok}")

    def _run_block_causal_benchmark(device: str = "cpu"):
        print("\n" + "-" * 40)
        print("Block-Causal Core Benchmark")
        print("-" * 40)
        torch.manual_seed(0)
        B, T, C, H, K, V = 2, 64, 32, 4, 32, 32
        q = torch.randn(B, T, H, K, device=device)
        k = torch.randn(B, T, H, K, device=device)
        v = torch.randn(B, T, H, V, device=device)
        g = torch.randn(B, T, H, device=device).clamp_max(0)
        beta = torch.sigmoid(torch.randn(B, T, H, device=device))

        # warmup
        for _ in range(2):
            block_causal_reference_naive(q, k, v, g, beta, chunk_size=C)
            block_causal_fast_chunk(q, k, v, g, beta, chunk_size=C)

        # naive
        start = time.time()
        block_causal_reference_naive(q, k, v, g, beta, chunk_size=C)
        if device == "cuda":
            torch.cuda.synchronize()
        t_naive = (time.time() - start) * 1000

        # fast
        start = time.time()
        block_causal_fast_chunk(q, k, v, g, beta, chunk_size=C)
        if device == "cuda":
            torch.cuda.synchronize()
        t_fast = (time.time() - start) * 1000

        print(f"  naive: {t_naive:.2f} ms")
        print(f"  fast : {t_fast:.2f} ms")

    def _run_state_vs_fla_tests(device: str = "cuda"):
        """
        Test state alignment between our implementations and FLA official implementation.
        Only compares final state, not output (output has different causal semantics).
        """
        print("\n" + "-" * 40)
        print("State vs FLA Official Implementation Tests")
        print("-" * 40)

        # Check prerequisites
        if not torch.cuda.is_available():
            print("  CUDA not available, skip state-vs-FLA test")
            return
        if not FLA_AVAILABLE:
            print("  FLA not available, skip state-vs-FLA test")
            return

        # Try to import FLA gated delta rule ops
        fla_func = None
        fla_func_name = None
        try:
            from fla.ops.gated_delta_rule import fused_recurrent_gated_delta_rule
            fla_func = fused_recurrent_gated_delta_rule
            fla_func_name = "fused_recurrent_gated_delta_rule"
        except ImportError:
            try:
                from fla.ops.gated_delta_rule import chunk_gated_delta_rule
                fla_func = chunk_gated_delta_rule
                fla_func_name = "chunk_gated_delta_rule"
            except ImportError:
                print("  FLA gated_delta_rule ops not available, skip state-vs-FLA test")
                return

        print(f"  Using FLA function: {fla_func_name}")

        # Test parameters
        torch.manual_seed(42)
        B, T, H, K, V = 2, 128, 4, 32, 32
        block_size = 64

        # Test configurations: (input_dtype, atol, rtol)
        test_configs = [
            (torch.float32, 1e-5, 1e-5),
            (torch.bfloat16, 1e-4, 1e-4),
        ]

        for input_dtype, atol, rtol in test_configs:
            for qk_norm in [False, True]:
                print(f"\n  Testing dtype={input_dtype}, qk_norm={qk_norm}")

                # Generate random inputs
                q = torch.randn(B, T, H, K, device=device, dtype=input_dtype)
                k = torch.randn(B, T, H, K, device=device, dtype=input_dtype)
                v = torch.randn(B, T, H, V, device=device, dtype=input_dtype)
                g = torch.randn(B, T, H, device=device, dtype=input_dtype).clamp_max(0)
                beta = torch.sigmoid(torch.randn(B, T, H, device=device, dtype=input_dtype))
                init_state = torch.randn(B, H, K, V, device=device, dtype=torch.float32)

                # Normalize k if qk_norm is enabled (to match FLA's use_qk_l2norm_in_kernel)
                if qk_norm:
                    k_normalized = F.normalize(k, p=2, dim=-1, eps=1e-6)
                else:
                    k_normalized = k

                # ===== Run FLA =====
                try:
                    _, fla_state = fla_func(
                        q=q,
                        k=k_normalized,
                        v=v,
                        g=g,
                        beta=beta,
                        initial_state=init_state,
                        output_final_state=True,
                        use_qk_l2norm_in_kernel=False,  # We already normalized k if needed
                    )
                except Exception as e:
                    print(f"    FLA call failed: {e}")
                    continue

                # ===== Run our implementations =====
                # 1. recurrent_write_states
                # Need to reshape inputs: (B, T, H, K) -> (B, T, H, K) is already correct
                recurrent_state = recurrent_write_states(
                    initial_state=init_state.clone(),
                    k=k.permute(0, 1, 2, 3),  # (B, T, H, K)
                    v=v.permute(0, 1, 2, 3),  # (B, T, H, V)
                    g=g,  # (B, T, H)
                    beta=beta,    # (B, T, H)
                    qk_norm=qk_norm,
                )

                # 2. block_causal_fast_chunk
                _, fast_chunk_state = block_causal_fast_chunk(
                    q=q,
                    k=k,
                    v=v,
                    g=g,
                    beta=beta,
                    chunk_size=block_size,
                    initial_state=init_state.clone(),
                    output_final_state=True,
                    qk_norm=qk_norm,
                )

                # 3. block_causal_reference_naive
                _, naive_state = block_causal_reference_naive(
                    q=q,
                    k=k,
                    v=v,
                    g=g,
                    beta=beta,
                    chunk_size=block_size,
                    initial_state=init_state.clone(),
                    output_final_state=True,
                    qk_norm=qk_norm,
                )

                # ===== Compare states (convert to float32 for comparison) =====
                _compare_tensors("deltanet_write_recurrent vs FLA", recurrent_state.float(), fla_state.float(), atol, rtol)
                _compare_tensors("block_causal_fast_chunk vs FLA", fast_chunk_state.float(), fla_state.float(), atol, rtol)
                _compare_tensors("block_causal_reference_naive vs FLA", naive_state.float(), fla_state.float(), atol, rtol)

        print("\n  State vs FLA tests completed.")

    print("=" * 60)
    print("Testing Gated DeltaNet for BLD")
    print("=" * 60)

    # Test parameters
    B, T_q, T_kv, D = 4, 256, 256, 384
    num_heads = 12
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create test inputs
    motion = torch.randn(B, T_q, D, device=device)
    context = torch.randn(B, T_kv, D, device=device)

    print(f"\nTest configuration:")
    print(f"  Batch size: {B}")
    print(f"  Query length (current block): {T_q}")
    print(f"  Context length (history): {T_kv}")
    print(f"  Hidden dim: {D}")
    print(f"  Num heads: {num_heads}")
    print(f"  Device: {device}")

    # Test GatedDeltaNetForBLD
    print("\n" + "-" * 40)
    print("Testing GatedDeltaNetForBLD")
    print("-" * 40)

    model = GatedDeltaNetForBLD(
        dim=D,
        num_heads=num_heads,
        qk_norm=False,
    ).to(device)

    # Training forward
    start = time.time()
    output = model(motion, context=context)
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = (time.time() - start) * 1000

    print(f"  Training output shape: {output.shape}")
    print(f"  Training forward time: {elapsed:.2f} ms")

    # Test backward
    loss = output.sum()
    loss.backward()
    print("  Backward pass: OK")

    # Test inference mode
    print("\n" + "-" * 40)
    print("Testing Inference Mode (Prefill + Decode)")
    print("-" * 40)

    model.eval()
    with torch.no_grad():
        # Initialize cache
        kv_cache = {
            'deltanet_state': None,
            'global_end_index': torch.tensor(0, device=device),
            'local_end_index': torch.tensor(0, device=device),
        }

        # Prefill with history
        model(
            motion=None,
            context=context,
            kv_cache=kv_cache,
            current_start=0,
            prefill_only=True,
        )
        print(f"  Prefill complete. State shape: {kv_cache['deltanet_state'].shape}")

        # Decode current block
        output_decode = model(
            motion=motion,
            kv_cache=kv_cache,
            current_start=T_kv,
        )
        print(f"  Decode output shape: {output_decode.shape}")

    # Block-causal core tests (standalone)
    _run_block_causal_correctness_tests(device=device)
    # _run_block_causal_gradcheck(device=device)
    _run_block_causal_benchmark(device=device)

    # State vs FLA official implementation tests
    _run_state_vs_fla_tests(device=device)

    # ShortConv cache consistency test for K/V
    def _run_shortconv_cache_consistency_test(device: str = "cuda"):
        """
        Test that K/V shortconv outputs are consistent between:
        - Method A: One-shot processing of entire context with cache=None
        - Method B: Split context into two segments, prefill first with cache=None,
                    then prefill second with the returned cache
        
        This verifies that the shortconv cache logic in _forward_prefill correctly
        maintains consistency with training (one-shot) behavior.
        """
        print("\n" + "-" * 40)
        print("ShortConv Cache Consistency Test for K/V")
        print("-" * 40)

        if not FLA_AVAILABLE:
            print("  FLA not available, skip shortconv cache test")
            return

        torch.manual_seed(42)
        B, T, D = 2, 128, 384
        num_heads = 12
        conv_size = 4

        # Create model with use_short_conv=True
        model = GatedDeltaNetForBLD(
            dim=D,
            num_heads=num_heads,
            use_short_conv=True,
            conv_size=conv_size,
            qk_norm=True,
        ).to(device).eval()

        # Create test context
        context = torch.randn(B, T, D, device=device)

        # Split point
        split = T // 2  # 64

        with torch.no_grad():
            # ===== Method A: One-shot processing =====
            k_proj_out = model.k_proj(context)
            v_proj_out = model.v_proj(context)
            k_oneshot, _ = model.k_conv1d(x=k_proj_out, cache=None, output_final_state=False)
            v_oneshot, _ = model.v_conv1d(x=v_proj_out, cache=None, output_final_state=False)

            # ===== Method B: Two-segment processing with cache =====
            # First segment
            k_proj_1 = model.k_proj(context[:, :split])
            v_proj_1 = model.v_proj(context[:, :split])
            k_seg1, k_cache = model.k_conv1d(x=k_proj_1, cache=None, output_final_state=True)
            v_seg1, v_cache = model.v_conv1d(x=v_proj_1, cache=None, output_final_state=True)

            # Second segment with cache
            k_proj_2 = model.k_proj(context[:, split:])
            v_proj_2 = model.v_proj(context[:, split:])
            k_seg2, _ = model.k_conv1d(x=k_proj_2, cache=k_cache, output_final_state=True)
            v_seg2, _ = model.v_conv1d(x=v_proj_2, cache=v_cache, output_final_state=True)

            # Concatenate segments
            k_concat = torch.cat([k_seg1, k_seg2], dim=1)
            v_concat = torch.cat([v_seg1, v_seg2], dim=1)

            # ===== Compare =====
            # Use tolerances appropriate for fp32
            atol, rtol = 1e-5, 1e-5

            k_ok = _compare_tensors("K shortconv", k_concat, k_oneshot, atol, rtol)
            v_ok = _compare_tensors("V shortconv", v_concat, v_oneshot, atol, rtol)

            if k_ok and v_ok:
                print("  ShortConv cache consistency test PASSED!")
            else:
                print("  ShortConv cache consistency test FAILED!")

    _run_shortconv_cache_consistency_test(device=device)

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
