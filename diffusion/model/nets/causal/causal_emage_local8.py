from copy import deepcopy

import torch
import torch.nn as nn
from omegaconf import DictConfig

from diffusion.model.nets.attention import attention, vanilla_attention
from diffusion.model.nets.basic_modules import t2i_modulate
from diffusion.model.nets.causal.causal_emage import (
    CausalEmage,
    CausalEmageBlock,
    DEFAULT_BLOCK_CONFIG,
)
from diffusion.model.nets.positional_embeding import apply_causal_rotary_emb
from diffusion.model.norms import build_norm


DEFAULT_BLOCK_CONFIG_LOCAL8 = deepcopy(DEFAULT_BLOCK_CONFIG)
DEFAULT_BLOCK_CONFIG_LOCAL8["target"] = (
    "diffusion.model.nets.causal.causal_emage_local8.CausalEmageBlockLocal8"
)
DEFAULT_BLOCK_CONFIG_LOCAL8["params"]["history_frames"] = 8

DEFAULT_BLOCK_CONFIG_LOCAL8_CROSSFIRST = deepcopy(DEFAULT_BLOCK_CONFIG_LOCAL8)
DEFAULT_BLOCK_CONFIG_LOCAL8_CROSSFIRST["target"] = (
    "diffusion.model.nets.causal.causal_emage_local8.CausalEmageBlockLocal8CrossFirst"
)


class Local8CrossCausalSelfAttention(nn.Module):
    """
    A drop-in cross-attention module with the SAME calling interface as
    [CrossCausalSelfAttention](diffusion/model/nets/causal/causal_block.py:345),
    but history interaction is explicitly constrained to recent K frames.

    Training:
      - For each query block, K/V only come from nearest `history_frames` clean tokens
        before current block start.

    Inference:
      - prefill_only=True: keep only the last `history_frames` clean tokens in cache.
      - decode: query this short cached context.
    """

    def __init__(
        self,
        dim,
        num_heads=8,
        head_dim=None,
        bias=True,
        attn_type="flash",
        qk_norm_type="identity",
        attn_drop=0.0,
        proj_drop=0.0,
        block_size=64,
        history_frames=8,
        **kwargs,
    ):
        super().__init__()
        assert (num_heads is not None and head_dim is None) or (
            num_heads is None and head_dim is not None
        ), "only one of num_heads and head_dim should be specified"
        if num_heads is not None:
            assert dim % num_heads == 0, "dim should be divisible by num_heads"
            self.num_heads = num_heads
        else:
            assert dim % head_dim == 0, "head_dim should be divisible by dim"
            self.num_heads = dim // head_dim

        assert attn_type in ["flash", "vanilla", "naive"]
        # Keep "vanilla" as a backward-compatible alias for the non-flash path.
        self.attn_func = attention if attn_type == "flash" else vanilla_attention

        self.block_size = int(block_size)
        self.history_frames = int(history_frames)
        self.m_q_linear = nn.Linear(dim, dim, bias=bias)
        self.kv_linear = nn.Linear(dim, 2 * dim, bias=bias)

        self.mq_norm = build_norm(name=qk_norm_type, num_features=dim)
        self.mk_norm = build_norm(name=qk_norm_type, num_features=dim)

        self.motion_proj = nn.Linear(dim, dim, bias=bias)
        self.motion_proj_drop = nn.Dropout(proj_drop)
        self.attn_drop = float(attn_drop)

    def _project_q(self, motion: torch.Tensor) -> torch.Tensor:
        b, n, c = motion.shape
        q = self.m_q_linear(motion)
        dtype = q.dtype
        q = self.mq_norm(q)
        q = q.reshape(b, n, self.num_heads, c // self.num_heads).to(dtype)
        return q

    def _project_kv(self, context: torch.Tensor):
        b, n, c = context.shape
        kv = self.kv_linear(context).view(b, n, 2, c)
        k, v = kv.unbind(2)
        dtype = context.dtype
        k = self.mk_norm(k)
        k = k.reshape(b, n, self.num_heads, c // self.num_heads).to(dtype)
        v = v.reshape(b, n, self.num_heads, c // self.num_heads).to(dtype)
        return k, v

    def _apply_rope(self, x: torch.Tensor, rotary_emb, start_frame: int):
        if rotary_emb is None:
            return x
        return apply_causal_rotary_emb(
            x,
            rotary_emb,
            use_real_unbind_dim=-1,
            start_frame=start_frame,
        )

    def _finalize_out(self, attn_out: torch.Tensor, shape_like: torch.Tensor):
        b, n, c = shape_like.shape
        out = attn_out.reshape(b, n, c)
        out = self.motion_proj_drop(self.motion_proj(out))
        if out.is_cuda and torch.get_autocast_dtype("cuda") == torch.float16:
            out = out.clip(-65504, 65504)
        return out

    def _forward_train(
        self,
        motion: torch.Tensor,
        context: torch.Tensor,
        rotary_emb=None,
        current_start: int = 0,
    ):
        b, n, c = motion.shape
        q_all = self._project_q(motion)
        out = torch.zeros_like(q_all)
        dropout_p = self.attn_drop if self.training else 0.0

        block_size = max(1, self.block_size)
        for blk_start in range(0, n, block_size):
            blk_end = min(blk_start + block_size, n)
            hist_end = blk_start
            hist_start = max(0, hist_end - self.history_frames)

            if hist_end <= hist_start:
                continue

            q_blk = q_all[:, blk_start:blk_end]
            ctx_blk = context[:, hist_start:hist_end]
            k_blk, v_blk = self._project_kv(ctx_blk)

            q_blk = self._apply_rope(q_blk, rotary_emb, start_frame=current_start + blk_start)
            k_blk = self._apply_rope(k_blk, rotary_emb, start_frame=current_start + hist_start)

            out[:, blk_start:blk_end] = self.attn_func(
                q_blk,
                k_blk,
                v_blk,
                dropout_p=dropout_p,
            )

        return self._finalize_out(out, motion)

    def _forward_prefill(
        self,
        context: torch.Tensor,
        kv_cache: dict,
        current_start: int,
        rotary_emb=None,
    ):
        if context is None:
            return None

        tail_len = min(self.history_frames, context.shape[1])
        ctx_tail = context[:, -tail_len:]
        k, v = self._project_kv(ctx_tail)

        hist_start = current_start + (context.shape[1] - tail_len)
        k = self._apply_rope(k, rotary_emb, start_frame=hist_start)

        kv_cache["local8_k"] = k
        kv_cache["local8_v"] = v

        if "global_end_index" in kv_cache:
            kv_cache["global_end_index"].fill_(current_start + context.shape[1])
        if "local_end_index" in kv_cache:
            kv_cache["local_end_index"].fill_(k.shape[1])
        return None

    def _forward_decode(
        self,
        motion: torch.Tensor,
        kv_cache: dict,
        current_start: int,
        rotary_emb=None,
    ):
        q = self._project_q(motion)
        q = self._apply_rope(q, rotary_emb, start_frame=current_start)

        k = kv_cache.get("local8_k", None)
        v = kv_cache.get("local8_v", None)
        if k is None or v is None:
            return torch.zeros_like(motion)

        k = k.to(device=q.device, dtype=q.dtype)
        v = v.to(device=q.device, dtype=q.dtype)
        out = self.attn_func(q, k, v, dropout_p=0.0)
        return self._finalize_out(out, motion)

    def forward(
        self,
        motion: torch.FloatTensor,
        context: torch.FloatTensor = None,
        mask=None,
        rotary_emb: torch.FloatTensor = None,
        use_liger_rope: bool = False,
        kv_cache=None,
        current_start: int = 0,
        update_kv_cache: bool = False,
        prefill_only: bool = False,
    ):
        del mask, use_liger_rope  # kept for call compatibility

        if kv_cache is None:
            assert context is not None, "training path requires clean context"
            return self._forward_train(
                motion=motion,
                context=context,
                rotary_emb=rotary_emb,
                current_start=current_start,
            )

        if prefill_only or update_kv_cache:
            return self._forward_prefill(
                context=context,
                kv_cache=kv_cache,
                current_start=current_start,
                rotary_emb=rotary_emb,
            )

        return self._forward_decode(
            motion=motion,
            kv_cache=kv_cache,
            current_start=current_start,
            rotary_emb=rotary_emb,
        )


class CausalEmageBlockLocal8(CausalEmageBlock):
    """
    Reuse [CausalEmageBlock](diffusion/model/nets/causal/causal_emage.py:110)
    but replace cross-history interaction with Local8 protocol.
    """

    def __init__(
        self,
        hidden_size,
        num_heads=8,
        head_dim=None,
        mapping_size=768,
        simple_modulation=False,
        bias=True,
        norm_type="rms",
        attn_type="flash",
        qk_norm_type="identity",
        ffn_type="mlp",
        ffn_act_type="gelu",
        ffn_proj_type="linear",
        ffn_mlp_ratio=4.0,
        attn_drop=0.0,
        attn_proj_drop=0.0,
        ffn_drop=0.0,
        drop_path=0.0,
        norm_eps=1e-6,
        is_teacher_forcing=False,
        num_frame_per_block=64,
        local_attn_size=-1,
        padding_unit_length=128,
        max_attention_size=512,
        sink_size=0,
        cross_attn_impl_type=None,
        use_timestep_fusion=False,
        timestep_dim=None,
        fusion_gate_hidden_dim=None,
        fusion_gate_activation="silu",
        history_frames: int = 8,
        **kwargs,
    ):
        # avoid duplicate keyword when upstream config still carries this field
        kwargs.pop("add_cross_attn_prefix", None)

        super().__init__(
            hidden_size=hidden_size,
            num_heads=num_heads,
            head_dim=head_dim,
            mapping_size=mapping_size,
            simple_modulation=simple_modulation,
            bias=bias,
            norm_type=norm_type,
            attn_type=attn_type,
            qk_norm_type=qk_norm_type,
            ffn_type=ffn_type,
            ffn_act_type=ffn_act_type,
            ffn_proj_type=ffn_proj_type,
            ffn_mlp_ratio=ffn_mlp_ratio,
            attn_drop=attn_drop,
            attn_proj_drop=attn_proj_drop,
            ffn_drop=ffn_drop,
            drop_path=drop_path,
            norm_eps=norm_eps,
            is_teacher_forcing=is_teacher_forcing,
            num_frame_per_block=num_frame_per_block,
            local_attn_size=local_attn_size,
            padding_unit_length=padding_unit_length,
            max_attention_size=max_attention_size,
            sink_size=sink_size,
            cross_attn_impl_type=cross_attn_impl_type,
            use_timestep_fusion=use_timestep_fusion,
            timestep_dim=timestep_dim,
            fusion_gate_hidden_dim=fusion_gate_hidden_dim,
            fusion_gate_activation=fusion_gate_activation,
            **kwargs,
        )
        cross_attn_impl_type = attn_type if cross_attn_impl_type is None else cross_attn_impl_type

        self.cross_attn = Local8CrossCausalSelfAttention(
            dim=hidden_size,
            num_heads=num_heads,
            head_dim=head_dim,
            bias=bias,
            attn_type=cross_attn_impl_type,
            qk_norm_type=qk_norm_type,
            attn_drop=attn_drop,
            proj_drop=attn_proj_drop,
            block_size=num_frame_per_block,
            history_frames=history_frames,
        )
        self.cross_attn2 = deepcopy(self.cross_attn)


class CausalEmageBlockLocal8CrossFirst(CausalEmageBlockLocal8):
    """
    Local8 block variant that keeps the Local8 history module unchanged, but
    swaps the serial execution order to:

        cross-attn -> self-attn -> ffn

    The parallel timestep-fusion path remains identical to the parent block.
    """

    def forward(
        self,
        hand: torch.FloatTensor,
        body: torch.FloatTensor,
        t: torch.FloatTensor,
        block_mask: torch.BoolTensor = None,
        clean_hand: torch.FloatTensor = None,
        clean_body: torch.FloatTensor = None,
        rotary_emb: torch.FloatTensor = None,
        use_liger_rope: bool = False,
        hand_kv_cache=None,
        body_kv_cache=None,
        current_start: int = 0,
        update_kv_cache: bool = False,
        is_inference: bool = False,
        **kwargs,
    ):
        if self.use_timestep_fusion:
            return super().forward(
                hand=hand,
                body=body,
                t=t,
                block_mask=block_mask,
                clean_hand=clean_hand,
                clean_body=clean_body,
                rotary_emb=rotary_emb,
                use_liger_rope=use_liger_rope,
                hand_kv_cache=hand_kv_cache,
                body_kv_cache=body_kv_cache,
                current_start=current_start,
                update_kv_cache=update_kv_cache,
                is_inference=is_inference,
                **kwargs,
            )

        assert hand.shape[0] == body.shape[0], "batch size must be the same"

        (
            hand_shift_msa, hand_scale_msa, hand_gate_msa,
            hand_shift_cross, hand_scale_cross, hand_gate_cross,
            hand_shift_mlp, hand_scale_mlp, hand_gate_mlp,
            body_shift_msa, body_scale_msa, body_gate_msa,
            body_shift_cross, body_scale_cross, body_gate_cross,
            body_shift_mlp, body_scale_mlp, body_gate_mlp,
        ) = self._compute_modulation_params(t)

        ####################################### Layer 1: Local8 Cross-Attention ############################################
        residual_hand = self.hand_norm2(hand)
        residual_body = self.body_norm2(body)

        residual_hand = t2i_modulate(residual_hand, hand_shift_cross, hand_scale_cross)
        residual_body = t2i_modulate(residual_body, body_shift_cross, body_scale_cross)

        residual_hand = self.cross_attn(
            motion=residual_hand,
            context=clean_hand,
            mask=block_mask,
            rotary_emb=rotary_emb,
            use_liger_rope=use_liger_rope,
            kv_cache=hand_kv_cache,
            current_start=current_start,
            update_kv_cache=update_kv_cache,
        )
        residual_body = self.cross_attn2(
            motion=residual_body,
            context=clean_body,
            mask=block_mask,
            rotary_emb=rotary_emb,
            use_liger_rope=use_liger_rope,
            kv_cache=body_kv_cache,
            current_start=current_start,
            update_kv_cache=update_kv_cache,
        )

        residual_hand = hand_gate_cross * residual_hand
        residual_body = body_gate_cross * residual_body

        hand = hand + self.hand_drop_path(residual_hand)
        body = body + self.body_drop_path(residual_body)

        ##################################### Layer 2: Block-wise Self-Attention #######################################
        residual_hand = self.hand_norm1(hand)
        residual_body = self.body_norm1(body)

        residual_hand = t2i_modulate(residual_hand, hand_shift_msa, hand_scale_msa)
        residual_body = t2i_modulate(residual_body, body_shift_msa, body_scale_msa)

        residual_hand = self.attn(
            residual_hand,
            rotary_emb=rotary_emb,
            current_start=current_start,
            is_inference=is_inference,
        )
        residual_body = self.attn2(
            residual_body,
            rotary_emb=rotary_emb,
            current_start=current_start,
            is_inference=is_inference,
        )

        residual_hand = hand_gate_msa * residual_hand
        residual_body = body_gate_msa * residual_body

        hand = hand + self.hand_drop_path(residual_hand)
        body = body + self.body_drop_path(residual_body)

        ######################################## Layer 3: FFN ########################################
        residual_hand = self.hand_norm3(hand)
        residual_body = self.body_norm3(body)

        residual_hand = t2i_modulate(residual_hand, hand_shift_mlp, hand_scale_mlp)
        residual_body = t2i_modulate(residual_body, body_shift_mlp, body_scale_mlp)

        residual_hand, residual_body = self.mlp(residual_hand, residual_body)

        residual_hand = hand_gate_mlp * residual_hand
        residual_body = body_gate_mlp * residual_body

        hand = hand + self.hand_drop_path(residual_hand)
        body = body + self.body_drop_path(residual_body)

        return hand, body


class CausalEmageLocal8(CausalEmage):
    """
    A clean variant model entry that keeps overall architecture unchanged,
    but allows selecting [CausalEmageBlockLocal8](diffusion/model/nets/causal/causal_emage_local8.py:286)
    from config without modifying
    [causal_emage.py](diffusion/model/nets/causal/causal_emage.py:1).
    """

    def __init__(self, *args, block_config: DictConfig = None, **kwargs):
        if block_config is None:
            block_config = deepcopy(DEFAULT_BLOCK_CONFIG_LOCAL8)
        super().__init__(*args, block_config=block_config, **kwargs)

