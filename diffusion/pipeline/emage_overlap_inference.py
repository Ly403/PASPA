"""
Overlap decode pipeline for causal EMAGE.

Consecutive blocks share O overlapping frames. The overlap region
is blended (linear or cosine) before being committed to output and
KV cache, producing smoother transitions at block boundaries.

Usage:
    pipeline = CausalEmageOverlapInferencePipeline(
        ..., overlap_frames=8, blend_mode="linear",
    )
    # Same interface as CausalEmageInferencePipeline
    motion, expression = pipeline.inference_with_seed(generator, noise, ...)
"""

import math
import torch
from diffusion.pipeline.emage_inference import CausalEmageInferencePipeline
from diffusion.utils import rotation_conversions as rc


class CausalEmageOverlapInferencePipeline(CausalEmageInferencePipeline):
    """Block-overlap variant of CausalEmageInferencePipeline.

    Instead of committing each denoised block immediately, the last O frames
    are held as *pending*. The next block independently generates O overlapping
    frames, which are blended with the pending tail before being committed
    (to both output buffer and KV cache). This smooths block-boundary jitter.
    """

    def __init__(self, *args, overlap_frames: int = 8, blend_mode: str = "linear", **kwargs):
        super().__init__(*args, **kwargs)
        assert 0 < overlap_frames < self.num_frame_per_block, (
            f"overlap_frames must be in (0, {self.num_frame_per_block}), got {overlap_frames}"
        )
        self.overlap_frames = overlap_frames
        self.blend_mode = blend_mode
        self.stride = self.num_frame_per_block - overlap_frames

    def inference(self, *args, **kwargs):
        raise NotImplementedError(
            "CausalEmageOverlapInferencePipeline only supports inference_with_seed(). "
            "Use the base CausalEmageInferencePipeline for inference() without seed."
        )

    def __repr__(self):
        return (
            f"CausalEmageOverlapInferencePipeline("
            f"block={self.num_frame_per_block}, overlap={self.overlap_frames}, "
            f"stride={self.stride}, blend={self.blend_mode})"
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _build_blend_weights(length: int, mode: str, device, dtype):
        """Returns (length,) tensor ramping from ~0 to ~1 (prev→curr transition).

        Endpoints 0 and 1 are excluded so blending is always a true mix.
        """
        if mode == "linear":
            return torch.linspace(0.0, 1.0, length + 2, device=device, dtype=dtype)[1:-1]
        elif mode == "cosine":
            t = torch.linspace(0.0, 1.0, length + 2, device=device, dtype=dtype)[1:-1]
            return 0.5 * (1.0 - torch.cos(math.pi * t))
        else:
            raise ValueError(f"Unknown blend_mode: {mode}. Supported: linear, cosine")

    def _blend_overlap(self, prev_tail: torch.Tensor, curr_head: torch.Tensor):
        """Blend two (B, O, C) motion tensors with rot6d re-normalization."""
        O = prev_tail.shape[1]
        w = self._build_blend_weights(O, self.blend_mode, prev_tail.device, prev_tail.dtype)
        w = w.unsqueeze(0).unsqueeze(-1)  # (1, O, 1)
        blended = (1.0 - w) * prev_tail + w * curr_head
        # Re-normalize rot6d after linear interpolation
        blended = torch.cat([
            rc.NormalizeRot6d(blended[:, :, :-3]),
            blended[:, :, -3:],
        ], dim=-1)
        return blended

    def _blend_overlap_expression(self, prev_tail: torch.Tensor, curr_head: torch.Tensor):
        """Blend two (B, O, 100) expression tensors (no rot6d structure)."""
        O = prev_tail.shape[1]
        w = self._build_blend_weights(O, self.blend_mode, prev_tail.device, prev_tail.dtype)
        w = w.unsqueeze(0).unsqueeze(-1)
        return (1.0 - w) * prev_tail + w * curr_head

    # ------------------------------------------------------------------
    # Sample without prefill
    # ------------------------------------------------------------------

    def _sample_block_only(
        self,
        generator,
        noisy_block: torch.Tensor,
        context_: torch.Tensor,
        expression_: torch.Tensor,
        speaker_id: torch.Tensor,
        current_start_frame: int,
        seq_len: int,
        cache_dtype=None,
        normalize_rot6d: bool = True,
        post_process_func=None,
    ):
        """Sample a denoised block WITHOUT writing to KV cache.

        Mirrors the sampling logic of ``_sample_block_with_cache()`` but
        omits the subsequent ``_prefill_kv_cache()`` call so that the
        caller can blend before committing.
        """
        if cache_dtype is None:
            cache_dtype = noisy_block.dtype
        denoised_pred, _extra = self.sampler.sample(
            generator,
            noisy_block,
            context=context_,
            speaker_id=speaker_id,
            expression=expression_,
            kv_cache={
                "hand_kv_cache": self.kv_cache1,
                "body_kv_cache": self.kv_cache2,
            },
            current_start=current_start_frame * self.frame_seq_length,
            seq_len=seq_len,
            prefill_only=False,
            update_kv_cache=False,
            return_intermediate=False,
            return_model_output=False,
            post_process_func=post_process_func,
        )
        if normalize_rot6d:
            denoised_pred = torch.cat([
                rc.NormalizeRot6d(denoised_pred[:, :, :-3]),
                denoised_pred[:, :, -3:],
            ], dim=-1)
        return denoised_pred

    # ------------------------------------------------------------------
    # Commit segment (output + expression + KV cache)
    # ------------------------------------------------------------------

    def _commit_segment(
        self,
        generator,
        output: torch.Tensor,
        expression_output: torch.Tensor,
        motion_segment: torch.Tensor,
        expression_segment: torch.Tensor,
        context_segment: torch.Tensor,
        speaker_id: torch.Tensor,
        global_start_frame: int,
        seq_len: int,
        cache_dtype,
    ):
        """Write a motion/expression segment to output buffers and KV cache."""
        seg_len = motion_segment.shape[1]
        end = global_start_frame + seg_len
        output[:, global_start_frame:end] = motion_segment
        expression_output[:, global_start_frame:end] = expression_segment
        self._prefill_kv_cache(
            generator,
            clean_motion=motion_segment.to(cache_dtype),
            context_=context_segment,
            expression_=expression_segment,
            speaker_id=speaker_id,
            current_start_frame=global_start_frame,
            seq_len=seq_len,
        )

    # ------------------------------------------------------------------
    # Main inference loop (override)
    # ------------------------------------------------------------------

    def inference_with_seed(
        self,
        generator,
        noise: torch.Tensor,
        context: torch.Tensor,
        context_pid: torch.Tensor,
        seed_motion: torch.Tensor = None,
        seed_pad_to_block: bool = False,
        profile: bool = False,
        noise_shuffle: bool = False,
        noise_shuffle_window: int = 4,
        quintic_bridge: bool = False,
        **kwargs,
    ):
        """Autoregressive block diffusion with overlap blending.

        Overlap parameters are set in __init__. The following parent-pipeline
        features are incompatible with overlap decode and will raise if enabled:
        quintic_bridge, noise_shuffle.
        """
        # ---- Guards: features incompatible with overlap decode ----
        if quintic_bridge:
            raise ValueError(
                "quintic_bridge is incompatible with overlap decode. "
                "Overlap blending already smooths block boundaries — "
                "disable quintic_bridge when using CausalEmageOverlapInferencePipeline."
            )
        if noise_shuffle:
            raise ValueError(
                "noise_shuffle is incompatible with overlap decode. "
                "The parent's _shuffle_boundary_noise assumes stride=block_size, "
                "which does not match overlap decode's stride-based block layout."
            )

        batch_size, num_frames_original, num_channels = noise.shape
        cache_dtype = kwargs.get("cache_dtype", noise.dtype)
        precomputed_expression = kwargs.get(
            "precomputed_expression_context",
            None,
        )
        num_frame_per_block = self.num_frame_per_block
        overlap = self.overlap_frames
        stride = self.stride

        # ---- Seed setup (identical to parent) ----
        seed_num_frames = 0
        seed_prefill_frames = 0
        seed_prefill_motion = None
        if seed_motion is not None:
            seed_num_frames = seed_motion.shape[1]
            if seed_num_frames == 0:
                seed_motion = None
            elif seed_num_frames > num_frames_original:
                raise ValueError(
                    f"seed_num_frames ({seed_num_frames}) > total frames ({num_frames_original})"
                )
            else:
                if seed_pad_to_block:
                    seed_prefill_frames = math.ceil(seed_num_frames / num_frame_per_block) * num_frame_per_block
                    padding_frames = seed_prefill_frames - seed_num_frames
                    if padding_frames > 0:
                        padding = torch.zeros(
                            [batch_size, padding_frames, num_channels],
                            device=seed_motion.device,
                            dtype=seed_motion.dtype,
                        )
                        seed_prefill_motion = torch.cat([seed_motion, padding], dim=1)
                    else:
                        seed_prefill_motion = seed_motion
                else:
                    seed_prefill_frames = seed_num_frames
                    seed_prefill_motion = seed_motion

        # ---- Stride-aligned padding ----
        frames_to_denoise = num_frames_original - seed_num_frames

        # Need: frames_to_denoise_padded = overlap + n_blocks * stride
        n_blocks = max(1, math.ceil((frames_to_denoise - overlap) / stride)) if frames_to_denoise > 0 else 0
        if n_blocks == 0:
            # Nothing to denoise — just return seed
            output = torch.zeros(
                [batch_size, num_frames_original, num_channels],
                device=noise.device, dtype=noise.dtype,
            )
            expression_output = torch.zeros(
                [batch_size, num_frames_original, 100],
                device=noise.device, dtype=noise.dtype,
            )
            if seed_motion is not None:
                output[:, :seed_num_frames] = seed_motion.to(output.dtype)
            return output, expression_output

        frames_to_denoise_padded = overlap + n_blocks * stride
        denoise_pad_length = frames_to_denoise_padded - frames_to_denoise
        num_denoise_blocks = n_blocks

        # Pad noise to stride-aligned length
        if denoise_pad_length > 0:
            noise = self._pad_noise_frames(noise, denoise_pad_length)

        num_frames_padded = noise.shape[1]

        # ---- Output buffers ----
        output = torch.zeros(
            [batch_size, num_frames_padded, num_channels],
            device=noise.device, dtype=noise.dtype,
        )
        expression_output = torch.zeros(
            [batch_size, num_frames_padded, 100],
            device=noise.device, dtype=noise.dtype,
        )

        # ---- Context preparation (same as parent) ----
        if seed_motion is not None:
            total_context_frames = seed_prefill_frames + frames_to_denoise_padded
        else:
            total_context_frames = num_frames_padded

        expression_context = None
        if self.config.train.task.startswith("audio"):
            context, expression_context = self._prepare_audio_conditions(
                context,
                context_pid,
                source_frames=num_frames_original,
                target_frames=total_context_frames,
                precomputed_expression=precomputed_expression,
            )

        # ---- Profiling (same as parent) ----
        if profile:
            init_start = torch.cuda.Event(enable_timing=True)
            init_end = torch.cuda.Event(enable_timing=True)
            diffusion_start = torch.cuda.Event(enable_timing=True)
            diffusion_end = torch.cuda.Event(enable_timing=True)
            block_times = []
            block_start_ev = torch.cuda.Event(enable_timing=True)
            block_end_ev = torch.cuda.Event(enable_timing=True)
            init_start.record()

        # ---- Initialize KV cache ----
        if self.kv_cache1 is None:
            self._initialize_kv_cache(batch_size=batch_size, dtype=cache_dtype, device=noise.device)
        else:
            self._reset_kv_cache()

        if profile:
            init_end.record()
            diffusion_start.record()

        # ---- Seed prefill ----
        if seed_motion is not None:
            total_seq_len = seed_prefill_frames + frames_to_denoise_padded
            expression_seed, _ = self._seed_prefill(
                generator=generator,
                seed_motion=seed_prefill_motion,
                context=context,
                speaker_id=context_pid,
                num_channels=num_channels,
                total_seq_len=total_seq_len,
                cache_dtype=cache_dtype,
                expression_context=expression_context,
            )
            output[:, :seed_num_frames] = seed_motion.to(output.dtype)
            if seed_pad_to_block and seed_prefill_frames > seed_num_frames:
                expression_output[:, :seed_num_frames] = \
                    expression_seed[:, :seed_num_frames].to(expression_output.dtype)
            else:
                expression_output[:, :seed_num_frames] = expression_seed.to(expression_output.dtype)

            kv_cache_offset = seed_num_frames
            context_offset = seed_num_frames
        else:
            total_seq_len = num_frames_padded
            kv_cache_offset = 0
            context_offset = 0

        # ---- Overlap decode loop ----
        pending_motion = None       # (B, O, C)
        pending_expression = None   # (B, O, 100)
        cache_cursor = kv_cache_offset

        for block_idx in range(num_denoise_blocks):
            if profile:
                block_start_ev.record()

            # Frame positions (stride-based)
            noise_start = seed_num_frames + block_idx * stride
            noise_end = noise_start + num_frame_per_block
            noisy_input = noise[:, noise_start:noise_end]

            # Context & expression
            ctx_start = context_offset + block_idx * stride
            ctx_end = ctx_start + num_frame_per_block
            context_ = self.context_encoding(context, ctx_start, ctx_end, num_frame_per_block)
            if expression_context is None:
                expression_ = self.speech_model(context_, pid=context_pid)
            else:
                expression_ = expression_context[:, ctx_start:ctx_end]
                if expression_.shape[1] != num_frame_per_block:
                    raise RuntimeError(
                        "expression cache is shorter than an overlap block"
                    )

            # Sample (no prefill)
            denoised = self._sample_block_only(
                generator,
                noisy_input,
                context_,
                expression_,
                context_pid,
                current_start_frame=cache_cursor,
                seq_len=total_seq_len,
                cache_dtype=cache_dtype,
                normalize_rot6d=True,
            )

            # ---- Blend & commit ----
            if block_idx == 0:
                # First block: commit [0, S), hold [S, B) as pending
                commit_motion = denoised[:, :stride]
                commit_expr = expression_[:, :stride]
                commit_ctx = context_[:, :stride]

                self._commit_segment(
                    generator, output, expression_output,
                    commit_motion, commit_expr, commit_ctx, context_pid,
                    global_start_frame=cache_cursor,
                    seq_len=total_seq_len,
                    cache_dtype=cache_dtype,
                )
                cache_cursor += stride

                pending_motion = denoised[:, stride:]
                pending_expression = expression_[:, stride:]
            else:
                # Blend overlap region
                curr_head_motion = denoised[:, :overlap]
                curr_head_expr = expression_[:, :overlap]
                blended_motion = self._blend_overlap(pending_motion, curr_head_motion)
                blended_expr = self._blend_overlap_expression(pending_expression, curr_head_expr)

                # Middle stable region: [O, S)
                middle_motion = denoised[:, overlap:stride]
                middle_expr = expression_[:, overlap:stride]

                # Commit: blended + middle = S frames
                commit_motion = torch.cat([blended_motion, middle_motion], dim=1)
                commit_expr = torch.cat([blended_expr, middle_expr], dim=1)
                commit_ctx = context_[:, :stride]

                self._commit_segment(
                    generator, output, expression_output,
                    commit_motion, commit_expr, commit_ctx, context_pid,
                    global_start_frame=cache_cursor,
                    seq_len=total_seq_len,
                    cache_dtype=cache_dtype,
                )
                cache_cursor += stride

                # New pending = current block's tail
                pending_motion = denoised[:, stride:]
                pending_expression = expression_[:, stride:]

            if profile:
                block_end_ev.record()
                torch.cuda.synchronize()
                block_times.append(block_start_ev.elapsed_time(block_end_ev))

        # ---- Final flush: commit pending tail ----
        if pending_motion is not None and pending_motion.shape[1] > 0:
            final_ctx_start = context_offset + num_denoise_blocks * stride
            final_ctx_end = final_ctx_start + overlap
            final_ctx = self.context_encoding(context, final_ctx_start, final_ctx_end, overlap)

            self._commit_segment(
                generator, output, expression_output,
                pending_motion, pending_expression, final_ctx, context_pid,
                global_start_frame=cache_cursor,
                seq_len=total_seq_len,
                cache_dtype=cache_dtype,
            )

        # ---- Profiling report ----
        if profile:
            diffusion_end.record()
            torch.cuda.synchronize()
            diffusion_time = diffusion_start.elapsed_time(diffusion_end)
            init_time = init_start.elapsed_time(init_end)
            torch.cuda.synchronize()
            total_time = init_time + diffusion_time

            info = "Profiling results (overlap decode):"
            info += f"  - Initialization/caching time: {init_time:.2f} ms ({100 * init_time / total_time:.2f}%)"
            info += f"  - Diffusion generation time: {diffusion_time:.2f} ms ({100 * diffusion_time / total_time:.2f}%)"
            for i, bt in enumerate(block_times):
                info += f"    - Block {i} generation time: {bt:.2f} ms ({100 * bt / diffusion_time:.2f}% of diffusion)"
            info += f"  - Total time: {total_time:.2f} ms"
            info += f"  - Overlap config: O={overlap}, S={stride}, blocks={num_denoise_blocks}, blend={self.blend_mode}"
            self.logger.info(info)

        # ---- Strip padding, return original length ----
        if denoise_pad_length > 0:
            return output[:, :num_frames_original], expression_output[:, :num_frames_original]
        return output, expression_output
