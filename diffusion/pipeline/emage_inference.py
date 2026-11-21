"""
    Code modified from self-forcing: https://github.com/guandeh17/Self-Forcing/blob/main/pipeline/causal_inference.py
"""
import math
import torch
from typing import Optional
from diffusion.utils import rotation_conversions as rc
from diffusion.pipeline.quintic_bridge import apply_quintic_hermite_bridge
import torch.nn.functional as F
import numpy as np
import os.path as osp
import os

class CausalEmageInferencePipeline:
    def __init__(
        self,
        config,
        context_encoder,
        speech_model,
        sampler,
        logger,
        device = "cuda",
    ):
        self.config = config
        self.kv_cache1 = None
        self.kv_cache2 = None
        self.crossattn_cache = None
        self.num_frame_per_block = self.config.train.num_frame_per_block
        self.frame_seq_length = 1 # 硬编码
        self.num_transformer_blocks = self.config.model.model.params.depth
        self.independent_first_frame = self.config.train.independent_first_frame # 一帧to多帧，类似i2v，兼容性的放在这里
        self.sampler = sampler
        self.logger = logger
        self.context_encoder = context_encoder
        self.pose_fps = self.config.data.pose_fps
        self.pose_length = self.config.data.pose_length
        self.speech_model = speech_model
        self.device = device
        self.audio_feature_scope = str(
            self.config.model.get("inference_audio_feature_scope", "block")
        ).lower()
        if self.audio_feature_scope not in {"block", "full_clip"}:
            raise ValueError(
                "model.inference_audio_feature_scope must be 'block' or "
                f"'full_clip', got {self.audio_feature_scope!r}"
            )
        bc = self.config.model.model.params.block_config.params
        self.local_attn_size = bc.get("local_attn_size", self.config.model.get("local_attn_size", -1))
        self.max_attention_size = bc.get("max_attention_size", self.config.model.get("max_attention_size", 256))
        if self.local_attn_size == -1:
            assert self.max_attention_size >= self.num_frame_per_block, \
                f"max_attention_size too small ({self.max_attention_size}) for block {self.num_frame_per_block}"
        
        # Determine cross-attention type from config (default to "crossattn" for backward compatibility)
        # Supported: crossattn / deltanet / local8
        self.cross_attn_type = self.config.model.get("cross_attn_type", "crossattn")
        
        # DeltaNet-specific config (read from block_config if available)
        if self.cross_attn_type == "deltanet":
            self.deltanet_use_short_conv = bc.get("deltanet_use_short_conv", False)

    def _shuffle_boundary_noise(
        self,
        noise: torch.Tensor,
        seed_num_frames: int,
        num_denoise_blocks: int,
        window_size: int,
    ) -> None:
        """BlockVid-style local shuffle on block boundaries (in-place)."""
        if window_size <= 0 or num_denoise_blocks <= 1:
            return
        s = min(window_size, self.num_frame_per_block)
        for bidx in range(num_denoise_blocks - 1):
            left_end = seed_num_frames + (bidx + 1) * self.num_frame_per_block
            left_start = left_end - s
            right_start = left_end
            right_end = right_start + s
            left_noise = noise[:, left_start:left_end]
            right_noise = noise[:, right_start:right_end]
            perm_left = torch.randperm(s, device=noise.device)
            perm_right = torch.randperm(s, device=noise.device)
            noise[:, left_start:left_end] = left_noise[:, perm_left]
            noise[:, right_start:right_end] = right_noise[:, perm_right]

    @staticmethod
    def _pad_noise_frames(
        noise: torch.Tensor,
        pad_length: int,
    ) -> torch.Tensor:
        """Right-pad time without reflect-padding failures on short clips."""

        if pad_length <= 0:
            return noise
        if noise.shape[1] <= 0:
            raise ValueError("cannot pad an empty motion sequence")
        # torch reflect padding requires pad_length < input_length.  Official
        # TalkSHOW clips do not hit this branch, but using replicate makes the
        # variable-length inference contract safe for shorter external clips.
        mode = "reflect" if pad_length < noise.shape[1] else "replicate"
        return F.pad(noise, (0, 0, 0, pad_length), mode=mode)


    def __repr__(self) -> str:
        return f"CausalInferencePipeline(local_attn_size={self.local_attn_size}, num_frame_per_block={self.num_frame_per_block}, independent_first_frame={self.independent_first_frame}, frame_seq_length={self.frame_seq_length})"
    
    def __str__(self) -> str:
        return f"CausalInferencePipeline(local_attn_size={self.local_attn_size}, num_frame_per_block={self.num_frame_per_block}, independent_first_frame={self.independent_first_frame}, frame_seq_length={self.frame_seq_length})"
    
    def visualize_sequence(
        self,
        motion_list,
        rec_exp,
        tar_beta,
        step,
        choice="test",
        global_step = None, 
        epoch = None,
        vis_type="intermediate",
    ):
        # TODO: visualize the intermediate or model output
        # 现在是只准备支持可视化一下obj和png，后续再补充，主要是怕太占用空间，也占计算速度
        vis_bs = 1
        prefix = f"gs{global_step:06d}_" if global_step is not None else ""
        prefix += f"epoch{epoch:06d}_" if epoch is not None else ""
        prefix += f"step{step:06d}_" if step is not None else ""

        npz_save_name = "debug_npz"
        os.makedirs(osp.join(self.config.work_dir, npz_save_name, choice), exist_ok=True)
        
        for idx, motion in enumerate(motion_list):
            poses = motion[...,:330]
            trans = motion[...,330:333]
            poses = rc.rotation_6d_to_matrix(poses.reshape(-1, 55, 6))
            poses = rc.global_to_local_batch(poses)
            poses = rc.matrix_to_rotation_6d(poses).reshape(motion.shape[0], -1, 55 * 6)
            poses = poses[0].unsqueeze(0) # 只可视化第一个数据
            
            motion_results_axis_angle = rc.rotation_6d_to_axis_angle(
                poses.view(vis_bs, poses.shape[1], -1, 6))\
                    .view(vis_bs, -1, 3 * 55)

            for i in range(vis_bs):
                rec_pose_np = motion_results_axis_angle[i].detach().cpu().numpy()
                rec_trans_np = trans[i].detach().cpu().numpy()
                tar_beta_np = tar_beta[i].detach().cpu().numpy()
                rec_exp_np = rec_exp[i].detach().cpu().numpy()

                
                np.savez(
                    osp.join(self.config.work_dir, npz_save_name, choice, 
                            f"{prefix}bs{i}_{vis_type}{idx}" +'.npz'),
                    betas=tar_beta_np[0],
                    poses=rec_pose_np,
                    expressions=rec_exp_np,
                    trans=rec_trans_np,
                    model='smplx2020',
                    gender='neutral',
                    mocap_frame_rate = 30,
                )
    
    def _reset_kv_cache(self):
        """Reset KV cache indices and clear cache contents.
        
        Supports crossattn / deltanet / local8 mechanisms.
        """
        for block_index in range(len(self.kv_cache1)):
            # Reset index tensors
            if "global_end_index" in self.kv_cache1[block_index]:
                self.kv_cache1[block_index]["global_end_index"].fill_(0)
            if "local_end_index" in self.kv_cache1[block_index]:
                self.kv_cache1[block_index]["local_end_index"].fill_(0)
            if "global_end_index" in self.kv_cache2[block_index]:
                self.kv_cache2[block_index]["global_end_index"].fill_(0)
            if "local_end_index" in self.kv_cache2[block_index]:
                self.kv_cache2[block_index]["local_end_index"].fill_(0)
            
            if self.cross_attn_type == "crossattn":
                # Clear K/V cache tensors
                if "k" in self.kv_cache1[block_index]:
                    self.kv_cache1[block_index]["k"].zero_()
                if "v" in self.kv_cache1[block_index]:
                    self.kv_cache1[block_index]["v"].zero_()
                if "k" in self.kv_cache2[block_index]:
                    self.kv_cache2[block_index]["k"].zero_()
                if "v" in self.kv_cache2[block_index]:
                    self.kv_cache2[block_index]["v"].zero_()
                # local8 optional cache entries may exist in crossattn-compatible cache dict
                if "local8_k" in self.kv_cache1[block_index]:
                    self.kv_cache1[block_index]["local8_k"] = None
                if "local8_v" in self.kv_cache1[block_index]:
                    self.kv_cache1[block_index]["local8_v"] = None
                if "local8_k" in self.kv_cache2[block_index]:
                    self.kv_cache2[block_index]["local8_k"] = None
                if "local8_v" in self.kv_cache2[block_index]:
                    self.kv_cache2[block_index]["local8_v"] = None
            elif self.cross_attn_type == "deltanet":
                # Clear deltanet_state
                self.kv_cache1[block_index]["deltanet_state"].zero_()
                self.kv_cache2[block_index]["deltanet_state"].zero_()
                # Reset shortconv caches to None (ShortConvolution handles None internally)
                if self.deltanet_use_short_conv:
                    self.kv_cache1[block_index]["k_conv_cache"] = None
                    self.kv_cache1[block_index]["v_conv_cache"] = None
                    self.kv_cache2[block_index]["k_conv_cache"] = None
                    self.kv_cache2[block_index]["v_conv_cache"] = None
            elif self.cross_attn_type == "local8":
                # Local8 caches are dynamic short-context tensors and should be reset to None
                self.kv_cache1[block_index]["local8_k"] = None
                self.kv_cache1[block_index]["local8_v"] = None
                self.kv_cache2[block_index]["local8_k"] = None
                self.kv_cache2[block_index]["local8_v"] = None

    def visualize_sequence(
        self,
        motion_list,
        rec_exp,
        tar_beta,
        step,
        choice="test",
        global_step = None, 
        epoch = None,
        vis_type="intermediate",
    ):
        # TODO: visualize the intermediate or model output
        # 现在是只准备支持可视化一下obj和png，后续再补充，主要是怕太占用空间，也占计算速度
        vis_bs = 1
        prefix = f"gs{global_step:06d}_" if global_step is not None else ""
        prefix += f"epoch{epoch:06d}_" if epoch is not None else ""
        prefix += f"step{step:06d}_" if step is not None else ""

        npz_save_name = "debug_npz"
        os.makedirs(osp.join(self.config.work_dir, npz_save_name, choice), exist_ok=True)
        
        for idx, motion in enumerate(motion_list):
            trans = motion[:,:,330:333]
            motion = motion[:,:,:330]
            motion_matrix = rc.rotation_6d_to_matrix(motion.reshape(-1, 55, 6))
            motion_matrix = rc.global_to_local_batch(motion_matrix)
            motion = rc.matrix_to_rotation_6d(motion_matrix).reshape(motion.shape[0], -1, 55 * 6)
            motion = motion[0].unsqueeze(0) # 只可视化第一个数据
            
            
            motion_results_axis_angle = rc.rotation_6d_to_axis_angle(
                motion.view(vis_bs, motion.shape[1], -1, 6))\
                    .view(vis_bs, -1, 3 * 55)

            for i in range(vis_bs):
                rec_pose_np = motion_results_axis_angle[i].detach().cpu().numpy()
                rec_trans_np = trans[i].detach().cpu().numpy()
                tar_beta_np = tar_beta[i].detach().cpu().numpy()
                rec_exp_np = rec_exp[i].detach().cpu().numpy()

                
                np.savez(
                    osp.join(self.config.work_dir, npz_save_name, choice, 
                            f"{prefix}bs{i}_{vis_type}{idx}" +'.npz'),
                    betas=tar_beta_np[0],
                    poses=rec_pose_np,
                    expressions=rec_exp_np,
                    trans=rec_trans_np,
                    model='smplx2020',
                    gender='neutral',
                    mocap_frame_rate = 30,
                )

    def _prefill_kv_cache(
        self,
        generator,
        clean_motion: torch.Tensor,      # (B, T, C) 需要写入 cache 的“最终 clean chunk”
        context_: torch.Tensor,          # (B, T, Dctx) 传入但 prefill 分支不会用，保留接口一致性
        expression_: torch.Tensor,       # (B, T, 100) 同上
        speaker_id: torch.Tensor,
        current_start_frame: int,
        seq_len: int,
    ):
        B, T, _ = clean_motion.shape
        dummy_t = torch.zeros([B, T], device=clean_motion.device, dtype=torch.long)

        _ = generator(
            clean_motion,                           # motion 参数在 prefill 分支不会被用到；传 clean_motion 最省心
            dummy_t,
            context=context_,
            speaker_id=speaker_id,
            expression=expression_,
            kv_cache={
                "hand_kv_cache": self.kv_cache1,
                "body_kv_cache": self.kv_cache2,
            },
            current_start=current_start_frame * self.frame_seq_length,
            seq_len=seq_len,
            prefill_only=True,
            update_kv_cache=True,
            clean_motion=clean_motion,
        )

    def _sample_block_with_cache(
        self,
        generator,
        noisy_block: torch.Tensor,
        context_: torch.Tensor,
        expression_: torch.Tensor,
        speaker_id: torch.Tensor,
        current_start_frame: int,
        seq_len: int,
        cache_dtype: torch.dtype | None = None,
        normalize_rot6d: bool = True,
        return_intermediate: bool = True,
        return_model_output: bool = True,
        post_process_func=None,
        start_sigma: float | None = None,
    ):
        if cache_dtype is None:
            cache_dtype = noisy_block.dtype
        denoised_pred, extra = self.sampler.sample(
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
            return_intermediate=return_intermediate,
            return_model_output=return_model_output,
            post_process_func=post_process_func,
            start_sigma=start_sigma,
        )
        if normalize_rot6d:
            denoised_pred = torch.cat([
                rc.NormalizeRot6d(denoised_pred[:, :, :-3]),
                denoised_pred[:, :, -3:],
            ], dim=-1)
        self._prefill_kv_cache(
            generator,
            clean_motion=denoised_pred.to(cache_dtype),
            context_=context_,
            expression_=expression_,
            speaker_id=speaker_id,
            current_start_frame=current_start_frame,
            seq_len=seq_len,
        )
        return denoised_pred, extra
    
    def _encode_audio_waveform(
        self,
        context: torch.Tensor,
        align_length: int,
    ) -> torch.Tensor:
        """Encode one complete waveform tensor and align it to motion frames."""

        if align_length <= 0:
            raise ValueError("audio alignment length must be positive")
        if self.context_encoder is None:
            raise RuntimeError("audio context encoder is not configured")

        sr = self.config.data.audio_sr
        audio_encoder_config = self.config.audio.audio_encoder
        audio_encoder_name = audio_encoder_config["name"]
        if audio_encoder_name not in {"hubert-large-ls960-ft", "wavlm"}:
            raise ValueError(f"Invalid audio encoder: {audio_encoder_name}")

        dtype = context.dtype
        processed = self.context_encoder[0](
            [waveform.cpu().numpy() for waveform in context],
            sampling_rate=sr,
            padding=True,
            return_tensors="pt",
        )
        context_mask = processed.attention_mask.to(self.device)
        input_values = processed.input_values.to(self.device)
        if not audio_encoder_config.get("quantization", False):
            encoded = self.context_encoder[1](
                input_values,
                attention_mask=context_mask,
            )
        else:
            encoded = self.context_encoder[1](
                input_values.to(torch.float16),
                attention_mask=context_mask.to(torch.float16),
            )
        hidden = encoded.last_hidden_state.to(dtype)
        return F.interpolate(
            hidden.permute(0, 2, 1),
            size=align_length,
            mode="linear",
            align_corners=True,
        ).permute(0, 2, 1)

    def _prepare_audio_context(
        self,
        context: torch.Tensor,
        *,
        source_frames: int,
        target_frames: int,
    ) -> torch.Tensor:
        """Prepare blockwise raw audio or a full-clip feature cache.

        ``block`` exactly preserves the legacy BEAT behavior: waveform padding
        happens before each block is encoded by :meth:`context_encoding`.
        ``full_clip`` is opt-in and used by the TalkSHOW config so inference
        matches the full-utterance WavLM scope used to train its frozen face
        network and to build its LMDB.
        """

        if source_frames <= 0 or target_frames < source_frames:
            raise ValueError(
                "invalid audio frame lengths: "
                f"source={source_frames}, target={target_frames}"
            )
        sr = self.config.data.audio_sr
        if self.audio_feature_scope == "block":
            required_samples = int(target_frames * sr / self.pose_fps)
            if context.shape[1] > required_samples:
                return context[:, :required_samples]
            return F.pad(
                context,
                (0, required_samples - context.shape[1], 0, 0),
                mode="constant",
                value=0.0,
            )

        if context.ndim == 3:
            if context.shape[1] < source_frames:
                raise ValueError(
                    "precomputed full_clip audio features are shorter than "
                    f"source_frames: {context.shape[1]} < {source_frames}"
                )
            features = context[:, :source_frames]
            if target_frames > source_frames:
                features = F.pad(
                    features,
                    (0, 0, 0, target_frames - source_frames),
                    mode="constant",
                    value=0.0,
                )
            return features
        if context.ndim != 2:
            raise ValueError(
                "full_clip audio context must be raw [B,S] waveform or "
                "precomputed [B,T,C] features"
            )

        required_samples = int(source_frames * sr / self.pose_fps)
        if context.shape[1] > required_samples:
            source_audio = context[:, :required_samples]
        else:
            source_audio = F.pad(
                context,
                (0, required_samples - context.shape[1], 0, 0),
                mode="constant",
                value=0.0,
            )
        features = self._encode_audio_waveform(source_audio, source_frames)
        if target_frames > source_frames:
            features = F.pad(
                features,
                (0, 0, 0, target_frames - source_frames),
                mode="constant",
                value=0.0,
            )
        return features

    def _prepare_audio_conditions(
        self,
        context: torch.Tensor,
        context_pid: torch.Tensor,
        *,
        source_frames: int,
        target_frames: int,
        precomputed_expression: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Prepare audio context and an optional full-clip expression cache.

        TalkSHOW's face head contains temporal convolutions, so running that
        head independently on diffusion blocks would introduce artificial
        zero-padding boundaries.  In ``full_clip`` mode both WavLM and the
        face head run once on the real source sequence.  Only then are their
        outputs right-padded for the diffusion block schedule.
        """

        prepared_context = self._prepare_audio_context(
            context,
            source_frames=source_frames,
            target_frames=target_frames,
        )
        if self.audio_feature_scope != "full_clip":
            if precomputed_expression is not None:
                raise ValueError(
                    "precomputed_expression requires full_clip audio scope"
                )
            return prepared_context, None
        if prepared_context.ndim != 3:
            raise RuntimeError(
                "full_clip audio preparation must produce [B,T,C] features"
            )

        source_context = prepared_context[:, :source_frames]
        if precomputed_expression is None:
            expression = self.speech_model(source_context, pid=context_pid)
        else:
            expression = precomputed_expression
        if (
            expression.ndim != 3
            or expression.shape[0] != source_context.shape[0]
            or expression.shape[1] != source_frames
        ):
            raise RuntimeError(
                "full_clip speech model must return [B,source_frames,C], "
                f"got {tuple(expression.shape)}"
            )
        if target_frames > source_frames:
            expression = F.pad(
                expression,
                (0, 0, 0, target_frames - source_frames),
                mode="constant",
                value=0.0,
            )
        return prepared_context, expression

    def context_encoding(self, context, start, end, align_length):
        if self.config.train.task.startswith("audio"):
            if context.ndim == 3:
                context_ = context[:, start:end]
                if context_.shape[1] != align_length:
                    context_ = F.interpolate(
                        context_.permute(0, 2, 1),
                        size=align_length,
                        mode="linear",
                        align_corners=True,
                    ).permute(0, 2, 1)
                return context_

            sr = self.config.data.audio_sr
            audio_start, audio_end = int(start * sr / self.pose_fps), int(end * sr / self.pose_fps)
            context_ = context[:, audio_start:audio_end]
            context_ = self._encode_audio_waveform(context_, align_length)
        elif self.config.train.task.startswith("text"):
            # TODO: 解析txt
            context_ = context[:, start:end]
        else:
            raise ValueError(f"Invalid task: {self.config.train.task}")
        return context_
    
    def _seed_prefill(
        self,
        generator,
        seed_motion: torch.Tensor,       # (B, N, C) seed motion (可能右侧有 padding)
        context: torch.Tensor,           # full audio context
        speaker_id: torch.Tensor,
        num_channels: int,
        total_seq_len: int,              # total sequence length for RoPE
        cache_dtype: torch.dtype,
        expression_context: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, int]:
        """
        Prefill KV cache with seed motion without padding.
        Splits seed into blocks; the last block can be shorter.

        注意：seed_motion 可能包含右侧的 padding，实际长度由其 shape[1] 确定
        """
        batch_size = seed_motion.shape[0]
        seed_prefill_frames = seed_motion.shape[1]  # 包含 padding 的总长度
        if seed_prefill_frames == 0:
            empty_expr = torch.zeros([batch_size, 0, 100], device=seed_motion.device, dtype=cache_dtype)
            return empty_expr, 0

        seed_block = seed_motion.to(cache_dtype)
        num_seed_blocks = math.ceil(seed_prefill_frames / self.num_frame_per_block)
        expression_seed_blocks = []

        # Prefill per block (last block can be shorter).
        for block_idx in range(num_seed_blocks):
            block_start = block_idx * self.num_frame_per_block
            block_end = min(block_start + self.num_frame_per_block, seed_prefill_frames)
            current_len = block_end - block_start
            if current_len <= 0:
                break

            current_seed_block = seed_block[:, block_start:block_end]
            current_context = self.context_encoding(
                context,
                start=block_start,
                end=block_end,
                align_length=current_len,
            )
            if expression_context is None:
                current_expression = self.speech_model(
                    current_context,
                    pid=speaker_id,
                )
            else:
                current_expression = expression_context[
                    :, block_start:block_end
                ]
                if current_expression.shape[1] != current_len:
                    raise RuntimeError(
                        "expression cache is shorter than the seed context"
                    )
            expression_seed_blocks.append(current_expression)

            self._prefill_kv_cache(
                generator,
                clean_motion=current_seed_block,
                context_=current_context,
                expression_=current_expression,
                speaker_id=speaker_id,
                current_start_frame=block_start,
                seq_len=total_seq_len,
            )

        expression_seed = torch.cat(expression_seed_blocks, dim=1)
        return expression_seed, seed_prefill_frames

    def inference_with_seed(
        self,
        generator,
        noise: torch.Tensor,
        context: torch.Tensor,
        context_pid: torch.Tensor,
        seed_motion: torch.Tensor = None,  # (B, N, C) 可选的 seed motion
        seed_pad_to_block: bool = False,
        profile: bool = False,
        noise_shuffle: bool = False,
        noise_shuffle_window: int = 4,
        quintic_bridge: bool = False,
        bridge_frames: int = 8,
        bridge_interp_space: str = "rot6d",
        **kwargs,
    ):
        """
        Autoregressive block diffusion with optional seed prefill.
        Supports both padding and no-padding modes for seed prefill.

        Args:
            generator: 扩散模型
            noise: (B, T, C) 初始噪声
            context: audio context
            context_pid: speaker id
            seed_motion: (B, N, C) 可选的 seed motion，N 可以是任意长度
                         如果提供，output[0:N] = seed，去噪从 noise[N:] 开始
            seed_pad_to_block: True 为 padding 模式，seed 左侧补零到 block 倍数后再 prefill
                               False 为 no-padding 模式，直接用原始 seed 长度 prefill
            profile: 是否开启性能分析
            noise_shuffle: 是否开启 BlockVid 式边界噪声 shuffle（仅推理）
            noise_shuffle_window: shuffle 窗口大小 s，通常为 2~8
            quintic_bridge: 是否启用 quintic Hermite bridge（硬 C² 连续性保证）
            bridge_frames: bridge 替换的帧数 m（推荐 8-12）
            bridge_interp_space: 插值空间 ("rot6d" 或 "axis_angle")
        
        Returns:
            output: (B, T, C) 生成的 motion
            expression_output: (B, T, 100) 生成的 expression
        """
        batch_size, num_frames_original, num_channels = noise.shape
        cache_dtype = kwargs.get("cache_dtype", noise.dtype)
        precomputed_expression = kwargs.get(
            "precomputed_expression_context",
            None,
        )
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
                    seed_prefill_frames = math.ceil(seed_num_frames / self.num_frame_per_block) * self.num_frame_per_block
                    padding_frames = seed_prefill_frames - seed_num_frames
                    if padding_frames > 0:
                        padding = torch.zeros(
                            [batch_size, padding_frames, num_channels],
                            device=seed_motion.device,
                            dtype=seed_motion.dtype,
                        )
                        # 修复：改为右侧 padding，而不是左侧
                        seed_prefill_motion = torch.cat([seed_motion, padding], dim=1)
                    else:
                        seed_prefill_motion = seed_motion
                else:
                    seed_prefill_frames = seed_num_frames
                    seed_prefill_motion = seed_motion


        frames_to_denoise = num_frames_original - seed_num_frames
        
        if frames_to_denoise % self.num_frame_per_block != 0:
            # 需要 padding 使去噪部分是 block 的整数倍
            denoise_pad_length = self.num_frame_per_block - (frames_to_denoise % self.num_frame_per_block)
        else:
            denoise_pad_length = 0
        
        # padding noise
        if denoise_pad_length > 0:
            noise = self._pad_noise_frames(noise, denoise_pad_length)
        
        num_frames_padded = noise.shape[1]  # padding 后的总帧数
        frames_to_denoise_padded = num_frames_padded - seed_num_frames  # padding 后的去噪帧数
        num_denoise_blocks = frames_to_denoise_padded // self.num_frame_per_block
        # BlockVid 风格噪声 shuffle（只对去噪段的 block 边界做局部置换）
        if noise_shuffle:
            self._shuffle_boundary_noise(
                noise=noise,
                seed_num_frames=seed_num_frames,
                num_denoise_blocks=num_denoise_blocks,
                window_size=noise_shuffle_window,
            )
        
        # 输出缓冲区
        output = torch.zeros(
            [batch_size, num_frames_padded, num_channels],
            device=noise.device,
            dtype=noise.dtype
        )
        expression_output = torch.zeros(
            [batch_size, num_frames_padded, 100],
            device=noise.device,
            dtype=noise.dtype
        )
        
        # 计算总的 context 长度（包含 seed block(s)）
        # context covers seed prefill length + denoise (padded) length.
        if seed_motion is not None:
            total_context_frames = seed_prefill_frames + frames_to_denoise_padded
        else:
            total_context_frames = num_frames_padded
        
        # TalkSHOW optionally encodes the original full clip once and slices
        # the aligned feature cache per diffusion block.  The default keeps
        # legacy blockwise BEAT encoding byte-for-byte equivalent.
        expression_context = None
        if self.config.train.task.startswith("audio"):
            context, expression_context = self._prepare_audio_conditions(
                context,
                context_pid,
                source_frames=num_frames_original,
                target_frames=total_context_frames,
                precomputed_expression=precomputed_expression,
            )
        
        if profile:
            init_start = torch.cuda.Event(enable_timing=True)
            init_end = torch.cuda.Event(enable_timing=True)
            diffusion_start = torch.cuda.Event(enable_timing=True)
            diffusion_end = torch.cuda.Event(enable_timing=True)
            block_times = []
            block_start = torch.cuda.Event(enable_timing=True)
            block_end = torch.cuda.Event(enable_timing=True)
            init_start.record()
        
        # 初始化 KV cache
        if self.kv_cache1 is None:
            self._initialize_kv_cache(batch_size=batch_size, dtype=cache_dtype, device=noise.device)
        else:
            self._reset_kv_cache()
        
        if profile:
            init_end.record()
            diffusion_start.record()
        
        # ------------------------------------------------------------
        # 0) Seed prefill (optional)
        # ------------------------------------------------------------
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

            # 修复：使用实际的 seed_num_frames 而不是 seed_prefill_frames
            # 这样可以确保 audio context 与 motion 正确对齐
            kv_cache_offset = seed_num_frames
            context_offset = seed_num_frames
        else:
            total_seq_len = num_frames_padded
            kv_cache_offset = 0
            context_offset = 0
        
        # ------------------------------------------------------------
        # 1) 去噪循环
        # ------------------------------------------------------------
        # 每个 block 都是 64 帧
        # - 第一个 block：noise[seed:seed+64] → output[seed:seed+64]
        # - 第二个 block：noise[seed+64:seed+128] → output[seed+64:seed+128]
        # - ...
        
        # Quintic bridge: initialize prev_tail from seed if available
        bridge_prev_tail = None
        if quintic_bridge:
            if seed_motion is not None and seed_num_frames >= 3:
                bridge_prev_tail = seed_motion[:, -3:]  # (B, 3, C)
        
        for block_idx in range(num_denoise_blocks):
            if profile:
                block_start.record()
            
            # 当前 block 在 output/noise 中的起始位置
            output_start = seed_num_frames + block_idx * self.num_frame_per_block
            output_end = output_start + self.num_frame_per_block
            
            # 取出当前 block 的 noisy input
            noisy_input = noise[:, output_start:output_end]
            
            # context encoding（需要加上 context_offset）
            context_start = context_offset + block_idx * self.num_frame_per_block
            context_end = context_start + self.num_frame_per_block
            context_ = self.context_encoding(
                context, context_start, context_end, self.num_frame_per_block
            )
            if expression_context is None:
                expression = self.speech_model(context_, pid=context_pid)
            else:
                expression = expression_context[:, context_start:context_end]
                if expression.shape[1] != self.num_frame_per_block:
                    raise RuntimeError(
                        "expression cache is shorter than a diffusion block"
                    )
            expression_output[:, output_start:output_end] = expression
            
            # 去噪
            # KV cache 中的位置 = kv_cache_offset + block_idx * num_frame_per_block
            kv_prefill_start = kv_cache_offset + block_idx * self.num_frame_per_block
            denoised_pred, extra = self._sample_block_with_cache(
                generator,
                noisy_input,
                context_,
                expression,
                context_pid,
                current_start_frame=kv_prefill_start,
                seq_len=total_seq_len,
                cache_dtype=cache_dtype,
                normalize_rot6d=True,
                return_intermediate=True,
                return_model_output=True,
            )
            
            # # 可视化（可选）
            # if "tar_beta" in kwargs:
            #     self.visualize_sequence(
            #         extra["motion_intermediates"],
            #         expression,
            #         kwargs["tar_beta"],
            #         0
            #     )
            #     self.visualize_sequence(
            #         extra["motion_model_outputs"],
            #         expression,
            #         kwargs["tar_beta"],
            #         0,
            #         vis_type="modelouts",
            #     )
            
            # Apply quintic Hermite bridge (hard C² continuity)
            if quintic_bridge and bridge_prev_tail is not None:
                denoised_pred = apply_quintic_hermite_bridge(
                    prev_frames=bridge_prev_tail,
                    current_block=denoised_pred,
                    bridge_frames=bridge_frames,
                    fps=self.pose_fps,
                    num_joints=55,
                    interp_space=bridge_interp_space,
                )
            
            # 写入 output
            output[:, output_start:output_end] = denoised_pred

            # Update bridge_prev_tail for next block's quintic bridge
            if quintic_bridge:
                bridge_prev_tail = denoised_pred[:, -3:]  # (B, 3, C)
            
            if profile:
                block_end.record()
                torch.cuda.synchronize()
                block_times.append(block_start.elapsed_time(block_end))
        
        if profile:
            diffusion_end.record()
            torch.cuda.synchronize()
            diffusion_time = diffusion_start.elapsed_time(diffusion_end)
            init_time = init_start.elapsed_time(init_end)
            torch.cuda.synchronize()
            total_time = init_time + diffusion_time
            
            info = "Profiling results:"
            info += f"  - Initialization/caching time: {init_time:.2f} ms ({100 * init_time / total_time:.2f}%)"
            info += f"  - Diffusion generation time: {diffusion_time:.2f} ms ({100 * diffusion_time / total_time:.2f}%)"
            for i, block_time in enumerate(block_times):
                info += f"    - Block {i} generation time: {block_time:.2f} ms ({100 * block_time / diffusion_time:.2f}% of diffusion)"
            info += f"  - Total time: {total_time:.2f} ms"
            self.logger.info(info)
        
        # 去掉 padding，返回原始长度
        if denoise_pad_length > 0:
            return output[:, :num_frames_original], expression_output[:, :num_frames_original]
        else:
            return output, expression_output

    # ============================================================
    # 旧的 inference 函数（已注释）
    # 如果不使用 seed prefill，可以取消注释使用此函数
    # ============================================================
    def inference(
        self,
        generator,
        noise: torch.Tensor,
        context: torch.Tensor,
        context_pid: torch.Tensor,
        profile: bool = False,
        quintic_bridge: bool = False,
        bridge_frames: int = 8,
        bridge_interp_space: str = "rot6d",
        **kwargs,
    ):
        """
        不带 seed prefill 的推理函数（保留原有功能）。
        如果需要使用 seed prefill，请使用 inference_with_seed 函数。

        Args:
            quintic_bridge: 是否启用 quintic Hermite bridge（硬 C² 连续性保证）
            bridge_frames: bridge 替换的帧数 m（推荐 8-12）
            bridge_interp_space: 插值空间 ("rot6d" 或 "axis_angle")
        """
        batch_size, num_frames, num_channels = noise.shape
        num_frames_original = num_frames
        precomputed_expression = kwargs.get(
            "precomputed_expression_context",
            None,
        )
        
        if num_frames % self.num_frame_per_block != 0:
            pad_length = self.num_frame_per_block - num_frames % self.num_frame_per_block
            noise = self._pad_noise_frames(noise, pad_length)
            num_frames = noise.shape[1]
        else:
            pad_length = 0
        num_output_frames = num_frames
        
        output = torch.zeros(
            [batch_size, num_output_frames, num_channels],
            device=noise.device,
            dtype=noise.dtype
        )
        expression_output = torch.zeros(
            [batch_size, num_output_frames, 100],
            device=noise.device,
            dtype=noise.dtype
        )
        
        expression_context = None
        if self.config.train.task.startswith("audio"):
            context, expression_context = self._prepare_audio_conditions(
                context,
                context_pid,
                source_frames=num_frames_original,
                target_frames=num_output_frames,
                precomputed_expression=precomputed_expression,
            )
        
        if profile:
            init_start = torch.cuda.Event(enable_timing=True)
            init_end = torch.cuda.Event(enable_timing=True)
            diffusion_start = torch.cuda.Event(enable_timing=True)
            diffusion_end = torch.cuda.Event(enable_timing=True)
            block_times = []
            block_start = torch.cuda.Event(enable_timing=True)
            block_end = torch.cuda.Event(enable_timing=True)
            init_start.record()

        cache_dtype = kwargs.get("cache_dtype", noise.dtype)
        if self.kv_cache1 is None:
            self._initialize_kv_cache(batch_size=batch_size, dtype=cache_dtype, device=noise.device)
        else:
            self._reset_kv_cache()

        if profile:
            init_end.record()
            diffusion_start.record()

        current_start_frame = 0   
        all_num_frames = []
        if num_output_frames > 0:
            assert num_output_frames % self.num_frame_per_block == 0, \
                f"internal error: num_output_frames={num_output_frames} not divisible by block={self.num_frame_per_block}"
            all_num_frames.extend([self.num_frame_per_block] * (num_output_frames // self.num_frame_per_block))
        
        # Quintic bridge: initialize prev_tail (no seed in this path)
        bridge_prev_tail = None

        for idx, current_num_frames in enumerate(all_num_frames):
            if profile:
                block_start.record()
            noisy_input = noise[
                :, current_start_frame:current_start_frame + current_num_frames]

            context_ = self.context_encoding(context,
                    current_start_frame, current_start_frame + current_num_frames, current_num_frames)
            if expression_context is None:
                expression = self.speech_model(context_, pid=context_pid)
            else:
                expression = expression_context[
                    :,
                    current_start_frame:
                    current_start_frame + current_num_frames,
                ]
                if expression.shape[1] != current_num_frames:
                    raise RuntimeError(
                        "expression cache is shorter than a diffusion block"
                    )
            expression_output[:, current_start_frame:current_start_frame + current_num_frames] = expression
            
            denoised_pred, extra = self.sampler.sample(
                generator,
                noisy_input,
                context=context_,
                speaker_id=context_pid,
                expression=expression,
                kv_cache={"hand_kv_cache": self.kv_cache1, "body_kv_cache": self.kv_cache2},
                current_start=current_start_frame * self.frame_seq_length,
                seq_len=num_output_frames,
                prefill_only=False,
                update_kv_cache=False,
                return_intermediate=True,
                return_model_output=True,
            )
            denoised_pred = torch.cat([
                rc.NormalizeRot6d(denoised_pred[:, :, :-3]),
                denoised_pred[:, :, -3:],
            ], dim=-1)
            
            if "tar_beta" in kwargs:
                self.visualize_sequence(
                    extra["motion_intermediates"],
                    expression,
                    kwargs["tar_beta"],
                    0
                )
                self.visualize_sequence(
                    extra["motion_model_outputs"],
                    expression,
                    kwargs["tar_beta"],
                    0,
                    vis_type="modelouts",
                )
            # Apply quintic Hermite bridge (hard C² continuity)
            if quintic_bridge and bridge_prev_tail is not None:
                denoised_pred = apply_quintic_hermite_bridge(
                    prev_frames=bridge_prev_tail,
                    current_block=denoised_pred,
                    bridge_frames=bridge_frames,
                    fps=self.pose_fps,
                    num_joints=55,
                    interp_space=bridge_interp_space,
                )
            
            output[:, current_start_frame:current_start_frame + current_num_frames] = denoised_pred
            
            self._prefill_kv_cache(
                generator,
                clean_motion=denoised_pred.to(cache_dtype),
                context_=context_,
                expression_=expression,
                speaker_id=context_pid,
                current_start_frame=current_start_frame,
                seq_len=num_output_frames,
            )

            # Update bridge_prev_tail for next block's quintic bridge
            if quintic_bridge:
                bridge_prev_tail = denoised_pred[:, -3:]  # (B, 3, C)

            if profile:
                block_end.record()
                torch.cuda.synchronize()
                block_times.append(block_start.elapsed_time(block_end))

            current_start_frame += current_num_frames

        if profile:
            diffusion_end.record()
            torch.cuda.synchronize()
            diffusion_time = diffusion_start.elapsed_time(diffusion_end)
            init_time = init_start.elapsed_time(init_end)
            torch.cuda.synchronize()
            total_time = init_time + diffusion_time

            info = "Profiling results:"
            info += f"  - Initialization/caching time: {init_time:.2f} ms ({100 * init_time / total_time:.2f}%)"
            info += f"  - Diffusion generation time: {diffusion_time:.2f} ms ({100 * diffusion_time / total_time:.2f}%)"
            for i, block_time in enumerate(block_times):
                info += f"    - Block {i} generation time: {block_time:.2f} ms ({100 * block_time / diffusion_time:.2f}% of diffusion)"
            info += f"  - Total time: {total_time:.2f} ms"
            self.logger.info(info)

        if pad_length > 0:
            return output[:, :-pad_length], expression_output[:, :-pad_length]
        else:
            return output, expression_output
    
    
    def _create_single_block_cache(self, batch_size, dtype, device, num_heads, head_dim, hidden_size, kv_cache_size=None):
        """Create a single block's KV cache based on cross_attn_type.
        
        Args:
            batch_size: Batch size
            dtype: Data type for tensors (used for crossattn K/V)
            device: Device to create tensors on
            num_heads: Number of attention heads
            head_dim: Dimension per head
            hidden_size: Hidden size (num_heads * head_dim)
            kv_cache_size: Size for K/V cache (only used for crossattn)
        
        Returns:
            dict: Cache dictionary for a single transformer block
        """
        if self.cross_attn_type == "crossattn":
            return {
                "k": torch.zeros([batch_size, kv_cache_size, num_heads, head_dim], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, kv_cache_size, num_heads, head_dim], dtype=dtype, device=device),
                "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                "local_end_index": torch.tensor([0], dtype=torch.long, device=device),
                # Optional Local8 cache entries for compatibility with Local8 cross-attn modules
                "local8_k": None,
                "local8_v": None,
            }
        elif self.cross_attn_type == "deltanet":
            cache_dict = {
                "deltanet_state": torch.zeros(
                    [batch_size, num_heads, head_dim, head_dim], 
                    dtype=torch.float32,  # Always fp32 for numerical stability
                    device=device
                ),
                "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                "local_end_index": torch.tensor([0], dtype=torch.long, device=device)
            }
            # Initialize shortconv caches as None (ShortConvolution handles None internally)
            if self.deltanet_use_short_conv:
                cache_dict["k_conv_cache"] = None
                cache_dict["v_conv_cache"] = None
            return cache_dict
        elif self.cross_attn_type == "local8":
            return {
                "local8_k": None,
                "local8_v": None,
                "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                "local_end_index": torch.tensor([0], dtype=torch.long, device=device),
            }
        else:
            raise ValueError(f"Unknown cross_attn_type: {self.cross_attn_type}")

    def _initialize_kv_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU KV cache.
        
        Supports both crossattn and deltanet mechanisms:
        - crossattn: K/V tensors with shape (B, kv_cache_size, num_heads, head_dim)
        - deltanet: deltanet_state tensor with shape (B, num_heads, head_dim, head_dim)
                    Shortconv caches are initialized as None (ShortConvolution handles this)
        """
        num_heads = self.config.model.model.params.block_config.params.num_heads
        head_dim = self.config.model.model.params.block_config.params.head_dim
        hidden_size = self.config.model.model.params.hidden_size
        if num_heads is None:
            num_heads = hidden_size // head_dim
        elif head_dim is None:
            head_dim = hidden_size // num_heads
        else:
            assert num_heads * head_dim == hidden_size, "num_heads * head_dim should be equal to hidden_size"
        
        # Compute kv_cache_size for crossattn (not used for deltanet)
        kv_cache_size = None
        if self.cross_attn_type == "crossattn":
            if self.local_attn_size != -1:
                kv_cache_size = self.local_attn_size * self.frame_seq_length
            else:
                kv_cache_size = self.max_attention_size
        
        # Create caches for all transformer blocks
        self.kv_cache1 = [
            self._create_single_block_cache(batch_size, dtype, device, num_heads, head_dim, hidden_size, kv_cache_size)
            for _ in range(self.num_transformer_blocks)
        ]
        self.kv_cache2 = [
            self._create_single_block_cache(batch_size, dtype, device, num_heads, head_dim, hidden_size, kv_cache_size)
            for _ in range(self.num_transformer_blocks)
        ]
