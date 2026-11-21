import torch
import math
from copy import deepcopy
from diffusion.trainer import EmageTrainer
from functools import partial
from diffusion.model.utils import mean_flat
from diffusion.model.respace import compute_density_for_timestep_sampling
import os.path as osp
import numpy as np
import soundfile as sf
from diffusion.utils import rotation_conversions as rc
from diffusion.pipeline.emage_inference import CausalEmageInferencePipeline
from termcolor import colored
from diffusion.metrics.gradient_conflict import (
    compute_block_motion_grad_debug_metrics,
    compute_interblock_motion_conflict_metrics,
    compute_simple_vs_rcxyz_conflict_metrics,
    compute_simple_vs_structural_conflict_metrics,
)
from diffusion.guider.guider import BLDCFGGuider

class BLDTrainer(EmageTrainer):
    """某种意义上的Block Diffusion，也算是一种teacher forcing"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.inference_pipeline = self.configure_pipeline()
        self._self_sampling_warmup_logged = False
        self._weak_mono_warmup_logged = False
        self._local_jitter_timestep_logged = False
        self.self_sampling_ema_model = None
        self._self_sampling_ema_last_step = None
        # 调试缓存：仅保留一份带计算图的数据，供 per-block loss/grad 联合分析
        self._debug_cache = None
    
    
    # def configure_audio_encoder(self):
    #     # 这个audio_encoder可以是一个复杂的深度学习模型提取特征，也可以就弄个mel谱
    #     # 我们会首先尝试从audio_encoder的配置中来获取，如果获取不到，我们就用mel谱
    #     if not self.config.train.task.startswith("audio2"):
    #         return None

    #     if not self.load_audio_feat or self.config._test or self.config._predict:    
    #         audio_encoder_config = getattr(self.config.audio, "audio_encoder", None)
    #         if audio_encoder_config is not None:
    #             audio_encoder_name = audio_encoder_config["name"]
    #             if audio_encoder_name == "hubert-large-ls960-ft": #
    #                 from modelscope import AutoProcessor, HubertModel, BitsAndBytesConfig # transformers库发癫一样，加载不了这个模型
    #                 local_model_path = audio_encoder_config.get("local_model_path", None)
    #                 is_quantization = self.config.audio.audio_encoder.get("quantization", False)
    #                 if is_quantization:
    #                     bnb_config = BitsAndBytesConfig(
    #                         load_in_8bit=True,
    #                         llm_int8_threshold=6.0
    #                     )
    #                 if local_model_path is not None:
    #                     processor = AutoProcessor.from_pretrained(local_model_path)
    #                     if is_quantization:
    #                         model = HubertModel.from_pretrained(
    #                             local_model_path, quantization_config=bnb_config, torch_dtype=torch.float16,
    #                             device_map=self.device, attn_implementation="sdpa")
    #                         model = model.eval()#.to(self.device)
    #                     else:
    #                         model = HubertModel.from_pretrained(local_model_path)
    #                         model = model.requires_grad_(False).eval().to(self.device)
    #                 else:
    #                     processor = AutoProcessor.from_pretrained("facebook/hubert-large-ls960-ft")
    #                     if is_quantization:
    #                         model = HubertModel.from_pretrained(
    #                         "facebook/hubert-large-ls960-ft", quantization_config=bnb_config, torch_dtype=torch.float16,
    #                         device_map=self.device, attn_implementation="sdpa")
    #                         model = model.eval()#.to(self.device)
    #                     else:
    #                         model = HubertModel.from_pretrained("facebook/hubert-large-ls960-ft")
    #                         model = model.eval().to(self.device)
    #                     model = model.eval().to(self.device)
    #                 audio_encoder = [processor, model]
    #                 params = sum(p.numel() for p in audio_encoder[1].parameters()) / 1e6
    #             elif audio_encoder_name == "wavlm":
    #                 from transformers import Wav2Vec2Processor, WavLMModel
    #                 local_model_path = audio_encoder_config.get("local_model_path", None)
    #                 if local_model_path is not None:
    #                     processor = Wav2Vec2Processor.from_pretrained(local_model_path)
    #                     model = WavLMModel.from_pretrained(local_model_path)
    #                 else:
    #                     processor = Wav2Vec2Processor.from_pretrained("facebook/wavlm-libri-clean-100h-large")
    #                     model = WavLMModel.from_pretrained("facebook/wavlm-libri-clean-100h-large")
    #                 model = model.requires_grad_(False).eval().to(self.device)
    #                 audio_encoder = [processor, model]
    #                 params = sum(p.numel() for p in audio_encoder[1].parameters()) / 1e6
    #             else:
    #                 raise ValueError(f"Invalid audio encoder: {audio_encoder_name}")
    #         else: # 简单的mel谱处理
    #             audio_encoder = MelSpec(**self.config.audio.mel_spec)
    #             params = sum(p.numel() for p in audio_encoder.parameters()) / 1e6
    #         self.logger.info(
    #             colored(
    #                 f"Audio Encoder: "
    #                 f"{audio_encoder.__class__.__name__}:{audio_encoder_config}, "
    #                 f"Model Parameters: {params:.2f} M",
    #                 "green",
    #                 attrs=["bold"],
    #             )
    #         )
    #     else:
    #         audio_encoder = None
    #         self.logger.info(
    #             colored(
    #                 f"Audio Encoder is not used since load_audio_feat is True",
    #                 "green",
    #                 attrs=["bold"],
    #             )
    #         )
    #     return audio_encoder
    
    def configure_pipeline(self):
        if self.config.train.task.startswith("audio"):
            context_encoder = self.audio_encoder
        elif self.config.train.task.startswith("text"):
            context_encoder = self.text_encoder
        else:
            raise ValueError(f"Invalid task: {self.config.train.task}")

        overlap_cfg = self.config.scheduler.get("overlap_decode", {})
        if overlap_cfg.get("enabled", False):
            from diffusion.pipeline.emage_overlap_inference import CausalEmageOverlapInferencePipeline
            pipeline = CausalEmageOverlapInferencePipeline(
                config=self.config,
                sampler=self.sampler,
                logger=self.logger,
                context_encoder=context_encoder,
                speech_model=self.speech_model,
                device=self.device,
                overlap_frames=overlap_cfg.get("overlap_frames", 8),
                blend_mode=overlap_cfg.get("blend_mode", "linear"),
            )
        else:
            pipeline = CausalEmageInferencePipeline(
                config=self.config,
                sampler=self.sampler,
                logger=self.logger,
                context_encoder=context_encoder,
                speech_model=self.speech_model,
                device=self.device,
            )
        self.logger.info(
            colored(
                f"Using {pipeline.__class__.__name__}:{pipeline}.",
                "green",
                attrs=["bold"],
        )
        )
        return pipeline

    def configure_guider(self):
        """Override parent's configure_guider to support BLD-specific CFG types."""
        guider_type = self.config.model.guider.guider_type
        if guider_type in ("classifier-free-bld", "classifier-free-bld-audio", "classifier-free-bld-history"):
            params = dict(self.config.model.guider)
            params.pop("guider_type")
            guider = BLDCFGGuider(**params)
            self.logger.info(
                colored(
                    f"Using {guider_type} Guider: {guider.__class__.__name__}, "
                    f"audio_cfg={guider.cfg_audio_scale}, history_cfg={guider.cfg_history_scale}, "
                    f"ucg_prob={guider.ucg_prob}",
                    "green",
                    attrs=["bold"],
                )
            )
            return guider
        # Fall back to parent for other guider types (identity, classifier-free, etc.)
        return super().configure_guider()

    def _build_sampling_model_fn(self, model_kwargs, model=None, is_sample: bool = True):
        if model is None:
            model = self.model
        def model_fn(x, t, *args, **kwargs):
            if len(t.shape) == 1:
                t = t.unsqueeze(1).repeat(1, x.shape[1])
            results = model(**self.guider(
                x, t, *args, **kwargs, model_kwargs=model_kwargs, is_sample=is_sample, task=self.config.train.task
            ))
            values = {}
            for key, value in results.items():
                if key in ["motion"]:
                    values[key] = value

            # KV cache prefill 路径下，模型可能只更新 cache 而不返回有效 motion。
            # 此时无需做 CFG 融合，直接返回即可（调用方不会使用该返回值）。
            if kwargs.get("prefill_only", False):
                return values.get("motion", None)

            if values.get("motion", None) is None:
                return None

            values = self.guider.guide(**values, timestep=t, task=self.config.train.task)
            return values["motion"]
        return model_fn

    def _get_weak_monotonic_strength(self, step: int | None = None) -> float:
        """
        Return rho in [0,1].
        rho=1 -> strong monotonic (sorted)
        rho=0 -> no monotonic (raw)
        """
        sch = self.config.scheduler
        if not sch.get("weak_monotonic_enable", True):
            return 1.0

        rho0 = float(sch.get("weak_monotonic_strength_init", 1.0))
        rho_min = float(sch.get("weak_monotonic_strength_min", rho0))
        schedule = str(sch.get("weak_monotonic_strength_schedule", "cosine"))
        warmup_steps = int(sch.get("weak_monotonic_warmup_steps", 0))
        anneal_end_steps = int(sch.get("weak_monotonic_anneal_end_steps", warmup_steps))
        anneal_end_steps = max(anneal_end_steps, warmup_steps)

        if step is None:
            step = 0

        if (step is not None and step > warmup_steps and not self._weak_mono_warmup_logged):
            if (not hasattr(self, "accelerator")) or self.accelerator is None or self.accelerator.is_main_process:
                self.logger.info(
                    f"Weak monotonic warmup finished at step {step} (warmup_steps={warmup_steps})."
                )
            self._weak_mono_warmup_logged = True

        if step <= warmup_steps:
            rho = rho0
        elif step >= anneal_end_steps:
            rho = rho_min
        else:
            q = (step - warmup_steps) / max(anneal_end_steps - warmup_steps, 1)
            if schedule == "linear":
                rho = rho0 + (rho_min - rho0) * q
            else:
                rho = rho_min + (rho0 - rho_min) * 0.5 * (1.0 + math.cos(math.pi * q))

        rho = max(0.0, min(1.0, float(rho)))
        return rho

    def _build_block_repeats(
        self,
        num_frame: int,
        num_frame_per_block: int,
        sampling_name: str,
    ):
        if self.config.train.independent_first_frame:
            if (num_frame - 1) % num_frame_per_block != 0:
                raise ValueError(
                    f"When independent_first_frame=True, {sampling_name} requires num_frame-1 to be divisible "
                    f"by num_frame_per_block, got num_frame={num_frame}, num_frame_per_block={num_frame_per_block}."
                )
            num_blocks = (num_frame - 1) // num_frame_per_block
            block_repeats = [1] + [num_frame_per_block] * num_blocks
        else:
            if num_frame % num_frame_per_block != 0:
                raise ValueError(
                    f"{sampling_name} requires num_frame to be divisible by num_frame_per_block, "
                    f"got num_frame={num_frame}, num_frame_per_block={num_frame_per_block}."
                )
            num_blocks = num_frame // num_frame_per_block
            block_repeats = [num_frame_per_block] * num_blocks
        return block_repeats

    def _sample_local_jitter_timestep(
        self,
        min_timestep: int,
        max_timestep: int,
        batch_size: int,
        num_frame: int,
        num_frame_per_block: int,
        monotonic_blocks: bool,
    ):
        """
        Weak-constrained same-level sampling:
        1) sample base_t shared by all blocks;
        2) sample per-block delta_t;
        3) t_block = clip(base_t + delta_t, [min_timestep, max_timestep)).

        This is backward-compatible by default because local_jitter_timestep.enabled=false.
        """
        cfg = self.config.scheduler.get("local_jitter_timestep", None)
        if cfg is None or (not cfg.get("enabled", False)):
            return None

        # Keep existing monotonic behavior unless explicitly allowed.
        allow_with_monotonic = bool(cfg.get("allow_with_monotonic", False))
        if monotonic_blocks and (not allow_with_monotonic):
            return None

        block_repeats = self._build_block_repeats(
            num_frame=num_frame,
            num_frame_per_block=num_frame_per_block,
            sampling_name="local_jitter_timestep",
        )
        num_blocks = len(block_repeats)

        base_t = self._sample_timestep(batch_size, 1, min_timestep, max_timestep)
        base_t = base_t.expand(-1, num_blocks).long()

        distribution = str(cfg.get("distribution", "uniform")).lower()
        delta = torch.zeros((batch_size, num_blocks), device=self.device, dtype=torch.long)
        delta_max = int(cfg.get("delta_max", 0))

        if distribution == "uniform":
            if delta_max < 0:
                raise ValueError(
                    f"local_jitter_timestep.delta_max must be >= 0 for uniform sampling, got {delta_max}."
                )
            if delta_max > 0:
                delta = torch.randint(
                    low=-delta_max,
                    high=delta_max + 1,
                    size=(batch_size, num_blocks),
                    device=self.device,
                    generator=self.generator,
                ).long()
        elif distribution in ["normal", "gaussian"]:
            delta_std = float(cfg.get("delta_std", 0.0))
            if delta_std < 0:
                raise ValueError(
                    f"local_jitter_timestep.delta_std must be >= 0 for normal sampling, got {delta_std}."
                )
            if delta_std > 0:
                delta = torch.round(
                    torch.randn(
                        (batch_size, num_blocks),
                        device=self.device,
                        generator=self.generator,
                    ) * delta_std
                ).long()
            if delta_max > 0:
                delta = delta.clamp(-delta_max, delta_max)
        else:
            raise ValueError(
                f"Unsupported local_jitter_timestep.distribution={distribution}. "
                "Use one of [uniform, normal]."
            )

        if bool(cfg.get("anchor_first_block_to_base", False)):
            delta[:, 0] = 0

        block_timesteps = (base_t + delta).clamp(min_timestep, max_timestep - 1).long()

        if bool(cfg.get("sort_blocks", False)):
            monotonic_order = str(self.config.scheduler.get("monotonic_order", "asc")).lower()
            if monotonic_order in ["asc", "increasing", "nondecreasing"]:
                sort_desc = False
            elif monotonic_order in ["desc", "decreasing", "nonincreasing"]:
                sort_desc = True
            else:
                raise ValueError(
                    f"Unsupported scheduler.monotonic_order={monotonic_order}. "
                    "Use one of [asc, desc]."
                )
            block_timesteps = torch.sort(block_timesteps, dim=1, descending=sort_desc).values

        block_repeats_tensor = torch.tensor(block_repeats, device=self.device)
        timestep = torch.repeat_interleave(block_timesteps, block_repeats_tensor, dim=1)

        if not self._local_jitter_timestep_logged:
            if (not hasattr(self, "accelerator")) or self.accelerator is None or self.accelerator.is_main_process:
                self.logger.info(
                    "Using local_jitter_timestep sampling "
                    f"(distribution={distribution}, delta_max={delta_max}, "
                    f"allow_with_monotonic={allow_with_monotonic})."
                )
            self._local_jitter_timestep_logged = True

        return timestep
    
    def _maybe_self_sample_motion(self, latent_motion, model_kwargs, step, model_fn=None):
        cfg = getattr(self.config.scheduler, "self_sampling", None)
        if cfg is None or not cfg.get("enabled", False):
            return latent_motion.clone().detach()
        if not self.config.train.teacher_forcing:
            raise ValueError(
                "self_sampling requires teacher_forcing=True. "
                "Self-sampled motion is used as clean_motion context for teacher forcing; "
                "without teacher_forcing the result would be discarded."
            )
        warmup = cfg.get("warmup_steps", 0)
        if step < warmup:
            return latent_motion.clone().detach()
        if step > warmup and not self._self_sampling_warmup_logged:
            if (not hasattr(self, "accelerator")) or self.accelerator is None or self.accelerator.is_main_process:
                self.logger.info(
                    f"Self-sampling warmup finished at step {step} (warmup_steps={warmup})."
                )
            self._self_sampling_warmup_logged = True

        use_ema = cfg.get("use_ema", True)
        ema_update_interval = int(cfg.get("ema_update_interval", 5))
        ema_update_interval = max(1, ema_update_interval)
        ema_rate = float(cfg.get("ema_rate", getattr(self, "ema_rate", 0.999)))
        if use_ema and (not self.config.train.use_fsdp):
            if self.self_sampling_ema_model is None:
                src_model = self.accelerator.unwrap_model(self.model) if hasattr(self, "accelerator") else self.model
                self.self_sampling_ema_model = deepcopy(src_model).eval()
                for p in self.self_sampling_ema_model.parameters():
                    p.requires_grad_(False)
                self.self_sampling_ema_model.to(self.device)
                self._self_sampling_ema_last_step = step
                if (not hasattr(self, "accelerator")) or self.accelerator is None or self.accelerator.is_main_process:
                    self.logger.info(
                        colored(
                            f"Created self-sampling EMA model (ema_rate={ema_rate:.4f}, update_interval={ema_update_interval}).",
                            "yellow",
                            attrs=["bold"],
                        )
                    )
            if (
                self._self_sampling_ema_last_step is None
                or (step - self._self_sampling_ema_last_step) >= ema_update_interval
            ):
                src_model = self.accelerator.unwrap_model(self.model) if hasattr(self, "accelerator") else self.model
                for p_src, p_ema in zip(src_model.parameters(), self.self_sampling_ema_model.parameters()):
                    if p_src.requires_grad:
                        p_ema.data.mul_(ema_rate).add_(p_src.data, alpha=1 - ema_rate)
                for b_src, b_ema in zip(src_model.buffers(), self.self_sampling_ema_model.buffers()):
                    b_ema.data.copy_(b_src.data)
                self._self_sampling_ema_last_step = step

        shift_s = cfg.get("shift", 1.0)
        B, L, _ = latent_motion.shape
        with torch.no_grad():
            # sample t_s ~ LogitNormal(0,1) -> sigmoid(N(0,1))
            # t_s is a single scalar shared across the entire batch (Algorithm 1, line 2)
            z = torch.randn((1, 1, 1), device=latent_motion.device, dtype=latent_motion.dtype)
            t_s = torch.sigmoid(z)
            t_s = shift_s * t_s / (1 + (shift_s - 1) * t_s)
            eps = self.get_initial_noise(latent_motion)
            x_ts = (1 - t_s) * latent_motion + t_s * eps

            # t_s is the same scalar for all samples in the batch
            t_s_scalar = t_s.view(-1)[0].item()

            # autoregressive over blocks using sampler + kv cache
            block_size = self.config.train.num_frame_per_block
            num_blocks = (L + block_size - 1) // block_size
            # KV cache: initialize if None or batch size changed, otherwise reset
            cache = self.inference_pipeline.kv_cache1
            cache_batch_ok = False
            if cache is not None:
                # Check batch dim of the first tensor in the first block's cache
                for v in cache[0].values():
                    if isinstance(v, torch.Tensor) and v.ndim >= 2:
                        cache_batch_ok = (v.shape[0] == B)
                        break
            if not cache_batch_ok:
                self.inference_pipeline._initialize_kv_cache(batch_size=B, dtype=latent_motion.dtype, device=latent_motion.device)
            else:
                self.inference_pipeline._reset_kv_cache()
            if model_fn is None:
                sample_model = self.self_sampling_ema_model if use_ema and self.self_sampling_ema_model is not None else self.model
                model_fn = self._build_sampling_model_fn(model_kwargs, model=sample_model, is_sample=True)
            denoised_blocks = []
            for bidx in range(num_blocks):
                start = bidx * block_size
                end = min(start + block_size, L)
                noisy_block = x_ts[:, start:end]
                context_block = model_kwargs["context"][:, start:end]
                expression_block = model_kwargs["expression"][:, start:end]
                denoised_pred, _ = self.inference_pipeline._sample_block_with_cache(
                    model_fn,
                    noisy_block,
                    context_block,
                    expression_block,
                    model_kwargs["speaker_id"],
                    current_start_frame=start,
                    seq_len=L,
                    cache_dtype=latent_motion.dtype,
                    normalize_rot6d=True,
                    return_intermediate=False,
                    return_model_output=False,
                    start_sigma=t_s_scalar,
                )
                denoised_blocks.append(denoised_pred)
            x0 = torch.cat(denoised_blocks, dim=1)
        return x0.detach()
    
    def _sample_timestep_unit(self, batch_size: int, num_frame: int) -> torch.Tensor:
        """
        Sample in unit interval [0, 1) according to current timestep weighting scheme.
        """
        if self.config.scheduler.weighting_scheme in ["logit_normal"]:
            # adapting from diffusers.training_utils
            u = compute_density_for_timestep_sampling(
                weighting_scheme=self.config.scheduler.weighting_scheme,
                batch_size=batch_size,
                num_frame=num_frame,
                logit_mean=self.config.scheduler.logit_mean,
                logit_std=self.config.scheduler.logit_std,
                mode_scale=None,  # not used
            )
            return u.to(self.device)
        return torch.rand((batch_size, num_frame), device=self.device, generator=self.generator)

    def _sample_timestep(self, batch_size, num_frame, min_timestep, max_timestep):
        if max_timestep <= min_timestep:
            raise ValueError(
                f"max_timestep must be larger than min_timestep, got min={min_timestep}, max={max_timestep}."
            )
        u = self._sample_timestep_unit(batch_size, num_frame)
        timesteps = (u * (max_timestep - min_timestep) + min_timestep).long().to(self.device)
        timesteps = timesteps.clamp(min_timestep, max_timestep - 1)
        return timesteps

    def _sample_timestep_per_sample_range(
        self,
        min_timestep_per_sample: torch.Tensor,
        max_timestep_per_sample: torch.Tensor,
    ) -> torch.Tensor:
        """
        Sample one timestep per sample with per-sample [min, max) range.
        """
        min_timestep_per_sample = min_timestep_per_sample.long().to(self.device)
        max_timestep_per_sample = max_timestep_per_sample.long().to(self.device)

        # Ensure valid exclusive range [min, max)
        max_timestep_per_sample = torch.maximum(max_timestep_per_sample, min_timestep_per_sample + 1)
        span = (max_timestep_per_sample - min_timestep_per_sample).long()

        u = self._sample_timestep_unit(min_timestep_per_sample.shape[0], 1).reshape(-1)
        sampled = (u * span.float() + min_timestep_per_sample.float()).long()

        sampled = torch.minimum(sampled, max_timestep_per_sample - 1)
        return sampled.long()

    def _sample_monotonic_block_timesteps_cascade(
        self,
        batch_size: int,
        num_blocks: int,
        min_timestep: int,
        max_timestep: int,
        sort_desc: bool,
    ) -> torch.Tensor:
        """
        Cascade monotonic block sampling:
        - asc : t1 ~ p([min,max)), ti ~ p([t(i-1), max))
        - desc: t1 ~ p([min,max)), ti ~ p([min, t(i-1)+1))
        """
        if num_blocks <= 0:
            return torch.empty((batch_size, 0), device=self.device, dtype=torch.long)

        block_timesteps = torch.empty((batch_size, num_blocks), device=self.device, dtype=torch.long)
        prev_t = self._sample_timestep(batch_size, 1, min_timestep, max_timestep).squeeze(1)
        block_timesteps[:, 0] = prev_t

        min_vec = torch.full((batch_size,), min_timestep, device=self.device, dtype=torch.long)
        max_vec = torch.full((batch_size,), max_timestep, device=self.device, dtype=torch.long)

        for i in range(1, num_blocks):
            if not sort_desc:
                low = prev_t
                high = max_vec
            else:
                low = min_vec
                high = prev_t + 1

            cur_t = self._sample_timestep_per_sample_range(low, high)
            block_timesteps[:, i] = cur_t
            prev_t = cur_t

        return block_timesteps.long()

    def _sample_monotonic_block_timesteps_cum(
        self,
        block_timesteps_raw: torch.Tensor,
        sort_desc: bool,
    ) -> torch.Tensor:
        """
        Monotonic projection by cumulative operator:
        - asc : cummax
        - desc: cummin
        """
        if not sort_desc:
            return torch.cummax(block_timesteps_raw, dim=1).values.long()
        return torch.cummin(block_timesteps_raw, dim=1).values.long()
    
    
    def sample_timestep(
        self,
        min_timestep: int,
        max_timestep: int,
        batch_size: int,
        num_frame: int,
        num_frame_per_block: int,
        uniform_timestep: bool = False,
        monotonic_blocks: bool = False,
        step: int | None = None,
    ) -> torch.Tensor:
        """
        Randomly generate a timestep tensor based on the generator's task type. It uniformly samples a timestep
        from the range [min_timestep, max_timestep], and returns a tensor of shape [batch_size, num_frame].
        - If uniform_timestep, it will use the same timestep for all frames.
        - If not uniform_timestep, it will use a different timestep for each block.
        - If monotonic_blocks, block timesteps follow monotonic order across blocks.
          Order is controlled by scheduler.monotonic_order:
            - "asc" (default): non-decreasing
            - "desc": non-increasing
          Sampling method is controlled by scheduler.monotonic_sampling_method:
            - "sort" (default): sample then sort
            - "sort_cyclic" (or "sortcyclic"): sort then apply a random cyclic shift per batch to preserve uniform marginals
            - "cummax": sample then cumulative monotonic projection
            - "cascade": autoregressive cascade sampling
        - step is the global training step for weak monotonic scheduling (in steps, not ratio).
        """
        if uniform_timestep:
            timestep = self._sample_timestep(batch_size, 1, min_timestep, max_timestep).repeat(1, num_frame)
            return timestep
        local_jitter_timestep = self._sample_local_jitter_timestep(
            min_timestep=min_timestep,
            max_timestep=max_timestep,
            batch_size=batch_size,
            num_frame=num_frame,
            num_frame_per_block=num_frame_per_block,
            monotonic_blocks=monotonic_blocks,
        )
        if local_jitter_timestep is not None:
            return local_jitter_timestep
        elif monotonic_blocks:
            # sample per-block timesteps then (soft) enforce non-decreasing across blocks
            block_repeats = self._build_block_repeats(
                num_frame=num_frame,
                num_frame_per_block=num_frame_per_block,
                sampling_name="monotonic block sampling",
            )
            expected_frames = sum(block_repeats)
            if expected_frames != num_frame:
                raise ValueError(
                    f"num_frame ({num_frame}) is not aligned with num_frame_per_block ({num_frame_per_block}) "
                    "for monotonic block sampling."
                )
            block_timesteps_raw = self._sample_timestep(
                batch_size, len(block_repeats), min_timestep, max_timestep
            )
            monotonic_order = str(self.config.scheduler.get("monotonic_order", "asc")).lower()
            if monotonic_order in ["asc", "increasing", "nondecreasing"]:
                sort_desc = False
            elif monotonic_order in ["desc", "decreasing", "nonincreasing"]:
                sort_desc = True
            else:
                raise ValueError(
                    f"Unsupported scheduler.monotonic_order={monotonic_order}. "
                    "Use one of [asc, desc]."
                )

            monotonic_sampling_method = str(
                self.config.scheduler.get("monotonic_sampling_method", "sort")
            ).lower()
            if monotonic_sampling_method == "sort":
                block_timesteps_monotonic = torch.sort(
                    block_timesteps_raw,
                    dim=1,
                    descending=sort_desc,
                ).values
            elif monotonic_sampling_method in ["sort_cyclic", "sortcyclic"]:
                # First sort them to get order statistics
                block_timesteps_sorted = torch.sort(
                    block_timesteps_raw,
                    dim=1,
                    descending=sort_desc,
                ).values
                # Then apply a random cyclic shift per sample in the batch
                num_blocks = len(block_repeats)
                shifts = torch.randint(0, num_blocks, (batch_size,), device=self.device, generator=self.generator)
                # idx: [batch_size, num_blocks]
                idx = (torch.arange(num_blocks, device=self.device).unsqueeze(0) - shifts.unsqueeze(1)) % num_blocks
                block_timesteps_monotonic = torch.gather(block_timesteps_sorted, 1, idx)
            elif monotonic_sampling_method == "cummax":
                block_timesteps_monotonic = self._sample_monotonic_block_timesteps_cum(
                    block_timesteps_raw=block_timesteps_raw,
                    sort_desc=sort_desc,
                )
            elif monotonic_sampling_method == "cascade":
                block_timesteps_monotonic = self._sample_monotonic_block_timesteps_cascade(
                    batch_size=batch_size,
                    num_blocks=len(block_repeats),
                    min_timestep=min_timestep,
                    max_timestep=max_timestep,
                    sort_desc=sort_desc,
                )
            else:
                raise ValueError(
                    f"Unsupported scheduler.monotonic_sampling_method={monotonic_sampling_method}. "
                    "Use one of [sort, sort_cyclic (or sortcyclic), cummax, cascade]."
                )

            rho = self._get_weak_monotonic_strength(step)
            mode = str(self.config.scheduler.get("weak_monotonic_mode", "mix_cummax"))

            if rho >= 0.999:
                block_timesteps = block_timesteps_monotonic
            elif rho <= 0.001:
                block_timesteps = block_timesteps_raw
            else:
                if mode == "bounded_drop":
                    delta_max = int(self.config.scheduler.get(
                        "weak_monotonic_delta_max",
                        max_timestep - min_timestep
                    ))
                    delta = int(round((1.0 - rho) * delta_max))
                    block_timesteps = block_timesteps_raw.clone()
                    for i in range(1, block_timesteps.shape[1]):
                        if not sort_desc:
                            # asc target: allow downward fluctuation bounded by delta
                            # t_i >= t_{i-1} - delta
                            block_timesteps[:, i] = torch.maximum(
                                block_timesteps[:, i],
                                block_timesteps[:, i - 1] - delta
                            )
                        else:
                            # desc target: allow upward fluctuation bounded by delta
                            # t_i <= t_{i-1} + delta
                            block_timesteps[:, i] = torch.minimum(
                                block_timesteps[:, i],
                                block_timesteps[:, i - 1] + delta
                            )
                else:
                    mix = rho * block_timesteps_monotonic.float() + (1.0 - rho) * block_timesteps_raw.float()
                    mix = torch.round(mix)
                    if not sort_desc:
                        block_timesteps = torch.cummax(mix, dim=1).values
                    else:
                        block_timesteps = torch.cummin(mix, dim=1).values

                block_timesteps = block_timesteps.clamp(min_timestep, max_timestep - 1).long()

            block_repeats_tensor = torch.tensor(block_repeats, device=self.device)
            timestep = torch.repeat_interleave(block_timesteps, block_repeats_tensor, dim=1)
            return timestep
        else:
            timestep = self._sample_timestep(batch_size, num_frame, min_timestep, max_timestep)
            # make the noise level the same within every block
            if self.config.train.independent_first_frame:
                # the first frame is always kept the same
                timestep_from_second = timestep[:, 1:]
                timestep_from_second = timestep_from_second.reshape(
                    timestep_from_second.shape[0], -1, num_frame_per_block)
                timestep_from_second[:, :, 1:] = timestep_from_second[:, :, 0:1]
                timestep_from_second = timestep_from_second.reshape(
                    timestep_from_second.shape[0], -1)
                timestep = torch.cat([timestep[:, 0:1], timestep_from_second], dim=1)
            else:
                timestep = timestep.reshape(
                    timestep.shape[0], -1, num_frame_per_block)
                timestep[:, :, 1:] = timestep[:, :, 0:1]
                timestep = timestep.reshape(timestep.shape[0], -1)
            return timestep
    
    def _compute_per_block_losses(
        self,
        pred,
        target,
        timestep,
        fk_func,
        block_start,
        block_end,
        extra_logvar=None,
        frame_supervision_weight=None,
    ):
        """
        计算单个 block 的损失项，直接复用 EmageDiffusion 的损失逻辑，
        确保与训练路径（smooth_l1 / mse / 权重等）保持一致。
        """
        pred_block = pred[:, block_start:block_end]
        target_block = target[:, block_start:block_end]
        timestep_block = timestep[:, block_start:block_end]

        extra_block = None
        if extra_logvar is not None:
            extra_block = {"logvar": extra_logvar[:, block_start:block_end]}
        if frame_supervision_weight is not None:
            if extra_block is None:
                extra_block = {}
            extra_block["frame_supervision_weight"] = frame_supervision_weight[:, block_start:block_end]

        loss_motion_weighting, geometric_loss_weighting = self.diffusion.loss_weighting(
            timestep_block, extra_block
        )

        return self.diffusion.loss_function(
            target_block,
            pred_block,
            loss_motion_weighting,
            geometric_loss_weighting,
            extra_block,
            timestep=timestep_block,
            forward_kinematics_func=fk_func,
            num_frame_per_block=self.config.train.num_frame_per_block,
        )

    def _build_frame_supervision_weight(self, batch_size: int, num_frame: int):
        """
        Build optional frame-wise supervision weighting for BLD block-level supervision shaping.

        Backward compatibility:
        - If config is missing or disabled, return None (no behavior change).
        """
        cfg = self.config.scheduler.get("block_supervision", None)
        if cfg is None or (not cfg.get("enabled", False)):
            return None

        block_size = int(self.config.train.num_frame_per_block)
        if block_size <= 0:
            return None

        if self.config.train.independent_first_frame:
            if num_frame < 1:
                return None
            if (num_frame - 1) % block_size != 0:
                raise ValueError(
                    f"When independent_first_frame=True, num_frame-1 must be divisible by block_size, "
                    f"got num_frame={num_frame}, block_size={block_size}."
                )
            num_blocks = (num_frame - 1) // block_size
            block_repeats = [1] + [block_size] * num_blocks
        else:
            if num_frame % block_size != 0:
                raise ValueError(
                    f"block supervision requires num_frame to be divisible by block_size, "
                    f"got num_frame={num_frame}, block_size={block_size}."
                )
            num_blocks = num_frame // block_size
            block_repeats = [block_size] * num_blocks

        mode = str(cfg.get("mode", "uniform"))
        if mode == "uniform":
            block_weight = torch.ones(len(block_repeats), device=self.device, dtype=torch.float32)
        elif mode == "tail_k":
            tail_k = int(cfg.get("tail_k", 1))
            tail_k = max(1, min(tail_k, len(block_repeats)))
            head_weight = float(cfg.get("head_weight", 1.0))
            tail_weight = float(cfg.get("tail_weight", 1.0))
            block_weight = torch.full(
                (len(block_repeats),),
                fill_value=head_weight,
                device=self.device,
                dtype=torch.float32,
            )
            block_weight[-tail_k:] = tail_weight
        elif mode == "custom":
            custom_weights = cfg.get("custom_weights", None)
            if custom_weights is None:
                raise ValueError("custom_weights must be provided when block_supervision.mode=custom")
            if len(custom_weights) != len(block_repeats):
                raise ValueError(
                    f"custom_weights length ({len(custom_weights)}) does not match number of blocks ({len(block_repeats)})"
                )
            block_weight = torch.tensor(custom_weights, device=self.device, dtype=torch.float32)
        elif mode == "random_k":
            keep_k = int(cfg.get("random_k", 1))
            keep_k = max(1, min(keep_k, len(block_repeats)))
            random_scope = str(cfg.get("random_scope", "global")).lower()
            on_weight = float(cfg.get("random_on_weight", 1.0))
            off_weight = float(cfg.get("random_off_weight", 0.0))

            if random_scope in ["global", "batch"]:
                block_weight = torch.full(
                    (len(block_repeats),),
                    fill_value=off_weight,
                    device=self.device,
                    dtype=torch.float32,
                )
                random_scores = torch.rand(
                    (len(block_repeats),),
                    device=self.device,
                    generator=self.generator,
                )
                keep_idx = torch.topk(random_scores, k=keep_k, dim=0, largest=True).indices
                block_weight[keep_idx] = on_weight
            elif random_scope in ["per_sample", "sample"]:
                block_weight = torch.full(
                    (batch_size, len(block_repeats)),
                    fill_value=off_weight,
                    device=self.device,
                    dtype=torch.float32,
                )
                random_scores = torch.rand(
                    (batch_size, len(block_repeats)),
                    device=self.device,
                    generator=self.generator,
                )
                keep_idx = torch.topk(random_scores, k=keep_k, dim=1, largest=True).indices
                block_weight.scatter_(1, keep_idx, on_weight)
            else:
                raise ValueError(
                    f"Unknown block_supervision.random_scope: {random_scope}. "
                    "Use one of [global, per_sample]."
                )
        else:
            raise ValueError(f"Unknown block_supervision.mode: {mode}")

        normalize = str(cfg.get("normalize", "none"))
        if normalize == "mean1":
            if block_weight.dim() == 1:
                block_weight = block_weight / (block_weight.mean() + 1e-8)
            elif block_weight.dim() == 2:
                block_weight = block_weight / (block_weight.mean(dim=1, keepdim=True) + 1e-8)
            else:
                raise ValueError(
                    f"Unexpected block_weight dim={block_weight.dim()} for mean1 normalization"
                )
        elif normalize != "none":
            raise ValueError(f"Unknown block_supervision.normalize: {normalize}")

        block_repeats_tensor = torch.tensor(block_repeats, device=self.device)
        if block_weight.dim() == 1:
            frame_weight = torch.repeat_interleave(block_weight, block_repeats_tensor, dim=0)
            frame_weight = frame_weight.unsqueeze(0).expand(batch_size, -1).contiguous()
        elif block_weight.dim() == 2:
            frame_weight = torch.repeat_interleave(block_weight, block_repeats_tensor, dim=1).contiguous()
        else:
            raise ValueError(f"Unexpected block_weight dim={block_weight.dim()}")
        return frame_weight

    def debug_gradient(self, global_step, loss_term):
        """
        覆写父类 debug_gradient，在原有 per-loss 梯度范数基础上，
        增加轻量的 block 间梯度不平衡统计（以 motion_loss 为代表）：
        1) max/min 梯度范数比值
        2) 每个 block 的梯度范数占比

        指标实现放在 diffusion.metrics.gradient_conflict 中，
        本函数只负责调度与写回日志字段，避免 trainer 代码臃肿。
        """
        # 1) 调用父类的梯度范数分析（EmageTrainer.debug_gradient -> per_loss_grad_norms）
        super().debug_gradient(global_step, loss_term)
        
        is_debug_step = (
            self.config.train.debug_gradient_interval > 0
            and global_step is not None
            and (global_step - 1) % self.config.train.debug_gradient_interval == 0
        )
        if not is_debug_step:
            return

        # 2) per-block 梯度统计（仅保留 motion_loss 代表项）
        if self._debug_cache is not None:
            pred_wg = self._debug_cache["pred"]
            target_wg = self._debug_cache["target"]
            params = [p for p in self.model.parameters() if p.requires_grad]

            block_metrics = compute_block_motion_grad_debug_metrics(
                pred=pred_wg,
                target=target_wg,
                timestep=self._debug_cache["timestep"],
                fk_func=self._debug_cache["fk_func"],
                block_size=self.config.train.num_frame_per_block,
                compute_block_losses_fn=self._compute_per_block_losses,
                params=params,
                extra_logvar=self._debug_cache.get("logvar", None),
                frame_supervision_weight=self._debug_cache.get("frame_supervision_weight", None),
            )
            loss_term.update(block_metrics)

            interblock_conflict_metrics = compute_interblock_motion_conflict_metrics(
                pred=pred_wg,
                target=target_wg,
                timestep=self._debug_cache["timestep"],
                fk_func=self._debug_cache["fk_func"],
                block_size=self.config.train.num_frame_per_block,
                compute_block_losses_fn=self._compute_per_block_losses,
                params=params,
                extra_logvar=self._debug_cache.get("logvar", None),
                frame_supervision_weight=self._debug_cache.get("frame_supervision_weight", None),
            )
            loss_term.update(interblock_conflict_metrics)

            conflict_metrics = compute_simple_vs_rcxyz_conflict_metrics(
                pred=pred_wg,
                target=target_wg,
                timestep=self._debug_cache["timestep"],
                fk_func=self._debug_cache["fk_func"],
                block_size=self.config.train.num_frame_per_block,
                compute_block_losses_fn=self._compute_per_block_losses,
                params=params,
                extra_logvar=self._debug_cache.get("logvar", None),
                frame_supervision_weight=self._debug_cache.get("frame_supervision_weight", None),
                boundary_window=int(self.config.train.get("debug_boundary_window", 4)),
                train_sampling_steps=int(self.config.scheduler.train_sampling_steps),
                num_timestep_buckets=int(self.config.train.get("debug_conflict_timestep_buckets", 0)),
            )
            loss_term.update(conflict_metrics)

            structural_conflict_metrics = compute_simple_vs_structural_conflict_metrics(
                pred=pred_wg,
                target=target_wg,
                timestep=self._debug_cache["timestep"],
                fk_func=self._debug_cache["fk_func"],
                block_size=self.config.train.num_frame_per_block,
                compute_block_losses_fn=self._compute_per_block_losses,
                params=params,
                extra_logvar=self._debug_cache.get("logvar", None),
                frame_supervision_weight=self._debug_cache.get("frame_supervision_weight", None),
                boundary_window=int(self.config.train.get("debug_boundary_window", 4)),
                train_sampling_steps=int(self.config.scheduler.train_sampling_steps),
                num_timestep_buckets=int(self.config.train.get("debug_conflict_timestep_buckets", 0)),
            )
            loss_term.update(structural_conflict_metrics)

            # 清理缓存
            self._debug_cache = None

    def training_step(self, batch, step, vae_time_all=None, global_step=None): 
        in_id = self.get_input(batch, "tar_id")
        # for index
        hand_indices = self.get_input(batch, "hand_indices")
        upper_indices = self.get_input(batch, "upper_indices")
        lower_indices = self.get_input(batch, "lower_indices")
        face_joint_indices = self.get_input(batch, "face_joint_indices")
        trans_indices = self.get_input(batch, "trans_indices")
        contact_indices = self.get_input(batch, "contact_indices")     
        tar_exps = self.get_input(batch, "tar_exps")
        tar_motion = self.get_input(batch, "tar_motion")
        #########################################################################
        if self.config.train.task.startswith("audio2"):
            in_audio = self.get_input(batch, "in_audio")
            expression = self.get_audio_expression(batch, in_id, in_audio)
            model_kwargs = dict(
                expression=expression,
                context=in_audio,
                speaker_id=in_id,
                hand_indices=hand_indices,
                upper_indices=upper_indices,
                lower_indices=lower_indices,
                face_joint_indices=face_joint_indices,
                trans_indices=trans_indices,
                contact_indices=contact_indices,
            )
        elif self.config.train.task.startswith("text2"):
            in_word = self.get_input(batch, "in_word")
            expression = self.get_expression(in_id, in_word)
            model_kwargs = dict(
                expression=expression,
                context=in_word,
                speaker_id=in_id,
                hand_indices=hand_indices,
                upper_indices=upper_indices,
                lower_indices=lower_indices,
                face_joint_indices=face_joint_indices,
                trans_indices=trans_indices,
                contact_indices=contact_indices,
            )
        else:
            raise ValueError(f"Invalid task: {self.config.train.task}")
        tar_motion = self.get_input(batch, "tar_motion")
        tar_beta = self.get_input(batch, "tar_beta")
        
        results = self.first_stage_encode(tar_motion, vae_time_all)
        latent_motion = results["latent_motion"]
        vae_time_all = results["vae_time_all"]
        
        if self.config.train.teacher_forcing:
            model_kwargs["clean_motion"] = latent_motion 
        
        # Sample a random timestep for each modality
        bs = tar_motion.shape[0]
        timestep = self.sample_timestep(
            min_timestep=0,
            max_timestep=self.config.scheduler.train_sampling_steps,
            batch_size=bs,
            num_frame=tar_motion.shape[1],
            num_frame_per_block=self.config.train.num_frame_per_block,
            uniform_timestep=self.config.scheduler.get("uniform_timestep", False),
            monotonic_blocks=self.config.scheduler.monotonic_blocks,
            step=global_step,
        )

        frame_supervision_weight = self._build_frame_supervision_weight(
            batch_size=bs,
            num_frame=tar_motion.shape[1],
        )
        
        # 包装 model_fn 以捕获模型输出，供 debug_gradient 中的 per-block 分析使用
        _captured_output = {}
        def model_fn(*args, **kwargs):
            result = self.model(**self.guider(
                *args, **kwargs, is_sample=False, task=self.config.train.task))
            _captured_output["motion"] = result["motion"]
            _captured_output["logvar"] = result.get("logvar", None)
            return result
        
        # Optional self-sampling (Resampling Forcing) after warmup
        # Use a copied kwargs dict for loss forward to avoid accidental in-place pollution
        # across later code paths that may still hold references to model_kwargs.
        latent_motion_for_loss = self._maybe_self_sample_motion(latent_motion, model_kwargs, global_step)
        model_kwargs_for_loss = dict(model_kwargs)
        if self.config.train.teacher_forcing:
            # Keep tensor value identical while breaking Python-level aliasing.
            model_kwargs_for_loss["clean_motion"] = latent_motion_for_loss
        #########################################################
        forward_kinematics_func = partial(self.forward_kinematics, batch=batch)
        loss_term = self.diffusion.training_losses(
            model_fn,
            latent_motion,
            timestep,
            model_kwargs=model_kwargs_for_loss,
            smpl_model=self.smplx,
            tar_beta=tar_beta,
            tar_exps=tar_exps,
            forward_kinematics_func=forward_kinematics_func,
            debug_func=self.get_debug_npz_file,
            num_frame_per_block=self.config.train.num_frame_per_block,
            frame_supervision_weight=frame_supervision_weight,
        )
        if self.config.train.get("finetune_speech_model", False):
            loss_term["mse_expression"] = (tar_exps - expression) ** 2 \
                * self.config.scheduler.scheduler.lambda_expression
            loss_term["mse_expression"] = mean_flat(loss_term["mse_expression"])
        
        # 缓存数据给 debug_gradient 使用
        if self.config.train.debug_gradient_interval > 0 and "motion" in _captured_output:
            is_debug_step = (
                global_step is not None
                and (global_step - 1) % self.config.train.debug_gradient_interval == 0
            )
            # 仅在 debug step 保留带图缓存，避免非 debug step 持有计算图
            if is_debug_step:
                self._debug_cache = {
                    "pred": _captured_output["motion"],
                    "target": latent_motion_for_loss,
                    "timestep": timestep,
                    "logvar": _captured_output["logvar"],
                    "frame_supervision_weight": frame_supervision_weight,
                    "fk_func": forward_kinematics_func,
                }
            else:
                self._debug_cache = None
        
        return loss_term, vae_time_all


    @torch.no_grad()
    def shared_validation_test_step(
        self, 
        batch, 
        null_embed, 
        step, 
        choice="validation", 
        global_step = None, 
        epoch = None, 
        visualize = True
    ): # 先不管这个名字了，预想的不对，这个只给validation用，test需要另写
        # for visualize
        log_interval = self.config.train.get("visualize_log_interval", 1)
        if_visualize = self.config.train.visualize and visualize and (
            (step + 1) % log_interval== 0 or (step + 1) == 1
        )

        in_id = self.get_input(batch, "tar_id")
        tar_motion = self.get_input(batch, "tar_motion")
        tar_trans = self.get_input(batch, "tar_trans")
        tar_exps = self.get_input(batch, "tar_exps")
        tar_beta = self.get_input(batch, "tar_beta")
        ori_id = self.get_input(batch, "ori_id")
        # for index
        hand_indices = self.get_input(batch, "hand_indices")
        upper_indices = self.get_input(batch, "upper_indices")
        lower_indices = self.get_input(batch, "lower_indices")
        face_joint_indices = self.get_input(batch, "face_joint_indices")
        trans_indices = self.get_input(batch, "trans_indices")
        contact_indices = self.get_input(batch, "contact_indices")
        #########################################################################
        
        if self.config.train.task.startswith("audio2"):
            in_audio = self.get_input(batch, "in_audio") # 这个是wav
            audio_original = self.get_input(batch, "in_audio_wav")
            expression = self.get_audio_expression(batch, in_id, in_audio)
            model_kwargs = dict(
                expression=expression,
                context=in_audio,
                speaker_id=in_id,
                hand_indices=hand_indices,
                upper_indices=upper_indices,
                lower_indices=lower_indices,
                face_joint_indices=face_joint_indices,
                trans_indices=trans_indices,
                contact_indices=contact_indices,
            )
        elif self.config.train.task.startswith("text2"):
            txt = self.get_input(batch, "in_word")
            txt_original = self.get_input(batch, "in_word_txt")
            expression = self.get_expression(in_id, txt)
            model_kwargs = dict(
                expression=expression,
                context=txt,
                speaker_id=in_id,
                hand_indices=hand_indices,
                upper_indices=upper_indices,
                lower_indices=lower_indices,
                face_joint_indices=face_joint_indices,
                trans_indices=trans_indices,
                contact_indices=contact_indices,
            )
        else:
            raise ValueError(f"Invalid task: {self.config.train.task}")
        
        # Sample a random timestep for each modality
        bs = tar_motion.shape[0]
        results = self.first_stage_encode(tar_motion)
        latent_motion = results["latent_motion"]
        if self.config.train.teacher_forcing:
            model_kwargs["clean_motion"] = latent_motion 
        if choice == "validation":
            # Sample a random timestep for each modality
            bs = tar_motion.shape[0]
            timestep = self.sample_timestep(
                min_timestep=0,
                max_timestep=self.config.scheduler.train_sampling_steps,
                batch_size=bs,
                num_frame=tar_motion.shape[1],
                num_frame_per_block=self.config.train.num_frame_per_block,
                uniform_timestep=self.config.scheduler.get("uniform_timestep", False),
                step=step,
            )
            #########################################################
            model_fn = lambda *args, **kwargs: \
                self.model_instance(
                    **self.guider(
                        *args, **kwargs, is_sample=False, task=self.config.train.task
                )
            )
            forward_kinematics_func = partial(self.forward_kinematics, batch=batch)
            loss_term = self.diffusion.training_losses(
                model_fn, 
                latent_motion, 
                timestep, 
                model_kwargs=model_kwargs,
                smpl_model=self.smplx,
                tar_beta=tar_beta,
                tar_exps=tar_exps,
                forward_kinematics_func=forward_kinematics_func,
                debug_func=self.get_debug_npz_file,
            )

        ########################################################################
        
        if self.config.train.task.split("2")[0] == "text":
            assert null_embed is not None
            model_kwargs = dict(
                expression=expression,
                context=txt,
                speaker_id=in_id, 
                uncond_context=null_embed.unsqueeze(0).repeat(bs, *[1] * len(txt.shape[1:])),
                hand_indices=hand_indices,
                upper_indices=upper_indices,
                lower_indices=lower_indices,
                face_joint_indices=face_joint_indices,
                trans_indices=trans_indices,
                contact_indices=contact_indices,
                clean_motion=latent_motion if self.config.train.teacher_forcing else None,
            )
        elif self.config.train.task.split("2")[0] == "audio":
            model_kwargs = dict(
                expression=expression,
                context=in_audio, 
                speaker_id=in_id, 
                uncond_context=torch.zeros_like(in_audio), # TODO，应该如何使用无条件语音呢？？
                hand_indices=hand_indices,
                upper_indices=upper_indices,
                lower_indices=lower_indices,
                face_joint_indices=face_joint_indices,
                trans_indices=trans_indices,
                contact_indices=contact_indices,
                clean_motion=latent_motion if self.config.train.teacher_forcing else None,
            )
        

        model_fn = self._build_sampling_model_fn(
            model_kwargs, model=self.model_instance, is_sample=True
        )
                
        motion_noise = self.get_initial_noise(latent_motion)
        
        if_vis_intermediate = if_visualize \
            and self.config.train.get("visualize_intermediate", False)
        if_vis_model_output = if_visualize \
            and self.config.train.get("visualize_model_output", False)
        if_vis_adding_noise = if_visualize \
            and self.config.train.get("visualize_adding_noise", False)
        
        motion_latents, extra = self.sampler.sample(
            model_fn,
            motion_noise,
            return_intermediate=if_vis_intermediate,
            return_model_output=if_vis_model_output,
            # post_process_func=lambda x: torch.cat([ 
            #     rc.NormalizeRot6d(x[:, :, :-3]),
            #     x[:, :, -3:],
            # ], dim=-1),
        )
        decode_results = self.first_stage_decode(motion_latents)
         
        motion_results = decode_results["motion"]
        motion_trans = decode_results["trans"]
        motion_results = self.post_process_motion(motion_results)
        # === add: denormalize (对齐底层 EmageTrainer) ===
        if self.config.train.normalize:
            motion_trans, motion_results[..., :6*self.joints] = \
                self.data._train_dataset.denormalize(motion_trans, motion_results[..., :6*self.joints])
            tar_trans, tar_motion[..., :6*self.joints] = \
                self.data._train_dataset.denormalize(tar_trans, tar_motion[..., :6*self.joints])
        
        tar_motion[..., :6*self.joints] = self.post_process_motion(tar_motion[..., :6*self.joints])

        if "motion_axis_angle" in decode_results:
            motion_results_axis_angle = decode_results["motion_axis_angle"]\
                .view(bs, -1, 3 * self.joints)
        else:
            motion_results_axis_angle = rc.rotation_6d_to_axis_angle(
                motion_results.view(bs, motion_results.shape[1], -1, 6))\
                .view(bs, -1, 3 * self.joints)
        tar_motion_axis_angle = rc.rotation_6d_to_axis_angle(
            tar_motion[..., :6*self.joints].view(bs, tar_motion.shape[1], -1, 6))\
                .view(bs, -1, 3 * self.joints)

        # TODO: visualize the motion, expression, audio
        vis_bs = self.config.train.get("visualize_batch_size", bs)
        vis_bs = min(vis_bs, bs)
        if self.accelerator.is_main_process: 
            # 这里相当于是只让主进程进行可视化，多进程可视化自然也是可以实现的，就是有点麻烦
            if if_visualize:
                self.visualize_motion(
                    motion_results_axis_angle, 
                    expression,  
                    motion_trans,
                    tar_beta,
                    tar_motion_axis_angle,
                    tar_exps,
                    tar_trans,
                    ori_id,
                    step=step,
                    global_step=global_step,
                    epoch=epoch,
                    choice=choice,
                    vis_bs=vis_bs,
                )
                
                if "audio" in self.config.train.task.split("2")[0] and not self.load_audio_feat:
                    self.visualize_audio(
                        audio_original, 
                        ori_id,
                        step=step, 
                        global_step=global_step,
                        epoch=epoch,
                        choice=choice,
                        vis_bs=vis_bs,
                    )
                
                    if self.config.train.get("visualize_motion_with_face", True):
                        self.visualize_motion_and_audio_together(
                            ori_id,
                            step=step,
                            global_step=global_step,
                            epoch=epoch,
                            choice=choice,
                            vis_bs=vis_bs,
                        )
                
                if "text" in self.config.train.task.split("2")[0]:
                    pass # TODO: visualize the text
                
                
            if if_vis_model_output:
                self.visualize_sequence(
                    extra["motion_model_outputs"],
                    tar_beta,
                    tar_motion_axis_angle,
                    tar_exps,
                    tar_trans,
                    ori_id,
                    step=step,
                    choice=choice,
                    global_step=global_step,
                    epoch=epoch,
                    vis_bs=vis_bs,
                    vis_type="modelouts",
                )
            
            if if_vis_intermediate:
                self.visualize_sequence(
                    extra["motion_intermediates"],
                    tar_beta,
                    tar_motion_axis_angle,
                    tar_exps,
                    tar_trans,
                    ori_id,
                    step=step,
                    choice=choice,
                    global_step=global_step,
                    epoch=epoch,
                    vis_bs=vis_bs,
                    vis_type="intermediates",
                )
                
            if if_vis_adding_noise:
                adding_noise_motion =\
                self.sampler.get_adding_noise_sequence(
                    latent_motion,
                )
                self.visualize_sequence(
                    adding_noise_motion,
                    tar_beta,
                    tar_motion_axis_angle,
                    tar_exps,
                    tar_trans,
                    ori_id,
                    step=step,
                    choice=choice,
                    global_step=global_step,
                    epoch=epoch,
                    vis_bs=vis_bs,
                    vis_type="adding_noise",
                )
        
        self.accelerator.wait_for_everyone() # 等主进程把文件写好
        if self.config.metrics.evaluator.get("evaluate_on_disk", False):
            # 如果在这种情况下，最好把可视化npz和audio wav给禁用掉，没必要了
            keep_every_epoch = self.config.metrics.evaluator.get("keep_every_epoch", False)
            prefix1 = f"gs{global_step:06d}_" if global_step is not None else ""
            prefix2 = f"epoch{epoch:06d}_" if epoch is not None else ""
            prefix3 = f"step{step:06d}_" if step is not None else ""
            tar_beta_np = tar_beta.detach().cpu().numpy()
            rec_pose_np = motion_results_axis_angle.detach().cpu().numpy()
            rec_exp_np = expression.detach().cpu().numpy()
            rec_trans_np = motion_trans.detach().cpu().numpy()
            tar_pose_np = tar_motion_axis_angle.detach().cpu().numpy()
            tar_exp_np = tar_exps.detach().cpu().numpy()
            tar_trans_np = tar_trans.detach().cpu().numpy()
            tar_audio_np = audio_original.detach().cpu().numpy()

            audio_sr = self.config.data.audio_sr
            
            res_paths = []
            gt_paths = []
            gt_audio_paths = []
            for i in range(bs):
                prefix4 = f"bs{i}_"
                if not keep_every_epoch:
                    prefix = prefix3 + prefix4
                else:
                    prefix = prefix1 + prefix2 + prefix3 + prefix4
                res_path = osp.join(self.config.work_dir, self.npz_save_name, choice,
                    f"rank{self.accelerator.process_index}_{prefix}res_" + ori_id[i] + ".npz")
                gt_path = osp.join(self.config.work_dir, self.npz_save_name, choice,
                    f"rank{self.accelerator.process_index}_{prefix}gt_" + ori_id[i] + ".npz")
                res_paths.append(res_path)
                gt_paths.append(gt_path)
                gt_audio_path = osp.join(self.config.work_dir, self.wav_save_name, choice,
                    f"rank{self.accelerator.process_index}_{prefix}gt_sr{audio_sr}_" + ori_id[i] + ".wav")
                gt_audio_paths.append(gt_audio_path)
                
                np.savez(
                    res_path,
                    betas=tar_beta_np[i,0], 
                    poses=rec_pose_np[i],
                    expressions=rec_exp_np[i],
                    trans=rec_trans_np[i],
                    model='smplx2020',
                    gender='neutral',
                    mocap_frame_rate = 30,
                )
                np.savez(
                    gt_path,
                    betas=tar_beta_np[i,0], 
                    poses=tar_pose_np[i],
                    expressions=tar_exp_np[i],
                    trans=tar_trans_np[i],
                    model='smplx2020',
                    gender='neutral',
                    mocap_frame_rate = 30,
                )
                sf.write(
                    gt_audio_path,
                    tar_audio_np[i],
                    samplerate=audio_sr,
                )
            del tar_beta_np, rec_pose_np, rec_exp_np, rec_trans_np, tar_pose_np, tar_exp_np, tar_trans_np
                
                         
            results = {
                "motion": res_paths,
                "expression": res_paths,
                "gt_motion": gt_paths,
                "gt_expression": gt_paths,
                "gt_audio": gt_audio_paths,
                "betas": gt_paths,
            }
        else:
            results = {
                "motion": motion_results_axis_angle,
                "expression": expression,
                "gt_motion": tar_motion_axis_angle,
                "gt_expression": tar_exps,
                "gt_audio": audio_original,
                "betas": tar_beta
            }

        if choice == "validation":
            return loss_term, results
        elif choice == "test":
            return results
        else:
            raise ValueError(f"Invalid choice: {choice}")
    
    def prepare_mask(self, data_dict):
        return data_dict
    
    @torch.no_grad()
    def shared_test_predict_step(
        self, 
        batch, 
        null_embed, 
        step, 
        choice="test", 
        global_step = None, 
        epoch = None, 
        visualize = True
    ):
        log_interval = self.config.train.get("visualize_log_interval", 1)
        
        in_id = self.get_input(batch, "tar_id")
        tar_motion = self.get_input(batch, "tar_motion")
        tar_trans = self.get_input(batch, "tar_trans")
        tar_exps = self.get_input(batch, "tar_exps")
        tar_beta = self.get_input(batch, "tar_beta")
        tar_pose = self.get_input(batch, "tar_pose_6d")
        ori_id = self.get_input(batch, "ori_id")
        is_padding_sample = self.get_input(batch, "is_padding_sample")
        # for index
        hand_indices = self.get_input(batch, "hand_indices")
        upper_indices = self.get_input(batch, "upper_indices")
        lower_indices = self.get_input(batch, "lower_indices")
        face_joint_indices = self.get_input(batch, "face_joint_indices")
        trans_indices = self.get_input(batch, "trans_indices")
        contact_indices = self.get_input(batch, "contact_indices")
        
        txt = self.get_input(batch, "in_word")
        in_audio = self.get_input(batch, "in_audio")
        audio_original = self.get_input(batch, "in_audio_wav")
        
        bs = tar_motion.shape[0]
        if is_padding_sample is None:
            valid_sample_mask = torch.ones(
                bs,
                dtype=torch.bool,
                device=tar_motion.device,
            )
        else:
            valid_sample_mask = ~is_padding_sample.reshape(-1).to(
                device=tar_motion.device,
                dtype=torch.bool,
            )
            if valid_sample_mask.numel() != bs:
                raise ValueError(
                    "is_padding_sample must contain one marker per sample"
                )

        results = self.first_stage_encode(tar_motion)
        latent_motion = results["latent_motion"]
            
        precomputed_expression_context = None
        if self.config.train.task.startswith("audio2"):
            use_talkshow_validation_cache = (
                choice == "validation"
                and getattr(
                    self.speech_model,
                    "requires_contextual_audio_features",
                    False,
                )
            )
            if use_talkshow_validation_cache:
                # A validation LMDB record is a window from a longer source
                # clip. Reuse its full-clip WavLM cache and halo-aware face
                # context instead of re-encoding the isolated window.
                context = in_audio
                precomputed_expression_context = self.get_audio_expression(
                    batch,
                    in_id,
                    in_audio,
                )
            else:
                # Test/predict records contain the complete source clip, so
                # the inference pipeline owns one raw-audio WavLM pass.
                context = audio_original
            context_pid = in_id
            model_kwargs = dict(
                hand_indices=hand_indices,
                upper_indices=upper_indices,
                lower_indices=lower_indices,
                face_joint_indices=face_joint_indices,
                trans_indices=trans_indices,
                contact_indices=contact_indices,
            )
        elif self.config.train.task.startswith("text2"):
            # expression = self.speech_model(txt, pid=in_id)
            context = txt
            context_pid = in_id
            model_kwargs = dict(
                hand_indices=hand_indices,
                upper_indices=upper_indices,
                lower_indices=lower_indices,
                face_joint_indices=face_joint_indices,
                trans_indices=trans_indices,
                contact_indices=contact_indices,
            )
        else:
            raise ValueError(f"Invalid task: {self.config.train.task}")
        
        model_func = self._build_sampling_model_fn(
            model_kwargs, model=self.model_instance, is_sample=True
        )
        
        motion_noise = self.get_initial_noise(latent_motion)
        motion_results, expression = self.inference_pipeline.inference_with_seed(
            model_func,
            motion_noise,
            context,
            context_pid,
            cache_dtype=next(self.model_instance.parameters()).dtype,
            seed_motion=latent_motion[:, :self.config.scheduler.overlap_length],
            profile=self.config.train.get("profile", False),
            noise_shuffle=self.config.scheduler.noise_shuffle,
            noise_shuffle_window=self.config.scheduler.noise_shuffle_window,
            quintic_bridge=self.config.scheduler.get("quintic_bridge", False),
            bridge_frames=self.config.scheduler.get("bridge_frames", 8),
            bridge_interp_space=self.config.scheduler.get("bridge_interp_space", "rot6d"),
            # 下面这两个都是给debug用的
            tar_beta=tar_beta,
            tar_motion=tar_motion,
            precomputed_expression_context=precomputed_expression_context,
        )
        decode_results = self.first_stage_decode(motion_results)
        
        motion_results = decode_results["motion"]
        trans_results = decode_results["trans"]
        
        # === add: denormalize ===
        if self.config.train.normalize:
            trans_results, motion_results[..., :6*self.joints] = \
                self.data._train_dataset.denormalize(trans_results, motion_results[..., :6*self.joints])
            tar_trans, tar_pose = self.data._train_dataset.denormalize(tar_trans, tar_pose)

        motion_results = rc.NormalizeRot6d(motion_results)
        motion_results = self.post_process_motion(motion_results)
        tar_pose = self.post_process_motion(tar_pose)
        
        motion_results_axis_angle = rc.rotation_6d_to_axis_angle(
            motion_results.view(bs, motion_results.shape[1], -1, 6))\
            .view(bs, -1, 3 * self.joints)
        tar_motion_axis_angle = rc.rotation_6d_to_axis_angle(tar_pose.view(bs, tar_pose.shape[1], -1, 6))\
            .view(bs, -1, 3 * self.joints)

        tar_beta_np = tar_beta.detach().cpu().numpy()
        rec_pose_np = motion_results_axis_angle.detach().cpu().numpy()
        rec_exp_np = expression.detach().cpu().numpy()
        rec_trans_np = trans_results.detach().cpu().numpy()
        tar_pose_np = tar_motion_axis_angle.detach().cpu().numpy()
        tar_exp_np = tar_exps.detach().cpu().numpy()
        tar_trans_np = tar_trans.detach().cpu().numpy()
        tar_audio_np = audio_original.detach().cpu().numpy()
        audio_sr = self.config.data.audio_sr
        
        res_paths = []
        gt_paths = []
        gt_audio_paths = []
        for i in range(bs):
            if not bool(valid_sample_mask[i].item()):
                continue
            res_path = osp.join(self.config.work_dir, self.npz_save_name, choice, "res_" + ori_id[i] + ".npz")
            gt_path = osp.join(self.config.work_dir, self.npz_save_name, choice, "gt_" + ori_id[i] + ".npz")
            res_paths.append(res_path)
            gt_paths.append(gt_path)
            gt_audio_path = osp.join(self.config.work_dir, self.wav_save_name, choice, \
                f"gt_sr{audio_sr}_" + ori_id[i] + ".wav")
            gt_audio_paths.append(gt_audio_path)
            
            np.savez(
                res_path,
                betas=tar_beta_np[i,0], 
                poses=rec_pose_np[i],
                expressions=rec_exp_np[i],
                trans=rec_trans_np[i],
                model='smplx2020',
                gender='neutral',
                mocap_frame_rate = 30,
            )
            np.savez(
                gt_path,
                betas=tar_beta_np[i,0], 
                poses=tar_pose_np[i],
                expressions=tar_exp_np[i],
                trans=tar_trans_np[i],
                model='smplx2020',
                gender='neutral',
                mocap_frame_rate = 30,
            )
            sf.write(
                gt_audio_path,
                tar_audio_np[i],
                samplerate=audio_sr,
            )

        if self.config.metrics.evaluator.get("evaluate_on_disk", False):
            del tar_beta_np, rec_pose_np, rec_exp_np, rec_trans_np, tar_pose_np, tar_exp_np, tar_trans_np                
            results = {
                "motion": res_paths,
                "expression": res_paths,
                "gt_motion": gt_paths,
                "gt_expression": gt_paths,
                "gt_audio": gt_audio_paths,
                "betas": gt_paths,
            }
        else:
            results = {
                "motion": motion_results_axis_angle,
                "expression": expression,
                "gt_motion": tar_motion_axis_angle,
                "gt_expression": tar_exps,
                "gt_audio": audio_original,
                "betas": tar_beta,
            }
            if is_padding_sample is not None:
                results = {
                    key: value[valid_sample_mask]
                    for key, value in results.items()
                }

        return results
    
    # def validation_step(self, batch, null_embed, step, global_step = None, epoch = None, visualize = True):
    #     return self.shared_validation_test_step(
    #         batch, null_embed, step, choice="validation", global_step = global_step, epoch = epoch, visualize = visualize)
    
    # @torch.no_grad()
    # def validation(self, global_step = None, epoch = None):
    #     if self.data._validation_dataloader is None:
    #         if self.accelerator.is_main_process:
    #             self.logger.error("No validation dataloader provided. Will not eval.")
    #         return 
        
    #     self.prepare_for_dist()
    #     self.evaluator.cuda()
    #     self.eval()
    #     # snapshot("val:enter", self.model, self.logger)
    #     # check_model_finite(self.model, "model")
    #     torch.cuda.empty_cache()
    #     if self.accelerator.is_main_process:
    #         self.logger.info("Running validation... ")
    #     # log_buffer = LogBuffer()
    #     all_results = {}
    #     if self.null_embed_path is not None:
    #         null_embed = torch.load(self.null_embed_path, map_location="cpu")
    #         null_embed = null_embed["uncond_prompt_embeds"].to(self.device)
    #     else:
    #         null_embed = None
        
    #     loss_log_buffer = LogBuffer()
    #     time_start, validation_start_time = time.time(), time.time()
    #     validation_dataloader_len = len(self.data._validation_dataloader)
    #     total_steps = validation_dataloader_len
    #     log_interval = self.config.train.get("validation_log_interval", self.config.train.log_interval)
    #     for step, batch in enumerate(self.data._validation_dataloader):
    #         data_dict = self.prepare_data(batch, step, null_embed, is_predict=False)
    #         loss_term, results = self.validation_step(data_dict, null_embed, step, global_step, epoch)
    #         # logs = {}
    #         # for key, val in metric_term:
    #         #     val = val.mean()
    #         #     logs = {f"val_{key}": self.accelerator.gather(val).mean().item()}    
    #         # log_buffer.update(logs)
    #         results = self.evaluator.update(results)
    #         all_results = self.postprocess_results(all_results, results)
            
    #         # loss = sum([val.mean() for _, val in loss_term.items()])
    #         # loss = self.accelerator.gather(loss_term["loss"]).mean().item()
    #         # loss_log_buffer.update({"val_loss": loss})
    #         loss_logs = {}
    #         for key, val in loss_term.items():
    #             val = val.mean()
    #             loss_logs[f"val/{key}"] = self.accelerator.gather(val).mean().item()    
    #         loss_log_buffer.update(loss_logs)
            
    #         if (step + 1) % log_interval== 0 or (step + 1) == 1:
    #             self.accelerator.wait_for_everyone()
    #             avg_time = (time.time() - time_start) / (step + 1)
    #             eta = str(datetime.timedelta(seconds=int(avg_time * (total_steps - step - 1))))
    #             # 重新设计验证日志格式，仿照训练日志的样式
    #             info = (
    #                 colored(
    #                     f"[Validation | Step {step:6d}/{validation_dataloader_len:6d}] ",
    #                     color="magenta",
    #                     attrs=["bold"],
    #                 ) + \
    #                 colored(
    #                     f"ETA: {eta} \n",
    #                     color="blue",
    #                 ) + \
    #                 colored(
    #                     f"[Loss] ",
    #                     color="cyan",
    #                 )
    #             )
    #             loss_log_buffer.average()
    #             info += " | ".join([f"{k}:{v:.4f}" for k, v in loss_log_buffer.output.items()])
    #             if self.accelerator.is_main_process:
    #                 self.logger.info(info)
    #                 self.accelerator.log(loss_log_buffer.output)
    #             loss_log_buffer.clear()
    #             self.extra_validation_log(loss_term=loss_term, results=results)
    #         # if self.accelerator.is_main_process:
    #         #     self.accelerator.log(logs)
        
    #     # log_buffer.average()
    #     # validation_output = log_buffer.output
    #     # info = ", ".join([f"{k}:{v:.4f}" for k, v in validation_output.items()])
    #     # log_buffer.clear()
    #     validation_output = self.evaluator(all_results)
    #     del all_results
    #     info = " | ".join([f"{k} = {v:.12f}" for k, v in validation_output.items()]) if validation_output is not None\
    #         else "No validation result."
    #     total_validation_time = time.time() - validation_start_time
    #     if self.accelerator.is_main_process:
    #         self.logger.info(
    #             colored(
    #                 "Validation Results : " + info,
    #                 "red",
    #                 attrs=["bold"],
    #             )
    #         )
    #         if validation_output is not None:
    #             self.accelerator.log(validation_output)
    #         self.logger.info(f"Validation Finished, Total Time Cost : {total_validation_time:.3f}")
    #     # check_model_finite(self.model, "model")
    #     # snapshot("val:exit", self.model, self.logger)
    #     self.training()
    #     self.evaluator.cpu()
    #     flush()
    #     return validation_output
