"""
    Code modified from self-forcing: https://github.com/guandeh17/Self-Forcing/blob/main/pipeline/causal_inference.py
"""
import torch
from typing import Optional
from diffusion.utils import rotation_conversions as rc
import torch.nn.functional as F
import numpy as np
import os.path as osp
import os
class CausalInferencePipeline:
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
        self.local_attn_size = self.config.model.get("local_attn_size", -1)
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
        self.max_attention_size = self.config.model.get("max_attention_size", 512)

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
            motion = motion[0].unsqueeze(0) # 只可视化第一个数据
            trans = motion[:,:,330:333]
            motion = motion[:,:,:330]
            
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
    
    def context_encoding(self, context, start, end, align_length):
        if self.config.train.task.startswith("audio"):
            sr = self.config.data.audio_sr
            audio_start, audio_end = int(start * sr / self.pose_fps), int(end * sr / self.pose_fps)
            context_ = context[:, audio_start:audio_end]
            audio_encoder_config = self.config.audio.audio_encoder
            audio_encoder_name = audio_encoder_config["name"]
            if audio_encoder_name == "hubert-large-ls960-ft" or audio_encoder_name == "wavlm":
                dtype = context_.dtype
                context_ = self.context_encoder[0](
                    list([_.cpu().numpy() for _ in context_]), 
                    sampling_rate=sr,
                    padding=True, 
                    return_tensors="pt"
                )
                context_mask = context_.attention_mask.to(self.device)
                context_ = context_.input_values.to(self.device)
                if not self.config.audio.audio_encoder.get("quantization", False):
                    context_ = self.context_encoder[1](
                        context_, 
                        attention_mask=context_mask)
                else:
                    context_ = self.context_encoder[1](
                        context_.to(torch.float16), 
                        attention_mask=context_mask.to(torch.float16))
                context_ = context_.last_hidden_state.to(dtype)
            else:
                raise ValueError(f"Invalid audio encoder: {audio_encoder_name}")
            context_ = torch.nn.functional.interpolate(
                context_.permute(0, 2, 1), size=(align_length,), mode='linear', align_corners=True)\
                    .permute(0, 2, 1)
        elif self.config.train.task.startswith("text"):
            # TODO: 解析txt
            context_ = context[:, start:end]
        else:
            raise ValueError(f"Invalid task: {self.config.train.task}")
        return context_
    
    def inference(
        self,
        generator,
        noise: torch.Tensor,
        context: torch.Tensor,
        context_pid: torch.Tensor,
        initial_latent: Optional[torch.Tensor] = None,
        profile: bool = False,
        **kwargs,
    ):
        batch_size, num_frames, num_channels = noise.shape
        
        if not self.independent_first_frame or (self.independent_first_frame and initial_latent is not None):
            # If the first frame is independent and the first frame is provided, then the number of frames in the
            # noise should still be a multiple of num_frame_per_block
            # assert num_frames % self.num_frame_per_block == 0
            if num_frames % self.num_frame_per_block != 0:
                pad_length = self.num_frame_per_block - num_frames % self.num_frame_per_block
                noise = F.pad(
                    noise, 
                    (0, 0, 0, pad_length), 
                    mode="reflect"
                )
                num_frames = noise.shape[1]
            else:
                pad_length = 0
            num_blocks = num_frames // self.num_frame_per_block
        else:
            # Using a [1, 4, 4, 4, 4, 4, ...] model to generate a video without image conditioning
            if (num_frames - 1) % self.num_frame_per_block != 0:
                pad_length = self.num_frame_per_block - (num_frames - 1) % self.num_frame_per_block
                noise = F.pad(
                    noise, 
                    (0, 0, 0, pad_length), 
                    mode="reflect"
                )
                num_frames = noise.shape[1]
            else:
                pad_length = 0
            # assert (num_frames - 1) % self.num_frame_per_block == 0
            num_blocks = (num_frames - 1) // self.num_frame_per_block
            
        num_input_frames = initial_latent.shape[1] if initial_latent is not None else 0
        num_output_frames = num_frames + num_input_frames  # add the initial latent frames
        
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
        
        # preprocess context
        if context.shape[1] > num_output_frames * self.config.data.audio_sr / self.config.data.pose_fps:
            context = context[:, :num_output_frames * self.config.data.audio_sr // self.config.data.pose_fps]
        else:
            context = F.pad(
                context,
                (0, num_output_frames * self.config.data.audio_sr // self.config.data.pose_fps - context.shape[1], 0, 0),
                mode="constant",
                value=0.0
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
        
        if self.kv_cache1 is None:
            self._initialize_kv_cache(
                batch_size=batch_size,
                dtype=noise.dtype,
                device=noise.device
            )
            # 分slice来的话，crossattn将无法cache
            # self._initialize_crossattn_cache(
            #     context_len=context_len,
            #     batch_size=batch_size,
            #     dtype=noise.dtype,
            #     device=noise.device
            # )
        else:
            # reset cross attn cache
            # 规避crossattn cache，因为slice来的话，crossattn将无法cache
            # for block_index in range(self.num_transformer_blocks):
            #     self.crossattn_cache[block_index]["is_init"] = False
            # reset kv cache
            for block_index in range(len(self.kv_cache1)):
                self.kv_cache1[block_index]["global_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noise.device)
                self.kv_cache1[block_index]["local_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noise.device)
                self.kv_cache2[block_index]["global_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noise.device)
                self.kv_cache2[block_index]["local_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noise.device)
        
        # Step 2: Cache context feature
        current_start_frame = 0
        if initial_latent is not None:
            timestep = torch.ones([batch_size, 1], device=noise.device, dtype=torch.int64) * 0
            if self.independent_first_frame:
                # Assume num_input_frames is 1 + self.num_frame_per_block * num_input_blocks
                assert (num_input_frames - 1) % self.num_frame_per_block == 0
                num_input_blocks = (num_input_frames - 1) // self.num_frame_per_block
                output[:, :1] = initial_latent[:, :1]
                context_ = self.context_encoding(context, 0, 1, 1)
                expression = self.speech_model(context_, pid=context_pid)
                expression_output[:, :1] = expression[:, :1]
                generator(
                    initial_latent[:, :1],
                    timestep * 0,
                    context=context_,
                    speaker_id=context_pid,
                    expression=expression,
                    kv_cache={
                        "hand_kv_cache": self.kv_cache1,
                        "body_kv_cache": self.kv_cache2,
                    },
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length,
                    seq_len=num_output_frames,
                )
                current_start_frame += 1
            else:
                # Assume num_input_frames is self.num_frame_per_block * num_input_blocks
                assert num_input_frames % self.num_frame_per_block == 0
                num_input_blocks = num_input_frames // self.num_frame_per_block

            for _ in range(num_input_blocks):
                current_ref_latents = \
                    initial_latent[:, current_start_frame:current_start_frame + self.num_frame_per_block]
                output[:, current_start_frame:current_start_frame + self.num_frame_per_block] = current_ref_latents
                context_ = self.context_encoding(context, 
                        current_start_frame, current_start_frame + self.num_frame_per_block, self.num_frame_per_block)
                expression = self.speech_model(context_, pid=context_pid)
                expression_output[:, current_start_frame:current_start_frame + self.num_frame_per_block]\
                    = expression[:, :self.num_frame_per_block]
                generator(
                    current_ref_latents,
                    timestep * 0,
                    context=context_,
                    speaker_id=context_pid,
                    expression=expression,
                    kv_cache={
                        "hand_kv_cache": self.kv_cache1,
                        "body_kv_cache": self.kv_cache2,
                    },
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length,
                    seq_len=num_output_frames,
                )
                current_start_frame += self.num_frame_per_block

        
        all_num_frames = [self.num_frame_per_block] * num_blocks
        # all_num_frames = [256] * (num_blocks // 4)
        if self.independent_first_frame and initial_latent is None:
            all_num_frames = [1] + all_num_frames
        # iter_num_frames = 0
        for current_num_frames in all_num_frames:
            if profile:
                block_start.record()

            noisy_input = noise[
                :, current_start_frame - num_input_frames:current_start_frame + current_num_frames - num_input_frames]

            # if iter_num_frames % self.pose_length == 0:
            #     end = min(current_start_frame + self.pose_length, num_output_frames)
            #     context_ = self.context_encoding(context, 
            #             current_start_frame,end, end - current_start_frame)
            #     expression = self.speech_model(context_, pid=context_pid)
            #     expression_output[:, current_start_frame:end] = expression
            # iter_num_frames += current_num_frames
            context_ = self.context_encoding(context, 
                    current_start_frame,current_start_frame + current_num_frames, current_num_frames)
            expression = self.speech_model(context_, pid=context_pid)
            expression_output[:, current_start_frame:current_start_frame + current_num_frames] = expression
            # Step 3.1: Spatial denoising loop
            denoised_pred, extra = self.sampler.sample(
                generator,
                noisy_input, 
                context=context_,
                speaker_id=context_pid,
                return_intermediate=True,
                return_model_output=True,
                # post_process_func=lambda x: torch.cat([ 
                #     rc.NormalizeRot6d(x[:, :, :-3]),
                #     x[:, :, -3:],
                # ], dim=-1),
                expression=expression,
                kv_cache={
                    "hand_kv_cache": self.kv_cache1,
                    "body_kv_cache": self.kv_cache2,
                },
                crossattn_cache=self.crossattn_cache,
                current_start=current_start_frame * self.frame_seq_length,
                seq_len=num_output_frames,
                # clean_motion=kwargs["tar_motion"][:,current_start_frame:current_start_frame + current_num_frames],
            )
            denoised_pred = torch.cat([
                rc.NormalizeRot6d(denoised_pred[:, :, :-3]),
                denoised_pred[:, :, -3:],
            ], dim=-1)
            # self.visualize_sequence(
            #     extra["motion_intermediates"],
            #     expression,
            #     kwargs["tar_beta"],
            #     0
            # )
            # self.visualize_sequence(
            #     extra["motion_model_outputs"],
            #     expression,
            #     kwargs["tar_beta"],
            #     0,
            #     vis_type="modelouts",
            # )
            # Step 3.2: record the model's output
            output[:, current_start_frame:current_start_frame + current_num_frames] = denoised_pred
            
            # Step 3.3: rerun with timestep zero to update KV cache using clean context
            context_timestep = torch.ones(
                [batch_size, current_num_frames],
                device=noise.device,
                dtype=torch.long
            ) * self.config.scheduler.get("context_noise", 0.0)
            generator(
                denoised_pred,
                context_timestep,
                context=context_,
                speaker_id=context_pid,
                expression=expression,
                kv_cache={
                    "hand_kv_cache": self.kv_cache1,
                    "body_kv_cache": self.kv_cache2,
                },
                crossattn_cache=self.crossattn_cache,
                current_start=current_start_frame * self.frame_seq_length,
                seq_len=num_output_frames,
            )

            if profile:
                block_end.record()
                torch.cuda.synchronize()
                block_time = block_start.elapsed_time(block_end)
                block_times.append(block_time)

            # Step 3.4: update the start and end frame indices
            current_start_frame += current_num_frames

        if profile:
            # End diffusion timing and synchronize CUDA
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
    
    
    def _initialize_kv_cache(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU KV cache for the Wan model.
        """
        kv_cache1 = []
        kv_cache2 = []
        if self.local_attn_size != -1:
            # Use the local attention size to compute the KV cache size
            kv_cache_size = self.local_attn_size * self.frame_seq_length
        else:
            # Use the default KV cache size
            kv_cache_size = self.max_attention_size # self-forcing硬编码了这个上限大小

        num_heads = self.config.model.model.params.block_config.params.num_heads
        head_dim = self.config.model.model.params.block_config.params.head_dim
        hidden_size = self.config.model.model.params.hidden_size
        if num_heads is None:
            num_heads = hidden_size // head_dim
        elif head_dim is None:
            head_dim = hidden_size // num_heads
        else:
            assert num_heads * head_dim == hidden_size, "num_heads * head_dim should be equal to hidden_size"
        
        for _ in range(self.num_transformer_blocks):
            kv_cache1.append({
                "k": torch.zeros([batch_size, kv_cache_size, num_heads, head_dim], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, kv_cache_size, num_heads, head_dim], dtype=dtype, device=device),
                "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                "local_end_index": torch.tensor([0], dtype=torch.long, device=device)
            })

        self.kv_cache1 = kv_cache1  # always store the clean cache
        for _ in range(self.num_transformer_blocks):
            kv_cache2.append({
                "k": torch.zeros([batch_size, kv_cache_size, num_heads, head_dim], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, kv_cache_size, num_heads, head_dim], dtype=dtype, device=device),
                "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                "local_end_index": torch.tensor([0], dtype=torch.long, device=device)
            })
        self.kv_cache2 = kv_cache2

    def _initialize_crossattn_cache(self, batch_size, context_len, dtype, device):
        """
        Initialize a Per-GPU cross-attention cache for the Wan model.
        """
        crossattn_cache = []
        num_heads = self.config.model.model.params.block_config.params.num_heads
        head_dim = self.config.model.model.params.block_config.params.head_dim
        hidden_size = self.config.model.model.params.hidden_size
        if num_heads is None:
            num_heads = hidden_size // head_dim
        elif head_dim is None:
            head_dim = hidden_size // num_heads
        else:
            assert num_heads * head_dim == hidden_size, "num_heads * head_dim should be equal to hidden_size"

        for _ in range(self.num_transformer_blocks):
            crossattn_cache.append({
                "k": torch.zeros([batch_size, context_len, num_heads, head_dim], dtype=dtype, device=device),
                "v": torch.zeros([batch_size, context_len, num_heads, head_dim], dtype=dtype, device=device),
                "is_init": False
            })
        self.crossattn_cache = crossattn_cache

class FullAudioCausalInferencePipeline(CausalInferencePipeline):
    def __init__(self, config, context_encoder, speech_model, sampler, logger, device = "cuda"):
        super().__init__(config, context_encoder, speech_model, sampler, logger, device)

    def inference(
        self,
        generator,
        noise: torch.Tensor,
        context: torch.Tensor,
        context_pid: torch.Tensor,
        initial_latent: Optional[torch.Tensor] = None,
        profile: bool = False,
        **kwargs,
    ):
        batch_size, num_frames, num_channels = noise.shape
        
        if not self.independent_first_frame or (self.independent_first_frame and initial_latent is not None):
            # If the first frame is independent and the first frame is provided, then the number of frames in the
            # noise should still be a multiple of num_frame_per_block
            # assert num_frames % self.num_frame_per_block == 0
            if num_frames % self.num_frame_per_block != 0:
                pad_length = self.num_frame_per_block - num_frames % self.num_frame_per_block
                noise = F.pad(
                    noise, 
                    (0, 0, 0, pad_length), 
                    mode="reflect"
                )
                num_frames = noise.shape[1]
            else:
                pad_length = 0
            num_blocks = num_frames // self.num_frame_per_block
        else:
            # Using a [1, 4, 4, 4, 4, 4, ...] model to generate a video without image conditioning
            if (num_frames - 1) % self.num_frame_per_block != 0:
                pad_length = self.num_frame_per_block - (num_frames - 1) % self.num_frame_per_block
                noise = F.pad(
                    noise, 
                    (0, 0, 0, pad_length), 
                    mode="reflect"
                )
                num_frames = noise.shape[1]
            else:
                pad_length = 0
            # assert (num_frames - 1) % self.num_frame_per_block == 0
            num_blocks = (num_frames - 1) // self.num_frame_per_block
            
        num_input_frames = initial_latent.shape[1] if initial_latent is not None else 0
        num_output_frames = num_frames + num_input_frames  # add the initial latent frames
        
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
        
        # preprocess context
        if context.shape[1] > num_output_frames * self.config.data.audio_sr / self.config.data.pose_fps:
            context = context[:, :num_output_frames * self.config.data.audio_sr // self.config.data.pose_fps]
        else:
            context = F.pad(
                context,
                (0, num_output_frames * self.config.data.audio_sr // self.config.data.pose_fps - context.shape[1], 0, 0),
                mode="constant",
                value=0.0
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
        
        if self.kv_cache1 is None:
            self._initialize_kv_cache(
                batch_size=batch_size,
                dtype=noise.dtype,
                device=noise.device
            )
        else:
            for block_index in range(len(self.kv_cache1)):
                self.kv_cache1[block_index]["global_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noise.device)
                self.kv_cache1[block_index]["local_end_index"] = torch.tensor(
                    [0], dtype=torch.long, device=noise.device)
        
        context = self.context_encoding(context, 0, num_output_frames, num_output_frames)
        expression = self.speech_model(context, pid=context_pid)
        # Step 2: Cache context feature
        current_start_frame = 0
        if initial_latent is not None:
            timestep = torch.ones([batch_size, 1], device=noise.device, dtype=torch.int64) * 0
            if self.independent_first_frame:
                # Assume num_input_frames is 1 + self.num_frame_per_block * num_input_blocks
                assert (num_input_frames - 1) % self.num_frame_per_block == 0
                num_input_blocks = (num_input_frames - 1) // self.num_frame_per_block
                output[:, :1] = initial_latent[:, :1]
                context_ = context[:,:1]
                expression_ = expression[:,:1]
                expression_output[:, :1] = expression_
                generator(
                    initial_latent[:, :1],
                    timestep * 0,
                    context=context_,
                    speaker_id=context_pid,
                    expression=expression_,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length,
                    seq_len=num_output_frames,
                )
                current_start_frame += 1
            else:
                # Assume num_input_frames is self.num_frame_per_block * num_input_blocks
                assert num_input_frames % self.num_frame_per_block == 0
                num_input_blocks = num_input_frames // self.num_frame_per_block

            for _ in range(num_input_blocks):
                current_ref_latents = \
                    initial_latent[:, current_start_frame:current_start_frame + self.num_frame_per_block]
                output[:, current_start_frame:current_start_frame + self.num_frame_per_block] = current_ref_latents
                
                context_ = context[:,current_start_frame:current_start_frame + self.num_frame_per_block]
                expression_ = expression[:,current_start_frame:current_start_frame + self.num_frame_per_block]
                
                expression_output[:, current_start_frame:current_start_frame + self.num_frame_per_block]\
                    = expression_
                generator(
                    current_ref_latents,
                    timestep * 0,
                    context=context_,
                    speaker_id=context_pid,
                    expression=expression_,
                    kv_cache=self.kv_cache1,
                    crossattn_cache=self.crossattn_cache,
                    current_start=current_start_frame * self.frame_seq_length,
                    seq_len=num_output_frames,
                )
                current_start_frame += self.num_frame_per_block

        
        all_num_frames = [self.num_frame_per_block] * num_blocks
        # all_num_frames = [256] * (num_blocks // 4)
        if self.independent_first_frame and initial_latent is None:
            all_num_frames = [1] + all_num_frames
        # iter_num_frames = 0
        for current_num_frames in all_num_frames:
            if profile:
                block_start.record()

            noisy_input = noise[
                :, current_start_frame - num_input_frames:current_start_frame + current_num_frames - num_input_frames]

            
            context_ = context[:,current_start_frame:current_start_frame + current_num_frames]
            expression_ = expression[:,current_start_frame:current_start_frame + current_num_frames]
            expression_output[:, current_start_frame:current_start_frame + current_num_frames] = expression_
            # Step 3.1: Spatial denoising loop
            denoised_pred, extra = self.sampler.sample(
                generator,
                noisy_input, 
                context=context_,
                speaker_id=context_pid,
                return_intermediate=True,
                return_model_output=True,
                # post_process_func=lambda x: torch.cat([ 
                #     rc.NormalizeRot6d(x[:, :, :-3]),
                #     x[:, :, -3:],
                # ], dim=-1),
                expression=expression_,
                kv_cache=self.kv_cache1,
                crossattn_cache=self.crossattn_cache,
                current_start=current_start_frame * self.frame_seq_length,
                seq_len=num_output_frames,
                # clean_motion=kwargs["tar_motion"][:,current_start_frame:current_start_frame + current_num_frames],
            )
            denoised_pred = torch.cat([
                rc.NormalizeRot6d(denoised_pred[:, :, :-3]),
                denoised_pred[:, :, -3:],
            ], dim=-1)
            
            output[:, current_start_frame:current_start_frame + current_num_frames] = denoised_pred
            
            # Step 3.3: rerun with timestep zero to update KV cache using clean context
            context_timestep = torch.ones(
                [batch_size, current_num_frames],
                device=noise.device,
                dtype=torch.long
            ) * self.config.scheduler.get("context_noise", 0.0)
            generator(
                denoised_pred,
                context_timestep,
                context=context_,
                speaker_id=context_pid,
                expression=expression_,
                kv_cache=self.kv_cache1,
                crossattn_cache=self.crossattn_cache,
                current_start=current_start_frame * self.frame_seq_length,
                seq_len=num_output_frames,
            )

            if profile:
                block_end.record()
                torch.cuda.synchronize()
                block_time = block_start.elapsed_time(block_end)
                block_times.append(block_time)

            # Step 3.4: update the start and end frame indices
            current_start_frame += current_num_frames

        if profile:
            # End diffusion timing and synchronize CUDA
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