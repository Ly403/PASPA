import torch
import time
import numpy as np
import os
import os.path as osp
import librosa
import smplx
from diffusion.trainer import Trainer
from diffusion.utils import rotation_conversions as rc
from diffusion.model.respace import compute_density_for_timestep_sampling
from diffusion import OmniFlowScheduler
from trimesh import Trimesh
import matplotlib.pyplot as plt
from huggingface_hub import hf_hub_download
from vocos import Vocos
from diffusion.utils.config import set_value
from termcolor import colored
import soundfile as sf
import wget
from diffusion.model.audio_encoder.mel_spec import MelSpec
from diffusion.metrics.emage import EmageMetric
import math
try:
    import diffusion.utils.fast_render as fast_render
    import diffusion.utils.npz2pose as npz2pose
except:
    fast_render = None
    npz2pose = None

class MotionTrainer(Trainer):
    def __init__(
        self,
        accelerator,
        data,
        model_config,
        config,
        num_replicas=1,
        logger=None,
        generator=None,
    ):
        super().__init__(
            accelerator,
            data,
            model_config,
            config,
            num_replicas=num_replicas,
            logger=logger,
            generator=generator,
        )
        self.joints = self.config.data.joints # self.data._train_dataset.joints
        self.smplx = self.configure_smplx()
        self.vocoder_name = self.config.audio.audio_vocoder.vocoder_name
        self.vocoder = self.configure_vocoder()
        self.mel_spec = self.configure_mel_spec()
        self.evaluator = self.configure_evaluator_post()
        self.check_task()
        self.prepare_save_dir()
    
    
    def prepare_save_dir(self):
        # 把生成文件夹都放到这里来，防止下面每一次重新检验
        self.npz_save_name = self.config.train.get("npz_save_dir", "npz")
        self.wav_save_name = self.config.train.get("wav_save_dir", "wav")
        self.mp4_save_name = self.config.train.get("mp4_save_dir", "mp4")
        self.obj_save_name = self.config.train.get("obj_save_dir", "obj")
        self.png_save_name = self.config.train.get("png_save_dir", "png")
        os.makedirs(osp.join(self.config.work_dir, self.npz_save_name), exist_ok=True)
        os.makedirs(osp.join(self.config.work_dir, self.wav_save_name), exist_ok=True)
        os.makedirs(osp.join(self.config.work_dir, self.mp4_save_name), exist_ok=True)
        os.makedirs(osp.join(self.config.work_dir, self.obj_save_name), exist_ok=True)
        os.makedirs(osp.join(self.config.work_dir, self.png_save_name), exist_ok=True)
        if self.config._validation or self.config.train.eval:
            os.makedirs(osp.join(self.config.work_dir, self.npz_save_name, "validation"), exist_ok=True)
            os.makedirs(osp.join(self.config.work_dir, self.wav_save_name, "validation"), exist_ok=True)
            os.makedirs(osp.join(self.config.work_dir, self.mp4_save_name, "validation"), exist_ok=True)
            os.makedirs(osp.join(self.config.work_dir, self.obj_save_name, "validation"), exist_ok=True)
            os.makedirs(osp.join(self.config.work_dir, self.png_save_name, "validation"), exist_ok=True)
        if self.config._test:
            os.makedirs(osp.join(self.config.work_dir, self.npz_save_name, "test"), exist_ok=True)
            os.makedirs(osp.join(self.config.work_dir, self.wav_save_name, "test"), exist_ok=True)
            os.makedirs(osp.join(self.config.work_dir, self.mp4_save_name, "test"), exist_ok=True)
            os.makedirs(osp.join(self.config.work_dir, self.obj_save_name, "test"), exist_ok=True)
            os.makedirs(osp.join(self.config.work_dir, self.png_save_name, "test"), exist_ok=True)
   
    
    def check_task(self):
        task = self.config.train.task
        assert task in [
            "text2all", 
            "text2motion", "text2expression", "text2audio"
            "text2motion+expression", "text2motion+audio", "text2expression+audio",
            "text+audio2motion", "text+audio2expression","text+audio2motion+expression",
            "text+motion2expression", "text+motion2audio","text+motion2expression+audio",
            "text+expression2motion", "text+expression2audio","text+expression2motion+audio",
            "text+motion+expression2audio", "text+motion+audio2expression", "text+expression+audio2motion",
            ################################################################################################
            "uncondition2all",
            "uncondition2motion", "uncondition2expression", "uncondition2audio"
            "uncondition2motion+expression", "uncondition2motion+audio", "uncondition2expression+audio",
            "audio2motion", "audio2expression","audio2motion+expression",
            "motion2expression", "motion2audio","motion2expression+audio",
            "expression2motion", "expression2audio","expression2motion+audio",
            "motion+expression2audio", "motion+audio2expression", "expression+audio2motion",
        ], f"Invalid task: {task}"

    def configure_smplx(self):
        smplx_model_dir = osp.abspath(
            osp.expanduser(
                osp.join(self.config.data.data_path_1, "smplx_models")
            )
        )
        # smplx.create(root, ...) resolves models from root/smplx/*.npz.
        # The old downloader wrote directly under root, so a fresh machine
        # could download successfully and still fail during model creation.
        smplx_model_file_path = osp.join(
            smplx_model_dir,
            "smplx",
            "SMPLX_NEUTRAL_2020.npz",
        )
        if not osp.isfile(smplx_model_file_path):
            os.makedirs(osp.dirname(smplx_model_file_path), exist_ok=True)
            self.logger.info(f"Downloading {smplx_model_file_path}")
            wget.download(
                "https://huggingface.co/spaces/H-Liu1997/EMAGE/resolve/main/EMAGE/smplx_models/smplx/SMPLX_NEUTRAL_2020.npz",
                smplx_model_file_path,
            )
        if (
            not osp.isfile(smplx_model_file_path)
            or osp.getsize(smplx_model_file_path) == 0
        ):
            raise FileNotFoundError(
                "SMPL-X neutral model is missing or empty: "
                f"{smplx_model_file_path}"
            )
        
        smplx_model = smplx.create(
            smplx_model_dir,
            model_type='smplx',
            gender='NEUTRAL_2020', 
            use_face_contour=False,
            num_betas=300,
            num_expression_coeffs=100, 
            ext='npz',
            use_pca=False,
        ).to(self.device).eval()
        return smplx_model

    def configure_evaluator(self):
        return None
    
    def configure_evaluator_post(self):
        evaluator = EmageMetric(
            **self.config.metrics.evaluator,
            accelerator=self.accelerator,
            smplx=self.smplx,
            device=self.device,
        )
        self.logger.info(
            colored(
                f"Using EmageMetric Evaluator: {evaluator.__class__.__name__}",
                "yellow",
                attrs=["bold"],
            )
        )
        return evaluator
    
    def configure_mel_spec(self):
        mel_spec = MelSpec(**self.config.audio.mel_spec)
        self.logger.info(
            colored(
                f"Mel Spec: "
                f"{mel_spec.__class__.__name__}:{self.vocoder_name}, "
                f"Model Parameters: {sum(p.numel() for p in mel_spec.parameters()) / 1e6:.2f}M",
                "green",
                attrs=["bold"],
            )
        )
        return mel_spec

    def configure_vocoder(self):
        if self.vocoder_name == "vocos":
            # vocoder = Vocos.from_pretrained("charactr/vocos-mel-24khz").to(device)
            if self.config.audio.audio_vocoder.get("local", None) is not None:
                self.logger.info(f"Load vocos from local path {self.config.audio.audio_vocoder.local}")
                config_path = f"{self.config.audio.audio_vocoder.local}/config.yaml"
                model_path = f"{self.config.audio.audio_vocoder.local}/pytorch_model.bin" 
            else:
                self.logger.info("Download Vocos from huggingface charactr/vocos-mel-24khz")
                repo_id = "charactr/vocos-mel-24khz"
                config_path = hf_hub_download(repo_id=repo_id, filename="config.yaml")
                model_path = hf_hub_download(repo_id=repo_id, filename="pytorch_model.bin")
            vocoder = Vocos.from_hparams(config_path)
            state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
            from vocos.feature_extractors import EncodecFeatures

            if isinstance(vocoder.feature_extractor, EncodecFeatures):
                encodec_parameters = {
                    "feature_extractor.encodec." + key: value
                    for key, value in vocoder.feature_extractor.encodec.state_dict().items()
                }
                state_dict.update(encodec_parameters)
            vocoder.load_state_dict(state_dict)
            vocoder = vocoder.eval().to(self.device)
        
        elif self.vocoder_name == "bigvgan":
            try:
                from diffusion.model.audio_encoder.BigVGAN import bigvgan
            except ImportError:
                self.logger.error("You need to follow the README to init submodule and change the BigVGAN source code.")
            if self.config.audio.audio_vocoder.get("local", None) is not None:
                # download generator from https://huggingface.co/nvidia/bigvgan_v2_24khz_100band_256x/tree/main
                vocoder = bigvgan.BigVGAN.from_pretrained(self.config.audio.audio_vocoder.local, use_cuda_kernel=False)
            else:
                vocoder = bigvgan.BigVGAN.from_pretrained(
                    "nvidia/bigvgan_v2_24khz_100band_256x", use_cuda_kernel=False
                )

            vocoder.remove_weight_norm()
            vocoder = vocoder.eval().to(self.device)
        
        elif self.vocoder_name == "hifigan":
            from speechbrain.inference.vocoders import HIFIGAN
            if self.config.audio.audio_vocoder.get("local", None) is not None:
                vocoder = HIFIGAN.from_hparams(
                    source="speechbrain/tts-hifigan-libritts-16kHz", 
                    savedir=self.config.audio.audio_vocoder.local,
                    run_opts={"device": self.device}
                )
            else:
                vocoder = HIFIGAN.from_hparams(
                    source="speechbrain/tts-hifigan-libritts-16kHz", 
                    savedir="pretrained_models/tts-hifigan-libritts-16kHz",
                    run_opts={"device": self.device}
                )
            vocoder = vocoder.eval()
        else:
            raise ValueError(f"Invalid vocoder: {self.vocoder_name}")
        
        self.logger.info(
            colored(
                f"Vocoder: "
                f"{vocoder.__class__.__name__}:{self.vocoder_name}, "
                f"Model Parameters: {sum(p.numel() for p in vocoder.parameters()) / 1e6:.2f}M",
                "green",
                attrs=["bold"],
            )
        )
        return vocoder
    
    def mel_to_wav(self, mel):
        if self.vocoder_name == "vocos":
            wav = self.vocoder.decode(mel)
        elif self.vocoder_name == "bigvgan":
            wav = self.vocoder(mel)
        elif self.vocoder_name == "hifigan":
            wav = self.vocoder.decode_batch(mel).squeeze(1)
        else:
            raise ValueError(f"Invalid vocoder: {self.vocoder_name}")
        return wav
    
    def configure_scheduler(self):
        set_value(self.config.scheduler.scheduler, "timestep_respacing",\
            str(self.config.scheduler.train_sampling_steps))
        diffusion = OmniFlowScheduler(
            **self.config.scheduler.scheduler,
        )
        return diffusion
    
    def prepare_data(self, batch, step, null_embed = None, lm_time_all=None, am_time_all=None, is_predict=False):
        j = self.joints
        tar_pose_raw = self.get_input(batch, "pose") 
        tar_pose = tar_pose_raw[:, :, :j*3].to(self.device)
        tar_contact = tar_pose_raw[:, :, j*3:j*3+4].to(self.device)
        tar_trans = self.get_input(batch, "trans").to(self.device)
        tar_exps = self.get_input(batch, "facial").to(self.device)
        in_audio = self.get_input(batch, "audio").to(self.device) 
        tar_beta = self.get_input(batch, "beta").to(self.device)
        tar_id = self.get_input(batch, "id").to(self.device).long()
        ori_id = self.get_input(batch, "ori_vid")
        # preprocess the tar_pose, switch to 6d representation
        bs, n = tar_pose.shape[0], tar_pose.shape[1]
        tar_pose_6d = rc.axis_angle_to_matrix(tar_pose.reshape(bs, n, j, 3))
        tar_pose_6d = rc.matrix_to_rotation_6d(tar_pose_6d).reshape(bs, n, j*6)
        
        tar_motion = torch.cat([tar_pose_6d, tar_trans, tar_contact], dim=-1)
        
        ####################### for text encoder ###############################
        if lm_time_all is not None:
            lm_time_start = time.time()
        if not self.config.train.task.startswith("text"):
            in_word = null_embed.unsqueeze(0).repeat(bs, *[1] * len(null_embed.shape))
            in_word_mask = None
        else:
            if is_predict:
                in_word = self.get_input(batch, "word")
                in_word_txt = in_word # 保留原始文本
                in_word_mask = self.get_input(batch, "word_mask")
            elif self.load_text_feat:
                in_word = self.get_input(batch, "word").to(self.device)  # bs, 1, N, C
                in_word_mask = self.get_input(batch, "word_mask")
                if in_word_mask is not None:
                    in_word_mask = in_word_mask.to(self.device)
            else:
                in_word = self.get_input(batch, "word")
                if "T5" in self.config.text_encoder.text_encoder_name:
                    with torch.no_grad():
                        txt_tokens = self.tokenizer(
                            in_word, max_length=self.config.text_encoder.model_max_length,\
                                padding="max_length", truncation=True, return_tensors="pt"
                        ).to(self.device)
                        untruncated_tokens = self.tokenizer(
                            in_word, padding="longest", truncation=False, return_tensors="pt"
                        ).to(self.device)
                        if untruncated_tokens.input_ids.shape[-1] > txt_tokens.input_ids.shape[-1] \
                            and not torch.equal(untruncated_tokens.input_ids, txt_tokens.input_ids):
                            self.logger.warning(
                                colored(
                                    f"The original text {in_word}, " 
                                    f"is longer than the max length {self.config.text_encoder.model_max_length}, " 
                                    f"so we truncate the text.",
                                    "yellow",
                                    attrs=["bold"],
                                ),
                            )
                            
                        in_word = self.text_encoder(txt_tokens.input_ids, attention_mask=txt_tokens.attention_mask)[0][:, None]
                        in_word_mask = txt_tokens.attention_mask[:, None, None]
                elif "clip" in self.config.text_encoder.text_encoder_name:
                    with torch.no_grad():
                        txt_tokens = self.tokenizer(
                            in_word, max_length=self.config.text_encoder.model_max_length,\
                                padding="max_length", truncation=True, return_tensors="pt"
                        ).to(self.device)
                        untruncated_tokens = self.tokenizer(
                            in_word, padding="longest", truncation=False, return_tensors="pt"
                        ).to(self.device)
                        if untruncated_tokens.input_ids.shape[-1] > txt_tokens.input_ids.shape[-1] \
                            and not torch.equal(untruncated_tokens.input_ids, txt_tokens.input_ids):
                            self.logger.warning(
                                colored(
                                    f"The original text {in_word}, " 
                                    f"is longer than the max length {self.config.text_encoder.model_max_length}, " 
                                    f"so we truncate the text.",
                                    "yellow",
                                    attrs=["bold"],
                                ),
                            )
                        in_word = self.text_encoder(
                        txt_tokens.input_ids, attention_mask=txt_tokens.attention_mask, output_hidden_states=True
                        )
                        if self.config.text_encoder.use_last_hidden_states:
                            in_word = in_word.last_hidden_state
                            in_word_mask = txt_tokens.attention_mask
                        else:
                            in_word = in_word[0][:, None]
                            in_word_mask = None                    
                elif (
                    "gemma" in self.config.text_encoder.text_encoder_name or "Qwen" in self.config.text_encoder.text_encoder_name
                ):
                    with torch.no_grad():
                        if not self.config.text_encoder.chi_prompt:
                            max_length_all = self.config.text_encoder.model_max_length
                            prompt = in_word
                        else:
                            chi_prompt = "\n".join(self.config.text_encoder.chi_prompt)
                            prompt = [chi_prompt + i for i in in_word]
                            num_sys_prompt_tokens = len(self.tokenizer.encode(chi_prompt))
                            max_length_all = (
                                num_sys_prompt_tokens + self.config.text_encoder.model_max_length - 2
                            )  # magic number 2: [bos], [_]
                        txt_tokens = self.tokenizer(
                            prompt,
                            padding="max_length",
                            max_length=max_length_all,
                            truncation=True,
                            return_tensors="pt",
                        ).to(self.device)
                        
                        untruncated_tokens = self.tokenizer(
                            prompt, padding="longest", truncation=False, return_tensors="pt"
                        ).to(self.device)
                        if untruncated_tokens.input_ids.shape[-1] > txt_tokens.input_ids.shape[-1] \
                            and not torch.equal(untruncated_tokens.input_ids, txt_tokens.input_ids):
                            self.logger.warning(
                                colored(
                                    f"The original text {prompt}, " 
                                    f"is longer than the max length {self.config.text_encoder.model_max_length}, " 
                                    f"so we truncate the text.",
                                    "yellow",
                                    attrs=["bold"],
                                ),
                            )
                        
                        select_index = [0] + list(
                            range(-self.config.text_encoder.model_max_length + 1, 0)
                        )  # first bos and end N-1
                        in_word = self.text_encoder(txt_tokens.input_ids, attention_mask=txt_tokens.attention_mask)[0]
                        # in_word = in_word[:, None][
                        #     :, :, select_index
                        # ]
                        # in_word_mask = txt_tokens.attention_mask[:, None, None][:, :, :, select_index]
                        in_word = in_word[:, select_index]
                        in_word_mask = txt_tokens.attention_mask[:, select_index]
                else:
                    self.logger.error("error")
                    exit()
        self.accelerator.wait_for_everyone()
        if lm_time_all is not None:
            lm_time_all += time.time() - lm_time_start
        ########################################################################
        
        return {
            "in_audio": in_audio,
            "in_word": in_word,
            "in_word_mask": in_word_mask,
            "tar_exps": tar_exps,
            "tar_trans": tar_trans,
            "tar_contact": tar_contact,
            "tar_beta": tar_beta,
            "tar_pose": tar_pose,
            "tar_pose_6d": tar_pose_6d,
            "tar_motion": tar_motion,
            "tar_id": tar_id,
            "ori_id": ori_id,
            "lm_time_all": lm_time_all,
        }
    
    def sample_timestep(self, bs):
        timesteps = torch.randint(
            0, self.config.scheduler.train_sampling_steps, (bs,), device=self.device,
            generator=self.generator
        ).long()
        if self.config.scheduler.weighting_scheme in ["logit_normal"]:
            # adapting from diffusers.training_utils
            u = compute_density_for_timestep_sampling(
                weighting_scheme=self.config.scheduler.weighting_scheme,
                batch_size=bs,
                logit_mean=self.config.scheduler.logit_mean,
                logit_std=self.config.scheduler.logit_std,
                mode_scale=None,  # not used
            )
            timesteps = (u * self.config.scheduler.train_sampling_steps).long().to(self.device)
        return timesteps
    
    def sample_all_timesteps(self, bs):
        task = self.config.train.task
        max_timestep = self.config.scheduler.train_sampling_steps - 1
        if task in ["text2all", "uncondition2all"]:
            motion_timesteps = self.sample_timestep(bs)
            expression_timesteps = self.sample_timestep(bs)
            audio_timesteps = self.sample_timestep(bs)
        # =========================================================================
        elif task in ["text2motion", "uncondition2motion"]:
            motion_timesteps = self.sample_timestep(bs)
            expression_timesteps = torch.ones_like(motion_timesteps) * max_timestep
            audio_timesteps = torch.ones_like(motion_timesteps) * max_timestep
        elif task in ["text2expression", "uncondition2expression"]:
            expression_timesteps = self.sample_timestep(bs)
            motion_timesteps = torch.ones_like(expression_timesteps) * max_timestep
            audio_timesteps = torch.ones_like(expression_timesteps) * max_timestep
        elif task in ["text2audio", "uncondition2audio"]:
            audio_timesteps = self.sample_timestep(bs)
            motion_timesteps = torch.ones_like(audio_timesteps) * max_timestep
            expression_timesteps = torch.ones_like(audio_timesteps) * max_timestep
        # =========================================================================
        elif task in ["text2motion+expression", "uncondition2motion+expression"]:
            motion_timesteps = self.sample_timestep(bs)
            expression_timesteps = self.sample_timestep(bs)
            audio_timesteps = torch.ones_like(motion_timesteps) * max_timestep
        elif task in ["text2motion+audio", "uncondition2motion+audio"]:
            motion_timesteps = self.sample_timestep(bs)
            audio_timesteps = self.sample_timestep(bs)
            expression_timesteps = torch.ones_like(motion_timesteps) * max_timestep
        elif task in ["text2expression+audio", "uncondition2expression+audio"]:
            expression_timesteps = self.sample_timestep(bs)
            audio_timesteps = self.sample_timestep(bs)
            motion_timesteps = torch.ones_like(expression_timesteps) * max_timestep
        # =========================================================================
        elif task in ["text+audio2motion", "audio2motion"]:
            motion_timesteps = self.sample_timestep(bs)
            audio_timesteps = torch.zeros_like(motion_timesteps)
            expression_timesteps = torch.ones_like(motion_timesteps) * max_timestep
        elif task in ["text+audio2expression", "audio2expression"]:
            expression_timesteps = self.sample_timestep(bs)
            audio_timesteps = torch.zeros_like(expression_timesteps)
            motion_timesteps = torch.ones_like(expression_timesteps) * max_timestep
        elif task in ["text+motion2expression", "motion2expression"]:
            expression_timesteps = self.sample_timestep(bs)
            motion_timesteps = torch.zeros_like(expression_timesteps)
            audio_timesteps = torch.ones_like(expression_timesteps) * max_timestep
        elif task in ["text+motion2audio", "motion2audio"]:
            audio_timesteps = self.sample_timestep(bs)
            motion_timesteps = torch.zeros_like(audio_timesteps)
            expression_timesteps = torch.ones_like(audio_timesteps) * max_timestep
        elif task in ["text+expression2motion", "expression2motion"]:
            motion_timesteps = self.sample_timestep(bs)
            expression_timesteps = torch.zeros_like(motion_timesteps)
            audio_timesteps = torch.ones_like(motion_timesteps) * max_timestep
        elif task in ["text+expression2audio", "expression2audio"]:
            audio_timesteps = self.sample_timestep(bs)
            expression_timesteps = torch.zeros_like(audio_timesteps)
            motion_timesteps = torch.ones_like(audio_timesteps) * max_timestep
        # =========================================================================
        elif task in ["text+motion+expression2audio", "motion+expression2audio"]:
            audio_timesteps = self.sample_timestep(bs)
            motion_timesteps = torch.zeros_like(audio_timesteps)
            expression_timesteps = torch.zeros_like(audio_timesteps)
        elif task in ["text+motion+audio2expression", "motion+audio2expression"]:
            expression_timesteps = self.sample_timestep(bs)
            motion_timesteps = torch.zeros_like(expression_timesteps)
            audio_timesteps = torch.zeros_like(expression_timesteps) 
        elif task in ["text+expression+audio2motion", "expression+audio2motion"]:
            motion_timesteps = self.sample_timestep(bs)
            expression_timesteps = torch.zeros_like(motion_timesteps)
            audio_timesteps = torch.zeros_like(motion_timesteps)
        # =========================================================================
        elif task in ["text+audio2motion+expression", "audio2motion+expression"]:
            motion_timesteps = self.sample_timestep(bs)
            expression_timesteps = self.sample_timestep(bs)
            audio_timesteps = torch.zeros_like(motion_timesteps)
        elif task in ["text+motion2expression+audio", "motion2expression+audio"]:
            expression_timesteps = self.sample_timestep(bs)
            audio_timesteps = self.sample_timestep(bs)
            motion_timesteps = torch.zeros_like(expression_timesteps)
        elif task in ["text+expression2motion+audio", "expression2motion+audio"]:
            motion_timesteps = self.sample_timestep(bs)
            audio_timesteps = self.sample_timestep(bs)
            expression_timesteps = torch.zeros_like(motion_timesteps)
        
        else:
            raise ValueError(f"Invalid task: {task}")
        return motion_timesteps, expression_timesteps, audio_timesteps
    
    def first_stage_encode(self, tar_motion, tar_exps, in_audio, vae_time_all = None):
        ####################### for vae ###############################
        if vae_time_all is not None:
            vae_time_start = time.time()
        
        if self.config.train.train_first_stage_model:
            results = self.first_stage_model.encode(motion=tar_motion, expression=tar_exps)
        else:
            with torch.no_grad():
                results = self.first_stage_model.encode(motion=tar_motion, expression=tar_exps)
        
        in_audio_mel = self.mel_spec(in_audio).transpose(-1, -2)
        if self.config.train.train_audio_vae_model:
            latent_audio = self.audio_vae.encode(audio=in_audio_mel)["audio"]
        else:
            with torch.no_grad():
                latent_audio = self.audio_vae.encode(audio=in_audio_mel)["audio"]
        
        if vae_time_all is not None:
            self.accelerator.wait_for_everyone()
            vae_time_all += time.time() - vae_time_start
        
        latent_motion = results["motion"]
        latent_expression = results["expression"]

        latent_motion_index = results.get("motion_index", None)
        latent_expression_index = results.get("expression_index", None)
        ########################################################################
        return {
            "latent_motion": latent_motion,
            "latent_expression": latent_expression,
            "latent_motion_index": latent_motion_index,
            "latent_expression_index": latent_expression_index,
            "latent_audio": latent_audio,
            "in_audio_mel": in_audio_mel,
            "vae_time_all": vae_time_all,
        }
    
    def first_stage_decode(self, latent_motion, latent_expression, latent_audio):
        if self.load_vae_feat:
            results = {}
            results["motion"] = latent_motion[..., :self.joints*6]
            results["expression"] = latent_expression[..., :100]
            results["trans"] = latent_motion[..., self.joints*6:self.joints*6+3]
            results["contact"] = latent_motion[..., self.joints*6+3:self.joints*6+4]
        else:
            if self.config.train.train_first_stage_model:
                results = self.first_stage_model.decode(motion=latent_motion, expression=latent_expression)
            else:
                with torch.no_grad():
                    results = self.first_stage_model.decode(motion=latent_motion, expression=latent_expression)
        
        if self.config.train.train_audio_vae_model:
            latent_audio = self.audio_vae.decode(audio=latent_audio)["audio"]
        else:
            with torch.no_grad():
                latent_audio = self.audio_vae.decode(audio=latent_audio)["audio"]

        return results, latent_audio
    
    def training_step(self, batch, step, vae_time_all=None, global_step=None): 
        in_audio = self.get_input(batch, "in_audio")
        in_word = self.get_input(batch, "in_word")
        in_word_mask = self.get_input(batch, "in_word_mask")
        in_id = self.get_input(batch, "tar_id")
        tar_motion = self.get_input(batch, "tar_motion")
        tar_exps = self.get_input(batch, "tar_exps")
        tar_beta = self.get_input(batch, "tar_beta")
        results = self.first_stage_encode(tar_motion, tar_exps, in_audio, vae_time_all)
        latent_motion = results["latent_motion"]
        latent_expression = results["latent_expression"]
        latent_audio = results["latent_audio"]
        vae_time_all = results["vae_time_all"]
        # Sample a random timestep for each modality
        bs = tar_motion.shape[0]
        motion_timestep, expression_timestep, audio_timestep = self.sample_all_timesteps(bs)
        #########################################################
        model_fn = lambda *args, **kwargs: self.model(**self.guider(*args, **kwargs, is_sample=False, task=self.config.train.task))
        loss_term = self.diffusion.training_losses(
            model_fn, 
            latent_motion, latent_expression, latent_audio, 
            motion_timestep, expression_timestep, audio_timestep, 
            model_kwargs=dict(txt=in_word, txt_mask=in_word_mask, speaker_id=in_id),
            smpl_model=self.smplx,
            tar_beta=tar_beta
        )
        return loss_term, vae_time_all
    
    def get_initial_noise(self, latent_motion, latent_expression, latent_audio):
        task = self.config.train.task
        if task in ["text2all", "uncondition2all"]:
            motion_noise = torch.randn_like(latent_motion)
            expression_noise = torch.randn_like(latent_expression)
            audio_noise = torch.randn_like(latent_audio)
        # =========================================================================
        elif task in ["text2motion", "uncondition2motion"]:
            motion_noise = self.randn_like(latent_motion)
            expression_noise = torch.zeros_like(latent_expression) # 无条件
            audio_noise = torch.zeros_like(latent_audio) # 无条件
        elif task in ["text2expression", "uncondition2expression"]:
            expression_noise = torch.randn_like(latent_expression)
            motion_noise = torch.zeros_like(latent_motion) # 无条件
            audio_noise = torch.zeros_like(latent_audio) # 无条件
        elif task in ["text2audio", "uncondition2audio"]:
            audio_noise = torch.randn_like(latent_audio)
            motion_noise = torch.zeros_like(latent_motion) # 无条件
            expression_noise = torch.zeros_like(latent_expression) # 无条件
        # =========================================================================
        elif task in ["text2motion+expression", "uncondition2motion+expression"]:
            motion_noise = torch.randn_like(latent_motion)
            expression_noise = torch.randn_like(latent_expression)
            audio_noise = torch.zeros_like(latent_audio) # 无条件
        elif task in ["text2motion+audio", "uncondition2motion+audio"]:
            motion_noise = torch.randn_like(latent_motion)
            audio_noise = torch.randn_like(latent_audio)
            expression_noise = torch.zeros_like(latent_expression) # 无条件
        elif task in ["text2expression+audio", "uncondition2expression+audio"]:
            expression_noise = torch.randn_like(latent_expression)
            audio_noise = torch.randn_like(latent_audio)
            motion_noise = torch.zeros_like(latent_motion) # 无条件
        # =========================================================================
        elif task in ["text+audio2motion", "audio2motion"]:
            motion_noise = torch.randn_like(latent_motion)
            audio_noise = latent_audio # 条件
            expression_noise = torch.zeros_like(latent_expression) # 无条件
        elif task in ["text+audio2expression", "audio2expression"]:
            expression_noise = torch.randn_like(latent_expression)
            audio_noise = latent_audio # 条件
            motion_noise = torch.zeros_like(latent_motion) # 无条件
        elif task in ["text+motion2expression", "motion2expression"]:
            expression_noise = torch.randn_like(latent_expression)
            motion_noise = latent_motion # 条件
            audio_noise = torch.zeros_like(latent_audio) # 无条件
        elif task in ["text+motion2audio", "motion2audio"]:
            audio_noise = torch.randn_like(latent_audio)
            motion_noise = latent_motion # 条件
            expression_noise = torch.zeros_like(latent_expression) # 无条件
        elif task in ["text+expression2motion", "expression2motion"]:
            motion_noise = torch.randn_like(latent_motion)
            expression_noise = latent_expression # 条件
            audio_noise = torch.zeros_like(latent_audio) # 无条件
        elif task in ["text+expression2audio", "expression2audio"]:
            audio_noise = torch.randn_like(latent_audio)
            expression_noise = latent_expression # 条件
            motion_noise = torch.zeros_like(latent_motion) # 无条件
        # =========================================================================
        elif task in ["text+motion+expression2audio", "motion+expression2audio"]:
            audio_noise = torch.randn_like(latent_audio)
            motion_noise = latent_motion # 条件
            expression_noise = latent_expression # 条件
        elif task in ["text+motion+audio2expression", "motion+audio2expression"]:
            expression_noise = torch.randn_like(latent_expression)
            motion_noise = latent_motion # 条件
            audio_noise = latent_audio # 条件
        elif task in ["text+expression+audio2motion", "expression+audio2motion"]:
            motion_noise = torch.randn_like(latent_motion)
            expression_noise = latent_expression # 条件
            audio_noise = latent_audio # 条件
        # =========================================================================
        elif task in ["text+audio2motion+expression", "audio2motion+expression"]:
            motion_noise = torch.randn_like(latent_motion)
            expression_noise = torch.randn_like(latent_expression)
            audio_noise = latent_audio # 条件
        elif task in ["text+motion2expression+audio", "motion2expression+audio"]:
            expression_noise = torch.randn_like(latent_expression)
            audio_noise = torch.randn_like(latent_audio)
            motion_noise = latent_motion # 条件
        elif task in ["text+expression2motion+audio", "expression2motion+audio"]:
            motion_noise = torch.randn_like(latent_motion)
            expression_noise = latent_expression # 条件
            audio_noise = torch.randn_like(latent_audio)
        
        else:
            raise ValueError(f"Invalid task: {task}")
        
        return motion_noise, expression_noise, audio_noise
    
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
    
        ## The same as training_step
        in_audio = self.get_input(batch, "in_audio") # 这个是wav
        in_word = self.get_input(batch, "in_word")
        in_word_mask = self.get_input(batch, "in_word_mask")
        in_id = self.get_input(batch, "tar_id")
        tar_motion = self.get_input(batch, "tar_motion")
        tar_trans = self.get_input(batch, "tar_trans")
        tar_exps = self.get_input(batch, "tar_exps")
        tar_beta = self.get_input(batch, "tar_beta")
        tar_pose = self.get_input(batch, "tar_pose_6d")
        ori_id = self.get_input(batch, "ori_id")
        # Sample a random timestep for each modality
        bs = tar_motion.shape[0]
        results = self.first_stage_encode(tar_motion, tar_exps, in_audio)
        latent_motion = results["latent_motion"]
        latent_expression = results["latent_expression"]
        latent_audio = results["latent_audio"] # 这个已经是mel谱了
        in_audio_mel = results["in_audio_mel"]
        if choice == "validation":
            motion_timestep, expression_timestep, audio_timestep = self.sample_all_timesteps(bs)
            #########################################################
            model_fn = lambda *args, **kwargs: \
                self.model_instance(**self.guider(*args, **kwargs, is_sample=False, task=self.config.train.task))
            loss_term = self.diffusion.training_losses(
                model_fn, 
                latent_motion, 
                latent_expression, 
                latent_audio, 
                motion_timestep, 
                expression_timestep, 
                audio_timestep, 
                model_kwargs=dict(
                    txt=in_word, 
                    txt_mask=in_word_mask, 
                    speaker_id=in_id
                ),
                smpl_model=self.smplx,
                tar_beta=tar_beta
            )
        ########################################################################
        
        model_kwargs = dict(
            txt=in_word, 
            txt_mask=in_word_mask, 
            speaker_id=in_id, 
            uncond_txt=null_embed.unsqueeze(0).repeat(bs, *[1] * len(in_word.shape[1:]))
        )
        
        def model_fn(*args, **kwargs):
            results =  self.model_instance(**self.guider(
                *args, **kwargs, model_kwargs=model_kwargs, is_sample=True, task=self.config.train.task))
            values = {}
            for key, value in results.items():
                if key in ["motion", "expression", "audio"]:
                    values[key] = value
            values = self.guider.guide(**values, task=self.config.train.task) # guidance here
            res = []
            for key, value in values.items():
                res.append(value)
            return res
                
        motion_noise, expression_noise, audio_noise = self.get_initial_noise(latent_motion, latent_expression, latent_audio)
        
        if_vis_intermediate = if_visualize and self.config.train.get("visualize_intermediate", False)
        if_vis_model_output = if_visualize and self.config.train.get("visualize_model_output", False)
        # if_vis_adding_noise = \
        #     self.config.train.visualize and \
        #     visualize and\
        #     step == adding_noise_vis_step and\
        #     self.config.train.get("visualize_adding_noise", False)
        if_vis_adding_noise = if_visualize and self.config.train.get("visualize_adding_noise", False)
        
        motion_latents, expression_latents, audio_latents, extra = self.sampler.sample(
            model_fn,
            motion_noise,
            expression_noise,
            audio_noise,
            return_intermediate=if_vis_intermediate,
            return_model_output=if_vis_model_output,
        )
        motion_expression_results, audio_results \
            = self.first_stage_decode(motion_latents, expression_latents, audio_latents)
         
        motion_results = motion_expression_results["motion"]
        expression_results = motion_expression_results["expression"]
        motion_trans = motion_expression_results["trans"]

        if "motion_axis_angle" in motion_expression_results:
            motion_results_axis_angle = motion_expression_results["motion_axis_angle"]\
                .view(bs, -1, 3 * self.joints)
        else:
            motion_results_axis_angle = rc.rotation_6d_to_axis_angle(motion_results.view(bs, motion_results.shape[1], -1, 6))\
                .view(bs, -1, 3 * self.joints)
        tar_motion_axis_angle = rc.rotation_6d_to_axis_angle(
            tar_motion[..., :6*self.joints].view(bs, tar_motion.shape[1], -1, 6))\
                .view(bs, -1, 3 * self.joints)

        audio_results = audio_results.transpose(-1, -2)
        in_audio_mel = in_audio_mel.transpose(-1, -2)
        audio_wav_gt = in_audio
        audio_wav_results = self.mel_to_wav(audio_results)

        # TODO: visualize the motion, expression, audio
        vis_bs = self.config.train.get("visualize_batch_size", bs)
        vis_bs = min(vis_bs, bs)
        if self.accelerator.is_main_process: # 这里相当于是只让主进程进行可视化，多进程可视化自然也是可以实现的，就是有点麻烦
            if if_visualize:
                self.visualize_motion(
                    motion_results_axis_angle, 
                    expression_results,  
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
                self.visualize_audio(
                    audio_results, 
                    in_audio_mel, 
                    audio_wav_results, 
                    audio_wav_gt,
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
                
                
            if if_vis_model_output:
                self.visualize_sequence(
                    extra["motion_model_outputs"],
                    extra["expression_model_outputs"],
                    extra["audio_model_outputs"],
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
                    extra["expression_intermediates"],
                    extra["audio_intermediates"],
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
                adding_noise_motion, adding_noise_expression, adding_noise_audio =\
                self.sampler.get_adding_noise_sequence(
                    latent_motion,
                    latent_expression,
                    latent_audio
                )
                self.visualize_sequence(
                    adding_noise_motion,
                    adding_noise_expression,
                    adding_noise_audio,
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
            if choice == "test":
                audio_sr = self.config.data.test.params.audio_sr
            elif choice == "validation":
                audio_sr = self.config.data.validation.params.audio_sr
            else:
                raise ValueError(f"Invalid choice: {choice}")
            prefix = f"gs{global_step:06d}_" if global_step is not None else ""
            prefix += f"epoch{epoch:06d}_" if epoch is not None else ""
            prefix += f"step{step:06d}_" if step is not None else ""
            tar_beta_np = tar_beta.detach().cpu().numpy()
            rec_pose_np = motion_results_axis_angle.detach().cpu().numpy()
            rec_exp_np = expression_results.detach().cpu().numpy()
            rec_trans_np = motion_trans.detach().cpu().numpy()
            tar_pose_np = tar_motion_axis_angle.detach().cpu().numpy()
            tar_exp_np = tar_exps.detach().cpu().numpy()
            tar_trans_np = tar_trans.detach().cpu().numpy()
            
            res_paths = []
            gt_paths = []
            res_audio_paths = []
            gt_audio_paths = []
            
            for i in range(bs):
                prefix += f"bs{i}_"
                if not keep_every_epoch:
                    prefix = ""
                res_path = osp.join(self.config.work_dir, self.npz_save_name, choice,
                            f"rank{self.accelerator.process_index}_{prefix}resface_" + ori_id[i] + ".npz")
                gt_path = osp.join(self.config.work_dir, self.npz_save_name, choice,
                            f"rank{self.accelerator.process_index}_{prefix}gt_" + ori_id[i] + ".npz")
                res_paths.append(res_path)
                gt_paths.append(gt_path)
                res_audio_path = osp.join(self.config.work_dir, self.wav_save_name, choice,
                            f"rank{self.accelerator.process_index}_{prefix}res_sr{audio_sr}_" + ori_id[i] + ".wav")
                gt_audio_path = osp.join(self.config.work_dir, self.wav_save_name, choice,
                            f"rank{self.accelerator.process_index}_{prefix}gt_sr{audio_sr}_" + ori_id[i] + ".wav")
                res_audio_paths.append(res_audio_path)
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
                    res_audio_path,
                    audio_wav_results[i].detach().cpu().numpy(),
                    samplerate=audio_sr,
                )
                sf.write(
                    gt_audio_path,
                    audio_wav_gt[i].detach().cpu().numpy(),
                    samplerate=audio_sr,
                )
            del tar_beta_np, rec_pose_np, rec_exp_np, rec_trans_np, tar_pose_np, tar_exp_np, tar_trans_np
                
                         
            results = {
                "motion": res_paths,
                "expression": res_paths,
                "audio": res_audio_paths,
                "gt_motion": gt_paths,
                "gt_expression": gt_paths,
                "gt_audio": gt_audio_paths,
                "betas": gt_paths,
            }
        else:
            results = {
                "motion": motion_results_axis_angle,
                "expression": expression_results,
                "audio": audio_wav_results,
                "gt_motion": tar_motion_axis_angle,
                "gt_expression": tar_exps,
                "gt_audio": audio_wav_gt,
                "betas": tar_beta
            }

        if choice == "validation":
            return loss_term, results
        elif choice == "test":
            return results
        else:
            raise ValueError(f"Invalid choice: {choice}")
    
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
        if_visualize = self.config.train.visualize and visualize and (
            (step + 1) % log_interval== 0 or (step + 1) == 1
        )
        if_vis_intermediate = if_visualize and self.config.train.get("visualize_intermediate", False)
        if_vis_model_output = if_visualize and self.config.train.get("visualize_model_output", False)
        if_vis_adding_noise = if_visualize and self.config.train.get("visualize_adding_noise", False)
        ## The same as training_step
        in_audio = self.get_input(batch, "in_audio") # 这个是wav
        in_word = self.get_input(batch, "in_word")
        in_word_mask = self.get_input(batch, "in_word_mask")
        in_id = self.get_input(batch, "tar_id")
        tar_motion = self.get_input(batch, "tar_motion")
        tar_trans = self.get_input(batch, "tar_trans")
        tar_exps = self.get_input(batch, "tar_exps")
        tar_beta = self.get_input(batch, "tar_beta")
        tar_pose = self.get_input(batch, "tar_pose_6d")
        ori_id = self.get_input(batch, "ori_id")
        bs = tar_motion.shape[0]
        
        tar_motion_axis_angle = rc.rotation_6d_to_axis_angle(
            tar_pose[..., :6*self.joints].view(bs, tar_pose.shape[1], -1, 6))\
                .view(bs, -1, 3 * self.joints
        )
                
        pose_length = self.config.data.pose_length # 这个是训练的pose长度，test必须以这个为基础单位进行
        test_length = tar_motion.shape[1] # 实际的长度
        overlap_length = self.config.scheduler.overlap_length
        pose_fps = self.config.data.train.params.pose_fps # 我们要对齐train的fps
        # if choice == "predict":
        #     sr = self.config.data.predict.params.audio_sr
        # elif choice == "test":
        #     sr = self.config.data.test.params.audio_sr
        # else:
        #     raise ValueError(f"Invalid choice: {choice}")
        sr = self.config.data.audio_sr
        
        # 留好位置
        motion_results_all = torch.empty_like(tar_motion_axis_angle)
        expression_results_all = torch.empty_like(tar_exps)
        motion_trans_all = torch.empty_like(tar_trans)
        audio_results_all = torch.empty_like(in_audio)
        for start in range(0, test_length, pose_length - overlap_length):
            end = start + pose_length
            if end > test_length:
                end = test_length
            audio_start = math.floor(start * sr / pose_fps)
            audio_end = math.floor(end * sr / pose_fps)
            tar_motion_ = tar_motion[:, start:end] if tar_motion is not None else None
            tar_exps_ = tar_exps[:, start:end] if tar_exps is not None else None
            in_audio_ = in_audio[:, audio_start:audio_end] if in_audio is not None else None
            in_word_ = in_word[:, start:end] if in_word is not None else None
            in_word_mask_ = in_word_mask[:, start:end] if in_word_mask is not None else None
            in_id_ = in_id[:, start:end] if in_id is not None else None

            ########################################################################
            # TODO: 去掉这里对first_stage_encode的调用，因为test过程其实是不大需要的，这里只为了好采样噪声，但是需要知道隐空间大小
            # 难说，可能有点复杂，先放在这吧
            results = self.first_stage_encode(tar_motion_, tar_exps_, in_audio_)
            latent_motion = results["latent_motion"]
            latent_expression = results["latent_expression"]
            latent_audio = results["latent_audio"] # 这个已经是mel谱了
            
            model_kwargs = dict(
                txt=in_word_, 
                txt_mask=in_word_mask_, 
                speaker_id=in_id_, 
                uncond_txt=null_embed.unsqueeze(0).repeat(bs, *[1] * len(in_word_.shape[1:]))
            )
            
            def model_fn(*args, **kwargs):
                results =  self.model_instance(**self.guider(
                    *args, **kwargs, model_kwargs=model_kwargs, is_sample=True, task=self.config.train.task))
                values = {}
                for key, value in results.items():
                    if key in ["motion", "expression", "audio"]:
                        values[key] = value
                values = self.guider.guide(**values, task=self.config.train.task) # guidance here
                res = []
                for key, value in values.items():
                    res.append(value)
                return res
                  
            motion_noise, expression_noise, audio_noise = self.get_initial_noise(
                latent_motion, latent_expression, latent_audio
            )
            motion_latents, expression_latents, audio_latents, extra = self.sampler.sample(
                model_fn,
                motion_noise,
                expression_noise,
                audio_noise,
                return_intermediate=if_vis_intermediate,
                return_model_output=if_vis_model_output,
            )
            motion_expression_results, audio_results \
                = self.first_stage_decode(motion_latents, expression_latents, audio_latents)
            
            motion_results = motion_expression_results["motion"]
            expression_results = motion_expression_results["expression"]
            motion_trans = motion_expression_results["trans"]

            if "motion_axis_angle" in motion_expression_results:
                motion_results_axis_angle = motion_expression_results["motion_axis_angle"]\
                    .view(bs, -1, 3 * self.joints)
            else:
                motion_results_axis_angle = rc.rotation_6d_to_axis_angle(motion_results.view(bs, motion_results.shape[1], -1, 6))\
                    .view(bs, -1, 3 * self.joints)
            

            audio_results = audio_results.transpose(-1, -2)
            audio_wav_results = self.mel_to_wav(audio_results)
            if audio_wav_results.shape[1] > in_audio_.shape[1]:
                audio_wav_results = audio_wav_results[:,:in_audio_.shape[1]]
            else:
                audio_wav_results = torch.cat([
                    audio_wav_results, 
                    torch.zeros(bs, in_audio_.shape[1] - audio_wav_results.shape[1]).to(
                        self.device, dtype=audio_wav_results.dtype)
                ], dim=-1)
            motion_results_all[:, start:end, :3*self.joints] = motion_results_axis_angle
            expression_results_all[:, start:end] = expression_results
            motion_trans_all[:, start:end, -7:] = motion_trans
            audio_results_all[:, audio_start:audio_end] = audio_wav_results

        if self.accelerator.is_main_process: 
            # 这里相当于是只让主进程进行可视化，多进程可视化自然也是可以实现的，就是有点麻烦
            # TODO: visualize the motion, expression, audio
            vis_bs = self.config.train.get("visualize_batch_size", bs)
            vis_bs = min(vis_bs, bs)
            if if_visualize:           
                in_audio_mel = self.mel_spec(in_audio)
                self.visualize_motion(
                    motion_results_axis_angle, 
                    expression_results,  
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
                self.visualize_audio(
                    audio_results, 
                    in_audio_mel, 
                    audio_wav_results, 
                    in_audio,
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
                
                
            if if_vis_model_output:
                self.visualize_sequence(
                    extra["motion_model_outputs"],
                    extra["expression_model_outputs"],
                    extra["audio_model_outputs"],
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
                    extra["expression_intermediates"],
                    extra["audio_intermediates"],
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
                adding_noise_motion, adding_noise_expression, adding_noise_audio =\
                self.sampler.get_adding_noise_sequence(
                    latent_motion,
                    latent_expression,
                    latent_audio
                )
                self.visualize_sequence(
                    adding_noise_motion,
                    adding_noise_expression,
                    adding_noise_audio,
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
            # if choice == "test":
            #     audio_sr = self.config.data.test.params.audio_sr
            # elif choice == "predict":
            #     audio_sr = self.config.data.predict.params.audio_sr
            # else:
            #     raise ValueError(f"Invalid choice: {choice}")
            audio_sr = self.config.data.audio_sr
            prefix = f"gs{global_step:06d}_" if global_step is not None else ""
            prefix += f"epoch{epoch:06d}_" if epoch is not None else ""
            prefix += f"step{step:06d}_" if step is not None else ""
            tar_beta_np = tar_beta.detach().cpu().numpy()
            rec_pose_np = motion_results_all.detach().cpu().numpy()
            rec_exp_np = expression_results_all.detach().cpu().numpy()
            rec_trans_np = motion_trans_all.detach().cpu().numpy()
            tar_pose_np = tar_motion_axis_angle.detach().cpu().numpy()
            tar_exp_np = tar_exps.detach().cpu().numpy()
            tar_trans_np = tar_trans.detach().cpu().numpy()
            
            res_paths = []
            gt_paths = []
            res_audio_paths = []
            gt_audio_paths = []
            
            for i in range(bs):
                prefix += f"bs{i}_"
                if not keep_every_epoch:
                    prefix = ""
                res_path = osp.join(self.config.work_dir, self.npz_save_name, choice,
                            f"rank{self.accelerator.process_index}_{prefix}resface_" + ori_id[i] + ".npz")
                gt_path = osp.join(self.config.work_dir, self.npz_save_name, choice,
                            f"rank{self.accelerator.process_index}_{prefix}gt_" + ori_id[i] + ".npz")
                res_paths.append(res_path)
                gt_paths.append(gt_path)
                res_audio_path = osp.join(self.config.work_dir, self.wav_save_name, choice,
                            f"rank{self.accelerator.process_index}_{prefix}res_sr{audio_sr}_" + ori_id[i] + ".wav")
                gt_audio_path = osp.join(self.config.work_dir, self.wav_save_name, choice,
                            f"rank{self.accelerator.process_index}_{prefix}gt_sr{audio_sr}_" + ori_id[i] + ".wav")
                res_audio_paths.append(res_audio_path)
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
                    res_audio_path,
                    audio_results_all[i].detach().cpu().numpy(),
                    samplerate=audio_sr,
                )
                sf.write(
                    gt_audio_path,
                    in_audio[i].detach().cpu().numpy(),
                    samplerate=audio_sr,
                )
            del tar_beta_np, rec_pose_np, rec_exp_np, rec_trans_np, tar_pose_np, tar_exp_np, tar_trans_np
                
                         
            results = {
                "motion": res_paths,
                "expression": res_paths,
                "audio": res_audio_paths,
                "gt_motion": gt_paths,
                "gt_expression": gt_paths,
                "gt_audio": gt_audio_paths,
                "betas": gt_paths,
            }
        else:
            results = {
                "motion": motion_results_axis_angle,
                "expression": expression_results,
                "audio": audio_wav_results,
                "gt_motion": tar_motion_axis_angle,
                "gt_expression": tar_exps,
                "gt_audio": in_audio,
                "betas": tar_beta
            }
        return results

    
    def validation_step(self, batch, null_embed, step, global_step = None, epoch = None, visualize = True):
        return self.shared_validation_test_step(
            batch, null_embed, step, choice="validation", global_step = global_step, epoch = epoch, visualize = visualize)
    
    def test_step(self, batch, null_embed, step):
        # return self.shared_validation_test_step(
        #     batch, null_embed, step, choice="test", global_step = None, epoch = None, visualize = False)
        return self.shared_test_predict_step(
            batch, null_embed, step, choice="test", global_step = None, epoch = None, visualize = False)

    def get_vertices(
        self, 
        rec_pose, 
        rec_exps, 
        rec_trans, 
        tar_beta, 
        tar_pose, 
        tar_exps, 
        tar_trans,
        vis_bs = 4,
    ):
        
        # rec_pose = rc.rotation_6d_to_matrix(rec_pose.reshape(vis_bs*n, j, 6))
        # rec_pose = rc.matrix_to_axis_angle(rec_pose).reshape(vis_bs*n, j*3)
        # tar_pose = rc.rotation_6d_to_matrix(tar_pose.reshape(vis_bs*n, j, 6))
        # tar_pose = rc.matrix_to_axis_angle(tar_pose).reshape(vis_bs*n, j*3)
        
        _, n, _ = rec_pose.shape
        j = self.joints
        rec_pose = rec_pose.view(vis_bs*n, j*3)
        tar_pose = tar_pose.view(vis_bs*n, j*3)
        
        vertices_rec = self.smplx(
            betas=tar_beta.reshape(vis_bs*n, 300), 
            transl=rec_trans.reshape(vis_bs*n, 3), 
            expression=tar_exps.reshape(vis_bs*n, 100),
            jaw_pose=rec_pose[:, 66:69], 
            global_orient=rec_pose[:,:3], 
            body_pose=rec_pose[:,3:21*3+3], 
            left_hand_pose=rec_pose[:,25*3:40*3], 
            right_hand_pose=rec_pose[:,40*3:55*3], 
            return_joints=True, 
            leye_pose=rec_pose[:, 69:72], 
            reye_pose=rec_pose[:, 72:75],
        )["vertices"]
        # vertices_tar = self.smplx(
        #         betas=tar_beta.reshape(bs*n, 300), 
        #         transl=rec_trans.reshape(bs*n, 3)-rec_trans.reshape(bs*n, 3), 
        #         expression=tar_exps.reshape(bs*n, 100)-tar_exps.reshape(bs*n, 100),
        #         jaw_pose=tar_pose[:, 66:69], 
        #         global_orient=tar_pose[:,:3], 
        #         body_pose=tar_pose[:,3:21*3+3], 
        #         left_hand_pose=tar_pose[:,25*3:40*3], 
        #         right_hand_pose=tar_pose[:,40*3:55*3], 
        #         return_joints=True, 
        #         leye_pose=tar_pose[:, 69:72], 
        #         reye_pose=tar_pose[:, 72:75],
        #     )
        vertices_rec_face = self.smplx(
            betas=tar_beta.reshape(vis_bs*n, 300), 
            transl=rec_trans.reshape(vis_bs*n, 3), 
            expression=rec_exps.reshape(vis_bs*n, 100), 
            jaw_pose=rec_pose[:, 66:69], 
            global_orient=rec_pose[:,:3]-rec_pose[:,:3], 
            body_pose=rec_pose[:,3:21*3+3]-rec_pose[:,3:21*3+3],
            left_hand_pose=rec_pose[:,25*3:40*3]-rec_pose[:,25*3:40*3],
            right_hand_pose=rec_pose[:,40*3:55*3]-rec_pose[:,40*3:55*3],
            return_verts=True, 
            return_joints=True,
            leye_pose=rec_pose[:, 69:72]-rec_pose[:, 69:72],
            reye_pose=rec_pose[:, 72:75]-rec_pose[:, 72:75],
        )["vertices"]
        # vertices_tar_face = self.smplx(
        #     betas=tar_beta.reshape(vis_bs*n, 300), 
        #     transl=tar_trans.reshape(vis_bs*n, 3)-tar_trans.reshape(vis_bs*n, 3), 
        #     expression=tar_exps.reshape(vis_bs*n, 100), 
        #     jaw_pose=tar_pose[:, 66:69], 
        #     global_orient=tar_pose[:,:3]-tar_pose[:,:3],
        #     body_pose=tar_pose[:,3:21*3+3]-tar_pose[:,3:21*3+3], 
        #     left_hand_pose=tar_pose[:,25*3:40*3]-tar_pose[:,25*3:40*3],
        #     right_hand_pose=tar_pose[:,40*3:55*3]-tar_pose[:,40*3:55*3],
        #     return_verts=True, 
        #     return_joints=True,
        #     leye_pose=tar_pose[:, 69:72]-tar_pose[:, 69:72],
        #     reye_pose=tar_pose[:, 72:75]-tar_pose[:, 72:75],
        # )["vertices"]
        vertices_tar_face = self.smplx(
            betas=tar_beta.reshape(vis_bs*n, 300), 
            transl=tar_trans.reshape(vis_bs*n, 3), 
            expression=tar_exps.reshape(vis_bs*n, 100), 
            jaw_pose=tar_pose[:, 66:69], 
            global_orient=tar_pose[:,:3],
            body_pose=tar_pose[:,3:21*3+3], 
            left_hand_pose=tar_pose[:,25*3:40*3],
            right_hand_pose=tar_pose[:,40*3:55*3],
            return_verts=True, 
            return_joints=True,
            leye_pose=tar_pose[:, 69:72],
            reye_pose=tar_pose[:, 72:75],
        )["vertices"]
        return vertices_rec, vertices_rec_face, vertices_tar_face

    def visualize_motion(
        self, 
        rec_pose, 
        rec_exps, 
        rec_trans,
        tar_beta,
        tar_pose,
        tar_exps,
        tar_trans,
        ori_id,
        step,
        choice="test",
        global_step = None,
        epoch = None,
        vis_bs = 4,
    ):
        bs, n, _ = rec_pose.shape
        j = self.joints
        if bs > vis_bs: # 限制一下可视化文件的数量
            rec_pose = rec_pose[0:vis_bs]
            rec_exps = rec_exps[0:vis_bs]
            rec_trans = rec_trans[0:vis_bs]
            tar_beta = tar_beta[0:vis_bs]
            tar_pose = tar_pose[0:vis_bs]
            tar_exps = tar_exps[0:vis_bs]
            tar_trans = tar_trans[0:vis_bs]
        
        # if choice == "test":
        #     pose_fps = self.config.data.test.params.pose_fps
        # elif choice == "validation":
        #     pose_fps = self.config.data.validation.params.pose_fps
        # else:
        #     raise ValueError(f"Invalid choice: {choice}")
        pose_fps = self.config.data.pose_fps
        
        npz_save_name = self.npz_save_name
        obj_save_name = self.obj_save_name
            
        if (30 / pose_fps) != 1:
            assert 30%pose_fps == 0
            n *= int(30/pose_fps)
            rec_pose = torch.nn.functional.interpolate(
                rec_pose.permute(0, 2, 1), scale_factor=30/pose_fps, mode='linear').permute(0,2,1)
            rec_exps = torch.nn.functional.interpolate(
                rec_exps.permute(0, 2, 1), scale_factor=30/pose_fps, mode='linear').permute(0,2,1)


        vertices_rec, vertices_rec_face, vertices_tar_face = self.get_vertices(
            rec_pose, rec_exps, rec_trans, tar_beta, tar_pose, tar_exps, tar_trans, vis_bs
        )
  
        
        ##############################  npz 保存的逻辑 ############################
        tar_pose = tar_pose.view(vis_bs, n, j*3)
        rec_pose = rec_pose.view(vis_bs, n, j*3)
        for i in range(vis_bs):
            tar_pose_np = tar_pose[i].detach().cpu().numpy()
            rec_pose_np = rec_pose[i].detach().cpu().numpy()
            rec_trans_np = rec_trans[i].detach().cpu().numpy()
            rec_exp_np = rec_exps[i].detach().cpu().numpy() 
            tar_exp_np = tar_exps[i].detach().cpu().numpy()
            tar_trans_np = tar_trans[i].detach().cpu().numpy()
            tar_beta_np = tar_beta[i].detach().cpu().numpy()
            
            prefix = f"gs{global_step:06d}_" if global_step is not None else ""
            prefix += f"epoch{epoch:06d}_" if epoch is not None else ""
            prefix += f"step{step:06d}_" if step is not None else ""
            
            if self.config.metrics.evaluator.get("evaluate_on_disk", False) is not True:
                np.savez(
                    osp.join(self.config.work_dir, npz_save_name, choice, 
                        f"{prefix}bs{i}_gt_"+ ori_id[i] +'.npz'),
                    betas=tar_beta_np[0], 
                    poses=tar_pose_np,
                    expressions=tar_exp_np,
                    trans=tar_trans_np,
                    model='smplx2020',
                    gender='neutral',
                    mocap_frame_rate = 30 ,
                )
                np.savez(
                    osp.join(self.config.work_dir, npz_save_name, choice, 
                            f"{prefix}bs{i}_res_"+ ori_id[i] +'.npz'),
                    betas=tar_beta_np[0], 
                    poses=rec_pose_np,
                    expressions=rec_exp_np,
                    trans=rec_trans_np,
                    model='smplx2020',
                    gender='neutral',
                    mocap_frame_rate = 30,
                )
            np.savez(
                osp.join(self.config.work_dir, npz_save_name, choice,
                         f"{prefix}bs{i}_resface_" + ori_id[i] + ".npz"),
                betas=tar_beta_np[0], 
                poses=rec_pose_np-rec_pose_np,
                expressions=rec_exp_np,
                trans=rec_trans_np,
                model='smplx2020',
                gender='neutral',
                mocap_frame_rate = 30,
            )
        
        ################################################################################
        
        ############################### obj 保存的逻辑 ############################
        visualize_obj_nums = self.config.train.get("visualize_obj_nums", 8) 
        # 限定一下可视化的obj数量，防止obj过多
        for i in range(vis_bs):
            if i >= 1:
                break # currently, only visualize the first batch
            for j in range(n):
                if j >= visualize_obj_nums:
                    break
                vertices_rec_index = vertices_rec[i*n+j].detach().cpu().numpy()
                vertices_rec_face_index = vertices_rec_face[i*n+j].detach().cpu().numpy()
                vertices_tar_face_index = vertices_tar_face[i*n+j].detach().cpu().numpy()
                rec_mesh = Trimesh(vertices=vertices_rec_index, faces=self.smplx.faces)
                rec_face_mesh = Trimesh(vertices=vertices_rec_face_index, faces=self.smplx.faces)
                tar_mesh = Trimesh(vertices=vertices_tar_face_index, faces=self.smplx.faces)
                rec_mesh.export(osp.join(self.config.work_dir, obj_save_name, choice,\
                    f"{prefix}bs{i}_res_frame{j:06d}_"+ ori_id[i] +'.obj'))
                rec_face_mesh.export(osp.join(self.config.work_dir, obj_save_name, choice,\
                    f"{prefix}bs{i}_res_face_frame{j:06d}_"+ ori_id[i] +'.obj'))
                tar_mesh.export(osp.join(self.config.work_dir, obj_save_name, choice,\
                    f"{prefix}bs{i}_gt_frame{j:06d}_"+ ori_id[i] +'.obj'))
        
    
    def visualize_audio(
        self, 
        rec_audio, 
        tar_audio,
        rec_audio_wav,
        tar_audio_wav,
        ori_id,
        step, 
        choice="test", 
        global_step = None, 
        epoch = None,
        vis_bs = 4,
    ):
        bs = rec_audio.shape[0]
        
        wav_save_name = self.wav_save_name
        png_save_name = self.png_save_name
        
        prefix = f"gs{global_step:06d}_" if global_step is not None else ""
        prefix += f"epoch{epoch:06d}_" if epoch is not None else ""
        prefix += f"step{step:06d}_" if step is not None else ""
        
        # if choice == "test":
        #     audio_sr = self.config.data.test.params.audio_sr
        #     hop_length = self.config.audio.mel_spec.hop_length
        # elif choice == "validation":
        #     audio_sr = self.config.data.validation.params.audio_sr
        #     hop_length = self.config.audio.mel_spec.hop_length
        # else:
        #     raise ValueError(f"Invalid choice: {choice}")
        audio_sr = self.config.data.audio_sr
        hop_length = self.config.audio.mel_spec.hop_length
        # rec_audio = rec_audio.transpose(-1, -2)
        # tar_audio = tar_audio.transpose(-1, -2)
        for i in range(vis_bs):
            # mel谱的频谱图
            plt.figure(figsize=(10, 5))
            # specshow 是 librosa 专门为绘制频谱图提供的函数
            librosa.display.specshow(rec_audio[i].detach().cpu().numpy(), 
                                    sr=audio_sr, 
                                    hop_length=hop_length,
                                    x_axis='time', 
                                    y_axis='mel')

            plt.colorbar(format='%+2.0f dB') # 添加颜色条，显示分贝值
            plt.title('Mel-frequency spectrogram of rec_audio') # 添加标题
            plt.tight_layout() # 调整图像布局，避免标签重叠
            plt.savefig(osp.join(self.config.work_dir, png_save_name, choice, 
                f"{prefix}bs{i}_res_mel_spectrogram_"+ ori_id[i] +'.png')
            )
            plt.close()
            
            # mel谱的频谱图
            plt.figure(figsize=(10, 5))
            # specshow 是 librosa 专门为绘制频谱图提供的函数
            librosa.display.specshow(tar_audio[i].detach().cpu().numpy(), 
                                    sr=audio_sr, 
                                    hop_length=hop_length,
                                    x_axis='time', 
                                    y_axis='mel')

            plt.colorbar(format='%+2.0f dB') # 添加颜色条，显示分贝值
            plt.title('Mel-frequency spectrogram of tar_audio') # 添加标题
            plt.tight_layout() # 调整图像布局，避免标签重叠
            plt.savefig(osp.join(self.config.work_dir, png_save_name, choice, 
                f"{prefix}bs{i}_gt_mel_spectrogram_"+ ori_id[i] +'.png')
            )
            plt.close()
            
        # rev_wav = self.mel_to_wav(rec_audio)
        # tar_wav = self.mel_to_wav(tar_audio)
        if self.config.metrics.evaluator.get("evaluate_on_disk", False) is not True:
            for i in range(vis_bs):
                # index = i + step * bs
                sf.write(
                    osp.join(self.config.work_dir, wav_save_name, choice, 
                        f"{prefix}bs{i}_res_"+ ori_id[i] +'.wav'), 
                    rec_audio_wav[i].detach().cpu().numpy(), 
                    samplerate=audio_sr,
                )
                sf.write(
                    osp.join(self.config.work_dir, wav_save_name, choice, 
                        f"{prefix}bs{i}_gt_"+ ori_id[i] +'.wav'), 
                    tar_audio_wav[i].detach().cpu().numpy(), 
                    samplerate=audio_sr,
                )
                # librosa.output.write_wav(
                #     osp.join(self.config.work_dir, wav_save_name, choice, f"{prefix}bs{i}_res.wav"), 
                #     rev_wav[i].detach().cpu().numpy(), 
                #     sr=audio_sr,
                # )
                # librosa.output.write_wav(
                #     osp.join(self.config.work_dir, wav_save_name, choice, f"{prefix}bs{i}_gt.wav"), 
                #     tar_wav[i].detach().cpu().numpy(), 
                #     sr=audio_sr,
                # )
    
    
    def visualize_motion_and_audio_together(
        self,
        ori_id,
        choice="test",
        global_step = None, 
        epoch = None,
        step = None,
        vis_bs = 4,
    ):
        if npz2pose is None or fast_render is None:
            self.logger.warning(
                "npz2pose or fast_render can not be imported, " 
                "maybe some packages of renderring (e.g. pytorch3d, trimesh, pyrender, etc.) are not installed, "
                "skipping visualization of motion and audio together")
            return
        prefix = f"gs{global_step:06d}_" if global_step is not None else ""
        prefix += f"epoch{epoch:06d}_" if epoch is not None else ""
        prefix += f"step{step:06d}_" if step is not None else ""
        
        wav_save_name = self.wav_save_name
        mp4_save_name = self.mp4_save_name
        npz_save_name = self.npz_save_name

    
        smplx_model_dir = osp.join(self.config.data.data_path_1, "smplx_models/")
        for i in range(vis_bs):
            # index = i + step * bs
            npz_path =  osp.join(
                self.config.work_dir, npz_save_name, choice, f"{prefix}bs{i}_res_"+ ori_id[i] +'.npz'
            )
            audio_path = osp.join(
                self.config.work_dir, wav_save_name, choice, 
                f"{prefix}bs{i}_res_"+ ori_id[i] +'.wav'
            )
            motion_dict = np.load(npz_path, allow_pickle=True)        
            v2d_face = npz2pose.render2d(motion_dict, (512, 512), face_only=True, remove_global=True, smplx_folder=smplx_model_dir)
            
            ##############################################################################################
            npz2pose.write_video(
                osp.join(self.config.work_dir, mp4_save_name, choice, f"{prefix}bs{i}_res_2dface.mp4"), 
                v2d_face.permute(0, 2, 3, 1), 
                fps=30
            )
            fast_render.add_audio_to_video(
                osp.join(self.config.work_dir, mp4_save_name, choice, f"{prefix}bs{i}_res_2dface.mp4"), 
                audio_path, 
                osp.join(self.config.work_dir, mp4_save_name, choice, f"{prefix}bs{i}_res_2dface_audio.mp4")
            )
            ###############################################################################################
            v2d_body = npz2pose.render2d(motion_dict, (720, 480), face_only=False, remove_global=True, smplx_folder=smplx_model_dir)
            npz2pose.write_video(
                osp.join(self.config.work_dir, mp4_save_name, choice, f"{prefix}bs{i}_res_2dbody.mp4"), 
                v2d_body.permute(0, 2, 3, 1), 
                fps=30
            )
            fast_render.add_audio_to_video(
                osp.join(self.config.work_dir, mp4_save_name, choice, f"{prefix}bs{i}_res_2dbody.mp4"), 
                audio_path, 
                osp.join(self.config.work_dir, mp4_save_name, choice, f"{prefix}bs{i}_res_2dbody_audio.mp4")
            )
            ###############################################################################################
            if self.config.train.get("visualize_sequence_with_face", True):
                fast_render.render_one_sequence_with_face(
                    npz_path, 
                    osp.join(self.config.work_dir, mp4_save_name, choice), 
                    audio_path, 
                    model_folder=smplx_model_dir
                )  

            if not self.config.train.get("visualize_gt_video", False):
                return
            
            npz_path =  osp.join(
                self.config.work_dir, npz_save_name, choice, f"{prefix}bs{i}_gt_"+ ori_id[i] +'.npz'
            )
            audio_path = osp.join(
                self.config.work_dir, wav_save_name, choice, 
                f"{prefix}bs{i}_gt_"+ ori_id[i] +'.wav'
            )
            motion_dict = np.load(npz_path, allow_pickle=True)        
            v2d_face = npz2pose.render2d(motion_dict, (512, 512), face_only=True, remove_global=True, smplx_folder=smplx_model_dir)
            
            ##############################################################################################
            npz2pose.write_video(
                osp.join(self.config.work_dir, mp4_save_name, choice, f"{prefix}bs{i}_gt_2dface.mp4"), 
                v2d_face.permute(0, 2, 3, 1), 
                fps=30
            )
            fast_render.add_audio_to_video(
                osp.join(self.config.work_dir, mp4_save_name, choice, f"{prefix}bs{i}_gt_2dface.mp4"), 
                audio_path, 
                osp.join(self.config.work_dir, mp4_save_name, choice, f"{prefix}bs{i}_gt_2dface_audio.mp4")
            )
            ###############################################################################################
            v2d_body = npz2pose.render2d(motion_dict, (720, 480), face_only=False, remove_global=True, smplx_folder=smplx_model_dir)
            npz2pose.write_video(
                osp.join(self.config.work_dir, mp4_save_name, choice, f"{prefix}bs{i}_gt_2dbody.mp4"), 
                v2d_body.permute(0, 2, 3, 1), 
                fps=30
            )
            fast_render.add_audio_to_video(
                osp.join(self.config.work_dir, mp4_save_name, choice, f"{prefix}bs{i}_gt_2dbody.mp4"), 
                audio_path, 
                osp.join(self.config.work_dir, mp4_save_name, choice, f"{prefix}bs{i}_gt_2dbody_audio.mp4")
            )
            ###############################################################################################
            if self.config.train.get("visualize_sequence_with_face", True):
                fast_render.render_one_sequence_with_face(
                    npz_path, 
                    osp.join(self.config.work_dir, mp4_save_name, choice), 
                    audio_path, 
                    model_folder=smplx_model_dir
                ) 
    
    def visualize_sequence(
        self,
        motion_list,
        expression_list,
        audio_list,
        tar_beta,
        tar_motion_axis_angle,
        tar_exps,
        tar_trans,
        ori_id,
        step,
        choice="test",
        global_step = None, 
        epoch = None,
        vis_bs = 1,
        vis_type="intermediate",
    ):
        # TODO: visualize the intermediate or model output
        # 现在是只准备支持可视化一下obj和png，后续再补充，主要是怕太占用空间，也占计算速度
        if vis_bs > 1:
            self.logger.info(f"Visualize the intermediate or model output, vis_bs is reduced to {vis_bs}")
            vis_bs = 1 # 现在只支持可视化一个数据，怕太耗时间，也太耗空间
        
        if vis_bs == 1:
            _tar_pose = tar_motion_axis_angle[0].unsqueeze(0)
            _tar_exps = tar_exps[0].unsqueeze(0)
            _tar_trans = tar_trans[0].unsqueeze(0)
            _tar_beta = tar_beta[0].unsqueeze(0)
        else:
            _tar_pose = tar_motion_axis_angle
            _tar_exps = tar_exps
            _tar_trans = tar_trans
            _tar_beta = tar_beta
        
        prefix = f"gs{global_step:06d}_" if global_step is not None else ""
        prefix += f"epoch{epoch:06d}_" if epoch is not None else ""
        prefix += f"step{step:06d}_" if step is not None else ""
        
        obj_save_name = self.obj_save_name
        png_save_name = self.png_save_name
        
        if choice == "test":
            audio_sr = self.config.data.test.params.audio_sr
            hop_length = self.config.audio.mel_spec.hop_length
        elif choice == "validation":
            audio_sr = self.config.data.validation.params.audio_sr
            hop_length = self.config.audio.mel_spec.hop_length
        else:
            raise ValueError(f"Invalid choice: {choice}")
        
        
        for idx, (motion, expression, audio) in enumerate(zip(motion_list, expression_list, audio_list)):
            n = motion.shape[1]
            if vis_bs == 1:
                motion = motion[0].unsqueeze(0) # 只可视化第一个数据
                expression = expression[0].unsqueeze(0)
                audio = audio[0].unsqueeze(0)
            else:
                motion = motion[0:vis_bs]
                expression = expression[0:vis_bs]
                audio = audio[0:vis_bs]
            
            motion_expression_results, audio_results \
                = self.first_stage_decode(motion, expression, audio)
            
            motion_results = motion_expression_results["motion"]
            expression_results = motion_expression_results["expression"]
            motion_trans = motion_expression_results["trans"]
            
            if "motion_axis_angle" in motion_expression_results:
                motion_results_axis_angle = motion_expression_results["motion_axis_angle"]\
                    .view(vis_bs, -1, 3 * self.joints)
            else:
                motion_results_axis_angle = rc.rotation_6d_to_axis_angle(
                    motion_results.view(vis_bs, motion_results.shape[1], -1, 6))\
                        .view(vis_bs, -1, 3 * self.joints)
            
            audio_results = audio.transpose(-1, -2)
            
            vertices_rec, vertices_rec_face, _ = self.get_vertices(
                motion_results_axis_angle, expression_results, motion_trans, 
                _tar_beta, _tar_pose, _tar_exps, _tar_trans, vis_bs
            )
            ############################### obj 保存的逻辑 ############################
            visualize_obj_nums = self.config.train.get("visualize_obj_nums", 8) 
            # 限定一下可视化的obj数量，防止obj过多
            for i in range(vis_bs):
                if i >= 1:
                    break # currently, only visualize the first batch
                for j in range(n):
                    if j >= visualize_obj_nums:
                        break
                    vertices_rec_index = vertices_rec[i*n+j].detach().cpu().numpy()
                    vertices_rec_face_index = vertices_rec_face[i*n+j].detach().cpu().numpy()
                    rec_mesh = Trimesh(vertices=vertices_rec_index, faces=self.smplx.faces)
                    rec_face_mesh = Trimesh(vertices=vertices_rec_face_index, faces=self.smplx.faces)
                    rec_mesh.export(osp.join(self.config.work_dir, obj_save_name, choice,\
                        f"{prefix}bs{i}_res_frame{j:06d}_{vis_type}{idx}_"+ ori_id[i] +'.obj'))
                    rec_face_mesh.export(osp.join(self.config.work_dir, obj_save_name, choice,\
                        f"{prefix}bs{i}_res_face_frame{j:06d}_{vis_type}{idx}_"+ ori_id[i] +'.obj'))

            for i in range(vis_bs):
                plt.figure(figsize=(10, 5))
                # specshow 是 librosa 专门为绘制频谱图提供的函数
                librosa.display.specshow(audio_results[i].detach().cpu().numpy(), 
                                        sr=audio_sr, 
                                        hop_length=hop_length,
                                        x_axis='time', 
                                        y_axis='mel')

                plt.colorbar(format='%+2.0f dB') # 添加颜色条，显示分贝值
                plt.title('Mel-frequency spectrogram of rec_audio') # 添加标题
                plt.tight_layout() # 调整图像布局，避免标签重叠
                plt.savefig(osp.join(self.config.work_dir, png_save_name, choice, 
                    f"{prefix}bs{i}_res_mel_spectrogram_{vis_type}{idx}_"+ ori_id[i] +'.png')
                )
                plt.close()
