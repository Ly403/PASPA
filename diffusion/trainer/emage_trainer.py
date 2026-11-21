import torch
import json
import os
from diffusion.trainer import MotionTrainer
from diffusion.scheduler.iddpm import EmageScheduler
from diffusion.utils.config import instantiate_from_config, set_value
import time
from diffusion.model.utils import expand_dims_like
from diffusion.utils import rotation_conversions as rc
from diffusion.utils import FK
from diffusion.model.utils import mean_flat
from diffusion.guider.guider import (
    VanillaCFGEmageGuider,
    IdentityEmageGuider,
)
from termcolor import colored
import os.path as osp
from trimesh import Trimesh
import numpy as np
import soundfile as sf
import random
from diffusion.utils.motion_io import MakeMask
from functools import partial
from diffusion.model.audio_encoder.mel_spec import MelSpec
import torch.distributed as dist
from diffusion.utils.filter import Filter1D
from diffusion.data.datasets.beat2.data_tools import GetBodyPartIndices
try:
    import diffusion.utils.fast_render as fast_render
    import diffusion.utils.npz2pose as npz2pose
except:
    fast_render = None
    npz2pose = None

@torch.no_grad()
def _allreduce_sum_(x: torch.Tensor) -> torch.Tensor:
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(x, op=dist.ReduceOp.SUM)
    return x

def grad_norm_of_loss(
    loss: torch.Tensor,
    params,
    norm_type: float = 2.0,
    retain_graph: bool = True,
    ddp_sync_norm: bool = True,
) -> torch.Tensor:
    """
    返回：|| d(loss)/d(params) ||_p （把所有参数梯度当成拼接后的一个大向量）
    注意：不会写入 params.grad，不影响正常训练。
    """
    grads = torch.autograd.grad(
        loss,
        params,
        retain_graph=retain_graph,   # 关键：后面还要对总 loss backward
        create_graph=False,
        allow_unused=True,           # 某些 loss 不依赖某些参数时避免报错
    )  # :contentReference[oaicite:1]{index=1}

    # 计算拼接向量的 p-norm： (sum ||g_i||_p^p)^(1/p)
    if norm_type == float("inf"):
        local_val = torch.zeros([], device=loss.device)
        for g in grads:
            if g is None:
                continue
            local_val = torch.maximum(local_val, g.detach().abs().max())
        if ddp_sync_norm:
            # inf-norm 严格同步要 all_reduce(max)，这里给出可选实现：
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(local_val, op=dist.ReduceOp.MAX)
        return local_val

    p = float(norm_type)
    local_sum_p = torch.zeros([], device=loss.device)
    for g in grads:
        if g is None:
            continue
        g = g.detach()
        local_sum_p = local_sum_p + (g.norm(p) ** p)

    if ddp_sync_norm:
        _allreduce_sum_(local_sum_p)

    return local_sum_p.pow(1.0 / p)

def per_loss_grad_norms(
    losses: dict,
    model,
    norm_type: float = 2.0,
    ddp_sync_norm: bool = True,
    only_keys=None,
) -> dict:
    params = [p for p in model.parameters() if p.requires_grad]
    out = {}
    keys = list(losses.keys()) if only_keys is None else [k for k in only_keys if k in losses]
    for k in keys:
        v = losses[k]
        if not torch.is_tensor(v):
            continue
        if v.numel() != 1:
            v = v.mean()
        out[k] = grad_norm_of_loss(
            v, params, norm_type=norm_type,
            retain_graph=True,
            ddp_sync_norm=ddp_sync_norm,
        )
    return out

class EmageTrainer(MotionTrainer):
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
        self.speech_model = self.configure_speech_model(config, logger)
        self.validate_speech_model_data_contract(data, logger=logger)
        super().__init__(
            accelerator,
            data,
            model_config,
            config,
            num_replicas=num_replicas,
            logger=logger,
            generator=generator,
        )
        self.load_audio_feat = getattr(self.config.data, "load_audio_feat", True)
        self.audio_encoder = self.configure_audio_encoder()
        if self.config.data.local:
            self.fk_process = FK.LFKWraper().to(self.device)
        else:
            self.fk_process = FK.GFKWraper().to(self.device)
        # ckpt = torch.load("/remote-home/liuyi/Sana/epoch_659.pth", map_location="cpu")
        # missing, unexpected = self.model.model.load_state_dict(ckpt, strict=False)
        # if len(missing) > 0 or len(unexpected) > 0:
        #     self.logger.warning(f"Missing keys: {missing}")
        #     self.logger.warning(f"Unexpected keys: {unexpected}")

    def configure_text_encoder(self):
        # 处理非text条件生成的问题
        if self.config.train.task.startswith("text2"):
            return super().configure_text_encoder()
        else:
            return None, None, None

    def configure_audio_vae(self):
        return None

    def prepare_null_embedding(self):
        # 处理非text条件生成的问题
        if self.config.train.task.startswith("text2"):
            return super().prepare_null_embedding()
        else:
            return None

    def configure_mel_spec(self):
        return None

    def configure_vocoder(self):
        return None

    def test_step(self, batch, null_embed, step):
        # return self.shared_validation_test_step(
        #     batch, null_embed, step, choice="test", global_step = None, epoch = None, visualize = False)
        return self.shared_test_predict_step(
            batch, null_embed, step, choice="test", global_step = None, epoch = None, visualize = True)

    def visualize_audio(
        self,
        tar_audio,
        ori_id,
        step,
        choice="test",
        global_step = None,
        epoch = None,
        vis_bs = 4,
    ):
        bs = tar_audio.shape[0]

        wav_save_name = self.wav_save_name

        prefix = f"gs{global_step:06d}_" if global_step is not None else ""
        prefix += f"epoch{epoch:06d}_" if epoch is not None else ""
        prefix += f"step{step:06d}_" if step is not None else ""

        audio_sr = self.config.data.audio_sr

        if self.config.metrics.evaluator.get("evaluate_on_disk", False) is not True:
            for i in range(vis_bs):
                sf.write(
                    osp.join(self.config.work_dir, wav_save_name, choice,
                        f"{prefix}bs{i}_gt_"+ ori_id[i] +'.wav'),
                    tar_audio[i].detach().cpu().numpy(),
                    samplerate=audio_sr,
                )

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
            npz_path =  osp.join(
                self.config.work_dir, npz_save_name, choice, f"{prefix}bs{i}_res_"+ ori_id[i] +'.npz'
            )
            audio_path = osp.join(
                self.config.work_dir, wav_save_name, choice,
                f"{prefix}bs{i}_gt_"+ ori_id[i] +'.wav'
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

    def configure_evaluator_post(self):
        if self.config.metrics.get("evaluator_type", "single") == "distributed":
            from diffusion.metrics.emage import DistributedEmageMetric
            evaluator = DistributedEmageMetric(
                **self.config.metrics.evaluator,
                accelerator=self.accelerator,
                smplx=self.smplx,
                device=self.device,
            )
        else:
            from diffusion.metrics.emage import EmageMetric
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
    
    def prepare_for_dist(self):
        if self.complete_prepare_for_dist:
            return
        finetune_speech = bool(self.config.train.get("finetune_speech_model", False))

        if self.talkshow_evaluation_only:
            if finetune_speech:
                evaluation_mode = (
                    "test-only"
                    if self.talkshow_test_only
                    else "validation-only"
                )
                raise ValueError(
                    f"TalkSHOW {evaluation_mode} runtime cannot finetune "
                    "the speech model"
                )
            self.model = self.accelerator.prepare(self.model)
            if self.model_ema is not None and (
                not self.config.train.use_fsdp
            ):
                self.model_ema.to(self.device)
            self.speech_model.to(self.device)
            self.complete_prepare_for_dist = True
            return
        
        opt_is_list = isinstance(self.optimizer, list)
        sch_is_list = isinstance(self.lr_scheduler, list)
        optimizers = self.optimizer if opt_is_list else [self.optimizer]
        schedulers = self.lr_scheduler if sch_is_list else [self.lr_scheduler]
        # 强烈建议：多 optimizer 就配套多 scheduler；不一致直接报错比静默复制引用安全
        if len(optimizers) != len(schedulers):
            raise ValueError(
                f"len(optimizers)={len(optimizers)} != len(lr_schedulers)={len(schedulers)}. "
                "Please provide one scheduler per optimizer (recommended)."
            )

        # 1) 组织一次性 prepare 的对象序列（顺序决定了后续如何解包）
        to_prepare = [self.model]
        if finetune_speech:
            to_prepare.append(self.speech_model)
        to_prepare.extend(optimizers)
        to_prepare.extend(schedulers)
        prepared = self.accelerator.prepare(*to_prepare)

        # 3) 按相同顺序解包回写
        idx = 0
        self.model = prepared[idx]; idx += 1
        if finetune_speech:
            self.speech_model = prepared[idx]; idx += 1
        # optimizers
        opt_block = prepared[idx: idx + len(optimizers)]
        idx += len(optimizers)
        self.optimizer = list(opt_block) if opt_is_list else opt_block[0]
        # schedulers
        sch_block = prepared[idx: idx + len(schedulers)]
        idx += len(schedulers)
        self.lr_scheduler = list(sch_block) if sch_is_list else sch_block[0]
        # EMA：通常不要 prepare，只放到 device
        if self.model_ema is not None and (not self.config.train.use_fsdp):
            self.model_ema.to(self.device)
        if not finetune_speech:
            self.speech_model.to(self.device)
        self.complete_prepare_for_dist = True
    
    def training(self):
        super().training()
        if self.config.train.get("finetune_speech_model", False):
            self.speech_model.train()

    def eval(self):
        super().eval()
        if self.speech_model is not None:
            self.speech_model.eval()

    def get_model_list(self):
        model_list = super().get_model_list()
        if self.config.train.get("finetune_speech_model", False):
            model_list.append(self.speech_model)
        return torch.nn.ModuleList(model_list)
        
    def configure_guider(self):
        if self.config.model.guider.guider_type == "classifier-free":     
            params = dict(self.config.model.guider)
            params.pop("guider_type")
            guider = VanillaCFGEmageGuider(
                **params
            )
        elif self.config.model.guider.guider_type == "identity":
            guider = IdentityEmageGuider()
        else:
            raise ValueError(f"Invalid guider type: {self.config.model.guider.guider_type}")
        self.logger.info(
            colored(
                f"Using {self.config.model.guider.guider_type} Guider: "
                f"{guider.__class__.__name__}:{guider}, "
                f"ucg_prob: {self.config.model.guider.get('ucg_prob', 0.0)}, "
                f"cfg_motion_scale: {self.config.model.guider.get('cfg_motion_scale', 'N/A')}, ",
                "green",
                attrs=["bold"],
            )
        )
        return guider
    
    def configure_speech_model(self, config, logger):
        speech_model_ckpt_path = config.model.get("speech_model_ckpt_path", None)
        speech_model = instantiate_from_config(config.model.speech_model)
        runtime_audio_validator = getattr(
            speech_model,
            "validate_runtime_audio_encoder_config",
            None,
        )
        if callable(runtime_audio_validator):
            runtime_audio_validator(config.audio.audio_encoder)
        runtime_data_validator = getattr(
            speech_model,
            "validate_runtime_data_config",
            None,
        )
        if callable(runtime_data_validator):
            runtime_data_validator(config.data)
        runtime_training_validator = getattr(
            speech_model,
            "validate_runtime_training_config",
            None,
        )
        if callable(runtime_training_validator):
            runtime_training_validator(config.train)
        required_mixed_precision = getattr(
            speech_model,
            "required_mixed_precision",
            None,
        )
        if (
            required_mixed_precision is not None
            and str(config.model.mixed_precision)
            != str(required_mixed_precision)
        ):
            raise ValueError(
                f"{speech_model.__class__.__name__} requires "
                f"model.mixed_precision={required_mixed_precision!r}"
            )
        if (
            config.train.get("finetune_speech_model", False)
            and not getattr(speech_model, "supports_finetuning", True)
        ):
            raise ValueError(
                f"{speech_model.__class__.__name__} is a frozen condition "
                "network and does not support finetune_speech_model=true"
            )
        if (
            getattr(
                speech_model,
                "requires_contextual_audio_features",
                False,
            )
            and config.model.get(
                "inference_audio_feature_scope",
                None,
            )
            != "full_clip"
        ):
            raise ValueError(
                f"{speech_model.__class__.__name__} requires "
                "model.inference_audio_feature_scope=full_clip"
            )
        supported_main_trainers = getattr(
            speech_model,
            "supported_main_trainers",
            None,
        )
        if supported_main_trainers is not None:
            trainer_name = str(config.trainer)
            if trainer_name not in supported_main_trainers:
                raise ValueError(
                    f"{speech_model.__class__.__name__} only supports main "
                    f"trainer(s) {tuple(supported_main_trainers)!r}; got "
                    f"trainer={trainer_name!r}"
                )
        if (
            getattr(
                speech_model,
                "requires_frame_aligned_audio_features",
                False,
            )
            and not bool(config.data.get("load_audio_feat", False))
        ):
            raise ValueError(
                f"{speech_model.__class__.__name__} requires "
                "data.load_audio_feat=true so training and validation use "
                "the strict full-clip LMDB cache"
            )
        if speech_model_ckpt_path is not None:
            checkpoint_loader = getattr(
                speech_model, "load_pretrained_checkpoint", None)
            if callable(checkpoint_loader):
                load_report = checkpoint_loader(speech_model_ckpt_path)
                logger.info(
                    f"load speech model from {speech_model_ckpt_path}: "
                    f"{load_report}")
            else:
                speech_model_state_dict = torch.load(
                    speech_model_ckpt_path, map_location="cpu")
                missing, unexpected = speech_model.load_state_dict(
                    speech_model_state_dict, strict=False)
                logger.info(f"load speech model from {speech_model_ckpt_path}")
                if len(missing) > 0 or len(unexpected) > 0:
                    logger.info(f"missing keys: {missing}")
                    logger.info(f"unexpected keys: {unexpected}")
            # TODO: support finetuning
            if config.train.get("finetune_speech_model", False):
                assert config.scheduler.scheduler.lambda_expression > 0.0, \
                    "lambda_expression must be greater than 0.0 when finetuning speech model"
            else:
                speech_model = speech_model.requires_grad_(False).eval()
        else:
            if getattr(
                speech_model, "requires_pretrained_checkpoint", False
            ):
                raise ValueError(
                    f"{speech_model.__class__.__name__} requires "
                    "model.speech_model_ckpt_path"
                )
            logger.info("no speech model checkpoint path provided")
        
        return speech_model

    def validate_speech_model_data_contract(self, data, logger=None):
        """Run optional dataset checks owned by specialized speech models."""

        validator = getattr(
            self.speech_model,
            "validate_dataset_contract",
            None,
        )
        if not callable(validator):
            return

        metadata_by_split = {}
        for split, attribute in (
            ("train", "train_dataset"),
            ("validation", "validation_dataset"),
            ("test", "test_dataset"),
        ):
            dataset = getattr(data, attribute, None)
            if dataset is None:
                continue
            metadata = getattr(dataset, "lmdb_metadata", None)
            if metadata is None:
                raise RuntimeError(
                    f"{self.speech_model.__class__.__name__} requires "
                    f"strict LMDB metadata for the {split} dataset"
                )
            metadata_by_split[split] = metadata
        validator(metadata_by_split)
        if logger is not None:
            logger.info(
                "Validated TalkSHOW face checkpoint, LMDB, and WavLM "
                "contracts"
            )
    
    def configure_scheduler(self):
        set_value(self.config.scheduler.scheduler, "timestep_respacing",\
            str(self.config.scheduler.train_sampling_steps))
        diffusion = EmageScheduler(
            **self.config.scheduler.scheduler,
            logger=self.logger, device=self.device,
        )
        return diffusion
    
    def prepare_mask(self, data_dict):
        ###########################################################################
        tar_pose_6d = data_dict["tar_pose_6d"]
        tar_trans = data_dict["tar_trans"]
        # for seed
        B, L  = tar_pose_6d.shape[:2]
        # assert B >= 20
        
        pose_prompt = tar_pose_6d.to(torch.float32)
        mask1 = torch.full((B - B//3, L), True, dtype=torch.bool, device=tar_pose_6d.device)
        mask1[:,:self.config.scheduler.overlap_length] = False
        mask2 = MakeMask(B//3, L, mask_ratio=random.random()*0.3 + 0.5, device=tar_pose_6d.device)
        mask1[:B//3, L - self.config.scheduler.overlap_length + random.randint(0,7)] = False
        prompt_mask = torch.cat([mask1, mask2], dim=0)
        # prompt_mask = torch.full((B, L), True, dtype=torch.bool, device=tar_pose_6d.device)
        # prompt_mask[:, :self.config.scheduler.overlap_length] = False
        pose_prompt = pose_prompt * (~prompt_mask).unsqueeze(-1).float()

        trans_prompt = tar_trans.to(torch.float32)
        # trans_mask = torch.full((B, L), True, dtype=torch.bool, device=tar_pose_6d.device)
        # trans_mask[:, :self.config.scheduler.overlap_length] = False
        trans_mask = prompt_mask.clone()
        trans_prompt = trans_prompt * (~trans_mask).unsqueeze(-1).float()
        
        data_dict["pose_prompt"] = pose_prompt
        data_dict["pose_prompt_mask"] = prompt_mask
        data_dict["trans_prompt"] = trans_prompt
        data_dict["trans_prompt_mask"] = trans_mask
        return data_dict
    
    def configure_audio_encoder(self):
        # 这个audio_encoder可以是一个复杂的深度学习模型提取特征，也可以就弄个mel谱
        # 我们会首先尝试从audio_encoder的配置中来获取，如果获取不到，我们就用mel谱
        if not self.config.train.task.startswith("audio2"):
            return None

        if (
            not self.load_audio_feat
            or self.config._test
            or self.config._predict
            or self.config._validation
        ):
            audio_encoder_config = getattr(self.config.audio, "audio_encoder", None)
            if audio_encoder_config is not None:
                audio_encoder_name = audio_encoder_config["name"]
                if audio_encoder_name == "hubert-large-ls960-ft": #
                    from modelscope import AutoProcessor, HubertModel, BitsAndBytesConfig # transformers库发癫一样，加载不了这个模型
                    local_model_path = audio_encoder_config.get("local_model_path", None)
                    is_quantization = self.config.audio.audio_encoder.get("quantization", False)
                    if is_quantization:
                        bnb_config = BitsAndBytesConfig(
                            load_in_8bit=True,
                            llm_int8_threshold=6.0
                        )
                    if local_model_path is not None:
                        processor = AutoProcessor.from_pretrained(local_model_path)
                        if is_quantization:
                            model = HubertModel.from_pretrained(
                                local_model_path, quantization_config=bnb_config, torch_dtype=torch.float16,
                                device_map=self.device, attn_implementation="sdpa")
                            model = model.eval()#.to(self.device)
                        else:
                            model = HubertModel.from_pretrained(local_model_path)
                            model = model.requires_grad_(False).eval().to(self.device)
                    else:
                        processor = AutoProcessor.from_pretrained("facebook/hubert-large-ls960-ft")
                        if is_quantization:
                            model = HubertModel.from_pretrained(
                            "facebook/hubert-large-ls960-ft", quantization_config=bnb_config, torch_dtype=torch.float16,
                            device_map=self.device, attn_implementation="sdpa")
                            model = model.eval()#.to(self.device)
                        else:
                            model = HubertModel.from_pretrained("facebook/hubert-large-ls960-ft")
                            model = model.eval().to(self.device)
                        model = model.eval().to(self.device)
                    audio_encoder = [processor, model]
                    params = sum(p.numel() for p in audio_encoder[1].parameters()) / 1e6
                elif audio_encoder_name == "wavlm":
                    from transformers import Wav2Vec2Processor, WavLMModel
                    local_model_path = audio_encoder_config.get("local_model_path", None)
                    if local_model_path is not None:
                        processor = Wav2Vec2Processor.from_pretrained(local_model_path)
                        model = WavLMModel.from_pretrained(local_model_path)
                    else:
                        processor = Wav2Vec2Processor.from_pretrained("facebook/wavlm-libri-clean-100h-large")
                        model = WavLMModel.from_pretrained("facebook/wavlm-libri-clean-100h-large")
                    model = model.requires_grad_(False).eval().to(self.device)
                    audio_encoder = [processor, model]
                    params = sum(p.numel() for p in audio_encoder[1].parameters()) / 1e6
                else:
                    raise ValueError(f"Invalid audio encoder: {audio_encoder_name}")
            else: # 简单的mel谱处理
                audio_encoder = MelSpec(**self.config.audio.mel_spec)
                params = sum(p.numel() for p in audio_encoder.parameters()) / 1e6
            self.logger.info(
                colored(
                    f"Audio Encoder: "
                    f"{audio_encoder.__class__.__name__}:{audio_encoder_config}, "
                    f"Model Parameters: {params:.2f} M",
                    "green",
                    attrs=["bold"],
                )
            )
        else:
            audio_encoder = None
            self.logger.info(
                colored(
                    f"Audio Encoder is not used since load_audio_feat is False",
                    "green",
                    attrs=["bold"],
                )
            )
        return audio_encoder
    
    def prepare_data(self, batch, step, null_embed = None, lm_time_all=None, am_time_all=None, is_predict=False):
        j = self.joints
        tar_pose_raw = self.get_input(batch, "pose")
        tar_pose = tar_pose_raw[:, :, :j*3].to(self.device)
        tar_contact = tar_pose_raw[:, :, j*3:j*3+4].to(self.device)
        tar_trans = self.get_input(batch, "trans").to(self.device)
        tar_exps = self.get_input(batch, "facial").to(self.device)
        tar_beta = self.get_input(batch, "beta").to(self.device)
        tar_id = self.get_input(batch, "id").to(self.device).long()
        ori_id = self.get_input(batch, "ori_vid")
        is_padding_sample = self.get_input(batch, "is_padding_sample")
        if is_padding_sample is not None:
            is_padding_sample = is_padding_sample.to(
                self.device,
                dtype=torch.bool,
            )
        # preprocess the tar_pose, switch to 6d representation
        bs, n = tar_pose.shape[0], tar_pose.shape[1]
        tar_pose_6d = rc.axis_angle_to_matrix(tar_pose.reshape(bs, n, j, 3))
        tar_pose_6d = rc.matrix_to_rotation_6d(tar_pose_6d).reshape(bs, n, j*6)

        tar_motion = torch.cat([tar_pose_6d, tar_trans, tar_contact], dim=-1)

        ####################### for text encoder ###############################
        if lm_time_all is not None:
            lm_time_start = time.time()
        if not self.config.train.task.startswith("text"):
            in_word = None
            in_word_mask = None
            in_word_txt = None
        else:
            if is_predict:
                in_word = self.get_input(batch, "word")
                in_word_txt = in_word # 保留原始文本
                in_word_mask = self.get_input(batch, "word_mask")
            elif self.load_text_feat:
                in_word = self.get_input(batch, "word").to(self.device)  # bs, 1, N, C
                in_word_txt = in_word # 保留原始文本
                in_word = self.get_input(batch, "word_feat").to(self.device)
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

                        in_word = self.text_encoder(
                            txt_tokens.input_ids, attention_mask=txt_tokens.attention_mask)[0][:, None]
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
                    "gemma" in self.config.text_encoder.text_encoder_name or \
                        "Qwen" in self.config.text_encoder.text_encoder_name
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
                        in_word = in_word[:, select_index]
                        in_word_mask = txt_tokens.attention_mask[:, select_index]
                else:
                    self.logger.error("error")
                    exit()
        self.accelerator.wait_for_everyone()
        if lm_time_all is not None:
            lm_time_all += time.time() - lm_time_start
        ########################################################################

        ####################### for audio encoder ###############################
        if am_time_all is not None:
            am_time_start = time.time()
        in_face_audio = None
        in_face_audio_valid_length = None
        in_face_audio_crop_start = None
        if not self.config.train.task.startswith("audio"):
            in_audio = None
            in_audio_mask = None
            in_audio_wav = None
        else:
            in_audio = self.get_input(batch, "audio").to(self.device)
            in_audio_wav = in_audio
            use_cached_evaluation_audio = (
                is_predict
                and self.load_audio_feat
                and getattr(
                    self.speech_model,
                    "requires_frame_aligned_audio_features",
                    False,
                )
            )
            if use_cached_evaluation_audio:
                cached_audio = self.get_input(batch, "audio_feat")
                if cached_audio is None:
                    raise RuntimeError(
                        f"{self.speech_model.__class__.__name__} requires "
                        "audio_feat for validation"
                    )
                in_audio = cached_audio.to(self.device)
                in_audio_mask = self.get_input(batch, "audio_mask")
                if in_audio_mask is not None:
                    in_audio_mask = in_audio_mask.to(self.device)
            elif is_predict:
                in_audio_mask = self.get_input(batch, "audio_mask")
                if in_audio_mask is not None:
                    in_audio_mask = in_audio_mask.to(self.device)
            elif self.load_audio_feat:
                in_audio = self.get_input(batch, "audio_feat").to(self.device)
                in_audio_mask = self.get_input(batch, "audio_mask")
                if in_audio_mask is not None:
                    in_audio_mask = in_audio_mask.to(self.device)
            elif self.config.audio.get("audio_encoder", None) is not None:
                # 使用hubert-large-ls960-ft
                audio_encoder_config = self.config.audio.audio_encoder
                audio_encoder_name = audio_encoder_config["name"]
                if audio_encoder_name == "hubert-large-ls960-ft":
                    with torch.no_grad():
                        dtype = in_audio.dtype
                        in_audio = self.audio_encoder[0](
                            list([_.cpu().numpy() for _ in in_audio]),
                            sampling_rate=self.config.data.audio_sr,
                            padding=True,
                            return_tensors="pt"
                        )
                        in_audio_mask = in_audio.attention_mask.to(self.device)
                        in_audio = in_audio.input_values.to(self.device)
                        if not self.config.audio.audio_encoder.get("quantization", False):
                            in_audio = self.audio_encoder[1](in_audio, attention_mask=in_audio_mask)
                        else:
                            in_audio = self.audio_encoder[1](
                                in_audio.to(torch.float16), attention_mask=in_audio_mask.to(torch.float16))
                        in_audio_mask = None
                        in_audio = in_audio.last_hidden_state.to(dtype)
                elif audio_encoder_name == "wavlm":
                    with torch.no_grad():
                        dtype = in_audio.dtype
                        in_audio = self.audio_encoder[0](
                            list([_.cpu().numpy() for _ in in_audio]),
                            sampling_rate=self.config.data.audio_sr,
                            padding=True,
                            return_tensors="pt"
                        )
                        in_audio_mask = in_audio.attention_mask.to(self.device)
                        in_audio = in_audio.input_values.to(self.device)
                        in_audio = self.audio_encoder[1](
                            in_audio, attention_mask=in_audio_mask)
                        in_audio_mask = None
                        in_audio = in_audio.last_hidden_state.to(dtype)
                else:
                    raise ValueError(f"Invalid audio encoder: {audio_encoder_name}")
            else: # 用mel谱保底
                with torch.no_grad():
                    in_audio = self.audio_encoder(in_audio).transpose(-1, -2)
                in_audio_mask = None
            if getattr(
                self.speech_model,
                "requires_contextual_audio_features",
                False,
            ):
                in_face_audio = self.get_input(batch, "face_audio_feat")
                in_face_audio_valid_length = self.get_input(
                    batch,
                    "face_audio_valid_length",
                )
                in_face_audio_crop_start = self.get_input(
                    batch,
                    "face_audio_crop_start",
                )
                if (
                    in_face_audio is None
                    or in_face_audio_valid_length is None
                    or in_face_audio_crop_start is None
                ):
                    raise RuntimeError(
                        f"{self.speech_model.__class__.__name__} requires "
                        "face_audio_feat, face_audio_valid_length, and "
                        "face_audio_crop_start from the TalkSHOW LMDB"
                    )
                in_face_audio = in_face_audio.to(self.device)
                in_face_audio_valid_length = (
                    in_face_audio_valid_length.to(self.device)
                )
                in_face_audio_crop_start = in_face_audio_crop_start.to(
                    self.device
                )
        self.accelerator.wait_for_everyone()
        if am_time_all is not None:
            am_time_all += time.time() - am_time_start
        ########################################################################
        # for index
        hand_indices = GetBodyPartIndices("hand", "rot6d")
        upper_indices = GetBodyPartIndices("upper", "rot6d")
        lower_indices = GetBodyPartIndices("lower", "rot6d")
        face_joint_indices = GetBodyPartIndices("face", "rot6d")
        poses_indices = np.arange(330)
        trans_indices = np.array([330, 331, 332])  # the last 3 dims are translation
        contact_indices = np.array([333, 334, 335, 336])  # the last 4 dims are contact

        ###############################################################################
        data_dict = {
            "in_audio": in_audio,
            "in_audio_mask": in_audio_mask,
            "in_audio_wav": in_audio_wav,
            "in_face_audio": in_face_audio,
            "in_face_audio_valid_length": in_face_audio_valid_length,
            "in_face_audio_crop_start": in_face_audio_crop_start,
            "in_word": in_word,
            "in_word_mask": in_word_mask,
            "in_word_txt": in_word_txt,
            "tar_exps": tar_exps,
            "tar_trans": tar_trans,
            "tar_contact": tar_contact,
            "tar_beta": tar_beta,
            "tar_pose": tar_pose,
            "tar_pose_6d": tar_pose_6d,
            "tar_motion": tar_motion,
            "tar_id": tar_id,
            "ori_id": ori_id,
            "is_padding_sample": is_padding_sample,
            "lm_time_all": lm_time_all,
            "am_time_all": am_time_all,
            "hand_indices": hand_indices,
            "upper_indices": upper_indices,
            "lower_indices": lower_indices,
            "face_joint_indices": face_joint_indices,
            "trans_indices": trans_indices,
            "contact_indices": contact_indices,
            "poses_indices": poses_indices,
        }
        if self.config.train.normalize:
            data_dict["tar_trans"], data_dict["tar_pose_6d"]\
                = self.data._train_dataset.normalize(data_dict["tar_trans"], data_dict["tar_pose_6d"])     
            tar_pose_6d = data_dict["tar_pose_6d"]
            tar_trans = data_dict["tar_trans"]
            tar_contact = data_dict["tar_contact"]
            data_dict["tar_motion"]  = torch.cat([tar_pose_6d, tar_trans, tar_contact], dim=-1)
        data_dict = self.prepare_mask(data_dict)
        data_dict["J"] = self.get_input(batch, "J")
        if data_dict["J"] is not None:
            data_dict["J"] = data_dict["J"].to(self.device)
        return data_dict
    
    def first_stage_encode(self, tar_motion, vae_time_all = None):
        ####################### for vae ###############################
        if vae_time_all is not None:
            vae_time_start = time.time()
        
        if self.config.train.train_first_stage_model:
            results = self.first_stage_model.encode(motion=tar_motion)
        else:
            with torch.no_grad():
                results = self.first_stage_model.encode(motion=tar_motion)

        if vae_time_all is not None:
            self.accelerator.wait_for_everyone()
            vae_time_all += time.time() - vae_time_start
        
        latent_motion = results["motion"]
        latent_motion_index = results.get("motion_index", None)
        ########################################################################
        return {
            "latent_motion": latent_motion,
            "latent_motion_index": latent_motion_index,
            "vae_time_all": vae_time_all,
        }   

    def first_stage_decode(self, latent_motion):
        if self.load_vae_feat:
            results = {}
            results["motion"] = latent_motion[..., :self.joints*6]
            results["trans"] = latent_motion[..., self.joints*6:self.joints*6+3]
            results["contact"] = latent_motion[..., self.joints*6+3:self.joints*6+4]
        else:
            if self.config.train.train_first_stage_model:
                results = self.first_stage_model.decode(motion=latent_motion)
            else:
                with torch.no_grad():
                    results = self.first_stage_model.decode(motion=latent_motion)

        return results

    def forward_kinematics(self, batch, motion, trans):
        if self.config.train.normalize:
            denorm_trans, denorm_motion = self.data._train_dataset.denormalize(trans, motion)
        else:
            denorm_trans, denorm_motion = trans, motion
        J = self.get_input(batch, "J")
        # 支持 blockwise reshape 后的调用：如果 batch 维度被放大，则重复 J
        if J.shape[0] != denorm_motion.shape[0]:
            if denorm_motion.shape[0] % J.shape[0] != 0:
                raise ValueError(
                    f"forward_kinematics batch mismatch: motion B={denorm_motion.shape[0]}, J B={J.shape[0]}"
                )
            repeat_factor = denorm_motion.shape[0] // J.shape[0]
            J = J.repeat_interleave(repeat_factor, dim=0)
        B, L = denorm_motion.shape[0], denorm_motion.shape[1]
        posemat = rc.rotation_6d_to_matrix(denorm_motion.reshape((-1, self.joints, 6)))
        rest_joints = J.unsqueeze(1).expand((-1, L, -1, -1)).reshape((-1, self.joints,  3))
        vertices = self.fk_process(posemat, rest_joints).reshape((B, L, -1, 3)) 
        if not self.config.scheduler.get("use_trans_for_fk", True):
            return vertices
        return vertices + denorm_trans.unsqueeze(2)

    def post_process_motion(self, motion):
        if self.config.data.local == False:
            pred_poses = rc.rotation_6d_to_matrix(motion.reshape(-1, self.joints, 6))
            pred_poses = rc.global_to_local_batch(pred_poses)
            pred_poses = rc.matrix_to_rotation_6d(pred_poses).reshape(motion.shape[0], -1, self.joints * 6)
            return pred_poses
        else:
            return motion

    def get_expression(
        self,
        id,
        *args,
        face_audio_features=None,
        face_audio_valid_lengths=None,
        face_audio_crop_starts=None,
        target_length=None,
        **kwargs,
    ):
        id_for_speech = id.clone()
        if not getattr(
            self.speech_model, "handles_global_speaker_ids", False
        ):
            id_flag = id > 30
            if torch.any(id_flag):
                id_for_speech[id_flag] = id_for_speech[id_flag] - 30
        def _forward_expression():
            if face_audio_features is not None:
                contextual_forward = getattr(
                    self.speech_model,
                    "forward_contextual",
                    None,
                )
                if not callable(contextual_forward):
                    raise RuntimeError(
                        f"{self.speech_model.__class__.__name__} does not "
                        "implement forward_contextual"
                    )
                return contextual_forward(
                    face_audio_features,
                    pid=id_for_speech,
                    valid_lengths=face_audio_valid_lengths,
                    crop_starts=face_audio_crop_starts,
                    target_length=target_length,
                )
            return self.speech_model(pid=id_for_speech, *args, **kwargs)

        if self.config.train.get("finetune_speech_model", False):
            expression = _forward_expression()
        else:
            with torch.no_grad():
                expression = _forward_expression()
        return expression

    def get_audio_expression(self, batch, id, in_audio):
        face_audio = self.get_input(batch, "in_face_audio")
        if face_audio is None:
            return self.get_expression(id, in_audio)
        if not isinstance(in_audio, torch.Tensor) or in_audio.ndim < 2:
            raise ValueError(
                "frame-aligned audio context must have shape [B,T,...]"
            )
        return self.get_expression(
            id,
            in_audio,
            face_audio_features=face_audio,
            face_audio_valid_lengths=self.get_input(
                batch,
                "in_face_audio_valid_length",
            ),
            face_audio_crop_starts=self.get_input(
                batch,
                "in_face_audio_crop_start",
            ),
            target_length=int(in_audio.shape[1]),
        )


    def training_step(self, batch, step, vae_time_all=None, global_step=None): 
        in_id = self.get_input(batch, "tar_id")
        # for index
        hand_indices = self.get_input(batch, "hand_indices")
        upper_indices = self.get_input(batch, "upper_indices")
        lower_indices = self.get_input(batch, "lower_indices")
        face_joint_indices = self.get_input(batch, "face_joint_indices")
        trans_indices = self.get_input(batch, "trans_indices")
        contact_indices = self.get_input(batch, "contact_indices")
        pose_prompt = self.get_input(batch, "pose_prompt")
        pose_prompt_mask = self.get_input(batch, "pose_prompt_mask")
        trans_prompt = self.get_input(batch, "trans_prompt")
        trans_mask = self.get_input(batch, "trans_prompt_mask")        
        tar_exps = self.get_input(batch, "tar_exps")
        poses_indices = self.get_input(batch, "poses_indices")
        #########################################################################
        if self.config.train.task.startswith("audio2"):
            in_audio = self.get_input(batch, "in_audio")
            in_audio_mask = self.get_input(batch, "in_audio_mask")
            expression = self.get_audio_expression(batch, in_id, in_audio)
            model_kwargs = dict(
                expression=expression,
                audio=in_audio,
                audio_mask=in_audio_mask,
                speaker_id=in_id,
                hand_indices=hand_indices,
                upper_indices=upper_indices,
                lower_indices=lower_indices,
                face_joint_indices=face_joint_indices,
                trans_indices=trans_indices,
                contact_indices=contact_indices,
                poses_indices=poses_indices,
                pose_prompt=pose_prompt,
                pose_prompt_mask=pose_prompt_mask,
                trans_prompt=trans_prompt,
                trans_prompt_mask=trans_mask,
            )
        elif self.config.train.task.startswith("text2"):
            in_word = self.get_input(batch, "in_word")
            in_word_mask = self.get_input(batch, "in_word_mask")
            expression = self.get_expression(in_id, in_word)
            model_kwargs = dict(
                expression=expression,
                txt=in_word,
                txt_mask=in_word_mask,
                speaker_id=in_id,
                hand_indices=hand_indices,
                upper_indices=upper_indices,
                lower_indices=lower_indices,
                face_joint_indices=face_joint_indices,
                trans_indices=trans_indices,
                contact_indices=contact_indices,
                poses_indices=poses_indices,
                pose_prompt=pose_prompt,
                pose_prompt_mask=pose_prompt_mask,
                trans_prompt=trans_prompt,
                trans_prompt_mask=trans_mask,
            )
        elif self.config.train.task.startswith("text+audio2"):
            in_word = self.get_input(batch, "in_word")
            in_word_mask = self.get_input(batch, "in_word_mask")
            in_audio = self.get_input(batch, "in_audio")
            in_audio_mask = self.get_input(batch, "in_audio_mask")
            expression = self.get_expression(in_id, in_word, in_audio)
            model_kwargs = dict(
                expression=expression,
                txt=in_word,
                txt_mask=in_word_mask,
                audio=in_audio,
                audio_mask=in_audio_mask,
                speaker_id=in_id,
                hand_indices=hand_indices,
                upper_indices=upper_indices,
                lower_indices=lower_indices,
                face_joint_indices=face_joint_indices,
                trans_indices=trans_indices,
                contact_indices=contact_indices,
                poses_indices=poses_indices,
                pose_prompt=pose_prompt,
                pose_prompt_mask=pose_prompt_mask,
                trans_prompt=trans_prompt,
                trans_prompt_mask=trans_mask,
            )
        else:
            raise ValueError(f"Invalid task: {self.config.train.task}")
        tar_motion = self.get_input(batch, "tar_motion")
        tar_beta = self.get_input(batch, "tar_beta")
        
        loss_term = {}
        results = self.first_stage_encode(tar_motion, vae_time_all)
        latent_motion = results["latent_motion"]
        vae_time_all = results["vae_time_all"]
        # vae_time_all = 0
        # latent_motion = tar_motion
        # Sample a random timestep for each modality
        bs = tar_motion.shape[0]
        timestep = self.sample_timestep(bs)
        #########################################################
        model_fn = lambda *args, **kwargs: self.model(**self.guider(
            *args, **kwargs, is_sample=False, task=self.config.train.task))
        
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
            pose_prompt_mask=pose_prompt_mask,
            trans_prompt_mask=trans_mask,
        )
        if self.config.train.get("finetune_speech_model", False):
            raise NotImplementedError("Finetuning speech model is not supported yet")
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
        pose_prompt = self.get_input(batch, "pose_prompt")
        pose_prompt_mask = self.get_input(batch, "pose_prompt_mask")
        trans_prompt = self.get_input(batch, "trans_prompt")
        trans_mask = self.get_input(batch, "trans_prompt_mask")
        # for index
        hand_indices = self.get_input(batch, "hand_indices")
        upper_indices = self.get_input(batch, "upper_indices")
        lower_indices = self.get_input(batch, "lower_indices")
        face_joint_indices = self.get_input(batch, "face_joint_indices")
        trans_indices = self.get_input(batch, "trans_indices")
        contact_indices = self.get_input(batch, "contact_indices")
        poses_indices = self.get_input(batch, "poses_indices")
        #########################################################################
        
        if self.config.train.task.startswith("audio2"):
            in_audio = self.get_input(batch, "in_audio") # 这个是wav
            in_audio_mask = self.get_input(batch, "in_audio_mask")
            audio_original = self.get_input(batch, "in_audio_wav")
            expression = self.get_audio_expression(batch, in_id, in_audio)
            model_kwargs = dict(
                expression=expression,
                audio=in_audio,
                audio_mask=in_audio_mask,
                speaker_id=in_id,
                hand_indices=hand_indices,
                upper_indices=upper_indices,
                lower_indices=lower_indices,
                face_joint_indices=face_joint_indices,
                trans_indices=trans_indices,
                contact_indices=contact_indices,
                poses_indices=poses_indices,
                pose_prompt=pose_prompt,
                pose_prompt_mask=pose_prompt_mask,
                trans_prompt=trans_prompt,
                trans_prompt_mask=trans_mask,
            )
        elif self.config.train.task.startswith("text2"):
            txt = self.get_input(batch, "in_word")
            txt_mask = self.get_input(batch, "in_word_mask")
            txt_original = self.get_input(batch, "in_word_txt")
            expression = self.get_expression(in_id, txt)
            model_kwargs = dict(
                expression=expression,
                txt=txt,
                txt_mask=txt_mask,
                speaker_id=in_id,
                hand_indices=hand_indices,
                upper_indices=upper_indices,
                lower_indices=lower_indices,
                face_joint_indices=face_joint_indices,
                trans_indices=trans_indices,
                contact_indices=contact_indices,
                pose_prompt=pose_prompt,
                pose_prompt_mask=pose_prompt_mask,
                trans_prompt=trans_prompt,
                trans_prompt_mask=trans_mask,
            )
        else:
            raise ValueError(f"Invalid task: {self.config.train.task}")
        
        # Sample a random timestep for each modality
        bs = tar_motion.shape[0]
        results = self.first_stage_encode(tar_motion)
        latent_motion = results["latent_motion"]
        if choice == "validation":
            timestep = self.sample_timestep(bs)
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
                pose_prompt_mask=pose_prompt_mask,
                trans_prompt_mask=trans_mask,
            )

        ########################################################################
        
        if self.config.train.task.split("2")[0] == "text":
            assert null_embed is not None
            model_kwargs = dict(
                expression=expression,
                txt=txt, 
                txt_mask=txt_mask, 
                speaker_id=in_id, 
                hand_indices=hand_indices,
                upper_indices=upper_indices,
                lower_indices=lower_indices,
                face_joint_indices=face_joint_indices,
                trans_indices=trans_indices,
                contact_indices=contact_indices,
                pose_prompt=pose_prompt,
                pose_prompt_mask=pose_prompt_mask,
                trans_prompt=trans_prompt,
                trans_prompt_mask=trans_mask,
            )
        elif self.config.train.task.split("2")[0] == "audio":
            model_kwargs = dict(
                expression=expression,
                audio=in_audio, 
                audio_mask=in_audio_mask,
                speaker_id=in_id, 
                hand_indices=hand_indices,
                upper_indices=upper_indices,
                lower_indices=lower_indices,
                face_joint_indices=face_joint_indices,
                trans_indices=trans_indices,
                contact_indices=contact_indices,
                poses_indices=poses_indices,
                pose_prompt=pose_prompt,
                pose_prompt_mask=pose_prompt_mask,
                trans_prompt=trans_prompt,
                trans_prompt_mask=trans_mask,
            )
        # elif self.config.train.task.split("2")[0] == "text+audio":
        #     model_kwargs = dict(
        #         expression=expression,
        #         txt=txt, 
        #         txt_mask=txt_mask, 
        #         audio=in_audio, 
        #         audio_mask=in_audio_mask,
        #         speaker_id=in_id, 
        #         hand_indices=hand_indices,
        #         upper_indices=upper_indices,
        #         lower_indices=lower_indices,
        #         face_joint_indices=face_joint_indices,
        #         trans_indices=trans_indices,
        #         contact_indices=contact_indices,
        #         poses_indices=poses_indices,
        #         pose_prompt=pose_prompt,
        #         pose_prompt_mask=pose_prompt_mask,
        #         trans_prompt=trans_prompt,
        #         trans_prompt_mask=trans_mask,
        #     )
        
        
        def model_fn(*args, **kwargs):
            results =  self.model_instance(**self.guider(
                *args, **kwargs, model_kwargs=model_kwargs, is_sample=True, task=self.config.train.task))
            values = {}
            for key, value in results.items():
                if key in ["motion"]:
                    values[key] = value
            values = self.guider.guide(
                **values, 
                timestep=args[1],
                task=self.config.train.task
            ) # guidance here
            return values["motion"]
            # results =  self.model_instance(*args, **kwargs, **model_kwargs)
            # return results["motion"]
                
        motion_noise = self.get_initial_noise(latent_motion)
        
        if_vis_intermediate = if_visualize \
            and self.config.train.get("visualize_intermediate", False)
        if_vis_model_output = if_visualize \
            and self.config.train.get("visualize_model_output", False)
        if_vis_adding_noise = if_visualize \
            and self.config.train.get("visualize_adding_noise", False)
        
        # times = reversed(torch.linspace(0, 1, 11).to(self.device, dtype=torch.float32))
        # next_times = times[1:]
        # xt = motion_noise.clone()
        # for time, next_time in zip(times, next_times):
        #     motion_results = model_fn(xt, time.unsqueeze(0).repeat(xt.shape[0]) * 1000)
        #     motion_results = torch.cat(
        #         [rc.NormalizeRot6d(motion_results[:, :, :-3]), motion_results[:, :, -3:]],
        #         dim=-1,
        #     )
        #     nt_bc = expand_dims_like(next_time, motion_results)
        #     xt = motion_noise *  nt_bc + motion_results * (1 - nt_bc)
        # decode_results = {
        #     "motion": xt[:, :, :self.joints*6],
        #     "trans": xt[:, :, self.joints*6:self.joints*6+3],
        # }
        # extra = {}
        # 是否收集 rot6d 流形统计
        return_rot6d_stats = self.config.train.get("analyze_rot6d_manifold", False)
        
        motion_latents, extra = self.sampler.sample(
            model_fn,
            motion_noise,
            return_intermediate=if_vis_intermediate,
            return_model_output=if_vis_model_output,
            return_rot6d_stats=return_rot6d_stats,
            rot6d_dim=self.joints * 6,
            compute_per_joint_stats=self.config.train.get("analyze_rot6d_per_joint", False),
            # 新接口：post_process_func 现在接收额外的上下文参数
            # post_process_func=lambda x, timestep=None, step_idx=None, total_steps=None, **kwargs: torch.cat([ 
            #     rc.NormalizeRot6d(x[:, :, :-3]),
            #     x[:, :, -3:],
            # ], dim=-1),
        )
        decode_results = self.first_stage_decode(motion_latents)
        
        motion_results = decode_results["motion"]
        motion_results = self.post_process_motion(motion_results)
        motion_trans = decode_results["trans"]
        if self.config.train.normalize:
            motion_trans, motion_results[..., :6*self.joints] \
                = self.data._train_dataset.denormalize(motion_trans, motion_results[..., :6*self.joints])
            tar_trans, tar_motion[..., :6*self.joints] = self.data._train_dataset.denormalize(tar_trans, tar_motion[..., :6*self.joints])
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
            
            # 保存 rot6d 流形统计
            if "rot6d_stats" in extra:
                self.save_rot6d_stats(
                    extra["rot6d_stats"],
                    step=step,
                    global_step=global_step,
                    epoch=epoch,
                    choice=choice,
                    ori_id=ori_id,
                )
        
        self.accelerator.wait_for_everyone() # 等主进程把文件写好
        if self.config.metrics.evaluator.get("evaluate_on_disk", False):
            # 如果在这种情况下，最好把可视化npz和audio wav给禁用掉，没必要了
            keep_every_epoch = self.config.metrics.evaluator.get("keep_every_epoch", False)
            prefix = f"gs{global_step:06d}_" if global_step is not None else ""
            prefix += f"epoch{epoch:06d}_" if epoch is not None else ""
            prefix += f"step{step:06d}_" if step is not None else ""
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
                prefix += f"bs{i}_"
                if not keep_every_epoch:
                    prefix = ""
                res_path = osp.join(self.config.work_dir, self.npz_save_name, choice,
                    f"rank{self.accelerator.process_index}_{prefix}resface_" + ori_id[i] + ".npz")
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
        
   
    def get_split(self, data, start, end):
        return data[:, start:end].clone() if data is not None else None
    
    def debug_gradient(self, global_step, loss_term):
        if self.config.train.debug_gradient_interval > 0 and (global_step - 1) % self.config.train.debug_gradient_interval == 0:
            norms = {}
            item_grad_norms = per_loss_grad_norms(loss_term, self.model, norm_type=2.0, ddp_sync_norm=True)
            for k, v in item_grad_norms.items():
                norms[f"gradnorm_item/{k}"] = v.item()
            if self.accelerator.is_main_process:
                self.logger.info(
                    colored(
                        f"Gradient norms: {sorted(norms.items(), key=lambda x: -x[1])[:10]}",
                        color="red",
                        attrs=["bold"],
                    )
                )
                self.accelerator.log(norms)
            self.model.zero_grad(set_to_none=True)
        
    def get_debug_npz_file(self, tar_beta, rec_pose, expression, trans, name="debug.npz"):
        rec_pose = self.post_process_motion(rec_pose)
        rec_pose = rc.rotation_6d_to_axis_angle(rec_pose.view(rec_pose.shape[0], -1, 6)).view(rec_pose.shape[0], -1, 3 * self.joints)
        
        tar_beta_np = tar_beta.detach().cpu().numpy()
        rec_pose_np = rec_pose.detach().cpu().numpy()
        rec_exp_np = expression.detach().cpu().numpy()
        rec_trans_np = trans.detach().cpu().numpy()

        np.savez(
            name,
            betas=tar_beta_np[0], 
            poses=rec_pose_np,
            expressions=rec_exp_np,
            trans=rec_trans_np,
            model='smplx2020',
            gender='neutral',
            mocap_frame_rate = 30,
        )

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
        in_id = self.get_input(batch, "tar_id")
        tar_motion = self.get_input(batch, "tar_motion")
        tar_trans = self.get_input(batch, "tar_trans")
        tar_exps = self.get_input(batch, "tar_exps")
        tar_beta = self.get_input(batch, "tar_beta")
        tar_pose = self.get_input(batch, "tar_pose_6d")
        ori_id = self.get_input(batch, "ori_id")
        # for index
        hand_indices = self.get_input(batch, "hand_indices")
        upper_indices = self.get_input(batch, "upper_indices")
        lower_indices = self.get_input(batch, "lower_indices")
        face_joint_indices = self.get_input(batch, "face_joint_indices")
        trans_indices = self.get_input(batch, "trans_indices")
        contact_indices = self.get_input(batch, "contact_indices")
        poses_indices = self.get_input(batch, "poses_indices")
        
        in_audio = self.get_input(batch, "in_audio")
        txt = self.get_input(batch, "in_word")
        txt_mask = self.get_input(batch, "in_word_mask")
        audio_original = self.get_input(batch, "in_audio_wav")
        txt_original = self.get_input(batch, "in_word_txt")
        
        bs = tar_motion.shape[0]
        base_pose_length = self.config.data.pose_length # 为单位进行切分
        pose_fps = self.config.data.pose_fps
        test_length = tar_motion.shape[1] # 实际的长度
        overlap_length = self.config.scheduler.overlap_length
        pose_prompt_seed = tar_pose[:, :overlap_length].clone()
        trans_prompt_seed = tar_trans[:, :overlap_length].clone()
        
        # 留好位置
        motion_results_all = []
        expression_results_all = []
        trans_results_all = []
        
        # full_audio_feat = None
        # if self.config.train.task.startswith("audio"):
        #     if self.load_audio_feat:
        #         full_audio_feat = in_audio
        #         full_audio_feat_mask = None
        #     else:
        #         sr = self.config.data.audio_sr
        #         audio_start, audio_end = 0, int(test_length * sr / pose_fps)
        #         full_audio_feat = self.get_split(in_audio, audio_start, audio_end)
        #         audio_encoder_config = self.config.audio.audio_encoder
        #         audio_encoder_name = audio_encoder_config["name"]
        #         if audio_encoder_name == "hubert-large-ls960-ft" or audio_encoder_name == "wavlm":
        #             dtype = full_audio_feat.dtype
        #             full_audio_feat = self.audio_encoder[0](
        #                 list([_.cpu().numpy() for _ in full_audio_feat]), 
        #                 sampling_rate=sr,
        #                 padding=True, 
        #                 return_tensors="pt"
        #             )
        #             full_audio_feat_mask = full_audio_feat.attention_mask.to(self.device)
        #             full_audio_feat = full_audio_feat.input_values.to(self.device)
        #             if not self.config.audio.audio_encoder.get("quantization", False):
        #                 full_audio_feat = self.audio_encoder[1](
        #                     full_audio_feat, 
        #                     attention_mask=full_audio_feat_mask)
        #             else:
        #                 full_audio_feat = self.audio_encoder[1](
        #                     full_audio_feat.to(torch.float16), 
        #                     attention_mask=full_audio_feat_mask.to(torch.float16))
        #             full_audio_feat_mask = None
        #             full_audio_feat = full_audio_feat.last_hidden_state.to(dtype)
        #         else:
        #             raise ValueError(f"Invalid audio encoder: {audio_encoder_name}")
        #         full_audio_feat = torch.nn.functional.interpolate(
        #             full_audio_feat.permute(0, 2, 1), size=(test_length,), mode='linear', align_corners=True)\
        #                 .permute(0, 2, 1)
        # smoothing_window_length = 7
        # filter = Filter1D(window_length=smoothing_window_length, polyorder=2).cuda()
        for start in range(0, test_length, base_pose_length - overlap_length):
            pose_length = min(base_pose_length, test_length - start)
            pose_prompt_mask = torch.full((bs, pose_length), True, device=self.device)
            pose_prompt_mask[:, :overlap_length] = False
            trans_prompt_mask = torch.full((bs, pose_length), True, device=self.device)
            trans_prompt_mask[:, :overlap_length] = False

            pose_prompt = torch.zeros((bs, pose_length, tar_pose.shape[2]), device=self.device, dtype=tar_pose.dtype)
            trans_prompt = torch.zeros((bs, pose_length, tar_trans.shape[2]), device=self.device, dtype=tar_trans.dtype)
            pose_seed_len = min(pose_prompt_seed.shape[1], pose_length)
            trans_seed_len = min(trans_prompt_seed.shape[1], pose_length)
            pose_prompt[:, :pose_seed_len] = pose_prompt_seed[:, :pose_seed_len]
            trans_prompt[:, :trans_seed_len] = trans_prompt_seed[:, :trans_seed_len]
            pose_prompt[:,pose_seed_len:] = 0
            trans_prompt[:,trans_seed_len:] = 0
            
            end = start + pose_length
            if end > test_length:
                end = test_length
            
            tar_motion_ = self.get_split(tar_motion, start, end)   
            
            if self.config.train.task.startswith("audio"):            
                sr = self.config.data.audio_sr
                audio_start, audio_end = int(start * sr / pose_fps), int(end * sr / pose_fps)
                audio_original_ = self.get_split(audio_original, audio_start, audio_end)
                audio_encoder_config = self.config.audio.audio_encoder
                audio_encoder_name = audio_encoder_config["name"]
                if audio_encoder_name == "hubert-large-ls960-ft" or audio_encoder_name == "wavlm":
                    dtype = audio_original_.dtype
                    in_audio_ = self.audio_encoder[0](
                        list([_.cpu().numpy() for _ in audio_original_]), 
                        sampling_rate=sr,
                        padding=True, 
                        return_tensors="pt"
                    )
                    in_audio_mask = in_audio_.attention_mask.to(self.device)
                    in_audio_ = in_audio_.input_values.to(self.device)
                    if not self.config.audio.audio_encoder.get("quantization", False):
                        in_audio_ = self.audio_encoder[1](
                            in_audio_, 
                            attention_mask=in_audio_mask)
                    else:
                        in_audio_ = self.audio_encoder[1](
                            in_audio_.to(torch.float16), 
                            attention_mask=in_audio_mask.to(torch.float16))
                    in_audio_mask_ = None
                    in_audio_ = in_audio_.last_hidden_state.to(dtype)
                else:
                    raise ValueError(f"Invalid audio encoder: {audio_encoder_name}")
                in_audio_ = torch.nn.functional.interpolate(
                    in_audio_.permute(0, 2, 1), size=(pose_length,), mode='linear', align_corners=True)\
                        .permute(0, 2, 1)
                txt_ = None
                txt_mask_ = None
            elif self.config.train.task.startswith("text"):
                in_audio_ = None
                in_audio_mask_ = None
                # TODO: 解析txt
                txt_ = self.get_split(txt, start, end)
                txt_mask_ = self.get_split(txt_mask, start, end)
            else:
                raise ValueError(f"Invalid task: {self.config.train.task}")
            
            txt_ = self.get_split(txt, start, end)
            txt_mask_ = self.get_split(txt_mask, start, end)
            
            results = self.first_stage_encode(tar_motion_)
            latent_motion = results["latent_motion"]
               
            if self.config.train.task.startswith("audio2"):
                expression = self.speech_model(in_audio_, pid=in_id)
                model_kwargs = dict(
                    expression=expression,
                    audio=in_audio_,
                    audio_mask=in_audio_mask_,
                    speaker_id=in_id,
                    hand_indices=hand_indices,
                    upper_indices=upper_indices,
                    lower_indices=lower_indices,
                    face_joint_indices=face_joint_indices,
                    trans_indices=trans_indices,
                    contact_indices=contact_indices,
                    poses_indices=poses_indices,
                    pose_prompt=pose_prompt,
                    pose_prompt_mask=pose_prompt_mask,
                    trans_prompt=trans_prompt,
                    trans_prompt_mask=trans_prompt_mask,
                )
            elif self.config.train.task.startswith("text2"):
                expression = self.speech_model(txt_, pid=in_id)
                model_kwargs = dict(
                    expression=expression,
                    txt=txt_,
                    txt_mask=txt_mask_,
                    speaker_id=in_id,
                    hand_indices=hand_indices,
                    upper_indices=upper_indices,
                    lower_indices=lower_indices,
                    face_joint_indices=face_joint_indices,
                    trans_indices=trans_indices,
                    contact_indices=contact_indices,
                    poses_indices=poses_indices,
                    pose_prompt=pose_prompt,
                    pose_prompt_mask=pose_prompt_mask,
                    trans_prompt=trans_prompt,
                    trans_prompt_mask=trans_prompt_mask,
                )
            else:
                raise ValueError(f"Invalid task: {self.config.train.task}")
            
            def model_fn(*args, model_fn_kwargs=None, **kwargs):
                results =  self.model_instance(**self.guider(
                    *args, **kwargs, model_kwargs=model_fn_kwargs, is_sample=True, task=self.config.train.task))
                values = {}
                for key, value in results.items():
                    if key in ["motion"]:
                        values[key] = value
                values = self.guider.guide(**values, timestep=args[1], task=self.config.train.task) # guidance here
                return values["motion"]
            model_func = partial(model_fn, model_fn_kwargs=model_kwargs)
            
            motion_noise = self.get_initial_noise(latent_motion)
            # 条件化的 post_process_func：可以根据 timestep/step_idx 决定是否应用
            # 例如：只在后 N 步（timestep < threshold）时应用归一化
            normalize_timestep_threshold = self.config.train.get("normalize_rot6d_timestep_threshold", None)
            
            def conditional_normalize_rot6d(x, timestep=None, step_idx=None, total_steps=None, **kwargs):
                """
                根据时间步决定是否应用 NormalizeRot6d。
                
                Args:
                    x: 预测的运动张量 (B, T, D)
                    timestep: 当前时间步值（float，通常从 1000 到接近 0）
                    step_idx: 当前步数索引（int，从 0 开始）
                    total_steps: 总采样步数
                """
                # 如果未设置阈值，则始终不应用归一化
                if normalize_timestep_threshold is None:
                    return x
                
                # 只在 timestep 小于阈值时应用归一化（即采样后期）
                if timestep is not None and timestep < normalize_timestep_threshold:
                    return torch.cat([ 
                        rc.NormalizeRot6d(x[:, :, :-3]),
                        x[:, :, -3:],
                    ], dim=-1)
                return x
            
            motion_latents, extra = self.sampler.sample(
                model_func,
                motion_noise,
                return_intermediate=False,
                return_model_output=False,
                post_process_func=conditional_normalize_rot6d,
            )
            decode_results = self.first_stage_decode(motion_latents)
            
            motion_results = decode_results["motion"]
            trans_results = decode_results["trans"]
            # if motion_results.shape[1] == base_pose_length: # 规避最后一段
            #     results = torch.cat([motion_results, trans_results], dim=-1)
            #     results[:,overlap_length-smoothing_window_length//2:] = filter(results[:,overlap_length-smoothing_window_length//2:])
            #     motion_results = results[:,:,:6*self.joints]
            #     trans_results = results[:,:,6*self.joints:]
            motion_results = rc.NormalizeRot6d(motion_results)
            

            motion_results_all.append(motion_results.clone().detach())
            expression_results_all.append(expression.clone().detach())
            trans_results_all.append(trans_results.clone().detach())

            if motion_results.shape[1] < overlap_length: # 实际并不重要，出现这种情况就是最后一段了，prompt反正也用不上了
                pose_prompt_seed = motion_results.clone().detach()
                trans_prompt_seed = trans_results.clone().detach()
            else:
                pose_prompt_seed = motion_results[:, -overlap_length:].clone().detach()
                trans_prompt_seed = trans_results[:, -overlap_length:].clone().detach()
            

        for i in range(1, len(motion_results_all)):
            n_overlap = min(motion_results_all[i].shape[1], overlap_length)
            motion_results_all[i-1][:,-n_overlap:] = \
                (motion_results_all[i-1][:,-n_overlap:] + motion_results_all[i][:,:n_overlap])/2.0
            motion_results_all[i] = motion_results_all[i][:,n_overlap:]
            expression_results_all[i-1][:,-n_overlap:] = \
                (expression_results_all[i-1][:,-n_overlap:] + expression_results_all[i][:,:n_overlap])/2.0
            expression_results_all[i] = expression_results_all[i][:,n_overlap:]
            trans_results_all[i-1][:,-n_overlap:] = \
                (trans_results_all[i-1][:,-n_overlap:] + trans_results_all[i][:,:n_overlap])/2.0
            trans_results_all[i] = trans_results_all[i][:,n_overlap:]
            
        motion_results_all = torch.cat(motion_results_all, dim=1)
        expression_results_all = torch.cat(expression_results_all, dim=1)
        trans_results_all = torch.cat(trans_results_all, dim=1)
        if self.config.train.normalize:
            trans_results_all, motion_results_all = self.data._train_dataset.denormalize(trans_results_all, motion_results_all)
            tar_trans, tar_pose = self.data._train_dataset.denormalize(tar_trans, tar_pose)
        motion_results_all = self.post_process_motion(motion_results_all)
        tar_pose_processed = self.post_process_motion(tar_pose)
        
        motion_results_all_axis_angle = rc.rotation_6d_to_axis_angle(
            motion_results_all.view(bs, motion_results_all.shape[1], -1, 6))\
            .view(bs, -1, 3 * self.joints)
        tar_motion_axis_angle = rc.rotation_6d_to_axis_angle(tar_pose_processed.view(bs, tar_pose_processed.shape[1], -1, 6))\
            .view(bs, -1, 3 * self.joints)

                        
        tar_beta_np = tar_beta.detach().cpu().numpy()
        rec_pose_np = motion_results_all_axis_angle.detach().cpu().numpy()
        rec_exp_np = expression_results_all.detach().cpu().numpy()
        rec_trans_np = trans_results_all.detach().cpu().numpy()
        tar_pose_np = tar_motion_axis_angle.detach().cpu().numpy()
        tar_exp_np = tar_exps.detach().cpu().numpy()
        tar_trans_np = tar_trans.detach().cpu().numpy()
        tar_audio_np = audio_original.detach().cpu().numpy()
        audio_sr = self.config.data.audio_sr
        
        res_paths = []
        gt_paths = []
        gt_audio_paths = []
        for i in range(bs):
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
                "motion": motion_results_all_axis_angle,
                "expression": expression_results_all,
                "gt_motion": tar_motion_axis_angle,
                "gt_expression": tar_exps,
                "gt_audio": audio_original,
                "betas": tar_beta
            }

        return results
        
    def get_initial_noise(self, latent_motion):
        # =========================================================================
        # motion_noise = torch.randn_like(latent_motion)
        motion_noise = torch.randn(
            (latent_motion.shape[0], latent_motion.shape[1], latent_motion.shape[2]),
            device=latent_motion.device,
            dtype=latent_motion.dtype,
            generator=self.generator,
        )
        return motion_noise

    def visualize_sequence(
        self,
        motion_list,
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
        npz_save_name = self.npz_save_name
        
        for idx, motion in enumerate(motion_list):
            n = motion.shape[1]
            if vis_bs == 1:
                motion = motion[0].unsqueeze(0) # 只可视化第一个数据
            else:
                motion = motion[0:vis_bs]
            
            decode_results = self.first_stage_decode(motion)
            
            motion_results = decode_results["motion"]
            motion_results = self.post_process_motion(motion_results)
            motion_trans = decode_results["trans"]
            
            if "motion_axis_angle" in decode_results:
                motion_results_axis_angle = decode_results["motion_axis_angle"]\
                    .view(vis_bs, -1, 3 * self.joints)
            else:
                motion_results_axis_angle = rc.rotation_6d_to_axis_angle(
                    motion_results.view(vis_bs, motion_results.shape[1], -1, 6))\
                        .view(vis_bs, -1, 3 * self.joints)
            
            vertices_rec, vertices_rec_face, _ = self.get_vertices(
                motion_results_axis_angle, _tar_exps, motion_trans, 
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
                rec_pose_np = motion_results_axis_angle[i].detach().cpu().numpy()
                rec_trans_np = motion_trans[i].detach().cpu().numpy()
                rec_exp_np = _tar_exps[i].detach().cpu().numpy() 
                tar_beta_np = tar_beta[i].detach().cpu().numpy()

                
                np.savez(
                    osp.join(self.config.work_dir, npz_save_name, choice, 
                            f"{prefix}bs{i}_{vis_type}{idx}_" + ori_id[i] +'.npz'),
                    betas=tar_beta_np[0],
                    poses=rec_pose_np,
                    expressions=rec_exp_np,
                    trans=rec_trans_np,
                    model='smplx2020',
                    gender='neutral',
                    mocap_frame_rate = 30,
                )

    def save_rot6d_stats(
        self, 
        rot6d_stats, 
        step=None, 
        global_step=None, 
        epoch=None, 
        choice="validation",
        ori_id=None,
    ):
        """
        保存 rot6d 流形统计到 JSON 文件，并打印关键统计到日志。
        
        Args:
            rot6d_stats: 从采样器返回的统计列表
            step: 当前 step
            global_step: 全局 step
            epoch: 当前 epoch
            choice: "validation" 或 "test"
            ori_id: 样本 ID
        """
        # 创建保存目录
        stats_dir = osp.join(self.config.work_dir, "rot6d_stats", choice)
        os.makedirs(stats_dir, exist_ok=True)
        
        # 构建文件名前缀
        prefix = ""
        if global_step is not None:
            prefix += f"gs{global_step:06d}_"
        if epoch is not None:
            prefix += f"epoch{epoch:06d}_"
        if step is not None:
            prefix += f"step{step:06d}_"
        if ori_id is not None and len(ori_id) > 0:
            prefix += f"{ori_id[0]}_"
        
        # 保存 JSON
        json_path = osp.join(stats_dir, f"{prefix}rot6d_stats.json")
        with open(json_path, "w") as f:
            json.dump(rot6d_stats, f, indent=2)
        
        # 打印关键统计到日志
        self.logger.info(
            colored(
                f"[Rot6D Manifold Stats] Saved to {json_path}",
                "cyan",
                attrs=["bold"],
            )
        )
        
        # 打印摘要：第一步和最后一步的对比
        if len(rot6d_stats) >= 2:
            first_stats = rot6d_stats[0]  # 初始噪声（step=-1）
            last_stats = rot6d_stats[-1]  # 最后一步
            
            summary = (
                f"\n{'='*60}\n"
                f"Rot6D Manifold Distance Summary (steps={len(rot6d_stats)-1}):\n"
                f"{'='*60}\n"
                f"{'Metric':<25} {'Initial (t=1)':<15} {'Final (t≈0)':<15} {'Change':<15}\n"
                f"{'-'*60}\n"
            )
            
            metrics_to_compare = [
                ("proj_l2_mean", "Projection L2"),
                ("a1_norm_err_mean", "a1 Norm Error"),
                ("a2_norm_err_mean", "a2 Norm Error"),
                ("orth_err_mean", "Orthogonality Err"),
                ("gs_correction_mean", "GS Correction"),
            ]
            
            for key, name in metrics_to_compare:
                if key in first_stats and key in last_stats:
                    init_val = first_stats[key]
                    final_val = last_stats[key]
                    change = final_val - init_val
                    change_pct = (change / (init_val + 1e-8)) * 100
                    summary += f"{name:<25} {init_val:<15.6f} {final_val:<15.6f} {change:+.6f} ({change_pct:+.1f}%)\n"
            
            summary += f"{'='*60}"
            self.logger.info(summary)
        
        # 返回 JSON 路径供后续使用
        return json_path

    def predict_step(self, batch, null_embed, step):
        return self.shared_test_predict_step(
            batch, null_embed, step, choice="predict", global_step = None, epoch = None, visualize = True)
        
    def validation_step(self, batch, null_embed, step, global_step = None, epoch = None, visualize = True):
        return {}, self.shared_test_predict_step(
            batch, null_embed, step, choice="validation", global_step = global_step, epoch = epoch, visualize = visualize)
