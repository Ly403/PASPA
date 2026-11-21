
import os
import os.path as osp
import time
import torch
import datetime
from copy import deepcopy
from diffusion.model.builder import build_model, get_tokenizer_and_text_encoder
from diffusion.utils.logger import LogBuffer, get_root_logger
from diffusion.utils.checkpoint import load_checkpoint, save_checkpoint, _resolve_latest_ckpt_path, _parse_epoch_step_from_name
from termcolor import colored
from diffusion import Scheduler
from diffusion.model.model_growth_utils import ModelGrowthInitializer
from diffusion.utils.optimizer import auto_scale_lr, build_optimizer
from diffusion.utils.lr_scheduler import build_lr_scheduler
from diffusion.utils.dist_utils import flush, get_world_size
from diffusion.utils.misc import DebugUnderflowOverflow
from diffusion.utils.config import instantiate_from_config, get_obj_from_str, set_value
from diffusion.model.utils import get_weight_dtype
from diffusion.guider.guider import VanillaOmniFlowCFGuider, IdentityGuider
from accelerate.utils import broadcast_object_list
import numpy as np
from torch.nn.utils import clip_grad_norm_, clip_grad_value_
import random
from diffusion.utils.snapshot import snapshot, sampler_fingerprint
from contextlib import contextmanager
import torch.distributed as dist

DEFAULT_FIRST_STAGE_MODEL_CONFIG = dict({
    "target": "diffusion.model.vae.base.IdentityAutoEncoder",
})

def grad_norm_of_loss(loss_scalar, model, retain_graph=True):
    model.zero_grad(set_to_none=True)
    loss_scalar.backward(retain_graph=retain_graph)
    # 只读不裁剪：用一个很大的 max_norm
    total_norm = clip_grad_norm_(model.parameters(), max_norm=1e9)
    return float(total_norm.detach().cpu())

def check_model_finite(model, tag):
    bad = []
    for n, p in model.named_parameters():
        if p is None: 
            continue
        if not torch.isfinite(p).all():
            bad.append(n)
            break
    if bad:
        raise RuntimeError(f"[{tag}] found NaN/Inf params, e.g. {bad[0]}")

class DummyEvaluator:
    def update(self, results):
        return results
    
    def __call__(self, results):
        return None

    def cuda(self):
        return self
    
    def cpu(self):
        return self

def set_fsdp_env():
    # Basic FSDP settings
    os.environ["ACCELERATE_USE_FSDP"] = "true"

    # Auto wrapping policy
    os.environ["FSDP_AUTO_WRAP_POLICY"] = "TRANSFORMER_BASED_WRAP"
    os.environ["FSDP_TRANSFORMER_CLS_TO_WRAP"] = "SanaMSBlock"  # Your transformer block name

    # Performance optimization settings
    os.environ["FSDP_BACKWARD_PREFETCH"] = "BACKWARD_PRE"
    os.environ["FSDP_FORWARD_PREFETCH"] = "false"

    # State dict settings
    os.environ["FSDP_STATE_DICT_TYPE"] = "FULL_STATE_DICT"
    os.environ["FSDP_SYNC_MODULE_STATES"] = "true"
    os.environ["FSDP_USE_ORIG_PARAMS"] = "true"

    # Sharding strategy
    os.environ["FSDP_SHARDING_STRATEGY"] = "FULL_SHARD"

    # Memory optimization settings (optional)
    os.environ["FSDP_CPU_RAM_EFFICIENT_LOADING"] = "false"
    os.environ["FSDP_OFFLOAD_PARAMS"] = "false"

    # Precision settings
    os.environ["FSDP_REDUCE_SCATTER_PRECISION"] = "fp32"
    os.environ["FSDP_ALL_GATHER_PRECISION"] = "fp32"
    os.environ["FSDP_OPTIMIZER_STATE_PRECISION"] = "fp32"

class Trainer:
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
        self.accelerator = accelerator
        self.data = data
        self.config = config
        self.talkshow_test_only = bool(
            config.get("talkshow_test_only", False)
        )
        self.talkshow_validation_only = bool(
            config.get("talkshow_validation_only", False)
        )
        if self.talkshow_test_only and self.talkshow_validation_only:
            raise ValueError(
                "talkshow_test_only and talkshow_validation_only are "
                "mutually exclusive"
            )
        self.talkshow_evaluation_only = (
            self.talkshow_test_only or self.talkshow_validation_only
        )
        self.num_replicas = num_replicas
        self.generator = generator if generator is not None \
            else torch.Generator(device=self.device).manual_seed(config.train.seed)
        if logger is None:
            logger = get_root_logger(osp.join(config.work_dir, "train_log.log"))
        self.logger = logger
        
        # 1.Initialize training mode
        if config.train.use_fsdp:
            set_fsdp_env()
            init_train = "FSDP"
        else:
            init_train = "DDP"
        
        logger.info(f"Initializing: {init_train} mode")
        self.load_from = True
        if config.resume_from or config.model.resume_from:
            self.load_from = False
            should_load_ema = bool(config.train.ema_update and (not config.train.use_fsdp))
            config.model.resume_from = dict(
                checkpoint=config.resume_from or config.model.resume_from,
                load_ema=should_load_ema,
                resume_optimizer=not self.talkshow_evaluation_only,
                resume_lr_scheduler=(
                    config.train.resume_lr_scheduler
                    and not self.talkshow_evaluation_only
                ),
            )

        os.environ["AUTOCAST_LINEAR_ATTN"] = "true" if config.model.autocast_linear_attn else "false"    
        ######################### first stage model #################################
        self.load_vae_feat = getattr(self.config.data, "load_vae_feat", True)
        self.first_stage_model = self.configure_first_stage_model()
        ############################ finishing first stage model ####################
        ######################### motion vae model #################################
        self.load_audio_feat = getattr(self.config.data, "load_audio_feat", True)
        self.audio_vae = self.configure_audio_vae()
        ############################ finishing motion vae model ####################
        self.load_text_feat = getattr(self.config.data, "load_text_feat", True)
        self.tokenizer, self.text_encoder, self.text_embed_dim = self.configure_text_encoder()
        # self.max_length = config.text_encoder.model_max_length
        ######################### build the diffusion scheduler #####################
        # pred_sigma = getattr(config.scheduler, "pred_sigma", False)
        # learn_sigma = getattr(config.scheduler, "learn_sigma", False) and pred_sigma
        self.diffusion = self.configure_scheduler()
        predict_info = (
            f"flow-prediction: {config.scheduler.scheduler.prediction_type}, "
            f"noise schedule: {config.scheduler.scheduler.noise_schedule}"
        )
        if "flow" in config.scheduler.scheduler.noise_schedule:
            predict_info += f", flow shift: {config.scheduler.scheduler.flow_shift}"
        if config.scheduler.weighting_scheme in ["logit_normal", "mode"]:
            predict_info += (
                f", flow weighting: {config.scheduler.weighting_scheme}, "
                f"logit-mean: {config.scheduler.logit_mean}, logit-std: {config.scheduler.logit_std}"
            )
        logger.info(predict_info)
        ######################### finishing build the diffusion scheduler ###########
        
        ######################### build the denoiser model ##########################
        self.model = build_model(
            model_config,
            config.train.grad_checkpointing,
            getattr(config.model, "fp32_attention", False),
        )
        if config.model.get("model_wrapper", None) is not None:
            self.model = get_obj_from_str(config.model.model_wrapper["target"])(
                model=self.model, 
                **config.model.model_wrapper.get("params", {}),
            )
        self.guider = self.configure_guider()

        logger.info(
            colored(
                f"{self.model.__class__.__name__}:{config.model.model}, "
                f"Model Parameters: {sum(p.numel() for p in self.model.parameters()) / 1e6:.2f}M",
                "green",
                attrs=["bold"],
            )
        )
        logger.info(f"Model: \n{self.model.__str__()}")
        if config.train.use_fsdp:
            self._fsdp_model_instance = deepcopy(self.model)
        
        self.null_embed_path = self.prepare_null_embedding()
        ############################# finishing model ###############################
        # # 4-2. model growth
        # if config.model_growth is not None:
        #     assert config.model.load_from is None
        #     self.model_growth_initializer = ModelGrowthInitializer(self.model, config.model_growth)
        #     self.model = self.model_growth_initializer.initialize(
        #         strategy=config.model_growth.init_strategy, **config.model_growth.init_params
        #     )
        
        if (not config.train.use_fsdp) and config.train.ema_update:
            self.model_ema = deepcopy(self.model).eval()
            for p in self.model_ema.parameters():
                p.requires_grad_(False)
            self.ema_rate = config.train.ema_rate
            logger.info(f"Creating EMA model for DDP mode with ema_rate: {self.ema_rate}")
        elif config.train.use_fsdp and config.train.ema_update:
            logger.warning("EMA update is not supported in FSDP mode. Setting model_ema to None.")
            self.model_ema = None
        else:
            self.model_ema = None
        
        self.ema_update(0.0)
        self.rng_state = self.load_model_checkpoint()
        if self.talkshow_evaluation_only:
            self.optimizer = None
            self.lr_scheduler = None
            evaluation_mode = (
                "test-only"
                if self.talkshow_test_only
                else "validation-only"
            )
            self.logger.info(
                f"TalkSHOW {evaluation_mode} runtime: optimizer and LR "
                "scheduler are not constructed"
            )
        else:
            self.optimizer, self.lr_scheduler = self.configure_optimizer()
        #####################################################################################
        text_encoder_info = self.config.text_encoder.text_encoder_name if not self.load_text_feat else 'not used'
        if self.talkshow_evaluation_only:
            evaluation_mode = (
                "test-only"
                if self.talkshow_test_only
                else "validation-only"
            )
            optimizer_info = (
                f"optimizer: disabled (TalkSHOW {evaluation_mode}), "
            )
        elif isinstance(self.optimizer, list):
            optimizer_info = "lr = [ " + ", ".join([f"{optimizer_i['params']['lr']:.5f}" for optimizer_i in self.config.train.optimizer]) + " ], "
        elif "muon" in self.config.train.optimizer["target"].lower():
            optimizer_info = f"muon lr: {self.config.train.optimizer['params']['muon_group']['lr']:.5f}, "
            optimizer_info += f"adam lr: {self.config.train.optimizer['params']['adam_groups']['lr']:.5f}, "
        else:
            optimizer_info = f"lr: {self.config.train.optimizer['params']['lr']:.5f}, "
        self.logger.warning(
            f"{colored(f'Basic Setting: ', 'green', attrs=['bold'])}"
            f"{optimizer_info}bs: {self.config.data.train_batch_size}, "
            f"gc: {self.config.train.grad_checkpointing}, "
            f"gc_accum_step: {self.config.train.gradient_accumulation_steps}, "
            f"guider type: {self.config.model.guider.guider_type}, "
            f"text encoder: {text_encoder_info}, "
            f"precision: {self.config.model.mixed_precision}"
        )
        #     f"qk norm: {self.config.model.model.params.block_config.params.qk_norm_type}, "
        #     f"fp32 attn: {self.config.model.fp32_attention}, "
        #     f"attn type: {self.config.model.model.params.block_config.params.attn_type}, "
        #     f"ffn type: {self.config.model.model.params.block_config.params.ffn_type}, "
        # )
        timestamp = time.strftime("%Y-%m-%d_%H:%M:%S", time.localtime())

        if self.accelerator.is_main_process:
            tracker_config = dict(vars(config))
            try:
                self.accelerator.init_trackers(config.tracker_project_name, tracker_config)
            except:
                self.accelerator.init_trackers(f"tb_{timestamp}")
        ####################################################################################
        self.complete_prepare_for_dist = False
        self.sampler = self.configure_sampler()
        self.evaluator = self.configure_evaluator()
        self.start_epoch, self.start_step = self.resume_training()

    @property
    def model_instance(self):
        if self.model_ema is not None:
            return self.model_ema
        if self.config.train.use_fsdp:
            # FSDP 模式下，validation 逻辑会手动把权重 load 到这个 shadow model 里
            return self._fsdp_model_instance
        return self.accelerator.unwrap_model(self.model)

    @property
    def device(self):
        return self.accelerator.device
    
    def load_model_checkpoint(self):
        if self.config.model.load_from is not None and self.load_from:
            epoch, missing, unexpected, rng_state = load_checkpoint(
                checkpoint=self.config.model.load_from,
                model=self.model,
                model_ema=self.model_ema if not self.config.train.use_fsdp else None,
                FSDP=self.config.train.use_fsdp,
                load_ema=self.model_ema is not None,
                # null_embed_path=self.null_embed_path,
            )
            self.logger.warning(f"Missing keys: {missing}")
            self.logger.warning(f"Unexpected keys: {unexpected}")
            if self.config.train.get("restore_rng_state", True):
                self._restore_rng_state(rng_state)
                return rng_state
        return None
    
    def resume_accelerator(self):
        if (
            self.config.train.use_fsdp
            and self.config.model.resume_from is not None
            and self.config.model.resume_from["checkpoint"] is not None
            and self.config.model.resume_from["resume_optimizer"]
            and self.config.model.resume_from["resume_lr_scheduler"]
        ):
            self.logger.info(f"FSDP resume: Loading optimizer, scheduler, scaler, random_states...")
            self.accelerator.load_state(
                os.path.join(self.config.model.resume_from["checkpoint"], "model"),
                state_dict_key=["optimizer", "scheduler", "scaler", "random_states"],
            )
    
    def prepare_for_dist(self):
        if self.complete_prepare_for_dist:
            return

        if isinstance(self.optimizer, list) or isinstance(self.lr_scheduler, list):
            # 强约束：要么都是 list 且同长度；要么你就不要走这个分支
            assert isinstance(self.optimizer, list) and isinstance(self.lr_scheduler, list), \
                "If using multiple optimizers/schedulers, both must be lists."
            assert len(self.optimizer) == len(self.lr_scheduler), \
                "The number of optimizers and lr_schedulers must match."

            prepared = self.accelerator.prepare(
                self.data._train_dataloader,
                self.model,
                *self.optimizer,
                *self.lr_scheduler,
            )
            self.data._train_dataloader, self.model = prepared[:2]
            n = len(self.optimizer)
            self.optimizer = list(prepared[2:2 + n])
            self.lr_scheduler = list(prepared[2 + n:])

        else:
            self.data._train_dataloader, self.model, self.optimizer, self.lr_scheduler = \
                self.accelerator.prepare(
                    self.data._train_dataloader,
                    self.model,
                    self.optimizer,
                    self.lr_scheduler,
                )

        if self.model_ema is not None and not self.config.train.use_fsdp:
            self.model_ema.to(self.device)

        self.complete_prepare_for_dist = True

    def training(self):
        self.accelerator.unwrap_model(self.model).train()
        if self.config.train.train_first_stage_model:
            self.first_stage_model.train()
        if self.config.train.train_audio_vae_model:
            self.audio_vae.train()
    
    def eval(self):
        self.model_instance.eval()
        if self.first_stage_model is not None:
            self.first_stage_model.eval()
        if self.audio_vae is not None:
            self.audio_vae.eval()

    def _capture_rng_state(self):
        state = {"torch": torch.get_rng_state()}
        if torch.cuda.is_available():
            # 建议保存 get_rng_state_all() 的返回
            # 有下面这段gather的代码，代表该函数必须所有卡一起调用，决不允许main process单独调用，而其他卡在wait，这样会死锁
            cuda_rng_state = torch.cuda.get_rng_state()
            world_size = get_world_size()
            cuda_rng_state_all = [None for _ in range(world_size)]
            dist.all_gather_object(cuda_rng_state_all, cuda_rng_state)
            state["torch_cuda"] = cuda_rng_state_all
        state["numpy"] = np.random.get_state()
        state["python"] = random.getstate()
        if self.generator is not None:
            state["generator"] = self.generator.get_state()
        return state

    @contextmanager
    def _rng_sandbox(self, start_from=None):
        outer = self._capture_rng_state()
        try:
            if start_from is not None:
                self._restore_rng_state(start_from)
            yield
        finally:
            # 无论中途发生什么，都尽量恢复外部 RNG，避免污染后续 phase
            self._restore_rng_state(outer)

    def _restore_rng_state(self, rng_state):
        if not rng_state:
            return

        errors = {}
        def _restore_cuda_rng_all_or_fallback(rng_state, logger):
            cuda_states = rng_state.get("torch_cuda", None)
            if cuda_states is None:
                cuda_states = rng_state.get("cuda", None)  # 兼容你现在的 snapshot 命名
            if (not torch.cuda.is_available()) or (cuda_states is None):
                return
            if not isinstance(cuda_states, (list, tuple)):
                # 认为是单 device 的 state
                dev = torch.cuda.current_device()
                torch.cuda.set_rng_state(cuda_states.cpu() if hasattr(cuda_states, "cpu") else cuda_states, device=dev)
                return
            n_visible = torch.cuda.device_count()
            n_saved = len(cuda_states)
            
            if n_saved == n_visible: # 全量恢复：长度一致才做
                cuda_states = [s.cpu() if hasattr(s, "cpu") else s for s in cuda_states]
                torch.cuda.set_rng_state_all(cuda_states)
                return
            # 不一致：降级为“当前 device”
            dev = torch.cuda.current_device()
            idx = min(dev, n_saved - 1) if n_saved > 0 else None
            if idx is None:
                raise ValueError("Saved cuda_states is empty; cannot restore.")
            s = cuda_states[idx]
            s = s.cpu() if hasattr(s, "cpu") else s
            logger.warning(
                f"CUDA RNG state count mismatch: saved={n_saved}, visible={n_visible}. "
                f"Fallback to restoring only current device={dev} from index={idx}."
            )
            torch.cuda.set_rng_state(s, device=dev)

        def _safe(name, fn):
            try:
                fn()
            except Exception as e:
                errors[name] = repr(e)
                if self.accelerator.is_main_process:
                    self.logger.warning(f"[rng_restore:{name}] failed: {e}")

        _safe("torch", lambda: torch.set_rng_state(rng_state["torch"]) if "torch" in rng_state else None)
        _safe("cuda", lambda: _restore_cuda_rng_all_or_fallback(rng_state, self.logger))
        _safe("numpy", lambda: np.random.set_state(rng_state["numpy"]) if "numpy" in rng_state else None)
        _safe("python", lambda: random.setstate(rng_state["python"]) if "python" in rng_state else None)
        _safe("generator", lambda: self.generator.set_state(rng_state["generator"])
            if (self.generator is not None and "generator" in rng_state) else None)
        if errors and self.accelerator.is_main_process:
            self.logger.warning(f"RNG restore finished with partial failures: {errors}")

    def resume_training(self):
        config = self.config
        start_epoch = 0
        start_step = 0
        if config.model.resume_from is not None and config.model.resume_from["checkpoint"] is not None:
            rng_state = None
            ckpt_path = osp.join(config.work_dir, "checkpoints")
            check_flag = osp.exists(ckpt_path) and len(os.listdir(ckpt_path)) != 0

            if config.model.resume_from["checkpoint"] == "latest":
                if check_flag:
                    config.model.resume_from["resume_optimizer"] = True
                    config.model.resume_from["resume_lr_scheduler"] = True
                    # checkpoints = os.listdir(ckpt_path)
                    # if "latest.pth" in checkpoints and osp.exists(osp.join(ckpt_path, "latest.pth")):
                    #     config.model.resume_from["checkpoint"] = osp.realpath(osp.join(ckpt_path, "latest.pth"))
                    # else:
                    #     checkpoints = [i for i in checkpoints if i.startswith("epoch_")]
                    #     checkpoints = sorted(checkpoints, key=lambda x: int(x.replace(".pth", "").split("_")[3]))
                    #     config.model.resume_from["checkpoint"] = osp.join(ckpt_path, checkpoints[-1])
                    resolved = _resolve_latest_ckpt_path(ckpt_path)
                    if resolved is None:
                        config.model.resume_from["checkpoint"] = config.model.load_from
                    else:
                        config.model.resume_from["checkpoint"] = resolved
                else:
                    config.model.resume_from["resume_optimizer"] = config.train.load_from_optimizer
                    config.model.resume_from["resume_lr_scheduler"] = config.train.load_from_lr_scheduler
                    config.model.resume_from["checkpoint"] = config.model.load_from

            if config.model.resume_from["checkpoint"] is not None:
                epoch, missing, unexpected, rng_state = load_checkpoint(
                    **config.model.resume_from,
                    model=self.model,
                    model_ema=self.model_ema if not config.train.use_fsdp else None,
                    FSDP=config.train.use_fsdp,
                    optimizer=self.optimizer,
                    lr_scheduler=self.lr_scheduler,
                    # null_embed_path=self.null_embed_path,
                )
                # Keep the historical resume behavior by default.  Explicitly
                # disabling RNG restoration is useful for repeated inference
                # from one checkpoint (for example TalkSHOW's official B=2
                # diversity protocol), while leaving all existing BEAT and
                # training runs unchanged.
                if config.train.get("restore_rng_state", True):
                    self._restore_rng_state(rng_state)
                    self.rng_state = rng_state
                else:
                    self.rng_state = None
                    self.logger.info(
                        "Checkpoint RNG restoration disabled; using "
                        "config.train.seed for this run."
                    )
                self.logger.warning(f"Missing keys: {missing}")
                self.logger.warning(f"Unexpected keys: {unexpected}")

                path = osp.basename(config.model.resume_from["checkpoint"])
            # try:
            #     start_epoch = int(path.replace(".pth", "").split("_")[1]) - 1
            #     start_step = int(path.replace(".pth", "").split("_")[3])
            # except:
            #     pass
            path = osp.basename(config.model.resume_from["checkpoint"])
            parsed = _parse_epoch_step_from_name(path)
            if parsed is not None:
                ep, st = parsed
                start_epoch = max(ep - 1, 0)
                start_step = st
        return start_epoch, start_step
    
    def configure_scheduler(self):
        set_value(self.config.scheduler.scheduler, "timestep_respacing",\
            str(self.config.scheduler.train_sampling_steps))
        diffusion = Scheduler(
            **self.config.scheduler.scheduler,
        )
        return diffusion
    
    def configure_guider(self):
        if self.config.model.guider.guider_type == "classifier-free":
            guider = VanillaOmniFlowCFGuider(
                ucg_prob=self.config.model.guider.ucg_prob,
                cfg_motion_scale=self.config.model.guider.motion_scale,
                cfg_expression_scale=self.config.model.guider.expression_scale,
                cfg_audio_scale=self.config.model.guider.audio_scale,
            )
        elif self.config.model.guider.guider_type == "identity":
            guider = IdentityGuider()
        else:
            raise ValueError(f"Invalid guider type: {self.config.model.guider.guider_type}")
        self.logger.info(
            colored(
                f"Using {self.config.model.guider.guider_type} Guider: "
                f"{guider.__class__.__name__}:{guider}, "
                f"ucg_prob: {self.config.model.guider.ucg_prob}, "
                f"motion_scale: {self.config.model.guider.motion_scale}, "
                f"expression_scale: {self.config.model.guider.expression_scale}, "
                f"audio_scale: {self.config.model.guider.audio_scale}",
                "green",
                attrs=["bold"],
            )
        )
        return guider
    
    
    def configure_evaluator(self):
        evaluator = DummyEvaluator()
        self.logger.warning(
            colored(
                "Using Default Dummy Evaluator. No calculation of evaluation metrics is performed.",
                "yellow",
                attrs=["bold"],
            )
        )
        return evaluator
    
    def configure_optimizer(self):
        # 6. build optimizer and lr scheduler
        lr_scale_ratio = 1
        if getattr(self.config.train, "auto_lr", None):
            lr_scale_ratio = auto_scale_lr(
                self.config.data.train_batch_size * get_world_size() * self.config.train.gradient_accumulation_steps,
                self.config.train.optimizer,
                **self.config.train.auto_lr,
            )
        optimizer = build_optimizer(
            self.get_model_list(), self.config.train.optimizer, self.config.train.get("optimizer_constructor", None))
        if self.config.train.lr_schedule_args and self.config.train.lr_schedule_args.get("num_warmup_steps", None):
            self.config.train.lr_schedule_args["num_warmup_steps"] = (
                self.config.train.lr_schedule_args["num_warmup_steps"] * self.num_replicas
            )
        lr_scheduler = build_lr_scheduler(self.config.train, optimizer, self.data._train_dataloader, lr_scale_ratio) 
        return optimizer, lr_scheduler
    
    def configure_first_stage_model(self):
        # typically for vae
        load_vae_feat = self.load_vae_feat
        if load_vae_feat == True:
            vae_config = DEFAULT_FIRST_STAGE_MODEL_CONFIG
            self.logger.info("Setting load_vae_feat == True, reducing to vae from {}".format(vae_config["target"]))
        else:
            vae_config = getattr(self.config, "vae", DEFAULT_FIRST_STAGE_MODEL_CONFIG)
        first_stage_model = instantiate_from_config(vae_config)
        weight_dtype = vae_config.get('weight_dtype', "float32")
        self.logger.info(
            colored(
                f"First Stage Model: "
                f"{first_stage_model.__class__.__name__}:{vae_config}, "
                f"Fisrt Stage Model Parameters: {sum(p.numel() for p in first_stage_model.parameters()) / 1e6:.2f}M, "
                f"weight type: {weight_dtype}",
                "green",
                attrs=["bold"],
            )
        )
        if self.config.train.train_first_stage_model:
            first_stage_model.train()
            self.logger.info("The first stage model will be trained along with the denoiser model")
        else:
            for named, parameters in first_stage_model.named_parameters():
                parameters.requires_grad_(False)
            first_stage_model.eval()
        dtype = get_weight_dtype(weight_dtype)
        return first_stage_model.to(dtype)
    
    def configure_audio_vae(self):
        load_audio_feat = self.load_audio_feat
        if load_audio_feat == True:
            audio_vae_config = DEFAULT_FIRST_STAGE_MODEL_CONFIG
            self.logger.info("Setting load_audio_feat == True, reducing to vae from {}".format(audio_vae_config["target"]))
        else:
            audio_vae_config = getattr(self.config.audio, "audio_vae", DEFAULT_FIRST_STAGE_MODEL_CONFIG)
        audio_vae = instantiate_from_config(audio_vae_config)
        weight_dtype = audio_vae_config.get('weight_dtype', "float32")
        self.logger.info(
            colored(
                f"Audio VAE: "
                f"{audio_vae.__class__.__name__}:{audio_vae_config}, "
                f"Audio VAE Parameters: {sum(p.numel() for p in audio_vae.parameters()) / 1e6:.2f}M, "
                f"weight type: {weight_dtype}",
                "green",
                attrs=["bold"],
            )
        )
        if self.config.train.train_audio_vae_model:
            audio_vae.train()
            self.logger.info("The audio vae model will be trained along with the denoiser model")
        else:
            for named, parameters in audio_vae.named_parameters():
                parameters.requires_grad_(False)
            audio_vae.eval()
        dtype = get_weight_dtype(weight_dtype)
        return audio_vae.to(dtype)
    
    def configure_text_encoder(self):
        tokenizer = text_encoder = None
        load_text_feat = self.load_text_feat
        if not load_text_feat:
            weight_dtype = get_weight_dtype(self.config.text_encoder.weight_dtype)
            tokenizer, text_encoder = get_tokenizer_and_text_encoder(
                name=self.config.text_encoder.text_encoder_name, device=self.device, dtype=weight_dtype,
                local_model_path=self.config.text_encoder.get("local_model_path", None),
                projection_dim=self.config.text_encoder.caption_channels # for clip
            )
            text_embed_dim = text_encoder.config.hidden_size
            self.logger.info(
                colored(
                    f"Text Encoder: "
                    f"{text_encoder.__class__.__name__}:{self.config.text_encoder}, "
                    f"Text Encoder Model Parameters: {sum(p.numel() for p in text_encoder.parameters()) / 1e6:.2f}M, "
                    f"weight type: {self.config.text_encoder.weight_dtype}",
                    "green",
                    attrs=["bold"],
                )
            )
            if self.config.text_encoder.get("chi_prompt", None) is not None:
                chi_prompt = "\n".join(self.config.text_encoder.chi_prompt)
                self.logger.info(f"Complex Human Instruct: {chi_prompt}")
        else:
            self.logger.info(
                colored(
                    "Text Encoder is not used since load_text_feat == True",
                    "green",
                    attrs=["bold"],
                )
            )
            # text_embed_dim = self.config.text_encoder.caption_channels
            text_embed_dim = None
        
        return (
            tokenizer,
            text_encoder,
            text_embed_dim
        )
    
    def configure_sampler(self):
        sampler = get_obj_from_str(self.config.scheduler.sampler["target"])(
            **self.config.scheduler.sampler.get("params", {}),
            diffusion=self.diffusion,
        )
        self.logger.info(f"Sampler: Using {sampler.__class__.__name__}:{sampler}")
        return sampler
            
    @torch.no_grad()
    def ema_update(self, ema_rate):
        if not self.config.train.ema_update or self.model_ema is None:
            return
        src_model = self.accelerator.unwrap_model(self.model)
        for p_src, p_ema in zip(src_model.parameters(), self.model_ema.parameters()):
            if p_src.requires_grad:
                p_ema.data.mul_(ema_rate).add_(p_src.data, alpha=1 - ema_rate)
        for b_src, b_ema in zip(src_model.buffers(), self.model_ema.buffers()):
            b_ema.data.copy_(b_src.data)
    
    def update_ema_rate(self, global_step: int):
        step_rules = self.config.train.get("ema_step_rules", None)
        if step_rules is None:
            return
        
        rule_list = step_rules.split(",")
        default_ema_rate = float(rule_list[-1])
        target_ema = default_ema_rate

        rules = []
        for rule_str in rule_list[:-1]:
            value_str, steps_str = rule_str.split(":")
            rules.append((int(steps_str), float(value_str)))
        rules.sort(key=lambda x: x[0])
        
        for limit_step, value in rules:
            if global_step <= limit_step:
                target_ema = value
                break
        self.ema_rate = target_ema
    
    def get_input(self, batch: dict, key: str):
        return batch.get(key, None)
    
    @torch.no_grad()
    def prepare_null_embedding(self):
        if self.load_text_feat:
            null_embed_path = self.config.train.null_embed_root
            if null_embed_path is None:
                raise ValueError("null_embed_path is None")
            self.logger.info(f"Using null embed path: {null_embed_path}")            
            return null_embed_path
        
        # 初始化null_embed_path为None，所有进程都需要这个变量
        null_embed_path = None
        
        if self.accelerator.is_main_process:
            null_embed_root = self.config.train.null_embed_root
            if null_embed_root is None:
                null_embed_root = osp.join(self.config.work_dir, "null_embed")
            os.makedirs(null_embed_root, exist_ok=True)
            null_embed_path = osp.join(
                null_embed_root,  
                f"null_embed_diffusers_{self.config.text_encoder.text_encoder_name}"
                f"_{self.config.text_encoder.model_max_length}token_{self.config.text_encoder.caption_channels}.pth",
            )
            null_tokens = self.tokenizer(
                "", max_length=self.config.text_encoder.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
            ).to(self.device)
            if "T5" in self.config.text_encoder.text_encoder_name:
                null_token_emb = self.text_encoder(null_tokens.input_ids, attention_mask=null_tokens.attention_mask)[0]
                attention_mask = null_tokens.attention_mask.squeeze(0)
            elif "gemma" in self.config.text_encoder.text_encoder_name or "Qwen" in self.config.text_encoder.text_encoder_name:
                null_token_emb = self.text_encoder(null_tokens.input_ids, attention_mask=null_tokens.attention_mask)[0]
                attention_mask = null_tokens.attention_mask.squeeze(0)
            elif "clip" in self.config.text_encoder.text_encoder_name:
                null_token_emb = self.text_encoder(
                    null_tokens.input_ids, attention_mask=null_tokens.attention_mask, output_hidden_states=True)
                if self.config.text_encoder.use_last_hidden_states:
                    null_token_emb = null_token_emb.last_hidden_state
                    attention_mask = null_tokens.attention_mask.squeeze(0)
                else:
                    null_token_emb = null_token_emb[0][:, None]
                    attention_mask = None
            else:
                raise ValueError(f"{self.config.text_encoder.text_encoder_name} is not supported!!")
            with open(null_embed_path, "wb") as f:
                torch.save({
                        "uncond_prompt_embeds": null_token_emb.squeeze(0), 
                        "uncond_prompt_embeds_mask": attention_mask
                    },
                    f
                )
                os.fsync(f.fileno()) # 强行在这里等一下，避免出现异步写的问题
        
        self.accelerator.wait_for_everyone() # 等主进程把文件写好
        # 使用 accelerate 的工具来广播 Python 对象
        if self.accelerator.is_main_process:
            # 主进程将路径放入一个 list
            path_list = [null_embed_path]
        else:
            # 其他进程创建一个包含占位符的相同大小的 list
            path_list = [None]

        # broadcast_object_list 会处理所有底层的同步和数据传输
        broadcast_object_list(path_list, from_process=0) # 进行一次广播
        # 现在所有进程的 path_list 都变成了 [null_embed_path]
        null_embed_path = path_list[0]
        self.accelerator.wait_for_everyone()
        return null_embed_path
        
    def prepare_data(self, batch, step, null_embed = None, lm_time_all=None, am_time_all=None, is_predict=False):
        raise NotImplementedError("Abstract method.")
    
    def training_step(self, batch, step, vae_time_all=None, global_step=None):
        raise NotImplementedError("Abstract method.")
    
    def extra_training_log(self, loss_term):
        pass
    
    def debug_gradient(self, global_step, loss_term):
        if self.config.train.debug_gradient_interval > 0 and (global_step - 1) % self.config.train.debug_gradient_interval == 0:
            norms = {}
            loss_scalars = {k: v.mean() for k, v in loss_term.items()} 
            for k, ls in loss_scalars.items():
                norms["grad_norm_" + k] = grad_norm_of_loss(ls, self.model, retain_graph=True)
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
 
    def save_checkpoints(self, step, epoch, global_step, ckpt_path=None, best_metric=None, best_metric_mode=None, rng_state=None):
        ckpt_saved_path = ckpt_path if ckpt_path is not None else osp.join(self.config.work_dir, "checkpoints")
        if self.config.train.use_fsdp:
            # FSDP mode
            os.umask(0o000)
            ckpt_saved_path = save_checkpoint(
                work_dir=ckpt_saved_path,
                epoch=epoch,
                model=self.model,
                accelerator=self.accelerator,
                optimizer=self.optimizer,
                lr_scheduler=self.lr_scheduler,
                step=global_step,
                add_symlink=True,
                keep_last=self.config.train.keep_last,
                keep_last_n=self.config.train.keep_last_n,
                best_metric=best_metric,
                best_metric_mode=best_metric_mode,
                rng_state=rng_state,
            )
        else:
            # DDP mode
            if self.accelerator.is_main_process:
                os.umask(0o000)
                ckpt_saved_path = save_checkpoint(
                    work_dir=ckpt_saved_path,
                    epoch=epoch,
                    model=self.accelerator.unwrap_model(self.model),
                    model_ema=self.accelerator.unwrap_model(self.model_ema) if self.model_ema is not None else None,
                    optimizer=self.optimizer,
                    lr_scheduler=self.lr_scheduler,
                    step=global_step,
                    generator=self.generator,
                    add_symlink=True,
                    keep_last=self.config.train.keep_last,
                    keep_last_n=self.config.train.keep_last_n,
                    best_metric=best_metric,
                    best_metric_mode=best_metric_mode,
                    rng_state=rng_state,
                )
    
        if self.accelerator.is_main_process:
            if self.config.train.online_metric and global_step % self.config.train.eval_metric_step == 0 and step > 1:
                online_metric_monitor_dir = osp.join(self.config.work_dir, self.config.train.online_metric_dir)
                os.makedirs(online_metric_monitor_dir, exist_ok=True)
                with open(f"{online_metric_monitor_dir}/{ckpt_saved_path.split('/')[-1]}.txt", "w") as f:
                    f.write(osp.join(self.config.work_dir, "config.py") + "\n")
                    f.write(ckpt_saved_path)
    
    def get_model_list(self):
        model_list = [self.model]
        if self.config.train.train_audio_vae_model:
            model_list.append(self.audio_vae)
        if self.config.train.train_first_stage_model:
            model_list.append(self.first_stage_model)
        return torch.nn.ModuleList(model_list)
    
    def train(self):
        if self.data._train_dataloader is None:
            self.logger.error("No train dataloader provided. Will not train.")
            return 
        if self.config.train.get("keep_best_ckpt", False):
            best_ckpt_value = float("-inf") if self.config.train.get("best_metric_mode", "max") == "max" else float("inf")
            best_ckpt_path = osp.join(self.config.work_dir, "checkpoints", "best")
            os.makedirs(best_ckpt_path, exist_ok=True)
        
        if getattr(self.config.train, "debug_nan", False):
            DebugUnderflowOverflow(self.model, max_frames_to_save=100)
            self.logger.info("NaN debugger registered. Start to detect overflow during training.")
        log_buffer = LogBuffer()
        start_epoch, start_step = self.start_epoch, self.start_step
        sampler = (
            self.data._train_dataloader.batch_sampler.sampler
            if (self.num_replicas > 1)
            else self.data._train_dataloader.sampler
        )
        if not hasattr(sampler, "set_start"):
            start_step = start_epoch * len(self.data._train_dataloader)
            if self.accelerator.is_main_process:
                self.logger.warning(
                    f"Sampler provided does not have set_start method, " 
                    f"will not skip starting steps and resume training from the very beginning of epoch #{start_epoch}."
                )
                self.logger.warning(f"Will reset the starting step to #{start_step}.")
        else:
            if start_step != start_epoch * len(self.data._train_dataloader):
                # 这种情况就是多半改变了batch size或者改变了训练的卡数，理论上最好不要出现这种问题
                # 如果只在每个epoch结束的时候保存权重，就应该不会存在这种问题
                # 但是还是要解决这种情况，尝试猜一次正确的starting step，猜错了就reset到epoch的开始
                loader_len = len(self.data._train_dataloader)
                world_bs = self.config.data.train_batch_size * get_world_size()
                assume_world_bs = world_bs * start_epoch * loader_len // start_step + 1
                # assume_eta_step = assume_world_bs * start_step - world_bs * start_epoch * loader_len
                # final_start_step = assume_eta_step // world_bs + start_epoch * loader_len
                assume_loader_len = len(self.data._train_dataset) // assume_world_bs +\
                    (len(self.data._train_dataset) % assume_world_bs > 0)
                # 上面这样算其实是有个固定假设的，认为上一次训练的时候没有drop last，但是这一点我们无从得知，只能这样假设
                # 尽量都不要drop last吧
                assume_eta_step = start_step % assume_loader_len
                final_start_step = assume_eta_step * assume_world_bs // world_bs +  start_epoch * loader_len
                
                if final_start_step - start_epoch * loader_len < loader_len:                 
                    if self.accelerator.is_main_process:
                        self.logger.warning(
                            colored(
                                f"Starting step: #{start_step} is not equal to starting epoch * dataloader length: "
                                f"#{start_epoch} * {len(self.data._train_dataloader)}."
                                f"Try to assume the starting step as #{final_start_step} in this training setting. "
                                f"This may not be the correct starting step. "
                                f"You'd better avoid this problem "
                                f"by keeping the same batch size and the same number of training cards during training.",
                                "red",
                                attrs=["bold"],
                            )
                        )
                    start_step = final_start_step
                else:
                    start_step = start_epoch * len(self.data._train_dataloader)
                    self.logger.warning(
                        colored(
                            f"Starting step: #{start_step} is not equal to starting epoch * dataloader length: "
                            f"#{start_epoch} * {len(self.data._train_dataloader)}."
                            f"Will reset the starting step to #{start_step} "
                            f"to resume training the model from the very beginning of epoch #{start_epoch}."
                            f"This may not be the correct starting step. "
                            f"You'd better avoid this problem "
                            f"by keeping the same batch size and the same number of training cards during training.",
                            "red",
                            attrs=["bold"],
                        )
                    )
                
            if self.accelerator.is_main_process:
                self.logger.info(f"Starting step: #{start_step}. Starting epoch: #{start_epoch}.")
        self.prepare_for_dist()
        self.resume_accelerator()
        # set_random_seed((start_step + 1) // self.config.train.save_model_steps + int(os.environ["LOCAL_RANK"]))
        # self.logger.info(f'Set seed: {(start_step + 1) // self.config.train.save_model_steps + int(os.environ["LOCAL_RANK"])}')
        
        config = self.config
        global_step = start_step + 1
        train_dataloader_len = len(self.data._train_dataloader)
        total_steps = train_dataloader_len * config.train.num_epochs
        skip_step = max(self.config.train.skip_step, global_step) % train_dataloader_len
        skip_step = skip_step if skip_step < (train_dataloader_len - 20) else 0
        loss_nan_timer = 0
        self.model_instance.to(self.device)
        if self.null_embed_path is not None:    
            null_embed = torch.load(self.null_embed_path, map_location="cpu")
            null_embed = null_embed["uncond_prompt_embeds"].to(self.device)
        else:
            null_embed = None
        
        # for sanity check
        if config.train.sanity_check_steps is not None and config.train.sanity_check_steps > 0:
            assert self.config.train.eval, "sanity check can only run in eval mode"
            if self.accelerator.is_main_process:
                self.logger.info("Running sanity check... ")
            with torch.no_grad():
                all_results = {}
                for i, batch in enumerate(self.data._validation_dataloader):
                    data_dict = self.prepare_data(batch, i, null_embed)
                    loss_term, results = self.validation_step(data_dict, null_embed, i, visualize=False)
                    loss = sum([val.mean() for _, val in loss_term.items()])
                    if isinstance(loss, torch.Tensor) and torch.any(torch.isnan(loss)):
                        self.logger.error("Loss is NaN, please check the training process")
                        raise ValueError("Loss is NaN, please check the training process")
                    all_results = self.postprocess_results(all_results, results)
                    
                    if i >= self.config.train.sanity_check_steps - 1:
                        metric_term = self.evaluator(all_results)
                        for key, val in metric_term.items():
                            if np.any(np.isnan(val)):
                                self.logger.error(f"Metric {key} is NaN, please check the training process")
                                raise ValueError(f"Metric {key} is NaN, please check the training process")
                        self.logger.info("Sanity check finished, no NaN found.")
                        break
        ####################################################################################            
        
        training_start_time = time.time()
        for epoch in range(start_epoch + 1, config.train.num_epochs + 1):
            time_start, last_tic = time.time(), time.time()
            sampler.set_epoch(epoch)
            if hasattr(sampler, "set_start"):
                sampler.set_start(max((skip_step - 1) * config.data.train_batch_size, 0))
                if skip_step > 1 and self.accelerator.is_main_process:
                    self.logger.info(f"Skipped Steps: {skip_step}")
            else:
                if self.accelerator.is_main_process:
                    self.logger.warning(f"Sampler provided does not have set_start method, will not skip starting steps." +\
                        "Skipping steps will be disabled.")
            skip_step = 1
            data_time_start = time.time()
            data_time_all = 0
            lm_time_all = 0 # language model time
            vae_time_all = 0 # first stage model time
            model_time_all = 0 #
            am_time_all = 0 # audio encoder time if needed
            epoch_time_start = time.time()
            for step, batch in enumerate(self.data._train_dataloader):
                # break
                # self.accelerator.wait_for_everyone()
                data_time_all += time.time() - data_time_start
                model_time_start = time.time()
                model_list = self.get_model_list()
                with self.accelerator.accumulate(model_list):
                    if self.accelerator.sync_gradients:
                        if isinstance(self.optimizer, list):
                            for opt in self.optimizer: opt.zero_grad(set_to_none=True)
                        else:
                            self.optimizer.zero_grad(set_to_none=True)
                    data_dict = self.prepare_data(batch, step, null_embed, lm_time_all, am_time_all) 
                    # for vae and language model and audio encoder (if needed)
                    lm_time_all = data_dict.get("lm_time_all", 0)
                    am_time_all = data_dict.get("am_time_all", 0)
                    loss_term, vae_time_all = self.training_step(data_dict, step, vae_time_all, global_step)
                    # loss = sum([val.mean() for _, val in loss_term.items()])
                    self.debug_gradient(global_step, loss_term)
                    loss = loss_term["loss"].mean()
                    loss = loss / self.config.train.gradient_accumulation_steps
                    self.accelerator.backward(loss)
                    if self.accelerator.sync_gradients:
                        # Optional: clip per-gradient values first to suppress outlier spikes.
                        gradient_clip_value = config.train.get("gradient_clip_value", None)
                        if gradient_clip_value is not None:
                            if hasattr(self.accelerator, "clip_grad_value_"):
                                self.accelerator.clip_grad_value_(self.model.parameters(), gradient_clip_value)
                            else:
                                clip_grad_value_(self.model.parameters(), gradient_clip_value)

                        if config.train.gradient_clip is not None:
                            grad_norm = self.accelerator.clip_grad_norm_(self.model.parameters(), config.train.gradient_clip)
                        else:
                            grad_norm = None
                        if isinstance(self.optimizer, list):
                            for optimizer_i in self.optimizer:
                                optimizer_i.step()
                        else:
                            self.optimizer.step()
                        if isinstance(self.lr_scheduler, list):
                            for lr_scheduler_i in self.lr_scheduler:
                                lr_scheduler_i.step()
                        else:
                            self.lr_scheduler.step()
                    else:
                        grad_norm = None    
                    
                    if self.accelerator.sync_gradients and \
                        not config.train.use_fsdp and config.train.ema_update and self.model_ema is not None:
                        self.update_ema_rate(global_step)
                        self.ema_update(self.ema_rate)
                    self.accelerator.wait_for_everyone()
                    model_time_all += time.time() - model_time_start
                if isinstance(loss, torch.Tensor) and torch.any(torch.isnan(loss)):
                    loss_nan_timer += 1
                if isinstance(self.lr_scheduler, list):
                    lr = []
                    for lr_scheduler_i in self.lr_scheduler:
                        lr.extend(lr_scheduler_i.get_last_lr())
                else:
                    lr = self.lr_scheduler.get_last_lr()
                logs = {}
                for key, val in loss_term.items():
                    val = val.mean()
                    # logs[f"train/{key}"] = self.accelerator.gather(val).mean().item()
                    logs[f"train/{key}"] = val.mean().item()
                # logs["train_total_loss"] = self.accelerator.gather(loss).mean().item()
                if grad_norm is not None:
                    logs["grad_norm"] = grad_norm.mean().item()
                    # logs["grad_norm"] = self.accelerator.gather(grad_norm).mean().item()
                log_buffer.update(logs)
                
                if (step + 1) % config.train.log_interval == 0 or (step + 1) == 1:
                    self.accelerator.wait_for_everyone()
                    t = (time.time() - last_tic) / config.train.log_interval
                    t_d = data_time_all / config.train.log_interval
                    t_m = model_time_all / config.train.log_interval
                    t_lm = lm_time_all / config.train.log_interval
                    t_am = am_time_all / config.train.log_interval
                    t_vae = vae_time_all / config.train.log_interval
                    avg_time = (time.time() - time_start) / (step + 1)
                    step_start = sampler.step_start if hasattr(sampler, "step_start") else 0
                    eta = str(datetime.timedelta(seconds=int(avg_time * (total_steps - global_step - 1))))
                    eta_epoch = str(
                        datetime.timedelta(
                            seconds=int(
                                avg_time
                                * (train_dataloader_len - step_start // config.data.train_batch_size - step - 1)
                            )
                        )
                    )
                    log_buffer.average()
                    
                    # current_step = (
                    #     global_step - step_start // config.data.train_batch_size
                    # ) % train_dataloader_len
                    current_step = global_step % train_dataloader_len
                    current_step = train_dataloader_len if current_step == 0 else current_step
                    epoch_time_cost = time.time() - epoch_time_start
                    epoch_time_cost = str(
                        datetime.timedelta(seconds=int(epoch_time_cost))
                    )
                    # 重新设计日志格式，提高可读性
                    info = (
                        colored(
                            f"[Epoch {epoch:6d}/{config.train.num_epochs:6d}" 
                            f" | Step {global_step:6d}/{total_steps:6d}" 
                            f" | Local {current_step:6d}/{train_dataloader_len:6d}] ",
                            color="green",
                            attrs=["bold"],
                        ) + \
                        colored(
                            f"ETA: {eta} | Epoch-ETA: {eta_epoch} | Epoch-Time: {epoch_time_cost} \n",
                            color="blue",
                        ) + \
                        colored(
                            f"[Time] ",
                            color="yellow",
                            attrs=["bold"],
                        ) + \
                        f"Total = {t:6.3f}s | Model = {t_m:6.3f}s | Data = {t_d:6.3f}s | "
                        f"LM = {t_lm:6.3f}s | VAE = {t_vae:6.3f}s | AM = {t_am:6.3f}s | "
                    )
                    if isinstance(lr, list):
                        info += "LR = [" + ", ".join([f"{lr_i:.2e}" for lr_i in lr]) + "] "
                    else:
                        info += f"LR = {lr:.2e} "
                    if not config.train.use_fsdp and self.config.train.ema_update and self.model_ema is not None:
                        info += f"| EMA = {self.ema_rate:.2e} "
                    info += "\n"
        
                    last_tic = time.time()
                    data_time_all = 0
                    model_time_all = 0
                    lm_time_all = 0
                    vae_time_all = 0
                    am_time_all = 0
                    log_buffer_output = log_buffer.output
                    for key, val in log_buffer_output.items():
                        log_buffer_output[key] = self.accelerator.gather(
                            torch.tensor(val, device=self.device, dtype=torch.float32)).mean().item()
                    info += colored(
                        f"[Loss] ",
                        color="cyan",
                        attrs=["bold"],
                    )
                    info += " | ".join([f"{k} = {v:.4f}" for k, v in log_buffer_output.items()])
                    if self.accelerator.is_main_process:
                        self.logger.info(info)
                        self.accelerator.log(log_buffer_output)
                    self.extra_training_log(loss_term=loss_term)
                    log_buffer.clear()

                logs["lr"] = lr
                if not config.train.use_fsdp and self.config.train.ema_update and self.model_ema is not None:
                    logs["ema_rate"] = self.ema_rate

                if loss_nan_timer > 20:
                    raise ValueError("Loss is NaN too much times. Break here.")
                
                if (
                    global_step % config.train.save_model_steps == 0
                    or (
                        config.train.early_stop_hours is not None
                        and (time.time() - training_start_time) / 3600 > config.train.early_stop_hours
                    )
                    or (
                        config.train.early_stop_steps is not None
                        and global_step > config.train.early_stop_steps
                    )
                ):
                    # torch.cuda.synchronize()
                    self.accelerator.wait_for_everyone()
                    self.save_checkpoints(step, epoch, global_step)
                
                    if (
                        config.train.early_stop_hours is not None
                        and (time.time() - training_start_time) / 3600 > config.train.early_stop_hours
                    ):
                        # for early stopping
                        all_training_time = time.time() - training_start_time
                        if self.accelerator.is_main_process:
                            self.logger.info(f"Stopping training at epoch {epoch}, step {global_step} due to time limit.")
                            self.logger.info(f"Training Finished, Total Time Cost : {all_training_time:.3f}.")
                        return
                    
                    if (
                        config.train.early_stop_steps is not None
                        and global_step > config.train.early_stop_steps
                    ):
                        all_training_time = time.time() - training_start_time
                        if self.accelerator.is_main_process:
                            self.logger.info(f"Stopping training at epoch {epoch}, step {global_step} due to step limit.")
                            self.logger.info(f"Training Finished, Total Time Cost : {all_training_time:.3f}.")
                        return

                global_step += 1
                data_time_start = time.time()
                
            if config.train.eval and (epoch % config.train.check_eval_n_epochs == 0):
                if config.train.use_fsdp:
                    merged_state_dict = self.accelerator.get_state_dict(self.model)
                    self.model_instance.load_state_dict(merged_state_dict)
                self.accelerator.wait_for_everyone()                        
                # 必须抢在validation前拿下rng_state做记录，否则validation过程中会污染rng_state
                rng_state = self._capture_rng_state()
                validation_output = self.validation(global_step, epoch)
                # def list_buffers(m):
                #     return [(n, tuple(b.shape), b.numel(), str(b.dtype), str(b.device))
                #         for n, b in m.named_buffers()]

                # bufs = list_buffers(self.accelerator.unwrap_model(self.model)) 
                # self.logger.info(bufs)
                if self.config.train.get("keep_best_ckpt", False) and \
                    (not self.config.train.use_fsdp) \
                        and self.accelerator.is_main_process: # 暂不支持FSDP模式下的最佳模型保存    

                    monitor_metric = self.config.train.get("monitor_metric", None)
                    # rolling back to validation_output.keys()[0] if monitor_metric is None else monitor_metric
                    monitor_metric = next(iter(validation_output)) if monitor_metric is None else monitor_metric
                    best_metric_mode = self.config.train.get("best_metric_mode", "max")
                    if best_metric_mode == "max":
                        if validation_output[monitor_metric] > best_ckpt_value:
                            best_ckpt_value = validation_output[monitor_metric]
                            self.save_checkpoints(
                                step, 
                                epoch, 
                                global_step, 
                                ckpt_path=best_ckpt_path, 
                                best_metric=best_ckpt_value, 
                                best_metric_mode=best_metric_mode,
                                rng_state=rng_state,
                            )
                    else:
                        if validation_output[monitor_metric] < best_ckpt_value:
                            best_ckpt_value = validation_output[monitor_metric]
                            self.save_checkpoints(
                                step, 
                                epoch, 
                                global_step, 
                                ckpt_path=best_ckpt_path, 
                                best_metric=best_ckpt_value, 
                                best_metric_mode=best_metric_mode,
                                rng_state=rng_state,
                            )
                    
                
            if epoch % config.train.save_model_epochs == 0 or epoch == config.train.num_epochs:
                self.accelerator.wait_for_everyone()
                # torch.cuda.synchronize()
                self.save_checkpoints(step, epoch, global_step)
        all_training_time = time.time() - training_start_time
        if self.accelerator.is_main_process:
            self.logger.info(f"Training Finished, Total Time Cost : {all_training_time:.3f}.")

    def extra_validation_log(self, loss_term, results):
        pass
    
    def validation_step(self, batch, null_embed, step, global_step = None, epoch = None, visualize = True):
        raise NotImplementedError("Abstract method.")
    
    
    def postprocess_results(self, all_results, results):
        for key, value in results.items():
            if key in all_results:
                if isinstance(value, torch.Tensor):
                    for b in range(value.shape[0]):
                        all_results[key].append(value[b].detach().cpu().numpy())
                elif isinstance(value, list):
                    all_results[key].extend(value)
                elif isinstance(value, np.ndarray):
                    for b in range(value.shape[0]):
                        all_results[key].append(value[b])
            else:
                all_results[key] = []
                if isinstance(value, torch.Tensor):
                    for b in range(value.shape[0]):
                        all_results[key].append(value[b].detach().cpu().numpy())
                elif isinstance(value, list):
                    all_results[key].extend(value)
                elif isinstance(value, np.ndarray):
                    for b in range(value.shape[0]):
                        all_results[key].append(value[b])
        return all_results

    @staticmethod
    def _evaluation_batch_has_valid_samples(batch):
        """Return false only for an explicitly marked all-padding eval batch."""

        padding = batch.get("is_padding_sample")
        if padding is None:
            return True
        if not isinstance(padding, torch.Tensor):
            raise TypeError("is_padding_sample must be a torch.Tensor")
        return bool(torch.any(~padding.to(dtype=torch.bool)).item())
    
    @torch.no_grad()
    def validation(self, global_step = None, epoch = None):
        if self.data._validation_dataloader is None:
            if self.accelerator.is_main_process:
                self.logger.error("No validation dataloader provided. Will not eval.")
            return 
        
        self.prepare_for_dist()
        self.evaluator.cuda()
        self.eval()
        # snapshot("val:enter", self.model, self.logger)
        # check_model_finite(self.model, "model")
        torch.cuda.empty_cache()
        if self.accelerator.is_main_process:
            self.logger.info("Running validation... ")
        # log_buffer = LogBuffer()
        all_results = {}
        if self.null_embed_path is not None:
            null_embed = torch.load(self.null_embed_path, map_location="cpu")
            null_embed = null_embed["uncond_prompt_embeds"].to(self.device)
        else:
            null_embed = None
        
        loss_log_buffer = LogBuffer()
        time_start, validation_start_time = time.time(), time.time()
        validation_dataloader_len = len(self.data._validation_dataloader)
        total_steps = validation_dataloader_len
        log_interval = self.config.train.get("validation_log_interval", self.config.train.log_interval)
        for step, batch in enumerate(self.data._validation_dataloader):
            data_dict = self.prepare_data(batch, step, null_embed, is_predict=True)
            loss_term, results = self.validation_step(data_dict, null_embed, step, global_step, epoch)
            # logs = {}
            # for key, val in metric_term:
            #     val = val.mean()
            #     logs = {f"val_{key}": self.accelerator.gather(val).mean().item()}    
            # log_buffer.update(logs)
            if self._evaluation_batch_has_valid_samples(data_dict):
                results = self.evaluator.update(results)
            else:
                results = {}
            all_results = self.postprocess_results(all_results, results)
            
            # loss = sum([val.mean() for _, val in loss_term.items()])
            # loss = self.accelerator.gather(loss_term["loss"]).mean().item()
            # loss_log_buffer.update({"val_loss": loss})
            loss_logs = {}
            for key, val in loss_term.items():
                val = val.mean()
                loss_logs[f"val/{key}"] = self.accelerator.gather(val).mean().item()    
            loss_log_buffer.update(loss_logs)
            
            if (step + 1) % log_interval== 0 or (step + 1) == 1:
                self.accelerator.wait_for_everyone()
                avg_time = (time.time() - time_start) / (step + 1)
                eta = str(datetime.timedelta(seconds=int(avg_time * (total_steps - step - 1))))
                # 重新设计验证日志格式，仿照训练日志的样式
                info = (
                    colored(
                        f"[Validation | Step {step:6d}/{validation_dataloader_len:6d}] ",
                        color="magenta",
                        attrs=["bold"],
                    ) + \
                    colored(
                        f"ETA: {eta} \n",
                        color="blue",
                    ) + \
                    colored(
                        f"[Loss] ",
                        color="cyan",
                    )
                )
                loss_log_buffer.average()
                info += " | ".join([f"{k}:{v:.4f}" for k, v in loss_log_buffer.output.items()])
                if self.accelerator.is_main_process:
                    self.logger.info(info)
                    self.accelerator.log(loss_log_buffer.output)
                loss_log_buffer.clear()
                self.extra_validation_log(loss_term=loss_term, results=results)
            # if self.accelerator.is_main_process:
            #     self.accelerator.log(logs)
        
        # log_buffer.average()
        # validation_output = log_buffer.output
        # info = ", ".join([f"{k}:{v:.4f}" for k, v in validation_output.items()])
        # log_buffer.clear()
        validation_output = self.evaluator(all_results)
        del all_results
        info = " | ".join([f"{k} = {v:.12f}" for k, v in validation_output.items()]) if validation_output is not None\
            else "No validation result."
        total_validation_time = time.time() - validation_start_time
        if self.accelerator.is_main_process:
            self.logger.info(
                colored(
                    "Validation Results : " + info,
                    "red",
                    attrs=["bold"],
                )
            )
            if validation_output is not None:
                self.accelerator.log(validation_output)
            self.logger.info(f"Validation Finished, Total Time Cost : {total_validation_time:.3f}")
        # check_model_finite(self.model, "model")
        # snapshot("val:exit", self.model, self.logger)
        self.training()
        self.evaluator.cpu()
        flush()
        return validation_output
    
    def test_step(self, batch, null_embed, step):
        raise NotImplementedError("Abstract method.")
    
    @torch.no_grad()
    def test(self):
        if self.data._test_dataloader is None:
            if self.accelerator.is_main_process:
                self.logger.error("No test dataloader provided. Will not test.")
            return 
        with self._rng_sandbox(start_from=self.rng_state):
            self.prepare_for_dist()
            self.evaluator.cuda()
            self.eval()
            # snapshot("test:enter", self.model, self.logger)
            flush()
            if self.accelerator.is_main_process:
                self.logger.info("Running Test... ")
            # log_buffer = LogBuffer()
            time_start, test_start_time = time.time(), time.time()
            test_dataloader_len = len(self.data._test_dataloader)
            total_steps = test_dataloader_len
            log_interval = self.config.train.get("test_log_interval", self.config.train.log_interval)
            all_results = {}

            if self.null_embed_path is not None:
                null_embed = torch.load(self.null_embed_path, map_location="cpu")
                null_embed = null_embed["uncond_prompt_embeds"].to(self.device)
            else:
                null_embed = None
            
            for step, batch in enumerate(self.data._test_dataloader):
                data_dict = self.prepare_data(batch, step, null_embed, is_predict=True)
                results = self.test_step(data_dict, null_embed, step)
                if self._evaluation_batch_has_valid_samples(data_dict):
                    results = self.evaluator.update(results)
                else:
                    results = {}
                all_results = self.postprocess_results(all_results, results)
                # logs = {}
                # for key, val in metric_term:
                #     val = val.mean()
                #     logs = {f"test_{key}": self.accelerator.gather(val).mean().item()}    
                # log_buffer.update(logs)
                
                if (step + 1) % log_interval== 0 or (step + 1) == 1:
                    self.accelerator.wait_for_everyone()
                    avg_time = (time.time() - time_start) / (step + 1)
                    eta = str(datetime.timedelta(seconds=int(avg_time * (total_steps - step - 1))))
                    # 重新设计测试日志格式，仿照训练日志的样式
                    info = (
                        colored(
                            f"[Test | Step {step:6d}/{test_dataloader_len:6d}] ",
                            color="red",
                            attrs=["bold"],
                        ) + \
                        colored(
                            f"ETA: {eta}",
                            color="blue",
                        )
                    )
                    if self.accelerator.is_main_process:
                        self.logger.info(info)
            # log_buffer.average()
            # test_output = log_buffer.output
            # info = ", ".join([f"{k}:{v:.4f}" for k, v in test_output.items()])
            # log_buffer.clear()
            test_output = self.evaluator(all_results)
            del all_results
            info = " | ".join([f"{k}:{v:.12f}" for k, v in test_output.items()]) if test_output is not None \
                else "No test result."
            total_test_time = time.time() - test_start_time
            if self.accelerator.is_main_process:
                self.logger.info(
                    colored(
                        "Test Results : " + info,
                        "red",
                        attrs=["bold"],
                    )
                )
                self.accelerator.log(test_output)
                self.logger.info(f"Test Finished, Total Time Cost : {total_test_time:.3f}")
            # snapshot("test:exit", self.model, self.logger)
            self.training()
            self.evaluator.cpu()
    
    @torch.no_grad()
    def predict(self):
        if self.data._predict_dataloader is None:
            if self.accelerator.is_main_process:
                self.logger.error("No predict dataloader provided. Will not predict.")
            return 
        
        with self._rng_sandbox(start_from=self.rng_state):
            self.prepare_for_dist()
            self.evaluator.cuda()
            self.eval()
            flush()
            if self.accelerator.is_main_process:
                self.logger.info("Running Predict... ")
            time_start, predict_start_time = time.time(), time.time()
            predict_dataloader_len = len(self.data._predict_dataloader)
            total_steps = predict_dataloader_len
            log_interval = self.config.train.get("predict_log_interval", self.config.train.log_interval)
            all_results = {}
            
            if self.null_embed_path is not None:
                null_embed = torch.load(self.null_embed_path, map_location="cpu")
                null_embed = null_embed["uncond_prompt_embeds"].to(self.device)
            else:
                null_embed = None
            
            for step, batch in enumerate(self.data._predict_dataloader):
                data_dict = self.prepare_data(batch, step, null_embed, is_predict=True)
                results = self.predict_step(data_dict, null_embed, step)
                results = self.evaluator.update(results)
                all_results = self.postprocess_results(all_results, results)
                if (step + 1) % log_interval== 0 or (step + 1) == 1:
                    self.accelerator.wait_for_everyone()
                    avg_time = (time.time() - time_start) / (step + 1)
                    eta = str(datetime.timedelta(seconds=int(avg_time * (total_steps - step - 1))))
                    info = (
                        colored(
                            f"[Predict | Step {step:6d}/{predict_dataloader_len:6d}] ",
                            color="green",
                            attrs=["bold"],
                        ) + \
                        colored(
                            f"ETA: {eta}",
                            color="blue",
                        )
                    )
                    if self.accelerator.is_main_process:
                        self.logger.info(info)
            predict_output = self.evaluator(all_results)
            del all_results
            info = " | ".join([f"{k}:{v:.12f}" for k, v in predict_output.items()]) if predict_output is not None \
                else "No predict result."
            total_predict_time = time.time() - predict_start_time
            if self.accelerator.is_main_process:
                self.logger.info(
                    colored(
                        "Predict Results : " + info,
                        "red",
                        attrs=["bold"],
                    )
                )
                self.accelerator.log(predict_output)
                self.logger.info(f"Predict Finished, Total Time Cost : {total_predict_time:.3f}")
            self.training()
            self.evaluator.cpu()
