import warnings
warnings.filterwarnings("ignore")  # ignore warning
import os
import hydra
import datetime
import os.path as osp
from accelerate import Accelerator, InitProcessGroupKwargs
from diffusion.data.base import DataModule
from diffusion.utils.config import set_config_value
from diffusion.utils.logger import get_root_logger
from omegaconf import OmegaConf
from diffusion.utils.dist_utils import get_world_size
from diffusion.utils.misc import init_random_seed, set_random_seed
import torch
from diffusion.data.datasets.beat2.build_vocab import Vocab
import torch.distributed as dist
import re
import hashlib, tempfile
from pathlib import Path
from hydra.experimental.callback import Callback
from omegaconf import DictConfig, OmegaConf


os.environ["TOKENIZERS_PARALLELISM"] = "true"

DEFAULT_SAMPLER_CONFIG = DictConfig({
    "target": "torch.utils.data.distributed.DistributedSampler",
    "params": {
        "drop_last": False,
        "shuffle": False,
    }
})


def _publish_seed_sweep_inference_completion() -> None:
    """Publish test completion before distributed teardown can block."""

    marker_path = os.environ.get("SEED_SWEEP_INFERENCE_COMPLETION_MARKER")
    if not marker_path:
        return
    run_id = os.environ.get("SEED_SWEEP_INFERENCE_COMPLETION_RUN_ID", "")
    token = os.environ.get("SEED_SWEEP_INFERENCE_COMPLETION_TOKEN", "")
    from scripts.seed_sweep.completion_marker import write_completion_marker

    write_completion_marker(
        marker_path,
        producer="main.py",
        run_id=run_id,
        token=token,
    )


def prepare_sampler_config(sampler_config, num_replicas, rank, seed):
    sampler_config = set_config_value(sampler_config, "num_replicas", num_replicas)
    sampler_config = set_config_value(sampler_config, "rank", rank)
    sampler_config = set_config_value(sampler_config, "seed", seed)
    return sampler_config

def get_resumable_dir(resume_path: str = None) -> str:
    """
    OmegaConf Resolver: 计算最终工作目录
    - Resume: 自动定位到原实验根目录，并在其下新建 outputN
    - New Run: 模拟 Hydra 默认行为，生成 outputs/YYYY-MM-DD/HH-MM-SS
    """
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    # 使用当前 CWD (项目根目录) 的 Hash 作为同步 ID
    run_id = hashlib.md5(os.getcwd().encode()).hexdigest()
    tmp_file = os.path.join(tempfile.gettempdir(), f"hydra_resolver_{run_id}.store")
    
    # 尝试建立通信 (Fail-safe)
    try:
        store = dist.FileStore(tmp_file, world_size)
    except:
        store = None

    key = "resolved_path"
    final_path_str = None

    # =========================================================
    # Rank 0: 决策逻辑
    # =========================================================
    if rank == 0:
        # 1. 判断是 Resume 还是 New Run
        if resume_path and os.path.exists(resume_path):
            # --- Resume 模式 ---
            ckpt_path = Path(resume_path).absolute()
            # 跳 3 层: file → checkpoints/ → output/ → hydra.run.dir
            exp_root = ckpt_path.parent.parent.parent
            # hydra.run.dir 可能是 exp_root/outputN（来自上次 resume），
            # 需要继续向上走到真正的实验根。
            while re.match(r"output\d+$", exp_root.name):
                exp_root = exp_root.parent
            
            # 在根目录下计算下一个 outputN
            target_dir_name = _get_next_output_dir(exp_root)
            final_path = exp_root / target_dir_name
            mode = "Resume"
        else:
            # --- New Run 模式 (模仿 Hydra 默认) ---
            # 由 Rank 0 生成时间戳，保证所有卡完全一致
            now = datetime.datetime.now()
            # 格式: outputs/2023-10-01/12-00-00
            rel_path = f"outputs/{now.strftime('%Y-%m-%d')}_{now.strftime('%H-%M-%S')}"
            final_path = Path(os.getcwd()) / rel_path
            mode = "New Run"

        final_path_str = str(final_path)
        
        # 广播给其他卡
        if store:
            store.set(key, final_path_str)
            
    # =========================================================
    # Rank > 0: 等待逻辑
    # =========================================================
    else:
        if store:
            val = store.get(key)
            final_path_str = val.decode("utf-8")
        else:
            # 如果 Store 挂了的保底（极罕见），只能冒险让大家自己算
            # 这种情况下无法保证时间戳秒级对齐
            final_path_str = str(Path(os.getcwd()) / "outputs/fallback_error")

    return final_path_str


def _get_next_output_dir(base_dir: Path) -> str:
    """逻辑不变：扫描并返回 outputN"""
    if not base_dir.exists(): return "output" # 理论上不会发生，因为 base_dir 是原实验目录
    
    # 扫描 output*
    subdirs = [d.name for d in base_dir.iterdir() if d.is_dir() and d.name.startswith("output")]
    
    # 如果这是第一次 Resume（根目录下只有 .hydra, checkpoints, main.log）
    # 还没有任何 output 文件夹，那么第一个就是 output1 (或者 output，看你喜好)
    if not subdirs:
        return "output1"
    
    max_idx = 0
    for name in subdirs:
        if name == "output": continue # 兼容旧习惯
        match = re.match(r"output(\d+)", name)
        if match:
            idx = int(match.group(1))
            if idx > max_idx: max_idx = idx
            
    return f"output{max_idx + 1}"

OmegaConf.register_new_resolver("calc_work_dir", get_resumable_dir)

@hydra.main(config_path="./configs", config_name="config")
def main(config) -> None:
    if config.debug:
        config.train.log_interval = 1
        config.data.train_batch_size = min(64, config.data.train_batch_size)
        config.report_to = "tensorboard"
    
    if config.get("allow_tf32", True):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    os.umask(0o000)
    os.makedirs(config.work_dir, exist_ok=True)

    init_handler = InitProcessGroupKwargs()
    init_handler.timeout = datetime.timedelta(seconds=5400)  # change timeout to avoid a strange NCCL bug

    # Initialize accelerator and tensorboard logging
    accelerator = Accelerator(
        mixed_precision=config.model.mixed_precision,
        gradient_accumulation_steps=config.train.gradient_accumulation_steps,
        log_with=config.report_to,
        project_dir=osp.join(config.work_dir, "logs"),
        kwargs_handlers=[init_handler],
    )
    log_name = "train_log.log"
    logger = get_root_logger(osp.join(config.work_dir, log_name))
    logger.info(accelerator.state)
    
    config.train.seed = init_random_seed(getattr(config.train, "seed", None))
    set_random_seed(config.train.seed + int(os.environ["LOCAL_RANK"]), deterministic=False)
    generator = torch.Generator(device=accelerator.device).manual_seed(config.train.seed)
    
    if accelerator.is_main_process:
        # pyrallis.dump(config, open(osp.join(config.work_dir, "config.yaml"), "w"), sort_keys=False, indent=4)
        with open(osp.join(config.work_dir, "config.yaml"), "w") as f:
            OmegaConf.save(config, f)
        if config.report_to == "wandb":
            import wandb
            wandb.init(project=config.tracker_project_name, experiment=config.name, name=config.name, resume="allow")
        if config.report_to == "swanlab":
            import swanlab
            swanlab.init(project=config.tracker_project_name, experiment=config.name, name=config.name, resume="allow")
    
    logger.info(f"Config: \n{config}")
    logger.info(f"World_size: {get_world_size()}, seed: {config.train.seed}")
        
    # 5. build dataloader
    num_replicas = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])

    # The frozen TalkSHOW evaluation launchers opt into a strict test-only or
    # validation-only runtime.  In those modes, instantiate only the requested
    # evaluation dataset: opening the large training LMDB is unnecessary and
    # can dominate startup time.  The checks below deliberately keep these
    # paths scoped to the supported TalkSHOW BLD setup so existing BEAT2 and
    # training behavior remains unchanged.
    talkshow_test_only = bool(config.get("talkshow_test_only", False))
    talkshow_validation_only = bool(
        config.get("talkshow_validation_only", False)
    )
    if talkshow_test_only and talkshow_validation_only:
        raise ValueError(
            "talkshow_test_only and talkshow_validation_only are mutually "
            "exclusive"
        )
    if talkshow_test_only:
        if config.trainer != "bld" or config.get("talkshow_profile") is None:
            raise ValueError(
                "talkshow_test_only is supported only by a TalkSHOW BLD profile"
            )
        if (
            bool(config._train)
            or bool(config._validation)
            or not bool(config._test)
            or bool(config._predict)
        ):
            raise ValueError(
                "talkshow_test_only requires _train=false, "
                "_validation=false, _test=true, and _predict=false"
            )
        if bool(config.train.get("finetune_speech_model", False)):
            raise ValueError(
                "talkshow_test_only requires train.finetune_speech_model=false"
            )
        if bool(config.train.get("normalize", False)):
            raise ValueError(
                "talkshow_test_only does not support train.normalize=true "
                "because no training dataset is instantiated"
            )
        if bool(config.train.get("use_fsdp", False)):
            raise ValueError("talkshow_test_only supports DDP inference only")
        if config.get("resume_from") or config.model.get(
            "resume_from",
            None,
        ):
            raise ValueError(
                "talkshow_test_only forbids resume_from/model.resume_from; "
                "load inference weights with model.load_from"
            )
        if not config.model.get("load_from", None):
            raise ValueError(
                "talkshow_test_only requires a model.load_from checkpoint"
            )
        train_dataset_config = None
        validation_dataset_config = None
        test_dataset_config = config.data.test
        predict_dataset_config = None
        logger.info(
            "TalkSHOW test-only runtime: instantiating only the test dataset"
        )
    elif talkshow_validation_only:
        if config.trainer != "bld" or config.get("talkshow_profile") is None:
            raise ValueError(
                "talkshow_validation_only is supported only by a TalkSHOW "
                "BLD profile"
            )
        if (
            bool(config._train)
            or not bool(config._validation)
            or bool(config._test)
            or bool(config._predict)
        ):
            raise ValueError(
                "talkshow_validation_only requires _train=false, "
                "_validation=true, _test=false, and _predict=false"
            )
        if bool(config.train.get("finetune_speech_model", False)):
            raise ValueError(
                "talkshow_validation_only requires "
                "train.finetune_speech_model=false"
            )
        if bool(config.train.get("normalize", False)):
            raise ValueError(
                "talkshow_validation_only does not support "
                "train.normalize=true because no training dataset is "
                "instantiated"
            )
        if bool(config.train.get("use_fsdp", False)):
            raise ValueError(
                "talkshow_validation_only supports DDP inference only"
            )
        if config.get("resume_from") or config.model.get(
            "resume_from",
            None,
        ):
            raise ValueError(
                "talkshow_validation_only forbids "
                "resume_from/model.resume_from; load inference weights with "
                "model.load_from"
            )
        if not config.model.get("load_from", None):
            raise ValueError(
                "talkshow_validation_only requires a model.load_from "
                "checkpoint"
            )
        train_dataset_config = None
        validation_dataset_config = config.data.validation
        test_dataset_config = None
        predict_dataset_config = None
        logger.info(
            "TalkSHOW validation-only runtime: instantiating only the "
            "validation dataset"
        )
    else:
        train_dataset_config = config.data.train
        validation_dataset_config = config.data.validation
        test_dataset_config = config.data.test
        predict_dataset_config = (
            config.data.predict if hasattr(config.data, "predict") else None
        )

    data = DataModule(
        config, 
        train_dataset_config=train_dataset_config,
        validation_dataset_config=validation_dataset_config,
        test_dataset_config=test_dataset_config,
        predict_dataset_config=predict_dataset_config,
    )
    accelerator.wait_for_everyone()
    
    if train_dataset_config is not None:
        train_sampler_config = config.data.get("train_sampler", DEFAULT_SAMPLER_CONFIG)
        train_sampler_config = prepare_sampler_config(train_sampler_config, num_replicas, rank, config.train.seed)
    else:
        train_sampler_config = None
    
    if validation_dataset_config is not None:
        validation_sampler_config = config.data.get("validation_sampler", DEFAULT_SAMPLER_CONFIG)
        validation_sampler_config = prepare_sampler_config(validation_sampler_config, num_replicas, rank, config.train.seed)
    else:
        validation_sampler_config = None
    
    if test_dataset_config is not None:
        test_sampler_config = config.data.get("test_sampler", DEFAULT_SAMPLER_CONFIG)
        test_sampler_config = prepare_sampler_config(test_sampler_config, num_replicas, rank, config.train.seed)
    else:
        test_sampler_config = None
    
    if predict_dataset_config is not None:
        predict_sampler_config = config.data.get("predict_sampler", DEFAULT_SAMPLER_CONFIG)
        predict_sampler_config = prepare_sampler_config(predict_sampler_config, num_replicas, rank, config.train.seed)
    else:
        predict_sampler_config = None
    
    data.prepare_data(
        train_sampler_config,
        validation_sampler_config,
        test_sampler_config,
        predict_sampler_config,
    )
    if config.trainer == "emage":
        from diffusion.trainer import EmageTrainer as Trainer
        trainer = Trainer(
            accelerator,
            data,
            config.model.model,
            config,
            num_replicas=num_replicas,
            logger=logger,
            generator=generator
        )
    elif config.trainer == "bld":
        from diffusion.trainer import BLDTrainer as Trainer
        trainer = Trainer(
            accelerator,
            data,
            config.model.model,
            config,
            num_replicas=num_replicas,
            logger=logger,
            generator=generator
        )
    else:
        from diffusion.trainer import MotionTrainer as Trainer
        trainer = Trainer(
            accelerator,
            data,
            config.model.model,
            config,
            num_replicas=num_replicas,
            logger=logger,
            generator=generator
        )
    
    # # allow checkpointing via USR1
    # def melk(*args, **kwargs):
    #     # run all checkpoint hooks
    #     if trainer.global_rank == 0:
    #         logger.info("Summoning checkpoint.")
    #         if melk_ckpt_name is None:
    #             ckpt_path = os.path.join(ckptdir, "last.ckpt")
    #         else:
    #             ckpt_path = os.path.join(ckptdir, melk_ckpt_name)
    #         trainer.save_checkpoint(ckpt_path)

    # def divein(*args, **kwargs):
    #     if trainer.global_rank == 0:
    #         import pudb

    #         pudb.set_trace()

    # import signal

    # signal.signal(signal.SIGUSR1, melk)
    # signal.signal(signal.SIGUSR2, divein)
    
    if config._train:
        trainer.train()
    if config._validation:
        trainer.validation()
    if config._test:
        trainer.test()
    if config._predict:
        trainer.predict()
    if os.environ.get("SEED_SWEEP_INFERENCE_COMPLETION_MARKER"):
        # A rank-0-only marker can otherwise race ahead of slower ranks and let
        # the parent kill a still-running distributed test. This barrier is
        # enabled only for completion-aware seed-sweep inference.
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            _publish_seed_sweep_inference_completion()
    logger.info("Done!")
    dist.destroy_process_group()
    
if __name__ == "__main__":
    main()
