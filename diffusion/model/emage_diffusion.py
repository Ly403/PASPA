import torch as th
import numpy as np
from diffusion.model.utils import expand_dims_like
from diffusion.model.respace import SpacedDiffusion
from diffusion.model.diffusion_utils import (
    discretized_gaussian_log_likelihood, 
    normal_kl
)
from diffusion.model.gaussian_diffusion import (
    LossType,
    ModelMeanType,
    ModelVarType,
    mean_flat,
    sum_flat,
    _extract_into_tensor
)
from diffusion.utils import rotation_conversions as rc
from typing import Dict, Union
from omegaconf import DictConfig
from torch.nn import functional as F
from diffusion.utils.FK import SMPLX_PARENTS
from diffusion.model.vae.VAE import MaskedVAE2, MaskedVAE3
from diffusion.utils.logger import get_root_logger
from termcolor import colored

def VEL(x: th.Tensor):
    return x[:, 1:] - x[:, :-1]


def ACC(x: th.Tensor):
    return VEL(VEL(x))

def geodesic_loss(m1, m2, reduction="mean"):
    """Compute the geodesic distance between two rotation matrices.

    Args:
        m1, m2: Two rotation matrices with the shape (batch x 3 x 3).

    Returns:
        The minimal angular difference between two rotation matrices in radian form [0, pi].
    """
    m1 = m1.reshape(-1, 3, 3)
    m2 = m2.reshape(-1, 3, 3)
    batch = m1.shape[0]
    m = th.bmm(m1, m2.transpose(1, 2))  # batch*3*3
    cos = (m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2] - 1) / 2
    cos = th.clamp(cos, min=-1 + 1e-6, max=1 - 1e-6)
    loss = th.acos(cos)

    if reduction == "mean":
        return loss.mean()
    elif reduction == "none":
        return loss
    else:
        raise RuntimeError(f"unsupported reduction: {reduction}")

def recon_loss_with_mask(motion_pred, motion_gt, mask):
    assert len(motion_gt.shape) == 3
    assert len(motion_pred.shape) == 3

    B, L = motion_pred.shape[0], motion_pred.shape[1]
    motion_pred_r = rc.rotation_6d_to_matrix_torch(motion_pred.reshape((-1, 6)))
    motion_gt_r = rc.rotation_6d_to_matrix_torch(motion_gt.reshape((-1, 6)))
    rotation_loss = geodesic_loss(motion_pred_r, motion_gt_r, reduction="none").reshape((B, L, -1))
    nJonts = rotation_loss.shape[-1]
    mask = ~mask.unsqueeze(-1).expand((B, L, nJonts))
    rotation_loss = (rotation_loss * mask).sum() / (mask.sum() + 1e-5)
    return rotation_loss

class EmageDiffusion(SpacedDiffusion):
    """
        这个版本比SimpleDiffusion还要简化，这是只有motion用diffusion生成，expression直接从audio中映射出来。
        考虑到虽然expression的模型大部分情况下都是先训好的，但是也有可能需要微调
        所以还是留下微调的接口，就是expression的loss
    """
    def __init__(
        self, 
        task="audio2motion+expression", 
        joints=55,
        overlap_length=8,
        lambda_motion=1.0,
        lambda_expression=1.0,
        normalize_rot6d=False,
        other_lambdas: Union[DictConfig, Dict]=None, 
        # 有些损失函数，涉及更多的参数调节，这里直接放一个字典，增强灵活性
        logger=None,
        device="cuda",
        vae2_path=None,
        vae3_path=None,
        # logvar 相关的稳定性配置
        logvar_clamp_min=-10.0,  # logvar 最小值，防止权重爆炸
        logvar_clamp_max=10.0,   # logvar 最大值
        geometric_loss_logvar_mode="detach",  # 几何损失的logvar模式: "full", "detach", "none"
        # motion loss weighting（可扩展）：支持 uniform / inverse_poly 等
        # inverse_poly: w(t) = 1 / (1 + k * t_norm^alpha)
        loss_weighting_config: Union[DictConfig, Dict, None] = None,
        # FK/几何损失仅在 timestep <= fk_loss_max_timestep 时启用。
        # None 表示默认使用训练采样总步数（num_timesteps）。
        fk_loss_max_timestep: int | None = None,
        # 是否按 block 计算 loss（形状从 B,T,C reshape 为 (B*块数, block_size, C)）
        blockwise_loss: bool = False,
        loss_block_size: int | None = None,
        *args,
        **kwargs
    ):
        self.task = task
        self.joints = joints
        self.overlap_length = overlap_length
        self.lambda_motion = lambda_motion
        self.normalize_rot6d = normalize_rot6d
        self.other_lambdas = other_lambdas
        self.logger = get_root_logger() if logger is None else logger
        self.device = device
        # logvar 稳定性配置
        self.logvar_clamp_min = logvar_clamp_min
        self.logvar_clamp_max = logvar_clamp_max
        self.geometric_loss_logvar_mode = geometric_loss_logvar_mode
        self.loss_weighting_config = self._build_loss_weighting_config(loss_weighting_config)
        self.fk_loss_max_timestep = fk_loss_max_timestep
        self.blockwise_loss = blockwise_loss
        self.loss_block_size = loss_block_size

        # Cache SMPLX_PARENTS as tensor to avoid repeated numpy array access and CPU-GPU transfer
        # This will be lazily moved to the correct device on first use
        self._smplx_parents_tensor = None

        super().__init__(*args, **kwargs)
        self.fk_loss_max_timestep = self._resolve_fk_loss_max_timestep(self.fk_loss_max_timestep)
        lambda_latent_loss = other_lambdas.get("lambda_latent_loss", 1.0)
        if lambda_latent_loss > 0:
            self.vae2, self.vae3 = self.configure_vae(vae2_path, vae3_path)
        else:
            self.vae2, self.vae3 = None, None

    def configure_vae(self, vae2_path, vae3_path):
        vae3 = MaskedVAE3(self.joints * 6 + 3, pred_logvar=True).to(self.device).requires_grad_(False).eval()
        missing, unexpected = vae3.load_state_dict(th.load(vae3_path))
        self.logger.info(colored(f"load vae3 from {vae3_path}", "green", attrs=["bold"]))
        if len(missing) > 0 or len(unexpected) > 0:
            self.logger.warning(f"missing keys: {missing}")
            self.logger.warning(f"unexpected keys: {unexpected}")
        
        vae2= MaskedVAE2(self.joints * 6 + 3, pred_logvar=True).to(self.device).requires_grad_(False).eval()
        missing, unexpected = vae2.load_state_dict(th.load(vae2_path))
        self.logger.info(colored(f"load vae2 from {vae2_path}", "green", attrs=["bold"]))
        if len(missing) > 0 or len(unexpected) > 0:
            self.logger.warning(f"missing keys: {missing}")
            self.logger.warning(f"unexpected keys: {unexpected}")
        return vae2, vae3

    def _get_smplx_parents_tensor(self, device):
        """
        Lazy loading and caching of SMPLX_PARENTS as a tensor.
        Converts numpy array to torch tensor on first access and caches it.

        Args:
            device: Target device for the tensor

        Returns:
            torch.Tensor: SMPLX_PARENTS as a long tensor on the specified device
        """
        if self._smplx_parents_tensor is None or self._smplx_parents_tensor.device != device:
            self._smplx_parents_tensor = th.from_numpy(np.array(SMPLX_PARENTS)).long().to(device)
        return self._smplx_parents_tensor

    def _build_loss_weighting_config(self, loss_weighting_config: Union[DictConfig, Dict, None]) -> Dict:
        """Build and validate loss weighting config.

        Keep backward compatibility with uniform weighting while exposing a
        generic interface for future weighting strategies.
        """
        config = {
            "mode": "uniform",
            # Parameters for inverse_poly mode: w(t)=1/(1+k*t_norm^alpha)
            "k": 4.0,
            "alpha": 1.0,
            "min_weight": 0.0,
            "max_weight": 1.0,
            # Optional normalization: none | mean1
            "normalize": "none",
            "eps": 1e-8,
            # Parameters for flow_x0_clipped mode: w(t) = min(1/t^2, gamma)
            "gamma": 100.0,  # clip threshold for 1/t^2
            "t_min": 1e-3,   # minimum t value to avoid numerical instability
        }

        if loss_weighting_config is None:
            return config

        if isinstance(loss_weighting_config, DictConfig):
            user_config = {k: loss_weighting_config[k] for k in loss_weighting_config.keys()}
        elif isinstance(loss_weighting_config, dict):
            user_config = dict(loss_weighting_config)
        else:
            raise TypeError(
                f"loss_weighting_config must be DictConfig/dict/None, got {type(loss_weighting_config)}"
            )

        config.update(user_config)
        return config

    def _compute_base_motion_weighting(self, timestep: th.Tensor) -> th.Tensor:
        """Compute base motion weighting from timestep before lambda/logvar/frame shaping."""
        cfg = self.loss_weighting_config
        mode = str(cfg.get("mode", "uniform")).lower()
        eps = float(cfg.get("eps", 1e-8))

        timestep_float = timestep.to(dtype=th.float32)

        if mode == "uniform":
            base_weight = th.ones_like(timestep_float)
        elif mode in ["inverse_poly", "timestep_inverse_poly", "inv_poly"]:
            k = float(cfg.get("k", 4.0))
            alpha = float(cfg.get("alpha", 1.0))
            min_weight = float(cfg.get("min_weight", 0.0))
            max_weight = float(cfg.get("max_weight", 1.0))

            if alpha <= 0:
                raise ValueError(f"loss_weighting_config.alpha must be > 0, got {alpha}")

            if hasattr(self, "num_timesteps") and self.num_timesteps is not None and int(self.num_timesteps) > 1:
                t_denom = float(int(self.num_timesteps) - 1)
                t_norm = timestep_float / t_denom
            else:
                # Fallback for safety; normal path should use self.num_timesteps.
                t_denom = th.clamp(timestep_float.max(), min=1.0)
                t_norm = timestep_float / t_denom

            t_norm = th.clamp(t_norm, min=0.0, max=1.0)
            denom = 1.0 + k * th.pow(t_norm, alpha)
            denom = th.clamp(denom, min=eps)
            base_weight = 1.0 / denom
            base_weight = th.clamp(base_weight, min=min_weight, max=max_weight)
        elif mode in ["flow_x0_clipped", "flow_x0_clip", "x0_clipped"]:
            # Flow Matching x0-MSE loss weighting with clipping
            # For flow matching with alpha_t = 1-t, sigma_t = t:
            # Exact weight from velocity-to-x0 conversion: w(t) = 1/t^2
            # Apply clipping to avoid extreme weights: w_clip(t) = min(1/t^2, gamma)
            gamma = float(cfg.get("gamma", 100.0))
            t_min = float(cfg.get("t_min", 1e-3))
            
            # Normalize timestep to [0, 1] range
            if hasattr(self, "num_timesteps") and self.num_timesteps is not None and int(self.num_timesteps) > 1:
                t_denom = float(int(self.num_timesteps) - 1)
                t_norm = timestep_float / t_denom
            else:
                # Fallback: assume timestep is already in [0, 1] or normalize by max
                t_max = timestep_float.max()
                if t_max > 1.0:
                    t_norm = timestep_float / t_max
                else:
                    t_norm = timestep_float
            
            # Clamp t to avoid division by zero or near-zero values
            t_clamped = th.clamp(t_norm, min=t_min, max=1.0)
            # Compute 1 / t^2
            inv_t2 = 1.0 / (t_clamped ** 2)
            # Apply clipping: w(t) = min(1/t^2, gamma)
            base_weight = th.clamp(inv_t2, max=gamma)
        elif mode in ["min_snr", "min_snr_x0", "minsnr"]:
            # Min-SNR weighting for Flow Matching x0-MSE loss
            # This is a training stabilization strategy, NOT strictly equivalent to velocity-loss c_t^2 weighting
            #
            # For flow matching with alpha_t = 1-t, sigma_t = t:
            # SNR(t) = alpha_t^2 / sigma_t^2 = (1-t)^2 / t^2
            #
            # Low noise corresponds to t -> 0, where SNR is very large
            # To emphasize low-noise samples while avoiding extreme weights dominating training,
            # we apply Min-SNR clipping: weight(t) = min(SNR(t), gamma)
            #
            # gamma is a configurable hyperparameter (e.g., 5.0 or 10.0)
            gamma = float(cfg.get("gamma", 5.0))
            t_min = float(cfg.get("t_min", 1e-4))
            
            # Normalize timestep to [0, 1] range
            if hasattr(self, "num_timesteps") and self.num_timesteps is not None and int(self.num_timesteps) > 1:
                t_denom = float(int(self.num_timesteps) - 1)
                t_norm = timestep_float / t_denom
            else:
                # Fallback: assume timestep is already in [0, 1] or normalize by max
                t_max = timestep_float.max()
                if t_max > 1.0:
                    t_norm = timestep_float / t_max
                else:
                    t_norm = timestep_float
            
            # Clamp t to avoid division by zero or near-zero values
            t_safe = th.clamp(t_norm, min=t_min, max=1.0)
            
            # Compute SNR(t) = (1-t)^2 / t^2
            alpha_t = 1.0 - t_safe
            sigma_t = t_safe
            snr = (alpha_t ** 2) / (sigma_t ** 2)
            
            # Apply Min-SNR clipping: weight(t) = min(SNR(t), gamma)
            base_weight = th.clamp(snr, max=gamma)
        else:
            raise ValueError(
                f"Unknown loss weighting mode: {mode}. "
                "Supported modes: ['uniform', 'inverse_poly', 'flow_x0_clipped', 'min_snr']"
            )

        normalize_mode = str(cfg.get("normalize", "none")).lower()
        if normalize_mode == "mean1":
            if base_weight.dim() >= 2:
                reduce_dims = tuple(range(1, base_weight.dim()))
                mean_weight = base_weight.mean(dim=reduce_dims, keepdim=True)
            else:
                mean_weight = base_weight.mean()
            base_weight = base_weight / (mean_weight + eps)
        elif normalize_mode == "none":
            pass
        else:
            raise ValueError(
                f"Unknown loss_weighting_config.normalize: {normalize_mode}. "
                "Supported: ['none', 'mean1']"
            )

        return base_weight

    def _resolve_fk_loss_max_timestep(self, fk_loss_max_timestep: int | None) -> int:
        """
        Resolve FK loss activation threshold.

        Default behavior (None): use training timestep count (self.num_timesteps),
        which is aligned with scheduler.train_sampling_steps in current setup.
        """
        if fk_loss_max_timestep is None:
            if hasattr(self, "num_timesteps") and self.num_timesteps is not None:
                fk_loss_max_timestep = int(self.num_timesteps)
            else:
                fk_loss_max_timestep = 1000
        fk_loss_max_timestep = int(fk_loss_max_timestep)
        if fk_loss_max_timestep < 0:
            raise ValueError(f"fk_loss_max_timestep must be >= 0, got {fk_loss_max_timestep}")
        return fk_loss_max_timestep

    def _build_fk_loss_gate(self, timestep: th.Tensor | None, ref_tensor: th.Tensor) -> th.Tensor:
        """
        Build gating mask for FK-related geometric losses.
        gate = 1 when timestep <= fk_loss_max_timestep else 0.
        """
        if timestep is None:
            gate = th.ones(ref_tensor.shape[:2], device=ref_tensor.device, dtype=ref_tensor.dtype)
            return expand_dims_like(gate, ref_tensor)

        t = timestep.to(device=ref_tensor.device)
        while t.dim() > 2 and t.shape[-1] == 1:
            t = t.squeeze(-1)
        gate = (t <= self.fk_loss_max_timestep).to(dtype=ref_tensor.dtype)
        gate = expand_dims_like(gate, ref_tensor)
        return gate

    def loss_weighting(self, timestep, extra):
        # 可扩展的基础 weighting：uniform / inverse_poly / future modes
        loss_motion_weighting = self._compute_base_motion_weighting(timestep)
        loss_motion_weighting = loss_motion_weighting * self.lambda_motion
        
        # 几何损失的权重（可能不使用logvar加权，或使用detached版本）
        geometric_loss_weighting = loss_motion_weighting.clone()
        
        if extra is not None and "logvar" in extra:
            logvar = extra["logvar"]
            
            # 只有当 min 或 max 不是 None 且不是 "-none" 时才进行 clamp
            if self.logvar_clamp_min is not None and str(self.logvar_clamp_min).lower() != "none" \
               and self.logvar_clamp_max is not None and str(self.logvar_clamp_max).lower() != "none":
                logvar_clamped = th.clamp(logvar, min=float(self.logvar_clamp_min), max=float(self.logvar_clamp_max))
            else:
                logvar_clamped = logvar

            loss_motion_weighting = expand_dims_like(loss_motion_weighting, logvar_clamped)
            loss_motion_weighting = (loss_motion_weighting / logvar_clamped.exp())
            
            # 根据配置决定几何损失的加权方式
            if self.geometric_loss_logvar_mode == "full":
                # 完全使用 logvar 加权（与主损失相同）
                geometric_loss_weighting = loss_motion_weighting.clone()
            elif self.geometric_loss_logvar_mode == "detach":
                # 使用 detached 的 logvar，不传递梯度
                geometric_loss_weighting = expand_dims_like(geometric_loss_weighting, logvar_clamped)
                geometric_loss_weighting = (geometric_loss_weighting / logvar_clamped.detach().exp())
            elif self.geometric_loss_logvar_mode == "none":
                # 不对几何损失使用 logvar 加权
                geometric_loss_weighting = expand_dims_like(geometric_loss_weighting, logvar_clamped)
            else:
                raise ValueError(f"Unknown geometric_loss_logvar_mode: {self.geometric_loss_logvar_mode}")

        # 可选：外部传入的逐帧监督权重（用于 BLD 的 block-level supervision shaping）
        # 兼容旧逻辑：默认不存在该 key 时，不影响任何行为
        if extra is not None and "frame_supervision_weight" in extra:
            frame_supervision_weight = extra["frame_supervision_weight"]
            frame_supervision_weight = frame_supervision_weight.to(
                device=loss_motion_weighting.device,
                dtype=loss_motion_weighting.dtype,
            )

            # 兼容 (B, L, 1) / (B, L) 两种输入
            while frame_supervision_weight.dim() > loss_motion_weighting.dim() and frame_supervision_weight.shape[-1] == 1:
                frame_supervision_weight = frame_supervision_weight.squeeze(-1)
            frame_supervision_weight = expand_dims_like(frame_supervision_weight, loss_motion_weighting)

            loss_motion_weighting = loss_motion_weighting * frame_supervision_weight
            geometric_loss_weighting = geometric_loss_weighting * frame_supervision_weight
        
        return loss_motion_weighting, geometric_loss_weighting

    def _reshape_blockwise(self, tensor: th.Tensor, block_size: int) -> th.Tensor:
        """
        将 (B, T, ...) reshape 为 (B * num_blocks, block_size, ...)，其中 T 必须能被 block_size 整除。
        """
        if tensor is None:
            return None
        if tensor.dim() < 2:
            return tensor
        B, T = tensor.shape[:2]
        if T % block_size != 0:
            raise ValueError(f"blockwise reshape 需要 T 可被 block_size 整除，当前 T={T}, block_size={block_size}")
        num_blocks = T // block_size
        new_shape = (B, num_blocks, block_size) + tensor.shape[2:]
        return tensor.reshape(new_shape).reshape(B * num_blocks, block_size, *tensor.shape[2:])
    
    def loss_function(
        self, 
        motion_target, 
        motion_output, 
        loss_motion_weighting,
        geometric_loss_weighting,
        extra,
        **kwargs
    ):
        if self.loss_type == LossType.MSE or self.loss_type == LossType.RESCALED_MSE:
            # 这是L2损失
            loss_motion = (motion_target - motion_output) ** 2 * loss_motion_weighting
            return {
                "mse_motion": th.mean(loss_motion),
            }
        elif self.loss_type == LossType.MAE or self.loss_type == LossType.RESCALED_MAE:
            # 这是L1损失
            loss_motion = th.abs(motion_target - motion_output) * loss_motion_weighting
            return {
                "mae_motion": th.mean(loss_motion),
            }
        elif self.loss_type == LossType.MOT_ANGLEDIS_EXP_MSE or self.loss_type == LossType.RESCALED_MOT_ANGLEDIS_EXP_MSE:
            # 这是motion angle, expression MSE, audio MAE损失
            motion_target_rot6d = motion_target[:, :, :self.joints*6]
            motion_output_rot6d = motion_output[:, :, :self.joints*6]
            motion_target_others = motion_target[:, :, self.joints*6:]
            motion_output_others = motion_output[:, :, self.joints*6:]
            motion_target_matrix = rc.rotation_6d_to_matrix(
                motion_target_rot6d.reshape(*motion_target_rot6d.shape[0:2], -1, 6))
            motion_output_matrix = rc.rotation_6d_to_matrix(
                motion_output_rot6d.reshape(*motion_output_rot6d.shape[0:2], -1, 6))
            motion_target_matrix = motion_target_matrix.reshape(*motion_target_matrix.shape[0:2], -1, 3, 3)
            motion_output_matrix = motion_output_matrix.reshape(*motion_output_matrix.shape[0:2], -1, 3, 3)
            matrix_product = motion_target_matrix @ motion_output_matrix.transpose(-1, -2)
            log_matrix_product = rc.matrix_to_axis_angle(matrix_product)
            loss_motion = th.sqrt(th.sum(
                log_matrix_product ** 2, dim=list(
                    range(1, log_matrix_product.dim())
                )
            )) * loss_motion_weighting.squeeze()
            loss_motion_others = (motion_target_others - motion_output_others) ** 2 * loss_motion_weighting
            return {
                "motion_angle_dis": th.mean(loss_motion),
                "mse_motion_others": th.mean(loss_motion_others),
            }
        elif self.loss_type == LossType.MOT_GEO_EXP_MSE \
            or self.loss_type == LossType.RESCALED_MOT_GEO_EXP_MSE \
            or self.loss_type == LossType.MOT_JointsGEO_EXP_MSE \
            or self.loss_type == LossType.RESCALED_MOT_JointsGEO_EXP_MSE:
            # 这是motion geometric loss，几何损失，用motion diffusion models里面提出的几个损失的计算方法
            loss_motion_mse = (motion_target - motion_output) ** 2 * loss_motion_weighting # mse保留
            # for motion geometric loss
            motion_target_rot6d = motion_target[:, :, :self.joints*6]
            motion_output_rot6d = motion_output[:, :, :self.joints*6]
            motion_target_others = motion_target[:, :, self.joints*6:]
            motion_output_others = motion_output[:, :, self.joints*6:]
            # 从下面需要计算几个部分的几何损失
            lambda_fc = self.other_lambdas.get("lambda_fc", 1.0) 
            lambda_vel = self.other_lambdas.get("lambda_vel", 1.0)
            lambda_vel_rcxyz = self.other_lambdas.get("lambda_vel_rcxyz", 1.0)
            lambda_rcxyz = self.other_lambdas.get("lambda_rcxyz", 1.0)
            smpl_model = kwargs.get("smpl_model", None)
            tar_beta = kwargs.get("tar_beta", None)
            if smpl_model is None or tar_beta is None:
                raise ValueError("smpl_model is needed for motion geometric loss")
            
            geometric_loss = {}
            if lambda_vel > 0:
                motion_target_rot6d_vel = motion_target_rot6d[:, 1:, :] - motion_target_rot6d[:, :-1, :]
                motion_output_rot6d_vel = motion_output_rot6d[:, 1:, :] - motion_output_rot6d[:, :-1, :]
                loss_motion_vel = (motion_target_rot6d_vel - motion_output_rot6d_vel) ** 2\
                    * lambda_vel * loss_motion_weighting
                geometric_loss["mse_motion_vel"] = th.mean(loss_motion_vel)

            if lambda_vel_rcxyz > 0 or lambda_fc > 0 or lambda_rcxyz > 0:
                bs = motion_target_rot6d.shape[0]
                n = motion_target_rot6d.shape[1]
                if self.loss_type == LossType.MOT_GEO_EXP_MSE \
                    or self.loss_type == LossType.RESCALED_MOT_GEO_EXP_MSE:
                    rec_pose = rc.rotation_6d_to_axis_angle(
                        motion_output_rot6d.reshape(bs, n, -1, 6)
                    ).reshape(bs*n, -1)
                    rec_trans = motion_output_others[:, :, -3:]
                    rec_exps = kwargs["tar_exps"]
                    tar_pose = rc.rotation_6d_to_axis_angle(
                        motion_target_rot6d.reshape(bs, n, -1, 6)
                    ).reshape(bs*n, -1)
                    tar_trans = motion_target_others[:, :, -3:]
                    tar_exps = kwargs["tar_exps"]
                    
                    vertices_rec = smpl_model(
                        betas=tar_beta.reshape(bs*n, 300), 
                        transl=rec_trans.reshape(bs*n, 3), 
                        expression=rec_exps.reshape(bs*n, 100),
                        jaw_pose=rec_pose[:, 66:69], 
                        global_orient=rec_pose[:,:3], 
                        body_pose=rec_pose[:,3:21*3+3], 
                        left_hand_pose=rec_pose[:,25*3:40*3], 
                        right_hand_pose=rec_pose[:,40*3:55*3], 
                        return_joints=True, 
                        leye_pose=rec_pose[:, 69:72], 
                        reye_pose=rec_pose[:, 72:75],
                    )["vertices"]
                    vertices_rec = vertices_rec.reshape(bs, n, *vertices_rec.shape[1:])

                    vertices_tar = smpl_model(
                        betas=tar_beta.reshape(bs*n, 300), 
                        transl=tar_trans.reshape(bs*n, 3), 
                        expression=tar_exps.reshape(bs*n, 100), 
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
                    vertices_tar = vertices_tar.reshape(bs, n, *vertices_tar.shape[1:])
                else:
                    vertices_rec = kwargs["forward_kinematics_func"](
                        motion=motion_output_rot6d, trans=motion_target_others[:,:,:3])
                    vertices_tar = kwargs["forward_kinematics_func"](
                        motion=motion_target_rot6d, trans=motion_target_others[:,:,:3])
                
                fk_loss_gate = self._build_fk_loss_gate(kwargs.get("timestep", None), vertices_rec)
                loss_motion_weighting = expand_dims_like(loss_motion_weighting, vertices_rec) * fk_loss_gate
                if lambda_rcxyz > 0:
                    loss_motion_rcxyz = (vertices_rec - vertices_tar) ** 2\
                        * lambda_rcxyz * loss_motion_weighting
                    geometric_loss["mse_motion_rcxyz"] = th.mean(loss_motion_rcxyz)
                                    
                if lambda_vel_rcxyz > 0:
                    vertices_rec_vel = vertices_rec[:, 1:, ...] - vertices_rec[:, :-1, ...]
                    vertices_tar_vel = vertices_tar[:, 1:, ...] - vertices_tar[:, :-1, ...]
                    loss_motion_vel_rcxyz = (vertices_rec_vel - vertices_tar_vel) ** 2\
                        * lambda_vel_rcxyz * loss_motion_weighting
                    geometric_loss["mse_motion_vel_rcxyz"] = th.mean(loss_motion_vel_rcxyz)
                                    
                if lambda_fc > 0:
                    # left_ankle_index = joints_list["beat_smplx_joints"]["left_ankle"][-1]
                    # right_ankle_index = joints_list["beat_smplx_joints"]["right_ankle"][-1]
                    # left_foot_index = joints_list["beat_smplx_joints"]["left_foot"][-1]
                    # right_foot_index = joints_list["beat_smplx_joints"]["right_foot"][-1]
                    # all_indices = [left_ankle_index, right_ankle_index, left_foot_index, right_foot_index]
                    l_ankle_idx, r_ankle_idx, l_foot_idx, r_foot_idx = 7, 8, 10, 11
                    relevant_joints = [l_ankle_idx, l_foot_idx, r_ankle_idx, r_foot_idx]
                    
                    vertices_rec_fc = vertices_rec[:, :, relevant_joints]
                    vertices_tar_fc = vertices_tar[:, :, relevant_joints]
                    vertices_rec_fc_vel = vertices_rec_fc[:, 1:, ...] - vertices_rec_fc[:, :-1, ...]
                    vertices_tar_fc_vel = vertices_tar_fc[:, 1:, ...] - vertices_tar_fc[:, :-1, ...]
                    loss_motion_fc_vel = (vertices_rec_fc_vel - vertices_tar_fc_vel) ** 2 \
                        * lambda_fc * loss_motion_weighting
                    geometric_loss["mse_motion_fc_vel"] = th.mean(loss_motion_fc_vel)
                
            return {
                "mse_motion": th.mean(loss_motion_mse),
                **geometric_loss,
            }
        elif self.loss_type == LossType.EMAGE or self.loss_type == LossType.RESCALED_EMAGE \
            or self.loss_type == LossType.EMAGE_MSE or self.loss_type == LossType.RESCALED_EMAGE_MSE:
            motion_pred_rot6d = motion_output[:, :, :self.joints*6]
            motion_pred_trans = motion_output[:, :, self.joints*6:]
            motion_tar_rot6d = motion_target[:, :, :self.joints*6]
            motion_tar_trans = motion_target[:, :, self.joints*6:]
            
            bs, n = motion_pred_rot6d.shape[0:2]
            main_loss = {}
            
            lambda_rot6d = self.other_lambdas.get("lambda_rot6d", 1.0)
            lambda_translation = self.other_lambdas.get("lambda_translation", 1.0)
            
            if self.loss_type == LossType.EMAGE or self.loss_type == LossType.RESCALED_EMAGE:
                if lambda_rot6d > 0:
                    if self.normalize_rot6d:
                        norm_motion_pred_rot6d = rc.NormalizeRot6d(motion_pred_rot6d)
                        norm_motion_tar_rot6d = rc.NormalizeRot6d(motion_tar_rot6d)
                    else:
                        norm_motion_pred_rot6d = motion_pred_rot6d
                        norm_motion_tar_rot6d = motion_tar_rot6d
                    # mae_loss = F.l1_loss(norm_motion_pred_rot6d, norm_motion_tar_rot6d, reduction="none")
                    # mae_loss = mae_loss * expand_dims_like(loss_motion_weighting, mae_loss) * lambda_rot6d
                    # mae_loss = th.mean(mae_loss)
                    # 暂行，全部用MSE Loss，后续再调整，为了对齐
                    mse_loss = F.mse_loss(norm_motion_pred_rot6d, norm_motion_tar_rot6d, reduction="none")
                    mse_loss = mse_loss * expand_dims_like(loss_motion_weighting, mse_loss) * lambda_rot6d
                    mse_loss = th.mean(mse_loss)
                    main_loss["motion_loss"] = mse_loss
                if lambda_translation > 0:
                    # trans_loss = F.l1_loss(motion_pred_trans, motion_tar_trans, reduction="none")
                    # 暂行，全部用MSE Loss，后续再调整，为了对齐
                    trans_loss = F.mse_loss(motion_pred_trans, motion_tar_trans, reduction="none")
                    trans_loss = trans_loss * expand_dims_like(loss_motion_weighting, trans_loss) * lambda_translation
                    trans_loss = th.mean(trans_loss)
                    main_loss["translation_loss"] = trans_loss
            elif self.loss_type == LossType.EMAGE_MSE or self.loss_type == LossType.RESCALED_EMAGE_MSE:
                if lambda_rot6d > 0:
                    if self.normalize_rot6d:
                        norm_motion_pred_rot6d = rc.NormalizeRot6d(motion_pred_rot6d)
                        norm_motion_tar_rot6d = rc.NormalizeRot6d(motion_tar_rot6d)
                    else:
                        norm_motion_pred_rot6d = motion_pred_rot6d
                        norm_motion_tar_rot6d = motion_tar_rot6d
                    mse_loss = F.mse_loss(norm_motion_pred_rot6d, norm_motion_tar_rot6d, reduction="none")
                    mse_loss = mse_loss * expand_dims_like(loss_motion_weighting, mse_loss) * lambda_rot6d
                    mse_loss = th.mean(mse_loss)
                    main_loss["motion_loss"] = mse_loss
                if lambda_translation > 0:
                    trans_loss = F.mse_loss(motion_pred_trans, motion_tar_trans, reduction="none")
                    trans_loss = trans_loss * expand_dims_like(loss_motion_weighting, trans_loss) * lambda_translation
                    trans_loss = th.mean(trans_loss)
                    main_loss["translation_loss"] = trans_loss
            lambda_rotation = self.other_lambdas.get("lambda_rotation", 1.0)
            
            if lambda_rotation > 0:
                if self.loss_type == LossType.EMAGE or self.loss_type == LossType.RESCALED_EMAGE:
                    #################### rotation loss #################################                    
                    motion_pred_r = rc.rotation_6d_to_matrix_torch(motion_pred_rot6d.reshape((-1, 6)))
                    motion_gt_r = rc.rotation_6d_to_matrix_torch(motion_tar_rot6d.reshape((-1, 6)))
                    # I = motion_gt_r @ motion_pred_r.transpose(2, 1)
                    # rotation_loss = F.l1_loss(I, th.eye(3, device=I.device).unsqueeze(0).expand((I.size(0), -1, -1)), reduction="none").view(bs, n, -1, 3, 3)
                    rotation_loss = geodesic_loss(motion_gt_r, motion_pred_r, reduction="none").view(bs, n, -1)
                    rotation_loss = rotation_loss * expand_dims_like(loss_motion_weighting, rotation_loss) * lambda_rotation
                    rotation_loss = th.mean(rotation_loss)
                    
                    main_loss.update({
                        "rotation_loss": rotation_loss,
                    })
                else:
                    #################### rotation loss #################################
                    motion_pred_r = rc.rotation_6d_to_matrix_torch(motion_pred_rot6d.reshape((-1, 6)))
                    motion_gt_r = rc.rotation_6d_to_matrix_torch(motion_tar_rot6d.reshape((-1, 6)))
                    # I = motion_gt_r @ motion_pred_r.transpose(2, 1)
                    # rotation_loss = F.mse_loss(I, th.eye(3, device=I.device).unsqueeze(0).expand((I.size(0), -1, -1)), reduction="none").view(bs, n, -1, 3, 3)
                    rotation_loss = geodesic_loss(motion_gt_r, motion_pred_r, reduction="none").view(bs, n, -1)
                    rotation_loss = rotation_loss * expand_dims_like(loss_motion_weighting, rotation_loss) * lambda_rotation
                    rotation_loss = th.mean(rotation_loss)
                    
                    main_loss.update({
                        "rotation_loss": rotation_loss,
                    })
            ###########################################################################
            geometric_loss = {}
            
            lambda_vel_trans = self.other_lambdas.get("lambda_vel_trans", 1.0)
            lambda_acc_trans = self.other_lambdas.get("lambda_acc_trans", 1.0)
            if lambda_vel_trans > 0 or lambda_acc_trans > 0:
                if lambda_vel_trans > 0:
                    rec_trans_vel = VEL(motion_pred_trans)
                    tar_trans_vel = VEL(motion_tar_trans)
                    if self.loss_type == LossType.EMAGE_MSE or self.loss_type == LossType.RESCALED_EMAGE_MSE:
                        loss_motion_vel_trans = F.mse_loss(rec_trans_vel, tar_trans_vel, reduction="none")
                    else:
                        loss_motion_vel_trans = F.smooth_l1_loss(rec_trans_vel, tar_trans_vel, reduction="none", beta=0.02)
                    loss_motion_vel_trans = loss_motion_vel_trans * lambda_vel_trans * expand_dims_like(loss_motion_weighting, loss_motion_vel_trans)
                    geometric_loss["translation_vel_loss"] = th.mean(loss_motion_vel_trans)
                if lambda_acc_trans > 0:
                    rec_trans_acc = ACC(motion_pred_trans)
                    tar_trans_acc = ACC(motion_tar_trans)
                    if self.loss_type == LossType.EMAGE_MSE or self.loss_type == LossType.RESCALED_EMAGE_MSE:
                        loss_motion_acc_trans = F.mse_loss(rec_trans_acc, tar_trans_acc, reduction="none")
                    else:
                        loss_motion_acc_trans = F.smooth_l1_loss(rec_trans_acc, tar_trans_acc, reduction="none", beta=0.02)
                    loss_motion_acc_trans = loss_motion_acc_trans * lambda_acc_trans * expand_dims_like(loss_motion_weighting, loss_motion_acc_trans)
                    geometric_loss["translation_acc_loss"] = th.mean(loss_motion_acc_trans)
            
            
            lambda_vel_rcxyz = self.other_lambdas.get("lambda_vel_rcxyz", 1.0)
            lambda_acc_rcxyz = self.other_lambdas.get("lambda_acc_rcxyz", 1.0)
            lambda_rcxyz = self.other_lambdas.get("lambda_rcxyz", 1.0)
            lambda_struct_loss = self.other_lambdas.get("lambda_struct_loss", 1.0)
            
            if lambda_vel_rcxyz > 0 or lambda_rcxyz > 0 or lambda_acc_rcxyz > 0 or lambda_struct_loss > 0:
                bs = motion_tar_rot6d.shape[0]
                n = motion_tar_rot6d.shape[1]
                vertices_rec = kwargs["forward_kinematics_func"](
                    motion=motion_pred_rot6d, trans=motion_pred_trans)
                vertices_tar = kwargs["forward_kinematics_func"](
                    motion=motion_tar_rot6d, trans=motion_tar_trans)
                # 使用几何损失专用的权重，避免 logvar 加权导致的数值不稳定
                fk_loss_gate = self._build_fk_loss_gate(kwargs.get("timestep", None), vertices_rec)
                _geometric_loss_weighting = expand_dims_like(geometric_loss_weighting, vertices_rec) * fk_loss_gate
                if lambda_rcxyz > 0:
                    if self.loss_type == LossType.EMAGE_MSE or self.loss_type == LossType.RESCALED_EMAGE_MSE:
                        loss_motion_rcxyz = F.mse_loss(vertices_rec, vertices_tar, reduction="none")
                    else:
                        loss_motion_rcxyz = F.smooth_l1_loss(vertices_rec, vertices_tar, reduction="none", beta=0.02)
                    loss_motion_rcxyz = loss_motion_rcxyz * lambda_rcxyz * _geometric_loss_weighting
                    geometric_loss["motion_rcxyz"] = th.mean(loss_motion_rcxyz)
                   
                if lambda_vel_rcxyz > 0:
                    vertices_rec_vel = VEL(vertices_rec)
                    vertices_tar_vel = VEL(vertices_tar)
                    if self.loss_type == LossType.EMAGE_MSE or self.loss_type == LossType.RESCALED_EMAGE_MSE:
                        loss_motion_vel_rcxyz = F.mse_loss(vertices_rec_vel, vertices_tar_vel, reduction="none")
                    else:
                        loss_motion_vel_rcxyz = F.smooth_l1_loss(vertices_rec_vel, vertices_tar_vel, reduction="none", beta=0.02)
                    loss_motion_vel_rcxyz = loss_motion_vel_rcxyz * lambda_vel_rcxyz * _geometric_loss_weighting
                    geometric_loss["motion_vel_rcxyz"] = th.mean(loss_motion_vel_rcxyz)
                            
                if lambda_acc_rcxyz > 0:
                    vertices_rec_acc = ACC(vertices_rec)
                    vertices_tar_acc = ACC(vertices_tar)
                    if self.loss_type == LossType.EMAGE_MSE or self.loss_type == LossType.RESCALED_EMAGE_MSE:
                        loss_motion_acc_rcxyz = F.mse_loss(vertices_rec_acc, vertices_tar_acc, reduction="none")
                    else:
                        loss_motion_acc_rcxyz = F.smooth_l1_loss(vertices_rec_acc, vertices_tar_acc, reduction="none", beta=0.02)
                    loss_motion_acc_rcxyz = loss_motion_acc_rcxyz * lambda_acc_rcxyz * _geometric_loss_weighting
                    geometric_loss["motion_acc_rcxyz"] = th.mean(loss_motion_acc_rcxyz)
                
                if lambda_struct_loss > 0:
                    eps = 1e-6
                    # Use cached tensor to avoid repeated numpy array access and CPU-GPU transfer
                    smplx_parents_tensor = self._get_smplx_parents_tensor(vertices_tar.device)
                    joint_bone_vec = (vertices_tar[:,:,:self.joints] - vertices_tar[:, :, smplx_parents_tensor])[:,:,1:]
                    joint_bone_vec = joint_bone_vec.reshape(-1, self.joints - 1, 3)
                    normalized_target = joint_bone_vec / (th.norm(joint_bone_vec, dim=-1, keepdim=True) + eps)
                    target_mat = th.bmm(normalized_target, normalized_target.transpose(-1, -2))
                    
                    pred_joint_bone_vec = (vertices_rec[:,:,:self.joints] - vertices_rec[:, :, smplx_parents_tensor])[:,:,1:]
                    pred_joint_bone_vec = pred_joint_bone_vec.reshape(-1, self.joints - 1, 3)
                    normalized_pred = pred_joint_bone_vec / (th.norm(pred_joint_bone_vec, dim=-1, keepdim=True) + eps)
                    pred_mat = th.bmm(normalized_pred, normalized_pred.transpose(-1, -2))
                    if self.loss_type == LossType.EMAGE_MSE or self.loss_type == LossType.RESCALED_EMAGE_MSE:
                        struct_loss = F.mse_loss(pred_mat, target_mat, reduction="none").view(bs, n, self.joints - 1, self.joints - 1)
                    else:
                        struct_loss = th.abs(pred_mat - target_mat).view(bs, n, self.joints - 1, self.joints - 1)
                    # Use properly shaped weighting for struct_loss to avoid memory explosion from broadcasting
                    _struct_loss_weighting = expand_dims_like(geometric_loss_weighting, struct_loss)
                    struct_loss = struct_loss * lambda_struct_loss * _struct_loss_weighting
                    geometric_loss["structural_loss"] = th.mean(struct_loss)

            lambda_latent_loss = self.other_lambdas.get("lambda_latent_loss", 1.0)
            if lambda_latent_loss > 0:
                pred_mu2, pred_logvar2 = self.vae2.encode(motion_output).chunk(2, dim=-1)
                pred_mu3, pred_logvar3 = self.vae3.encode(motion_output).chunk(2, dim=-1)
                gt_mu2, gt_logvar2 = self.vae2.encode(motion_target).chunk(2, dim=-1)
                gt_mu3, gt_logvar3 = self.vae3.encode(motion_target).chunk(2, dim=-1)
                if self.loss_type == LossType.EMAGE_MSE or self.loss_type == LossType.RESCALED_EMAGE_MSE:
                    latent_loss2 = F.mse_loss(pred_mu2, gt_mu2, reduction="none")
                    latent_loss3 = F.mse_loss(pred_mu3, gt_mu3, reduction="none")
                else:
                    latent_loss2 = F.smooth_l1_loss(pred_mu2, gt_mu2, beta=0.02, reduction="none")
                    latent_loss3 = F.smooth_l1_loss(pred_mu3, gt_mu3, beta=0.02, reduction="none")
                latent_loss2 = latent_loss2 * lambda_latent_loss * expand_dims_like(loss_motion_weighting, latent_loss2)
                latent_loss3 = latent_loss3 * lambda_latent_loss * expand_dims_like(loss_motion_weighting, latent_loss3)
                geometric_loss["latent_loss"] = th.mean(latent_loss2) + th.mean(latent_loss3)
            
            prompt_loss = {}
            lambda_motion_prompt = self.other_lambdas.get("lambda_motion_prompt", 1.0)
            lambda_trans_prompt = self.other_lambdas.get("lambda_trans_prompt", 1.0)
            if lambda_motion_prompt > 0 or lambda_trans_prompt > 0:
                if lambda_motion_prompt > 0:
                    # pose_prompt_mask = kwargs.get("pose_prompt_mask", None)
                    # pose_prompt_loss = recon_loss_with_mask(motion_pred_rot6d, motion_tar_rot6d, pose_prompt_mask) * lambda_motion_prompt
                    # prompt_loss["motion_prompt_loss"] = pose_prompt_loss
                    seed_motion_pred = motion_pred_rot6d[:, :self.overlap_length]
                    seed_motion_gt = motion_tar_rot6d[:, :self.overlap_length]
                    seed_motion_pred_r = rc.rotation_6d_to_matrix_torch(seed_motion_pred.reshape((-1, 6)))
                    seed_motion_gt_r = rc.rotation_6d_to_matrix_torch(seed_motion_gt.reshape((-1, 6)))
                    seed_rotation_loss = geodesic_loss(seed_motion_gt_r, seed_motion_pred_r, reduction="none").view(bs, self.overlap_length, -1)
                    seed_rotation_loss = seed_rotation_loss * lambda_motion_prompt * expand_dims_like(loss_motion_weighting, seed_rotation_loss)
                    pose_prompt_loss = th.mean(seed_rotation_loss)
                    prompt_loss["motion_prompt_loss"] = pose_prompt_loss
                
                if lambda_trans_prompt > 0:
                    # trans_prompt_mask = kwargs.get("trans_prompt_mask", None)
                    # den = (~trans_prompt_mask).sum() + 1e-6 # 规避除以0
                    # if self.loss_type == LossType.EMAGE_MSE or self.loss_type == LossType.RESCALED_EMAGE_MSE:
                    #     num = (th.square(motion_pred_trans - motion_tar_trans).sum(dim=-1) * (~trans_prompt_mask)).sum()
                    # else:
                    #     num = (th.abs(motion_pred_trans - motion_tar_trans).sum(dim=-1) * (~trans_prompt_mask)).sum()
                    # trans_prompt_loss = (num / den) * lambda_trans_prompt
                    seed_trans_pred = motion_pred_trans[:, :self.overlap_length]
                    seed_trans_gt = motion_tar_trans[:, :self.overlap_length]
                    if self.loss_type == LossType.EMAGE_MSE or self.loss_type == LossType.RESCALED_EMAGE_MSE:
                        seed_trans_loss = F.mse_loss(seed_trans_pred, seed_trans_gt, reduction="none")
                    else:
                        seed_trans_loss = F.l1_loss(seed_trans_pred, seed_trans_gt, reduction="none")
                    seed_trans_loss = seed_trans_loss * lambda_trans_prompt * expand_dims_like(loss_motion_weighting, seed_trans_loss)
                    trans_prompt_loss = th.mean(seed_trans_loss)
                    prompt_loss["trans_prompt_loss"] = trans_prompt_loss

            # Block boundary overlap loss for reducing discontinuities at block junctions
            block_boundary_loss = {}
            lambda_block_boundary = self.other_lambdas.get("lambda_block_boundary", 0.0)
            lambda_block_boundary_vel = self.other_lambdas.get("lambda_block_boundary_vel", 1.0)
            lambda_block_boundary_acc = self.other_lambdas.get("lambda_block_boundary_acc", 1.0)
            if lambda_block_boundary > 0:
                # Get block size from kwargs (num_frame_per_block from training config)
                num_frame_per_block = kwargs.get("num_frame_per_block", None)
                if num_frame_per_block is not None and num_frame_per_block > 0:
                    # Number of overlapping frames at block boundaries (default to 4)
                    boundary_overlap_frames = kwargs.get("boundary_overlap_frames", 4)
                    seq_len = motion_pred_rot6d.shape[1]

                    # Only compute if sequence is longer than one block
                    if seq_len > num_frame_per_block:
                        # Compute number of complete blocks
                        num_blocks = seq_len // num_frame_per_block

                        # Collect boundary regions for loss computation
                        boundary_pred_rot6d_list = []
                        boundary_tar_rot6d_list = []
                        boundary_pred_trans_list = []
                        boundary_tar_trans_list = []

                        for i in range(num_blocks - 1):
                            # Boundary position between block i and block i+1
                            boundary_pos = (i + 1) * num_frame_per_block

                            # Extract last boundary_overlap_frames from block i
                            start_idx = max(0, boundary_pos - boundary_overlap_frames)
                            # Extract first boundary_overlap_frames from block i+1
                            end_idx = min(seq_len, boundary_pos + boundary_overlap_frames)

                            # Extract boundary region
                            boundary_pred_rot6d_list.append(motion_pred_rot6d[:, start_idx:end_idx])
                            boundary_tar_rot6d_list.append(motion_tar_rot6d[:, start_idx:end_idx])
                            boundary_pred_trans_list.append(motion_pred_trans[:, start_idx:end_idx])
                            boundary_tar_trans_list.append(motion_tar_trans[:, start_idx:end_idx])

                        if len(boundary_pred_rot6d_list) > 0:
                            # Concatenate all boundary regions
                            boundary_pred_rot6d = th.cat(boundary_pred_rot6d_list, dim=1)
                            boundary_tar_rot6d = th.cat(boundary_tar_rot6d_list, dim=1)
                            boundary_pred_trans = th.cat(boundary_pred_trans_list, dim=1)
                            boundary_tar_trans = th.cat(boundary_tar_trans_list, dim=1)

                            # Compute rot6d loss on boundary regions in representation space
                            if self.loss_type == LossType.EMAGE_MSE or self.loss_type == LossType.RESCALED_EMAGE_MSE:
                                boundary_rotation_loss = F.mse_loss(
                                    boundary_pred_rot6d,
                                    boundary_tar_rot6d,
                                    reduction="mean",
                                )
                            else:
                                boundary_rotation_loss = F.l1_loss(
                                    boundary_pred_rot6d,
                                    boundary_tar_rot6d,
                                    reduction="mean",
                                )
                            boundary_rotation_loss = boundary_rotation_loss * lambda_block_boundary
                            block_boundary_loss["block_boundary_rotation_loss"] = boundary_rotation_loss

                            # Compute translation loss on boundary regions
                            if self.loss_type == LossType.EMAGE_MSE or self.loss_type == LossType.RESCALED_EMAGE_MSE:
                                boundary_trans_loss = F.mse_loss(boundary_pred_trans, boundary_tar_trans, reduction="mean")
                            else:
                                boundary_trans_loss = F.l1_loss(boundary_pred_trans, boundary_tar_trans, reduction="mean")
                            boundary_trans_loss = boundary_trans_loss * lambda_block_boundary
                            block_boundary_loss["block_boundary_trans_loss"] = boundary_trans_loss

                            # Compute velocity / acceleration loss on boundary regions
                            # using FK vertex positions (reuse vertices_rec/vertices_tar computed above)
                            if vertices_rec is not None and vertices_tar is not None:
                                # Slice boundary regions from the full FK results
                                boundary_vertices_pred_list = []
                                boundary_vertices_tar_list = []
                                for i in range(num_blocks - 1):
                                    boundary_pos = (i + 1) * num_frame_per_block
                                    start_idx = max(0, boundary_pos - boundary_overlap_frames)
                                    end_idx = min(seq_len, boundary_pos + boundary_overlap_frames)
                                    boundary_vertices_pred_list.append(vertices_rec[:, start_idx:end_idx])
                                    boundary_vertices_tar_list.append(vertices_tar[:, start_idx:end_idx])

                                boundary_vertices_pred = th.cat(boundary_vertices_pred_list, dim=1)
                                boundary_vertices_tar = th.cat(boundary_vertices_tar_list, dim=1)

                                if boundary_vertices_pred.shape[1] >= 2:
                                    boundary_pred_vel = VEL(boundary_vertices_pred)
                                    boundary_tar_vel = VEL(boundary_vertices_tar)
                                    if self.loss_type == LossType.EMAGE_MSE or self.loss_type == LossType.RESCALED_EMAGE_MSE:
                                        boundary_vel_loss = F.mse_loss(
                                            boundary_pred_vel,
                                            boundary_tar_vel,
                                            reduction="mean",
                                        )
                                    else:
                                        boundary_vel_loss = F.l1_loss(
                                            boundary_pred_vel,
                                            boundary_tar_vel,
                                            reduction="mean",
                                        )
                                    boundary_vel_loss = boundary_vel_loss * lambda_block_boundary * lambda_block_boundary_vel
                                    block_boundary_loss["block_boundary_vel_loss"] = boundary_vel_loss

                                if boundary_vertices_pred.shape[1] >= 3:
                                    boundary_pred_acc = ACC(boundary_vertices_pred)
                                    boundary_tar_acc = ACC(boundary_vertices_tar)
                                    if self.loss_type == LossType.EMAGE_MSE or self.loss_type == LossType.RESCALED_EMAGE_MSE:
                                        boundary_acc_loss = F.mse_loss(
                                            boundary_pred_acc,
                                            boundary_tar_acc,
                                            reduction="mean",
                                        )
                                    else:
                                        boundary_acc_loss = F.l1_loss(
                                            boundary_pred_acc,
                                            boundary_tar_acc,
                                            reduction="mean",
                                        )
                                    boundary_acc_loss = boundary_acc_loss * lambda_block_boundary * lambda_block_boundary_acc
                                    block_boundary_loss["block_boundary_acc_loss"] = boundary_acc_loss

            return {
                **main_loss,
                **geometric_loss,
                **prompt_loss,
                **block_boundary_loss,
            }
        else:
            raise NotImplementedError(self.loss_type)
    
    def training_losses(
        self,
        model,
        motion_start,
        timestep,
        model_kwargs=None,
        motion_noise=None,
        skip_noise=False,
        **kwargs
    ):
        if model_kwargs is None:
            model_kwargs = {}
        if skip_noise:
            motion_t = motion_start
        else:
            if motion_noise is None:
                motion_noise = th.randn_like(motion_start)
            motion_t = self.q_sample(motion_start, t=timestep, noise=motion_noise)
        
        terms = {}
        
        if self.loss_type == LossType.KL or self.loss_type == LossType.RESCALED_KL:
            terms["loss"] = self._vb_terms_bpd(
                model=model,
                motion_start=motion_start,
                motion_t=motion_t,
                timestep=timestep,
                clip_denoised=False,
                model_kwargs=model_kwargs,
            )["output"]
            if self.loss_type == LossType.RESCALED_KL:
                terms["loss"] *= self.num_timesteps
        else:
            model_output = model(motion_t, timestep, model_kwargs)
            
            motion_output = model_output.pop("motion", None)
            extra = model_output

            # 可选：来自 trainer 的逐帧监督权重（BLD block-level supervision shaping）
            frame_supervision_weight = kwargs.get("frame_supervision_weight", None)
            if frame_supervision_weight is not None:
                extra = dict(extra) if extra is not None else {}
                extra["frame_supervision_weight"] = frame_supervision_weight

            if self.return_startx and self.model_mean_type == ModelMeanType.EPSILON:
                # FOR motion
                B, N, C = motion_t.shape
                assert motion_output.shape == (B, N, C * 2)
                motion_output, motion_var_values = th.split(motion_output, C, dim=1)
                pred_motion_start = self._predict_xstart_from_eps(x_t=motion_t, t=timestep, eps=motion_output)
                # RETURN
                # _output后缀的是noise，模型直接输出noise
                return motion_output, pred_motion_start, motion_t

            if self.model_var_type in [
                ModelVarType.LEARNED,
                ModelVarType.LEARNED_RANGE,
            ]:
                # FOR motion
                B, N, C = motion_t.shape
                assert motion_output.shape == (B, N, C * 2)
                motion_output, motion_var_values = th.split(motion_output, C, dim=1)

                #################################################################
                # Learn the variance using the variational bound, but don't let it affect our mean prediction.
                motion_frozen_out = th.cat([motion_output.detach(), motion_var_values], dim=1)
                terms["vb"] = self._vb_terms_bpd(                    
                    model=lambda *args, r1=motion_frozen_out, **kwargs: {
                        "motion": r1,
                    },
                    motion_start=motion_start,
                    motion_t=motion_t,
                    timestep=timestep,
                    clip_denoised=False,
                    model_kwargs=model_kwargs,
                )["output"]
                if self.loss_type == LossType.RESCALED_MSE \
                    or self.loss_type == LossType.RESCALED_MOT_ANGLEDIS_EXP_MSE \
                    or self.loss_type == LossType.RESCALED_MOT_GEO_EXP_MSE \
                    or self.loss_type == LossType.RESCALED_MAE \
                    or self.loss_type == LossType.RESCALED_EMAGE \
                    or self.loss_type == LossType.RESCALED_EMAGE_MSE:
                    # Divide by 1000 for equivalence with initial implementation.
                    # Without a factor of 1/1000, the VB term hurts the MSE term.
                    terms["vb"] *= self.num_timesteps / 1000.0

            # FOR motion
            motion_target = {
                ModelMeanType.PREVIOUS_X: 
                    self.q_posterior_mean_variance(x_start=motion_start, x_t=motion_t, t=timestep)[0],
                ModelMeanType.START_X: motion_start,
                ModelMeanType.EPSILON: motion_noise,
                ModelMeanType.FLOW_VELOCITY: motion_noise - motion_start,
            }[self.model_mean_type]
            #################################################################
            #################################################################
            assert motion_output.shape == motion_target.shape == motion_start.shape
            #################################################################
            # FOR loss calculation ##########################################
            # 可选：按 block reshape，再计算损失（B, T, C） -> (B * num_blocks, block_size, C）
            if self.blockwise_loss:
                assert extra is None or "logvar" not in extra, (
                    "blockwise_loss=True and return_logvar=True cannot be used together. "
                    "logvar has shape (B, 1, 1) for scalar timestep or (B, num_blocks, 1) for block-level "
                    "timestep, neither of which can be correctly reshaped by _reshape_blockwise to "
                    "(B * num_blocks, block_size, 1). Disable either blockwise_loss or return_logvar."
                )
                block_size = self.loss_block_size or motion_target.shape[1]
                motion_target = self._reshape_blockwise(motion_target, block_size)
                motion_output = self._reshape_blockwise(motion_output, block_size)
                motion_start = self._reshape_blockwise(motion_start, block_size)
                motion_noise = self._reshape_blockwise(motion_noise, block_size)
                timestep = self._reshape_blockwise(timestep, block_size)
                if extra is not None and "logvar" in extra:
                    extra = dict(extra)
                    extra["logvar"] = self._reshape_blockwise(extra["logvar"], block_size)
                if extra is not None and "frame_supervision_weight" in extra:
                    extra = dict(extra)
                    extra["frame_supervision_weight"] = self._reshape_blockwise(extra["frame_supervision_weight"], block_size)

            loss_motion_weighting, geometric_loss_weighting = self.loss_weighting(timestep, extra)
            loss_motion_weighting = expand_dims_like(loss_motion_weighting, motion_target)
            geometric_loss_weighting = expand_dims_like(geometric_loss_weighting, motion_target)
            
            loss_items = self.loss_function(
                motion_target, motion_output, 
                loss_motion_weighting,
                geometric_loss_weighting,
                extra,                                                   
                timestep=timestep,
                **kwargs
            )
            terms["loss"] = sum(loss_items.values())
            for k, v in loss_items.items():
                if k != "loss":
                    terms[k] = v
            if extra is not None and "logvar" in extra:
                # 正则项必须使用 clamp 后的 logvar，与 loss_weighting 中的 clamp 保持一致。
                # 否则当原始 logvar 超出 clamp 范围时，MSE 项的梯度被截断但正则项梯度不截断，
                # 导致梯度信号不一致，logvar 在 clamp 边界震荡，训练崩溃。
                if self.logvar_clamp_min is not None and str(self.logvar_clamp_min).lower() != "none" \
                   and self.logvar_clamp_max is not None and str(self.logvar_clamp_max).lower() != "none":
                    logvar_clamped = th.clamp(
                        extra["logvar"],
                        min=float(self.logvar_clamp_min),
                        max=float(self.logvar_clamp_max),
                    )
                else:
                    logvar_clamped = extra["logvar"]

                terms["logvar"] = logvar_clamped.mean()
                terms["loss"] += terms["logvar"]
        return terms

    def training_losses_diffusers(
        self, 
        model, 
        motion_start, 
        expression, 
        timestep, 
        model_kwargs=None, 
        motion_noise=None, 
        skip_noise=False
    ):
        """
        Compute training losses for a single timestep.
        :param model: the model to evaluate loss on.
        :param x_start: the [N x C x ...] tensor of inputs.
        :param t: a batch of timestep indices.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :param noise: if specified, the specific Gaussian noise to try to remove.
        :return: a dict with the key "loss" containing a tensor of shape [N].
                 Some mean or variance settings may also have other keys.
        """
        return self.training_losses(
            model, motion_start, expression, timestep, model_kwargs, motion_noise, skip_noise
        )

    def p_mean_variance(
        self, 
        model, 
        motion_t,
        timestep,
        clip_denoised=True, 
        denoised_fn=None, 
        model_kwargs=None
    ):
        """
        Apply the model to get p(x_{t-1} | x_t), as well as a prediction of
        the initial x, x_0.
        :param model: the model, which takes a signal and a batch of timesteps
                      as input.
        :param x: the [N x C x ...] tensor at time t.
        :param t: a 1-D Tensor of timesteps.
        :param clip_denoised: if True, clip the denoised signal into [-1, 1].
        :param denoised_fn: if not None, a function which applies to the
            x_start prediction before it is used to sample. Applies before
            clip_denoised.
        :param model_kwargs: if not None, a dict of extra keyword arguments to
            pass to the model. This can be used for conditioning.
        :return: a dict with the following keys:
                 - 'mean': the model mean output.
                 - 'variance': the model variance output.
                 - 'log_variance': the log of 'variance'.
                 - 'pred_xstart': the prediction for x_0.
        """
        if model_kwargs is None:
            model_kwargs = {}

        B, mN, mC = motion_t.shape
        assert timestep.shape == (B,) 
        model_output = model(
            motion_t,
            timestep,
            model_kwargs
        )
        
        motion_output = model_output.pop("motion", None)
        extra = model_output

        if self.model_var_type in [ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE]:
            assert motion_output.shape == (B, mN, mC * 2)
            motion_output, motion_var_values = th.split(motion_output, mC, dim=-1)
            
            min_log = _extract_into_tensor(self.posterior_log_variance_clipped, timestep, motion_output.shape)
            max_log = _extract_into_tensor(np.log(self.betas), timestep, motion_output.shape)
            
            # The model_var_values is [-1, 1] for [min_var, max_var].
            motion_frac = (motion_var_values + 1) / 2
            motion_log_variance = motion_frac * max_log + (1 - motion_frac) * min_log
            motion_variance = th.exp(motion_log_variance)
        elif self.model_var_type in [ModelVarType.FIXED_LARGE, ModelVarType.FIXED_SMALL]:
            model_variance, model_log_variance = {
                # for fixedlarge, we set the initial (log-)variance like so
                # to get a better decoder log likelihood.
                ModelVarType.FIXED_LARGE: (
                    np.append(self.posterior_variance[1], self.betas[1:]),
                    np.log(np.append(self.posterior_variance[1], self.betas[1:])),
                ),
                ModelVarType.FIXED_SMALL: (
                    self.posterior_variance,
                    self.posterior_log_variance_clipped,
                ),
            }[self.model_var_type]
            
            motion_variance = _extract_into_tensor(model_variance, timestep, motion_output.shape)
            motion_log_variance = _extract_into_tensor(model_log_variance, timestep, motion_output.shape)
        else:
            motion_variance = th.zeros_like(motion_output)
            motion_log_variance = th.zeros_like(motion_output)
            

        def process_xstart(x):
            if denoised_fn is not None:
                x = denoised_fn(x)
            if clip_denoised:
                return x.clamp(-1, 1)
            return x

        if self.model_mean_type == ModelMeanType.START_X:
            pred_motion_start = process_xstart(motion_output)
        else:
            pred_motion_start = process_xstart(
                self._predict_xstart_from_eps(x_t=motion_t, t=timestep, eps=motion_output))
        motion_mean, _, _ = self.q_posterior_mean_variance(x_start=pred_motion_start, x_t=motion_t, t=timestep)
        
        assert motion_mean.shape == motion_log_variance.shape == pred_motion_start.shape == motion_t.shape
        return {
            "motion_mean": motion_mean,
            "motion_variance": motion_variance,
            "motion_log_variance": motion_log_variance,
            "pred_motion_start": pred_motion_start,
            "extra": extra,
        }
    
    def _vb_terms_bpd(
        self, 
        model, 
        motion_start, 
        motion_t, 
        timestep,
        clip_denoised=True, 
        model_kwargs=None
    ):
        """
        Get a term for the variational lower-bound.
        The resulting units are bits (rather than nats, as one might expect).
        This allows for comparison to other papers.
        :return: a dict with the following keys:
                 - 'output': a shape [N] tensor of NLLs or KLs.
                 - 'pred_xstart': the x_0 predictions.
        """
        # 这个vb的损失是不适合expression的，因为它不是生成式模型生成的
        motion_true_mean, _, motion_true_log_variance_clipped = \
            self.q_posterior_mean_variance(x_start=motion_start, x_t=motion_t, t=timestep)
        out = self.p_mean_variance(
            model, motion_t=motion_t,
            timestep=timestep, 
            clip_denoised=clip_denoised, model_kwargs=model_kwargs
        )
        motion_kl = normal_kl(motion_true_mean, motion_true_log_variance_clipped, out["motion_mean"], out["motion_log_variance"])
        motion_kl = mean_flat(motion_kl) / np.log(2.0)

        motion_decoder_nll = -discretized_gaussian_log_likelihood(
            motion_start, means=out["motion_mean"], log_scales=0.5 * out["motion_log_variance"]
        )
        assert motion_decoder_nll.shape == motion_start.shape
        motion_decoder_nll = mean_flat(motion_decoder_nll) / np.log(2.0)
        # At the first timestep return the decoder NLL,
        # otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
        motion_loss = th.where((timestep == 0), motion_decoder_nll, motion_kl)
        return {
            "output": motion_loss, 
            "pred_motion_start": out["pred_motion_start"],
        }
