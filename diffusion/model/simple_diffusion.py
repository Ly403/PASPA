import torch as th
import torch.nn.functional as F
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
    _extract_into_tensor
)
from diffusion.utils import rotation_conversions as rc
from diffusion.data.datasets.beat2.data_tools import joints_list
from typing import Dict, Union
from omegaconf import DictConfig

class SimpleDiffusion(SpacedDiffusion):
    """
        设想是支持text2motion+expression、audio2motion+expression、uncondition2motion+expression、
        text2motion、audio2motion、uncondition2motion的task，后面这三个task是给没有expression的task预备的
        只管一个timestep的输入，motion和expression共用一个timestep
        属于一种简化版本的omniflow diffusion
    """
    def __init__(
        self, 
        task="audio2motion+expression", 
        joints=55,
        lambda_motion=1.0,
        lambda_expression=1.0,
        other_lambdas: Union[DictConfig, Dict]=None, # 有些损失函数，涉及更多的参数调节，这里直接放一个字典，增强灵活性
        *args, 
        **kwargs
    ):
        self.task = task
        self.joints = joints
        self.lambda_motion = lambda_motion
        self.lambda_expression = lambda_expression
        self.other_lambdas = other_lambdas
        super().__init__(*args, **kwargs)
    
    def loss_weighting(self, timestep):
        # 现在先支持均匀的加权，并且把多task的逻辑放到这里来，避免对主逻辑的侵入性修改
        # TODO: 支持其他类型的加权方式
        if self.task.endswith("2motion"):
            loss_motion_weighting = th.ones_like(timestep)
            with th.no_grad():
                loss_expression_weighting = th.zeros_like(timestep)
        elif self.task.endswith("2motion+expression"):
            loss_motion_weighting = th.ones_like(timestep)
            loss_expression_weighting = th.ones_like(timestep)
        else:
            raise NotImplementedError(self.task)
        loss_motion_weighting = loss_motion_weighting * self.lambda_motion
        loss_expression_weighting = loss_expression_weighting * self.lambda_expression
        return loss_motion_weighting, loss_expression_weighting
    
    def loss_function(
        self, 
        motion_target, 
        motion_output, 
        loss_motion_weighting,
        expression_target, 
        expression_output, 
        loss_expression_weighting,
        **kwargs
    ):
        if self.loss_type == LossType.MSE or self.loss_type == LossType.RESCALED_MSE:
            # 这是L2损失
            loss_motion = (motion_target - motion_output) ** 2 * loss_motion_weighting
            loss_expression = (expression_target - expression_output) ** 2 * loss_expression_weighting
            return {
                "mse_motion": mean_flat(loss_motion),
                "mse_expression": mean_flat(loss_expression),
            }
        elif self.loss_type == LossType.MAE or self.loss_type == LossType.RESCALED_MAE:
            # 这是L1损失
            loss_motion = th.abs(motion_target - motion_output) * loss_motion_weighting
            loss_expression = th.abs(expression_target - expression_output) * loss_expression_weighting
            return {
                "mae_motion": mean_flat(loss_motion),
                "mae_expression": mean_flat(loss_expression),
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
            loss_expression = (expression_target - expression_output) ** 2 * loss_expression_weighting
            return {
                "motion_angle_dis": mean_flat(loss_motion),
                "mse_motion_others": mean_flat(loss_motion_others),
                "mse_expression": mean_flat(loss_expression),
            }
        elif self.loss_type == LossType.MOT_GEO_EXP_MSE or self.loss_type == LossType.RESCALED_MOT_GEO_EXP_MSE:
            # 这是motion geometric loss，几何损失，用motion diffusion models里面提出的几个损失的计算方法
            loss_motion_mse = (motion_target - motion_output) ** 2 * loss_motion_weighting # mse保留
            loss_expression = (expression_target - expression_output) ** 2 * loss_expression_weighting
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
                loss_motion_vel = (motion_target_rot6d_vel - motion_output_rot6d_vel) ** 2 * lambda_vel * loss_motion_weighting
                geometric_loss["mse_motion_vel"] = mean_flat(loss_motion_vel)

            if lambda_vel_rcxyz > 0 or lambda_fc > 0 or lambda_rcxyz > 0:
                bs = motion_target_rot6d.shape[0]
                n = motion_target_rot6d.shape[1]
                rec_pose = rc.rotation_6d_to_axis_angle(
                    motion_output_rot6d.reshape(bs, n, -1, 6)
                ).reshape(bs*n, -1)
                rec_trans = motion_output_others[:, :, -3:]
                rec_exps = expression_output
                tar_pose = rc.rotation_6d_to_axis_angle(
                    motion_target_rot6d.reshape(bs, n, -1, 6)
                ).reshape(bs*n, -1)
                tar_trans = motion_target_others[:, :, -3:]
                tar_exps = expression_target
                
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
                loss_motion_weighting = expand_dims_like(loss_motion_weighting, vertices_rec)
                if lambda_rcxyz > 0:
                    loss_motion_rcxyz = (vertices_rec - vertices_tar) ** 2 * lambda_rcxyz * loss_motion_weighting
                    geometric_loss["mse_motion_rcxyz"] = mean_flat(loss_motion_rcxyz)
                                    
                if lambda_vel_rcxyz > 0:
                    vertices_rec_vel = vertices_rec[:, 1:, ...] - vertices_rec[:, :-1, ...]
                    vertices_tar_vel = vertices_tar[:, 1:, ...] - vertices_tar[:, :-1, ...]
                    loss_motion_vel_rcxyz = (vertices_rec_vel - vertices_tar_vel) ** 2 * lambda_vel_rcxyz * loss_motion_weighting
                    geometric_loss["mse_motion_vel_rcxyz"] = mean_flat(loss_motion_vel_rcxyz)
                                    
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
                    geometric_loss["mse_motion_fc_vel"] = mean_flat(loss_motion_fc_vel)
                
            return {
                "mse_motion": mean_flat(loss_motion_mse),
                "mse_expression": mean_flat(loss_expression),
                **geometric_loss,
            }
        else:
            raise NotImplementedError(self.loss_type)
    
    def training_losses(
        self,
        model,
        motion_start,
        expression_start,
        timestep,
        model_kwargs=None,
        motion_noise=None,
        expression_noise=None,
        skip_noise=False,
        **kwargs
    ):
        if model_kwargs is None:
            model_kwargs = {}
        if skip_noise:
            motion_t = motion_start
            expression_t = expression_start
        else:
            if motion_noise is None:
                motion_noise = th.randn_like(motion_start)
            if expression_noise is None:
                expression_noise = th.randn_like(expression_start)
            motion_t = self.q_sample(motion_start, t=timestep, noise=motion_noise)
            expression_t = self.q_sample(expression_start, t=timestep, noise=expression_noise)
        
        terms = {}
        
        if self.loss_type == LossType.KL or self.loss_type == LossType.RESCALED_KL:
            terms["loss"] = self._vb_terms_bpd(
                model=model,
                motion_start=motion_start,
                motion_t=motion_t,
                expression_start=expression_start,
                expression_t=expression_t,
                timestep=timestep,
                clip_denoised=False,
                model_kwargs=model_kwargs,
            )["output"]
            if self.loss_type == LossType.RESCALED_KL:
                terms["loss"] *= self.num_timesteps
        else:
            model_output = model(motion_t, expression_t, timestep, model_kwargs)
            
            motion_output = model_output.pop("motion", None)
            expression_output = model_output.pop("expression", None)
            extra = model_output

            if self.return_startx and self.model_mean_type == ModelMeanType.EPSILON:
                # FOR motion
                B, N, C = motion_t.shape
                assert motion_output.shape == (B, N, C * 2)
                motion_output, motion_var_values = th.split(motion_output, C, dim=1)
                pred_motion_start = self._predict_xstart_from_eps(x_t=motion_t, t=timestep, eps=motion_output)
                #################################################################
                # FOR expression
                B, N, C = expression_t.shape
                assert expression_output.shape == (B, N, C * 2)
                expression_output, expression_var_values = th.split(expression_output, C, dim=1)
                pred_expression_start = self._predict_xstart_from_eps(x_t=expression_t, t=timestep, eps=expression_output)
                #################################################################
                # RETURN
                # _output后缀的是noise，模型直接输出noise
                return motion_output, expression_output, pred_motion_start, pred_expression_start, motion_t, expression_t

            if self.model_var_type in [
                ModelVarType.LEARNED,
                ModelVarType.LEARNED_RANGE,
            ]:
                # FOR motion
                B, N, C = motion_t.shape
                assert motion_output.shape == (B, N, C * 2)
                motion_output, motion_var_values = th.split(motion_output, C, dim=1)
                #################################################################
                # FOR expression
                B, N, C = expression_t.shape
                assert expression_output.shape == (B, N, C * 2)
                expression_output, expression_var_values = th.split(expression_output, C, dim=1)
                #################################################################
                # Learn the variance using the variational bound, but don't let it affect our mean prediction.
                motion_frozen_out = th.cat([motion_output.detach(), motion_var_values], dim=1)
                expression_frozen_out = th.cat([expression_output.detach(), expression_var_values], dim=1)
                terms["vb"] = self._vb_terms_bpd(                    
                    model=lambda *args, r1=motion_frozen_out, r2=expression_frozen_out, **kwargs: {
                        "motion": r1,
                        "expression": r2,
                    },
                    motion_start=motion_start,
                    motion_t=motion_t,
                    expression_start=expression_start,
                    expression_t=expression_t,
                    timestep=timestep,
                    clip_denoised=False,
                    model_kwargs=model_kwargs,
                )["output"]
                if self.loss_type == LossType.RESCALED_MSE \
                    or self.loss_type == LossType.RESCALED_MOT_ANGLEDIS_EXP_MSE \
                    or self.loss_type == LossType.RESCALED_MOT_GEO_EXP_MSE \
                    or self.loss_type == LossType.RESCALED_MAE:
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
            # FOR expression
            expression_target = {
                ModelMeanType.PREVIOUS_X: 
                    self.q_posterior_mean_variance(x_start=expression_start, x_t=expression_t, t=timestep)[0],
                ModelMeanType.START_X: expression_start,
                ModelMeanType.EPSILON: expression_noise,
                ModelMeanType.FLOW_VELOCITY: expression_noise - expression_start,
            }[self.model_mean_type]
            #################################################################
            #################################################################
            assert motion_output.shape == motion_target.shape == motion_start.shape
            assert expression_output.shape == expression_target.shape == expression_start.shape
            #################################################################
            # FOR loss calculation ##########################################
            loss_motion_weighting, loss_expression_weighting = self.loss_weighting(timestep)
            loss_motion_weighting = expand_dims_like(loss_motion_weighting, motion_target)
            loss_expression_weighting = expand_dims_like(loss_expression_weighting, expression_target)
            
            loss_items = self.loss_function(
                motion_target, motion_output, loss_motion_weighting,                                                       
                expression_target, expression_output, loss_expression_weighting,
                **kwargs
            )
            for k, v in loss_items.items():
                terms[k] = v
            terms["loss"] = sum(terms.values())
        return terms

    def training_losses_diffusers(
        self, 
        model, 
        motion_start, 
        expression_start, 
        timestep, 
        model_kwargs=None, 
        motion_noise=None, 
        expression_noise=None, 
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
            model, motion_start, expression_start, timestep, model_kwargs, motion_noise, expression_noise, skip_noise
        )

    def p_mean_variance(
        self, 
        model, 
        motion_t,
        expression_t, 
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

        assert motion_t.shape[0] == expression_t.shape[0], "batch size mismatch"
        B, mN, mC = motion_t.shape
        B, eN, eC = expression_t.shape
        assert timestep.shape == (B,) 
        model_output = model(
            motion_t,
            expression_t,
            timestep,
            model_kwargs
        )
        
        motion_output = model_output.pop("motion", None)
        expression_output = model_output.pop("expression", None)
        extra = model_output

        if self.model_var_type in [ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE]:
            assert motion_output.shape == (B, mN, mC * 2) and expression_output.shape == (B, eN, eC * 2)
            motion_output, motion_var_values = th.split(motion_output, mC, dim=-1)
            expression_output, expression_var_values = th.split(expression_output, eC, dim=-1)
            
            min_log = _extract_into_tensor(self.posterior_log_variance_clipped, timestep, motion_output.shape)
            max_log = _extract_into_tensor(np.log(self.betas), timestep, motion_output.shape)
            
            # The model_var_values is [-1, 1] for [min_var, max_var].
            motion_frac = (motion_var_values + 1) / 2
            expression_frac = (expression_var_values + 1) / 2
            
            motion_log_variance = motion_frac * max_log + (1 - motion_frac) * min_log
            expression_log_variance = expression_frac * max_log + (1 - expression_frac) * min_log
            
            motion_variance = th.exp(motion_log_variance)
            expression_variance = th.exp(expression_log_variance)
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
            
            expression_variance = _extract_into_tensor(model_variance, timestep, expression_output.shape)
            expression_log_variance = _extract_into_tensor(model_log_variance, timestep, expression_output.shape)
        else:
            motion_variance = th.zeros_like(motion_output)
            motion_log_variance = th.zeros_like(motion_output)
            
            expression_variance = th.zeros_like(expression_output)
            expression_log_variance = th.zeros_like(expression_output)

        def process_xstart(x):
            if denoised_fn is not None:
                x = denoised_fn(x)
            if clip_denoised:
                return x.clamp(-1, 1)
            return x

        if self.model_mean_type == ModelMeanType.START_X:
            pred_motion_start = process_xstart(motion_output)
            pred_expression_start = process_xstart(expression_output)
        else:
            pred_motion_start = process_xstart(self._predict_xstart_from_eps(x_t=motion_t, t=timestep, eps=motion_output))
            pred_expression_start = process_xstart(
                self._predict_xstart_from_eps(x_t=expression_t, t=timestep, eps=expression_output))
        motion_mean, _, _ = self.q_posterior_mean_variance(x_start=pred_motion_start, x_t=motion_t, t=timestep)
        expression_mean, _, _ = self.q_posterior_mean_variance(x_start=pred_expression_start, x_t=expression_t, t=timestep)
        
        assert motion_mean.shape == motion_log_variance.shape == pred_motion_start.shape == motion_t.shape
        assert expression_mean.shape == expression_log_variance.shape == pred_expression_start.shape == expression_t.shape
        return {
            "motion_mean": motion_mean,
            "motion_variance": motion_variance,
            "motion_log_variance": motion_log_variance,
            "pred_motion_start": pred_motion_start,
            "expression_mean": expression_mean,
            "expression_variance": expression_variance,
            "expression_log_variance": expression_log_variance,
            "pred_expression_start": pred_expression_start,
            "extra": extra,
        }
    
    def _vb_terms_bpd(
        self, 
        model, 
        motion_start, 
        motion_t, 
        expression_start,
        expression_t,
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
        motion_true_mean, _, motion_true_log_variance_clipped = \
            self.q_posterior_mean_variance(x_start=motion_start, x_t=motion_t, t=timestep)
        expression_true_mean, _, expression_true_log_variance_clipped = \
            self.q_posterior_mean_variance(x_start=expression_start, x_t=expression_t, t=timestep)
        out = self.p_mean_variance(
            model, motion_t=motion_t, expression_t=expression_t,
            timestep=timestep, 
            clip_denoised=clip_denoised, model_kwargs=model_kwargs
        )
        motion_kl = normal_kl(motion_true_mean, motion_true_log_variance_clipped, out["motion_mean"], out["motion_log_variance"])
        expression_kl = normal_kl(expression_true_mean, expression_true_log_variance_clipped, out["expression_mean"], out["expression_log_variance"])
        
        motion_kl = mean_flat(motion_kl) / np.log(2.0)
        expression_kl = mean_flat(expression_kl) / np.log(2.0)

        motion_decoder_nll = -discretized_gaussian_log_likelihood(
            motion_start, means=out["motion_mean"], log_scales=0.5 * out["motion_log_variance"]
        )
        assert motion_decoder_nll.shape == motion_start.shape
        motion_decoder_nll = mean_flat(motion_decoder_nll) / np.log(2.0)
        
        expression_decoder_nll = -discretized_gaussian_log_likelihood(
            expression_start, means=out["expression_mean"], log_scales=0.5 * out["expression_log_variance"]
        )
        assert expression_decoder_nll.shape == expression_start.shape
        expression_decoder_nll = mean_flat(expression_decoder_nll) / np.log(2.0)

        # At the first timestep return the decoder NLL,
        # otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
        motion_loss = th.where((timestep == 0), motion_decoder_nll, motion_kl)
        expression_loss = th.where((timestep == 0), expression_decoder_nll, expression_kl)
        return {
            "output": motion_loss + expression_loss, 
            "pred_motion_start": out["pred_motion_start"],
            "pred_expression_start": out["pred_expression_start"],
        }

            