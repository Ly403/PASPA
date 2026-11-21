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
class OmniFlowDiffusion(SpacedDiffusion):
    def __init__(
        self, 
        task="text2all", 
        lambda_motion=1.0,
        lambda_expression=1.0,
        lambda_audio=1.0,
        other_lambdas: Union[DictConfig, Dict]=None, # 有些损失函数，涉及更多的参数调节，这里直接放一个字典，增强灵活性
        *args, 
        **kwargs
    ):
        self.task = task
        self.lambda_motion = lambda_motion
        self.lambda_expression = lambda_expression
        self.lambda_audio = lambda_audio
        self.other_lambdas = other_lambdas
        super().__init__(*args, **kwargs)
    
    def loss_weighting(self, motion_timestep, expression_timestep, audio_timestep):
        # 现在先支持均匀的加权，并且把多task的逻辑放到这里来，避免对主逻辑的侵入性修改
        # TODO: 支持其他类型的加权方式
        if self.task.endswith("all"):
            loss_motion_weighting = th.ones_like(motion_timestep)
            loss_expression_weighting = th.ones_like(expression_timestep)
            loss_audio_weighting = th.ones_like(audio_timestep)
        elif self.task.endswith("2motion"):
            loss_motion_weighting = th.ones_like(motion_timestep)
            with th.no_grad():
                loss_expression_weighting = th.zeros_like(expression_timestep)
                loss_audio_weighting = th.zeros_like(audio_timestep)
        elif self.task.endswith("2expression"):
            loss_expression_weighting = th.ones_like(expression_timestep)
            with th.no_grad():
                loss_motion_weighting = th.zeros_like(motion_timestep)
                loss_audio_weighting = th.zeros_like(audio_timestep)
        elif self.task.endswith("2audio"):
            loss_audio_weighting = th.ones_like(audio_timestep)
            with th.no_grad():
                loss_motion_weighting = th.zeros_like(motion_timestep)
                loss_expression_weighting = th.zeros_like(expression_timestep)
        elif self.task.endswith("2motion+expression"):
            loss_motion_weighting = th.ones_like(motion_timestep)
            loss_expression_weighting = th.ones_like(expression_timestep)
            with th.no_grad():
                loss_audio_weighting = th.zeros_like(audio_timestep)
        elif self.task.endswith("2motion+audio"):
            loss_motion_weighting = th.ones_like(motion_timestep)
            with th.no_grad():
                loss_expression_weighting = th.zeros_like(expression_timestep)
            loss_audio_weighting = th.ones_like(audio_timestep)
        elif self.task.endswith("2expression+audio"):
            with th.no_grad():
                loss_motion_weighting = th.zeros_like(motion_timestep)
            loss_expression_weighting = th.ones_like(expression_timestep)
            loss_audio_weighting = th.ones_like(audio_timestep)
        else:
            raise NotImplementedError(self.task)
        loss_motion_weighting = loss_motion_weighting * self.lambda_motion
        loss_expression_weighting = loss_expression_weighting * self.lambda_expression
        loss_audio_weighting = loss_audio_weighting * self.lambda_audio
        return loss_motion_weighting, loss_expression_weighting, loss_audio_weighting
    
    def loss_function(
        self, 
        motion_target, 
        motion_output, 
        loss_motion_weighting,
        expression_target, 
        expression_output, 
        loss_expression_weighting,
        audio_target, 
        audio_output, 
        loss_audio_weighting,
        **kwargs
    ):
        if self.loss_type == LossType.MSE or self.loss_type == LossType.RESCALED_MSE:
            # 这是L2损失
            loss_motion = (motion_target - motion_output) ** 2 * loss_motion_weighting
            loss_expression = (expression_target - expression_output) ** 2 * loss_expression_weighting
            loss_audio = (audio_target - audio_output) ** 2 * loss_audio_weighting
            return {
                "mse_motion": mean_flat(loss_motion),
                "mse_expression": mean_flat(loss_expression),
                "mse_audio": mean_flat(loss_audio),
            }
        elif self.loss_type == LossType.MAE or self.loss_type == LossType.RESCALED_MAE:
            # 这是L1损失
            loss_motion = th.abs(motion_target - motion_output) * loss_motion_weighting
            loss_expression = th.abs(expression_target - expression_output) * loss_expression_weighting
            loss_audio = th.abs(audio_target - audio_output) * loss_audio_weighting
            return {
                "mae_motion": mean_flat(loss_motion),
                "mae_expression": mean_flat(loss_expression),
                "mae_audio": mean_flat(loss_audio),
            }
        elif self.loss_type == LossType.MOT_ANGLEDIS_EXP_MSE_AUDIO_MAE or self.loss_type == LossType.RESCALED_MOT_ANGLEDIS_EXP_MSE_AUDIO_MAE:
            # 这是motion angle, expression MSE, audio MAE损失
            motion_target_rot6d = motion_target[:, :, :-7]
            motion_output_rot6d = motion_output[:, :, :-7]
            motion_target_others = motion_target[:, :, -7:]
            motion_output_others = motion_output[:, :, -7:]
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
            loss_audio = (audio_target - audio_output) ** 2 * loss_audio_weighting
            return {
                "motion_angle_dis": mean_flat(loss_motion),
                "mse_motion_others": mean_flat(loss_motion_others),
                "mse_expression": mean_flat(loss_expression),
                "mse_audio": mean_flat(loss_audio),
            }
        elif self.loss_type == LossType.MOT_GEO_EXP_MSE_AUDIO_MAE or self.loss_type == LossType.RESCALED_MOT_GEO_EXP_MSE_AUDIO_MAE:
            # 这是motion geometric loss，几何损失，用motion diffusion models里面提出的几个损失的计算方法
            loss_motion_mse = (motion_target - motion_output) ** 2 * loss_motion_weighting # mse保留
            loss_expression = (expression_target - expression_output) ** 2 * loss_expression_weighting
            loss_audio = (audio_target - audio_output) ** 2 * loss_audio_weighting
            # for motion geometric loss
            motion_target_rot6d = motion_target[:, :, :-7]
            motion_output_rot6d = motion_output[:, :, :-7]
            motion_target_others = motion_target[:, :, -7:]
            motion_output_others = motion_output[:, :, -7:]
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
                "mse_audio": mean_flat(loss_audio),
                **geometric_loss,
            }
        else:
            raise NotImplementedError(self.loss_type)
    
    def training_losses(
        self,
        model,
        motion_start,
        expression_start,
        audio_start,
        motion_timestep,
        expression_timestep,
        audio_timestep,
        model_kwargs=None,
        motion_noise=None,
        expression_noise=None,
        audio_noise=None, 
        skip_noise=False,
        **kwargs
    ):
        if model_kwargs is None:
            model_kwargs = {}
        if skip_noise:
            motion_t = motion_start
            expression_t = expression_start
            audio_t = audio_start
        else:
            if motion_noise is None:
                motion_noise = th.randn_like(motion_start)
            if expression_noise is None:
                expression_noise = th.randn_like(expression_start)
            if audio_noise is None:
                audio_noise = th.randn_like(audio_start)
            motion_t = self.q_sample(motion_start, t=motion_timestep, noise=motion_noise)
            expression_t = self.q_sample(expression_start, t=expression_timestep, noise=expression_noise)
            audio_t = self.q_sample(audio_start, t=audio_timestep, noise=audio_noise)
        
        terms = {}
        
        if self.loss_type == LossType.KL or self.loss_type == LossType.RESCALED_KL:
            terms["loss"] = self._vb_terms_bpd(
                model=model,
                motion_start=motion_start,
                motion_t=motion_t,
                motion_timstep=motion_timestep,
                expression_start=expression_start,
                expression_t=expression_t,
                expression_timstep=expression_timestep,
                audio_start=audio_start,
                audio_t=audio_t,
                audio_timstep=audio_timestep,
                clip_denoised=False,
                model_kwargs=model_kwargs,
            )["output"]
            if self.loss_type == LossType.RESCALED_KL:
                terms["loss"] *= self.num_timesteps
        else:
            model_output = model(motion_t, expression_t, audio_t, \
                motion_timestep, expression_timestep, audio_timestep, model_kwargs)
            
            motion_output = model_output.pop("motion", None)
            expression_output = model_output.pop("expression", None)
            audio_output = model_output.pop("audio", None)
            extra = model_output


            if self.return_startx and self.model_mean_type == ModelMeanType.EPSILON:
                # FOR motion
                B, N, C = motion_t.shape
                assert motion_output.shape == (B, N, C * 2)
                motion_output, motion_var_values = th.split(motion_output, C, dim=1)
                pred_motion_start = self._predict_xstart_from_eps(x_t=motion_t, t=motion_timestep, eps=motion_output)
                #################################################################
                # FOR expression
                B, N, C = expression_t.shape
                assert expression_output.shape == (B, N, C * 2)
                expression_output, expression_var_values = th.split(expression_output, C, dim=1)
                pred_expression_start = self._predict_xstart_from_eps(x_t=expression_t, t=expression_timestep, eps=expression_output)
                #################################################################
                # FOR audio
                B, N, C = audio_t.shape
                assert audio_output.shape == (B, N, C * 2)
                audio_output, audio_var_values = th.split(audio_output, C, dim=1)
                pred_audio_start = self._predict_xstart_from_eps(x_t=audio_t, t=audio_timestep, eps=audio_output)
                #################################################################
                # RETURN
                # _output后缀的是noise，模型直接输出noise
                return motion_output, expression_output, audio_output, pred_motion_start, pred_expression_start, pred_audio_start, motion_t, expression_t, audio_t

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
                # FOR audio
                B, N, C = audio_t.shape
                assert audio_output.shape == (B, N, C * 2)
                audio_output, audio_var_values = th.split(audio_output, C, dim=1)
                #################################################################
                # Learn the variance using the variational bound, but don't let it affect our mean prediction.
                motion_frozen_out = th.cat([motion_output.detach(), motion_var_values], dim=1)
                expression_frozen_out = th.cat([expression_output.detach(), expression_var_values], dim=1)
                audio_frozen_out = th.cat([audio_output.detach(), audio_var_values], dim=1)
                terms["vb"] = self._vb_terms_bpd(                    
                    model=lambda *args, r1=motion_frozen_out, r2=expression_frozen_out, r3=audio_frozen_out, **kwargs: {
                        "motion": r1,
                        "expression": r2,
                        "audio": r3,
                    },
                    motion_start=motion_start,
                    motion_t=motion_t,
                    motion_timstep=motion_timestep,
                    expression_start=expression_start,
                    expression_t=expression_t,
                    expression_timstep=expression_timestep,
                    audio_start=audio_start,
                    audio_t=audio_t,
                    audio_timstep=audio_timestep,
                    clip_denoised=False,
                    model_kwargs=model_kwargs,
                )["output"]
                if self.loss_type == LossType.RESCALED_MSE or self.loss_type == LossType.RESCALED_MOT_ANGLEDIS_EXP_MSE_AUDIO_MAE \
                    or self.loss_type == LossType.RESCALED_MAE:
                    # Divide by 1000 for equivalence with initial implementation.
                    # Without a factor of 1/1000, the VB term hurts the MSE term.
                    terms["vb"] *= self.num_timesteps / 1000.0

            # FOR motion
            motion_target = {
                ModelMeanType.PREVIOUS_X: self.q_posterior_mean_variance(x_start=motion_start, x_t=motion_t, t=motion_timestep)[0],
                ModelMeanType.START_X: motion_start,
                ModelMeanType.EPSILON: motion_noise,
                ModelMeanType.FLOW_VELOCITY: motion_noise - motion_start,
            }[self.model_mean_type]
            #################################################################
            # FOR expression
            expression_target = {
                ModelMeanType.PREVIOUS_X: self.q_posterior_mean_variance(x_start=expression_start, x_t=expression_t, t=expression_timestep)[0],
                ModelMeanType.START_X: expression_start,
                ModelMeanType.EPSILON: expression_noise,
                ModelMeanType.FLOW_VELOCITY: expression_noise - expression_start,
            }[self.model_mean_type]
            #################################################################
            # FOR audio
            audio_target = {
                ModelMeanType.PREVIOUS_X: self.q_posterior_mean_variance(x_start=audio_start, x_t=audio_t, t=audio_timestep)[0],
                ModelMeanType.START_X: audio_start,
                ModelMeanType.EPSILON: audio_noise,
                ModelMeanType.FLOW_VELOCITY: audio_noise - audio_start,
            }[self.model_mean_type]
            #################################################################
            assert motion_output.shape == motion_target.shape == motion_start.shape
            assert expression_output.shape == expression_target.shape == expression_start.shape
            assert audio_output.shape == audio_target.shape == audio_start.shape
            #################################################################
            # FOR loss calculation ##########################################
            loss_motion_weighting, loss_expression_weighting, loss_audio_weighting \
                = self.loss_weighting(motion_timestep, expression_timestep, audio_timestep)
            loss_motion_weighting = expand_dims_like(loss_motion_weighting, motion_target)
            loss_expression_weighting = expand_dims_like(loss_expression_weighting, expression_target)
            loss_audio_weighting = expand_dims_like(loss_audio_weighting, audio_target)

            loss_items = self.loss_function(
                motion_target, motion_output, loss_motion_weighting,
                expression_target, expression_output, loss_expression_weighting,
                audio_target, audio_output, loss_audio_weighting,
                **kwargs
            )
            # loss_motion = (motion_target - motion_output) ** 2 * loss_motion_weighting
            # loss_expression = (expression_target - expression_output) ** 2 * loss_expression_weighting
            # loss_audio = (audio_target - audio_output) ** 2 * loss_audio_weighting
                
            #################################################################
            # if model_kwargs.get("mask_ratio", False) and model_kwargs["mask_ratio"] > 0:
            #     assert "mask" in model_output
            #     loss = F.avg_pool2d(loss.mean(dim=1), model.model.module.patch_size).flatten(1)
            #     mask = model_output["mask"]
            #     unmask = 1 - mask
            #     terms["mse"] = mean_flat(loss * unmask) * unmask.shape[1] / unmask.sum(1)
            #     if model_kwargs["mask_loss_coef"] > 0:
            #         terms["mae"] = model_kwargs["mask_loss_coef"] * mean_flat(loss * mask) * mask.shape[1] / mask.sum(1)
            # else:
            #     terms["mse"] = mean_flat(loss)
            # 暂时先抛弃上面的逻辑，上面这个应该是适用图像的
            for k, v in loss_items.items():
                terms[k] = v
            terms["loss"] = sum(terms.values())
        # else:
        #     raise NotImplementedError(self.loss_type)

        return terms

    def training_losses_diffusers(
        self, 
        model, 
        motion_start, 
        expression_start, 
        audio_start, 
        motion_timestep, 
        expression_timestep, 
        audio_timestep, 
        model_kwargs=None, 
        motion_noise=None, 
        expression_noise=None, 
        audio_noise=None, 
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
        if model_kwargs is None:
            model_kwargs = {}
        if skip_noise:
            motion_t = motion_start
            expression_t = expression_start
            audio_t = audio_start
        else:
            if motion_noise is None:
                motion_noise = th.randn_like(motion_start)
            if expression_noise is None:
                expression_noise = th.randn_like(expression_start)
            if audio_noise is None:
                audio_noise = th.randn_like(audio_start)
            motion_t = self.q_sample(motion_start, t=motion_timestep, noise=motion_noise)
            expression_t = self.q_sample(expression_start, t=expression_timestep, noise=expression_noise)
            audio_t = self.q_sample(audio_start, t=audio_timestep, noise=audio_noise)

        terms = {}

        if self.loss_type == LossType.KL or self.loss_type == LossType.RESCALED_KL:
            terms["loss"] = self._vb_terms_bpd(
                model=model,
                motion_start=motion_start,
                motion_t=motion_t,
                motion_timestep=motion_timestep,
                expression_start=expression_start,
                expression_t=expression_t,
                expression_timestep=expression_timestep,
                audio_start=audio_start,
                audio_t=audio_t,
                audio_timestep=audio_timestep,
                clip_denoised=False,
                model_kwargs=model_kwargs,
            )["output"]
            if self.loss_type == LossType.RESCALED_KL:
                terms["loss"] *= self.num_timesteps
        else:
            model_output = model(motion_t, expression_t, audio_t, \
                motion_timestep, expression_timestep, audio_timestep, model_kwargs)
            
            motion_output = model_output.pop("motion", None)
            expression_output = model_output.pop("expression", None)
            audio_output = model_output.pop("audio", None)
            extra = model_output

            if self.return_startx and self.model_mean_type == ModelMeanType.EPSILON:
                B, N, C = motion_t.shape
                assert motion_output.shape == (B, N, C * 2)
                motion_output = th.split(motion_output, C, dim=1)[0]
                pred_motion_start = self._predict_xstart_from_eps(x_t=motion_t, t=motion_timestep, eps=motion_output)
                
                B, N, C = expression_t.shape
                assert expression_output.shape == (B, N, C * 2)
                expression_output = th.split(expression_output, C, dim=1)[0]
                pred_expression_start = self._predict_xstart_from_eps(x_t=expression_t, t=expression_timestep, eps=expression_output)
                
                B, N, C = audio_t.shape
                assert audio_output.shape == (B, N, C * 2)
                audio_output = th.split(audio_output, C, dim=1)[0]
                pred_audio_start = self._predict_xstart_from_eps(x_t=audio_t, t=audio_timestep, eps=audio_output)
                
                return motion_output, expression_output, audio_output, pred_motion_start, pred_expression_start, pred_audio_start, motion_t, expression_t, audio_t
                

            if self.model_var_type in [
                ModelVarType.LEARNED,
                ModelVarType.LEARNED_RANGE,
            ]:
                B, N, C = motion_t.shape
                assert motion_output.shape == (B, N, C * 2)
                motion_output, motion_var_values = th.split(motion_output, C, dim=1)
                
                B, N, C = expression_t.shape
                assert expression_output.shape == (B, N, C * 2)
                expression_output, expression_var_values = th.split(expression_output, C, dim=1)
                
                B, N, C = audio_t.shape
                assert audio_output.shape == (B, N, C * 2)
                audio_output, audio_var_values = th.split(audio_output, C, dim=1)
                # Learn the variance using the variational bound, but don't let it affect our mean prediction.
                motion_frozen_out = th.cat([motion_output.detach(), motion_var_values], dim=1)
                expression_frozen_out = th.cat([expression_output.detach(), expression_var_values], dim=1)
                audio_frozen_out = th.cat([audio_output.detach(), audio_var_values], dim=1)
                
                terms["vb"] = self._vb_terms_bpd(
                    model=lambda *args, r1=motion_frozen_out, r2=expression_frozen_out, r3=audio_frozen_out, **kwargs: {
                        "motion": r1,
                        "expression": r2,
                        "audio": r3,
                    },
                    motion_start=motion_start,
                    motion_t=motion_t,
                    motion_timestep=motion_timestep,
                    expression_start=expression_start,
                    expression_t=expression_t,
                    expression_timestep=expression_timestep,
                    audio_start=audio_start,
                    audio_t=audio_t,
                    audio_timestep=audio_timestep,
                    clip_denoised=False,
                )["output"]
                if self.loss_type == LossType.RESCALED_MSE or self.loss_type == LossType.RESCALED_MOT_ANGLEDIS_EXP_MSE_AUDIO_MAE \
                    or self.loss_type == LossType.RESCALED_MAE:
                    # Divide by 1000 for equivalence with initial implementation.
                    # Without a factor of 1/1000, the VB term hurts the MSE term.
                    terms["vb"] *= self.num_timesteps / 1000.0

            motion_target = {
                ModelMeanType.PREVIOUS_X: self.q_posterior_mean_variance(x_start=motion_start, x_t=motion_t, t=motion_timestep)[0],
                ModelMeanType.START_X: motion_start,
                ModelMeanType.EPSILON: motion_noise,
                ModelMeanType.FLOW_VELOCITY: motion_noise - motion_start,
            }[self.model_mean_type]
            expression_target = {
                ModelMeanType.PREVIOUS_X: self.q_posterior_mean_variance(x_start=expression_start, x_t=expression_t, t=expression_timestep)[0],
                ModelMeanType.START_X: expression_start,
                ModelMeanType.EPSILON: expression_noise,
                ModelMeanType.FLOW_VELOCITY: expression_noise - expression_start,
            }[self.model_mean_type]
            audio_target = {
                ModelMeanType.PREVIOUS_X: self.q_posterior_mean_variance(x_start=audio_start, x_t=audio_t, t=audio_timestep)[0],
                ModelMeanType.START_X: audio_start,
                ModelMeanType.EPSILON: audio_noise,
                ModelMeanType.FLOW_VELOCITY: audio_noise - audio_start,
            }[self.model_mean_type]
            
            assert motion_output.shape == motion_target.shape == motion_start.shape
            assert expression_output.shape == expression_target.shape == expression_start.shape
            assert audio_output.shape == audio_target.shape == audio_start.shape
            #################################################################
            # FOR loss calculation ##########################################
            loss_motion_weighting, loss_expression_weighting, loss_audio_weighting \
                = self.loss_weighting(motion_timestep, expression_timestep, audio_timestep)
            loss_motion_weighting = expand_dims_like(loss_motion_weighting, motion_target)
            loss_expression_weighting = expand_dims_like(loss_expression_weighting, expression_target)
            loss_audio_weighting = expand_dims_like(loss_audio_weighting, audio_target)
            
            loss_items = self.loss_function(
                motion_target, motion_output, loss_motion_weighting,
                expression_target, expression_output, loss_expression_weighting,
                audio_target, audio_output, loss_audio_weighting
            )
            #################################################################
            for k, v in loss_items.items():
                terms[k] = v
            terms["loss"] = sum(terms.values())
        # else:
        #     raise NotImplementedError(self.loss_type)

        return terms  

    def p_mean_variance(
        self, 
        model, 
        motion_t,
        expression_t,
        audio_t, 
        motion_timestep,
        expression_timestep,
        audio_timestep, 
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

        assert motion_t.shape[0] == expression_t.shape[0] == audio_t.shape[0], "batch size mismatch"
        B, mN, mC = motion_t.shape
        B, eN, eC = expression_t.shape
        B, aN, aC = audio_t.shape
        assert motion_timestep.shape == (B,) \
            and expression_timestep.shape == (B,) \
            and audio_timestep.shape == (B,)
        model_output = model(
            motion_t,
            expression_t,
            audio_t,
            motion_timestep,
            expression_timestep,
            audio_timestep, 
            model_kwargs
        )
        
        motion_output = model_output.pop("motion", None)
        expression_output = model_output.pop("expression", None)
        audio_output = model_output.pop("audio", None)
        extra = model_output

        if self.model_var_type in [ModelVarType.LEARNED, ModelVarType.LEARNED_RANGE]:
            assert motion_output.shape == (B, mN, mC * 2) and expression_output.shape == (B, eN, eC * 2) \
                and audio_output.shape == (B, aN, aC * 2)
            motion_output, motion_var_values = th.split(motion_output, mC, dim=-1)
            expression_output, expression_var_values = th.split(expression_output, eC, dim=-1)
            audio_output, audio_var_values = th.split(audio_output, aC, dim=-1)
            
            motion_min_log = _extract_into_tensor(self.posterior_log_variance_clipped, motion_timestep, motion_output.shape)
            motion_max_log = _extract_into_tensor(np.log(self.betas), motion_timestep, motion_output.shape)
            
            expression_min_log = _extract_into_tensor(self.posterior_log_variance_clipped, expression_timestep, expression_output.shape)
            expression_max_log = _extract_into_tensor(np.log(self.betas), expression_timestep, expression_output.shape)
            
            audio_min_log = _extract_into_tensor(self.posterior_log_variance_clipped, audio_timestep, audio_output.shape)
            audio_max_log = _extract_into_tensor(np.log(self.betas), audio_timestep, audio_output.shape)
            
            # The model_var_values is [-1, 1] for [min_var, max_var].
            motion_frac = (motion_var_values + 1) / 2
            expression_frac = (expression_var_values + 1) / 2
            audio_frac = (audio_var_values + 1) / 2
            
            motion_log_variance = motion_frac * motion_max_log + (1 - motion_frac) * motion_min_log
            expression_log_variance = expression_frac * expression_max_log + (1 - expression_frac) * expression_min_log
            audio_log_variance = audio_frac * audio_max_log + (1 - audio_frac) * audio_min_log
            
            motion_variance = th.exp(motion_log_variance)
            expression_variance = th.exp(expression_log_variance)
            audio_variance = th.exp(audio_log_variance)
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
            
            motion_variance = _extract_into_tensor(model_variance, motion_timestep, motion_output.shape)
            motion_log_variance = _extract_into_tensor(model_log_variance, motion_timestep, motion_output.shape)
            
            expression_variance = _extract_into_tensor(model_variance, expression_timestep, expression_output.shape)
            expression_log_variance = _extract_into_tensor(model_log_variance, expression_timestep, expression_output.shape)
            
            audio_variance = _extract_into_tensor(model_variance, audio_timestep, audio_output.shape)
            audio_log_variance = _extract_into_tensor(model_log_variance, audio_timestep, audio_output.shape)
        else:
            motion_variance = th.zeros_like(motion_output)
            motion_log_variance = th.zeros_like(motion_output)
            
            expression_variance = th.zeros_like(expression_output)
            expression_log_variance = th.zeros_like(expression_output)
            
            audio_variance = th.zeros_like(audio_output)
            audio_log_variance = th.zeros_like(audio_output)

        def process_xstart(x):
            if denoised_fn is not None:
                x = denoised_fn(x)
            if clip_denoised:
                return x.clamp(-1, 1)
            return x

        if self.model_mean_type == ModelMeanType.START_X:
            pred_motion_start = process_xstart(motion_output)
            pred_expression_start = process_xstart(expression_output)
            pred_audio_start = process_xstart(audio_output)
        else:
            pred_motion_start = process_xstart(self._predict_xstart_from_eps(x_t=motion_t, t=motion_timestep, eps=motion_output))
            pred_expression_start = process_xstart(self._predict_xstart_from_eps(x_t=expression_t, t=expression_timestep, eps=expression_output))
            pred_audio_start = process_xstart(self._predict_xstart_from_eps(x_t=audio_t, t=audio_timestep, eps=audio_output))
        motion_mean, _, _ = self.q_posterior_mean_variance(x_start=pred_motion_start, x_t=motion_t, t=motion_timestep)
        expression_mean, _, _ = self.q_posterior_mean_variance(x_start=pred_expression_start, x_t=expression_t, t=expression_timestep)
        audio_mean, _, _ = self.q_posterior_mean_variance(x_start=pred_audio_start, x_t=audio_t, t=audio_timestep)
        
        assert motion_mean.shape == motion_log_variance.shape == pred_motion_start.shape == motion_t.shape
        assert expression_mean.shape == expression_log_variance.shape == pred_expression_start.shape == expression_t.shape
        assert audio_mean.shape == audio_log_variance.shape == pred_audio_start.shape == audio_t.shape
        return {
            "motion_mean": motion_mean,
            "motion_variance": motion_variance,
            "motion_log_variance": motion_log_variance,
            "pred_motion_start": pred_motion_start,
            "expression_mean": expression_mean,
            "expression_variance": expression_variance,
            "expression_log_variance": expression_log_variance,
            "pred_expression_start": pred_expression_start,
            "audio_mean": audio_mean,
            "audio_variance": audio_variance,
            "audio_log_variance": audio_log_variance,
            "pred_audio_start": pred_audio_start,
            "extra": extra,
        }
    
    
    def _vb_terms_bpd(
        self, 
        model, 
        motion_start, 
        motion_t, 
        motion_timstep,
        expression_start,
        expression_t,
        expression_timstep,
        audio_start,
        audio_t,
        audio_timstep,
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
            self.q_posterior_mean_variance(x_start=motion_start, x_t=motion_t, t=motion_timstep)
        expression_true_mean, _, expression_true_log_variance_clipped = \
            self.q_posterior_mean_variance(x_start=expression_start, x_t=expression_t, t=expression_timstep)
        audio_true_mean, _, audio_true_log_variance_clipped = \
            self.q_posterior_mean_variance(x_start=audio_start, x_t=audio_t, t=audio_timstep)
        out = self.p_mean_variance(model, motion_t=motion_t, expression_t=expression_t, audio_t=audio_t,
            motion_timestep=motion_timstep, expression_timestep=expression_timstep, audio_timestep=audio_timstep, 
            clip_denoised=clip_denoised, model_kwargs=model_kwargs)
        motion_kl = normal_kl(motion_true_mean, motion_true_log_variance_clipped, out["motion_mean"], out["motion_log_variance"])
        expression_kl = normal_kl(expression_true_mean, expression_true_log_variance_clipped, out["expression_mean"], out["expression_log_variance"])
        audio_kl = normal_kl(audio_true_mean, audio_true_log_variance_clipped, out["audio_mean"], out["audio_log_variance"])
        
        motion_kl = mean_flat(motion_kl) / np.log(2.0)
        expression_kl = mean_flat(expression_kl) / np.log(2.0)
        audio_kl = mean_flat(audio_kl) / np.log(2.0)

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
        
        audio_decoder_nll = -discretized_gaussian_log_likelihood(
            audio_start, means=out["audio_mean"], log_scales=0.5 * out["audio_log_variance"]
        )
        assert audio_decoder_nll.shape == audio_start.shape
        audio_decoder_nll = mean_flat(audio_decoder_nll) / np.log(2.0)

        # At the first timestep return the decoder NLL,
        # otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
        motion_loss = th.where((motion_timstep == 0), motion_decoder_nll, motion_kl)
        expression_loss = th.where((expression_timstep == 0), expression_decoder_nll, expression_kl)
        audio_loss = th.where((audio_timstep == 0), audio_decoder_nll, audio_kl)
        return {
            "output": motion_loss + expression_loss + audio_loss, 
            "pred_motion_start": out["pred_motion_start"],
            "pred_expression_start": out["pred_expression_start"],
            "pred_audio_start": out["pred_audio_start"],
        }

            