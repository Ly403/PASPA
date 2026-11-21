import torch
import numpy as np
from diffusion.utils import rotation_conversions as rc


def get_motion_rep_tensor(smplx_model, motion_tensor, pose_fps=30, device="cuda", expressions=None, expression_only=False, betas=None):
    bs, n, _ = motion_tensor.shape
    motion_tensor = motion_tensor.float().to(device)
    motion_tensor_reshaped = motion_tensor.reshape(bs * n, 165)
    if betas is None:
        betas = torch.zeros(bs * n, 300, device=device)
    else:
        if len(betas.shape) == 3:
            betas = betas.to(device).view(-1, 300)
        if len(betas.shape) == 2:
            betas = betas.to(device)
        elif len(betas.shape) == 1:
            betas = betas.to(device).unsqueeze(0).repeat(bs * n, 1)
        else:
            raise ValueError(f"Invalid betas shape: {betas.shape}")
    if expressions is not None and expression_only:
        # print("xx")
        expressions = expressions.view(-1, 100).float().to(device)
        output = smplx_model(
            betas=betas,
            transl=torch.zeros(bs * n, 3, device=device),
            expression=expressions,
            jaw_pose=motion_tensor_reshaped[:, 22 * 3:23 * 3],
            global_orient=torch.zeros(bs * n, 3, device=device),
            body_pose=torch.zeros(bs * n, 21*3, device=device),
            left_hand_pose=torch.zeros(bs * n, 15*3, device=device),
            right_hand_pose=torch.zeros(bs * n, 15*3, device=device),
            return_joints=True,
            leye_pose=torch.zeros(bs * n, 3, device=device),
            reye_pose=torch.zeros(bs * n, 3, device=device),
            )
        joints = output["vertices"].reshape(bs, n, -1)
        return {"vertices": joints}

    output = smplx_model(
        betas=betas,
        transl=torch.zeros(bs * n, 3, device=device),
        expression=torch.zeros(bs * n, 100, device=device),
        jaw_pose=torch.zeros(bs * n, 3, device=device),
        global_orient=torch.zeros(bs * n, 3, device=device),
        body_pose=motion_tensor_reshaped[:, 3:21 * 3 + 3],
        left_hand_pose=motion_tensor_reshaped[:, 25 * 3:40 * 3],
        right_hand_pose=motion_tensor_reshaped[:, 40 * 3:55 * 3],
        return_joints=True,
        leye_pose=torch.zeros(bs * n, 3, device=device),
        reye_pose=torch.zeros(bs * n, 3, device=device),
    )
    joints = output['joints'].reshape(bs, n, 127, 3)[:, :, :55, :]
    dt = 1 / pose_fps
    init_vel = (joints[:, 1:2] - joints[:, 0:1]) / dt
    middle_vel = (joints[:, 2:] - joints[:, :-2]) / (2 * dt)
    final_vel = (joints[:, -1:] - joints[:, -2:-1]) / dt
    vel = torch.cat([init_vel, middle_vel, final_vel], dim=1)
    position = joints
    rot_matrices = rc.axis_angle_to_matrix(motion_tensor.reshape(bs, n, 55, 3))
    rot6d = rc.matrix_to_rotation_6d(rot_matrices).reshape(bs, n, 55, 6)
    init_vel_ang = (motion_tensor[:, 1:2] - motion_tensor[:, 0:1]) / dt
    middle_vel_ang = (motion_tensor[:, 2:] - motion_tensor[:, :-2]) / (2 * dt)
    final_vel_ang = (motion_tensor[:, -1:] - motion_tensor[:, -2:-1]) / dt
    angular_velocity = torch.cat([init_vel_ang, middle_vel_ang, final_vel_ang], dim=1).reshape(bs, n, 55, 3)
    rep15d = torch.cat([position, vel, rot6d, angular_velocity], dim=3).reshape(bs, n, 55 * 15)
    return {
        "position": position,
        "velocity": vel,
        "rotation": rot6d,
        "axis_angle": motion_tensor,
        "angular_velocity": angular_velocity,
        "rep15d": rep15d,
    }

def get_motion_rep_numpy(smplx_model, poses_np, pose_fps=30, device="cuda", \
    expressions=None, expression_only=False, betas=None):
    n = poses_np.shape[0]

    # Convert numpy to torch tensor for SMPL-X forward pass
    poses_ts = torch.from_numpy(poses_np).float().to(device).unsqueeze(0)  # (1, n, 165)
    poses_ts_reshaped = poses_ts.reshape(-1, 165)  # (n, 165)
    betas = torch.zeros(n, 300, device=device) if betas is None else torch.from_numpy(betas)
    if len(betas.shape) == 2 and betas.shape[0] == n:
        betas = betas.to(device)
    elif len(betas.shape) == 1:
        betas = betas.to(device).unsqueeze(0).repeat(n, 1)
    else:
        raise ValueError(f"Invalid betas shape: {betas.shape}")
    if expressions is not None and expression_only:
        # print("xx")
        expressions = torch.from_numpy(expressions).float().to(device)
        output = smplx_model(
            betas=betas,
            transl=torch.zeros(n, 3, device=device),
            expression=expressions,
            jaw_pose=poses_ts_reshaped[:, 22 * 3:23 * 3],
            global_orient=torch.zeros(n, 3, device=device),
            body_pose=torch.zeros(n, 21*3, device=device),
            left_hand_pose=torch.zeros(n, 15*3, device=device),
            right_hand_pose=torch.zeros(n, 15*3, device=device),
            return_joints=True,
            leye_pose=torch.zeros(n, 3, device=device),
            reye_pose=torch.zeros(n, 3, device=device),
            )
        joints = output["vertices"].detach().cpu().numpy().reshape(n, -1)
        return {"vertices": joints}

    # Run smplx model to get joints
    output = smplx_model(
        betas=betas,
        transl=torch.zeros(n, 3, device=device),
        expression=torch.zeros(n, 100, device=device),
        jaw_pose=torch.zeros(n, 3, device=device),
        global_orient=torch.zeros(n, 3, device=device),
        body_pose=poses_ts_reshaped[:, 3:21 * 3 + 3],
        left_hand_pose=poses_ts_reshaped[:, 25 * 3:40 * 3],
        right_hand_pose=poses_ts_reshaped[:, 40 * 3:55 * 3],
        return_joints=True,
        leye_pose=torch.zeros(n, 3, device=device),
        reye_pose=torch.zeros(n, 3, device=device),
    )
    joints = output["joints"].detach().cpu().numpy().reshape(n, 127, 3)[:, :55, :]

    dt = 1 / pose_fps
    # Compute linear velocity
    init_vel = (joints[1:2] - joints[0:1]) / dt
    middle_vel = (joints[2:] - joints[:-2]) / (2 * dt)
    final_vel = (joints[-1:] - joints[-2:-1]) / dt
    vel = np.concatenate([init_vel, middle_vel, final_vel], axis=0)

    position = joints

    # Compute rotation 6D from axis-angle
    poses_ts_reshaped_aa = poses_ts.reshape(1, n, 55, 3)
    rot_matrices = rc.axis_angle_to_matrix(poses_ts_reshaped_aa)[0]  # (n, 55, 3, 3)
    rot6d = rc.matrix_to_rotation_6d(rot_matrices).reshape(n, 55, 6).cpu().numpy()

    # Compute angular velocity
    init_vel_ang = (poses_np[1:2] - poses_np[0:1]) / dt
    middle_vel_ang = (poses_np[2:] - poses_np[:-2]) / (2 * dt)
    final_vel_ang = (poses_np[-1:] - poses_np[-2:-1]) / dt
    angular_velocity = np.concatenate([init_vel_ang, middle_vel_ang, final_vel_ang], axis=0).reshape(n, 55, 3)

    # rep15d: position(55*3), vel(55*3), rot6d(55*6), angular_velocity(55*3) => total 55*(3+3+6+3)=55*15
    rep15d = np.concatenate([position, vel, rot6d, angular_velocity], axis=2).reshape(n, 55 * 15)

    return {
        "position": position,
        "velocity": vel,
        "rotation": rot6d,
        "axis_angle": poses_np,
        "angular_velocity": angular_velocity,
        "rep15d": rep15d,
    }