"""
Quintic Hermite Bridge for block-boundary C² continuity.

Given the last few frames of the previous block and the current block's
denoised prediction, replace the first `m` frames of the current block
with a quintic Hermite curve that exactly matches position, velocity,
and acceleration at both endpoints.

Supports two interpolation spaces:
  - "rot6d": interpolate directly in rot6d + translation space
  - "axis_angle": convert rot6d → axis-angle, interpolate, convert back

Usage:
    from diffusion.pipeline.quintic_bridge import apply_quintic_hermite_bridge

    denoised_pred = apply_quintic_hermite_bridge(
        prev_frames=prev_tail,       # (B, >=3, C)
        current_block=denoised_pred,  # (B, block_size, C)
        bridge_frames=8,
        fps=30.0,
        num_joints=55,
        interp_space="rot6d",
    )
"""

import torch
from diffusion.utils import rotation_conversions as rc


def _quintic_hermite_basis(tau: torch.Tensor):
    """Compute the 6 quintic Hermite basis functions at normalized time tau.

    Args:
        tau: (m+1,) tensor of normalized time values in [0, 1].

    Returns:
        Tuple of 6 tensors, each shape (m+1,):
            h00, h10, h20, h01, h11, h21
    """
    t2 = tau * tau
    t3 = t2 * tau
    t4 = t3 * tau
    t5 = t4 * tau

    h00 = 1.0 - 10.0 * t3 + 15.0 * t4 - 6.0 * t5
    h10 = tau - 6.0 * t3 + 8.0 * t4 - 3.0 * t5
    h20 = 0.5 * t2 - 1.5 * t3 + 1.5 * t4 - 0.5 * t5
    h01 = 10.0 * t3 - 15.0 * t4 + 6.0 * t5
    h11 = -4.0 * t3 + 7.0 * t4 - 3.0 * t5
    h21 = 0.5 * t3 - t4 + 0.5 * t5

    return h00, h10, h20, h01, h11, h21


def _estimate_vel_acc_backward(frames: torch.Tensor, dt: float):
    """Estimate velocity and acceleration at the last frame using backward differences.

    Args:
        frames: (B, >=3, D) tensor. Uses last 3 frames.
        dt: time step (1/fps).

    Returns:
        x0: (B, 1, D) position at last frame
        v0: (B, 1, D) velocity at last frame
        a0: (B, 1, D) acceleration at last frame
    """
    x_m1 = frames[:, -1:]   # t = -1 (last)
    x_m2 = frames[:, -2:-1] # t = -2
    x_m3 = frames[:, -3:-2] # t = -3

    x0 = x_m1
    v0 = (x_m1 - x_m2) / dt
    a0 = (x_m1 - 2.0 * x_m2 + x_m3) / (dt * dt)
    return x0, v0, a0


def _estimate_vel_acc_central(block: torch.Tensor, idx: int, dt: float):
    """Estimate velocity and acceleration at frame `idx` using central differences.

    Args:
        block: (B, T, D) tensor.
        idx: frame index (must be >= 1 and <= T-2).
        dt: time step (1/fps).

    Returns:
        x1: (B, 1, D) position at frame idx
        v1: (B, 1, D) velocity at frame idx
        a1: (B, 1, D) acceleration at frame idx
    """
    x_prev = block[:, idx - 1:idx]
    x_curr = block[:, idx:idx + 1]
    x_next = block[:, idx + 1:idx + 2]

    x1 = x_curr
    v1 = (x_next - x_prev) / (2.0 * dt)
    a1 = (x_next - 2.0 * x_curr + x_prev) / (dt * dt)
    return x1, v1, a1


def _quintic_hermite_interp(
    x0: torch.Tensor,
    v0: torch.Tensor,
    a0: torch.Tensor,
    x1: torch.Tensor,
    v1: torch.Tensor,
    a1: torch.Tensor,
    m: int,
    T: float,
):
    """Compute quintic Hermite interpolation for m+1 points (j=0..m).

    Args:
        x0, v0, a0: (B, 1, D) left endpoint conditions.
        x1, v1, a1: (B, 1, D) right endpoint conditions.
        m: number of bridge frames (bridge covers frames 0..m).
        T: total time span = m * dt.

    Returns:
        bridge: (B, m+1, D) interpolated values.
    """
    device = x0.device
    dtype = x0.dtype

    # tau = j/m for j = 0, 1, ..., m
    tau = torch.linspace(0.0, 1.0, m + 1, device=device, dtype=dtype)
    h00, h10, h20, h01, h11, h21 = _quintic_hermite_basis(tau)

    # Reshape basis for broadcasting: (1, m+1, 1)
    h00 = h00.unsqueeze(0).unsqueeze(-1)
    h10 = h10.unsqueeze(0).unsqueeze(-1)
    h20 = h20.unsqueeze(0).unsqueeze(-1)
    h01 = h01.unsqueeze(0).unsqueeze(-1)
    h11 = h11.unsqueeze(0).unsqueeze(-1)
    h21 = h21.unsqueeze(0).unsqueeze(-1)

    # x0, v0, a0, x1, v1, a1 are (B, 1, D) → broadcast with (1, m+1, 1)
    bridge = (
        h00 * x0
        + T * h10 * v0
        + T * T * h20 * a0
        + h01 * x1
        + T * h11 * v1
        + T * T * h21 * a1
    )
    return bridge


def apply_quintic_hermite_bridge(
    prev_frames: torch.Tensor,
    current_block: torch.Tensor,
    bridge_frames: int = 8,
    fps: float = 30.0,
    num_joints: int = 55,
    interp_space: str = "rot6d",
) -> torch.Tensor:
    """Apply quintic Hermite bridge to the first `bridge_frames` of current_block.

    Args:
        prev_frames: (B, >=3, C) last frames from previous block.
            C = num_joints * 6 + 3 (rot6d + translation).
        current_block: (B, block_size, C) current block denoised prediction.
        bridge_frames: number of frames to replace with bridge (m).
        fps: frame rate for dt computation.
        num_joints: number of joints (default 55 for SMPLX).
        interp_space: "rot6d" or "axis_angle".

    Returns:
        Modified current_block with first `bridge_frames` replaced by bridge.
    """
    B, block_size, C = current_block.shape
    m = min(bridge_frames, block_size - 2)  # need at least idx+1 for central diff
    if m < 1:
        return current_block
    if prev_frames.shape[1] < 3:
        return current_block

    dt = 1.0 / fps
    T = m * dt
    rot6d_dim = num_joints * 6

    result = current_block.clone()

    if interp_space == "rot6d":
        # Interpolate directly in rot6d + translation space
        x0, v0, a0 = _estimate_vel_acc_backward(prev_frames, dt)
        x1, v1, a1 = _estimate_vel_acc_central(current_block, m, dt)

        bridge = _quintic_hermite_interp(x0, v0, a0, x1, v1, a1, m, T)
        # bridge is (B, m+1, C), we use frames 0..m-1 (skip the last which is the anchor point)
        result[:, :m] = bridge[:, :m]

    elif interp_space == "axis_angle":
        # Convert rot6d → axis_angle, interpolate, convert back
        # Split rotation and translation
        prev_rot6d = prev_frames[:, :, :rot6d_dim]
        prev_trans = prev_frames[:, :, rot6d_dim:]
        curr_rot6d = current_block[:, :, :rot6d_dim]
        curr_trans = current_block[:, :, rot6d_dim:]

        # rot6d → matrix → axis_angle
        prev_aa = rc.rotation_6d_to_axis_angle(
            prev_rot6d.reshape(B, prev_rot6d.shape[1], num_joints, 6)
        ).reshape(B, prev_rot6d.shape[1], num_joints * 3)
        curr_aa = rc.rotation_6d_to_axis_angle(
            curr_rot6d.reshape(B, curr_rot6d.shape[1], num_joints, 6)
        ).reshape(B, curr_rot6d.shape[1], num_joints * 3)

        # Concatenate axis_angle + translation for unified bridge
        prev_aa_trans = torch.cat([prev_aa, prev_trans], dim=-1)
        curr_aa_trans = torch.cat([curr_aa, curr_trans], dim=-1)

        x0, v0, a0 = _estimate_vel_acc_backward(prev_aa_trans, dt)
        x1, v1, a1 = _estimate_vel_acc_central(curr_aa_trans, m, dt)

        bridge = _quintic_hermite_interp(x0, v0, a0, x1, v1, a1, m, T)
        bridge_used = bridge[:, :m]  # (B, m, num_joints*3 + trans_dim)

        # Split back
        bridge_aa = bridge_used[:, :, :num_joints * 3]
        bridge_trans = bridge_used[:, :, num_joints * 3:]

        # axis_angle → matrix → rot6d
        bridge_rot6d = rc.matrix_to_rotation_6d(
            rc.axis_angle_to_matrix(
                bridge_aa.reshape(B, m, num_joints, 3)
            )
        ).reshape(B, m, rot6d_dim)

        result[:, :m, :rot6d_dim] = bridge_rot6d
        result[:, :m, rot6d_dim:] = bridge_trans

    else:
        raise ValueError(
            f"Unknown interp_space: {interp_space}. "
            "Supported: 'rot6d', 'axis_angle'"
        )

    return result
