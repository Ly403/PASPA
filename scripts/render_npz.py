"""
================================================================================
SMPLX NPZ 可视化渲染脚本
================================================================================

功能总览
--------
把 SMPLX NPZ 动作文件（必须包含 poses/betas/expressions/trans）渲染成：
  1. 视频（mesh 或 skeleton，可选带音频）              — 默认模式
  2. 透明背景 PNG 序列                                — 传 --export_png
  3. Motion Bundle：多帧叠影拼成一张 PNG                — 传 --bundle
  4. Clip PNG：指定区间按帧导出 PNG                    — 传 --export_png + --clip_*

输入约束
--------
NPZ 文件需包含以下数组：
  poses       (N, 165)    — SMPLX 165 维 axis-angle pose
  betas       (N, 300) 或 (300,)  — 体型参数（单帧也可）
  expressions (N, 100)    — 表情参数
  trans       (N, 3)      — 全局平移

================================================================================
参数详解（按作用分组）
================================================================================

--------------------------------------------------------------------------------
[1] 基础参数
--------------------------------------------------------------------------------
npz_path                  必填。输入 NPZ 文件路径。
--mode {mesh,skeleton,face}
                          渲染模式：
                            mesh     — SMPLX 实体网格（默认）
                            skeleton — 关节球 + 骨棒
                            face     — 仅人脸特写：把身体姿态置零、只保留 jaw/表情，
                                        再以双眼中点 (joints 23/24 mean) 为原点缩放
                                        face_zoom 倍，把脸部锁定到相机正中心 (0, 1, 0)。
                                        与 NPZ 的 trans / betas 无关，始终居中。
                                        适合做情绪/口型对比图。
--face_zoom FLOAT         face 模式专用：以头部为中心的缩放系数，默认 7.0。
                          越大脸越大（可能裁掉额头/下巴）；越小脸越小。推荐 5.0-9.0。
--device {cuda,cpu}       SMPLX forward 用的设备。默认有 GPU 时用 cuda。
--model_folder PATH       SMPLX 模型文件夹，默认 ./emage_evaltools/smplx_models/。
                          首次运行会自动下载 SMPLX_NEUTRAL_2020.npz。
--audio PATH              可选：给视频附加音频（仅视频模式生效）。
--output PATH             视频输出路径。默认 <npz名>.mp4。

--------------------------------------------------------------------------------
[2] 帧处理与居中
--------------------------------------------------------------------------------
--keep_transl             保留逐帧全局平移。默认关：把所有帧 trans 锁到第 0 帧，避免角色走出画面。
                          Bundle 模式想看"挥手"之类的原地动作不需要开；想看行走轨迹才开。
--no_center               禁用 X 轴居中。默认行为是把所有帧 X 的中位数移到 0，让角色居中。
                          如果 NPZ 本身已对齐好，可以传这个参数。
--no_fingers              骨架模式跳过手指关节（只画身体 25 个关节）。适合看全身粗粒度动作。

--------------------------------------------------------------------------------
[3] 分辨率 / 帧率
--------------------------------------------------------------------------------
--width  INT              渲染宽度，默认 1440。
--height INT              渲染高度，默认 2160。比例 2:3（人物竖构图友好）。
                          论文图建议至少 1440x2160；想更清晰直接再加 --ssaa 2。
--fps INT                 视频帧率，默认 30。NPZ 原始 fps 大多是 30；如是 60 改这里。

--------------------------------------------------------------------------------
[4] 颜色控制（按 body / left_arm / right_arm 三段控制）
--------------------------------------------------------------------------------
区域定义（SMPLX 关节索引）：
  left_arm  : 16 左肩 + 18 左肘 + 20 左腕 + 25-39 左手（整条左臂，肩到指尖）
  right_arm : 17 右肩 + 19 右肘 + 21 右腕 + 40-54 右手
  body      : 其余（躯干 / 头 / 双腿 / 颈 / 锁骨）

--body_color HEX          身体颜色。默认：DCDCDC（浅灰）。
--left_arm_color HEX      左臂颜色。默认：跟随 body_color。
--right_arm_color HEX     右臂颜色。默认：跟随 body_color。

作用范围：
  - 骨架模式：按关节索引分段，对应球/骨棒上色。
  - Mesh 模式：用 SMPLX 的 lbs_weights argmax 做顶点级分段（每个顶点归给权重最大的关节），
              按同一套 left_arm / right_arm / body 分组上色。肩到指尖整条手臂都会被染色。

--color HEX               快捷：三段同时设成同一颜色，等价于 --body_color=X --left_arm_color=X --right_arm_color=X。
                          与 per-part 冲突时 per-part 优先。
--color_hands             Backward-compat：emage 默认三色预设（身体蓝 + 左手绿 + 右手红，按旧的"只染手"分段）。
                          只有在 per-part 和 --color 都没传时才生效。

优先级（从高到低）：
  per-part (body / left_arm / right_arm) > --color > --color_hands > 默认浅灰统一色

--------------------------------------------------------------------------------
[5] 画质增强（默认全关，保证与老脚本完全一致）
--------------------------------------------------------------------------------
※ 以下参数仅在 --bundle 和 --export_png+--clip_* 路径下生效；
  普通视频 / 普通 PNG 序列走 fast_render.py 老逻辑，不受影响。

--ssaa INT                超采样倍率，默认 1（关）。
                          传 2：以 2*width × 2*height 渲染后 Lanczos 下采样回目标尺寸。
                          4K 级锐度，是抗锯齿最有效手段。代价：渲染时间/显存 4 倍。
                          推荐：bundle 出图 --ssaa 2；想极致清晰用 --ssaa 3（慎用，显存吃紧）。

--better_lighting         加环境光 [0.35,0.35,0.35] + 反向补光（intensity 1.8）。
                          默认场景只有一盏方向光 → 背光面纯黑；开这个后层次更自然。
                          推荐：几乎所有出图场景都该开。

--rim_light               加一盏后上方的轮廓光（intensity 2.5），让角色边缘"亮起来"。
                          bundle 叠 40 帧时尤其有用 —— 相邻帧的肢体轮廓能被区分开。
                          推荐：和 --better_lighting 一起开 = 三点光照。

--hq_skeleton             骨架模式：关节球 8×8 → 24×24，圆柱 sections → 24。
                          不开：球面能看出八边形棱角（即使 ssaa 也救不回来）。
                          开：球面平滑。成本几乎为零。强烈推荐骨架出图时开。

--pbr                     启用 PBR 材质（MetallicRoughness，roughness 0.55，metallic 0）。
                          给表面一点柔和高光，更像真实肌肤/皮革；mesh 模式观感提升最明显。
                          skeleton 模式下会让球/棒有一点塑料反光感，看个人审美。

--ground                  在角色脚下投一层 fake 投影：把角色几何沿 Y 轴压扁到最低点附近，
                          以半透明深灰渲染一份当作 drop shadow。地面本身是透明的，
                          只看得到脚下那一条阴影带，视觉上角色"落在地上"。
                          实现上不走 pyrender 的 SHADOWS_DIRECTIONAL 通道（该 shader 在
                          WSL / 软件 OpenGL 下会抛 "NoneType is not callable"，非常不稳定），
                          而是额外加一份压扁 + 半透明深色的 mesh，稳定可控。
                          想调深浅/宽窄，改 _add_shadow_mesh 里的 squash / alpha。

--------------------------------------------------------------------------------
[6] Motion Bundle 模式
--------------------------------------------------------------------------------
开启方式：传 --bundle。
输出单张 PNG：把区间内所有帧按 alpha 0.12→1.0 线性递增叠成"叠影"。

--bundle                  启用 bundle 模式。
--bundle_start INT        起始帧，默认 0。
--bundle_length INT       帧数，默认 30。论文主图一般 30-60 帧效果最好。
--bundle_output PATH      输出 PNG 路径，默认 bundle.png。

--------------------------------------------------------------------------------
[7] Clip PNG 模式
--------------------------------------------------------------------------------
开启方式：同时传 --export_png DIR 和 --clip_start / --clip_length。
输出：[clip_start, clip_start+clip_length) 区间内每帧一张透明 PNG。
文件名保留原始帧号（例如 frame_000100.png），便于追溯。

--export_png DIR          PNG 输出目录。单独传表示全序列按 sample_interval 间隔导出。
--clip_start INT          Clip 起始帧。传了就走 clip 分支。
--clip_length INT         Clip 长度（帧数）。
--sample_interval INT     clip 分支下默认改为 1（每帧都导）；普通分支默认 30。

--------------------------------------------------------------------------------
[8] 噪声对照版（可选）
--------------------------------------------------------------------------------
--render_noise            额外渲染一份"对 poses/expressions 加高斯噪声"的版本。
                          视频模式：输出 <base>_noised.mp4；PNG 模式：<dir>_noised/。
--noise_scale FLOAT       噪声强度，默认 1.0（相当于纯高斯）。

================================================================================
参数推荐组合（按目的选配）
================================================================================

[目的 A] 论文主图 / 封面图（最高画质，不计成本）
--------------------------------------------------------------------------------
Motion Bundle 模式：
  --bundle --bundle_length 30~60
  --ssaa 2 --better_lighting --rim_light --ground
  骨架再加 --hq_skeleton；mesh 再加 --pbr
  分辨率保持默认 1440x2160 或更高
显存 ~8GB（3090），30 帧耗时 ~2-3 分钟。

[目的 B] 论文方法对比图（清晰度够用，省一半时间）
--------------------------------------------------------------------------------
  --ssaa 2 --better_lighting --hq_skeleton
  不开 --rim_light / --pbr / --ground
30 帧 ~1-2 分钟。

[目的 C] 汇报 / 幻灯片视频（流畅 + 快速）
--------------------------------------------------------------------------------
  默认视频模式，不加任何画质参数（走 fast_render 老路径最快）
  --width 720 --height 1080 --fps 30
1000 帧 ~2-5 分钟。

[目的 D] Debug / 数据检查（只看动作正确性）
--------------------------------------------------------------------------------
  --mode skeleton --no_fingers （25 关节全身骨架，整体更清晰）
  --width 720 --height 1080
  快速扫查：--export_png out/ --sample_interval 10

[目的 E] 行走 / 大位移动作（角色沿轨迹移动）
--------------------------------------------------------------------------------
  必须加 --keep_transl，否则角色被锁在原点
  bundle 模式下想看"行走轨迹"效果更要加

--------------------------------------------------------------------------------
参数之间的相互作用与陷阱
--------------------------------------------------------------------------------
✔ --ssaa N 把渲染分辨率放大 N 倍，显存/耗时 ~ N²。ssaa=2 最划算；ssaa=3 慎用；ssaa=4 基本不现实。

✔ --better_lighting 和 --rim_light 建议同时开（标准三点光）。单独开 --rim_light 会导致暗面依然死黑。

✔ --ground 依赖 directional-light 阴影，不配 --better_lighting 阴影会非常硬，审美上不推荐单用。

✔ --color 优先级高于 --color_hands。同时传时 --color_hands 被忽略。

✔ 普通视频 / 普通 PNG 序列（未传 --bundle，且未同时传 --export_png + --clip_*）走 fast_render 老路径，
  以下参数完全无效：--ssaa / --better_lighting / --rim_light / --hq_skeleton / --pbr / --ground。
  要让视频享受高画质，只能先 --export_png + --clip_* 导出 PNG，再用 ffmpeg 合成视频。

✔ --bundle 与 --export_png / --clip_* 是互斥路径：--bundle 优先，其它分支会被跳过。

--------------------------------------------------------------------------------
bundle_length 推荐区间（按动作幅度）
--------------------------------------------------------------------------------
静态动作（挥手 / 指向）       : 20-30 帧（~0.7-1.0s）
中等幅度（走几步 / 转身）     : 30-50 帧
完整句子级别的动作            : 50-90 帧
> 90 帧不建议：肢体叠加过密，看不清结构。

叠影效果不理想时的调参顺序：
  1. 先降低 bundle_length
  2. 再调低 alpha_min（函数内常量，默认 0.12）
  3. 最后提升 --width / --height / --ssaa

--------------------------------------------------------------------------------
颜色十六进制参考（--color 直接填，不带 #）
--------------------------------------------------------------------------------
DCDCDC    浅灰（默认 mesh 色，统一感最强）
FFFFFF    纯白（适合深色背景 / 论文黑底）
E8E1D4    米白（柔和，暖调，不刺眼）
B0C4DE    淡蓝灰（冷调，科研风）
FF6B6B    珊瑚红（强调版本对比时用）
333333    深灰（适合浅色背景下剪影式展示）

================================================================================
典型使用示例
================================================================================

# 1) 最简：渲染成 mesh mp4
python scripts/render_npz.py file.npz

# 2) 带音频的骨架视频
python scripts/render_npz.py file.npz --mode skeleton --audio a.wav --output out.mp4

# 3) 论文主图：骨架 bundle，30 帧，全画质套餐
python scripts/render_npz.py file.npz --mode skeleton --bundle \
    --bundle_start 0 --bundle_length 30 \
    --ssaa 2 --better_lighting --rim_light --hq_skeleton --ground \
    --bundle_output fig1_bundle.png

# 4) Mesh bundle + PBR，出彩图
python scripts/render_npz.py file.npz --mode mesh --bundle \
    --bundle_length 30 --ssaa 2 --better_lighting --rim_light --pbr --ground

# 5) 导出一段 clip 的每一帧 PNG（做动图 / 幻灯片）
python scripts/render_npz.py file.npz --export_png ./pngs/ \
    --clip_start 100 --clip_length 60 \
    --ssaa 2 --better_lighting --rim_light --hq_skeleton

# 6) 对比图：同一序列同时出干净版和加噪版
python scripts/render_npz.py file.npz --export_png ./pngs/ --render_noise \
    --clip_start 0 --clip_length 30
================================================================================
"""

import argparse
import os
import shutil
import tempfile

import imageio
import numpy as np
import pyrender
import smplx
import torch
import trimesh
import wget
from PIL import Image

from diffusion.utils.fast_render import (
    args as render_args,
    generate_silent_videos_no_gt,
    generate_silent_videos_skeleton,
    export_png_sequence,
    add_audio_to_video,
    create_pose_camera,
    create_pose_light,
    create_scene_with_mesh,
    create_scene_with_skeleton,
    SMPLX_PARENTS,
    _joint_color,
)


def smplx_forward(model, poses, betas, expressions, trans):
    n = poses.shape[0]
    output = model(
        betas=betas,
        transl=trans,
        expression=expressions[:n],
        jaw_pose=poses[:, 66:69],
        global_orient=poses[:, :3],
        body_pose=poses[:, 3:21 * 3 + 3],
        left_hand_pose=poses[:, 25 * 3:40 * 3],
        right_hand_pose=poses[:, 40 * 3:55 * 3],
        leye_pose=poses[:, 69:72],
        reye_pose=poses[:, 72:75],
        return_verts=True,
    )
    vertices = output["vertices"].cpu().numpy()
    joints = output["joints"].cpu().numpy()[:, :55, :]
    return vertices, joints


def _apply_better_lighting(scene):
    """Add ambient + opposite-side fill light to soften single-directional lighting."""
    scene.ambient_light = np.array([0.35, 0.35, 0.35], dtype=np.float32)
    # Fill light: opposite side, lower intensity
    fill_pose = create_pose_light(angle_deg=30)
    fill_pose[0, 3] = -1.0  # offset to the other side
    fill_light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=1.8)
    scene.add(fill_light, pose=fill_pose)


def _apply_rim_light(scene):
    """Add a back-top rim light to outline silhouette."""
    rim_pose = create_pose_light(angle_deg=-150)  # from behind-above, shining toward camera
    rim_pose[2, 3] = -3.0  # placed behind subject (aesthetic; directional light only uses rotation)
    rim_light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2.5)
    scene.add(rim_light, pose=rim_pose)


def _build_skeleton_trimesh(joints, skip_fingers, hq_skeleton,
                             joint_radius=0.015, bone_radius=0.008):
    """Build skeleton geometry as a single uncolored trimesh (used for shadow flattening)."""
    sphere_count = (24, 24) if hq_skeleton else (8, 8)
    cylinder_sections = 24 if hq_skeleton else 16
    n_joints = 25 if skip_fingers else min(len(joints), 55)
    parts = []
    for i in range(n_joints):
        r = joint_radius * 0.5 if i >= 25 else joint_radius
        sphere = trimesh.creation.uv_sphere(radius=r, count=sphere_count)
        sphere.apply_translation(joints[i])
        parts.append(sphere)
        pid = SMPLX_PARENTS[i]
        if pid == -1 or pid >= n_joints:
            continue
        seg = np.array([joints[pid], joints[i]])
        if np.linalg.norm(seg[1] - seg[0]) < 1e-6:
            continue
        br = bone_radius * 0.5 if i >= 25 else bone_radius
        cyl = trimesh.creation.cylinder(radius=br, segment=seg, sections=cylinder_sections)
        parts.append(cyl)
    return trimesh.util.concatenate(parts)


def _add_shadow_mesh(scene, vertices, faces, ground_y, squash=0.01, alpha=0.35):
    """Fake drop shadow by squashing the geometry's Y down to ~ground_y and rendering it
    as a semi-transparent dark mesh. Avoids pyrender's unstable SHADOWS_DIRECTIONAL pass."""
    shadow_verts = np.asarray(vertices, dtype=np.float32).copy()
    shadow_verts[..., 1] = ground_y + (shadow_verts[..., 1] - ground_y) * squash
    shadow_tm = trimesh.Trimesh(vertices=shadow_verts, faces=faces, process=False)
    shadow_material = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=[0.04, 0.04, 0.04, alpha],
        metallicFactor=0.0,
        roughnessFactor=1.0,
        alphaMode='BLEND',
    )
    scene.add(pyrender.Mesh.from_trimesh(shadow_tm, smooth=False, material=shadow_material))


def _make_pbr_material():
    return pyrender.MetallicRoughnessMaterial(
        baseColorFactor=[1.0, 1.0, 1.0, 1.0],
        roughnessFactor=0.55,
        metallicFactor=0.0,
    )


def _build_mesh_scene_hq(vertices, faces, pose_camera, pose_light, pbr, vertex_colors=None):
    if vertex_colors is None:
        vertex_colors = render_args.get('mesh_color') or [220, 220, 220, 255]
    trimesh_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_colors=vertex_colors)
    material = _make_pbr_material() if pbr else None
    mesh = pyrender.Mesh.from_trimesh(trimesh_mesh, smooth=True, material=material)
    scene = pyrender.Scene(bg_color=[0, 0, 0, 0])
    scene.add(mesh)
    scene.add(pyrender.OrthographicCamera(xmag=1.0, ymag=1.0), pose=pose_camera)
    scene.add(pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=4.0), pose=pose_light)
    return scene


# SMPLX arm joints (shoulder + elbow + wrist + all hand joints)
LEFT_ARM_JOINTS = frozenset({16, 18, 20} | set(range(25, 40)))
RIGHT_ARM_JOINTS = frozenset({17, 19, 21} | set(range(40, 55)))


def _resolve_part_color(i, skeleton_colors):
    """skeleton_colors: tuple (body_rgba, left_arm_rgba, right_arm_rgba) or None → use fast_render default.
    SMPLX joint mapping: left arm = shoulder/elbow/wrist/hand, right arm symmetric, rest = body."""
    if skeleton_colors is None:
        return _joint_color(i)
    body_c, left_c, right_c = skeleton_colors
    if i in LEFT_ARM_JOINTS:
        return np.asarray(left_c, dtype=np.uint8)
    if i in RIGHT_ARM_JOINTS:
        return np.asarray(right_c, dtype=np.uint8)
    return np.asarray(body_c, dtype=np.uint8)


def _build_skeleton_scene_hq(joints, pose_camera, pose_light, skip_fingers, hq_skeleton, pbr,
                              joint_radius=0.015, bone_radius=0.008, skeleton_colors=None):
    sphere_count = (24, 24) if hq_skeleton else (8, 8)
    cylinder_sections = 24 if hq_skeleton else 16
    n_joints = 25 if skip_fingers else min(len(joints), 55)
    parts = []
    for i in range(n_joints):
        color = _resolve_part_color(i, skeleton_colors)
        r = joint_radius * 0.5 if i >= 25 else joint_radius
        sphere = trimesh.creation.uv_sphere(radius=r, count=sphere_count)
        sphere.apply_translation(joints[i])
        sphere.visual.vertex_colors = np.tile(color, (len(sphere.vertices), 1))
        parts.append(sphere)
        pid = SMPLX_PARENTS[i]
        if pid == -1 or pid >= n_joints:
            continue
        seg = np.array([joints[pid], joints[i]])
        length = np.linalg.norm(seg[1] - seg[0])
        if length < 1e-6:
            continue
        br = bone_radius * 0.5 if i >= 25 else bone_radius
        cyl = trimesh.creation.cylinder(radius=br, segment=seg, sections=cylinder_sections)
        cyl.visual.vertex_colors = np.tile(color, (len(cyl.vertices), 1))
        parts.append(cyl)
    mesh_tm = trimesh.util.concatenate(parts)
    material = _make_pbr_material() if pbr else None
    mesh = pyrender.Mesh.from_trimesh(mesh_tm, smooth=False, material=material)
    scene = pyrender.Scene(bg_color=[0, 0, 0, 0])
    scene.add(mesh)
    scene.add(pyrender.OrthographicCamera(xmag=1.0, ymag=1.0), pose=pose_camera)
    scene.add(pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=4.0), pose=pose_light)
    return scene


def _build_scene(mode, vertices_i, faces, joints_i, pose_camera, pose_light,
                 skip_fingers, hq_skeleton, pbr, shadow_ground_y=None,
                 skeleton_colors=None, mesh_vertex_colors=None):
    """Use HQ local builder only when any quality flag is on; otherwise fall back to fast_render.
    If shadow_ground_y is provided, also add an invisible-ground drop shadow.
    If skeleton_colors is provided (and mode=skeleton), force local builder with per-part colors.
    If mesh_vertex_colors is provided (and mode=mesh), force local builder with per-vertex colors."""
    if mode == "mesh":
        use_mesh_local = hq_skeleton or pbr or mesh_vertex_colors is not None
        if use_mesh_local:
            scene = _build_mesh_scene_hq(vertices_i, faces, pose_camera, pose_light, pbr,
                                          vertex_colors=mesh_vertex_colors)
        else:
            uniform_color = render_args.get('mesh_color') or [220, 220, 220, 255]
            scene = create_scene_with_mesh(vertices_i, faces, uniform_color, pose_camera, pose_light)
        if shadow_ground_y is not None:
            _add_shadow_mesh(scene, vertices_i, faces, shadow_ground_y)
        return scene
    # skeleton
    use_skel_local = hq_skeleton or pbr or skeleton_colors is not None
    if use_skel_local:
        scene = _build_skeleton_scene_hq(joints_i, pose_camera, pose_light, skip_fingers,
                                          hq_skeleton, pbr, skeleton_colors=skeleton_colors)
    else:
        scene = create_scene_with_skeleton(joints_i, pose_camera, pose_light, skip_fingers=skip_fingers)
    if shadow_ground_y is not None:
        skeleton_tm = _build_skeleton_trimesh(joints_i, skip_fingers, hq_skeleton)
        _add_shadow_mesh(scene, skeleton_tm.vertices, skeleton_tm.faces, shadow_ground_y)
    return scene


def _downscale_rgba(image_u8, target_w, target_h):
    """High-quality Lanczos downscale to (target_w, target_h)."""
    if image_u8.shape[1] == target_w and image_u8.shape[0] == target_h:
        return image_u8
    img = Image.fromarray(image_u8, mode="RGBA")
    img = img.resize((target_w, target_h), Image.Resampling.LANCZOS)
    return np.asarray(img)


def export_clip_pngs(vertices_all, faces, joints_all, output_dir, start, length,
                      sample_interval, mode, skip_fingers, width, height,
                      ssaa=1, better_lighting=False, rim_light=False,
                      hq_skeleton=False, pbr=False, ground=False,
                      skeleton_colors=None, mesh_vertex_colors=None):
    """Export PNGs for a specific frame range, keeping original frame indices in filenames."""
    n_total = len(vertices_all) if mode == "mesh" else len(joints_all)
    end = min(start + length, n_total)
    if start >= end:
        raise ValueError(f"No frames in clip range [{start}, {end})")

    os.makedirs(output_dir, exist_ok=True)
    render_w, render_h = width * ssaa, height * ssaa
    renderer = pyrender.OffscreenRenderer(render_w, render_h)
    pose_camera = create_pose_camera(angle_deg=-2)
    pose_light = create_pose_light(angle_deg=-30)
    rgba_flags = pyrender.constants.RenderFlags.RGBA
    ground_y = None
    if ground:
        y_src = vertices_all if mode == "mesh" else joints_all
        ground_y = float(np.min(y_src[start:end, :, 1])) - 0.005

    saved = 0
    for i in range(start, end, sample_interval):
        scene = _build_scene(mode, vertices_all[i] if mode == "mesh" else None,
                             faces, joints_all[i] if mode != "mesh" else None,
                             pose_camera, pose_light, skip_fingers, hq_skeleton, pbr,
                             shadow_ground_y=ground_y, skeleton_colors=skeleton_colors,
                             mesh_vertex_colors=mesh_vertex_colors)
        if better_lighting:
            _apply_better_lighting(scene)
        if rim_light:
            _apply_rim_light(scene)

        color, _ = renderer.render(scene, flags=rgba_flags)
        if ssaa > 1:
            color = _downscale_rgba(color, width, height)
        out_path = os.path.join(output_dir, f"frame_{i:06d}.png")
        imageio.imwrite(out_path, color)
        saved += 1

    renderer.delete()
    print(f"Exported {saved} PNGs to {output_dir}")


def render_bundle(vertices_all, faces, joints_all, mode, skip_fingers,
                   start, length, width, height, ssaa=1, better_lighting=False,
                   rim_light=False, hq_skeleton=False, pbr=False, ground=False,
                   skeleton_colors=None, mesh_vertex_colors=None):
    """Render a motion bundle: overlay multiple frames with increasing opacity."""
    n_total = len(vertices_all) if mode == "mesh" else len(joints_all)
    end = min(start + length, n_total)
    frame_indices = list(range(start, end))
    N = len(frame_indices)
    if N == 0:
        raise ValueError(f"No frames in range [{start}, {end})")

    render_w, render_h = width * ssaa, height * ssaa
    renderer = pyrender.OffscreenRenderer(render_w, render_h)
    pose_camera = create_pose_camera(angle_deg=-2)
    pose_light = create_pose_light(angle_deg=-30)
    rgba_flags = pyrender.constants.RenderFlags.RGBA
    ground_y = None
    if ground:
        y_src = vertices_all if mode == "mesh" else joints_all
        ground_y = float(np.min(y_src[start:end, :, 1])) - 0.005

    # Alpha schedule: linearly from 0.12 to 1.0
    alpha_min, alpha_max = 0.12, 1.0
    if N == 1:
        alphas = [1.0]
    else:
        alphas = [alpha_min + (alpha_max - alpha_min) * i / (N - 1) for i in range(N)]

    canvas = np.zeros((render_h, render_w, 4), dtype=np.float32)

    for idx, frame_i in enumerate(frame_indices):
        scene = _build_scene(mode,
                             vertices_all[frame_i] if mode == "mesh" else None,
                             faces,
                             joints_all[frame_i] if mode != "mesh" else None,
                             pose_camera, pose_light, skip_fingers, hq_skeleton, pbr,
                             shadow_ground_y=ground_y, skeleton_colors=skeleton_colors,
                             mesh_vertex_colors=mesh_vertex_colors)
        if better_lighting:
            _apply_better_lighting(scene)
        if rim_light:
            _apply_rim_light(scene)

        color, _ = renderer.render(scene, flags=rgba_flags)
        fg = color.astype(np.float32) / 255.0
        fg[..., 3] *= alphas[idx]

        a_fg = fg[..., 3:4]
        a_bg = canvas[..., 3:4]
        a_out = a_fg + a_bg * (1.0 - a_fg)
        canvas[..., :3] = (fg[..., :3] * a_fg + canvas[..., :3] * a_bg * (1.0 - a_fg)) / (a_out + 1e-8)
        canvas[..., 3:4] = a_out

        if (idx + 1) % 10 == 0 or idx == N - 1:
            print(f"  composited frame {idx + 1}/{N}")

    renderer.delete()
    out = (canvas * 255).clip(0, 255).astype(np.uint8)
    if ssaa > 1:
        out = _downscale_rgba(out, width, height)
    return out


def main():
    parser = argparse.ArgumentParser(description="Render SMPLX NPZ to video")
    parser.add_argument("npz_path", type=str, help="Path to the NPZ file")
    parser.add_argument("--mode", type=str, default="mesh", choices=["mesh", "skeleton", "face"],
                        help="Render mode: mesh / skeleton / face (close-up head only)")
    parser.add_argument("--face_zoom", type=float, default=7.0,
                        help="Vertex scale factor for --mode face (default 7.0). "
                             "Larger → bigger face; smaller → smaller face.")
    parser.add_argument("--audio", type=str, default=None, help="Optional audio file to attach")
    parser.add_argument("--output", type=str, default=None, help="Output video path (default: <npz_name>.mp4)")
    parser.add_argument("--model_folder", type=str, default="./emage_evaltools/smplx_models/",
                        help="Path to SMPLX model folder")
    parser.add_argument("--keep_transl", action="store_true",
                        help="Keep per-frame translation (default: fix to first frame)")
    parser.add_argument("--no_center", action="store_true",
                        help="Disable X-axis centering")
    parser.add_argument("--no_fingers", action="store_true",
                        help="Skip finger joints in skeleton mode")
    parser.add_argument("--export_png", type=str, default=None,
                        help="Export transparent PNGs to this directory (skips video)")
    parser.add_argument("--sample_interval", type=int, default=30,
                        help="Frame sampling interval for PNG export (default: 30)")
    parser.add_argument("--clip_start", type=int, default=None,
                        help="Clip start frame (use with --export_png to limit range)")
    parser.add_argument("--clip_length", type=int, default=None,
                        help="Clip length in frames (use with --export_png to limit range)")
    parser.add_argument("--render_noise", action="store_true",
                        help="Also render a fully-noised version (gaussian noise on poses)")
    parser.add_argument("--noise_scale", type=float, default=1.0,
                        help="Scale of gaussian noise applied to poses (default: 1.0)")
    parser.add_argument("--bundle", action="store_true",
                        help="Enable motion bundle mode (overlay frames into one PNG)")
    parser.add_argument("--bundle_start", type=int, default=0,
                        help="Start frame for bundle (default: 0)")
    parser.add_argument("--bundle_length", type=int, default=30,
                        help="Number of frames for bundle (default: 30)")
    parser.add_argument("--bundle_output", type=str, default="bundle.png",
                        help="Output PNG path for bundle (default: bundle.png)")
    parser.add_argument("--color", type=str, default=None,
                        help="Shortcut: set body/left_hand/right_hand all to this HEX (e.g. DCDCDC). "
                             "Overridden by per-part --body_color/--left_hand_color/--right_hand_color.")
    parser.add_argument("--body_color", type=str, default=None,
                        help="Body color HEX (torso/head/legs). Default: DCDCDC light gray.")
    parser.add_argument("--left_arm_color", type=str, default=None,
                        help="Left arm color HEX (shoulder+elbow+wrist+hand, joints 16/18/20 + 25-39). "
                             "Default: follow --body_color.")
    parser.add_argument("--right_arm_color", type=str, default=None,
                        help="Right arm color HEX (shoulder+elbow+wrist+hand, joints 17/19/21 + 40-54). "
                             "Default: follow --body_color.")
    parser.add_argument("--color_hands", action="store_true",
                        help="Backward-compat preset: emage three-color (body blue + left green + right red). "
                             "Only active when none of --color/--body_color/--left_hand_color/--right_hand_color is set.")
    parser.add_argument("--fps", type=int, default=30, help="Video FPS")
    parser.add_argument("--width", type=int, default=1440, help="Render width")
    parser.add_argument("--height", type=int, default=2160, help="Render height")
    parser.add_argument("--ssaa", type=int, default=1,
                        help="Supersample factor: render at width*ssaa x height*ssaa then Lanczos downscale "
                             "(only for --bundle and --export_png+clip paths). Default 1 (off).")
    parser.add_argument("--better_lighting", action="store_true",
                        help="Add ambient + fill light to soften single-directional lighting "
                             "(only for --bundle and --export_png+clip paths).")
    parser.add_argument("--rim_light", action="store_true",
                        help="Add a back-top rim light for silhouette separation "
                             "(combine with --better_lighting for three-point lighting).")
    parser.add_argument("--hq_skeleton", action="store_true",
                        help="Skeleton mode: raise sphere tessellation from 8x8 to 24x24 "
                             "and cylinder sections to 24. Eliminates low-poly facets.")
    parser.add_argument("--pbr", action="store_true",
                        help="Use PBR material (MetallicRoughness, roughness=0.55) for subtle specular. "
                             "Most visible on mesh mode.")
    parser.add_argument("--ground", action="store_true",
                        help="Add a ground plane below the subject and cast directional-light shadows "
                             "(light gray opaque ground, 4x4 units). Only for --bundle and --export_png+clip.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    # Override fast_render global settings
    render_args['render_video_fps'] = args.fps
    render_args['render_video_width'] = args.width
    render_args['render_video_height'] = args.height

    # --- Color resolution (per-part > --color > --color_hands > default) ---
    def _hex_to_rgba(s):
        s = s.lstrip('#')
        return [int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16), 255]

    any_part = bool(args.body_color or args.left_arm_color or args.right_arm_color)
    skeleton_colors = None  # None → use fast_render's _joint_color default (emage three-color preset)
    part_colors_set = False  # whether body/left/right were resolved to explicit rgba

    if any_part or args.color:
        base = _hex_to_rgba(args.color) if args.color else [220, 220, 220, 255]
        body_rgba = _hex_to_rgba(args.body_color) if args.body_color else base
        left_rgba = _hex_to_rgba(args.left_arm_color) if args.left_arm_color else body_rgba
        right_rgba = _hex_to_rgba(args.right_arm_color) if args.right_arm_color else body_rgba
        skeleton_colors = (body_rgba, left_rgba, right_rgba)
        render_args['mesh_color'] = body_rgba  # fallback for any uniform-mesh code path
        part_colors_set = True
    elif args.color_hands:
        # Emage three-color preset: leave skeleton_colors=None and mesh_color unset → fast_render default.
        pass
    else:
        # Default: everything unified light gray.
        body_rgba = left_rgba = right_rgba = [220, 220, 220, 255]
        skeleton_colors = (body_rgba, body_rgba, body_rgba)
        render_args['mesh_color'] = body_rgba
        part_colors_set = True

    # Output path
    if args.output is None:
        base = os.path.splitext(os.path.basename(args.npz_path))[0]
        args.output = f"{base}.mp4"

    # --- Load NPZ ---
    print(f"Loading {args.npz_path} ...")
    data = np.load(args.npz_path, allow_pickle=True)
    poses = torch.from_numpy(data["poses"]).float().to(args.device)
    betas = torch.from_numpy(data["betas"]).float().to(args.device)
    expressions = torch.from_numpy(data["expressions"]).float().to(args.device)
    trans = torch.from_numpy(data["trans"]).float().to(args.device)

    n = poses.shape[0]
    if betas.dim() == 1:
        betas = betas.unsqueeze(0).expand(n, -1)
    if betas.shape[0] != n:
        betas = betas[:1].expand(n, -1)

    if not args.keep_transl:
        trans = trans[0:1].expand(n, -1)

    # --- Load SMPLX model (auto-download if missing) ---
    smplx_npz_path = os.path.join(args.model_folder, "smplx", "SMPLX_NEUTRAL_2020.npz")
    if not os.path.exists(smplx_npz_path):
        os.makedirs(os.path.join(args.model_folder, "smplx"), exist_ok=True)
        print(f"Downloading SMPLX model to {smplx_npz_path} ...")
        wget.download(
            "https://huggingface.co/spaces/H-Liu1997/EMAGE/resolve/main/EMAGE/smplx_models/smplx/SMPLX_NEUTRAL_2020.npz",
            smplx_npz_path,
        )
        print()

    faces = np.load(smplx_npz_path, allow_pickle=True)["f"]
    print("Creating SMPLX model ...")
    model = smplx.create(
        args.model_folder, model_type="smplx", gender="NEUTRAL_2020",
        use_face_contour=False, num_betas=300, num_expression_coeffs=100,
        ext="npz", use_pca=False,
    ).to(args.device).eval()

    # --- Per-vertex color segmentation for mesh mode (using LBS argmax → dominant joint) ---
    # Left arm  = shoulder(16) + elbow(18) + wrist(20) + hand(25-39)
    # Right arm = shoulder(17) + elbow(19) + wrist(21) + hand(40-54)
    # Body      = everything else
    # Face mode: skip per-part coloring (whole mesh is essentially just the head).
    mesh_vertex_colors = None
    if part_colors_set and args.mode == "mesh":
        lbs = model.lbs_weights.detach().cpu().numpy()  # (N_verts, 55)
        vert_joint = np.argmax(lbs, axis=1)             # (N_verts,)
        left_arm_ids  = np.array(list(LEFT_ARM_JOINTS))
        right_arm_ids = np.array(list(RIGHT_ARM_JOINTS))
        left_mask  = np.isin(vert_joint, left_arm_ids)
        right_mask = np.isin(vert_joint, right_arm_ids)
        body_mask  = ~(left_mask | right_mask)
        mesh_vertex_colors = np.zeros((lbs.shape[0], 4), dtype=np.uint8)
        mesh_vertex_colors[body_mask]  = body_rgba
        mesh_vertex_colors[left_mask]  = left_rgba
        mesh_vertex_colors[right_mask] = right_rgba

    # --- SMPLX forward (chunked to avoid OOM) ---
    print(f"Computing mesh for {n} frames ...")
    chunk_size = 128
    vertices_list, joints_list = [], []
    with torch.no_grad():
        for i in range(0, n, chunk_size):
            end = min(i + chunk_size, n)
            verts, jts = smplx_forward(model, poses[i:end], betas[i:end], expressions[i:end], trans[i:end])
            vertices_list.append(verts)
            joints_list.append(jts)
    vertices_all = np.concatenate(vertices_list, axis=0)
    joints_all = np.concatenate(joints_list, axis=0)

    # --- Face mode: replace vertices with neutral-body + face-articulation mesh ---
    # Strategy: zero body/hands/eyes/global_orient, keep jaw_pose + expression.
    # Center per frame around the eyes midpoint (joints 23/24, visually on the face)
    # rather than joint 15 (head — sits at base of skull, causes face to drift upward).
    # Scale by face_zoom, landing eyes midpoint at (0, 1, 0) — the ortho camera's
    # look-at point for xmag=ymag=1.0 at camera Y=1.0. Works for any NPZ regardless
    # of trans / betas.
    if args.mode == "face":
        print(f"Computing face-only mesh (face_zoom={args.face_zoom}) ...")
        zero_poses = torch.zeros_like(poses)
        face_verts_list, face_center_list = [], []
        with torch.no_grad():
            for i in range(0, n, chunk_size):
                end = min(i + chunk_size, n)
                out = model(
                    betas=betas[i:end], transl=trans[i:end],
                    expression=expressions[i:end],
                    jaw_pose=poses[i:end, 66:69],
                    global_orient=zero_poses[i:end, :3],
                    body_pose=zero_poses[i:end, 3:66],
                    left_hand_pose=zero_poses[i:end, 75:120],
                    right_hand_pose=zero_poses[i:end, 120:165],
                    leye_pose=zero_poses[i:end, 69:72],
                    reye_pose=zero_poses[i:end, 72:75],
                    return_verts=True,
                )
                face_verts_list.append(out["vertices"].cpu().numpy())
                eyes_mid = (out["joints"][:, 23] + out["joints"][:, 24]) * 0.5
                face_center_list.append(eyes_mid.cpu().numpy())
        face_verts = np.concatenate(face_verts_list, axis=0)                # (N, V, 3)
        face_center = np.concatenate(face_center_list, axis=0)[:, None, :]  # (N, 1, 3)
        camera_lookat = np.array([[[0.0, 1.0, 0.0]]], dtype=face_verts.dtype)
        vertices_all = (face_verts - face_center) * args.face_zoom + camera_lookat

    # --- Noised version (gaussian noise on poses only, keep trans to stay in frame) ---
    noised_verts_all, noised_joints_all = None, None
    if args.render_noise:
        print(f"Computing noised mesh (noise_scale={args.noise_scale}) ...")
        noised_poses = torch.randn_like(poses) * args.noise_scale
        noised_expressions = torch.randn_like(expressions) * args.noise_scale
        nv_list, nj_list = [], []
        with torch.no_grad():
            for i in range(0, n, chunk_size):
                end = min(i + chunk_size, n)
                nv, nj = smplx_forward(
                    model, noised_poses[i:end], betas[i:end],
                    noised_expressions[i:end], trans[i:end],
                )
                nv_list.append(nv)
                nj_list.append(nj)
        noised_verts_all = np.concatenate(nv_list, axis=0)
        noised_joints_all = np.concatenate(nj_list, axis=0)

        if args.mode == "face":
            # Re-forward noised version with the same body-zeroed setup + noised jaw/expression,
            # then re-center around per-frame head joint → camera look-at.
            zero_poses = torch.zeros_like(poses)
            nfv_list, nfc_list = [], []
            with torch.no_grad():
                for i in range(0, n, chunk_size):
                    end = min(i + chunk_size, n)
                    out = model(
                        betas=betas[i:end], transl=trans[i:end],
                        expression=noised_expressions[i:end],
                        jaw_pose=noised_poses[i:end, 66:69],
                        global_orient=zero_poses[i:end, :3],
                        body_pose=zero_poses[i:end, 3:66],
                        left_hand_pose=zero_poses[i:end, 75:120],
                        right_hand_pose=zero_poses[i:end, 120:165],
                        leye_pose=zero_poses[i:end, 69:72],
                        reye_pose=zero_poses[i:end, 72:75],
                        return_verts=True,
                    )
                    nfv_list.append(out["vertices"].cpu().numpy())
                    eyes_mid = (out["joints"][:, 23] + out["joints"][:, 24]) * 0.5
                    nfc_list.append(eyes_mid.cpu().numpy())
            nfv = np.concatenate(nfv_list, axis=0)
            nfc = np.concatenate(nfc_list, axis=0)[:, None, :]
            camera_lookat = np.array([[[0.0, 1.0, 0.0]]], dtype=nfv.dtype)
            noised_verts_all = (nfv - nfc) * args.face_zoom + camera_lookat

    # --- X-axis centering (fix offset-to-the-right) ---
    # Face mode positions head via fixed translate; never re-center.
    if not args.no_center and args.mode != "face":
        if args.mode == "mesh":
            x_center = np.median(vertices_all[:, :, 0])
            vertices_all[:, :, 0] -= x_center
            if noised_verts_all is not None:
                noised_verts_all[:, :, 0] -= x_center
        else:
            x_center = np.median(joints_all[:, :, 0])
            joints_all[:, :, 0] -= x_center
            if noised_joints_all is not None:
                noised_joints_all[:, :, 0] -= x_center

    # Face mode is rendered as mesh (the face-only vertices) downstream.
    effective_mode = "mesh" if args.mode == "face" else args.mode
    # Ground shadow is meaningless for face close-ups (floor not in frame). Force off.
    use_ground = args.ground and args.mode != "face"
    if args.ground and args.mode == "face":
        print("Note: --ground ignored for --mode face (floor not in frame).")

    # --- Bundle mode ---
    if args.bundle:
        print(f"Rendering motion bundle: frames [{args.bundle_start}, {args.bundle_start + args.bundle_length}) ...")
        result = render_bundle(
            vertices_all=vertices_all, faces=faces, joints_all=joints_all,
            mode=effective_mode, skip_fingers=args.no_fingers,
            start=args.bundle_start, length=args.bundle_length,
            width=args.width, height=args.height,
            ssaa=args.ssaa, better_lighting=args.better_lighting,
            rim_light=args.rim_light, hq_skeleton=args.hq_skeleton, pbr=args.pbr,
            ground=use_ground, skeleton_colors=skeleton_colors,
            mesh_vertex_colors=mesh_vertex_colors,
        )
        imageio.imwrite(args.bundle_output, result)
        print(f"Done! Bundle saved to {args.bundle_output}")
        return

    # --- PNG export mode ---
    if args.export_png:
        use_clip = args.clip_start is not None or args.clip_length is not None
        if use_clip:
            clip_start = args.clip_start if args.clip_start is not None else 0
            clip_length = args.clip_length if args.clip_length is not None else (n - clip_start)
            # Default to every frame in clip mode unless user set sample_interval explicitly via a non-default value
            clip_interval = args.sample_interval if args.sample_interval != 30 else 1
            print(f"Exporting clip PNGs frames [{clip_start}, {clip_start + clip_length}) "
                  f"every {clip_interval} frame(s) to {args.export_png} ...")
            export_clip_pngs(
                vertices_all=vertices_all, faces=faces, joints_all=joints_all,
                output_dir=args.export_png, start=clip_start, length=clip_length,
                sample_interval=clip_interval, mode=effective_mode, skip_fingers=args.no_fingers,
                width=args.width, height=args.height,
                ssaa=args.ssaa, better_lighting=args.better_lighting,
                rim_light=args.rim_light, hq_skeleton=args.hq_skeleton, pbr=args.pbr,
                ground=use_ground, skeleton_colors=skeleton_colors,
                mesh_vertex_colors=mesh_vertex_colors,
            )
            if args.render_noise:
                noised_png_dir = args.export_png.rstrip("/") + "_noised"
                print(f"Exporting noised clip PNGs to {noised_png_dir} ...")
                export_clip_pngs(
                    vertices_all=noised_verts_all, faces=faces, joints_all=noised_joints_all,
                    output_dir=noised_png_dir, start=clip_start, length=clip_length,
                    sample_interval=clip_interval, mode=effective_mode, skip_fingers=args.no_fingers,
                    width=args.width, height=args.height,
                    ssaa=args.ssaa, better_lighting=args.better_lighting,
                    rim_light=args.rim_light, hq_skeleton=args.hq_skeleton, pbr=args.pbr,
                    skeleton_colors=skeleton_colors,
                    mesh_vertex_colors=mesh_vertex_colors,
                )
            print("Done!")
            return

        print(f"Exporting PNGs every {args.sample_interval} frames to {args.export_png} ...")
        export_png_sequence(
            vertices_all=vertices_all, faces=faces, joints_all=joints_all,
            output_dir=args.export_png, sample_interval=args.sample_interval,
            mode=effective_mode, skip_fingers=args.no_fingers,
            render_width=args.width, render_height=args.height,
        )
        if args.render_noise:
            noised_png_dir = args.export_png.rstrip("/") + "_noised"
            print(f"Exporting noised PNGs to {noised_png_dir} ...")
            export_png_sequence(
                vertices_all=noised_verts_all, faces=faces, joints_all=noised_joints_all,
                output_dir=noised_png_dir, sample_interval=args.sample_interval,
                mode=effective_mode, skip_fingers=args.no_fingers,
                render_width=args.width, render_height=args.height,
            )
        print("Done!")
        return

    # --- Video render ---
    print(f"Rendering {n} frames ({args.mode} mode) at {args.width}x{args.height} ...")
    tmpdir = tempfile.mkdtemp()
    try:
        seconds = n // args.fps
        if effective_mode == "mesh":
            silent_path = generate_silent_videos_no_gt(
                int(seconds * args.fps), vertices_all, faces, tmpdir,
            )
        else:
            silent_path = generate_silent_videos_skeleton(
                int(seconds * args.fps), joints_all, tmpdir,
                skip_fingers=args.no_fingers,
            )

        if args.audio:
            add_audio_to_video(silent_path, args.audio, args.output)
            os.remove(silent_path)
        else:
            shutil.move(silent_path, args.output)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    if args.render_noise:
        base, ext = os.path.splitext(args.output)
        noised_output = f"{base}_noised{ext}"
        print(f"Rendering noised version ...")
        tmpdir = tempfile.mkdtemp()
        try:
            if effective_mode == "mesh":
                silent_path = generate_silent_videos_no_gt(
                    int(seconds * args.fps), noised_verts_all, faces, tmpdir,
                )
            else:
                silent_path = generate_silent_videos_skeleton(
                    int(seconds * args.fps), noised_joints_all, tmpdir,
                    skip_fingers=args.no_fingers,
                )

            if args.audio:
                add_audio_to_video(silent_path, args.audio, noised_output)
                os.remove(silent_path)
            else:
                shutil.move(silent_path, noised_output)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
        print(f"Noised output: {noised_output}")

    print(f"Done! Output: {args.output}")


if __name__ == "__main__":
    main()
