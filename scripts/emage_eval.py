# 和on-the-fly eval的方法不同，这个是尽可能接近EMAGE的原始实现，以精度为第一目标
import os
import wget
import json
import hashlib
import smplx
import torch
import pandas as pd
import numpy as np
import argparse
import librosa
import sys
import warnings
from pathlib import Path
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.EMAGE.emage_evaltools.mertic import (
    FGD,
    BC as _UpstreamBC,
    L1div,
    LVDFace,
    MSEFace,
)  # 使用最原始的实现方式保证一致
from scripts.gesture_style_metrics import (
    GestureStyleAccumulator,
    STYLE_FEATURE_NAMES,
)
import diffusion.utils.rotation_conversions as rc


class BC(_UpstreamBC):
    """Keep upstream EMAGE BC while defining its empty-beat edge case."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.empty_audio_onset_clip_count = 0

    @staticmethod
    def GAHR(a, b, sigma):
        # Upstream divides by len(b). A valid speech clip can still contain no
        # detected onset after the two-second boundary trim. With no alignment
        # evidence, score this directed comparison as zero instead of crashing.
        if len(a) == 0 or len(b) == 0:
            return 0.0
        return _UpstreamBC.GAHR(a, b, sigma)

    def compute(self, onset_bt_rms, beat_vel, length=1, pose_fps=30):
        if len(onset_bt_rms) == 0:
            self.empty_audio_onset_clip_count += 1
        return super().compute(
            onset_bt_rms,
            beat_vel,
            length=length,
            pose_fps=pose_fps,
        )

    def avg(self):
        if self.counter <= 0:
            return 0.0
        return super().avg()


TALKSHOW_SPEAKER_IDS = {
    "oliver": 21,
    "chemistry": 22,
    "seth": 23,
    "conan": 24,
}


def _validate_talkshow_clip_id(clip_id, source):
    if "/" in clip_id or "\\" in clip_id:
        raise ValueError(
            f"{source}: TalkSHOW clip ID must not contain a path separator: "
            f"{clip_id!r}"
        )
    speaker, separator, sequence = clip_id.partition("__")
    if not separator or not sequence:
        raise ValueError(
            f"{source}: expected '<speaker>__<sequence>', got {clip_id!r}"
        )
    if speaker not in TALKSHOW_SPEAKER_IDS:
        raise ValueError(
            f"{source}: unknown TalkSHOW speaker {speaker!r}; expected one of "
            f"{list(TALKSHOW_SPEAKER_IDS)}"
        )
    return TALKSHOW_SPEAKER_IDS[speaker]


def _read_talkshow_manifest(path):
    manifest_path = Path(path).expanduser()
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"TalkSHOW manifest does not exist: {manifest_path}"
        )

    clip_ids = []
    first_line_by_id = {}
    with manifest_path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            clip_id = line.strip()
            if not clip_id or clip_id.startswith("#"):
                continue
            _validate_talkshow_clip_id(
                clip_id,
                f"{manifest_path}:{line_number}",
            )
            if clip_id in first_line_by_id:
                raise ValueError(
                    f"{manifest_path}:{line_number}: duplicate clip ID "
                    f"{clip_id!r}; first seen on line "
                    f"{first_line_by_id[clip_id]}"
                )
            first_line_by_id[clip_id] = line_number
            clip_ids.append(clip_id)

    if not clip_ids:
        raise ValueError(
            f"TalkSHOW manifest contains no clip IDs: {manifest_path}"
        )
    return clip_ids


def _discover_prefixed_ids(directory, prefix, suffix):
    return {
        path.name[len(prefix):-len(suffix)]
        for path in directory.glob(f"{prefix}*{suffix}")
        if path.is_file()
    }


def _discover_talkshow_samples(input_path, audio_path, manifest_path=None):
    prediction_dir = Path(input_path).expanduser()
    wav_dir = Path(audio_path).expanduser()
    if not prediction_dir.is_dir():
        raise FileNotFoundError(
            f"TalkSHOW NPZ directory does not exist: {prediction_dir}"
        )
    if not wav_dir.is_dir():
        raise FileNotFoundError(
            f"TalkSHOW audio directory does not exist: {wav_dir}"
        )

    if manifest_path is None:
        prediction_ids = _discover_prefixed_ids(
            prediction_dir,
            "res_",
            ".npz",
        )
        if not prediction_ids:
            suspicious = sorted(prediction_dir.glob("*res_*.npz"))
            hint = (
                f"; found non-canonical file {suspicious[0].name!r}"
                if suspicious
                else ""
            )
            raise FileNotFoundError(
                f"no canonical res_*.npz files in {prediction_dir}{hint}"
            )
        clip_ids = sorted(prediction_ids)

        ground_truth_ids = _discover_prefixed_ids(
            prediction_dir,
            "gt_",
            ".npz",
        )
        audio_ids = _discover_prefixed_ids(
            wav_dir,
            "gt_sr16000_",
            ".wav",
        )
        mismatch_messages = []
        for label, observed in (
            ("gt_*.npz", ground_truth_ids),
            ("gt_sr16000_*.wav", audio_ids),
        ):
            missing = sorted(prediction_ids - observed)
            extra = sorted(observed - prediction_ids)
            if missing:
                mismatch_messages.append(
                    f"{label} missing {len(missing)} clip(s), first={missing[:10]}"
                )
            if extra:
                mismatch_messages.append(
                    f"{label} has {len(extra)} clip(s) without prediction, "
                    f"first={extra[:10]}"
                )
        if mismatch_messages:
            raise ValueError(
                "TalkSHOW res/gt/audio clip sets do not match:\n  "
                + "\n  ".join(mismatch_messages)
                + "\nUse --talkshow-manifest only when an intentional subset "
                "has been frozen explicitly."
            )
    else:
        clip_ids = _read_talkshow_manifest(manifest_path)

    samples = []
    missing_paths = []
    for clip_id in clip_ids:
        speaker_id = _validate_talkshow_clip_id(
            clip_id,
            "TalkSHOW evaluation input",
        )
        paths = (
            prediction_dir / f"res_{clip_id}.npz",
            prediction_dir / f"gt_{clip_id}.npz",
            wav_dir / f"gt_sr16000_{clip_id}.wav",
        )
        for path in paths:
            if not path.is_file():
                missing_paths.append(path)
        samples.append((clip_id, speaker_id))

    if missing_paths:
        preview = "\n  ".join(str(path) for path in missing_paths[:20])
        remaining = len(missing_paths) - min(len(missing_paths), 20)
        suffix = f"\n  ... and {remaining} more" if remaining else ""
        raise FileNotFoundError(
            "TalkSHOW evaluation inputs are incomplete:\n  "
            f"{preview}{suffix}"
        )
    return samples


def _discover_beat_samples(split_file, persons):
    df = pd.read_csv(split_file)
    df = df[df["type"] == "test"]
    df = df[
        df["id"].apply(lambda value: str(value).split("_")[0]).isin(
            [str(person) for person in persons]
        )
    ]
    return [
        (str(file_id), int(str(file_id).split("_", 1)[0]))
        for file_id in df["id"].tolist()
    ]


parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input_path", type=str, required=True, help="Input to the npz paths.")
parser.add_argument("-o", "--output_path", type=str, required=True, help="Output path of metric computing results.")
parser.add_argument("-ia", "--input_audio_path", type=str, required=True, help="Input audio path.")
parser.add_argument(
    "-dp",
    "--download_path",
    type=str,
    default=os.environ.get(
        "EMAGE_EVALTOOLS_DIR",
        str(REPO_ROOT / "tools" / "EMAGE" / "emage_evaltools"),
    ),
    help="Directory containing the EMAGE evaluation assets.",
)
parser.add_argument(
    "--dataset",
    choices=("beat", "talkshow"),
    default="beat",
    help=(
        "Dataset-specific sample discovery. 'beat' preserves the original "
        "split-file/person behavior; 'talkshow' pairs canonical "
        "res_/gt_/gt_sr16000_ files."
    ),
)
parser.add_argument(
    "-sf",
    "--split_file",
    type=str,
    default=os.environ.get("BEAT2_SPLIT_FILE"),
    help="BEAT2 train_test_split.csv; required with --dataset beat.",
)
parser.add_argument("-d", "--device", type=str, default="cuda", help="Device to use.")
parser.add_argument("-p", "--person", type=int, nargs='+', 
                    default=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29], 
                    help="Person to evaluate.")
parser.add_argument(
    "--talkshow_manifest",
    "--talkshow-manifest",
    dest="talkshow_manifest",
    type=str,
    default=None,
    help=(
        "Optional UTF-8 text file containing one canonical TalkSHOW clip ID "
        "('<speaker>__<sequence>') per line. Blank lines and lines beginning "
        "with '#' are ignored. Only valid with --dataset talkshow."
    ),
)
parser.add_argument("--only_fgd", type=bool, default=False, help="only test fgd.")
parser.add_argument(
    "--no_style_metrics",
    "--no-style-metrics",
    dest="no_style_metrics",
    action="store_true",
    help=(
        "Disable the default Style-SWD and long-lag Style-ACF Error "
        "evaluation. --only_fgd also disables these metrics."
    ),
)
config = parser.parse_args()
if config.dataset == "beat" and config.talkshow_manifest is not None:
    parser.error("--talkshow-manifest requires --dataset talkshow")
if config.dataset == "beat" and not config.split_file:
    parser.error("--split_file is required with --dataset beat")
config.download_path = os.path.abspath(
    os.path.expanduser(config.download_path)
).rstrip(os.sep) + os.sep

if config.dataset == "talkshow":
    try:
        eval_samples = _discover_talkshow_samples(
            config.input_path,
            config.input_audio_path,
            config.talkshow_manifest,
        )
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))
    sample_source = (
        f"manifest {Path(config.talkshow_manifest).expanduser()}"
        if config.talkshow_manifest is not None
        else "canonical directory scan"
    )
    print(
        f"Evaluation dataset: TalkSHOW ({len(eval_samples)} clips; "
        f"{sample_source})"
    )
else:
    eval_samples = _discover_beat_samples(config.split_file, config.person)
    print(f"Evaluation dataset: BEAT ({len(eval_samples)} clips)")

device = torch.device(config.device)
download_path = config.download_path
style_metrics_enabled = not config.no_style_metrics and not config.only_fgd

fgd_evaluator = FGD(download_path=download_path)
if not config.only_fgd:
    bc_evaluator = BC(download_path=download_path, sigma=0.3, order=7)
    l1_evaluator= L1div()
    lvd_evaluator = LVDFace()
    mse_evaluator = MSEFace()
style_evaluator = GestureStyleAccumulator() if style_metrics_enabled else None
print(
    "Interpretable style metrics: "
    + ("enabled" if style_metrics_enabled else "disabled")
)

smplx_model_dir = os.path.join(download_path, "smplx_models", "smplx")
if not os.path.exists(smplx_model_dir):
    smplx_model_file_path = os.path.join(smplx_model_dir, "SMPLX_NEUTRAL_2020.npz")
    os.makedirs(smplx_model_dir, exist_ok=True)
    if not os.path.exists(smplx_model_file_path):
        print(f"Downloading {smplx_model_file_path}")
        wget.download(
            "https://huggingface.co/spaces/H-Liu1997/EMAGE/resolve/main/EMAGE/smplx_models/smplx/SMPLX_NEUTRAL_2020.npz",
            smplx_model_file_path,
        )

smplx_model = smplx.create(
    download_path + "smplx_models/",
    model_type='smplx',
    gender='NEUTRAL_2020',
    use_face_contour=False,
    num_betas=300,
    num_expression_coeffs=100,
    ext='npz',
    use_pca=False,
).eval()

# 保留原始实现，没有用diffusion里面的
def get_motion_rep_numpy(poses_np, pose_fps=30, device="cuda", expressions=None, expression_only=False, betas=None):
    # motion["poses"] is expected to be numpy array of shape (n, 165)
    # (n, 55*3), axis-angle for 55 joints
    global smplx_model
    smplx_model = smplx_model.to(device)
    n = poses_np.shape[0]

    # Convert numpy to torch tensor for SMPL-X forward pass
    poses_ts = torch.from_numpy(poses_np).float().to(device).unsqueeze(0)  # (1, n, 165)
    poses_ts_reshaped = poses_ts.reshape(-1, 165)  # (n, 165)
    betas = torch.zeros(n, 300, device=device) if betas is None else torch.from_numpy(betas).to(device).unsqueeze(0).repeat(n, 1)
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

#################################################################################
# 读取数据
all_motion_pred = []
all_motion_gt = []
all_audio = []
all_expression_pred = []
all_expression_gt = []
all_betas = []
all_file_ids = []
all_speaker_ids = []
bc_short_clip_count = 0

for file, speaker_id in eval_samples:
    npz_path = os.path.join(config.input_path, "res_" + file + ".npz")
    gt_npz_path = os.path.join(config.input_path, "gt_" + file + ".npz")
    audio_path = os.path.join(config.input_audio_path, "gt_sr16000_" + file + ".wav")
    npz = np.load(npz_path, allow_pickle=True)
    gt_npz = np.load(gt_npz_path, allow_pickle=True)
    all_motion_pred.append(npz["poses"])
    all_motion_gt.append(gt_npz["poses"])
    all_audio.append(librosa.load(audio_path, sr=16000)[0])
    all_expression_pred.append(npz["expressions"])
    all_expression_gt.append(gt_npz["expressions"])
    all_betas.append(gt_npz["betas"])
    all_file_ids.append(str(file))
    all_speaker_ids.append(speaker_id)

################################################################################

# Example usage
for (
    motion_pred,
    motion_gt,
    audio,
    expression_pred,
    expression_gt,
    betas,
    file_id,
    speaker_id,
) in tqdm(
    zip(
        all_motion_pred,
        all_motion_gt,
        all_audio,
        all_expression_pred,
        all_expression_gt,
        all_betas,
        all_file_ids,
        all_speaker_ids,
    ),
    total=len(all_file_ids),
):
    # truncate to the shorter length between pred and gt
    t = min(motion_pred.shape[0], motion_gt.shape[0])
    motion_pred = motion_pred[:t]
    motion_gt = motion_gt[:t]
    expression_pred = expression_pred[:t]
    expression_gt = expression_gt[:t]
    if not config.only_fgd:
        motion_position_pred_joints = get_motion_rep_numpy(
            motion_pred,
            device=device,
            betas=betas,
        )["position"]  # t*55*3
        motion_position_pred = motion_position_pred_joints.reshape(t, -1)
        # ignore the start and end 2s, this may for beat dataset only
        if t > 120:
            audio_beat = bc_evaluator.load_audio(
                audio,
                t_start=2 * 16000,
                t_end=int((t - 60) / 30 * 16000),
                without_file=True,
            )
            motion_beat = bc_evaluator.load_motion(
                motion_position_pred,
                t_start=60,
                t_end=t - 60,
                pose_fps=30,
                without_file=True,
            )
            bc_evaluator.compute(
                audio_beat,
                motion_beat,
                length=t - 120,
                pose_fps=30,
            )
        else:
            bc_short_clip_count += 1

        l1_evaluator.compute(motion_position_pred)

        if style_evaluator is not None:
            motion_position_gt_joints = get_motion_rep_numpy(
                motion_gt,
                device=device,
                betas=betas,
            )["position"]
            accepted = style_evaluator.update(
                motion_position_pred_joints,
                motion_position_gt_joints,
                speaker_id=speaker_id,
                clip_id=file_id,
            )
            if not accepted:
                warnings.warn(
                    f"Style metrics skipped invalid/short clip: {file_id}"
                )

        face_position_pred = get_motion_rep_numpy(
            motion_pred, device=device, expressions=expression_pred, expression_only=True, betas=betas)["vertices"] # t -1
        face_position_gt = get_motion_rep_numpy(
            motion_gt, device=device, expressions=expression_gt, expression_only=True, betas=betas)["vertices"]
        lvd_evaluator.compute(face_position_pred, face_position_gt)
        mse_evaluator.compute(face_position_pred, face_position_gt)

    # fgd requires rotation 6d representaiton
    motion_gt = torch.from_numpy(motion_gt).to(device).unsqueeze(0)
    motion_pred = torch.from_numpy(motion_pred).to(device).unsqueeze(0)
    motion_gt = rc.axis_angle_to_rotation_6d(motion_gt.reshape(1, t, 55, 3)).reshape(1, t, 55*6)
    motion_pred = rc.axis_angle_to_rotation_6d(motion_pred.reshape(1, t, 55, 3)).reshape(1, t, 55*6)
    fgd_evaluator.update(motion_pred.float(), motion_gt.float())
    
metrics = {}
if config.dataset == "talkshow":
    metrics["dataset"] = "talkshow"
    metrics["person"] = sorted(set(all_speaker_ids))
    metrics["talkshow_speaker_ids"] = TALKSHOW_SPEAKER_IDS
    metrics["clip_count"] = len(all_file_ids)
    metrics["clip_manifest_sha256"] = hashlib.sha256(
        ("\n".join(all_file_ids) + "\n").encode("utf-8")
    ).hexdigest()
else:
    metrics["person"] = [int(p) for p in config.person]
metrics["fgd"] = float(fgd_evaluator.compute())
if not config.only_fgd:
    metrics["bc"] = float(bc_evaluator.avg())
    metrics["l1"] = float(l1_evaluator.avg())
    metrics["lvd"] = float(lvd_evaluator.avg())
    metrics["mse"] = float(mse_evaluator.avg())
    if bc_evaluator.empty_audio_onset_clip_count:
        metrics["bc_empty_audio_onset_clip_count"] = (
            bc_evaluator.empty_audio_onset_clip_count
        )
        warnings.warn(
            "BC found no audio onset after boundary trimming for "
            f"{bc_evaluator.empty_audio_onset_clip_count} clip(s); their "
            "undefined GAHR contribution was scored as 0.0."
        )
    if bc_short_clip_count:
        metrics["bc_short_clip_skip_count"] = bc_short_clip_count
        warnings.warn(
            f"BC skipped {bc_short_clip_count} clip(s) with at most 120 "
            "frames because trimming 60 frames from each boundary leaves no "
            "valid evaluation interval. Other metrics still include them."
        )
if style_evaluator is not None:
    try:
        style_results = style_evaluator.compute()
        metrics["style_swd"] = style_results["style_swd"]
        metrics["style_acf_error"] = style_results["style_acf_error"]
        for feature_name in STYLE_FEATURE_NAMES:
            metrics[f"style_{feature_name}_w2"] = style_results[
                "feature_w2"
            ][feature_name]
        metrics["style_metrics"] = style_results["details"]
    except Exception as exc:
        warnings.warn(
            "Style metric aggregation failed; preserving the standard EMAGE "
            f"metrics: {type(exc).__name__}: {exc}"
        )
        metrics["style_swd"] = None
        metrics["style_acf_error"] = None
        metrics["style_metrics"] = {
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
        }

with open(config.output_path, "w") as f:
    json.dump(metrics, f, indent=4)
print(metrics)
