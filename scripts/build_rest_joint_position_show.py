#!/usr/bin/env python3
"""
Build a per-speaker SMPL-X rest joint position table for the SHOW dataset.

Reads betas from a SHOW LMDB (output of `scripts/create_lmdb_show.py`), groups by pid,
runs SMPL-X(NEUTRAL_2020, num_betas=300) with pose=0/trans=0 to recover the
(55, 3) rest joint position per speaker, and writes the table as a MATLAB-format
.mat file. Overlapping LMDB windows from the same source clip are counted once,
so one long clip cannot dominate the per-speaker beta average.

Output schema matches the BEAT2 file at
`diffusion/data/datasets/beat2/rest_joint_position.mat` — top-level keys are pid as a
digit string ("0", "1", ...), each value is a (55, 3) float32 array. This is exactly what
`BEATXDataset.LMDBDataset.__init__` expects (it only picks up keys matching `k.isdigit()`).

Workflow:
    # 1) one-time computation (writes to tools/show/rest_joint_position_show.mat)
    python scripts/build_rest_joint_position_show.py \
        --lmdb_path /data/show_train_seq_size_256_stride_size_20_global.lmdb \
        --smplx_model_path /absolute/path/to/emage_evaltools/smplx_models

The canonical main-model contract keeps raw LMDB/rest-table PIDs 21..24 and
maps them to model-local IDs 0..3 only after the rest-joint lookup.  Therefore
the source LMDB must declare ``pid_offset=1``.

    # 2) point the SHOW data config at the separate output; do not overwrite
    #    BEAT2's table:
    export SHOW_REST_JOINT_PATH=/absolute/path/rest_joint_position_show.mat

The legacy --install option is retained for old workflows, but the dedicated
dataset parameter is safer when BEAT and SHOW experiments coexist. This script
only writes the explicitly selected SHOW output; it does not alter BEAT unless
the caller deliberately passes --install.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
import sys
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from numbers import Integral
from typing import Any

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATASET_MAT_PATH = os.path.join(
    _REPO_ROOT, "diffusion", "data", "datasets", "beat2", "rest_joint_position.mat"
)
DEFAULT_EXPECTED_PIDS = (21, 22, 23, 24)
DEFAULT_EXPECTED_PID_OFFSET = 1
OFFICIAL_SPLIT_SHA256 = (
    "2df6e745cdf7473f13ce3ae2ed759c3cceb60c9197e7f3fd65110e7bc20b6f2d"
)
OFFICIAL_HAND_COMPONENT_SHA256 = (
    "a9db3d2f04fe161e09553b17aaf5922791b0ed826593e640f4ddf8ebb8baf423"
)
FACE_CONTEXT_PROTOCOL = "source_clip_valid_slice_v1"


class RestJointBuildError(RuntimeError):
    """Raised when an input LMDB cannot safely produce SHOW rest joints."""


def validate_source_lmdb_metadata(
    raw_metadata: bytes | None,
    *,
    expected_pids: Sequence[int],
    expected_pid_offset: int = DEFAULT_EXPECTED_PID_OFFSET,
    lmdb_length: int,
) -> dict[str, Any]:
    """Require the canonical strict TalkSHOW train-LMDB provenance."""

    if raw_metadata is None:
        raise RestJointBuildError(
            "SHOW rest joints require a strict LMDB __meta__ record"
        )
    try:
        metadata = json.loads(raw_metadata.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise RestJointBuildError(
            "LMDB __meta__ must be a UTF-8 JSON object"
        ) from error
    if not isinstance(metadata, Mapping):
        raise RestJointBuildError("LMDB __meta__ must be a JSON object")

    required_values = {
        "schema_version": 2,
        "dataset": "SHOW",
        "split": "train",
        "pid_offset": expected_pid_offset,
        "local_rotation": False,
        "rotation_space": "global",
        "pose_representation": "axis_angle",
        "feature_scope": "full_clip",
        "face_context_frames": 6,
        "face_context_protocol": FACE_CONTEXT_PROTOCOL,
        "split_manifest_sha256": OFFICIAL_SPLIT_SHA256,
        "hand_component_sha256": OFFICIAL_HAND_COMPONENT_SHA256,
        "output_sample_count": lmdb_length,
    }
    missing = sorted(
        key for key in required_values if key not in metadata
    )
    if missing:
        raise RestJointBuildError(
            "LMDB __meta__ is incomplete; missing: " + ", ".join(missing)
        )
    for key, expected in required_values.items():
        if metadata[key] != expected:
            raise RestJointBuildError(
                f"LMDB metadata {key}={metadata[key]!r}, "
                f"expected {expected!r}"
            )

    speaker_ids = metadata.get("speaker_ids")
    if not isinstance(speaker_ids, Mapping):
        raise RestJointBuildError(
            "LMDB metadata speaker_ids must be an object"
        )
    observed_pids = sorted(int(pid) for pid in speaker_ids.values())
    if observed_pids != sorted(int(pid) for pid in expected_pids):
        raise RestJointBuildError(
            f"LMDB metadata PIDs {observed_pids} != expected "
            f"{sorted(int(pid) for pid in expected_pids)}"
        )
    return dict(metadata)


def _strict_beta_vector(value: Any, *, record_name: str) -> np.ndarray:
    if not isinstance(value, np.ndarray):
        raise RestJointBuildError(
            f"{record_name}.betas must be a numpy.ndarray"
        )
    if value.shape != (300,):
        raise RestJointBuildError(
            f"{record_name}.betas shape={value.shape}, expected (300,)"
        )
    if value.dtype != np.float32:
        raise RestJointBuildError(
            f"{record_name}.betas dtype={value.dtype}, expected float32"
        )
    if not np.isfinite(value).all():
        raise RestJointBuildError(
            f"{record_name}.betas contains NaN or Inf"
        )
    return value


def collect_unique_betas(
    records: Iterable[tuple[int, Any]],
    *,
    max_clips_per_pid: int,
    expected_pids: Sequence[int] = DEFAULT_EXPECTED_PIDS,
) -> tuple[dict[int, list[np.ndarray]], dict[str, Any]]:
    """Validate records and select at most N unique source clips per PID.

    Deduplication is keyed by ``(pid, ori_vid)``. Every overlapping window is
    still validated, and conflicting beta vectors for the same source clip are
    rejected instead of silently choosing one.
    """

    if (
        isinstance(max_clips_per_pid, bool)
        or not isinstance(max_clips_per_pid, Integral)
        or max_clips_per_pid < 0
    ):
        raise ValueError("max_clips_per_pid must be an integer >= 0")
    normalized_expected_pids = tuple(expected_pids)
    if not normalized_expected_pids:
        raise ValueError("expected_pids cannot be empty")
    if any(
        isinstance(pid, bool) or not isinstance(pid, Integral)
        for pid in normalized_expected_pids
    ):
        raise ValueError("expected_pids must contain only integers")
    expected_pid_set = {int(pid) for pid in normalized_expected_pids}
    if len(expected_pid_set) != len(normalized_expected_pids):
        raise ValueError("expected_pids cannot contain duplicates")

    selected: defaultdict[int, list[np.ndarray]] = defaultdict(list)
    seen_clip_betas: dict[tuple[int, str], np.ndarray] = {}
    observed_pids: set[int] = set()
    record_count = 0
    duplicate_window_count = 0

    for record_index, record in records:
        record_name = f"record {record_index}"
        record_count += 1
        if not isinstance(record, Mapping):
            raise RestJointBuildError(f"{record_name} must be a mapping")
        for field in ("pid", "ori_vid", "betas"):
            if field not in record:
                raise RestJointBuildError(
                    f"{record_name} is missing field {field!r}"
                )

        raw_pid = record["pid"]
        if isinstance(raw_pid, bool) or not isinstance(raw_pid, Integral):
            raise RestJointBuildError(
                f"{record_name}.pid must be an integer, got {raw_pid!r}"
            )
        pid = int(raw_pid)
        if pid not in expected_pid_set:
            raise RestJointBuildError(
                f"{record_name}.pid={pid} is outside expected IDs "
                f"{sorted(expected_pid_set)}"
            )
        observed_pids.add(pid)

        ori_vid = record["ori_vid"]
        ori_parts = (
            ori_vid.split("__", 1)
            if isinstance(ori_vid, str)
            else ()
        )
        if (
            len(ori_parts) != 2
            or not ori_parts[0]
            or not ori_parts[1]
        ):
            raise RestJointBuildError(
                f"{record_name}.ori_vid must be '<speaker>__<clip>'"
            )
        betas = _strict_beta_vector(
            record["betas"],
            record_name=record_name,
        )

        source_key = (pid, ori_vid)
        previous = seen_clip_betas.get(source_key)
        if previous is not None:
            duplicate_window_count += 1
            if not np.array_equal(previous, betas):
                raise RestJointBuildError(
                    "conflicting betas for repeated source clip "
                    f"pid={pid}, ori_vid={ori_vid!r}"
                )
            continue

        # Keep a copy so a mutable input record cannot alter later consistency
        # checks. This remains modest for the four-person SHOW split.
        seen_clip_betas[source_key] = betas.copy()
        if (
            max_clips_per_pid == 0
            or len(selected[pid]) < max_clips_per_pid
        ):
            selected[pid].append(betas.copy())

    if record_count == 0:
        raise RestJointBuildError("LMDB contains no records")
    if observed_pids != expected_pid_set:
        raise RestJointBuildError(
            f"observed PID set {sorted(observed_pids)} != expected "
            f"{sorted(expected_pid_set)}"
        )
    empty_selected = [
        pid for pid in sorted(expected_pid_set) if not selected[pid]
    ]
    if empty_selected:
        raise RestJointBuildError(
            f"no source clips selected for PIDs {empty_selected}"
        )

    per_pid = {
        pid: selected[pid]
        for pid in sorted(expected_pid_set)
    }
    unique_counts = {
        pid: sum(1 for source_pid, _ in seen_clip_betas if source_pid == pid)
        for pid in sorted(expected_pid_set)
    }
    stats = {
        "record_count": record_count,
        "unique_source_clip_count": len(seen_clip_betas),
        "duplicate_window_count": duplicate_window_count,
        "unique_clips_per_pid": unique_counts,
        "selected_clips_per_pid": {
            pid: len(per_pid[pid]) for pid in sorted(per_pid)
        },
    }
    return per_pid, stats


def collect_betas_per_pid(
    lmdb_path,
    max_clips_per_pid,
    expected_pids: Sequence[int] = DEFAULT_EXPECTED_PIDS,
    *,
    expected_pid_offset: int = DEFAULT_EXPECTED_PID_OFFSET,
    return_stats: bool = False,
):
    """Walk the LMDB and gather strict, source-clip-deduplicated betas."""

    try:
        import lmdb
    except ImportError as error:
        raise RestJointBuildError(
            "python-lmdb is required to build SHOW rest joints"
        ) from error
    if not os.path.isfile(lmdb_path):
        raise RestJointBuildError(
            f"SHOW LMDB does not exist or is not a file: {lmdb_path}"
        )

    env = lmdb.open(
        lmdb_path,
        subdir=False,
        readonly=True,
        lock=False,
        readahead=False,
        meminit=False,
        max_readers=8,
    )
    try:
        with env.begin() as txn:
            length_bytes = txn.get(b"__len__")
            if length_bytes is None:
                raise RestJointBuildError(
                    f"LMDB key '__len__' missing in {lmdb_path}"
                )
            try:
                length = int(length_bytes.decode("ascii"))
            except (UnicodeDecodeError, ValueError) as error:
                raise RestJointBuildError(
                    "LMDB __len__ is not an ASCII integer"
                ) from error
            if length <= 0:
                raise RestJointBuildError(
                    f"LMDB is empty (__len__={length}): {lmdb_path}"
                )
            validate_source_lmdb_metadata(
                txn.get(b"__meta__"),
                expected_pids=expected_pids,
                expected_pid_offset=expected_pid_offset,
                lmdb_length=length,
            )

            def iter_minimal_records() -> Iterable[tuple[int, Any]]:
                # Yield one minimal projection at a time. Full audio/feature
                # windows are released before the next LMDB record is loaded.
                for record_index in range(length):
                    key = f"{record_index:010d}".encode("ascii")
                    byte_data = txn.get(key)
                    if byte_data is None:
                        raise RestJointBuildError(
                            f"LMDB record key {key!r} is missing"
                        )
                    try:
                        record = pickle.loads(byte_data)
                    except Exception as error:
                        raise RestJointBuildError(
                            f"could not unpickle LMDB record {record_index}"
                        ) from error
                    if isinstance(record, Mapping):
                        record = {
                            field: record[field]
                            for field in ("pid", "ori_vid", "betas")
                            if field in record
                        }
                    yield record_index, record

            per_pid, stats = collect_unique_betas(
                iter_minimal_records(),
                max_clips_per_pid=max_clips_per_pid,
                expected_pids=expected_pids,
            )
    finally:
        env.close()
    if return_stats:
        return per_pid, stats
    return per_pid


def compute_J_table(per_pid, smplx_model, device):
    """Run SMPL-X(pose=0, trans=0, betas=mean(betas_for_pid)) per pid -> (55, 3)."""
    import torch

    table = {}
    with torch.no_grad():
        for pid in sorted(per_pid.keys()):
            betas_arr = np.stack(per_pid[pid], axis=0)
            if betas_arr.ndim != 2 or betas_arr.shape[1] != 300:
                raise RestJointBuildError(
                    f"PID {pid} beta table has shape {betas_arr.shape}, "
                    "expected [N,300]"
                )
            betas_mean = betas_arr.mean(axis=0, dtype=np.float64).astype(
                np.float32
            )

            betas_t = (
                torch.from_numpy(betas_mean).to(device).unsqueeze(0)
            )
            out = smplx_model(
                betas=betas_t,
                global_orient=torch.zeros((1, 3), device=device),
                body_pose=torch.zeros((1, 21 * 3), device=device),
                jaw_pose=torch.zeros((1, 3), device=device),
                leye_pose=torch.zeros((1, 3), device=device),
                reye_pose=torch.zeros((1, 3), device=device),
                left_hand_pose=torch.zeros((1, 15 * 3), device=device),
                right_hand_pose=torch.zeros((1, 15 * 3), device=device),
                expression=torch.zeros((1, 100), device=device),
                transl=torch.zeros((1, 3), device=device),
                return_verts=True,
                return_joints=True,
            )
            joints = (
                out.joints[0, :55]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )
            if joints.shape != (55, 3) or not np.isfinite(joints).all():
                raise RestJointBuildError(
                    f"SMPL-X produced invalid rest joints for PID {pid}: "
                    f"shape={joints.shape}"
                )
            table[pid] = joints
    return table


def install(out_path, dataset_mat_path):
    """Copy `out_path` to dataset_mat_path with a one-time backup of the original."""
    bak = dataset_mat_path + ".bak_beat2"
    if os.path.exists(dataset_mat_path) and not os.path.exists(bak):
        shutil.copy2(dataset_mat_path, bak)
        print(f"[install] backed up original BEAT2 mat: {dataset_mat_path} -> {bak}")
    elif os.path.exists(bak):
        print(f"[install] existing backup at {bak}; not overwriting it")
    shutil.copy2(out_path, dataset_mat_path)
    print(f"[install] copied SHOW mat into dataset path: {out_path} -> {dataset_mat_path}")
    print(f"[install] to restore BEAT2 mat later: cp {bak} {dataset_mat_path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--lmdb_path", type=str, required=True,
                        help="SHOW train LMDB (from scripts/create_lmdb_show.py)")
    parser.add_argument("--smplx_model_path", type=str, required=True,
                        help="Directory containing SMPLX_NEUTRAL_2020.npz "
                             "(same as EmageTrainer.configure_smplx)")
    parser.add_argument("--output", type=str,
                        default=os.path.join(_REPO_ROOT, "tools", "show",
                                             "rest_joint_position_show.mat"),
                        help="Output mat file path")
    parser.add_argument("--install", action="store_true",
                        help="Legacy: copy the mat into the shared BEAT2 location. "
                             "Prefer data.*.params.rest_joint_position_path.")
    parser.add_argument("--max_clips_per_pid", type=int, default=100,
                        help="Cap unique source clips averaged per pid "
                             "(overlapping windows count once; 0 = use all)")
    parser.add_argument("--expected_pids", type=int, nargs="+",
                        default=list(DEFAULT_EXPECTED_PIDS),
                        help="Exact SHOW PID set required in the input LMDB")
    parser.add_argument(
        "--expected-pid-offset",
        "--expected_pid_offset",
        dest="expected_pid_offset",
        type=int,
        default=DEFAULT_EXPECTED_PID_OFFSET,
        help=(
            "Exact metadata pid_offset required in the input LMDB "
            f"(default: {DEFAULT_EXPECTED_PID_OFFSET})"
        ),
    )
    args = parser.parse_args()

    if args.max_clips_per_pid < 0:
        parser.error("--max_clips_per_pid must be >= 0")
    if not args.expected_pids:
        parser.error("--expected_pids cannot be empty")
    if len(set(args.expected_pids)) != len(args.expected_pids):
        parser.error("--expected_pids cannot contain duplicates")

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1) scan LMDB
    print(f"[scan] reading betas from {args.lmdb_path}")
    per_pid, collection_stats = collect_betas_per_pid(
        args.lmdb_path,
        args.max_clips_per_pid,
        expected_pids=args.expected_pids,
        expected_pid_offset=args.expected_pid_offset,
        return_stats=True,
    )
    print(f"[scan] {len(per_pid)} unique pids: {sorted(per_pid.keys())}")
    print(
        "[scan] "
        f"windows={collection_stats['record_count']}, "
        f"unique_source_clips="
        f"{collection_stats['unique_source_clip_count']}, "
        f"duplicate_windows="
        f"{collection_stats['duplicate_window_count']}"
    )
    for pid in sorted(per_pid.keys()):
        bs = np.stack(per_pid[pid], axis=0)
        std_mean = float(bs.std(axis=0).mean()) if bs.shape[0] > 1 else 0.0
        available = collection_stats["unique_clips_per_pid"][pid]
        print(
            f"  pid {pid:>3d}: selected {bs.shape[0]:>4d} / "
            f"{available:>4d} unique source clip(s), "
            f"inter-clip betas std-mean = {std_mean:.6f}"
        )

    # 2) build SMPL-X (lazy import so the script's argparse --help works without smplx)
    import smplx as smpl
    smplx_model_root = os.path.abspath(
        os.path.expanduser(args.smplx_model_path)
    )
    smplx_neutral_path = os.path.join(
        smplx_model_root,
        "smplx",
        "SMPLX_NEUTRAL_2020.npz",
    )
    if not os.path.isfile(smplx_neutral_path):
        raise RestJointBuildError(
            "SMPL-X neutral model is missing; expected "
            f"{smplx_neutral_path}"
        )
    print(f"[smplx] loading from {smplx_model_root}")
    smplx_model = smpl.create(
        smplx_model_root,
        model_type="smplx",
        gender="NEUTRAL_2020",
        use_face_contour=False,
        num_betas=300,
        num_expression_coeffs=100,
        ext="npz",
        use_pca=False,
    ).to(device).eval()

    # 3) compute J per pid
    print(f"[compute] running SMPL-X forward (pose=0, trans=0) per pid on {device}")
    table = compute_J_table(per_pid, smplx_model, device)
    for pid in sorted(table.keys()):
        J = table[pid]
        print(f"  pid {pid:>3d}: J range "
              f"x=[{J[:,0].min():+.3f},{J[:,0].max():+.3f}]  "
              f"y=[{J[:,1].min():+.3f},{J[:,1].max():+.3f}]  "
              f"z=[{J[:,2].min():+.3f},{J[:,2].max():+.3f}]")

    # 4) save mat (keys are digit strings so BEATXDataset's `k.isdigit()` filter picks them up)
    from scipy.io import savemat

    output_parent = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(output_parent, exist_ok=True)
    out_dict = {str(pid): table[pid] for pid in sorted(table.keys())}
    savemat(args.output, out_dict)
    print(f"[save] {args.output}  ({len(out_dict)} entries)")

    # 5) optional install
    if args.install:
        install(args.output, _DATASET_MAT_PATH)


if __name__ == "__main__":
    main()
