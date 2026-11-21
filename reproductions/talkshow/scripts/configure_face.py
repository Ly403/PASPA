#!/usr/bin/env python3
"""Create a run-local TalkSHOW face config from the pinned upstream config."""

import argparse
import hashlib
import json
from pathlib import Path


EXPECTED_ASSETS = (
    "SMPLX_NEUTRAL_2020.npz",
    "smplx_extra_joints.yaml",
    "SMPLX_to_J14.pkl",
)

MODEL_WEIGHT_FILES = (
    "model.safetensors",
    "pytorch_model.bin",
    "model.safetensors.index.json",
    "pytorch_model.bin.index.json",
)
MODEL_METADATA_FILES = (
    "config.json",
    "preprocessor_config.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "added_tokens.json",
)


def _canonical_json_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _collect_audio_model_file_hashes(root: Path) -> dict[str, str]:
    files = {
        root / name
        for name in MODEL_METADATA_FILES + MODEL_WEIGHT_FILES
        if (root / name).is_file()
    }
    for index_name in (
        "model.safetensors.index.json",
        "pytorch_model.bin.index.json",
    ):
        index_path = root / index_name
        if not index_path.is_file():
            continue
        try:
            index = json.loads(index_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(
                f"invalid audio model weight index: {index_path}"
            ) from error
        weight_map = (
            index.get("weight_map") if isinstance(index, dict) else None
        )
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError(
                f"audio model weight index has no weight_map: {index_path}"
            )
        for relative_name in set(weight_map.values()):
            if not isinstance(relative_name, str) or not relative_name:
                raise ValueError(
                    "audio model weight index contains an unsafe shard: "
                    f"{relative_name!r}"
                )
            relative_path = Path(relative_name)
            if relative_path.is_absolute() or ".." in relative_path.parts:
                raise ValueError(
                    "audio model weight index contains an unsafe shard: "
                    f"{relative_name!r}"
                )
            shard = root / relative_path
            if not shard.is_file():
                raise FileNotFoundError(
                    f"audio model weight shard is missing: {shard}"
                )
            files.add(shard)
    return {
        path.relative_to(root).as_posix(): _file_sha256(path)
        for path in sorted(files)
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream-dir", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--smplx-assets", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-tag", required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--save-every", type=int, required=True)
    parser.add_argument("--print-every", type=int, required=True)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="per-device batch size",
    )
    parser.add_argument(
        "--world-size",
        type=int,
        default=1,
        help="requested number of distributed worker processes",
    )
    parser.add_argument(
        "--variable-length-batching",
        action="store_true",
        help="enable batching for variable-length whole-video clips",
    )
    parser.add_argument(
        "--length-bucket",
        type=int,
        help="positive length-bucket multiplier used by variable-length batching",
    )
    drop_last_group = parser.add_mutually_exclusive_group()
    drop_last_group.add_argument(
        "--drop-last",
        dest="drop_last",
        action="store_true",
        help="discard an incomplete final per-device batch",
    )
    drop_last_group.add_argument(
        "--no-drop-last",
        dest="drop_last",
        action="store_false",
        help="retain an incomplete final per-device batch",
    )
    parser.set_defaults(drop_last=True)
    parser.add_argument(
        "--audio-encoder",
        choices=("wav2vec2", "wavlm"),
        default="wav2vec2",
    )
    parser.add_argument("--audio-model-path", type=Path)
    parser.add_argument(
        "--face-loss",
        choices=("legacy", "corrected"),
        default="legacy",
    )
    parser.add_argument("--resume-from", type=Path)
    args = parser.parse_args()

    if args.epochs < 1 or args.save_every < 1 or args.print_every < 1:
        parser.error("epochs/save-every/print-every must be positive")
    if args.batch_size < 1 or args.world_size < 1:
        parser.error("batch-size/world-size must be positive")
    if args.length_bucket is not None and args.length_bucket < 1:
        parser.error("length-bucket must be positive")
    if args.length_bucket is not None and not args.variable_length_batching:
        parser.error("--length-bucket requires --variable-length-batching")
    if (
        args.batch_size > 1 or args.world_size > 1
    ) and not args.variable_length_batching:
        parser.error(
            "--batch-size greater than 1 or --world-size greater than 1 requires "
            "--variable-length-batching"
        )
    if args.variable_length_batching and args.length_bucket is None:
        parser.error(
            "--variable-length-batching requires --length-bucket"
        )
    if args.variable_length_batching and args.audio_encoder != "wavlm":
        parser.error(
            "--variable-length-batching currently requires --audio-encoder=wavlm"
        )
    if args.audio_encoder == "wavlm" and args.audio_model_path is None:
        parser.error("--audio-model-path is required when --audio-encoder=wavlm")

    audio_model_path = None
    audio_model_config = None
    audio_config_sha256 = None
    audio_preprocessor_sha256 = None
    audio_model_files_sha256 = None
    audio_do_normalize = False
    audio_hidden_size = 768
    if args.audio_model_path is not None:
        audio_model_path = args.audio_model_path.resolve()
        if not audio_model_path.is_dir():
            parser.error(f"audio model directory does not exist: {audio_model_path}")
        if not (audio_model_path / "config.json").is_file():
            parser.error(
                f"audio model directory has no config.json: {audio_model_path}"
            )
        with (audio_model_path / "config.json").open(
            "r", encoding="utf-8"
        ) as handle:
            audio_model_config = json.load(handle)
        audio_config_sha256 = _canonical_json_sha256(audio_model_config)
        if audio_model_config.get("model_type") != args.audio_encoder:
            parser.error(
                "audio model type does not match --audio-encoder: "
                f"expected {args.audio_encoder}, found "
                f"{audio_model_config.get('model_type')!r}"
            )
        audio_hidden_size = audio_model_config.get("hidden_size")
        if not isinstance(audio_hidden_size, int) or audio_hidden_size < 1:
            parser.error(
                f"audio model config has invalid hidden_size: {audio_hidden_size!r}"
            )
        if args.audio_encoder == "wav2vec2" and audio_hidden_size != 768:
            parser.error(
                "the TalkSHOW reproduction path requires a 768-dimensional "
                "Wav2Vec2-base checkpoint"
            )
        if not any(
            (audio_model_path / name).is_file()
            for name in MODEL_WEIGHT_FILES
        ):
            parser.error(
                "audio model directory has no Hugging Face model weights: "
                f"{audio_model_path}"
            )
        try:
            audio_model_files_sha256 = _collect_audio_model_file_hashes(
                audio_model_path
            )
        except (OSError, ValueError) as error:
            parser.error(str(error))
        if args.audio_encoder == "wavlm":
            preprocessor_path = audio_model_path / "preprocessor_config.json"
            if not preprocessor_path.is_file():
                parser.error(
                    "WavLM model directory has no preprocessor_config.json: "
                    f"{audio_model_path}"
                )
            with preprocessor_path.open("r", encoding="utf-8") as handle:
                preprocessor_config = json.load(handle)
            audio_preprocessor_sha256 = _canonical_json_sha256(
                preprocessor_config
            )
            if not isinstance(preprocessor_config.get("do_normalize"), bool):
                parser.error(
                    "WavLM preprocessor_config.json must contain a boolean "
                    "do_normalize value"
                )
            audio_do_normalize = preprocessor_config["do_normalize"]
            if (
                args.variable_length_batching
                and audio_model_config.get("feat_extract_norm") != "layer"
            ):
                parser.error(
                    "variable-length WavLM batching requires "
                    "feat_extract_norm='layer'"
                )

    source = args.upstream_dir.resolve() / "config" / "face.json"
    with source.open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    config["Data"]["data_root"] = str(args.data_root.resolve())
    config["Data"]["pklname"] = f"_talkshow_face_{args.cache_tag}.pkl"
    if args.smplx_assets is not None:
        assets = args.smplx_assets.resolve()
        missing = [name for name in EXPECTED_ASSETS if not (assets / name).is_file()]
        if missing:
            parser.error(
                "SMPL-X asset directory is incomplete; missing: "
                + ", ".join(missing)
            )
        config["smplx_npz_path"] = str(assets / "SMPLX_NEUTRAL_2020.npz")
        config["extra_joint_path"] = str(assets / "smplx_extra_joints.yaml")
        config["j14_regressor_path"] = str(assets / "SMPLX_to_J14.pkl")
    config["DataLoader"]["batch_size"] = args.batch_size
    config["DataLoader"]["variable_length_batching"] = (
        args.variable_length_batching
    )
    config["DataLoader"]["length_bucket"] = args.length_bucket
    config["DataLoader"]["drop_last"] = args.drop_last
    config["Distributed"] = {
        "requested_world_size": args.world_size,
        "effective_global_batch_size": args.batch_size * args.world_size,
    }
    config["Train"]["epochs"] = args.epochs
    config["Log"]["save_every"] = args.save_every
    config["Log"]["print_every"] = args.print_every
    config["Model"]["audio_encoder"] = args.audio_encoder
    config["Model"]["audio_model_path"] = (
        str(audio_model_path) if audio_model_path is not None else None
    )
    config["Model"]["audio_model_source"] = (
        str(audio_model_path)
        if audio_model_path is not None
        else "facebook/wav2vec2-base-960h"
    )
    config["Model"]["audio_model_revision"] = (
        None
        if audio_model_path is not None
        else "22aad52d435eb6dbaf354bdad9b0da84ce7d6156"
    )
    config["Model"]["audio_do_normalize"] = audio_do_normalize
    config["Model"]["audio_hidden_size"] = audio_hidden_size
    config["Model"]["audio_config_sha256"] = audio_config_sha256
    config["Model"][
        "audio_preprocessor_sha256"
    ] = audio_preprocessor_sha256
    config["Model"][
        "audio_model_files_sha256"
    ] = audio_model_files_sha256
    # PASPA uses WavLM as a fixed feature extractor. Freezing it
    # here keeps the representation reusable by the body and face heads.
    config["Model"]["freeze_audio_encoder"] = args.audio_encoder == "wavlm"
    config["Reproduction"] = {
        "face_loss": args.face_loss,
    }
    if args.resume_from is not None:
        resume_from = args.resume_from.resolve()
        if not resume_from.is_file():
            parser.error(f"resume checkpoint does not exist: {resume_from}")
        config["Resume"] = {
            "checkpoint": str(resume_from),
            "boundary": "end_of_epoch",
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(config, handle, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
