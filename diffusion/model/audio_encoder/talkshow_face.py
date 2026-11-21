"""TalkSHOW WavLM face head adapted to the main diffusion interface.

The TalkSHOW training code consumes raw waveforms and owns a frozen WavLM
backbone.  The main project already supplies frame-aligned WavLM Large hidden
states, so this module intentionally contains only the post-WavLM face head.
Its public ``forward`` follows the existing speech-model contract and returns
the 100 expression coefficients.  ``forward_face`` additionally exposes the
three jaw-pose channels stored in the TalkSHOW checkpoint.
"""

from __future__ import annotations

import hashlib
import json
import warnings
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


_TALKSHOW_SOURCE_SPEAKER_IDS = {
    "oliver": 20,
    "chemistry": 21,
    "seth": 22,
    "conan": 23,
}


def _canonical_json_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(
            f"required WavLM metadata file does not exist: {path}"
        ) from None
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid JSON in WavLM metadata file: {path}") from error


class _ConvNormRelu(nn.Module):
    """State-dict-compatible copy of TalkSHOW's 1-D ConvNormRelu."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        residual: bool = False,
        norm: str = "ln",
    ) -> None:
        super().__init__()
        if norm != "ln":
            raise ValueError("TalkSHOW face condition only supports LayerNorm")

        self.residual = residual
        self.norm_type = norm
        if residual:
            if in_channels == out_channels:
                self.residual_layer = nn.Identity()
            else:
                self.residual_layer = nn.Sequential(
                    nn.Conv1d(
                        in_channels,
                        out_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1,
                    )
                )

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.norm = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout(p=0.0)
        self.relu = nn.ReLU()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.dropout(self.conv(inputs))
        outputs = self.norm(outputs.transpose(1, 2)).transpose(1, 2)
        if self.residual:
            outputs = outputs + self.residual_layer(inputs)
        return self.relu(outputs)


class _SeqTranslator1D(nn.Module):
    """Three residual convolution layers used by the TalkSHOW audio head."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.num_layers = 3
        self.conv_layers = nn.Sequential(
            _ConvNormRelu(in_channels, out_channels, residual=True),
            _ConvNormRelu(out_channels, out_channels, residual=True),
            _ConvNormRelu(out_channels, out_channels, residual=True),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.conv_layers(inputs)


class _TalkSHOWAudioHead(nn.Module):
    """Identity-conditioned post-WavLM encoder from TalkSHOW."""

    def __init__(
        self,
        hidden_dim: int,
        identity_dim: int,
        num_person: int,
    ) -> None:
        super().__init__()
        self.identity = True
        self.id_mlp = nn.Conv1d(num_person, identity_dim, kernel_size=1, stride=1)
        self.first_net = _SeqTranslator1D(
            hidden_dim + identity_dim,
            hidden_dim,
        )
        # TalkSHOW registers a GRU but never calls it.  It is deliberately not
        # instantiated here; its checkpoint keys are explicitly ignored.
        self.dropout = nn.Dropout(0.1)

    def forward(
        self,
        features: torch.Tensor,
        identity: torch.Tensor,
    ) -> torch.Tensor:
        features = self.dropout(features)
        identity_features = identity.reshape(identity.shape[0], -1, 1)
        identity_features = identity_features.repeat(1, 1, features.shape[2])
        identity_features = self.id_mlp(
            identity_features.to(dtype=features.dtype)
        )
        return self.first_net(
            torch.cat([features, identity_features], dim=1)
        )


class TalkSHOWFaceConditionModel(nn.Module):
    """Use a trained TalkSHOW WavLM face head as diffusion face condition.

    Args:
        input_feature_dim: Dimension of the main project's WavLM hidden state.
        hidden_dim: TalkSHOW face-head channel width.
        output_feat_dim: Number of expression coefficients required by CausalEmage.
        jaw_dim: Number of axis-angle jaw channels in the TalkSHOW face output.
        num_person: Number of identities used to train this checkpoint.
        speaker_id_offset: Runtime network ID corresponding to local class zero.
            The TalkSHOW main-model dataset emits local IDs 0..3, so the
            integrated configuration uses zero.
        identity_dim: Width of TalkSHOW's one-hot identity projection.
        wavlm_model_path: Runtime WavLM directory used by validation/test.
        require_audio_contract: Require modern face-checkpoint audio metadata.
        required_face_loss: Optional face-loss protocol required by this run.

    ``pid`` is in the network-local identity space shared by the face head and
    diffusion backbone.  The strict TalkSHOW dataset keeps its raw LMDB/rest
    IDs separate and maps raw IDs 21..24 to runtime classes 0..3.
    """

    _LOADABLE_PREFIXES = (
        "audio_feature_map.",
        "audio_middle.",
        "decoder.",
        "final_out.",
    )
    _IGNORED_SOURCE_PREFIXES = (
        "audio_encoder.",
        "audio_middle.grus.",
    )
    requires_pretrained_checkpoint = True
    # The shared trainer interprets ``False`` as permission to apply BEAT's
    # legacy ``pid > 30 -> pid - 30`` symmetry mapping.  TalkSHOW owns a
    # stricter local 0..3 identity contract, so keep every incoming value
    # unchanged and validate it in ``_local_speaker_ids`` below.
    handles_global_speaker_ids = True
    supports_finetuning = False
    requires_frame_aligned_audio_features = True
    requires_contextual_audio_features = True
    context_radius_frames = 6
    supported_main_trainers = ("bld",)
    required_mixed_precision = "no"

    def __init__(
        self,
        input_feature_dim: int = 1024,
        hidden_dim: int = 256,
        output_feat_dim: int = 100,
        jaw_dim: int = 3,
        num_person: int = 4,
        speaker_id_offset: int = 0,
        identity_dim: int = 64,
        wavlm_model_path: str | Path | None = None,
        require_audio_contract: bool = False,
        required_face_loss: str | None = None,
        expected_pose_length: int = 256,
        expected_block_size: int = 64,
        expected_speakers: tuple[str, ...] | list[str] = (
            "oliver",
            "seth",
            "conan",
            "chemistry",
        ),
        expected_speaker_ids: Mapping[str, int] | None = None,
        expected_lmdb_pid_offset: int = 1,
    ) -> None:
        super().__init__()
        if min(
            input_feature_dim,
            hidden_dim,
            output_feat_dim,
            jaw_dim,
            num_person,
            identity_dim,
        ) <= 0:
            raise ValueError("TalkSHOW face dimensions must all be positive")

        self.input_feature_dim = int(input_feature_dim)
        self.hidden_dim = int(hidden_dim)
        self.output_feat_dim = int(output_feat_dim)
        self.jaw_dim = int(jaw_dim)
        self.num_person = int(num_person)
        self.speaker_id_offset = int(speaker_id_offset)
        if (
            isinstance(expected_lmdb_pid_offset, bool)
            or not isinstance(expected_lmdb_pid_offset, int)
        ):
            raise TypeError("expected_lmdb_pid_offset must be an integer")
        self.expected_lmdb_pid_offset = expected_lmdb_pid_offset
        self.identity_dim = int(identity_dim)
        self.wavlm_model_path = (
            Path(wavlm_model_path).expanduser().resolve()
            if wavlm_model_path is not None
            else None
        )
        if (
            isinstance(expected_pose_length, bool)
            or not isinstance(expected_pose_length, int)
            or expected_pose_length < 1
        ):
            raise ValueError("expected_pose_length must be a positive integer")
        if (
            isinstance(expected_block_size, bool)
            or not isinstance(expected_block_size, int)
            or expected_block_size < 1
        ):
            raise ValueError("expected_block_size must be a positive integer")
        if expected_pose_length % expected_block_size != 0:
            raise ValueError(
                "expected_pose_length must be divisible by "
                "expected_block_size"
            )
        self.expected_pose_length = expected_pose_length
        self.expected_block_size = expected_block_size
        self.require_audio_contract = bool(require_audio_contract)
        if required_face_loss not in {None, "legacy", "corrected"}:
            raise ValueError(
                "required_face_loss must be null, 'legacy', or 'corrected'"
            )
        self.required_face_loss = required_face_loss
        self.expected_speakers = tuple(str(name) for name in expected_speakers)
        if len(self.expected_speakers) != self.num_person:
            raise ValueError(
                "expected_speakers length must equal num_person: "
                f"{len(self.expected_speakers)} != {self.num_person}"
            )
        if expected_speaker_ids is None:
            expected_speaker_ids = {
                "oliver": 21,
                "chemistry": 22,
                "seth": 23,
                "conan": 24,
            }
        # This mapping describes the raw LMDB/rest-joint identity namespace.
        # It is intentionally independent of speaker_id_offset, which governs
        # only the already-normalized IDs presented to the two networks.
        self.expected_speaker_ids = {
            str(name): int(pid)
            for name, pid in expected_speaker_ids.items()
        }
        if len(self.expected_speaker_ids) != self.num_person:
            raise ValueError(
                "expected_speaker_ids must contain exactly num_person raw "
                f"LMDB identities: {len(self.expected_speaker_ids)} != "
                f"{self.num_person}"
            )
        if set(self.expected_speaker_ids) != set(self.expected_speakers):
            raise ValueError(
                "expected_speaker_ids and expected_speakers must name the "
                "same speakers"
            )
        if len(set(self.expected_speaker_ids.values())) != self.num_person:
            raise ValueError(
                "expected_speaker_ids raw LMDB IDs must be unique"
            )
        canonical_lmdb_speaker_ids = {
            speaker: source_pid + self.expected_lmdb_pid_offset
            for speaker, source_pid in _TALKSHOW_SOURCE_SPEAKER_IDS.items()
        }
        if self.expected_speaker_ids != canonical_lmdb_speaker_ids:
            raise ValueError(
                "expected_speaker_ids must equal TalkSHOW's canonical source "
                "mapping plus expected_lmdb_pid_offset: "
                f"{self.expected_speaker_ids} != "
                f"{canonical_lmdb_speaker_ids}"
            )

        self.audio_feature_map = nn.Linear(
            self.input_feature_dim,
            self.hidden_dim,
        )
        self.audio_middle = _TalkSHOWAudioHead(
            self.hidden_dim,
            self.identity_dim,
            self.num_person,
        )

        self.decoder = nn.ModuleList(
            [
                nn.Sequential(
                    _ConvNormRelu(self.hidden_dim, 64),
                    _ConvNormRelu(64, 64),
                    _ConvNormRelu(64, 64),
                ),
                nn.Sequential(
                    _ConvNormRelu(self.hidden_dim, self.hidden_dim),
                    _ConvNormRelu(self.hidden_dim, self.hidden_dim),
                    _ConvNormRelu(self.hidden_dim, self.hidden_dim),
                ),
            ]
        )
        self.final_out = nn.ModuleList(
            [
                nn.Conv1d(64, self.jaw_dim, kernel_size=1, stride=1),
                nn.Conv1d(
                    self.hidden_dim,
                    self.output_feat_dim,
                    kernel_size=1,
                    stride=1,
                ),
            ]
        )
        self.checkpoint_load_report: dict[str, Any] | None = None
        self.checkpoint_audio_contract: dict[str, Any] | None = None
        self.runtime_audio_contract = self._runtime_audio_contract()

    def _runtime_audio_contract(self) -> dict[str, Any] | None:
        if self.wavlm_model_path is None:
            if self.require_audio_contract:
                raise ValueError(
                    "wavlm_model_path is required when "
                    "require_audio_contract=true"
                )
            return None
        if not self.wavlm_model_path.is_dir():
            raise FileNotFoundError(
                f"WavLM model directory does not exist: {self.wavlm_model_path}"
            )
        config = _load_json(self.wavlm_model_path / "config.json")
        preprocessor = _load_json(
            self.wavlm_model_path / "preprocessor_config.json"
        )
        if not isinstance(config, Mapping) or not isinstance(
            preprocessor,
            Mapping,
        ):
            raise ValueError("WavLM config and preprocessor JSON must be objects")
        if config.get("model_type") != "wavlm":
            raise ValueError(
                "runtime audio model must be WavLM; config model_type is "
                f"{config.get('model_type')!r}"
            )
        hidden_size = config.get("hidden_size")
        if hidden_size != self.input_feature_dim:
            raise ValueError(
                "runtime WavLM hidden size does not match face input: "
                f"{hidden_size!r} != {self.input_feature_dim}"
            )
        if config.get("add_adapter", False):
            raise ValueError("WavLM checkpoints with adapters are unsupported")
        if config.get("feat_extract_norm") != "layer":
            raise ValueError(
                "runtime WavLM feat_extract_norm must be 'layer'"
            )
        do_normalize = preprocessor.get("do_normalize")
        if not isinstance(do_normalize, bool):
            raise ValueError(
                "WavLM preprocessor_config.json must contain boolean "
                "do_normalize"
            )
        return {
            "audio_encoder": "wavlm",
            "audio_hidden_size": hidden_size,
            "audio_config_sha256": _canonical_json_sha256(config),
            "audio_preprocessor_sha256": _canonical_json_sha256(preprocessor),
            "audio_do_normalize": do_normalize,
        }

    def validate_runtime_audio_encoder_config(
        self,
        audio_encoder_config: Any,
    ) -> None:
        """Ensure the trainer will load the same WavLM directory we hashed.

        Hydra permits arbitrary command-line overrides.  Without this check,
        ``wavlm_model_path`` could validate one snapshot while the inference
        pipeline loads another snapshot through
        ``audio.audio_encoder.local_model_path``.
        """

        if self.wavlm_model_path is None:
            if self.require_audio_contract:
                raise RuntimeError(
                    "strict TalkSHOW integration has no validated WavLM path"
                )
            return
        getter = getattr(audio_encoder_config, "get", None)
        if not callable(getter):
            raise TypeError(
                "audio.audio_encoder must be a mapping-like configuration"
            )
        if getter("name") != "wavlm":
            raise RuntimeError(
                "TalkSHOW face condition requires "
                "audio.audio_encoder.name=wavlm"
            )
        if getter("quantization", False):
            raise RuntimeError(
                "TalkSHOW face condition requires "
                "audio.audio_encoder.quantization=false"
            )
        configured_path = getter("local_model_path")
        if not isinstance(configured_path, (str, Path)) or not str(
            configured_path
        ).strip():
            raise RuntimeError(
                "TalkSHOW face condition requires an explicit local "
                "audio.audio_encoder.local_model_path"
            )
        resolved_path = Path(configured_path).expanduser().resolve()
        if resolved_path != self.wavlm_model_path:
            raise RuntimeError(
                "trainer WavLM path differs from the path validated by the "
                "TalkSHOW face condition: "
                f"{resolved_path} != {self.wavlm_model_path}"
            )

    def validate_runtime_data_config(self, data_config: Any) -> None:
        """Reject CLI overrides that break the trained TalkSHOW contract."""

        getter = getattr(data_config, "get", None)
        if not callable(getter):
            raise TypeError("data config must be mapping-like")
        required_values = {
            "audio_sr": 16000,
            "audio_fps": 16000,
            "pose_fps": 30,
            "pose_length": self.expected_pose_length,
            "joints": 55,
            "ori_joints": "beat_smplx_joints",
            "load_audio_feat": True,
        }
        for name, expected in required_values.items():
            observed = getter(name)
            if observed != expected:
                raise RuntimeError(
                    "TalkSHOW face condition data contract mismatch: "
                    f"data.{name}={observed!r}, expected {expected!r}"
                )

    def validate_runtime_training_config(self, train_config: Any) -> None:
        """Reject a block size that disagrees with the selected SHOW profile."""

        getter = getattr(train_config, "get", None)
        if not callable(getter):
            raise TypeError("train config must be mapping-like")
        observed = getter("num_frame_per_block")
        if observed != self.expected_block_size:
            raise RuntimeError(
                "TalkSHOW face condition training contract mismatch: "
                f"train.num_frame_per_block={observed!r}, expected "
                f"{self.expected_block_size!r}"
            )

    def _validate_checkpoint_audio_contract(
        self,
        payload: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        contract = payload.get("resume_contract")
        if not isinstance(contract, Mapping):
            if self.require_audio_contract:
                raise RuntimeError(
                    "TalkSHOW face checkpoint has no resume_contract; use a "
                    "checkpoint produced by the current large-batch trainer, "
                    "or explicitly disable require_audio_contract for legacy "
                    "experiments"
                )
            return None

        required = {
            "schema_version",
            "trainer",
            "audio_encoder",
            "audio_hidden_size",
            "audio_config_sha256",
            "audio_preprocessor_sha256",
            "audio_do_normalize",
            "freeze_audio_encoder",
            "num_classes",
            "speakers",
            "face_loss",
        }
        missing = sorted(required.difference(contract))
        if missing:
            raise RuntimeError(
                "TalkSHOW face checkpoint audio contract is incomplete; "
                f"missing: {missing}"
            )
        normalized = {key: contract[key] for key in required}
        schema_version = normalized["schema_version"]
        if schema_version not in {1, 2}:
            raise RuntimeError(
                "unsupported TalkSHOW face resume_contract schema_version: "
                f"{schema_version!r}"
            )
        if schema_version == 2:
            model_file_hashes = contract.get("audio_model_files_sha256")
            if not isinstance(model_file_hashes, Mapping) or not model_file_hashes:
                raise RuntimeError(
                    "TalkSHOW face schema 2 contract has no "
                    "audio_model_files_sha256"
                )
            normalized["audio_model_files_sha256"] = {
                str(name): str(digest).lower()
                for name, digest in model_file_hashes.items()
            }
            for name, digest in normalized[
                "audio_model_files_sha256"
            ].items():
                if (
                    not name
                    or Path(name).is_absolute()
                    or ".." in Path(name).parts
                    or len(digest) != 64
                    or any(
                        character not in "0123456789abcdef"
                        for character in digest
                    )
                ):
                    raise RuntimeError(
                        "TalkSHOW face checkpoint contains an invalid WavLM "
                        f"file digest for {name!r}"
                    )
        else:
            warnings.warn(
                "TalkSHOW face checkpoint uses resume_contract schema 1; "
                "WavLM config/preprocessor hashes are checked, but the face "
                "training weight files cannot be proven. This legacy resume "
                "lineage intentionally remains schema 1.",
                RuntimeWarning,
                stacklevel=2,
            )
        if normalized["trainer"] != "talkshow_face_large_batch":
            raise RuntimeError(
                "TalkSHOW face checkpoint must come from the modern "
                "talkshow_face_large_batch trainer"
            )
        if normalized["audio_encoder"] != "wavlm":
            raise RuntimeError("TalkSHOW face checkpoint was not trained with WavLM")
        if normalized["audio_hidden_size"] != self.input_feature_dim:
            raise RuntimeError(
                "face checkpoint audio hidden size does not match adapter: "
                f"{normalized['audio_hidden_size']} != {self.input_feature_dim}"
            )
        if normalized["num_classes"] != self.num_person:
            raise RuntimeError(
                "face checkpoint identity count does not match adapter: "
                f"{normalized['num_classes']} != {self.num_person}"
            )
        if not isinstance(normalized["speakers"], (list, tuple)):
            raise RuntimeError(
                "face checkpoint speakers contract must be a list"
            )
        checkpoint_speakers = tuple(str(name) for name in normalized["speakers"])
        if (
            len(checkpoint_speakers) != self.num_person
            or set(checkpoint_speakers) != set(self.expected_speakers)
        ):
            raise RuntimeError(
                "face checkpoint speakers do not match the configured "
                "reproduction protocol: "
                f"{checkpoint_speakers!r} != "
                f"{self.expected_speakers!r}"
            )
        if normalized["freeze_audio_encoder"] is not True:
            raise RuntimeError(
                "main-model integration requires a face checkpoint trained "
                "with a frozen WavLM"
            )
        if (
            self.required_face_loss is not None
            and normalized["face_loss"] != self.required_face_loss
        ):
            raise RuntimeError(
                "face checkpoint loss protocol does not match this run: "
                f"{normalized['face_loss']!r} != "
                f"{self.required_face_loss!r}"
            )
        if self.runtime_audio_contract is not None:
            for key in (
                "audio_encoder",
                "audio_hidden_size",
                "audio_config_sha256",
                "audio_preprocessor_sha256",
                "audio_do_normalize",
            ):
                if normalized[key] != self.runtime_audio_contract[key]:
                    raise RuntimeError(
                        "face checkpoint and runtime WavLM disagree on "
                        f"{key}: {normalized[key]!r} != "
                        f"{self.runtime_audio_contract[key]!r}"
                    )
        return normalized

    def validate_dataset_contract(
        self,
        metadata_by_split: Mapping[str, Mapping[str, Any]],
    ) -> None:
        """Cross-check LMDB metadata against face and runtime WavLM metadata."""

        if not metadata_by_split:
            if self.require_audio_contract:
                raise RuntimeError("TalkSHOW LMDB metadata is unavailable")
            return
        first_split, first_metadata = next(iter(metadata_by_split.items()))
        shared_lmdb_fields = (
            "split_manifest_sha256",
            "split_view_summary_sha256",
            "split_view_counts",
            "hand_component_sha256",
            "audio_model_files_sha256",
            "face_context_frames",
            "face_context_protocol",
            "pid_offset",
        )
        for split, metadata in metadata_by_split.items():
            declared_split = metadata.get("split")
            # The validation loader may explicitly opt into a strict test
            # LMDB. TalkSHOWLMDBDataset has already checked its configured
            # expected_split before metadata reaches this cross-split check.
            split_matches = declared_split == split or (
                {declared_split, split} == {"val", "validation"}
            ) or (
                split == "validation" and declared_split == "test"
            )
            if not split_matches:
                raise RuntimeError(
                    f"TalkSHOW {split} LMDB declares split "
                    f"{declared_split!r}"
                )
            if metadata.get("feature_scope") != "full_clip":
                raise RuntimeError(
                    f"TalkSHOW {split} LMDB must use full_clip WavLM features"
                )
            if metadata.get("face_context_frames") != self.context_radius_frames:
                raise RuntimeError(
                    f"TalkSHOW {split} LMDB face context radius is "
                    f"{metadata.get('face_context_frames')!r}, expected "
                    f"{self.context_radius_frames}"
                )
            if (
                metadata.get("face_context_protocol")
                != "source_clip_valid_slice_v1"
            ):
                raise RuntimeError(
                    f"TalkSHOW {split} LMDB has unsupported face context "
                    f"protocol {metadata.get('face_context_protocol')!r}"
                )
            if metadata.get("audio_feature_dim") != self.input_feature_dim:
                raise RuntimeError(
                    f"TalkSHOW {split} LMDB feature dim is "
                    f"{metadata.get('audio_feature_dim')!r}, expected "
                    f"{self.input_feature_dim}"
                )
            if (
                metadata.get("pid_offset")
                != self.expected_lmdb_pid_offset
            ):
                raise RuntimeError(
                    f"TalkSHOW {split} LMDB pid_offset is "
                    f"{metadata.get('pid_offset')!r}, expected "
                    f"{self.expected_lmdb_pid_offset}"
                )
            speaker_ids = metadata.get("speaker_ids")
            if not isinstance(speaker_ids, Mapping):
                raise RuntimeError(
                    f"TalkSHOW {split} LMDB has no speaker_ids mapping"
                )
            expected_mapping = self.expected_speaker_ids
            if dict(speaker_ids) != expected_mapping:
                raise RuntimeError(
                    f"TalkSHOW {split} LMDB speaker/PID mapping does not "
                    f"match {expected_mapping}"
                )
            if split != first_split:
                for key in shared_lmdb_fields:
                    if metadata.get(key) != first_metadata.get(key):
                        raise RuntimeError(
                            f"TalkSHOW {split} and {first_split} LMDBs "
                            f"disagree on {key}"
                        )

            reference = self.checkpoint_audio_contract
            if reference is None:
                reference = self.runtime_audio_contract
            if reference is not None:
                for key in (
                    "audio_config_sha256",
                    "audio_preprocessor_sha256",
                    "audio_do_normalize",
                ):
                    if metadata.get(key) != reference[key]:
                        raise RuntimeError(
                            f"TalkSHOW {split} LMDB and face/runtime WavLM "
                            f"disagree on {key}: {metadata.get(key)!r} != "
                            f"{reference[key]!r}"
                        )
                if "audio_model_files_sha256" in reference:
                    checkpoint_hashes = {
                        str(name): str(digest).lower()
                        for name, digest in reference[
                            "audio_model_files_sha256"
                        ].items()
                    }
                    lmdb_hashes = {
                        str(name): str(digest).lower()
                        for name, digest in metadata.get(
                            "audio_model_files_sha256",
                            {},
                        ).items()
                    }
                    if checkpoint_hashes != lmdb_hashes:
                        raise RuntimeError(
                            f"TalkSHOW {split} LMDB and face checkpoint "
                            "disagree on WavLM model file hashes"
                        )

    def _local_speaker_ids(
        self,
        pid: torch.Tensor,
        *,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        if not isinstance(pid, torch.Tensor):
            raise TypeError(f"pid must be a torch.Tensor, got {type(pid)!r}")
        if pid.numel() != batch_size:
            raise ValueError(
                "pid must contain exactly one speaker ID per sample: "
                f"got shape {tuple(pid.shape)} for batch size {batch_size}"
            )

        flat_pid = pid.reshape(batch_size).to(device=device)
        if torch.is_floating_point(flat_pid):
            rounded_pid = flat_pid.round()
            if not torch.equal(flat_pid, rounded_pid):
                raise ValueError("pid must contain integer-valued speaker IDs")
            flat_pid = rounded_pid
        local_pid = flat_pid.to(dtype=torch.long) - self.speaker_id_offset

        valid = (local_pid >= 0) & (local_pid < self.num_person)
        if not torch.all(valid):
            invalid_ids = torch.unique(flat_pid[~valid]).detach().cpu().tolist()
            first_id = self.speaker_id_offset
            last_id = first_id + self.num_person - 1
            raise ValueError(
                "TalkSHOW face checkpoint only supports runtime network IDs "
                f"{first_id}..{last_id}; got {invalid_ids}"
            )
        return local_pid

    def _encode(
        self,
        audio_features: torch.Tensor,
        pid: torch.Tensor,
    ) -> torch.Tensor:
        if not isinstance(audio_features, torch.Tensor):
            raise TypeError(
                "audio_features must be a torch.Tensor, "
                f"got {type(audio_features)!r}"
            )
        if audio_features.ndim != 3:
            raise ValueError(
                "audio_features must have shape [B, T, C], "
                f"got {tuple(audio_features.shape)}"
            )
        if audio_features.shape[-1] != self.input_feature_dim:
            raise ValueError(
                "TalkSHOW face condition expects WavLM hidden size "
                f"{self.input_feature_dim}, got {audio_features.shape[-1]}"
            )
        if audio_features.dtype != torch.float32:
            raise TypeError(
                "TalkSHOW face condition requires float32 audio features; "
                f"got {audio_features.dtype}"
            )

        local_pid = self._local_speaker_ids(
            pid,
            batch_size=audio_features.shape[0],
            device=audio_features.device,
        )
        identity = F.one_hot(
            local_pid,
            num_classes=self.num_person,
        ).to(dtype=audio_features.dtype)

        features = self.audio_feature_map(audio_features).transpose(1, 2)
        return self.audio_middle(features, identity)

    def forward_face(
        self,
        audio_features: torch.Tensor,
        pid: torch.Tensor,
    ) -> torch.Tensor:
        """Return TalkSHOW's full ``[jaw(3), expression(100)]`` prediction."""

        features = self._encode(audio_features, pid)
        outputs = [
            output_layer(decoder(features))
            for decoder, output_layer in zip(self.decoder, self.final_out)
        ]
        return torch.cat(outputs, dim=1).transpose(1, 2)

    def forward(
        self,
        audio_features: torch.Tensor,
        pid: torch.Tensor,
    ) -> torch.Tensor:
        """Return the 100-D expression condition expected by the main model."""

        face = self.forward_face(audio_features, pid)
        return face[..., self.jaw_dim :]

    def forward_contextual(
        self,
        audio_features: torch.Tensor,
        pid: torch.Tensor,
        *,
        valid_lengths: torch.Tensor,
        crop_starts: torch.Tensor,
        target_length: int,
    ) -> torch.Tensor:
        """Run the face head on source-valid context and crop central frames.

        ``audio_features`` is a fixed-length carrier used only for batching.
        Frames after each sample's ``valid_length`` never enter the nonlinear
        temporal network.  Samples with equal bounds are grouped so the common
        interior-window case remains a single batched face-head call.
        """

        if not isinstance(audio_features, torch.Tensor) or audio_features.ndim != 3:
            raise ValueError(
                "contextual audio_features must have shape [B, Tctx, C]"
            )
        batch_size, carrier_length, _ = audio_features.shape
        if (
            isinstance(target_length, bool)
            or not isinstance(target_length, int)
            or target_length <= 0
        ):
            raise ValueError("target_length must be a positive integer")
        if not isinstance(valid_lengths, torch.Tensor) or not isinstance(
            crop_starts,
            torch.Tensor,
        ):
            raise TypeError("valid_lengths and crop_starts must be tensors")
        if valid_lengths.numel() != batch_size or crop_starts.numel() != batch_size:
            raise ValueError(
                "face context bounds must contain one value per sample"
            )

        valid_values = valid_lengths.detach().reshape(-1).cpu().tolist()
        crop_values = crop_starts.detach().reshape(-1).cpu().tolist()
        groups: dict[tuple[int, int], list[int]] = {}
        for sample_index, (valid_raw, crop_raw) in enumerate(
            zip(valid_values, crop_values)
        ):
            valid_length = int(valid_raw)
            crop_start = int(crop_raw)
            if valid_length != valid_raw or crop_start != crop_raw:
                raise ValueError("face context bounds must be integer-valued")
            if not (
                0 < valid_length <= carrier_length
                and 0 <= crop_start <= self.context_radius_frames
                and crop_start + target_length <= valid_length
                and valid_length - crop_start - target_length
                <= self.context_radius_frames
            ):
                raise ValueError(
                    "invalid face context bounds for sample "
                    f"{sample_index}: valid_length={valid_length}, "
                    f"crop_start={crop_start}, target_length={target_length}, "
                    f"carrier_length={carrier_length}"
                )
            groups.setdefault((valid_length, crop_start), []).append(
                sample_index
            )

        output = audio_features.new_empty(
            (batch_size, target_length, self.output_feat_dim)
        )
        for (valid_length, crop_start), sample_indices in groups.items():
            index = torch.tensor(
                sample_indices,
                dtype=torch.long,
                device=audio_features.device,
            )
            grouped_features = audio_features.index_select(0, index)[
                :, :valid_length
            ]
            grouped_pid = pid.reshape(batch_size).index_select(0, index)
            grouped_expression = self(
                grouped_features,
                pid=grouped_pid,
            )
            grouped_expression = grouped_expression[
                :, crop_start:crop_start + target_length
            ]
            output.index_copy_(0, index, grouped_expression)
        return output

    @classmethod
    def _normalize_source_key(cls, key: str) -> str:
        normalized = key
        while normalized.startswith("module."):
            normalized = normalized[len("module.") :]
        if normalized.startswith("generator."):
            candidate = normalized[len("generator.") :]
            if candidate.startswith(
                cls._LOADABLE_PREFIXES + cls._IGNORED_SOURCE_PREFIXES
            ):
                normalized = candidate
        return normalized

    @classmethod
    def _looks_like_generator_state(
        cls,
        candidate: Mapping[str, Any],
    ) -> bool:
        for key, value in candidate.items():
            if not isinstance(key, str) or not isinstance(value, torch.Tensor):
                continue
            normalized = cls._normalize_source_key(key)
            if normalized.startswith(cls._LOADABLE_PREFIXES):
                return True
        return False

    @classmethod
    def _extract_generator_state(
        cls,
        checkpoint: Mapping[str, Any],
    ) -> tuple[Mapping[str, Any], str]:
        current: Any = checkpoint
        layout: list[str] = []
        for _ in range(4):
            if not isinstance(current, Mapping):
                break
            if cls._looks_like_generator_state(current):
                return current, ".".join(layout) if layout else "<root>"
            nested = current.get("generator")
            if not isinstance(nested, Mapping):
                break
            current = nested
            layout.append("generator")
        raise ValueError(
            "not a supported TalkSHOW face checkpoint: expected generator "
            "weights at the root, ['generator'], or "
            "['generator']['generator']"
        )

    def load_pretrained_checkpoint(
        self,
        checkpoint: str | Path | Mapping[str, Any],
    ) -> dict[str, Any]:
        """Strictly load the post-WavLM head from a TalkSHOW checkpoint.

        The frozen WavLM backbone and the registered-but-unused TalkSHOW GRU
        are intentionally ignored.  Every parameter used by ``forward_face``
        must otherwise exist and match exactly.
        """

        source = str(checkpoint) if isinstance(checkpoint, (str, Path)) else None
        if source is not None:
            try:
                payload = torch.load(
                    source,
                    map_location="cpu",
                    weights_only=True,
                )
            except TypeError:
                # PyTorch versions predating the weights_only argument.
                payload = torch.load(source, map_location="cpu")
        else:
            payload = checkpoint
        if not isinstance(payload, Mapping):
            raise TypeError(
                "TalkSHOW checkpoint must contain a mapping, "
                f"got {type(payload)!r}"
            )

        self.checkpoint_audio_contract = (
            self._validate_checkpoint_audio_contract(payload)
        )
        generator_state, layout = self._extract_generator_state(payload)
        normalized_state: dict[str, torch.Tensor] = {}
        ignored_prefix_counts: Counter[str] = Counter()
        unexpected: list[str] = []

        expected_state = self.state_dict()
        expected_keys = set(expected_state)
        for source_key, value in generator_state.items():
            if not isinstance(source_key, str):
                unexpected.append(repr(source_key))
                continue
            key = self._normalize_source_key(source_key)
            if key in expected_keys:
                if not isinstance(value, torch.Tensor):
                    raise TypeError(
                        f"checkpoint value for {source_key!r} is not a tensor"
                    )
                if key in normalized_state:
                    raise ValueError(
                        f"duplicate TalkSHOW checkpoint key after normalization: {key}"
                    )
                normalized_state[key] = value
                continue

            ignored_prefix = next(
                (
                    prefix
                    for prefix in self._IGNORED_SOURCE_PREFIXES
                    if key.startswith(prefix)
                ),
                None,
            )
            if ignored_prefix is not None:
                ignored_prefix_counts[ignored_prefix] += 1
            else:
                unexpected.append(source_key)

        missing = sorted(expected_keys - set(normalized_state))
        if missing or unexpected:
            details: list[str] = []
            if missing:
                details.append(f"missing keys: {missing}")
            if unexpected:
                details.append(f"unexpected keys: {sorted(unexpected)}")
            raise RuntimeError(
                "TalkSHOW face checkpoint is incompatible; " + "; ".join(details)
            )

        non_fp32 = sorted(
            key
            for key, value in normalized_state.items()
            if torch.is_floating_point(value) and value.dtype != torch.float32
        )
        if non_fp32:
            raise RuntimeError(
                "TalkSHOW face checkpoint must remain float32; non-FP32 keys: "
                f"{non_fp32}"
            )
        non_finite = sorted(
            key
            for key, value in normalized_state.items()
            if torch.is_floating_point(value)
            and not torch.isfinite(value).all().item()
        )
        if non_finite:
            raise RuntimeError(
                "TalkSHOW face checkpoint contains NaN or Inf values in keys: "
                f"{non_finite}"
            )

        try:
            self.load_state_dict(normalized_state, strict=True)
        except RuntimeError as error:
            raise RuntimeError(
                "TalkSHOW face checkpoint tensor shapes are incompatible with "
                "the configured face condition model"
            ) from error

        report = {
            "source": source or "<mapping>",
            "layout": layout,
            "loaded_keys": len(normalized_state),
            "ignored_prefix_counts": dict(sorted(ignored_prefix_counts.items())),
            "audio_contract": (
                "validated"
                if self.checkpoint_audio_contract is not None
                else "legacy-unavailable"
            ),
        }
        self.checkpoint_load_report = report
        return report


__all__ = ["TalkSHOWFaceConditionModel"]
