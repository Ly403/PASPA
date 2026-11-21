#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
usage: train_face.sh \
  --upstream-dir DIR --data-root DIR --output-dir DIR --run-id ID \
  [--smplx-assets DIR] \
  [--audio-encoder wav2vec2|wavlm] [--audio-model-path DIR] \
  [--wav2vec-path DIR] \
  [--num-gpus N] [--batch-size-per-gpu N] [--epochs N] \
  [--resume-from FILE] \
  [--smoke] [--corrected-face-loss]

--wav2vec-path is a backward-compatible alias for:
  --audio-encoder wav2vec2 --audio-model-path DIR

WavLM requires --audio-model-path.  Use the same local
patrickvonplaten/wavlm-libri-clean-100h-large Hugging Face snapshot as the main
codebase; the directory must include preprocessor_config.json.

The defaults remain the released single-GPU protocol:
  --num-gpus 1 --batch-size-per-gpu 1 --epochs 100

More than one GPU or a per-GPU batch larger than one selects the opt-in
variable-length WavLM trainer.  --smoke always overrides --epochs to 1.
Outside --smoke, the released trainer requires 30 or a multiple of 50 epochs
so its final epoch is one of the upstream checkpoint milestones.

--resume-from restores an epoch-boundary face checkpoint into a new run ID.
--epochs remains the total target epoch count, not the number of extra epochs.
EOF
}

UPSTREAM_DIR=""
DATA_ROOT=""
SMPLX_ASSETS=""
OUTPUT_DIR=""
RUN_ID=""
AUDIO_ENCODER="wav2vec2"
AUDIO_MODEL_PATH=""
WAV2VEC_PATH=""
NUM_GPUS=1
BATCH_SIZE_PER_GPU=1
EPOCHS=100
RESUME_FROM=""
SMOKE=0
LEGACY_FACE_LOSS=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --upstream-dir|--data-root|--smplx-assets|--output-dir|--run-id|\
    --audio-encoder|--audio-model-path|--wav2vec-path|--num-gpus|\
    --batch-size-per-gpu|--epochs|--resume-from)
      if [[ $# -lt 2 || "$2" == --* ]]; then
        echo "missing value for argument: $1" >&2
        usage
        exit 2
      fi
      ;;
  esac
  case "$1" in
    --upstream-dir) UPSTREAM_DIR="$2"; shift 2 ;;
    --data-root) DATA_ROOT="$2"; shift 2 ;;
    --smplx-assets) SMPLX_ASSETS="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --run-id) RUN_ID="$2"; shift 2 ;;
    --audio-encoder) AUDIO_ENCODER="$2"; shift 2 ;;
    --audio-model-path) AUDIO_MODEL_PATH="$2"; shift 2 ;;
    --wav2vec-path) WAV2VEC_PATH="$2"; shift 2 ;;
    --num-gpus) NUM_GPUS="$2"; shift 2 ;;
    --batch-size-per-gpu) BATCH_SIZE_PER_GPU="$2"; shift 2 ;;
    --epochs) EPOCHS="$2"; shift 2 ;;
    --resume-from) RESUME_FROM="$2"; shift 2 ;;
    --smoke) SMOKE=1; shift ;;
    --corrected-face-loss) LEGACY_FACE_LOSS=0; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

for VALUE_NAME in UPSTREAM_DIR DATA_ROOT OUTPUT_DIR RUN_ID; do
  if [[ -z "${!VALUE_NAME}" ]]; then
    echo "missing required argument: ${VALUE_NAME}" >&2
    usage
    exit 2
  fi
done

for INTEGER_VALUE_NAME in NUM_GPUS BATCH_SIZE_PER_GPU EPOCHS; do
  if [[ ! "${!INTEGER_VALUE_NAME}" =~ ^[1-9][0-9]*$ ]]; then
    echo "${INTEGER_VALUE_NAME} must be a positive integer" >&2
    exit 2
  fi
done

case "${AUDIO_ENCODER}" in
  wav2vec2|wavlm) ;;
  *)
    echo "unsupported audio encoder: ${AUDIO_ENCODER}" >&2
    usage
    exit 2
    ;;
esac

if [[ -n "${WAV2VEC_PATH}" ]]; then
  if [[ "${AUDIO_ENCODER}" != "wav2vec2" ]]; then
    echo "--wav2vec-path cannot be combined with --audio-encoder ${AUDIO_ENCODER}" >&2
    exit 2
  fi
  if [[ -n "${AUDIO_MODEL_PATH}" ]]; then
    echo "--wav2vec-path and --audio-model-path are mutually exclusive" >&2
    exit 2
  fi
  AUDIO_MODEL_PATH="${WAV2VEC_PATH}"
fi

if [[ "${AUDIO_ENCODER}" == "wavlm" && -z "${AUDIO_MODEL_PATH}" ]]; then
  echo "--audio-model-path is required when --audio-encoder wavlm" >&2
  exit 2
fi

if [[ "${SMOKE}" -eq 1 && -n "${RESUME_FROM}" ]]; then
  echo "--smoke and --resume-from are mutually exclusive" >&2
  exit 2
fi

USE_LARGE_BATCH_TRAINER=0
if [[ "${NUM_GPUS}" -gt 1 || "${BATCH_SIZE_PER_GPU}" -gt 1 ]]; then
  USE_LARGE_BATCH_TRAINER=1
  if [[ "${AUDIO_ENCODER}" != "wavlm" ]]; then
    echo "multi-GPU or batch-size-per-gpu > 1 currently requires WavLM" >&2
    exit 2
  fi
fi

if [[ ! "${RUN_ID}" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "run ID may contain only letters, digits, dot, underscore, and dash" >&2
  exit 2
fi

UPSTREAM_DIR="$(cd -- "${UPSTREAM_DIR}" && pwd)"
DATA_ROOT="$(cd -- "${DATA_ROOT}" && pwd)"
if [[ -n "${SMPLX_ASSETS}" ]]; then
  SMPLX_ASSETS="$(cd -- "${SMPLX_ASSETS}" && pwd)"
fi
if [[ -n "${AUDIO_MODEL_PATH}" ]]; then
  AUDIO_MODEL_PATH="$(cd -- "${AUDIO_MODEL_PATH}" && pwd)"
fi
if [[ -n "${RESUME_FROM}" ]]; then
  if [[ ! -f "${RESUME_FROM}" ]]; then
    echo "resume checkpoint does not exist: ${RESUME_FROM}" >&2
    exit 1
  fi
  RESUME_FROM="$(
    cd -- "$(dirname -- "${RESUME_FROM}")"
    printf '%s/%s\n' "$(pwd)" "$(basename -- "${RESUME_FROM}")"
  )"
fi
mkdir -p "${OUTPUT_DIR}"
OUTPUT_DIR="$(cd -- "${OUTPUT_DIR}" && pwd)"

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
EXPECTED_COMMIT="9aef82df5ff1082f0cfa0cfc116c0b7208e85d5b"
ACTUAL_COMMIT="$(git -C "${UPSTREAM_DIR}" rev-parse HEAD)"
if [[ "${ACTUAL_COMMIT}" != "${EXPECTED_COMMIT}" ]]; then
  echo "unexpected TalkSHOW commit: ${ACTUAL_COMMIT}" >&2
  exit 1
fi
python "${SCRIPT_DIR}/scripts/patch_upstream.py" \
  --upstream-dir "${UPSTREAM_DIR}" --check

if [[ ! -d "${DATA_ROOT}" ]]; then
  echo "data root does not exist: ${DATA_ROOT}" >&2
  exit 1
fi

SAVE_EVERY=50
PRINT_EVERY=1000
if [[ "${SMOKE}" -eq 1 ]]; then
  EPOCHS=1
  SAVE_EVERY=1
  PRINT_EVERY=1
fi

if [[
  "${USE_LARGE_BATCH_TRAINER}" -eq 0 &&
  "${SMOKE}" -eq 0 &&
  "${EPOCHS}" -ne 30 &&
  $((EPOCHS % SAVE_EVERY)) -ne 0
 ]]; then
  echo "the released trainer only saves its final epoch at 30 or a multiple of ${SAVE_EVERY}; use --smoke for a one-epoch check" >&2
  exit 2
fi

RUN_ROOT="${OUTPUT_DIR}/${RUN_ID}"
if [[ -e "${RUN_ROOT}" ]]; then
  echo "refusing to reuse an existing run directory: ${RUN_ROOT}" >&2
  exit 1
fi
mkdir -p "${RUN_ROOT}/config" "${RUN_ROOT}/cache" "${RUN_ROOT}/experiments"

CONFIG_PATH="${RUN_ROOT}/config/face.json"
CONFIG_ARGS=(
  --upstream-dir "${UPSTREAM_DIR}" \
  --data-root "${DATA_ROOT}" \
  --output "${CONFIG_PATH}" \
  --cache-tag "${RUN_ID}" \
  --epochs "${EPOCHS}" \
  --save-every "${SAVE_EVERY}" \
  --print-every "${PRINT_EVERY}" \
  --batch-size "${BATCH_SIZE_PER_GPU}" \
  --world-size "${NUM_GPUS}" \
  --audio-encoder "${AUDIO_ENCODER}"
)
if [[ "${LEGACY_FACE_LOSS}" -eq 1 ]]; then
  CONFIG_ARGS+=(--face-loss legacy)
else
  CONFIG_ARGS+=(--face-loss corrected)
fi
if [[ "${USE_LARGE_BATCH_TRAINER}" -eq 1 ]]; then
  CONFIG_ARGS+=(
    --variable-length-batching \
    --length-bucket 8 \
    --no-drop-last
  )
fi
if [[ -n "${SMPLX_ASSETS}" ]]; then
  CONFIG_ARGS+=(--smplx-assets "${SMPLX_ASSETS}")
fi
if [[ -n "${AUDIO_MODEL_PATH}" ]]; then
  CONFIG_ARGS+=(--audio-model-path "${AUDIO_MODEL_PATH}")
fi
if [[ -n "${RESUME_FROM}" ]]; then
  CONFIG_ARGS+=(--resume-from "${RESUME_FROM}")
fi
python "${SCRIPT_DIR}/scripts/configure_face.py" "${CONFIG_ARGS[@]}"

export TALKSHOW_LEGACY_FACE_LOSS="${LEGACY_FACE_LOSS}"
export TALKSHOW_WRITE_DATA_CACHE=0
# The run-local config is authoritative.  Do not let a stale parent-shell
# variable silently replace its recorded model source.
unset TALKSHOW_WAV2VEC_PATH
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export TALKSHOW_RUN_DATE="$(date +%F)"

EXP_NAME="talkshow_face_${RUN_ID}"
(
  cd "${UPSTREAM_DIR}"
  if [[ "${USE_LARGE_BATCH_TRAINER}" -eq 1 ]]; then
    LARGE_BATCH_ARGS=(
      "${SCRIPT_DIR}/scripts/train_face_large_batch.py"
      --upstream-dir "${UPSTREAM_DIR}"
      --save-dir "${RUN_ROOT}/experiments"
      --exp-name "${EXP_NAME}"
      --speakers oliver seth conan chemistry
      --config-file "${CONFIG_PATH}"
    )
    if [[ -n "${RESUME_FROM}" ]]; then
      LARGE_BATCH_ARGS+=(--resume-from "${RESUME_FROM}")
    fi
    if [[ "${NUM_GPUS}" -gt 1 ]]; then
      torchrun --standalone --nproc_per_node "${NUM_GPUS}" \
        "${LARGE_BATCH_ARGS[@]}"
    else
      python -W ignore "${LARGE_BATCH_ARGS[@]}"
    fi
  else
    if [[ -n "${RESUME_FROM}" ]]; then
      python -W ignore \
        "${SCRIPT_DIR}/scripts/train_face_single_resume.py" \
        --upstream-dir "${UPSTREAM_DIR}" \
        --gpu 0 \
        --save-dir "${RUN_ROOT}/experiments" \
        --exp-name "${EXP_NAME}" \
        --speakers oliver seth conan chemistry \
        --config-file "${CONFIG_PATH}" \
        --resume-from "${RESUME_FROM}"
    else
      python -W ignore scripts/train.py \
        --gpu 0 \
        --save_dir "${RUN_ROOT}/experiments" \
        --exp_name "${EXP_NAME}" \
        --speakers oliver seth conan chemistry \
        --config_file "${CONFIG_PATH}"
    fi
  fi
)

TRAIN_DIR="${RUN_ROOT}/experiments/${TALKSHOW_RUN_DATE}-${EXP_NAME}-face"
LAST_EPOCH=$((EPOCHS - 1))
CHECKPOINT="${TRAIN_DIR}/ckpt-${LAST_EPOCH}.pth"

VERIFY_ARGS=(
  --checkpoint "${CHECKPOINT}"
  --expected-epoch "${LAST_EPOCH}"
  --min-global-steps 1
  --require-face-wrapper-state
)
if [[ "${USE_LARGE_BATCH_TRAINER}" -eq 1 ]]; then
  VERIFY_ARGS+=(--require-resume-metadata)
fi
python "${SCRIPT_DIR}/scripts/verify_face_checkpoint.py" "${VERIFY_ARGS[@]}"

echo "TalkSHOW face training completed: ${CHECKPOINT}"
