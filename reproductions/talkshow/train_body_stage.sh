#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
usage: train_body_stage.sh \
  --stage {vq,pixel} --upstream-dir DIR --data-root DIR \
  --output-dir DIR --run-id ID [--vq-checkpoint FILE] [--smoke]
EOF
}

STAGE_NAME=""
UPSTREAM_DIR=""
DATA_ROOT=""
OUTPUT_DIR=""
RUN_ID=""
VQ_CHECKPOINT=""
SMOKE=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --stage) STAGE_NAME="$2"; shift 2 ;;
    --upstream-dir) UPSTREAM_DIR="$2"; shift 2 ;;
    --data-root) DATA_ROOT="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --run-id) RUN_ID="$2"; shift 2 ;;
    --vq-checkpoint) VQ_CHECKPOINT="$2"; shift 2 ;;
    --smoke) SMOKE=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

if [[ "${STAGE_NAME}" != "vq" && "${STAGE_NAME}" != "pixel" ]]; then
  echo "--stage must be vq or pixel" >&2
  exit 2
fi
for VALUE_NAME in UPSTREAM_DIR DATA_ROOT OUTPUT_DIR RUN_ID; do
  if [[ -z "${!VALUE_NAME}" ]]; then
    echo "missing required argument: ${VALUE_NAME}" >&2
    usage
    exit 2
  fi
done
if [[ ! "${RUN_ID}" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "run ID may contain only letters, digits, dot, underscore, and dash" >&2
  exit 2
fi
if [[ "${STAGE_NAME}" == "pixel" && -z "${VQ_CHECKPOINT}" ]]; then
  echo "--vq-checkpoint is required for the pixel stage" >&2
  exit 2
fi
if [[ "${STAGE_NAME}" == "vq" && -n "${VQ_CHECKPOINT}" ]]; then
  echo "--vq-checkpoint is only valid for the pixel stage" >&2
  exit 2
fi
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" || "${CUDA_VISIBLE_DEVICES}" == *,* ]]; then
  echo "CUDA_VISIBLE_DEVICES must contain exactly one physical GPU ID" >&2
  exit 2
fi

UPSTREAM_DIR="$(cd -- "${UPSTREAM_DIR}" && pwd)"
DATA_ROOT="$(cd -- "${DATA_ROOT}" && pwd)"
mkdir -p "${OUTPUT_DIR}"
OUTPUT_DIR="$(cd -- "${OUTPUT_DIR}" && pwd)"
if [[ -n "${VQ_CHECKPOINT}" ]]; then
  VQ_DIR="$(cd -- "$(dirname -- "${VQ_CHECKPOINT}")" && pwd)"
  VQ_CHECKPOINT="${VQ_DIR}/$(basename -- "${VQ_CHECKPOINT}")"
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
EXPECTED_COMMIT="9aef82df5ff1082f0cfa0cfc116c0b7208e85d5b"
ACTUAL_COMMIT="$(git -C "${UPSTREAM_DIR}" rev-parse HEAD)"
if [[ "${ACTUAL_COMMIT}" != "${EXPECTED_COMMIT}" ]]; then
  echo "unexpected TalkSHOW commit: ${ACTUAL_COMMIT}" >&2
  exit 1
fi
python "${SCRIPT_DIR}/scripts/patch_upstream.py" \
  --upstream-dir "${UPSTREAM_DIR}" --check

RUN_ROOT="${OUTPUT_DIR}/${RUN_ID}"
if [[ -e "${RUN_ROOT}" ]]; then
  echo "refusing to reuse an existing run directory: ${RUN_ROOT}" >&2
  exit 1
fi
mkdir -p "${RUN_ROOT}/config" "${RUN_ROOT}/experiments"

EPOCHS=100
BATCH_SIZE=128
SAVE_EVERY=50
PRINT_EVERY=200
if [[ "${SMOKE}" -eq 1 ]]; then
  EPOCHS=1
  BATCH_SIZE=1
  SAVE_EVERY=1
  PRINT_EVERY=1
fi

CONFIG_PATH="${RUN_ROOT}/config/body_${STAGE_NAME}.json"
CONFIG_ARGS=(
  --stage "${STAGE_NAME}"
  --upstream-dir "${UPSTREAM_DIR}"
  --data-root "${DATA_ROOT}"
  --output "${CONFIG_PATH}"
  --cache-tag "${RUN_ID}"
  --epochs "${EPOCHS}"
  --batch-size "${BATCH_SIZE}"
  --save-every "${SAVE_EVERY}"
  --print-every "${PRINT_EVERY}"
)
if [[ -n "${VQ_CHECKPOINT}" ]]; then
  CONFIG_ARGS+=(--vq-checkpoint "${VQ_CHECKPOINT}")
fi
python "${SCRIPT_DIR}/scripts/configure_body.py" "${CONFIG_ARGS[@]}"

export TALKSHOW_WRITE_DATA_CACHE=0
export PYTHONUNBUFFERED=1
export TALKSHOW_RUN_DATE="$(date +%F)"

LOG_NAME="body-vq"
if [[ "${STAGE_NAME}" == "pixel" ]]; then
  LOG_NAME="body-pixel2"
fi
EXP_NAME="talkshow_body_${STAGE_NAME}_${RUN_ID}"
(
  cd "${UPSTREAM_DIR}"
  python -W ignore scripts/train.py \
    --gpu 0 \
    --save_dir "${RUN_ROOT}/experiments" \
    --exp_name "${EXP_NAME}" \
    --speakers oliver seth conan chemistry \
    --config_file "${CONFIG_PATH}"
)

LAST_EPOCH=99
if [[ "${SMOKE}" -eq 1 ]]; then
  LAST_EPOCH=0
fi
CHECKPOINT="${RUN_ROOT}/experiments/${TALKSHOW_RUN_DATE}-${EXP_NAME}-${LOG_NAME}/ckpt-${LAST_EPOCH}.pth"
python "${SCRIPT_DIR}/scripts/verify_face_checkpoint.py" \
  --checkpoint "${CHECKPOINT}" \
  --expected-epoch "${LAST_EPOCH}" \
  --min-global-steps 1

echo "TalkSHOW body ${STAGE_NAME} training completed: ${CHECKPOINT}"
