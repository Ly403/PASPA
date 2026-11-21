#!/usr/bin/env bash
set -euo pipefail

UPSTREAM_COMMIT="9aef82df5ff1082f0cfa0cfc116c0b7208e85d5b"
UPSTREAM_URL="${TALKSHOW_UPSTREAM_URL:-https://github.com/yhw-yhw/TalkSHOW.git}"

if [[ $# -ne 1 ]]; then
  echo "usage: $0 /path/to/talkshow_upstream" >&2
  exit 2
fi

DEST="$1"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MARKER="${DEST}/.paspa_talkshow_patch.json"

if [[ -f "${MARKER}" ]]; then
  ACTUAL="$(git -C "${DEST}" rev-parse HEAD)"
  if [[ "${ACTUAL}" != "${UPSTREAM_COMMIT}" ]]; then
    echo "patched checkout has unexpected commit: ${ACTUAL}" >&2
    exit 1
  fi
  python "${SCRIPT_DIR}/scripts/patch_upstream.py" \
    --upstream-dir "${DEST}" --check
  echo "TalkSHOW checkout is already pinned and patched: ${DEST}"
  exit 0
fi

if [[ -e "${DEST}" ]]; then
  echo "destination exists but is not a recognized patched checkout: ${DEST}" >&2
  exit 1
fi

git clone --filter=blob:none "${UPSTREAM_URL}" "${DEST}"
git -C "${DEST}" checkout --detach "${UPSTREAM_COMMIT}"
python "${SCRIPT_DIR}/scripts/patch_upstream.py" --upstream-dir "${DEST}"

echo "Prepared TalkSHOW ${UPSTREAM_COMMIT} at ${DEST}"
