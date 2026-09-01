#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${MEETING_TRANSLATION_PYTHON:-${SCRIPT_DIR}/.venv/bin/python}"

if [[ ! -x "${PYTHON}" ]]; then
  echo "Caption environment not found: ${PYTHON}" >&2
  echo "Set MEETING_TRANSLATION_PYTHON or recreate .venv with uv." >&2
  exit 1
fi

export PYTHONNOUSERSITE=1
export HF_HOME="${HF_HOME:-${SCRIPT_DIR}/models/huggingface}"
export TORCH_HOME="${TORCH_HOME:-${SCRIPT_DIR}/models/torch}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-${SCRIPT_DIR}/cache}"

cd "${SCRIPT_DIR}"
MODULE="app"
if [[ "${1:-}" == "--metrics" ]]; then
  MODULE="replay"
  shift
fi
exec "${PYTHON}" -m "meeting_translation.${MODULE}" "$@"
