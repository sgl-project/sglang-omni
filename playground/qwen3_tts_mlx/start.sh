#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------------------
# Start the MLX Qwen3-TTS voice-cloning playground.
#
# There is no separate backend: MLX serving is not wired into Omni's scheduler
# yet, so the UI loads the checkpoint in its own process.
#
# Usage:
#   ./playground/qwen3_tts_mlx/start.sh
#   ./playground/qwen3_tts_mlx/start.sh --model-path ~/models/qwen3-tts-base
#   ./playground/qwen3_tts_mlx/start.sh --port 7861 --share
# ---------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

MODEL_PATH="${QWEN3_TTS_CKPT:-${MODEL_PATH:-Qwen/Qwen3-TTS-12Hz-0.6B-Base}}"
GRADIO_PORT="7860"
GRADIO_HOST="127.0.0.1"
GRADIO_SHARE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-path) MODEL_PATH="$2"; shift 2 ;;
    --port)       GRADIO_PORT="$2"; shift 2 ;;
    --host)       GRADIO_HOST="$2"; shift 2 ;;
    --share)      GRADIO_SHARE="--share"; shift ;;
    -h|--help)    sed -n '4,14p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

if ! "${PYTHON_BIN}" -c "import mlx.core" >/dev/null 2>&1; then
  echo "MLX is not importable. On Apple Silicon: pip install mlx mlx-lm" >&2
  exit 1
fi

cd "${REPO_DIR}"
echo "Model:  ${MODEL_PATH}"
echo "UI:     http://${GRADIO_HOST}:${GRADIO_PORT}"
exec "${PYTHON_BIN}" -m playground.qwen3_tts_mlx.app \
  --model-path "${MODEL_PATH}" \
  --host "${GRADIO_HOST}" \
  --port "${GRADIO_PORT}" \
  ${GRADIO_SHARE}
