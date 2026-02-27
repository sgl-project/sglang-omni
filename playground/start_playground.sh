#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------------------
# Start the playground: single-process backend serving API + UI + file browser.
#
# Usage:
#   ./playground/start_playground.sh --pipeline qwen3-omni --model-id Qwen/Qwen3-Omni-30B-A3B-Instruct
#   CUDA_VISIBLE_DEVICES=5 ./playground/start_playground.sh --pipeline qwen3-omni --model-id <id>
#   ./playground/start_playground.sh --pipeline qwen3-omni --model-id <id> --port 8080
# ---------------------------------------------------------------------------

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PLAYGROUND_DIR="${ROOT_DIR}/playground"

PORT="${PORT:-8000}"
PYTHON_BIN="${PYTHON_BIN:-python}"

# Parse arguments: extract our flags, forward the rest to sglang-omni-server
BACKEND_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --port)  PORT="$2"; shift 2 ;;
    *)       BACKEND_ARGS+=("$1"); shift ;;
  esac
done

if [[ ${#BACKEND_ARGS[@]} -eq 0 ]]; then
  echo "Usage: $0 --pipeline <name> --model-id <model> [--port PORT]"
  echo ""
  echo "Example:"
  echo "  CUDA_VISIBLE_DEVICES=5 $0 --pipeline qwen3-omni --model-id Qwen/Qwen3-Omni-30B-A3B-Instruct"
  exit 1
fi

echo "============================================================"
echo "  SGLang-Omni Playground"
echo "============================================================"
echo ""
echo "  URL: http://localhost:${PORT}"
echo ""
echo "============================================================"
echo ""

exec "${PYTHON_BIN}" -m sglang_omni.serve.launcher \
  "${BACKEND_ARGS[@]}" \
  --port "${PORT}" \
  --serve-playground "${PLAYGROUND_DIR}"
