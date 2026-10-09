#!/usr/bin/env bash
# Serve the model under test on /v1/realtime. Runs in the foreground; stop with Ctrl-C.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

cd "$SGLANG_OMNI_ROOT"
CUDA_VISIBLE_DEVICES="$SERVER_GPU" exec "$OMNI_VENV/bin/python" -m sglang_omni.cli serve \
    --config "$SERVER_CONFIG" \
    --model-path "$MODEL_PATH" \
    --enable-realtime \
    --port "$SERVER_PORT"
