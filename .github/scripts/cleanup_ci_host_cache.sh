#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -gt 0 ]; then
  VENV_NAMES=("$@")
else
  VENV_NAMES=(omni omni-qwen3 omni-s2pro)
fi

for name in "${VENV_NAMES[@]}"; do
  path="/github/home/${name}"
  if [ -e "${path}" ]; then
    echo "Removing ${path}..."
    rm -rf "${path}"
  fi
done

EPHEMERAL_PATHS=(
  /github/home/.torchinductor
  /github/home/.cache/flashinfer
  /github/home/.cache/uv
  /github/home/.cache/torch
  /github/home/.cache/triton
)

for path in "${EPHEMERAL_PATHS[@]}"; do
  if [ -e "${path}" ]; then
    echo "Removing ${path}..."
    rm -rf "${path}"
  fi
done

if command -v nvidia-smi >/dev/null 2>&1; then
  if [ -f ".github/scripts/delete_gpu_process.sh" ]; then
    bash .github/scripts/delete_gpu_process.sh || true
  fi
fi

echo "CI host cache cleanup complete"
