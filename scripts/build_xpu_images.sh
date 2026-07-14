#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_JOBS="${BUILD_JOBS:-3}"
SGLANG_XPU_IMAGE="${SGLANG_XPU_IMAGE:-sglang-xpu:0.5.12.post1-pt2.11}"
SGLANG_OMNI_XPU_IMAGE="${SGLANG_OMNI_XPU_IMAGE:-sglang-omni-xpu:0.1.0-moss}"
SGLANG_OMNI_COMMIT="${SGLANG_OMNI_COMMIT:-$(git -C "$ROOT" rev-parse HEAD)}"

docker build \
  --build-arg "BUILD_JOBS=${BUILD_JOBS}" \
  -t "$SGLANG_XPU_IMAGE" \
  -f "$ROOT/docker/sglang-xpu-base.Dockerfile" \
  "$ROOT"

docker build \
  --build-arg "SGLANG_XPU_IMAGE=${SGLANG_XPU_IMAGE}" \
  --build-arg "SGLANG_OMNI_COMMIT=${SGLANG_OMNI_COMMIT}" \
  -t "$SGLANG_OMNI_XPU_IMAGE" \
  -f "$ROOT/docker/xpu.Dockerfile" \
  "$ROOT"
