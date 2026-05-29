#!/usr/bin/env bash
# Host-resident flashinfer-jit-cache wheel + per-venv install.
#
# The 1.1GiB wheel lives under /github/home/.cache/flashinfer-jit-cache/ and
# survives PR cleanup. Network download happens only when the pinned version
# changes or the wheel file is missing. Each fresh venv still receives a local
# install from that wheel (seconds, no network).
set -euo pipefail

venv_name="${1:?venv name required}"

# Pin matches flashinfer-python 0.6.1 on frankleeeee/sglang-omni:dev (CUDA 12.8).
# Bump this string when upgrading flashinfer in CI; host cache refreshes once.
readonly FLASHINFER_JIT_CACHE_PIN="0.6.1.dev20260121+cu128"
readonly WHEEL_FILENAME="flashinfer_jit_cache-${FLASHINFER_JIT_CACHE_PIN}-cp39-abi3-manylinux_2_28_x86_64.whl"
readonly WHEEL_URL="https://github.com/flashinfer-ai/flashinfer/releases/download/nightly-v0.6.1-20260121/${WHEEL_FILENAME}"
readonly HOST_CACHE_ROOT="${FLASHINFER_JIT_CACHE_HOST_DIR:-/github/home/.cache/flashinfer-jit-cache}"
readonly HOST_WHEEL="${HOST_CACHE_ROOT}/${WHEEL_FILENAME}"
readonly HOST_VERSION_FILE="${HOST_CACHE_ROOT}/VERSION"

ensure_host_wheel() {
  if [ -f "${HOST_VERSION_FILE}" ] \
    && [ "$(cat "${HOST_VERSION_FILE}")" = "${FLASHINFER_JIT_CACHE_PIN}" ] \
    && [ -f "${HOST_WHEEL}" ]; then
    echo "Host flashinfer-jit-cache cache hit: ${HOST_WHEEL}"
    return 0
  fi

  mkdir -p "${HOST_CACHE_ROOT}"
  echo "Refreshing host flashinfer-jit-cache wheel (pin ${FLASHINFER_JIT_CACHE_PIN})"
  curl -fsSL --retry 3 --retry-delay 5 -o "${HOST_WHEEL}" "${WHEEL_URL}"
  echo "${FLASHINFER_JIT_CACHE_PIN}" > "${HOST_VERSION_FILE}"
  echo "Host flashinfer-jit-cache wheel ready: ${HOST_WHEEL}"
}

source "${venv_name}/bin/activate"

if python -c "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('flashinfer_jit_cache') else 1)"; then
  python - <<'PY'
import importlib.metadata as md

print(f"flashinfer-jit-cache already installed: {md.distribution('flashinfer-jit-cache').version}")
PY
  exit 0
fi

ensure_host_wheel

echo "Installing flashinfer-jit-cache into venv from host wheel"
uv pip install "${HOST_WHEEL}"

python - <<'PY'
import importlib.metadata as md
import importlib.util

print(importlib.util.find_spec("flashinfer_jit_cache").origin)
print(md.distribution("flashinfer-jit-cache").version)
PY
