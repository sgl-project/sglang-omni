#!/usr/bin/env bash
# Ensure flashinfer-jit-cache is present so MoE CUDA graph capture does not
# fall back to runtime nvcc/ninja JIT (which routinely exceeds CI startup
# timeouts). Stale ~/.cache/flashinfer dirs are removed separately in omni-setup.
set -euo pipefail

venv_name="${1:?venv name required}"

source "${venv_name}/bin/activate"

if python -c "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('flashinfer_jit_cache') else 1)"; then
  python - <<'PY'
import importlib.metadata as md

print(f"flashinfer-jit-cache already installed: {md.distribution('flashinfer-jit-cache').version}")
PY
  exit 0
fi

# Pin matches flashinfer-python 0.6.1 on frankleeeee/sglang-omni:dev (CUDA 12.8).
wheel_filename="flashinfer_jit_cache-0.6.1.dev20260121+cu128-cp39-abi3-manylinux_2_28_x86_64.whl"
wheel_url="https://github.com/flashinfer-ai/flashinfer/releases/download/nightly-v0.6.1-20260121/${wheel_filename}"
wheel_cache_dir="${FLASHINFER_JIT_CACHE_WHEEL_DIR:-/github/home/.cache/wheels}"
cached_wheel="${wheel_cache_dir}/${wheel_filename}"

mkdir -p "${wheel_cache_dir}"

if [ ! -f "${cached_wheel}" ]; then
  echo "Downloading flashinfer-jit-cache wheel to ${cached_wheel}"
  curl -fsSL --retry 3 --retry-delay 5 -o "${cached_wheel}" "${wheel_url}"
else
  echo "Using cached flashinfer-jit-cache wheel: ${cached_wheel}"
fi

echo "Installing flashinfer-jit-cache from ${cached_wheel}"
uv pip install "${cached_wheel}"

python - <<'PY'
import importlib.metadata as md
import importlib.util

print(importlib.util.find_spec("flashinfer_jit_cache").origin)
print(md.distribution("flashinfer-jit-cache").version)
PY
