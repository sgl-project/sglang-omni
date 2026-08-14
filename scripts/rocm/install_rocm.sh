#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
manifest="${repo_dir}/pyproject_rocm.toml"

check_manifest() {
    if grep -Eiq 'flashinfer|flash-attn|nixl-cu|mooncake-transfer-engine-cuda|nvidia-' "${manifest}"; then
        echo "ROCm manifest contains a CUDA-only dependency" >&2
        return 1
    fi
}

check_runtime() {
    python3 - <<'PY'
import importlib.metadata
import sys

import torch

hip = getattr(torch.version, "hip", None)
if not hip:
    raise SystemExit("PyTorch is not a ROCm/HIP build")

try:
    sglang_version = importlib.metadata.version("sglang")
except importlib.metadata.PackageNotFoundError as exc:
    raise SystemExit("SGLang is missing from the ROCm base image") from exc
from nixl import _api as nixl_api  # noqa: F401

architectures = sorted(
    {
        str(getattr(torch.cuda.get_device_properties(index), "gcnArchName", ""))
        .split(":", 1)[0]
        .lower()
        for index in range(torch.cuda.device_count())
        if getattr(torch.cuda.get_device_properties(index), "gcnArchName", "")
    }
)
print(
    f"ROCm stack OK: torch={torch.__version__} hip={hip} "
    f"sglang={sglang_version} devices={torch.cuda.device_count()} "
    f"architectures={architectures}"
)
PY
}

check_manifest
if [[ "${1:-}" == "--check" ]]; then
    check_runtime
    exit 0
fi

python3 -m pip install --no-deps --no-build-isolation -e "${repo_dir}"
python3 "${repo_dir}/scripts/rocm/install_dependencies.py" --manifest "${manifest}"
python3 -m pip install --no-deps qwen-tts==0.1.1
check_runtime
