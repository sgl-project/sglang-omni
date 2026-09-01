# musa-sglang-omni Installation Guide

This note points to the verified `musa-sglang-omni` cloud install workflow.
For the MUSA SDK itself, follow the official Moore Threads installation guide:

```text
https://docs.mthreads.com/musa-sdk/musa-sdk-doc-online/install_guide/
```

The local workflow below assumes the cloud image already provides the MUSA
driver/runtime and a working `torch_musa` stack. Do **not** replace that stack
with unconstrained public wheels.

## Assumptions

- Python 3.10 to 3.12
- `pip`, `git`, `setuptools>=77.0.0`, and a normal build toolchain
- the cloud image already exposes MUSA devices and a working `torch.musa`
- you will install a MUSA-enabled SGLang build from source

## Public packages to install

`sglang-omni` expects `sglang` to be present. For MUSA, use the public SGLang source build
path instead of the CUDA wheel pins:

```bash
git clone https://github.com/sgl-project/sglang.git
cd sglang
git checkout v0.5.16
cd python/sglang/kernels/aot
python setup_musa.py install
cd ../../../..
rm -f python/pyproject.toml && mv python/pyproject_other.toml python/pyproject.toml
pip install -e "python[dev_musa]"
```

That path pulls the public MUSA packages declared by SGLang itself, including `torch_musa`,
`torchada`, `flash_attn_3`, `deep-gemm`, and `mate`.

For the model-by-model view, see the
[`MUSA Hardware / Backend / Model Support Matrix`](musa_support_matrix.md).
That page summarizes the model coverage and bench evidence in a self-contained
table.

After that, install the shared runtime and model-support packages required by
your smoke path. Keep any wheelhouse-specific pins aligned with the MUSA stack
already present in the cloud image.

Keep model/test extras out of the base smoke install. Add them only for a
specific model test after confirming they do not pull generic CUDA `torch`,
`torchaudio`, or `torchcodec` wheels.

## Install sglang-omni

Back in this repository:

```bash
cd /path/to/sglang-omni-pr1651
python -m pip install -e . --no-deps
```

`--no-deps` keeps pip from trying to replace the MUSA stack with the CUDA-specific pins from
`pyproject.toml`.

## Smoke test

```bash
python - <<'PY'
import torch
import torchada
import sglang
import sglang_omni
from sglang_omni.platforms import current_platform

print("torch:", torch.__version__)
print("musa available:", getattr(torch, "musa", None) and torch.musa.is_available())
print("platform:", current_platform.device_type)
print("sglang_omni:", sglang_omni.__file__)
PY
```

If `torch.musa.is_available()` is false, fix the base cloud image first; do not try to patch
that with extra Python packages.
