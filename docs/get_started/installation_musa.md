# 🚀 Installation — MUSA

This document describes how to install SGLang-Omni on Moore Threads GPUs.

SGLang owns the base MUSA runtime environment. Install or build SGLang's MUSA
environment first, then install SGLang-Omni as an overlay.

## Prerequisites

Install the Moore Threads driver and MUSA runtime on the host first. For the
base SGLang environment, follow the SGLang Moore Threads GPU installation guide.

## 🐳 Option A: Docker

Build the SGLang MUSA image first:

```bash
git clone https://github.com/sgl-project/sglang.git
cd sglang
docker build -f docker/musa.Dockerfile -t sglang:main-musa520-s5000 .
```

Then build the SGLang-Omni image on top of it:

```bash
git clone https://github.com/sgl-project/sglang-omni.git
cd sglang-omni
docker build -f docker/musa.Dockerfile \
  --build-arg SGLANG_MUSA_IMAGE=sglang:main-musa520-s5000 \
  -t sglang-omni:main-musa520-s5000 .
```

Run the image with MUSA devices exposed by the host runtime. If `mthreads` is
already configured as the Docker default runtime on your host, omit
`--runtime=mthreads`.

```bash
docker run -it --rm \
  --runtime=mthreads \
  --env MTHREADS_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
  --env MTHREADS_DRIVER_CAPABILITIES=all \
  --env HF_HOME=/cache/huggingface \
  -v ~/.cache/huggingface:/cache/huggingface \
  --shm-size=32g \
  -p 8000:8000 \
  sglang-omni:main-musa520-s5000
```

## 🛠️ Option B: Install from Source

Start from an environment where SGLang has already been installed with MUSA
support.

```bash
git clone https://github.com/sgl-project/sglang-omni.git
cd sglang-omni

python -m pip install --upgrade pip
cp pyproject_musa.toml pyproject.toml
python -m pip install -e . \
  --index-url https://dl.mthreads.com/repo/api/pypi/pypi/simple \
  --extra-index-url https://pypi.org/simple \
  --trusted-host dl.mthreads.com
```

The MUSA pyproject installs only SGLang-Omni overlay dependencies. The base
MUSA torch stack, SGLang, sgl-kernel, Triton, TileLang, and MATE are inherited
from the SGLang MUSA environment.

## Verify

```bash
python - <<'PY'
import torch
import torchada  # noqa: F401
import sglang
import sglang_omni

print("torch:", torch.__version__)
print("musa:", torch.version.musa)
print("devices:", torch.musa.device_count())
print("sglang:", sglang.__version__)
print("sglang_omni:", sglang_omni.__file__)
PY
```
