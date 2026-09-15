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
git clone --branch v0.5.19 --single-branch \
  https://github.com/sgl-project/sglang.git
cd sglang
docker build -f docker/musa.Dockerfile -t sglang:v0.5.19-musa520-s5000 .
```

Then build the SGLang-Omni image on top of it:

```bash
git clone https://github.com/sgl-project/sglang-omni.git
cd sglang-omni
docker build -f docker/musa.Dockerfile \
  --build-arg SGLANG_MUSA_IMAGE=sglang:v0.5.19-musa520-s5000 \
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

python -m pip install --upgrade pip "setuptools<82" wheel
sudo apt-get update
sudo apt-get install -y \
  build-essential ca-certificates git libdav1d-dev libmp3lame-dev \
  libsndfile1 libsox-dev libsox-fmt-all libssl-dev libx264-dev nasm \
  pkg-config pybind11-dev sox yasm

ffmpeg_root="$(mktemp -d)"
git clone --depth 1 --branch mt-7.0.2-public \
  https://github.com/MooreThreads/FFmpeg.git "${ffmpeg_root}/FFmpeg"
cd "${ffmpeg_root}/FFmpeg"
mkdir -p build
cd build
../configure \
  --enable-shared \
  --enable-libmp3lame \
  --enable-libdav1d \
  --enable-openssl \
  --enable-libx264 \
  --enable-gpl \
  --enable-nonfree
make -j"$(nproc)"
sudo make install
sudo ldconfig
cd - >/dev/null

cp pyproject_musa.toml pyproject.toml
TORCH_DEVICE_BACKEND_AUTOLOAD=0 \
I_CONFIRM_THIS_IS_NOT_A_LICENSE_VIOLATION=1 \
ENABLE_MUSA=1 \
python -m pip install --no-build-isolation --no-deps \
  "torchcodec @ git+https://github.com/MooreThreads/torchcodec.git@release/0.5-musa-public" \
  --index-url https://pypi.org/simple \
  --extra-index-url https://dl.mthreads.com/repo/api/pypi/pypi/simple \
  --trusted-host dl.mthreads.com

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
