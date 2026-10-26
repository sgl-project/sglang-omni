# Installation — CUDA

Current stable release: **v0.1.9** on [PyPI](https://pypi.org/project/sglang-omni/).

This page covers NVIDIA CUDA. Docker is recommended — UCX, flash-attn, SGLang,
and CUDA are prebuilt.

## Option A: Docker (recommended)

**1. Pull the image**

```bash
docker pull hongccc/sglang-omni:dev
```

Only the `dev` tag is published today. It moves with main — pin by digest for reproducible runs:

```bash
docker pull lmsysorg/sglang-omni@sha256:<digest>
```

**2. Run the container**

```bash
docker run -it \
    --shm-size 32g \
    --gpus all \
    --ipc host \
    --network host \
    --privileged \
    hongccc/sglang-omni:dev \
    /bin/zsh
```

**3. Install `sglang-omni` inside the container**

```bash
pip install --upgrade pip
pip install uv

uv venv .venv -p 3.12
source .venv/bin/activate

uv pip install --prerelease=allow "sglang-omni==0.1.9"
```

## Option B: Manual install

Build prerequisites first:

- **UCX 1.20.x** with CUDA + verbs — [upstream](https://github.com/openucx/ucx), or reuse flags in [`docker/Dockerfile`](../../docker/Dockerfile).
- **flash-attn-4** `>=4.0.0b18`, matching `torch==2.14.1` and SGLang 0.5.22's `nvidia-cutlass-dsl` 4.8.0 pin.

Then:

```bash
pip install --upgrade pip
pip install uv

uv venv .venv -p 3.12
source .venv/bin/activate

uv pip install --prerelease=allow "sglang-omni==0.1.9"
```

Latest on the index without a pin: `uv pip install --prerelease=allow sglang-omni`.

### Install from source

For development or unreleased changes:

```bash
git clone git@github.com:sgl-project/sglang-omni.git
cd sglang-omni

pip install --upgrade pip
pip install uv

uv venv .venv -p 3.12
source .venv/bin/activate

uv pip install --prerelease=allow -v -e .   # drop -e for a non-editable install
```

For installation problems, see [Troubleshooting](./troubleshooting.md). `uv pip check` prints a few expected lines on CUDA installs, listed under [Expected output of uv pip check](./troubleshooting.md#expected-output-of-uv-pip-check).

## Other platforms

For other platforms, use the dedicated pages:
[Apple Silicon](./installation_apple_silicon.md),
[Ascend NPU](./installation_npu.md),
[Intel XPU](./installation_xpu.md),
[Intel CPU](./installation_cpu.md), and
[MUSA](./installation_musa.md).
