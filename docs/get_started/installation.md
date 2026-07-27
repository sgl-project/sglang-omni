# 🚀 Installation

We provide two installation paths. Docker is recommended — the image ships with UCX, flash-attn, sglang, and CUDA prebuilt.

## 🐳 Option A: Docker (recommended)

**1. Pull the image**

```bash
docker pull lmsysorg/sglang-omni:dev
```

**2. Run the container**

```bash
docker run -it \
    --shm-size 32g \
    --gpus all \
    --ipc host \
    --network host \
    --privileged \
    lmsysorg/sglang-omni:dev \
    /bin/zsh
```

**3. Install `sglang-omni` inside the container**

```bash
git clone git@github.com:sgl-project/sglang-omni.git
cd sglang-omni

uv venv .venv -p 3.12
source .venv/bin/activate

uv sync
```

## 🛠️ Option B: Manual install

Build **UCX 1.20.x** with CUDA + verbs support first. Follow [upstream](https://github.com/openucx/ucx), or reuse the exact build flags in [`docker/Dockerfile`](../../docker/Dockerfile).

Then install:

```bash
git clone git@github.com:sgl-project/sglang-omni.git
cd sglang-omni
uv venv .venv -p 3.12
source .venv/bin/activate

uv sync
```

## Optional external FlashAttention 4 package

The documented `uv sync` installation uses SGLang's vendored FA4 implementation and excludes the external `flash-attn-4` package. This exclusion is a uv project setting, so use `uv sync` or `uv lock` rather than `uv pip install` when creating the environment.

If you explicitly set `SGLANG_INKLING_FA4_USE_PIP=1` to route SGLang through the external pip FA4 implementation, install its dependency with:

```bash
uv sync --extra fa4
```
