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

uv pip install -v -e .   # drop `-e` for a non-editable install
```

## 🛠️ Option B: Manual install

Build the prerequisites first:

- **UCX 1.20.x** with CUDA + verbs support — follow [upstream](https://github.com/openucx/ucx), or reuse the exact build flags in [`docker/Dockerfile`](../../docker/Dockerfile).
- **flash-attn** — install a prebuilt wheel matching your `torch==2.9.1` / CUDA version.

Then install:

```bash
git clone git@github.com:sgl-project/sglang-omni.git
cd sglang-omni

uv venv .venv -p 3.12
source .venv/bin/activate

uv pip install -v -e .   # drop `-e` for a non-editable install
```
