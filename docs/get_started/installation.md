# 🚀 Installation

Current stable release: **v0.1.1** on [PyPI](https://pypi.org/project/sglang-omni/).

Two install paths. Docker is recommended — UCX, flash-attn, sglang, and CUDA are prebuilt.

## 🐳 Option A: Docker (recommended)

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
uv venv .venv -p 3.12
source .venv/bin/activate

uv pip install "sglang-omni==0.1.1"
```

## 🛠️ Option B: Manual install

Build prerequisites first:

- **UCX 1.20.x** with CUDA + verbs — [upstream](https://github.com/openucx/ucx), or reuse flags in [`docker/Dockerfile`](../../docker/Dockerfile).
- **flash-attn-4** `>=4.0.0b18`, matching `torch==2.11.0` and SGLang 0.5.16's `nvidia-cutlass-dsl` 4.6.0 pin.

Then:

```bash
uv venv .venv -p 3.12
source .venv/bin/activate

uv pip install "sglang-omni==0.1.1"
```

Latest on the index without a pin: `uv pip install sglang-omni`.

### Install from source

For development or unreleased changes:

```bash
git clone git@github.com:sgl-project/sglang-omni.git
cd sglang-omni

uv venv .venv -p 3.12
source .venv/bin/activate

uv pip install -v -e .   # drop -e for a non-editable install
```
