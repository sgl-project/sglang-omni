# 🚀 Installation

Current stable release: **v0.1.5** on [PyPI](https://pypi.org/project/sglang-omni/).

Choose the path for your platform. Docker is recommended for NVIDIA CUDA —
UCX, flash-attn, SGLang, and CUDA are prebuilt. Apple Silicon has a dedicated
source installer below.

> **Intel GPU (XPU)?** For Intel Arc GPUs, see [Installation — Intel XPU](./installation_xpu.md), which uses [`pyproject_xpu.toml`](../../pyproject_xpu.toml) + the PyTorch XPU wheel index instead of the CUDA-only pins below.

> **Intel CPU?** Also not this page. See [Installation — Intel CPU](./installation_cpu.md), which uses [`pyproject_cpu.toml`](../../pyproject_cpu.toml) + the PyTorch CPU wheel index.

> **Ascend NPU?** See [Installation — Ascend NPU](./installation_npu.md) for the supported software stack, prerequisites, and installation helper.

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
pip install --upgrade pip
pip install uv

uv venv .venv -p 3.12
source .venv/bin/activate

uv pip install --prerelease=allow "sglang-omni==0.1.5"
```

<a id="macos-apple-silicon"></a>

## 🛠️ Option B: Manual install

Build prerequisites first:

- **UCX 1.20.x** with CUDA + verbs — [upstream](https://github.com/openucx/ucx), or reuse flags in [`docker/Dockerfile`](../../docker/Dockerfile).
- **flash-attn-4** `>=4.0.0b18`, matching `torch==2.13.0` and SGLang 0.5.19's `nvidia-cutlass-dsl` 4.6.2 pin.

Then:

```bash
pip install --upgrade pip
pip install uv

uv venv .venv -p 3.12
source .venv/bin/activate

uv pip install --prerelease=allow "sglang-omni==0.1.5"
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
