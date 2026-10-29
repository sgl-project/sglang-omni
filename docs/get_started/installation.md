# Installation — CUDA

Current stable release: **v0.2.0** on [PyPI](https://pypi.org/project/sglang-omni/).

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

uv pip install --prerelease=allow "sglang-omni==0.2.0"
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

uv pip install --prerelease=allow "sglang-omni==0.2.0"
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

SGLANG_BUILD_RUST_EXTS=none uv pip install --prerelease=allow -v -e .   # drop -e for a non-editable install
```

For installation problems, see [Troubleshooting](./troubleshooting.md). `uv pip check` prints a few expected lines on CUDA installs, listed under [Expected output of uv pip check](./troubleshooting.md#expected-output-of-uv-pip-check).

## Other platforms

For other platforms, use the dedicated pages:
[Apple Silicon](./installation_apple_silicon.md),
[Ascend NPU](./installation_npu.md),
[Intel XPU](./installation_xpu.md),
[Intel CPU](./installation_cpu.md), and
[MUSA](./installation_musa.md).

## Prepare the bundled SGLang runtime

Source installs set SGLANG_BUILD_RUST_EXTS=none to retain the Python cache implementation used in qualification. This branch pins SGLang source revision 9f6678aa and carries the GDN prefill graph and prompt-only prefix-cache policy changes in the Omni wheel. After installing Omni, prepare the runtime before starting Python workers:

```bash
sgl-omni-prepare-runtime
sgl-omni-prepare-runtime --check
```

For an editable SGLang checkout, pass its unmodified package directory with --source /path/to/sglang/python/sglang. The command verifies the expected base source hashes, creates a private copy in the active environment, applies the bundled patches, verifies the resulting source hashes, and activates that copy for new Python processes. The original installation remains unchanged. Restart already-running Python processes after preparation. Model workers reject an absent or stale overlay before allocating model state or capturing graphs.

Docker startup and CI environment preparation run this command automatically. Patch files and their expected hashes are included in the CI environment fingerprint. Rebuild the CUDA image when adopting these dependency pins; an old image's FlashInfer installation does not satisfy this branch. The source-pinned CUDA path is the qualification target; provider-specific CPU/XPU/ROCm/NPU images require their own dependency alignment and validation.
