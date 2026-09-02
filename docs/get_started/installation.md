# 🚀 Installation

Current stable release: **v0.1.4** on [PyPI](https://pypi.org/project/sglang-omni/).

Choose the path for your platform. Docker is recommended for NVIDIA CUDA —
UCX, flash-attn, SGLang, and CUDA are prebuilt. Apple Silicon has a dedicated
source installer below.

> **Intel GPU (XPU)?** For Intel Arc GPUs, see [Installation — Intel XPU](./installation_xpu.md), which uses [`pyproject_xpu.toml`](../../pyproject_xpu.toml) + the PyTorch XPU wheel index instead of the CUDA-only pins below.

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

uv pip install --prerelease=allow "sglang-omni==0.1.4"
```

<a id="macos-apple-silicon"></a>

## 🍎 Option B: macOS Apple Silicon installer

From a checkout of this branch, run:

```bash
./install.sh
source .venv-apple/bin/activate
```

The script is idempotent and creates (or reuses) `.venv-apple`, installs the
Homebrew formulae `ffmpeg@7` and `uv` (and `git` only when a working git is not
already available), installs SGLang `v0.5.18` from source with its `all_mps`
extra, and installs this checkout with `uv pip`. SGLang's optional Rust
extensions are not needed by this Apple Silicon path and are skipped.
`ffmpeg@7` is intentional: `torchcodec==0.11.1` does not support the current
unversioned FFmpeg 9 formula. At runtime, expose its libraries:

```bash
export DYLD_LIBRARY_PATH="$(brew --prefix ffmpeg@7)/lib${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
```

Homebrew must be installed before running the script. If `brew` is missing, the
script prints an error and exits; install it yourself from
[brew.sh](https://brew.sh), then rerun. The installer never invokes `sudo` or
Homebrew's bootstrapper. Use `--non-interactive` (or `NONINTERACTIVE=1`) to
disable Homebrew auto-update in CI, `SGLANG_OMNI_VENV=/path/to/venv` to choose a virtualenv, and
`SGLANG_OMNI_EXTRAS=audar-tts,fun-cosyvoice3` to enable optional extras.
The persistent SGLang source checkout defaults to
`~/.cache/sglang-omni/sglang-v0.5.18` and can be changed with
`SGLANG_SOURCE_DIR`. Slow or proxied networks can override the installer's uv
defaults with `UV_HTTP_TIMEOUT` and `UV_HTTP_RETRIES`.

This path currently supports macOS `arm64` only and is intended for the
Apple-Silicon Qwen3-ASR MLX/Torch-MPS paths. Other platforms should use the
Docker, manual, or Intel XPU instructions below. Common failures are a missing
Homebrew/uv on `PATH`, an unavailable Python 3.12 toolchain, or forgetting the
`DYLD_LIBRARY_PATH` export when starting an audio server.

### Run from a hosted installer

The script also supports a downloaded or `curl | bash` invocation: when it is
not inside an sglang-omni checkout, it clones the repository specified by
`SGLANG_OMNI_REPO` and `SGLANG_OMNI_REF` into the cache and installs that
checkout. Prefer downloading, reviewing, and then running a pinned script:

```bash
curl -fsSLo /tmp/sglang-omni-install.sh \
  https://raw.githubusercontent.com/sgl-project/sglang-omni/<commit>/install.sh
less /tmp/sglang-omni-install.sh
chmod +x /tmp/sglang-omni-install.sh
SGLANG_OMNI_REF=<commit> /tmp/sglang-omni-install.sh
```

Piping a remote script directly to Bash executes code without a review step;
use it only when that trade-off is acceptable:

```bash
curl -fsSL https://raw.githubusercontent.com/sgl-project/sglang-omni/<commit>/install.sh \
  | SGLANG_OMNI_REF=<commit> bash
```

For a fork or an internal mirror, set `SGLANG_OMNI_REPO` and
`SGLANG_OMNI_REF` explicitly. The hosted mode stores the project checkout at
`~/.cache/sglang-omni/sglang-omni-<ref>` by default; override it with
`SGLANG_OMNI_PROJECT_DIR`.

## 🛠️ Option C: Manual install

Build prerequisites first:

- **UCX 1.20.x** with CUDA + verbs — [upstream](https://github.com/openucx/ucx), or reuse flags in [`docker/Dockerfile`](../../docker/Dockerfile).
- **flash-attn-4** `>=4.0.0b18`, matching `torch==2.13.0` and SGLang 0.5.18's `nvidia-cutlass-dsl` 4.6.2 pin.

Then:

```bash
pip install --upgrade pip
pip install uv

uv venv .venv -p 3.12
source .venv/bin/activate

uv pip install --prerelease=allow "sglang-omni==0.1.4"
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
