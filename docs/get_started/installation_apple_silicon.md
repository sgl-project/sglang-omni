# 🍎 Apple Silicon Installation

This page covers installing `sglang-omni` on Apple Silicon (macOS `arm64`) for
models that support the MLX or Torch-MPS backend. It is the shared base for
every MLX-supported model cookbook; model-specific extras and serve commands
live in each cookbook's **Apple Silicon** section.

> **Other platforms?** NVIDIA CUDA uses the [main Installation guide](./installation.md)
> (Docker recommended). Intel XPU, Intel CPU, and Ascend NPU each have their own
> page linked from there.

## Prerequisites

### MLX(0.32.2) fully supports
- **macOS 14 or newer** on `arm64` (Apple Silicon). The pinned
  `torch==2.13.0`, `torchvision==0.28.0`, and `torchcodec==0.15.0` wheels are
  built for `macosx_14_0_arm64`.
- **Homebrew** installed and on `PATH`. The installer never invokes `sudo` or
  Homebrew's bootstrapper; install it yourself from [brew.sh](https://brew.sh)
  if needed.
- **Python 3.12** (the installer creates an isolated `uv` venv with Python 3.12).

## Method 1: Using the `install.sh` Script (recommended)

```bash
git clone https://github.com/sgl-project/sglang-omni.git && cd sglang-omni
 ./install.sh
source .venv-apple/bin/activate
```

The script is idempotent and creates (or reuses) `.venv-apple`, installs the
Homebrew formulae `ffmpeg@7` and `uv` (and `git` only when a working git is not
already available), installs SGLang `v0.5.19` from source with its `all_mps`
extra, and installs this checkout with `uv pip`. SGLang's optional Rust
extensions are not needed by this Apple Silicon path and are skipped.

### Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `SGLANG_OMNI_VENV` | `.venv-apple` (in checkout) | Virtual environment path |
| `SGLANG_OMNI_EXTRAS` | _(empty)_ | Optional extras, comma-separated |
| `SGLANG_OMNI_CACHE` | `~/.cache/sglang-omni` | Cache root for source checkouts |
| `SGLANG_SOURCE_DIR` | `<cache>/sglang-v0.5.19` | SGLang source checkout path |
| `SGLANG_VERSION` | `v0.5.19` | SGLang git tag/branch |
| `SGLANG_REPO` | upstream sglang | SGLang repository URL |
| `SGLANG_OMNI_REPO` | upstream sglang-omni | Repository URL for hosted use |
| `SGLANG_OMNI_REF` | `main` | Branch/tag for hosted use |
| `SGLANG_OMNI_PROJECT_DIR` | `<cache>/sglang-omni-<ref>` | Checkout path for hosted use |
| `NONINTERACTIVE=1` | _(off)_ | Disable Homebrew auto-update (CI) |
| `UV_HTTP_TIMEOUT` | `300` | Per-request uv timeout (seconds) |
| `UV_HTTP_RETRIES` | `5` | uv network retry count |


The installer never invokes `sudo` or Homebrew's bootstrapper. Use `--non-interactive` (or `NONINTERACTIVE=1`) to
disable Homebrew auto-update in CI, `SGLANG_OMNI_VENV=/path/to/venv` to choose a virtualenv, and
`SGLANG_OMNI_EXTRAS=audar-tts,fun-cosyvoice3` to enable optional extras.
The persistent SGLang source checkout defaults to
`~/.cache/sglang-omni/sglang-v0.5.19` and can be changed with
`SGLANG_SOURCE_DIR`. Slow or proxied networks can override the installer's uv
defaults with `UV_HTTP_TIMEOUT` and `UV_HTTP_RETRIES`.

## Method 2: Run from a hosted installer

This method runs the installer directly without cloning the repository.

```bash
curl -fsSL https://raw.githubusercontent.com/sgl-project/sglang-omni/main/install.sh | bash
```

## Common failures

- Missing Homebrew or `uv` on `PATH`.
- An unavailable Python 3.12 toolchain.
- Forgetting the `DYLD_LIBRARY_PATH` export when starting an audio server
  (compressed audio fails while WAV still works).
- Running on macOS older than 14 or on non-`arm64` hardware.

## Next steps

Once the base environment is installed, follow the **Apple Silicon (MLX)**
section in your model's cookbook for model-specific extras, the converted MLX
checkpoint, and the exact `sgl-omni serve` command:

- [Qwen3-ASR](../cookbook/qwen3_asr.md)
- [Fun-CosyVoice3](../cookbook/fun_cosyvoice3.md)
