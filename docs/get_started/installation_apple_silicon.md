# 🍎 Apple Silicon (MPS / MLX) Installation

This page covers installing `sglang-omni` on Apple Silicon (macOS `arm64`) for
models that support the MLX or Torch-MPS backend. It is the shared base for
every MLX-supported model cookbook; model-specific extras and serve commands
live in each cookbook's **Apple Silicon** section.

> **Other platforms?** NVIDIA CUDA uses the [main Installation guide](./installation.md)
> (Docker recommended). Intel XPU, Intel CPU, and Ascend NPU each have their own
> page linked from there.

## Prerequisites

- **macOS 14 or newer** on `arm64` (Apple Silicon). The pinned
  `torch==2.13.0`, `torchvision==0.28.0`, and `torchcodec==0.15.0` wheels are
  built for `macosx_14_0_arm64`.
- **Homebrew** installed and on `PATH`. The installer never invokes `sudo` or
  Homebrew's bootstrapper; install it yourself from [brew.sh](https://brew.sh)
  if needed.
- **Python 3.12** (the installer creates an isolated `uv` venv with Python 3.12).

## Method 1: Using the `install.sh` Script (recommended)

### Qwen3-ASR

```bash
git clone https://github.com/sgl-project/sglang-omni.git && cd sglang-omni
 ./install.sh
source .venv-apple/bin/activate
```

### Fun-Cosyvoice3

```bash
git clone https://github.com/sgl-project/sglang-omni.git && cd sglang-omni
SGLANG_OMNI_EXTRAS=Fun-CosyVoice3 ./install.sh
source .venv-apple/bin/activate
```

Set `SGLANG_OMNI_EXTRAS` to the comma-separated extras your model needs (see
the model's cookbook). Omit it for models with no extra dependency.

The script is idempotent and creates (or reuses) `.venv-apple`, installs the
Homebrew formulae `ffmpeg@7` and `uv` (and `git` only when a working git is not
already available), installs SGLang `v0.5.19` from source with its `all_mps`
extra, and installs this checkout with `uv pip`. SGLang's optional Rust
extensions are not needed by this Apple Silicon path and are skipped.

### FFmpeg 7 and `DYLD_LIBRARY_PATH`

`ffmpeg@7` is intentional: `torchcodec==0.15.0` ships loaders for FFmpeg 4
through 8 only, and the unversioned `ffmpeg` formula installs FFmpeg 9. At
runtime, expose its libraries before starting the server:

```bash
export DYLD_LIBRARY_PATH="$(brew --prefix ffmpeg@7)/lib${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
```

Because `ffmpeg@7` is keg-only, this export must be present whenever the server
starts. macOS may remove `DYLD_*` variables when a SIP-protected system
executable launches the server; set it on the final `sgl-omni` process (for
example, place `/usr/bin/env DYLD_LIBRARY_PATH=...` after wrappers such as
`/usr/bin/time`). Test a compressed input such as M4A or MP3, since WAV
decoding can succeed without loading FFmpeg.

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

Use `--non-interactive` (or `NONINTERACTIVE=1`) to disable Homebrew auto-update
in CI. Slow or proxied networks can override the installer's uv defaults with
`UV_HTTP_TIMEOUT` and `UV_HTTP_RETRIES`.

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
SGLANG_OMNI_EXTRAS=<model-extra> SGLANG_OMNI_REF=<commit> /tmp/sglang-omni-install.sh
```

Piping a remote script directly to Bash executes code without a review step;
use it only when that trade-off is acceptable:

```bash
curl -fsSL https://raw.githubusercontent.com/sgl-project/sglang-omni/<commit>/install.sh \
  | SGLANG_OMNI_EXTRAS=<model-extra> SGLANG_OMNI_REF=<commit> bash
```

For a fork or an internal mirror, set `SGLANG_OMNI_REPO` and
`SGLANG_OMNI_REF` explicitly.

## Method 2: Manual Configuration

If you prefer not to use `install.sh`, set up the environment by hand.

### 1. Install FFmpeg 7

```bash
brew install ffmpeg@7
export DYLD_LIBRARY_PATH="$(brew --prefix ffmpeg@7)/lib${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
```

Do not replace `ffmpeg@7` with the unversioned `ffmpeg` formula. See
[FFmpeg 7 and `DYLD_LIBRARY_PATH`](#ffmpeg-7-and-dyld_library_path) above for
details.

### 2. Create a virtual environment and install

Create one virtual environment for both repositories, then install the pinned
SGLang tag from source with its `all_mps` dependencies before installing
SGLang-Omni:

```bash
git clone --branch v0.5.19 https://github.com/sgl-project/sglang.git
git clone https://github.com/sgl-project/sglang-omni.git
uv venv -p 3.12 sglang-omni/.venv-apple
source sglang-omni/.venv-apple/bin/activate
cd sglang
cp python/pyproject_other.toml python/pyproject.toml
uv pip install -e "python[all_mps]"
cd ../sglang-omni
uv pip install -e ".[<model-extra>]"
```

Replace `<model-extra>` with the extra named in your model's cookbook, or omit
it if none is required. This installs MLX through SGLang; it does not install
or use the `mlx-audio` package.

### 3. Verify the runtime

Before downloading a model, verify both Metal and FFmpeg loading:

```bash
SGLANG_USE_MLX=1 python - <<'PY'
import mlx.core as mx
from torchcodec.decoders import AudioDecoder
assert mx.metal.is_available()
print("MLX Metal and TorchCodec FFmpeg loading are available")
PY
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
