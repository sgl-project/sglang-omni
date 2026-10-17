# Installation — Apple Silicon

<a id="macos-apple-silicon"></a>

Install the Python Torch/MPS path of `sglang-omni` for Qwen3-ASR and Fun-ASR
on macOS Apple Silicon (arm64) with `./install.sh`. CUDA users should follow
the [CUDA installation](./installation.md).

The native C++ MLX runtime under `sglang_omni_mlx/native/` has a separate build
and run workflow. See the [MLX path documentation](../developer_reference/mlx.md)
and the [Voxt backend guide](https://github.com/sgl-project/sglang-omni/blob/main/Voxt/backend/README.md) for that path.

```bash
git clone https://github.com/sgl-project/sglang-omni.git && cd sglang-omni
./install.sh
source .venv-apple/bin/activate
```

The script is idempotent and creates (or reuses) `.venv-apple`, installs the
Homebrew formulae `ffmpeg@7` and `uv` (and `git` only when a working git is not
already available), installs SGLang `v0.5.21` from source with its `all_mps`
extra, and installs this checkout with `uv pip`. SGLang's optional Rust
extensions are not needed by this Apple Silicon path and are skipped.
`ffmpeg@7` is intentional: `torchcodec==0.15.0` ships loaders for FFmpeg 4 through 8
only, and the unversioned formula installs FFmpeg 9. At runtime, expose its libraries:

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
`~/.cache/sglang-omni/sglang-v0.5.21` and can be changed with
`SGLANG_SOURCE_DIR`. Slow or proxied networks can override the installer's uv
defaults with `UV_HTTP_TIMEOUT` and `UV_HTTP_RETRIES`.

This path currently supports macOS 14 or newer on `arm64` only (the pinned
`torch==2.13.0`, `torchvision==0.28.0` and `torchcodec==0.15.0` wheels are built
for `macosx_14_0_arm64`) and is intended for the Apple-Silicon Qwen3-ASR
Torch-MPS path and the
[Fun-ASR Torch/MPS path](../cookbook/fun_asr.md#apple-silicon).
Common failures are a missing Homebrew/uv on `PATH`, an unavailable Python 3.12
toolchain, or forgetting the `DYLD_LIBRARY_PATH` export when starting an audio
server.

## Run from a hosted installer

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

## Next steps

- [Run Qwen3-ASR with Torch/MPS](../cookbook/qwen3_asr.md#apple-silicon).
- [Run Fun-ASR with Torch/MPS](../cookbook/fun_asr.md#apple-silicon).
