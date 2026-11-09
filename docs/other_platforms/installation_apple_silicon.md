# Installation — Apple Silicon

<a id="macos-apple-silicon"></a>

Choose the runtime you want to use on Apple Silicon:

- **Torch MPS:** Follow the [installation steps on this page](#prerequisites)
  to run Qwen3-ASR and Fun-ASR with PyTorch on macOS 14 or newer.
- **Native MLX / Voxt:** The C++ runtime in `sglang_omni_mlx/native/` has its
  own build and run workflow. See [Native MLX development](../developer_reference/mlx/index.md)
  and [Voxt's README](https://github.com/sgl-project/sglang-omni/blob/main/Voxt/README.md).

For NVIDIA GPUs, see the [CUDA installation](../get_started/installation.md).

## Prerequisites

Homebrew must be installed before running the script. If `brew` is missing, the
script prints an error and exits; install it yourself from
[brew.sh](https://brew.sh), then rerun. The installer never invokes `sudo` or
Homebrew's bootstrapper.

The installer uses Python 3.12 and requires the macOS 14 or newer arm64 wheels
for `torch==2.14.1`, `torchvision==0.29.1`, and `torchcodec==0.17.0`.

(run-from-a-hosted-installer)=
## Install with the hosted installer

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

## Install from a checkout

If you want to work from a local checkout, run the same installer from the
repository root:

```bash
git clone https://github.com/sgl-project/sglang-omni.git && cd sglang-omni
./install.sh
source .venv-apple/bin/activate
```

The script is idempotent and creates (or reuses) `.venv-apple`, installs the
Homebrew formulae `ffmpeg@7` and `uv` (and `git` only when a working git is not
already available), installs SGLang `v0.5.23` from source with its `all_mps`
extra, and installs this checkout with `uv pip`. SGLang's optional Rust
extensions are not needed by this Apple Silicon path and are skipped.

Use `--non-interactive` (or `NONINTERACTIVE=1`) to
disable Homebrew auto-update in CI, `SGLANG_OMNI_VENV=/path/to/venv` to choose a virtualenv, and
`SGLANG_OMNI_EXTRAS=audar-tts,fun-cosyvoice3` to enable optional extras.
The persistent SGLang source checkout defaults to
`~/.cache/sglang-omni/sglang-v0.5.23` and can be changed with
`SGLANG_SOURCE_DIR`. Slow or proxied networks can override the installer's uv
defaults with `UV_HTTP_TIMEOUT` and `UV_HTTP_RETRIES`.

## Next steps

1. Activate the environment printed by the installer if it is not already active.

2. Export the FFmpeg libraries in every shell that starts a server. `ffmpeg@7`
   is intentional: the installer pins one FFmpeg major, and `torchcodec==0.17.0`
   ships loaders for FFmpeg 4 through 9.

   ```bash
   export DYLD_LIBRARY_PATH="$(brew --prefix ffmpeg@7)/lib${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
   ```

   macOS may remove `DYLD_*` variables when a SIP-protected system executable
   launches the server. Set `DYLD_LIBRARY_PATH` on the final `sgl-omni` process;
   for example, place `/usr/bin/env DYLD_LIBRARY_PATH=...` after wrappers such as
   `/usr/bin/time`.

3. Check that Metal and FFmpeg load before you download a checkpoint or start a
   server. TorchCodec loads Homebrew's keg-only FFmpeg libraries through the
   export above, so this also proves the export took effect:

   ```bash
   python - <<'PY'
   import torch
   from torchcodec.decoders import AudioDecoder

   assert torch.backends.mps.is_available()
   print("PyTorch MPS and TorchCodec FFmpeg loading are available")
   PY
   ```

4. Start a server from one of the cookbooks, then send a compressed input such
   as M4A or MP3 first: a WAV request can decode without loading FFmpeg.

   - [Run Qwen3-ASR with Torch/MPS](../cookbook/qwen3_asr.md#apple-silicon).
   - [Run Fun-ASR with Torch/MPS](../cookbook/fun_asr.md#apple-silicon).
