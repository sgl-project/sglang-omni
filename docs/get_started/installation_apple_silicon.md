# 🍎 Apple Silicon Installation

The Apple Silicon path targets macOS 14+, arm64, and Python 3.12. The installer
creates an isolated environment, builds the pinned SGLang MLX/MPS runtime, and
installs sglang-omni together with the shared audio prerequisites (`ffmpeg@7`,
`sox`, `uv`, and `git` when needed). It does not install or import the
`mlx-audio` package, download model weights, or create MLX-converted artifacts.

`ffmpeg@7` is intentional, not an old SGLang pin: the current Apple extra for
SGLang 0.5.19 installs `torchcodec==0.15.0`, whose wheel supports FFmpeg 4–8,
while Homebrew's unversioned `ffmpeg` formula is currently FFmpeg 9. Revisit
this formula only when the TorchCodec pin moves to a release with FFmpeg 9
support.

## One-line install

Review the script or pin the URL to a release/commit when reproducibility is
important. This single command installs the shared Apple audio runtime and the
model-side Python/runtime prerequisites currently provisioned by `install.sh`:

```bash
curl -fsSL https://raw.githubusercontent.com/sgl-project/sglang-omni/main/install.sh | bash
source ~/.cache/sglang-omni/.venv-apple/bin/activate
```

That includes the `fun-cosyvoice3` extra, the pinned CosyVoice/Matcha-TTS
source checkouts, and `qwen-tts==0.1.1` plus `einops` installed without their
conflicting transitive dependencies. Model weights and MLX-converted artifacts
remain separate downloads.

If Homebrew is missing, the script runs Homebrew's official bootstrapper. That
step can request administrator approval and Xcode Command Line Tools. To
require a pre-existing native Homebrew installation instead, set
`SGLANG_OMNI_BOOTSTRAP_HOMEBREW=0`. Set `--non-interactive` for CI; it disables
Homebrew auto-update, but cannot bypass macOS administrator requirements.

The script is safe to rerun. Its checkout and virtual-environment locations
are configurable with `SGLANG_OMNI_CACHE`, `SGLANG_OMNI_VENV`,
`SGLANG_SOURCE_DIR`, and `SGLANG_OMNI_PROJECT_DIR`. The venv activation scripts
automatically add keg-only FFmpeg 7 to `PATH` and `DYLD_LIBRARY_PATH`; no manual
FFmpeg export is needed after `source`.

For a local checkout, run `./install.sh` from its root and then source
`.venv-apple/bin/activate`. For a fork, set `SGLANG_OMNI_REPO` and
`SGLANG_OMNI_REF` before the command. Hosted mode keeps the checkout and venv
under `~/.cache/sglang-omni/` by default.

The installer keeps the Apple prerequisite registry (Homebrew formulas, Omni
extras, no-dependency model packages, and pinned source checkouts) in one place;
adding a future validated model updates that registry without adding another
user-facing install command.

For reproducible hosted installs, the raw script ref and the checkout ref must
match. For example:

```bash
INSTALL_REF=<tag-or-commit>
curl -fsSL "https://raw.githubusercontent.com/sgl-project/sglang-omni/${INSTALL_REF}/install.sh" \
  | SGLANG_OMNI_REF="$INSTALL_REF" bash
source ~/.cache/sglang-omni/.venv-apple/bin/activate
```

The model cookbooks contain model-specific artifacts and server settings:
[Qwen3-ASR](../cookbook/qwen3_asr.md) and
[Fun-CosyVoice3](../cookbook/fun_cosyvoice3.md).

