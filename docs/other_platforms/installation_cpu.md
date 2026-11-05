# Installation — Intel CPU

Install SGLang-Omni for CPU-only inference on Intel Xeon. For NVIDIA GPUs,
use the [CUDA installation](../get_started/installation.md).

## Prerequisites

- Python >= 3.10,<3.13
- `uv`
- A CPU build of SGLang from the matching upstream release.
- Standard audio runtime libraries such as `ffmpeg` and `libsndfile`.

<a id="option-a-docker"></a>

## Docker

```bash
# Clone the SGLang-omni repository
git clone https://github.com/sgl-project/sglang-omni.git
cd sglang-omni

# Build the docker image
docker build -f docker/cpu.Dockerfile -t sglang-omni:cpu .

# Initiate a docker container
docker run -it --shm-size 32g --ipc host --network host sglang-omni:cpu
```

The image installs upstream SGLang with its CPU pyproject, then installs `sglang-omni`
with `pyproject_cpu.toml`. It sets `SGLANG_USE_CPU_ENGINE=1` for the runtime.

<a id="option-b-manual-install"></a>

## Manual install

<a id="why-a-separate-pyproject"></a>

```{note}
[`pyproject_cpu.toml`](https://github.com/sgl-project/sglang-omni/blob/main/pyproject_cpu.toml)
selects CPU wheels. A plain editable install uses the CUDA
[`pyproject.toml`](https://github.com/sgl-project/sglang-omni/blob/main/pyproject.toml)
and can replace your CPU torch build. The editable CPU install requires
`--no-build-isolation`.
```

Create and activate an environment first:

```bash
git clone https://github.com/sgl-project/sglang-omni.git
cd sglang-omni
OMNI_DIR="$(pwd)"

uv venv .venv -p 3.12
source .venv/bin/activate
uv pip install --upgrade pip "packaging>=24.2" "setuptools>=77.0.0" wheel
```

Install the matching CPU SGLang build:

```bash
git clone https://github.com/sgl-project/sglang ../sglang
cd ../sglang
git checkout v0.5.23

cd python
cp pyproject_cpu.toml pyproject.toml
uv pip install -e . --no-build-isolation --extra-index-url https://download.pytorch.org/whl/cpu

cd sglang/kernels/aot
cp pyproject_cpu.toml pyproject.toml
uv pip install -e . --no-build-isolation --extra-index-url https://download.pytorch.org/whl/cpu
```

Install `sglang-omni` with the CPU pyproject:

```bash
cd "$OMNI_DIR"
bash scripts/cpu/install_cpu.sh
```

<a id="verify"></a>

## Verify the installation

```bash
python -c "import sglang_omni, torch; print(sglang_omni.__file__, torch.__version__)"
which sgl-omni
```

The torch version should resolve to a CPU build. CPU-specific unit tests live in
one directory, so CI (and you) can select them without touching the accelerator
suites:

```bash
SGLANG_USE_CPU_ENGINE=1 pytest tests/unit_test/cpu -v
```

> **`SGLANG_USE_CPU_ENGINE=1` is required at runtime.** Without it the platform
> layer reports `device_type == "cpu"` while `is_cpu()` stays `False`, so code
> that branches on the platform silently takes the accelerator path.

### Audio decoding fails with `libtorchcodec_core*.so`

`utils/audio.py` decodes through `torchcodec`, which loads FFmpeg's shared
libraries at import. An unsupported FFmpeg version can surface as the unhelpful
`Could not load this library` error.

`torchcodec` ships loaders for FFmpeg majors 4–8 only; FFmpeg 9 satisfies none
of them. Pin an older major if needed. Docker users get a supported major from
`apt`.

<a id="serve"></a>

## Serve a model

### Whisper ASR (speech-to-text, Intel Xeon CPU)

Whisper-base and Whisper-large-v3 have been validated on an AMX-capable Intel Xeon CPU. SGLang selects `intel_amx` by default on x86 CPUs and automatically falls back to `torch_native` when AMX is unavailable; graph capture, torch.compile, and async decode are disabled by default on CPU.

Run inside the CPU environment installed above:

```bash
SGLANG_USE_CPU_ENGINE=1 sgl-omni serve \
  --model-path openai/whisper-large-v3 \
  --port 8000
```

To explicitly select PyTorch native attention, including on AMX-capable Xeon CPUs, append `--asr.engine.attention_backend torch_native` to the server command.

See the [Whisper cookbook](../cookbook/whisper_asr.md) for request parameters, supported response formats, and the SeedTTS WER benchmark.

### Qwen3-TTS (text-to-speech, Intel Xeon CPU)

Qwen3-TTS Base has been validated on an AMX-capable Intel Xeon CPU. Install the `qwen-tts` runtime dependency as described in the [Qwen3-TTS cookbook](../cookbook/qwen3_tts.md#prerequisites).

Run inside the CPU environment installed above:

```bash
SGLANG_USE_CPU_ENGINE=1 sgl-omni serve \
  --model-path Qwen/Qwen3-TTS-12Hz-0.6B-Base \
  --port 8000
```

The Base checkpoint requires reference audio for synthesis. The reference transcript is optional. See the [Qwen3-TTS cookbook](../cookbook/qwen3_tts.md) for request examples and other checkpoints.
