# 🚀 Installation — Intel CPU

Installs `sglang-omni` for **CPU-only inference**. The default
[installation](./installation.md) targets CUDA, so this path uses the separate
[`pyproject_cpu.toml`](../../pyproject_cpu.toml) and the PyTorch CPU wheel index.

This CPU build supports **Qwen3-TTS only**. Other model families are not covered by
this installation path.

> **`--no-build-isolation` is required** for the editable `sglang-omni` install.

## Why a separate pyproject

`pip install -e .` resolves [`pyproject.toml`](../../pyproject.toml). In CUDA-oriented
checkouts, that can pull CUDA-only wheels and replace a CPU torch stack.
[`pyproject_cpu.toml`](../../pyproject_cpu.toml) pins the torch family to CPU wheels and
keeps the dependency set scoped to Qwen3-TTS serving.

## Prerequisites

- Python >= 3.10.
- A CPU build of SGLang from the matching upstream release.
- Standard audio runtime libraries such as `ffmpeg` and `libsndfile`.

## 🐳 Option A: Docker

```bash Command
# Clone the SGLang-omni repository
git clone https://github.com/sgl-project/sglang-omni.git

# Build the docker image
docker build -f docker/cpu.Dockerfile -t sglang-omni:cpu .

# Initiate a docker container
docker run -it --shm-size 32g --ipc host --network host sglang-omni:cpu
```

The image installs upstream SGLang with its CPU pyproject, then installs `sglang-omni`
with `pyproject_cpu.toml`. It sets `SGLANG_USE_CPU_ENGINE=1` for the runtime.

## 🛠️ Option B: Manual install

Create and activate an environment first:

```bash
git clone https://github.com/sgl-project/sglang-omni.git
cd sglang-omni
OMNI_DIR="$(pwd)"

uv venv .venv -p 3.12
source .venv/bin/activate
uv pip install --upgrade pip setuptools wheel
```

Install the matching CPU SGLang build:

```bash
git clone https://github.com/sgl-project/sglang ../sglang
cd ../sglang
git checkout v0.5.16

cd python
cp pyproject_cpu.toml pyproject.toml
uv pip install . --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple

cd sglang/kernels/aot
cp pyproject_cpu.toml pyproject.toml
uv pip install . --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple
```

Install `sglang-omni` with the CPU pyproject:

```bash
cd OMNI_DIR
bash scripts/cpu/install_cpu.sh
```
Qwen3-TTS needs the upstream `qwen-tts` package. Option A already includes it; for
Option B install it here, because `pyproject_xpu.toml` deliberately does not pin it.
`--no-deps` is required on both lines: `qwen-tts` pins Transformers 4.57.3, which
would replace this project's 5.12.1, and resolving `sox` lifts `numpy` past the
`numba==0.65.1` ceiling. See
[docs/cookbook/qwen3_tts.md](../cookbook/qwen3_tts.md).

```bash
sudo apt-get update
sudo apt-get install -y sox

uv pip install --no-deps sox
uv pip install --no-deps qwen-tts==0.1.1
```

--no-deps is required because qwen-tts pins an older Transformers version,
which would otherwise replace the Transformers version used by SGLang-Omni.
Installing Python sox without dependency resolution also avoids changing the
existing NumPy stack.

## Verify

```bash
python -c "import sglang_omni, torch; print(sglang_omni.__file__, torch.__version__)"
which sgl-omni
```

The torch version should resolve to a CPU build. If you installed the optional test
dependencies, you can also run the Qwen3-TTS unit tests:

```bash
pytest tests/unit_test/qwen3_tts -v
```

## Serve Qwen3-TTS

Run with the CPU engine enabled:

```bash
export SGLANG_USE_CPU_ENGINE=1

sgl-omni serve \
  --model-path Qwen/Qwen3-TTS-12Hz-0.6B-Base \
  --config examples/configs/qwen3_tts_0_6b.yaml \
  --host 0.0.0.0 \
  --port 8000
```

For the 1.7B Base checkpoint, use the matching config:

```bash
export SGLANG_USE_CPU_ENGINE=1

sgl-omni serve \
  --model-path Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --config examples/configs/qwen3_tts_1_7b.yaml \
  --host 0.0.0.0 \
  --port 8000
```

Qwen3-TTS Base checkpoints require a reference voice. Pass `ref_audio` and `ref_text`,
or the equivalent `references` array:

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    "voice": "default",
    "input": "Hello, this is Qwen3-TTS running on CPU.",
    "ref_audio": "/path/to/ref.wav",
    "ref_text": "Reference transcript for the voice prompt.",
    "language": "English",
    "response_format": "wav",
    "do_sample": false,
    "subtalker_dosample": false
  }' \
  --output output.wav
```

Health check:

```bash
curl http://localhost:8000/v1/models
```

> ✅ Support status: this CPU installation path supports **Qwen3-TTS** serving only.
