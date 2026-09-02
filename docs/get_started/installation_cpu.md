# 🚀 Installation — Intel CPU

Installs `sglang-omni` for **CPU-only inference**. The default
[installation](./installation.md) targets CUDA, so this path uses the separate
[`pyproject_cpu.toml`](../../pyproject_cpu.toml) and the PyTorch CPU wheel index.

> **`--no-build-isolation` is required** for the editable `sglang-omni` install.

## Why a separate pyproject

`pip install -e .` resolves [`pyproject.toml`](../../pyproject.toml). In CUDA-oriented
checkouts, that can pull CUDA-only wheels and replace a CPU torch stack.
[`pyproject_cpu.toml`](../../pyproject_cpu.toml) pins the torch family to CPU wheels and
omits accelerator-only packages.

## Prerequisites

- Python >= 3.10,<3.13
- `uv`
- A CPU build of SGLang from the matching upstream release.
- Standard audio runtime libraries such as `ffmpeg` and `libsndfile`.

## 🐳 Option A: Docker

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

## 🛠️ Option B: Manual install

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
git checkout v0.5.18

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

## Verify

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
