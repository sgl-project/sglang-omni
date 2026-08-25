# 🚀 Installation — Intel XPU

Installs `sglang-omni` for **Intel GPUs (XPU)**. The default
[installation](./installation.md) pins CUDA-only wheels and would clobber a `torch+xpu` stack.
Mirroring upstream SGLang ([Intel XPU docs](https://docs.sglang.io/docs/hardware-platforms/xpu),
`docker/xpu.Dockerfile`), the XPU path uses a **separate `pyproject_xpu.toml`** plus the PyTorch
XPU wheel index.

## Why a separate pyproject

`pip install -e .` resolves the CUDA [`pyproject.toml`](../../pyproject.toml), whose torch
family and CUDA-only wheels would replace the `+xpu` stack.
[`pyproject_xpu.toml`](../../pyproject_xpu.toml) encodes the XPU replacements.

Core deps cover the supported models (Qwen3-ASR / TTS / Omni) plus the API server;
`[eval]` adds SeedTTS/WER tooling and `[all]` aliases it. Other model families
(S2-Pro, Ming-Omni, Voxtral-TTS) are CUDA-only and are not offered here.

> **`--no-build-isolation` is required** — without it pip emits a legacy in-tree
> `egg-info` instead of a PEP 660 editable install. The installer always passes it.
> Because of that pip does not install build requirements either, so this
> environment's own `setuptools` must be **≥ 77.0.0**: older releases reject the
> PEP 639 license metadata with ``invalid pyproject.toml config: `project.license` ``.
> The installer checks this before building; upgrade with
> `pip install -U 'setuptools>=77.0.0'`.

## Prerequisites

- Python ≥ 3.10, and an Intel GPU driver (`/dev/dri/renderD*` present).
- `setuptools` ≥ 77.0.0 in the target environment (see the note above).
- The **PyTorch XPU stack** and an **XPU SGLang build** — reuse an existing working
  `torch+xpu` env if you have one. See [Runtime environment](#runtime-environment-important)
  for the oneAPI caveat.

## 🐳 Option A: Docker

```bash
docker build -f docker/xpu.Dockerfile -t sglang-omni:xpu .
docker run -it --device /dev/dri --shm-size 32g --ipc host --network host sglang-omni:xpu
```

Built on Intel Deep Learning Essentials with the `+xpu` torch wheels. It deliberately does **not**
source oneAPI — see [Runtime environment](#runtime-environment-important).

Pinning SGLang does not pin the SYCL kernels: its XPU manifest requires `sgl-kernel-xpu`
from git with no revision. The Dockerfile therefore pins that commit itself, so rebuilds
are reproducible by default. Override it only to move deliberately:

```bash
docker build -f docker/xpu.Dockerfile \
  --build-arg SGL_KERNEL_XPU_REF=<sgl-kernel-xpu commit sha> \
  -t sglang-omni:xpu .
```

## 🛠️ Option B: Install into an existing XPU env (recommended here)

The helper swaps in `pyproject_xpu.toml`, installs with the XPU index, then restores the CUDA one:

```bash
git clone git@github.com:sgl-project/sglang-omni.git
cd sglang-omni

# dry-run first — shows the commands, installs nothing
PYTHON=$(which python) scripts/xpu/install_xpu.sh --check

# editable install against the PyTorch XPU index
PYTHON=$(which python) scripts/xpu/install_xpu.sh
```

Pick extras with `--extras` (comma-separated):

```bash
scripts/xpu/install_xpu.sh --extras eval           # core + SeedTTS/WER eval + tests
scripts/xpu/install_xpu.sh --extras all            # alias for eval
```

Or do it manually (the same steps the script automates):

```bash
cp pyproject.toml .pyproject.cuda.bak
cp pyproject_xpu.toml pyproject.toml
pip install -e . --no-build-isolation --extra-index-url https://download.pytorch.org/whl/xpu
cp -f .pyproject.cuda.bak pyproject.toml && rm .pyproject.cuda.bak   # restore CUDA pyproject
```

### SGLang (installed separately)

`sglang` is intentionally **not** pinned, so the install above leaves an existing XPU build alone.
It cannot be pinned even as a range: every published wheel requires `flashinfer_python[cu13]` and the
`nvidia-*` runtime, so **any** specifier pulls the CUDA stack over `torch+xpu`. Build from source:

```bash
git clone https://github.com/sgl-project/sglang && cd sglang
git checkout v0.5.16   # the pinned release
cd python && cp pyproject_xpu.toml pyproject.toml
pip install -e . --no-build-isolation --extra-index-url https://download.pytorch.org/whl/xpu
```

Use that commit: the XPU port targets this SGLang revision's APIs and does not carry
version-compatibility shims. A VCS requirement (`pip install "sglang @ git+…"`) does **not** work:
pip reads the checkout's `python/pyproject.toml`, which pins CUDA torch; only the swap above
selects `+xpu`.

## Verify

```bash
# import works from anywhere now (package installed, not just cwd-on-path)
python -c "import sglang_omni, torch; print(sglang_omni.__file__, torch.__version__)"
which sgl-omni

# device-layer unit tests (CPU, no GPU) — needs pytest, which ships in the
# `[eval]` extra (install with `.[eval]`, or `pip install pytest` first)
pytest tests/unit_test/xpu/test_device_layer.py -v
```

## Serve

### Runtime environment (important)

Run in the **PyTorch-XPU environment as-is** — do **not** `source /opt/intel/oneapi/setvars.sh`.
The `+xpu` wheels ship their own oneCCL/SYCL/Level-Zero; a system oneAPI puts a different oneCCL/UCX
on the library path, conflicting with the bundled `libccl` and crashing multi-XPU `xccl` collectives.

No extra environment variables are needed — the XPU backend is auto-detected. If a Triton JIT
build reports `fatal error: sycl/sycl.hpp: No such file or directory`, point the compiler at the
`intel-sycl-rt` wheel's headers:
```bash
export CPATH="$(python -c 'import sysconfig; print(sysconfig.get_paths()["include"])')"
```

### Qwen3-ASR (speech-to-text, single XPU)

```bash
sgl-omni serve --model-path /path/to/Qwen3-ASR-1.7B --host 0.0.0.0 --port 8000
# transcribe:
curl -s -X POST http://localhost:8000/v1/audio/transcriptions \
  -F "file=@sample.wav" -F "model=/path/to/Qwen3-ASR-1.7B"
```

### Qwen3-TTS (text-to-speech, single XPU)

Qwen3-TTS needs the upstream `qwen-tts` package. Option A already includes it; for
Option B install it here, because `pyproject_xpu.toml` deliberately does not pin it.
`--no-deps` is required on both lines: `qwen-tts` pins Transformers 4.57.3, which
would replace this project's 5.12.1, and resolving `sox` lifts `numpy` past the
`numba==0.65.1` ceiling. See
[docs/cookbook/qwen3_tts.md](../cookbook/qwen3_tts.md).

```bash
apt-get update && apt-get install -y sox   # the Python sox package shells out to it
pip install --no-deps sox einops
pip install --no-deps qwen-tts==0.1.1
```

```bash
sgl-omni serve --model-path /path/to/Qwen3-TTS-12Hz-1.7B-Base --host 0.0.0.0 --port 8000
# Base checkpoint clones a reference voice — pass ref_audio (+ ref_text):
curl -s -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model":"/path/to/Qwen3-TTS-12Hz-1.7B-Base","input":"Hello from Intel XPU.",
       "voice":"default","ref_audio":"/path/to/ref.wav","ref_text":"reference transcript",
       "response_format":"wav"}' -o out.wav
```

### Qwen3-Omni (30B-A3B MoE, multi-XPU tensor parallel)

The 30B MoE does not fit one 24 GB card; shard the thinker across GPUs with tensor parallelism.
`--text-only` serves the thinker (chat) without the talker/speech stages. The text-only config normally puts every stage in the `pipeline` process, so give the TP thinker an otherwise-unused process name before enabling TP:

```bash
# thinker across 8 cards (TP=8). Large shards over shared storage load slowly, so give
# startup more headroom than the default 600 s.
export SGLANG_OMNI_STARTUP_TIMEOUT=1800
sgl-omni serve --model-path /path/to/Qwen3-Omni-30B-A3B-Instruct \
  --text-only --thinker.process thinker \
  --thinker.tp_size 8 --thinker.gpu "[0, 1, 2, 3, 4, 5, 6, 7]" \
  --host 0.0.0.0 --port 8000
# chat:
curl -s -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"/path/to/Qwen3-Omni-30B-A3B-Instruct",
       "messages":[{"role":"user","content":"What is Intel XPU?"}],"max_tokens":64}'
```

Health check for any of the above: `curl http://localhost:8000/v1/models`.

> **Expected on XPU:** `Failed to import mooncake` / `Failed to import nixl` warnings are harmless
> — those CUDA-only transfer backends are omitted; tensors move through the `shm` relay instead.

> ✅ Support status: **Qwen3-ASR, Qwen3-TTS, and Qwen3-Omni all serve end-to-end on Intel XPU**
> (ASR single-card, TTS single-card, Qwen3-Omni thinker across 8 cards with tensor parallelism).
