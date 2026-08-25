# 🚀 Installation — Huawei Ascend NPU

Installs `sglang-omni` for **Huawei Ascend NPUs**. The default
[installation](./installation.md) pins CUDA-only wheels and would clobber a `torch+npu` stack.
Mirroring upstream SGLang ([Ascend NPU docs](https://docs.sglang.io/docs/hardware-platforms/ascend-npus/ascend_npu),
`docker/npu.Dockerfile`), the NPU path uses a **separate `pyproject_npu.toml`** plus a pre-installed
`torch_npu` / `triton-ascend` stack.

## Why a separate pyproject

`pip install -e .` resolves the CUDA [`pyproject.toml`](../../pyproject.toml), whose torch
family and CUDA-only wheels would replace the `torch+npu` stack.
[`pyproject_npu.toml`](../../pyproject_npu.toml) encodes the NPU replacements.

Core deps cover the supported models (Qwen3-ASR / TTS / Omni) plus the API server;
`[eval]` adds SeedTTS/WER tooling, `[all]` aliases it, and `[fun-cosyvoice3]`
adds the Fun-CosyVoice3 dependencies. Other model families
(S2-Pro, Ming-Omni, Voxtral-TTS) are CUDA-only and are not offered here.

> **`--no-build-isolation` is required** — without it pip emits a legacy in-tree
> `egg-info` instead of a PEP 660 editable install. The installer always passes it.
> Because of that pip does not install build requirements either, so this
> environment's own `setuptools` must be **≥ 77.0.0**: older releases reject the
> PEP 639 license metadata with ``invalid pyproject.toml config: `project.license` ``.
> The installer checks this before building; upgrade with
> `pip install -U 'setuptools>=77.0.0'`.

## Prerequisites

The NPU stack is **not** installed by `pyproject_npu.toml` or the install script.
You must set up the following components first, following the
[SGLang Ascend NPU guide](https://docs.sglang.io/docs/hardware-platforms/ascend-npus/ascend_npu):

| Component | Version | Install |
|-----------|---------|---------|
| CANN toolkit | 9.0.0 | [Installation guide](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/softwareinst/instg/instg_0008.html) |
| HDK (driver/firmware) | 25.5.2 | [Download](https://www.hiascend.com/hardware/firmware-drivers/community) |
| torch (CPU) | 2.10.0 | `pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cpu` |
| torch_npu | 2.10.0 | [gitcode.com/Ascend/pytorch/releases](https://gitcode.com/Ascend/pytorch/releases) (arch-specific wheel) |
| triton-ascend | 3.2.1.dev20260530 | `pip install triton-ascend==3.2.1.dev20260530 --extra-index-url=https://mirrors.huaweicloud.com/ascend/repos/pypi/nightly --trusted-host mirrors.huaweicloud.com` |
| memfabric-hybrid | 1.0.8 | `pip install memfabric-hybrid==1.0.8` (PD disaggregation only) |
| sgl-kernel-npu | latest | [sgl-project/sgl-kernel-npu](https://github.com/sgl-project/sgl-kernel-npu) releases |

> **Python 3.11 is the verified configuration.** The wheel examples below are
> CPython 3.11 builds; select matching `torch_npu` wheels if you use another
> supported Python version. To reproduce the verified environment, use
> `conda create -n sglang-omni-npu python=3.11`.

### torch_npu installation

`torch_npu` wheels are arch-specific and hosted on Huawei's gitcode mirror, not on PyPI.
Download the matching wheel for your architecture:

```bash
# aarch64 (Atlas 800I A2 / A3)
pip install https://gitcode.com/Ascend/pytorch/releases/download/v26.0.0-pytorch2.10.0/torch_npu-2.10.0-cp311-cp311-manylinux_2_28_aarch64.whl

# x86_64
pip install https://gitcode.com/Ascend/pytorch/releases/download/v26.0.0-pytorch2.10.0/torch_npu-2.10.0-cp311-cp311-manylinux_2_28_x86_64.whl
```

## 🛠️ Install sglang-omni

The helper swaps in `pyproject_npu.toml`, installs with `--no-build-isolation`, then
restores the CUDA one:

```bash
git clone https://github.com/sgl-project/sglang-omni.git
cd sglang-omni

# dry-run first — shows the commands, installs nothing
PYTHON=$(which python) scripts/npu/install_npu.sh --check

# editable install
PYTHON=$(which python) scripts/npu/install_npu.sh
```

Pick extras with `--extras` (comma-separated):

```bash
scripts/npu/install_npu.sh --extras eval           # core + SeedTTS/WER eval + tests
scripts/npu/install_npu.sh --extras all            # alias for eval
scripts/npu/install_npu.sh --extras fun-cosyvoice3 # Fun-CosyVoice3 runtime
```

Or do it manually (the same steps the script automates):

```bash
cp pyproject.toml .pyproject.cuda.bak
cp pyproject_npu.toml pyproject.toml
pip install -e . --no-build-isolation
cp -f .pyproject.cuda.bak pyproject.toml && rm .pyproject.cuda.bak   # restore CUDA pyproject
```

### SGLang (installed separately)

`sglang` is intentionally **not** pinned, so the install above leaves an existing NPU build alone.
It cannot be pinned even as a range: every published wheel requires `flashinfer_python[cu13]` and the
`nvidia-*` runtime, so **any** specifier pulls the CUDA stack over `torch+npu`. Build from source:

```bash
git clone https://github.com/sgl-project/sglang && cd sglang
git checkout v0.5.16   # the pinned release
cd python && cp pyproject_npu.toml pyproject.toml
pip install -e . --no-build-isolation
```

Use that commit: the NPU port targets this SGLang revision's APIs and does not carry
version-compatibility shims. A VCS requirement (`pip install "sglang @ git+…"`) does **not** work:
pip reads the checkout's `python/pyproject.toml`, which pins CUDA torch; only the swap above
selects the NPU manifest.

## Verify

```bash
# import works from anywhere now (package installed, not just cwd-on-path)
python -c "import sglang_omni, torch; print(sglang_omni.__file__, torch.__version__)"
which sgl-omni

# device-layer unit tests (CPU, no NPU) — needs pytest, which ships in the
# `[eval]` extra (install with `.[eval]`, or `pip install pytest` first)
pytest tests/unit_test/test_platforms.py -v
```

## Serve

### Qwen3-ASR (speech-to-text, single NPU)

```bash
sgl-omni serve --model-path /path/to/Qwen3-ASR-1.7B --host 0.0.0.0 --port 8000
# transcribe:
curl -s -X POST http://localhost:8000/v1/audio/transcriptions \
  -F "file=@sample.wav" -F "model=/path/to/Qwen3-ASR-1.7B"
```

### Qwen3-TTS (text-to-speech, single NPU)

Qwen3-TTS needs the upstream `qwen-tts` package, which is not pinned in `pyproject_npu.toml`.
Install it with `--no-deps` to avoid version conflicts:

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
  -d '{"model":"/path/to/Qwen3-TTS-12Hz-1.7B-Base","input":"Hello from Ascend NPU.",
       "voice":"default","ref_audio":"/path/to/ref.wav","ref_text":"reference transcript",
       "response_format":"wav"}' -o out.wav
```

### Qwen3-Omni (30B-A3B MoE, multi-NPU tensor parallel)

The 30B MoE does not fit one card; shard the thinker across NPUs with tensor parallelism.
`--text-only` serves the thinker (chat) without the talker/speech stages. The text-only
config normally puts every stage in the `pipeline` process, so give the TP thinker an
otherwise-unused process name before enabling TP:

```bash
# thinker across 8 cards (TP=8). Large shards over shared storage load slowly, so give
# startup more headroom than the default 600 s.
export SGLANG_OMNI_STARTUP_TIMEOUT=1800
sgl-omni serve --model-path /path/to/Qwen3-Omni-30B-A3B-Instruct \
  --text-only --stages.thinker.process thinker \
  --thinker-tp-size 8 --thinker-gpus 0,1,2,3,4,5,6,7 \
  --host 0.0.0.0 --port 8000
# chat:
curl -s -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"/path/to/Qwen3-Omni-30B-A3B-Instruct",
       "messages":[{"role":"user","content":"What is Ascend NPU?"}],"max_tokens":64}'
```

Health check for any of the above: `curl http://localhost:8000/v1/models`.

> **Expected on NPU:** `Failed to import mooncake` / `Failed to import nixl` warnings are harmless
> — those CUDA-only transfer backends are omitted; tensors move through the `shm` relay instead
> (or `memfabric` if installed).
