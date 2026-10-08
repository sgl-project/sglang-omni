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

Core deps cover the supported models (Qwen3-ASR / TTS / Omni / MiniMax Music 3, Fun-ASR-Nano, Nemotron 3.5 ASR, MOSS-Transcribe-Diarize, MiniCPM-o, Ming-Omni-TTS, PersonaPlex, AuK/AuK-Flash and dots.tts) plus the API server;
`[eval]` adds SeedTTS/WER tooling and `[all]` aliases it. ZONOS2 also serves here,
but its DAC codec is not a core dep on any platform — see
[ZONOS2](#zonos2-moe-tts-single-xpu) for the XPU-safe way to add it. Other model
families (S2-Pro, Ming-Omni, Voxtral-TTS) are CUDA-only and are not offered here.

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
# torch+xpu provides triton-xpu; do not let openai-whisper replace it with CUDA Triton.
pip install --no-deps openai-whisper==20250625
cp -f .pyproject.cuda.bak pyproject.toml && rm .pyproject.cuda.bak   # restore CUDA pyproject
```

### SGLang (installed separately)

`sglang` is intentionally **not** pinned, so the install above leaves an existing XPU build alone.
It cannot be pinned even as a range: every published wheel requires `flashinfer_python[cu13]` and the
`nvidia-*` runtime, so **any** specifier pulls the CUDA stack over `torch+xpu`. Build from source:

```bash
git clone https://github.com/sgl-project/sglang && cd sglang
git checkout v0.5.21   # the pinned release
cd python && cp pyproject_xpu.toml pyproject.toml
pip install -e . --no-build-isolation --extra-index-url https://download.pytorch.org/whl/xpu
pip install --no-deps xgrammar==0.1.33
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

### Fun-ASR-Nano (speech-to-text, single XPU)

Same endpoint as Qwen3-ASR, one uploaded clip of 30 s or less per request. See
[docs/cookbook/fun_asr.md](../cookbook/fun_asr.md) for the request parameters.

```bash
sgl-omni serve --model-path /path/to/Fun-ASR-Nano-2512-hf --host 0.0.0.0 --port 8000
# transcribe:
curl -s -X POST http://localhost:8000/v1/audio/transcriptions \
  -F "file=@sample.wav" -F "model=/path/to/Fun-ASR-Nano-2512-hf" -F "language=en"
```

The audio encoder captures a graph per (batch, length) bucket the first time it
sees one, on XPU as on CUDA. Buckets that fail to capture log a warning and run
eager, so a transcript is never at stake.

### Nemotron 3.5 ASR (speech-to-text, single XPU)

Run from the repository root after the XPU installation above, selecting the device with `ZE_AFFINITY_MASK`:

```bash
ZE_AFFINITY_MASK=0 sgl-omni serve \
  --model-path nvidia/nemotron-3.5-asr-streaming-0.6b \
  --host 0.0.0.0 --port 8000
```

In another terminal, upload the cookbook's reference recording:

```bash
curl -s -X POST http://localhost:8000/v1/audio/transcriptions \
  -F "file=@tests/data/query_to_cars.wav" \
  -F "model=nvidia/nemotron-3.5-asr-streaming-0.6b" \
  -F "language=auto" \
  -F "response_format=verbose_json"
```

See the [Nemotron cookbook](../cookbook/nemotron3_5_asr.md) for request parameters.

### Qwen3-TTS (text-to-speech, single XPU)

Qwen3-TTS needs the upstream `qwen-tts` package. Option A already includes it; for
Option B install it here, because `pyproject_xpu.toml` deliberately does not pin it.
`--no-deps` is required on both lines: `qwen-tts` pins Transformers 4.57.3, which
would replace this project's 5.12.1, and resolving `sox` lifts `numpy` past the
`numba==0.65.1` ceiling. See
[docs/cookbook/qwen3_tts.md](../cookbook/qwen3_tts.md).

```bash
apt-get update && apt-get install -y sox   # the Python sox package shells out to it
pip install --no-deps sox
pip install --no-deps qwen-tts==0.1.1
```

#### Base

```bash
sgl-omni serve --model-path /path/to/Qwen3-TTS-12Hz-1.7B-Base --host 0.0.0.0 --port 8000
# Base checkpoint clones a reference voice — pass ref_audio (+ ref_text):
curl -s -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model":"/path/to/Qwen3-TTS-12Hz-1.7B-Base","input":"Hello from Intel XPU.",
       "voice":"default","ref_audio":"/path/to/ref.wav","ref_text":"reference transcript",
       "response_format":"wav"}' -o out.wav
```

#### CustomVoice

`CustomVoice` checkpoints synthesize speech with built-in speakers and need no reference audio. Use the matching config and select a speaker in the request:

```bash
sgl-omni serve \
  --model-path Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  --config examples/configs/qwen3_tts_1_7b_customvoice.yaml \
  --host 0.0.0.0 --port 8000
curl -s -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice","input":"Hello from Intel XPU.",
       "voice":"Ryan","task_type":"CustomVoice","language":"English",
       "instructions":"Speak clearly and calmly.","response_format":"wav"}' -o out.wav
```

#### VoiceDesign

The `VoiceDesign` checkpoint synthesizes speech from text and a voice description, so it needs no reference audio. Serve it with its config:

```bash
sgl-omni serve \
  --model-path Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign \
  --config examples/configs/qwen3_tts_1_7b_voicedesign.yaml \
  --host 0.0.0.0 --port 8000
# VoiceDesign requires task_type and non-empty instructions:
curl -s -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model":"Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign","input":"Hello, how are you?",
       "voice":"default","task_type":"VoiceDesign",
       "instructions":"A warm, natural young adult voice.","response_format":"wav"}' -o out.wav
```

#### Codec decoding on XPU

The stateful incremental codec decoder runs, but the pipeline starts it with
`async_decode: false`, and the vocoder captures its decode graphs during the
asynchronous decode warmup, so none are captured. Capturing the shape set the CUDA
defaults imply has not been shown to pay here yet: on one Arc Pro B60 it ran past
the 600 s stage startup budget, and the engine then ran out of memory on a 24 GB
card once the graphs were resident.

The speaker encoder graphs do capture, on the default bucket ladder, for about 2 s of
extra startup. The reference encoder stays eager whatever the ladder says: its
transformers Mimi encoder reads a mask tensor on the host mid-forward, which a capture
cannot record, so the platform declines that one capability and the batcher logs
`qwen3_tts_reference_encoder_graph resolved=eager`.

To try the fast path, turn the asynchronous path back on; `incremental_codec_compile`
is worth dropping with it, since compiling the codec kernels is what dominated that
startup:

```yaml
stages:
  vocoder:
    factory:
      async_decode: true
      incremental_codec_compile: false
```

An explicit stage value wins over the pipeline default. See the platform-neutral
defaults in [docs/cookbook/qwen3_tts.md](../cookbook/qwen3_tts.md).

### AuK (speech generation and editing, single XPU)

Run one checkpoint at a time:

```bash
ZE_AFFINITY_MASK=0 sgl-omni serve \
  --model-path tencent/AuK \
  --host 0.0.0.0 --port 8000
```

```bash
ZE_AFFINITY_MASK=0 sgl-omni serve \
  --model-path tencent/AuK-Flash \
  --host 0.0.0.0 --port 8000
```

With either server, generate a 24 kHz WAV from text and voice instructions. Without reference audio, `stage_params.auk_engine.gen_seconds` is required:

```bash
curl -s -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input":"Welcome home.","instructions":"A warm, relaxed voice.",
       "stage_params":{"auk_engine":{"gen_seconds":3}},
       "seed":1234,"response_format":"wav"}' -o out.wav
```

AuK uses 32 sampling steps by default; AuK-Flash uses its fixed four-step recipe. See the [AuK cookbook](../cookbook/auk.md) for voice cloning and speech editing through `/generate`.

### dots.tts (text-to-speech, single XPU)

The XPU installation includes `dots.tts==0.2.1`.
All three checkpoints were validated on one 24 GB Intel Arc Pro B60 with bf16,
`mem_fraction_static=0.20`, and `max_generate_length=500`:

| Checkpoint | Config | `num_steps` | `max_running_requests` tested |
|---|---|---|---|
| `dots-studio/dots.tts-mf` | `examples/configs/dots_tts.yaml` | 4 | 4 |
| `dots-studio/dots.tts-soar` | `examples/configs/dots_tts_soar.yaml` | 10 | 1 |
| `dots-studio/dots.tts-base` | `examples/configs/dots_tts_soar.yaml` | 10 | 1 |

For MF on B60, lower the config's default 16 request slots to the 4 slots used in validation. Keep the checkpoint revision pinned by the config:

```bash
ZE_AFFINITY_MASK=0 sgl-omni serve \
  --config examples/configs/dots_tts.yaml \
  --latent_engine.engine.max_running_requests 4 \
  --latent_engine.engine.cuda_graph_max_bs 4 \
  --allowed-local-media-path docs/_static/audio \
  --host 0.0.0.0 --port 8000
```

For SOAR, use its single-request config:

```bash
ZE_AFFINITY_MASK=0 sgl-omni serve \
  --model-path dots-studio/dots.tts-soar \
  --config examples/configs/dots_tts_soar.yaml \
  --allowed-local-media-path docs/_static/audio \
  --host 0.0.0.0 --port 8000
```

For base, use the same command with
`--model-path dots-studio/dots.tts-base`. Base and SOAR require
`max_running_requests=1`; continuous batching is MF-only.

The `disable_cuda_graph` and `cuda_graph_max_bs` config names also control
SGLang backbone decode graphs on XPU. The configs enable them; use
`--latent_engine.engine.disable_cuda_graph true` for eager backbone decode.
The batched acoustic tail runs eager on XPU. Its memory admission precheck
queries free XPU memory and rejects oversized pools before allocation.
Lower `max_running_requests` and/or
`max_generate_length` explicitly; `mem_fraction_static` only budgets
the backbone KV cache.

Each request needs one reference clip and its matching transcript:

```bash
curl -sS -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "dots-studio/dots.tts-mf",
    "input": "Have a nice day and enjoy south california sunshine.",
    "references": [{
      "audio_path": "docs/_static/audio/male-voice.wav",
      "text": "Hey, Adam here. Let'\''s create something that feels real, sounds human, and connects every time."
    }],
    "seed": 42
  }' \
  --output output.wav
```

Set `model` to the checkpoint being served. The validation runs produced
48 kHz mono audio and passed ASR checks against the requested text.
See the [dots.tts cookbook](../cookbook/dots_tts.md) for streaming and
solver parameters.

### ZONOS2 (MoE TTS, single XPU)

ZONOS2 needs the Descript DAC codec. `--no-deps` is required:
`descript-audiotools==0.7.2` pins `protobuf<3.20`, which would downgrade this
environment's `protobuf` to 3.19.6. The other packages on the line are the codec's
own dependencies.

```bash
pip install --no-deps "descript-audiotools==0.7.2" "descript-audio-codec==1.0.0" \
  argbind julius pyloudnorm pystoi torch-stoi flatten-dict randomname \
  ffmpy fire markdown2 importlib_resources \
  tensorboard tensorboard-data-server absl-py Markdown Werkzeug
python -c "import dac; print('ok')"
```

Voice cloning transcodes the reference clip with **ffmpeg**, so `ffmpeg` must be on
`PATH`. Then serve — `params.json` auto-selects the architecture, so `--model-path`
is all that is needed:

```bash
sgl-omni serve --model-path Zyphra/zonos2 --host 0.0.0.0 --port 8000
# clone a voice from a reference clip:
curl -s -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input":"Hello from Intel XPU.",
       "references":[{"audio_path":"/path/to/ref.wav","text":"reference transcript"}]}' \
  -o out.wav
```

On XPU, ZONOS2 keeps its MoE experts in bf16 and leaves `torch.compile` off; decode
graphs stay on by default. All three are applied automatically.

### PersonaPlex (speech-to-speech, single XPU)

Follow the [PersonaPlex prerequisites](../cookbook/personaplex.md#prerequisites)
to accept the checkpoint license and download the model. PersonaPlex was
validated on one 24 GB Intel Arc Pro B60 with
`--lm.engine.mem_fraction_static 0.70`. Pick the XPU with `ZE_AFFINITY_MASK`:

```bash
ZE_AFFINITY_MASK=0 python examples/run_personaplex.py \
  --model-path nvidia/personaplex-7b-v1 \
  --audio /path/to/caller.wav \
  --voice NATF2 \
  --text-prompt "You are a wise and friendly teacher. Answer questions or provide advice in a clear and engaging way." \
  --out reply.wav \
  --lm.engine.mem_fraction_static 0.70
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

### MiniMax Music 3 (text-to-music, two XPUs)
```bash
# server
sgl-omni serve --model-path MiniMaxAI/MiniMax-Music3 --port 8000 --mem-fraction-static 0.7
# client request - Genre, instrumentation, tempo, and a production note
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "MiniMaxAI/MiniMax-Music3",
    "input": "[Chorus]\nWe are the fire that never dies\nBurning bright against the sky",
    "instructions": "An energetic arena rock anthem with distorted electric guitars, punchy live drums and a soaring male vocal at 130 BPM, wide stereo image, lightly compressed",
    "seed": 7,
    "max_new_tokens": 750
  }' \
  --output rock_1.wav
```

### Ming-Omni-TTS (16.8B-A3B MoE, two XPUs)

The bf16 AR backbone does not fit one 24 GB card, so `tts_engine` runs with TP=2. Its joint
RoPE is sgl-kernel's SYCL JIT kernel, which needs `icpx` on `PATH`; add the compiler directory
alone rather than sourcing `setvars.sh`. Its fp32 MoE routing needs the `sglang-kernel-xpu` 0.3.0
wheel that SGLang v0.5.21 pins; older sgl-kernel builds fail with
`"fused_topk_softmax_kernel" not implemented for 'Float'`.
```bash
export PATH="/opt/intel/oneapi/compiler/latest/bin:$PATH" SGLANG_OMNI_STARTUP_TIMEOUT=1800
sgl-omni serve --model-path /path/to/Ming-omni-tts-16.8B-A3B \
  --config examples/configs/ming_omni_tts.yaml \
  --tts_engine.tp_size 2 --tts_engine.gpu "[0, 1]" \
  --reference_encode.gpu 0 --audio_decode.gpu 0 \
  --tts_engine.gpu_memory_fraction 0.78 --tts_engine.engine.mem_fraction_static 0.78 \
  --host 0.0.0.0 --port 8000
curl -s -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"model":"ming-omni-tts","input":"Hello from Intel XPU.","voice":"default","response_format":"wav"}' \
  -o out.wav
```
The config's AudioVAE streaming graph runs eager on XPU, since oneMKL's FFT cannot be recorded
in a graph.

Health check for any of the above: `curl http://localhost:8000/v1/models`.

> **Expected on XPU:** `Failed to import mooncake` / `Failed to import nixl` warnings are harmless
> — those CUDA-only transfer backends are omitted; tensors move through the `shm` relay instead.

> ✅ Support status: **Qwen3-ASR, Fun-ASR-Nano, Nemotron 3.5 ASR, MOSS-Transcribe-Diarize, Qwen3-TTS, ZONOS2,
> Qwen3-Omni, MiniMax Music 3, MiniCPM-o, Ming-Omni-TTS, PersonaPlex and dots.tts all serve end-to-end on Intel XPU**
> (Qwen3-ASR, Fun-ASR-Nano, Nemotron 3.5 ASR, MOSS-Transcribe-Diarize, Qwen3-TTS, MiniCPM-o, PersonaPlex and dots.tts single-card;
> ZONOS2 single-card with decode graphs; MiniMax Music 3 and Ming-Omni-TTS need two cards;
> Qwen3-Omni thinker across 8 cards with tensor parallelism).
