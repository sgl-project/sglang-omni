# Omni Model Usage

This guide uses [Qwen3-Omni](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct) as an example omni model with SGLang-Omni and the OpenAI-compatible API. Qwen3-Omni supports multi-modal input (text, image, audio) and can produce text-only or text + audio output depending on the mode.

## Prerequisites

Install `sglang-omni` by following [Installation](../get_started/installation.md).

## Text-Only Mode

Text-only mode runs the thinker pipeline on a single GPU. It accepts multi-modal input (text, image, audio) and produces text output only.

### Launch the Server

```bash
sgl-omni serve \
  --model-path Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --text-only \
  --port 8008
```

Multimodal preprocessing (tokenization, image/video/audio feature extraction)
runs serially by default. Add `--preprocessing.factory.max_concurrency 4` to
run it on a thread pool when CPU preprocessing is the bottleneck under
concurrent load. Threaded preprocessing changes how requests arrive at the
thinker, so greedy outputs of the bf16 MoE can differ from the serial default;
keep the default when comparing accuracy across runs.

For MMSU-style audio-input / text-output benchmarks with short requests, use
the fused text-path config so the full text path stays inside one worker
process:

```bash
sgl-omni serve \
  --model-path Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --config examples/configs/qwen3_omni_mmsu.yaml \
  --text-only \
  --port 8008
```

### Image and Text Input

Send an image with a text question to get a text response.

**cURL**

```bash
curl -X POST http://localhost:8008/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-omni",
    "messages": [{"role": "user", "content": "How many cars are there in the picture?"}],
    "images": ["tests/data/cars.jpg"],
    "modalities": ["text"],
    "max_tokens": 16
  }'
```

**Python**

```python
import requests

resp = requests.post(
    "http://localhost:8008/v1/chat/completions",
    json={
        "model": "qwen3-omni",
        "messages": [{"role": "user", "content": "How many cars are there in the picture?"}],
        "images": ["tests/data/cars.jpg"],
        "modalities": ["text"],
        "max_tokens": 16,
    },
)
resp.raise_for_status()
result = resp.json()
print(result["choices"][0]["message"]["content"])
```

### Audio and Image Input

Send an audio file together with an image. The audio contains the spoken question ("How many cars are there in the picture?") and the model answers based on both inputs.

> **Note:** Set `"content": ""` (empty string) on the user message when all semantic content comes from audio, video, or images rather than text.

**cURL**

```bash
curl -X POST http://localhost:8008/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-omni",
    "messages": [{"role": "user", "content": ""}],
    "images": ["tests/data/cars.jpg"],
    "audios": ["tests/data/query_to_cars.wav"],
    "modalities": ["text"],
    "max_tokens": 16
  }'
```

**Python**

```python
import requests

resp = requests.post(
    "http://localhost:8008/v1/chat/completions",
    json={
        "model": "qwen3-omni",
        "messages": [{"role": "user", "content": ""}],
        "images": ["tests/data/cars.jpg"],
        "audios": ["tests/data/query_to_cars.wav"],
        "modalities": ["text"],
        "max_tokens": 16,
    },
)
resp.raise_for_status()
result = resp.json()
print(result["choices"][0]["message"]["content"])
```

### Video and Audio Input

Send a video with a spoken audio question. The model watches the video, hears the question, and responds with text.

The Video-AMME CI benchmark uses this same modality combination: video input
plus a spoken question/options WAV, with only routing and answer-format
instructions in the text message.

**cURL**

```bash
curl -X POST http://localhost:8008/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-omni",
    "messages": [{"role": "user", "content": ""}],
    "videos": ["tests/data/draw.mp4"],
    "audios": ["tests/data/query_to_draw.wav"],
    "modalities": ["text"],
    "max_tokens": 16
  }'
```

**Python**

```python
import requests

resp = requests.post(
    "http://localhost:8008/v1/chat/completions",
    json={
        "model": "qwen3-omni",
        "messages": [{"role": "user", "content": ""}],
        "videos": ["tests/data/draw.mp4"],
        "audios": ["tests/data/query_to_draw.wav"],
        "modalities": ["text"],
        "max_tokens": 16,
    },
)
resp.raise_for_status()
result = resp.json()
print(result["choices"][0]["message"]["content"])
```

## Speech Mode

Speech mode runs the full eight-stage pipeline on one or more GPUs. It produces
both text (from the thinker) and audio (from the talker) output.

### Launch the Server

Speech mode can run as a colocated one-GPU worker using the colocated config:

```bash
sgl-omni serve \
  --model-path Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --config examples/configs/qwen3_omni_colocated_h20.yaml \
  --colocate \
  --port 8008
```

Use `examples/configs/qwen3_omni_colocated_h200.yaml` on single-H200 workers.

Exact-shape CUDA Graph replay is enabled by default for Qwen3-Omni Code2Wav.
The default stage config supplies a 2% typed GPU memory budget; colocated
example configs override it with their hardware-specific budget.

To disable replay, set it on the stage in the YAML config:

```yaml
stages:
  code2wav:
    factory:
      enable_cuda_graph: false
```

When replay is enabled, a custom Code2Wav stage must define
`gpu_memory_fraction`; startup rejects a missing typed budget before loading
the model.

The feature derives the exact `B=1` threshold windows from
`stream_chunk_size` and `left_context_size`; the defaults capture
`T{10,20,30,35}`. Unsupported shapes and final stream tails run eagerly.
Capture-time incompatibilities also fall back to eager execution.

Output overlap is also enabled by default on CUDA devices: each threshold
window's waveform readback runs as an asynchronous device-to-host copy into a
pinned staging buffer and is materialized while the GPU computes the next
window, and the codec EOS check runs once per window instead of once per
frame. The first window of every request stays synchronous, so
time-to-first-audio is unchanged. For successfully completed requests, audio
bytes and message boundaries are identical to the synchronous path. A request
that fails or is aborted can drop an un-emitted pending window while its
in-flight staging buffer is retired. To disable it:

```yaml
stages:
  code2wav:
    factory:
      enable_output_overlap: false
```

For manual multi-GPU placement, use the example script:

```bash
python examples/run_omni.py qwen3-speech-server \
  --model-path Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --gpu-thinker 0 \
  --gpu-talker 1 \
  --gpu-code-predictor 1 \
  --gpu-code2wav 0 \
  --port 8008
```

Or use the CLI without `--text-only` for the standard speech pipeline:

```bash
sgl-omni serve \
  --model-path Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --port 8008
```

By default, leave `mem_fraction_static` unset and let SGLang-Omni auto-size the
SGLang AR memory budget. If a specific machine needs manual tuning, you can pin
the value globally or per AR stage:

```bash
sgl-omni serve \
  --model-path Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --port 8008 \
  --mem-fraction-static 0.88
```

Use per-stage flags when the thinker and talker need different budgets:

```bash
sgl-omni serve \
  --model-path Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --port 8008 \
  --thinker.engine.mem_fraction_static 0.88 \
  --talker_ar.engine.mem_fraction_static 0.88
```

The speech server launcher exposes the same per-stage controls:

```bash
python examples/run_omni.py qwen3-speech-server \
  --model-path Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --gpu-thinker 0 \
  --gpu-talker 1 \
  --gpu-code-predictor 1 \
  --gpu-code2wav 0 \
  --port 8008 \
  --thinker-mem-fraction-static 0.88 \
  --talker-mem-fraction-static 0.88
```

`--mem-fraction-static` applies to every SGLang engine stage. A dotted
per-stage path overrides the global value for that stage. Values must be
greater than `0` and less than `1`.

The thinker admits up to 64 running requests by default. Use the
thinker-specific flag to lower or raise that limit in either text-only or
speech mode:

```bash
sgl-omni serve \
  --model-path Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --thinker.engine.max_running_requests 16
```

To configure the thinker through a pipeline YAML file instead, set the same
path under the stage entry:

```yaml
stages:
  thinker:
    engine:
      max_running_requests: 16
```

### Speech Stage Placement

At concurrency 8, the talker is the heaviest speech stage: it holds its GPU
at about 86% median utilization. Give it a GPU of its own. The code2wav
vocoder is light by comparison (9–13% median utilization) and shares the
thinker's GPU by default. That default holds only when the thinker stays on
its own default GPU — a `--thinker.gpu` override moves the thinker alone,
not code2wav, so pass `--code2wav.gpu` explicitly too if you relocate the
thinker.

This is the default topology for `sgl-omni serve` without GPU overrides:
thinker alone, talker alone, code2wav on the thinker's GPU. When code2wav
shares the thinker's GPU, the thinker's auto-sized KV pool shrinks to make
room for it — about 4.3 GiB smaller, measured on H200, since the vocoder
itself needs roughly 1.4–1.6 GiB. That adjustment only happens for the
auto-sized budget: an explicitly pinned `--thinker.engine.mem_fraction_static`
gets no automatic carve-out, so a tightly pinned fraction should leave
headroom for code2wav.

A concurrency-8 experiment held code2wav fixed on the thinker's GPU and
varied only the talker's placement, from sharing the thinker's GPU to
having its own. Every concurrent metric improved:

| metric (concurrency 8, two measured pairs) | talker shares the thinker's GPU | talker alone | change |
|---|---|---|---|
| wall-clock time | 22.8–24.1 s | 20.2–20.7 s | 9–16% lower |
| time to first audio (TTFA), p50 | 1.43–1.71 s | 0.99–1.20 s | 16–42% lower |
| TTFA, p90 | 2.66–2.70 s | 1.42–1.46 s | 46–47% lower |
| end-to-end latency, p50 | 10.04–10.05 s | 8.94–9.44 s | 6–11% lower |

A follow-up single-variable pair tested moving code2wav to its own GPU
instead, with the talker already isolated in both arms. Wall-clock time and
end-to-end latency were flat (−0.7% and −0.9%, well within noise for a
single interleaved pair) — giving the vocoder a dedicated GPU bought
nothing, which is why code2wav shares the thinker's GPU instead of getting
isolated like the talker.

Single-stream traffic (one request at a time) sees a different trade-off.
In the talker-placement experiment above, isolating the talker — moving it
off the thinker's GPU — made single-stream end-to-end latency 14–17% worse:
with one request in flight there is no contention to escape, while that
experiment's thinker-to-talker handoff started crossing a device boundary.
Single-stream TTFA still improved by about 30% (0.48–0.49 s to 0.34–0.35 s),
because the talker no longer waits its turn on a shared GPU. Both the old
and the new default layout already keep the thinker and talker on separate
GPUs, so these numbers describe that experiment's topology change, not the
default code2wav placement. Streaming TTS workloads should prefer this
layout regardless, since TTFA is the latency users notice first.

### Realtime Speech with Server-Side Turn Detection

The speech pipeline can stream spoken responses over `/v1/realtime`. Enable the
WebSocket endpoint on the standard speech pipeline:

```bash
sgl-omni serve \
  --model-path Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --port 8008 \
  --enable-realtime
```

After connecting to `ws://localhost:8008/v1/realtime`, request text and audio
output:

```json
{
  "type": "session.update",
  "session": {
    "modalities": ["text", "audio"],
    "input_audio_format": "pcm16",
    "output_audio_format": "pcm16",
    "turn_detection": {
      "type": "semantic_vad",
      "eagerness": "medium"
    }
  }
}
```

Stream mono 16 kHz PCM16 input with `input_audio_buffer.append`. Turn
detection auto-commits each utterance and starts generation. Check
`session.created.capabilities.turn_detection` before requesting
`semantic_vad` — older servers only support `server_vad`.

`server_vad` (default) ends a turn after a fixed silence duration.
`semantic_vad` adds a GPU Smart Turn v3.2 model on top of Silero speech
detection, so a natural mid-thought pause doesn't end the turn early.
`eagerness` (`low`/`medium`/`high`, default `medium`) trades latency for
patience; `silence_duration_ms` only applies to `server_vad`.

To enable `semantic_vad`, provision the BSD-2 licensed
[Smart Turn v3.2](https://huggingface.co/pipecat-ai/smart-turn-v3)
`smart-turn-v3.2-gpu.onnx` model and set `SGLANG_OMNI_SMART_TURN_MODEL_PATH`
to its path (file or containing directory). The server never downloads it and
verifies its SHA-256 on load. If the model is missing or invalid, the
endpoint still works — semantic requests just fall back to `server_vad`.

Text arrives in `response.text.delta` events; spoken output arrives as
base64-encoded mono 24 kHz PCM16 in `response.audio.delta` events, followed by
`response.audio.done` and `response.done`.

Audio output is opt-in: sessions remain text-only unless both modalities are
requested. A thinker-only server rejects audio negotiation because it has no
`code2wav` stage.

For text-and-audio sessions, server-owned barge-in is enabled by default. When
the active turn detector emits `input_audio_buffer.speech_started`, the active
response is cancelled with reason `turn_detected`; its user transcription still
completes and enters conversation history before the next queued turn runs.
Cancelled assistant output is not added to conversation history.

Clients must stop buffered playback on `speech_started` and reject every later
`response.audio.delta` for that response until its `response.done`. If speech
starts before `response.created`, retain a pending-interruption flag and reject
the response when its ID arrives. Automatic barge-in ends with
`response.done.status="cancelled"` and reason `turn_detected`; explicit
`response.cancel` uses reason `client_cancelled`.

If assistant audio has already been scheduled for playback, the client must
also send `conversation.item.truncate` with the assistant `item_id` from
`response.audio.delta`, `content_index: 0`, and the played duration in
`audio_end_ms`. The server replies with `conversation.item.truncated` and
removes that assistant item from conversation history. The whole assistant
transcript is removed because the endpoint cannot align text with played audio.

Set `turn_detection.interrupt_response` to `false` in `session.update` to opt
out. Partial updates preserve the active detector type and settings, so clients
can change interruption behavior without dropping semantic VAD or its
eagerness. Changing detector behavior rebuilds the detector and clears pending
input audio; `interrupt_response` changes independently. Text-only responses
are not interrupted automatically.

The browser example in `playground/qwen-omni/realtime` captures microphone
input, negotiates turn-detection support per connection, and lets the user
select text-only output or text plus streamed PCM16 audio playback.

## Single-GPU FP8 on H100/H20

SGLang-Omni can also serve native FP8 Qwen3-Omni checkpoints. Native FP8 uses
the checkpoint quantization config when loading the thinker and talker AR stages,
while keeping the same Qwen3-Omni request format shown below.

For one-GPU H100/H20 colocated launch, use the FP8 colocated config:

```bash
sgl-omni serve \
  --config examples/configs/qwen3_omni_fp8_colocated.yaml \
  --colocate \
  --model-name qwen3-omni \
  --port 8008
```

The config file contains the FP8 checkpoint path:
`marksverdhei/Qwen3-Omni-30B-A3B-FP8`. You can still pass `--model-path` to
override the config value.

The FP8 path keeps dense FP8 GEMM on SGLang `auto` and defaults native FP8 MoE
to CUTLASS when supported. For Qwen3-Omni pipeline launches,
`SGLANG_JIT_DEEPGEMM_PRECOMPILE=0` is set as a default unless the operator has
already set that environment variable. This disables SGLang's all-M DeepGEMM
precompile session while keeping DeepGEMM available for dense FP8 GEMMs.

To opt back into SGLang's all-M DeepGEMM precompile behavior:

```bash
SGLANG_JIT_DEEPGEMM_PRECOMPILE=1 sgl-omni serve \
  --config examples/configs/qwen3_omni_fp8_colocated.yaml \
  --colocate \
  --model-name qwen3-omni \
  --port 8008
```

## Single-GPU AutoRound INT4 Thinker on H100/H20

SGLang-Omni also supports AutoRound INT4 quantized Qwen3-Omni checkpoints.
AutoRound uses a 4-bit quantization scheme with group size 128, significantly
reducing memory footprint compared to BF16 or FP8.

The public AutoRound checkpoint quantizes the thinker transformer layers. In
speech mode, the talker and code2wav stages load as BF16 from the same
checkpoint. For one-GPU H100/H20 colocated launch, use the colocated config
with the AutoRound checkpoint:

```bash
sgl-omni serve \
  --config examples/configs/qwen3_omni_colocated_h20.yaml \
  --colocate \
  --model-name qwen3-omni \
  --model-path Intel/Qwen3-Omni-30B-A3B-Instruct-int4-AutoRound \
  --port 8008
```

AutoRound quantization provides:
- **~50% memory reduction** compared to BF16 (from ~60GB to ~30GB)
- **~25% memory reduction** compared to FP8 (from ~40GB to ~30GB)
- **Accuracy at ultra-low bit widths**: maintains high accuracy even at 2–4 bits, requiring minimal tuning effort thanks to its sign-gradient descent optimization.

### Image and Text Input

Send an image with a text question to get both text and audio responses. Set `"modalities": ["text", "audio"]` to enable audio output.

**cURL**

```bash
curl -X POST http://localhost:8008/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-omni",
    "messages": [{"role": "user", "content": "How many cars are there in the picture?"}],
    "images": ["tests/data/cars.jpg"],
    "modalities": ["text", "audio"],
    "max_tokens": 16
  }'
```

**Python**

```python
import base64
import requests

resp = requests.post(
    "http://localhost:8008/v1/chat/completions",
    json={
        "model": "qwen3-omni",
        "messages": [{"role": "user", "content": "How many cars are there in the picture?"}],
        "images": ["tests/data/cars.jpg"],
        "modalities": ["text", "audio"],
        "max_tokens": 16,
    },
)
resp.raise_for_status()
result = resp.json()
choice = result["choices"][0]["message"]

print(choice["content"])

audio_data = base64.b64decode(choice["audio"]["data"])
with open("output.wav", "wb") as f:
    f.write(audio_data)
```

### Audio and Image Input

Send an audio file with an image. The model hears the spoken question and sees the image, then responds with both text and audio.

**cURL**

```bash
curl -X POST http://localhost:8008/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-omni",
    "messages": [{"role": "user", "content": ""}],
    "images": ["tests/data/cars.jpg"],
    "audios": ["tests/data/query_to_cars.wav"],
    "modalities": ["text", "audio"],
    "max_tokens": 16
  }'
```

**Python**

```python
import base64
import requests

resp = requests.post(
    "http://localhost:8008/v1/chat/completions",
    json={
        "model": "qwen3-omni",
        "messages": [{"role": "user", "content": ""}],
        "images": ["tests/data/cars.jpg"],
        "audios": ["tests/data/query_to_cars.wav"],
        "modalities": ["text", "audio"],
        "max_tokens": 16,
    },
)
resp.raise_for_status()
result = resp.json()
choice = result["choices"][0]["message"]

print(choice["content"])

audio_data = base64.b64decode(choice["audio"]["data"])
with open("output.wav", "wb") as f:
    f.write(audio_data)
```

### Video and Audio Input

Send a video with a spoken audio question. The model watches the video, hears the question, and responds with both text and audio.

**cURL**

```bash
curl -X POST http://localhost:8008/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-omni",
    "messages": [{"role": "user", "content": ""}],
    "videos": ["tests/data/draw.mp4"],
    "audios": ["tests/data/query_to_draw.wav"],
    "modalities": ["text", "audio"],
    "max_tokens": 16
  }'
```

**Python**

```python
import base64
import requests

resp = requests.post(
    "http://localhost:8008/v1/chat/completions",
    json={
        "model": "qwen3-omni",
        "messages": [{"role": "user", "content": ""}],
        "videos": ["tests/data/draw.mp4"],
        "audios": ["tests/data/query_to_draw.wav"],
        "modalities": ["text", "audio"],
        "max_tokens": 16,
    },
)
resp.raise_for_status()
result = resp.json()
choice = result["choices"][0]["message"]

print(choice["content"])

audio_data = base64.b64decode(choice["audio"]["data"])
with open("output.wav", "wb") as f:
    f.write(audio_data)
```

<a id="apple-silicon-mlx-and-torch-mps"></a>
## Apple Silicon (MLX and Torch MPS)

Use `mlx>=0.32.2` and `mlx-lm>=0.31.2` for the native MLX path; `mlx-vlm`
is not required. MLX 0.32.2 adds fused attention for the vision encoder's
72-dimensional heads. Existing environments can upgrade MLX with
`python -m pip install --upgrade 'mlx>=0.32.2'`.

Qwen3-Omni also runs on Apple Silicon (`arm64`) through the same backend
switch used by Qwen3-ASR: `SGLANG_USE_MLX=1` selects the all-native MLX
Apple path, and an unset (or falsy) `SGLANG_USE_MLX` selects the eager Torch
MPS Apple path. Both use the standard `sgl-omni serve` CLI (equivalently,
`python -m sglang_omni.cli serve`). See the
[Qwen3-Omni cookbook](../cookbook/qwen3_omni.md#apple-silicon-mlx-and-torch-mps)
for installation prerequisites and supported checkpoint layouts.

Set the pinned community checkpoint once:

```bash
export REPO="/path/to/sglang-omni"
export PY="$REPO/.venv-apple/bin/python"
export MODEL_DIR="$HOME/models/Qwen3-Omni-30B-A3B-Instruct-4bit-93b3cbdd"
export MODEL_REVISION="93b3cbddd65ed4babff8f22fba491cdba7a21778"
cd "$REPO"
```

```bash
"$PY" - <<'PY'
import os
from huggingface_hub import snapshot_download

print(
    snapshot_download(
        repo_id="mlx-community/Qwen3-Omni-30B-A3B-Instruct-4bit",
        revision=os.environ["MODEL_REVISION"],
        local_dir=os.path.expanduser(os.environ["MODEL_DIR"]),
    )
)
PY
```

Canonical native MLX launch (speech mode):

```bash
SGLANG_USE_MLX=1 "$PY" -m sglang_omni.cli serve \
  --model-path "$MODEL_DIR" \
  --port 8008
```

Text-only native MLX launch:

```bash
SGLANG_USE_MLX=1 "$PY" -m sglang_omni.cli serve \
  --model-path "$MODEL_DIR" \
  --text-only \
  --port 8008
```

Ownership for the Apple MLX path:

- native MLX: vision, audio, thinker, talker, code predictor, code2wav
- CPU: preprocessing, token decoding

Talker prefill also runs natively in MLX. In Torch MPS mode the model
components, including image/audio encoders and code2wav, remain on Torch MPS.

The Apple runtime profile is intentionally conservative and identical for
both backends:

- One Metal device (`tp_size=1`), `max_running_requests=1` — one request runs
  at a time.
- Greedy generation only; other sampling, penalty, and logprob combinations
  are rejected at request time.
- Eager execution — no radix cache, overlap, mixed prefill, chunked prefill,
  CUDA graphs, `torch.compile`, async decode lookahead, logprobs, or partial
  talker start.
- SHM inter-stage transport, the same as CUDA single-node deployments.
- Backend selection is strict: MLX never falls back to Torch MPS and vice
  versa, so a load failure on one backend is a real failure, not a silent
  fallback.
- Native MLX does not imply radix cache, multi-request batching, or
  CUDA-oriented optimizations.

Checkpoint layouts:

- **Dense Torch MPS** expects the dense, officially supported split Hugging Face
  checkpoint layout (thinker/talker/code2wav weights alongside the official
  processor and tokenizer assets), launched with `SGLANG_USE_MLX` unset.
- **Quantized Torch MPS** accepts symmetric HF INT4 checkpoints in
  `compressed-tensors/pack-quantized` or AutoRound `auto_round:auto_gptq`
  format, detected automatically from `config.json`.
  Loading and eager inference use Torch, not MLX or a CUDA quantization backend.
- **MLX** launches the downloaded pinned
  `mlx-community/Qwen3-Omni-30B-A3B-Instruct-4bit` checkpoint directory
  directly.
- Other MLX-compatible 4-bit layouts may also load when they satisfy the
  Apple checkpoint validator, but the pinned mlx-community checkpoint above is
  the tested and recommended deployment.
- Validator-accepted MLX examples include component-local thinker/talker
  shards and the root-namespaced MLX-VLM layout.

With both backend variables unset, the normal Torch MPS path selects dense
or supported INT4 loading from checkpoint metadata. The legacy
`SGLANG_QWEN3_OMNI_MPS_QUANTIZATION=int4` flag remains accepted but is unnecessary;
it does not quantize dense checkpoints. Unsupported formats fail during preflight.
The packed adapter repacks existing calibrated INT4 codes for Metal.
MLX-specific tensor names, convolution-layout conversion, and INT8
requantization are no longer part of the MPS loader.

Dense Torch MPS example:

```bash
env -u SGLANG_USE_MLX -u SGLANG_QWEN3_OMNI_MPS_QUANTIZATION \
  "$PY" -m sglang_omni.cli serve \
  --model-path /absolute/path/to/Qwen3-Omni-30B-A3B-Instruct \
  --port 8008
```

Native Torch MPS weight-only INT4 uses a **different checkpoint** from MLX.
The supported public exports have these pinned layouts:

| Export | Revision | Packing |
|---|---|---|
| `Intel/Qwen3-Omni-30B-A3B-Instruct-int4-AutoRound` | `c3ef7bbf0b9d866866136c1d08b6314d27f7e151` | Symmetric GPTQ-style INT4, group size 128 |
| `cyankiwi/Qwen3-Omni-30B-A3B-Instruct-AWQ-4bit` | `d6e1eff8d3414580a276744361d5b6d7d4798a56` | Symmetric compressed-tensors INT4, group size 32 |

Download a pinned export, then launch its local directory:

```bash
hf download Intel/Qwen3-Omni-30B-A3B-Instruct-int4-AutoRound \
  --revision c3ef7bbf0b9d866866136c1d08b6314d27f7e151 \
  --local-dir "$HOME/models/Qwen3-Omni-AutoRound"

env -u SGLANG_USE_MLX -u SGLANG_QWEN3_OMNI_MPS_QUANTIZATION \
  PYTORCH_ENABLE_MPS_FALLBACK=0 \
  "$PY" -m sglang_omni.cli serve \
  --model-path "$HOME/models/Qwen3-Omni-AutoRound" \
  --model-name qwen3-omni --host 127.0.0.1 --port 8008
```

These exports quantize the Thinker transformer, not the entire speech pipeline.
The Talker, code predictor, encoders and vocoder use the existing dense loaders,
not the quantized loader. Embeddings and routers also stay in their configured
floating dtype. Activations
and KV caches also remain floating point. Checkpoint file size and CUDA/XPU
memory figures are not guarantees of peak memory on a 48-GB Mac.

The loader reads one projection at a time, including each routed expert,
and preserves the existing scales and integer codes. Forward calls PyTorch's
native `aten._weight_int4pack_mm` MPS operator, with no dense weight cache.
It does not require AutoRound, AutoAWQ, GPTQModel, TorchAO, or custom Metal
kernels at runtime. The eager stage runners and serving profile are unchanged.

This is not a delegation to Transformers' AutoRound loader:
[AutoRound v0.13 registers MLX as its MPS backend](https://github.com/intel/auto-round/blob/v0.13.0/auto_round/inference/backend.py),
not native Torch packed INT4 execution. The narrow Thinker adapter remains
necessary for this pure-Torch path.

Only the specified symmetric W4A16 layouts are accepted: the label "AWQ" or
"GPTQ" alone does not imply compatibility. Asymmetric weights, activation-order
permutations, mixed quantization schemes, MLX affine checkpoints, and quantized
encoders/Talker/code2wav are rejected. The latter components must remain dense.
Do not combine the legacy INT4 flag with `SGLANG_USE_MLX=1`; run servers sequentially.

Run the focused native-MPS packing and loader regression tests:

```bash
"$PY" -m pytest -q \
  tests/unit_test/qwen3_omni/test_torch_mps_hf_quantization.py
```

These tests establish packing, native dispatch, routing, and loader behavior.
They do not establish full-model memory safety or multimodal semantic quality.

Both pinned exports above also completed eight real serving requests on a
48-GiB Apple Silicon Mac with PyTorch 2.13.0: text, full-resolution image,
audio, and video+audio inputs, each with text-only and text+speech outputs.
The runs left `SGLANG_QWEN3_OMNI_MPS_QUANTIZATION` unset, used
`PYTORCH_ENABLE_MPS_FALLBACK=0`, loaded 18,624 quantized Thinker linears per export,
and used the ordinary dense loaders for other components, one server at a time.
This establishes serving
behavior for that matrix, not comprehensive semantic quality or a peak-memory
guarantee for other inputs.

## Request Parameters

The table below lists all parameters accepted by the `/v1/chat/completions` endpoint for Qwen3-Omni.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model` | string | `null` | Model identifier |
| `messages` | list | (required) | List of chat messages, each with `role` and `content` |
| `modalities` | list | `["text"]` | Output modalities: `["text"]` for text only, `["text", "audio"]` for text and audio |
| `images` | list | `null` | List of image file paths (local paths or URLs) |
| `audios` | list | `null` | List of audio file paths (local paths or URLs) |
| `videos` | list | `null` | List of video file paths (local paths or URLs) |
| `max_tokens` | int | `null` | Maximum number of tokens to generate |
| `max_completion_tokens` | int | `null` | OpenAI-compatible alias for `max_tokens` |
| `temperature` | float | `null` | Sampling temperature |
| `top_p` | float | `null` | Top-p sampling |
| `top_k` | int | `null` | Top-k sampling |
| `repetition_penalty` | float | `null` | Repetition penalty |
| `seed` | int | `null` | Random seed for reproducibility |
| `stream` | bool | `false` | Enable streaming via SSE |
| `audio` | dict | `null` | Speech response format configuration, e.g. `{"format": "wav"}` |
| `stage_sampling` | dict | `null` | Per-stage sampling overrides, e.g. `{"thinker": {"temperature": 0.8}}` |
