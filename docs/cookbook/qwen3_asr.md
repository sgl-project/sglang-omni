# Qwen3-ASR

[Qwen3-ASR](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) is an audio transcription model served through the OpenAI-compatible `/v1/audio/transcriptions` endpoint. It accepts one uploaded audio file per request and returns text.

Qwen3-ASR does not support `/v1/audio/translations`; that endpoint returns HTTP 400. Use `/v1/audio/transcriptions`.

## Prerequisites

Install `sglang-omni` by following [Installation](../get_started/installation.md), then download the model:

```bash
MODEL_REVISION=7278e1e70fe206f11671096ffdd38061171dd6e5
MODEL_PATH=$(hf download Qwen/Qwen3-ASR-1.7B --revision "${MODEL_REVISION}")
```

### Apple Silicon (MLX)

The Apple Silicon path requires macOS, Python 3.12, Homebrew, and SGLang's MLX
runtime. Audio decoding also requires Homebrew's versioned FFmpeg 7 formula:

```bash
brew install ffmpeg@7
export DYLD_LIBRARY_PATH="$(brew --prefix ffmpeg@7)/lib${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
```

Do not replace `ffmpeg@7` with the unversioned `ffmpeg` formula. The latter
currently installs FFmpeg 9, while Apple installs `torchcodec==0.11.1`, which
supports FFmpeg 4 through 8. Because `ffmpeg@7` is keg-only, its library
directory must also be present in `DYLD_LIBRARY_PATH` whenever the server starts.

macOS may remove `DYLD_*` variables when a SIP-protected system executable
launches the server. Set `DYLD_LIBRARY_PATH` on the final `sgl-omni` process;
for example, place `/usr/bin/env DYLD_LIBRARY_PATH=...` after wrappers such as
`/usr/bin/time`. Test a compressed input such as M4A or MP3, since WAV decoding
can succeed without loading FFmpeg.

Create one virtual environment for both repositories, then install the pinned
SGLang tag from source with its `all_mps` dependencies before installing
SGLang-Omni:

```bash
git clone --branch v0.5.18 https://github.com/sgl-project/sglang.git
git clone https://github.com/sgl-project/sglang-omni.git

uv venv -p 3.12 sglang-omni/.venv-apple
source sglang-omni/.venv-apple/bin/activate

cd sglang
cp python/pyproject_other.toml python/pyproject.toml
uv pip install -e "python[all_mps]"

cd ../sglang-omni
uv pip install -e .
```

This installs MLX through SGLang. It does not install or use the `mlx-audio`
package. Before downloading a model, verify both Metal and FFmpeg loading:

```bash
SGLANG_USE_MLX=1 python - <<'PY'
import mlx.core as mx
from torchcodec.decoders import AudioDecoder

assert mx.metal.is_available()
print("MLX Metal and TorchCodec FFmpeg loading are available")
PY
```

Use an MLX-converted Qwen3-ASR checkpoint and opt into the MLX runner:

```bash
export SGLANG_USE_MLX=1
export DYLD_LIBRARY_PATH="$(brew --prefix ffmpeg@7)/lib${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"

sgl-omni serve \
  --model-path mlx-community/Qwen3-ASR-0.6B-4bit \
  --model-name Qwen/Qwen3-ASR-0.6B \
  --asr.engine.max_running_requests 1 \
  --port 8000
```

The MLX path currently supports one device (`tp_size=1`) and greedy decoding.
Radix caching, chunked prefill, and CUDA graphs are not used by this path. The
HTTP and SSE transcription interfaces below are the same as on CUDA;
`stream=true` provides pseudo-streaming transcript deltas as tokens are decoded.
The Apple paths do not provide sampling penalties or token logprobs yet. MLX
can batch multiple requests, but `max_running_requests=1` is recommended when
single-request latency matters; increase it only when throughput is preferred.

To use the Torch MPS compatibility path instead, leave `SGLANG_USE_MLX` unset
and pass an official PyTorch Qwen3-ASR checkpoint. It currently uses one device,
greedy decoding, and the eager `torch_native`/`sdpa` profile:

```bash
unset SGLANG_USE_MLX
sgl-omni serve \
  --model-path Qwen/Qwen3-ASR-0.6B \
  --model-name Qwen/Qwen3-ASR-0.6B \
  --asr.engine.max_running_requests 1 \
  --port 8000
```

The initial Torch MPS profile is bounded to a 2,048-token KV budget, uses
`audio_chunking.max_audio_clip_s` for non-streaming chunks (30 seconds by
default), and caps native/whole-upload requests at 60 seconds. Values above
that qualified limit are rejected. Use the MLX path for long audio and the
larger native context limit.

## Server Configuration

Qwen3-ASR runs a single ASR stage on one GPU. Its default `auto` dtype follows
the checkpoint configuration (BF16 for Qwen3-ASR-1.7B); pass
`--asr.factory.dtype float16` to force FP16.
Async decode is enabled by default for all decode batch sizes, allowing the
shared one-step-lookahead path to overlap host-side result processing with the
next GPU decode forward even for a single request. Use
`--asr.factory.enable_async_decode false` to disable it, or tune the crossover
with `--asr.factory.async_decode_min_batch_size`.
The request builders also use the shared LM prefill-admission gate: prefill
starts when 16 built requests are ready or after the oldest ready request waits
40 ms. Once request-build work drains, a ready prefill is released immediately
if decode is idle; while decode is active, it continues coalescing until the
same request target or deadline.

```bash
sgl-omni serve \
  --model-path "${MODEL_PATH}" \
  --model-name Qwen/Qwen3-ASR-1.7B \
  --port 8000
```

For a single 24 GB RTX 4090 (SM89), use the checked-in consumer profile:

```bash
sgl-omni serve \
  --config examples/configs/qwen3_asr_rtx4090.yaml \
  --port 8000
```

This qualified profile keeps the model in BF16, limits the stage to 16 running
requests, and sets `mem_fraction_static` to `0.65`. Its bounds are specific to
the validated RTX 4090 layout; use the default configuration or a separately
qualified profile on other GPU architectures.

For example, force synchronous decode when comparing modes:

```bash
sgl-omni serve \
  --model-path Qwen/Qwen3-ASR-1.7B \
  --asr.factory.enable_async_decode false \
  --port 8000
```

## Transcribe Audio

```bash
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F model=Qwen/Qwen3-ASR-1.7B \
  -F file=@tests/data/query_to_cars.wav \
  -F response_format=json
```

```python
import requests

with open("tests/data/query_to_cars.wav", "rb") as f:
    resp = requests.post(
        "http://localhost:8000/v1/audio/transcriptions",
        data={
            "model": "Qwen/Qwen3-ASR-1.7B",
            "response_format": "json",
        },
        files={"file": ("query_to_cars.wav", f, "audio/wav")},
        timeout=300,
    )

resp.raise_for_status()
print(resp.json()["text"])
```

## Stream Transcription

Set `stream=true` to receive incremental transcript deltas over SSE. Use
`curl -N` to disable client-side response buffering:

```bash
curl -N -X POST http://localhost:8000/v1/audio/transcriptions \
  -F model=Qwen/Qwen3-ASR-1.7B \
  -F file=@tests/data/query_to_cars.wav \
  -F language=en \
  -F response_format=json \
  -F stream=true
```

The stream contains zero or more delta events, followed by the complete final
transcript and the SSE sentinel:

```text
data: {"type":"transcript.text.delta","delta":"..."}

data: {"type":"transcript.text.done","text":"..."}

data: [DONE]
```

Qwen3-ASR batches deltas for up to 50 ms by default. EOS and other terminal
conditions flush any buffered text before the final transcript event.

## Request Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `file` | file | required | Audio file uploaded as multipart form data |
| `model` | string | server default | Model identifier |
| `language` | string | none | Optional language hint as a supported code or canonical name (case-insensitive); omit it for automatic detection |
| `prompt` | string | none | Accepted for OpenAI compatibility; Qwen3-ASR currently ignores it |
| `response_format` | string | `json` | `json`, `verbose_json`, or `text` |
| `temperature` | float | `0` | Sampling temperature; `0` uses greedy decoding |
| `max_new_tokens` | integer | server stage limit | Per-request generation-token limit |
| `stream` | boolean | `false` | Return SSE transcript deltas; supports `json` or `text` response format |

`verbose_json` uses the model adapter's verbose response schema and includes
duration-based usage (rounded-up audio seconds) when duration probing succeeds.

### Language Hints

When `language` is omitted, Qwen3-ASR detects the spoken language before
transcribing. Set an explicit hint when the language is known or automatic
detection is unreliable for short or ambiguous audio.

Qwen3-ASR accepts the following 30 explicit language codes and their canonical names:

| Codes | Canonical names |
|---|---|
| `ar`, `yue`, `zh`, `cs`, `da`, `nl`, `en`, `fil`, `fi`, `fr` | Arabic, Cantonese, Chinese, Czech, Danish, Dutch, English, Filipino, Finnish, French |
| `de`, `el`, `hi`, `hu`, `id`, `it`, `ja`, `ko`, `mk`, `ms` | German, Greek, Hindi, Hungarian, Indonesian, Italian, Japanese, Korean, Macedonian, Malay |
| `fa`, `pl`, `pt`, `ro`, `ru`, `es`, `sv`, `th`, `tr`, `vi` | Persian, Polish, Portuguese, Romanian, Russian, Spanish, Swedish, Thai, Turkish, Vietnamese |

For example, `language=es` and `language=Spanish` both force the prompt suffix
`language Spanish<asr_text>`. The legacy `cn` and regional `zh-*` spellings are
also accepted as Chinese. Unsupported language hints return HTTP 400 instead of
silently falling back to English.

The model also has ASR coverage for 22 Chinese dialects, but those dialect names
are not supported as forced `language` hints; use `Chinese`/`zh` for them.

## Long Audio

The current Qwen3-ASR model accepts at most 1,200 seconds of audio in one
request, so we transcribe longer uploads in chunks: we split the audio, run
each chunk as its own engine request, and join the transcripts back in
order. The behavior follows two kinds of values.

The scheduling policy is yours to tune, with dotted flags or the matching
YAML keys:

| Name | Default | Meaning |
|---|---|---|
| `--audio_chunking.max_audio_clip_s` | `30` | Longest clip we send to the engine in one request, and therefore the chunk length. It sits well below the model's native 1,200s on purpose: shorter chunks batch better, and the output-token budget scales with clip length on its own. Capped at the native clip limit. |
| `--audio_chunking.max_concurrent_chunks` | `8` | How many chunks of one request run in the engine at once. A per-request cap so one long upload can't crowd out everyone else's requests. |
| `--audio_chunking.max_total_audio_s` | `3600` | Upper limit on the whole upload; you get HTTP 400 above it. This is a memory guard: we keep the decoded waveform in memory while its chunks run. |

The model properties are ClassVars on `Qwen3ASRPipelineConfig`; no
configuration path reaches them:

| Name | Value | Meaning |
|---|---|---|
| `allow_audio_chunking` | `true` | Qwen3-ASR transcribes an isolated chunk correctly, so chunking is on. |
| `max_native_clip_s` | `1200` | Longest clip the model takes as one request (its native limit). Streaming cannot chunk, so this is the streaming cutoff; the Torch MPS compatibility path resolves it to its qualified 60-second cap. |
| `min_tail_s` | `0.5` | Shortest final chunk worth transcribing; if the tail would be shorter, we move the previous cut earlier to absorb it. This matches the model's own minimum input length. |

Note: Raising `audio_chunking.max_audio_clip_s` also resizes the encoder CUDA-graph bucket
ladder, which is derived from the chunk length: a longer chunk means more and
larger captured graphs, and their static buffers stay resident for the life of
the server (roughly 6.6 KB per token of ladder ceiling; at 1,200s the ceiling
is 124,800 tokens). Budget for that when you raise the flag on small GPUs.

Behavior notes:

- **`verbose_json` returns one segment per chunk** with the chunk's real
  start/end timestamps -- chunk-level granularity, not word-level (Qwen3-ASR
  does not emit word timestamps).
- A few unusual audio formats may not expose a readable duration; we fall
  back to the non-chunked path for those uploads.
- Streamed responses (`stream=true`) do not support chunking yet; a stream
  request runs as one engine request. MLX and CUDA accept audio up to the
  model-native `max_native_clip_s` (1,200s), while Torch MPS accepts up to its
  qualified 60-second cap and returns HTTP 400 above that -- use `stream=false`
  for longer uploads.

## Benchmarking

Use `benchmarks/eval/benchmark_asr_seedtts.py` to sweep ASR concurrency on
SeedTTS reference audio through `/v1/audio/transcriptions`. It defaults to
`--model-path Qwen/Qwen3-ASR-1.7B`; the shared request and metric logic lives in
`benchmarks.tasks.asr` and also supports Fun-ASR through `--model-path`.
The report includes RTF (processing time divided by audio duration) and RTFx
(successful input-audio seconds divided by wall-clock seconds).

```bash
sgl-omni serve \
  --model-path "${MODEL_PATH}" \
  --model-name Qwen/Qwen3-ASR-1.7B \
  --port 8000

# Sweep the full SeedTTS EN set (1088 clips), 3 repeats per concurrency:
# Set SERVER_GPU_PID to the server process PID reported by nvidia-smi.
python -m benchmarks.eval.benchmark_asr_seedtts \
  --port 8000 \
  --gpu-process-pid "${SERVER_GPU_PID}" \
  --dataset-revision 27f4c1adee83b5b29b7c4b375f6b976324bda308 \
  --model-revision 7278e1e70fe206f11671096ffdd38061171dd6e5 \
  --concurrencies 1,2,4,8,16,32,64 \
  --repeats 3 --warmup
```

The result JSON includes the applied dataset revision, declared model revision,
an effective evaluation-input content hash, normalization, repository and
dependency fingerprints, complete sample counts, and latency/RTF/throughput.
When local NVML and `psutil` sampling are available, it also includes CPU use,
power, and peak/steady GPU memory. Pass each server GPU PID reported by NVML via
`--gpu-process-pid`; without explicit PIDs, process-specific metrics remain
unavailable rather than including unrelated workloads on the same GPU. In a
Docker container, use the host PID namespace (`--pid=host`) to collect process
CPU metrics. Unavailable metrics and monitor errors remain explicit. Optional
server settings and an exact launch command can be declared with the benchmark's
provenance flags.

The ASR CI gate runs the selected ASR CI model preset on this same benchmark
entry point (`tests/test_model/test_asr_ci_seedtts.py`). Qwen3-ASR remains
the transcriber for the TTS and talker WER stages.

For the current-main concurrency baseline, the fixed-baseline comparison, and
the per-stage bottleneck decomposition (issue #1324), see
[Qwen3-ASR concurrency profile](../developer_reference/qwen3_asr_concurrency_profile.md).
The benchmark's `--profile-events`, `--sample-util`, `--save-raw-dir`, and
`--fingerprint` flags capture the telemetry that report uses.

## Concurrency tuning

The request-build, admission, and CUDA-graph policy defaults come from a
measured sweep (issue #1324 Q-PR5): `request_build_max_workers` {2, 4, 8} ×
`request_build_max_pending` {16, 32, 64} × `max_running_requests` {16, 32, 64}
with matching CUDA-graph coverage, each configuration a full SeedTTS EN
concurrency sweep (1–64, three repeats plus warmup) on one 141 GB GPU with the
pre-LM encoder enabled and its embedding cache disabled (unique-input regime).
Requests/s by client concurrency:

| config (workers/pending/running) | c=8 | c=16 | c=32 | c=64 | shed at c=64 |
|---|---:|---:|---:|---:|---:|
| 2 / 16 / 32 | 39.1 | 47.5 | 52.3 | 51.0 | 704/3264 |
| 4 / 16 / 32 | 47.6 | 60.3 | 70.4 | 55.4 | 301/3264 |
| 8 / 16 / 32 | 48.5 | 75.6 | 89.7 | 64.6 | 173/3264 |
| 8 / 16 / 16 | 57.6 | 75.4 | 42.2 | 46.7 | 250/3264 |
| 8 / 32 / 32 | 57.7 | 76.5 | 87.1 | 65.1 | 0 |
| 8 / 64 / 32 | 55.2 | 76.6 | 87.9 | 64.7 | 0 |
| **8 / 32 / 64 (default)** | 57.4 | 77.0 | 90.2 | 96.8 | 0 |
| 8 / 64 / 64 | 57.0 | 74.3 | 88.8 | 100.3 | 0 |

Reading, and the resulting defaults:

- **Build workers scale monotonically to 8** at every concurrency ≥ 8 and cost
  nothing at concurrency 1 (0.099–0.101 s mean everywhere), so 8 is the
  default. Those workers do CPU request construction (decode audio,
  optional mel FFT) and submit encoder work asynchronously. When no extra
  builds are queued, the request builder waits for encode and returns a
  ready request like the sync path; when pending+backlog exceeds the
  worker pool, it returns a deferred admission so workers can pull the
  backlog. A cache hit still skips mel extraction entirely.
- **Pending 16 → 32 removes all concurrency-64 shedding** and lifts
  concurrency-8 throughput ~19 %; 64 adds nothing further. 32 is the default.
- **`max_running_requests` 16 collapses concurrency 32** (queue-bound) with no
  light-load latency benefit, so there is no latency-first case for lowering
  it. The default is 64 because it unlocks the concurrency-64 regime (+~50 %
  requests/s, zero shedding), at the price of larger CUDA-graph and KV memory.
  On memory-constrained GPUs, use the memory-conservative override:

```bash
sgl-omni serve --model-path Qwen/Qwen3-ASR-1.7B \
  --asr.engine.max_running_requests 32
```

- Corpus WER stayed 0.0122 for every configuration at every level.

## Known Limitations

- The endpoint accepts one uploaded file per request.
- Non-streaming uploads up to `max_total_audio_s` (default one hour) are
  transcribed in full via chunking; see Long Audio above. Streaming requests
  are limited to `max_native_clip_s` (1,200s) on MLX/CUDA; Torch MPS caps both
  native and whole-upload requests at 60 seconds.
- `prompt` is accepted by the HTTP endpoint for OpenAI compatibility, but Qwen3-ASR currently ignores it.
- Audio is resampled to 16 kHz before transcription.
