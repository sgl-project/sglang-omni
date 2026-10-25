# Qwen3-TTS

[Qwen3-TTS Base](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base) clones voices from short reference clips and supports 10 languages. SGLang-Omni serves the 0.6B and 1.7B checkpoints through a `preprocessing → tts_engine → vocoder` pipeline, with 24 kHz streaming through the OpenAI-compatible `/v1/audio/speech` endpoint.

## Prerequisites

Install **sglang-omni** by following [Installation](../get_started/installation.md).

Qwen3-TTS Base depends on the upstream `qwen-tts` package. When installing, **you must use** `--no-deps` to avoid changing SGLang-Omni’s dependency versions:

```bash
apt-get update && apt-get install -y sox
uv pip install --no-deps sox einops
uv pip install --no-deps qwen-tts==0.1.1
```

Both lines require `--no-deps`:

- `qwen-tts` pins Transformers to 4.57.3, which would overwrite the project’s 5.19.0.
- A normal install of `sox` pulls a numpy version that exceeds the upper bound of `numba==0.65.1` (`numpy<=2.4`). The upgraded numpy causes librosa to fail, and `import qwen_tts` then raises: `Numba needs NumPy 2.4 or less`.

Do not append `onnxruntime` (it is already a dependency of SGLang-Omni and would likewise trigger the numpy version conflict).

Specifically, `qwen-tts` 0.1.1 pins Transformers 4.57.3, while its model code calls interfaces that were renamed or removed in Transformers 5.12. SGLang-Omni applies compatibility patches in `sglang_omni/models/qwen3_tts/compat.py`; every Qwen3-TTS entry point loads this patch before importing `qwen_tts`. The locked combination of Transformers 5.19 / SGLang 0.5.22 is the officially supported configuration.

> **Warning**: If a `TypeError` is raised from inside `qwen_tts`, do **not** install the Transformers version declared by that package—it will break the runtime. Please open an issue so the compatibility layer can be extended.

Download a checkpoint (both repositories are public, no token required):

```bash
hf download Qwen/Qwen3-TTS-12Hz-0.6B-Base
hf download Qwen/Qwen3-TTS-12Hz-1.7B-Base
```

## Server Configuration

Start the server with `sgl-omni serve`. Provide a model path and let `sglang-omni` build the pipeline config for you. CUDA graph capture can take several minutes at first startup.

```bash
# 0.6B
sgl-omni serve \
  --model-path Qwen/Qwen3-TTS-12Hz-0.6B-Base \
  --config examples/configs/qwen3_tts_0_6b.yaml \
  --port 8000
```

```bash
# 1.7B
sgl-omni serve \
  --model-path Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --config examples/configs/qwen3_tts_1_7b.yaml \
  --port 8000
```

### Deterministic Inference

For byte-identical PCM across batch sizes with the same prompt, reference audio, and seed, enable deterministic inference in either Base checkpoint's pipeline YAML (disabled by default):

```yaml
enable_deterministic_inference: true
```

This reduces throughput by serializing reference preprocessing and vocoder decoding and disabling the `initial_cuda_graph` and `followup_cuda_graph` capture paths.


### Overload / admission policy

Two SGLang generation-stage knobs control behavior past saturation:

| Knob | Meaning | Qwen3-TTS default |
|------|---------|-------------------|
| `--tts_engine.engine.max_running_requests` | Concurrent running slots | 64 |
| `--tts_engine.engine.max_queued_requests` | Waiting-queue depth before fast-reject | 64 |

Every request enters the waiting queue first, so `max_queued_requests` must be **≥ 1**. Capacity is roughly `running + queued`. Excess arrivals get HTTP **503** (The request queue is full) before preprocessing, or later if the AR waiting queue or request-build backlog is full. Qwen3-TTS defaults to 4 request-build workers with pending depth 16.

A finished request no longer counts toward capacity, but its streamed audio stays in the API process until the client reads it. `--stream_delivery.max_total_backlog_bytes` caps undelivered stream bytes across all requests per API process (default `1073741824` / 1 GiB; `none` disables it).

A client that reads at least as fast as audio is generated holds almost nothing. A slower client holds generated but unread audio; 24 kHz float32 is ~96 kB/s. This includes clients reading at playback speed while generation is faster than real time, once socket buffers fill.

Over the cap, streams with the longest-idle readers fail first until the total fits. A failed stream’s generation is aborted if still running. SSE (`stream_format=sse`) and WebSocket clients get an error event with **The request queue is full**; a raw PCM stream is closed early. New requests are not refused because of this cap. `/health` reports the backlog under `stream_delivery` (held bytes, open streams, failed streams).


### Server Readiness

Before reporting ready, the server sequentially runs one streaming warmup request per replica of the process with the most replicas. CustomVoice uses its first built-in speaker, VoiceDesign a short voice description, and Base a one-second reference tone.

`/health` returns HTTP 503 with status `starting` until warmup completes. Warmup loads kernels and initializes the speech path; single-request and batched samplers compile during model loading. A failed warmup stops the server.

Set `SGLANG_OMNI_WARMUP_TIMEOUT` to limit each warmup request (seconds; default `600`), or use `--skip-server-warmup` to skip warmup.

### Prefill CUDA graphs

Qwen3-TTS replays prefills as CUDA graphs. Base and CustomVoice use the full
backend, VoiceDesign the breakable backend.

| Knob | Meaning | Default |
|---|---|---|
| `--tts_engine.engine.cuda_graph_backend_prefill` | Prefill graph backend (`full`, `breakable` or `disabled`) | `full` on Base and CustomVoice, `breakable` on VoiceDesign |
| `--tts_engine.engine.cuda_graph_bs_prefill` | Prefill token counts to capture | a ladder from `1` up to the chunked prefill size |
| `--tts_engine.engine.cuda_graph_max_bs_prefill` | Cap for the ladder | the chunked prefill size |

`full` captures the whole prefill transformer, attention included, as one
graph per token count. `breakable` captures the layers in segments and runs
attention between them. `full` needs the `fa3`, `fa4` or `flashinfer` attention
backend. With another attention backend, Base and CustomVoice use `breakable`,
and an explicit `full` fails at startup.

On Hopper GPUs the full graph runs its prefill attention on `fa4` and holds every
running request, unless you set `--tts_engine.engine.attention_backend`,
`--tts_engine.engine.prefill_attention_backend` or an FP8
`--tts_engine.engine.kv_cache_dtype`. Decode keeps its own attention backend.

The ladder reaches the chunked prefill size, which the server picks from the
GPU's memory (8192 tokens on an 80 GB GPU), so prefills batched together under
load still replay. It starts at `1` because a prompt that is already cached
prefills a single token.

Opt out with `--tts_engine.engine.cuda_graph_backend_prefill disabled`. The
default adds a few seconds of graph capture at startup. Setting
`cuda_graph_max_bs_prefill` rebuilds the ladder up to the new cap, and declaring
`cuda_graph_bs_prefill` yourself keeps your list as is.

To change the ceiling, set `max_running_requests` and `max_queued_requests`
together:

```bash
sgl-omni serve \
  --model-path Qwen/Qwen3-TTS-12Hz-0.6B-Base \
  --config examples/configs/qwen3_tts_0_6b.yaml \
  --tts_engine.engine.max_running_requests 32 \
  --tts_engine.engine.max_queued_requests 16 \
  --port 8000
```

Stepped `--concurrencies` is a closed-loop client: it never holds more than
N in-flight requests, so past-ceiling load is a burst that drains. Keep
offered load above `max_running_requests + max_queued_requests` for a
duration with open-loop sustained overshoot:

```bash
python -m benchmarks.eval.benchmark_tts_seedtts \
  --generate-only --use-existing-server --stream \
  --model Qwen/Qwen3-TTS-12Hz-0.6B-Base \
  --port 8000 \
  --max-running-requests 32 \
  --max-queued-requests 16 \
  --sustained-overshoot \
  --overshoot-duration-s 10 \
  --max-samples 64
```

Arrivals default to `2 × capacity` (`--request-rate` overrides). Stats are
on successes only; artifacts land in `<output-dir>/overshoot/`.

A closed-loop `--concurrencies 16,32,48,64` sweep is still available for
comparing healthy vs past-ceiling points, but it does not hold overshoot. Each
concurrency writes inspectable artifacts under `<output-dir>/c<N>/`.

### Prefill Admission Coalescing

Prefill coalescing briefly delays admission to batch ready requests. It reduces scheduler overhead but can increase latency and leave fewer requests decoding. It is disabled by default.

Enable it with:

```bash
sgl-omni serve \
  --model-path Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --config examples/configs/qwen3_tts_1_7b.yaml \
  --tts_engine.factory.prefill_coalesce_requests 2 \
  --tts_engine.factory.prefill_coalesce_wait_ms 30 \
  --port 8000
```

Equivalent pipeline YAML:

```yaml
stages:
  tts_engine:
    factory:
      prefill_coalesce_requests: 2
      prefill_coalesce_wait_ms: 30.0
```

Coalescing takes effect when `prefill_coalesce_requests >= 2`. Admission resumes as soon as any condition is met:

- Decode is idle.
- The waiting queue reaches `prefill_coalesce_requests`.
- The oldest waiting request reaches `prefill_coalesce_wait_ms`.

Tune the request threshold and wait for your workload; `2` requests and `30` ms are examples. Leave coalescing disabled for latency-sensitive traffic or when the delay outweighs the batching benefit.

### Process Topology

All stages share a Python process by default. Reference preprocessing (speech-tokenizer encoding and speaker/prompt embeddings) competes with AR scheduling and vocoder decoding, which can limit throughput above roughly 32 concurrent requests per replica.

Separate preprocessing to reduce contention. The extra process loads a prompt-only frontend (embedding tables, text projection, predictor codec embeddings, and speaker encoder) and speech tokenizer, then sends prepared tensors to the engine. For 1.7B, this adds about 2.2 GB of GPU memory plus a CUDA context.

Declare a memory fraction for every GPU stage, and match `tts_engine.engine.mem_fraction_static` to `tts_engine.gpu_memory_fraction`:

```bash
sgl-omni serve \
  --model-path Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --preprocessing.process tts_frontend \
  --preprocessing.gpu 0 \
  --preprocessing.gpu_memory_fraction 0.05 \
  --tts_engine.gpu_memory_fraction 0.75 \
  --tts_engine.engine.mem_fraction_static 0.75 \
  --vocoder.gpu_memory_fraction 0.12
```

add `--vocoder.process vocoder`, set both engine fractions to `0.72`, and set the vocoder fraction to `0.15`. Six SeedTTS samples at a fixed seed produced identical PCM in both layouts;the extra process costs its CUDA context plus the frontend.
weights.

## Synthesizing Speech

### Text-only Requests

Base checkpoints require a reference clip. For text-only requests, use a CustomVoice or VoiceDesign checkpoint; launch commands are in [TTS Model Usage](../basic_usage/tts.md).

### Voice Cloning

Each `references` item contains `audio_path` (a server-local path or HTTP URL) and optional `text`. Include the clip's transcript for in-context learning (ICL) and better cloning quality; otherwise, the model uses speaker embeddings (x-vector mode).

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    "voice": "default",
    "input": "SGLang-Omni is a great project!",
    "references": [{
      "audio_path": "https://huggingface.co/datasets/zhaochenyang20/seed-tts-eval-mini/resolve/main/en/prompt-wavs/common_voice_en_10119832.wav",
      "text": "We asked over twenty different people, and they all said it was his."
    }]
  }' \
  --output output.wav
```

`ref_audio` and `ref_text` are shorthand for `references[0].audio_path` and `references[0].text`.

Python example:

```python
import requests

resp = requests.post(
    "http://localhost:8000/v1/audio/speech",
    json={
        "model": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
        "voice": "default",
        "input": "Get the trust fund to the bank early.",
        "references": [{
            "audio_path": "https://huggingface.co/datasets/zhaochenyang20/seed-tts-eval-mini/resolve/main/en/prompt-wavs/common_voice_en_10119832.wav",
            "text": "We asked over twenty different people, and they all said it was his.",
        }],
    },
)
resp.raise_for_status()
with open("output.wav", "wb") as f:
    f.write(resp.content)
```

Non-streaming responses include an `X-Finish-Reason` header:

- `stop`: generation reached the codec end-of-sequence token.
- `length`: generation reached `max_new_tokens`; the audio is decodable but may be incomplete.
- `unknown`: the model did not report a finish reason.

Batch responses expose the same value in each item's `finish_reason`.

**Leading silence in x-vector mode**

In x-vector mode, the engine masks silence codec IDs in the first two frames to reduce leading silence. At startup, it derives and logs these IDs by encoding stationary noise up to -50 dBFS with the checkpoint's codec. ICL, CustomVoice, and VoiceDesign are unaffected.

SeedTTS EN results (1088 clips, no reference text, seed 0):

| Checkpoint | Frames masked | Median onset | Onsets > 160 ms | WER | Speaker similarity |
|---|---|---|---|---|---|
| 1.7B Base | 0 | 495 ms | 95% | 0.854% | 61.00 |
| 1.7B Base | 2 (default) | 155 ms | 47% | 0.904% | 60.87 |
| 0.6B Base | 0 | 480 ms | 86% | 1.532% | 58.41 |
| 0.6B Base | 2 (default) | 75 ms | 23% | 1.072% | 58.10 |

Masking changes the generated output. To disable it:

```bash
sgl-omni serve \
  --model-path Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --config examples/configs/qwen3_tts_1_7b.yaml \
  --tts_engine.factory.leading_silence_mask_frames 0 \
  --port 8000
```

### Language Hint

**language** sets a target-language hint. The default, **auto**, lets the model detect the language. Supported languages are Chinese, English, Japanese, Korean, German, French, Russian, Portuguese, Spanish, and Italian.

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    "voice": "default",
    "input": "今天天气不错，就该出去晒晒太阳。",
    "references": [{
      "audio_path": "https://huggingface.co/datasets/zhaochenyang20/seed-tts-eval-mini/resolve/main/en/prompt-wavs/common_voice_en_10119832.wav",
      "text": "We asked over twenty different people, and they all said it was his."
    }],
    "language": "Chinese"
  }' \
  --output output.wav
```

### Streaming

Set `"stream": true` and `"response_format": "pcm"` for real-time raw PCM chunks:

```bash
curl -N -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
    "voice": "default",
    "input": "Get the trust fund to the bank early.",
    "references": [{
      "audio_path": "https://huggingface.co/datasets/zhaochenyang20/seed-tts-eval-mini/resolve/main/en/prompt-wavs/common_voice_en_10119832.wav",
      "text": "We asked over twenty different people, and they all said it was his."
    }],
    "stream": true,
    "response_format": "pcm"
  }' \
  --output output.pcm
```

Streaming returns `audio/pcm` 16-bit mono PCM with sample-rate metadata in the headers. See the [Higgs TTS cookbook](../cookbook/higgs_tts.md#streaming) for a full Python raw PCM consumer.

All three task types (Base/reference-cloning, CustomVoice, VoiceDesign) use true incremental codec and vocoder streaming on both this HTTP endpoint and `/v1/audio/speech/stream` WebSocket sessions with `stream_audio=true`. Pass `"stream_codec_output": false` per request, or launch with `--preprocessing.factory.stream_codec_output false`, to restore whole-utterance decoding.

Streamed CustomVoice on validated voice/language pairs (currently Ryan/English with default sampling) withholds the model’s silent bootstrap codec frame, removing ~80 ms of leading silence from the first chunk. The frame still feeds the vocoder, so later samples are unchanged; a runtime silence check emits the audio unmodified if the first frame is not silent. Opt out per request with `"suppress_bootstrap_silence": false` or per deployment with `--vocoder.factory.suppress_bootstrap_silence false`.

When `initial_codec_chunk_frames` is omitted, Qwen3-TTS ramps the first chunks `1 -> 2 -> 4` codec frames before the steady stride, so first audio leaves after a single AR step while the playback cushion rebuilds within four chunks. Pass an explicit value to trade continuity against time-to-first-audio.

Utterances that finish in fewer than the first chunk’s generated codec frames never reach the first chunk; their audio arrives complete in a single final flush.

**Codec decoding defaults**

Streaming uses the stateful incremental codec by default: each follow-up chunk decodes only its fresh frames against per-stream state in a preallocated arena, and follow-up workers collect for 4 ms. On NVIDIA, AMD and MUSA GPUs, steady-state cohorts also decode asynchronously on a priority stream and replay CUDA graphs whose decode step is `torch.compile`d (~1 min startup for the steady shapes). The left-context decoder remains available as a rollback:

```yaml
stages:
  vocoder:
    factory:
      enable_stateful_codec_decoder: false
```

`incremental_codec_cuda_graph`, `incremental_codec_compile`, `async_decode` and `followup_batch_wait_ms` are the individual switches. On one H100 80GB at 20 RPS (three client seeds, ~1200 requests each), the default path holds 0.6–2.3% of streams underrun vs 20.9% for the left-context decoder, with first playable audio at 55–58 ms vs 82–89 ms.

Each stream holds one of `codec_state_slots` (default 64, matching default `max_running_requests`) while decoding; streams past the slots decode eagerly on the left-context path. At startup the vocoder logs the captured decode graphs and warns if slots are fewer than the streams it may hold at once: the engine’s `max_running_requests` when the vocoder shares its process, or that times the engine replicas when the vocoder runs in its own process (requests bind round-robin). Raise `--vocoder.factory.codec_state_slots` together with `max_running_requests`.

Intel XPU, CPU and Apple MPS decode synchronously by default. For NPU defaults and configuration, see [Ascend NPU baseline](#ascend-npu-baseline). To opt in on Intel GPUs, follow the [Qwen3-TTS XPU recipe](../get_started/installation_xpu.md#qwen3-tts-text-to-speech-single-xpu).

Each follow-up batch sorts the stateful decoder’s cohorts (and rows within each) by playback deadline: the nearest cohort launches first and its nearest rows fill the first graph-sized group. Left-context decodes keep queue order. Follow-up workers also commit each in-flight cohort as soon as its decode finishes, checked at the top of the worker loop and before each launch, instead of at the next drain.

**First-audio chunk ramp**

For latency-sensitive deployments the early chunk schedule can be configured server-side with `stream_chunk_ramp` on the vocoder stage: entry `i` sizes streaming decode chunk `i + 1` in codec frames; past the ramp the steady stride takes over, so `[2, 4, 8]` yields `2 -> 4 -> 8 -> 8 -> ...`. Set it in a pipeline config:

```yaml
config_cls: Qwen3TTSPipelineConfig
model_path: Qwen/Qwen3-TTS-12Hz-0.6B-Base
stages:
  vocoder:
    factory:
      stream_chunk_ramp: [2, 4, 8]
```

```bash
python -m sglang_omni.cli serve --config qwen3_tts_ramp.yaml
```

Smaller early chunks lower time-to-first-audio but start with less buffered audio, so continuity cost grows with concurrency. Default `[1, 2, 4]` is the most aggressive and suits low concurrency; prefer `[2, 4, 8]` at moderate concurrency and `[4, 8]` for saturated serving. The ramp is mutually exclusive with the legacy `initial_chunk_frames` / `stream_initial_followup_stride` options; its first entry must not exceed the steady stride; a per-request `initial_codec_chunk_frames` still overrides only the first chunk.

## Generation Parameters

| Parameter | Default | Notes |
|---|---|---|
| `model` | Served model | Model identifier |
| `input` | Required | Text to synthesize |
| `voice` | `default` | Voice identifier; Base uses the reference clip for speaker conditioning |
| `references` | `null` | Reference clips for cloning; each item has `audio_path` and optional `text`. Required for Base unless `ref_audio` is supplied |
| `ref_audio` / `ref_text` | `null` | Shorthand for `references[0].audio_path` / `references[0].text` |
| `language` | `auto` | Target-language hint; see [Language Hint](#language-hint) |
| `temperature` | `0.9` | Sampling temperature |
| `top_p` | `1.0` | Top-p sampling |
| `top_k` | `50` | Top-k sampling |
| `repetition_penalty` | `1.05` | Repetition penalty |
| `max_new_tokens` | `2048` | Maximum generated codec tokens |
| `seed` | `null` | Random seed; use [deterministic inference](#deterministic-inference) for identical PCM across batch sizes |
| `stream` | `false` | Stream audio; use `response_format="pcm"` for raw PCM |
| `initial_codec_chunk_frames` | First chunk of the configured ramp (default `1 → 2 → 4`) | First chunk size in codec frames; `0` starts at the steady stride. See [chunk ramp](#first-audio-chunk-ramp) |
| `stream_codec_output` | `true` | Forward codec frames as they are generated; `false` decodes complete utterances |
| `suppress_bootstrap_silence` | `true` | Omit the silent bootstrap frame from streamed CustomVoice audio on validated voice/language pairs, subject to a runtime silence check |

## Model Variants

| Checkpoint | Parameters | Config |
|---|---|---|
| `Qwen/Qwen3-TTS-12Hz-0.6B-Base` | 0.6B | `examples/configs/qwen3_tts_0_6b.yaml` |
| `Qwen/Qwen3-TTS-12Hz-1.7B-Base` | 1.7B | `examples/configs/qwen3_tts_1_7b.yaml` |

Both use the same API. Choose 0.6B for lower memory use and latency, or 1.7B for typically better quality.

## CustomVoice Checkpoints

CustomVoice generates speech with built-in speakers through the same pipeline. Use it without reference audio; omit `ref_audio`, `ref_text`, `references`, and `x_vector_only_mode`. Omit `task_type` or set it to `CustomVoice`.


| Checkpoint | Config | instructions support |
|---|---|---|
| `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice` | `examples/configs/qwen3_tts_0_6b_customvoice.yaml` | Accepted for compatibility; unreliable style control |
| `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` | `examples/configs/qwen3_tts_1_7b_customvoice.yaml` | Supported |

Both checkpoints provide `Serena`, `Vivian`, `Uncle_Fu`, `Ryan`, `Aiden`, `Ono_Anna`, `Sohee`, `Eric`, and `Dylan`. Speaker matching is case-insensitive; an omitted or `default` voice selects `Vivian`. `GET /v1/audio/voices` lists `default` and available speakers. Unknown speakers or cloning fields return HTTP 400; uploaded voices are unsupported.

Both sizes support non-streaming responses, batch requests, HTTP PCM streaming, and WebSocket audio; see [Streaming](#streaming). WebSocket sessions use `stream_audio=true` and `response_format="pcm"`.

For Eric/Dylan, `language: Auto` selects Sichuan/Beijing dialect tokens; `language: Chinese` uses the Chinese token. The upstream `qwen-tts` 0.1.1 wrapper selects dialect tokens for both. Token selection does not guarantee an accent change.

Start 1.7B CustomVoice:

```bash
sgl-omni serve \
  --model-path Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  --config examples/configs/qwen3_tts_1_7b_customvoice.yaml \
  --port 8000
```

Then select a speaker:

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    "input": "SGLang-Omni serves Qwen CustomVoice.",
    "voice": "Ryan",
    "language": "English",
    "instructions": "Speak clearly and calmly."
  }' \
  --output custom-voice.wav
```

For 0.6B, use its model and configuration from the table, update the request's `model`, and omit `instructions`.

## Benchmark Results

### 0.6B Base

Qwen3-TTS-12Hz-0.6B-Base on Seed-TTS EN: 1088 utterances, voice cloning from each reference prompt, concurrency 16, and WER scored with HF Whisper-large-v3. Hardware: 1× H200 SXM.

| Metric | Value |
|---|---|
| WER (corpus, excluding runaway outliers) | 1.07% |
| WER (per-sample median / p95) | 0.00% / 9.09% |
| WER (corpus micro-average, all samples) | 18.29% |
| Runaway samples (>50% WER) | 2 / 1088 (0.2%) |
| Latency mean / median (s) | 6.61 / 6.24 |
| RTF mean / median | 1.51 / 1.48 |
| Output throughput (tok/s) | 115.4 |
| Completed / failed requests | 1088 / 0 |

Two utterances generated about 164 seconds of repeated audio before reaching `max_new_tokens`, accounting for the high unfiltered WER. Real-time factor (RTF) reflects concurrency 16, not single-stream latency.

### 1.7B CustomVoice

Qwen3-TTS-12Hz-1.7B-CustomVoice on the full Seed-TTS-Eval EN and ZH splits, concurrency 16, with 16 warmup requests per language/mode and `max_new_tokens=2048`. EN used Ryan/English; ZH used Vivian/Chinese. Neither used reference audio or instructions. WER/CER was scored with Qwen3-ASR-1.7B at concurrency 32. Hardware: 1× H200 141 GB, BF16, TP1. Sampling overrides and seed were unset.

Server settings:

```text
--tts_engine.engine.max_running_requests 64
--tts_engine.engine.cuda_graph_max_bs 64
--tts_engine.engine.torch_compile_max_bs 64
--vocoder.process vocoder
--tts_engine.gpu_memory_fraction 0.85
--vocoder.gpu_memory_fraction 0.10
```

`torch.compile` was disabled. Streaming used the default `1 → 2 → 4` chunk ramp with no request-level override. Each language/mode ran once on the same warmed server, in this order: non-streaming EN, non-streaming ZH, streaming EN, streaming ZH. No external process used the target GPU during timed runs; host CPU, memory, and I/O were shared with another profiling task.

| Metric | Non-streaming EN | Non-streaming ZH | Streaming EN | Streaming ZH |
|---|---:|---:|---:|---:|
| Samples | 1088 | 2020 | 1088 | 2020 |
| Corpus WER/CER | 1.608% | 0.984% | 2.085% | 0.927% |
| Corpus WER/CER (excluding >50% outliers) | 1.359% | 0.984% | 1.454% | 0.927% |
| Samples above 50% WER/CER | 3 | 0 | 4 | 0 |
| UTMOS | 4.1723 | 3.1824 | 4.1500 | 3.1789 |
| QPS | 14.788 | 13.307 | 10.098 | 8.333 |
| Latency mean (s) | 1.075 | 1.198 | 1.573 | 1.915 |
| RTF mean | 0.2335 | 0.2079 | 0.3380 | 0.3332 |
| TTFA mean (s) | N/A | N/A | 0.1213 | 0.1047 |

Corpus WER/CER divides total edit distance by EN reference words/ZH characters. The filtered row excludes samples above 50% WER/CER and recalculates the rate. UTMOS is a mean predicted audio-quality score, not a listening-test result.

Streaming and non-streaming were sampled independently, so quality scores are not a paired comparison. Base also uses different conditioning and a different ASR evaluator.

TTFA measures the first PCM payload's arrival, which may precede audible speech. The mean first-payload duration was 80 ms in both languages. Among all 3,108 streaming requests, 98.99% EN and 93.76% ZH had no playback underrun longer than 50 ms. Maximum underruns were 396.1 ms EN and 2072.0 ms ZH.

## Known Limitations

- **Reference audio recommended.** Without a reference clip, Qwen3-TTS Base tends to sound robotic.
- **Transcript improves cloning.** Providing `text` in `references` (ICL mode) gives better speaker similarity than speaker-embedding-only (x-vector) mode.
- **Language detection.** `language: auto` can misdetect short or code-switched inputs; set `language` explicitly when known.
- **Rare runaway generation.** ~0.2% of utterances (mainly 0.6B) may loop and run to `max_new_tokens`. Raise `repetition_penalty` (default `1.05`) or lower `max_new_tokens`; 1.7B is less affected.


## Reference
- [API Server Quickstart](https://github.com/sgl-project/sglang-omni/blob/main/docs/get_started/apiserver_quickstart.md)
