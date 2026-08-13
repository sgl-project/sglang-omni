# Whisper ASR

Whisper ASR checkpoints can be started through the OpenAI-compatible `/v1/audio/transcriptions` endpoint, but this path is experimental in the current SGLang-Omni tree. Prefer [Qwen3-ASR](qwen3_asr.md) for validated ASR serving.

## Prerequisites

Install `sglang-omni` by following [Installation](../get_started/installation.md), then download a Whisper checkpoint:

```bash
hf download openai/whisper-large-v3
```

## Server Configuration

Whisper ASR runs a single ASR stage on one GPU.

```bash
sgl-omni serve \
  --model-path openai/whisper-large-v3 \
  --port 8000
```

## Encoder CUDA Graph

The encoder CUDA Graph is enabled by default for the pipeline. The final bucket set is resolved from the serving prefill budget and the checkpoint's encoder prefix length; with the default 4,096-token budget and 1,500-token Whisper encoder prefix, only batches 1 and 2 are captured. To use eager encoder execution, override the pipeline configuration:

```yaml
config_cls: WhisperASRPipelineConfig
name: whisper
model_path: openai/whisper-large-v3-turbo

runtime_overrides:
  asr:
    enable_encoder_cuda_graph: false
```

The graph is captured after SGLang's generation graphs. Raise `max_prefill_tokens` before configuring larger buckets. Each request uses the smallest captured bucket that fits its batch. Requests larger than every captured bucket, with a different feature shape, or without a successful capture run eagerly. Startup and first-replay logs identify the captured and executed buckets.

## Prefill Coalescing

Whisper builds up to two requests concurrently and coalesces prefill at the serving-reachable batch size of two. A partial batch waits for at most 6 ms only while another request build is pending; a single request and a partial batch with no remaining build work are released immediately. This allows concurrent traffic to replay the encoder batch-2 graph without adding a fixed wait to the idle c=1 path.

`request_build_max_pending` bounds submitted request-build futures, not the request backlog. When `max_queued_requests` is unset, requests beyond that pending-build limit remain queued for later construction. Setting `max_queued_requests` retains the configured finite-queue rejection behavior.

Use `prefill_coalesce_requests` and `prefill_coalesce_wait_ms` to tune the gate. Set `prefill_coalesce_requests: 0` to disable only coalescing, or also set `request_build_max_workers: 1` to restore the pre-optimization request-build path:

```yaml
runtime_overrides:
  asr:
    request_build_max_workers: 1
    prefill_coalesce_requests: 0
```

## Transcribe Audio

```bash
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F model=openai/whisper-large-v3 \
  -F file=@tests/data/query_to_cars.wav \
  -F response_format=json
```

```python
import requests

with open("tests/data/query_to_cars.wav", "rb") as f:
    resp = requests.post(
        "http://localhost:8000/v1/audio/transcriptions",
        data={
            "model": "openai/whisper-large-v3",
            "response_format": "json",
        },
        files={"file": ("query_to_cars.wav", f, "audio/wav")},
        timeout=300,
    )

resp.raise_for_status()
print(resp.json()["text"])
```

## Request Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `file` | file | required | Audio file uploaded as multipart form data |
| `model` | string | server default | Model identifier |
| `language` | string | unset | Optional language hint |
| `prompt` | string | unset | Optional text used as Whisper prev-context conditioning |
| `response_format` | string | `json` | Use `json` for the current Whisper path |
| `temperature` | float | `0.0` | Sampling temperature; defaults to greedy decoding |

The request builder also supports `task` (`transcribe` by default) and
`max_new_tokens`, but the public transcription endpoint currently exposes only
the fields above. The route uses the ASR stage default unless the pipeline is
configured another way. For smoke tests, keep the request minimal and use
`response_format=json`.

## Long Audio

Whisper reads at most 30 seconds of audio in one request: the feature extractor works on a fixed 30-second mel window and drops everything past it.
In SGLang-Omni, we transcribe longer uploads in chunks by splitting the audio at the quietest point near each 30-second boundary,
running each chunk as its own engine request, and joining the transcripts back in order. The behavior follows these values, which Whisper
declares in code (`WhisperASRPipelineConfig.audio_chunking`). They are fixed model defaults in this release:

| Name | Value | Meaning                                                                                                                                                                     |
|---|---|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `max_audio_clip_s` | `30` | Longest clip we send to the engine in one request, and therefore the chunk length. Unlike Qwen3-ASR this is not a scheduling choice: 30s is the hard edge of the model's mel window. |
| `max_native_clip_s` | `30` | Same as the chunk length. Streaming cannot chunk, so `stream=true` takes audio up to 30s and gets HTTP 400 above that.                                                      |
| `max_total_audio_s` | `3600` | Upper limit on the whole upload; you get HTTP 400 above it. This is a memory guard: we keep the decoded waveform in memory while its chunks run.                            |
| `max_concurrent_chunks` | `8` | How many chunks of one request run in the engine at once. A per-request cap so one long upload can't crowd out everyone else's requests.                                    |
| `min_tail_s` | `1` | Shortest final chunk worth transcribing; if the tail would be shorter, we move the previous cut earlier to absorb it, which keeps Whisper from hallucinating on very short clips.      |

## Benchmarking

Use the shared SeedTTS benchmark for end-to-end concurrency, WER, latency, and throughput:

```bash
python -m benchmarks.eval.benchmark_asr_seedtts \
  --port 8000 --model-path openai/whisper-base \
  --max-samples 20 --concurrencies 1,2,4,8 \
  --repeats 5 --warmup --output whisper_concurrency.json
```

## Benchmark Results

The following W-PR1 results used the 20-sample SeedTTS EN subset on a single H200 with `openai/whisper-base` in FP16. Each mode ran one discarded warmup and three measured repeats per concurrency.

| Concurrency | Eager req/s | CUDA Graph req/s | Throughput gain | Eager mean latency (s) | CUDA Graph mean latency (s) | Corpus WER |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 19.57 | 20.29 | 3.7% | 0.051 | 0.049 | 0.0415 |
| 2 | 28.41 | 30.87 | 8.7% | 0.070 | 0.065 | 0.0415 |
| 4 | 37.90 | 41.70 | 10.0% | 0.104 | 0.094 | 0.0415 |
| 8 | 42.10 | 49.00 | 16.4% | 0.185 | 0.158 | 0.0415 |

All 480 W-PR1 measured requests completed successfully. Corpus WER was unchanged across eager and CUDA Graph modes at every concurrency.

The following W-PR2 results were measured separately on the same H200 and 20-sample subset with five measured repeats plus one discarded warmup per concurrency. The baseline used one request-build worker with coalescing disabled; the attribution run used two workers with coalescing disabled; the optimized run used two workers, a batch target of two, and a pending-build-aware 6 ms deadline.

| Concurrency | Baseline req/s | Two workers req/s | Coalesced req/s | Total gain | Gate gain | Baseline latency (s) | Coalesced latency (s) | Corpus WER |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 21.04 | 22.51 | 22.46 | 6.8% | -0.3% | 0.047 | 0.044 | 0.0415 |
| 2 | 30.45 | 36.68 | 41.96 | 37.8% | 14.4% | 0.066 | 0.047 | 0.0415 |
| 4 | 40.24 | 55.62 | 62.83 | 56.2% | 13.0% | 0.097 | 0.063 | 0.0415 |
| 8 | 48.03 | 75.93 | 82.15 | 71.0% | 8.2% | 0.161 | 0.092 | 0.0415 |

All 1,200 measured requests completed successfully. Corpus WER remained 0.0415 in all three modes and at every concurrency. Logs from the optimized run showed `Replaying Whisper encoder CUDA graph batch=2 request_batch=2` and prefill batches with two sequences and 3,008 new tokens.

## Known Limitations

- This path is experimental and not yet correctness-validated. Prefer Qwen3-ASR
  for validated ASR serving.
- Encoder CUDA Graph is enabled by default and requires SGLang generation CUDA
  Graph to remain enabled. Validate the selected buckets before production use.
- Chunked prefill is disabled because the Whisper encoder prefix must be
  admitted atomically. Requests that exceed the current prefill budget wait
  for the next batch instead of splitting the encoder prefix.
- Use `response_format=json`; other response formats are not validated for this
  experimental path.
- First startup can take several minutes.
- The endpoint accepts one uploaded file per request.
- Audio is resampled to 16 kHz before transcription.
- `prompt` conditions decoding via Whisper prev-context tokens. Only the last
  223 prompt tokens are kept (224 prev-context tokens including
  `<|startofprev|>`) — fewer when `max_new_tokens` is large, since prompt,
  task prefix, and output share Whisper's 448-token decoder context.
  `max_new_tokens` is likewise clamped to that context. The prompt must not
  contain Whisper special tokens.
