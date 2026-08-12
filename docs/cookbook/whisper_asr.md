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

## Benchmarking

Use the shared SeedTTS benchmark for end-to-end concurrency, WER, latency, and throughput:

```bash
python -m benchmarks.eval.benchmark_asr_seedtts \
  --port 8000 --model-path openai/whisper-base \
  --max-samples 20 --concurrencies 1,2,4,8 \
  --repeats 3 --warmup --output whisper_concurrency.json
```

## Benchmark Results

End-to-end results used the 20-sample SeedTTS EN subset on a single H200 with `openai/whisper-base` in FP16. Each mode ran one discarded warmup and three measured repeats per concurrency.

| Concurrency | Eager req/s | CUDA Graph req/s | Throughput gain | Eager mean latency (s) | CUDA Graph mean latency (s) | Corpus WER |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 19.57 | 20.29 | 3.7% | 0.051 | 0.049 | 0.0415 |
| 2 | 28.41 | 30.87 | 8.7% | 0.070 | 0.065 | 0.0415 |
| 4 | 37.90 | 41.70 | 10.0% | 0.104 | 0.094 | 0.0415 |
| 8 | 42.10 | 49.00 | 16.4% | 0.185 | 0.158 | 0.0415 |

All 480 measured requests completed successfully. Corpus WER was unchanged across eager and CUDA Graph modes at every concurrency.

## Known Limitations

- This path is experimental and not yet correctness-validated. Prefer Qwen3-ASR
  for validated ASR serving.
- Encoder CUDA Graph is opt-in and requires SGLang generation CUDA Graph to be
  enabled. Validate the selected buckets before production use.
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
