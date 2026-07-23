# Fun-ASR-Nano

[Fun-ASR-Nano](https://arxiv.org/abs/2509.12508) is a multilingual audio
transcription model served
through the OpenAI-compatible `/v1/audio/transcriptions` endpoint. It accepts
one uploaded audio file per request and returns text.

## Prerequisites

Install `sglang-omni` by following [Installation](../get_started/installation.md),
then download the model:

```bash
# Use the -hf variant
hf download FunAudioLLM/Fun-ASR-Nano-2512-hf
```

## Server Configuration

Fun-ASR-Nano runs a single ASR stage on one GPU.

```bash
sgl-omni serve \
  --model-path FunAudioLLM/Fun-ASR-Nano-2512-hf \
  --port 8000
```

## Transcribe Audio

```bash
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F model=FunAudioLLM/Fun-ASR-Nano-2512-hf \
  -F file=@tests/data/query_to_cars.wav \
  -F language=en \
  -F response_format=json
```

```python
import requests

with open("tests/data/query_to_cars.wav", "rb") as f:
    resp = requests.post(
        "http://localhost:8000/v1/audio/transcriptions",
        data={
            "model": "FunAudioLLM/Fun-ASR-Nano-2512-hf",
            "language": "en",
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
| `language` | string | unset | Language hint. `en`/`english`/`英文` transcribe to English; `zh`/`cn`/`chinese`/`中文` (or unset) transcribes to Chinese; other values pass through as the target language |
| `response_format` | string | `json` | `json`, `verbose_json`, or `text` |
| `temperature` | float | `0.0` | Sampling temperature; `0.0` (greedy) is the correct decoding mode for Fun-ASR-Nano and the default |
| `max_new_tokens` | integer | duration-based | Generation budget scaled to the audio duration. Explicit values must be between 1 and 200 |

## Benchmarking

SeedTTS EN/ZH concurrency/WER benchmarking for Fun-ASR-Nano lives in
`benchmarks/eval/benchmark_asr_seedtts.py`. Pass the Fun-ASR-Nano model
path with `--model-path`.

```bash
# Download the test set once:
python -m benchmarks.dataset.prepare --dataset seedtts

# Launch Fun-ASR-Nano:
sgl-omni serve --model-path FunAudioLLM/Fun-ASR-Nano-2512-hf --port 8000

# Sweep the full SeedTTS EN set (1088 clips) at 1..64 concurrency, 3 repeats:
python -m benchmarks.eval.benchmark_asr_seedtts \
  --model-path FunAudioLLM/Fun-ASR-Nano-2512-hf --port 8000 \
  --concurrencies 1,2,4,8,16,32,64 --repeats 3

# Quick smoke on a 20-sample subset:
python -m benchmarks.eval.benchmark_asr_seedtts \
  --model-path FunAudioLLM/Fun-ASR-Nano-2512-hf --port 8000 \
  --max-samples 20 --concurrencies 2 --repeats 1
```

## Benchmark Results

Measured on a single H100 80 GB (bf16, DP=1, default server settings)
against the full SeedTTS sets. Each row is the mean of 3 runs with one
discarded warmup pass per level. RTF is processing time divided by audio
duration (lower is better). audio_s/s is seconds of audio transcribed per
wall-clock second.

SeedTTS EN (1088 clips, mean clip length 4.69 s). Corpus WER was 0.0171 at
every level through concurrency 32:

| Concurrency | Throughput (samples/s) | Mean latency (s) | p95 latency (s) | RTF mean | audio_s/s |
|---:|---:|---:|---:|---:|---:|
| 1 | 26.44 | 0.038 | 0.047 | 0.0082 | 124 |
| 2 | 42.55 | 0.047 | 0.058 | 0.0102 | 200 |
| 4 | 62.35 | 0.064 | 0.088 | 0.0139 | 293 |
| 8 | 90.24 | 0.088 | 0.121 | 0.0192 | 423 |
| 16 | 127.46 | 0.125 | 0.167 | 0.0270 | 598 |
| 32 | 127.44 | 0.249 | 0.334 | 0.0539 | 598 |
| 64 | 137.98 | 0.453 | 0.542 | 0.0988 | 647 |

SeedTTS ZH (2020 clips, mean clip length 4.68 s). Corpus WER, effectively
character level after normalization, was 0.0135 at every level through
concurrency 32:

| Concurrency | Throughput (samples/s) | Mean latency (s) | p95 latency (s) | RTF mean | audio_s/s |
|---:|---:|---:|---:|---:|---:|
| 1 | 26.96 | 0.037 | 0.048 | 0.0080 | 126 |
| 2 | 45.97 | 0.043 | 0.056 | 0.0094 | 215 |
| 4 | 58.28 | 0.069 | 0.093 | 0.0148 | 273 |
| 8 | 79.76 | 0.100 | 0.138 | 0.0216 | 373 |
| 16 | 138.23 | 0.116 | 0.160 | 0.0249 | 647 |
| 32 | 167.42 | 0.190 | 0.264 | 0.0410 | 784 |
| 64 | 165.75 | 0.381 | 0.475 | 0.0825 | 776 |

At concurrency 64 a single worker rejects roughly 2 to 5 percent of
requests with HTTP 500 by design, because the request-build backlog admits
at most 16 pending builds per worker. Qwen3-ASR shows the same shedding
behavior at this level. For higher client concurrency, serve behind the
DP=2 managed router, matching the ASR CI topology.

## Known Limitations

- The endpoint accepts one uploaded file per request.
- Each uploaded audio segment must be 30 seconds or shorter, matching the
  official Fun-ASR VAD segment limit. Split longer recordings before upload.
- `itn` and `hotwords` are supported by the model request builder but not
  exposed as form fields on the public transcription endpoint.
- `prompt` is accepted by the HTTP endpoint for OpenAI compatibility, but
  Fun-ASR-Nano currently ignores it (use `hotwords` inside the builder for
  context biasing instead).
- Audio is resampled to 16 kHz before transcription.
- bf16 is strongly recommended; fp16 can overflow to NaN in the adaptor path.
