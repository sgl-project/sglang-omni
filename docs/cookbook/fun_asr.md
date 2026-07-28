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
MODEL_REVISION=854d88f94205cd17d2afdb24332130d86fbe654a
MODEL_PATH=$(hf download FunAudioLLM/Fun-ASR-Nano-2512-hf \
  --revision "${MODEL_REVISION}")
```

## Server Configuration

Fun-ASR-Nano runs a single ASR stage on one GPU.

```bash
sgl-omni serve \
  --model-path FunAudioLLM/Fun-ASR-Nano-2512-hf \
  --port 8000
```

### RTX 4090 (24 GB)

The consumer profile uses BF16, FlashInfer language-model attention, Triton
multimodal attention, CUDA Graph batches through 16,
`max_running_requests=16`, `mem_fraction_static=0.65`, and no
`torch.compile`.

```bash
CUDA_VISIBLE_DEVICES=0 sgl-omni serve \
  --config examples/configs/fun_asr_rtx4090.yaml \
  --model-path "${MODEL_PATH}" \
  --model-name FunAudioLLM/Fun-ASR-Nano-2512-hf \
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

# Launch the RTX 4090 profile:
sgl-omni serve --config examples/configs/fun_asr_rtx4090.yaml --port 8000

# Sweep the full SeedTTS EN set (1088 clips), 3 measured repeats:
python -m benchmarks.eval.benchmark_asr_seedtts \
  --model-path FunAudioLLM/Fun-ASR-Nano-2512-hf --port 8000 \
  --model-revision 854d88f94205cd17d2afdb24332130d86fbe654a \
  --dataset-revision 27f4c1adee83b5b29b7c4b375f6b976324bda308 \
  --concurrencies 1,2,4,8,16,32 --repeats 3 --warmup

# Quick smoke on a 20-sample subset:
python -m benchmarks.eval.benchmark_asr_seedtts \
  --model-path FunAudioLLM/Fun-ASR-Nano-2512-hf --port 8000 \
  --max-samples 20 --concurrencies 2 --repeats 1
```

## Benchmark Results

### RTX 4090

Measured on Linux 6.8 with one RTX 4090 24,564 MiB (SM89), driver 580.126.20,
CUDA 13.0, PyTorch 2.11.0, SGLang 0.5.12.post1, and Transformers 5.6.0.
Each level completed the full pinned dataset with zero skips.

SeedTTS EN (1088 clips), mean of three measured repeats:

| Concurrency | Requests/s | Mean latency (s) | p95 latency (s) | Mean RTF | Audio s/s | Worst corpus WER |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 20.16 | 0.050 | 0.065 | 0.0107 | 95.46 | 0.0171 |
| 2 | 34.68 | 0.058 | 0.073 | 0.0125 | 164.24 | 0.0171 |
| 4 | 55.09 | 0.072 | 0.096 | 0.0157 | 260.90 | 0.0188 |
| 8 | 85.37 | 0.093 | 0.130 | 0.0203 | 404.30 | 0.0178 |
| 16 | 114.26 | 0.139 | 0.192 | 0.0302 | 541.14 | 0.0175 |
| 32 | 111.75 | 0.284 | 0.350 | 0.0619 | 529.25 | 0.0178 |

SeedTTS ZH (2020 clips), mean of three measured repeats:

| Concurrency | Requests/s | Mean latency (s) | p95 latency (s) | Mean RTF | Audio s/s | Worst corpus CER |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 14.36 | 0.071 | 0.102 | 0.0153 | 67.21 | 0.0168 |
| 2 | 36.95 | 0.054 | 0.068 | 0.0117 | 173.00 | 0.0162 |
| 4 | 58.50 | 0.068 | 0.091 | 0.0147 | 273.88 | 0.0162 |
| 8 | 93.33 | 0.085 | 0.123 | 0.0185 | 436.95 | 0.0164 |
| 16 | 127.25 | 0.125 | 0.182 | 0.0271 | 595.75 | 0.0164 |
| 32 | 130.64 | 0.243 | 0.305 | 0.0526 | 611.63 | 0.0167 |

The 30-minute mixed workload completed 105,067 requests with zero unexpected
errors. All 28 malformed-input and 28 cancel/reconnect events passed. Sampled
free GPU memory never fell below 7,104 MiB, cooldown retained 104 MiB, and
`/health` remained available. The same run confirmed 29.9-second English audio
succeeds and 30.1-second audio returns HTTP 400.

EN remains within the existing H100 CI tolerance, but RTX 4090 ZH CER
(`0.0148`–`0.0168`) is above the H100 raw reference (`0.0135`). This is a
measured cross-architecture quality difference rather than an H100 parity
claim.

### H100 reference

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
- Nsight Compute kernel counters were unavailable on the validation host due to
  `ERR_NVGPUCTRPERM`; NVML and scheduler-level resource metrics were collected.
