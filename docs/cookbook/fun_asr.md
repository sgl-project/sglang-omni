# Fun-ASR-Nano

[Fun-ASR-Nano](https://arxiv.org/abs/2509.12508) is a multilingual audio
transcription model served
through the OpenAI-compatible `/v1/audio/transcriptions` endpoint. It accepts
one uploaded audio file per request and returns text.

Fun-ASR does not support `/v1/audio/translations`; that endpoint returns HTTP 400. Use `/v1/audio/transcriptions`.

## Prerequisites

Install `sglang-omni` by following [Installation](../get_started/installation.md),
then download the model:

```bash
# Use the -hf variant and pin the validated revision.
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

The consumer profile uses BF16, disables `torch.compile`, caps running
requests at 16, and reserves 65% of device memory for static allocations:

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

## Stream Transcription

Set the multipart `stream` field to `true` and keep `response_format` as
`json` or `text` to receive Server-Sent Events (SSE):

```bash
curl -N -X POST http://localhost:8000/v1/audio/transcriptions \
  -F model=FunAudioLLM/Fun-ASR-Nano-2512-hf \
  -F file=@tests/data/query_to_cars.wav \
  -F language=en \
  -F response_format=json \
  -F stream=true
```

The response contains zero or more `transcript.text.delta` events, followed
by one `transcript.text.done` event with the complete post-processed
transcript, then `data: [DONE]`. Streaming primarily reduces time to first
text; it does not change the final transcript.

## Request Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `file` | file | required | Audio file uploaded as multipart form data |
| `model` | string | server default | Model identifier |
| `language` | string | unset | Language hint. `en`/`english`/`英文` transcribe to English; `zh`/`cn`/`chinese`/`中文` (or unset) transcribes to Chinese; other values pass through as the target language |
| `response_format` | string | `json` | `json`, `verbose_json`, or `text` |
| `stream` | boolean | `false` | Emit SSE text deltas. Streaming accepts only `json` or `text` response formats |
| `temperature` | float | `0.0` | Sampling temperature; `0.0` (greedy) is the correct decoding mode for Fun-ASR-Nano and the default |
| `max_new_tokens` | integer | duration-based | Generation budget scaled to the audio duration. Explicit values must be between 1 and 200 |

## Benchmarking

SeedTTS EN/ZH concurrency/WER benchmarking for Fun-ASR-Nano lives in
`benchmarks/eval/benchmark_asr_seedtts.py`. Pass the Fun-ASR-Nano model
path with `--model-path`.

```bash
# Download the test set once:
python -m benchmarks.dataset.prepare --dataset seedtts

# Launch the RTX 4090 profile as shown above.
# Set this to the server's GPU worker PID reported by `nvidia-smi`.
SERVER_HOST_PID=12345

# Sweep the full SeedTTS EN set (1088 clips), 3 measured repeats:
python -m benchmarks.eval.benchmark_asr_seedtts \
  --model-path FunAudioLLM/Fun-ASR-Nano-2512-hf --port 8000 \
  --model-revision 854d88f94205cd17d2afdb24332130d86fbe654a \
  --dataset-revision 27f4c1adee83b5b29b7c4b375f6b976324bda308 \
  --lang en --concurrencies 1,2,4,8,16,32 --repeats 3 --warmup

# Run the 30-minute correctness, chaos, and memory-retention workload:
python -m benchmarks.eval.benchmark_asr_stability \
  --model-path FunAudioLLM/Fun-ASR-Nano-2512-hf --port 8000 \
  --model-revision 854d88f94205cd17d2afdb24332130d86fbe654a \
  --dataset-revision 27f4c1adee83b5b29b7c4b375f6b976324bda308 \
  --duration-s 1800 --concurrencies 1,4,8,16 \
  --request-timeout-s 60 --check-audio-boundary \
  --dtype bfloat16 --attention-backend flashinfer \
  --mm-attention-backend triton_attn --cuda-graph --no-torch-compile \
  --max-running-requests 16 --mem-fraction-static 0.65 \
  --gpu-process-pid "${SERVER_HOST_PID}" \
  --min-free-memory-mib 2048 --max-retained-memory-mib 256 \
  --output fun_asr_stability.json

# Quick smoke on a 20-sample subset:
python -m benchmarks.eval.benchmark_asr_seedtts \
  --model-path FunAudioLLM/Fun-ASR-Nano-2512-hf --port 8000 \
  --max-samples 20 --concurrencies 2 --repeats 1

# Measure text TTFT and inter-chunk latency through the SSE endpoint:
python -m benchmarks.eval.benchmark_asr_seedtts \
  --model-path FunAudioLLM/Fun-ASR-Nano-2512-hf --port 8000 \
  --max-samples 20 --concurrencies 2 --repeats 1 --stream
```

## Benchmark Results

### RTX 4090 validation on base `cd45a47a`

The PR stack based on `cd45a47a` was validated in the digest-pinned ASR CI
image on one RTX 4090 (24,564 MiB), Linux 6.8, driver 590.48.01, driver CUDA
13.1, PyTorch 2.11.0+cu130, SGLang 0.5.16, SGLang-Omni 0.1.2, and Transformers
5.12.1. The outer image was
`hongccc/sglang-omni@sha256:374d0b1c30b2bff685b1716fc64a02ad3b3d0a90fe2ce73ce9861a6992c28101`.
The model and SeedTTS revisions are the values shown above. Each row is the
mean of three measured repeats after one discarded warmup; every repeat
evaluated the full split with zero skips.

SeedTTS EN (1,088 clips):

| Concurrency | Requests/s | Mean latency (s) | p95 latency (s) | Audio s/s | Corpus WER |
|---:|---:|---:|---:|---:|---:|
| 1 | 15.59 | 0.064 | 0.079 | 73.86 | 0.0172 |
| 2 | 26.59 | 0.075 | 0.097 | 125.93 | 0.0172 |
| 4 | 44.68 | 0.089 | 0.122 | 211.61 | 0.0185 |
| 8 | 69.43 | 0.114 | 0.159 | 328.83 | 0.0180 |
| 16 | 95.81 | 0.166 | 0.240 | 453.78 | 0.0178 |
| 32 | 79.13 | 0.401 | 0.497 | 374.77 | 0.0177 |

SeedTTS ZH (2,020 clips; normalized error is character-spaced):

| Concurrency | Requests/s | Mean latency (s) | p95 latency (s) | Audio s/s | Corpus error rate |
|---:|---:|---:|---:|---:|---:|
| 1 | 17.59 | 0.057 | 0.070 | 82.35 | 0.0138 |
| 2 | 30.58 | 0.065 | 0.085 | 143.15 | 0.0133 |
| 4 | 52.82 | 0.075 | 0.096 | 247.31 | 0.0133 |
| 8 | 80.33 | 0.099 | 0.129 | 376.09 | 0.0134 |
| 16 | 113.33 | 0.141 | 0.199 | 530.58 | 0.0137 |
| 32 | 84.72 | 0.376 | 0.480 | 396.64 | 0.0137 |

The 30-minute staged workload completed 110,514 normal requests with zero
unexpected errors. All 28 cancel/reconnect and 32 malformed-audio events
passed, final `/health` returned HTTP 200, sampled free GPU memory stayed above
7,328 MiB, and retained device memory after cooldown was 224 MiB (limit:
256 MiB). The preceding four-second smoke passed every functional and health
check but retained 302 MiB; it was accepted only as the predeclared bounded
short-smoke warning, not as the formal memory result.

The fixed validation base was
`cd45a47a1838017c89fb2178f167aac0cd7412a3`, and every lane used clean
integrated candidate `e54422c2de28e72c17dd2b744908e69c1b7ffd20`. The final
14-patch integrated stack was then rebased to
`b3bbff6ea1f48d50c78a4c12d059af8705f6f4f0` with a 14/14 exact `=`
range-diff. The publication history has 15 commits because the final test
cleanup was split between its PR #2 and PR #4 owners; its tree is
byte-identical to the integrated stack. The two intervening upstream commits
change only Qwen3-TTS/Moss-TTS runtime and tests plus `mps_dp.md`, not the
Fun-ASR, Whisper, benchmark, or focused-test paths validated here.

The [issue #1170](https://github.com/sgl-project/sglang-omni/issues/1170)
validation record identifies the provider run as `vastai-48058271` and the
artifact set as `run-artifacts-e54422c2`. Its 233-entry remote `SHA256SUMS`
manifest has SHA-256
`de49284660acd6a312c4f540a90232ffd3757a928578d160f5d87d81441522c8`;
the collected 248-file tree also has a verified 247-entry local manifest. The
set contains exact commands, per-repeat output, dependency inventory, and
checksums. These clean pinned-base runs supersede the issue's older RTX 4090
numbers.

No Nsight Compute hardware-counter profile was run; the table reports
application metrics and NVML resource telemetry only.

### Historical H100 reference

Measured on a single H100 80 GB (bf16, DP=1, default server settings)
against the full SeedTTS sets. Each row is the mean of 3 runs with one
discarded warmup pass per level. RTF is processing time divided by audio
duration (lower is better). RTFx is successful input-audio seconds divided by
wall-clock seconds (higher is better).

SeedTTS EN (1088 clips, mean clip length 4.69 s). Corpus WER was 0.0171 at
every level through concurrency 32:

| Concurrency | Throughput (samples/s) | Mean latency (s) | p95 latency (s) | RTF mean | RTFx |
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

| Concurrency | Throughput (samples/s) | Mean latency (s) | p95 latency (s) | RTF mean | RTFx |
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
