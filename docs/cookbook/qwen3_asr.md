# Qwen3-ASR

[Qwen3-ASR](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) is an audio transcription model served through the OpenAI-compatible `/v1/audio/transcriptions` endpoint. It accepts one uploaded audio file per request and returns text.

## Prerequisites

Install `sglang-omni` by following [Installation](../get_started/installation.md), then download the model:

```bash
MODEL_REVISION=7278e1e70fe206f11671096ffdd38061171dd6e5
MODEL_PATH=$(hf download Qwen/Qwen3-ASR-1.7B --revision "${MODEL_REVISION}")
```

## Server Configuration

Qwen3-ASR runs a single ASR stage on one GPU. Its default `auto` dtype follows
the checkpoint configuration (BF16 for Qwen3-ASR-1.7B); pass
`--stages.asr.factory-args.dtype float16` to force FP16.
Async decode is enabled by default for decode batches of at least two requests,
allowing the shared one-step-lookahead path to overlap host-side result
processing with the next GPU decode forward. Use `--decode-mode sync` to disable
it, or tune the crossover with `--async-lookahead-min-batch-size`.
The request builders also use the shared LM prefill-admission gate while more
request builds are pending: prefill starts when 16 built requests are ready,
after the oldest ready request waits 24 ms, or immediately when build work drains.

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
  --decode-mode sync \
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
| `stream` | boolean | `false` | Return transcript events over SSE |

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
  default (it is also what lets the pre-LM encoder form real batches).
- **Pending 16 → 32 removes all concurrency-64 shedding** and lifts
  concurrency-8 throughput ~19 %; 64 adds nothing further. 32 is the default.
- **`max_running_requests` 16 collapses concurrency 32** (queue-bound) with no
  light-load latency benefit, so there is no latency-first case for lowering
  it. The default is 64 because it unlocks the concurrency-64 regime (+~50 %
  requests/s, zero shedding), at the price of larger CUDA-graph and KV memory.
  On memory-constrained GPUs, use the memory-conservative override:

```bash
sgl-omni serve --model-path Qwen/Qwen3-ASR-1.7B \
  --max-running-requests 32
```

- Corpus WER stayed 0.0122 for every configuration at every level.

## Known Limitations

- The endpoint accepts one uploaded file per request.
- Audio duration is bounded by the configured context and requested
  `max_new_tokens`, rather than a fixed 30-second window. Split audio or reduce
  `max_new_tokens` if the request exceeds that token budget.
- `prompt` is accepted by the HTTP endpoint for OpenAI compatibility, but Qwen3-ASR currently ignores it.
- Audio is resampled to 16 kHz before transcription.
