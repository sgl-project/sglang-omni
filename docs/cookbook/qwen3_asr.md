# Qwen3-ASR

[Qwen3-ASR](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) is an audio transcription model served through the OpenAI-compatible `/v1/audio/transcriptions` endpoint. It accepts one uploaded audio file per request and returns text.

## Prerequisites

Install `sglang-omni` by following [Installation](../get_started/installation.md), then download the model:

```bash
hf download Qwen/Qwen3-ASR-1.7B
```

## Server Configuration

Qwen3-ASR runs a single ASR stage on one GPU.
Async decode is enabled by default for decode batches of at least two requests,
allowing the shared one-step-lookahead path to overlap host-side result
processing with the next GPU decode forward. Use `--decode-mode sync` to disable
it, or tune the crossover with `--async-lookahead-min-batch-size`.

```bash
sgl-omni serve \
  --model-path Qwen/Qwen3-ASR-1.7B \
  --port 8000
```

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
  -F language=en \
  -F response_format=json
```

```python
import requests

with open("tests/data/query_to_cars.wav", "rb") as f:
    resp = requests.post(
        "http://localhost:8000/v1/audio/transcriptions",
        data={
            "model": "Qwen/Qwen3-ASR-1.7B",
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
| `language` | string | `en` | Language hint; `zh`/`cn` select Chinese, other values use English prompting |
| `response_format` | string | `json` | `json`, `verbose_json`, or `text` |
| `temperature` | float | `0.01` effective | Sampling temperature; `0` is converted to near-greedy `0.01` |

`verbose_json` is accepted, but currently returns the same minimal JSON shape as `json`:
`{"text": "..."}`.

`max_new_tokens` can be supplied as a multipart form field. The route uses the
Qwen3-ASR stage default of 4,096 when it is omitted.

## Long Audio

Qwen3-ASR accepts up to 1,200 seconds of audio in one native model request.
This is a model-owned limit rather than Whisper's 30-second feature-extractor
window. Audio at or below the limit is therefore processed without serving-layer
chunking.

Longer uploads are decoded once, split at low-energy boundaries into
non-overlapping chunks of at most 1,200 seconds, transcribed in order, and
stitched into one response. This applies to blocking and streaming
transcriptions. Chunk failures fail the whole upload instead of returning a
transcript with a silent gap.

Qwen3-ASR does not currently produce timestamped segments, so chunk offsets are
not exposed in `verbose_json`.

## Benchmarking

Use `benchmarks/eval/benchmark_asr_seedtts.py` to sweep ASR concurrency on
SeedTTS reference audio through `/v1/audio/transcriptions`. It defaults to
`--model-path Qwen/Qwen3-ASR-1.7B`; the shared request and metric logic lives in
`benchmarks.tasks.asr` and also supports Fun-ASR through `--model-path`.
The report includes RTF (processing time divided by audio duration) and RTFx
(successful input-audio seconds divided by wall-clock seconds).

```bash
sgl-omni serve --model-path Qwen/Qwen3-ASR-1.7B --port 8000

# Sweep the full SeedTTS EN set (1088 clips) at 1..64 concurrency, 3 repeats:
python -m benchmarks.eval.benchmark_asr_seedtts \
  --port 8000 --concurrencies 1,2,4,8,16,32,64 --repeats 3 --warmup
```

The ASR CI gate runs the selected ASR CI model preset on this same benchmark
entry point (`tests/test_model/test_asr_ci_seedtts.py`). Qwen3-ASR remains
the transcriber for the TTS and talker WER stages.

For the current-main concurrency baseline, the fixed-baseline comparison, and
the per-stage bottleneck decomposition (issue #1324), see
[Qwen3-ASR concurrency profile](../developer_reference/qwen3_asr_concurrency_profile.md).
The benchmark's `--profile-events`, `--sample-util`, `--save-raw-dir`, and
`--fingerprint` flags capture the telemetry that report uses.

## Known Limitations

- The endpoint accepts one uploaded file per request.
- `prompt` is accepted by the HTTP endpoint for OpenAI compatibility, but Qwen3-ASR currently ignores it.
- Audio is resampled to 16 kHz before transcription.
