# Qwen3-ASR

[Qwen3-ASR](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) is an audio transcription model served through the OpenAI-compatible `/v1/audio/transcriptions` endpoint. It accepts one uploaded audio file per request and returns text.

## Prerequisites

Install `sglang-omni` by following [Installation](../get_started/installation.md), then download the model:

```bash
hf download Qwen/Qwen3-ASR-1.7B
```

## Server Configuration

Qwen3-ASR runs a single ASR stage on one GPU.

```bash
sgl-omni serve \
  --model-path Qwen/Qwen3-ASR-1.7B \
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
| `file` | file | required | Audio file uploaded as multipart form data; maximum size is 25 MiB |
| `model` | string | server default | Model identifier |
| `language` | string | `en` | Language hint; `zh`/`cn` select Chinese, other values use English prompting |
| `response_format` | string | `json` | `json`, `verbose_json`, or `text` |
| `temperature` | float | `0.01` effective | Sampling temperature; `0` is converted to near-greedy `0.01` |
| `max_new_tokens` | integer | stage default | Maximum generated transcription tokens |

`verbose_json` is accepted, but currently returns the same minimal JSON shape as `json`:
`{"text": "..."}`.

## Benchmarking

Use `benchmarks/eval/benchmark_asr_seedtts.py` to sweep ASR concurrency on
SeedTTS reference audio through `/v1/audio/transcriptions`. It defaults to
`--model-path Qwen/Qwen3-ASR-1.7B`; the shared request and metric logic lives in
`benchmarks.tasks.asr` and also supports Fun-ASR through `--model-path`.

```bash
sgl-omni serve --model-path Qwen/Qwen3-ASR-1.7B --port 8000

# Sweep the full SeedTTS EN set (1088 clips) at 1..64 concurrency, 3 repeats:
python -m benchmarks.eval.benchmark_asr_seedtts \
  --port 8000 --concurrencies 1,2,4,8,16,32,64 --repeats 3 --warmup
```

The ASR CI gate runs Fun-ASR-Nano on this same benchmark entry point
(`tests/test_model/test_asr_ci_fun_asr.py`). Qwen3-ASR remains the
transcriber for the TTS and talker WER stages.

## Known Limitations

- The endpoint accepts one uploaded file per request.
- Audio duration is bounded by the configured context and requested
  `max_new_tokens`, rather than a fixed 30-second window. Split audio or reduce
  `max_new_tokens` if the request exceeds that token budget.
- `prompt` is accepted by the HTTP endpoint for OpenAI compatibility, but Qwen3-ASR currently ignores it.
- Audio is resampled to 16 kHz before transcription.
