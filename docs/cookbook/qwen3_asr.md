# Qwen3-ASR

[Qwen3-ASR](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) is an audio transcription model served through the OpenAI-compatible `/v1/audio/transcriptions` endpoint. It accepts one uploaded audio file per request and returns text.

## Prerequisites

Install `sglang-omni` by following [Installation](../get_started/installation.md), then download the model:

```bash
MODEL_REVISION=7278e1e70fe206f11671096ffdd38061171dd6e5
MODEL_PATH=$(hf download Qwen/Qwen3-ASR-1.7B --revision "${MODEL_REVISION}")
```

## Server Configuration

Qwen3-ASR runs a single ASR stage on one GPU.

```bash
sgl-omni serve \
  --model-path "${MODEL_PATH}" \
  --model-name Qwen/Qwen3-ASR-1.7B \
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
| `prompt` | string | none | Accepted for OpenAI compatibility; Qwen3-ASR currently ignores it |
| `response_format` | string | `json` | `json`, `verbose_json`, or `text` |
| `temperature` | float | `0.01` effective | Sampling temperature; `0` is converted to near-greedy `0.01` |
| `max_new_tokens` | integer | server stage limit | Per-request generation-token limit |
| `stream` | boolean | `false` | Return transcript events over SSE |

`verbose_json` uses the model adapter's verbose response schema and includes
duration-based usage (rounded-up audio seconds) when duration probing succeeds.

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

## Known Limitations

- The endpoint accepts one uploaded file per request.
- `prompt` is accepted by the HTTP endpoint for OpenAI compatibility, but Qwen3-ASR currently ignores it.
- Audio is resampled to 16 kHz before transcription.
