# Nemotron 3.5 ASR

[Nemotron 3.5 ASR Streaming 0.6B](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b)
is a multilingual RNN-T speech-recognition model served through the
OpenAI-compatible `/v1/audio/transcriptions` endpoint. SGLang-Omni resamples
each uploaded file to mono 16 kHz audio and batches compatible requests into
one model `generate()` call.

This integration accepts complete uploaded audio files. Client-driven
incremental PCM ingestion and persistent cross-chunk model caches are outside
the scope of this offline integration.

Nemotron 3.5 ASR does not support `/v1/audio/translations`; use
`/v1/audio/transcriptions`.

## Prerequisites

Install `sglang-omni` by following [Installation](../get_started/installation.md),
then download the model:

```bash
hf download nvidia/nemotron-3.5-asr-streaming-0.6b
```

## Server Configuration

Nemotron 3.5 ASR runs as one model-owned ASR stage on one GPU. The validated
default dtype is `float32`. The scheduler admits up to eight compatible
requests to one model batch and waits for at most 2 ms to form that batch.

```bash
sgl-omni serve \
  --model-path nvidia/nemotron-3.5-asr-streaming-0.6b \
  --port 8000
```

The checkpoint supports lookahead values `0`, `3`, `6`, and `13`; the default
is `3`. Configure lookahead and batching on the ASR stage when needed:

```bash
sgl-omni serve \
  --model-path nvidia/nemotron-3.5-asr-streaming-0.6b \
  --asr.factory.num_lookahead_tokens 3 \
  --asr.factory.max_batch_size 8 \
  --asr.factory.max_batch_wait_ms 2 \
  --port 8000
```

Requests with different explicit `max_new_tokens` values are placed in
separate model batches so each request keeps its requested output limit.

## Transcribe Audio

```bash
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F model=nvidia/nemotron-3.5-asr-streaming-0.6b \
  -F file=@tests/data/query_to_cars.wav \
  -F language=auto \
  -F response_format=verbose_json
```

```python
import requests

with open("tests/data/query_to_cars.wav", "rb") as audio_file:
    response = requests.post(
        "http://localhost:8000/v1/audio/transcriptions",
        data={
            "model": "nvidia/nemotron-3.5-asr-streaming-0.6b",
            "language": "auto",
            "response_format": "verbose_json",
        },
        files={"file": ("query_to_cars.wav", audio_file, "audio/wav")},
        timeout=300,
    )

response.raise_for_status()
print(response.json())
```

## Request Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `file` | file | required | Audio file uploaded as multipart form data |
| `model` | string | server default | Model identifier |
| `language` | string | `auto` | Checkpoint-defined locale or language code, matched case-insensitively; `auto` enables language detection |
| `response_format` | string | `json` | `json`, `verbose_json`, or `text` |
| `temperature` | float | `0` | Nemotron uses greedy RNN-T decoding and rejects non-zero values |
| `max_new_tokens` | integer | model default | Optional positive output-token limit |
| `prompt` | string | unset | Text prompts are not supported; non-empty values are rejected |

With `language=auto`, `verbose_json.language` is populated when the model emits
one unambiguous locale tag. Locale tags are removed from the returned clean
transcript. If the output contains multiple different locale tags, the API
does not invent a single language for the response.

## Batching and Limitations

- Audio is prepared through the shared mono 16 kHz transcription path.
- The processor's checkpoint-provided prompt dictionary is authoritative;
  unsupported language values fail before model inference.
- The model's generation path owns mutable encoder and decoder state, so model
  calls are serialized while each admitted scheduler batch is executed as one
  true batched `generate()` call.
- This integration does not expose live incremental audio ingestion. Upload the
  complete audio file in each transcription request.
