# Nemotron 3.5 ASR

[Nemotron 3.5 ASR Streaming 0.6B](https://huggingface.co/nvidia/nemotron-3.5-asr-streaming-0.6b)
is a multilingual speech-recognition model with a FastConformer encoder and
an RNN-T decoder. SGLang-Omni supports complete-file transcription through
`/v1/audio/transcriptions` and native cache-aware PCM streaming in the internal
pipeline runtime.

## Prerequisites

Follow [Installation](../get_started/installation.md), then run the examples
from the repository root. Use the repository's pinned Transformers version;
the Nemotron compatibility implementation is included.

## Server Configuration

The default pipeline runs one ASR stage on one GPU in `float32`:

```bash
sgl-omni serve \
  --model-path nvidia/nemotron-3.5-asr-streaming-0.6b \
  --port 8000
```

Tune the ASR stage with `--asr.factory.*` flags:

| Option | Default | Description |
|---|---|---|
| `dtype` | `float32` | Model dtype |
| `num_lookahead_tokens` | `3` | Encoder right context; the checkpoint supports `0`, `3`, `6`, and `13` |
| `max_batch_size` | `8` | Maximum number of requests in a scheduler batch |
| `max_batch_wait_ms` | `2.0` | Maximum wait to form a batch, in milliseconds |
| `max_pending_stream_messages` | `256` | Pending input-message limit for the internal streaming runtime |

For example:

```bash
sgl-omni serve \
  --model-path nvidia/nemotron-3.5-asr-streaming-0.6b \
  --asr.factory.num_lookahead_tokens 3 \
  --asr.factory.max_batch_size 8 \
  --asr.factory.max_batch_wait_ms 2 \
  --port 8000
```

Within each scheduler batch, complete-file requests with the same
`max_new_tokens` value share one model `generate()` call. Different token limits
are processed in separate batches to preserve each request's output limit.

## Transcribe Audio

Upload a complete audio file; the server converts it to mono 16 kHz audio:

```bash
curl http://localhost:8000/v1/audio/transcriptions \
  -F model=nvidia/nemotron-3.5-asr-streaming-0.6b \
  -F file=@tests/data/query_to_cars.wav \
  -F language=auto \
  -F response_format=verbose_json
```

Use `response_format=text` for the transcript alone, or `json` for a JSON
response. `verbose_json` also includes duration and language information.
Locale tags are removed from the transcript. With `language=auto`, the
response reports a language when the model emits one unambiguous locale tag.

## Request Parameters

| Parameter | Default | Description |
|---|---|---|
| `file` | required | Audio file uploaded as multipart form data |
| `model` | server default | Model identifier |
| `language` | `auto` | Checkpoint-defined locale or language code, matched case-insensitively; `auto` enables language detection |
| `response_format` | `json` | `json`, `verbose_json`, or `text`; streaming responses accept only `json` or `text` |
| `temperature` | `0` | Greedy RNN-T decoding only; non-zero values are rejected |
| `max_new_tokens` | model default | Optional positive output-token limit |
| `prompt` | unset | Text prompts are unsupported; non-empty values are rejected |
| `stream` | `false` | Stream the HTTP response after a complete file upload; see below |

Supported language values come from the checkpoint's prompt dictionary.
Unsupported values fail before model inference. Nemotron supports transcription
only; `/v1/audio/translations` returns HTTP 400. Segment timestamps, SRT/VTT
output, and speaker diarization are not supported.

## Native Streaming

The internal streaming runtime accepts mono PCM16 at 16 kHz. It forms
lookahead-dependent audio windows and retains each request's attention,
convolution-padding, and RNN-T decoder caches across chunks. Compatible chunks
from different requests can share a model batch. End-of-input flushes the final
partial window; completion, cancellation, or failure releases the request state.

This PCM input path is an internal runtime capability. A public Nemotron
WebSocket/session API, VAD controls, and stable input-event schema are outside
the current integration. Setting `stream=true` on `/v1/audio/transcriptions`
streams the response to a complete uploaded file; it does not open a persistent
PCM input session or select the native cache-aware input path.
