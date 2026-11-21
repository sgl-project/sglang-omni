# Audio Translations

This guide uses [Whisper ASR](../cookbook/whisper_asr.md) as an example model
with the OpenAI-compatible `POST /v1/audio/translations` endpoint, which
translates source speech to English. The endpoint is served only by pipelines
that declare audio-translation capability; other ASR models return an explicit
HTTP 400 instead of silently transcribing.

## Prerequisites

Install `sglang-omni` by following [Installation](../get_started/installation.md),
then download a multilingual Whisper checkpoint:

```bash
hf download openai/whisper-large-v3
```

Translation requires a multilingual, non-turbo checkpoint: `*.en` checkpoints
have no translate task, and `whisper-large-v3-turbo` was distilled without it,
so serving those checkpoints returns transcriptions regardless of endpoint.

## Launch the Server

```bash
sgl-omni serve \
  --model-path openai/whisper-large-v3 \
  --port 8000
```

## Translate Audio

```bash
curl -X POST http://localhost:8000/v1/audio/translations \
  -F model=openai/whisper-large-v3 \
  -F file=@tests/data/query_to_cars.wav \
  -F response_format=json
```

Whisper detects the source language when `language` is omitted. This optional source-language hint is an SGLang-Omni extension to OpenAI's translations schema. Both APIs translate to English.

## Model Support

| Model | `/v1/audio/translations` |
|---|---|
| [Whisper ASR](../cookbook/whisper_asr.md) | Supported |
| [Hy-SimulMT](../cookbook/hy_simulmt.md) | Chinese to English |
| [Qwen3-ASR](../cookbook/qwen3_asr.md) | HTTP 400 |
| [Fun-ASR](../cookbook/fun_asr.md) | HTTP 400 |
| [ARK-ASR](../cookbook/arkasr.md) | HTTP 400 |
| [Nemotron 3.5 ASR](../cookbook/nemotron3_5_asr.md) | HTTP 400 |
| [MOSS-Transcribe-Diarize](../cookbook/moss_transcribe_diarize.md) | HTTP 400 |

See each model's cookbook for its transcription workflow.

## Response Formats

| `response_format` | Behavior |
|---|---|
| `json` | JSON object containing `text` |
| `verbose_json` | JSON with `task="translate"`, text, duration, and segments |
| `text` | Raw translated text with `text/plain` content type |
| `srt`, `vtt` | Whisper subtitles with model-derived segment timestamps |

For a Whisper clip with translated text, `verbose_json` returns a segment spanning the clip. Subtitle formats require a model adapter with real segment timestamps. An unsupported model returns HTTP 400 before inference.

Streaming is available with `stream=true` for `json` and `text`, using the same
SSE lifecycle as transcriptions: `transcript.text.delta` events, one
`transcript.text.done` event, then `data: [DONE]`.
