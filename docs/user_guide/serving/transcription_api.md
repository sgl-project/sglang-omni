# Transcription API

`POST /v1/audio/transcriptions` accepts one multipart audio upload and returns a
transcript. Model cookbooks define language coverage, long-audio behavior,
timestamp granularity, and any ignored compatibility fields.

## Send a request

```bash
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F model=Qwen/Qwen3-ASR-1.7B \
  -F file=@tests/data/query_to_cars.wav \
  -F response_format=json
```

The server rejects empty uploads before they consume model resources.

## Request fields

| Field | Type | Default | Behavior |
|---|---|---|---|
| `file` | file | Required | One multipart audio upload |
| `model` | string | Served model | Model identifier |
| `language` | string | None | Optional model-specific language hint |
| `prompt` | string | None | Compatibility prompt; model support varies |
| `response_format` | string | `json` | `json`, `verbose_json`, `text`, `srt`, or `vtt`; subtitle formats require model-provided segment timestamps |
| `temperature` | float | Model default | Sampling temperature |
| `max_new_tokens` | integer | Stage limit | Positive per-request generation limit |
| `stream` | boolean | `false` | Return transcript events over SSE |

Do not infer language-hint or prompt support from the shared schema. Unsupported
hints return a model-specific error or have the behavior documented in the
cookbook.

## Response formats

| Format | Content type | Shape |
|---|---|---|
| `json` | JSON | `text` plus duration usage when available |
| `verbose_json` | JSON | Task, language, duration, text, segments, and usage |
| `text` | Plain text | Transcript body only |
| `srt` | Plain text | SubRip subtitles built from model-provided timestamped segments |
| `vtt` | Plain text | WebVTT subtitles built from model-provided timestamped segments |

Duration usage is rounded up to whole audio seconds when the server can probe
the input duration. Segment contents are produced by the active model adapter.
They may represent model timestamps, server-side long-audio chunks, or one
whole-file segment; consult the cookbook before treating them as word-level
timestamps.

`srt` and `vtt` are available only when the active model adapter exposes
segment timestamps. The server returns HTTP 400 for other models. These formats
are non-streaming only and are not synthesized from server-side long-audio
chunk boundaries.

## Streaming transcription

Set `stream=true` with `response_format=json` or `text`. The server emits
`transcript.text.delta` SSE events and finishes with a
`transcript.text.done` event containing the complete text and duration usage
when available. `verbose_json` is not a streaming response format.

See [Streaming](../advanced_features/streaming.md) for the event contract and
client guidance. A model may impose a shorter streaming duration limit than its
non-streaming upload limit.

## Transcription versus translation

`/v1/audio/transcriptions` preserves the spoken language. Speech-to-English
translation uses `/v1/audio/translations` and is capability-gated by model.
See [Audio translation support](../../basic_usage/audio_translations.md) before
switching routes.

## Errors

The endpoint returns HTTP 400 for empty files, unsupported response formats,
subtitle formats without model-provided segment timestamps, invalid
model-specific hints, and request limits such as excessive audio duration.
Runtime failures and queue saturation use the shared serving error mapping.
See the model cookbook for its concrete constraints and
[Admission control](../advanced_features/admission_control.md) for overload
behavior.

## Related documentation

- [Streaming](../advanced_features/streaming.md)
- [Audio translation support](../../basic_usage/audio_translations.md)
- [Qwen3-ASR cookbook](../../cookbook/qwen3_asr.md)
- [Supported models](../../supported_models.md)
