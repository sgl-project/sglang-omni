# Speech API

SGLang-Omni exposes OpenAI-compatible text-to-speech serving with extensions
for voice cloning, model-specific controls, batching, uploaded voices, and
stateful text input.

Check the [supported-model matrix](../../supported_models.md) and the model
cookbook before using an extension. The request schema accepts a shared set of
fields, but each model decides which conditioning and generation controls it
supports.

## Endpoints

| Endpoint | Transport | Purpose |
|---|---|---|
| `POST /v1/audio/speech` | HTTP | Generate one audio response or a raw PCM stream |
| `POST /v1/audio/speech/batch` | HTTP | Generate independent speech items with shared defaults |
| `/v1/audio/speech/stream` | WebSocket | Send and commit text segments over one persistent session |
| `GET /v1/audio/voices` | HTTP | List preset and uploaded voices |
| `POST /v1/audio/voices` | HTTP | Register a reusable uploaded voice reference |
| `DELETE /v1/audio/voices/{name}` | HTTP | Delete an uploaded voice reference |

See [Streaming](../advanced_features/streaming.md) for the HTTP and WebSocket
streaming contracts.

## Send one request

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "fishaudio/s2-pro",
    "voice": "default",
    "input": "SGLang-Omni serves speech models."
  }' \
  --output output.wav
```

The non-streaming response body contains encoded audio. `Content-Type` matches
the selected format and `Content-Disposition` supplies a filename. When the
pipeline reports them, these response headers carry terminal and usage data:

| Header | Meaning |
|---|---|
| `X-Finish-Reason` | `stop`, `length`, or another model-provided terminal reason |
| `X-Prompt-Tokens` | Prompt token count |
| `X-Completion-Tokens` | Generated token count |
| `X-Engine-Time` | Engine processing time in seconds |

## Request fields

### Core fields

| Field | Type | Default | Behavior |
|---|---|---|---|
| `model` | string | Served model | Model identifier |
| `input` | string | Required | Non-empty text to synthesize |
| `voice` | string | `default` | Preset or uploaded voice name; `speaker` is an alias |
| `response_format` | string | `wav` | `wav`, `mp3`, `flac`, `opus`, `aac`, or `pcm` |
| `speed` | float | `1.0` | Playback speed from `0.25` through `4.0` |
| `stream` | boolean | `false` | Return incremental raw PCM; requires `response_format: pcm` |

Encoded formats require their runtime encoder dependency. If the encoder is
unavailable, the server returns HTTP 503 instead of silently changing formats.

### Model extensions

The shared schema also accepts:

- voice conditioning: `references`, `ref_audio`, `ref_text`,
  `x_vector_only_mode`
- model modes: `task_type`, `language`, `instructions`
- length and cadence: `max_new_tokens`, `token_count`, `duration_tokens`,
  `initial_codec_chunk_frames`
- sampling: `temperature`, `top_p`, `top_k`, `repetition_penalty`, `seed`
- advanced per-stage values: `stage_params`

These fields do not imply universal support. Use the model cookbook for
required references, accepted languages, defaults, unsupported fields, and
quality tradeoffs.

`references` contains objects with an audio source and optional transcript.
`ref_audio` and `ref_text` are shorthand for a single reference. Audio sources
can be local paths, file or data URLs, or HTTP URLs when the server's media
policy allows them.

## Batch speech

`POST /v1/audio/speech/batch` accepts shared request defaults plus an `items`
list. Each item overrides the shared values and follows the normal speech
validation and generation path.

```bash
curl -X POST http://localhost:8000/v1/audio/speech/batch \
  -H "Content-Type: application/json" \
  -d '{
    "model": "fishaudio/s2-pro",
    "response_format": "wav",
    "items": [
      {"input": "First item."},
      {"input": "Second item.", "speed": 1.1}
    ]
  }'
```

The JSON response preserves item order. A successful item contains
base64-encoded audio, format, media type, and an optional finish reason. A
failed item contains an OpenAI-style error object. Invalid batch envelopes fail
the whole HTTP request.

## Errors

Validation and generation failures use an OpenAI-style envelope:

```json
{
  "error": {
    "message": "stream=true requires response_format='pcm'",
    "type": "BadRequestError",
    "param": "response_format",
    "code": 400
  }
}
```

Use model cookbooks for model-specific validation such as required reference
audio, unsupported voices, or language restrictions. Queue saturation can
return HTTP 503; see [Admission control](../advanced_features/admission_control.md).

## Related documentation

- [TTS model usage](../../basic_usage/tts.md)
- [Streaming](../advanced_features/streaming.md)
- [Admission control](../advanced_features/admission_control.md)
- [Supported models](../../supported_models.md)
