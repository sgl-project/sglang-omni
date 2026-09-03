# Streaming

SGLang-Omni uses different streaming transports for different interaction
patterns. Choose the endpoint by the type of input and output; `stream=true`
does not imply one universal framing protocol.

## Transport matrix

| Use case | Endpoint | Transport | Payload |
|---|---|---|---|
| One speech request, incremental audio | `POST /v1/audio/speech` | HTTP body stream | Raw PCM bytes |
| One transcription, incremental text | `POST /v1/audio/transcriptions` | SSE | Transcript delta and done events |
| One chat completion, incremental output | `POST /v1/chat/completions` | SSE | Chat completion events |
| Persistent text input for speech | `/v1/audio/speech/stream` | WebSocket | JSON control events plus binary audio frames |
| Bidirectional realtime conversation | `/v1/realtime` | WebSocket | Realtime text, audio, VAD, and lifecycle events |

Model and checkpoint support varies. Check the
[supported-model matrix](../../supported_models.md) and model cookbook before
building a client around a transport.

## HTTP speech streaming

Set both `stream=true` and `response_format=pcm`:

```bash
curl -N -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "fishaudio/s2-pro",
    "voice": "default",
    "input": "Stream this sentence.",
    "stream": true,
    "response_format": "pcm"
  }' \
  --output output.pcm
```

The body is headerless signed 16-bit mono PCM, not JSON or a WAV container.
Read its format from the response:

| Header | Meaning |
|---|---|
| `Content-Type: audio/pcm` | Raw PCM body |
| `X-Sample-Rate` | Model output sample rate |
| `X-Channels` | Channel count; currently `1` for this route |
| `X-Bit-Depth` | Sample width; currently `16` |

The stream ends when the HTTP body closes. It has no in-band terminal sentinel
or final usage event. A client that saves the body must add the correct WAV
header itself if it wants a WAV file.

`initial_codec_chunk_frames` is a model-specific latency-versus-continuity
control. Omit it for the model default; use the cookbook before overriding it.

## Transcription SSE

Set `stream=true` with `response_format=json` or `text`. The event sequence is:

```text
transcript.text.delta  # zero or more incremental text fragments
transcript.text.done   # complete transcript and optional duration usage
```

Do not request `verbose_json` while streaming. Long-audio chunking and maximum
stream duration are model-specific.

## Stateful speech WebSocket

Use `/v1/audio/speech/stream` when text arrives over time and one connection
should produce multiple committed speech segments.

1. Send `session.config` as the first message.
2. Wait for `session.configured`.
3. Send one or more `input.text` messages.
4. Send `input.commit` to flush a segment and keep the connection open.
5. Wait for `input.committed`, then continue or send `input.done`.
6. Wait for `session.done`; the server then closes the session.

`stream_audio` defaults to `false`. In that mode, each committed segment emits
one binary encoded-audio frame between `audio.start` and `audio.done`. With
`stream_audio=true`, `response_format` must be `pcm`, and the server emits
incremental binary PCM frames within the same boundary events.

`split_granularity` accepts `sentence` or `clause`. `input.commit` flushes any
remaining buffered text even when it does not end on the configured boundary.
Malformed JSON and unknown message types produce an `error` event; an invalid
initial configuration also closes the session.

See [TTS model usage](../../basic_usage/tts.md) for a complete Python WebSocket
client.

## Chat and realtime streaming

`/v1/chat/completions` uses SSE when the request sets `stream=true`. The output
modalities and event contents depend on whether the server runs a text-only or
speech pipeline.

`/v1/realtime` is a separate bidirectional WebSocket protocol for incremental
audio input, server VAD, text/audio output, cancellation, and barge-in. It must
be enabled at server startup and requires a pipeline with the requested output
stages. See [Omni model usage](../../basic_usage/qwen3_omni.md) for the current
realtime session and interruption contract.

## Benchmarking streaming

Report at least time to first text or audio, end-to-end latency, concurrency,
audio duration, output cadence or chunk policy, and whether the client consumed
or buffered chunks during measurement. Compare streaming and non-streaming
paths with identical prompts, seeds, and launch settings when possible. See
[Benchmark methodology](../../benchmarks/methodology.md).

## Related documentation

- [Speech API](../serving/speech_api.md)
- [Transcription API](../serving/transcription_api.md)
- [TTS model usage](../../basic_usage/tts.md)
- [Omni model usage](../../basic_usage/qwen3_omni.md)
