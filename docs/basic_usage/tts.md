# TTS Model Usage

Use the OpenAI-compatible `/v1/audio/speech` interface to synthesize text. This
page covers common part in TTS cookbook inlcluding: model setup, required inputs, and
implementation-specific options belong in the linked cookbooks. The examples
use [Qwen3-TTS CustomVoice](../cookbook/qwen3_tts.md#customvoice-checkpoints), which accepts text-only requests. Replace the model and add any fields required by your checkpoint.

## Prerequisites

Install SGLang-Omni using the [installation guide](../get_started/installation.md)
and follow your model's cookbook for its dependencies and weights. 

## Supported TTS Models

The streaming column describes incremental audio for the Python serving pipeline. follow the cookbook for the
model you are serving.

| Model | Incremental audio | Request requirements and model-specific behavior |
|---|---|---|
| [Fish Speech S2-Pro](../cookbook/fishaudio_s2_pro.md) | Yes | Text-only or voice cloning with `references` |
| [Qwen3-TTS Base](../cookbook/qwen3_tts.md) | Yes | Reference audio or a precomputed speaker embedding |
| [Qwen3-TTS CustomVoice](../cookbook/qwen3_tts.md#customvoice-checkpoints) | Yes | Built-in speakers; 1.7B supports instruction control |
| [Qwen3-TTS VoiceDesign](../cookbook/qwen3_tts.md) | Yes | `task_type: "VoiceDesign"` and non-empty `instructions`; checkpoint `Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign` uses `examples/configs/qwen3_tts_1_7b_voicedesign.yaml` |
| [Fun-CosyVoice3](../cookbook/fun_cosyvoice3.md) | Yes | One reference clip; cloning, cross-lingual and instruction modes |
| [MOSS-TTS](../cookbook/moss_tts.md) | Yes | Optional reference and model-specific duration control |
| [MOSS-TTS Local](../cookbook/moss_tts_local.md) | Yes | Optional reference; 48 kHz output |
| [Higgs TTS](../cookbook/higgs_tts.md) | Yes | Optional reference, inline control tokens |
| [Voxtral TTS](../cookbook/voxtral_tts.md) | Yes | Named preset voices; no reference cloning |
| [dots.tts](../cookbook/dots_tts.md) | Yes | Reference audio and transcript; MeanFlow and SOAR have different settings |
| [Ming-Omni-TTS](../cookbook/ming_tts.md) | Yes | Optional reference and transcript; `initial_codec_chunk_frames` is rejected |
| [Ming-3.0-TTS](../cookbook/ming3_tts.md) | CUDA; CPU decodes a complete waveform | Optional reference; the Tiny and 0.5B checkpoints handle `ref_text` differently |
| [Ming-3.0 audio](../cookbook/ming3_audio.md) | See cookbook | Speech uses one reference clip and its transcript |
| [ZONOS2](../cookbook/zonos2.md) | Yes | Reference cloning and model-specific chunk configuration |
| [AuK](../cookbook/auk.md) | No | Serial, non-streaming engine; `stage_params.auk_engine.gen_seconds` is required for speech |

Ming-Omni-TTS sets its first and steady chunk cadence through the
`audio_decode.factory.initial_chunk_patches` and `steady_chunk_patches` server
settings; it does not accept `initial_codec_chunk_frames` in a request.

<a id="launch-the-server"></a>

## Start a TTS Server

Choose a checkpoint and the matching configuration from its cookbook. For
example, after installing the [Qwen3-TTS dependencies](../cookbook/qwen3_tts.md#prerequisites):

```bash
sgl-omni serve \
  --model-path Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  --config examples/configs/qwen3_tts_1_7b_customvoice.yaml \
  --port 8000
```

The server reads reference paths on the server host and requires the relevant
`--allowed-local-media-path`. Remote references require an
`--allowed-media-domain`; redirect hosts must also be allowed. See the
[installation guide](../get_started/installation.md) and your model cookbook for
setup, and [TTS Process Topology](tts_process_topology.md) for stage placement.

<a id="use-curl"></a>

## Non-streaming Speech

The default response is a complete WAV file. This CustomVoice example selects a built-in speaker and sends no reference audio. Base checkpoints require a reference clip.

### cURL

```bash
curl -fS -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    "input": "Hello from SGLang-Omni.",
    "voice": "Ryan",
    "language": "English"
  }' \
  --output output.wav
```

<a id="use-python"></a>
<a id="basic-tts"></a>

### Python

```python
from pathlib import Path

import requests

response = requests.post(
    "http://localhost:8000/v1/audio/speech",
    json={
        "model": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        "input": "Hello from SGLang-Omni.",
        "voice": "Ryan",
        "language": "English",
    },
    timeout=600,
)
response.raise_for_status()
Path("output.wav").write_bytes(response.content)
```

### OpenAI Python SDK

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
response = client.audio.speech.create(
    model="Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    voice="Ryan",
    input="Hello from SGLang-Omni.",
    response_format="wav",
)
response.stream_to_file("output.wav")
```

## Streaming

Streaming requires a model and backend that produce incremental audio. Use
`stream=true` for raw HTTP audio, or `stream_format="sse"` for SSE. The latter
enables streaming even if `stream` is omitted. Raw WAV and SSE WAV are the
default; set `response_format="pcm"` for headerless PCM.

### Raw WAV and PCM

```bash
curl -fS -N -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    "input": "Hello from SGLang-Omni.",
    "voice": "Ryan",
    "language": "English",
    "stream": true,
    "response_format": "wav"
  }' \
  --output streaming.wav
```

Raw WAV has one streaming WAV header followed by mono 16-bit PCM. Its header
uses placeholder length fields because the final size is not yet known. Some
file-oriented tools require a finalized WAV header; use the PCM-to-WAV client
below when saving for those tools. With `response_format="pcm"`, the body is
headerless signed 16-bit little-endian PCM (`audio/pcm`). Both responses provide
`X-Sample-Rate`, `X-Channels`, and `X-Bit-Depth` headers. Read the sample rate
from the response instead of assuming it is the same for every model.

Raw streams contain no in-band completion or error event. Once a stream ends,
look up its `X-SGLang-Omni-Speech-Id` at
`GET /v1/audio/speech/{request_id}` for `finish_reason` and `usage`. The speech
ID is distinct from the `X-Request-Id` correlation header. Behind a router, echo
`X-SGLang-Omni-Worker` as `X-SGLang-Omni-Route-Worker` on the lookup. Only
recent outcomes are retained; a missing outcome does not prove completion.

### Python Streaming Client

This client writes the PCM stream incrementally and finalizes the WAV header
when it closes. It reads the format from response headers so it works with
models at different sample rates. Install `requests` first.

```python
import wave

import requests

with requests.post(
    "http://localhost:8000/v1/audio/speech",
    json={
        "model": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        "input": "Hello from SGLang-Omni.",
        "voice": "Ryan",
        "language": "English",
        "stream": True,
        "response_format": "pcm",
    },
    stream=True,
    timeout=600,
) as response:
    response.raise_for_status()
    sample_rate = int(response.headers["X-Sample-Rate"])
    channels = int(response.headers["X-Channels"])
    bit_depth = int(response.headers["X-Bit-Depth"])
    if bit_depth != 16:
        raise ValueError(f"Unexpected PCM bit depth: {bit_depth}")
    with wave.open("output_stream.wav", "wb") as output:
        output.setnchannels(channels)
        output.setsampwidth(bit_depth // 8)
        output.setframerate(sample_rate)
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                output.writeframesraw(chunk)
```

### Server-Sent Events (SSE)

```bash
curl -fS -N -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    "input": "Hello from SGLang-Omni.",
    "voice": "Ryan",
    "language": "English",
    "stream_format": "sse",
    "response_format": "pcm"
  }'
```

Each `speech.audio.delta` carries base64-encoded audio and its
`response_format`; WAV starts with one header delta. A successful stream ends
with `speech.audio.done`, containing `finish_reason`, duration in seconds, and
`usage` (which may be `null`). There is no `[DONE]` sentinel. The response
has the same sample-rate, channel and bit-depth headers as raw streaming.
Errors after streaming begins use SSE events; see
[Error Handling](#error-handling).

### SSE Event Framing

SSE responses use data-only frames by default to preserve clients that listen for the default `message` event. The client reads the JSON `type` to distinguish audio chunks, completion, and errors.

To enable named events, add `--enable-speech-sse-named-events` when starting the server:

```bash
sgl-omni serve \
  --model-path Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice \
  --config examples/configs/qwen3_tts_1_7b_customvoice.yaml \
  --enable-speech-sse-named-events \
  --port 8000
```

This is a server startup flag, rather than a JSON request field. It applies to every SSE speech response from that server. Requests still select SSE with `"stream_format": "sse"`; `response_format` independently selects WAV or PCM. Requests using the raw `audio` transport keep their binary response format.

| Server startup option | SSE framing | Client event listener | JSON error `type` after streaming starts |
| --- | --- | --- | --- |
| Flag omitted (default) | `data:` only | `message`; dispatch using JSON `type` | `error` |
| `--enable-speech-sse-named-events` | `event:` followed by `data:` | `speech.audio.delta`, `speech.audio.done`, and `speech.audio.error` | `speech.audio.error` |

For the same request with `"stream_format": "sse"` and `"response_format": "pcm"`, a successful response without the flag has this form. Audio is abbreviated and duration is illustrative; generation may produce multiple audio delta frames:

```text
data: {"type":"speech.audio.delta","audio":"<base64 PCM>","response_format":"pcm"}

data: {"type":"speech.audio.done","duration":0.2,"usage":null,"finish_reason":"stop"}

```

With the flag enabled, the same events include explicit SSE event names:

```text
event: speech.audio.delta
data: {"type":"speech.audio.delta","audio":"<base64 PCM>","response_format":"pcm"}

event: speech.audio.done
data: {"type":"speech.audio.done","duration":0.2,"usage":null,"finish_reason":"stop"}

```

The `event:` line controls SSE event dispatch; it is separate from the `type` field inside JSON. A client that subscribes only to `message` will not receive these named events. Enable the flag when clients subscribe to the named events or their parser handles both framing styles.

### WebSocket Speech Streaming

Use `/v1/audio/speech/stream` for stateful text input over a persistent
WebSocket. The first message must be `session.config`. Then send `input.text`
messages. Send `input.commit` to flush the current text segment while keeping
the WebSocket open, or finish the session with `input.done`. The server
acknowledges the initial configuration with `session.configured`.

`stream_audio` defaults to `false`. With the default, each completed text
segment returns one binary audio frame between `audio.start` and `audio.done`.
For `stream_audio=true`, `response_format` must be `pcm`, and the server sends
incremental binary PCM frames between `audio.start` and `audio.done`.

```python
import asyncio
import json

import websockets


async def main():
    async with websockets.connect(
        "ws://localhost:8000/v1/audio/speech/stream"
    ) as ws:
        await ws.send(json.dumps({
            "type": "session.config",
            "session": {
                "model": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
                "voice": "Ryan",
                "language": "English",
                "response_format": "pcm",
                "stream_audio": True,
                "split_granularity": "sentence",
            },
        }))
        print(await ws.recv())

        pcm_chunks = []
        await ws.send(json.dumps({
            "type": "input.text",
            "text": "Hello from the speech WebSocket. This is the second sentence.",
        }))
        await ws.send(json.dumps({"type": "input.commit"}))

        # input.committed is emitted after all audio for the segment. More
        # input.text/input.commit pairs can follow on the same WebSocket.
        while True:
            message = await ws.recv()
            if isinstance(message, bytes):
                pcm_chunks.append(message)
                continue
            event = json.loads(message)
            print(event)
            if event["type"] == "input.committed":
                break

        await ws.send(json.dumps({"type": "input.done"}))

        while True:
            message = await ws.recv()
            if isinstance(message, bytes):
                pcm_chunks.append(message)
                continue
            event = json.loads(message)
            print(event)
            if event["type"] == "session.done":
                break

        with open("websocket_output.pcm", "wb") as f:
            f.write(b"".join(pcm_chunks))


asyncio.run(main())
```

Each `input.commit` forces a flush of any remaining buffered text (including text that does not end at the configured sentence
or clause boundary). After all audio for that segment, the server emits
`input.committed` with `segment_index`, `segment_sentences`, and cumulative
`total_sentences`, then accepts more input on the same connection. An empty
commit is valid: it flushes nothing and reports `segment_sentences: 0`.
`input.done` performs the same final buffer flush, emits `session.done`, and
closes the WebSocket.

`split_granularity` can be `sentence` or `clause`. Unknown message types and
malformed JSON return a WebSocket `error` event. Missing or invalid initial
configuration returns an error and closes the session.

## Voice Cloning

Models that accept reference audio can use `references`, with an `audio_path`
and optionally the reference transcript as `text`. `ref_audio` and `ref_text`
are shorthand for the first reference on models supporting these fields.
Requirements vary: some require a transcript, others use it only for a
particular task. Consult the model cookbook. A reference can be a permitted
server-local path, data URL, or allowed HTTP URL; configure media access and
redirect hosts before using a remote URL.

### Qwen3-TTS Base Voice Cloning

CustomVoice rejects reference audio. Start the Base checkpoint for this request, instead of the CustomVoice server above. The clip below is from [`seed-tts-eval-mini`](https://huggingface.co/datasets/zhaochenyang20/seed-tts-eval-mini). `references` accepts `audio_path` (a local path, file URL, data URL, or HTTP URL) and `text` (transcript of that audio). Supplying the transcript selects in-context cloning.

```bash
sgl-omni serve \
  --model-path Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --config examples/configs/qwen3_tts_1_7b.yaml \
  --allowed-media-domain huggingface.co \
  --port 8000
```

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    "voice": "default",
    "input": "Get the trust fund to the bank early.",
    "references": [{
      "audio_path": "https://huggingface.co/datasets/zhaochenyang20/seed-tts-eval-mini/resolve/main/en/prompt-wavs/common_voice_en_10119832.wav",
      "text": "We asked over twenty different people, and they all said it was his."
    }]
  }' \
  --output output.wav
```

To stream the same reference, add `"stream": true` and use the [streaming examples](#streaming). Permit the download redirect host as well as `huggingface.co`.

## Batch Speech

Use `/v1/audio/speech/batch` when one request should synthesize several
independent utterances. Batch defaults are merged with each item. Item fields
override the defaults, and each item runs through the normal `/v1/audio/speech`
path.

Batch speech does not support streaming, including `stream_format="sse"`.
It returns a single JSON response containing the completed results. For streaming
output, use `/v1/audio/speech`. The server accepts up to 32 items by default;
set `--tts-batch-max-items` at startup to change the envelope limit.

CustomVoice rejects `ref_audio` on a batch item. A Base server accepts it, using the same reference fields as [Qwen3-TTS Base voice cloning](#qwen3-tts-base-voice-cloning).

```bash
curl -X POST http://localhost:8000/v1/audio/speech/batch \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    "voice": "Ryan",
    "language": "English",
    "response_format": "wav",
    "items": [
      {"input": "First sentence."},
      {"input": "Second sentence.", "speed": 1.1},
      {"input": "Third sentence.", "voice": "Vivian"}
    ]
  }'
```

The response preserves item order. Successful items contain base64-encoded
audio bytes and the selected media type. Failed items contain an OpenAI-style
error object at the item level. Invalid batch envelopes, such as too many
items, fail the HTTP request.

## Uploaded Voices

Use `/v1/audio/voices` to register reference clips once and reuse them by name
in later `/v1/audio/speech` requests. Uploaded samples are stored as
`.safetensors` files under `SPEAKER_SAMPLES_DIR` and are restored when the
server restarts. If `SPEAKER_SAMPLES_DIR` is not set, the server uses
`~/.cache/sglang-omni/speakers`. `SPEAKER_MAX_UPLOADED` limits the number of
stored voices and defaults to `1000`.

Upload a voice sample:

```bash
curl -X POST http://localhost:8000/v1/audio/voices \
  -F "name=narrator" \
  -F "consent=consent-recording-id" \
  -F "ref_text=Transcript of the uploaded reference clip." \
  -F "speaker_description=Clear narration voice" \
  -F "audio_sample=@reference.wav;type=audio/wav"
```

List preset and uploaded voices:

```bash
curl http://localhost:8000/v1/audio/voices
```

For CustomVoice, the response's `voices` list contains `default` and the served checkpoint's built-in speakers. Uploaded reference voices are not used for CustomVoice synthesis. Reuse an uploaded voice from a Base server:

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    "input": "The uploaded voice can now be reused without resending audio.",
    "voice": "narrator",
    "response_format": "wav"
  }' \
  --output narrator.wav
```

Delete an uploaded voice:

```bash
curl -X DELETE http://localhost:8000/v1/audio/voices/narrator
```

Accepted upload formats are WAV, MP3, FLAC, OGG, AAC, WebM, and MP4. Each file
must be at most 10 MiB and contain 1-30 seconds of non-silent reference audio.
Uploading the same `name` overwrites the previous sample. Deleting a voice
removes the persisted sample. The list response includes API-process
`cache_stats` for uploaded-voice reference lookup observability.

## Request Parameters

These fields are accepted by the shared speech endpoint. Model support and defaults vary; consult the relevant cookbook before relying on a model-specific field.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model` | string | served model | Served model identifier |
| `input` | string | (required) | Text to synthesize |
| `voice` | string | `"default"` | Preset or uploaded voice identifier |
| `response_format` | string | `"wav"` | Output audio format: `wav`, `mp3`, `flac`, `pcm`, `aac`, or `opus`. Explicit values override the default in every mode. HTTP streaming accepts only `wav` or `pcm` |
| `speed` | float | `1.0` | Playback speed multiplier from `0.25` to `4.0` |
| `stream` | bool | `false` | Enable raw WAV streaming by default. Set `response_format="pcm"` for raw PCM, or `stream_format="sse"` for SSE events |
| `stream_format` | string | `"audio"` | Streaming transport: `audio` returns raw WAV or PCM bytes when streaming is enabled; `sse` enables SSE even when `stream` is omitted or false. SSE uses data-only `message` events by default. The server startup flag `--enable-speech-sse-named-events` selects the framing and error type shown in the [SSE Event Framing table](#sse-event-framing) |
| `initial_codec_chunk_frames` | int | `null` | Model-specific first streaming chunk override. Qwen3-TTS, Higgs, MOSS-TTS Local, and ZONOS2 document support; `0` selects their steady chunk size. Ming-Omni-TTS rejects this field |
| `stream_codec_output` | bool | `true` | Qwen3-TTS only. Forward codec frames to the vocoder as they are generated. Set `false` to restore whole-utterance decoding for CustomVoice / VoiceDesign |
| `suppress_bootstrap_silence` | bool | `true` | Qwen3-TTS only. Withhold the silent bootstrap codec frame's audio from streamed CustomVoice output on validated voice/language pairs; an audible first frame is always emitted unchanged. Set `false` to keep the leading silence |
| `references` | list | `null` | Reference audio for voice cloning. Each item has `audio_path` (local path / file URL / data URL / remote URL) and `text`. For Qwen3-TTS Base, the only item may instead carry `speaker_embedding`: the x-vector the same checkpoint's speaker encoder computes from the clip, 1024 numbers for 0.6B and 2048 for 1.7B, each finite in the model's dtype, which implies x-vector-only mode |
| `ref_audio` | string | `null` | Reference audio path / URL / base64 string. Equivalent to `references[0].audio_path` |
| `ref_text` | string | `null` | Transcript for `ref_audio`. Equivalent to `references[0].text` |
| `language` | string | `null` | Model-specific language hint; supported values vary by model |
| `task_type` | string | `null` | Model-specific task selector; Qwen3-TTS supports `Base`, `CustomVoice`, and `VoiceDesign` |
| `instructions` | string | `null` | Model-specific style or voice instructions |
| `max_new_tokens` | int | `null` | Maximum number of generated tokens |
| `token_count` | int | `null` | Model-specific duration token target |
| `duration_tokens` | int | `null` | Alias-style duration token target for models that expose duration control |
| `x_vector_only_mode` | bool | `null` | Qwen3-TTS Base speaker-embedding mode |
| `temperature` | float | `null` | Sampling temperature |
| `top_p` | float | `null` | Top-p sampling |
| `top_k` | int | `null` | Top-k sampling |
| `epsilon_cutoff` | float | `null` | Model-specific sampling cutoff |
| `repetition_penalty` | float | `null` | Repetition penalty |
| `seed` | int | `null` | Model-specific. Qwen3-TTS Base accepts request-scoped seed, Voxtral TTS currently rejects seed |
| `stage_params` | object | `null` | Model-specific stage overrides; see the model cookbook |
| `return_spec_tokens_details` | bool | `false` | With a speculative draft, return this request's acceptance statistics: `X-SGLang-Omni-Spec-*` headers on buffered audio, `sglext.spec_tokens_details` on `speech.audio.done` and on the raw audio outcome |

## Error Handling

Invalid speech requests return an OpenAI-style error envelope. For example, streaming with `response_format="mp3"` or `"flac"` returns HTTP 400:

```json
{
  "error": {
    "message": "Streaming requires response_format='pcm' or 'wav'",
    "type": "BadRequestError",
    "param": "response_format",
    "code": 400
  }
}
```

Validation and generation failures before the first chunk return HTTP errors.
After an SSE response has started, failures arrive as an `error` or
`speech.audio.error` event, depending on the framing mode, instead of a done
event. If audio was already emitted, the event includes `partial_audio: true`
and `action: "discard"`. A raw audio stream has no in-band error message; use
the outcome lookup described above and treat a missing outcome as unknown.

<a id="h200-seedtts-benchmark-commands"></a>

For SeedTTS benchmarking, see the [TTS benchmark reference](../developer_reference/tts_bench.md)
and the model-specific cookbook. Results depend on the checkpoint, hardware,
reference inputs, and workload.

## Interactive Playground

SGLang-Omni ships with a Gradio-based playground for interactive TTS experimentation:

```bash
./playground/s2pro/start.sh
```

The playground now exposes two demo modes against the same S2 Pro backend:

- `Non-Streaming` starts a standard request and shows the final WAV after generation finishes.
- `Streaming` consumes the `/v1/audio/speech` raw PCM stream, converts incremental chunks for playback, and also writes a final combined WAV artifact for inspection.

The launcher starts the backend first, waits for `/health`, then starts the Gradio UI with:

```bash
python -m playground.s2pro.app --api-base http://localhost:8000
```

A demo play video is available [here](https://x.com/lmsysorg/status/2031412267213008984/video/1). We highly recommend using playground since audio data is hard to interact with by CLI.
