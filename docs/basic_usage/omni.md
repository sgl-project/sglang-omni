# Omni Model Usage

SGLang-Omni serves omni models through the OpenAI-compatible `/v1/chat/completions` endpoint. A request can carry text, image, audio, and video, and the reply is text-only or text and speech.

This page covers the request format shared by all omni models, including streaming and realtime sessions. Launch commands, tuning, and model-specific fields are on each model's page. Examples use `qwen3-omni` as the model name.

## Prerequisites

Install SGLang-Omni using the [installation guide](../get_started/installation.md) and follow your model's cookbook for its dependencies and weights.

## Supported Omni Models

| Model | Input | Output | Notes |
|---|---|---|---|
| [Qwen3-Omni](qwen3_omni.md) | Text, image, audio, video | Text, or text and speech | Text-only mode or the full speech pipeline; realtime sessions |
| [Ming-Omni](../cookbook/ming_omni.md) | Text, image, audio, video | Text, or text and speech | 44.1 kHz speech; thinker and talker stages |
| [Ming-3.0-audio](../cookbook/ming3_audio.md) | Audio, with optional text | Text, or text and speech | Transcription, audio question answering, and speech synthesis |
| [MiniCPM-o](../cookbook/minicpm_o.md) | Text, image, audio, video | Text, or text and speech | Chat here; live full-duplex calls use `/v1/realtime` |


## General Usage

A chat request carries the conversation in `messages` and the requested outputs in `modalities`. The default is `["text"]`; adding `"audio"` asks for speech and requires a server that runs the model's speech stages.

### Text Output

```bash
curl -X POST http://localhost:8008/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-omni",
    "messages": [{"role": "user", "content": "Summarize the plot in one sentence."}],
    "modalities": ["text"],
    "max_tokens": 128
  }'
```

The answer is in `choices[0].message.content`:

```python
import requests

response = requests.post(
    "http://localhost:8008/v1/chat/completions",
    json={
        "model": "qwen3-omni",
        "messages": [{"role": "user", "content": "Summarize the plot in one sentence."}],
        "modalities": ["text"],
        "max_tokens": 128,
    },
    timeout=600,
)
response.raise_for_status()
print(response.json()["choices"][0]["message"]["content"])
```

The same request works through the OpenAI Python SDK when the client points at the SGLang-Omni server. Extra fields such as `modalities` and the media lists pass through unchanged:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8008/v1", api_key="EMPTY")
response = client.chat.completions.create(
    model="qwen3-omni",
    messages=[{"role": "user", "content": "Summarize the plot in one sentence."}],
    modalities=["text"],
    max_tokens=128,
)
print(response.choices[0].message.content)
```

### Media Input

Media reaches the model in two ways, and a request may use either one:

- Top-level `images`, `audios`, and `videos` lists, each entry a local path visible to the server, an HTTP URL, or a data URL.
- OpenAI content parts inside a message: `text`, `image_url`, `video_url`, `audio_url`, and `input_audio`, where `input_audio` carries base64 `wav` or `mp3` data.

```json
{
  "role": "user",
  "content": [
    {"type": "image_url", "image_url": {"url": "tests/data/cars.jpg"}},
    {"type": "text", "text": "How many cars are there in the picture?"}
  ]
}
```

Media parts keep their message and their position in it across conversation turns. Any other part type is rejected with a 400. The optional `detail` field does not override the model's image preprocessing settings.

Where top-level media land in the prompt depends on the model. Qwen3-Omni places them after the parts of the final user message, MiniCPM-o before that message's text, and Ming-Omni in the first user message. Use content parts when the position of each item matters.

When every part of the question comes from audio, video, or image input, set `"content": ""` on the user message. Leaving a text question in `content` alongside the media makes the model read both.

### Image and Text Input

The examples below show the curl form. In Python, send the same JSON body as in Text Output and read the answer from `choices[0].message.content`.

Send an image with a text question to get a text response.

```bash
curl -X POST http://localhost:8008/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-omni",
    "messages": [{"role": "user", "content": "How many cars are there in the picture?"}],
    "images": ["tests/data/cars.jpg"],
    "modalities": ["text"],
    "max_tokens": 16
  }'
```

### Audio and Image Input

Send an audio file together with an image. The audio contains the spoken question ("How many cars are there in the picture?"), and the model answers from both inputs. The question is spoken, so `content` is `""`.

```bash
curl -X POST http://localhost:8008/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-omni",
    "messages": [{"role": "user", "content": ""}],
    "images": ["tests/data/cars.jpg"],
    "audios": ["tests/data/query_to_cars.wav"],
    "modalities": ["text"],
    "max_tokens": 16
  }'
```

### Video and Audio Input

Send a video with a spoken audio question. The model watches the video, hears the question, and responds with text.

The text message does not have to be empty. It can carry instructions that are not part of the question, such as routing or answer format, while the question itself stays in the audio. The Video-AMME CI benchmark uses this layout: video input, a WAV with the spoken question and options, and a text message that holds only routing and answer-format instructions.

```bash
curl -X POST http://localhost:8008/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-omni",
    "messages": [{"role": "user", "content": ""}],
    "videos": ["tests/data/draw.mp4"],
    "audios": ["tests/data/query_to_draw.wav"],
    "modalities": ["text"],
    "max_tokens": 16
  }'
```

### Speech Output

Set `"modalities": ["text", "audio"]` and select the audio container with `audio`. The reply text stays in `choices[0].message.content`; the speech arrives base64-encoded in `choices[0].message.audio.data`.

Every input combination above can ask for speech. Keep the same `messages` and media lists, set `modalities`, and add `audio`. For example, an image question answered in text and speech:

```bash
curl -X POST http://localhost:8008/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-omni",
    "messages": [{"role": "user", "content": "How many cars are there in the picture?"}],
    "images": ["tests/data/cars.jpg"],
    "modalities": ["text", "audio"],
    "audio": {"format": "wav"},
    "max_tokens": 16
  }'
```

```python
import base64
from pathlib import Path

import requests

response = requests.post(
    "http://localhost:8008/v1/chat/completions",
    json={
        "model": "qwen3-omni",
        "messages": [
            {"role": "user", "content": "How many cars are there in the picture?"}
        ],
        "images": ["tests/data/cars.jpg"],
        "modalities": ["text", "audio"],
        "audio": {"format": "wav"},
        "max_tokens": 16,
    },
    timeout=600,
)
response.raise_for_status()
message = response.json()["choices"][0]["message"]
print(message["content"])
Path("reply.wav").write_bytes(base64.b64decode(message["audio"]["data"]))
```

Some models return the spoken text as `message.audio.transcript` as well, and take an optional `audio.ref_audio` reference clip for voice conditioning; both are documented with the model. A text-only server accepts `"modalities": ["text", "audio"]` without raising an error but returns no audio, so check that the model page's launch command starts the speech stages.

### Streaming

Set `"stream": true` to receive Server-Sent Events. Text deltas arrive in `choices[0].delta.content`; speech chunks arrive base64-encoded in `choices[0].delta.audio.data`. A stream ends with `data: [DONE]`.

Chunk size depends on the model's pipeline. A pipeline without streaming stages can send the text or the audio as one chunk after generation finishes, so clients should not assume token-sized text deltas or short audio chunks. A request with `return_spec_tokens_details: true` gets one extra chunk before `[DONE]` with empty `choices`. An error event also has no `choices`.

```bash
curl -N -X POST http://localhost:8008/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-omni",
    "messages": [{"role": "user", "content": "Say one friendly sentence."}],
    "modalities": ["text", "audio"],
    "audio": {"format": "wav"},
    "stream": true,
    "max_tokens": 64
  }'
```

This client prints text as it arrives and writes each audio chunk to its own file. Each chunk is a complete container, so concatenating the files byte for byte does not produce one playable stream; decode the PCM frames if you need a single file.

```python
import base64
import json
from pathlib import Path

import requests

chunks: list[Path] = []
with requests.post(
    "http://localhost:8008/v1/chat/completions",
    json={
        "model": "qwen3-omni",
        "messages": [{"role": "user", "content": "Say one friendly sentence."}],
        "modalities": ["text", "audio"],
        "audio": {"format": "wav"},
        "stream": True,
        "max_tokens": 64,
    },
    stream=True,
    timeout=600,
) as response:
    response.raise_for_status()
    for line in response.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data: "):
            continue
        data = line.removeprefix("data: ")
        if data == "[DONE]":
            break
        event = json.loads(data)
        if "error" in event:
            raise RuntimeError(event["error"]["message"])
        if not event.get("choices"):
            continue
        delta = event["choices"][0].get("delta", {})
        if delta.get("content"):
            print(delta["content"], end="", flush=True)
        audio = delta.get("audio") or {}
        if audio.get("data"):
            path = Path(f"chunk_{len(chunks):03d}.wav")
            path.write_bytes(base64.b64decode(audio["data"]))
            chunks.append(path)
```

## Realtime Interaction

Realtime interaction is separate from both a completed chat request and an SSE response. The client and server keep a WebSocket session open and exchange input and output events continuously. Model pages document the supported event names, audio formats, and session limits.

### WebSocket / Realtime API

Connect to `/v1/realtime`, send a session configuration event, and then send
input events as they become available:

```json
{
  "type": "session.update",
  "session": {
    "modalities": ["text", "audio"],
    "input_audio_format": "pcm16",
    "output_audio_format": "pcm16"
  }
}
```

Audio is commonly sent as base64 PCM16 in `input_audio_buffer.append`. A client can end a turn with `input_audio_buffer.commit`, or let server-side VAD commit it automatically. A realtime session is not created by setting `stream: true`.

Models that take camera input during a session, such as MiniCPM-o and the Qwen3-Omni session profile, receive frames in `sglang.input_image.append` events. Each event carries a base64 JPEG or PNG in `image` and the frame's position on the audio timeline in `sglang.t_ms`.

### Turn-based, streaming cascade, and native full-duplex

- **Turn-based:** the client or server commits one user turn, then generation starts. The session usually does not listen while the reply is spoken.
- **Streaming cascade:** ASR, language generation, and TTS stream partial results through separate stages. This reduces latency but requires explicit transcript, timing, and cancellation handling between stages.
- **Native full-duplex:** one Omni session consumes and produces streams concurrently. It can listen while speaking, but requires explicit model and session support.

### Turn detection and interruption

Turn-based sessions may use client commits, `server_vad`, or semantic VAD when the deployment provides it. For barge-in, stop queued playback and ignore later audio deltas for the canceled response. If the listener heard only part of the reply, send the model-specific truncate event so conversation history matches what was played.

## Server Readiness

Before it reports ready, the server sends itself a chat request with a small image, one second of audio, and a short instruction — text and audio output in
speech mode, text output in text-only mode, at most 8 new tokens from each stage that generates tokens — once per replica of its most replicated process, one after
another, so that every replica serves one. `/health` returns 503 with status `starting` until these requests have passed through every stage, so the one-time kernel builds of that request path happen before the first user request. Inputs of other shapes, such as video or a large concurrent batch, can still build a kernel
variant on first use. A failed warmup stops the server.`SGLANG_OMNI_WARMUP_TIMEOUT` (seconds, default 600) bounds each request, and `--skip-server-warmup` reports ready without them.

## Request Parameters

Fields accepted by `/v1/chat/completions`. Model support and defaults vary; the model page or cookbook lists the fields, defaults, and sampling controls that apply to a specific checkpoint.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model` | string | served model | Model identifier; the `--model-name` value from launch |
| `messages` | list | (required) | Chat messages, each with `role` and `content` |
| `modalities` | list | `["text"]` | Requested outputs: `["text"]`, or `["text", "audio"]` for text and speech |
| `images` | list | `null` | Image paths, HTTP(S) URLs, or data URLs |
| `audios` | list | `null` | Audio paths, HTTP(S) URLs, or data URLs |
| `videos` | list | `null` | Video paths, HTTP(S) URLs, or data URLs |
| `max_tokens` | int | model default | Maximum tokens to generate |
| `max_completion_tokens` | int | `null` | OpenAI-compatible alias for `max_tokens` |
| `temperature` | float | model default | Sampling temperature |
| `top_p` | float | model default | Top-p sampling |
| `top_k` | int | model default | Top-k sampling |
| `min_p` | float | model default | Minimum-token probability |
| `repetition_penalty` | float | model default | Repetition penalty |
| `stop` | string or list | `null` | Stop sequences |
| `seed` | int | `null` | Random seed for reproducibility |
| `stream` | bool | `false` | Stream results as Server-Sent Events |
| `audio` | object | `null` | Speech output settings, such as `{"format": "wav"}`; some models add `ref_audio` |
| `stage_sampling` | object | `null` | Per-stage sampling overrides, such as `{"thinker": {"temperature": 0.8}}` |
| `stage_params` | object | `null` | Per-stage non-sampling overrides |
| `use_audio_in_video` | bool | `null` | Set `true` to process audio embedded in video input |
| `video_fps` | float | `null` | Frame sampling rate for video input; the server default applies when unset |
| `video_max_frames` | int | `null` | Maximum number of frames sampled from a video |
| `video_min_pixels` | int | `null` | Minimum pixels per video frame |
| `video_max_pixels` | int | `null` | Maximum pixels per video frame |
| `video_total_pixels` | int | `null` | Total pixel budget across all sampled frames |
| `return_spec_tokens_details` | bool | `false` | With speculative decoding, return acceptance statistics in `sglext.spec_tokens_details` |

Limiting `video_max_frames` or the pixel budget keeps video latency predictable.

## Known Limitations

- Non-streaming requests return HTTP 400 for invalid media, HTTP 503 when the request queue is full, and HTTP 500 for other server failures.
- With `stream=true`, an error detected after the event stream starts arrives as an SSE error event, and the stream still closes with `data: [DONE]`.
- In a stream, a full request queue uses `type: "server_error"` with `code: 503`, and other server failures use `type: "server_error"` with `code: 500`.
- The HTTP status stays 200 once streaming has started, so clients must check for an error event, including after partial output.
- A request whose prompt, or prompt plus `max_tokens`, exceeds the model's context length is rejected. Shorten the input or lower `max_tokens`.
- `"modalities": ["text", "audio"]` on a text-only server returns no audio and raises no error.
- Speculative decoding drafts restrict sampling. Requests that use penalties, `min_p`, or logprobs may be rejected; the model page lists the exact rules.
- Model-specific caps still apply, such as video frame-rate rules, context limits, and which outputs a pipeline can stream.

