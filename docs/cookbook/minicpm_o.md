# MiniCPM-o

[MiniCPM-o 4.5](https://huggingface.co/openbmb/MiniCPM-o-4_5) understands text, images, audio and video, and answers in text and speech. SGLang-Omni serves it in two ways:

| Mode | Endpoint | Use it for |
|---|---|---|
| Chat | `/v1/chat/completions` | One request, one reply, in text or speech |
| Full duplex | `/v1/realtime` (WebSocket) | A live voice or video call where the model listens and speaks at the same time |

## Prerequisites

Follow [Installation](../get_started/installation.md), then run from the repository root.

Chat with speech output:

```bash
python -m sglang_omni.cli serve --model-path openbmb/MiniCPM-o-4_5 --port 8000
```

Chat with text output only:

```bash
python -m sglang_omni.cli serve --model-path openbmb/MiniCPM-o-4_5 --text-only --port 8000
```

Full duplex:

```bash
hf download openbmb/MiniCPM-o-4_5 --local-dir models/MiniCPM-o-4_5

python -m sglang_omni.cli serve \
  --config examples/full_duplex/minicpmo.yaml \
  --model-path models/MiniCPM-o-4_5 \
  --enable-realtime --port 8000
```

The full-duplex server is ready when this returns JSON containing `"native_full_duplex":true`:

```bash
curl --fail http://localhost:8000/v1/realtime/capabilities
```

Full duplex has been tested on one H200. For a browser page with microphone and camera, see [playground/realtime](../../playground/realtime/README.md).

## Chat with a cloned voice

Pass a reference recording in `audio.ref_audio` and the reply is spoken in that voice:

```python
import base64
from pathlib import Path

from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")
reference = base64.b64encode(Path("reference.wav").read_bytes()).decode("ascii")
response = client.chat.completions.create(
    model="MiniCPM-o-4_5",
    messages=[{"role": "user", "content": "Please say hello."}],
    modalities=["text", "audio"],
    audio={
        "format": "wav",
        "ref_audio": f"data:audio/wav;base64,{reference}",
    },
)
```

The reference must be sent as a base64 audio data URI; paths and HTTP URLs are not fetched. Without a reference, the model speaks in the checkpoint's default voice. Audio inside `messages` is something for the model to listen to and does not change its voice.

## Full-duplex conversation

Open a WebSocket to `/v1/realtime`, send microphone audio as it is captured, and play the audio that comes back. The model works in units of one second: after each second of audio it decides whether to keep listening or to speak, and a spoken unit carries up to one second of reply.

| | Format |
|---|---|
| Input audio | mono PCM16, 16 kHz, base64 in `input_audio_buffer.append` |
| Output audio | mono PCM16, 24 kHz, base64 in `response.output_audio.delta` |
| Output text | `response.output_audio_transcript.delta` |

A session goes through these steps:

1. The server sends `session.created`.
2. Send `session.update` with the prompt and any settings from the sections below. The server answers `session.updated`.
3. Send audio with `input_audio_buffer.append` at the pace it is captured. Replies arrive while you send.
4. Send `sglang.input_audio.end` when there is no more audio, and wait for `sglang.input_audio.drained`.
5. Send `session.close`.

This client plays a 16 kHz mono WAV file to the model and saves what it says:

```python
import asyncio
import base64
import json
import wave

import websockets

PACKET_SAMPLES = 1280  # 80 ms at 16 kHz


async def main() -> None:
    with wave.open("question.wav", "rb") as source:
        pcm = source.readframes(source.getnframes())
    pcm += b"\x00\x00" * 16000 * 5  # five seconds of silence so the model can answer
    reply = bytearray()
    async with websockets.connect(
        "ws://localhost:8000/v1/realtime", max_size=16 * 1024 * 1024
    ) as websocket:
        async for message in websocket:
            event = json.loads(message)
            if event["type"] == "session.created":
                await websocket.send(
                    json.dumps(
                        {
                            "type": "session.update",
                            "event_id": "config",
                            "session": {
                                "instructions": "You are a helpful voice assistant.",
                                "output_modalities": ["audio"],
                            },
                        }
                    )
                )
            elif event["type"] == "session.updated":
                packet_bytes = PACKET_SAMPLES * 2
                for sequence, offset in enumerate(range(0, len(pcm), packet_bytes)):
                    await websocket.send(
                        json.dumps(
                            {
                                "type": "input_audio_buffer.append",
                                "event_id": f"audio-{sequence}",
                                "audio": base64.b64encode(
                                    pcm[offset : offset + packet_bytes]
                                ).decode(),
                                "sglang": {"seq": sequence},
                            }
                        )
                    )
                    await asyncio.sleep(PACKET_SAMPLES / 16000)
                await websocket.send(
                    json.dumps({"type": "sglang.input_audio.end", "event_id": "end"})
                )
            elif event["type"] == "response.output_audio_transcript.delta":
                print(event["delta"], end="", flush=True)
            elif event["type"] == "response.output_audio.delta":
                reply += base64.b64decode(event["delta"])
            elif event["type"] == "sglang.input_audio.drained":
                await websocket.send(
                    json.dumps({"type": "session.close", "event_id": "close"})
                )
            elif event["type"] == "session.closed":
                break
            elif event["type"] == "error":
                raise RuntimeError(event["error"])
    with wave.open("reply.wav", "wb") as output:
        output.setparams((1, 2, 24000, 0, "NONE", "not compressed"))
        output.writeframes(bytes(reply))


asyncio.run(main())
```

All settings below go in `session.update` before the first audio packet and stay fixed for the session. Start a new session to change them.

### Voice

Send a reference recording to choose the voice for one session:

```python
reference = base64.b64encode(Path("reference.wav").read_bytes()).decode("ascii")
update = {
    "type": "session.update",
    "event_id": "voice",
    "session": {
        "sglang": {
            "reference_audio": {"media_type": "audio/wav", "data": reference}
        }
    },
}
```

The reference must be a PCM16 WAV file, mono or stereo, 8 to 48 kHz, at most 30 seconds and 1 MiB. Without one, the session uses the server's `reference_audio` from the config file, or the checkpoint's default voice. An optional `tts_reference_audio` with the same structure changes only the output voice.

### Sampling

```json
{
  "type": "session.update",
  "event_id": "sampling",
  "session": {"sglang": {"sampling": {"temperature": 0.7, "top_p": 0.8}}}
}
```

| Field | Default | Range | Effect |
|---|---|---|---|
| `greedy` | `false` | | Always pick the most likely token |
| `temperature` | 0.7 | 0 to 2 | Randomness of the reply text; 0 picks the most likely token |
| `top_k` | 20 | -1 or more | Sample from the k most likely tokens; -1 or 0 turns it off |
| `top_p` | 0.8 | above 0, up to 1 | Nucleus sampling |
| `repetition_penalty` | 1.05 | 1 or more | Discourage repeated text |
| `repetition_window_size` | 512 | 1 or more | How many recent tokens the penalty looks at |
| `listen_prob_scale` | 1.0 | 0 or more | Above 1 makes the model listen more and speak less |
| `force_listen_count` | 3 | 0 or more | Units (3 = the first 3 s) in which the model only listens at the start of a session |
| `max_new_tokens_per_unit` | 20 | 1 or more | Most text tokens produced in one unit (1 s) |
| `talker_temperature` | 0.8 | 0 to 2 | Randomness of the voice |
| `talker_repetition_penalty` | 1.05 | 1 or more | Discourage repeated sounds in the voice |

Fields you leave out keep the server defaults, which come from the `sampling` section of `examples/full_duplex/minicpmo.yaml`. Unknown fields are rejected.

### Camera frames

Send JPEG or PNG frames while audio is flowing, and the model sees them together with the audio of the same unit:

```json
{
  "type": "sglang.input_image.append",
  "event_id": "frame-1",
  "image": "<base64 JPEG or PNG>",
  "sglang": {"t_ms": 1500}
}
```

`t_ms` is the frame's time on the audio timeline, counted from the first audio sample. A frame is at most 512 KiB and 4096 × 4096 pixels. By default a session accepts up to 4 frames per unit (1 s of audio); a frame whose unit has already been processed is rejected.

For more detail in each frame, ask for HD slicing before the first audio packet:

```json
{
  "type": "session.update",
  "event_id": "vision",
  "session": {"sglang": {"max_slice_nums": 4}}
}
```

A higher slice count lowers the number of frames accepted per unit. The reply to `session.update` reports the limit in `sglang.granted.input_image_format.max_frames_per_unit`.

### Server limits

These are set in `examples/full_duplex/minicpmo.yaml`:

| Setting | Default | Meaning |
|---|---|---|
| `max_sessions` | 2 | Conversations served at the same time |
| `reference_audio` | checkpoint default | Voice used when a session sends no reference |
| `vision.max_frames_per_unit` | 4 | Frames accepted per unit (1 s of audio) |
| `vision.max_slice_nums_limit` | 9 | Highest slice count a session may request |

One conversation can hold 8192 tokens of history, which is the model's limit. When a conversation fills it, the server sends a `context_exhausted` error and closes the session. Start a new session to continue.

For repeatable output, start the server from `examples/full_duplex/minicpmo-parity.yaml`, which uses greedy sampling.
