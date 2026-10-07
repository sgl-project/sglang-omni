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

Full duplex has been tested on one H100 and one H200.

Two configs are provided. Pass either with `--config`:

| Config | Use it for |
|---|---|
| `examples/full_duplex/minicpmo.yaml` | Normal serving. Sampling matches the MiniCPM-o demo |
| `examples/full_duplex/minicpmo-parity.yaml` | Repeatable output for regression and parity recordings. Differs only in greedy sampling and `top_k: 100` |

| Setting | Default | Meaning |
|---|---|---|
| `max_sessions` | 2 | Conversations at the same time. Further connections get HTTP 503 |
| `reference_audio` | checkpoint default | Voice used when a session sends no reference |
| `speech_state_bytes_per_session` | 2 GiB | Memory the speech stage may hold per conversation. A conversation that needs more is closed and the others keep running |
| `sampling` | see the config | Default sampling when a session does not set its own |
| `vision` | see the config | Camera-frame limits per unit (1 s of audio) |

A session holds at most 8192 tokens of history, the model's limit. When that fills, the server sends `context_exhausted` and closes the session.

## Browser demo

With the full-duplex server running, start the demo page in another terminal:

```bash
python playground/realtime/app.py --api-base http://127.0.0.1:8000 --port 8080
```

The page warms up first. Once the terminal prints `Running on`, open <http://localhost:8080> and click **Start talking** to talk through your microphone; the camera can be turned on during the call. If the server is remote, first run `ssh -N -L 8080:127.0.0.1:8080 USER@SERVER` on your own machine. Open the page through `localhost`, or the browser will not grant microphone access.

The settings panel switches between English and Chinese, changes the voice and adjusts sampling. The download button in the top bar saves the conversation trace; attach it when reporting a problem.

## Voice cloning

The chat endpoint follows the OpenAI Chat Completions API. Pass a reference recording in `audio.ref_audio`, and the reply is spoken in that voice:

```python
import base64
from pathlib import Path

from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")
reference = base64.b64encode(Path("docs/_static/audio/male-voice.wav").read_bytes()).decode("ascii")
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

The reference must be a base64 data URI; file paths and URLs are not fetched. Without it, the model uses its default voice.

## Full-duplex protocol

Clients connect to `/v1/realtime` over WebSocket. They send 16 kHz mono PCM16 audio in `input_audio_buffer.append`; the server returns 24 kHz audio in `response.output_audio.delta` and text in `response.output_audio_transcript.delta`. The model decides once per second of audio whether to keep listening or to speak.

1. After `session.created`, send `session.update` and wait for `session.updated`.
2. Send audio with `input_audio_buffer.append` at the pace it is captured; replies arrive while you send.
3. When the audio ends, send `sglang.input_audio.end`, wait for `sglang.input_audio.drained`, then send `session.close`.

For a Python client, see `warmup` in `playground/realtime/app.py`.

Session settings go in the `sglang` field of `session.update`, before the first audio packet, and stay fixed for the session:

| Setting | Field | Notes |
|---|---|---|
| Voice | `reference_audio` | `{"media_type": "audio/wav", "data": "<base64>"}`, a PCM16 WAV of at most 30 s and 1 MiB; `tts_reference_audio` changes only the output voice |
| Sampling | `sampling` | For example `temperature`, `top_p` and `listen_prob_scale`; unset fields keep the defaults in `examples/full_duplex/minicpmo.yaml` |
| Image detail | `max_slice_nums` | Higher is sharper but accepts fewer frames per second |

Send camera frames with `sglang.input_image.append`: a base64 JPEG or PNG in `image`, and its position on the audio timeline in `sglang.t_ms`. By default up to 4 frames per second are accepted; `session.updated` reports the actual limit.
