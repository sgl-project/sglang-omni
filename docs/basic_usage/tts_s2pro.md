# TTS Model Usage

This guide uses [Fish Speech S2-Pro](https://huggingface.co/fishaudio/s2-pro) as an example TTS (text-to-speech) model with SGLang-Omni and the OpenAI-compatible API.

## Prerequisites

```bash
docker pull frankleeeee/sglang-omni:dev
docker run -it --shm-size 32g --gpus all frankleeeee/sglang-omni:dev /bin/zsh
```

```bash
git clone https://github.com/sgl-project/sglang-omni.git
cd sglang-omni
uv venv .venv -p 3.12 && source .venv/bin/activate
uv pip install -v .
hf download fishaudio/s2-pro
```

## Launch the Server

```bash
sgl-omni serve \
  --model-path fishaudio/s2-pro \
  --config examples/configs/s2pro_tts.yaml \
  --port 8000
```

## Use Curl

Generate speech from text without any reference audio:

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
    -H "Content-Type: application/json" \
    -d '{"input": "Hello, how are you?"}' \
    --output output.wav
```

Note that without reference audio, the generated voice will sound robotic. For natural-sounding results, use Voice Cloning with a reference audio clip.

### Voice Cloning

The examples below use a sample clip from [`seed-tts-eval-mini`](https://huggingface.co/datasets/zhaochenyang20/seed-tts-eval-mini). The `references` field accepts `audio_path` (a local path or HTTP URL) and `text` (transcript of that audio).

1. Non-streaming request

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Get the trust fund to the bank early.",
    "references": [{
      "audio_path": "https://huggingface.co/datasets/zhaochenyang20/seed-tts-eval-mini/resolve/main/en/prompt-wavs/common_voice_en_10119832.wav",
      "text": "We asked over twenty different people, and they all said it was his."
    }]
  }' \
  --output output.wav
```

2. Streaming

Enable streaming to receive audio chunks in real time via Server-Sent Events (SSE). Set `"stream": true`:

```bash
curl -N -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Get the trust fund to the bank early.",
    "references": [{
      "audio_path": "https://huggingface.co/datasets/zhaochenyang20/seed-tts-eval-mini/resolve/main/en/prompt-wavs/common_voice_en_10119832.wav",
      "text": "We asked over twenty different people, and they all said it was his."
    }],
    "stream": true
  }'
```

The server returns a stream of SSE events. Each event contains an `audio.speech.chunk` object with a base64-encoded audio chunk. The stream ends with `data: [DONE]`.

## Use Python

### Basic TTS

```python
import requests

resp = requests.post(
    "http://localhost:8000/v1/audio/speech",
    json={"input": "Hello, how are you?"},
)
resp.raise_for_status()
with open("output.wav", "wb") as f:
    f.write(resp.content)
```

### Voice Cloning

```python
REFERENCE_AUDIO = "https://huggingface.co/datasets/zhaochenyang20/seed-tts-eval-mini/resolve/main/en/prompt-wavs/common_voice_en_10119832.wav"
REFERENCE_TEXT = "We asked over twenty different people, and they all said it was his."
SPEECH_INPUT = "Get the trust fund to the bank early."
```

1. Non-streaming Request

```python
import requests

resp = requests.post(
    "http://localhost:8000/v1/audio/speech",
    json={
        "input": SPEECH_INPUT,
        "references": [{"audio_path": REFERENCE_AUDIO, "text": REFERENCE_TEXT}],
    },
)
resp.raise_for_status()
with open("output.wav", "wb") as f:
    f.write(resp.content)
```

2. Streaming Request

```python
import base64, io, json, wave

import requests

payload = {
    "input": SPEECH_INPUT,
    "references": [{"audio_path": REFERENCE_AUDIO, "text": REFERENCE_TEXT}],
    "stream": True,
    "response_format": "wav",
}

chunks = []
fmt = None
with requests.post(
    "http://localhost:8000/v1/audio/speech",
    json=payload,
    stream=True,
    timeout=600,
) as stream:
    stream.raise_for_status()
    for line in stream.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data: "):
            continue
        data = line[len("data:"):].lstrip()
        if data == "[DONE]":
            break
        b64 = (json.loads(data).get("audio") or {}).get("data")
        if not b64:
            continue
        with wave.open(io.BytesIO(base64.b64decode(b64)), "rb") as w:
            if fmt is None:
                fmt = w.getnchannels(), w.getsampwidth(), w.getframerate()
            chunks.append(w.readframes(w.getnframes()))

assert fmt
nc, sw, fr = fmt
with wave.open("output_stream.wav", "wb") as w:
    w.setnchannels(nc)
    w.setsampwidth(sw)
    w.setframerate(fr)
    w.writeframes(b"".join(chunks))
```

## Request Parameters

The table below lists all parameters accepted by the `/v1/audio/speech` endpoint.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `input` | string | (required) | Text to synthesize |
| `voice` | string | `"default"` | Voice identifier |
| `response_format` | string | `"wav"` | Output audio format |
| `speed` | float | `1.0` | Playback speed multiplier |
| `stream` | bool | `false` | Enable streaming via SSE |
| `references` | list | `null` | Reference audio for voice cloning; each item has `audio_path` (local path / remote url) and `text` |
| `max_new_tokens` | int | `null` | Maximum number of generated tokens |
| `temperature` | float | `null` | Sampling temperature |
| `top_p` | float | `null` | Top-p sampling |
| `top_k` | int | `null` | Top-k sampling |
| `repetition_penalty` | float | `null` | Repetition penalty |
| `seed` | int | `null` | Random seed for reproducibility |

## Interactive Playground

SGLang-Omni ships with a Gradio-based playground for interactive TTS experimentation:

```bash
./playground/tts/start.sh
```

The playground now exposes two demo modes against the same S2 Pro backend:

- `Non-Streaming` starts a standard request and shows the final WAV after generation finishes.
- `Streaming` consumes the `/v1/audio/speech` SSE stream, starts playback from incremental WAV chunks, and also writes a final combined WAV artifact for inspection.

The launcher starts the backend first, waits for `/health`, then starts the Gradio UI with:

```bash
python -m playground.tts.app --api-base http://localhost:8000
```

A demo play video is available [here](https://x.com/lmsysorg/status/2031412267213008984/video/1). We highly recommend using playground since audio data is hard to intertact with by CLI.
