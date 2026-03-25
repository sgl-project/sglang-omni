# TTS with Fish Speech S2-Pro

This guide walks you through serving the [Fish Speech S2-Pro](https://huggingface.co/fishaudio/s2-pro) text-to-speech model with SGLang-Omni and generating speech via the OpenAI-compatible API.

## Prerequisites

We recommend using the official Docker image, which bundles all system-level dependencies.

```bash
docker pull frankleeeee/sglang-omni:dev
docker run -it --shm-size 32g --gpus all frankleeeee/sglang-omni:dev /bin/zsh
```

Inside the container, install SGLang-Omni with S2-Pro support and download the model weights:

```bash
git clone https://github.com/sgl-project/sglang-omni.git
cd sglang-omni
uv venv .venv -p 3.12 && source .venv/bin/activate
uv pip install -v ".[s2pro]"
huggingface-cli download fishaudio/s2-pro
```

## Launch the Server

Start the TTS server with the following command:

```bash
sgl-omni serve \
  --model-path fishaudio/s2-pro \
  --config examples/configs/s2pro_tts.yaml \
  --port 8000
```

Wait for the server to be ready, then verify with a health check:

```bash
curl -s http://localhost:8000/health
```

## Basic Text-to-Speech

Generate speech from text without any reference audio:

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
    -H "Content-Type: application/json" \
    -d '{"input": "Hello, how are you?"}' \
    --output output.wav
```

> **Note:** Without reference audio, the generated voice will sound robotic. For natural-sounding results, use voice cloning with a reference audio clip.

## Voice Cloning

Provide a reference audio file and its transcript to clone a specific voice:

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
    -H "Content-Type: application/json" \
    -d '{
        "input": "Hello, how are you?",
        "references": [{"audio_path": "ref.wav", "text": "Transcript of ref audio."}]
    }' \
    --output output.wav
```

The `references` field accepts a list of objects, each containing:
- `audio_path` -- path to the reference audio file.
- `text` -- transcript of the reference audio.

## Streaming

Enable streaming to receive audio chunks in real time via Server-Sent Events (SSE):

```bash
curl -N http://localhost:8000/v1/audio/speech \
    -H "Content-Type: application/json" \
    -d '{"input": "Hello, how are you?", "stream": true}'
```

The server returns a stream of SSE events. Each event contains an `audio.speech.chunk` object with a base64-encoded audio chunk. The stream ends with `data: [DONE]`.

## Python Client Examples

### Basic TTS

```python
import requests

resp = requests.post(
    "http://localhost:8000/v1/audio/speech",
    json={"input": "Hello, how are you?"},
)
with open("output.wav", "wb") as f:
    f.write(resp.content)
```

### Voice Cloning

```python
import requests

resp = requests.post(
    "http://localhost:8000/v1/audio/speech",
    json={
        "input": "Hello, how are you?",
        "references": [{"audio_path": "ref.wav", "text": "Transcript of ref audio."}],
    },
)
with open("output.wav", "wb") as f:
    f.write(resp.content)
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
| `references` | list | `null` | Reference audio for voice cloning; each item has `audio_path` and `text` |
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
