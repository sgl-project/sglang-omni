# Omni Model Usage

This guide uses [Qwen3-Omni](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct) as an example omni model with SGLang-Omni and the OpenAI-compatible API.

Hugging Face assets below use the **`hf download`** command from `huggingface-hub` (the old `huggingface-cli download` name is deprecated).

## Prerequisites

```bash
docker pull frankleeeee/sglang-omni:dev
docker run -it --shm-size 32g --gpus all frankleeeee/sglang-omni:dev /bin/zsh
```

```bash
git clone https://github.com/sgl-project/sglang-omni.git
cd sglang-omni
uv venv .venv -p 3.12 && source .venv/bin/activate
uv pip install -e .
hf download Qwen/Qwen3-Omni-30B-A3B-Instruct
```

## Launch the Server

Qwen3-Omni has two practical usage modes: **text-first mode** for text output and **speech mode** for text + audio output.

### Text-first mode

Text-first mode produces text only. It runs the following stages:

- **preprocessing**: tokenizes user input and resolves media inputs
- **image_encoder**: encodes images and video frames into embeddings
- **audio_encoder**: encodes audio input into embeddings
- **mm_aggregate**: gathers preprocessing and encoder outputs and prepares thinker inputs
- **thinker**: the core LLM that generates text tokens
- **decode**: decodes token ids into text

```bash
sgl-omni serve \
  --model-path Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --model-name qwen3-omni
```

### Speech mode

Speech mode adds 3 extra stages after the thinker, enabling both text and audio output:

- **talker_ar**: generates speech tokens from thinker hidden states
- **code_predictor**: converts speech tokens into codec codes
- **code2wav**: decodes codec codes into audio waveforms

The example below uses 3 GPUs, placing the thinker on GPU 0, the talker on GPU 1, and the code predictor on GPU 2:

```bash
python examples/run_qwen3_omni_speech_server.py \
  --model-path Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --gpu-thinker 0 \
  --gpu-talker 1 \
  --gpu-code-predictor 2
```

## Use Curl

### Text Chat

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-omni",
    "messages": [
      {"role": "user", "content": "Hello! Give me a one-sentence greeting."}
    ],
    "modalities": ["text"],
    "max_tokens": 128
  }'
```

The response follows the OpenAI chat completion shape. The text is in `choices[0].message.content`.

### Streaming

Set `"stream": true` to receive Server-Sent Events:

```bash
curl -N http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-omni",
    "messages": [
      {"role": "user", "content": "Write a short greeting."}
    ],
    "modalities": ["text"],
    "stream": true
  }'
```

A few details:

- the response type is `text/event-stream`
- the first chunk may contain only `role="assistant"`
- the stream ends with `data: [DONE]`
- `usage` is attached to the final completion chunk

### Multi-modal Input

Use the `images`, `videos`, and `audios` fields to pass multi-modal inputs alongside your text prompt. Each field accepts a list of local file paths or HTTP(S) URLs.

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-omni",
    "messages": [
      {"role": "user", "content": "Describe the image, the video, and the audio."}
    ],
    "images": [
      "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/demo/cars.jpg"
    ],
    "videos": [
      "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/demo/draw.mp4"
    ],
    "audios": [
      "https://huggingface.co/datasets/zhaochenyang20/seed-tts-eval-mini/resolve/main/en/prompt-wavs/common_voice_en_10119832.wav"
    ],
    "modalities": ["text"],
    "max_tokens": 256
  }'
```

### Speech Output

Set `"modalities": ["text", "audio"]` to receive both text and audio in a single response. The audio data is base64-encoded:

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3-omni",
    "messages": [
      {"role": "user", "content": "Say hello and return both text and audio."}
    ],
    "modalities": ["text", "audio"],
    "max_tokens": 128
  }'
```

## Use Python (OpenAI SDK)

### Text Chat (OpenAI SDK)

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")

resp = client.chat.completions.create(
    model="qwen3-omni",
    messages=[
        {"role": "user", "content": "Hello! Give me a one-sentence greeting."}
    ],
    max_tokens=128,
    extra_body={"modalities": ["text"]},
)

print(resp.choices[0].message.content)
```

### Streaming (OpenAI SDK)

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")

stream = client.chat.completions.create(
    model="qwen3-omni",
    messages=[{"role": "user", "content": "Write a short greeting."}],
    stream=True,
    extra_body={"modalities": ["text"]},
)

for chunk in stream:
    if not chunk.choices:
        continue
    delta = chunk.choices[0].delta
    if delta.content:
        print(delta.content, end="", flush=True)
print()
```

### Speech Output (OpenAI SDK)

```python
import base64
from pathlib import Path

from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")

resp = client.chat.completions.create(
    model="qwen3-omni",
    messages=[
        {"role": "user", "content": "Say hello and return both text and audio."}
    ],
    max_tokens=128,
    extra_body={"modalities": ["text", "audio"]},
)

message = resp.choices[0].message
print(message.content or "")
if message.audio:
    Path("reply.wav").write_bytes(base64.b64decode(message.audio.data))
```

## Use Python (requests)

### Text Chat (Python)

```python
import requests

resp = requests.post(
    "http://localhost:8000/v1/chat/completions",
    json={
        "model": "qwen3-omni",
        "messages": [{"role": "user", "content": "Hello! Give me a one-sentence greeting."}],
        "modalities": ["text"],
        "max_tokens": 128,
    },
)
resp.raise_for_status()
print(resp.json()["choices"][0]["message"]["content"])
```

### Streaming (Python)

```python
import json

import requests

with requests.post(
    "http://localhost:8000/v1/chat/completions",
    json={
        "model": "qwen3-omni",
        "messages": [{"role": "user", "content": "Write a short greeting."}],
        "modalities": ["text"],
        "stream": True,
    },
    stream=True,
    timeout=60,
) as resp:
    resp.raise_for_status()
    for line in resp.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data: "):
            continue
        payload = line[len("data: "):]
        if payload == "[DONE]":
            break
        chunk = json.loads(payload)
        delta = chunk["choices"][0]["delta"]
        if delta.get("content"):
            print(delta["content"], end="", flush=True)
print()
```

### Multi-modal Input (Python)

```python
import requests

resp = requests.post(
    "http://localhost:8000/v1/chat/completions",
    json={
        "model": "qwen3-omni",
        "messages": [
            {"role": "user", "content": "Describe the image, the video, and the audio."}
        ],
        "images": [
            "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/demo/cars.jpg"
        ],
        "videos": [
            "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen3-Omni/demo/draw.mp4"
        ],
        "audios": [
            "https://huggingface.co/datasets/zhaochenyang20/seed-tts-eval-mini/resolve/main/en/prompt-wavs/common_voice_en_10119832.wav"
        ],
        "modalities": ["text"],
        "max_tokens": 256,
    },
)
resp.raise_for_status()
print(resp.json()["choices"][0]["message"]["content"])
```

### Speech Output (Python)

```python
import base64

import requests

resp = requests.post(
    "http://localhost:8000/v1/chat/completions",
    json={
        "model": "qwen3-omni",
        "messages": [
            {"role": "user", "content": "Say hello and return both text and audio."}
        ],
        "modalities": ["text", "audio"],
        "max_tokens": 128,
    },
)
resp.raise_for_status()
body = resp.json()
message = body["choices"][0]["message"]
print(message.get("content", ""))
audio_data = message.get("audio", {}).get("data")
if audio_data:
    with open("reply.wav", "wb") as f:
        f.write(base64.b64decode(audio_data))
```

## Request Parameters

The table below lists the request fields accepted by `POST /v1/chat/completions`.

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `model` | string | `null` | Model name. If omitted, the server uses the active model. |
| `messages` | list | (required) | Chat history in OpenAI message format. |
| `temperature` | float | `1.0` | Sampling temperature. |
| `top_p` | float | `1.0` | Top-p sampling. |
| `top_k` | int | `null` | Top-k sampling. |
| `min_p` | float | `0.0` | Minimum probability cutoff. |
| `repetition_penalty` | float | `1.0` | Repetition penalty. |
| `max_tokens` | int | `null` | Maximum number of generated tokens. |
| `max_completion_tokens` | int | `null` | Alternative to `max_tokens`. If both are set, this one takes precedence. |
| `stop` | string or list | `null` | Stop sequence or list of stop sequences. |
| `seed` | int | `null` | Random seed for reproducibility. |
| `stream` | bool | `false` | Enable streaming via SSE. |
| `modalities` | list | `["text"]` | Requested output modalities such as `["text"]` or `["text", "audio"]`. |
| `audio` | object | `null` | Audio output config. Use `{"format": "wav"}` to control the returned audio format. |
| `audios` | list | `null` | Input audio file paths or URLs. |
| `images` | list | `null` | Input image file paths or URLs. |
| `videos` | list | `null` | Input video file paths or URLs. |
| `stage_sampling` | object | `null` | Per-stage sampling overrides. |
| `stage_params` | object | `null` | Per-stage runtime parameters. |
| `request_id` | string | `null` | Optional request identifier. |
| `user` | string | `null` | Optional end-user identifier. |
