# Omni Model Usage

This guide uses [Qwen3-Omni](https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct) as an example omni model with SGLang-Omni and the OpenAI-compatible API.

## Prerequisites

- Install SGLang-Omni by following [Installation](../get_started/installation.md).

## Launch the Server

Qwen3-Omni currently has two practical usage modes:

- **Text-first mode** for text output
- **Speech mode** for text + audio output

### Text-first mode

Text-first mode only produces text output. It runs the following stages:

 - **preprocessing**: tokenizes user input and resolves media inputs
 - **image_encoder**: encodes images and video frames into embeddings
 - **audio_encoder**: encodes audio input into embeddings
 - **mm_aggregate**: gathers preprocessing and encoder outputs and prepares thinker inputs
 - **thinker**: the core LLM that generates text tokens
 - **decode**: decodes token ids into text

You can try it directly without starting an HTTP server. It supports text, image, video, and audio inputs:

```bash
python examples/run_qwen3_omni_text_first.py \
  --model-path Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --prompt "What is in this image?" \
  --image-path /path/to/image.jpg
```

The result is printed directly to the terminal as the raw pipeline output.

Or launch it as an HTTP server via `sgl-omni serve`:

```bash
sgl-omni serve \
  --model-path Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --model-name qwen3-omni
```

### Speech mode

Speech mode builds on text-first mode by adding 3 extra stages after the thinker, enabling both text and audio output:

- **talker_ar**: generates speech tokens from thinker hidden states
- **code_predictor**: converts speech tokens into codec codes
- **code2wav**: decodes codec codes into audio waveforms

The example below uses 3 GPUs. This places the thinker on GPU 0, the talker on GPU 1, and the code predictor on GPU 2. The image encoder, audio encoder, and vocoder default to GPU 0.

```bash
python examples/run_qwen3_omni_speech_server.py \
  --model-path Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --gpu-thinker 0 \
  --gpu-talker 1 \
  --gpu-code-predictor 2
```

## Text Chat

### Curl

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

The response follows the OpenAI chat completion shape. In the common case, the text is in `choices[0].message.content`.

### OpenAI SDK

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

## Streaming

Set `"stream": true` to receive Server-Sent Events.

### Curl

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

A few details matter here:

- the response type is `text/event-stream`
- the first chunk may contain only `role="assistant"`
- the stream ends with `data: [DONE]`
- `usage` is attached to the final completion chunk

### OpenAI SDK

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

## Multi-modal Input

Use the `images`, `videos`, and `audios` fields to pass multi-modal inputs alongside your text prompt. Each field accepts a list of local file paths or HTTP(S) URLs.

### Curl

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

### OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")

resp = client.chat.completions.create(
    model="qwen3-omni",
    messages=[
        {"role": "user", "content": "Describe the image, the video, and the audio."}
    ],
    max_tokens=256,
    extra_body={
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
    },
)

print(resp.choices[0].message.content)
```

## Speech Output

To receive both text and audio in a single response, set `"modalities": ["text", "audio"]`.

### Curl

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

### OpenAI SDK

The audio data in the response is base64-encoded WAV by default. Decode it and save to a file:

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
    extra_body={
        "modalities": ["text", "audio"],
    },
)

message = resp.choices[0].message
audio = message.audio

Path("reply.wav").write_bytes(base64.b64decode(audio.data))
print(message.content or "")
print(audio.transcript or "")
```

## Request Parameters

The table below lists the request fields accepted by `POST /v1/chat/completions`.

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `model` | string | `null` | Model name. If omitted, the server uses the active model. |
| `messages` | list | (required) | Chat history in OpenAI message format. |
| `temperature` | float | `1.0` | Sampling temperature. |
| `top_p` | float | `1.0` | Top-p sampling. |
| `top_k` | int | `-1` | Top-k sampling. |
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
