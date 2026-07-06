# LLaDA2.0-Uni

[LLaDA2.0-Uni](https://huggingface.co/inclusionAI/LLaDA2.0-Uni) is a multimodal model that accepts text and image input, and can return text or non-streaming image output.

## Highlights

- Unified dLLM-MoE Backbone — Built on LLaDA 2.0, unifying multimodal understanding and generation.
- Top-Tier Understanding & Generation — Matches dedicated VLMs in visual QA and document understanding, while generating high-quality images.
- Interleaved Generation & Reasoning — Empowered by unified discrete representations, unlocking interleaved generation and reasoning.

## Architecture

![LLaDA2.0-Uni Architecture](../_static/image/llada2.0_uni_architecture.png)

LLaDA2.0-Uni unifies multimodal understanding and generation into a simple Mask Token Prediction paradigm. Visual inputs are encoded by the SigLIP-VQ tokenizer into discrete semantic tokens, then mapped alongside text tokens to backbone hidden states under a unified mask prediction objective. Output tokens are decoded back to text via the Text De-Tokenizer, or reconstructed into high-fidelity images through the Diffusion Decoder. Empowered by unified discrete representations, it effortlessly handles complex interleaved generation and unlocks advanced interleaved reasoning, interleaving <|image|>...<|/image|> chunks to enable end-to-end training and inference within a single coherent framework.

## Prerequisites

Install `sglang-omni` by following [Installation](../get_started/installation.md).

## Server Configuration

LLaDA2.0-Uni runs a 5-stage pipeline
(`preprocessing → image_encoder → thinker → image_decode → decode`) on a single
GPU. Text requests use the SGLang dLLM thinker path. Image-output requests use
the model checkpoint's upstream image-token generator in the thinker stage, then
decode the generated VQ tokens to PNG/JPEG bytes in `image_decode`.

```bash
sgl-omni serve --model-path inclusionAI/LLaDA2.0-Uni --port 8000
```

## Text Input

Send a text-only prompt and get a text response.

**cURL**

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "inclusionAI/LLaDA2.0-Uni",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 256
  }'
```

**Python**

```python
import requests

resp = requests.post(
    "http://localhost:8000/v1/chat/completions",
    json={
        "model": "inclusionAI/LLaDA2.0-Uni",
        "messages": [{"role": "user", "content": "Hello!"}],
        "max_tokens": 256,
    },
)
resp.raise_for_status()
result = resp.json()
print(result["choices"][0]["message"]["content"])
```

## Image and Text Input

Send an image with a text prompt to get a text response.

**cURL**

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "inclusionAI/LLaDA2.0-Uni",
    "messages": [{"role": "user", "content": "Briefly describe the cars in this image."}],
    "images": ["tests/data/cars.jpg"],
    "modalities": ["text"],
    "max_tokens": 16
  }'
```

**Python**

```python
import requests

resp = requests.post(
    "http://localhost:8000/v1/chat/completions",
    json={
        "model": "inclusionAI/LLaDA2.0-Uni",
        "messages": [{"role": "user", "content": "Briefly describe the cars in this image."}],
        "images": ["tests/data/cars.jpg"],
        "modalities": ["text"],
        "max_tokens": 16,
    },
)
resp.raise_for_status()
result = resp.json()
print(result["choices"][0]["message"]["content"])
```

Images can also be passed inline using the OpenAI multi-content format:

```python
import requests

resp = requests.post(
    "http://localhost:8000/v1/chat/completions",
    json={
        "model": "inclusionAI/LLaDA2.0-Uni",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "tests/data/cars.jpg"}},
                    {"type": "text", "text": "Briefly describe the cars in this image."},
                ],
            }
        ],
        "modalities": ["text"],
        "max_tokens": 16,
    },
)
resp.raise_for_status()
result = resp.json()
print(result["choices"][0]["message"]["content"])
```

Text-to-image requests return base64-encoded PNG/JPEG image payloads:

```python
import base64
import requests

resp = requests.post(
    "http://localhost:8000/v1/chat/completions",
    json={
        "model": "inclusionAI/LLaDA2.0-Uni",
        "messages": [{"role": "user", "content": "Draw a lighthouse at dusk."}],
        "modalities": ["image"],
        "image": {
            "width": 512,
            "height": 512,
            "steps": 8,
            "cfg_scale": 2.0,
            "decode_mode": "decoder-turbo",
            "format": "png",
        },
    },
)
resp.raise_for_status()
image = resp.json()["choices"][0]["message"]["images"][0]
with open("llada2_output.png", "wb") as f:
    f.write(base64.b64decode(image["data"]))
```

## Request Parameters

The table below lists all parameters accepted by the `/v1/chat/completions` endpoint for LLaDA2.0-Uni.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model` | string | `null` | Model identifier |
| `messages` | list | (required) | List of chat messages, each with `role` and `content` |
| `modalities` | list | `["text"]` | Output modalities; use `["image"]` for text-to-image responses |
| `image` | object | `null` | Image-output generation settings such as `width`, `height`, `steps`, `cfg_scale`, `decoder_steps`, `decode_mode`, and `format` |
| `images` | list | `null` | List of image file paths (local paths or URLs) |
| `max_tokens` | int | `null` | Maximum number of tokens to generate |

### Incoming Features

- Image editing
- Interleaved Generation
- Text-to-image generation with thinking

## Known Limitations

- Text output is supported for text and image input. Image output currently
  supports text-to-image generation and returns non-streaming base64 PNG/JPEG
  payloads. Image editing, image-output streaming, and interleaved generation
  are not wired yet.
- The image-token generator and image decoder are loaded lazily, but image
  output can require substantially more GPU memory than text-only serving. For
  mixed text and image workloads, budget for both thinker paths or run separate
  workers behind the router.
