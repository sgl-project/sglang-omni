# LLaDA2.0-Uni

[LLaDA2.0-Uni](https://huggingface.co/inclusionAI/LLaDA2.0-Uni) accepts text and image input and supports text output, text-to-image generation, and image editing.

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

The default `omni` pipeline runs preprocessing, image encoding, and the
DLLM thinker, then routes to text and image decoders. The image decoder
uses diffusers' ZImage backbone, SigVQ conditioning, and a VAE. This is
LLaDA2-Uni's semantic decoder, not the LLaDA-Image text-conditioned model.
The `text` variant retains the four-stage text-output pipeline.

The CFG thinker currently uses synchronous eager execution. An explicit
CUDA graph request is rejected until CFG graph metadata is supported.

```bash
sgl-omni serve --model-path inclusionAI/LLaDA2.0-Uni --port 8000
```

## Image Generation and Editing

### Native Decoder Sequence Parallelism

The native SGLang adapter supports single-rank decoding and opt-in sequence
parallelism. The thinker TP setting is independent of decoder SP. Set the
following stage overrides in the pipeline YAML for two decoder GPUs:

```yaml
stages:
  image_decode:
    gpu: [1, 2]
    sp_size: 2
    factory:
      backend: sglang
      attention_backend: torch_sdpa
      ulysses_degree: 2
      ring_degree: 1
```

GPU IDs are relative to the parent's visible devices. Each decoder rank owns
one process; SP followers do not participate in the thinker's TP/KV groups.
SigVQ conditioning and VAE/image encoding run on rank zero. All ranks use the
same noise seed and SGLang's spatial sharding, RoPE, attention, and gather.

| Configuration | `gpu` | `sp_size` | Ulysses | Ring | Attention backend |
| --- | --- | --- | --- | --- | --- |
| Native SP1 | `1` | 1 | 1 | 1 | `torch_sdpa` |
| Native SP2 Ulysses | `[1, 2]` | 2 | 2 | 1 | `torch_sdpa` |
| Native SP2 ring | `[1, 2]` | 2 | 1 | 2 | `torch_sdpa` is unsupported |

SGLang's ring path requires `fa` or `sage_attn`; configuration rejects SDPA
with ring instead of changing the backend. A ring comparison must also run
SP1 and Ulysses with the same explicitly selected backend. Ulysses degree
must divide the checkpoint's attention head count. Spatial layouts that add
learned padding tokens relative to SP1 are rejected to preserve semantics.

The opt-in GPU test checks a small checkpoint against diffusers and verifies
the instantiated attention backend. It does not replace full server validation:

```bash
LLADA_DECODER_GPU_TEST=1 python -m pytest -q \
  tests/unit_test/llada2_uni/test_decoder_native_gpu.py
```

For generation/performance comparisons, use the same checkpoint, request,
seed, resolution, CFG, dtype, decode mode and step count. Warm each server
before timing; report request latency separately from decoder GPU time.

### Requests

Send non-streaming requests with `modalities: ["image"]`. `dllm_steps` controls
VQ token generation; `decoder_steps` controls diffusion sampling.

```python
import base64
from pathlib import Path

import requests

request = {
    "model": "inclusionAI/LLaDA2.0-Uni",
    "messages": [{"role": "user", "content": "A sailboat on a calm lake."}],
    "modalities": ["image"],
    "stream": False,
    "image_generation": {
        "mode": "normal",
        "decode_mode": "decoder-turbo",
        "decoder_steps": 8,
        "dllm_steps": 8,
        "cfg_scale": 4.0,
        "seed": 42,
    },
}
response = requests.post(
    "http://localhost:8000/v1/chat/completions", json=request, timeout=600
)
response.raise_for_status()
image = response.json()["choices"][0]["message"]["image"]
Path("generated.png").write_bytes(base64.b64decode(image["data"]))
```

For editing, use an instruction such as `"Change the background to a beach."`
and include one source `image_url` content item alongside the text in the user
message. Set `cfg_text_scale` and `cfg_image_scale` in `image_generation` to
control editing guidance. Omitting those values retains task-specific defaults.
For text-to-image requests, set `mode: "thinking"` to generate a description
before the image tokens. Use `modalities: ["text", "image"]` to return both
the thinking text and the image; `["image"]` returns only the image.
The text pass has a 2048-token budget and stops at `<boi>`. The image pass
retains the generated context and applies CFG to the VQ tokens. Both passes
must fit the thinker's configured context length. Thinking mode does not
support editing.

The server selects a patch-aligned source grid near a 512x512 pixel budget,
then resizes proportionally and center-crops the image to that grid. Small
images are enlarged without black padding; images already matching the grid
are preserved. Aspect ratios beyond 4:1 or 1:4 require additional cropping.
Image understanding retains its separate preprocessing and pixel budgets.

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

## Request Parameters

The table below lists all parameters accepted by the `/v1/chat/completions` endpoint for LLaDA2.0-Uni.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model` | string | `null` | Model identifier |
| `messages` | list | (required) | List of chat messages, each with `role` and `content` |
| `modalities` | list | `["text"]` | Use `["text"]` for understanding or `["image"]` for generation/editing |
| `image_generation` | object | `null` | Image generation options shown above |
| `images` | list | `null` | List of image file paths (local paths or URLs) |
| `max_tokens` | int | `null` | Maximum number of tokens to generate |

### Incoming Features

- Interleaved Generation

## Known Limitations

- Image generation and editing return one image per non-streaming request.
- Interleaved generation is not supported.
