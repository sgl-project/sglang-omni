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

The CFG thinker uses synchronous execution and defaults to eager mode.
CUDA Graph replay is available for fixed-width DLLM blocks, including
multi-way CFG and tensor parallel execution.

```bash
sgl-omni serve --model-path inclusionAI/LLaDA2.0-Uni --port 8000
```

### Thinker TP and CUDA Graphs

Enable graphs alongside TP to reduce the repeated model-launch overhead of
DLLM unmasking. Small blocks can make TP slower in eager mode: each rank does
less compute, while Python dispatch and the per-layer collectives remain.
Graphs capture the model forward, including graph-compatible collectives;
they do not capture the Python sampling loop or remove communication.

```bash
CUDA_VISIBLE_DEVICES=0,1 sgl-omni serve \
  --model-path inclusionAI/LLaDA2.0-Uni --port 8000 \
  --thinker.tp_size 2 --thinker.gpu '[0, 1]' \
  --thinker.engine.disable_cuda_graph false \
  --thinker.engine.cuda_graph_bs '[1, 2, 3, 4]'
```

Graph batch sizes count CFG branches, not user requests. Two-way T2I CFG
uses two rows and three-way edit CFG uses three. The graph runner is scoped
to the LLaDA-Uni CFG thinker. Blocks with padding in the current query stay
eager; once that padding is entirely in the cached prefix, replay excludes
it from cached attention. Dynamic KV lengths and token values are refreshed
before replay. This uses the existing SGLang operators without adding custom
Triton kernels. Set `disable_cuda_graph: true` for the eager comparison, and
exclude model loading, graph capture, and initial warmup requests from timing.

Group-limited expert routing uses SGLang's `TopK` component in both eager and
graph execution. It preserves FP32 router logits, sigmoid scoring, correction
bias, and normalized routing weights. The fused implementation can change
expert ordering and floating-point rounding, so validate model accuracy rather
than expecting pixel-identical images across routing implementations.

LLaDA-Uni thinker workers retain the same `CUDA_VISIBLE_DEVICES` list across
TP ranks and select distinct local devices. The stage defaults
`SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS=false` so custom all-reduce v2 can
identify each GPU when initializing symmetric memory. Other models retain
their existing placement defaults. Check startup logs to confirm custom
all-reduce initialization when comparing TP performance. The existing
`SGLANG_OPT_USE_CUSTOM_ALL_REDUCE_V2=0` environment setting selects legacy
custom all-reduce for a separate comparison; this pipeline does not change
the global communication default.

## Image Generation and Editing

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
