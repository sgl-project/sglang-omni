# LLaDA2.0-Uni

[LLaDA2.0-Uni](https://huggingface.co/inclusionAI/LLaDA2.0-Uni) is a multimodal model that accepts text and image input. This SGLang-Omni cookbook covers the experimental text-output serving path.

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

LLaDA2.0-Uni runs a 4-stage pipeline
(`preprocessing → image_encoder → thinker → decode`). The thinker disables CUDA
graph by default for this experimental DLLM path.

```bash
sgl-omni serve --model-path inclusionAI/LLaDA2.0-Uni --port 8000
```

### Intel XPU

The thinker's bf16 weights are ~30 GB, so on 24 GB cards it has to be sharded.
A TP stage must own its own process, and `tp_size` needs a `gpu` list of exactly
that many card ids (indices into the visible set, so `ZE_AFFINITY_MASK` chooses
the physical cards):

```bash
ZE_AFFINITY_MASK=0,1 sgl-omni serve \
  --model-path inclusionAI/LLaDA2.0-Uni --port 8000 \
  --thinker.process thinker --thinker.gpu '[0,1]' --thinker.tp_size 2 \
  --thinker.gpu_memory_fraction 0.80 --image_encoder.gpu_memory_fraction 0.12
```

Three things are decided by the platform rather than by this command line:

- **The attention backend is triton.** A dLLM denoises a whole block at once, so
  its attention must stay bidirectional. triton honours the model's
  `AttentionType.ENCODER_ONLY`; `intel_xpu` derives causality from
  `is_cross_attention` alone and would mask each block's later positions,
  returning wrong tokens with no error.
- **Decode graphs stay off.** Every forward in this pipeline is a
  `ForwardMode.DLLM_EXTEND`, and no attention backend reachable on XPU can put
  that mode in a graph: the triton backend raises while building its graph
  metadata, and `intel_xpu` asserts `is_decode_or_idle()`.
  Only flashinfer and flashattention handle it, both CUDA-only — which is why
  SGLang's own dLLM pass switches to flashinfer once capture is on. Turning
  capture on here fails at startup rather than running slowly.
- **One request denoises at a time.** A batched dLLM round returns fluent
  nonsense for some of its requests here: eight concurrent requests garbled at
  least one reply in 3 of 6 trials, and the dirty trials were also the slow ones.
  Concurrent requests are still accepted and still pipelined against the image
  encoder — they queue for the thinker instead of sharing its forward, which
  cleared 13 of 13 trials at no measurable cost in wall-clock at this
  concurrency.

Measured on two Intel Arc Pro B60 cards: ~15-20 tok/s on a warm 64-token text
reply, and 8 concurrent 64-token replies in 63-93s. The first request after
startup is several times slower (~3 tok/s) while triton autotunes, so time a
second one before drawing conclusions.

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
| `modalities` | list | `["text"]` | Output modalities (only `["text"]` is supported) |
| `images` | list | `null` | List of image file paths (local paths or URLs) |
| `max_tokens` | int | `null` | Maximum number of tokens to generate |

### Incoming Features

- Text-to-image generation
- Text-to-Image Generation with Thinking
- Interleaved Generation

## Known Limitations

- Text output is supported for text and image input. Image generation and
  interleaved generation are not wired to the OpenAI-compatible response path
  yet.
