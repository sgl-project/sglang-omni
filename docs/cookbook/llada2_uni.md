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
sgl-omni serve --model-path inclusionAI/LLaDA2.0-Uni --port 8000 \
  --thinker.engine.enable_torch_compile false
```

The cookbook explicitly disables `torch.compile` to match the validated
generation settings. CUDA Graph execution is controlled separately.

### Thinker TP and CUDA Graphs

Enable graphs alongside TP to reduce the repeated model-launch overhead of
DLLM unmasking. Small blocks can make TP slower in eager mode: each rank does
less compute, while Python dispatch and the per-layer collectives remain.
Graphs capture the model forward, including graph-compatible collectives;
they do not capture the Python sampling loop or remove communication.

```bash
CUDA_VISIBLE_DEVICES=0,1 sgl-omni serve \
  --model-path inclusionAI/LLaDA2.0-Uni --port 8000 \
  --thinker.process thinker --thinker.tp_size 2 --thinker.gpu '[0, 1]' \
  --thinker.engine.disable_cuda_graph false \
  --thinker.engine.enable_torch_compile false \
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

Group-limited expert routing defaults to the reference Torch implementation
for both TP1 and TP2, in eager and graph execution. To opt into SGLang's fused
`TopK`, add this startup option:

```bash
--thinker.engine.json_model_override_args '{"llada2_uni_topk_backend":"sglang"}'
```

Use `"torch"` instead of `"sglang"` to explicitly select the default. If other
model overrides are needed, include them in the same JSON object. This setting
is applied when the model loads on every thinker rank; restart the server to
change it. No checkpoint files need editing.

Both paths use FP32 router logits, sigmoid scoring, correction bias and
normalized weights, but the fused implementation can change expert ordering
and sigmoid rounding. In a controlled 1,050-question MMMU run, Torch/SGLang
scored 517/500 at TP1 and 517/494 at TP2. SGLang reduced TP2 T2I/edit E2E from
14.918/12.052 seconds to 14.070/11.715 seconds on H20-3e. These single-run
results describe an observed tradeoff, not a guaranteed score or speedup.

LLaDA-Uni thinker workers retain the same `CUDA_VISIBLE_DEVICES` list across
TP ranks and select distinct local devices. The stage defaults
`SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS=false` so custom all-reduce v2 can
identify each GPU when initializing symmetric memory. Other models retain
their existing placement defaults. Check startup logs to confirm custom
all-reduce initialization when comparing TP performance. The existing
`SGLANG_OPT_USE_CUSTOM_ALL_REDUCE_V2=0` environment setting selects legacy
custom all-reduce for a separate comparison; this pipeline does not change
the global communication default.

### Optional H20 TP2 MoE Tuning

For LLaDA2.0-Uni BF16 on NVIDIA H20-3e with TP2, this checkout includes gate/up and down-projection configurations originally tuned with SGLang 0.5.20 and Triton 3.7.1. They were also validated with SGLang 0.5.22, Torch 2.14.1, and Triton 3.8.0 through SGLang's configuration-version fallback. This is reuse of existing tuning, not a new Triton 3.8.0 tuning sweep.

Set `SGLANG_MOE_CONFIG_DIR` explicitly when starting the server from the repository root. Installing or upgrading SGLang does not install these repository configurations into its runtime directory. A launcher that clears inherited environment variables must pass this setting to the server process again.

```bash
SGLANG_MOE_CONFIG_DIR="$PWD/examples/tuning/llada2_uni/h20_tp2" \
CUDA_VISIBLE_DEVICES=0,1 sgl-omni serve \
  --model-path inclusionAI/LLaDA2.0-Uni --port 8000 \
  --thinker.process thinker --thinker.tp_size 2 --thinker.gpu '[0, 1]' \
  --thinker.engine.disable_cuda_graph false \
  --thinker.engine.enable_torch_compile false \
  --thinker.engine.cuda_graph_bs '[1, 2, 3, 4]'
```

The TP workers inherit the environment variable. Point it at the absolute directory containing `configs/`, not at `configs/` itself. Keep both JSON files: `_down.json` enables the existing down-projection TMA path. No SGLang installation files or kernel implementations need changing.

Check startup or first-forward logs for both `E=256,N=256` configuration paths inside this checkout. Triton 3.7.1 reports `Using MoE kernel config from`; the validated SGLang 0.5.22 / Triton 3.8.0 stack reports `Fallback to triton version 3.7.1 and use MoE kernel config from`. That version-fallback warning is expected for these files. `Using default MoE kernel config` instead means the requested tuned configuration was not loaded.

The configurations were tuned with SGLang's fused-MoE benchmark for
32/64/96/128 input-token rows, covering common DLLM block and CFG shapes.
`N=256` is the per-rank expert intermediate width, not the GEMM tile width.
SGLang chooses the nearest token-count entry for other shapes; larger
prefill shapes were not separately tuned.

This environment variable replaces the entire MoE configuration search root, rather than extending the installed configuration library. Omit it for TP1, other models, quantized weights, or unvalidated hardware/software combinations. Set it before launching workers and restart the server when changing files because SGLang caches loaded configurations. The regular TP command above remains available without this optional tuning.

On H20-3e, a matched-request comparison measured the following Thinker stage times (seconds, median of three requests after two warmups):

| Configuration | T2I | Edit |
| --- | ---: | ---: |
| SGLang 0.5.20, TP1 | 6.557 | 3.130 |
| SGLang 0.5.20, TP2 with tuning | 5.186 | 2.803 |
| SGLang 0.5.22, TP1 | 6.543 | 3.097 |
| SGLang 0.5.22, TP2 with default configuration | 5.794 | 2.897 |
| SGLang 0.5.22, TP2 with repository tuning | 5.195 | 2.806 |

Both versions used BF16, SGLang TopK, CUDA Graphs, compilation disabled, 32 dLLM steps, text CFG 4, edit image CFG 1.5, CFG rescale 0.7, seed 42, and eight decoder-turbo steps with the SGLang `torch_sdpa` decoder. T2I used a 1024x1024 shipwreck prompt; edit used a raw turtle image with a shell-smoothing instruction. The TopK override above must be added to the launch command to reproduce these measurements; the default remains Torch routing. Runs were serial with exclusive GPUs. Times cover Thinker dispatch through completion, not HTTP E2E or isolated kernel execution.

Explicit tuning reduced the new stack's TP2 Thinker latency by 10.3% for T2I and 3.1% for edit, bringing it within 0.2% of the old tuned stack on these requests. Its TP1-to-TP2 speedups were 1.26x and 1.10x. T2I output was pixel-identical before and after tuning, while edit output changed (RGB MAE 9.47 on a 0-255 scale). These sample checks do not establish full-benchmark accuracy equivalence; no new MMMU or ImgEdit score is implied.

## Image Generation and Editing

Use `POST /v1/images/generations` for T2I and `POST /v1/images/edits` for
editing. These routes use SGLang Diffusion's image request and response
schemas while executing Omni's Thinker and selected image decoder backend.
`dllm_steps` controls VQ token generation; `num_inference_steps` controls
decoder diffusion sampling. `guidance_scale` sets Thinker CFG, not an
additional CFG pass in the image decoder.

```python
import base64
from pathlib import Path

import requests

request = {
    "model": "inclusionAI/LLaDA2.0-Uni",
    "prompt": "A sailboat on a calm lake.",
    "size": "1024x1024",
    "response_format": "b64_json",
    "decode_mode": "decoder-turbo",
    "num_inference_steps": 8,
    "dllm_steps": 32,
    "guidance_scale": 4.0,
    "seed": 42,
}
response = requests.post(
    "http://localhost:8000/v1/images/generations", json=request, timeout=600
)
response.raise_for_status()
image = response.json()["data"][0]
Path("generated.png").write_bytes(base64.b64decode(image["b64_json"]))
```

Edits accept multipart form data with exactly one `image`/`image[]` upload or
`url`/`url[]` reference. Dimensions follow the processed source grid; omit
`size`, `width`, and `height`. Set `cfg_text_scale` and `cfg_image_scale` for
editing guidance. Omitting them retains the model's task-specific defaults.

```bash
curl http://localhost:8000/v1/images/edits \
  -F 'image=@source.png' \
  -F 'prompt=Change the background to a beach.' \
  -F 'response_format=b64_json' \
  -F 'decode_mode=decoder-turbo' \
  -F 'num_inference_steps=8' \
  -F 'dllm_steps=8' \
  -F 'cfg_text_scale=4.0' -F 'cfg_image_scale=1.5' -F 'seed=42'
```

Both routes are non-streaming, support one PNG (`n=1`), and return
`{id, created, data: [...]}`. `response_format=b64_json` returns raw base64;
`response_format=url` returns an inline PNG data URL without server-side file
retention. T2I accepts either `size` or paired `width`/`height`, defaulting to
1024x1024. Dimensions must be positive multiples of 32; if both forms are
supplied, they must agree. Unsupported native sampling controls are rejected
rather than ignored.
Image endpoints imply image output; no `modalities` field is required.
`/v1/chat/completions` retains text and image-understanding requests; use the
image endpoints for T2I and editing.

The response's `usage` counts prompt and generated tokens, including both
passes in thinking mode.

Text-to-image supports `mode: "normal"` and `mode: "thinking"`. Editing supports
only `mode: "normal"`. Generation mode is separate from `decode_mode`, which
selects `normal` or `decoder-turbo` diffusion decoding.
For normal decoding, `num_inference_steps` specifies at least two sampling
points, giving `num_inference_steps - 1` solver intervals. For decoder-turbo, it specifies at
least one decoder iteration. The examples use eight turbo iterations.
When `decode_mode` is omitted, validation uses the decoder factory's configured
mode, so a turbo factory accepts `num_inference_steps=1` without a mode override.

Common controls include `seed`, `cfg_rescale`, and `dllm_steps`. T2I uses
`guidance_scale` (the thinker's `cfg_scale`); editing uses `cfg_text_scale` and
`cfg_image_scale`. `resolution_multiplier` is a decoder factory setting, not a
request field. Its default value of 2 maps the semantic grid to output pixels.

The server selects a patch-aligned source grid near a 512x512 pixel budget,
then resizes proportionally and center-crops the image to that grid. Small
images are enlarged without black padding; images already matching the grid
are preserved. Aspect ratios beyond 4:1 or 1:4 require additional cropping.
Image understanding retains its separate preprocessing and pixel budgets.

## Thinking Image Generation

Use the same `POST /v1/images/generations` endpoint with top-level
`mode: "thinking"`. For example, add it to the T2I request above:

```python
request["mode"] = "thinking"
response = requests.post(
    "http://localhost:8000/v1/images/generations", json=request, timeout=600
)
response.raise_for_status()
Path("thinking.png").write_bytes(
    base64.b64decode(response.json()["data"][0]["b64_json"])
)
```

The thinker first generates text with a 2048-token budget, stopping at
`<boi>`. It then retains that context and generates the requested VQ grid
with CFG before image decoding. Both phases must fit the configured context
length. The native response remains `data[]` containing one PNG; it does not
expose the thinking text. No `modalities` field is needed. Thinking-mode
editing is unsupported, and chat completions remain for understanding.

## Interleaved Output

The interleaved variant answers with text and images in one response. Each
image decodes while the model writes the next text.

```bash
sgl-omni serve --model-path inclusionAI/LLaDA2.0-Uni --variant interleaved --port 8000 \
  --thinker.engine.enable_torch_compile false
```

Send text-only `messages` to `POST /v1/chat/completions`:

```python
import requests

response = requests.post(
    "http://localhost:8000/v1/chat/completions",
    json={
        "model": "inclusionAI/LLaDA2.0-Uni",
        "messages": [{"role": "user", "content": "Illustrate a three-frame story"}],
        "stage_params": {"orchestrator": {"max_frames": 3, "seed": 42}},
    },
    timeout=1800,
)
segments = response.json()["choices"][0]["message"]["segments"]
```

`choices[0].message.segments` lists the text and image segments in the order
the model wrote them, and `message.content` holds the text. With
`stream: true`, each segment arrives in `choices[0].delta.segment` as soon as it
is ready. An image segment carries the PNG with its size and SHA-256, and `url` is a
PNG data URL:

```json
{
  "type": "segment",
  "execution_id": "...",
  "segment_index": 1,
  "kind": "image",
  "data": {
    "kind": "image",
    "mime_type": "image/png",
    "size_bytes": 12345,
    "sha256": "...",
    "url": "..."
  }
}
```

Optional controls go in `stage_params.orchestrator`: `max_frames` (default
10, the most the default server admits), `text_max_new_tokens`,
`image_max_new_tokens`, `max_image_tokens`, `dllm_steps`, `cfg_scale`,
`cfg_text_scale`, `cfg_image_scale`, `cfg_rescale`, `decode_mode`,
`decoder_steps`, and `seed`.

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

The table below lists common text and image-understanding parameters for
`/v1/chat/completions`.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `model` | string | `null` | Model identifier |
| `messages` | list | (required) | List of chat messages, each with `role` and `content` |
| `modalities` | list | `["text"]` | Use `["text"]` for text and image understanding |
| `images` | list | `null` | List of image file paths (local paths or URLs) |
| `max_tokens` | int | `null` | Maximum number of tokens to generate |

## Known Limitations

- Image generation and editing return one image per non-streaming request.
- Thinking mode supports text-to-image only.
- Interleaved output takes text-only input.
