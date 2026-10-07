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

For LLaDA2.0-Uni BF16 on NVIDIA H20-3e with TP2, SGLang 0.5.20 and
Triton 3.7.1, this checkout includes tuned gate/up and down-projection
configurations in `examples/tuning/llada2_uni/h20_tp2`. Enable them explicitly
when starting the server from the repository root:

```bash
SGLANG_MOE_CONFIG_DIR="$PWD/examples/tuning/llada2_uni/h20_tp2" \
CUDA_VISIBLE_DEVICES=0,1 sgl-omni serve \
  --model-path inclusionAI/LLaDA2.0-Uni --port 8000 \
  --thinker.process thinker --thinker.tp_size 2 --thinker.gpu '[0, 1]' \
  --thinker.engine.disable_cuda_graph false \
  --thinker.engine.enable_torch_compile false \
  --thinker.engine.cuda_graph_bs '[1, 2, 3, 4]'
```

The TP workers inherit the environment variable. SGLang's existing MoE
loader reads `configs/triton_3_7_1/` beneath that directory; do not point the
variable at the `configs` subdirectory. Both JSON files are needed: the
`_down.json` configuration enables the existing down-projection TMA path.
No SGLang installation files or kernel implementations need changing.
Startup or first-forward logs should report `Using MoE kernel config from`
paths inside this checkout for both files.

The configurations were tuned with SGLang's fused-MoE benchmark for
32/64/96/128 input-token rows, covering common DLLM block and CFG shapes.
`N=256` is the per-rank expert intermediate width, not the GEMM tile width.
SGLang chooses the nearest token-count entry for other shapes; larger
prefill shapes were not separately tuned.

This environment variable replaces the entire MoE configuration search
root, rather than extending the installed configuration library. Omit it
for TP1, other models, quantized weights, or other hardware/software
combinations unless they have their own validated configurations. Set it
before launching workers and restart the server when changing files because
SGLang caches loaded configurations. The regular TP command above remains
available without this optional tuning.

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
    "dllm_steps": 8,
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
  -F 'cfg_text_scale=4.0' -F 'cfg_image_scale=1.5' -F 'seed=42'
```

Both routes are non-streaming, support one PNG (`n=1`), and return
`{id, created, data: [...]}`. `response_format=b64_json` returns raw base64;
`response_format=url` returns an inline PNG data URL without server-side file
retention. T2I accepts either `size` or paired `width`/`height`, defaulting to
1024x1024. Unsupported native sampling controls are rejected rather than ignored.
The old chat image-generation entrypoint remains available for existing clients.

For thinking T2I, set `mode: "thinking"`. To retrieve both thinking text and
the image, use `/v1/chat/completions` with `modalities: ["text", "image"]`
and `image_generation.mode: "thinking"`.
The text pass has a 2048-token budget and stops at `<boi>`. The image pass
retains the generated context and applies CFG to the VQ tokens. Both passes
must fit the thinker's configured context length. Thinking mode does not
support editing.

The server selects a patch-aligned source grid near a 512x512 pixel budget,
then resizes proportionally and center-crops the image to that grid. Small
images are enlarged without black padding; images already matching the grid
are preserved. Aspect ratios beyond 4:1 or 1:4 require additional cropping.
Image understanding retains its separate preprocessing and pixel budgets.

## Interleaved Output

Start the interleaved variant with `examples/configs/llada2_uni_interleaved.yaml`.
Send `/v1/chat/completions` with text-only `messages`,
`modalities: ["text", "image"]`, `stream: false`, and
`image_generation: {"mode": "interleaved", "max_frames": 3}`.
LLaDA-specific generation controls remain in `image_generation`; it does not
use Cosmos3's Reasoner decision loop or media re-ingestion policy.

The response exposes ordered `choices[0].message.segments` following the
[Cosmos3 segment contract](https://github.com/sgl-project/sglang-omni/issues/2183).
That shared contract is proposed in PRs #2204/#2205 and is not yet merged into
main. `message.content` is the concatenated text view. The previous
`message.images` table and `content[].image_ref` format are no longer emitted
for interleaved requests.

```json
{
  "type": "segment",
  "session_id": "request-execution-id",
  "segment_index": 1,
  "kind": "image",
  "data": {
    "kind": "image",
    "mime_type": "image/png",
    "url": "data:image/png;base64,...",
    "sha256": "sha256-of-png-bytes",
    "size_bytes": 12345
  }
}
```

Text segments use `kind: "text"` and a string `data`. Indices start at zero
and are contiguous across text and images. The session ID identifies this
request execution. SDK `CompletionResult.segments` and buffered `/generate`
results expose the same list. This adapter returns a completed snapshot;
incremental `delta.segment` delivery is not enabled by this API migration.

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

## Known Limitations

- Image generation and editing return one image per non-streaming request.
- Interleaved generation returns a buffered segment snapshot; streaming is not supported.
