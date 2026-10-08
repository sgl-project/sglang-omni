# SenseNova-U1.5 Image Generation

SGLang-Omni supports the native image-generation path of
`SenseNova-U1.5-8B-MoT`, including text-to-image (T2I) and image-to-image (I2I)
through the OpenAI-compatible Images API. It loads the checkpoint once in a
single stage and returns RGB PNG images as base64 data.

The Ascend path includes NPU fused infer attention (FIA), NPU RMSNorm, fused
SwiGLU, and denoise timestep embedding reuse when the installed runtime exposes
the corresponding operators. Operator availability is logged at model startup;
an Ascend trace is still required to prove that an operator ran for a specific
request.

## Installation

Install the NPU build from the repository root. The command checks the existing
CANN, PyTorch NPU, and SGLang environment without downloading the model:

```bash
bash scripts/npu/install_npu.sh --check
bash scripts/npu/install_npu.sh
```

For an offline deployment, place the complete checkpoint on local or shared
storage. The examples below use:

```bash
export MODEL_PATH=/nas/disk1/SenseNova/SenseNova-U1.5-8B-MoT
```

## Launch the Server

The default is BF16, NPU 0, and single-request execution:

```bash
python -m sglang_omni.cli serve \
  --model-path "$MODEL_PATH" \
  --generate.gpu 0 \
  --port 8000
```

## Generate an Image

```bash
curl http://127.0.0.1:8000/v1/images/generations \
  -H 'Content-Type: application/json' \
  -d '{
    "prompt": "A blue cat sitting by a window",
    "size": "1024x1024",
    "n": 1,
    "response_format": "b64_json",
    "seed": 42,
    "num_inference_steps": 10,
    "guidance_scale": 4.0
  }' > response.json
```

The response follows the Images API shape:

```json
{
  "created": 0,
  "data": [{"b64_json": "..."}]
}
```

`size` dimensions must be positive multiples of 32. The current route supports
one image per request, base64 PNG output, and `think_mode=false`.

## Cache-DiT Acceleration

Cache-DiT is off by default. Enable it for all requests in the pipeline config
or at launch, and optionally tune the five DBCache parameters:

```bash
python -m sglang_omni.cli serve \
  --model-path "$MODEL_PATH" \
  --generate.factory.enable_cache_dit true \
  --generate.factory.cache_dit_params '{"residual_diff_threshold": 0.1}'
```

Individual image requests can override the server setting with
`enable_cache_dit` and `cache_dit_params`. The edits endpoint accepts the same
fields as multipart form data, with `cache_dit_params` encoded as a JSON
object. Only `Fn_compute_blocks`, `Bn_compute_blocks`, `max_warmup_steps`,
`residual_diff_threshold`, and `max_continuous_cached_steps` are supported.

The cache wraps the Qwen decoder only during pure image denoising. Prefix,
Think, text, and KV-cache updates continue through the original decoder. Three
branch image guidance and timestep-gated CFG run without Cache-DiT. In a
dynamic T2I batch, requests with different cache settings are dispatched in
separate batches.

Cache parameters are validated after resolving the server default and request
override. Disabled requests and unsupported guidance schedules ignore cache
parameters. When caching is enabled, unsupported names and invalid values are
rejected.

## Edit an Image

The same server exposes the native SenseNova image-to-image (I2I) path through
the multipart Images API:

```bash
curl http://127.0.0.1:8000/v1/images/edits \
  -F "image=@reference.png" \
  -F 'prompt=Keep the subject unchanged and replace the background with snow.' \
  -F 'size=512x512' \
  -F 'n=1' \
  -F 'response_format=b64_json' \
  -F 'seed=42' \
  -F 'num_inference_steps=30' \
  -F 'guidance_scale=4.0' \
  -F 'img_cfg_scale=1.0' > response.json
```

When `size` is omitted, the output keeps the first reference image's aspect
ratio within the default pixel budget. I2I requests stay single-item even when
T2I dynamic batching is enabled.

On Ascend, the condition, image-condition, and unconditional I2I branches each
prepare their own prefix length. This enables the BNSD FIA cache when the
installed runtime exposes FIA and preserves the SDPA fallback otherwise. The
service logs the selected cache layout for every I2I request; an Ascend trace is
still required to prove kernel execution.

## Dynamic Batching

Batching is opt-in until hardware validation establishes safe memory limits.
Only T2I requests with equal width, height, inference steps, guidance scale,
and Cache-DiT settings share a model batch. Prompts and seeds remain per
request. Incompatible requests are dispatched separately, and image-edit
requests remain single-item.

For two concurrent `1024x1024`, 10-step requests with CFG enabled, the following
cost limit admits exactly two requests (`width * height * steps * 2` per
request):

```bash
python -m sglang_omni.cli serve \
  --model-path "$MODEL_PATH" \
  --generate.factory.max_batch_size 2 \
  --generate.factory.max_batch_wait_ms 10 \
  --generate.factory.max_batch_cost 41943040
```

When batching is used, the service log includes
`SenseNova-U1 T2I batch: size=2`. Concurrent HTTP success without this message
does not prove that model execution was batched.

## Current Scope

- Single-card execution; tensor parallelism is not wired into this native HF
  stage.
- BF16 is the supported initial dtype.
- Dynamic batching defaults to off (`max_batch_size=1`).
- Quantization, CPU offload, compact mode, and thinking/SRT KV transfer are not
  exposed by the Omni stage.
- Ascend I2I basic functionality and memory behavior have been validated on
  real hardware. Kernel-hit profiling and broader image-quality and performance
  characterization remain pending.
