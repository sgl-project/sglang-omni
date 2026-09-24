# SenseNova-U1.5 Image Generation

SGLang-Omni supports the native image-generation path of
`SenseNova-U1.5-8B-MoT`. The first integration scope is text-to-image (T2I)
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

## Dynamic Batching

Batching is opt-in until hardware validation establishes safe memory limits.
Only T2I requests with equal width, height, inference steps, and guidance scale
share a model batch. Prompts and seeds remain per request. Incompatible requests
are dispatched separately, and image-edit requests remain single-item.

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
- Ascend 910C correctness, memory, kernel-hit, and performance numbers must be
  recorded from real hardware before the NPU path is described as validated.
