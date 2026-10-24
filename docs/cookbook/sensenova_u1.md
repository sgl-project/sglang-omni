# SenseNova-U1.5 Image Generation

SGLang-Omni supports `SenseNova-U1.5-8B-MoT` through the shared UMM image
generation pipeline, including text-to-image (T2I) and image-to-image (I2I)
through the OpenAI-compatible Images API. It loads the checkpoint once in the
`generate` stage, routes requests through the UMM orchestrator/output stages,
and returns RGB PNG images as base64 data.

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

## Data-Parallel Serving

To serve concurrent requests on two GPUs, run one complete model replica on
each GPU:

```bash
python -m sglang_omni.cli serve \
  --config examples/configs/sensenova_u1_dp2.yaml \
  --model-path "$MODEL_PATH" \
  --port 8000
```

The `sensenova_generate` process uses `num_replicas: 2` and
`replica_devices: [0, 1]`. Requests are assigned to replicas at admission.
Each GPU must have enough memory for the full model and its activations. DP
increases concurrent throughput; it does not split one request across GPUs or
reduce single-request latency. Use a distinct GPU for each SenseNova replica.
The existing T2I dynamic batching setting applies independently within each
replica; I2I requests still run one at a time.

## Tensor-Parallel Serving

Split the dense Qwen3 backbone across two GPUs:

```bash
python -m sglang_omni.cli serve \
  --config examples/configs/sensenova_u1_tp2.yaml \
  --model-path "$MODEL_PATH" \
  --port 8000
```

Alternatively, set `--generate.tp_size 2 --generate.gpu '[0, 1]'`.
Supported TP sizes are 1, 2, 4, and 8. The attention head count, KV head count,
and MLP intermediate width must all be divisible by the TP size. The current
implementation supports the dense SenseNova-U1.5 Qwen3 backbone on CUDA
(NCCL) and Ascend NPU (HCCL); MoE backbones are rejected.

Both understanding and image-generation attention and MLP weights are sharded.
Output projections use All-Reduce to keep hidden states replicated. Vision
encoders, embeddings, normalization, and the pixel head remain replicated.
Weights are loaded on CPU and sharded before accelerator placement, so TP
reduces backbone accelerator memory but each worker still needs host memory
to load the original checkpoint. Initial image noise is broadcast from rank
zero for both T2I and I2I.

TP requires `generate.factory.max_batch_size: 1`: independent dynamic batching
on different ranks could produce different collective sequences. A single T2I
request can still use `n > 1`. Cache-DiT and CUDA layerwise offload can be used
with TP. Only the leader publishes the generated result.

For the UMM pipeline, TP cancellation is cooperative at model checkpoints:
ranks agree before leaving a sequence of collectives, and a cancelled batch
member does not stop live members. Shutdown drains dispatched requests before
unloading Cache-DiT and destroying the TP process group. The worker supervisor
can still force termination if its shutdown timeout expires before draining
finishes.

For two replicas with two TP ranks each, use
`examples/configs/sensenova_u1_dp2_tp2.yaml`. The flat
`replica_devices: [0, 1, 2, 3]` assigns GPUs `[0, 1]` to the first replica and
`[2, 3]` to the second. Every replica and TP rank must use a distinct GPU.

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

`size` dimensions must be positive multiples of 32. Text-to-image requests
support `n` from 1 to 10 and return base64 PNG images. `/v1/images/edits` also
accepts `n` from 1 to 10 and sends each output as its own pipeline request with
`seed + output_index`, so every output gets the full `timeout_s`. A `/generate`
image edit produces one image, so it must use `n=1`.
Both routes require `think_mode=false`.

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

The cache wraps the Qwen decoder only during pure image denoising. Prefix
prefill and KV-cache updates continue through the original decoder. Three
branch image guidance and timestep-gated CFG run without Cache-DiT. In a
dynamic T2I batch, requests with different cache settings are dispatched in
separate batches.

Cache parameters are validated after resolving the server default and request
override. Disabled requests and unsupported guidance schedules ignore cache
parameters. When caching is enabled, unsupported names and invalid values are
rejected.

## Edit an Image

The same server exposes the SenseNova image-to-image (I2I) path through
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
ratio using the default pixel budget. I2I requests stay single-item even when
T2I dynamic batching is enabled.

The edits endpoint accepts `input_max_pixels` to control reference-image
resizing and `do_resize` as a `true` or `false` form field. For example,
`-F 'input_max_pixels=262144' -F 'do_resize=true'` lowers the reference pixel
budget to 512x512 pixels while retaining the aspect ratio. The minimum accepted
budget is 262144. These options do not replace `size`, which controls the
generated image dimensions when provided.

`do_resize=false` sends every reference to the model at its original
dimensions, so each reference's width and height must be multiples of 32
pixels. An unaligned reference, such as a 4032x3024 photo, is rejected with
400 rather than resized. Without resizing, reference pixels are bounded per
request, not per image: all references in one request together may contain at
most 33554432 pixels (`max_reference_pixels`, 32 x 1024 x 1024), and the edits
endpoint also caps each upload at 4096x4096 pixels. `do_resize=false` does not
disable the other admission limits or output-size selection.

I2I accelerator memory errors identify the CUDA or NPU backend and suggest
reducing references or output size. Without TP they return 400 on every
endpoint. With TP they still stop the TP group: the generate stage exits, and
the server stops serving until it is restarted. Lowering `input_max_pixels` only
helps when `do_resize=true`; CUDA requests can also lower
`generate.factory.dit_layerwise_resident_layers` with layerwise offload enabled.
A CPU memory error remains a server failure rather than being reported as a
reference-image accelerator error. Empty or invalid boolean form values are
rejected rather than treated as omitted fields.

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

- Each DP replica uses one GPU with TP=1, or a distinct GPU group with TP>1.
  Dense Qwen3 tensor parallelism is supported; sequence parallelism is not.
- BF16 is the supported initial dtype.
- Dynamic batching defaults to off (`max_batch_size=1`).
- Quantization, compact mode, and SRT KV transfer are not exposed by the Omni
  stage. Thinking mode and interleaved text/image output are a later #2831
  phase and remain unexposed. CUDA layerwise CPU offload is available through
  `generate.factory.dit_layerwise_offload`.
- The UMM path uses fixed admission limits (`DEFAULT_UMM_ADMISSION_LIMITS`);
  `max_outputs` defaults to 10 to cover up to 10 images.
- Ascend I2I basic functionality and memory behavior have been validated on
  real hardware. Kernel-hit profiling and broader image-quality and performance
  characterization remain pending.

## Shared UMM Serving Contract

The default SenseNova pipeline is the shared UMM pipeline:
`orchestrator -> generate -> output`. It keeps the SenseNova model stage and
its batching, TP/DP, Cache-DiT, and offload settings. Thinking and interleaved
model generation remain out of scope for this pipeline; they are tracked as a
later #2831 phase and depend on the UMM progress-streaming contract.

The controller dispatches one `generate` phase and accepts image segments in
generation order. The phase returns
the original request, including its continuation metadata. It owns the model
and device caches; the adapter owns only request translation and retained
request bookkeeping. The output stage shares the controller process, as
UMMPipelineConfig requires.

Each image segment uses the shared inline-media fields: kind, MIME type, byte
size, SHA-256 digest, and data URL. The Images endpoints decode these segments
from the final result; `/generate` clients receive them in the segment list.

All five controller bounds are required. `max_outputs` must cover
`max_images` (the default is 10).
Output count and input size are checked before generation; image dimensions
and step count also have explicit model admission limits. The admission limits
are fixed model constants (`DEFAULT_UMM_ADMISSION_LIMITS`), because the UMM
controller factory does not accept per-model adapter options. Reference-image
bounds cover both original and resized pixels, using the same size calculation
as preprocessing. They respect default resize settings and `do_resize=False`.
I2I output pixel limits and retained-byte reservations use the resolved output
dimensions, including aspect-ratio rounding, before dispatching model work.
These bounds do not represent total accelerator memory.

For `/generate`, overrides in `stage_params.generate` are applied consistently
to admission, request cost, batching, TP validation, and model execution.
Overrides cannot bypass the model's step, image-count, or pixel limits.
A null `stage_params.generate` entry behaves like an omitted entry; strings
and lists are rejected.

Cancellation is cooperative at model checkpoints. TP ranks decide together
before leaving a sequence of collectives. One cancelled member of a fused batch
must not fail live members; their shared computation can continue until the
batch has no live requests. Cancellation and failure release per-request state,
while model weights remain available for later requests. Invalid input must be
rejected before TP fan-out.

Acceptance compares this UMM path with commit `706053fa0` using the same
checkpoint, seeds, hardware, parameters, concurrency, and warmup. Required
coverage includes T2I/I2I, ordered multi-output T2I, compatible request
batching, TP/DP, Cache-DiT, supported offload, cancellation, deadlines, and
failure recovery. CPU contract tests, GPU/NPU inference, performance
measurements, and business acceptance are reported separately. Unexecuted
combinations remain unvalidated.

### Run the UMM Pipeline

With the model dependencies installed and `MODEL_PATH` pointing to the
checkpoint, launch the default configuration:

```bash
python -m sglang_omni.cli serve \
  --model-path "$MODEL_PATH" \
  --generate.gpu 0 \
  --port 8000
```

The Images generation and edit requests above use the same response contract.
Generation settings remain under `generate.factory`; the fixed admission limits
are checked before generation. The controller still requires
`max_dispatches`, `max_outputs`, `max_active`, `max_retained_bytes`, and
`timeout_s` under `orchestrator.factory`. The retained-byte limit reserves a
conservative inline PNG budget before generation, so a request can be rejected
before its actual encoded image size is known.
