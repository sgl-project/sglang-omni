# MiniMax-H3-Image on Moore Threads MUSA

This recipe runs MiniMax-H3-Image text-to-image and image editing on MTT S5000.
It covers one-card offline generation plus two-card Ulysses and
tensor-parallel serving.

## Prerequisites

- A MUSA 5.2 environment with `torch_musa` and SGLang-Omni installed.
- A local MiniMax-H3-Image checkpoint.
- The SGLang MUSA compatibility changes in
  [sgl-project/sglang#50713](https://github.com/sgl-project/sglang/pull/50713).
- MiniMax-H3 VAE graph decoding requires PyTorch 2.11 or newer on MUSA.
  PyTorch 2.9 remains functional through the eager VAE fast path and logs that
  graph decoding is disabled.

## Generate an image on one card

```bash
export MODEL_PATH=/path/to/MiniMax-H3-Image
export PYTHONUNBUFFERED=1
export TORCHDYNAMO_DISABLE=1
export HF_HUB_OFFLINE=1
export SGLANG_DIFFUSION_PLATFORM_OVERRIDE=musa

cat > /tmp/h3-request.json <<'JSON'
{
  "prompt": "A sunlit still life of a cobalt-blue ceramic vase holding magenta peonies on a warm oak table",
  "width": 1024,
  "height": 1024,
  "seed": 42,
  "generator_device": "cpu",
  "min_generation_area": 1048576
}
JSON

sglang generate \
  --model-path "${MODEL_PATH}" \
  --config /tmp/h3-request.json \
  --output-file-path output.png \
  --attention-backend fa \
  --component-attention-backends text_encoder=torch_sdpa,transformer=fa \
  --warmup-mode off \
  --dit-cpu-offload \
  --vae-cpu-offload \
  --text-encoder-cpu-offload \
  --dit-offload-prefetch-size 16
```

The CPU offload options are required on one 96 GB device because the model
components do not fit resident at the same time. `TORCHDYNAMO_DISABLE=1` keeps
this example on the functional eager path.

`generator_device: "cpu"` fixes the initial-noise generator so the same seed
produces the same initial latent across machines. It does not change the
inference device.

## Serve with two-card Ulysses

SGLang uses Ulysses sequence parallelism for multi-card H3-Image serving.
`OMP_NUM_THREADS=1` is required for the replicated MUSA text-encoder workers.

```bash
export MODEL_PATH=/path/to/MiniMax-H3-Image
export PYTHONUNBUFFERED=1
export TORCHDYNAMO_DISABLE=1
export HF_HUB_OFFLINE=1
export SGLANG_DIFFUSION_PLATFORM_OVERRIDE=musa
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

sglang serve \
  --model-path "${MODEL_PATH}" \
  --served-model-name MiniMax-H3-Image \
  --host 127.0.0.1 \
  --port 30000 \
  --backend sglang \
  --num-gpus 2 \
  --ulysses-degree 2 \
  --ring-degree 1 \
  --encoder-parallel replicate \
  --dit-cpu-offload \
  --vae-cpu-offload \
  --text-encoder-cpu-offload \
  --dit-offload-prefetch-size 16 \
  --attention-backend fa \
  --component-attention-backends text_encoder=torch_sdpa,transformer=fa \
  --warmup-mode off
```

Generate an image through the OpenAI-compatible image API:

```bash
curl http://127.0.0.1:30000/v1/images/generations \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "MiniMax-H3-Image",
    "prompt": "A sunlit still life of a cobalt-blue ceramic vase holding magenta peonies on a warm oak table",
    "size": "1024x1024",
    "seed": 42,
    "response_format": "b64_json"
  }' \
  -o /tmp/h3-t2i.json
```

Edit an existing image:

```bash
curl http://127.0.0.1:30000/v1/images/edits \
  -F model=MiniMax-H3-Image \
  --form-string 'prompt=<Picture 1> is the still life. Add a small brass key beside the vase and keep the composition unchanged.' \
  -F size=1024x1024 \
  -F seed=7 \
  -F response_format=b64_json \
  -F image=@input.png \
  -o /tmp/h3-edit.json
```

## Serve with tensor parallelism

Pure tensor parallelism shards the DiT and uses native MCCL all-reduce on
MUSA. Component offload keeps the text encoder, DiT, and VAE from remaining
resident together.

```bash
export MODEL_PATH=/path/to/MiniMax-H3-Image
export GPUS=2  # Validated values: 2 and 4

sglang serve \
  --model-path "${MODEL_PATH}" \
  --served-model-name MiniMax-H3-Image \
  --host 127.0.0.1 \
  --port 30000 \
  --backend auto \
  --num-gpus "${GPUS}" \
  --tp-size "${GPUS}" \
  --dit-cpu-offload \
  --vae-cpu-offload \
  --text-encoder-cpu-offload \
  --attention-backend fa \
  --component-attention-backends transformer=fa \
  --warmup-mode off
```

Use the same `/v1/images/generations` and `/v1/images/edits` requests shown
above.

## Validate

1. `sglang generate` exits with status 0.
2. `/v1/images/generations` and `/v1/images/edits` return status 200.
3. Returned images decode as the requested RGB resolution and are visually
   coherent.
4. Logs contain `Pipeline instantiated` and
   `MiniMax-H3 VAE: installed quality-gated fast path`.
5. On PyTorch 2.11, VAE decoding logs either
   `captured decode graph for tile batch shape` or uses a previously captured
   graph.
6. Logs do not contain `Pipeline class 'MiniMaxH3ImagePipeline' not found` or
   `Segfault`.

## Known limitations

- Use one-card execution for reproducibility checks. Same-seed reproducibility
  has not been established for multi-card execution.
- Two-card Ulysses requires component offload. Keeping all H3 components
  resident exceeds one S5000 device.
- SGLang disables its CUDA-only `srt_custom_allreduce` path outside CUDA.
  MUSA tensor groups use native MCCL collectives.
- Do not enable `PYTORCH_MUSA_ALLOC_CONF=expandable_segments:True` during component loading.
- The H3-specific fused Triton norm+RoPE block currently uses the unfused fallback on MUSA.
- MATE provides RMSNorm primitives, but currently has no drop-in replacement for
  the H3 fused QK-Norm+RoPE interface.
- Component-offload transitions can invalidate and recapture the VAE decode
  graph. This recipe does not claim persistent graph reuse across component
  moves.

This recipe is a functional validation, not a throughput benchmark.
