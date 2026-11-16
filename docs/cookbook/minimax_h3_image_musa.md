# MiniMax-H3-Image on Moore Threads MUSA

This recipe runs MiniMax-H3-Image text-to-image inference on one MTT S5000.

## Prerequisites

- A MUSA 5.2 environment with `torch_musa` and SGLang-Omni installed.
- A local MiniMax-H3-Image checkpoint.
- The SGLang MUSA compatibility changes in
  [sgl-project/sglang#50713](https://github.com/sgl-project/sglang/pull/50713).

## Generate an image

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

The CPU offload options are required on a 96 GB device because the model components do not fit resident at the same time. `TORCHDYNAMO_DISABLE=1` keeps this example on the functional eager path.

`generator_device: "cpu"` fixes the initial-noise generator so the same seed produces the same initial latent across machines. It does not change the inference device.

## Validate

1. `sglang generate` exits with status 0.
2. `output.png` is a valid RGB PNG.
3. Logs contain `Pipeline instantiated` and `MiniMax-H3 VAE: installed quality-gated fast path`.
4. Logs do not contain `Failed to build JIT module`, `Pipeline class 'MiniMaxH3ImagePipeline' not found`, or `Segfault`.

## Known limitations

- Use TP1 for reproducibility checks. TP4 can run but does not produce reproducible images with the same seed.
- Do not enable `PYTORCH_MUSA_ALLOC_CONF=expandable_segments:True` during component loading.
- The H3-specific fused Triton norm+RoPE block currently uses the unfused fallback on MUSA.
- MATE RMSNorm reuse and fused QK-Norm+RoPE optimization are follow-up work.
- Image editing through `/v1/images/edits` is not covered by this recipe.
- HTTP serving through `/v1/images/generations` is not covered by this offline recipe.

This recipe uses the eager functional path and is not a throughput or compiled-serving benchmark.
