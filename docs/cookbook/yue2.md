# YuE2

[YuE2](https://huggingface.co/m-a-p/YuE2-3B) generates 48 kHz stereo music from lyrics and a style description. Its shared transformer predicts an ABC score and semantic tokens autoregressively, then generates acoustic latents with flow matching. A separate FP32 VAE decodes the audio.

## Start the server

Install SGLang-Omni following [Installation](../get_started/installation.md), then start one CUDA worker:

```bash
CUDA_VISIBLE_DEVICES=0 sgl-omni serve \
  --model-path m-a-p/YuE2-3B \
  --port 8000
```

The worker loads the BF16 transformer and FP32 decoder once. By default it processes queued requests sequentially. Each request owns its sampling generators, lengths and sampling history; optional AR batching is described below.

Both the transformer and decoder accept local checkpoint directories or Hugging Face model identifiers. The default decoder is `m-a-p/YuE2-Vae`. Select the benchmark decoder with `--synthesis.factory.vae_path m-a-p/YuE2-Vae-legacy`. Checkpoint identifiers can pin a revision with `repository@revision`.

## Generate a song

```bash
curl http://localhost:8000/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "m-a-p/YuE2-3B",
    "input": "[Verse]\nMorning light is shining through\nEvery road leads back to you",
    "instructions": "Gentle acoustic pop, warm female vocals, piano and guitar",
    "seed": 42,
    "max_new_tokens": 9000,
    "response_format": "wav"
  }' \
  --output song.wav
```

`input` is plain lyrics and `instructions` is the musical style. `seed` controls each generation phase independently. `max_new_tokens` limits semantic audio tokens, excluding the ABC planning phase; it is a token budget, not a requested audio duration. The semantic sampling fields `temperature`, `top_p`, `top_k` and `repetition_penalty` are supported.

The response's `X-Finish-Reason` is `length` if either planning or semantic generation reaches its token budget; it is `stop` when both phases finish normally. Increasing the semantic budget cannot repair a truncated ABC plan: adjust `abc_max_tokens` for that phase. Requests that exceed the model context fail instead of silently shortening their budgets.

## Control the score and acoustic solver

Model-specific settings use the existing `stage_params` field:

```json
{
  "input": "[Verse]\nMorning light is shining through",
  "instructions": "Acoustic folk, gentle vocals",
  "seed": 42,
  "stage_params": {
    "synthesis": {
      "cot": "melody",
      "abc": "X:1\nT:Morning\nM:4/4\nL:1/4\nK:C\nC D E F | G A B c |",
      "ode_steps": 32
    }
  }
}
```

| Setting | Meaning |
| --- | --- |
| `cot` | `full` plans melody and chords; `melody` plans melody; `off` skips score planning. Default: `full`. |
| `abc` | An optional supplied score, used without generating a new one. Requires `full` or `melody`. |
| `abc_max_tokens` | Planning budget. Default: 4096. |
| `cfg_scale` | Semantic classifier-free guidance. Defaults to 1.0 for score modes and 1.01 for `off`, matching the model's generation protocol. |
| `ode_steps` | Midpoint flow-matching steps. Default: 32. |

Voice selection, reference-audio conditioning, speed changes and streaming output are unsupported. Cancellation is checked between AR tokens, acoustic solver evaluations and VAE tiles; an already submitted GPU operation finishes before cancellation takes effect.

## Optional acceleration

The default path uses eager execution. On NVIDIA CUDA devices, enable the separately selectable optimizations with:

```bash
sgl-omni serve --model-path m-a-p/YuE2-3B \
  --synthesis.factory.ar_cuda_graph true \
  --synthesis.factory.nar_cuda_graph true \
  --synthesis.factory.sampling_cuda_graph true \
  --synthesis.factory.fuse_projections true
```

AR decode graphs reserve one full-context KV buffer per guidance branch and reuse it across requests. NAR graphs capture the velocity evaluation for each acoustic chunk. Sampling graphs preserve the vocabulary mask, top-k ties, repetition window and request-local random generator. Projection fusion combines QKV and gate/up projections without changing checkpoint names.

Set `--synthesis.factory.fused_kernels true` to use `sgl_kernel` for RMS normalization, residual addition with normalization, Q/K normalization with RoPE, and SiLU activation with gating in both AR and NAR. This also enables projection fusion and works with eager execution or CUDA graphs.

AR graph attention uses the packed FlashAttention path, matching the model's reference graph implementation. Graph attention, projection fusion and fused kernels can change BF16 rounding and sampled songs. Fused kernels also change acoustic latents; full song-quality evaluation is still required before changing the defaults.

### Optional AR batching

With AR and sampling graphs enabled, set `--synthesis.factory.max_batch_size 4` to batch concurrent score planning and semantic generation. `max_batch_wait_ms` controls the collection window (default 25 ms). `batch_graph_cache_size` bounds the number of cached batch shapes (default 2); each shape reserves KV capacity for its request count and token budgets.

Requests retain independent seeds, sampling parameters, token budgets and cancellation state. Invalid requests fail individually. External ABC and `cot=off` with `cfg_scale=1` can join a batch; requests with guidance other than 1 use the single-request path. Acoustic synthesis and VAE decoding remain sequential. Batch results are returned after all active requests finish, so throughput gains can increase a short request's latency when it shares a batch with a long song.

Batch shapes can change BF16 rounding and sampled tokens even with the same seed. Compare both song quality and request latency under the intended concurrency before enabling batching in production.

## Limitations

The official YuE2 implementation offers additional [optional execution modes](https://github.com/multimodal-art-projection/YuE/blob/1116bea76032bee5668e5f097294d082fa0b84f0/docs/hardware.md) that are not currently supported by this Omni integration:

- Experimental AR FP8 quantization.
- Low-VRAM weight offloading.
- AR decoding with torch.compile.

## Validation

Run the CPU regression suite with:

```bash
pytest tests/unit_test/yue2 -q
```

Run the real-checkpoint smoke tests on one CUDA GPU with:

```bash
pytest tests/test_model/test_yue2.py -q
```

These short tests cover the score modes, external ABC, CFG, cancellation and seeded recovery. They do not measure song quality. Set `YUE2_MODEL_PATH` and `YUE2_VAE_PATH` to use local or revision-pinned checkpoints.

Quality comparisons must use the same transformer revision, VAE revision, prompts, seeds and generation settings. The official [WildSongBench protocol](https://huggingface.co/datasets/m-a-p/WildSongBench) uses `YuE2-Vae-legacy`; scores from the standard decoder are not directly comparable to that baseline. Keep truncated outputs in evaluation and report them separately.
