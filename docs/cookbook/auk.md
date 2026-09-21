# AuK

[AuK](https://huggingface.co/tencent/AuK) and [AuK-Flash](https://huggingface.co/tencent/AuK-Flash) support instruction-driven speech generation and editing. They share a four-stage pipeline: preprocessing → conditioning → DiT sampling → VAE decoding. AuK-Flash is the DMD-distilled four-step recipe.

The released checkpoints use:

| Component | Configuration |
|---|---|
| Conditioner | Frozen Qwen2.5-Omni-3B Thinker, text and audio only |
| DiT | Flux-style MMDiT: 10 double-stream blocks, 20 single-stream blocks, dim=1536, 24 heads |
| VAE | Shared reference encoder and audio decoder; 50 Hz, 64-channel latents |

| Checkpoint | Sampling |
|---|---|
| [`tencent/AuK`](https://huggingface.co/tencent/AuK) | Euler, NFE=32, CFG=2.0, sway=-1.0 |
| [`tencent/AuK-Flash`](https://huggingface.co/tencent/AuK-Flash) | Released four-step time grid, CFG=0. Factory `nfe` / `cfg_strength` / `sway_sampling_coef` are ignored |

## Prerequisites

Follow [Installation](../get_started/installation.md), then run from the repository root:

```bash
python -m sglang_omni.cli serve --model-path tencent/AuK --port 8000
```

```bash
python -m sglang_omni.cli serve --model-path tencent/AuK-Flash --port 8000
```

## Speech Generation

`/v1/audio/speech` takes the text in `input`. Without reference audio, `instructions` describes the voice and defaults to `A clear, natural voice.` An explicit target duration is required in this mode:

```bash
curl http://localhost:8000/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{
    "input": "Welcome home.",
    "instructions": "warm, relaxed female voice",
    "stage_params": {"auk_engine": {"gen_seconds": 3}},
    "seed": 1234,
    "response_format": "wav"
  }' --output speech.wav
```

For voice cloning, provide `ref_audio` or one structured reference. The model uses a same-voice instruction and ignores the voice description. HTTP(S) URLs, audio data URLs, and server-local paths are accepted. To use a `file://` URL, start the server with `--allowed-local-media-path /abs/reference-dir` and reference a file within that directory, such as `file:///abs/reference-dir/reference.wav`. Without this flag, `file://` references return HTTP 400.

```bash
curl http://localhost:8000/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{
    "input": "Welcome home.",
    "ref_audio": "https://huggingface.co/datasets/zhaochenyang20/seed-tts-eval-mini/resolve/main/en/prompt-wavs/common_voice_en_10119832.wav",
    "ref_text": "We asked over twenty different people, and they all said it was his.",
    "seed": 1234,
    "response_format": "wav"
  }' --output speech.wav
```

When `gen_seconds` is omitted, voice cloning requires the reference transcript (`ref_text`, or `references[0].text`). Target duration is estimated as:

```text
target_seconds = reference_seconds × UTF8_bytes(input) / UTF8_bytes(ref_text)
```

Explicit `gen_seconds` takes priority and must be positive. Target duration rounds up to 20 ms frames and is capped at 30 seconds by default. To change the cap, set `max_seconds` on both stages: `--preprocessing.factory.max_seconds` and `--auk_engine.factory.max_seconds`.

## Speech Editing

`/generate` accepts a raw AuK instruction in `prompt` and returns JSON. Set `output_modalities` to `["audio"]` and `return_logprob` to `false` (AuK does not produce token log probabilities). Supply reference audio through `metadata.tts_params.ref_audio`:

```bash
curl http://localhost:8000/generate \
  -H 'Content-Type: application/json' \
  -d '{
    "prompt": "Remove the background noise.",
    "metadata": {"tts_params": {"ref_audio": "https://huggingface.co/datasets/zhaochenyang20/seed-tts-eval-mini/resolve/main/en/prompt-wavs/common_voice_en_10119832.wav"}},
    "output_modalities": ["audio"],
    "return_logprob": false
  }'
```

Override duration with `stage_params.auk_engine.gen_seconds`. Otherwise, editing uses the source's complete 20 ms frames, subject to the duration cap. Raw requests without reference audio or explicit duration default to 5 seconds.

## Sampling

Base AuK uses Euler integration with factory defaults `nfe=32`, `cfg_strength=2.0`, and `sway_sampling_coef=-1.0`. Override them with `--auk_engine.factory.*` flags. Flash locks to the released four-step grid with CFG disabled, so those flags have no effect on `tencent/AuK-Flash`. Request overrides of these settings and `max_seconds` are rejected. Qwen uses BF16 autocast; the VAE runs in FP32.

The DiT stores its weights in BF16 and runs without autocast by default (`--auk_engine.factory.weight_dtype bfloat16`), which removes the per-step FP32-to-BF16 weight casts. Set `--auk_engine.factory.weight_dtype float32` to keep FP32 weights with BF16 autocast instead; that is the upstream-exact recipe the parity test compares against, at roughly 1.3x the sampling time. The ODE state is integrated in FP32 in both modes.

Multi-request DiT batches run packed by default. The transformer blocks run
only valid text/reference/target tokens through Linear, Norm, FFN and non-
causal variable-length FlashAttention. Original position IDs and separate CFG
branches are preserved. Convolutional audio embeddings and the FP32 Euler state
retain their padded layout. Layout indices, rotary tables, text projections,
reference embeddings and timestep embeddings are built once per trajectory
and handed to every Euler step as one plan.
Singleton batches retain the padded execution path. Packing requires a CUDA
BF16 backbone, a checkpoint with `attn_mask_enabled`, and SGLang's varlen
FlashAttention. The version follows SGLang's own predicates: FA4 wherever
`is_blackwell` holds (sm100/sm110/sm120, CUDA >= 12.8; sm12x binds SGLang's
`flash_attention_v4_sm120` entry point as its FlashAttentionBackend does) and
FA3 wherever `_is_fa3_supported` holds (sm80-sm90, CUDA >= 12.3). FA4 on
sm120 (RTX 5090) has been validated against the padded path; FA3 and FA4 on
sm100 have not been exercised on hardware yet. The stage factory checks the
checkpoint flag and runs one probe attention call before loading the DiT
weights. When a check
fails (`weight_dtype float32`, HIP, other devices, a missing kernel build) the
engine logs the reason and keeps the padded path; set
`--auk_engine.factory.enable_packed_dit true` to turn that into a startup
error, or `false` to keep the padded path unconditionally. Packing does not
enable CUDA graph capture. Different attention and GEMM reductions can change
generated waveforms relative to the padded path.

`seed` initializes separate request-local generators for target noise and reference VAE posterior sampling, without changing the process RNG. Sampling is reproducible for fixed inputs; different batch shapes or compute backends can still produce numerical differences. Multiple structured references are rejected.

Conditioning and DiT sampling use dynamic batching, with default maximum batch sizes of 8 and 16. VAE decoding groups equal-length latents (up to 4 requests) to preserve boundary behavior. The stages can overlap on separate CUDA streams and share VAE weights within the same process/device. Conditioning loads the Qwen encoder, the VAE and the two hidden-state fusion parameters; only the sampling stage loads the DiT. Set `--conditioning.factory.max_batch_size`, `--auk_engine.factory.max_batch_size`, or `--decode.factory.max_batch_size` to tune them. Audio is returned after decoding completes; incremental audio streaming is not implemented.

## DiT Q/K fusion

On CUDA, the DiT uses a Triton kernel that fuses per-head RMSNorm with
interleaved rotary embedding. Disable it to use the native PyTorch path:

```bash
python -m sglang_omni.cli serve --model-path tencent/AuK \
  --auk_engine.factory.enable_dit_fused_qk_norm_rope false
```

The kernel derives the head dimension and output dtype from the model and
serves both the padded and the packed DiT paths. It leaves the conditioner,
VAE, and sampling recipe unchanged. Non-CUDA devices and AuK-Flash use the
native path. The first request may include Triton JIT compilation; the 32-step
AuK checkpoint has been validated on H100 with FP32 weights under BF16 autocast
and with native BF16 weights.

## SeedTTS Evaluation

The standard benchmark detects `tencent/AuK` and `tencent/AuK-Flash` and starts the server from `--model-path`. It defaults to the full English dataset, concurrency 1, one warmup, and seed 1234. It estimates duration from the reference audio and transcript, then automatically starts and stops the TTS and ASR servers:

```bash
CUDA_VISIBLE_DEVICES=0 python -m benchmarks.eval.benchmark_tts_seedtts \
  --model tencent/AuK --output-dir results/auk_en
```

Add `--concurrency 16` to evaluate with 16 in-flight requests. Use `--max-samples` and `--sample-offset` for a subset. `--generate-only` and `--transcribe-only` run individual phases; add `--use-existing-server` to either mode to use a running server. Explicit CLI options override the AuK defaults.

`wer_results.json` includes full sample mean WER, `wer_below_50_per_sample_mean` (excluding samples strictly above 50%), and `n_above_50_pct_wer`. Corpus WER is reported separately and is word-weighted.

## Upstream Parity

The checkpoint test compares reference latents, fused Qwen conditioning, generated latents, and waveforms with upstream, with and without reference audio. It aligns the upstream process RNG with the request seed to compare the same random inputs. Install `torchdiffeq`, `qwen-omni-utils`, and `audioread` in addition to the serving dependencies, and use a GPU with memory for both implementations:

```bash
pip install torchdiffeq qwen-omni-utils audioread
git clone https://github.com/Tencent-Hunyuan/AuK.git /tmp/AuK
git -C /tmp/AuK checkout d9f30ffe4231dbc90b48cc83a35d310fece0b060
AUK_UPSTREAM_SOURCE=/tmp/AuK \
AUK_PARITY_CHECKPOINT=tencent/AuK \
python -m pytest tests/test_model/test_auk_parity.py -v
```

```bash
AUK_UPSTREAM_SOURCE=/tmp/AuK \
AUK_PARITY_CHECKPOINT=tencent/AuK-Flash \
python -m pytest tests/test_model/test_auk_parity.py -v
```

`AUK_QWEN_CHECKPOINT` optionally selects a local encoder. The test skips unless both `AUK_UPSTREAM_SOURCE` and `AUK_PARITY_CHECKPOINT` are set.

## Attribution

The implementation derives from [Tencent-Hunyuan/AuK](https://github.com/Tencent-Hunyuan/AuK) at the revision above. Its MIT notice is preserved in `sglang_omni/models/auk/LICENSE`. The VAE source also retains NVIDIA and alias-free-torch attribution.
