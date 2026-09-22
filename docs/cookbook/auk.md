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

## Apple Silicon (native MLX)

After the [Apple Silicon installation](../get_started/installation.md), enable
the native backend for either official checkpoint:

```bash
SGLANG_USE_MLX=1 python -m sglang_omni.cli serve \
  --model-path tencent/AuK-Flash --port 8000
```

Replace the model path with `tencent/AuK` for the base model. The loader reads
the original AuK and `Qwen/Qwen2.5-Omni-3B` checkpoints directly. To use a local
conditioner, set `--conditioning.factory.text_encoder_path /path/to/Qwen2.5-Omni-3B`.
No `mlx-audio` package is needed. Text/audio conditioning, DiT sampling and VAE
encoding/decoding execute in MLX; audio preprocessing and stage payloads use CPU
tensors. The existing speech, cloning and editing APIs remain the same.

Apple defaults to one request per stage batch. Larger batches can be selected
with each stage's `max_batch_size` after checking unified-memory use. Outputs are
complete waveforms; incremental audio streaming is not implemented. CUDA graph,
Torch compilation and CUDA fusion settings do not select MLX kernels.

The conditioner and DiT use BF16 matrix weights by default. Text residuals,
hidden-state fusion, rotary positions, VAE computation and Euler integration
retain FP32. For a strict FP32 comparison, set `MLX_ENABLE_TF32=0` before Python
starts and set all three options to `float32`:
`--conditioning.factory.dtype`, `--auk_engine.factory.dtype`, and
`--auk_engine.factory.weight_dtype`. The MLX engine requires matching compute
and weight precision. This path currently supports BF16 and FP32, not FP16 or
quantized checkpoints.

Reference audio uses posterior sampling, matching the original PyTorch recipe.
Reference and generation noise have independent request-local streams. A fixed
seed is reproducible within a backend; it does not imply identical Torch/MLX
random samples or bitwise equality across batch shapes.

Run component tests and opt-in real-checkpoint parity on an Apple GPU:

```bash
MLX_ENABLE_TF32=0 python -m pytest tests/unit_test/auk/test_mlx_*.py -q
AUK_MLX_CHECKPOINT=/path/to/AuK-Flash \
AUK_QWEN_CHECKPOINT=/path/to/Qwen2.5-Omni-3B MLX_ENABLE_TF32=0 \
  python -m pytest tests/test_model/test_auk_mlx.py -q
AUK_MLX_SERVER_URL=http://localhost:8000 \
  python -m pytest tests/test_model/test_auk_mlx_http.py -q
```

Repeat the checkpoint and HTTP tests with the base model. Component parity uses
identical weights and explicit noise, and complements end-to-end speech-quality
evaluation; it does not establish quality or throughput on its own.

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

Two further sampling-speed options are on by default. `--auk_engine.factory.enable_dit_torch_compile` compiles each transformer block once for all blocks of its kind, folding the modulation, gating, and normalization chain into the matmul stream. `--auk_engine.factory.enable_dit_cuda_graph` captures one whole Euler step as a CUDA graph and replays it for every NFE step: only the noised latent and the timestep change between steps, so one capture serves a whole trajectory, and every later request whose shape it covers. Set either to `false` to fall back. The graph removes op-issue time and nothing else, so what it is worth depends on whether issuing a step costs more than running it; benchmark it on your own card before assuming it does.

Graph capture requires padding, so a batch pads up to a declared capture shape -- a `(batch, frames, reference frames, text tokens)` tuple -- and only to one that covers it on every axis. The default list is the cross product of five frame rungs (192 to 768 frames, roughly 4s to 15s of audio), two conditioning profiles (an instruction alone, and a cloned voice carrying its reference's own tokens), and batch sizes 1 and 2; set your own with `--auk_engine.factory.dit_cuda_graph_capture_shapes`. A batch no declared shape covers -- a long reference, a wide batch -- runs the step eagerly and unpadded, since padding only pays for itself when it buys a replay.

Padded rows are computed and discarded, not blended in: the attention bias sends masked keys to `-inf`, the convolutional position embedding masks its own input, and rope positions come from the real lengths, so a padded batch produces what the unpadded one does. `enable_dit_cuda_graph` is refused on a checkpoint whose config disables `attn_mask_enabled`, since that bias is what makes padding neutral. Block compilation and every declared capture are paid at startup rather than under the first request that needs each shape, because entering a capture synchronizes the whole device and empties the allocator cache while the conditioning and decode stages are running in the same process. A shape left without enough free device memory to capture logs a warning and runs eagerly; trim the list if startup time or graph memory matters. Sampling stays reproducible for a fixed input shape, but a different padded width is a different numerical path.

`seed` initializes separate request-local generators for target noise and reference VAE posterior sampling, without changing the process RNG. Sampling is reproducible for fixed inputs; different batch shapes or compute backends can still produce numerical differences. Multiple structured references are rejected.

Conditioning and DiT sampling use dynamic batching, with default maximum batch sizes of 8 and 16. VAE decoding groups equal-length latents (up to 4 requests) to preserve boundary behavior. The stages can overlap on separate CUDA streams and share VAE weights within the same process/device. Conditioning loads the Qwen encoder, the VAE and the two hidden-state fusion parameters; only the sampling stage loads the DiT. Set `--conditioning.factory.max_batch_size`, `--auk_engine.factory.max_batch_size`, or `--decode.factory.max_batch_size` to tune them. Audio is returned after decoding completes; incremental audio streaming is not implemented.

## DiT Q/K fusion

On CUDA, the DiT uses a Triton kernel that fuses per-head RMSNorm with
interleaved rotary embedding. Disable it to use the native PyTorch path:

```bash
python -m sglang_omni.cli serve --model-path tencent/AuK \
  --auk_engine.factory.enable_dit_fused_qk_norm_rope false
```

The kernel derives the head dimension and output dtype from the model. It
leaves the conditioner, VAE, and sampling recipe unchanged. Non-CUDA devices
and AuK-Flash use the native path. The first request may include Triton JIT
compilation; the 32-step AuK checkpoint has been validated on H100 with FP32
weights under BF16 autocast and with native BF16 weights.

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
