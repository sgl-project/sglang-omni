# SGLang Day-0 Support for FishAudio S2 Text-to-Speech

## TL;DR

We are excited to announce SGLang's day-0 support for FishAudio S2, a frontier text-to-speech (TTS) model with high-quality voice cloning capabilities. By integrating S2's backbone into SGLang, we achieve an RTF of 0.34 and 63.3 tok/s on single H200 GPU at single batch size.

This work is a collaboration between the SGLang Omni Team and [FishAudio Team](https://fish.audio). We thank the FishAudio team for their support in model architecture and implementation detais.

Acknowledgments: Jingwen Gu, Yitong Guan, Xiaole Guo, Shidong Li, Shuai Shi, Junrong Lin, Fan Yin, Leng Yue, Shenggui Li, Chenyang Zhao

## Background and Motivation

Text-to-speech has converged on LLM-style autoregressive architectures: a transformer predicts discrete audio tokens, which a codec vocoder decodes into waveforms. It means TTS models face the same inference challenges as LLMs, including growing KV caches to be managed efficiently and the need for production-grade serving infrastructure.

FishAudio S2 is a leading example of this trend. Built on a Dual-autoregressive (Dual-AR) architecture, S2 achieves state-of-the-art quality across multiple benchmarks while supporting fine-grained inline control of prosody and emotion through natural-language tags. Trained on over 10 million hours of audio across approximately 100 languages and aligned with GRPO-based reinforcement learning, S2 tops the Audio Turing Test (0.515 posterior mean) and EmergentTTS-Eval (81.88% win rate against gpt-4o-mini-tts) while achieving the lowest word error rate (WER) on Seed-TTS Eval among all evaluated models including closed-source systems. For more details on S2's model design and training, see FishAudio's S2 release blog post.

 S2's Dual-AR architecture is structurally isomorphic to standard autoregressive LLMs, so it can directly inherit LLM-native serving optimizations with minimal modification, perfectly matching the strenghth of SGLang.

The integration challenge is that TTS models aren't pure text-in, text-out transformers. S2 interleaves VQ codebook embeddings into the token stream during decoding, runs multiple Fast AR decoder steps after each Slow AR step, and requires constrained decoding to enforce codebook structure. Integrating this into SGLang's runtime while preserving prefix caching required careful adaptation of the Model Runner and scheduling.

## Architecture

S2 uses a 3-stage pipeline:

```
Text input ──► Preprocessing ──► SGLang AR Engine ──► DAC Vocoder ──► Audio output
                 (CPU)              (GPU)               (GPU)
```

**Stage 1 — Preprocessing:** Tokenizes the input text into a Qwen3-style chat prompt. For voice cloning, it encodes the reference audio into VQ codes via the DAC codec and prepends them to the prompt as a system message.

**Stage 2 — Dual-AR Generation:** The Slow AR runs inside SGLang along the time axis. At each decode step, it predicts a semantic token, then the Fast AR (4-layer transformer) generates the remaining 9 residual codebook tokens conditioned on the hidden state. VQ embeddings are injected into the input embedding at masked positions, allowing the model to attend over both text and audio context through SGLang's KV cache. Startup constructs the Fast AR directly and strictly loads only the checkpoint's `audio_decoder.*` tensors; the unused Hugging Face Slow AR wrapper is not instantiated.

**Stage 3 — Vocoder:** The accumulated codebook indices are decoded into a waveform by a DAC codec, producing the final audio output.


## Usage

Please refer to [TTS Model Usage](https://github.com/sgl-project/sglang-omni/blob/main/docs/basic_usage/tts.md) for more details.

## Ascend NPU Support

S2-Pro can run on Ascend NPU through the sglang-omni platform abstraction:

- `attention_backend="ascend"` is selected automatically on NPU; single-token
  Fast-AR decode requires `torch.ops.npu.npu_fused_infer_attention_score`.
- Decode runs under NPU graph (`cuda_graph_backend_decode="full"`); the eager
  decode path is avoided because the ascend backend corrupts content under
  concurrent decode. `torch.compile` defaults to off on NPU because its
  interaction with the NPU fused operator and NPU graph has not yet been
  validated for correctness, memory use, or numerical stability. It can be
  enabled explicitly for validation with `engine.enable_torch_compile: true`.

  ```yaml
  stages:
    tts_engine:
      engine:
        enable_torch_compile: true
        torch_compile_max_bs: 16
  ```
- Cross-process payloads use SHM transport instead of CUDA IPC.
- The standard `sgl-omni serve --config examples/configs/s2pro_tts.yaml` command
  works unchanged; device strings are resolved by `resolve_device_spec`.

### NPU Validation

Start the service with:

```bash
sgl-omni serve --model-path fishaudio/s2-pro \
  --config examples/configs/s2pro_tts.yaml \
  --port 8000 --allowed-local-media-path .
```

Pure TTS, voice cloning, and streaming requests were checked through
`/v1/audio/speech`; all returned HTTP 200 and the generated content matched the
input text. For example, voice cloning was tested with:

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{"input":"The quick brown fox jumps over the lazy dog.","voice":"default","ref_audio":"ref.wav","ref_text":"Hello world! This is a Fish Audio S2-Pro reference.","response_format":"wav","max_new_tokens":1024}' \
  --output vc.wav
```

The full 1,088-utterance SeedTTS English set was measured at concurrency 16
after one warmup request:

```bash
python -m benchmarks.eval.benchmark_tts_seedtts \
  --use-existing-server --generate-only \
  --meta zhaochenyang20/seed-tts-eval-arrow \
  --max-concurrency 16 --model fishaudio/s2-pro --port 8000 \
  --output-dir results/s2pro_npu_en
```

| Hardware | Throughput (req/s) | Mean latency (s) | RTF | Success |
|---|---:|---:|---:|---:|
| Ascend 910, 64 GiB | 1.428 | 11.14 | 2.88 | 1088/1088 |
| H100 reference | 1.700 | 9.38 | 2.48 | 1088/1088 |
| H200 reference | 1.005 | 15.84 | 4.27 | 1088/1088 |

The CUDA rows are official reference results and are directional comparisons;
the accelerator and software stacks differ from the NPU run.

## Apple Silicon Support (Experimental)

S2-Pro also runs on Apple Metal, through a native MLX path
(`SGLANG_USE_MLX=1`) or a Torch/MPS compatibility path (`SGLANG_USE_MLX=0`).
`SGLANG_USE_MLX=1` off Apple Metal fails at startup rather than falling back
silently. Both share the Fish request, sampler, and stream adapters, the
`max_running_requests=1` admission profile, and the same rejection of quantized
weights and unqualified overrides.

Native MLX (`FishS2ProMlxModel`):

The shared MLX worker resolves Fish through the lazy runner registry. Its
`FishMlxWorkerAdapter` provides model initialization, the scheduler-facing model,
and the KV-release hook; Fish-specific dispatch is confined to its registration.

- The Slow AR transformer, its KV cache, and the entire Fast-AR greedy residual
  chain run in MLX. Only the semantic logits cross to CPU, for the shared Fish
  mask/RAS/top-k sampler. The residual chain then synchronizes once to publish
  all codebooks, instead of once per codebook.
- RoPE phases are taken from the canonical Torch `precompute_freqs_cis` and cast
  once to BF16, rather than recomputed in MLX, because recomputation can round
  across a BF16 boundary and drift from the CUDA reference.
- Weights load straight from the official checkpoint's safetensors under their
  own names, strictly: a missing, extra, or mis-shaped tensor is an error. There
  is no `mlx-audio` dependency and no converted MLX artifact.
- Fish owns its per-request MLX cache, so the worker skips SGLang's MLX
  KV-release bookkeeping and releases on scheduler completion/abort instead.

Torch/MPS (`S2ProTorchMpsTextModel`):

- The Slow AR runs as a native eager Torch module that keeps Fish's interleaved
  BF16 RoPE, VQ embedding injection, and codebook sampler, and replaces only the
  CUDA attention/KV-cache contract. One scheduler request owns the native cache
  at a time.
- Fast-AR attention appends one position per step into the dense NHD cache and
  attends over its initialized prefix with `scaled_dot_product_attention`, with
  GQA expanded explicitly because MPS has no grouped kernel.
- The seeded semantic sampler reproduces SGLang's MurmurHash3/Gumbel draw on CPU
  in int64: MPS supports neither uint64 nor float64, and the Triton kernel is
  unavailable. Only the small top-k distribution crosses to CPU, so seeded
  draws are deterministic for a fixed probability distribution.
- CUDA graphs, `torch.compile`, radix caching, and chunked prefill are disabled;
  `attention_backend="torch_native"` with `sampling_backend="pytorch"`. Quantized
  weights are rejected, and unqualified overrides fail fast rather than degrade.

### Apple Validation

Validation uses the official `fishaudio/s2-pro` checkpoint at revision
`1de9996b6be38b745688de084d87a5633f714e4e` on an Apple M5 Pro with 48 GB RAM.
The unit suite covers cached versus full prefill on CPU and MPS, Fast-AR cache
masking, strict weight loading, deterministic sampler vectors, and request-cache
cleanup; the MLX suite additionally checks the native Slow AR against the Torch
module, the native Fast-AR chain against the MPS decoder, and reference-codebook
mixing. All six live HTTP checks passed on both the MLX and the Torch/MPS
server. Three additional English plain-TTS, English reference-conditioned, and
Chinese smoke clips produced finite mono
44.1 kHz audio and stopped at EOS. Local Fun-ASR recovered the intended text
(ignoring punctuation). These short samples took about 3.4–3.6 seconds of wall
time per second of generated audio on Torch/MPS and 2.3–2.5 on MLX, excluding
ASR scoring; these are smoke measurements, not controlled throughput benchmarks. On the same host, steady-state
Slow-AR decode was about 10 tokens/s under MLX against about 8 tokens/s under
Torch/MPS, both at batch size 1.

Run the live checks against a server started with the Apple cookbook command:

```bash
FISH_APPLE_URL=http://127.0.0.1:8000 python -m pytest \
  tests/test_model/test_fish_apple.py -q
```

These exercise seeded repetition, reference-audio transport, streaming,
queued requests, and disconnect recovery, and are backend-agnostic: run them once
per `SGLANG_USE_MLX` setting. They do not measure speaker similarity,
corpus-level transcription accuracy, or cross-device numerical parity.
Production performance qualification remains future work.

The MLX numerical test compares MLX against Torch on a tiny random model. On M5,
MLX defaults to reduced-precision (TF32) FP32 GEMM, so that check runs against a
tolerance scaled to the logit magnitude; set `MLX_ENABLE_TF32=0` to run it at
full FP32 precision (2e-5).

## Optimizations with SGLang Omni

By integrating S2's Dual-AR backbone into SGLang's paged-attention engine, we inherit LLM-native optimizations:

- **Paged KV cache** — SGLang manages KV cache for the Slow AR path, enabling efficient memory usage and high concurrency.
- **Radix prefix caching** — Shared system prompt and reference audio prefixes are cached across requests, keeping TTFT (~18ms) and Time-to-First-Audio (~140ms) consistently low.
- **Decode CUDA Graph coverage of Slow AR and Fast AR** — Bounded decode graph capture and replay have been validated on SM89 and SM120 for batch sizes 1, 2, and 4 with `torch.compile` disabled. FlashInfer-backed SM89/SM100/SM120 profiles therefore default to uncompiled Fast-AR layers while retaining CUDA Graph capture; prefill graphs, larger batches, and CUDA Graph execution with `torch.compile` remain unvalidated. SM90 retains the existing compile default. Details on the dual-AR graph design are available in [Revisiting CUDA Graph: Core Mechanisms, Multi-Graph Memory Sharing, and Unified Coverage for Dual AR Models](https://github.com/zhaochenyang20/Awesome-ML-SYS-Tutorial/blob/main/torch/cuda-graph/readme-2-en.md).
- **Architecture-aware attention** — S2-Pro selects FA3 on SM90 and FlashInfer on SM89, SM100, and SM120 so neither attention path enters an unsupported kernel.

### Attention Backend Policy

S2-Pro has separate attention implementations for the SGLang-hosted Slow-AR path and the Fast-AR decoder's dense NHD KV cache:

| Compute capability | Slow-AR | Fast-AR KV cache |
|---|---|---|
| SM89 | FlashInfer | FlashInfer |
| SM90 | FA3 | FA3 |
| SM100 | FlashInfer | FlashInfer |
| SM120 | FlashInfer | FlashInfer |

An explicit `attention_backend` setting overrides automatic selection only for Slow-AR. Fast-AR uses its own architecture policy because it dispatches a separate KV-cache kernel. Do not force FA3 on SM89, SM100, or SM120: FA3 has no compatible S2-Pro kernel on those architectures. Unsupported compute capabilities fail at startup with an actionable error.

## Future Optimization

To further improve throughput and latency in the future:

- **Validate CUDA Graphs with `torch.compile` enabled.** Decode CUDA Graph capture and replay are validated for batch sizes 1, 2, and 4 with `torch.compile` disabled. Their correctness, memory usage, and numerical behavior when `torch.compile` is enabled still require separate validation.

- **Batched Fast AR head processing.** Currently, the Fast AR codebook decoding loop runs sequentially per request. Batching these steps across concurrent requests would improve GPU utilization at higher batch sizes potentially improving throughput.

## Engineering Appendix

<details>
<summary>Engineering Appendix</summary>

### BF16 RoPE Precision Mismatch

SGLang's default RoPE implementation precomputes `cos_sin_cache` in float32, but S2's model was trained entirely in bfloat16 including the RoPE frequencies. The precision difference caused logit divergence producing garbled audio with abnormal long sequence of tokens.

It's worth attention for any future engineering for fish audio inference infrastructure, since it's uncommon and hard to debug when accuracy of inference engine is higher than the precision of the model. Below is a simple fix once problem identified.

```python
def _truncate_rope_to_bf16(model: torch.nn.Module) -> None:
    for module in model.modules():
        if hasattr(module, "cos_sin_cache"):
            module.cos_sin_cache.data = module.cos_sin_cache.data.to(torch.bfloat16).to(
                torch.float32
            )
```

### Historical Attention Backend Divergence Causing Early Stopping

Historically, SGLang defaulted to FlashInfer for attention while S2 was trained with FlashAttention, and early EOS was observed during development. This observation was not a controlled or quality-qualified comparison and is retained only as historical debugging context.

</details>
