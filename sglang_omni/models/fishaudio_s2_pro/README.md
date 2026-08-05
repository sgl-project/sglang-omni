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
