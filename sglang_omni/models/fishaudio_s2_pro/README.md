# SGLang Day-0 Support for FishAudio S2 Text-to-Speech

## TL;DR

We are excited to announce SGLang's day-0 support for FishAudio S2, a frontier text-to-speech model with high-quality voice cloning capabilities. By integrating S2's backbone into SGLang, we achieve an RTF of 0.34 and 63.3 tok/s on single H200 GPU at single batch size.

This work is a collaboration between the SGLang Omni Team and [FishAudio Team](https://fish.audio). We thank the FishAudio team for their support in model architecture and implementation detais.

Acknowledgments: Jingwen Gu, Yitong Guan, Xiaole Guo, Shidong Li, Shuai Shi, Junrong Lin, Fan Yin, Leng Yue, Shenggui Li, Chenyang Zhao

## Background and Motivation

Text-to-speech has converged on LLM-style autoregressive architectures: a transformer predicts discrete audio tokens, which a codec vocoder decodes into waveforms. It means TTS models face the same inference challenges as LLMs, including growing KV caches to be managed efficiently and the need for production-grade serving infrastructure.

FishAudio S2 is a leading example of this trend. Built on a Dual-AR architecture, S2 achieves state-of-the-art quality across multiple benchmarks while supporting fine-grained inline control of prosody and emotion through natural-language tags. Trained on over 10 million hours of audio across approximately 100 languages and aligned with GRPO-based reinforcement learning, S2 tops the Audio Turing Test (0.515 posterior mean) and EmergentTTS-Eval (81.88% win rate against gpt-4o-mini-tts) while achieving the lowest WER on Seed-TTS Eval among all evaluated models including closed-source systems. For more details on S2's model design and training, see FishAudio's S2 release blog post.

 S2's Dual-AR architecture is structurally isomorphic to standard autoregressive LLMs, so it can directly inherit LLM-native serving optimizations with minimal modification, perfectly matching the strenghth of SGLang.

The integration challenge is that TTS models aren't pure text-in, text-out transformers. S2 interleaves VQ codebook embeddings into the token stream during decoding, runs multiple Fast AR decoder steps after each Slow AR step, and requires constrained decoding to enforce codebook structure. Integrating this into SGLang's forward path while preserving prefix caching required careful adaptation of the Model Runner and scheduling.

## Architecture

S2 uses a 3-stage pipeline:

```
Text input ──► Preprocessing ──► SGLang AR Engine ──► DAC Vocoder ──► Audio output
                 (CPU)              (GPU)               (GPU)
```

**Stage 1 — Preprocessing:** Tokenizes the input text into a Qwen3-style chat prompt. For voice cloning, encodes the reference audio into VQ codes via the DAC codec and prepends them to the prompt as a system message.

**Stage 2 — Dual-AR Generation:** The Slow AR runs inside SGLang along the time axis. At each decode step, it predicts a semantic token, then the Fast AR (4-layer transformer) generates the remaining 9 residual codebook tokens conditioned on the hidden state. VQ embeddings are injected into the input embedding at masked positions, allowing the model to attend over both text and audio context through SGLang's KV cache.

**Stage 3 — Vocoder:** The accumulated codebook indices are decoded into a waveform by a DAC codec, producing the final audio output.


## Performance

Evaluated on the full seed-tts-eval EN testset (1,088 samples) on a single H200 GPU.

| Metric | BS=1 | BS=2 | BS=4 | BS=8 |
|---|---|---|---|---|
| Tok/s (mean) | 63.3 | 45.8 | 31.9 | 19.6 |
| RTF (mean) | 0.340 | 0.473 | 0.676 | 1.097 |
| Latency (mean) | 1.33s | 1.80s | 2.69s | 4.36s |
| TTFT (mean) | 19.6 ms | 22.0 ms | 31.6 ms | 50.7 ms |
| TTFB (mean) | 172.8 ms | 249.9 ms | 319.1 ms | 509.6 ms |

## Installation and Quick Start

### Docker

```bash
docker pull frankleeeee/sglang-omni:dev

docker run -it --shm-size 32g --gpus all frankleeeee/sglang-omni:dev /bin/zsh
```

### Install sglang-omni (inside Docker)

```bash
git clone https://github.com/sgl-project-dev/sglang-omni.git
cd sglang-omni
uv venv .venv -p 3.12 && source .venv/bin/activate
uv pip install -v ".[s2pro]"
huggingface-cli download fishaudio/s2-pro
```

### Playground and Server

We provide a Gradio-based interactive playground and a server for production deployment. We highly recommend using playground since audio data is hard to intertact with by CLI.

1. Interactive Playground

```bash
 ./playground/tts/start.sh
```

2. Server

```bash
python -m sglang_omni.cli.cli serve \
    --model-path fishaudio/s2-pro \
    --config examples/configs/s2pro_tts.yaml \
    --port 8000
```

<details>
<summary>curl commands</summary>

1. Text-to-Speech

Note that without reference audio, the generated audio sounds like a robot.

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
    -H "Content-Type: application/json" \
    -d '{"input": "Hello, how are you?"}' \
    --output output.wav
```

2. Voice Cloning

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
    -H "Content-Type: application/json" \
    -d '{
        "input": "Hello, how are you?",
        "references": [{"audio_path": "ref.wav", "text": "Transcript of ref audio."}]
    }' \
    --output output.wav
```

</details>

We highly recommend using playground since audio data is hard to intertact with by CLI.

## Optimizations with SGLang Omni

By integrating S2's Dual-AR backbone into SGLang's paged-attention engine, we inherit LLM-native optimizations:

- **Paged KV cache** — SGLang manages KV cache for the Slow AR path, enabling efficient memory usage and high concurrency.
- **Radix prefix caching** — Shared system prompt and reference audio prefixes are cached across requests, keeping TTFT consistently low (~18ms).
- **torch.compile on Fast AR** — The 9-step codebook loop is compiled with torch.compile, achieving 5x speedup over eager mode.
- **FlashAttention 3** — Forced FA3 backend to match training-time attention numerics, avoiding early-EOS divergence from flashinfer.

## Future Optimization

To further improve throughput and latency in the future:

- **CUDA Graphs while torch.compile enabled.** The current implementation uses torch.compile on the Fast AR codebook loop (achieving 5x over eager), but does not capture CUDA graphs for the Slow AR path. Enabling CUDA graphs requires resolving numerical divergence from deterministic-mode constraints and adapting SGLang's graph capture to S2's interleaved VQ embedding injection, involving significant engineering that we leave for a future release.

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

### Attention Backend Divergence Causing Early Stopping

SGLang defaults to flashinfer for attention, but S2 was trained with FlashAttention. When future engineering meet early EOS token issue, this could suggest the fix.

</details>
