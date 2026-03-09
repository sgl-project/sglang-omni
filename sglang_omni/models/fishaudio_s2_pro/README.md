# FishAudio S2 Text-to-Speech

SGLang's day-0 support for [FishAudio S2](https://fish.audio), a frontier text-to-speech model with high-quality voice cloning capabilities. S2 tops the Audio Turing Test and EmergentTTS-Eval while achieving the lowest WER on Seed-TTS Eval among all evaluated models including closed-source systems, with support for ~100 languages and fine-grained prosody/emotion control via natural-language tags.

## Installation

```bash
# Docker (recommended)
docker pull frankleeeee/sglang-omni:dev
docker run -it --shm-size 32g --gpus all \
    -v /data/cache/huggingface:/root/.cache/huggingface \
    --ipc=host --privileged \
    frankleeeee/sglang-omni:dev /bin/zsh

# Inside Docker
git clone git@github.com:sgl-project-dev/sglang-omni.git
cd sglang-omni
uv venv .venv -p 3.12 && source .venv/bin/activate
uv pip install msgpack
uv pip install -v ".[s2pro]"
huggingface-cli download fishaudio/s2-pro
```

## Web UI (Playground)

```bash
./playground/tts/start.sh
```

This launches the backend server and a Gradio UI. Options:

```bash
./playground/tts/start.sh --port 8080 --gradio-port 7861 --share
```

## Server

```bash
python -m sglang_omni.cli.cli serve \
    --model-path fishaudio/s2-pro \
    --config examples/configs/s2pro_tts.yaml \
    --port 8000
```

### API — Text-to-Speech

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
    -H "Content-Type: application/json" \
    -d '{"input": "Hello, how are you?"}' \
    --output output.wav
```

### API — Voice Cloning

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
    -H "Content-Type: application/json" \
    -d '{
        "input": "Hello, how are you?",
        "references": [{"audio_path": "ref.wav", "text": "Transcript of ref audio."}]
    }' \
    --output output.wav
```

## Optimizations

By integrating S2's Dual-AR backbone into SGLang's paged-attention engine, we inherit LLM-native optimizations:

- **Paged KV cache** — SGLang manages KV cache for the Slow AR path, enabling efficient memory usage and high concurrency.
- **Radix prefix caching** — Shared system prompt and reference audio prefixes are cached across requests, keeping TTFT consistently low (~18ms).
- **torch.compile on Fast AR** — The 9-step codebook loop is compiled with torch.compile, achieving 5x speedup over eager mode.
- **FlashAttention 3** — Forced FA3 backend to match training-time attention numerics, avoiding early-EOS divergence from flashinfer.
- **BF16 RoPE truncation** — RoPE frequencies are round-tripped through bfloat16 to match S2's training precision, preventing logit divergence.

## Performance

Evaluated on the full seed-tts-eval EN testset (1,088 samples) on a single H200 GPU.

| Metric | BS=1 |
|---|---|
| Tok/s (mean) | 61.6 |
| RTF (mean) | 0.352 |
| Latency (mean) | 1.43s |
| TTFT (mean) | 18.0 ms |
| TTFB (mean) | 176.0 ms |

## Future Optimization

- **CUDA Graphs + torch.compile** — Capture CUDA graphs for the Slow AR decode path alongside torch.compile on the Fast AR loop. Requires resolving numerical divergence from deterministic-mode constraints and adapting graph capture to S2's interleaved VQ embedding injection.
- **Batched Fast AR processing** — Batch the 9-step codebook loop across concurrent requests to improve GPU utilization at higher batch sizes.
