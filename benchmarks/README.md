# SGLang Omni Benchmarks

Benchmark suite for SGLang Omni, covering performance (latency, throughput, RTF)
and accuracy (WER) across supported modality combinations.

## Directory Structure

```
benchmarks/
├── tasks/          # Task definitions (voice_clone, tts_speed)
├── metrics/        # Atomic evaluation tools (wer, performance)
├── dataset/        # Dataset loaders + download helpers
├── benchmarker/    # Framework: runner, data structures, utilities
├── eval/           # Entry-point scripts (one per task x model)
├── cache/          # (gitignored) dataset caches
└── results/        # (gitignored) evaluation outputs
```

## Quick Start

```bash
# 1. Start the server
python -m sglang_omni.cli.cli serve \
    --model-path fishaudio/s2-pro \
    --config examples/configs/s2pro_tts.yaml --port 8000

# 2a. Speed benchmark (voice cloning, non-streaming)
python benchmarks/eval/benchmark_tts_speed.py \
    --model fishaudio/s2-pro --port 8000 \
    --testset seedtts_testset/en/meta.lst --max-samples 10

# 2b. Speed benchmark (streaming)
python benchmarks/eval/benchmark_tts_speed.py \
    --model fishaudio/s2-pro --port 8000 \
    --testset seedtts_testset/en/meta.lst --max-samples 10 --stream

# 2c. WER evaluation (voice cloning)
python benchmarks/eval/voice_clone_s2pro_wer.py \
    --meta seedtts_testset/en/meta.lst \
    --output-dir results/s2pro_en --lang en --max-samples 50
```

## Eval Scripts

| Script | Task | Model | API |
|--------|------|-------|-----|
| `eval/benchmark_tts_speed.py` | TTS speed | Any TTS model | `/v1/audio/speech` |
| `eval/voice_clone_s2pro_wer.py` | Voice clone WER | S2 Pro | `/v1/audio/speech` |
| `eval/voice_clone_qwen3_omni_wer.py` | Voice clone WER | Qwen3 Omni | `/v1/chat/completions` |

## Adding a New Model

For a model using the same API type (e.g., another OAI TTS API model):
1. Add eval script in `eval/voice_clone_your_model_wer.py` using existing task class

For a new API type: add a new class in the relevant `tasks/` file.
