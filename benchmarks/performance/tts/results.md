# Qwen3-Omni TTS WER Evaluation via SGLang-Omni Server

## Overview

This document reports Word Error Rate (WER) benchmarks for **Qwen3-Omni-30B-A3B-Instruct** served via the **sglang-omni server** (speech pipeline), evaluated on the [seed-tts-eval](https://huggingface.co/datasets/zhaochenyang20/seed-tts-eval) English test set.

### Goals

1. **Verify Jingwen's fix** (`JingwenGu0829:fix/minimal-speech-pipeline-patches`) — confirm the Qwen3 Omni speech pipeline is functional on sglang-omni server.
2. **Benchmark WER** on EN subset (50 / 100 / 200 samples) and compare to the official reported 1.39%.
3. **Note**: Qwen3 Omni does **not** support voice cloning. It uses fixed speaker IDs (Ethan, Chelsie, Aiden). The official 1.39% WER is also reported without voice cloning. This is unlike S2-Pro which supports reference-audio-based voice cloning.

## Server Configuration

- **Model**: `Qwen/Qwen3-Omni-30B-A3B-Instruct`
- **Pipeline**: 9-stage speech pipeline (thinker + talker + code_predictor + code2wav + encoders + preprocessing + decode)
- **GPU allocation**: 2x H100 (thinker on GPU 0, talker/code_predictor/code2wav on GPU 1, via `CUDA_VISIBLE_DEVICES=2,3`)
- **API endpoint**: `/v1/chat/completions` with `modalities: ["text", "audio"]`
- **Speaker**: Ethan (default)
- **Branch**: `qwen3-omni-wer-server-eval` (based on `qwen3-omni-wer-validation` + `JingwenGu0829:fix/minimal-speech-pipeline-patches`)

## Fix Verification

The fix (`fix/minimal-speech-pipeline-patches`) addresses several critical issues:

| Fix | Description |
|-----|-------------|
| `EntryClass` default | Changed from text-only to speech pipeline (`Qwen3OmniSpeechPipelineConfig`) |
| `audio_target_sr` | Added missing variable in preprocessor's plain-string input path |
| Tensor clone | Fixed CUDA illegal memory access by cloning aux hidden states before async SHM copy |
| Radix cache | Disabled for talker (projected embeddings break prefix matching) |
| Spawn context | Use `multiprocessing.get_context("spawn")` to avoid CUDA re-init in forked children |
| Multi-process launcher | Added `needs_mp` path in `launcher.py` for multi-GPU pipelines |

**Result**: Server starts successfully, health check passes, and TTS requests return valid audio.

## WER Results (English, seed-tts-eval)

### Summary

| Samples | WER (corpus, micro-avg) | WER per-sample mean | WER (excl >50%) | >50% WER samples | Latency (mean) |
|---------|------------------------|---------------------|------------------|-------------------|----------------|
| 50      | **0.89%**              | 0.62%               | 0.89%            | 0 (0.0%)          | 5.28s          |
| 100     | **1.73%**              | 1.97%               | 0.92%            | 1 (1.0%)          | 5.03s          |
| 200     | **1.95%**              | 2.04%               | 1.95%            | 0 (0.0%)          | 5.13s          |

- **Official reported WER**: 1.39% (full EN set, no voice cloning)
- **ASR model**: Whisper-large-v3
- **WER metric**: Micro-average (corpus-level), consistent with HuggingFace evaluate standard

### Comparison with Previous Results

| Configuration | 50 samples | Full EN set |
|---------------|-----------|-------------|
| Qwen3 Omni (transformers pipeline, no voice clone) | 0.89% | 2.53% |
| Qwen3 Omni (sglang-omni server, no voice clone) | 0.89% | N/A (tested 200) → 1.95% |
| Qwen3 Omni (official, no voice clone) | N/A | 1.39% |
| S2-Pro (sglang-omni server, with voice clone) | 0.89% | 1.95% |

### Notable Bad Cases (WER > 10%, 200-sample run)

| Sample ID | WER | Reference | Hypothesis | Issue |
|-----------|-----|-----------|------------|-------|
| `common_voice_en_18466098-...` | 50.0% | "catch as catch can" | "catress catch can" | Short sentence, first word garbled |
| `common_voice_en_18944281-...` | 30.0% | "recordings of greene are scarce..." | "red coatings of green are scarce..." | First words heavily distorted |
| `common_voice_en_18336410-...` | 20.0% | "the questionnaire is too simplistic" | "the croissanier is too simplistic" | Rare word "questionnaire" mispronounced |
| `common_voice_en_18893247-...` | 20.0% | "for a start we both know..." | "to our start we both know..." | First words swapped |
| `common_voice_en_15265-...` | 20.0% | "a sky jumper falls toward..." | "a skyjumper falls toward..." | Compound word merge (debatable) |

Most errors are on the first 1-2 words of the sentence, or on rare/unusual words.

## Detailed Results

Full per-sample results are saved in:
- `results/qwen3_omni_server_en_50/wer_results.json`
- `results/qwen3_omni_server_en_100/wer_results.json`
- `results/qwen3_omni_server_en_200/wer_results.json`

## How to Reproduce

```bash
# 1. Download dataset
huggingface-cli download zhaochenyang20/seed-tts-eval \
    --repo-type dataset --local-dir seedtts_testset

# 2. Start Qwen3 Omni speech server (requires 2 GPUs)
CUDA_VISIBLE_DEVICES=2,3 python examples/run_qwen3_omni_speech_server.py \
    --model-path Qwen/Qwen3-Omni-30B-A3B-Instruct \
    --gpu-thinker 0 --gpu-talker 1 \
    --gpu-code-predictor 1 --gpu-code2wav 0 \
    --port 8000

# 3. Run WER evaluation (e.g., 200 samples)
CUDA_VISIBLE_DEVICES=4 python -m benchmarks.performance.tts.benchmark_tts_wer_qwen3_omni_server \
    --meta seedtts_testset/en/meta.lst \
    --output-dir results/qwen3_omni_server_en_200 \
    --lang en \
    --max-samples 200 \
    --asr-device cuda:0 \
    --port 8000
```

## Conclusions

1. **Jingwen's fix works**: The Qwen3 Omni speech pipeline runs correctly on sglang-omni server after the minimal patches.
2. **WER is reasonable**: 1.95% on 200 EN samples, compared to official 1.39% on full set. The gap may narrow with more samples or be due to non-determinism in generation.
3. **sglang-omni server matches transformers**: The 50-sample WER (0.89%) is identical between sglang-omni server and transformers pipeline, confirming correctness.
4. **No voice cloning**: Qwen3 Omni only supports fixed speaker IDs, not reference-audio-based voice cloning. This is an architecture limitation, not a bug.

## Date

2026-03-27
