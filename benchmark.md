# Qwen3-Omni Accuracy Benchmark (WER)

## Overview

This benchmark measures the **Word Error Rate (WER)** of Qwen3-Omni's speech
synthesis on the **seed-tts-eval** dataset, replicating the evaluation from
Section 5.2 of the Qwen3-Omni paper (Table 13).

The pipeline:
1. Qwen3-Omni generates speech from a text prompt
2. Whisper-large-v3 transcribes the generated audio
3. WER is computed between the target text and the Whisper transcription

## Inference Backend

**HuggingFace Transformers (offline)** -- not the SGLang serving pipeline.

The SGLang speech pipeline on main is currently broken due to multiple issues:
- Text-only server crashes with `audio_target_sr` unbound variable
- Speech pipeline's `Qwen3OmniSpeechPipelineConfig` is not registered as `EntryClass`
- Multi-process pipeline runner has CUDA illegal memory access errors for cross-GPU streaming
- Single-process runner fails with distributed state conflict (two SGLang model runners)

Using HF Transformers ensures correctness of the accuracy measurement,
independent of the serving infrastructure bugs. The model is loaded via
`Qwen3OmniMoeForConditionalGeneration.from_pretrained()` with `enable_audio_output=True`.

## Results

### English (seed-tts-eval test-en)

| Metric | Value |
|--------|-------|
| Samples evaluated | 20 / 1088 |
| **WER mean** | **1.60%** |
| WER median | 0.00% |
| WER std | 4.15% |
| WER p95 | 8.14% |
| >50% WER samples | 0 (0.0%) |
| Latency mean | 8.73s |
| Audio duration mean | 3.83s |

### Chinese (seed-tts-eval test-zh)

For Chinese, we use **Character Error Rate (CER)**, which is the standard
metric for Chinese TTS evaluation. The Qwen3-Omni paper reports this as "WER"
but it is effectively CER (each Chinese character is treated as a word).

| Metric | Value |
|--------|-------|
| Samples evaluated | 20 / 2020 |
| **CER mean** | **0.92%** |
| CER median | 0.00% |
| CER std | 3.10% |
| CER p95 | 5.21% |
| >50% CER samples | 0 (0.0%) |
| Latency mean | 8.87s |
| Audio duration mean | 4.21s |

### Comparison with Official Results

| Testset | Ours (20 samples) | Published (full set) | Delta |
|---------|--------------------|----------------------|-------|
| test-en WER | 1.60% | 1.39% | +0.21% |
| test-zh CER | 0.92% | 1.07% | -0.15% |

Our results are very close to the published numbers, with both deltas within
0.25 percentage points. The small differences are expected due to:
- Small sample size (20 vs 1088/2020)
- Stochastic generation (temperature=0.7)
- Whisper transcription variance

## Error Analysis

### English errors (3/20 samples with non-zero WER)

1. **"fifty" vs "50"**: Whisper transcribed the spoken "fifty" as the digit
   "50". This is a text normalization issue, not a speech quality issue.
   (WER=16.7% for this sample)

2. **"doesn't" vs "doesnt"**: The target text had no apostrophe; Whisper
   output included it. After normalization, "doesn't" vs "doesnt" mismatch.
   (WER=7.7%)

3. **"do" vs "did"**: Minor word substitution: "I do not like sushi" was
   synthesized as "I did not like sushi". (WER=7.7%)

### Chinese errors (2/20 samples with non-zero CER)

1. **"可是" vs "持续"** + **"窝策" vs "无策"**: Two character substitutions in a
   22-character sentence. The first is a homophone confusion (by the model or
   Whisper); the second involves a rare character "窝" in the ground truth
   where standard Chinese would use "无" (束手无策 is the correct idiom).
   (CER=13.6%)

2. **"怀安" vs "淮安"**: Homophone substitution for a place name. Both are
   valid Chinese place names with identical pronunciation. (CER=4.8%)

## Configuration

- **Model**: `Qwen/Qwen3-Omni-30B-A3B-Instruct`
- **Whisper**: `openai/whisper-large-v3`
- **Speaker**: Chelsie
- **Temperature**: 0.7
- **Max new tokens**: 2048
- **GPU**: NVIDIA H200 (143GB)
- **Backend**: HuggingFace Transformers 4.57.1
- **PyTorch**: 2.9.1+cu128

## Script Location

```
benchmarks/accuracy/omni/benchmark_omni_wer.py
```

## How to Run

```bash
# Download dataset
huggingface-cli download zhaochenyang20/seed-tts-eval \
    --repo-type dataset --local-dir seedtts_testset

# Run EN benchmark (20 samples)
CUDA_VISIBLE_DEVICES=1 python -m benchmarks.accuracy.omni.benchmark_omni_wer \
    --model-path Qwen/Qwen3-Omni-30B-A3B-Instruct \
    --testset seedtts_testset/en \
    --max-samples 20

# Run ZH benchmark (20 samples)
CUDA_VISIBLE_DEVICES=1 python -m benchmarks.accuracy.omni.benchmark_omni_wer \
    --model-path Qwen/Qwen3-Omni-30B-A3B-Instruct \
    --testset seedtts_testset/zh \
    --max-samples 20

# Full evaluation (takes hours with HF Transformers backend)
CUDA_VISIBLE_DEVICES=1 python -m benchmarks.accuracy.omni.benchmark_omni_wer \
    --model-path Qwen/Qwen3-Omni-30B-A3B-Instruct \
    --testset seedtts_testset/en
```

## Conclusion

The Qwen3-Omni model's speech synthesis accuracy, measured via Whisper-based
WER/CER on the seed-tts-eval dataset, is consistent with the published results.
On a 20-sample subset:
- **EN WER: 1.60%** (published 1.39%, delta +0.21%)
- **ZH CER: 0.92%** (published 1.07%, delta -0.15%)

These results validate the model's accuracy claims. The HF Transformers
backend is extremely slow (~8-9s per sample) but produces correct results.
Once the SGLang speech pipeline is fixed, this benchmark should be re-run
with the serving backend for production-representative evaluation.
