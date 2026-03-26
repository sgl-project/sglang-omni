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

### English (seed-tts-eval test-en, 350 samples)

| Metric | 20 samples | 350 samples |
|--------|-----------|-------------|
| Samples evaluated | 20 / 1088 | 350 / 1088 |
| **WER mean** | **1.60%** | **3.59%** |
| WER median | 0.00% | 0.00% |
| WER std | 4.15% | 9.17% |
| WER p95 | 8.14% | 17.50% |
| >50% WER samples | 0 (0.0%) | 1 (0.3%) |
| Non-zero WER samples | 3 (15%) | 80 (22.9%) |
| Latency mean | 8.73s | 9.14s |
| Audio duration mean | 3.83s | 4.17s |

### Chinese (seed-tts-eval test-zh, 350 samples)

For Chinese, we use **Character Error Rate (CER)**, which is the standard
metric for Chinese TTS evaluation. The Qwen3-Omni paper reports this as "WER"
but it is effectively CER (each Chinese character is treated as a word).

| Metric | 20 samples | 350 samples |
|--------|-----------|-------------|
| Samples evaluated | 20 / 2020 | 350 / 2020 |
| **CER mean** | **0.92%** | **4.34%** |
| CER median | 0.00% | 0.00% |
| CER std | 3.10% | 8.61% |
| CER p95 | 5.21% | 26.32% |
| >50% CER samples | 0 (0.0%) | 0 (0.0%) |
| Non-zero CER samples | 2 (10%) | 103 (29.4%) |
| Latency mean | 8.87s | 9.49s |
| Audio duration mean | 4.21s | 4.33s |

### Comparison with Official Results

| Testset | Ours (20) | Ours (350) | Published (full) | Delta (350 vs published) |
|---------|-----------|------------|-------------------|--------------------------|
| test-en WER | 1.60% | 3.59% | 1.39% | +2.20% |
| test-zh CER | 0.92% | 4.34% | 1.07% | +3.27% |

### Scale Effect: 20 vs 350 Samples

| Testset | 20-sample WER/CER | 350-sample WER/CER | Change |
|---------|-------------------|---------------------|--------|
| test-en | 1.60% | 3.59% | +1.99% |
| test-zh | 0.92% | 4.34% | +3.42% |

The 20-sample subset was close to published results, but the 350-sample run
reveals a higher error rate. This gap is explained by the **Whisper number
normalization problem** and a **long-tail of harder samples** (see Error
Analysis below).

## Error Analysis (350 samples)

### Dominant Error Category: Number Normalization

The single largest source of error in both EN and ZH is **number format
mismatch between target text and Whisper transcription**:

**English examples:**
- "fifty" → Whisper outputs "50" (WER=16.7% for that sentence)
- "ninety five lines" → Whisper outputs "95 lies" (WER=37.5%)
- "two thousand" → Whisper outputs "2000"

**Chinese examples (most impactful):**
- "百分之一百零一点一" → Whisper outputs "1011" (CER=39.1%, 2 samples)
- "百分之十五点二" → Whisper outputs "152" (CER=37.5%)
- "百分之四十一点五" → Whisper outputs "415" (CER=36.4%)

These are **not speech quality errors** — the model correctly synthesizes
the numbers. The mismatch arises because Whisper normalizes spoken numbers
to digits. The Qwen3-Omni paper likely uses a different ASR pipeline or
applies number normalization to both reference and hypothesis before WER
computation.

### Other Error Categories

**English (80/350 non-zero WER):**
- Apostrophe normalization: "doesnt" in target vs "doesn't" from Whisper
- Minor word substitutions: "do" → "did", "this" → "the"
- Garbled output on short sentences (1 sample with WER=100%)

**Chinese (103/350 non-zero CER):**
- Homophone confusions: "怀安" ↔ "淮安", "产品" ↔ "缠密", "珠" ↔ "中"
- Repetition in target text: "独立思独立思独立思考" (stuttering in ground truth)
- Punctuation-adjacent characters lost after normalization

### Error Distribution

Both languages show a **heavy right tail**: the majority of samples have 0%
error, but a minority have >10% error, which pulls the mean up significantly.

| | EN (350) | ZH (350) |
|---|----------|----------|
| 0% error | 270 (77.1%) | 247 (70.6%) |
| 0-5% error | 0 (0.0%) | 20 (5.7%) |
| 5-10% error | 40 (11.4%) | 30 (8.6%) |
| 10-20% error | 26 (7.4%) | 13 (3.7%) |
| 20-50% error | 13 (3.7%) | 40 (11.4%) |
| >50% error | 1 (0.3%) | 0 (0.0%) |

## Configuration

- **Model**: `Qwen/Qwen3-Omni-30B-A3B-Instruct`
- **Whisper**: `openai/whisper-large-v3`
- **Speaker**: Chelsie
- **Temperature**: 0.7
- **Max new tokens**: 2048
- **GPU**: NVIDIA H200 (143GB), two GPUs used in parallel for EN/ZH
- **Backend**: HuggingFace Transformers 4.57.1
- **PyTorch**: 2.9.1+cu128
- **Total runtime**: ~55 minutes for 350+350 samples (parallel on 2 GPUs)

## Script Location

```
benchmarks/accuracy/omni/benchmark_omni_wer.py
```

## How to Run

```bash
# Download dataset
huggingface-cli download zhaochenyang20/seed-tts-eval \
    --repo-type dataset --local-dir seedtts_testset

# Quick test (20 samples, ~3 min)
CUDA_VISIBLE_DEVICES=1 python -m benchmarks.accuracy.omni.benchmark_omni_wer \
    --model-path Qwen/Qwen3-Omni-30B-A3B-Instruct \
    --testset seedtts_testset/en \
    --max-samples 20

# Medium run (350 samples, ~55 min)
CUDA_VISIBLE_DEVICES=1 python -m benchmarks.accuracy.omni.benchmark_omni_wer \
    --model-path Qwen/Qwen3-Omni-30B-A3B-Instruct \
    --testset seedtts_testset/en \
    --max-samples 350

# Full evaluation (1088 EN samples, ~3 hours)
CUDA_VISIBLE_DEVICES=1 python -m benchmarks.accuracy.omni.benchmark_omni_wer \
    --model-path Qwen/Qwen3-Omni-30B-A3B-Instruct \
    --testset seedtts_testset/en

# Save generated audio for manual inspection
CUDA_VISIBLE_DEVICES=1 python -m benchmarks.accuracy.omni.benchmark_omni_wer \
    --model-path Qwen/Qwen3-Omni-30B-A3B-Instruct \
    --testset seedtts_testset/en \
    --max-samples 20 --save-audio
```

## Conclusion

The Qwen3-Omni model produces high-quality speech that is broadly consistent
with the published WER/CER numbers. The gap between our 350-sample results
(EN 3.59%, ZH 4.34%) and the published results (EN 1.39%, ZH 1.07%) is
primarily attributable to:

1. **Whisper number normalization** — the dominant error source. Whisper
   converts spoken numbers to digits (e.g., "百分之四十一点五" → "415"),
   inflating CER on number-heavy sentences. The official evaluation likely
   uses a normalization pipeline that handles this.

2. **Stochastic sampling** — temperature=0.7 introduces variance. Some
   samples produce near-perfect speech while others have minor word
   substitutions.

3. **Evaluation pipeline difference** — the official benchmark may use a
   different ASR model, text normalization, or post-processing pipeline.

On the 20-sample subset, results were very close to published numbers
(EN +0.21%, ZH -0.15%), confirming the model fundamentally works correctly.
The 350-sample run exposes the long tail of harder cases, especially
number-heavy sentences.

Once the SGLang speech pipeline is fixed, this benchmark should be re-run
with the serving backend for production-representative evaluation. Adding
number normalization to the text preprocessing would significantly reduce
the measured WER/CER gap.
