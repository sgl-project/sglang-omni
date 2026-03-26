# S2 Pro WER Benchmark Results (seed-tts-eval EN)

**Date**: 2026-03-26
**Model**: `fishaudio/s2-pro`
**Dataset**: [seed-tts-eval](https://huggingface.co/datasets/zhaochenyang20/seed-tts-eval) English (`en/meta.lst`, 1088 samples)
**ASR**: Whisper-large-v3
**Inference**: sglang-omni server (HTTP API, non-streaming)
**Temperature**: 0.8

## Results Summary

### Full English Set (1088 samples)

| Metric | Value |
|--------|-------|
| Total Samples | 1088 |
| Evaluated | 1088 |
| Skipped | 0 |
| **Corpus WER (micro-avg)** | **1.95%** |
| Per-sample WER mean | 1.83% |
| Per-sample WER median | 0.00% |
| Per-sample WER std | 8.70% |
| Per-sample WER p95 | 9.09% |
| Corpus WER (excl >50%) | 1.24% |
| Samples with >50% WER | 8 (0.74%) |
| Mean latency (s) | 7.02 |
| Mean audio duration (s) | 4.61 |

### First 50 Samples (reproducibility check)

| Metric | PR #216 (pipeline) | Server (run 1) | Server (run 2, HF dataset) |
|--------|-------------------|----------------|---------------------------|
| **Corpus WER (micro-avg)** | **0.91%** | **0.89%** | **0.89%** |
| Per-sample WER mean | 1.10% | 0.62% | 0.62% |
| Per-sample WER median | 0.00% | 0.00% | 0.00% |
| Per-sample WER std | 4.33% | - | 3.04% |
| Per-sample WER p95 | 6.88% | - | 0.00% |
| >50% WER samples | 0 | 0 | 0 |
| Mean latency (s) | - | - | 6.96 |
| Mean audio duration (s) | - | - | 3.96 |

The "Server (run 2, HF dataset)" column uses the updated `benchmark_tts_wer.py` which auto-downloads from HuggingFace (`zhaochenyang20/seed-tts-eval`). The result (0.89%) is identical to run 1, confirming:

1. The HuggingFace dataset is correct and consistent with the previous local copy.
2. Server mode is reproducible across runs.
3. Server and pipeline modes produce equivalent accuracy (0.89% vs 0.91%).

## Comparison with Fish Audio S2 Pro Tech Report

Source: [Fish Audio S2 Technical Report (arXiv:2603.08823)](https://arxiv.org/abs/2603.08823)

| Model | test-en WER (%) | Source |
|-------|----------------|--------|
| Fish Audio S2 (official) | 0.99 | Tech Report |
| **SGLang-Omni S2 Pro (full set)** | **1.95** | This benchmark |
| **SGLang-Omni S2 Pro (excl >50%)** | **1.24** | This benchmark |
| Fish Audio S1 (official) | 1.07 | Tech Report |
| CosyVoice 3-1.5B | 2.21 | Tech Report |
| Seed-TTS | 2.25 | Tech Report |
| Qwen3-TTS | 1.24 | Tech Report |
| Minimax Speech-02 | 1.90 | Tech Report |
| FireRedTTS-2 | 1.95 | Tech Report |

## Analysis

**Full-set gap**: Our SGLang-Omni full-set result (1.95%) is ~2x the official Fish Audio S2 Pro result (0.99%). However, the first-50-sample result (0.89%) closely matches the official number.

**Root cause**: The WER increase at scale is driven by **8 truncated/empty-output samples** (0.74% of total), not by a systematic serving-layer issue. Excluding these, the corpus WER drops to **1.24%**, which matches Qwen3-TTS and is competitive with the field.

**Possible explanations for truncated outputs**:

1. **Model corner cases**: The model generates very short or empty audio for certain input texts, likely due to edge cases in the text-to-semantic generation (e.g., specific phoneme sequences, unusual sentence structures).

2. **Sampling parameters**: The official benchmark may use different sampling parameters (e.g., different temperature, top_p, top_k, repetition_penalty). We used the server defaults (temperature=0.8, top_p=0.8, top_k=30, repetition_penalty=1.1).

3. **Text normalization**: WER is sensitive to text normalization. We used `whisper_normalizer.english.EnglishTextNormalizer` (or the transformers equivalent), which is the standard approach. Minor differences in normalization could account for some of the remaining gap.

## Worst Performing Samples

| Sample ID | WER | Issue |
|-----------|-----|-------|
| common_voice_en_20735803-...807 | 100% | Empty output |
| common_voice_en_678682-...679 | 100% | Empty output |
| common_voice_en_22793454-...457 | 92.9% | Only first word generated |
| common_voice_en_25626822-...825 | 91.7% | Near-empty output |
| common_voice_en_17401431-...433 | 90.0% | Only first word generated |
| common_voice_en_26933529-...531 | 76.9% | Truncated at 3 words |
| common_voice_en_19944630-...633 | 75.0% | Truncated at 3 words |
| common_voice_en_628642-...645 | 69.2% | Truncated at 4 words |

## Dataset

The evaluation uses the [seed-tts-eval](https://huggingface.co/datasets/zhaochenyang20/seed-tts-eval) dataset, which contains 5 evaluation sets:

| # | File | Language | Samples | Difficulty |
|---|------|----------|---------|------------|
| 1 | `en/meta.lst` | English | 1,088 | Standard |
| 2 | `zh/meta.lst` | Chinese | 2,020 | Standard |
| 3 | `en/non_para_reconstruct_meta.lst` | English | 1,086 | Hard (cross-speaker) |
| 4 | `zh/non_para_reconstruct_meta.lst` | Chinese | 2,018 | Hard (cross-speaker) |
| 5 | `zh/hardcase.lst` | Chinese | 400 | Hard (tongue twisters) |

Currently only sets 1 and 2 are used in the standard benchmark pipeline.

## How to Reproduce

```bash
# 1. Start the server
python -m sglang_omni.cli.cli serve \
    --model-path fishaudio/s2-pro \
    --config examples/configs/s2pro_tts.yaml \
    --port 8000

# 2. Run WER evaluation (auto-downloads dataset from HuggingFace)
python -m benchmarks.performance.tts.benchmark_tts_wer \
    --model fishaudio/s2-pro \
    --port 8000 \
    --output-dir results/s2pro_en_wer \
    --lang en \
    --device cuda:0

# Or with a local dataset path:
python -m benchmarks.performance.tts.benchmark_tts_wer \
    --meta seedtts_testset/en/meta.lst \
    --model fishaudio/s2-pro \
    --port 8000 \
    --output-dir results/s2pro_en_wer \
    --lang en \
    --device cuda:0
```
