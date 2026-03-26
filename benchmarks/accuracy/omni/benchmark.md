# Qwen3-Omni Accuracy Benchmark (WER)

Benchmark Word Error Rate (WER) of Qwen3-Omni speech synthesis on the
[seed-tts-eval](https://huggingface.co/datasets/zhaochenyang20/seed-tts-eval)
dataset, comparing generated audio transcriptions (via Whisper) against
ground-truth target texts.

## Setup

| Item | Value |
|------|-------|
| Model | `Qwen/Qwen3-Omni-30B-A3B-Instruct` |
| Backend | HuggingFace Transformers (offline, single-sample inference) |
| ASR Model | `openai/whisper-large-v3` |
| Dataset | seed-tts-eval (BytedanceSpeech) |
| Speaker | Chelsie |
| Temperature | 0.7 |
| Max New Tokens | 2048 |
| Hardware | NVIDIA H200 (143 GB) |
| Date | 2026-03-26 |

## Full Dataset Results

Results on the **complete** seed-tts-eval dataset (1088 EN + 2020 ZH samples).

| Metric | EN (1088 samples) | ZH (2020 samples) |
|--------|-------------------:|-------------------:|
| **WER mean** | **3.63%** | **7.47%** |
| WER median | 0.00% | 0.00% |
| WER std | 12.67% | 36.36% |
| WER p95 | 18.18% | 22.73% |
| Samples > 50% WER | 6 (0.6%) | 21 (1.0%) |
| Published WER (paper Table 13) | 1.39% | 1.07% |
| **Delta (ours - published)** | **+2.24%** | **+6.40%** |
| Latency mean (s) | 9.491 | 10.708 |
| Latency median (s) | 9.290 | 10.431 |
| Latency p95 (s) | 13.606 | 13.932 |
| Audio duration mean (s) | 4.288 | 4.985 |
| Evaluated / Total | 1088 / 1088 | 2020 / 2020 |
| Skipped | 0 | 0 |

## Subset Size Analysis

WER varies significantly across subset sizes, confirming feedback from the
Qwen team that **the full dataset should be used for reliable evaluation**.

| Subset | EN WER | ZH WER |
|--------|-------:|-------:|
| 5 samples | 0.00% | -- |
| 20 samples | 1.60% | 0.92% |
| 350 samples | 3.59% | 4.34% |
| **Full dataset** | **3.63%** | **7.47%** |

## Notes

### Delta from Published Results

Our WER is higher than the published results in the Qwen3-Omni paper
(Table 13: EN 1.39%, ZH 1.07%). Possible contributing factors:

1. **Batch inference precision loss**: The Qwen team has confirmed that the
   Transformers batch inference path introduces slight precision loss in the
   audio encoder batch. Our benchmark uses single-sample inference, but the
   model checkpoint and Transformers integration may still carry related
   numerical differences.

2. **Inference configuration**: The published results may use different
   decoding parameters (temperature, speaker, max tokens) or a different
   inference backend.

3. **Whisper model version**: Differences in Whisper model version or
   decoding settings can affect transcription accuracy.

4. **ZH evaluation metric**: For Chinese, we compute character-level error
   rate (CER) by splitting text into individual characters before comparison.
   This is standard for Chinese TTS evaluation and matches the Qwen paper's
   approach.

### Subset Variance

The WER score is sensitive to which subset of the data is evaluated. Small
subsets (5-20 samples) can give misleadingly low WER, while the full dataset
reveals a higher error rate due to the long tail of difficult samples. The ZH
full-set WER (7.47%) is notably higher than the 350-sample subset (4.34%),
indicating that harder samples are concentrated in the latter portion of the
dataset.

## Reproduction

```bash
# Download full dataset
huggingface-cli download zhaochenyang20/seed-tts-eval \
    --repo-type dataset --local-dir seedtts_testset

# Run EN evaluation
CUDA_VISIBLE_DEVICES=0 python -m benchmarks.accuracy.omni.benchmark_omni_wer \
    --model-path Qwen/Qwen3-Omni-30B-A3B-Instruct \
    --testset seedtts_testset/en \
    --output-dir benchmarks/accuracy/omni/results/en_full

# Run ZH evaluation
CUDA_VISIBLE_DEVICES=0 python -m benchmarks.accuracy.omni.benchmark_omni_wer \
    --model-path Qwen/Qwen3-Omni-30B-A3B-Instruct \
    --testset seedtts_testset/zh \
    --output-dir benchmarks/accuracy/omni/results/zh_full
```
