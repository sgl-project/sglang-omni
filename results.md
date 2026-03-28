# S2 Pro Benchmark Verification Results

**Date:** 2026-03-28
**Dataset:** seedtts_testset/en/meta.lst (first 50 samples)
**Server:** S2 Pro on single GPU (NVIDIA H100)
**Branch:** benchmark-redesign

## Speed Benchmark (Voice Cloning, Non-Streaming)

| Metric | Value |
|--------|-------|
| Completed requests | 50 |
| Failed requests | 0 |
| Latency mean (s) | 6.926 |
| Latency median (s) | 6.524 |
| Latency p95 (s) | 10.287 |
| Latency p99 (s) | 11.096 |
| Audio duration mean (s) | 3.961 |
| RTF mean | 1.9108 |
| RTF median | 1.7751 |
| Tok/s (per-req mean) | 83.9 |
| Tok/s (per-req median) | 85.1 |
| Tok/s (aggregate) | 84.2 |
| Gen tokens (mean) | 85.0 |
| Gen tokens (total) | 4265 |
| Prompt tokens (mean) | 177.0 |
| Prompt tokens (total) | 8863 |
| Throughput (req/s) | 0.144 |

All CI thresholds would pass:
- tok_per_s_agg = 84.2 >= 80 (VC_NON_STREAM_MIN_TOK_PER_S)
- rtf_mean = 1.91 <= 2.85 (VC_NON_STREAM_MAX_RTF)

## WER Benchmark (Voice Cloning, English)

| Metric | Value |
|--------|-------|
| Total samples | 50 |
| Evaluated | 50 |
| Skipped | 0 |
| **WER (corpus, micro-avg)** | **2.66%** |
| WER per-sample mean | 2.62% |
| WER per-sample median | 0.00% |
| WER per-sample std | 14.24% |
| WER per-sample p95 | 7.86% |
| WER corpus (excl >50%) | 0.90% |
| Bad cases (>50% WER) | 1 (2.0%) |
| Latency mean (s) | 6.641 |
| Audio duration mean (s) | 4.120 |

### Bad Case Analysis

1 sample had WER > 50%:

- **ID:** `common_voice_en_15265-common_voice_en_15268`
- **WER:** 100%
- **Target:** "A sky jumper falls toward the sea and the earth."
- **Whisper transcription:** "." (nearly empty — model generated very short/silent audio)

2 samples had nonzero WER <= 50%:

- `common_voice_en_123125-common_voice_en_123126`: WER=14.29%
- `common_voice_en_155313-common_voice_en_155315`: WER=16.67%

### Summary

47 out of 50 samples (94%) had **perfect WER = 0.00%**. The corpus-level micro-average WER of **2.66%** is dominated by one bad case. Excluding bad cases (>50% WER), the corpus WER drops to **0.90%**, indicating excellent voice cloning quality.

## Output Files

- Speed results: `/tmp/s2pro_speed_50/speed_results.json`
- WER results: `/tmp/s2pro_wer_50/wer_results.json`
- WER per-sample CSV: `/tmp/s2pro_wer_50/wer_results.csv`
- Generated audio: `/tmp/s2pro_wer_50/audio/`
