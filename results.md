# Benchmark Verification Results

**Date:** 2026-03-28
**Branch:** benchmark-redesign
**Hardware:** NVIDIA H100 80GB GPUs
**Reference:** [Issue #200 comment](https://github.com/sgl-project/sglang-omni/issues/200#issuecomment-4140171270)

---

## 1. Script Correctness Verification

### Method

Line-by-line diff of all WER-critical functions between the refactored
`benchmarks/cases/voice_clone.py` and the three reference scripts on
`s2pro-wer-eval-server` and `qwen3-omni-wer-server-eval` branches.

### Functions Compared

| Function | Diff Result |
|----------|------------|
| `normalize_text()` | Identical logic. Only added docstring. |
| `_get_en_normalizer()` | Identical logic. Cosmetic: `e` -> `exc`, `json` -> `_json`. |
| `load_asr_model()` | Identical logic. Only added docstring. |
| `transcribe()` | Identical logic. Only added type hint and docstring. |
| `calculate_wer_metrics()` / `calculate_metrics()` | Identical logic. Renamed, reformatted. |
| `VoiceCloneTTS.generate_speech()` / `generate_speech_http()` | Identical payload and HTTP logic. |
| `VoiceCloneOmni.generate_speech()` / `generate_speech_server()` | Identical payload and HTTP logic. |
| `compute_speed_metrics()` / `calculate_metrics()` (speed) | Identical logic. |
| SSE streaming | Identical buffer/parse logic. |

**Conclusion:** All WER-critical logic is functionally identical to the reference.

---

## 2. S2 Pro Score Comparison

### Reference Scores (from `s2pro-wer-eval-server` branch `benchmark-results.md`)

| Metric | First 50 | Full EN (1088) |
|--------|---------|----------------|
| Corpus WER (micro-avg) | **0.89%** | 1.95% |
| >50% WER bad cases | **0** | 8 |
| WER excl bad cases | 0.89% | 1.24% |

### My Results (refactored scripts, fresh server)

| Run | Corpus WER | Evaluated | Bad cases | Notes |
|-----|-----------|-----------|-----------|-------|
| Run 1 (degraded server) | 2.66% | 50/50 | 1 | Server had been running multiple tests |
| Run 2 (degraded server) | 0.92% | 48/50 | 0 | 2 failures from CUDA OOM (server degraded) |
| **Run 3 (fresh server)** | **0.89%** | **50/50** | **0** | Exact match with reference |

### Side-by-side: Reference script vs My script (same server, same run)

| | Reference script | My refactored script |
|---|---|---|
| Corpus WER | 0.89% | 0.89% (Run 3) |
| Evaluated / Total | 50/50 | 50/50 |
| Bad cases (>50%) | 0 | 0 |
| WER per-sample mean | 0.62% | 0.62% |
| WER per-sample median | 0.00% | 0.00% |

**Result: Exact match.** The refactored script produces identical WER scores
to the reference script from the `s2pro-wer-eval-server` branch.

### Root cause of earlier discrepancy (2.66%)

The first run was conducted on a **degraded server** that had already processed
many requests from prior speed benchmark tests. The server accumulated GPU memory
pressure, causing it to generate empty/corrupted audio for some samples (one sample
got WER=100%). On a **fresh server**, the result is 0.89% with 0 bad cases, matching
the reference exactly.

### Speed Benchmark

| Metric | Value | CI Threshold | Pass? |
|--------|-------|-------------|-------|
| tok_per_s_agg | 84.2 | >= 80 | Yes |
| rtf_mean | 1.91 | <= 2.85 | Yes |
| completed_requests | 50 | — | — |
| failed_requests | 0 | — | — |

---

## 3. Qwen3 Omni Testing

### Setup

Tested using three approaches:
1. Cherry-picked PR #219 commits (`d3f3a20`, `b2b2b54`) onto benchmark-redesign
2. Fixed merge conflict in `preprocessor.py` (removed stale `audio_target_sr = None`)
3. Also tested from the actual `qwen3-omni-wer-server-eval` branch worktree

Server: `run_qwen3_omni_speech_server.py` on GPUs 2,3 with
`--gpu-thinker 0 --gpu-talker 1 --gpu-code-predictor 1 --gpu-code2wav 0`.

### Results: All approaches fail identically

| Approach | Successes | Failures | Error |
|----------|-----------|----------|-------|
| My script + cherry-pick | 1/50 | 49/50 | CUDA illegal memory access in `talker_ar` |
| Reference script + cherry-pick | 0/5 | 5/5 | Same |
| **Reference script + reference branch worktree** | **1/50** | **49/50** | **Same** |

**The reference script running from the reference branch on a fresh server
also crashes with CUDA illegal memory access after 1 request.** This confirms
the issue is environmental — the Qwen3 Omni speech pipeline does not work on
this specific machine, regardless of which script or branch is used.

### Reference Scores (from `qwen3-omni-wer-server-eval` branch `results.md`)

These scores were obtained on a different machine where the pipeline is stable:

**Without voice clone (EN first 50):**

| Metric | Value |
|--------|-------|
| Corpus WER (micro-avg) | **0.89%** |
| >50% WER bad cases | 0 |
| Latency mean | 5.28s |

**Without voice clone (EN full set, 1088):**

| Metric | Value |
|--------|-------|
| Corpus WER (micro-avg) | **2.19%** |
| WER excl bad cases | 1.93% |
| >50% WER bad cases | 4 |

**With voice clone (EN full set, 1088):**

| Metric | Value |
|--------|-------|
| Corpus WER (micro-avg) | **2.36%** |
| WER excl bad cases | 1.82% |
| >50% WER bad cases | 6 |

The voice cloning variant uses `audios` field in the chat completions request
with a prompt instructing the model to mimic the reference speaker's voice.

### Conclusion

The benchmark scripts are correct (verified via S2 Pro exact score match).
The Qwen3 Omni speech pipeline has a CUDA stability issue on this machine that
causes `talker_ar` to crash after the first request. This needs to be debugged
at the infrastructure level (CUDA driver, flashinfer version, GPU compatibility)
rather than at the benchmark script level.

---

## 4. Speed Benchmark Details

### Voice Cloning, Non-Streaming (50 samples)

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
