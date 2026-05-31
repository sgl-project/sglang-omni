# CI Threshold Observation Report

## 1. WHISPER ASR Wer

— 2× NVIDIA H20 from precheck.json, 20 samples (SEEDTTS_ASR_CORRECTNESS_SAMPLES), concurrency=2, 5 runs.

| Run | Samples run | Samples ok | Corpus WER (%) | Max per-sample WER (%) |
|-----|--------|--------|--------|--------|
| 1 | 20 | 20 | 0.69 | 6.67 |
| 2 | 20 | 20 | 0.69 | 6.67 |
| 3 | 20 | 20 | 0.69 | 6.67 |
| 4 | 20 | 20 | 0.69 | 6.67 |
| 5 | 20 | 20 | 0.69 | 6.67 |
| **Worst-of-5** | — | — | **0.69** | **6.67** |

## 2. WHISPER ASR Speed

— 2× NVIDIA H20 from precheck.json, 20 samples, concurrency=2, 5 runs.

| Run | Samples run | Samples ok | Throughput (req/s) | Latency mean (s) | Latency p95 (s) | RTF mean | RTF p95 |
|-----|--------|--------|--------|--------|--------|--------|--------|
| 1 | 20 | 20 | 10.320 | 0.193 | 0.553 | 0.0408 | 0.1417 |
| 2 | 20 | 20 | 10.405 | 0.191 | 0.550 | 0.0405 | 0.1408 |
| 3 | 20 | 20 | 10.295 | 0.193 | 0.555 | 0.0409 | 0.1421 |
| 4 | 20 | 20 | 10.476 | 0.190 | 0.548 | 0.0401 | 0.1404 |
| 5 | 20 | 20 | 10.430 | 0.191 | 0.554 | 0.0405 | 0.1419 |
| **Worst-of-5** | — | — | **10.295** | **0.193** | **0.555** | **0.0409** | **0.1421** |

## Applied changes

| Stage | Metric | Old | New | Direction |
|-------|--------|-----|-----|-----------|
| whisper_asr_speed | WHISPER_ASR_THROUGHPUT_MIN | 10.153 | 10.294983887949183 | tightens (+1.4%) |
| whisper_asr_speed | WHISPER_ASR_LATENCY_MEAN_MAX_S | 0.196 | 0.19333569328882733 | tightens (-1.4%) |
| whisper_asr_speed | WHISPER_ASR_LATENCY_P95_MAX_S | 0.57 | 0.555 | tightens (-2.6%) |
| whisper_asr_speed | WHISPER_ASR_RTF_MEAN_MAX | 0.0415 | 0.0409 | tightens (-1.4%) |
| whisper_asr_speed | WHISPER_ASR_RTF_P95_MAX | 0.1459 | 0.1421 | tightens (-2.6%) |

## Provenance

- Model: tts
- Branch: hayden/fix-issue-627 @ b9d687ef (dirty) — see `workspace.diff`
- Venv Python: /sgl-workspace/sglang-omni/omni/bin/python (flag)
- sglang 0.5.8 · torch 2.9.1+cu128
- GPU: 2× NVIDIA H20
- tune-ci-thresholds v0.4.3
- Ran 2026-05-31T08:25:24Z – 2026-05-31T08:31:46Z
