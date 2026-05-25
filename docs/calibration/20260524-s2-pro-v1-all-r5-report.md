# CI Threshold Observation Report

## 1. TTS NONSTREAM Wer

— 2× NVIDIA H20, STREAMING_BENCHMARK_MAX_SAMPLES=32, 5 runs

| Run | Samples run | Samples ok | Corpus WER (%) | Max per-sample WER (%) |
|-----|--------|--------|--------|--------|
| 1 | 50 | 50 | 0.89 | 14.29 |
| 2 | 50 | 50 | 0.89 | 14.29 |
| 3 | 50 | 50 | 1.06 | 14.29 |
| 4 | 50 | 50 | 1.06 | 16.67 |
| 5 | 50 | 50 | 0.89 | 14.29 |
| **Worst-of-5** | — | — | **1.06** | **16.67** |

## 2. TTS NONSTREAM Speed

— 2× NVIDIA H20, STREAMING_BENCHMARK_MAX_SAMPLES=32, 5 runs

| Run | Samples run | Samples ok | Throughput (req/s) | Output tok/req-s | Latency mean (s) | RTF mean |
|-----|--------|--------|--------|--------|--------|--------|
| 1 | 50 | 50 | 1.465 | 66.9 | 9.757 | 3.0377 |
| 2 | 50 | 50 | 1.528 | 68.0 | 9.269 | 2.8495 |
| 3 | 50 | 50 | 1.503 | 67.4 | 9.405 | 2.8404 |
| 4 | 50 | 50 | 1.520 | 68.0 | 9.321 | 2.8593 |
| 5 | 50 | 50 | 1.517 | 69.0 | 9.327 | 2.8189 |
| **Worst-of-5** | — | — | **1.465** | **66.9** | **9.757** | **3.0377** |

## 3. TTS STREAM Wer

— 2× NVIDIA H20, STREAMING_BENCHMARK_MAX_SAMPLES=32, 5 runs

| Run | Samples run | Samples ok | Corpus WER (%) | Max per-sample WER (%) |
|-----|--------|--------|--------|--------|
| 1 | 32 | 32 | 1.06 | 14.29 |
| 2 | 32 | 32 | 1.06 | 14.29 |
| 3 | 32 | 32 | 1.33 | 16.67 |
| 4 | 32 | 32 | 1.06 | 14.29 |
| 5 | 32 | 32 | 1.33 | 16.67 |
| **Worst-of-5** | — | — | **1.33** | **16.67** |

## 4. TTS STREAM Speed

— 2× NVIDIA H20, STREAMING_BENCHMARK_MAX_SAMPLES=32, 5 runs

| Run | Samples run | Samples ok | Throughput (req/s) | Output tok/req-s | Latency mean (s) | RTF mean |
|-----|--------|--------|--------|--------|--------|--------|
| 1 | 32 | 32 | 1.309 | 60.6 | 10.229 | 2.8393 |
| 2 | 32 | 32 | 1.287 | 58.7 | 10.035 | 2.8508 |
| 3 | 32 | 32 | 1.371 | 61.0 | 9.881 | 2.6539 |
| 4 | 32 | 32 | 1.417 | 58.5 | 9.740 | 2.7394 |
| 5 | 32 | 32 | 1.415 | 60.0 | 9.826 | 2.6914 |
| **Worst-of-5** | — | — | **1.287** | **58.5** | **10.229** | **2.8508** |

## Applied changes

| Stage | Metric | Old | New | Direction |
|-------|--------|-----|-----|-----------|
| tts_nonstream_wer | VC_WER_MAX_CORPUS | 0.012411347517730497 | 0.010638297872340425 | tightens (-14.3%) |
| tts_nonstream_wer | VC_WER_MAX_PER_SAMPLE | 0.17 | 0.16666666666666666 | tightens (-2.0%) |
| tts_stream_wer | VC_STREAM_WER_MAX_CORPUS | 0.010610079575596816 | 0.013262599469496022 | loosens (+25.0%) |
| tts_stream_wer | VC_STREAM_WER_MAX_PER_SAMPLE | 0.14285714285714285 | 0.16666666666666666 | loosens (+16.7%) |
| tts_nonstream_speed | _VC_NON_STREAM_P95[16]['throughput_qps'] | 1.433 | 1.465 | tightens (+2.2%) |
| tts_nonstream_speed | _VC_NON_STREAM_P95[16]['latency_mean_s'] | 9.769 | 9.757 | tightens (-0.1%) |
| tts_stream_speed | _VC_STREAM_P95[16]['throughput_qps'] | 1.285 | 1.287 | tightens (+0.2%) |
| tts_stream_speed | _VC_STREAM_P95[16]['latency_mean_s'] | 10.289 | 10.229 | tightens (-0.6%) |
| tts_stream_speed | _VC_STREAM_P95[16]['rtf_mean'] | 2.8576 | 2.8508 | tightens (-0.2%) |

## Provenance

- Model: s2-pro-v1
- Branch: calibration-chenyang-05-24 @ e6e9e428 (dirty) — see `workspace.diff`
- Venv Python: /sgl-workspace/sglang-omni/omni-qwen3/bin/python (flag)
- sglang 0.5.8 · torch 2.9.1+cu128
- GPU: 2× NVIDIA H20
- tune-ci-thresholds v0.4.1
- Ran 2026-05-25T02:54:54Z – 2026-05-25T03:15:20Z
