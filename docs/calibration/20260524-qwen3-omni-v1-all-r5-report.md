# CI Threshold Observation Report

## 1. MMMU Accuracy

— 2× NVIDIA H20, CONCURRENCY=16, 5 runs

| Run | Samples run | Samples ok | Acc (%) |
|-----|--------|--------|--------|
| 1 | 50 | 50 | 62.00 |
| 2 | 50 | 50 | 58.00 |
| 3 | 50 | 50 | 64.00 |
| 4 | 50 | 50 | 56.00 |
| 5 | 50 | 50 | 60.00 |
| **Worst-of-5** | — | — | **56.00** |

## 2. MMMU Speed

— 2× NVIDIA H20, CONCURRENCY=16, 5 runs

| Run | Samples run | Samples ok | Throughput (req/s) | Output tok/req-s | Latency mean (s) |
|-----|--------|--------|--------|--------|--------|
| 1 | 50 | 50 | 1.318 | 58.7 | 10.131 |
| 2 | 50 | 50 | 1.364 | 57.0 | 10.320 |
| 3 | 50 | 50 | 1.349 | 54.0 | 9.956 |
| 4 | 50 | 50 | 1.351 | 58.4 | 10.266 |
| 5 | 50 | 50 | 1.346 | 58.8 | 10.085 |
| **Worst-of-5** | — | — | **1.318** | **54.0** | **10.320** |

## 3. MMMU TALKER Accuracy

— 2× NVIDIA H20, MAX_SAMPLES=20, MAX_TOKENS=256, CONCURRENCY=16, 5 runs

| Run | Samples run | Samples ok | Acc (%) |
|-----|--------|--------|--------|
| 1 | 20 | 20 | 75.00 |
| 2 | 20 | 20 | 75.00 |
| 3 | 20 | 20 | 75.00 |
| 4 | 20 | 20 | 75.00 |
| 5 | 20 | 20 | 75.00 |
| **Worst-of-5** | — | — | **75.00** |

## 4. MMMU TALKER Wer

— 2× NVIDIA H20, MAX_SAMPLES=20, MAX_TOKENS=256, CONCURRENCY=16, 5 runs

| Run | Samples run | Samples ok | Corpus WER ≤50% (%) | Samples >50% WER |
|-----|--------|--------|--------|--------|
| 1 | 20 | 20 | 15.01 | 2 |
| 2 | 20 | 20 | 15.27 | 3 |
| 3 | 20 | 20 | 18.15 | 2 |
| 4 | 20 | 20 | 11.52 | 4 |
| 5 | 20 | 20 | 18.15 | 3 |
| **Worst-of-5** | — | — | **18.15** | **4** |

## 5. MMMU TALKER Speed

— 2× NVIDIA H20, MAX_SAMPLES=20, MAX_TOKENS=256, CONCURRENCY=16, 5 runs

| Run | Samples run | Samples ok | Throughput (req/s) | Output tok/req-s | Latency mean (s) | RTF mean |
|-----|--------|--------|--------|--------|--------|--------|
| 1 | 20 | 20 | 0.717 | 8.1 | 16.126 | 0.4309 |
| 2 | 20 | 20 | 0.713 | 8.2 | 16.666 | 0.4252 |
| 3 | 20 | 20 | 0.689 | 8.2 | 16.090 | 0.4178 |
| 4 | 20 | 20 | 0.617 | 8.0 | 16.187 | 0.4223 |
| 5 | 20 | 20 | 0.687 | 8.2 | 15.995 | 0.4157 |
| **Worst-of-5** | — | — | **0.617** | **8.0** | **16.666** | **0.4309** |

## 6. MMSU Accuracy

— 2× NVIDIA H20, CONCURRENCY=16, 5 runs

| Run | Samples run | Samples ok | Acc (%) |
|-----|--------|--------|--------|
| 1 | 2000 | 2000 | 70.05 |
| 2 | 2000 | 2000 | 70.25 |
| 3 | 2000 | 2000 | 70.10 |
| 4 | 2000 | 2000 | 69.45 |
| 5 | 2000 | 2000 | 70.80 |
| **Worst-of-5** | — | — | **69.45** |

## 7. MMSU Speed

— 2× NVIDIA H20, CONCURRENCY=16, 5 runs

| Run | Samples run | Samples ok | Throughput (req/s) | Output tok/req-s | Latency mean (s) |
|-----|--------|--------|--------|--------|--------|
| 1 | 2000 | 2000 | 54.416 | 7.1 | 0.293 |
| 2 | 2000 | 2000 | 58.788 | 7.6 | 0.272 |
| 3 | 2000 | 2000 | 58.615 | 7.6 | 0.272 |
| 4 | 2000 | 2000 | 57.770 | 7.5 | 0.276 |
| 5 | 2000 | 2000 | 58.594 | 7.6 | 0.272 |
| **Worst-of-5** | — | — | **54.416** | **7.1** | **0.293** |

## 8. MMSU TALKER Accuracy

— 2× NVIDIA H20, MAX_SAMPLES=40, MAX_TOKENS=256, CONCURRENCY=16, 5 runs

| Run | Samples run | Samples ok | Acc (%) |
|-----|--------|--------|--------|
| 1 | 40 | 40 | 65.00 |
| 2 | 40 | 40 | 62.50 |
| 3 | 40 | 40 | 65.00 |
| 4 | 40 | 40 | 62.50 |
| 5 | 40 | 40 | 65.00 |
| **Worst-of-5** | — | — | **62.50** |

## 9. MMSU TALKER Wer

— 2× NVIDIA H20, MAX_SAMPLES=40, MAX_TOKENS=256, CONCURRENCY=16, 5 runs

| Run | Samples run | Samples ok | Corpus WER ≤50% (%) | Samples >50% WER |
|-----|--------|--------|--------|--------|
| 1 | 40 | 40 | 3.41 | 0 |
| 2 | 40 | 40 | 2.72 | 0 |
| 3 | 40 | 40 | 2.46 | 0 |
| 4 | 40 | 40 | 2.34 | 0 |
| 5 | 40 | 40 | 3.01 | 0 |
| **Worst-of-5** | — | — | **3.41** | **0** |

## 10. MMSU TALKER Speed

— 2× NVIDIA H20, MAX_SAMPLES=40, MAX_TOKENS=256, CONCURRENCY=16, 5 runs

| Run | Samples run | Samples ok | Throughput (req/s) | Output tok/req-s | Latency mean (s) | RTF mean |
|-----|--------|--------|--------|--------|--------|--------|
| 1 | 40 | 40 | 1.709 | 7.2 | 8.441 | 0.4719 |
| 2 | 40 | 40 | 1.713 | 7.2 | 8.382 | 0.4679 |
| 3 | 40 | 40 | 1.670 | 7.1 | 8.703 | 0.4792 |
| 4 | 40 | 40 | 1.696 | 7.3 | 8.474 | 0.4735 |
| 5 | 40 | 40 | 1.680 | 7.2 | 8.420 | 0.4678 |
| **Worst-of-5** | — | — | **1.670** | **7.1** | **8.703** | **0.4792** |

## 11. TTS Wer

— 2× NVIDIA H20, MAX_SAMPLES=50, CONCURRENCY=16, 5 runs

| Run | Samples run | Samples ok | Corpus WER ≤50% (%) | Samples >50% WER |
|-----|--------|--------|--------|--------|
| 1 | 50 | 50 | 1.42 | 0 |
| 2 | 50 | 50 | 1.24 | 0 |
| 3 | 50 | 50 | 1.42 | 0 |
| 4 | 50 | 50 | 1.24 | 0 |
| 5 | 50 | 50 | 1.24 | 0 |
| **Worst-of-5** | — | — | **1.42** | **0** |

## 12. TTS Speed

— 2× NVIDIA H20, MAX_SAMPLES=50, CONCURRENCY=16, 5 runs

| Run | Samples run | Samples ok | Throughput (req/s) | Output tok/req-s | Latency mean (s) | RTF mean |
|-----|--------|--------|--------|--------|--------|--------|
| 1 | 50 | 50 | 5.750 | 5.8 | 2.537 | 0.8085 |
| 2 | 50 | 50 | 5.899 | 5.8 | 2.522 | 0.7965 |
| 3 | 50 | 50 | 5.889 | 5.9 | 2.502 | 0.7874 |
| 4 | 50 | 50 | 5.842 | 5.7 | 2.581 | 0.8149 |
| 5 | 50 | 50 | 5.899 | 5.8 | 2.510 | 0.7918 |
| **Worst-of-5** | — | — | **5.750** | **5.7** | **2.581** | **0.8149** |

## 13. VIDEOAMME Accuracy

— 2× NVIDIA H20, MAX_SAMPLES=50, CONCURRENCY=16, 5 runs

| Run | Samples run | Samples ok | Acc (%) |
|-----|--------|--------|--------|
| 1 | 50 | 50 | 70.00 |
| 2 | 50 | 50 | 70.00 |
| 3 | 50 | 50 | 70.00 |
| 4 | 50 | 50 | 66.00 |
| 5 | 50 | 50 | 70.00 |
| **Worst-of-5** | — | — | **66.00** |

## 14. VIDEOAMME Speed

— 2× NVIDIA H20, MAX_SAMPLES=50, CONCURRENCY=16, 5 runs

| Run | Samples run | Samples ok | Throughput (req/s) | Output tok/req-s | Latency mean (s) |
|-----|--------|--------|--------|--------|--------|
| 1 | 50 | 50 | 1.030 | 3.0 | 14.651 |
| 2 | 50 | 50 | 1.200 | 3.5 | 11.930 |
| 3 | 50 | 50 | 1.156 | 3.3 | 12.573 |
| 4 | 50 | 50 | 1.143 | 3.5 | 12.402 |
| 5 | 50 | 50 | 0.995 | 2.7 | 15.716 |
| **Worst-of-5** | — | — | **0.995** | **2.7** | **15.716** |

## 15. VIDEOAMME TALKER Accuracy

— 2× NVIDIA H20, MAX_SAMPLES=20, MAX_TOKENS=256, CONCURRENCY=16, 5 runs

| Run | Samples run | Samples ok | Acc (%) |
|-----|--------|--------|--------|
| 1 | 20 | 20 | 65.00 |
| 2 | 20 | 20 | 65.00 |
| 3 | 20 | 20 | 60.00 |
| 4 | 20 | 20 | 60.00 |
| 5 | 20 | 20 | 65.00 |
| **Worst-of-5** | — | — | **60.00** |

## 16. VIDEOAMME TALKER Wer

— 2× NVIDIA H20, MAX_SAMPLES=20, MAX_TOKENS=256, CONCURRENCY=16, 5 runs

| Run | Samples run | Samples ok | Corpus WER ≤50% (%) | Samples >50% WER |
|-----|--------|--------|--------|--------|
| 1 | 20 | 20 | 1.38 | 1 |
| 2 | 20 | 20 | 0.44 | 1 |
| 3 | 20 | 20 | 0.00 | 0 |
| 4 | 20 | 20 | 0.54 | 1 |
| 5 | 20 | 20 | 0.88 | 1 |
| **Worst-of-5** | — | — | **1.38** | **1** |

## 17. VIDEOAMME TALKER Speed

— 2× NVIDIA H20, MAX_SAMPLES=20, MAX_TOKENS=256, CONCURRENCY=16, 5 runs

| Run | Samples run | Samples ok | Throughput (req/s) | Output tok/req-s | Latency mean (s) | RTF mean |
|-----|--------|--------|--------|--------|--------|--------|
| 1 | 20 | 20 | 0.637 | 2.1 | 21.298 | 3.0362 |
| 2 | 20 | 20 | 0.709 | 2.2 | 19.270 | 2.5422 |
| 3 | 20 | 20 | 0.606 | 2.1 | 22.427 | 3.3211 |
| 4 | 20 | 20 | 0.633 | 2.1 | 21.585 | 3.1129 |
| 5 | 20 | 20 | 0.705 | 2.2 | 19.689 | 2.2201 |
| **Worst-of-5** | — | — | **0.606** | **2.1** | **22.427** | **3.3211** |

## 18. VIDEOAMME TALKER TP2 Accuracy

— 2× NVIDIA H20, MAX_SAMPLES=10, MAX_TOKENS=256, CONCURRENCY=8, 5 runs

| Run | Samples run | Samples ok | Acc (%) |
|-----|--------|--------|--------|
| 1 | 10 | 10 | 50.00 |
| 2 | 10 | 10 | 60.00 |
| 3 | 10 | 10 | 50.00 |
| 4 | 10 | 10 | 50.00 |
| 5 | 10 | 10 | 50.00 |
| **Worst-of-5** | — | — | **50.00** |

## 19. VIDEOAMME TALKER TP2 Wer

— 2× NVIDIA H20, MAX_SAMPLES=10, MAX_TOKENS=256, CONCURRENCY=8, 5 runs

| Run | Samples run | Samples ok | Corpus WER ≤50% (%) | Samples >50% WER |
|-----|--------|--------|--------|--------|
| 1 | 10 | 10 | 0.85 | 1 |
| 2 | 10 | 10 | 0.83 | 1 |
| 3 | 10 | 10 | 0.57 | 1 |
| 4 | 10 | 10 | 0.74 | 0 |
| 5 | 10 | 10 | 0.30 | 1 |
| **Worst-of-5** | — | — | **0.85** | **1** |

## 20. VIDEOAMME TALKER TP2 Speed

— 2× NVIDIA H20, MAX_SAMPLES=10, MAX_TOKENS=256, CONCURRENCY=8, 5 runs

| Run | Samples run | Samples ok | Throughput (req/s) | Output tok/req-s | Latency mean (s) | RTF mean |
|-----|--------|--------|--------|--------|--------|--------|
| 1 | 10 | 10 | 0.050 | 0.3 | 137.945 | 16.9597 |
| 2 | 10 | 10 | 0.071 | 0.4 | 97.814 | 19.0820 |
| 3 | 10 | 10 | 0.063 | 0.4 | 112.602 | 22.4697 |
| 4 | 10 | 10 | 0.069 | 0.5 | 99.543 | 6.8810 |
| 5 | 10 | 10 | 0.070 | 0.4 | 98.256 | 19.6497 |
| **Worst-of-5** | — | — | **0.050** | **0.3** | **137.945** | **22.4697** |

## 21. VIDEOMME Accuracy

— 2× NVIDIA H20, MAX_SAMPLES=50, CONCURRENCY=16, 5 runs

| Run | Samples run | Samples ok | Acc (%) |
|-----|--------|--------|--------|
| 1 | 50 | 50 | 58.00 |
| 2 | 50 | 50 | 62.00 |
| 3 | 50 | 50 | 54.00 |
| 4 | 50 | 50 | 52.00 |
| 5 | 50 | 50 | 56.00 |
| **Worst-of-5** | — | — | **52.00** |

## 22. VIDEOMME Speed

— 2× NVIDIA H20, MAX_SAMPLES=50, CONCURRENCY=16, 5 runs

| Run | Samples run | Samples ok | Throughput (req/s) | Output tok/req-s | Latency mean (s) |
|-----|--------|--------|--------|--------|--------|
| 1 | 50 | 50 | 0.975 | 7.8 | 15.147 |
| 2 | 50 | 50 | 1.029 | 7.4 | 14.614 |
| 3 | 50 | 50 | 1.005 | 8.4 | 14.642 |
| 4 | 50 | 50 | 0.983 | 7.6 | 15.434 |
| 5 | 50 | 50 | 0.928 | 7.2 | 16.306 |
| **Worst-of-5** | — | — | **0.928** | **7.2** | **16.306** |

## 23. VIDEOMME TALKER Accuracy

— 2× NVIDIA H20, MAX_SAMPLES=20, MAX_TOKENS=256, CONCURRENCY=16, 5 runs

| Run | Samples run | Samples ok | Acc (%) |
|-----|--------|--------|--------|
| 1 | 20 | 20 | 65.00 |
| 2 | 20 | 20 | 60.00 |
| 3 | 20 | 20 | 60.00 |
| 4 | 20 | 20 | 60.00 |
| 5 | 20 | 20 | 60.00 |
| **Worst-of-5** | — | — | **60.00** |

## 24. VIDEOMME TALKER Wer

— 2× NVIDIA H20, MAX_SAMPLES=20, MAX_TOKENS=256, CONCURRENCY=16, 5 runs

| Run | Samples run | Samples ok | Corpus WER ≤50% (%) | Samples >50% WER |
|-----|--------|--------|--------|--------|
| 1 | 20 | 20 | 1.26 | 0 |
| 2 | 20 | 20 | 1.89 | 0 |
| 3 | 20 | 20 | 1.43 | 0 |
| 4 | 20 | 20 | 2.39 | 0 |
| 5 | 20 | 20 | 2.03 | 0 |
| **Worst-of-5** | — | — | **2.39** | **0** |

## 25. VIDEOMME TALKER Speed

— 2× NVIDIA H20, MAX_SAMPLES=20, MAX_TOKENS=256, CONCURRENCY=16, 5 runs

| Run | Samples run | Samples ok | Throughput (req/s) | Output tok/req-s | Latency mean (s) | RTF mean |
|-----|--------|--------|--------|--------|--------|--------|
| 1 | 20 | 20 | 0.624 | 2.2 | 20.226 | 2.0660 |
| 2 | 20 | 20 | 0.603 | 2.2 | 20.150 | 2.1134 |
| 3 | 20 | 20 | 0.612 | 2.2 | 19.792 | 2.0204 |
| 4 | 20 | 20 | 0.635 | 2.2 | 20.301 | 2.0087 |
| 5 | 20 | 20 | 0.608 | 2.3 | 20.486 | 1.9935 |
| **Worst-of-5** | — | — | **0.603** | **2.2** | **20.486** | **2.1134** |

## Applied changes

| Stage | Metric | Old | New | Direction |
|-------|--------|-----|-----|-----------|
| mmmu_speed | _MMMU_P95['throughput_qps'] | 1.245 | 1.318 | tightens (+5.9%) |
| mmmu_speed | _MMMU_P95['latency_mean_s'] | 10.881 | 10.32 | tightens (-5.2%) |
| mmsu_speed | _MMSU_P95['throughput_qps'] | 50.399 | 54.416 | tightens (+8.0%) |
| mmsu_speed | _MMSU_P95['output_tok_per_req_s'] | 6.5 | 7.1 | tightens (+9.2%) |
| mmsu_speed | _MMSU_P95['latency_mean_s'] | 0.317 | 0.293 | tightens (-7.6%) |
| videoamme_talker_speed | _VIDEOAMME_TALKER_AUDIO_P95['rtf_mean'] | 3.7926 | 3.3211 | tightens (-12.4%) |
| videoamme_talker_tp2_accuracy | VIDEOAMME_TALKER_TP2_THINKER_TEXT_MIN_ACCURACY | 0.4 | 0.5 | tightens (+25.0%) |
| videomme_talker_wer | VIDEOMME_TALKER_WER_BELOW_50_CORPUS_MAX | 0.037868162692847124 | 0.023876404494382022 | tightens (-36.9%) |
| videomme_talker_wer | VIDEOMME_TALKER_N_ABOVE_50_MAX | 2.0 | 0.0 | tightens (-100.0%) |
| videomme_talker_speed | _VIDEOMME_TALKER_AUDIO_P95['output_tok_per_req_s'] | 2.1 | 2.2 | tightens (+4.8%) |

## Provenance

- Model: qwen3-omni-v1
- Branch: calibration-chenyang-05-24 @ e6e9e428 (dirty) — see `workspace.diff`
- Venv Python: /sgl-workspace/sglang-omni/omni-qwen3/bin/python (flag)
- sglang 0.5.8 · torch 2.9.1+cu128
- GPU: 2× NVIDIA H20
- tune-ci-thresholds v0.4.1
- Ran 2026-05-25T02:43:02Z – 2026-05-25T02:54:44Z
