# CI Threshold Observation Report

**Calibration commit:** `8521cf6865686ad28b858f338e96d323caac4c60`
**Branch:** HEAD
**Run directory:** `/data/calibrations/20260925T0135Z/run_tts`
**Calibration started:** 2026-09-25T01:59:03Z
**Report generated:** 2026-09-25T02:48:21Z

## 1. TTS LATENCY QWEN3-TTS R1 Speed

{{CONTEXT:tts_latency_qwen3-tts_r1_speed}}

| Run | Samples run | Samples ok | First playable median (s) |
|-----|--------|--------|--------|
| 1 | 60 | 60 | 0.0581 |
| 2 | 60 | 60 | 0.0575 |
| 3 | 60 | 60 | 0.0573 |
| 4 | 60 | 60 | 0.0580 |
| 5 | 60 | 60 | 0.0577 |
| **Worst-of-5** | — | — | **0.0581** |

### Metric calibration

| Metric | Worst | Median | Min | Max | Range | Std | CV | Outliers |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| First playable median (s) | 0.0581 | 0.0577 | 0.0573 | 0.0581 | 0.0008 | 0.0003 | 0.006 | none |

## 2. TTS LATENCY QWEN3-TTS R20 Speed

{{CONTEXT:tts_latency_qwen3-tts_r20_speed}}

| Run | Samples run | Samples ok | First playable median (s) | First playable p95 (s) | Continuity c50 (%) |
|-----|--------|--------|--------|--------|--------|
| 1 | 1088 | 1088 | 0.1029 | 0.1852 | 99.72 |
| 2 | 1088 | 1088 | 0.1037 | 0.1760 | 99.36 |
| 3 | 1088 | 1088 | 0.1031 | 0.1816 | 99.54 |
| 4 | 1088 | 1088 | 0.1030 | 0.1782 | 99.72 |
| 5 | 1088 | 1088 | 0.1041 | 0.1739 | 99.45 |
| **Worst-of-5** | — | — | **0.1041** | **0.1852** | **99.36** |

### Metric calibration

| Metric | Worst | Median | Min | Max | Range | Std | CV | Outliers |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| First playable median (s) | 0.1041 | 0.1031 | 0.1029 | 0.1041 | 0.0012 | 0.0005 | 0.005 | none |
| First playable p95 (s) | 0.1852 | 0.1782 | 0.1739 | 0.1852 | 0.0113 | 0.0045 | 0.025 | none |
| Continuity c50 (%) | 99.36 | 99.54 | 99.36 | 99.72 | 0.36 | 0.16 | 0.002 | none |

## 3. TTS LATENCY QWEN3-TTS-CUSTOM-VOICE R1 Speed

{{CONTEXT:tts_latency_qwen3-tts-custom-voice_r1_speed}}

| Run | Samples run | Samples ok | First playable median (s) |
|-----|--------|--------|--------|
| 1 | 60 | 60 | 0.0212 |
| 2 | 60 | 60 | 0.0212 |
| 3 | 60 | 60 | 0.0215 |
| 4 | 60 | 60 | 0.0218 |
| 5 | 60 | 60 | 0.0213 |
| **Worst-of-5** | — | — | **0.0218** |

### Metric calibration

| Metric | Worst | Median | Min | Max | Range | Std | CV | Outliers |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| First playable median (s) | 0.0218 | 0.0213 | 0.0212 | 0.0218 | 0.0006 | 0.0003 | 0.012 | none |

## 4. TTS LATENCY QWEN3-TTS-CUSTOM-VOICE R20 Speed

{{CONTEXT:tts_latency_qwen3-tts-custom-voice_r20_speed}}

| Run | Samples run | Samples ok | First playable median (s) | First playable p95 (s) | Continuity c50 (%) |
|-----|--------|--------|--------|--------|--------|
| 1 | 1088 | 1088 | 0.0333 | 0.0477 | 100.00 |
| 2 | 1088 | 1088 | 0.0333 | 0.0466 | 100.00 |
| 3 | 1088 | 1088 | 0.0355 | 0.0487 | 100.00 |
| 4 | 1088 | 1088 | 0.0333 | 0.0475 | 100.00 |
| 5 | 1088 | 1088 | 0.0333 | 0.0475 | 100.00 |
| **Worst-of-5** | — | — | **0.0355** | **0.0487** | **100.00** |

### Metric calibration

| Metric | Worst | Median | Min | Max | Range | Std | CV | Outliers |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| First playable median (s) | 0.0355 | 0.0333 | 0.0333 | 0.0355 | 0.0022 | 0.0010 | 0.029 | run3 |
| First playable p95 (s) | 0.0487 | 0.0475 | 0.0466 | 0.0487 | 0.0021 | 0.0007 | 0.016 | run2, run3 |
| Continuity c50 (%) | 100.00 | 100.00 | 100.00 | 100.00 | 0.00 | 0.00 | 0.000 | none |

## Operational reliability

| Stage | Runs | Attempts | Retry successes | Failed attempts | Startup | OOM | Timeout | Partial |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| tts_latency_qwen3-tts_r1_speed | 5 | 5 | 0 | 0 | 0 | 0 | 0 | 0 |
| tts_latency_qwen3-tts_r20_speed | 5 | 5 | 0 | 0 | 0 | 0 | 0 | 0 |
| tts_latency_qwen3-tts-custom-voice_r1_speed | 5 | 5 | 0 | 0 | 0 | 0 | 0 | 0 |
| tts_latency_qwen3-tts-custom-voice_r20_speed | 5 | 5 | 0 | 0 | 0 | 0 | 0 | 0 |

## Provenance

- Model: tts
- Calibration commit: `8521cf6865686ad28b858f338e96d323caac4c60`
- Branch: HEAD
- Calibration started: 2026-09-25T01:59:03Z
- Venv Python: /github/home/calibration/omni/bin/python (flag)
- sglang 0.5.20 · torch 2.13.0+cu130
- GPU: 2× NVIDIA H100 80GB HBM3
- GPU group: [0, 1]
- Environment comparability: core-pins-and-image-identity-recorded
- Container image digest: sha256:ebe4239e29a764ee3a2806385c061c5fd438a26f01458e503d3822dcba5790df
- Dependency freeze SHA256: 832769ce389bc5a151d4b7b060b48cc771648570fd2131e15de1b22bb7f08a76
- tune-ci-thresholds v0.8.0
- Report generated: 2026-09-25T02:48:21Z
