# CI Calibration Raw Metrics (worst-of-5 inputs)

## qwen3-omni-v1
Run dir: `.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5`

### mmmu_accuracy

#### run 1
```json
{
  "status": "ok",
  "reason": "",
  "metrics": {
    "accuracy": 0.62
  },
  "sample_counts": {
    "total": 50,
    "ok": 50
  },
  "duration_s": 159.54,
  "attempts": 1,
  "pytest_log": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_mmmu_ci/run1.log",
  "basetemp": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_mmmu_ci/basetemp_run1",
  "pytest_rc": 0
}
```

#### run 2
```json
{
  "status": "ok",
  "reason": "threshold_assertion (exit 1)",
  "metrics": {
    "accuracy": 0.58
  },
  "sample_counts": {
    "total": 50,
    "ok": 50
  },
  "duration_s": 159.5,
  "attempts": 1,
  "pytest_log": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_mmmu_ci/run2.log",
  "basetemp": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_mmmu_ci/basetemp_run2",
  "pytest_rc": 1
}
```

#### run 3
```json
{
  "status": "ok",
  "reason": "",
  "metrics": {
    "accuracy": 0.64
  },
  "sample_counts": {
    "total": 50,
    "ok": 50
  },
  "duration_s": 159.6,
  "attempts": 1,
  "pytest_log": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_mmmu_ci/run3.log",
  "basetemp": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_mmmu_ci/basetemp_run3",
  "pytest_rc": 0
}
```

#### run 4
```json
{
  "status": "ok",
  "reason": "threshold_assertion (exit 1)",
  "metrics": {
    "accuracy": 0.56
  },
  "sample_counts": {
    "total": 50,
    "ok": 50
  },
  "duration_s": 159.46,
  "attempts": 1,
  "pytest_log": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_mmmu_ci/run4.log",
  "basetemp": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_mmmu_ci/basetemp_run4",
  "pytest_rc": 1
}
```

#### run 5
```json
{
  "status": "ok",
  "reason": "threshold_assertion (exit 1)",
  "metrics": {
    "accuracy": 0.6
  },
  "sample_counts": {
    "total": 50,
    "ok": 50
  },
  "duration_s": 159.57,
  "attempts": 1,
  "pytest_log": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_mmmu_ci/run5.log",
  "basetemp": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_mmmu_ci/basetemp_run5",
  "pytest_rc": 1
}
```

### mmmu_speed

#### run 1
```json
{
  "status": "ok",
  "reason": "",
  "metrics": {
    "throughput_qps": 1.318,
    "output_tok_per_req_s": 58.7,
    "latency_mean_s": 10.131
  },
  "sample_counts": {
    "total": 50,
    "ok": 50
  },
  "duration_s": 159.54,
  "attempts": 1,
  "pytest_log": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_mmmu_ci/run1.log",
  "basetemp": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_mmmu_ci/basetemp_run1",
  "pytest_rc": 0
}
```

#### run 2
```json
{
  "status": "ok",
  "reason": "threshold_assertion (exit 1)",
  "metrics": {
    "throughput_qps": 1.364,
    "output_tok_per_req_s": 57.0,
    "latency_mean_s": 10.32
  },
  "sample_counts": {
    "total": 50,
    "ok": 50
  },
  "duration_s": 159.5,
  "attempts": 1,
  "pytest_log": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_mmmu_ci/run2.log",
  "basetemp": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_mmmu_ci/basetemp_run2",
  "pytest_rc": 1
}
```

#### run 3
```json
{
  "status": "ok",
  "reason": "",
  "metrics": {
    "throughput_qps": 1.349,
    "output_tok_per_req_s": 54.0,
    "latency_mean_s": 9.956
  },
  "sample_counts": {
    "total": 50,
    "ok": 50
  },
  "duration_s": 159.6,
  "attempts": 1,
  "pytest_log": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_mmmu_ci/run3.log",
  "basetemp": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_mmmu_ci/basetemp_run3",
  "pytest_rc": 0
}
```

#### run 4
```json
{
  "status": "ok",
  "reason": "threshold_assertion (exit 1)",
  "metrics": {
    "throughput_qps": 1.351,
    "output_tok_per_req_s": 58.4,
    "latency_mean_s": 10.266
  },
  "sample_counts": {
    "total": 50,
    "ok": 50
  },
  "duration_s": 159.46,
  "attempts": 1,
  "pytest_log": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_mmmu_ci/run4.log",
  "basetemp": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_mmmu_ci/basetemp_run4",
  "pytest_rc": 1
}
```

#### run 5
```json
{
  "status": "ok",
  "reason": "threshold_assertion (exit 1)",
  "metrics": {
    "throughput_qps": 1.346,
    "output_tok_per_req_s": 58.8,
    "latency_mean_s": 10.085
  },
  "sample_counts": {
    "total": 50,
    "ok": 50
  },
  "duration_s": 159.57,
  "attempts": 1,
  "pytest_log": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_mmmu_ci/run5.log",
  "basetemp": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_mmmu_ci/basetemp_run5",
  "pytest_rc": 1
}
```

### mmmu_talker_accuracy

#### run 1
```json
{
  "status": "ok",
  "reason": "",
  "metrics": {
    "accuracy": 0.75
  },
  "sample_counts": {
    "total": 20,
    "ok": 20
  },
  "duration_s": 219.65,
  "attempts": 1,
  "pytest_log": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_mmmu_talker_ci/run1.log",
  "basetemp": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_mmmu_talker_ci/basetemp_run1",
  "pytest_rc": 0
}
```

#### run 2
```json
{
  "status": "ok",
  "reason": "threshold_assertion (exit 1)",
  "metrics": {
    "accuracy": 0.75
  },
  "sample_counts": {
    "total": 20,
    "ok": 20
  },
  "duration_s": 219.57,
  "attempts": 1,
  "pytest_log": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_mmmu_talker_ci/run2.log",
  "basetemp": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_mmmu_talker_ci/basetemp_run2",
  "pytest_rc": 1
}
```

#### run 3
```json
{
  "status": "ok",
  "reason": "",
  "metrics": {
    "accuracy": 0.75
  },
  "sample_counts": {
    "total": 20,
    "ok": 20
  },
  "duration_s": 219.55,
  "attempts": 1,
  "pytest_log": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_mmmu_talker_ci/run3.log",
  "basetemp": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_mmmu_talker_ci/basetemp_run3",
  "pytest_rc": 0
}
```

#### run 4
```json
{
  "status": "ok",
  "reason": "threshold_assertion (exit 1)",
  "metrics": {
    "accuracy": 0.75
  },
  "sample_counts": {
    "total": 20,
    "ok": 20
  },
  "duration_s": 279.77,
  "attempts": 1,
  "pytest_log": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_mmmu_talker_ci/run4.log",
  "basetemp": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_mmmu_talker_ci/basetemp_run4",
  "pytest_rc": 1
}
```

#### run 5
```json
{
  "status": "ok",
  "reason": "",
  "metrics": {
    "accuracy": 0.75
  },
  "sample_counts": {
    "total": 20,
    "ok": 20
  },
  "duration_s": 219.65,
  "attempts": 1,
  "pytest_log": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_mmmu_talker_ci/run5.log",
  "basetemp": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_mmmu_talker_ci/basetemp_run5",
  "pytest_rc": 0
}
```

### mmmu_talker_wer

#### run 1
```json
{
  "status": "ok",
  "reason": "",
  "metrics": {
    "wer_below_50_corpus": 0.1500815660685155,
    "n_above_50": 2.0
  },
  "sample_counts": {
    "total": 20,
    "ok": 20
  },
  "duration_s": 219.65,
  "attempts": 1,
  "pytest_log": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_mmmu_talker_ci/run1.log",
  "basetemp": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_mmmu_talker_ci/basetemp_run1",
  "pytest_rc": 0
}
```

#### run 2
```json
{
  "status": "ok",
  "reason": "threshold_assertion (exit 1)",
  "metrics": {
    "wer_below_50_corpus": 0.15267175572519084,
    "n_above_50": 3.0
  },
  "sample_counts": {
    "total": 20,
    "ok": 20
  },
  "duration_s": 219.57,
  "attempts": 1,
  "pytest_log": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_mmmu_talker_ci/run2.log",
  "basetemp": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_mmmu_talker_ci/basetemp_run2",
  "pytest_rc": 1
}
```

#### run 3
```json
{
  "status": "ok",
  "reason": "",
  "metrics": {
    "wer_below_50_corpus": 0.18145483613109512,
    "n_above_50": 2.0
  },
  "sample_counts": {
    "total": 20,
    "ok": 20
  },
  "duration_s": 219.55,
  "attempts": 1,
  "pytest_log": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_mmmu_talker_ci/run3.log",
  "basetemp": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_mmmu_talker_ci/basetemp_run3",
  "pytest_rc": 0
}
```

#### run 4
```json
{
  "status": "ok",
  "reason": "threshold_assertion (exit 1)",
  "metrics": {
    "wer_below_50_corpus": 0.11523046092184369,
    "n_above_50": 4.0
  },
  "sample_counts": {
    "total": 20,
    "ok": 20
  },
  "duration_s": 279.77,
  "attempts": 1,
  "pytest_log": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_mmmu_talker_ci/run4.log",
  "basetemp": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_mmmu_talker_ci/basetemp_run4",
  "pytest_rc": 1
}
```

#### run 5
```json
{
  "status": "ok",
  "reason": "",
  "metrics": {
    "wer_below_50_corpus": 0.18152610441767067,
    "n_above_50": 3.0
  },
  "sample_counts": {
    "total": 20,
    "ok": 20
  },
  "duration_s": 219.65,
  "attempts": 1,
  "pytest_log": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_mmmu_talker_ci/run5.log",
  "basetemp": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_mmmu_talker_ci/basetemp_run5",
  "pytest_rc": 0
}
```

### mmmu_talker_speed

#### run 1
```json
{
  "status": "ok",
  "reason": "",
  "metrics": {
    "throughput_qps": 0.717,
    "output_tok_per_req_s": 8.1,
    "latency_mean_s": 16.126,
    "rtf_mean": 0.4309
  },
  "sample_counts": {
    "total": 20,
    "ok": 20
  },
  "duration_s": 219.65,
  "attempts": 1,
  "pytest_log": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_mmmu_talker_ci/run1.log",
  "basetemp": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_mmmu_talker_ci/basetemp_run1",
  "pytest_rc": 0
}
```

#### run 2
```json
{
  "status": "ok",
  "reason": "threshold_assertion (exit 1)",
  "metrics": {
    "throughput_qps": 0.713,
    "output_tok_per_req_s": 8.2,
    "latency_mean_s": 16.666,
    "rtf_mean": 0.4252
  },
  "sample_counts": {
    "total": 20,
    "ok": 20
  },
  "duration_s": 219.57,
  "attempts": 1,
  "pytest_log": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_mmmu_talker_ci/run2.log",
  "basetemp": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_mmmu_talker_ci/basetemp_run2",
  "pytest_rc": 1
}
```

#### run 3
```json
{
  "status": "ok",
  "reason": "",
  "metrics": {
    "throughput_qps": 0.689,
    "output_tok_per_req_s": 8.2,
    "latency_mean_s": 16.09,
    "rtf_mean": 0.4178
  },
  "sample_counts": {
    "total": 20,
    "ok": 20
  },
  "duration_s": 219.55,
  "attempts": 1,
  "pytest_log": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_mmmu_talker_ci/run3.log",
  "basetemp": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_mmmu_talker_ci/basetemp_run3",
  "pytest_rc": 0
}
```

#### run 4
```json
{
  "status": "ok",
  "reason": "threshold_assertion (exit 1)",
  "metrics": {
    "throughput_qps": 0.617,
    "output_tok_per_req_s": 8.0,
    "latency_mean_s": 16.187,
    "rtf_mean": 0.4223
  },
  "sample_counts": {
    "total": 20,
    "ok": 20
  },
  "duration_s": 279.77,
  "attempts": 1,
  "pytest_log": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_mmmu_talker_ci/run4.log",
  "basetemp": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_mmmu_talker_ci/basetemp_run4",
  "pytest_rc": 1
}
```

#### run 5
```json
{
  "status": "ok",
  "reason": "",
  "metrics": {
    "throughput_qps": 0.687,
    "output_tok_per_req_s": 8.2,
    "latency_mean_s": 15.995,
    "rtf_mean": 0.4157
  },
  "sample_counts": {
    "total": 20,
    "ok": 20
  },
  "duration_s": 219.65,
  "attempts": 1,
  "pytest_log": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_mmmu_talker_ci/run5.log",
  "basetemp": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_mmmu_talker_ci/basetemp_run5",
  "pytest_rc": 0
}
```

### mmsu_accuracy

#### run 1
```json
{
  "status": "ok",
  "reason": "",
  "metrics": {
    "accuracy": 0.7005
  },
  "sample_counts": {
    "total": 2000,
    "ok": 2000
  },
  "duration_s": 129.44,
  "attempts": 1,
  "pytest_log": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_mmsu_ci/run1.log",
  "basetemp": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_mmsu_ci/basetemp_run1",
  "pytest_rc": 0
}
```

#### run 2
```json
{
  "status": "ok",
  "reason": "",
  "metrics": {
    "accuracy": 0.7025
  },
  "sample_counts": {
    "total": 2000,
    "ok": 2000
  },
  "duration_s": 129.52,
  "attempts": 1,
  "pytest_log": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_mmsu_ci/run2.log",
  "basetemp": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_mmsu_ci/basetemp_run2",
  "pytest_rc": 0
}
```

#### run 3
```json
{
  "status": "ok",
  "reason": "",
  "metrics": {
    "accuracy": 0.701
  },
  "sample_counts": {
    "total": 2000,
    "ok": 2000
  },
  "duration_s": 129.46,
  "attempts": 1,
  "pytest_log": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_mmsu_ci/run3.log",
  "basetemp": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_mmsu_ci/basetemp_run3",
  "pytest_rc": 0
}
```

#### run 4
```json
{
  "status": "ok",
  "reason": "threshold_assertion (exit 1)",
  "metrics": {
    "accuracy": 0.6945
  },
  "sample_counts": {
    "total": 2000,
    "ok": 2000
  },
  "duration_s": 129.41,
  "attempts": 1,
  "pytest_log": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_mmsu_ci/run4.log",
  "basetemp": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_mmsu_ci/basetemp_run4",
  "pytest_rc": 1
}
```

#### run 5
```json
{
  "status": "ok",
  "reason": "",
  "metrics": {
    "accuracy": 0.708
  },
  "sample_counts": {
    "total": 2000,
    "ok": 2000
  },
  "duration_s": 129.5,
  "attempts": 1,
  "pytest_log": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_mmsu_ci/run5.log",
  "basetemp": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_mmsu_ci/basetemp_run5",
  "pytest_rc": 0
}
```

### mmsu_speed

#### run 1
```json
{
  "status": "ok",
  "reason": "",
  "metrics": {
    "throughput_qps": 54.416,
    "output_tok_per_req_s": 7.1,
    "latency_mean_s": 0.293
  },
  "sample_counts": {
    "total": 2000,
    "ok": 2000
  },
  "duration_s": 129.44,
  "attempts": 1,
  "pytest_log": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_mmsu_ci/run1.log",
  "basetemp": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_mmsu_ci/basetemp_run1",
  "pytest_rc": 0
}
```

#### run 2
```json
{
  "status": "ok",
  "reason": "",
  "metrics": {
    "throughput_qps": 58.788,
    "output_tok_per_req_s": 7.6,
    "latency_mean_s": 0.272
  },
  "sample_counts": {
    "total": 2000,
    "ok": 2000
  },
  "duration_s": 129.52,
  "attempts": 1,
  "pytest_log": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_mmsu_ci/run2.log",
  "basetemp": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_mmsu_ci/basetemp_run2",
  "pytest_rc": 0
}
```

#### run 3
```json
{
  "status": "ok",
  "reason": "",
  "metrics": {
    "throughput_qps": 58.615,
    "output_tok_per_req_s": 7.6,
    "latency_mean_s": 0.272
  },
  "sample_counts": {
    "total": 2000,
    "ok": 2000
  },
  "duration_s": 129.46,
  "attempts": 1,
  "pytest_log": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_mmsu_ci/run3.log",
  "basetemp": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_mmsu_ci/basetemp_run3",
  "pytest_rc": 0
}
```

#### run 4
```json
{
  "status": "ok",
  "reason": "threshold_assertion (exit 1)",
  "metrics": {
    "throughput_qps": 57.77,
    "output_tok_per_req_s": 7.5,
    "latency_mean_s": 0.276
  },
  "sample_counts": {
    "total": 2000,
    "ok": 2000
  },
  "duration_s": 129.41,
  "attempts": 1,
  "pytest_log": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_mmsu_ci/run4.log",
  "basetemp": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_mmsu_ci/basetemp_run4",
  "pytest_rc": 1
}
```

#### run 5
```json
{
  "status": "ok",
  "reason": "",
  "metrics": {
    "throughput_qps": 58.594,
    "output_tok_per_req_s": 7.6,
    "latency_mean_s": 0.272
  },
  "sample_counts": {
    "total": 2000,
    "ok": 2000
  },
  "duration_s": 129.5,
  "attempts": 1,
  "pytest_log": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_mmsu_ci/run5.log",
  "basetemp": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_mmsu_ci/basetemp_run5",
  "pytest_rc": 0
}
```

### mmsu_talker_accuracy

#### run 1
```json
{
  "status": "ok",
  "reason": "",
  "metrics": {
    "accuracy": 0.65
  },
  "sample_counts": {
    "total": 40,
    "ok": 40
  },
  "duration_s": 189.57,
  "attempts": 1,
  "pytest_log": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_mmsu_talker_ci/run1.log",
  "basetemp": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_mmsu_talker_ci/basetemp_run1",
  "pytest_rc": 0
}
```

#### run 2
```json
{
  "status": "ok",
  "reason": "threshold_assertion (exit 1)",
  "metrics": {
    "accuracy": 0.625
  },
  "sample_counts": {
    "total": 40,
    "ok": 40
  },
  "duration_s": 189.58,
  "attempts": 1,
  "pytest_log": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_mmsu_talker_ci/run2.log",
  "basetemp": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_mmsu_talker_ci/basetemp_run2",
  "pytest_rc": 1
}
```

#### run 3
```json
{
  "status": "ok",
  "reason": "",
  "metrics": {
    "accuracy": 0.65
  },
  "sample_counts": {
    "total": 40,
    "ok": 40
  },
  "duration_s": 219.68,
  "attempts": 1,
  "pytest_log": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_mmsu_talker_ci/run3.log",
  "basetemp": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_mmsu_talker_ci/basetemp_run3",
  "pytest_rc": 0
}
```

#### run 4
```json
{
  "status": "ok",
  "reason": "threshold_assertion (exit 1)",
  "metrics": {
    "accuracy": 0.625
  },
  "sample_counts": {
    "total": 40,
    "ok": 40
  },
  "duration_s": 219.6,
  "attempts": 1,
  "pytest_log": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_mmsu_talker_ci/run4.log",
  "basetemp": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_mmsu_talker_ci/basetemp_run4",
  "pytest_rc": 1
}
```

#### run 5
```json
{
  "status": "ok",
  "reason": "",
  "metrics": {
    "accuracy": 0.65
  },
  "sample_counts": {
    "total": 40,
    "ok": 40
  },
  "duration_s": 189.57,
  "attempts": 1,
  "pytest_log": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_mmsu_talker_ci/run5.log",
  "basetemp": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_mmsu_talker_ci/basetemp_run5",
  "pytest_rc": 0
}
```

### mmsu_talker_wer

#### run 1
```json
{
  "status": "ok",
  "reason": "",
  "metrics": {
    "wer_below_50_corpus": 0.03411513859275053,
    "n_above_50": 0.0
  },
  "sample_counts": {
    "total": 40,
    "ok": 40
  },
  "duration_s": 189.57,
  "attempts": 1,
  "pytest_log": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_mmsu_talker_ci/run1.log",
  "basetemp": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_mmsu_talker_ci/basetemp_run1",
  "pytest_rc": 0
}
```

#### run 2
```json
{
  "status": "ok",
  "reason": "threshold_assertion (exit 1)",
  "metrics": {
    "wer_below_50_corpus": 0.027243589743589744,
    "n_above_50": 0.0
  },
  "sample_counts": {
    "total": 40,
    "ok": 40
  },
  "duration_s": 189.58,
  "attempts": 1,
  "pytest_log": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_mmsu_talker_ci/run2.log",
  "basetemp": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_mmsu_talker_ci/basetemp_run2",
  "pytest_rc": 1
}
```

#### run 3
```json
{
  "status": "ok",
  "reason": "",
  "metrics": {
    "wer_below_50_corpus": 0.02462022001047669,
    "n_above_50": 0.0
  },
  "sample_counts": {
    "total": 40,
    "ok": 40
  },
  "duration_s": 219.68,
  "attempts": 1,
  "pytest_log": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_mmsu_talker_ci/run3.log",
  "basetemp": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_mmsu_talker_ci/basetemp_run3",
  "pytest_rc": 0
}
```

#### run 4
```json
{
  "status": "ok",
  "reason": "threshold_assertion (exit 1)",
  "metrics": {
    "wer_below_50_corpus": 0.02336448598130841,
    "n_above_50": 0.0
  },
  "sample_counts": {
    "total": 40,
    "ok": 40
  },
  "duration_s": 219.6,
  "attempts": 1,
  "pytest_log": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_mmsu_talker_ci/run4.log",
  "basetemp": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_mmsu_talker_ci/basetemp_run4",
  "pytest_rc": 1
}
```

#### run 5
```json
{
  "status": "ok",
  "reason": "",
  "metrics": {
    "wer_below_50_corpus": 0.03007518796992481,
    "n_above_50": 0.0
  },
  "sample_counts": {
    "total": 40,
    "ok": 40
  },
  "duration_s": 189.57,
  "attempts": 1,
  "pytest_log": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_mmsu_talker_ci/run5.log",
  "basetemp": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_mmsu_talker_ci/basetemp_run5",
  "pytest_rc": 0
}
```

### mmsu_talker_speed

#### run 1
```json
{
  "status": "ok",
  "reason": "",
  "metrics": {
    "throughput_qps": 1.709,
    "output_tok_per_req_s": 7.2,
    "latency_mean_s": 8.441,
    "rtf_mean": 0.4719
  },
  "sample_counts": {
    "total": 40,
    "ok": 40
  },
  "duration_s": 189.57,
  "attempts": 1,
  "pytest_log": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_mmsu_talker_ci/run1.log",
  "basetemp": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_mmsu_talker_ci/basetemp_run1",
  "pytest_rc": 0
}
```

#### run 2
```json
{
  "status": "ok",
  "reason": "threshold_assertion (exit 1)",
  "metrics": {
    "throughput_qps": 1.713,
    "output_tok_per_req_s": 7.2,
    "latency_mean_s": 8.382,
    "rtf_mean": 0.4679
  },
  "sample_counts": {
    "total": 40,
    "ok": 40
  },
  "duration_s": 189.58,
  "attempts": 1,
  "pytest_log": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_mmsu_talker_ci/run2.log",
  "basetemp": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_mmsu_talker_ci/basetemp_run2",
  "pytest_rc": 1
}
```

#### run 3
```json
{
  "status": "ok",
  "reason": "",
  "metrics": {
    "throughput_qps": 1.67,
    "output_tok_per_req_s": 7.1,
    "latency_mean_s": 8.703,
    "rtf_mean": 0.4792
  },
  "sample_counts": {
    "total": 40,
    "ok": 40
  },
  "duration_s": 219.68,
  "attempts": 1,
  "pytest_log": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_mmsu_talker_ci/run3.log",
  "basetemp": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_mmsu_talker_ci/basetemp_run3",
  "pytest_rc": 0
}
```

#### run 4
```json
{
  "status": "ok",
  "reason": "threshold_assertion (exit 1)",
  "metrics": {
    "throughput_qps": 1.696,
    "output_tok_per_req_s": 7.3,
    "latency_mean_s": 8.474,
    "rtf_mean": 0.4735
  },
  "sample_counts": {
    "total": 40,
    "ok": 40
  },
  "duration_s": 219.6,
  "attempts": 1,
  "pytest_log": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_mmsu_talker_ci/run4.log",
  "basetemp": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_mmsu_talker_ci/basetemp_run4",
  "pytest_rc": 1
}
```

#### run 5
```json
{
  "status": "ok",
  "reason": "",
  "metrics": {
    "throughput_qps": 1.68,
    "output_tok_per_req_s": 7.2,
    "latency_mean_s": 8.42,
    "rtf_mean": 0.4678
  },
  "sample_counts": {
    "total": 40,
    "ok": 40
  },
  "duration_s": 189.57,
  "attempts": 1,
  "pytest_log": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_mmsu_talker_ci/run5.log",
  "basetemp": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_mmsu_talker_ci/basetemp_run5",
  "pytest_rc": 0
}
```

### tts_wer

#### run 1
```json
{
  "status": "ok",
  "reason": "",
  "metrics": {
    "wer_below_50_corpus": 0.014184397163120567,
    "n_above_50": 0.0
  },
  "sample_counts": {
    "total": 50,
    "ok": 50
  },
  "duration_s": 279.8,
  "attempts": 1,
  "pytest_log": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_tts_ci/run1.log",
  "basetemp": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_tts_ci/basetemp_run1",
  "pytest_rc": 0
}
```

#### run 2
```json
{
  "status": "ok",
  "reason": "",
  "metrics": {
    "wer_below_50_corpus": 0.012411347517730497,
    "n_above_50": 0.0
  },
  "sample_counts": {
    "total": 50,
    "ok": 50
  },
  "duration_s": 159.46,
  "attempts": 1,
  "pytest_log": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_tts_ci/run2.log",
  "basetemp": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_tts_ci/basetemp_run2",
  "pytest_rc": 0
}
```

#### run 3
```json
{
  "status": "ok",
  "reason": "",
  "metrics": {
    "wer_below_50_corpus": 0.014184397163120567,
    "n_above_50": 0.0
  },
  "sample_counts": {
    "total": 50,
    "ok": 50
  },
  "duration_s": 189.54,
  "attempts": 1,
  "pytest_log": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_tts_ci/run3.log",
  "basetemp": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_tts_ci/basetemp_run3",
  "pytest_rc": 0
}
```

#### run 4
```json
{
  "status": "ok",
  "reason": "",
  "metrics": {
    "wer_below_50_corpus": 0.012411347517730497,
    "n_above_50": 0.0
  },
  "sample_counts": {
    "total": 50,
    "ok": 50
  },
  "duration_s": 189.68,
  "attempts": 1,
  "pytest_log": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_tts_ci/run4.log",
  "basetemp": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_tts_ci/basetemp_run4",
  "pytest_rc": 0
}
```

#### run 5
```json
{
  "status": "ok",
  "reason": "",
  "metrics": {
    "wer_below_50_corpus": 0.012411347517730497,
    "n_above_50": 0.0
  },
  "sample_counts": {
    "total": 50,
    "ok": 50
  },
  "duration_s": 159.55,
  "attempts": 1,
  "pytest_log": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_tts_ci/run5.log",
  "basetemp": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_tts_ci/basetemp_run5",
  "pytest_rc": 0
}
```

### tts_speed

#### run 1
```json
{
  "status": "ok",
  "reason": "",
  "metrics": {
    "throughput_qps": 5.75,
    "output_tok_per_req_s": 5.8,
    "latency_mean_s": 2.537,
    "rtf_mean": 0.8085
  },
  "sample_counts": {
    "total": 50,
    "ok": 50
  },
  "duration_s": 279.8,
  "attempts": 1,
  "pytest_log": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_tts_ci/run1.log",
  "basetemp": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_tts_ci/basetemp_run1",
  "pytest_rc": 0
}
```

#### run 2
```json
{
  "status": "ok",
  "reason": "",
  "metrics": {
    "throughput_qps": 5.899,
    "output_tok_per_req_s": 5.8,
    "latency_mean_s": 2.522,
    "rtf_mean": 0.7965
  },
  "sample_counts": {
    "total": 50,
    "ok": 50
  },
  "duration_s": 159.46,
  "attempts": 1,
  "pytest_log": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_tts_ci/run2.log",
  "basetemp": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_tts_ci/basetemp_run2",
  "pytest_rc": 0
}
```

#### run 3
```json
{
  "status": "ok",
  "reason": "",
  "metrics": {
    "throughput_qps": 5.889,
    "output_tok_per_req_s": 5.9,
    "latency_mean_s": 2.502,
    "rtf_mean": 0.7874
  },
  "sample_counts": {
    "total": 50,
    "ok": 50
  },
  "duration_s": 189.54,
  "attempts": 1,
  "pytest_log": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_tts_ci/run3.log",
  "basetemp": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_tts_ci/basetemp_run3",
  "pytest_rc": 0
}
```

#### run 4
```json
{
  "status": "ok",
  "reason": "",
  "metrics": {
    "throughput_qps": 5.842,
    "output_tok_per_req_s": 5.7,
    "latency_mean_s": 2.581,
    "rtf_mean": 0.8149
  },
  "sample_counts": {
    "total": 50,
    "ok": 50
  },
  "duration_s": 189.68,
  "attempts": 1,
  "pytest_log": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_tts_ci/run4.log",
  "basetemp": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_tts_ci/basetemp_run4",
  "pytest_rc": 0
}
```

#### run 5
```json
{
  "status": "ok",
  "reason": "",
  "metrics": {
    "throughput_qps": 5.899,
    "output_tok_per_req_s": 5.8,
    "latency_mean_s": 2.51,
    "rtf_mean": 0.7918
  },
  "sample_counts": {
    "total": 50,
    "ok": 50
  },
  "duration_s": 159.55,
  "attempts": 1,
  "pytest_log": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_tts_ci/run5.log",
  "basetemp": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_tts_ci/basetemp_run5",
  "pytest_rc": 0
}
```

### videoamme_accuracy

#### run 1
```json
{
  "status": "ok",
  "reason": "",
  "metrics": {
    "accuracy": 0.7
  },
  "sample_counts": {
    "total": 50,
    "ok": 50
  },
  "duration_s": 159.74,
  "attempts": 1,
  "pytest_log": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_videoamme_ci/run1.log",
  "basetemp": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_videoamme_ci/basetemp_run1",
  "pytest_rc": 0
}
```

#### run 2
```json
{
  "status": "ok",
  "reason": "",
  "metrics": {
    "accuracy": 0.7
  },
  "sample_counts": {
    "total": 50,
    "ok": 50
  },
  "duration_s": 159.54,
  "attempts": 1,
  "pytest_log": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_videoamme_ci/run2.log",
  "basetemp": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_videoamme_ci/basetemp_run2",
  "pytest_rc": 0
}
```

#### run 3
```json
{
  "status": "ok",
  "reason": "",
  "metrics": {
    "accuracy": 0.7
  },
  "sample_counts": {
    "total": 50,
    "ok": 50
  },
  "duration_s": 159.48,
  "attempts": 1,
  "pytest_log": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_videoamme_ci/run3.log",
  "basetemp": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_videoamme_ci/basetemp_run3",
  "pytest_rc": 0
}
```

#### run 4
```json
{
  "status": "ok",
  "reason": "",
  "metrics": {
    "accuracy": 0.66
  },
  "sample_counts": {
    "total": 50,
    "ok": 50
  },
  "duration_s": 159.53,
  "attempts": 1,
  "pytest_log": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_videoamme_ci/run4.log",
  "basetemp": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_videoamme_ci/basetemp_run4",
  "pytest_rc": 0
}
```

#### run 5
```json
{
  "status": "ok",
  "reason": "threshold_assertion (exit 1)",
  "metrics": {
    "accuracy": 0.7
  },
  "sample_counts": {
    "total": 50,
    "ok": 50
  },
  "duration_s": 159.61,
  "attempts": 1,
  "pytest_log": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_videoamme_ci/run5.log",
  "basetemp": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_videoamme_ci/basetemp_run5",
  "pytest_rc": 1
}
```

### videoamme_speed

#### run 1
```json
{
  "status": "ok",
  "reason": "",
  "metrics": {
    "throughput_qps": 1.03,
    "output_tok_per_req_s": 3.0,
    "latency_mean_s": 14.651
  },
  "sample_counts": {
    "total": 50,
    "ok": 50
  },
  "duration_s": 159.74,
  "attempts": 1,
  "pytest_log": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_videoamme_ci/run1.log",
  "basetemp": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_videoamme_ci/basetemp_run1",
  "pytest_rc": 0
}
```

#### run 2
```json
{
  "status": "ok",
  "reason": "",
  "metrics": {
    "throughput_qps": 1.2,
    "output_tok_per_req_s": 3.5,
    "latency_mean_s": 11.93
  },
  "sample_counts": {
    "total": 50,
    "ok": 50
  },
  "duration_s": 159.54,
  "attempts": 1,
  "pytest_log": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_videoamme_ci/run2.log",
  "basetemp": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_videoamme_ci/basetemp_run2",
  "pytest_rc": 0
}
```

#### run 3
```json
{
  "status": "ok",
  "reason": "",
  "metrics": {
    "throughput_qps": 1.156,
    "output_tok_per_req_s": 3.3,
    "latency_mean_s": 12.573
  },
  "sample_counts": {
    "total": 50,
    "ok": 50
  },
  "duration_s": 159.48,
  "attempts": 1,
  "pytest_log": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_videoamme_ci/run3.log",
  "basetemp": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_videoamme_ci/basetemp_run3",
  "pytest_rc": 0
}
```

#### run 4
```json
{
  "status": "ok",
  "reason": "",
  "metrics": {
    "throughput_qps": 1.143,
    "output_tok_per_req_s": 3.5,
    "latency_mean_s": 12.402
  },
  "sample_counts": {
    "total": 50,
    "ok": 50
  },
  "duration_s": 159.53,
  "attempts": 1,
  "pytest_log": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_videoamme_ci/run4.log",
  "basetemp": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_videoamme_ci/basetemp_run4",
  "pytest_rc": 0
}
```

#### run 5
```json
{
  "status": "ok",
  "reason": "threshold_assertion (exit 1)",
  "metrics": {
    "throughput_qps": 0.995,
    "output_tok_per_req_s": 2.7,
    "latency_mean_s": 15.716
  },
  "sample_counts": {
    "total": 50,
    "ok": 50
  },
  "duration_s": 159.61,
  "attempts": 1,
  "pytest_log": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_videoamme_ci/run5.log",
  "basetemp": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_videoamme_ci/basetemp_run5",
  "pytest_rc": 1
}
```

### videoamme_talker_accuracy

#### run 1
```json
{
  "status": "ok",
  "reason": "threshold_assertion (exit 1)",
  "metrics": {
    "accuracy": 0.65
  },
  "sample_counts": {
    "total": 20,
    "ok": 20
  },
  "duration_s": 159.55,
  "attempts": 4,
  "pytest_log": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_videoamme_talker_ci/run1.log",
  "basetemp": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_videoamme_talker_ci/basetemp_run1",
  "pytest_rc": 1
}
```

#### run 2
```json
{
  "status": "ok",
  "reason": "",
  "metrics": {
    "accuracy": 0.65
  },
  "sample_counts": {
    "total": 20,
    "ok": 20
  },
  "duration_s": 159.54,
  "attempts": 3,
  "pytest_log": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_videoamme_talker_ci/run2.log",
  "basetemp": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_videoamme_talker_ci/basetemp_run2",
  "pytest_rc": 0
}
```

#### run 3
```json
{
  "status": "ok",
  "reason": "threshold_assertion (OOM)",
  "metrics": {
    "accuracy": 0.6
  },
  "sample_counts": {
    "total": 20,
    "ok": 20
  },
  "duration_s": 159.52,
  "attempts": 4,
  "pytest_log": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_videoamme_talker_ci/run3.log",
  "basetemp": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_videoamme_talker_ci/basetemp_run3",
  "pytest_rc": 1
}
```

#### run 4
```json
{
  "status": "ok",
  "reason": "threshold_assertion (exit 1)",
  "metrics": {
    "accuracy": 0.6
  },
  "sample_counts": {
    "total": 20,
    "ok": 20
  },
  "duration_s": 159.53,
  "attempts": 2,
  "pytest_log": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_videoamme_talker_ci/run4.log",
  "basetemp": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_videoamme_talker_ci/basetemp_run4",
  "pytest_rc": 1
}
```

#### run 5
```json
{
  "status": "ok",
  "reason": "",
  "metrics": {
    "accuracy": 0.65
  },
  "sample_counts": {
    "total": 20,
    "ok": 20
  },
  "duration_s": 159.51,
  "attempts": 3,
  "pytest_log": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_videoamme_talker_ci/run5.log",
  "basetemp": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_videoamme_talker_ci/basetemp_run5",
  "pytest_rc": 0
}
```

### videoamme_talker_wer

#### run 1
```json
{
  "status": "ok",
  "reason": "threshold_assertion (exit 1)",
  "metrics": {
    "wer_below_50_corpus": 0.013831258644536652,
    "n_above_50": 1.0
  },
  "sample_counts": {
    "total": 20,
    "ok": 20
  },
  "duration_s": 159.55,
  "attempts": 4,
  "pytest_log": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_videoamme_talker_ci/run1.log",
  "basetemp": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_videoamme_talker_ci/basetemp_run1",
  "pytest_rc": 1
}
```

#### run 2
```json
{
  "status": "ok",
  "reason": "",
  "metrics": {
    "wer_below_50_corpus": 0.004379562043795621,
    "n_above_50": 1.0
  },
  "sample_counts": {
    "total": 20,
    "ok": 20
  },
  "duration_s": 159.54,
  "attempts": 3,
  "pytest_log": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_videoamme_talker_ci/run2.log",
  "basetemp": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_videoamme_talker_ci/basetemp_run2",
  "pytest_rc": 0
}
```

#### run 3
```json
{
  "status": "ok",
  "reason": "threshold_assertion (OOM)",
  "metrics": {
    "wer_below_50_corpus": 0.0,
    "n_above_50": 0.0
  },
  "sample_counts": {
    "total": 20,
    "ok": 20
  },
  "duration_s": 159.52,
  "attempts": 4,
  "pytest_log": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_videoamme_talker_ci/run3.log",
  "basetemp": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_videoamme_talker_ci/basetemp_run3",
  "pytest_rc": 1
}
```

#### run 4
```json
{
  "status": "ok",
  "reason": "threshold_assertion (exit 1)",
  "metrics": {
    "wer_below_50_corpus": 0.005376344086021506,
    "n_above_50": 1.0
  },
  "sample_counts": {
    "total": 20,
    "ok": 20
  },
  "duration_s": 159.53,
  "attempts": 2,
  "pytest_log": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_videoamme_talker_ci/run4.log",
  "basetemp": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_videoamme_talker_ci/basetemp_run4",
  "pytest_rc": 1
}
```

#### run 5
```json
{
  "status": "ok",
  "reason": "",
  "metrics": {
    "wer_below_50_corpus": 0.008823529411764706,
    "n_above_50": 1.0
  },
  "sample_counts": {
    "total": 20,
    "ok": 20
  },
  "duration_s": 159.51,
  "attempts": 3,
  "pytest_log": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_videoamme_talker_ci/run5.log",
  "basetemp": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_videoamme_talker_ci/basetemp_run5",
  "pytest_rc": 0
}
```

### videoamme_talker_speed

#### run 1
```json
{
  "status": "ok",
  "reason": "threshold_assertion (exit 1)",
  "metrics": {
    "throughput_qps": 0.637,
    "output_tok_per_req_s": 2.1,
    "latency_mean_s": 21.298,
    "rtf_mean": 3.0362
  },
  "sample_counts": {
    "total": 20,
    "ok": 20
  },
  "duration_s": 159.55,
  "attempts": 4,
  "pytest_log": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_videoamme_talker_ci/run1.log",
  "basetemp": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_videoamme_talker_ci/basetemp_run1",
  "pytest_rc": 1
}
```

#### run 2
```json
{
  "status": "ok",
  "reason": "",
  "metrics": {
    "throughput_qps": 0.709,
    "output_tok_per_req_s": 2.2,
    "latency_mean_s": 19.27,
    "rtf_mean": 2.5422
  },
  "sample_counts": {
    "total": 20,
    "ok": 20
  },
  "duration_s": 159.54,
  "attempts": 3,
  "pytest_log": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_videoamme_talker_ci/run2.log",
  "basetemp": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_videoamme_talker_ci/basetemp_run2",
  "pytest_rc": 0
}
```

#### run 3
```json
{
  "status": "ok",
  "reason": "threshold_assertion (OOM)",
  "metrics": {
    "throughput_qps": 0.606,
    "output_tok_per_req_s": 2.1,
    "latency_mean_s": 22.427,
    "rtf_mean": 3.3211
  },
  "sample_counts": {
    "total": 20,
    "ok": 20
  },
  "duration_s": 159.52,
  "attempts": 4,
  "pytest_log": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_videoamme_talker_ci/run3.log",
  "basetemp": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_videoamme_talker_ci/basetemp_run3",
  "pytest_rc": 1
}
```

#### run 4
```json
{
  "status": "ok",
  "reason": "threshold_assertion (exit 1)",
  "metrics": {
    "throughput_qps": 0.633,
    "output_tok_per_req_s": 2.1,
    "latency_mean_s": 21.585,
    "rtf_mean": 3.1129
  },
  "sample_counts": {
    "total": 20,
    "ok": 20
  },
  "duration_s": 159.53,
  "attempts": 2,
  "pytest_log": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_videoamme_talker_ci/run4.log",
  "basetemp": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_videoamme_talker_ci/basetemp_run4",
  "pytest_rc": 1
}
```

#### run 5
```json
{
  "status": "ok",
  "reason": "",
  "metrics": {
    "throughput_qps": 0.705,
    "output_tok_per_req_s": 2.2,
    "latency_mean_s": 19.689,
    "rtf_mean": 2.2201
  },
  "sample_counts": {
    "total": 20,
    "ok": 20
  },
  "duration_s": 159.51,
  "attempts": 3,
  "pytest_log": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_videoamme_talker_ci/run5.log",
  "basetemp": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_videoamme_talker_ci/basetemp_run5",
  "pytest_rc": 0
}
```

### videoamme_talker_tp2_accuracy

#### run 1
```json
{
  "status": "ok",
  "reason": "threshold_assertion (exit 1)",
  "metrics": {
    "accuracy": 0.5
  },
  "sample_counts": {
    "total": 10,
    "ok": 10
  },
  "duration_s": 514.08,
  "attempts": 1,
  "pytest_log": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_videoamme_talker_tp2_ci/run1.log",
  "basetemp": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_videoamme_talker_tp2_ci/basetemp_run1",
  "pytest_rc": 1
}
```

#### run 2
```json
{
  "status": "ok",
  "reason": "",
  "metrics": {
    "accuracy": 0.6
  },
  "sample_counts": {
    "total": 10,
    "ok": 10
  },
  "duration_s": 243.6,
  "attempts": 1,
  "pytest_log": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_videoamme_talker_tp2_ci/run2.log",
  "basetemp": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_videoamme_talker_tp2_ci/basetemp_run2",
  "pytest_rc": 0
}
```

#### run 3
```json
{
  "status": "ok",
  "reason": "",
  "metrics": {
    "accuracy": 0.5
  },
  "sample_counts": {
    "total": 10,
    "ok": 10
  },
  "duration_s": 273.66,
  "attempts": 1,
  "pytest_log": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_videoamme_talker_tp2_ci/run3.log",
  "basetemp": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_videoamme_talker_tp2_ci/basetemp_run3",
  "pytest_rc": 0
}
```

#### run 4
```json
{
  "status": "ok",
  "reason": "",
  "metrics": {
    "accuracy": 0.5
  },
  "sample_counts": {
    "total": 10,
    "ok": 10
  },
  "duration_s": 243.63,
  "attempts": 2,
  "pytest_log": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_videoamme_talker_tp2_ci/run4.log",
  "basetemp": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_videoamme_talker_tp2_ci/basetemp_run4",
  "pytest_rc": 0
}
```

#### run 5
```json
{
  "status": "ok",
  "reason": "",
  "metrics": {
    "accuracy": 0.5
  },
  "sample_counts": {
    "total": 10,
    "ok": 10
  },
  "duration_s": 243.66,
  "attempts": 1,
  "pytest_log": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_videoamme_talker_tp2_ci/run5.log",
  "basetemp": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_videoamme_talker_tp2_ci/basetemp_run5",
  "pytest_rc": 0
}
```

### videoamme_talker_tp2_wer

#### run 1
```json
{
  "status": "ok",
  "reason": "threshold_assertion (exit 1)",
  "metrics": {
    "wer_below_50_corpus": 0.0084985835694051,
    "n_above_50": 1.0
  },
  "sample_counts": {
    "total": 10,
    "ok": 10
  },
  "duration_s": 514.08,
  "attempts": 1,
  "pytest_log": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_videoamme_talker_tp2_ci/run1.log",
  "basetemp": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_videoamme_talker_tp2_ci/basetemp_run1",
  "pytest_rc": 1
}
```

#### run 2
```json
{
  "status": "ok",
  "reason": "",
  "metrics": {
    "wer_below_50_corpus": 0.008333333333333333,
    "n_above_50": 1.0
  },
  "sample_counts": {
    "total": 10,
    "ok": 10
  },
  "duration_s": 243.6,
  "attempts": 1,
  "pytest_log": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_videoamme_talker_tp2_ci/run2.log",
  "basetemp": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_videoamme_talker_tp2_ci/basetemp_run2",
  "pytest_rc": 0
}
```

#### run 3
```json
{
  "status": "ok",
  "reason": "",
  "metrics": {
    "wer_below_50_corpus": 0.0056657223796034,
    "n_above_50": 1.0
  },
  "sample_counts": {
    "total": 10,
    "ok": 10
  },
  "duration_s": 273.66,
  "attempts": 1,
  "pytest_log": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_videoamme_talker_tp2_ci/run3.log",
  "basetemp": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_videoamme_talker_tp2_ci/basetemp_run3",
  "pytest_rc": 0
}
```

#### run 4
```json
{
  "status": "ok",
  "reason": "",
  "metrics": {
    "wer_below_50_corpus": 0.007352941176470588,
    "n_above_50": 0.0
  },
  "sample_counts": {
    "total": 10,
    "ok": 10
  },
  "duration_s": 243.63,
  "attempts": 2,
  "pytest_log": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_videoamme_talker_tp2_ci/run4.log",
  "basetemp": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_videoamme_talker_tp2_ci/basetemp_run4",
  "pytest_rc": 0
}
```

#### run 5
```json
{
  "status": "ok",
  "reason": "",
  "metrics": {
    "wer_below_50_corpus": 0.0029940119760479044,
    "n_above_50": 1.0
  },
  "sample_counts": {
    "total": 10,
    "ok": 10
  },
  "duration_s": 243.66,
  "attempts": 1,
  "pytest_log": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_videoamme_talker_tp2_ci/run5.log",
  "basetemp": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_videoamme_talker_tp2_ci/basetemp_run5",
  "pytest_rc": 0
}
```

### videoamme_talker_tp2_speed

#### run 1
```json
{
  "status": "ok",
  "reason": "threshold_assertion (exit 1)",
  "metrics": {
    "throughput_qps": 0.05,
    "output_tok_per_req_s": 0.3,
    "latency_mean_s": 137.945,
    "rtf_mean": 16.9597
  },
  "sample_counts": {
    "total": 10,
    "ok": 10
  },
  "duration_s": 514.08,
  "attempts": 1,
  "pytest_log": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_videoamme_talker_tp2_ci/run1.log",
  "basetemp": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_videoamme_talker_tp2_ci/basetemp_run1",
  "pytest_rc": 1
}
```

#### run 2
```json
{
  "status": "ok",
  "reason": "",
  "metrics": {
    "throughput_qps": 0.071,
    "output_tok_per_req_s": 0.4,
    "latency_mean_s": 97.814,
    "rtf_mean": 19.082
  },
  "sample_counts": {
    "total": 10,
    "ok": 10
  },
  "duration_s": 243.6,
  "attempts": 1,
  "pytest_log": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_videoamme_talker_tp2_ci/run2.log",
  "basetemp": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_videoamme_talker_tp2_ci/basetemp_run2",
  "pytest_rc": 0
}
```

#### run 3
```json
{
  "status": "ok",
  "reason": "",
  "metrics": {
    "throughput_qps": 0.063,
    "output_tok_per_req_s": 0.4,
    "latency_mean_s": 112.602,
    "rtf_mean": 22.4697
  },
  "sample_counts": {
    "total": 10,
    "ok": 10
  },
  "duration_s": 273.66,
  "attempts": 1,
  "pytest_log": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_videoamme_talker_tp2_ci/run3.log",
  "basetemp": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_videoamme_talker_tp2_ci/basetemp_run3",
  "pytest_rc": 0
}
```

#### run 4
```json
{
  "status": "ok",
  "reason": "",
  "metrics": {
    "throughput_qps": 0.069,
    "output_tok_per_req_s": 0.5,
    "latency_mean_s": 99.543,
    "rtf_mean": 6.881
  },
  "sample_counts": {
    "total": 10,
    "ok": 10
  },
  "duration_s": 243.63,
  "attempts": 2,
  "pytest_log": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_videoamme_talker_tp2_ci/run4.log",
  "basetemp": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_videoamme_talker_tp2_ci/basetemp_run4",
  "pytest_rc": 0
}
```

#### run 5
```json
{
  "status": "ok",
  "reason": "",
  "metrics": {
    "throughput_qps": 0.07,
    "output_tok_per_req_s": 0.4,
    "latency_mean_s": 98.256,
    "rtf_mean": 19.6497
  },
  "sample_counts": {
    "total": 10,
    "ok": 10
  },
  "duration_s": 243.66,
  "attempts": 1,
  "pytest_log": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_videoamme_talker_tp2_ci/run5.log",
  "basetemp": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_videoamme_talker_tp2_ci/basetemp_run5",
  "pytest_rc": 0
}
```

### videomme_accuracy

#### run 1
```json
{
  "status": "ok",
  "reason": "",
  "metrics": {
    "accuracy": 0.58
  },
  "sample_counts": {
    "total": 50,
    "ok": 50
  },
  "duration_s": 159.55,
  "attempts": 1,
  "pytest_log": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_videomme_ci/run1.log",
  "basetemp": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_videomme_ci/basetemp_run1",
  "pytest_rc": 0
}
```

#### run 2
```json
{
  "status": "ok",
  "reason": "",
  "metrics": {
    "accuracy": 0.62
  },
  "sample_counts": {
    "total": 50,
    "ok": 50
  },
  "duration_s": 159.44,
  "attempts": 1,
  "pytest_log": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_videomme_ci/run2.log",
  "basetemp": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_videomme_ci/basetemp_run2",
  "pytest_rc": 0
}
```

#### run 3
```json
{
  "status": "ok",
  "reason": "",
  "metrics": {
    "accuracy": 0.54
  },
  "sample_counts": {
    "total": 50,
    "ok": 50
  },
  "duration_s": 189.51,
  "attempts": 1,
  "pytest_log": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_videomme_ci/run3.log",
  "basetemp": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_videomme_ci/basetemp_run3",
  "pytest_rc": 0
}
```

#### run 4
```json
{
  "status": "ok",
  "reason": "",
  "metrics": {
    "accuracy": 0.52
  },
  "sample_counts": {
    "total": 50,
    "ok": 50
  },
  "duration_s": 159.53,
  "attempts": 1,
  "pytest_log": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_videomme_ci/run4.log",
  "basetemp": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_videomme_ci/basetemp_run4",
  "pytest_rc": 0
}
```

#### run 5
```json
{
  "status": "ok",
  "reason": "",
  "metrics": {
    "accuracy": 0.56
  },
  "sample_counts": {
    "total": 50,
    "ok": 50
  },
  "duration_s": 159.58,
  "attempts": 1,
  "pytest_log": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_videomme_ci/run5.log",
  "basetemp": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_videomme_ci/basetemp_run5",
  "pytest_rc": 0
}
```

### videomme_speed

#### run 1
```json
{
  "status": "ok",
  "reason": "",
  "metrics": {
    "throughput_qps": 0.975,
    "output_tok_per_req_s": 7.8,
    "latency_mean_s": 15.147
  },
  "sample_counts": {
    "total": 50,
    "ok": 50
  },
  "duration_s": 159.55,
  "attempts": 1,
  "pytest_log": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_videomme_ci/run1.log",
  "basetemp": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_videomme_ci/basetemp_run1",
  "pytest_rc": 0
}
```

#### run 2
```json
{
  "status": "ok",
  "reason": "",
  "metrics": {
    "throughput_qps": 1.029,
    "output_tok_per_req_s": 7.4,
    "latency_mean_s": 14.614
  },
  "sample_counts": {
    "total": 50,
    "ok": 50
  },
  "duration_s": 159.44,
  "attempts": 1,
  "pytest_log": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_videomme_ci/run2.log",
  "basetemp": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_videomme_ci/basetemp_run2",
  "pytest_rc": 0
}
```

#### run 3
```json
{
  "status": "ok",
  "reason": "",
  "metrics": {
    "throughput_qps": 1.005,
    "output_tok_per_req_s": 8.4,
    "latency_mean_s": 14.642
  },
  "sample_counts": {
    "total": 50,
    "ok": 50
  },
  "duration_s": 189.51,
  "attempts": 1,
  "pytest_log": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_videomme_ci/run3.log",
  "basetemp": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_videomme_ci/basetemp_run3",
  "pytest_rc": 0
}
```

#### run 4
```json
{
  "status": "ok",
  "reason": "",
  "metrics": {
    "throughput_qps": 0.983,
    "output_tok_per_req_s": 7.6,
    "latency_mean_s": 15.434
  },
  "sample_counts": {
    "total": 50,
    "ok": 50
  },
  "duration_s": 159.53,
  "attempts": 1,
  "pytest_log": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_videomme_ci/run4.log",
  "basetemp": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_videomme_ci/basetemp_run4",
  "pytest_rc": 0
}
```

#### run 5
```json
{
  "status": "ok",
  "reason": "",
  "metrics": {
    "throughput_qps": 0.928,
    "output_tok_per_req_s": 7.2,
    "latency_mean_s": 16.306
  },
  "sample_counts": {
    "total": 50,
    "ok": 50
  },
  "duration_s": 159.58,
  "attempts": 1,
  "pytest_log": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_videomme_ci/run5.log",
  "basetemp": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_videomme_ci/basetemp_run5",
  "pytest_rc": 0
}
```

### videomme_talker_accuracy

#### run 1
```json
{
  "status": "ok",
  "reason": "",
  "metrics": {
    "accuracy": 0.65
  },
  "sample_counts": {
    "total": 20,
    "ok": 20
  },
  "duration_s": 159.54,
  "attempts": 1,
  "pytest_log": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_videomme_talker_ci/run1.log",
  "basetemp": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_videomme_talker_ci/basetemp_run1",
  "pytest_rc": 0
}
```

#### run 2
```json
{
  "status": "ok",
  "reason": "",
  "metrics": {
    "accuracy": 0.6
  },
  "sample_counts": {
    "total": 20,
    "ok": 20
  },
  "duration_s": 189.68,
  "attempts": 1,
  "pytest_log": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_videomme_talker_ci/run2.log",
  "basetemp": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_videomme_talker_ci/basetemp_run2",
  "pytest_rc": 0
}
```

#### run 3
```json
{
  "status": "ok",
  "reason": "",
  "metrics": {
    "accuracy": 0.6
  },
  "sample_counts": {
    "total": 20,
    "ok": 20
  },
  "duration_s": 189.54,
  "attempts": 1,
  "pytest_log": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_videomme_talker_ci/run3.log",
  "basetemp": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_videomme_talker_ci/basetemp_run3",
  "pytest_rc": 0
}
```

#### run 4
```json
{
  "status": "ok",
  "reason": "",
  "metrics": {
    "accuracy": 0.6
  },
  "sample_counts": {
    "total": 20,
    "ok": 20
  },
  "duration_s": 159.44,
  "attempts": 1,
  "pytest_log": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_videomme_talker_ci/run4.log",
  "basetemp": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_videomme_talker_ci/basetemp_run4",
  "pytest_rc": 0
}
```

#### run 5
```json
{
  "status": "ok",
  "reason": "",
  "metrics": {
    "accuracy": 0.6
  },
  "sample_counts": {
    "total": 20,
    "ok": 20
  },
  "duration_s": 159.54,
  "attempts": 1,
  "pytest_log": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_videomme_talker_ci/run5.log",
  "basetemp": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_videomme_talker_ci/basetemp_run5",
  "pytest_rc": 0
}
```

### videomme_talker_wer

#### run 1
```json
{
  "status": "ok",
  "reason": "",
  "metrics": {
    "wer_below_50_corpus": 0.012622720897615708,
    "n_above_50": 0.0
  },
  "sample_counts": {
    "total": 20,
    "ok": 20
  },
  "duration_s": 159.54,
  "attempts": 1,
  "pytest_log": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_videomme_talker_ci/run1.log",
  "basetemp": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_videomme_talker_ci/basetemp_run1",
  "pytest_rc": 0
}
```

#### run 2
```json
{
  "status": "ok",
  "reason": "",
  "metrics": {
    "wer_below_50_corpus": 0.018922852983988356,
    "n_above_50": 0.0
  },
  "sample_counts": {
    "total": 20,
    "ok": 20
  },
  "duration_s": 189.68,
  "attempts": 1,
  "pytest_log": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_videomme_talker_ci/run2.log",
  "basetemp": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_videomme_talker_ci/basetemp_run2",
  "pytest_rc": 0
}
```

#### run 3
```json
{
  "status": "ok",
  "reason": "",
  "metrics": {
    "wer_below_50_corpus": 0.014285714285714285,
    "n_above_50": 0.0
  },
  "sample_counts": {
    "total": 20,
    "ok": 20
  },
  "duration_s": 189.54,
  "attempts": 1,
  "pytest_log": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_videomme_talker_ci/run3.log",
  "basetemp": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_videomme_talker_ci/basetemp_run3",
  "pytest_rc": 0
}
```

#### run 4
```json
{
  "status": "ok",
  "reason": "",
  "metrics": {
    "wer_below_50_corpus": 0.023876404494382022,
    "n_above_50": 0.0
  },
  "sample_counts": {
    "total": 20,
    "ok": 20
  },
  "duration_s": 159.44,
  "attempts": 1,
  "pytest_log": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_videomme_talker_ci/run4.log",
  "basetemp": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_videomme_talker_ci/basetemp_run4",
  "pytest_rc": 0
}
```

#### run 5
```json
{
  "status": "ok",
  "reason": "",
  "metrics": {
    "wer_below_50_corpus": 0.02027027027027027,
    "n_above_50": 0.0
  },
  "sample_counts": {
    "total": 20,
    "ok": 20
  },
  "duration_s": 159.54,
  "attempts": 1,
  "pytest_log": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_videomme_talker_ci/run5.log",
  "basetemp": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_videomme_talker_ci/basetemp_run5",
  "pytest_rc": 0
}
```

### videomme_talker_speed

#### run 1
```json
{
  "status": "ok",
  "reason": "",
  "metrics": {
    "throughput_qps": 0.624,
    "output_tok_per_req_s": 2.2,
    "latency_mean_s": 20.226,
    "rtf_mean": 2.066
  },
  "sample_counts": {
    "total": 20,
    "ok": 20
  },
  "duration_s": 159.54,
  "attempts": 1,
  "pytest_log": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_videomme_talker_ci/run1.log",
  "basetemp": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_videomme_talker_ci/basetemp_run1",
  "pytest_rc": 0
}
```

#### run 2
```json
{
  "status": "ok",
  "reason": "",
  "metrics": {
    "throughput_qps": 0.603,
    "output_tok_per_req_s": 2.2,
    "latency_mean_s": 20.15,
    "rtf_mean": 2.1134
  },
  "sample_counts": {
    "total": 20,
    "ok": 20
  },
  "duration_s": 189.68,
  "attempts": 1,
  "pytest_log": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_videomme_talker_ci/run2.log",
  "basetemp": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_videomme_talker_ci/basetemp_run2",
  "pytest_rc": 0
}
```

#### run 3
```json
{
  "status": "ok",
  "reason": "",
  "metrics": {
    "throughput_qps": 0.612,
    "output_tok_per_req_s": 2.2,
    "latency_mean_s": 19.792,
    "rtf_mean": 2.0204
  },
  "sample_counts": {
    "total": 20,
    "ok": 20
  },
  "duration_s": 189.54,
  "attempts": 1,
  "pytest_log": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_videomme_talker_ci/run3.log",
  "basetemp": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_videomme_talker_ci/basetemp_run3",
  "pytest_rc": 0
}
```

#### run 4
```json
{
  "status": "ok",
  "reason": "",
  "metrics": {
    "throughput_qps": 0.635,
    "output_tok_per_req_s": 2.2,
    "latency_mean_s": 20.301,
    "rtf_mean": 2.0087
  },
  "sample_counts": {
    "total": 20,
    "ok": 20
  },
  "duration_s": 159.44,
  "attempts": 1,
  "pytest_log": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_videomme_talker_ci/run4.log",
  "basetemp": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_videomme_talker_ci/basetemp_run4",
  "pytest_rc": 0
}
```

#### run 5
```json
{
  "status": "ok",
  "reason": "",
  "metrics": {
    "throughput_qps": 0.608,
    "output_tok_per_req_s": 2.3,
    "latency_mean_s": 20.486,
    "rtf_mean": 1.9935
  },
  "sample_counts": {
    "total": 20,
    "ok": 20
  },
  "duration_s": 159.54,
  "attempts": 1,
  "pytest_log": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_videomme_talker_ci/run5.log",
  "basetemp": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_qwen3-omni-v1_all_r5/_pytest/test_qwen3_omni_videomme_talker_ci/basetemp_run5",
  "pytest_rc": 0
}
```

## s2-pro-v1
Run dir: `.tune-runs/20260524T221718Z_s2-pro-v1_all_r5`

### tts_nonstream_wer

#### run 1
```json
{
  "status": "ok",
  "reason": "",
  "metrics": {
    "corpus_wer": 0.008865248226950355,
    "per_sample_wer_max": 0.14285714285714285
  },
  "sample_counts": {
    "total": 50,
    "ok": 50
  },
  "duration_s": 309.91,
  "attempts": 1,
  "pytest_log": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_s2-pro-v1_all_r5/_pytest/test_s2pro_tts_ci/run1.log",
  "basetemp": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_s2-pro-v1_all_r5/_pytest/test_s2pro_tts_ci/basetemp_run1",
  "pytest_rc": 0
}
```

#### run 2
```json
{
  "status": "ok",
  "reason": "",
  "metrics": {
    "corpus_wer": 0.008865248226950355,
    "per_sample_wer_max": 0.14285714285714285
  },
  "sample_counts": {
    "total": 50,
    "ok": 50
  },
  "duration_s": 219.61,
  "attempts": 1,
  "pytest_log": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_s2-pro-v1_all_r5/_pytest/test_s2pro_tts_ci/run2.log",
  "basetemp": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_s2-pro-v1_all_r5/_pytest/test_s2pro_tts_ci/basetemp_run2",
  "pytest_rc": 0
}
```

#### run 3
```json
{
  "status": "ok",
  "reason": "threshold_assertion (exit 1)",
  "metrics": {
    "corpus_wer": 0.010638297872340425,
    "per_sample_wer_max": 0.14285714285714285
  },
  "sample_counts": {
    "total": 50,
    "ok": 50
  },
  "duration_s": 219.56,
  "attempts": 1,
  "pytest_log": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_s2-pro-v1_all_r5/_pytest/test_s2pro_tts_ci/run3.log",
  "basetemp": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_s2-pro-v1_all_r5/_pytest/test_s2pro_tts_ci/basetemp_run3",
  "pytest_rc": 1
}
```

#### run 4
```json
{
  "status": "ok",
  "reason": "",
  "metrics": {
    "corpus_wer": 0.010638297872340425,
    "per_sample_wer_max": 0.16666666666666666
  },
  "sample_counts": {
    "total": 50,
    "ok": 50
  },
  "duration_s": 219.69,
  "attempts": 1,
  "pytest_log": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_s2-pro-v1_all_r5/_pytest/test_s2pro_tts_ci/run4.log",
  "basetemp": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_s2-pro-v1_all_r5/_pytest/test_s2pro_tts_ci/basetemp_run4",
  "pytest_rc": 0
}
```

#### run 5
```json
{
  "status": "ok",
  "reason": "threshold_assertion (exit 1)",
  "metrics": {
    "corpus_wer": 0.008865248226950355,
    "per_sample_wer_max": 0.14285714285714285
  },
  "sample_counts": {
    "total": 50,
    "ok": 50
  },
  "duration_s": 219.64,
  "attempts": 1,
  "pytest_log": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_s2-pro-v1_all_r5/_pytest/test_s2pro_tts_ci/run5.log",
  "basetemp": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_s2-pro-v1_all_r5/_pytest/test_s2pro_tts_ci/basetemp_run5",
  "pytest_rc": 1
}
```

### tts_nonstream_speed

#### run 1
```json
{
  "status": "ok",
  "reason": "",
  "metrics": {
    "throughput_qps": 1.465,
    "output_tok_per_req_s": 66.9,
    "latency_mean_s": 9.757,
    "rtf_mean": 3.0377
  },
  "sample_counts": {
    "total": 50,
    "ok": 50
  },
  "duration_s": 309.91,
  "attempts": 1,
  "pytest_log": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_s2-pro-v1_all_r5/_pytest/test_s2pro_tts_ci/run1.log",
  "basetemp": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_s2-pro-v1_all_r5/_pytest/test_s2pro_tts_ci/basetemp_run1",
  "pytest_rc": 0
}
```

#### run 2
```json
{
  "status": "ok",
  "reason": "",
  "metrics": {
    "throughput_qps": 1.528,
    "output_tok_per_req_s": 68.0,
    "latency_mean_s": 9.269,
    "rtf_mean": 2.8495
  },
  "sample_counts": {
    "total": 50,
    "ok": 50
  },
  "duration_s": 219.61,
  "attempts": 1,
  "pytest_log": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_s2-pro-v1_all_r5/_pytest/test_s2pro_tts_ci/run2.log",
  "basetemp": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_s2-pro-v1_all_r5/_pytest/test_s2pro_tts_ci/basetemp_run2",
  "pytest_rc": 0
}
```

#### run 3
```json
{
  "status": "ok",
  "reason": "threshold_assertion (exit 1)",
  "metrics": {
    "throughput_qps": 1.503,
    "output_tok_per_req_s": 67.4,
    "latency_mean_s": 9.405,
    "rtf_mean": 2.8404
  },
  "sample_counts": {
    "total": 50,
    "ok": 50
  },
  "duration_s": 219.56,
  "attempts": 1,
  "pytest_log": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_s2-pro-v1_all_r5/_pytest/test_s2pro_tts_ci/run3.log",
  "basetemp": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_s2-pro-v1_all_r5/_pytest/test_s2pro_tts_ci/basetemp_run3",
  "pytest_rc": 1
}
```

#### run 4
```json
{
  "status": "ok",
  "reason": "",
  "metrics": {
    "throughput_qps": 1.52,
    "output_tok_per_req_s": 68.0,
    "latency_mean_s": 9.321,
    "rtf_mean": 2.8593
  },
  "sample_counts": {
    "total": 50,
    "ok": 50
  },
  "duration_s": 219.69,
  "attempts": 1,
  "pytest_log": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_s2-pro-v1_all_r5/_pytest/test_s2pro_tts_ci/run4.log",
  "basetemp": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_s2-pro-v1_all_r5/_pytest/test_s2pro_tts_ci/basetemp_run4",
  "pytest_rc": 0
}
```

#### run 5
```json
{
  "status": "ok",
  "reason": "threshold_assertion (exit 1)",
  "metrics": {
    "throughput_qps": 1.517,
    "output_tok_per_req_s": 69.0,
    "latency_mean_s": 9.327,
    "rtf_mean": 2.8189
  },
  "sample_counts": {
    "total": 50,
    "ok": 50
  },
  "duration_s": 219.64,
  "attempts": 1,
  "pytest_log": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_s2-pro-v1_all_r5/_pytest/test_s2pro_tts_ci/run5.log",
  "basetemp": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_s2-pro-v1_all_r5/_pytest/test_s2pro_tts_ci/basetemp_run5",
  "pytest_rc": 1
}
```

### tts_stream_wer

#### run 1
```json
{
  "status": "ok",
  "reason": "",
  "metrics": {
    "corpus_wer": 0.010610079575596816,
    "per_sample_wer_max": 0.14285714285714285
  },
  "sample_counts": {
    "total": 32,
    "ok": 32
  },
  "duration_s": 309.91,
  "attempts": 1,
  "pytest_log": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_s2-pro-v1_all_r5/_pytest/test_s2pro_tts_ci/run1.log",
  "basetemp": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_s2-pro-v1_all_r5/_pytest/test_s2pro_tts_ci/basetemp_run1",
  "pytest_rc": 0
}
```

#### run 2
```json
{
  "status": "ok",
  "reason": "",
  "metrics": {
    "corpus_wer": 0.010610079575596816,
    "per_sample_wer_max": 0.14285714285714285
  },
  "sample_counts": {
    "total": 32,
    "ok": 32
  },
  "duration_s": 219.61,
  "attempts": 1,
  "pytest_log": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_s2-pro-v1_all_r5/_pytest/test_s2pro_tts_ci/run2.log",
  "basetemp": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_s2-pro-v1_all_r5/_pytest/test_s2pro_tts_ci/basetemp_run2",
  "pytest_rc": 0
}
```

#### run 3
```json
{
  "status": "ok",
  "reason": "threshold_assertion (exit 1)",
  "metrics": {
    "corpus_wer": 0.013262599469496022,
    "per_sample_wer_max": 0.16666666666666666
  },
  "sample_counts": {
    "total": 32,
    "ok": 32
  },
  "duration_s": 219.56,
  "attempts": 1,
  "pytest_log": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_s2-pro-v1_all_r5/_pytest/test_s2pro_tts_ci/run3.log",
  "basetemp": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_s2-pro-v1_all_r5/_pytest/test_s2pro_tts_ci/basetemp_run3",
  "pytest_rc": 1
}
```

#### run 4
```json
{
  "status": "ok",
  "reason": "",
  "metrics": {
    "corpus_wer": 0.010610079575596816,
    "per_sample_wer_max": 0.14285714285714285
  },
  "sample_counts": {
    "total": 32,
    "ok": 32
  },
  "duration_s": 219.69,
  "attempts": 1,
  "pytest_log": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_s2-pro-v1_all_r5/_pytest/test_s2pro_tts_ci/run4.log",
  "basetemp": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_s2-pro-v1_all_r5/_pytest/test_s2pro_tts_ci/basetemp_run4",
  "pytest_rc": 0
}
```

#### run 5
```json
{
  "status": "ok",
  "reason": "threshold_assertion (exit 1)",
  "metrics": {
    "corpus_wer": 0.013262599469496022,
    "per_sample_wer_max": 0.16666666666666666
  },
  "sample_counts": {
    "total": 32,
    "ok": 32
  },
  "duration_s": 219.64,
  "attempts": 1,
  "pytest_log": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_s2-pro-v1_all_r5/_pytest/test_s2pro_tts_ci/run5.log",
  "basetemp": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_s2-pro-v1_all_r5/_pytest/test_s2pro_tts_ci/basetemp_run5",
  "pytest_rc": 1
}
```

### tts_stream_speed

#### run 1
```json
{
  "status": "ok",
  "reason": "",
  "metrics": {
    "throughput_qps": 1.309,
    "output_tok_per_req_s": 60.6,
    "latency_mean_s": 10.229,
    "rtf_mean": 2.8393
  },
  "sample_counts": {
    "total": 32,
    "ok": 32
  },
  "duration_s": 309.91,
  "attempts": 1,
  "pytest_log": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_s2-pro-v1_all_r5/_pytest/test_s2pro_tts_ci/run1.log",
  "basetemp": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_s2-pro-v1_all_r5/_pytest/test_s2pro_tts_ci/basetemp_run1",
  "pytest_rc": 0
}
```

#### run 2
```json
{
  "status": "ok",
  "reason": "",
  "metrics": {
    "throughput_qps": 1.287,
    "output_tok_per_req_s": 58.7,
    "latency_mean_s": 10.035,
    "rtf_mean": 2.8508
  },
  "sample_counts": {
    "total": 32,
    "ok": 32
  },
  "duration_s": 219.61,
  "attempts": 1,
  "pytest_log": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_s2-pro-v1_all_r5/_pytest/test_s2pro_tts_ci/run2.log",
  "basetemp": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_s2-pro-v1_all_r5/_pytest/test_s2pro_tts_ci/basetemp_run2",
  "pytest_rc": 0
}
```

#### run 3
```json
{
  "status": "ok",
  "reason": "threshold_assertion (exit 1)",
  "metrics": {
    "throughput_qps": 1.371,
    "output_tok_per_req_s": 61.0,
    "latency_mean_s": 9.881,
    "rtf_mean": 2.6539
  },
  "sample_counts": {
    "total": 32,
    "ok": 32
  },
  "duration_s": 219.56,
  "attempts": 1,
  "pytest_log": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_s2-pro-v1_all_r5/_pytest/test_s2pro_tts_ci/run3.log",
  "basetemp": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_s2-pro-v1_all_r5/_pytest/test_s2pro_tts_ci/basetemp_run3",
  "pytest_rc": 1
}
```

#### run 4
```json
{
  "status": "ok",
  "reason": "",
  "metrics": {
    "throughput_qps": 1.417,
    "output_tok_per_req_s": 58.5,
    "latency_mean_s": 9.74,
    "rtf_mean": 2.7394
  },
  "sample_counts": {
    "total": 32,
    "ok": 32
  },
  "duration_s": 219.69,
  "attempts": 1,
  "pytest_log": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_s2-pro-v1_all_r5/_pytest/test_s2pro_tts_ci/run4.log",
  "basetemp": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_s2-pro-v1_all_r5/_pytest/test_s2pro_tts_ci/basetemp_run4",
  "pytest_rc": 0
}
```

#### run 5
```json
{
  "status": "ok",
  "reason": "threshold_assertion (exit 1)",
  "metrics": {
    "throughput_qps": 1.415,
    "output_tok_per_req_s": 60.0,
    "latency_mean_s": 9.826,
    "rtf_mean": 2.6914
  },
  "sample_counts": {
    "total": 32,
    "ok": 32
  },
  "duration_s": 219.64,
  "attempts": 1,
  "pytest_log": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_s2-pro-v1_all_r5/_pytest/test_s2pro_tts_ci/run5.log",
  "basetemp": "/sgl-workspace/sglang-omni/.tune-runs/20260524T221718Z_s2-pro-v1_all_r5/_pytest/test_s2pro_tts_ci/basetemp_run5",
  "pytest_rc": 1
}
```

