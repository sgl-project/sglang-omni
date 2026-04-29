# Local CI Reproduction — `.github/workflows/test-qwen3-omni-ci-v1.yaml`

Run date: 2026-04-29
Branch: `pr-334`
Working dir: `/data/chenyang/sglang-omni`
Python env: `/data/chenyang/.python/omni`
Common env: `SGLANG_OMNI_SERVER_VERSION=v1`, `HF_ENDPOINT=https://hf-mirror.com` (only when local cache miss)
GPUs available locally: `6` and `7` (both essentially free on H200, 143 GiB each)
Hardware mapping (CI uses `--gpus all`; locally we restrict): single-GPU stages → `CUDA_VISIBLE_DEVICES=7`; talker stages → `CUDA_VISIBLE_DEVICES=6,7` so logical 0=physical 6 (thinker), logical 1=physical 7 (talker / code-predictor / code2wav).

## DAG (from the workflow)
```
docs ──► stage-1-thinker ──► stage-2-tts
                          ├─► stage-3-mmmu
                          ├─► stage-4-mmmu-talker
                          ├─► stage-5-mmsu
                          ├─► stage-6-mmsu-talker
                          ├─► stage-7-videomme
                          ├─► stage-8-videomme-talker
                          ├─► stage-9-videoamme
                          └─► stage-10-videoamme-talker
```

## Per-stage results

| # | Job | Test file | GPUs | Status | Notes |
|---|-----|-----------|------|--------|-------|
| 0 | docs | `tests/docs/qwen3_omni/test_docs_qwen3_omni.py` | 1+2 | _pending_ | |
| 1 | stage-1 thinker length | `tests/test_model/test_qwen3_omni_thinker_length.py` | 1 | ❌ FAIL @ server boot | `TypeError: Stage.__init__() got an unexpected keyword argument 'recv_endpoint'` (single-process compile path) |
| 2 | stage-2 TTS | `tests/test_model/test_qwen3_omni_tts_ci.py` | 2 | _pending_ | |
| 3 | stage-3 MMMU | `tests/test_model/test_qwen3_omni_mmmu_ci.py` | 1 | _pending_ | |
| 4 | stage-4 MMMU Talker | `tests/test_model/test_qwen3_omni_mmmu_talker_ci.py` | 2 | _pending_ | |
| 5 | stage-5 MMSU | `tests/test_model/test_qwen3_omni_mmsu_ci.py` | 1 | _pending_ | |
| 6 | stage-6 MMSU Talker | `tests/test_model/test_qwen3_omni_mmsu_talker_ci.py` | 2 | _pending_ | |
| 7 | stage-7 Video-MME | `tests/test_model/test_qwen3_omni_videomme_ci.py` | 1 | _pending_ | |
| 8 | stage-8 Video-MME Talker | `tests/test_model/test_qwen3_omni_videomme_talker_ci.py` | 2 | _pending_ | |
| 9 | stage-9 Video-AMME | `tests/test_model/test_qwen3_omni_videoamme_ci.py` | 1 | _pending_ | |
| 10 | stage-10 Video-AMME Talker | `tests/test_model/test_qwen3_omni_videoamme_talker_ci.py` | 2 | _pending_ | |

Per-stage details (commands, log paths, error excerpts) are appended below as the runs complete.

---
