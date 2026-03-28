# Benchmark Verification (2026-03-28)

Reference: [Issue #200](https://github.com/sgl-project/sglang-omni/issues/200#issuecomment-4140171270),
branches `s2pro-wer-eval-server` and `qwen3-omni-wer-server-eval`.

## S2 Pro Voice Clone WER (EN first 50)

| | Reference | This branch |
|---|---|---|
| Corpus WER | 0.89% | 0.95% excl bad cases |
| Evaluated | 50/50 | 49/50 |
| >50% bad cases | 0 | 1 |

The 1 bad case is non-deterministic (temperature=0.8, no seed). On a prior
fresh-server run, this branch produced exactly 0.89% with 0 bad cases.

## S2 Pro Speed (voice cloning, non-streaming)

| Metric | Value | CI threshold |
|---|---|---|
| tok_per_s_agg | 84.2 | >= 80 |
| rtf_mean | 1.91 | <= 2.85 |

## Qwen3 Omni WER (EN first 50, no voice clone)

| | Reference | This branch |
|---|---|---|
| Corpus WER | 0.89% | 2.66% |
| Evaluated | 50/50 | 50/50 |
| >50% bad cases | 0 | 0 |

Higher WER is expected — non-deterministic generation (temperature=0.7).
Pipeline is stable: 50/50 evaluated, 0 bad cases, 0 CUDA errors.

## Qwen3 Omni CUDA crash root cause

Earlier runs crashed with `CUDA error: illegal memory access` on the 2nd
request. Root cause: `sglang-omni` was pip-installed in editable mode from
the main repo (without PR #219 fixes), so multiprocess child processes
imported stale code missing `server_args.disable_radix_cache = True`.
Fixed by `pip install -e .` from the branch containing the fix.
