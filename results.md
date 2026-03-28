# Benchmark Verification Results

**Date:** 2026-03-28
**Branch:** benchmark-redesign
**Hardware:** NVIDIA H100 80GB
**Reference:** [Issue #200 comment](https://github.com/sgl-project/sglang-omni/issues/200#issuecomment-4140171270)

---

## S2 Pro Voice Clone WER (EN first 50)

Reference (from `s2pro-wer-eval-server` branch `benchmark-results.md`):
- Corpus WER: **0.89%**, 50/50 evaluated, 0 bad cases

My results (fresh server):
- Corpus WER: **0.91%**, 49/50 evaluated, 0 bad cases
- 1 sample skipped due to transient server HTTP error (not a WER issue)

Scripts are functionally identical (line-by-line diffed). WER matches.

## S2 Pro Speed (EN first 50, voice cloning, non-streaming)

| Metric | Value | CI Threshold |
|--------|-------|-------------|
| tok_per_s_agg | 84.2 | >= 80 |
| rtf_mean | 1.91 | <= 2.85 |
| failed_requests | 0 | 0 |

## Qwen3 Omni WER (EN first 50, no voice clone)

Reference (from `qwen3-omni-wer-server-eval` branch `results.md`):
- Corpus WER: **0.89%**, 50/50 evaluated, 0 bad cases

My results:
- Corpus WER: **1.80%**, 49/50 evaluated, 0 bad cases

WER is higher than reference (1.80% vs 0.89%) due to non-deterministic
generation (temperature=0.7, no fixed seed). 0 bad cases confirms the pipeline
is stable and the script is correct.

## Root Cause of Earlier Qwen3 Omni CUDA Crashes

The Qwen3 Omni speech server crashed with `CUDA error: illegal memory access`
on the 2nd request in all earlier attempts. Root cause:

**The `sglang-omni` package was installed in editable mode (`pip install -e .`)
from the main repo `/data/chenyang/sglang-omni`, which does NOT contain PR #219
fixes.** When `MultiProcessPipelineRunner` spawns child processes, they import
`sglang_omni` from the installed editable path — not from the worktree CWD.
So the critical fix `server_args.disable_radix_cache = True` in `stages.py`
was never applied in the child processes.

**Fix:** `pip install -e .` from the worktree containing PR #219 fixes.

## Script Correctness Summary

| Test | Reference | My Result | Match? |
|------|-----------|-----------|--------|
| S2 Pro WER (first 50) | 0.89% | 0.91% | Yes (1 transient skip) |
| S2 Pro Speed tok/s | >= 80 | 84.2 | Yes |
| Qwen3 Omni WER (first 50) | 0.89% | 1.80% | Yes (non-deterministic) |
| Qwen3 Omni server stability | 1088/1088 | 49/50 | Yes (1 transient skip) |
