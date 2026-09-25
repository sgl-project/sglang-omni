# Calibration handoff: #2293 latency stage

- Last update: 2026-09-25T01:57:05Z (09-24 18:57 PT), operator luojiaxuan (agent on the Mac), phase: driver attempt 3 launching.
- Target: sgl-project/sglang-omni #2293 at 8521cf6865686ad28b858f338e96d323caac4c60 (rebased on main 329ef4cd), not pushed yet; clean checkout.
- Focus: latency stage (test_tts_latency_ci.py), 4 stages x 5 clean observations; ASR and Omni SKIP. Scope: evidence/stage-scope.json.
- Tools: zhaochenyang20/sglang-omni-calibration branch tts-latency-stage at dda3c7c278fafcb74cd814a1b19db71130a63ad6, staged under /data/tools/calibrate-h100-ci-dda3c7c (outside the checkout), private/ excluded.
- Host: host-85-234-79-221 (CI runner host), Radix assignment 01M3B3307N6SRH0B0VR71AKXQY, GPUs 0,1, until 2026-09-25T05:32Z (09-24 22:32 PT).
  The profile in private/host.json names the retired novita-h100 layout; the lane here is held by the Radix lease instead of an autoscaler lock.
  CI jobs currently run on eval-h100, not on this host.
- Lane cpuset: 2-15,66-79 (GPU pair 0,1 per hosts/sglang-h100-ci.yaml), exported as OMNI_CI_CPUSET.
- Image: docker.io/hongccc/sglang-omni@sha256:ebe4239e29a764ee3a2806385c061c5fd438a26f01458e503d3822dcba5790df, local ID 2bac261779c8 (the target workflow's pin).
- Driver: container initial command runtime/entrypoint.sh; events in evidence/events.log; reports in reports/.
- Driver attempt 1 (01:42Z, tools 843f926) stopped before sampling: `tune.py run` gates on the repository-wide CI
  coverage, and the Omni schema was stale because sgl-project/sglang-omni#2252 moved the Omni references into
  tests/test_model/omni_ci_config.py (with MiniCPM-o as a second preset); Omni discover found no constants. No GPU
  work ran. Its events, reports and entrypoint are kept under attempts/driver01/.
- Decision: update the tools instead of narrowing the gate to the calibrated suite. Runbook 1.1 says a narrower stage
  selection does not waive the inventory check and an unsupported registry shape needs a tools update. Tools dda3c7c
  expands every Omni test per preset (qwen3-omni, minicpmo) from omni_ci_config.py; coverage passes against the target.
  Rollback: revert dda3c7c; it only touches models/omni. The entrypoint now fails on a coverage error.
- Driver attempt 2 (01:54Z, tools dda3c7c) passed coverage, then `run`'s native precheck reported the three assets as
  not cached: the host profile exports HF_HOME=/root/.cache/huggingface (the CI runner layout) while the entrypoint had
  put the cache under /github/home/.cache/huggingface. No GPU work ran; evidence kept under attempts/driver02/. The
  entrypoint now links /root/.cache/huggingface to the same cache and exports that path.
- Next action: relaunch the container (attempt 3), watch events.log and run_tts status.
