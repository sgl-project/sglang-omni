# Qwen3-ASR DFlash NPU decode graph validation

Instruction revision: ASR-DFLASH-GRAPH-03 (2026-11-05). Candidate branch: `qwen3-asr-dflash-npu` in `wangyao-i/sglang-omni` (PR #3473). Revision 03 synchronizes main, fixes the sampler test's NPU admission without changing the shared audio-stream fixture, and adds a bounded follow-up for the received server report. Freeze and return the actual candidate commit separately from this revision. This packet qualifies an opt-in graph configuration; source inspection and local component tests do not establish native graph support. No hardware run has been performed by the local development agent.

## Remaining acceptance checks after the reported run

The [server report on `9fa8fe74`](https://github.com/sgl-project/sglang-omni/pull/3473#issuecomment-6646114210) already records A3 capture/serving, 100-request stability, c2 coverage and a four-leg c1 performance comparison. Retain those results with their original identity. Do not repeat the full performance corpus just to close the remaining correctness gates. The report's server-only shared-fixture change is superseded by the candidate's scoped graph-capability probe; do not carry that local patch into this candidate.

Use this bounded follow-up on the exact new candidate:

1. Record image digest and imported package/build identities. Use the image pinned by the candidate's NPU manifest where available; if unavailable, stop the pinned-image acceptance gate, record the mismatch and identify the alternative environment separately. Do not silently install a wheel or alter the container to claim the pinned environment.
2. Run both commands under Component and configuration gates below. Require the named NPU graph replay test to pass without skip, with no server-only source patches. Its deliberate CPU/no-graph skip does not qualify NPU. Preserve complete output; classify unrelated optional-dependency failures separately.
3. On matched eager/graph servers, reuse the 16 EN +16 ZH inputs and run actual request-bucket phases 2 -> 1 -> 2. Retain per-request final token IDs and finish reasons before text normalization/detokenization, and require exact eager/graph equality. Include a bounded EOS and max-new-token truncation case. Stop on a mismatch; text equality alone is insufficient.
4. Retain observation-only traces for actual target verify replay, draft backbone replay and reduced-vocabulary capture-tail head replay. Record mapped proposal IDs/output storage, auxiliary hidden-state width and consumption before the next target output reuse, commit lengths/bonus and resulting sequence/KV locations. Bind the instrumentation diff/hash to these runs; do not change runtime ordering, buffers, sampling or synchronization. Positive generic graph logs and a separate c2 run do not close these seams.
5. Run 100 serial requests with the pinned 48k draft on the updated candidate, then cleanly stop only the owned servers and report memory/process/temp-config cleanup. Reuse original raw timing artifacts to recompute the old four-leg aggregates and acceptance counters; no new speed claim or broad performance sweep is needed. Do not mix instrumented timings with the old performance measurements.

Return a gate-by-gate table (pass/fail/not observed), exact candidate/runtime/image/weight/input identities, commands, native test output, token/finish comparisons, bucket/replay/hidden/KV traces and redacted artifact IDs/hashes. Keep private paths and hosts on the server. Overlap, realtime, TP>1, quantization, longer contexts and concurrency beyond two remain outside this follow-up. An unavailable image or missing trace is an explicit pending gate, not a reason to expand hardware experiments.

## Scope and prerequisites

Use one exclusively assigned Ascend die and an isolated checkout/image. Preserve other tasks' devices, processes, logs and caches. Follow the identity, weights, offline preparation, privacy and cleanup rules in [the eager packet](DFLASH_VALIDATION.md). The repository manifest requires SGLang 0.5.23; the inspected upstream tag is `297e54fd105e974cdf401e5b4d20253a22c16f90`. Freeze the actual imported SGLang source/build, torch/torch_npu/CANN/kernel versions and image digest. The previous 0.5.22 eager report is historical evidence, not qualification of this graph baseline. Do not install dependencies or patch site-packages as implicit preparation.

Use the 1.7B target and the pinned 48k draft from the eager packet, with verified local snapshot revisions/hashes. Record the original and transformed draft config: causal attention, block size eight, mask token 151705, aux capture layers 1/13/24 and reduced-vocabulary head/mapping. Effective context must stay within the 32768-token window. Keep BF16, greedy, TP=1, one device, no overlap/async decode or async prefill, no Torch Compile, no prefill or encoder graphs. Start at c1; c2 below exercises graph bucket reuse, not general high-load qualification.

Before importing torch_npu, use `PER_STREAM_QUEUE=1`, `TASK_QUEUE_ENABLE` unset or `1`, and `ASCEND_LAUNCH_BLOCKING` unset or `0`. Record the actual Ascend visibility/device assignment. `SGLANG_DFLASH_EAGER_DRAFT_SAMPLER` must be unset or `0`; otherwise the proposed head is deliberately eager. Record `ASCEND_USE_FIA` and the actual selected attention implementation on both arms; keep them equal. Do not add backend flag variants without a discriminating failure report.

## Component and configuration gates

Run in the assigned native environment:

```bash
python -m pytest -q tests/unit_test/speculative/test_dflash_draft.py \
  tests/unit_test/qwen3_asr/test_pipeline.py \
  tests/unit_test/speculative/test_target_worker.py

python -m pytest -q tests/unit_test/config/test_example_configs.py \
  -k qwen3_asr_1_7b_dflash_npu_graph
```

Require the accelerator test `test_draft_sampler_graph_replay_updates_mapped_tokens_in_shared_storage` to execute on NPU, not skip or run on CUDA. It captures a toy draft's head/mapping at two block counts, changes hidden states between replays, alternates 2->1->2 requests and checks known target IDs plus a stable output address. It uses the platform's graph/stream APIs and explicit producer/capture stream ordering. It does not run the SGLang graph runner, target attention or a trained ASR checkpoint; those are the next gate. The CPU storage-reuse test is a component invariant check only.

Keep full test commands/output and classify missing optional dependencies separately from candidate failures. Stop on a failure or skipped native component test; do not replace it with a mock result.

## Matched eager and graph servers

Both arms must use the same candidate code, environment, weights, data, device, memory budget and scheduling settings. Generate an eager reference from the graph example so max-running requests and graph bucket metadata cannot become unintended differences:

```python
from pathlib import Path
import yaml

config = yaml.safe_load(Path("examples/configs/qwen3_asr_1_7b_dflash_npu_graph.yaml").read_text())
config["name"] = "qwen3-asr-dflash-npu-graph-reference"
engine = config["stages"]["asr"]["engine"]
engine.pop("cuda_graph_backend_decode")
engine["disable_cuda_graph"] = True
Path("qwen3_asr_dflash_npu_graph_reference.yaml").write_text(yaml.safe_dump(config))
```

Removing the explicit full backend matters: a disable switch combined with an explicit graph backend may resolve differently. Inspect effective parsed settings rather than trusting only the YAML diff. Reference: both prefill/decode disabled. Candidate: prefill disabled, decode full, request buckets [1,2], request width eight (8/16 token rows). Both: encoder eager, compile off, same ascend target/draft backends. Use local snapshot overrides from the eager packet.

```bash
sgl-omni serve --config qwen3_asr_dflash_npu_graph_reference.yaml \
  --model-path "$TARGET_SNAPSHOT" \
  --asr.engine.speculative_draft_model_path "$DRAFT_SNAPSHOT" --port 8000

# Stop only the owned reference server, confirm device health, then run:
sgl-omni serve --config examples/configs/qwen3_asr_1_7b_dflash_npu_graph.yaml \
  --model-path "$TARGET_SNAPSHOT" \
  --asr.engine.speculative_draft_model_path "$DRAFT_SNAPSHOT" --port 8000
```

Allow at most 15 minutes for startup and 120 seconds per request/no-progress interval. On the first capture failure, device error, timeout or silent eager fallback, stop dependent gates and retain the first complete stack/log. Do not force eager sampling, disable one of the tested graphs, adjust accuracy criteria or patch a dependency to claim graph acceptance.

## Positive replay and correctness gates

Use the fixed audio selection/revisions in the eager packet and retain file IDs/hashes. First run 16 EN +16 ZH clips serially on fresh eager and graph servers. Then run the same set with deterministic phases of c2 -> c1 -> c2, using a start barrier for c2. Record actual request batches: two in-flight HTTP requests alone do not prove the worker executed a two-request batch. If c2 never produces the two-request graph bucket, report that coverage pending rather than pass it. Follow with 100 graph-mode serial requests cycling 20 fixed clips, and one clean shutdown.

For the candidate require separate positive evidence of:

- Target and draft `NPUGraphRunner` capture coverage for request buckets one/two, actual backend graph replay calls during serving and no fallback. A nonzero generic `can_run_graph` field or capture log alone is insufficient. Distinguish 8/16 token rows from request counts.
- Reduced-vocabulary capture-tail sampling into the worker's fixed output buffer, with mapped proposal IDs changing on different audio/hidden inputs. The draft head must not silently remain eager, and proposals must not use the target vocabulary head.
- Target verification auxiliary hidden states at the expected concatenated width, consumed for draft KV materialization before the next target forward reuses its graph output. Preserve the current worker's consume-before-reuse order; do not add a new stream/event scheme.
- Correct MRoPE positions, commit lengths including bonus, next sequence lengths, EOS/length truncation, valid KV locations and no replay of stale request data across bucket transitions. Capture-only and HTTP success do not establish these invariants.

If normal logs omit a field, retain a separately identified observation-only instrumentation diff/hash and bind it to the corresponding run. Instrument actual replay/worker handoff calls; do not infer execution from a class name. Do not change sampling, synchronization, graph buffers or metadata. Return per-request final token IDs, transcript, finish reason, correct draft count and verify rounds alongside the trace. Require eager/graph token IDs and finish reasons to match; investigate baseline drift or any mismatch before expanding. A corpus WER tolerance cannot replace this seam gate.

Require no duplicate/missing/padded tokens, unexpected empty output, ACL error, graph update stall, growing outstanding request count, stale hidden-state reuse or buffer overwrite. The 100-request test must use the pinned 48k draft, not inherit the prior 32k stability result. Verify temporary flattened configs are released, original weight snapshots remain, owned processes exit and device memory returns to its starting idle range. Cancellation, KV pool exhaustion and contexts beyond the chosen bound remain unqualified.

## Performance and return

Only after graph correctness passes, a separately allocated performance window may compare eager/graph at c1 and c2 sequentially on the same die in A-B-B-A order. Use equal warmup, fixed sample identities and cache policy, fresh logs and raw per-request results. Keep no-draft comparisons separate from this graph-only comparison. Report throughput, mean/p50/p95, failed requests and weighted acceptance from raw counters, plus repeated-leg spread; do not present one timing or periodic log averages as a qualified speedup.

Return instruction/candidate/runtime/image/weight/input identities; commands actually run; effective phase graph and sampler settings; target/draft capture and replay counts by request bucket; per-request correctness/counters; first failure and its owning layer; stability/cleanup; instrumentation identity and server evidence IDs. Redact private paths/host details. Source-only analysis locates a candidate failure owner; a new upstream/runtime patch requires a separate evidence-backed scope. Stop here for review before encoder graphs, prefill graphs, overlap, realtime, larger batches or another speculative algorithm.
