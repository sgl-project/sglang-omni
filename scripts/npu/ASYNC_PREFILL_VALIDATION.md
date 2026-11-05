# Qwen3-ASR NPU async prefill initial validation

Instruction revision: NPU-ASYNC-PREFILL-02 (2026-11-03). This packet defines validation for an experimental opt-in adaptation, not a performance claim. Branch: `qwen3-asr-npu-async-prefill` in `wangyao-i/sglang-omni`; base: `30da209c2330eaf50b1838fc82a2975c1ddc78f4`. Freeze the fetched branch commit before execution and return its full SHA separately from this instruction revision.

The [server report on PR #3460](https://github.com/sgl-project/sglang-omni/pull/3460#issuecomment-6604095899) records bounded eager/graph correctness, positive async scheduling traces and no stable performance gain on the tested workloads. The report includes server-only fixes; it does not qualify this revised local implementation as byte-identical to that server branch. The gates below remain a reference for future authorized runs, not a request to consume further hardware budget.

## Scope and runtime identity

Use only an assigned isolated NPU and task-owned processes, ports, logs and caches. Preserve existing checkouts and other workloads. Do not patch SGLang, torch_npu or site-packages, install dependencies, reset devices or disable the tested feature to obtain a pass. If the environment needs changes, return the mismatch first.

Record clean/dirty Omni HEAD and actual imported source paths, SGLang commit or exact build, torch/torch_npu/CANN/driver versions, device model/topology, model revision, audio sample manifest and content hashes, launch command and effective factory/server configuration. Check compatibility with the installation guide rather than assuming the workstation's dependencies describe the server.

The target is single-stream launch/resolve lookahead: prefill forward and sample → pinned host token snapshot → FutureMap token publication → device completion event → next eligible forward launch → previous step event wait and host collection. The existing two staging buffers and two completion events are reused through `post_decode_launch()` and the execution bridge. This path differs from the CUDA-only `stage_token_ids()` helper. Device forwards remain stream-ordered; the intended overlap is host result processing with device execution, not simultaneous prefill/decode compute streams. Native asynchronous submission alone does not establish a useful overlap window or performance gain. The pre-LM encoder remains independently stream-owned. No new worker or cross-stream wait is introduced by this patch. Native event, host-copy, FutureMap and graph-buffer behavior require evidence bound to the tested code and workload.

## Local and native tests

Run these in the server's existing matching environment before model workloads:

```bash
python -m pytest tests/unit_test/qwen3_asr/test_pipeline.py tests/unit_test/pipeline/test_async_decode.py tests/unit_test/model_runner/test_sglang_execution.py tests/unit_test/scheduling/test_prefill_coalesce_gate.py -q
```

The new factory tests mock infrastructure; they verify config propagation, not native NPU execution. Existing scheduler tests cover whole-prompt eligibility and hook order. Record collection failures as environment failures. Local Windows static/source-isolated checks do not replace this gate.

## Paired runs and target-path evidence

On the same candidate commit, compare A (`--asr.factory.enable_async_prefill false`) with B (`--asr.factory.enable_async_prefill true`), keeping `--asr.factory.enable_async_decode true` in both arms; async decode is already enabled by default on the ordinary NPU factory path. Leave `async_decode_min_batch_size=1` and other settings unchanged. An omitted/null prefill flag keeps CUDA on and NPU off while async decode is enabled; disabling async decode also disables the omitted prefill default. Explicit true with async decode false is rejected before infrastructure allocation. The explicit flag enables only CUDA/NPU, with DFlash and Torch MPS excluded. An attached `session_bridge` excludes prefill lookahead without excluding ordinary HTTP SSE transcription as a whole. This switch also selects upstream's scoped request-build admission behavior, so any speed difference is a whole-feature result rather than isolated kernel overlap. The ordinary non-MPS generation profile defaults to breakable prefill graphs; record the resolved backend rather than assuming an eager baseline.

Launch from the documented NPU environment with `PER_STREAM_QUEUE=1`, `TASK_QUEUE_ENABLE=1`, and `ASCEND_LAUNCH_BLOCKING=0` set before Python imports. The ordinary launcher already supplies the per-stream queue default; direct factory callers must establish the environment themselves. Retain all existing queue/graph restrictions in the NPU installation guide. Use a fresh service and fresh logs for each arm.

1. Smoke: 8 fixed English/Chinese clips at concurrency 1 in A and B, both ordinary and streaming transcription. Require completion, nonempty expected text, valid stream termination and no device/runtime errors. Use a 120-second readiness limit after the environment's established model-load phase and a 60-second no-request-progress limit.
2. Positive path: use a bounded profiler/temporary server-side trace to observe a whole-prompt extend entering `run_batch_launch`/`execute_launch`, token publication and completion-event recording, then a decode launch before that prefill's resolve. A scheduler flag, decode-only launch count or HTTP 200 alone is insufficient. Return trace identity and event ordering; retain the instrumentation patch separately. If workload coalescing prevents this sequence, report unexercised rather than passed. Test eager and the deployment's normal NPU graph configuration separately; eager success does not qualify graph execution.
3. Correctness/stability: freeze 32 clips spanning short/long audio and EN/ZH, including different shapes across reuse. At concurrency 1 and 8, run 3 rounds per arm and response mode (ordinary/streaming). Require all requests to terminate, zero empty/truncated outputs or malformed streams, no ACL 107033, graph-update errors, watchdog, allocator errors or service hangs. Retain every transcript and compare A/B normalized text and corpus WER. Any changed output requires inspection; do not assume numerical equivalence from HTTP success or flat aggregate WER. Add one cancel/disconnect during a mixed prefill/decode workload, then require subsequent requests to finish without KV/resource errors.
4. Fallbacks: positively observe synchronous routing for a partial/chunked prefill and a history-dependent sampling batch. Fallback in these cases is expected; fallback of every eligible whole-prompt prefill cannot qualify B.
5. Optional performance after correctness: fixed 32-clip manifest, warm each fresh service with 8 unscored clips, then three ABBA blocks at concurrency 1 and 8. Preserve per-request records, order, throughput, response and first-text mean/p95, WER, peak memory and startup costs. Keep streaming and non-streaming separate. Do not infer a gain from CUDA PR results; report regressions and variability.

Stop dependent gates on the first complete failure or 60 seconds without request progress. Preserve the first failure and task-owned logs; do not stack timing/queue/lock variants. Record the launch/resolve step, batch mode, effective configuration and runtime identity needed to distinguish setup, baseline and adaptation failures. Shut down only task-owned services and confirm assigned resources return to their pre-run state. Roll back by restarting with `enable_async_prefill=false` while preserving async decode.

## Return packet

Return instruction revision, frozen candidate SHA, imported dependency/runtime identities, effective A/B configuration, input/trace identifiers, gates actually executed, first failure if any, output differences/WER, target-path and fallback counts, resource/cleanup result, any server-only patch identity, and remaining unknowns. Full logs and private server paths stay on the isolated server; return redacted evidence identifiers. No upstream PR or merge is requested by this packet.
