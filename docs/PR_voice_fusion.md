# Higgs TTS: voice timbre fusion (multi-reference output-distribution blending)

## What

Adds **voice timbre fusion** to the Higgs TTS (`higgs_multimodal_qwen3`) path: a single
synthesis can be conditioned on **N reference voices at once**, blended by weight, producing
one new voice that interpolates the references — not a switch between them, not a concatenation.

API (OpenAI-compatible `/v1/audio/speech`), backward compatible:

```jsonc
{
  "input": "Text to speak in the blended voice.",
  "references": [
    {"audio_path": "a.wav", "text": "...", "weight": 0.6},
    {"audio_path": "b.wav", "text": "...", "weight": 0.4}
  ]
}
```

`>= 2` references, at least one carrying a `weight` ⇒ fusion request (the weight is what marks
the intent to blend rather than the legacy "first ref wins" behavior — refs that omit it default
to weight `1.0`). Anything else keeps the exact legacy single-voice behavior. Weights are ratios
(need not sum to 1). `N > 2` supported.

## How it works

Higgs has **no speaker/style embedding** — timbre is conditioned entirely by reference audio
codes in-context. So fusion is done at the **output-distribution layer**, the one place a
clean, continuous blend is possible:

1. **Fan-out** (`request_builders.py`): a fusion request is split into `N` *sibling* rows,
   each prefilling one reference voice into its own KV context. All siblings share one
   `fusion_group_id` and **one concrete sampling seed**.
2. **Blend** (`fusion.py`, `model.py`): at every AR decode step, the `N` siblings'
   per-codebook output distributions are weighted-averaged **before** sampling
   (`log(Σ wᵢ·softmax(logitsᵢ/T))`). All siblings then sample the **same** frame (shared
   seed), so their `N` KV contexts stay in lock-step and decode identical codes; only the
   group **leader** is emitted as audio.
3. **Co-retire** (group barrier): "any sibling done ⇒ all done", so the group's rows finish
   on the same step.

The blend op is CUDA-graph friendly (fixed-shape `scatter_add_`/advanced-index, no host
control flow) and is a **byte-identical no-op** for non-fusion rows (each is its own singleton
group), so the production decode path is unchanged for ordinary requests.

## Co-batching: no forced admission, just decode-time enforcement

The siblings must be co-batched every step for the shared-seed lock-step to hold. Continuous
batching schedules rows independently, so an earlier version of this PR tried to force it by
having `OmniScheduler.get_next_batch_to_run` defer partial-group prefill batches back to the
waiting queue. **That mechanism was removed — it was unsound.** Upstream's
`ScheduleBatch.prepare_for_extend()` runs *before* this override ever sees the batch, already
flattening every req's tokens/KV-slots into batch-wide tensors sized to the full, untrimmed req
list; trimming `.reqs` afterward desyncs those tensors and corrupts the whole batch, not just the
deferred sibling (there's no supported way to shrink an already-prepared extend batch — upstream's
own `filter_batch` is decode-only).

What's left is a single enforcement point, entirely at decode time:

- **Prefill side is now read-only observability**: `get_next_batch_to_run` still scans for
  partial-group prefill batches and logs one when seen, but never mutates anything. Whether
  siblings land in the same prefill batch depends on ordinary scheduler behavior (they're enqueued
  adjacently, so this is the common case, but not guaranteed).
- **`model.py::_batch_local_fusion`** (prefill's own first-decode-step, no longer guarded by any
  admission mechanism): a split group's present rows are isolated (demoted to singletons, no
  blend) and the group is flagged *poisoned* in the registry — because this layer has no `Req`
  handle, it can't abort the rows itself, and because a group that splits here can later "heal"
  (every member eventually reaching decode together, looking complete by presence count alone)
  even though each side already sampled an unblended frame the moment it was isolated, permanently
  desyncing their KV contexts. The poison flag is what makes the next check below fail loud
  instead of silently resuming a fusion that can never be correct again.
- **`HiggsTTSModelRunner._populate_fusion_buffers`** (decode CG path — the only place a genuine
  mid-decode split is reachable, since KV-pressure retraction only acts on the running/decode
  batch): isolates a split group's present rows and aborts them (`FINISH_ABORT`), *also* aborting
  a group that looks complete again but was poisoned earlier.
- **`OmniScheduler._cascade_abort_split_fusion_group`** (in `stream_output`): the isolation above
  only touches rows *present* in that step's batch. The *absent* sibling — retracted, or just not
  there yet — is cascade-aborted too (via the existing `abort()`, which handles both
  `waiting_queue` and running-batch members), so it can't become a zombie that never gets cleaned
  up, and a client-facing error is emitted for it (including when the absent member happens to be
  the group's leader — the only rid the client is actually listening for).
- Fusion is registered *before* any batch is launched (at request-build time), and
  `HiggsTTSModelRunner.lookahead_eligible` returns `False` whenever any fusion request is
  registered, forcing fusion-carrying decode batches onto the synchronous path — this repo's own
  one-step-lookahead async-decode pipeline launches a step before resolving the previous one, and
  a launch-time abort from the isolation above would otherwise get mistaken for a stale prior-step
  overrun and silently dropped before it ever reaches the cascade above.

**Trade-off, stated plainly**: without prefill-side enforcement, fusion requests can now abort
under scheduling contention (partial admission that never heals) instead of transparently
retrying. Clients need to handle "this came back as an error, not audio" and retry themselves.
Deployments should still budget KV/concurrency as **1 fusion request = N rows** to keep this rare.

## Files

| File | Change |
|---|---|
| `models/higgs_tts/fusion.py` | **new** — `fuse_group_logits` / `fuse_group_generation_done` (pure torch) + `FusionRegistry` (pure Python, thread-safe group/weight/leader/poisoned bookkeeping), all unit-tested with no engine dependency |
| `models/higgs_tts/model.py` | delegates to `FusionRegistry`, blends logits in both decode paths, drives the group barrier, isolates + poisons a group split at prefill |
| `models/higgs_tts/model_runner.py` | CG fusion-buffer population (with an early-out for zero-fusion traffic, full-buffer reset on the dirty-to-clean transition), follower output dedup via `_finish_fusion_follower`, `lookahead_eligible` override forcing fusion traffic onto the sync decode path |
| `models/higgs_tts/request_builders.py` | sibling fan-out, shared seed, leader/follower |
| `models/higgs_tts/stages.py` | preprocessing/audio-encoder multi-reference handling |
| `models/higgs_tts/payload_types.py` | `HiggsTtsState.fusion_refs` |
| `scheduling/omni_scheduler.py` | read-only partial-group observability at prefill, decode-time cascade-abort, follower lifecycle |
| `serve/protocol.py` | `SpeechReference.weight` / `reference_codes` |
| `tests/unit_test/higgs_tts/` | blend numerics + `FusionRegistry` bookkeeping (26 cases, no engine) + pipeline fan-out and `_populate_fusion_buffers` tests (needs engine) |

## Testing

- **`test_voice_fusion.py`** (no GPU, no engine): 26 cases — blend ops (singleton
  byte-identity, weighted average, weight ratio, shared-seed identical draw, mixed batch,
  group barrier, temperature-before-blend, the greedy AND non-greedy sampling regression
  guards) plus `FusionRegistry` bookkeeping (register/clear/reuse/snapshot/poisoned).
  **All pass locally.**
- **`test_voice_fusion_pipeline.py`**: fan-out shape/seed/leader/weight, plus
  `HiggsTTSModelRunner._populate_fusion_buffers` exercised directly (intact-group blend,
  split-group isolation, poisoned-but-healed abort, stale-buffer-slot regression) via the same
  `object.__new__` + `SimpleNamespace` construction `test_async_decode_runner.py` already uses.
  Skips cleanly without the engine installed — not run on the author's Windows dev box, only
  read.

## Known items to verify on Linux + GPU (could not run here)

The author's dev box is Windows (no `sgl_kernel`/engine), so the blend math, registry logic, and
`_populate_fusion_buffers` isolation/poisoning are unit-tested and traced against the pinned
upstream source (`sglang==0.5.12.post1`) by hand, but none of it has run against a real engine:

1. **decode-time isolation + cascade-abort, end to end**: does a real KV-pressure retract
   actually get caught, isolated, and cascaded the way the unit tests (mocking the registry and
   runner methods) assume?
2. **How often prefill-side splits actually happen and heal**: without forced admission, what
   fraction of fusion requests hit the prefill-side isolation/poison path under real load, and of
   those, how many produce a user-audible bad first frame before the poisoned-group abort catches
   up versus getting cleanly aborted before any audio is emitted.
3. **CUDA-graph replay**: confirm per-step `_cg_fusion_group`/`_cg_fusion_weight` repopulation
   (including the new full-buffer reset on dirty-to-clean transitions) composes with graph
   capture/replay.
4. **`lookahead_eligible` forcing sync under fusion traffic**: confirmed by reading
   `_event_loop_async_decode` that this fully bypasses the launch/resolve split, but not run
   against a real `enable_async_decode=True` deployment.
5. **Deadlock guard**: deployments must budget `max_running_requests`/KV as **1 fusion request =
   N rows**, since sampler-pool capacity is shared and KV pressure is now the main source of
   splits (no admission-side backpressure any more).
