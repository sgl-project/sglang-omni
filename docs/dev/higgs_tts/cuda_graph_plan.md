# Higgs TTS — CUDA Graph Capture Plan

Tracking doc for the AR-decode CUDA Graph capture effort. **Status:
not started.** Linked to roadmap issue #478.

## Why

AR decode dominates Higgs TTS end-to-end latency. Each step launches
~80 CUDA kernels, and on A100 the Python + launch dispatch overhead is
3-5 ms — about as much as the actual GPU compute (5-8 ms). CUDA Graph
records a fixed sequence of kernel launches once and replays it without
any Python in the critical path, removing that overhead.

Currently disabled via `disable_cuda_graph: True` in
`stages.py::create_sglang_tts_engine_executor` because:

1. Per-request sampler state lives in a Python `dict[str, _RequestSlot]`
   on `HiggsTTSModel._slots`.
2. `sampler.step()` has Python `if/elif` branches (delay window,
   wind-down, EOC detection).
3. CUDA Graph needs a fixed kernel sequence per captured batch size;
   any Python branching per step breaks the recording invariant.

## Expected impact

Estimates from per-step breakdown (single A100):

| Metric                    | Before  | After (target) |  Δ   |
|---------------------------|---------|----------------|------|
| AR step time              | ~13 ms  | ~8 ms          | -38% |
| RTF C=1                   | 0.55    | 0.35           | -36% |
| RTF C=32                  | 0.071   | 0.045          | -37% |
| audio_s/s C=32            | 14      | 22             | +57% |

Single biggest single-knob optimization on the roadmap.

## Design

Move per-request sampler state from Python dict to GPU tensors so the
sampler step becomes a fixed CUDA Graph-compatible kernel sequence,
then hand off capture to sglang's existing `CudaGraphRunner`.

### Architecture diff

```
Before:
  HiggsTTSModel._slots: dict[str, _RequestSlot]
                          ├── sampler: HiggsSamplerState  (Python ints)
                          └── output_codes: list[Tensor]   (per-step append)

  sampler.step(logits, state):
      if state.delay_count < N: ...           ← Python branch
      elif state.eoc_countdown is not None: ...
      elif codes_N[0] == eoc_id: ...
      state.delay_count += 1                    ← Python int mutate

After:
  HiggsTTSModel._sampler_state: HiggsBatchedSamplerState
                                  ├── delay_count:     [max_bs] int32   on GPU
                                  ├── eoc_countdown:   [max_bs] int32   on GPU
                                  ├── generation_done: [max_bs] bool    on GPU
                                  └── last_codes:      [max_bs, N] long on GPU

  HiggsTTSModel._rid_to_row: dict[str, int]    (CPU-side row mapping)
  HiggsTTSModel._free_rows: list[int]

  sampler.batched_step(logits, state, batch_indices):
      ... pure torch.where + scatter, no Python control flow ...
```

## Stages

### Stage 1 — Tensorize sampler state (3 days)

**Files**: `sampler.py`, `model.py`

- Add `HiggsBatchedSamplerState` dataclass holding 4 GPU tensors sized
  `[max_bs, ...]`.
- Replace `HiggsTTSModel._slots` with `HiggsBatchedSamplerState` +
  `_rid_to_row: dict[str, int]` + `_free_rows: list[int]`.
- Implement `acquire_row(rid) → int` (resets row state) and
  `release_row(rid)`.
- Keep the old per-step `sampler.step()` working for non-CUDA-Graph
  path (no behavior change yet).

**Acceptance**

- Single-request E2E produces bit-identical codes as today (same seed).
- Mixed-batch state stays per-row clean.
- `disable_cuda_graph` still `True`; no perf change expected at this
  stage.

### Stage 2 — Vectorize sampler step (3 days) ✅

**Files**: `sampler.py`

Add `batched_step()` that takes `logits[B, N, V]` + state + row
indices, returns `codes[B, N]`, and updates state in place — no Python
branching:

```python
def batched_step(logits_BNV, state, batch_indices, *,
                 temperature, top_p, top_k, boc_id=BOC_ID, eoc_id=EOC_ID):
    codes_BN = _sample_independent_batched(logits_BNV, temperature, top_p, top_k)

    delay_count   = state.delay_count[batch_indices]       # [B]
    eoc_countdown = state.eoc_countdown[batch_indices]     # [B]

    # delay window: codes at cb_idx > delay_count+1 → BOC
    cb_idx = torch.arange(N, device=logits_BNV.device)
    in_delay = (delay_count.unsqueeze(-1) < N) & (cb_idx.unsqueeze(0) > delay_count.unsqueeze(-1))
    codes_BN = torch.where(in_delay, boc_id, codes_BN)

    # EOC detection
    just_eoc = (codes_BN[:, 0] == eoc_id) & (delay_count >= N) & (eoc_countdown < 0)
    new_eoc = torch.where(
        just_eoc, torch.full_like(eoc_countdown, N - 2),
        torch.where(eoc_countdown >= 0, eoc_countdown - 1, eoc_countdown),
    )

    state.delay_count[batch_indices]     = delay_count + 1
    state.eoc_countdown[batch_indices]   = new_eoc
    state.generation_done[batch_indices] = new_eoc == 0
    state.last_codes[batch_indices]      = codes_BN
    return codes_BN
```

**Acceptance**

- Numerical parity vs `step()` on a large fuzz set (1000 random states
  × 8 codebook configurations) — codes must match bit-for-bit at
  matching seed.
- Microbench: `batched_step` < 0.6 ms at batch=8 (slightly slower than
  per-row `step` is fine; CUDA Graph recovers the cost).

### Stage 3 — Row allocation / recycling (2 days)

**Files**: `model.py`, `model_runner.py`

- `HiggsTTSModel.acquire_row(rid)` called from `HiggsTTSModelRunner.
  prepare_prefill`; pulls one free row, resets it.
- `HiggsTTSModel.release_row(rid)` called from `make_higgs_scheduler_
  adapters.result_adapter` (replaces `model.reset_request(rid)`) and
  from the OmniScheduler `on_abort` callback.
- `_collect_step_outputs` reads `state.generation_done[batch_indices]`
  via a single D2H copy each step (~10 μs) and sets
  `req.finished_reason = FINISH_MATCHED_TOKEN(EOC_ID)` for finished
  rows.

**Acceptance**

- Pool exhaustion test: submit `max_bs + 1` concurrent requests; the
  (max_bs+1)th gets rejected through `_request_kv_capacity_error` (or a
  similar path), not via a hard crash.
- Abort mid-decode releases the row; subsequent request reuses it.

### Stage 4 — Hook into sglang `CudaGraphRunner` (1-2 days)

**Files**: `stages.py`

```python
overrides = {
    "disable_cuda_graph": False,        # flip on
    "cuda_graph_max_bs": 32,
    # default capture set: [1, 2, 4, 8, 16, 32]
    ...
}
```

If sglang's capture machinery needs hints about per-model static
buffers, expose them via a `HiggsTTSModel.get_cuda_graph_static_
buffers()` method returning the 4 `HiggsBatchedSamplerState` tensors.

**Acceptance**

- Server log shows `"Capture cuda graph bs [1, 2, 4, 8, 16, 32]"` on
  startup.
- First inference after capture is fast (no JIT warm-up); subsequent
  warmer.

### Stage 5 — Padding (1 day)

**Files**: `model_runner.py`

CUDA Graph captures specific batch sizes (1, 2, 4, 8, 16, 32). When
actual batch != captured size, sglang pads with dummy rows. The padded
rows must not pollute real rows:

- Use a "valid rows" mask in `_collect_step_outputs` so finish detection
  only looks at real rows.
- Padding rows touch `delay_count` / `eoc_countdown` / `last_codes` —
  acceptable because `acquire_row` resets state when the row is later
  taken by a real request.

**Acceptance**

- Run mixed batch_size = 3, 5, 7, 9 concurrent loads. Per-request
  output is identical to running each request alone with the same seed.

### Stage 6 — Test + numerical parity (3 days)

**Tests** (new file `tests/test_higgs_tts_cuda_graph.py`):

- `test_batched_sampler_matches_per_row` — fuzz parity vs Stage 1.
- `test_mixed_batch_sizes` — capture set vs raw run, codes bit-identical.
- `test_abort_releases_row` — abort mid-decode, row count restored.

**Bench** (`_perf_bench/bench_cuda_graph.py`):

- Same `bench_concurrent.py` workload, flag on vs off.
- Compare RTF and AR step latency at C=1, 4, 8, 16, 32.

**Quality eval** (N=100 seed-tts en):

- Δ WER < 0.005
- Δ speaker_sim < 1.0

## Risks

| Risk                                        | Likelihood | Mitigation                            |
|---------------------------------------------|------------|---------------------------------------|
| Float drift from `where` vs `if`            | Medium     | Stage 2 strict bit-for-bit parity     |
| sglang capture rejects our `model.forward`  | Medium     | Validate first on a Qwen baseline     |
| Padding row pollutes real state             | Low        | Stage 5 dedicated test                |
| Pool exhaustion at high concurrency         | Low        | Promote to kv_capacity_error path     |

## Timeline

| Stage                          | Days | Cumulative |
|--------------------------------|------|------------|
| 1. Tensorize state             | 3    | 3          |
| 2. Vectorize sampler step      | 3    | 6          |
| 3. Row allocation              | 2    | 8          |
| 4. sglang capture wiring       | 1    | 9          |
| 5. Padding handling            | 1    | 10         |
| 6. Test + numerical parity     | 3    | **13 (~2.5 weeks)** |

## Exit criteria

- [ ] Numerical parity: N=100 seed-tts WER Δ < 0.005, sim Δ < 1.0
- [ ] Perf: RTF C=1 ≤ 0.40, audio_s/s C=32 ≥ 20
- [ ] Boundary cases pass: abort, chunked prefill, max_new_tokens
- [ ] Server startup penalty for capture < 60 s
