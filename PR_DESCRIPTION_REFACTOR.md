<!-- Title suggestion -->
# refactor: deduplicate TTS Engine Builder boilerplate via shared base helpers

## Motivation

The 8 concrete `TtsEngineBuilder` subclasses (Higgs, ZONOS2, Ming, MOSS,
MOSS-Local, Qwen3, FishAudio S2-Pro, Voxtral) repeat the same three pieces of
boilerplate verbatim:

- `make_model_runner` — an `importlib.import_module(...)` + 2-argument
  `SomeModelRunner(model_worker, output_proc)` instantiation.
- `post_scheduler_setup` — `model_runner.set_stream_outbox(scheduler.outbox)`.
- `make_abort_callback` — `assert self.model is not None; return
  self.model.reset_request`.

This duplication raises the cost of adding a new TTS model and makes it easy to
drift behavior between builders. The base class already exposes a
template-method `build()` flow with `NotImplementedError` hooks, so it is the
natural home for these shared defaults.

## Modifications

Added three opt-in helpers on `TtsEngineBuilder` in
`sglang_omni/scheduling/engine_factory.py` (base-class default behaviors are
unchanged — `post_scheduler_setup` stays a no-op and `make_abort_callback`
still returns `None` by default):

- `make_model_runner_from_path(model_worker, output_proc, *, module_path,
  class_name)` — generic importlib-based runner construction for the common
  2-argument shape.
- `bind_stream_outbox(scheduler, model_runner)` — shared
  `set_stream_outbox` wiring.
- `model_reset_request_abort_callback()` — shared `self.model.reset_request`
  wiring.

Refactored the 8 builders to delegate to these helpers instead of repeating the
code:

| Builder | `make_model_runner` | `post_scheduler_setup` | `make_abort_callback` |
| --- | --- | --- | --- |
| higgs_tts | uses helper | uses helper | uses helper |
| zonos2 | kept (extra args) | uses helper | uses helper |
| ming_tts | uses helper (keeps `_model_runner` cache) | unchanged | unchanged (uses `_model_runner`) |
| moss_tts | uses helper | unchanged | unchanged |
| moss_tts_local | uses helper | uses helper | kept (closure) |
| qwen3_tts | uses helper | unchanged | unchanged |
| fishaudio_s2_pro | uses helper | unchanged | unchanged |
| voxtral_tts | uses helper | unchanged | unchanged |

Builders with model-specific signatures (`zonos2` multi-arg runner,
`moss_tts_local` cleanup closure, `ming_tts` `_model_runner` cache) keep their
own overrides — no semantic change.

Extended `tests/unit_test/scheduling/test_engine_factory.py` with contract
tests for the three new helpers (runner instantiation args, outbox binding,
abort callback wiring), all passing.

## Related Issues

None (pure refactor / de-duplication; no tracked bug fixed).

## Accuracy Test

Not applicable — no model/runtime behavior changed.

## Benchmark & Profiling

Not applicable — no control-flow or numerical behavior changed.

## Checklist

- [x] Format your code according with pre-commit.
- [x] Add unit tests. (extended `test_engine_factory.py`)
- [x] Update documentation / docstrings / example tutorials as needed. (docstrings on new helpers)
- [ ] Provide throughput / latency benchmark results and accuracy evaluation results as needed. (N/A — refactor only)
- [ ] For reviewers: If you haven't made any contributions to this PR and are only assisting with merging the main branch, please remove yourself as a co-author when merging the PR.

## CI

CI runs on self-hosted GPU runners and requires a maintainer to add the
`run-ci` label. Once labeled, every subsequent push re-triggers CI as
long as the label remains. This PR is a behavior-preserving refactor; existing
CI (`test_zonos2_tts_ci.py`, `test_tts_ci.py`, `test_qwen3_omni_tts_ci.py`,
`test_ming_tp_parity_ci.py`) should remain green.

---

**Base branch:** `main`
**Local branch:** `refactor/tts-engine-builder-dedup`
**Reviewer quick check (CPU-only):** `python -m pytest tests/unit_test/scheduling/test_engine_factory.py -q`
