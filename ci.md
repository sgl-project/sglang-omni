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
| 0 | docs | `tests/docs/qwen3_omni/test_docs_qwen3_omni.py` | 1+2 | ✅ 14 passed in 309s | TextOnly 7/7 + SpeechMode 7/7 (incl. video+audio WER vs Whisper). Required Fix 1 (compiler). |
| 1 | stage-1 thinker length | `tests/test_model/test_qwen3_omni_thinker_length.py` | 1 | ✅ 3 passed in 42.49s | Initial fail: compiler `recv_endpoint` TypeError. 2nd fail (post-compiler-fix): API didn't reject overlong → scheduler crash → ReadTimeout cascade. 3rd fail: `finish_reason` always `"stop"`. **All three fixed**: see "Fixes applied during this run". |
| 2 | stage-2 TTS | `tests/test_model/test_qwen3_omni_tts_ci.py` | 2 | _pending_ | |
| 3 | stage-3 MMMU | `tests/test_model/test_qwen3_omni_mmmu_ci.py` | 1 | ❌ FAIL @ assertion | 50/50 requests succeeded, accuracy and latency pass. Fails on `KeyError: 'tok_per_s_agg'` — V1 benchmark summary dict is missing this key. Pipeline itself works; benchmark schema gap. |
| 4 | stage-4 MMMU Talker | `tests/test_model/test_qwen3_omni_mmmu_talker_ci.py` | 2 | _pending_ | |
| 5 | stage-5 MMSU | `tests/test_model/test_qwen3_omni_mmsu_ci.py` | 1 | _pending_ | |
| 6 | stage-6 MMSU Talker | `tests/test_model/test_qwen3_omni_mmsu_talker_ci.py` | 2 | _pending_ | |
| 7 | stage-7 Video-MME | `tests/test_model/test_qwen3_omni_videomme_ci.py` | 1 | _pending_ | |
| 8 | stage-8 Video-MME Talker | `tests/test_model/test_qwen3_omni_videomme_talker_ci.py` | 2 | _pending_ | |
| 9 | stage-9 Video-AMME | `tests/test_model/test_qwen3_omni_videoamme_ci.py` | 1 | _pending_ | |
| 10 | stage-10 Video-AMME Talker | `tests/test_model/test_qwen3_omni_videoamme_talker_ci.py` | 2 | _pending_ | |

Per-stage details (commands, log paths, error excerpts) are appended below as the runs complete.

---

## Fixes applied during this run

These fixes were merged into the working tree to unblock the V1 path. They are **prerequisites** for the CI workflow to make any progress on V1; without them, every single stage fails before the model finishes loading.

### Fix 1 — Single-process pipeline compile path (`sglang_omni_v1/config/compiler.py`)

`_compile_stage` was passing `recv_endpoint` / `coordinator_endpoint` / `abort_endpoint` directly to `Stage.__init__`, which doesn't accept them; it expects a `control_plane` (`StageControlPlane`) and a `role`. Mirrors the multi-process path in `pipeline/stage_process.py`.

```diff
+ from sglang_omni_v1.pipeline.control_plane import StageControlPlane
...
+ control_plane = StageControlPlane(
+     stage_name=stage_cfg.name,
+     recv_endpoint=stage_endpoints[stage_cfg.name],
+     coordinator_endpoint=endpoints["completion"],
+     abort_endpoint=endpoints["abort"],
+ )
  return Stage(
      name=stage_cfg.name,
+     role="single",
      get_next=get_next,
      gpu_id=relay_config["gpu_id"],
-     recv_endpoint=stage_endpoints[stage_cfg.name],
-     coordinator_endpoint=endpoints["completion"],
-     abort_endpoint=endpoints["abort"],
      endpoints=stage_endpoints,
+     control_plane=control_plane,
      ...
  )
```

### Fix 2 — API-edge prompt length validation (stage-1 tests 1 + 2)

V1 had no API-side validation for overlong prompts or `prompt + max_tokens > thinker_max_seq_len`. Without rejection, the request entered the scheduler and crashed with `scheduler crashed: len(new_indices)=132, len(keys)=256`, which killed the scheduler thread and made every subsequent request hang to `ReadTimeout`.

Files touched:
- `sglang_omni_v1/models/qwen3_omni/components/preprocessor.py` — port `validate_prompt_seq_len(...)` from v0 (verbatim error wording, since the test substring-matches), accept `max_seq_len` in `__init__`, call validator after tokenization.
- `sglang_omni_v1/models/qwen3_omni/stages.py` — `create_preprocessing_executor` now accepts `thinker_max_seq_len` and passes it to the preprocessor.
- `sglang_omni_v1/models/qwen3_omni/config.py` — `preprocessing` stage now carries `factory_args={"thinker_max_seq_len": 8192}` in both pipeline configs.
- `examples/run_qwen3_omni_server.py` and `examples/run_qwen3_omni_speech_server.py` — when `--thinker-max-seq-len` is overridden, route the new value to **both** the `thinker` stage and the `preprocessing` stage.
- `sglang_omni_v1/serve/openai_api.py` — port `_BAD_REQUEST_MARKERS` + `_is_bad_request_error` from v0; `_chat_non_stream`'s `Exception` handler now returns 400 when the marker matches, 500 otherwise.

### Fix 3 — `finish_reason` propagation (stage-1 test 3)

Nothing in the V1 chain wired SGLang's `req.finished_reason` into the result dict consumed by `Client._default_result_builder`, so every response defaulted to `finish_reason="stop"` regardless of how the request actually ended.

Files touched:
- `sglang_omni_v1/scheduling/sglang_backend/request_data.py` — added `finish_reason: str | None = None` to `SGLangARRequestData`.
- `sglang_omni_v1/scheduling/omni_scheduler.py` — `stream_output` now reads `req.finished_reason.to_json()["type"]` and stores it on the data object before calling the result adapter.
- `sglang_omni_v1/models/qwen3_omni/request_builders.py` — `apply_thinker_result` propagates `finish_reason` into `thinker_out`.
- `sglang_omni_v1/models/qwen3_omni/stages.py` — decode stage merges `thinker_out["finish_reason"]` into the result dict.

---

## Known V1 issues outside this PR's reach

These surfaced during the run but were **not** fixed (they don't gate stage-1):

- **`tok_per_s_agg` missing in V1 benchmark summaries.** `compute_speed_metrics` only adds the key when `total_engine_time > 0 AND total_tokens > 0`. V1's per-request `engine_time_s` and/or `completion_tokens` are not populated, so the key is dropped. CI's `assert_speed_thresholds` reads `summary["tok_per_s_agg"]` unconditionally → `KeyError`. Stage 3 hit this; stages 5/7/9 (and possibly the talker speed paths) are likely to hit it too.

---
