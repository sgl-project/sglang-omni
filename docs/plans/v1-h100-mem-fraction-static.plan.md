# [V1] Validate PR #426 Hardware-Aware mem_fraction_static on H100

## Goal Description

Validate, not re-port, PR #426 (`Ratish1/v1-hardware-aware-mem-fraction`, single commit `e9adf61` ahead of `main`) which introduces V1 hardware-aware `mem_fraction_static` autosizing plus per-role memory overrides and a thinker-only `--encoder-mem-reserve` reserve. The Humanize run drives a fix/review/benchmark loop until two terminal conditions are both met:

1. **Review-clean**: Codex `/review` reports no unresolved correctness, regression, security, performance, or missing-test findings against the PR branch state.
2. **Benchmark-clean**: H100 SeedTTS (EN then ZH) completes with zero failed requests and WER within the existing H100 reference range recorded in `benchmarks/eval/benchmark_omni_seedtts.py`.

Only minimal, targeted edits are produced in response to `/review` findings or benchmark regressions; no broad refactor is in scope unless evidence shows the implementation is structurally wrong.

In addition, the run verifies non-contradiction with the V1 architecture RFC (the "biggest RFC", covering Coordinator / Stage / Scheduler / ModelRunner / Pipeline Layer / Runtime parameter plumbing). PR #426 is the first concrete instance of the RFC's "runtime parameter plumbing" direction (`mem_fraction_static`, soon `thinker_max_seq_len` / `video_fps`). The validation checks that PR #426 advances the RFC vision rather than contradicting it; full RFC realization is out of scope and tracked separately.

## RFC Alignment Findings

The following intersection points between PR #426 and the V1 architecture RFC have been examined. Each verdict is recorded so the validation knows what to test and what to defer.

- RFC item — "Pick one memory-fraction semantics deliberately. Don't inherit the current ambiguity" (Chenyang note under "Stage placement — co-location as first-class").
  - PR #426 status: NEUTRAL. PR #426 exposes `mem_fraction_static` through CLI but does NOT choose between SGLang's "fraction of remaining VRAM after weights" and vLLM's "fraction of total VRAM." The numeric meaning is whatever SGLang's `ServerArgs.mem_fraction_static` defines today. The H100 server-launch logs (AC-10) record the actual post-resolution value so the eventual RFC decision has empirical grounding.
  - Verdict: NON-CONTRADICTORY. The semantics choice has not been made yet, so PR #426 cannot violate it. Future migration would change the value's interpretation; the CLI flag name and the override mechanism remain valid under either semantics.

- RFC item — "One canonical mechanism: a typed, stage-addressable override primitive at the PipelineConfig layer, with CLI / config / env as thin adapters on top. All runtime params go through it."
  - PR #426 status: PARTIAL ALIGNMENT.
    - Aligned: PR #426 adds `PipelineConfig.mem_fraction_role_to_stage()` (class method) as a discovery API at the PipelineConfig layer. This is RFC-shaped.
    - Misaligned: The override-APPLICATION primitive `_apply_stage_server_args_override` lives in `sglang_omni_v1/cli/serve.py` (CLI layer), not on `PipelineConfig`. V0's legacy `sglang_omni.config.schema.PipelineConfig.apply_server_args_overrides(stage_name, overrides)` IS the RFC-shaped primitive; PR #426 did not port this method to V1's `PipelineConfig`.
    - Consequence: Future config-file loaders, env adapters, or runtime overrides applying per-stage `mem_fraction_static` on V1 would have to either re-implement the dict-mutation logic, import a private CLI-layer helper, or motivate a refactor that promotes `_apply_stage_server_args_override` to V1's `PipelineConfig`.
  - Verdict: NON-CONTRADICTORY but PARTIAL. PR #426 does not block the RFC; it just leaves the canonical primitive at the wrong layer for V1. DEC-8 surfaces whether to expand PR #426 scope to fix this, or accept it as a known follow-up.

- RFC item — "Stages must be allowed to share GPUs. vLLM co-locates thinker + talker on one device via per-stage memory budgeting."
  - PR #426 status: ENABLING. Per-role memory split (`--thinker-mem-fraction-static` + `--talker-mem-fraction-static`) is structurally exactly what a future co-located placement model would need to budget thinker vs `talker_ar` on the same GPU. PR #426 does not enable same-GPU placement itself (current V1 still hard-rejects same-GPU speech-stage placement), but it does not block the future enablement either.
  - Verdict: NON-CONTRADICTORY, ENABLING. No runtime co-location test required by this plan; the structural compatibility is a documentation-only check (AC-15).

- RFC item — "Length validation today only guards the thinker input side; talker also needs an output-length cap."
  - PR #426 status: OUT OF SCOPE. PR #426 only touches memory params; it does not add `--talker-max-seq-len` or any output-length cap. The existing `--thinker-max-seq-len` is preserved.
  - Verdict: NEUTRAL. Not a blocker. Tracked separately.

- RFC item — Role-keyed CLI surface (`--thinker-mem-fraction-static`, `--talker-mem-fraction-static`) translating via `mem_fraction_role_to_stage` to actual stage names.
  - PR #426 status: NEW CONVENTION. Roles are a CLI-friendly abstraction layered on top of the stage vocabulary. The RFC's vocabulary is stage-based; PR #426 introduces "role" as a public concept without RFC endorsement.
  - Verdict: NON-CONTRADICTORY (the underlying mechanism remains stage-keyed; roles are a translation surface). DEC-9 surfaces whether to keep roles or strip the role layer in favor of pure stage-keyed flags.

- RFC item — Error handling at Scheduler layer (unified catch in `run_batch()`, no broad `except Exception` in model executors, fallbacks architecturally disallowed).
  - PR #426 status: UNRELATED. PR #426 does not touch error-handling paths or executor catches.
  - Verdict: OUT OF SCOPE.

- RFC item — `compile_pipeline()` / multi-process runner / `tp_size` schema.
  - PR #426 status: UNRELATED. PR #426 does not modify these layers.
  - Verdict: OUT OF SCOPE.

## Acceptance Criteria

Each criterion uses TDD-style positive/negative tests for deterministic verification.

- AC-1: Working tree is positioned on the PR #426 state before any validation runs.
  - Positive Tests (expected to PASS):
    - `git rev-parse HEAD` matches commit `e9adf61` (or a fast-forward descendant containing it).
    - `tests/test_v1_mem_fraction_static.py` exists on the working tree.
    - `git diff --name-only main...HEAD` lists exactly the ten files PR #426 touches: seven V1 source files (`sglang_omni_v1/cli/serve.py`, `sglang_omni_v1/config/schema.py`, `sglang_omni_v1/models/qwen3_omni/config.py`, `sglang_omni_v1/models/qwen3_omni/stages.py`, `sglang_omni_v1/scheduling/sglang_backend/__init__.py`, `sglang_omni_v1/scheduling/sglang_backend/server_args_builder.py`, `sglang_omni_v1/utils/misc.py`), one legacy dispatcher (`sglang_omni/cli/serve.py`), and two test files (`tests/test_v1_cli_version_dispatch.py`, `tests/test_v1_mem_fraction_static.py`).
  - Negative Tests (expected to FAIL / be rejected):
    - Running validation while `HEAD == a0a563b` (current `main`) is rejected because `tests/test_v1_mem_fraction_static.py` is absent.
    - Validation aborts if `git status --porcelain` reports modifications inside the PR-touched file set that did not come from the PR commit.

- AC-2: V1 `build_sglang_server_args` no longer forces `mem_fraction_static=0.7`.
  - Positive Tests:
    - With no explicit memory override, `build_sglang_server_args(model_path, ctx_len)` produces a `ServerArgs` whose `mem_fraction_static` is the SGLang hardware-aware value, not a hard-coded `0.7`.
    - On H100, the value reported in the server startup log (`Merged Configuration` block and SGLang's own init log) is at or above 0.85 (consistent with the H20 floor test pattern already in `tests/test_mem_fraction_static.py::TestH20AutoMemFractionFloor`).
  - Negative Tests:
    - A unit test asserts that the constructor does NOT receive `mem_fraction_static=0.7` when the user passed nothing.
    - An assertion that `build_sglang_server_args(..., mem_fraction_static=0.85).mem_fraction_static == 0.85` still holds, proving the parameter remains forwardable.

- AC-3: V1 CLI accepts and validates the four public memory flags.
  - Positive Tests:
    - `--mem-fraction-static 0.80`, `--thinker-mem-fraction-static 0.70`, `--talker-mem-fraction-static 0.65`, and `--encoder-mem-reserve 0.05` are accepted by `sglang_omni_v1/cli/serve.py` `serve` and reach the appropriate stages.
    - Per-role values override the global fallback: when both `--mem-fraction-static 0.80` and `--thinker-mem-fraction-static 0.70` are supplied to Qwen3-Omni speech, thinker receives 0.70 and `talker_ar` receives 0.80.
  - Negative Tests:
    - Values outside `(0, 1)` raise `typer.BadParameter` BEFORE `launch_server` is invoked.
    - `--encoder-mem-reserve` outside `[0, 1)` is rejected at the CLI boundary.

- AC-4: Qwen3-Omni V1 role mappings are exactly text=`{"thinker": "thinker"}` and speech=`{"thinker": "thinker", "talker": "talker_ar"}`.
  - Positive Tests:
    - `Qwen3OmniPipelineConfig.mem_fraction_role_to_stage()` returns `{"thinker": "thinker"}`.
    - `Qwen3OmniSpeechPipelineConfig.mem_fraction_role_to_stage()` returns `{"thinker": "thinker", "talker": "talker_ar"}`.
    - Global `--mem-fraction-static` on speech variant applies to BOTH thinker and `talker_ar`.
  - Negative Tests:
    - `--talker-mem-fraction-static` against the text-only Qwen3 pipeline is rejected.
    - Thinker-specific override does NOT leak into `talker_ar` when no global or talker flag is set.

- AC-5: Base and non-Qwen V1 pipelines do not silently inherit the public memory API.
  - Positive Tests:
    - Base `PipelineConfig.mem_fraction_role_to_stage()` returns `{}` (or otherwise reports "no public memory roles").
    - `sglang_omni_v1` Ming-Omni / FishAudio-S2-Pro pipelines (if registered) report empty role mappings.
  - Negative Tests:
    - `--mem-fraction-static` against any pipeline with empty role mapping raises `typer.BadParameter`.
    - `--thinker-mem-fraction-static` and `--talker-mem-fraction-static` against any non-Qwen pipeline are rejected at the CLI boundary.

- AC-6: `--encoder-mem-reserve` only affects the thinker auto-memory path and is mutually exclusive with every form of explicit thinker/global memory pin. When the CLI flag is omitted, the thinker factory default applies — per DEC-7 (PR #426 currently sets this default to `0.05` in `sglang_omni_v1/models/qwen3_omni/stages.py`).
  - Positive Tests:
    - With ONLY `--encoder-mem-reserve 0.20`, the thinker stage's `server_args.mem_fraction_static` equals `round(auto − 0.20, 3)`; `talker_ar` is untouched.
    - With `--encoder-mem-reserve` omitted on Qwen3 thinker, the factory default reserve from PR #426 (`0.05` unless DEC-7 changes it) is applied; `server_args.mem_fraction_static` equals `round(auto − 0.05, 3)`.
    - With `--encoder-mem-reserve 0` explicitly passed, `apply_encoder_mem_reserve` is a no-op (matches the V0 invariant in `tests/test_mem_fraction_static.py::test_apply_encoder_mem_reserve_noop_when_zero`).
  - Negative Tests:
    - `--encoder-mem-reserve 0.05` combined with `--mem-fraction-static 0.80` is rejected at the CLI boundary.
    - `--encoder-mem-reserve 0.05` combined with `--thinker-mem-fraction-static 0.70` is rejected at the CLI boundary.
    - `--encoder-mem-reserve 0.05` is rejected when the merged config already contains `server_args_overrides.mem_fraction_static` on the thinker stage (via config file or `global_cfg.runtime_overrides["thinker"]`); this exercises the precedence path in `sglang_omni_v1/config/compiler.py:296-302`.
    - `--encoder-mem-reserve` is rejected on any pipeline lacking a thinker stage.
    - A reserve that drops `mem_fraction_static` below the 0.1 floor raises `ValueError("... below the safe floor ...")` (matches V0 `TestEncoderMemReserveFloor`).

- AC-7: Legacy entry `sgl-omni serve --version v1 ...` forwards V1 memory flags through to the V1 CLI module instead of rejecting them.
  - Positive Tests:
    - `_build_v1_exec_argv(["sgl-omni", "serve", "--version", "v1", "--mem-fraction-static", "0.80", ...])` produces an argv that retains `--mem-fraction-static 0.80` and no longer contains `--version`.
    - `tests/test_v1_cli_version_dispatch.py` shows that the four memory flags survive forwarding for `--version v1`.
  - Negative Tests:
    - Passing `--mem-fraction-static` together with `--version v1` no longer raises "only supported by the legacy server" (the current main-branch behavior in `sglang_omni/cli/serve.py:141-157` must be inverted on the PR branch).
    - `--version legacy` continues to use the legacy serve path unchanged; the legacy path's existing memory-flag handling is preserved.

- AC-8: Focused V1 tests pass on the PR branch.
  - Positive Tests:
    - `pytest -q tests/test_v1_mem_fraction_static.py tests/test_v1_cli_version_dispatch.py` exits 0 with non-zero collected items in each file.
    - `pytest -q tests/test_v1_*.py` (the broader V1 suite) also exits 0.
  - Negative Tests:
    - A skipped or zero-collected run is NOT counted as a pass.
    - Failures in `tests/test_v1_mem_fraction_static.py` block proceeding to the `/review` loop step.

- AC-9: Codex `/review` is a hard prerequisite (per DEC-3). The run halts if `/review` is unavailable; the loop terminates only when an available `/review` reports no unresolved findings.
  - Positive Tests:
    - The Codex availability probe confirms `/review` is callable and the probe output is recorded verbatim.
    - The final iteration's `/review` output shows zero unresolved findings in the gating categories: correctness, regression, security, performance, missing-test.
  - Negative Tests:
    - If `/review` is unavailable for any reason (auth, network, skill not installed), the run STOPS at AC-9 and reports the failure mode — no unstructured-review fallback is taken (DEC-3).
    - If `/review` is available but the last run reports any unresolved finding in the gating categories, the loop continues; the plan does NOT terminate.
    - Style-only / nit findings do NOT block termination, but each must be recorded as "waived (style/nit)".

- AC-10: H100 server starts successfully under V1 auto memory on a known, recorded port.
  - Positive Tests:
    - Port preflight: `ss -lntH 'sport = :8000'` (or equivalent) returns empty BEFORE launching, OR the launcher's `_find_available_port` fallback port (logged as "Using port N instead." by `sglang_omni_v1/serve/launcher.py:51-64`) is captured and used in every downstream benchmark command — never assume port 8000.
    - `CUDA_VISIBLE_DEVICES=2,3 python -m sglang_omni.cli serve --model-path Qwen/Qwen3-Omni-30B-A3B-Instruct --version v1 --port 8000` reaches the ready state without OOM (or on the captured fallback port).
    - Server logs include the merged configuration block AND each AR stage's final `ServerArgs.mem_fraction_static`; both thinker and `talker_ar` values are recorded (SGLang-auto value or DEC-7-defaulted value, never the legacy `0.7`).
    - The actual physical GPU index map (from `CUDA_VISIBLE_DEVICES`) and the logical CUDA IDs reported in the SGLang init log are both recorded.
  - Negative Tests:
    - OOM or CUDA init failure blocks acceptance until resolved.
    - A silent port fallback that is NOT propagated into the benchmark commands is a defect (the benchmark would target the wrong server or refuse to connect); detection requires explicit comparison of the requested `--port` against the launcher's actual bound port.

- AC-11: H100 SeedTTS English benchmark (full set, generate then transcribe sequential, `voice_clone=True` per DEC-5) completes with zero failed requests and `wer_corpus` within tolerance.
  - Positive Tests:
    - `python -m benchmarks.eval.benchmark_omni_seedtts --generate-only --voice-clone --meta seedtts_testset/en/meta.lst --output-dir results/qwen3_omni_h100_en_mem_auto_vc --max-concurrency 16 --model qwen3-omni --port <captured-port>` exits 0, writes `speed_results.json` and `generated.json` into the output dir, and `speed_results.json` shows `summary.completed_requests == 1088` and `summary.failed_requests == 0` (split-mode artifacts; the combined `eval_results.json` is only emitted by the non-split run).
    - `python -m benchmarks.eval.benchmark_omni_seedtts --transcribe-only --meta seedtts_testset/en/meta.lst --output-dir results/qwen3_omni_h100_en_mem_auto_vc --model qwen3-omni --lang en --device cuda:0` exits 0 and writes `wer_results.json` and `asr_speed_results.json` whose summary reports `evaluated == 1088`, `skipped == 0`, and each per-sample record has `is_success == True`.
    - `wer_results.json`'s `accuracy.wer.summary.wer_corpus` is within `1.86% ± 0.30 pp` (i.e. `[1.56%, 2.16%]`) per DEC-2 and DEC-5 (H100 EN/VC=T reference from `benchmarks/eval/benchmark_omni_seedtts.py:65`).
  - Negative Tests:
    - Any non-zero `summary.failed_requests` in `speed_results.json`, any non-zero `summary.skipped` in `wer_results.json`, any timeout, empty audio, or missing WER aborts and triggers the next fix loop.
    - `wer_corpus` outside `[1.56%, 2.16%]` aborts and triggers the next fix loop.
    - Omitting `--voice-clone` invalidates the run (the gating reference is VC=T, not VC=F).

- AC-12: H100 SeedTTS Chinese benchmark (`voice_clone=True` per DEC-5) completes with zero failed requests and `wer_corpus` within the H100 VC=T reference range.
  - Positive Tests:
    - Same shape as AC-11 but with `seedtts_testset/zh/meta.lst`, `--voice-clone`, `--lang zh`, and a distinct `--output-dir` (e.g. `results/qwen3_omni_h100_zh_mem_auto_vc`). `speed_results.json` reports `summary.completed_requests == 2020` and `summary.failed_requests == 0`. `wer_results.json` reports `summary.evaluated == 2020` and `summary.skipped == 0`.
    - `wer_corpus` within `1.49% ± 0.30 pp` (i.e. `[1.19%, 1.79%]`) per DEC-2 and DEC-5 (H100 ZH/VC=T reference from `benchmarks/eval/benchmark_omni_seedtts.py:67`).
  - Negative Tests: Same as AC-11, with ZH corpus counts and the `[1.19%, 1.79%]` tolerance band.

- AC-13: Final summary contains the full evidence package required by the draft and the RFC-alignment record.
  - Positive Tests: The summary records:
    - Exact test commands and outcomes (focused + broader V1 suite).
    - `/review` availability status and final no-unresolved-findings evidence.
    - Server launch command and final `mem_fraction_static` for thinker and `talker_ar`.
    - Benchmark commands for EN and ZH (generate + transcribe), with output dirs.
    - WER (and ASR-WER), failed-request count, throughput / RTF.
    - Deviations from H200 and H100 reference rows in `benchmarks/eval/benchmark_omni_seedtts.py`.
    - The RFC-alignment verdicts from AC-14 and AC-15 (one line per intersection point listed in the "RFC Alignment Findings" section).
  - Negative Tests: Missing any of the items above blocks acceptance.

- AC-14: PR #426 is non-contradictory with the V1 architecture RFC's runtime-parameter-plumbing direction and its stage-addressable primitive vision.
  - Positive Tests:
    - `PipelineConfig.mem_fraction_role_to_stage` is defined at the PipelineConfig layer (verified by reading `sglang_omni_v1/config/schema.py` after preflight); it returns a `dict[str, str]` keyed by role name.
    - The four CLI flags (`--mem-fraction-static`, `--thinker-mem-fraction-static`, `--talker-mem-fraction-static`, `--encoder-mem-reserve`) route through one consistent path in `sglang_omni_v1/cli/serve.py` — there is no parallel mechanism (e.g. a per-stage memory field on `StageConfig`) that bypasses the override dict pattern.
    - The semantics of `mem_fraction_static` (SGLang fraction-of-remaining-VRAM-after-weights) is preserved unchanged; PR #426 does not redefine the metric.
  - Negative Tests:
    - A duplicate per-role memory field on `StageConfig` or `PipelineConfig` outside the `server_args_overrides` mechanism would be a contradiction; AC-14 fails if any such field is found.
    - Hard-coding total-VRAM-fraction semantics into PR #426 (e.g. computing the value from `torch.cuda.get_device_properties().total_memory` before passing to SGLang) would also be a contradiction; AC-14 fails if such logic is found.
    - A re-introduction of the legacy `mem_fraction_static=0.7` forced default would contradict the RFC's "don't inherit the current ambiguity" position (it would mean V1 forces a wrong-by-default value instead of using SGLang's auto-sized value).

- AC-15: PR #426 is forward-compatible with the RFC's announced future directions (the cases the RFC names but hasn't decided yet).
  - Positive Tests:
    - The CLI flag name `--mem-fraction-static` and its docstring describe behavior (a value passed to SGLang's `ServerArgs.mem_fraction_static`) without committing to a fraction-of-total vs fraction-of-remaining definition that a future RFC decision would have to un-promise.
    - The per-role memory split (`--thinker-mem-fraction-static`, `--talker-mem-fraction-static`) is structurally compatible with co-located placement: a user could in principle pin `--thinker-mem-fraction-static 0.55` and `--talker-mem-fraction-static 0.35` and have them sum below 1.0 on a single device, even though current V1 still rejects same-GPU speech-stage placement.
    - The role-to-stage mapping is overridable per pipeline class (each model's PipelineConfig subclass can declare its own `mem_fraction_role_to_stage`), so adding a third model with different stage names does not require touching CLI code.
  - Negative Tests:
    - Documentation that says "fraction of total VRAM" or "fraction of remaining VRAM" without flagging the RFC's open question is a forward-compat hazard; AC-15 fails if found.
    - A `runtime_overrides` write path that silently supersedes CLI-injected memory (without raising) would break the precedence story PR #426 needs to defend (already covered by AC-6 negative tests, restated here for the RFC-alignment view).

## Path Boundaries

### Upper Bound (Maximum Acceptable Scope)
The Humanize run is allowed to: operate on the pre-positioned PR branch state (DEC-1; the working tree is positioned by the harness before the loop starts — the loop itself does NOT run `git checkout`); run focused and broader V1 pytest suites; invoke Codex `/review`; apply minimal, targeted patches to the seven V1 source files PR #426 already touches and to the two PR-touched test files (`tests/test_v1_mem_fraction_static.py`, `tests/test_v1_cli_version_dispatch.py`); modify the legacy dispatcher in `sglang_omni/cli/serve.py` (also PR-touched) only to the extent of forwarding V1 memory flags; launch the V1 server on H100 (GPUs 2 and 3 preferred, auto-pick next free pair per DEC-4); run SeedTTS EN and ZH `--generate-only --voice-clone` then `--transcribe-only` (VC=T per DEC-5); perform RFC-alignment static analysis (AC-14, AC-15) on the PR branch source; and record the full evidence package including RFC verdicts. DEC-8 RESOLVED to "defer": no edits to V1's `sglang_omni_v1/config/schema.py` are permitted in this validation run. The override-primitive port to PipelineConfig is recorded as a known RFC follow-up item in the final summary, not implemented here.

### Lower Bound (Minimum Acceptable Scope)
The Humanize run must, at minimum: position the working tree on the PR #426 state; execute the focused tests in `tests/test_v1_mem_fraction_static.py` and `tests/test_v1_cli_version_dispatch.py`; invoke Codex `/review`; address all valid unresolved findings; launch the V1 server on H100 with a recorded port; run SeedTTS for BOTH English and Chinese end-to-end (both are gating per AC-11 and AC-12); and record the final evidence package. AC-11 and AC-12 are mandatory acceptance; "one language only" is NOT a passing state — it is a deferred state that requires a new fix loop or user-granted scope reduction.

### Allowed Choices
- Can use:
  - Codex `/review` skill for review iterations.
  - Existing test infrastructure (pytest, the V1 test fixtures already on the PR branch).
  - The existing benchmark script `benchmarks/eval/benchmark_omni_seedtts.py` as-is — only patch the script if a real bug blocks running the validation, never to inflate the reported WER margin.
  - The existing override hooks: `PipelineConfig.mem_fraction_role_to_stage()`, `apply_server_args_overrides`, `apply_encoder_mem_reserve`.
  - Working tree pre-positioned on the PR #426 state by the RLCR harness before the loop starts (DEC-1 RESOLVED). The RLCR loop verifies HEAD; it does not run `git checkout` or `git fetch`.
- Cannot use:
  - Any broad refactor unrelated to a `/review` finding or benchmark regression.
  - Renaming existing public APIs.
  - Mocks that hide real SGLang behavior at the AC-2 boundary (the auto-value assertion needs a real `ServerArgs` construction path, not a `SimpleNamespace`).
  - Adding new public CLI memory flags beyond the four listed in AC-3.
  - Skipping the legacy dispatcher inversion — the PR explicitly requires `--version v1` to forward, not reject, memory flags.

> **Note on Deterministic Designs**: The draft pins specific reference behavior (the four CLI flags, the exact role-mapping shapes, the legacy-forwarding inversion, the WER reference table). For these items the upper and lower bounds collapse to the draft text — implementations may not silently introduce different role shapes or different default behaviors.

## Feasibility Hints and Suggestions

> Reference only — not prescriptive.

### Conceptual Approach
A single iteration of the loop looks like:

```
preflight:
  # working tree pre-positioned on PR #426 by harness; loop does NOT switch branches
  assert HEAD == e9adf61 (or descendant)
  assert tests/test_v1_mem_fraction_static.py exists

iterate until clean:
  pytest -q tests/test_v1_mem_fraction_static.py tests/test_v1_cli_version_dispatch.py
  if any failure: claude fixes minimally -> goto iterate
  pytest -q tests/test_v1_*.py
  if any failure: claude fixes minimally -> goto iterate
  ask Codex: is /review available?
  if NO: HALT and report (DEC-3 disallows fallback)
  Codex runs /review against the PR branch state
  if unresolved correctness/regression/security/performance/missing-test findings:
    claude fixes minimally -> goto iterate

benchmark (H100, prefer GPUs 2/3, auto-pick next free pair if busy):
  probe port 8000; capture launcher fallback port if needed
  launch server (V1, auto memory; thinker is auto - 0.05 reserve per DEC-7)
  capture: merged config, thinker mem_fraction_static, talker_ar mem_fraction_static,
           CUDA_VISIBLE_DEVICES, logical CUDA IDs, bound port

  EN VC=T: generate-only --voice-clone -> transcribe-only (sequential)
    if summary.failed_requests > 0 OR summary.skipped > 0 OR
       wer_corpus not in [1.56%, 2.16%]: diagnose, fix minimally, goto iterate

  ZH VC=T: generate-only --voice-clone -> transcribe-only (sequential)
    if summary.failed_requests > 0 OR summary.skipped > 0 OR
       wer_corpus not in [1.19%, 1.79%]: diagnose, fix minimally, goto iterate

summarize:
  test commands and outcomes
  /review status (must be available + zero unresolved)
  benchmark commands (with --voice-clone), wer_corpus, ASR-WER, failed count, throughput/RTF
  deviations from H100/H200 reference table; CUDA_VISIBLE_DEVICES; bound port
```

### Relevant References

- `sglang_omni_v1/scheduling/sglang_backend/server_args_builder.py` — `build_sglang_server_args` currently has `mem_fraction_static: float = 0.7` (line 17). PR #426 removes the forced default and adds `apply_encoder_mem_reserve`.
- `sglang_omni_v1/cli/serve.py` — currently exposes no memory flags. PR #426 adds the four flags plus validation.
- `sglang_omni/cli/serve.py:141-157` — current main rejects V1 memory flags when `--version v1`. PR #426 inverts this.
- `sglang_omni_v1/config/schema.py` — adds `mem_fraction_role_to_stage` class hook and per-stage override helper.
- `sglang_omni_v1/config/compiler.py:296-302` — `_resolve_factory_args` applies `global_cfg.runtime_overrides` on top of `stage.factory_args`. AC-6's runtime-override exclusion test must exercise this code path.
- `sglang_omni_v1/models/qwen3_omni/config.py` and `stages.py` — Qwen3 text and speech variants; PR #426 wires the role mapping and the speech `talker_ar` factory.
- `sglang_omni_v1/utils/misc.py` — PR #426 adds 11 lines; likely the `apply_encoder_mem_reserve` helper or a rounding helper.
- `tests/test_v1_mem_fraction_static.py` — new file, 246 lines. Mirror the V0 test patterns from `tests/test_mem_fraction_static.py` (encoder-mem-reserve subtraction, floor enforcement, per-role precedence, atomic apply on partial cast failure).
- `tests/test_v1_cli_version_dispatch.py` — already exists; PR #426 adds 34 lines covering memory-flag forwarding.
- `benchmarks/eval/benchmark_omni_seedtts.py:50-117` — H200 and H100 reference tables with WER, ASR-WER, RTF, sample counts, failed counts.

## Dependencies and Sequence

### Milestones

1. Milestone A — Working Tree Preflight
   - Step 1: Verify the working tree is already positioned on the PR #426 state — the RLCR harness pre-positions the worktree before the loop starts; do NOT run `git fetch` or `git checkout`.
   - Step 2: Verify `git rev-parse HEAD` resolves to `e9adf61` (or a descendant containing it).
   - Step 3: Verify the ten PR-touched files exist and `git diff --name-only main...HEAD` matches the expected list (seven V1 source + one legacy dispatcher + two tests).
   - Step 4: Snapshot any pre-existing local changes outside the PR file set so they cannot contaminate validation.

2. Milestone B — Focused Test Pass
   - Step 1: Run `pytest -q tests/test_v1_mem_fraction_static.py tests/test_v1_cli_version_dispatch.py`.
   - Step 2: If any failure, apply minimal targeted patch; repeat.
   - Step 3: Once focused tests pass, run broader `pytest -q tests/test_v1_*.py`.
   - Step 4: Same minimal-patch loop until clean.

3. Milestone C — Review Loop (gated by `/review` availability per DEC-3)
   - Step 1: Codex checks `/review` availability and records the outcome. If unavailable, HALT and report the failure mode; do NOT proceed without `/review`.
   - Step 2: Run `/review` against the current branch state.
   - Step 3: Claude applies minimal patches for each valid unresolved finding (correctness / regression / security / performance / missing-test).
   - Step 4: Re-run focused + broader V1 tests.
   - Step 5: Re-run `/review`.
   - Step 6: Repeat until `/review` returns no unresolved findings (in the five gating categories); waive style-only findings explicitly with reason.

4. Milestone D — H100 Server Launch
   - Step 1: Probe GPU availability. Prefer GPUs 2 and 3; if either is busy, auto-pick the next free consecutive H100 pair (DEC-4 RESOLVED). Record the selected `CUDA_VISIBLE_DEVICES` values.
   - Step 2: Probe port 8000; if it is occupied, capture the launcher fallback port (the "Using port N instead." log line from `sglang_omni_v1/serve/launcher.py:51-64`).
   - Step 3: Start the V1 server with the draft's launch command (replacing `CUDA_VISIBLE_DEVICES` and `--port` to match Step 1/2 if needed).
   - Step 4: Capture the merged configuration block, SGLang's final per-stage `mem_fraction_static`, and the bound port. Both thinker and `talker_ar` final values must be SGLang-auto (or auto − 0.05 on thinker per DEC-7), never `0.7`.

5. Milestone E — SeedTTS English Benchmark (VC=T per DEC-5)
   - Step 1: Prepare dataset if missing (`python -m benchmarks.dataset.prepare --dataset seedtts`).
   - Step 2: Run `--generate-only --voice-clone` for `seedtts_testset/en/meta.lst` into `results/qwen3_omni_h100_en_mem_auto_vc/`. Use the port from Milestone D Step 2.
   - Step 3: Run `--transcribe-only` for the same output dir.
   - Step 4: Parse `speed_results.json` for `summary.completed_requests` and `summary.failed_requests`; parse `wer_results.json` for `summary.evaluated`, `summary.skipped`, and `accuracy.wer.summary.wer_corpus`; parse `asr_speed_results.json` for ASR throughput.
   - Step 5: Compare `wer_corpus` against `1.86% ± 0.30 pp` (H100 EN/VC=T reference); on regression, fix minimally and reset to Milestone C.

6. Milestone F — SeedTTS Chinese Benchmark (VC=T per DEC-5)
   - Same shape as Milestone E for `seedtts_testset/zh/meta.lst` → `results/qwen3_omni_h100_zh_mem_auto_vc/`. Compare `wer_corpus` against `1.49% ± 0.30 pp` (H100 ZH/VC=T reference).

7. Milestone G — RFC Alignment Static Analysis
   - Step 1: Verify AC-14 — `mem_fraction_role_to_stage` lives at the PipelineConfig layer, no parallel mechanism on `StageConfig`, no hard-coded total-VRAM semantics, no re-introduction of forced `0.7` default.
   - Step 2: Verify AC-15 — CLI docstrings remain semantics-neutral, role-to-stage mapping is per-subclass overridable, per-role split is structurally co-location-ready.
   - Step 3: Record the verdict line per RFC item (one line per intersection point from the "RFC Alignment Findings" section).

8. Milestone H — Final Evidence Package
   - Step 1: Compile all required summary items (AC-13), including the AC-14/15 verdicts.
   - Step 2: Report results.

Dependencies: B depends on A. C (review loop) depends on B fully completing (both focused tests and the broader V1 suite), reflected in `task6` depending on `task5`. D depends on C reaching no-unresolved. E depends on D. F depends on E. G (RFC static analysis via task15/task16) depends only on A (preflight) and may run in parallel with B–F. H (final evidence) depends on E, F, and G.

## Task Breakdown

| Task ID | Description | Target AC | Tag | Depends On |
|---------|-------------|-----------|-----|------------|
| task1 | Verify the working tree is already positioned on the PR #426 state (harness pre-positions before the loop starts); confirm HEAD includes `e9adf61` and `tests/test_v1_mem_fraction_static.py` exists. Do NOT run `git fetch` or `git checkout`. Abort if HEAD is wrong. | AC-1 | coding | - |
| task2 | Inventory the ten files PR #426 touches and snapshot any pre-existing local diff outside that set; abort if contamination is detected. | AC-1 | coding | task1 |
| task3 | Run `pytest -q tests/test_v1_mem_fraction_static.py tests/test_v1_cli_version_dispatch.py`; capture verbatim output. | AC-2, AC-3, AC-4, AC-5, AC-6, AC-7, AC-8 | coding | task2 |
| task4 | If task3 fails, apply minimal targeted patches inside the PR-touched file set; re-run until green. | AC-2..AC-8 | coding | task3 |
| task5 | Run `pytest -q tests/test_v1_*.py` (broader V1 suite); fix regressions with the same minimal-patch loop. | AC-8 | coding | task4 |
| task6 | Codex probes `/review` availability; record outcome (path, version, error if any). | AC-9 | analyze | task5 |
| task7 | If `/review` is available, run it against the PR branch state and persist its findings categorized as correctness / regression / security / performance / missing-test / style. | AC-9 | analyze | task6 |
| task8 | Claude applies minimal targeted patches to address each unresolved gating finding; re-run task3..task5 between rounds. | AC-2..AC-9 | coding | task7 |
| task9 | Loop task7..task8 until `/review` reports zero unresolved gating findings. | AC-9 | analyze | task8 |
| task10 | Verify port 8000 is free (or capture the launcher's `_find_available_port` fallback); launch H100 V1 server on preferred GPUs 2/3 with the draft's serve command; capture merged config, final per-stage `mem_fraction_static`, physical and logical GPU IDs, and the actual bound port for use by downstream benchmark commands. | AC-10 | coding | task9 |
| task11 | Run SeedTTS EN `--generate-only --voice-clone` (VC=T per DEC-5) and `--transcribe-only` sequentially using the bound port from task10; persist `speed_results.json`, `generated.json`, `wer_results.json`, and `asr_speed_results.json` into `results/qwen3_omni_h100_en_mem_auto_vc/`. | AC-11 | coding | task10 |
| task12 | Compare EN `wer_corpus` against `[1.56%, 2.16%]` (H100 EN/VC=T 1.86% ± 0.30 pp per DEC-2/DEC-5), `summary.failed_requests` (generation) against `0`, and `summary.skipped` (transcription) against `0`; on regression, drop back to task7 with the failure as a new finding. | AC-11 | analyze | task11 |
| task13 | Run SeedTTS ZH `--generate-only --voice-clone` (VC=T per DEC-5) and `--transcribe-only` sequentially using the bound port from task10; persist `speed_results.json`, `generated.json`, `wer_results.json`, and `asr_speed_results.json` into `results/qwen3_omni_h100_zh_mem_auto_vc/`. | AC-12 | coding | task12 |
| task14 | Compare ZH `wer_corpus` against `[1.19%, 1.79%]` (H100 ZH/VC=T 1.49% ± 0.30 pp per DEC-2/DEC-5), `summary.failed_requests` (generation) against `0`, and `summary.skipped` (transcription) against `0`; on regression, drop back to task7. | AC-12 | analyze | task13 |
| task15 | Read `sglang_omni_v1/config/schema.py` and `sglang_omni_v1/cli/serve.py` on the PR branch; record where `mem_fraction_role_to_stage` and `_apply_stage_server_args_override` are defined; verify there is no parallel per-role memory field on `StageConfig` or `PipelineConfig` that bypasses the override dict pattern; record the verdict per AC-14 positive/negative test. | AC-14 | analyze | task2 |
| task16 | Read the same files plus the four CLI flag docstrings; verify the docstrings do not commit to fraction-of-total vs fraction-of-remaining semantics; check that role-to-stage mapping is overridable per pipeline subclass; record the verdict per AC-15. | AC-15 | analyze | task2 |
| task17 | Compile the final evidence package (commands, `/review` status, server-side memory values, WER, ASR-WER, failed count, throughput/RTF, deviations from H200 and H100 references, RFC-alignment verdicts from task15 and task16, DEC-8 / DEC-9 final state). | AC-13 | coding | task14 |

## Claude-Codex Deliberation

### Agreements

- The current branch is `main` and the validation target is the PR #426 state at commit `e9adf61`; the plan must explicitly position the working tree before running validation.
- `tests/test_v1_mem_fraction_static.py` does not exist on `main` and only appears on the PR branch; the focused test set referenced in the draft cannot be executed without first positioning the tree.
- The existing `tests/test_mem_fraction_static.py` covers the legacy V0 engine and must NOT be substituted for the V1 test surface.
- AC-6 must cover three different forms of explicit thinker memory pin (CLI, `factory_args.server_args_overrides`, `runtime_overrides`) because `sglang_omni_v1/config/compiler.py:296-302` can let `runtime_overrides` silently supersede CLI inputs.
- The legacy dispatcher in `sglang_omni/cli/serve.py:141-157` currently REJECTS the V1 memory flags; the plan must verify PR #426 inverts this rather than treat the existing behavior as correct.
- Server startup must record the final per-stage `mem_fraction_static` from logs, because unit tests alone cannot prove SGLang chose a hardware-aware value on real H100 hardware.
- The H100 reference values in `benchmarks/eval/benchmark_omni_seedtts.py` are the WER comparison source.
- Style-only `/review` findings do not gate termination; correctness/regression/security/performance/missing-test findings do.

### Resolved Disagreements

- Topic: "Validate `main` vs validate the PR branch" — Codex flagged that 'current local branch' is ambiguous given the user is sitting on `main`. Resolution: the plan explicitly requires positioning the working tree on the PR #426 state, with DEC-1 selecting between checkout and cherry-pick. Rationale: the draft says "PR #426 已经基本包含了预期的设计", so the validation target is unambiguously the PR's code, not bare `main`.
- Topic: "Should the plan also CPU-mock the SGLang ServerArgs path?" — Codex worried mocks may hide real behavior; Claude noted that the V0 test pattern in `tests/test_mem_fraction_static.py` mixes mocked factory tests with a real H20 floor test that constructs `ServerArgs(...)` directly. Resolution: the V1 test surface follows the same mixed pattern. AC-2's positive test on H100 explicitly asserts the real SGLang-picked value, which removes the mock blind spot.
- Topic: "Full-set benchmark vs subset smoke first" — Codex recommended a subset smoke before the full set. Resolution: keep the draft's full-set instruction as the AC, but allow Milestone D's server-launch evidence to satisfy the smoke-equivalent role (server reaches ready state without OOM). This keeps acceptance aligned with the draft while still front-loading failure detection.
- Topic: "PR file count mismatch (round 2)" — Codex flagged that the round-1 plan said "eight V1 source files plus the legacy dispatcher plus two test files" which sums to eleven, not ten. Resolution: AC-1's positive test now enumerates the exact ten files: seven V1 source files + one legacy dispatcher + two test files. Verified against `git show --stat e9adf61`.
- Topic: "encoder_mem_reserve default behavior (round 2)" — Codex flagged a semantic conflict between AC-6's "with `--encoder-mem-reserve` omitted, no reserve is applied" and PR #426's actual `encoder_mem_reserve: float = 0.05` thinker-factory default. Resolution: AC-6 now reflects the PR's actual behavior (default `0.05` when omitted), and the choice is surfaced as DEC-7 for explicit user sign-off.
- Topic: "Benchmark output artifact names (round 2)" — Codex flagged that AC-11/AC-12 referenced `eval_results.json` (only produced in the non-split path at line 526) and `failed_requests` (a generation-side counter). Resolution: AC-11 and AC-12 now reference the actual split-mode artifacts (`speed_results.json`, `generated.json`, `wer_results.json`, `asr_speed_results.json`) and split the gating checks by metric: generation = `summary.completed_requests` / `summary.failed_requests`, transcription = `summary.evaluated` / `summary.skipped` / `accuracy.wer.summary.wer_corpus`.
- Topic: "Port fallback (round 2)" — Codex flagged that `sglang_omni_v1/serve/launcher.py:51-64` silently falls back to a different free port when 8000 is taken, which would invalidate downstream benchmark `--port 8000` calls. Resolution: AC-10 now requires either a port preflight or capture-and-propagate of the bound port, and task10 owns that responsibility.
- Topic: "Lower Bound vs AC-12 contradiction (round 2)" — Codex flagged that Lower Bound's "at least one SeedTTS language" clause contradicted AC-12 (Chinese benchmark mandatory). Resolution: Lower Bound now requires BOTH languages, consistent with AC-11 and AC-12 being gating criteria.
- Topic: "DEC-5 internal contradiction (round 2)" — Codex flagged that DEC-5's Claude position simultaneously said "run BOTH modes" and "add VC=T if time allows." Resolution: DEC-5 now commits to VC=F as the gating mode for both languages with VC=T as optional, eliminating the contradiction. (User subsequently overrode this to VC=T gating only; see DEC-5 Decision Status.)
- Topic: "RFC override primitive location (post-convergence, user-driven)" — User requested a cross-check against the V1 architecture RFC. The RFC asks for the override-APPLICATION primitive to live at `PipelineConfig`; PR #426 leaves it at the CLI layer. Resolution: This is recorded as PARTIAL ALIGNMENT (non-contradictory) in the "RFC Alignment Findings" section, gated by AC-14 (no contradiction allowed) and AC-15 (forward-compatible). The fix-vs-defer choice is surfaced as DEC-8. Roles-as-CLI-only is surfaced as DEC-9.

### Convergence Status
- Final Status: `converged` (with RFC-alignment review applied post-convergence). Two Codex passes ran on the original draft (first-pass analysis + round-2 reasonability review) and all REQUIRED_CHANGES were applied. A subsequent user-driven RFC cross-check surfaced one non-contradictory but partial-alignment item (override-application primitive at CLI layer vs PipelineConfig layer); this is recorded in AC-14 / AC-15 and surfaced as DEC-8 and DEC-9 for explicit user direction. Remaining open items are user decisions, not Claude/Codex disagreements.

## Pending User Decisions

- DEC-1: How should the working tree be positioned on the PR #426 state?
  - Claude Position: Prefer working directly from the PR branch state because it is pure PR validation and reproduces the PR author's exact tree.
  - Codex Position: Either choice is valid; cherry-pick onto `main` catches merge drift but is no longer a pure PR-branch validation.
  - Tradeoff Summary: PR-branch state is cleanest and matches "validate PR #426"; cherry-pick onto `main` better simulates the merge state but adds risk of unrelated `main` changes contaminating the run.
  - Decision Status: RESOLVED — Working tree is pre-positioned on the PR #426 state (commit `e9adf61` on local branch `pr-426`, tracking `Ratish1/v1-hardware-aware-mem-fraction`) by the RLCR harness before the loop starts. The loop only verifies HEAD; it never runs `git fetch` or `git checkout`. The RLCR loop runs inside a git worktree at `/data/chenyang/sglang-omni-pr426`; on completion the harness pushes the final `pr-426` commit(s) back to `Ratish1/v1-hardware-aware-mem-fraction`.

- DEC-2: What WER tolerance defines "within reference range" against the H100 row in `benchmarks/eval/benchmark_omni_seedtts.py`?
  - Claude Position: Use absolute deviation `<= 0.30 percentage points` on `wer_results.json`'s `accuracy.wer.summary.wer_corpus` (corpus-level WER, NOT per-sample mean) versus the H100 reference WER per `(language, voice_clone)` cell, matching the typical run-to-run variance in the H200 PR #393 entries.
  - Codex Position: Define explicitly — pure-max, max + absolute margin, relative percent, or human judgment.
  - Tradeoff Summary: A loose tolerance hides real regressions; a tight tolerance flakes on benign run variance. ±0.30 pp on `wer_corpus` is conservative without being noisy, and `wer_corpus` is the metric the reference table itself reports.
  - Decision Status: RESOLVED — Absolute deviation `<= 0.30 percentage points` on `accuracy.wer.summary.wer_corpus` versus the H100 reference per `(language, voice_clone)` cell. Treated as a HARD requirement (not an optimization direction).

- DEC-3: If Codex reports `/review` is unavailable, what is the fallback?
  - Claude Position: Continue with Codex unstructured review of the diff against `main` and record `/review unavailable: <reason>` in the final summary; do not block acceptance solely on tooling absence.
  - Codex Position: Options are fail, fall back to manual Codex review, or record a waived criterion.
  - Tradeoff Summary: Blocking on `/review` availability would make the entire validation gated on a tool that is environmental; manual Codex review preserves the safety intent without false negatives.
  - Decision Status: RESOLVED — BLOCK the run until `/review` is available. `/review` is a hard prerequisite for AC-9. Diagnose and resolve unavailability (auth, network, skill install) before resuming the loop; no unstructured-review fallback is permitted.

- DEC-4: If H100 GPUs 2 and 3 are not free, what is the fallback?
  - Claude Position: Try the next two consecutive idle H100 GPUs (lowest IDs first); abort only if no two are free.
  - Codex Position: N/A — open question.
  - Tradeoff Summary: Hard-coding GPU 2/3 fails fast but is brittle on shared hosts; auto-select is robust but may pick GPUs with thermal contention.
  - Decision Status: RESOLVED — Auto-pick the next free consecutive H100 pair (lowest IDs first). Record the chosen `CUDA_VISIBLE_DEVICES` values and the SGLang-logged logical CUDA IDs in the final summary. Abort only if no two GPUs are free.

- DEC-5: Which voice_clone modes are required for AC-11 and AC-12?
  - Claude Position: `voice_clone=False` is the gating mode for both EN and ZH (matches the draft's example commands, which omit `--voice-clone`); `voice_clone=True` runs are optional and recorded if executed, but not required for acceptance. This commits to two gating cells (EN/VC=F, ZH/VC=F) rather than four.
  - Codex Position: Either both modes gating with four output dirs, or one selected mode gating and the other optional — pick one consistently.
  - Tradeoff Summary: Two gating cells halve the benchmark time vs four-cell coverage but miss regressions specific to the reference-audio code path. The draft's command shape signals VC=F is the canonical run, so two-cell gating + optional VC=T is the lowest-friction commitment.
  - Decision Status: RESOLVED — `voice_clone=True` is the gating mode for both EN (AC-11) and ZH (AC-12). The benchmark generate command must include `--voice-clone`. Reference WER becomes H100/VC=T: EN `wer_corpus` reference 1.86%, ZH `wer_corpus` reference 1.49% (from `benchmarks/eval/benchmark_omni_seedtts.py:65, :67`). VC=False runs are not required.

- DEC-6: Should the plan require underscore CLI aliases (e.g. `--mem_fraction_static`) in addition to dashed spellings, matching the existing V1 CLI convention?
  - Claude Position: Do not require — out of validation scope unless PR #426 already includes them. If PR omits them and `/review` flags it, fix in scope.
  - Codex Position: N/A — open question.
  - Tradeoff Summary: Adding aliases keeps V1 CLI consistent but turns this run from "validate" into "redesign"; the draft explicitly forbids unrelated refactor.
  - Decision Status: RESOLVED — Out of scope. Aliases are NOT a validation criterion. If `/review` raises them as a gating finding (correctness / regression), address in-scope; otherwise leave untouched.

- DEC-7: When `--encoder-mem-reserve` is omitted, should the thinker factory apply a default reserve, or no reserve?
  - Claude Position: Accept PR #426's current behavior — `sglang_omni_v1/models/qwen3_omni/stages.py` sets `encoder_mem_reserve: float = 0.05` in the thinker factory signature, so the SGLang auto value is shaved by 0.05 even when the CLI flag is absent. The validation's job is to confirm PR behavior, not redesign it; the V0 engine uses the same default and the same test pattern (see `tests/test_mem_fraction_static.py::test_factory_uses_default_reserve_when_omitted`).
  - Codex Position: Either "no reserve unless CLI/config sets it" or PR's current default `0.05`; pick one and align AC-6, tests, and benchmark expectations.
  - Tradeoff Summary: Defaulting to `0.05` mirrors V0 semantics and prevents long-video encoder OOM out of the box, at the cost of slightly less KV cache headroom on workloads that do not need encoder reserve. Defaulting to `0` (no reserve) maximizes KV cache headroom but changes V0→V1 behavior for Qwen3 thinker.
  - Decision Status: RESOLVED — Accept PR #426's `encoder_mem_reserve: float = 0.05` thinker-factory default. AC-6 and the H100 server-launch log capture must show `mem_fraction_static = round(SGLang_auto − 0.05, 3)` on the thinker stage when no explicit CLI/config pin is present.

- DEC-8: The RFC asks for the override-APPLICATION primitive to live at the PipelineConfig layer; PR #426 leaves it at the CLI layer (`_apply_stage_server_args_override` in `sglang_omni_v1/cli/serve.py`). Should PR #426 expand scope to port `apply_server_args_overrides` from V0's `sglang_omni.config.schema.PipelineConfig` to V1's `sglang_omni_v1.config.schema.PipelineConfig`?
  - Claude Position: Defer. PR #426's stated scope is hardware-aware memory autosizing; adding a config-layer method would widen the diff and the review surface. Track as a follow-up PR motivated by the RFC's "one canonical mechanism" goal once the next runtime param (`thinker_max_seq_len`, `video_fps`) needs the same plumbing.
  - Codex Position: N/A — RFC-driven question newly surfaced.
  - Tradeoff Summary: Porting now would close the RFC gap in one PR and avoid future contributors discovering they must reach into the CLI layer for stage overrides. Deferring keeps PR #426 focused on its memory-management contract and protects the validation timeline. The misalignment is non-contradictory; the project can defer without violating the RFC.
  - Decision Status: RESOLVED — Defer to a follow-up PR. PR #426's scope stays as memory autosizing; no schema edits to V1's `PipelineConfig` are permitted in this validation run. The follow-up motivation is recorded in the final summary (AC-13) as a known RFC follow-up item, motivated by the RFC's "one canonical mechanism" goal.

- DEC-9: The RFC's vocabulary is stage-keyed (e.g. `stage="talker_ar"`); PR #426 introduces a role-keyed CLI surface (`--thinker-mem-fraction-static`, `--talker-mem-fraction-static`) and uses `mem_fraction_role_to_stage` to translate. Should the role layer be kept, replaced with pure stage-keyed flags, or treated as a CLI-only convenience that does NOT enter the RFC's vocabulary?
  - Claude Position: Keep roles as a CLI-only convenience and explicitly document them as such. Roles solve a real UX problem: a user does not need to know whether the talker stage is named `talker_ar` or `talker_mtp`. The RFC's stage vocabulary remains canonical at the config/schema layer.
  - Codex Position: N/A — RFC-driven question newly surfaced.
  - Tradeoff Summary: Keeping roles preserves CLI ergonomics but adds a translation step. Stripping roles in favor of stage-keyed flags (e.g. `--stage talker_ar mem_fraction_static=0.65`) matches the RFC's vocabulary but produces a more verbose CLI. A third option — treat `--thinker-mem-fraction-static` as sugar that desugars to a stage-keyed override after parsing — is what PR #426 already does.
  - Decision Status: RESOLVED — Keep roles as CLI-only sugar. PR #426's role-keyed flags are accepted as the public CLI surface. The schema layer (`PipelineConfig`, `StageConfig`) stays stage-keyed; `mem_fraction_role_to_stage` is the translation boundary. AC-14 verifies the schema layer does not gain a parallel role-keyed primitive.

## Implementation Notes

### Code Style Requirements
- Implementation code, comments, and docstrings must NOT contain plan-specific workflow terminology such as `AC-`, `Milestone`, `Step`, `Phase`, or `DEC-`. Those identifiers belong only in this plan document.
- Use domain-appropriate names: e.g. `mem_fraction_role_to_stage`, `apply_encoder_mem_reserve`, `_resolve_factory_args`. Do not coin new public APIs solely for this validation.
- Comments are reserved for the non-obvious "why" (e.g. the floor check on `apply_encoder_mem_reserve`, the runtime-override precedence quirk in `_resolve_factory_args`); routine "what" comments are forbidden.
- Tests must reuse existing fixtures and helper patterns from `tests/test_mem_fraction_static.py` rather than introduce parallel infrastructure.
- This is a validation run — when in doubt between "patch the code" and "record the finding for user review", prefer the latter.

--- Original Design Draft Start ---

# [V1] 在 H100 上验证 hardware-aware mem_fraction_static autosizing

## 目标
PR #426 已经基本包含了预期的 V1 `mem_fraction_static` 设计。本次 Humanize 运行的目标不是从零重新 port 这个 PR，而是严格验证当前本地分支：通过 Codex review 和 Claude Code 修复不断循环，直到 review 没有问题；之后再跑 H100 benchmark，直到实现同时达到 review-clean 和 benchmark-clean。

如果 review 或 benchmark 结果暴露 bug，只修复这个 bug 并继续循环。除非证据表明当前实现存在结构性错误，否则不要做大范围重写。

## 需要保持的参考行为
- V1 `build_sglang_server_args` 不应该强制 `mem_fraction_static=0.7`；默认行为应该允许 SGLang main 自己选择硬件感知的值。
- 用户显式设置 `mem_fraction_static` 时，显式 pin 仍然应该被传递下去。
- Qwen3-Omni V1 只应该为支持的公开 role 暴露 memory override：
  - text pipeline: `{"thinker": "thinker"}`
  - speech pipeline: `{"thinker": "thinker", "talker": "talker_ar"}`
- base/non-Qwen pipeline 不应该意外继承通用的公开 memory API。
- V1 CLI 应该接受并验证：
  - `--mem-fraction-static`
  - `--thinker-mem-fraction-static`
  - `--talker-mem-fraction-static`
  - `--encoder-mem-reserve`
- per-role 值应该覆盖 global fallback。
- `--encoder-mem-reserve` 只在 thinker auto-memory 路径上有效。
- 当 global 或 thinker memory 已经被 CLI、config 或 runtime overrides 显式 pin 时，必须拒绝 `--encoder-mem-reserve`。
- legacy 入口 `sgl-omni serve --version v1 ...` 应该转发 V1 memory flags，而不是拒绝它们。

## Humanize Review 循环
使用 Humanize 作为编排循环，并让 Codex 使用它可用的 `/review` agent 作为 review 标准：

1. 运行 focused tests。
2. 让 Codex 检查自己是否有可用的 `/review` agent。
3. 如果 `/review` 可用，让 Codex 使用这个 agent 做 review。
4. 让 Claude Code 修复 `/review` 提出的每一个有效 finding。
5. 每轮修复后，再让 Codex 使用 `/review` 重新检查。
6. 重复上述流程，直到 `/review` 不再报告未解决的 correctness、regression、security、performance 或 missing-test 问题。

注意：Codex 应该先检查自己可用的 agents/skills；如果存在 `/review`，就使用它。任何 `/review` 未解决 finding 都应该视为 blocking。

## 给 Humanize 的 Prompt

使用下面这段作为 Humanize 的任务输入：

```markdown
我们正在 H100 本地验证 PR #426 的 V1 hardware-aware `mem_fraction_static` 工作。不要假设这需要重新 port。先检查当前本地分支，并和 PR #426 的预期行为对照。如果实现已经匹配，就把重点放在测试、review 和 benchmark 验证上。

必须执行的循环：
1. 运行 V1 memory override 行为的 focused unit tests。
2. 让 Codex 检查自己可用的 agents/skills 里是否有 `/review`。
3. 如果 `/review` 可用，让 Codex 使用 `/review` 做 review。
4. 用最小、定向的改动修复每一个有效的 `/review` finding。
5. 每轮修复后，再让 Codex 使用 `/review` 复查。
6. 重复直到 `/review` 没有 unresolved findings。
7. 运行 H100 SeedTTS benchmarks。如果 benchmark 的精度、失败率或 memory 行为有问题，诊断、修复，并重新执行 review 和 benchmark 循环。

参考行为：
- V1 默认不能强制 `mem_fraction_static=0.7`。
- V1 AR stages 的默认行为应该允许 SGLang 进行硬件感知 autosizing。
- 显式 global/per-role memory pin 必须仍然有效。
- Qwen text 支持 thinker memory override。
- Qwen speech 支持 thinker 和 talker_ar memory overrides。
- Non-Qwen pipelines 应该拒绝公开 memory flags。
- `--encoder-mem-reserve` 只应用在 thinker auto path，并且和显式 thinker memory pin 互斥。
- Legacy `sgl-omni serve --version v1` 应该转发 V1 memory flags。

不要做无关 refactor。只有当测试能验证公开行为或防止真实 regression 时，才添加或调整测试。
```

## 需要检查的文件
- `sglang_omni/cli/serve.py`
- `sglang_omni_v1/cli/serve.py`
- `sglang_omni_v1/config/schema.py`
- `sglang_omni_v1/models/qwen3_omni/config.py`
- `sglang_omni_v1/models/qwen3_omni/stages.py`
- `sglang_omni_v1/scheduling/sglang_backend/server_args_builder.py`
- `sglang_omni_v1/scheduling/sglang_backend/__init__.py`
- `sglang_omni_v1/utils/misc.py`
- `tests/test_v1_cli_version_dispatch.py`
- `tests/test_v1_mem_fraction_static.py`
- `benchmarks/eval/benchmark_omni_seedtts.py`

## 验证命令
先运行 focused tests：

```bash
pytest -q tests/test_v1_mem_fraction_static.py tests/test_v1_cli_version_dispatch.py
```

focused tests 通过后，再运行更广的 V1 tests：

```bash
pytest -q tests/test_v1_*.py
```

测试修复完成后，如果 Codex 报告 `/review` 可用，就通过 Humanize/Codex 使用 `/review` 做 review。review-clean 的定义是 `/review` 没有 unresolved findings。

## H100 Benchmark
优先使用 H100 GPU 2 和 3，除非它们不可用。推荐使用 sequential generate/transcribe，这样可以把 server memory 行为和 ASR memory 使用隔离开。

启动 server：

```bash
CUDA_VISIBLE_DEVICES=2,3 python -m sglang_omni.cli serve \
  --model-path Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --version v1 \
  --port 8000
```

如有需要，准备数据集：

```bash
python -m benchmarks.dataset.prepare --dataset seedtts
```

运行 English full-set generation：

```bash
python -m benchmarks.eval.benchmark_omni_seedtts \
  --generate-only \
  --meta seedtts_testset/en/meta.lst \
  --output-dir results/qwen3_omni_h100_en_mem_auto \
  --max-concurrency 16 \
  --model qwen3-omni \
  --port 8000
```

运行 English transcription 和 WER：

```bash
python -m benchmarks.eval.benchmark_omni_seedtts \
  --transcribe-only \
  --meta seedtts_testset/en/meta.lst \
  --output-dir results/qwen3_omni_h100_en_mem_auto \
  --model qwen3-omni \
  --lang en \
  --device cuda:0
```

如果 English 通过，再用 `seedtts_testset/zh/meta.lst` 和单独的 output directory 跑 Chinese。

## 验收标准
- Focused 和相关 V1 tests 通过。
- Codex 检查了 `/review` 是否可用，并在可用时使用它做 review。
- `/review` 没有 unresolved findings。
- H100 server 能够用 V1 auto memory 行为成功启动。
- H100 SeedTTS benchmark 完成，failed requests 为 0。
- H100 WER 保持在 `benchmarks/eval/benchmark_omni_seedtts.py` 里的现有参考区间内；任何有意义的 regression 都会触发下一轮 fix/review/benchmark 循环。
- 最终总结包含测试命令、review 状态、H100 benchmark 命令行、WER、失败数量、throughput，以及相对 H200/H100 参考结果的任何偏差。

--- Original Design Draft End ---
