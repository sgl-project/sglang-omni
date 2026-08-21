# PR #1209 rebase audit

Date: 2026-08-20

## Rebase inputs

- Original PR: `JiaxinD/sglang-omni:perf/shared-chain-graph-harness`
- Original PR head: `e6a6d8b9a0d1d1341426a7f764dfc3b872e0218a`
- Original merge base: `7e837c22420c32e8894756feabb1dbfb224ab8fc`
- Rebasing onto: `upstream/main` at `a5249369d1bea7296c5c359d7334b90d340cc615`
- Rebasing worktree: `/private/tmp/sglang-omni-pr1209-rebase`
- Protected original head: `backup/pr1209-pre-rebase`

## Conflict and semantic-resolution log

### `2330d87b` — shared cache and persistent state

Conflicts:

- `sglang_omni/models/moss_tts_local/sglang_model.py`
- `sglang_omni/models/qwen3_omni/components/talker.py`
- `sglang_omni/models/qwen3_tts/sglang_model.py`
- `tests/unit_test/qwen3_omni/test_talker.py`
- `tests/unit_test/qwen3_tts/test_predictor_cuda_graph.py`

Resolution:

- MOSS retained the current-main compile configuration and adopted the shared
  `KeyedGraphCache`/`PersistentStateRegistry` integration.
- Qwen3-Omni talker and its test were kept at current-main. Upstream has a
  newer per-bucket graph map, GQA path, lifecycle, and failure handling than
  the PR's proposed shared-cache migration.
- Qwen3-TTS predictor and its test were kept at current-main. Upstream's
  custom predictor graph implementation includes newer sampling, staging,
  failure-fuse, and graph-gating behavior that the PR version would regress.

### `46714611` — Higgs codec shared runner

Current main already contains a newer direct Higgs codec CUDA-graph
implementation with frame-count configuration, capture accounting, single-flight
capture, capture-safe quantization, and eager fallback. The older shared
`HiggsVocoderCudaGraphRunner` proposal was therefore dropped rather than
reintroduced over that implementation.

Dropped as superseded:

- `sglang_omni/models/higgs_tts/vocoder_cuda_graph.py`
- `tests/unit_test/higgs_tts/test_vocoder_cuda_graph.py`
- `tests/unit_test/higgs_tts/test_vocoder_cuda_graph_real_model.py`

The current-main Higgs codec and stage wiring were preserved. The later test
double warmup-hook fix remains because current stage setup still calls that
hook.

### `c8e27d66` — Higgs class default and CUDA guards

The class-level `_cg_runner` default was stale against current main's direct
codec graph implementation and was dropped. The valid CUDA-only skip guards in
the Voxtral frame-noise test and Zonos tail-runner test were retained.

### `bd79aeae` — Higgs pipeline test doubles

Replayed without conflict. The three codec test doubles now provide the warmup
hook expected by the current pipeline setup.

### `7253cbd6` — Whisper and Code2Wav cache adoption

The current-main Code2Wav implementation was preserved. Its tier-0/tier-1
capture policy, memory budget, retry/shrink behavior, atomic publication,
process ownership checks, borrowed-output contract, and eager fallback are
newer than the PR's all-or-nothing `KeyedGraphCache` migration. Code2Wav cache
migration is deferred.

The generic cache enhancements (`pool_factory`, stable graph view, and clear)
were retained because they are independently used by the surviving shared-cache
integrations. The PR's final `e6a6d8b9` revert was replayed and removes the
Whisper encoder cache adoption, as intended by the original PR series.

### Post-rebase cleanup

The PR-only
`tests/unit_test/qwen3_omni/test_talker_predictor_graph_gates.py` was removed.
It asserted the abandoned `_predictor_graph_cache` API and would have created a
false contract against current main's preserved talker implementation.

## Resulting scope

Retained integrations are the shared CUDA-graph cache/state primitives for MOSS
TTS Local and Voxtral TTS, the Zonos2 keyed tail graphs, the existing MOSS
vocoder state-address checks, and the Ming TTS eager fallback for uncovered
batches. Current-main implementations are preserved for Qwen3-Omni talker,
Qwen3-TTS predictor, Qwen3-Omni Code2Wav, Higgs codec graphs, and Whisper
encoder cache behavior.

No PR body was rewritten and no new PR was created.

## Validation

- `git diff --check upstream/main...HEAD`: passed.
- `python -m compileall -q sglang_omni`: passed.
- Touched test files: Python compilation passed.
- Targeted pytest collection was attempted, but this macOS environment has no
  PyTorch installation. The default Python is 3.14, while the project requires
  Python 3.10–3.12. Therefore no pytest pass result is claimed here; CUDA
  runtime validation remains pending in a project-compatible CUDA environment.
