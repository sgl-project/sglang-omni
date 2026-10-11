# MLX Path Policy and Contribution Requirements

This page sets two rules for Apple Silicon work in SGLang-Omni:

1. MLX is a separate path. It is not abstracted together with the CUDA path.
2. MLX contributions meet the bar described below before they merge.

Both rules follow what the merged native runtime PRs under the Voxt roadmap
([#2537](https://github.com/sgl-project/sglang-omni/issues/2537)) already do.
Where this page states something that is new, it says so.

## Where MLX code lives

| Path | What it is |
| --- | --- |
| `sglang_omni_mlx/native/` | The native runtime: C++ on MLX, no Python at run time. One server binary per model (`qwen3_asr_server`, `whisper_server`, `cohere_transcribe_server`, `moss_transcribe_diarize_server`, ...) and one parity CLI per model under `tools/`. Shared code lives in `src/layers.{h,cpp}`, `src/swift_port.{h,cpp}`, `src/asr_service.{h,cpp}` and similar files. |
| `sglang_omni_mlx/native/ci/` | Model pins (`models.json`, `model_pins.json`), the frozen 392-clip corpus, golden outputs (`golden/`, `model_golden/`) and the server API tests. |
| `sglang_omni_mlx/<model>/` | Python MLX servers. `sglang_omni_mlx/qwen3_asr` is the reference server the native Qwen3-ASR runtime matches. Shared Python pieces are `serving.py`, `text_decoder.py`, `checkpoint.py`, `transcription.py` and `wav.py`. |
| `Voxt/` | The macOS app. It starts and owns the native server over local HTTP and WebSocket. The client is `Voxt/Voxt/Transcription/Omni*.swift`; build, run and CI notes are in [`Voxt/backend/README.md`](https://github.com/sgl-project/sglang-omni/blob/main/Voxt/backend/README.md). |
| `sglang_omni/` | The CUDA serving framework. It is not part of the MLX path. |

## Rule 1: MLX is a separate path

`sglang_omni_mlx` is MLX-only serving, independent of the CUDA pipeline in
`sglang_omni`. It does not import `sglang_omni`, `sglang` or `torch`, and the
CUDA package does not need to know that it exists.

Going forward:

- New MLX models, kernels and servers go under `sglang_omni_mlx/`.
- Do not add MLX branches, MLX worker types or MLX platform hooks to
  `sglang_omni/` (platforms, model runners, engine factories, stage pipelines,
  model `engine_builder.py` files).
- Do not introduce a shared backend interface, base class or registry that
  both paths implement.
- Code is shared inside the MLX path freely (see Rule 2). Code is not shared
  across the two paths.
- The two paths meet at the HTTP API. MLX servers expose the same
  OpenAI-style routes (`/v1/models`, `/v1/audio/transcriptions` with JSON or
  SSE, `/v1/realtime`, `/health`), so clients do not care which path serves
  them. That contract is the only coupling.

### Why

- **Different runtimes.** `sglang_omni` is built on SGLang's CUDA stack:
  multi-process stage pipelines, per-stage schedulers, RadixAttention, CUDA
  graphs, zero-copy shared memory, PyTorch models. The MLX path is one
  process per model on unified memory, and the native runtime ships as C++
  binaries with no Python. An interface that fits both has to hide most of
  what makes each one fast.
- **Different dependencies.** SGLang's PyPI wheel has unconditional CUDA
  dependencies, so Apple Silicon has to install SGLang from source to load
  `sglang_omni` at all. Keeping MLX outside `sglang_omni` lets Voxt run a
  model without the CUDA stack, PyTorch or a Python runtime.
- **Different CI.** CUDA changes are tested on Linux GPU runners. MLX changes
  are tested by `Voxt Mac CI` on the Apple Silicon runner, which only
  triggers on `Voxt/**` and `sglang_omni_mlx/**`. A layer shared by both
  paths would be changed by CUDA PRs that never run on a Mac, and by MLX PRs
  that never run on a GPU. Each side would break the other without CI
  noticing.
- **Different correctness targets.** MLX models match a reference on Apple
  hardware (Voxt's Swift backend or a Python MLX reference), checked per chip
  against golden outputs. CUDA models match their PyTorch reference on GPUs.
  One abstraction cannot serve both checks.
- **Cost of change.** With a shared abstraction, every CUDA refactor has to
  consider MLX and every MLX model has to fit CUDA's shapes. Separate paths
  let each side move at its own pace.

### Existing MLX code inside `sglang_omni`

`sglang_omni` already has some Apple code from before this policy:
`sglang_omni/platforms/apple.py`, `sglang_omni/model_runner/mlx_model_worker.py`
and `audio_mlx.py`, the `MlxTpModelWorker` path in
`sglang_omni/scheduling/engine_factory.py`, and `MlxTpModelWorker` types in most model
`engine_builder.py` files. This policy does not remove them. New work does
not extend them; a model that needs MLX gets an `sglang_omni_mlx` server
instead. Moving or removing the existing code is a separate decision for its
owners.

This page covers MLX. Running `sglang_omni`'s PyTorch models on the MPS device
is a different question and is not decided here.

If you think a change really needs a shared abstraction across the two paths,
open an issue and get a maintainer's agreement before writing code (new
policy).

## Rule 2: Requirements for MLX contributions

### Scope and placement

- Code goes in the places listed above. A native model adds
  `src/<model>_model.{h,cpp}`, `src/<model>_transcriber.{h,cpp}`, a server
  (or a new model kind on the shared server), a parity CLI under `tools/`,
  and its CMake targets.
- One server per model kind, built on the shared server code
  (`asr_service` natively, `serving.py` in Python). Register as a kind
  instead of copying handlers, SSE, argument parsing or the supervisor
  protocol.
- No changes to `sglang_omni/` are needed for an MLX model. If yours needs
  one, see Rule 1.
- Voxt changes stay in the Omni files (`Voxt/Voxt/Transcription/Omni*.swift`,
  `Voxt/VoxtTests/Omni*`, `Voxt/backend/`) and the smallest routing changes in
  existing Voxt files. The rest of `Voxt/` is the upstream import.

### Reuse shared code

- Before writing a layer, check `src/layers.{h,cpp}`, `src/swift_port.{h,cpp}`,
  `src/speech_segments.{h,cpp}`, the Sortformer feature extractor and the
  Silero VAD port. Reuse them.
- When your code repeats something that already exists elsewhere (a relative
  shift, an LSTM step, a SentencePiece decoder, a VAD), move it into the
  shared file in the same PR and switch the existing callers. Do not land a
  second copy and promise a follow-up.
- Build on the PR that introduces a component instead of adding a parallel
  copy that will conflict with it.

### Parity with the reference

- Name the reference: Voxt's Swift backend (with the mlx-audio-swift commit),
  or a Python MLX reference written from the reference sources rather than
  from your port.
- Use the same weights and the same MLX version. The native build pins MLX
  exactly in `sglang_omni_mlx/native/CMakeLists.txt`.
- Keep the reference's behavior: request fields, chunking, language handling,
  stop rules, output segments. Changing behavior (for example, a better
  chunking strategy) is a separate PR and a separate decision.
- Report accuracy on the frozen 392-clip corpus with the language each clip is
  sent with: English WER, Chinese CER and mixed MER, reference against native,
  with the delta in percentage points. Omit metrics the model does not support
  and say why.
- Report how many clips are textually identical to the reference. When clips
  differ, find the first point of divergence (front end, encoder layer,
  decoder step) and explain it. Accuracy on par is not enough on its own.

### Golden outputs and model pins

- Pin the checkpoint (and tokenizer, if separate) by revision with a SHA-256
  for every file in `ci/model_pins.json` (or `ci/models.json`).
- Add a golden file under `ci/model_golden/` (or `ci/golden/`) with the
  reference baseline metrics, the tolerance, and per-chip outputs.
- Greedy decoding follows each chip's GPU arithmetic, so the gate is exact only
  on chips recorded in the golden file. Import the Mac CI runner's outputs
  (the `golden-outputs` artifact, via `--import`) so the runner checks
  exactly instead of falling back to the error-rate tolerance.
- If the corpus does not reach a code path (long-audio chunking, VAD
  segmentation, multi-chunk token budgets), add a second golden that does,
  for example with a shorter chunk duration.
- If a golden freezes known bad outputs (loops that hit the token limit, a
  missing language detection), say so in the PR so it is regenerated when the
  behavior is fixed.

### Tests

- Native servers: API tests in `ci/test_model_servers.py` (or the matching
  `test_*_api.py`). Cover:
  - the success path, JSON and SSE, and realtime when the model streams;
  - realtime at the native chunk giving the Final result, where applicable;
  - every invalid input returning 400 (`invalid_request`), not a worker
    failure: negative or non-finite numbers, bad chunk sizes, bad prompts,
    invalid UTF-8, corrupt checkpoint files;
  - session state errors (settings after audio, audio after commit) and
    cancel on disconnect;
  - a server rejecting a model kind it does not serve.
- Assert structure (speaker count, ordering, text prefix), not one chip's
  exact text. Exact text belongs in the golden file.
- Python MLX servers: unit tests under `tests/unit_test/mlx_<model>/` guarded
  by `pytest.importorskip("mlx.core")`. No CI job runs these today (they skip
  on Linux), so run them on Apple Silicon and include the result in the PR.
- Voxt: Omni unit tests for new request, routing and failure behavior, added
  to the `voxt-unit` job's list in `voxt-mac-ci.yaml`.

### Robustness

- Validate the checkpoint config at load time (vocabulary sizes, head
  divisibility, required fields) so an unsupported checkpoint fails at load,
  not with an out-of-bounds write mid-request.
- Validate requests at the boundary, before work is submitted to the worker.
- Bounds-check every read of an untrusted file or payload (npz directories,
  UTF-8, Base64).
- Streaming and realtime sessions use bounded memory: trim consumed audio,
  recompute only the new tail, cap session length, and clear the MLX buffer
  cache as the reference does. A one-hour session must not grow without
  limit or cost O(N^2).
- Keep resident models bounded (for example, at most one VAD model at a time).
- Check cancellation before expensive work, and honor the startup deadline.

### Performance

- Mac CI checks correctness only. Measure performance by hand against the
  reference on the same 392 clips, for example with Voxt's
  `OmniPhase1BenchmarkTests`.
- Report the device (chip and memory), total time, realtime factor, peak
  memory, idle memory after load, and model load time.

### CI

- Open the PR against `main`. `Voxt Mac CI` only runs on PRs into `main`, so a
  PR stacked on another branch is never checked.
- Mac CI needs a non-draft PR and the `run-ci` label (or
  `/tag-and-rerun-ci`).
- Both jobs, `native runtime (golden + API)` and `Voxt Omni unit tests`, must
  pass on a real run. A run that died at checkout or provisioning does not
  count. The Linux GPU CI does not test MLX code.

### Style and documentation

- Run `pre-commit run --all-files`: black 24.10.0, isort and ruff for Python,
  clang-format 18.1.8 for C++.
- Follow [coding-style.md](https://github.com/sgl-project/sglang-omni/blob/main/.claude/skills/code-review/coding-style.md):
  full-word names that state the unit, sparse comments that explain why,
  signed as `// Note (Your Name): ...` in C++ and `# Note (Your Name): ...`
  in Python.
- Update [`Voxt/backend/README.md`](https://github.com/sgl-project/sglang-omni/blob/main/Voxt/backend/README.md):
  the checkpoint table, the binaries, the CI notes and known limitations.

### PR description

Use the repository's PR template. For a native model:

- Title: `[Voxt] Run <model> on the native runtime`.
- Related Issues: `Part of #2537`.
- Modifications: native runtime, Voxt and CI changes listed separately.
- Accuracy Test: the corpus, the language sent per clip, the metric table, the
  golden result with the chip, Voxt end to end against the CLI, and the
  identical-clip count with the divergence analysis.
- Benchmark & Profiling: the performance table with the device stated.

[#2676](https://github.com/sgl-project/sglang-omni/pull/2676),
[#2688](https://github.com/sgl-project/sglang-omni/pull/2688) and
[#2689](https://github.com/sgl-project/sglang-omni/pull/2689) are examples of
this format.
