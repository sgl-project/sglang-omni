# MLX Path

Rules for Apple Silicon work. They write down what the native runtime PRs under
[#2537](https://github.com/sgl-project/sglang-omni/issues/2537) already do.

## Layout

| Path | Contents |
| --- | --- |
| `sglang_omni_mlx/native/` | C++ runtime on MLX: one server binary per model, a parity CLI per model in `tools/`, shared code in `src/layers`, `src/swift_port`, and the model services. |
| `sglang_omni_mlx/native/ci/` | Model pins, the 392-clip corpus, golden outputs, server API tests, and model evaluation tools. |
| `sglang_omni_mlx/<model>/` | Optional Python MLX reference implementations used for parity; shipped servers belong under `sglang_omni_mlx/native/`. |
| `Voxt/` | The macOS app that runs the native servers. See [`Voxt/backend/README.md`](https://github.com/sgl-project/sglang-omni/blob/main/Voxt/backend/README.md). |

## Rule 1: MLX is a separate path

MLX code is not abstracted together with the CUDA path in `sglang_omni`.

- New MLX models, kernels and servers go under `sglang_omni_mlx/`.
- Do not add MLX branches, workers or platform hooks to `sglang_omni/`.
- No shared backend interface, base class or registry across the two paths.
  Share code freely inside the MLX path, not across paths.
- The paths meet only at the OpenAI-style HTTP API (`/v1/audio/transcriptions`,
  `/v1/audio/speech`, `/v1/realtime`, `/health`).

Why:

- **Runtimes differ.** `sglang_omni` is a multi-process CUDA pipeline on
  PyTorch; the MLX path is one process per model on unified memory, shipped as
  C++ binaries. An interface over both hides what makes each fast.
- **Dependencies differ.** SGLang's wheel pulls CUDA. Keeping MLX out lets Voxt
  run without CUDA, PyTorch or Python.
- **CI differs.** Mac CI only runs on `Voxt/**` and `sglang_omni_mlx/**`, and
  GPU CI never runs on a Mac. A shared layer would break one side without CI
  noticing.

The legacy MLX code in `sglang_omni` (`mlx_model_worker.py`,
`MlxTpModelWorker`, the per-model `mlx/` packages) is removed in
[#2725](https://github.com/sgl-project/sglang-omni/pull/2725); do not add it
back. If a change really needs a cross-path abstraction, open an issue and
agree on it with a maintainer first.

## Rule 2: Contribution requirements

**Placement and reuse**

- A native ASR model adds `<model>_model`, `<model>_transcriber`, a server on
  the shared ASR service, a CLI in `tools/`, and CMake targets.
- A native TTS model adds `<model>_model`, a model-specific TTS service, a
  parity or evaluation CLI in `tools/`, and CMake targets. TTS services share
  the HTTP and supervisor lifecycle code but do not extend the ASR service.
- Reuse `layers`, `swift_port`, the Sortformer front end and the Silero VAD
  port. If your code duplicates something on `main`, move it into the shared
  file in the same PR.
- Voxt changes stay in `Omni*.swift`, `VoxtTests/Omni*` and `Voxt/backend/`,
  plus minimal routing edits.

**Parity**

- Name the reference (Voxt's Swift backend at a pinned mlx-audio-swift commit,
  or a Python MLX reference) and use the same weights and MLX version.
- Keep the reference's behavior. Behavior changes go in a separate PR.
- Report English WER, Chinese CER and mixed MER on the 392-clip corpus,
  reference vs native, plus the count of identical clips. Explain where
  differing clips diverge.
- For TTS, compare fixed-seed generated tokens with the named reference, then
  report transcription error through a pinned ASR model and speaker similarity
  on a declared evaluation set. Sampling differences do not replace a parity
  check; record the seed, checkpoint revisions, and MLX version.

**Golden and pins**

- Pin every checkpoint file by revision and SHA-256 in `ci/model_pins.json`.
- Add a golden under `ci/model_golden/`. Import the Mac CI runner's outputs
  (`golden-outputs` artifact, `--import`) so the runner checks exactly.
- Add a second golden for paths the corpus misses (long-audio chunking, VAD).

**Tests**

- Server tests in `ci/test_model_servers.py`: JSON, SSE and realtime success
  paths; every invalid input returns 400; session-state errors, cancel on
  disconnect, and wrong model kind.
- TTS server tests cover WAV and PCM responses, streaming, invalid request
  fields, busy and shutdown states, and cancellation after a client disconnect.
- Assert structure, not one chip's exact text; exact text lives in the golden.
- New Voxt Omni tests go into the `voxt-unit` job in `voxt-mac-ci.yaml`.

**Robustness**

- Check the checkpoint config at load, so bad checkpoints fail at load, not
  mid-request.
- Validate requests before they reach the worker.
- Streaming sessions use bounded memory: trim consumed audio, cap session
  length, clear the MLX cache.

**Performance**

- Mac CI checks correctness only. Report total time, realtime factor, peak
  memory and load time against the reference, with the device stated.

**CI and style**

- Target `main` (stacked PRs get no Mac CI), non-draft, with the `run-mac-ci`
  or `run-all-ci` label. Both Mac CI jobs must pass on a real run.
- Run pre-commit and follow
  [coding-style.md](https://github.com/sgl-project/sglang-omni/blob/main/.claude/skills/code-review/coding-style.md).
- Update `Voxt/backend/README.md`. Follow the PR format of
  [#2676](https://github.com/sgl-project/sglang-omni/pull/2676) or
  [#2689](https://github.com/sgl-project/sglang-omni/pull/2689).
