# MLX Path

```{toctree}
:hidden:

long_audio
```

Rules for Apple Silicon work. They write down what the native runtime PRs under
[#2537](https://github.com/sgl-project/sglang-omni/issues/2537) already do.

## Layout

| Path | Contents |
| --- | --- |
| `sglang_omni_mlx/native/` | C++ runtime on MLX: one server binary per model, a parity CLI per model in `tools/`. Each model's code is in `src/<model>/`; code shared across models (`audio`, `form`, `layers`, `npz`, `reference_audio`, `tokenizer`, built as the `common` library, plus `swift_port` and `http`) is in `src/common/`. The shared ASR server code (`asr_service`, `worker`, `realtime`) is still in `src/qwen3_asr/`, and so is the `/v1/audio/speech` API that `voxtral_tts_server`, `fishaudio_drama3_server` and `fish_s2_pro_server` share (`speech_service`), built on `asr_service`. `ming3_audio_server` serves `/v1/chat/completions` from its own service in `src/ming3_audio/`. |
| `sglang_omni_mlx/native/ci/` | Model pins, the 392-clip corpus, golden outputs, server API tests and the [long-recording benchmark](long_audio.md). |
| `sglang_omni_mlx/<model>/` | Python MLX servers, e.g. the Qwen3-ASR reference server. |
| `Voxt/` | Voxt, the macOS app that runs the native servers. See [`Voxt/README.md`](https://github.com/sgl-project/sglang-omni/blob/main/Voxt/README.md). |

## Rule 1: MLX is a separate path

MLX code is not abstracted together with the CUDA path in `sglang_omni`.

- New MLX models, kernels and servers go under `sglang_omni_mlx/`.
- Do not add MLX branches, workers or platform hooks to `sglang_omni/`.
- No shared backend interface, base class or registry across the two paths.
  Share code freely inside the MLX path, not across paths.
- The paths meet only at the OpenAI-style HTTP API (`/v1/audio/transcriptions`,
  `/v1/audio/speech`, `/v1/chat/completions`, `/v1/realtime`, `/health`).

Why:

- **Runtimes differ.** `sglang_omni` is a multi-process CUDA pipeline on
  PyTorch; the MLX path is one process per model on unified memory, shipped as
  C++ binaries. An interface over both hides what makes each fast.
- **Dependencies differ.** SGLang's wheel pulls CUDA. Keeping MLX out lets Voxt
  run without CUDA, PyTorch or Python.
- **CI differs.** Mac CI only runs on `sglang_omni_mlx/**` and the app's Swift packages, and
  GPU CI never runs on a Mac. A shared layer would break one side without CI
  noticing.

The legacy MLX code in `sglang_omni` (`mlx_model_worker.py`,
`MlxTpModelWorker`, the per-model `mlx/` packages) is removed in
[#2725](https://github.com/sgl-project/sglang-omni/pull/2725); do not add it
back. If a change really needs a cross-path abstraction, open an issue and
agree on it with a maintainer first.

## Speculative decoding

`qwen3_asr_server` and `qwen3_asr_transcribe` take `--draft-model-path DIR` (and `--draft-bits 4|6|8` to quantize the draft at load). The Qwen3-ASR target then verifies a DFlash draft's block of proposals each step and keeps its own greedy transcript; the parity tool reports `verify_rounds` per clip. The draft is a Speculators-layout DFlash checkpoint (`aux_hidden_state_layer_ids`, `block_size`, `mask_token_id`, `transformer_layer_config`; weights `fc`, `hidden_norm`, `layers.*`, `norm`, `lm_head`, `d2t`) checked against the target at load: hidden size, the layers it reads (in order), the `fc`, `lm_head` and `d2t` shapes, `d2t` landing inside the target's vocabulary, and a causal block; one that does not fit is refused at load. A request that sets `repetition_penalty` decodes without the draft.

If the verify rounds of a request commit fewer than 2 tokens a round on average, the request decodes the rest of its tokens without the draft. The runtime checks this average every 32 rounds. `verify_rounds` counts only the rounds before the change.

| Target | Draft | Golden |
| --- | --- | --- |
| `mlx-community/Qwen3-ASR-0.6B-4bit` | `wilsonzheng0327/Qwen3-ASR-0.6B-dflash` | `ci/golden/qwen3-asr-0.6b-4bit-dflash.json` |
| `mlx-community/Qwen3-ASR-1.7B-{6bit,8bit}` | `wilsonzheng0327/Qwen3-ASR-1.7B-dflash` | `ci/golden/qwen3-asr-1.7b-8bit-dflash.json` |
| `OpenMOSS-Team/MOSS-Transcribe-Diarize` | `wilsonzheng0327/MOSS-Transcribe-Diarize-dflash` | `ci/model_golden/moss-transcribe-diarize-dflash.json` |

The Qwen3-ASR drafts are pinned in `ci/models.json`; the MOSS-TD draft is pinned in `ci/model_pins.json`. A draft for another target is one pin and one golden whose request names it under `draft_model_path`. Verifying a block moves the target's low-precision logits slightly, so a few percent of clips differ from plain decoding in punctuation or a near-tie word; the goldens hold each chip's draft outputs like any other.

`moss_transcribe_diarize_server` and `moss_transcribe` take the same two flags; the CLI's `verify_rounds` sums over audio chunks. Each audio chunk has its own average.

## Rule 2: Contribution requirements

**Placement and reuse**

- A native model adds `src/<model>/` with `<model>_model`, `<model>_transcriber`
  and a server on the shared `asr_service` (a speech model's on
  `speech_service`), a CLI in `tools/`, and CMake
  targets: a library for the model's own sources that links what it reuses
  (`common`, `swift_port`, `silero_vad`, `sortformer`, `fish_dac`). Shared code goes in
  `src/common/`, not in another model's directory or library. Include project
  headers by their path under `src/`, e.g. `#include "common/layers.h"`.
- Shared audio, checkpoint/layer, tokenizer and multipart APIs use the `omni_common` namespace and the `common` library. HTTP helpers use `omni_server` and the `common_http` library; neither library depends on a model. Qwen3-ASR audio layouts, log-mel extraction and encoder token counts belong to `qwen3_asr/qwen3_audio.h`. Shared decoder and ASR serving types still live under `qwen3_asr/`; their extraction is separate from the common primitive APIs.
- Reuse `layers`, `swift_port`, the Sortformer front end, the Silero VAD
  port and the Fish DAC. If your code duplicates something on `main`, move it into the shared
  file in the same PR.

**Parity**

- Name the reference (mlx-audio-swift at a pinned commit,
  or a Python MLX reference) and use the same weights and MLX version.
- Keep the reference's behavior. Behavior changes go in a separate PR.
- Report English WER, Chinese CER and mixed MER on the 392-clip corpus,
  reference vs native, plus the count of identical clips. Explain where
  differing clips diverge.

**Golden and pins**

- Pin every checkpoint file by revision and SHA-256 in `ci/model_pins.json`.
- Add a golden under `ci/model_golden/`: record a chip's outputs with
  `check_model_golden.py --outputs` and add them with `--import`. Mac CI does
  not run it, so give its result on your Mac in the PR. A speech model's
  golden holds seeded requests' codes (`ci/speech_golden.py`).
- Add a second golden for paths the corpus misses (long-audio chunking, VAD).

**Tests**

- Real-model server tests in `ci/test_model_servers.py`, or in the model's
  own `ci/test_<model>_api.py` once that file would pass ~400 lines, run by
  hand: JSON, SSE and realtime success paths; every invalid input returns 400;
  session-state errors and cancel on disconnect.
- Assert structure, not one chip's exact text; exact text lives in the golden.
- Mac CI runs `tests/` with no checkpoint. List a new server and its model
  kinds in `tests/test_runtime_binaries.py`, with a row for each flag its
  startup rejects. A change to the shared protocol (`common/service.cpp`,
  `qwen3_asr/asr_service.cpp`, `speech_service.cpp`, `realtime.cpp`) gets a
  case against `tests/stub_server.cpp`.

**Robustness**

- Check the checkpoint config at load, so bad checkpoints fail at load, not
  mid-request.
- Validate requests before they reach the worker.
- Streaming sessions use bounded memory: trim consumed audio, cap session
  length, clear the MLX cache.

**Performance**

- Mac CI checks neither outputs nor speed. Report total time, realtime
  factor, peak memory and load time against the reference, with the device
  stated.

**CI and style**

- Target `main` (stacked PRs get no Mac CI), non-draft, with the `run-mac-ci` label. Both Mac CI jobs must pass on a real run.
- Run pre-commit and follow
  [coding-style.md](https://github.com/sgl-project/sglang-omni/blob/main/.claude/skills/code-review/coding-style.md).
- Follow the PR format of
  [#2676](https://github.com/sgl-project/sglang-omni/pull/2676) or
