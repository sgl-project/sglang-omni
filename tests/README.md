## Folder Structure
```text
tests/
├── README.md
├── __init__.py
├── utils.py
├── data/
├── docs/
│   └── s2pro/
├── test_model/
│   ├── conftest.py
│   ├── test_qwen3_omni_*_ci.py
│   ├── test_qwen3_omni_videoamme_talker_tp2_ci.py
│   ├── test_tts_ci.py
│   ├── test_whisper_asr_ci.py
│   └── omni_whisper_wer_utils.py
└── unit_test/
    ├── fixtures/
    │   ├── fish_fakes.py
    │   ├── pipeline_fakes.py
    │   └── qwen_fakes.py
    ├── pipeline/
    │   ├── helpers.py
    │   ├── test_compile.py
    │   ├── test_coordinator.py
    │   ├── test_gpu_memory.py
    │   ├── test_ipc.py
    │   ├── test_placement.py
    │   ├── test_runtime_adapter.py
    │   ├── test_runtime_schema.py
    │   ├── test_scheduler.py
    │   ├── test_simple_scheduler_concurrent.py
    │   ├── test_stage.py
    │   ├── test_stage_process_env.py
    │   └── test_stage_streaming.py
    ├── qwen3_omni/
    │   ├── test_cli.py
    │   ├── test_code2wav.py
    │   ├── test_colocation_config.py
    │   ├── test_config_manager.py
    │   ├── test_fp8_backend_config.py
    │   ├── test_example_launcher.py
    │   ├── test_logit_shaping.py
    │   ├── test_pipeline.py
    │   ├── test_quantization.py
    │   ├── test_sglang_ar_budget.py
    │   ├── test_streaming.py
    │   ├── test_talker.py
    │   └── test_text_template.py
    ├── ming_omni/
    │   ├── test_omni_serve.py
    │   ├── test_pipeline.py
    │   ├── test_talker.py
    │   ├── test_talker_voice_validation.py
    │   ├── test_thinker.py
    │   ├── test_tokenizer.py
    │   ├── test_tp.py
    │   └── test_vision_patch_embed_linear.py
    ├── qwen3_tts/
    │   └── test_pipeline.py
    ├── higgs_tts/
    │   ├── test_batched_step.py
    │   ├── test_pipeline.py
    │   └── test_request_builders.py
    ├── router/
    │   ├── test_app.py
    │   └── test_core.py
    ├── serve/
    │   └── test_openai_api.py
    ├── fishaudio_s2_pro/
    │   ├── test_pipeline.py
    │   ├── test_streaming_vocoder.py
    │   ├── test_tts.py
    │   └── test_vocoder.py
    └── voxtral_tts/
        └── test_pipeline.py
```

## How To Add A Test


General rules:

- Protect user-visible contracts and component ownership, not incidental implementation structure.
- Keep imports thin and consistent. If a test monkeypatches a module object,
  call through that module alias instead of mixing direct symbol imports.
- Reuse existing helpers and fakes before adding another scheduler, relay, or
  lifecycle helper.
- Add a one-sentence docstring to non-obvious contract tests.
- Do not add root-level `tests/test_*.py` files.


## Markers

Markers are registered in `pyproject.toml` under `[tool.pytest.ini_options]`.
Tag each test with the marker that matches its lane and use it to filter runs.

- `benchmark`: GPU performance / parity tests in `test_model/`. May require a
  populated HF cache and tens of GB of GPU memory; per-test docstrings call
  out hardware needs.
- `docs`: documented-example tests in `docs/`. Verify documented request
  shapes and CLI snippets still work.
- `tts_stage(name)`: in-file CI stage selector for TTS benchmarks.
  Combined with `--tts-stage` (see `test_model/conftest.py`).


## Root Files

- `README.md`: This file. It explains test ownership and where new tests belong.
- `__init__.py`: Keeps `tests` importable as a package.
- `utils.py`: Shared helpers used by docs and model CI tests.

## `data/`

Small static fixtures shared by tests, such as images, audio, and short videos.
Keep these files small and deterministic. Large model artifacts, generated
outputs, and benchmark datasets should live outside the unit test tree.

## `docs/`

Documentation/example tests. These verify that documented user-facing examples
still work.

Use this lane when the test protects:

- install/docs snippets,
- client examples,
- documented request/response shapes,
- examples that may need optional docs dependencies.

These tests are not the default fast unit lane.

Expected command:

```bash
pytest tests/docs -m docs -v
```

## `test_model/`

End-to-end and model CI tests. These are allowed to depend on real servers,
model snapshots, benchmark artifacts, optional packages, and GPU/runtime
resources.

Expected command (GPU benchmark subset):

```bash
pytest tests/test_model -m benchmark -v -s
```

Relevant model CI ownership:

- `qwen3_omni_thinker_server` / `qwen3_omni_talker_server`: expose the shared
  router-backed Qwen3-Omni endpoint from `conftest.py`.
- `test_qwen3_omni_tts_ci.py`: gates the SeedTTS speed/WER path through the
  router and verifies both colocated workers receive traffic.
- `test_whisper_asr_ci.py`: Whisper large-v3 ASR correctness + speed via
  SGLang Omni router (DP=2, `/v1/audio/transcriptions`). Uses the first 20
  English SeedTTS clips; writes `whisper_asr_results.json` for threshold
  calibration (`whisper-asr-v1` in `tune-ci-thresholds`).
- `omni_whisper_wer_utils.py`: shared fixture/helpers for talker/TTS WER CI —
  stops the upstream model server, runs `ensure_gpus_idle.sh`, then launches
  a DP=2 Whisper router for ASR. Used by Qwen3 talker WER tests and TTS
  WER tests instead of the in-process transformers Whisper pipeline.
- Talker / video WER CI (`test_qwen3_omni_*_talker_ci.py`, `test_tts_ci.py`):
  generate audio with the model router first, tear down that server, free both
  GPUs, then transcribe saved WAVs through the Omni Whisper router. Long talker
  clips (>30 s) are chunked client-side in `benchmarks/tasks/tts.py` to match
  the transformers `chunk_length_s=30` behavior.
- CI env alignment on the H20 repro host: `source .github/scripts/ci_env.sh`
  then `source omni/bin/activate`.
  Omni CI (`omni-ci.yaml`) runs benchmark suites sequentially after one shared
  setup: TTS CI → Qwen3-Omni CI → PR Test (`test.yaml` unit tests). A failure in
  an earlier suite does not skip later ones; only a failed setup blocks the chain.
  Full WER sweep: `.github/scripts/run_all_wer_ci_aligned.sh` (milestones on
  stdout; details in `/tmp/wer_ci_qwen3.log` and `/tmp/wer_ci_tts.log`).
- GPU handoff between stages: `.github/scripts/ensure_gpus_idle.sh` (kills orphan
  spawn/router workers, waits for VRAM below threshold).
- `qwen3_omni_vision_sglang_env`: session-scoped SGLang dist + DP-attention
  init from `conftest.py`, shared by every Qwen3-Omni vision-encoder benchmark
  module — avoids re-initializing the process-global TP group when the combined
  `-m benchmark` command runs more than one module.
- `test_qwen3_omni_realtime.py`: starts `examples/run_qwen3_omni_server.py`
  with `--enable-realtime` and drives `/v1/realtime` through a real WebSocket
  client to cover text responses, server VAD transcription, and disconnect
  teardown.
- `test_tts_ci.py`: default TTS CI gate. It starts the TTS managed router
  with two one-GPU workers using the default model config, runs the
  full SeedTTS EN set (1088 samples) in non-streaming / streaming stages at
  concurrency 16, and frees the server GPUs before ASR/WER and
  speaker-similarity checks.
- `test_tts_consistency_artifacts.py`: CPU-only stage-3 check that compares
  TTS non-stream and streaming `speed_results.json` under
  `${OMNI_CI_HOME}/tts-stage-results/{nonstream,stream}/`.
- CLI flags `--tts-stage {tts-stage-1-nonstream,tts-stage-2-stream,tts-stage-3-consistency,all}`
  and `--concurrency {1,2,4,8,16,all}`: scope a TTS CI sweep without
  editing source.

## `unit_test/`

Fast contract tests that should run without model downloads or real server
startup. Keep these focused on the smallest component that owns the behavior.

Expected command:

```bash
pytest tests/unit_test -q
```
Choose the location by the behavior contract being protected, not by the file
that happened to contain an older version of the test.

- `unit_test/pipeline/`: Model-agnostic pipeline tests:
  - compile
  - placement planning
  - runtime wiring
  - runtime schema/adapter behavior
  - coordinator behavior
  - stage routing
  - local-object fan-out selector contracts, including negative coverage for
    shared mutable payload containers while preserving tensor leaf sharing
  - stage process environment
  - relay handling
  - stream relay/IPC selector contracts, including negative coverage for CPU
    tensor metadata and large inline metadata on same-GPU stream chunks
  - GPU memory accounting helpers
  - IPC lifecycle
  - scheduler batching
  - scheduler errors
  - scheduler concurrency
  - scheduler callable contracts, including sync wrappers and callable objects
    that return awaitables.
- `unit_test/qwen3_omni/` Qwen3-Omni unit tests:

  - public CLI/config behavior
  - example launcher config contract (TP/GPU/mem-fraction overrides)
  - SGLang argument builders
  - backend policy and quantization compatibility contracts
  - tokenizer and preprocessing fallback behavior
  - memory flag contracts
  - colocation config and SGLang AR budget contracts
  - `PipelineState` request builders, including projected payload container
    isolation for mutable streaming state
  - talker behavior, including projected prefill tensor storage/slicing, decode
    feedback/text FIFO consumption, and replay of generated-token input embeds
    after decode retract
  - `PipelineState` request builders, including projected payload container
    isolation for mutable streaming state
  - talker behavior, including partial-prefix startup gate, the real
    `_build_talker_request_data` propagation contract (input_ids,
    tts_pad_embed, sampling_seed, fallback chunks, thinker_done), and the
    `_rollback_decode_prep_after_skip` idempotency contract, projected prefill
    tensor storage/slicing, decode feedback/text FIFO consumption, and replay
    of generated-token input embeds after decode retract
  - Code2Wav streaming/cleanup behavior
  - logit-shaping helpers (e.g. repetition penalty) numerical equivalence with the original per-row scalar formulas.

- `unit_test/ming_omni/` Ming-Omni unit tests:

  - text + speech pipeline config and stage schema
  - omni serve CLI/config merge, default speech vs. text-only selection,
    launcher handoff, GPU placement, TP wiring, and unsupported flag capability
    boundaries
  - stage factory and scheduler contracts (preprocessing, encoders, thinker, talker, decode)
  - thinker bootstrap registration and Ming model runner wiring
  - multimodal embed injection (per-modality consumed state, pad-value fallback, short-embeds detection)
  - image/vision encoder TP context preservation
  - audio/image preprocessor placeholder construction and cache-key plumbing
  - talker executor request gating and result-builder modality merging
  - talker voice-preset validation (load-time manifest / wav existence, request-time prompt_wav_path priority), duration-cap heuristic, and `generate()` final-chunk flush across stop-token and step-ceiling exits
  - Bailing tokenizer loader fallback for vocab compatibility
  - TP topology validation (rank-specific stage specs, talker/thinker GPU collision detection, server_args alignment before infra init)
  - vision encoder `patch_embed` numerical equivalence: `nn.Conv3d` vs `F.linear` reshape at the substitution boundary, using synthetic weights without loading real Ming checkpoints.

- `unit_test/qwen3_tts/`: Qwen3-TTS unit tests:
  - pipeline config and registry contracts
  - OmniScheduler-backed AR stage factory wiring
  - request mapping for `ref_audio` / `ref_text` and `references`
  - model-owned default preservation for language and sampling parameters
  - Base, CustomVoice, and VoiceDesign request validation
  - voice-clone reference validation
  - pipeline payload state serialization.

- `unit_test/higgs_tts/`: Higgs TTS unit tests:
  - OmniScheduler-backed AR stage factory wiring
  - sampler-driven finish handling for eager and CUDA-graph paths
  - request builder sampling normalization and server-side token caps
  - model slot cleanup and engine timing in scheduler result adapters.

- `unit_test/router/`: SGLang-Omni Router unit tests:
  - router CLI/config behavior
  - worker metadata and health-state contracts
  - request routing, proxying, and streaming relay
  - worker selection policy behavior
  - managed launcher command construction and cleanup.

- `unit_test/serve/`: In-process serving API unit tests:
  - OpenAI-compatible request/response behavior
  - streaming response framing and failure semantics.

- `unit_test/fishaudio_s2_pro/`: FishAudio S2-Pro unit tests:
  - tokenizer/state contracts
  - TTS scheduler behavior
  - model-runner state transitions
  - vocoder batching/trim behavior
  - streaming vocoder chunking, flush, and abort behavior.

- `unit_test/voxtral_tts/`: Voxtral-TTS unit tests:
  - pipeline config and registry contracts
  - current `StageConfig` schema wiring
  - SGLang-backed generation and vocoder GPU placement contracts
  - terminal stage behavior.

- `unit_test/profiler/`: Request-level profiler unit tests:
  - `RequestEvent` schema and JSONL emit/append behavior
  - concurrent emit safety under multiple threads
  - lifecycle (start / stop / run_id mismatch / stage substitution)
  - timeline reconstruction, stage breakdown, hop breakdown, malformed-line tolerance.

- `unit_test/fixtures/`: Shared fakes. Single-test
  helpers should stay local until a second test needs them.
