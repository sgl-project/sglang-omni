## Folder Structure
```text
tests/
├── README.md
├── __init__.py
├── utils.py
├── data/
├── docs/
│   ├── qwen3_omni/
│   └── s2pro/
├── test_model/
│   ├── conftest.py
│   ├── test_qwen3_omni_*_ci.py
│   └── test_s2pro_tts_ci.py
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
    │   ├── test_code2wav.py
    │   ├── test_colocation_config.py
    │   ├── test_config_manager.py
    │   ├── test_logit_shaping.py
    │   ├── test_pipeline.py
    │   ├── test_sglang_ar_budget.py
    │   ├── test_streaming.py
    │   └── test_talker.py
    ├── ming_omni/
    │   ├── test_pipeline.py
    │   ├── test_talker.py
    │   ├── test_thinker.py
    │   ├── test_tokenizer.py
    │   └── test_tp.py
    ├── router/
    │   ├── test_app.py
    │   └── test_core.py
    ├── serve/
    │   └── test_openai_api.py
    └── fishaudio_s2_pro/
        ├── test_pipeline.py
        ├── test_streaming_vocoder.py
        ├── test_tts.py
        └── test_vocoder.py
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
- `s2pro_stage(name)`: in-file CI stage selector for S2-Pro benchmarks.
  Combined with `--s2pro-stage` (see `test_model/conftest.py`).


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
- `qwen3_omni_vision_sglang_env`: session-scoped SGLang dist + DP-attention
  init from `conftest.py`, shared by every Qwen3-Omni vision-encoder benchmark
  module — avoids re-initializing the process-global TP group when the combined
  `-m benchmark` command runs more than one module.
- `test_qwen3_omni_realtime.py`: starts `examples/run_qwen3_omni_server.py`
  with `--enable-realtime` and drives `/v1/realtime` through a real WebSocket
  client to cover text responses, server VAD transcription, and disconnect
  teardown.
- CLI flags `--s2pro-stage {nonstream,stream,consistency,all}` and
  `--concurrency {1,2,4,8,16,all}`: scope an S2-Pro CI sweep without editing
  source.


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
  - stage process environment
  - relay handling
  - GPU memory accounting helpers
  - IPC lifecycle
  - scheduler batching
  - scheduler errors
  - scheduler concurrency
  - scheduler callable contracts, including sync wrappers and callable objects
    that return awaitables.
- `unit_test/qwen3_omni/` Qwen3-Omni unit tests:

  - public CLI/config behavior
  - SGLang argument builders
  - memory flag contracts
  - colocation config and SGLang AR budget contracts
  - `PipelineState` request builders
  - talker behavior
  - Code2Wav streaming/cleanup behavior
  - logit-shaping helpers (e.g. repetition penalty) numerical equivalence with the original per-row scalar formulas.

- `unit_test/ming_omni/` Ming-Omni unit tests:

  - text + speech pipeline config and stage schema
  - launcher argparse, GPU placement, and TP wiring
  - stage factory and scheduler contracts (preprocessing, encoders, thinker, talker, decode)
  - thinker bootstrap registration and Ming model runner wiring
  - multimodal embed injection (per-modality consumed state, pad-value fallback, short-embeds detection)
  - image/vision encoder TP context preservation
  - audio/image preprocessor placeholder construction and cache-key plumbing
  - talker executor request gating and result-builder modality merging
  - Bailing tokenizer loader fallback for vocab compatibility
  - TP topology validation (rank-specific stage specs, talker/thinker GPU collision detection, server_args alignment before infra init).

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

- `unit_test/fixtures/`: Shared fakes. Single-test
  helpers should stay local until a second test needs them.
