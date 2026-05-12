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
└── unit_test/
    ├── fixtures/
    │   ├── fish_fakes.py
    │   ├── pipeline_fakes.py
    │   └── qwen_fakes.py
    ├── pipeline/
    │   ├── helpers.py
    │   ├── test_compile.py
    │   ├── test_coordinator.py
    │   ├── test_ipc.py
    │   ├── test_scheduler.py
    │   └── test_stage.py
    ├── qwen3_omni/
    │   ├── test_code2wav.py
    │   ├── test_pipeline.py
    │   └── test_talker.py
    └── fishaudio_s2_pro/
        ├── test_pipeline.py
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

## `test_model/`

End-to-end and model CI tests. These are allowed to depend on real servers,
model snapshots, benchmark artifacts, optional packages, and GPU/runtime
resources.


## `unit_test/`

Fast contract tests that should run without model downloads or real server
startup. Keep these focused on the smallest component that owns the behavior.

Expected command:

```bash
pytest tests/unit_test -q
```
Choose the location by the behavior contract being protected, not by the file
that happened to contain an older version of the test.

- `unit_test/pipeline/`: Model-agnostic V1 pipeline tests:
  - compile
  - runtime wiring
  - coordinator behavior
  - stage routing
  - relay handling
  - IPC lifecycle
  - scheduler batching
  - scheduler errors
  - scheduler concurrency.
- `unit_test/qwen3_omni/` Qwen3-Omni unit tests:

  - public CLI/config behavior
  - SGLang argument builders
  - memory flag contracts
  - `PipelineState` request builders
  - talker behavior
  - Code2Wav streaming/cleanup behavior.

- `unit_test/fishaudio_s2_pro/`: FishAudio S2-Pro unit tests:
  - tokenizer/state contracts
  - TTS scheduler behavior
  - model-runner state transitions
  - vocoder batching/trim behavior.

- `unit_test/fixtures/`: Shared fakes. Single-test
  helpers should stay local until a second test needs them.
