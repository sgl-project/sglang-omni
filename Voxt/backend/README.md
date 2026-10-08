# Voxt on a local sglang-omni MLX server

This directory runs Voxt's local Qwen3-ASR 0.6B 4-bit on sglang-omni's
standalone MLX server (`sglang_omni_mlx.qwen3_asr.server`), a single process
on Apple Silicon that shares no code with sglang-omni's CUDA serving stack.
Every other model keeps Voxt's original Swift backend.

| Checkpoint | Server | Voxt behavior kept |
| --- | --- | --- |
| `mlx-community/Qwen3-ASR-0.6B-4bit` | `sglang_omni_mlx.qwen3_asr.server` | Final with context bias and language hint, Swift's audio layout and stop rules, 1200 s energy-cut chunks sharing one token budget, first detected language carried forward; live preview over the realtime socket, first decode after 100 ms of audio, then once a second |

Not migrated, and still on the Swift backend: MOSS-Transcribe-Diarize, the
Whisper and other Qwen3-ASR variants, Cohere, Parakeet, Nemotron, SenseVoice,
speaker analysis, VAD and the local LLMs.

## Set up

Requirements: an Apple Silicon Mac, Xcode and [uv](https://docs.astral.sh/uv/).

```bash
Voxt/backend/setup_env.sh ~/voxt-omni-env
```

The script prints the `VOXT_OMNI_PYTHON` to use. It installs the packages
pinned in `requirements-mac.lock` (MLX, `tokenizers` and a small Starlette
server; no PyTorch or SGLang) and this checkout's sglang-omni without its
dependencies.

## Build and run

```bash
Voxt/backend/run_omni_dev.sh build
VOXT_OMNI_PYTHON=~/voxt-omni-env/venv/bin/python Voxt/backend/run_omni_dev.sh run
```

"Voxt Omni Dev" has its own bundle identifier, runs without the sandbox so it
can start the backend, and sees `~/.voxt-omni-dev` as its home, so its database,
history, preferences and models stay apart from any installed Voxt. Set
`VOXT_SHARED_MODELS` to an existing `<root>/mlx-audio` directory to reuse
downloaded weights. `run_omni_dev.sh run --swift-backend` runs the same build on
the original Swift backend for comparison.

Download the models from Voxt's model settings as usual. With the Omni backend
enabled, selecting Qwen3-ASR 0.6B 4-bit starts a server for it on a free
loopback port; switching models, idle unload, deletion and quitting stop it,
and a server that dies is replaced on the next use. After the models are
cached, dictation needs no network.

## How it fits together

- `voxt_omni_backend/supervisor.py` owns one server process per loaded
  model. It reports `ready` only after the server answers with the
  unique model name it was started with, and stops the server and every process
  it started when Voxt sends `shutdown`, when Voxt's control pipe closes (Voxt
  quit or crashed) or on a termination signal. It never signals other processes.
- `Voxt/Transcription/Omni*.swift` is the client: `OmniASRRuntime` (launch,
  requests, an awaitable retire that drains in-flight work), the request
  planning, the live session and its adapter to Voxt's streaming session
  interface.
- `sglang_omni_mlx/qwen3_asr/` (repository root) is the server: the WAV and
  log-mel front end, the MLX model, greedy decoding with Voxt's stop rules, the
  realtime session and the HTTP app.

## Tests

The server tests need `pytest`, `pytest-asyncio`, `httpx` and `transformers`
(for reference features) on top of the backend environment; set
`QWEN3_ASR_MLX_MODEL_PATH` to the installed checkpoint to include the tests
that load it.

```bash
cd Voxt/backend && "$VOXT_OMNI_PYTHON" -m pytest tests
cd ../.. && python -m pytest tests/unit_test/mlx_qwen3_asr
```

Voxt's own tests include `OmniASRRuntimeLaunchTests` (no model needed) and two
opt-in suites that need the installed Qwen model and `VOXT_RUN_MODEL_TESTS=1`:

- `OmniPhase1LifecycleTests`: load/Final/unload rounds that must leave no
  process behind, a server killed mid-Final, a cancelled cold start,
  termination during a Final and a cancelled live session. Set
  `VOXT_ASR_BACKEND=omni` with the backend variables, `VOXT_MODEL_STORAGE_ROOT`
  and `VOXT_LIFECYCLE_CLIPS` (a directory with `short.wav` and `long.wav`).
- `OmniPhase1BenchmarkTests`: the measurement used for acceptance, identical on
  the upstream build and this one; see its header for the `VOXT_BENCH_*`
  variables.

Pass environment variables to an `xcodebuild test-without-building` run through
the `.xctestrun` file. xcodebuild resolves symlinks in those values, which turns
a venv's `bin/python` link into the base interpreter, so point
`VOXT_OMNI_PYTHON` at a script that runs `exec <venv>/bin/python "$@"` there.

## Known limitations

- Greedy decoding only.
- The dev build is ad hoc signed without keychain access groups, so remote
  provider API keys may not persist in it.
- The server accepts requests from any local client on its loopback port; it
  holds no user data beyond in-flight audio.
- The live preview decodes once a second, like Voxt's Swift session, but has
  neither its 0.2 s cadence right after an 8 s window boundary nor its
  agreement-based promotion of provisional text.
