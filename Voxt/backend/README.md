# Voxt on sglang-omni's native MLX runtime

Voxt's local Qwen3-ASR 0.6B 4-bit runs on sglang-omni's native runtime
(`sglang_omni_mlx/native`): one C++ binary on MLX, with no Python. Voxt starts
it and owns it. Every other model keeps Voxt's original Swift backend.

| Checkpoint | Runtime | Voxt behavior kept |
| --- | --- | --- |
| `mlx-community/Qwen3-ASR-0.6B-4bit` | `qwen3_asr_server` | Final with context bias and language hint, Swift's audio layout and stop rules, 1200 s energy-cut chunks sharing one token budget, first detected language carried forward; live preview over the realtime socket, first decode after 100 ms of audio, then once a second |

## Build and run

Requirements: an Apple Silicon Mac, Xcode and [uv](https://docs.astral.sh/uv/).

```bash
Voxt/backend/run_omni_dev.sh build
Voxt/backend/run_omni_dev.sh run
```

`build` builds the runtime into `Voxt/build/omni-runtime` with
`sglang_omni_mlx/native/scripts/build_runtime.sh`, then builds "Voxt Omni Dev".
`bin/` holds `qwen3_asr_server`, `qwen3_asr_transcribe`, and the pinned MLX
library and Metal kernels next to them.

"Voxt Omni Dev" has its own bundle identifier. It runs without the sandbox so it
can start the runtime, and it sees `~/.voxt-omni-dev` as its home, so its
database, history, preferences and models stay apart from any installed Voxt.
Set `VOXT_SHARED_MODELS` to an existing `<root>/mlx-audio` directory to reuse
downloaded weights. `run_omni_dev.sh run --swift-backend` runs the same build on
the original Swift backend for comparison.

With the Omni backend enabled (`VOXT_ASR_BACKEND=omni`, `VOXT_OMNI_RUNTIME=<binary>`),
selecting Qwen3-ASR 0.6B 4-bit starts the runtime on a free loopback port.
Switching models, idle unload, deletion and quitting stop it, and a runtime
that dies is replaced on the next use.

## How it fits together

- `qwen3_asr_server --supervised` speaks the launch protocol Voxt expects:
  - On stdout it prints `{"event":"ready",…}` once serving, or `failed` if it can't.
  - On stdin, `{"command":"shutdown"}` makes it reply `stopped` and exit.
  - End of stdin, which happens when Voxt quits or crashes, also stops it, and so does a termination signal.
  - It is a single process, so stopping it leaves nothing behind.
- The server API is the same as the reference Python server
  (`sglang_omni_mlx.qwen3_asr.server`):
  - `/health` with `request_states`;
  - `/v1/models`;
  - `/v1/audio/transcriptions` (JSON or SSE);
  - the `/v1/realtime` manual-turn socket.
- `Voxt/Transcription/Omni*.swift` is the client:
  - `OmniASRRuntime` handles launch, requests, and an awaitable retire that drains in-flight work.
  - It also contains the request planning, the live session and its adapter to Voxt's streaming session interface.

## Tests and CI

- Runtime correctness is checked by `sglang_omni_mlx/native/ci`:
  - `provision.py` fetches the pinned checkpoint and rebuilds the frozen corpus from its public sources, checking every file's SHA-256.
  - `check_golden.py` runs the runtime over the 392-clip corpus with Voxt's Final request and requires every clip to match the golden output. It reports error rates next to the original Swift backend's.
- The `Voxt Mac CI` workflow runs these on the repository's Apple Silicon runner, together with the server API tests and Voxt's Omni unit tests.
- Voxt's opt-in suites need the installed model and `VOXT_RUN_MODEL_TESTS=1`:
  - `OmniPhase1LifecycleTests`:
    - load/Final/unload rounds that must leave no process behind;
    - a server killed mid-Final;
    - a cancelled cold start;
    - termination during a Final;
    - a cancelled live session.

    It also needs `VOXT_ASR_BACKEND=omni`, `VOXT_OMNI_RUNTIME`, `VOXT_MODEL_STORAGE_ROOT` and `VOXT_LIFECYCLE_CLIPS`.
  - `OmniPhase1BenchmarkTests`: the measurement against the original backend. See its header for the `VOXT_BENCH_*` variables.

Pass environment variables to an `xcodebuild test-without-building` run through
the `.xctestrun` file.

## Known limitations

- Greedy decoding only.
- The dev build is ad hoc signed without keychain access groups, so remote
  provider API keys may not persist in it.
- The server accepts requests from any local client on its loopback port; it
  holds no user data beyond in-flight audio.
- The live preview decodes once a second, like Voxt's Swift session, but has
  neither its 0.2 s cadence right after an 8 s window boundary nor its
  agreement-based promotion of provisional text.
