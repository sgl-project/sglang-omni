# Voxt on sglang-omni's native MLX runtime

Voxt can run selected local ASR checkpoints, Silero VAD and Sortformer speaker
diarization on sglang-omni's native runtime (`sglang_omni_mlx/native`): C++
binaries on MLX, with no Python runtime. Voxt starts and owns the selected
server. Other models keep their Swift backend.

To add or change a model here, read the
[MLX path policy and contribution requirements](../../docs/developer_reference/mlx.md)
first.

| Checkpoint | Runtime | Voxt behavior kept |
| --- | --- | --- |
| `mlx-community/Qwen3-ASR-0.6B-4bit`, `mlx-community/Qwen3-ASR-1.7B-6bit`, `mlx-community/Qwen3-ASR-1.7B-8bit` | `qwen3_asr_server` | Final with context bias and language hint, Swift's audio layout and stop rules, 1200 s energy-cut chunks sharing one token budget, first detected language carried forward; live preview over the realtime socket, first decode after 100 ms of audio, then once a second |
| `mlx-community/whisper-large-v3-turbo` | `whisper_server` | Final and batch preview with Voxt's language, token budget, temperature, and 30 s audio windows |
| `mlx-community/whisper-large-v3-mlx` | `whisper_server` | Final and batch preview with Voxt's language, token budget, temperature, and 30 s audio windows; weights load from `weights.npz` |
| `mlx-community/whisper-small-mlx` | `whisper_server` | Final and batch preview with Voxt's language, token budget, temperature, and 30 s audio windows |
| `mlx-community/silero-vad-v6` | `qwen3_asr_server --model-kind silero_vad` | Streaming speech probability per 512-sample chunk with one stream state per audio stream (`/v1/vad/stream`), and offline speech ranges with the meeting sensitivity profile's options (`/v1/vad/speech_timestamps`); one server shared by every detector, started on first use |
| `beshkenadze/cohere-transcribe-03-2026-mlx-fp16` | `cohere_transcribe_server` | Final with Voxt's language, punctuation, and token budget; long recordings use Silero VAD when voice-activity segmentation is selected; live text uses batch preview |
| `mlx-community/diar_streaming_sortformer_4spk-v2.1-fp16` | `qwen3_asr_server --model-kind sortformer` | Meeting speaker analysis: Voxt's feed policy (4.96 s feeds, tails padded to one frame), one streaming state per contiguous run of audio (`/v1/diarization/stream`), `feed`'s threshold, merge gap and state limits, the state-size checks and the timestamp mapping; one server shared by every analysis, started on first use |
| `OpenMOSS-Team/MOSS-Transcribe-Diarize` | `moss_transcribe_diarize_server` | Final with the dictation or meeting prompt (hotwords included), 1200 s energy-cut chunks each with Voxt's token budget, timestamped speaker segments; live 4 s windows finalized as audio arrives, a preview of the last 2.5 s at most once a second |

## Build and run

Requirements: an Apple Silicon Mac, Xcode and [uv](https://docs.astral.sh/uv/).

```bash
Voxt/backend/run_omni_dev.sh build
Voxt/backend/run_omni_dev.sh run
```

`build` builds the runtime into `Voxt/build/omni-runtime` with
`sglang_omni_mlx/native/scripts/build_runtime.sh`, then builds "Voxt Omni Dev".
`bin/` holds `qwen3_asr_server`, `qwen3_asr_transcribe`, `whisper_server`,
`whisper_transcribe`, `cohere_transcribe_server`, `cohere_transcribe`,
`moss_transcribe_diarize_server`, `moss_transcribe`, and the pinned MLX library
and Metal kernels next to them.

"Voxt Omni Dev" has its own bundle identifier. It runs without the sandbox so it
can start the runtime, and it sees `~/.voxt-omni-dev` as its home, so its
database, history, preferences and models stay apart from any installed Voxt.
Set `VOXT_SHARED_MODELS` to an existing `<root>/mlx-audio` directory to reuse
downloaded weights. `run_omni_dev.sh run --swift-backend` runs the same build on
the original Swift backend for comparison.

With the Omni backend enabled (`VOXT_ASR_BACKEND=omni`, `VOXT_OMNI_RUNTIME=<binary>`),
selecting a listed checkpoint starts its runtime on a free loopback port.
Switching models, idle unload, deletion and quitting stop it, and a runtime
that dies is replaced on the next use. Silero VAD gets its own server the first
time a detector needs it; it stops when the last detector unloads or Voxt quits.
Sortformer likewise gets its own server the first time a meeting is analyzed;
a file analysis returns it when done, as the Swift engine dropped its model,
and quitting stops it.

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
  - `OmniSharedModelRuntime` is the one server per model kind that Silero VAD (`OmniVoiceActivity.swift`) and Sortformer (`OmniSpeakerDiarization.swift`) callers lease.

## Tests and CI

- Runtime correctness is checked by `sglang_omni_mlx/native/ci`:
  - `provision.py` fetches the pinned checkpoint and rebuilds the frozen corpus from its public sources, checking every file's SHA-256.
  - `check_golden.py` runs the runtime over the 392-clip corpus with Voxt's Final request and requires every clip to match the golden output. It reports error rates next to the original Swift backend's.
  - Silero VAD and Sortformer golden files hold the original Swift outputs; `vad_golden.py` and `sortformer_golden.py` check them with tolerances for MLX kernel differences.
- The `Voxt Mac CI` workflow runs these on the repository's Apple Silicon runner, together with the server API tests and Voxt's Omni unit tests.
- The Whisper check pins its checkpoint and tokenizer, runs all 392 frozen clips through `check_model_golden.py`, and tests the transcription server. Its golden keeps per-chip outputs with `check_golden.py`'s tolerance, and records the original Swift backend's error rates as its baseline.
- The Cohere check pins its checkpoint and Silero VAD v6, runs the same frozen corpus through `check_model_golden.py`, and tests the server. Its golden keeps per-chip outputs with the same tolerance, and records the original Swift backend's error rates as its baseline; a second golden freezes the energy cut's multi-chunk path on the long clips.
- Voxt's opt-in suites need the installed model and `VOXT_RUN_MODEL_TESTS=1`:
  - `OmniPhase1LifecycleTests`:
    - load/Final/unload rounds that must leave no process behind;
    - a server killed mid-Final;
    - a cancelled cold start;
    - termination during a Final;
    - a cancelled live session.

    It also needs `VOXT_ASR_BACKEND=omni`, `VOXT_OMNI_RUNTIME`, `VOXT_MODEL_STORAGE_ROOT` and `VOXT_LIFECYCLE_CLIPS`.
  - `OmniPhase1BenchmarkTests`: the measurement against the original backend. See its header for the `VOXT_BENCH_*` variables.
  - `OmniVoiceActivityIntegrationTests` and `OmniSpeakerDiarizationIntegrationTests` run Voxt's Silero detectors and Sortformer engine against the server and MLXAudioVAD in the same process. See their headers for the variables.

Pass environment variables to an `xcodebuild test-without-building` run through
the `.xctestrun` file.

## Known limitations

- Greedy decoding only.
- Whisper uses batch preview; its server has no realtime socket.
- Cohere uses batch preview until a native streaming session is implemented.
- The dev build is ad hoc signed without keychain access groups, so remote
  provider API keys may not persist in it.
- The server accepts requests from any local client on its loopback port; it
  holds no user data beyond in-flight audio.
- The live preview decodes once a second, like Voxt's Swift session, but has
  neither its 0.2 s cadence right after an 8 s window boundary nor its
  agreement-based promotion of provisional text.
