# Voxt on sglang-omni's native MLX runtime

Voxt's local Qwen3-ASR, Silero VAD and Sortformer speaker diarization run on
sglang-omni's native runtime
(`sglang_omni_mlx/native`): one C++ binary on MLX, with no Python. Voxt starts
it and owns it. Every other model keeps Voxt's original Swift backend.

| Checkpoint | Runtime | Voxt behavior kept |
| --- | --- | --- |
| `mlx-community/Qwen3-ASR-0.6B-4bit`, `mlx-community/Qwen3-ASR-1.7B-6bit`, `mlx-community/Qwen3-ASR-1.7B-8bit` | `qwen3_asr_server` | Final with context bias and language hint, Swift's audio layout and stop rules, 1200 s energy-cut chunks sharing one token budget, first detected language carried forward; live preview over the realtime socket, first decode after 100 ms of audio, then once a second |
| `mlx-community/silero-vad-v6` | `qwen3_asr_server --model-kind silero_vad` | Streaming speech probability per 512-sample chunk with one stream state per audio stream (`/v1/vad/stream`), and offline speech ranges with the meeting sensitivity profile's options (`/v1/vad/speech_timestamps`); one server shared by every detector, started on first use |
| `mlx-community/diar_streaming_sortformer_4spk-v2.1-fp16` | `qwen3_asr_server --model-kind sortformer` | Meeting speaker analysis: Voxt's feed policy (4.96 s feeds, tails padded to one frame), one streaming state per contiguous run of audio (`/v1/diarization/stream`), `feed`'s threshold, merge gap and state limits, the state-size checks and the timestamp mapping; one server shared by every analysis, started on first use |

Not migrated, and still on the Swift backend: MOSS-Transcribe-Diarize,
Whisper, Cohere, Parakeet, Nemotron, SenseVoice, the OmniVAD-Kit and energy VAD
modes, and the local LLMs.

## Run from scratch

Requirements:

- An Apple Silicon Mac with macOS 15 or later.
- Xcode 26 or later, with its Metal Toolchain.
- [uv](https://docs.astral.sh/uv/).
- 7.0 GB of free disk space:
  - 0.92 GB for the Metal Toolchain.
  - 0.71 GB for the model.
  - 0.24 GB for the VAD and meeting models that Voxt downloads at first launch.
  - 0.74 GB for the runtime build. Its build tools take 0.38 GB of this.
  - 4.4 GB for the app build.

Run all commands from the repository root.

1. Install the Metal Toolchain. Xcode 26 and later do not include it, and the
   MLX Swift package that Voxt links needs it to compile its Metal kernels
   (839 MB download):

   ```bash
   xcodebuild -downloadComponent MetalToolchain
   ```

2. Optional: download the model now instead of from Voxt's settings:

   ```bash
   uvx --from huggingface_hub hf download mlx-community/Qwen3-ASR-0.6B-4bit --local-dir ~/voxt-models/mlx-audio/mlx-community_Qwen3-ASR-0.6B-4bit
   ```

3. Build the runtime and "Voxt Omni Dev". With the Swift packages and the
   build tools' wheels already in the local caches, the runtime took 16 s and
   the app took 74 s on an Apple M5 Pro:

   ```bash
   Voxt/backend/run_omni_dev.sh build
   ```

   `build` builds the runtime into `Voxt/build/omni-runtime` with
   `sglang_omni_mlx/native/scripts/build_runtime.sh`, then builds the app into
   `Voxt/build/omni-dev`. The runtime's `bin/` holds `qwen3_asr_server`,
   `qwen3_asr_transcribe`, the `silero_vad_probe` and `sortformer_probe` tools,
   and the pinned MLX library and Metal kernels next to them.

4. Run it on the Omni backend. Leave out `VOXT_SHARED_MODELS` if you skipped
   step 2:

   ```bash
   VOXT_SHARED_MODELS=~/voxt-models/mlx-audio Voxt/backend/run_omni_dev.sh run
   ```

5. In Voxt's onboarding:
   1. Grant Microphone and Accessibility access.
   2. In the model step, select the local model "Qwen3 0.6B (4bit)". If you
      skipped step 2, download it there.
   3. Put the cursor in a text field, press `fn` once, speak, and press `fn`
      again. The text appears at the cursor.

6. Within 90 s after the dictation, check that it ran on the Omni backend.
   After 90 s without use, Voxt unloads the model and stops the runtime:

   ```bash
   pgrep -fl qwen3_asr_server
   ```

"Voxt Omni Dev" has its own bundle identifier. It runs without the sandbox so it
can start the runtime, and it sees `~/.voxt-omni-dev` as its home, so its
database, history, preferences and models stay apart from any installed Voxt.
`VOXT_SHARED_MODELS` must be an existing `<root>/mlx-audio` directory. It
applies only when `~/.voxt-omni-dev` has no `mlx-audio` model directory yet,
which is normally the first run.
`run_omni_dev.sh run --swift-backend` runs the same build on the original Swift
backend for comparison.

With the Omni backend enabled (`VOXT_ASR_BACKEND=omni`, `VOXT_OMNI_RUNTIME=<binary>`),
selecting one of these Qwen3-ASR checkpoints starts the runtime on a free loopback port.
Switching models, idle unload, deletion and quitting stop it, and a runtime
that dies is replaced on the next use. Silero VAD gets its own server the first
time a detector needs it; it stops when the last detector unloads or Voxt quits.
Sortformer likewise gets its own server the first time a meeting is analyzed;
a file analysis returns it when done, as the Swift engine dropped its model,
and quitting stops it. After the models are cached, dictation needs no
network.

### Troubleshooting

| Symptom | Cause | Fix |
| --- | --- | --- |
| `run_omni_dev.sh build` fails with "cannot execute tool 'metal' due to missing Metal Toolchain" | Xcode 26 and later ship without it. | Run step 1. |
| codesign fails with "resource fork, Finder information, or similar detritus not allowed" | The checkout, and so its build directory `Voxt/build/omni-dev`, is in a folder that iCloud Drive syncs, such as Desktop or Documents. | Move the checkout to a folder that iCloud Drive does not sync. Then delete `Voxt/build` and build again, because the moved app keeps the iCloud attributes. |
| Dictation inserts no text after a rebuild | Each ad hoc signed build is a new app to macOS, so the old Accessibility grant no longer applies. | In System Settings > Privacy & Security > Accessibility, remove "Voxt Omni Dev", then add it again. |
| No `qwen3_asr_server` process during dictation | The app was started with `--swift-backend`, a model on the Swift backend is selected, or the model was unused for 90 s and Voxt unloaded it. | Start it as in step 4, select "Qwen3 0.6B (4bit)", and run step 6 within 90 s after a dictation. |

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

Build the tests with the same xcconfig as the app. They run inside "Voxt Omni
Dev", so one build serves both the Swift and the Omni backend:

```bash
xcodebuild build-for-testing -project Voxt/Voxt.xcodeproj -scheme Voxt -configuration Debug \
  -destination 'platform=macOS' -xcconfig Voxt/Config/OmniDev.xcconfig \
  -derivedDataPath Voxt/build/omni-dev -skipPackagePluginValidation
```

xcodebuild passes each `TEST_RUNNER_<NAME>` variable to the tests as `<NAME>`.
For example, the live session benchmark on the Omni backend:

```bash
TEST_RUNNER_VOXT_RUN_MODEL_TESTS=1 \
TEST_RUNNER_VOXT_MODEL_STORAGE_ROOT=~/voxt-models \
TEST_RUNNER_VOXT_BENCH_MANIFEST=<corpus>/manifest.jsonl TEST_RUNNER_VOXT_BENCH_CLIPS=<corpus>/clips \
TEST_RUNNER_VOXT_BENCH_OUT=<output> TEST_RUNNER_VOXT_BENCH_RUN=C-r1 \
TEST_RUNNER_VOXT_ASR_BACKEND=omni \
TEST_RUNNER_VOXT_OMNI_RUNTIME="$PWD/Voxt/build/omni-runtime/bin/qwen3_asr_server" \
xcodebuild test-without-building -destination 'platform=macOS' \
  -xctestrun Voxt/build/omni-dev/Build/Products/Voxt_macosx*-arm64.xctestrun \
  -only-testing:VoxtTests/OmniPhase1BenchmarkTests/testSessionBenchmark
```

Leave out the two backend variables to measure the Swift backend. Keep the Mac
awake and on power during a benchmark: a sleep stretches the clips it interrupts.

## Known limitations

- Greedy decoding only.
- The dev build is ad hoc signed without keychain access groups, so remote
  provider API keys may not persist in it.
- The server accepts requests from any local client on its loopback port; it
  holds no user data beyond in-flight audio.
- The live preview decodes once a second, like Voxt's Swift session, but has
  neither its 0.2 s cadence right after an 8 s window boundary nor its
  agreement-based promotion of provisional text.
