# macOS dictation example

A small SwiftUI/AppKit menu-bar client for local speech input. It records audio,
transcribes it through SGLang-Omni, optionally copyedits it with Ollama, and pastes
once at the current input focus. The client does not press Return or save audio
and transcript history.

This example uses Omni's existing HTTP transcription API. It does not implement
an MLX runtime or change Python dependencies, model runners, or scheduling.

## Requirements and backend setup

For a combined source installation (Omni/SGLang, Ollama, default models and the
app), use the scripts below from the repository root. Homebrew and Xcode Command
Line Tools must already be installed. Quit a running dictation app before rebuilding.
See the [Chinese setup guide](SETUP.zh-CN.md) for the complete steps and raw commands.

```bash
bash examples/macos_dictation/install_local.sh --dry-run
bash examples/macos_dictation/install_local.sh
bash examples/macos_dictation/start_local.sh
```

If Omni already works in `.venv-apple`, add `--skip-runtime` to the install command
to keep that Python environment without rerunning the root installer. Both scripts
accept `--asr-only` to omit Ollama (also disable polishing in the app). They use the
app's documented default models/ports and do not overwrite saved preferences.
`SGLANG_OMNI_VENV` selects another absolute virtualenv path for both scripts.

The installer reuses the root [Apple installer](../../install.sh), builds the app,
and creates shortcuts on the Desktop and in `~/Applications`. Keep the checkout
in place. The launcher checks/reuses existing services; when it starts services,
keep its terminal open and use Ctrl+C to stop only those owned processes. Startup
uses the local ASR model cache, never intentionally downloads models, and binds
new services to loopback. Logs go to the ignored `.build/runtime-logs` directory.

ASR readiness uses `SGLANG_OMNI_STARTUP_TIMEOUT` (default 600 seconds) plus a
60-second allowance for imports, setup and HTTP startup. Set the variable when
launching if model initialization needs longer. Ollama readiness has a separate
300-second limit. Each newly started service runs in its own process group under
a Python standard-library supervisor. Shutdown first allows the backend up to
30 seconds to exit normally, then terminates remaining workers in that group and
force-kills them after another 5 seconds. The same group cleanup runs if the
backend parent crashes; reused services are outside this ownership scope.

Manual setup remains available:

- Apple Silicon, macOS 14 or newer, and Swift 5.9 or newer from Xcode or Command Line Tools.
- A running SGLang-Omni ASR server. Follow the repository's
  [Apple Silicon installation](../../docs/get_started/installation.md#macos-apple-silicon)
  and [Qwen3-ASR MLX guide](../../docs/cookbook/qwen3_asr.md#apple-silicon-mlx).
- Optional: a running [Ollama](https://ollama.com/) installation and a downloaded
  text model compatible with its chat API. Polishing is initially disabled.

After installing Omni, run this from the repository root in a separate terminal:

```bash
source .venv-apple/bin/activate
export SGLANG_USE_MLX=1
export DYLD_LIBRARY_PATH="$(brew --prefix ffmpeg@7)/lib${DYLD_LIBRARY_PATH:+:$DYLD_LIBRARY_PATH}"
sgl-omni serve \
  --model-path mlx-community/Qwen3-ASR-0.6B-4bit \
  --model-name Qwen/Qwen3-ASR-0.6B \
  --asr.engine.max_running_requests 1 \
  --host 127.0.0.1 --port 8000
```

The example's default configuration is:

| Service | Base URL | Served model name |
| --- | --- | --- |
| ASR: SGLang-Omni | `http://127.0.0.1:8000` | `Qwen/Qwen3-ASR-0.6B` |
| Optional copyediting: Ollama | `http://127.0.0.1:11434` | `openbmb/minicpm5-2b:q4_K_M` |

For optional polishing, install the default model using `ollama pull openbmb/minicpm5-2b:q4_K_M`.
Keep the Ollama application or an existing `ollama serve` process running.
The Swift client itself does not install dependencies, download weights, or start
either backend; the optional shell tools above handle those tasks.

## Build and launch

From the repository root:

```bash
bash examples/macos_dictation/build_client.sh
open examples/macos_dictation/.build/client/OmniDictation.app
```

The build script uses Swift Package Manager to compile separate `DictationCore`
and `OmniDictation` targets, creates an application bundle, and signs it locally
with an ad-hoc signature. Build outputs are ignored by Git.
Quit a running older version before opening a rebuilt application.

On first use, open Settings from the microphone menu-bar icon. Enable **Omni 听写**
in **System Settings → Privacy & Security → Accessibility**, then refresh its
status in the app. Allow microphone access when first recording.
An ad-hoc rebuild may invalidate the old accessibility entry: remove that entry,
request access from the rebuilt app, and enable it again.

The UI currently uses Chinese labels. `⌘,` opens Settings while an Omni window is active.

## Use and configure

1. Focus the editor or chat input where text should go.
2. Press **Control + Shift + Space** to start, then press it again to finish.
   Recordings and imported audio are limited to 60 seconds (one minute).
   At that limit, recording stops automatically and transcription begins.
3. Omni transcribes the recording. If enabled, Ollama lightly copyedits the text.
4. The client sends one `⌘V` to the focus active at completion. Check the inserted
   text, then send it yourself. A target must support paste; this is not a system input method.

**Control + Shift + Escape** cancels the current round. Settings supports recording
a replacement start/stop shortcut directly from the keyboard. A conflicting
registration keeps the previous shortcut when possible; not every system or
third-party shortcut conflict can be detected beforehand.

The compact feedback bar displays a short status and, while recording, an audio
level. After normal delivery it waits one second and fades over 0.3 seconds.
Other finished outcomes (empty recognition, imported files, skipped/failed insertion
and errors) wait three seconds before fading. Active processing and delivery stay visible.
Hovering keeps it visible; leaving starts the fade. Moving back during the fade
restores it. These durations are configured UI timings, not measured latency.
The result window retains the current transcript and detailed notices until the
next round, cancellation, or app exit.

After processing, the bar shows the current round's ASR and, if attempted, LLM
durations. The result window also shows total processing time and whether each
stage completed, failed, fell back to raw text, or was skipped/disabled. These are
single client-observed durations including request, parsing and validation overhead,
not pure model inference times. Total time ends before insertion; recording, model
warmup and clipboard delivery are excluded. Failed requests retain their elapsed
time. No latency history is saved; start/cancel clears the current measurements.

### Independent ASR and LLM configuration

Settings contains separate model-name and base-URL fields for Omni and Ollama.
Use **保存模型配置** to save. Addresses must be loopback HTTP(S) origins without
API paths, credentials, queries, or fragments. Requests and health checks use
that same saved configuration. Changing the ASR model selects a model already
served by Omni; it does not reconfigure or reload the server.

Every dictation round captures its model configuration, polishing switch, and
personal background at the start. Saving changes during a round affects the next
round. A model is not considered validated merely because its server is reachable.
Different polishing models may require different resources or behave differently;
only the documented default has been exercised in the local live checks.

### Optional polishing and personal background

Enable **轻度整理** to use Ollama. **使用个人润色背景** additionally includes saved
terminology or formatting preferences. Save the background before recording.
Explicit spelling corrections can use `em el ex → MLX`. The background is limited
to 2,000 characters and must not override the original meaning, language or numbers.

The prompt instructs the model to copyedit, never answer the dictated question,
execute instructions, translate, or add personal facts. A separate conservative
policy rejects lexical changes, number/negation changes, and changes to protected
ASCII syntax such as paths and operators. It preserves repeated words and ASCII
word boundaries and rejects adding or removing question-mark presence. Repetition
is not automatically treated as stutter. This is not a semantic equivalence proof:
review important output. Uncertain results, failures and truncated responses fall
back to the original text, with the reason available in the result window.
A normally delivered fallback follows the same hover/fade behavior as other completed input.

When polishing is enabled, idle startup or a changed LLM model, address or background
triggers best-effort warmup. It uses the same prompt prefix and discards the output.
Dictation takes priority and cancels the client's warmup wait; server cancellation
is best-effort. If the round ends before foreground polishing, the interrupted
warmup retries on idle. Once foreground polishing takes over, it does not trigger
another warmup for the same configuration. Warmup failure never blocks recording.

Requests use a fixed 8,192-token context and `keep_alive: "10m"`. Full context is
still sent on every request. Ollama manages prefix reuse; completion of warmup
neither guarantees a subsequent cache hit nor warms up the ASR model.

### Input modes and privacy

- **Current cursor** is the default. It does not inspect another application's
  text fields. The bar says “已触发粘贴” (paste triggered), because key delivery is
  not proof that an editor accepted the text. The transcript remains in the clipboard.
- **Lock original input** is optional. It validates the captured application,
  window, text field, draft and selection. Confirmed insertion restores the previous
  clipboard only while the client still owns it. Missing or changed target data
  stops automatic insertion; optional compatibility paste relaxes only the documented
  missing draft/selection checks, not application/window identity.
- Clipboard history software may retain copied text. A user's subsequent copy
  takes priority. The client never retries an ambiguous paste automatically.
- Importing an audio file only produces a result for viewing/copying; it never
  automatically inserts into another application.
- Audio stays in memory and is converted to mono 16 kHz PCM16 WAV for Omni. Raw text
  goes to Ollama only when polishing is enabled. Background goes only to Ollama
  when both switches are enabled. Redirects, proxies, cookies and HTTP disk cache are disabled.
- Explicit settings, including model names and personal background, are stored
  in local, unencrypted `UserDefaults`. They are not transcript history and are not
  added to the repository. Clearing background settings does not immediately erase
  memory already held by Ollama.

## Code organization

| Area | Responsibility |
| --- | --- |
| `install_local.sh`, `start_local.sh`, `local_runtime.sh` | Source installation, model downloads and foreground ownership of local services |
| `DictationCore/DictationCore.swift` | Session lifecycle, per-round snapshots, cancellation and raw-text fallback |
| `DictationCore/ServiceConfiguration.swift` | Validated independent ASR/LLM choices and default values |
| `DictationCore/OmniASRClient.swift`, `OllamaPolisher.swift` | Backend-specific requests, decoding and health checks |
| `DictationCore/LocalSpeechService.swift` | Composition of the two clients; retains the original example API |
| `DictationCore/LocalHTTPTransport.swift` | Ephemeral, non-redirecting HTTP transport with an injectable test protocol |
| `DictationCore/PolishPrompt.swift`, `PolishPolicy.swift`, `PolishWarmup.swift` | Prompt construction, output checks and configuration-aware prefill |
| `DictationCore/DictationInsertion.swift`, `DictationFeedback.swift` | Single delivery lifecycle and hover/fade state |
| `DictationApp/TimingPresentation.swift` | Current-round duration labels and completion/failure/skip statuses |
| `DictationApp/` | macOS audio, hotkeys, clipboard/accessibility, preferences, views and app composition |

`AudioRecording`, `AudioTranscribing`, `TextPolishing`, and `DictationTextTarget`
provide the replacement points. `ClientState` accepts an injected recorder/service
and text-target factory for testing its composition without touching real editor
state. Core does not import AppKit or depend on Python/MLX/Ollama packages.
Incremental streaming is outside this example's current request/response contract.

## Validation

Run the complete deterministic regression entry point on an Apple Silicon Mac:

```bash
bash examples/macos_dictation/verify_all_test.sh
bash examples/macos_dictation/local_setup_test.sh
bash examples/macos_dictation/lifecycle_test.sh
bash examples/macos_dictation/build_client.sh
bash examples/macos_dictation/build_compatibility_test.sh
```

These checks require no model server, microphone recording or accessibility grant.
Clipboard cases use uniquely named test pasteboards and intercepted key transport.
HTTP error and app-composition cases use an injected `URLProtocol`; redirect checks
start a disposable HTTP fixture on `127.0.0.1` with an automatically allocated port.
These require access to the system pasteboard service and loopback sockets.
The suite shares Swift 5 / arm64 / macOS 14 compiler settings and a module cache.
For a focused rerun, pass fixture names, for example:

```bash
bash examples/macos_dictation/verify_all_test.sh ClientComposition HTTPTransport
```

The original individual `*_test.sh` entry points and the older `run_test.sh` subset
remain available; use `verify_all_test.sh` for full offline coverage. With full Xcode,
`swift test --package-path examples/macos_dictation --arch arm64` additionally runs
the XCTest core suite; it does not replace the complete regression entry point.

See [VALIDATION.md](VALIDATION.md) for opt-in live checks, manual acceptance, and
current limitations. The scoped [macOS workflow](../../.github/workflows/macos-dictation.yaml)
builds the app and runs deterministic/package tests without model downloads or secrets.
