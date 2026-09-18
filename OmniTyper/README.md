# OmniTyper

**Local voice typing, powered by SGLang-Omni.**

OmniTyper is an open-source voice input app for Apple Silicon Macs, built with
SwiftUI and AppKit. It uses SGLang-Omni's native MLX Qwen3-ASR service for speech
recognition. An optional OpenAI-compatible text API provides cleanup, translation,
voice editing, and answers to spoken questions.

Verbatim dictation runs locally without a text model. For other modes, connect
your own Ollama instance or another compatible service. OmniTyper is an independent
project and is not affiliated with Typeless.

## Requirements

- macOS 14 or later on Apple Silicon.
- Xcode Command Line Tools; a Swift 6 toolchain is required to run the tests.
- Homebrew, installed before running setup.
- At least 16 GB of memory is recommended, plus several GB of free disk space for
  the Python environment and model weights.
- Internet access for the initial dependency and model downloads.

## Quick start

Run these commands from the `sglang-omni` repository root:

```bash
bash OmniTyper/scripts/setup.sh
open OmniTyper/dist/OmniTyper.app
```

The setup script reuses the repository's [installer](../install.sh). It creates
`OmniTyper/.venv` with Python 3.12 and installs the Apple Silicon dependencies for
SGLang `v0.5.19`, the current SGLang-Omni checkout, and `ffmpeg@7`. It does not
install CUDA packages or replace system Python.

On first launch:

1. Allow **Microphone** and **Accessibility** access from the home screen. macOS
   requires these permissions to be granted through its system UI.
2. Open **Settings → Local speech model → Download & prepare ASR**. The first run
   downloads model weights from Hugging Face. Cached weights support offline ASR.
3. Place the cursor in the destination input field. Press **Control + Option +
   Space**, wait for **Listening**, and speak. Press the shortcut again to finish.
   Keep the input focused until the result is inserted.
4. Press **Esc** to cancel. To use push-to-talk, enable **Hold shortcut to talk**
   in Settings, hold the entire shortcut while speaking, and release it to
   finish. You can also record a custom shortcut, including a single modifier
   key such as **Fn**: press and release it on its own while recording.
5. The default writing style is **verbatim**, which needs no text API. Configure
   [a text model](#text-model-api) when you want cleanup, translation, editing, or
   answers to questions.

Clicking **Start speaking** in OmniTyper's main window produces a result you can
copy. Use the global shortcut from the destination app for automatic insertion.

Drag the main window by its native title bar and the recording popup by its
header. The popup remembers its position while the app runs and keeps keyboard
focus in the destination app. Settings toggles and disclosure rows also respond
to clicks on their labels. Notifications stay visible above the page while
scrolling through Settings.

Closing the main window keeps OmniTyper in the menu bar. **Quit** exits the app
and shuts down its model processes. To use **Open at login**, first place the app
at a stable location, such as `~/Applications`. Keep the repository and Python
environment available; this is a source build, not a self-contained installer.

## Features

| Feature | Behavior |
| --- | --- |
| Dictate | Local ASR with verbatim output by default; optional text cleanup before insertion. |
| Translate | Transcribe speech and translate it into the configured target language through the text API. |
| Voice edit | Select text in another app, speak an editing instruction, and replace the selection. The original selection is not stored in history. |
| Ask | Ask about selected text or a general topic. The answer appears in OmniTyper without replacing the selection. No web search is performed. |
| Shortcuts | Custom shortcuts, toggle or hold-to-talk recording, Esc to cancel, and a floating recording panel that does not take focus. |
| Audio | Microphone selection, a live level meter, start/stop sounds, and a five-minute recording limit. |
| Dictionary | Preferred spelling, literal replacements, CSV import/export, and entries created from history corrections. |
| Writing style | Global and per-app preferences: `verbatim`, `clean`, `casual`, `formal`, and `concise`. |
| History | Search, mode filters, original transcripts, copying, corrections, export, deletion, and retention settings. |
| Audio retention | Off by default. When enabled, recordings support dictation/translation retries and WAV export. Deleting a history entry also deletes its audio. |
| Appearance and language | Light/dark/system appearance and English/Simplified Chinese/system language. App language changes apply immediately; macOS permission dialogs follow the system language. |

OmniTyper does not include cloud sync, a mobile keyboard, passive learning across
apps, web browsing, or automated website actions. Dictionary learning happens
only when you explicitly save an entry. Model quality is not guaranteed to match
other voice input products.

## Text model API

SGLang-Omni handles ASR. Your text API service handles text model downloads,
configuration, loading, and inference. OmniTyper records audio, builds text
requests, and inserts the result. It does not install, start, stop, or manage
Ollama or its models.

To connect Ollama:

1. Start your Ollama service and use `ollama list` to find an installed model.
2. In **Settings → Text API**, set the base URL to
   `http://127.0.0.1:11434/v1`, the default value.
3. Click **Connect & load models**, or enter the model name directly. Custom names
   created with an Ollama Modelfile are supported.
4. Choose a writing style other than `verbatim`, or use Translate, Voice edit, or
   Ask to enable text processing.

OmniTyper calls `GET <Base URL>/models` and `POST <Base URL>/chat/completions`.
Include the service prefix, usually `/v1`, in the base URL; do not enter the full
`/chat/completions` path. Services without a model-list endpoint can be used by
entering a model name directly. See the
[Ollama OpenAI compatibility documentation](https://docs.ollama.com/api/openai-compatibility)
for the server-side API.

**Request options (JSON)** defaults to `{}`. Requests specify `model`, `messages`,
and `stream: false`, leaving sampling defaults to the server. You can add fields
supported by your service, for example:

```json
{"temperature": 0.2, "max_tokens": 2048}
```

Model weights, context-window settings, and Modelfile parameters remain under your
control in Ollama. If you rename the model, update its name in OmniTyper. Unsupported
request options are reported as errors.

Local Ollama usually needs no API key. An optional key is kept only in memory and
cleared when the endpoint changes or the app exits. It is not written to settings,
history, or logs. Use HTTPS for remote endpoints; requests do not follow redirects.
A local endpoint can still route to a cloud model, so offline operation depends on
your service configuration.

If text cleanup fails during dictation, OmniTyper keeps the original transcript
and shows a warning. Translation, editing, and question-answering failures keep
the original transcript available for copying or retrying instead of inserting
it as a successful result.

## Live transcription

OmniTyper starts or reuses the ASR service before recording. Wait for **Listening**
before speaking. Preparing the model in Settings avoids the first-load delay.
In hold-to-talk mode, releasing the shortcut during startup cancels that attempt.

Enable **Settings → Local speech model → Keep the speech model loaded** to
preload at launch and keep the service ready between recordings, including after
cancellation. This is opt-in and uses memory while idle. Turning it off or
choosing **Unload ASR** releases the service; quitting always stops it. A recording
or history retry started during preload waits for the preparation already running.

Audio is streamed to `/v1/realtime?intent=transcription`, with partial transcripts
shown in the recording panel and main window. The current upstream defaults
process approximately two seconds of new audio per partial update and segment
long recordings at 30-second boundaries. Display latency also depends on inference
time. Partial text can be revised; the client replaces revised segments rather
than appending duplicates. Short recordings may produce only a final result.
This uses periodic audio-window inference, not continuous token-by-token decoding.

Only the final transcript is processed and inserted after recording stops. Live
preview text is not inserted into the destination app. If streaming fails or
falls behind, OmniTyper shows a warning and transcribes the complete WAV after
recording ends.

The realtime path does not currently pass dictionary hotwords to ASR. Dictionary
replacements still apply to the final transcript; full-WAV transcription also
passes supported hotword hints to the server.

## Text insertion and clipboard behavior

Enable **Insert text automatically at the original cursor** in Settings and start
recording with the global shortcut from the destination input field.

For accessible fields, OmniTyper checks that the app, field, contents, cursor, and
selection still match the captured target before writing. Native fields may accept
a direct Accessibility write; web-hosted fields use clipboard paste.

Apps such as WeChat may not expose an accessible input field. In that case,
OmniTyper checks the original foreground app and window, then sends **Command + V**.
This fallback cannot detect cursor or conversation changes within the same window.
Keep the intended input focused until processing finishes. Voice edit requires a
readable selection and is unavailable when the app does not expose one.

Clipboard paste temporarily replaces the clipboard, then restores its previous
contents if no other copy has occurred. It does not permanently copy every result.
Use **Copy** in the result or history view to keep text on the clipboard. Ask and
retried recordings do not insert automatically.

Password fields reported by Accessibility, and system secure-input mode, block
recording and insertion. If target validation fails, the result remains available
to copy. Some custom editors and remote desktops may not accept simulated paste.

## Architecture

```text
SwiftUI / AppKit
  ├─ AVAudioEngine → 16 kHz mono PCM16 → /v1/realtime WebSocket
  │    └─ Live transcript preview + temporary WAV for recovery
  ├─ Global shortcuts, Accessibility, conditional clipboard restoration
  └─ Private stdin/stdout JSON-lines worker
       ├─ Managed SGLANG_USE_MLX=1 sgl-omni serve
       │    └─ Native Qwen3-ASR MLX
       │         ├─ /v1/realtime
       │         └─ /v1/audio/transcriptions for full-WAV recovery and retries
       └─ OpenAI-compatible HTTP API → Ollama or another text model service
            └─ Cleanup, translation, editing, and answers
```

ASR uses a pinned revision of `mlx-community/Qwen3-ASR-0.6B-4bit`, configured in
[backend/server.py](backend/server.py). It reuses SGLang-Omni's native
[Qwen3-ASR MLX implementation](../sglang_omni/models/qwen3_asr/mlx/) and does not
depend on `mlx-audio`. No text model is bundled or loaded through MLX-LM in the
worker; the configured API service controls the text model's lifecycle.

Code ownership:

| File | Responsibility |
| --- | --- |
| `Sources/OmniTyper/AppModel.swift` | Recording sessions, cancellation, retries, and insertion orchestration |
| `Sources/OmniTyper/AudioRecorder.swift` | Microphone capture, PCM conversion, and temporary WAV ownership |
| `Sources/OmniTyper/GlobalShortcut.swift` | Keyboard event tap and held-key state |
| `Sources/OmniTyper/TextInsertion.swift` | Accessibility, destination checks, and clipboard restoration |
| `Sources/OmniTyper/ASRStream.swift` | Bounded WebSocket transport and transcript revisions |
| `Sources/OmniTyper/WorkerClient.swift` | Worker process, JSON-lines framing, timeouts, and cancellation |
| `Sources/OmniTyper/Store.swift` | Settings, history, dictionary, and retention |
| `Sources/OmniTyper/Views.swift` | Main window, shared view components, and voice panel |
| `Sources/OmniTyper/LibraryViews.swift` | History, dictionary, writing rules, and import/export |
| `Sources/OmniTyper/PreferencesView.swift` | Settings and shortcut capture |
| `backend/worker.py` | Private request validation and ASR/text-processing orchestration |
| `backend/server.py` | Pinned model setup and ownership of the native ASR process group |
| `backend/text_api.py` | Text prompts and bounded OpenAI-compatible HTTP requests |

The ASR service stays loaded between recordings. Quitting, cancelling, or stopping
the worker cleans up its service process group. Initial model preparation allows
up to 30 minutes; ordinary worker requests allow up to 10 minutes. Text API
connection and read timeouts are 10 and 180 seconds, respectively.

## Privacy and local data

Microphone audio is processed by the local ASR service. When a text API is used,
OmniTyper sends the transcript, writing preferences, and selected text needed for
Voice edit or Ask to that endpoint.

Destination-field contents, window identity, and selection state are compared in
memory to avoid inserting into a changed target. A field can expose an entire
document; those validation snapshots are not saved or sent to models. Explicitly
selected text used by Voice edit or Ask is the exception described above. The app
does not capture screenshots, read browsing history, or collect analytics.

The worker exposes no public control API. The ASR HTTP service binds to a random
port on `127.0.0.1`, not the LAN. Its inference endpoints currently have no
authentication, so other processes on the same machine can access them.

Settings and history are stored at:

```text
~/Library/Application Support/OmniTyper/library.json
~/Library/Application Support/OmniTyper/Audio/
```

Data directories use permissions `0700`, data files use `0600`, and writes are
atomic. OmniTyper provides no additional disk encryption. History is limited to
1,000 entries, with retention choices of 24 hours, 7 days, 30 days, one year, or
forever. Disabling history deletes existing entries and recordings; disabling
audio retention deletes saved audio.

Failed recordings can be retained temporarily for retries during the current
session and are removed on normal exit. Crashes or forced termination can leave
temporary files. Model weights use the standard Hugging Face cache.

Diagnostics are available from Settings and stored at:

```text
~/Library/Logs/OmniTyper/diagnostics.log
```

The log records JSON events containing error codes, destination bundle IDs,
insertion paths, permission status, and the app version. It excludes transcripts,
selected text, field contents, window titles, and file paths. It is bounded to
128 KiB, discarding older complete lines when necessary. Individual events larger
than the log limit are omitted.

## Development

Run all commands from the repository root:

```bash
# Build against an existing Python environment.
OMNITYPER_PYTHON=/absolute/path/to/python bash OmniTyper/scripts/build.sh

# Build a debug version.
CONFIGURATION=debug bash OmniTyper/scripts/build.sh

# Run unit and integration tests without model downloads or microphone access.
bash OmniTyper/scripts/test.sh

# Check live ASR with synthesized speech streamed in real time; no text API needed.
OmniTyper/.venv/bin/python OmniTyper/backend/smoke_stream.py

# Check real ASR and a running text API. Replace my-model with a server model name.
OmniTyper/.venv/bin/python OmniTyper/backend/smoke.py --model my-model

# Use an existing WAV recording.
OmniTyper/.venv/bin/python OmniTyper/backend/smoke.py --model my-model --audio /absolute/path/to/audio.wav

# Use another compatible endpoint; set OMNITYPER_API_KEY if it requires authentication.
OmniTyper/.venv/bin/python OmniTyper/backend/smoke.py --base-url http://127.0.0.1:8080/v1 --model my-model
```

Automated tests cover audio conversion, streaming, worker lifecycle and request
validation, shortcut state across repeated configuration, insertion-target
validation, localization, byte-bounded JSON diagnostics, history, dictionary
handling, and data migration. Real microphone behavior, physical
shortcuts, permissions, and third-party input compatibility also require desktop
acceptance testing.

Model smoke tests require cached or downloadable ASR weights and, where applicable,
a running text API. Set `HF_HUB_OFFLINE=1` to prevent Hugging Face downloads when
weights are cached. This does not prevent network access to the configured text API.

The app bundle includes worker source files and records the Python executable's
absolute path in `Info.plist`. It does not bundle Python or model weights. On
another Mac, rerun setup or configure an existing compatible environment. Do not
move or delete the repository or virtual environment while the app relies on it.

Builds use ad-hoc signing by default. Set `CODE_SIGN_IDENTITY` to an appropriate
code-signing identity for a stable designated requirement across rebuilds. Public
distribution requires your own Developer ID signing and Apple notarization.

### Migrating from OpenTypeless

Quit the old app before launching OmniTyper. The first launch migrates the old data
directory to `~/Library/Application Support/OmniTyper`, preserving settings,
dictionary entries, history, and recordings. An existing OmniTyper data directory
is not overwritten. A stale Python path from a renamed project directory is
updated when the replacement executable exists.

The bundle identifier is `org.sglang.OmniTyper`. Grant Microphone and Accessibility
permissions to this app and re-enable its login item if needed.

## Troubleshooting

### Shortcuts do not respond or text is not inserted

Check Accessibility permission and the automatic-insertion setting. Start from the
destination input field and keep it focused until processing finishes. Inspect
`capture.failed` and `insert.failed` events in the diagnostic log. The
`paste-window` path indicates the fallback for apps without accessible input
fields. Switching apps or windows cancels automatic insertion.

### Permissions stop working after an update

With the default ad-hoc signature, macOS ties an Accessibility grant to the build's
code hash. Rebuilding can invalidate the grant while System Settings still shows
its switch enabled. Toggling the stale entry may not fix it.

Quit OmniTyper, remove its entry from **System Settings → Privacy & Security →
Accessibility**, or reset that app's grant:

```bash
tccutil reset Accessibility org.sglang.OmniTyper
```

Add the rebuilt app, enable access, and relaunch it. If microphone permission also
stops working, grant it again under **Privacy & Security → Microphone**. Frequent
local rebuilds can use a stable signing identity through `CODE_SIGN_IDENTITY`
instead of ad-hoc signing.

### Hold-to-talk closes before recording starts

Keep the entire shortcut held, including its modifier keys, until you finish
speaking. Releasing it while the model is loading cancels startup. Use **Download
& prepare ASR** before recording to avoid waiting for the initial load. Use a
current build containing the physical-key-state fix if the panel closes while
all keys remain held.

### Microphone or ASR is unavailable

Allow microphone access and check that the selected input device is connected.
The system-default device is resolved at the start of each recording. For model
startup failures, rerun `OmniTyper/scripts/setup.sh` and check Python 3.12,
`ffmpeg@7`, and Hugging Face connectivity. Setup includes HTTPX's SOCKS support for
proxy environments.

### Text processing fails

Check that the configured service is running, the base URL has the correct `/v1`
prefix, and the model name matches the server. For remote services, check the API
key and HTTPS endpoint. Remove custom request options to rule out unsupported
parameters. Failed translation or editing does not automatically insert the raw
transcript; it remains available to copy.

### The history file is corrupt

OmniTyper preserves the original file and stops overwriting it. Back it up before
repairing it or moving it aside, using the path reported by the app.

## License

[Apache-2.0](../LICENSE). Model weights and dependencies retain their respective
licenses.
