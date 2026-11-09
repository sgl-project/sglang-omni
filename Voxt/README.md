# Voxt

**English** · [简体中文](README.zh-CN.md)

Voice input for macOS: press a shortcut in any app, speak, and the text appears where your cursor is. Speech is recognised **on your Mac**; the language model that cleans up, translates and rewrites is one you choose: a service's API, or one you run on this Mac with Ollama.

- **转录 Transcribe**: what you say, typed where your cursor is, with live text while you speak and an optional clean-up by a language model (fillers and false starts out, punctuation in).
- **翻译 Translate**: the selected text, or what you say, into another language.
- **转写 Rewrite**: change the selected text by saying how ("shorter", "more formal"), or have new text written from a spoken request; follow-ups in the answer card.
- **应用增强 App enhancement**: groups of apps and websites with their own scene prompt and a key pressed after inserting (a casual tone in chats, Return to send).
- **词典 Dictionary**: names and jargon the speech model expects and the language model writes as you spell them.

With a history, the overlay, shortcuts, the menu bar, a setup guide and settings.

**Shortcuts** (change them on each feature's page): tap the **right Option** key to transcribe, right Option with Shift to translate, Control with right Option to rewrite; Control-Option-V pastes the last result. These keys leave fn to Auralink, so the two run side by side.

## Download and install

Download the newest `Voxt-X.Y.Z-arm64.dmg` from [the Voxt releases](https://github.com/sgl-project/sglang-omni/releases?q=voxt-v&expanded=true), open it and drag Voxt to Applications. Voxt needs a Mac with Apple silicon and macOS 15 or later.

- **New to Voxt**: open it and follow the setup guide: grant the microphone and Accessibility, download the recommended speech model, and set up a language model for clean-up, Translate and Rewrite.
- **Coming from Voxt 1.x**: quit Voxt 1.x first, then replace it with Voxt. In the setup guide, **Import from the previous Voxt** brings over the dictionary, history, downloaded speech models, and your language model and speech service setups. Shortcuts and other settings are not imported. Grant the microphone and Accessibility again: macOS treats Voxt as a new app. Voxt 1.x's Check for Updates does not install Voxt 2.

Step by step: [Install Voxt](https://sgl-project.github.io/sglang-omni/get_started/voxt.html). The import in detail: [docs/voxt-migration.md](docs/voxt-migration.md).

**Updates**: Voxt does not update itself; install each new version from its disk image. Changes by version: [CHANGELOG.md](CHANGELOG.md).

## What is new in Voxt 2

Voxt 2 replaces the Swift Voxt 1.x:

- Speech recognition runs on SGLang-Omni's native MLX servers, with live text; **Faster transcription** speeds up Qwen3-ASR by speculative decoding.
- You choose the language model: a service's API, or Ollama, LM Studio or oMLX on this Mac. Voxt 2 no longer runs language models itself.
- Rewrite keeps a conversation in the answer card; the dictionary suggests terms from your history and your corrections.
- One-click import from Voxt 1.x, and the app is open source in this repository.

## What runs on your Mac

Everything that hears your words:

- **Speech recognition** and **voice detection** run on sglang-omni's native MLX servers ([`sglang_omni_mlx/native`](../sglang_omni_mlx/native), C++, no Python), one process per model, started when needed, stopped after 10 minutes unused (Local models → Advanced changes it).
- A small Swift helper does what needs Apple's frameworks: the microphone, global shortcuts, Accessibility, pasting.

Voxt offers a few models, tuned for it, and downloads them from Hugging Face (or a mirror) when you ask, each file checked against a pinned SHA-256:

| Model | What it is good at | Download |
|---|---|---|
| Qwen3-ASR 1.7B (8bit) (recommended) | the most accurate; accents, dialects, names, several languages in one sentence; live text | 2.5 GB |
| Qwen3-ASR 0.6B (4bit) | quicker and lighter, for clear speech in a quiet place | 713 MB |

Voxt 1.x's other local models are there too: Qwen3-ASR 1.7B (6bit), Whisper Large v3 Turbo, Whisper Large v3 and Whisper Small (about 100 languages), Parakeet TDT 0.6B v3 (25 European languages), SenseVoice Small (Chinese, Cantonese, English, Japanese, Korean), Cohere Transcribe 03-2026 (14 languages), Nemotron 3.5 ASR 0.6B (28 languages, live text) and MOSS-Transcribe-Diarize (Chinese and English, live text). Only Qwen3-ASR takes your dictionary. Models Voxt 1.x downloaded come over without a download (docs/voxt-migration.md).

Voice detection (Silero VAD v6, 1.2 MB) comes with the first speech model. The Qwen3-ASR models know 30 languages and 22 Chinese dialects. The Qwen3-ASR 1.7B (8bit) and 0.6B (4bit) cards also offer **Faster transcription**: a draft model (726 MB and 262 MB) for speculative decoding, which proposes several words at a time for the speech model to check.

**Language models** are the services you set up on the Language models page, over their OpenAI-compatible API: OpenAI, Anthropic, Google Gemini, DeepSeek, Qwen (Alibaba Cloud), Kimi, GLM, Doubao (Volcengine), MiniMax, StepFun, Xiaomi MiMo, xAI, OpenRouter, any other compatible server, or one on this Mac (Ollama, LM Studio, oMLX: free, and the text stays here). Without one, Transcribe types the text as recognised; Translate and Rewrite need one.

**Needs** a Mac with Apple silicon and macOS 15 or later. The app is about 280 MB; the recommended speech model is a 2.5 GB download.

## Privacy

Speech is recognised on the Mac, unless you set up a remote speech service for a feature on the Speech models page (OpenAI, Doubao, Aliyun and others): that feature's recordings then go to that service. The text to clean up, translate or rewrite goes to the language model you set up, and only there: with Ollama, LM Studio or oMLX it stays on this Mac. API keys stay in the macOS Keychain. Voxt has no server of its own, no account and no analytics. Otherwise it goes online only to download the models you choose (from Hugging Face, through the system's proxy, or hf-mirror.com when Hugging Face cannot be reached). The history, its optional recordings, the dictionary and the settings stay in `~/Library/Application Support/VoxtNew/`.

## Build from source

Needs Go 1.27.1 (`GOTOOLCHAIN=auto` fetches it) and, to build the app, an Apple silicon Mac with macOS 15 or later, Xcode (the helper), and [uv](https://docs.astral.sh/uv/) or Python 3 for the Omni runtime's build tools (`scripts/build-omni.sh` builds `sglang_omni_mlx/native` with its pinned MLX; CMake fetches its C++ dependencies from GitHub, through the shell's `https_proxy` if needed). Run from `Voxt/`:

```sh
scripts/build-helper.sh              # the Swift helper, into resources/darwin-arm64/helper/
scripts/build-omni.sh                # the Omni runtime (speech models), into resources/darwin-arm64/omni/
go tool mygo dev                     # run with hot reload
go tool mygo build                   # build/darwin-arm64/Voxt.app and its disk image
```

The helper must be built before `go tool mygo build` or `mygo dev`: `mygo.json` gives it its entitlements and fails without it. Without the Omni runtime the app runs but cannot recognise speech (or point `VOXTNEW_OMNI_RUNTIME` at one built with `scripts/build-omni.sh --out DIR`).

**The release build** is one command, on the Mac: `scripts/release.sh` builds the three, packages and signs the app, and checks the result (`--runtimes DIR` takes prebuilt runtimes; with a Developer ID identity, `--notarize` notarises it (Voxt 2 has no update feed, so nothing to `--publish`); over ssh it runs through `scripts/mac-gui.sh`). A release is that build on GitHub: a `voxt-vX.Y.Z` tag runs the Voxt Release workflow, which signs it with the Developer ID of the team RY2TQTYSKQ, notarises it and puts its disk image on a draft GitHub release. See [docs/release.md](docs/release.md).

From Linux, `scripts/mac.sh <command>` mirrors the project to a Mac (`MAC_HOST`, default `mac-air`; `MAC_DIR`, default `code/VoxtNew`) and runs the command there: `scripts/mac.sh 'scripts/build-helper.sh && go tool mygo build'`.

<details>
<summary>Signing during development (keeping the permissions across rebuilds)</summary>

macOS keeps the Microphone and Accessibility grants for a signature. An ad hoc signed build (the default) has a new one each time the app is rebuilt, so after a rebuild macOS forgets them: the microphone turns **denied without asking again**, and Accessibility still shows on in System Settings but does not work. The helper's grants are the app's (macOS checks the helper against Voxt.app, its responsible process), so the same happens to it. Ways around it:

- **Change only the helper:** keep one built app and run it with a helper outside the bundle. The helper inherits the app's grants wherever it lives, so rebuilding it costs nothing:
  ```sh
  scripts/build-helper.sh --out ~/voxtnew-dev
  open --env VOXTNEW_HELPER=$HOME/voxtnew-dev/voxtnew-helper build/darwin-arm64/Voxt.app
  ```
- **Start over:** `tccutil reset Microphone dev.hehehai.voxtnew` (likewise `Accessibility`), then open the app and grant again. If a stale Accessibility entry stays on but does nothing, remove it with − in System Settings and grant again.
- **Sign with a fixed identity:** an Apple Development certificate keeps the grants across rebuilds: `go tool mygo build -sign "Apple Development: …"` (or `macos.signingIdentity` in a local mygo.json; `VOXTNEW_SIGN_IDENTITY` for `scripts/build-helper.sh`). Over ssh, `codesign` fails with `errSecInternalComponent` until the key may be used without the GUI: once, in Terminal on the Mac, `security set-key-partition-list -S apple-tool:,apple: -s ~/Library/Keychains/login.keychain-db`.

Switching between your own builds and a release build (Developer ID) needs the same reset once ([docs/release.md](docs/release.md) §4). Downloaded models stay.

</details>

## Development

Voxt is a Go app on [MyGo](https://github.com/egoist/mygo) (native macOS UI from Go, no cgo). Go decides; the helper and the model runtimes do what they are told, over JSON lines on stdin and stdout (the helper) or loopback HTTP (the Omni servers). Everything but the Swift code is plain Go and is tested on Linux, with fakes of the helper and the runtimes. Run from `Voxt/`:

```sh
CGO_ENABLED=0 go test ./...          # everything; runs on Linux
go tool mygo vet .                   # go vet and MyGo's UI-lifetime checks
VOXTNEW_SNAPSHOTS=/tmp/shots go test -run TestSnapshots .   # every page as PNG, light/dark, en/zh-Hans
cd helper && swift test              # the Swift helper (on a Mac), against the same protocol fixtures
scripts/mac.sh scripts/mac-e2e.sh    # headless Mac tests: the real helper, fake models
scripts/mac.sh scripts/mac-regress.sh   # the real models on a Mac: runtimes, eval, the packaged app
```

The store's tests use MyGo's SQLite plugin, which downloads its library once (checked against its SHA-256) or uses `MYGO_SQLITE_LIBRARY`. On Linux the overlay's pixel checks of Chinese text need a CJK font (Ubuntu: `fonts-noto-cjk`).

- [docs/architecture.md](docs/architecture.md): packages, threading, the session, the runtimes, data on disk, tests.
- [docs/plan.md](docs/plan.md): what the app is and its milestones.
- [docs/behavior.md](docs/behavior.md): what the app does differently from the Swift Voxt ([hehehai/voxt](https://github.com/hehehai/voxt)), and why.
- [docs/llm-models.md](docs/llm-models.md): the retired local language models, their settings and measurements (history).
- [docs/eval.md](docs/eval.md): the models' quality on an eval set, and the decisions it led to.
- [docs/helper-protocol.md](docs/helper-protocol.md): the helper's protocol.
- [docs/history-access.md](docs/history-access.md): reading the dictation history from scripts and agents on this Mac (off by default).
- [docs/manual-test.md](docs/manual-test.md): the checks by hand on a Mac (in Chinese), and the automated Mac tests.
- [docs/release.md](docs/release.md): building, signing, notarising and publishing a release; updates (in Chinese).
- [AGENTS.md](AGENTS.md): the rules for changes (strings, threading, the protocols).
- [CHANGELOG.md](CHANGELOG.md): changes by version. [CONTRIBUTING.md](CONTRIBUTING.md): how to send a change.

Voxt takes the place of the Swift Voxt (1.x) that this repository held before 2.0.0: installing it replaces that app's Voxt.app in Applications, and the two keep their data apart; Voxt can import the old app's dictionary, history, downloaded speech models, and language model and speech service setups ([docs/voxt-migration.md](docs/voxt-migration.md)). Some names inside stay VoxtNew (its Swift packages, processes, bundle id and data folders). It started as a copy of [Auralink](https://github.com/hehehai/Auralink), the same author's app on remote models, which rewrote [Voxt](https://github.com/hehehai/voxt) on Go and MyGo with [AudioSmith](https://github.com/hehehai/AudioSmith)'s interface; its local model runtimes and their request policies follow Voxt's.

## Licence

Apache License 2.0 ([LICENSE](LICENSE)), as the rest of this repository. Copyright the SGLang-Omni Team. Third-party code, runtimes and their licences are listed in [NOTICE](NOTICE); the models are downloaded under their own licences.
