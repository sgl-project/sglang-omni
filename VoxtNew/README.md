# VoxtNew

**English** · [简体中文](README.zh-CN.md)

Voice input for macOS on **local models only**: press a shortcut in any app, speak, and the text appears where your cursor is.

- **转录 Transcribe**: what you say, typed where your cursor is, with live text while you speak and an optional clean-up by a language model (fillers and false starts out, punctuation in).
- **翻译 Translate**: the selected text, or what you say, into another language.
- **转写 Rewrite**: change the selected text by saying how ("shorter", "more formal"), or have new text written from a spoken request; follow-ups in the answer card.
- **应用增强 App enhancement**: groups of apps and websites with their own scene prompt and a key pressed after inserting (a casual tone in chats, Return to send).
- **词典 Dictionary**: names and jargon the speech model expects and the language model writes as you spell them.

With a history, the overlay, shortcuts, the menu bar, a setup guide, settings and updates.

**Shortcuts** (change them on each feature's page): tap the **right Option** key to transcribe, right Option with Shift to translate, Control with right Option to rewrite; Control-Option-V pastes the last result. VoxtNew is one of the [ActNow](https://actnow.dev) apps: these keys leave fn to Auralink, so the two run side by side.

**Website** [actnow.dev/voxtnew](https://actnow.dev/voxtnew) · **Download** [actnow.dev/download/voxtnew](https://actnow.dev/download/voxtnew) (sign in first) · updates come from actnow.dev.

> **Status: 0.1.0** (first beta: `0.1.0-beta.1`). Every feature above works on the local models; see [CHANGELOG.md](CHANGELOG.md).

## What runs on your Mac

Everything that hears or reads your words:

- **Speech recognition** and **voice detection** run on sglang-omni's native MLX servers ([`sglang_omni_mlx/native`](../sglang_omni_mlx/native), C++, no Python), one process per model, started when needed, stopped after 10 minutes unused (Local models → Advanced changes it).
- **Language models** run in `voxtnew-llm`, a Swift process of VoxtNew's own (MLX, and llama.cpp for one model), stopped after 5 minutes unused (or as Local models → Advanced says).
- A small Swift helper does what needs Apple's frameworks: the microphone, global shortcuts, Accessibility, pasting.

VoxtNew offers a few models, tuned for it, and downloads them from Hugging Face (or a mirror) when you ask, each file checked against a pinned SHA-256:

| Model | What it is good at | Download |
|---|---|---|
| Accurate speech recognition (recommended) | the most accurate; accents, dialects, names, several languages in one sentence; live text | 2.5 GB |
| Fast speech recognition | quicker and lighter, for clear speech in a quiet place | 713 MB |
| Balanced language model (recommended) | clean-up, translation and rewriting: the best balance of quality and speed | 3.3 GB |
| Fast language model | answers sooner, uses less memory; short clean-ups and simple rewrites | 1.7 GB |
| Translation model | made for translation, quick and natural; only Translate uses it | 1.1 GB |

Voice detection (1.2 MB) comes with the first speech model. The speech models know 30 languages and 22 Chinese dialects; the general language models are best in Chinese and English.

**Needs** a Mac with Apple silicon and macOS 15 or later. The app is about 280 MB; the recommended pair is a 5.8 GB download (all five models: 9.4 GB). While both recommended models are loaded they take about 7 GB of memory, so 16 GB of memory is comfortable; on an 8 GB Mac choose the fast models.

## Privacy

Audio and text never leave the Mac: recognition, clean-up, translation and rewriting all run on it. VoxtNew has no server of its own, no account and no analytics. It goes online for two things only: downloading the models you choose (from Hugging Face, through the system's proxy, or hf-mirror.com when Hugging Face cannot be reached) and checking for updates on [actnow.dev](https://actnow.dev). The history, its optional recordings, the dictionary and the settings stay in `~/Library/Application Support/VoxtNew/`.

## Build from source

Needs Go 1.27.1 (`GOTOOLCHAIN=auto` fetches it) and, to build the app, an Apple silicon Mac with macOS 15 or later, Xcode (the helper and `voxtnew-llm`), and [uv](https://docs.astral.sh/uv/) or Python 3 for the Omni runtime's build tools (`scripts/build-omni.sh` builds `sglang_omni_mlx/native` with its pinned MLX; CMake fetches its C++ dependencies from GitHub, through the shell's `https_proxy` if needed). Run from `VoxtNew/`:

```sh
scripts/build-helper.sh              # the Swift helper, into resources/darwin-arm64/helper/
scripts/build-omni.sh                # the Omni runtime (speech models), into resources/darwin-arm64/omni/
scripts/build-llm.sh                 # voxtnew-llm (language models), into resources/darwin-arm64/llm/
go tool mygo dev                     # run with hot reload
go tool mygo build                   # build/darwin-arm64/VoxtNew.app and its disk image
```

The helper and `voxtnew-llm` must be built before `go tool mygo build` or `mygo dev`: `mygo.json` gives both their entitlements and fails without them. Without the Omni runtime the app runs but cannot recognise speech (or point `VOXTNEW_OMNI_RUNTIME` at one built with `scripts/build-omni.sh --out DIR`; likewise `VOXTNEW_LLM` for `scripts/build-llm.sh --out DIR`).

**The release build** is one command, on the Mac: `scripts/release.sh` builds the three, packages and signs the app, and checks the result (`--runtimes DIR` takes prebuilt runtimes; with the Developer ID of the team 779M533DL5 (as the other ActNow apps), `--notarize` and `--publish` notarise it and publish it on actnow.dev; over ssh it runs through `scripts/mac-gui.sh`). See [docs/release.md](docs/release.md).

From Linux, `scripts/mac.sh <command>` mirrors the project to a Mac (`MAC_HOST`, default `mac-air`; `MAC_DIR`, default `code/VoxtNew`) and runs the command there: `scripts/mac.sh 'scripts/build-helper.sh && scripts/build-llm.sh && go tool mygo build'`.

<details>
<summary>Signing during development (keeping the permissions across rebuilds)</summary>

macOS keeps the Microphone and Accessibility grants for a signature. An ad hoc signed build (the default) has a new one each time the app is rebuilt, so after a rebuild macOS forgets them: the microphone turns **denied without asking again**, and Accessibility still shows on in System Settings but does not work. The helper's grants are the app's (macOS checks the helper against VoxtNew.app, its responsible process), so the same happens to it. Ways around it:

- **Change only the helper:** keep one built app and run it with a helper outside the bundle. The helper inherits the app's grants wherever it lives, so rebuilding it costs nothing:
  ```sh
  scripts/build-helper.sh --out ~/voxtnew-dev
  open --env VOXTNEW_HELPER=$HOME/voxtnew-dev/voxtnew-helper build/darwin-arm64/VoxtNew.app
  ```
- **Start over:** `tccutil reset Microphone dev.hehehai.voxtnew` (likewise `Accessibility`), then open the app and grant again. If a stale Accessibility entry stays on but does nothing, remove it with − in System Settings and grant again.
- **Sign with a fixed identity:** an Apple Development certificate keeps the grants across rebuilds: `go tool mygo build -sign "Apple Development: …"` (or `macos.signingIdentity` in a local mygo.json; `VOXTNEW_SIGN_IDENTITY` for `scripts/build-helper.sh` and `scripts/build-llm.sh`). Over ssh, `codesign` fails with `errSecInternalComponent` until the key may be used without the GUI: once, in Terminal on the Mac, `security set-key-partition-list -S apple-tool:,apple: -s ~/Library/Keychains/login.keychain-db`.

Switching between your own builds and a release build (Developer ID) needs the same reset once ([docs/release.md](docs/release.md) §4). Downloaded models stay.

</details>

## Development

VoxtNew is a Go app on [MyGo](https://github.com/egoist/mygo) (native macOS UI from Go, no cgo). Go decides; the helper and the model runtimes do what they are told, over JSON lines on stdin and stdout (the helper, `voxtnew-llm`) or loopback HTTP (the Omni servers). Everything but the Swift code is plain Go and is tested on Linux, with fakes of the helper and the runtimes. Run from `VoxtNew/`:

```sh
CGO_ENABLED=0 go test ./...          # everything; runs on Linux
go tool mygo vet .                   # go vet and MyGo's UI-lifetime checks
VOXTNEW_SNAPSHOTS=/tmp/shots go test -run TestSnapshots .   # every page as PNG, light/dark, en/zh-Hans
cd helper && swift test              # the Swift helper (on a Mac), against the same protocol fixtures
scripts/build-llm.sh --test          # voxtnew-llm's Swift tests (on a Mac)
scripts/mac.sh scripts/mac-e2e.sh    # headless Mac tests: the real helper, fake models
scripts/mac.sh scripts/mac-regress.sh   # the real models on a Mac: runtimes, eval, the packaged app
```

The store's tests use MyGo's SQLite plugin, which downloads its library once (checked against its SHA-256) or uses `MYGO_SQLITE_LIBRARY`. On Linux the overlay's pixel checks of Chinese text need a CJK font (Ubuntu: `fonts-noto-cjk`).

- [docs/architecture.md](docs/architecture.md): packages, threading, the session, the runtimes, data on disk, tests.
- [docs/plan.md](docs/plan.md): what VoxtNew is and its milestones.
- [docs/behavior.md](docs/behavior.md): what VoxtNew does differently from Voxt, and why.
- [docs/llm-models.md](docs/llm-models.md): the language models, their settings and measurements.
- [docs/eval.md](docs/eval.md): the models' quality on an eval set, and the decisions it led to.
- [docs/helper-protocol.md](docs/helper-protocol.md), [docs/llm-protocol.md](docs/llm-protocol.md): the helper's and `voxtnew-llm`'s protocols.
- [docs/manual-test.md](docs/manual-test.md): the checks by hand on a Mac (in Chinese), and the automated Mac tests.
- [docs/release.md](docs/release.md): building, signing, notarising and publishing a release; updates (in Chinese).
- [AGENTS.md](AGENTS.md): the rules for changes (strings, threading, the protocols).
- [CHANGELOG.md](CHANGELOG.md): changes by version. [CONTRIBUTING.md](CONTRIBUTING.md): how to send a change.

VoxtNew started as a copy of [Auralink](https://github.com/hehehai/Auralink), the same author's app on remote models, which rewrote [Voxt](https://github.com/hehehai/voxt) (here as [`../Voxt`](../Voxt)) on Go and MyGo with [AudioSmith](https://github.com/hehehai/AudioSmith)'s interface; its local model runtimes and their request policies follow Voxt's.

## Licence

Apache License 2.0 ([LICENSE](LICENSE)), as the rest of this repository. Copyright ActNow and the SGLang-Omni Team. Third-party code, runtimes and their licences are listed in [NOTICE](NOTICE); the models are downloaded under their own licences.
