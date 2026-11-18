# Building and developing Voxt

**English** · [简体中文](development.zh-CN.md)

How to build Voxt from source, sign it during development and run its tests. To install and use Voxt, see the [README](../README.md); to send a change, [CONTRIBUTING.md](../CONTRIBUTING.md).

## What is inside

- **Speech recognition** and **voice detection** run on sglang-omni's native MLX servers ([`sglang_omni_mlx/native`](../../sglang_omni_mlx/native), C++, no Python), one process per model, started when needed, stopped after 10 minutes unused (Speech models → Advanced changes it).
- A small Swift helper does what needs Apple's frameworks: the microphone, global shortcuts, Accessibility, pasting.

## Build from source

Needs Go 1.27.1 (`GOTOOLCHAIN=auto` fetches it) and, to build the app, an Apple silicon Mac with macOS 15 or later, Xcode (the helper), and [uv](https://docs.astral.sh/uv/) or Python 3 for the Omni runtime's build tools (`scripts/build-omni.sh` builds `sglang_omni_mlx/native` and, the first time, the MLX commit it pins; git and CMake fetch MLX and the C++ dependencies from GitHub and metal-cpp from developer.apple.com, through the shell's `https_proxy` if needed). Run from `Voxt/`:

```sh
scripts/build-helper.sh              # the Swift helper, into resources/darwin-arm64/helper/
scripts/build-omni.sh                # the Omni runtime (speech models), into resources/darwin-arm64/omni/
go tool mygo dev                     # run with hot reload
go tool mygo build                   # build/darwin-arm64/Voxt.app and its disk image
```

The helper must be built before `go tool mygo build` or `mygo dev`: `mygo.json` gives it its entitlements and fails without it. Without the Omni runtime the app runs but cannot recognise speech (or point `VOXTNEW_OMNI_RUNTIME` at one built with `scripts/build-omni.sh --out DIR`).

**The release build** is one command, on the Mac: `scripts/release.sh` builds the helper and the Omni runtime, packages and signs the app, and checks the result (`--runtimes DIR` takes prebuilt runtimes; with a Developer ID identity, `--notarize` notarises it (Voxt 2 has no update feed, so nothing to `--publish`); over ssh it runs through `scripts/mac-gui.sh`). A release is that build on GitHub: a `voxt-vX.Y.Z` tag runs the Voxt Release workflow, which signs it with the Developer ID of the team RY2TQTYSKQ, notarises it and puts its disk image on a draft GitHub release. See [docs/release.md](release.md).

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

Switching between your own builds and a release build (Developer ID) needs the same reset once ([docs/release.md](release.md) §4). Downloaded models stay.

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

- [docs/architecture.md](architecture.md): packages, threading, the session, the runtimes, data on disk, tests.
- [docs/plan.md](plan.md): what the app is and its milestones.
- [docs/behavior.md](behavior.md): what the app does differently from the Swift Voxt ([hehehai/voxt](https://github.com/hehehai/voxt)), and why.
- [docs/llm-models.md](llm-models.md): the retired local language models, their settings and measurements (history).
- [docs/eval.md](eval.md): the models' quality on an eval set, and the decisions it led to.
- [docs/helper-protocol.md](helper-protocol.md): the helper's protocol.
- [docs/history-access.md](history-access.md): reading the dictation history from scripts and agents on this Mac (off by default).
- [docs/manual-test.md](manual-test.md): the checks by hand on a Mac (in Chinese), and the automated Mac tests.
- [docs/release.md](release.md): building, signing, notarising and publishing a release; updates (in Chinese).
- [AGENTS.md](../AGENTS.md): the rules for changes (strings, threading, the protocols).
- [CHANGELOG.md](../CHANGELOG.md): changes by version. [CONTRIBUTING.md](../CONTRIBUTING.md): how to send a change.

Voxt takes the place of the Swift Voxt (1.x) that this repository held before 2.0.0: installing it replaces that app's Voxt.app in Applications, and the two keep their data apart; Voxt can import the old app's dictionary, history, downloaded speech models, and language model and speech service setups ([docs/voxt-migration.md](voxt-migration.md)). Some names inside stay VoxtNew (its Swift packages, processes, bundle id and data folders). It started as a copy of [Auralink](https://github.com/hehehai/Auralink), the same author's app on remote models, which rewrote [Voxt](https://github.com/hehehai/voxt) on Go and MyGo with [AudioSmith](https://github.com/hehehai/AudioSmith)'s interface; its local model runtimes and their request policies follow Voxt's.
