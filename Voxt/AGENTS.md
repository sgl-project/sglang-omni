# Working on VoxtNew

VoxtNew is a macOS voice input app: Go + MyGo (native UI, no cgo) and a small Swift helper. Speech recognition and voice activity detection run on this Mac, on sglang-omni's native MLX servers (`sglang_omni_mlx/native`); the language models are remote services the user sets up (any OpenAI-compatible API, Ollama on this Mac included). Voxt runs no language model of its own: the local ones retired in 2.1, and their Swift process (`voxtnew-llm`) is gone. It started as a copy of Auralink, the same author's app on remote models (NOTICE). Read [docs/architecture.md](docs/architecture.md) first, then [docs/plan.md](docs/plan.md) for the milestones; [docs/behavior.md](docs/behavior.md) compares its behaviour with Voxt's.

Users see the app as **Voxt**: a user-visible text says Voxt, never VoxtNew, ActNow or actnow.dev (`TestNoOldNames`). Some inside names stay VoxtNew: the Swift packages, the processes, the bundle id, the data folders and `voxtnew://` (docs/architecture.md, Names).

These rules apply inside `Voxt/`. The repository's root `AGENTS.md` and its coding style are for the Python tree; Voxt follows its own Go rules below.

## Rules

- **Speech on this Mac by default, language models remote.** Speech is recognised on the Mac unless the user sets up a remote speech service for a feature on the Speech models page (`internal/asr`, `providers.NewASR`): the audio then goes only to that service, its key in the Keychain (`asr.<id>`). Text goes only to the language model the user set up on the Language models page (`internal/llm`'s OpenAI-compatible transport and presets, `internal/providers`), and only the recognised or selected text; its API key lives in the Keychain (`internal/secrets`), never in the settings, logs or git. The network is for update checks, Voxt's announcements (`internal/announce`: nothing of the user's goes with them), model downloads and those services. No account comes back (signing in to ChatGPT for the "ChatGPT (Codex)" language model is the user's own sign-in to a third-party service, for their own plan, not a Voxt account). The local language models are retired (`models.Model.Retired`: never offered, opened or downloaded). The speech models live in the ActNow family's model library (`$HUB`, the Hugging Face cache, shared with the family's other apps: `internal/models` on `actnowkit`), and downloads follow the family's shared network settings (proxy, Hub endpoints; "automatic" = the Mac's system proxy). VoxtNew is a member of the ActNow family (`internal/actnow`, the spec in the actnow repository `docs/family/actnow-family-spec.md`): never delete another app's models or claims. Models are behind `internal/local` (`local.Services`; `providers.Services` adds the remote language models): the session and the pages never talk to a runtime or a service directly.
- **Every user-visible string goes through `i18n.T`** (or `N`, `X`) with a literal English message, and whoever adds one adds its zh-Hans translation to `internal/i18n/locales/zh-Hans.json`. `TestKeyParity` fails on a missing translation, an unused one, or format verbs that differ. Product words in Chinese: 转录 (Transcribe), 翻译 (Translate), 改写 (Rewrite), 应用增强 (App enhancement), 词典 (Dictionary), 语音识别模型 (Speech models: never 语音模型, one character from 语言模型), 语言模型 (Language models), 辅助进程 (the helper), 转录稿 (a transcript), App (an app).
- **The build pass does no I/O.** Views read `app` fields, snapshots and memory-only getters (`local.Services.Installed` too). Store, runtime and helper calls run on goroutines (`a.async`, `helperCall`) and come back through `a.post`. `app` is main-thread only.
- **Voxt and Auralink are sources.** Both are the same author's, Apache-2.0 for VoxtNew: port their code where it fits (Voxt's Omni runtime lifecycle, model request policies and prompts; Auralink's everything else), written our way (one session owner, a layered prompt composer, pure logic apart from platform code), and record a behaviour that differs from Voxt's in `docs/behavior.md` (Voxt / VoxtNew / why).
- **The UI is AudioSmith's.** Build pages from `ui_kit.go` and the tokens in `ui_tokens.go` (one ink primary button per view, the accent for state only, Solar icons, `settingRow` with ⓘ help). New icons: add the name to `internal/icons/names.txt`, then `go generate ./internal/icons`.
- **Pure logic in `internal/`, platform code behind narrow interfaces** (the helper, `permission`, the local runtimes), with fakes for tests: everything must be testable on Linux. A runtime's client is tested against a fake server on loopback.
- **The helper protocol** changes in every place at once: `internal/helper/proto`, the Swift types (`helper/Sources/VoxtNewHelperKit/Wire*.swift`), `helper/Fixtures`, the fake helper (`internal/helper/helpertest`) and `docs/helper-protocol.md`. Go and Swift tests decode the same fixtures.
- **The helper does, Go decides**: policy (when ⌘C may run, delays, keys after a paste, which device) goes in the params; the helper keeps no settings. The same holds for the speech models' runtime: Go supervises it and chooses the parameters.
- **Files stay small**: at most 600 lines of code (aim for 400), 800 of tests; split by topic into `<area>_<part>.go`. `internal/archtest` checks the sizes (a ratchet over `allowlist.txt`) and the import layering (`imports.txt` lists every internal → internal import: add a line for a new one). Refactors that move code prove it with `declhash` and `scripts/snapshots.sh check` (docs/architecture.md, "Code structure rules").
- **Licence**: VoxtNew is Apache-2.0, as the repository. Third-party code, data, models' runtimes or artwork need a licence compatible with it and an entry in `NOTICE`, with its licence text in `LICENSES/` (a new Go module: its licence file in `LICENSES/go/`). No per-file licence headers.
- **The Mac (`mac-air`) is shared and short of disk** (about 32 GB free): build in `code/VoxtNew` (`scripts/mac.sh`), delete large build trees and models you downloaded for a test, and never touch `~/code/Auralink*`. The six catalogue models are in the real `$HUB` (`~/Library/Application Support/SGLang Omni/models` once VoxtNew 2.0.1 or later has started, hard links of those in `~/Library/Application Support/ActNow/models/hub`, which the user's AudioSmith and Dicta use; claimed as `dev.hehehai.voxtnew`) and the Omni runtime in `~/voxtnew-runtime/omni/` for every thread (its README says how to refresh it): use them instead of downloading or building again, never delete or move anything in the HF cache, and test deletions with a scratch `ACTNOW_FAMILY_DIR` + `HF_HUB_CACHE`. Do not sync (`scripts/mac.sh`) into a checkout while it builds, nor before pulling back files the Mac made (`--pull`): the sync deletes what the Linux side lacks.
- Code comments and commit messages in English. Keep the MyGo lifetime rules (`.agents/skills/mygo-maintenance`).

## Commands

Run from `Voxt/`; Go 1.27.1 (`GOTOOLCHAIN=auto` fetches it).

```sh
CGO_ENABLED=0 go test ./...                                  # all tests (Linux and Mac)
go tool mygo vet .                                           # vet + MyGo's UI checks
CGO_ENABLED=1 go test -race ./...                            # needs cgo
VOXTNEW_SNAPSHOTS=/tmp/shots go test -run TestSnapshots .    # pages as PNG
go test -run TestKeyParity ./internal/i18n                   # strings and zh-Hans
scripts/mac.sh '<command>'                                   # run on the Mac (MAC_HOST, MAC_DIR=code/VoxtNew)
scripts/mac.sh 'scripts/build-helper.sh && go tool mygo build'   # the Swift helper, then the app
scripts/mac.sh 'scripts/build-omni.sh && go tool mygo build'     # the Omni runtime (speech models) into the app; --out DIR for VOXTNEW_OMNI_RUNTIME
scripts/mac.sh 'scripts/release.sh --runtimes $HOME/voxtnew-runtime'   # the release build: helper, omni, mygo build, release-verify.sh (docs/release.md); VOXTNEW_SIGN_IDENTITY signs it (team 779M533DL5, as the other ActNow apps: scripts/release-identity.sh), over ssh through scripts/mac-gui.sh; official releases: a voxt-vX.Y.Z tag runs .github/workflows/voxt-release.yaml (team RY2TQTYSKQ, docs/release.md)
MAC_DIR=code/VoxtNew-xyz scripts/mac.sh 'VOXTNEW_COEXIST_OK=1 scripts/mac-hotkey-coexist.sh'   # a built app's helper beside a running Auralink: synthetic keys, right ⌥ fires only VoxtNew
MAC_DIR=code/VoxtNew-sign scripts/mac.sh scripts/mac-sign-e2e.sh  # the release's signing with a throwaway identity: every nested binary hardened and timestamped
scripts/mac.sh 'cd helper && swift test'                     # the helper's tests (same fixtures as Go)
MAC_DIR=code/VoxtNew-mt scripts/mac.sh scripts/mac-e2e.sh    # headless Mac tests: real helper, fake models, no window or key
scripts/mac-latency.sh ACCURATE_DIR FAST_DIR [ROUNDS [SETTLE]]   # on the Mac: the first transcription's stages after a launch and a change of model, real speech models, fake helper (TestMacE2EFirstLatency)
MAC_DIR=code/VoxtNew-logsel scripts/mac.sh scripts/mac-logs-shots.sh  # the logs' selection worked with the real pointer and captured (only while nobody uses the Mac; VOXTNEW_SHOTS_DRY=1: the window only)
MAC_DIR=code/VoxtNew-xyz scripts/mac.sh scripts/mac-regress.sh  # the real models ($HUB, ~/voxtnew-runtime): Swift tests, omni, LLM, quick eval, app smoke
scripts/mac.sh 'VOXTNEW_EVAL=1 VOXTNEW_OMNI_RUNTIME=$HOME/voxtnew-runtime/omni go test -run TestEval -v .'  # the eval (docs/eval.md)
scripts/vendor-mygo.sh --check                               # third_party/mygo is MyGo + our patches
scripts/snapshots.sh check                                   # every snapshot, pixel for pixel, against the merge base
go run ./internal/tools/declhash . ./internal/...            # a hash per declaration: equal before and after a pure move
go vet -tags updatecheck .                                   # the update-check build
```
