# Working on VoxtNew

VoxtNew is a macOS voice input app on local models only: Go + MyGo (native UI, no cgo) and a small Swift helper. Speech recognition and voice activity detection come from sglang-omni's native MLX servers (`sglang_omni_mlx/native`), the language models from a Swift process of their own (mlx-swift-lm and a few GGUF models). It started as a copy of Auralink, the same author's app on remote models (NOTICE). Read [docs/architecture.md](docs/architecture.md) first, then [docs/plan.md](docs/plan.md) for the milestones; [docs/behavior.md](docs/behavior.md) compares its behaviour with Voxt's.

These rules apply inside `VoxtNew/`. The repository's root `AGENTS.md` and its coding style are for the Python tree; VoxtNew follows its own Go rules below.

## Rules

- **Local models only.** Audio and text never leave the Mac; the network is for update checks and model downloads. No remote provider, API key or account comes back. The models live in the ActNow family's model library (`$HUB`, the Hugging Face cache, shared with the family's other apps: `internal/models` on `actnowkit`), and downloads follow the family's shared network settings (proxy, Hub endpoints; "automatic" = the Mac's system proxy). VoxtNew is a member of the ActNow family (`internal/actnow`, the spec in the actnow repository `docs/family/actnow-family-spec.md`): never delete another app's models or claims. Models are behind `internal/local` (`local.Services`): the session and the pages never talk to a runtime directly.
- **Every user-visible string goes through `i18n.T`** (or `N`, `X`) with a literal English message, and whoever adds one adds its zh-Hans translation to `internal/i18n/locales/zh-Hans.json`. `TestKeyParity` fails on a missing translation, an unused one, or format verbs that differ. Product words in Chinese: 转录 (Transcribe), 翻译 (Translate), 转写 (Rewrite), 应用增强 (App enhancement), 词典 (Dictionary), 本地模型 (Local models), 辅助进程 (the helper).
- **The build pass does no I/O.** Views read `app` fields, snapshots and memory-only getters (`local.Services.Installed` too). Store, runtime and helper calls run on goroutines (`a.async`, `helperCall`) and come back through `a.post`. `app` is main-thread only.
- **Voxt and Auralink are sources.** Both are the same author's, Apache-2.0 for VoxtNew: port their code where it fits (Voxt's Omni runtime lifecycle, model request policies and prompts; Auralink's everything else), written our way (one session owner, a layered prompt composer, pure logic apart from platform code), and record a behaviour that differs from Voxt's in `docs/behavior.md` (Voxt / VoxtNew / why).
- **The UI is AudioSmith's.** Build pages from `ui_kit.go` and the tokens in `ui_tokens.go` (one ink primary button per view, the accent for state only, Solar icons, `settingRow` with ⓘ help). New icons: add the name to `internal/icons/names.txt`, then `go generate ./internal/icons`.
- **Pure logic in `internal/`, platform code behind narrow interfaces** (the helper, `permission`, the local runtimes), with fakes for tests: everything must be testable on Linux. A runtime's client is tested against a fake server on loopback.
- **The helper protocol** changes in every place at once: `internal/helper/proto`, the Swift types (`helper/Sources/VoxtNewHelperKit/Wire*.swift`), `helper/Fixtures`, the fake helper (`internal/helper/helpertest`) and `docs/helper-protocol.md`. Go and Swift tests decode the same fixtures. So does the language models' llm protocol: `internal/llmrt/wire`, `llm/Sources/VoxtNewLLMKit/Wire*.swift`, `llm/Fixtures`, `internal/llmrt/llmrttest` and `docs/llm-protocol.md`.
- **The helper does, Go decides**: policy (when ⌘C may run, delays, keys after a paste, which device) goes in the params; the helper keeps no settings. The same holds for the model runtimes: Go supervises them and chooses the parameters.
- **Files stay small**: at most 600 lines of code (aim for 400), 800 of tests; split by topic into `<area>_<part>.go`. `internal/archtest` checks the sizes (a ratchet over `allowlist.txt`) and the import layering (`imports.txt` lists every internal → internal import: add a line for a new one). Refactors that move code prove it with `declhash` and `scripts/snapshots.sh check` (docs/architecture.md, "Code structure rules").
- **Licence**: VoxtNew is Apache-2.0, as the repository. Third-party code, data, models' runtimes or artwork need a licence compatible with it and an entry in `NOTICE`, with its licence text in `LICENSES/` (a new Go module: its licence file in `LICENSES/go/`). No per-file licence headers.
- **The Mac (`mac-air`) is shared and short of disk** (about 32 GB free): build in `code/VoxtNew` (`scripts/mac.sh`), delete large build trees and models you downloaded for a test, and never touch `~/code/Auralink*`. The six catalogue models are in the real `$HUB` (`~/.cache/huggingface/hub`, claimed as `dev.hehehai.voxtnew`; the user's AudioSmith and Dicta use it too) and the runtimes in `~/voxtnew-runtime/{omni,llm}/` for every thread (its README says how to refresh it): use them instead of downloading or building again, never delete or move anything in the HF cache, and test deletions with a scratch `ACTNOW_FAMILY_DIR` + `HF_HUB_CACHE`. Do not sync (`scripts/mac.sh`) into a checkout while it builds, nor before pulling back files the Mac made (`--pull`): the sync deletes what the Linux side lacks.
- Code comments and commit messages in English. Keep the MyGo lifetime rules (`.agents/skills/mygo-maintenance`).

## Commands

Run from `VoxtNew/`; Go 1.27.1 (`GOTOOLCHAIN=auto` fetches it).

```sh
CGO_ENABLED=0 go test ./...                                  # all tests (Linux and Mac)
go tool mygo vet .                                           # vet + MyGo's UI checks
CGO_ENABLED=1 go test -race ./...                            # needs cgo
VOXTNEW_SNAPSHOTS=/tmp/shots go test -run TestSnapshots .    # pages as PNG
go test -run TestKeyParity ./internal/i18n                   # strings and zh-Hans
scripts/mac.sh '<command>'                                   # run on the Mac (MAC_HOST, MAC_DIR=code/VoxtNew)
scripts/mac.sh 'scripts/build-helper.sh && scripts/build-llm.sh && go tool mygo build'   # the Swift helper and voxtnew-llm, then the app (mygo.json gives llm/voxtnew-llm its entitlements: no app without it)
scripts/mac.sh 'scripts/build-omni.sh && go tool mygo build'     # the Omni runtime (speech models) into the app; --out DIR for VOXTNEW_OMNI_RUNTIME
scripts/mac.sh 'scripts/build-llm.sh && go tool mygo build'      # voxtnew-llm (language models) into the app; --out DIR for VOXTNEW_LLM; on mac-air cp -Rc ~/voxtnew-runtime/llm resources/darwin-arm64/llm is quicker
scripts/mac.sh 'scripts/release.sh --runtimes $HOME/voxtnew-runtime'   # the release build: helper, omni, LLM, mygo build, release-verify.sh (docs/release.md); VOXTNEW_SIGN_IDENTITY signs it (team 779M533DL5, as the other ActNow apps: scripts/release-identity.sh), over ssh through scripts/mac-gui.sh
MAC_DIR=code/VoxtNew-xyz scripts/mac.sh 'VOXTNEW_COEXIST_OK=1 scripts/mac-hotkey-coexist.sh'   # a built app's helper beside a running Auralink: synthetic keys, right ⌥ fires only VoxtNew
MAC_DIR=code/VoxtNew-sign scripts/mac.sh scripts/mac-sign-e2e.sh  # the release's signing with a throwaway identity: every nested binary hardened and timestamped
scripts/mac.sh 'scripts/build-llm.sh --test'                     # its Swift tests (the same fixtures as Go)
scripts/mac.sh 'cd helper && swift test'                     # the helper's tests (same fixtures as Go)
MAC_DIR=code/VoxtNew-mt scripts/mac.sh scripts/mac-e2e.sh    # headless Mac tests: real helper, fake models, no window or key
MAC_DIR=code/VoxtNew-logsel scripts/mac.sh scripts/mac-logs-shots.sh  # the logs' selection worked with the real pointer and captured (only while nobody uses the Mac; VOXTNEW_SHOTS_DRY=1: the window only)
MAC_DIR=code/VoxtNew-xyz scripts/mac.sh scripts/mac-regress.sh  # the real models ($HUB, ~/voxtnew-runtime): Swift tests, omni, LLM, quick eval, app smoke
scripts/mac.sh 'VOXTNEW_EVAL=1 VOXTNEW_OMNI_RUNTIME=$HOME/voxtnew-runtime/omni VOXTNEW_LLM=$HOME/voxtnew-runtime/llm/voxtnew-llm go test -run TestEval -v .'  # the eval (docs/eval.md)
scripts/vendor-mygo.sh --check                               # third_party/mygo is MyGo + our patches
scripts/snapshots.sh check                                   # every snapshot, pixel for pixel, against the merge base
go run ./internal/tools/declhash . ./internal/...            # a hash per declaration: equal before and after a pure move
go vet -tags updatecheck .                                   # the update-check build
```
