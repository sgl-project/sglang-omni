# Building and developing Voxt

**English** · [简体中文](development.zh-CN.md)

How to build Voxt from source, sign it during development and run its tests. To install and use Voxt, see the [README](../README.md); to send a change, [CONTRIBUTING.md](../CONTRIBUTING.md).

## What is inside

- **Speech recognition** and **voice detection** run on sglang-omni's native MLX servers ([`sglang_omni_mlx/native`](../../sglang_omni_mlx/native), C++, no Python), one process per model, started when needed, stopped after 10 minutes unused (Speech models → Advanced changes it).
- A small Swift helper does what needs Apple's frameworks: the microphone, global shortcuts, Accessibility, pasting.

## Build from source

Needs Go 1.27.4 (`GOTOOLCHAIN=auto` fetches it) and, to build the app, an Apple silicon Mac with macOS 15 or later, Xcode (the helper), and [uv](https://docs.astral.sh/uv/) or Python 3 for the Omni runtime's build tools (`scripts/build-omni.sh` builds `sglang_omni_mlx/native` and, the first time, the MLX commit it pins; git and CMake fetch MLX and the C++ dependencies from GitHub and metal-cpp from developer.apple.com, through the shell's `https_proxy` if needed). Run from `Voxt/`:

```sh
scripts/build-helper.sh              # the Swift helper, into resources/darwin-arm64/helper/
scripts/build-omni.sh                # the Omni runtime (speech models), into resources/darwin-arm64/omni/
go tool mygo dev                     # run with hot reload
go tool mygo build                   # build/darwin-arm64/Voxt.app and its disk image
```

The helper must be built before `go tool mygo build` or `mygo dev`: `mygo.json` gives it its entitlements and fails without it. Without the Omni runtime the app runs but cannot recognise speech (or point `VOXTNEW_OMNI_RUNTIME` at one built with `scripts/build-omni.sh --out DIR`).

**The release build** is one command, on the Mac: `scripts/release.sh` builds the helper and the Omni runtime, packages and signs the app, and checks the result (`--runtimes DIR` takes prebuilt runtimes; with a Developer ID identity, `--notarize` notarises it; over ssh it runs through `scripts/mac-gui.sh`). A release is that build on GitHub: a `voxt-vX.Y.Z` tag runs the Voxt Release workflow, which signs it with the Developer ID of the team RY2TQTYSKQ, notarises it and puts its disk image and signed update files on a draft GitHub release. Local builds without a feed keep the GitHub update check. See [docs/release.md](release.md).

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

## Environment variables

This table lists every `VOXTNEW_*` variable that the Go code, the Swift helper or `scripts/` reads. `TestDocsEnvironmentVariables` fails when a variable has no row. When the table does not give a default, the variable is unset by default and the feature or test that it controls is off.

| Variable | What it does | Default |
|---|---|---|
| `VOXTNEW_ACCEPT_PROXY` | Upstream proxy for the network that `TestAccept` in `internal/models` uses. | Unset: direct connection |
| `VOXTNEW_APP` | The app whose Omni runtime `scripts/mac-latency.sh` uses. | `/Applications/Voxt.app` |
| `VOXTNEW_COEXIST_OK` | `1` lets `scripts/mac-hotkey-coexist.sh` post key events on the Mac. | Unset: the script stops |
| `VOXTNEW_DEMO_DIR` | Starts the app as the fixture demo with the fixtures in this directory (needs `-tags=meetingdemo`). | Unset: the normal app |
| `VOXTNEW_DEMO_SHOTS` | Directory where `TestMeetingDemoScreenshots` writes every fixture as PNG. | Unset: assertions run without writing PNGs |
| `VOXTNEW_DSP_OUT` | Directory where `TestCompareVariants` in `internal/dsp` writes each variant as WAV. | Unset: log only |
| `VOXTNEW_E2E_ICONS` | Directory where the Mac end-to-end test saves the app icons that the real helper returns. | Unset: icons are not saved |
| `VOXTNEW_EVAL` | `1` runs the eval on the real models (`TestEvalASR`, [eval.md](eval.md)). | Unset: the eval skips |
| `VOXTNEW_EVAL_ASR` | Comma-separated speech model ids for the dictionary and long-audio evals. | Dictionary eval: `asr.accurate` and `asr.fast`; long-audio eval: `asr.accurate` |
| `VOXTNEW_EVAL_ONLY` | Comma-separated clip ids: the eval runs only these clips. | Unset: all clips |
| `VOXTNEW_EVAL_OUT` | Directory where the eval writes its reports. | Unset: no files |
| `VOXTNEW_EVAL_PIECES` | Comma-separated ways that the long-audio eval cuts audio into pieces. | `model` |
| `VOXTNEW_EVAL_QUICK` | `1` runs one clip of each kind and checks the scores against floors. | Unset: the full eval |
| `VOXTNEW_EVAL_WAYS` | Comma-separated ways (`whole`, `pieces`) that the long-audio eval runs. | Unset: all ways |
| `VOXTNEW_FAKE_HELPER` | Options for the fake helper. When it is set, a test binary runs as the fake helper. | Unset: a normal test run |
| `VOXTNEW_FAMILY_ACCEPT` | `1` runs the model library acceptance test on the real Hugging Face cache and network (macOS). | Unset: the test skips |
| `VOXTNEW_FLOWS` | Any value makes `TestOverlayFlowsTrace` log the steps of each overlay flow. | Unset: the test skips |
| `VOXTNEW_FLOW_SHOTS` | Directory where the overlay flow tests write a picture of each step. | Unset: no pictures |
| `VOXTNEW_HELPER` | Path of the `voxtnew-helper` executable to run. | Unset: the helper in the app bundle |
| `VOXTNEW_HELPER_CAPTURE_FILE` | Tests only: the helper records this audio file instead of the microphone. | Unset: the microphone |
| `VOXTNEW_HELPER_CAPTURE_SPEED` | Tests only: how many times faster than real time the helper plays the capture file. | `1` |
| `VOXTNEW_HELPER_SYSTEM_FILE` | Tests only: the helper plays this audio file as the system audio capture. | Unset: the real system audio |
| `VOXTNEW_HOME_SHOTS` | Directory where `TestHomeSnapshots` writes the Home page as PNG. | Unset: the test skips |
| `VOXTNEW_HUB` | Model library that `scripts/mac-latency.sh` reads. | `~/Library/Application Support/SGLang Omni/models` |
| `VOXTNEW_INK` | Directory where the overlay tests write the rendered cards as PNG. | Unset: no pictures |
| `VOXTNEW_LATENCY` | `1` runs `TestMacE2EFirstLatency` on a Mac. | Unset: the test skips |
| `VOXTNEW_LATENCY_ARM` | A/B arm label written to each latency JSON row, for profiling runs on a Mac. | Unset: no arm label |
| `VOXTNEW_LATENCY_ATTEMPT` | Independent attempt label written to each latency JSON row, for profiling runs on a Mac. | Unset: attempt `1` |
| `VOXTNEW_LATENCY_OUT` | File where the latency test appends one JSON object per session. | Unset: no JSON output |
| `VOXTNEW_MAIN_SHA` | Source commit recorded in latency JSON, for profiling runs on a Mac. | Unset: `unknown` |
| `VOXTNEW_LATENCY_PCM` | Recording that the latency test plays. | Unset: the bundled speech sample |
| `VOXTNEW_LATENCY_ROUNDS` | Number of rounds of the latency test. | `3` |
| `VOXTNEW_LATENCY_SETTLE` | Wait after the app is ready, before the first session of the latency test. | `3s` |
| `VOXTNEW_LATENCY_SPEAK` | How long a latency test session records after its first speech. | `2s` |
| `VOXTNEW_LEGACY_HOME` | Home directory where the import of Voxt 1.x data looks, for tests that run the app with a fake home. | Unset: the real home |
| `VOXTNEW_MAC` | Any value runs the tests that use real macOS services (Keychain, permissions, app activation). | Unset: these tests skip |
| `VOXTNEW_MAC_E2E` | `1` runs the headless end-to-end tests on a real Mac. | Unset: the tests skip |
| `VOXTNEW_MAC_MIC_CAPTURE` | `1` lets the Mac end-to-end test record from the real microphone. | Unset: the test skips |
| `VOXTNEW_MAC_OUTPUT` | UID of an output device that the output device test can switch to. | Unset: the test skips |
| `VOXTNEW_MAC_OUTPUT_DEFAULT` | `1` lets the output device test change the default output device. | Unset: not allowed |
| `VOXTNEW_MIN_FREE` | `scripts/mac-beta-e2e.sh` and `scripts/mac-regress.sh` stop when less than this percentage of memory is free. | `40` in `mac-beta-e2e.sh`, `30` in `mac-regress.sh` |
| `VOXTNEW_MODELS_REAL` | `1` runs the `internal/models` tests that use the real Hugging Face Hub. | Unset: the tests skip |
| `VOXTNEW_MOVE_AS` | App id that claims the models that `TestRealMoveToHub` moves. | Unset: the VoxtNew id |
| `VOXTNEW_MOVE_FROM` | Old model directory that `TestRealMoveToHub` moves into the model library. | Unset: the test skips |
| `VOXTNEW_MOVE_ONLY` | Comma-separated model ids: `TestRealMoveToHub` moves only these models. | Unset: all models |
| `VOXTNEW_MTD_DIR` | MTD model snapshot that `TestEvalMTD` measures. | Unset: catalogue models only |
| `VOXTNEW_MTD_DRAFT_DIR` | DFlash draft model snapshot that `TestEvalMTD` uses with the MTD model. | Unset: no draft |
| `VOXTNEW_OMNI_MODELS` | Speech models for the latency test, as `id=folder;…`. | Set by `scripts/mac-latency.sh` |
| `VOXTNEW_OMNI_RUNTIME` | Directory of the Omni runtime (`scripts/build-omni.sh --out DIR`). | Unset: the runtime in the app bundle |
| `VOXTNEW_INTERACTIVE_CONFIG` | JSON config for the interactive-priority profiling experiment, run explicitly on a Mac. | Unset: `TestMeetingInteractive` skips |
| `VOXTNEW_OVERLAP_CONFIG` | JSON config for the real-ASR overlap profiling experiment, run explicitly on a Mac. | Unset: `TestMeetingOverlap` skips |
| `VOXTNEW_PROFILE_CONFIG` | JSON config for long-recording profiling, run explicitly on a Mac. | Unset: `TestMeetingProfile` skips |
| `VOXTNEW_OVERLAY_SHOTS` | Directory where the overlay snapshot tests write PNGs. | Unset: the tests skip |
| `VOXTNEW_REAL_CUSTOM` | Custom language model for `TestRealServices`, as `<address>\|<model>`. | Unset: the custom case skips |
| `VOXTNEW_REAL_SERVICES` | `1` runs `TestRealServices` against the remote services whose keys are in the login Keychain (macOS). | Unset: the test skips |
| `VOXTNEW_RUNTIME_DIR` | Directory with the runtimes that `scripts/mac-regress.sh` uses. | `~/voxtnew-runtime` |
| `VOXTNEW_SHOTS` | Directory where the Transcribe page layout test writes its PNGs. | Unset: no pictures |
| `VOXTNEW_SHOTS_DRY` | `1` makes `scripts/mac-logs-shots.sh` only open the window, without moving the pointer. | Unset: the full run |
| `VOXTNEW_SHOTS_IDLE` | Seconds without user input before `scripts/mac-logs-shots.sh` moves the pointer. | `300` |
| `VOXTNEW_SHOTS_ORDER` | `1` makes `scripts/mac-logs-shots.sh` capture the log order instead. | Unset: the normal captures |
| `VOXTNEW_SIGN_IDENTITY` | Code signing identity for the helper and the release build. | Unset: ad hoc signing |
| `VOXTNEW_SMOKE_LEGACY` | `1` runs the import check of the upgrade test in the app. | Unset: no check |
| `VOXTNEW_SMOKE_MODELS` | Speech model id that the app smoke test opens. | Unset: the first installed speech model |
| `VOXTNEW_SNAPSHOTS` | Directory where `TestSnapshots` and the page snapshot tests write PNGs. | Unset: the tests skip |
| `VOXTNEW_SNAPSHOTS_OUT` | File where `TestRealSnapshots` in `internal/models` writes its list. | Unset: the test skips |
| `VOXTNEW_SNAPSHOTS_SIZE` | Window size for the snapshots, as `WxH` or `min`. | Unset: the default window size |
| `VOXTNEW_SNAPSHOT_ICONS` | Directory of app icon PNGs that the fake helper of the snapshot tests returns. | Unset: no icons |
| `VOXTNEW_SWIFT_DISABLE_SANDBOX` | `1` makes `scripts/build-helper.sh` pass `--disable-sandbox` to SwiftPM, for a caller that already isolates the build. | Unset: SwiftPM's own sandbox |
| `VOXTNEW_TEXTFIT_AUDIT` | `1` lists text that does not fit. `strict` fails on any. | Unset: no audit |
| `VOXTNEW_TEXTFIT_AUDIT_OUT` | File where the text fit audit writes its list. | Unset: log only |
| `VOXTNEW_TEXTFIT_PROBE` | Any value runs the probe of the MyGo text engine (useful on macOS). | Unset: the probe skips |
| `VOXTNEW_UPDATE_GOLDEN` | `1` rewrites golden files and fixtures instead of comparing with them. | Unset: compare |
| `VOXTNEW_UPDATE_KEY` | File with the update signing key for `scripts/release.sh`. | `~/.config/voxtnew-release/update-keys/mygo-update.key` |
| `VOXTNEW_W0_RESOURCES` | `1` runs the resource baseline measurement (`scripts/benchmarks.sh`). | Unset: the test skips |
| `VOXTNEW_W0_STREAM_LENGTHS` | File where the real stream path test writes the answer stream lengths. | Unset: not written |
| `VOXTNEW_W0_STREAM_PATH` | `1` runs the test of the real answer stream path. | Unset: the test skips |

## Change fixture-backed view state and actions

The [UI development guide](ui-development.md) explains how to render fixtures and change meeting, history, session-overlay, updates, home, onboarding, app enhancement, dictionary, Notes, remote, models, feature, settings, or shell views without the helper or models. Feature-side changes use the same boundary:

1. Add display data to the input and result of the meeting builders, the history builders, the session builders in `session_overlay_state.go`, `session_overlay_card_state.go`, and `session_overlay_menu_state.go`, `buildUpdatesViewState` and `buildReleaseViewState`, `buildHomeViewState`, `buildOnboardingViewState`, `buildAppEnhancementViewState`, the dictionary and Notes builders, `buildRemotePageViewState`, `buildRemoteDialogViewState`, and `buildLLMDebugViewState`, `buildModelsViewState`, the feature builders in `feature_view_state.go`, `feature_shortcut_view_state.go`, `feature_prompt_view_state.go`, `feature_page_view_state.go`, and `try_view_state.go`, or the settings section builders and the shell builders in `shell_nav_view_state.go`, `shell_notices_view_state.go`, `shell_logs_view_state.go`, `shell_commands_view_state.go`, `shell_announcement_view_state.go`, and `shell_view_state.go`. Keep domain types and I/O out of the view state.
2. Add a scenario to `meeting_fixture_scenarios_test.go`, `history_fixture_scenarios_test.go`, `session_fixture_scenarios_test.go`, `updates_fixture_scenarios_test.go`, `home_fixture_scenarios_test.go`, `onboarding_fixture_scenarios_test.go`, `appenhance_fixture_scenarios_test.go`, `dictionary_fixture_scenarios_test.go`, `remote_fixture_scenarios_test.go`, `models_fixture_scenarios_test.go`, `feature_fixture_scenarios_test.go`, `settings_fixture_scenarios_test.go`, or `shell_fixture_test.go` and regenerate the fixtures with `VOXTNEW_UPDATE_GOLDEN=1 go test -count=1 -run '^TestMeetingViewStateGolden$' .`.
3. For a new user intent, add the method to the area's actions in `meeting_actions.go`, `history_actions.go`, `sessionOverlayActions`, `updates_actions.go`, `home_actions.go`, `onboarding_actions.go`, `appenhance_actions.go`, `dictionary_actions.go`, `remote_actions.go`, `models_actions.go`, `feature_actions.go`, `try_actions.go`, `settings_actions.go`, `settings_dialog_actions.go`, `settings_about_actions.go`, `permissions.go`, or `shell_actions.go`, implement it on `app`, and record it on `meetingDemoHost`. Views call one intent instead of assembling a business workflow.
4. Add a click test to `TestMeetingDemoActionControls`, `TestHistoryDemoActionControls`, `TestSessionDemoActionControls`, `TestUpdatesDemoActionControls`, `TestHomeDemoActionControls`, `TestOnboardingDemoActionControls`, `TestAppEnhancementDemoActionControls`, `TestDictionaryDemoActionControls`, `TestRemoteDemoActionControls`, `TestModelsDemoActionControls`, `TestFeatureDemoActionControls`, `TestSettingsDemoActionControls`, or `TestShellDemoActionControls`, then run `go test ./internal/archtest` and `go test -run '^(TestMeeting(ViewStateGolden|FixtureCoverage)|TestHistoryFixtureCoverage|TestSessionFixtureCoverage|TestUpdatesFixtureCoverage|TestHomeFixtureCoverage|TestOnboardingFixtureCoverage|TestAppEnhancementFixtureCoverage|TestDictionaryFixtureCoverage|TestRemoteFixtureCoverage|TestModelsFixtureCoverage|TestFeatureFixtureCoverage|TestSettingsFixtureCoverage|TestShellFixtureCoverage)$' .`.

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
