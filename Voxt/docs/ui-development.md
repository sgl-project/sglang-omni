# Develop views with fixtures

Use this guide to change a meeting, history, session-overlay, updates, home, onboarding, app enhancement, dictionary, Notes, remote, models, feature, settings, or shell view without starting Voxt's helper, loading a model, granting microphone permission, or opening Voxt's data directory.

## Open the first fixture

Install Go 1.27.1 or a newer Go 1.27 patch release as described in [the development guide](development.md), then run every command from `Voxt/`.

1. Build and start the demo:

   ```sh
   MYGO_LOCALE=en VOXTNEW_DEMO_DIR="$PWD/testdata/meeting" GOFLAGS='-tags=meetingdemo' go tool mygo dev ./demo
   ```

2. Check that the **Voxt UI Demo** window shows `en/appenhance-empty`. This is the first non-modal fixture in sorted order.

3. Click **Continue** or **Back** to switch fixtures without restarting the window. The selector wraps at either end and resets local widget state.
4. Click **Close** in a detail fixture. The terminal prints the recorded action without changing the fixture:

   ```text
   meeting demo action: closeMeetingDetail
   ```

Keep `mygo dev` running while you change view code. It rebuilds and restarts the demo after a successful build. Fixture selection loads JSON once at startup, so restart the demo after regenerating fixture files.

The demo reads only `testdata/meeting`. Its MyGo configuration has no helper entitlement, model runtime, or permission declaration. `MYGO_LOCALE` prevents MyGo from reading or saving a language preference; the demo sets it to the first fixture's locale when you omit it. Actions only print records to stdout.

Normal builds do not enable the demo. Setting `VOXTNEW_DEMO_DIR` without `-tags=meetingdemo` returns an error before normal app startup. No user-facing setting enables demo mode.

To check the demo package without opening a window, run:

```sh
GOFLAGS='-tags=meetingdemo' go tool mygo build -skip-dmg ./demo
```

To render fixtures without a window, use [the screenshot command](#render-screenshots-and-check-actions). It creates the output directory.

## Change a view

The meeting views are `meetings_page.go`, `meetings_detail.go`, `meetings_playback.go`, `meeting_overlay.go`, and `meeting_overlay_view.go`. History uses `history_page.go`, `history_detail.go`, `history_menu.go`, `history_retry.go`, `history_settings.go`, `history_export.go`, and `history_access.go`. Session overlay views use `overlay.go`, `overlay_face.go`, `overlay_card.go`, `overlay_card_view.go`, `overlay_lang.go`, `overlay_size.go`, `overlay_think.go`, and `overlay_wave.go`. Updates uses `updates_view.go` for Settings → About and the release dialog. Home uses `home_page.go` and `home_ready.go`. Onboarding uses `onboarding_keys.go`, `onboarding_models.go`, `onboarding_steps.go`, `onboarding_try_guide.go`, and `onboarding_view.go`. App enhancement uses `appenhance_page.go`, `appenhance_group.go`, `appenhance_dialogs.go`, `appenhance_preview.go`, and `appicons.go`. Dictionary uses `dictionary_page.go`, `dictionary_notes.go`, `dictionary_scan.go`, `dictionary_suggest.go`, and `notes_page.go`. Remote uses `remote_page.go`, `remote_dialog.go`, `remote_generation.go`, `remote_chatgpt.go`, `llm_wait.go`, and `llm_debug.go`. Models uses `models_page.go`, `models_cards.go`, `models_context.go`, `models_delete.go`, `models_keep.go`, and `models_network.go`. The feature area uses `feature_page.go`, `feature_keys.go`, `feature_models.go`, `feature_prompts.go`, `feature_endword.go`, and `try_card.go`. Settings uses `shell_settings.go`, `permissions.go`, `key_recorder.go`, and the section view files. The shell uses `shell.go`, `shell_nav.go`, `shell_commands.go`, `shell_notices.go`, `shell_logs.go`, `shell_logs_order.go`, `shell_logs_select.go`, `pages.go`'s fallback page, and `announcements.go`. These files read view state, call actions, and own local widget state such as scroll position and open menus. They do not read stores, snapshots, settings, or `app`.

The matching logic files build view state and implement actions:

- `meetings_view_state.go`, `meeting_start_state.go`, and `meeting_overlay_state.go` build meeting display data. `meetings_controller.go`, `meetings_detail_controller.go`, `meetings_playback_controller.go`, and `meeting_overlay_window.go` own store reads, detail updates, audio work, and native resources.
- `history_view_state.go`, `history_settings_view_state.go`, and `history_access_view_state.go` build history display data. The history controllers own store reads, retries, recording jobs, exports, and the access socket.
- `session_overlay_state.go`, `session_overlay_card_state.go`, and `session_overlay_menu_state.go` build session display data. `session_overlay_window.go`, `session_overlay_size.go`, `session_overlay_menu.go`, and `session_overlay_stream.go` own native resources, measurement, menus, and stream timing. `sessionOverlayActions` in `session_overlay_state.go` defines user intents.
- `updates_view_state.go` builds Settings → About and release-dialog display data. `updates.go` and `updates_github.go` own updater I/O, release checks, watchers, and cards.
- `home_view_state.go` and `home_readiness_state.go` build home display data. `home_controller.go` owns history reads, readiness snapshots, and actions.
- `onboarding_view_state.go`, `onboarding_models_state.go`, `onboarding_steps_state.go`, and `onboarding_try_state.go` build onboarding display data. `onboarding.go` owns the guide's lifecycle and model surveys.
- `appenhance_view_state.go` builds app enhancement display data. `appenhance_controller.go` owns rule edits, settings changes, and app-list reads. `appicons_controller.go` owns icon reads and bitmap decoding.
- `dictionary_view_state.go`, `dictionary_notes_state.go`, and `notes_view_state.go` build dictionary and Notes display data. The dictionary and notes controllers own store reads, settings writes, imports and exports, scans, autosave, and shortcut recording.
- `remote_view_state.go`, `remote_dialog_view_state.go`, and `llm_debug_view_state.go` build remote display data. The remote controllers own settings, credentials, model discovery, connection tests, sign-in, and debug requests.
- `models_view_state.go` builds models display data. The models controllers own downloads, deletion, shared-family settings, runtime memory measurements, and the page-state cache.
- `feature_view_state.go`, `feature_shortcut_view_state.go`, `feature_prompt_view_state.go`, `feature_page_view_state.go`, and `try_view_state.go` build feature display data. The feature controllers own settings updates, shortcut parsing, prompt composition, model selection, and Try sessions.
- `settings_dialog_view_state.go` selects one Settings section, and the section builders create its display data. The settings controllers own preferences, helper and permission reads, storage and diagnostics work, sounds, and model operations.
- `shell_nav_view_state.go`, `shell_notices_view_state.go`, `shell_logs_view_state.go`, `shell_commands_view_state.go`, `shell_announcement_view_state.go`, and `shell_view_state.go` build the shell's display data. The shell's controllers own navigation, the command registry, the notification centre's snapshot, the log's tail, the announcements, and the window's wiring (`shell_controller.go`, `shell_nav_controller.go`, `shell_notices_controller.go`, `shell_logs_controller.go`, `shell_commands_controller.go`, `shell_announcement_controller.go`).
- `meeting_actions.go`, `history_actions.go`, `updates_actions.go`, `home_actions.go`, `onboarding_actions.go`, `appenhance_actions.go`, `dictionary_actions.go`, `remote_actions.go`, `models_actions.go`, `feature_actions.go`, `try_actions.go`, `settings_actions.go`, `settings_dialog_actions.go`, `settings_about_actions.go`, `permissions.go`, and `shell_actions.go` define user intents. `app` implements every intent in production. `meetingDemoHost` records them in the demo.

Keep dimensions, colors, and widget choices in the view files. Run `go tool mygo vet .` and `go test ./internal/archtest` after moving code.

## Add or change fixture data

Do not edit fixture JSON by hand. Change the production builder and its scenario in `meeting_fixture_scenarios_test.go`, `history_fixture_scenarios_test.go`, `session_fixture_scenarios_test.go`, `updates_fixture_scenarios_test.go`, `home_fixture_scenarios_test.go`, `onboarding_fixture_scenarios_test.go`, `appenhance_fixture_scenarios_test.go`, `dictionary_fixture_scenarios_test.go`, `remote_fixture_scenarios_test.go`, `models_fixture_scenarios_test.go`, `feature_fixture_scenarios_test.go`, `settings_fixture_scenarios_test.go`, or `shell_fixture_test.go`, then regenerate both languages:

```sh
VOXTNEW_UPDATE_GOLDEN=1 go test -count=1 -run '^TestMeetingViewStateGolden$' .
```

`TestMeetingViewStateGolden` builds every fixture through the production view-state builders. `TestMeetingFixtureCoverage`, `TestHistoryFixtureCoverage`, and `TestSessionFixtureCoverage` derive required states and problem codes from the domain declarations. `TestUpdatesFixtureCoverage` enumerates the updates display and release-check error branches, `TestHomeFixtureCoverage` enumerates the home readiness actions, `TestOnboardingFixtureCoverage` enumerates the onboarding steps, permission states, and shortcut presets, `TestAppEnhancementFixtureCoverage` enumerates the app enhancement match sources, `TestDictionaryFixtureCoverage` derives the dictionary suggestion sources and checks the scan stages, `TestRemoteFixtureCoverage` enumerates the remote missing and problem codes, generation choices, and model waits, `TestModelsFixtureCoverage` enumerates the model kinds, runtimes, download errors, and keep-loaded choices, `TestFeatureFixtureCoverage` enumerates the feature and shortcut problem codes, `TestSettingsFixtureCoverage` checks the settings sections, permission states, sound sources, and capture modes, and `TestShellFixtureCoverage` checks the shell's fixture kinds, notice severities, and log levels and derives the helper failure reasons. These tests fail when a fixture is missing.

If a view needs new data, ask the feature side to add it to the builder input and view state. Add a scenario that proves the value comes from real logic. For a new production user intent, add the action to the area's `*_actions.go` file, implement it on `app`, record it on `meetingDemoHost`, and add a click test to the area's demo action test.

For a genuinely new area, add its prefix to `catalogAreas` in `internal/i18n/parity_test.go` and add a `catalogAliases` entry if its root file prefix differs from the area name. Put the area's messages in `<area>.json`, not `common.json`.

## Add the next area or a fixture-backed page

Extend the existing fixture window instead of adding another startup path. A new area supplies its view state, actions, scenarios, and views. The integration points below are the only demo changes; `main.go`, `meeting_demo_enabled.go`, `meeting_demo_disabled.go`, and `demo/mygo.json` stay unchanged. Extract shared loading or selection code when another area needs it, and do not add a UI-version selector.

Use a short kind such as `note` for the new page. Keep the view in a `*_page.go` file with only its view state, actions, and local UI state. Then wire each required path:

1. Add an optional field for the new kind to `meetingFixtureState` in `meeting_fixture.go` and a matching case to `buildMeetingFixture` in `meeting_fixture_test.go`. Add a scenario to `meeting_fixture_scenarios_test.go` and regenerate the fixtures.
2. Accept the kind in `readMeetingDemoFixture` in `meeting_demo.go` and add its render branch to `meetingDemoView` in `meeting_demo_view.go`. Initialize and reset any local widget state in `selectMeetingFixture`.
3. Add the action methods to the area's `*_actions.go` file and record them on `meetingDemoHost`. A production page also implements them on `app`. A demo-only exercise needs only the recording implementation.
4. Add the expected visible text to `meetingDemoSentinel` in `meeting_demo_test.go` and add controls to the area's demo action test.
5. For a production page, construct its state in the owning controller and register the view in `pages.go`. A demo-only exercise can stop after the fixture, view, and action tests.

`TestFileSizes` limits production files to 600 lines and test files to 800. If the integration would cross either limit, split the demo code by topic into `<area>_demo.go` or `<area>_demo_test.go` before adding it.

Run `go test ./internal/archtest` to check the new view's imports and `go test -run '^(TestMeeting(ViewStateGolden|FixtureCoverage)|TestHistoryFixtureCoverage|TestSessionFixtureCoverage|TestUpdatesFixtureCoverage|TestHomeFixtureCoverage|TestOnboardingFixtureCoverage|TestAppEnhancementFixtureCoverage|TestDictionaryFixtureCoverage|TestRemoteFixtureCoverage|TestModelsFixtureCoverage|TestFeatureFixtureCoverage|TestSettingsFixtureCoverage|TestShellFixtureCoverage)$' .` to check existing fixture coverage. Add the new area's own golden, rendering, and action tests beside those suites.

## Render screenshots and check actions

Render every fixture in both languages:

```sh
VOXTNEW_DEMO_SHOTS=/tmp/voxt-meeting-demo go test -count=1 -run '^TestMeetingDemoScreenshots$' .
```

The test writes one PNG per fixture as `<locale>-<scenario>.png` and checks each fixture's expected text and non-empty frame.

Check fixture switching and control-to-action wiring:

```sh
go test -count=1 -run '^Test(MeetingDemo(ActionControls|SwitchesFixtures)|HistoryDemoActionControls|SessionDemo.*|SessionOverlayDemoMatchesProduction|UpdatesDemoActionControls|HomeDemoActionControls|OnboardingDemo|AppEnhancementDemoActionControls|DictionaryDemoActionControls|RemoteDemoActionControls|ModelsDemo|FeatureDemoActionControls|SettingsDemoActionControls|ShellDemoActionControls)$' .
go test -tags=meetingdemo -count=1 -run '^Test(Meeting|History|Session|Updates|Home|Onboarding|AppEnhancement|Dictionary|Remote|Models|Feature|Settings|Shell)(Demo|OverlayDemoMatchesProduction)' .
```

The first command clicks the action-bearing meeting, history, session-overlay, updates, home, onboarding, app enhancement, dictionary, remote, models, feature, settings, and shell controls covered by these demo suites and the local cancellation controls, then compares the exact recorded calls. It covers menus, confirmation dialogs, playback, search, settings, exports, cleanup, Privacy access, pill controls, answer-card controls, update preferences, the release dialog, readiness fixes, recent results, permissions, model choices, onboarding tries, rule editors, browser lists, dictionary terms, notes, provider cards, service dialogs, the LLM debug view, model downloads, deletion, Network, keep-loaded choices, shortcut editing, model selection, prompt controls, Try sessions, the settings sections, recorder states, and storage and diagnostics controls, and the shell's navigation, notice cards, log filtering and copy-all, command palette, shortcut sheet, and announcement dialog. It also compares production rendering with JSON fixtures. The second command compiles the enabled demo entry point.

These tests use `ui.Tester`. They need no window, helper binary, model, microphone, network, or Voxt data directory. They also run on Linux.

The shell demo leaves log selection, sidebar-width changes, `openMainSettings`, `openLogFile`, `closeLogsPanel`, and `toggleLogsPanel` to production tests and real-app checks.

The demo does not verify native overlay placement, vibrancy, content protection, real audio capture, helper behavior, model output, real downloads or deletion, runtime behavior, proxy or network behavior, global shortcuts, persistence, or file output. Check those on a Mac with the real app. The feature-owned Try card is also outside the Home demo, so Home fixtures hide that section. Onboarding and remote-provider views also remain outside models fixtures. Main Settings and meeting-owned Meeting settings have fixtures. The updater and feature-shortcut subviews stay with their areas.
