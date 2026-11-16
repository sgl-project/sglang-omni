# Develop views with fixtures

Use this guide to change a meeting, history, session-overlay, updates, or home view without starting Voxt's helper, loading a model, granting microphone permission, or opening Voxt's data directory.

## Open the first fixture

Install Go 1.27.1 or a newer Go 1.27 patch release as described in [the development guide](development.md), then run every command from `Voxt/`.

1. Build and start the demo:

   ```sh
   MYGO_LOCALE=en VOXTNEW_DEMO_DIR="$PWD/testdata/meeting" GOFLAGS='-tags=meetingdemo' go tool mygo dev ./demo
   ```

2. Check that the **Voxt UI Demo** window shows `en/detail-answering`. This is the first fixture in sorted order.
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

The meeting views are `meetings_page.go`, `meetings_detail.go`, `meetings_playback.go`, `meeting_overlay.go`, and `meeting_overlay_view.go`. History uses `history_page.go`, `history_detail.go`, `history_menu.go`, `history_retry.go`, `history_settings.go`, `history_export.go`, and `history_access.go`. Session overlay views use `overlay.go`, `overlay_face.go`, `overlay_card.go`, `overlay_card_view.go`, `overlay_lang.go`, `overlay_size.go`, `overlay_think.go`, and `overlay_wave.go`. Updates uses `updates_view.go` for Settings → About and the release dialog. Home uses `home_page.go` and `home_ready.go`. These files read view state, call actions, and own local widget state such as scroll position and open menus. They do not read stores, snapshots, settings, or `app`.

The matching logic files build view state and implement actions:

- `meetings_view_state.go` and `meeting_overlay_state.go` build meeting display data. `meetings_controller.go`, `meetings_detail_controller.go`, `meetings_playback_controller.go`, and `meeting_overlay_window.go` own store reads, detail updates, audio work, and native resources.
- `history_view_state.go`, `history_settings_view_state.go`, and `history_access_view_state.go` build history display data. The history controllers own store reads, retries, recording jobs, exports, and the access socket.
- `session_overlay_state.go`, `session_overlay_card_state.go`, and `session_overlay_menu_state.go` build session display data. `session_overlay_window.go`, `session_overlay_size.go`, `session_overlay_menu.go`, and `session_overlay_stream.go` own native resources, measurement, menus, and stream timing. `sessionOverlayActions` in `session_overlay_state.go` defines user intents.
- `updates_view_state.go` builds Settings → About and release-dialog display data. `updates.go` and `updates_github.go` own updater I/O, release checks, watchers, and cards.
- `home_view_state.go` and `home_readiness_state.go` build home display data. `home_controller.go` owns history reads, readiness snapshots, and actions.
- `meeting_actions.go`, `history_actions.go`, `updates_actions.go`, and `home_actions.go` define user intents. `app` implements every intent in production. `meetingDemoHost` records them in the demo.

Keep dimensions, colors, and widget choices in the view files. Run `go tool mygo vet .` and `go test ./internal/archtest` after moving code.

## Add or change fixture data

Do not edit fixture JSON by hand. Change the production builder and its scenario in `meeting_fixture_scenarios_test.go`, `history_fixture_scenarios_test.go`, `session_fixture_scenarios_test.go`, `updates_fixture_scenarios_test.go`, or `home_fixture_scenarios_test.go`, then regenerate both languages:

```sh
VOXTNEW_UPDATE_GOLDEN=1 go test -count=1 -run '^TestMeetingViewStateGolden$' .
```

`TestMeetingViewStateGolden` builds every fixture through the production view-state builders. `TestMeetingFixtureCoverage`, `TestHistoryFixtureCoverage`, and `TestSessionFixtureCoverage` derive required states and problem codes from the domain declarations. `TestUpdatesFixtureCoverage` enumerates the updates display and release-check error branches, and `TestHomeFixtureCoverage` enumerates the home readiness actions. These tests fail when a fixture is missing.

If a view needs new data, ask the feature side to add it to the builder input and view state. Add a scenario that proves the value comes from real logic. For a new user intent, add the action to `meeting_actions.go`, `history_actions.go`, `sessionOverlayActions`, `updates_actions.go`, or `home_actions.go`, implement it on `app`, record it on `meetingDemoHost`, and add a click test to `TestMeetingDemoActionControls`, `TestHistoryDemoActionControls`, `TestSessionDemoActionControls`, `TestUpdatesDemoActionControls`, or `TestHomeDemoActionControls`.

## Add the next area or a fixture-backed page

Extend the existing fixture window instead of adding another startup path. A new area supplies its view state, actions, scenarios, and views. The integration points below are the only demo changes; `main.go`, `meeting_demo_enabled.go`, `meeting_demo_disabled.go`, and `demo/mygo.json` stay unchanged. Extract shared loading or selection code when another area needs it, and do not add a UI-version selector.

Use a short kind such as `note` for the new page. Keep the view in a `*_page.go` file with only its view state, actions, and local UI state. Then wire each required path:

1. Add an optional field for the new kind to `meetingFixtureState` in `meeting_fixture.go` and a matching case to `buildMeetingFixture` in `meeting_fixture_test.go`. Add a scenario to `meeting_fixture_scenarios_test.go` and regenerate the fixtures.
2. Accept the kind in `readMeetingDemoFixture` in `meeting_demo.go` and add its render branch to `meetingDemoView` in `meeting_demo_view.go`. Initialize and reset any local widget state in `selectMeetingFixture`.
3. Add the action methods to `meeting_actions.go`, implement them on `app`, and record them on `meetingDemoHost`.
4. Add the expected visible text to `meetingDemoSentinel` in `meeting_demo_test.go` and add controls to `TestMeetingDemoActionControls`.
5. For a production page, construct its state in the owning controller and register the view in `pages.go`. A demo-only exercise can stop after the fixture, view, and action tests.

Run `go test ./internal/archtest` to check the new view's imports and `go test -run '^(TestMeeting(ViewStateGolden|FixtureCoverage)|TestHistoryFixtureCoverage|TestSessionFixtureCoverage)$' .` to check existing fixture coverage. Add the new area's own golden, rendering, and action tests beside those suites.

## Render screenshots and check actions

Render every fixture in both languages:

```sh
VOXTNEW_DEMO_SHOTS=/tmp/voxt-meeting-demo go test -count=1 -run '^TestMeetingDemoScreenshots$' .
```

The test writes one PNG per fixture as `<locale>-<scenario>.png` and checks each fixture's expected text and non-empty frame.

Check fixture switching and control-to-action wiring:

```sh
go test -count=1 -run '^Test(MeetingDemo(ActionControls|SwitchesFixtures)|HistoryDemoActionControls|SessionDemo.*|SessionOverlayDemoMatchesProduction|UpdatesDemoActionControls|HomeDemoActionControls)$' .
go test -tags=meetingdemo -count=1 -run '^Test(Meeting|History|Session|Updates|Home)(Demo|OverlayDemoMatchesProduction)' .
```

The first command clicks every action-bearing meeting, history, session-overlay, updates, and home control and the local cancellation controls, then compares the exact recorded calls. It covers menus, confirmation dialogs, playback, search, settings, exports, cleanup, Privacy access, pill controls, answer-card controls, update preferences, the release dialog, readiness fixes, and recent results. It also compares production rendering with JSON fixtures. The second command compiles the enabled demo entry point.

These tests use `ui.Tester`. They need no window, helper binary, model, microphone, network, or Voxt data directory. They also run on Linux.

The demo does not verify native overlay placement, vibrancy, content protection, real audio capture, helper behavior, model output, global shortcuts, persistence, or file output. Check those on a Mac with the real app. The meeting settings-area dialog remains outside the demo, so meeting page fixtures hide its button. The feature-owned Try card is also outside the Home demo, so Home fixtures hide that section.
