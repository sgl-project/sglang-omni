# Develop meeting views with fixtures

Use this guide to change a meeting view without starting Voxt's helper, loading a model, granting microphone permission, or opening Voxt's data directory. Run every command from `Voxt/`.

## Open the first fixture

1. Build and start the demo:

   ```sh
   MYGO_LOCALE=en VOXTNEW_DEMO_DIR="$PWD/testdata/meeting" GOFLAGS='-tags=meetingdemo' go tool mygo dev ./demo
   ```

2. Click **Continue** or **Back** to switch fixtures without restarting the window.

The demo reads only `testdata/meeting`. Its MyGo configuration has no helper entitlement, model runtime, or permission declaration. The window has no saved state. Actions print records to stdout and do not change the fixture.

Normal builds do not enable the demo. Setting `VOXTNEW_DEMO_DIR` without `-tags=meetingdemo` returns an error before normal app startup.

To check the demo package without opening a window, run:

```sh
GOFLAGS='-tags=meetingdemo' go tool mygo build -skip-dmg ./demo
```

## Change a view

The meeting views are `meetings_page.go`, `meetings_detail.go`, `meetings_playback.go`, `meeting_overlay.go`, and `meeting_overlay_view.go`. They read view state, call actions, and own local widget state such as scroll position and open menus. They do not read stores, snapshots, settings, or `app`.

The matching logic files build view state and implement actions:

- `meetings_view_state.go` builds list, detail, and playback display data.
- `meeting_overlay_state.go` builds overlay display data.
- `meetings_controller.go`, `meetings_detail_controller.go`, and `meetings_playback_controller.go` own store reads, detail updates, and audio work.
- `meeting_overlay_window.go` owns the native overlay window and cached overlay state.
- `meeting_actions.go` defines the user intents. `app` implements them in production. `meetingDemoHost` records them in the demo.

Keep dimensions, colors, and widget choices in the view files. Run `go tool mygo vet .` and `go test ./internal/archtest` after moving code.

## Add or change fixture data

Do not edit fixture JSON by hand. Change the production builder and its scenario in `meeting_fixture_scenarios_test.go`, then regenerate both languages:

```sh
VOXTNEW_UPDATE_GOLDEN=1 go test -count=1 -run '^TestMeetingViewStateGolden$' .
```

`TestMeetingViewStateGolden` builds every fixture through the production view-state builders. `TestMeetingFixtureCoverage` derives required states and problem codes from the domain declarations and fails when a fixture is missing.

If a view needs new data, ask the feature side to add it to the builder input and view state. Add a scenario that proves the value comes from real logic. For a new user intent, add the action to `meeting_actions.go`, implement it on `app`, record it on `meetingDemoHost`, and add a click test to `TestMeetingDemoActionControls`.

## Render screenshots and check actions

Render every fixture in both languages:

```sh
VOXTNEW_DEMO_SHOTS=/tmp/voxt-meeting-demo go test -count=1 -run '^TestMeetingDemoScreenshots$' .
```

Check fixture switching and control-to-action wiring:

```sh
go test -count=1 -run '^TestMeetingDemo(ActionControls|SwitchesFixtures)$' .
```

These tests use `ui.Tester`. They need no window, helper binary, model, microphone, network, or Voxt data directory. They also run on Linux.

The demo does not verify native overlay placement, vibrancy, content protection, real audio capture, helper behavior, model output, global shortcuts, persistence, or file output. Check those on a Mac with the real app.
