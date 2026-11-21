package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"os"

	"github.com/sgl-project/sglang-omni/Voxt/internal/overlay/layout"
)

// meetingFixtureAdapt rewrites a fixture's JSON before decoding. Tests
// set it to adapt the checked-in macOS captures to the platform they run
// on (privacy deep links, shortcut glyphs); nil in the app.
var meetingFixtureAdapt func([]byte) []byte

func readMeetingDemoFixture(path string) (meetingFixtureState, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return meetingFixtureState{}, err
	}
	if meetingFixtureAdapt != nil {
		data = meetingFixtureAdapt(data)
	}
	decoder := json.NewDecoder(bytes.NewReader(data))
	decoder.DisallowUnknownFields()
	var fixture meetingFixtureState
	if err := decoder.Decode(&fixture); err != nil {
		return meetingFixtureState{}, err
	}
	if err := decoder.Decode(&struct{}{}); err != io.EOF {
		return meetingFixtureState{}, fmt.Errorf("trailing JSON value")
	}
	valid := fixture.Kind == "overlay" && fixture.Overlay != nil ||
		fixture.Kind == "page" && fixture.Page != nil ||
		fixture.Kind == "start" && fixture.Start != nil ||
		fixture.Kind == "history" && fixture.History != nil ||
		fixture.Kind == "history-settings" && fixture.HistorySettings != nil ||
		fixture.Kind == "history-access" && fixture.HistoryAccess != nil ||
		fixture.Kind == "session-overlay" && fixture.Session != nil && fixture.SessionUI != nil ||
		fixture.Kind == "updates-settings" && fixture.UpdatesSettings != nil ||
		fixture.Kind == "updates-release" && fixture.UpdatesRelease != nil ||
		fixture.Kind == "home" && fixture.Home != nil ||
		fixture.Kind == "onboarding" && fixture.Onboarding != nil ||
		fixture.Kind == "appenhance" && fixture.AppEnhancement != nil ||
		fixture.Kind == "dictionary" && fixture.Dictionary != nil ||
		fixture.Kind == "notes" && fixture.Notes != nil ||
		fixture.Kind == "remote" && fixture.Remote != nil ||
		fixture.Kind == "remote-cards" && fixture.RemoteCards != nil ||
		fixture.Kind == "remote-dialog" && fixture.RemoteDialog != nil ||
		fixture.Kind == "remote-debug" && fixture.RemoteDebug != nil ||
		fixture.Kind == "models" && fixture.Models != nil ||
		fixture.Kind == "feature" && fixture.Feature != nil ||
		fixture.Kind == "settings" && fixture.Settings != nil ||
		fixture.Kind == "meeting-settings" && fixture.MeetingSettings != nil ||
		fixture.Kind == "shortcut-prompt" && fixture.ShortcutPrompt != nil ||
		fixture.Kind == "key-recorder" && fixture.KeyRecorder != nil ||
		fixture.Kind == "shell" && fixture.Shell != nil && fixture.ShellUI != nil ||
		fixture.Kind == "shell-nav" && fixture.Nav != nil ||
		fixture.Kind == "shell-notices" && fixture.Notices != nil ||
		fixture.Kind == "shell-logs" && fixture.Logs != nil ||
		fixture.Kind == "shell-palette" && fixture.Palette != nil && fixture.ShellUI != nil ||
		fixture.Kind == "shell-sheet" && fixture.Sheet != nil && fixture.ShellUI != nil ||
		fixture.Kind == "shell-announcement" && fixture.Announcement != nil && fixture.ShellUI != nil
	if !valid {
		return meetingFixtureState{}, fmt.Errorf("kind %q has no matching state", fixture.Kind)
	}
	if fixture.Kind == "session-overlay" {
		menuKind := ""
		if fixture.Session.Menu != nil {
			menuKind = fixture.Session.Menu.Kind
		}
		if fixture.SessionUI.Menu != menuKind {
			return meetingFixtureState{}, fmt.Errorf("session menu seed %q does not match state %q", fixture.SessionUI.Menu, menuKind)
		}
		if card := fixture.Session.Card; card != nil {
			if card.SizeStep < 0 || card.SizeStep >= len(layout.CardSizes) {
				return meetingFixtureState{}, fmt.Errorf("session card size step %d is outside 0..%d", card.SizeStep, len(layout.CardSizes)-1)
			}
			if card.LiveCount < 0 || card.LiveCount > len(card.Live) {
				return meetingFixtureState{}, fmt.Errorf("session card live count %d is outside 0..%d", card.LiveCount, len(card.Live))
			}
		}
	}
	return fixture, nil
}
