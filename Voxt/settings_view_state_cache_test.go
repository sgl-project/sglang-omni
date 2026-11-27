package main

import (
	"slices"
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/permission"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
)

func TestGeneralSettingsModelCacheInvalidatesOnPreferences(t *testing.T) {
	a, _, _ := newTestApp(t)
	first := a.generalSettingsModel()
	cached := a.generalSettingsModel()
	if &first.LanguageOptions[0] != &cached.LanguageOptions[0] {
		t.Fatal("the same preference revision rebuilt the general view state")
	}

	revision := a.settingsRevision
	a.updateSettings(func(current *settings.Settings) { current.General.CopyResult = !first.CopyResult })
	if a.settingsRevision == revision {
		t.Fatal("updateSettings did not advance the preference revision")
	}
	if got := a.generalSettingsModel(); got.CopyResult == first.CopyResult {
		t.Fatal("updateSettings reused stale general view state")
	}

	oldPreferences := a.prefs
	newPreferences := oldPreferences
	newPreferences.General.CloseToTray = !oldPreferences.General.CloseToTray
	revision = a.settingsRevision
	a.settingsChanged(oldPreferences, newPreferences)
	if a.settingsRevision == revision {
		t.Fatal("settingsChanged did not advance the preference revision")
	}
	if got := a.generalSettingsModel(); got.CloseToTray != newPreferences.General.CloseToTray {
		t.Fatal("settingsChanged reused stale general view state")
	}
}

func TestPermissionsModelCacheInvalidatesOnValues(t *testing.T) {
	tests := []struct {
		name   string
		change func(*app)
		shown  func(permissionsViewState) bool
	}{
		{name: "microphone", change: func(a *app) { a.perms.mic = permission.Granted }, shown: func(state permissionsViewState) bool {
			return state.Microphone.Allowed
		}},
		{name: "speech", change: func(a *app) { a.perms.speech = permission.Granted }, shown: func(state permissionsViewState) bool {
			return state.SpeechRecognition.Allowed
		}},
		{name: "accessibility", change: func(a *app) { a.perms.access = true }, shown: func(state permissionsViewState) bool {
			return state.Accessibility.Allowed
		}},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			a := &app{}
			a.perms.mic, a.perms.speech = permission.NotDetermined, permission.NotDetermined
			if before := a.permissionsModel(); test.shown(before) {
				t.Fatal("the initial permission state already shows the change")
			}
			test.change(a)
			if after := a.permissionsModel(); !test.shown(after) {
				t.Fatal("a permission value change reused stale view state")
			}
		})
	}
}

func TestAudioSettingsModelCachePreservesPreviewChanges(t *testing.T) {
	a, _, _ := newTestApp(t)
	now := time.Date(2026, 1, 1, 9, 0, 0, 0, time.UTC)
	pair := func(state settingsAudioViewState, start string) settingsSoundPairViewState {
		t.Helper()
		for _, pair := range state.Sounds.Pairs {
			if pair.Start == start {
				return pair
			}
		}
		t.Fatalf("no sound pair starts with %q", start)
		return settingsSoundPairViewState{}
	}

	a.sounds.seq++
	a.sounds.previewing = recordStartEvent.ID
	a.sounds.playing = "pebble"
	a.sounds.playingEnd = now.Add(time.Second)
	first := a.audioSettingsModel(now)
	if got := pair(first, "pebble"); !got.Playing || got.Remaining != time.Second {
		t.Fatalf("first preview: playing=%t remaining=%v", got.Playing, got.Remaining)
	}
	later := a.audioSettingsModel(now.Add(250 * time.Millisecond))
	if got := pair(later, "pebble"); !got.Playing || got.Remaining != 750*time.Millisecond {
		t.Fatalf("later preview: playing=%t remaining=%v", got.Playing, got.Remaining)
	}
	if !later.Sounds.SystemCues[0].TryAccent {
		t.Fatal("the preview lost its Try accent")
	}

	a.sounds.seq++
	a.sounds.playing = "tally"
	a.sounds.playingEnd = now.Add(2 * time.Second)
	changed := a.audioSettingsModel(now.Add(500 * time.Millisecond))
	if got := pair(changed, "tally"); !got.Playing || got.Remaining != 1500*time.Millisecond {
		t.Fatalf("changed preview: playing=%t remaining=%v", got.Playing, got.Remaining)
	}
	if pair(changed, "pebble").Playing {
		t.Fatal("the previous preview source stayed active")
	}

	a.sysRead, a.sysSounds = false, nil
	expired := a.audioSettingsModel(now.Add(3 * time.Second))
	if pair(expired, "tally").Playing || !expired.Sounds.SystemCues[0].TryAccent {
		t.Fatal("preview expiry did not rebuild the playing state")
	}
	cached := a.audioSettingsModel(now.Add(3 * time.Second))
	if &expired.Sounds.SourceOptions[0] != &cached.Sounds.SourceOptions[0] {
		t.Fatal("an unchanged empty preview rebuilt the audio view state")
	}
	a.sounds.previewing = ""
	cleared := a.audioSettingsModel(now.Add(3 * time.Second))
	if cleared.Sounds.SystemCues[0].TryAccent {
		t.Fatal("clearing the preview accent reused stale audio view state")
	}

	a.updateSettings(func(current *settings.Settings) {
		current.Audio.StartSound, current.Audio.StopSound = "system:Pop", "system:Tink"
	})
	system := a.audioSettingsModel(now.Add(3 * time.Second))
	if system.Sounds.SourceID != soundSystemID {
		t.Fatalf("sound source %q, want %q", system.Sounds.SourceID, soundSystemID)
	}
	a.sysSounds, a.sysRead = []string{"Fixture Ping"}, true
	loaded := a.audioSettingsModel(now.Add(3 * time.Second))
	if !slices.Contains(loaded.Sounds.SystemCues[0].Options, "Fixture Ping") {
		t.Fatal("loading system sounds reused stale audio view state")
	}
}
