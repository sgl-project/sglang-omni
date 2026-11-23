package main

import (
	"context"
	"os"
	"path/filepath"
	"testing"

	"github.com/egoist/mygo"
	"github.com/sgl-project/sglang-omni/Voxt/internal/helper"
	"github.com/sgl-project/sglang-omni/Voxt/internal/helper/proto"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/local"
	"github.com/sgl-project/sglang-omni/Voxt/internal/models"
)

func newSettingsFixtureApp(t *testing.T) *app {
	t.Helper()
	if err := os.RemoveAll(settingsFixtureHome); err != nil {
		t.Fatal(err)
	}
	dataDirectory := filepath.Join(settingsFixtureHome, "data")
	if err := os.MkdirAll(dataDirectory, 0o755); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() {
		if err := os.RemoveAll(settingsFixtureHome); err != nil {
			t.Errorf("remove fixture home: %v", err)
		}
	})
	locale := i18n.Locale()
	a, _, _ := newTestAppIn(t, dataDirectory)
	i18n.SetLocale(locale)
	a.upd = nil
	return a
}

func settingsFixtureApplyInput(t *testing.T, a *app, input settingsFixtureInput) {
	t.Helper()
	a.prefs = input.preferences
	a.settingsRevision++
	a.settingsOpen = true
	a.setRouter.Replace("/" + input.section)
	if input.privacy.HasUpdateFeed || input.updates.Available {
		a.upd = newFakeUpdates()
	}

	switch input.section {
	case "permissions":
		a.perms.mic = input.permissions.Microphone
		a.perms.access = input.permissions.Accessibility
		a.perms.speech = input.permissions.SpeechRecognition
		a.perms.systemAudioAsked = input.permissions.SystemAudioAsked
	case "audio":
		a.prefs.Audio = input.audio.Audio
		a.prefs.Audio.Sounds = input.sounds.Prefs.On
		a.prefs.Audio.SetSoundVolume(input.sounds.Prefs.Volume)
		a.prefs.Audio.StartSound = input.sounds.Prefs.Choice(recordStartEvent)
		a.prefs.Audio.StopSound = input.sounds.Prefs.Choice(recordStopEvent)
		a.audio.devices = input.audio.Devices
		a.audio.read = input.audio.DevicesRead
		a.audio.loading = input.audio.DevicesLoading
		a.audio.err = input.audio.DevicesError
		a.hstatus.State = helper.Stopped
		if input.audio.HelperReady {
			a.hstatus.State = helper.Ready
		}
		a.soundCat = input.sounds.Catalog
		a.sysSounds = input.sounds.SystemSounds
		a.sysRead = true
		a.sounds.previewing = input.sounds.Previewing
		a.sounds.playing = input.sounds.Playing
		a.sounds.playingEnd = meetingFixtureNow.Add(input.sounds.Remaining)
	case "shortcuts":
		a.prefs.Features = input.shortcuts.Features
		a.famConflicts = input.shortcuts.FamilyConflicts
		a.hotkeys.Installed = input.shortcuts.AccessibilityInstalled
	case "privacy":
		a.prefs.Privacy = input.privacy.Privacy
		a.prefs.Privacy.HistoryAccess = input.historyAccess.On
		a.prefs.Privacy.HistorySelection = input.historyAccess.Selection
		a.clearHistoryAsk = input.privacy.ClearHistoryAsk
		a.histAccessUI = historyAccessLocalUI{
			historyAccess: input.historyAccess.On, historySelection: input.historyAccess.Selection,
			ask: input.historyAccess.Ask, listOpen: input.historyAccess.ListOpen,
		}
		a.histAccess.exe = input.historyAccess.Exe
		a.histAccess.reads = input.historyAccess.Reads
	case "storage":
		storage := input.storage
		a.homeDir = storage.HomeDir
		a.prefs.History.AudioDir = storage.RecordingsDirectory
		a.stReport = storage.Report
		a.stLoc.Previous = storage.PreviousDirectory
		a.stLoc.Failed = storage.FailedMove
		a.moveErr = storage.MoveError
		a.moving = storage.Moving
		a.moveDone = storage.MoveDone
		a.moveTotal = storage.MoveTotal
		a.moveTo = storage.MoveTo
		if storage.CanCancelMove {
			a.moveCancel = func() {}
		}
		a.resetAsk = storage.ResetAsk
		a.resetText = storage.ResetText
		a.erasing = storage.Erasing
		a.eraseKept = input.keptModelIDs
		a.eraseKeptBytes = input.keptModelBytes
		modelStore := *a.models.Store()
		modelStore.Hub = storage.ModelDirectory
		a.models.Close()
		a.models = models.Open(models.Options{Store: &modelStore})
		if storage.FamilyOthers {
			addPeerIn(t, a, "dev.hehehai.audiosmith", filepath.Join(settingsFixtureHome, "peer"), nil)
			a.familyFound(a.fam.Discover())
		}
		if err := a.fam.SetBackup(context.Background(), storage.ModelBackup); err != nil {
			t.Fatal(err)
		}
		a.modelMoving = storage.ModelMoving
		a.modelMoveTo = storage.ModelMoveTo
		if storage.CanCancelModelMove {
			a.modelMoveCancel = func() {}
		}
	case "about":
		if input.updates.Available && input.updates.RunningVersion != input.about.Version {
			t.Fatalf("about version %q and updates version %q differ", input.about.Version, input.updates.RunningVersion)
		}
		a.version = input.about.Version
		a.hstatus.Hello = nil
		if input.about.HelperVersion != "—" {
			a.hstatus.Hello = &proto.HelloResult{Version: input.about.HelperVersion}
		}
		a.sys = input.about.SystemInfo
		a.selfChecking = input.about.SelfChecking
		a.checks = input.about.Checks
		a.updKnown = input.updates.Known
		a.updPrefs = input.updates.Prefs
		a.bundleUpdated = input.updates.BundleUpdated
		beta := input.updates.Beta
		a.prefs.Updates.Beta = &beta
	}
}

func buildSettingsFixture(t *testing.T, input settingsFixtureInput) settingsDialogViewState {
	t.Helper()
	a := newSettingsFixtureApp(t)
	settingsFixtureApplyInput(t, a, input)
	previousWindow := a.win
	if input.section == "storage" && input.storage.CanOpen {
		a.win = new(mygo.Window)
	}
	state := a.settingsDialogModel(meetingFixtureNow)
	a.win = previousWindow
	return state
}

type settingsFixtureLocal struct {
	fakeLocal
	installedModels []local.Model
}

func (f *settingsFixtureLocal) Installed(kind local.Kind) []local.Model {
	var installed []local.Model
	for _, model := range f.installedModels {
		if model.Kind == kind {
			installed = append(installed, model)
		}
	}
	return installed
}

func buildMeetingSettingsFixture(t *testing.T, input settingsMeetingStateInput) settingsMeetingViewState {
	t.Helper()
	a := newSettingsFixtureApp(t)
	a.prefs.Features.Meeting = input.Meeting
	installedModels := append([]local.Model{}, input.ReadyASR...)
	installedModels = append(installedModels, input.ReadyLLM...)
	a.local = &settingsFixtureLocal{installedModels: installedModels}
	return a.meetingSettingsModel()
}

func buildShortcutPromptFixture(t *testing.T, input shortcutPromptFixtureInput) shortcutPromptViewState {
	t.Helper()
	a := newSettingsFixtureApp(t)
	a.prefs.Features = input.features
	a.scPrompt = &input.conflict
	state := a.shortcutPromptModel()
	if state == nil {
		t.Fatal("shortcut prompt fixture did not exercise the non-nil model branch")
	}
	return *state
}
