package main

import (
	"path/filepath"
	"time"

	"github.com/hehehai/actnowkit/registry"
	"github.com/hehehai/actnowkit/shortcuts"
	"github.com/hehehai/actnowkit/sounds"
	"github.com/sgl-project/sglang-omni/Voxt/internal/helper/proto"
	"github.com/sgl-project/sglang-omni/Voxt/internal/hotkey"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/local"
	"github.com/sgl-project/sglang-omni/Voxt/internal/permission"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
)

const (
	settingsFixtureSoundRemaining = 1250 * time.Millisecond
	settingsFixtureHome           = "testdata/fixture-home"
)

var (
	settingsFixtureNow       = meetingFixtureNow
	settingsFixtureLastCheck = settingsFixtureNow.Add(-2 * time.Hour)
)

type settingsFixtureInput struct {
	section        string
	preferences    settings.Settings
	permissions    permissionsStateInput
	audio          settingsAudioStateInput
	sounds         settingsSoundsStateInput
	shortcuts      settingsShortcutsStateInput
	privacy        settingsPrivacyStateInput
	historyAccess  historyAccessStateInput
	storage        settingsStorageStateInput
	about          settingsAboutStateInput
	updates        settingsUpdatesStateInput
	keptModelIDs   []string
	keptModelBytes int64
}

type shortcutPromptFixtureInput struct {
	features settings.Features
	conflict shortcuts.Conflict
}

func settingsFixtureScenarios() []meetingFixtureScenario {
	var scenarios []meetingFixtureScenario
	for _, section := range buildSettingsSections(i18n.NewTranslator("en")) {
		scenario := newSettingsFixtureScenario(settingsNormalFixtureName(section.id), section.id)
		switch section.id {
		case "general":
			scenario.settings.preferences.General.MainLanguage = "en"
			scenario.settings.preferences.General.OtherLanguages = []string{"ja", "fr"}
			scenario.settings.preferences.Features.Hidden = []string{settings.HideTranslate}
		case "overlay":
			scenario.settings.preferences.Overlay.LiveLines = false
		case "privacy":
			scenario.settings.privacy.Privacy.KeepAudio = true
			scenario.settings.privacy.Privacy.SendWindow = true
			scenario.settings.privacy.Privacy.SendClipboardImage = true
		}
		scenarios = append(scenarios, scenario)
	}

	scenarios = append(scenarios, newSettingsFixtureScenario("settings-general-empty-languages", "general"))
	for _, name := range []string{"settings-loading", "settings-empty", "settings-error", "settings-audio-no-helper"} {
		scenario := newSettingsFixtureScenario(name, "audio")
		scenario.settings.audio.Audio.Devices = nil
		scenario.settings.audio.Devices = nil
		switch name {
		case "settings-loading":
			scenario.settings.audio.DevicesRead = false
			scenario.settings.audio.DevicesLoading = true
		case "settings-error":
			scenario.settings.audio.DevicesRead = false
			scenario.settings.audio.DevicesError = "fixture device read failed"
		case "settings-audio-no-helper":
			scenario.settings.audio.HelperReady = false
			scenario.settings.audio.DevicesRead = false
		}
		scenarios = append(scenarios, scenario)
	}

	customPair := newSettingsFixtureScenario("settings-audio-source-built-in-custom-playing", "audio")
	customPair.settings.sounds.Prefs.Choose(recordStartEvent, "chime")
	customPair.settings.sounds.Prefs.Choose(recordStopEvent, "tink")
	customPair.settings.sounds.Playing = "chime"
	customPair.settings.sounds.Remaining = settingsFixtureSoundRemaining
	settingsFixtureSoundTargets(&customPair.settings.sounds)
	scenarios = append(scenarios, customPair)

	systemSounds := newSettingsFixtureScenario("settings-audio-source-system", "audio")
	systemSounds.settings.sounds.Source = soundSystem
	systemSounds.settings.sounds.Prefs.Choose(recordStartEvent, sounds.System("Fixture Pop"))
	systemSounds.settings.sounds.Prefs.Choose(recordStopEvent, sounds.System("Fixture Tink"))
	settingsFixtureSoundTargets(&systemSounds.settings.sounds)
	scenarios = append(scenarios, systemSounds)

	noSounds := newSettingsFixtureScenario("settings-audio-source-none", "audio")
	noSounds.settings.sounds.Source = soundNone
	noSounds.settings.sounds.Prefs.On = false
	settingsFixtureSoundTargets(&noSounds.settings.sounds)
	scenarios = append(scenarios, noSounds)

	for state := permission.Unknown; state <= permission.Granted; state++ {
		name := "settings-permissions-state-" + settingsPermissionStateName(state)
		scenario := newSettingsFixtureScenario(name, "permissions")
		scenario.settings.permissions = permissionsStateInput{
			Microphone: state, Accessibility: state == permission.Granted,
			SpeechRecognition: permission.Granted - state,
		}
		scenarios = append(scenarios, scenario)
	}

	appearance := newSettingsFixtureScenario("settings-appearance-dark-icons-reduce-motion", "appearance")
	appearance.settings.preferences.Appearance.Theme = "dark"
	appearance.settings.preferences.Appearance.Scale = settings.Scales[len(settings.Scales)-1]
	appearance.settings.preferences.Appearance.Rail = true
	appearance.settings.preferences.Appearance.RailChosen = true
	appearance.settings.preferences.Appearance.ReduceMotion = true
	scenarios = append(scenarios, appearance)
	lightAppearance := newSettingsFixtureScenario("settings-appearance-light", "appearance")
	lightAppearance.settings.preferences.Appearance.Theme = "light"
	scenarios = append(scenarios, lightAppearance)
	scale90 := newSettingsFixtureScenario("settings-appearance-scale-90", "appearance")
	scale90.settings.preferences.Appearance.Scale = 0.9
	scenarios = append(scenarios, scale90)
	scale110 := newSettingsFixtureScenario("settings-appearance-scale-110", "appearance")
	scale110.settings.preferences.Appearance.Scale = 1.1
	scenarios = append(scenarios, scale110)

	overlay := newSettingsFixtureScenario("settings-overlay-top-live-lines", "overlay")
	overlay.settings.preferences.Overlay.Position = "top"
	overlay.settings.preferences.Overlay.LiveLines = true
	scenarios = append(scenarios, overlay)

	shortcutProblems := newSettingsFixtureScenario("settings-shortcuts-duplicate-problem-strict-fn-no-accessibility", "shortcuts")
	shortcutProblems.settings.shortcuts.Features = settingsFixtureProblemFeatures()
	shortcutProblems.settings.shortcuts.AccessibilityInstalled = false
	scenarios = append(scenarios, shortcutProblems)

	familyConflict := newSettingsFixtureScenario("settings-shortcuts-family-conflict", "shortcuts")
	familyConflict.settings.shortcuts.FamilyConflicts = []shortcuts.Conflict{settingsFixtureConflict()}
	scenarios = append(scenarios, familyConflict)

	offlinePrivacy := newSettingsFixtureScenario("settings-privacy-offline-disabled", "privacy")
	offlinePrivacy.settings.privacy.Privacy = settings.Privacy{}
	offlinePrivacy.settings.privacy.HasUpdateFeed = false
	offlinePrivacy.settings.historyAccess = historyAccessStateInput{}
	scenarios = append(scenarios, offlinePrivacy)
	clearHistory := newSettingsFixtureScenario("settings-privacy-clear-confirmation", "privacy")
	clearHistory.settings.privacy.ClearHistoryAsk = true
	scenarios = append(scenarios, clearHistory)

	storageLoading := newSettingsFixtureScenario("settings-storage-loading", "storage")
	storageLoading.settings.storage.Report = nil
	scenarios = append(scenarios, storageLoading)
	moveError := newSettingsFixtureScenario("settings-storage-move-error", "storage")
	moveError.settings.storage.MoveError = "fixture move failed"
	moveError.settings.storage.FailedMove = "fixture move failed"
	scenarios = append(scenarios, moveError)
	dataMove := newSettingsFixtureScenario("settings-storage-data-move-progress", "storage")
	dataMove.settings.storage.Moving = true
	dataMove.settings.storage.MoveDone = 4096
	dataMove.settings.storage.MoveTotal = 8192
	dataMove.settings.storage.MoveTo = filepath.Join(settingsFixtureHome, "moved-data")
	dataMove.settings.storage.CanCancelMove = true
	scenarios = append(scenarios, dataMove)
	modelMove := newSettingsFixtureScenario("settings-storage-model-move-progress", "storage")
	modelMove.settings.storage.ModelMoving = true
	modelMove.settings.storage.ModelMoveTo = filepath.Join(settingsFixtureHome, "moved-models")
	modelMove.settings.storage.CanCancelModelMove = true
	scenarios = append(scenarios, modelMove)
	resetSettings := newSettingsFixtureScenario("settings-storage-reset-settings", "storage")
	resetSettings.settings.storage.ResetAsk = "settings"
	scenarios = append(scenarios, resetSettings)
	eraseEverything := newSettingsFixtureScenario("settings-storage-erase-confirmation", "storage")
	eraseEverything.settings.storage.ResetAsk = "everything"
	eraseEverything.settings.storage.ResetText = "DELETE"
	eraseEverything.settings.keptModelIDs = []string{"fixture-shared-model"}
	eraseEverything.settings.keptModelBytes = 4096
	scenarios = append(scenarios, eraseEverything)

	aboutLoading := newSettingsFixtureScenario("settings-about-loading", "about")
	aboutLoading.settings.about.SystemInfo = nil
	aboutLoading.settings.about.HelperVersion = "—"
	scenarios = append(scenarios, aboutLoading)
	selfChecking := newSettingsFixtureScenario("settings-about-self-checking", "about")
	selfChecking.settings.about.SelfChecking = true
	scenarios = append(scenarios, selfChecking)
	aboutResults := newSettingsFixtureScenario("settings-about-results", "about")
	aboutResults.settings.about.Checks = []selfCheck{
		{id: "fixture-library", title: "Fixture library", status: "pass", detail: "fixture check passed"},
		{id: "fixture-helper", title: "Fixture helper", status: "warn", detail: "fixture check warned"},
		{id: "fixture-data", title: "Fixture data", status: "fail", detail: "fixture check failed"},
	}
	scenarios = append(scenarios, aboutResults)

	updates := newSettingsFixtureScenario("settings-about-updates", "about")
	updates.settings.about.Version = "1.2.3"
	updates.settings.updates = settingsUpdatesStateInput{
		Present: true, Known: true, Version: "1.2.3",
		Prefs: updatePrefs{Enabled: true, AutoCheck: true, LastCheck: settingsFixtureLastCheck},
	}
	scenarios = append(scenarios, updates)
	checkOnly := newSettingsFixtureScenario("settings-about-updates-check-only", "about")
	checkOnly.settings.about.Version = "1.2.3"
	checkOnly.settings.updates = settingsUpdatesStateInput{
		Present: true, Known: true, BundleUpdated: true, Beta: true, Version: "1.2.3",
		Prefs: updatePrefs{Enabled: true, CheckOnly: true, AutoCheck: true, LastCheck: settingsFixtureLastCheck},
	}
	scenarios = append(scenarios, checkOnly)

	scenarios = append(scenarios, newMeetingSettingsFixtureScenario("meeting-settings-normal"))
	scenarios = append(scenarios, meetingFixtureScenario{
		name: "meeting-settings-no-models", kind: "meeting-settings",
		meetingSettings: settingsMeetingStateInput{Meeting: settings.Meeting{Lang: "en"}},
	})
	for _, mode := range []string{"subtitles", "recording"} {
		scenario := newMeetingSettingsFixtureScenario("meeting-settings-capture-" + mode)
		scenario.meetingSettings.Meeting.Mode = mode
		scenarios = append(scenarios, scenario)
	}
	scenarios = append(scenarios, meetingFixtureScenario{
		name: "shortcut-prompt-conflict", kind: "shortcut-prompt",
		shortcutPrompt: shortcutPromptFixtureInput{features: settings.Defaults().Features, conflict: settingsFixtureConflict()},
	})
	return scenarios
}

func settingsNormalFixtureName(sectionID string) string {
	if sectionID == "general" {
		return "settings-normal"
	}
	return "settings-" + sectionID + "-normal"
}

func newSettingsFixtureScenario(name, section string) meetingFixtureScenario {
	scenario := meetingFixtureScenario{name: name, kind: "settings"}
	scenario.settings.section = section
	scenario.settings.preferences = settings.Defaults()
	switch section {
	case "permissions":
		scenario.settings.permissions = permissionsStateInput{
			Microphone: permission.Granted, Accessibility: true, SpeechRecognition: permission.Denied,
		}
	case "audio":
		scenario.settings.audio, scenario.settings.sounds = settingsFixtureAudioInputs()
	case "shortcuts":
		scenario.settings.shortcuts = settingsShortcutsStateInput{
			Features: scenario.settings.preferences.Features, AccessibilityInstalled: true,
		}
	case "privacy":
		scenario.settings.privacy = settingsPrivacyStateInput{
			Privacy: scenario.settings.preferences.Privacy, HasUpdateFeed: true,
		}
		scenario.settings.historyAccess = historyAccessFixtureInput()
	case "storage":
		scenario.settings.storage = settingsFixtureStorageInput()
	case "about":
		scenario.settings.about = settingsFixtureAboutInput()
	}
	return scenario
}

func settingsFixtureAudioInputs() (settingsAudioStateInput, settingsSoundsStateInput) {
	catalog, err := sounds.Load(testSoundsFS(), "sounds")
	if err != nil {
		panic(err)
	}
	audio := settings.Defaults().Audio
	audio.Devices = []string{"fixture-mic-primary", "fixture-mic-secondary"}
	audio.DeviceNames = map[string]string{
		"fixture-mic-primary": "Fixture Primary Microphone", "fixture-mic-secondary": "Fixture Secondary Microphone",
	}
	audio.LiveResult = true
	audio.UploadMB = 12
	audio.MaxSeconds = 1800
	devices := []proto.CaptureDevice{
		{ID: "fixture-mic-primary", Name: "Fixture Primary Microphone", Default: true},
		{ID: "fixture-mic-secondary", Name: "Fixture Secondary Microphone"},
		{ID: "fixture-mic-other", Name: "Fixture Other Microphone"},
	}
	soundInput := settingsSoundsStateInput{
		Prefs: sounds.DefaultPrefs(), Source: soundBuiltIn, Catalog: catalog,
		SystemSounds: []string{"Fixture Pop", "Fixture Tink"},
	}
	settingsFixtureSoundTargets(&soundInput)
	return settingsAudioStateInput{
		Audio: audio, Devices: devices, HelperReady: true, DevicesRead: true,
	}, soundInput
}

func settingsFixtureSoundTargets(input *settingsSoundsStateInput) {
	input.StartTarget = input.Prefs.Target(input.Catalog, recordStartEvent)
	input.StopTarget = input.Prefs.Target(input.Catalog, recordStopEvent)
}

func settingsPermissionStateName(state permission.State) string {
	return []string{"unknown", "not-determined", "denied", "restricted", "granted"}[state]
}

func settingsFixtureProblemFeatures() settings.Features {
	features := settings.Defaults().Features
	settings.ApplyPreset(&features, settings.PresetFn)
	features.Transcribe.Hotkeys = append(features.Transcribe.Hotkeys, features.Transcribe.Hotkeys[0])
	features.Translate.Hotkeys = []settings.Hotkey{{Keys: "Bad+"}}
	features.Rewrite.Hotkeys = []settings.Hotkey{{Keys: "A"}}
	features.PasteLast = features.Transcribe.Hotkeys[0]
	features.Cancel = settings.Hotkey{Keys: "Command+C"}
	features.StrictShortcuts = true
	return features
}

func settingsFixtureConflict() shortcuts.Conflict {
	theirs := shortcuts.Shortcut{
		ID: "fixture-transcription", Title: "Fixture transcription", Keys: settings.DefaultTranscribe,
		Trigger: shortcuts.Tap, Enabled: true,
	}
	return shortcuts.Conflict{
		Mine: shortcuts.Shortcut{ID: hotkey.Transcribe, Keys: settings.DefaultTranscribe, Trigger: shortcuts.Tap, Enabled: true},
		Other: shortcuts.Holder{
			BundleID: "fixture.other-app", Name: "Fixture Other App",
			Links:     map[string]string{registry.LinkShortcuts: "fixture-other:///shortcuts"},
			Shortcuts: []shortcuts.Shortcut{theirs},
		},
		Theirs: theirs,
		Prompt: true,
	}
}

func settingsFixtureStorageInput() settingsStorageStateInput {
	dataDirectory := filepath.Join(settingsFixtureHome, "data")
	recordingsDirectory := filepath.Join(settingsFixtureHome, "recordings")
	modelDirectory := filepath.Join(settingsFixtureHome, "models")
	return settingsStorageStateInput{
		HomeDir: settingsFixtureHome, DataDirectory: dataDirectory, RecordingsDirectory: recordingsDirectory,
		PreviousDirectory: filepath.Join(settingsFixtureHome, "previous-data"), CanOpen: true,
		Report: &storageReport{
			cats: []storageCat{
				{id: "audio", title: "Fixture recordings", bytes: 2048, dir: recordingsDirectory},
				{id: "database", title: "Fixture database", bytes: 4096, dir: dataDirectory},
			},
			data: 8192, free: 65536, models: modelDirectory, modelsMine: 4096, modelsShared: 2048, shared: true,
		},
		ModelDirectory: modelDirectory, FamilyOthers: true, CanChangeModelFolder: true, ModelBackup: true,
	}
}

func settingsFixtureAboutInput() settingsAboutStateInput {
	return settingsAboutStateInput{
		Version: "fixture-version", HelperVersion: "fixture-helper-version",
		SystemInfo: &sysInfo{OS: "Fixture OS", Chip: "Fixture Chip", Memory: 16 << 30},
	}
}

func newMeetingSettingsFixtureScenario(name string) meetingFixtureScenario {
	readyASR := []local.Model{
		{ID: "fixture-speech-one", Name: "Fixture Speech One", Kind: local.ASR},
		{ID: "fixture-speech-two", Name: "Fixture Speech Two", Kind: local.ASR},
	}
	return meetingFixtureScenario{
		name: name, kind: "meeting-settings",
		meetingSettings: settingsMeetingStateInput{
			Meeting: settings.Meeting{
				Feature: settings.Feature{ASR: "fixture-speech-two", LLM: "fixture-language-two"},
				Lang:    "en", HideFromSharing: true,
			},
			ReadyASR: readyASR, ShownASR: readyASR,
			ReadyLLM: []local.Model{
				{ID: "fixture-language-one", Name: "Fixture Language One", Kind: local.LLM},
				{ID: "fixture-language-two", Name: "Fixture Language Two", Kind: local.LLM},
			},
		},
	}
}
