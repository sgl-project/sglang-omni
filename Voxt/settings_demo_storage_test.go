package main

import (
	"testing"

	"github.com/egoist/mygo/ui"
)

func settingsDemoStorageCases() []settingsDemoCase {
	return []settingsDemoCase{
		{name: "open data folder", fixture: "en/settings-storage-normal", prepare: settingsDemoSearch("~/data"), steps: historySteps("Open"), want: []string{"openStorageFolder /fixture/home/data"}},
		{name: "open model folder", fixture: "en/settings-storage-normal", prepare: settingsDemoSearch("~/models"), steps: historySteps("Open"), want: []string{"openStorageFolder /fixture/home/models"}},
		{name: "open recordings category", fixture: "en/settings-storage-normal", prepare: settingsDemoSearch("Reset look"), row: "Fixture recordings", want: []string{"openStorageFolder /fixture/home/recordings"}},
		{name: "open database category", fixture: "en/settings-storage-normal", prepare: settingsDemoSearch("Reset look"), row: "Fixture database", want: []string{"openStorageFolder /fixture/home/data"}},
		{name: "choose data folder", fixture: "en/settings-storage-normal", steps: historySteps("Move…"), want: []string{"chooseStorageDataFolder"}},
		{name: "choose model folder", fixture: "en/settings-storage-normal", steps: historySteps("Change…"), want: []string{"chooseStorageModelFolder"}},
		{name: "storage history settings", fixture: "en/settings-storage-normal", steps: historySteps("History settings…"), want: []string{"showHistorySettings"}},
		{name: "remove previous data", fixture: "en/settings-storage-normal", steps: historySteps("Remove it"), want: []string{"removePrevious"}},
		{name: "dismiss move failure", fixture: "en/settings-storage-move-error", steps: historySteps("Dismiss"), want: []string{"dismissMoveFailure"}},
		{name: "cancel data move", fixture: "en/settings-storage-data-move-progress", steps: historySteps("Cancel"), want: []string{"cancelStorageDataMove"}},
		{name: "cancel model move", fixture: "en/settings-storage-model-move-progress", steps: historySteps("Cancel"), want: []string{"cancelStorageModelMove"}},
		{name: "model backup", fixture: "en/settings-storage-normal", scroll: "Back up models with Time Machine", steps: historySteps("Back up models with Time Machine"), want: []string{"setStorageModelBackup false"}},
		{name: "reset look", fixture: "en/settings-storage-normal", scroll: "Reset the look", steps: historySteps("Reset"), want: []string{"resetLook"}},
		{name: "reset settings", fixture: "en/settings-storage-normal", scroll: "Reset all settings", steps: historySteps("Reset…", "Reset settings"), want: []string{"resetSettings"}, check: func(t *testing.T, host *meetingDemoHost) {
			if host.settingsStorageUI.resetAsk != "" {
				t.Fatal("reset confirmation stayed open")
			}
		}},
		{name: "reset settings cancel", fixture: "en/settings-storage-normal", scroll: "Reset all settings", steps: historySteps("Reset…", "Cancel"), check: func(t *testing.T, host *meetingDemoHost) {
			if host.settingsStorageUI.resetAsk != "" {
				t.Fatal("reset confirmation stayed open")
			}
		}},
		{name: "erase everything", fixture: "en/settings-storage-erase-confirmation", steps: historySteps("Erase and restart"), want: []string{"eraseAll"}, check: func(t *testing.T, host *meetingDemoHost) {
			if host.settingsStorageUI.resetAsk != "" || host.settingsStorageUI.resetText != "" {
				t.Fatal("erase completion kept confirmation state")
			}
		}},
		{name: "ask erase", fixture: "en/settings-storage-normal", scroll: "Erase everything", steps: historySteps("Erase…"), want: []string{"askErase"}, check: func(t *testing.T, host *meetingDemoHost) {
			if host.settingsStorageUI.resetAsk != "everything" || host.settingsStorageUI.resetText != "" {
				t.Fatal("erase confirmation did not open with empty text")
			}
		}},
		{name: "erase cancel", fixture: "en/settings-storage-normal", scroll: "Erase everything", steps: historySteps("Erase…", "Cancel"), want: []string{"askErase"}, check: func(t *testing.T, host *meetingDemoHost) {
			if host.settingsStorageUI.resetAsk != "" {
				t.Fatal("erase confirmation stayed open")
			}
		}},
		{name: "self check", fixture: "en/settings-about-normal", steps: historySteps("Run check"), want: []string{"runSelfCheck"}},
		{name: "copy diagnostics", fixture: "en/settings-about-normal", steps: historySteps("Copy"), want: []string{"copyDiagnostics"}},
		{name: "save diagnostics", fixture: "en/settings-about-normal", steps: historySteps("Save…"), want: []string{"saveBundle"}},
		{name: "report bug", fixture: "en/settings-about-normal", steps: historySteps("Report…"), want: []string{"reportBug"}},
		{name: "meeting capture mode", fixture: "en/meeting-settings-normal", steps: settingsDemoSelect("Capture mode", ui.KeyDown, ui.KeyDown), want: []string{"setMeetingCaptureMode recording"}},
		{name: "meeting hide from sharing", fixture: "en/meeting-settings-normal", steps: historySteps("Hide meeting overlay"), want: []string{"setMeetingHideFromSharing false"}},
		{name: "meeting speech model", fixture: "en/meeting-settings-normal", steps: settingsDemoSelect("Speech model for meetings", ui.KeyUp), want: []string{"setMeetingASR fixture-speech-one"}},
		{name: "meeting language", fixture: "en/meeting-settings-normal", steps: settingsDemoSelect("Spoken language in meetings", ui.KeyDown), want: []string{"setMeetingLanguage ja"}},
		{name: "meeting language model", fixture: "en/meeting-settings-normal", steps: settingsDemoSelect("Language model", ui.KeyUp), want: []string{"setMeetingLLM fixture-language-one"}},
		{name: "meeting settings done", fixture: "en/meeting-settings-normal", steps: historySteps("Done"), check: func(t *testing.T, host *meetingDemoHost) {
			if host.pageUI.setOpen {
				t.Fatal("meeting settings stayed open")
			}
		}},
		{name: "meeting settings close", fixture: "en/meeting-settings-normal", steps: historySteps("Close"), check: func(t *testing.T, host *meetingDemoHost) {
			if host.pageUI.setOpen {
				t.Fatal("meeting settings stayed open")
			}
		}},
		{name: "prompt ignore", fixture: "en/shortcut-prompt-conflict", steps: historySteps("Ignore"), want: []string{"answerShortcutPrompt"}},
		{name: "prompt escape", fixture: "en/shortcut-prompt-conflict", steps: []historyDemoStep{{key: ui.KeyEscape}}, want: []string{"answerShortcutPrompt"}},
		{name: "prompt other settings", fixture: "en/shortcut-prompt-conflict", steps: historySteps("Open Fixture Other App settings"), want: []string{"openShortcutPromptLink fixture-other:///shortcuts"}},
		{name: "prompt change", fixture: "en/shortcut-prompt-conflict", steps: historySteps("Change shortcut"), want: []string{"changeShortcutPrompt /transcribe"}},
	}
}
