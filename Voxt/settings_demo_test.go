package main

import (
	"slices"
	"testing"

	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
)

type settingsDemoCase struct {
	name     string
	fixture  string
	prepare  func(*meetingDemoHost)
	scroll   string
	row      string
	slider   string
	steps    []historyDemoStep
	want     []string
	wantText string
	check    func(*testing.T, *meetingDemoHost)
}

func TestSettingsDemoActionControls(t *testing.T) {
	tests := settingsDemoGeneralCases()
	tests = append(tests, settingsDemoAudioCases()...)
	tests = append(tests, settingsDemoControlCases()...)
	tests = append(tests, settingsDemoKeyRecorderCases()...)
	tests = append(tests, settingsDemoStorageCases()...)
	tests = append(tests, settingsDemoUpdateCases()...)
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			host, err := loadMeetingDemoHost("testdata/meeting")
			if err != nil {
				t.Fatal(err)
			}
			selectMeetingDemoFixture(t, host, test.fixture)
			if test.prepare != nil {
				test.prepare(host)
			}
			tester := newMeetingDemoTester(t, host)
			if test.scroll != "" {
				scrollToText(tester, test.scroll)
				frames(tester)
			}
			if test.row != "" {
				button, buttonOK := tester.Find("Open")
				row, rowOK := tester.Find(test.row)
				if !buttonOK || !rowOK {
					t.Fatalf("cannot locate Open in row %q (texts %q)", test.row, tester.Texts())
				}
				tester.ClickAt(button.X+button.W/2, row.Y+row.H/2)
				frames(tester)
			}
			if test.slider != "" {
				value, ok := tester.Find(test.slider)
				if !ok {
					t.Fatalf("cannot locate slider value %q", test.slider)
				}
				tester.ClickAt(value.X-120, value.Y+value.H/2)
				frames(tester)
			}
			for _, step := range test.steps {
				step.label = platformText(step.label)
				var err error
				switch {
				case step.context:
					err = tester.RightClick(step.label)
				case step.menu:
					err = tester.ChooseMenuItem(step.label)
				case step.text != "":
					err = tester.Click(step.label)
					if err == nil {
						tester.Type(step.text)
					}
				case step.command != "":
					tester.Command(step.command)
				case step.key != 0:
					tester.Key(step.mods, step.key)
				case step.click:
					err = tester.Click(step.label)
				default:
					err = tester.ClickControl(step.label)
				}
				if err != nil {
					t.Fatalf("%s: %v (controls %q)", step.label, err, tester.AccessLabels())
				}
				frames(tester)
			}
			if !slices.Equal(host.calls, test.want) {
				t.Fatalf("calls %q, want %q", host.calls, test.want)
			}
			if test.wantText != "" && !tester.HasText(test.wantText) {
				t.Fatalf("view does not show %q; texts %q", test.wantText, tester.Texts())
			}
			if test.check != nil {
				test.check(t, host)
			}
		})
	}
}

func settingsDemoSearch(query string) func(*meetingDemoHost) {
	return func(host *meetingDemoHost) { host.settingsForm.search = query }
}

func settingsDemoSelect(label string, keys ...ui.Key) []historyDemoStep {
	steps := []historyDemoStep{{label: label}}
	for _, key := range keys {
		steps = append(steps, historyDemoStep{key: key})
	}
	return append(steps, historyDemoStep{key: ui.KeyEnter})
}

func settingsDemoGeneralCases() []settingsDemoCase {
	return []settingsDemoCase{
		{name: "sections", fixture: "en/settings-normal", steps: historySteps("General", "Permissions", "Audio", "Overlay style", "Shortcuts", "Appearance", "Privacy", "Storage", "About"), want: []string{
			"navigate /settings/general", "navigate /settings/permissions", "navigate /settings/audio", "navigate /settings/overlay",
			"navigate /settings/shortcuts", "navigate /settings/appearance", "navigate /settings/privacy", "navigate /settings/storage", "navigate /settings/about",
		}},
		{name: "close", fixture: "en/settings-normal", steps: historySteps("Close"), check: func(t *testing.T, host *meetingDemoHost) {
			if host.settingsOpen {
				t.Fatal("settings dialog stayed open")
			}
		}},
		{name: "app language", fixture: "en/settings-normal", steps: settingsDemoSelect("App language", ui.KeyDown), want: []string{"setAppLanguage en"}},
		{name: "main language", fixture: "en/settings-normal", steps: settingsDemoSelect("Main language", ui.KeyDown), want: []string{"setMainLanguage ja"}},
		{name: "add language", fixture: "en/settings-normal", steps: []historyDemoStep{{label: "Add a language"}, {label: "Korean · 한국어", menu: true}}, want: []string{"addOtherLanguage ko"}},
		{name: "remove language", fixture: "en/settings-normal", steps: historySteps("Remove Japanese · 日本語"), want: []string{"removeOtherLanguage ja"}},
		{name: "copy result", fixture: "en/settings-normal", steps: historySteps("Also copy the result to the clipboard"), want: []string{"setCopyResult true"}},
		{name: "menu bar", fixture: "en/settings-normal", steps: historySteps("Show in the menu bar"), want: []string{"setMenuBar false"}},
		{name: "dock", fixture: "en/settings-normal", steps: historySteps("Show in the Dock"), want: []string{"setDock false"}},
		{name: "close to tray", fixture: "en/settings-normal", steps: historySteps("Keep running when the window closes"), want: []string{"setCloseToTray false"}},
		{name: "open at login", fixture: "en/settings-normal", steps: historySteps("Open at login"), want: []string{"setOpenAtLogin true"}},
		{name: "show translate", fixture: "en/settings-normal", scroll: "Show Translate", steps: historySteps("Show Translate"), want: []string{"setFeatureShown translate true"}},
		{name: "hide rewrite", fixture: "en/settings-normal", scroll: "Show Rewrite", steps: historySteps("Show Rewrite"), want: []string{"setFeatureShown rewrite false"}},
		{name: "hide app enhancement", fixture: "en/settings-normal", scroll: "Show App enhancement", steps: historySteps("Show App enhancement"), want: []string{"setFeatureShown app-enhancement false"}},
		{name: "setup guide", fixture: "en/settings-normal", scroll: "Open setup guide", steps: historySteps("Open setup guide"), want: []string{"openOnboarding"}},
		{name: "theme", fixture: "en/settings-appearance-normal", steps: historySteps("Dark"), want: []string{"setTheme dark"}},
		{name: "interface scale", fixture: "en/settings-appearance-normal", steps: historySteps("125%"), want: []string{"setInterfaceScale 1.25"}},
		{name: "sidebar", fixture: "en/settings-appearance-normal", steps: historySteps("Icons only"), want: []string{"setSidebarChoice icons"}},
		{name: "reduce motion", fixture: "en/settings-appearance-normal", steps: historySteps("Reduce motion"), want: []string{"setReduceMotion true"}},
		{name: "required permissions", fixture: "en/settings-permissions-state-not-determined", steps: historySteps("Ask for the required permissions"), want: []string{"askRequired"}},
		{name: "recheck permissions", fixture: "en/settings-permissions-normal", steps: historySteps("Check again"), want: []string{"recheckPermissions"}},
		{name: "ask microphone", fixture: "en/settings-permissions-state-not-determined", prepare: settingsDemoSearch("Microphone"), steps: historySteps("Ask"), want: []string{"requestPermission microphone"}},
		{name: "ask accessibility", fixture: "en/settings-permissions-state-not-determined", prepare: settingsDemoSearch("Lets Voxt type"), steps: historySteps("Ask"), want: []string{"requestPermission accessibility"}},
		{name: "ask system audio", fixture: "en/settings-permissions-state-not-determined", prepare: settingsDemoSearch("System audio recording"), steps: historySteps("Ask"), want: []string{"requestPermission systemAudio"}},
		{name: "ask speech recognition", fixture: "en/settings-permissions-state-restricted", prepare: settingsDemoSearch("Speech Recognition"), steps: historySteps("Ask"), want: []string{"requestPermission speechRecognition"}},
		{name: "open microphone permission", fixture: "en/settings-permissions-state-not-determined", prepare: settingsDemoSearch("Microphone"), steps: historySteps("Open System Settings"), want: []string{"openPrivacy " + privacyMicrophone}},
		{name: "open accessibility permission", fixture: "en/settings-permissions-state-not-determined", prepare: settingsDemoSearch("Lets Voxt type"), steps: historySteps("Open System Settings"), want: []string{"openPrivacy " + privacyAccessibility}},
		{name: "open system audio permission", fixture: "en/settings-permissions-state-not-determined", prepare: settingsDemoSearch("System audio recording"), steps: historySteps("Open System Settings"), want: []string{"openPrivacy " + privacyAudioCapture}},
		{name: "open speech permission", fixture: "en/settings-permissions-state-restricted", prepare: settingsDemoSearch("Speech Recognition"), steps: historySteps("Open System Settings"), want: []string{"openPrivacy " + privacySpeech}},
	}
}

func settingsDemoExpectedText(dialog *settingsDialogViewState) string {
	switch {
	case dialog.Audio != nil:
		audio := dialog.Audio
		if audio.DevicesError != "" {
			return audio.DevicesError
		}
		if !audio.HelperReady {
			return i18n.T("The helper is not running, so microphones cannot be listed.")
		}
		for _, pair := range audio.Sounds.Pairs {
			if pair.Playing {
				return pair.PlayLabel
			}
		}
		if audio.Sounds.SourceID == soundSystemID {
			return audio.Sounds.SystemCues[0].Selected
		}
		if audio.DevicesLoading || audio.Sounds.SourceID == soundNoneID {
			return i18n.T("Microphones")
		}
		if len(audio.Devices) > 0 {
			return audio.Devices[0].Name
		}
		if audio.DevicesRead {
			return i18n.T("No other microphone is connected.")
		}
		return i18n.T("No microphone chosen, so Voxt records from the system's default input.")
	case dialog.General != nil:
		if len(dialog.General.OtherLanguages) == 0 {
			return i18n.T("Other languages")
		}
		return dialog.General.OtherLanguages[0].Label
	case dialog.Permissions != nil:
		return dialog.Permissions.Microphone.Status
	case dialog.Appearance != nil:
		if dialog.Appearance.ReduceMotion {
			return i18n.T("Reduce motion")
		}
		return i18n.T("Appearance")
	case dialog.Overlay != nil:
		if dialog.Overlay.Position == 1 && dialog.Overlay.LiveLayout == 1 {
			return i18n.T("Live text layout")
		}
		return i18n.T("Preview")
	case dialog.Shortcuts != nil:
		if !dialog.Shortcuts.AccessibilityInstalled {
			return i18n.T("Open Permissions")
		}
		for _, row := range dialog.Shortcuts.Rows {
			if len(row.Problems) > 0 {
				return row.Problems[0]
			}
		}
		return dialog.Shortcuts.Rows[0].KeyText
	case dialog.Privacy != nil:
		privacy := dialog.Privacy
		if privacy.ClearHistoryAsk {
			return i18n.T("Clear history?")
		}
		if privacy.HistoryAccess.ShowLastRead {
			return privacy.HistoryAccess.LastRead
		}
		return privacy.Online
	case dialog.Storage != nil:
		storage := dialog.Storage
		if storage.ResetAsk == "settings" {
			return i18n.T("Reset all settings?")
		}
		if storage.ResetAsk == "everything" {
			return i18n.T("Erase everything?")
		}
		if storage.Moving {
			return storage.MoveProgressText
		}
		if storage.ModelMoving {
			return storage.ModelMoveText
		}
		if storage.MoveError != "" {
			return storage.MoveError
		}
		if storage.Report == nil {
			return i18n.T("Measuring…")
		}
		return storage.Report.DataSize
	case dialog.About != nil:
		about := dialog.About
		if about.SelfChecking {
			return i18n.T("Self-check")
		}
		if len(about.Checks) > 0 {
			return about.Checks[0].Detail
		}
		if len(about.Rows) < 4 {
			return about.Rows[1].Value
		}
		return about.Rows[0].Value
	}
	return i18n.T("Settings")
}

func keyRecorderDemoExpectedText(recorder *keyRecorderViewState) string {
	switch {
	case recorder.RecordDenied:
		return i18n.T("Recording a shortcut needs Accessibility access.")
	case recorder.Unknown:
		return i18n.T("Voxt cannot use that key. Press another, or type the keys instead.")
	case recorder.Typing && !recorder.Typed.Valid:
		return recorder.Typed.ErrorText
	case recorder.Typing && recorder.Edited:
		return i18n.T("Press Return to use %s.", recorder.Typed.FormattedKeys)
	case recorder.Typing:
		return i18n.T("Modifiers first, then one key: Fn, RightCommand, Control+Option+Space, Fn+F5.")
	case recorder.Focused && recorder.HelperRecordsKeys:
		return i18n.T("Press the shortcut you want, including fn, a left or right modifier, or a mouse button. Each press replaces the last.")
	case recorder.Focused:
		return i18n.T("Press the keys you want, such as ⌥Space. Each press replaces the last.")
	default:
		return recorder.DisplayedKeys
	}
}
