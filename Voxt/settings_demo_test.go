package main

import (
	"slices"
	"testing"

	"github.com/egoist/mygo/ui"
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
