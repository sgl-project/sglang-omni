package main

import (
	"slices"
	"testing"

	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
)

func TestHomeDemoActionControls(t *testing.T) {
	for _, test := range []struct {
		name    string
		fixture string
		steps   []historyDemoStep
		want    []string
		check   func(*testing.T, *meetingDemoHost)
	}{
		{name: "guide", fixture: "en/home-empty", steps: historySteps("Open setup guide"), want: []string{"openHomeGuide"}},
		{name: "history", fixture: "en/home-normal", steps: historySteps("Open History"), want: []string{"openHomeHistory"}},
		{name: "add shortcut", fixture: "en/home-shortcut-missing", steps: historySteps("Add a shortcut to start"), want: []string{"addHomeShortcut"}},
		{name: "speech download", fixture: "en/home-speech-missing", steps: historySteps("Speech recognition: Download (2.5 GB)"), want: []string{"fixHomeReadiness download-speech"}},
		{name: "speech on disk", fixture: "en/home-speech-on-disk", steps: historySteps("Speech recognition: Open Speech models"), want: []string{"fixHomeReadiness speech-models"}},
		{name: "speech downloading", fixture: "en/home-speech-downloading", steps: historySteps("Speech recognition: Open Speech models"), want: []string{"fixHomeReadiness speech-models"}},
		{name: "language models", fixture: "en/home-llm-missing", steps: historySteps("Language model: Open Language models"), want: []string{"fixHomeReadiness language-models"}},
		{name: "microphone ask", fixture: "en/home-microphone-notdetermined", steps: historySteps("Microphone: Ask now"), want: []string{"fixHomeReadiness microphone"}},
		{name: "microphone permissions", fixture: "en/home-microphone-denied", steps: historySteps("Microphone: Open Permissions"), want: []string{"fixHomeReadiness permissions"}},
		{name: "accessibility permissions", fixture: "en/home-access-missing", steps: historySteps("Accessibility: Open Permissions"), want: []string{"fixHomeReadiness permissions"}},
		{name: "shortcut settings", fixture: "en/home-shortcut-missing", steps: historySteps("Shortcut: Open Transcribe"), want: []string{"fixHomeReadiness transcribe"}},
		{name: "helper restart", fixture: "en/home-helper-stopped", steps: historySteps("Helper: Restart helper"), want: []string{"fixHomeReadiness restart-helper"}},
		{name: "app relaunch", fixture: "en/home-helper-reason-updated", steps: historySteps("Helper: Restart Voxt"), want: []string{"fixHomeReadiness relaunch"}},
		{name: "open recent", fixture: "en/home-normal", steps: historySteps("Weekly plan"), want: []string{"openHomeRecent home-transcribe"}},
		{name: "inline copy", fixture: "en/home-normal", steps: []historyDemoStep{{label: "Copy: Weekly plan", click: true}}, want: []string{"copyHomeRecent home-transcribe"}},
		{name: "menu copy", fixture: "en/home-normal", steps: []historyDemoStep{{label: "Weekly plan", context: true}, {label: "Copy", menu: true}}, want: []string{`copyHistoryText "Weekly plan\nI'll send the new agenda tonight."`}},
		{name: "menu raw", fixture: "en/home-normal", steps: []historyDemoStep{{label: "Weekly plan", context: true}, {label: "Copy recognised text", menu: true}}, want: []string{`copyHistoryText "weekly plan\nI'll send the new agenda tonight"`}},
		{name: "menu details", fixture: "en/home-normal", steps: []historyDemoStep{{label: "Weekly plan", context: true}, {label: "Show details", menu: true}}, want: []string{"openHomeRecent home-transcribe"}},
		{name: "menu reveal", fixture: "en/home-normal", steps: []historyDemoStep{{label: "Weekly plan", context: true}, {label: "Show recording in Finder", menu: true}}, want: []string{"showHistoryRecording home-transcribe"}},
		{name: "menu retry", fixture: "en/home-normal", steps: []historyDemoStep{{label: "Weekly plan", context: true}, {label: "Transcribe again", menu: true}}, want: []string{"retryHistoryEntry home-transcribe"}},
		{name: "menu delete", fixture: "en/home-normal", steps: []historyDemoStep{{label: "Weekly plan", context: true}, {label: "Delete", menu: true}, {label: "Delete"}}, want: []string{"deleteHistoryEntry home-transcribe"}},
		{name: "delete cancel", fixture: "en/home-normal", steps: []historyDemoStep{{label: "Weekly plan", context: true}, {label: "Delete", menu: true}, {label: "Cancel"}}, check: func(t *testing.T, host *meetingDemoHost) {
			if host.historyUI.delAsk {
				t.Fatal("delete confirmation stayed open")
			}
		}},
	} {
		t.Run(test.name, func(t *testing.T) {
			host, err := loadMeetingDemoHost("testdata/meeting")
			if err != nil {
				t.Fatal(err)
			}
			selectMeetingDemoFixture(t, host, test.fixture)
			tester := newMeetingDemoTester(t, host)
			for _, step := range test.steps {
				var err error
				switch {
				case step.context:
					err = tester.RightClick(step.label)
				case step.menu:
					err = tester.ChooseMenuItem(step.label)
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
			if test.check != nil {
				test.check(t, host)
			}
		})
	}
}

func TestHomeErrorInitialScenarioShowsTry(t *testing.T) {
	for _, scenario := range homeFixtureScenarios() {
		if scenario.name != "home-error-initial" {
			continue
		}
		fixture := buildMeetingFixture(t, scenario, "en")
		if !fixture.Home.ShowTry {
			t.Fatal("home-error-initial does not offer the initial Try area")
		}
		return
	}
	t.Fatal("home-error-initial scenario is missing")
}

func TestHomeDemoFixtureTry(t *testing.T) {
	host, err := loadMeetingDemoHost("testdata/meeting")
	if err != nil {
		t.Fatal(err)
	}
	for index, fixture := range host.fixtures {
		if fixture.state.Home == nil {
			continue
		}
		t.Run(fixture.name, func(t *testing.T) {
			host.selectMeetingFixture(index - host.selected)
			tester := newMeetingDemoTester(t, host)
			tryTitle, startLabel := i18n.T("Try it"), i18n.T("Start a test")
			if !fixture.state.Home.ShowTry {
				if tester.HasText(tryTitle) {
					t.Fatal("Try shows with show_try=false")
				}
				return
			}
			if !tester.HasText(tryTitle) || !tester.HasText(startLabel) {
				t.Fatalf("no fixture-backed Try area: %q", tester.Texts())
			}
			host.calls = nil
			if err := tester.ClickControl(startLabel); err != nil {
				t.Fatal(err)
			}
			frames(tester)
			if !slices.Equal(host.calls, []string{"startTry"}) {
				t.Fatalf("calls %q, want [startTry]", host.calls)
			}
		})
	}
}
