package main

import (
	"slices"
	"testing"
)

func TestHomeDemoActionControls(t *testing.T) {
	for _, test := range []struct {
		name    string
		fixture string
		steps   []historyDemoStep
		want    []string
	}{
		{name: "guide", fixture: "en/home-empty", steps: historySteps("Open setup guide"), want: []string{"openHomeGuide"}},
		{name: "add shortcut", fixture: "en/home-shortcut-missing", steps: historySteps("Add a shortcut to start"), want: []string{"addHomeShortcut"}},
		{name: "activity pages", fixture: "en/home-normal", steps: historySteps("Earlier weeks"), want: []string{"activityBack"}},
	} {
		t.Run(test.name, func(t *testing.T) {
			host, err := loadMeetingDemoHost("testdata/meeting")
			if err != nil {
				t.Fatal(err)
			}
			selectMeetingDemoFixture(t, host, test.fixture)
			tester := newMeetingDemoTester(t, host)
			for _, step := range test.steps {
				if err := tester.ClickControl(step.label); err != nil {
					t.Fatalf("%s: %v (controls %q)", step.label, err, tester.AccessLabels())
				}
				frames(tester)
			}
			if !slices.Equal(host.calls, test.want) {
				t.Fatalf("calls %q, want %q", host.calls, test.want)
			}
		})
	}
}

func TestHomeDemoUsage(t *testing.T) {
	host, err := loadMeetingDemoHost("testdata/meeting")
	if err != nil {
		t.Fatal(err)
	}
	selectMeetingDemoFixture(t, host, "en/home-normal")
	tester := newMeetingDemoTester(t, host)
	for _, text := range []string{"Today", "30 days", "Per minute", "Latency", "120", "150", "200", "2.0 s", "fixture-speech"} {
		if !tester.HasText(text) {
			t.Errorf("no %q in %q", text, tester.Texts())
		}
	}
	for _, text := range []string{"Recent results", "Try it", "By feature", "steps to go"} {
		if tester.HasText(text) {
			t.Errorf("removed Home section %q is still visible", text)
		}
	}

	selectMeetingDemoFixture(t, host, "en/home-error")
	tester = newMeetingDemoTester(t, host)
	if !tester.HasText("Could not read the history.") {
		t.Fatalf("history-error Home: %q", tester.Texts())
	}

	selectMeetingDemoFixture(t, host, "en/home-history-off")
	tester = newMeetingDemoTester(t, host)
	if !tester.HasText("History is off, so nothing is counted.") || tester.HasText("Per minute") {
		t.Fatalf("history-off Home: %q", tester.Texts())
	}
}
