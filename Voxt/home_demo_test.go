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
		{name: "fold looking back", fixture: "en/home-looking-back", steps: historySteps("Looking back"), want: []string{"setInsightsClosed true"}},
		{name: "open looking back", fixture: "en/home-looking-back-closed", steps: historySteps("Looking back"), want: []string{"setInsightsClosed false"}},
		{name: "share card", fixture: "en/home-normal", steps: historySteps("Share a card of Voxt and you"), want: []string{"openShare"}},
		{name: "share folded looking back", fixture: "en/home-looking-back-closed", steps: historySteps("Share a look back"), want: []string{"openInsightsShare"}},
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
				tester.Advance(shareSparkleLength) // Share opens once its stars have flown
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

func TestHomeDemoHidesFeatureTry(t *testing.T) {
	host, err := loadMeetingDemoHost("testdata/meeting")
	if err != nil {
		t.Fatal(err)
	}
	selectMeetingDemoFixture(t, host, "en/home-empty")
	tester := newMeetingDemoTester(t, host)
	if tester.HasText("Try it") {
		t.Fatal("the feature-owned Try card leaked into the Home demo")
	}
}

// TestLookingBackNumbersWhole: Looking back's cards never cut a number,
// its unit or its label ("226 characters a minute" was "2… characters a
// min…"), and they go four in a row while they fit, else two by two.
func TestLookingBackNumbersWhole(t *testing.T) {
	host, err := loadMeetingDemoHost("testdata/meeting")
	if err != nil {
		t.Fatal(err)
	}
	selectMeetingDemoFixture(t, host, "en/home-looking-back")
	tester := newMeetingDemoTester(t, host)
	widths := func() (number, unit float32) {
		t.Helper()
		frames(tester)
		n, ok := tester.Find("226")
		u, ok2 := tester.Find("characters a minute")
		if !ok || !ok2 {
			t.Fatalf("no speed card: %q", tester.Texts())
		}
		return n.W, u.W
	}
	number, unit := widths() // the demo's own window: room for all four
	label := func() float32 {
		t.Helper()
		l, ok := tester.Find("Time spoken")
		if !ok {
			t.Fatalf("label cut: %q", tester.Texts())
		}
		return l.W
	}
	whole := label()
	// rows is how many rows the four cards take, by their numbers' tops.
	rows := func() int {
		t.Helper()
		tops := map[float32]bool{}
		for _, text := range []string{"2,220", "42", "226", "9"} {
			r, ok := tester.Find(text)
			if !ok {
				t.Fatalf("no %q: %q", text, tester.Texts())
			}
			tops[r.Y] = true
		}
		return len(tops)
	}
	for _, test := range []struct{ width, rows int }{{400, 2}, {480, 2}, {560, 2}, {627, 2}, {700, 1}, {735, 1}} {
		tester.SetSize(test.width, 700)
		if n, u := widths(); n < number-1 || u < unit-1 { // a pixel of rounding
			t.Errorf("%d wide: the speed is %.0f and its unit %.0f wide, want %.0f and %.0f", test.width, n, u, number, unit)
		}
		if l := label(); l < whole-1 {
			t.Errorf("%d wide: Time spoken is %.0f wide, want %.0f", test.width, l, whole)
		}
		if got := rows(); got != test.rows {
			t.Errorf("%d wide: the cards take %d rows, want %d", test.width, got, test.rows)
		}
	}
}
