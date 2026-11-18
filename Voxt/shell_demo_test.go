package main

// The shell's fixture demo: clicking every action-bearing control
// records the action the app would run.

import (
	"slices"
	"testing"
)

func TestShellDemoActionControls(t *testing.T) {
	for _, test := range []struct {
		name    string
		fixture string
		prepare func(*meetingDemoHost)
		steps   []meetingDemoStep
		want    []string
		check   func(*testing.T, *meetingDemoHost)
	}{
		{name: "nav page", fixture: "en/nav-sidebar", steps: steps("History"), want: []string{"navigate /history"}},
		{name: "nav search", fixture: "en/nav-sidebar", steps: steps("Command palette"), want: []string{"openPalette"}},
		{name: "nav rail on", fixture: "en/nav-sidebar", steps: steps("Show icons only"), want: []string{"setRail true"}},
		{name: "nav rail off", fixture: "en/nav-rail", steps: steps("Show sidebar"), want: []string{"setRail false"}},
		{name: "nav rail page", fixture: "en/nav-rail", steps: steps("History"), want: []string{"navigate /history"}},
		{name: "nav settings", fixture: "en/nav-sidebar", steps: steps("Settings"), want: []string{"navigate /settings"}},
		{name: "notice dismiss", fixture: "en/notices-warning", steps: steps("Dismiss"), want: []string{"dismissNotice n1"}},
		{name: "notice clear", fixture: "en/notices-stack", steps: steps("Clear all"), want: []string{"clearNotices"}},
		{name: "notice action", fixture: "en/notices-models-both", steps: steps("Open Speech models"), want: []string{"noticeCardAction problem/models"}},
		{name: "notice helper", fixture: "en/notices-helper-missing", steps: steps("Restart helper"), want: []string{"noticeCardAction problem/helper"}},
		{name: "notice link", fixture: "en/notices-actions", steps: steps("Open"), want: []string{"noticeCardAction n5"}},
		{name: "logs copy all", fixture: "en/logs-lines", steps: steps("Copy all lines shown"), want: []string{"copyLogLines [5 4 3 2 1]"}},
		{name: "logs clear filter", fixture: "en/logs-filter-empty", steps: steps("Clear filter"), check: func(t *testing.T, host *meetingDemoHost) {
			t.Helper()
			if host.logsQuery != "" {
				t.Fatalf("clear filter left %q", host.logsQuery)
			}
		}},
		{name: "palette run", fixture: "en/palette-all", steps: steps("Dictionary"), want: []string{"runCommandID go/dictionary"}},
		{name: "sheet close", fixture: "en/sheet-hotkeys", steps: steps("Close"), check: func(t *testing.T, host *meetingDemoHost) {
			t.Helper()
			if host.sheetOpen {
				t.Fatal("Close did not close the sheet")
			}
		}},
		{name: "announcement done", fixture: "en/shell-announcement-version", steps: steps("Done"), check: func(t *testing.T, host *meetingDemoHost) {
			t.Helper()
			if host.announcementOpen {
				t.Fatal("Done did not close the announcement")
			}
		}},
		{name: "shell notices", fixture: "en/shell-busy", steps: steps("Clear all"), want: []string{"clearNotices"}},
	} {
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
			for _, step := range test.steps {
				if step.menu {
					if err := tester.ChooseMenuItem(step.label); err != nil {
						t.Fatalf("%s: %v (texts %q)", step.label, err, tester.Texts())
					}
				} else if step.text != "" {
					if err := tester.Click(step.label); err != nil {
						t.Fatalf("%s: %v (texts %q)", step.label, err, tester.Texts())
					}
					tester.Type(step.text)
				} else if err := tester.ClickControl(step.label); err != nil {
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
