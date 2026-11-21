package main

import (
	"slices"
	"strconv"
	"testing"

	"github.com/egoist/mygo/ui"
)

type historyDemoStep struct {
	label   string
	text    string
	menu    bool
	context bool
	click   bool
	command string
	mods    ui.Modifiers
	key     ui.Key
}

func TestHistoryDemoActionControls(t *testing.T) {
	copyResult := "Weekly plan\nI'll send the new agenda tonight."
	copyRaw := "weekly plan\nI'll send the new agenda tonight"
	for _, test := range []struct {
		name    string
		fixture string
		prepare func(*meetingDemoHost)
		steps   []historyDemoStep
		want    []string
		check   func(*testing.T, *meetingDemoHost)
	}{
		{name: "load", fixture: "en/history-page-error", steps: historySteps("Try again"), want: []string{"loadHistory"}},
		{name: "search", fixture: "en/history-page-normal", steps: []historyDemoStep{{label: "Search history", text: "agenda"}}, want: []string{`searchHistory "agenda"`}},
		{name: "clear search", fixture: "en/history-page-search-empty", steps: historySteps("Clear search"), want: []string{`searchHistory ""`}},
		{name: "tab", fixture: "en/history-page-normal", steps: historySteps("Translate"), want: []string{"showHistoryTab 1"}},
		{name: "settings", fixture: "en/history-page-normal", steps: historySteps("History settings"), want: []string{"openHistorySettingsDialog"}},
		{name: "missing folder settings", fixture: "en/history-page-audio-missing", steps: historySteps("History settings"), want: []string{"openHistorySettingsDialog"}},
		{name: "privacy", fixture: "en/history-page-off", steps: historySteps("Privacy settings"), want: []string{"openHistoryPrivacy"}},
		{name: "empty transcribe", fixture: "en/history-page-empty-transcribe", steps: historySteps("Set up Transcribe"), want: []string{"openHistoryFeature transcribe"}},
		{name: "empty translate", fixture: "en/history-page-empty-translate", steps: historySteps("Set up Translate"), want: []string{"openHistoryFeature translate"}},
		{name: "empty rewrite", fixture: "en/history-page-empty-rewrite", steps: historySteps("Set up Rewrite"), want: []string{"openHistoryFeature rewrite"}},
		{name: "open detail", fixture: "en/history-page-normal", steps: historySteps("Weekly plan"), want: []string{"openHistoryDetail history-inserted"}},
		{name: "close selected row", fixture: "en/history-detail-normal", steps: historySteps("Weekly plan"), want: []string{"closeHistoryDetail"}},
		{name: "close detail", fixture: "en/history-detail-normal", steps: historySteps("Close"), want: []string{"closeHistoryDetail"}},
		{name: "copy result", fixture: "en/history-detail-normal", steps: historySteps("Copy result"), want: []string{`copyHistoryText ` + quoted(copyResult)}},
		{name: "copy raw", fixture: "en/history-detail-normal", steps: historySteps("Copy recognised text"), want: []string{`copyHistoryText ` + quoted(copyRaw)}},
		{name: "copy instruction", fixture: "en/history-detail-rewrite-selection", steps: historySteps("Copy instruction"), want: []string{`copyHistoryText "Rewrite formally"`}},
		{name: "copy selected text", fixture: "en/history-detail-translate-selection", steps: historySteps("Copy selected text"), want: []string{`copyHistoryText "selected source"`}},
		{name: "detail reveal", fixture: "en/history-detail-normal", steps: historySteps("Show recording in Finder"), want: []string{"showHistoryRecording history-inserted"}},
		{name: "transcribe again", fixture: "en/history-detail-transcribe-again", steps: historySteps("Transcribe again"), want: []string{"retryHistoryEntry history-inserted"}},
		{name: "detail retry", fixture: "en/history-detail-retry", steps: historySteps("Retry"), want: []string{"retryHistoryEntry history-retry"}},
		{name: "row retry", fixture: "en/history-page-normal", steps: historySteps("Retry"), want: []string{"retryHistoryEntry history-cancelled"}},
		{name: "menu copy", fixture: "en/history-page-normal", steps: []historyDemoStep{{label: "Weekly plan", context: true}, {label: "Copy", menu: true}}, want: []string{`copyHistoryText ` + quoted(copyResult)}},
		{name: "menu raw", fixture: "en/history-page-normal", steps: []historyDemoStep{{label: "Weekly plan", context: true}, {label: "Copy recognised text", menu: true}}, want: []string{`copyHistoryText ` + quoted(copyRaw)}},
		{name: "menu details", fixture: "en/history-page-normal", steps: []historyDemoStep{{label: "Weekly plan", context: true}, {label: "Show details", menu: true}}, want: []string{"openHistoryDetail history-inserted"}},
		{name: "menu reveal", fixture: "en/history-page-normal", steps: []historyDemoStep{{label: "Weekly plan", context: true}, {label: "Show recording in Finder", menu: true}}, want: []string{"showHistoryRecording history-inserted"}},
		{name: "menu retry", fixture: "en/history-page-normal", steps: []historyDemoStep{{label: "No text", context: true}, {label: "Retry", menu: true}}, want: []string{"retryHistoryEntry history-cancelled"}},
		{name: "menu delete", fixture: "en/history-page-normal", steps: []historyDemoStep{{label: "Weekly plan", context: true}, {label: "Delete", menu: true}, {label: "Delete"}}, want: []string{"deleteHistoryEntry history-inserted"}},
		{name: "delete cancel", fixture: "en/history-detail-normal", steps: historySteps("Delete entry…", "Cancel"), check: func(t *testing.T, host *meetingDemoHost) {
			if host.historyUI.delAsk {
				t.Fatal("delete confirmation stayed open")
			}
		}},
		{name: "keyboard delete", fixture: "en/history-page-normal", steps: []historyDemoStep{{label: "Weekly plan"}, {key: ui.KeyBackspace}, {label: "Delete"}}, want: []string{"openHistoryDetail history-inserted", "deleteHistoryEntry history-inserted"}},
		{name: "keyboard copy", fixture: "en/history-page-normal", steps: []historyDemoStep{{label: "Weekly plan"}, {mods: ui.Cmd, key: ui.KeyC}}, want: []string{"openHistoryDetail history-inserted", `copyHistoryText ` + quoted(copyResult)}},
		{name: "command copy", fixture: "en/history-page-normal", steps: []historyDemoStep{{label: "Weekly plan"}, {command: "copy"}}, want: []string{"openHistoryDetail history-inserted", `copyHistoryText ` + quoted(copyResult)}},
		{name: "choose folder", fixture: "en/history-settings-normal", steps: historySteps("Choose…"), want: []string{"chooseHistoryRecordingFolder"}},
		{name: "reveal folder", fixture: "en/history-settings-normal", steps: historySteps("Show in Finder"), want: []string{"showHistoryRecordingFolder"}},
		{name: "default folder", fixture: "en/history-settings-normal", steps: historySteps("Use default"), want: []string{"useDefaultHistoryRecordingFolder"}},
		{name: "export recordings", fixture: "en/history-settings-normal", steps: historySteps("Export…"), want: []string{"exportHistoryRecordings false"}},
		{name: "export shown recordings", fixture: "en/history-settings-normal", prepare: func(host *meetingDemoHost) {
			host.historySettingsUI.exportSel = host.current().state.HistorySettings.RecordingScopes[1]
		}, steps: historySteps("Export…"), want: []string{"exportHistoryRecordings true"}},
		{name: "export text", fixture: "en/history-settings-normal", steps: historySteps("Export text…"), want: []string{"exportShownHistoryText false md"}},
		{name: "export shown jsonl", fixture: "en/history-settings-filtered-jsonl", prepare: func(host *meetingDemoHost) {
			host.historySettingsUI.textScope = host.current().state.HistorySettings.TextScopes[1]
		}, steps: historySteps("Export text…"), want: []string{"exportShownHistoryText true jsonl"}},
		{name: "delete recordings", fixture: "en/history-settings-normal", steps: historySteps("Delete all…", "Delete recordings"), want: []string{"deleteHistoryRecordings"}},
		{name: "delete recordings cancel", fixture: "en/history-settings-normal", steps: historySteps("Delete all…", "Cancel"), check: func(t *testing.T, host *meetingDemoHost) {
			if host.historySettingsUI.delAudioAsk {
				t.Fatal("recording deletion stayed open")
			}
		}},
		{name: "cancel move", fixture: "en/history-settings-move", steps: historySteps("Cancel"), want: []string{"cancelHistoryWork"}},
		{name: "cleanup off", fixture: "en/history-settings-normal", steps: historySteps("Clean up history"), want: []string{"setHistoryCleanup false"}},
		{name: "cleanup days", fixture: "en/history-settings-normal", steps: []historyDemoStep{{label: "30 days", click: true}, {label: "1 week", click: true}}, want: []string{"setHistoryCleanupDays 7"}},
		{name: "cleanup count", fixture: "en/history-settings-normal", steps: []historyDemoStep{{label: "No limit", click: true}, {label: "100", click: true}}, want: []string{"setHistoryCleanupCount 100"}},
		{name: "cleanup confirm", fixture: "en/history-settings-prune-ask", steps: historySteps("Delete"), want: []string{"savePendingHistoryCleanup"}},
		{name: "cleanup cancel", fixture: "en/history-settings-prune-ask", steps: historySteps("Cancel"), check: func(t *testing.T, host *meetingDemoHost) {
			if host.historySettingsUI.pruneAsk {
				t.Fatal("cleanup confirmation stayed open")
			}
		}},
		{name: "settings done", fixture: "en/history-settings-normal", steps: historySteps("Done"), check: func(t *testing.T, host *meetingDemoHost) {
			if host.historySettingsUI.open {
				t.Fatal("settings dialog stayed open")
			}
		}},
		{name: "access ask", fixture: "en/history-access-off", steps: historySteps("Allow reading"), check: func(t *testing.T, host *meetingDemoHost) {
			if !host.historyAccessUI.ask {
				t.Fatal("access confirmation did not open")
			}
		}},
		{name: "access on", fixture: "en/history-access-ask", steps: historySteps("Turn on"), want: []string{"setHistoryAccess true"}},
		{name: "access cancel", fixture: "en/history-access-ask", steps: historySteps("Cancel"), check: func(t *testing.T, host *meetingDemoHost) {
			if host.historyAccessUI.ask {
				t.Fatal("access confirmation stayed open")
			}
		}},
		{name: "access off", fixture: "en/history-access-on", steps: historySteps("Allow reading"), want: []string{"setHistoryAccess false"}},
		{name: "access selection", fixture: "en/history-access-on", steps: historySteps("Include selected text"), want: []string{"setHistorySelection false"}},
		{name: "access log", fixture: "en/history-access-on", steps: historySteps("Read log…", "Done"), check: func(t *testing.T, host *meetingDemoHost) {
			if host.historyAccessUI.listOpen {
				t.Fatal("read log stayed open")
			}
		}},
		{name: "access copy", fixture: "en/history-access-on", steps: historySteps("Copy"), want: []string{"copyHistoryAccessCommand"}},
		{name: "access docs", fixture: "en/history-access-on", steps: historySteps("How to use it"), want: []string{"openHistoryAccessDocs"}},
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
			if test.check != nil {
				test.check(t, host)
			}
		})
	}
}

func historySteps(labels ...string) []historyDemoStep {
	steps := make([]historyDemoStep, len(labels))
	for index, label := range labels {
		steps[index].label = label
	}
	return steps
}

func quoted(text string) string {
	return strconv.Quote(text)
}
