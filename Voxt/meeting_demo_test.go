package main

import (
	"image/png"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
)

func newMeetingDemoTester(t *testing.T, host *meetingDemoHost) *ui.Tester {
	t.Helper()
	fixture := host.current()
	width, height := 1000, 700
	if fixture.state.Kind == "overlay" && fixture.state.Overlay.Visible {
		model := meetingOverlayLayoutModel(*fixture.state.Overlay, host.overlayUI.menu)
		width, height = model.Layout.Width, model.Layout.Height+48
	}
	if fixture.state.Kind == "appenhance" {
		height = 1400
	}
	tester := ui.NewTester(func(c *ui.Context) { meetingDemoView(c, host) }, width, height)
	tester.SetLocale(fixture.locale)
	tester.SetFontSmoothing(hudSmoothing)
	frames(tester)
	return tester
}

func selectMeetingDemoFixture(t *testing.T, host *meetingDemoHost, name string) {
	t.Helper()
	for index, fixture := range host.fixtures {
		if fixture.name == name {
			host.selectMeetingFixture(index - host.selected)
			return
		}
	}
	t.Fatalf("no fixture %s", name)
}

func TestMeetingDemoScreenshots(t *testing.T) {
	host, err := loadMeetingDemoHost("testdata/meeting")
	if err != nil {
		t.Fatal(err)
	}
	output := os.Getenv("VOXTNEW_DEMO_SHOTS")
	for index, fixture := range host.fixtures {
		t.Run(fixture.name, func(t *testing.T) {
			host.selectMeetingFixture(index - host.selected)
			tester := newMeetingDemoTester(t, host)
			want := meetingDemoSentinel(fixture)
			if want != "" && !tester.HasText(want) {
				t.Fatalf("fixture does not show %q; texts %q", want, tester.Texts())
			}
			image := tester.Image()
			if image.Bounds().Empty() || len(image.Pix) == 0 {
				t.Fatal("empty frame")
			}
			if output == "" {
				return
			}
			if err := os.MkdirAll(output, 0o755); err != nil {
				t.Fatal(err)
			}
			path := filepath.Join(output, strings.ReplaceAll(fixture.name, "/", "-")+".png")
			file, err := os.Create(path)
			if err != nil {
				t.Fatal(err)
			}
			encodeErr := png.Encode(file, image)
			closeErr := file.Close()
			if encodeErr != nil {
				t.Fatal(encodeErr)
			}
			if closeErr != nil {
				t.Fatal(closeErr)
			}
		})
	}
}

func meetingDemoSentinel(fixture meetingDemoFixture) string {
	state := fixture.state
	switch state.Kind {
	case "start":
		return state.Start.Text
	case "overlay":
		overlay := state.Overlay
		if !overlay.Visible {
			return fixture.name
		}
		if overlay.ShowAsking {
			return i18n.T("Save this meeting?")
		}
		if overlay.ShowFinishing {
			return overlay.FinishingText
		}
		if overlay.ProblemText != "" {
			return overlay.ProblemText[:min(20, len(overlay.ProblemText))]
		}
		return overlay.ElapsedText
	case "page":
		page := state.Page
		if page.Detail != nil {
			if page.Detail.Loading {
				return i18n.T("Loading…")
			}
			if page.Detail.Error != "" {
				return page.Detail.Error
			}
			return page.Detail.Title
		}
		if page.Error != "" {
			return page.Error
		}
		if page.Query != "" && page.MeetingCount == 0 {
			return i18n.T("Nothing matches “%s”", page.Query)
		}
		if page.Read && page.MeetingCount == 0 && page.Query == "" {
			return i18n.T("No meetings yet.")
		}
		return i18n.T("Meetings")
	case "history":
		history := state.History
		if history.Detail != nil {
			return i18n.T("Entry details")
		}
		if history.Error != "" {
			return history.Error
		}
		if history.Empty != nil {
			return history.Empty.Title
		}
		if history.FolderNotice != "" {
			return history.FolderNotice
		}
		for _, row := range history.Rows {
			if row.Retry.Active {
				return row.Retry.StageText
			}
		}
		if history.CountLabel != "" {
			return history.CountLabel
		}
		return i18n.T("History")
	case "history-settings":
		settings := state.HistorySettings
		if settings.PruneAsk {
			return settings.PrunePrompt
		}
		if settings.DeleteAudioAsk {
			return i18n.T("Delete every recording?")
		}
		if settings.Busy {
			return settings.BusyLabel
		}
		if settings.MissingNotice != "" {
			return settings.MissingNotice
		}
		return i18n.T("History settings")
	case "history-access":
		access := state.HistoryAccess
		if access.Ask {
			return i18n.T("Let programs on this Mac read the history?")
		}
		if access.ListOpen {
			return i18n.T("Read log")
		}
		if access.ShowLastRead {
			return access.LastRead
		}
		return i18n.T("Other programs")
	case "appenhance":
		enhancement := state.AppEnhancement
		if enhancement.Apps != nil {
			if enhancement.Apps.Loading {
				return i18n.T("Looking for apps…")
			}
			if enhancement.Apps.Error != "" {
				return enhancement.Apps.Error
			}
			if enhancement.Apps.NoMatches {
				return i18n.T("No app matches.")
			}
			return enhancement.Apps.Title
		}
		if enhancement.Browsers != nil {
			return i18n.T("Voxt reads the current tab's address when you press a shortcut.")
		}
		if enhancement.Delete != nil {
			return enhancement.Delete.Title
		}
		if enhancement.Error != "" {
			return enhancement.Error
		}
		if !enhancement.Enabled {
			return i18n.T("App enhancement is off, so every app uses only the features' own rules.")
		}
		if enhancement.Detail != nil {
			if enhancement.Detail.AutoKey.Error != "" {
				return enhancement.Detail.AutoKey.Error
			}
			if enhancement.Detail.URLError != "" {
				return enhancement.Detail.URLError
			}
			for _, row := range enhancement.Detail.URLs {
				if row.Warning != "" {
					return row.Warning
				}
				if row.Canonical != "" {
					return row.Canonical
				}
			}
		}
		if enhancement.Preview.Empty {
			return enhancement.Preview.FrontText
		}
		if enhancement.Preview.Address != "" {
			return enhancement.Preview.Address
		}
		if enhancement.Preview.AddSite {
			return enhancement.Preview.AddSiteLabel
		}
		if enhancement.Preview.AddApp {
			return enhancement.Preview.AddAppLabel
		}
		if enhancement.Detail == nil {
			return i18n.T("Rules for the frontmost app or website")
		}
		return enhancement.Preview.MatchText
	default:
		return ""
	}
}

func TestMeetingDemoSwitchesFixtures(t *testing.T) {
	previousLocale := i18n.Locale()
	t.Cleanup(func() { i18n.SetLocale(previousLocale) })
	host, err := loadMeetingDemoHost("testdata/meeting")
	if err != nil {
		t.Fatal(err)
	}
	initial := host.selected
	host.pageUI.query = "stale"
	host.pageUI.detail = &meetingDetailLocalUI{question: "stale"}
	host.overlayUI.menu = meetingMenuMore
	host.historyUI.query, host.historyUI.sel = "stale", "stale"
	host.historySettingsUI.exportSel = "stale"
	host.historyAccessUI.listOpen = true
	host.appEnhancementUI.sel = "stale"
	host.appEnhancementUI.appsQuery = "stale"
	tester := newMeetingDemoTester(t, host)
	if err := tester.ClickControl("Continue"); err != nil {
		t.Fatal(err)
	}
	frames(tester)
	if host.selected != (initial+1)%len(host.fixtures) {
		t.Fatalf("selected %d, want %d", host.selected, initial+1)
	}
	if want := meetingDemoSentinel(host.current()); want != "" && !tester.HasText(want) {
		t.Fatalf("next fixture does not show %q", want)
	}
	if host.pageUI.query != "" || host.overlayUI.menu != meetingMenuNone || host.pageUI.detail != nil && host.pageUI.detail.question != "" ||
		host.historyUI.query != "" || host.historyUI.sel != "" || host.historySettingsUI.exportSel != "" || host.historyAccessUI.listOpen ||
		host.appEnhancementUI.appsQuery != "" || host.appEnhancementUI.sel == "stale" {
		t.Fatal("fixture switch kept local UI state")
	}
	if err := tester.ClickControl("Back"); err != nil {
		t.Fatal(err)
	}
	frames(tester)
	if host.selected != initial {
		t.Fatalf("previous selected %d, want %d", host.selected, initial)
	}
	if err := tester.ClickControl("Back"); err != nil {
		t.Fatal(err)
	}
	frames(tester)
	want := (initial - 1 + len(host.fixtures)) % len(host.fixtures)
	if host.selected != want {
		t.Fatalf("wrapped selected %d, want %d", host.selected, want)
	}
	if i18n.Locale() != host.current().locale {
		t.Fatalf("locale %q, fixture locale %q", i18n.Locale(), host.current().locale)
	}
	if len(host.calls) != 0 {
		t.Fatalf("selector called domain actions: %v", host.calls)
	}
}

func TestMeetingDemoRejectsUnknownFields(t *testing.T) {
	directory := t.TempDir()
	if err := os.Mkdir(filepath.Join(directory, "en"), 0o755); err != nil {
		t.Fatal(err)
	}
	data := []byte(`{"name":"bad","kind":"page","page":{},"unknown":true}`)
	if err := os.WriteFile(filepath.Join(directory, "en", "bad.json"), data, 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := loadMeetingDemoHost(directory); err == nil || !strings.Contains(err.Error(), "unknown") {
		t.Fatalf("load error %v", err)
	}
}

type meetingDemoStep struct {
	label string
	menu  bool
	text  string
}

func TestMeetingDemoActionControls(t *testing.T) {
	for _, test := range []struct {
		name    string
		fixture string
		prepare func(*meetingDemoHost)
		steps   []meetingDemoStep
		want    []string
	}{
		{name: "start", fixture: "en/page-empty", steps: steps("Start meeting"), want: []string{"startMeeting"}},
		{name: "finish", fixture: "en/page-normal", steps: steps("End meeting"), want: []string{"meetingFinish"}},
		{name: "retry", fixture: "en/page-error", steps: steps("Try again"), want: []string{"loadMeetings"}},
		{name: "search", fixture: "en/page-normal", steps: []meetingDemoStep{{label: "Search meetings", text: "needle"}}, want: []string{`searchMeetings "needle"`}},
		{name: "clear search", fixture: "en/page-search-empty", steps: steps("Clear search"), want: []string{`searchMeetings ""`}},
		{name: "next page", fixture: "en/page-normal", steps: steps("Next page"), want: []string{"showMeetingsPage 1"}},
		{name: "previous page", fixture: "en/page-normal", prepare: func(host *meetingDemoHost) { host.pageUI.page = 1 }, steps: steps("Previous page"), want: []string{"showMeetingsPage 0"}},
		{name: "open detail", fixture: "en/page-normal", steps: steps("Fixture recording"), want: []string{"openMeetingDetail meeting-recording"}},
		{name: "close selected row", fixture: "en/page-normal", steps: steps("Fixture done"), want: []string{"closeMeetingDetail"}},
		{name: "close detail", fixture: "en/detail-normal", steps: steps("Close"), want: []string{"closeMeetingDetail"}},
		{name: "close loading detail", fixture: "en/detail-loading", steps: steps("Close"), want: []string{"closeMeetingDetail"}},
		{name: "close error detail", fixture: "en/detail-error", steps: steps("Close"), want: []string{"closeMeetingDetail"}},
		{name: "delete", fixture: "en/detail-normal", steps: steps("Delete meeting…", "Delete meeting"), want: []string{"deleteMeeting detail-1"}},
		{name: "delete cancel", fixture: "en/detail-normal", steps: steps("Delete meeting…", "Cancel")},
		{name: "continue", fixture: "en/detail-interrupted", steps: steps("Continue processing"), want: []string{"meetingReprocess detail-1"}},
		{name: "reprocess", fixture: "en/detail-normal", steps: []meetingDemoStep{{label: "More actions"}, {label: "Reprocess transcript", menu: true}}, want: []string{"meetingReprocess detail-1"}},
		{name: "correct", fixture: "en/detail-normal", steps: []meetingDemoStep{{label: "More actions"}, {label: "Correct with the language model", menu: true}}, want: []string{"meetingCorrect detail-1"}},
		{name: "finder", fixture: "en/detail-normal", steps: []meetingDemoStep{{label: "More actions"}, {label: "Show in Finder", menu: true}}, want: []string{"showMeetingInFinder detail-1"}},
		{name: "regenerate", fixture: "en/detail-normal", steps: steps("Regenerate summary"), want: []string{"regenerateMeetingSummary detail-1"}},
		{name: "follow up", fixture: "en/detail-normal", prepare: func(host *meetingDemoHost) { host.pageUI.detail = &meetingDetailLocalUI{question: "What happened?"} }, steps: steps("Send question"), want: []string{`askMeetingFollowUp detail-1 "What happened?"`}},
		{name: "language models", fixture: "en/detail-empty-no-llm", steps: steps("Open Language models"), want: []string{"openMeetingLanguageModels"}},
		{name: "copy", fixture: "en/detail-normal", steps: steps("Copy transcript"), want: []string{"copyMeetingTranscript detail-1"}},
		{name: "export text", fixture: "en/detail-normal", steps: []meetingDemoStep{{label: "Export transcript"}, {label: "Export .txt…", menu: true}}, want: []string{"saveMeetingExport detail-1 txt"}},
		{name: "export markdown", fixture: "en/detail-normal", steps: []meetingDemoStep{{label: "Export transcript"}, {label: "Export Markdown…", menu: true}}, want: []string{"saveMeetingExport detail-1 md"}},
		{name: "play", fixture: "en/detail-normal", steps: steps("Play"), want: []string{"toggleMeetingPlayback detail-1"}},
		{name: "pause playback", fixture: "en/detail-playing", steps: steps("Pause"), want: []string{"toggleMeetingPlayback detail-1"}},
		{name: "seek first", fixture: "en/detail-normal", steps: steps("00:01 You"), want: []string{"seekMeetingPlayback detail-1 1000"}},
		{name: "seek second", fixture: "en/detail-normal", steps: steps("00:03 Alice"), want: []string{"seekMeetingPlayback detail-1 3000"}},
		{name: "pause", fixture: "en/state-recording-meeting", steps: steps("Pause meeting"), want: []string{"meetingPauseResume"}},
		{name: "resume", fixture: "en/state-paused", steps: steps("Resume meeting"), want: []string{"meetingPauseResume"}},
		{name: "end overlay", fixture: "en/state-recording-meeting", steps: steps("End meeting"), want: []string{"meetingFinish"}},
		{name: "end starting", fixture: "en/state-starting", steps: steps("End meeting"), want: []string{"meetingFinish"}},
		{name: "save", fixture: "en/state-asking", steps: steps("Save"), want: []string{"meetingSave"}},
		{name: "discard", fixture: "en/state-asking", steps: steps("Discard"), want: []string{"meetingDiscard"}},
		{name: "mode meeting", fixture: "en/state-recording-subtitles", steps: steps("Meeting mode", "Meeting"), want: []string{"toggleMeetingMenu 1", "meetingSetModeID meeting"}},
		{name: "mode subtitles", fixture: "en/state-recording-meeting", steps: steps("Meeting mode", "Subtitles"), want: []string{"toggleMeetingMenu 1", "meetingSetModeID subtitles"}},
		{name: "mode recording", fixture: "en/state-recording-meeting", steps: steps("Meeting mode", "Recording"), want: []string{"toggleMeetingMenu 1", "meetingSetModeID recording"}},
		{name: "mode close", fixture: "en/state-recording-meeting", steps: steps("Meeting mode", "Meeting mode"), want: []string{"toggleMeetingMenu 1", "toggleMeetingMenu 0"}},
		{name: "translate", fixture: "en/state-recording-meeting", steps: steps("More", "Live translation"), want: []string{"toggleMeetingMenu 2", "toggleMeetingTranslation fixture-live"}},
		{name: "copy live", fixture: "en/state-paused", steps: steps("More", "Copy transcript"), want: []string{"toggleMeetingMenu 2", "copyLiveMeetingTranscript"}},
		{name: "cancel live", fixture: "en/state-recording-meeting", steps: steps("More", "Cancel meeting", "Cancel meeting"), want: []string{"toggleMeetingMenu 2", "requestMeetingDiscard", "meetingDiscard"}},
		{name: "keep recording", fixture: "en/state-recording-meeting", steps: steps("More", "Cancel meeting", "Keep recording"), want: []string{"toggleMeetingMenu 2", "requestMeetingDiscard"}},
		{name: "collapse", fixture: "en/state-recording-meeting", steps: steps("Collapse"), want: []string{"setMeetingOverlayCollapsed true"}},
		{name: "expand", fixture: "en/state-collapsed", steps: steps("Expand"), want: []string{"setMeetingOverlayCollapsed false"}},
		{name: "card larger", fixture: "en/state-recording-meeting", steps: steps("Make the card larger"), want: []string{"setMeetingOverlayCardSize 2"}},
		{name: "card smaller", fixture: "en/state-card-largest", steps: steps("Make the card smaller"), want: []string{"setMeetingOverlayCardSize 0"}},
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
			switch test.name {
			case "delete cancel":
				if host.pageUI.delID != "" {
					t.Fatal("cancel did not close the delete dialog")
				}
			case "keep recording":
				if host.overlayUI.abandon {
					t.Fatal("keep recording did not close the cancel dialog")
				}
			case "collapse":
				if !tester.HasText("Expand") {
					t.Fatal("collapse did not collapse the overlay")
				}
			case "expand":
				if !tester.HasText("Collapse") {
					t.Fatal("expand did not expand the overlay")
				}
			}
		})
	}
}

func steps(labels ...string) []meetingDemoStep {
	steps := make([]meetingDemoStep, len(labels))
	for index, label := range labels {
		steps[index].label = label
	}
	return steps
}
