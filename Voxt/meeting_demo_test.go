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
	default:
		return ""
	}
}

func TestMeetingDemoSwitchesFixtures(t *testing.T) {
	host, err := loadMeetingDemoHost("testdata/meeting")
	if err != nil {
		t.Fatal(err)
	}
	initial := host.selected
	tester := newMeetingDemoTester(t, host)
	if err := tester.ClickControl("Continue"); err != nil {
		t.Fatal(err)
	}
	frames(tester)
	if host.selected != (initial+1)%len(host.fixtures) {
		t.Fatalf("selected %d, want %d", host.selected, initial+1)
	}
	if len(host.calls) != 0 {
		t.Fatalf("selector called domain actions: %v", host.calls)
	}
	if err := tester.ClickControl("Back"); err != nil {
		t.Fatal(err)
	}
	if host.selected != initial {
		t.Fatalf("previous selected %d, want %d", host.selected, initial)
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
		{name: "clear search", fixture: "en/page-search-empty", steps: steps("Clear search"), want: []string{`searchMeetings ""`}},
		{name: "next page", fixture: "en/page-normal", steps: steps("Next page"), want: []string{"showMeetingsPage 1"}},
		{name: "open detail", fixture: "en/page-normal", steps: steps("Fixture recording"), want: []string{"openMeetingDetail meeting-recording"}},
		{name: "close detail", fixture: "en/detail-normal", steps: steps("Close"), want: []string{"closeMeetingDetail"}},
		{name: "delete", fixture: "en/detail-normal", steps: steps("Delete meeting…", "Delete meeting"), want: []string{"deleteMeeting detail-1"}},
		{name: "reprocess", fixture: "en/detail-normal", steps: []meetingDemoStep{{"More actions", false}, {"Reprocess transcript", true}}, want: []string{"meetingReprocess detail-1"}},
		{name: "correct", fixture: "en/detail-normal", steps: []meetingDemoStep{{"More actions", false}, {"Correct with the language model", true}}, want: []string{"meetingCorrect detail-1"}},
		{name: "finder", fixture: "en/detail-normal", steps: []meetingDemoStep{{"More actions", false}, {"Show in Finder", true}}, want: []string{"showMeetingInFinder detail-1"}},
		{name: "regenerate", fixture: "en/detail-normal", steps: steps("Regenerate summary"), want: []string{"regenerateMeetingSummary detail-1"}},
		{name: "follow up", fixture: "en/detail-normal", prepare: func(host *meetingDemoHost) { host.pageUI.detail = &meetingDetailLocalUI{question: "What happened?"} }, steps: steps("Send question"), want: []string{`askMeetingFollowUp detail-1 "What happened?"`}},
		{name: "language models", fixture: "en/detail-empty-no-llm", steps: steps("Open Language models"), want: []string{"openMeetingLanguageModels"}},
		{name: "copy", fixture: "en/detail-normal", steps: steps("Copy transcript"), want: []string{"copyMeetingTranscript detail-1"}},
		{name: "export text", fixture: "en/detail-normal", steps: []meetingDemoStep{{"Export transcript", false}, {"Export .txt…", true}}, want: []string{"saveMeetingExport detail-1 txt"}},
		{name: "export markdown", fixture: "en/detail-normal", steps: []meetingDemoStep{{"Export transcript", false}, {"Export Markdown…", true}}, want: []string{"saveMeetingExport detail-1 md"}},
		{name: "play", fixture: "en/detail-normal", steps: steps("Play"), want: []string{"toggleMeetingPlayback detail-1"}},
		{name: "seek", fixture: "en/detail-normal", steps: steps("00:01 You"), want: []string{"seekMeetingPlayback detail-1 1000"}},
		{name: "pause", fixture: "en/state-recording-meeting", steps: steps("Pause meeting"), want: []string{"meetingPauseResume"}},
		{name: "end overlay", fixture: "en/state-recording-meeting", steps: steps("End meeting"), want: []string{"meetingFinish"}},
		{name: "save", fixture: "en/state-asking", steps: steps("Save"), want: []string{"meetingSave"}},
		{name: "discard", fixture: "en/state-asking", steps: steps("Discard"), want: []string{"meetingDiscard"}},
		{name: "mode", fixture: "en/state-recording-meeting", steps: steps("Meeting mode", "Subtitles"), want: []string{"toggleMeetingMenu 1", "meetingSetModeID subtitles"}},
		{name: "translate", fixture: "en/state-recording-meeting", steps: steps("More", "Live translation"), want: []string{"toggleMeetingMenu 2", "toggleMeetingTranslation fixture-live"}},
		{name: "copy live", fixture: "en/state-paused", steps: steps("More", "Copy transcript"), want: []string{"toggleMeetingMenu 2", "copyLiveMeetingTranscript"}},
		{name: "cancel live", fixture: "en/state-recording-meeting", steps: steps("More", "Cancel meeting", "Cancel meeting"), want: []string{"toggleMeetingMenu 2", "requestMeetingDiscard", "meetingDiscard"}},
		{name: "collapse", fixture: "en/state-recording-meeting", steps: steps("Collapse"), want: []string{"setMeetingOverlayCollapsed true"}},
		{name: "expand", fixture: "en/state-recording-meeting", prepare: func(host *meetingDemoHost) {
			state := host.current().state.Overlay
			state.Collapsed, state.ShowExpanded = true, false
			host.overlayUI.collapsed = true
		}, steps: steps("Expand"), want: []string{"setMeetingOverlayCollapsed false"}},
		{name: "card size", fixture: "en/state-recording-meeting", steps: steps("Make the card larger"), want: []string{"setMeetingOverlayCardSize 2"}},
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
				} else if err := tester.ClickControl(step.label); err != nil {
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

func steps(labels ...string) []meetingDemoStep {
	steps := make([]meetingDemoStep, len(labels))
	for index, label := range labels {
		steps[index].label = label
	}
	return steps
}
