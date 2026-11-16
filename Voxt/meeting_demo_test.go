package main

import (
	"image/png"
	"os"
	"path/filepath"
	"reflect"
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
	case "settings":
		dialog := state.Settings
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
	case "meeting-settings":
		meeting := state.MeetingSettings
		if meeting.NoLLM {
			return i18n.T("No language model is set up, so summaries and translation are off.")
		}
		return meeting.CaptureModeSelected
	case "shortcut-prompt":
		return state.ShortcutPrompt.OtherApp
	case "key-recorder":
		recorder := state.KeyRecorder
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
	host.settingsForm.search = "stale"
	host.settingsForm.findN = 3
	host.settingsForm.clearHistoryAsk = true
	host.settingsForm.ovStyle.margin = 42
	host.settingsStorageUI.resetAsk, host.settingsStorageUI.resetText = "everything", "stale"
	host.settingsUpdatesUI.AutoCheck = true
	host.settingsMeetingUI.hide = true
	host.settingsOpen = true
	host.keyRow = keyRow{custom: true, typing: true, text: "stale", edited: true, unknown: true, recordDenied: true, recFocused: true}
	host.recorderActive = true
	host.settingsRouter.Replace("/about")
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
		host.historyUI.query != "" || host.historyUI.sel != "" || host.historySettingsUI.exportSel != "" || host.historyAccessUI.listOpen {
		t.Fatal("fixture switch kept local UI state")
	}
	if !reflect.DeepEqual(host.settingsForm, settingsForm{}) {
		t.Fatalf("fixture switch kept settings form state: %+v", host.settingsForm)
	}
	if host.settingsStorageUI != (settingsStorageLocalUI{}) || host.settingsUpdatesUI != (settingsUpdatesLocalUI{}) ||
		host.settingsMeetingUI != (meetingSettingsUI{}) || host.settingsOpen || host.pageUI.setOpen {
		t.Fatal("fixture switch kept settings dialog state")
	}
	if host.settingsRouter.Path() != "/general" {
		t.Fatal("fixture switch kept settings router")
	}
	if host.keyRow != (keyRow{}) || host.recorderActive {
		t.Fatalf("fixture switch kept key recorder state: %+v, active=%t", host.keyRow, host.recorderActive)
	}
	if host.settingsDialogUI.Open != &host.settingsOpen || host.settingsDialogUI.Router != host.settingsRouter ||
		host.settingsDialogUI.Form != &host.settingsForm || host.settingsDialogUI.HistoryAccess != &host.historyAccessUI ||
		host.settingsDialogUI.Storage != &host.settingsStorageUI || host.settingsDialogUI.Updates != &host.settingsUpdatesUI {
		t.Fatal("settings dialog pointers do not address the host's local state")
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
