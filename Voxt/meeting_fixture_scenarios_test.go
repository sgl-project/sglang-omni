package main

import (
	"errors"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/local"
	"github.com/sgl-project/sglang-omni/Voxt/internal/meeting"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

type meetingFixtureScenario struct {
	name            string
	kind            string
	snapshot        meeting.Snapshot
	page            meetingsViewStateInput
	err             error
	collapsed       bool
	cardSize        int
	history         historyViewStateInput
	historySettings historySettingsStateInput
	historyAccess   historyAccessStateInput
	historyProblem  string
	settings        settingsFixtureInput
	meetingSettings settingsMeetingStateInput
	shortcutPrompt  shortcutPromptFixtureInput
}

var meetingFixtureNow = time.Date(2026, time.January, 2, 15, 4, 0, 0, time.UTC)

func meetingFixtureScenarios() []meetingFixtureScenario {
	var scenarios []meetingFixtureScenario
	addOverlay := func(name string, snapshot meeting.Snapshot) {
		scenarios = append(scenarios, meetingFixtureScenario{name: name, kind: "overlay", snapshot: snapshot, cardSize: 1})
	}
	addOverlay("empty-initial", meeting.Snapshot{})
	for _, state := range []meeting.State{
		meeting.StateIdle, meeting.StateStarting, meeting.StatePaused, meeting.StateAsking,
		meeting.StateDone, meeting.StateDiscarded, meeting.StateFailed, meeting.StateInterrupted,
	} {
		snapshot := meetingFixtureSnapshot()
		snapshot.State = state
		addOverlay("state-"+string(state), snapshot)
	}
	for _, mode := range []meeting.Mode{meeting.ModeMeeting, meeting.ModeSubtitles, meeting.ModeRecording} {
		snapshot := meetingFixtureSnapshot()
		snapshot.Mode = mode
		addOverlay("state-recording-"+string(mode), snapshot)
	}
	for _, stage := range []meeting.Stage{meeting.StageSaving, meeting.StageTranscribing, meeting.StageSummary} {
		snapshot := meetingFixtureSnapshot()
		snapshot.State = meeting.StateFinishing
		snapshot.Stage = stage
		snapshot.Progress = 0.4
		addOverlay("state-finishing-"+string(stage), snapshot)
	}
	for _, problem := range []struct {
		name string
		key  string
		mode meeting.Mode
	}{
		{"system-silent", meeting.ProblemKeySystemSilent, meeting.ModeMeeting},
		{"live-backlog", meeting.ProblemKeyLiveBacklog, meeting.ModeMeeting},
		{"disk-low", meeting.ProblemKeyDiskLow, meeting.ModeMeeting},
		{"helper-old", meeting.ProblemKeyHelperOld, meeting.ModeMeeting},
		{"track-degraded-mic", meeting.ProblemKeyTrackDegraded + string(meeting.SourceMic), meeting.ModeSubtitles},
		{"track-degraded-system", meeting.ProblemKeyTrackDegraded + string(meeting.SourceSystem), meeting.ModeRecording},
	} {
		snapshot := meetingFixtureSnapshot()
		snapshot.Mode = problem.mode
		snapshot.Problem = &meeting.Problem{Key: problem.key}
		addOverlay("problem-"+problem.name, snapshot)
	}
	scenarios = append(scenarios,
		meetingFixtureScenario{name: "state-collapsed", kind: "overlay", snapshot: meetingFixtureSnapshot(), collapsed: true, cardSize: 1},
		meetingFixtureScenario{name: "state-card-largest", kind: "overlay", snapshot: meetingFixtureSnapshot(), cardSize: len(meetingOvWidths) - 1},
	)

	for _, scenario := range []struct {
		name  string
		error error
	}{
		{"model-not-downloaded", local.ErrNotInstalled},
		{"busy", meeting.ErrBusy},
		{"model-unavailable", &meeting.StartError{Source: meeting.SourceMic, Err: errors.New("load failed"), Guidance: meeting.GuidanceModel}},
		{"mic-unavailable", &meeting.StartError{Source: meeting.SourceMic, Err: errors.New("microphone failed"), Guidance: meeting.GuidanceMic}},
		{"system-unavailable", &meeting.StartError{Source: meeting.SourceSystem, Err: errors.New("system failed"), Guidance: meeting.GuidanceSystem}},
		{"generic", errors.New("disk full")},
	} {
		scenarios = append(scenarios, meetingFixtureScenario{name: "start-" + scenario.name, kind: "start", err: scenario.error})
	}

	for _, scenario := range []struct {
		name   string
		mutate func(*meetingsViewStateInput)
	}{
		{"page-loading", func(input *meetingsViewStateInput) { input.Read, input.Loading = false, true }},
		{"page-empty", func(input *meetingsViewStateInput) { input.List, input.Active = nil, false }},
		{"page-normal", func(input *meetingsViewStateInput) {}},
		{"page-error", func(input *meetingsViewStateInput) { input.Error = "fixture read failed" }},
		{"page-search-empty", func(input *meetingsViewStateInput) { input.Query, input.List = "missing", nil }},
		{"page-search-match", func(input *meetingsViewStateInput) {
			input.Query = "needle"
			input.Lines = map[string]string{"meeting-done": "a needle in the transcript"}
		}},
		{"detail-loading", func(input *meetingsViewStateInput) {
			input.Detail = &meetingDetailStateInput{Meeting: meetingFixtureDetailMeeting(), Loading: true}
		}},
		{"detail-error", func(input *meetingsViewStateInput) {
			input.Detail = &meetingDetailStateInput{Meeting: meetingFixtureDetailMeeting(), Error: "fixture detail failed"}
		}},
		{"detail-interrupted", func(input *meetingsViewStateInput) {
			input.Detail = meetingFixtureDetailInput()
			input.Detail.Meeting.Status = store.MeetingInterrupted
		}},
		{"detail-empty-no-llm", func(input *meetingsViewStateInput) {
			stored := meetingFixtureDetailMeeting()
			stored.Summary, stored.Chat = "", ""
			input.Detail = &meetingDetailStateInput{Meeting: stored}
		}},
		{"detail-normal", func(input *meetingsViewStateInput) { input.Detail = meetingFixtureDetailInput() }},
		{"detail-correcting", func(input *meetingsViewStateInput) {
			input.Detail = meetingFixtureDetailInput()
			input.Detail.Correcting = true
		}},
		{"detail-answering", func(input *meetingsViewStateInput) {
			input.Detail = meetingFixtureDetailInput()
			input.Detail.Answering = true
		}},
		{"detail-playing", func(input *meetingsViewStateInput) {
			input.Detail = meetingFixtureDetailInput()
			input.Detail.Playback = meetingPlaybackStateInput{Position: 65 * time.Second, Playing: true, Highlight: 1}
		}},
	} {
		input := meetingFixturePageInput()
		scenario.mutate(&input)
		scenarios = append(scenarios, meetingFixtureScenario{name: scenario.name, kind: "page", page: input})
	}
	scenarios = append(scenarios, historyFixtureScenarios()...)
	scenarios = append(scenarios, settingsFixtureScenarios()...)
	return scenarios
}

func meetingFixtureSnapshot() meeting.Snapshot {
	return meeting.Snapshot{
		ID: "fixture-live", Mode: meeting.ModeMeeting, State: meeting.StateRecording,
		Started: meetingFixtureNow.Add(-time.Hour), Elapsed: 65*time.Minute + 2*time.Second,
		Levels: map[meeting.Source][]float64{
			meeting.SourceMic: {0.1, 0.4, 0.8}, meeting.SourceSystem: {0.2, 0.5, 0.7},
		},
		Live: map[meeting.Source]string{meeting.SourceMic: "still listening", meeting.SourceSystem: "system live"},
		Segments: []meeting.Segment{
			{ID: "segment-mic", Source: meeting.SourceMic, Speaker: "me", Start: time.Second, End: 2 * time.Second, Text: "First part", Final: true, StoreIdx: 0},
			{ID: "segment-system", Source: meeting.SourceSystem, Speaker: "s1", Start: 3 * time.Second, End: 4 * time.Second, Text: "A complete answer.", Final: true, StoreIdx: 1},
		},
	}
}

func meetingFixturePageInput() meetingsViewStateInput {
	return meetingsViewStateInput{
		List: meetingFixtureStoredMeetings(), Read: true, Query: "", Page: 0, HasMore: true,
		Active: true, SelectedID: "meeting-done", Live: meetingFixtureSnapshot(), Now: meetingFixtureNow,
	}
}

func meetingFixtureStoredMeetings() []store.Meeting {
	statuses := []store.MeetingStatus{
		store.MeetingRecording, store.MeetingInterrupted, store.MeetingAsking,
		store.MeetingFinishing, store.MeetingDone, store.MeetingFailed,
	}
	modes := []store.MeetingMode{store.MeetingModeMeeting, store.MeetingModeSubtitles, store.MeetingModeRecording}
	var meetings []store.Meeting
	for index, status := range statuses {
		id := "meeting-" + string(status)
		title := ""
		if index%2 == 0 {
			title = "Fixture " + string(status)
		}
		meetings = append(meetings, store.Meeting{
			ID: id, CreatedAt: meetingFixtureNow.Add(-time.Duration(index) * 24 * time.Hour),
			Status: status, Mode: modes[index%len(modes)], Title: title, DurationMS: int64(index+1) * 60_000,
		})
	}
	return meetings
}

func meetingFixtureDetailMeeting() store.Meeting {
	summary, _ := (meeting.Summary{
		Title: "Fixture summary", Body: "A generated summary.\nSecond line.", Todos: []string{"Follow up"},
		GeneratedAt: meetingFixtureNow,
	}).EncodeJSON()
	chat, _ := meeting.EncodeChatMessages([]meeting.ChatMessage{{Role: "user", Content: "What happened?"}, {Role: "assistant", Content: "A generated answer."}})
	return store.Meeting{
		ID: "detail-1", CreatedAt: meetingFixtureNow.Add(-time.Hour), Status: store.MeetingDone,
		Mode: store.MeetingModeMeeting, Title: "Detailed meeting", DurationMS: 3_900_000,
		Speakers: `{"s1":"Alice"}`, Summary: summary, Chat: chat, Audio: "detail-audio",
	}
}

func meetingFixtureDetailInput() *meetingDetailStateInput {
	chat, _ := meeting.ParseChatMessages(meetingFixtureDetailMeeting().Chat)
	return &meetingDetailStateInput{
		Meeting: meetingFixtureDetailMeeting(), Chat: chat, CanUseLLM: true, AudioReady: true,
		Segments: []store.MeetingSegment{
			{Idx: 0, Source: store.SegmentMic, Speaker: "me", StartMS: 1000, EndMS: 2000, Text: "Microphone text", Translation: "麦克风文本", Final: true},
			{Idx: 1, Source: store.SegmentSystem, Speaker: "s1", StartMS: 3000, EndMS: 4000, Text: "System text", Final: true},
		},
		Playback: meetingPlaybackStateInput{Highlight: -1},
	}
}
