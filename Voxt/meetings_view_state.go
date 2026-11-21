package main

import (
	"encoding/json"
	"strings"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/meeting"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

type meetingRowViewState struct {
	ID           string `json:"id"`
	Title        string `json:"title"`
	Snippet      string `json:"snippet"`
	StatusCode   string `json:"status_code"`
	StatusLabel  string `json:"status_label"`
	ModeCode     string `json:"mode_code"`
	ClockText    string `json:"clock_text"`
	DurationText string `json:"duration_text"`
	Selected     bool   `json:"selected"`
}

type meetingItemViewState struct {
	Key     string               `json:"key"`
	Heading string               `json:"heading"`
	Meeting *meetingRowViewState `json:"meeting,omitempty"`
}

type meetingSummaryViewState struct {
	Present bool     `json:"present"`
	Title   string   `json:"title"`
	Body    string   `json:"body"`
	Todos   []string `json:"todos"`
}

type meetingChatViewState struct {
	User bool   `json:"user"`
	Text string `json:"text"`
}

type meetingSegmentViewState struct {
	Key         string `json:"key"`
	Index       int    `json:"index"`
	StartMS     int64  `json:"start_ms"`
	TimeText    string `json:"time_text"`
	Speaker     string `json:"speaker"`
	Me          bool   `json:"me"`
	Text        string `json:"text"`
	Translation string `json:"translation"`
	Source      string `json:"source"`
}

type meetingPlaybackViewState struct {
	Available    bool   `json:"available"`
	Playing      bool   `json:"playing"`
	PositionText string `json:"position_text"`
	Highlight    int    `json:"highlight"`
}

const meetingLiveSegmentCap = 200

type meetingDetailViewState struct {
	ID             string                    `json:"id"`
	Title          string                    `json:"title"`
	Meta           string                    `json:"meta"`
	Loading        bool                      `json:"loading"`
	Error          string                    `json:"error"`
	Correcting     bool                      `json:"correcting"`
	Answering      bool                      `json:"answering"`
	CanUseLLM      bool                      `json:"can_use_llm"`
	ShowMore       bool                      `json:"show_more"`
	CanReprocess   bool                      `json:"can_reprocess"`
	CanCorrect     bool                      `json:"can_correct"`
	CanRevealAudio bool                      `json:"can_reveal_audio"`
	CanContinue    bool                      `json:"can_continue"`
	DeleteDisabled bool                      `json:"delete_disabled"`
	HasTranscript  bool                      `json:"has_transcript"`
	SpeakerHelp    bool                      `json:"speaker_help,omitempty"`
	Summary        meetingSummaryViewState   `json:"summary"`
	Chat           []meetingChatViewState    `json:"chat"`
	Segments       []meetingSegmentViewState `json:"segments"`
	Playback       meetingPlaybackViewState  `json:"playback"`
}

type meetingsViewState struct {
	Read              bool                    `json:"read"`
	Loading           bool                    `json:"loading"`
	Error             string                  `json:"error"`
	Query             string                  `json:"query"`
	Page              int                     `json:"page"`
	HasMore           bool                    `json:"has_more"`
	Active            bool                    `json:"active"`
	SettingsAvailable bool                    `json:"settings_available"`
	MeetingCount      int                     `json:"meeting_count"`
	Items             []meetingItemViewState  `json:"items"`
	Detail            *meetingDetailViewState `json:"detail,omitempty"`
}

type meetingPlaybackStateInput struct {
	Position  time.Duration
	Playing   bool
	Highlight int
}

type meetingDetailStateInput struct {
	Meeting       store.Meeting
	Segments      []store.MeetingSegment
	Chat          []meeting.ChatMessage
	Loading       bool
	Answering     bool
	Correcting    bool
	Error         string
	CanUseLLM     bool
	MeetingActive bool
	AudioReady    bool
	MTDInstalled  bool // the diarizer that labels the other side as Speaker 1–4
	Playback      meetingPlaybackStateInput
}

type meetingsViewStateInput struct {
	List       []store.Meeting
	Lines      map[string]string
	Read       bool
	Loading    bool
	Error      string
	Query      string
	Page       int
	HasMore    bool
	Active     bool
	SelectedID string
	Live       meeting.Snapshot
	Now        time.Time
	Detail     *meetingDetailStateInput
}

func buildMeetingsViewState(translator *i18n.Translator, input meetingsViewStateInput) meetingsViewState {
	state := meetingsViewState{
		Read: input.Read, Loading: input.Loading, Error: input.Error, Query: input.Query,
		Page: input.Page, HasMore: input.HasMore, Active: input.Active, SettingsAvailable: true, MeetingCount: len(input.List),
	}
	lastGroup := ""
	for index := range input.List {
		stored := input.List[index]
		group := meetingGroup(translator, stored.CreatedAt, input.Now)
		if group != lastGroup {
			state.Items = append(state.Items, meetingItemViewState{Key: "head/" + group, Heading: group})
			lastGroup = group
		}
		snippet := input.Lines[stored.ID]
		if snippet == "" {
			snippet = meetingSnippet(translator, stored)
		}
		milliseconds := stored.DurationMS
		if stored.ID == input.Live.ID && input.Live.State.Active() {
			milliseconds = input.Live.Elapsed.Milliseconds()
		}
		duration := ""
		if milliseconds > 0 {
			duration = meetingDuration(translator, milliseconds)
		}
		row := &meetingRowViewState{
			ID: stored.ID, Title: meetingDisplayTitle(translator, stored), Snippet: snippet,
			StatusCode: string(stored.Status), StatusLabel: meetingStatusLabel(translator, stored.Status),
			ModeCode: string(stored.Mode), ClockText: stored.CreatedAt.Format("15:04"),
			DurationText: duration, Selected: input.SelectedID == stored.ID,
		}
		state.Items = append(state.Items, meetingItemViewState{Key: stored.ID, Meeting: row})
	}
	if input.Detail != nil {
		state.Detail = buildMeetingDetailViewState(translator, *input.Detail)
	}
	return state
}

func buildMeetingDetailViewState(translator *i18n.Translator, input meetingDetailStateInput) *meetingDetailViewState {
	stored := input.Meeting
	state := &meetingDetailViewState{
		ID: stored.ID, Title: meetingDisplayTitle(translator, stored), Meta: meetingMeta(translator, stored),
		Loading: input.Loading, Error: input.Error, Correcting: input.Correcting,
		Answering: input.Answering, CanUseLLM: input.CanUseLLM,
		ShowMore:       stored.Status == store.MeetingDone || stored.Audio != "",
		CanReprocess:   stored.Status == store.MeetingDone,
		CanCorrect:     stored.Status == store.MeetingDone && input.CanUseLLM && !input.Correcting,
		CanRevealAudio: stored.Audio != "",
		CanContinue:    stored.Status == store.MeetingInterrupted || stored.Status == store.MeetingFinishing,
		DeleteDisabled: input.MeetingActive && stored.Status == store.MeetingRecording,
		HasTranscript:  len(input.Segments) > 0,
		SpeakerHelp:    stored.Status == store.MeetingDone && stored.Mode == store.MeetingModeMeeting && !input.MTDInstalled && !meetingHasNumberedSpeaker(input.Segments),
		Playback: meetingPlaybackViewState{
			Available: input.AudioReady, Playing: input.Playback.Playing, Highlight: input.Playback.Highlight,
		},
	}
	if input.Playback.Position > 0 || input.Playback.Playing {
		state.Playback.PositionText = meeting.FormatPlaybackTime(input.Playback.Position)
	}
	if summary, ok := meeting.ParseStoredSummary(stored.Summary); ok {
		state.Summary = meetingSummaryViewState{Present: true, Title: summary.Title, Body: summary.Body, Todos: summary.Todos}
	}
	for _, message := range input.Chat {
		state.Chat = append(state.Chat, meetingChatViewState{User: message.Role == "user", Text: message.Content})
	}
	names := meetingViewSpeakerNameMap(stored.Speakers)
	state.Segments = make([]meetingSegmentViewState, 0, len(input.Segments)+meetingLiveSegmentCap)
	for _, segment := range input.Segments {
		state.Segments = append(state.Segments, meetingSegmentViewStateFromStore(translator, segment, names))
	}
	return state
}

func meetingSegmentViewStateFromStore(translator *i18n.Translator, segment store.MeetingSegment, names map[string]string) meetingSegmentViewState {
	start := time.Duration(segment.StartMS) * time.Millisecond
	return meetingSegmentViewState{
		Key: "seg/" + itoa(segment.Idx), Index: segment.Idx, StartMS: segment.StartMS,
		TimeText: meeting.FormatPlaybackTime(start), Speaker: meetingViewSpeakerText(translator, segment.Speaker, names),
		Me: segment.Speaker == "me", Text: segment.Text, Translation: segment.Translation, Source: string(segment.Source),
	}
}

func meetingTitle(stored store.Meeting) string {
	if stored.Title != "" {
		return stored.Title
	}
	return stored.CreatedAt.Format("2006-01-02 15:04")
}

func meetingDisplayTitle(translator *i18n.Translator, stored store.Meeting) string {
	if stored.Title != "" {
		return stored.Title
	}
	return translator.T("Untitled meeting")
}

func meetingStatusLabel(translator *i18n.Translator, status store.MeetingStatus) string {
	switch status {
	case store.MeetingRecording, store.MeetingAsking:
		return translator.T("In progress")
	case store.MeetingInterrupted:
		return translator.T("Interrupted")
	case store.MeetingFinishing:
		return translator.T("Finishing")
	case store.MeetingFailed:
		return translator.T("Failed")
	default:
		return ""
	}
}

func meetingMeta(translator *i18n.Translator, stored store.Meeting) string {
	parts := []string{stored.CreatedAt.Format("2006-01-02 15:04")}
	if stored.DurationMS > 0 {
		parts = append(parts, meetingDuration(translator, stored.DurationMS))
	}
	switch stored.Mode {
	case store.MeetingModeSubtitles:
		parts = append(parts, translator.T("System audio only"))
	case store.MeetingModeRecording:
		parts = append(parts, translator.T("Microphone only"))
	case store.MeetingModeImport:
		parts = append(parts, translator.T("Imported recording"))
	default:
		parts = append(parts, translator.T("Microphone and system audio"))
	}
	return strings.Join(parts, " · ")
}

func meetingDuration(translator *i18n.Translator, milliseconds int64) string {
	minutes := int((time.Duration(milliseconds)*time.Millisecond + 30*time.Second) / time.Minute)
	switch {
	case minutes < 1:
		return translator.T("Under a minute")
	case minutes < 60:
		return translator.T("%d min", minutes)
	default:
		return translator.T("%d h %d min", minutes/60, minutes%60)
	}
}

func meetingGroup(translator *i18n.Translator, at, now time.Time) string {
	year, month, day := now.Date()
	today := time.Date(year, month, day, 0, 0, 0, 0, now.Location())
	switch {
	case !at.Before(today):
		return translator.T("Today")
	case !at.Before(today.AddDate(0, 0, -1)):
		return translator.T("Yesterday")
	case !at.Before(today.AddDate(0, 0, -7)):
		return translator.T("Previous 7 days")
	default:
		return at.Format("2006-01")
	}
}

func meetingSnippet(translator *i18n.Translator, stored store.Meeting) string {
	if summary, ok := meeting.ParseStoredSummary(stored.Summary); ok {
		if line, _, _ := strings.Cut(strings.TrimSpace(summary.Body), "\n"); line != "" {
			return strings.TrimLeft(line, "-*# ")
		}
	}
	switch stored.Mode {
	case store.MeetingModeSubtitles:
		return translator.T("System audio only")
	case store.MeetingModeRecording:
		return translator.T("Microphone only")
	case store.MeetingModeImport:
		return translator.T("Imported recording")
	default:
		return translator.T("Microphone and system audio")
	}
}

func meetingViewSpeakerNameMap(raw string) map[string]string {
	names := map[string]string{}
	_ = json.Unmarshal([]byte(raw), &names)
	return names
}

func meetingViewSpeakerText(translator *i18n.Translator, speaker string, names map[string]string) string {
	label := meeting.SpeakerLabel(speaker, names)
	switch label {
	case "Me":
		return translator.T("You")
	case "Them":
		return translator.T("Others")
	case "Speaker " + speaker[1:]:
		return translator.T("Speaker %s", speaker[1:])
	default:
		return label
	}
}

// meetingHasNumberedSpeaker reports whether a diarizer already labelled
// the other side (speakers s1–s4).
func meetingHasNumberedSpeaker(segments []store.MeetingSegment) bool {
	for _, seg := range segments {
		if len(seg.Speaker) == 2 && seg.Speaker[0] == 's' && seg.Speaker[1] >= '1' && seg.Speaker[1] <= '4' {
			return true
		}
	}
	return false
}
