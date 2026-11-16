package main

import (
	"fmt"
	"strings"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/meeting"
	"github.com/sgl-project/sglang-omni/Voxt/internal/uitext"
)

type meetingModeOptionState struct {
	ID       string `json:"id"`
	Title    string `json:"title"`
	Hint     string `json:"hint"`
	Selected bool   `json:"selected"`
}

type meetingMoreOptionState struct {
	ID       string `json:"id"`
	Label    string `json:"label"`
	Selected bool   `json:"selected"`
	Danger   bool   `json:"danger"`
}

type meetingOverlayViewState struct {
	Visible           bool                          `json:"visible"`
	ID                string                        `json:"id"`
	State             string                        `json:"state"`
	Stage             string                        `json:"stage"`
	Mode              string                        `json:"mode"`
	Progress          float64                       `json:"progress"`
	ShowBar           bool                          `json:"show_bar"`
	ShowExpanded      bool                          `json:"show_expanded"`
	ShowAsking        bool                          `json:"show_asking"`
	ShowFinishing     bool                          `json:"show_finishing"`
	Live              bool                          `json:"live"`
	Paused            bool                          `json:"paused"`
	Starting          bool                          `json:"starting"`
	StatusText        string                        `json:"status_text"`
	ElapsedText       string                        `json:"elapsed_text"`
	ModeTitle         string                        `json:"mode_title"`
	ModeHint          string                        `json:"mode_hint"`
	FinishingText     string                        `json:"finishing_text"`
	ProblemKey        string                        `json:"problem_key"`
	ProblemText       string                        `json:"problem_text"`
	ShowMicrophone    bool                          `json:"show_microphone"`
	ShowSystem        bool                          `json:"show_system"`
	MicrophoneLevels  []float64                     `json:"microphone_levels"`
	SystemLevels      []float64                     `json:"system_levels"`
	Sources           []string                      `json:"sources"`
	Rows              []meetingOverlayTranscriptRow `json:"rows"`
	Modes             []meetingModeOptionState      `json:"modes"`
	MoreItems         []meetingMoreOptionState      `json:"more_items"`
	TranslationActive bool                          `json:"translation_active"`
	TranslationLabel  string                        `json:"translation_label"`
	CardSize          int                           `json:"card_size"`
}

func buildMeetingOverlayViewState(snapshot meeting.Snapshot, collapsed bool, cardSize int) meetingOverlayViewState {
	asking := snapshot.State == meeting.StateAsking
	finishing := snapshot.State == meeting.StateFinishing
	live := snapshot.State == meeting.StateRecording || snapshot.State == meeting.StatePaused
	paused := snapshot.State == meeting.StatePaused
	state := meetingOverlayViewState{
		Visible:           snapshot.State.Active(),
		ID:                snapshot.ID,
		State:             string(snapshot.State),
		Stage:             string(snapshot.Stage),
		Mode:              string(snapshot.Mode),
		Progress:          snapshot.Progress,
		ShowBar:           live || snapshot.State == meeting.StateStarting || finishing,
		ShowExpanded:      !collapsed && !asking && !finishing && snapshot.State.Active(),
		ShowAsking:        asking,
		ShowFinishing:     finishing,
		Live:              live,
		Paused:            paused,
		Starting:          snapshot.State == meeting.StateStarting,
		StatusText:        i18n.T("Meeting in progress"),
		ElapsedText:       formatMeetingOverlayElapsed(snapshot.Elapsed),
		ModeTitle:         meetingOverlayModeTitle(snapshot.Mode),
		ModeHint:          meetingOverlayModeHint(snapshot.Mode),
		FinishingText:     meetingOverlayFinishingText(snapshot),
		ShowMicrophone:    snapshot.Mode != meeting.ModeSubtitles,
		ShowSystem:        snapshot.Mode != meeting.ModeRecording,
		MicrophoneLevels:  snapshot.Levels[meeting.SourceMic],
		SystemLevels:      snapshot.Levels[meeting.SourceSystem],
		Rows:              meetingOverlayTranscriptRows(snapshot),
		TranslationActive: snapshot.Translate != "",
		CardSize:          cardSize,
	}
	if paused {
		state.StatusText = i18n.T("Meeting paused")
	}
	if snapshot.Problem != nil {
		state.ProblemKey = snapshot.Problem.Key
		state.ProblemText = meetingOverlayStateProblemText(snapshot.Problem.Key)
	}
	for _, source := range []meeting.Source{meeting.SourceMic, meeting.SourceSystem} {
		_, levels := snapshot.Levels[source]
		_, liveText := snapshot.Live[source]
		hasSegment := false
		for _, segment := range snapshot.Segments {
			if segment.Source == source {
				hasSegment = true
				break
			}
		}
		if levels || liveText || hasSegment {
			state.Sources = append(state.Sources, string(source))
		}
	}
	for _, mode := range []meeting.Mode{meeting.ModeMeeting, meeting.ModeSubtitles, meeting.ModeRecording} {
		state.Modes = append(state.Modes, meetingModeOptionState{
			ID: string(mode), Title: meetingOverlayModeTitle(mode), Hint: meetingOverlayModeHint(mode), Selected: snapshot.Mode == mode,
		})
	}
	translationLabel := i18n.T("Live translation")
	if snapshot.Translate != "" {
		translationLabel = i18n.T("Translating to %s", uitext.TargetLanguageLabel(snapshot.Translate))
	}
	state.TranslationLabel = translationLabel
	state.MoreItems = append(state.MoreItems, meetingMoreOptionState{ID: "translate", Label: translationLabel, Selected: snapshot.Translate != ""})
	if paused {
		state.MoreItems = append(state.MoreItems, meetingMoreOptionState{ID: "copy", Label: i18n.T("Copy transcript")})
	}
	state.MoreItems = append(state.MoreItems, meetingMoreOptionState{ID: "cancel", Label: i18n.T("Cancel meeting"), Danger: true})
	return state
}

func meetingOverlayModeTitle(mode meeting.Mode) string {
	switch mode {
	case meeting.ModeSubtitles:
		return i18n.T("Subtitles")
	case meeting.ModeRecording:
		return i18n.T("Recording")
	default:
		return i18n.T("Meeting")
	}
}

func meetingOverlayModeHint(mode meeting.Mode) string {
	switch mode {
	case meeting.ModeSubtitles:
		return i18n.T("System audio only")
	case meeting.ModeRecording:
		return i18n.T("Microphone only")
	default:
		return i18n.T("Microphone and system audio")
	}
}

func meetingOverlayFinishingText(snapshot meeting.Snapshot) string {
	message := i18n.T("Finishing the meeting…")
	switch snapshot.Stage {
	case meeting.StageTranscribing:
		message = i18n.T("Finalising transcript…")
		if snapshot.Progress > 0 {
			message += fmt.Sprintf(" %.0f%%", snapshot.Progress*100)
		}
	case meeting.StageSummary:
		message = i18n.T("Summarising…")
	}
	return message
}

func formatMeetingOverlayElapsed(duration time.Duration) string {
	duration = duration.Round(time.Second)
	if duration < time.Hour {
		return fmt.Sprintf("%d:%02d", int(duration.Minutes()), int(duration.Seconds())%60)
	}
	return fmt.Sprintf("%d:%02d:%02d", int(duration.Hours()), int(duration.Minutes())%60, int(duration.Seconds())%60)
}

// meetingOverlayTranscriptRow is one display row. Key is stable across snapshots.
type meetingOverlayTranscriptRow struct {
	Key    string `json:"key"`
	Who    string `json:"who"`
	Text   string `json:"text"`
	Me     bool   `json:"me"`
	Live   bool   `json:"live"`
	First  bool   `json:"first"`
	Source string `json:"source"`
}

const meetingOverlayJoinGap = 2 * time.Second

func meetingOverlayTranscriptRows(snapshot meeting.Snapshot) []meetingOverlayTranscriptRow {
	var rows []meetingOverlayTranscriptRow
	var lastEnd time.Duration
	add := func(row meetingOverlayTranscriptRow, start time.Duration, joinable bool) {
		if count := len(rows); count > 0 {
			previous := &rows[count-1]
			if previous.Who == row.Who {
				if joinable && !previous.Live && !meetingOverlayEndsSentence(previous.Text) && start-lastEnd <= meetingOverlayJoinGap {
					previous.Text = joinMeetingOverlayText(previous.Text, row.Text)
					return
				}
				row.First = false
				rows = append(rows, row)
				return
			}
		}
		row.First = true
		rows = append(rows, row)
	}
	for _, segment := range snapshot.Segments {
		if strings.TrimSpace(segment.Text) == "" {
			continue
		}
		me := segment.Speaker == "me" || segment.Speaker == "" && segment.Source == meeting.SourceMic
		add(meetingOverlayTranscriptRow{
			Key: segment.ID, Who: meetingOverlaySpeaker(segment.Speaker, segment.Source), Text: segment.Text,
			Me: me, Source: string(segment.Source),
		}, segment.Start, true)
		lastEnd = segment.End
	}
	for _, source := range []meeting.Source{meeting.SourceMic, meeting.SourceSystem} {
		if text := snapshot.Live[source]; text != "" {
			add(meetingOverlayTranscriptRow{
				Key: "live/" + string(source), Who: meetingOverlaySpeaker("", source), Text: text,
				Me: source == meeting.SourceMic, Live: true, Source: string(source),
			}, 0, false)
		}
	}
	return rows
}

func meetingOverlayEndsSentence(text string) bool {
	runes := []rune(strings.TrimRight(text, " \"'”’)）"))
	return len(runes) > 0 && strings.ContainsRune(".?!。？！…", runes[len(runes)-1])
}

func joinMeetingOverlayText(first, second string) string {
	first, second = strings.TrimSpace(first), strings.TrimSpace(second)
	firstRunes, secondRunes := []rune(first), []rune(second)
	if len(firstRunes) == 0 || len(secondRunes) == 0 {
		return first + second
	}
	if firstRunes[len(firstRunes)-1] < 0x2E80 && secondRunes[0] < 0x2E80 {
		return first + " " + second
	}
	return first + second
}

func meetingOverlayStateProblemText(key string) string {
	if key == "" {
		return ""
	}
	switch key {
	case meeting.ProblemKeySystemSilent:
		return i18n.T("If you see no text when others speak, allow Voxt in System Settings → Privacy & Security → Screen & System Audio Recording.")
	case meeting.ProblemKeyLiveBacklog:
		return i18n.T("Live captions are falling behind. The final transcript will catch up after you end the meeting.")
	case meeting.ProblemKeyDiskLow:
		return i18n.T("Disk space is running low. End the meeting soon or free space on this Mac.")
	case meeting.ProblemKeyHelperOld:
		return i18n.T("Recording system audio needs a newer Voxt helper. Update Voxt, or record the microphone only.")
	default:
		if strings.HasPrefix(key, meeting.ProblemKeyTrackDegraded) {
			return i18n.T("One audio track stopped. Voxt switched to a simpler capture mode.")
		}
		return ""
	}
}

func meetingOverlaySpeaker(name string, source meeting.Source) string {
	if name == "" && source == meeting.SourceMic {
		return i18n.T("You")
	}
	return meetingViewSpeakerText(name, nil)
}
