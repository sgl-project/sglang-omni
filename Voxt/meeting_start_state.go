package main

import (
	"errors"

	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/local"
	"github.com/sgl-project/sglang-omni/Voxt/internal/meeting"
)

type meetingStartViewState struct {
	Code string `json:"code"`
	Text string `json:"text"`
}

func buildMeetingStartViewState(err error) meetingStartViewState {
	var startError *meeting.StartError
	switch {
	case errors.Is(err, local.ErrNotInstalled):
		return meetingStartViewState{Code: "meeting.model.not_downloaded", Text: i18n.T("The speech model for meetings is not downloaded. Download it in Settings → Meetings.")}
	case errors.Is(err, meeting.ErrBusy):
		return meetingStartViewState{Code: "meeting.busy", Text: i18n.T("A meeting is already recording.")}
	case errors.As(err, &startError) && startError.Guidance == meeting.GuidanceModel:
		return meetingStartViewState{Code: meeting.GuidanceModel, Text: i18n.T("Could not load the speech model for meetings: %v", startError.Err)}
	case errors.As(err, &startError) && startError.Source == meeting.SourceSystem:
		code := startError.Guidance
		if code == "" {
			code = meeting.GuidanceSystem
		}
		return meetingStartViewState{Code: code, Text: i18n.T("Could not record system audio. Allow Voxt in System Settings → Privacy & Security → Screen & System Audio Recording, or record the microphone only.")}
	case errors.As(err, &startError):
		code := startError.Guidance
		if code == "" {
			code = meeting.GuidanceMic
		}
		return meetingStartViewState{Code: code, Text: i18n.T("Could not record the microphone. Allow Voxt in System Settings → Privacy & Security → Microphone.")}
	default:
		return meetingStartViewState{Code: "meeting.start.failed", Text: i18n.T("Could not start the meeting: %v", err)}
	}
}

// meetingStartProblem is what the user reads when a meeting does not start.
func meetingStartProblem(err error) string {
	return buildMeetingStartViewState(err).Text
}
