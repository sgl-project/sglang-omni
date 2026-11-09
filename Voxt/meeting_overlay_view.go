package main

import (
	"strings"

	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/meeting"
)

const (
	meetingOvCollapsedW = 340
	meetingOvCollapsedH = 48
	meetingOvExpandedW  = 560
	meetingOvExpandedH  = 360
)

// meetingOverlayLayout is the meeting floating window's size.
type meetingOverlayLayout struct {
	Width, Height int
	Collapsed     bool
}

// meetingOverlayModel is the pure view model for the meeting overlay.
type meetingOverlayModel struct {
	Layout       meetingOverlayLayout
	ProblemKey   string
	ShowBar      bool
	ShowExpanded bool
	ShowAsking   bool
	ShowFinishing bool
}

// meetingOverlaySize returns width and height for a collapsed or expanded overlay.
func meetingOverlaySize(collapsed bool) (w, h int) {
	if collapsed {
		return meetingOvCollapsedW, meetingOvCollapsedH
	}
	return meetingOvExpandedW, meetingOvExpandedH
}

// meetingOverlayModel builds the overlay model from the meeting snapshot.
func overlayModel(s meeting.Snapshot, collapsed bool) meetingOverlayModel {
	w, h := meetingOverlaySize(collapsed)
	m := meetingOverlayModel{
		Layout: meetingOverlayLayout{Width: w, Height: h, Collapsed: collapsed},
		ShowBar: s.State == meeting.StateRecording || s.State == meeting.StatePaused ||
			s.State == meeting.StateStarting || s.State == meeting.StateFinishing,
		ShowExpanded: !collapsed && s.State.Active(),
		ShowAsking:   s.State == meeting.StateAsking,
		ShowFinishing: s.State == meeting.StateFinishing,
	}
	if s.Problem != nil {
		m.ProblemKey = s.Problem.Key
	}
	return m
}

func meetingOverlayProblemText(key string) string {
	if key == "" {
		return ""
	}
	switch key {
	case meeting.ProblemKeySystemSilent:
		return i18n.T("If others are speaking but you see no text from them, open System Settings → Privacy & Security → Screen & System Audio Recording and allow Voxt.")
	case meeting.ProblemKeyLiveBacklog:
		return i18n.T("Live captions are falling behind; the final transcript will catch up after you end the meeting.")
	case meeting.ProblemKeyDiskLow:
		return i18n.T("Disk space is running low. End the meeting soon or free space on this Mac.")
	case meeting.ProblemKeyHelperOld:
		return i18n.T("This meeting needs a newer Voxt helper with system audio capture. Update Voxt or record microphone only.")
	default:
		if strings.HasPrefix(key, "meeting.track.degraded.") {
			return i18n.T("One audio track stopped. Voxt switched to a simpler capture mode.")
		}
		return ""
	}
}
