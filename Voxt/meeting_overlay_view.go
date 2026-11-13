package main

import (
	"strings"

	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/meeting"
	"github.com/sgl-project/sglang-omni/Voxt/internal/overlay/layout"
)

// The meeting window's sizes in DIPs: the face's plus layout.Pad on each
// side for its shadow.
const (
	meetingOvBarH       = 44
	meetingOvProblemH   = 42 // two lines of the warning and its margin
	meetingOvCollapsedW = 360 + 2*layout.Pad
	meetingOvCollapsedH = meetingOvBarH + 2*layout.Pad
	meetingOvExpandedW  = 520 + 2*layout.Pad
	meetingOvExpandedH  = 340 + 2*layout.Pad
)

// meetingOverlayLayout is the meeting floating window's size.
type meetingOverlayLayout struct {
	Width, Height int
	Collapsed     bool
}

// meetingOverlayModel is the pure view model for the meeting overlay.
type meetingOverlayModel struct {
	Layout        meetingOverlayLayout
	ProblemKey    string
	ShowBar       bool
	ShowExpanded  bool
	ShowAsking    bool
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
// Asking and finishing always show the capsule.
func overlayModel(s meeting.Snapshot, collapsed bool) meetingOverlayModel {
	asking, finishing := s.State == meeting.StateAsking, s.State == meeting.StateFinishing
	pill := collapsed || asking || finishing
	w, h := meetingOverlaySize(pill)
	m := meetingOverlayModel{
		Layout: meetingOverlayLayout{Width: w, Height: h, Collapsed: collapsed},
		ShowBar: s.State == meeting.StateRecording || s.State == meeting.StatePaused ||
			s.State == meeting.StateStarting || finishing,
		ShowExpanded:  !pill && s.State.Active(),
		ShowAsking:    asking,
		ShowFinishing: finishing,
	}
	if s.Problem != nil {
		m.ProblemKey = s.Problem.Key
		if pill && !asking && !finishing && meetingOverlayProblemText(m.ProblemKey) != "" {
			m.Layout.Height += meetingOvProblemH
		}
	}
	return m
}

func meetingOverlayProblemText(key string) string {
	if key == "" {
		return ""
	}
	switch key {
	case meeting.ProblemKeySystemSilent:
		return i18n.T("If you see no text when others speak, allow Voxt in System Settings → Privacy & Security → Screen & System Audio Recording.")
	case meeting.ProblemKeyLiveBacklog:
		return i18n.T("Live captions are falling behind. The final transcript will catch up after you end the meeting.")
	case meeting.ProblemKeyDiskLow:
		return i18n.Platform("Disk space is running low. End the meeting soon or free space on this Mac.", "Disk space is running low. End the meeting soon or free space on this PC.")
	case meeting.ProblemKeyHelperOld:
		return i18n.T("Recording system audio needs a newer Voxt helper. Update Voxt, or record the microphone only.")
	default:
		if strings.HasPrefix(key, "meeting.track.degraded.") {
			return i18n.T("One audio track stopped. Voxt switched to a simpler capture mode.")
		}
		return ""
	}
}
