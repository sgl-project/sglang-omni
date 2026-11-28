package main

import (
	"testing"

	"github.com/sgl-project/sglang-omni/Voxt/internal/meeting"
)

// Pause and Resume show at once: the overlay changes on the press, before
// the manager has stopped or started the tracks (a pause waits for their
// last words, which took seconds on a busy Mac while the button seemed not
// to answer), and a second press meanwhile is not sent.
func TestMeetingPauseShowsAtOnce(t *testing.T) {
	s := meetingOfWords(t, "let us begin the weekly meeting now")
	s.startMeeting()
	s.pump("the meeting", func() bool { return s.meetSnap.State == meeting.StateRecording })
	s.meetingPauseResume()
	if !s.meetOvView.Paused || s.meetOvPending != meeting.StatePaused {
		t.Fatalf("the press did not show: paused %v, pending %q", s.meetOvView.Paused, s.meetOvPending)
	}
	s.meetingPauseResume() // again, before the pause is through: not sent
	s.pump("the pause", func() bool { return s.meetSnap.State == meeting.StatePaused && s.meetOvPending == "" })
	if !s.meetOvView.Paused || len(s.toasts) != 0 {
		t.Fatalf("paused: view %v, toasts %q", s.meetOvView.Paused, s.toasts)
	}
	s.meetingPauseResume()
	if s.meetOvView.Paused {
		t.Fatal("Resume did not show at once")
	}
	s.pump("the resume", func() bool { return s.meetSnap.State == meeting.StateRecording && s.meetOvPending == "" })
	if s.meetOvView.Paused || len(s.toasts) != 0 {
		t.Fatalf("resumed: view paused %v, toasts %q", s.meetOvView.Paused, s.toasts)
	}
	s.meetingDiscard()
	s.pump("discarded", func() bool { return s.meetSnap.State == meeting.StateIdle })
}

// The overlay's buttons act after the frame that saw the click: a resize
// made while that frame built had it drawn for the old size and stretched
// over the new one, a flash on every expand and collapse (11-25).
func TestMeetingOverlayActsAfterItsFrame(t *testing.T) {
	a, _, _ := newTestApp(t)
	var queued []func()
	a.postQueue = func(fn func()) { queued = append(queued, fn) }
	actions := meetingOverlayAfterFrame{a}
	actions.setMeetingOverlayCollapsed(true)
	actions.toggleMeetingMenu(meetingMenuMore)
	actions.requestMeetingFinish()
	if a.meetOvUI.collapsed || a.meetOvUI.menu != meetingMenuNone || a.meetOvUI.endAsk {
		t.Fatalf("changed in the frame of the click: %+v", a.meetOvUI)
	}
	for _, fn := range queued {
		fn()
	}
	if !a.meetOvUI.collapsed || !a.meetOvUI.endAsk {
		t.Fatalf("after the frame: %+v", a.meetOvUI)
	}
}
