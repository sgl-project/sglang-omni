package main

import (
	"testing"

	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/meeting"
)

// The capsule's controls fit in its window, with a long warning too.
func TestMeetingOverlayControlsFit(t *testing.T) {
	a, _, _ := newTestApp(t)
	for _, problem := range []string{"", meeting.ProblemKeySystemSilent} {
		s := meeting.Snapshot{ID: "m1", State: meeting.StateRecording, Mode: meeting.ModeMeeting}
		if problem != "" {
			s.Problem = &meeting.Problem{Key: problem}
		}
		a.meetSnap, a.meetOvCollapsed = s, true
		m := overlayModel(s, true)
		tt := ui.NewTester(a.meetingOverlayView, m.Layout.Width, m.Layout.Height)
		frames(tt)
		for _, label := range []string{"Pause meeting", "End meeting", "Expand", meetingOverlayProblemText(problem)} {
			if label == "" {
				continue
			}
			r, ok := tt.Find(label)
			if !ok || r.W < 10 || r.H < 10 || r.X < 0 || r.Y < 0 || r.X+r.W > float32(m.Layout.Width) || r.Y+r.H > float32(m.Layout.Height) {
				t.Errorf("problem %q: %q at %+v (found %v) outside %dx%d", problem, label, r, ok, m.Layout.Width, m.Layout.Height)
			}
		}
	}
}

// A start that hangs still offers End, and no Pause.
func TestMeetingOverlayStartingCanEnd(t *testing.T) {
	a, _, _ := newTestApp(t)
	s := meeting.Snapshot{ID: "m1", State: meeting.StateStarting, Mode: meeting.ModeMeeting}
	a.meetSnap, a.meetOvCollapsed = s, true
	m := overlayModel(s, true)
	tt := ui.NewTester(a.meetingOverlayView, m.Layout.Width, m.Layout.Height)
	frames(tt)
	if _, ok := tt.Find("End meeting"); !ok {
		t.Error("no End meeting while starting")
	}
	if _, ok := tt.Find("Pause meeting"); ok {
		t.Error("Pause meeting while starting")
	}
}

func TestMeetingOverlayContentProtection(t *testing.T) {
	a, _, _ := newTestApp(t)
	var got []bool
	t.Cleanup(func() { meetingContentProtectionHook = nil })
	meetingContentProtectionHook = func(on bool) { got = append(got, on) }

	a.prefs.Features.Meeting.HideFromSharing = false
	a.applyMeetingOverlayProtection()
	a.prefs.Features.Meeting.HideFromSharing = true
	a.applyMeetingOverlayProtection()
	if len(got) != 2 || got[0] || !got[1] {
		t.Fatalf("content protection calls %v", got)
	}
}
