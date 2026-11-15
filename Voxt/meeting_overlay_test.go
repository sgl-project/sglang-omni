package main

import (
	"testing"

	"github.com/egoist/mygo"
	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/meeting"
	"github.com/sgl-project/sglang-omni/Voxt/internal/overlay/layout"
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

// TestMeetingOverlayFrame: the overlay first shows at the bottom centre,
// then stays where the user dragged it as it grows or shrinks (#3984),
// kept inside the work area of the screen it is on.
func TestMeetingOverlayFrame(t *testing.T) {
	area := mygo.Rectangle{X: 0, Y: 25, Width: 1280, Height: 775}
	if f := meetingOverlayFrame(mygo.Rectangle{}, area, 400, 60); f != (mygo.Rectangle{X: 440, Y: 25 + 775 - 60 - 28, Width: 400, Height: 60}) {
		t.Errorf("first show: %+v", f)
	}
	dragged := mygo.Rectangle{X: 100, Y: 200, Width: 400, Height: 60}
	if f := meetingOverlayFrame(dragged, area, 400, 60); f != dragged {
		t.Errorf("an update after a drag moved it: %+v, want %+v", f, dragged)
	}
	if f := meetingOverlayFrame(dragged, area, 560, 420); f.X != 100 || f.Y != 200 || f.Width != 560 || f.Height != 420 {
		t.Errorf("expanded after a drag: %+v", f)
	}
	low := mygo.Rectangle{X: 1000, Y: 730, Width: 400, Height: 60}
	if f := meetingOverlayFrame(low, area, 560, 420); f.X != 1280-560+layout.Pad || f.Y != 25+775-420+layout.Pad {
		t.Errorf("expanded by the bottom-right corner: %+v, want the face inside %+v", f, area)
	}
	edge := mygo.Rectangle{X: -layout.Pad, Y: 25 - layout.Pad, Width: 400, Height: 60} // the face flush with the top-left corner
	if f := meetingOverlayFrame(edge, area, 400, 60); f != edge {
		t.Errorf("dragged flush with the corner, then updated: %+v, want %+v", f, edge)
	}
	other := mygo.Rectangle{X: -1920, Y: 0, Width: 1920, Height: 1080} // a screen left of the main one
	at := mygo.Rectangle{X: -1500, Y: 300, Width: 400, Height: 60}
	if f := meetingOverlayFrame(at, other, 400, 60); f != at {
		t.Errorf("on the other screen: %+v, want %+v", f, at)
	}
}
