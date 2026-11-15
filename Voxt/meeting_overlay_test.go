package main

import (
	"fmt"
	"strings"
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

// The live transcript stays at the newest line while it shows the end, and
// keeps the user's place once they scroll up (#3985).
func TestMeetingOverlayFollowsNewestLine(t *testing.T) {
	a, _, _ := newTestApp(t)
	lineText := func(i int) string {
		return fmt.Sprintf("Line %d: 我们先看 the release notes，再讨论下一步。", i)
	}
	s := meeting.Snapshot{ID: "m1", State: meeting.StateRecording, Mode: meeting.ModeMeeting}
	next := 0 // the snapshot keeps the newest 200 lines, as meeting.Manager does
	add := func(n int) {
		for range n {
			i := next
			next++
			s.Segments = append(s.Segments, meeting.Segment{ID: fmt.Sprint("seg-", i), Source: meeting.SourceSystem, Speaker: "s1",
				Text: lineText(i), Final: true})
			if len(s.Segments) > 200 {
				s.Segments = s.Segments[1:]
			}
		}
		a.meetSnap = s
	}
	a.meetOvCollapsed = false
	m := overlayModel(s, false)
	tt := ui.NewTester(a.meetingOverlayView, m.Layout.Width, m.Layout.Height)
	shows := func(text string) bool {
		r, ok := tt.Find(text)
		return ok && r.H > 0 && r.Y >= 0 && r.Y+r.H <= float32(m.Layout.Height)
	}
	newest := func() string { return lineText(next - 1) }
	a.meetSnap = s
	frames(tt)
	if !shows("Listening…") {
		t.Fatal("no Listening… before any text")
	}
	add(3)
	frames(tt)
	for range 3 { // one line at a time, then a burst, as a long call
		add(1)
		frames(tt)
		if !shows(newest()) {
			t.Fatalf("after %d lines: %q not in view", next, newest())
		}
	}
	add(400)
	frames(tt)
	if !shows(newest()) {
		t.Fatalf("after a burst to %d lines: %q not in view", next, newest())
	}
	// Scrolled up to read, new lines leave the view where it is.
	cx, cy := float32(m.Layout.Width)/2, float32(m.Layout.Height)*0.7
	tt.Scroll(cx, cy, 0, -600)
	frames(tt)
	if shows(newest()) {
		t.Fatal("still at the end after scrolling up")
	}
	reading := ""
	for _, txt := range tt.Texts() {
		if strings.HasPrefix(txt, "Line ") && shows(txt) {
			reading = txt
			break
		}
	}
	add(5) // the oldest lines leave the snapshot as these come
	frames(tt)
	if reading == "" || !shows(reading) || shows(newest()) {
		t.Fatalf("scrolled up: reading %q, shows it %v, shows newest %v", reading, shows(reading), shows(newest()))
	}
	// Back at the end, it follows again, the sentence still heard too.
	tt.Scroll(cx, cy, 0, 100000)
	frames(tt)
	add(1)
	s.Live = map[meeting.Source]string{meeting.SourceMic: "我还在说 the next sentence"}
	a.meetSnap = s
	frames(tt)
	if !shows(newest()) || !shows(s.Live[meeting.SourceMic]) {
		t.Fatalf("back at the end: %q or the live line not in view", newest())
	}
	// Collapsed while more is said, it opens again at the end.
	a.meetOvCollapsed = true
	frames(tt)
	s.Live = nil
	add(50)
	a.meetOvCollapsed = false
	frames(tt)
	if !shows(newest()) {
		t.Fatalf("expanded again: %q not in view", newest())
	}
	// Another meeting starts at its own end, whatever the last one's place.
	tt.Scroll(cx, cy, 0, -600)
	frames(tt)
	s, next = meeting.Snapshot{ID: "m2", State: meeting.StateRecording, Mode: meeting.ModeMeeting}, 0
	add(30)
	frames(tt)
	if !shows(newest()) {
		t.Fatalf("next meeting: %q not in view", newest())
	}
}
