package main

import (
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/session"
)

// TestShortNotices: Nothing was heard leaves after noSpeechTime, through
// the fade, a green check after okTime; other notices keep noticeTime.
func TestShortNotices(t *testing.T) {
	a, _, _ := newTestApp(t)
	ft := new(fakeTimers).install(a)
	snap(a, session.Snapshot{ID: "s1", State: session.Recording})
	snap(a, session.Snapshot{ID: "s1", State: session.Done, Problem: &session.Problem{Code: session.ProblemNoSpeech}})
	if a.notice == nil || a.notice.hold != noSpeechTime {
		t.Fatalf("no speech: %+v", a.notice)
	}
	ft.advance(noSpeechTime)
	if a.notice != nil || a.ov.shown {
		t.Fatalf("no speech stays after %v: %+v", noSpeechTime, a.notice)
	}
	if !a.ov.visible {
		t.Error("no speech left without the fade")
	}
	ft.advance(ovGone)
	if a.ov.visible {
		t.Error("no speech still on screen after the fade")
	}
	snap(a, session.Snapshot{ID: "s2", State: session.Recording, Note: true})
	snap(a, session.Snapshot{ID: "s2", State: session.Done, Note: true, Result: "A note."})
	if a.notice == nil || !a.notice.ok {
		t.Fatalf("saved: %+v", a.notice)
	}
	ft.advance(okTime - time.Millisecond)
	if a.notice == nil {
		t.Fatal("saved left early")
	}
	ft.advance(time.Millisecond)
	if a.notice != nil {
		t.Fatalf("saved stays after %v", okTime)
	}
	ft.advance(ovGone)
	snap(a, session.Snapshot{ID: "s3", State: session.Recording})
	snap(a, session.Snapshot{ID: "s3", State: session.Failed, Problem: &session.Problem{Code: session.ProblemNoASR}})
	ft.advance(noticeTime - time.Millisecond)
	if a.notice == nil {
		t.Fatal("another notice left before noticeTime")
	}
}

// TestOverlayQuickSessions: what the last session left on its way out
// (the fade, a notice's timer) never hides the next one.
func TestOverlayQuickSessions(t *testing.T) {
	a, _, _ := newTestApp(t)
	ft := new(fakeTimers).install(a)
	snap(a, session.Snapshot{ID: "s1", State: session.Recording})
	snap(a, session.Snapshot{ID: "s1", State: session.Done, Result: "One.", Inserted: true})
	ft.advance(20 * time.Millisecond) // mid-fade
	snap(a, session.Snapshot{ID: "s2", State: session.Recording})
	ft.advance(time.Second)
	if a.ov.mode != ovRecording || !a.ov.shown || !a.ov.visible {
		t.Fatalf("the second recording hidden: %s shown %v visible %v", a.ov.mode, a.ov.shown, a.ov.visible)
	}
	// A notice of the second, then a third session failing: the second's
	// timer leaves the third's notice alone.
	snap(a, session.Snapshot{ID: "s2", State: session.Done, Problem: &session.Problem{Code: session.ProblemNoSpeech}})
	ft.advance(time.Second)
	snap(a, session.Snapshot{ID: "s3", State: session.Recording})
	snap(a, session.Snapshot{ID: "s3", State: session.Failed, Problem: &session.Problem{Code: session.ProblemNoASR}})
	ft.advance(time.Second) // the second's timer is due
	if a.notice == nil || a.notice.action != actionModels || !a.ov.visible {
		t.Fatalf("the third's notice went with the second's timer: %+v", a.notice)
	}
	ft.advance(noticeTime)
	if a.notice != nil || a.ov.visible {
		t.Errorf("the third's notice stays: %+v", a.notice)
	}
}
