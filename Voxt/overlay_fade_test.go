package main

import (
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/overlay/layout"
	"github.com/sgl-project/sglang-omni/Voxt/internal/session"
)

// TestOverlayFadeKeepsLastFrame: while the overlay fades out, it draws what
// it showed last, not the ended session's empty pill (the live text gone,
// the microphone row alone in the wide face).
func TestOverlayFadeKeepsLastFrame(t *testing.T) {
	a, _, _ := newTestApp(t)
	a.prefs.Overlay.LiveLines = true
	ft := new(fakeTimers).install(a)
	snap(a, session.Snapshot{ID: "s1", State: session.Recording, Partial: "the nature of things"})
	if got := a.ov.projection.view.Pill.LiveText; got == "" {
		t.Fatalf("recording: no live text: %+v", a.ov.projection.view.Pill)
	}
	shown := a.ov.projection.view
	w, h := a.ov.w, a.ov.h
	snap(a, session.Snapshot{ID: "s1", State: session.Done, Result: "The nature of things.", Inserted: true})
	for a.ov.visible {
		a.refreshSessionOverlay(time.Now())
		if v := a.ov.projection.view; v.Mode != ovHidden || v.State != string(session.Done) {
			t.Fatalf("fading: projection %s %s, want it to follow the session", v.Mode, v.State)
		}
		if v := a.ov.projection.drawn(); v.Mode != shown.Mode || v.Pill.LiveText != shown.Pill.LiveText || a.ov.w != w || a.ov.h != h {
			t.Fatalf("fading: draws %s %q at %dx%d, showed %s %q at %dx%d", v.Mode, v.Pill.LiveText, a.ov.w, a.ov.h, shown.Mode, shown.Pill.LiveText, w, h)
		}
		tester := hudTester(a.overlayView, a.ov.w+2*layout.Pad, a.ov.h+2*layout.Pad)
		frames(tester)
		if !tester.HasText("the nature of things") {
			t.Fatal("fading: the frame lost its live text")
		}
		ft.advance(layout.FadeStep)
	}
	snap(a, session.Snapshot{ID: "s2", State: session.Recording})
	if v := a.ov.projection.view; v.Mode != ovRecording || v.SessionID != "s2" || v.Pill.LiveText != "" {
		t.Fatalf("next session: %s %s %q", v.Mode, v.SessionID, v.Pill.LiveText)
	}
}

// TestPillFadeKeepsLastFrame: the pill's own window (a Transcribe beside an
// open answer card) also fades out on its last frame, not on the ended
// session's empty pill.
func TestPillFadeKeepsLastFrame(t *testing.T) {
	chat := &fakeChat{texts: []string{"one", "two"}}
	a, tr, _ := rewriteSelecting(t, chat, chatA, "summarize")
	a.prefs.Overlay.LiveLines = true
	startSaying(a, tr, "transcribe/0", textEditApp, "dictated")
	a.refreshPillOverlay(time.Now(), a.pill.mode)
	shown := a.pill.projection.view
	if a.pill.mode != ovRecording || shown.Pill.Title == "" {
		t.Fatalf("recording: pill %s %+v", a.pill.mode, shown.Pill)
	}
	tap(a, "transcribe/0", textEditApp)
	a.pump("pill fading", func() bool { return !a.pill.shown })
	if !a.pill.visible {
		t.Fatal("the pill hid without fading")
	}
	a.refreshPillOverlay(time.Now(), a.pill.mode)
	if v := a.pill.projection.drawn(); v.Mode == ovHidden || v.SessionID != shown.SessionID || v.Pill.Title != shown.Pill.Title {
		t.Fatalf("fading: draws %s %s %q, showed %s %s %q", v.Mode, v.SessionID, v.Pill.Title, shown.Mode, shown.SessionID, shown.Pill.Title)
	}
	if v := a.pill.projection.view; v.Mode != ovHidden {
		t.Fatalf("projection: %s, want it to follow the session", v.Mode)
	}
}
