package main

import (
	"testing"

	"github.com/egoist/mygo"
	"github.com/sgl-project/sglang-omni/Voxt/internal/macx"
	"github.com/sgl-project/sglang-omni/Voxt/internal/overlay/layout"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

// TestRewriteBoxHidesTheOverlayCard: the Rewrite card takes over the
// answer card shown in the overlay window, which fades out. A second sync
// during the fade (the next state change) used to cancel its hide, and
// the overlay window stayed on screen, dimmed, at the old card's size,
// drawing the current session's pill: the translucent empty card.
func TestRewriteBoxHidesTheOverlayCard(t *testing.T) {
	a, _, _ := newTestApp(t)
	timers := (&fakeTimers{}).install(a)
	a.answer.open, a.answer.kind, a.answer.text = true, store.KindRewrite, "the answer"
	a.overlaySync()
	if a.ov.mode != ovAnswer || !a.ov.visible {
		t.Fatalf("the card: mode %s, visible %v", a.ov.mode, a.ov.visible)
	}

	a.ov.native = true
	previous, oldFrame, oldPointer := newBoxNativeWindow, overlayFocusedWindowFrame, overlayPointerDisplay
	newBoxNativeWindow = func(mygo.WindowOptions) boxDropWindow { return &fakeBoxDropWindow{} }
	overlayFocusedWindowFrame = func() (macx.WindowFrame, bool) { return macx.WindowFrame{}, false }
	overlayPointerDisplay = func() mygo.Display { return mygo.Display{WorkArea: ovTestArea} }
	t.Cleanup(func() {
		newBoxNativeWindow, overlayFocusedWindowFrame, overlayPointerDisplay = previous, oldFrame, oldPointer
	})
	a.box = rewriteBox{open: true, shown: true, winShown: true, area: ovTestArea}
	a.overlaySync() // the Rewrite card takes the answer: the overlay fades out
	a.overlaySync() // the next change, while it fades
	timers.advance(3 * layout.FadeStep)
	if a.ov.shown || a.ov.visible {
		t.Fatalf("the overlay window stays on screen behind the Rewrite card: shown %v, visible %v", a.ov.shown, a.ov.visible)
	}
}
