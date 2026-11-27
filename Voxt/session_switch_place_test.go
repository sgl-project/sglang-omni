package main

import (
	"strings"
	"testing"

	"github.com/egoist/mygo"
)

// TestSessionSwitchKeepsTheCardInPlace: from the Rewrite card, a new
// session opens in the same card, idle, where it was, and takes a typed
// first instruction; any card a switch shows opens at the same top-left,
// so the session row stays under the pointer, until the user drags it.
// The new session used to show a second, short card at the bottom, and
// every switch between cards of other heights moved the session row.
func TestSessionSwitchKeepsTheCardInPlace(t *testing.T) {
	chat := &fakeChat{texts: []string{"one", "two"}}
	a, _, _ := rewriteSelecting(t, chat, chatA, "summarize")
	a.ov.area, a.box.area = ovTestArea, ovTestArea
	at := mygo.Rectangle{X: 300, Y: 120, Width: boxWinW, Height: boxWinH}
	a.box.frame, a.box.winShown = at, true
	a.newSession()
	if !a.box.open || !a.box.shown || !a.boxIdle() {
		t.Fatalf("the new session left the Rewrite card: open %v, shown %v, idle %v", a.box.open, a.box.shown, a.boxIdle())
	}
	a.box.area = ovTestArea // boxSync's, on a screen
	if f := a.boxFrame(); f.X != at.X || f.Y != at.Y {
		t.Fatalf("the Rewrite card moved from %+v to %+v", at, f)
	}
	if f := a.overlayFrame(ovAnswer, 300, 100); f.X != at.X || f.Y != at.Y {
		t.Fatalf("an answer card opens at %+v, not %+v", f, at)
	}
	if f := a.overlayFrame(ovRecording, 300, 40); f.X == at.X && f.Y == at.Y {
		t.Fatal("the recording pill moved to the card's place")
	}
	if v := a.boxViewState(); !v.CanSpeak || v.CanSend || v.Subtitle != "Say or type what to write" {
		t.Fatalf("the idle card: speak %v, send %v, subtitle %q", v.CanSpeak, v.CanSend, v.Subtitle)
	}
	a.box.text = "write a haiku"
	if !a.boxViewState().CanSend {
		t.Fatal("the typed first instruction cannot be sent")
	}
	a.boxSendNow()
	a.pump("the new session's answer", func() bool { return a.answer.text == "two" && a.sess.State.Ended() })
	if !strings.Contains(lastUser(chat, 1), "write a haiku") || !a.box.open || !a.boxAnswered() {
		t.Fatalf("asked %q, card open %v answered %v", lastUser(chat, 1), a.box.open, a.boxAnswered())
	}
	a.boxDraggedTo(mygo.Rectangle{X: 40, Y: 60, Width: boxWinW, Height: boxWinH})
	if a.slotPin != nil {
		t.Fatal("a drag kept the switch's place")
	}
}
