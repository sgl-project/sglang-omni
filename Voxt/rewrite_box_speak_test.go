package main

import (
	"testing"

	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/helper/proto"
	"github.com/sgl-project/sglang-omni/Voxt/internal/overlay/layout"
	"github.com/sgl-project/sglang-omni/Voxt/internal/session"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

// TestRewriteCardDictationKeysSpeakTheNextInstruction: in the Rewrite card
// holding an answer, Transcribe's and Note's shortcuts record the next
// instruction and their next press sends it, as Rewrite's does. They used
// to dictate into the card's input, which waited for Enter, so only the
// first instruction could be spoken.
func TestRewriteCardDictationKeysSpeakTheNextInstruction(t *testing.T) {
	chat := &fakeChat{texts: []string{"one", "two", "three"}}
	a, tr, _ := rewriteSelecting(t, chat, chatA, "summarize")
	self := proto.App{BundleID: selfBundleID, Name: "Voxt"}
	for i, c := range []struct{ binding, said, answer string }{
		{"transcribe/0", "shorter", "two"},
		{"note", "in French", "three"},
	} {
		if !a.boxHoldsCard(ovAnswer) {
			t.Fatalf("turn %d: the answer is not in the Rewrite card", i+2)
		}
		startSaying(a, tr, c.binding, self, c.said)
		if a.sess.Kind != store.KindRewrite || a.followUp != a.answer.id {
			t.Fatalf("%s in the Rewrite card: a %s session, follow-up %q of card %q", c.binding, a.sess.Kind, a.followUp, a.answer.id)
		}
		tap(a, c.binding, self)
		a.pump("the answer", func() bool { return a.sess.State.Ended() && a.answer.text == c.answer })
		if n := len(a.answer.turns); n != i+2 || a.answer.turns[n-1].Instruction != c.said {
			t.Fatalf("%s: turns %+v", c.binding, a.answer.turns)
		}
	}
}

// TestRewriteCardSpeakButton: once an answer is in, the Rewrite card shows
// Speak with the Rewrite shortcut beside it, and Speak records the next
// instruction there. The card showed only the text input, so the next
// instruction could not be spoken without knowing the shortcut.
func TestRewriteCardSpeakButton(t *testing.T) {
	chat := &fakeChat{texts: []string{"one", "two"}}
	a, tr, _ := rewriteSelecting(t, chat, chatA, "summarize")
	v := a.boxViewState()
	if !v.CanSpeak || v.SpeakKeys == "" {
		t.Fatalf("the Rewrite card with an answer: speak %v, keys %q", v.CanSpeak, v.SpeakKeys)
	}
	tr.setText("shorter")
	a.boxActions().speak()
	a.pump("recording", func() bool { return a.sess.State == session.Recording && a.followUp == a.answer.id })
	if v := a.boxViewState(); v.CanSpeak || !v.CanSend {
		t.Fatalf("recording: speak %v, send %v", v.CanSpeak, v.CanSend)
	}
	a.pump("audio", func() bool { return len(a.sess.Levels) >= 5 })
	a.boxActions().send()
	a.pump("the answer", func() bool { return a.sess.State.Ended() && a.answer.text == "two" })
	if n := len(a.answer.turns); n != 2 || a.answer.turns[1].Instruction != "shorter" {
		t.Fatalf("turns %+v", a.answer.turns)
	}
}

// TestRewriteCardKeepsOneLayout: the Rewrite card is one chat face from
// the first instruction to the last. The header always holds Insert, Copy
// and Close, and the composer at the foot always holds Add, the
// microphone and Send, each in the same place, inside the card, and the
// size button sits in the header, clear of Send. Send and
// the live wave used to move into the header while it listened, Insert
// and Copy came and went, the foot's key hints and Speak overflowed, and
// Send ran into the size button in the corner.
func TestRewriteCardKeepsOneLayout(t *testing.T) {
	type place struct{ add, mic, send, insert, copy, close ui.Rect }
	measure := func(t *testing.T, view func(*ui.Context), mic string) place {
		t.Helper()
		tt := hudTester(view, boxWinW, boxWinH)
		frames(tt)
		var p place
		for _, f := range []struct {
			label string
			at    *ui.Rect
		}{{"Add", &p.add}, {mic, &p.mic}, {"Send", &p.send}, {"Insert", &p.insert}, {"Copy", &p.copy}, {"Close", &p.close}} {
			r, ok := tt.Find(f.label)
			if !ok {
				t.Fatalf("%q is missing from %q", f.label, firstTexts(tt))
			}
			if r.X+r.W > boxWinW-layout.Pad {
				t.Fatalf("%q runs out of the card: %+v", f.label, r)
			}
			*f.at = r
		}
		if size, ok := tt.Find("Make the card larger"); !ok || size.Y != p.close.Y || size.X+size.W > p.close.X {
			t.Fatalf("the size button %+v is not in the header before Close %+v", size, p.close)
		}
		for _, gone := range []string{"Enter", "Esc"} {
			if hasExact(tt, gone) {
				t.Fatalf("the foot still spells out %q: %q", gone, firstTexts(tt))
			}
		}
		return p
	}

	first, _, _ := newTestApp(t)
	first.box = rewriteBox{open: true, shown: true, focusText: true, session: "v1"}
	snap(first, session.Snapshot{ID: "v1", Kind: store.KindRewrite, State: session.Recording, Selection: "source"})
	listening := measure(t, first.boxView, "Stop and send")

	chat := &fakeChat{texts: []string{"one", "two"}}
	a, tr, _ := rewriteSelecting(t, chat, chatA, "summarize")
	answered := measure(t, a.boxView, "Speak")
	tr.setText("shorter")
	a.boxActions().speak()
	a.pump("recording", func() bool { return a.sess.State == session.Recording })
	again := measure(t, a.boxView, "Stop and send")

	for name, p := range map[string]place{"with an answer": answered, "listening again": again} {
		if p.add.X != listening.add.X || p.mic.X != listening.mic.X || p.send.X != listening.send.X ||
			p.insert.X != listening.insert.X || p.copy.X != listening.copy.X || p.close.X != listening.close.X {
			t.Errorf("%s the controls moved: %+v, first instruction %+v", name, p, listening)
		}
	}
}

// TestRewriteCardFirstAnswerIsAChat: the first answer in the Rewrite card
// shows under the instruction that asked for it, as each later turn does.
// It showed as bare text until a follow-up started, then turned into
// bubbles.
func TestRewriteCardFirstAnswerIsAChat(t *testing.T) {
	chat := &fakeChat{texts: []string{"one"}}
	a, _, _ := rewriteSelecting(t, chat, chatA, "summarize")
	tt := hudTester(a.boxView, boxWinW, boxWinH)
	frames(tt)
	if !hasExact(tt, "summarize") || !hasExact(tt, "one") {
		t.Fatalf("the first answer is not shown under its instruction: %q", firstTexts(tt))
	}
}
