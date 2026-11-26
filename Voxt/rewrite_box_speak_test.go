package main

import (
	"testing"

	"github.com/sgl-project/sglang-omni/Voxt/internal/helper/proto"
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
