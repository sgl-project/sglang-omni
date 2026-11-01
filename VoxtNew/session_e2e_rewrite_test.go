package main

// Whole sessions of Rewrite on Linux (the fakes are in
// local_fakes_test.go, the app in session_e2e_test.go).

import (
	"strings"
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/VoxtNew/internal/helper/helpertest"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/helper/proto"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/session"
)

// TestRewriteAnswerFollowUp: nothing selected and nothing takes text: the
// JSON answer streams into the card with its title; the Rewrite shortcut
// on the card records a follow-up, sent with the conversation so far,
// answered in plain text into the card's conversation.
func TestRewriteAnswerFollowUp(t *testing.T) {
	chat := &fakeChat{texts: []string{`{"title":"Late","content":"Sorry, I am running late."}`, "Running late."}}
	set, keys, tr := fileSetup(t, "write that I am late", chat)
	ev := script(t, hotkeyLine(0, "rewrite/0", proto.PhaseTrigger), hotkeyLine(500, "rewrite/0", proto.PhaseTrigger))
	a := newSessionApp(t, "readonly,"+ev, set, keys)
	a.pump("the end", func() bool { return a.sess.State.Ended() })
	if a.sess.Result != "Sorry, I am running late." || a.sess.Title != "Late" || !a.answer.open || a.answer.title != "Late" || len(a.answer.turns) != 1 {
		t.Fatalf("end %+v card %+v", a.sess, a.answer)
	}
	if reqs := chat.requests(); reqs[0]["response_format"] == nil {
		t.Errorf("no JSON asked: %v", reqs[0])
	}
	// The follow-up: the shortcut, a moment of speech, the shortcut.
	tr.setText("shorter")
	front := helpertest.Apps["com.tinyspeck.slackmacgap"]
	a.hotkeyFired(proto.HotkeyEventParams{Binding: "rewrite/0", Phase: proto.PhaseTrigger, Behavior: proto.BehaviorTap, Front: &front})
	a.pump("the follow-up's recording", func() bool { return a.sess.State == session.Recording && a.sess.HeardAt.After(a.sess.StartedAt) })
	// It records in the card, now a conversation (Voxt's).
	if v := a.cardNow(); !a.answer.open || a.ov.mode != ovAnswer || !v.conversation || !v.recording {
		t.Errorf("card %v overlay %s while recording: %+v", a.answer.open, a.ov.mode, v)
	}
	time.Sleep(400 * time.Millisecond)
	a.hotkeyFired(proto.HotkeyEventParams{Binding: "rewrite/0", Phase: proto.PhaseTrigger, Behavior: proto.BehaviorTap, Front: &front})
	a.pump("the follow-up's end", func() bool { return a.sess.State.Ended() && a.answer.text == "Running late." })
	if a.answer.text != "Running late." || len(a.answer.turns) != 2 || !a.answer.open || !a.answer.conversation || len(a.pasted()) != 0 {
		t.Errorf("card %+v", a.answer)
	}
	// A follow-up answers in plain text, without a title (Voxt's).
	if v := a.cardNow(); len(v.bubbles) != 4 || !v.bubbles[2].user || v.bubbles[2].text != "shorter" || v.bubbles[3].title != "" || v.bubbles[3].text != "Running late." || !v.canFollowUp {
		t.Errorf("the conversation %+v", v.bubbles)
	}
	msgs := chatMessages(chat.requests()[1])
	if len(msgs) != 4 || msgs[2][0] != "assistant" || !strings.Contains(msgs[2][1], "Sorry, I am running late.") || msgs[3][1] != "<instruction>\nshorter\n</instruction>" {
		t.Errorf("follow-up messages %q", msgs)
	}
	a.hotkeyFired(proto.HotkeyEventParams{Binding: proto.HotkeyEscape, Phase: proto.PhaseTrigger})
	if a.answer.open {
		t.Error("Escape did not close the card")
	}
}
