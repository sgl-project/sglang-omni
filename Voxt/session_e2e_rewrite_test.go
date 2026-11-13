package main

// Whole sessions of Rewrite on Linux (the fakes are in
// local_fakes_test.go, the app in session_e2e_test.go).

import (
	"strings"
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/helper/helpertest"
	"github.com/sgl-project/sglang-omni/Voxt/internal/helper/proto"
	"github.com/sgl-project/sglang-omni/Voxt/internal/session"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
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
	a.pump("the follow-up's recording", func() bool { return a.sess.State == session.Recording && !a.sess.HeardAt.Before(a.sess.StartedAt) })
	// It records in the card, now a conversation (Voxt's).
	if v := a.cardNow(); !a.answer.open || a.ov.mode != ovAnswer || !v.conversation || !v.recording {
		t.Errorf("card %v overlay %s while recording: %+v", a.answer.open, a.ov.mode, v)
	}
	// At least 400 ms recorded, over the shortest recording kept: the
	// fake helper sends a level before each 100 ms frame.
	a.pump("the follow-up's audio", func() bool { return len(a.sess.Levels) >= 5 })
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
	a.hotkeyFired(proto.HotkeyEventParams{Binding: proto.HotkeyCancel, Phase: proto.PhaseTrigger})
	if a.answer.open {
		t.Error("Escape did not close the card")
	}
}

// TestStrictShortcuts: Rewrite started with its shortcut, then the
// Transcribe shortcut. By default it stops the recording (Voxt 1.x's
// common stop); with Strict shortcuts it does nothing, and the session
// keeps recording until Rewrite's own shortcut stops it (#3823).
func TestStrictShortcuts(t *testing.T) {
	for _, strict := range []bool{false, true} {
		t.Run(map[bool]string{false: "default", true: "strict"}[strict], func(t *testing.T) {
			chat := &fakeChat{text: "Sorry, I am running late."}
			set, keys, _ := fileSetup(t, "write that I am late", chat)
			a := newSessionApp(t, "readonly,"+script(t, hotkeyLine(0, "rewrite/0", proto.PhaseTrigger)), func(s *settings.Settings) {
				set(s)
				s.Features.StrictShortcuts = strict
			}, keys)
			a.pump("the recording", func() bool { return a.sess.State == session.Recording && len(a.sess.Levels) >= 5 })
			front := helpertest.Apps["com.tinyspeck.slackmacgap"]
			press := func(binding string) {
				a.hotkeyFired(proto.HotkeyEventParams{Binding: binding, Phase: proto.PhaseTrigger, Behavior: proto.BehaviorTap, Front: &front})
			}
			press("transcribe/0")
			if !strict {
				a.pump("the stop", func() bool { return a.sess.State != session.Recording })
				if a.sess.Kind != store.KindRewrite {
					t.Errorf("switched to %s", a.sess.Kind)
				}
				return
			}
			// Half a second of audio: a stop would have ended the recording by then.
			pressed := time.Now()
			a.pump("more audio", func() bool { return time.Since(pressed) > 500*time.Millisecond || a.sess.State != session.Recording })
			if a.sess.State != session.Recording || !a.sess.StoppedAt.IsZero() || a.sess.Kind != store.KindRewrite {
				t.Fatalf("the Transcribe shortcut acted: %s %s", a.sess.Kind, a.sess.State)
			}
			press("rewrite/0")
			a.pump("the stop", func() bool { return a.sess.State != session.Recording })
			a.pump("the end", func() bool { return a.sess.State.Ended() })
			if a.sess.State != session.Done || a.sess.Kind != store.KindRewrite {
				t.Errorf("end %s %s", a.sess.Kind, a.sess.State)
			}
		})
	}
}
