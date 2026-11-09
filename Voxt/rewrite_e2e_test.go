package main

import (
	"strings"
	"testing"

	"github.com/sgl-project/sglang-omni/Voxt/internal/helper/helpertest"
	"github.com/sgl-project/sglang-omni/Voxt/internal/helper/proto"
	"github.com/sgl-project/sglang-omni/Voxt/internal/session"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

// Rewrite's scenarios end to end, after Voxt's (10-15, docs/behavior.md
// §6.9): the real session manager, the fake helper as a process (its
// recording, its focus and selection, its paste log), a loopback
// transcription service and a loopback chat service. With the card
// switch on, every answer waits in the card; Insert pastes it (over the
// selection when there was one); Continue (or the Rewrite shortcut)
// records a follow-up in the card, a conversation; Done stops it; Escape
// cancels a follow-up, then closes. With it off, the answer goes in at
// once.

// rewriteWith runs a Rewrite session from its shortcut (pressed twice,
// speech between) to its end, with the answer card switch card.
func rewriteWith(t *testing.T, opts string, chat *fakeChat, said string, card bool) (*sessionApp, *fakeTranscriptions) {
	t.Helper()
	set, keys, tr := fileSetup(t, said, chat)
	ev := script(t, hotkeyLine(0, "rewrite/0", proto.PhaseTrigger), hotkeyLine(500, "rewrite/0", proto.PhaseTrigger))
	a := newSessionApp(t, opts+ev, func(s *settings.Settings) { set(s); s.Features.ShowCard = card }, keys)
	a.pump("the end", func() bool { return a.sess.State.Ended() })
	return a, tr
}

// rewriteOnce is rewriteWith with the card switch on: the card's
// scenarios.
func rewriteOnce(t *testing.T, opts string, chat *fakeChat, said string) (*sessionApp, *fakeTranscriptions) {
	t.Helper()
	return rewriteWith(t, opts, chat, said, true)
}

// followUp records a follow-up from the card's Continue and stops it with
// Done, as the card's buttons do.
func (a *sessionApp) recordFollowUp(t *testing.T, tr *fakeTranscriptions, said string) {
	t.Helper()
	tr.setText(said)
	first := a.sess.ID
	if !a.followUpAnswer("") {
		t.Fatalf("Continue did nothing: %+v", a.answer)
	}
	a.pump("the follow-up's recording", func() bool {
		return a.sess.ID != first && a.sess.State == session.Recording && a.sess.HeardAt.After(a.sess.StartedAt)
	})
	if v := a.cardNow(); a.ov.mode != ovAnswer || !v.conversation || !v.recording || !v.canStop || v.canFollowUp {
		t.Errorf("recording the follow-up: %s %+v", a.ov.mode, v)
	}
	// At least 400 ms recorded, over the shortest recording kept: the
	// fake helper sends a level before each 100 ms frame.
	a.pump("the follow-up's audio", func() bool { return len(a.sess.Levels) >= 5 })
	a.stopSession() // Done
	a.pump("the follow-up's end", func() bool { return a.sess.ID != first && a.sess.State.Ended() })
}

// TestRewriteShortAnswerInCard: 10-15, the user's case, with the card
// switch on: nothing selected, an app that takes text, a short answer: it
// waits in the card (it was typed, with no way to follow up), with
// Continue; Insert pastes it.
func TestRewriteShortAnswerInCard(t *testing.T) {
	chat := &fakeChat{text: `{"title":"Help","content":"Sure: tell me the problem."}`}
	a, _ := rewriteOnce(t, "", chat, "help me with a problem")
	v := a.cardNow()
	if a.sess.State != session.Done || a.sess.Inserted || !a.answer.open || a.ov.mode != ovAnswer || len(a.pasted()) != 0 {
		t.Fatalf("end %+v card %+v", a.sess, a.answer)
	}
	if v.title != "Help" || v.text != "Sure: tell me the problem." || !v.canFollowUp || !v.canInsert || !v.canCopy || v.conversation {
		t.Errorf("card %+v", v)
	}
	a.insertAnswer()
	a.pump("the insert", func() bool { return !a.answer.open })
	if p := a.pasted(); len(p) != 1 || p[0].Text != "Sure: tell me the problem." || a.ov.shown {
		t.Errorf("pasted %+v", p)
	}
	a.pump("the history", func() bool { es := a.history(); return len(es) == 1 && es[0].Delivery == "insert" })
}

// TestRewriteSelectionInCard: a selection that can be edited: the
// rewrite waits in the card too (Voxt's); Insert pastes it over the
// selection; Continue follows up with the selection in the conversation.
func TestRewriteSelectionInCard(t *testing.T) {
	chat := &fakeChat{texts: []string{"Team: the release moves to Friday.", "Team: Friday."}}
	a, tr := rewriteOnce(t, helpertest.Opt("selection", "hey all, ship friday")+",", chat, "make it formal")
	if a.sess.Inserted || !a.answer.open || a.answer.input.Selection != "hey all, ship friday" || len(a.pasted()) != 0 {
		t.Fatalf("end %+v card %+v", a.sess, a.answer)
	}
	if v := a.cardNow(); v.title != "AI Answer" || !v.canFollowUp || !v.canInsert {
		t.Errorf("card %+v", v)
	}
	msgs := chatMessages(chat.requests()[0])
	if msgs[1][1] != "<instruction>\nmake it formal\n</instruction>\n\n<selected_text>\nhey all, ship friday\n</selected_text>" {
		t.Errorf("user message %q", msgs[1][1])
	}
	es := a.history()
	if len(es) != 1 || es[0].Kind != store.KindRewrite || es[0].PromptText != "make it formal" || es[0].SourceText != "hey all, ship friday" || es[0].Delivery != "card" {
		t.Errorf("history %+v", es)
	}
	a.recordFollowUp(t, tr, "shorter")
	if a.answer.text != "Team: Friday." || !a.answer.conversation || len(a.answer.turns) != 2 {
		t.Errorf("after the follow-up %+v", a.answer)
	}
	msgs = chatMessages(chat.requests()[1])
	if len(msgs) != 4 || !strings.Contains(msgs[1][1], "<selected_text>\nhey all, ship friday") || msgs[2][1] != "Team: the release moves to Friday." {
		t.Errorf("follow-up messages %q", msgs)
	}
	a.insertAnswer()
	a.pump("the insert", func() bool { return !a.answer.open })
	if p := a.pasted(); len(p) != 1 || p[0].Text != "Team: Friday." {
		t.Errorf("pasted %+v", p)
	}
}

// TestRewriteConversation: Continue, Done, a second follow-up cancelled
// with Escape (the card stays, with its last answer), Escape again closes.
func TestRewriteConversation(t *testing.T) {
	chat := &fakeChat{texts: []string{`{"title":"Late","content":"Sorry, I am running late."}`, "Running late.", "unused"}}
	a, tr := rewriteOnce(t, "", chat, "write that I am late")
	if !a.answer.open || a.answer.conversation {
		t.Fatalf("card %+v", a.answer)
	}
	a.recordFollowUp(t, tr, "shorter")
	v := a.cardNow()
	if !v.conversation || len(v.bubbles) != 4 || v.bubbles[0].text != "write that I am late" || v.bubbles[1].title != "Late" ||
		v.bubbles[2].text != "shorter" || v.bubbles[3].text != "Running late." || !v.canFollowUp || v.recording {
		t.Errorf("conversation %+v", v)
	}
	// A second follow-up, cancelled while it records.
	tr.setText("in French")
	second := a.sess.ID
	if !a.followUpAnswer("") {
		t.Fatal("Continue did nothing")
	}
	a.pump("the second follow-up", func() bool { return a.sess.ID != second && a.sess.State == session.Recording })
	a.hotkeyFired(proto.HotkeyEventParams{Binding: proto.HotkeyCancel, Phase: proto.PhaseTrigger})
	a.pump("the cancel", func() bool { return a.sess.State == session.Cancelled })
	if !a.answer.open || a.ov.mode != ovAnswer || a.answer.text != "Running late." || a.followUp != "" || len(chat.requests()) != 2 {
		t.Errorf("after Escape: %s %+v", a.ov.mode, a.answer)
	}
	if v := a.cardNow(); !v.conversation || len(v.bubbles) != 4 || !v.canFollowUp {
		t.Errorf("the conversation after Escape %+v", v)
	}
	a.hotkeyFired(proto.HotkeyEventParams{Binding: proto.HotkeyCancel, Phase: proto.PhaseTrigger})
	if a.answer.open || a.ov.shown || len(a.pasted()) != 0 {
		t.Errorf("the second Escape: %+v", a.answer)
	}
}

// TestRewriteNothingTakesText: nothing selected and nothing takes text:
// the card (Voxt's has no Insert then); ours inserts when asked, whatever
// has the focus: the user chose to.
func TestRewriteNothingTakesText(t *testing.T) {
	chat := &fakeChat{text: `{"title":"Note","content":"A note."}`}
	a, _ := rewriteOnce(t, "readonly,", chat, "write a note")
	if !a.answer.open || a.sess.Inserted || len(a.pasted()) != 0 {
		t.Fatalf("card %+v", a.answer)
	}
	a.insertAnswer()
	a.pump("the insert", func() bool { return !a.answer.open })
	if p := a.pasted(); len(p) != 1 || p[0].Text != "A note." {
		t.Errorf("pasted %+v", p)
	}
}

// TestRewriteNoCard: the card switch off (one for the three features): a
// rewrite replaces the selection, and an answer written from an
// instruction is typed where the cursor is, at once: no card, so no
// follow-up. What cannot go in waits in the card all the same, and takes
// follow-ups there.
func TestRewriteNoCard(t *testing.T) {
	t.Run("selection", func(t *testing.T) {
		chat := &fakeChat{text: "Team: the release moves to Friday."}
		a, _ := rewriteWith(t, helpertest.Opt("selection", "hey all, ship friday")+",", chat, "make it formal", false)
		if p := a.pasted(); a.sess.State != session.Done || !a.sess.Inserted || a.answer.open || len(p) != 1 || p[0].Text != "Team: the release moves to Friday." {
			t.Fatalf("end %+v card %+v pasted %+v", a.sess, a.answer, p)
		}
		a.pump("the history", func() bool {
			es := a.history()
			return len(es) == 1 && es[0].Delivery == "insert" && es[0].SourceText == "hey all, ship friday"
		})
	})
	t.Run("from an instruction", func(t *testing.T) {
		chat := &fakeChat{text: `{"title":"Help","content":"Sure: tell me the problem."}`}
		a, _ := rewriteWith(t, "", chat, "help me with a problem", false)
		if p := a.pasted(); a.sess.State != session.Done || !a.sess.Inserted || a.answer.open || a.canFollowUp() || len(p) != 1 || p[0].Text != "Sure: tell me the problem." {
			t.Fatalf("end %+v card %+v pasted %+v", a.sess, a.answer, p)
		}
	})
	t.Run("nothing takes text", func(t *testing.T) {
		chat := &fakeChat{text: `{"title":"Note","content":"A note."}`}
		a, _ := rewriteWith(t, "readonly,", chat, "write a note", false)
		if v := a.cardNow(); a.sess.Inserted || !a.answer.open || len(a.pasted()) != 0 || v.text != "A note." || !v.canFollowUp {
			t.Fatalf("end %+v card %+v", a.sess, v)
		}
	})
}
