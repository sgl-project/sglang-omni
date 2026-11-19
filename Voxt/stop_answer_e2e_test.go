package main

import (
	"strings"
	"testing"

	"github.com/sgl-project/sglang-omni/Voxt/internal/helper/proto"
	"github.com/sgl-project/sglang-omni/Voxt/internal/overlay/layout"
	"github.com/sgl-project/sglang-omni/Voxt/internal/session"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

// TestStopAnswerEndToEnd (#3886): while Rewrite's answer streams into the
// card, the card's Stop, or the cancel shortcut, ends the model's request;
// the card keeps the answer so far (also the words the throttled draft
// had not shown yet), to insert, copy or follow up, and the
// history keeps it. The cancel shortcut used to cancel the session and the
// written text was lost.
func TestStopAnswerEndToEnd(t *testing.T) {
	for _, how := range []string{"button", "shortcut"} {
		t.Run(how, func(t *testing.T) {
			chat := &fakeChat{text: `{"title":"Plan","content":"先订会议室 then email Ann。"}`, stall: true}
			set, keys, _ := fileSetup(t, "plan the offsite", chat)
			ev := script(t, hotkeyLine(0, "rewrite/0", proto.PhaseTrigger), hotkeyLine(500, "rewrite/0", proto.PhaseTrigger))
			a := newSessionApp(t, ev, func(s *settings.Settings) { set(s); s.Features.Rewrite.ShowCard = true }, keys)
			a.pump("the answer in the card", func() bool {
				return a.sess.CanStop() && a.ov.mode == ovAnswer
			})
			if v := a.cachedCard(); !v.Writing || !v.CanStopAnswer {
				t.Fatalf("the writing card %+v", v)
			}
			if how == "button" {
				tt := hudTester(a.overlayView, a.ov.w+2*layout.Pad, a.ov.h+2*layout.Pad)
				click(t, tt, "Stop")
			} else {
				a.hotkeyFired(proto.HotkeyEventParams{Binding: proto.HotkeyCancel, Phase: proto.PhaseTrigger})
			}
			a.pump("the end", func() bool { return a.sess.State.Ended() })
			v := a.cachedCard()
			if a.sess.State != session.Done || !a.answer.open || cardTextForTest(v) != "先订会议室 then email Ann。" || v.Title != "Plan" || !v.CanInsert || !v.CanFollowUp || v.CanStopAnswer {
				t.Fatalf("end %s, card %+v", a.sess.State, v)
			}
			if p := a.pasted(); len(p) != 0 {
				t.Errorf("pasted %+v", p)
			}
			if es := a.history(); len(es) != 1 || es[0].Status != store.StatusDone || es[0].Result != "先订会议室 then email Ann。" {
				t.Errorf("history %+v", es)
			}
		})
	}
}

// TestStopFollowUpAnswer (#3886): the cancel shortcut while a follow-up's
// answer streams stops it there: the conversation keeps the earlier turn
// and the follow-up's answer so far, to follow up on again. It used to
// cancel the follow-up and drop its answer.
func TestStopFollowUpAnswer(t *testing.T) {
	chat := &fakeChat{texts: []string{`{"title":"Plan","content":"one"}`, "two, and more to come"}}
	a, tr := rewriteOnce(t, "", chat, "plan the offsite")
	chat.mu.Lock()
	chat.stall = true
	chat.mu.Unlock()
	tr.setText("longer")
	first := a.sess.ID
	if !a.followUpAnswer("") {
		t.Fatalf("Continue did nothing: %+v", a.answer)
	}
	a.pump("the follow-up's audio", func() bool {
		return a.sess.ID != first && a.sess.State == session.Recording && len(a.sess.Levels) >= 5
	})
	a.stopSession()
	a.pump("the follow-up's answer", func() bool { return a.sess.CanStop() })
	if v := a.cachedCard(); !v.Conversation || !v.CanStopAnswer {
		t.Fatalf("the writing follow-up %+v", v)
	}
	a.hotkeyFired(proto.HotkeyEventParams{Binding: proto.HotkeyCancel, Phase: proto.PhaseTrigger})
	a.pump("the end", func() bool { return a.sess.ID != first && a.sess.State.Ended() })
	if a.sess.State != session.Done || !a.answer.open || len(a.answer.turns) != 2 || a.answer.turns[1].Answer.Content != "two, and more to come" || !a.canFollowUp() {
		t.Errorf("end %s %+v, card %+v", a.sess.State, a.sess.Problem, a.answer)
	}
}

// TestTranscribeCardStreams: with Transcribe's answer card on, the
// clean-up streams into the card; it used to wait in the pill for the
// whole text. The card offers Use transcript, not Stop (a clean-up cut
// short would drop the rest of what was said): the button, or the cancel
// shortcut, puts all of it in the card, as recognised.
func TestTranscribeCardStreams(t *testing.T) {
	for _, how := range []string{"button", "shortcut"} {
		t.Run(how, func(t *testing.T) {
			chat := &fakeChat{text: "So the release moves to Thursday.", stall: true}
			set, keys, _ := fileSetup(t, "so the release moves to thursday", chat)
			ev := script(t, hotkeyLine(0, "transcribe/0", proto.PhaseTrigger), hotkeyLine(500, "transcribe/0", proto.PhaseTrigger))
			a := newSessionApp(t, ev, func(s *settings.Settings) {
				set(s)
				s.Features.Transcribe.Enhance, s.Features.Transcribe.ShowCard = true, true
			}, keys)
			a.pump("the clean-up in the card", func() bool { return a.sess.State == session.Composing && a.ov.mode == ovAnswer })
			if v := a.cardNow(); !v.writing || v.kind != store.KindTranscribe || v.canStopAnswer || !v.canUseTranscript || !strings.HasPrefix(a.sess.Draft, v.text) {
				t.Fatalf("the writing card %+v", v)
			}
			if how == "button" {
				tt := hudTester(a.overlayView, a.ov.w+2*layout.Pad, a.ov.h+2*layout.Pad)
				click(t, tt, "Use transcript")
			} else {
				a.hotkeyFired(proto.HotkeyEventParams{Binding: proto.HotkeyCancel, Phase: proto.PhaseTrigger})
			}
			a.pump("the end", func() bool { return a.sess.State.Ended() })
			if v := a.cardNow(); a.sess.State != session.Done || !a.answer.open || v.text != a.sess.Raw || !v.canInsert {
				t.Fatalf("end %s %+v, card %+v", a.sess.State, a.sess.Problem, v)
			}
			if p := a.pasted(); len(p) != 0 {
				t.Errorf("pasted %+v", p)
			}
		})
	}
}
