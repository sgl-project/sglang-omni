package main

// Wall 116: the transcribe shortcut works while a Rewrite answer streams
// into its card. The answer backgrounds and finishes into the unmoved
// card; the session's pill shows in its own window beside it. The
// shortcut used to be refused while the model wrote, and a transcription
// closed the card and reopened it at the default place.

import (
	"testing"
	"time"

	"github.com/egoist/mygo"
	"github.com/sgl-project/sglang-omni/Voxt/internal/helper/proto"
	"github.com/sgl-project/sglang-omni/Voxt/internal/session"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

// framesOverlap reports whether two window frames intersect. The pill
// window must never cover the answer card (wall 116: both stay
// readable); placing it on the card's anchor hid the card's body.
func framesOverlap(a, b mygo.Rectangle) bool {
	return a.X < b.X+b.Width && b.X < a.X+a.Width && a.Y < b.Y+b.Height && b.Y < a.Y+a.Height
}

// pillBesideCard asserts the pill window shows and does not touch the
// card's frame.
func pillBesideCard(t *testing.T, a *sessionApp, when string) {
	t.Helper()
	if !a.pill.shown || a.ov.mode != ovAnswer || framesOverlap(a.ov.frame, a.pill.frame) {
		t.Fatalf("%s: card %v, pill %v (shown %v, main %s)", when, a.ov.frame, a.pill.frame, a.pill.shown, a.ov.mode)
	}
}

// TestTranscribeWhileRewriteStreams: Option+Space mid-stream. The answer
// keeps streaming in the background, the streaming card never leaves,
// moves or changes, the pill records beside it; when the transcription
// ends, the card is as it was, and the cancel shortcut stops the
// still-streaming answer (#3886's, now in the background).
func TestTranscribeWhileRewriteStreams(t *testing.T) {
	// chunkDelay over draftEvery: every piece publishes, so the whole
	// draft is there deterministically, under any load.
	full := "先订会议室 then email Ann。"
	chat := &fakeChat{text: `{"title":"Plan","content":"` + full + `"}`, stall: true, chunkDelay: 50 * time.Millisecond}
	set, keys, tr := fileSetup(t, "plan the offsite", chat)
	ev := script(t, hotkeyLine(0, "rewrite/0", proto.PhaseTrigger), hotkeyLine(500, "rewrite/0", proto.PhaseTrigger))
	a := newSessionApp(t, ev, func(s *settings.Settings) {
		set(s)
		s.Features.Rewrite.ShowCard = true
		s.Features.Transcribe.Enhance = false // no second, stalled, model request
	}, keys)
	a.pump("the whole answer in the card", func() bool {
		return a.sess.CanStop() && a.ov.mode == ovAnswer && a.sess.Draft == full
	})
	if a.pill.mode != ovHidden || a.pill.shown {
		t.Fatalf("the streaming answer's own session in the pill window: %s", a.pill.mode)
	}
	frame := a.ov.frame

	tr.setText("dictated words")
	tap(a, "transcribe/0", textEditApp)
	a.pump("recording beside the card", func() bool {
		return a.sess.Kind == store.KindTranscribe && a.sess.State == session.Recording && len(a.sess.Levels) >= 5
	})
	if a.bgID == "" || !a.bgSess.State.Live() || a.answer.open || a.under.open ||
		a.ov.mode != ovAnswer || a.pill.mode != ovRecording || a.ov.frame != frame ||
		cardTextForTest(a.cachedCard()) != full {
		t.Fatalf("recording: bg=%q card open=%v under=%v mode=%s pill=%s frame=%v (was %v), card %q",
			a.bgID, a.answer.open, a.under.open, a.ov.mode, a.pill.mode, a.ov.frame, frame, cardTextForTest(a.cachedCard()))
	}
	pillBesideCard(t, a, "recording beside the streaming card")
	tap(a, "transcribe/0", textEditApp)
	a.pump("the transcription's end", func() bool { return a.sess.State.Ended() })
	if p := a.pasted(); len(p) != 1 || p[0].Text != "dictated words" {
		t.Fatalf("pasted %+v", p)
	}
	// The card is as it was; the answer streams on in the background.
	if a.bgID == "" || a.answer.open || a.ov.mode != ovAnswer || a.pill.shown || a.ov.frame != frame {
		t.Fatalf("after the transcription: bg=%q card open=%v mode=%s pill=%s frame=%v (was %v)",
			a.bgID, a.answer.open, a.ov.mode, a.pill.mode, a.ov.frame, frame)
	}
	// The cancel shortcut stops the backgrounded answer where it is; the
	// card finishes as any stopped answer's (#3886).
	a.hotkeyFired(proto.HotkeyEventParams{Binding: proto.HotkeyCancel, Phase: proto.PhaseTrigger})
	a.pump("the answer's end", func() bool { return a.bgID == "" })
	if !a.answer.open || a.answer.text != full || a.answer.title != "Plan" || a.ov.mode != ovAnswer {
		t.Fatalf("the finished card %+v, mode=%s", a.answer, a.ov.mode)
	}
	a.pump("the history", func() bool { return len(a.history()) == 2 })
	es := a.history()
	if es[0].Kind != store.KindTranscribe || es[0].Result != "dictated words" ||
		es[1].Kind != store.KindRewrite || es[1].Status != store.StatusDone || es[1].Result != full {
		t.Fatalf("history %+v", es)
	}
}

// TestRewriteAnswerFinishesBesideTranscribe: the answer completes on its
// own while the transcription still records; the card gets it, unmoved.
func TestRewriteAnswerFinishesBesideTranscribe(t *testing.T) {
	// A long answer: the stream outlasts the shortcut's round trip.
	body := "Dear team, the release moves to Thursday next week. "
	for len([]rune(body)) < 2400 {
		body += body
	}
	long := `{"title":"Letter","content":"` + body + `"}`
	chat := &fakeChat{text: long}
	set, keys, tr := fileSetup(t, "plan the offsite", chat)
	ev := script(t, hotkeyLine(0, "rewrite/0", proto.PhaseTrigger), hotkeyLine(500, "rewrite/0", proto.PhaseTrigger))
	a := newSessionApp(t, ev, func(s *settings.Settings) { set(s); s.Features.Rewrite.ShowCard = true }, keys)
	a.pump("the answer streaming", func() bool { return a.sess.CanStop() && a.ov.mode == ovAnswer })

	tr.setText("dictated words")
	tap(a, "transcribe/0", textEditApp)
	a.pump("recording beside the card", func() bool {
		return a.sess.Kind == store.KindTranscribe && a.sess.State == session.Recording && len(a.sess.Levels) >= 5
	})
	if a.bgID == "" || a.ov.mode != ovAnswer || a.pill.mode != ovRecording {
		t.Fatalf("recording: bg=%q mode=%s pill=%s", a.bgID, a.ov.mode, a.pill.mode)
	}
	pillBesideCard(t, a, "recording beside the streaming long card")
	// The answer ends while the transcription still records.
	a.pump("the answer done in the background", func() bool { return a.bgID == "" })
	pillBesideCard(t, a, "the long answer done")
	if !a.answer.open || a.answer.title != "Letter" || a.ov.mode != ovAnswer || a.pill.mode != ovRecording {
		t.Fatalf("the done answer: card=%+v mode=%s pill=%s", a.answer, a.ov.mode, a.pill.mode)
	}
	tap(a, "transcribe/0", textEditApp)
	a.pump("the transcription's end", func() bool { return a.sess.State.Ended() })
	if !a.answer.open || a.answer.title != "Letter" || a.ov.mode != ovAnswer || a.pill.shown {
		t.Fatalf("after the transcription: card=%+v mode=%s pill=%s", a.answer, a.ov.mode, a.pill.mode)
	}
	a.pump("the history", func() bool { return len(a.history()) == 2 })
}

// TestTranscribeRefusedKeepsTheAnswerStreaming: the transcribe shortcut
// while the answer streams, with Transcribe's own speech model missing:
// the notice says so, the answer keeps streaming in the foreground (it
// must not background for a session that cannot start), and the next
// shortcut press is judged anew.
func TestTranscribeRefusedKeepsTheAnswerStreaming(t *testing.T) {
	// A longer answer: the stream spans draftEvery, so the draft publishes.
	chat := &fakeChat{text: `{"title":"Plan","content":"先订会议室 then email Ann。"}`, stall: true}
	// The speech model serves Rewrite, not Transcribe: the transcribe
	// shortcut has no model to start with.
	svc := withChat(&fakeLocal{ASRP: &fakeTranscriptions{text: "plan the offsite"}, ASRFeatures: []string{"rewrite"}}, chat)
	ev := script(t, hotkeyLine(0, "rewrite/0", proto.PhaseTrigger), hotkeyLine(500, "rewrite/0", proto.PhaseTrigger))
	a := newSessionApp(t, ev, func(s *settings.Settings) { s.Features.Rewrite.ShowCard = true }, svc)
	a.pump("the answer streaming", func() bool { return a.sess.CanStop() && a.ov.mode == ovAnswer })
	tap(a, "transcribe/0", textEditApp)
	a.pump("the notice", func() bool { return a.notice != nil })
	if a.notice.text != "No speech model is installed." || a.bgID != "" || !a.sess.CanStop() || a.pill.mode != ovNotice {
		t.Fatalf("notice=%+v bg=%q sess=%s pill=%s", a.notice, a.bgID, a.sess.State, a.pill.mode)
	}
	// The refused press changed nothing: the answer still streams in the
	// foreground, and stopping it works as ever (#3886).
	a.hotkeyFired(proto.HotkeyEventParams{Binding: proto.HotkeyCancel, Phase: proto.PhaseTrigger})
	a.pump("the answer's end", func() bool { return a.sess.State.Ended() })
	if a.sess.State != session.Done || !a.answer.open || a.answer.text != "先订会议室 then email Ann。" {
		t.Fatalf("end %s, card %+v", a.sess.State, a.answer)
	}
	// Not bricked: the next press is judged on its own.
	tap(a, "transcribe/0", textEditApp)
	a.pump("the notice again", func() bool { return a.notice != nil && a.notice.text == "No speech model is installed." })
	if a.bgID != "" {
		t.Fatalf("a refused start backgrounded: %q", a.bgID)
	}
}

// TestConversationSurvivesATranscribeCard: a follow-up streams in the
// background and the transcription beside it lands in a card of its own:
// the new card waits under the conversation, which the follow-up's end
// completes with its turns intact; closing it brings the new card back.
func TestConversationSurvivesATranscribeCard(t *testing.T) {
	chat := &fakeChat{texts: []string{`{"title":"Plan","content":"one"}`, "two, and more to come"}}
	set, keys, tr := fileSetup(t, "plan the offsite", chat)
	ev := script(t, hotkeyLine(0, "rewrite/0", proto.PhaseTrigger), hotkeyLine(500, "rewrite/0", proto.PhaseTrigger))
	a := newSessionApp(t, ev, func(s *settings.Settings) {
		set(s)
		s.Features.Rewrite.ShowCard = true
		s.Features.Transcribe.ShowCard = true // the transcription lands in a card of its own
		s.Features.Transcribe.Enhance = false
	}, keys)
	a.pump("the first answer's card", func() bool { return a.sess.State.Ended() && a.answer.open })
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
	if a.pill.mode != ovHidden || a.pill.shown {
		t.Fatalf("the card's own follow-up in the pill window: %s", a.pill.mode)
	}
	conv := a.answer.conv

	tr.setText("dictated words")
	tap(a, "transcribe/0", textEditApp)
	a.pump("recording beside the card", func() bool {
		return a.sess.Kind == store.KindTranscribe && a.sess.State == session.Recording && len(a.sess.Levels) >= 5
	})
	tap(a, "transcribe/0", textEditApp)
	a.pump("the transcription's end", func() bool { return a.sess.State.Ended() })
	// The transcription's card waits under the conversation, which the
	// follow-up still streams into.
	if a.bgID == "" || !a.answer.open || a.followUp != a.answer.id || len(a.answer.turns) != 1 ||
		!a.under.open || a.under.kind != store.KindTranscribe || a.under.text != "dictated words" {
		t.Fatalf("bg=%q card=%+v under=%+v", a.bgID, a.answer, a.under)
	}
	// The follow-up ends (stopped, as #3886's): the conversation completes.
	a.hotkeyFired(proto.HotkeyEventParams{Binding: proto.HotkeyCancel, Phase: proto.PhaseTrigger})
	a.pump("the follow-up's end", func() bool { return a.bgID == "" })
	if !a.answer.open || len(a.answer.turns) != 2 || a.answer.turns[1].Answer.Content != "two, and more to come" ||
		a.answer.conv != conv || !a.under.open {
		t.Fatalf("card=%+v under=%+v", a.answer, a.under)
	}
	// Closing the conversation brings the transcription's card back.
	a.closeAnswer()
	if !a.answer.open || a.answer.kind != store.KindTranscribe || a.answer.text != "dictated words" || a.under.open {
		t.Fatalf("card=%+v under=%+v", a.answer, a.under)
	}
}

// TestPillWindowBesideTheCardTop: with the overlay anchored to the top,
// the pill window goes below the card instead of covering it.
func TestPillWindowBesideTheCardTop(t *testing.T) {
	chat := &fakeChat{texts: []string{"one"}}
	set, keys, tr := fileSetup(t, "summarize", chat)
	ev := script(t, hotkeyLine(0, "rewrite/0", proto.PhaseTrigger), hotkeyLine(500, "rewrite/0", proto.PhaseTrigger))
	a := newSessionApp(t, ev, func(s *settings.Settings) {
		set(s)
		s.Features.Rewrite.ShowCard = true
		s.Overlay.Position = "top"
	}, keys)
	a.pump("the first answer's card", func() bool { return a.sess.State.Ended() && a.answer.open })
	frame := a.ov.frame
	startSaying(a, tr, "transcribe/0", textEditApp, "dictated")
	pillBesideCard(t, a, "recording beside the open card (top)")
	if a.pill.frame.Y < a.ov.frame.Y+a.ov.frame.Height {
		t.Fatalf("the pill is not below the card: card %v, pill %v", a.ov.frame, a.pill.frame)
	}
	tap(a, "transcribe/0", textEditApp)
	a.pump("the end", func() bool { return a.sess.State.Ended() })
	if !a.answer.open || a.answer.text != "one" || a.ov.frame != frame || a.pill.shown {
		t.Fatalf("after the transcription: card %+v frame %v (was %v), pill %s", a.answer, a.ov.frame, frame, a.pill.mode)
	}
}

// TestPillWindowBesideALongCardTop: the same tall card with the overlay
// anchored to the top: the pill goes below it, and never covers it even
// when the card fills the work area.
func TestPillWindowBesideALongCardTop(t *testing.T) {
	body := "Dear team, the release moves to Thursday next week. "
	for len([]rune(body)) < 2400 {
		body += body
	}
	chat := &fakeChat{text: `{"title":"Letter","content":"` + body + `"}`}
	set, keys, tr := fileSetup(t, "plan the offsite", chat)
	ev := script(t, hotkeyLine(0, "rewrite/0", proto.PhaseTrigger), hotkeyLine(500, "rewrite/0", proto.PhaseTrigger))
	a := newSessionApp(t, ev, func(s *settings.Settings) {
		set(s)
		s.Features.Rewrite.ShowCard = true
		s.Overlay.Position = "top"
	}, keys)
	a.pump("the answer streaming", func() bool { return a.sess.CanStop() && a.ov.mode == ovAnswer })
	tr.setText("dictated words")
	tap(a, "transcribe/0", textEditApp)
	a.pump("recording beside the card", func() bool {
		return a.sess.Kind == store.KindTranscribe && a.sess.State == session.Recording && len(a.sess.Levels) >= 5
	})
	pillBesideCard(t, a, "recording beside the long card (top)")
	a.pump("the answer done in the background", func() bool { return a.bgID == "" })
	pillBesideCard(t, a, "the long card done (top)")
	tap(a, "transcribe/0", textEditApp)
	a.pump("the transcription's end", func() bool { return a.sess.State.Ended() })
	if !a.answer.open || a.answer.title != "Letter" || a.pill.shown {
		t.Fatalf("after the transcription: card=%+v pill=%s", a.answer, a.pill.mode)
	}
}

// TestReviewTranscribeCardSurvivesInitialBackgroundAnswer: the
// transcription finishes first and opens its card (Transcribe's answer
// card on), then the backgrounded Rewrite ends: its card opens over it,
// and the transcription's card waits under and comes back on close. It
// used to vanish: setAside holds only Rewrite and Translate cards.
func TestReviewTranscribeCardSurvivesInitialBackgroundAnswer(t *testing.T) {
	chat := &fakeChat{text: `{"title":"Plan","content":"先订会议室 then email Ann。"}`, stall: true}
	set, keys, tr := fileSetup(t, "plan the offsite", chat)
	ev := script(t, hotkeyLine(0, "rewrite/0", proto.PhaseTrigger), hotkeyLine(500, "rewrite/0", proto.PhaseTrigger))
	a := newSessionApp(t, ev, func(s *settings.Settings) {
		set(s)
		s.Features.Rewrite.ShowCard = true
		s.Features.Transcribe.ShowCard = true
		s.Features.Transcribe.Enhance = false
	}, keys)
	a.pump("the whole answer in the card", func() bool {
		return a.sess.CanStop() && a.ov.mode == ovAnswer && a.sess.Draft == "先订会议室 then email Ann。"
	})
	tr.setText("dictated words")
	tap(a, "transcribe/0", textEditApp)
	a.pump("recording beside the card", func() bool {
		return a.sess.Kind == store.KindTranscribe && a.sess.State == session.Recording && len(a.sess.Levels) >= 5
	})
	tap(a, "transcribe/0", textEditApp)
	a.pump("the transcription's end", func() bool { return a.sess.State.Ended() })
	if !a.answer.open || a.answer.kind != store.KindTranscribe || a.answer.text != "dictated words" {
		t.Fatalf("the transcription did not open its card: %+v", a.answer)
	}
	transcribeID := a.answer.id
	a.hotkeyFired(proto.HotkeyEventParams{Binding: proto.HotkeyCancel, Phase: proto.PhaseTrigger})
	a.pump("the background answer's end", func() bool { return a.bgID == "" })
	if !a.answer.open || a.answer.kind != store.KindRewrite || a.answer.text != "先订会议室 then email Ann。" ||
		!a.under.open || a.under.id != transcribeID {
		t.Fatalf("answer=%+v under=%+v", a.answer, a.under)
	}
	a.closeAnswer()
	if !a.answer.open || a.answer.id != transcribeID || a.under.open {
		t.Fatalf("the transcription's card did not come back: answer=%+v under=%+v", a.answer, a.under)
	}
}

// TestReviewBackgroundFollowUpDoesNotAllowAnotherFollowUp: while the
// card's own follow-up still streams in the background, Continue is off:
// two follow-ups cannot run together.
func TestReviewBackgroundFollowUpDoesNotAllowAnotherFollowUp(t *testing.T) {
	chat := &fakeChat{texts: []string{`{"title":"Plan","content":"one"}`, "two, and more to come", "three"}}
	set, keys, tr := fileSetup(t, "plan the offsite", chat)
	ev := script(t, hotkeyLine(0, "rewrite/0", proto.PhaseTrigger), hotkeyLine(500, "rewrite/0", proto.PhaseTrigger))
	a := newSessionApp(t, ev, func(s *settings.Settings) {
		set(s)
		s.Features.Rewrite.ShowCard = true
		s.Features.Transcribe.ShowCard = true
		s.Features.Transcribe.Enhance = false
	}, keys)
	a.pump("the first answer's card", func() bool { return a.sess.State.Ended() && a.answer.open })
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
	tr.setText("dictated words")
	tap(a, "transcribe/0", textEditApp)
	a.pump("recording beside the card", func() bool {
		return a.sess.Kind == store.KindTranscribe && a.sess.State == session.Recording && len(a.sess.Levels) >= 5
	})
	tap(a, "transcribe/0", textEditApp)
	a.pump("the transcription's end", func() bool { return a.sess.State.Ended() })
	if a.bgID == "" || a.sess.State.Live() || !a.answer.open {
		t.Fatalf("unexpected setup: bg=%q session=%+v answer=%+v", a.bgID, a.sess, a.answer)
	}
	if a.canFollowUp() {
		t.Fatalf("a second follow-up is enabled while the first still streams in the background: bg=%q", a.bgID)
	}
	a.hotkeyFired(proto.HotkeyEventParams{Binding: proto.HotkeyCancel, Phase: proto.PhaseTrigger})
	a.pump("the follow-up's end", func() bool { return a.bgID == "" })
	if !a.canFollowUp() {
		t.Fatal("Continue stayed off after the follow-up ended")
	}
}
