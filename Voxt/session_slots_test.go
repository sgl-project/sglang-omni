package main

import (
	"strings"
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/helper/proto"
	"github.com/sgl-project/sglang-omni/Voxt/internal/session"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

// #4005: the multi-session prototype. The session shortcuts (session-1 …)
// switch the slot Rewrite, Translate and Transcribe run in; each slot
// keeps its own answer card and conversation. The first card is from
// Slack (rewriteSelecting); the user goes on in TextEdit.

// TestSessionSlotSwitchKeepsConversations: a switch parks the current
// slot's card, the next slot starts its own conversation (no earlier
// turns go to the model), and switching back brings the first card and
// its conversation back, where a follow-up continues.
func TestSessionSlotSwitchKeepsConversations(t *testing.T) {
	chat := &fakeChat{texts: []string{"one", "two", "three"}}
	a, tr, _ := rewriteSelecting(t, chat, chatA, "summarize")
	conv1 := a.answer.conv

	tap(a, "session-2", textEditApp)
	if a.slot != 1 || a.answer.open || !a.slotsUsed {
		t.Fatalf("switched: slot=%d answer=%+v slotsUsed=%v", a.slot, a.answer, a.slotsUsed)
	}
	if p := a.slots[0]; !p.answer.open || p.answer.text != "one" || len(p.answer.turns) != 1 {
		t.Fatalf("parked: %+v", p.answer)
	}
	if a.notice == nil || a.notice.text != "Session 2" {
		t.Fatalf("notice %+v", a.notice)
	}

	// A Rewrite in session 2 starts its own conversation: no earlier turn.
	rewriteSaid(t, a, tr, textEditApp, "write a haiku", "two")
	if len(a.answer.turns) != 1 || a.answer.conv == conv1 {
		t.Fatalf("session 2 card: conv=%s (session 1: %s) turns=%d", a.answer.conv, conv1, len(a.answer.turns))
	}
	msgs := chatMessages(chat.requests()[len(chat.requests())-1])
	if len(msgs) != 2 {
		t.Fatalf("session 2 request carries %d messages, want 2: %q", len(msgs), msgs)
	}

	// Back in session 1, its card and conversation are as they were.
	tap(a, "session-1", textEditApp)
	slackCardBack(t, a, "back in session 1")
	if v := a.cachedCard(); v.Session != "Session 1" {
		t.Fatalf("card label %q", v.Session)
	}

	// Its follow-up continues its conversation, with the first turn.
	rewriteSaid(t, a, tr, a.answer.target, "shorter", "three")
	msgs = chatMessages(chat.requests()[len(chat.requests())-1])
	if len(msgs) != 4 || len(a.answer.turns) != 2 || !strings.Contains(msgs[1][1], "summarize") || !strings.Contains(msgs[3][1], "shorter") {
		t.Fatalf("follow-up: %d messages, %d turns: %q", len(msgs), len(a.answer.turns), msgs)
	}
	// Session 2's card waits in its slot.
	if p := a.slots[1]; !p.answer.open || p.answer.text != "two" {
		t.Fatalf("session 2 parked: %+v", p.answer)
	}
}

// TestSessionSlotSwitchWhileAnswerStreams: the owner's case (#4005). A
// Rewrite answer streams in Session 1; Option+2 switches anyway — the
// answer finishes in the background in Session 1 — and a Transcribe
// runs in Session 2 beside it. The finished card is Session 1's alone,
// with its conversation; Session 2 never shows it.
func TestSessionSlotSwitchWhileAnswerStreams(t *testing.T) {
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
		return a.sess.CanStop() && a.sess.Draft == full
	})

	// The switch backgrounds the stream, in Session 1; Session 2 shows
	// nothing of it.
	tap(a, "session-2", textEditApp)
	if a.slot != 1 || a.bgID == "" || !a.bgSess.State.Live() || a.answer.open || a.ov.mode == ovAnswer {
		t.Fatalf("switched mid-stream: slot=%d bg=%q answer open=%v mode=%s", a.slot, a.bgID, a.answer.open, a.ov.mode)
	}

	// A Transcribe runs in Session 2 beside the stream (wall 116).
	startSaying(a, tr, "transcribe/0", textEditApp, "dictated words")
	tap(a, "transcribe/0", textEditApp)
	a.pump("the transcription's end", func() bool { return a.sess.State.Ended() })
	if p := a.pasted(); len(p) != 1 || p[0].Text != "dictated words" {
		t.Fatalf("pasted %+v", p)
	}
	if a.answer.open || a.slots[1].answer.open {
		t.Fatalf("the transcription left a card: %+v / %+v", a.answer, a.slots[1].answer)
	}

	// Back in Session 1 the stream shows again; stopping it finishes
	// the card there, with its own conversation (#3886).
	tap(a, "session-1", textEditApp)
	a.pump("the stream visible again", func() bool { return a.ov.mode == ovAnswer })
	a.hotkeyFired(proto.HotkeyEventParams{Binding: proto.HotkeyCancel, Phase: proto.PhaseTrigger})
	a.pump("the answer's end", func() bool { return a.bgID == "" })
	if c := a.answer; !c.open || c.text != full || c.slot != 0 || len(c.turns) != 1 {
		t.Fatalf("session 1's card %+v", c)
	}
	if v := a.cachedCard(); v.Session != "Session 1" {
		t.Fatalf("card label %q", v.Session)
	}
	if p := a.slots[1]; p.answer.open || p.under.open {
		t.Fatalf("session 2 was given a card: %+v", p)
	}
	a.pump("the history", func() bool { return len(a.history()) == 2 })
	es := a.history()
	if es[0].Kind != store.KindTranscribe || es[0].Conversation == es[1].Conversation {
		t.Fatalf("conversations mixed: %+v", es)
	}
}

// TestSessionSlotAnswerFinishesInItsSlot: the streaming answer ends
// while Session 2 is current: its card waits in Session 1, conversation
// and all, instead of taking Session 2's place.
func TestSessionSlotAnswerFinishesInItsSlot(t *testing.T) {
	full := "先订会议室 then email Ann。"
	chat := &fakeChat{text: `{"title":"Plan","content":"` + full + `"}`, stall: true, chunkDelay: 50 * time.Millisecond}
	set, keys, _ := fileSetup(t, "plan the offsite", chat)
	ev := script(t, hotkeyLine(0, "rewrite/0", proto.PhaseTrigger), hotkeyLine(500, "rewrite/0", proto.PhaseTrigger))
	a := newSessionApp(t, ev, func(s *settings.Settings) {
		set(s)
		s.Features.Rewrite.ShowCard = true
	}, keys)
	a.pump("the whole answer in the card", func() bool {
		return a.sess.CanStop() && a.sess.Draft == full
	})
	tap(a, "session-2", textEditApp)
	if a.bgID == "" || a.slot != 1 {
		t.Fatalf("switched mid-stream: slot=%d bg=%q", a.slot, a.bgID)
	}
	a.mgr.StopLLM(a.bgID) // the answer ends here, kept (#3886)
	a.pump("the answer's end", func() bool { return a.bgID == "" })
	if a.answer.open || a.ov.mode == ovAnswer {
		t.Fatalf("Session 2 shows the card: %+v mode=%s", a.answer, a.ov.mode)
	}
	p := a.slots[0]
	if !p.answer.open || p.answer.text != full || p.answer.slot != 0 || len(p.answer.turns) != 1 {
		t.Fatalf("parked in session 1: %+v", p.answer)
	}
	tap(a, "session-1", textEditApp)
	if c := a.answer; !c.open || c.text != full || len(c.turns) != 1 {
		t.Fatalf("restored: %+v", c)
	}
}

// TestSessionSlotSwitchRestoresParkedCard: a slot parks both the open
// card and the card waiting under it (#4013); switching back restores
// both, and the waiting one still resurfaces.
func TestSessionSlotSwitchRestoresParkedCard(t *testing.T) {
	chat := &fakeChat{texts: []string{"one", "two"}}
	a, tr, _ := rewriteSelecting(t, chat, chatA, "summarize")
	// A Rewrite in another app takes the card's place; it waits under.
	rewriteSaid(t, a, tr, textEditApp, "write a haiku", "two")
	if !a.answer.open || a.answer.text != "two" || !a.under.open || a.under.text != "one" {
		t.Fatalf("before switch: answer=%q under=%q", a.answer.text, a.under.text)
	}
	tap(a, "session-2", textEditApp)
	if a.answer.open || a.under.open {
		t.Fatalf("switched away: answer=%q under=%q", a.answer.text, a.under.text)
	}
	tap(a, "session-1", textEditApp)
	if !a.answer.open || a.answer.text != "two" || !a.under.open || a.under.text != "one" {
		t.Fatalf("restored: answer=%q under=%q", a.answer.text, a.under.text)
	}
	// Closing the card still brings the waiting one back.
	a.closeAnswer()
	slackCardBack(t, a, "after close")
}

// TestSessionSlotSwitchWhileRecordingIgnored: a slot switch waits for an
// idle app: the keys do nothing while a session records.
func TestSessionSlotSwitchWhileRecordingIgnored(t *testing.T) {
	chat := &fakeChat{texts: []string{"one"}}
	a, tr, _ := rewriteSelecting(t, chat, chatA, "summarize")
	startSaying(a, tr, "transcribe/0", textEditApp, "dictated")
	tap(a, "session-2", textEditApp)
	if a.slot != 0 || a.slotsUsed || a.sess.State != session.Recording {
		t.Fatalf("switch during recording: slot=%d slotsUsed=%v state=%s", a.slot, a.slotsUsed, a.sess.State)
	}
	tap(a, "transcribe/0", textEditApp)
	a.pump("the end", func() bool { return a.sess.State.Ended() })
	slackCardBack(t, a, "after the transcription")
}

// TestSessionSlotSwitchWhileInsertingIgnored: an insert's paste must
// find its card in place, so a switch waits for it too.
func TestSessionSlotSwitchWhileInsertingIgnored(t *testing.T) {
	chat := &fakeChat{texts: []string{"one"}}
	a, _, _ := rewriteSelecting(t, chat, chatA, "summarize")
	a.answer.inserting = true
	tap(a, "session-2", textEditApp)
	a.answer.inserting = false
	if a.slot != 0 || a.slotsUsed {
		t.Fatalf("switch during insert: slot=%d slotsUsed=%v", a.slot, a.slotsUsed)
	}
}

// TestSessionSlotsDefaultUnchanged: with no session shortcut pressed,
// setAside and resurface behave as before (#4013) and the card shows no
// session label.
func TestSessionSlotsDefaultUnchanged(t *testing.T) {
	chat := &fakeChat{texts: []string{"one", "two"}}
	a, tr, _ := rewriteSelecting(t, chat, chatA, "summarize")
	rewriteSaid(t, a, tr, textEditApp, "write a haiku", "two")
	if !a.answer.open || a.answer.text != "two" || !a.under.open || a.under.text != "one" {
		t.Fatalf("set aside: answer=%q under=%q", a.answer.text, a.under.text)
	}
	a.closeAnswer()
	slackCardBack(t, a, "resurfaced")
	if a.slotsUsed {
		t.Fatal("no session shortcut was pressed")
	}
	if v := a.cachedCard(); v.Session != "" {
		t.Fatalf("label without sessions: %q", v.Session)
	}
}
