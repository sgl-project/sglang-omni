package main

import (
	"strconv"
	"strings"
	"testing"
	"time"

	"github.com/sgl-project/sglang-omni/Voxt/internal/helper/proto"
	"github.com/sgl-project/sglang-omni/Voxt/internal/overlay/layout"
	"github.com/sgl-project/sglang-omni/Voxt/internal/session"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

// #4005: the multi-session prototype. Rewrite, Translate and Transcribe
// run in the current session of a list the user adds to; each session
// keeps its own answer card and conversation. The first card is from
// Slack (rewriteSelecting); the user goes on in TextEdit.

// sessionsOn turns Settings → Experimental's Sessions switch on.
func sessionsOn(a *sessionApp) {
	a.updateSettings(func(s *settings.Settings) { s.Features.SessionsOn = true })
}

// chipLine is the card's session row as text: "1 title*" is current.
func chipLine(a *sessionApp) string {
	var out []string
	chips := a.cachedCard().Sessions
	if a.box.open { // the Rewrite card draws the row
		chips = a.boxViewState().Sessions
	}
	for i, chip := range chips {
		text := strings.TrimSpace(strconv.Itoa(i+1) + " " + chip.Title)
		if chip.Current {
			text += "*"
		}
		out = append(out, text)
	}
	return strings.Join(out, " | ")
}

// TestSessionSlotSwitchKeepsConversations: a switch parks the current
// slot's card, the next slot starts its own conversation (no earlier
// turns go to the model), and switching back brings the first card and
// its conversation back, where a follow-up continues.
func TestSessionSlotSwitchKeepsConversations(t *testing.T) {
	chat := &fakeChat{texts: []string{"one", "two", "three"}}
	a, tr, _ := rewriteSelecting(t, chat, chatA, "summarize")
	conv1 := a.answer.conv

	a.newSession()
	if a.slot != 1 || a.answer.open || len(a.slots) != 2 {
		t.Fatalf("switched: slot=%d answer=%+v sessions=%d", a.slot, a.answer, len(a.slots))
	}
	if p := a.slots[0]; !p.answer.open || p.answer.text != "one" || len(p.answer.turns) != 1 {
		t.Fatalf("parked: %+v", p.answer)
	}
	if a.notice != nil {
		t.Fatalf("a switch shows a notice: %+v", a.notice)
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
	a.newSession()
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
	if c := a.answer; !c.open || c.text != full || len(c.turns) != 1 {
		t.Fatalf("session 1's card %+v", c)
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
	a.newSession()
	if a.bgID == "" || a.slot != 1 {
		t.Fatalf("switched mid-stream: slot=%d bg=%q", a.slot, a.bgID)
	}
	a.mgr.StopLLM(a.bgID) // the answer ends here, kept (#3886)
	a.pump("the answer's end", func() bool { return a.bgID == "" })
	if a.answer.open || a.ov.mode == ovAnswer {
		t.Fatalf("Session 2 shows the card: %+v mode=%s", a.answer, a.ov.mode)
	}
	p := a.slots[0]
	if !p.answer.open || p.answer.text != full || len(p.answer.turns) != 1 {
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
	a.newSession()
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
	a.newSession()
	if a.slot != 0 || len(a.slots) != 1 || a.sess.State != session.Recording {
		t.Fatalf("switch during recording: slot=%d sessions=%d state=%s", a.slot, len(a.slots), a.sess.State)
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
	a.newSession()
	a.answer.inserting = false
	if a.slot != 0 || len(a.slots) != 1 {
		t.Fatalf("switch during insert: slot=%d sessions=%d", a.slot, len(a.slots))
	}
}

// TestSessionSlotsDefaultUnchanged: with sessions off, setAside and
// resurface behave as before (#4013), the card has no session row and
// is as high as it was.
func TestSessionSlotsDefaultUnchanged(t *testing.T) {
	chat := &fakeChat{texts: []string{"one", "two"}}
	a, tr, _ := rewriteSelecting(t, chat, chatA, "summarize")
	rewriteSaid(t, a, tr, textEditApp, "write a haiku", "two")
	if !a.answer.open || a.answer.text != "two" || !a.under.open || a.under.text != "one" {
		t.Fatalf("set aside: answer=%q under=%q", a.answer.text, a.under.text)
	}
	a.closeAnswer()
	slackCardBack(t, a, "resurfaced")
	if v := a.cachedCard(); len(v.Sessions) != 0 || a.boxViewState().Sessions != nil {
		t.Fatalf("session row without sessions: %+v", v.Sessions)
	}
	_, off, _ := a.cardSize()
	sessionsOn(a)
	a.overlaySync()
	// One session: the row shows already, CardSessionsH taller, so its
	// "+" can add the next one.
	if _, on, _ := a.cardSize(); on != off+layout.CardSessionsH || len(a.cachedCard().Sessions) != 1 {
		t.Fatalf("one session: card height %d, %d without sessions, row %q", on, off, chipLine(a))
	}
	// A second one keeps the card's height.
	a.slots = append(a.slots, parkedSlot{id: 9, title: "A long title here"})
	a.overlaySync()
	if _, on, _ := a.cardSize(); on != off+layout.CardSessionsH {
		t.Fatalf("card height %d with the session row, %d without", on, off)
	}
	// The other session's title is cut short; the tooltip has all of it.
	chips := a.boxViewState().Sessions
	if len(chips) != 2 || chips[1].Title != "A long t…" || chips[1].Full != "A long title here" || chips[0].Full != "" {
		t.Fatalf("session row %+v", chips)
	}
	// A long name of the current session is cut too, so the row stays short.
	a.nameSession(a.slot, "A name that is much longer than a chip")
	if chip := a.boxViewState().Sessions[0]; chip.Title != "A name that is muc…" || chip.Full != "A name that is much longer than a chip" {
		t.Fatalf("long name %+v", chip)
	}
}

// TestSessionListNewDeleteNext: "+" adds sessions without a limit and,
// from the Rewrite card, opens each in that card, idle, with no notice; the next-session shortcut walks the
// list and around; "×" deletes, the current one leaving for its
// neighbour and the last one for a new empty session.
func TestSessionListNewDeleteNext(t *testing.T) {
	chat := &fakeChat{texts: []string{"one"}}
	a, _, _ := rewriteSelecting(t, chat, chatA, "summarize")
	sessionsOn(a)
	for range 4 {
		a.newSession()
	}
	if got := chipLine(a); got != "1 | 2 | 3 | 4 | 5*" {
		t.Fatalf("five sessions: %q", got)
	}
	if !a.boxIdle() || a.notice != nil {
		t.Fatalf("empty session: Rewrite card open=%v idle=%v, notice=%+v", a.box.open, a.boxIdle(), a.notice)
	}
	// The idle card has nothing to copy or redo: c and r stay with the
	// app in front.
	if a.cardHotkeyGeneration() != 0 {
		t.Fatal("the empty session card binds the card's letter keys")
	}
	for _, want := range []int{0, 1, 2} {
		tap(a, "session-next", textEditApp)
		if a.slot != want {
			t.Fatalf("next: session %d, want %d", a.slot, want)
		}
	}
	tap(a, "session-1", textEditApp)
	slackCardBack(t, a, "session 1")

	// Deleting another session keeps the current one; deleting the
	// current one goes to its neighbour, with its card.
	a.deleteSession(3)
	if got := chipLine(a); got != "1* | 2 | 3 | 4" {
		t.Fatalf("after deleting session 4: %q", got)
	}
	a.switchSession(1)
	a.deleteSession(1)
	if a.slot != 2 || len(a.slots) != 3 {
		t.Fatalf("deleted the current one: session %d of %d", a.slot, len(a.slots))
	}
	a.deleteSession(2)
	a.deleteSession(4)
	slackCardBack(t, a, "the one session left")
	// The last session: its answer goes, a new idle session takes its
	// place in the Rewrite card.
	a.deleteSession(0)
	if len(a.slots) != 1 || a.answer.open || !a.boxIdle() {
		t.Fatalf("deleted the last one: %d sessions, card open=%v, Rewrite card idle=%v", len(a.slots), a.answer.open, a.boxIdle())
	}
	only := a.slot
	tap(a, "session-3", textEditApp) // no such session: nothing
	if a.slot != only {
		t.Fatalf("a shortcut for a session that is not there switched to %d", a.slot)
	}
	// Closing the idle card takes it off the screen.
	a.closeAnswer()
	if a.overlayModeNow() == ovAnswer || a.box.open {
		t.Fatal("the idle card stayed after its close")
	}
}

// TestSessionUseCounts: the user's own switches count (the shortcut, a
// session on the card, "+"), and one back to a session with a card as a
// return; a delete and turning sessions off do not.
func TestSessionUseCounts(t *testing.T) {
	chat := &fakeChat{texts: []string{"one"}}
	a, _, _ := rewriteSelecting(t, chat, chatA, "summarize")
	sessionsOn(a)
	a.newSession()                      // 1
	tap(a, "session-next", textEditApp) // 2, back to the card
	a.switchSession(1)                  // 3
	a.deleteSession(1)                  // to session 1: not counted
	a.newSession()                      // 4
	if s := a.prefs.Setup; s.SessionSwitchTotal != 4 || s.SessionReturns != 1 {
		t.Fatalf("use counted %d switches, %d returns", s.SessionSwitchTotal, s.SessionReturns)
	}
	// Turning sessions off goes back to the first session, row gone.
	a.setSessionsOn(false)
	if s := a.prefs.Setup; s.SessionSwitchTotal != 4 || s.SessionReturns != 1 {
		t.Fatalf("turning sessions off counted as a switch: %d, %d", s.SessionSwitchTotal, s.SessionReturns)
	}
	if a.slot != 0 || !a.answer.open || len(a.cachedCard().Sessions) != 0 {
		t.Fatalf("off: session=%d card=%v row=%+v", a.slot, a.answer.open, a.cachedCard().Sessions)
	}
}

// TestSessionTitleAndRename: a session is named after the head of its
// first instruction at once, then by the language model's summary of
// it; a title the user types replaces both for good, and an empty one
// brings the automatic title back. While a title is typed the cancel
// shortcut ends the typing, not the card.
func TestSessionTitleAndRename(t *testing.T) {
	chat := &fakeChat{texts: []string{"one", "two", "\"Autumn rain haiku.\"\nIt is about rain.", "three"}}
	a, tr, _ := rewriteSelecting(t, chat, chatA, "summarize")
	sessionsOn(a)
	a.newSession()
	rewriteSaid(t, a, tr, textEditApp, "write a haiku about the autumn rain in Kyoto", "two")
	if got := a.slots[1].titled; !strings.HasPrefix(got, "write a haiku") {
		t.Fatalf("the instruction a title is asked for: %q", got)
	}
	a.pump("the model's title", func() bool { return a.slots[1].title == "Autumn rain haiku" })
	if got := chipLine(a); got != "1 | 2 Autumn rain haiku*" {
		t.Fatalf("titled: %q", got)
	}
	// The title request is no turn of the conversation, and a later
	// instruction does not rename the session.
	rewriteSaid(t, a, tr, textEditApp, "shorter", "three")
	if got := a.slots[1].title; got != "Autumn rain haiku" || len(a.answer.turns) != 2 {
		t.Fatalf("after a follow-up: title %q, %d turns", got, len(a.answer.turns))
	}
	a.renameSession(1)
	if !a.cachedCard().Sessions[1].Renaming || a.cardHotkeyGeneration() != 0 {
		t.Fatal("renaming: the card still takes its keys")
	}
	a.hotkeyFired(proto.HotkeyEventParams{Binding: proto.HotkeyCancel, Phase: proto.PhaseTrigger})
	if a.renaming || !a.answer.open {
		t.Fatalf("cancel while renaming: renaming=%v card open=%v", a.renaming, a.answer.open)
	}
	a.renameSession(1)
	a.nameSession(1, "  Haiku  draft ")
	if got := chipLine(a); got != "1 | 2 Haiku draft*" {
		t.Fatalf("renamed: %q", got)
	}
	// While it is typed the field shows the automatic title, and nothing
	// typed gives that title back.
	a.renameSession(1)
	if chips := a.sessionChips(); !chips[1].Renaming || chips[1].Title != a.slots[1].title {
		t.Fatalf("renaming: %+v", chips)
	}
	a.nameSession(1, " ")
	if got := chipLine(a); got != strings.TrimSpace("1 | 2 "+a.slots[1].title)+"*" || a.slots[1].name != "" || a.renaming {
		t.Fatalf("empty name: %q renaming=%v", got, a.renaming)
	}
	// The only session is renamed from the menu bar: its card comes on
	// screen with the row.
	a.deleteSession(0)
	if got := len(a.sessionChips()); got != 1 {
		t.Fatalf("one session: %d chips", got)
	}
	a.renameCurrentSession()
	if chips := a.sessionChips(); len(chips) != 1 || !chips[0].Renaming || !(a.answer.open || a.emptyCard) {
		t.Fatalf("rename from the menu: %+v open=%v empty=%v", chips, a.answer.open, a.emptyCard)
	}
	a.nameSession(a.slot, "Only one")
	if labels, _ := menuLabels(a); chipLine(a) != "1 Only one*" || labels[1] != "1 Only one*" {
		t.Fatalf("named: row %q menu %q", chipLine(a), labels)
	}
	a.newSession()
	a.switchSession(a.slots[0].id)
	// A title that arrives for a deleted session is dropped.
	late := a.slots[1].id
	a.slots[1].titled = "x"
	a.deleteSession(late)
	a.setSessionTitle(late, "x", "Late")
	if got := chipLine(a); strings.Contains(got, "Late") || len(a.sessionChips()) != 1 {
		t.Fatalf("after the delete: %q", got)
	}
}

// TestSessionSlotCloseLeavesOtherSlotStreaming: closing Session 2's card
// while Session 1's answer streams in the background closes that card
// only; the stream goes on and finishes in Session 1.
func TestSessionSlotCloseLeavesOtherSlotStreaming(t *testing.T) {
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
	a.newSession()
	if a.bgID == "" || a.slot != 1 {
		t.Fatalf("switched mid-stream: slot=%d bg=%q", a.slot, a.bgID)
	}
	id := a.bgID
	// Session 2 has an ended card of its own, and a card waiting under it.
	a.answer = answerCard{open: true, id: "other", kind: store.KindTranslate, text: "bonjour"}
	a.under = answerCard{open: true, id: "under", kind: store.KindTranslate, text: "hallo"}
	if a.writingAnswer() {
		t.Fatal("Session 2's card reads as Session 1's answer being written")
	}
	a.closeAnswer()
	if a.answer.id != "under" {
		t.Fatalf("the card under did not come back: %+v", a.answer)
	}
	a.closeAnswer()
	if a.answer.open || a.bgID != id || !a.bgSess.State.Live() {
		t.Fatalf("after the close: card open=%v bg=%q live=%v", a.answer.open, a.bgID, a.bgSess.State.Live())
	}

	// Session 1's stream is still there to stop, and ends as its card.
	tap(a, "session-1", textEditApp)
	a.pump("the stream visible again", func() bool { return a.ov.mode == ovAnswer })
	a.hotkeyFired(proto.HotkeyEventParams{Binding: proto.HotkeyCancel, Phase: proto.PhaseTrigger})
	a.pump("the answer's end", func() bool { return a.bgID == "" })
	if c := a.answer; !c.open || c.text != full {
		t.Fatalf("session 1's card %+v", c)
	}
}

// TestSessionSlotSwitchBackKeepsRewriteCard: a Rewrite answer comes back
// in the Rewrite card, as in 2.1.3's single session: the input to type
// or say the next change, not the answer card with Continue.
func TestSessionSlotSwitchBackKeepsRewriteCard(t *testing.T) {
	chat := &fakeChat{texts: []string{"one", "two"}}
	a, tr, _ := rewriteSelecting(t, chat, chatA, "summarize")
	if !a.boxHoldsCard(ovAnswer) {
		t.Fatalf("the first answer is not in the Rewrite card: box %+v", a.box.open)
	}
	a.newSession()
	rewriteSaid(t, a, tr, textEditApp, "write a haiku", "two")
	tap(a, "session-1", textEditApp)
	slackCardBack(t, a, "back in session 1")
	if !a.boxHoldsCard(ovAnswer) {
		t.Fatalf("back in session 1, the answer card stands in for the Rewrite card: box open %v, shown %v", a.box.open, a.box.shown)
	}
	// The Rewrite card has an input and its own Insert and Copy: c and r
	// are letters typed there or in the app in front.
	if a.cardHotkeyGeneration() != 0 {
		t.Fatal("the Rewrite card binds the answer card's letter keys")
	}
}

// TestSessionTitleOfAnswer: the model's title loses its markdown and
// quotes, and a Chinese title is cut at half the letters of an English
// one, as each character is twice as wide.
func TestSessionTitleOfAnswer(t *testing.T) {
	for answer, want := range map[string]string{
		"**“我”与父亲之间的感情**\n这段话讲述了……": "“我”与父亲之间的感…",
		"## `Autumn rain` haiku.": "Autumn rain haiku",
		"这段话主要讲述了作者与父亲之间深厚的感情":    "这段话主要讲述了作…",
	} {
		if got := titleOfAnswer(answer); got != want {
			t.Errorf("titleOfAnswer(%q) = %q, want %q", answer, got, want)
		}
	}
}
