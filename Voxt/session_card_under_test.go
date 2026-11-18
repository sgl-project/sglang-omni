package main

import (
	"strings"
	"testing"

	"github.com/sgl-project/sglang-omni/Voxt/internal/helper/helpertest"
	"github.com/sgl-project/sglang-omni/Voxt/internal/helper/proto"
	"github.com/sgl-project/sglang-omni/Voxt/internal/session"
)

// #4013: an answer card waits under a new session and comes back when that
// session leaves no card. The card is from Slack (rewriteSelecting); the
// user goes on in TextEdit.

var textEditApp = helpertest.Apps["com.apple.TextEdit"]

// tap presses binding once, app in front.
func tap(a *sessionApp, binding string, front proto.App) {
	a.hotkeyFired(proto.HotkeyEventParams{Binding: binding, Phase: proto.PhaseTrigger, Behavior: proto.BehaviorTap, Front: &front})
}

// startSaying taps binding in app and waits until the new session records said.
func startSaying(a *sessionApp, tr *fakeTranscriptions, binding string, app proto.App, said string) {
	tr.setText(said)
	first := a.sess.ID
	tap(a, binding, app)
	a.pump("recording", func() bool {
		return a.sess.ID != first && a.sess.State == session.Recording && a.sess.HeardAt.After(a.sess.StartedAt)
	})
	a.pump("audio", func() bool { return len(a.sess.Levels) >= 5 })
}

// rewriteSaid is one Rewrite (or follow-up) in app by the shortcut, its answer answer.
func rewriteSaid(t *testing.T, a *sessionApp, tr *fakeTranscriptions, app proto.App, said, answer string) {
	t.Helper()
	startSaying(a, tr, "rewrite/0", app, said)
	tap(a, "rewrite/0", app)
	a.pump("the end", func() bool { return a.sess.State.Ended() && a.answer.text == answer })
}

// slackCardBack checks that Slack's card shows again, as it was.
func slackCardBack(t *testing.T, a *sessionApp, when string) {
	t.Helper()
	if c := a.answer; !c.open || c.target.Name != "Slack" || c.text != "one" || len(c.turns) != 1 || a.under.open || a.overlayModeNow() != ovAnswer {
		t.Fatalf("%s: card open=%v target=%s text=%q turns=%d, under=%v, mode=%s", when, c.open, c.target.Name, c.text, len(c.turns), a.under.open, a.overlayModeNow())
	}
}

// A transcription in between: the card hides while it runs and comes back,
// and the next Rewrite in Slack still follows it up.
func TestCardBackAfterTranscribe(t *testing.T) {
	chat := &fakeChat{texts: []string{"one", "two", "three"}}
	a, tr, _ := rewriteSelecting(t, chat, chatA, "summarize")
	startSaying(a, tr, "transcribe/0", textEditApp, "dictated")
	if a.answer.open || !a.under.open || a.ov.mode != ovRecording {
		t.Fatalf("recording: card open=%v under=%v mode=%s", a.answer.open, a.under.open, a.ov.mode)
	}
	tap(a, "transcribe/0", textEditApp)
	a.pump("the end", func() bool { return a.sess.State.Ended() })
	slackCardBack(t, a, "after the transcription")
	answer := "two"
	if len(chat.requests()) > 1 { // the transcription's clean-up asked the model too
		answer = "three"
	}
	rewriteSaid(t, a, tr, a.answer.target, "shorter", answer)
	msgs := chatMessages(chat.requests()[len(chat.requests())-1])
	if len(msgs) != 4 || len(a.answer.turns) != 2 || !strings.Contains(msgs[1][1], "summarize") || !strings.Contains(msgs[3][1], "shorter") {
		t.Fatalf("follow-up: %d messages, %d turns: %q", len(msgs), len(a.answer.turns), msgs)
	}
}

// The cancel shortcut during the transcription cancels it; the card comes back.
func TestCardBackAfterCancelledTranscribe(t *testing.T) {
	chat := &fakeChat{texts: []string{"one"}}
	a, tr, _ := rewriteSelecting(t, chat, chatA, "summarize")
	startSaying(a, tr, "transcribe/0", textEditApp, "dictated")
	a.hotkeyFired(proto.HotkeyEventParams{Binding: proto.HotkeyCancel, Phase: proto.PhaseTrigger})
	a.pump("cancelled", func() bool { return a.sess.State == session.Cancelled })
	slackCardBack(t, a, "after cancel")
	if p := a.pasted(); len(p) != 0 {
		t.Fatalf("pasted %+v", p)
	}
}

// The transcription's notice (nothing heard) shows first, then the card.
func TestCardBackAfterTranscribeNotice(t *testing.T) {
	chat := &fakeChat{texts: []string{"one"}}
	a, tr, _ := rewriteSelecting(t, chat, chatA, "summarize")
	tr.setText("")
	first := a.sess.ID
	tap(a, "transcribe/0", textEditApp)
	a.pump("recording", func() bool { return a.sess.ID != first && a.sess.State == session.Recording })
	a.pump("audio", func() bool { return len(a.sess.Levels) >= 5 })
	tap(a, "transcribe/0", textEditApp)
	a.pump("the end", func() bool { return a.sess.State.Ended() })
	if a.notice == nil || a.overlayModeNow() != ovNotice || !a.answer.open {
		t.Fatalf("notice=%+v mode=%s card open=%v: want the notice, the card back behind it", a.notice, a.overlayModeNow(), a.answer.open)
	}
	a.closeNotice()
	slackCardBack(t, a, "after the notice")
}

// The Rewrite shortcut in another app is a new conversation; Slack's card
// waits and comes back when the new card is closed (twice over).
func TestRewriteElsewhereIsNewConversation(t *testing.T) {
	chat := &fakeChat{texts: []string{"one", "two", "three"}}
	a, tr, path := rewriteSelecting(t, chat, chatA, "summarize")
	selectNow(t, path, chatB)
	startSaying(a, tr, "rewrite/0", textEditApp, "translate this")
	if a.answer.open || !a.under.open {
		t.Fatalf("recording: card open=%v under=%v", a.answer.open, a.under.open)
	}
	tap(a, "rewrite/0", textEditApp)
	a.pump("the end", func() bool { return a.sess.State.Ended() && a.answer.text == "two" })
	if msgs := chatMessages(chat.requests()[1]); len(msgs) != 2 || a.answer.target.Name != "TextEdit" || len(a.answer.turns) != 1 {
		t.Fatalf("%d messages (want 2: no Slack turns), card target=%s turns=%d", len(msgs), a.answer.target.Name, len(a.answer.turns))
	}
	a.closeAnswer() // Esc
	slackCardBack(t, a, "after Esc")
	rewriteSaid(t, a, tr, textEditApp, "again", "three")
	a.closeAnswer()
	slackCardBack(t, a, "after the second Esc")
}

// Inserting the new card's answer brings Slack's card back.
func TestRewriteElsewhereInsertBringsBack(t *testing.T) {
	chat := &fakeChat{texts: []string{"one", "two"}}
	a, tr, _ := rewriteSelecting(t, chat, chatA, "summarize")
	rewriteSaid(t, a, tr, textEditApp, "translate this", "two")
	a.insertAnswer()
	a.pump("Slack's card back", func() bool { return a.answer.open && a.answer.text == "one" })
	slackCardBack(t, a, "after Insert")
	if p := a.pasted(); len(p) != 1 || p[0].Text != "two" {
		t.Fatalf("pasted %+v", p)
	}
}

// The card's Continue still follows up from another app, with what is selected there.
func TestContinueElsewhereFollowsUp(t *testing.T) {
	chat := &fakeChat{texts: []string{"one", "two"}}
	a, tr, path := rewriteSelecting(t, chat, chatA, "summarize")
	selectNow(t, path, chatB)
	a.front = textEditApp
	a.recordFollowUp(t, tr, "shorter")
	a.pump("the end", func() bool { return a.sess.State.Ended() && a.answer.text == "two" })
	msgs := chatMessages(chat.requests()[1])
	if len(msgs) != 4 || len(a.answer.turns) != 2 || a.answer.target.Name != "Slack" || a.answer.turns[1].Selection != chatB {
		t.Fatalf("%d messages, turns=%d target=%s, turn 2 selection %q", len(msgs), len(a.answer.turns), a.answer.target.Name, a.answer.turns[1].Selection)
	}
}

// History keeps each entry's conversation: a follow-up has its first
// session's ID, a new conversation its own.
func TestHistoryConversation(t *testing.T) {
	chat := &fakeChat{texts: []string{"one", "two", "three"}}
	a, tr, _ := rewriteSelecting(t, chat, chatA, "summarize")
	rewriteSaid(t, a, tr, a.answer.target, "shorter", "two")
	a.closeAnswer()
	rewriteSaid(t, a, tr, textEditApp, "decline politely", "three")
	a.pump("history", func() bool { return len(a.history()) == 3 })
	conv, id := map[string]string{}, map[string]string{}
	for _, e := range a.history() {
		conv[e.PromptText], id[e.PromptText] = e.Conversation, e.ID
	}
	first := id["summarize"]
	if first == "" || conv["summarize"] != first || conv["shorter"] != first || conv["decline politely"] != id["decline politely"] || id["decline politely"] == first {
		t.Fatalf("conversations %v, ids %v", conv, id)
	}
}
