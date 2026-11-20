package main

import (
	"strings"
	"testing"

	"github.com/egoist/mygo"
	"github.com/sgl-project/sglang-omni/Voxt/internal/compose"
	"github.com/sgl-project/sglang-omni/Voxt/internal/helper/proto"
	"github.com/sgl-project/sglang-omni/Voxt/internal/overlay/layout"
	"github.com/sgl-project/sglang-omni/Voxt/internal/session"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

// TestRewriteBoxIsTheOnlyCard: from the shortcut to the answer, Rewrite
// shows one card (wall 122). The input card stays, with no pill beside
// or after it, while the speech is recognised and the model works; the
// answer is then written in that card, and no answer window opens.
func TestRewriteBoxIsTheOnlyCard(t *testing.T) {
	a, _, _ := newTestApp(t)
	a.box = rewriteBox{open: true, shown: true, session: "r1"}
	for _, state := range []session.State{session.Recording, session.Recognizing, session.Composing} {
		snap(a, session.Snapshot{ID: "r1", Kind: store.KindRewrite, State: state, Card: true, Spoken: "make it shorter"})
		if !a.box.shown || a.overlayModeNow() != ovHidden {
			t.Fatalf("%s: the input card shows %v, the overlay %q", state, a.box.shown, a.overlayModeNow())
		}
	}
	if view := a.boxViewState(); view.VoiceLine != "make it shorter" || view.Subtitle == "" || view.Conversation {
		t.Errorf("the working card: %+v", view)
	}

	snap(a, session.Snapshot{ID: "r1", Kind: store.KindRewrite, State: session.Composing, Card: true, Draft: "Shorter."})
	if !a.box.shown || !a.boxHoldsCard(a.overlayModeNow()) || !a.boxViewState().Conversation {
		t.Fatalf("the answer: the input card shows %v, the overlay %q", a.box.shown, a.overlayModeNow())
	}
	if card := a.boxCardView(); card == nil || !card.Writing || sessionCardBlockCount(card) == 0 {
		t.Errorf("the card writes %+v", card)
	}
}

// TestRewriteBoxResendKeepsTheCard: sending again after a failure keeps
// the input card on screen (the pill is hidden for its session, so
// hiding the card left nothing).
func TestRewriteBoxResendKeepsTheCard(t *testing.T) {
	chat := &fakeChat{status: 400}
	a := newBoxApp(t, boxScript(t), chat)
	a.pump("the failure", func() bool { return a.box.shown && a.box.err != "" })
	a.postQueue(a.boxSend)
	a.pump("the resend", func() bool { return a.box.sending != "" })
	if !a.box.shown || a.overlayModeNow() != ovHidden {
		t.Errorf("the resend: the input card shows %v, the overlay %q", a.box.shown, a.overlayModeNow())
	}
}

// TestRewriteCardSizeIsTheSettingOnly: the card is as wide as Rewrite's
// card size and always as high, whatever it holds; its size button steps
// through the sizes (wall 122: the cards differed at the Fill size).
func TestRewriteCardSizeIsTheSettingOnly(t *testing.T) {
	a, _, _ := newTestApp(t)
	a.box = rewriteBox{open: true, shown: true, area: ovTestArea}
	empty := a.boxFrame()
	a.answer = answerCard{open: true, id: "r1", kind: store.KindRewrite, text: strings.Repeat("A long answer. ", 400)}
	if got := a.boxFrame(); got != empty {
		t.Errorf("a long answer resized the card: %+v, empty %+v", got, empty)
	}
	for step := 1; step <= len(layout.CardSizes); step++ {
		a.boxCycleSize()
		want := step % len(layout.CardSizes)
		if got := a.prefs.Features.CardSize(string(store.KindRewrite)); got != want {
			t.Fatalf("after %d clicks the size is %d, want %d", step, got, want)
		}
		frame := a.boxFrame()
		if width := layout.CardSizes[want].Width(ovTestArea, a.prefs.Overlay.Margin) + 2*layout.Pad; frame.Width != width || frame.Height != empty.Height {
			t.Errorf("size %d: the card %+v, want %d wide and %d high", want, frame, width, empty.Height)
		}
	}
}

// TestRewriteCardInputStaysAtTheFoot: before the first answer the text
// box sits at the card's foot, the room above it empty; with an answer
// the conversation takes that room and the text box stays.
func TestRewriteCardInputStaysAtTheFoot(t *testing.T) {
	a, _, _ := newTestApp(t)
	a.box = rewriteBox{open: true, shown: true, focusText: true, session: "v1", area: ovTestArea}
	snap(a, session.Snapshot{ID: "v1", Kind: store.KindRewrite, State: session.Recording})
	tt := hudTester(a.boxView, boxWinW, boxWinH)
	frames(tt)
	editor, ok := tt.Find("More text for the rewrite")
	if !ok {
		t.Fatalf("texts %q", firstTexts(tt))
	}
	if editor.Y < boxWinH/2 {
		t.Errorf("the text box is at %.0f of %d: not at the card's foot", editor.Y, boxWinH)
	}

	a.box.session = ""
	snap(a, session.Snapshot{ID: "v1", Kind: store.KindRewrite, State: session.Done})
	a.answer = answerCard{open: true, id: "v1", conv: "v1", kind: store.KindRewrite, text: "Shorter.", conversation: true,
		turns: []compose.Turn{{Instruction: "make it shorter", Answer: compose.Answer{Content: "Shorter."}}}}
	frames(tt)
	asked, ok1 := tt.Find("make it shorter")
	answer, ok2 := tt.Find("Shorter.")
	after, ok3 := tt.Find("More text for the rewrite")
	if !ok1 || !ok2 || !ok3 {
		t.Fatalf("the answered card's texts %q", firstTexts(tt))
	}
	if asked.Y >= answer.Y || answer.Y >= after.Y || after.Y != editor.Y {
		t.Errorf("asked at %.0f, the answer at %.0f, the text box at %.0f (was %.0f)", asked.Y, answer.Y, after.Y, editor.Y)
	}
	if _, ok := tt.Find("Insert"); !ok {
		t.Errorf("no Insert in the card's header: %q", firstTexts(tt))
	}
}

// TestRewriteFollowUpTypedInTheCard: the card stays after its answer,
// and what is typed there and sent is the follow-up: no recording, no
// second window (a Mac with no microphone could not follow up). The new
// answer joins the same card, which is emptied for the next one.
func TestRewriteFollowUpTypedInTheCard(t *testing.T) {
	chat := &fakeChat{text: `{"title":"Help","content":"Sure: tell me the problem."}`}
	a, _ := rewriteOnce(t, "", chat, "help me with a problem")
	first := a.sess.ID
	if v := a.boxViewState(); !a.box.open || !a.box.shown || !v.Conversation || v.Text != "" || v.VoiceLine != "" {
		t.Fatalf("after the answer: box %+v, view %+v", a.box, v)
	}
	if card := a.boxCardView(); card == nil || !card.CanInsert || sessionCardBlockCount(card) == 0 {
		t.Fatalf("the card's answer %+v", card)
	}
	chat.mu.Lock()
	chat.text = "Tell me the problem, please."
	chat.mu.Unlock()
	a.boxActions().textChanged("make it polite")
	if !a.boxViewState().CanSend {
		t.Error("no Send for the typed follow-up")
	}
	a.boxSend()
	a.pump("the follow-up's end", func() bool { return a.sess.ID != first && a.sess.State.Ended() })
	if len(a.sess.Levels) != 0 {
		t.Errorf("the typed follow-up recorded: %d levels", len(a.sess.Levels))
	}
	reqs := chat.requests()
	if got, want := lastUserOf(reqs[len(reqs)-1]), "<instruction>\nmake it polite\n</instruction>"; got != want {
		t.Errorf("the follow-up asked %q, want %q", got, want)
	}
	if turns := a.answer.turns; len(turns) != 2 || turns[1].Instruction != "make it polite" || turns[1].Typed != "" {
		t.Errorf("the turns %+v", turns)
	}
	if v := a.boxViewState(); !a.box.open || !a.boxHoldsCard(a.overlayModeNow()) || v.Text != "" || a.answer.text != "Tell me the problem, please." {
		t.Errorf("the end: box open %v, overlay %s, text %q, card %+v", a.box.open, a.overlayModeNow(), v.Text, a.answer)
	}

	// Close closes the one card: the answer goes with it.
	a.boxCloseCard()
	if a.box.open || a.answer.open {
		t.Errorf("after Close: box open %v, answer open %v", a.box.open, a.answer.open)
	}
}

// TestRewriteCardClosesAfterInserting: with Rewrite's card switch off the
// result goes into the app, and the card closes.
func TestRewriteCardClosesAfterInserting(t *testing.T) {
	a, _, _ := newTestApp(t)
	a.box = rewriteBox{open: true, shown: true, session: "r1"}
	snap(a, session.Snapshot{ID: "r1", Kind: store.KindRewrite, State: session.Composing})
	snap(a, session.Snapshot{ID: "r1", Kind: store.KindRewrite, State: session.Done, Result: "Shorter.", Inserted: true})
	if a.box.open || a.answer.open {
		t.Errorf("after inserting: box open %v, answer open %v", a.box.open, a.answer.open)
	}
}

// TestRewriteCardDrags: the card moves where the user drags it (its face
// drags the window), keeps that place through a change of size and a
// follow-up, stays on the screen, and opens at its usual place again once
// it was closed, as the answer card does (TestCardDragged).
func TestRewriteCardDrags(t *testing.T) {
	a, _, _ := newTestApp(t)
	native := &fakeBoxDropWindow{}
	previous := newBoxNativeWindow
	newBoxNativeWindow = func(mygo.WindowOptions) boxDropWindow { return native }
	t.Cleanup(func() { newBoxNativeWindow = previous })
	a.box = rewriteBox{open: true, shown: true, winShown: true, area: ovTestArea}
	a.box.win = newBoxWindow(a.boxViewState, a.boxActions(), &a.box.ui, floatingWindowOptions("Rewrite", boxWinW, boxWinH, nil))
	anchored := a.boxFrame()
	a.box.frame = anchored
	if native.onMove == nil {
		t.Fatal("the window reports no move")
	}
	native.bounds = mygo.Rectangle{X: anchored.X - 300, Y: anchored.Y - 200, Width: anchored.Width, Height: anchored.Height}
	native.onMove()
	if f := a.boxFrame(); f != native.bounds {
		t.Fatalf("dragged to %+v, the card's frame is %+v", native.bounds, f)
	}

	a.boxCycleSize()
	if f := a.boxFrame(); f.Width == anchored.Width || f.Y != anchored.Y-200 {
		t.Errorf("another size moved the dragged card: %+v, anchored %+v", f, anchored)
	}
	a.boxKeep() // the answer is in: the card takes the follow-up where it is
	if f := a.boxFrame(); f.Y != anchored.Y-200 {
		t.Errorf("the follow-up moved the dragged card: %+v", f)
	}
	a.box.dragged.y = -5000
	if f := a.boxFrame(); f.Y != ovTestArea.Y-layout.Pad {
		t.Errorf("dragged past the top: %+v", f)
	}

	a.closeBox()
	a.setCardSize(string(store.KindRewrite), 0)
	a.box.open, a.box.shown, a.box.area = true, true, ovTestArea
	if f := a.boxFrame(); f != anchored {
		t.Errorf("the next card: %+v, want %+v", f, anchored)
	}
}

// TestRewriteCardIsNotAlwaysOnTop: the card stays at the windows' level,
// so the user can put another window over it. Each press of the Rewrite
// shortcut on it is a click on it and brings it in front: the press that
// asks in it and the press that sends. A refresh of what it shows does not.
func TestRewriteCardIsNotAlwaysOnTop(t *testing.T) {
	a, _, _ := newTestApp(t)
	a.ov.native = true
	native := &fakeBoxDropWindow{onTop: true}
	previous := newBoxNativeWindow
	newBoxNativeWindow = func(mygo.WindowOptions) boxDropWindow { return native }
	t.Cleanup(func() { newBoxNativeWindow = previous })
	a.box = rewriteBox{open: true, shown: true, winShown: true, area: ovTestArea} // already placed: no screen to ask in a test
	a.boxSync()
	if native.onTop || native.raised != 0 {
		t.Fatalf("the new card: always on top %v, raised %d times", native.onTop, native.raised)
	}
	a.showBox(proto.App{PID: 42}, "r2") // the Rewrite shortcut asks in the card
	if native.raised != 1 {
		t.Errorf("the shortcut raised the card %d times, want 1", native.raised)
	}
	a.boxSync()
	if native.raised != 1 {
		t.Errorf("a refresh raised the card: %d times", native.raised)
	}
	a.boxShortcutSend() // the shortcut again (or its hold let go): it sends
	if native.raised != 2 {
		t.Errorf("the shortcut's send raised the card %d times in all, want 2", native.raised)
	}
}
