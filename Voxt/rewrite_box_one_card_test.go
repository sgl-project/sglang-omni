package main

import (
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/egoist/mygo"
	"github.com/egoist/mygo/transfer"
	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/compose"
	"github.com/sgl-project/sglang-omni/Voxt/internal/helper"
	"github.com/sgl-project/sglang-omni/Voxt/internal/helper/helpertest"
	"github.com/sgl-project/sglang-omni/Voxt/internal/helper/proto"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
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
	for _, state := range []session.State{session.Recording, session.Recognizing} {
		snap(a, session.Snapshot{ID: "r1", Kind: store.KindRewrite, State: state, Card: true, Partial: "make it shorter"})
		if !a.box.shown || a.overlayModeNow() != ovHidden {
			t.Fatalf("%s: the input card shows %v, the overlay %q", state, a.box.shown, a.overlayModeNow())
		}
	}
	if view := a.boxViewState(); view.VoiceLine != "make it shorter" || view.Conversation {
		t.Errorf("the working card: %+v", view)
	}
	// Sent: the card shows the turn, its answer on the way, still alone.
	snap(a, session.Snapshot{ID: "r1", Kind: store.KindRewrite, State: session.Composing, Card: true, Spoken: "make it shorter"})
	if !a.box.shown || !a.boxHoldsCard(a.overlayModeNow()) || !a.boxViewState().Conversation {
		t.Fatalf("sent: the input card shows %v, the overlay %q", a.box.shown, a.overlayModeNow())
	}
	if card := a.boxCardView(); card == nil || sessionCardBlockCount(card) != 2 || sessionCardBlockAt(card, 0).Text != "make it shorter" {
		t.Fatalf("sent: the card does not show what was asked: %+v", card)
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

// TestRewriteNativePasteAfterFollowUpKeepsOnlyNewInput: a completed request's
// typed segments must not reappear when the next follow-up pastes a block.
func TestRewriteNativePasteAfterFollowUpKeepsOnlyNewInput(t *testing.T) {
	chat := &fakeChat{texts: []string{"First answer", "Second answer"}}
	a := newBoxApp(t, boxScript(t), chat)
	a.pump("the recording", func() bool { return a.box.open && a.sess.State == session.Recording })
	tt := hudTester(a.boxView, boxWinW, boxWinH)
	frames(tt)
	long := "hello world\nsecond\nthird\nfourth"
	paste := func(text string) {
		tt.SetClipboardData(transfer.TextData(text))
		tt.Command("paste")
		frames(tt)
	}
	tt.Type("keep me")
	paste(long)
	tt.Type("after")
	frames(tt)
	a.pump("the first answer", func() bool { return a.answer.open && a.sess.State.Ended() })
	requests := chat.requests()
	if len(requests) != 1 || !strings.Contains(lastUserOf(requests[0]), "keep me\n"+long+"\nafter") {
		t.Fatalf("the first request did not keep the typed and pasted input: %q", requests)
	}

	frames(tt)
	paste("hello")
	paste(strings.Repeat("x", (32<<20)+1))
	if a.box.err == "" {
		t.Fatal("the oversize paste was not rejected")
	}
	paste(long)
	tt.Type("after native")
	frames(tt)
	a.boxSend()
	a.pump("the follow-up answer", func() bool { return len(chat.requests()) == 2 && a.sess.State.Ended() })

	requests = chat.requests()
	want := "<instruction>\nhello\n" + long + "\nafter native\n</instruction>"
	if got := lastUserOf(requests[1]); got != want {
		t.Fatalf("the follow-up request %q, want %q", got, want)
	}
}

// TestRewriteTypedFollowUpSurvivesStartRefusal: a missing helper refuses
// the start. The typed follow-up stays in the card for another attempt.
func TestRewriteTypedFollowUpSurvivesStartRefusal(t *testing.T) {
	chat := &fakeChat{text: "First answer"}
	a, _ := rewriteOnce(t, "", chat, "help me with a problem")
	a.hstatus.State = helper.Stopped
	a.boxActions().textChanged("make it polite")
	a.boxSend()
	if a.notice == nil {
		t.Fatal("the follow-up start was not refused")
	}
	if got := a.boxTextContent(); got != "make it polite" {
		t.Errorf("the refused start erased the typed follow-up: %q", got)
	}
}

// TestRewriteTypingDuringFollowUpSurvivesItsEnd: text typed after a
// follow-up went to the model belongs to the next one. Finishing the
// request, or cancelling it with Escape, must not erase that text.
func TestRewriteTypingDuringFollowUpSurvivesItsEnd(t *testing.T) {
	for _, cancelled := range []bool{false, true} {
		name := "done"
		if cancelled {
			name = "escape"
		}
		t.Run(name, func(t *testing.T) {
			chat := &fakeChat{text: "First answer"}
			a, _ := rewriteOnce(t, "", chat, "help me with a problem")
			first := a.sess.ID
			chat.set("Second answer", 0)
			chat.mu.Lock()
			chat.stall = cancelled
			if !cancelled {
				chat.delay = time.Second
			}
			chat.mu.Unlock()
			a.boxActions().textChanged("first follow-up")
			a.boxSend()
			a.pump("the follow-up request", func() bool { return len(chat.requests()) == 2 })
			a.pump("the follow-up card", func() bool {
				return a.sess.ID != first && a.sess.ID == a.boxSessionID() && a.sess.State.Live()
			})
			a.boxActions().textChanged("typed while waiting")
			if cancelled {
				a.boxEscape()
			}
			a.pump("the follow-up end", func() bool { return a.sess.ID != first && a.sess.State.Ended() })
			if got := a.boxTextContent(); got != "typed while waiting" {
				t.Errorf("the follow-up end erased the next typed input: %q", got)
			}
			if !a.box.open || !a.answer.open {
				t.Errorf("the one card closed: box %v, answer %v", a.box.open, a.answer.open)
			}
		})
	}
}

// TestRewriteTypedFollowUpRetryKeepsClearOnEnd: a resend keeps its
// original clear-on-end behavior, including text typed while the first
// attempt was running. No new edits are made during the resend.
func TestRewriteTypedFollowUpRetryKeepsClearOnEnd(t *testing.T) {
	for _, cancelled := range []bool{false, true} {
		name := "done"
		if cancelled {
			name = "escape"
		}
		t.Run(name, func(t *testing.T) {
			chat := &fakeChat{text: "First answer"}
			a, _ := rewriteOnce(t, "", chat, "help me with a problem")
			first := a.sess.ID
			chat.set("", 401)
			chat.mu.Lock()
			chat.delay = time.Second
			chat.mu.Unlock()
			a.boxActions().textChanged("first follow-up")
			a.boxSend()
			a.pump("the failed follow-up request", func() bool { return len(chat.requests()) == 2 })
			a.pump("the failed follow-up card", func() bool {
				return a.sess.ID != first && a.sess.ID == a.boxSessionID() && a.sess.State.Live()
			})
			a.boxActions().textChanged("typed while waiting")
			a.pump("the failed follow-up", func() bool { return a.sess.ID != first && a.sess.State == session.Failed })
			if got := a.boxTextContent(); got != "typed while waiting" {
				t.Fatalf("the failure lost the editor setup: %q", got)
			}

			failed := a.sess.ID
			chat.set("Recovered", 0)
			chat.mu.Lock()
			chat.delay = 0
			chat.stall = cancelled
			chat.mu.Unlock()
			a.boxSend()
			a.pump("the retry request", func() bool { return len(chat.requests()) == 3 })
			a.pump("the retry card", func() bool {
				return a.sess.ID != failed && a.sess.ID == a.boxSessionID() && a.sess.State.Live()
			})
			if cancelled {
				a.boxEscape()
			}
			a.pump("the retry end", func() bool { return a.sess.ID != failed && a.sess.State.Ended() })
			if got := a.boxTextContent(); got != "" {
				t.Errorf("the resend kept its sent input: %q", got)
			}
		})
	}
}

// TestRewriteTypedFollowUpSurvivesNoLLM: a typed follow-up can fail
// before the write step copied Raw into Spoken. Raw still restores the
// instruction for the retry.
func TestRewriteTypedFollowUpSurvivesNoLLM(t *testing.T) {
	a, _, _ := newTestApp(t)
	a.answer = answerCard{open: true, id: "first", kind: store.KindRewrite, text: "First answer"}
	a.followUp = "first"
	a.box = rewriteBox{open: true, shown: true, session: "follow"}
	a.boxEnded(session.Snapshot{ID: "follow", State: session.Failed, Raw: "make it polite",
		Problem: &session.Problem{Code: session.ProblemNoLLM, Err: errors.New("no model")}})
	if got := a.boxVoiceLine(); got != "make it polite" {
		t.Errorf("the early failure erased the typed follow-up: %q", got)
	}
}

// TestRewriteKeepsTheLiveTextWhileRecognising: once the shortcut ends, the
// card keeps the words heard so far until the recognition finishes; it shows
// "Recognising…" only when nothing was heard live (VX-141).
func TestRewriteKeepsTheLiveTextWhileRecognising(t *testing.T) {
	a, _, _ := newTestApp(t)
	a.box = rewriteBox{open: true, shown: true, session: "r1"}
	snap(a, session.Snapshot{ID: "r1", Kind: store.KindRewrite, State: session.Recording, Partial: "make it polite"})
	snap(a, session.Snapshot{ID: "r1", Kind: store.KindRewrite, State: session.Recognizing, Partial: "make it polite"})
	if got := a.boxVoiceLine(); got != "make it polite" {
		t.Errorf("recognising replaced the live text with %q", got)
	}
	snap(a, session.Snapshot{ID: "r1", Kind: store.KindRewrite, State: session.Recognizing})
	if got := a.boxVoiceLine(); got != i18n.T("Recognising…") {
		t.Errorf("with nothing heard live the card shows %q", got)
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
// drags the window), keeps that place through a change of size, a follow-up,
// and close and reopen, and stays on the screen (TestCardDragged).
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
	current := a.boxFrame()
	native.bounds = mygo.Rectangle{X: current.X, Y: anchored.Y - 5000, Width: current.Width, Height: current.Height}
	native.onMove()
	if f := a.boxFrame(); f.Y != ovTestArea.Y-layout.Pad {
		t.Errorf("dragged past the top: %+v", f)
	}

	a.setCardSize(string(store.KindRewrite), 0)
	moved := a.boxFrame()
	a.closeBox()
	a.showBox(proto.App{}, "next")
	a.box.area = ovTestArea
	if f := a.boxFrame(); f != moved {
		t.Errorf("the next card: %+v, want %+v", f, moved)
	}
}

// TestRewriteCardTakesTheFront: each press of the Rewrite shortcut on the
// card is a click on it. Pressed in WeChat, the card showed under WeChat's
// window: macOS keeps an ordinary window of an app that is not in front
// under the front app's. The card now floats above every window at once,
// and VoxtNew comes in front with it once the selection of the app in
// front is read (it may get a ⌘C); the card is at the windows' level then,
// where the user's next app covers it and a window manager moves it. A
// refresh does not raise it, and a user who went to another app while the
// selection was read keeps the front.
func TestRewriteCardTakesTheFront(t *testing.T) {
	a, _, _ := newTestApp(t)
	a.ov.native = true
	native := &fakeBoxDropWindow{onTop: true}
	previous, front := newBoxNativeWindow, frontmostPID
	newBoxNativeWindow = func(mygo.WindowOptions) boxDropWindow { return native }
	frontmostPID = func() int { return 42 } // the user is in WeChat
	t.Cleanup(func() { newBoxNativeWindow, frontmostPID = previous, front })
	a.box = rewriteBox{open: true, shown: true, winShown: true, area: ovTestArea} // already placed: no screen to ask in a test
	a.boxSync()
	if !native.onTop || native.raised != 0 || native.focused != 0 {
		t.Fatalf("the new card: on top %v, raised %d, focused %d", native.onTop, native.raised, native.focused)
	}

	a.sess = session.Snapshot{ID: "r2", Kind: store.KindRewrite, State: session.Recording, ReadingSelection: true}
	a.showBox(proto.App{PID: 42}, "r2") // the shortcut, while the selection is read
	if !native.onTop || native.raised != 1 || native.focused != 0 {
		t.Fatalf("while the selection is read: on top %v, raised %d, focused %d", native.onTop, native.raised, native.focused)
	}
	a.sess.ReadingSelection = false
	a.boxSync()
	if native.onTop || native.focused != 1 {
		t.Fatalf("once it is read: on top %v, focused %d", native.onTop, native.focused)
	}
	a.boxSync()
	if native.raised != 1 || native.focused != 1 {
		t.Errorf("a refresh raised the card: raised %d, focused %d", native.raised, native.focused)
	}

	a.sess = session.Snapshot{}
	a.boxShortcutSend() // the shortcut again (or its hold let go): it sends
	if native.onTop || native.raised != 2 || native.focused != 2 {
		t.Errorf("the shortcut's send: on top %v, raised %d, focused %d", native.onTop, native.raised, native.focused)
	}

	a.sess = session.Snapshot{ID: "r3", Kind: store.KindRewrite, State: session.Recording, ReadingSelection: true}
	a.showBox(proto.App{PID: 42}, "r3")
	frontmostPID = func() int { return 77 } // the user went to another app
	a.sess.ReadingSelection = false
	a.boxSync()
	if native.onTop || native.focused != 2 {
		t.Errorf("the user went elsewhere: on top %v, focused %d", native.onTop, native.focused)
	}
}

// TestRewriteCardOverAFullScreenApp: the wish wall's "Rewrite switched my
// full-screen app back to the desktop". In another app's full screen the
// card's ordinary window is not on the user's Space (macOS shows no
// ordinary window of an app with a Dock icon there), and VoxtNew taking the
// front with it switched the user to a desktop. The card is a panel
// there instead: it shows over the full-screen app and takes the keyboard
// without activating VoxtNew, and it floats (at the windows' level the app
// in front would cover it). It stays a panel until it closes; the next card
// on a desktop is an ordinary window again, and a card left open on a
// desktop becomes a panel when the shortcut is pressed in full screen.
func TestRewriteCardOverAFullScreenApp(t *testing.T) {
	a, _, _ := newTestApp(t)
	a.ov.native = true
	var made []*fakeBoxDropWindow
	var options []mygo.WindowOptions
	fullScreen := true
	previous, front := newBoxNativeWindow, frontmostPID
	newBoxNativeWindow = func(o mygo.WindowOptions) boxDropWindow {
		w := &fakeBoxDropWindow{onTop: o.AlwaysOnTop}
		w.offSpace = fullScreen && !o.Panel // macOS: a panel shows there
		made, options = append(made, w), append(options, o)
		return w
	}
	frontmostPID = func() int { return 42 } // the full-screen app
	t.Cleanup(func() { newBoxNativeWindow, frontmostPID = previous, front })

	a.box = rewriteBox{open: true, shown: true, winShown: true, area: ovTestArea} // already placed: no screen to ask in a test
	a.sess = session.Snapshot{ID: "r1", Kind: store.KindRewrite, State: session.Recording, ReadingSelection: true}
	a.showBox(proto.App{PID: 42}, "r1") // the shortcut, in the full-screen app
	if len(made) != 2 || !made[0].destroyed || made[0].focused != 0 || made[0].raised != 0 {
		t.Fatalf("the ordinary window: %d windows, first destroyed %v, raised %d, focused %d", len(made), len(made) > 0 && made[0].destroyed, made[0].raised, made[0].focused)
	}
	panel := made[1]
	if !options[1].Panel || options[1].NonActivating || !a.box.panel || a.box.win != boxWindow(panel) {
		t.Fatalf("the card over the full-screen app: options %+v, panel %v", options[1], a.box.panel)
	}
	if panel.raised != 1 || panel.focused != 0 || panel.style != a.box.style || panel.bounds != a.box.frame {
		t.Fatalf("the panel while the selection is read: raised %d, focused %d, style %+v, bounds %+v", panel.raised, panel.focused, panel.style, panel.bounds)
	}
	a.sess.ReadingSelection = false
	a.boxSync()
	if panel.focused != 1 || !panel.onTop || !a.box.focusText {
		t.Fatalf("once the selection is read the panel takes the keyboard and floats: focused %d, on top %v, text focus %v", panel.focused, panel.onTop, a.box.focusText)
	}
	a.boxKeep() // the answer is in: the follow-up stays in the panel
	a.boxShortcutSend()
	if len(made) != 2 || !a.box.panel || panel.raised != 2 || !panel.onTop {
		t.Errorf("a follow-up: %d windows, panel %v, raised %d, on top %v", len(made), a.box.panel, panel.raised, panel.onTop)
	}

	a.closeBox()
	if !panel.destroyed || a.box.panel {
		t.Errorf("closing: panel destroyed %v, still a panel %v", panel.destroyed, a.box.panel)
	}
	fullScreen = false // back on a desktop
	a.sess = session.Snapshot{}
	a.box = rewriteBox{open: true, shown: true, winShown: true, area: ovTestArea}
	a.showBox(proto.App{PID: 42}, "r2")
	if len(made) != 3 || options[2].Panel || a.box.panel || made[2].focused != 1 || made[2].onTop {
		t.Fatalf("the next card on a desktop: %d windows, panel %v/%v, focused %d, on top %v", len(made), options[len(options)-1].Panel, a.box.panel, made[len(made)-1].focused, made[len(made)-1].onTop)
	}

	fullScreen, made[2].offSpace = true, true // the user went to a full-screen app, the card open
	a.box.focusText = false                   // the text box had taken the focus
	a.boxShortcutSend()
	if len(made) != 4 || !made[2].destroyed || !options[3].Panel || !a.box.panel || made[3].focused != 1 || !made[3].onTop || !a.box.focusText {
		t.Errorf("the shortcut in full screen: %d windows, old destroyed %v, panel %v, focused %d, on top %v, text focus %v",
			len(made), made[2].destroyed, a.box.panel, made[len(made)-1].focused, made[len(made)-1].onTop, a.box.focusText)
	}
}

// TestRewriteCardEnterFollowsTheReturnSwitch: while the card records,
// Enter sends only with Settings → Shortcuts' Return switch on (it is
// taken, no newline). Off, Enter is a newline there, as it reaches any app
// then, and Send's tooltip does not say that Enter sends.
func TestRewriteCardEnterFollowsTheReturnSwitch(t *testing.T) {
	a, _, _ := newTestApp(t)
	a.prefs.Features.ReturnFinishes = false
	tt := hudTester(a.boxView, boxWinW, boxWinH)
	a.box = rewriteBox{open: true, shown: true, focusText: true, session: "v1"}
	snap(a, session.Snapshot{ID: "v1", Kind: store.KindRewrite, State: session.Recording})
	frames(tt)
	tt.Type("some text")
	frames(tt)
	if a.boxViewState().EnterSends {
		t.Error("Return off: the card says Enter sends")
	}
	tt.Key(0, ui.KeyEnter)
	frames(tt)
	if a.box.text != "some text\n" {
		t.Fatalf("Return off: text %q after Enter, want a newline", a.box.text)
	}
	a.prefs.Features.ReturnFinishes = true
	frames(tt)
	if !a.boxViewState().EnterSends {
		t.Error("Return on: the card does not say Enter sends")
	}
	tt.Key(0, ui.KeyEnter)
	frames(tt)
	if a.box.text != "some text\n" {
		t.Errorf("Return on: text %q after Enter, want Enter taken", a.box.text)
	}
}

// TestRewriteCardWaitsForTheCopiedSelection: Notion item 141's lost Slack
// selection. The card opens right after the session starts, before its
// first snapshot reaches the app; it took the front then, so the ⌘C that
// copies the selection went to VoxtNew and nothing was read. It waits
// until the selection is read, then takes the front.
func TestRewriteCardWaitsForTheCopiedSelection(t *testing.T) {
	native := &fakeBoxDropWindow{onTop: true}
	previous, front := newBoxNativeWindow, frontmostPID
	newBoxNativeWindow = func(mygo.WindowOptions) boxDropWindow { return native }
	frontmostPID = func() int { return slackApp.PID }
	t.Cleanup(func() { newBoxNativeWindow, frontmostPID = previous, front }) // registered first: it runs after the app's own cleanup, which still syncs the card
	a := newBoxApp(t, helpertest.Opt("delay", "textio.selection:800"), &fakeChat{text: "A shorter text."})
	focusedAtOpen := -1
	a.postQueue(func() {
		a.ov.native = true
		a.box = rewriteBox{open: true, shown: true, winShown: true, area: ovTestArea} // already placed: no screen to ask in a test
		a.openBox(slackApp, session.StartRequest{Kind: store.KindRewrite, Target: slackApp})
		focusedAtOpen = native.focused
	})
	a.pump("the selection read", func() bool { return a.box.open && a.sess.Selection != "" })
	if focusedAtOpen != 0 {
		t.Fatalf("the card took the front while the selection was read (focused %d)", focusedAtOpen)
	}
	a.pump("the card in front", func() bool { return native.focused == 1 })
}
