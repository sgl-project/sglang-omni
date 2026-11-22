package main

import (
	"strings"
	"testing"

	"github.com/sgl-project/sglang-omni/Voxt/internal/overlay/layout"
	"github.com/sgl-project/sglang-omni/Voxt/internal/session"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

// TestStreamedAnswerNoScrollWhileGrowing (#3836 feedback: "the scroll bar
// keeps flashing while it writes"): a Rewrite answer streams into a card
// under the pointer. As on the Mac, the view measures a new text in the
// frame that first draws it, and the card grows to it a turn of the main
// loop later. Every frame, while the card still has room to grow, its
// body neither scrolls (under the pointer its scroll bar flashed, and
// following the end the text jumped a line up and back) nor has text
// past its bottom: the card grew before the text got there.
func TestStreamedAnswerNoScrollWhileGrowing(t *testing.T) {
	for _, c := range []struct{ name, line string }{
		{"en", "The meeting moves to Thursday at three, and the room is the big one upstairs. "},
		{"zh", "会议改到周四下午三点，地点是楼上那间大会议室，请大家提前十分钟到场。"},
	} {
		t.Run(c.name, func(t *testing.T) { streamUnderPointer(t, strings.Repeat(c.line, 10)) })
	}
}

func streamUnderPointer(t *testing.T, answer string) {
	a, _, _ := newTestApp(t)
	a.measureHook = nil // the overlay's view measures its texts, as on the Mac
	defer func() { a.measureHook = guessText }()
	var posts []func()
	a.postQueue = func(fn func()) { posts = append(posts, fn) }
	flush := func() {
		for len(posts) > 0 {
			fn := posts[0]
			posts = posts[1:]
			fn()
		}
	}
	clock := (&fakeTimers{}).install(a)
	s := session.Snapshot{ID: "r1", Kind: store.KindRewrite, State: session.Composing, Card: true, Partial: "make it longer", Selection: "x", Draft: answer}
	snap(a, s)
	tt := hudTester(a.overlayView, layout.AnswerW+2*layout.Pad, 640)
	tt.SetScale(2)
	frame := func() { // the window as large as the card is now, as the Mac's
		tt.SetSize(a.ov.w+2*layout.Pad, a.ov.h+2*layout.Pad)
		tt.Frame()
	}
	frame()
	flush()
	tt.Move(float32(layout.Pad+layout.AnswerW/2), float32(layout.Pad+layout.CardChrome)) // the pointer rests on the card
	scrolled, past, grew := 0, 0, 0
	for i := 0; i < 2000 && len(cardTextForTest(a.cachedCard())) < len(strings.TrimSpace(answer)); i++ {
		clock.advance(revealFrame)
		frame() // the new text, in the card as high as it was
		card := a.cachedCard()
		body := a.currentSessionOverlayLayout().Height - layout.CardChrome
		room := body < a.cardBodyMax(&card)
		st := a.ov.ui.cardScroll
		flush() // the view's measures come back: the card grows
		frame()
		if !room {
			continue
		}
		if a.currentSessionOverlayLayout().Height-layout.CardChrome > body {
			grew++
		}
		if st.MaxY > 0 {
			scrolled++
			t.Logf("at %d runes the %d pt body scrolled (%.0f of %.0f pt)", len(cardTextForTest(card)), body, st.Y, st.MaxY)
		}
		if h, exact := cardTextHeight(a, &card); exact && h > body {
			past++
			t.Logf("at %d runes the text, %d pt, ran past the %d pt body", len(cardTextForTest(card)), h, body)
		}
	}
	if grew < 3 {
		t.Fatalf("the card grew %d times: the answer is too short to test", grew)
	}
	if scrolled > 0 || past > 0 {
		t.Errorf("while the card had room: its body scrolled in %d frames, its text ran past it in %d", scrolled, past)
	}
}

// cardTextHeight is how high card's text is, as the view measured it.
func cardTextHeight(a *app, card *sessionCardViewState) (int, bool) {
	whole := *card
	whole.Writing, whole.Busy = false, false // its height, not the card's steps
	measure := func(key ovSized) (float32, bool) {
		measured, ok := a.ovText(key)
		return measured.h, ok
	}
	view := &a.ov.projection.view
	width := max(layout.CardSizes[whole.SizeStep].W, layout.AnswerW) // the test's fixed sizes
	height, _, exact := sessionOverlayCardBodyHeight(view.SessionID, &whole, measure, a.sessionOverlayPreviewHeights(view), 1<<20, 0, width)
	return height, exact
}
