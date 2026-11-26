package main

import (
	"slices"
	"testing"

	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/session"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

// TestRewriteCardShowsTheSentTurnFirst: once the first instruction is
// sent, the Rewrite card shows it, with the selection and the picture it
// sent, before any answer comes; the answer then streams under it, and
// the ended turn keeps both.
func TestRewriteCardShowsTheSentTurnFirst(t *testing.T) {
	a, _, _ := newTestApp(t)
	jpeg := boxJPEGOf(t, 20, 10)
	bitmap, _ := ui.DecodeBitmap(jpeg)
	a.box = rewriteBox{open: true, shown: true, sending: "r1", images: [][]byte{jpeg}, thumbs: []*ui.Bitmap{bitmap},
		imageInfo: []boxImageInfo{{Source: "draft.png", Width: 20, Height: 10}}}
	sent := session.Snapshot{ID: "r1", Kind: store.KindRewrite, State: session.Composing, Card: true,
		Raw: "describe this", Selection: "source text", ImageHint: "box"}
	userTurn := func(when string) {
		t.Helper()
		v := a.boxViewState()
		if !v.Conversation || v.Card == nil || sessionCardBlockCount(v.Card) != 2 {
			t.Fatalf("%s: the card shows no turn: %+v", when, v.Card)
		}
		user, answer := sessionCardBlockAt(v.Card, 0), sessionCardBlockAt(v.Card, 1)
		if !user.User || user.Text != "describe this" || user.Quote != "source text" || len(user.Images) != 1 || answer.User {
			t.Fatalf("%s: user side %+v, then %+v", when, user, answer)
		}
		if previews := a.ov.ui.cardPreviews; user.Images[0] >= len(previews) || previews[user.Images[0]].Label != "draft.png" {
			t.Fatalf("%s: the bubble's picture is not the one sent: %+v", when, previews)
		}
	}
	snap(a, sent)
	userTurn("sent, no answer yet")
	if v := a.boxViewState(); len(v.Thumbnails) != 0 || v.Selection != "" {
		t.Fatalf("the composer still holds what the turn sent: %d pictures, selection %q", len(v.Thumbnails), v.Selection)
	}
	sent.Draft = "A small"
	snap(a, sent)
	userTurn("streaming")
	tt := hudTester(a.boxView, boxWinW, boxWinH)
	frames(tt)
	texts := firstTexts(tt)
	if said, answer := slices.Index(texts, "describe this"), slices.Index(texts, "A small"); said < 0 || answer < said {
		t.Fatalf("the instruction is not above the answer: %q", texts)
	}
	sent.State, sent.Result = session.Done, "A small blue box."
	snap(a, sent)
	userTurn("answered")
}
