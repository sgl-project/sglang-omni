package main

import (
	"context"
	"testing"
	"time"
	"unicode/utf8"

	"github.com/egoist/mygo/transfer"
	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/helper/helpertest"
	"github.com/sgl-project/sglang-omni/Voxt/internal/helper/proto"
	"github.com/sgl-project/sglang-omni/Voxt/internal/session"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

func TestBoxPastedTextCollapseThreshold(t *testing.T) {
	a, _, _ := newTestApp(t)
	a.box = rewriteBox{open: true, shown: true}
	if a.boxPasteText("one\ntwo\nthree") {
		t.Fatal("a three-line paste became a block")
	}
	a.box.text = "before\n"
	long := "中🙂\n二\n三\n四"
	if !a.boxPasteText(long) {
		t.Fatal("a four-line paste stayed in the editor")
	}
	a.box.text = "after"
	if got, want := a.boxTextContent(), "before\n"+long+"after"; got != want {
		t.Fatalf("request text %q, want %q", got, want)
	}
	if len(a.box.textSegments) != 2 {
		t.Fatalf("%d segments, want typed prefix plus pasted block", len(a.box.textSegments))
	}
	block := a.box.textSegments[1]
	if !block.Pasted || block.Lines != 4 || block.Chars != utf8.RuneCountInString(long) || block.Chars == len(long) || block.Text != long {
		t.Fatalf("block %+v", block)
	}
	a.boxRemoveText(1)
	if got, want := a.boxTextContent(), "before\nafter"; got != want {
		t.Fatalf("text after removing the block %q, want %q", got, want)
	}
}

func TestBoxImagePreviewMetadataAndFullView(t *testing.T) {
	a, _, _ := newTestApp(t)
	a.box = rewriteBox{open: true, shown: true}
	a.boxAttach(boxJPEGOf(t, 20, 10), "draft.png")
	state := a.boxViewState()
	if len(state.ImageInfo) != 1 || state.ImageInfo[0].Source != "draft.png" || state.ImageInfo[0].Width != 20 || state.ImageInfo[0].Height != 10 {
		t.Fatalf("image metadata %+v", state.ImageInfo)
	}
	tt := hudTester(a.boxView, boxWinW, boxWinH)
	frames(tt)
	if !has(tt, "draft.png") || !has(tt, "20 × 10") {
		t.Fatalf("caption missing from %q", firstTexts(tt))
	}
	click(t, tt, "Preview draft.png")
	frames(tt)
	if a.box.ui.previewImage != 1 || !has(tt, "Back to input") {
		t.Fatalf("full preview did not open: index %d, texts %q", a.box.ui.previewImage, firstTexts(tt))
	}
	click(t, tt, "Back to input")
	frames(tt)
	if a.box.ui.previewImage != 0 {
		t.Fatal("full preview did not close")
	}
	a.boxRemoveImage(0)
	if len(a.boxViewState().ImageInfo) != 0 {
		t.Fatal("the caption survived its image")
	}
}

func TestBoxSelectionUsesCollapsibleTextBlock(t *testing.T) {
	a, _, _ := newTestApp(t)
	a.box = rewriteBox{open: true, shown: true, session: "v1"}
	selection := "first line\nsecond line\nthird line\nfourth line"
	snap(a, session.Snapshot{ID: "v1", Kind: store.KindRewrite, State: session.Recording, Selection: selection, SelectionVia: "copy"})
	tt := hudTester(a.boxView, boxWinW, boxWinH)
	frames(tt)
	if !has(tt, "Copied text") || !has(tt, "Show all") || has(tt, "fourth line") {
		t.Fatalf("collapsed selection texts %q", firstTexts(tt))
	}
	click(t, tt, "Show all")
	frames(tt)
	if !has(tt, "fourth line") || !has(tt, "Show less") {
		t.Fatalf("expanded selection texts %q", firstTexts(tt))
	}
}

func TestBoxShortSelectionCanBeRemoved(t *testing.T) {
	for _, selection := range []string{"hello", "first\nsecond\nthird"} {
		a, _, _ := newTestApp(t)
		a.box = rewriteBox{open: true, shown: true, session: "v1"}
		snap(a, session.Snapshot{ID: "v1", Kind: store.KindRewrite, State: session.Recording, Selection: selection})
		tt := hudTester(a.boxView, boxWinW, boxWinH)
		frames(tt)
		click(t, tt, "Remove pasted text")
		frames(tt)
		if !a.box.selectionSuppressed || a.boxViewState().Selection != "" {
			t.Fatalf("short selection %q survived removal", selection)
		}
	}
}

func TestBoxTwoPastedBlocksKeepTrailingEditorVisible(t *testing.T) {
	a, _, _ := newTestApp(t)
	a.box = rewriteBox{open: true, shown: true, focusText: true, session: "v1"}
	a.boxPasteText("first\nsecond\nthird\nfourth")
	a.boxPasteText("fifth\nsixth\nseventh\neighth")
	snap(a, session.Snapshot{ID: "v1", Kind: store.KindRewrite, State: session.Recording, Selection: "selected first\nselected second\nselected third\nselected fourth"})
	tt := hudTester(a.boxView, boxWinW, boxWinH)
	frames(tt)
	editor, ok := tt.Find("More text for the rewrite")
	if !ok || editor.H <= 0 || editor.Y < 0 || editor.Y+editor.H > boxWinH {
		t.Fatalf("the trailing editor is clipped: bounds %+v, present %v, window height %d", editor, ok, boxWinH)
	}
}

func TestBoxLimitStaysVisibleWithEarlierInputs(t *testing.T) {
	a, _, _ := newTestApp(t)
	a.box = rewriteBox{open: true, shown: true, focusText: true, session: "v1", limit: true}
	a.boxPasteText("first\nsecond\nthird\nfourth")
	a.boxPasteText("fifth\nsixth\nseventh\neighth")
	a.box.thumbs = []*ui.Bitmap{shotBitmap(t), shotBitmap(t), shotBitmap(t), shotBitmap(t)}
	a.box.images = [][]byte{{1}, {2}, {3}, {4}}
	snap(a, session.Snapshot{ID: "v1", Kind: store.KindRewrite, State: session.Recording, Selection: "selected first\nselected second\nselected third\nselected fourth"})
	tt := hudTester(a.boxView, boxWinW, boxWinH)
	frames(tt)
	editor, editorOK := tt.Find("More text for the rewrite")
	limit, limitOK := tt.Find("At most 4 images")
	if !editorOK || !limitOK || limit.Y < 0 || limit.Y+limit.H > editor.Y {
		t.Fatalf("the image limit is clipped: limit %+v (present %v), editor %+v (present %v)", limit, limitOK, editor, editorOK)
	}
}

func TestAnswerCardInputChipsOpenOriginals(t *testing.T) {
	a, _, _ := newTestApp(t)
	jpeg := boxJPEGOf(t, 20, 10)
	a.box = rewriteBox{open: true, session: "r1", images: [][]byte{jpeg}, imageInfo: []boxImageInfo{{Source: "draft.png", Width: 20, Height: 10}}}
	bitmap, _ := ui.DecodeBitmap(jpeg)
	a.box.thumbs = []*ui.Bitmap{bitmap}
	long := "first line\nsecond line\nthird line\nfourth line"
	snap(a, session.Snapshot{ID: "r1", Kind: store.KindRewrite, State: session.Done, Card: true,
		Raw: "make it formal", Selection: long, SelectionVia: "copy", Typed: "typed source", Result: "Formal answer.", ImageHint: "box"})
	if len(a.answer.previews) != 3 || a.cachedCard().InputCount != 3 {
		t.Fatalf("%d previews, DTO count %d", len(a.answer.previews), a.cachedCard().InputCount)
	}
	tt := cardTester(t, a)
	frames(tt)
	click(t, tt, "Copied text")
	frames(tt)
	if !has(tt, "fourth line") || !has(tt, "Back to answer") {
		t.Fatalf("text original did not open: %q", firstTexts(tt))
	}
	click(t, tt, "Back to answer")
	frames(tt)
	click(t, tt, "draft.png")
	frames(tt)
	if !has(tt, "20 × 10") || !has(tt, "Back to answer") {
		t.Fatalf("image original did not open: %q", firstTexts(tt))
	}
}

func TestBoxWindowPreviewStaysVisibleUntilConfirmed(t *testing.T) {
	a, _, _ := newTestApp(t)
	jpeg := boxJPEGOf(t, 20, 10)
	a.box = rewriteBox{open: true, shown: true, session: "v1"}
	a.sess = session.Snapshot{ID: "v1", Kind: store.KindRewrite, State: session.Recording, Rev: 1}
	a.sessionChanged(session.Snapshot{ID: "v1", Kind: store.KindRewrite, State: session.Recognizing, Rev: 2, ImagePreviewKind: "window", ImagePreview: jpeg})
	if !a.box.shown || !a.boxViewState().CanSend {
		t.Fatal("the window preview cannot be reviewed before sending")
	}
	a.boxSend()
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	if err := a.boxShare.waitForReview(ctx); err != nil {
		t.Fatalf("Send did not confirm the preview: %v", err)
	}
	a.sessionChanged(session.Snapshot{ID: "v1", Kind: store.KindRewrite, State: session.Composing, Rev: 3})
	if a.box.shown {
		t.Fatal("the confirmed box stayed over the answer")
	}
}

func TestBoxResendWindowPreviewCanBeConfirmed(t *testing.T) {
	chat := &fakeChat{status: 400, delay: 100 * time.Millisecond}
	set, keys, _ := fileSetup(t, "make it shorter", chat)
	ev := script(t, hotkeyLine(0, "rewrite/0", proto.PhaseTrigger))
	a := newSessionApp(t, helpertest.Opt("selection", "the selected text")+","+ev, func(s *settings.Settings) {
		set(s)
		s.Privacy.SendWindow = true
	}, keys)
	a.pump("the recording", func() bool { return a.box.open && a.sess.State == session.Recording })
	a.postQueue(a.boxSend)
	a.pump("the first window review", func() bool {
		return a.sess.ID == a.box.session && a.sess.State == session.Recognizing && len(a.sess.ImagePreview) > 0
	})
	a.postQueue(func() {
		a.boxSend()
		a.boxSend()
	})
	a.pump("the first failure", func() bool { return a.box.shown && a.box.err != "" })
	sent := len(chat.requests())
	chat.set("Recovered.", 0)
	// A second click after the first review finished leaves this approval
	// for the next send unless the resend clears it.
	a.boxShare.reviewApproved = true

	a.postQueue(a.boxSend)
	a.pump("the resend's window review", func() bool {
		return a.box.sending != "" && a.sess.ID == a.box.sending && a.sess.State == session.Composing && a.sess.ImagePreviewKind == "window" && len(a.sess.ImagePreview) > 0
	})
	if !a.box.shown || !a.boxViewState().CanSend {
		t.Fatalf("the resend's screenshot cannot be confirmed: shown %v, canSend %v", a.box.shown, a.boxViewState().CanSend)
	}
	a.drain(100 * time.Millisecond)
	if got := len(chat.requests()); got != sent {
		t.Fatalf("the resend used the first send's approval: %d requests before its own confirmation, want %d", got, sent)
	}
	a.postQueue(func() {
		a.boxSend()
		if a.box.shown {
			t.Error("the confirmed resend stayed visible")
		}
	})
	a.postQueue(func() {
		next := a.sess
		next.Rev++
		next.Draft = "streaming"
		a.sessionChanged(next)
		if a.box.shown {
			t.Error("a later draft reopened the confirmed resend")
		}
	})
	a.pump("the resend's answer", func() bool { return a.answer.open })
	if a.sess.State != session.Done || a.sess.Result != "Recovered." {
		t.Fatalf("resend ended %s with %q", a.sess.State, a.sess.Result)
	}
	reqs := chat.requests()
	if len(reqs) <= sent {
		t.Fatal("the resend never reached the language model")
	}
	images, _ := reqs[len(reqs)-1]["images"].([][]byte)
	if len(images) != 1 {
		t.Fatalf("the confirmed resend sent %d images, want the window screenshot", len(images))
	}
}

func TestBoxResendWithoutWindowReviewHides(t *testing.T) {
	chat := &fakeChat{status: 400, delay: 100 * time.Millisecond}
	set, keys, _ := fileSetup(t, "make it shorter", chat)
	ev := script(t, hotkeyLine(0, "rewrite/0", proto.PhaseTrigger))
	a := newSessionApp(t, helpertest.Opt("selection", "the selected text")+","+ev, set, keys)
	a.pump("the recording", func() bool { return a.box.open && a.sess.State == session.Recording })
	a.postQueue(a.boxSend)
	a.pump("the first failure", func() bool { return a.box.shown && a.box.err != "" })
	sent := len(chat.requests())
	chat.set("Recovered.", 0)

	a.postQueue(a.boxSend)
	a.pump("the resend", func() bool { return a.box.sending != "" })
	if a.box.shown {
		t.Fatal("a resend without a window review stayed visible")
	}
	a.pump("the resend's answer", func() bool { return a.answer.open })
	if a.sess.State != session.Done || len(chat.requests()) <= sent {
		t.Fatalf("resend ended %s after %d requests", a.sess.State, len(chat.requests()))
	}
}

func TestBoxCloseCancelsResendWindowReview(t *testing.T) {
	chat := &fakeChat{status: 400}
	set, keys, _ := fileSetup(t, "make it shorter", chat)
	ev := script(t, hotkeyLine(0, "rewrite/0", proto.PhaseTrigger))
	a := newSessionApp(t, helpertest.Opt("selection", "the selected text")+","+ev, func(s *settings.Settings) {
		set(s)
		s.Privacy.SendWindow = true
	}, keys)
	a.pump("the recording", func() bool { return a.box.open && a.sess.State == session.Recording })
	a.postQueue(a.boxSend)
	a.pump("the first window review", func() bool {
		return a.sess.ID == a.box.session && a.sess.State == session.Recognizing && len(a.sess.ImagePreview) > 0
	})
	a.postQueue(a.boxSend)
	a.pump("the first failure", func() bool { return a.box.shown && a.box.err != "" })
	sent := len(chat.requests())
	chat.set("Recovered.", 0)

	a.postQueue(a.boxSend)
	a.pump("the resend's window review", func() bool {
		return a.box.sending != "" && a.sess.ID == a.box.sending && a.sess.State == session.Composing && a.sess.ImagePreviewKind == "window" && len(a.sess.ImagePreview) > 0
	})
	target, resend := a.box.target, a.box.sending
	a.postQueue(a.closeBox)
	a.pump("the resend's cancellation", func() bool { return !a.box.open && a.sess.State.Ended() })
	if a.sess.State != session.Cancelled {
		t.Fatalf("Close ended the resend as %s, want cancelled", a.sess.State)
	}
	if got := len(chat.requests()); got != sent {
		t.Fatalf("Close sent the reviewed screenshot: %d requests, want %d", got, sent)
	}
	a.postQueue(func() { a.openBox(target, session.StartRequest{Kind: store.KindRewrite, Target: target}) })
	a.pump("a new recording", func() bool {
		return a.box.open && a.box.session != resend && a.sess.ID == a.box.session && a.sess.State == session.Recording
	})
}

func TestAutomaticImageAppearsOnAnswerCardOnlyWhenSent(t *testing.T) {
	a, _, _ := newTestApp(t)
	jpeg := boxJPEGOf(t, 20, 10)
	snap(a, session.Snapshot{ID: "r1", Kind: store.KindRewrite, State: session.Done, Card: true,
		Raw: "describe it", Result: "A rectangle.", ImageHint: "clipboard", ImagePreviewKind: "clipboard", ImagePreview: jpeg})
	if len(a.answer.previews) != 1 || a.answer.previews[0].Label != "Copied image" {
		t.Fatalf("automatic previews %+v", a.answer.previews)
	}
	a, _, _ = newTestApp(t)
	snap(a, session.Snapshot{ID: "r2", Kind: store.KindRewrite, State: session.Done, Card: true,
		Raw: "describe it", Result: "Text only.", ImagePreviewKind: "clipboard", ImagePreview: jpeg})
	if len(a.answer.previews) != 0 {
		t.Fatal("a refused image was listed as sent")
	}
}

func TestRedoDropsAutomaticImagePreviewWhenItWasNotResent(t *testing.T) {
	a, _, _ := newTestApp(t)
	a.answer = answerCard{open: true, kind: store.KindRewrite, attach: "clipboard", previews: []answerInputPreview{{Label: "Copied image", Image: shotBitmap(t)}}}
	a.endRedo(session.Snapshot{ID: "redo", Kind: store.KindRewrite, State: session.Done, Result: "New answer.", Selection: "source", Typed: "typed"})
	if a.answer.attach != "" {
		t.Fatalf("redo kept the old automatic image attachment %q", a.answer.attach)
	}
	for _, preview := range a.answer.previews {
		if preview.Image != nil {
			t.Fatalf("redo kept an automatic image that was not resent: %+v", preview)
		}
	}
	if len(a.answer.previews) != 2 || a.answer.previews[0].Text != "source" || a.answer.previews[1].Text != "typed" {
		t.Fatalf("redo text previews %+v", a.answer.previews)
	}
}

func TestRedoAutomaticImageChipsClearWhileStreaming(t *testing.T) {
	chat := &fakeChat{texts: []string{"First answer", "Second answer"}, delay: 200 * time.Millisecond}
	a := newBoxApp(t, boxScript(t), chat)
	a.pump("the first answer", func() bool { return a.answer.open })
	bitmap, _ := ui.DecodeBitmap(boxJPEGOf(t, 20, 10))
	a.answer.attach = "clipboard"
	a.answer.previews = append(a.answer.previews, answerInputPreview{Label: "Copied image", Image: bitmap})
	input := a.answer.input
	a.postQueue(func() {
		a.redoAnswer(session.StartRequest{Kind: a.answer.kind, Target: a.answer.target, Text: &input})
	})
	a.pump("the redo streaming", func() bool {
		return len(chat.requests()) == 2 && a.sess.ID == a.answer.redo && a.sess.State == session.Composing && a.sess.Draft != ""
	})
	if attachment := a.cachedCard().Attachment; attachment != "" {
		t.Fatalf("the streaming redo kept %q", attachment)
	}
	for _, preview := range a.ov.ui.cardPreviews {
		if preview.Image != nil {
			t.Fatalf("the streaming redo kept an unsent image: %+v", a.answer.previews)
		}
	}
	a.pump("the redo answer", func() bool { return a.answer.redo == "" })
}

func TestBoxLongPasteThroughEditorDoesNotDuplicate(t *testing.T) {
	a, _, _ := newTestApp(t)
	a.box = rewriteBox{open: true, shown: true, focusText: true}
	tt := hudTester(a.boxView, boxWinW, boxWinH)
	frames(tt)
	tt.Type("before\n")
	frames(tt)
	long := "first\nsecond\nthird\nfourth"
	tt.SetClipboardData(transfer.TextData(long))
	tt.Command("paste")
	frames(tt)
	if got, want := a.boxTextContent(), "before\n"+long; got != want {
		t.Fatalf("request text %q, want %q", got, want)
	}
	tt.Type("after")
	frames(tt)
	if got, want := a.boxTextContent(), "before\n"+long+"after"; got != want {
		t.Fatalf("request text after typing %q, want %q", got, want)
	}
}

func TestBoxTypingImmediatelyAfterPaste(t *testing.T) {
	a, _, _ := newTestApp(t)
	a.box = rewriteBox{open: true, shown: true, focusText: true}
	tt := hudTester(a.boxView, boxWinW, boxWinH)
	frames(tt)
	tt.Type("before\n")
	frames(tt)
	long := "first\nsecond\nthird\nfourth"
	tt.SetClipboardData(transfer.TextData(long))
	tt.Command("paste")
	tt.Type("X")
	frames(tt)
	if got, want := a.boxTextContent(), "before\n"+long+"X"; got != want {
		t.Fatalf("request text after fast typing %q, want %q", got, want)
	}
}

func TestBoxCaretMovesAfterPastedBlock(t *testing.T) {
	a, _, _ := newTestApp(t)
	a.box = rewriteBox{open: true, shown: true, focusText: true}
	tt := hudTester(a.boxView, boxWinW, boxWinH)
	frames(tt)
	tt.Type("hello world")
	frames(tt)
	for range 6 {
		tt.Key(0, ui.KeyLeft)
	}
	frames(tt)
	long := "one\ntwo\nthree\nfour"
	tt.SetClipboardData(transfer.TextData(long))
	tt.Command("paste")
	tt.Type("X")
	frames(tt)
	if got, want := a.boxTextContent(), "hello"+long+"X world"; got != want {
		t.Fatalf("request text after typing at the suffix caret %q, want %q", got, want)
	}
}

func TestBoxSelectedReplacementFollowedByTyping(t *testing.T) {
	a, _, _ := newTestApp(t)
	a.box = rewriteBox{open: true, shown: true, focusText: true}
	tt := hudTester(a.boxView, boxWinW, boxWinH)
	frames(tt)
	tt.Type("abc")
	frames(tt)
	tt.Key(ui.Cmd, ui.KeyA)
	frames(tt)
	long := "one\ntwo\nthree\nfour"
	tt.SetClipboardData(transfer.TextData(long))
	tt.Command("paste")
	tt.Type("X")
	frames(tt)
	if got, want := a.boxTextContent(), long+"X"; got != want {
		t.Fatalf("request text after selected replacement and typing %q, want %q", got, want)
	}
}

func TestBoxBackToBackPastesKeepTheirOrder(t *testing.T) {
	a, _, _ := newTestApp(t)
	a.box = rewriteBox{open: true, shown: true, focusText: true}
	tt := hudTester(a.boxView, boxWinW, boxWinH)
	frames(tt)
	tt.Type("hello world")
	frames(tt)
	for range 6 {
		tt.Key(0, ui.KeyLeft)
	}
	frames(tt)
	first := "one\ntwo\nthree\nfour"
	second := "five\nsix\nseven\neight"
	tt.SetClipboardData(transfer.TextData(first))
	tt.Command("paste")
	if got, want := a.boxTextContent(), "hello"+first+" world"; got != want {
		t.Fatalf("request text after the first paste %q, want %q", got, want)
	}
	tt.SetClipboardData(transfer.TextData(second))
	tt.Command("paste")
	tt.Type("X")
	frames(tt)
	if got, want := a.boxTextContent(), "hello"+first+second+"X world"; got != want {
		t.Fatalf("request text after back-to-back pastes %q, want %q", got, want)
	}
}

func TestBoxMixedPasteLengthsWithoutAFrame(t *testing.T) {
	long := "one\ntwo\nthree\nfour"
	for _, tc := range []struct {
		name   string
		pastes []string
		want   string
	}{
		{name: "long then short", pastes: []string{long, "x"}, want: long + "x"},
		{name: "short then long", pastes: []string{"x", long}, want: "x" + long},
	} {
		t.Run(tc.name, func(t *testing.T) {
			a, _, _ := newTestApp(t)
			a.box = rewriteBox{open: true, shown: true, focusText: true}
			tt := hudTester(a.boxView, boxWinW, boxWinH)
			frames(tt)
			for _, pasted := range tc.pastes {
				tt.SetClipboardData(transfer.TextData(pasted))
				tt.Command("paste")
			}
			frames(tt)
			if got := a.boxTextContent(); got != tc.want {
				t.Fatalf("request text %q, want %q", got, tc.want)
			}
		})
	}
}

func TestBoxLongPasteUsesTheCaretAndKeepsTextEditable(t *testing.T) {
	a, _, _ := newTestApp(t)
	a.box = rewriteBox{open: true, shown: true}
	a.box.text = "hello world"
	long := "one\ntwo\nthree\nfour"
	if !a.boxPasteTextAt(-1, long, 5, 5) {
		t.Fatal("the long paste was not taken")
	}
	if got, want := a.boxTextContent(), "hello"+long+" world"; got != want {
		t.Fatalf("request text %q, want %q", got, want)
	}
	a.boxRemoveText(1)
	a.boxSegmentChanged(0, "HELLO")
	if got, want := a.boxTextContent(), "HELLO world"; got != want {
		t.Fatalf("text after removing and editing %q, want %q", got, want)
	}
	if !a.boxPasteTextAt(0, long, 3, 3) {
		t.Fatal("a paste into an earlier typed segment was not taken")
	}
	if got, want := a.boxTextContent(), "HEL"+long+"LO world"; got != want {
		t.Fatalf("text after pasting into an earlier segment %q, want %q", got, want)
	}
}

func TestAnswerCardChipsCoverTextOnlyRewrite(t *testing.T) {
	a, _, _ := newTestApp(t)
	snap(a, session.Snapshot{ID: "r1", Kind: store.KindRewrite, State: session.Done, Card: true,
		Raw: "make it formal", Selection: "source text", Result: "Formal answer."})
	if len(a.answer.previews) != 1 || a.cachedCard().InputCount != 1 {
		t.Fatalf("previews %+v, DTO count %d", a.answer.previews, a.cachedCard().InputCount)
	}
	tt := cardTester(t, a)
	frames(tt)
	click(t, tt, "Selected text")
	frames(tt)
	if !has(tt, "source text") || !has(tt, "Back to answer") {
		t.Fatalf("text-only original did not open: %q", firstTexts(tt))
	}
}

func TestFollowUpChipsKeepNewTextAndOriginalImages(t *testing.T) {
	a, _, _ := newTestApp(t)
	jpeg := boxJPEGOf(t, 20, 10)
	a.box = rewriteBox{open: true, session: "r1", images: [][]byte{jpeg}, imageInfo: []boxImageInfo{{Source: "draft.png", Width: 20, Height: 10}}}
	bitmap, _ := ui.DecodeBitmap(jpeg)
	a.box.thumbs = []*ui.Bitmap{bitmap}
	snap(a, session.Snapshot{ID: "r1", Kind: store.KindRewrite, State: session.Done, Card: true,
		Raw: "first", Selection: "old source", Result: "First answer.", ImageHint: "box"})
	a.followUp = "r1"
	a.box = rewriteBox{}
	snap(a, session.Snapshot{ID: "r2", Kind: store.KindRewrite, State: session.Recording})
	snap(a, session.Snapshot{ID: "r2", Kind: store.KindRewrite, State: session.Done, Card: true,
		Raw: "second", Selection: "new source", Result: "Second answer.", ImageHint: "box"})
	labels := make([]string, 0, len(a.answer.previews))
	for _, preview := range a.answer.previews {
		labels = append(labels, preview.Label)
	}
	if len(a.answer.previews) != 2 || a.answer.previews[0].Text != "new source" || a.answer.previews[1].Label != "draft.png" {
		t.Fatalf("follow-up previews %+v, labels %q", a.answer.previews, labels)
	}
}

func TestAutomaticInputsCanBeRemoved(t *testing.T) {
	a, _, _ := newTestApp(t)
	jpeg := boxJPEGOf(t, 20, 10)
	a.box = rewriteBox{open: true, shown: true, session: "v1"}
	snap(a, session.Snapshot{ID: "v1", Kind: store.KindRewrite, State: session.Recording, Selection: "first\nsecond\nthird\nfourth", SelectionVia: "copy", ImagePreviewKind: "clipboard", ImagePreview: jpeg})
	tt := hudTester(a.boxView, boxWinW, boxWinH)
	frames(tt)
	click(t, tt, "Remove this image")
	frames(tt)
	if len(a.boxViewState().Thumbnails) != 0 {
		t.Fatal("the automatic image survived removal")
	}
	click(t, tt, "Remove pasted text")
	frames(tt)
	if a.boxViewState().Selection != "" {
		t.Fatal("the automatic text survived removal")
	}
	selectionSuppressed, imageSuppressed := a.boxShare.automaticSuppressed()
	if !selectionSuppressed || !imageSuppressed {
		t.Fatalf("suppression did not reach the session: text %v, image %v", selectionSuppressed, imageSuppressed)
	}
}
