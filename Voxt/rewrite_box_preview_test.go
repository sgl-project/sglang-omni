package main

import (
	"testing"
	"unicode/utf8"

	"github.com/egoist/mygo/transfer"
	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/session"
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

func TestBoxLongPasteThroughEditorDoesNotDuplicate(t *testing.T) {
	a, _, _ := newTestApp(t)
	a.box = rewriteBox{open: true, shown: true, focusText: true}
	tt := hudTester(a.boxView, boxWinW, boxWinH)
	frames(tt)
	tt.Type("before\n")
	frames(tt)
	long := "first\nsecond\nthird\nfourth"
	tt.SetClipboardData(transfer.TextData(long))
	tt.Key(ui.Cmd, ui.KeyV)
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
	tt.Key(ui.Cmd, ui.KeyV)
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
	tt.Key(ui.Cmd, ui.KeyV)
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
	tt.Key(ui.Cmd, ui.KeyV)
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
	tt.Key(ui.Cmd, ui.KeyV)
	tt.SetClipboardData(transfer.TextData(second))
	tt.Key(ui.Cmd, ui.KeyV)
	tt.Type("X")
	frames(tt)
	if got, want := a.boxTextContent(), "hello"+first+second+"X world"; got != want {
		t.Fatalf("request text after back-to-back pastes %q, want %q", got, want)
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

func TestAutomaticImageHasNoRemoveButton(t *testing.T) {
	a, _, _ := newTestApp(t)
	jpeg := boxJPEGOf(t, 20, 10)
	a.box = rewriteBox{open: true, shown: true, session: "v1"}
	snap(a, session.Snapshot{ID: "v1", Kind: store.KindRewrite, State: session.Recording, ImagePreviewKind: "clipboard", ImagePreview: jpeg})
	tt := hudTester(a.boxView, boxWinW, boxWinH)
	frames(tt)
	if has(tt, "Remove this image") {
		t.Fatal("an automatic image has a button that cannot remove it")
	}
}
