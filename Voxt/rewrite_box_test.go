package main

import (
	"bytes"
	"image"
	"image/jpeg"
	"image/png"
	"strings"
	"testing"

	"github.com/egoist/mygo/transfer"
	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/helper/helpertest"
	"github.com/sgl-project/sglang-omni/Voxt/internal/helper/proto"
	"github.com/sgl-project/sglang-omni/Voxt/internal/overlay/layout"
	"github.com/sgl-project/sglang-omni/Voxt/internal/session"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

// boxJPEG is a real JPEG for the box's thumbnails.
func boxJPEG(t *testing.T) []byte {
	t.Helper()
	var out bytes.Buffer
	if err := jpeg.Encode(&out, image.NewRGBA(image.Rect(0, 0, 8, 8)), nil); err != nil {
		t.Fatal(err)
	}
	return out.Bytes()
}

// boxPNG is a PNG for a clipboard picture.
func boxPNG(t *testing.T) []byte {
	t.Helper()
	var out bytes.Buffer
	if err := png.Encode(&out, image.NewRGBA(image.Rect(0, 0, 8, 8))); err != nil {
		t.Fatal(err)
	}
	return out.Bytes()
}

// newBoxApp is a sessionApp with the input box on and a fake speech and
// language model, ev the hotkey script.
func newBoxApp(t *testing.T, ev string, chat *fakeChat) *sessionApp {
	set, keys, _ := fileSetup(t, "make it shorter", chat)
	return newSessionApp(t, helpertest.Opt("selection", "the selected text")+","+ev, func(s *settings.Settings) {
		set(s)
		s.Features.Rewrite.InputBox = true
	}, keys)
}

// boxScript opens the box, starts the voice and stops it.
func boxScript(t *testing.T) string {
	return script(t,
		hotkeyLine(0, "rewrite/0", proto.PhaseTrigger),
		hotkeyLine(400, "rewrite/0", proto.PhaseTrigger),
		hotkeyLine(900, "rewrite/0", proto.PhaseTrigger),
	)
}

// chatTexts is the text of the chat request's messages, joined.
func chatTexts(body map[string]any) string {
	var b strings.Builder
	for _, m := range body["messages"].([]any) {
		b.WriteString(m.(map[string]any)["content"].(string))
		b.WriteByte('\n')
	}
	return b.String()
}

// TestRewriteBoxEndToEnd: with the input box on, the Rewrite shortcut
// opens the box (no session, the selection pre-fills), the second press
// hears the instruction, and Enter sends the box's text as the source
// with its pictures (at most boxMaxImages): the answer opens in the
// card, which shows the pictures; the voice session keeps no history.
func TestRewriteBoxEndToEnd(t *testing.T) {
	chat := &fakeChat{text: "A shorter text."}
	a := newBoxApp(t, boxScript(t), chat)
	a.pump("the box", func() bool { return a.box.open })
	if a.sess.State.Live() {
		t.Fatal("the box opening started a session")
	}
	a.pump("the selection in the box", func() bool { return a.box.text == "the selected text" })
	a.pump("the instruction", func() bool { return a.box.instruction == "make it shorter" })

	// Five pictures: four go in, the fifth is refused.
	jpeg := boxJPEG(t)
	for range 5 {
		a.postQueue(func() { a.boxAttach(jpeg) })
	}
	a.pump("the fifth refused", func() bool { return a.box.limit })
	if len(a.box.images) != boxMaxImages {
		t.Errorf("%d images", len(a.box.images))
	}

	a.postQueue(a.boxSend)
	a.pump("the answer", func() bool { return a.answer.open })
	if a.box.open {
		t.Error("the box stayed open after the answer")
	}
	reqs := chat.requests()
	if len(reqs) != 1 {
		t.Fatalf("%d chats", len(reqs))
	}
	texts := chatTexts(reqs[0])
	if !strings.Contains(texts, "the selected text") || !strings.Contains(texts, "make it shorter") {
		t.Errorf("the request's source and instruction:\n%s", texts)
	}
	imgs, _ := reqs[0]["images"].([][]byte)
	if len(imgs) != boxMaxImages {
		t.Errorf("%d images sent", len(imgs))
	}
	if a.answer.attach != "box" || len(a.answer.thumbs) != boxMaxImages {
		t.Errorf("the card's attach %q, %d thumbs", a.answer.attach, len(a.answer.thumbs))
	}
	if entries := a.history(); len(entries) != 1 || entries[0].Kind != store.KindRewrite {
		t.Errorf("history %v", entries)
	}
}

// TestRewriteBoxOffUnchanged: with the input box off, the Rewrite
// shortcut records and sends as ever; no box opens.
func TestRewriteBoxOffUnchanged(t *testing.T) {
	chat := &fakeChat{text: "A shorter text."}
	set, keys, _ := fileSetup(t, "make it shorter", chat)
	ev := script(t, hotkeyLine(0, "rewrite/0", proto.PhaseTrigger), hotkeyLine(400, "rewrite/0", proto.PhaseTrigger))
	a := newSessionApp(t, helpertest.Opt("selection", "the selected text")+","+ev, set, keys)
	a.pump("the end", func() bool { return a.sess.State.Ended() })
	if a.box.open {
		t.Error("the box opened with the mode off")
	}
	if a.sess.State != session.Done {
		t.Fatalf("end %s %+v", a.sess.State, a.sess.Problem)
	}
	if len(chat.requests()) != 1 {
		t.Errorf("%d chats", len(chat.requests()))
	}
}

// TestRewriteBoxKeepsContentOnRefusal: the language model refuses the
// request (a model that takes no images): the box comes back with the
// error and keeps the text and the pictures; no failed card opens.
func TestRewriteBoxKeepsContentOnRefusal(t *testing.T) {
	chat := &fakeChat{status: 400}
	a := newBoxApp(t, boxScript(t), chat)
	a.pump("the box", func() bool { return a.box.open })
	a.pump("the selection in the box", func() bool { return a.box.text == "the selected text" })
	a.pump("the instruction", func() bool { return a.box.instruction != "" })
	a.postQueue(func() { a.boxAttach(boxJPEG(t)) })
	a.pump("the picture", func() bool { return len(a.box.images) == 1 })
	a.postQueue(a.boxSend)
	a.pump("the failure", func() bool { return a.box.shown && a.box.err != "" })
	if !a.box.open || a.box.text != "the selected text" || len(a.box.images) != 1 || a.box.instruction == "" {
		t.Errorf("the box lost its content: %+v", a.box)
	}
	if a.answer.open {
		t.Error("a failed card opened beside the box")
	}
	if entries := a.history(); len(entries) != 1 || entries[0].Status != store.StatusFailed {
		t.Errorf("history %v", entries)
	}
}

// TestRewriteBoxEscWhileListening: Esc while the box's voice records
// cancels it and closes the box; nothing is sent.
func TestRewriteBoxEscWhileListening(t *testing.T) {
	chat := &fakeChat{text: "unused"}
	ev := script(t,
		hotkeyLine(0, "rewrite/0", proto.PhaseTrigger),
		hotkeyLine(300, "rewrite/0", proto.PhaseTrigger),
		hotkeyLine(600, "cancel", proto.PhaseTrigger),
	)
	a := newBoxApp(t, ev, chat)
	a.pump("the voice", func() bool { return a.box.voice != "" })
	a.pump("the cancel", func() bool { return !a.box.open })
	if len(chat.requests()) != 0 {
		t.Errorf("%d chats after Esc", len(chat.requests()))
	}
}

// TestRewriteBoxView: the box's states as its view shows them: typing,
// Enter (sends only with an instruction; Shift+Enter is a new line), the
// pictures and their limit, the voice line, Esc.
func TestRewriteBoxView(t *testing.T) {
	a, _, _ := newTestApp(t)
	tt := hudTester(a.boxView, boxWinW, boxWinH)
	a.box = rewriteBox{open: true, shown: true, focusText: true}
	frames(tt)
	tt.Type("some text")
	frames(tt)
	if a.box.text != "some text" {
		t.Fatalf("text %q", a.box.text)
	}
	// Enter without an instruction does nothing (no newline); Shift+Enter
	// breaks the line.
	tt.Key(0, ui.KeyEnter)
	tt.Key(ui.Shift, ui.KeyEnter)
	frames(tt)
	if a.box.text != "some text\n" {
		t.Errorf("text %q after Enter and Shift+Enter", a.box.text)
	}
	// Paste a picture: it goes into the box.
	tt.SetClipboardData(transfer.New(transfer.NewItem(transfer.Bytes(transfer.PNG, boxPNG(t)))))
	tt.Key(ui.Cmd, ui.KeyV)
	frames(tt)
	if len(a.box.images) != 1 {
		t.Fatalf("%d images after paste", len(a.box.images))
	}
	// The cap and its note.
	for range boxMaxImages {
		a.boxAttach(boxJPEG(t))
	}
	frames(tt)
	if len(a.box.images) != boxMaxImages || !a.box.limit || !has(tt, "At most 4 images") {
		t.Errorf("cap: %d images, limit %v", len(a.box.images), a.box.limit)
	}
	// A thumbnail's remove button.
	click(t, tt, "Remove this image")
	if len(a.box.images) != boxMaxImages-1 {
		t.Errorf("%d images after remove", len(a.box.images))
	}
	// The voice line: the hint, then the live instruction, then the
	// recognised one with Send.
	if !has(tt, "Say the instruction") {
		t.Error("no hint on the voice line")
	}
	a.box.voice = "v1"
	snap(a, session.Snapshot{ID: "v1", Kind: store.KindRewrite, State: session.Recording, Partial: "make it"})
	frames(tt)
	if !has(tt, "make it") {
		t.Error("no live instruction")
	}
	a.box.voice, a.box.instruction = "", "make it shorter"
	snap(a, session.Snapshot{ID: "v1", Kind: store.KindRewrite, State: session.Cancelled})
	frames(tt)
	if !has(tt, "Send") {
		t.Error("no Send with an instruction")
	}
	// Esc closes the box without sending.
	tt.Key(0, ui.KeyEscape)
	frames(tt)
	if a.box.open {
		t.Error("Esc did not close the box")
	}
}

// TestAnswerCardAttach: the card says which picture went to the language
// model, in both modes: the window's, the clipboard's, or the box's as
// thumbnails.
func TestAnswerCardAttach(t *testing.T) {
	for _, tc := range []struct{ hint, want string }{
		{"window", "Screenshot"},
		{"clipboard", "Copied image"},
	} {
		t.Run(tc.hint, func(t *testing.T) {
			a, _, _ := newTestApp(t)
			snap(a, session.Snapshot{ID: "r1", Kind: store.KindRewrite, State: session.Done, Card: true,
				Raw: "describe it", Title: "A picture", Result: "A red square.", ImageHint: tc.hint})
			tt := cardTester(t, a)
			frames(tt)
			if !has(tt, tc.want) {
				t.Errorf("no %q on the card (texts %q)", tc.want, firstTexts(tt))
			}
			_, with, _ := a.cardSize()
			a.answer.attach = ""
			_, without, _ := a.cardSize()
			if with != without+layout.CardAttachH {
				t.Errorf("card height %d with the row, %d without", with, without)
			}
		})
	}
	// The box's pictures as thumbnails.
	a, _, _ := newTestApp(t)
	a.box = rewriteBox{open: true, sending: "r1", images: [][]byte{boxJPEG(t), boxJPEG(t)}}
	a.box.thumbs = nil
	for _, j := range a.box.images {
		bmp, _ := ui.DecodeBitmap(j)
		a.box.thumbs = append(a.box.thumbs, bmp)
	}
	snap(a, session.Snapshot{ID: "r1", Kind: store.KindRewrite, State: session.Done, Card: true,
		Raw: "describe them", Title: "Two pictures", Result: "Two red squares.", ImageHint: "box"})
	if a.answer.attach != "box" || len(a.answer.thumbs) != 2 {
		t.Fatalf("the card's attach %q, %d thumbs", a.answer.attach, len(a.answer.thumbs))
	}
	_, with, _ := a.cardSize()
	if with == 0 {
		t.Error("no card")
	}
}
