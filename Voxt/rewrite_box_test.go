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
	"github.com/sgl-project/sglang-omni/Voxt/internal/llm"
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

// newBoxApp is a sessionApp with a fake speech and language model, ev
// the hotkey script.
func newBoxApp(t *testing.T, ev string, chat *fakeChat) *sessionApp {
	set, keys, _ := fileSetup(t, "make it shorter", chat)
	return newSessionApp(t, helpertest.Opt("selection", "the selected text")+","+ev, set, keys)
}

// boxScript presses the Rewrite shortcut twice: the card and the
// microphone, then the stop that sends.
func boxScript(t *testing.T) string {
	return script(t,
		hotkeyLine(0, "rewrite/0", proto.PhaseTrigger),
		hotkeyLine(500, "rewrite/0", proto.PhaseTrigger),
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

// lastUser is the chat request's last user message.
func lastUserOf(body map[string]any) string {
	msgs := body["messages"].([]any)
	return msgs[len(msgs)-1].(map[string]any)["content"].(string)
}

// TestRewriteBoxEndToEnd: the Rewrite shortcut opens the card with the
// input box and starts the microphone together (the pill stays hidden:
// the card shows the recording); the selection goes into the card by
// itself; typed text and pasted pictures go between the selection and
// the spoken instruction in the request; the second press finishes the
// recording and sends. The answer opens in the card, which shows the
// pictures.
func TestRewriteBoxEndToEnd(t *testing.T) {
	chat := &fakeChat{text: "A shorter text."}
	a := newBoxApp(t, boxScript(t), chat)
	a.pump("the card and the microphone", func() bool {
		return a.box.open && a.sess.State == session.Recording
	})
	if a.overlayModeNow() != ovHidden {
		t.Errorf("a preview showed beside the card: %v", a.overlayModeNow())
	}
	a.pump("the selection in the card", func() bool { return a.box.selection == "the selected text" })

	// Typed text and five pictures while it records: four go in, the
	// fifth is refused.
	a.postQueue(func() { a.box.text = "and the tickets"; a.boxSync() })
	jpeg := boxJPEG(t)
	for range 5 {
		a.postQueue(func() { a.boxAttach(jpeg) })
	}
	a.pump("the fifth refused", func() bool { return a.box.limit })
	if len(a.box.images) != boxMaxImages {
		t.Errorf("%d images", len(a.box.images))
	}

	a.pump("the answer", func() bool { return a.answer.open })
	if a.box.open {
		t.Error("the box stayed open after the answer")
	}
	reqs := chat.requests()
	if len(reqs) != 1 {
		t.Fatalf("%d chats", len(reqs))
	}
	want := "<selected_text>\nthe selected text\n</selected_text>\n\n<typed_text>\nand the tickets\n</typed_text>\n\n<instruction>\nmake it shorter\n</instruction>"
	if got := lastUserOf(reqs[0]); got != want {
		t.Errorf("the request's order:\n%s", got)
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

// TestRewriteBoxSpeechOnlyUnchanged: nothing selected, nothing typed,
// no picture: the quick path works as before — press, speak, press —
// and the request is the spoken instruction alone.
func TestRewriteBoxSpeechOnlyUnchanged(t *testing.T) {
	chat := &fakeChat{text: `{"title":"Friday","content":"We ship on Friday."}`}
	set, keys, _ := fileSetup(t, "write that we ship on friday", chat)
	a := newSessionApp(t, helpertest.Opt("selection", "")+","+boxScript(t), set, keys)
	a.pump("the card and the microphone", func() bool {
		return a.box.open && a.sess.State == session.Recording
	})
	a.pump("the end", func() bool { return a.sess.State.Ended() })
	if a.sess.State != session.Done || a.sess.Result != "We ship on Friday." {
		t.Fatalf("end %s %q %+v", a.sess.State, a.sess.Result, a.sess.Problem)
	}
	reqs := chat.requests()
	if len(reqs) != 1 {
		t.Fatalf("%d chats", len(reqs))
	}
	if got, want := lastUserOf(reqs[0]), "<instruction>\nwrite that we ship on friday\n</instruction>"; got != want {
		t.Errorf("the speech-only request:\n%s", got)
	}
	if imgs, _ := reqs[0]["images"].([][]byte); len(imgs) != 0 {
		t.Errorf("%d images with speech only", len(imgs))
	}
}

// TestRewriteBoxEscWithoutSending: Esc while the card records cancels
// it and closes the card; nothing is sent.
func TestRewriteBoxEscWithoutSending(t *testing.T) {
	chat := &fakeChat{text: "unused"}
	ev := script(t,
		hotkeyLine(0, "rewrite/0", proto.PhaseTrigger),
		hotkeyLine(400, "cancel", proto.PhaseTrigger),
	)
	a := newBoxApp(t, ev, chat)
	a.pump("the recording", func() bool { return a.box.open && a.sess.State == session.Recording })
	a.pump("the cancel", func() bool { return !a.box.open })
	a.pump("the session's end", func() bool { return a.sess.State.Ended() })
	if len(chat.requests()) != 0 {
		t.Errorf("%d chats after Esc", len(chat.requests()))
	}
}

// TestRewriteBoxRetrySameBytes: the language model refuses the pictures
// (a model that takes no images): the card comes back with the error
// and keeps everything; Enter sends again, and the second request
// carries the same image bytes in the same position, so the server's
// prefix cache could have reused them.
func TestRewriteBoxRetrySameBytes(t *testing.T) {
	chat := &fakeChat{status: 400}
	a := newBoxApp(t, boxScript(t), chat)
	a.pump("the recording", func() bool { return a.box.open && a.sess.State == session.Recording })
	a.pump("the selection", func() bool { return a.box.selection == "the selected text" })
	a.postQueue(func() { a.box.text = "and the tickets"; a.boxSync() })
	a.postQueue(func() { a.boxAttach(boxJPEG(t)) })
	a.postQueue(func() { a.boxAttach(boxJPEG(t)) })
	a.pump("the pictures", func() bool { return len(a.box.images) == 2 })

	a.pump("the failure", func() bool { return a.box.shown && a.box.err != "" })
	if !a.box.open || a.box.text != "and the tickets" || len(a.box.images) != 2 || a.box.instruction != "make it shorter" {
		t.Fatalf("the box lost its content: %+v", a.box)
	}
	if a.answer.open {
		t.Error("a failed card opened beside the box")
	}

	sent := len(chat.requests()) // the first send's, with the client's own retries
	a.postQueue(a.boxSend)
	a.pump("the resend", func() bool { return a.box.sending != "" })
	a.pump("the second failure", func() bool { return a.box.sending == "" && a.box.shown && a.box.err != "" })
	reqs := chat.requests()
	if len(reqs) <= sent {
		t.Fatalf("no resend request (%d of %d)", sent, len(reqs))
	}
	first, _ := reqs[0]["images"].([][]byte)
	again, _ := reqs[sent]["images"].([][]byte)
	if len(first) != 2 || len(again) != 2 || !bytes.Equal(first[0], again[0]) || !bytes.Equal(first[1], again[1]) {
		t.Errorf("the resend's images differ: %q vs %q", first, again)
	}
	if lastUserOf(reqs[0]) != lastUserOf(reqs[sent]) {
		t.Errorf("the resend's text differs:\n%s\n---\n%s", lastUserOf(reqs[0]), lastUserOf(reqs[sent]))
	}
	if entries := a.history(); len(entries) != 2 || entries[0].Status != store.StatusFailed {
		t.Errorf("history %v", entries)
	}
}

// TestRewriteBoxRedoSameBytes: asking the box's answer again at another
// thinking level sends the same typed text and the same image bytes in
// the same position, so the server's prefix cache reuses them.
func TestRewriteBoxRedoSameBytes(t *testing.T) {
	chat := &fakeChat{texts: []string{"A shorter text.", "A shorter text!"}}
	a := newBoxApp(t, boxScript(t), chat)
	a.pump("the recording", func() bool { return a.box.open && a.sess.State == session.Recording })
	a.pump("the selection", func() bool { return a.box.selection == "the selected text" })
	a.postQueue(func() { a.box.text = "and the tickets"; a.boxSync() })
	a.postQueue(func() { a.boxAttach(boxJPEG(t)) })
	a.postQueue(func() { a.boxAttach(boxJPEG(t)) })
	a.pump("the answer", func() bool { return a.answer.open })
	if len(chat.requests()) != 1 || a.answer.attach != "box" || len(a.answer.images) != 2 {
		t.Fatalf("requests %d, card %+v", len(chat.requests()), a.answer)
	}

	// Another thinking level asks the same input again.
	a.updateSettings(func(s *settings.Settings) { s.Providers.LLM = []settings.Provider{{ID: "deepseek"}} })
	a.answer.llmID = "deepseek"
	if !a.cardNow().canThink {
		t.Fatal("no thinking levels on the card")
	}
	a.postQueue(func() { a.chooseThinking(&llm.Thinking{Mode: llm.ThinkOn}) })
	a.pump("the redo", func() bool { return len(chat.requests()) == 2 && a.answer.redo == "" })
	reqs := chat.requests()
	first, _ := reqs[0]["images"].([][]byte)
	again, _ := reqs[1]["images"].([][]byte)
	if len(again) != 2 || !bytes.Equal(first[0], again[0]) || !bytes.Equal(first[1], again[1]) {
		t.Errorf("the redo's images differ: %q vs %q", first, again)
	}
	if lastUserOf(reqs[0]) != lastUserOf(reqs[1]) {
		t.Errorf("the redo's text differs:\n%s\n---\n%s", lastUserOf(reqs[0]), lastUserOf(reqs[1]))
	}
	if a.answer.text != "A shorter text!" {
		t.Errorf("answer %q", a.answer.text)
	}
}

// TestRewriteBoxView: the box's states as its view shows them: the
// selection, typing, Enter (taken; Shift+Enter is a new line), the
// pictures and their limit, the voice line, Esc.
func TestRewriteBoxView(t *testing.T) {
	a, _, _ := newTestApp(t)
	tt := hudTester(a.boxView, boxWinW, boxWinH)
	a.box = rewriteBox{open: true, shown: true, focusText: true, session: "v1"}
	snap(a, session.Snapshot{ID: "v1", Kind: store.KindRewrite, State: session.Recording, Selection: "the selected text"})
	frames(tt)
	if !has(tt, "the selected text") {
		t.Errorf("no selection in the card (texts %q)", firstTexts(tt))
	}
	tt.Type("some text")
	frames(tt)
	if a.box.text != "some text" {
		t.Fatalf("text %q", a.box.text)
	}
	// Enter is taken (no newline); Shift+Enter breaks the line.
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
	// The voice line: the live instruction, then the recognised one
	// kept after the recording, with Send.
	snap(a, session.Snapshot{ID: "v1", Kind: store.KindRewrite, State: session.Recording, Selection: "the selected text", Partial: "make it"})
	frames(tt)
	if !has(tt, "make it") {
		t.Error("no live instruction")
	}
	a.box.session, a.box.instruction = "", "make it shorter"
	snap(a, session.Snapshot{ID: "v1", Kind: store.KindRewrite, State: session.Cancelled})
	frames(tt)
	if !has(tt, "Send") {
		t.Error("no Send with an instruction kept")
	}
	// Esc closes the box without sending.
	tt.Key(0, ui.KeyEscape)
	frames(tt)
	if a.box.open {
		t.Error("Esc did not close the box")
	}
}

// TestAnswerCardAttach: the card says which picture went to the language
// model: the window's, the clipboard's, or the box's as thumbnails.
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
	a.box = rewriteBox{open: true, session: "r1", images: [][]byte{boxJPEG(t), boxJPEG(t)}}
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

// fakeBoxWindow records the panel's life (boxWindow).
type fakeBoxWindow struct{ hidden, destroyed bool }

func (f *fakeBoxWindow) Invalidate() {}
func (f *fakeBoxWindow) Center()     {}
func (f *fakeBoxWindow) Show()       {}
func (f *fakeBoxWindow) Focus()      {}
func (f *fakeBoxWindow) Hide()       { f.hidden = true }
func (f *fakeBoxWindow) Destroy()    { f.destroyed = true }

// TestRewriteBoxCloseClosesWindow: Esc drops the box's state, but the
// panel must go with it — zeroing the window pointer first would
// strand the visible panel on screen, and a reopen would add another.
func TestRewriteBoxCloseClosesWindow(t *testing.T) {
	a, _, _ := newTestApp(t)
	w := &fakeBoxWindow{}
	a.box = rewriteBox{open: true, shown: true, win: w, winShown: true}
	a.closeBox()
	if !w.destroyed || a.box.win != nil || a.box.open {
		t.Errorf("destroyed %v, box %+v", w.destroyed, a.box)
	}
}

// TestRewriteBoxStaleConversionDropped: a picture whose conversion
// ends after the box was closed must not land in the reopened one.
func TestRewriteBoxStaleConversionDropped(t *testing.T) {
	a, _, _ := newTestApp(t)
	a.box = rewriteBox{open: true, shown: true}
	gen := a.boxGen
	a.boxPending++
	a.closeBox()
	a.box.open = true // the shortcut again
	a.boxConverted(gen, boxJPEG(t), "")
	if len(a.box.images) != 0 || a.boxPending != 0 {
		t.Errorf("%d images, %d pending", len(a.box.images), a.boxPending)
	}
}

// TestRewriteBoxSendWaitsForConversion: Enter while a pasted picture
// still converts sends when it lands, with the picture; the accepted
// picture is never left behind.
func TestRewriteBoxSendWaitsForConversion(t *testing.T) {
	chat := &fakeChat{text: "A shorter text."}
	a := newBoxApp(t, boxScript(t), chat)
	a.pump("the card and the microphone", func() bool {
		return a.box.open && a.sess.State == session.Recording
	})
	a.postQueue(func() {
		gen := a.boxGen
		a.boxPending++
		a.boxSend() // Enter with the conversion still running
		if !a.boxSendQueued || a.sess.State != session.Recording {
			t.Errorf("queued %v, session %s", a.boxSendQueued, a.sess.State)
		}
		a.boxConverted(gen, boxJPEG(t), "")
	})
	a.pump("the answer", func() bool { return a.answer.open })
	reqs := chat.requests()
	if len(reqs) != 1 {
		t.Fatalf("%d chats", len(reqs))
	}
	if imgs, _ := reqs[0]["images"].([][]byte); len(imgs) != 1 {
		t.Errorf("%d images sent", len(imgs))
	}
}
