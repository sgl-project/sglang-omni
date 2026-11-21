package main

import (
	"bytes"
	"encoding/json"
	"image"
	"image/jpeg"
	"image/png"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/egoist/mygo"
	"github.com/egoist/mygo/transfer"
	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/helper/helpertest"
	"github.com/sgl-project/sglang-omni/Voxt/internal/helper/proto"
	"github.com/sgl-project/sglang-omni/Voxt/internal/imgconv"
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

// boxJPEGOf is a real JPEG of w × h, distinct bytes per size.
func boxJPEGOf(t *testing.T, w, h int) []byte {
	t.Helper()
	var out bytes.Buffer
	if err := jpeg.Encode(&out, image.NewRGBA(image.Rect(0, 0, w, h)), nil); err != nil {
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
	if v := a.boxViewState(); !a.box.open || !v.Conversation || len(a.box.images) != 0 {
		t.Errorf("after the answer: the card open %v, %d images left, %+v", a.box.open, len(a.box.images), v)
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
	if a.answer.attach != "box" || len(a.answer.ui.thumbs) != boxMaxImages {
		t.Errorf("the card's attach %q, %d thumbs", a.answer.attach, len(a.answer.ui.thumbs))
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
	set, keys, _ := fileSetup(t, "make it shorter", chat)
	keys.LLMModel = fakeLLMModel
	keys.LLMModel.ID = "deepseek"
	a := newSessionApp(t, helpertest.Opt("selection", "the selected text")+","+boxScript(t), set, keys)
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
	a.boxShare.store("someone else's text", [][]byte{[]byte("someone else's image")})
	if !a.cachedCard().CanThink {
		t.Fatal("no thinking levels on the card")
	}
	a.postQueue(func() { a.chooseThinking(sessionThinkingValueOf(&llm.Thinking{Mode: llm.ThinkOn})) })
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

func TestRewriteBoxFollowUpsKeepImages(t *testing.T) {
	chat := &fakeChat{texts: []string{"First answer", "Second answer", "Third answer"}}
	set, keys, transcriptions := fileSetup(t, "make it shorter", chat)
	a := newSessionApp(t, helpertest.Opt("selection", "the selected text")+","+boxScript(t), set, keys)
	a.pump("the recording", func() bool { return a.box.open && a.sess.State == session.Recording })
	a.postQueue(func() { a.boxAttach(boxJPEG(t)) })
	a.pump("the first answer", func() bool { return a.answer.open })
	if len(a.answer.images) != 1 || len(a.answer.ui.thumbs) != 1 {
		t.Fatalf("the first card lost the image: %+v", a.answer)
	}
	first := a.answer.images[0]
	for index, instruction := range []string{"first follow-up", "second follow-up"} {
		a.recordFollowUp(t, transcriptions, instruction)
		requests := chat.requests()
		images, _ := requests[index+1]["images"].([][]byte)
		if len(images) != 1 || !bytes.Equal(images[0], first) {
			t.Fatalf("%s images %q, want the first image", instruction, images)
		}
		if len(a.answer.images) != 1 || len(a.answer.ui.thumbs) != 1 {
			t.Fatalf("%s left %d images and %d thumbnails", instruction, len(a.answer.images), len(a.answer.ui.thumbs))
		}
	}
}

func TestRewriteBoxTypedOnlyFollowUp(t *testing.T) {
	chat := &fakeChat{texts: []string{"First answer", "Second answer"}}
	set, keys, transcriptions := fileSetup(t, "make it shorter", chat)
	a := newSessionApp(t, helpertest.Opt("selection", "")+","+boxScript(t), set, keys)
	a.pump("the recording", func() bool { return a.box.open && a.sess.State == session.Recording })
	a.postQueue(func() { a.box.text = "a rough draft"; a.boxSync() })
	a.pump("the first answer", func() bool { return a.answer.open })
	requests := chat.requests()
	firstUser := "<selected_text>\na rough draft\n</selected_text>\n\n<instruction>\nmake it shorter\n</instruction>"
	if got := lastUserOf(requests[0]); got != firstUser {
		t.Fatalf("the first request %q, want %q", got, firstUser)
	}
	a.recordFollowUp(t, transcriptions, "more formal")
	messages := chatMessages(chat.requests()[1])
	if len(messages) != 4 || messages[1][1] != firstUser {
		t.Fatalf("the follow-up changed the first turn: %q", messages)
	}
}

// TestReopenBoxTypedAfterRestart: a restored box answer keeps typed text
// for both another thinking level and a follow-up, with the same messages.
func TestReopenBoxTypedAfterRestart(t *testing.T) {
	for _, tc := range []struct{ name, selected, typed string }{
		{"typed only", "", "a rough draft"},
		{"selected and typed", "the selected text", "and the tickets"},
	} {
		t.Run(tc.name, func(t *testing.T) {
			dir := t.TempDir()
			var conversation, originalUser string
			t.Run("before restart", func(t *testing.T) {
				chat := &fakeChat{text: "A shorter text."}
				set, models, _ := fileSetup(t, "make it shorter", chat)
				a := newSessionAppIn(t, dir, helpertest.Opt("selection", tc.selected)+","+boxScript(t), func(s *settings.Settings) {
					set(s)
					s.Features.Rewrite.ShowCard = true
				}, models)
				a.pump("the recording", func() bool { return a.box.open && a.sess.State == session.Recording })
				a.postQueue(func() { a.box.text = tc.typed; a.boxSync() })
				a.pump("the answer", func() bool { return a.answer.open })
				originalUser = lastUserOf(chat.requests()[0])
				conversation = a.answer.conv
				entries := a.history()
				if len(entries) != 1 || entries[0].TypedText != tc.typed {
					t.Fatalf("history %+v", entries)
				}
			})

			chat := &fakeChat{texts: []string{"A shorter text!", "More formal."}}
			set, models, transcriptions := fileSetup(t, "more formal", chat)
			models.LLMModel = fakeLLMModel
			models.LLMModel.ID = "deepseek"
			a := newSessionAppIn(t, dir, "", func(s *settings.Settings) {
				set(s)
				s.Features.Rewrite.ShowCard = true
			}, models)
			a.loadTrayRecent()
			a.restoreAnswer(conversation)
			a.pump("the restored card", func() bool { return a.answer.open })
			if a.answer.input.Typed != tc.typed || len(a.answer.turns) != 1 {
				t.Fatalf("restored %+v", a.answer)
			}
			turn := a.answer.turns[0]
			wantSelection, wantTyped := tc.selected, tc.typed
			if tc.selected == "" {
				wantSelection, wantTyped = tc.typed, ""
			}
			if turn.Selection != wantSelection || turn.Typed != wantTyped {
				t.Fatalf("restored turn %+v", turn)
			}

			a.updateSettings(func(s *settings.Settings) { s.Providers.LLM = []settings.Provider{{ID: "deepseek"}} })
			a.answer.llmID = "deepseek"
			if !a.cachedCard().CanThink {
				t.Fatal("no thinking levels on the restored card")
			}
			a.postQueue(func() { a.chooseThinking(sessionThinkingValueOf(&llm.Thinking{Mode: llm.ThinkOn})) })
			a.pump("the redo", func() bool { return len(chat.requests()) == 1 && a.answer.redo == "" })
			if got := lastUserOf(chat.requests()[0]); got != originalUser {
				t.Fatalf("redo changed the input:\n%s\n--- want ---\n%s", got, originalUser)
			}

			a.recordFollowUp(t, transcriptions, "more formal")
			messages := chatMessages(chat.requests()[1])
			if len(messages) != 4 || messages[1][1] != originalUser {
				t.Fatalf("follow-up changed the first turn:\n%s\n--- want ---\n%s", messages, originalUser)
			}
		})
	}
}

// TestRewriteBoxView: the box's states as its view shows them: the
// selection, typing, Enter (taken; Shift+Enter is a new line), the
// pictures and their limit, the voice line, Esc.
func TestRewriteBoxView(t *testing.T) {
	a, fs, _ := newTestApp(t)
	fs.handlers[proto.MethodClipboardImage] = func(json.RawMessage) (any, error) {
		return proto.ClipboardImageResult{JPEG: boxJPEG(t)}, nil
	}
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
	tt.Command("paste")
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
			a.refreshSessionOverlay(time.Now())
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
	if a.answer.attach != "box" || len(a.answer.ui.thumbs) != 2 {
		t.Fatalf("the card's attach %q, %d thumbs", a.answer.attach, len(a.answer.ui.thumbs))
	}
	_, with, _ := a.cardSize()
	if with == 0 {
		t.Error("no card")
	}

	// The first answer streams while the recording's session still owns the box.
	a, _, _ = newTestApp(t)
	a.box = rewriteBox{open: true, session: "r1", images: [][]byte{boxJPEG(t), boxJPEG(t)}}
	for _, encoded := range a.box.images {
		bitmap, _ := ui.DecodeBitmap(encoded)
		a.box.thumbs = append(a.box.thumbs, bitmap)
	}
	snap(a, session.Snapshot{ID: "r1", Kind: store.KindRewrite, State: session.Composing, Card: true, Draft: "Two red squares.", ImageHint: "box"})
	card := a.cachedCard()
	if card.Attachment != "box" || card.ThumbnailCount != 2 || len(a.ov.ui.cardThumbs) != 2 {
		t.Fatalf("the streaming card's attachments %q, %d thumbs, %d native", card.Attachment, card.ThumbnailCount, len(a.ov.ui.cardThumbs))
	}
}

// fakeBoxWindow records the panel's life (boxWindow).
type fakeBoxWindow struct {
	hidden, destroyed bool
	style             mygo.VibrancyStyle
	bounds            mygo.Rectangle
}

func (f *fakeBoxWindow) Invalidate()                           {}
func (f *fakeBoxWindow) SetContentBounds(r mygo.Rectangle)     { f.bounds = r }
func (f *fakeBoxWindow) Show()                                 {}
func (f *fakeBoxWindow) Focus()                                {}
func (f *fakeBoxWindow) Hide()                                 { f.hidden = true }
func (f *fakeBoxWindow) Destroy()                              { f.destroyed = true }
func (f *fakeBoxWindow) SetVibrancyStyle(s mygo.VibrancyStyle) { f.style = s }

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
	seq := a.boxReserve()
	a.closeBox()
	a.box.open = true // the shortcut again
	a.boxConverted(gen, seq, boxJPEG(t), "")
	if len(a.box.images) != 0 || a.boxPending != 0 {
		t.Errorf("%d images, %d pending", len(a.box.images), a.boxPending)
	}
}

func TestOpenBoxResetsAutomaticInputSuppression(t *testing.T) {
	a := newBoxApp(t, "", &fakeChat{text: "A shorter text."})
	a.box = rewriteBox{open: true, selectionSuppressed: true, automaticImageSuppressed: true}
	a.postQueue(func() { a.openBox(slackApp, session.StartRequest{Kind: store.KindRewrite}) })
	a.pump("the new recording", func() bool { return a.box.open && a.sess.State == session.Recording })
	if a.box.selectionSuppressed || a.box.automaticImageSuppressed {
		t.Fatal("the new box kept the previous box's removed inputs")
	}
	selectionSuppressed, imageSuppressed := a.boxShare.automaticSuppressed()
	if selectionSuppressed || imageSuppressed {
		t.Fatal("the session still suppresses the new automatic inputs")
	}
}

// TestRewriteBoxQueuedSendDiesWithTheBox: Enter queued behind a pasted
// picture's conversion belongs to that box's life. Esc and a reopen
// start a new one: the old life's conversion, landing now, must not
// stop and send the new recording — the user's Enter does, once.
func TestRewriteBoxQueuedSendDiesWithTheBox(t *testing.T) {
	chat := &fakeChat{text: "A shorter text."}
	a := newBoxApp(t, script(t, hotkeyLine(0, "rewrite/0", proto.PhaseTrigger)), chat)
	a.pump("the first recording", func() bool { return a.box.open && a.sess.State == session.Recording })
	var gen, seq int
	a.postQueue(func() {
		gen = a.boxGen
		seq = a.boxReserve() // the pasted picture still converts
		a.boxSend()          // Enter: the send waits for it
		a.closeBox()         // Esc before it lands
	})
	a.pump("the box closed", func() bool { return !a.box.open && a.sess.State.Ended() })
	a.drain(500 * time.Millisecond) // the cancelled capture's teardown
	a.postQueue(func() { a.openBox(proto.App{}, session.StartRequest{Kind: store.KindRewrite}) })
	a.pump("the second recording", func() bool {
		return a.box.open && a.box.session != "" && a.sess.ID == a.box.session && a.sess.State == session.Recording
	})
	// The old life's conversion lands: no send may come of it.
	a.postQueue(func() { a.boxConverted(gen, seq, boxJPEG(t), "") })
	a.drain(500 * time.Millisecond)
	if a.sess.State != session.Recording {
		t.Fatalf("a stale conversion's queued send stopped the new recording (%s)", a.sess.State)
	}
	if n := len(chat.requests()); n != 0 {
		t.Fatalf("%d chats without the user's Enter", n)
	}
	// The user's own Enter still sends, once.
	a.postQueue(a.boxSend)
	a.pump("the answer", func() bool { return a.answer.open })
	if n := len(chat.requests()); n != 1 {
		t.Fatalf("%d chats", n)
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
		seq := a.boxReserve()
		a.boxSend() // Enter with the conversion still running
		if !a.boxSendQueued || a.sess.State != session.Recording {
			t.Errorf("queued %v, session %s", a.boxSendQueued, a.sess.State)
		}
		a.boxConverted(gen, seq, boxJPEG(t), "")
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

// TestRewriteBoxConversionFailureNeedsConfirmation: a failed conversion
// cancels a queued send but keeps the box and its error. Enter confirms
// the remaining content instead of sending it unseen.
func TestRewriteBoxConversionFailureNeedsConfirmation(t *testing.T) {
	chat := &fakeChat{text: "A shorter text."}
	a := newBoxApp(t, script(t, hotkeyLine(0, "rewrite/0", proto.PhaseTrigger)), chat)
	a.pump("the recording", func() bool { return a.box.open && a.sess.State == session.Recording })
	a.postQueue(func() {
		gen := a.boxGen
		seq := a.boxReserve()
		a.boxSend()
		if !a.boxSendQueued {
			t.Error("the send did not wait for the conversion")
		}
		a.boxConverted(gen, seq, nil, "This image could not be read.")
	})
	a.drain(100 * time.Millisecond)
	if !a.box.open || !a.box.shown || a.box.err == "" || a.boxSendQueued || a.sess.State != session.Recording {
		t.Fatalf("box %+v, session %s", a.box, a.sess.State)
	}
	if n := len(chat.requests()); n != 0 {
		t.Fatalf("%d chats after the failed conversion", n)
	}
	a.postQueue(a.boxSend)
	a.pump("the confirmed answer", func() bool { return a.answer.open })
	reqs := chat.requests()
	if len(reqs) != 1 {
		t.Fatalf("%d chats", len(reqs))
	}
	if imgs, _ := reqs[0]["images"].([][]byte); len(imgs) != 0 {
		t.Errorf("%d images sent after the failed conversion", len(imgs))
	}
}

// TestRewriteBoxClipboardFailureNeedsConfirmation: a clipboard.image
// failure is a failed conversion too, never a silent text-only send.
func TestRewriteBoxClipboardFailureNeedsConfirmation(t *testing.T) {
	chat := &fakeChat{text: "A shorter text."}
	a := newBoxApp(t, helpertest.Opt("clipboard-error", "1")+","+script(t, hotkeyLine(0, "rewrite/0", proto.PhaseTrigger)), chat)
	a.pump("the recording", func() bool { return a.box.open && a.sess.State == session.Recording })
	tt := hudTester(a.boxView, boxWinW, boxWinH)
	data, err := transfer.FileData(filepath.Join(t.TempDir(), "picture.heic"))
	if err != nil {
		t.Fatal(err)
	}
	tt.SetClipboardData(data)
	tt.Command("paste")
	a.boxSend()
	if !a.boxSendQueued {
		t.Fatal("the send did not wait for clipboard.image")
	}
	frames(tt)
	a.drain(100 * time.Millisecond)
	if !a.box.open || !a.box.shown || a.box.err == "" || a.boxSendQueued || a.sess.State != session.Recording {
		t.Fatalf("box %+v, session %s", a.box, a.sess.State)
	}
	if n := len(chat.requests()); n != 0 {
		t.Fatalf("%d chats after clipboard.image failed", n)
	}
}

// TestRewriteBoxHotkeyStopWaitsForConversion: the Rewrite shortcut and a
// hold's release use the same conversion wait as Enter in the box.
func TestRewriteBoxHotkeyStopWaitsForConversion(t *testing.T) {
	for _, tc := range []struct{ name, startPhase, stopPhase string }{
		{"tap", proto.PhaseTrigger, proto.PhaseTrigger},
		{"hold release", proto.PhaseDown, proto.PhaseUp},
	} {
		t.Run(tc.name, func(t *testing.T) {
			chat := &fakeChat{text: "A shorter text."}
			a := newBoxApp(t, script(t, hotkeyLine(0, "rewrite/0", tc.startPhase)), chat)
			a.pump("the recording", func() bool { return a.box.open && a.sess.State == session.Recording })
			front := helpertest.Apps["com.tinyspeck.slackmacgap"]
			a.postQueue(func() {
				gen := a.boxGen
				seq := a.boxReserve()
				a.hotkeyFired(proto.HotkeyEventParams{Binding: "rewrite/0", Phase: tc.stopPhase, Behavior: proto.BehaviorTap, Front: &front})
				if !a.boxSendQueued || a.sess.State != session.Recording {
					t.Errorf("queued %v, session %s", a.boxSendQueued, a.sess.State)
				}
				a.boxConverted(gen, seq, boxJPEG(t), "")
			})
			a.pump("the answer", func() bool { return a.answer.open })
			reqs := chat.requests()
			if len(reqs) != 1 {
				t.Fatalf("%d chats", len(reqs))
			}
			if imgs, _ := reqs[0]["images"].([][]byte); len(imgs) != 1 {
				t.Errorf("%d images sent", len(imgs))
			}
		})
	}
}

// TestRewriteBoxDropOrderAcrossBatches: a big picture of a first batch
// converts after a second batch's small one (each batch converts on
// its own goroutine). The box and the request must keep the drop
// order: the first batch's picture stays first.
func TestRewriteBoxDropOrderAcrossBatches(t *testing.T) {
	chat := &fakeChat{text: "A shorter text."}
	a := newBoxApp(t, script(t, hotkeyLine(0, "rewrite/0", proto.PhaseTrigger)), chat)
	a.pump("the recording", func() bool { return a.box.open && a.sess.State == session.Recording })
	slow, fast := boxJPEGOf(t, 8, 8), boxJPEGOf(t, 16, 16)
	a.postQueue(func() {
		gen := a.boxGen
		first := a.boxReserve()  // batch 1's big picture, still converting
		second := a.boxReserve() // batch 2's, dropped meanwhile
		a.boxConverted(gen, second, fast, "")
		if len(a.box.images) != 0 {
			t.Errorf("%d images before the first batch's picture landed", len(a.box.images))
		}
		a.boxConverted(gen, first, slow, "")
		if len(a.box.images) != 2 || !bytes.Equal(a.box.images[0], slow) || !bytes.Equal(a.box.images[1], fast) {
			t.Errorf("the box lost the drop order: %q", a.box.images)
		}
	})
	a.postQueue(a.boxSend) // Enter: finish the recording and send
	a.pump("the answer", func() bool { return a.answer.open })
	reqs := chat.requests()
	if len(reqs) != 1 {
		t.Fatalf("%d chats", len(reqs))
	}
	imgs, _ := reqs[0]["images"].([][]byte)
	if len(imgs) != 2 || !bytes.Equal(imgs[0], slow) || !bytes.Equal(imgs[1], fast) {
		t.Errorf("the request lost the drop order: %q", imgs)
	}
}

// TestRewriteBoxDropKeepsBatchOrder: files dropped in two batches land
// in the box in the drop order, and the pending count returns to zero.
func TestRewriteBoxDropKeepsBatchOrder(t *testing.T) {
	chat := &fakeChat{text: "A shorter text."}
	a := newBoxApp(t, script(t, hotkeyLine(0, "rewrite/0", proto.PhaseTrigger)), chat)
	a.pump("the recording", func() bool { return a.box.open && a.sess.State == session.Recording })
	dir := t.TempDir()
	var files []string
	var want [][]byte
	for i, side := range []int{8, 16, 24} {
		var raw bytes.Buffer
		if err := png.Encode(&raw, image.NewRGBA(image.Rect(0, 0, side, side))); err != nil {
			t.Fatal(err)
		}
		f := filepath.Join(dir, strings.Repeat("f", i+1)+".png")
		if err := os.WriteFile(f, raw.Bytes(), 0o600); err != nil {
			t.Fatal(err)
		}
		enc, err := imgconv.JPEG(raw.Bytes())
		if err != nil {
			t.Fatal(err)
		}
		files, want = append(files, f), append(want, enc)
	}
	a.postQueue(func() { a.boxDrop(files[:1]) })
	a.postQueue(func() { a.boxDrop(files[1:]) })
	a.pump("the pictures", func() bool { return a.boxPending == 0 && len(a.box.images) == 3 })
	for i, enc := range want {
		if !bytes.Equal(a.box.images[i], enc) {
			t.Errorf("image %d is not the %dth dropped file", i, i+1)
		}
	}
}

func (a *app) boxView(context *ui.Context) {
	rewriteBoxView(context, a.boxViewState(), a.boxActions(), &a.box.ui)
}
