package main

// Rewrite's card with the input box (rewrite_box_view.go draws it): the
// Rewrite shortcut starts the microphone and opens the card together.
// The text selected at the press shows in it as the input; while the
// card records, the user can type more text and paste or drop up to
// boxMaxImages pictures. The shortcut again (or Enter) finishes the
// recording and sends: the selected text, then the typed text and the
// pictures, then the spoken instruction, in one Rewrite request
// (session StartRequest.Box; the box's content reaches the session
// through Config.BoxInput, encoded once when each picture was pasted,
// so a send again has the same bytes). Esc closes the card without
// sending. Nothing selected, typed or pasted: the request is the speech
// alone, as before.
//
// The card's window is a panel (mygo WindowOptions.Panel): it floats
// and takes the keyboard without activating VoxtNew, so the app the
// user was in stays in front and gets the keyboard back when the card
// hides.
//
// A send that fails (a model that takes no images refuses them) brings
// the card back with the error and everything kept: Enter sends again
// with the same instruction (StartRequest.Text).

import (
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"time"

	"github.com/egoist/mygo"
	"github.com/egoist/mygo/transfer"
	"github.com/sgl-project/sglang-omni/Voxt/internal/helper/proto"
	"github.com/sgl-project/sglang-omni/Voxt/internal/hotkey"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/imgconv"
	"github.com/sgl-project/sglang-omni/Voxt/internal/session"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
	"github.com/sgl-project/sglang-omni/Voxt/internal/uitext"
)

// boxMaxImages is how many pictures the box holds (and one request
// sends); the wish wall asked for few, not a gallery.
const boxMaxImages = 4

// rewriteBox is the input box's state (rewrite_box_view.go draws it).
type rewriteBox struct {
	open   bool
	shown  bool      // the window shows (it hides while the send runs)
	target proto.App // the app in front when the box opened
	// session is the recording session's id ("" when it ended); sending
	// a resend's after a failure.
	session, sending string
	// selection is the text selected when the shortcut fired (the
	// session reads it; the box shows it).
	selection string
	text      string
	// images are the box's pictures (JPEGs), thumbs their decoded
	// bitmaps; limit: one over boxMaxImages was refused.
	images [][]byte
	thumbs []*boxThumbnail
	limit  bool
	// instruction is the recognised spoken instruction, kept so a send
	// that failed can go again with Enter.
	instruction string
	// err is a line the box shows: the send's failure, a picture that
	// would not read.
	err string
	// focusText: the text box takes the focus on show; winShown: the
	// window is on screen (native only).
	focusText bool
	win       boxWindow
	winShown  bool
}

// boxWindow is the box's panel as boxSync and closeBoxWindow use it
// (*mygo.Window; a fake in tests, so a close's hide is observable).
type boxWindow interface {
	Invalidate()
	Center()
	Show()
	Focus()
	Hide()
	Destroy()
}

// boxShare is the box's content as the sending session reads it
// (Config.BoxInput, on the session's goroutine): stored on every box
// change and every frame the card shows (a keystroke is one).
type boxShare struct {
	mu     sync.Mutex
	text   string
	images [][]byte
}

func (b *boxShare) store(text string, images [][]byte) {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.text, b.images = text, slices.Clone(images)
}

func (b *boxShare) load() (string, [][]byte) {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.text, b.images
}

// openBox opens the input box and starts the microphone with it (the
// Rewrite shortcut): the card shows what the session reads and hears,
// the typed text and the pictures going with it when the recording is
// stopped. A box that stayed open (a send failed) keeps its content.
func (a *app) openBox(target proto.App, req session.StartRequest) {
	req.Box = true
	id, n := a.startSessionID(req)
	if n != nil || id == "" {
		return // the notice says what is missing
	}
	if !a.box.open {
		a.box = rewriteBox{}
	}
	a.box.open, a.box.shown, a.box.err = true, true, ""
	a.box.target, a.box.session = target, id
	a.box.sending, a.box.instruction, a.box.selection = "", "", ""
	a.box.focusText = true
	a.boxSync()
}

// closeBox closes the box and drops its content (Esc, its Close button,
// the window's close); the recording is cancelled.
func (a *app) closeBox() {
	if a.box.session != "" && a.sess.State.Live() && a.sess.ID == a.box.session {
		a.cancelSession()
	}
	a.dropBox()
	a.boxSync()
}

// dropBox drops the box's state, closing the window first: zeroing win
// before boxSync ran would strand the panel on screen, unreachable.
func (a *app) dropBox() {
	a.closeBoxWindow()
	a.box = rewriteBox{}
	a.boxShare.store("", nil)
	a.boxGen++
	// A send queued behind a picture's conversion dies with the box: an
	// old life's conversion landing later must not send the new life.
	a.boxSendQueued = false
}

// boxSend is Enter in the box: while it records, it finishes the
// recording and the session sends; after a send failed, it sends the
// kept instruction, text and pictures again.
func (a *app) boxSend() {
	if !a.box.open {
		return
	}
	if a.boxPending > 0 {
		// A pasted or dropped picture is still converting: send when it
		// lands, so the accepted picture cannot be left behind.
		a.boxSendQueued = true
		return
	}
	if a.box.session != "" && a.sess.ID == a.box.session {
		if a.sess.State == session.Recording {
			a.stopSession()
		}
		return
	}
	if a.box.instruction == "" || a.box.sending != "" || a.sess.State.Live() {
		return
	}
	in := session.TextInput{Raw: a.box.instruction, Selection: a.box.selection}
	id, n := a.startSessionID(session.StartRequest{Kind: store.KindRewrite, Box: true, Target: a.box.target, Text: &in})
	if n == nil && id != "" {
		a.box.sending, a.box.err = id, ""
		a.box.shown = false
		a.boxSync()
	}
}

// boxEnded takes the end of one of the box's sessions (sessionEnded).
func (a *app) boxEnded(s session.Snapshot) {
	switch s.ID {
	case a.box.session:
		a.box.session = ""
	case a.box.sending:
		a.box.sending = ""
	default:
		return
	}
	switch s.State {
	case session.Done:
		a.dropBox() // sent: the card holds the answer
	case session.Failed:
		// Nothing of the box's is dropped: it comes back with the
		// error, the text and the pictures kept, the instruction ready
		// for Enter.
		a.box.err, _, _ = sessionProblemText(s.Problem)
		if s.Spoken != "" {
			a.box.instruction = s.Spoken
		}
		if s.Selection != "" {
			a.box.selection = s.Selection
		}
		a.box.shown = true
	default: // cancelled: as Esc always did
		a.dropBox()
	}
	a.boxSync()
}

// boxLanding is one converted picture waiting for its turn (boxHeld).
type boxLanding struct {
	gen     int
	jpeg    []byte
	errText string
}

// boxReserve takes a dropped or pasted picture's number: the send
// waits for it (boxPending), and it lands after every earlier number,
// so batches keep the drop order whatever converts first.
func (a *app) boxReserve() int {
	a.boxPending++
	seq := a.boxSeq
	a.boxSeq++
	return seq
}

// boxAttach adds a picture to the box (paste, drop), capped at
// boxMaxImages. The JPEG is encoded once, here, and kept: a send again
// (a retry, another thinking level, a follow-up) has the same bytes in
// the same position, which the server's prefix cache reuses.
func (a *app) boxAttach(jpeg []byte) {
	if len(jpeg) == 0 || !a.box.open {
		return
	}
	if len(a.box.images) >= boxMaxImages {
		a.box.limit = true
		a.boxSync()
		return
	}
	bmp := boxDecodeBitmap(jpeg)
	a.box.images = append(a.box.images, jpeg)
	a.box.thumbs = append(a.box.thumbs, bmp)
	a.box.limit = false
	a.boxSync()
}

// boxRemoveImage takes the picture i out of the box.
func (a *app) boxRemoveImage(i int) {
	if i < 0 || i >= len(a.box.images) {
		return
	}
	a.box.images = slices.Delete(a.box.images, i, i+1)
	a.box.thumbs = slices.Delete(a.box.thumbs, i, i+1)
	a.box.limit = false
	a.boxSync()
}

type boxPasteRoute int

const (
	boxPasteText boxPasteRoute = iota
	boxPasteDirect
	boxPasteHelper
)

func boxPasteRouteFor(formats []transfer.Format, files []string) boxPasteRoute {
	has := func(format transfer.Format) bool { return slices.Contains(formats, format) }
	switch {
	case has(transfer.FileList):
		if boxImageFiles(files) {
			return boxPasteHelper
		}
		return boxPasteText
	case has(transfer.Text) || has(transfer.HTML):
		return boxPasteText
	case has(transfer.PNG):
		return boxPasteDirect
	case len(formats) > 0:
		return boxPasteHelper
	default:
		return boxPasteText
	}
}

func boxImageFiles(files []string) bool {
	for _, path := range files {
		switch strings.ToLower(filepath.Ext(path)) {
		case ".png", ".jpg", ".jpeg", ".gif", ".heic", ".heif", ".tif", ".tiff", ".webp":
			return true
		}
	}
	return false
}

// boxPaste is ⌘V in the box: an image file copied in Finder goes through
// the helper even when the clipboard also holds its filename; other image
// formats use the helper too, while text without an image pastes as ever.
func (a *app) boxPaste(svc boxUIServices) bool {
	formats := svc.ClipboardFormats()
	var files []string
	if slices.Contains(formats, transfer.FileList) {
		files, _ = svc.ReadClipboardData(transfer.FileList).Files()
	}
	route := boxPasteRouteFor(formats, files)
	if route == boxPasteText {
		return false
	}
	if route == boxPasteHelper {
		return a.boxPasteFromHelper()
	}
	raw, err := svc.ReadClipboardData(transfer.PNG).Read(transfer.PNG)
	if err != nil {
		return true
	}
	gen := a.boxGen
	seq := a.boxReserve()
	a.async(func() {
		jpeg, err := imgconv.JPEG(raw)
		a.post(func() {
			errText := ""
			if err != nil {
				errText, jpeg = i18n.T("This image could not be read."), nil
			}
			a.boxConverted(gen, seq, jpeg, errText)
		})
	})
	return true
}

func (a *app) boxPasteFromHelper() bool {
	gen := a.boxGen
	seq := a.boxReserve()
	helperCall(a, proto.MethodClipboardImage, proto.ClipboardImageParams{}, 3*time.Second, func(res proto.ClipboardImageResult, err error) {
		if err != nil {
			res.JPEG = nil
		}
		a.boxConverted(gen, seq, res.JPEG, "")
	})
	return true
}

// boxDrop takes picture files dropped on the box.
func (a *app) boxDrop(files []string) {
	if !a.box.open {
		return
	}
	gen := a.boxGen
	seqs := make([]int, len(files))
	for i := range files {
		seqs[i] = a.boxReserve()
	}
	a.async(func() {
		for i, f := range files {
			jpeg, err := imgconv.JPEGFile(f)
			seq := seqs[i]
			a.post(func() {
				errText := ""
				if err != nil {
					errText, jpeg = i18n.T("This image could not be read; paste it (⌘V) instead."), nil
				}
				a.boxConverted(gen, seq, jpeg, errText)
			})
		}
	})
}

// boxConverted takes one converted picture on the main thread (errText
// when it would not read, "" on success or a silent skip): it lands
// only in the box's own life (a close started another) and in its
// number's turn — each batch converts on its own goroutine, so a big
// picture of an earlier batch can finish after a later batch's, and
// without the turn the box and the request would follow the finish
// order, not the drop order. A send that waited for it goes now.
func (a *app) boxConverted(gen, seq int, jpeg []byte, errText string) {
	a.boxPending--
	if a.boxHeld == nil {
		a.boxHeld = map[int]boxLanding{}
	}
	a.boxHeld[seq] = boxLanding{gen, jpeg, errText}
	for {
		land, ok := a.boxHeld[a.boxNext]
		if !ok {
			break
		}
		delete(a.boxHeld, a.boxNext)
		a.boxNext++
		if land.gen != a.boxGen {
			continue
		}
		if land.errText != "" {
			a.box.err = land.errText
		} else {
			a.boxAttach(land.jpeg)
		}
		a.boxSync()
	}
	if a.boxPending == 0 && a.boxSendQueued {
		a.boxSendQueued = false
		a.boxSend()
	}
}

// boxSync shows, hides or refreshes the box's window. Main thread.
func (a *app) boxSync() {
	a.boxShare.store(a.box.text, a.box.images)
	show := a.box.open && a.box.shown
	if !show {
		if a.box.win != nil && a.box.winShown {
			a.box.win.Hide()
		}
		a.box.winShown = false
		return
	}
	if !a.overlayNative() {
		return // tests: the state is enough
	}
	if a.box.win == nil {
		a.box.win = mygo.NewWindow(mygo.WindowOptions{
			Title: i18n.T("Rewrite"), Width: boxWinW, Height: boxWinH, UseContentSize: true,
			Hidden: true, Frameless: true, Transparent: true, Vibrancy: mygo.VibrancyHUD,
			Panel: true, DisableResize: true, DisableClose: true,
			Content: hudContent(a.boxView),
		})
	}
	a.box.win.Invalidate()
	if !a.box.winShown {
		a.box.win.Center()
		a.box.win.Show()
		a.box.win.Focus()
		a.box.winShown = true
	}
}

// boxWinW and boxWinH are the box window's size.
const boxWinW, boxWinH = 480, 340

// closeBoxWindow destroys the window (quit).
func (a *app) closeBoxWindow() {
	if a.box.win != nil {
		a.box.win.Destroy()
		a.box.win = nil
	}
}

func (a *app) boxSelection() string {
	if a.box.session != "" && a.sess.ID == a.box.session {
		return a.sess.Selection
	}
	return a.box.selection
}

func (a *app) boxVoiceLine() string {
	box := &a.box
	if box.session != "" && a.sess.ID == box.session && a.sess.State.Live() {
		switch a.sess.State {
		case session.Recording:
			if partial := a.sess.Partial; partial != "" {
				return partial
			}
			return i18n.T("Listening…")
		default:
			return i18n.T("Recognising…")
		}
	}
	if box.instruction != "" {
		return box.instruction
	}
	if keys := a.featureKeys(hotkey.Rewrite); keys != "" {
		return i18n.T("Say the instruction (%s), then Enter sends", keys)
	}
	return i18n.T("Say the instruction, then Enter sends")
}

func (a *app) boxView(context *boxUIContext) {
	box := &a.box
	a.boxShare.store(box.text, box.images)
	send := box.sending == "" && (box.session != "" && a.sess.ID == box.session && a.sess.State == session.Recording || box.session == "" && box.instruction != "")
	state := boxViewState{
		title: uitext.KindTitle(store.KindRewrite), selection: a.boxSelection(), voiceLine: a.boxVoiceLine(),
		err: box.err, send: send, limit: box.limit, text: &box.text, focusText: &box.focusText, thumbs: box.thumbs,
	}
	actions := boxViewActions{close: a.closeBox, send: a.boxSend, drop: a.boxDrop, paste: a.boxPaste, remove: a.boxRemoveImage}
	boxView(context, state, actions)
}
