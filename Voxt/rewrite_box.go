package main

// Rewrite's input box (settings Features.Rewrite.InputBox): the Rewrite
// shortcut opens a small panel with an editable box instead of recording
// at once. The user types or pastes the text to rewrite (and up to
// boxMaxImages pictures, pasted or dropped), says the instruction as
// before (hold the shortcut, or press it again), and Enter sends: the
// same Rewrite prompt, the box's text in place of the selection
// (session StartRequest.Box). Esc closes the box without sending.
//
// The box's window is a panel (mygo WindowOptions.Panel): it floats and
// takes the keyboard without activating VoxtNew, so the app the user was
// in stays in front and gets the keyboard back when the box hides.
//
// Two sessions are the box's (session_link.go tracks them by id): the
// voice's (StartRequest.BoxVoice) only hears the instruction and keeps
// nothing; the send's (StartRequest.Box) rewrites the box's text with
// its pictures. A send that fails (a model that takes no images refuses
// them) reopens the box with the error and everything kept.

import (
	"slices"
	"time"

	"github.com/egoist/mygo"
	"github.com/egoist/mygo/transfer"
	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/helper/proto"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/imgconv"
	"github.com/sgl-project/sglang-omni/Voxt/internal/session"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

// boxMaxImages is how many pictures the box holds (and one request
// sends); the wish wall asked for few, not a gallery.
const boxMaxImages = 4

// rewriteBox is the input box's state (rewrite_box_view.go draws it).
type rewriteBox struct {
	open   bool
	shown  bool      // the window shows (it hides while the send runs)
	target proto.App // the app in front when the box opened
	text   string
	// images are the box's pictures (JPEGs), thumbs their decoded
	// bitmaps; limit: one over boxMaxImages was refused.
	images [][]byte
	thumbs []*ui.Bitmap
	limit  bool
	// instruction is the recognised spoken instruction, ready for Enter.
	instruction string
	// voice and sending are the box's sessions' ids ("" for none).
	voice, sending string
	// err is a line the box shows: the send's failure, a picture that
	// would not read.
	err string
	// focusText: the text box takes the focus on show; winShown: the
	// window is on screen (native only).
	focusText bool
	win       *mygo.Window
	winShown  bool
}

// openBox opens the input box (the Rewrite shortcut, InputBox on): the
// selected text, when there is one, starts in the box.
func (a *app) openBox(target proto.App) {
	if !a.box.open {
		a.box = rewriteBox{}
	}
	a.box.open, a.box.shown, a.box.err = true, true, ""
	a.box.target = target
	a.box.focusText = true
	a.boxSync()
	id := a.box.target.PID
	helperCall(a, proto.MethodTextSelection, proto.SelectionParams{AllowCopy: true}, 1500*time.Millisecond, func(res proto.SelectionResult, err error) {
		// Only fill a box still empty, still the same one's.
		if err != nil || res.Text == "" || !a.box.open || a.box.target.PID != id || a.box.text != "" {
			return
		}
		a.box.text = res.Text
		a.boxSync()
	})
}

// closeBox closes the box and drops its content (Esc, its Close button,
// the window's close); a voice on its way is cancelled.
func (a *app) closeBox() {
	if a.box.voice != "" && a.sess.State.Live() && a.sess.ID == a.box.voice {
		a.cancelSession()
	}
	a.box = rewriteBox{}
	a.boxSync()
}

// boxVoiceStart hears the instruction (the shortcut with the box open).
func (a *app) boxVoiceStart(hold string) {
	if !a.box.open || a.sess.State.Live() {
		return
	}
	a.box.err = ""
	id, n := a.startSessionID(session.StartRequest{Kind: store.KindRewrite, BoxVoice: true, Target: a.box.target, HoldBinding: hold})
	if n == nil && id != "" {
		a.box.voice = id
	}
	a.boxSync()
}

// boxSend sends the box (Enter): the same Rewrite prompt, the box's text
// as the source, its pictures with it. The box hides while it runs; the
// answer comes to the answer card.
func (a *app) boxSend() {
	if !a.box.open || a.box.instruction == "" || a.box.sending != "" || a.sess.State.Live() {
		return
	}
	in := session.TextInput{Raw: a.box.instruction, Selection: a.box.text}
	req := session.StartRequest{Kind: store.KindRewrite, Box: true, Target: a.box.target, Text: &in, Images: a.box.images}
	id, n := a.startSessionID(req)
	if n == nil && id != "" {
		a.box.sending, a.box.err = id, ""
		a.box.shown = false
		a.boxSync()
	}
}

// boxEnded takes the end of one of the box's sessions (sessionEnded).
func (a *app) boxEnded(s session.Snapshot) {
	switch s.ID {
	case a.box.voice:
		a.box.voice = ""
		switch s.State {
		case session.Done:
			a.box.instruction = s.Spoken
		case session.Cancelled:
			a.box = rewriteBox{} // Esc while it listened: the box closes
		}
		a.boxSync()
	case a.box.sending:
		a.box.sending = ""
		switch s.State {
		case session.Done:
			a.box = rewriteBox{} // sent: the card holds the answer
		case session.Failed:
			// Nothing of the box's is dropped: it comes back with the
			// error, the text and the pictures kept.
			a.box.err, _, _ = sessionProblemText(s.Problem)
			a.box.shown = true
		default: // cancelled while it wrote: as Esc always did
			a.box = rewriteBox{}
		}
		a.boxSync()
	}
}

// boxAttach adds a picture to the box (paste, drop), capped at
// boxMaxImages.
func (a *app) boxAttach(jpeg []byte) {
	if len(jpeg) == 0 || !a.box.open {
		return
	}
	if len(a.box.images) >= boxMaxImages {
		a.box.limit = true
		a.boxSync()
		return
	}
	bmp, _ := ui.DecodeBitmap(jpeg)
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

// boxPaste is ⌘V in the box: a picture on the clipboard goes into the
// box (text it also holds wins, as with the copied-image setting); an
// image file copied in Finder comes through the helper (HEIC too); text
// pastes as ever. It reports whether the key was taken.
func (a *app) boxPaste(svc ui.Services) bool {
	formats := svc.ClipboardFormats()
	has := func(f transfer.Format) bool { return slices.Contains(formats, f) }
	switch {
	case has(transfer.Text) || has(transfer.HTML):
		return false
	case has(transfer.PNG):
		raw, err := svc.ReadClipboardData(transfer.PNG).Read(transfer.PNG)
		if err != nil {
			return true
		}
		a.async(func() {
			jpeg, err := imgconv.JPEG(raw)
			a.post(func() {
				if err != nil {
					a.box.err = i18n.T("This image could not be read.")
				} else {
					a.boxAttach(jpeg)
				}
				a.boxSync()
			})
		})
		return true
	case has(transfer.FileList):
		helperCall(a, proto.MethodClipboardImage, proto.ClipboardImageParams{}, 3*time.Second, func(res proto.ClipboardImageResult, err error) {
			if err == nil && len(res.JPEG) > 0 {
				a.boxAttach(res.JPEG)
			}
		})
		return true
	}
	return false
}

// boxDrop takes picture files dropped on the box.
func (a *app) boxDrop(files []string) {
	if !a.box.open {
		return
	}
	a.async(func() {
		for _, f := range files {
			jpeg, err := imgconv.JPEGFile(f)
			a.post(func() {
				if err != nil {
					a.box.err = i18n.T("This image could not be read; paste it (⌘V) instead.")
				} else {
					a.boxAttach(jpeg)
				}
				a.boxSync()
			})
		}
	})
}

// boxSync shows, hides or refreshes the box's window. Main thread.
func (a *app) boxSync() {
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
