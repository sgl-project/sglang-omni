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
// The card's window is an ordinary window (cardWindowOptions), as any
// app's: each press of the Rewrite shortcut on it is a click on it, so it
// shows above every window at once and VoxtNew comes in front with it
// once the selection is read (cardFront); the user drags it by its
// margins, a window manager moves it, and the user's next app covers it.
// Closing it gives the front back to the app the user was in
// (giveFrontBack); an insert activates that app first (deliver).
//
// A send that fails (a model that takes no images refuses them) brings
// the card back with the error and everything kept: Enter sends again
// with the same instruction (StartRequest.Text).

import (
	"path/filepath"
	"slices"
	"strings"
	"time"

	"github.com/egoist/mygo"
	"github.com/egoist/mygo/transfer"
	"github.com/sgl-project/sglang-omni/Voxt/internal/helper/proto"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/imgconv"
	"github.com/sgl-project/sglang-omni/Voxt/internal/session"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

// boxMaxImages is how many pictures the box holds (and one request
// sends); the wish wall asked for few, not a gallery.
const boxMaxImages = 4

// rewriteBox is the input box's state (rewrite_box_view.go draws it).
type rewriteBox struct {
	open   bool
	shown  bool           // the window shows
	area   mygo.Rectangle // the screen the window opened on
	frame  mygo.Rectangle
	target proto.App // the app in front when the box opened
	// dragged is how far the user moved the card from where it opens
	// (boxDraggedTo): showBox restores the shared card position after a close.
	dragged struct{ x, y int }
	// raise brings the window in front at the next boxSync: each press of
	// the Rewrite shortcut on the card (it opens it, sends, asks in it)
	// is a click on it. front is how it takes the front (cardFront).
	raise bool
	front cardFront
	// session is the recording session's id ("" when it ended); sending
	// a resend's after a failure.
	session, sending string
	// selection is the text selected when the shortcut fired (the
	// session reads it; the box shows it).
	selection         string
	selectionVia      string
	text              string
	textSegments      []boxTextSegment
	nextTextID        int
	textSegmentsDirty bool
	textCache         string
	cachedEditor      string
	textCacheValid    bool
	// images are the box's pictures (JPEGs), thumbs their decoded
	// bitmaps; limit: one over boxMaxImages was refused.
	images    [][]byte
	thumbs    []*boxBitmap
	imageInfo []boxImageInfo
	limit     bool
	// autoImage caches the clipboard/window preview's decoded bitmap.
	autoImage                []byte
	autoKind                 string
	autoThumb                *boxBitmap
	autoInfo                 boxImageInfo
	selectionSuppressed      bool
	automaticImageSuppressed bool
	// instruction is the recognised spoken instruction, kept so a send
	// that failed can go again with Enter.
	instruction string
	// err is a line the box shows: the send's failure, a picture that
	// would not read.
	err string
	// focusText: the text box takes the focus on show; winShown: the
	// window is on screen (native only).
	focusText bool
	showTip   bool
	win       boxWindow
	winShown  bool
	style     mygo.VibrancyStyle
	ui        rewriteBoxLocalUI
}

// rewriteBoxViewState is the input box's display state.
type rewriteBoxViewState struct {
	Selection string
	// Conversation: the card holds an answer, or one is being written:
	// the answer card's content shows above the text box (wall 122).
	Conversation      bool
	Card              *sessionCardViewState
	CardSessionID     string
	CardWidth         int
	SizeStep          int
	SelectionBlock    boxTextSegment
	Text              string
	TextSegments      []boxTextSegment
	Thumbnails        []*boxBitmap
	ImageInfo         []boxImageInfo
	ManualImages      int
	VoiceLine         string
	EditorPlaceholder string
	Subtitle          string
	SubtitleError     bool
	CanSend           bool
	EnterSends        bool // Enter sends: not while it records with Settings → Shortcuts' Return switch off
	Limit             bool
	FocusText         bool
	ShowTip           bool
	Recording         bool
	Heard             bool
	Levels            [session.Bars]float64
	Motion            bool
	Phase             float64
	Radius            float32
	Alpha             float32
	Palette           string
}

// rewriteBoxActions are the input box's user intents.
type rewriteBoxActions struct {
	viewState       func() rewriteBoxViewState
	close           func()
	escape          func()
	card            sessionOverlayActions // the answer card's, for the blocks drawn here
	cardLocal       func() sessionOverlayLocal
	cardMeasured    func([]sessionOverlayMeasurement)
	cycleSize       func()
	send            func()
	drop            func([]string)
	chooseImages    func()
	chooseFile      func()
	paste           func(uiServices) bool
	pasteText       func(int, string, int, int) bool
	pasteClipboard  func(uiServices, int, int, int)
	remove          func(int)
	removeText      func(int)
	removeSelection func()
	removeAutoImage func()
	textChanged     func(string)
	segmentChanged  func(int, string)
	dismissTip      func()
	focused         func()
	moved           func() // the user dragged the window
}

// boxWindow is the box's panel as boxSync and closeBoxWindow use it
// (*mygo.Window; a fake in tests, so a close's hide is observable).
type boxWindow interface {
	Invalidate()
	SetContentBounds(mygo.Rectangle)
	ContentBounds() mygo.Rectangle
	SetAlwaysOnTop(bool)
	SetVisibleOnAllWorkspaces(bool)
	ShowInactive()
	Focus()
	Hide()
	Destroy()
	SetVibrancyStyle(mygo.VibrancyStyle)
}

// openBox opens the input box and starts the microphone with it (the
// Rewrite shortcut): the card shows what the session reads and hears,
// the typed text and the pictures going with it when the recording is
// stopped. A box that stayed open (a send failed) keeps its content.
func (a *app) openBox(target proto.App, req session.StartRequest) {
	req.Box = true
	if !a.sess.State.Live() {
		a.boxShare.clearReview()
	}
	id, n := a.startSessionID(req)
	if n != nil || id == "" {
		return // the notice says what is missing
	}
	a.showBox(target, id)
}

// showBox shows the box for its session id.
func (a *app) showBox(target proto.App, id string) {
	if !a.box.open {
		a.box = rewriteBox{showTip: !a.prefs.Setup.RewriteBoxTipDismissed && a.prefs.Setup.RewriteBoxOpens < 3, dragged: a.ov.cardDragged}
		if a.prefs.Setup.RewriteBoxOpens < 3 {
			a.updateSettings(func(s *settings.Settings) {
				if s.Setup.RewriteBoxOpens < 3 {
					s.Setup.RewriteBoxOpens++
				}
			})
		}
	}
	a.box.open, a.box.shown, a.box.err, a.box.raise = true, true, "", true
	a.box.target, a.box.session = target, id
	a.box.sending, a.box.instruction, a.box.selection, a.box.selectionVia = "", "", "", ""
	a.box.selectionSuppressed, a.box.automaticImageSuppressed = false, false
	a.box.focusText = true
	a.boxSync()
}

// closeBox closes the box and drops its content: the session running is
// cancelled. An answer moved to the background (wall 116) is no longer
// the box's to cancel.
func (a *app) closeBox() {
	if id := a.boxSessionID(); id != "" && id != a.bgID && a.sess.State.Live() && a.sess.ID == id {
		a.cancelSession()
	}
	a.dropBox()
	a.boxSync()
}

// boxCloseCard is the card's Close button (and its window's close): the
// answer it holds closes with it.
func (a *app) boxCloseCard() {
	answered, target := a.boxAnswered(), a.box.target.PID
	a.closeBox()
	if answered {
		a.closeAnswer() // which gives the front back
		return
	}
	a.giveFrontBack(target)
}

// boxEscape is Esc in the card: a follow-up on its way is cancelled and
// the card keeps its answer; otherwise the card closes.
func (a *app) boxEscape() {
	if id := a.boxSessionID(); a.boxAnswered() && id != "" && a.sess.ID == id && a.sess.State.Live() {
		a.cancelSession()
		return
	}
	a.boxCloseCard()
}

// boxAnswered says whether the card holds a Rewrite answer, or the
// answer its follow-up changes (a translation's too): it then shows the
// conversation, and what is typed is a follow-up.
func (a *app) boxAnswered() bool {
	return a.box.open && a.answer.open && (a.answer.kind == store.KindRewrite || a.followUp != "" && a.followUp == a.answer.id)
}

// boxKeep keeps the card open for its answer once a session of its ends:
// the input is emptied for the next follow-up.
func (a *app) boxKeep() {
	win, winShown, style, area, frame, ui := a.box.win, a.box.winShown, a.box.style, a.box.area, a.box.frame, a.box.ui
	target, dragged, front := a.box.target, a.box.dragged, a.box.front
	a.box = rewriteBox{open: true, shown: true, target: target, focusText: true, win: win, winShown: winShown, style: style, area: area, frame: frame, dragged: dragged, front: front, ui: ui}
	a.boxShare.reset()
	a.boxShare.suppressAutomatic(false, false)
	a.boxGen++
	a.boxSendQueued = false
}

// dropBox drops the box's state, closing the window first: zeroing win
// before boxSync ran would strand the panel on screen, unreachable.
func (a *app) dropBox() {
	a.closeBoxWindow()
	a.box = rewriteBox{}
	a.boxShare.reset()
	a.boxShare.suppressAutomatic(false, false)
	a.boxGen++
	// A send queued behind a picture's conversion dies with the box: an
	// old life's conversion landing later must not send the new life.
	a.boxSendQueued = false
}

// boxShortcutSend is the Rewrite shortcut pressed again, or its hold let
// go, while the card records: like a click on the card, it brings the
// card in front of the windows over it; then it sends (boxSend).
func (a *app) boxShortcutSend() {
	a.box.raise = true
	a.boxSync()
	a.boxSend()
}

// boxSend is Enter or the shortcut's stop in the box: while it records,
// it finishes the recording and the session sends; a newly captured window
// preview waits for one more confirmation. After a send failed, it sends
// the kept instruction, text and pictures again.
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
	a.boxShare.clearError() // a failed conversion is confirmed by this send
	if a.box.session != "" && a.sess.ID == a.box.session {
		if a.sess.State == session.Recording {
			a.stopSession()
		}
		if a.sess.State == session.Recognizing && a.sess.ImagePreviewKind == "window" && len(a.sess.ImagePreview) > 0 {
			a.boxShare.approveReview()
		}
		return
	}
	if a.box.sending != "" {
		if a.sess.ID == a.box.sending && a.box.shown && a.sess.ImagePreviewKind == "window" && len(a.sess.ImagePreview) > 0 {
			a.boxShare.approveReview()
			a.boxSync()
		}
		return
	}
	if a.box.instruction == "" && !a.sess.State.Live() && a.boxAnswered() {
		// The card holds an answer: what is typed is the follow-up.
		if typed := strings.TrimSpace(a.boxTextContent()); typed != "" {
			a.followUpTyped(typed)
		}
		return
	}
	if a.box.instruction == "" || a.sess.State.Live() {
		return
	}
	a.boxShare.clearReview()
	in := session.TextInput{Raw: a.box.instruction, Selection: a.box.selection}
	req := session.StartRequest{Kind: store.KindRewrite, Box: true, Target: a.box.target, Text: &in}
	if a.followUp != "" && a.followUp == a.answer.id {
		req.Previous, req.Conversation, req.Images = a.answer.turns, a.answer.conv, a.answer.images // still the card's follow-up
	}
	id, n := a.startSessionID(req)
	if n == nil && id != "" {
		a.box.sending, a.box.err = id, ""
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
		if a.boxAnswered() {
			a.boxKeep() // the answer is in the card, which takes the next follow-up
		} else {
			a.dropBox() // inserted, or nothing to show
		}
	case session.Failed:
		// Nothing of the box's is dropped: it comes back with the
		// error, the text and the pictures kept, the instruction ready
		// for Enter. A model failure replaces an informational skip notice.
		if a.box.err == "" || s.Problem != nil && s.Problem.Code == session.ProblemLLM {
			a.box.err, _, _ = sessionProblemText(s.Problem)
		}
		if s.Spoken != "" {
			a.box.instruction = s.Spoken
		}
		if s.Selection != "" {
			a.box.selection = s.Selection
			a.box.selectionVia = s.SelectionVia
		}
		a.box.shown = true
	default: // cancelled: as Esc always did; a follow-up's leaves the answer
		if a.boxAnswered() {
			a.boxKeep()
		} else {
			a.dropBox()
		}
	}
	a.boxSync()
}

// boxReserve takes a dropped or pasted picture's number: the send
// waits for it (boxPending), and it lands after every earlier number,
// so batches keep the drop order whatever converts first.
func (a *app) boxReserve() int {
	// A new input replaces an earlier failure's hint; a failure within its
	// own batch then stays when a later file of the batch lands.
	a.box.err = ""
	a.boxShare.clearError()
	a.boxPending++
	a.boxShare.start()
	seq := a.boxSeq
	a.boxSeq++
	return seq
}

// boxAttach adds a picture to the box (paste, drop), capped at
// boxMaxImages. The JPEG is encoded once, here, and kept: a send again
// (a retry, another thinking level, a follow-up) has the same bytes in
// the same position, which the server's prefix cache reuses.
func (a *app) boxAttach(jpeg []byte, source ...string) {
	if len(jpeg) == 0 || !a.box.open {
		return
	}
	if len(a.box.images) >= boxMaxImages {
		a.box.limit = true
		a.boxSync()
		return
	}
	name := i18n.T("Pasted image")
	if len(source) > 0 && source[0] != "" {
		name = source[0]
	}
	bmp, _ := decodeBoxBitmap(jpeg)
	a.box.images = append(a.box.images, jpeg)
	a.box.thumbs = append(a.box.thumbs, bmp)
	a.box.imageInfo = append(a.box.imageInfo, boxImageCaption(bmp, name))
	a.boxSync()
}

// boxRemoveImage takes the picture i out of the box.
func (a *app) boxRemoveImage(i int) {
	if i < 0 || i >= len(a.box.images) {
		return
	}
	a.box.images = slices.Delete(a.box.images, i, i+1)
	a.box.thumbs = slices.Delete(a.box.thumbs, i, i+1)
	if i < len(a.box.imageInfo) {
		a.box.imageInfo = slices.Delete(a.box.imageInfo, i, i+1)
	}
	a.box.limit = false
	a.boxSync()
}

func (a *app) boxRemoveSelection() {
	a.box.selectionSuppressed = true
	a.boxSync()
}

func (a *app) boxRemoveAutoImage() {
	a.box.automaticImageSuppressed = true
	a.boxSync()
}

// boxClipboardImages are the clipboard's picture formats the helper reads
// (clipboard.image: PNG, JPEG, HEIC, TIFF, WebP; MyGo lists TIFF as PNG).
var boxClipboardImages = []transfer.Format{transfer.PNG, "image/jpeg", "image/heic", "image/webp"}

// boxImageFileExtensions are the copied files the helper turns into a
// picture (an image type ImageIO decodes).
var boxImageFileExtensions = []string{".png", ".jpg", ".jpeg", ".heic", ".heif", ".tif", ".tiff", ".gif", ".webp", ".bmp"}

// boxCopiedImageFile says whether the clipboard's first copied file is a
// picture: the helper reads only that one.
func boxCopiedImageFile(svc uiServices) bool {
	paths, err := svc.ReadClipboardData(transfer.FileList).Files()
	return err == nil && len(paths) > 0 && slices.Contains(boxImageFileExtensions, strings.ToLower(filepath.Ext(paths[0])))
}

// boxPaste is ⌘V in the box: a picture on the clipboard goes into the
// box. An image file copied in Finder is the picture although Finder
// also puts its name there as text (the helper reads the file, HEIC
// too); any other text wins, as with the copied-image setting (a copied
// spreadsheet range or styled text brings a picture of itself along).
// Text pastes as ever. It reports whether the key was taken.
func (a *app) boxPaste(svc uiServices) bool {
	formats := svc.ClipboardFormats()
	has := func(f transfer.Format) bool { return slices.Contains(formats, f) }
	switch {
	case has(transfer.FileList):
		if !boxCopiedImageFile(svc) {
			return false // another file: its name pastes
		}
	case has(transfer.Text), !slices.ContainsFunc(boxClipboardImages, has):
		return false
	}
	switch {
	case len(a.box.images)+a.boxPending >= boxMaxImages:
		a.box.limit = true
		a.boxSync()
		return true
	default:
		gen := a.boxGen
		seq := a.boxReserve()
		helperCall(a, proto.MethodClipboardImage, proto.ClipboardImageParams{}, 3*time.Second, func(res proto.ClipboardImageResult, err error) {
			errText := ""
			switch {
			case err != nil:
				res.JPEG, errText = nil, i18n.T("This image could not be read.")
			case res.TooLarge || len(res.JPEG) > imgconv.MaxFileBytes:
				res.JPEG, errText = nil, i18n.T("This image is too large.")
			case len(res.JPEG) == 0:
				errText = i18n.T("This image could not be read.")
			}
			a.boxConverted(gen, seq, res.JPEG, errText, i18n.T("Clipboard"))
		})
		return true
	}
}

// boxDrop takes picture files dropped on the box.
func (a *app) boxDrop(files []string) {
	if !a.box.open {
		return
	}
	remaining := max(0, boxMaxImages-len(a.box.images)-a.boxPending)
	rejected := len(files) > remaining
	files = files[:min(len(files), remaining)]
	if len(files) == 0 {
		if rejected {
			a.box.limit = true
			a.boxSync()
		}
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
				a.boxConverted(gen, seq, jpeg, errText, filepath.Base(f))
			})
		}
		if rejected {
			a.post(func() {
				if gen == a.boxGen {
					a.box.limit = true
					a.boxSync()
				}
			})
		}
	})
}
func (a *app) boxSessionID() string {
	if a.box.sending != "" {
		return a.box.sending
	}
	return a.box.session
}

// boxSelection is the selection the box shows: the session's while it
// runs, the kept one after a send failed.
func (a *app) boxSelection() string {
	if a.box.selectionSuppressed {
		return ""
	}
	if a.box.session != "" && a.sess.ID == a.box.session {
		return a.sess.Selection
	}
	return a.box.selection
}
