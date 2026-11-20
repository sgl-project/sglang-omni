package main

import (
	"bytes"
	"slices"
	"strings"
	"time"

	"github.com/egoist/mygo"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/overlay/layout"
	"github.com/sgl-project/sglang-omni/Voxt/internal/session"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
	"github.com/sgl-project/sglang-omni/Voxt/internal/uitext"
)

// boxVoiceLine is the recognised instruction, live or kept for a resend.
func (a *app) boxVoiceLine() string {
	b := &a.box
	if b.session != "" && a.sess.ID == b.session && a.sess.State.Live() {
		switch {
		case a.sess.State == session.Recording:
			return a.sess.Partial
		case a.sess.State == session.Recognizing:
			return i18n.T("Recognising…")
		case a.sess.Spoken != "":
			return a.sess.Spoken // the model works: what it was asked stays
		}
	}
	return b.instruction
}

func (a *app) boxSubtitle(selection string) string {
	if a.box.err != "" {
		return a.box.err
	}
	if a.sess.ID == "" || a.sess.ID != a.boxSessionID() || !a.sess.State.Live() {
		return ""
	}
	if a.sess.State != session.Recording {
		// The card stays while the model works (wall 122): it says so
		// where the pill did.
		if a.sess.State == session.Recognizing {
			return ""
		}
		return uitext.WorkingText(a.sess)
	}
	if selection != "" {
		return i18n.T("Listening… say how to change the selected text")
	}
	if a.boxAnswered() {
		return i18n.T("Listening… say what to change in the answer")
	}
	return i18n.T("Listening… say what to write")
}

func (a *app) boxViewState() rewriteBoxViewState {
	box := &a.box
	selectionSource := i18n.T("Selected text")
	selectionVia := box.selectionVia
	if box.session != "" && a.sess.ID == box.session {
		selectionVia = a.sess.SelectionVia
	}
	if selectionVia == "copy" {
		selectionSource = i18n.T("Copied text")
	}
	thumbnails := slices.Clone(box.thumbs)
	imageInfo := slices.Clone(box.imageInfo)
	if !box.automaticImageSuppressed && len(thumbnails) == 0 && a.sess.ID != "" && a.sess.ID == a.boxSessionID() && len(a.sess.ImagePreview) > 0 {
		if !bytes.Equal(box.autoImage, a.sess.ImagePreview) || box.autoKind != a.sess.ImagePreviewKind {
			box.autoImage = bytes.Clone(a.sess.ImagePreview)
			box.autoKind = a.sess.ImagePreviewKind
			box.autoThumb, _ = decodeBoxBitmap(a.sess.ImagePreview)
			label := i18n.T("Copied image")
			if a.sess.ImagePreviewKind == "window" {
				label = i18n.T("Screenshot")
			}
			box.autoInfo = boxImageCaption(box.autoThumb, label)
		}
		thumbnails = append(thumbnails, box.autoThumb)
		imageInfo = append(imageInfo, box.autoInfo)
	}
	selection := a.boxSelection()
	resendReview := box.sending != "" && box.shown && a.sess.ID == box.sending && a.sess.ImagePreviewKind == "window" && len(a.sess.ImagePreview) > 0
	recording := box.session != "" && a.sess.ID == box.session && a.sess.State == session.Recording
	state := rewriteBoxViewState{
		Selection: selection, Conversation: a.boxConversing(), SizeStep: a.prefs.Features.CardSize(string(store.KindRewrite)), SelectionBlock: newBoxTextSegment(selection, selectionSource, true), Text: box.text, TextSegments: slices.Clone(box.textSegments),
		Thumbnails: thumbnails, ImageInfo: imageInfo, ManualImages: len(box.images), VoiceLine: a.boxVoiceLine(), EditorPlaceholder: i18n.T("Or type here"), Subtitle: a.boxSubtitle(selection), SubtitleError: box.err != "",
		CanSend: resendReview || box.sending == "" && (box.session != "" && a.sess.ID == box.session && (a.sess.State == session.Recording || a.sess.State == session.Recognizing && a.sess.ImagePreviewKind == "window" && len(a.sess.ImagePreview) > 0) || box.session == "" && box.instruction != ""),
		Limit:   box.limit, FocusText: box.focusText, ShowTip: box.showTip,
		Recording: recording, Heard: !a.sess.HeardAt.IsZero(), Motion: a.overlayMotion(), Phase: a.overlayPhase(),
		Radius: faceRadius(a.prefs.Overlay, ovAnswer, boxWinH), Alpha: layout.FaceAlpha(a.prefs.Overlay), Palette: a.prefs.Appearance.Palette,
	}
	if state.Conversation {
		state.Card, state.CardSessionID, state.CardWidth = a.boxCardView(), a.ov.projection.view.SessionID, a.boxFaceW()
	}
	if a.boxAnswered() && box.instruction == "" && !a.sess.State.Live() && strings.TrimSpace(a.boxTextContent()) != "" {
		state.CanSend = true // a typed follow-up
	}
	if state.Subtitle == "" && a.boxAnswered() {
		state.Subtitle = a.answer.msg // "Copied", why Insert did not work
	}
	if recording {
		copySessionLevels(&state.Levels, a.sess.Levels)
	}
	return state
}

func (a *app) boxActions() rewriteBoxActions {
	return rewriteBoxActions{
		viewState: a.boxViewState,
		close:     a.boxCloseCard, escape: a.boxEscape, card: boxCardActions{a}, cardLocal: func() sessionOverlayLocal { return sessionOverlayLocal{Overlay: &a.ov.ui, Card: &a.answer.ui} }, cardMeasured: a.postSessionOverlayMeasurements, cycleSize: a.boxCycleSize, send: a.boxSend, drop: a.boxDrop, chooseImages: a.boxChooseImages, chooseFile: a.boxChooseFile,
		paste: a.boxPaste, pasteText: a.boxPasteTextAt, pasteClipboard: a.boxPasteClipboard,
		remove: a.boxRemoveImage, removeText: a.boxRemoveText, removeSelection: a.boxRemoveSelection, removeAutoImage: a.boxRemoveAutoImage, segmentChanged: a.boxSegmentChanged,
		dismissTip: a.dismissBoxTip,
		textChanged: func(text string) {
			if !a.box.open {
				return
			}
			a.box.text = text
			a.boxShare.store(a.boxTextContent(), a.box.images)
		},
		focused: func() { a.box.focusText = false },
	}
}

func (a *app) dismissBoxTip() {
	a.box.showTip = false
	if !a.prefs.Setup.RewriteBoxTipDismissed {
		a.updateSettings(func(s *settings.Settings) { s.Setup.RewriteBoxTipDismissed = true })
	}
	a.boxSync()
}

// boxSync shows, hides or refreshes the box's window. Main thread.
func (a *app) boxSync() {
	a.boxShare.store(a.boxTextContent(), a.box.images)
	a.boxShare.suppressAutomatic(a.box.selectionSuppressed, a.box.automaticImageSuppressed)
	// Insert pastes the answer: the card is out of the way meanwhile.
	// Another feature's card over the conversation has the screen until it closes.
	show := a.box.open && a.box.shown && !(a.answer.inserting && a.boxAnswered()) && !(a.answer.open && !a.boxAnswered())
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
		a.box.win = newBoxWindow(a.boxViewState, a.boxActions(), &a.box.ui, floatingWindowOptions(i18n.T("Rewrite"), boxWinW, boxWinH, nil))
	}
	style := floatingVibrancyStyle(faceRadius(a.prefs.Overlay, ovAnswer, boxWinH))
	if style != a.box.style {
		a.box.style = style
		a.box.win.SetVibrancyStyle(style)
	}
	a.box.win.Invalidate()
	if !a.box.winShown {
		a.box.area = a.overlayArea()
	}
	if frame := a.boxFrame(); frame != a.box.frame {
		a.box.frame = frame
		a.box.win.SetContentBounds(frame)
	}
	if !a.box.winShown {
		a.box.win.Show()
		a.box.win.Focus()
		a.box.winShown = true
	}
}

// boxConversing says whether the card shows the answer card's content:
// it holds an answer, or its session writes one.
func (a *app) boxConversing() bool {
	return a.boxAnswered() || a.box.open && a.sess.ID != "" && a.sess.ID == a.boxSessionID() && a.writingAnswer()
}

// boxHoldsCard says whether the Rewrite card stands in for the answer
// card's window: the overlay keeps its state and shows no window.
func (a *app) boxHoldsCard(mode overlayMode) bool {
	return mode == ovAnswer && a.box.open && a.box.shown && a.boxConversing()
}

// boxCardView is the answer card's view state, for the card to draw.
func (a *app) boxCardView() *sessionCardViewState {
	if !a.boxConversing() {
		return nil
	}
	a.refreshSessionOverlay(time.Now())
	return a.ov.projection.view.Card
}

// boxCardActions are the answer card's actions as the Rewrite card uses
// them: its text is drawn natively, the overlay window's WebView is not
// this window's.
type boxCardActions struct{ *app }

func (boxCardActions) overlayTranscriptResult(*uiContext, sessionOverlayPreviewRequest) bool {
	return false
}

// boxCycleSize is the card's size button: the next card size.
func (a *app) boxCycleSize() {
	next := a.prefs.Features.CardSize(string(store.KindRewrite)) + 1
	if next == settings.CardSizeSteps {
		next = 0
	}
	a.setCardSize(string(store.KindRewrite), next)
	a.boxSync()
}

// boxWinW and boxWinH are the box window's size at the small card
// size; boxFaceH is its face's height, at every size.
const (
	boxFaceH         = 460
	boxWinW, boxWinH = 480 + 2*layout.Pad, boxFaceH + 2*layout.Pad
)

// boxFrame is where the box's window sits: where Rewrite's answer card
// opens, as wide as its card size, so the answer takes the box's place
// without a move or a resize (wall 122).
func (a *app) boxFrame() mygo.Rectangle {
	return layout.Frame(a.box.area, a.prefs.Overlay, a.boxFaceW(), boxFaceH)
}

// boxFaceW is the card's width: Rewrite's card size.
func (a *app) boxFaceW() int {
	return layout.CardSizes[a.prefs.Features.CardSize(string(store.KindRewrite))].Width(a.box.area, a.prefs.Overlay.Margin)
}

// closeBoxWindow destroys the window (quit).
func (a *app) closeBoxWindow() {
	if a.box.win != nil {
		a.box.win.Destroy()
		a.box.win = nil
	}
}
