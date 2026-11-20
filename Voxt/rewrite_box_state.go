package main

import (
	"bytes"
	"slices"

	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/overlay/layout"
	"github.com/sgl-project/sglang-omni/Voxt/internal/session"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
)

// boxVoiceLine is the recognised instruction, live or kept for a resend.
func (a *app) boxVoiceLine() string {
	b := &a.box
	if b.session != "" && a.sess.ID == b.session && a.sess.State.Live() {
		if a.sess.State == session.Recording {
			return a.sess.Partial
		}
		return i18n.T("Recognising…")
	}
	return b.instruction
}

func (a *app) boxSubtitle(selection string) string {
	if a.box.err != "" {
		return a.box.err
	}
	if a.box.session == "" || a.sess.ID != a.box.session || a.sess.State != session.Recording {
		return ""
	}
	if selection != "" {
		return i18n.T("Listening… say how to change the selected text")
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
		Selection: selection, SelectionBlock: newBoxTextSegment(selection, selectionSource, true), Text: box.text, TextSegments: slices.Clone(box.textSegments),
		Thumbnails: thumbnails, ImageInfo: imageInfo, ManualImages: len(box.images), VoiceLine: a.boxVoiceLine(), EditorPlaceholder: i18n.T("Add text (optional)"), Subtitle: a.boxSubtitle(selection), SubtitleError: box.err != "",
		CanSend: resendReview || box.sending == "" && (box.session != "" && a.sess.ID == box.session && (a.sess.State == session.Recording || a.sess.State == session.Recognizing && a.sess.ImagePreviewKind == "window" && len(a.sess.ImagePreview) > 0) || box.session == "" && box.instruction != ""),
		Limit:   box.limit, FocusText: box.focusText, ShowTip: box.showTip,
		Recording: recording, Heard: !a.sess.HeardAt.IsZero(), Motion: a.overlayMotion(), Phase: a.overlayPhase(),
		Radius: faceRadius(a.prefs.Overlay, ovAnswer, boxWinH), Alpha: layout.FaceAlpha(a.prefs.Overlay), Palette: a.prefs.Appearance.Palette,
	}
	if recording {
		copySessionLevels(&state.Levels, a.sess.Levels)
	}
	return state
}

func (a *app) boxActions() rewriteBoxActions {
	return rewriteBoxActions{
		viewState: a.boxViewState,
		close:     a.closeBox, send: a.boxSend, drop: a.boxDrop, chooseImages: a.boxChooseImages, chooseFile: a.boxChooseFile,
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
		a.box.win = newBoxWindow(a.boxViewState, a.boxActions(), &a.box.ui, floatingWindowOptions(i18n.T("Rewrite"), boxWinW, boxWinH, nil))
	}
	style := floatingVibrancyStyle(faceRadius(a.prefs.Overlay, ovAnswer, boxWinH))
	if style != a.box.style {
		a.box.style = style
		a.box.win.SetVibrancyStyle(style)
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
const boxWinW, boxWinH = 480 + 2*layout.Pad, 340 + 2*layout.Pad

// closeBoxWindow destroys the window (quit).
func (a *app) closeBoxWindow() {
	if a.box.win != nil {
		a.box.win.Destroy()
		a.box.win = nil
	}
}
