package main

import (
	"github.com/egoist/mygo"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/overlay/layout"
)

// meetingAskWindow is the window of the card that offers to record a
// detected meeting, and what it shows.
type meetingAskWindow struct {
	win   *mygo.Window
	shown bool
	view  meetingAskViewState
}

// syncMeetingAsk shows the card as its state says, or hides it. The
// window never takes the front: the user is in the meeting's app.
func (a *app) syncMeetingAsk() {
	a.ask.view = buildMeetingAskViewState(i18n.CurrentTranslator(), meetingAskInput{App: a.ask.app, Compact: a.ask.compact, MenuOpen: a.ask.menu})
	if !a.overlayNative() {
		return
	}
	if !a.ask.view.Visible {
		if a.ask.win != nil && a.ask.shown {
			a.ask.win.Hide()
			a.ask.shown = false
		}
		return
	}
	size := meetingAskLayoutModel(a.ask.view)
	if a.ask.win == nil {
		content := meetingAskContent(
			func() meetingAskViewState { return a.ask.view },
			a,
			func() meetingOverlayFaceStyle {
				return meetingOverlayFaceStyle{radius: meetingAskRadius, alpha: layout.FaceAlpha(a.prefs.Overlay), palette: a.prefs.Appearance.Palette}
			},
		)
		opts := floatingWindowOptions("Voxt meeting detected", size.Width, size.Height, content)
		opts.HoverWhileInactive = true // the pointer over it shows its close button while another app is in front
		opts.Vibrancy = mygo.VibrancyHUD
		a.ask.win = mygo.NewWindow(opts)
		a.ask.win.SetMovable(true)
	}
	at := mygo.Rectangle{}
	if a.ask.shown {
		at = a.ask.win.ContentBounds()
	}
	area := a.overlayArea()
	if at.Width > 0 {
		area = mygo.Screen.DisplayMatching(at).WorkArea
	}
	frame := meetingAskFrame(at, area, size.Width, size.Height)
	if frame != at {
		a.ask.win.SetContentBounds(frame)
	}
	a.ask.win.SetVibrancyStyle(floatingVibrancyStyle(meetingAskRadius))
	a.ask.win.Invalidate()
	if !a.ask.shown {
		showFloating(a.ask.win, frame.X, frame.Y)
		a.ask.shown = true
	}
}

func (a *app) destroyMeetingAsk() {
	if a.ask.win != nil {
		a.ask.win.Destroy()
		a.ask.win, a.ask.shown = nil, false
	}
}
