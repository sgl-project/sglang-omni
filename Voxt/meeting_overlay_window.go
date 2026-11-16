package main

import (
	"github.com/egoist/mygo"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/meeting"
	"github.com/sgl-project/sglang-omni/Voxt/internal/overlay/layout"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
)

type meetingOverlayState struct {
	meetOvWin    *mygo.Window
	meetOvShown  bool
	meetOvView   meetingOverlayViewState
	meetOvModel  meetingOverlayModel
	meetOvLocale string
	meetOvUI     meetingOverlayLocalUI
}

// meetingContentProtectionHook is set in tests to observe SetContentProtection.
var meetingContentProtectionHook func(bool)

const meetingCardSize = "meeting"

func (a *app) meetingOverlayCollapsed() bool {
	return a.meetOvUI.collapsed || a.prefs.Features.Meeting.Collapsed
}

func (a *app) rebuildMeetingOverlay() {
	a.meetOvView = buildMeetingOverlayViewState(a.meetSnap, a.meetingOverlayCollapsed(), a.prefs.Features.CardSize(meetingCardSize))
	a.meetOvModel = meetingOverlayLayoutModel(a.meetOvView, a.meetOvUI.menu)
	a.meetOvLocale = i18n.Locale()
}

func (a *app) meetingOverlayModel() meetingOverlayModel {
	a.rebuildMeetingOverlay()
	return a.meetOvModel
}

func (a *app) syncMeetingOverlay() {
	model := a.meetingOverlayModel()
	if !a.overlayNative() {
		return
	}
	show := a.meetingActive() || a.meetSnap.State == meeting.StateFinishing
	if !show {
		if a.meetOvWin != nil && a.meetOvShown {
			a.meetOvWin.Hide()
			a.meetOvShown = false
		}
		return
	}
	width, height := model.Layout.Width, model.Layout.Height
	if a.meetOvWin == nil {
		content := meetingOverlayContent(
			func() meetingOverlayViewState { return a.meetOvView },
			func() meetingOverlayModel { return a.meetOvModel },
			a,
			&a.meetOvUI,
			func() meetingOverlayAnimation {
				return meetingOverlayAnimation{motion: a.overlayMotion(), phase: a.overlayPhase()}
			},
			func() meetingOverlayFaceStyle {
				mode, faceHeight := meetingOverlayFace(a.meetOvModel)
				return meetingOverlayFaceStyle{radius: faceRadius(a.prefs.Overlay, mode, faceHeight), alpha: layout.FaceAlpha(a.prefs.Overlay)}
			},
		)
		opts := floatingWindowOptions("Voxt meeting", width, height, content)
		opts.Vibrancy = mygo.VibrancyHUD
		opts.StateKey = "meeting-overlay"
		a.meetOvWin = mygo.NewWindow(opts)
		a.meetOvWin.SetMovable(true)
	}
	at := mygo.Rectangle{}
	if a.meetOvShown {
		at = a.meetOvWin.ContentBounds()
	}
	area := a.overlayArea()
	if at.Width > 0 {
		area = mygo.Screen.DisplayMatching(at).WorkArea
	}
	frame := meetingOverlayFrame(at, area, width, height)
	x, y := frame.X, frame.Y
	if frame != at {
		a.meetOvWin.SetContentBounds(frame)
	}
	mode, faceHeight := meetingOverlayFace(model)
	a.meetOvWin.SetVibrancyStyle(mygo.VibrancyStyle{Top: layout.Pad, Right: layout.Pad, Bottom: layout.Pad, Left: layout.Pad,
		Radius: float64(faceRadius(a.prefs.Overlay, mode, faceHeight)), Dark: true})
	a.meetOvWin.Invalidate()
	a.applyMeetingOverlayProtection()
	if !a.meetOvShown {
		showFloating(a.meetOvWin, x, y)
		a.meetOvShown = true
	}
}

func meetingOverlayFrame(at, area mygo.Rectangle, width, height int) mygo.Rectangle {
	if at.Width == 0 {
		return mygo.Rectangle{X: area.X + (area.Width-width)/2, Y: area.Y + area.Height - height - 28, Width: width, Height: height}
	}
	x := max(min(at.X, area.X+area.Width-width+layout.Pad), area.X-layout.Pad)
	y := at.Y
	if at.Y+at.Height/2 > area.Y+area.Height/2 {
		y = at.Y + at.Height - height
	}
	y = max(min(y, area.Y+area.Height-height+layout.Pad), area.Y-layout.Pad)
	return mygo.Rectangle{X: x, Y: y, Width: width, Height: height}
}

func (a *app) applyMeetingOverlayProtection() {
	hide := a.prefs.Features.Meeting.HideFromSharing
	if meetingContentProtectionHook != nil {
		meetingContentProtectionHook(hide)
	}
	if a.meetOvWin != nil {
		a.meetOvWin.SetContentProtection(hide)
	}
}

func (a *app) toggleMeetingMenu(menu meetingMenu) {
	if a.meetOvUI.menu == menu {
		menu = meetingMenuNone
	}
	a.meetOvUI.menu = menu
	a.syncMeetingOverlay()
}

func (a *app) setMeetingOverlayCollapsed(collapsed bool) {
	a.meetOvUI.collapsed, a.meetOvUI.menu = collapsed, meetingMenuNone
	a.updateSettings(func(state *settings.Settings) { state.Features.Meeting.Collapsed = collapsed })
	a.syncMeetingOverlay()
}

func (a *app) setMeetingOverlayCardSize(size int) {
	a.setCardSize(meetingCardSize, size)
	a.syncMeetingOverlay()
}

func (a *app) meetingSetModeID(id string) {
	a.meetOvUI.menu = meetingMenuNone
	if meeting.Mode(id) != a.meetSnap.Mode {
		a.meetingSetMode(meeting.Mode(id))
	}
	a.syncMeetingOverlay()
}

func (a *app) copyLiveMeetingTranscript() {
	a.meetOvUI.menu = meetingMenuNone
	a.copyLiveMeetingExport()
	a.toast(i18n.T("Copied"))
	a.syncMeetingOverlay()
}

func (a *app) requestMeetingDiscard() {
	a.meetOvUI.menu = meetingMenuNone
	a.meetOvUI.abandon = true
	a.syncMeetingOverlay()
}

func (a *app) destroyMeetingOverlay() {
	if a.meetOvWin != nil {
		a.meetOvWin.Destroy()
		a.meetOvWin = nil
		a.meetOvShown = false
		a.meetOvUI.collapsed = false
	}
}
