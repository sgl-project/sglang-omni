package main

// The meeting overlay: a second floating window with the timer, level
// meters and pause / end while recording, and save / discard when history
// is off (design §4.12, collapsed layout only in C1-1b).

import (
	"fmt"
	"time"

	"github.com/egoist/mygo"
	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/meeting"
)

type meetingOverlayState struct {
	meetOvWin   *mygo.Window
	meetOvShown bool
}

const (
	meetingOvW = 340
	meetingOvH = 48
)

func (a *app) syncMeetingOverlay() {
	if !a.overlayNative() || !a.meetingEnabled() {
		return
	}
	show := a.meetingActive()
	if !show {
		if a.meetOvWin != nil && a.meetOvShown {
			a.meetOvWin.Hide()
			a.meetOvShown = false
		}
		return
	}
	area := a.overlayArea()
	x := area.X + (area.Width-meetingOvW)/2
	y := area.Y + area.Height - meetingOvH - 28
	if a.meetOvWin == nil {
		opts := floatingWindowOptions("Voxt meeting", meetingOvW, meetingOvH, hudContent(a.meetingOverlayView))
		opts.Vibrancy = mygo.VibrancyHUD
		a.meetOvWin = mygo.NewWindow(opts)
	}
	a.meetOvWin.SetContentBounds(mygo.Rectangle{X: x, Y: y, Width: meetingOvW, Height: meetingOvH})
	a.meetOvWin.Invalidate()
	if !a.meetOvShown {
		showFloating(a.meetOvWin, x, y)
		a.meetOvShown = true
	}
	a.syncMeetingOverlayProtection()
}

func (a *app) syncMeetingOverlayProtection() {
	if a.meetOvWin == nil {
		return
	}
	a.meetOvWin.SetContentProtection(a.prefs.Features.Meeting.HideFromSharing)
}

func (a *app) meetingOverlayView(c *ui.Context) {
	t := c.Theme()
	s := a.meetSnap
	ui.Row(c).Padding(px(t, 10), px(t, 8)).AlignItems(ui.Center).Gap(px(t, 8)).Children(func() {
		ui.Text(c, formatMeetingElapsed(s.Elapsed))
		ui.Text(c, fmt.Sprintf("mic %.0f · sys %.0f", s.Level[meeting.SourceMic], s.Level[meeting.SourceSystem])).TextColor(tok(t).Text2)
		ui.Box(c).Grow(1)
		if s.State == meeting.StateAsking {
			if primaryButton(c.Key("mt-save"), i18n.T("Save")).Clicked() {
				a.meetingSave()
			}
			if ghostButton(c.Key("mt-discard"), i18n.T("Discard")).Clicked() {
				a.meetingDiscard()
			}
			return
		}
		if s.State == meeting.StateRecording || s.State == meeting.StatePaused {
			pause := i18n.T("Pause")
			if s.State == meeting.StatePaused {
				pause = i18n.T("Resume")
			}
			if ghostButton(c.Key("mt-pause"), pause).Clicked() {
				a.meetingPauseResume()
			}
			if ghostButton(c.Key("mt-end"), i18n.T("End")).Clicked() {
				a.meetingFinish()
			}
			if ghostButton(c.Key("mt-abandon"), i18n.T("Cancel meeting")).Clicked() {
				a.meetingAbandon = true
			}
		}
	})
	a.meetingAbandonDialog(c)
}

func (a *app) meetingAbandonDialog(c *ui.Context) {
	t := c.Theme()
	ui.Modal(c.Key("mt-abandon-ask"), &a.meetingAbandon, func() {
		ui.Text(c, i18n.T("Cancel this meeting?")).Bold()
		ui.Text(c, i18n.T("The recording and any unsaved text are deleted. This cannot be undone.")).Width(px(t, 380))
		ui.Row(c).Gap(px(t, sp2)).Justify(ui.End).Children(func() {
			if ui.Button(c.Key("mt-abandon-no"), i18n.T("Keep recording")).AutoFocus().Clicked() {
				a.meetingAbandon = false
			}
			if recordButton(c.Key("mt-abandon-yes"), i18n.T("Cancel meeting")).Clicked() {
				a.meetingAbandon = false
				a.meetingDiscard()
			}
		})
	})
}

func formatMeetingElapsed(d time.Duration) string {
	d = d.Round(time.Second)
	h := int(d.Hours())
	m := int(d.Minutes()) % 60
	sec := int(d.Seconds()) % 60
	if h > 0 {
		return fmt.Sprintf("%d:%02d:%02d", h, m, sec)
	}
	return fmt.Sprintf("%02d:%02d", m, sec)
}

func (a *app) destroyMeetingOverlay() {
	if a.meetOvWin != nil {
		a.meetOvWin.Destroy()
		a.meetOvWin = nil
		a.meetOvShown = false
	}
}
