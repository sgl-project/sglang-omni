package main

import (
	"time"

	"github.com/egoist/mygo"
	"github.com/sgl-project/sglang-omni/Voxt/internal/overlay/layout"
	"github.com/sgl-project/sglang-omni/Voxt/internal/session"
)

// overlayState owns the session overlay's native and current-frame state.
type overlayState struct {
	overlayTranscript transcriptPreview
	bubbleTranscripts []*transcriptPreview
	ov                struct {
		win     *mygo.Window
		shown   bool
		visible bool
		mode    overlayMode
		w, h    int
		frame   mygo.Rectangle
		live    struct {
			id string
			w  int
		}
		textW       map[ovSized]ovMeasured
		area        mygo.Rectangle
		dragged     struct{ x, y int }
		style       mygo.VibrancyStyle
		passThrough bool
		reveal      bool
		fade        int
		epoch       time.Time
		native      bool
		stream      streamReveal
		projection  sessionOverlayProjection
		layout      sessionOverlayLayout
		lang        langMenuState
		ui          sessionOverlayLocalUI
	}
	laterHook   func(time.Duration, func())
	measureHook func(ovSized) (float32, float32)
	measure     func(ovSized) (ovMeasured, bool)
}

// langMenuState owns the native language or thinking menu window.
type langMenuState struct {
	win   *mygo.Window
	shown bool
	frame mygo.Rectangle
}

func (a *app) overlayNative() bool { return a.win != nil || a.ov.native }

func (a *app) later(delay time.Duration, fn func()) {
	if a.laterHook != nil {
		a.laterHook(delay, fn)
		return
	}
	if !a.overlayNative() {
		return
	}
	time.AfterFunc(delay, func() { a.post(fn) })
}

func (a *app) overlayModeNow() overlayMode {
	snapshot := a.sess
	live := snapshot.State.Live() && !snapshot.Try
	card := a.answer.open && (!live || a.cardSessionLive())
	switch {
	case a.notice != nil && a.notice.over:
		return ovNotice
	case card && a.answer.inserting:
		return ovHidden
	case card, live && a.writingAnswer(), a.liveCard():
		return ovAnswer
	case live && snapshot.State == session.Delivering:
		return ovHidden
	case live && snapshot.State == session.Recording:
		return ovRecording
	case live:
		return ovWorking
	case a.notice != nil:
		return ovNotice
	}
	return ovHidden
}

func (a *app) overlaySync() {
	a.paceDraft()
	a.refreshSessionOverlay(time.Now())
	mode := a.overlayModeNow()
	was := a.ov.shown
	if mode == ovHidden {
		a.ov.mode, a.ov.shown = mode, false
		a.ov.live.id, a.ov.live.w = "", 0
		a.ov.dragged.x, a.ov.dragged.y = 0, 0
		if was {
			a.overlayFadeOut()
		}
		a.syncLangMenu()
		return
	}
	if !was {
		a.ov.area = a.overlayArea()
	}
	width, height := a.overlayFace(mode)
	frame := a.overlayFrame(width, height)
	moved := frame != a.ov.frame
	a.ov.mode, a.ov.w, a.ov.h, a.ov.frame = mode, width, height, frame
	a.ov.shown = true
	a.ov.fade++
	a.ov.visible = true
	if a.ov.epoch.IsZero() {
		a.ov.epoch = time.Now()
	}
	defer a.syncLangMenu()
	if !a.overlayNative() {
		a.syncPassThrough()
		return
	}
	if a.ov.win == nil {
		content := sessionOverlayContent(
			func() *sessionOverlayViewState {
				a.refreshSessionOverlay(time.Now())
				return &a.ov.projection.view
			},
			func() sessionOverlayLayout { return a.currentSessionOverlayLayout() },
			a,
			sessionOverlayLocal{Overlay: &a.ov.ui, Card: &a.answer.ui},
			func() sessionOverlayAnimation {
				return sessionOverlayAnimation{Motion: a.overlayMotion(), Phase: a.overlayPhase()}
			},
			a.postSessionOverlayMeasurements,
			func() { a.ov.reveal = false; a.post(a.revealOverlay) },
		)
		opts := floatingWindowOptions("Voxt", frame.Width, frame.Height, hudContent(content))
		opts.Vibrancy = mygo.VibrancyHUD
		a.ov.win = mygo.NewWindow(opts)
		a.ov.win.OnMove(func() { a.post(a.overlayDragged) })
		a.ov.passThrough = false
	}
	a.syncPassThrough()
	if !was {
		a.ov.win.SetOpacity(0)
		a.ov.reveal = true
		sequence := a.ov.fade
		a.later(ovRevealMax, func() {
			if a.ov.fade == sequence && a.ov.shown {
				a.revealOverlay()
			}
		})
	}
	if moved {
		a.ov.win.SetContentBounds(frame)
	}
	a.syncOverlayMaterial()
	a.ov.win.Invalidate()
	if !was {
		showFloating(a.ov.win, frame.X, frame.Y)
	}
}

func (a *app) currentSessionOverlayLayout() sessionOverlayLayout {
	model := a.ov.layout
	model.Width, model.Height = a.ov.w, a.ov.h
	model.Radius = faceRadius(a.prefs.Overlay, a.ov.mode, a.ov.h)
	model.Alpha = layout.FaceAlpha(a.prefs.Overlay)
	model.Palette = a.prefs.Appearance.Palette
	model.ShowTime, model.LiveLines = a.prefs.Overlay.ShowTime, a.prefs.Overlay.LiveLines
	model.Top = a.prefs.Overlay.Position == "top"
	model.Reveal, model.Measured = a.ov.reveal, a.ov.textW
	return model
}

func (a *app) overlayFrame(width, height int) mygo.Rectangle {
	frame, area := layout.Frame(a.ov.area, a.prefs.Overlay, width, height), a.ov.area
	frame.X = min(max(frame.X+a.ov.dragged.x, area.X-layout.Pad), area.X+area.Width-frame.Width+layout.Pad)
	frame.Y = min(max(frame.Y+a.ov.dragged.y, area.Y-layout.Pad), area.Y+area.Height-frame.Height+layout.Pad)
	return frame
}

func (a *app) overlayDragged() {
	bounds := a.ov.win.ContentBounds()
	if !a.ov.shown || bounds.X == a.ov.frame.X && bounds.Y == a.ov.frame.Y {
		return
	}
	frame := layout.Frame(a.ov.area, a.prefs.Overlay, a.ov.w, a.ov.h)
	a.ov.dragged.x, a.ov.dragged.y = bounds.X-frame.X, bounds.Y-frame.Y
	a.ov.frame.X, a.ov.frame.Y = bounds.X, bounds.Y
	a.syncLangMenu()
}

func (a *app) syncPassThrough() {
	pill := &a.ov.projection.view.Pill
	pass := (a.ov.mode == ovRecording || a.ov.mode == ovWorking && !pill.CanSkip && !pill.CanThink) && !a.prefs.Overlay.LiveLines
	if pass == a.ov.passThrough {
		return
	}
	a.ov.passThrough = pass
	if a.ov.win != nil {
		a.ov.win.SetIgnoreMouseEvents(pass)
	}
}

const ovRevealMax = 120 * time.Millisecond

func (a *app) revealOverlay() {
	a.ov.reveal = false
	if a.ov.win != nil && a.ov.shown {
		a.ov.win.SetOpacity(1)
	}
}

func (a *app) syncOverlayMaterial() {
	if a.ov.win == nil {
		return
	}
	style := mygo.VibrancyStyle{Top: layout.Pad, Right: layout.Pad, Bottom: layout.Pad, Left: layout.Pad,
		Radius: float64(faceRadius(a.prefs.Overlay, a.ov.mode, a.ov.h)), Dark: true}
	if style != a.ov.style {
		a.ov.style = style
		a.ov.win.SetVibrancyStyle(style)
	}
}

func (a *app) overlayFadeOut() {
	a.ov.fade++
	sequence := a.ov.fade
	current := func() bool { return a.ov.fade == sequence && !a.ov.shown }
	hide := func() {
		if !current() {
			return
		}
		a.ov.visible = false
		if a.ov.win != nil {
			a.ov.win.Hide()
		}
	}
	if a.prefs.Appearance.ReduceMotion {
		hide()
		return
	}
	dim := func(opacity float64) func() {
		return func() {
			if current() && a.ov.win != nil {
				a.ov.win.SetOpacity(opacity)
			}
		}
	}
	dim(0.6)()
	a.later(layout.FadeStep, dim(0.25))
	a.later(2*layout.FadeStep, hide)
}

func (a *app) overlayFace(mode overlayMode) (int, int) {
	width, height, _ := a.overlaySize(mode)
	return width, height
}

func (a *app) overlayRestyle() {
	if !a.ov.shown {
		return
	}
	a.refreshSessionOverlay(time.Now())
	width, height := a.overlayFace(a.ov.mode)
	frame := a.overlayFrame(width, height)
	moved := frame != a.ov.frame
	a.ov.w, a.ov.h, a.ov.frame = width, height, frame
	a.syncLangMenu()
	a.syncPassThrough()
	if a.ov.win == nil {
		return
	}
	if moved {
		a.ov.win.SetContentBounds(frame)
	}
	a.syncOverlayMaterial()
	a.ov.win.Invalidate()
}

// invalidatePaletteOverlays redraws the overlay windows in a new palette.
func (a *app) invalidatePaletteOverlays() {
	for _, window := range []*mygo.Window{a.ov.win, a.ov.lang.win, a.meetOvWin} {
		if window != nil {
			window.Invalidate()
		}
	}
}

func (a *app) closeOverlay() {
	a.closeLangWindow()
	if a.ov.win != nil {
		a.ov.win.Destroy()
		a.ov.win = nil
	}
	a.ov.shown, a.ov.visible = false, false
}

func (a *app) overlayMotion() bool { return a.overlayNative() && !a.prefs.Appearance.ReduceMotion }

func (a *app) overlayPhase() float64 {
	if !a.overlayMotion() || a.ov.epoch.IsZero() {
		return 0
	}
	return time.Since(a.ov.epoch).Seconds()
}
