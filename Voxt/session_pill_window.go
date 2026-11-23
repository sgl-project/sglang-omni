package main

// The session pill's own window (wall 116): while the answer card holds
// the overlay window — a Rewrite or Translate answer open or still
// streaming — a Transcribe session's pill (or its notice) shows here,
// so both stay on screen and the card is never moved. When the overlay
// window shows the session itself this window hides. It mirrors the
// overlay window's lifetime (session_overlay_window.go), minus the card,
// the menus and the answer's stream pacing.

import (
	"time"

	"github.com/egoist/mygo"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/overlay/layout"
)

// pillOverlaySync keeps the pill window on pillModeNow; overlaySync
// defers to it with the overlay window's mode.
func (a *app) pillOverlaySync(main overlayMode) {
	mode := a.pillModeNow(main)
	a.refreshPillOverlay(time.Now(), mode)
	was := a.pill.shown
	if mode == ovHidden {
		a.pill.mode, a.pill.shown = mode, false
		a.pill.live.id, a.pill.live.w = "", 0
		a.pill.dragged.x, a.pill.dragged.y = 0, 0
		if was {
			a.pillFadeOut()
		}
		return
	}
	if !was {
		a.pill.area = a.overlayArea()
	}
	width, height := a.pillFace(mode)
	frame := a.pillFrame(width, height)
	moved := frame != a.pill.frame
	a.pill.mode, a.pill.w, a.pill.h, a.pill.frame = mode, width, height, frame
	a.pill.shown = true
	a.pill.fade++
	a.pill.visible = true
	if a.pill.epoch.IsZero() {
		a.pill.epoch = time.Now()
	}
	if !a.overlayNative() {
		a.syncPillPassThrough()
		return
	}
	if a.pill.win == nil {
		content := sessionOverlayContent(
			func() *sessionOverlayViewState {
				a.refreshPillOverlay(time.Now(), a.pill.mode)
				return &a.pill.projection.view
			},
			func() sessionOverlayLayout { return a.currentPillOverlayLayout() },
			a,
			sessionOverlayLocal{Overlay: &a.ov.ui, Card: &a.answer.ui},
			func() sessionOverlayAnimation {
				return sessionOverlayAnimation{Motion: a.overlayMotion(), Phase: a.pillPhase()}
			},
			a.postPillOverlayMeasurements,
			func() { a.pill.reveal = false; a.post(a.revealPillOverlay) },
		)
		opts := cardWindowOptions("Voxt", frame.Width, frame.Height, hudContent(content))
		opts.Vibrancy = mygo.VibrancyHUD
		a.pill.win = mygo.NewWindow(opts)
		a.pill.win.OnMove(func() { a.post(a.pillDragged) })
		a.pill.passThrough = false
	}
	a.syncPillPassThrough()
	if !was {
		a.pill.win.SetOpacity(0)
		a.pill.reveal = true
		sequence := a.pill.fade
		a.later(ovRevealMax, func() {
			if a.pill.fade == sequence && a.pill.shown {
				a.revealPillOverlay()
			}
		})
	}
	if moved {
		a.pill.win.SetContentBounds(frame)
	}
	a.syncPillOverlayMaterial()
	a.pill.win.Invalidate()
	if !was {
		showFloating(a.pill.win, frame.X, frame.Y)
	}
}

// refreshPillOverlay updates the pill window's projection: the session
// pill or the notice, never the card or its menus.
func (a *app) refreshPillOverlay(now time.Time, mode overlayMode) {
	projection := &a.pill.projection
	locale := i18n.Locale()
	if projection.translator == nil || projection.locale != locale {
		projection.locale = locale
		projection.translator = i18n.NewTranslator(locale)
		projection.cardReady, projection.menuReady = false, false
	}
	input := sessionOverlayProjectionInput{
		Snapshot: a.sess, Mode: mode,
		HideLiveText: a.prefs.Overlay.HideLiveText, ShowTime: a.prefs.Overlay.ShowTime,
		PillNoteID: a.pillNote.id, PillNoteText: a.pillNote.text, Notice: a.notice,
	}
	// shortcut: no thinking menu here — langMenuWanted anchors it to the
	// overlay window's working pill, which this window is not; upgrade by
	// teaching the menu about both windows.
	next := &projection.view
	next.Revision, next.SessionID, next.State, next.Mode = a.sess.Rev, a.sess.ID, string(a.sess.State), mode
	next.ProblemCode = ""
	if a.sess.Problem != nil {
		next.ProblemCode = a.sess.Problem.Code
	}
	if mode == ovRecording || mode == ovWorking {
		key := sessionPillKeyFor(input, locale, now)
		if !projection.pillReady || key != projection.pillKey {
			next.Pill = buildSessionPillViewState(projection.translator, input, now)
			projection.pillKey, projection.pillReady = key, true
		}
		copySessionLevels(&next.Pill.Levels, a.sess.Levels)
	} else {
		next.Pill = sessionPillViewState{}
		projection.pillReady = false
	}
	next.Card, next.Menu, next.Notice = nil, nil, nil
	if mode == ovNotice && a.notice != nil {
		key := sessionNoticeProjectionKey{
			sequence: a.notice.seq, text: a.notice.text, action: a.notice.action,
			locale: locale, warn: a.notice.warn, success: a.notice.ok,
		}
		if !projection.noticeReady || key != projection.noticeKey {
			projection.notice = buildSessionNoticeViewState(projection.translator, *a.notice)
			projection.noticeKey, projection.noticeReady = key, true
		}
		next.Notice = &projection.notice
	}
}

func (a *app) currentPillOverlayLayout() sessionOverlayLayout {
	model := a.pill.layout
	model.Width, model.Height = a.pill.w, a.pill.h
	model.Radius = faceRadius(a.prefs.Overlay, a.pill.mode, a.pill.h)
	model.Alpha = layout.FaceAlpha(a.prefs.Overlay)
	model.Palette = a.prefs.Appearance.Palette
	model.ShowTime, model.LiveLines = a.prefs.Overlay.ShowTime, a.prefs.Overlay.LiveLines
	model.Top = a.prefs.Overlay.Position == "top"
	model.Reveal, model.Measured = a.pill.reveal, a.pill.textW
	return model
}

func (a *app) pillFace(mode overlayMode) (int, int) {
	width, height, _ := a.pillOverlaySize(mode)
	return width, height
}

// pillOverlaySize sizes the pill like the overlay window's pill, without
// the card's bubble previews (they are the overlay window's).
func (a *app) pillOverlaySize(mode overlayMode) (width, height int, exact bool) {
	overlay := a.prefs.Overlay
	if a.measure == nil {
		a.measure = a.ovText
	}
	model := sessionOverlayLayoutFor(&a.pill.projection.view, sessionOverlayLayoutInput{
		Measure:  a.measure,
		WorkArea: a.pill.area,
		Overlay: sessionOverlayLayoutStyle{
			Position: overlay.Position, Opacity: overlay.Opacity, Radius: overlay.Radius, Margin: overlay.Margin,
			ShowTime: overlay.ShowTime, LiveLines: overlay.LiveLines,
		},
		Memory:   sessionOverlayLayoutMemory{LiveID: a.pill.live.id, LiveWidth: a.pill.live.w},
		Measured: a.pill.textW,
		Reveal:   a.pill.reveal,
	})
	a.pill.live.id, a.pill.live.w = model.LiveID, model.LiveWidth
	a.pill.layout = model
	return model.Width, model.Height, model.Exact
}

// pillAnchor is the pill window's frame before clamping and the user's
// drag: beside the card's current frame — above a bottom-anchored card,
// below a top-anchored one (adjacent never intersects it). The default
// anchor when the card is not showing.
func (a *app) pillAnchor(width, height int) mygo.Rectangle {
	frame := layout.Frame(a.pill.area, a.prefs.Overlay, width, height)
	if a.ov.shown && a.ov.mode == ovAnswer {
		if a.prefs.Overlay.Position == "top" {
			frame.Y = a.ov.frame.Y + a.ov.frame.Height
		} else {
			frame.Y = a.ov.frame.Y - frame.Height
		}
	}
	return frame
}

func (a *app) pillFrame(width, height int) mygo.Rectangle {
	frame, area := a.pillAnchor(width, height), a.pill.area
	frame.X = min(max(frame.X+a.pill.dragged.x, area.X-layout.Pad), area.X+area.Width-frame.Width+layout.Pad)
	frame.Y = a.pillFrameY(frame, frame.Y+a.pill.dragged.y)
	return frame
}

// pillFrameY clamps y to the work area — unless that puts the pill over
// the card: then the card's other side, and when a tall card leaves no
// room on either, the roomier side adjacent to the card without the
// clamp. The work area's far edge may be crossed; the card is never
// covered.
func (a *app) pillFrameY(frame mygo.Rectangle, y int) int {
	area := a.pill.area
	clamp := func(v int) int {
		return min(max(v, area.Y-layout.Pad), area.Y+area.Height-frame.Height+layout.Pad)
	}
	if cy := clamp(y); !a.pillCoversCard(frame.X, cy, frame.Width, frame.Height) {
		return cy
	}
	card := a.ov.frame
	other := card.Y + card.Height
	if a.prefs.Overlay.Position == "top" {
		other = card.Y - frame.Height
	}
	if cy := clamp(other); !a.pillCoversCard(frame.X, cy, frame.Width, frame.Height) {
		return cy
	}
	if card.Y-area.Y >= area.Y+area.Height-(card.Y+card.Height) {
		return card.Y - frame.Height
	}
	return card.Y + card.Height
}

// pillCoversCard reports whether a pill frame at (x, y) intersects the
// answer card's frame.
func (a *app) pillCoversCard(x, y, w, h int) bool {
	if !a.ov.shown || a.ov.mode != ovAnswer {
		return false
	}
	c := a.ov.frame
	return x < c.X+c.Width && c.X < x+w && y < c.Y+c.Height && c.Y < y+h
}

// pillDragged keeps the offset the pill window was dragged by.
func (a *app) pillDragged() {
	bounds := a.pill.win.ContentBounds()
	if !a.pill.shown || bounds.X == a.pill.frame.X && bounds.Y == a.pill.frame.Y {
		return
	}
	anchor := a.pillAnchor(a.pill.w, a.pill.h)
	a.pill.dragged.x, a.pill.dragged.y = bounds.X-anchor.X, bounds.Y-anchor.Y
	a.pill.frame.X, a.pill.frame.Y = bounds.X, bounds.Y
}

// postPillOverlayMeasurements is the pill window's
// postSessionOverlayMeasurements.
func (a *app) postPillOverlayMeasurements(measurements []sessionOverlayMeasurement) {
	if len(measurements) == 0 {
		return
	}
	a.post(func() {
		if a.pill.textW == nil || len(a.pill.textW) > 256 {
			a.pill.textW = map[ovSized]ovMeasured{}
		}
		changed := false
		for _, measurement := range measurements {
			measured := ovMeasured{measurement.Width, measurement.Height}
			if previous, ok := a.pill.textW[measurement.Key]; !ok || previous != measured {
				a.pill.textW[measurement.Key], changed = measured, true
			}
		}
		if changed {
			a.pillOverlayRestyle()
		}
	})
}

// pillOverlayRestyle resizes the pill window to its measured text
// (overlayRestyle's).
func (a *app) pillOverlayRestyle() {
	if !a.pill.shown {
		return
	}
	a.refreshPillOverlay(time.Now(), a.pill.mode)
	width, height := a.pillFace(a.pill.mode)
	frame := a.pillFrame(width, height)
	moved := frame != a.pill.frame
	a.pill.w, a.pill.h, a.pill.frame = width, height, frame
	a.syncPillPassThrough()
	if a.pill.win == nil {
		return
	}
	if moved {
		a.pill.win.SetContentBounds(frame)
	}
	a.syncPillOverlayMaterial()
	a.pill.win.Invalidate()
}

func (a *app) syncPillPassThrough() {
	pill := &a.pill.projection.view.Pill
	pass := (a.pill.mode == ovRecording || a.pill.mode == ovWorking && !pill.CanSkip && !pill.CanThink) && !a.prefs.Overlay.LiveLines
	if pass == a.pill.passThrough {
		return
	}
	a.pill.passThrough = pass
	if a.pill.win != nil {
		a.pill.win.SetIgnoreMouseEvents(pass)
	}
}

func (a *app) syncPillOverlayMaterial() {
	if a.pill.win == nil {
		return
	}
	style := mygo.VibrancyStyle{Top: layout.Pad, Right: layout.Pad, Bottom: layout.Pad, Left: layout.Pad,
		Radius: float64(faceRadius(a.prefs.Overlay, a.pill.mode, a.pill.h)), Dark: true}
	if style != a.pill.style {
		a.pill.style = style
		a.pill.win.SetVibrancyStyle(style)
	}
}

// pillFadeOut hides the pill window as the overlay window fades out.
func (a *app) pillFadeOut() {
	a.pill.fade++
	sequence := a.pill.fade
	current := func() bool { return a.pill.fade == sequence && !a.pill.shown }
	hide := func() {
		if !current() {
			return
		}
		a.pill.visible = false
		if a.pill.win != nil {
			a.pill.win.Hide()
		}
	}
	if a.prefs.Appearance.ReduceMotion {
		hide()
		return
	}
	dim := func(opacity float64) func() {
		return func() {
			if current() && a.pill.win != nil {
				a.pill.win.SetOpacity(opacity)
			}
		}
	}
	dim(0.6)()
	a.later(layout.FadeStep, dim(0.25))
	a.later(2*layout.FadeStep, hide)
}

func (a *app) revealPillOverlay() {
	a.pill.reveal = false
	if a.pill.win != nil && a.pill.shown {
		a.pill.win.SetOpacity(1)
	}
}

func (a *app) pillPhase() float64 {
	if !a.overlayMotion() || a.pill.epoch.IsZero() {
		return 0
	}
	return time.Since(a.pill.epoch).Seconds()
}

// closePillOverlay destroys the pill window (closeOverlay's).
func (a *app) closePillOverlay() {
	if a.pill.win != nil {
		a.pill.win.Destroy()
		a.pill.win = nil
	}
	a.pill.shown, a.pill.visible = false, false
}
