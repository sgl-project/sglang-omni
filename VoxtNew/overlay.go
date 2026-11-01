package main

// The recording overlay: VoxtNew's one floating window,
// redone after Voxt's (10-15): a small pill with no buttons. It is
// non-activating (floating.go, mygo patch 0003): clicking it leaves the
// app the user dictates into in front, with its caret. It always uses the
// dark tokens, whatever the appearance.
//
//	recording  148×48 (Voxt's compact card): the feature's icon and a
//	           16-bar waveform; the recording's time when Overlay.ShowTime
//	           says so; the matched app enhancement group, short ("Team
//	           chat"); with live text (the speech model's, the text
//	           being cleaned up, the answer as the model writes it) 76
//	           high, one line of it under them, the face as wide as the
//	           text (measured) up to 260 and its padding, then the text
//	           scrolled to its newest words, the start fading out at the
//	           left edge (Voxt's)
//	working    the same pill with a five-dot loader in the waveform's
//	           place: recognising, cleaning up, translating, writing,
//	           inserting
//	notice     an error (or a warning) in the waveform's place: an icon and
//	           one line, for 5 s; a click opens where to fix it
//	answer     the answer card (overlay_card.go, Voxt's): 440 wide, as
//	           high as its text up to Voxt's 220 for it; the result that
//	           was not inserted, its actions in the header
//
// Nothing shows after an insert: the pill fades out (100 ms). Every
// action is a shortcut: the feature's shortcut stops, Escape cancels,
// releasing a long press stops.
//
// The face is a translucent dark fill (Overlay.Opacity) over the window's
// HUD material, which blurs what is behind it (mygo patch 0004), with a
// white hairline and a small shadow the window leaves room for (layout.Pad).
// Its place comes from settings.Overlay: the top or bottom of the display
// under the pointer, and the margin from that edge.

import (
	"time"

	"github.com/egoist/mygo"
	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/overlay/layout"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/session"
)

// overlayMode is what the overlay shows.
type overlayMode int

const (
	ovHidden overlayMode = iota
	ovRecording
	ovWorking
	ovNotice
	ovAnswer
)

func (m overlayMode) String() string {
	return [...]string{"hidden", "recording", "working", "notice", "answer"}[m]
}

// overlayState is the overlay's window and what it shows.
type overlayState struct {
	ov struct {
		win   *mygo.Window
		shown bool // it shows (or should: the tests have no window)
		// visible: the window is on screen; it stays while it fades out.
		visible bool
		mode    overlayMode
		w, h    int // the face's size
		// frame is the window's, in screen DIPs, as last given to it (or
		// as it would be in tests): the face and the shadow's margin
		// around it, its anchor edge (Overlay.Position) Overlay.Margin
		// from the work area's, centred. Hidden, the window keeps it, so
		// the next showing compares with it (10-15: a pill that had grown
		// with live text came back grown and empty in the next session).
		frame mygo.Rectangle
		// live is the session whose live text widened the face, and to
		// what: within it the face only grows (a revised or restarted
		// text does not make it jump back), the next starts as a pill.
		live struct {
			id string
			w  int
		}
		// textW holds the sizes of the texts the face's size depends on
		// (the live text, the group's name, a notice's line, the answer
		// card's texts, wrapped), as the overlay's view measured them
		// (overlayMeasure); a text not there yet is guessed (guessText)
		// until it is.
		textW map[ovSized]ovMeasured
		// area is the work area of the display it shows on, chosen when
		// it appears (the display under the pointer); a resize stays there.
		area mygo.Rectangle
		// style is the material's shape last given to the window.
		style mygo.VibrancyStyle
		// passThrough: the window lets the pointer through (recording and
		// working: nothing in it takes a click).
		passThrough bool
		// reveal: the window was just shown transparent; the next build of
		// its view (the new face's) makes it opaque, so that no frame of
		// the last face shows.
		reveal bool
		// fade numbers the fades out: showing again, or another fade,
		// makes the steps of an older one do nothing.
		fade int
		// epoch is when the overlay first showed: the waveform's and the
		// loader's motion count from it.
		epoch time.Time
		// native: the overlay makes its window without the main one (the
		// overlay check, overlay_check.go).
		native bool
		// The answer card's (overlay_card.go): how far its text scrolls,
		// and whose text that is; the bubble whose Copy shows its mark,
		// bubbleSeq numbering the copies for their timers.
		cardScroll    ui.ScrollState
		cardScrollKey string
		bubbleCopied  string
		bubbleSeq     int
		// lang is the card's language menu (overlay_lang.go).
		lang langMenuState
	}
	// laterHook replaces the timers of later in tests (nil: none run
	// without a window).
	laterHook func(time.Duration, func())
	// measureHook measures the face's texts (width, height) in tests,
	// which build no view (nil: the view measures them, overlayMeasure).
	measureHook func(ovSized) (float32, float32)
}

// overlayNative says whether the overlay has a real window: in the app,
// not in tests.
func (a *app) overlayNative() bool { return a.win != nil || a.ov.native }

// later runs fn on the main thread after d: the notice's and the fade's
// timers. Without a window (tests) only laterHook runs them.
func (a *app) later(d time.Duration, fn func()) {
	if a.laterHook != nil {
		a.laterHook(d, fn)
		return
	}
	if !a.overlayNative() {
		return
	}
	time.AfterFunc(d, func() { a.post(fn) })
}

// overlayModeNow decides what the overlay shows: the answer card (an
// ended session's result, with its own session on its way: a follow-up,
// another language), a live session in the pill (not a try: the page
// shows those), a notice, or nothing. A session that will end in the card
// stays in the pill until it ends, its answer one line under the row as
// the model writes it (Voxt's: the card comes with the result; 10-15:
// the answer streaming into a card of its own width was one more way to
// show text, and the card grew line by line); a session that ended shows
// nothing of its own: not the insert, which the text in place says. A session that ended
// shows nothing of its own: not the insert, which the text in place says.
func (a *app) overlayModeNow() overlayMode {
	s := a.sess
	live := s.State.Live() && !s.Try
	card := a.answer.open && (!live || a.cardSessionLive())
	switch {
	case card && a.answer.inserting:
		return ovHidden // Insert pastes: the card comes back if it could not (Voxt's)
	case card:
		return ovAnswer
	case live && s.State == session.Recording:
		return ovRecording
	case live:
		return ovWorking
	case a.notice != nil:
		return ovNotice
	}
	return ovHidden
}

// overlaySync shows, resizes or hides the overlay after a change of the
// session, the notice or the answer card. Main thread.
//
// The window is as large as the face (and the shadow's margin), so that
// the desktop around it takes clicks; it grows and shrinks about its
// anchor (the bottom edge and the centre, or the top edge), the size and
// the place set in one move: the material (patch 0004) follows it in the
// same Core Animation transaction, and MyGo draws the new size's frame
// before it commits (darwin/surface.go). The view state is the new face's
// before the window changes, so that frame is the new face. A window that
// shows again is first transparent, until its view built the new face
// (reveal): no frame of the last face shows.
func (a *app) overlaySync() {
	mode := a.overlayModeNow()
	was := a.ov.shown
	if mode == ovHidden {
		// The face stays as it was while it fades; the next showing sizes
		// it again.
		a.ov.mode, a.ov.shown = mode, false
		a.ov.live.id, a.ov.live.w = "", 0
		if was {
			a.overlayFadeOut()
		}
		a.syncLangMenu()
		return
	}
	w, h := a.overlayFace(mode)
	if !was {
		a.ov.area = a.overlayArea()
	}
	frame := layout.Frame(a.ov.area, a.prefs.Overlay, w, h)
	moved := frame != a.ov.frame
	a.ov.mode, a.ov.w, a.ov.h, a.ov.frame = mode, w, h, frame
	a.ov.shown = true
	a.ov.fade++ // a fade on its way stops where it is
	a.ov.visible = true
	if a.ov.epoch.IsZero() {
		a.ov.epoch = time.Now()
	}
	pass := mode == ovRecording || mode == ovWorking
	defer a.syncLangMenu() // overlay_lang.go: the language menu follows the card
	if !a.overlayNative() {
		a.ov.passThrough = pass
		return // tests: the state is enough
	}
	if a.ov.win == nil {
		opts := floatingWindowOptions("VoxtNew", frame.Width, frame.Height, hudContent(a.overlayView))
		opts.Vibrancy = mygo.VibrancyHUD
		a.ov.win = mygo.NewWindow(opts)
		a.ov.passThrough = false
	}
	if pass != a.ov.passThrough {
		a.ov.passThrough = pass
		a.ov.win.SetIgnoreMouseEvents(pass)
	}
	if !was {
		a.ov.win.SetOpacity(0)
		a.ov.reveal = true
		seq := a.ov.fade
		a.later(ovRevealMax, func() { // whatever the view does
			if a.ov.fade == seq && a.ov.shown {
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

// ovRevealMax is how long a window that shows again may wait for its view
// to build before it turns opaque anyway.
const ovRevealMax = 120 * time.Millisecond

// revealOverlay makes the window opaque once it shows the new face.
func (a *app) revealOverlay() {
	a.ov.reveal = false
	if a.ov.win != nil && a.ov.shown {
		a.ov.win.SetOpacity(1)
	}
}

// syncOverlayMaterial shapes the window's material to the face: inside
// the shadow's margin, with the face's corners, dark.
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

// overlayFadeOut hides the overlay: at once with reduced motion, else
// after a short fade. Showing it again meanwhile stops the fade.
func (a *app) overlayFadeOut() {
	a.ov.fade++
	seq := a.ov.fade
	current := func() bool { return a.ov.fade == seq && !a.ov.shown }
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
	dim := func(v float64) func() {
		return func() {
			if current() && a.ov.win != nil {
				a.ov.win.SetOpacity(v)
			}
		}
	}
	dim(0.6)()
	a.later(layout.FadeStep, dim(0.25))
	a.later(2*layout.FadeStep, hide)
}

// overlayFace is the face's size in mode, kept as the session's live
// width when it is a measured one with live text (a guess may be wider
// than the text: the face would never fit it).
func (a *app) overlayFace(mode overlayMode) (int, int) {
	w, h, exact := a.overlaySize(mode)
	switch {
	case h == layout.LiveH && mode != ovAnswer:
		if a.ov.live.id != a.sess.ID {
			a.ov.live.id, a.ov.live.w = a.sess.ID, 0
		}
		if exact {
			a.ov.live.w = max(a.ov.live.w, w)
		}
	case a.ov.live.id != a.sess.ID:
		a.ov.live.id, a.ov.live.w = "", 0
	}
	return w, h
}

// overlayRestyle applies a change of settings.Overlay, or of the measured
// texts, to the overlay while it shows: its place, its size (the time,
// the texts) and its face.
func (a *app) overlayRestyle() {
	if !a.ov.shown {
		return
	}
	w, h := a.overlayFace(a.ov.mode)
	frame := layout.Frame(a.ov.area, a.prefs.Overlay, w, h)
	moved := frame != a.ov.frame
	a.ov.w, a.ov.h, a.ov.frame = w, h, frame
	a.syncLangMenu()
	if a.ov.win == nil {
		return
	}
	if moved {
		a.ov.win.SetContentBounds(frame)
	}
	a.syncOverlayMaterial()
	a.ov.win.Invalidate()
}

// closeOverlay destroys the window (quit).
func (a *app) closeOverlay() {
	a.closeLangWindow()
	if a.ov.win != nil {
		a.ov.win.Destroy()
		a.ov.win = nil
	}
	a.ov.shown, a.ov.visible = false, false
}

// overlayTheme is the overlay's theme: the dark tokens, always.
func overlayTheme() *ui.Theme {
	th := *ui.DarkTheme()
	styleTheme(&th)
	return &th
}

// overlayMotion says whether the waveform and the loader move.
func (a *app) overlayMotion() bool { return a.overlayNative() && !a.prefs.Appearance.ReduceMotion }

// overlayPhase is the time the motion is at, in seconds; 0 when nothing
// moves (and in tests).
func (a *app) overlayPhase() float64 {
	if !a.overlayMotion() || a.ov.epoch.IsZero() {
		return 0
	}
	return time.Since(a.ov.epoch).Seconds()
}
