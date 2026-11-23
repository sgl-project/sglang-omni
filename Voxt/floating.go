package main

// Floating windows over other apps (the recording pill, the answer
// card): they must take clicks without activating VoxtNew, or the app
// the user dictates into loses its focus and caret and ⌘V pastes
// elsewhere. MyGo patch 0003 (third_party/mygo-patches) makes such a
// window a non-activating panel at creation; it never takes the keyboard,
// so it holds no text input (the cancel shortcut comes from the helper's event tap
// while a session is active). A window that needs typing is an ordinary
// one, and before inserting, VoxtNew gives the front back to the target
// with activateTarget (plan B of the M0 spike S3).
//
// A preview (the recording pill and its live text, the meeting's live
// window) floats above other windows. A card (the answer card of
// Transcribe, Translate and Rewrite, the Rewrite card) is an ordinary
// window: the user's action on it brings VoxtNew in front with it
// (cardFront), so a window manager such as Rectangle moves it and the
// user's next app covers it. Both are ordinary windows that a click
// brings in front, as any app's (cardWindowOptions).

import (
	"os"
	"time"

	"github.com/egoist/mygo"
	"github.com/sgl-project/sglang-omni/Voxt/internal/macx"
	"github.com/sgl-project/sglang-omni/Voxt/internal/overlay/layout"
)

// floatingWindowOptions are the options of a floating window of width ×
// height DIPs showing content: borderless, transparent, above other
// windows, out of the Dock and the window cycle, hidden until
// showFloating, and non-activating.
func floatingWindowOptions(title string, width, height int, content mygo.Content) mygo.WindowOptions {
	return mygo.WindowOptions{
		Title: title, Width: width, Height: height, UseContentSize: true,
		Frameless: true, Transparent: true, DisableShadow: true, AlwaysOnTop: true, Hidden: true, SkipTaskbar: true,
		DisableResize: true, DisableMinimize: true, DisableMaximize: true, DisableFullScreen: true,
		NonActivating: true,
		Content:       content,
	}
}

// cardWindowOptions are a card's or a preview's: floatingWindowOptions,
// but an ordinary window a click brings in front with VoxtNew, as any
// app's, so it is the front app's focused window a window manager such
// as Rectangle acts on. Shown with showFloating or cardFront, it does
// not take the front by itself.
func cardWindowOptions(title string, width, height int, content mygo.Content) mygo.WindowOptions {
	options := floatingWindowOptions(title, width, height, content)
	options.NonActivating = false
	// Hover follows the pointer while the user works in another app: else
	// a button hovered while VoxtNew was in front stayed hovered, its
	// tooltip over the preview's text (MyGo patch 0020).
	options.HoverWhileInactive = true
	return options
}

func floatingVibrancyStyle(radius float32) mygo.VibrancyStyle {
	return mygo.VibrancyStyle{Top: layout.Pad, Right: layout.Pad, Bottom: layout.Pad, Left: layout.Pad,
		Radius: float64(radius), Dark: true}
}

// showFloating shows a floating window at x, y (screen DIPs) on every
// Space, without activating VoxtNew. Main thread.
func showFloating(w *mygo.Window, x, y int) {
	w.SetVisibleOnAllWorkspaces(true)
	w.SetPosition(x, y)
	w.ShowInactive()
}

// activateTarget brings the app with pid back to the front (at most
// 600 ms), for a paste after a click on an ordinary VoxtNew window.
// Call it from a goroutine: it waits.
func activateTarget(pid int) (bool, time.Duration) {
	return macx.Activate(pid, 600*time.Millisecond, mygo.RunOnMain)
}

// cardWindow is what taking the front needs of a card's window
// (*mygo.Window; a fake in tests).
type cardWindow interface {
	SetAlwaysOnTop(bool)
	ShowInactive()
	Focus()
}

// cardMay says whether a card the user acted on may take the front now.
type cardMay int

const (
	// cardWait: the selection is still being read from the app in front,
	// which may get a ⌘C (session.Snapshot.ReadingSelection).
	cardWait cardMay = iota
	// cardTake: VoxtNew comes in front with the card.
	cardTake
	// cardLeave: the user went to another app meanwhile; the card stays
	// where it is, at the windows' level.
	cardLeave
)

// cardFront is how a card takes the front when the user acts on it (the
// shortcut that opens it or sends from it is a click on it), shared by
// the Rewrite card and the answer card. macOS keeps an ordinary window of
// an app that is not in front under the front app's windows, so the card
// floats above every window at once (raise), then VoxtNew becomes the
// app in front with the card as its key window and the card is back at
// the windows' level (settle): the user's next app covers it, and a
// window manager moves it. A window starts above every window.
type cardFront struct {
	lowered bool // at the windows' level
	waiting bool // raised, VoxtNew not in front yet
}

// float keeps win above every window: a preview's level.
func (c *cardFront) float(win cardWindow) {
	if c.lowered {
		win.SetAlwaysOnTop(true)
	}
	c.lowered, c.waiting = false, false
}

// raise shows win above every window at once: the user acted on it.
func (c *cardFront) raise(win cardWindow) {
	win.SetAlwaysOnTop(true)
	win.ShowInactive()
	c.lowered, c.waiting = false, true
}

// lower puts win at the windows' level, where other windows cover it.
func (c *cardFront) lower(win cardWindow) {
	if !c.lowered {
		win.SetAlwaysOnTop(false)
	}
	c.lowered, c.waiting = true, false
}

// settle finishes a raise when may allows (asked only while it waits):
// VoxtNew in front with win as its key window, then win at the windows'
// level. Main thread.
func (c *cardFront) settle(win cardWindow, may func() cardMay) {
	if !c.waiting {
		return
	}
	switch may() {
	case cardWait:
		return
	case cardTake:
		win.Focus() // an ordinary window: VoxtNew comes in front with it
	}
	c.lower(win)
}

// overlayFront is the overlay window's: a preview floats; the answer
// card takes the front (cardFront) when it opens and with each new answer.
type overlayFront struct {
	card   bool
	answer string
	front  cardFront
}

// sync follows the overlay's mode; opened: it was hidden. A card that
// opens while the user is in another app waits at the windows' level
// instead of taking the front from it. Main thread.
func (o *overlayFront) sync(win cardWindow, card, opened bool, answer string, may func() cardMay) {
	if !card {
		o.front.float(win)
		o.card = false
		return
	}
	if !o.card || opened || answer != o.answer {
		o.answer = answer
		if may() == cardLeave {
			o.front.lower(win)
		} else {
			o.front.raise(win)
		}
	}
	o.card = true
	o.front.settle(win, may)
}

// frontmostPID is the app in front's pid (main thread); tests set it.
var frontmostPID = func() int { return macx.Frontmost().PID }

// cardMayTake says whether a card the user acted on in target may take
// the front now: not while a live session reads the selection, and only
// while the user is still in target (or VoxtNew). Main thread.
func (a *app) cardMayTake(target int) cardMay {
	if a.sess.State.Live() && a.sess.ReadingSelection {
		return cardWait
	}
	if front := frontmostPID(); target == 0 || front == target || front == os.Getpid() {
		return cardTake
	}
	return cardLeave
}

// giveFrontBack hands the front back to target when a card closes and
// leaves VoxtNew in front with no window of its own open, as closing a
// launcher's window does. Main thread.
func (a *app) giveFrontBack(target int) {
	if !a.overlayNative() || target == 0 || target == os.Getpid() || frontmostPID() != os.Getpid() || a.win != nil && a.win.IsVisible() {
		return
	}
	go func() { _, _ = activateTarget(target) }() // it waits: the card is gone, nothing to report
}
