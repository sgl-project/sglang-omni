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
// window) stays above other windows. A card (an answer card, the Rewrite
// card) is dragged by its face and sits at the windows' level, so the
// user can put another window over it (cardLevel).

import (
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

// levelWindow is what cardLevel changes on a window (*mygo.Window).
type levelWindow interface {
	SetAlwaysOnTop(bool)
	ShowInactive()
}

// cardLevel is a floating window's level while it changes between a
// preview and a card: created above other windows, it goes to the
// windows' level as a card and comes in front then, and again with each
// new answer, so an answer the user covered shows when it changes.
type cardLevel struct {
	card   bool   // at the windows' level
	answer string // the answer it last came in front with
}

// sync puts win at a card's or a preview's level. Main thread.
func (l *cardLevel) sync(win levelWindow, card bool, answer string) {
	if card != l.card {
		win.SetAlwaysOnTop(!card)
		l.card = card
		if card {
			win.ShowInactive() // in front of the windows that may cover it later
			l.answer = answer
		}
		return
	}
	if card && answer != l.answer {
		win.ShowInactive()
		l.answer = answer
	}
}
