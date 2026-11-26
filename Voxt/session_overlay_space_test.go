package main

import (
	"testing"

	"github.com/egoist/mygo"
)

// spaceWin is a window on the user's Space or not.
type spaceWin struct{ on bool }

func (w *spaceWin) IsOnActiveSpace() bool { return w.on }

// Over another app's full screen the overlay shows a panel, created once,
// and back on a desktop the ordinary window again: macOS shows no
// ordinary window of an app with a Dock icon on another app's full-screen
// Space, so a Transcribe pill there was never on screen.
func TestOverlayShowsAPanelOverAFullScreenApp(t *testing.T) {
	ordinary := &spaceWin{on: true}
	made := 0
	newPanel := func() *spaceWin { made++; return &spaceWin{on: true} }

	show, keep, panel := spaceSwap(ordinary, nil, false, newPanel)
	if show != ordinary || keep != nil || panel || made != 0 {
		t.Fatalf("on a desktop: show %p, keep %p, panel %v, made %d", show, keep, panel, made)
	}

	ordinary.on = false // the user is in another app's full screen
	show, keep, panel = spaceSwap(show, keep, panel, newPanel)
	if show == ordinary || keep != ordinary || !panel || made != 1 {
		t.Fatalf("over a full-screen app: the ordinary window %v, panel %v, made %d", show == ordinary, panel, made)
	}
	again, _, _ := spaceSwap(show, keep, panel, newPanel)
	if again != show || made != 1 {
		t.Fatalf("a second show there: same panel %v, made %d", again == show, made)
	}

	ordinary.on = true
	show, keep, panel = spaceSwap(show, keep, panel, newPanel)
	if show != ordinary || keep == nil || panel || made != 1 {
		t.Fatalf("back on a desktop: the ordinary window %v, panel %v, made %d", show == ordinary, panel, made)
	}
}

// The pill's own window never takes the front: a non-activating panel,
// which shows over another app's full screen as an ordinary window does not.
func TestPillWindowNeverTakesTheFront(t *testing.T) {
	if opts := pillWindowOptions(mygo.Rectangle{Width: 220, Height: 48}, nil); !opts.NonActivating || opts.Vibrancy != mygo.VibrancyHUD {
		t.Fatalf("pill window options: non-activating %v, vibrancy %v", opts.NonActivating, opts.Vibrancy)
	}
}
