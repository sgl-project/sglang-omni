package main

import (
	"runtime"
	"testing"
)

// TestFloatingWindowOptions: floating windows never activate VoxtNew
// (mygo patch 0003), and start hidden above other windows.
func TestFloatingWindowOptions(t *testing.T) {
	o := floatingWindowOptions("pill", 220, 48, nil)
	if !o.NonActivating || !o.AlwaysOnTop || !o.Hidden || !o.SkipTaskbar || !o.Frameless || o.Width != 220 || o.Height != 48 {
		t.Errorf("options %+v", o)
	}
	// Off macOS nothing can be activated.
	if runtime.GOOS != "darwin" {
		if ok, _ := activateTarget(1); ok {
			t.Error("activated off macOS")
		}
	}
}

// levelRecorder records what cardLevel asks of a window.
type levelRecorder struct {
	onTop  bool
	raised int
}

func (w *levelRecorder) SetAlwaysOnTop(on bool) { w.onTop = on }

func (w *levelRecorder) ShowInactive() { w.raised++ }

// TestCardLevel: a preview stays above other windows; a card goes to
// the windows' level, so another window can cover it, and comes in
// front when it opens and with each new answer, not on every refresh.
func TestCardLevel(t *testing.T) {
	win := &levelRecorder{onTop: true} // floatingWindowOptions: created on top
	var level cardLevel
	level.sync(win, false, "")
	if !win.onTop || win.raised != 0 {
		t.Fatalf("the preview: on top %v, raised %d", win.onTop, win.raised)
	}
	level.sync(win, true, "")
	if win.onTop || win.raised != 1 {
		t.Fatalf("the card opens: on top %v, raised %d", win.onTop, win.raised)
	}
	level.sync(win, true, "")
	if win.raised != 1 {
		t.Errorf("a refresh raised the card: %d", win.raised)
	}
	level.sync(win, true, "t2")
	if win.onTop || win.raised != 2 {
		t.Errorf("a new answer: on top %v, raised %d", win.onTop, win.raised)
	}
	level.sync(win, false, "")
	if !win.onTop {
		t.Error("the next recording's preview is not on top")
	}
}
