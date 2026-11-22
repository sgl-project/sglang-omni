package main

import (
	"runtime"
	"testing"
)

// TestFloatingWindowOptions: floating windows never activate VoxtNew
// (mygo patch 0003), and start hidden above other windows. A card's or a
// preview's is an ordinary window that hovers while another app is in
// front (patch 0020): tracked only while VoxtNew was active, the
// preview's Use transcript stayed hovered and its tooltip covered the
// live text.
func TestFloatingWindowOptions(t *testing.T) {
	o := floatingWindowOptions("pill", 220, 48, nil)
	if !o.NonActivating || !o.AlwaysOnTop || !o.Hidden || !o.SkipTaskbar || !o.Frameless || o.Width != 220 || o.Height != 48 {
		t.Errorf("options %+v", o)
	}
	if card := cardWindowOptions("pill", 220, 48, nil); card.NonActivating || !card.HoverWhileInactive || !card.AlwaysOnTop || !card.Hidden {
		t.Errorf("card options %+v", card)
	}
	// Off macOS nothing can be activated.
	if runtime.GOOS != "darwin" {
		if ok, _ := activateTarget(1); ok {
			t.Error("activated off macOS")
		}
	}
}

// frontRecorder records what cardFront asks of a window.
type frontRecorder struct {
	onTop           bool
	raised, focused int
}

func (w *frontRecorder) SetAlwaysOnTop(on bool) { w.onTop = on }

func (w *frontRecorder) ShowInactive() { w.raised++ }

func (w *frontRecorder) Focus() { w.focused++ }

// TestCardFront: the overlay window's preview floats above every window;
// its answer card floats at once when it opens, then VoxtNew takes the
// front with it and it is at the windows' level, where other windows cover
// it. A new answer does it again, a refresh does not. While the selection
// is read it waits; a card that opens after the user went to another app
// stays at the windows' level without taking the front.
func TestCardFront(t *testing.T) {
	win := &frontRecorder{onTop: true} // cardWindowOptions: created on top
	may := cardTake
	ask := func() cardMay { return may }
	var front overlayFront
	front.sync(win, false, true, "", ask)
	if !win.onTop || win.raised != 0 || win.focused != 0 {
		t.Fatalf("the preview: on top %v, raised %d, focused %d", win.onTop, win.raised, win.focused)
	}
	may = cardWait
	front.sync(win, true, false, "t1", ask)
	if !win.onTop || win.raised != 1 || win.focused != 0 {
		t.Fatalf("the card opens while the selection is read: on top %v, raised %d, focused %d", win.onTop, win.raised, win.focused)
	}
	may = cardTake
	front.sync(win, true, false, "t1", ask)
	if win.onTop || win.focused != 1 {
		t.Fatalf("the card takes the front: on top %v, focused %d", win.onTop, win.focused)
	}
	front.sync(win, true, false, "t1", ask)
	if win.raised != 1 || win.focused != 1 {
		t.Errorf("a refresh: raised %d, focused %d", win.raised, win.focused)
	}
	front.sync(win, true, false, "t2", ask)
	if win.onTop || win.raised != 2 || win.focused != 2 {
		t.Errorf("a new answer: on top %v, raised %d, focused %d", win.onTop, win.raised, win.focused)
	}
	front.sync(win, false, false, "", ask)
	if !win.onTop {
		t.Error("the next preview does not float")
	}
	may = cardLeave
	front.sync(win, true, false, "t3", ask)
	if win.onTop || win.raised != 2 || win.focused != 2 {
		t.Errorf("a card for a user who went elsewhere: on top %v, raised %d, focused %d", win.onTop, win.raised, win.focused)
	}
}
