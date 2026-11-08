package main

import (
	"testing"

	"github.com/egoist/mygo"
)

// fakeWindow is the main window in a test. Close runs the close listeners as
// the window does; a listener that prevents the close keeps the window open.
// A close that goes through is recorded, not run: its OnClosed quits the app.
type fakeWindow struct {
	visible bool
	closed  bool
	onClose []func(*mygo.CloseEvent)
}

func (w *fakeWindow) Show()           { w.visible = true }
func (w *fakeWindow) Hide()           { w.visible = false }
func (w *fakeWindow) IsVisible() bool { return w.visible }

func (w *fakeWindow) OnClose(fn func(*mygo.CloseEvent)) func() {
	w.onClose = append(w.onClose, fn)
	return func() {}
}

func (w *fakeWindow) OnClosed(func()) func() { return func() {} }

func (w *fakeWindow) Close() {
	e := &mygo.CloseEvent{}
	for _, fn := range w.onClose {
		fn(e)
	}
	w.closed = !e.DefaultPrevented()
}

// findMenuItem finds the item with the label, in its submenus too.
func findMenuItem(items []*mygo.MenuItem, label string) *mygo.MenuItem {
	for _, it := range items {
		if it.Label == label {
			return it
		}
		if found := findMenuItem(it.Submenu, label); found != nil {
			return found
		}
	}
	return nil
}

// newMenuApp is a test app with a menu bar. Its hotkeys need MyGo's main loop,
// which tests lack, so the cleanup turns the menu off before the app quits.
func newMenuApp(t *testing.T) *app {
	a, _, _ := newTestApp(t)
	t.Cleanup(func() { a.tray = nil; a.hasMenu = false }) // runs before a.quit
	a.hasMenu = true                                      // buildMenu sets the menu only in the running app
	return a
}

// TestCloseWindowMenu: ⌘W closes the window (Window → Close Window), ⌘Q quits.
func TestCloseWindowMenu(t *testing.T) {
	a := newMenuApp(t)
	a.buildMenu()
	for label, acc := range map[string]string{"Close Window": "CmdOrCtrl+W", "Quit Voxt": "CmdOrCtrl+Q"} {
		it := findMenuItem(mygo.App.Menu().Items(), label)
		if it == nil {
			t.Errorf("no %q in the menu", label)
		} else if it.Accelerator != acc {
			t.Errorf("%q has %q, want %q", label, it.Accelerator, acc)
		}
	}
}

// TestCloseWindowKeepsRunning: ⌘W hides the window and VoxtNew keeps running
// (the menu bar icon is there); opening VoxtNew again shows the window.
func TestCloseWindowKeepsRunning(t *testing.T) {
	a := newMenuApp(t)
	w := &fakeWindow{visible: true}
	a.life = w
	a.tray = &mygo.Tray{} // the menu bar icon: only that it exists counts
	a.connectWindowLife()
	a.buildMenu()
	it := findMenuItem(mygo.App.Menu().Items(), "Close Window")
	if it == nil {
		t.Fatal("no Close Window item: ⌘W does nothing")
	}
	it.Click(it, nil)
	if w.IsVisible() || w.closed {
		t.Fatalf("after ⌘W: visible %v, closed %v; want hidden and not closed", w.IsVisible(), w.closed)
	}
	a.reopenWindow() // what the Dock icon does (App.OnActivate)
	if !w.IsVisible() {
		t.Error("the window did not come back")
	}
}
