//go:build !js

package mygo

import (
	"encoding/json"
	"errors"
	"io/fs"
	"maps"
	"os"
	"path/filepath"
	"time"

	"github.com/egoist/mygo/internal/platform"
)

// Window state is what WindowOptions.StateKey remembers of a window, kept
// by the WindowStateStore, in window-state.json in PathUserData unless the
// app set its own. It is captured shortly after the window moves, resizes
// or changes state, when it settled (maximizing and full screen
// transitions resize the window on the way), and when it closes; it is
// saved when a window closes and when the app quits.

const windowStateFile = "window-state.json"

// How long a window's state must stay unchanged to be captured.
var windowStateDelay = 300 * time.Millisecond

// windowStates are the states of the store, loaded on first use. Main
// thread only.
var windowStates struct {
	loaded bool
	byKey  map[string]WindowState
	dirty  bool
}

func loadWindowStates() {
	if windowStates.loaded {
		return
	}
	windowStates.loaded = true
	states, err := App.windowStateStore().LoadWindowStates()
	if err != nil || states == nil { // without states, windows get their options
		states = map[string]WindowState{}
	}
	windowStates.byKey = states
}

// saveWindowStates saves the states if they changed. A save that fails
// keeps them for the next one, at the latest when the app quits.
func saveWindowStates() {
	if !windowStates.dirty {
		return
	}
	windowStates.dirty = App.windowStateStore().SaveWindowStates(maps.Clone(windowStates.byKey)) != nil
}

// windowStateStore is the store of App.SetWindowStateStore, or the file.
func (a *Application) windowStateStore() WindowStateStore {
	a.mu.Lock()
	s := a.stateStore
	a.mu.Unlock()
	if s == nil {
		return stateFile{}
	}
	return s
}

// stateFile is the default WindowStateStore: window-state.json in
// PathUserData.
type stateFile struct{}

func (stateFile) path() (string, error) {
	dir, err := App.Path(PathUserData)
	return filepath.Join(dir, windowStateFile), err
}

func (f stateFile) LoadWindowStates() (map[string]WindowState, error) {
	path, err := f.path()
	if err != nil {
		return nil, err
	}
	data, err := readAppFile(path)
	if errors.Is(err, fs.ErrNotExist) {
		return nil, nil
	} else if err != nil {
		return nil, err
	}
	var states map[string]WindowState
	if err := json.Unmarshal(data, &states); err != nil {
		return nil, err // a damaged file is ignored, and replaced by the next save
	}
	return states, nil
}

func (f stateFile) SaveWindowStates(states map[string]WindowState) error {
	path, err := f.path()
	if err != nil {
		return err
	}
	data, err := json.MarshalIndent(states, "", "  ")
	if err != nil {
		return err
	}
	// An error, as Windows still reporting the file in use after rename's
	// retries, keeps the states for the next save.
	if err := os.WriteFile(path+".tmp", data, 0o644); err != nil {
		return err
	}
	return rename(path+".tmp", path)
}

// restoreWindowState applies the state saved under key to p, if the
// window would show on a connected display. Otherwise the window keeps
// its saved size, as far as the display allows, and is centered.
func restoreWindowState(key string, p *platform.WindowOptions) {
	loadWindowStates()
	st, ok := windowStates.byKey[key]
	if !ok || st.Width <= 0 || st.Height <= 0 {
		return
	}
	r := Rectangle{st.X, st.Y, st.Width, st.Height}
	displays := backend().Screen().Displays()
	p.UseContentSize = false
	p.Maximized, p.FullScreen = st.Maximized, st.FullScreen
	if onDisplay(r, displays) {
		p.X, p.Y, p.Width, p.Height = r.X, r.Y, r.Width, r.Height
		p.Center = false
		return
	}
	p.Width, p.Height = r.Width, r.Height
	for _, d := range displays {
		if d.Primary {
			p.Width, p.Height = min(r.Width, d.WorkArea.Width), min(r.Height, d.WorkArea.Height)
		}
	}
	p.Center = true
}

// onDisplay reports whether enough of r shows on a display to reach it.
func onDisplay(r Rectangle, displays []platform.Display) bool {
	for _, d := range displays {
		a := d.WorkArea
		w := min(a.X+a.Width, r.X+r.Width) - max(a.X, r.X)
		h := min(a.Y+a.Height, r.Y+r.Height) - max(a.Y, r.Y)
		if w >= min(100, r.Width) && h >= min(50, r.Height) {
			return true
		}
	}
	return false
}

// initState records the state of a window just created with the options
// p. A window that starts maximized or full screen has not shown its normal
// bounds yet, so they are the requested ones.
func (w *Window) initState(p *platform.WindowOptions) {
	if _, ok := windowStates.byKey[w.stateKey]; !ok && (p.Maximized || p.FullScreen) {
		st := WindowState{X: p.X, Y: p.Y, Width: p.Width, Height: p.Height, Maximized: p.Maximized, FullScreen: p.FullScreen}
		if p.Center {
			for _, d := range backend().Screen().Displays() {
				if d.Primary {
					st.X = d.WorkArea.X + (d.WorkArea.Width-st.Width)/2
					st.Y = d.WorkArea.Y + (d.WorkArea.Height-st.Height)/2
				}
			}
		}
		windowStates.byKey[w.stateKey] = st
		windowStates.dirty = true
		return
	}
	w.captureState()
}

// stateChanged captures the window's state once it stopped changing.
func (w *Window) stateChanged() {
	if w.stateKey == "" {
		return
	}
	if w.stateTimer != nil {
		w.stateTimer.Reset(windowStateDelay)
		return
	}
	w.stateTimer = time.AfterFunc(windowStateDelay, func() { postMain(w.captureState) })
}

// captureState records the window's current state under its key.
func (w *Window) captureState() {
	n := w.native
	if w.stateKey == "" || n == nil {
		return
	}
	loadWindowStates()
	st := windowStates.byKey[w.stateKey]
	st.Maximized, st.FullScreen = n.IsMaximized(), n.IsFullScreen()
	if !st.Maximized && !st.FullScreen && !n.IsMinimized() {
		if b := n.Bounds(); b.Width > 0 && b.Height > 0 {
			st.X, st.Y, st.Width, st.Height = b.X, b.Y, b.Width, b.Height
		}
	}
	if st.Width <= 0 || st.Height <= 0 {
		return
	}
	if prev, ok := windowStates.byKey[w.stateKey]; !ok || prev != st {
		windowStates.byKey[w.stateKey] = st
		windowStates.dirty = true
	}
}

// closeState captures the state of a closing window for the last time,
// and writes the states.
func (w *Window) closeState() {
	if w.stateTimer != nil {
		w.stateTimer.Stop()
		w.stateTimer = nil
	}
	w.captureState()
	saveWindowStates()
}
