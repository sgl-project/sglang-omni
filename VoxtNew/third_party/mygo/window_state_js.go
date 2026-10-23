//go:build js

package mygo

import "github.com/egoist/mygo/internal/platform"

// Browsers keep no window state (WindowOptions.StateKey): the page lays
// the windows out, and there is no PathUserData to keep the state file
// in. Builds for them leave out window_state.go, with encoding/json, and
// never call the WindowStateStore.

func saveWindowStates() {}

func restoreWindowState(key string, p *platform.WindowOptions) {}

func (w *Window) initState(p *platform.WindowOptions) {}

func (w *Window) stateChanged() {}

func (w *Window) closeState() {}
