//go:build linux && (amd64 || arm64)

package linux

import "sync"

var (
	panelOnce             sync.Once
	gdkWindowSetFunctions func(w ptr, functions uint32)
)

// GdkWMFunction: what the window manager lets the user do to a panel.
const (
	funcResize = 1 << 1
	funcMove   = 1 << 2
	funcClose  = 1 << 5
)

// initPanel makes a new window a panel (WindowOptions.Panel), before it
// shows, as launchers make theirs (Albert, Ulauncher): a utility window on
// every workspace, out of the pager, which the window manager neither
// minimizes nor maximizes (the Motif hints GDK sets). The options keep it
// above other windows and out of the taskbar.
func (w *window) initPanel() {
	if w.layer {
		return // initLayer made it a layer surface
	}
	loadFlyouts()
	panelOnce.Do(func() { mustBind(libGDK, &gdkWindowSetFunctions, "gdk_window_set_functions") })
	gtkWindowSetTypeHint(w.win, typeHintUtility)
	gtkWindowSetSkipPagerHint(w.win, true)
	gtkWindowStick(w.win)
	gtkWidgetRealize(w.win)
	gdkWindowSetFunctions(gtkWidgetGetWindow(w.win), funcResize|funcMove|funcClose)
}
