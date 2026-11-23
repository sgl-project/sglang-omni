//go:build linux && (amd64 || arm64)

package linux

import (
	"os"
	"path/filepath"
	"sync"

	"github.com/egoist/mygo/internal/platform"
)

// gtk-layer-shell makes panels surfaces of the wlr-layer-shell protocol
// on Wayland compositors that speak it (Sway, Hyprland, KDE Plasma,
// COSMIC…), as launchers there do (Ulauncher, Walker, anyrun): above
// other windows, full-screen ones too, on the output where they are put,
// out of the taskbar. Wayland lets other windows do none of that. The
// library is optional: without it, or where the compositor does not speak
// the protocol (GNOME), a panel is a window as others.
var (
	layerOnce      sync.Once
	layerSupported bool

	gtkLayerIsSupported              func() bool
	gtkLayerInitForWindow            func(w ptr)
	gtkLayerSetNamespace             func(w ptr, namespace *byte)
	gtkLayerSetLayer                 func(w ptr, layer int32)
	gtkLayerSetAnchor                func(w ptr, edge int32, anchor bool)
	gtkLayerSetMargin                func(w ptr, edge int32, margin int32)
	gtkLayerSetMonitor               func(w ptr, monitor ptr)
	gtkLayerSetKeyboardMode          func(w ptr, mode int32) // 0.6
	gtkLayerSetKeyboardInteractivity func(w ptr, v bool)     // before 0.6
	gdkDisplayGetMonitorAtPoint      func(d ptr, x, y int32) ptr
)

const (
	layerOverlay = 3 // GTK_LAYER_SHELL_LAYER_OVERLAY

	edgeLeft = 0 // GtkLayerShellEdge
	edgeTop  = 2

	keyboardExclusive = 1 // GtkLayerShellKeyboardMode
	keyboardOnDemand  = 2
)

// loadLayerShell loads gtk-layer-shell once, and reports whether panels
// can be layer surfaces.
func loadLayerShell() bool {
	layerOnce.Do(func() {
		lib, err := open("libgtk-layer-shell.so.0")
		if err != nil {
			return
		}
		ok := bind(lib, &gtkLayerIsSupported, "gtk_layer_is_supported") &&
			bind(lib, &gtkLayerInitForWindow, "gtk_layer_init_for_window") &&
			bind(lib, &gtkLayerSetNamespace, "gtk_layer_set_namespace") &&
			bind(lib, &gtkLayerSetLayer, "gtk_layer_set_layer") &&
			bind(lib, &gtkLayerSetAnchor, "gtk_layer_set_anchor") &&
			bind(lib, &gtkLayerSetMargin, "gtk_layer_set_margin") &&
			bind(lib, &gtkLayerSetMonitor, "gtk_layer_set_monitor") &&
			bind(libGDK, &gdkDisplayGetMonitorAtPoint, "gdk_display_get_monitor_at_point")
		if !bind(lib, &gtkLayerSetKeyboardMode, "gtk_layer_set_keyboard_mode") {
			ok = ok && bind(lib, &gtkLayerSetKeyboardInteractivity, "gtk_layer_set_keyboard_interactivity")
		}
		layerSupported = ok && gtkLayerIsSupported()
	})
	return layerSupported
}

// initLayer makes a new panel a layer surface where it can be one, before
// GTK realizes it, and reports whether it did. Without a position, the
// compositor centers it on the output it chooses, the one the user is on.
func (w *window) initLayer() bool {
	if w.b.onX11 || !loadLayerShell() {
		return false
	}
	gtkLayerInitForWindow(w.win)
	// The namespace tells compositors' rules which surfaces are the app's.
	name := w.b.name
	if name == "" {
		name = filepath.Base(os.Args[0])
	}
	gtkLayerSetNamespace(w.win, cs(name))
	gtkLayerSetLayer(w.win, layerOverlay)
	w.layer = true
	w.layerKeyboard(true)
	if !w.opts.Center {
		w.placeLayer(w.opts.X, w.opts.Y)
	}
	return true
}

// layerKeyboard asks for the keyboard of a layer surface as it shows,
// or not, as ShowInactive shows it. Compositors give the keyboard to a
// surface that asks for it alone (exclusive), not to one that would take
// it on demand, which only a click on it gets, and Sway takes it back from
// a surface that goes from one to the other: a panel keeps the keyboard
// while it shows, as launchers do, which blurs it only as it hides.
// Before gtk-layer-shell 0.6, a panel always asks for it.
func (w *window) layerKeyboard(take bool) {
	switch {
	case gtkLayerSetKeyboardMode == nil:
		gtkLayerSetKeyboardInteractivity(w.win, true)
	case take:
		gtkLayerSetKeyboardMode(w.win, keyboardExclusive)
	default:
		gtkLayerSetKeyboardMode(w.win, keyboardOnDemand)
	}
}

// placeLayer puts a layer surface's top-left corner at x, y, on the output
// there, with margins from that output's top and left edges.
func (w *window) placeLayer(x, y int) {
	m := gdkDisplayGetMonitorAtPoint(gdkDisplayGetDefault(), int32(x), int32(y))
	var g gdkRectangle
	gdkMonitorGetGeometry(m, &g)
	gtkLayerSetMonitor(w.win, m)
	gtkLayerSetAnchor(w.win, edgeTop, true)
	gtkLayerSetAnchor(w.win, edgeLeft, true)
	gtkLayerSetMargin(w.win, edgeTop, int32(y)-g.Y)
	gtkLayerSetMargin(w.win, edgeLeft, int32(x)-g.X)
	w.layerAt = &platform.Point{X: x, Y: y}
}

// centerLayer lets the compositor center a layer surface on its output.
func (w *window) centerLayer() {
	gtkLayerSetAnchor(w.win, edgeTop, false)
	gtkLayerSetAnchor(w.win, edgeLeft, false)
	w.layerAt = nil
}

// setLayerBounds places and sizes a layer surface.
func (w *window) setLayerBounds(r platform.Rect) {
	width, height := w.constrain(int32(r.Width), int32(r.Height))
	w.placeLayer(r.X, r.Y)
	w.sizeLayer(width, height)
}

// sizeLayer sizes a layer surface, which gtk-layer-shell sizes as the
// window asks to be, not by its default size.
func (w *window) sizeLayer(width, height int32) {
	gtkWidgetSetSizeRequest(w.win, width, height)
	gtkWindowResize(w.win, width, height)
}

// layerBounds returns where a layer surface is: where it was put, or the
// middle of its output, as the compositor centers it.
func (w *window) layerBounds() platform.Rect {
	var width, height int32
	gtkWindowGetSize(w.win, &width, &height)
	r := platform.Rect{Width: int(width), Height: int(height)}
	if p := w.layerAt; p != nil {
		r.X, r.Y = p.X, p.Y
		return r
	}
	d := gdkDisplayGetDefault()
	var m ptr
	if gw := gtkWidgetGetWindow(w.win); gw != 0 && gtkWidgetGetMapped(w.win) {
		m = gdkDisplayGetMonitorAtWin(d, gw)
	}
	if m == 0 {
		m = gdkDisplayGetPrimaryMonitor(d)
	}
	if m == 0 {
		return r
	}
	var g gdkRectangle
	gdkMonitorGetGeometry(m, &g)
	r.X, r.Y = int(g.X)+(int(g.Width)-r.Width)/2, int(g.Y)+(int(g.Height)-r.Height)/2
	return r
}
