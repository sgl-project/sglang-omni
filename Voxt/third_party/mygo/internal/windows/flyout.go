//go:build windows

package windows

import (
	"unsafe"

	"github.com/egoist/mygo/internal/platform"
)

// A flyout (WindowOptions.Flyout) is a WS_POPUP window owned by its
// parent, which keeps it above, out of the taskbar (WS_EX_TOOLWINDOW), and
// transparent where its content is: without a redirection bitmap, the
// content shows through DirectComposition with its alpha. One that is not
// focusable is never activated (WS_EX_NOACTIVATE, MA_NOACTIVATE), and one
// that is closes when another window is activated. Owned windows do not
// follow their owner: the core places flyouts again as their parent moves.

const (
	wmMouseActivate  = 0x0021
	maActivateAndEat = 2
	maNoActivate     = 3
	// wmAppDismiss asks a flyout whether it closes, after the activation
	// that dismissed it.
	wmAppDismiss = wmApp + 6

	// flyoutShadowClass is the window class of flyouts with a shadow: the
	// drop shadow of menus and tooltips (CS_DROPSHADOW).
	flyoutShadowClass = "MyGoFlyoutShadow"
)

var procMonitorFromRect = user32.NewProc("MonitorFromRect")

// roundFlyout rounds the corners of a flyout with a shadow, as Windows 11
// rounds its menus and WinUI's flyouts: DWM clips the window, outlines it
// and casts the shadow around all of it. Windows 10 rounds nothing, and
// casts the shadow of its menus, square, along the right and bottom edges.
func (w *window) roundFlyout() {
	pref := int32(dwmwcpRound)
	procDwmSetWindowAttribute.Call(w.hwnd, dwmwaWindowCornerPreference, uintptr(unsafe.Pointer(&pref)), 4)
}

// flyoutStyles returns the styles of a flyout's window.
func (w *window) flyoutStyles() (style, ex uint32) {
	style = wsPopup | wsClipChildren
	ex = wsExToolWindow
	if w.opts.Flyout.Tray != nil {
		ex |= wsExTopmost // over other apps' windows, without an owner
	}
	if !w.opts.Flyout.Focusable {
		ex |= wsExNoActivate
	}
	if w.noRedirect {
		ex |= wsExNoRedirect
	}
	return style, ex
}

// PlaceFlyout places a flyout next to its anchor, in the parent's client
// area or a tray's icon, as Flyout.Place resolves it in the work area of
// the monitor that has most of the anchor, in physical pixels at the
// DPI of the parent, or of the icon's monitor.
func (w *window) PlaceFlyout(f platform.Flyout, size platform.Size) {
	if w.opts.Flyout == nil || w.closed {
		return
	}
	w.flyout, w.flyoutSize = f, size
	var a rect // the anchor
	var dpi int
	if t, ok := f.Tray.(*tray); ok {
		var shows bool
		if a, shows = t.rect(); !shows {
			// In the middle of the primary monitor's work area.
			mon, _, _ := procMonitorFromPoint.Call(0, monitorDefaultToPrimary)
			dpi = monitorDPI(mon)
			r := platform.Centered(platform.Size{Width: int(toPx(size.Width, dpi)), Height: int(toPx(size.Height, dpi))}, rectOf(monitorWorkArea(mon)))
			procSetWindowPos.Call(w.hwnd, 0, uintptr(r.X), uintptr(r.Y), uintptr(r.Width), uintptr(r.Height), swpNoZOrder|swpNoActivate)
			return
		}
		mon, _, _ := procMonitorFromRect.Call(uintptr(unsafe.Pointer(&a)), monitorDefaultToNearest)
		dpi = monitorDPI(mon)
	} else {
		p := w.parent
		if p == nil || p.closed {
			return
		}
		dpi = dpiOf(p.hwnd)
		origin := point{}
		procClientToScreen.Call(p.hwnd, uintptr(unsafe.Pointer(&origin)))
		a.Left, a.Top = origin.X+toPx(f.Anchor.X, dpi), origin.Y+toPx(f.Anchor.Y, dpi)
		a.Right, a.Bottom = a.Left+toPx(f.Anchor.Width, dpi), a.Top+toPx(f.Anchor.Height, dpi)
	}
	px := f
	px.Gap = int(toPx(f.Gap, dpi))
	px.Anchor = rectOf(a)
	search := rect{a.Left, a.Top, max(a.Right, a.Left+1), max(a.Bottom, a.Top+1)}
	mon, _, _ := procMonitorFromRect.Call(uintptr(unsafe.Pointer(&search)), monitorDefaultToNearest)
	r, _, _ := px.Place(px.Anchor, platform.Size{Width: int(toPx(size.Width, dpi)), Height: int(toPx(size.Height, dpi))}, rectOf(monitorWorkArea(mon)))
	procSetWindowPos.Call(w.hwnd, 0, uintptr(r.X), uintptr(r.Y), uintptr(r.Width), uintptr(r.Height), swpNoZOrder|swpNoActivate)
}

// rectOf returns r as a platform rectangle.
func rectOf(r rect) platform.Rect {
	return platform.Rect{X: int(r.Left), Y: int(r.Top), Width: int(r.Right - r.Left), Height: int(r.Bottom - r.Top)}
}

// flyoutMessage handles the messages of a flyout's window that differ
// from other windows'.
func (w *window) flyoutMessage(m uint32, wp, lp uintptr) (uintptr, bool) {
	switch m {
	case wmMouseActivate:
		if !w.flyout.Focusable {
			return maNoActivate, true
		}
	case wmActivate:
		if loword(wp) == waInactive && w.flyout.Focusable {
			// Unless the window activated is a flyout of its own.
			for x := w.b.windows[lp]; x != nil; x = x.parent {
				if x == w {
					return 0, false
				}
			}
			// A press on its own tray icon, which activates the taskbar:
			// the icon's click toggles it.
			if t, ok := w.flyout.Tray.(*tray); ok {
				var pt point
				procGetCursorPos.Call(uintptr(unsafe.Pointer(&pt)))
				if r, shows := t.rect(); shows && pt.X >= r.Left && pt.X < r.Right && pt.Y >= r.Top && pt.Y < r.Bottom {
					return 0, false
				}
			}
			postMessage(w.hwnd, wmAppDismiss, 0, 0)
		}
	case wmAppDismiss:
		w.dismissFlyout()
		return 0, true
	case wmNCCalcSize:
		if wp != 0 {
			return 0, true // all of the window is client area
		}
	}
	return 0, false
}

// ownsFocusableFlyout reports whether a focusable flyout of w, or of its
// flyouts, shows: a click activating w only closes it, as it closes menus
// and WinUI's flyouts, which would otherwise come back as the click on
// what opened them toggles them.
func (w *window) ownsFocusableFlyout() bool {
	for _, x := range w.b.windows {
		if f := x.opts.Flyout; f == nil || !f.Focusable || x.closed || !x.IsVisible() {
			continue
		}
		for p := x.parent; p != nil; p = p.parent {
			if p == w {
				return true
			}
		}
	}
	return false
}

// dismissFlyout closes a focusable flyout another window was activated
// over, as the user closing it.
func (w *window) dismissFlyout() {
	if !w.closed && w.IsVisible() && w.h.ShouldClose() {
		w.Close()
	}
}
