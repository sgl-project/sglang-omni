//go:build linux && (amd64 || arm64)

package linux

import (
	"slices"
	"sync"

	"github.com/ebitengine/purego"

	"github.com/egoist/mygo/internal/platform"
)

// A flyout of a tray icon has no parent, which a popup needs on Wayland,
// and AppIndicator tells neither clicks nor where its icon is: it is an
// undecorated toplevel utility window over the others, in the middle of
// the primary display's work area where the platform lets apps place
// windows, which a focusable one closes as it loses the keyboard.
//
// A flyout (WindowOptions.Flyout) is a GTK_WINDOW_POPUP window, as GTK's
// menus and tooltips are: override-redirect on X11, where it goes exactly
// where Flyout.Place puts it in the work area of a monitor, and an
// xdg_popup of its parent on Wayland, which gdk_window_move_to_rect places
// with the compositor's positioner and which follows the parent by
// itself. Such a window gets the keyboard only through a grab: a
// focusable flyout grabs the seat (gdk_seat_grab), which Wayland turns
// into the popup's grab, and the application's input (gtk_grab_add), and
// closes as GtkMenu does, on a press outside it, when the grab breaks, or
// when the compositor dismisses the popup.

var (
	flyoutOnce sync.Once

	gtkWindowSetTypeHint          func(w ptr, hint int32)
	gdkWindowSetTransientFor      func(w, parent ptr)
	gdkWindowMoveToRect           func(w ptr, r *gdkRectangle, rectAnchor, windowAnchor, hints, dx, dy int32)
	gdkSeatGrab                   func(seat, w ptr, caps int32, ownerEvents bool, cursor, event, prepare, data ptr) int32
	gdkSeatUngrab                 func(seat ptr)
	gtkGrabAdd, gtkGrabRemove     func(w ptr)
	gtkWidgetTranslateCoordinates func(src, dst ptr, x, y int32, dx, dy *int32) bool
	gtkWindowSetAcceptFocus       func(w ptr, v bool)
	gtkWindowSetSkipPagerHint     func(w ptr, v bool)

	cbFlyoutPress, cbFlyoutGrabBroken, cbFlyoutUnmap, cbFlyoutPrepare ptr
)

const (
	typeHintUtility   = 5  // GDK_WINDOW_TYPE_HINT_UTILITY
	typeHintPopupMenu = 9  // GDK_WINDOW_TYPE_HINT_POPUP_MENU
	anchorHintsAll    = 63 // GDK_ANCHOR_FLIP | GDK_ANCHOR_SLIDE | GDK_ANCHOR_RESIZE
	seatCapabilityAll = 15 // GDK_SEAT_CAPABILITY_ALL
)

func loadFlyouts() {
	flyoutOnce.Do(func() {
		t, d := libGTK, libGDK
		mustBind(t, &gtkWindowSetTypeHint, "gtk_window_set_type_hint")
		mustBind(d, &gdkWindowSetTransientFor, "gdk_window_set_transient_for")
		mustBind(d, &gdkWindowMoveToRect, "gdk_window_move_to_rect")
		mustBind(d, &gdkSeatGrab, "gdk_seat_grab")
		mustBind(d, &gdkSeatUngrab, "gdk_seat_ungrab")
		mustBind(t, &gtkGrabAdd, "gtk_grab_add")
		mustBind(t, &gtkGrabRemove, "gtk_grab_remove")
		mustBind(t, &gtkWidgetTranslateCoordinates, "gtk_widget_translate_coordinates")
		mustBind(t, &gtkWindowSetAcceptFocus, "gtk_window_set_accept_focus")
		mustBind(t, &gtkWindowSetSkipPagerHint, "gtk_window_set_skip_pager_hint")
	})
}

// gravity returns the GdkGravity of a point of a rectangle.
func gravity(g platform.Gravity) int32 { return int32((g.Y+1)*3 + g.X + 1 + 1) }

// initFlyout makes a new window a flyout, before it shows.
func (w *window) initFlyout() {
	loadFlyouts()
	data := ptr(w.id)
	if w.opts.Flyout.Tray != nil {
		gtkWindowSetTypeHint(w.win, typeHintUtility)
		gtkWindowSetSkipPagerHint(w.win, true)
		w.SetAlwaysOnTop(true)
		gtkWindowSetAcceptFocus(w.win, w.opts.Flyout.Focusable)
		w.PlaceFlyout(*w.opts.Flyout, platform.Size{Width: w.opts.Width, Height: w.opts.Height})
		return
	}
	gtkWindowSetTypeHint(w.win, typeHintPopupMenu)
	// Its size is the one asked for, smaller than its content's natural
	// size too.
	gtkWindowSetResizable(w.win, true)
	connect(w.win, "button-press-event", cbFlyoutPress, data)
	connect(w.win, "grab-broken-event", cbFlyoutGrabBroken, data)
	connect(w.win, "unmap-event", cbFlyoutUnmap, data)
	w.PlaceFlyout(*w.opts.Flyout, platform.Size{Width: w.opts.Width, Height: w.opts.Height})
}

// PlaceFlyout places a flyout next to its anchor: on X11 where Flyout.Place
// puts it, on Wayland with the compositor's positioner, which takes a
// popup's position as it maps: there a flyout that shows maps again,
// unless only its parent moved, which the popup follows.
func (w *window) PlaceFlyout(f platform.Flyout, size platform.Size) {
	if w.opts.Flyout != nil && f.Tray != nil && !w.closed {
		w.flyout, w.flyoutSize, w.placed = f, size, true
		r := platform.Centered(size, platform.PrimaryWorkArea(screen{}.Displays()))
		w.SetBounds(r) // Wayland compositors place it themselves
		return
	}
	p, ok := w.opts.Parent.(*window)
	if w.opts.Flyout == nil || w.closed || !ok || p.closed {
		return
	}
	same := w.placed && f == w.flyout && size == w.flyoutSize
	w.flyout, w.flyoutSize, w.placed = f, size, true
	if w.b.onX11 {
		x, y := p.contentOrigin()
		anchor := f.Anchor
		anchor.X += int(x)
		anchor.Y += int(y)
		r, _, _ := f.Place(anchor, size, platform.WorkAreaFor(anchor, screen{}.Displays()))
		w.SetBounds(r)
		return
	}
	visible := gtkWidgetGetVisible(w.win)
	if visible && same {
		return
	}
	if visible {
		w.hideFlyout()
	}
	gtkWindowResize(w.win, int32(size.Width), int32(size.Height))
	gtkWidgetRealize(w.win)
	gdkWin := gtkWidgetGetWindow(w.win)
	gdkWindowSetTransientFor(gdkWin, gtkWidgetGetWindow(p.win))
	// Relative to the parent's GdkWindow, its shadows included.
	var x, y int32
	gtkWidgetTranslateCoordinates(p.contentWidget(), p.win, int32(f.Anchor.X), int32(f.Anchor.Y), &x, &y)
	var alloc gdkRectangle
	gtkWidgetGetAllocation(p.win, &alloc)
	r := gdkRectangle{X: x + alloc.X, Y: y + alloc.Y, Width: int32(max(f.Anchor.Width, 1)), Height: int32(max(f.Anchor.Height, 1))}
	ra, wa, dx, dy := f.Gravities()
	gdkWindowMoveToRect(gdkWin, &r, gravity(ra), gravity(wa), anchorHintsAll, int32(dx), int32(dy))
	if visible {
		w.showFlyout(f.Focusable)
	}
}

// contentOrigin returns where the window's content is on the screen.
func (w *window) contentOrigin() (x, y int32) {
	gtkWidgetRealize(w.win)
	gdkWindowGetOrigin(gtkWidgetGetWindow(w.win), &x, &y)
	var dx, dy int32
	gtkWidgetTranslateCoordinates(w.contentWidget(), w.win, 0, 0, &dx, &dy)
	return x + dx, y + dy
}

// showFlyout shows a flyout, which takes the keyboard when grab is set.
func (w *window) showFlyout(grab bool) {
	if w.trayFlyout() {
		w.willShow()
		gtkWidgetShow(w.win)
		if grab {
			gtkWindowPresent(w.win)
		}
		return
	}
	if gtkWidgetGetVisible(w.win) {
		return
	}
	if !w.b.onX11 {
		p, _ := w.opts.Parent.(*window)
		w.b.closePopupsAbove(p)
		w.b.flyouts = append(w.b.flyouts, w)
	}
	w.willShow()
	if !grab {
		gtkWidgetShow(w.win)
		return
	}
	gtkWidgetRealize(w.win)
	seat := gdkDisplayGetDefaultSeat(gdkDisplayGetDefault())
	// The prepare function shows the window: Wayland maps the popup with
	// the grab.
	if gdkSeatGrab(seat, gtkWidgetGetWindow(w.win), seatCapabilityAll, true, 0, 0, cbFlyoutPrepare, ptr(w.id)) == 0 {
		w.grabbed = true
		gtkGrabAdd(w.win)
	}
	gtkWidgetShow(w.win) // unless the prepare function did
}

// hideFlyout hides a flyout, and lets the grab of a focusable one go.
func (w *window) hideFlyout() {
	w.ungrab()
	w.b.flyouts = slices.DeleteFunc(w.b.flyouts, func(x *window) bool { return x == w })
	if gtkWidgetGetMapped(w.win) {
		w.unmaps++ // its unmap event comes later, maybe once it shows again
	}
	gtkWidgetHide(w.win)
}

// closePopupsAbove closes the popups above parent, the topmost first, as
// the user dismissing them, or hides those that stay: Wayland maps a popup
// only over the topmost one, or a toplevel, which GDK makes its parent.
func (b *Backend) closePopupsAbove(parent *window) {
	for len(b.flyouts) > 0 {
		top := b.flyouts[len(b.flyouts)-1]
		if top == parent {
			return
		}
		top.dismissFlyout(false)
		if !top.closed && gtkWidgetGetVisible(top.win) {
			top.hideFlyout()
		}
		b.flyouts = slices.DeleteFunc(b.flyouts, func(x *window) bool { return x == top })
	}
}

// dismissFlyouts closes the focusable flyouts that grab the input as
// another window takes the keyboard, which their grab keeps from it.
func (b *Backend) dismissFlyouts(focus *window) {
	for _, w := range b.windows {
		if w.grabbed && w != focus {
			w.dismissFlyout(false)
		}
	}
}

func (w *window) ungrab() {
	if !w.grabbed {
		return
	}
	w.grabbed = false
	gtkGrabRemove(w.win)
	gdkSeatUngrab(gdkDisplayGetDefaultSeat(gdkDisplayGetDefault()))
}

// dismissFlyout closes a flyout the user dismissed, as the user closing
// it. One that stays, but lost its popup (the compositor hid it, or its
// grab broke), hides.
func (w *window) dismissFlyout(lost bool) {
	if w.closed || w.dismissing {
		return
	}
	w.dismissing = true
	defer func() { w.dismissing = false }()
	if w.h.ShouldClose() {
		w.Close()
	} else if lost {
		w.hideFlyout()
	}
}

func initFlyoutCallbacks() {
	b := func() *Backend { return theBackend }
	cbFlyoutPrepare = purego.NewCallback(func(seat, gdkWin, data ptr) {
		if w := b().window(data); w != nil {
			gtkWidgetShow(w.win)
		}
	})
	// The application's input goes to a focusable flyout while it grabs
	// it: a press elsewhere, in another window or, on X11, another app,
	// closes it.
	cbFlyoutPress = purego.NewCallback(func(widget, event, data ptr) bool {
		w := b().window(data)
		if w == nil || !w.grabbed {
			return false
		}
		// Where the press is on the screen, as GtkMenu looks: the event's
		// window and coordinates may be the grab's or the window's under
		// the pointer. GdkEventButton: x_root 64, y_root 72.
		var ox, oy int32
		gdkWindowGetOrigin(gtkWidgetGetWindow(w.win), &ox, &oy)
		var a gdkRectangle
		gtkWidgetGetAllocation(w.win, &a)
		x, y := field[float64](event, 64)-float64(ox), field[float64](event, 72)-float64(oy)
		if x >= 0 && y >= 0 && x < float64(a.Width) && y < float64(a.Height) {
			return false
		}
		w.dismissFlyout(false)
		return true
	})
	cbFlyoutGrabBroken = purego.NewCallback(func(widget, event, data ptr) bool {
		if w := b().window(data); w != nil && w.grabbed {
			w.grabbed = false
			gtkGrabRemove(w.win)
			w.dismissFlyout(true)
		}
		return false
	})
	// A Wayland compositor dismissed the popup, and GDK hid its window:
	// an unmap the backend's own hiding (unmaps) does not account for.
	// X11 servers do not dismiss popups.
	cbFlyoutUnmap = purego.NewCallback(func(widget, event, data ptr) bool {
		w := b().window(data)
		if w == nil || w.b.onX11 {
			return false
		}
		if w.unmaps > 0 {
			w.unmaps--
			return false
		}
		if gtkWidgetGetVisible(w.win) {
			w.ungrab()
			w.dismissFlyout(true)
		}
		return false
	})
}

// trayFlyout reports a flyout of a tray icon.
func (w *window) trayFlyout() bool { return w.opts.Flyout != nil && w.opts.Flyout.Tray != nil }

// grabbingChild reports whether a flyout of w's grabs the input, and so
// has the keyboard w lost to it.
func (w *window) grabbingChild() bool {
	for _, x := range w.b.windows {
		if !x.grabbed {
			continue
		}
		for p, _ := x.opts.Parent.(*window); p != nil; p, _ = p.opts.Parent.(*window) {
			if p == w {
				return true
			}
		}
	}
	return false
}
