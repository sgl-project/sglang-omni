//go:build darwin

package darwin

import (
	"github.com/ebitengine/purego/objc"

	"github.com/egoist/mygo/internal/platform"
)

// A flyout with Flyout.Popover shows its web view or surface in an
// NSPopover, which draws the system's material and arrow, and animates.
// AppKit places it against the anchor: the arrow points at the anchor's
// middle, on the side the flyout asks for, which AppKit flips where the
// screen has no room. The popover makes its window as it shows: w.win is
// that window while it shows. AppKit gives a popover that shows the
// keyboard, focusable or not, while its parent stays the key window, which
// the popover's window says it is too. A focusable flyout's popover is
// transient: it asks the delegate (popoverShouldClose:) before it closes on
// a click elsewhere, as WindowHandler.ShouldClose decides.

const (
	popoverBehaviorApplicationDefined = 0
	popoverBehaviorTransient          = 1
)

func (w *window) createPopover(content NSRect) {
	w.delegate = alloc("MyGoWindowDelegate")
	w.createContent(content)
	send(w.view, "setAutoresizingMask:", nsViewWidthHeightSizable)
	w.popoverController = alloc("NSViewController")
	send(w.popoverController, "setView:", uintptr(w.view))
	w.popover = alloc("NSPopover")
	send(w.popover, "setContentViewController:", uintptr(w.popoverController))
	msgSetSize(w.popover, sel("setContentSize:"), content.Size)
	behavior := uintptr(popoverBehaviorApplicationDefined)
	if w.flyout.Focusable {
		behavior = popoverBehaviorTransient
	}
	send(w.popover, "setBehavior:", behavior)
	send(w.popover, "setDelegate:", uintptr(w.delegate))
	w.register()
}

// popoverAnchor returns the view the popover shows from, the anchor in it
// and the edge of it the popover goes to: the parent's content view, which
// is not flipped, or the button of a tray's status item. view is 0 when
// there is none.
func (w *window) popoverAnchor() (view id, r NSRect, edge uint) {
	// NSRectEdge
	edge = map[platform.Side]uint{platform.SideBottom: 1, platform.SideTop: 3, platform.SideRight: 2, platform.SideLeft: 0}[w.flyout.Side]
	if t, ok := w.flyout.Tray.(*tray); ok {
		if t.item == 0 {
			return 0, r, edge
		}
		return t.button, msgRect(t.button, sel("bounds")), edge
	}
	a, p := w.flyout.Anchor, w.parent
	if p == nil || p.closed || p.view == 0 {
		return 0, r, edge
	}
	h := msgRect(p.view, sel("bounds")).Size.Height
	r = NSRect{
		Origin: NSPoint{float64(a.X), h - float64(a.Y) - float64(max(a.Height, 1))},
		Size:   NSSize{float64(max(a.Width, 1)), float64(max(a.Height, 1))},
	}
	return p.view, r, edge
}

func (w *window) showPopover() {
	view, r, edge := w.popoverAnchor()
	if view == 0 || sendBool(w.popover, "isShown") {
		return
	}
	w.popoverClosing = false
	if w.flyout.Tray != nil {
		// The popover of a menu bar app takes the keyboard of an app the
		// user did not activate, as AppKit's own status items' do.
		send(w.b.app, "activateIgnoringOtherApps:", 1)
	}
	msgSetSize(w.popover, sel("setContentSize:"), NSSize{float64(w.flyoutSize.Width), float64(w.flyoutSize.Height)})
	msgRectIDUint(w.popover, sel("showRelativeToRect:ofView:preferredEdge:"), r, view, edge)
	w.adoptPopoverWindow()
}

// adoptPopoverWindow makes the window the popover shows in the flyout's.
func (w *window) adoptPopoverWindow() {
	win := send(w.view, "window")
	if win == 0 || win == w.win {
		return
	}
	w.forgetPopoverWindow()
	w.win = win
	w.b.byNSWindow[win] = w
	if w.surface != nil {
		send(win, "makeFirstResponder:", uintptr(w.surface.view))
	}
}

func (w *window) forgetPopoverWindow() {
	if w.win == 0 {
		return
	}
	delete(w.b.byNSWindow, w.win)
	w.win = 0
}

// placePopover moves a popover that shows to the anchor, at the size.
func (w *window) placePopover() {
	if !sendBool(w.popover, "isShown") {
		return
	}
	msgSetSize(w.popover, sel("setContentSize:"), NSSize{float64(w.flyoutSize.Width), float64(w.flyoutSize.Height)})
	if _, r, _ := w.popoverAnchor(); w.flyout.Tray == nil {
		msgSetRect(w.popover, sel("setPositioningRect:"), r)
	}
}

// hidePopover closes the popover, which shows again with Show, and gives
// the keyboard it had back to the parent.
func (w *window) hidePopover() {
	if !sendBool(w.popover, "isShown") {
		return
	}
	key := w.win != 0 && sendBool(w.win, "isKeyWindow")
	w.popoverClosing = true
	send(w.popover, "close")
	if p := w.parent; key && p != nil && !p.closed && sendBool(p.win, "isVisible") {
		send(p.win, "makeKeyWindow")
	}
}

// closePopover destroys the flyout, at once.
func (w *window) closePopover() {
	send(w.popover, "setAnimates:", 0)
	w.hidePopover()
	w.cleanup()
	w.h.Closed()
}

func (w *window) cleanupPopover() {
	w.forgetPopoverWindow()
	send(w.popover, "setDelegate:", 0)
	if sendBool(w.popover, "isShown") {
		send(w.popover, "setAnimates:", 0)
		send(w.popover, "close")
	}
	send(w.popover, "setContentViewController:", 0)
	release(w.popover)
	release(w.popoverController)
}

func popoverDelegateMethods() []objc.MethodDef {
	b := func() *Backend { return theBackend }
	return []objc.MethodDef{
		// The user clicked elsewhere, in a transient popover. A press on
		// its own tray icon leaves it to the icon's click, which toggles
		// it, rather than closing it for the click to open it again.
		method("popoverShouldClose:", func(self id, _ objc.SEL, popover id) bool {
			w := b().windowFor(self)
			if w == nil {
				return true
			}
			if t, ok := w.flyout.Tray.(*tray); ok && t.pressed() {
				return false
			}
			return w.h.ShouldClose()
		}),
		method("popoverDidClose:", func(self id, _ objc.SEL, n id) {
			w := b().windowFor(self)
			if w == nil {
				return
			}
			w.forgetPopoverWindow()
			if w.popoverClosing {
				w.popoverClosing = false
				return
			}
			w.cleanup()
			w.h.Closed()
		}),
	}
}

// pressed reports whether the event being handled is a press on the
// icon's button.
func (t *tray) pressed() bool {
	ev := send(t.b.app, "currentEvent")
	if ev == 0 || t.item == 0 {
		return false
	}
	switch sendInt(ev, "type") {
	case 1, 3: // NSEventTypeLeftMouseDown, RightMouseDown
	default:
		return false
	}
	win := send(t.button, "window")
	if win == 0 || send(ev, "window") != win {
		return false
	}
	p := msgPointFromView(t.button, sel("convertPoint:fromView:"), msgPoint(ev, sel("locationInWindow")), 0)
	b := msgRect(t.button, sel("bounds"))
	return p.X >= b.Origin.X && p.Y >= b.Origin.Y && p.X < b.Origin.X+b.Size.Width && p.Y < b.Origin.Y+b.Size.Height
}
