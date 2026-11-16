//go:build darwin

package darwin

import (
	"slices"
	"sync"

	"github.com/ebitengine/purego"
	"github.com/ebitengine/purego/objc"

	"github.com/egoist/mygo/internal/platform"
)

// A web view in a window of native UI (platform.Surface.NewWebView) is a
// window of this package without an NSWindow of its own: win is its
// host's, which its sheets and panels use, and its MyGoWebView sits in a
// clip view, a subview of the host's content view under the surface. The
// surface's frames are transparent where the content shows the web view
// (holes), so what the content paints after it shows over the page, and
// its hitTest: lets the pointer through to the page there; the web view
// takes no mouse moves where the content painted over it. Its delegate is
// the web view's alone, never the NSWindow's, so it hears only of its
// page.

// nsViewMinYMargin is NSViewMinYMargin: the margin below a view grows,
// which keeps its distance to the top of an unflipped superview.
const nsViewMinYMargin = 8

func (s *surface) NewWebView(o *platform.WindowOptions, h platform.WindowHandler) (platform.WebView, error) {
	host := s.w
	b := host.b
	v := &window{b: b, h: h, opts: o, win: host.win, host: host, downloads: map[id][2]string{}}
	withPool(func() {
		v.delegate = alloc("MyGoWindowDelegate")
		v.createWebView(NSRect{})
		send(v.web, "setAutoresizingMask:", 0)
		v.clip = msgInitRect(send(class("MyGoClipView"), "alloc"), sel("initWithFrame:"), NSRect{})
		send(v.clip, "setWantsLayer:", 1)
		send(send(v.clip, "layer"), "setMasksToBounds:", 1)
		send(v.clip, "setAutoresizingMask:", nsViewMinYMargin)
		send(v.clip, "setHidden:", 1)
		if o.BackgroundColor != nil && !o.Transparent {
			v.clipBackground(*o.BackgroundColor)
		}
		send(v.clip, "addSubview:", uintptr(v.web))
		send(host.view, "addSubview:positioned:relativeTo:", uintptr(v.clip), nsWindowBelow, uintptr(s.view))
	})
	b.byDelegate[v.delegate] = v
	b.byWebView[v.web] = v
	host.webViews = append(host.webViews, v)
	return v, nil
}

// nsWindowBelow is NSWindowBelow, for addSubview:positioned:relativeTo:.
const nsWindowBelow = ^uintptr(0)

// clipBackground fills the clip view with c until the page paints.
func (w *window) clipBackground(c platform.Color) {
	send(send(w.clip, "layer"), "setBackgroundColor:", uintptr(send(nsColor(c), "CGColor")))
}

// PlaceWebViews shows the window's web views where the content shows
// them, in its paint order under the surface, and hides the others.
func (s *surface) PlaceWebViews(views []platform.WebViewPlacement) {
	host := s.w
	if host.closed {
		return
	}
	s.placed = append(s.placed[:0], views...)
	height := msgRect(host.view, sel("bounds")).Size.Height
	shown := make([]*window, 0, len(views))
	withPool(func() {
		for _, p := range views {
			v, ok := p.WebView.(*window)
			if !ok || v.host != host || v.closed {
				continue
			}
			c, f := p.Clip, p.Frame
			msgSetRect(v.clip, sel("setFrame:"), NSRect{Origin: NSPoint{c.X, height - c.Y - c.H}, Size: NSSize{c.W, c.H}})
			// The clip view is flipped.
			msgSetRect(v.web, sel("setFrame:"), NSRect{Origin: NSPoint{f.X - c.X, f.Y - c.Y}, Size: NSSize{f.W, f.H}})
			if sendBool(v.clip, "isHidden") {
				send(v.clip, "setHidden:", 0)
			}
			shown = append(shown, v)
		}
		for _, v := range host.webViews {
			if !slices.Contains(shown, v) && !sendBool(v.clip, "isHidden") {
				if v.hasKeyboard() {
					// A hidden view keeps no keyboard: it goes back to the
					// content.
					send(host.win, "makeFirstResponder:", uintptr(s.view))
				}
				send(v.clip, "setHidden:", 1)
			}
		}
		s.stackWebViews(shown)
	})
}

// stackWebViews puts the clip views of the web views shown in the order
// the content painted them, under the surface, unless they are.
func (s *surface) stackWebViews(shown []*window) {
	if len(shown) < 2 {
		return
	}
	subviews := send(s.w.view, "subviews")
	n := sendInt(subviews, "count")
	at := 0
	for i := range n {
		if at < len(shown) && send(subviews, "objectAtIndex:", uintptr(i)) == shown[at].clip {
			at++
		}
	}
	if at == len(shown) {
		return
	}
	// Moving a view out of its superview takes the keyboard from it: the
	// first responder goes back to it.
	first := send(s.w.win, "firstResponder")
	for _, v := range shown {
		retain(v.clip)
		send(v.clip, "removeFromSuperview")
		send(s.w.view, "addSubview:positioned:relativeTo:", uintptr(v.clip), nsWindowBelow, uintptr(s.view))
		release(v.clip)
	}
	if first != 0 && first != send(s.w.win, "firstResponder") {
		send(s.w.win, "makeFirstResponder:", uintptr(first))
	}
}

// webViewAt returns the web view that takes the pointer at x, y in the
// surface: the last placed, as the content painted it over the others,
// whose placement takes it.
func (s *surface) webViewAt(x, y float64) *window {
	for i := len(s.placed) - 1; i >= 0; i-- {
		if p := &s.placed[i]; p.At(x, y) {
			if v, ok := p.WebView.(*window); ok && !v.closed {
				return v
			}
		}
	}
	return nil
}

// covered reports whether the content painted over the web view w where
// the mouse event ev happened, or does not show it there.
func (w *window) covered(ev id) bool {
	s := w.host.surface
	if s == nil {
		return false
	}
	x, y := s.location(ev)
	return s.webViewAt(x, y) != w
}

// top returns the window that shows w: its host for a web view.
func (w *window) top() *window {
	if w.host != nil {
		return w.host
	}
	return w
}

// hasKeyboard reports whether the web view, or a view in it, is the first
// responder of its window.
func (w *window) hasKeyboard() bool {
	r := send(w.win, "firstResponder")
	return r != 0 && sendBool(r, "isKindOfClass:", uintptr(class("NSView"))) &&
		sendBool(r, "isDescendantOf:", uintptr(w.web))
}

// pressed tells the content of the window that a button went down on the
// web view w, which takes the press.
func (w *window) pressed(ev id, button int) {
	if s := w.host.surface; s != nil && !w.host.closed {
		x, y := s.location(ev)
		s.send(platform.SurfaceEvent{Kind: platform.WebViewPress, X: x, Y: y, Button: button, Mods: eventMods(ev)})
	}
}

// TabInto gives the web view the keyboard as the window's key view loop
// does, which tells WebKit to focus the page's first element, or its last
// going back: the web view follows the surface in the loop, or precedes
// it.
func (w *window) TabInto(back bool) {
	s := w.host.surface
	if w.closed || s == nil {
		return
	}
	withPool(func() {
		if back {
			send(w.web, "setNextKeyView:", uintptr(s.view))
			send(w.win, "selectKeyViewPrecedingView:", uintptr(s.view))
		} else {
			send(s.view, "setNextKeyView:", uintptr(w.web))
			send(w.win, "selectKeyViewFollowingView:", uintptr(s.view))
		}
	})
}

// tabOut gives the content the keyboard back as Tab leaves the page of the
// web view w past its last element, or Shift+Tab past its first.
func (w *window) tabOut(back bool) {
	s := w.host.surface
	if s == nil || w.host.closed {
		return
	}
	send(w.win, "makeFirstResponder:", uintptr(s.view))
	ev := platform.SurfaceEvent{Kind: platform.WebViewTabOut, Key: platform.KeyTab, WebView: w}
	if back {
		ev.Mods = platform.ModShift
	}
	s.send(ev)
}

// webViewOf returns the web view that is view or holds it.
func webViewOf(view id) *window {
	for v := view; v != 0; v = send(v, "superview") {
		if w := theBackend.byWebView[v]; w != nil {
			return w
		}
	}
	return nil
}

// closeWebView closes a web view: by Close, or as its window closes.
func (w *window) closeWebView() {
	if w.closed {
		return
	}
	w.closed = true
	b := w.b
	withPool(func() {
		if w.hasKeyboard() && w.host.surface != nil {
			send(w.win, "makeFirstResponder:", uintptr(w.host.surface.view))
		}
		send(w.web, "removeObserver:forKeyPath:", uintptr(w.delegate), uintptr(nsString("title")))
		send(w.ucc, "removeScriptMessageHandlerForName:", uintptr(nsString("mygo")))
		send(w.ucc, "removeAllUserScripts")
		send(w.web, "stopLoading")
		send(w.web, "setNavigationDelegate:", 0)
		send(w.web, "setUIDelegate:", 0)
		send(w.clip, "removeFromSuperview")
		delete(b.byDelegate, w.delegate)
		delete(b.byWebView, w.web)
		release(w.lastMouseDown)
		w.lastMouseDown = 0
		release(w.ucc)
		release(w.web)
		release(w.clip)
		// WebKit may still hold the delegate as the page goes.
		autorelease(w.delegate)
	})
	h := w.host
	h.webViews = slices.DeleteFunc(h.webViews, func(v *window) bool { return v == w })
	if s := h.surface; s != nil {
		s.placed = slices.DeleteFunc(s.placed, func(p platform.WebViewPlacement) bool { return p.WebView == platform.WebView(w) })
	}
}

// closeWebViews closes the web views of a window that closes.
func (w *window) closeWebViews() {
	for len(w.webViews) > 0 {
		w.webViews[len(w.webViews)-1].closeWebView()
	}
}

var (
	exitOnce     sync.Once
	msgExitEvent func(cls id, sel objc.SEL, typ uint, loc NSPoint, flags uint, ts float64, wn int, ctx id, en, tn int, data uintptr) id
)

// leave tells the page of the web view w that the pointer left it, as it
// went onto what the content painted over the page: an exit outside the
// view, which WebKit takes as a move there, unhovering what was under the
// pointer.
func (w *window) leave(ev id) {
	exitOnce.Do(func() { purego.RegisterFunc(&msgExitEvent, msgSendAddr) })
	const typeMouseExited = 9 // NSEventTypeMouseExited
	exit := msgExitEvent(class("NSEvent"), sel("enterExitEventWithType:location:modifierFlags:timestamp:windowNumber:context:eventNumber:trackingNumber:userData:"),
		typeMouseExited, NSPoint{X: -1e5, Y: -1e5}, 0, msgFloat(ev, sel("timestamp")), sendInt(w.win, "windowNumber"), 0, 0, 0, 0)
	sendSuper(w.web, "MyGoWebView", sel("mouseExited:"), uintptr(exit))
}

// webViewMethods are the overrides of MyGoWebView for web views in a
// window of native UI: a press tells the content, which closes its
// popovers, and the mouse moving where the content painted over the page
// is the content's alone, which sets the cursor there; the page hears
// that it left.
func webViewMethods() []objc.MethodDef {
	press := func(button int) func(id, objc.SEL, id) {
		return func(self id, cmd objc.SEL, ev id) {
			if w := theBackend.byWebView[self]; w != nil && w.host != nil {
				w.pressed(ev, button)
			}
			sendSuper(self, "MyGoWebView", cmd, uintptr(ev))
		}
	}
	moved := func(self id, cmd objc.SEL, ev id) {
		if w := theBackend.byWebView[self]; w != nil && w.host != nil {
			if w.covered(ev) {
				if w.pointerIn {
					w.pointerIn = false
					w.leave(ev)
				}
				return
			}
			w.pointerIn = true
		}
		sendSuper(self, "MyGoWebView", cmd, uintptr(ev))
	}
	exited := func(self id, cmd objc.SEL, ev id) {
		if w := theBackend.byWebView[self]; w != nil {
			w.pointerIn = false
		}
		sendSuper(self, "MyGoWebView", cmd, uintptr(ev))
	}
	return []objc.MethodDef{
		method("rightMouseDown:", press(1)),
		method("otherMouseDown:", press(2)),
		method("mouseMoved:", moved),
		method("mouseEntered:", moved),
		method("mouseExited:", exited),
		method("cursorUpdate:", moved),
		// The content resizes a web view as its window resizes, frame by
		// frame: told of the window's live resize, WebKit would show the
		// page's scrollers throughout it, at the page's last width.
		method("viewWillStartLiveResize", func(self id, cmd objc.SEL) {
			if w := theBackend.byWebView[self]; w == nil || w.host == nil {
				sendSuper(self, "MyGoWebView", cmd)
			}
		}),
		method("viewDidEndLiveResize", func(self id, cmd objc.SEL) {
			if w := theBackend.byWebView[self]; w == nil || w.host == nil {
				sendSuper(self, "MyGoWebView", cmd)
			}
		}),
		method("accessibilityParent", func(self id, cmd objc.SEL) id {
			if w := theBackend.byWebView[self]; w != nil && w.accessParent != 0 {
				return w.accessParent
			}
			return sendSuper(self, "MyGoWebView", cmd)
		}),
	}
}

func registerClipViewClass() {
	classDef("MyGoClipView", "NSView", nil, []objc.MethodDef{
		method("isFlipped", func(self id, _ objc.SEL) bool { return true }),
		// Assistive technology finds the page in the element of the
		// content showing it, not beside the content.
		method("accessibilityChildren", func(self id, _ objc.SEL) id { return nsArray() }),
	})
}
