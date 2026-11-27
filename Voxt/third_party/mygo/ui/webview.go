package ui

import (
	"math"
	"slices"

	"github.com/egoist/mygo/internal/platform"
	"github.com/egoist/mygo/internal/scene"
	"github.com/egoist/mygo/internal/surface"
)

// NativeWebView is a web view a window made for its content to show,
// as *mygo.WebView, of mygo.Window.NewWebView.
type NativeWebView = surface.WebView

// WebView shows a web view, made with mygo.Window.NewWebView, in the
// element's content box. A web view has no size of its own: the element
// stretches across its container (AlignSelf(Stretch)), and takes the room
// along it with Grow or a size:
//
//	ui.Row(c).Fill().Children(func() {
//		sidebar(c)
//		ui.WebView(c, app.docs).Grow(1)
//	})
//
// The web view shows while a frame builds its element and hides when one
// does not, keeping its page: a view switching tabs builds the web view
// of the tab shown. The element paints a hole through the window's
// content (the web view is under it, the system's own view), as rounded
// as its Radius and cut by the scroll containers and clips around it, so
// that what paints after it shows over the page: its children, popovers,
// menus, dialogs, tooltips and toasts. The page takes the pointer where
// it shows and nothing painted over it takes it, and the keyboard once
// clicked, focused (mygo.WebView.Focus) or tabbed into: Tab stops at the
// element. Its background and border paint around the page, which its
// opacity fades, and assistive technology finds the page inside it.
func coreWebView(c *context, v NativeWebView) *node {
	e := c.newElement(kindBox)
	e.webView = v
	e.self = Stretch
	// Tab stops at it to give its page the keyboard, which shows its
	// own focus.
	e.flags |= flagFocusable | flagOwnRing
	return e
}

// webPlace is where a frame shows a web view: hit is the index of its
// element's own hit, inert that it shows but takes no pointer.
type webPlace struct {
	v     NativeWebView
	frame Rect
	clip  Rect
	hit   int
	inert bool
}

// noteWebView notes where the frame shows the web view of e, laid out
// within clip, unless none of it shows. Called while committing, before
// the element's own hit.
func (rt *engine) noteWebView(e *node, clip Rect, invisible bool) {
	if e.flags&flagInvisible != 0 {
		return
	}
	// The hole the painter makes: the content box on whole pixels. Native
	// web views stay upright, covering a transformed box's bounds.
	b := e.visualRect(e.contentBox())
	if s := rt.painted[2]; s > 0 {
		x0, y0 := float32(math.Round(float64(b.X*s)))/s, float32(math.Round(float64(b.Y*s)))/s
		x1, y1 := float32(math.Round(float64((b.X+b.W)*s)))/s, float32(math.Round(float64((b.Y+b.H)*s)))/s
		b = Rect{x0, y0, x1 - x0, y1 - y0}
	}
	v := intersect(b, clip)
	if b.W <= 0 || b.H <= 0 || v.W <= 0 || v.H <= 0 {
		return
	}
	rt.webPlaces = append(rt.webPlaces, webPlace{v: e.webView, frame: b, clip: v, hit: len(rt.hits), inert: invisible})
}

// placedWebView is where a frame shows a web view, for the host.
type placedWebView struct {
	v           NativeWebView
	frame, clip platform.RectF
	covers      []platform.RectF
}

// webViewPlacements returns where the frame shows its web views, in the
// order it paints them, each with the boxes of what is painted over it
// and takes the pointer: the hits after its own. A web view built twice,
// as by an exit transition, shows where it takes the pointer.
func (rt *engine) webViewPlacements() []placedWebView {
	out := rt.webOut[:0]
	for i := range rt.webPlaces {
		wp := &rt.webPlaces[i]
		dup := slices.IndexFunc(rt.webPlaces[:i], func(o webPlace) bool { return o.v == wp.v })
		if dup >= 0 && (wp.inert || !rt.webPlaces[dup].inert) {
			continue
		}
		if dup >= 0 {
			// The inert one went first: this one takes its place.
			out = slices.DeleteFunc(out, func(o placedWebView) bool { return o.v == wp.v })
		}
		out = append(out, placedWebView{v: wp.v, frame: rectF(wp.frame), clip: rectF(wp.clip), covers: wp.covers(rt.hits)})
	}
	rt.webOut = out
	return out
}

// covers returns the boxes, within the web view's clip, of the hits after
// its own: those inside a box already found add nothing.
func (wp *webPlace) covers(hits []hit) []platform.RectF {
	if wp.inert {
		return []platform.RectF{rectF(wp.clip)}
	}
	var out []platform.RectF
	start := wp.hit
	if start < len(hits) && hits[start].st != nil && hits[start].st.webView == wp.v {
		start++ // its own
	}
next:
	for _, h := range hits[start:] {
		r := intersect(h.r, wp.clip)
		if r.W <= 0 || r.H <= 0 {
			continue
		}
		f := rectF(r)
		for _, c := range out {
			if f.X >= c.X && f.Y >= c.Y && f.X+f.W <= c.X+c.W && f.Y+f.H <= c.Y+c.H {
				continue next
			}
		}
		out = append(out, f)
	}
	return out
}

func rectF(r Rect) platform.RectF {
	return platform.RectF{X: float64(r.X), Y: float64(r.Y), W: float64(r.W), H: float64(r.H)}
}

// placeWebViews tells the host where the frame shows its web views.
func (rt *engine) placeWebViews() {
	if len(rt.webPlaces) == 0 && !rt.webShown {
		return
	}
	rt.webShown = len(rt.webPlaces) > 0
	rt.host.placeWebViews(rt.webViewPlacements())
}

// webViewPress handles a press that went to the web view at x, y: the
// elements under it hover, and the focus leaves the content, as the
// page has the keyboard, but nothing is pressed; popovers see a press
// outside them (PressedOutside).
func (rt *engine) webViewPress(x, y float32, button int) {
	chain := rt.hitChain(x, y)
	rt.pointerX, rt.pointerY, rt.pointerIn = x, y, true
	rt.setHover(chain)
	if len(chain) > 0 {
		rt.downs = append(rt.downs, chain[0])
	} else {
		rt.downs = append(rt.downs, 0)
	}
	if button == 0 && rt.focused != 0 {
		rt.focused = 0
		rt.focusVisible = false
	}
	rt.requestFrame()
}

// webViewTabOut moves the focus on from the element showing v, whose page
// Tab left past its last element, or Shift+Tab past its first when back.
func (rt *engine) webViewTabOut(v NativeWebView, back bool) {
	for _, id := range rt.focusOrder {
		if s := rt.states[id]; s != nil && s.webView == v {
			rt.focused = id
			break
		}
	}
	rt.moveFocus(back)
	rt.requestFrame()
}

// overWebView reports whether the element under the pointer is a web
// view, whose page has the cursor.
func (rt *engine) overWebView() bool {
	if len(rt.hover) == 0 {
		return false
	}
	s := rt.states[rt.hover[0]]
	return s != nil && s.webView != nil
}

// hole paints the hole showing the web view of e: its content box,
// rounded as its border's inner edge less its padding.
func (p *Painter) hole(e *node) {
	b := e.contentBox()
	if b.W <= 0 || b.H <= 0 {
		return
	}
	in := [4]float32{e.border[0] + e.pad[0], e.border[1] + e.pad[1], e.border[2] + e.pad[2], e.border[3] + e.pad[3]}
	box := p.snap(Rect{e.x, e.y, e.w, e.h})
	_, radii := scene.InnerRadii(box, scene.Corners(box, p.radii(e.radius), continuousCorners), p.radii(in))
	for i := range radii {
		radii[i] = abs32(radii[i])
	}
	r := p.snap(b)
	// Translucent, the page blends over what was painted under it.
	p.add(scene.Op{Kind: scene.OpHole, Rect: r, Radii: radii, Continuous: continuousCorners, Opacity: p.opacity})
	if c := intersect(transformRect(p.transform, b), p.clip); c.W > 0 && c.H > 0 {
		p.holes = append(p.holes, c)
	}
}
