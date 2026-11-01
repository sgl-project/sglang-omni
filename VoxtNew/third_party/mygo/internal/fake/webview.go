package fake

import "github.com/egoist/mygo/internal/platform"

// WebView is a web view under a fake Surface (Surface.NewWebView): a
// Window of its own for its page, which its handler hears of, without the
// window's methods.
type WebView struct {
	*Window
	host *Window

	// Placement is where the content last showed the web view, while
	// Shown; Focused tells that Focus or TabInto gave it the keyboard
	// since, TabbedBack that TabInto did so going back.
	Placement  platform.WebViewPlacement
	Shown      bool
	Focused    bool
	TabbedBack bool
}

func (s *Surface) NewWebView(o *platform.WindowOptions, h platform.WindowHandler) (platform.WebView, error) {
	v := &WebView{Window: &Window{b: s.w.b, H: h, Opts: o, zoom: o.Zoom}, host: s.w}
	if o.BackgroundColor != nil {
		v.Background = *o.BackgroundColor
	}
	s.mu.Lock()
	s.webViews = append(s.webViews, v)
	s.mu.Unlock()
	return v, nil
}

func (s *Surface) PlaceWebViews(views []platform.WebViewPlacement) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.placements++
	for _, v := range s.webViews {
		v.Shown = false
	}
	s.order = s.order[:0]
	for _, p := range views {
		v, ok := p.WebView.(*WebView)
		if !ok || v.host != s.w || v.IsClosed() {
			continue
		}
		v.Placement, v.Shown = p, true
		v.Placement.Covers = append([]platform.RectF(nil), p.Covers...)
		s.order = append(s.order, v)
	}
	for _, v := range s.webViews {
		if !v.Shown {
			v.Focused = false // a hidden view keeps no keyboard
		}
	}
}

// WebViews returns the surface's web views not closed, in the order they
// were made.
func (s *Surface) WebViews() []*WebView {
	s.mu.Lock()
	defer s.mu.Unlock()
	var list []*WebView
	for _, v := range s.webViews {
		if !v.IsClosed() {
			list = append(list, v)
		}
	}
	return list
}

// Placements counts the PlaceWebViews calls.
func (s *Surface) Placements() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.placements
}

// WebViewAt returns the web view that takes the pointer at x, y: the last
// shown, in the order the content painted them, whose placement takes it.
func (s *Surface) WebViewAt(x, y float64) *WebView {
	s.mu.Lock()
	defer s.mu.Unlock()
	for i := len(s.order) - 1; i >= 0; i-- {
		if v := s.order[i]; v.Shown && v.Placement.At(x, y) {
			return v
		}
	}
	return nil
}

// Press presses the pointer at x, y as the platform would: on the web
// view taking it there, which tells the content (WebViewPress) and
// returns it, else on the content, and reports nil.
func (s *Surface) Press(x, y float64) *WebView {
	if v := s.WebViewAt(x, y); v != nil {
		s.Send(platform.SurfaceEvent{Kind: platform.WebViewPress, X: x, Y: y})
		v.Focus()
		return v
	}
	s.Send(platform.SurfaceEvent{Kind: platform.PointerDown, X: x, Y: y, Clicks: 1})
	s.Send(platform.SurfaceEvent{Kind: platform.PointerUp, X: x, Y: y, Clicks: 1})
	return nil
}

func (v *WebView) Focus() {
	v.mu.Lock()
	v.Focused = true
	v.mu.Unlock()
}

func (v *WebView) TabInto(back bool) {
	v.mu.Lock()
	v.Focused, v.TabbedBack = true, back
	v.mu.Unlock()
}

// TabOut moves the keyboard out of the web view as Tab past its page's
// last element does, or Shift+Tab past its first when back.
func (v *WebView) TabOut(back bool) {
	v.mu.Lock()
	v.Focused = false
	v.mu.Unlock()
	ev := platform.SurfaceEvent{Kind: platform.WebViewTabOut, Key: platform.KeyTab, WebView: v}
	if back {
		ev.Mods = platform.ModShift
	}
	v.host.surface.Send(ev)
}

// Close closes the web view; its handler hears no more of it.
func (v *WebView) Close() {
	v.mu.Lock()
	v.closed, v.Shown, v.Focused = true, false, false
	v.mu.Unlock()
}

// Host returns the window the web view is in.
func (v *WebView) Host() *Window { return v.host }

// closeWebViews closes the web views of the window's surface, as the
// window closes.
func (w *Window) closeWebViews() {
	if w.surface == nil {
		return
	}
	for _, v := range w.surface.WebViews() {
		v.Close()
	}
}
