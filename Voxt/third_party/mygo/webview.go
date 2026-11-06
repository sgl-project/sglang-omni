package mygo

import (
	"errors"
	"fmt"
	"log"

	"github.com/egoist/mygo/internal/platform"
	"github.com/egoist/mygo/internal/surface"
)

// WebView is a web page inside a window's native UI, which the window's
// view places with ui.WebView: a document the web engine renders beside
// native controls, a page per tab, a preview of HTML the app makes.
//
//	docs, err := win.NewWebView(mygo.WebViewOptions{URL: "/docs"})
//
//	// In the view:
//	ui.Row(c).Fill().Children(func() {
//		sidebar(c)
//		ui.WebView(c, docs).Grow(1)
//	})
//
// Its page works as a window's does: it loads the app's frontend, calls
// bound Go methods (CallerPage tells which page called, CallerWindow the
// window it is in), hears events (Event.EmitPage, Broadcast) and
// evaluates scripts. The web view shows where the view places it, under
// what the view paints after it, as popovers and menus, while the view
// builds it; it keeps its page while it does not show, until Destroy or
// until its window closes. Its methods are safe from any goroutine.
type WebView struct{ w *Window }

// WebViewOptions configure Window.NewWebView.
type WebViewOptions struct {
	// URL is loaded right away. "/" and other URLs without a scheme are
	// pages of the app's frontend (see Page.LoadURL).
	URL string
	// Page configures the web view's page.
	Page PageOptions
	// BackgroundColor fills the web view until its page paints, as
	// WindowOptions.BackgroundColor does; by default the page's, or white.
	BackgroundColor string
	// Transparent shows what the window's view paints under the web view
	// through the transparent parts of its page.
	Transparent bool
}

// errNoContent is returned by NewWebView for windows showing a page.
var errNoContent = errors.New("mygo: web views go in windows showing native UI (WindowOptions.Content)")

// NewWebView creates a web view for the window's native UI, which shows it
// where its view places it with ui.WebView. The window must show Content;
// the web view closes with it.
func (w *Window) NewWebView(opts WebViewOptions) (*WebView, error) {
	bg, err := backgroundOption(opts.BackgroundColor)
	if err != nil {
		return nil, err
	}
	if w.content == nil {
		return nil, errNoContent
	}
	var v *WebView
	err = errDestroyed
	onMain(func() {
		if w.native != nil && w.conn != nil {
			v, err = w.newWebView(opts, bg)
		}
	})
	return v, err
}

// newWebView creates a web view of the window. Main thread only.
func (w *Window) newWebView(opts WebViewOptions, bg *background) (*WebView, error) {
	windows.Lock()
	windows.nextID++
	id := windows.nextID
	windows.Unlock()

	// The web view's page lives in a Window of its own, which no list
	// holds: everything a page does works through it as for a window.
	pw := &Window{id: id, host: w, trustedOrigins: opts.Page.TrustedOrigins, secret: randomText(), background: bg}
	pw.pg = &Page{pw}
	pw.resetPage()
	popts := pw.platformOptions(&WindowOptions{Page: opts.Page, Transparent: opts.Transparent})
	popts.BackgroundColor = pw.backgroundColor()
	if opts.Transparent {
		pw.background = nil
	}
	pw.devTools = popts.DevTools
	nv, err := w.conn.Surface.NewWebView(popts, &windowHandler{pw})
	if err != nil {
		return nil, fmt.Errorf("mygo: cannot create a web view: %w", err)
	}
	pw.native = &webViewNative{Window: w.native, web: nv}
	v := &WebView{pw}
	pw.webView = v

	windows.Lock()
	w.webViews = append(w.webViews, pw)
	windows.Unlock()

	if opts.URL != "" {
		if err := pw.pg.LoadURL(opts.URL); err != nil {
			log.Printf("mygo: web view: %v", err)
		}
	}
	return v, nil
}

// Page returns the web view's page.
func (v *WebView) Page() *Page { return v.w.pg }

// Window returns the window the web view is in.
func (v *WebView) Window() *Window { return v.w.host }

// Focus gives the web view the keyboard, once it shows.
func (v *WebView) Focus() {
	onMain(func() {
		if n, ok := v.w.native.(*webViewNative); ok {
			n.web.Focus()
		}
	})
}

// SetBackgroundColor sets the color filling the web view until its page
// paints, in the syntax of WebViewOptions.BackgroundColor, as when the
// app's own theme changes; one given with light-dark follows the
// appearance by itself.
func (v *WebView) SetBackgroundColor(color string) error { return v.w.SetBackgroundColor(color) }

// CapturePage returns a PNG screenshot of the web view's page.
func (v *WebView) CapturePage() ([]byte, error) { return v.w.CapturePage() }

// Destroy closes the web view and its page. Its window no longer shows it
// where the view still places it.
func (v *WebView) Destroy() { onMain(v.w.destroyWebView) }

// IsDestroyed reports whether the web view was destroyed, or its window
// closed.
func (v *WebView) IsDestroyed() bool { return v.w.destroyed.Load() }

// SurfaceWebView returns the native web view for package ui to place, nil
// once destroyed or for the content of another window, which is logged
// once. Main thread only.
func (v *WebView) SurfaceWebView(conn *surface.Conn) platform.WebView {
	if v == nil {
		return nil
	}
	n, ok := v.w.native.(*webViewNative)
	if !ok {
		return nil // destroyed
	}
	if conn.Window != v.w.host {
		if !n.misplaced {
			n.misplaced = true
			log.Printf("mygo: the content of window %d shows a web view of window %d", conn.Window.(*Window).ID(), v.w.host.ID())
		}
		return nil
	}
	return n.web
}

// placeWebViews shows the window's web views where its content's last
// frame shows them, and notes where each is. Main thread only.
func (w *Window) placeWebViews(views []platform.WebViewPlacement) {
	windows.Lock()
	for _, pw := range w.webViews {
		if n, ok := pw.native.(*webViewNative); ok {
			n.shown = false
			for _, p := range views {
				if p.WebView == n.web {
					n.frame, n.shown = p.Frame, true
				}
			}
		}
	}
	windows.Unlock()
	if w.conn != nil {
		w.conn.Surface.PlaceWebViews(views)
	}
}

// destroyWebView closes the native web view of a web view's page and
// forgets it. Main thread only.
func (w *Window) destroyWebView() {
	n, ok := w.native.(*webViewNative)
	if !ok {
		return
	}
	n.web.Close()
	w.native = nil
	w.destroyed.Store(true)
	windows.Lock()
	if h := w.host; h != nil {
		for i, x := range h.webViews {
			if x == w {
				h.webViews = append(h.webViews[:i:i], h.webViews[i+1:]...)
				break
			}
		}
	}
	windows.Unlock()
	w.mu.Lock()
	cancel := w.pageCancel
	w.mu.Unlock()
	cancel()
}

// destroyWebViews closes the web views of a window that closed. Main
// thread only.
func (w *Window) destroyWebViews() {
	windows.Lock()
	views := append([]*Window(nil), w.webViews...)
	windows.Unlock()
	for _, v := range views {
		v.destroyWebView()
	}
}

// top returns the window showing the page: the window of a web view,
// else the window itself.
func (w *Window) top() *Window {
	if w.host != nil {
		return w.host
	}
	return w
}

// pageWindows returns the windows with pages: every window, and the pages
// of the web views of those showing native UI.
func pageWindows() []*Window {
	windows.Lock()
	defer windows.Unlock()
	list := append([]*Window(nil), windows.list...)
	for _, w := range windows.list {
		list = append(list, w.webViews...)
	}
	return list
}

// webViewNative is the native side of a web view's page: its page
// methods are the web view's, and its window methods those of the window
// it is in, which the core calls for drag regions (StartDrag,
// TitleBarDoubleClicked).
type webViewNative struct {
	platform.Window
	web platform.WebView
	// frame is where the window's view last showed the web view, while
	// shown; misplaced is set once a view of another window placed it,
	// which is logged once.
	frame     platform.RectF
	shown     bool
	misplaced bool
}

func (n *webViewNative) Surface() platform.Surface   { return nil }
func (n *webViewNative) TitleBar() platform.TitleBar { return platform.TitleBar{} }
func (n *webViewNative) SetTitle(string)             {}
func (n *webViewNative) Close()                      { n.web.Close() }
func (n *webViewNative) Focus()                      { n.web.Focus() }
func (n *webViewNative) SetBackgroundColor(c platform.Color) {
	n.web.SetBackgroundColor(c)
}

func (n *webViewNative) WebViewHandle() uintptr              { return n.web.WebViewHandle() }
func (n *webViewNative) LoadURL(url string)                  { n.web.LoadURL(url) }
func (n *webViewNative) LoadHTML(html, baseURL string)       { n.web.LoadHTML(html, baseURL) }
func (n *webViewNative) LoadFile(path, readAccessDir string) { n.web.LoadFile(path, readAccessDir) }
func (n *webViewNative) Reload(ignoreCache bool)             { n.web.Reload(ignoreCache) }
func (n *webViewNative) StopLoading()                        { n.web.StopLoading() }
func (n *webViewNative) GoBack()                             { n.web.GoBack() }
func (n *webViewNative) GoForward()                          { n.web.GoForward() }
func (n *webViewNative) CanGoBack() bool                     { return n.web.CanGoBack() }
func (n *webViewNative) CanGoForward() bool                  { return n.web.CanGoForward() }
func (n *webViewNative) URL() string                         { return n.web.URL() }
func (n *webViewNative) IsLoading() bool                     { return n.web.IsLoading() }
func (n *webViewNative) Eval(js string)                      { n.web.Eval(js) }
func (n *webViewNative) SetUserScripts(scripts []platform.UserScript) {
	n.web.SetUserScripts(scripts)
}
func (n *webViewNative) CallAsyncFunction(body string, cb func(string, error)) {
	n.web.CallAsyncFunction(body, cb)
}
func (n *webViewNative) SetZoom(factor float64) { n.web.SetZoom(factor) }
func (n *webViewNative) Zoom() float64          { return n.web.Zoom() }
func (n *webViewNative) SetUserAgent(ua string) { n.web.SetUserAgent(ua) }
func (n *webViewNative) UserAgent() string      { return n.web.UserAgent() }
func (n *webViewNative) OpenDevTools()          { n.web.OpenDevTools() }
func (n *webViewNative) CloseDevTools()         { n.web.CloseDevTools() }
func (n *webViewNative) IsDevToolsOpened() bool { return n.web.IsDevToolsOpened() }
func (n *webViewNative) DroppedFiles() []string { return n.web.DroppedFiles() }
func (n *webViewNative) Print()                 { n.web.Print() }
func (n *webViewNative) CapturePage(cb func([]byte, error)) {
	n.web.CapturePage(cb)
}
func (n *webViewNative) PrintToPDF(opts platform.PDFOptions, cb func([]byte, error)) {
	n.web.PrintToPDF(opts, cb)
}
