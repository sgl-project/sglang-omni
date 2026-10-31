//go:build js

package web

import (
	"syscall/js"

	"github.com/egoist/mygo/internal/platform"
)

// window is a window of the page: an element filling the page's element
// with the ID "mygo", or else the page, holding the canvas its surface
// draws on. The page's title is the title of the window opened last.
type window struct {
	b    *Backend
	h    platform.WindowHandler
	opts *platform.WindowOptions
	el   js.Value
	s    *surface

	title   string
	visible bool
	focused bool
	closed  bool
	full    bool
}

// css is the style sheet of windows, added to the page once.
const css = `
.mygo-window{position:absolute;inset:0;overflow:hidden;contain:strict}
body>.mygo-window{position:fixed}
.mygo-window>canvas{display:block;width:100%;height:100%;touch-action:none;outline:none;user-select:none;-webkit-user-select:none}
.mygo-window>textarea{position:absolute;left:0;top:0;width:1px;height:1em;padding:0;margin:0;border:0;outline:none;resize:none;overflow:hidden;opacity:0;color:transparent;background:transparent;caret-color:transparent;white-space:pre;font-size:16px;z-index:-1}
`

var styled bool

func newWindow(b *Backend, o *platform.WindowOptions, h platform.WindowHandler) (*window, error) {
	doc := js.Global().Get("document")
	if !styled {
		styled = true
		style := doc.Call("createElement", "style")
		style.Set("textContent", css)
		doc.Get("head").Call("appendChild", style)
	}
	parent := doc.Call("getElementById", "mygo")
	if parent.IsNull() {
		parent = doc.Get("body")
	} else if js.Global().Call("getComputedStyle", parent).Get("position").String() == "static" {
		parent.Get("style").Set("position", "relative")
	}
	w := &window{b: b, h: h, opts: o, title: o.Title, full: o.FullScreen}
	w.el = doc.Call("createElement", "div")
	w.el.Set("className", "mygo-window")
	w.el.Get("style").Set("display", "none")
	if c := o.BackgroundColor; c != nil {
		w.SetBackgroundColor(*c)
	}
	parent.Call("appendChild", w.el)
	s, err := newSurface(w)
	if err != nil {
		w.el.Call("remove")
		return nil, err
	}
	w.s = s
	w.SetTitle(o.Title)
	return w, nil
}

func (w *window) Handle() uintptr           { return 0 }
func (w *window) Surface() platform.Surface { return w.s }

func (w *window) SetTitle(t string) {
	w.title = t
	if t != "" {
		js.Global().Get("document").Set("title", t)
	}
}

func (w *window) Title() string { return w.title }

func (w *window) SetRepresentedFile(string) {}
func (w *window) SetDocumentEdited(bool)    {}

// bounds returns the window's element's rectangle in the page.
func (w *window) bounds() platform.Rect {
	r := w.el.Call("getBoundingClientRect")
	return platform.Rect{X: r.Get("x").Int(), Y: r.Get("y").Int(), Width: r.Get("width").Int(), Height: r.Get("height").Int()}
}

// The page sets where windows are and how large: the methods setting it do
// nothing.
func (w *window) SetBounds(platform.Rect)                    {}
func (w *window) Bounds() platform.Rect                      { return w.bounds() }
func (w *window) SetContentBounds(platform.Rect)             {}
func (w *window) ContentBounds() platform.Rect               { return w.bounds() }
func (w *window) SetMinimumSize(platform.Size)               {}
func (w *window) SetMaximumSize(platform.Size)               {}
func (w *window) SetResizable(bool)                          {}
func (w *window) IsResizable() bool                          { return false }
func (w *window) SetMovable(bool)                            {}
func (w *window) IsMovable() bool                            { return false }
func (w *window) SetMinimizable(bool)                        {}
func (w *window) IsMinimizable() bool                        { return false }
func (w *window) SetMaximizable(bool)                        {}
func (w *window) IsMaximizable() bool                        { return false }
func (w *window) SetClosable(bool)                           {}
func (w *window) IsClosable() bool                           { return w.opts.Closable }
func (w *window) SetAlwaysOnTop(bool)                        {}
func (w *window) IsAlwaysOnTop() bool                        { return false }
func (w *window) Minimize()                                  {}
func (w *window) IsMinimized() bool                          { return false }
func (w *window) Maximize()                                  {}
func (w *window) Unmaximize()                                {}
func (w *window) IsMaximized() bool                          { return false }
func (w *window) Restore()                                   {}
func (w *window) Center()                                    {}
func (w *window) SetOpacity(float64)                         {}
func (w *window) Opacity() float64                           { return 1 }
func (w *window) SetHasShadow(bool)                          {}
func (w *window) HasShadow() bool                            { return false }
func (w *window) SetIgnoreMouseEvents(bool)                  {}
func (w *window) SetContentProtection(bool)                  {}
func (w *window) SetVibrancy(string)                         {}
func (w *window) SetProgressBar(string, float64)             {}
func (w *window) FlashFrame(bool)                            {}
func (w *window) SetSkipTaskbar(bool)                        {}
func (w *window) SetVisibleOnAllWorkspaces(bool)             {}
func (w *window) SetIcon([]byte) error                       { return errUnsupported }
func (w *window) SetMenu(*platform.Menu)                     {}
func (w *window) SetAutoHideMenu(bool)                       {}
func (w *window) TitleBar() platform.TitleBar                { return platform.TitleBar{} }
func (w *window) StartDrag()                                 {}
func (w *window) TitleBarDoubleClicked()                     {}
func (w *window) PlaceFlyout(platform.Flyout, platform.Size) {}

// Show shows the window and gives it the keyboard, unless the app is in a
// frame of another page: that page keeps the keyboard until the window is
// clicked, as frames loading do not take it from the page scrolling them.
func (w *window) Show() {
	w.ShowInactive()
	if js.Global().Get("top").Equal(js.Global().Get("self")) {
		w.Focus()
	}
}

func (w *window) ShowInactive() {
	if w.closed || w.visible {
		return
	}
	w.visible = true
	w.el.Get("style").Set("display", "")
	w.s.measure()
	w.s.RequestFrame()
}

func (w *window) Hide() {
	if !w.visible {
		return
	}
	w.visible = false
	w.el.Get("style").Set("display", "none")
}

func (w *window) IsVisible() bool { return w.visible }
func (w *window) Focus()          { w.s.focus() }
func (w *window) Blur()           { w.s.input.Call("blur") }
func (w *window) IsFocused() bool { return w.focused }

// SetFullScreen asks the browser to show the window's element on the whole
// screen, which it allows while the user acts.
func (w *window) SetFullScreen(v bool) {
	if v == w.full {
		return
	}
	w.full = v
	doc := js.Global().Get("document")
	if v {
		if fn := w.el.Get("requestFullscreen"); fn.Type() == js.TypeFunction {
			w.el.Call("requestFullscreen").Call("catch", js.Global().Get("Function").New(""))
		}
		w.h.EnteredFullScreen()
	} else {
		if !doc.Get("fullscreenElement").IsNull() {
			doc.Call("exitFullscreen").Call("catch", js.Global().Get("Function").New(""))
		}
		w.h.LeftFullScreen()
	}
}

func (w *window) IsFullScreen() bool { return w.full }

func (w *window) SetBackgroundColor(c platform.Color) {
	w.el.Get("style").Set("background", "rgba("+itoa(int(c.R))+","+itoa(int(c.G))+","+itoa(int(c.B))+","+ftoa(float64(c.A)/255)+")")
}

func (w *window) Close() {
	if w.closed {
		return
	}
	w.closed = true
	w.s.close()
	w.el.Call("remove")
	w.b.removeWindow(w)
	w.h.Closed()
}

// The page of a window is native UI's: none to load.
func (w *window) WebViewHandle() uintptr                                   { return 0 }
func (w *window) LoadURL(string)                                           {}
func (w *window) LoadHTML(string, string)                                  {}
func (w *window) LoadFile(string, string)                                  {}
func (w *window) Reload(bool)                                              {}
func (w *window) StopLoading()                                             {}
func (w *window) GoBack()                                                  {}
func (w *window) GoForward()                                               {}
func (w *window) CanGoBack() bool                                          { return false }
func (w *window) CanGoForward() bool                                       { return false }
func (w *window) URL() string                                              { return "" }
func (w *window) IsLoading() bool                                          { return false }
func (w *window) Eval(string)                                              {}
func (w *window) SetUserScripts([]platform.UserScript)                     {}
func (w *window) SetZoom(float64)                                          {}
func (w *window) Zoom() float64                                            { return 1 }
func (w *window) SetUserAgent(string)                                      {}
func (w *window) UserAgent() string                                        { return js.Global().Get("navigator").Get("userAgent").String() }
func (w *window) OpenDevTools()                                            {}
func (w *window) CloseDevTools()                                           {}
func (w *window) IsDevToolsOpened() bool                                   { return false }
func (w *window) DroppedFiles() []string                                   { return nil }
func (w *window) Print()                                                   {}
func (w *window) CapturePage(cb func([]byte, error))                       { cb(nil, errUnsupported) }
func (w *window) PrintToPDF(_ platform.PDFOptions, cb func([]byte, error)) { cb(nil, errUnsupported) }

func (w *window) CallAsyncFunction(_ string, cb func(string, error)) {
	cb("", errUnsupported)
}

// SetVibrancyStyle does nothing: a page shows no material.
func (w *window) SetVibrancyStyle(platform.VibrancyStyle) {}
