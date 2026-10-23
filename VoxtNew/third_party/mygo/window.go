package mygo

import (
	"context"
	"errors"
	"fmt"
	"log"
	"net/url"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"sync"
	"sync/atomic"
	"time"
	"unsafe"

	"github.com/egoist/mygo/internal/platform"
	"github.com/egoist/mygo/internal/surface"
)

// TitleBarStyle selects how the title bar of a framed window looks.
type TitleBarStyle string

// Title bar styles.
const (
	TitleBarDefault TitleBarStyle = "default"
	// TitleBarHidden gives the page the whole window while keeping the
	// window controls, which sit over it (see WindowOptions.TitleBarStyle).
	TitleBarHidden TitleBarStyle = "hidden"
	// TitleBarHiddenInset is TitleBarHidden with the window controls inset
	// further from the edges (macOS; TitleBarHidden elsewhere).
	TitleBarHiddenInset TitleBarStyle = "hiddenInset"
)

// DevTools selects whether the web inspector is available.
type DevTools int

// DevTools modes.
const (
	// DevToolsAuto enables the inspector unless the app was built for
	// production with `mygo build`.
	DevToolsAuto DevTools = iota
	DevToolsEnabled
	DevToolsDisabled
)

// Point is a position in screen coordinates.
type Point struct{ X, Y int }

// Size is a width and height in DIPs.
type Size struct{ Width, Height int }

// Rectangle is a position and size in screen coordinates (DIPs, origin at
// the top-left corner of the primary display).
type Rectangle struct{ X, Y, Width, Height int }

// WindowOptions configures NewWindow. The zero value is a visible,
// resizable 800x600 window centered on screen.
type WindowOptions struct {
	// Title of the window. Defaults to the application name. The title
	// follows the page's <title> unless an OnPageTitleUpdated listener
	// prevents it.
	Title string
	// URL is loaded right after the window is created. "/" and other URLs
	// without a scheme are pages of the app's frontend (see LoadURL).
	URL string

	// Width and Height of the window in DIPs (default 800x600).
	Width, Height int
	// UseContentSize makes Width and Height describe the page area instead
	// of the whole window.
	UseContentSize bool
	// X and Y of the top-left corner. The window is centered when both are
	// zero.
	X, Y                int
	MinWidth, MinHeight int
	MaxWidth, MaxHeight int

	// Hidden creates the window without showing it. Call Show, typically
	// from OnReadyToShow, to avoid a blank window while the page loads.
	Hidden bool
	// Frameless removes the title bar and window chrome. Mark draggable
	// areas with the CSS `--app-region: drag`.
	Frameless bool
	// TitleBarStyle hides the title bar but keeps the window controls, so
	// the page fills the window and draws its own title bar under them:
	// the traffic lights on macOS, and minimize, maximize and close at the
	// top corner on Linux and Windows. On Linux they are GTK's own title
	// buttons, and the desktop's button layout decides which show and on
	// which side, possibly none. Pages keep clear of them with the
	// --mygo-titlebar-* CSS variables and drag the window by their title
	// bar with --app-region: drag, as in a frameless window; native UI
	// with ui.Context.TitleBar and ui.Element.DragWindow.
	TitleBarStyle TitleBarStyle
	// TrafficLightPosition moves the window controls of a window with a
	// hidden title bar (macOS): the top-left corner of the close button
	// goes this far from the top-left corner of the window.
	TrafficLightPosition *Point
	// TitleBarHeight is the height of the title bar that the page of a
	// window with a hidden title bar draws (Linux, Windows): the window
	// controls fill it on Windows and are centered in it on Linux. Zero is
	// 32 on Windows, as Windows 11's own title bars, and the height of the
	// desktop's header bars on Linux.
	TitleBarHeight int

	DisableResize     bool
	DisableMove       bool
	DisableMinimize   bool
	DisableMaximize   bool
	DisableClose      bool
	DisableFullScreen bool
	DisableShadow     bool
	AlwaysOnTop       bool
	FullScreen        bool
	// Maximized creates the window maximized.
	Maximized bool
	// SkipTaskbar hides the window from the taskbar (Linux, Windows).
	SkipTaskbar bool
	// NonActivating makes a window that never activates the app nor takes
	// the keyboard: a press on it reaches its content while the app in
	// front keeps the focus and its caret, as for a floating HUD over the
	// text the user is typing into (dictation, a color picker). On macOS
	// the window is an NSPanel with NSWindowStyleMaskNonactivatingPanel
	// that never becomes the key or main window; on Windows it has
	// WS_EX_NOACTIVATE. Linux ignores it. Its content cannot take the
	// keyboard: no text inputs.
	NonActivating bool
	// AutoHideMenuBar keeps the window's menu bar out of sight until the
	// user presses Alt or F10, and hides it again once they leave it
	// (Linux, Windows). Its keyboard shortcuts work all along.
	AutoHideMenuBar bool
	// Transparent makes the window background transparent, so a page with a
	// transparent background shows the desktop through.
	Transparent bool
	// BackgroundColor fills the window until the page paints, which avoids
	// a flash of another color while it loads. CSS syntax: "#1e1e1e",
	// "#rgba", "rgb(30 30 30)", or "light-dark(#f5f5f7, #1e1e1e)" for a
	// page that follows the light or dark appearance.
	BackgroundColor string
	// Vibrancy puts a translucent, blurred material behind a transparent
	// page, or behind native UI where it draws no background (macOS and
	// Windows 11 22H2), e.g. VibrancySidebar; ui.Context.Vibrancy tells
	// native UI whether it shows. On Windows, only a window created with a
	// material can show one, and it has no menu bar: Alt and F10 open its
	// menus in a popup.
	Vibrancy Vibrancy
	// Opacity of the window between 0 and 1 (default 1).
	Opacity float64
	// Parent makes this window a child window that stays on top of it.
	Parent *Window
	// Modal makes a child window modal to its Parent.
	Modal bool

	// Panel makes the window a panel, as the windows of launchers and
	// quick entry (Spotlight, Raycast) are: it floats above the windows of
	// all apps, on whichever Space or virtual desktop the user is on, over
	// full-screen apps too, and stays out of the taskbar, the Dock and the
	// app switcher. It cannot be minimized, maximized or go full screen. On
	// macOS, showing or focusing it gives it the keyboard without
	// activating the app, so the app the user was in stays active and gets
	// the keyboard back when the panel hides.
	Panel bool

	// Page configures the window's web page.
	Page PageOptions

	// StateKey remembers the window's position, size, and maximized and
	// full screen state under this key, in window-state.json in
	// PathUserData or the store of App.SetWindowStateStore, saved when
	// the window closes and when the app quits. A window created with the
	// same key, for example at the next launch, gets them back as long as
	// it would show on a connected display: they override X, Y, Width,
	// Height, UseContentSize, Maximized and FullScreen.
	StateKey string

	// Content makes the window show a user interface MyGo draws itself,
	// on the GPU where it can, instead of a web page: create it with
	// package ui. The window then has no page (Window.Page is nil): URL
	// and Page are ignored.
	Content Content
}

// WindowState is what WindowOptions.StateKey remembers of a window: its
// bounds in the normal state, which it gets back when it leaves the
// maximized or full screen state, and that state.
type WindowState struct {
	X          int  `json:"x"`
	Y          int  `json:"y"`
	Width      int  `json:"width"`
	Height     int  `json:"height"`
	Maximized  bool `json:"maximized,omitempty"`
	FullScreen bool `json:"fullScreen,omitempty"`
}

// WindowStateStore keeps the states of the windows with a
// WindowOptions.StateKey, by key, for App.SetWindowStateStore. MyGo calls
// it on the main thread, so it should return quickly.
type WindowStateStore interface {
	// LoadWindowStates returns the saved states. It is called once, when
	// the first window with a StateKey is created. Without states (nil, or
	// an error) windows get their options.
	LoadWindowStates() (map[string]WindowState, error)
	// SaveWindowStates replaces the saved states with states, which the
	// store may keep. It is called when a window with a StateKey closes
	// and when the app quits, if a state changed. After an error the
	// next call saves the states again.
	SaveWindowStates(states map[string]WindowState) error
}

// SetWindowStateStore keeps the states of windows (WindowOptions.StateKey)
// in s, such as the app's own database, instead of window-state.json in
// PathUserData. nil restores the file. Call it before the first window
// with a StateKey is created, usually before Run. Browsers keep no window
// state, and never call s.
func (a *Application) SetWindowStateStore(s WindowStateStore) {
	a.mu.Lock()
	a.stateStore = s
	a.mu.Unlock()
}

// Window is a native window hosting a web page, or the Content of package
// ui. Create windows with NewWindow once the application is ready.
type Window struct {
	id     int
	parent *Window
	// flyout places a window NewFlyout created, at flyoutSize, the size
	// it asked for. Main thread only.
	flyout     *platform.Flyout
	flyoutSize platform.Size
	// panel is WindowOptions.Panel.
	panel bool
	// host is the window a web view is in, for the page of a web view
	// (webView); webViews are the pages of a window's web views, guarded
	// by the lock of windows.
	host     *Window
	webView  *WebView
	webViews []*Window
	// pg is the window's page, which Page returns unless the window shows
	// native UI.
	pg *Page
	// content is WindowOptions.Content; conn connects it to the native
	// surface (main thread only). devTools tells that PageOptions.DevTools
	// turned the developer tools on: the web inspector, or the inspector
	// of the Content.
	content      Content
	conn         *surface.Conn
	devTools     bool
	invalidating atomic.Bool
	// userScripts are the scripts the window's pages run at document
	// start: the bridge, the preferences and the preload script. Main
	// thread only.
	userScripts []platform.UserScript
	// secret starts every message of the bridge in this window's pages.
	// Only the bridge knows it, so messages posted from elsewhere (an
	// iframe of another origin) are ignored.
	secret string
	// preload is PageOptions.PreloadScript, which the scripts of the pages
	// loaded later keep (pageScripts).
	preload string

	// native is only accessed on the main thread; nil once destroyed.
	native    platform.Window
	destroyed atomic.Bool
	shown     bool
	readyShow bool
	// background is the BackgroundColor, if any. Main thread only.
	background *background
	// hiddenTitleBar is a TitleBarStyle that hides the title bar: the pages
	// hear the room the window controls take. Main thread only.
	hiddenTitleBar bool
	menu           *Menu
	// title is the title the app or the page set; representedFile and
	// documentEdited are the document the window shows. Main thread only.
	title           string
	representedFile string
	documentEdited  bool
	// stateKey is WindowOptions.StateKey; stateTimer captures the state
	// once it settled. Main thread only.
	stateKey   string
	stateTimer *time.Timer

	// trusted reports whether the current page may call bound methods.
	// Main thread only.
	trusted        bool
	trustedOrigins []string
	// html is the document LoadHTML loaded, while it is the page's or
	// about to be. Main thread only.
	html *htmlDocument

	mu                sync.Mutex
	openHandler       func(WindowOpenRequest) *WindowOptions
	permissionHandler func(PermissionRequest) bool
	pageCtx           context.Context
	pageCancel        context.CancelFunc
	// channels are those of the current page, by the page's id for them.
	channels map[int64]*channel
	// closedEarly holds the channels the current page closed before
	// their calls made them, with the page's token.
	closedEarly map[int64]string

	outMu    sync.Mutex
	outbox   []message
	flushing bool
	// held keeps events until the page's DOM is ready, so events sent right
	// after creating a window or during a navigation are not lost.
	held     []message
	domReady bool

	onClose            listeners[func(*CloseEvent)]
	onClosed           listeners[func()]
	onFocus            listeners[func()]
	onBlur             listeners[func()]
	onShow             listeners[func()]
	onHide             listeners[func()]
	onReadyToShow      listeners[func()]
	onResize           listeners[func()]
	onFileDrop         listeners[func(*FileDropEvent)]
	onWillDownload     listeners[func(*DownloadEvent)]
	onDownloadDone     listeners[func(*Download)]
	onMove             listeners[func()]
	onMaximize         listeners[func()]
	onUnmaximize       listeners[func()]
	onMinimize         listeners[func()]
	onRestore          listeners[func()]
	onEnterFullScreen  listeners[func()]
	onLeaveFullScreen  listeners[func()]
	onPageTitleUpdated listeners[func(*TitleEvent)]
	onWillNavigate     listeners[func(*NavigateEvent)]
	onDidNavigate      listeners[func(string)]
	onDOMReady         listeners[func()]
	onDidFinishLoad    listeners[func()]
	onDidFailLoad      listeners[func(*LoadError)]
	onRenderGone       listeners[func(string)]
}

// LoadError describes a failed page load.
type LoadError struct {
	URL         string
	Code        int
	Description string
}

func (e *LoadError) Error() string {
	return fmt.Sprintf("mygo: failed to load %s: %s (%d)", e.URL, e.Description, e.Code)
}

// WindowOpenRequest describes a window.open() call or a click on a
// target="_blank" link.
type WindowOpenRequest struct {
	URL       string
	FrameName string
	// Width and Height requested through window.open() features; 0 when
	// not specified.
	Width, Height int
}

var windows struct {
	sync.Mutex
	nextID  int
	list    []*Window
	focused *Window
}

// NewWindow creates a window. It must be called after the application is
// ready (see Application.WhenReady); when called from another goroutine it
// waits for that.
func NewWindow(opts WindowOptions) *Window {
	// Validate in the caller's goroutine so mistakes point at the call.
	bg, err := backgroundOption(opts.BackgroundColor)
	if err != nil {
		panic(err)
	}
	return createWindow("NewWindow", opts, bg, nil)
}

// createWindow creates a window, or a flyout, once the application is
// ready.
func createWindow(caller string, opts WindowOptions, bg *background, flyout *flyoutOptions) *Window {
	if !isMainThread() {
		App.waitReady()
	} else if !App.IsReady() {
		panic("mygo: " + caller + " called before the application is ready; create windows in App.WhenReady")
	}
	var w *Window
	onMain(func() { w = newWindow(opts, bg, 0, flyout) })
	if w == nil {
		// The event loop has already stopped.
		w = &Window{}
		w.destroyed.Store(true)
	}
	return w
}

func backgroundOption(s string) (*background, error) {
	if s == "" {
		return nil, nil
	}
	bg, err := parseBackground(s)
	if err != nil {
		return nil, err
	}
	return &bg, nil
}

// backgroundColor returns the window's background for the current
// appearance, or nil. Main thread only.
func (w *Window) backgroundColor() *platform.Color {
	if w.background == nil {
		return nil
	}
	c := w.background.light
	if w.background.dark != c && backend().Theme().IsDark() {
		c = w.background.dark
	}
	return &c
}

// updateBackgrounds gives windows with a light and a dark background the
// one of the current appearance. Main thread only.
func updateBackgrounds() {
	for _, w := range pageWindows() {
		if bg := w.background; w.native != nil && bg != nil && bg.light != bg.dark {
			w.native.SetBackgroundColor(*w.backgroundColor())
		}
	}
}

func newWindow(opts WindowOptions, bg *background, native uintptr, flyout *flyoutOptions) *Window {
	windows.Lock()
	windows.nextID++
	id := windows.nextID
	windows.Unlock()

	w := &Window{id: id, parent: opts.Parent, trustedOrigins: opts.Page.TrustedOrigins, secret: randomText(), stateKey: opts.StateKey, background: bg, content: opts.Content, panel: opts.Panel && flyout == nil}
	w.pg = &Page{w}
	w.hiddenTitleBar = !opts.Frameless && (opts.TitleBarStyle == TitleBarHidden || opts.TitleBarStyle == TitleBarHiddenInset)
	w.resetPage()
	popts := w.platformOptions(&opts)
	popts.BackgroundColor = w.backgroundColor()
	popts.Surface = opts.Content != nil
	if opts.Transparent {
		// Backends keep transparent windows clear; so do theme changes.
		w.background = nil
	}
	popts.Native = native
	if flyout != nil {
		w.flyout = flyout.flyout
		if t := flyout.tray; t != nil {
			if t.native == nil {
				panic("mygo: NewFlyout: the tray was destroyed")
			}
			w.flyout.Tray = t.native
		}
		w.flyoutSize = platform.Size{Width: popts.Width, Height: popts.Height}
		popts.Flyout = w.flyout
		popts.Focusable = w.flyout.Focusable
		popts.Center = false
	}
	w.devTools = popts.DevTools
	if w.stateKey != "" {
		restoreWindowState(w.stateKey, popts)
	}
	w.title = popts.Title
	nw, err := backend().NewWindow(popts, &windowHandler{w})
	if err != nil {
		panic(fmt.Sprintf("mygo: cannot create window: %v", err))
	}
	w.native = nw
	if w.stateKey != "" {
		w.initState(popts)
	}
	if w.content != nil {
		w.attachContent()
	}

	windows.Lock()
	windows.list = append(windows.list, w)
	windows.Unlock()

	fire1(&App.onWindowCreated, w)
	if opts.URL != "" && w.content == nil {
		w.pg.LoadURL(opts.URL)
	}
	if !opts.Hidden {
		w.Show()
	}
	return w
}

func (w *Window) platformOptions(o *WindowOptions) *platform.WindowOptions {
	po := o.Page
	p := &platform.WindowOptions{
		Title:          o.Title,
		X:              o.X,
		Y:              o.Y,
		Width:          or(o.Width, 800),
		Height:         or(o.Height, 600),
		Center:         o.X == 0 && o.Y == 0,
		UseContentSize: o.UseContentSize,
		MinSize:        platform.Size{Width: o.MinWidth, Height: o.MinHeight},
		MaxSize:        platform.Size{Width: o.MaxWidth, Height: o.MaxHeight},
		Resizable:      !o.DisableResize,
		Movable:        !o.DisableMove,
		Minimizable:    !o.DisableMinimize,
		Maximizable:    !o.DisableMaximize,
		Closable:       !o.DisableClose,
		Focusable:      !o.NonActivating,
		Fullscreenable: !o.DisableFullScreen,
		AlwaysOnTop:    o.AlwaysOnTop,
		FullScreen:     o.FullScreen,
		Maximized:      o.Maximized,
		SkipTaskbar:    o.SkipTaskbar,
		AutoHideMenu:   o.AutoHideMenuBar,
		HasShadow:      !o.DisableShadow,
		Frameless:      o.Frameless,
		Transparent:    o.Transparent,
		TitleBarStyle:  string(o.TitleBarStyle),
		TitleBarHeight: max(o.TitleBarHeight, 0),
		Vibrancy:       string(o.Vibrancy),
		Opacity:        o.Opacity,
		Modal:          o.Modal,
		UserAgent:      po.UserAgent,
		Zoom:           po.ZoomFactor,
		Schemes:        Protocol.schemes(),
		Panel:          o.Panel,
	}
	if o.Panel {
		p.Minimizable, p.Maximizable, p.Fullscreenable = false, false, false
		p.AlwaysOnTop, p.SkipTaskbar = true, true
	}
	if p.Title == "" {
		p.Title = App.Name()
	}
	if p.Opacity <= 0 || p.Opacity > 1 {
		p.Opacity = 1
	}
	if p.Zoom <= 0 {
		p.Zoom = 1
	}
	switch po.DevTools {
	case DevToolsEnabled:
		p.DevTools = true
	case DevToolsAuto:
		p.DevTools = IsDev()
	}
	if o.TrafficLightPosition != nil {
		p.TrafficLightPosition = &platform.Point{X: o.TrafficLightPosition.X, Y: o.TrafficLightPosition.Y}
	}
	if o.Parent != nil && o.Parent.native != nil {
		p.Parent = o.Parent.native
	}
	w.preload = po.PreloadScript
	p.UserScripts = w.pageScripts()
	// Kept to replace the preferences' and the locale's (sendPreferences,
	// pagesLocaleChanged).
	w.userScripts = p.UserScripts
	return p
}

func jsPlatform() string {
	if runtime.GOOS == "windows" {
		return "win32"
	}
	return runtime.GOOS
}

func or[T comparable](v, fallback T) T {
	var zero T
	if v == zero {
		return fallback
	}
	return v
}

// Windows returns all open windows in creation order.
func Windows() []*Window {
	windows.Lock()
	defer windows.Unlock()
	return append([]*Window(nil), windows.list...)
}

// FocusedWindow returns the focused window of this application, or nil.
func FocusedWindow() *Window {
	windows.Lock()
	defer windows.Unlock()
	return windows.focused
}

// WindowByID returns the window with the given id, or nil.
func WindowByID(id int) *Window {
	windows.Lock()
	defer windows.Unlock()
	for _, w := range windows.list {
		if w.id == id {
			return w
		}
	}
	return nil
}

// ID returns the unique id of the window. The page sees it as
// window.mygo.windowId.
func (w *Window) ID() int { return w.id }

// IsDestroyed reports whether the window has been closed.
func (w *Window) IsDestroyed() bool { return w.destroyed.Load() }

// Parent returns the parent window, or nil.
func (w *Window) Parent() *Window { return w.parent }

// do runs fn with the native window on the main thread, unless the window
// has been destroyed.
func (w *Window) do(fn func(n platform.Window)) {
	onMain(func() {
		if w.native != nil {
			fn(w.native)
		}
	})
}

// get is do for functions returning a value.
func get[T any](w *Window, fn func(n platform.Window) T) T {
	return onMainValue(func() T {
		if w.native == nil {
			var zero T
			return zero
		}
		return fn(w.native)
	})
}

// Close closes the window as if the user clicked its close button, so
// OnClose listeners can cancel it.
func (w *Window) Close() { onMain(func() { w.close() }) }

// close runs on the main thread and reports whether the window was closed.
func (w *Window) close() bool {
	if w.native == nil {
		return true
	}
	e := &CloseEvent{Window: w}
	fire1(&w.onClose, e)
	if e.prevented {
		return false
	}
	w.destroy()
	return true
}

// Destroy closes the window without emitting OnClose.
func (w *Window) Destroy() { onMain(w.destroy) }

func (w *Window) destroy() {
	if n := w.native; n != nil {
		w.closeState()
		n.Close()
		// Backends report Closed synchronously, but be defensive.
		(&windowHandler{w}).Closed()
	}
}

// Show shows and focuses the window.
func (w *Window) Show() {
	onMain(func() {
		if w.native == nil {
			return
		}
		visible := w.native.IsVisible()
		w.native.Show()
		w.shown = true
		if !visible {
			fire(&w.onShow)
		}
	})
}

// ShowInactive shows the window without focusing it.
func (w *Window) ShowInactive() {
	onMain(func() {
		if w.native == nil {
			return
		}
		visible := w.native.IsVisible()
		w.native.ShowInactive()
		w.shown = true
		if !visible {
			fire(&w.onShow)
		}
	})
}

// Hide hides the window.
func (w *Window) Hide() {
	onMain(func() {
		if w.native == nil || !w.native.IsVisible() {
			return
		}
		w.native.Hide()
		fire(&w.onHide)
	})
}

// IsPanel reports whether the window was created with
// WindowOptions.Panel.
func (w *Window) IsPanel() bool { return w.panel }

// IsVisible reports whether the window is shown.
func (w *Window) IsVisible() bool { return get(w, platform.Window.IsVisible) }

// Focus focuses the window.
func (w *Window) Focus() { w.do(platform.Window.Focus) }

// Blur removes focus from the window.
func (w *Window) Blur() { w.do(platform.Window.Blur) }

// IsFocused reports whether the window has keyboard focus.
func (w *Window) IsFocused() bool { return get(w, platform.Window.IsFocused) }

// Minimize minimizes the window.
func (w *Window) Minimize() { w.do(platform.Window.Minimize) }

// IsMinimized reports whether the window is minimized.
func (w *Window) IsMinimized() bool { return get(w, platform.Window.IsMinimized) }

// Maximize maximizes the window.
func (w *Window) Maximize() { w.do(platform.Window.Maximize) }

// Unmaximize restores a maximized window.
func (w *Window) Unmaximize() { w.do(platform.Window.Unmaximize) }

// ToggleMaximize maximizes the window or restores it if it is maximized.
func (w *Window) ToggleMaximize() {
	w.do(func(n platform.Window) {
		if n.IsMaximized() {
			n.Unmaximize()
		} else {
			n.Maximize()
		}
	})
}

// IsMaximized reports whether the window is maximized.
func (w *Window) IsMaximized() bool { return get(w, platform.Window.IsMaximized) }

// Restore restores a minimized window.
func (w *Window) Restore() { w.do(platform.Window.Restore) }

// SetFullScreen enters or leaves full screen.
func (w *Window) SetFullScreen(v bool) {
	w.do(func(n platform.Window) { n.SetFullScreen(v) })
}

// ToggleFullScreen enters full screen or leaves it.
func (w *Window) ToggleFullScreen() {
	w.do(func(n platform.Window) { n.SetFullScreen(!n.IsFullScreen()) })
}

// IsFullScreen reports whether the window is in full screen.
func (w *Window) IsFullScreen() bool { return get(w, platform.Window.IsFullScreen) }

// Center moves the window to the center of its screen.
func (w *Window) Center() { w.do(platform.Window.Center) }

// SetTitle sets the native window title.
func (w *Window) SetTitle(title string) {
	w.do(func(n platform.Window) {
		w.title = title
		w.applyTitle(n)
	})
}

// Title returns the window title, without the mark of an edited document
// (see SetDocumentEdited).
func (w *Window) Title() string {
	return get(w, func(platform.Window) string { return w.title })
}

// SetRepresentedFile tells which file the window shows, "" for none. On
// macOS the title bar shows its icon, which the user can drag, and
// Command-clicking the title lists the folders it is in. Other platforms
// show nothing. Documents (NewDocuments) set it for their windows.
func (w *Window) SetRepresentedFile(path string) {
	if path != "" {
		if abs, err := filepath.Abs(path); err == nil {
			path = abs
		}
	}
	w.do(func(n platform.Window) {
		w.representedFile = path
		n.SetRepresentedFile(path)
	})
}

// RepresentedFile returns the path SetRepresentedFile set.
func (w *Window) RepresentedFile() string {
	return onMainValue(func() string { return w.representedFile })
}

// SetDocumentEdited tells whether the window's document has unsaved
// changes: a dot in the close button on macOS, and a "*" before the title
// on Linux and Windows, as Notepad and gedit show it. Documents
// (NewDocuments) set it for their windows.
func (w *Window) SetDocumentEdited(edited bool) {
	w.do(func(n platform.Window) {
		if w.documentEdited == edited {
			return
		}
		w.documentEdited = edited
		n.SetDocumentEdited(edited)
		w.applyTitle(n)
	})
}

// IsDocumentEdited reports what SetDocumentEdited set.
func (w *Window) IsDocumentEdited() bool {
	return onMainValue(func() bool { return w.documentEdited })
}

// applyTitle gives the native window its title, marked when the document
// is edited where the title bar has no other mark. Main thread only.
func (w *Window) applyTitle(n platform.Window) {
	title := w.title
	if w.documentEdited && runtime.GOOS != "darwin" {
		title = "*" + title
	}
	n.SetTitle(title)
}

// SetBounds moves and resizes the window.
func (w *Window) SetBounds(r Rectangle) {
	w.do(func(n platform.Window) { n.SetBounds(platform.Rect(r)) })
}

// Bounds returns the position and size of the window.
func (w *Window) Bounds() Rectangle {
	return get(w, func(n platform.Window) Rectangle { return Rectangle(n.Bounds()) })
}

// SetContentBounds moves and resizes the window so the page area has the
// given bounds.
func (w *Window) SetContentBounds(r Rectangle) {
	w.do(func(n platform.Window) { n.SetContentBounds(platform.Rect(r)) })
}

// ContentBounds returns the bounds of the page area.
func (w *Window) ContentBounds() Rectangle {
	return get(w, func(n platform.Window) Rectangle { return Rectangle(n.ContentBounds()) })
}

// SetSize resizes the window.
func (w *Window) SetSize(width, height int) {
	w.do(func(n platform.Window) {
		if w.setFlyoutSize(width, height) {
			return
		}
		b := n.Bounds()
		b.Width, b.Height = width, height
		n.SetBounds(b)
	})
}

// Size returns the size of the window.
func (w *Window) Size() (width, height int) {
	b := w.Bounds()
	return b.Width, b.Height
}

// SetContentSize resizes the window so the page area has the given size.
func (w *Window) SetContentSize(width, height int) {
	w.do(func(n platform.Window) {
		if w.setFlyoutSize(width, height) {
			return
		}
		b := n.ContentBounds()
		b.Width, b.Height = width, height
		n.SetContentBounds(b)
	})
}

// ContentSize returns the size of the page area.
func (w *Window) ContentSize() (width, height int) {
	b := w.ContentBounds()
	return b.Width, b.Height
}

// SetPosition moves the window's top-left corner.
func (w *Window) SetPosition(x, y int) {
	w.do(func(n platform.Window) {
		b := n.Bounds()
		b.X, b.Y = x, y
		n.SetBounds(b)
	})
}

// Position returns the window's top-left corner.
func (w *Window) Position() (x, y int) {
	b := w.Bounds()
	return b.X, b.Y
}

// SetMinimumSize limits how small the window can be resized; 0 means no
// limit.
func (w *Window) SetMinimumSize(width, height int) {
	w.do(func(n platform.Window) { n.SetMinimumSize(platform.Size{Width: width, Height: height}) })
}

// SetMaximumSize limits how large the window can be resized; 0 means no
// limit.
func (w *Window) SetMaximumSize(width, height int) {
	w.do(func(n platform.Window) { n.SetMaximumSize(platform.Size{Width: width, Height: height}) })
}

// SetResizable sets whether the user can resize the window.
func (w *Window) SetResizable(v bool) { w.do(func(n platform.Window) { n.SetResizable(v) }) }

// IsResizable reports whether the user can resize the window.
func (w *Window) IsResizable() bool { return get(w, platform.Window.IsResizable) }

// SetMovable sets whether the user can move the window.
func (w *Window) SetMovable(v bool) { w.do(func(n platform.Window) { n.SetMovable(v) }) }

// IsMovable reports whether the user can move the window.
func (w *Window) IsMovable() bool { return get(w, platform.Window.IsMovable) }

// SetMinimizable sets whether the window can be minimized.
func (w *Window) SetMinimizable(v bool) { w.do(func(n platform.Window) { n.SetMinimizable(v) }) }

// IsMinimizable reports whether the window can be minimized.
func (w *Window) IsMinimizable() bool { return get(w, platform.Window.IsMinimizable) }

// SetMaximizable sets whether the window can be maximized.
func (w *Window) SetMaximizable(v bool) { w.do(func(n platform.Window) { n.SetMaximizable(v) }) }

// IsMaximizable reports whether the window can be maximized.
func (w *Window) IsMaximizable() bool { return get(w, platform.Window.IsMaximizable) }

// SetClosable sets whether the user can close the window.
func (w *Window) SetClosable(v bool) { w.do(func(n platform.Window) { n.SetClosable(v) }) }

// IsClosable reports whether the user can close the window.
func (w *Window) IsClosable() bool { return get(w, platform.Window.IsClosable) }

// SetAlwaysOnTop keeps the window above other windows.
func (w *Window) SetAlwaysOnTop(v bool) { w.do(func(n platform.Window) { n.SetAlwaysOnTop(v) }) }

// IsAlwaysOnTop reports whether the window stays above other windows.
func (w *Window) IsAlwaysOnTop() bool { return get(w, platform.Window.IsAlwaysOnTop) }

// SetBackgroundColor sets the color shown behind the page, in the syntax of
// WindowOptions.BackgroundColor.
func (w *Window) SetBackgroundColor(color string) error {
	bg, err := parseBackground(color)
	if err != nil {
		return err
	}
	w.do(func(n platform.Window) {
		w.background = &bg
		n.SetBackgroundColor(*w.backgroundColor())
	})
	return nil
}

// SetOpacity sets the window opacity between 0 and 1.
func (w *Window) SetOpacity(v float64) {
	w.do(func(n platform.Window) { n.SetOpacity(min(max(v, 0), 1)) })
}

// Opacity returns the window opacity.
func (w *Window) Opacity() float64 { return get(w, platform.Window.Opacity) }

// SetHasShadow sets whether the window has a shadow.
func (w *Window) SetHasShadow(v bool) { w.do(func(n platform.Window) { n.SetHasShadow(v) }) }

// HasShadow reports whether the window has a shadow.
func (w *Window) HasShadow() bool { return get(w, platform.Window.HasShadow) }

// SetIgnoreMouseEvents makes the window transparent to mouse events.
func (w *Window) SetIgnoreMouseEvents(v bool) {
	w.do(func(n platform.Window) { n.SetIgnoreMouseEvents(v) })
}

// SetContentProtection keeps the window content out of screenshots and
// screen recordings.
func (w *Window) SetContentProtection(v bool) {
	w.do(func(n platform.Window) { n.SetContentProtection(v) })
}

// SetVibrancy sets the material behind the page; VibrancyNone removes it.
// On Windows, it changes the material of a window created with one, and
// shows none in other windows. See WindowOptions.Vibrancy.
func (w *Window) SetVibrancy(v Vibrancy) {
	w.do(func(n platform.Window) { n.SetVibrancy(string(v)) })
	// Native UI looks again whether the material shows.
	w.Invalidate()
}

// VibrancyStyle shapes a window's material (WindowOptions.Vibrancy,
// SetVibrancy) on macOS: the material fills a rectangle inset from the
// edges of the window's content by Top, Right, Bottom and Left DIPs, its
// corners rounded by Radius, as for a transparent window that draws a
// rounded HUD with room for its own shadow around it. Dark shows the
// material as in the dark appearance whatever the app's, for a HUD that
// stays dark over anything. The zero value fills the window, as without a
// style. Other platforms ignore it.
type VibrancyStyle struct {
	Top, Right, Bottom, Left float64
	Radius                   float64
	Dark                     bool
}

// SetVibrancyStyle shapes the window's material (see VibrancyStyle); it
// holds for a material set later too.
func (w *Window) SetVibrancyStyle(s VibrancyStyle) {
	ps := platform.VibrancyStyle(s)
	w.do(func(n platform.Window) { n.SetVibrancyStyle(ps) })
}

// ProgressState is the state of a progress bar; see ProgressBar.
type ProgressState string

// Progress states.
const (
	ProgressNone          ProgressState = ""
	ProgressNormal        ProgressState = "normal"
	ProgressIndeterminate ProgressState = "indeterminate"
	// ProgressPaused and ProgressError color the bar yellow and red on
	// Windows, and show it like ProgressNormal elsewhere.
	ProgressPaused ProgressState = "paused"
	ProgressError  ProgressState = "error"
)

// ProgressBar is the progress of a task, shown on the window's taskbar
// button (Windows), on the application's Dock icon (macOS) or on its
// launcher entry (Linux docks that implement the Unity launcher API, such
// as KDE Plasma's and Ubuntu's). The zero value shows no progress.
type ProgressBar struct {
	// State defaults to ProgressNormal when Value is above 0, else to
	// ProgressNone.
	State ProgressState
	// Value is the fraction done, between 0 and 1.
	Value float64
}

// SetProgressBar shows the progress of a task:
//
//	win.SetProgressBar(mygo.ProgressBar{Value: done / total})
//	win.SetProgressBar(mygo.ProgressBar{}) // done: remove it
//
// On macOS and Linux the progress belongs to the application: the window
// that set it last wins.
func (w *Window) SetProgressBar(p ProgressBar) {
	state, value := p.State, min(max(p.Value, 0), 1)
	if state == ProgressNone && p.Value > 0 {
		state = ProgressNormal
	}
	w.do(func(n platform.Window) { n.SetProgressBar(string(state), value) })
}

// FlashFrame draws the user's attention to the window, or stops doing so:
// its taskbar button flashes until it is focused (Windows), it is marked
// urgent (Linux), or the Dock icon bounces (macOS, while the app is not
// active).
func (w *Window) FlashFrame(flash bool) {
	w.do(func(n platform.Window) { n.FlashFrame(flash) })
}

// SetSkipTaskbar hides the window from the taskbar, or shows it there again
// (Linux, Windows). See WindowOptions.SkipTaskbar.
func (w *Window) SetSkipTaskbar(v bool) {
	w.do(func(n platform.Window) { n.SetSkipTaskbar(v) })
}

// SetVisibleOnAllWorkspaces shows the window on every workspace (macOS
// Spaces, Linux virtual desktops), or on the current one only. On macOS it
// also shows over other apps in full screen, even with DisableFullScreen,
// if the app is an accessory (ActivationPolicyAccessory, Dock.Hide): macOS
// shows no windows of regular apps there.
func (w *Window) SetVisibleOnAllWorkspaces(v bool) {
	w.do(func(n platform.Window) { n.SetVisibleOnAllWorkspaces(v) })
}

// SetIcon sets the icon of the window, shown in its title bar and taskbar
// button (Linux, Windows), from a PNG image; nil restores the application
// icon. macOS windows show no icon of their own.
func (w *Window) SetIcon(png []byte) error {
	var err error
	w.do(func(n platform.Window) { err = n.SetIcon(png) })
	return err
}

// SetMenu sets the menu bar of this window (Linux, Windows). On macOS the
// menu bar belongs to the application; see Application.SetMenu.
func (w *Window) SetMenu(m *Menu) {
	onMain(func() {
		w.menu = m
		if w.native != nil {
			w.native.SetMenu(m.snapshot())
		}
	})
}

// SetAutoHideMenuBar keeps the menu bar of this window out of sight until
// the user presses Alt or F10, or shows it for good again (Linux,
// Windows). See WindowOptions.AutoHideMenuBar.
func (w *Window) SetAutoHideMenuBar(v bool) {
	w.do(func(n platform.Window) { n.SetAutoHideMenu(v) })
}

// NativeHandle returns the native window: NSWindow* on macOS, GtkWindow* on
// Linux and HWND on Windows.
func (w *Window) NativeHandle() uintptr { return get(w, platform.Window.Handle) }

// LoadURL navigates the page to url. A URL without a scheme, such as "/" or
// "/settings?tab=1", is a page of the app's frontend: the dev server at
// devUrl during `mygo dev`, the frontend built into the app otherwise (see
// SetFrontend). Besides http(s) URLs, schemes registered with
// Protocol.Handle can be used.
func (p *Page) LoadURL(rawURL string) error {
	if p.w.content != nil {
		return errNoPage
	}
	resolved, err := resolveURL(rawURL)
	if err != nil {
		return err
	}
	p.w.page(func(n platform.Window) {
		p.w.html = nil
		n.LoadURL(resolved)
	})
	return nil
}

// LoadFile loads a local HTML file. Relative paths are resolved against the
// working directory, then against the directory of the executable (and the
// Resources directory of a macOS app bundle).
func (p *Page) LoadFile(path string) error {
	if p.w.content != nil {
		return errNoPage
	}
	abs, err := resolveFile(path)
	if err != nil {
		return err
	}
	p.w.page(func(n platform.Window) {
		p.w.html = nil
		n.LoadFile(abs, filepath.Dir(abs))
	})
	return nil
}

func resolveFile(path string) (string, error) {
	if filepath.IsAbs(path) {
		if _, err := os.Stat(path); err != nil {
			return "", err
		}
		return path, nil
	}
	candidates := []string{}
	if wd, err := os.Getwd(); err == nil {
		candidates = append(candidates, filepath.Join(wd, path))
	}
	if exe, err := os.Executable(); err == nil {
		dir := filepath.Dir(exe)
		candidates = append(candidates, filepath.Join(dir, path), filepath.Join(dir, "..", "Resources", path))
	}
	for _, c := range candidates {
		if _, err := os.Stat(c); err == nil {
			return c, nil
		}
	}
	return "", fmt.Errorf("mygo: file %q not found", path)
}

// isTrusted reports whether a page at rawURL may call bound methods.
func (w *Window) isTrusted(rawURL string) bool {
	u, err := url.Parse(rawURL)
	if err != nil {
		return false
	}
	scheme := strings.ToLower(u.Scheme)
	switch {
	case scheme == "about" || scheme == "file" || scheme == frontendScheme:
		return true
	case Protocol.IsHandled(scheme):
		return true
	}
	origin := scheme + "://" + strings.ToLower(u.Host)
	if isDevOrigin(origin) {
		return true
	}
	for _, o := range w.trustedOrigins {
		if o == "*" || strings.EqualFold(strings.TrimSuffix(o, "/"), origin) {
			return true
		}
	}
	if IsDev() && (scheme == "http" || scheme == "https") {
		switch u.Hostname() {
		case "localhost", "127.0.0.1", "::1":
			return true
		}
	}
	return false
}

// LoadHTML loads an HTML string. Relative URLs in it resolve against
// baseURL, which may be empty, and the page is at baseURL, or about:blank.
// Reloading the page, with Reload or from the page, loads html again.
func (p *Page) LoadHTML(html, baseURL string) {
	p.w.page(func(n platform.Window) { p.w.loadHTML(n, html, baseURL) })
}

// htmlDocument is a document LoadHTML loaded.
type htmlDocument struct {
	html, baseURL string
	// shown is set once the page committed it.
	shown bool
}

// loadHTML loads html into the page and remembers it for its reloads: web
// views reload the page's URL instead, the base URL or a blank page. Main
// thread only.
func (w *Window) loadHTML(n platform.Window, html, baseURL string) {
	w.html = &htmlDocument{html: html, baseURL: baseURL}
	n.LoadHTML(html, baseURL)
}

// isAt reports whether url, which the page committed, is the document's,
// fragments aside.
func (d *htmlDocument) isAt(url string) bool {
	base := d.baseURL
	if base == "" {
		base = "about:blank"
	}
	trim := func(u string) string {
		u, _, _ = strings.Cut(u, "#")
		return strings.TrimSuffix(u, "/")
	}
	return trim(url) == trim(base)
}

// Reload reloads the page.
func (p *Page) Reload() { p.w.page(func(n platform.Window) { n.Reload(false) }) }

// ReloadIgnoringCache reloads the page bypassing the cache.
func (p *Page) ReloadIgnoringCache() { p.w.page(func(n platform.Window) { n.Reload(true) }) }

// Stop stops loading the page.
func (p *Page) Stop() { p.w.page(platform.Window.StopLoading) }

// GoBack navigates back in history.
func (p *Page) GoBack() { p.w.page(platform.Window.GoBack) }

// GoForward navigates forward in history.
func (p *Page) GoForward() { p.w.page(platform.Window.GoForward) }

// CanGoBack reports whether there is a previous page in history.
func (p *Page) CanGoBack() bool { return pageGet(p.w, platform.Window.CanGoBack) }

// CanGoForward reports whether there is a next page in history.
func (p *Page) CanGoForward() bool { return pageGet(p.w, platform.Window.CanGoForward) }

// URL returns the URL of the current page.
func (p *Page) URL() string { return pageGet(p.w, platform.Window.URL) }

// IsLoading reports whether the page is still loading.
func (p *Page) IsLoading() bool { return pageGet(p.w, platform.Window.IsLoading) }

// SetZoomFactor zooms the page; 1 is 100%.
func (p *Page) SetZoomFactor(f float64) {
	if f <= 0 {
		return
	}
	p.w.page(func(n platform.Window) {
		n.SetZoom(f)
		p.w.sendTitleBar() // the controls take other CSS pixels
	})
}

// ZoomFactor returns the page zoom; 1 is 100%.
func (p *Page) ZoomFactor() float64 { return pageGet(p.w, platform.Window.Zoom) }

// SetUserAgent overrides the user agent for subsequent requests.
func (p *Page) SetUserAgent(ua string) { p.w.page(func(n platform.Window) { n.SetUserAgent(ua) }) }

// UserAgent returns the user agent of the page.
func (p *Page) UserAgent() string { return pageGet(p.w, platform.Window.UserAgent) }

// OpenDevTools opens the web inspector (unless disabled with
// PageOptions.DevTools).
func (p *Page) OpenDevTools() { p.w.page(platform.Window.OpenDevTools) }

// CloseDevTools closes the web inspector.
func (p *Page) CloseDevTools() { p.w.page(platform.Window.CloseDevTools) }

// IsDevToolsOpened reports whether the web inspector is open.
func (p *Page) IsDevToolsOpened() bool { return pageGet(p.w, platform.Window.IsDevToolsOpened) }

// ToggleDevTools opens or closes the web inspector.
func (p *Page) ToggleDevTools() {
	p.w.page(func(n platform.Window) {
		if n.IsDevToolsOpened() {
			n.CloseDevTools()
		} else {
			n.OpenDevTools()
		}
	})
}

// Print opens the print dialog for the page.
func (p *Page) Print() { p.w.page(platform.Window.Print) }

// CapturePage returns a PNG screenshot of the visible page. For a window
// showing Content it renders the content as it is.
func (w *Window) CapturePage() ([]byte, error) {
	if w.content != nil {
		return w.captureContent()
	}
	type result struct {
		png []byte
		err error
	}
	ch := make(chan result, 1)
	w.do(func(n platform.Window) {
		n.CapturePage(func(png []byte, err error) { deliver(ch, result{png, err}) })
	})
	if w.IsDestroyed() {
		return nil, errDestroyed
	}
	r := await(ch)
	return r.png, r.err
}

var errDestroyed = errors.New("mygo: window has been destroyed")

// errNoPage is returned by the page methods of windows showing Content.
var errNoPage = errors.New("mygo: the window shows Content, not a web page")

// Eval evaluates JavaScript in the page and returns its result, decoded
// from JSON. An expression's value is returned, with promises awaited:
//
//	title, err := win.Page().Eval("document.title")
//	data, err := win.Page().Eval("fetch('/data.json').then(r => r.json())")
//
// Statements run as the body of an async function, so use return to produce
// a value:
//
//	n, err := win.Page().Eval("const items = document.querySelectorAll('li'); return items.length")
//
// Eval is not subject to the page's Content Security Policy.
func (p *Page) Eval(code string) (any, error) {
	return p.EvalContext(context.Background(), code)
}

// EvalContext is Eval with a context that can abort the wait.
func (p *Page) EvalContext(ctx context.Context, code string) (any, error) {
	raw, err := p.w.eval(ctx, code)
	if err != nil || len(raw) == 0 {
		return nil, err
	}
	var v any
	if err := decodeResult(raw, &v); err != nil {
		return nil, err
	}
	return v, nil
}

// EvalAs evaluates JavaScript in a page like Page.Eval and decodes the
// result into a value of type T:
//
//	title, err := mygo.EvalAs[string](win.Page(), "document.title")
//
// A nil page, as a window showing native UI has, returns an error.
func EvalAs[T any](p *Page, code string) (T, error) {
	var v T
	if p == nil {
		return v, errNoPage
	}
	raw, err := p.w.eval(context.Background(), code)
	if err != nil || len(raw) == 0 {
		return v, err
	}
	err = decodeResult(raw, &v)
	return v, err
}

// EvalError is returned by Eval when the script throws.
type EvalError struct {
	Message string
}

func (e *EvalError) Error() string { return "mygo: javascript error: " + e.Message }

// SetWindowOpenHandler decides what happens on window.open() and clicks on
// target="_blank" links. The handler returns the options of the window to
// open, or nil to deny the request. Without a handler, http(s) URLs open in
// the default browser and everything else is denied.
//
// On macOS the new page keeps its relation to the opener (window.opener).
// On Linux it opens as an independent page, because WebKitGTK would share
// the opener's script message routing with a related page.
func (p *Page) SetWindowOpenHandler(fn func(req WindowOpenRequest) *WindowOptions) {
	p.w.mu.Lock()
	p.w.openHandler = fn
	p.w.mu.Unlock()
}

// resetPage starts a new page context; the previous one is canceled.
func (w *Window) resetPage() {
	ctx := context.WithValue(context.WithValue(context.Background(), callerKey{}, w.top()), pageKey{}, w.pg)
	ctx, cancel := context.WithCancel(ctx)
	w.mu.Lock()
	prev := w.pageCancel
	w.pageCtx, w.pageCancel = ctx, cancel
	w.channels = nil // they close with the previous page's context
	w.closedEarly = nil
	w.mu.Unlock()
	if prev != nil {
		prev()
	}
}

// pageContext returns the context of the current page. It is canceled when
// the page navigates away or the window is destroyed.
func (w *Window) pageContext() context.Context {
	w.mu.Lock()
	defer w.mu.Unlock()
	return w.pageCtx
}

// message is a message for the page: a JSON object, in parts that are
// joined when it is sent, so that an encoded value, which can be large, is
// not copied into it.
type message struct {
	head, value, tail []byte
}

func (m message) len() int { return len(m.head) + len(m.value) + len(m.tail) }

func (m message) appendTo(b []byte) []byte {
	return append(append(append(b, m.head...), m.value...), m.tail...)
}

// maxHeldEvents bounds the events kept for a page that never becomes ready
// (e.g. an image or a failed load).
const maxHeldEvents = 1024

// enqueue schedules a message for the page. Messages are batched into a
// single script evaluation per main loop iteration. Events are held until
// the DOM of the page is ready; replies to calls are sent right away since
// the page is waiting for them.
func (w *Window) enqueue(msg message, event bool) {
	if w.content != nil {
		return
	}
	w.outMu.Lock()
	if event && !w.domReady {
		if len(w.held) == maxHeldEvents {
			w.held = w.held[1:]
		}
		w.held = append(w.held, msg)
		w.outMu.Unlock()
		return
	}
	w.outbox = append(w.outbox, msg)
	schedule := !w.flushing
	w.flushing = true
	w.outMu.Unlock()
	if schedule {
		postMain(w.flush)
	}
}

func (w *Window) flush() {
	w.outMu.Lock()
	msgs := w.outbox
	w.outbox = nil
	w.flushing = false
	w.outMu.Unlock()
	if w.native == nil || len(msgs) == 0 {
		return
	}
	size := 64
	for _, m := range msgs {
		size += m.len() + 1
	}
	js := make([]byte, 0, size)
	// A script shaped like a.b(JSON) runs without being compiled: WebKit's
	// JavaScriptCore parses the value as JSON unless the inspector is on,
	// several times faster. Pages without the runtime only throw.
	js = append(js, "__mygo.receive(["...)
	for i, m := range msgs {
		if i > 0 {
			js = append(js, ',')
		}
		js = m.appendTo(js)
	}
	js = append(js, "])"...)
	// js is not used again, so the script can share its memory.
	w.native.Eval(unsafe.String(unsafe.SliceData(js), len(js)))
}

// OnClose is called when the window is about to close. Call
// e.PreventDefault to keep it open.
func (w *Window) OnClose(fn func(e *CloseEvent)) (off func()) { return w.onClose.add(fn, false) }

// OnClosed is called after the window has been closed.
func (w *Window) OnClosed(fn func()) (off func()) { return w.onClosed.add(fn, false) }

// OnFocus is called when the window gains focus.
func (w *Window) OnFocus(fn func()) (off func()) { return w.onFocus.add(fn, false) }

// OnBlur is called when the window loses focus.
func (w *Window) OnBlur(fn func()) (off func()) { return w.onBlur.add(fn, false) }

// OnShow is called when the window is shown.
func (w *Window) OnShow(fn func()) (off func()) { return w.onShow.add(fn, false) }

// OnHide is called when the window is hidden.
func (w *Window) OnHide(fn func()) (off func()) { return w.onHide.add(fn, false) }

// OnReadyToShow is called once, when the first page is ready to be
// displayed. Create the window with Hidden and call Show here to avoid a
// visual flash. A window showing Content is ready once it drew its first
// frame, which a hidden window may not draw until shown.
func (w *Window) OnReadyToShow(fn func()) (off func()) { return w.onReadyToShow.add(fn, false) }

// OnResize is called after the window was resized.
func (w *Window) OnResize(fn func()) (off func()) { return w.onResize.add(fn, false) }

// OnFileDrop is called when files, for example from Finder or Explorer, are
// dropped on the page, with their paths. The app's own pages also get them,
// through onFileDrop of mygo-runtime. In a window showing native UI, it
// gets the files no element takes (ui.Element.DroppedFiles), with X and Y
// in DIPs.
//
// The page's own drag and drop keeps working: drop events still reach it
// with the File objects, and drags that start in the page are left alone.
// Only where the page does not handle dragged files are they accepted, so
// that dropping them there reaches OnFileDrop instead of replacing the page
// with the file.
func (w *Window) OnFileDrop(fn func(e *FileDropEvent)) (off func()) {
	return w.onFileDrop.add(fn, false)
}

// OnMove is called after the window was moved.
func (w *Window) OnMove(fn func()) (off func()) { return w.onMove.add(fn, false) }

// OnMaximize is called when the window is maximized.
func (w *Window) OnMaximize(fn func()) (off func()) { return w.onMaximize.add(fn, false) }

// OnUnmaximize is called when the window leaves the maximized state.
func (w *Window) OnUnmaximize(fn func()) (off func()) { return w.onUnmaximize.add(fn, false) }

// OnMinimize is called when the window is minimized.
func (w *Window) OnMinimize(fn func()) (off func()) { return w.onMinimize.add(fn, false) }

// OnRestore is called when the window is restored from minimized.
func (w *Window) OnRestore(fn func()) (off func()) { return w.onRestore.add(fn, false) }

// OnEnterFullScreen is called when the window entered full screen.
func (w *Window) OnEnterFullScreen(fn func()) (off func()) {
	return w.onEnterFullScreen.add(fn, false)
}

// OnLeaveFullScreen is called when the window left full screen.
func (w *Window) OnLeaveFullScreen(fn func()) (off func()) {
	return w.onLeaveFullScreen.add(fn, false)
}

// OnPageTitleUpdated is called when the page's <title> changes. Preventing
// the event keeps the native window title.
func (p *Page) OnPageTitleUpdated(fn func(e *TitleEvent)) (off func()) {
	return p.w.onPageTitleUpdated.add(fn, false)
}

// OnWillNavigate is called before the page navigates to another URL, for
// example after a link click. Preventing the event cancels the navigation.
// It is not called for LoadURL and friends.
func (p *Page) OnWillNavigate(fn func(e *NavigateEvent)) (off func()) {
	return p.w.onWillNavigate.add(fn, false)
}

// OnDidNavigate is called when a navigation committed and a new page
// started.
func (p *Page) OnDidNavigate(fn func(url string)) (off func()) {
	return p.w.onDidNavigate.add(fn, false)
}

// OnDOMReady is called when the page's DOM is ready (DOMContentLoaded).
func (p *Page) OnDOMReady(fn func()) (off func()) { return p.w.onDOMReady.add(fn, false) }

// OnDidFinishLoad is called when the page finished loading.
func (p *Page) OnDidFinishLoad(fn func()) (off func()) { return p.w.onDidFinishLoad.add(fn, false) }

// OnDidFailLoad is called when the page failed to load.
func (p *Page) OnDidFailLoad(fn func(err *LoadError)) (off func()) {
	return p.w.onDidFailLoad.add(fn, false)
}

// OnRenderProcessGone is called when the web content process crashed or
// was killed. Call Reload to recover.
func (p *Page) OnRenderProcessGone(fn func(reason string)) (off func()) {
	return p.w.onRenderGone.add(fn, false)
}

func (w *Window) readyToShow() {
	if w.readyShow {
		return
	}
	w.readyShow = true
	fire(&w.onReadyToShow)
}

// windowHandler receives native window events.
type windowHandler struct{ w *Window }

func (h *windowHandler) ShouldClose() bool {
	e := &CloseEvent{Window: h.w}
	fire1(&h.w.onClose, e)
	if !e.prevented {
		h.w.closeState()
	}
	return !e.prevented
}

func (h *windowHandler) Closed() {
	w := h.w
	if w.native == nil {
		return
	}
	w.cancelDataDrag()
	w.destroyWebViews()
	w.detachContent()
	w.native = nil
	w.destroyed.Store(true)

	windows.Lock()
	for i, x := range windows.list {
		if x == w {
			windows.list = append(windows.list[:i:i], windows.list[i+1:]...)
			break
		}
	}
	if windows.focused == w {
		windows.focused = nil
	}
	children := []*Window{}
	for _, x := range windows.list {
		if x.parent == w {
			children = append(children, x)
		}
	}
	windows.Unlock()

	// Children close with their parent, which reports the last window
	// once they are all gone.
	closingParents++
	for _, c := range children {
		c.destroy()
	}
	closingParents--
	w.mu.Lock()
	cancel := w.pageCancel
	w.mu.Unlock()
	cancel()

	fire(&w.onClosed)
	if closingParents == 0 && len(Windows()) == 0 {
		App.lastWindowClosed()
	}
}

// closingParents counts the windows closing their children (main thread
// only).
var closingParents int

func (h *windowHandler) Focused() {
	windows.Lock()
	windows.focused = h.w
	windows.Unlock()
	fire(&h.w.onFocus)
}

func (h *windowHandler) Blurred() {
	windows.Lock()
	if windows.focused == h.w {
		windows.focused = nil
	}
	windows.Unlock()
	fire(&h.w.onBlur)
}

func (h *windowHandler) MenuItemClicked(id int) {
	if h.w.native != nil {
		menuItemClicked(id, h.w)
	}
}

func (h *windowHandler) Resized()           { h.w.stateChanged(); h.w.placeFlyouts(); fire(&h.w.onResize) }
func (h *windowHandler) Moved()             { h.w.stateChanged(); h.w.placeFlyouts(); fire(&h.w.onMove) }
func (h *windowHandler) Minimized()         { h.w.stateChanged(); fire(&h.w.onMinimize) }
func (h *windowHandler) Restored()          { h.w.stateChanged(); fire(&h.w.onRestore) }
func (h *windowHandler) Maximized()         { h.w.stateChanged(); fire(&h.w.onMaximize) }
func (h *windowHandler) Unmaximized()       { h.w.stateChanged(); fire(&h.w.onUnmaximize) }
func (h *windowHandler) EnteredFullScreen() { h.w.stateChanged(); fire(&h.w.onEnterFullScreen) }
func (h *windowHandler) LeftFullScreen()    { h.w.stateChanged(); fire(&h.w.onLeaveFullScreen) }
func (h *windowHandler) TitleBarChanged() {
	h.w.sendTitleBar()
	if c := h.w.conn; c != nil && c.TitleBarChanged != nil {
		c.TitleBarChanged()
	}
}

func (h *windowHandler) Message(msg string) { h.w.handleMessage(msg) }

func (h *windowHandler) WillNavigate(nav platform.Navigation) bool {
	if doc := h.w.html; nav.IsMainFrame && nav.Reload && doc != nil && doc.shown {
		// The reload of a document LoadHTML loaded loads it again.
		postMain(func() {
			if w := h.w; w.html == doc && w.native != nil {
				w.loadHTML(w.native, doc.html, doc.baseURL)
			}
		})
		return false
	}
	if !nav.IsMainFrame || nav.Unasked {
		return true
	}
	e := &NavigateEvent{URL: nav.URL, UserInitiated: nav.UserInitiated}
	fire1(&h.w.onWillNavigate, e)
	return !e.prevented
}

func (h *windowHandler) NavigationStarted(string) {}

func (h *windowHandler) NavigationCommitted(url string) {
	if doc := h.w.html; doc != nil {
		switch {
		case doc.shown:
			h.w.html = nil // another document replaced it
		case doc.isAt(url):
			doc.shown = true
		}
	}
	h.w.trusted = h.w.isTrusted(url)
	h.w.outMu.Lock()
	h.w.domReady = false
	h.w.outMu.Unlock()
	h.w.resetPage()
	fire1(&h.w.onDidNavigate, url)
}

func (h *windowHandler) LoadFinished() {
	fire(&h.w.onDidFinishLoad)
	// Pages without the bridge (e.g. images) never report DOM ready.
	h.w.readyToShow()
}

func (h *windowHandler) LoadFailed(url string, code int, desc string) {
	fire1(&h.w.onDidFailLoad, &LoadError{URL: url, Code: code, Description: desc})
}

func (h *windowHandler) TitleChanged(title string) {
	if title == "" {
		return
	}
	e := &TitleEvent{Title: title}
	fire1(&h.w.onPageTitleUpdated, e)
	if !e.prevented && h.w.native != nil {
		h.w.title = title
		h.w.applyTitle(h.w.native)
	}
}

func (h *windowHandler) NewWindow(req platform.NewWindowRequest) platform.Window {
	h.w.mu.Lock()
	handler := h.w.openHandler
	h.w.mu.Unlock()
	r := WindowOpenRequest{URL: req.URL, FrameName: req.FrameName, Width: req.Width, Height: req.Height}
	if handler == nil {
		if u, err := url.Parse(req.URL); err == nil && (u.Scheme == "http" || u.Scheme == "https") {
			go Shell.OpenExternal(req.URL)
		}
		return nil
	}
	opts := handler(r)
	if opts == nil {
		return nil
	}
	o := *opts
	if o.Width == 0 && req.Width > 0 {
		o.Width, o.Height = req.Width, req.Height
	}
	// The webview loads the URL itself.
	o.URL = ""
	bg, err := backgroundOption(o.BackgroundColor)
	if err != nil {
		log.Printf("mygo: window open handler: %v", err)
	}
	child := newWindow(o, bg, req.Native, nil)
	return child.native
}

func (h *windowHandler) ClosedByPage() {
	if h.w.host == nil { // a web view's page does not close its window
		h.w.close()
	}
}

func (h *windowHandler) RenderProcessGone(reason string) { fire1(&h.w.onRenderGone, reason) }

// filesDropped tells OnFileDrop listeners, and the app's own pages, about
// dropped files.
func (w *Window) filesDropped(paths []string, x, y int) {
	if len(paths) == 0 {
		return
	}
	if n, ok := w.native.(*webViewNative); ok {
		// Where the web view shows in its window.
		x, y = x+int(n.frame.X), y+int(n.frame.Y)
	}
	fire1(&w.top().onFileDrop, &FileDropEvent{Paths: paths, X: x, Y: y})
	// Local paths are none of other pages' business.
	if w.trusted {
		if msg, err := encodeEvent(fileDropEvent, FileDropEvent{Paths: paths, X: x, Y: y}); err == nil {
			w.enqueue(msg, true)
		}
	}
}
