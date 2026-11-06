//go:build linux && (amd64 || arm64)

package linux

import (
	"bytes"
	"errors"
	"fmt"
	"net/url"
	"os"
	"slices"
	"strings"
	"sync"
	"unsafe"

	"github.com/ebitengine/purego"

	"github.com/egoist/mygo/internal/platform"
)

// GDK window state flags.
const (
	stateIconified  = 1 << 1
	stateMaximized  = 1 << 2
	stateFullscreen = 1 << 4
	stateAbove      = 1 << 5
	stateFocused    = 1 << 7
	stateTiled      = 1 << 8
	// The edges the window manager lets resize (GDK 3.22.23 and later).
	stateTopResizable    = 1 << 10
	stateRightResizable  = 1 << 12
	stateBottomResizable = 1 << 14
	stateLeftResizable   = 1 << 16
)

type window struct {
	b    *Backend
	id   int
	h    platform.WindowHandler
	opts *platform.WindowOptions

	win     ptr // GtkWindow
	box     ptr // GtkBox holding the menu bar and the webview
	web     ptr // WebKitWebView
	ucm     ptr // WebKitUserContentManager
	menubar ptr
	accel   ptr
	owner   int
	ownMenu bool
	// autoHideMenu shows the menu bar only while its menus are open;
	// altAlone is an Alt press no other key or click has joined.
	autoHideMenu, altAlone bool
	// surface shows the content MyGo draws, in place of the web view.
	surface *surface
	// webViews are the web views under the surface (webview.go), and host
	// the window whose surface a web view is under; accessParent is the
	// node of the content its page's accessible is in.
	webViews     []*window
	host         *window
	accessParent ptr
	// controls are the title buttons over the page of a window with a
	// hidden title bar (titlebar.go), in an overlay with the web view.
	controls *windowControls

	// flyout is how a flyout (WindowOptions.Flyout) was placed last, at
	// flyoutSize, once placed is set. grabbed tells that a focusable one
	// grabs the input, dismissing that it asks whether to close; unmaps
	// counts the unmap events its hiding has yet to bring.
	flyout                      platform.Flyout
	flyoutSize                  platform.Size
	placed, grabbed, dismissing bool
	unmaps                      int
	// layer tells that a panel is a layer surface (layer.go), put at
	// layerAt, or centered by the compositor without it.
	layer   bool
	layerAt *platform.Point

	closed       bool
	programmatic bool
	keepAbove    bool
	// requested holds bounds set with SetBounds until the window manager
	// answers: GTK resizes and moves asynchronously. requestFrame is the
	// window manager's frame they were asked for with, and requestInside
	// tells that the size asked for is the one inside it (as with
	// SetContentBounds). previous is the size GTK gave the window then,
	// which reports sent before still carry. placing is set while the
	// window manager places a window that shows.
	requested     *platform.Rect
	requestFrame  extents
	requestInside bool
	previous      [2]int32
	placing       bool
	// framed tells that the window manager framed the window that shows,
	// and frame is its frame as last seen, once seenFrame is set (see
	// frameExtents).
	framed, seenFrame bool
	frame             extents
	state             uint32
	x, y, w, hgt      int32
	minW, minH        int32
	maxW, maxH        int32

	// activationResize holds the latest bounds asked for while a window
	// drawing with GL on Wayland is not focused, and activationPrevious the
	// size it had before: the configure event of its activation can bring
	// that size back. resizeIdle is the idle source that then asks for the
	// bounds again (cbActivationResize).
	activationResize   *platform.Rect
	activationPrevious [2]int32
	resizeIdle         uint32

	// dragged holds the paths of the files dragged over the page, dropped
	// those of the files dropped on it, for DroppedFiles.
	dragged, dropped []string

	// The last button press on the page, used for window dragging and
	// context menu positioning.
	press struct {
		button     int32
		rootX      float64
		rootY      float64
		time       uint32
		event      ptr
		hasPressed bool
	}

	// dragPending holds whether a window drag was requested with the last
	// press and waits for the pointer to move; see StartDrag.
	dragPending bool

	// cursor is the resize cursor shown over an edge of a frameless
	// window, and the page's cursor it replaced.
	cursor struct {
		on    bool
		edge  int32
		shown ptr
		saved ptr // a reference
	}
}

func (b *Backend) window(data ptr) *window {
	w := b.windows[int(data)]
	if w == nil {
		w = b.webViews[int(data)]
	}
	if w == nil || w.closed {
		return nil
	}
	return w
}

func (b *Backend) NewWindow(o *platform.WindowOptions, h platform.WindowHandler) (platform.Window, error) {
	if !o.Surface {
		if err := webKit(); err != nil {
			return nil, err
		}
	}
	b.nextID++
	w := &window{b: b, id: b.nextID, h: h, opts: o, autoHideMenu: o.AutoHideMenu}
	data := ptr(w.id)

	typ := int32(0) // GTK_WINDOW_TOPLEVEL
	if o.Flyout != nil && o.Flyout.Tray == nil {
		typ = 1 // GTK_WINDOW_POPUP
	}
	w.win = gtkWindowNew(typ)
	gtkWindowSetTitle(w.win, cs(o.Title))
	if o.Panel {
		w.initLayer() // before anything realizes the window
	}
	if o.Center {
		gtkWindowSetPosition(w.win, 1) // GTK_WIN_POS_CENTER
	} else {
		gtkWindowMove(w.win, int32(o.X), int32(o.Y))
	}
	gtkWindowSetResizable(w.win, o.Resizable)
	gtkWindowSetDeletable(w.win, o.Closable)
	if w.undecorated() {
		gtkWindowSetDecorated(w.win, false)
	}
	if o.AlwaysOnTop {
		w.SetAlwaysOnTop(true)
	}
	if o.SkipTaskbar {
		gtkWindowSetSkipTaskbarHint(w.win, true)
	}
	if o.Opacity > 0 && o.Opacity < 1 {
		gtkWidgetSetOpacity(w.win, o.Opacity)
	}
	w.minW, w.minH = int32(o.MinSize.Width), int32(o.MinSize.Height)
	w.maxW, w.maxH = int32(o.MaxSize.Width), int32(o.MaxSize.Height)
	w.applyGeometry()
	if o.Transparent {
		if visual := gdkScreenGetRGBAVisual(gtkWidgetGetScreen(w.win)); visual != 0 {
			gtkWidgetSetVisual(w.win, visual)
		}
		gtkWidgetSetAppPaintable(w.win, true)
	}
	if p, ok := o.Parent.(*window); ok && p != nil {
		gtkWindowSetTransientFor(w.win, p.win)
		if o.Modal {
			gtkWindowSetModal(w.win, true)
		}
	}

	w.box = gtkBoxNew(1, 0) // vertical
	// GTK makes a window whose child has no natural size, as the web view,
	// 200x200 by nature, and one the user cannot resize never smaller than
	// its natural size.
	gtkWidgetSetSizeRequest(w.box, 1, 1)
	gtkContainerAdd(w.win, w.box)
	if w.hiddenTitleBar() {
		w.newControls()
	}
	if o.Surface {
		w.createSurface()
	} else {
		w.createWebView()
	}
	if w.controls != nil {
		gtkContainerAdd(w.controls.overlay, w.contentWidget())
	} else {
		gtkBoxPackStart(w.box, w.contentWidget(), true, true, 0)
	}
	w.accel = gtkAccelGroupNew()
	gtkWindowAddAccelGroup(w.win, w.accel)
	if b.appMenu != nil && w.hasAppMenu() {
		w.installMenu(b.appMenu)
	}
	// GTK sizes the window inside the window manager's frame, with the menu
	// bar.
	width, height := int32(o.Width), int32(o.Height)
	if o.UseContentSize {
		in := w.contentInsets()
		width, height = width+in.left+in.right, height+in.top+in.bottom
	} else {
		f := w.frameExtents()
		width, height = width-f.left-f.right, height-f.top-f.bottom
	}
	gtkWindowSetDefaultSize(w.win, max(width, 1), max(height, 1))
	if w.layer {
		w.sizeLayer(max(width, 1), max(height, 1))
	}

	connect(w.win, "delete-event", cbDeleteEvent, data)
	connect(w.win, "destroy", cbDestroy, data)
	connect(w.win, "focus-in-event", cbFocusIn, data)
	connect(w.win, "focus-out-event", cbFocusOut, data)
	connect(w.win, "configure-event", cbConfigure, data)
	connect(w.win, "window-state-event", cbWindowState, data)
	connect(w.win, "key-press-event", cbMenuKey, data)
	connect(w.win, "key-release-event", cbMenuKey, data)

	b.windows[w.id] = w
	if w.web != 0 {
		b.byWebView[w.web] = w
	}
	if o.Frameless && o.Flyout == nil && !w.layer && b.announceCSD != nil {
		// GTK asks a Wayland compositor that speaks
		// org_kde_kwin_server_decoration (KWin, COSMIC, Sway…) to decorate
		// every window it does not decorate itself, an undecorated one
		// too: say the window decorates itself, before it is mapped, so
		// that the compositor draws no title bar.
		gtkWidgetRealize(w.win)
		b.announceCSD(gtkWidgetGetWindow(w.win))
	}
	gtkWidgetShowAll(w.box)
	if o.Flyout != nil {
		w.initFlyout()
	}
	if o.Panel {
		w.initPanel()
	}
	if o.FullScreen {
		gtkWindowFullscreen(w.win)
	} else if o.Maximized {
		gtkWindowMaximize(w.win)
	}
	return w, nil
}

func (w *window) createWebView() {
	o := w.opts
	data := ptr(w.id)
	w.ucm = webkitUserContentManagerNew()
	if !w.b.webLanguages {
		// The first web view: WebKit just loaded.
		w.b.webLanguages = true
		w.b.applyWebLanguages()
	}
	for _, s := range o.Schemes {
		w.b.registerScheme(s)
	}
	w.web = webkitWebViewNewWithUserContentManager(w.ucm)
	webkitUserContentManagerRegisterHandler(w.ucm, cs("mygo"))
	connect(w.ucm, "script-message-received::mygo", cbScriptMessage, data)
	w.addUserScripts()

	settings := webkitWebViewGetSettings(w.web)
	webkitSettingsSetEnableDeveloperExtras(settings, o.DevTools)
	webkitSettingsSetAllowFileAccessFromFileURLs(settings, true)
	webkitSettingsSetJavascriptCanAccessClipboard(settings, true)
	if o.UserAgent != "" {
		webkitSettingsSetUserAgent(settings, cs(o.UserAgent))
	}
	if o.Zoom > 0 && o.Zoom != 1 {
		webkitWebViewSetZoomLevel(w.web, o.Zoom)
	}
	switch {
	case o.Transparent:
		webkitWebViewSetBackgroundColor(w.web, &gdkRGBA{})
	case o.BackgroundColor != nil:
		w.SetBackgroundColor(*o.BackgroundColor)
	}

	connect(w.web, "load-changed", cbLoadChanged, data)
	connect(w.web, "load-failed", cbLoadFailed, data)
	connect(w.web, "notify::title", cbTitle, data)
	connect(w.web, "decide-policy", cbDecidePolicy, data)
	connect(w.web, "create", cbCreate, data)
	connect(w.web, "close", cbClose, data)
	connect(w.web, "web-process-terminated", cbCrashed, data)
	connect(w.web, "button-press-event", cbButtonPress, data)
	if w.undecorated() {
		connect(w.web, "motion-notify-event", cbMotion, data)
	}
	connect(w.web, "drag-data-received", cbDragData, data)
	connect(w.web, "drag-drop", cbDragDrop, data)
	connect(w.web, "permission-request", cbPermission, data)
	hookDownloads()
}

func (w *window) applyGeometry() {
	g := gdkGeometry{MinWidth: w.minW, MinHeight: w.minH, MaxWidth: w.maxW, MaxHeight: w.maxH}
	mask := int32(0)
	if w.minW > 0 || w.minH > 0 {
		mask |= 1 << 1 // GDK_HINT_MIN_SIZE
	}
	if w.maxW > 0 || w.maxH > 0 {
		if g.MaxWidth == 0 {
			g.MaxWidth = 1 << 20
		}
		if g.MaxHeight == 0 {
			g.MaxHeight = 1 << 20
		}
		mask |= 1 << 2 // GDK_HINT_MAX_SIZE
	}
	gtkWindowSetGeometryHints(w.win, 0, &g, mask)
}

func (w *window) cleanup() {
	if w.closed {
		return
	}
	w.closeWebViews()
	w.closed = true
	delete(w.b.windows, w.id)
	w.b.flyouts = slices.DeleteFunc(w.b.flyouts, func(x *window) bool { return x == w })
	delete(w.b.byWebView, w.web)
	dropOwner(w.owner)
	if w.surface != nil {
		w.surface.destroy()
	}
	if w.ucm != 0 {
		webkitUserContentManagerUnregisterHandler(w.ucm, cs("mygo"))
		webkitUserContentManagerRemoveAllScripts(w.ucm)
		gObjectUnref(w.ucm)
	}
	if w.press.event != 0 {
		gdkEventFree(w.press.event)
		w.press.event = 0
	}
	w.releaseCursor()
}

func (w *window) Handle() uintptr        { return w.win }
func (w *window) WebViewHandle() uintptr { return w.web }

func (w *window) SetTitle(title string) { gtkWindowSetTitle(w.win, cs(title)) }
func (w *window) Title() string         { return goStr(gtkWindowGetTitle(w.win)) }

// The core marks the title of an edited document; GTK shows no file.
func (w *window) SetRepresentedFile(string) {}
func (w *window) SetDocumentEdited(bool)    {}

// extents are the widths of the edges around a rectangle.
type extents struct{ left, top, right, bottom int32 }

// grow returns r with the extents around it.
func (e extents) grow(r platform.Rect) platform.Rect {
	return platform.Rect{
		X: r.X - int(e.left), Y: r.Y - int(e.top),
		Width: r.Width + int(e.left+e.right), Height: r.Height + int(e.top+e.bottom),
	}
}

// shrink returns r without the extents around its inside.
func (e extents) shrink(r platform.Rect) platform.Rect {
	return platform.Rect{
		X: r.X + int(e.left), Y: r.Y + int(e.top),
		Width: r.Width - int(e.left+e.right), Height: r.Height - int(e.top+e.bottom),
	}
}

// frameExtents returns the extents of the frame a reparenting window
// manager puts around the window on X11, title bar and borders: GTK sizes
// the window inside it and positions the frame (NORTH_WEST gravity). They
// are known once the window manager framed the window that shows, which
// its first own (synthetic) configure report tells. Until then the window
// expects the frame it had last, or the one seen last around a decorated
// window in its normal state that the user can resize or not as this one,
// or none. Wayland tells no positions, and its compositors' decorations
// have no say in GTK's sizes.
func (w *window) frameExtents() extents {
	if !w.b.onX11 {
		return extents{}
	}
	resizable := 0
	if gtkWindowGetResizable(w.win) {
		resizable = 1
	}
	if !w.framed || !gtkWidgetGetMapped(w.win) {
		switch {
		case w.seenFrame:
			return w.frame
		case w.undecorated():
			return extents{}
		}
		return w.b.frames[resizable]
	}
	gdkWin := gtkWidgetGetWindow(w.win)
	var frame gdkRectangle
	gdkWindowGetFrameExtents(gdkWin, &frame)
	var x, y, width, height int32
	gdkWindowGetOrigin(gdkWin, &x, &y)
	gdkWindowGetGeometry(gdkWin, nil, nil, &width, &height) // from the server, as the frame
	f := extents{x - frame.X, y - frame.Y, frame.X + frame.Width - x - width, frame.Y + frame.Height - y - height}
	w.frame, w.seenFrame = f, true
	const special = stateMaximized | stateFullscreen | stateTiled | stateIconified
	if !w.undecorated() && w.state&special == 0 {
		w.b.frames[resizable] = f
	}
	return f
}

// contentInsets returns where the content is inside the window GTK sizes:
// below the menu bar, which GTK gives its natural height. That holds
// before GTK lays out a menu bar that just showed or went, unlike where
// it has the content.
func (w *window) contentInsets() extents {
	if w.menubar == 0 || !gtkWidgetGetVisible(w.menubar) {
		return extents{}
	}
	var minimum, natural int32
	gtkWidgetGetPreferredHeight(w.menubar, &minimum, &natural)
	return extents{top: natural}
}

// SetBounds asks for the bounds of the window manager's frame: GTK moves
// the frame and sizes the window inside it.
func (w *window) SetBounds(r platform.Rect) {
	if w.layer {
		w.setLayerBounds(r)
		return
	}
	w.setBounds(r, false)
}

func (w *window) SetContentBounds(r platform.Rect) {
	w.setBounds(w.frameExtents().grow(w.contentInsets().grow(r)), true)
}

// setBounds asks for the bounds of the frame, for the size inside it when
// inside is set: a frame other than expected keeps that size (see
// cbConfigure).
func (w *window) setBounds(r platform.Rect, inside bool) {
	f := w.frameExtents()
	width, height := w.constrain(int32(r.Width)-f.left-f.right, int32(r.Height)-f.top-f.bottom)
	r.Width, r.Height = int(width+f.left+f.right), int(height+f.top+f.bottom)
	if w.requested == nil {
		gtkWindowGetSize(w.win, &w.previous[0], &w.previous[1])
	}
	w.requested, w.requestFrame, w.requestInside = &r, f, inside
	if w.resizeIdle != 0 {
		w.activationResize = &r // the idle callback asks for it
		return
	}
	if !w.b.onX11 && w.surface != nil && w.surface.rendered && w.surface.drawsGL() && w.state&stateFocused == 0 {
		if w.activationResize == nil {
			gtkWindowGetSize(w.win, &w.activationPrevious[0], &w.activationPrevious[1])
		}
		w.activationResize = &r
	}
	// GTK keeps a window the user cannot resize at least as large as its
	// default size.
	gtkWindowSetDefaultSize(w.win, width, height)
	gtkWindowMove(w.win, int32(r.X), int32(r.Y))
	gtkWindowResize(w.win, width, height)
}

// constrain returns the size GTK gives the window when asked for one:
// within its minimum and maximum sizes, and no smaller than its content, or
// than the content's natural size when the user cannot resize the window.
func (w *window) constrain(width, height int32) (int32, int32) {
	var minW, natW, minH, natH int32
	gtkWidgetGetPreferredWidth(w.box, &minW, &natW)
	gtkWidgetGetPreferredHeight(w.box, &minH, &natH)
	if !gtkWindowGetResizable(w.win) {
		minW, minH = natW, natH
	}
	if w.maxW > 0 {
		width = min(width, w.maxW)
	}
	if w.maxH > 0 {
		height = min(height, w.maxH)
	}
	return max(width, w.minW, minW), max(height, w.minH, minH)
}

// Bounds returns the bounds of the window manager's frame, which are the
// window's own without one.
func (w *window) Bounds() platform.Rect {
	if w.layer {
		return w.layerBounds()
	}
	if w.requested != nil {
		return *w.requested
	}
	if w.framed && gtkWidgetGetMapped(w.win) {
		var r gdkRectangle
		gdkWindowGetFrameExtents(gtkWidgetGetWindow(w.win), &r)
		return platform.Rect{X: int(r.X), Y: int(r.Y), Width: int(r.Width), Height: int(r.Height)}
	}
	var x, y, width, height int32
	gtkWindowGetPosition(w.win, &x, &y) // the frame's
	gtkWindowGetSize(w.win, &width, &height)
	f := w.frameExtents()
	return platform.Rect{X: int(x), Y: int(y), Width: int(width + f.left + f.right), Height: int(height + f.top + f.bottom)}
}

// size returns the size GTK gives the window, or was asked to give it.
func (w *window) size() (width, height int32) {
	if r := w.requested; r != nil {
		f := w.requestFrame
		return int32(r.Width) - f.left - f.right, int32(r.Height) - f.top - f.bottom
	}
	gtkWindowGetSize(w.win, &width, &height)
	return width, height
}

// ContentBounds returns where the content is: on X11, where X has the
// window inside the window manager's frame, below the menu bar.
func (w *window) ContentBounds() platform.Rect {
	var r platform.Rect
	if w.requested == nil && w.b.onX11 && gtkWidgetGetMapped(w.win) {
		var x, y int32
		gdkWindowGetOrigin(gtkWidgetGetWindow(w.win), &x, &y)
		width, height := w.size()
		r = platform.Rect{X: int(x), Y: int(y), Width: int(width), Height: int(height)}
	} else {
		f := w.requestFrame
		if w.requested == nil {
			f = w.frameExtents()
		}
		r = f.shrink(w.Bounds())
	}
	return w.contentInsets().shrink(r)
}

func (w *window) SetMinimumSize(s platform.Size) {
	w.minW, w.minH = int32(s.Width), int32(s.Height)
	w.applyGeometry()
}

func (w *window) SetMaximumSize(s platform.Size) {
	w.maxW, w.maxH = int32(s.Width), int32(s.Height)
	w.applyGeometry()
}

func (w *window) SetResizable(v bool) {
	if !v {
		// Keep the size it has or was given (see SetBounds).
		width, height := w.size()
		gtkWindowSetDefaultSize(w.win, width, height)
	}
	gtkWindowSetResizable(w.win, v)
	if w.controls != nil {
		w.layoutControls() // with a maximize button or without
	}
}

func (w *window) IsResizable() bool   { return gtkWindowGetResizable(w.win) }
func (w *window) SetMovable(bool)     {}
func (w *window) IsMovable() bool     { return true }
func (w *window) SetMinimizable(bool) {}
func (w *window) IsMinimizable() bool { return !w.opts.Panel }
func (w *window) SetMaximizable(bool) {}
func (w *window) IsMaximizable() bool { return gtkWindowGetResizable(w.win) && !w.opts.Panel }
func (w *window) SetClosable(v bool) {
	gtkWindowSetDeletable(w.win, v)
	if w.controls != nil {
		w.layoutControls() // with a close button or without
	}
}

func (w *window) IsClosable() bool { return gtkWindowGetDeletable(w.win) }
func (w *window) SetAlwaysOnTop(v bool) {
	w.keepAbove = v
	gtkWindowSetKeepAbove(w.win, v)
}

// IsAlwaysOnTop reports the requested state: only some window managers
// reflect it in the window state.
func (w *window) IsAlwaysOnTop() bool { return w.keepAbove || w.state&stateAbove != 0 }

func (w *window) Show() {
	if w.opts.Flyout != nil {
		w.showFlyout(w.flyout.Focusable)
		return
	}
	if w.layer {
		w.layerKeyboard(true)
	}
	w.willShow()
	gtkWidgetShow(w.win)
	gtkWindowPresent(w.win)
}

func (w *window) ShowInactive() {
	if w.opts.Flyout != nil {
		w.showFlyout(false)
		return
	}
	if w.layer {
		w.layerKeyboard(false)
	}
	w.willShow()
	gtkWidgetShow(w.win)
}

// willShow makes Bounds report where GTK asks for a window about to show
// until the window manager put it there. Until then, X has the window where
// GTK created it or where it was, and a reparenting window manager has its
// frame where it created that.
func (w *window) willShow() {
	if gtkWidgetGetVisible(w.win) {
		return
	}
	w.framed = false // until the window manager frames it again
	if w.requested == nil {
		b := w.Bounds() // what GTK asks for, while the window does not show
		// The size it was made with, or had, unless it was made with
		// that of its frame.
		inside := w.opts.UseContentSize || w.seenFrame
		w.requested, w.requestFrame, w.requestInside = &b, w.frameExtents(), inside
		w.previous[0], w.previous[1] = w.size()
	}
	w.placing = w.b.windowManager(w.win)
}

func (w *window) Hide() {
	if w.opts.Flyout != nil {
		w.hideFlyout()
		return
	}
	gtkWidgetHide(w.win)
}

func (w *window) IsVisible() bool { return gtkWidgetGetVisible(w.win) }

func (w *window) Focus() {
	if w.host != nil {
		if !w.closed {
			gtkWidgetGrabFocus(w.web)
		}
		return
	}
	if w.opts.Flyout != nil {
		// A popup takes the keyboard only by its grab, as it shows.
		w.showFlyout(w.flyout.Focusable)
		return
	}
	w.b.dismissFlyouts(w)
	if w.layer {
		w.layerKeyboard(true)
	}
	w.willShow() // presenting shows a hidden window
	gtkWindowPresent(w.win)
}

func (w *window) Blur()             {}
func (w *window) IsFocused() bool   { return gtkWindowIsActive(w.win) }
func (w *window) Minimize()         { gtkWindowIconify(w.win) }
func (w *window) IsMinimized() bool { return w.state&stateIconified != 0 }
func (w *window) Maximize()         { gtkWindowMaximize(w.win) }
func (w *window) Unmaximize()       { gtkWindowUnmaximize(w.win) }
func (w *window) IsMaximized() bool { return gtkWindowIsMaximized(w.win) }
func (w *window) Restore()          { gtkWindowDeiconify(w.win) }

func (w *window) SetFullScreen(v bool) {
	if v {
		gtkWindowFullscreen(w.win)
	} else {
		gtkWindowUnfullscreen(w.win)
	}
}

func (w *window) IsFullScreen() bool { return w.state&stateFullscreen != 0 }

func (w *window) SetProgressBar(state string, value float64) {
	l := &w.b.launcher
	l.showing = state != ""
	l.progress = value
	if state == "indeterminate" {
		l.progress = 1
	}
	w.b.updateLauncherEntry()
}

func (w *window) FlashFrame(flash bool) { gtkWindowSetUrgencyHint(w.win, flash) }
func (w *window) SetSkipTaskbar(v bool) { gtkWindowSetSkipTaskbarHint(w.win, v) }

func (w *window) SetVisibleOnAllWorkspaces(v bool) {
	if v {
		gtkWindowStick(w.win)
	} else {
		gtkWindowUnstick(w.win)
	}
}

func (w *window) SetIcon(png []byte) error {
	if png == nil {
		gtkWindowSetIcon(w.win, 0) // back to the default icon
		return nil
	}
	pix, err := pixbufFromPNG(png)
	if err != nil {
		return err
	}
	defer gObjectUnref(pix)
	gtkWindowSetIcon(w.win, pix)
	return nil
}

func (w *window) DroppedFiles() []string {
	paths := w.dropped
	w.dropped = nil
	return paths
}

func (w *window) Center() {
	if w.layer {
		w.centerLayer()
		return
	}
	b := w.Bounds()
	for _, d := range (screen{}).Displays() {
		if d.Primary {
			area := d.WorkArea
			x, y := area.X+(area.Width-b.Width)/2, area.Y+(area.Height-b.Height)/2
			if w.requested != nil {
				// Bounds that GTK has not confirmed yet move with it.
				w.requested.X, w.requested.Y = x, y
			}
			gtkWindowMove(w.win, int32(x), int32(y))
			return
		}
	}
}

func (w *window) SetBackgroundColor(c platform.Color) {
	if w.web != 0 {
		webkitWebViewSetBackgroundColor(w.web, &gdkRGBA{float64(c.R) / 255, float64(c.G) / 255, float64(c.B) / 255, float64(c.A) / 255})
	}
}

func (w *window) SetOpacity(v float64)      { gtkWidgetSetOpacity(w.win, v) }
func (w *window) Opacity() float64          { return gtkWidgetGetOpacity(w.win) }
func (w *window) SetHasShadow(bool)         {}
func (w *window) HasShadow() bool           { return true }
func (w *window) SetIgnoreMouseEvents(bool) {}
func (w *window) SetContentProtection(bool) {}
func (w *window) SetVibrancy(string)        {}

// StartDrag begins moving the window with the last button press. On an
// undecorated window the compositor takes the pointer once the move
// starts, so the move waits until the pointer moves, keeping a second
// press free to reach the page as a double click. Decorated windows
// observe no motion on the page, so they move at once.
func (w *window) StartDrag() {
	if !w.press.hasPressed {
		return
	}
	if !w.undecorated() {
		gtkWindowBeginMoveDrag(w.win, w.press.button, int32(w.press.rootX), int32(w.press.rootY), w.press.time)
		return
	}
	w.dragPending = true
}

// checkDragStart starts a pending window drag once the pointer moves past
// the drag threshold with the button still down.
func (w *window) checkDragStart(event ptr) {
	if !w.dragPending {
		return
	}
	// GdkEventMotion: state 48, x_root 64, y_root 72.
	if field[uint32](event, 48)&(1<<8) == 0 {
		// The button went up without the pointer moving.
		w.dragPending = false
		return
	}
	const dragThreshold = 8 // pixels, as GTK's
	dx, dy := field[float64](event, 64)-w.press.rootX, field[float64](event, 72)-w.press.rootY
	if dx*dx+dy*dy < dragThreshold*dragThreshold {
		return
	}
	w.dragPending = false
	gtkWindowBeginMoveDrag(w.win, w.press.button, int32(w.press.rootX), int32(w.press.rootY), w.press.time)
}

// GTK gives windows without decorations no resize edges, so the outer
// pixels of the page resize frameless windows, as in Electron.
const (
	resizeInset  = 5  // pixels of the page along the edges
	resizeCorner = 16 // pixels along the edges from a corner that resize it
)

// resizeEdges are the GdkWindowEdge values, with their cursors and the
// states in which the window manager lets them resize the window.
var resizeEdges = [8]struct {
	cursor string
	states uint32
}{
	{"nw-resize", stateTopResizable | stateLeftResizable},
	{"n-resize", stateTopResizable},
	{"ne-resize", stateTopResizable | stateRightResizable},
	{"w-resize", stateLeftResizable},
	{"e-resize", stateRightResizable},
	{"sw-resize", stateBottomResizable | stateLeftResizable},
	{"s-resize", stateBottomResizable},
	{"se-resize", stateBottomResizable | stateRightResizable},
}

// resizeEdge returns the GdkWindowEdge a pointer event on the page (a
// GdkEventMotion or GdkEventButton) would resize, or -1.
func (w *window) resizeEdge(event ptr) int32 {
	// GdkEventMotion and GdkEventButton: window 8, x 24, y 32.
	if !w.undecorated() || w.state&(stateMaximized|stateFullscreen) != 0 ||
		field[ptr](event, 8) != w.contentWindow() || !gtkWindowGetResizable(w.win) {
		return -1
	}
	x, y := field[float64](event, 24), field[float64](event, 32)
	var page gdkRectangle
	gtkWidgetGetAllocation(w.contentWidget(), &page)
	width, height := float64(page.Width), float64(page.Height)
	atTop := page.Y == 0 // no menu bar above the page
	top, bottom := atTop && y < resizeInset, y >= height-resizeInset
	left, right := x < resizeInset, x >= width-resizeInset
	if !top && !bottom && !left && !right {
		return -1
	}
	// Near a corner, the edges resize the corner.
	top = top || (left || right) && atTop && y < resizeCorner
	bottom = bottom || (left || right) && y >= height-resizeCorner
	left = left || (top || bottom) && x < resizeCorner
	right = right || (top || bottom) && x >= width-resizeCorner
	var edge int32
	switch {
	case top && left:
		edge = 0
	case top && right:
		edge = 2
	case bottom && left:
		edge = 5
	case bottom && right:
		edge = 7
	case top:
		edge = 1
	case bottom:
		edge = 6
	case left:
		edge = 3
	default:
		edge = 4
	}
	// Tiled windows resize at the edges the window manager allows, as
	// with GTK's own decorations.
	const resizable = stateTopResizable | stateRightResizable | stateBottomResizable | stateLeftResizable
	if need := resizeEdges[edge].states; w.state&resizable != 0 && w.state&need != need ||
		w.state&resizable == 0 && w.state&stateTiled != 0 {
		return -1
	}
	return edge
}

// showResizeCursor shows the cursor of a resize edge over the page, or
// the page's own cursor again for -1.
func (w *window) showResizeCursor(edge int32) {
	c := &w.cursor
	if edge < 0 && !c.on {
		return
	}
	page := w.contentWindow()
	current := gdkWindowGetCursor(page)
	if edge < 0 {
		if c.on && current == c.shown {
			gdkWindowSetCursor(page, c.saved)
		}
		w.releaseCursor()
		return
	}
	if !c.on || current != c.shown {
		// The pointer comes onto an edge, or WebKit changed the cursor.
		w.releaseCursor()
		if current != 0 {
			gObjectRef(current)
		}
		c.on, c.edge, c.saved = true, -1, current
	}
	if c.edge != edge {
		cursor := gdkCursorNewFromName(gdkWindowGetDisplay(page), cs(resizeEdges[edge].cursor))
		gdkWindowSetCursor(page, cursor)
		if cursor != 0 {
			gObjectUnref(cursor) // the GdkWindow holds it
		}
		c.edge, c.shown = edge, cursor
	}
}

// releaseCursor forgets the page's cursor kept to restore.
func (w *window) releaseCursor() {
	if w.cursor.saved != 0 {
		gObjectUnref(w.cursor.saved)
	}
	w.cursor.on, w.cursor.edge, w.cursor.shown, w.cursor.saved = false, -1, 0, 0
}

// hiddenTitleBar reports a TitleBarStyle that hides the title bar (not a
// frameless window's, which has no buttons either).
func (w *window) hiddenTitleBar() bool {
	o := w.opts
	return !o.Frameless && (o.TitleBarStyle == "hidden" || o.TitleBarStyle == "hiddenInset")
}

// undecorated reports a window without the window manager's decorations,
// whose page's outer pixels resize it.
func (w *window) undecorated() bool { return w.opts.Frameless || w.hiddenTitleBar() }

func (w *window) TitleBarDoubleClicked() {
	if w.IsMaximized() {
		w.Unmaximize()
	} else {
		w.Maximize()
	}
}

func (w *window) Close() {
	if w.host != nil {
		w.closeWebView()
		return
	}
	if !w.closed {
		if w.opts.Flyout != nil {
			w.ungrab()
		}
		gtkWidgetDestroy(w.win)
	}
}

func (w *window) LoadURL(url string) {
	w.programmatic = true
	webkitWebViewLoadURI(w.web, cs(url))
}

func (w *window) LoadHTML(html, baseURL string) {
	w.programmatic = true
	webkitWebViewLoadHTML(w.web, cs(html), optCS(baseURL))
}

func (w *window) LoadFile(path, _ string) {
	f := gFileNewForPath(cs(path))
	uri := takeStr(gFileGetURI(f))
	gObjectUnref(f)
	w.LoadURL(uri)
}

func (w *window) Reload(ignoreCache bool) {
	if ignoreCache {
		webkitWebViewReloadBypassCache(w.web)
	} else {
		webkitWebViewReload(w.web)
	}
}

func (w *window) StopLoading()       { webkitWebViewStopLoading(w.web) }
func (w *window) GoBack()            { webkitWebViewGoBack(w.web) }
func (w *window) GoForward()         { webkitWebViewGoForward(w.web) }
func (w *window) CanGoBack() bool    { return webkitWebViewCanGoBack(w.web) }
func (w *window) CanGoForward() bool { return webkitWebViewCanGoForward(w.web) }
func (w *window) URL() string        { return goStr(webkitWebViewGetURI(w.web)) }
func (w *window) IsLoading() bool    { return webkitWebViewIsLoading(w.web) }

// addUserScripts adds the user scripts of the options to the content
// manager, and the title bar's after them.
func (w *window) addUserScripts() {
	add := func(src string, frames, when int32) {
		script := webkitUserScriptNew(cs(src), frames, when, 0, 0)
		webkitUserContentManagerAddScript(w.ucm, script)
		webkitUserScriptUnref(script)
	}
	for _, s := range w.opts.UserScripts {
		frames, when := int32(1), int32(0) // top frame, document start
		if s.AllFrames {
			frames = 0
		}
		if s.AtDocumentEnd {
			when = 1
		}
		add(s.Source, frames, when)
	}
	if w.controls != nil {
		// After the bridge, which it tells; top frame, document start.
		add(w.titleBarScript(), 1, 0)
	}
}

// SetUserScripts replaces the user scripts, all of them, as the content
// manager removes scripts one at a time only from WebKitGTK 2.32 on.
func (w *window) SetUserScripts(scripts []platform.UserScript) {
	if w.ucm == 0 || w.closed {
		return
	}
	w.opts.UserScripts = scripts
	webkitUserContentManagerRemoveAllScripts(w.ucm)
	w.addUserScripts()
}

func (w *window) Eval(js string) {
	if webkitWebViewEvaluateJavascript != nil {
		// With its length, the script needs no NUL terminated copy.
		webkitWebViewEvaluateJavascript(w.web, unsafe.StringData(js), len(js), nil, nil, 0, 0, 0)
	} else if webkitWebViewRunJavascript != nil {
		webkitWebViewRunJavascript(w.web, cs(js), 0, 0, 0)
	}
}

func (w *window) CallAsyncFunction(body string, cb func(string, error)) {
	if webkitWebViewCallAsyncJavascriptFunction == nil {
		cb("", errors.New("mygo: Eval requires WebKitGTK 2.40 or later"))
		return
	}
	id := pending.add(func(source, res ptr) {
		var gerr ptr
		v := webkitWebViewCallAsyncJavascriptFunctionFinish(source, res, &gerr)
		if gerr != 0 {
			cb("", gErr(gerr))
			return
		}
		s := takeStr(jscValueToString(v))
		gObjectUnref(v)
		cb(s, nil)
	})
	webkitWebViewCallAsyncJavascriptFunction(w.web, unsafe.StringData(body), len(body), 0, nil, nil, 0, cbAsyncReady, id)
}

func (w *window) SetZoom(f float64) { webkitWebViewSetZoomLevel(w.web, f) }
func (w *window) Zoom() float64     { return webkitWebViewGetZoomLevel(w.web) }

func (w *window) SetUserAgent(ua string) {
	webkitSettingsSetUserAgent(webkitWebViewGetSettings(w.web), optCS(ua))
}

func (w *window) UserAgent() string {
	return goStr(webkitSettingsGetUserAgent(webkitWebViewGetSettings(w.web)))
}

func (w *window) OpenDevTools() {
	if w.opts.DevTools {
		webkitWebInspectorShow(webkitWebViewGetInspector(w.web))
	}
}

func (w *window) CloseDevTools() { webkitWebInspectorClose(webkitWebViewGetInspector(w.web)) }

func (w *window) IsDevToolsOpened() bool {
	return webkitWebInspectorGetWebView(webkitWebViewGetInspector(w.web)) != 0
}

func (w *window) CapturePage(cb func([]byte, error)) {
	id := pending.add(func(source, res ptr) {
		var gerr ptr
		surface := webkitWebViewGetSnapshotFinish(source, res, &gerr)
		if gerr != 0 {
			cb(nil, gErr(gerr))
			return
		}
		png, err := surfaceToPNG(surface)
		cairoSurfaceDestroy(surface)
		cb(png, err)
	})
	webkitWebViewGetSnapshot(w.web, 0, 0, 0, cbAsyncReady, id)
}

// decideResponse makes attachments, and what WebKit cannot show,
// downloads. WebKit cannot download what a custom scheme serves, so the app
// serves it again into a download.
func (w *window) decideResponse(decision ptr) bool {
	resp := webkitResponsePolicyDecisionGetResponse(decision)
	uri := goStr(webkitURIResponseGetURI(resp))
	disposition := ""
	if h := webkitURIResponseGetHTTPHeaders(resp); h != 0 {
		disposition = goStr(soupMessageHeadersGetOne(h, cs("Content-Disposition")))
	}
	attachment := strings.HasPrefix(strings.ToLower(strings.TrimSpace(disposition)), "attachment")
	if !attachment && webkitResponsePolicyDecisionIsMIMETypeSupported(decision) {
		return false
	}
	scheme, _, _ := strings.Cut(uri, ":")
	if w.b.schemes[strings.ToLower(scheme)] {
		webkitPolicyDecisionIgnore(decision)
		w.h.SchemeDownload(uri)
		return true
	}
	webkitPolicyDecisionDownload(decision)
	return true
}

// printJobs holds the PrintToPDF operations running.
var printJobs = map[ptr]*printJob{}

type printJob struct {
	err  error // from "failed", which "finished" follows
	done func(err error)
}

// PrintToPDF prints to the "Print to File" printer of GTK, which writes
// the PDF to a temporary file.
func (w *window) PrintToPDF(o platform.PDFOptions, cb func([]byte, error)) {
	// WebKitGTK crashes printing a web view that never loaded anything,
	// whose URI is still NULL.
	if webkitWebViewGetURI(w.web) == 0 {
		cb(nil, errors.New("mygo: printing to PDF: the page has loaded nothing"))
		return
	}
	f, err := os.CreateTemp("", "mygo-*.pdf")
	if err != nil {
		cb(nil, err)
		return
	}
	path := f.Name()
	f.Close()
	settings := gtkPrintSettingsNew()
	for k, v := range map[string]string{
		"printer":            "Print to File",
		"output-file-format": "pdf",
		"output-uri":         (&url.URL{Scheme: "file", Path: path}).String(),
	} {
		gtkPrintSettingsSet(settings, cs(k), cs(v))
	}
	setup := gtkPageSetupNew()
	const unitInch = 2 // GTK_UNIT_INCH
	paper := gtkPaperSizeNewCustom(cs("mygo"), cs("MyGo"), o.PageWidth, o.PageHeight, unitInch)
	gtkPageSetupSetPaperSize(setup, paper)
	gtkPaperSizeFree(paper)
	if o.Landscape {
		gtkPageSetupSetOrientation(setup, 1) // GTK_PAGE_ORIENTATION_LANDSCAPE
	}
	gtkPageSetupSetTopMargin(setup, o.MarginTop, unitInch)
	gtkPageSetupSetRightMargin(setup, o.MarginRight, unitInch)
	gtkPageSetupSetBottomMargin(setup, o.MarginBottom, unitInch)
	gtkPageSetupSetLeftMargin(setup, o.MarginLeft, unitInch)

	web := webkitWebViewGetSettings(w.web)
	backgrounds := webkitSettingsGetPrintBackgrounds(web)
	webkitSettingsSetPrintBackgrounds(web, o.Background)
	op := webkitPrintOperationNew(w.web)
	webkitPrintOperationSetPrintSettings(op, settings)
	webkitPrintOperationSetPageSetup(op, setup)
	gObjectUnref(settings)
	gObjectUnref(setup)
	printJobs[op] = &printJob{done: func(err error) {
		delete(printJobs, op)
		webkitSettingsSetPrintBackgrounds(web, backgrounds)
		gObjectUnref(op)
		data, rerr := os.ReadFile(path)
		os.Remove(path)
		if err == nil && (rerr != nil || len(data) == 0) {
			err = errors.New("mygo: printing to PDF produced no file")
		}
		if err != nil {
			data = nil
		}
		cb(data, err)
	}}
	connect(op, "finished", cbPrintFinished, 0)
	connect(op, "failed", cbPrintFailed, 0)
	webkitPrintOperationPrint(op)
}

func (w *window) Print() {
	op := webkitPrintOperationNew(w.web)
	webkitPrintOperationRunDialog(op, w.win)
	gObjectUnref(op)
}

// pending holds callbacks of asynchronous GIO operations by id.
var pending = &asyncCalls{m: map[ptr]func(source, res ptr){}}

type asyncCalls struct {
	mu   sync.Mutex
	next ptr
	m    map[ptr]func(source, res ptr)
}

func (a *asyncCalls) add(fn func(source, res ptr)) ptr {
	a.mu.Lock()
	defer a.mu.Unlock()
	a.next++
	a.m[a.next] = fn
	return a.next
}

func (a *asyncCalls) take(id ptr) func(source, res ptr) {
	a.mu.Lock()
	defer a.mu.Unlock()
	fn := a.m[id]
	delete(a.m, id)
	return fn
}

// surfaceToPNG encodes a cairo surface as PNG through a write callback.
var pngBuffers = map[ptr]*bytes.Buffer{}

func surfaceToPNG(surface ptr) ([]byte, error) {
	key := ptr(len(pngBuffers) + 1)
	for pngBuffers[key] != nil {
		key++
	}
	buf := &bytes.Buffer{}
	pngBuffers[key] = buf
	defer delete(pngBuffers, key)
	if status := cairoSurfaceWriteToPNGStream(surface, cbPNGWrite, key); status != 0 {
		return nil, errors.New("mygo: cannot encode snapshot")
	}
	return buf.Bytes(), nil
}

// Signal handlers. data carries the window id.
var (
	cbDeleteEvent, cbDestroy, cbFocusIn, cbFocusOut, cbConfigure, cbWindowState ptr
	cbScriptMessage, cbLoadChanged, cbLoadFailed, cbTitle, cbDecidePolicy       ptr
	cbCreate, cbClose, cbCrashed, cbButtonPress, cbAsyncReady, cbPNGWrite       ptr
	cbDragData, cbDragDrop, cbPrintFinished, cbPrintFailed, cbPermission        ptr
	cbMotion, cbControlsAllocated, cbDecorationLayout, cbActivationResize       ptr
)

func field[T any](p ptr, offset uintptr) T {
	return *(*T)(unsafe.Add(*(*unsafe.Pointer)(unsafe.Pointer(&p)), offset))
}

func initWindowCallbacks() {
	b := func() *Backend { return theBackend }
	cbDeleteEvent = purego.NewCallback(func(widget, event, data ptr) bool {
		if w := b().window(data); w != nil {
			return !w.h.ShouldClose()
		}
		return false
	})
	cbDestroy = purego.NewCallback(func(widget, data ptr) {
		if w := b().window(data); w != nil {
			w.cleanup()
			w.h.Closed()
		}
	})
	cbFocusIn = purego.NewCallback(func(widget, event, data ptr) bool {
		if w := b().window(data); w != nil {
			w.h.Focused()
		}
		return false
	})
	cbFocusOut = purego.NewCallback(func(widget, event, data ptr) bool {
		if w := b().window(data); w != nil {
			w.altAlone = false
			w.h.Blurred()
			if w.trayFlyout() && w.flyout.Focusable && !w.grabbingChild() {
				w.dismissFlyout(false)
			}
		}
		return false
	})
	// A window drawing with GL on Wayland got focus: the configure event of
	// the activation can carry the size of the last buffer it drew, from
	// before a resize GTK already took, and GTK went back to it. Ask for the
	// latest bounds again then, unless the window manager sized the window.
	cbActivationResize = purego.NewCallback(func(data ptr) int32 {
		w := b().window(data)
		if w == nil {
			return 0
		}
		r := w.activationResize
		w.activationResize, w.resizeIdle = nil, 0
		var width, height int32
		gtkWindowGetSize(w.win, &width, &height)
		if w.state&(stateMaximized|stateFullscreen|stateTiled) == 0 &&
			(w.requested != nil || [2]int32{width, height} == w.activationPrevious) {
			// GTK skips a request for the size it asked for last: the first
			// check takes the configure, the second asks for the size the
			// window has, so that the bounds are asked for again. Handlers
			// of the new size may close the window.
			gtkWindowResize(w.win, width, height)
			gtkContainerCheckResize(w.win)
			if !w.closed {
				gtkContainerCheckResize(w.win)
			}
			if !w.closed {
				w.SetBounds(*r)
			}
		} else {
			w.requested = nil
		}
		return 0
	})
	cbConfigure = purego.NewCallback(func(widget, event, data ptr) bool {
		w := b().window(data)
		if w == nil {
			return false
		}
		// GdkEventConfigure: x, y, width, height at offsets 20..32.
		x, y := field[int32](event, 20), field[int32](event, 24)
		width, height := field[int32](event, 28), field[int32](event, 32)
		moved := x != w.x || y != w.y
		resized := width != w.w || height != w.hgt
		w.x, w.y, w.w, w.hgt = x, y, width, height
		// The window manager's own report (GdkEventConfigure's send_event,
		// at 16) comes once it framed and moved the window.
		synthetic := field[int8](event, 16) != 0
		if synthetic && w.b.onX11 {
			w.framed = true
		}
		if r := w.requested; r != nil && w.resizeIdle == 0 {
			// Without the decorations GTK draws, as requested.
			var cw, ch int32
			gtkWindowGetSize(w.win, &cw, &ch)
			f := w.frameExtents()
			if old := w.requestFrame; f != old {
				// The window manager framed the window otherwise than
				// expected as it placed it. GTK positions the frame, so
				// the bounds grow with the frame around the size asked
				// for inside it, or GTK is asked for the size inside the
				// frame again, which is done once GTK has it there.
				if w.requestInside {
					r.Width += int(f.left + f.right - old.left - old.right)
					r.Height += int(f.top + f.bottom - old.top - old.bottom)
					w.requestFrame = f
				} else {
					rw, rh := w.constrain(int32(r.Width)-f.left-f.right, int32(r.Height)-f.top-f.bottom)
					r.Width, r.Height = int(rw+f.left+f.right), int(rh+f.top+f.bottom)
					w.requestFrame, w.previous, w.placing = f, [2]int32{cw, ch}, false
					gtkWindowSetDefaultSize(w.win, rw, rh)
					gtkWindowResize(w.win, rw, rh)
				}
			}
			rw, rh := w.size()
			done := false
			switch {
			case cw != rw || ch != rh:
				if cw == w.previous[0] && ch == w.previous[1] {
					// A report from before the window manager took the
					// request, as openbox sends when the size hints change.
					// GTK would ask for this size again from it (and keeps a
					// window the user cannot resize as large), unless asked
					// for the new one again.
					gtkWindowResize(w.win, rw, rh)
				} else {
					done = true // the window manager, or the user, chose it
				}
			case synthetic || !w.b.onX11:
				done = true
			case !w.placing:
				// X11 reports where the window is, in the frame of the
				// window manager: it is done once that is where it was
				// asked to be.
				var cx, cy int32
				gtkWindowGetPosition(w.win, &cx, &cy)
				done = cx == int32(r.X) && cy == int32(r.Y)
			}
			if done {
				w.requested, w.placing = nil, false
			}
		}
		if resized {
			w.h.Resized()
		}
		if moved {
			w.h.Moved()
		}
		return false
	})
	cbWindowState = purego.NewCallback(func(widget, event, data ptr) bool {
		w := b().window(data)
		if w == nil {
			return false
		}
		// GdkEventWindowState: changed_mask at 20, new_window_state at 24.
		changed, state := field[uint32](event, 20), field[uint32](event, 24)
		w.state = state
		if changed&stateFocused != 0 && state&stateFocused != 0 && w.activationResize != nil && w.resizeIdle == 0 {
			// G_PRIORITY_DEFAULT_IDLE: after GTK took the configure event
			// of the activation.
			w.resizeIdle = gIdleAddFull(200, cbActivationResize, ptr(w.id), 0)
		}
		if changed&stateIconified != 0 {
			if state&stateIconified != 0 {
				w.h.Minimized()
			} else {
				w.h.Restored()
			}
		}
		if changed&stateMaximized != 0 {
			if state&stateMaximized != 0 {
				w.h.Maximized()
			} else {
				w.h.Unmaximized()
			}
		}
		if changed&stateFullscreen != 0 {
			if state&stateFullscreen != 0 {
				w.h.EnteredFullScreen()
			} else {
				w.h.LeftFullScreen()
			}
			if w.menubar != 0 {
				gtkWidgetSetVisible(w.menubar, !w.menuHides())
			}
			if w.controls != nil {
				w.fullScreenChanged()
			}
		}
		return false
	})
	cbControlsAllocated = purego.NewCallback(func(widget, allocation, data ptr) {
		if w := b().window(data); w != nil && w.controls != nil {
			w.measureControls()
		}
	})
	// The desktop's button layout changed: every window's title buttons
	// follow it.
	cbDecorationLayout = purego.NewCallback(func(settings, pspec, data ptr) {
		for _, w := range b().windows {
			if w.controls != nil && !w.closed {
				w.layoutControls()
			}
		}
	})
	cbButtonPress = purego.NewCallback(func(widget, event, data ptr) bool {
		w := b().window(data)
		if w == nil {
			return false
		}
		if w.host != nil {
			// GdkEventButton: type 0, x 24, y 32, state 48, button 52.
			if field[int32](event, 0) == 4 {
				w.webViewPressed(field[float64](event, 24), field[float64](event, 32), field[uint32](event, 52), gdkMods(field[uint32](event, 48)))
			}
			// A web view's page drags the window it is in (StartDrag).
			w = w.top()
		}
		w.altAlone = false
		// GdkEventButton: time 20, button 52, x_root 64, y_root 72.
		w.press.time = field[uint32](event, 20)
		w.press.button = int32(field[uint32](event, 52))
		w.press.rootX = field[float64](event, 64)
		w.press.rootY = field[float64](event, 72)
		w.press.hasPressed = true
		if w.press.event != 0 {
			gdkEventFree(w.press.event)
		}
		w.press.event = gdkEventCopy(event)
		if edge := w.resizeEdge(event); edge >= 0 {
			// GdkEventButton: type 0; GDK_BUTTON_PRESS, not a double click.
			if field[int32](event, 0) == 4 && w.press.button == 1 {
				gtkWindowBeginResizeDrag(w.win, edge, w.press.button, int32(w.press.rootX), int32(w.press.rootY), w.press.time)
			}
			return true // the edges are not the page's
		}
		return false
	})
	cbMotion = purego.NewCallback(func(widget, event, data ptr) bool {
		w := b().window(data)
		if w == nil {
			return false
		}
		edge := w.resizeEdge(event)
		w.showResizeCursor(edge)
		w.checkDragStart(event)
		return edge >= 0
	})
	// WebKit asks for the data of a drag while it moves over the page,
	// before the drop; the uri-list of files gives their paths.
	cbDragData = purego.NewCallback(func(widget, context ptr, x, y int32, sel ptr, info, time uint32, data ptr) {
		if w := b().window(data); w != nil {
			w.dragged = selectionPaths(sel)
		}
	})
	// Runs before WebKit's own handler, which gets the drop to the page.
	cbDragDrop = purego.NewCallback(func(widget, context ptr, x, y int32, time uint32, data ptr) bool {
		if w := b().window(data); w != nil {
			w.dropped, w.dragged = w.dragged, nil
		}
		return false
	})
	cbScriptMessage = purego.NewCallback(func(ucm, result, data ptr) {
		if w := b().window(data); w != nil {
			w.h.Message(takeStr(jscValueToString(webkitJavascriptResultGetJSValue(result))))
		}
	})
	cbLoadChanged = purego.NewCallback(func(web ptr, event int32, data ptr) {
		w := b().window(data)
		if w == nil {
			return
		}
		switch event {
		case 0: // WEBKIT_LOAD_STARTED
			w.h.NavigationStarted(w.URL())
		case 2: // WEBKIT_LOAD_COMMITTED
			w.h.NavigationCommitted(w.URL())
		case 3: // WEBKIT_LOAD_FINISHED
			w.h.LoadFinished()
		}
	})
	cbLoadFailed = purego.NewCallback(func(web ptr, event int32, uri, gerr, data ptr) bool {
		w := b().window(data)
		if w == nil || gerr == 0 {
			return false
		}
		code := field[int32](gerr, 4)
		// Cancelled loads (302) and navigations stopped by policy (102)
		// are not failures.
		if code == 302 || code == 102 {
			return false
		}
		w.h.LoadFailed(goStr(uri), int(code), goStr(field[ptr](gerr, 8)))
		return false
	})
	cbTitle = purego.NewCallback(func(web, pspec, data ptr) {
		if w := b().window(data); w != nil {
			w.h.TitleChanged(goStr(webkitWebViewGetTitle(w.web)))
		}
	})
	cbDecidePolicy = purego.NewCallback(func(web, decision ptr, kind int32, data ptr) bool {
		w := b().window(data)
		if w == nil {
			return false
		}
		if kind == 2 { // WEBKIT_POLICY_DECISION_TYPE_RESPONSE
			return w.decideResponse(decision)
		}
		if kind != 0 { // WEBKIT_POLICY_DECISION_TYPE_NAVIGATION_ACTION
			return false
		}
		action := webkitNavigationPolicyDecisionGetNavigationAction(decision)
		navType := webkitNavigationActionGetNavigationType(action)
		mainFrame := true
		if webkitNavigationPolicyDecisionGetFrameName != nil {
			mainFrame = webkitNavigationPolicyDecisionGetFrameName(decision) == 0
		}
		// WebKitNavigationType: link, form, back/forward, reload, …
		nav := platform.Navigation{
			URL:           goStr(webkitURIRequestGetURI(webkitNavigationActionGetRequest(action))),
			IsMainFrame:   mainFrame,
			UserInitiated: webkitNavigationActionIsUserGesture(action) || navType == 0 || navType == 1,
			Unasked:       navType == 2 || navType == 3,
			Reload:        navType == 3,
		}
		if nav.IsMainFrame && w.programmatic {
			w.programmatic = false
			nav.Unasked = true
		}
		if w.h.WillNavigate(nav) {
			webkitPolicyDecisionUse(decision)
		} else {
			webkitPolicyDecisionIgnore(decision)
		}
		return true
	})
	cbCreate = purego.NewCallback(func(web, action, data ptr) ptr {
		w := b().window(data)
		if w == nil {
			return 0
		}
		// WebKitGTK makes a related view share the opener's user content
		// manager, which would route the new window's IPC to the opener.
		// Open an independent window instead; its page has no opener.
		url := goStr(webkitURIRequestGetURI(webkitNavigationActionGetRequest(action)))
		if child, ok := w.h.NewWindow(platform.NewWindowRequest{URL: url}).(*window); ok && child != nil {
			child.LoadURL(url)
		}
		return 0
	})
	cbClose = purego.NewCallback(func(web, data ptr) {
		if w := b().window(data); w != nil {
			w.h.ClosedByPage()
		}
	})
	cbCrashed = purego.NewCallback(func(web ptr, reason int32, data ptr) {
		if w := b().window(data); w != nil {
			w.h.RenderProcessGone(map[int32]string{0: "crashed", 1: "exceeded memory limit", 2: "terminated by API"}[reason])
		}
	})
	cbAsyncReady = purego.NewCallback(func(source, res, data ptr) {
		if fn := pending.take(data); fn != nil {
			fn(source, res)
		}
	})
	// Camera, microphone, location and notifications; other requests get
	// WebKit's default answer.
	cbPermission = purego.NewCallback(func(web, req, data ptr) bool {
		w := b().window(data)
		if w == nil {
			return false
		}
		var kinds []string
		switch {
		case gTypeCheckInstanceIsA(req, webkitUserMediaPermissionRequestGetType()):
			if webkitUserMediaPermissionIsForVideoDevice(req) {
				kinds = append(kinds, "camera")
			}
			if webkitUserMediaPermissionIsForAudioDevice(req) {
				kinds = append(kinds, "microphone")
			}
		case gTypeCheckInstanceIsA(req, webkitGeolocationPermissionRequestGetType()):
			kinds = []string{"geolocation"}
		case gTypeCheckInstanceIsA(req, webkitNotificationPermissionRequestGetType()):
			kinds = []string{"notifications"}
		default:
			return false
		}
		origin := goStr(webkitWebViewGetURI(web))
		if u, err := url.Parse(origin); err == nil {
			origin = (&url.URL{Scheme: u.Scheme, Host: u.Host}).String()
		}
		if w.h.PermissionRequested(kinds, origin) {
			webkitPermissionRequestAllow(req)
		} else {
			webkitPermissionRequestDeny(req)
		}
		return true
	})
	cbPrintFinished = purego.NewCallback(func(op, data ptr) {
		if job := printJobs[op]; job != nil {
			job.done(job.err)
		}
	})
	cbPrintFailed = purego.NewCallback(func(op, gerr, data ptr) {
		if job := printJobs[op]; job != nil {
			// GError: domain, code, then the message.
			job.err = fmt.Errorf("mygo: printing to PDF: %s", goStr(*(*ptr)(unsafe.Add(*(*unsafe.Pointer)(unsafe.Pointer(&gerr)), 8))))
		}
	})
	cbPNGWrite = purego.NewCallback(func(closure, data ptr, length uint32) int32 {
		if buf := pngBuffers[closure]; buf != nil && length > 0 {
			buf.Write(unsafe.Slice(*(**byte)(unsafe.Pointer(&data)), length))
		}
		return 0 // CAIRO_STATUS_SUCCESS
	})
}

// SetVibrancyStyle does nothing: Linux has no material.
func (w *window) SetVibrancyStyle(platform.VibrancyStyle) {}
