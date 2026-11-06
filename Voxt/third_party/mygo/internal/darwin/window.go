//go:build darwin

package darwin

import (
	"errors"
	"log"
	"math"
	"os"
	"strconv"
	"strings"

	"github.com/ebitengine/purego/objc"

	"github.com/egoist/mygo/internal/bridge"
	"github.com/egoist/mygo/internal/platform"
)

const (
	styleTitled              = 1 << 0
	styleClosable            = 1 << 1
	styleMiniaturizable      = 1 << 2
	styleResizable           = 1 << 3
	styleFullScreen          = 1 << 14
	styleFullSizeContentView = 1 << 15
	styleNonactivatingPanel  = 1 << 7

	nsBackingStoreBuffered   = 2
	nsFloatingWindowLevel    = 3
	nsPopUpMenuWindowLevel   = 101
	nsNormalWindowLevel      = 0
	nsViewWidthHeightSizable = 2 | 16

	collectionCanJoinAllSpaces    = 1 << 0
	collectionTransient           = 1 << 3
	collectionIgnoresCycle        = 1 << 6
	collectionFullScreenPrimary   = 1 << 7
	collectionFullScreenAuxiliary = 1 << 8
	collectionFullScreenNone      = 1 << 9

	// NSApplicationPresentationOptions
	presentationAutoHideMenuBar = 1 << 2
	presentationFullScreen      = 1 << 10
	presentationAutoHideToolbar = 1 << 11
)

var (
	errInvalidImage = errors.New("mygo: invalid image data")
	errClosed       = errors.New("mygo: window has been closed")
)

type window struct {
	b        *Backend
	h        platform.WindowHandler
	opts     *platform.WindowOptions
	win      id
	view     id // the content view: holds the web view and a docked inspector
	web      id
	delegate id
	ucc      id
	effect   id
	// vstyle shapes the effect view (SetVibrancyStyle).
	vstyle platform.VibrancyStyle
	parent *window
	// flyout is how a flyout (WindowOptions.Flyout) was placed last, at
	// flyoutSize.
	flyout     platform.Flyout
	flyoutSize platform.Size
	// popover shows a flyout with the look of AppKit's popovers
	// (Flyout.Popover), in its window, win while it shows; popoverClosing
	// tells that the backend closes it, not the user.
	popover, popoverController id
	popoverClosing             bool

	lastMouseDown id
	dropped       []string         // DroppedFiles
	downloads     map[id][2]string // WKDownload: URL and path
	attention     int              // the request of FlashFrame
	maximized     bool
	// auxiliary tells that SetVisibleOnAllWorkspaces made a window that
	// cannot go full screen an auxiliary of full-screen Spaces, to undo.
	auxiliary     bool
	programmatic  bool
	closed        bool
	sheet         bool
	trafficLights *platform.Point
	// surface shows the content MyGo draws, in place of the web view.
	surface *surface
	// webViews are the web views under the surface (webview.go), and host
	// the window whose surface a web view is under, in the clip view clip.
	webViews []*window
	host     *window
	clip     id
	// pointerIn tells that the page of a web view has the pointer, where
	// nothing the content painted covers it.
	pointerIn bool
	// accessParent is the element of the content showing a web view,
	// which assistive technology finds its page in.
	accessParent id
}

func (b *Backend) NewWindow(o *platform.WindowOptions, h platform.WindowHandler) (platform.Window, error) {
	w := &window{b: b, h: h, opts: o, trafficLights: o.TrafficLightPosition, downloads: map[id][2]string{}}
	if p, ok := o.Parent.(*window); ok && p != nil {
		w.parent = p
	}
	withPool(w.create)
	return w, nil
}

func (w *window) create() {
	o := w.opts

	if o.Flyout != nil {
		w.createFlyout()
		return
	}
	style := uint(styleTitled)
	if o.Closable {
		style |= styleClosable
	}
	if o.Minimizable {
		style |= styleMiniaturizable
	}
	if o.Resizable {
		style |= styleResizable
	}
	hiddenTitleBar := o.Frameless || o.TitleBarStyle == "hidden" || o.TitleBarStyle == "hiddenInset"
	if hiddenTitleBar {
		style |= styleFullSizeContentView
	}

	size := NSRect{Size: NSSize{float64(o.Width), float64(o.Height)}}
	content := size
	if !o.UseContentSize {
		content = msgRectForRect(class("NSWindow"), sel("contentRectForFrameRect:styleMask:"), size, style)
		content.Origin = NSPoint{}
	}
	cls := "MyGoWindow"
	switch {
	case !o.Focusable:
		// WindowOptions.NonActivating: the style must be there at init,
		// for the window server to know the window never activates.
		cls = "MyGoNonactivatingPanel"
		style |= styleNonactivatingPanel
	case o.Panel:
		// The nonactivating style must be there from the start: the window
		// server only learns that the window never activates its app from
		// the style it is created with.
		cls = "MyGoPanel"
		style |= styleNonactivatingPanel
	}
	w.win = msgInitWindow(send(class(cls), "alloc"), sel("initWithContentRect:styleMask:backing:defer:"),
		content, style, nsBackingStoreBuffered, false)
	send(w.win, "setReleasedWhenClosed:", 0)
	if !o.Focusable {
		send(w.win, "setFloatingPanel:", 1)
		send(w.win, "setBecomesKeyOnlyIfNeeded:", 1)
		send(w.win, "setHidesOnDeactivate:", 0)
	}
	w.delegate = alloc("MyGoWindowDelegate")
	send(w.win, "setDelegate:", uintptr(w.delegate))
	send(w.win, "setTitle:", uintptr(nsString(o.Title)))
	send(w.win, "setTabbingMode:", 2) // NSWindowTabbingModeDisallowed

	if hiddenTitleBar {
		send(w.win, "setTitlebarAppearsTransparent:", 1)
		send(w.win, "setTitleVisibility:", 1) // NSWindowTitleHidden
	}
	if o.Frameless {
		for i := uintptr(0); i < 3; i++ {
			send(send(w.win, "standardWindowButton:", i), "setHidden:", 1)
		}
	}
	if o.TitleBarStyle == "hiddenInset" && !o.Frameless {
		tb := send(send(class("NSToolbar"), "alloc"), "initWithIdentifier:", uintptr(nsString("mygo.titlebar")))
		send(tb, "setShowsBaselineSeparator:", 0)
		send(w.win, "setToolbar:", uintptr(tb))
		release(tb)
	}
	if o.MinSize.Width > 0 || o.MinSize.Height > 0 {
		w.SetMinimumSize(o.MinSize)
	}
	if o.MaxSize.Width > 0 || o.MaxSize.Height > 0 {
		w.SetMaximumSize(o.MaxSize)
	}
	send(w.win, "setMovable:", boolArg(o.Movable))
	if !o.Maximizable {
		w.SetMaximizable(false)
	}
	behavior := uint(collectionFullScreenPrimary)
	if !o.Fullscreenable {
		behavior = collectionFullScreenNone
	}
	if o.Panel {
		// On the Space the user is on, full-screen ones included, as they
		// switch, and out of Mission Control and the window cycle.
		behavior = collectionCanJoinAllSpaces | collectionFullScreenAuxiliary | collectionIgnoresCycle | collectionTransient
		// A panel stays as the app is deactivated: the app hides it.
		send(w.win, "setHidesOnDeactivate:", 0)
	}
	send(w.win, "setCollectionBehavior:", uintptr(behavior))
	if o.AlwaysOnTop {
		send(w.win, "setLevel:", nsFloatingWindowLevel)
	}
	w.fillWindow(content)

	if o.Center {
		send(w.win, "center")
	} else {
		frame := msgRect(w.win, sel("frame"))
		r := rectFromMac(frame)
		r.X, r.Y = o.X, o.Y
		msgSetRectBool(w.win, sel("setFrame:display:"), rectToMac(r), false)
	}
	if w.trafficLights != nil {
		w.layoutTrafficLights()
		// The window's appearance changes with the application's, or when
		// it is given its own; either lays the title bar out again.
		send(w.win, "addObserver:forKeyPath:options:context:", uintptr(w.delegate), uintptr(nsString("effectiveAppearance")), 0, 0)
	}
	w.register()

	if o.FullScreen {
		send(w.win, "toggleFullScreen:", 0)
	} else if o.Maximized {
		send(w.win, "zoom:", 0)
	}
}

// createFlyout creates the panel of a flyout (WindowOptions.Flyout):
// borderless and nonactivating, so that it never activates the app,
// and the key window only when it is focusable, while its parent stays
// the main window. It is a child window of its parent while it shows.
func (w *window) createFlyout() {
	o := w.opts
	content := NSRect{Size: NSSize{float64(o.Width), float64(o.Height)}}
	w.flyout = *o.Flyout
	w.flyoutSize = platform.Size{Width: o.Width, Height: o.Height}
	if w.flyout.Popover {
		w.createPopover(content)
		return
	}
	w.win = msgInitWindow(send(class("MyGoFlyout"), "alloc"), sel("initWithContentRect:styleMask:backing:defer:"),
		content, styleNonactivatingPanel, nsBackingStoreBuffered, false)
	send(w.win, "setReleasedWhenClosed:", 0)
	w.delegate = alloc("MyGoWindowDelegate")
	send(w.win, "setDelegate:", uintptr(w.delegate))
	send(w.win, "setTitle:", uintptr(nsString(o.Title)))
	// NSPanel hides when the app is deactivated: a flyout stays with its
	// parent instead, and one that takes the keyboard closes.
	send(w.win, "setHidesOnDeactivate:", 0)
	send(w.win, "setCollectionBehavior:", collectionTransient|collectionIgnoresCycle|collectionFullScreenAuxiliary)
	if w.flyout.Tray != nil {
		// Over the windows of other apps, as the menus of the menu bar.
		send(w.win, "setLevel:", nsPopUpMenuWindowLevel)
	}
	w.fillWindow(content)
	w.register()
	w.PlaceFlyout(w.flyout, w.flyoutSize)
}

// fillWindow gives a new window its look and its web view or surface.
func (w *window) fillWindow(content NSRect) {
	o := w.opts
	send(w.win, "setHasShadow:", boolArg(o.HasShadow))
	if o.Opacity < 1 {
		msgSetFloat(w.win, sel("setAlphaValue:"), o.Opacity)
	}
	if o.Transparent {
		send(w.win, "setOpaque:", 0)
		send(w.win, "setBackgroundColor:", uintptr(send(class("NSColor"), "clearColor")))
	} else if o.BackgroundColor != nil {
		send(w.win, "setBackgroundColor:", uintptr(nsColor(*o.BackgroundColor)))
	}

	w.createContent(content)
	if w.surface != nil {
		send(w.win, "makeFirstResponder:", uintptr(w.surface.view))
	}
	send(w.win, "setContentView:", uintptr(w.view))
	if o.Vibrancy != "" {
		w.SetVibrancy(o.Vibrancy)
	}
}

// createContent creates the web view or the surface, in w.view.
func (w *window) createContent(content NSRect) {
	if w.opts.Surface {
		w.createSurface(content)
	} else {
		w.createWebView(content)
	}
	// WebKit docks the inspector next to the web view, in its superview.
	// That must not be the window's frame view: AppKit would draw a broken
	// legacy title bar from then on.
	w.view = msgInitRect(send(class("NSView"), "alloc"), sel("initWithFrame:"), NSRect{Size: content.Size})
	if w.surface != nil {
		send(w.view, "addSubview:", uintptr(w.surface.view))
	} else {
		send(w.view, "addSubview:", uintptr(w.web))
	}
}

// register lets the delegate, the web view and the window find w.
func (w *window) register() {
	b := w.b
	b.byDelegate[w.delegate] = w
	b.byWebView[w.web] = w
	if w.win != 0 {
		b.byNSWindow[w.win] = w
	}
}

// PlaceFlyout places a flyout next to its anchor, in the parent's content
// area, as Flyout.Place resolves it in the visible frame of a screen.
func (w *window) PlaceFlyout(f platform.Flyout, size platform.Size) {
	if w.opts.Flyout == nil || w.closed {
		return
	}
	w.flyout, w.flyoutSize = f, size
	if w.popover != 0 {
		w.placePopover()
		return
	}
	displays := screen{}.Displays()
	anchor := f.Anchor
	if t, ok := f.Tray.(*tray); ok {
		if anchor = t.Bounds(); anchor.Width == 0 || anchor.Height == 0 {
			r := platform.Centered(size, platform.PrimaryWorkArea(displays))
			msgSetRectBool(w.win, sel("setFrame:display:"), rectToMac(r), true)
			return
		}
	} else if p := w.parent; p != nil && !p.closed {
		c := p.ContentBounds()
		anchor.X += c.X
		anchor.Y += c.Y
	}
	r, _, _ := f.Place(anchor, size, platform.WorkAreaFor(anchor, displays))
	msgSetRectBool(w.win, sel("setFrame:display:"), rectToMac(r), true)
}

func (w *window) createWebView(content NSRect) {
	o := w.opts
	var cfg id
	if o.Native != 0 {
		// window.open(): WebKit hands us the configuration to use.
		cfg = retain(id(o.Native))
	} else {
		cfg = alloc("WKWebViewConfiguration")
		for _, s := range o.Schemes {
			send(cfg, "setURLSchemeHandler:forURLScheme:", uintptr(schemeHandler()), uintptr(nsString(s)))
		}
	}
	prefs := send(cfg, "preferences")
	send(prefs, "setValue:forKey:", uintptr(nsBool(o.DevTools)), uintptr(nsString("developerExtrasEnabled")))
	send(prefs, "setValue:forKey:", uintptr(nsBool(true)), uintptr(nsString("allowFileAccessFromFileURLs")))
	if respondsTo(prefs, "setElementFullscreenEnabled:") {
		send(prefs, "setElementFullscreenEnabled:", 1)
	}

	// Every window gets its own content controller so messages, scripts and
	// the bridge configuration never leak between windows.
	w.ucc = alloc("WKUserContentController")
	send(w.ucc, "addScriptMessageHandler:name:", uintptr(w.delegate), uintptr(nsString("mygo")))
	w.addUserScripts(o.Zoom)
	send(cfg, "setUserContentController:", uintptr(w.ucc))

	w.web = msgInitRectID(send(class("MyGoWebView"), "alloc"), sel("initWithFrame:configuration:"), NSRect{Size: content.Size}, cfg)
	release(cfg)
	send(w.web, "setNavigationDelegate:", uintptr(w.delegate))
	send(w.web, "setUIDelegate:", uintptr(w.delegate))
	send(w.web, "setAutoresizingMask:", nsViewWidthHeightSizable)
	if o.Transparent || o.BackgroundColor != nil || o.Vibrancy != "" {
		// Let the window color show until the page paints.
		send(w.web, "setValue:forKey:", uintptr(nsBool(false)), uintptr(nsString("drawsBackground")))
		if o.BackgroundColor != nil && respondsTo(w.web, "setUnderPageBackgroundColor:") {
			send(w.web, "setUnderPageBackgroundColor:", uintptr(nsColor(*o.BackgroundColor)))
		}
	}
	if respondsTo(w.web, "setInspectable:") {
		send(w.web, "setInspectable:", boolArg(o.DevTools))
	}
	if o.UserAgent != "" {
		send(w.web, "setCustomUserAgent:", uintptr(nsString(o.UserAgent)))
	}
	if o.Zoom > 0 && o.Zoom != 1 {
		msgSetFloat(w.web, sel("setPageZoom:"), o.Zoom)
	}
	send(w.web, "addObserver:forKeyPath:options:context:", uintptr(w.delegate), uintptr(nsString("title")), 1, 0)
}

func nsColor(c platform.Color) id {
	return msgColor(class("NSColor"), sel("colorWithSRGBRed:green:blue:alpha:"),
		float64(c.R)/255, float64(c.G)/255, float64(c.B)/255, float64(c.A)/255)
}

// cleanup releases native resources; called from windowWillClose:.
func (w *window) cleanup() {
	if w.closed {
		return
	}
	w.closeWebViews()
	w.closed = true
	b := w.b
	send(w.web, "removeObserver:forKeyPath:", uintptr(w.delegate), uintptr(nsString("title")))
	if w.trafficLights != nil {
		send(w.win, "removeObserver:forKeyPath:", uintptr(w.delegate), uintptr(nsString("effectiveAppearance")))
	}
	send(w.ucc, "removeScriptMessageHandlerForName:", uintptr(nsString("mygo")))
	send(w.ucc, "removeAllUserScripts")
	send(w.web, "stopLoading")
	send(w.web, "setNavigationDelegate:", 0)
	send(w.web, "setUIDelegate:", 0)
	if w.popover != 0 {
		w.cleanupPopover()
	} else {
		send(w.win, "setDelegate:", 0)
		if w.parent != nil && !w.parent.closed {
			send(w.parent.win, "removeChildWindow:", uintptr(w.win))
		}
	}
	delete(b.byDelegate, w.delegate)
	delete(b.byWebView, w.web)
	delete(b.byNSWindow, w.win)
	release(w.lastMouseDown)
	w.lastMouseDown = 0
	release(w.ucc)
	for _, j := range printJobs {
		if j.w == w && j.cb != nil {
			cb := j.cb
			j.cb = nil
			cb(nil, errDestroyed)
		}
	}
	if w.surface != nil {
		w.surface.destroy()
	}
	release(w.effect)
	release(w.web)
	release(w.view)
	// AppKit is still closing the window; let the pool release it.
	autorelease(w.delegate)
	if w.popover == 0 {
		autorelease(w.win)
	}
}

func (w *window) Handle() uintptr        { return uintptr(w.win) }
func (w *window) WebViewHandle() uintptr { return uintptr(w.web) }

func (w *window) SetTitle(title string) {
	withPool(func() { send(w.win, "setTitle:", uintptr(nsString(title))) })
	w.layoutTrafficLights()
}

// SetRepresentedFile shows the file's proxy icon in the title bar, which
// the user drags, and whose ⌘-click menu lists its folders.
func (w *window) SetRepresentedFile(path string) {
	withPool(func() {
		var url id
		if path != "" {
			url = fileURL(path)
		}
		send(w.win, "setRepresentedURL:", uintptr(url))
	})
	// AppKit lays the title bar's buttons out again.
	w.layoutTrafficLights()
}

// SetDocumentEdited shows the dot in the close button, and dims the proxy
// icon.
func (w *window) SetDocumentEdited(edited bool) {
	send(w.win, "setDocumentEdited:", boolArg(edited))
	w.layoutTrafficLights()
}

func (w *window) Title() string {
	var s string
	withPool(func() { s = goString(send(w.win, "title")) })
	return s
}

func (w *window) SetBounds(r platform.Rect) {
	msgSetRectBool(w.win, sel("setFrame:display:"), rectToMac(r), true)
}

func (w *window) Bounds() platform.Rect { return rectFromMac(msgRect(w.win, sel("frame"))) }

func (w *window) SetContentBounds(r platform.Rect) {
	frame := msgRectToRect(w.win, sel("frameRectForContentRect:"), rectToMac(r))
	msgSetRectBool(w.win, sel("setFrame:display:"), frame, true)
}

func (w *window) ContentBounds() platform.Rect {
	frame := msgRect(w.win, sel("frame"))
	return rectFromMac(msgRectToRect(w.win, sel("contentRectForFrameRect:"), frame))
}

func (w *window) SetMinimumSize(s platform.Size) {
	msgSetSize(w.win, sel("setMinSize:"), NSSize{float64(s.Width), float64(s.Height)})
}

func (w *window) SetMaximumSize(s platform.Size) {
	max := NSSize{math.MaxFloat32, math.MaxFloat32}
	if s.Width > 0 {
		max.Width = float64(s.Width)
	}
	if s.Height > 0 {
		max.Height = float64(s.Height)
	}
	msgSetSize(w.win, sel("setMaxSize:"), max)
}

func (w *window) setStyle(bit uint, on bool) {
	mask := uint(send(w.win, "styleMask"))
	if on {
		mask |= bit
	} else {
		mask &^= bit
	}
	send(w.win, "setStyleMask:", uintptr(mask))
}

func (w *window) hasStyle(bit uint) bool { return uint(send(w.win, "styleMask"))&bit != 0 }

func (w *window) SetResizable(v bool)   { w.setStyle(styleResizable, v) }
func (w *window) IsResizable() bool     { return w.hasStyle(styleResizable) }
func (w *window) SetMovable(v bool)     { send(w.win, "setMovable:", boolArg(v)) }
func (w *window) IsMovable() bool       { return sendBool(w.win, "isMovable") }
func (w *window) SetMinimizable(v bool) { w.setStyle(styleMiniaturizable, v) }
func (w *window) IsMinimizable() bool   { return w.hasStyle(styleMiniaturizable) }
func (w *window) SetClosable(v bool)    { w.setStyle(styleClosable, v) }
func (w *window) IsClosable() bool      { return w.hasStyle(styleClosable) }

func (w *window) SetMaximizable(v bool) {
	send(send(w.win, "standardWindowButton:", 2), "setEnabled:", boolArg(v))
}

func (w *window) IsMaximizable() bool {
	return sendBool(send(w.win, "standardWindowButton:", 2), "isEnabled")
}

func (w *window) SetAlwaysOnTop(v bool) {
	level := uintptr(nsNormalWindowLevel)
	if v {
		level = nsFloatingWindowLevel
	}
	send(w.win, "setLevel:", level)
}

func (w *window) IsAlwaysOnTop() bool { return sendInt(w.win, "level") != nsNormalWindowLevel }

func (w *window) Show() {
	if w.popover != 0 {
		w.showPopover()
		return
	}
	if w.opts.Flyout != nil {
		w.showFlyout(w.flyout.Focusable)
		return
	}
	if w.parent != nil && w.opts.Modal && !w.sheet {
		w.sheet = true
		send(w.parent.win, "beginSheet:completionHandler:", uintptr(w.win), 0)
		return
	}
	send(w.win, "makeKeyAndOrderFront:", 0)
	if w.parent != nil && !w.parent.closed {
		send(w.parent.win, "addChildWindow:ordered:", uintptr(w.win), 1)
	}
	if !w.opts.Panel {
		send(w.b.app, "activateIgnoringOtherApps:", 1)
	}
}

// showFlyout shows a flyout above its parent, as its child window, and
// gives it the keyboard when it takes it. Showing it does not activate
// the app: it shows over a parent the user is using.
func (w *window) showFlyout(key bool) {
	if key {
		w.b.watchOutsideClicks()
		send(w.win, "makeKeyAndOrderFront:", 0)
	} else {
		send(w.win, "orderFront:", 0)
	}
	if p := w.parent; p != nil && !p.closed && send(w.win, "parentWindow") == 0 {
		send(p.win, "addChildWindow:ordered:", uintptr(w.win), 1) // NSWindowAbove
	}
}

// hideFlyout hides a flyout. It stops being a child window, which AppKit
// would order in again with its parent, and gives the keyboard it had
// back to the parent.
func (w *window) hideFlyout() {
	key := sendBool(w.win, "isKeyWindow")
	if p := w.parent; p != nil && !p.closed {
		send(p.win, "removeChildWindow:", uintptr(w.win))
		if key && sendBool(p.win, "isVisible") {
			send(p.win, "makeKeyWindow")
		}
	}
	send(w.win, "orderOut:", 0)
}

// outsideClicks is the monitor of clicks outside the app, in other apps or
// the menu bar, which close focusable flyouts as AppKit's transient
// popovers close: their panels may take the keyboard without the app
// being activated, and a click in the menu bar activates no other app, so
// they would not resign the key window. It is made once, as the first
// focusable flyout shows.
var outsideClicks id

func (b *Backend) watchOutsideClicks() {
	if outsideClicks != 0 {
		return
	}
	const mask = 1<<1 | 1<<3 | 1<<25 // left, right and other mouse down
	blk := newBlock(func(_ objc.Block, ev id) {
		var open []*window
		for _, w := range b.byNSWindow {
			if w.opts.Flyout != nil && w.popover == 0 && w.flyout.Focusable && !w.closed && sendBool(w.win, "isVisible") {
				open = append(open, w)
			}
		}
		for _, w := range open {
			if !w.closed && w.h.ShouldClose() {
				w.Close()
			}
		}
	})
	outsideClicks = retain(send(class("NSEvent"), "addGlobalMonitorForEventsMatchingMask:handler:", mask, uintptr(blk)))
	b.watchAnchorPresses()
}

// anchorPresses is the monitor of presses in the app that eats a press on
// the anchor of a focusable flyout, which only closes the flyout, as
// AppKit's transient popovers eat a press on their positioning rect: the
// press would make the parent key, which closes the flyout before the
// release clicks what opened it, which would open it again. The press's
// drags and release are eaten too. Presses elsewhere go on to what is
// under the pointer. It is made once, with outsideClicks.
var anchorPresses id

// eatingPress is the type of the release that ends a press anchorPresses
// ate, or 0. Main thread only.
var eatingPress int

func (b *Backend) watchAnchorPresses() {
	// left, right and other mouse down, up and dragged
	const mask = 1<<1 | 1<<3 | 1<<25 | 1<<2 | 1<<4 | 1<<26 | 1<<6 | 1<<7 | 1<<27
	blk := newBlock(func(_ objc.Block, ev id) id {
		switch typ := sendInt(ev, "type"); typ {
		case 2, 4, 26: // NSEventTypeLeftMouseUp, RightMouseUp, OtherMouseUp
			if typ == eatingPress {
				eatingPress = 0
				return 0
			}
		case 6, 7, 27: // NSEventTypeLeftMouseDragged, RightMouseDragged, OtherMouseDragged
			if eatingPress != 0 {
				return 0
			}
		default: // a press, which ends any other
			eatingPress = 0
			if w := b.flyoutAnchoredAt(ev); w != nil {
				eatingPress = typ + 1
				if w.h.ShouldClose() {
					w.Close()
				}
				return 0
			}
		}
		return ev
	})
	anchorPresses = retain(send(class("NSEvent"), "addLocalMonitorForEventsMatchingMask:handler:", mask, uintptr(blk)))
}

// flyoutAnchoredAt returns the focusable flyout that shows in a panel
// whose anchor is under the press ev in its parent, or nil.
func (b *Backend) flyoutAnchoredAt(ev id) *window {
	p := b.byNSWindow[send(ev, "window")]
	if p == nil || p.closed {
		return nil
	}
	frame := msgRect(p.win, sel("frame"))
	c := msgRectToRect(p.win, sel("contentRectForFrameRect:"), frame)
	loc := msgPoint(ev, sel("locationInWindow"))
	x := loc.X - (c.Origin.X - frame.Origin.X)
	y := c.Origin.Y - frame.Origin.Y + c.Size.Height - loc.Y
	for _, w := range b.byNSWindow {
		if w.parent != p || w.opts.Flyout == nil || w.popover != 0 || !w.flyout.Focusable || w.closed || !sendBool(w.win, "isVisible") {
			continue
		}
		if a := w.flyout.Anchor; x >= float64(a.X) && y >= float64(a.Y) && x < float64(a.X+a.Width) && y < float64(a.Y+a.Height) {
			return w
		}
	}
	return nil
}

// dismissFlyout closes a focusable flyout that lost the keyboard, as the
// user closing it, unless it went to a flyout of its own.
func (w *window) dismissFlyout() {
	if w.closed || !w.flyout.Focusable || !sendBool(w.win, "isVisible") {
		return
	}
	for k := w.b.byNSWindow[send(w.b.app, "keyWindow")]; k != nil; k = k.parent {
		if k == w {
			return
		}
	}
	if w.h.ShouldClose() {
		w.Close()
	}
}

func (w *window) ShowInactive() {
	if w.popover != 0 {
		w.showPopover()
		return
	}
	if w.opts.Flyout != nil {
		w.showFlyout(false)
		return
	}
	send(w.win, "orderFrontRegardless")
	if w.parent != nil && !w.parent.closed {
		send(w.parent.win, "addChildWindow:ordered:", uintptr(w.win), 1)
	}
}

func (w *window) Hide() {
	if w.popover != 0 {
		w.hidePopover()
		return
	}
	if w.opts.Flyout != nil {
		w.hideFlyout()
		return
	}
	if w.sheet {
		w.sheet = false
		send(w.parent.win, "endSheet:", uintptr(w.win))
		return
	}
	send(w.win, "orderOut:", 0)
}

func (w *window) IsVisible() bool {
	if w.popover != 0 {
		return sendBool(w.popover, "isShown")
	}
	return sendBool(w.win, "isVisible")
}

func (w *window) Focus() {
	if w.host != nil {
		if !w.closed {
			send(w.win, "makeFirstResponder:", uintptr(w.web))
		}
		return
	}
	if w.opts.Flyout != nil {
		w.Show() // the key window only when it takes the keyboard
		return
	}
	send(w.win, "makeKeyAndOrderFront:", 0)
	if !w.opts.Panel {
		send(w.b.app, "activateIgnoringOtherApps:", 1)
	}
}

func (w *window) Blur() { send(w.win, "orderBack:", 0) }
func (w *window) IsFocused() bool {
	return sendBool(w.win, "isKeyWindow")
}
func (w *window) Minimize()         { send(w.win, "miniaturize:", 0) }
func (w *window) IsMinimized() bool { return sendBool(w.win, "isMiniaturized") }

func (w *window) Maximize() {
	if !w.IsMaximized() {
		send(w.win, "zoom:", 0)
	}
}

func (w *window) Unmaximize() {
	if w.IsMaximized() {
		send(w.win, "zoom:", 0)
	}
}

func (w *window) IsMaximized() bool { return sendBool(w.win, "isZoomed") && !w.IsFullScreen() }

func (w *window) Restore() {
	if w.IsMinimized() {
		send(w.win, "deminiaturize:", 0)
	}
}

func (w *window) SetFullScreen(v bool) {
	if v != w.IsFullScreen() {
		send(w.win, "toggleFullScreen:", 0)
	}
}

func (w *window) IsFullScreen() bool { return w.hasStyle(styleFullScreen) }
func (w *window) Center()            { send(w.win, "center") }

func (w *window) SetBackgroundColor(c platform.Color) {
	withPool(func() {
		if w.host != nil {
			// A web view's window is its host's.
			w.clipBackground(c)
		} else {
			send(w.win, "setBackgroundColor:", uintptr(nsColor(c)))
		}
		send(w.web, "setValue:forKey:", uintptr(nsBool(false)), uintptr(nsString("drawsBackground")))
		if respondsTo(w.web, "setUnderPageBackgroundColor:") {
			send(w.web, "setUnderPageBackgroundColor:", uintptr(nsColor(c)))
		}
	})
}

func (w *window) SetOpacity(v float64) { msgSetFloat(w.win, sel("setAlphaValue:"), v) }

func (w *window) SetProgressBar(state string, value float64) { w.b.setDockProgress(state, value) }

func (w *window) FlashFrame(flash bool) {
	if w.attention != 0 {
		send(w.b.app, "cancelUserAttentionRequest:", uintptr(w.attention))
		w.attention = 0
	}
	if flash {
		w.attention = sendInt(w.b.app, "requestUserAttention:", 10) // NSInformationalRequest
	}
}

// SetSkipTaskbar does nothing: the Dock shows applications, not windows.
func (w *window) SetSkipTaskbar(bool) {}

func (w *window) SetVisibleOnAllWorkspaces(v bool) {
	const canJoinAllSpaces = 1 << 0
	behavior := uint(sendInt(w.win, "collectionBehavior"))
	if v {
		behavior |= canJoinAllSpaces
		// FullScreenNone (DisableFullScreen) also keeps a window out of
		// other apps' full-screen Spaces. As an auxiliary, the window shows
		// there too (an accessory app's: macOS shows no regular app's
		// windows there), and still cannot go full screen itself.
		if behavior&collectionFullScreenNone != 0 {
			behavior = behavior&^collectionFullScreenNone | collectionFullScreenAuxiliary
			w.auxiliary = true
		}
	} else {
		behavior &^= canJoinAllSpaces
		if w.auxiliary {
			behavior = behavior&^collectionFullScreenAuxiliary | collectionFullScreenNone
			w.auxiliary = false
		}
	}
	send(w.win, "setCollectionBehavior:", uintptr(behavior))
}

// SetIcon does nothing: macOS windows show no icon of their own.
func (w *window) SetIcon([]byte) error        { return nil }
func (w *window) Opacity() float64            { return msgFloat(w.win, sel("alphaValue")) }
func (w *window) SetHasShadow(v bool)         { send(w.win, "setHasShadow:", boolArg(v)) }
func (w *window) HasShadow() bool             { return sendBool(w.win, "hasShadow") }
func (w *window) SetIgnoreMouseEvents(v bool) { send(w.win, "setIgnoresMouseEvents:", boolArg(v)) }

func (w *window) SetContentProtection(v bool) {
	sharing := uintptr(1) // NSWindowSharingReadOnly
	if v {
		sharing = 0 // NSWindowSharingNone
	}
	send(w.win, "setSharingType:", sharing)
}

// vibrancyMaterials maps mygo.Vibrancy values to NSVisualEffectMaterial.
var vibrancyMaterials = map[string]uintptr{
	"titlebar": 3, "selection": 4, "menu": 5, "popover": 6, "sidebar": 7, "header": 10,
	"sheet": 11, "window": 12, "hud": 13, "fullscreen-ui": 15, "tooltip": 17, "content": 18,
	"under-window": 21, "under-page": 22,
	// Windows 11 backdrops.
	"mica": 21, "tabbed": 21, "acrylic": 13,
}

func (w *window) SetVibrancy(material string) {
	withPool(func() {
		m, ok := vibrancyMaterials[material]
		if !ok {
			if material != "" {
				log.Printf("mygo: unknown vibrancy %q", material)
			}
			if w.effect != 0 {
				send(w.effect, "removeFromSuperview")
				release(w.effect)
				w.effect = 0
			}
			return
		}
		if w.effect == 0 {
			w.effect = msgInitRect(send(class("NSVisualEffectView"), "alloc"), sel("initWithFrame:"), msgRect(w.view, sel("bounds")))
			send(w.effect, "setBlendingMode:", 0) // NSVisualEffectBlendingModeBehindWindow
			send(w.effect, "setState:", 1)        // NSVisualEffectStateActive
			send(w.effect, "setAutoresizingMask:", nsViewWidthHeightSizable)
			// Behind the page and a docked inspector.
			send(w.view, "addSubview:positioned:relativeTo:", uintptr(w.effect), ^uintptr(0), 0) // NSWindowBelow
			send(w.web, "setValue:forKey:", uintptr(nsBool(false)), uintptr(nsString("drawsBackground")))
			w.styleEffect()
		}
		send(w.effect, "setMaterial:", m)
	})
}

func (w *window) SetVibrancyStyle(s platform.VibrancyStyle) {
	w.vstyle = s
	withPool(w.styleEffect)
}

// styleEffect fits the effect view to vstyle: its frame inset from the
// content view's bounds (which it follows as they resize), its corners
// rounded by a mask, its appearance.
func (w *window) styleEffect() {
	if w.effect == 0 {
		return
	}
	s := w.vstyle
	b := msgRect(w.view, sel("bounds"))
	// The content view is not flipped: Bottom is the origin's side.
	f := NSRect{
		Origin: NSPoint{X: s.Left, Y: s.Bottom},
		Size:   NSSize{Width: max(b.Size.Width-s.Left-s.Right, 0), Height: max(b.Size.Height-s.Top-s.Bottom, 0)},
	}
	msgSetRect(w.effect, sel("setFrame:"), f)
	var mask id
	if s.Radius > 0 {
		mask = roundedMask(s.Radius)
	}
	send(w.effect, "setMaskImage:", uintptr(mask))
	var appearance id
	if s.Dark {
		appearance = send(class("NSAppearance"), "appearanceNamed:", uintptr(nsString("NSAppearanceNameDarkAqua")))
	}
	send(w.effect, "setAppearance:", uintptr(appearance))
}

// roundedMask is a mask image for NSVisualEffectView: a rounded
// rectangle of radius r whose corners stay as they are while its middle
// stretches to the view's size (cap insets). Autoreleased.
func roundedMask(r float64) id {
	size := 2*r + 1
	img := autorelease(msgInitSize(send(class("NSImage"), "alloc"), sel("initWithSize:"), NSSize{Width: size, Height: size}))
	send(img, "lockFocus")
	send(send(class("NSColor"), "blackColor"), "set")
	path := msgRectRadii(class("NSBezierPath"), sel("bezierPathWithRoundedRect:xRadius:yRadius:"), NSRect{Size: NSSize{Width: size, Height: size}}, r, r)
	send(path, "fill")
	send(img, "unlockFocus")
	// NSEdgeInsets (top, left, bottom, right) has NSRect's layout.
	msgSetRect(img, sel("setCapInsets:"), NSRect{Origin: NSPoint{X: r, Y: r}, Size: NSSize{Width: r, Height: r}})
	send(img, "setResizingMode:", 1) // NSImageResizingModeStretch
	return img
}

func (w *window) SetMenu(*platform.Menu) {}

// SetAutoHideMenu does nothing: the menu bar belongs to the application.
func (w *window) SetAutoHideMenu(bool) {}

func (w *window) StartDrag() {
	if w.lastMouseDown != 0 {
		send(w.win, "performWindowDragWithEvent:", uintptr(w.lastMouseDown))
	}
}

// hiddenTitleBar reports a TitleBarStyle that hides the title bar but keeps
// the traffic lights.
func (w *window) hiddenTitleBar() bool {
	o := w.opts
	return !o.Frameless && (o.TitleBarStyle == "hidden" || o.TitleBarStyle == "hiddenInset")
}

// TitleBar returns the room the traffic lights take, from the left edge to
// the right edge of the zoom button, in the title bar, which is as tall as
// the window's own or, with TrafficLightPosition, leaves as much room below
// them as above. They hide in full screen.
func (w *window) TitleBar() platform.TitleBar {
	if !w.hiddenTitleBar() || w.closed || w.IsFullScreen() {
		return platform.TitleBar{}
	}
	closeBtn := send(w.win, "standardWindowButton:", 0)
	zoom := send(w.win, "standardWindowButton:", 2)
	if closeBtn == 0 || zoom == 0 {
		return platform.TitleBar{}
	}
	c, z := msgRect(closeBtn, sel("frame")), msgRect(zoom, sel("frame"))
	if p := w.trafficLights; p != nil {
		// Where layoutTrafficLights puts them, which may not have run yet.
		right := float64(p.X) + z.Origin.X - c.Origin.X + z.Size.Width
		return platform.TitleBar{Height: int(math.Round(c.Size.Height + 2*float64(p.Y))), Left: int(math.Ceil(right))}
	}
	frame := msgRect(w.win, sel("frame"))
	content := msgRect(w.win, sel("contentLayoutRect"))
	height := frame.Size.Height - content.Origin.Y - content.Size.Height
	return platform.TitleBar{Height: int(math.Round(height)), Left: int(math.Ceil(z.Origin.X + z.Size.Width))}
}

func (w *window) TitleBarDoubleClicked() {
	var action string
	withPool(func() {
		defaults := send(class("NSUserDefaults"), "standardUserDefaults")
		action = goString(send(defaults, "stringForKey:", uintptr(nsString("AppleActionOnDoubleClick"))))
	})
	switch action {
	case "None":
	case "Minimize":
		send(w.win, "performMiniaturize:", 0)
	default:
		send(w.win, "performZoom:", 0)
	}
}

func (w *window) Close() {
	if w.host != nil {
		w.closeWebView()
		return
	}
	if w.closed {
		return
	}
	if w.popover != 0 {
		w.closePopover()
		return
	}
	if w.sheet {
		w.sheet = false
		send(w.parent.win, "endSheet:", uintptr(w.win))
	}
	if w.opts.Flyout != nil {
		w.hideFlyout()
	}
	send(w.win, "close")
}

// layoutTrafficLights puts the top-left corner of the close button at the
// requested position, and the other buttons after it, in a title bar that
// leaves as much room below them as above. AppKit lays the title bar out
// again when the window resizes, its title changes or the appearance
// changes, so this runs after each of them.
func (w *window) layoutTrafficLights() {
	p := w.trafficLights
	if p == nil || w.closed || w.IsFullScreen() {
		return
	}
	closeBtn := send(w.win, "standardWindowButton:", 0)
	if closeBtn == 0 {
		return
	}
	container := send(send(closeBtn, "superview"), "superview")
	if container == 0 {
		return
	}
	btnFrame := msgRect(closeBtn, sel("frame"))
	height := btnFrame.Size.Height + 2*float64(p.Y)
	winFrame := msgRect(w.win, sel("frame"))
	cf := msgRect(container, sel("frame"))
	cf.Size.Height = height
	cf.Origin.Y = winFrame.Size.Height - height
	msgSetRect(container, sel("setFrame:"), cf)

	mini := send(w.win, "standardWindowButton:", 1)
	gap := msgRect(mini, sel("frame")).Origin.X - btnFrame.Origin.X
	for i := uintptr(0); i < 3; i++ {
		btn := send(w.win, "standardWindowButton:", i)
		msgSetPoint(btn, sel("setFrameOrigin:"), NSPoint{float64(p.X) + float64(i)*gap, float64(p.Y)})
	}
}

// Navigation.

func (w *window) LoadURL(url string) {
	w.programmatic = true
	withPool(func() {
		req := send(class("NSURLRequest"), "requestWithURL:", uintptr(nsURL(url)))
		send(w.web, "loadRequest:", uintptr(req))
	})
}

func (w *window) LoadHTML(html, baseURL string) {
	w.programmatic = true
	withPool(func() {
		var base id
		if baseURL != "" {
			base = nsURL(baseURL)
		}
		send(w.web, "loadHTMLString:baseURL:", uintptr(nsString(html)), uintptr(base))
	})
}

func (w *window) LoadFile(path, readAccessDir string) {
	w.programmatic = true
	withPool(func() {
		send(w.web, "loadFileURL:allowingReadAccessToURL:", uintptr(fileURL(path)), uintptr(fileURL(readAccessDir)))
	})
}

func (w *window) Reload(ignoreCache bool) {
	if ignoreCache {
		send(w.web, "reloadFromOrigin")
	} else {
		send(w.web, "reload")
	}
}

func (w *window) StopLoading()       { send(w.web, "stopLoading") }
func (w *window) GoBack()            { send(w.web, "goBack") }
func (w *window) GoForward()         { send(w.web, "goForward") }
func (w *window) CanGoBack() bool    { return sendBool(w.web, "canGoBack") }
func (w *window) CanGoForward() bool { return sendBool(w.web, "canGoForward") }
func (w *window) IsLoading() bool    { return sendBool(w.web, "isLoading") }

func (w *window) URL() string {
	var s string
	withPool(func() { s = goString(send(send(w.web, "URL"), "absoluteString")) })
	return s
}

// addUserScripts adds the user scripts of the options to the content
// controller, and the title bar's after them. Pool required.
func (w *window) addUserScripts(zoom float64) {
	add := func(src string, injection uintptr, mainFrameOnly bool) {
		script := send(send(class("WKUserScript"), "alloc"), "initWithSource:injectionTime:forMainFrameOnly:",
			uintptr(nsString(src)), injection, boolArg(mainFrameOnly))
		send(w.ucc, "addUserScript:", uintptr(script))
		release(script)
	}
	for _, s := range w.opts.UserScripts {
		injection := uintptr(0) // WKUserScriptInjectionTimeAtDocumentStart
		if s.AtDocumentEnd {
			injection = 1
		}
		add(s.Source, injection, !s.AllFrames)
	}
	if w.hiddenTitleBar() {
		// After the bridge, which it tells: the room of the traffic lights.
		add(bridge.TitleBarScript(w.TitleBar(), zoom), 0, true)
	}
}

// SetUserScripts replaces the user scripts: a content controller can only
// remove all of its scripts, so they are all added again.
func (w *window) SetUserScripts(scripts []platform.UserScript) {
	if w.ucc == 0 || w.closed {
		return
	}
	w.opts.UserScripts = scripts
	withPool(func() {
		send(w.ucc, "removeAllUserScripts")
		w.addUserScripts(w.Zoom())
	})
}

func (w *window) Eval(js string) {
	withPool(func() {
		send(w.web, "evaluateJavaScript:completionHandler:", uintptr(nsString(js)), 0)
	})
}

func (w *window) CallAsyncFunction(body string, cb func(string, error)) {
	if !respondsTo(w.web, "callAsyncJavaScript:arguments:inFrame:inContentWorld:completionHandler:") {
		cb("", errors.New("mygo: Eval requires macOS 11 or later"))
		return
	}
	withPool(func() {
		blk := newBlock(func(_ objc.Block, result id, err id) {
			if err != 0 {
				cb("", nsError(err))
				return
			}
			cb(goString(result), nil)
		})
		defer blk.Release()
		send(w.web, "callAsyncJavaScript:arguments:inFrame:inContentWorld:completionHandler:",
			uintptr(nsString(body)), 0, 0, uintptr(send(class("WKContentWorld"), "pageWorld")), uintptr(blk))
	})
}

func (w *window) SetZoom(f float64) { msgSetFloat(w.web, sel("setPageZoom:"), f) }
func (w *window) Zoom() float64     { return msgFloat(w.web, sel("pageZoom")) }

func (w *window) SetUserAgent(ua string) {
	withPool(func() {
		var s id
		if ua != "" {
			s = nsString(ua)
		}
		send(w.web, "setCustomUserAgent:", uintptr(s))
	})
}

func (w *window) UserAgent() string {
	var s string
	withPool(func() {
		s = goString(send(w.web, "customUserAgent"))
		if s == "" && respondsTo(w.web, "_userAgent") {
			s = goString(send(w.web, "_userAgent"))
		}
	})
	return s
}

func (w *window) inspector() id {
	if !w.opts.DevTools || !respondsTo(w.web, "_inspector") {
		return 0
	}
	return send(w.web, "_inspector")
}

func (w *window) OpenDevTools() {
	if in := w.inspector(); in != 0 {
		send(in, "show")
	}
}

func (w *window) CloseDevTools() {
	if in := w.inspector(); in != 0 {
		send(in, "close")
	}
}

func (w *window) IsDevToolsOpened() bool {
	in := w.inspector()
	return in != 0 && respondsTo(in, "isVisible") && sendBool(in, "isVisible")
}

func (w *window) CapturePage(cb func([]byte, error)) {
	withPool(func() {
		blk := newBlock(func(_ objc.Block, img id, err id) {
			if err != 0 || img == 0 {
				cb(nil, nsError(err))
				return
			}
			withPool(func() { cb(pngFromImage(img), nil) })
		})
		defer blk.Release()
		send(w.web, "takeSnapshotWithConfiguration:completionHandler:", 0, uintptr(blk))
	})
}

// pngFromImage encodes an NSImage as PNG.
func pngFromImage(img id) []byte {
	tiff := send(img, "TIFFRepresentation")
	rep := send(class("NSBitmapImageRep"), "imageRepWithData:", uintptr(tiff))
	if rep == 0 {
		return nil
	}
	data := send(rep, "representationUsingType:properties:", 4, uintptr(send(class("NSDictionary"), "dictionary")))
	return goBytes(data)
}

// securityOrigin renders a WKSecurityOrigin like a URL origin.
func securityOrigin(o id) string {
	origin := goString(send(o, "protocol")) + "://" + goString(send(o, "host"))
	if port := sendInt(o, "port"); port != 0 {
		origin += ":" + strconv.Itoa(port)
	}
	return origin
}

// printJob is a print operation of PrintToPDF, by its context in
// printJobs until it ends. One whose window closes may never end: WebKit
// leaves it waiting for the page, so the window's cleanup gives cb
// errDestroyed, and done releases what the job holds if it ends after all.
type printJob struct {
	w    *window
	cb   func(pdf []byte, err error) // nil once it got its result
	done func(success bool)
}

var printJobs = map[uintptr]*printJob{}

var errDestroyed = errors.New("mygo: window has been destroyed")

func (w *window) PrintToPDF(o platform.PDFOptions, cb func([]byte, error)) {
	if !respondsTo(w.web, "printOperationWithPrintInfo:") {
		cb(nil, errors.New("mygo: printing to PDF needs macOS 11 or later"))
		return
	}
	f, err := os.CreateTemp("", "mygo-*.pdf")
	if err != nil {
		cb(nil, err)
		return
	}
	path := f.Name()
	f.Close()
	withPool(func() {
		info := autorelease(send(send(class("NSPrintInfo"), "sharedPrintInfo"), "copy"))
		send(info, "setJobDisposition:", uintptr(appKitString("NSPrintSaveJob")))
		url := send(class("NSURL"), "fileURLWithPath:", uintptr(nsString(path)))
		send(send(info, "dictionary"), "setObject:forKey:", uintptr(url), uintptr(appKitString("NSPrintJobSavingURL")))
		const points = 72
		width, height := o.PageWidth, o.PageHeight
		orientation := 0 // NSPaperOrientationPortrait
		if o.Landscape {
			width, height, orientation = height, width, 1
		}
		msgSetSize(info, sel("setPaperSize:"), NSSize{width * points, height * points})
		send(info, "setOrientation:", uintptr(orientation))
		msgSetFloat(info, sel("setTopMargin:"), o.MarginTop*points)
		msgSetFloat(info, sel("setRightMargin:"), o.MarginRight*points)
		msgSetFloat(info, sel("setBottomMargin:"), o.MarginBottom*points)
		msgSetFloat(info, sel("setLeftMargin:"), o.MarginLeft*points)
		send(info, "setHorizontalPagination:", 1) // fit the width, like browsers
		send(info, "setVerticalPagination:", 0)   // as many pages as it takes
		send(info, "setHorizontallyCentered:", 0)
		send(info, "setVerticallyCentered:", 0)
		// The preferences are shared with the configuration's copy. They are
		// kept for the job, which may end after the window closed.
		prefs := retain(send(send(w.web, "configuration"), "preferences"))
		backgrounds := respondsTo(prefs, "setShouldPrintBackgrounds:") // macOS 13.3
		printedBackgrounds := false
		if backgrounds {
			printedBackgrounds = sendBool(prefs, "shouldPrintBackgrounds")
			send(prefs, "setShouldPrintBackgrounds:", boolArg(o.Background))
		}
		// Kept until it ran: nothing else holds on to it.
		op := retain(send(w.web, "printOperationWithPrintInfo:", uintptr(info)))
		send(op, "setShowsPrintPanel:", 0)
		send(op, "setShowsProgressPanel:", 0)
		msgSetRect(send(op, "view"), sel("setFrame:"), msgRect(w.web, sel("bounds")))
		job := uintptr(len(printJobs) + 1)
		for printJobs[job] != nil {
			job++
		}
		j := &printJob{w: w, cb: cb}
		j.done = func(success bool) {
			release(op)
			if backgrounds {
				send(prefs, "setShouldPrintBackgrounds:", boolArg(printedBackgrounds))
			}
			release(prefs)
			data, err := os.ReadFile(path)
			os.Remove(path)
			if j.cb == nil {
				return
			}
			if err == nil && (!success || len(data) == 0) {
				err = errors.New("mygo: printing to PDF failed")
			}
			j.cb(data, err)
		}
		printJobs[job] = j
		// The app's delegate hears of the end, as the window's may be gone.
		send(op, "runOperationModalForWindow:delegate:didRunSelector:contextInfo:", uintptr(w.win), uintptr(w.b.delegate),
			uintptr(sel("mygoPrintOperationDidRun:success:contextInfo:")), job)
	})
}

func (w *window) Print() {
	if !respondsTo(w.web, "printOperationWithPrintInfo:") {
		return
	}
	withPool(func() {
		info := send(class("NSPrintInfo"), "sharedPrintInfo")
		// Kept until it ran: nothing else holds on to it.
		op := retain(send(w.web, "printOperationWithPrintInfo:", uintptr(info)))
		view := send(op, "view")
		msgSetRect(view, sel("setFrame:"), msgRect(w.web, sel("bounds")))
		send(op, "runOperationModalForWindow:delegate:didRunSelector:contextInfo:", uintptr(w.win), 0, 0, 0)
	})
}

// Coordinates: AppKit puts the origin at the bottom-left of the primary
// screen, MyGo at its top-left.

func primaryScreenHeight() float64 {
	screens := send(class("NSScreen"), "screens")
	if sendInt(screens, "count") == 0 {
		return 0
	}
	return msgRect(send(screens, "objectAtIndex:", 0), sel("frame")).Size.Height
}

func rectFromMac(r NSRect) platform.Rect {
	h := primaryScreenHeight()
	return platform.Rect{
		X:      int(math.Round(r.Origin.X)),
		Y:      int(math.Round(h - r.Origin.Y - r.Size.Height)),
		Width:  int(math.Round(r.Size.Width)),
		Height: int(math.Round(r.Size.Height)),
	}
}

func rectToMac(r platform.Rect) NSRect {
	h := primaryScreenHeight()
	return NSRect{
		Origin: NSPoint{float64(r.X), h - float64(r.Y) - float64(r.Height)},
		Size:   NSSize{float64(r.Width), float64(r.Height)},
	}
}

// draggedFiles returns the paths of the files on the pasteboard of a
// drag.
func draggedFiles(info id) []string {
	var paths []string
	withPool(func() {
		opts := send(class("NSDictionary"), "dictionaryWithObject:forKey:", uintptr(nsBool(true)), uintptr(nsString("NSPasteboardURLReadingFileURLsOnlyKey")))
		urls := send(send(info, "draggingPasteboard"), "readObjectsForClasses:options:", uintptr(nsArray(class("NSURL"))), uintptr(opts))
		for _, u := range arrayItems(urls) {
			if p := goString(send(send(u, "filePathURL"), "path")); p != "" {
				paths = append(paths, p)
			}
		}
	})
	return paths
}

func (w *window) DroppedFiles() []string {
	paths := w.dropped
	w.dropped = nil
	return paths
}

func (b *Backend) windowFor(delegate id) *window {
	w := b.byDelegate[delegate]
	if w == nil || w.closed {
		return nil
	}
	return w
}

func registerWindowClasses() {
	// WebKit moves the keyboard out of a page past its ends through the
	// window's key view loop: a web view in native UI gives it back to
	// the content.
	selectKeyView := func(class string, back bool) func(id, objc.SEL, id) {
		return func(self id, cmd objc.SEL, view id) {
			if w := webViewOf(view); w != nil && w.host != nil {
				w.tabOut(back)
				return
			}
			sendSuper(self, class, cmd, uintptr(view))
		}
	}
	classDef("MyGoWindow", "NSWindow", nil, []objc.MethodDef{
		method("canBecomeKeyWindow", func(self id, _ objc.SEL) bool { return true }),
		method("canBecomeMainWindow", func(self id, _ objc.SEL) bool { return true }),
		method("selectKeyViewFollowingView:", selectKeyView("MyGoWindow", false)),
		method("selectKeyViewPrecedingView:", selectKeyView("MyGoWindow", true)),
	})

	// A panel (WindowOptions.Panel) takes the keyboard, without
	// activating the app, and is never the main window.
	classDef("MyGoPanel", "NSPanel", nil, []objc.MethodDef{
		method("canBecomeKeyWindow", func(self id, _ objc.SEL) bool { return true }),
		method("canBecomeMainWindow", func(self id, _ objc.SEL) bool { return false }),
		// NSPanel closes on an Escape its content leaves (cancelOperation:),
		// past windowShouldClose: MyGo closes it then, as the user closing
		// it, which OnClose listeners may prevent.
		method("cancelOperation:", func(self id, _ objc.SEL, sender id) {}),
		method("selectKeyViewFollowingView:", selectKeyView("MyGoPanel", false)),
		method("selectKeyViewPrecedingView:", selectKeyView("MyGoPanel", true)),
	})

	// A non-activating window (WindowOptions.NonActivating) is a panel
	// that never becomes the key or the main window: clicks reach its
	// content and leave the app in front, and its first responder alone.
	classDef("MyGoNonactivatingPanel", "NSPanel", nil, []objc.MethodDef{
		method("canBecomeKeyWindow", func(self id, _ objc.SEL) bool { return false }),
		method("canBecomeMainWindow", func(self id, _ objc.SEL) bool { return false }),
	})

	// A flyout's panel becomes the key window only when it takes the
	// keyboard, and never the main window: its parent stays it.
	classDef("MyGoFlyout", "NSPanel", nil, []objc.MethodDef{
		method("canBecomeKeyWindow", func(self id, _ objc.SEL) bool {
			w := theBackend.byNSWindow[self]
			return w != nil && w.flyout.Focusable
		}),
		method("canBecomeMainWindow", func(self id, _ objc.SEL) bool { return false }),
		method("selectKeyViewFollowingView:", selectKeyView("MyGoFlyout", false)),
		method("selectKeyViewPrecedingView:", selectKeyView("MyGoFlyout", true)),
	})

	registerClipViewClass()
	classDef("MyGoWebView", "WKWebView", nil, append(webViewMethods(), []objc.MethodDef{
		method("mouseDown:", func(self id, cmd objc.SEL, ev id) {
			if w := theBackend.byWebView[self]; w != nil {
				if w.host != nil {
					w.pressed(ev, 0)
				}
				// A web view's page drags the window it is in (StartDrag).
				w = w.top()
				release(w.lastMouseDown)
				w.lastMouseDown = retain(ev)
			}
			sendSuper(self, "MyGoWebView", cmd, uintptr(ev))
		}),
		// Files dropped on the page: the page gets File objects, the app
		// their paths.
		method("performDragOperation:", func(self id, cmd objc.SEL, info id) bool {
			if w := theBackend.byWebView[self]; w != nil {
				w.dropped = draggedFiles(info)
			}
			return byte(sendSuper(self, "MyGoWebView", cmd, uintptr(info))) != 0
		}),
	}...))

	b := func() *Backend { return theBackend }
	classDef("MyGoWindowDelegate", "NSObject",
		[]string{"NSWindowDelegate", "WKNavigationDelegate", "WKUIDelegate", "WKScriptMessageHandler", "WKDownloadDelegate", "NSPopoverDelegate"},
		append(popoverDelegateMethods(), []objc.MethodDef{
			// NSWindowDelegate
			method("windowShouldClose:", func(self id, _ objc.SEL, sender id) bool {
				if w := b().windowFor(self); w != nil {
					return w.h.ShouldClose()
				}
				return true
			}),
			// WKUIDelegate (macOS 12): camera and microphone requests.
			method("webView:requestMediaCapturePermissionForOrigin:initiatedByFrame:type:decisionHandler:", func(self id, _ objc.SEL, web, origin, frame id, typ int, handler uintptr) {
				const grant, deny = 1, 2 // WKPermissionDecision
				decision := deny
				if w := b().windowFor(self); w != nil {
					kinds := map[int][]string{0: {"camera"}, 1: {"microphone"}, 2: {"camera", "microphone"}}[typ]
					if w.h.PermissionRequested(kinds, securityOrigin(origin)) {
						decision = grant
					}
				}
				callBlock(handler, uintptr(decision))
			}),
			method("windowWillClose:", func(self id, _ objc.SEL, n id) {
				if w := b().windowFor(self); w != nil {
					w.cleanup()
					w.h.Closed()
				}
			}),
			method("windowDidBecomeKey:", func(self id, _ objc.SEL, n id) {
				if w := b().windowFor(self); w != nil {
					w.h.Focused()
					w.surfaceKeyChanged(true)
				}
			}),
			method("windowDidResignKey:", func(self id, _ objc.SEL, n id) {
				if w := b().windowFor(self); w != nil {
					w.h.Blurred()
					w.surfaceKeyChanged(false)
					if w.opts.Flyout != nil && w.popover == 0 {
						// Once AppKit made another window key, or none,
						// unless the press on its tray icon toggles it.
						if t, ok := w.flyout.Tray.(*tray); !ok || !t.pressed() {
							b().post(w.dismissFlyout)
						}
					}
				}
			}),
			method("windowDidResize:", func(self id, _ objc.SEL, n id) {
				w := b().windowFor(self)
				if w == nil {
					return
				}
				w.layoutTrafficLights()
				if z := w.IsMaximized(); z != w.maximized {
					w.maximized = z
					if z {
						w.h.Maximized()
					} else {
						w.h.Unmaximized()
					}
				}
				w.h.Resized()
			}),
			method("windowDidMove:", func(self id, _ objc.SEL, n id) {
				if w := b().windowFor(self); w != nil {
					w.h.Moved()
				}
			}),
			method("windowDidChangeOcclusionState:", func(self id, _ objc.SEL, n id) {
				if w := b().windowFor(self); w != nil && w.surface != nil && w.surface.visible() {
					w.surface.send(platform.SurfaceEvent{Kind: platform.SurfaceShown})
				}
			}),
			method("windowDidMiniaturize:", func(self id, _ objc.SEL, n id) {
				if w := b().windowFor(self); w != nil {
					w.h.Minimized()
				}
			}),
			method("windowDidDeminiaturize:", func(self id, _ objc.SEL, n id) {
				if w := b().windowFor(self); w != nil {
					w.h.Restored()
				}
			}),
			// The toolbar that insets the traffic lights holds no items: in
			// full screen it hides with the menu bar instead of covering the
			// top of the page.
			method("window:willUseFullScreenPresentationOptions:", func(self id, _ objc.SEL, win id, proposed uint) uint {
				const autoHide = presentationFullScreen | presentationAutoHideMenuBar
				if send(win, "toolbar") != 0 && proposed&autoHide == autoHide {
					return proposed | presentationAutoHideToolbar
				}
				return proposed
			}),
			method("windowDidEnterFullScreen:", func(self id, _ objc.SEL, n id) {
				if w := b().windowFor(self); w != nil {
					w.h.EnteredFullScreen()
					if w.hiddenTitleBar() {
						w.h.TitleBarChanged() // the traffic lights hid
					}
				}
			}),
			method("windowDidExitFullScreen:", func(self id, _ objc.SEL, n id) {
				if w := b().windowFor(self); w != nil {
					w.layoutTrafficLights()
					w.h.LeftFullScreen()
					if w.hiddenTitleBar() {
						w.h.TitleBarChanged()
					}
				}
			}),

			// WKScriptMessageHandler
			method("userContentController:didReceiveScriptMessage:", func(self id, _ objc.SEL, ucc, msg id) {
				w := b().windowFor(self)
				if w == nil {
					return
				}
				// The handler is reachable from every frame; only the
				// page's own bridge may talk to Go.
				if frame := send(msg, "frameInfo"); frame == 0 || !sendBool(frame, "isMainFrame") {
					return
				}
				body := send(msg, "body")
				if body == 0 || !sendBool(body, "isKindOfClass:", uintptr(class("NSString"))) {
					return
				}
				w.h.Message(goString(body))
			}),

			// Key-value observing of the page title and the window's
			// appearance.
			method("observeValueForKeyPath:ofObject:change:context:", func(self id, _ objc.SEL, keyPath, object, change id, ctx uintptr) {
				w := b().windowFor(self)
				if w == nil {
					return
				}
				switch goString(keyPath) {
				case "title":
					w.h.TitleChanged(goString(send(w.web, "title")))
				case "effectiveAppearance":
					// AppKit lays the title bar out again after telling us.
					b().post(w.layoutTrafficLights)
				}
			}),

			// WKNavigationDelegate
			method("webView:decidePolicyForNavigationAction:decisionHandler:", func(self id, _ objc.SEL, web, action id, handler uintptr) {
				// A link with the download attribute (macOS 11.3).
				if respondsTo(action, "shouldPerformDownload") && sendBool(action, "shouldPerformDownload") {
					callBlock(handler, 2) // WKNavigationActionPolicyDownload
					return
				}
				allow := true
				if w := b().windowFor(self); w != nil {
					frame := send(action, "targetFrame")
					navType := sendInt(action, "navigationType")
					// WKNavigationType: link, form, back/forward, reload, …
					nav := platform.Navigation{
						URL:           goString(send(send(send(action, "request"), "URL"), "absoluteString")),
						IsMainFrame:   frame != 0 && sendBool(frame, "isMainFrame"),
						UserInitiated: navType == 0 || navType == 1,
						Unasked:       navType == 2 || navType == 3,
						Reload:        navType == 3,
					}
					if nav.IsMainFrame && w.programmatic {
						w.programmatic = false
						nav.Unasked = true
					}
					allow = w.h.WillNavigate(nav)
				}
				callBlock(handler, boolArg(allow))
			}),
			// Responses the page cannot show, or sent as attachments, are
			// downloads (macOS 11.3).
			method("webView:decidePolicyForNavigationResponse:decisionHandler:", func(self id, _ objc.SEL, web, nav id, handler uintptr) {
				const cancel, allow, download = 0, 1, 2 // WKNavigationResponsePolicy
				resp := send(nav, "response")
				url := goString(send(send(resp, "URL"), "absoluteString"))
				disposition, custom := schemeDispositions[url]
				if !custom && respondsTo(resp, "allHeaderFields") {
					disposition = goString(send(send(resp, "allHeaderFields"), "objectForKey:", uintptr(nsString("Content-Disposition"))))
				}
				attachment := strings.HasPrefix(strings.ToLower(strings.TrimSpace(disposition)), "attachment")
				switch {
				case !attachment && sendBool(nav, "canShowMIMEType"):
					callBlock(handler, allow)
				case custom:
					// WebKit cannot turn what a scheme handler serves into a
					// download: serve it again into one.
					callBlock(handler, cancel)
					if w := b().windowFor(self); w != nil {
						w.h.SchemeDownload(url)
					}
				case hasClass("WKDownload"):
					callBlock(handler, download)
				default:
					callBlock(handler, cancel)
				}
			}),
			method("webView:navigationAction:didBecomeDownload:", func(self id, _ objc.SEL, web, action, download id) {
				send(download, "setDelegate:", uintptr(self))
			}),
			method("webView:navigationResponse:didBecomeDownload:", func(self id, _ objc.SEL, web, nav, download id) {
				send(download, "setDelegate:", uintptr(self))
			}),
			// WKDownloadDelegate.
			method("download:decideDestinationUsingResponse:suggestedFilename:completionHandler:", func(self id, _ objc.SEL, download, resp, suggested id, handler uintptr) {
				w := b().windowFor(self)
				if w == nil {
					callBlock(handler, 0)
					return
				}
				url := goString(send(send(send(download, "originalRequest"), "URL"), "absoluteString"))
				path := w.h.DownloadStarted(url, goString(suggested))
				if path == "" {
					callBlock(handler, 0) // cancels the download
					return
				}
				os.Remove(path) // WebKit refuses to replace a file
				w.downloads[download] = [2]string{url, path}
				withPool(func() { callBlock(handler, uintptr(send(class("NSURL"), "fileURLWithPath:", uintptr(nsString(path))))) })
			}),
			method("downloadDidFinish:", func(self id, _ objc.SEL, download id) {
				if w := b().windowFor(self); w != nil {
					if d, ok := w.downloads[download]; ok {
						delete(w.downloads, download)
						w.h.DownloadFinished(d[0], d[1], nil)
					}
				}
			}),
			method("download:didFailWithError:resumeData:", func(self id, _ objc.SEL, download, nserr, resume id) {
				if w := b().windowFor(self); w != nil {
					if d, ok := w.downloads[download]; ok {
						delete(w.downloads, download)
						w.h.DownloadFinished(d[0], d[1], errors.New(goString(send(nserr, "localizedDescription"))))
					}
				}
			}),
			method("webView:didStartProvisionalNavigation:", func(self id, _ objc.SEL, web, nav id) {
				if w := b().windowFor(self); w != nil {
					w.h.NavigationStarted(w.URL())
				}
			}),
			method("webView:didCommitNavigation:", func(self id, _ objc.SEL, web, nav id) {
				if w := b().windowFor(self); w != nil {
					w.h.NavigationCommitted(w.URL())
				}
			}),
			method("webView:didFinishNavigation:", func(self id, _ objc.SEL, web, nav id) {
				if w := b().windowFor(self); w != nil {
					w.h.LoadFinished()
				}
			}),
			method("webView:didFailProvisionalNavigation:withError:", func(self id, _ objc.SEL, web, nav, err id) {
				if w := b().windowFor(self); w != nil {
					w.loadFailed(err)
				}
			}),
			method("webView:didFailNavigation:withError:", func(self id, _ objc.SEL, web, nav, err id) {
				if w := b().windowFor(self); w != nil {
					w.loadFailed(err)
				}
			}),
			method("webViewWebContentProcessDidTerminate:", func(self id, _ objc.SEL, web id) {
				if w := b().windowFor(self); w != nil {
					w.h.RenderProcessGone("terminated")
				}
			}),

			// WKUIDelegate
			method("webView:createWebViewWithConfiguration:forNavigationAction:windowFeatures:", func(self id, _ objc.SEL, web, cfg, action, features id) id {
				w := b().windowFor(self)
				if w == nil {
					return 0
				}
				num := func(key string) int {
					if n := send(features, key); n != 0 {
						return sendInt(n, "integerValue")
					}
					return 0
				}
				req := platform.NewWindowRequest{
					URL:    goString(send(send(send(action, "request"), "URL"), "absoluteString")),
					X:      num("x"),
					Y:      num("y"),
					Width:  num("width"),
					Height: num("height"),
					Native: uintptr(cfg),
				}
				child, ok := w.h.NewWindow(req).(*window)
				if !ok || child == nil {
					return 0
				}
				return child.web
			}),
			method("webViewDidClose:", func(self id, _ objc.SEL, web id) {
				if w := b().windowFor(self); w != nil {
					w.h.ClosedByPage()
				}
			}),
			method("webView:runJavaScriptAlertPanelWithMessage:initiatedByFrame:completionHandler:", func(self id, _ objc.SEL, web, message, frame id, handler uintptr) {
				w := b().windowFor(self)
				jsAlert(w, goString(message), handler)
			}),
			method("webView:runJavaScriptConfirmPanelWithMessage:initiatedByFrame:completionHandler:", func(self id, _ objc.SEL, web, message, frame id, handler uintptr) {
				w := b().windowFor(self)
				jsConfirm(w, goString(message), handler)
			}),
			method("webView:runJavaScriptTextInputPanelWithPrompt:defaultText:initiatedByFrame:completionHandler:", func(self id, _ objc.SEL, web, prompt, defaultText, frame id, handler uintptr) {
				w := b().windowFor(self)
				jsPrompt(w, goString(prompt), goString(defaultText), handler)
			}),
			method("webView:runOpenPanelWithParameters:initiatedByFrame:completionHandler:", func(self id, _ objc.SEL, web, params, frame id, handler uintptr) {
				w := b().windowFor(self)
				jsFileInput(w, params, handler)
			}),
		}...))
}

func (w *window) loadFailed(err id) {
	code := sendInt(err, "code")
	// Cancelled (-999) and "frame load interrupted" (102) happen when a
	// navigation is replaced or denied; they are not failures.
	if code == -999 || code == 102 {
		return
	}
	var url, desc string
	withPool(func() {
		info := send(err, "userInfo")
		url = goString(send(info, "objectForKey:", uintptr(nsString("NSErrorFailingURLStringKey"))))
		desc = goString(send(err, "localizedDescription"))
	})
	w.h.LoadFailed(url, code, desc)
}
