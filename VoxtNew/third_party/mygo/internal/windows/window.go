//go:build windows && (amd64 || arm64)

package windows

import (
	"errors"
	"unsafe"

	"github.com/egoist/mygo/internal/platform"
)

var errDestroyed = errors.New("mygo: window has been destroyed")

type window struct {
	b      *Backend
	h      platform.WindowHandler
	opts   *platform.WindowOptions
	hwnd   uintptr
	parent *window
	// flyout is how a flyout (WindowOptions.Flyout) was placed last, at
	// flyoutSize.
	flyout     platform.Flyout
	flyoutSize platform.Size

	// The WebView2 controller is created asynchronously; calls that need
	// it wait in pending until it exists, or fail with webViewErr, why it
	// could not be created.
	controller   uintptr // ICoreWebView2Controller
	webview      uintptr // ICoreWebView2
	settings     uintptr // ICoreWebView2Settings
	ready        bool
	pending      []pendingCall
	webViewErr   error
	webViewFails int // creations that failed
	// scriptIDs are the ids of the scripts added to every document, which
	// SetUserScripts removes; scriptGen counts the replacements, so an id
	// that arrives after one is removed at once.
	scriptIDs  []string
	scriptGen  int
	closed     bool
	destroying bool

	minW, minH, maxW, maxH int // DIPs
	movable, closable      bool
	frameless              bool
	hiddenTitleBar         bool        // a TitleBarStyle that hides the caption
	caption                *captionBar // the controls in its place, nil if they failed
	noRedirect             bool        // no redirection bitmap, for a material behind the page
	fullScreen             bool
	reframing              bool     // SetFullScreen is changing the frame
	showMaximized          bool     // on the first show (WindowOptions.Maximized)
	dropped                []string // DroppedFiles
	skipTaskbar            bool
	progress               struct {
		state string
		value float64
	}
	icons [2]uintptr // small and big, from SetIcon
	// surface shows the content MyGo draws, in place of the webview.
	surface *surface
	// webViews are the web views under the surface (webview_embed.go), and
	// host the window whose surface a web view is under, which shows it
	// through its composition controller comp (comp3 its drag and drop,
	// on runtimes having it; uia the provider of its page for UI
	// Automation), in the visual webVis inside clipVis, at frame clipped
	// to clip while shown. focused tells that it has the keyboard, and
	// cursor is the cursor its page wants.
	webViews         []*window
	host             *window
	comp, comp3, uia uintptr
	clipVis, webVis  uintptr
	frame, clip      platform.RectF
	shown, focused   bool
	cursor           uintptr
	saved            struct {
		style, exStyle uintptr
		placement      windowPlacement
	}
	state       int // last sizeRestored, sizeMinimized or sizeMaximized
	opacity     float64
	bgBrush     uintptr
	bg          *platform.Color
	vibrancy    string
	shadow      bool
	ignoreMouse bool

	menu    *platform.Menu // menu bar
	hmenu   uintptr
	owner   int
	ownMenu bool
	accels  map[accelKey]uint16
	// autoHideMenu keeps hmenu off the window except while revealed, when
	// the keyboard is in the menu bar.
	autoHideMenu, revealed bool
	inMenu                 bool    // in a menu loop, of the bar or a popup
	menuKey                menuKey // the letter that opened popupMenuBar

	programmatic bool
	loading      bool
	zoom         float64
	devTools     bool
	htmlFor      map[string]string // LoadHTML documents by the URL they load at
	htmlNext     bool              // LoadHTML started the next navigation
	calls        map[int]func(string, error)
	nextCall     int
}

func (b *Backend) NewWindow(o *platform.WindowOptions, h platform.WindowHandler) (platform.Window, error) {
	if !o.Surface {
		if err := b.startEnvironment(); err != nil {
			return nil, err
		}
	}
	w := &window{
		b: b, h: h, opts: o,
		movable: o.Movable, closable: o.Closable, frameless: o.Frameless,
		opacity: 1, zoom: 1, shadow: o.HasShadow, autoHideMenu: o.AutoHideMenu,
		minW: o.MinSize.Width, minH: o.MinSize.Height, maxW: o.MaxSize.Width, maxH: o.MaxSize.Height,
		htmlFor: map[string]string{}, calls: map[int]func(string, error){},
	}
	w.hiddenTitleBar = !o.Frameless && (o.TitleBarStyle == "hidden" || o.TitleBarStyle == "hiddenInset")
	// A page or native UI with a material behind it needs a window without
	// a redirection bitmap, whose opaque surface would cover the system
	// backdrop. GDI draws nothing there: the controls of a hidden title bar
	// and the frames of native UI show through DirectComposition
	// (compositor.go, and the renderer's swap chain), and the menus open in
	// a popup (barless). Where DirectComposition is unavailable, the window
	// keeps its bitmap: what it shows shows, and the material does not. The
	// style is the window's for life: Windows neither adds it nor removes it
	// later.
	w.noRedirect = o.Vibrancy != "" && systemBackdrops() && (!w.hiddenTitleBar && !o.Surface || b.composition() != nil)
	if o.Flyout != nil {
		// What the content leaves transparent shows what is behind.
		w.noRedirect = b.composition() != nil
	}
	var owner uintptr
	if p, ok := o.Parent.(*window); ok && p != nil && !p.closed {
		w.parent, owner = p, p.hwnd
	} else if o.Panel {
		// A tool window alone may still get a taskbar button, on the
		// taskbars of other displays: an owned one never does. The
		// application window is hidden for good.
		owner = b.appHwnd
	}
	style, ex := w.styles()
	class := windowClass
	if o.Flyout != nil {
		style, ex = w.flyoutStyles()
		if o.HasShadow {
			class = flyoutShadowClass
		}
	}
	w.hwnd = createWindow(ex, class, o.Title, style, 0, 0, 0, 0, owner)
	if w.hwnd == 0 {
		return nil, errors.New("mygo: cannot create a window")
	}
	b.windows[w.hwnd] = w
	if o.Panel {
		// Windows 11 gives tool windows the small corners of menus.
		w.roundFlyout()
	}
	if o.Flyout != nil {
		if o.HasShadow {
			w.roundFlyout()
		}
		w.PlaceFlyout(*o.Flyout, platform.Size{Width: o.Width, Height: o.Height})
	} else {
		w.placeInitially()
	}

	if !o.Closable {
		w.SetClosable(false)
	}
	if o.AlwaysOnTop {
		w.SetAlwaysOnTop(true)
	}
	if o.Opacity > 0 && o.Opacity < 1 {
		w.SetOpacity(o.Opacity)
	}
	if o.BackgroundColor != nil {
		w.SetBackgroundColor(*o.BackgroundColor)
	}
	if w.captionless() && w.shadow && o.Flyout == nil {
		w.SetHasShadow(true)
	}
	if w.hiddenTitleBar {
		w.caption = newCaptionBar(w)
	}
	if o.Surface {
		if w.surface = newSurface(w); w.surface == nil {
			procDestroyWindow.Call(w.hwnd)
			return nil, errors.New("mygo: cannot create the window's surface")
		}
	}
	b.applyWindowTheme(w)
	if o.Vibrancy != "" {
		w.SetVibrancy(o.Vibrancy)
	}
	if b.icon != 0 {
		w.setIcon(b.icon)
	}
	if b.appMenu != nil && w.hasAppMenu() {
		w.installMenu(b.appMenu)
	}
	if o.Modal && w.parent != nil {
		procEnableWindow.Call(w.parent.hwnd, 0)
	}
	if o.FullScreen {
		w.SetFullScreen(true)
	} else if o.Maximized {
		w.showMaximized = true
	}
	w.skipTaskbar = o.SkipTaskbar // applied once the taskbar button exists
	if w.caption != nil {
		w.caption.layout()
	}
	if !o.Surface {
		b.whenEnvironment(w.createWebView)
	}
	return w, nil
}

func (w *window) styles() (style, ex uint32) {
	o := w.opts
	style = wsOverlappedWindow | wsClipChildren
	if !o.Resizable {
		style &^= wsThickFrame | wsMaximizeBox
	}
	if !o.Minimizable {
		style &^= wsMinimizeBox
	}
	if !o.Maximizable {
		style &^= wsMaximizeBox
	}
	if !o.Focusable {
		ex |= wsExNoActivate
	}
	if o.Panel {
		// Out of the taskbar and Alt+Tab, on whichever virtual desktop
		// is current as it shows, and above other apps' windows. Without
		// a window menu: DWM draws the close button of a tool window's
		// menu where a material extends the frame, over the content.
		ex |= wsExToolWindow | wsExTopmost
		style &^= wsSysMenu
	}
	if w.noRedirect {
		ex |= wsExNoRedirect
	}
	return style, ex
}

// placeInitially sizes and positions a new window, in the DPI of the
// monitor it appears on.
func (w *window) placeInitially() {
	o := w.opts
	var mon uintptr
	if o.Center {
		mon, _, _ = procMonitorFromPoint.Call(0, monitorDefaultToPrimary)
	} else {
		mon = monitorAt(o.X, o.Y)
	}
	dpi := monitorDPI(mon)
	width, height := toPx(o.Width, dpi), toPx(o.Height, dpi)
	if o.UseContentSize && !w.captionless() {
		width, height = w.outerSize(width, height, dpi)
	}
	x, y := toPx(o.X, dpi), toPx(o.Y, dpi)
	if o.Center {
		work := monitorWorkArea(mon)
		x = work.Left + (work.Right-work.Left-width)/2
		y = work.Top + (work.Bottom-work.Top-height)/2
	}
	procSetWindowPos.Call(w.hwnd, 0, uintptr(x), uintptr(y), uintptr(width), uintptr(height), swpNoZOrder|swpNoActivate)
}

// outerSize returns the window size for a content size, in pixels.
func (w *window) outerSize(width, height int32, dpi int) (int32, int32) {
	r := rect{0, 0, width, height}
	menu := uintptr(0)
	if !w.autoHideMenu && !w.noRedirect && (w.hmenu != 0 || w.b.appMenu != nil && w.hasAppMenu()) {
		menu = 1
	}
	style, ex := windowLong(w.hwnd, gwlStyle), windowLong(w.hwnd, gwlExStyle)
	if has(procAdjustWindowRectExForDpi) {
		procAdjustWindowRectExForDpi.Call(uintptr(unsafe.Pointer(&r)), style, menu, ex, uintptr(dpi))
	}
	return r.Right - r.Left, r.Bottom - r.Top
}

func monitorAt(x, y int) uintptr {
	pt := uintptr(uint32(int32(x))) | uintptr(uint32(int32(y)))<<32
	m, _, _ := procMonitorFromPoint.Call(pt, monitorDefaultToNearest)
	return m
}

func monitorDPI(mon uintptr) int {
	if mon != 0 && has(procGetDpiForMonitor) {
		var dx, dy uint32
		if r, _, _ := procGetDpiForMonitor.Call(mon, 0, uintptr(unsafe.Pointer(&dx)), uintptr(unsafe.Pointer(&dy))); r == 0 && dx != 0 {
			return int(dx)
		}
	}
	return dpiOf(0)
}

func monitorInfo(mon uintptr) monitorInfoEx {
	var mi monitorInfoEx
	mi.Size = uint32(unsafe.Sizeof(mi))
	procGetMonitorInfoW.Call(mon, uintptr(unsafe.Pointer(&mi)))
	return mi
}

func monitorWorkArea(mon uintptr) rect { return monitorInfo(mon).Work }

func (w *window) monitor() uintptr {
	m, _, _ := procMonitorFromWindow.Call(w.hwnd, monitorDefaultToNearest)
	return m
}

// message handles a window message; ok false lets DefWindowProc run.
func (w *window) message(m uint32, wp, lp uintptr) (uintptr, bool) {
	if m == w.b.taskbarButtonCreated && m != 0 {
		w.applyTaskbar()
		return 0, true
	}
	if w.opts.Flyout != nil {
		if r, ok := w.flyoutMessage(m, wp, lp); ok {
			return r, true
		}
	}
	if m == wmMouseActivate && w.ownsFocusableFlyout() {
		return maActivateAndEat, true
	}
	if r, ok := w.darkMenuMessage(m, wp, lp); ok {
		return r, true
	}
	switch m {
	case wmClose:
		if w.h.ShouldClose() {
			w.destroy()
		}
		return 0, true
	case wmDestroy:
		w.cleanup()
		w.h.Closed()
		return 0, true
	case wmTimer:
		if wp != timerWebView {
			return 0, false
		}
		procKillTimer.Call(w.hwnd, timerWebView)
		w.createWebView()
		return 0, true
	case wmSize:
		if !w.reframing { // SetFullScreen lays the window out once it is done
			w.sized(wp)
		}
		return 0, true
	case wmMove:
		if w.controller != 0 {
			comCall(w.controller, ctlNotifyParentWindowPositionChanged)
		}
		for _, v := range w.webViews {
			if v.controller != 0 {
				comCall(v.controller, ctlNotifyParentWindowPositionChanged)
			}
		}
		w.h.Moved()
		return 0, true
	case wmActivate:
		if w.caption != nil {
			w.caption.activate(loword(wp) != waInactive)
		}
		if loword(wp) == waInactive {
			w.h.Blurred()
		} else {
			w.h.Focused()
		}
		return 0, false
	case wmSetFocus:
		if w.surface != nil {
			procSetFocus.Call(w.surface.hwnd)
		} else {
			w.focusWebView()
		}
		return 0, true
	case wmGetMinMaxInfo:
		info := (*minMaxInfo)(native(lp))
		dpi := dpiOf(w.hwnd)
		if w.minW > 0 || w.minH > 0 {
			info.MinTrackSize = point{toPx(w.minW, dpi), toPx(w.minH, dpi)}
		}
		if w.maxW > 0 {
			info.MaxTrackSize.X = toPx(w.maxW, dpi)
		}
		if w.maxH > 0 {
			info.MaxTrackSize.Y = toPx(w.maxH, dpi)
		}
		return 0, true
	case wmDpiChanged:
		r := (*rect)(native(lp))
		procSetWindowPos.Call(w.hwnd, 0, uintptr(r.Left), uintptr(r.Top), uintptr(r.Right-r.Left), uintptr(r.Bottom-r.Top), swpNoZOrder|swpNoActivate)
		return 0, true
	case wmNCCalcSize:
		if w.captionless() && wp != 0 && !w.fullScreen {
			return w.frameCalcSize(wp, lp), true
		}
	case wmEraseBkgnd:
		if w.bgBrush != 0 && w.vibrancy == "" {
			var r rect
			procGetClientRect.Call(w.hwnd, uintptr(unsafe.Pointer(&r)))
			procFillRect.Call(wp, uintptr(unsafe.Pointer(&r)), w.bgBrush)
		}
		return 1, true
	case wmWindowPosChanging:
		if !w.movable {
			pos := (*windowPos)(native(lp))
			pos.Flags |= swpNoMove
		}
		return 0, false
	case wmCommand:
		if hiword(wp) == 0 && lp == 0 { // a menu item
			w.b.menuCommand(loword(wp), w)
			return 0, true
		}
	case wmEnterMenuLoop:
		w.inMenu = true
	case wmExitMenuLoop:
		w.inMenu = false
	case wmMenuChar:
		if r, ok := w.menuChar(wp, lp); ok {
			return r, true
		}
	case wmSysCommand:
		// Alt or F10 alone (lp 0) take the keyboard to the menu bar, and Alt
		// and a letter (lp the letter) open the menu of the letter; Alt and
		// Space open the window menu. WebView2 passes no Alt+letter on: the
		// letters come from native UI only.
		if wp&0xFFF0 == scKeyMenu && lp != ' ' && w.hmenu != 0 {
			if w.barless() {
				w.popupMenuBar(lp)
				return 0, true
			}
			if lp == 0 && w.autoHideMenu && !w.revealed {
				return w.revealMenu(wp, lp), true
			}
		}
	}
	return 0, false
}

// sized lays the window out after it changed size or state (a WM_SIZE kind),
// and tells the app.
func (w *window) sized(kind uintptr) {
	w.resizeWebView()
	if w.surface != nil {
		w.surface.fit()
	}
	if w.caption != nil && kind != sizeMinimized {
		w.caption.layout()
	}
	switch kind {
	case sizeMinimized:
		if w.state != sizeMinimized {
			w.state = sizeMinimized
			w.h.Minimized()
		}
	case sizeMaximized:
		if w.state != sizeMaximized {
			if w.state == sizeMinimized {
				w.h.Restored()
			}
			w.state = sizeMaximized
			w.h.Maximized()
		}
	case sizeRestored:
		switch w.state {
		case sizeMinimized:
			w.h.Restored()
		case sizeMaximized:
			w.h.Unmaximized()
		}
		w.state = sizeRestored
	}
	w.h.Resized()
}

// captionless reports a window without a caption: frameless, or with a
// hidden title bar.
func (w *window) captionless() bool { return w.frameless || w.hiddenTitleBar }

// frameCalcSize removes the title bar of windows without a caption.
// Resizable ones keep their left, right and bottom borders, which Windows
// 10 and later draw invisible outside the window, so they still resize.
// Maximized windows would overflow the screen by their borders: fit the
// work area.
//
// A hidden title bar on Windows 11 keeps a pixel of it, which the window's
// border covers: without any non-client area at the top, Windows 11 opens no
// snap layouts over the maximize button of a window that is not maximized.
// Windows 10 has no snap layouts, and over a window that keeps any of its top
// it draws its whole title bar, above the page and the controls.
func (w *window) frameCalcSize(wp, lp uintptr) uintptr {
	params := (*ncCalcSizeParams)(native(lp))
	if w.IsMaximized() {
		// The monitor of the proposed rectangle, not of the window: while a
		// window is restored from minimized it is still the minimized one,
		// parked off screen, whose nearest monitor is not the one it comes
		// back to. Fitting the work area of the wrong monitor leaves the
		// client area outside the window, and the content shows nothing.
		// A rectangle on no monitor keeps the whole window as its client
		// area, as winit and tao do.
		if mon, _, _ := procMonitorFromRect.Call(uintptr(unsafe.Pointer(&params.Rgrc[0])), monitorDefaultToNull); mon != 0 {
			params.Rgrc[0] = monitorWorkArea(mon)
		}
		return 0
	}
	if windowLong(w.hwnd, gwlStyle)&wsThickFrame != 0 {
		top := params.Rgrc[0].Top
		procDefWindowProcW.Call(w.hwnd, wmNCCalcSize, wp, lp)
		params.Rgrc[0].Top = top
		if w.hiddenTitleBar && windows11() {
			params.Rgrc[0].Top++
		}
	}
	return 0
}

func (w *window) destroy() {
	if w.closed || w.destroying {
		return
	}
	w.destroying = true
	// Re-enable the owner first, or Windows activates another app.
	if w.opts.Modal && w.parent != nil && !w.parent.closed {
		procEnableWindow.Call(w.parent.hwnd, 1)
	}
	procDestroyWindow.Call(w.hwnd)
}

func (w *window) cleanup() {
	if w.closed {
		return
	}
	w.closeWebViews()
	w.closed = true
	if w.opts.Modal && w.parent != nil && !w.parent.closed {
		procEnableWindow.Call(w.parent.hwnd, 1)
	}
	for id, cb := range w.calls {
		delete(w.calls, id)
		cb("", errDestroyed)
	}
	w.failPending(errDestroyed)
	if w.caption != nil {
		w.caption.forget()
	}
	if w.controller != 0 {
		comCall(w.controller, ctlClose)
		release(w.settings)
		release(w.webview)
		release(w.controller)
		w.controller, w.webview, w.settings = 0, 0, 0
	}
	w.b.menus.drop(w.owner)
	// Windows destroys the menu bar on the window, not one hidden off it.
	if w.hmenu != 0 && !w.menuShown() {
		procDestroyMenu.Call(w.hmenu)
		w.hmenu = 0
	}
	if w.bgBrush != 0 {
		procDeleteObject.Call(w.bgBrush)
		w.bgBrush = 0
	}
	w.freeIcons()
	delete(w.b.windows, w.hwnd)
}

func (w *window) Handle() uintptr        { return w.hwnd }
func (w *window) WebViewHandle() uintptr { return w.webview }

func (w *window) SetTitle(title string) {
	procSetWindowTextW.Call(w.hwnd, uintptr(unsafe.Pointer(u16(title))))
}

func (w *window) Title() string {
	n, _, _ := procGetWindowTextLengthW.Call(w.hwnd)
	buf := make([]uint16, n+1)
	procGetWindowTextW.Call(w.hwnd, uintptr(unsafe.Pointer(&buf[0])), uintptr(len(buf)))
	return string(utf16Decode(buf))
}

// The core marks the title of an edited document; Windows shows no file.
func (w *window) SetRepresentedFile(string) {}
func (w *window) SetDocumentEdited(bool)    {}

func utf16Decode(b []uint16) []rune {
	for i, c := range b {
		if c == 0 {
			b = b[:i]
			break
		}
	}
	out := make([]rune, 0, len(b))
	for i := 0; i < len(b); i++ {
		c := rune(b[i])
		if c >= 0xD800 && c < 0xDC00 && i+1 < len(b) {
			c = (c-0xD800)<<10 + (rune(b[i+1]) - 0xDC00) + 0x10000
			i++
		}
		out = append(out, c)
	}
	return out
}

func (w *window) rectToDIP(r rect) platform.Rect {
	dpi := dpiOf(w.hwnd)
	return platform.Rect{X: toDIP(r.Left, dpi), Y: toDIP(r.Top, dpi), Width: toDIP(r.Right-r.Left, dpi), Height: toDIP(r.Bottom-r.Top, dpi)}
}

func (w *window) SetBounds(r platform.Rect) {
	dpi := dpiOf(w.hwnd)
	procSetWindowPos.Call(w.hwnd, 0, uintptr(toPx(r.X, dpi)), uintptr(toPx(r.Y, dpi)), uintptr(toPx(r.Width, dpi)), uintptr(toPx(r.Height, dpi)), swpNoZOrder|swpNoActivate)
}

func (w *window) Bounds() platform.Rect {
	var r rect
	procGetWindowRect.Call(w.hwnd, uintptr(unsafe.Pointer(&r)))
	return w.rectToDIP(r)
}

func (w *window) SetContentBounds(r platform.Rect) {
	dpi := dpiOf(w.hwnd)
	outer := rect{toPx(r.X, dpi), toPx(r.Y, dpi), toPx(r.X+r.Width, dpi), toPx(r.Y+r.Height, dpi)}
	if !w.captionless() && has(procAdjustWindowRectExForDpi) {
		menu := uintptr(0)
		if w.menuShown() {
			menu = 1
		}
		procAdjustWindowRectExForDpi.Call(uintptr(unsafe.Pointer(&outer)), windowLong(w.hwnd, gwlStyle), menu, windowLong(w.hwnd, gwlExStyle), uintptr(dpi))
	}
	procSetWindowPos.Call(w.hwnd, 0, uintptr(outer.Left), uintptr(outer.Top), uintptr(outer.Right-outer.Left), uintptr(outer.Bottom-outer.Top), swpNoZOrder|swpNoActivate)
}

func (w *window) ContentBounds() platform.Rect {
	var r rect
	procGetClientRect.Call(w.hwnd, uintptr(unsafe.Pointer(&r)))
	var origin point
	procClientToScreen.Call(w.hwnd, uintptr(unsafe.Pointer(&origin)))
	return w.rectToDIP(rect{origin.X, origin.Y, origin.X + r.Right, origin.Y + r.Bottom})
}

func (w *window) SetMinimumSize(s platform.Size) { w.minW, w.minH = s.Width, s.Height }
func (w *window) SetMaximumSize(s platform.Size) { w.maxW, w.maxH = s.Width, s.Height }

func (w *window) setStyle(bits uint32, on bool) {
	style := windowLong(w.hwnd, gwlStyle)
	if on {
		style |= uintptr(bits)
	} else {
		style &^= uintptr(bits)
	}
	setWindowLong(w.hwnd, gwlStyle, style)
	procSetWindowPos.Call(w.hwnd, 0, 0, 0, 0, 0, swpNoMove|swpNoSize|swpNoZOrder|swpNoActivate|swpFrameChanged)
}

func (w *window) setExStyle(bits uint32, on bool) {
	ex := windowLong(w.hwnd, gwlExStyle)
	if on {
		ex |= uintptr(bits)
	} else {
		ex &^= uintptr(bits)
	}
	setWindowLong(w.hwnd, gwlExStyle, ex)
}

func (w *window) hasStyle(bits uint32) bool { return windowLong(w.hwnd, gwlStyle)&uintptr(bits) != 0 }

func (w *window) SetResizable(v bool) {
	bits := uint32(wsThickFrame)
	if w.opts.Maximizable {
		bits |= wsMaximizeBox
	}
	w.setStyle(bits, v)
	w.captionChanged()
}

func (w *window) IsResizable() bool { return w.hasStyle(wsThickFrame) }
func (w *window) SetMovable(v bool) { w.movable = v }
func (w *window) IsMovable() bool   { return w.movable }

func (w *window) SetMinimizable(v bool) {
	w.setStyle(wsMinimizeBox, v)
	w.captionChanged()
}

func (w *window) IsMinimizable() bool { return w.hasStyle(wsMinimizeBox) }

func (w *window) SetMaximizable(v bool) {
	w.setStyle(wsMaximizeBox, v)
	w.captionChanged()
}

func (w *window) IsMaximizable() bool { return w.hasStyle(wsMaximizeBox) }
func (w *window) IsClosable() bool    { return w.closable }

func (w *window) SetClosable(v bool) {
	w.closable = v
	menu, _, _ := procGetSystemMenu.Call(w.hwnd, 0)
	flags := uintptr(mfByCommand)
	if !v {
		flags |= mfGrayed
	}
	procEnableMenuItem.Call(menu, scClose, flags)
	if w.caption != nil {
		w.caption.paint()
	}
}

func (w *window) SetAlwaysOnTop(v bool) {
	after := hwndNoTopmost
	if v {
		after = hwndTopmost
	}
	procSetWindowPos.Call(w.hwnd, after, 0, 0, 0, 0, swpNoMove|swpNoSize|swpNoActivate)
}

func (w *window) IsAlwaysOnTop() bool { return windowLong(w.hwnd, gwlExStyle)&wsExTopmost != 0 }

func (w *window) Show() {
	if f := w.opts.Flyout; f != nil && !f.Focusable {
		procShowWindow.Call(w.hwnd, swShowNoActivate)
		return
	}
	if w.showMaximized {
		w.showMaximized = false
		procShowWindow.Call(w.hwnd, swShowMaximized)
	} else {
		procShowWindow.Call(w.hwnd, swShow)
	}
	procSetForegroundWindow.Call(w.hwnd)
}

func (w *window) ShowInactive() {
	if w.showMaximized {
		// Maximizing activates the window; there is no inactive variant.
		w.showMaximized = false
		procShowWindow.Call(w.hwnd, swShowMaximized)
		return
	}
	procShowWindow.Call(w.hwnd, swShowNoActivate)
}
func (w *window) Hide() { procShowWindow.Call(w.hwnd, swHide) }

func (w *window) IsVisible() bool {
	r, _, _ := procIsWindowVisible.Call(w.hwnd)
	return r != 0
}

func (w *window) Focus() {
	if w.host != nil {
		w.focusWebView()
		return
	}
	if f := w.opts.Flyout; f != nil && !f.Focusable {
		w.Show() // never activated
		return
	}
	if w.IsMinimized() {
		procShowWindow.Call(w.hwnd, swRestore)
	}
	procSetForegroundWindow.Call(w.hwnd)
	w.focusWebView()
}

func (w *window) Blur() {
	const gwHwndNext = 2
	next, _, _ := procGetNextWindow.Call(w.hwnd, gwHwndNext)
	if next != 0 {
		procSetForegroundWindow.Call(next)
	}
}

func (w *window) IsFocused() bool {
	if w.host != nil {
		return w.focused
	}
	fg, _, _ := procGetForegroundWindow.Call()
	return fg == w.hwnd
}

func (w *window) Minimize() { procShowWindow.Call(w.hwnd, swMinimize) }

func (w *window) IsMinimized() bool {
	r, _, _ := procIsIconic.Call(w.hwnd)
	return r != 0
}

func (w *window) Maximize() {
	if !w.IsVisible() {
		w.showMaximized = true // maximizing would show it
		return
	}
	procShowWindow.Call(w.hwnd, swShowMaximized)
}

func (w *window) Unmaximize() {
	if w.showMaximized {
		w.showMaximized = false
		return
	}
	if w.IsMaximized() {
		procShowWindow.Call(w.hwnd, swRestore)
	}
}

func (w *window) IsMaximized() bool {
	if w.showMaximized {
		return true
	}
	r, _, _ := procIsZoomed.Call(w.hwnd)
	return r != 0
}

func (w *window) Restore() { procShowWindow.Call(w.hwnd, swRestore) }

// SetFullScreen takes the window's frame and menu bar away, or gives them
// back. Each step sizes the window again: it is laid out once, at the end.
func (w *window) SetFullScreen(v bool) {
	if v == w.fullScreen {
		return
	}
	if w.inMenu {
		procEndMenu.Call() // its menus come from a bar that comes or goes
	}
	w.reframing = true
	if v {
		w.saved.style = windowLong(w.hwnd, gwlStyle)
		w.saved.exStyle = windowLong(w.hwnd, gwlExStyle)
		w.saved.placement.Length = uint32(unsafe.Sizeof(w.saved.placement))
		procGetWindowPlacement.Call(w.hwnd, uintptr(unsafe.Pointer(&w.saved.placement)))
		w.fullScreen = true
		style := w.saved.style &^ (wsCaption | wsThickFrame)
		setWindowLong(w.hwnd, gwlStyle, style)
		w.attachMenu() // no menu bar in full screen
		m := monitorInfo(w.monitor()).Monitor
		procSetWindowPos.Call(w.hwnd, 0, uintptr(m.Left), uintptr(m.Top), uintptr(m.Right-m.Left), uintptr(m.Bottom-m.Top), swpNoZOrder|swpFrameChanged)
	} else {
		w.fullScreen = false
		setWindowLong(w.hwnd, gwlStyle, w.saved.style)
		setWindowLong(w.hwnd, gwlExStyle, w.saved.exStyle)
		w.attachMenu()
		procSetWindowPlacement.Call(w.hwnd, uintptr(unsafe.Pointer(&w.saved.placement)))
		procSetWindowPos.Call(w.hwnd, 0, 0, 0, 0, 0, swpNoMove|swpNoSize|swpNoZOrder|swpFrameChanged)
	}
	w.reframing = false
	kind := uintptr(sizeRestored)
	if w.IsMinimized() {
		kind = sizeMinimized
	} else if zoomed, _, _ := procIsZoomed.Call(w.hwnd); zoomed != 0 {
		kind = sizeMaximized
	}
	w.sized(kind)
	if v {
		w.h.EnteredFullScreen()
	} else {
		w.h.LeftFullScreen()
	}
	w.captionChanged() // hidden in full screen
}

func (w *window) IsFullScreen() bool { return w.fullScreen }

func (w *window) Center() {
	var r rect
	procGetWindowRect.Call(w.hwnd, uintptr(unsafe.Pointer(&r)))
	work := monitorWorkArea(w.monitor())
	width, height := r.Right-r.Left, r.Bottom-r.Top
	x := work.Left + (work.Right-work.Left-width)/2
	y := work.Top + (work.Bottom-work.Top-height)/2
	procSetWindowPos.Call(w.hwnd, 0, uintptr(x), uintptr(y), 0, 0, swpNoSize|swpNoZOrder|swpNoActivate)
}

func (w *window) SetBackgroundColor(c platform.Color) {
	w.bg = &c
	// A web view in native UI has no window of its own to paint: only
	// WebView2 fills it, and a brush would outlive it.
	if w.host == nil {
		if w.bgBrush != 0 {
			procDeleteObject.Call(w.bgBrush)
		}
		w.bgBrush, _, _ = procCreateSolidBrush.Call(uintptr(c.R) | uintptr(c.G)<<8 | uintptr(c.B)<<16)
	}
	w.withWebView(w.applyWebViewBackground)
	if w.hwnd != 0 {
		procInvalidateRect(w.hwnd)
	}
}

// applyWebViewBackground paints the webview before its page does:
// transparent with a material behind it, else the background color.
// WebView2 only supports fully opaque or fully transparent backgrounds.
// A transparent page shows the window's background color behind it, which
// GDI paints; in a window without a redirection bitmap, the webview does.
func (w *window) applyWebViewBackground() {
	ctl2 := queryInterface(w.controller, &iidICoreWebView2Controller2)
	if ctl2 == 0 {
		return
	}
	defer release(ctl2)
	var c uint32 = 0xFFFFFFFF // opaque white: A, R, G, B bytes
	switch {
	case w.vibrancy != "":
		c = 0
	case w.opts.Transparent && (w.bg == nil || !w.noRedirect):
		c = 0
	case w.bg != nil && w.bg.A == 0:
		c = 0
	case w.bg != nil:
		c = 0xFF | uint32(w.bg.R)<<8 | uint32(w.bg.G)<<16 | uint32(w.bg.B)<<24
	}
	comCall(ctl2, ctl2PutDefaultBackgroundColor, uintptr(c))
}

func (w *window) SetOpacity(v float64) {
	w.opacity = v
	w.updateLayered()
}

func (w *window) Opacity() float64 { return w.opacity }

// updateLayered makes the window layered while it is translucent or lets
// the mouse through.
func (w *window) updateLayered() {
	layered := w.opacity < 1 || w.ignoreMouse
	w.setExStyle(wsExLayered, layered)
	if layered {
		procSetLayeredWindowAttributes.Call(w.hwnd, 0, uintptr(byte(w.opacity*255+0.5)), lwaAlpha)
	}
}

func (w *window) SetHasShadow(v bool) {
	w.shadow = v
	if !w.captionless() || w.vibrancy != "" {
		return
	}
	m := margins{}
	if v {
		m.Top = 1 // a frame the size of a pixel brings the shadow back
	}
	procDwmExtendFrameIntoClientArea.Call(w.hwnd, uintptr(unsafe.Pointer(&m)))
}

func (w *window) HasShadow() bool { return w.shadow }

func (w *window) SetIgnoreMouseEvents(v bool) {
	w.ignoreMouse = v
	w.setExStyle(wsExTransparent, v)
	w.updateLayered()
}

func (w *window) SetContentProtection(v bool) {
	if !has(procSetWindowDisplayAffinity) {
		return
	}
	affinity := uintptr(wdaNone)
	if v {
		affinity = wdaExcludeFromCapture
	}
	if r, _, _ := procSetWindowDisplayAffinity.Call(w.hwnd, affinity); r == 0 && v {
		procSetWindowDisplayAffinity.Call(w.hwnd, 1) // WDA_MONITOR before Windows 10 2004
	}
}

// Windows 11 system backdrops (DWM_SYSTEMBACKDROP_TYPE).
const (
	backdropNone    = 1
	backdropMica    = 2
	backdropAcrylic = 3
	backdropMicaAlt = 4
)

func backdropFor(material string) int32 {
	switch material {
	case "":
		return backdropNone
	case "mica":
		return backdropMica
	case "tabbed":
		return backdropMicaAlt
	case "acrylic", "menu", "popover", "hud", "sheet", "tooltip", "selection", "fullscreen-ui":
		return backdropAcrylic
	}
	return backdropMica
}

func (w *window) SetVibrancy(material string) {
	w.vibrancy = material
	backdrop := backdropFor(material)
	m := margins{}
	if material != "" {
		// The material shows where the frame extends, behind the page.
		m = margins{-1, -1, -1, -1}
	}
	procDwmExtendFrameIntoClientArea.Call(w.hwnd, uintptr(unsafe.Pointer(&m)))
	procDwmSetWindowAttribute.Call(w.hwnd, dwmwaSystemBackdropType, uintptr(unsafe.Pointer(&backdrop)), 4)
	if material == "" && w.captionless() && w.shadow {
		w.SetHasShadow(true)
	}
	w.withWebView(w.applyWebViewBackground)
	procInvalidateRect(w.hwnd)
}

// hasAppMenu tells whether the window shows the application menu in its
// menu bar: flyouts and panels show none.
func (w *window) hasAppMenu() bool { return w.opts.Flyout == nil && !w.opts.Panel }

// SetVibrancyStyle does nothing: a system backdrop fills the window.
func (w *window) SetVibrancyStyle(platform.VibrancyStyle) {}

func (w *window) SetMenu(m *platform.Menu) {
	w.ownMenu = m != nil
	if m == nil && w.hasAppMenu() {
		m = w.b.appMenu
	}
	w.installMenu(m)
}

func (w *window) SetAutoHideMenu(v bool) {
	w.autoHideMenu = v
	if w.hmenu != 0 && !w.revealed {
		w.attachMenu()
	}
}

func (w *window) StartDrag() {
	procReleaseCapture.Call()
	procSendMessageW.Call(w.hwnd, wmNCLButtonDown, htCaption, 0)
}

func (w *window) TitleBarDoubleClicked() {
	if w.IsMaximized() {
		w.Unmaximize()
	} else if w.hasStyle(wsMaximizeBox) {
		w.Maximize()
	}
}

func (w *window) Close() {
	if w.host != nil {
		w.closeWebView()
		return
	}
	if !w.closed {
		w.destroy()
	}
}

func (w *window) setIcon(icon uintptr) {
	const iconSmall, iconBig = 0, 1
	procSendMessageW.Call(w.hwnd, wmSetIcon, iconSmall, icon)
	procSendMessageW.Call(w.hwnd, wmSetIcon, iconBig, icon)
}

func procInvalidateRect(hwnd uintptr) { procInvalidateRectW.Call(hwnd, 0, 1) }

func boolArg(v bool) uintptr {
	if v {
		return 1
	}
	return 0
}
