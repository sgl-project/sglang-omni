//go:build windows && (amd64 || arm64)

package windows

import (
	"time"
	"unicode/utf16"
	"unsafe"

	"github.com/egoist/mygo/internal/platform"
)

// The surface of a window that shows content MyGo draws itself is a child
// window filling its client area, below the caption controls of a hidden
// title bar, as the webview's window is. Renderers draw into it with a
// swap chain, or the backend copies frames drawn in memory onto it.

const surfaceClass = "MyGoSurface"

var (
	imm32 = systemDLL("imm32.dll")

	procSetCapture            = user32.NewProc("SetCapture")
	procSetCursor             = user32.NewProc("SetCursor")
	procBeginPaint            = user32.NewProc("BeginPaint")
	procEndPaint              = user32.NewProc("EndPaint")
	procRedrawWindow          = user32.NewProc("RedrawWindow")
	procSystemParametersInfoW = user32.NewProc("SystemParametersInfoW")
	procGetDoubleClickTime    = user32.NewProc("GetDoubleClickTime")
	procGetMessageTime        = user32.NewProc("GetMessageTime")
	procGetSystemMetrics      = user32.NewProc("GetSystemMetrics")
	procSetDIBitsToDevice     = gdi32.NewProc("SetDIBitsToDevice")
	procImmGetContext         = imm32.NewProc("ImmGetContext")
	procImmReleaseContext     = imm32.NewProc("ImmReleaseContext")
	procImmGetCompositionStrW = imm32.NewProc("ImmGetCompositionStringW")
	procImmSetCompositionWnd  = imm32.NewProc("ImmSetCompositionWindow")
	procImmSetCandidateWindow = imm32.NewProc("ImmSetCandidateWindow")
	procImmAssociateContextEx = imm32.NewProc("ImmAssociateContextEx")
	procImmNotifyIME          = imm32.NewProc("ImmNotifyIME")
	surfaceClassRegistered    bool
	surfaceCursors            = map[platform.Cursor]uintptr{}
	surfaceWheelLines         = 0
	surfaceWheelLinesQueried  = false
)

const (
	wmPaint              = 0x000F
	wmKillFocus          = 0x0008
	wmSetCursor          = 0x0020
	wmGetDlgCode         = 0x0087
	wmKeyDown            = 0x0100
	wmKeyUp              = 0x0101
	wmChar               = 0x0102
	wmSysKeyDown         = 0x0104
	wmSysKeyUp           = 0x0105
	wmSysChar            = 0x0106
	wmImeStartComp       = 0x010D
	wmImeEndComp         = 0x010E
	wmImeComposition     = 0x010F
	wmMouseMove          = 0x0200
	wmLButtonDown        = 0x0201
	wmLButtonDblClk      = 0x0203
	wmRButtonDown        = 0x0204
	wmRButtonDblClk      = 0x0206
	wmMButtonDown        = 0x0207
	wmMButtonUp          = 0x0208
	wmMButtonDblClk      = 0x0209
	wmMouseWheel         = 0x020A
	wmMouseHWheel        = 0x020E
	wmXButtonDown        = 0x020B
	wmXButtonUp          = 0x020C
	wmXButtonDblClk      = 0x020D
	wmAppCommand         = 0x0319
	wmCaptureChanged     = 0x0215
	wmImeSetContext      = 0x0281
	wmImeChar            = 0x0286
	wmGetObject          = 0x003D
	dlgcWantAllKeys      = 0x0004
	dlgcWantChars        = 0x0080
	iscShowUICompWindow  = 0x80000000
	gcsCompStr           = 0x0008
	gcsCursorPos         = 0x0080
	gcsResultStr         = 0x0800
	cfsPoint             = 0x0002
	cfsExclude           = 0x0080
	iaceDefault          = 0x0010
	niCompositionStr     = 0x0015
	cpsComplete          = 0x0001
	cpsCancel            = 0x0004
	rdwInvalidate        = 0x0001
	rdwUpdateNow         = 0x0100
	spiGetWheelScrollLns = 0x0068
	tmeLeaveFlag         = 0x0002
)

type paintStruct struct {
	HDC       uintptr
	Erase     int32
	Paint     rect
	Restore   int32
	IncUpdate int32
	Reserved  [32]byte
}

type compositionForm struct {
	Style uint32
	Pos   point
	Area  rect
}

type candidateForm struct {
	Index uint32
	Style uint32
	Pos   point
	Area  rect
}

type surface struct {
	clientComposition platform.InputComposition
	w                 *window
	hwnd              uintptr

	paintDC  uintptr
	tracking bool
	buttons  int
	clicks   platform.ClickCounter
	cursor   platform.Cursor
	high     uint16 // a high surrogate of WM_CHAR waiting for its pair
	ime      bool
	caret    platform.RectF
	input    platform.TextInputState

	// keyTaken tells that the content took the key down, whose
	// WM_SYSCHAR then opens no menu.
	keyTaken bool

	reconvert  *[2]int // the runes a reconversion replaces
	dropTarget uintptr // IDropTarget
	access     *uiaTree
	dragSource *oleDragSource

	// comp shows the frames drawn in memory in a window without a
	// redirection bitmap, where GDI cannot (compositor.go); nil while none
	// shows.
	comp *compositor

	// layer holds the web views under the frames, once there are some
	// (webview_embed.go): placed are where the content shows them, and
	// stacked those shown, in the order of their visuals. webHover is the
	// web view under the pointer, and webCapture the one a press went to,
	// which gets the mouse until the buttons are let go.
	layer                *webLayer
	placed               []platform.WebViewPlacement
	stacked              []*window
	webHover, webCapture *window
}

func registerSurfaceClass() {
	if surfaceClassRegistered {
		return
	}
	surfaceClassRegistered = true
	wc := wndClassEx{
		WndProc:   wndProcCallback,
		Instance:  instance(),
		ClassName: u16(surfaceClass),
	}
	wc.Size = uint32(unsafe.Sizeof(wc))
	procRegisterClassExW.Call(uintptr(unsafe.Pointer(&wc)))
}

func newSurface(w *window) *surface {
	registerSurfaceClass()
	s := &surface{w: w}
	s.hwnd = createWindow(0, surfaceClass, "", wsChild|wsVisible|wsClipSiblings, 0, 0, 0, 0, w.hwnd)
	if s.hwnd == 0 {
		return nil
	}
	w.b.surfaces[s.hwnd] = s
	// Input methods only come to text inputs.
	procImmAssociateContextEx.Call(s.hwnd, 0, 0)
	s.acceptFileDrops()
	s.fit()
	return s
}

// fit makes the surface fill the client area of its window.
func (s *surface) fit() {
	var r rect
	procGetClientRect.Call(s.w.hwnd, uintptr(unsafe.Pointer(&r)))
	procSetWindowPos.Call(s.hwnd, 0, 0, 0, uintptr(r.Right-r.Left), uintptr(r.Bottom-r.Top), swpNoZOrder|swpNoActivate)
}

func (s *surface) dpi() int { return dpiOf(s.w.hwnd) }

func (s *surface) toDIP(v int32) float64 { return float64(v) * 96 / float64(s.dpi()) }

// Native returns the surface's window, which a GPU renderer is about to
// draw into: in a window without a redirection bitmap, it composes the
// window from then on, and the frames drawn in memory let go of it.
func (s *surface) Native() platform.SurfaceNative {
	s.freeComp()
	return platform.SurfaceNative{HWND: s.hwnd, Composed: s.viaComposition()}
}

// viaComposition reports whether the frames show through
// DirectComposition: in a window without a redirection bitmap, and over
// web views.
func (s *surface) viaComposition() bool { return s.w.noRedirect || s.layer != nil }

// ShowsMaterial reports whether the window shows its material behind the
// content: it has one, and no redirection bitmap to cover it.
func (s *surface) ShowsMaterial() bool { return s.w.noRedirect && s.w.vibrancy != "" }

func (s *surface) freeComp() {
	if s.comp != nil {
		s.comp.free()
		s.comp = nil
	}
}

func (s *surface) Size() (float64, float64, float64) {
	var r rect
	procGetClientRect.Call(s.hwnd, uintptr(unsafe.Pointer(&r)))
	return s.toDIP(r.Right - r.Left), s.toDIP(r.Bottom - r.Top), float64(s.dpi()) / 96
}

func (s *surface) RequestFrame() { procInvalidateRectW.Call(s.hwnd, 0, 0) }

// RefreshRate returns the refresh rate of the monitor showing the window,
// in its current display mode.
func (s *surface) RefreshRate() float64 {
	mi := monitorInfo(s.w.monitor())
	var dm devMode
	dm.Size = uint16(unsafe.Sizeof(dm))
	if ok, _, _ := procEnumDisplaySettingsW.Call(uintptr(unsafe.Pointer(&mi.Device[0])), enumCurrentSettings, uintptr(unsafe.Pointer(&dm))); ok == 0 {
		return 0
	}
	if dm.DisplayFrequency <= 1 {
		return 0 // the hardware's default rate, unknown
	}
	return float64(dm.DisplayFrequency)
}

func (s *surface) PresentPixels(pix []byte, stride, width, height int) {
	if len(pix) < stride*height || width == 0 || height == 0 {
		return
	}
	if s.viaComposition() {
		// GDI draws nothing in a window without a redirection bitmap, and
		// would draw under the web views: the frame shows through
		// DirectComposition, with its alpha.
		if s.comp == nil {
			s.comp = newCompositor(s.w.b, s.hwnd)
		}
		px := s.comp.pixels(width * height)
		for y := range height {
			copy(px[y*width:(y+1)*width], unsafe.Slice((*uint32)(unsafe.Pointer(&pix[y*stride])), width))
		}
		s.comp.show(px, int32(width), int32(height))
		return
	}
	dc := s.paintDC
	if dc == 0 {
		dc, _, _ = procGetDC.Call(s.hwnd)
		defer procReleaseDC.Call(s.hwnd, dc)
	}
	bi := bitmapInfoHeader{Width: int32(stride / 4), Height: -int32(height), Planes: 1, BitCount: 32}
	bi.Size = uint32(unsafe.Sizeof(bi))
	procSetDIBitsToDevice.Call(dc, 0, 0, uintptr(width), uintptr(height), 0, 0, 0, uintptr(height),
		uintptr(unsafe.Pointer(&pix[0])), uintptr(unsafe.Pointer(&bi)), 0)
}

func (s *surface) SetCursor(c platform.Cursor) {
	s.cursor = c
	var pt point
	procGetCursorPos.Call(uintptr(unsafe.Pointer(&pt)))
	procScreenToClient.Call(s.hwnd, uintptr(unsafe.Pointer(&pt)))
	var r rect
	procGetClientRect.Call(s.hwnd, uintptr(unsafe.Pointer(&r)))
	if pt.X >= r.Left && pt.Y >= r.Top && pt.X < r.Right && pt.Y < r.Bottom {
		procSetCursor.Call(cursorHandle(c))
	}
}

func cursorHandle(c platform.Cursor) uintptr {
	if h, ok := surfaceCursors[c]; ok {
		return h
	}
	ids := map[platform.Cursor]uintptr{
		platform.CursorDefault: 32512, platform.CursorPointer: 32649, platform.CursorText: 32513,
		platform.CursorMove: 32646, platform.CursorResizeEW: 32644, platform.CursorResizeNS: 32645,
		platform.CursorResizeNWSE: 32642, platform.CursorResizeNESW: 32643, platform.CursorNotAllowed: 32648,
		platform.CursorCrosshair: 32515, platform.CursorGrab: 32649, platform.CursorGrabbing: 32646,
		// Windows has no cursors of one direction, of columns and rows, or
		// for vertical text: those of both directions and the I-beam.
		platform.CursorResizeN: 32645, platform.CursorResizeS: 32645, platform.CursorResizeE: 32644,
		platform.CursorResizeW: 32644, platform.CursorResizeColumn: 32644, platform.CursorResizeRow: 32645,
		platform.CursorVerticalText: 32513,
	}
	if c == platform.CursorNone {
		surfaceCursors[c] = 0 // SetCursor(NULL) hides it
		return 0
	}
	id, ok := ids[c]
	if !ok {
		id = 32512
	}
	h, _, _ := procLoadCursorW.Call(0, id)
	surfaceCursors[c] = h
	return h
}

func (s *surface) SetTextInput(t platform.TextInputState) {
	// Without input methods, the window has no input context: keys type
	// the layout's characters, as WM_CHAR, as winit's do.
	active, caret := t.Active && !t.NoInputMethod, t.Caret
	if s.input.Active && s.input.Client != t.Client {
		s.clientComposition.Reset()
		if himc, _, _ := procImmGetContext.Call(s.hwnd); himc != 0 {
			procImmNotifyIME.Call(himc, niCompositionStr, cpsCancel, 0)
			procImmReleaseContext.Call(s.hwnd, himc)
		}
		s.reconvert = nil
	}
	s.input = t
	if active != s.ime {
		s.ime = active
		if active {
			procImmAssociateContextEx.Call(s.hwnd, 0, iaceDefault)
		} else {
			s.reconvert = nil
			if himc, _, _ := procImmGetContext.Call(s.hwnd); himc != 0 {
				procImmNotifyIME.Call(himc, niCompositionStr, cpsComplete, 0)
				procImmReleaseContext.Call(s.hwnd, himc)
			}
			procImmAssociateContextEx.Call(s.hwnd, 0, 0)
		}
	}
	s.caret = caret
	if active {
		s.placeIME()
	}
}

// placeIME puts the input method's composition and candidate windows at
// the caret.
func (s *surface) placeIME() {
	himc, _, _ := procImmGetContext.Call(s.hwnd)
	if himc == 0 {
		return
	}
	defer procImmReleaseContext.Call(s.hwnd, himc)
	dpi := s.dpi()
	x, y := toPx(int(s.caret.X), dpi), toPx(int(s.caret.Y), dpi)
	h := toPx(int(s.caret.H+0.5), dpi)
	cf := compositionForm{Style: cfsPoint, Pos: point{x, y}}
	procImmSetCompositionWnd.Call(himc, uintptr(unsafe.Pointer(&cf)))
	cand := candidateForm{Style: cfsExclude, Pos: point{x, y + h}, Area: rect{x, y, x + 1, y + h}}
	procImmSetCandidateWindow.Call(himc, uintptr(unsafe.Pointer(&cand)))
}

func (s *surface) send(ev platform.SurfaceEvent) bool {
	if ev.Kind == platform.PointerDown || ev.Kind == platform.SurfaceBlur {
		s.clientComposition.Reset()
	}
	if s.w.closed {
		return false
	}
	return s.w.h.SurfaceEvent(ev)
}

// modifierKey is whether the virtual key vk is Shift, Ctrl, Alt or a Windows key.
func modifierKey(vk uintptr) bool {
	switch vk {
	case vkShift, vkControl, vkMenu, vkLWin, vkRWin, 0xA0, 0xA1, 0xA2, 0xA3, 0xA4, 0xA5:
		return true
	}
	return false
}

func mods() platform.Modifiers {
	var m platform.Modifiers
	down := func(vk uintptr) bool {
		r, _, _ := procGetKeyState.Call(vk)
		return r&0x8000 != 0
	}
	if down(vkShift) {
		m |= platform.ModShift
	}
	if down(vkControl) {
		m |= platform.ModCtrl
	}
	if down(vkMenu) {
		m |= platform.ModAlt
	}
	if down(vkLWin) || down(vkRWin) {
		m |= platform.ModSuper
	}
	return m
}

// systemMetric is GetSystemMetrics' metric for the DPI of the window hwnd.
func systemMetric(metric, hwnd uintptr) uintptr {
	if has(procGetSystemMetricsForDpi) {
		if n, _, _ := procGetSystemMetricsForDpi.Call(metric, uintptr(dpiOf(hwnd))); n != 0 {
			return n
		}
	}
	n, _, _ := procGetSystemMetrics.Call(metric)
	return n
}

func (s *surface) pointer(kind platform.SurfaceEventKind, lp uintptr, button int) {
	x, y := int16(loword(lp)), int16(hiword(lp))
	clicks := 0
	if kind == platform.PointerDown {
		// A press carries no count: it follows the double-click time and
		// the rectangle centered on the last press, in the window's pixels,
		// as set.
		ms, _, _ := procGetDoubleClickTime.Call()
		at, _, _ := procGetMessageTime.Call()
		w, h := systemMetric(smCxDoubleClk, s.hwnd), systemMetric(smCyDoubleClk, s.hwnd)
		clicks = s.clicks.Press(button, float64(x), float64(y), uint32(at), platform.DoubleClick{Time: time.Duration(ms) * time.Millisecond, DX: float64(w) / 2, DY: float64(h) / 2})
	}
	s.send(platform.SurfaceEvent{Kind: kind, X: s.toDIP(int32(x)), Y: s.toDIP(int32(y)), Button: button, Clicks: clicks, Mods: mods()})
}

func (s *surface) message(hwnd uintptr, m uint32, wp, lp uintptr) (uintptr, bool) {
	switch m {
	case wmMouseMove, wmLButtonDown, wmLButtonDblClk, wmLButtonUp, wmRButtonDown, wmRButtonDblClk, wmRButtonUp,
		wmMButtonDown, wmMButtonDblClk, wmMButtonUp, wmXButtonDown, wmXButtonDblClk, wmXButtonUp, wmMouseWheel, wmMouseHWheel:
		if s.forwardMouse(m, wp, lp) {
			return 0, true // the web view under the pointer takes it
		}
	case wmTimer:
		if wp == timerWebView {
			procKillTimer.Call(hwnd, timerWebView)
			for _, v := range s.w.webViews {
				if v.controller == 0 && v.webViewErr == nil {
					v.createWebView()
				}
			}
			return 0, true
		}
	}
	switch m {
	case wmPaint:
		var ps paintStruct
		dc, _, _ := procBeginPaint.Call(hwnd, uintptr(unsafe.Pointer(&ps)))
		s.paintDC = dc
		s.send(platform.SurfaceEvent{Kind: platform.SurfaceFrame})
		s.paintDC = 0
		procEndPaint.Call(hwnd, uintptr(unsafe.Pointer(&ps)))
		return 0, true
	case wmEraseBkgnd:
		return 1, true
	case wmSize:
		if s.layer != nil {
			// The web views keep their place in DIPs, in other pixels after
			// a change of DPI.
			for _, v := range s.w.webViews {
				v.placeEmbedded()
			}
			s.layer.commit()
		}
		s.send(platform.SurfaceEvent{Kind: platform.SurfaceResize})
		// Draw at once, so the content follows the window while it is
		// resized.
		procRedrawWindow.Call(hwnd, 0, 0, rdwInvalidate|rdwUpdateNow)
		return 0, true
	case wmMouseMove:
		if !s.tracking {
			tme := trackMouseEvent{Flags: tmeLeaveFlag, Track: hwnd}
			tme.Size = uint32(unsafe.Sizeof(tme))
			procTrackMouseEvent.Call(uintptr(unsafe.Pointer(&tme)))
			s.tracking = true
		}
		s.pointer(platform.PointerMove, lp, 0)
		return 0, true
	case wmMouseLeave:
		s.tracking = false
		s.mouseLeft()
		s.send(platform.SurfaceEvent{Kind: platform.PointerLeave})
		return 0, true
	case wmLButtonDown, wmLButtonDblClk, wmRButtonDown, wmRButtonDblClk, wmMButtonDown, wmMButtonDblClk:
		button := 0
		switch m {
		case wmRButtonDown, wmRButtonDblClk:
			button = 1
		case wmMButtonDown, wmMButtonDblClk:
			button = 2
		}
		procSetFocus.Call(hwnd)
		if s.buttons == 0 {
			procSetCapture.Call(hwnd)
		}
		s.buttons |= 1 << button
		s.pointer(platform.PointerDown, lp, button)
		return 0, true
	case wmLButtonUp, wmRButtonUp, wmMButtonUp:
		button := 0
		switch m {
		case wmRButtonUp:
			button = 1
		case wmMButtonUp:
			button = 2
		}
		s.buttons &^= 1 << button
		if s.buttons == 0 {
			procReleaseCapture.Call()
		}
		s.pointer(platform.PointerUp, lp, button)
		if m == wmRButtonUp {
			return 0, false // WM_CONTEXTMENU
		}
		return 0, true
	case wmXButtonDown, wmXButtonDblClk, wmXButtonUp:
		// The side buttons go back and forward, as keys do, as they are
		// let go: Windows's apps go there then, as DefWindowProc sends
		// WM_APPCOMMAND, which taking them leaves out.
		if m == wmXButtonUp {
			k := platform.KeyBack
			if hiword(wp) == 2 { // XBUTTON2
				k = platform.KeyForward
			}
			s.send(platform.SurfaceEvent{Kind: platform.KeyPressed, Key: k, Mods: mods()})
			s.send(platform.SurfaceEvent{Kind: platform.KeyReleased, Key: k, Mods: mods()})
		}
		return 1, true
	case wmAppCommand:
		// Devices that send commands rather than keys or buttons.
		k := platform.KeyUnknown
		switch hiword(lp) &^ 0xF000 {
		case 1: // APPCOMMAND_BROWSER_BACKWARD
			k = platform.KeyBack
		case 2: // APPCOMMAND_BROWSER_FORWARD
			k = platform.KeyForward
		}
		if k == platform.KeyUnknown {
			return 0, false
		}
		s.send(platform.SurfaceEvent{Kind: platform.KeyPressed, Key: k, Mods: mods()})
		s.send(platform.SurfaceEvent{Kind: platform.KeyReleased, Key: k, Mods: mods()})
		return 1, true
	case wmCaptureChanged:
		s.buttons = 0
		if lp != hwnd {
			s.webCapture = nil
		}
		return 0, true
	case wmMouseWheel, wmMouseHWheel:
		pt := point{int32(int16(loword(lp))), int32(int16(hiword(lp)))}
		procScreenToClient.Call(hwnd, uintptr(unsafe.Pointer(&pt)))
		delta := float64(int16(hiword(wp)))
		dist := delta / 120 * float64(wheelLines()) * 33
		ev := platform.SurfaceEvent{Kind: platform.PointerScroll, X: s.toDIP(pt.X), Y: s.toDIP(pt.Y), Mods: mods(), Precise: int(delta)%120 != 0}
		if m == wmMouseWheel {
			ev.DY = -dist
		} else {
			ev.DX = dist
		}
		s.send(ev)
		return 0, true
	case wmSetCursor:
		if loword(lp) == htClient {
			if s.webCursor() {
				return 1, true // the page's
			}
			procSetCursor.Call(cursorHandle(s.cursor))
			return 1, true
		}
	case wmGetDlgCode:
		return dlgcWantAllKeys | dlgcWantChars, true
	case wmSetFocus:
		s.send(platform.SurfaceEvent{Kind: platform.SurfaceFocus})
		return 0, true
	case wmKillFocus:
		s.send(platform.SurfaceEvent{Kind: platform.SurfaceBlur})
		return 0, true
	case wmKeyDown, wmSysKeyDown:
		s.clientComposition.Reset()
		s.keyTaken = false
		modifiers := mods()
		if modifierKey(wp) {
			s.send(platform.SurfaceEvent{Kind: platform.ModifiersChanged, Mods: modifiers})
		}
		if k, physical, ok := keyAndPosition(wp, lp, false, modifiers); ok {
			s.keyTaken = s.send(platform.SurfaceEvent{Kind: platform.KeyPressed, Key: k, PhysicalKey: physical, Mods: modifiers, Repeat: lp&(1<<30) != 0})
		}
		// Alt+F4, Alt+Space and F10 keep working.
		return 0, m == wmKeyDown
	case wmKeyUp, wmSysKeyUp:
		modifiers := mods()
		if modifierKey(wp) {
			s.send(platform.SurfaceEvent{Kind: platform.ModifiersChanged, Mods: modifiers})
		}
		if k, physical, ok := keyAndPosition(wp, lp, true, modifiers); ok {
			s.send(platform.SurfaceEvent{Kind: platform.KeyReleased, Key: k, PhysicalKey: physical, Mods: modifiers})
		}
		return 0, m == wmKeyUp
	case wmSysChar:
		if s.keyTaken {
			// The content took Alt and the key, as a terminal does.
			return 0, true
		}
		// Alt and a letter go to the window, which opens the menu of the
		// letter: DefWindowProc would open the window's bar without it.
		if wp > ' ' {
			procSendMessageW.Call(s.w.hwnd, wmSysCommand, scKeyMenu, wp)
			return 0, true
		}
	case wmChar:
		c := uint16(wp)
		switch {
		case utf16.IsSurrogate(rune(c)) && c < 0xDC00:
			s.high = c
			return 0, true
		case utf16.IsSurrogate(rune(c)):
			r := utf16.DecodeRune(rune(s.high), rune(c))
			s.high = 0
			s.send(platform.SurfaceEvent{Kind: platform.TextInput, Text: string(r)})
			return 0, true
		case c < 0x20 || c == 0x7F:
			return 0, true // control characters come as keys
		}
		s.send(platform.SurfaceEvent{Kind: platform.TextInput, Text: string(rune(c))})
		return 0, true
	case wmImeSetContext:
		r, _, _ := procDefWindowProcW.Call(hwnd, uintptr(m), wp, lp&^iscShowUICompWindow)
		return r, true
	case wmImeStartComp:
		s.placeIME()
		return 0, true
	case wmImeComposition:
		himc, _, _ := procImmGetContext.Call(hwnd)
		if himc == 0 {
			return 0, true
		}
		if lp&gcsResultStr != 0 {
			if text := imeString(himc, gcsResultStr); text != "" {
				s.composed(platform.SurfaceEvent{Kind: platform.TextInput, Text: text})
			}
		}
		if lp&gcsCompStr != 0 {
			text := imeString(himc, gcsCompStr)
			pos, _, _ := procImmGetCompositionStrW.Call(himc, gcsCursorPos, 0, 0)
			units := utf16.Encode([]rune(text))
			caret := len(utf16.Decode(units[:min(int(pos), len(units))]))
			s.composed(platform.SurfaceEvent{Kind: platform.TextComposition, Text: text, Caret: caret})
		}
		procImmReleaseContext.Call(hwnd, himc)
		s.placeIME()
		return 0, true
	case wmImeEndComp:
		s.reconvert = nil
		if c := s.input.Client; c != nil {
			s.clientComposition.End(c)
			return 0, true
		}
		s.send(platform.SurfaceEvent{Kind: platform.TextComposition})
		return 0, true
	case wmImeChar:
		return 0, true
	case wmImeRequest:
		return s.imeRequest(wp, lp), true
	case wmGetObject:
		return s.getObject(wp, lp)
	case wmDestroy:
		s.CancelDataDrag()
		s.destroyAccess()
		s.revokeFileDrops()
		s.freeComp()
		delete(s.w.b.surfaces, hwnd)
		return 0, true
	}
	return 0, false
}

// imeString reads a composition string of an input context.
func imeString(himc, which uintptr) string {
	n, _, _ := procImmGetCompositionStrW.Call(himc, which, 0, 0)
	if int32(n) <= 0 {
		return ""
	}
	buf := make([]uint16, n/2)
	procImmGetCompositionStrW.Call(himc, which, uintptr(unsafe.Pointer(&buf[0])), n)
	return string(utf16.Decode(buf))
}

func wheelLines() int {
	if !surfaceWheelLinesQueried {
		surfaceWheelLinesQueried = true
		var n uint32
		if r, _, _ := procSystemParametersInfoW.Call(spiGetWheelScrollLns, 0, uintptr(unsafe.Pointer(&n)), 0); r != 0 && n > 0 && n < 100 {
			surfaceWheelLines = int(n)
		} else {
			surfaceWheelLines = 3
		}
	}
	return surfaceWheelLines
}

// keyAndPosition returns the key of a WM_KEYDOWN or WM_KEYUP by virtual
// key, and by position (the scancode in lp). ok is false for a key to leave
// out: one whose virtual key is unmapped, unless a shortcut modifier is
// held and its position is a US key. Known virtual keys keep their meaning
// (Russian letters already use VK_A through VK_Z). A release
// is sent whatever the modifiers held now, so that a press sent as a chord
// is let go of even after Ctrl was.
func keyAndPosition(vk, lp uintptr, release bool, modifiers platform.Modifiers) (k, physical platform.Key, ok bool) {
	k = vkKey(vk)
	// An input method (VK_PROCESSKEY) or injected Unicode text (VK_PACKET)
	// keeps the key from the application: the scancode is not a shortcut.
	if vk == vkProcessKey || vk == vkPacket {
		return k, platform.KeyUnknown, k != platform.KeyUnknown
	}
	// Bit 24 is the extended flag: the keypad's slash shares the code of /.
	if lp&(1<<24) == 0 {
		physical = platform.KeyForScancode(uint16(lp >> 16 & 0xFF))
	}
	if k == platform.KeyUnknown && !(release && physical != platform.KeyUnknown) && !platform.KeyByPosition(k, physical, modifiers) {
		return k, physical, false
	}
	return k, physical, true
}

// vkKey maps a virtual key code to the key it stands for.
func vkKey(vk uintptr) platform.Key {
	switch {
	case vk >= 'A' && vk <= 'Z':
		return platform.KeyA + platform.Key(vk-'A')
	case vk >= '0' && vk <= '9':
		return platform.Key0 + platform.Key(vk-'0')
	case vk >= 0x60 && vk <= 0x69: // the numeric keypad
		return platform.Key0 + platform.Key(vk-0x60)
	case vk >= 0x70 && vk <= 0x7B:
		return platform.KeyF1 + platform.Key(vk-0x70)
	}
	switch vk {
	case 0x0D:
		return platform.KeyEnter
	case 0x1B:
		return platform.KeyEscape
	case 0x08:
		return platform.KeyBackspace
	case 0x09:
		return platform.KeyTab
	case 0x20:
		return platform.KeySpace
	case 0x2E:
		return platform.KeyDelete
	case 0x2D:
		return platform.KeyInsert
	case 0x24:
		return platform.KeyHome
	case 0x23:
		return platform.KeyEnd
	case 0x21:
		return platform.KeyPageUp
	case 0x22:
		return platform.KeyPageDown
	case 0x25:
		return platform.KeyLeft
	case 0x26:
		return platform.KeyUp
	case 0x27:
		return platform.KeyRight
	case 0x28:
		return platform.KeyDown
	case 0xBD, 0x6D:
		return platform.KeyMinus
	case 0xBB:
		return platform.KeyEqual
	case 0xBC:
		return platform.KeyComma
	case 0xBE, 0x6E:
		return platform.KeyPeriod
	case 0xBF, 0x6F:
		return platform.KeySlash
	case 0xBA:
		return platform.KeySemicolon
	case 0xDE:
		return platform.KeyQuote
	case 0xDB:
		return platform.KeyBracketLeft
	case 0xDD:
		return platform.KeyBracketRight
	case 0xDC:
		return platform.KeyBackslash
	case 0xC0:
		return platform.KeyBackquote
	case 0x5D:
		return platform.KeyContextMenu
	case 0xA6: // VK_BROWSER_BACK
		return platform.KeyBack
	case 0xA7: // VK_BROWSER_FORWARD
		return platform.KeyForward
	}
	return platform.KeyUnknown
}

// Surface returns the surface of a window created with
// WindowOptions.Surface.
func (w *window) Surface() platform.Surface {
	if w.surface == nil {
		return nil
	}
	return w.surface
}
