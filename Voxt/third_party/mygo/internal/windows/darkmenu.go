//go:build windows && (amd64 || arm64)

package windows

import (
	"sync"
	"syscall"
	"unsafe"
)

// Win32 menus have no dark appearance of their own. Popup menus follow the
// preferred app mode of uxtheme, which Windows exports by ordinal only
// (since Windows 10 1903, build 18362). The menu bar follows nothing: the
// window draws it, and the line below it, when the system asks through the
// messages UAH ("user API hook") menus send to their window.

const (
	wmNCPaint         = 0x0085
	wmNCActivate      = 0x0086
	wmUAHDrawMenu     = 0x0091
	wmUAHDrawMenuItem = 0x0092
	objidMenu         = 0xFFFFFFFD // OBJID_MENU
	dtHidePrefix      = 0x00100000
	rdwFrame          = 0x0400

	odsSelected = 0x0001
	odsGrayed   = 0x0002
	odsDisabled = 0x0004
	odsHotLight = 0x0040
	odsNoAccel  = 0x0100

	// SetPreferredAppMode's modes.
	appModeForceDark  = 2
	appModeForceLight = 3
)

var (
	procGetMenuBarInfo = user32.NewProc("GetMenuBarInfo")
	procGetWindowDC    = user32.NewProc("GetWindowDC")
)

// The colors of a dark menu bar, those of the dark title bar above it and,
// for hovered and open items, of the caption buttons drawn in its place
// (white at 10% and 20% over it).
const (
	darkMenuBar      = 0x202020 // COLORREF, 0x00BBGGRR
	darkMenuHot      = 0x363636
	darkMenuOpen     = 0x4D4D4D
	darkMenuText     = 0xFFFFFF
	darkMenuDisabled = 0x6D6D6D
)

// uxtheme's preferred app mode, and the flush that makes menus created
// before it follow.
var appMode = sync.OnceValues(func() (setPreferredAppMode, flushMenuThemes uintptr) {
	var v osVersionInfo
	v.Size = uint32(unsafe.Sizeof(v))
	procRtlGetVersion.Call(uintptr(unsafe.Pointer(&v)))
	if v.Major < 10 || v.Major == 10 && v.Build < 18362 {
		return 0, 0 // ordinal 135 was AllowDarkModeForApp before
	}
	const loadLibrarySearchSystem32 = 0x800
	h, _, _ := procLoadLibraryExW.Call(uintptr(unsafe.Pointer(u16("uxtheme.dll"))), 0, loadLibrarySearchSystem32)
	if h == 0 {
		return 0, 0
	}
	set, _, _ := procGetProcAddress.Call(h, 135)
	flush, _, _ := procGetProcAddress.Call(h, 136)
	return set, flush
})

// applyMenuTheme gives the popup menus of the app, those of its trays
// included, the appearance.
func (b *Backend) applyMenuTheme() {
	set, flush := appMode()
	if set == 0 {
		return
	}
	mode := uintptr(appModeForceLight)
	if b.isDark() {
		mode = appModeForceDark
	}
	syscall.SyscallN(set, mode)
	if flush != 0 {
		syscall.SyscallN(flush)
	}
}

type drawItemStruct struct {
	CtlType, CtlID, ItemID, ItemAction, ItemState uint32
	HwndItem, DC                                  uintptr
	Item                                          rect
	ItemData                                      uintptr
}

type uahMenu struct {
	Menu, DC uintptr
	Flags    uint32
}

type uahDrawMenuItem struct {
	Draw     drawItemStruct
	Menu     uahMenu
	Position int32
	// UAHMENUITEMMETRICS and UAHMENUPOPUPMETRICS follow.
}

type menuBarInfo struct {
	Size        uint32
	Bar         rect
	Menu, Hwnd  uintptr
	FocusedBits int32
}

// darkMenuBar reports whether the window draws its menu bar dark. A
// contrast theme keeps the system's colors, as it does for popup menus.
func (w *window) darkMenuBar() bool {
	return w.menuShown() && w.b.isDark() && !highContrastOn()
}

// menuBarRect is the menu bar in window coordinates, which the DCs of
// these messages use.
func (w *window) menuBarRect() (rect, bool) {
	mbi := menuBarInfo{Size: uint32(unsafe.Sizeof(menuBarInfo{}))}
	if ok, _, _ := procGetMenuBarInfo.Call(w.hwnd, objidMenu, 0, uintptr(unsafe.Pointer(&mbi))); ok == 0 {
		return rect{}, false
	}
	var wr rect
	procGetWindowRect.Call(w.hwnd, uintptr(unsafe.Pointer(&wr)))
	r := mbi.Bar
	r.Left, r.Right = r.Left-wr.Left, r.Right-wr.Left
	r.Top, r.Bottom = r.Top-wr.Top, r.Bottom-wr.Top
	return r, true
}

func fillColor(dc uintptr, r *rect, color uint32) {
	brush, _, _ := procCreateSolidBrush.Call(uintptr(color))
	procFillRect.Call(dc, uintptr(unsafe.Pointer(r)), brush)
	procDeleteObject.Call(brush)
}

// drawMenuBar paints the background of the bar.
func (w *window) drawMenuBar(lp uintptr) {
	um := (*uahMenu)(native(lp))
	if r, ok := w.menuBarRect(); ok {
		fillColor(um.DC, &r, darkMenuBar)
	}
}

// drawMenuBarItem paints an item of the bar: its backplate and its label.
func (w *window) drawMenuBarItem(lp uintptr) {
	d := (*uahDrawMenuItem)(native(lp))
	dc, r, state := d.Draw.DC, d.Draw.Item, d.Draw.ItemState
	back, fore := uint32(darkMenuBar), uint32(darkMenuText)
	switch {
	case state&(odsGrayed|odsDisabled) != 0:
		fore = darkMenuDisabled
	case state&odsSelected != 0:
		back = darkMenuOpen
	case state&odsHotLight != 0:
		back = darkMenuHot
	}
	fillColor(dc, &r, back)
	buf := make([]uint16, 256)
	n, _, _ := procGetMenuStringW.Call(d.Menu.Menu, uintptr(d.Position), uintptr(unsafe.Pointer(&buf[0])), uintptr(len(buf)), mfByPosition)
	flags := uintptr(dtCenter | dtVCenter | dtSingleLine)
	if state&odsNoAccel != 0 {
		flags |= dtHidePrefix
	}
	procSetBkMode.Call(dc, 1) // TRANSPARENT
	procSetTextColor.Call(dc, uintptr(fore))
	procDrawTextW.Call(dc, uintptr(unsafe.Pointer(&buf[0])), n, uintptr(unsafe.Pointer(&r)), flags)
}

// coverMenuLine paints over the light line the frame draws below the menu
// bar, after the frame is drawn.
func (w *window) coverMenuLine() {
	bar, ok := w.menuBarRect()
	if !ok {
		return
	}
	line := rect{bar.Left, bar.Bottom, bar.Right, bar.Bottom + 1}
	dc, _, _ := procGetWindowDC.Call(w.hwnd)
	if dc == 0 {
		return
	}
	fillColor(dc, &line, darkMenuBar)
	procReleaseDC.Call(w.hwnd, dc)
}

// darkMenuMessage draws the menu bar dark. It handles the messages only
// while the bar shows in the dark appearance.
func (w *window) darkMenuMessage(m uint32, wp, lp uintptr) (uintptr, bool) {
	switch m {
	case wmUAHDrawMenu, wmUAHDrawMenuItem, wmNCPaint, wmNCActivate:
	default:
		return 0, false
	}
	if !w.darkMenuBar() {
		return 0, false
	}
	switch m {
	case wmUAHDrawMenu:
		w.drawMenuBar(lp)
	case wmUAHDrawMenuItem:
		w.drawMenuBarItem(lp)
	default: // the frame, then the line over it
		r, _, _ := procDefWindowProcW.Call(w.hwnd, uintptr(m), wp, lp)
		w.coverMenuLine()
		return r, true
	}
	return 0, true
}
