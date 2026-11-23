//go:build windows && (amd64 || arm64)

package windows

import (
	"errors"
	"fmt"
	"strings"
	"syscall"
	"time"
	"unsafe"

	"github.com/egoist/mygo/internal/accelerator"
)

// The functions in this file drive native UI the way a user would, for the
// GUI tests in internal/e2e. They must run on the main thread.

// TestActivateMenuItem activates an item of a window's menu bar, found by
// the labels along its path, e.g. TestActivateMenuItem(hwnd, "File", "New").
func TestActivateMenuItem(hwnd uintptr, path ...string) error {
	w := theBackend.windows[hwnd]
	if w == nil || w.menu == nil {
		return fmt.Errorf("mygo: the window has no menu bar")
	}
	items := w.menu.Items
	for i, label := range path {
		found := false
		for _, it := range items {
			if it.Label != label {
				continue
			}
			found = true
			if i == len(path)-1 {
				for cmd, e := range theBackend.menus.entries {
					if e.uid == it.ID && e.owner == w.owner {
						theBackend.menuCommand(cmd, w)
						return nil
					}
				}
				return fmt.Errorf("mygo: menu item %q is not a command", label)
			}
			if it.Submenu == nil {
				return fmt.Errorf("mygo: menu %q has no submenu", label)
			}
			items = it.Submenu.Items
			break
		}
		if !found {
			return fmt.Errorf("mygo: no menu item %q", label)
		}
	}
	return nil
}

// TestMenuBarShown reports whether a window shows its menu bar.
func TestMenuBarShown(hwnd uintptr) bool {
	bar, _, _ := user32.NewProc("GetMenu").Call(hwnd)
	return bar != 0
}

// TestEnterMenuBar takes the keyboard to a window's menu bar, as Alt and
// F10 do, and returns once it leaves the menus. It reports false when the
// window cannot come to the front, which the keyboard needs.
func TestEnterMenuBar(hwnd uintptr) bool {
	if !TestForeground(hwnd) {
		return false
	}
	procSendMessageW.Call(hwnd, wmSysCommand, scKeyMenu, 0)
	return true
}

// TestForeground brings a window to the front, where the keyboard types,
// and reports whether it came.
func TestForeground(hwnd uintptr) bool {
	procSetForegroundWindow.Call(hwnd)
	fg, _, _ := procGetForegroundWindow.Call()
	return fg == hwnd
}

// TestEndMenu leaves the menus, as Escape does.
func TestEndMenu() { procEndMenu.Call() }

// TestPanelStyles reports whether a window is a topmost tool window
// without a window menu, owned by the hidden application window, as
// panels are.
func TestPanelStyles(hwnd uintptr) (tool, topmost, owned bool) {
	const gwOwner = 4
	ex := windowLong(hwnd, gwlExStyle)
	owner, _, _ := procGetNextWindow.Call(hwnd, gwOwner)
	menu := windowLong(hwnd, gwlStyle)&wsSysMenu != 0
	return ex&wsExToolWindow != 0 && !menu, ex&wsExTopmost != 0, owner != 0 && owner == theBackend.appHwnd
}

// TestActivateAccelerator runs the menu item of a window's shortcut, as its
// keys do in the page, and reports whether there is one.
func TestActivateAccelerator(hwnd uintptr, acc string) bool {
	w := theBackend.windows[hwnd]
	a, err := accelerator.Parse(acc, "windows")
	if w == nil || err != nil {
		return false
	}
	vk, ok := virtualKey(a.Key)
	if !ok {
		return false
	}
	cmd, ok := w.accels[accelKey{vk: vk, mods: a.Modifiers}]
	if ok {
		theBackend.menuCommand(cmd, w)
	}
	return ok
}

// TestSetDroppedFiles makes paths the files of the next drop on a window's
// page, as if they had been dragged there.
func TestSetDroppedFiles(hwnd uintptr, paths []string) {
	if w := theBackend.windows[hwnd]; w != nil {
		w.dropped = paths
	}
}

// TestPressKeys presses virtual keys together and releases them, like a
// keyboard would.
func TestPressKeys(keys ...byte) {
	proc := user32.NewProc("keybd_event")
	const keyUp = 0x2
	for _, k := range keys {
		proc.Call(uintptr(k), 0, 0, 0)
	}
	for i := len(keys) - 1; i >= 0; i-- {
		proc.Call(uintptr(keys[i]), 0, keyUp, 0)
	}
}

// TestCaptionButtons returns the window controls of a window with a hidden
// title bar, left to right, named after what WM_NCHITTEST answers over
// their middles, which Windows' snap layouts rely on.
func TestCaptionButtons(hwnd uintptr) []string {
	w := theBackend.windows[hwnd]
	if w == nil || w.caption == nil {
		return nil
	}
	var names []string
	for i := range w.caption.shown {
		hit, _, _ := procSendMessageW.Call(w.caption.buttons, wmNCHitTest, 0, captionButtonPoint(w.caption, i))
		switch hit {
		case htMinButton:
			names = append(names, "minimize")
		case htMaxButton:
			names = append(names, "maximize")
		case htClose:
			names = append(names, "close")
		default:
			names = append(names, fmt.Sprint(hit))
		}
	}
	return names
}

// TestPressCaptionButton clicks a window control of a window with a hidden
// title bar: "minimize", "maximize" or "close".
func TestPressCaptionButton(hwnd uintptr, name string) bool {
	w := theBackend.windows[hwnd]
	if w == nil || w.caption == nil {
		return false
	}
	for i, b := range w.caption.shown {
		if [...]string{"minimize", "maximize", "close"}[b] != name {
			continue
		}
		lp := captionButtonPoint(w.caption, i)
		procSendMessageW.Call(w.caption.buttons, wmNCLButtonDown, b.hitTest(), lp)
		procSendMessageW.Call(w.caption.buttons, wmNCLButtonUp, b.hitTest(), lp)
		return true
	}
	return false
}

// captionButtonPoint is the screen position of the middle of the i-th
// button, as the lParam of mouse messages.
func captionButtonPoint(c *captionBar, i int) uintptr {
	var r rect
	procGetWindowRect.Call(c.buttons, uintptr(unsafe.Pointer(&r)))
	bw := (r.Right - r.Left) / int32(len(c.shown))
	x, y := r.Left+int32(i)*bw+bw/2, r.Top+(r.Bottom-r.Top)/2
	return uintptr(uint16(x)) | uintptr(uint16(y))<<16
}

// TestCaptionColor puts the pointer over a window control of a hidden
// title bar, "minimize", "maximize" or "close", as far as the control
// knows, and returns the color on the screen beside its glyph once Windows
// has composed it: what the user sees, through UpdateLayeredWindow or
// DirectComposition. The pointer leaves after.
func TestCaptionColor(hwnd uintptr, name string) (r, g, b uint8, ok bool) {
	w := theBackend.windows[hwnd]
	if w == nil || w.caption == nil {
		return 0, 0, 0, false
	}
	c := w.caption
	for i, btn := range c.shown {
		if [...]string{"minimize", "maximize", "close"}[btn] != name {
			continue
		}
		c.setHot(i)
		defer c.setHot(-1)
		dwmapi.NewProc("DwmFlush").Call() // until the next frame shows what was drawn
		var rc rect
		procGetWindowRect.Call(c.buttons, uintptr(unsafe.Pointer(&rc)))
		bw := (rc.Right - rc.Left) / int32(len(c.shown))
		x, y := rc.Left+int32(i)*bw+3, rc.Top+(rc.Bottom-rc.Top)/2
		screen, _, _ := procGetDC.Call(0)
		defer procReleaseDC.Call(0, screen)
		px, _, _ := gdi32.NewProc("GetPixel").Call(screen, uintptr(x), uintptr(y))
		if px == 0xFFFFFFFF { // CLR_INVALID
			return 0, 0, 0, false
		}
		return uint8(px), uint8(px >> 8), uint8(px >> 16), true
	}
	return 0, 0, 0, false
}

// TestComposed reports whether a window has no redirection bitmap, for the
// material behind its page, and whether the controls of its hidden title
// bar show through the DirectComposition device in use.
func TestComposed(hwnd uintptr) (noRedirect, composed bool) {
	w := theBackend.windows[hwnd]
	if w == nil {
		return false, false
	}
	c := w.caption
	return w.noRedirect, c != nil && c.comp != nil && c.comp.dev != nil && c.comp.dev == theBackend.comp
}

// TestLoseComposition takes the DirectComposition device away, as a draw
// that finds it invalid does, or, with removed, signals that its GPU device
// was removed. It reports false without a device, or, with removed,
// without the event (Windows 10 before 1607).
func TestLoseComposition(removed bool) bool {
	c := theBackend.comp
	switch {
	case c == nil, removed && c.removed == 0:
		return false
	case removed:
		procSetEvent.Call(c.removed)
	default:
		theBackend.loseComposition(errors.New("taken away by a test"))
	}
	return true
}

// testFailWebViews is how many creations of webviews still fail
// (TestFailWebViews).
var testFailWebViews int

// TestFailWebViews makes the next n creations of webviews fail, as WebView2
// reports a failure, when it would have created them; 0 lets them be.
func TestFailWebViews(n int) { testFailWebViews = n }

// TestTopNonClient returns how many pixels at the top of a window are not
// its client area, and how many a hidden title bar keeps there on this
// version of Windows: one on Windows 11, none on Windows 10.
func TestTopNonClient(hwnd uintptr) (px, want int32) {
	var r rect
	procGetWindowRect.Call(hwnd, uintptr(unsafe.Pointer(&r)))
	var origin point
	procClientToScreen.Call(hwnd, uintptr(unsafe.Pointer(&origin)))
	if windows11() {
		want = 1
	}
	return origin.Y - r.Top, want
}

// TestClickSurface clicks (x, y), in DIPs, in a window showing native UI,
// with the messages a mouse sends. It reports false for a window showing a
// web page.
func TestClickSurface(hwnd uintptr, x, y float64) bool {
	w := theBackend.windows[hwnd]
	if w == nil || w.surface == nil {
		return false
	}
	scale := float64(w.surface.dpi()) / 96
	lp := uintptr(uint16(int16(y*scale)))<<16 | uintptr(uint16(int16(x*scale)))
	const mkLButton = 1
	procSendMessageW.Call(w.surface.hwnd, wmMouseMove, 0, lp)
	procSendMessageW.Call(w.surface.hwnd, wmLButtonDown, mkLButton, lp)
	procSendMessageW.Call(w.surface.hwnd, wmLButtonUp, 0, lp)
	return true
}

// TestInputClick clicks a point of a window showing native UI with the
// system's input, as a mouse does: the window it lands on activates as
// for the user. The system delivers the click later, through the thread's
// queue: TestInputClick handles the messages that come until the release
// reaches a window of the app, so that the app handled the click when it
// returns. It reports false for a window showing a web page, and when no
// release came in 10 seconds, missed describes the window under the
// point, which took the click.
func TestInputClick(hwnd uintptr, x, y float64) (ok bool, missed string) {
	w := theBackend.windows[hwnd]
	if w == nil || w.surface == nil {
		return false, ""
	}
	scale := float64(w.surface.dpi()) / 96
	pt := point{int32(x * scale), int32(y * scale)}
	procClientToScreen.Call(w.surface.hwnd, uintptr(unsafe.Pointer(&pt)))
	user32.NewProc("SetCursorPos").Call(uintptr(pt.X), uintptr(pt.Y))
	mouse := user32.NewProc("mouse_event")
	const leftDown, leftUp = 0x0002, 0x0004
	mouse.Call(leftDown, 0, 0, 0, 0)
	mouse.Call(leftUp, 0, 0, 0, 0)
	for deadline := time.Now().Add(10 * time.Second); time.Now().Before(deadline); {
		procMsgWaitForMultipleObjectsEx.Call(0, 0, 100, qsAllInput, mwmoInputAvailable)
		var m msg
		for {
			if r, _, _ := procPeekMessageW.Call(uintptr(unsafe.Pointer(&m)), 0, 0, 0, pmRemove); r == 0 {
				break
			}
			if m.Message == wmQuit {
				procPostQuitMessage.Call(m.WParam) // for the loop running outside
				return true, "the app quit"
			}
			procTranslateMessage.Call(uintptr(unsafe.Pointer(&m)))
			procDispatchMessageW.Call(uintptr(unsafe.Pointer(&m)))
			if m.Message == wmLButtonUp {
				return true, ""
			}
		}
	}
	return true, windowAt(pt)
}

// windowAt describes the top-level window at a point of the screen, in
// physical pixels: its class, title and process.
func windowAt(pt point) string {
	h, _, _ := user32.NewProc("WindowFromPoint").Call(uintptr(uint32(pt.X)) | uintptr(uint32(pt.Y))<<32)
	if h == 0 {
		return "no window"
	}
	const gaRoot = 2
	if root, _, _ := user32.NewProc("GetAncestor").Call(h, gaRoot); root != 0 {
		h = root
	}
	class := make([]uint16, 256)
	user32.NewProc("GetClassNameW").Call(h, uintptr(unsafe.Pointer(&class[0])), uintptr(len(class)))
	title := make([]uint16, 256)
	procGetWindowTextW.Call(h, uintptr(unsafe.Pointer(&title[0])), uintptr(len(title)))
	var pid uint32
	user32.NewProc("GetWindowThreadProcessId").Call(h, uintptr(unsafe.Pointer(&pid)))
	return fmt.Sprintf("%#x, class %q, title %q, process %d (this one %d)", h, syscall.UTF16ToString(class), syscall.UTF16ToString(title), pid, syscall.Getpid())
}

// TestSideButton clicks a mouse's back or forward button (XBUTTON1 and
// XBUTTON2) in a window showing native UI, with the messages a mouse
// sends.
func TestSideButton(hwnd uintptr, back bool) bool {
	w := theBackend.windows[hwnd]
	if w == nil || w.surface == nil {
		return false
	}
	button, held := uintptr(2), uintptr(0x40) // XBUTTON2, MK_XBUTTON2
	if back {
		button, held = 1, 0x20 // XBUTTON1, MK_XBUTTON1
	}
	scale := float64(w.surface.dpi()) / 96
	lp := uintptr(uint16(int16(20*scale)))<<16 | uintptr(uint16(int16(20*scale)))
	procSendMessageW.Call(w.surface.hwnd, wmXButtonDown, button<<16|held, lp)
	procSendMessageW.Call(w.surface.hwnd, wmXButtonUp, button<<16, lp)
	return true
}

// TestDragSurface presses the primary button at the first of points, in
// DIPs, in a window showing native UI, moves the pointer through the
// others, and releases it at the last, with the messages a mouse sends.
func TestDragSurface(hwnd uintptr, points [][2]float64) bool {
	w := theBackend.windows[hwnd]
	if w == nil || w.surface == nil || len(points) < 2 {
		return false
	}
	scale := float64(w.surface.dpi()) / 96
	lp := func(p [2]float64) uintptr {
		return uintptr(uint16(int16(p[1]*scale)))<<16 | uintptr(uint16(int16(p[0]*scale)))
	}
	const mkLButton = 1
	procSendMessageW.Call(w.surface.hwnd, wmMouseMove, 0, lp(points[0]))
	procSendMessageW.Call(w.surface.hwnd, wmLButtonDown, mkLButton, lp(points[0]))
	for _, p := range points[1:] {
		procSendMessageW.Call(w.surface.hwnd, wmMouseMove, mkLButton, lp(p))
	}
	procSendMessageW.Call(w.surface.hwnd, wmLButtonUp, 0, lp(points[len(points)-1]))
	return true
}

// TestRightClickSurface clicks (x, y), in DIPs, in a window showing native
// UI with the secondary button, as TestClickSurface does with the primary.
func TestRightClickSurface(hwnd uintptr, x, y float64) bool {
	w := theBackend.windows[hwnd]
	if w == nil || w.surface == nil {
		return false
	}
	scale := float64(w.surface.dpi()) / 96
	lp := uintptr(uint16(int16(y*scale)))<<16 | uintptr(uint16(int16(x*scale)))
	const mkRButton = 2
	procSendMessageW.Call(w.surface.hwnd, wmMouseMove, 0, lp)
	procSendMessageW.Call(w.surface.hwnd, wmRButtonDown, mkRButton, lp)
	procSendMessageW.Call(w.surface.hwnd, wmRButtonUp, 0, lp)
	return true
}

// TestPopups returns the labels of the items of the popup menus the main
// thread shows, "-" for separators.
func TestPopups() [][]string {
	var menus [][]string
	for _, h := range threadPopups() {
		menus = append(menus, popupLabels(h))
	}
	return menus
}

// TestChoosePopupItem chooses the item labeled label of a popup menu the
// main thread shows, with the keys a keyboard presses in it, and reports
// whether there is one.
func TestChoosePopupItem(label string) bool {
	const vkReturn, vkUp, vkDown = 0x0D, 0x26, 0x28
	for _, h := range threadPopups() {
		labels := popupLabels(h)
		target := -1
		for i, l := range labels {
			if l == label {
				target = i
			}
		}
		if target < 0 {
			continue
		}
		// The arrows move to the next item past separators, from the one
		// highlighted; Down to the first when none is.
		hmenu, _, _ := procSendMessageW.Call(h, mnGetHMenu, 0, 0)
		from := -1
		for i := range labels {
			if state, _, _ := procGetMenuState.Call(hmenu, uintptr(i), mfByPosition); state&mfHilite != 0 {
				from = i
			}
		}
		for i := from + 1; i <= target; i++ {
			if labels[i] != "-" {
				postMessage(h, wmKeyDown, vkDown, 0)
			}
		}
		for i := target; i < from; i++ {
			if labels[i] != "-" {
				postMessage(h, wmKeyDown, vkUp, 0)
			}
		}
		postMessage(h, wmKeyDown, vkReturn, 0)
		return true
	}
	return false
}

const (
	mnGetHMenu = 0x01E1
	mfHilite   = 0x0080
)

// threadPopups returns the windows of the popup menus the main thread
// shows: menus of other threads and processes are not the app's.
func threadPopups() []uintptr {
	findWindowEx := user32.NewProc("FindWindowExW")
	threadOf := user32.NewProc("GetWindowThreadProcessId")
	class := u16("#32768")
	thread := uintptr(currentThreadID())
	var popups []uintptr
	for after := uintptr(0); ; {
		h, _, _ := findWindowEx.Call(0, after, uintptr(unsafe.Pointer(class)), 0)
		if h == 0 {
			return popups
		}
		after = h
		if t, _, _ := threadOf.Call(h, 0); t != thread {
			continue
		}
		if visible, _, _ := procIsWindowVisible.Call(h); visible != 0 {
			popups = append(popups, h)
		}
	}
}

// popupLabels returns the labels of the items of the popup menu of a
// menu window, without the accelerators they show.
func popupLabels(h uintptr) []string {
	hmenu, _, _ := procSendMessageW.Call(h, mnGetHMenu, 0, 0)
	if hmenu == 0 {
		return nil
	}
	n, _, _ := procGetMenuItemCount.Call(hmenu)
	labels := []string{}
	for i := range int(int32(n)) {
		if state, _, _ := procGetMenuState.Call(hmenu, uintptr(i), mfByPosition); state&mfSeparator != 0 {
			labels = append(labels, "-")
			continue
		}
		buf := make([]uint16, 256)
		procGetMenuStringW.Call(hmenu, uintptr(i), uintptr(unsafe.Pointer(&buf[0])), uintptr(len(buf)), mfByPosition)
		label, _, _ := strings.Cut(syscall.UTF16ToString(buf), "\t")
		labels = append(labels, label)
	}
	return labels
}

var (
	procPrintWindow            = user32.NewProc("PrintWindow")
	procGetPixel               = gdi32.NewProc("GetPixel")
	procCreateCompatibleBitmap = gdi32.NewProc("CreateCompatibleBitmap")
)

// TestMenuBarColors draws a window as it shows and returns the colors, as
// COLORREFs, of its menu bar past the last item and of the line below it.
func TestMenuBarColors(hwnd uintptr) (bar, line uint32, ok bool) {
	w := theBackend.windows[hwnd]
	if w == nil {
		return 0, 0, false
	}
	r, ok := w.menuBarRect()
	if !ok {
		return 0, 0, false
	}
	var wr rect
	procGetWindowRect.Call(hwnd, uintptr(unsafe.Pointer(&wr)))
	screen, _, _ := procGetDC.Call(0)
	defer procReleaseDC.Call(0, screen)
	dc, _, _ := procCreateCompatibleDC.Call(screen)
	defer procDeleteDC.Call(dc)
	bmp, _, _ := procCreateCompatibleBitmap.Call(screen, uintptr(wr.Right-wr.Left), uintptr(wr.Bottom-wr.Top))
	defer procDeleteObject.Call(bmp)
	old, _, _ := procSelectObject.Call(dc, bmp)
	defer procSelectObject.Call(dc, old)
	const pwRenderFullContent = 2
	procPrintWindow.Call(hwnd, dc, pwRenderFullContent)
	x := uintptr(r.Right - 4)
	b, _, _ := procGetPixel.Call(dc, x, uintptr((r.Top+r.Bottom)/2))
	l, _, _ := procGetPixel.Call(dc, x, uintptr(r.Bottom))
	return uint32(b), uint32(l), true
}

// TestRecentDocuments returns the paths in the Recent category of the app's
// jump list, newest first.
func TestRecentDocuments() []string {
	var (
		clsid = guid("86bec222-30f2-47e0-9f25-60d11cd75c28") // ApplicationDocumentLists
		iid   = guid("3c594f9f-9f30-47a1-979a-c9e83d3d0a06") // IApplicationDocumentLists
	)
	const adlGetList, adltRecent = 4, 0
	lists, err := coCreate(clsid, iid)
	if err != nil {
		return nil
	}
	defer release(lists)
	var arr uintptr
	if failed(comCall(lists, adlGetList, adltRecent, 0, uintptr(unsafe.Pointer(&iidIObjectArray)), uintptr(unsafe.Pointer(&arr)))) {
		return nil
	}
	defer release(arr)
	var n uint32
	comCall(arr, oaGetCount, uintptr(unsafe.Pointer(&n)))
	var paths []string
	for i := range n {
		var item uintptr
		if failed(comCall(arr, oaGetAt, uintptr(i), uintptr(unsafe.Pointer(&iidIShellItem)), uintptr(unsafe.Pointer(&item)))) {
			continue
		}
		paths = append(paths, itemPath(item))
		release(item)
	}
	return paths
}

// TestWindowTitle returns the text of a window's title bar.
func TestWindowTitle(hwnd uintptr) string { return (&window{hwnd: hwnd}).Title() }
