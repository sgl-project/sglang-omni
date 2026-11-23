//go:build windows && (amd64 || arm64)

package windows

import (
	"encoding/json"
	"strings"
	"unsafe"

	"github.com/egoist/mygo/internal/accelerator"
	"github.com/egoist/mygo/internal/platform"
)

// menuEntry is a native menu item: WM_COMMAND reports 16-bit command ids,
// which map back to the items they were built from.
type menuEntry struct {
	uid   int
	role  string
	owner int
	menu  uintptr // the HMENU holding the item
	radio bool
	accel string
}

type menuTable struct {
	entries   map[uint16]*menuEntry
	next      uint16
	nextOwner int
}

func newMenuTable() menuTable { return menuTable{entries: map[uint16]*menuEntry{}} }

func (t *menuTable) newOwner() int {
	t.nextOwner++
	return t.nextOwner
}

func (t *menuTable) add(e *menuEntry) uint16 {
	for {
		t.next++
		if t.next == 0 || t.next >= 0xF000 { // 0xF000 and up are system commands
			t.next = 1
		}
		if t.entries[t.next] == nil {
			t.entries[t.next] = e
			return t.next
		}
	}
}

// drop forgets the items of an owner (a menu bar, popup or tray menu).
func (t *menuTable) drop(owner int) {
	if owner == 0 {
		return
	}
	for id, e := range t.entries {
		if e.owner == owner {
			delete(t.entries, id)
		}
	}
}

// buildMenu creates a native menu bar or popup menu.
func (b *Backend) buildMenu(m *platform.Menu, bar bool, owner int, accels map[accelKey]uint16) uintptr {
	var h uintptr
	if bar {
		h, _, _ = procCreateMenu.Call()
	} else {
		h, _, _ = procCreatePopupMenu.Call()
	}
	for _, it := range m.Items {
		if !it.Visible {
			continue
		}
		if it.Type == platform.MenuItemSeparator {
			procAppendMenuW.Call(h, mfSeparator, 0, 0)
			continue
		}
		flags := uintptr(mfString)
		if !it.Enabled {
			flags |= mfGrayed
		}
		if it.Submenu != nil {
			sub := b.buildMenu(it.Submenu, false, owner, accels)
			procAppendMenuW.Call(h, flags|mfPopup, sub, uintptr(unsafe.Pointer(u16(it.Label))))
			continue
		}
		if it.Checked {
			flags |= mfChecked
		}
		e := &menuEntry{uid: it.ID, role: it.Role, owner: owner, menu: h, radio: it.Type == platform.MenuItemRadio}
		cmd := b.menus.add(e)
		label := it.Label
		if a, err := accelerator.Parse(it.Accelerator, "windows"); err == nil && it.Accelerator != "" {
			e.accel = acceleratorText(a)
			label += "\t" + e.accel
			if vk, ok := virtualKey(a.Key); ok && accels != nil {
				accels[accelKey{vk: vk, mods: a.Modifiers}] = cmd
			}
		}
		procAppendMenuW.Call(h, flags, uintptr(cmd), uintptr(unsafe.Pointer(u16(label))))
		if e.radio {
			setItemInfo(h, cmd, label, it.Enabled, it.Checked, true)
		}
	}
	return h
}

func setItemInfo(menu uintptr, cmd uint16, label string, enabled, checked, radio bool) {
	type menuItemInfo struct {
		Size         uint32
		Mask         uint32
		Type         uint32
		State        uint32
		ID           uint32
		SubMenu      uintptr
		BmpChecked   uintptr
		BmpUnchecked uintptr
		ItemData     uintptr
		TypeData     *uint16
		Cch          uint32
		BmpItem      uintptr
	}
	mi := menuItemInfo{Mask: miimState | miimFType | miimString, TypeData: u16(label)}
	mi.Size = uint32(unsafe.Sizeof(mi))
	if radio {
		mi.Type = mftRadioCheck
	}
	if checked {
		mi.State |= mfsChecked
	}
	if !enabled {
		mi.State |= mfsDisabled
	}
	procSetMenuItemInfoW.Call(menu, uintptr(cmd), 0, uintptr(unsafe.Pointer(&mi)))
}

// installMenu shows m as the window's menu bar (nil removes it).
func (w *window) installMenu(m *platform.Menu) {
	old := w.hmenu
	w.b.menus.drop(w.owner)
	w.menu, w.hmenu, w.owner, w.accels = m, 0, 0, map[accelKey]uint16{}
	if m != nil && len(m.Items) > 0 {
		w.owner = w.b.menus.newOwner()
		w.hmenu = w.b.buildMenu(m, true, w.owner, w.accels)
	}
	w.attachMenu()
	if old != 0 {
		procDestroyMenu.Call(old)
	}
}

// menuShown reports whether the menu bar is on the window. One that hides
// is there only while the keyboard is in it, and a window without a caption,
// in full screen or with a material behind its page has none.
func (w *window) menuShown() bool {
	return w.hmenu != 0 && !w.barless() && (!w.autoHideMenu || w.revealed)
}

// barless reports a window that has no menu bar: Alt and F10 open its menus
// in a popup instead. A window without a caption, or in full screen, has no
// room for it, and one without a redirection bitmap would not show what GDI
// draws of it, though it took clicks.
func (w *window) barless() bool { return w.captionless() || w.fullScreen || w.noRedirect }

// attachMenu puts the menu bar on the window, or takes it off one whose bar
// hides. Its shortcuts work either way: they come from the webview.
func (w *window) attachMenu() {
	bar := uintptr(0)
	if w.menuShown() {
		bar = w.hmenu
	}
	procSetMenu.Call(w.hwnd, bar)
}

// revealMenu shows a menu bar that hides for the menu loop, which Alt or
// F10 starts with SC_KEYMENU and which runs inside DefWindowProc.
func (w *window) revealMenu(wp, lp uintptr) uintptr {
	w.revealed = true
	w.attachMenu()
	r, _, _ := procDefWindowProcW.Call(w.hwnd, wmSysCommand, wp, lp)
	w.revealed = false
	if !w.closed {
		w.attachMenu()
	}
	return r
}

// popupMenuBar opens the menus of the menu bar from the top-left corner of
// a window without room for the bar, below the title bar the page draws,
// when Alt or F10 would take the keyboard to the bar; Alt and a letter
// (key) open the menu of the letter in it, as in the bar. The popup holds
// the bar's own submenus, so their items keep their states and commands,
// and gives them back before it is destroyed.
func (w *window) popupMenuBar(key uintptr) {
	n, _, _ := procGetMenuItemCount.Call(w.hmenu)
	count := int(int32(n))
	if count <= 0 {
		return
	}
	popup, _, _ := procCreatePopupMenu.Call()
	for i := range count {
		buf := make([]uint16, 256)
		procGetMenuStringW.Call(w.hmenu, uintptr(i), uintptr(unsafe.Pointer(&buf[0])), uintptr(len(buf)), mfByPosition)
		state, _, _ := procGetMenuState.Call(w.hmenu, uintptr(i), mfByPosition)
		flags := uintptr(mfString) | state&mfGrayed
		item, _, _ := procGetSubMenu.Call(w.hmenu, uintptr(i))
		if item != 0 {
			flags |= mfPopup
		} else {
			item, _, _ = procGetMenuItemID.Call(w.hmenu, uintptr(i))
		}
		procAppendMenuW.Call(popup, flags, item, uintptr(unsafe.Pointer(&buf[0])))
	}
	pt := point{Y: toPx(w.TitleBar().Height, dpiOf(w.hwnd))}
	procClientToScreen.Call(w.hwnd, uintptr(unsafe.Pointer(&pt)))
	if key != 0 {
		// The menu loop takes the letter as typed in the popup.
		w.menuKey = menuKey{popup: popup, key: key}
		postMessage(w.hwnd, wmChar, key, 0)
	}
	cmd := w.b.trackPopup(popup, w.hwnd, pt, 0)
	w.menuKey = menuKey{}
	for i := count - 1; i >= 0; i-- {
		procRemoveMenu.Call(popup, uintptr(i), mfByPosition)
	}
	procDestroyMenu.Call(popup)
	if cmd != 0 && !w.closed {
		w.b.menuCommand(cmd, w)
	}
}

// menuKey is the letter of Alt and a letter that opened popupMenuBar, and
// the popup.
type menuKey struct{ popup, key uintptr }

// menuChar answers WM_MENUCHAR, which a menu sends its window for a letter
// that names none of its items. Alt and a letter that names no menu of the
// bar open none, as in the bar: the popup closes. A letter typed in it
// later cannot be that one, which names none of its items.
func (w *window) menuChar(wp, lp uintptr) (uintptr, bool) {
	if k := w.menuKey; k.popup != 0 && lp == k.popup && wp&0xFFFF == k.key {
		return mncClose << 16, true
	}
	return 0, false
}

func (b *Backend) SetApplicationMenu(m *platform.Menu) {
	b.appMenu = m
	for _, w := range b.windows {
		if !w.ownMenu && w.hasAppMenu() {
			w.installMenu(m)
		}
	}
}

func (b *Backend) UpdateMenuItem(it *platform.MenuItem) {
	for cmd, e := range b.menus.entries {
		if e.uid != it.ID {
			continue
		}
		label := it.Label
		if e.accel != "" {
			label += "\t" + e.accel
		}
		setItemInfo(e.menu, cmd, label, it.Enabled, it.Checked, e.radio)
	}
	for _, w := range b.windows {
		if w.menuShown() {
			procDrawMenuBar.Call(w.hwnd)
		}
	}
}

// menuCommand runs the item behind a command id.
func (b *Backend) menuCommand(cmd uint16, w *window) {
	e := b.menus.entries[cmd]
	if e == nil {
		return
	}
	if w == nil {
		w = b.focusedWindow()
	}
	if w != nil && editRole(w, e.role) {
		return
	}
	if w != nil {
		w.h.MenuItemClicked(e.uid)
	} else {
		b.h.MenuItemClicked(e.uid)
	}
}

func (b *Backend) focusedWindow() *window {
	fg, _, _ := procGetForegroundWindow.Call()
	return b.windows[fg]
}

// editRole performs the edit roles in the page. The page needs a user
// gesture for them, which DevTools evaluation provides; paste types the
// clipboard's text, since pages may not read the clipboard.
func editRole(w *window, role string) bool {
	commands := map[string]string{
		"undo": "undo", "redo": "redo", "cut": "cut", "copy": "copy",
		"delete": "delete", "selectAll": "selectAll",
	}
	if v := w.focusedWebView(); v != nil {
		// The page of a web view in the content has the keyboard.
		return editRole(v, role)
	}
	if w.surface != nil {
		if _, ok := commands[role]; !ok && role != "paste" && role != "pasteAndMatchStyle" {
			return false
		}
		w.h.SurfaceEvent(platform.SurfaceEvent{Kind: platform.SurfaceCommand, Text: role})
		return true
	}
	switch role {
	case "paste", "pasteAndMatchStyle":
		text := clipboard{w.b}.ReadText()
		params, _ := json.Marshal(map[string]string{"text": text})
		w.withWebView(func() { w.devtools("Input.insertText", string(params), nil) })
		return true
	}
	cmd, ok := commands[role]
	if !ok {
		// The Format menu's, in the page with the keyboard.
		cmd, ok = formatCommands[role]
	}
	if !ok {
		return false
	}
	params, _ := json.Marshal(map[string]any{"expression": "document.execCommand(" + jsString(cmd) + ")", "userGesture": true})
	w.withWebView(func() { w.devtools("Runtime.evaluate", string(params), nil) })
	return true
}

// formatCommands are the page's commands of the Format menu's roles; in
// native UI, they go to the content as the other roles.
var formatCommands = map[string]string{
	"bold": "bold", "italic": "italic", "underline": "underline", "strikethrough": "strikeThrough",
	"alignLeft": "justifyLeft", "alignCenter": "justifyCenter", "alignRight": "justifyRight",
}

func jsString(s string) string {
	b, _ := json.Marshal(s)
	return string(b)
}

func (b *Backend) PopupMenu(m *platform.Menu, pw platform.Window, pos *platform.Point) {
	owner := b.appHwnd
	w, _ := pw.(*window)
	if w != nil && !w.closed {
		owner = w.hwnd
	} else {
		w = nil
	}
	id := b.menus.newOwner()
	defer b.menus.drop(id)
	menu := b.buildMenu(m, false, id, nil)
	defer procDestroyMenu.Call(menu)

	var pt point
	if pos != nil && w != nil {
		dpi := dpiOf(w.hwnd)
		pt = point{toPx(pos.X, dpi), toPx(pos.Y, dpi)}
		procClientToScreen.Call(w.hwnd, uintptr(unsafe.Pointer(&pt)))
	} else {
		procGetCursorPos.Call(uintptr(unsafe.Pointer(&pt)))
	}
	var flags uintptr
	if m.RightToLeft {
		// Its right edge at the point, laid out from the right.
		flags = tpmRightAlign | tpmLayoutRTL
	}
	cmd := b.trackPopup(menu, owner, pt, flags)
	if cmd != 0 {
		b.menuCommand(cmd, w)
	}
}

// trackPopup shows a popup menu, with TrackPopupMenuEx's flags besides
// those it always gives, and returns the chosen command, or 0.
func (b *Backend) trackPopup(menu, owner uintptr, pt point, flags uintptr) uint16 {
	// The owner must be in the foreground, or the menu does not close
	// when the user clicks elsewhere.
	procSetForegroundWindow.Call(owner)
	cmd, _, _ := procTrackPopupMenuEx.Call(menu, tpmReturnCmd|tpmRightButton|flags, uintptr(pt.X), uintptr(pt.Y), owner, 0)
	postMessage(owner, 0, 0, 0) // WM_NULL, as the documentation asks
	return uint16(cmd)
}

// Keyboard accelerators.

type accelKey struct {
	vk   uint32
	mods accelerator.Modifiers
}

func currentModifiers() accelerator.Modifiers {
	down := func(vk uintptr) bool {
		r, _, _ := procGetKeyState.Call(vk)
		return int16(r) < 0
	}
	var m accelerator.Modifiers
	if down(vkControl) {
		m |= accelerator.Ctrl
	}
	if down(vkMenu) {
		m |= accelerator.Alt
	}
	if down(vkShift) {
		m |= accelerator.Shift
	}
	if down(vkLWin) || down(vkRWin) {
		m |= accelerator.Super
	}
	return m
}

func acceleratorText(a accelerator.Accelerator) string {
	var parts []string
	if a.Has(accelerator.Ctrl) {
		parts = append(parts, "Ctrl")
	}
	if a.Has(accelerator.Alt) {
		parts = append(parts, "Alt")
	}
	if a.Has(accelerator.Shift) {
		parts = append(parts, "Shift")
	}
	if a.Has(accelerator.Super) {
		parts = append(parts, "Win")
	}
	key := a.Key
	if len(key) == 1 {
		key = strings.ToUpper(key)
	}
	return strings.Join(append(parts, key), "+")
}

var namedVirtualKeys = map[string]uint32{
	"Enter": 0x0D, "Tab": 0x09, "Space": 0x20, "Backspace": 0x08, "Delete": 0x2E, "Insert": 0x2D,
	"Escape": 0x1B, "Up": 0x26, "Down": 0x28, "Left": 0x25, "Right": 0x27, "Home": 0x24, "End": 0x23,
	"PageUp": 0x21, "PageDown": 0x22, "VolumeUp": 0xAF, "VolumeDown": 0xAE, "VolumeMute": 0xAD,
	"MediaNextTrack": 0xB0, "MediaPreviousTrack": 0xB1, "MediaStop": 0xB2, "MediaPlayPause": 0xB3,
	"PrintScreen": 0x2C, "NumDec": 0x6E, "NumAdd": 0x6B, "NumSub": 0x6D, "NumMult": 0x6A, "NumDiv": 0x6F,
	"CapsLock": 0x14, "NumLock": 0x90, "ScrollLock": 0x91,
	",": 0xBC, "-": 0xBD, ".": 0xBE, "=": 0xBB, "+": 0xBB, ";": 0xBA, "/": 0xBF, "`": 0xC0,
	"[": 0xDB, "\\": 0xDC, "]": 0xDD, "'": 0xDE,
}

// virtualKey returns the virtual-key code of an accelerator key.
func virtualKey(key string) (uint32, bool) {
	if vk, ok := namedVirtualKeys[key]; ok {
		return vk, true
	}
	if len(key) == 1 {
		c := key[0]
		switch {
		case c >= 'a' && c <= 'z':
			return uint32(c - 'a' + 'A'), true
		case c >= '0' && c <= '9':
			return uint32(c), true
		}
	}
	if strings.HasPrefix(key, "F") {
		if n := atoi(key[1:]); n >= 1 && n <= 24 {
			return 0x70 + uint32(n-1), true
		}
	}
	if strings.HasPrefix(key, "Num") {
		if n := atoi(key[3:]); n >= 0 && n <= 9 && len(key) == 4 {
			return 0x60 + uint32(n), true
		}
	}
	return 0, false
}

func atoi(s string) int {
	if s == "" {
		return -1
	}
	n := 0
	for _, c := range s {
		if c < '0' || c > '9' {
			return -1
		}
		n = n*10 + int(c-'0')
	}
	return n
}
