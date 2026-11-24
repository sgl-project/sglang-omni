//go:build linux && (amd64 || arm64)

package linux

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"unicode/utf8"

	"github.com/ebitengine/purego"

	"github.com/egoist/mygo/internal/accelerator"
	"github.com/egoist/mygo/internal/platform"
)

// Native menu items are tracked per owner (a window's menu bar, a popup, a
// tray) so they can be updated in place and forgotten when rebuilt.
type nativeItem struct {
	owner    int
	item     ptr
	check    bool
	windowID int // menu bar or popup's window; 0 for a tray
}

var (
	menuOwners int
	itemsByID  = map[int][]nativeItem{}
	itemRoles  = map[int]string{}

	cbMenuActivate   ptr
	cbMenuDeactivate ptr

	// Menu bars that hide (window.menuHides).
	cbMenuKey, cbCanActivateAccel, cbMenuBarDeactivate ptr
)

// modifierMask holds the GDK modifiers that make a key a shortcut: Shift,
// Control, Alt, Super, Hyper and Meta.
const modifierMask = 1<<0 | 1<<2 | 1<<3 | 1<<26 | 1<<27 | 1<<28

// editCommands are handled by the focused webview itself.
var editCommands = map[string]string{
	"undo": "Undo", "redo": "Redo", "cut": "Cut", "copy": "Copy", "paste": "Paste",
	"pasteAndMatchStyle": "PasteAsPlainText", "delete": "Delete", "selectAll": "SelectAll",
	// The Format menu's.
	"bold": "Bold", "italic": "Italic", "underline": "Underline", "strikethrough": "Strikethrough",
	"alignLeft": "JustifyLeft", "alignCenter": "JustifyCenter", "alignRight": "JustifyRight",
}

func initMenuCallbacks() {
	cbMenuActivate = purego.NewCallback(func(item, data ptr) {
		if settingState {
			return
		}
		b := theBackend
		id := int(data)
		// A submenu may have focus. Resolve this native item's window, as
		// the same Go item may appear in several windows' menu bars.
		var w *window
		for _, n := range itemsByID[id] {
			if n.item == item && n.windowID != 0 {
				w = b.window(ptr(n.windowID))
				if w == nil {
					return
				}
				break
			}
		}
		if cmd, ok := editCommands[itemRoles[id]]; ok {
			if w == nil {
				for _, x := range b.windows {
					if !x.closed && gtkWindowIsActive(x.win) {
						w = x
						break
					}
				}
			}
			var v *window
			if w != nil {
				v = w.focusedWebView()
			}
			if v != nil {
				// The page of a web view in the content has the keyboard.
				webkitWebViewExecuteEditingCommand(v.web, cs(cmd))
			} else if w != nil {
				if w.surface != nil {
					role := itemRoles[id]
					w.h.SurfaceEvent(platform.SurfaceEvent{Kind: platform.SurfaceCommand, Text: role})
				} else {
					webkitWebViewExecuteEditingCommand(w.web, cs(cmd))
				}
			}
			return
		}
		if w != nil {
			w.h.MenuItemClicked(id)
		} else {
			b.h.MenuItemClicked(id)
		}
	})
	cbMenuDeactivate = purego.NewCallback(func(menu, loop ptr) {
		gMainLoopQuit(loop)
	})
	// Alt pressed and released alone, or F10, show a menu bar that hides.
	cbMenuKey = purego.NewCallback(func(widget, event, data ptr) bool {
		w := theBackend.window(data)
		if w == nil {
			return false
		}
		if !w.menuHides() || w.menubar == 0 || gtkWidgetGetVisible(w.menubar) {
			w.altAlone = false
			return false
		}
		// GdkEventKey: type 0, state 24, keyval 28.
		press := field[int32](event, 0) == 8 // GDK_KEY_PRESS
		state, key := field[uint32](event, 24), field[uint32](event, 28)
		alt := key == 0xffe9 || key == 0xffea // Alt_L, Alt_R
		switch {
		case press && key == 0xffc7 && state&modifierMask == 0: // F10
			w.revealMenu()
			return true
		case press:
			w.altAlone = alt && state&modifierMask == 0
		case alt && w.altAlone:
			w.altAlone = false
			w.revealMenu()
		default:
			w.altAlone = false
		}
		return false
	})
	// GTK activates the shortcuts of widgets on screen only: a menu bar that
	// hides keeps its own.
	cbCanActivateAccel = purego.NewCallback(func(widget ptr, signal uint32, data ptr) bool {
		w := theBackend.window(data)
		return w != nil && w.menuHides()
	})
	cbMenuBarDeactivate = purego.NewCallback(func(shell, data ptr) {
		if w := theBackend.window(data); w != nil && w.menuHides() {
			gtkWidgetHide(shell)
		}
	})
}

func newOwner() int {
	menuOwners++
	return menuOwners
}

func dropOwner(owner int) {
	if owner == 0 {
		return
	}
	for id, list := range itemsByID {
		kept := list[:0]
		for _, n := range list {
			if n.owner != owner {
				kept = append(kept, n)
			}
		}
		if len(kept) == 0 {
			delete(itemsByID, id)
		} else {
			itemsByID[id] = kept
		}
	}
}

// buildMenu fills a GtkMenuShell with items.
func buildMenu(shell ptr, m *platform.Menu, accel ptr, owner, windowID int) {
	for _, it := range m.Items {
		gtkMenuShellAppend(shell, buildItem(it, accel, owner, windowID))
	}
}

func buildItem(it *platform.MenuItem, accel ptr, owner, windowID int) ptr {
	var item ptr
	check := false
	switch it.Type {
	case platform.MenuItemSeparator:
		item = gtkSeparatorMenuItemNew()
	case platform.MenuItemCheckbox, platform.MenuItemRadio:
		item = gtkCheckMenuItemNewWithLabel(cs(it.Label))
		gtkCheckMenuItemSetDrawAsRadio(item, it.Type == platform.MenuItemRadio)
		check = true
	default:
		item = gtkMenuItemNewWithLabel(cs(it.Label))
	}
	if it.Submenu != nil {
		sub := gtkMenuNew()
		buildMenu(sub, it.Submenu, accel, owner, windowID)
		gtkMenuItemSetSubmenu(item, sub)
	} else if it.Type != platform.MenuItemSeparator {
		connect(item, "activate", cbMenuActivate, ptr(it.ID))
		itemRoles[it.ID] = it.Role
		if accel != 0 && it.Accelerator != "" {
			if key, mods, ok := gtkAccelerator(it.Accelerator); ok {
				gtkWidgetAddAccelerator(item, cs("activate"), accel, key, mods, 1) // GTK_ACCEL_VISIBLE
			}
		}
	}
	applyState(item, it, check)
	itemsByID[it.ID] = append(itemsByID[it.ID], nativeItem{owner: owner, item: item, check: check, windowID: windowID})
	return item
}

// settingState is true while a check item is changed programmatically:
// gtk_check_menu_item_set_active emits "activate" as if the user clicked.
var settingState bool

func applyState(item ptr, it *platform.MenuItem, check bool) {
	gtkWidgetSetSensitive(item, it.Enabled)
	gtkWidgetSetNoShowAll(item, !it.Visible)
	gtkWidgetSetVisible(item, it.Visible)
	if check {
		settingState = true
		gtkCheckMenuItemSetActive(item, it.Checked)
		settingState = false
	}
	if it.ToolTip != "" {
		gtkWidgetSetTooltipText(item, cs(it.ToolTip))
	}
}

var gdkKeyNames = map[string]string{
	"Enter": "Return", "Tab": "Tab", "Space": "space", "Backspace": "BackSpace", "Delete": "Delete",
	"Insert": "Insert", "Escape": "Escape", "Up": "Up", "Down": "Down", "Left": "Left", "Right": "Right",
	"Home": "Home", "End": "End", "PageUp": "Page_Up", "PageDown": "Page_Down", "+": "plus",
	"VolumeUp": "XF86AudioRaiseVolume", "VolumeDown": "XF86AudioLowerVolume", "VolumeMute": "XF86AudioMute",
	"MediaNextTrack": "XF86AudioNext", "MediaPreviousTrack": "XF86AudioPrev", "MediaStop": "XF86AudioStop",
	"MediaPlayPause": "XF86AudioPlay", "PrintScreen": "Print", "NumDec": "KP_Decimal", "NumAdd": "KP_Add",
	"NumSub": "KP_Subtract", "NumMult": "KP_Multiply", "NumDiv": "KP_Divide", "CapsLock": "Caps_Lock",
	"NumLock": "Num_Lock", "ScrollLock": "Scroll_Lock",
}

// gtkAccelerator converts an accelerator to a GDK keyval and modifier mask.
func gtkAccelerator(acc string) (uint32, uint32, bool) {
	a, err := accelerator.Parse(acc, "linux")
	if err != nil {
		return 0, 0, false
	}
	var mods uint32
	if a.Has(accelerator.Shift) {
		mods |= 1 << 0
	}
	if a.Has(accelerator.Ctrl) {
		mods |= 1 << 2
	}
	if a.Has(accelerator.Alt) {
		mods |= 1 << 3
	}
	if a.Has(accelerator.Super) {
		mods |= 1 << 26
	}
	var key uint32
	switch {
	case gdkKeyNames[a.Key] != "":
		key = gdkKeyvalFromName(cs(gdkKeyNames[a.Key]))
	case strings.HasPrefix(a.Key, "F") && len(a.Key) > 1:
		key = gdkKeyvalFromName(cs(a.Key))
	case strings.HasPrefix(a.Key, "Num") && len(a.Key) == 4:
		key = gdkKeyvalFromName(cs("KP_" + a.Key[3:]))
	default:
		r, _ := utf8.DecodeRuneInString(a.Key)
		key = gdkUnicodeToKeyval(uint32(r))
	}
	return key, mods, key != 0
}

func (b *Backend) SetApplicationMenu(m *platform.Menu) {
	b.appMenu = m
	for _, w := range b.windows {
		if !w.ownMenu && w.hasAppMenu() {
			w.installMenu(m)
		}
	}
}

// hasAppMenu tells whether the window shows the application menu in its
// menu bar: flyouts and panels show none.
func (w *window) hasAppMenu() bool { return w.opts.Flyout == nil && !w.opts.Panel }

func (w *window) SetMenu(m *platform.Menu) {
	w.ownMenu = m != nil
	if m == nil && w.hasAppMenu() {
		m = w.b.appMenu
	}
	w.installMenu(m)
}

// installMenu replaces the menu bar at the top of the window.
func (w *window) installMenu(m *platform.Menu) {
	if w.menubar != 0 {
		gtkWidgetDestroy(w.menubar)
		w.menubar = 0
		dropOwner(w.owner)
		// Accelerators live in the group; start with a fresh one.
		gtkWindowRemoveAccelGroup(w.win, w.accel)
		gObjectUnref(w.accel)
		w.accel = gtkAccelGroupNew()
		gtkWindowAddAccelGroup(w.win, w.accel)
	}
	if m == nil || len(m.Items) == 0 {
		return
	}
	w.owner = newOwner()
	w.menubar = gtkMenuBarNew()
	buildMenu(w.menubar, m, w.accel, w.owner, w.id)
	gtkBoxPackStart(w.box, w.menubar, false, false, 0)
	gtkBoxReorderChild(w.box, w.menubar, 0)
	gtkWidgetShowAll(w.menubar)
	// The bar shows as menuHides says, whatever shows the window.
	gtkWidgetSetNoShowAll(w.menubar, true)
	gtkWidgetSetVisible(w.menubar, !w.menuHides())
	data := ptr(w.id)
	connect(w.menubar, "can-activate-accel", cbCanActivateAccel, data)
	connect(w.menubar, "deactivate", cbMenuBarDeactivate, data)
}

func (w *window) SetAutoHideMenu(v bool) {
	w.autoHideMenu = v
	if w.menubar != 0 {
		gtkWidgetSetVisible(w.menubar, !w.menuHides())
	}
}

// menuHides reports whether the menu bar shows only while its menus are
// open: one that hides, and any in full screen, which leaves the page the
// whole screen.
func (w *window) menuHides() bool { return w.autoHideMenu || w.state&stateFullscreen != 0 }

// revealMenu shows a menu bar that hides and opens its first menu, as F10
// does in GTK, while the key event is current: the menu grabs its device.
// The bar hides again when its menus close.
func (w *window) revealMenu() {
	gtkWidgetShow(w.menubar)
	// The menu opens under its item: lay the bar out now.
	gtkContainerCheckResize(w.win)
	if item := firstMenu(w.menubar); item != 0 {
		gtkWidgetMnemonicActivate(item, false) // as the item's mnemonic
	} else {
		gtkWidgetHide(w.menubar) // nothing to open, so no deactivate to come
	}
}

// firstMenu returns the first menu of a menu bar that the user can open,
// or 0.
func firstMenu(bar ptr) ptr {
	list := gtkContainerGetChildren(bar)
	defer gListFree(list)
	for node := list; node != 0; node = field[ptr](node, 8) {
		if item := field[ptr](node, 0); gtkWidgetGetVisible(item) && gtkWidgetIsSensitive(item) {
			return item
		}
	}
	return 0
}

func (b *Backend) UpdateMenuItem(it *platform.MenuItem) {
	for _, n := range itemsByID[it.ID] {
		if it.Type != platform.MenuItemSeparator {
			gtkMenuItemSetLabel(n.item, cs(it.Label))
		}
		applyState(n.item, it, n.check)
	}
}

// PopupMenu shows a context menu and blocks until it is dismissed.
func (b *Backend) PopupMenu(m *platform.Menu, pw platform.Window, pos *platform.Point) {
	// GTK positions menus relative to a window: the given one, else the
	// active one.
	w, _ := pw.(*window)
	if w == nil || w.closed {
		w = nil
		for _, x := range b.windows {
			if !x.closed && gtkWindowIsActive(x.win) {
				w = x
				break
			}
		}
	}
	if w == nil {
		return // nowhere to show it
	}
	owner := newOwner()
	defer dropOwner(owner)
	menu := gtkMenuNew()
	buildMenu(menu, m, 0, owner, w.id)
	gtkWidgetShowAll(menu)
	event, anchor := w.popupTrigger(), w.contentWindow()
	const northWest = 1 // GDK_GRAVITY_NORTH_WEST
	corner := int32(northWest)
	if m.RightToLeft {
		// GTK_TEXT_DIR_RTL lays the menu out from the right (its items
		// take the default direction, which GTK takes from the locale),
		// and GDK_GRAVITY_NORTH_EAST opens it to the left of the point.
		gtkWidgetSetDirection(menu, 2)
		corner = 3
	}
	switch {
	case pos != nil && anchor != 0:
		gtkMenuPopupAtRect(menu, anchor, &gdkRectangle{X: int32(pos.X), Y: int32(pos.Y), Width: 1, Height: 1}, northWest, corner, event)
	case event != 0:
		gtkMenuPopupAtPointer(menu, event)
	case anchor != 0:
		// No mouse press to anchor to: at the pointer.
		var x, y int32
		pointer := gdkSeatGetPointer(gdkDisplayGetDefaultSeat(gdkDisplayGetDefault()))
		gdkWindowGetDevicePosition(anchor, pointer, &x, &y, nil)
		gtkMenuPopupAtRect(menu, anchor, &gdkRectangle{X: x, Y: y, Width: 1, Height: 1}, northWest, northWest, 0)
	}
	if !gtkWidgetGetVisible(menu) {
		// GTK could not show it: there is no dismissal to wait for.
		gtkWidgetDestroy(menu)
		return
	}
	loop := gMainLoopNew(0, false)
	connect(menu, "deactivate", cbMenuDeactivate, loop)
	b.popups = append(b.popups, menu)
	b.quitLoops = append(b.quitLoops, loop)
	gMainLoopRun(loop)
	b.quitLoops = b.quitLoops[:len(b.quitLoops)-1]
	b.popups = b.popups[:len(b.popups)-1]
	gMainLoopUnref(loop)
	gtkWidgetDestroy(menu)
}

// Tray icons use AppIndicator (StatusNotifierItem), which shows a menu on
// click; click events are not available.
type tray struct {
	b       *Backend
	id      int
	ind     ptr
	menu    ptr
	owner   int
	iconDir string
	icons   int
}

func (b *Backend) NewTray(h platform.TrayHandler) (platform.Tray, error) {
	if appIndicatorNew == nil {
		return nil, fmt.Errorf("mygo: tray icons need libayatana-appindicator3: %w", platform.ErrUnsupported)
	}
	b.nextTray++
	t := &tray{b: b, id: b.nextTray}
	name := fmt.Sprintf("mygo-%d-%d", os.Getpid(), t.id)
	t.ind = appIndicatorNew(cs(name), cs("application-x-executable"), 0) // APPLICATION_STATUS
	appIndicatorSetStatus(t.ind, 1)                                      // ACTIVE
	// AppIndicator only shows up once it has a menu.
	t.SetMenu(&platform.Menu{})
	b.trays[t.id] = t
	return t, nil
}

func (t *tray) SetImage(png []byte, template bool) error {
	if t.iconDir == "" {
		dir, err := os.MkdirTemp("", "mygo-tray-")
		if err != nil {
			return err
		}
		t.iconDir = dir
	}
	// A new file name makes the indicator host reload the icon.
	t.icons++
	name := fmt.Sprintf("icon-%d", t.icons)
	if err := os.WriteFile(filepath.Join(t.iconDir, name+".png"), png, 0o644); err != nil {
		return err
	}
	appIndicatorSetIconThemePath(t.ind, cs(t.iconDir))
	appIndicatorSetIconFull(t.ind, cs(name), cs(""))
	return nil
}

func (t *tray) SetTitle(title string) { appIndicatorSetLabel(t.ind, cs(title), cs(title)) }

func (t *tray) SetToolTip(tip string) {
	if appIndicatorSetTitle != nil {
		appIndicatorSetTitle(t.ind, cs(tip))
	}
}

func (t *tray) SetMenu(m *platform.Menu) {
	dropOwner(t.owner)
	t.owner = newOwner()
	menu := gtkMenuNew()
	if m != nil {
		buildMenu(menu, m, 0, t.owner, 0)
	}
	gtkWidgetShowAll(menu)
	appIndicatorSetMenu(t.ind, menu)
	t.menu = menu
}

func (t *tray) PopUpMenu(*platform.Menu) {}
func (t *tray) Bounds() platform.Rect    { return platform.Rect{} }

func (t *tray) Destroy() {
	if t.ind == 0 {
		return
	}
	appIndicatorSetStatus(t.ind, 0) // PASSIVE
	gObjectUnref(t.ind)
	t.ind = 0
	dropOwner(t.owner)
	delete(t.b.trays, t.id)
	if t.iconDir != "" {
		_ = os.RemoveAll(t.iconDir)
	}
}
