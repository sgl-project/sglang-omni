//go:build darwin

package darwin

import (
	"github.com/ebitengine/purego/objc"

	"github.com/egoist/mygo/internal/accelerator"
	"github.com/egoist/mygo/internal/platform"
)

// Native menu items are tracked per owner (the app menu, a tray, a popup)
// so they can be updated in place and released when the owner rebuilds.
type nativeItem struct {
	owner int
	item  id
}

const ownerAppMenu = 0

var (
	menuOwners    = 1
	itemsByID     = map[int][]nativeItem{}
	nativeRoleSel = map[string]string{
		"undo":               "undo:",
		"redo":               "redo:",
		"cut":                "cut:",
		"copy":               "copy:",
		"paste":              "paste:",
		"pasteAndMatchStyle": "pasteAsPlainText:",
		"delete":             "delete:",
		"selectAll":          "selectAll:",
		"minimize":           "performMiniaturize:",
		"zoom":               "performZoom:",
		"close":              "performClose:",
		"front":              "arrangeInFront:",
		"togglefullscreen":   "toggleFullScreen:",
		"about":              "orderFrontStandardAboutPanel:",
		"hide":               "hide:",
		"hideOthers":         "hideOtherApplications:",
		"unhide":             "unhideAllApplications:",
		"quit":               "terminate:",
		"startSpeaking":      "startSpeaking:",
		"stopSpeaking":       "stopSpeaking:",
	}
)

func newOwner() int {
	menuOwners++
	return menuOwners
}

func track(owner, itemID int, item id) {
	itemsByID[itemID] = append(itemsByID[itemID], nativeItem{owner, retain(item)})
}

func dropOwner(owner int) {
	for itemID, list := range itemsByID {
		kept := list[:0]
		for _, n := range list {
			if n.owner == owner {
				release(n.item)
			} else {
				kept = append(kept, n)
			}
		}
		if len(kept) == 0 {
			delete(itemsByID, itemID)
		} else {
			itemsByID[itemID] = kept
		}
	}
}

func registerMenuTarget() {
	classDef("MyGoMenuTarget", "NSObject", nil, []objc.MethodDef{
		method("mygoMenuItemClicked:", func(self id, _ objc.SEL, item id) {
			theBackend.h.MenuItemClicked(sendInt(item, "tag"))
		}),
	})
}

func (b *Backend) SetApplicationMenu(m *platform.Menu) {
	withPool(func() {
		dropOwner(ownerAppMenu)
		if m == nil {
			send(b.app, "setMainMenu:", 0)
			return
		}
		send(b.app, "setMainMenu:", uintptr(b.buildMenu(m, "", ownerAppMenu)))
	})
}

// buildMenu creates an autoreleased NSMenu.
func (b *Backend) buildMenu(m *platform.Menu, title string, owner int) id {
	menu := autorelease(send(send(class("NSMenu"), "alloc"), "initWithTitle:", uintptr(nsString(title))))
	send(menu, "setAutoenablesItems:", 0)
	for _, it := range m.Items {
		if it.Type == platform.MenuItemSeparator {
			item := send(class("NSMenuItem"), "separatorItem")
			send(item, "setHidden:", boolArg(!it.Visible))
			send(menu, "addItem:", uintptr(item))
			track(owner, it.ID, item)
			continue
		}
		item := b.buildItem(it, owner)
		send(menu, "addItem:", uintptr(item))
		if it.Role == "zoomIn" {
			// Also accept Cmd+= (no shift) like other macOS apps.
			alt := b.buildItem(&platform.MenuItem{ID: it.ID, Label: it.Label, Role: it.Role, Accelerator: "CmdOrCtrl+=", Enabled: it.Enabled}, owner)
			send(alt, "setHidden:", 1)
			if respondsTo(alt, "setAllowsKeyEquivalentWhenHidden:") {
				send(alt, "setAllowsKeyEquivalentWhenHidden:", 1)
			}
			send(menu, "addItem:", uintptr(alt))
		}
	}
	return menu
}

func (b *Backend) buildItem(it *platform.MenuItem, owner int) id {
	action := sel("mygoMenuItemClicked:")
	target := b.menuTarget
	if s, ok := nativeRoleSel[it.Role]; ok {
		action, target = sel(s), 0
	}
	if it.Submenu != nil {
		action, target = 0, 0
	}
	key, mods := keyEquivalent(it.Accelerator)
	item := autorelease(send(send(class("NSMenuItem"), "alloc"), "initWithTitle:action:keyEquivalent:",
		uintptr(nsString(it.Label)), uintptr(action), uintptr(nsString(key))))
	if key != "" {
		send(item, "setKeyEquivalentModifierMask:", uintptr(mods))
	}
	send(item, "setTarget:", uintptr(target))
	send(item, "setTag:", uintptr(it.ID))
	applyItemState(item, it)
	if it.Submenu != nil {
		sub := b.buildMenu(it.Submenu, it.Label, owner)
		send(item, "setSubmenu:", uintptr(sub))
		switch it.Role {
		case "services":
			send(b.app, "setServicesMenu:", uintptr(sub))
		case "window", "windowMenu":
			send(b.app, "setWindowsMenu:", uintptr(sub))
		case "help":
			send(b.app, "setHelpMenu:", uintptr(sub))
		}
	}
	track(owner, it.ID, item)
	return item
}

func applyItemState(item id, it *platform.MenuItem) {
	send(item, "setEnabled:", boolArg(it.Enabled))
	send(item, "setHidden:", boolArg(!it.Visible))
	state := uintptr(0)
	if it.Checked && (it.Type == platform.MenuItemCheckbox || it.Type == platform.MenuItemRadio) {
		state = 1
	}
	send(item, "setState:", state)
	if it.ToolTip != "" {
		send(item, "setToolTip:", uintptr(nsString(it.ToolTip)))
	}
}

func (b *Backend) UpdateMenuItem(it *platform.MenuItem) {
	withPool(func() {
		key, mods := keyEquivalent(it.Accelerator)
		for _, n := range itemsByID[it.ID] {
			// The hidden Cmd+= twin of "Zoom In" only follows the enabled state.
			if it.Role == "zoomIn" && sendBool(n.item, "isHidden") && respondsTo(n.item, "allowsKeyEquivalentWhenHidden") &&
				sendBool(n.item, "allowsKeyEquivalentWhenHidden") {
				send(n.item, "setEnabled:", boolArg(it.Enabled))
				continue
			}
			if it.Type != platform.MenuItemSeparator {
				send(n.item, "setTitle:", uintptr(nsString(it.Label)))
				if sub := send(n.item, "submenu"); sub != 0 {
					send(sub, "setTitle:", uintptr(nsString(it.Label)))
				}
				send(n.item, "setKeyEquivalent:", uintptr(nsString(key)))
				send(n.item, "setKeyEquivalentModifierMask:", uintptr(mods))
			}
			applyItemState(n.item, it)
		}
	})
}

func (b *Backend) PopupMenu(m *platform.Menu, pw platform.Window, pos *platform.Point) {
	withPool(func() {
		owner := newOwner()
		defer dropOwner(owner)
		menu := b.buildMenu(m, "", owner)
		var view id
		var loc NSPoint
		if w, ok := pw.(*window); ok && w != nil && !w.closed {
			view = w.web
			if w.surface != nil {
				view = w.surface.view
			}
			if pos != nil {
				loc = NSPoint{float64(pos.X), float64(pos.Y)}
				if !sendBool(view, "isFlipped") {
					loc.Y = msgRect(view, sel("bounds")).Size.Height - loc.Y
				}
			} else {
				p := msgPoint(w.win, sel("mouseLocationOutsideOfEventStream"))
				loc = msgPointFromView(view, sel("convertPoint:fromView:"), p, 0)
			}
		} else {
			loc = msgPoint(class("NSEvent"), sel("mouseLocation"))
		}
		if m.RightToLeft {
			// NSUserInterfaceLayoutDirectionRightToLeft lays the items out
			// from the right and opens submenus to the left. A menu popped
			// up in a view takes the view's direction, which the surface
			// has only while the menu shows.
			send(menu, "setUserInterfaceLayoutDirection:", 1)
			if view != 0 {
				was := sendInt(view, "userInterfaceLayoutDirection")
				send(view, "setUserInterfaceLayoutDirection:", 1)
				defer send(view, "setUserInterfaceLayoutDirection:", uintptr(was))
			}
		}
		msgPopUpMenu(menu, sel("popUpMenuPositioningItem:atLocation:inView:"), 0, loc, view)
	})
}

// keyEquivalent converts an accelerator to an NSMenuItem key equivalent and
// modifier mask.
func keyEquivalent(acc string) (string, uint) {
	if acc == "" {
		return "", 0
	}
	a, err := accelerator.Parse(acc, "darwin")
	if err != nil {
		return "", 0
	}
	var mods uint
	if a.Has(accelerator.Super) {
		mods |= 1 << 20
	}
	if a.Has(accelerator.Ctrl) {
		mods |= 1 << 18
	}
	if a.Has(accelerator.Alt) {
		mods |= 1 << 19
	}
	if a.Has(accelerator.Shift) {
		mods |= 1 << 17
	}
	if len(a.Key) == 1 {
		return a.Key, mods
	}
	named := map[string]rune{
		"Enter": '\r', "Tab": '\t', "Space": ' ', "Backspace": 0x08, "Delete": 0xF728, "Escape": 0x1b,
		"Up": 0xF700, "Down": 0xF701, "Left": 0xF702, "Right": 0xF703, "Home": 0xF729, "End": 0xF72B,
		"PageUp": 0xF72C, "PageDown": 0xF72D, "Insert": 0xF727, "PrintScreen": 0xF72E,
	}
	if r, ok := named[a.Key]; ok {
		return string(r), mods
	}
	var n int
	if len(a.Key) > 1 && a.Key[0] == 'F' {
		for _, c := range a.Key[1:] {
			n = n*10 + int(c-'0')
		}
		return string(rune(0xF704 + n - 1)), mods
	}
	return "", 0
}

// Tray icons (NSStatusItem).

type tray struct {
	b      *Backend
	h      platform.TrayHandler
	item   id
	button id
	owner  int
}

func registerTrayTarget() {
	classDef("MyGoTrayTarget", "NSObject", nil, []objc.MethodDef{
		method("mygoTrayClicked:", func(self id, _ objc.SEL, sender id) {
			t := theBackend.trays[sender]
			if t == nil {
				return
			}
			ev := send(theBackend.app, "currentEvent")
			if ev != 0 && sendInt(ev, "type") == 4 { // NSEventTypeRightMouseUp
				t.h.RightClicked()
			} else {
				t.h.Clicked()
			}
		}),
	})
}

var trayTarget id

func (b *Backend) NewTray(h platform.TrayHandler) (platform.Tray, error) {
	if trayTarget == 0 {
		trayTarget = alloc("MyGoTrayTarget")
	}
	t := &tray{b: b, h: h, owner: newOwner()}
	withPool(func() {
		bar := send(class("NSStatusBar"), "systemStatusBar")
		t.item = retain(msgFloatID(bar, sel("statusItemWithLength:"), -1)) // NSVariableStatusItemLength
		t.button = send(t.item, "button")
		send(t.button, "setTarget:", uintptr(trayTarget))
		send(t.button, "setAction:", uintptr(sel("mygoTrayClicked:")))
		send(t.button, "sendActionOn:", 1<<2|1<<4) // left and right mouse up
	})
	b.trays[t.button] = t
	return t, nil
}

func (t *tray) SetImage(png []byte, template bool) error {
	var err error
	withPool(func() {
		img := autorelease(send(send(class("NSImage"), "alloc"), "initWithData:", uintptr(nsData(png))))
		if img == 0 {
			err = errInvalidImage
			return
		}
		// Menu bar icons are 18 points tall.
		size := msgSize(img, sel("size"))
		if size.Height > 0 {
			msgSetSize(img, sel("setSize:"), NSSize{size.Width * 18 / size.Height, 18})
		}
		send(img, "setTemplate:", boolArg(template))
		send(t.button, "setImage:", uintptr(img))
	})
	return err
}

func (t *tray) SetTitle(title string) {
	withPool(func() { send(t.button, "setTitle:", uintptr(nsString(title))) })
}

func (t *tray) SetToolTip(tip string) {
	withPool(func() { send(t.button, "setToolTip:", uintptr(nsString(tip))) })
}

func (t *tray) SetMenu(m *platform.Menu) {
	withPool(func() {
		dropOwner(t.owner)
		if m == nil {
			send(t.item, "setMenu:", 0)
			return
		}
		send(t.item, "setMenu:", uintptr(t.b.buildMenu(m, "", t.owner)))
	})
}

func (t *tray) PopUpMenu(m *platform.Menu) {
	if m == nil || t.item == 0 {
		return
	}
	withPool(func() {
		owner := newOwner()
		defer dropOwner(owner)
		menu := t.b.buildMenu(m, "", owner)
		// Let the status item place and track its menu, including the
		// button's highlight. Keep its previous menu alive while swapped.
		previous := retain(send(t.item, "menu"))
		defer release(previous)
		send(t.item, "setMenu:", uintptr(menu))
		send(t.button, "performClick:", 0)
		// Menu actions run in a nested event loop: they may destroy the
		// tray or install another menu, which we must leave intact.
		if t.item != 0 && send(t.item, "menu") == menu {
			send(t.item, "setMenu:", uintptr(previous))
		}
	})
}

func (t *tray) Bounds() platform.Rect {
	win := send(t.button, "window")
	if win == 0 {
		return platform.Rect{}
	}
	return rectFromMac(msgRect(win, sel("frame")))
}

func (t *tray) Destroy() {
	if t.item == 0 {
		return
	}
	delete(t.b.trays, t.button)
	dropOwner(t.owner)
	send(send(class("NSStatusBar"), "systemStatusBar"), "removeStatusItem:", uintptr(t.item))
	release(t.item)
	t.item = 0
}
