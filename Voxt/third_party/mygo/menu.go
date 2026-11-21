package mygo

import (
	"runtime"
	"sync"
	"weak"

	"github.com/egoist/mygo/internal/platform"
)

// MenuItemType is the kind of a menu item.
type MenuItemType string

// Menu item kinds. The type is inferred when empty: items with a Submenu are
// submenus, everything else is a normal item.
const (
	MenuItemNormal    MenuItemType = "normal"
	MenuItemSeparator MenuItemType = "separator"
	MenuItemSubmenu   MenuItemType = "submenu"
	MenuItemCheckbox  MenuItemType = "checkbox"
	MenuItemRadio     MenuItemType = "radio"
)

// MenuRole gives a menu item a predefined behavior, label and accelerator.
type MenuRole string

// Menu roles. Roles marked (macOS) are ignored elsewhere.
const (
	RoleUndo               MenuRole = "undo"
	RoleRedo               MenuRole = "redo"
	RoleCut                MenuRole = "cut"
	RoleCopy               MenuRole = "copy"
	RolePaste              MenuRole = "paste"
	RolePasteAndMatchStyle MenuRole = "pasteAndMatchStyle"
	RoleDelete             MenuRole = "delete"
	RoleSelectAll          MenuRole = "selectAll"
	RoleReload             MenuRole = "reload"
	RoleForceReload        MenuRole = "forceReload"
	RoleToggleDevTools     MenuRole = "toggleDevTools"
	RoleResetZoom          MenuRole = "resetZoom"
	RoleZoomIn             MenuRole = "zoomIn"
	RoleZoomOut            MenuRole = "zoomOut"
	RoleToggleFullScreen   MenuRole = "togglefullscreen"
	RoleMinimize           MenuRole = "minimize"
	RoleZoom               MenuRole = "zoom"
	RoleClose              MenuRole = "close"
	// Documents (NewDocuments): they act on the focused window's
	// document, and New and Open on the app's first kind of documents.
	RoleNew           MenuRole = "new"
	RoleOpen          MenuRole = "open"
	RoleOpenRecent    MenuRole = "openRecent" // submenu of the recent documents
	RoleSave          MenuRole = "save"
	RoleSaveAs        MenuRole = "saveAs"
	RoleDuplicate     MenuRole = "duplicate"
	RoleRename        MenuRole = "rename"
	RoleMoveTo        MenuRole = "moveTo"
	RoleRevert        MenuRole = "revert"
	RoleQuit          MenuRole = "quit"
	RoleAbout         MenuRole = "about"         // (macOS)
	RoleHide          MenuRole = "hide"          // (macOS)
	RoleHideOthers    MenuRole = "hideOthers"    // (macOS)
	RoleUnhide        MenuRole = "unhide"        // (macOS)
	RoleFront         MenuRole = "front"         // (macOS)
	RoleServices      MenuRole = "services"      // (macOS) submenu
	RoleStartSpeaking MenuRole = "startSpeaking" // (macOS)
	RoleStopSpeaking  MenuRole = "stopSpeaking"  // (macOS)
	RoleWindow        MenuRole = "window"        // submenu listing open windows (macOS)
	RoleHelp          MenuRole = "help"          // submenu with the search field (macOS)
	RoleAppMenu       MenuRole = "appMenu"       // default application menu
	RoleFileMenu      MenuRole = "fileMenu"      // default File menu
	RoleEditMenu      MenuRole = "editMenu"      // default Edit menu
	RoleViewMenu      MenuRole = "viewMenu"      // default View menu
	RoleWindowMenu    MenuRole = "windowMenu"    // default Window menu
	// The Format menu's roles format the selection of a rich text editor
	// of native UI (ui/richtext), or of an editable web page.
	RoleBold          MenuRole = "bold"
	RoleItalic        MenuRole = "italic"
	RoleUnderline     MenuRole = "underline"
	RoleStrikethrough MenuRole = "strikethrough"
	RoleAlignLeft     MenuRole = "alignLeft"
	RoleAlignCenter   MenuRole = "alignCenter"
	RoleAlignRight    MenuRole = "alignRight"
	RoleFormatMenu    MenuRole = "formatMenu" // default Format menu
)

// MenuItem is an entry of a Menu. Fill in the exported fields to build a
// menu template; after the item has been added to a menu, change its state
// with the Set methods so native menus update.
type MenuItem struct {
	// ID identifies the item for Menu.ItemByID.
	ID    string
	Label string
	Type  MenuItemType
	// Role gives the item a predefined behavior; Label and Accelerator
	// default to the role's.
	Role MenuRole
	// Accelerator is a keyboard shortcut such as "CmdOrCtrl+Shift+N".
	Accelerator string
	Disabled    bool
	Hidden      bool
	// Checked is the state of checkbox and radio items. It is toggled
	// before Click is called.
	Checked bool
	ToolTip string
	Submenu []*MenuItem
	// Click is called on the main thread when the item is chosen. On Linux
	// and Windows, win is the window whose menu bar or context menu was
	// chosen. Other menus use the focused window, which may be nil.
	Click func(item *MenuItem, win *Window)

	uid    int
	parent *Menu
	sub    *Menu
	// roleLabel is the label the item took from its role, which says it
	// in the language the app shows for as long as the app gives the item
	// no label of its own (relabelMenus).
	roleLabel string
}

// Menu is a list of menu items used as application menu, window menu,
// context menu or tray menu.
type Menu struct {
	items []*MenuItem
}

// Separator returns a separator item.
func Separator() *MenuItem { return &MenuItem{Type: MenuItemSeparator} }

// menuMu guards the state of all menus and items.
var menuMu sync.Mutex

var menuItems struct {
	next int
	byID map[int]weak.Pointer[MenuItem]
}

// NewMenu builds a menu from a template:
//
//	menu := mygo.NewMenu([]*mygo.MenuItem{
//		{Role: mygo.RoleAppMenu},
//		{Label: "File", Submenu: []*mygo.MenuItem{
//			{Label: "New Window", Accelerator: "CmdOrCtrl+N", Click: newWindow},
//			mygo.Separator(),
//			{Role: mygo.RoleClose},
//		}},
//		{Role: mygo.RoleEditMenu},
//	})
func NewMenu(items []*MenuItem) *Menu {
	m := &Menu{}
	menuMu.Lock()
	for _, it := range items {
		m.items = append(m.items, prepareItem(it, m))
	}
	menuMu.Unlock()
	return m
}

// Items returns the items of the menu.
func (m *Menu) Items() []*MenuItem {
	menuMu.Lock()
	defer menuMu.Unlock()
	return append([]*MenuItem(nil), m.items...)
}

// Append adds items at the end of the menu.
func (m *Menu) Append(items ...*MenuItem) {
	menuMu.Lock()
	for _, it := range items {
		m.items = append(m.items, prepareItem(it, m))
	}
	menuMu.Unlock()
	refreshMenus()
}

// Insert adds item at position pos.
func (m *Menu) Insert(pos int, item *MenuItem) {
	menuMu.Lock()
	pos = min(max(pos, 0), len(m.items))
	m.items = append(m.items[:pos], append([]*MenuItem{prepareItem(item, m)}, m.items[pos:]...)...)
	menuMu.Unlock()
	refreshMenus()
}

// ItemByID finds an item by ID in the menu and its submenus.
func (m *Menu) ItemByID(id string) *MenuItem {
	menuMu.Lock()
	defer menuMu.Unlock()
	return m.itemByID(id)
}

func (m *Menu) itemByID(id string) *MenuItem {
	for _, it := range m.items {
		if it.ID == id {
			return it
		}
		if it.sub != nil {
			if found := it.sub.itemByID(id); found != nil {
				return found
			}
		}
	}
	return nil
}

// Popup shows the menu as a context menu at the mouse position over win.
// It returns once the menu is closed.
func (m *Menu) Popup(win *Window) {
	needsApp("Menu.Popup")
	m.popup(win, nil, false)
}

// PopupAt shows the menu as a context menu at a position relative to the
// top-left corner of win's page area.
func (m *Menu) PopupAt(win *Window, x, y int) {
	needsApp("Menu.PopupAt")
	m.popup(win, &platform.Point{X: x, Y: y}, false)
}

// popup shows the menu at pos, laid out right to left with rtl, as those of
// native UI laid out so.
func (m *Menu) popup(win *Window, pos *platform.Point, rtl bool) {
	snap := m.snapshot()
	snap.RightToLeft = rtl
	onMain(func() {
		var n platform.Window
		if win != nil {
			n = win.native
		}
		backend().PopupMenu(snap, n, pos)
	})
	runtime.KeepAlive(m)
}

// SetEnabled enables or disables the item.
func (it *MenuItem) SetEnabled(v bool) { it.update(func() { it.Disabled = !v }) }

// IsEnabled reports whether the item is enabled.
func (it *MenuItem) IsEnabled() bool { return !it.get(func() bool { return it.Disabled }) }

// SetVisible shows or hides the item.
func (it *MenuItem) SetVisible(v bool) { it.update(func() { it.Hidden = !v }) }

// IsVisible reports whether the item is visible.
func (it *MenuItem) IsVisible() bool { return !it.get(func() bool { return it.Hidden }) }

// SetChecked checks or unchecks a checkbox or radio item.
func (it *MenuItem) SetChecked(v bool) {
	it.update(func() {
		if v && it.Type == MenuItemRadio {
			it.uncheckRadioSiblings()
		}
		it.Checked = v
	})
}

// IsChecked reports whether a checkbox or radio item is checked.
func (it *MenuItem) IsChecked() bool { return it.get(func() bool { return it.Checked }) }

// SetLabel changes the label of the item.
func (it *MenuItem) SetLabel(label string) { it.update(func() { it.Label = label }) }

// SetAccelerator changes the keyboard shortcut of the item.
func (it *MenuItem) SetAccelerator(acc string) { it.update(func() { it.Accelerator = acc }) }

func (it *MenuItem) get(fn func() bool) bool {
	menuMu.Lock()
	defer menuMu.Unlock()
	return fn()
}

func (it *MenuItem) update(fn func()) {
	menuMu.Lock()
	fn()
	var snaps []*platform.MenuItem
	if it.uid != 0 {
		snaps = append(snaps, it.snapshotLocked())
		if it.Type == MenuItemRadio && it.parent != nil {
			for _, s := range it.radioGroup() {
				if s != it {
					snaps = append(snaps, s.snapshotLocked())
				}
			}
		}
	}
	menuMu.Unlock()
	if len(snaps) > 0 {
		postMain(func() {
			for _, s := range snaps {
				backend().UpdateMenuItem(s)
			}
		})
	}
}

// radioGroup returns the run of adjacent radio items containing it.
func (it *MenuItem) radioGroup() []*MenuItem {
	items := it.parent.items
	idx := -1
	for i, x := range items {
		if x == it {
			idx = i
			break
		}
	}
	if idx < 0 {
		return []*MenuItem{it}
	}
	start, end := idx, idx
	for start > 0 && items[start-1].Type == MenuItemRadio {
		start--
	}
	for end < len(items)-1 && items[end+1].Type == MenuItemRadio {
		end++
	}
	return items[start : end+1]
}

func (it *MenuItem) uncheckRadioSiblings() {
	if it.parent == nil {
		return
	}
	for _, s := range it.radioGroup() {
		s.Checked = false
	}
}

// prepareItem registers an item and expands its role. menuMu must be held.
func prepareItem(it *MenuItem, parent *Menu) *MenuItem {
	it.parent = parent
	if it.uid == 0 {
		if menuItems.byID == nil {
			menuItems.byID = map[int]weak.Pointer[MenuItem]{}
		}
		menuItems.next++
		it.uid = menuItems.next
		menuItems.byID[it.uid] = weak.Make(it)
		if len(menuItems.byID) > 256 && len(menuItems.byID)&255 == 0 {
			for id, wp := range menuItems.byID {
				if wp.Value() == nil {
					delete(menuItems.byID, id)
				}
			}
		}
	}
	if it.Role != "" {
		label, accel := roleDefaults(it.Role)
		if it.Label == "" {
			it.Label, it.roleLabel = label, label
		}
		if it.Accelerator == "" {
			it.Accelerator = accel
		}
		if it.Submenu == nil {
			it.Submenu = roleSubmenu(it.Role)
		}
		if runtime.GOOS != "darwin" && macOnlyRoles[it.Role] {
			it.Hidden = true
		}
	}
	if it.Type == "" {
		it.Type = MenuItemNormal
		if it.Submenu != nil {
			it.Type = MenuItemSubmenu
		}
	}
	if it.Submenu != nil {
		sub := &Menu{}
		for _, child := range it.Submenu {
			sub.items = append(sub.items, prepareItem(child, sub))
		}
		it.sub = sub
	}
	return it
}

func lookupMenuItem(uid int) *MenuItem {
	menuMu.Lock()
	defer menuMu.Unlock()
	return menuItems.byID[uid].Value()
}

// refreshMenus re-installs menus in use after their structure changed.
func refreshMenus() {
	if !App.IsReady() {
		return
	}
	postMain(func() {
		if m := App.Menu(); m != nil {
			backend().SetApplicationMenu(m.snapshot())
		}
		for _, w := range Windows() {
			if w.menu != nil && w.native != nil {
				w.native.SetMenu(w.menu.snapshot())
			}
		}
	})
}

func (m *Menu) snapshot() *platform.Menu {
	if m == nil {
		return nil
	}
	menuMu.Lock()
	defer menuMu.Unlock()
	return m.snapshotLocked()
}

func (m *Menu) snapshotLocked() *platform.Menu {
	pm := &platform.Menu{Items: make([]*platform.MenuItem, 0, len(m.items))}
	for _, it := range m.items {
		pm.Items = append(pm.Items, it.snapshotLocked())
	}
	return pm
}

func (it *MenuItem) snapshotLocked() *platform.MenuItem {
	p := &platform.MenuItem{
		ID:          it.uid,
		Label:       it.Label,
		Role:        string(it.Role),
		Accelerator: it.Accelerator,
		Enabled:     !it.Disabled,
		Visible:     !it.Hidden,
		Checked:     it.Checked,
		ToolTip:     it.ToolTip,
	}
	switch it.Type {
	case MenuItemSeparator:
		p.Type = platform.MenuItemSeparator
	case MenuItemSubmenu:
		p.Type = platform.MenuItemSubmenu
	case MenuItemCheckbox:
		p.Type = platform.MenuItemCheckbox
	case MenuItemRadio:
		p.Type = platform.MenuItemRadio
	}
	if it.sub != nil {
		p.Submenu = it.sub.snapshotLocked()
	}
	return p
}

// menuItemClicked runs on the main thread when the backend reports a click.
func menuItemClicked(uid int, win *Window) {
	it := lookupMenuItem(uid)
	if it == nil {
		return
	}
	switch it.Type {
	case MenuItemCheckbox:
		it.SetChecked(!it.IsChecked())
	case MenuItemRadio:
		it.SetChecked(true)
	}
	performRole(it.Role, win)
	if it.Click != nil {
		it.Click(it, win)
	}
}

var zoomSteps = []float64{0.25, 0.33, 0.5, 0.67, 0.75, 0.8, 0.9, 1, 1.1, 1.25, 1.5, 1.75, 2, 2.5, 3, 4, 5}

func nextZoom(cur float64, dir int) float64 {
	if dir > 0 {
		for _, s := range zoomSteps {
			if s > cur+0.001 {
				return s
			}
		}
		return zoomSteps[len(zoomSteps)-1]
	}
	for i := len(zoomSteps) - 1; i >= 0; i-- {
		if zoomSteps[i] < cur-0.001 {
			return zoomSteps[i]
		}
	}
	return zoomSteps[0]
}

// performRole implements roles that backends report back to Go.
func performRole(role MenuRole, win *Window) {
	switch role {
	case RoleQuit:
		App.Quit()
		return
	case "":
		return
	case RoleNew, RoleOpen, RoleSave, RoleSaveAs, RoleDuplicate, RoleRename, RoleMoveTo, RoleRevert:
		if perform, _ := linkedDocuments(); perform != nil {
			perform(role, win)
		}
		return
	}
	if win == nil || win.native == nil {
		return
	}
	n := win.native
	// A window showing native UI has no page to reload or zoom; it has
	// its own inspector.
	switch role {
	case RoleToggleDevTools:
		if c := win.conn; c != nil {
			if c.ToggleDevTools != nil {
				c.ToggleDevTools()
			}
			return
		}
	case RoleReload, RoleForceReload, RoleResetZoom, RoleZoomIn, RoleZoomOut:
		if win.content != nil {
			return
		}
	}
	switch role {
	case RoleReload:
		n.Reload(false)
	case RoleForceReload:
		n.Reload(true)
	case RoleToggleDevTools:
		if n.IsDevToolsOpened() {
			n.CloseDevTools()
		} else {
			n.OpenDevTools()
		}
	case RoleResetZoom:
		n.SetZoom(1)
	case RoleZoomIn:
		n.SetZoom(nextZoom(n.Zoom(), 1))
	case RoleZoomOut:
		n.SetZoom(nextZoom(n.Zoom(), -1))
	case RoleToggleFullScreen:
		n.SetFullScreen(!n.IsFullScreen())
	case RoleMinimize:
		n.Minimize()
	case RoleZoom:
		if n.IsMaximized() {
			n.Unmaximize()
		} else {
			n.Maximize()
		}
	case RoleClose:
		win.close()
	case RoleUndo, RoleRedo, RoleCut, RoleCopy, RolePaste, RoleDelete, RoleSelectAll, RolePasteAndMatchStyle:
		cmd := map[MenuRole]string{
			RoleUndo: "undo", RoleRedo: "redo", RoleCut: "cut", RoleCopy: "copy", RolePaste: "paste",
			RoleDelete: "delete", RoleSelectAll: "selectAll", RolePasteAndMatchStyle: "pasteAndMatchStyle",
		}[role]
		if win.content != nil {
			win.contentCommand(cmd)
			return
		}
		if role == RolePasteAndMatchStyle {
			cmd = "paste"
		}
		n.Eval("document.execCommand(" + `"` + cmd + `"` + ")")
	case RoleBold, RoleItalic, RoleUnderline, RoleStrikethrough, RoleAlignLeft, RoleAlignCenter, RoleAlignRight:
		if win.content != nil {
			win.contentCommand(string(role))
			return
		}
		cmd := map[MenuRole]string{
			RoleBold: "bold", RoleItalic: "italic", RoleUnderline: "underline", RoleStrikethrough: "strikeThrough",
			RoleAlignLeft: "justifyLeft", RoleAlignCenter: "justifyCenter", RoleAlignRight: "justifyRight",
		}[role]
		n.Eval("document.execCommand(" + `"` + cmd + `"` + ")")
	}
}

var macOnlyRoles = map[MenuRole]bool{
	RoleAbout: true, RoleHide: true, RoleHideOthers: true, RoleUnhide: true, RoleFront: true,
	RoleServices: true, RoleStartSpeaking: true, RoleStopSpeaking: true, RoleAppMenu: true,
	RolePasteAndMatchStyle: true, RoleWindow: true, RoleHelp: true,
}

func roleDefaults(r MenuRole) (label, accel string) {
	mac := runtime.GOOS == "darwin"
	name := App.Name()
	tx := menuMessages()
	switch r {
	case RoleUndo:
		return tx.MenuUndo(), "CmdOrCtrl+Z"
	case RoleRedo:
		if runtime.GOOS == "windows" {
			return tx.MenuRedo(), "Ctrl+Y"
		}
		return tx.MenuRedo(), "Shift+CmdOrCtrl+Z"
	case RoleCut:
		return tx.MenuCut(), "CmdOrCtrl+X"
	case RoleCopy:
		return tx.MenuCopy(), "CmdOrCtrl+C"
	case RolePaste:
		return tx.MenuPaste(), "CmdOrCtrl+V"
	case RolePasteAndMatchStyle:
		return tx.MenuPasteAndMatchStyle(), "Shift+Alt+CmdOrCtrl+V"
	case RoleDelete:
		return tx.MenuDelete(), ""
	case RoleSelectAll:
		return tx.MenuSelectAll(), "CmdOrCtrl+A"
	case RoleReload:
		return tx.MenuReload(), "CmdOrCtrl+R"
	case RoleForceReload:
		return tx.MenuForceReload(), "Shift+CmdOrCtrl+R"
	case RoleToggleDevTools:
		if mac {
			return tx.MenuToggleDevTools(), "Alt+Cmd+I"
		}
		return tx.MenuToggleDevTools(), "Ctrl+Shift+I"
	case RoleResetZoom:
		return tx.MenuActualSize(), "CmdOrCtrl+0"
	case RoleZoomIn:
		return tx.MenuZoomIn(), "CmdOrCtrl+Plus"
	case RoleZoomOut:
		return tx.MenuZoomOut(), "CmdOrCtrl+-"
	case RoleToggleFullScreen:
		if mac {
			return tx.MenuToggleFullScreen(), "Ctrl+Cmd+F"
		}
		return tx.MenuToggleFullScreen(), "F11"
	case RoleMinimize:
		return tx.MenuMinimize(), "CmdOrCtrl+M"
	case RoleZoom:
		return tx.MenuZoom(), ""
	case RoleClose:
		if hasDocuments() {
			return tx.MenuClose(), "CmdOrCtrl+W"
		}
		return tx.MenuCloseWindow(), "CmdOrCtrl+W"
	case RoleNew:
		return tx.MenuNew(), "CmdOrCtrl+N"
	case RoleOpen:
		return tx.MenuOpen(), "CmdOrCtrl+O"
	case RoleOpenRecent:
		return tx.MenuOpenRecent(), ""
	case RoleSave:
		return tx.MenuSave(), "CmdOrCtrl+S"
	case RoleSaveAs:
		if mac {
			return tx.MenuSaveAs(), "Alt+Shift+Cmd+S"
		}
		return tx.MenuSaveAs(), "Ctrl+Shift+S"
	case RoleDuplicate:
		if mac {
			return tx.MenuDuplicate(), "Shift+Cmd+S"
		}
		return tx.MenuDuplicate(), ""
	case RoleRename:
		return tx.MenuRename(), ""
	case RoleMoveTo:
		return tx.MenuMoveTo(), ""
	case RoleRevert:
		if mac {
			return tx.MenuRevertToSaved(), ""
		}
		return tx.MenuRevert(), ""
	case RoleQuit:
		if mac {
			return tx.MenuQuitApp(name), "Cmd+Q"
		}
		if runtime.GOOS == "windows" {
			return tx.MenuExit(), ""
		}
		return tx.MenuQuit(), "Ctrl+Q"
	case RoleAbout:
		return tx.MenuAbout(name), ""
	case RoleHide:
		return tx.MenuHide(name), "Cmd+H"
	case RoleHideOthers:
		return tx.MenuHideOthers(), "Alt+Cmd+H"
	case RoleUnhide:
		return tx.MenuShowAll(), ""
	case RoleFront:
		return tx.MenuBringAllToFront(), ""
	case RoleServices:
		return tx.MenuServices(), ""
	case RoleStartSpeaking:
		return tx.MenuStartSpeaking(), ""
	case RoleStopSpeaking:
		return tx.MenuStopSpeaking(), ""
	case RoleWindow, RoleWindowMenu:
		return tx.MenuWindow(), ""
	case RoleHelp:
		return tx.MenuHelp(), ""
	case RoleAppMenu:
		return name, ""
	case RoleFileMenu:
		return tx.MenuFile(), ""
	case RoleEditMenu:
		return tx.MenuEdit(), ""
	case RoleViewMenu:
		return tx.MenuView(), ""
	case RoleFormatMenu:
		return tx.MenuFormat(), ""
	case RoleBold:
		return tx.MenuBold(), "CmdOrCtrl+B"
	case RoleItalic:
		return tx.MenuItalic(), "CmdOrCtrl+I"
	case RoleUnderline:
		return tx.MenuUnderline(), "CmdOrCtrl+U"
	case RoleStrikethrough:
		return tx.MenuStrikethrough(), "Shift+CmdOrCtrl+X"
	case RoleAlignLeft:
		if mac {
			return tx.MenuAlignLeft(), "Cmd+{"
		}
		return tx.MenuAlignLeft(), "Ctrl+L"
	case RoleAlignCenter:
		if mac {
			return tx.MenuCenter(), "Cmd+|"
		}
		return tx.MenuCenter(), "Ctrl+E"
	case RoleAlignRight:
		if mac {
			return tx.MenuAlignRight(), "Cmd+}"
		}
		return tx.MenuAlignRight(), "Ctrl+R"
	}
	return string(r), ""
}

// roleSubmenu returns the default items of submenu roles.
func roleSubmenu(r MenuRole) []*MenuItem {
	mac := runtime.GOOS == "darwin"
	switch r {
	case RoleAppMenu:
		return []*MenuItem{
			{Role: RoleAbout},
			Separator(),
			{Role: RoleServices},
			Separator(),
			{Role: RoleHide},
			{Role: RoleHideOthers},
			{Role: RoleUnhide},
			Separator(),
			{Role: RoleQuit},
		}
	case RoleFileMenu:
		if hasDocuments() {
			// DocumentGroup's File menu on macOS; KStandardAction's and
			// Notepad's elsewhere.
			if mac {
				return []*MenuItem{
					{Role: RoleNew}, {Role: RoleOpen}, {Role: RoleOpenRecent}, Separator(),
					{Role: RoleClose}, {Role: RoleSave}, {Role: RoleDuplicate},
					{Role: RoleRename}, {Role: RoleMoveTo}, {Role: RoleRevert},
				}
			}
			return []*MenuItem{
				{Role: RoleNew}, {Role: RoleOpen}, {Role: RoleOpenRecent}, Separator(),
				{Role: RoleSave}, {Role: RoleSaveAs}, {Role: RoleRevert}, Separator(),
				{Role: RoleClose}, Separator(), {Role: RoleQuit},
			}
		}
		if mac {
			return []*MenuItem{{Role: RoleClose}}
		}
		return []*MenuItem{{Role: RoleQuit}}
	case RoleOpenRecent:
		return recentMenuItems()
	case RoleEditMenu:
		items := []*MenuItem{
			{Role: RoleUndo},
			{Role: RoleRedo},
			Separator(),
			{Role: RoleCut},
			{Role: RoleCopy},
			{Role: RolePaste},
		}
		if mac {
			return append(items,
				&MenuItem{Role: RolePasteAndMatchStyle},
				&MenuItem{Role: RoleDelete},
				&MenuItem{Role: RoleSelectAll},
				Separator(),
				&MenuItem{Label: menuMessages().MenuSpeech(), Submenu: []*MenuItem{
					{Role: RoleStartSpeaking},
					{Role: RoleStopSpeaking},
				}},
			)
		}
		return append(items, &MenuItem{Role: RoleDelete}, Separator(), &MenuItem{Role: RoleSelectAll})
	case RoleFormatMenu:
		return []*MenuItem{
			{Role: RoleBold},
			{Role: RoleItalic},
			{Role: RoleUnderline},
			{Role: RoleStrikethrough},
			Separator(),
			{Role: RoleAlignLeft},
			{Role: RoleAlignCenter},
			{Role: RoleAlignRight},
		}
	case RoleViewMenu:
		return []*MenuItem{
			{Role: RoleReload},
			{Role: RoleForceReload},
			{Role: RoleToggleDevTools},
			Separator(),
			{Role: RoleResetZoom},
			{Role: RoleZoomIn},
			{Role: RoleZoomOut},
			Separator(),
			{Role: RoleToggleFullScreen},
		}
	case RoleWindowMenu:
		if mac {
			return []*MenuItem{
				{Role: RoleMinimize},
				{Role: RoleZoom},
				Separator(),
				{Role: RoleFront},
			}
		}
		return []*MenuItem{{Role: RoleMinimize}, {Role: RoleClose}}
	}
	return nil
}

// defaultMenu is installed when the app does not set one. Only macOS gets a
// default menu bar: it hosts the standard shortcuts (copy, paste, quit, ...),
// and apps with documents, whose File menu saves them.
func defaultMenu() *Menu {
	if runtime.GOOS != "darwin" {
		if hasDocuments() {
			return NewMenu([]*MenuItem{{Role: RoleFileMenu}, {Role: RoleEditMenu}})
		}
		return nil
	}
	return NewMenu([]*MenuItem{
		{Role: RoleAppMenu},
		{Role: RoleFileMenu},
		{Role: RoleEditMenu},
		{Role: RoleViewMenu},
		{Role: RoleWindowMenu},
	})
}
