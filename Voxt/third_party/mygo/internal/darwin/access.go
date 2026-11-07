//go:build darwin

package darwin

import (
	"slices"
	"sync"
	"unicode/utf16"

	"github.com/ebitengine/purego"
	"github.com/ebitengine/purego/objc"

	"github.com/egoist/mygo/internal/platform"
)

// VoiceOver and other assistive technology see the content of a surface
// as NSAccessibilityElements, one per node of the content's accessibility
// tree, which the surface view contains. Elements are kept from update to
// update by node ID, so that VoiceOver's cursor stays on them, and only
// what changed is set again. The content describes frames once AppKit
// first asks the view for its children. Lists are tables of rows, as AppKit's
// own lists, and VoiceOver reads their rows "row 5 of 10,000" from their
// index among all of their table's (AXIndex) and how many it has
// (AXRowCount), rather than from the rows built, as a list builds only those
// in view; the table says which rows show (AXVisibleRows) and which are
// chosen (AXSelectedRows). What is in a scroll container takes
// AXScrollToVisible.

// accessElement is the element of a node.
type accessElement struct {
	obj      id // MyGoAccessibilityElement, owned
	s        *surface
	node     platform.AccessNode
	parent   id
	children []uint64
	// webView is the web view whose page is the first child.
	webView *window
	// chosen are the rows of a list or table that are chosen, and count
	// its rows, built or not.
	chosen []uint64
	count  int
}

var (
	accessOnce     sync.Once
	postAccessNote uintptr // NSAccessibilityPostNotification
	// postAccessInfo is NSAccessibilityPostNotificationWithUserInfo.
	postAccessInfo uintptr
	// actionDescription is NSAccessibilityActionDescription.
	actionDescription uintptr
	msgPointToPoint   func(obj id, sel objc.SEL, p NSPoint) NSPoint
)

func loadAccess() {
	accessOnce.Do(func() {
		postAccessNote = mustDlsym(libAppKit, "NSAccessibilityPostNotification")
		postAccessInfo = mustDlsym(libAppKit, "NSAccessibilityPostNotificationWithUserInfo")
		actionDescription = mustDlsym(libAppKit, "NSAccessibilityActionDescription")
		purego.RegisterFunc(&msgPointToPoint, msgSendAddr)
	})
}

// postNote posts an accessibility notification about obj.
func postNote(obj id, name string) {
	if obj != 0 {
		purego.SyscallN(postAccessNote, uintptr(obj), uintptr(nsString(name)))
	}
}

// announce asks VoiceOver to read text out, after what it is reading
// (NSAccessibilityAnnouncementRequestedNotification).
func announce(app id, text string) {
	info := send(class("NSMutableDictionary"), "dictionary")
	send(info, "setObject:forKey:", uintptr(nsString(text)), uintptr(nsString("AXAnnouncementKey")))
	send(info, "setObject:forKey:", uintptr(nsNumberInt(50)), uintptr(nsString("AXPriorityKey"))) // NSAccessibilityPriorityMedium
	purego.SyscallN(postAccessInfo, uintptr(app), uintptr(nsString("AXAnnouncementRequested")), uintptr(info))
}

// accessRoles are the roles and subroles of node roles.
var accessRoles = map[platform.AccessRole][2]string{
	platform.RoleGroup:        {"AXGroup", ""},
	platform.RoleText:         {"AXStaticText", ""},
	platform.RoleButton:       {"AXButton", ""},
	platform.RoleLink:         {"AXLink", ""},
	platform.RoleCheckBox:     {"AXCheckBox", ""},
	platform.RoleRadio:        {"AXRadioButton", ""},
	platform.RoleSwitch:       {"AXCheckBox", "AXSwitch"},
	platform.RoleSlider:       {"AXSlider", ""},
	platform.RoleProgress:     {"AXProgressIndicator", ""},
	platform.RoleTextField:    {"AXTextField", ""},
	platform.RoleImage:        {"AXImage", ""},
	platform.RoleList:         {"AXTable", ""},
	platform.RoleScroll:       {"AXScrollArea", ""},
	platform.RoleDialog:       {"AXGroup", "AXDialog"},
	platform.RolePopup:        {"AXPopover", ""},
	platform.RoleTooltip:      {"AXHelpTag", ""},
	platform.RolePopUpButton:  {"AXPopUpButton", ""},
	platform.RoleTabList:      {"AXTabGroup", ""},
	platform.RoleTab:          {"AXRadioButton", "AXTabButton"},
	platform.RoleSplitter:     {"AXSplitter", ""},
	platform.RoleStatus:       {"AXGroup", "AXApplicationStatus"},
	platform.RoleTable:        {"AXTable", ""},
	platform.RoleRow:          {"AXRow", "AXTableRow"},
	platform.RoleCell:         {"AXCell", ""},
	platform.RoleColumnHeader: {"AXCell", ""},
	platform.RoleTree:         {"AXOutline", ""},
	platform.RoleTreeItem:     {"AXRow", "AXOutlineRow"},
	platform.RoleListItem:     {"AXRow", "AXTableRow"},
	platform.RoleMenuButton:   {"AXMenuButton", ""},
	platform.RoleToolbar:      {"AXToolbar", ""},
	platform.RoleRadioGroup:   {"AXRadioGroup", ""},
	platform.RoleToggleButton: {"AXCheckBox", "AXToggle"},
	platform.RoleComboBox:     {"AXComboBox", ""},
	platform.RoleDisclosure:   {"AXDisclosureTriangle", ""},
	platform.RoleMeter:        {"AXLevelIndicator", ""},
	platform.RoleStepper:      {"AXIncrementor", ""},
	platform.RoleColorWell:    {"AXColorWell", ""},
	platform.RoleAlertDialog:  {"AXGroup", "AXApplicationAlertDialog"},
	platform.RoleMenu:         {"AXMenu", ""},
	platform.RoleMenuBar:      {"AXMenuBar", ""},
	// AppKit's items show a choice with a mark (AXMenuItemMarkChar).
	platform.RoleMenuItem:         {"AXMenuItem", ""},
	platform.RoleMenuItemCheckBox: {"AXMenuItem", ""},
	platform.RoleMenuItemRadio:    {"AXMenuItem", ""},
	// WebKit's headings, whose value is their level.
	platform.RoleHeading: {"AXHeading", ""},
}

// menuItem reports whether nodes of a role are items of a menu.
func menuItem(r platform.AccessRole) bool {
	return r == platform.RoleMenuItem || r == platform.RoleMenuItemCheckBox || r == platform.RoleMenuItemRadio
}

// markChar returns the mark an item of a menu shows for its choice, as
// AppKit's: a check mark when on, a dash when mixed, none otherwise.
func markChar(n platform.AccessNode) string {
	switch {
	case !menuItem(n.Role):
	case n.States&platform.AccessMixed != 0:
		return "-"
	case n.States&platform.AccessChecked != 0:
		return "✓"
	}
	return ""
}

// chooses reports whether a node is the row of a list, a table or an
// outline that its list chooses.
func chooses(n platform.AccessNode) bool {
	return (n.Role == platform.RoleListItem || n.Role == platform.RoleRow || n.Role == platform.RoleTreeItem) && n.States&platform.AccessSelectable != 0
}

func roleOf(n platform.AccessNode) (role, subrole string) {
	r := accessRoles[n.Role]
	role, subrole = r[0], r[1]
	if n.States&platform.AccessSegment != 0 {
		subrole = "AXSegment" // as NSSegmentedControl's
	}
	if n.Role == platform.RoleTextField {
		switch {
		case n.States&platform.AccessSearch != 0:
			subrole = "AXSearchField"
		case n.States&platform.AccessPassword != 0:
			subrole = "AXSecureTextField"
		case n.States&platform.AccessMultiline != 0:
			role = "AXTextArea"
		}
	}
	return role, subrole
}

// titled reports whether elements of a role show their name as a title.
func titled(r platform.AccessRole) bool {
	switch r {
	case platform.RoleButton, platform.RoleLink, platform.RoleCheckBox, platform.RoleRadio, platform.RoleSwitch, platform.RolePopUpButton, platform.RoleTab,
		platform.RoleMenuButton, platform.RoleToggleButton, platform.RoleMenuItem, platform.RoleMenuItemCheckBox, platform.RoleMenuItemRadio:
		return true
	}
	return false
}

// textual reports whether elements of a role edit text: text fields and
// combo boxes.
func textual(r platform.AccessRole) bool {
	return r == platform.RoleTextField || r == platform.RoleComboBox
}

// toggle reports whether elements of a role have their state as their
// value.
func toggle(r platform.AccessRole) bool {
	switch r {
	case platform.RoleCheckBox, platform.RoleRadio, platform.RoleSwitch, platform.RoleTab, platform.RoleToggleButton:
		return true
	}
	return false
}

// valueOf returns the value of a node as AppKit wants it: a number for
// toggles, disclosures and ranges, a string for texts.
func valueOf(n platform.AccessNode) id {
	switch {
	case n.Role == platform.RoleDisclosure:
		// 1 while open, as AppKit's disclosure triangles.
		return nsNumberInt(int(n.States&platform.AccessExpanded) / int(platform.AccessExpanded))
	case toggle(n.Role):
		v := 0
		switch {
		case n.States&platform.AccessMixed != 0:
			v = 2
		case n.States&platform.AccessChecked != 0:
			v = 1
		}
		return nsNumberInt(v)
	case n.Role.Ranged():
		if n.Now < n.Min {
			return 0 // a progress of unknown length
		}
		return msgFloatID(class("NSNumber"), sel("numberWithDouble:"), n.Now)
	case n.Role == platform.RoleText:
		return nsString(n.Label)
	case n.Role == platform.RoleHeading && n.Level > 0:
		return nsNumberInt(n.Level)
	}
	if n.Value != "" || textual(n.Role) {
		return nsString(n.Value)
	}
	return 0
}

// apply sets the properties of a node that changed, all of them for a new
// element, and reports whether its value changed.
func (el *accessElement) apply(n platform.AccessNode, fresh bool) (valueChanged bool) {
	o := el.node
	el.node = n
	obj := el.obj
	r, sr := roleOf(n)
	if or, osr := roleOf(o); fresh || r != or || sr != osr {
		send(obj, "setAccessibilityRole:", uintptr(nsString(r)))
		var sub id
		if sr != "" {
			sub = nsString(sr)
		}
		send(obj, "setAccessibilitySubrole:", uintptr(sub))
	}
	if fresh || n.Label != o.Label || n.Role != o.Role {
		// Controls show their name, which AppKit's own give as their
		// title; other elements are described by it.
		if titled(n.Role) {
			send(obj, "setAccessibilityTitle:", uintptr(nsString(n.Label)))
			send(obj, "setAccessibilityLabel:", 0)
		} else {
			send(obj, "setAccessibilityTitle:", 0)
			send(obj, "setAccessibilityLabel:", uintptr(nsString(n.Label)))
		}
	}
	if fresh || n.States&platform.AccessDisabled != o.States&platform.AccessDisabled {
		send(obj, "setAccessibilityEnabled:", boolArg(n.States&platform.AccessDisabled == 0))
	}
	if fresh || n.States&platform.AccessExpanded != o.States&platform.AccessExpanded {
		send(obj, "setAccessibilityExpanded:", boolArg(n.States&platform.AccessExpanded != 0))
	}
	if n.Role == platform.RoleSlider && (fresh || n.States&platform.AccessVertical != o.States&platform.AccessVertical) {
		// NSAccessibilityOrientationVertical or Horizontal.
		orient := 2
		if n.States&platform.AccessVertical != 0 {
			orient = 1
		}
		send(obj, "setAccessibilityOrientation:", uintptr(orient))
	}
	if n.Role.Ranged() {
		if fresh || n.Min != o.Min || n.Max != o.Max {
			send(obj, "setAccessibilityMinValue:", uintptr(msgFloatID(class("NSNumber"), sel("numberWithDouble:"), n.Min)))
			send(obj, "setAccessibilityMaxValue:", uintptr(msgFloatID(class("NSNumber"), sel("numberWithDouble:"), n.Max)))
		}
	}
	// The state of a toggle is its value; a row's choice is not, which its
	// table tells (AXSelectedRowsChanged).
	toggled := toggle(n.Role) && n.States&(platform.AccessChecked|platform.AccessMixed) != o.States&(platform.AccessChecked|platform.AccessMixed) ||
		n.Role == platform.RoleDisclosure && n.States&platform.AccessExpanded != o.States&platform.AccessExpanded
	if fresh || n.Value != o.Value || n.Now != o.Now || toggled || n.Role == platform.RoleText && n.Label != o.Label ||
		n.Role == platform.RoleHeading && n.Level != o.Level {
		send(obj, "setAccessibilityValue:", uintptr(valueOf(n)))
		valueChanged = !fresh
	}
	if n.Level > 0 && n.Role != platform.RoleHeading && (fresh || n.Level != o.Level) {
		// How deep the row of an outline is, from 0.
		send(obj, "setAccessibilityDisclosureLevel:", uintptr(n.Level-1))
	}
	if n.Role == platform.RoleTreeItem && (fresh || n.States&platform.AccessExpanded != o.States&platform.AccessExpanded) {
		send(obj, "setAccessibilityDisclosed:", boolArg(n.States&platform.AccessExpanded != 0))
	}
	if n.PosInSet > 0 && (fresh || n.PosInSet != o.PosInSet) {
		send(obj, "setAccessibilityIndex:", uintptr(n.PosInSet-1))
	}
	if chooses(n) && (fresh || !chooses(o) || n.States&platform.AccessChecked != o.States&platform.AccessChecked) {
		send(obj, "setAccessibilitySelected:", boolArg(n.States&platform.AccessChecked != 0))
	}
	if fresh || n.Placeholder != o.Placeholder {
		var p id
		if n.Placeholder != "" {
			p = nsString(n.Placeholder)
		}
		send(obj, "setAccessibilityPlaceholderValue:", uintptr(p))
	}
	if sorted := platform.AccessSortAscending | platform.AccessSortDescending; n.Role == platform.RoleColumnHeader && (fresh || n.States&sorted != o.States&sorted) {
		// NSAccessibilitySortDirection.
		dir := 0
		switch {
		case n.States&platform.AccessSortAscending != 0:
			dir = 1
		case n.States&platform.AccessSortDescending != 0:
			dir = 2
		}
		send(obj, "setAccessibilitySortDirection:", uintptr(dir))
	}
	if fresh || n.Description != o.Description {
		// AppKit's help, which it gives tooltips as.
		var d id
		if n.Description != "" {
			d = nsString(n.Description)
		}
		send(obj, "setAccessibilityHelp:", uintptr(d))
	}
	if textual(n.Role) && (fresh || n.Value != o.Value || n.SelStart != o.SelStart || n.SelEnd != o.SelEnd) {
		text := []rune(n.Value)
		a, b := max(0, min(n.SelStart, len(text))), max(0, min(n.SelEnd, len(text)))
		send(obj, "setAccessibilityNumberOfCharacters:", uintptr(units(text)))
		send(obj, "setAccessibilitySelectedTextRange:", uintptr(units(text[:a])), uintptr(units(text[a:max(a, b)])))
	}
	return valueChanged
}

// accessRequested starts the content describing its frames, the first
// time AppKit asks about the surface.
func (s *surface) accessRequested() {
	if !s.accessOn {
		s.accessOn = true
		s.send(platform.SurfaceEvent{Kind: platform.AccessibilityOn})
	}
}

func (s *surface) UpdateAccessibility(tree *platform.AccessTree) {
	if tree == nil || s.w.closed {
		return
	}
	loadAccess()
	withPool(func() {
		s.updating = true
		defer func() { s.updating = false }()
		if s.elements == nil {
			s.elements = map[uint64]*accessElement{}
		}
		changed := false
		objs := make([]id, len(tree.Nodes))
		var valueChanged []id
		fresh := map[uint64]bool{}
		for i, n := range tree.Nodes {
			el := s.elements[n.ID]
			if el == nil {
				fresh[n.ID] = true
				el = &accessElement{obj: send(send(class("MyGoAccessibilityElement"), "alloc"), "init"), s: s}
				s.elements[n.ID] = el
				s.w.b.byAccess[el.obj] = el
				el.apply(n, true)
				changed = true
			} else if el.apply(n, false) {
				valueChanged = append(valueChanged, el.obj)
			}
			objs[i] = el.obj
		}
		children := make([][]uint64, len(tree.Nodes))
		childObjs := make([][]id, len(tree.Nodes))
		var top []id
		for i, n := range tree.Nodes {
			if n.Parent < 0 || n.Parent >= len(tree.Nodes) {
				top = append(top, objs[i])
				continue
			}
			children[n.Parent] = append(children[n.Parent], n.ID)
			childObjs[n.Parent] = append(childObjs[n.Parent], objs[i])
		}
		grafted := map[*window]bool{}
		for i, n := range tree.Nodes {
			el := s.elements[n.ID]
			parent := s.view
			if n.Parent >= 0 && n.Parent < len(tree.Nodes) {
				parent = objs[n.Parent]
			}
			if el.parent != parent {
				el.parent = parent
				send(el.obj, "setAccessibilityParent:", uintptr(parent))
			}
			// The page of a web view is inside the element showing it.
			var web *window
			if v, ok := n.WebView.(*window); ok && v.host == s.w && !v.closed && !grafted[v] {
				web, grafted[v] = v, true
				v.accessParent = el.obj
			}
			if !slices.Equal(el.children, children[i]) || el.webView != web {
				el.children, el.webView = children[i], web
				kids := childObjs[i]
				if web != nil {
					kids = append([]id{web.web}, kids...)
				}
				send(el.obj, "setAccessibilityChildren:", uintptr(nsArray(kids...)))
				changed = true
			}
		}
		for _, v := range s.w.webViews {
			if !grafted[v] {
				v.accessParent = 0
			}
		}
		// Lists and tables: their rows, those that show, those chosen, and
		// how many they have, built or not. A list's own rows say where
		// they are among all; the rows of other tables, where among those.
		var selectionChanged []*accessElement
		for i, n := range tree.Nodes {
			if n.Role != platform.RoleList && n.Role != platform.RoleTable && n.Role != platform.RoleTree {
				continue
			}
			el := s.elements[n.ID]
			var rows, shown, chosen []id
			var chosenIDs []uint64
			for _, cid := range children[i] {
				c := s.elements[cid]
				if r := c.node.Role; r != platform.RoleRow && r != platform.RoleListItem && r != platform.RoleTreeItem || n.SetSize > 0 && c.node.PosInSet == 0 {
					continue // a table's header
				}
				if c.node.PosInSet == 0 {
					send(c.obj, "setAccessibilityIndex:", uintptr(len(rows)))
				}
				rows = append(rows, c.obj)
				if c.node.States&platform.AccessOffscreen == 0 {
					shown = append(shown, c.obj)
				}
				if chooses(c.node) && c.node.States&platform.AccessChecked != 0 {
					chosen = append(chosen, c.obj)
					chosenIDs = append(chosenIDs, cid)
				}
			}
			count := n.SetSize
			if count == 0 {
				count = len(rows)
			}
			if count != el.count {
				el.count = count
				send(el.obj, "setAccessibilityRowCount:", uintptr(count))
			}
			send(el.obj, "setAccessibilityRows:", uintptr(nsArray(rows...)))
			send(el.obj, "setAccessibilityVisibleRows:", uintptr(nsArray(shown...)))
			send(el.obj, "setAccessibilitySelectedRows:", uintptr(nsArray(chosen...)))
			if !slices.Equal(el.chosen, chosenIDs) {
				el.chosen = chosenIDs
				if !fresh[n.ID] { // a new list's choice did not change
					selectionChanged = append(selectionChanged, el)
				}
			}
		}
		release(s.topLevel)
		s.topLevel = retain(nsArray(top...))
		live := make(map[uint64]bool, len(tree.Nodes))
		for _, n := range tree.Nodes {
			live[n.ID] = true
		}
		for nid, el := range s.elements {
			if !live[nid] {
				postNote(el.obj, "AXUIElementDestroyed")
				delete(s.w.b.byAccess, el.obj)
				release(el.obj)
				delete(s.elements, nid)
				changed = true
			}
		}
		if changed {
			postNote(s.view, "AXLayoutChanged")
		}
		for _, obj := range valueChanged {
			postNote(obj, "AXValueChanged")
		}
		for _, el := range selectionChanged {
			postNote(el.obj, "AXSelectedRowsChanged")
		}
		if tree.Focus != s.access.Focus {
			if el := s.elements[tree.Focus]; el != nil {
				postNote(el.obj, "AXFocusedUIElementChanged")
			}
		}
		for _, text := range tree.Announcements {
			announce(s.w.b.app, text)
		}
		s.access = *tree
		s.access.Announcements = nil
	})
}

// destroyAccess releases the elements of a surface.
func (s *surface) destroyAccess() {
	for _, v := range s.w.webViews {
		v.accessParent = 0
	}
	for nid, el := range s.elements {
		postNote(el.obj, "AXUIElementDestroyed")
		delete(s.w.b.byAccess, el.obj)
		release(el.obj)
		delete(s.elements, nid)
	}
	release(s.topLevel)
	s.topLevel = 0
}

// accessAt returns the innermost element at a point of the view.
func (s *surface) accessAt(x, y float64) id {
	found := id(0)
	for _, n := range s.access.Nodes {
		b := n.Bounds
		if x >= b.X && y >= b.Y && x < b.X+b.W && y < b.Y+b.H {
			if el := s.elements[n.ID]; el != nil {
				found = el.obj // later nodes are deeper or above
			}
		}
	}
	return found
}

// screenRect converts a rectangle of the view to the screen.
func (s *surface) screenRect(b platform.RectF) NSRect {
	inWindow := msgConvertRectView(s.view, sel("convertRect:toView:"), NSRect{Origin: NSPoint{b.X, b.Y}, Size: NSSize{b.W, b.H}}, 0)
	return msgRectToRect(s.w.win, sel("convertRectToScreen:"), inWindow)
}

func (b *Backend) accessElementOf(obj id) *accessElement {
	el := b.byAccess[obj]
	if el == nil || el.s.w.closed {
		return nil
	}
	return el
}

// act sends an action of assistive technology to the content.
func (el *accessElement) act(a platform.AccessActionKind, text string) bool {
	if el.s.updating {
		return false
	}
	el.s.send(platform.SurfaceEvent{Kind: platform.AccessAction, ID: el.node.ID, Action: a, Text: text})
	return true
}

// legacyActions are the actions of nodes by their names in AppKit's older
// API: AXScrollToVisible is NSAccessibilityScrollToVisibleAction, which
// AppKit exports only since macOS 26.
var legacyActions = []struct {
	name   string
	action platform.AccessActions
	kind   platform.AccessActionKind
}{
	{"AXPress", platform.ActionPress, platform.AccessPress},
	{"AXIncrement", platform.ActionIncrement, platform.AccessIncrement},
	{"AXDecrement", platform.ActionDecrement, platform.AccessDecrement},
	{"AXScrollToVisible", platform.ActionScrollIntoView, platform.AccessScrollIntoView},
}

func registerAccessClass() {
	b := func() *Backend { return theBackend }
	allowed := map[string]platform.AccessActions{
		"accessibilityPerformPress":     platform.ActionPress,
		"accessibilityPerformIncrement": platform.ActionIncrement,
		"accessibilityPerformDecrement": platform.ActionDecrement,
		"setAccessibilityValue:":        platform.ActionSetValue,
		"setAccessibilityFocused:":      platform.ActionFocus,
		"setAccessibilityDisclosed:":    platform.ActionExpand,
	}
	classDef("MyGoAccessibilityElement", "NSAccessibilityElement", nil, []objc.MethodDef{
		method("accessibilityFrame", func(self id, _ objc.SEL) NSRect {
			if el := b().accessElementOf(self); el != nil {
				return el.s.screenRect(el.node.Bounds)
			}
			return NSRect{}
		}),
		method("isAccessibilityFocused", func(self id, _ objc.SEL) bool {
			el := b().accessElementOf(self)
			return el != nil && el.s.access.Focus == el.node.ID
		}),
		// AXInvalid, which the newer API lacks, says a value is not valid,
		// as WebKit's fields do, and AXMenuItemMarkChar what an item of a
		// menu shows of its choice.
		method("accessibilityAttributeNames", func(self id, cmd objc.SEL) id {
			names := sendSuper(self, "MyGoAccessibilityElement", cmd)
			el := b().accessElementOf(self)
			if el == nil {
				return names
			}
			if el.node.States&platform.AccessInvalid != 0 {
				names = send(names, "arrayByAddingObject:", uintptr(nsString("AXInvalid")))
			}
			if menuItem(el.node.Role) {
				names = send(names, "arrayByAddingObject:", uintptr(nsString("AXMenuItemMarkChar")))
			}
			return names
		}),
		method("accessibilityAttributeValue:", func(self id, cmd objc.SEL, attr id) id {
			if el := b().accessElementOf(self); el != nil {
				switch stringOf(attr) {
				case "AXInvalid":
					if el.node.States&platform.AccessInvalid != 0 {
						return nsString("true")
					}
				case "AXMenuItemMarkChar":
					if m := markChar(el.node); m != "" {
						return nsString(m)
					}
					if menuItem(el.node.Role) {
						return 0
					}
				}
			}
			return sendSuper(self, "MyGoAccessibilityElement", cmd, uintptr(attr))
		}),
		// The older API asks which attributes are settable, which
		// NSAccessibilityElement does not answer (calling super throws):
		// AppKit's answer without it is the attributes of the setters the
		// class overrides, AXValue, AXFocused and a row's AXDisclosing,
		// kept but for the value of a read-only text field, which is not
		// settable, as WebKit's.
		method("accessibilityIsAttributeSettable:", func(self id, _ objc.SEL, attr id) bool {
			el := b().accessElementOf(self)
			if el == nil {
				return false
			}
			switch stringOf(attr) {
			case "AXValue":
				return !textual(el.node.Role) || el.node.Actions&platform.ActionSetValue != 0
			case "AXFocused":
				return true
			case "AXDisclosing":
				return el.node.Role == platform.RoleTreeItem
			}
			return false
		}),
		method("isAccessibilitySelectorAllowed:", func(self id, cmd objc.SEL, selector objc.SEL) bool {
			if el := b().accessElementOf(self); el != nil {
				for name, action := range allowed {
					if sel(name) == selector {
						return el.node.Actions&action != 0
					}
				}
			}
			return byte(sendSuper(self, "MyGoAccessibilityElement", cmd, uintptr(selector))) != 0
		}),
		// Scrolling into view is an action of the older API alone, which
		// VoiceOver asks for as it moves to what is out of view: answering
		// it, an element answers for all its actions there.
		method("accessibilityActionNames", func(self id, _ objc.SEL) id {
			var names []id
			if el := b().accessElementOf(self); el != nil {
				for _, a := range legacyActions {
					if el.node.Actions&a.action != 0 {
						names = append(names, nsString(a.name))
					}
				}
			}
			return nsArray(names...)
		}),
		method("accessibilityActionDescription:", func(self id, _ objc.SEL, action id) id {
			r, _, _ := purego.SyscallN(actionDescription, uintptr(action))
			return id(r)
		}),
		method("accessibilityPerformAction:", func(self id, _ objc.SEL, action id) {
			el := b().accessElementOf(self)
			if el == nil {
				return
			}
			name := stringOf(action)
			for _, a := range legacyActions {
				if a.name == name && el.node.Actions&a.action != 0 {
					el.act(a.kind, "")
				}
			}
		}),
		method("accessibilityPerformPress", func(self id, _ objc.SEL) bool {
			el := b().accessElementOf(self)
			return el != nil && el.act(platform.AccessPress, "")
		}),
		method("accessibilityPerformIncrement", func(self id, _ objc.SEL) bool {
			el := b().accessElementOf(self)
			return el != nil && el.act(platform.AccessIncrement, "")
		}),
		method("accessibilityPerformDecrement", func(self id, _ objc.SEL) bool {
			el := b().accessElementOf(self)
			return el != nil && el.act(platform.AccessDecrement, "")
		}),
		// Opening and closing a row of an outline, as VoiceOver does.
		method("setAccessibilityDisclosed:", func(self id, cmd objc.SEL, open bool) {
			sendSuper(self, "MyGoAccessibilityElement", cmd, boolArg(open))
			if el := b().accessElementOf(self); el != nil && el.node.Actions&platform.ActionExpand != 0 {
				kind := platform.AccessCollapse
				if open {
					kind = platform.AccessExpand
				}
				el.act(kind, "")
			}
		}),
		method("setAccessibilityFocused:", func(self id, _ objc.SEL, focused bool) {
			if el := b().accessElementOf(self); el != nil && focused {
				el.act(platform.AccessFocus, "")
			}
		}),
		method("setAccessibilityValue:", func(self id, cmd objc.SEL, value id) {
			sendSuper(self, "MyGoAccessibilityElement", cmd, uintptr(value))
			if el := b().accessElementOf(self); el != nil && textual(el.node.Role) {
				el.act(platform.AccessSetValue, stringOf(value))
			}
		}),
		// Moving the selection of a text field, as VoiceOver does.
		method("setAccessibilitySelectedTextRange:", func(self id, cmd objc.SEL, r nsRange) {
			sendSuper(self, "MyGoAccessibilityElement", cmd, uintptr(r.Location), uintptr(r.Length))
			el := b().accessElementOf(self)
			if el == nil || !textual(el.node.Role) || el.s.updating {
				return
			}
			u := utf16.Encode([]rune(el.node.Value))
			lo := int(min(r.Location, uint(len(u))))
			hi := int(min(uint(lo)+r.Length, uint(len(u))))
			start := len(utf16.Decode(u[:lo]))
			end := start + len(utf16.Decode(u[lo:hi]))
			el.s.send(platform.SurfaceEvent{Kind: platform.AccessAction, ID: el.node.ID, Action: platform.AccessSelectText, From: start, To: end})
		}),
		// The text of text fields, for reading by characters and lines.
		method("accessibilityStringForRange:", func(self id, _ objc.SEL, r nsRange) id {
			el := b().accessElementOf(self)
			if el == nil {
				return 0
			}
			u := utf16.Encode([]rune(el.node.Value))
			lo := int(min(r.Location, uint(len(u))))
			hi := int(min(uint(lo)+r.Length, uint(len(u))))
			return nsString(string(utf16.Decode(u[lo:hi])))
		}),
		// The formats of a rich text's runs, as NSTextView gives them, in
		// accessibility's own keys: VoiceOver reads them as bold, italic,
		// links.
		method("accessibilityAttributedStringForRange:", func(self id, _ objc.SEL, r nsRange) id {
			el := b().accessElementOf(self)
			if el == nil {
				return 0
			}
			return attributedAccessString(&el.node, r)
		}),
		method("accessibilityLineForIndex:", func(self id, _ objc.SEL, i int) int { return 0 }),
		method("accessibilityRangeForLine:", func(self id, _ objc.SEL, line int) nsRange {
			if el := b().accessElementOf(self); el != nil {
				return nsRange{Length: uint(units([]rune(el.node.Value)))}
			}
			return nsRange{}
		}),
		method("accessibilityFrameForRange:", func(self id, _ objc.SEL, r nsRange) NSRect {
			if el := b().accessElementOf(self); el != nil {
				return el.s.screenRect(el.node.Bounds)
			}
			return NSRect{}
		}),
	})
}

// accessViewMethods are the methods of the surface view that make it the
// container of the elements.
func accessViewMethods() []objc.MethodDef {
	b := func() *Backend { return theBackend }
	return []objc.MethodDef{
		method("isAccessibilityElement", func(self id, _ objc.SEL) bool { return false }),
		method("accessibilityChildren", func(self id, _ objc.SEL) id {
			s := b().surfaceOf(self)
			if s == nil {
				return 0
			}
			s.accessRequested()
			if s.topLevel == 0 {
				return nsArray()
			}
			return s.topLevel
		}),
		method("accessibilityHitTest:", func(self id, _ objc.SEL, p NSPoint) id {
			s := b().surfaceOf(self)
			if s == nil {
				return self
			}
			loadAccess()
			s.accessRequested()
			inWindow := msgPointToPoint(s.w.win, sel("convertPointFromScreen:"), p)
			v := msgPointFromView(self, sel("convertPoint:fromView:"), inWindow, 0)
			if obj := s.accessAt(v.X, v.Y); obj != 0 {
				return obj
			}
			return self
		}),
		method("accessibilityFocusedUIElement", func(self id, _ objc.SEL) id {
			s := b().surfaceOf(self)
			if s == nil {
				return self
			}
			s.accessRequested()
			if el := s.elements[s.access.Focus]; el != nil {
				return el.obj
			}
			return self
		}),
	}
}
