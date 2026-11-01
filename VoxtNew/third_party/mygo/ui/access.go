package ui

import (
	"slices"
	"strings"

	"github.com/egoist/mygo/internal/platform"
)

// Assistive technology, such as screen readers, sees the elements of each
// frame as a tree once it asks for the window: widgets, texts, and the
// elements that take the focus or have a Label, with their states. It acts
// on them as the keyboard and the pointer would: pressing clicks, setting
// a text field's value edits it.

// Role is what an element is to assistive technology. Widgets set their
// own; other elements get one from what they do: one that is clickable and
// takes the focus is a button, a text is a text, a scroll container a
// scroll area, and one with a Label, or that takes the focus, a group.
type Role uint8

const (
	// RoleAuto finds the role from what the element does.
	RoleAuto Role = iota
	// RoleNone leaves the element out, but not its children.
	RoleNone
	RoleGroup
	RoleText
	RoleButton
	RoleLink
	RoleCheckBox
	RoleRadio
	RoleSwitch
	RoleSlider
	RoleProgress
	RoleTextField
	RoleImage
	RoleList
	RoleScroll
	RoleDialog
	RolePopup
	RoleTooltip
	RolePopUpButton
	RoleTabList
	RoleTab
	RoleSplitter
	RoleStatus
	RoleTable
	RoleRow
	RoleCell
	RoleColumnHeader
	RoleTree
	RoleTreeItem
	// RoleListItem is an item of a list: the rows of a List are.
	RoleListItem
	// RoleMenuButton is a button opening a menu (Element.Menu).
	RoleMenuButton
	// RoleToolbar holds controls (Toolbar), RoleRadioGroup radio buttons
	// (RadioGroup, Segmented), and RoleToggleButton is a button that stays
	// pressed (Toggle).
	RoleToolbar
	RoleRadioGroup
	RoleToggleButton
	// RoleComboBox is a text input with a popup of options (ComboboxBase).
	RoleComboBox
	// RoleDisclosure is a button showing or hiding content below it, as
	// the trigger of a collapsible (CollapsibleBase), expanded while open.
	RoleDisclosure
	// RoleMeter shows a value in a range (Meter), and RoleStepper steps a
	// value in one (Stepper).
	RoleMeter
	RoleStepper
	// RoleColorWell is a button showing a color, which it opens a picker
	// of (ColorWell).
	RoleColorWell
	// RoleAlertDialog is a dialog asking about something important
	// (AlertDialog).
	RoleAlertDialog
	// RoleMenu holds the items of a menu drawn in the window, as a drop-down
	// menu's panel, and RoleMenuBar the titles of menus along it.
	// RoleMenuItem is an item of either, and RoleMenuItemCheckBox and
	// RoleMenuItemRadio items showing a choice, checked with Checked.
	RoleMenu
	RoleMenuBar
	RoleMenuItem
	RoleMenuItemCheckBox
	RoleMenuItemRadio
	// RoleHeading is the title of a section, named by its text, its rank
	// set with Level.
	RoleHeading
)

// Role sets what the element is to assistive technology, for an element
// drawn as a widget it is not built from, such as a custom toggle.
func (e *node) Role(r Role) *node {
	e.role = r
	return e
}

// The states of an element of your own, which a widget's base sets
// itself, tell assistive technology what it shows, as ARIA's states do
// on the web.

// Checked tells assistive technology whether the element, as a check box,
// switch, radio button, toggle button or menu item of your own, is on.
// Bases set it from their value.
func (e *node) Checked(on bool) *node {
	e.checked = 1 + int8(b2f(on))
	return e
}

// Mixed tells assistive technology that the element, a check box, is
// partly on, as one checking a group whose boxes differ.
func (e *node) Mixed() *node {
	e.checked = 3
	return e
}

// Expanded tells assistive technology whether what the element opens
// shows, as the popup of a button or the section below a header.
func (e *node) Expanded(open bool) *node {
	e.expanded = open
	return e
}

// Value sets what assistive technology reads as the element's value, as
// the choice a button opening a popup shows.
func (e *node) Value(s string) *node {
	e.accValue = s
	return e
}

// Range tells assistive technology the range and the value of a slider,
// progress bar, meter or stepper of your own: value, from lo to hi. Step
// tells it how far the keys move the value.
func (e *node) Range(lo, hi, value float64) *node {
	e.hasRange, e.accRange = true, [3]float64{lo, hi, value}
	return e
}

// Level tells assistive technology the rank of a heading, from 1 for the
// highest, or how deep an item of a tree is, from 1 at the top.
func (e *node) Level(n int) *node {
	e.level = max(n, 0)
	return e
}

// ActiveDescendant tells assistive technology that d has the keyboard
// focus while the element does: the option the arrows are on in a list or
// a menu that keeps the focus itself, as a combobox's input does. Call it
// once d is built, inside the element or in its popup.
func (e *node) ActiveDescendant(d *node) *node {
	e.activeDescendant = d
	return e
}

// accessRole returns the element's role, and false for elements assistive
// technology does not see.
func (e *node) accessRole() (platform.AccessRole, bool) {
	switch e.role {
	case RoleNone:
		return 0, false
	case RoleAuto:
	default:
		return platform.AccessRole(e.role - RoleGroup), true
	}
	switch {
	case e.flags&flagEditable != 0:
		return platform.RoleTextField, true
	case e.kind == kindText:
		return platform.RoleText, e.text != ""
	case e.kind == kindImage || e.kind == kindIcon:
		return platform.RoleImage, e.label != "" // others are decoration
	case e.flags&flagClickable != 0 && e.flags&flagFocusable != 0:
		return platform.RoleButton, true
	case e.flags&(flagScrollX|flagScrollY) != 0:
		return platform.RoleScroll, true
	case e.label != "" || e.flags&flagFocusable != 0:
		return platform.RoleGroup, true
	}
	return 0, false
}

// leafRole reports whether elements of a role take the text inside them
// as their name, rather than showing it as elements of its own.
func leafRole(r platform.AccessRole) bool {
	switch r {
	case platform.RoleGroup, platform.RoleList, platform.RoleScroll, platform.RoleDialog, platform.RoleAlertDialog, platform.RolePopup, platform.RoleStatus,
		platform.RoleTabList, platform.RoleTable, platform.RoleRow, platform.RoleCell, platform.RoleTree, platform.RoleListItem,
		platform.RoleToolbar, platform.RoleRadioGroup, platform.RoleMenu, platform.RoleMenuBar:
		return false
	}
	return true
}

// innerText returns the texts of an element and those inside it, joined
// by spaces.
func (e *node) innerText() string {
	var b strings.Builder
	var walk func(e *node)
	walk = func(e *node) {
		if e.flags&flagInvisible != 0 {
			return
		}
		if e.kind == kindText {
			if e.text != "" {
				if b.Len() > 0 {
					b.WriteByte(' ')
				}
				b.WriteString(e.text)
			}
			return // its text holds that of the elements inside it
		}
		for ch := e.first; ch != nil; ch = ch.next {
			walk(ch)
		}
	}
	walk(e)
	return b.String()
}

// accessTree describes the last frame for assistive technology.
func (rt *engine) accessTree() *platform.AccessTree {
	t := &platform.AccessTree{}
	var focused *node
	switch root := rt.c.root; {
	case root == nil:
	case rt.modal != 0 && rt.modalLayer != 0 && rt.c.overlay != nil:
		// What is behind a dialog is inert: the dialog on top and what
		// shows above it alone.
		on := false
		for ch := rt.c.overlay.first; ch != nil; ch = ch.next {
			if on = on || ch.id == rt.modalLayer; on {
				rt.accessElement(t, ch, -1, false, &focused)
			}
		}
	default:
		rt.accessElement(t, root, -1, false, &focused)
	}
	if rt.windowFocus() && rt.focused != 0 {
		focus := rt.focused
		// A list choosing rows with the arrows has the focus for them:
		// assistive technology follows the row chosen, as the arrows move
		// the choice, while it shows.
		if f := focused.rowsOfElement(); f != nil && f.s.cursor() != nil {
			for _, r := range f.rows {
				if r.i == *f.s.cursor() {
					focus = r.e.id
				}
			}
		}
		// A combobox has the focus for the option its arrows are on.
		if focused != nil && focused.activeDescendant != nil {
			focus = focused.activeDescendant.id
		}
		for _, id := range []uint64{focus, rt.focused} {
			if t.Focus == 0 && slices.ContainsFunc(t.Nodes, func(n platform.AccessNode) bool { return n.ID == id }) {
				t.Focus = id
			}
		}
	}
	if len(rt.announcements) > 0 {
		t.Announcements = slices.Clone(rt.announcements)
	}
	return t
}

// listOf returns the List the element is, if any.
func (e *node) listOf() *listFrame {
	if e == nil {
		return nil
	}
	return e.list
}

// rowsOfElement returns the list whose rows the element holds, if any.
func (e *node) rowsOfElement() *listFrame {
	if e == nil {
		return nil
	}
	return e.rowsOf
}

// accessElement adds the nodes of e and the elements inside it, below
// node parent; scrolled is set inside a scroll container, and focused gets
// the element with the keyboard focus.
func (rt *engine) accessElement(t *platform.AccessTree, e *node, parent int, scrolled bool, focused **node) {
	if e.flags&(flagInvisible|flagInert) != 0 {
		return
	}
	if e.id == rt.focused {
		*focused = e
	}
	if role, ok := e.accessRole(); ok {
		n := platform.AccessNode{
			ID: e.id, Parent: parent, Role: role, Label: e.label,
			Bounds: accessBounds(e),
		}
		if scrolled {
			n.Actions |= platform.ActionScrollIntoView
		}
		rt.accessDetails(e, &n)
		t.Nodes = append(t.Nodes, n)
		parent = len(t.Nodes) - 1
		if e.kind == kindText {
			rt.accessInline(t, e, parent)
			return
		}
		if leafRole(role) {
			return
		}
	}
	if e.webView != nil {
		// Its page first, as it paints, in a node of its own that the
		// platform's elements of the page are inside.
		if v := rt.host.platformWebView(e.webView); v != nil {
			b := e.contentBox()
			t.Nodes = append(t.Nodes, platform.AccessNode{
				ID: e.id ^ pageNodeID, Parent: parent, Role: platform.RoleGroup, WebView: v,
				Bounds: platform.RectF{X: float64(b.X), Y: float64(b.Y), W: float64(b.W), H: float64(b.H)},
			})
		}
	}
	// Children in flow first, absolute ones above them, as they paint.
	scrolled = scrolled || e.scrolls()
	for ch := e.first; ch != nil; ch = ch.next {
		if ch.flags&flagAbsolute == 0 {
			rt.accessElement(t, ch, parent, scrolled, focused)
		}
	}
	for ch := e.first; ch != nil; ch = ch.next {
		if ch.flags&flagAbsolute != 0 {
			rt.accessElement(t, ch, parent, scrolled, focused)
		}
	}
}

// pageNodeID tells the node of a web view's page from its element's.
const pageNodeID = 0x9e3779b97f4a7c15

// accessDetails fills in the name, value, states and actions of a node.
func (rt *engine) accessDetails(e *node, n *platform.AccessNode) {
	disabled := e.IsDisabled()
	if disabled {
		n.States |= platform.AccessDisabled
	}
	if e.segment {
		n.States |= platform.AccessSegment
	}
	if e.vertical {
		n.States |= platform.AccessVertical
	}
	if e.search {
		n.States |= platform.AccessSearch
	}
	if e.invalid {
		n.States |= platform.AccessInvalid
	}
	switch e.sort {
	case 1:
		n.States |= platform.AccessSortAscending
	case 2:
		n.States |= platform.AccessSortDescending
	}
	n.Description = e.description
	if f := e.nameFrom; f != nil {
		switch {
		case n.Label == "":
			n.Label = f.nameOf()
		case e.nameJoin && f.nameOf() != "":
			n.Label = f.nameOf() + " " + n.Label
		}
	}
	if e.setSize > 0 {
		n.PosInSet, n.SetSize = e.setPos, e.setSize
	}
	if e.choosesItems {
		n.States |= platform.AccessSelectable
	}
	if e.chooseMany {
		n.States |= platform.AccessMultiselectable
	}
	// Rows and items of lists, and statuses, are named by their content,
	// as leaves are, but show what is inside them too, as a toast's button.
	if n.Label == "" && (leafRole(n.Role) || n.Role == platform.RoleListItem || n.Role == platform.RoleRow || n.Role == platform.RoleStatus) {
		n.Label = e.innerText()
	}
	switch e.checked {
	case 2:
		n.States |= platform.AccessChecked
	case 3:
		n.States |= platform.AccessMixed
	}
	if e.expanded {
		n.States |= platform.AccessExpanded
	}
	if e.expandable {
		n.States |= platform.AccessExpandable
	}
	n.Level = e.level
	n.Value = e.accValue
	if e.hasRange {
		n.Min, n.Max, n.Now, n.Step = e.accRange[0], e.accRange[1], e.accRange[2], e.accStep
	}
	// A row of a list says which of all it is, built or not, and the list
	// how many it has.
	if f := e.parent.listOf(); e.listRow && f != nil {
		n.PosInSet, n.SetSize = e.rowIndex+1, f.n
	}
	if f := e.rowsOf; f != nil {
		n.SetSize = f.n
		if f.s.cursor() != nil {
			n.States |= platform.AccessSelectable // as do its rows
		}
		if f.s.Selection != nil {
			n.States |= platform.AccessMultiselectable
		}
	}
	if e.flags&flagChoosable != 0 {
		n.States |= platform.AccessSelectable
	}
	// What a scroll container built out of view, as the rows of a list
	// beyond its edges.
	if st := e.st; (st.vw <= 0 || st.vh <= 0) && e.w > 0 && e.h > 0 {
		n.States |= platform.AccessOffscreen
	}
	if disabled {
		return
	}
	if e.flags&(flagFocusable|flagEditable|flagChoosable) != 0 {
		// Focusing a row a list chooses chooses it.
		n.States |= platform.AccessFocusable
		n.Actions |= platform.ActionFocus
	}
	if e.flags&flagClickable != 0 {
		n.Actions |= platform.ActionPress
	}
	if n.Role == platform.RoleSlider || n.Role == platform.RoleStepper {
		n.Actions |= platform.ActionIncrement | platform.ActionDecrement
	}
	if e.expandable {
		n.Actions |= platform.ActionExpand
	}
	if ed := e.st.editor; ed != nil && e.flags&flagEditable != 0 {
		if ed.readOnly {
			n.States |= platform.AccessReadOnly
		} else {
			n.Actions |= platform.ActionSetValue
		}
		if ed.multiline {
			n.States |= platform.AccessMultiline
		}
		n.Placeholder = ed.placeholder
		if ed.password {
			n.States |= platform.AccessPassword
		} else {
			n.Value = ed.buf.string()
			n.SelStart, n.SelEnd = ed.selection()
		}
	}
	if e.accFill != nil {
		e.accFill(n)
	}
}

// accessibilityOn starts describing frames for assistive technology,
// with the last one at once.
func (rt *engine) accessibilityOn() {
	if !rt.access {
		rt.access = true
		if rt.c.root == nil {
			rt.requestFrame()
		}
	}
	rt.host.updateAccessibility(rt.accessTree())
}

// accessAction performs an action of assistive technology on an element
// of the last frame, as the keyboard or the pointer would.
func (rt *engine) accessAction(ev platform.SurfaceEvent) {
	s := rt.states[ev.ID]
	if s == nil || s.flags&flagDisabled != 0 {
		return
	}
	switch ev.Action {
	case platform.AccessPress:
		if s.flags&flagMenuButton != 0 {
			rt.openMenuButton(s)
			break
		}
		if s.flags&flagClickable != 0 {
			s.clicks++
			s.clickMods = 0
		} else {
			rt.focusOn(s)
		}
	case platform.AccessFocus:
		if s.flags&flagChoosable != 0 {
			// A row of a list choosing rows: its list takes the focus for
			// it, choosing it.
			s.clicks++
			s.clickMods = 0
			break
		}
		rt.focusOn(s)
	case platform.AccessScrollIntoView:
		// The rows of lists around it scroll into view by their place, as
		// the next frame may not build them, then it does.
		for row := s; row != nil; row = rt.states[row.parent] {
			if l := rt.states[row.parent]; l != nil && l.list != nil {
				if r, ok := l.list.rows[row.id]; ok {
					l.list.ScrollIntoView(r.row)
				}
			}
			if row.parent == 0 {
				break
			}
		}
		rt.reveal(s.id)
	case platform.AccessIncrement, platform.AccessDecrement:
		rt.focusOn(s)
		// Sliders step right and left, spin buttons up and down, as do
		// sliders laid out right to left, whose Right goes down.
		up, down := KeyRight, KeyLeft
		if s.role == RoleStepper || s.rtl {
			up, down = KeyUp, KeyDown
		}
		k := up
		if ev.Action == platform.AccessDecrement {
			k = down
		}
		rt.keys = append(rt.keys, keyEvent{0, k})
	case platform.AccessExpand, platform.AccessCollapse:
		// The tree opens or closes the item as it builds it.
		s.expand = 1
		if ev.Action == platform.AccessCollapse {
			s.expand = -1
		}
	case platform.AccessSelectText:
		if ed := s.editor; ed != nil && s.flags&flagEditable != 0 {
			n := ed.buf.n
			ed.editorSelection = editorSelection{anchor: max(0, min(ev.From, n)), caret: max(0, min(ev.To, n))}
			if ed.area != nil {
				ed.area.reveal = true
			}
			break
		}
		if s.input != nil {
			s.input(InputEvent{Kind: InputSelect, Start: ev.From, End: ev.To, Time: rt.now()})
		}
	case platform.AccessSetValue:
		if s.editor == nil || s.flags&flagEditable == 0 || s.editor.readOnly {
			return
		}
		rt.focusOn(s)
		s.editor.queue = append(s.editor.queue, editEvent{kind: editCommand, text: "selectAll"}, editEvent{kind: editInsert, text: ev.Text})
	}
	rt.requestFrame()
}

// focusOn gives an element the keyboard focus, as Tab would.
func (rt *engine) focusOn(s *state) {
	if s.flags&(flagFocusable|flagEditable) == 0 {
		return
	}
	if rt.focused != s.id {
		rt.focused = s.id
		rt.blinkStart = rt.now()
	}
	rt.focusVisible = true
	rt.reveal(s.id)
}
