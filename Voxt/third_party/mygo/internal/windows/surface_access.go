//go:build windows && (amd64 || arm64)

package windows

import (
	"math"
	"slices"
	"strings"
	"sync"
	"syscall"
	"unsafe"

	"github.com/egoist/mygo/internal/platform"
)

// Assistive technology, such as Narrator, reads native UI through UI
// Automation. The surface's window answers WM_GETOBJECT with a fragment
// root whose fragments are the nodes of the content's tree: COM objects
// with the control patterns of their roles. Invoke presses buttons and
// links, Toggle check boxes and switches, SelectionItem radio buttons and
// the rows lists choose, in the Selection of their list; RangeValue gives
// sliders and progress bars, Value text fields and pop-up buttons, which
// ExpandCollapse opens; ScrollItem scrolls what is in scroll containers
// into view. Rows say which of all of their list's they are
// (PositionInSet, SizeOfSet), as a list builds only those in view.
//
// Two methods take doubles, which Go callbacks cannot read: thunks in
// assembly move their bits to integer registers first (uia_*.s).

var (
	uiaCore  = systemDLL("uiautomationcore.dll")
	oleaut32 = systemDLL("oleaut32.dll")

	procUiaReturnRawElementProvider            = uiaCore.NewProc("UiaReturnRawElementProvider")
	procUiaHostProviderFromHwnd                = uiaCore.NewProc("UiaHostProviderFromHwnd")
	procUiaRaiseAutomationEvent                = uiaCore.NewProc("UiaRaiseAutomationEvent")
	procUiaRaiseAutomationPropertyChangedEvent = uiaCore.NewProc("UiaRaiseAutomationPropertyChangedEvent")
	procUiaRaiseStructureChangedEvent          = uiaCore.NewProc("UiaRaiseStructureChangedEvent")
	procUiaClientsAreListening                 = uiaCore.NewProc("UiaClientsAreListening")
	procUiaDisconnectProvider                  = uiaCore.NewProc("UiaDisconnectProvider")
	procSysAllocString                         = oleaut32.NewProc("SysAllocString")
	procSysFreeString                          = oleaut32.NewProc("SysFreeString")
	procSafeArrayCreateVector                  = oleaut32.NewProc("SafeArrayCreateVector")
	procSafeArrayPutElement                    = oleaut32.NewProc("SafeArrayPutElement")
	procVariantClear                           = oleaut32.NewProc("VariantClear")

	uiaOnce sync.Once
	// The vtables of the interfaces, by uiaIface index.
	uiaVtbls [uiaIfaces][]uintptr
	// uiaElements keeps the elements COM holds alive.
	uiaElements = map[*uiaElement]bool{}

	// The callbacks of the methods that take doubles, which the thunks
	// call with their bits as integers.
	uiaFromPointCallback, uiaSetValueCallback uintptr
)

// uiaThunks returns the entries of the methods that take doubles.
func uiaThunks() (fromPoint, setValue uintptr)

// The thunks, in assembly, called by UI Automation only.
func uiaFromPointThunk()
func uiaSetValueThunk()

// The interfaces of elements.
const (
	ifaceSimple = iota
	ifaceFragment
	ifaceRoot
	ifaceInvoke
	ifaceToggle
	ifaceSelectionItem
	ifaceRangeValue
	ifaceValue
	ifaceExpandCollapse
	ifaceSelection
	ifaceScrollItem
	ifaceHwndOverride
	ifaceText
	uiaIfaces
)

var uiaIIDs = [uiaIfaces]GUID{
	guid("d6dd68d1-86fd-4332-8666-9abedea2d24c"), // IRawElementProviderSimple
	guid("f7063da8-8359-439c-9297-bbc5299a7d87"), // IRawElementProviderFragment
	guid("620ce2a5-ab8f-40a9-86cb-de3c75599b58"), // IRawElementProviderFragmentRoot
	guid("54fcb24b-e18e-47a2-b4d3-eccbe77599a2"), // IInvokeProvider
	guid("56d00bd0-c4f4-433c-a836-1a52a57e0892"), // IToggleProvider
	guid("2acad808-b2d4-452d-a407-91ff1ad167b2"), // ISelectionItemProvider
	guid("36dc7aef-33e6-4691-afe1-2be7274b3d33"), // IRangeValueProvider
	guid("c7935180-6fb3-4201-b174-7df73adbf64a"), // IValueProvider
	guid("d847d3a5-cab0-4a98-8c32-ecb45c59ad24"), // IExpandCollapseProvider
	guid("fb8b03af-3bdf-48d4-bd36-1a65793be168"), // ISelectionProvider
	guid("2360c714-4bf1-4b26-ba65-9b21316127eb"), // IScrollItemProvider
	guid("1d5df27c-8947-4425-b8d9-79787bb460b8"), // IRawElementProviderHwndOverride
	guid("3589c92c-63f3-4367-99bb-ada653b77cf2"), // ITextProvider
}

// Pattern identifiers of UI Automation, by interface.
var uiaPatterns = map[uintptr]int{10000: ifaceInvoke, 10001: ifaceSelection, 10002: ifaceValue, 10003: ifaceRangeValue,
	10005: ifaceExpandCollapse, 10010: ifaceSelectionItem, 10014: ifaceText, 10015: ifaceToggle, 10017: ifaceScrollItem}

const (
	uiaRootObjectID  = -25
	objidClient      = -4
	eventObjectFocus = 0x8005

	uiaRuntimeIDProperty        = 30000
	uiaControlTypeProperty      = 30003
	uiaLocalizedControlProperty = 30004
	uiaNameProperty             = 30005
	uiaHasKeyboardFocusProperty = 30008
	uiaIsFocusableProperty      = 30009
	uiaIsEnabledProperty        = 30010
	uiaHelpTextProperty         = 30013
	uiaIsDataValidProperty      = 30103
	uiaItemStatusProperty       = 30026
	uiaLevelProperty            = 30154
	uiaFullDescriptionProperty  = 30159
	uiaIsPasswordProperty       = 30019
	uiaFrameworkIDProperty      = 30024
	uiaValueProperty            = 30045
	uiaRangeValueProperty       = 30047
	uiaExpandStateProperty      = 30070
	uiaIsSelectedProperty       = 30079
	uiaToggleStateProperty      = 30086
	uiaIsDialogProperty         = 30174
	uiaHeadingLevelProperty     = 30173
	uiaOrientationProperty      = 30023
	uiaIsOffscreenProperty      = 30022
	uiaPositionInSetProperty    = 30152
	uiaSizeOfSetProperty        = 30153
	uiaLiveSettingProperty      = 30135

	uiaFocusChangedEvent                = 20005
	uiaElementAddedToSelectionEvent     = 20010
	uiaElementRemovedFromSelectionEvent = 20011
	uiaElementSelectedEvent             = 20012
	uiaLiveRegionChangedEvent           = 20024
	uiaTextSelectionChangedEvent        = 20014
	uiaTextChangedEvent                 = 20015

	// announcerID identifies the live region holding announcements.
	announcerID = ^uint64(0)

	vtEmpty   = 0
	vtI4      = 3
	vtR8      = 5
	vtBSTR    = 8
	vtBool    = 11
	vtUnknown = 13

	uiaElementNotEnabled   = 0x80040200
	uiaElementNotAvailable = 0x80040201
	uiaInvalidOperation    = 0x80131509
)

// Control types of roles.
var uiaControlTypes = map[platform.AccessRole]int32{
	platform.RoleGroup: 50026, platform.RoleText: 50020, platform.RoleButton: 50000, platform.RoleLink: 50005,
	platform.RoleCheckBox: 50002, platform.RoleRadio: 50013, platform.RoleSwitch: 50000, platform.RoleSlider: 50015,
	platform.RoleProgress: 50012, platform.RoleTextField: 50004, platform.RoleImage: 50006, platform.RoleList: 50008,
	platform.RoleScroll: 50033, platform.RoleDialog: 50033, platform.RolePopup: 50033, platform.RoleTooltip: 50022,
	platform.RolePopUpButton: 50003, platform.RoleTabList: 50018, platform.RoleTab: 50019, platform.RoleSplitter: 50038,
	platform.RoleStatus: 50017, platform.RoleTable: 50036, platform.RoleRow: 50029, platform.RoleCell: 50025,
	platform.RoleColumnHeader: 50035, platform.RoleTree: 50023, platform.RoleTreeItem: 50024,
	platform.RoleListItem: 50007, platform.RoleMenuButton: 50000, platform.RoleToolbar: 50021, platform.RoleRadioGroup: 50026,
	platform.RoleToggleButton: 50000, platform.RoleComboBox: 50003, platform.RoleDisclosure: 50000,
	platform.RoleMeter: 50012, platform.RoleStepper: 50016, platform.RoleColorWell: 50000, platform.RoleAlertDialog: 50033,
	platform.RoleMenu: 50009, platform.RoleMenuBar: 50010, platform.RoleMenuItem: 50011, platform.RoleMenuItemCheckBox: 50011,
	platform.RoleMenuItemRadio: 50011, platform.RoleHeading: 50020,
}

// variant is VARIANT, with the value of the types used here.
type variant struct {
	VT  uint16
	_   [3]uint16
	Val uint64
	_   uintptr
}

// uiaRect is UiaRect.
type uiaRect struct{ Left, Top, Width, Height float64 }

// uiaIface is an interface pointer of an element: COM calls its methods
// with its address.
type uiaIface struct {
	vtbl *uintptr
	e    *uiaElement
}

// uiaElement is a node of the tree as UI Automation sees it, or the root.
type uiaElement struct {
	ifaces [uiaIfaces]uiaIface
	refs   int32
	tree   *uiaTree
	root   bool
	dead   bool
	n      platform.AccessNode
	// textRunes are the runes of n.Value, runesOf the value they are of.
	textRunes []rune
	runesOf   string
	parent    *uiaElement
	children  []*uiaElement
}

// uiaTree is what UI Automation sees of a surface.
type uiaTree struct {
	s     *surface
	root  *uiaElement
	nodes map[uint64]*uiaElement
	order []*uiaElement // in the order of the content's tree
	focus uint64
	quiet bool // building the tree a client asked for
	// announced is the text of the live region of announcements.
	announced string
}

func (e *uiaElement) ptr(i int) uintptr { return uintptr(unsafe.Pointer(&e.ifaces[i])) }

func uiaOf(this uintptr) *uiaElement { return (*uiaIface)(native(this)).e }

func newUIAElement(t *uiaTree) *uiaElement {
	e := &uiaElement{refs: 1, tree: t}
	for i := range e.ifaces {
		e.ifaces[i] = uiaIface{vtbl: &uiaVtbls[i][0], e: e}
	}
	uiaElements[e] = true
	return e
}

func (e *uiaElement) addRef() int32 {
	e.refs++
	return e.refs
}

func (e *uiaElement) release() int32 {
	if e.refs--; e.refs == 0 {
		delete(uiaElements, e)
	}
	return e.refs
}

// supports reports whether the element implements an interface.
func (e *uiaElement) supports(i int) bool {
	n := e.n
	switch i {
	case ifaceSimple, ifaceFragment:
		return true
	case ifaceRoot, ifaceHwndOverride:
		return e.root
	}
	if e.root {
		return false
	}
	switch i {
	case ifaceInvoke:
		switch n.Role {
		case platform.RoleCheckBox, platform.RoleSwitch, platform.RoleRadio, platform.RolePopUpButton, platform.RoleTab, platform.RoleMenuButton,
			platform.RoleToggleButton, platform.RoleDisclosure:
			return false
		}
		return n.Actions&platform.ActionPress != 0
	case ifaceToggle:
		// Items of menus showing a choice toggle it, as WPF's checkable
		// items, and are invoked as well.
		return n.Role == platform.RoleCheckBox || n.Role == platform.RoleSwitch || n.Role == platform.RoleToggleButton ||
			n.Role == platform.RoleMenuItemCheckBox || n.Role == platform.RoleMenuItemRadio
	case ifaceSelectionItem:
		switch n.Role {
		case platform.RoleRadio, platform.RoleTab, platform.RoleTreeItem:
			return true
		case platform.RoleListItem, platform.RoleRow:
			return n.States&platform.AccessSelectable != 0
		}
		return false
	case ifaceSelection:
		// A list or table choosing its rows.
		return (n.Role == platform.RoleList || n.Role == platform.RoleTable || n.Role == platform.RoleTree) && n.States&platform.AccessSelectable != 0
	case ifaceScrollItem:
		return n.Actions&platform.ActionScrollIntoView != 0
	case ifaceRangeValue:
		return n.Role.Ranged()
	case ifaceValue:
		return n.Role == platform.RoleTextField || n.Role == platform.RolePopUpButton || n.Role == platform.RoleComboBox ||
			n.Role == platform.RoleColorWell
	case ifaceText:
		// Text fields' text, by characters, formats, words and lines,
		// with the formats of a rich text's runs; a password's is not.
		return n.Role == platform.RoleTextField && n.States&platform.AccessPassword == 0
	case ifaceExpandCollapse:
		// A menu button expands into its menu, as WinUI's DropDownButton.
		return n.Role == platform.RolePopUpButton || n.Role == platform.RoleTreeItem || n.Role == platform.RoleMenuButton ||
			n.Role == platform.RoleComboBox || n.Role == platform.RoleDisclosure
	}
	return false
}

// out stores an interface of an element, with a reference, at a pointer.
func out(p uintptr, e *uiaElement, i int) uintptr {
	if e == nil {
		*(*uintptr)(native(p)) = 0
		return sOK
	}
	e.addRef()
	*(*uintptr)(native(p)) = e.ptr(i)
	return sOK
}

func setBool(p uintptr, v bool) {
	*(*int32)(native(p)) = 0
	if v {
		*(*int32)(native(p)) = 1
	}
}

func setFloat(p uintptr, v float64) { *(*float64)(native(p)) = v }

func bstr(s string) uintptr {
	b, _, _ := procSysAllocString.Call(uintptr(unsafe.Pointer(u16(s))))
	return b
}

func boolVariant(v bool) variant {
	if v {
		return variant{VT: vtBool, Val: 0xFFFF} // VARIANT_TRUE
	}
	return variant{VT: vtBool}
}

// accessRoot returns the root of the tree UI Automation sees, asking the
// content for the tree while it has none, as before its first frame.
func (s *surface) accessRoot() *uiaElement {
	uiaOnce.Do(initUIA)
	t := s.access
	if t == nil {
		t = &uiaTree{s: s, nodes: map[uint64]*uiaElement{}}
		t.root = newUIAElement(t)
		t.root.root = true
		s.access = t
	}
	if len(t.order) == 0 {
		t.quiet = true
		s.send(platform.SurfaceEvent{Kind: platform.AccessibilityOn})
		t.quiet = false
	}
	return t.root
}

// getObject answers WM_GETOBJECT for UI Automation.
func (s *surface) getObject(wp, lp uintptr) (uintptr, bool) {
	// UI Automation asks for the root (UiaRootObjectId), and, with object
	// IDs from 0 up, for the elements whose events its clients receive:
	// UiaReturnRawElementProvider answers both, and OBJID_CLIENT with the
	// MSAA view of the tree. Other MSAA objects (the title bar, the
	// scroll bars, …) are the window's.
	if id := int32(uint32(lp)); id < 0 && id != uiaRootObjectID && id != objidClient || s.w.closed {
		return 0, false
	}
	root := s.accessRoot()
	r, _, _ := procUiaReturnRawElementProvider.Call(s.hwnd, wp, lp, root.ptr(ifaceSimple))
	return r, true
}

// UpdateAccessibility shows UI Automation the content's last frame, and
// tells its clients what changed.
func (s *surface) UpdateAccessibility(tree *platform.AccessTree) {
	if t := s.access; t != nil && tree != nil {
		t.update(tree)
	}
}

// destroyAccess disconnects the tree when the surface's window goes.
func (s *surface) destroyAccess() {
	t := s.access
	if t == nil {
		return
	}
	s.access = nil
	procUiaReturnRawElementProvider.Call(s.hwnd, 0, 0, 0)
	for _, e := range t.order {
		e.disconnect()
	}
	t.root.disconnect()
	t.nodes, t.order = nil, nil
}

// disconnect lets go of an element that left the tree.
func (e *uiaElement) disconnect() {
	e.dead = true
	if has(procUiaDisconnectProvider) {
		procUiaDisconnectProvider.Call(e.ptr(ifaceSimple))
	}
	e.release()
}

func listening() bool {
	r, _, _ := procUiaClientsAreListening.Call()
	return r != 0
}

func (t *uiaTree) update(tree *platform.AccessTree) {
	// Announcements go into a polite live region at the end of the tree,
	// whose LiveRegionChanged screen readers read, as Flutter's alerts:
	// notification events (UiaRaiseNotificationEvent), which a provider of
	// another process raises, reached no client in tests, unlike its other
	// events.
	if len(tree.Announcements) > 0 {
		t.announced = strings.Join(tree.Announcements, " ")
	}
	if t.announced != "" {
		announcer := platform.AccessNode{ID: announcerID, Parent: -1, Role: platform.RoleText, Label: t.announced, Bounds: platform.RectF{W: 1, H: 1}}
		tree = &platform.AccessTree{Nodes: append(slices.Clip(tree.Nodes), announcer), Focus: tree.Focus, Announcements: tree.Announcements}
	}
	notify := !t.quiet && listening()
	old := t.nodes
	t.nodes = make(map[uint64]*uiaElement, len(tree.Nodes))
	order := make([]*uiaElement, len(tree.Nodes))
	prev := make([]platform.AccessNode, len(tree.Nodes))
	fresh := make([]bool, len(tree.Nodes))
	oldKids := map[*uiaElement][]*uiaElement{t.root: t.root.children}
	for i, n := range tree.Nodes {
		e := old[n.ID]
		if e != nil {
			delete(old, n.ID)
			oldKids[e] = e.children
			prev[i] = e.n
		} else {
			e = newUIAElement(t)
			fresh[i] = true
		}
		e.n, e.children = n, nil
		t.nodes[n.ID] = e
		order[i] = e
	}
	t.root.children = nil
	for _, e := range order {
		e.parent = t.root
		if e.n.Parent >= 0 {
			e.parent = order[e.n.Parent]
		}
		e.parent.children = append(e.parent.children, e)
	}
	t.order = order
	if notify {
		for parent, kids := range oldKids {
			var added, removed []*uiaElement
			for _, k := range kids {
				if !slices.Contains(parent.children, k) {
					removed = append(removed, k)
				}
			}
			for _, k := range parent.children {
				if !slices.Contains(kids, k) {
					added = append(added, k)
				}
			}
			switch {
			case len(added)+len(removed) > 1:
				id := parent.runtimeID()
				procUiaRaiseStructureChangedEvent.Call(parent.ptr(ifaceSimple), 2, // StructureChangeType_ChildrenInvalidated
					uintptr(unsafe.Pointer(unsafe.SliceData(id))), uintptr(len(id)))
			case len(added) == 1:
				id := added[0].runtimeID()
				procUiaRaiseStructureChangedEvent.Call(added[0].ptr(ifaceSimple), 0, // StructureChangeType_ChildAdded
					uintptr(unsafe.Pointer(&id[0])), uintptr(len(id)))
			case len(removed) == 1:
				id := removed[0].runtimeID()
				procUiaRaiseStructureChangedEvent.Call(parent.ptr(ifaceSimple), 1, // StructureChangeType_ChildRemoved
					uintptr(unsafe.Pointer(&id[0])), uintptr(len(id)))
			}
		}
		for i, e := range order {
			if !fresh[i] {
				e.notifyChanges(prev[i])
			}
		}
	}
	for _, e := range old {
		e.disconnect()
	}
	if tree.Focus != t.focus {
		t.focus = tree.Focus
		if e := t.nodes[t.focus]; e != nil && notify {
			procUiaRaiseAutomationEvent.Call(e.ptr(ifaceSimple), uiaFocusChangedEvent)
			// Clients that follow the focus through WinEvents, as those of
			// System.Windows.Automation, then ask the window for it.
			procNotifyWinEvent.Call(eventObjectFocus, t.s.hwnd, uintptr(objidClient&0xffffffff), 0)
		}
	}
	if e := t.nodes[announcerID]; e != nil && notify && len(tree.Announcements) > 0 {
		procUiaRaiseAutomationEvent.Call(e.ptr(ifaceSimple), uiaLiveRegionChangedEvent)
	}
}

// notifyChanges tells UI Automation's clients how an element changed since
// the last tree.
func (e *uiaElement) notifyChanges(prev platform.AccessNode) {
	n := e.n
	changed := func(property int, before, after variant) {
		procUiaRaiseAutomationPropertyChangedEvent.Call(e.ptr(ifaceSimple), uintptr(property),
			uintptr(unsafe.Pointer(&before)), uintptr(unsafe.Pointer(&after)))
	}
	str := func(property int, before, after string) {
		b, a := variant{VT: vtBSTR, Val: uint64(bstr(before))}, variant{VT: vtBSTR, Val: uint64(bstr(after))}
		changed(property, b, a)
		procVariantClear.Call(uintptr(unsafe.Pointer(&b)))
		procVariantClear.Call(uintptr(unsafe.Pointer(&a)))
	}
	if prev.Label != n.Label {
		str(uiaNameProperty, prev.Label, n.Label)
	}
	if prev.States&platform.AccessDisabled != n.States&platform.AccessDisabled {
		changed(uiaIsEnabledProperty, boolVariant(prev.States&platform.AccessDisabled == 0), boolVariant(n.States&platform.AccessDisabled == 0))
	}
	switch {
	case e.supports(ifaceToggle):
		if before, after := toggleState(prev), toggleState(n); before != after {
			changed(uiaToggleStateProperty, variant{VT: vtI4, Val: uint64(before)}, variant{VT: vtI4, Val: uint64(after)})
		}
	case e.supports(ifaceSelectionItem):
		if before, after := prev.States&platform.AccessChecked != 0, n.States&platform.AccessChecked != 0; before != after {
			changed(uiaIsSelectedProperty, boolVariant(before), boolVariant(after))
			// A row of a list choosing several joins the choice or leaves
			// it, unless it is now the only one.
			event := uintptr(uiaElementSelectedEvent)
			if c := e.container(); c != nil && c.n.States&platform.AccessMultiselectable != 0 {
				switch {
				case !after:
					event = uiaElementRemovedFromSelectionEvent
				case len(c.chosen()) > 1:
					event = uiaElementAddedToSelectionEvent
				}
			}
			if after || event != uiaElementSelectedEvent {
				procUiaRaiseAutomationEvent.Call(e.ptr(ifaceSimple), event)
			}
		}
	case e.supports(ifaceRangeValue):
		if prev.Now != n.Now {
			changed(uiaRangeValueProperty, variant{VT: vtR8, Val: math.Float64bits(rangeValue(prev))},
				variant{VT: vtR8, Val: math.Float64bits(rangeValue(n))})
		}
	}
	if e.supports(ifaceValue) && prev.Value != n.Value {
		str(uiaValueProperty, prev.Value, n.Value)
	}
	if e.supports(ifaceText) {
		if prev.Value != n.Value || !slices.Equal(prev.TextRuns, n.TextRuns) {
			procUiaRaiseAutomationEvent.Call(e.ptr(ifaceSimple), uiaTextChangedEvent)
		}
		if prev.SelStart != n.SelStart || prev.SelEnd != n.SelEnd {
			procUiaRaiseAutomationEvent.Call(e.ptr(ifaceSimple), uiaTextSelectionChangedEvent)
		}
	}
	if e.supports(ifaceExpandCollapse) && prev.States&platform.AccessExpanded != n.States&platform.AccessExpanded {
		changed(uiaExpandStateProperty, variant{VT: vtI4, Val: uint64(expandState(prev))}, variant{VT: vtI4, Val: uint64(expandState(n))})
	}
}

// smallChange returns how far the arrow keys move the value of a range:
// its step, else a hundredth of it.
func smallChange(n platform.AccessNode) float64 {
	if n.Step > 0 {
		return n.Step
	}
	return (n.Max - n.Min) / 100
}

func toggleState(n platform.AccessNode) int32 {
	switch {
	case n.States&platform.AccessMixed != 0:
		return 2 // ToggleState_Indeterminate
	case n.States&platform.AccessChecked != 0:
		return 1
	}
	return 0
}

func expandState(n platform.AccessNode) int32 {
	switch {
	case n.States&platform.AccessExpanded != 0:
		return 1 // ExpandCollapseState_Expanded
	case n.Role == platform.RoleTreeItem && n.States&platform.AccessExpandable == 0:
		return 3 // ExpandCollapseState_LeafNode
	}
	return 0
}

// rangeValue is the value of a slider or progress bar, the minimum for a
// progress of unknown length.
func rangeValue(n platform.AccessNode) float64 { return max(n.Now, n.Min) }

// runtimeID returns the element's runtime identifier: UiaAppendRuntimeId
// and its node's identifier, or nothing for the root, which its window
// identifies.
func (e *uiaElement) runtimeID() []int32 {
	if e.root {
		return nil
	}
	return []int32{3, int32(uint32(e.n.ID)), int32(uint32(e.n.ID >> 32))}
}

// act performs an action of assistive technology on the element.
func (e *uiaElement) act(action platform.AccessActionKind, text string) uintptr {
	switch {
	case e.dead:
		return uiaElementNotAvailable
	case e.n.States&platform.AccessDisabled != 0:
		return uiaElementNotEnabled
	}
	e.tree.s.send(platform.SurfaceEvent{Kind: platform.AccessAction, ID: e.n.ID, Action: action, Text: text})
	return sOK
}

// screenRect returns the element's bounds in pixels of the screen.
func (e *uiaElement) screenRect() uiaRect {
	s := e.tree.s
	var origin point
	procClientToScreen.Call(s.hwnd, uintptr(unsafe.Pointer(&origin)))
	scale := float64(s.dpi()) / 96
	b := e.n.Bounds
	return uiaRect{float64(origin.X) + b.X*scale, float64(origin.Y) + b.Y*scale, b.W * scale, b.H * scale}
}

// at returns the deepest element at a point of the screen, in pixels,
// the topmost of those that overlap.
func (t *uiaTree) at(x, y float64) *uiaElement {
	for i := len(t.order) - 1; i >= 0; i-- {
		r := t.order[i].screenRect()
		if x >= r.Left && y >= r.Top && x < r.Left+r.Width && y < r.Top+r.Height {
			return t.order[i]
		}
	}
	return nil
}

func initUIA() {
	cb := syscall.NewCallback
	unknown := []uintptr{
		cb(func(this, riid, p uintptr) uintptr {
			e := uiaOf(this)
			iid := *(*GUID)(native(riid))
			if iid == iidIUnknown {
				return out(p, e, ifaceSimple)
			}
			for i, id := range uiaIIDs {
				if id == iid && e.supports(i) {
					return out(p, e, i)
				}
			}
			*(*uintptr)(native(p)) = 0
			return eNoInterface
		}),
		cb(func(this uintptr) uintptr { return uintptr(uiaOf(this).addRef()) }),
		cb(func(this uintptr) uintptr { return uintptr(uiaOf(this).release()) }),
	}
	vtbl := func(methods ...uintptr) []uintptr { return append(slices.Clone(unknown), methods...) }
	live := func(this uintptr) (*uiaElement, uintptr) {
		e := uiaOf(this)
		if e.dead {
			return nil, uiaElementNotAvailable
		}
		return e, sOK
	}

	// The node of a web view's page is WebView2's provider of the page,
	// in the place of the window WebView2 shows it with (ifaceHwndOverride):
	// it answers with that provider (pageCall), and has no runtime
	// identifier of its own.
	uiaVtbls[ifaceSimple] = vtbl(
		cb(func(this, p uintptr) uintptr { // get_ProviderOptions
			if hr, ok := uiaOf(this).pageCall(uiaGetProviderOptions, p); ok {
				return hr
			}
			*(*int32)(native(p)) = 0x1 | 0x20 // ServerSideProvider, UseComThreading
			return sOK
		}),
		cb(func(this, pattern, p uintptr) uintptr { // GetPatternProvider
			e, hr := live(this)
			if e == nil {
				return hr
			}
			if hr, ok := e.pageCall(uiaGetPatternProvider, pattern, p); ok && hr != eNotImpl {
				return hr
			}
			if i, ok := uiaPatterns[pattern]; ok && e.supports(i) {
				return out(p, e, i)
			}
			return out(p, nil, 0)
		}),
		cb(func(this, property, p uintptr) uintptr { // GetPropertyValue
			e, hr := live(this)
			v := (*variant)(native(p))
			*v = variant{}
			if e == nil {
				return hr
			}
			if hr, ok := e.pageCall(uiaGetPropertyValue, property, p); ok && hr != eNotImpl {
				return hr
			}
			e.property(int(property), v)
			return sOK
		}),
		cb(func(this, p uintptr) uintptr { // get_HostRawElementProvider
			e := uiaOf(this)
			if e.root && !e.dead {
				r, _, _ := procUiaHostProviderFromHwnd.Call(e.tree.s.hwnd, p)
				return r
			}
			if hr, ok := e.pageCall(uiaGetHostRawElementProvider, p); ok {
				return hr
			}
			return out(p, nil, 0)
		}),
	)

	uiaVtbls[ifaceFragment] = vtbl(
		cb(func(this, direction, p uintptr) uintptr { // Navigate
			e, hr := live(this)
			if e == nil {
				return hr
			}
			var to *uiaElement
			siblings := func(d int) *uiaElement {
				if e.root {
					return nil
				}
				kids := e.parent.children
				if i := slices.Index(kids, e) + d; i >= 0 && i < len(kids) {
					return kids[i]
				}
				return nil
			}
			switch direction {
			case 0: // NavigateDirection_Parent
				if !e.root {
					to = e.parent
				}
			case 1: // NextSibling
				to = siblings(1)
			case 2: // PreviousSibling
				to = siblings(-1)
			case 3: // FirstChild
				if len(e.children) > 0 {
					to = e.children[0]
				}
			case 4: // LastChild
				if len(e.children) > 0 {
					to = e.children[len(e.children)-1]
				}
			}
			return out(p, to, ifaceFragment)
		}),
		cb(func(this, p uintptr) uintptr { // GetRuntimeId
			e, hr := live(this)
			*(*uintptr)(native(p)) = 0
			if e == nil {
				return hr
			}
			id := e.runtimeID()
			if id == nil || e.page() != nil { // a page's is its window's
				return sOK
			}
			sa, _, _ := procSafeArrayCreateVector.Call(vtI4, 0, uintptr(len(id)))
			for i := range id {
				index := int32(i)
				procSafeArrayPutElement.Call(sa, uintptr(unsafe.Pointer(&index)), uintptr(unsafe.Pointer(&id[i])))
			}
			*(*uintptr)(native(p)) = sa
			return sOK
		}),
		cb(func(this, p uintptr) uintptr { // get_BoundingRectangle
			e, hr := live(this)
			*(*uiaRect)(native(p)) = uiaRect{}
			if e == nil {
				return hr
			}
			if !e.root { // the root's window gives the root's
				*(*uiaRect)(native(p)) = e.screenRect()
			}
			return sOK
		}),
		cb(func(this, p uintptr) uintptr { // GetEmbeddedFragmentRoots
			*(*uintptr)(native(p)) = 0
			return sOK
		}),
		cb(func(this uintptr) uintptr { // SetFocus
			e, hr := live(this)
			if e == nil {
				return hr
			}
			if v := e.page(); v != nil {
				v.focusWebView()
				return sOK
			}
			procSetFocus.Call(e.tree.s.hwnd)
			if e.root || e.n.Actions&platform.ActionFocus == 0 {
				return sOK
			}
			return e.act(platform.AccessFocus, "")
		}),
		cb(func(this, p uintptr) uintptr { // get_FragmentRoot
			e, hr := live(this)
			if e == nil {
				return hr
			}
			return out(p, e.tree.root, ifaceRoot)
		}),
	)

	uiaFromPointCallback = cb(func(this, x, y, p uintptr) uintptr { // ElementProviderFromPoint
		out(p, nil, 0)
		e, hr := live(this)
		if e == nil {
			return hr
		}
		return out(p, e.tree.at(math.Float64frombits(uint64(x)), math.Float64frombits(uint64(y))), ifaceFragment)
	})
	uiaSetValueCallback = cb(func(this, v uintptr) uintptr { // IRangeValueProvider::SetValue
		e, hr := live(this)
		if e == nil {
			return hr
		}
		return e.setRangeValue(math.Float64frombits(uint64(v)))
	})
	fromPoint, setValue := uiaThunks()

	uiaVtbls[ifaceHwndOverride] = vtbl(
		cb(func(this, hwnd, p uintptr) uintptr { // GetOverrideProviderForHwnd
			e, hr := live(this)
			*(*uintptr)(native(p)) = 0
			if e == nil {
				return hr
			}
			return out(p, e.tree.pageOfWindow(hwnd), ifaceSimple)
		}),
	)

	uiaVtbls[ifaceText] = vtbl(textProviderMethods()...)
	initTextRanges()

	uiaVtbls[ifaceRoot] = vtbl(
		fromPoint,
		cb(func(this, p uintptr) uintptr { // GetFocus
			e, hr := live(this)
			if e == nil {
				return hr
			}
			return out(p, e.tree.nodes[e.tree.focus], ifaceFragment)
		}),
	)

	press := func(this uintptr) uintptr {
		e, hr := live(this)
		if e == nil {
			return hr
		}
		return e.act(platform.AccessPress, "")
	}
	uiaVtbls[ifaceInvoke] = vtbl(cb(press))

	uiaVtbls[ifaceToggle] = vtbl(
		cb(press),
		cb(func(this, p uintptr) uintptr { // get_ToggleState
			*(*int32)(native(p)) = toggleState(uiaOf(this).n)
			return sOK
		}),
	)

	selected := func(e *uiaElement) bool { return e.n.States&platform.AccessChecked != 0 }
	selectItem := cb(func(this uintptr) uintptr { // Select, AddToSelection
		e, hr := live(this)
		if e == nil || selected(e) {
			return hr
		}
		return e.act(platform.AccessPress, "")
	})
	uiaVtbls[ifaceSelectionItem] = vtbl(
		selectItem,
		selectItem,
		cb(func(this uintptr) uintptr { return uiaInvalidOperation }), // RemoveFromSelection
		cb(func(this, p uintptr) uintptr { // get_IsSelected
			setBool(p, selected(uiaOf(this)))
			return sOK
		}),
		cb(func(this, p uintptr) uintptr { // get_SelectionContainer
			e, hr := live(this)
			if e == nil {
				return hr
			}
			return out(p, e.container(), ifaceSimple)
		}),
	)

	uiaVtbls[ifaceSelection] = vtbl(
		cb(func(this, p uintptr) uintptr { // GetSelection
			e, hr := live(this)
			*(*uintptr)(native(p)) = 0
			if e == nil {
				return hr
			}
			chosen := e.chosen()
			sa, _, _ := procSafeArrayCreateVector.Call(vtUnknown, 0, uintptr(len(chosen)))
			for i, d := range chosen {
				index := int32(i)
				procSafeArrayPutElement.Call(sa, uintptr(unsafe.Pointer(&index)), d.ptr(ifaceSimple))
			}
			*(*uintptr)(native(p)) = sa
			return sOK
		}),
		cb(func(this, p uintptr) uintptr { // get_CanSelectMultiple
			e, hr := live(this)
			if e == nil {
				setBool(p, false)
				return hr
			}
			setBool(p, e.n.States&platform.AccessMultiselectable != 0)
			return sOK
		}),
		cb(func(this, p uintptr) uintptr { // get_IsSelectionRequired
			setBool(p, false)
			return sOK
		}),
	)

	uiaVtbls[ifaceScrollItem] = vtbl(
		cb(func(this uintptr) uintptr { // ScrollIntoView
			e, hr := live(this)
			if e == nil {
				return hr
			}
			return e.act(platform.AccessScrollIntoView, "")
		}),
	)

	rangeOf := func(get func(n platform.AccessNode) float64) uintptr {
		return cb(func(this, p uintptr) uintptr {
			setFloat(p, get(uiaOf(this).n))
			return sOK
		})
	}
	uiaVtbls[ifaceRangeValue] = vtbl(
		setValue,
		rangeOf(rangeValue),
		cb(func(this, p uintptr) uintptr { // get_IsReadOnly
			setBool(p, uiaOf(this).n.Actions&platform.ActionIncrement == 0)
			return sOK
		}),
		rangeOf(func(n platform.AccessNode) float64 { return n.Max }),
		rangeOf(func(n platform.AccessNode) float64 { return n.Min }),
		rangeOf(func(n platform.AccessNode) float64 { return max(smallChange(n), (n.Max-n.Min)/10) }), // LargeChange, as Page Up and Down
		rangeOf(smallChange), // SmallChange, as the arrow keys
	)

	uiaVtbls[ifaceValue] = vtbl(
		cb(func(this, s uintptr) uintptr { // SetValue
			e, hr := live(this)
			if e == nil {
				return hr
			}
			if e.n.Actions&platform.ActionSetValue == 0 {
				return uiaInvalidOperation
			}
			return e.act(platform.AccessSetValue, wstr(s))
		}),
		cb(func(this, p uintptr) uintptr { // get_Value
			*(*uintptr)(native(p)) = bstr(uiaOf(this).n.Value)
			return sOK
		}),
		cb(func(this, p uintptr) uintptr { // get_IsReadOnly
			setBool(p, uiaOf(this).n.Actions&platform.ActionSetValue == 0)
			return sOK
		}),
	)

	expand := func(open bool) uintptr {
		return cb(func(this uintptr) uintptr {
			e, hr := live(this)
			if e == nil || (e.n.States&platform.AccessExpanded != 0) == open {
				return hr
			}
			if e.n.Actions&platform.ActionExpand != 0 {
				// An item of a tree, which pressing chooses.
				if !open {
					return e.act(platform.AccessCollapse, "")
				}
				return e.act(platform.AccessExpand, "")
			}
			return e.act(platform.AccessPress, "")
		})
	}
	uiaVtbls[ifaceExpandCollapse] = vtbl(
		expand(true),
		expand(false),
		cb(func(this, p uintptr) uintptr { // get_ExpandCollapseState
			*(*int32)(native(p)) = expandState(uiaOf(this).n)
			return sOK
		}),
	)
}

// setRangeValue moves a slider toward a value, as the arrow keys do.
func (e *uiaElement) setRangeValue(v float64) uintptr {
	switch {
	case e.n.Actions&platform.ActionIncrement == 0:
		return uiaInvalidOperation
	case v > e.n.Now:
		return e.act(platform.AccessIncrement, "")
	case v < e.n.Now:
		return e.act(platform.AccessDecrement, "")
	}
	return sOK
}

// property fills in a property of the element, leaving the ones it has
// not empty for UI Automation's defaults.
func (e *uiaElement) property(id int, v *variant) {
	if e.root {
		return // the window's
	}
	n := e.n
	str := func(s string) {
		if s != "" {
			*v = variant{VT: vtBSTR, Val: uint64(bstr(s))}
		}
	}
	switch id {
	case uiaControlTypeProperty:
		*v = variant{VT: vtI4, Val: uint64(uiaControlTypes[n.Role])}
	case uiaLocalizedControlProperty:
		switch n.Role {
		case platform.RoleSwitch:
			str("toggle switch")
		case platform.RoleMenuButton:
			str("menu button")
		case platform.RoleRadioGroup:
			str("radio group")
		case platform.RoleDialog:
			str("dialog")
		case platform.RoleAlertDialog:
			str("alert dialog")
		case platform.RolePopup:
			str("popup")
		case platform.RoleHeading:
			str("heading")
		}
	case uiaNameProperty:
		str(n.Label)
	case uiaHasKeyboardFocusProperty:
		*v = boolVariant(n.ID == e.tree.focus)
	case uiaIsFocusableProperty:
		*v = boolVariant(n.States&platform.AccessFocusable != 0)
	case uiaIsEnabledProperty:
		*v = boolVariant(n.States&platform.AccessDisabled == 0)
	case uiaHelpTextProperty:
		// The placeholder of a text field, else the description, as
		// Chromium's.
		if n.Placeholder != "" {
			str(n.Placeholder)
		} else {
			str(n.Description)
		}
	case uiaFullDescriptionProperty:
		str(n.Description)
	case uiaItemStatusProperty:
		// The order of the column a table is sorted by, as Chromium says.
		switch {
		case n.States&platform.AccessSortAscending != 0:
			str("ascending")
		case n.States&platform.AccessSortDescending != 0:
			str("descending")
		}
	case uiaIsDataValidProperty:
		*v = boolVariant(n.States&platform.AccessInvalid == 0)
	case uiaIsPasswordProperty:
		*v = boolVariant(n.States&platform.AccessPassword != 0)
	case uiaFrameworkIDProperty:
		str("MyGo")
	case uiaIsDialogProperty:
		*v = boolVariant(n.Role == platform.RoleDialog || n.Role == platform.RoleAlertDialog)
	case uiaIsOffscreenProperty:
		*v = boolVariant(n.States&platform.AccessOffscreen != 0)
	case uiaOrientationProperty:
		// OrientationType_Horizontal or Vertical, of a slider.
		if n.Role == platform.RoleSlider {
			o := uint64(1)
			if n.States&platform.AccessVertical != 0 {
				o = 2
			}
			*v = variant{VT: vtI4, Val: o}
		}
	case uiaHeadingLevelProperty:
		// HeadingLevel1 to HeadingLevel9, as Chromium's headings.
		if n.Role == platform.RoleHeading && n.Level > 0 {
			*v = variant{VT: vtI4, Val: uint64(80050 + min(n.Level, 9))}
		}
	case uiaLevelProperty:
		if n.Level > 0 && n.Role != platform.RoleHeading {
			*v = variant{VT: vtI4, Val: uint64(n.Level)}
		}
	case uiaPositionInSetProperty:
		if n.PosInSet > 0 {
			*v = variant{VT: vtI4, Val: uint64(n.PosInSet)}
		}
	case uiaSizeOfSetProperty:
		if n.SetSize > 0 {
			*v = variant{VT: vtI4, Val: uint64(n.SetSize)}
		}
	case uiaLiveSettingProperty:
		// Statuses, as toasts, and announcements are polite live regions.
		if n.Role == platform.RoleStatus || n.ID == announcerID {
			*v = variant{VT: vtI4, Val: 1} // Polite
		}
	}
}

// chosen returns the elements chosen in the element's Selection.
func (e *uiaElement) chosen() []*uiaElement {
	var chosen []*uiaElement
	for _, d := range e.tree.order {
		if d.n.States&platform.AccessChecked != 0 && d.supports(ifaceSelectionItem) && d.container() == e {
			chosen = append(chosen, d)
		}
	}
	return chosen
}

// container returns the element whose Selection holds the element's, the
// list or table around it choosing it.
func (e *uiaElement) container() *uiaElement {
	for p := e.parent; p != nil && !p.root; p = p.parent {
		if p.supports(ifaceSelection) {
			return p
		}
	}
	return nil
}

// The methods of IRawElementProviderSimple.
const (
	uiaGetProviderOptions        = 3
	uiaGetPatternProvider        = 4
	uiaGetPropertyValue          = 5
	uiaGetHostRawElementProvider = 6
)

// page returns the web view whose page the element stands for, if any.
func (e *uiaElement) page() *window {
	v, ok := e.n.WebView.(*window)
	if !ok || v.closed || v.comp == 0 || v.host == nil || v.host.surface != e.tree.s {
		return nil
	}
	return v
}

// pageCall calls a method of WebView2's provider of the page the element
// stands for, and reports whether it is one's.
func (e *uiaElement) pageCall(method int, args ...uintptr) (uintptr, bool) {
	v := e.page()
	if v == nil || e.dead {
		return 0, false
	}
	p := v.automationProvider()
	if p == 0 {
		return 0, false
	}
	simple := queryInterface(p, &uiaIIDs[ifaceSimple])
	if simple == 0 {
		return 0, false
	}
	defer release(simple)
	return comCall(simple, method, args...), true
}

// pageOfWindow returns the node of the page that WebView2 shows with the
// window hwnd, a child of the surface's, if one does.
func (t *uiaTree) pageOfWindow(hwnd uintptr) *uiaElement {
	env4 := queryInterface(t.s.w.b.env, &iidICoreWebView2Environment4)
	if env4 == 0 {
		return nil
	}
	defer release(env4)
	var p uintptr
	if failed(comCall(env4, env4GetAutomationProviderForWindow, hwnd, uintptr(unsafe.Pointer(&p)))) || p == 0 {
		return nil
	}
	defer release(p)
	want := queryInterface(p, &iidIUnknown)
	defer release(want)
	for _, e := range t.order {
		if v := e.page(); v != nil {
			if q := v.automationProvider(); q != 0 {
				got := queryInterface(q, &iidIUnknown)
				release(got)
				if got == want {
					return e
				}
			}
		}
	}
	return nil
}
