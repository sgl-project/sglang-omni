//go:build linux && (amd64 || arm64)

package linux

import (
	"math"
	"slices"
	"strconv"
	"sync"
	"unicode"
	"unicode/utf8"
	"unsafe"

	"github.com/ebitengine/purego"

	"github.com/egoist/mygo/internal/platform"
)

// Assistive technology, such as Orca, reads applications through AT-SPI,
// which GTK bridges to ATK. The widget of a surface is a subclass of
// GtkDrawingArea whose accessible, a subclass of GtkWidgetAccessible, has
// the nodes of the content's tree below it: ATK objects of types
// registered here, which answer from the last tree and turn what assistive
// technology does into AccessAction events. The page of a web view is
// inside the node standing for it: the node's child is the web view's
// accessible, an AtkSocket the web process's tree is embedded in, which
// the overlay holding the web views and the surface (a subclass) does not
// list beside the content.
//
// Registering types writes functions into ATK's class and interface
// structures at offsets of its headers. The class's size is checked
// against the one GObject reports, and the interfaces' layout against
// ATK's version; native UI goes without accessibility rather than write
// past them.

var (
	accessOnce sync.Once

	// The types: the drawing area, its accessible, and the nodes', by
	// the interfaces they implement.
	areaType, glAreaType, rootType, nodeType, rangeType, textType, editableType, selectionType uintptr
	overlayType, overlayAccessType                                                             uintptr

	gTypeRegisterStaticSimple func(parent uintptr, name *byte, classSize uint32, classInit ptr, instanceSize uint32, instanceInit ptr, flags uint32) uintptr
	gTypeAddInterfaceStatic   func(instanceType, ifaceType uintptr, info *gInterfaceInfo)
	gTypeQuery                func(t uintptr, q *gTypeQueryInfo)
	gObjectNew                func(t uintptr, end ptr) ptr
	gValueInit                func(v ptr, t uintptr) ptr
	gValueSetDouble           func(v ptr, d float64)
	gStrdup                   func(s *byte) ptr
	gMalloc0                  func(n uintptr) ptr
	gSlistAppend              func(list, data ptr) ptr
	gSignalEmitChild          func(obj ptr, signal *byte, index uint32, child ptr)
	gSignalEmitText           func(obj ptr, signal *byte, pos, length int32, text *byte)
	gSignalEmitInt            func(obj ptr, signal *byte, v int32)
	gSignalEmit               func(obj ptr, signal *byte)
	gObjectNotify             func(obj ptr, property *byte)
	// gSignalEmitAnnouncement emits AtkObject's announcement signal, which
	// ATK has had since 2.46: announces is set where it has.
	gSignalLookup           func(name *byte, t uintptr) uint32
	gSignalEmitAnnouncement func(obj ptr, signal *byte, text *byte)
	announces               bool

	gtkWidgetAccessibleGetType      func() uintptr
	gtkDrawingAreaGetType           func() uintptr
	gtkGLAreaGetType                func() uintptr
	gtkWidgetClassSetAccessibleType func(class ptr, t uintptr)
	gtkWidgetClassSetAccessibleRole func(class ptr, role int32)
	gtkAccessibleGetWidget          func(accessible ptr) ptr
	gtkWidgetGetAccessible          func(widget ptr) ptr
	gtkOverlayGetType               func() uintptr
	gtkContainerAccessibleGetType   func() uintptr
	atkSocketGetType                func() uintptr

	atkGetMajorVersion, atkGetMinorVersion                  func() uint32
	atkObjectGetType, atkComponentGetType, atkActionGetType func() uintptr
	atkValueGetType, atkTextGetType, atkEditableTextGetType func() uintptr
	atkSelectionGetType                                     func() uintptr
	atkRoleForName, atkStateTypeForName                     func(name *byte) int32
	atkObjectSetName, atkObjectSetDescription               func(obj ptr, name *byte)
	atkObjectSetRole                                        func(obj ptr, role int32)
	atkObjectSetParent                                      func(obj, parent ptr)
	atkObjectNotifyStateChange                              func(obj ptr, state uint64, value bool)
	atkStateSetNew                                          func() ptr
	atkStateSetAddState                                     func(set ptr, state int32) bool
	atkRangeNew                                             func(lower, upper float64, description *byte) ptr
	atkComponentGetExtents                                  func(obj ptr, x, y, w, h *int32, coords int32)

	cbAreaClassInit, cbRootClassInit, cbNodeClassInit, cbRootComponent     ptr
	cbComponentInit, cbActionInit, cbValueInit, cbTextInit, cbEditableInit ptr
	cbSelectionInit, cbOverlayClassInit, cbOverlayAccessClassInit          ptr

	// rootParentIndex is GtkWidgetAccessible's get_index_in_parent.
	rootParentIndex ptr

	atkRoles  = map[platform.AccessRole]int32{}
	atkRole   struct{ panel, text, password, listBox int32 }
	atkStates struct {
		enabled, sensitive, visible, showing, focusable, focused, checkable, checked, indeterminate,
		expandable, expanded, editable, readOnly, multiLine, singleLine, selectableText, defunct int32
		selectable, selected, multiselectable, hasPopup, invalidEntry, vertical, horizontal int32
	}

	// The trees of surfaces, by the surface and by its accessible, and the
	// nodes, by their ATK object.
	surfaceTrees  = map[*surface]*accessTree{}
	accessTrees   = map[ptr]*accessTree{}
	accessObjects = map[ptr]*accessNode{}

	// Static strings of actions.
	clickName, expandName, emptyName = []byte("click\x00"), []byte("expand or contract\x00"), []byte("\x00")
)

// gTypeQueryInfo is GTypeQuery.
type gTypeQueryInfo struct {
	typ                     uintptr
	name                    ptr
	classSize, instanceSize uint32
}

// gInterfaceInfo is GInterfaceInfo.
type gInterfaceInfo struct{ init, finalize, data ptr }

const (
	atkObjectClassSize = 352 // GObjectClass, 26 functions and a pad
	gTypeDouble        = 15 << 2
	atkXYParent        = 2
)

// Offsets of the functions of AtkObjectClass.
const (
	atkGetNChildren      = 160
	atkRefChild          = 168
	atkGetIndexInParent  = 176
	atkRefStateSet       = 216
	atkGetAttributes     = 328
	gTypeInterfaceHeader = 16 // the functions of interfaces follow a GTypeInterface
)

// newSurfaceArea creates the widget of a surface, a GtkGLArea or a drawing
// area, that assistive technology sees the content of when ATK allows it.
func newSurfaceArea(gl bool) ptr {
	accessOnce.Do(registerAccess)
	switch {
	case gl && glAreaType != 0:
		return gObjectNew(glAreaType, 0)
	case gl:
		return gtkGLAreaNew()
	case areaType != 0:
		return gObjectNew(areaType, 0)
	}
	return gtkDrawingAreaNew()
}

// slot returns the address of a function pointer in a C structure.
func slot(p ptr, offset uintptr) *ptr {
	return (*ptr)(unsafe.Add(*(*unsafe.Pointer)(unsafe.Pointer(&p)), offset))
}

func setIface(iface ptr, fns map[int]ptr) {
	for i, fn := range fns {
		*slot(iface, gTypeInterfaceHeader+uintptr(i)*8) = fn
	}
}

func registerAccess() {
	libATK, err := open("libatk-1.0.so.0")
	if err != nil {
		return
	}
	a, o, g, t := libATK, libGObject, libGLib, libGTK
	ok := bind(a, &atkGetMajorVersion, "atk_get_major_version") && bind(a, &atkGetMinorVersion, "atk_get_minor_version")
	for _, b := range []struct {
		lib  ptr
		fn   any
		name string
	}{
		{o, &gTypeRegisterStaticSimple, "g_type_register_static_simple"}, {o, &gTypeAddInterfaceStatic, "g_type_add_interface_static"},
		{o, &gTypeQuery, "g_type_query"}, {o, &gObjectNew, "g_object_new"}, {o, &gValueInit, "g_value_init"},
		{o, &gValueSetDouble, "g_value_set_double"}, {g, &gStrdup, "g_strdup"}, {g, &gMalloc0, "g_malloc0"},
		{g, &gSlistAppend, "g_slist_append"}, {o, &gSignalEmitChild, "g_signal_emit_by_name"},
		{o, &gSignalEmitText, "g_signal_emit_by_name"}, {o, &gSignalEmitInt, "g_signal_emit_by_name"},
		{o, &gSignalEmit, "g_signal_emit_by_name"}, {o, &gObjectNotify, "g_object_notify"},
		{t, &gtkWidgetAccessibleGetType, "gtk_widget_accessible_get_type"}, {t, &gtkDrawingAreaGetType, "gtk_drawing_area_get_type"},
		{t, &gtkWidgetClassSetAccessibleType, "gtk_widget_class_set_accessible_type"},
		{t, &gtkWidgetClassSetAccessibleRole, "gtk_widget_class_set_accessible_role"},
		{t, &gtkAccessibleGetWidget, "gtk_accessible_get_widget"}, {t, &gtkWidgetGetAccessible, "gtk_widget_get_accessible"},
		{a, &atkObjectGetType, "atk_object_get_type"}, {a, &atkComponentGetType, "atk_component_get_type"},
		{a, &atkActionGetType, "atk_action_get_type"}, {a, &atkValueGetType, "atk_value_get_type"},
		{a, &atkTextGetType, "atk_text_get_type"}, {a, &atkEditableTextGetType, "atk_editable_text_get_type"},
		{a, &atkSelectionGetType, "atk_selection_get_type"},
		{a, &atkRoleForName, "atk_role_for_name"}, {a, &atkStateTypeForName, "atk_state_type_for_name"},
		{a, &atkObjectSetName, "atk_object_set_name"}, {a, &atkObjectSetRole, "atk_object_set_role"},
		{a, &atkObjectSetDescription, "atk_object_set_description"},
		{a, &atkObjectSetParent, "atk_object_set_parent"}, {a, &atkObjectNotifyStateChange, "atk_object_notify_state_change"},
		{a, &atkStateSetNew, "atk_state_set_new"}, {a, &atkStateSetAddState, "atk_state_set_add_state"},
		{a, &atkRangeNew, "atk_range_new"}, {a, &atkComponentGetExtents, "atk_component_get_extents"},
	} {
		ok = ok && bind(b.lib, b.fn, b.name)
	}
	// The value and text functions written below arrived in ATK 2.12.
	if !ok || atkGetMajorVersion() < 2 || atkGetMajorVersion() == 2 && atkGetMinorVersion() < 12 {
		return
	}
	var q gTypeQueryInfo
	if gTypeQuery(atkObjectGetType(), &q); q.classSize != atkObjectClassSize {
		return
	}
	loadAccessNames()
	initAccessCallbacks()
	announces = bind(o, &gSignalLookup, "g_signal_lookup") && bind(o, &gSignalEmitAnnouncement, "g_signal_emit_by_name") &&
		gSignalLookup(cs("announcement"), atkObjectGetType()) != 0
	register := func(parent uintptr, name string, classInit ptr) uintptr {
		var q gTypeQueryInfo
		gTypeQuery(parent, &q)
		return gTypeRegisterStaticSimple(parent, cs(name), q.classSize, classInit, q.instanceSize, 0, 0)
	}
	implement := func(t, iface uintptr, init ptr) {
		gTypeAddInterfaceStatic(t, iface, &gInterfaceInfo{init: init})
	}
	rootType = register(gtkWidgetAccessibleGetType(), "MyGoSurfaceAccessible", cbRootClassInit)
	nodeType = register(atkObjectGetType(), "MyGoAccessible", cbNodeClassInit)
	if rootType == 0 || nodeType == 0 {
		return
	}
	rangeType = register(nodeType, "MyGoAccessibleRange", 0)
	textType = register(nodeType, "MyGoAccessibleText", 0)
	editableType = register(textType, "MyGoAccessibleEditable", 0)
	selectionType = register(nodeType, "MyGoAccessibleSelection", 0)
	if rangeType == 0 || textType == 0 || editableType == 0 || selectionType == 0 {
		return
	}
	implement(rootType, atkComponentGetType(), cbRootComponent)
	implement(nodeType, atkComponentGetType(), cbComponentInit)
	implement(nodeType, atkActionGetType(), cbActionInit)
	implement(rangeType, atkValueGetType(), cbValueInit)
	implement(textType, atkTextGetType(), cbTextInit)
	implement(editableType, atkEditableTextGetType(), cbEditableInit)
	implement(selectionType, atkSelectionGetType(), cbSelectionInit)
	areaType = register(gtkDrawingAreaGetType(), "MyGoSurfaceArea", cbAreaClassInit)
	// GtkGLArea came in GTK 3.16.
	if bind(t, &gtkGLAreaGetType, "gtk_gl_area_get_type") {
		glAreaType = register(gtkGLAreaGetType(), "MyGoSurfaceGLArea", cbAreaClassInit)
	}
	// The overlay holding web views lists the surface alone, whose nodes
	// hold the pages.
	if bind(t, &gtkOverlayGetType, "gtk_overlay_get_type") && bind(t, &gtkContainerAccessibleGetType, "gtk_container_accessible_get_type") &&
		bind(a, &atkSocketGetType, "atk_socket_get_type") {
		if overlayAccessType = register(gtkContainerAccessibleGetType(), "MyGoSurfaceOverlayAccessible", cbOverlayAccessClassInit); overlayAccessType != 0 {
			overlayType = register(gtkOverlayGetType(), "MyGoSurfaceOverlay", cbOverlayClassInit)
		}
	}
}

// newOverlay creates the overlay putting a surface's area over its web
// views.
func newOverlay() ptr {
	if overlayType != 0 {
		return gObjectNew(overlayType, 0)
	}
	return gtkOverlayNew()
}

// layersOf returns the surface whose web views the overlay holds.
func layersOf(overlay ptr) *surface {
	for _, s := range surfaceAreas {
		if s.layers == overlay && overlay != 0 {
			return s
		}
	}
	return nil
}

// pageAccessible returns the accessible of a web view, the AtkSocket its
// page's tree is embedded in, or 0.
func (w *window) pageAccessible() ptr {
	if atkSocketGetType == nil || w.web == 0 {
		return 0
	}
	if acc := gtkWidgetGetAccessible(w.web); acc != 0 && gTypeCheckInstanceIsA(acc, atkSocketGetType()) {
		return acc
	}
	return 0
}

// loadAccessNames finds ATK's roles and states by name, which, unlike
// their numbers, do not change between versions.
func loadAccessNames() {
	role := func(names ...string) int32 {
		for _, n := range names {
			if r := atkRoleForName(cs(n)); r != 0 {
				return r
			}
		}
		return 0
	}
	atkRole.panel, atkRole.text, atkRole.password = role("panel"), role("text"), role("password text")
	atkRole.listBox = role("list box")
	for r, names := range map[platform.AccessRole][]string{
		platform.RoleGroup: {"panel"}, platform.RoleText: {"label"}, platform.RoleButton: {"push button", "button"},
		platform.RoleLink: {"link"}, platform.RoleCheckBox: {"check box"}, platform.RoleRadio: {"radio button"},
		platform.RoleSwitch: {"switch", "toggle button"}, platform.RoleSlider: {"slider"},
		platform.RoleProgress: {"progress bar"}, platform.RoleTextField: {"entry"}, platform.RoleImage: {"image"},
		platform.RoleList: {"list"}, platform.RoleScroll: {"scroll pane"}, platform.RoleDialog: {"dialog"},
		platform.RolePopup: {"popup menu"}, platform.RoleTooltip: {"tool tip"}, platform.RolePopUpButton: {"combo box"},
		platform.RoleTabList: {"page tab list"}, platform.RoleTab: {"page tab"}, platform.RoleSplitter: {"separator"},
		platform.RoleStatus: {"notification", "status bar"}, platform.RoleTable: {"table"}, platform.RoleRow: {"table row"},
		platform.RoleCell: {"table cell"}, platform.RoleColumnHeader: {"column header", "table column header"},
		platform.RoleTree: {"tree", "tree table"}, platform.RoleTreeItem: {"tree item", "list item"},
		platform.RoleListItem: {"list item"}, platform.RoleMenuButton: {"push button menu", "push button", "button"},
		platform.RoleToolbar: {"tool bar"}, platform.RoleRadioGroup: {"panel"}, platform.RoleToggleButton: {"toggle button"},
		platform.RoleComboBox: {"combo box"}, platform.RoleDisclosure: {"toggle button"},
		platform.RoleMeter: {"level bar", "progress bar"}, platform.RoleStepper: {"spin button"},
		platform.RoleColorWell: {"push button", "button"}, platform.RoleAlertDialog: {"alert", "dialog"},
		platform.RoleMenu: {"menu"}, platform.RoleMenuBar: {"menu bar"}, platform.RoleMenuItem: {"menu item"},
		platform.RoleMenuItemCheckBox: {"check menu item", "menu item"}, platform.RoleMenuItemRadio: {"radio menu item", "menu item"},
		platform.RoleHeading: {"heading", "label"},
	} {
		atkRoles[r] = role(names...)
	}
	st := &atkStates
	for p, name := range map[*int32]string{
		&st.enabled: "enabled", &st.sensitive: "sensitive", &st.visible: "visible", &st.showing: "showing",
		&st.focusable: "focusable", &st.focused: "focused", &st.checkable: "checkable", &st.checked: "checked",
		&st.indeterminate: "indeterminate", &st.expandable: "expandable", &st.expanded: "expanded",
		&st.editable: "editable", &st.readOnly: "read-only", &st.multiLine: "multi-line",
		&st.singleLine: "single-line", &st.selectableText: "selectable-text", &st.defunct: "defunct",
		&st.selectable: "selectable", &st.selected: "selected", &st.multiselectable: "multiselectable",
		&st.hasPopup: "has-popup", &st.invalidEntry: "invalid-entry", &st.vertical: "vertical", &st.horizontal: "horizontal",
	} {
		*p = atkStateTypeForName(cs(name))
	}
}

// accessTree is what assistive technology sees of a surface.
type accessTree struct {
	s     *surface
	root  ptr // the surface's accessible
	nodes map[uint64]*accessNode
	top   []*accessNode
	focus uint64
	quiet bool // building the tree assistive technology asked for
}

// accessNode is a node of the tree, with its ATK object.
type accessNode struct {
	obj      ptr // owned
	typ      uintptr
	tree     *accessTree
	parent   *accessNode
	children []*accessNode
	n        platform.AccessNode
	// chosen are the rows a list choosing its rows chose, among those
	// built.
	chosen []uint64
	// page is the accessible of the web view's page the node stands for,
	// its only child.
	page ptr
}

// chooses reports whether a node is the row of a list, a table or an
// outline that its list chooses.
func chooses(n platform.AccessNode) bool {
	return (n.Role == platform.RoleListItem || n.Role == platform.RoleRow || n.Role == platform.RoleTreeItem) && n.States&platform.AccessSelectable != 0
}

// chosenRows returns the rows of a list or table that are chosen.
func (an *accessNode) chosenRows() []*accessNode {
	var rows []*accessNode
	for _, c := range an.children {
		if chooses(c.n) && c.n.States&platform.AccessChecked != 0 {
			rows = append(rows, c)
		}
	}
	return rows
}

// rootTree returns the tree below a surface's accessible, asking the
// content for it while it has none, as before its first frame.
func rootTree(root ptr) *accessTree {
	t := accessTrees[root]
	if t == nil {
		s := surfaceAreas[gtkAccessibleGetWidget(root)]
		if s == nil || s.w.closed {
			return nil
		}
		t = &accessTree{s: s, root: root, nodes: map[uint64]*accessNode{}}
		accessTrees[root], surfaceTrees[s] = t, t
	}
	if len(t.top) == 0 {
		t.quiet = true
		t.s.send(platform.SurfaceEvent{Kind: platform.AccessibilityOn})
		t.quiet = false
	}
	return t
}

// UpdateAccessibility shows assistive technology the content's last
// frame, and tells it what changed.
func (s *surface) UpdateAccessibility(tree *platform.AccessTree) {
	if t := surfaceTrees[s]; t != nil && tree != nil {
		t.update(tree)
	}
}

func (s *surface) destroyAccess() {
	t := surfaceTrees[s]
	if t == nil {
		return
	}
	delete(surfaceTrees, s)
	delete(accessTrees, t.root)
	t.graft(nil)
	for _, an := range t.nodes {
		an.release()
	}
	t.nodes, t.top = nil, nil
}

// release lets go of a node that left the tree.
func (an *accessNode) release() {
	atkObjectNotifyStateChange(an.obj, uint64(atkStates.defunct), true)
	delete(accessObjects, an.obj)
	gObjectUnref(an.obj)
}

func accessType(n platform.AccessNode) uintptr {
	switch n.Role {
	case platform.RoleSlider, platform.RoleProgress, platform.RoleMeter, platform.RoleStepper:
		return rangeType
	case platform.RoleTextField, platform.RoleComboBox:
		return editableType
	case platform.RoleText:
		return textType
	case platform.RoleList, platform.RoleTable, platform.RoleTree:
		if n.States&platform.AccessSelectable != 0 {
			return selectionType // choosing its rows
		}
	}
	return nodeType
}

func (an *accessNode) role() int32 {
	r := atkRoles[an.n.Role]
	if an.n.Role == platform.RoleList && an.n.States&platform.AccessSelectable != 0 && atkRole.listBox != 0 {
		r = atkRole.listBox // choosing its items
	}
	if an.n.Role == platform.RoleTextField {
		switch {
		case an.n.States&platform.AccessPassword != 0:
			r = atkRole.password
		case an.n.States&platform.AccessMultiline != 0:
			r = atkRole.text
		}
	}
	if r == 0 {
		r = atkRole.panel
	}
	return r
}

func (t *accessTree) update(tree *platform.AccessTree) {
	notify := !t.quiet
	old := t.nodes
	t.nodes = make(map[uint64]*accessNode, len(tree.Nodes))
	list := make([]*accessNode, len(tree.Nodes))
	prev := make([]platform.AccessNode, len(tree.Nodes))
	fresh := make([]bool, len(tree.Nodes))
	oldKids := map[*accessNode][]*accessNode{nil: t.top}
	for i, n := range tree.Nodes {
		typ := accessType(n)
		an := old[n.ID]
		if an != nil && an.typ == typ {
			delete(old, n.ID)
			oldKids[an] = an.children
			prev[i] = an.n
		} else {
			an = &accessNode{obj: gObjectNew(typ, 0), typ: typ, tree: t}
			accessObjects[an.obj] = an
			fresh[i] = true
		}
		an.n, an.children = n, nil
		t.nodes[n.ID] = an
		list[i] = an
	}
	t.top = nil
	for i, an := range list {
		parent, parentObj := (*accessNode)(nil), t.root
		if an.n.Parent >= 0 {
			parent = list[an.n.Parent]
			parentObj = parent.obj
			parent.children = append(parent.children, an)
		} else {
			t.top = append(t.top, an)
		}
		if fresh[i] || an.parent != parent {
			atkObjectSetParent(an.obj, parentObj)
		}
		an.parent = parent
		if roleStates := platform.AccessPassword | platform.AccessMultiline | platform.AccessSelectable | platform.AccessMultiselectable; fresh[i] || prev[i].Role != an.n.Role || prev[i].States&roleStates != an.n.States&roleStates {
			atkObjectSetRole(an.obj, an.role())
		}
		if fresh[i] || prev[i].Label != an.n.Label {
			atkObjectSetName(an.obj, cs(an.n.Label))
		}
		if fresh[i] || prev[i].Description != an.n.Description {
			atkObjectSetDescription(an.obj, cs(an.n.Description))
		}
	}
	if notify {
		// Children that left their parent, then those that joined it, for
		// the parents that were in the tree before.
		for parent, kids := range oldKids {
			obj, now := t.root, t.top
			if parent != nil {
				obj, now = parent.obj, parent.children
			}
			for i := len(kids) - 1; i >= 0; i-- {
				if !slices.Contains(now, kids[i]) {
					gSignalEmitChild(obj, cs("children-changed::remove"), uint32(i), kids[i].obj)
				}
			}
			for i, kid := range now {
				if !slices.Contains(kids, kid) {
					gSignalEmitChild(obj, cs("children-changed::add"), uint32(i), kid.obj)
				}
			}
		}
		for i, an := range list {
			if !fresh[i] {
				an.notifyChanges(prev[i])
			}
		}
	}
	// Lists choosing their rows tell when the choice moves.
	for _, an := range list {
		if an.typ != selectionType {
			continue
		}
		var chosen []uint64
		for _, r := range an.chosenRows() {
			chosen = append(chosen, r.n.ID)
		}
		if !slices.Equal(chosen, an.chosen) {
			an.chosen = chosen
			if notify {
				gSignalEmit(an.obj, cs("selection-changed"))
			}
		}
	}
	t.graft(list)
	for _, an := range old {
		an.release()
	}
	if focus := tree.Focus; focus != t.focus {
		if an := t.nodes[t.focus]; an != nil && notify {
			atkObjectNotifyStateChange(an.obj, uint64(atkStates.focused), false)
		}
		t.focus = focus
		if an := t.nodes[focus]; an != nil && notify {
			atkObjectNotifyStateChange(an.obj, uint64(atkStates.focused), true)
		}
	}
	if announces && notify {
		for _, text := range tree.Announcements {
			gSignalEmitAnnouncement(t.root, cs("announcement"), cs(text))
		}
	}
}

// notifyChanges tells assistive technology how a node changed since the
// last tree, but for its focus.
func (an *accessNode) notifyChanges(prev platform.AccessNode) {
	n := an.n
	before, after := stateList(prev, false), stateList(n, false)
	for _, s := range before {
		if !slices.Contains(after, s) {
			atkObjectNotifyStateChange(an.obj, uint64(s), false)
		}
	}
	for _, s := range after {
		if !slices.Contains(before, s) {
			atkObjectNotifyStateChange(an.obj, uint64(s), true)
		}
	}
	switch an.typ {
	case rangeType:
		if prev.Now != n.Now || prev.Min != n.Min || prev.Max != n.Max {
			gObjectNotify(an.obj, cs("accessible-value"))
		}
	case editableType:
		if prev.Value != n.Value {
			// What changed, between the text both have at their ends.
			a, b := []rune(prev.Value), []rune(n.Value)
			start := 0
			for start < len(a) && start < len(b) && a[start] == b[start] {
				start++
			}
			end := 0
			for end < len(a)-start && end < len(b)-start && a[len(a)-1-end] == b[len(b)-1-end] {
				end++
			}
			if removed := a[start : len(a)-end]; len(removed) > 0 {
				gSignalEmitText(an.obj, cs("text-remove"), int32(start), int32(len(removed)), cs(string(removed)))
			}
			if inserted := b[start : len(b)-end]; len(inserted) > 0 {
				gSignalEmitText(an.obj, cs("text-insert"), int32(start), int32(len(inserted)), cs(string(inserted)))
			}
		}
		if prev.SelEnd != n.SelEnd {
			gSignalEmitInt(an.obj, cs("text-caret-moved"), int32(n.SelEnd))
		}
		if prev.SelStart != n.SelStart || prev.SelEnd != n.SelEnd {
			if prev.SelStart != prev.SelEnd || n.SelStart != n.SelEnd {
				gSignalEmit(an.obj, cs("text-selection-changed"))
			}
		}
	}
}

// stateList returns the ATK states of a node.
func stateList(n platform.AccessNode, focused bool) []int32 {
	st := &atkStates
	list := []int32{st.visible}
	if n.States&platform.AccessOffscreen == 0 {
		list = append(list, st.showing)
	}
	if n.States&platform.AccessDisabled == 0 {
		list = append(list, st.enabled, st.sensitive)
	}
	if n.States&platform.AccessFocusable != 0 {
		list = append(list, st.focusable)
	}
	if focused {
		list = append(list, st.focused)
	}
	switch n.Role {
	case platform.RoleCheckBox, platform.RoleRadio, platform.RoleSwitch, platform.RoleMenuItemCheckBox, platform.RoleMenuItemRadio:
		list = append(list, st.checkable)
	case platform.RoleTab, platform.RoleTreeItem, platform.RoleRow, platform.RoleListItem:
		// Rows and items of lists only where their list chooses them.
		if n.States&platform.AccessSelectable == 0 && (n.Role == platform.RoleRow || n.Role == platform.RoleListItem) {
			break
		}
		list = append(list, st.selectable)
		if n.States&platform.AccessChecked != 0 {
			list = append(list, st.selected) // not checked
		}
	case platform.RolePopUpButton:
		list = append(list, st.expandable)
	case platform.RoleSlider:
		// As GtkScale's.
		if n.States&platform.AccessVertical != 0 {
			list = append(list, st.vertical)
		} else {
			list = append(list, st.horizontal)
		}
	case platform.RoleDisclosure:
		// Checked while open, as GTK's expanders.
		list = append(list, st.expandable)
		if n.States&platform.AccessExpanded != 0 {
			list = append(list, st.checked)
		}
	case platform.RoleMenuButton:
		list = append(list, st.hasPopup)
	case platform.RoleTextField, platform.RoleComboBox:
		if n.Role == platform.RoleComboBox {
			list = append(list, st.expandable)
		}
		list = append(list, st.selectableText)
		if n.States&platform.AccessReadOnly == 0 {
			list = append(list, st.editable)
		} else {
			list = append(list, st.readOnly)
		}
		if n.States&platform.AccessMultiline != 0 {
			list = append(list, st.multiLine)
		} else {
			list = append(list, st.singleLine)
		}
	}
	// A toggle button is checked while pressed, as GTK's are.
	if n.States&platform.AccessChecked != 0 && n.Role != platform.RoleTab && n.Role != platform.RoleTreeItem && n.Role != platform.RoleRow && n.Role != platform.RoleListItem {
		list = append(list, st.checked)
	}
	if n.States&platform.AccessMixed != 0 || n.Now < n.Min {
		list = append(list, st.indeterminate) // a progress of unknown length
	}
	if n.States&platform.AccessExpandable != 0 {
		list = append(list, st.expandable)
	}
	if n.States&platform.AccessExpanded != 0 {
		list = append(list, st.expanded)
	}
	if n.States&platform.AccessMultiselectable != 0 {
		list = append(list, st.multiselectable)
	}
	if n.States&platform.AccessInvalid != 0 {
		list = append(list, st.invalidEntry)
	}
	return slices.DeleteFunc(list, func(s int32) bool { return s == 0 })
}

// act performs an action of assistive technology on the node.
func (an *accessNode) act(action platform.AccessActionKind, text string) {
	an.tree.s.send(platform.SurfaceEvent{Kind: platform.AccessAction, ID: an.n.ID, Action: action, Text: text})
}

// extents returns the node's bounds in ATK's coordinates.
func (an *accessNode) extents(coords int32) (x, y, w, h int32) {
	b := an.n.Bounds
	if coords == atkXYParent {
		if an.parent != nil {
			b.X, b.Y = b.X-an.parent.n.Bounds.X, b.Y-an.parent.n.Bounds.Y
		}
	} else {
		var rx, ry, rw, rh int32
		atkComponentGetExtents(an.tree.root, &rx, &ry, &rw, &rh, coords)
		if rx == math.MinInt32 {
			return rx, rx, int32(b.W), int32(b.H) // not on the screen
		}
		b.X, b.Y = b.X+float64(rx), b.Y+float64(ry)
	}
	return int32(math.Floor(b.X)), int32(math.Floor(b.Y)), int32(math.Round(b.W)), int32(math.Round(b.H))
}

func (an *accessNode) contains(x, y, coords int32) bool {
	nx, ny, w, h := an.extents(coords)
	return x >= nx && y >= ny && x < nx+w && y < ny+h
}

// at returns a new reference to the topmost of nodes at a point, or 0.
func at(nodes []*accessNode, x, y, coords int32) ptr {
	for i := len(nodes) - 1; i >= 0; i-- {
		if nodes[i].contains(x, y, coords) {
			return gObjectRef(nodes[i].obj)
		}
	}
	return 0
}

// text returns the text of a node that ATK reads as text.
func (an *accessNode) text() []rune {
	if an.typ == textType {
		return []rune(an.n.Label)
	}
	return []rune(an.n.Value)
}

// span clamps a range of runes of ATK, whose end is -1 for the end.
func span(start, end int32, n int) (int, int) {
	if end < 0 || int(end) > n {
		end = int32(n)
	}
	a := max(0, min(int(start), n))
	return a, max(a, int(end))
}

// segment returns the range of the character, word, or line, for the
// granularities of ATK 0, 1, and above, at a rune offset.
func segment(r []rune, offset, granularity int) (int, int) {
	offset = max(0, min(offset, len(r)))
	switch granularity {
	case 0:
		return offset, min(offset+1, len(r))
	case 1:
		start := offset
		for start > 0 && unicode.IsSpace(r[start-1]) && (start == len(r) || unicode.IsSpace(r[start])) {
			start-- // between words: the one before
		}
		for start > 0 && !unicode.IsSpace(r[start-1]) {
			start--
		}
		end := start
		for end < len(r) && !unicode.IsSpace(r[end]) {
			end++
		}
		for end < len(r) && unicode.IsSpace(r[end]) {
			end++
		}
		return start, end
	}
	start, end := offset, offset
	for start > 0 && r[start-1] != '\n' {
		start--
	}
	for end < len(r) && r[end] != '\n' {
		end++
	}
	return start, min(end+1, len(r))
}

func cString(s string) ptr { return gStrdup(cs(s)) }

func setInt(p ptr, v int) {
	if p != 0 {
		*(*int32)(unsafe.Pointer(slot(p, 0))) = int32(v)
	}
}

func setDouble(gvalue ptr, v float64) {
	gValueInit(gvalue, gTypeDouble)
	gValueSetDouble(gvalue, v)
}

// cBytes copies length bytes of a C string, or all of it for -1.
func cBytes(p ptr, length int32) string {
	if length < 0 || p == 0 {
		return goStr(p)
	}
	return string(unsafe.Slice((*byte)(unsafe.Pointer(slot(p, 0))), length))
}

func initAccessCallbacks() {
	node := func(obj ptr) *accessNode { return accessObjects[obj] }
	cbAreaClassInit = purego.NewCallback(func(class, data ptr) {
		gtkWidgetClassSetAccessibleType(class, rootType)
		gtkWidgetClassSetAccessibleRole(class, atkRole.panel)
	})
	rootChildren := purego.NewCallback(func(root ptr) int32 {
		if t := rootTree(root); t != nil {
			return int32(len(t.top))
		}
		return 0
	})
	rootChild := purego.NewCallback(func(root ptr, i int32) ptr {
		if t := rootTree(root); t != nil && i >= 0 && int(i) < len(t.top) {
			return gObjectRef(t.top[i].obj)
		}
		return 0
	})
	// In the overlay holding web views, the surface is its only child.
	rootIndex := purego.NewCallback(func(root ptr) int32 {
		if s := surfaceAreas[gtkAccessibleGetWidget(root)]; s != nil && s.layers != 0 && overlayType != 0 {
			return 0
		}
		r, _, _ := purego.SyscallN(rootParentIndex, root)
		return int32(r)
	})
	cbRootClassInit = purego.NewCallback(func(class, data ptr) {
		*slot(class, atkGetNChildren) = rootChildren
		*slot(class, atkRefChild) = rootChild
		rootParentIndex = *slot(class, atkGetIndexInParent)
		*slot(class, atkGetIndexInParent) = rootIndex
	})
	overlayChildren := purego.NewCallback(func(obj ptr) int32 {
		if layersOf(gtkAccessibleGetWidget(obj)) != nil {
			return 1
		}
		return 0
	})
	overlayChild := purego.NewCallback(func(obj ptr, i int32) ptr {
		if s := layersOf(gtkAccessibleGetWidget(obj)); s != nil && i == 0 {
			return gObjectRef(gtkWidgetGetAccessible(s.area))
		}
		return 0
	})
	cbOverlayAccessClassInit = purego.NewCallback(func(class, data ptr) {
		*slot(class, atkGetNChildren) = overlayChildren
		*slot(class, atkRefChild) = overlayChild
	})
	cbOverlayClassInit = purego.NewCallback(func(class, data ptr) {
		gtkWidgetClassSetAccessibleType(class, overlayAccessType)
	})
	rootAt := purego.NewCallback(func(root ptr, x, y, coords int32) ptr {
		if t := rootTree(root); t != nil {
			return at(t.top, x, y, coords)
		}
		return 0
	})
	cbRootComponent = purego.NewCallback(func(iface, data ptr) {
		setIface(iface, map[int]ptr{2: rootAt}) // ref_accessible_at_point; the rest is GtkWidgetAccessible's
	})

	children := purego.NewCallback(func(obj ptr) int32 {
		if an := node(obj); an != nil && an.page != 0 {
			return 1
		} else if an != nil {
			return int32(len(an.children))
		}
		return 0
	})
	child := purego.NewCallback(func(obj ptr, i int32) ptr {
		if an := node(obj); an != nil && an.page != 0 && i == 0 {
			return gObjectRef(an.page)
		}
		if an := node(obj); an != nil && i >= 0 && int(i) < len(an.children) {
			return gObjectRef(an.children[i].obj)
		}
		return 0
	})
	index := purego.NewCallback(func(obj ptr) int32 {
		an := node(obj)
		if an == nil {
			return -1
		}
		siblings := an.tree.top
		if an.parent != nil {
			siblings = an.parent.children
		}
		return int32(slices.Index(siblings, an))
	})
	states := purego.NewCallback(func(obj ptr) ptr {
		set := atkStateSetNew()
		an := node(obj)
		if an == nil {
			atkStateSetAddState(set, atkStates.defunct)
			return set
		}
		for _, s := range stateList(an.n, an.n.ID == an.tree.focus) {
			atkStateSetAddState(set, s)
		}
		return set
	})
	// Object attributes: the placeholder of a text field, and the place of
	// a row among its list's, which Orca reads as "5 of 10000".
	attributes := purego.NewCallback(func(obj ptr) ptr {
		an := node(obj)
		if an == nil {
			return 0
		}
		var list ptr
		add := func(name, value string) {
			attr := gMalloc0(16) // AtkAttribute: name, value
			*slot(attr, 0), *slot(attr, 8) = cString(name), cString(value)
			list = gSlistAppend(list, attr)
		}
		if an.n.Placeholder != "" {
			add("placeholder-text", an.n.Placeholder)
		}
		if an.n.PosInSet > 0 {
			add("posinset", strconv.Itoa(an.n.PosInSet))
		}
		if an.n.SetSize > 0 {
			add("setsize", strconv.Itoa(an.n.SetSize))
		}
		if an.n.Level > 0 {
			add("level", strconv.Itoa(an.n.Level))
		}
		// The order of the column a table is sorted by, as Chromium says.
		switch {
		case an.n.States&platform.AccessSortAscending != 0:
			add("sort", "ascending")
		case an.n.States&platform.AccessSortDescending != 0:
			add("sort", "descending")
		}
		return list
	})
	cbNodeClassInit = purego.NewCallback(func(class, data ptr) {
		*slot(class, atkGetNChildren) = children
		*slot(class, atkRefChild) = child
		*slot(class, atkGetIndexInParent) = index
		*slot(class, atkRefStateSet) = states
		*slot(class, atkGetAttributes) = attributes
	})

	contains := purego.NewCallback(func(obj ptr, x, y, coords int32) bool {
		an := node(obj)
		return an != nil && an.contains(x, y, coords)
	})
	nodeAt := purego.NewCallback(func(obj ptr, x, y, coords int32) ptr {
		if an := node(obj); an != nil && an.page != 0 {
			return gObjectRef(an.page)
		}
		if an := node(obj); an != nil {
			return at(an.children, x, y, coords)
		}
		return 0
	})
	extents := purego.NewCallback(func(obj, px, py, pw, ph ptr, coords int32) {
		if an := node(obj); an != nil {
			x, y, w, h := an.extents(coords)
			setInt(px, int(x))
			setInt(py, int(y))
			setInt(pw, int(w))
			setInt(ph, int(h))
		}
	})
	grabFocus := purego.NewCallback(func(obj ptr) bool {
		an := node(obj)
		if an == nil || an.n.Actions&platform.ActionFocus == 0 {
			return false
		}
		gtkWidgetGrabFocus(an.tree.s.area)
		an.act(platform.AccessFocus, "")
		return true
	})
	// What assistive technology moves to out of view scrolls into view.
	scrollTo := purego.NewCallback(func(obj ptr, how int32) bool {
		an := node(obj)
		if an == nil || an.n.Actions&platform.ActionScrollIntoView == 0 {
			return false
		}
		an.act(platform.AccessScrollIntoView, "")
		return true
	})
	// scroll_to arrived in ATK 2.30, past the end of the interface before.
	scrolls := atkGetMajorVersion() > 2 || atkGetMinorVersion() >= 30
	cbComponentInit = purego.NewCallback(func(iface, data ptr) {
		fns := map[int]ptr{1: contains, 2: nodeAt, 3: extents, 6: grabFocus}
		if scrolls {
			fns[15] = scrollTo
		}
		setIface(iface, fns)
	})

	// The actions of a node: click, then expand or contract, for an item
	// of a tree with children, as GTK's tree views have.
	actions := func(obj ptr) (*accessNode, []platform.AccessActions) {
		an := node(obj)
		if an == nil {
			return nil, nil
		}
		var list []platform.AccessActions
		for _, a := range []platform.AccessActions{platform.ActionPress, platform.ActionExpand} {
			if an.n.Actions&a != 0 {
				list = append(list, a)
			}
		}
		return an, list
	}
	action := func(obj ptr, i int32) (*accessNode, platform.AccessActions) {
		an, list := actions(obj)
		if i < 0 || int(i) >= len(list) {
			return nil, 0
		}
		return an, list[i]
	}
	doAction := purego.NewCallback(func(obj ptr, i int32) bool {
		an, a := action(obj, i)
		switch {
		case a == platform.ActionPress:
			an.act(platform.AccessPress, "")
		case a == platform.ActionExpand && an.n.States&platform.AccessExpanded != 0:
			an.act(platform.AccessCollapse, "")
		case a == platform.ActionExpand:
			an.act(platform.AccessExpand, "")
		}
		return an != nil
	})
	nActions := purego.NewCallback(func(obj ptr) int32 {
		_, list := actions(obj)
		return int32(len(list))
	})
	actionName := purego.NewCallback(func(obj ptr, i int32) ptr {
		switch _, a := action(obj, i); a {
		case platform.ActionPress:
			return ptr(unsafe.Pointer(&clickName[0]))
		case platform.ActionExpand:
			return ptr(unsafe.Pointer(&expandName[0]))
		}
		return 0
	})
	actionEmpty := purego.NewCallback(func(obj ptr, i int32) ptr {
		if an, _ := action(obj, i); an != nil {
			return ptr(unsafe.Pointer(&emptyName[0]))
		}
		return 0
	})
	cbActionInit = purego.NewCallback(func(iface, data ptr) {
		// do_action, get_n_actions, get_description, get_name,
		// get_keybinding and get_localized_name.
		setIface(iface, map[int]ptr{0: doAction, 1: nActions, 2: actionEmpty, 3: actionName, 4: actionEmpty, 6: actionName})
	})

	valueOf := func(pick func(n platform.AccessNode) float64) ptr {
		return purego.NewCallback(func(obj, gvalue ptr) {
			if an := node(obj); an != nil {
				setDouble(gvalue, pick(an.n))
			}
		})
	}
	current := valueOf(func(n platform.AccessNode) float64 { return max(n.Now, n.Min) })
	maximum := valueOf(func(n platform.AccessNode) float64 { return n.Max })
	minimum := valueOf(func(n platform.AccessNode) float64 { return n.Min })
	// The step, through the GValue of get_minimum_increment: callbacks
	// return no floats for get_increment, which AT-SPI's bridge then
	// leaves for this.
	increment := valueOf(func(n platform.AccessNode) float64 { return n.Step })
	valueAndText := purego.NewCallback(func(obj, value, text ptr) {
		if an := node(obj); an != nil && value != 0 {
			*(*float64)(unsafe.Pointer(slot(value, 0))) = max(an.n.Now, an.n.Min)
		}
		if text != 0 {
			*slot(text, 0) = 0
		}
	})
	valueRange := purego.NewCallback(func(obj ptr) ptr {
		if an := node(obj); an != nil {
			return atkRangeNew(an.n.Min, an.n.Max, nil)
		}
		return 0
	})
	// Values change as the arrow keys change them.
	setValue := purego.NewCallback(func(obj ptr, v float64) {
		an := node(obj)
		switch {
		case an == nil || an.n.Actions&platform.ActionIncrement == 0:
		case v > an.n.Now:
			an.act(platform.AccessIncrement, "")
		case v < an.n.Now:
			an.act(platform.AccessDecrement, "")
		}
	})
	cbValueInit = purego.NewCallback(func(iface, data ptr) {
		// get_current_value, get_maximum_value, get_minimum_value,
		// get_minimum_increment, get_value_and_text, get_range and
		// set_value.
		setIface(iface, map[int]ptr{0: current, 1: maximum, 2: minimum, 4: increment, 5: valueAndText, 6: valueRange, 9: setValue})
	})

	getText := purego.NewCallback(func(obj ptr, start, end int32) ptr {
		an := node(obj)
		if an == nil {
			return 0
		}
		r := an.text()
		a, b := span(start, end, len(r))
		return cString(string(r[a:b]))
	})
	charAt := purego.NewCallback(func(obj ptr, offset int32) uint32 {
		if an := node(obj); an != nil {
			if r := an.text(); offset >= 0 && int(offset) < len(r) {
				return uint32(r[offset])
			}
		}
		return 0
	})
	caret := purego.NewCallback(func(obj ptr) int32 {
		if an := node(obj); an != nil && an.typ == editableType {
			return int32(an.n.SelEnd)
		}
		return -1
	})
	count := purego.NewCallback(func(obj ptr) int32 {
		if an := node(obj); an != nil {
			return int32(len(an.text()))
		}
		return 0
	})
	selections := purego.NewCallback(func(obj ptr) int32 {
		if an := node(obj); an != nil && an.n.SelStart != an.n.SelEnd {
			return 1
		}
		return 0
	})
	selection := purego.NewCallback(func(obj ptr, i int32, start, end ptr) ptr {
		an := node(obj)
		if an == nil || i != 0 || an.n.SelStart == an.n.SelEnd {
			setInt(start, 0)
			setInt(end, 0)
			return 0
		}
		r := an.text()
		a, b := span(int32(an.n.SelStart), int32(an.n.SelEnd), len(r))
		setInt(start, a)
		setInt(end, b)
		return cString(string(r[a:b]))
	})
	stringAt := func(obj ptr, offset int32, granularity int, start, end ptr) ptr {
		an := node(obj)
		if an == nil {
			return 0
		}
		r := an.text()
		a, b := segment(r, int(offset), granularity)
		setInt(start, a)
		setInt(end, b)
		return cString(string(r[a:b]))
	}
	stringAtOffset := purego.NewCallback(func(obj ptr, offset, granularity int32, start, end ptr) ptr {
		return stringAt(obj, offset, int(granularity), start, end)
	})
	// Boundaries of the older text_at_offset: characters, the starts and
	// ends of words, sentences and lines.
	textAtOffset := purego.NewCallback(func(obj ptr, offset, boundary int32, start, end ptr) ptr {
		return stringAt(obj, offset, int([]int32{0, 1, 1, 3, 3, 3, 3}[max(0, min(boundary, 6))]), start, end)
	})
	// The formats of a rich text's runs, in ATK's text attributes, which
	// Orca reads as bold, italic, underlined.
	runAttributes := purego.NewCallback(func(obj ptr, offset int32, start, end ptr) ptr {
		an := node(obj)
		if an == nil {
			setInt(start, 0)
			setInt(end, 0)
			return 0
		}
		run, a, z := textRunAt(an.n.TextRuns, int(offset), len(an.text()))
		setInt(start, a)
		setInt(end, z)
		return textAttributeSet(run)
	})
	defaultAttributes := purego.NewCallback(func(obj ptr) ptr {
		return textAttributeSet(platform.AccessTextRun{})
	})
	cbTextInit = purego.NewCallback(func(iface, data ptr) {
		// get_text, get_text_at_offset, get_character_at_offset,
		// get_caret_offset, get_run_attributes, get_default_attributes,
		// get_character_count, get_n_selections, get_selection and
		// get_string_at_offset.
		setIface(iface, map[int]ptr{0: getText, 2: textAtOffset, 3: charAt, 5: caret, 6: runAttributes, 7: defaultAttributes,
			9: count, 11: selections, 12: selection, 23: stringAtOffset})
	})

	// The rows a list chooses, which choosing a row as a click does
	// changes: it chooses that row alone.
	childAt := func(obj ptr, i int32) *accessNode {
		if an := node(obj); an != nil && i >= 0 && int(i) < len(an.children) {
			return an.children[i]
		}
		return nil
	}
	addSelection := purego.NewCallback(func(obj ptr, i int32) bool {
		c := childAt(obj, i)
		if c == nil || !chooses(c.n) || c.n.Actions&platform.ActionPress == 0 {
			return false
		}
		c.act(platform.AccessPress, "")
		return true
	})
	refuse := purego.NewCallback(func(obj ptr) bool { return false })
	refuseAt := purego.NewCallback(func(obj ptr, i int32) bool { return false })
	refSelection := purego.NewCallback(func(obj ptr, i int32) ptr {
		if an := node(obj); an != nil {
			if rows := an.chosenRows(); i >= 0 && int(i) < len(rows) {
				return gObjectRef(rows[i].obj)
			}
		}
		return 0
	})
	selectionCount := purego.NewCallback(func(obj ptr) int32 {
		if an := node(obj); an != nil {
			return int32(len(an.chosenRows()))
		}
		return 0
	})
	isSelected := purego.NewCallback(func(obj ptr, i int32) bool {
		c := childAt(obj, i)
		return c != nil && chooses(c.n) && c.n.States&platform.AccessChecked != 0
	})
	cbSelectionInit = purego.NewCallback(func(iface, data ptr) {
		// add_selection, clear_selection, ref_selection,
		// get_selection_count, is_child_selected, remove_selection and
		// select_all_selection.
		setIface(iface, map[int]ptr{0: addSelection, 1: refuse, 2: refSelection, 3: selectionCount, 4: isSelected,
			5: refuseAt, 6: refuse})
	})

	editable := func(obj ptr) *accessNode {
		if an := node(obj); an != nil && an.n.Actions&platform.ActionSetValue != 0 {
			return an
		}
		return nil
	}
	setContents := purego.NewCallback(func(obj, text ptr) {
		if an := editable(obj); an != nil {
			an.act(platform.AccessSetValue, goStr(text))
		}
	})
	insert := purego.NewCallback(func(obj, text ptr, length int32, position ptr) {
		an := editable(obj)
		if an == nil || position == 0 {
			return
		}
		s, r := cBytes(text, length), an.text()
		at := max(0, min(int(*(*int32)(unsafe.Pointer(slot(position, 0)))), len(r)))
		an.act(platform.AccessSetValue, string(r[:at])+s+string(r[at:]))
		setInt(position, at+utf8.RuneCountInString(s))
	})
	remove := purego.NewCallback(func(obj ptr, start, end int32) {
		if an := editable(obj); an != nil {
			r := an.text()
			a, b := span(start, end, len(r))
			an.act(platform.AccessSetValue, string(r[:a])+string(r[b:]))
		}
	})
	cbEditableInit = purego.NewCallback(func(iface, data ptr) {
		// set_text_contents, insert_text and delete_text.
		setIface(iface, map[int]ptr{1: setContents, 2: insert, 5: remove})
	})
}

// graft puts the pages of the surface's web views in the nodes of list
// standing for them, and takes those of the others out.
func (t *accessTree) graft(list []*accessNode) {
	grafted := map[*window]bool{}
	for _, an := range list {
		an.page = 0
		v, ok := an.n.WebView.(*window)
		if !ok || grafted[v] || v.closed || v.host == nil || v.host.surface != t.s {
			continue
		}
		if acc := v.pageAccessible(); acc != 0 {
			grafted[v], an.page = true, acc
			if v.accessParent != an.obj {
				v.accessParent = an.obj
				atkObjectSetParent(acc, an.obj)
			}
		}
	}
	for _, v := range t.s.w.webViews {
		if !grafted[v] && v.accessParent != 0 {
			v.accessParent = 0
			if acc := v.pageAccessible(); acc != 0 {
				atkObjectSetParent(acc, 0)
			}
		}
	}
}

// textRunAt returns the run of runs holding rune i of a text of n runes,
// and where it starts and ends: between runs, a run without formats
// over the runes between them.
func textRunAt(runs []platform.AccessTextRun, i, n int) (platform.AccessTextRun, int, int) {
	start := 0
	for _, r := range runs {
		switch {
		case i < r.Start:
			return platform.AccessTextRun{}, start, r.Start
		case i < r.End:
			return r, r.Start, r.End
		}
		start = r.End
	}
	return platform.AccessTextRun{}, min(start, n), n
}

// textAttributeSet returns the formats of a run as an AtkAttributeSet,
// which ATK frees.
func textAttributeSet(r platform.AccessTextRun) ptr {
	var list ptr
	add := func(name, value string) {
		attr := gMalloc0(16) // AtkAttribute: name, value
		*slot(attr, 0), *slot(attr, 8) = cString(name), cString(value)
		list = gSlistAppend(list, attr)
	}
	if r.Family != "" {
		add("family-name", r.Family)
	}
	if r.Size > 0 {
		add("size", strconv.FormatFloat(float64(r.Size)*0.75, 'g', 4, 32)) // points
	}
	weight := r.Weight
	if weight <= 0 {
		weight = 400
	}
	add("weight", strconv.Itoa(weight))
	if r.Italic {
		add("style", "italic")
	} else {
		add("style", "normal")
	}
	if r.Underline {
		add("underline", "single")
	} else {
		add("underline", "none")
	}
	add("strikethrough", strconv.FormatBool(r.Strikethrough))
	color := func(c [4]uint8) string {
		return strconv.Itoa(int(c[0])*257) + "," + strconv.Itoa(int(c[1])*257) + "," + strconv.Itoa(int(c[2])*257)
	}
	if r.Color[3] > 0 {
		add("fg-color", color(r.Color))
	}
	if r.Background[3] > 0 {
		add("bg-color", color(r.Background))
	}
	return list
}
