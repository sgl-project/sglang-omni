// Package listsel is the row selection of the ActNow apps' log views (and
// any list of lines to copy): which rows are selected, the anchor a
// Shift-click extends from, and the pointer gesture that tells a click
// from a drag. It has no UI: the app reports presses, moves and releases
// with the key of the row under the pointer and draws what the Model says.
//
// The rules, the same in every app:
//
//   - A plain click (no modifier, no drag) copies its row (Intent CopyRow),
//     clears the selection and makes the row the anchor, without showing
//     it selected.
//   - Shift-click selects the rows from the anchor to the row clicked, in
//     display order, added to what was selected before the anchor was set.
//   - Toggle-click (⌘ on macOS, Ctrl elsewhere) adds the row or takes it
//     out, and makes it the anchor.
//   - Pressing and moving the pointer more than DragSlop before releasing
//     selects the rows from the one pressed to the one under the pointer,
//     up or down; with Toggle held it adds them to the selection.
//   - SelectAll (⌘A) selects every row shown, Clear (Esc, and after a
//     copy) empties the selection and keeps the anchor.
//
// Rows are named by stable keys (a log line's sequence number, an id), not
// by their place in the list: new lines and other filters do not move the
// selection to other rows. Functions taking order want the keys of the rows
// shown, in display order; Prune drops the keys no longer shown.
//
// A Model is not safe for concurrent use: keep it with the view's state,
// on the UI thread. Its zero value is an empty selection.
package listsel

// Intent is what a gesture asks the app to do besides selecting.
type Intent int

const (
	// None: nothing to do; redraw the selection.
	None Intent = iota
	// CopyRow: copy the row clicked (a plain click), then show that it
	// was copied.
	CopyRow
)

// Mods are the modifier keys held as a row was pressed, as the UI maps
// them: Extend is Shift; Toggle is ⌘ (Command) on macOS and Ctrl
// elsewhere. Control-click on macOS is a secondary click (the context
// menu) and never reaches the Model.
type Mods uint8

const (
	Extend Mods = 1 << iota
	Toggle
)

// DragSlop is how far, in DIPs (points), the pointer moves away from where
// it pressed before the press is a drag that selects rather than a click
// that copies.
const DragSlop float32 = 4

// Model is a selection of rows by key.
type Model[K comparable] struct {
	sel map[K]struct{}
	// base is what a range (Shift-click, drag) is added to: the selection
	// as the anchor was last set by a click, a toggle or a drag's start.
	base      map[K]struct{}
	anchor    K
	hasAnchor bool

	press   press[K]
	drag    bool
	dragKey K              // the row the drag started on
	before  map[K]struct{} // the selection as the drag started, for Cancel
}

type press[K comparable] struct {
	on   bool
	key  K
	x, y float32
	mods Mods
}

// Count returns how many rows are selected.
func (m *Model[K]) Count() int { return len(m.sel) }

// Empty reports whether no row is selected.
func (m *Model[K]) Empty() bool { return len(m.sel) == 0 }

// IsSelected reports whether the row k is selected.
func (m *Model[K]) IsSelected(k K) bool {
	_, ok := m.sel[k]
	return ok
}

// Anchor returns the row a Shift-click extends from, if any.
func (m *Model[K]) Anchor() (K, bool) { return m.anchor, m.hasAnchor }

// Dragging reports whether a drag is selecting rows: the pointer pressed a
// row and moved more than DragSlop, and has not been released.
func (m *Model[K]) Dragging() bool { return m.drag }

// Pressed reports whether a press is in progress (a click or a drag to be).
func (m *Model[K]) Pressed() bool { return m.press.on }

// Selected returns the selected rows of order, in its order: the rows
// shown, as they are shown. Selected keys missing from order are left out.
func (m *Model[K]) Selected(order []K) []K {
	if len(m.sel) == 0 {
		return nil
	}
	out := make([]K, 0, len(m.sel))
	for _, k := range order {
		if _, ok := m.sel[k]; ok {
			out = append(out, k)
		}
	}
	return out
}

// Click is a plain click on row k: the selection empties, k becomes the
// anchor, and the app copies the row (CopyRow).
func (m *Model[K]) Click(k K) Intent {
	m.sel, m.base = nil, nil
	m.setAnchor(k)
	return CopyRow
}

// ShiftClick is a Shift-click on row k: the rows from the anchor to k, in
// order, are selected, besides what was selected as the anchor was set.
// Without an anchor shown in order, k alone is selected and becomes the
// anchor.
func (m *Model[K]) ShiftClick(k K, order []K) {
	if !m.hasAnchor || index(order, m.anchor) < 0 || index(order, k) < 0 {
		m.sel, m.base = set(k), nil
		m.setAnchor(k)
		return
	}
	m.sel = union(m.base, span(order, m.anchor, k))
}

// ToggleClick is a ⌘-click on row k: k is added to the selection or taken
// out of it, and becomes the anchor.
func (m *Model[K]) ToggleClick(k K) {
	if _, ok := m.sel[k]; ok {
		delete(m.sel, k)
	} else {
		if m.sel == nil {
			m.sel = map[K]struct{}{}
		}
		m.sel[k] = struct{}{}
	}
	m.base = clone(m.sel)
	m.setAnchor(k)
}

// ClickWith is a click on row k with mods: Extend wins over Toggle, as in
// Finder; no modifier is Click.
func (m *Model[K]) ClickWith(k K, mods Mods, order []K) Intent {
	switch {
	case mods&Extend != 0:
		m.ShiftClick(k, order)
	case mods&Toggle != 0:
		m.ToggleClick(k)
	default:
		return m.Click(k)
	}
	return None
}

// DragStart starts selecting by dragging from row k, which becomes the
// anchor and is selected; add keeps the selection and adds the rows
// dragged over to it (⌘ held), else they replace it.
func (m *Model[K]) DragStart(k K, add bool) {
	if add {
		m.base = clone(m.sel)
	} else {
		m.base = nil
	}
	m.drag, m.dragKey = true, k
	m.setAnchor(k)
	m.sel = union(m.base, []K{k})
}

// DragOver is the pointer over row k while dragging: the rows from the
// drag's first row to k are selected, up or down, besides the selection
// the drag added to.
func (m *Model[K]) DragOver(k K, order []K) {
	if !m.drag {
		return
	}
	r := span(order, m.dragKey, k)
	if r == nil {
		r = []K{k}
	}
	m.sel = union(m.base, r)
}

// DragEnd ends a drag; the rows stay selected, and a Shift-click extends
// from the drag's first row.
func (m *Model[K]) DragEnd() { m.drag = false }

// SelectAll selects the rows of keys (⌘A): those shown.
func (m *Model[K]) SelectAll(keys []K) {
	m.sel = make(map[K]struct{}, len(keys))
	for _, k := range keys {
		m.sel[k] = struct{}{}
	}
	m.base = nil
	if !m.hasAnchor && len(keys) > 0 {
		m.setAnchor(keys[0])
	}
}

// Clear empties the selection (Esc, after a copy) and keeps the anchor.
func (m *Model[K]) Clear() { m.sel, m.base = nil, nil }

// Reset forgets everything: the selection, the anchor and any gesture.
func (m *Model[K]) Reset() { *m = Model[K]{} }

// Prune keeps only the rows of visible, as the filters or the lines shown
// change: the selection, the anchor and the drag's first row of rows no
// longer shown are dropped (a drag whose first row went ends).
func (m *Model[K]) Prune(visible []K) {
	if len(m.sel) == 0 && len(m.base) == 0 && !m.hasAnchor && !m.press.on && !m.drag {
		return
	}
	in := make(map[K]struct{}, len(visible))
	for _, k := range visible {
		in[k] = struct{}{}
	}
	for k := range m.sel {
		if _, ok := in[k]; !ok {
			delete(m.sel, k)
		}
	}
	for k := range m.base {
		if _, ok := in[k]; !ok {
			delete(m.base, k)
		}
	}
	if _, ok := in[m.anchor]; m.hasAnchor && !ok {
		var zero K
		m.anchor, m.hasAnchor = zero, false
	}
	if _, ok := in[m.press.key]; m.press.on && !ok {
		m.press = press[K]{}
	}
	if _, ok := in[m.dragKey]; m.drag && !ok {
		m.drag, m.before = false, nil
	}
	for k := range m.before {
		if _, ok := in[k]; !ok {
			delete(m.before, k)
		}
	}
}

func (m *Model[K]) setAnchor(k K) { m.anchor, m.hasAnchor = k, true }

// index returns where k is in order, -1 when it is not.
func index[K comparable](order []K, k K) int {
	for i, o := range order {
		if o == k {
			return i
		}
	}
	return -1
}

// span returns the keys of order from a to b, both included, whichever
// comes first; nil when either is missing.
func span[K comparable](order []K, a, b K) []K {
	i, j := index(order, a), index(order, b)
	if i < 0 || j < 0 {
		return nil
	}
	if i > j {
		i, j = j, i
	}
	return order[i : j+1]
}

func set[K comparable](keys ...K) map[K]struct{} {
	s := make(map[K]struct{}, len(keys))
	for _, k := range keys {
		s[k] = struct{}{}
	}
	return s
}

func clone[K comparable](s map[K]struct{}) map[K]struct{} {
	if len(s) == 0 {
		return nil
	}
	c := make(map[K]struct{}, len(s))
	for k := range s {
		c[k] = struct{}{}
	}
	return c
}

func union[K comparable](s map[K]struct{}, keys []K) map[K]struct{} {
	u := make(map[K]struct{}, len(s)+len(keys))
	for k := range s {
		u[k] = struct{}{}
	}
	for _, k := range keys {
		u[k] = struct{}{}
	}
	return u
}
