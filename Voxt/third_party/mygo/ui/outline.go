package ui

import (
	"runtime"
	"time"
)

// OutlineState is the state of an Outline or an OutlineTable: its rows, as
// a List's, and the items that are open, showing their children. The rows'
// keys are the items, of type K, so that their place, choice and state
// follow them as items open and close above them.
//
//	type app struct {
//		files ui.OutlineState[string] // paths
//		file  int
//	}
//
//	app.files.List.Selected = &app.file
type OutlineState[K comparable] struct {
	// List is the state of the rows: their place, and how they are chosen,
	// as a List's. Its Key is the rows' items; a Selection takes them, of
	// type K.
	List ListState
	// Open holds the items showing their children.
	Open Selection[K]
	// Indent is the width in DIPs each level adds before a row's arrow,
	// the theme's Space(4) when zero. It must not be negative.
	Indent float32
	rows   []outlineRow[K]
}

// outlineRow is a row of an outline: its item, how deep it is, the row of
// its parent (-1 for the items at the top), and whether it has children.
type outlineRow[K comparable] struct {
	item   K
	depth  int
	parent int
	branch bool
}

// Rows returns how many rows the outline shows: its items at the top, and
// the children of those open, in the last frame.
func (s *OutlineState[K]) Rows() int { return len(s.rows) }

// Item returns the item row i shows, in the last frame.
func (s *OutlineState[K]) Item(row int) K { return s.rows[row].item }

// Depth returns how deep the item of row i is: 0 for the items at the top.
func (s *OutlineState[K]) Depth(row int) int { return s.rows[row].depth }

// flatten lists the rows the outline shows: roots, and the children of
// the items open.
func (s *OutlineState[K]) flatten(roots []K, children func(K) []K) {
	if !(s.Indent >= 0) {
		panic("ui: an OutlineState's Indent must not be negative")
	}
	s.rows = s.rows[:0]
	var walk func(items []K, depth, parent int)
	walk = func(items []K, depth, parent int) {
		for _, item := range items {
			kids := children(item)
			s.rows = append(s.rows, outlineRow[K]{item: item, depth: depth, parent: parent, branch: kids != nil})
			if kids != nil && s.Open.Has(item) {
				walk(kids, depth+1, len(s.rows)-1)
			}
		}
	}
	walk(roots, 0, -1)
}

// setOpen opens or closes an item, and with all its children's children.
func (s *OutlineState[K]) setOpen(item K, open, all bool, children func(K) []K) {
	if open {
		s.Open.Add(item)
	} else {
		s.Open.Remove(item)
	}
	if all {
		for _, kid := range children(item) {
			if children(kid) != nil {
				s.setOpen(kid, open, true, children)
			}
		}
	}
}

// Outline creates a list of items in a tree, as AppKit's outline view and
// SwiftUI's List of children: roots at the top, and below each item open
// its children, indented, which children returns, nil for an item without
// any. A click on an item's arrow opens or closes it, with all its
// children's children with Option (Alt). row builds the content of an
// item's row, after its arrow; Gap, Padding and the rest are a List's.
//
// The outline builds only the rows in view, as a List does, so trees of
// any size are as fast as small ones. With s.List.Selected or Selection,
// rows are chosen as a List's, and while the outline has the focus, Right
// opens the item chosen or goes to its first child, and Left closes it or
// goes to its parent, with Option on macOS and Shift elsewhere, as in GTK,
// opening and closing all inside.
//
//	app.files.List.Selected = &app.file
//	ui.Outline(c, &app.files, []string{"/"}, func(dir string) []string {
//		return app.children[dir] // nil for files
//	}, func(path string) {
//		ui.Text(c, filepath.Base(path))
//	}).Grow(1)
func coreOutline[K comparable](c *context, s *OutlineState[K], roots []K, children func(K) []K, row func(item K)) *node {
	e := coreScroll(c)
	e.widget, e.role = "Outline", RoleTree
	s.flatten(roots, children)
	s.List.Key = func(i int) any { return s.rows[i].item }
	buildList(c, e, e, &s.List, len(s.rows), func(i int) {
		r := coreRow(c).AlignItems(Center).Gap(c.theme.Space(1)).Padding(c.theme.Space(1), c.theme.Space(2))
		r.Role(RoleNone)
		r.Children(func() { s.prefix(c, i, children, func() { row(s.rows[i].item) }) })
	}, treeList)
	e.afterInput(func() { s.keys(e, children) })
	return e
}

// OutlineTable creates an outline whose rows are a Table's, as AppKit's
// outline view with columns: the first of columns shows the items' arrows,
// indented as they are deep, before what cell builds. It takes the
// columns, the sort and the keys of a Table, and those of an Outline.
//
//	ui.OutlineTable(c, &app.files, cols, roots, children, func(path string, col int) {
//		switch col {
//		case 0:
//			ui.Text(c, filepath.Base(path)).SingleLine()
//		case 1:
//			ui.Text(c, app.size(path))
//		}
//	})
func coreOutlineTable[K comparable](c *context, s *OutlineState[K], columns []TableColumn, roots []K, children func(K) []K, cell func(item K, col int)) *node {
	s.flatten(roots, children)
	s.List.Key = func(i int) any { return s.rows[i].item }
	e := table(c, &s.List, columns, len(s.rows), func(i, col int) {
		if col != 0 {
			cell(s.rows[i].item, col)
			return
		}
		s.prefix(c, i, children, func() { cell(s.rows[i].item, 0) })
	}, treeTableList)
	e.afterInput(func() { s.keys(e, children) })
	return e
}

// prefix builds the indentation and the arrow of row i, then content, and
// tells assistive technology how deep the row is and whether it is open.
func (s *OutlineState[K]) prefix(c *context, i int, children func(K) []K, content func()) {
	t := c.theme
	r := s.rows[i]
	open := r.branch && s.Open.Has(r.item)
	// The list's element holding the row is the tree's item.
	var item *node
	for p := c.parent; p != nil; p = p.parent {
		if p.listRow {
			item = p
			p.level, p.expandable, p.expanded = r.depth+1, r.branch, open
			break
		}
	}
	if r.depth > 0 {
		indent := s.Indent
		if indent == 0 {
			indent = t.Space(4)
		}
		coreBox(c).Width(float32(r.depth) * indent).Shrink(0)
	}
	arrow := coreBox(c).Size(t.Space(4), t.Space(4)).Shrink(0).Role(RoleNone)
	if r.branch {
		arrow.flags |= flagClickable | flagKeepFocus
		arrow.afterInput(func() {
			toggle := arrow.Clicked()
			if item != nil && item.st.expand != 0 {
				// Assistive technology opening or closing it.
				toggle = (item.st.expand > 0) != open
				item.st.expand = 0
				c.rt.consumed = true
			}
			if toggle {
				s.setOpen(r.item, !open, arrow.ClickModifiers()&Alt != 0, children)
				s.closed(i, open)
				open = !open
				if item != nil {
					item.expanded = open
				}
			}

		})
		turn := arrow.Animate("open", 90*b2f(open), 150*time.Millisecond)
		rtl := arrow.rtl()
		arrow.Draw(func(p *Painter, rect Rect) {
			p.StrokePath(disclosurePath(rect, turn, rtl), 1.5, arrowColor(t, item))
		})
	}
	content()
}

// closed moves the choice to row i as it closes, from a row inside it,
// which no longer shows.
func (s *OutlineState[K]) closed(i int, wasOpen bool) {
	sel := s.List.cursor()
	if !wasOpen || sel == nil || *sel <= i {
		return
	}
	for p := *sel; p >= 0; p = s.rows[p].parent {
		if p == i {
			s.List.frame.choose(i)
			return
		}
	}
}

// keys opens and closes the item chosen with Right and Left while the
// outline has the focus, or moves to its first child or its parent.
func (s *OutlineState[K]) keys(owner *node, children func(K) []K) {
	sel := s.List.cursor()
	if sel == nil {
		return
	}
	// Right opens, toward where the rows' content goes; Left right to
	// left.
	back, forth := owner.arrows()
	right, left := owner.Shortcut(0, forth), owner.Shortcut(0, back)
	allRight, allLeft := owner.Shortcut(allMod, forth), owner.Shortcut(allMod, back)
	i := *sel
	if i < 0 || i >= len(s.rows) {
		return
	}
	r := s.rows[i]
	open := r.branch && s.Open.Has(r.item)
	f := &s.List.frame
	switch {
	case (right || allRight) && r.branch && (!open || allRight):
		s.setOpen(r.item, true, allRight, children)
	case right && open && i+1 < len(s.rows) && s.rows[i+1].parent == i:
		f.choose(i + 1)
	case (left || allLeft) && open:
		s.setOpen(r.item, false, allLeft, children)
	case (left || allLeft) && r.parent >= 0:
		f.choose(r.parent)
	}
}

// allMod opens and closes all inside an item with the arrows: Option on
// macOS, Shift elsewhere, as in GTK, where Alt and the arrows go back and
// forward.
var allMod = func() Modifiers {
	if runtime.GOOS == "darwin" {
		return Alt
	}
	return Shift
}()

// arrowColor is the color of the arrow of an item of a tree: muted, or
// the accent's text on the item chosen.
func arrowColor(t *Theme, item *node) Color {
	if item != nil && item.checked == 2 {
		return t.AccentText
	}
	return t.TextMuted
}

// rotate returns a function turning points around (cx, cy) by deg degrees
// clockwise.
func rotate(cx, cy, deg float32) func(x, y float32) (float32, float32) {
	sin, cos := sincos(deg)
	return func(x, y float32) (float32, float32) { return cx + x*cos - y*sin, cy + x*sin + y*cos }
}
