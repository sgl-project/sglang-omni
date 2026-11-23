package ui

import (
	"maps"
	"slices"
	"strconv"
)

// TableColumn describes a column of a Table.
type TableColumn struct {
	// Title heads the column. ID identifies it among the others, for the
	// order and the widths the user gives the columns (ListState.Columns)
	// and for the sort (ListState.Sort and SortBy): the Title when empty.
	Title, ID string
	// Width is the column's width in DIPs, or 0 to share the room the
	// others leave. The user resizes the column by dragging the right edge
	// of its header, between MinWidth and MaxWidth (none for 0), and fits
	// it to its cells with a double click there.
	Width, MinWidth, MaxWidth float32
	// Align aligns the content of its cells.
	Align Align
	// Sortable lets a click on the column's header sort the rows by it,
	// with ListState.Sort or SortBy.
	Sortable bool
	// Fixed keeps the column where it is and as wide as it is: the user
	// neither moves nor resizes it.
	Fixed bool
}

func (col *TableColumn) id() string {
	if col.ID != "" {
		return col.ID
	}
	return col.Title
}

// SortOrder is how a table's rows are sorted: by the column whose ID is
// Column, in ascending order unless Descending.
type SortOrder struct {
	Column     string
	Descending bool
}

// TableLayout is the order and the widths the user gave the columns of a
// Table, by their IDs. Columns it does not name keep the order and the
// widths the table gives them.
type TableLayout struct {
	Order  []string
	Widths map[string]float32
}

// arrange returns the indexes of columns in the order the layout gives
// them: those it names in its order, the others after them in theirs, and
// fixed columns where they are.
func (l *TableLayout) arrange(columns []TableColumn) []int {
	order := make([]int, 0, len(columns))
	var named []int
	for _, id := range l.Order {
		if j := slices.IndexFunc(columns, func(col TableColumn) bool { return col.id() == id }); j >= 0 && !columns[j].Fixed && !slices.Contains(named, j) {
			named = append(named, j)
		}
	}
	// Fixed columns at their place, the others in the layout's order, then
	// in theirs.
	movable := append([]int(nil), named...)
	for j := range columns {
		if !columns[j].Fixed && !slices.Contains(named, j) {
			movable = append(movable, j)
		}
	}
	for j := range columns {
		if columns[j].Fixed {
			order = append(order, j)
		} else {
			order = append(order, movable[0])
			movable = movable[1:]
		}
	}
	return order
}

// width returns the width of a column: the user's, or the table's, or
// false for one sharing the room left.
func (l *TableLayout) width(col *TableColumn) (float32, bool) {
	w := col.Width
	if v, ok := l.Widths[col.id()]; ok && !col.Fixed {
		w = v
	}
	if w <= 0 {
		return 0, false
	}
	return clampWidth(col, w), true
}

func clampWidth(col *TableColumn, w float32) float32 {
	if col.MaxWidth > 0 {
		w = min(w, col.MaxWidth)
	}
	return max(w, col.MinWidth, 16)
}

// tableDrag is a press on a table's header: the column pressed, by ID,
// how far the pointer moved since, and whether it moves the column.
type tableDrag struct {
	id     string
	dx     float32
	moving bool
	// fit is the column to fit to its cells in this frame, by ID.
	fit string
}

// Table creates a table of n rows under a header of columns, whose rows
// are a List's: it builds only those in view, cell building the content
// of a row's column, usually a Text. Rows are as high as their tallest
// cell, and at least as high as the header.
//
// s keeps the place of the rows and says how they behave, as a List's, or
// nil: with s.Selected, a click chooses a row, as Up, Down, Home and End
// do while the table has the keyboard focus, Changed reports a new choice,
// and Submitted a double click or Enter on it; with s.Selection, the user
// chooses several. A row s.Header names spans
// every column, cell building it as column 0, and stays at the top while
// the rows of its section scroll under it.
//
// The user resizes columns by dragging the edges of their headers, and
// moves them by dragging the headers, which s.Columns keeps; columns wider
// than the table scroll sideways, the header with them. With s.Sort, a
// click on the header of a Sortable column sorts the rows by it; with
// s.SortBy, a Cmd-click or a Shift-click sorts by it after the others;
// with s.HeaderMenu, the header has a context menu.
//
//	cols := []ui.TableColumn{{Title: "Name", Sortable: true}, {Title: "Size", Width: 90, Align: ui.End, Sortable: true}}
//	app.files.Selected, app.files.Sort = &app.file, &app.sort
//	if ui.Table(c, &app.files, cols, len(files), func(row, col int) {
//		f := files[row]
//		switch col {
//		case 0:
//			ui.Text(c, f.Name).SingleLine()
//		case 1:
//			ui.Text(c, f.Size())
//		}
//	}).Grow(1).Submitted() {
//		app.open(files[app.file])
//	}
func coreTable(c *context, s *ListState, columns []TableColumn, n int, cell func(row, col int)) *node {
	return table(c, s, columns, n, cell, tableList)
}

// table creates a Table, or with kind treeTableList an OutlineTable.
func table(c *context, s *ListState, columns []TableColumn, n int, cell func(row, col int), kind listKind) *node {
	t := c.theme
	// The height of the header, and the least of the rows.
	tableRow := t.Space(8)
	table := coreColumn(c).Role(RoleTable).Focusable().Clip()
	table.widget = "Table"
	if kind == treeTableList {
		table.widget, table.role = "OutlineTable", RoleTree
	}
	table.flags |= flagOwnRing
	if s == nil {
		s = coreLocal(table, "rows", func() ListState { return ListState{} })
	}
	drag := coreLocal(table, "drag", func() tableDrag { return tableDrag{} })
	s.syncSort()
	layout := &s.Columns
	order := layout.arrange(columns)
	// The least width of the rows: the columns', those sharing the room
	// left at their least.
	flexMin := t.Space(15)
	var least float32
	for _, j := range order {
		if w, ok := layout.width(&columns[j]); ok {
			least += w
		} else {
			least += max(columns[j].MinWidth, flexMin)
		}
	}
	// fitting are the cells of the column fitting its cells.
	var fitting []*node
	// cells builds a row's cells, with fill building the content of each.
	cells := func(role Role, fill func(col int)) []*node {
		boxes := make([]*node, 0, len(order))
		for _, j := range order {
			col := &columns[j]
			// Keyed by its column, a cell keeps its state as the columns
			// move: a header being dragged keeps the press.
			box := coreRow(c).Key(col.id()).Padding(t.Space(1.5), t.Space(2.5)).AlignItems(Center).Shrink(0).Clip().Role(role)
			if w, ok := layout.width(col); ok {
				box.Width(w)
			} else {
				box.Grow(1).Shrink(1).MinWidth(max(col.MinWidth, flexMin))
			}
			switch col.Align {
			case Center:
				box.Justify(Center)
			case End:
				box.Justify(End)
			}
			box.Children(func() { fill(j) })
			if drag.fit != "" && drag.fit == col.id() {
				fitting = append(fitting, box)
			}
			boxes = append(boxes, box)
		}
		return boxes
	}
	var list *node
	table.Children(func() {
		// The header, which scrolls sideways with the rows.
		head := coreRow(c).Height(tableRow).Shrink(0).AlignItems(Stretch).Role(RoleRow).Clip()
		if menu := s.HeaderMenu; menu != nil {
			// The part of the header past the columns, as AppKit's
			// clickedColumn -1.
			head.ContextMenu(func(m *Menu) { menu(m, -1) })
		}
		head.Children(func() {
			heads := cells(RoleColumnHeader, func(j int) {
				col := &columns[j]
				coreText(c, col.Title).SingleLine().FontWeight(600).TextColor(t.TextMuted)
				if at, descending := s.sortPlace(col); at >= 0 {
					place := 0
					if s.SortBy != nil && len(*s.SortBy) > 1 {
						place = at + 1
					}
					sortArrow(c, descending, place)
				}
			})
			for k, j := range order {
				tableHeader(c, table, heads[k], j, s, drag, heads, order, columns)
			}
		})
		coreDivider(c)
		// The rows are a List's, whose choice the table takes the focus
		// and the keys for.
		list = coreScroll(c).Grow(1).Role(RoleNone)
		list.widget = "List"
		list.flags |= flagScrollX
		list.rowMinW = least
		head.followX = list
		buildList(c, list, table, s, n, func(i int) {
			row := coreRow(c).MinHeight(tableRow).AlignItems(Stretch)
			// The list's element holding the row is the row.
			row.Role(RoleNone)
			if s.Header != nil && s.Header(i) {
				row.Padding(t.Space(1.5), t.Space(2.5)).AlignItems(Center).Background(t.Surface).FontWeight(600)
				row.Children(func() { cell(i, 0) })
				return
			}
			if s.cursor() == nil {
				// Chosen rows show the pointer over them already.
				row.flags |= flagHover
				if s.RoundRows {
					row.Radius(t.Radius)
				}
				row.styleFn = func(row *node) {
					if row.Hovered() {
						row.bg = t.SurfaceHover
					}
				}
			}
			row.Children(func() {
				cells(RoleCell, func(j int) { cell(i, j) })
			})
		}, kind)
	})
	if id := drag.fit; id != "" {
		table.colFit = &tableFit{id: id, cells: fitting, layout: layout, drag: drag}
	}
	table.DrawOver(func(p *Painter, r Rect) {
		if table.FocusVisible() {
			p.FocusRing(r, [4]float32{})
		}
	})
	return table
}

// tableHeader makes the header of columns[j] sort the rows when clicked,
// move the column when dragged, resize it from its end edge (the right,
// or the left right to left), and open s.HeaderMenu.
func tableHeader(c *context, table, h *node, j int, s *ListState, drag *tableDrag, heads []*node, order []int, columns []TableColumn) {
	t := c.theme
	col := &columns[j]
	id := col.id()
	if menu := s.HeaderMenu; menu != nil {
		h.ContextMenu(func(m *Menu) { menu(m, j) })
	}
	if s.sorts() && col.Sortable {
		h.sort = 0
		if at, descending := s.sortPlace(col); at >= 0 {
			h.sort = 1
			if descending {
				h.sort = 2
			}
		}
		h.Label(col.Title) // the title alone, not the place number beside it
		h.flags |= flagClickable | flagHover
		h.styleFn = func(h *node) {
			if h.Hovered() {
				h.bg = t.SurfaceHover
			}
		}
	}
	// Moving the column, once the pointer went a few DIPs.
	if dx, _, held := h.Dragged(); held && !col.Fixed {
		if drag.id != id {
			*drag = tableDrag{id: id}
		}
		drag.dx += dx
		if !drag.moving && abs(drag.dx) > 4 {
			drag.moving = true
		}
		if drag.moving {
			moveColumn(c, s, drag, heads, order, columns)
			h.Left(drag.dx).Background(t.SurfacePressed).Opacity(0.9)
		}
	} else if drag.id == id && !held {
		// Released: a click, unless it moved the column.
		moved := drag.moving
		*drag = tableDrag{fit: drag.fit}
		if moved {
			h.st.clicks = 0
		}
	}
	h.afterInput(func() {
		if s.sorts() && col.Sortable && h.Clicked() {
			if s.SortBy != nil {
				// Cmd-click as Finder, Shift-click as web grids (TanStack
				// Table, AG Grid, MUI).
				*s.SortBy = sortClicked(*s.SortBy, id, h.ClickModifiers()&(Cmd|Shift) != 0)
				s.syncSort()
			} else {
				*s.Sort = sortClicked([]SortOrder{*s.Sort}, id, false)[0]
			}
			table.st.markChanged()
		}
	})
	if col.Fixed {
		return
	}
	// The edge, which resizes the column, and fits it to its cells on a
	// double click.
	grip := t.Space(2)
	h.Children(func() {
		edge := coreBox(c).Absolute().Top(0).Bottom(0).End(0).Width(grip).Cursor(CursorResizeEW).Role(RoleNone)
		rtl := edge.rtl()
		edge.flags |= flagHover
		if edge.DoubleClicked() {
			drag.fit = id
			c.rt.consumed = true
		}
		if dx, _, held := edge.Dragged(); held && dx != 0 {
			if rtl {
				// The edge is on the left: dragging it left widens.
				dx = -dx
			}
			// The columns before it sharing the room left keep their
			// widths, for the edge to follow the pointer: the room the
			// column takes or gives comes from the columns after it, or
			// scrolls.
			for k, j := range order {
				if heads[k] == h {
					break
				}
				if _, ok := s.Columns.width(&columns[j]); !ok {
					s.setWidth(&columns[j], heads[k].Bounds().W)
				}
			}
			w, ok := s.Columns.width(col)
			if !ok {
				w = h.Bounds().W // sharing the room left, as wide as it was
			}
			s.setWidth(col, w+dx)
		}
		edge.Draw(func(p *Painter, r Rect) {
			x := r.X + r.W - 0.5
			if rtl {
				x = r.X + 0.5
			}
			color := t.Border
			if edge.Hovered() || edge.Pressed() {
				color = t.Accent
			}
			p.Fill(Rect{x - 0.5, r.Y + r.H/4, 1, r.H / 2}, color, 0)
		})
	})
}

// setWidth gives a column a width the user chose.
func (s *ListState) setWidth(col *TableColumn, w float32) {
	if s.Columns.Widths == nil {
		s.Columns.Widths = map[string]float32{}
	} else {
		// A copy, as the app may hold the map it restored.
		s.Columns.Widths = maps.Clone(s.Columns.Widths)
	}
	s.Columns.Widths[col.id()] = clampWidth(col, w)
}

// moveColumn moves the column being dragged past a neighbor whose middle
// it passed, keeping it under the pointer.
func moveColumn(c *context, s *ListState, drag *tableDrag, heads []*node, order []int, columns []TableColumn) {
	at := slices.IndexFunc(order, func(j int) bool { return columns[j].id() == drag.id })
	if at < 0 {
		return
	}
	to := at
	// The columns after it are to its left right to left.
	sign := float32(1)
	if heads[at].rtl() {
		sign = -1
	}
	dx := drag.dx * sign
	switch {
	case dx > 0 && at+1 < len(order):
		if next := heads[at+1].Bounds(); next.W > 0 && dx > next.W/2 {
			to = at + 1
		}
	case dx < 0 && at > 0:
		if prev := heads[at-1].Bounds(); prev.W > 0 && -dx > prev.W/2 {
			to = at - 1
		}
	}
	if to == at || columns[order[to]].Fixed {
		return
	}
	// Under the pointer still: the column's place moved by the other's
	// width.
	if to > at {
		drag.dx -= sign * heads[to].Bounds().W
	} else {
		drag.dx += sign * heads[to].Bounds().W
	}
	ids := make([]string, len(order))
	for k, j := range order {
		ids[k] = columns[j].id()
	}
	ids[at], ids[to] = ids[to], ids[at]
	s.Columns.Order = ids
	c.rt.consumed = true
}

// tableFit fits a column of a table to its cells as the table lays out,
// the cells built in the frame and the header.
type tableFit struct {
	id     string
	cells  []*node
	layout *TableLayout
	drag   *tableDrag
	done   bool
}

// apply sets the column's width to the widest of its cells, once, before
// the table is measured or laid out.
func (tf *tableFit) apply() {
	if tf.done {
		return
	}
	tf.done, tf.drag.fit = true, ""
	var w float32
	for _, cell := range tf.cells {
		cell.width = length{}
		w = max(w, intrinsic(cell, true))
	}
	for _, cell := range tf.cells {
		cell.width, cell.grow, cell.shrink = px(w), 0, 0
	}
	if tf.layout.Widths == nil {
		tf.layout.Widths = map[string]float32{}
	} else {
		tf.layout.Widths = maps.Clone(tf.layout.Widths)
	}
	tf.layout.Widths[tf.id] = w
}

// syncSort keeps Sort, when set with SortBy, the first column SortBy sorts
// by, or the zero SortOrder when it sorts by none.
func (s *ListState) syncSort() {
	if s.Sort == nil || s.SortBy == nil {
		return
	}
	var first SortOrder
	if by := *s.SortBy; len(by) > 0 {
		first = by[0]
	}
	*s.Sort = first
}

// sorts reports whether a Table's headers sort its rows.
func (s *ListState) sorts() bool { return s.Sort != nil || s.SortBy != nil }

// sortPlace returns the place of col among the columns the rows are sorted
// by, from 0, and whether it sorts them descending, or -1.
func (s *ListState) sortPlace(col *TableColumn) (int, bool) {
	if !col.Sortable {
		return -1, false
	}
	id := col.id()
	if s.SortBy != nil {
		for i, o := range *s.SortBy {
			if o.Column == id {
				return i, o.Descending
			}
		}
		return -1, false
	}
	if s.Sort != nil && s.Sort.Column == id {
		return 0, s.Sort.Descending
	}
	return -1, false
}

// sortClicked returns, as a new slice, the columns sorted by after a click
// on the header of the column whose ID is id, a Cmd-click when add.
func sortClicked(by []SortOrder, id string, add bool) []SortOrder {
	i := slices.IndexFunc(by, func(o SortOrder) bool { return o.Column == id })
	if !add {
		if i == 0 {
			return []SortOrder{{Column: id, Descending: !by[0].Descending}}
		}
		return []SortOrder{{Column: id}}
	}
	by = slices.Clone(by)
	switch {
	case i < 0:
		return append(by, SortOrder{Column: id})
	case !by[i].Descending:
		by[i].Descending = true
		return by
	default:
		return slices.Delete(by, i, i+1)
	}
}

// sortArrow draws the arrow of a column the rows are sorted by, after its
// place among them from 1, or alone for 0.
func sortArrow(c *context, descending bool, place int) {
	t := c.theme
	if place > 0 {
		coreText(c, strconv.Itoa(place)).FontSize(t.FontSize * 0.75).TextColor(t.TextMuted).Shrink(0).MarginStart(t.Space(1)).Role(RoleNone)
	}
	coreBox(c).Size(t.Space(2.5), t.Space(2.5)).Shrink(0).MarginStart(t.Space(1)).Draw(func(p *Painter, r Rect) {
		var path Path
		if descending {
			path.MoveTo(r.X+r.W*0.1, r.Y+r.H*0.3).LineTo(r.X+r.W*0.5, r.Y+r.H*0.7).LineTo(r.X+r.W*0.9, r.Y+r.H*0.3)
		} else {
			path.MoveTo(r.X+r.W*0.1, r.Y+r.H*0.7).LineTo(r.X+r.W*0.5, r.Y+r.H*0.3).LineTo(r.X+r.W*0.9, r.Y+r.H*0.7)
		}
		p.StrokePath(&path, 1.5, t.TextMuted)
	})
}

func abs(v float32) float32 {
	if v < 0 {
		return -v
	}
	return v
}

// Selected shows the element as chosen among its siblings, in the accent
// color, as a row of a list, and tells assistive technology it is.
func (e *node) Selected(on bool) *node {
	if !on {
		e.checked = 1
		return e
	}
	t := e.c.theme
	e.checked = 2
	e.Background(t.Accent).TextColor(t.AccentText)
	return e
}
