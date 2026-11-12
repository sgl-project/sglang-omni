package ui

import (
	"math"
	"math/bits"
	"runtime"
	"slices"
	"strings"
)

// A List builds only the rows in view, and keeps its place by a row, its
// anchor, and how far its top is from the top of the list, rather than by
// an offset into its content: rows above and below may change height as
// they are measured, rows may come and go, and the rows in view stay
// where they are. Building, it makes the rows its place shows, as far as
// the heights it knows tell; laying out, it measures them, places them
// from the anchor, and builds the rows still missing, so a frame never
// shows a gap. The offset of its content, which its scroll bar shows and
// the wheel moves, is where the heights known (measured, or estimated as
// their average) put the anchor, in float64: millions of rows scroll by
// fractions of a DIP.

// listOverscan is how many rows a List builds beyond each edge of its
// view, for Tab to move the focus to and assistive technology to read.
const listOverscan = 2

// listMaxRows bounds the rows a List builds in a frame, against rows
// taking no room.
const listMaxRows = 4096

// listSearch is how far from where it was a List looks for an item's key
// when its rows changed.
const listSearch = 1000

// ListState is a List's place among its rows, how it behaves, and what it
// knows of the heights of its rows, kept in the app's state. Give each
// List a ListState of its own; set its fields before building the List.
type ListState struct {
	// Handle names this list independently of its per-pass node.
	Handle

	// Key returns an identity for the item row i shows, such as its ID:
	// comparable, and unique among the rows. Without it, a row is its
	// index. With it, the state of a row (focus, text being edited, …)
	// follows its item, and the list keeps its place and its choice when
	// rows are added or removed above them, as when older messages load.
	Key func(row int) any
	// KeepOffset keeps a list with a Key where it is scrolled as its rows
	// move, rather than with the item at its top: for rows sorted again as
	// they change, as a process monitor's, which the list would otherwise
	// scroll along with the item at its top. The state of a row, the choice
	// and the row Shift chooses from still follow their items.
	KeepOffset bool
	// FollowEnd starts the list at its end, and keeps the end in view as
	// rows are added or grow while it shows it, as a chat or a log does:
	// scrolling away from the end stops following it, and scrolling back
	// to it follows again.
	FollowEnd bool
	// FollowAnimation, when set, makes a list following its end glide to
	// it as rows are added or grow, as the animation says (Smooth, a
	// spring, or a curve), rather than show it at once: what is added
	// comes up from below, as the chat apps scroll. A glide on its way
	// goes on from where it is as more comes, and the user scrolling stops
	// it. It shows the end at once where the desktop asks for less motion.
	FollowAnimation *Animation
	// Selected, when set, lets a click choose a row, as Up, Down, Home and
	// End do while the list has the keyboard focus, and Page Up and Page
	// Down on Linux and Windows: *Selected is the chosen row, -1 for none,
	// which shows in the accent color. The list's Changed reports a new
	// choice, and its Submitted a double click or Enter.
	Selected *int
	// Selection, when set, lets the user choose several rows, which it
	// holds by their keys: a click chooses one, Cmd-click (Ctrl-click on
	// Linux and Windows) adds a row or takes it out, and Shift-click
	// chooses those from the row last chosen to it, as Shift with the keys
	// that move the choice does; Cmd+A chooses all. On Linux and Windows,
	// Ctrl with those keys moves without choosing, and Ctrl+Space adds the
	// row there or takes it out. Selected, set as well, is the row last
	// chosen, where the keys move from.
	Selection Selector
	// Label returns the text of row i: typing the first letters of a
	// row's text while the list has the keyboard focus chooses it, and
	// assistive technology reads it as the row's name.
	Label func(row int) string
	// Header, when set, reports whether row i heads a section: the header
	// of the section at the top of the list stays there while the rows of
	// its section scroll under it, until the next header pushes it away.
	// Give headers a background. Headers are not chosen.
	Header func(row int) bool
	// Reorder, when set, lets the user drag rows to another place in the
	// list: the rows chosen, when the row dragged is one, else that row,
	// which the list passes in order, and the row they go before, the
	// number of rows for the end. Move them there: the list shows where
	// they would go as they are dragged, and scrolls near its edges.
	Reorder func(rows []int, to int)
	// Sort, when set, is the column a Table's rows are sorted by: a click
	// on the header of a Sortable column sorts by it, ascending, and
	// another reverses the order, which the header shows. Sort the rows as
	// it says, when it changes. SortBy sorts by several columns, which
	// Sort, set with it, follows.
	Sort *SortOrder
	// SortBy sorts a Table's rows by several columns, the first first. A
	// click on a Sortable column's header sorts by it alone, as with Sort,
	// and reverses it when it was first; a Cmd-click (Ctrl-click on Linux
	// and Windows) or a Shift-click adds it after the others, ascending,
	// another makes it descending, and a third takes it out. The headers
	// show their direction and, when there are several, their place. It
	// gets a new slice as it changes, so one the app kept stays as it was.
	// When Sort is set too, the table keeps it the first of SortBy, the
	// zero SortOrder when SortBy is empty, for the app to read the column
	// that sorts first.
	SortBy *[]SortOrder
	// Columns keeps the order and the widths the user gave a Table's
	// columns, by ID: restore it from saved settings, and save it as it
	// changes.
	Columns TableLayout
	// RoundRows rounds the corners of a Table's chosen and hovered rows at
	// the theme's Radius, as a List's are, rather than leaving them square
	// from edge to edge.
	RoundRows bool
	// HeaderMenu, when set, gives a Table's header a context menu, built
	// as ContextMenu's is: col is the index of the column clicked in the
	// columns given to Table, not in the order the user moved them into,
	// or -1 for the part of the header past the columns, as AppKit's
	// clickedColumn. A menu built without items does not open.
	HeaderMenu func(m *Menu, col int)

	// The place: the first row in view, and how far the list is scrolled
	// past its top: the row's top is that far above where the content
	// starts, so that the offset is the row's plus inset.
	anchor    int
	inset     float64
	anchorKey any
	following bool
	started   bool
	// followLag is how far below its place the end shows as it glides
	// there (FollowAnimation); followRow and followKey are the last row of
	// the frame before, followBelow how much content was from its top to
	// the end, and followWidth the width of rows then: more content below
	// that row, as it grows or rows come after it, is the end moving down.
	followLag   channel
	followRow   int
	followKey   any
	followBelow float64
	followWidth float32
	followSeen  bool
	laidOut     bool
	// wroteY is the offset the list set, to tell it from scrolling.
	wroteY  float64
	req     listRequest
	heights listHeights
	// width, gap and padTop are those the rows were placed with: the width
	// of the rows, the gap between them, and where the first starts below
	// the list's top. border is its top border.
	width       float32
	gap         float64
	padTop      float64
	border      float64
	first, last int
	atEnd       bool
	// read is set when the view read where the list is, which a frame
	// that moves it then builds another for.
	read bool
	// selRow is the row chosen in the last frame and selKey its key, to
	// tell the app choosing another from rows moving.
	selRow int
	selKey any
	// lead is the row last chosen in a list choosing several without
	// Selected. Shift chooses the rows from pivot, whose key is pivotKey,
	// and chose those from it to pivot+span last.
	lead, pivot, span int
	pivotKey          any

	header listHeader
	// rows are the last frame's rows by their elements' IDs, to find the
	// row holding the focus.
	rows  map[uint64]listRowID
	frame listFrame
	// editables reports rows holding an EditableText in the last frame,
	// and editablesNow in this one; editKey is the row whose text the keys
	// asked to edit, while editAsked.
	editables, editablesNow bool
	editKey                 any
	editAsked               bool
}

// listRequest is where the app or the user asked the list to go: to row
// at align, as little as shows it (near), to its end, or to offset y
// (with row, when hint is set, the row to place from there).
type listRequest struct {
	set, end, near, offset, hint bool
	row                          int
	align                        Align
	y                            float64
}

type listRowID struct {
	row int
	key any
}

// ScrollTo scrolls the list to show row at align: Start shows it at the
// top, Center in the middle and End at the bottom. Call it from the view:
// the list scrolls as the frame is laid out, to where the row is once it
// is measured.
func (s *ListState) ScrollTo(row int, align Align) {
	s.req = listRequest{set: true, row: row, align: align}
}

// ScrollIntoView scrolls the list as little as shows row whole, as
// Element.ScrollIntoView does an element.
func (s *ListState) ScrollIntoView(row int) {
	s.req = listRequest{set: true, row: row, near: true}
}

// ScrollToEnd scrolls the list to its end, where FollowEnd follows it
// again.
func (s *ListState) ScrollToEnd() { s.req = listRequest{set: true, end: true} }

// Visible returns the first and the last row the list showed in the last
// frame, last below first when it showed none: a list nearing the end of
// what was loaded, say, loads more. A frame that changes them builds
// another, for what read them.
func (s *ListState) Visible() (first, last int) {
	s.read = true
	if !s.laidOut {
		return 0, -1
	}
	return s.first, s.last
}

// AtEnd reports whether the list showed its end in the last frame, as
// Visible does its rows.
func (s *ListState) AtEnd() bool {
	s.read = true
	return s.atEnd
}

// Focused reports whether the List, Table or Outline using s has
// the keyboard focus. Call it from the view, after building the list or
// while building its rows. It returns false when s was not built in the
// current pass of c, including when the list is hidden.
func (s *ListState) coreFocused(c *context) bool { return s.current(c).Focused() }

// FocusWithin reports whether the list or one of its descendants has the
// keyboard focus, with the same lifetime as Focused. Row builders can use
// it before List returns, without saving an Element from an earlier frame.
func (s *ListState) coreFocusWithin(c *context) bool { return s.current(c).FocusWithin() }

// Focus gives the list the keyboard focus and reports whether it was
// built in the current pass of c. A pending focus request can wait until
// a hidden list returns:
//
//	if app.focusFiles && app.files.Focus(c) {
//		app.focusFiles = false
//	}
//
// Call it after building the list, or while building its rows.
func (s *ListState) coreFocus(c *context) bool {
	e := s.current(c)
	if e == nil {
		return false
	}
	e.Focus()
	return true
}

// Shortcut reports whether mods+key was pressed while the list or one of
// its descendants had the focus, as Element.Shortcut does. Call it after
// building the list, or while building its rows. A list not built in the
// current pass of c registers and handles no shortcuts.
func (s *ListState) coreShortcut(c *context, mods Modifiers, key Key) bool {
	return s.current(c).Shortcut(mods, key)
}

// current resolves the focus owner of this pass before reading an arena
// pointer: an absent list's old slot may now hold an unrelated element.
func (s *ListState) current(c *context) *node {
	if s == nil || c == nil || c.rt == nil || c.rt.closed || !c.rt.inFrame {
		return nil
	}
	f := &s.frame
	if f.c != c || f.frame != c.rt.frame || f.pass != c.rt.pass {
		return nil
	}
	return f.owner
}

// listFrame is what a List built in the frame, for laying it out.
type listFrame struct {
	c *context
	e *node
	// owner takes the focus and reports choices: the list, or its Table.
	owner *node
	s     *ListState
	n     int
	row   func(i int)
	frame uint64
	pass  int
	// flat leaves the corners of chosen rows square, as a Table's unless
	// ListState.RoundRows; tree makes the rows items of a tree, as an
	// Outline's.
	flat, tree bool
	// grid makes the rows those of a GridView, which holds its items.
	grid bool
	// theme is the theme the rows are built with, laying out as well.
	theme *Theme
	rows  []listRow
	// at finds rows by their index while the list lays out.
	at map[int]int
	// rb is the row being built (Context.row), kept here rather than
	// allocated for each row.
	rb rowBuild
}

type listRow struct {
	i   int
	key any
	e   *node
	y   float64
}

// rowBuild is the row of a list being built, for what is inside it: its
// list, index and key, whether it was chosen before a click on it in this
// frame, and whether it was double-clicked.
type rowBuild struct {
	f                       *listFrame
	i                       int
	key                     any
	clicked, chosen, double bool
}

// List creates a vertical scroll container of n rows, built with row(i),
// which builds only those in view, and a few beyond. Rows take the height
// of their content: the list measures them as they show and estimates the
// others from them, so a list of millions of rows is as fast as one of
// ten. Give it a size, or Grow it in its parent:
//
//	ui.List(c, nil, len(app.files), func(i int) {
//		ui.Text(c, app.files[i].Name).Padding(6, 12)
//	}).Grow(1)
//
// s keeps its place and says how it behaves, or nil keeps its place in
// the list's own state: with a ListState, the app scrolls the list to a
// row, rows keep their state with their items, the list follows its end,
// chooses rows, and pins the headers of sections. Gap spaces the rows,
// Padding pads the content, scrolling with it, and Justify(End) puts rows
// that do not fill the list at its bottom, as a chat's first messages.
func coreList(c *context, s *ListState, n int, row func(i int)) *node {
	e := coreScroll(c)
	e.widget, e.role = "List", RoleList
	if s == nil {
		s = coreLocal(e, "list", func() ListState { return ListState{} })
	}
	buildList(c, e, e, s, n, row, plainList)
	return e
}

// buildList builds the rows of list e that its place shows, with owner
// taking the focus and reporting choices.
// listKind is what a list's rows are: a List's, a Table's, an
// Outline's, or an OutlineTable's.
type listKind uint8

const (
	plainList listKind = iota
	tableList
	treeList
	treeTableList
	// gridList's rows hold the items of a GridView.
	gridList
)

func buildList(c *context, e, owner *node, s *ListState, n int, row func(i int), kind listKind) {
	flat, tree := kind == tableList || kind == treeTableList, kind == treeList || kind == treeTableList
	rt := c.rt
	f := &s.frame
	if f.e != nil && f.frame == rt.frame && f.pass == rt.pass {
		panic("ui: a ListState shown by two Lists")
	}
	n = max(n, 0)
	*f = listFrame{c: c, e: e, owner: owner, s: s, n: n, row: row, frame: rt.frame, pass: rt.pass, flat: flat, tree: tree, grid: kind == gridList, theme: c.theme, rows: f.rows[:0], at: f.at}
	if rt.pass == 0 {
		s.editables, s.editablesNow = s.editablesNow, false
	}
	e.list, owner.rowsOf = f, f
	wrapElement(owner).Bind(&s.Handle)
	s.sync(e, n)
	if s.cursor() != nil && owner == e {
		e.Focusable()
	}
	a, first, last := s.plan(e, n)
	e.Children(func() {
		for i := first; i < last; i++ {
			f.build(i)
		}
		// The row holding the focus stays out of view as well, with the
		// text being edited in it; and the header of the section at the
		// top is built where the others are.
		focus, ok := s.focusedRow(e, n)
		if ok && (focus < first || focus >= last) {
			f.build(focus)
		}
		if s.Header != nil && n > 0 {
			if h := s.headerAbove(a, n); h >= 0 && (h < first || h >= last) && (!ok || h != focus) {
				f.build(h)
			}
		}
	})
	owner.onValueInput(listInput)
}

// sync follows what changed since the last frame: rows added or removed,
// items moving, the user or the app scrolling.
func (s *ListState) sync(e *node, n int) {
	st, rt := e.st, e.c.rt
	if !s.started {
		s.started = true
		s.following = s.FollowEnd
		s.selRow, s.lead, s.pivot = -1, -1, -1
		s.header.row = -1
	}
	s.heights.def = float64(e.c.theme.Space(8))
	if s.Key != nil && n > 0 {
		if k := s.anchorKey; k != nil {
			if j, ok := s.find(k, s.anchor, n); ok && j != s.anchor {
				// Rows came or went above it: the heights known move with
				// the rows around it.
				s.heights.resize(max(n, s.heights.n))
				s.heights.shift(j - s.anchor)
				s.anchor = j
				s.header.ok = false
			}
		}
		if sel := s.cursor(); sel != nil && *sel == s.selRow && s.selKey != nil {
			// The choice follows its item, and is gone with it.
			j, ok := s.find(s.selKey, *sel, n)
			if !ok {
				j = -1
			}
			*sel, s.selRow = j, j
		}
		if s.pivotKey != nil {
			j, ok := s.find(s.pivotKey, s.pivot, n)
			if !ok {
				j = -1
			}
			s.pivot = j
		}
	}
	if n != s.heights.n {
		s.heights.resize(n)
		s.header.ok = false
	}
	s.anchor = max(0, min(s.anchor, n-1))
	switch {
	case st.list != s || st.born == rt.frame || !s.laidOut:
		// Built anew, as when its page shows again, or showing another
		// place, or a new one, which did not write the offset its element
		// has: the place stands, unless the app set the offset with a
		// ScrollState this frame.
		if st.movedIn(rt.frame) && !s.req.set {
			s.req = listRequest{set: true, offset: true, y: max(0, st.scrollY)}
		}
		st.list = s
	case st.scrollY != s.wroteY && !s.req.set && s.following && s.followLag.value > 0.5 && st.scrollY > s.wroteY:
		// Scrolled toward the end it glides to: it goes on following.
	case st.scrollY != s.wroteY && !s.req.set:
		// Scrolled by the wheel, the keys, the scroll bar or the app: a
		// step moves the rows by as much, from the place; a jump goes
		// where the heights known put the offset, or to the end.
		top := max(0, st.contentH-float64(st.h))
		y := max(0, min(st.scrollY, top))
		switch d := y - s.wroteY; {
		case top > 0 && y >= top-0.5:
			s.req = listRequest{set: true, end: true}
		case d >= -2*float64(st.h) && d <= 2*float64(st.h):
			s.inset += d
		default:
			s.req = listRequest{set: true, offset: true, y: y}
		}
		s.following = s.FollowEnd && top > 0 && y >= top-0.5
	}
	s.wroteY = st.scrollY
}

// anchorAt returns the row at the top of the view scrolled to y, and where
// its top goes in the list's box, as the heights known put them.
func (s *ListState) anchorAt(y float64) (int, float64) {
	a := s.heights.rowAt(max(0, y+s.border-s.padTop), s.gap)
	return a, s.padTop + s.heights.top(a, s.gap) - y
}

// find returns the row whose key is k, looking around row near.
func (s *ListState) find(k any, near, n int) (int, bool) {
	for d := 0; d <= listSearch; d++ {
		lo, hi := near-d, near+d
		if lo < 0 && hi >= n {
			break
		}
		if lo >= 0 && lo < n && s.Key(lo) == k {
			return lo, true
		}
		if d > 0 && hi >= 0 && hi < n && s.Key(hi) == k {
			return hi, true
		}
	}
	return 0, false
}

// plan returns the row the place starts from and the rows to build, as far
// as the heights known tell where the place puts them.
func (s *ListState) plan(e *node, n int) (a, first, last int) {
	if n == 0 {
		return 0, 0, 0
	}
	view := float64(e.st.h)
	if view <= 0 {
		view = float64(e.c.h)
	}
	hs, gap := &s.heights, s.gap
	a, y, above := s.anchor, s.padTop-s.inset, 0.0
	switch r := s.req; {
	case r.set && r.offset:
		a, y = s.anchorAt(r.y)
		if r.hint {
			a, y = max(0, min(r.row, n-1)), s.padTop+hs.top(r.row, gap)-r.y
		}
	case r.set && !r.end:
		// Rows on both sides cover any alignment.
		a, y, above = max(0, min(r.row, n-1)), 0, -view
	case r.end || s.following:
		a = n - 1
		y = view - hs.height(a)
	}
	last = a
	for yy := y; last < n && yy < view && last-a < listMaxRows; last++ {
		yy += hs.height(last) + gap
	}
	first = a
	for yy := y; first > 0 && yy > above && a-first < listMaxRows; {
		first--
		yy -= hs.height(first) + gap
	}
	return a, max(0, first-listOverscan), min(n, last+listOverscan)
}

// build builds row i in the list.
func (f *listFrame) build(i int) *node {
	c, s := f.c, f.s
	var key any = i
	if s.Key != nil {
		key = s.Key(i)
	}
	w := coreBox(c).Key(key).Shrink(0)
	w.listRow, w.rowIndex = true, i
	if s.Label != nil {
		w.Label(s.Label(i))
	}
	// Assistive technology sees a row of a table, an item of a tree or of
	// a list.
	switch {
	case f.grid:
		w.Role(RoleNone)
	case f.tree:
		w.Role(RoleTreeItem)
	case f.flat:
		w.Role(RoleRow)
	default:
		w.Role(RoleListItem)
	}
	prev := f.rb
	f.rb = rowBuild{f: f, i: i, key: key, clicked: w.st.clicks > 0 && w.st.clickMods == 0, double: w.st.doubleClicks > 0}
	rb := &f.rb
	if sel := s.cursor(); sel != nil && !f.isHeader(i) {
		t := c.theme
		w.flags |= flagClickable | flagHover | flagChoosable
		rb.chosen = f.chosen(i, key)
		*valueBinding[listRowInput](w) = listRowInput{f: f, i: i}
		w.onValueInput(listRowValueInput)
		on := f.chosen(i, key)
		w.Selected(on)
		switch r := t.Radius; {
		case f.flat && !s.RoundRows:
		case s.Selection != nil && s.gap == 0 && on:
			// Rows chosen together make one block, as in Finder.
			above := i > 0 && !f.isHeader(i-1) && f.chosen(i-1, f.key(i-1))
			below := i < f.n-1 && !f.isHeader(i+1) && f.chosen(i+1, f.key(i+1))
			w.radius = [4]float32{r, r, r, r}
			if above {
				w.radius[0], w.radius[1] = 0, 0
			}
			if below {
				w.radius[2], w.radius[3] = 0, 0
			}
		default:
			w.Radius(r)
		}
		w.styleFn = func(w *node) {
			if w.checked != 2 && w.Hovered() {
				w.bg = t.SurfaceHover
			}
		}
		if owner := f.owner; s.Selection != nil && *sel == i && !on {
			// Where the keys moved without choosing.
			w.DrawOver(func(p *Painter, r Rect) {
				if owner.FocusVisible() {
					p.FocusRing(Rect{r.X + 3, r.Y + 3, r.W - 6, r.H - 6}, w.radius)
				}
			})
		}
	}
	if s.Reorder != nil && !f.isHeader(i) {
		w.dragFrom(func() any { return &rowDrag{s: s, rows: f.dragged(i)} })
		if w.Dragging() {
			// Where the row was, as it moves with the pointer.
			w.Opacity(0.4)
		}
	}
	saved := c.row
	c.row = rb
	w.Children(func() { f.row(i) })
	c.row, f.rb = saved, prev
	f.rows = append(f.rows, listRow{i: i, key: key, e: w})
	return w
}

// buildLate builds row i while the list lays out, as the view would have:
// with the theme it built the list with, and its input forgotten once the
// frame is laid out (layoutTree).
func (f *listFrame) buildLate(i int) *node {
	c := f.c
	parent, theme := c.parent, c.theme
	c.parent, c.theme = f.e, f.theme
	w := f.build(i)
	c.parent, c.theme = parent, theme
	c.rt.late = true
	return w
}

func (f *listFrame) isHeader(i int) bool { return f.s.Header != nil && f.s.Header(i) }

// cursor returns where the keys move the choice from, or nil when the list
// does not choose rows: *Selected, or the row last chosen in a list
// choosing several without it.
func (s *ListState) cursor() *int {
	switch {
	case s.Selected != nil:
		return s.Selected
	case s.Selection != nil:
		return &s.lead
	}
	return nil
}

// key returns the key of row i, its index without ListState.Key.
// rowDrag is rows of a list being dragged to another place in it
// (ListState.Reorder).
type rowDrag struct {
	s    *ListState
	rows []int
}

func (d *rowDrag) dragCount() int { return len(d.rows) }

// dragged returns the rows a drag from row i moves: those chosen when it
// is one of them, else row i.
func (f *listFrame) dragged(i int) []int {
	s := f.s
	if s.Selection == nil || !f.chosen(i, f.key(i)) {
		return []int{i}
	}
	var rows []int
	for j := range f.n {
		if !f.isHeader(j) && f.chosen(j, f.key(j)) {
			rows = append(rows, j)
		}
	}
	return rows
}

// reorder makes the list take its own rows dragged over it: it shows where
// they would go, and moves them there as they drop (ListState.Reorder).
func (f *listFrame) reorder() {
	s, e := f.s, f.e
	if s.Reorder == nil {
		return
	}
	e.flags |= flagValueDrop
	e.st.accepts = func(v any) bool { d, ok := v.(*rowDrag); return ok && d.s == s }
	rt := f.c.rt
	var d *rowDrag
	dropped, y := false, rt.pointerY
	switch {
	case e.st.hasDropped:
		// Where the rows dropped, which the pointer may have left since,
		// as when Windows moves it back to the mouse after a test's drag.
		d, _ = e.st.droppedValue.(*rowDrag)
		e.st.hasDropped, dropped, y = false, true, e.st.dropY
	case rt.drag != nil && !rt.drag.canceled && rt.drag.over == e.id:
		d, _ = rt.drag.value.(*rowDrag)
	}
	if d == nil || d.s != s {
		return
	}
	x := rt.pointerX
	if dropped {
		x = e.st.dropX
	}
	_, y = e.st.local(x, y)
	y += e.st.y
	to := f.dropAt(y)
	if dropped {
		// Rows already there stay.
		if !(len(d.rows) == 1 && (to == d.rows[0] || to == d.rows[0]+1)) {
			s.Reorder(d.rows, to)
			rt.consumed = true
		}
		return
	}
	t := f.c.theme
	e.DrawOver(func(p *Painter, r Rect) {
		y, ok := f.boundary(to)
		if !ok {
			return
		}
		x := r.X + e.contentX()
		p.Fill(Rect{x + 4, y - 1, r.W - e.padX() - 4, 2}, t.Accent, 1)
		p.Fill(Rect{x, y - 4, 8, 8}, t.Accent, 4)
	})
}

// dropAt returns the row before which rows dropped at y go: that of the
// row of the last frame whose middle is below y, the number of rows past
// the last.
func (f *listFrame) dropAt(y float32) int {
	to, below := -1, -1
	for id, r := range f.s.rows {
		st := f.c.rt.states[id]
		if st == nil || f.isHeader(r.row) {
			continue
		}
		if y < st.y+st.h/2 {
			if to < 0 || r.row < to {
				to = r.row
			}
		} else if r.row > below {
			below = r.row
		}
	}
	if to < 0 {
		return min(below+1, f.n)
	}
	return to
}

// boundary returns where row i starts, in this frame, or the end of the
// row before it, for the last.
func (f *listFrame) boundary(i int) (float32, bool) {
	for _, r := range f.rows {
		if r.i == i {
			return r.e.y - max(f.e.gapY, 0)/2, true
		}
	}
	for _, r := range f.rows {
		if r.i == i-1 {
			return r.e.y + r.e.h + max(f.e.gapY, 0)/2, true
		}
	}
	return 0, false
}

func (f *listFrame) key(i int) any {
	if f.s.Key != nil {
		return f.s.Key(i)
	}
	return i
}

// chosen reports whether row i, of key k, is chosen.
func (f *listFrame) chosen(i int, k any) bool {
	if sel := f.s.Selection; sel != nil {
		return sel.has(k)
	}
	return *f.s.Selected == i
}

// changed reports a new choice.
func (f *listFrame) changed() {
	f.owner.st.markChanged()
	f.c.rt.consumed = true
}

// click chooses row i as a click with mods does.
func (f *listFrame) click(i int, mods Modifiers) {
	switch {
	case f.s.Selection == nil:
		f.choose(i)
	case mods&Shift != 0:
		f.extend(i, mods&Cmd != 0)
	case mods&Cmd != 0:
		f.toggle(i)
	default:
		f.choose(i)
	}
}

// choose makes row i the choice, alone, and shows it.
func (f *listFrame) choose(i int) {
	if sel := f.s.Selection; sel != nil {
		k := f.key(i)
		if sel.size() != 1 || !sel.has(k) {
			sel.clear()
			sel.set(k, true)
			f.changed()
		}
	}
	f.lead(i, true)
}

// toggle adds row i to the choice or takes it out, and shows it.
func (f *listFrame) toggle(i int) {
	k := f.key(i)
	f.s.Selection.set(k, !f.s.Selection.has(k))
	f.changed()
	f.lead(i, true)
}

// extend chooses the rows from the pivot to row i, as Shift does: on
// macOS in place of those it last chose, elsewhere in place of all others
// unless keep.
func (f *listFrame) extend(i int, keep bool) {
	s, sel := f.s, f.s.Selection
	p := s.pivot
	if p < 0 || p >= f.n || f.isHeader(p) {
		f.choose(i)
		return
	}
	lo, hi := min(p, i), max(p, i)
	changed := false
	switch {
	case runtime.GOOS == "darwin":
		last := max(0, min(p+s.span, f.n-1))
		for j := min(p, last); j <= max(p, last); j++ {
			if (j < lo || j > hi) && !f.isHeader(j) {
				changed = sel.set(f.key(j), false) || changed
			}
		}
	case !keep:
		in := 0
		for j := lo; j <= hi; j++ {
			if !f.isHeader(j) && sel.has(f.key(j)) {
				in++
			}
		}
		if sel.size() > in {
			changed = sel.clear()
		}
	}
	for j := lo; j <= hi; j++ {
		if !f.isHeader(j) {
			changed = sel.set(f.key(j), true) || changed
		}
	}
	if changed {
		f.changed()
	}
	s.span = i - p
	f.lead(i, false)
}

// lead makes row i the one the keys move from, and with pivot the one
// Shift extends the choice from, and shows it.
func (f *listFrame) lead(i int, pivot bool) {
	s := f.s
	if sel := s.cursor(); *sel != i {
		*sel = i
		if s.Selection == nil || s.Selected != nil {
			f.changed()
		}
		f.c.rt.consumed = true // the rows show where it is
	}
	if pivot {
		s.pivot, s.span = i, 0
	}
	s.ScrollIntoView(i)
}

// step returns the first row from i on by d that is not a header, or -1.
func (f *listFrame) step(i, d int) int {
	for i += d; i >= 0 && i < f.n; i += d {
		if !f.isHeader(i) {
			return i
		}
	}
	return -1
}

// page returns the row a page down from the one the keys move from (up
// for d -1): the last row in view, or a page further from there.
func (f *listFrame) page(d int) int {
	s, n := f.s, f.n
	i := *s.cursor()
	view := float64(f.e.st.h)
	if i < 0 || i >= n || !s.laidOut || view <= 0 {
		return f.step(-1, 1)
	}
	to := s.heights.rowAt(max(0, s.heights.top(i, s.gap)+float64(d)*view), s.gap)
	switch {
	case d > 0 && i >= s.first && i < s.last:
		to = s.last
	case d < 0 && i > s.first && i <= s.last:
		to = s.first
	}
	to = max(0, min(to, n-1))
	if f.isHeader(to) {
		if j := f.step(to, d); j >= 0 {
			return j
		}
		return f.step(to, -d)
	}
	return to
}

// navigate moves the choice with the keys, past headers.
func (f *listFrame) navigate() {
	o, n, sel, s := f.owner, f.n, f.s.cursor(), f.s
	multi, mac := s.Selection != nil, runtime.GOOS == "darwin"
	if s.Label != nil {
		o.flags |= flagTypeSelect
	}
	if n == 0 {
		return
	}
	// Every key is asked for in every frame, which keeps it.
	const (
		move = iota + 1
		extend
		cursor
	)
	keys := []Key{KeyDown, KeyUp, KeyHome, KeyEnd, KeyPageDown, KeyPageUp}
	if mac {
		keys = keys[:4] // the others scroll, on macOS
	}
	how, key := 0, Key(0)
	for _, k := range keys {
		if o.Shortcut(0, k) {
			how, key = move, k
		}
		if multi && o.Shortcut(Shift, k) {
			how, key = extend, k
		}
		if multi && !mac && o.Shortcut(Ctrl, k) {
			how, key = cursor, k
		}
	}
	all := multi && o.Shortcut(Cmd, KeyA)
	flip := multi && !mac && o.Shortcut(Ctrl, KeySpace)
	in := *sel >= 0 && *sel < n
	// Enter on macOS, as in Finder, and F2 elsewhere edit the text of the
	// row chosen in place (EditableText), where it has one.
	enter := o.Shortcut(0, KeyEnter)
	edit := s.editables && (o.Shortcut(0, KeyF2) || mac && enter)
	switch {
	case edit && in && !f.isHeader(*sel):
		s.editKey, s.editAsked = f.key(*sel), true
		s.ScrollIntoView(*sel)
	case enter && in:
		o.st.markSubmitted()
	}
	to := -1
	switch key {
	case KeyDown:
		to = f.step(max(*sel, -1), 1)
	case KeyUp:
		if to = f.step(*sel, -1); !in {
			to = f.step(-1, 1)
		}
	case KeyHome:
		to = f.step(-1, 1)
	case KeyEnd:
		to = f.step(n, -1)
	case KeyPageDown:
		to = f.page(1)
	case KeyPageUp:
		to = f.page(-1)
	}
	switch {
	case how == 0:
	case to < 0 && in:
		// Nowhere further: the choice shows again, if it scrolled away.
		s.ScrollIntoView(*sel)
	case to < 0:
	case how == move:
		f.choose(to)
	case how == extend:
		f.extend(to, false)
	case how == cursor:
		f.lead(to, false)
	}
	if all {
		changed := false
		for j := range n {
			if !f.isHeader(j) {
				changed = s.Selection.set(f.key(j), true) || changed
			}
		}
		if changed {
			f.changed()
		}
	}
	if flip && *sel >= 0 && *sel < n && !f.isHeader(*sel) {
		f.toggle(*sel)
	}
	if o.st.typing && s.Label != nil {
		if j := f.typed(o.st.typed); j >= 0 {
			f.choose(j)
		}
	}
}

// typed returns the row whose text starts with what was typed, from the
// row the keys move from on, or -1: the same letter again goes to the
// next row starting with it.
func (f *listFrame) typed(text string) int {
	n, cur := f.n, *f.s.cursor()
	if text == "" {
		return -1
	}
	from := max(cur, 0)
	if strings.Count(text, text[:1]) == len(text) {
		text = text[:1]
		if cur >= 0 {
			from = cur + 1
		}
	}
	for d := range n {
		i := (from + d) % n
		if l := f.s.Label(i); len(l) >= len(text) && strings.EqualFold(l[:len(text)], text) && !f.isHeader(i) {
			return i
		}
	}
	return -1
}

// focusedRow returns the row holding the keyboard focus in the last
// frame, where it is now.
func (s *ListState) focusedRow(e *node, n int) (int, bool) {
	rt := e.c.rt
	if rt.focused == 0 || len(s.rows) == 0 {
		return 0, false
	}
	for st := rt.states[rt.focused]; st != nil && st.id != e.id && st.parent != 0; st = rt.states[st.parent] {
		if st.parent != e.id {
			continue
		}
		r, ok := s.rows[st.id]
		if !ok {
			return 0, false
		}
		i := r.row
		if s.Key != nil && (i >= n || s.Key(i) != r.key) {
			if i, ok = s.find(r.key, i, n); !ok {
				return 0, false
			}
		}
		return i, i < n
	}
	return 0, false
}

// layoutList measures and places the rows of a List w×h from its place,
// building those missing, pins the header of the section at the top, and
// sets the size of the content and the offset. Without rows, what else was
// built in the list shows, as a scroll container's content: a message that
// it is empty, say.
func (e *node) layoutList(w, h float32) {
	f := e.list
	s, n := f.s, f.n
	hs := &s.heights
	st := e.st
	cw := max(w-e.padX(), 0)
	if e.flags&flagScrollX != 0 {
		// Rows at least as wide as they ask, as a table's columns, which
		// scroll sideways.
		cw = max(cw, e.rowMinW)
	}
	if g := e.gridFit; g != nil && max(1, int(math.Floor(float64((cw+g.gap)/(g.minW+g.gap))))) != g.cols {
		// The items of a grid take other columns: the next frame builds
		// them so.
		e.c.rt.animating = true
	}
	if cw != s.width {
		// The rows wrap anew: heights measured at another width are gone.
		hs.clear()
		s.width = cw
	}
	gap := float64(max(e.gapY, 0))
	padTop, padBottom := float64(e.contentY()), float64(e.pad[2]+e.border[2])
	H := float64(h)
	top, bottom := float64(e.border[0]), H-float64(e.border[2])
	s.gap, s.padTop, s.border = gap, padTop, top
	st.list = s
	if n == 0 {
		_, uh := boxLayout(e, cw, inf, true)
		e.contentW, e.contentH = float64(cw+e.padX()), float64(max(uh, max(h-e.padY(), 0))+e.padY())
		layoutAbsolute(e)
		e.scrollBase = 0
		s.remember(e, 0, -1, true, st.scrollY)
		return
	}
	if f.at == nil {
		f.at = map[int]int{}
	}
	clear(f.at)
	for k := range f.rows {
		r := &f.rows[k]
		f.at[r.i] = k
		hs.set(r.i, heightAt(r.e, cw, inf))
	}
	// ensure builds and measures row i if the list lacks it.
	ensure := func(i int) bool {
		if _, ok := f.at[i]; ok {
			return true
		}
		if len(f.rows) >= listMaxRows {
			return false
		}
		r := f.buildLate(i)
		f.at[i] = len(f.rows) - 1
		hs.set(i, heightAt(r, cw, inf))
		return true
	}
	ht := hs.height

	// The row placed first, and where its top goes.
	var a int
	var ya float64
	switch req := s.req; {
	case req.end || s.following && !req.set:
		a = n - 1
		ensure(a)
		ya = H - padBottom - ht(a)
	case req.offset:
		// Where the heights known, now with the rows built measured, put
		// the offset.
		a, ya = s.anchorAt(req.y)
		if req.hint {
			a = max(0, min(req.row, n-1))
			ya = padTop + hs.top(a, gap) - req.y
		}
		ensure(a)
	case req.set:
		a = max(0, min(req.row, n-1))
		ensure(a)
		size := ht(a)
		// The header pinned over the row's section hides the top of the
		// list: the row shows below it.
		shown, start := top, padTop
		if s.Header != nil && !s.Header(a) {
			if hh := s.headerAbove(a, n); hh >= 0 && ensure(hh) {
				shown = top + ht(hh)
				start = max(start, shown)
			}
		}
		switch {
		case req.near:
			// Where the place puts it now.
			ya = padTop + hs.top(a, gap) - (hs.top(s.anchor, gap) + s.inset)
			switch {
			case ya < shown || size > bottom-shown:
				ya = start
			case ya+size > bottom:
				ya = H - padBottom - size
			}
		case req.align == Center:
			ya = (shown + bottom - size) / 2
		case req.align == End:
			ya = H - padBottom - size
		default:
			ya = start
		}
	default:
		a = s.anchor
		ensure(a)
		ya = padTop - s.inset
	}

	// Rows lo to hi are placed, row lo at yLo and row hi at yHi.
	lo, hi, yLo, yHi := a, a, ya, ya
	cover := func() {
		for hi < n-1 && yHi+ht(hi)+gap < bottom && ensure(hi+1) {
			yHi += ht(hi) + gap
			hi++
		}
		for lo > 0 && yLo > top && ensure(lo-1) {
			lo--
			yLo -= ht(lo) + gap
		}
	}
	shift := func(d float64) { yLo, yHi = yLo+d, yHi+d }
	end := func() float64 { return yHi + ht(hi) + padBottom }
	cover()
	// Keep the content's ends at the list's: the end may show above the
	// bottom after rows went away, the start below the top after rows
	// above turned out shorter than estimated.
	for k := 0; k < 4 && !(lo == 0 && hi == n-1); k++ {
		if hi == n-1 && end() < H {
			shift(H - end())
		} else if lo == 0 && yLo > padTop {
			shift(padTop - yLo)
		} else {
			break
		}
		cover()
	}
	if lo == 0 && hi == n-1 {
		switch start := yLo - padTop; {
		case end()-start <= H:
			// The rows fit: Justify places them.
			switch e.justify {
			case End:
				shift(H - end())
			case Center:
				shift((H - end() - start) / 2)
			default:
				shift(-start)
			}
		case start > 0:
			shift(-start)
		case end() < H:
			shift(H - end())
		}
	}
	glide := s.glideToEnd(e, n, lo, hi, H, func(d float64) { shift(d); cover() })
	for k := 0; k < listOverscan && hi < n-1 && ensure(hi+1); k++ {
		yHi += ht(hi) + gap
		hi++
	}
	for k := 0; k < listOverscan && lo > 0 && ensure(lo-1); k++ {
		lo--
		yLo -= ht(lo) + gap
	}

	// The offset: where the heights known put row lo's top, less where it
	// shows.
	content := max(padTop+padBottom+hs.top(n, gap)-gap, H)
	scroll := max(0, min(padTop+hs.top(lo, gap)-yLo, content-H))
	y := yLo
	for i := lo; i <= hi; i++ {
		f.rows[f.at[i]].y = y
		y += ht(i) + gap
	}
	for k := range f.rows {
		if r := &f.rows[k]; r.i < lo || r.i > hi {
			r.y = padTop + hs.top(r.i, gap) - scroll
		}
	}

	// The rows in view, the first of which anchors the place.
	first, last := -1, -1
	for i := lo; i <= hi; i++ {
		if y := f.rows[f.at[i]].y; y+ht(i) > top && y < bottom {
			if first < 0 {
				first = i
			}
			last = i
		}
	}
	if first < 0 {
		first, last = a, a-1
	}
	inset := padTop - f.rows[f.at[first]].y
	var pinned *node
	if s.Header != nil && last >= first {
		pinned = e.pinHeader(first, hi, top)
	}

	// Lay the rows out in order, as they show, Tab moves and assistive
	// technology reads, then what else was built above them, which lies
	// over the rows.
	var extras []*node
	for ch := e.first; ch != nil; ch = ch.next {
		if !ch.listRow && ch.flags&flagAbsolute != 0 {
			extras = append(extras, ch)
		}
	}
	slices.SortFunc(f.rows, func(a, b listRow) int { return a.i - b.i })
	var prev *node
	link := func(ch *node) {
		ch.next = nil
		if prev == nil {
			e.first = ch
		} else {
			prev.next = ch
		}
		prev = ch
	}
	for k := range f.rows {
		r := &f.rows[k]
		link(r.e)
		r.e.x, r.e.y = e.contentX(), float32(r.y)
		layoutBox(r.e, cw, float32(ht(r.i)))
	}
	for _, ch := range extras {
		link(ch)
	}
	e.last = prev
	layoutAbsolute(e)
	if pinned != nil {
		// Above the rows scrolling under it, and still at the top when the
		// offset moves before placing.
		pinned.flags |= flagAbsolute
	}
	e.contentW, e.contentH = float64(cw+e.padX()), content
	e.scrollBase = scroll
	st.scrollTo(st.scrollX, scroll)
	s.inset = inset
	s.remember(e, first, last, hi == n-1 && end()-glide <= H+0.5, scroll)
}

// glideToEnd moves the rows of a list following its end with a
// FollowAnimation below their place, by as much as the end moved down
// since the frame before, and on toward their place, as the animation
// says; it returns how far below it they show. Rows lo to hi are placed,
// and shift moves them. What moved is measured in the content, so that
// the list growing or shrinking moves nothing.
func (s *ListState) glideToEnd(e *node, n, lo, hi int, H float64, shift func(float64)) float64 {
	rt, now := e.c.rt, e.c.now
	if s.FollowAnimation == nil || !s.following || s.req.set || hi != n-1 || rt.preferences().ReduceMotion || rt.instant() {
		s.followLag, s.followSeen = channel{}, false
		return 0
	}
	hs, gap := &s.heights, s.gap
	below := func(i int) float64 { return hs.top(n, gap) - hs.top(i, gap) }
	ch := &s.followLag
	if ch.at.IsZero() {
		ch.at = now
	}
	ch.advance(now)
	if s.followSeen && s.width == s.followWidth {
		// The row last at the end: as much as came below it, the rows
		// come up from below; as much as went, the glide takes back.
		prev, ok := s.followRow, s.followRow < n
		if s.Key != nil && s.followKey != nil {
			prev, ok = s.find(s.followKey, s.followRow, n)
		}
		if !ok {
			*ch = channel{at: now}
		} else if d := below(prev) - s.followBelow; d > 0.5 {
			ch.value = min(ch.value+d, H)
			ch.retarget(0, *s.FollowAnimation)
		} else if d < -0.5 {
			// The rows stay where they show, above their place, and go
			// down to it.
			ch.value = max(ch.value+d, -H)
			ch.retarget(0, *s.FollowAnimation)
		}
	}
	s.followSeen, s.followRow, s.followBelow, s.followWidth, s.followKey = true, n-1, below(n-1), s.width, nil
	if s.Key != nil {
		s.followKey = s.Key(n - 1)
	}
	if ch.moving || ch.pending {
		rt.animating = true
	}
	lag := ch.value
	if lag > 0.01 || lag < -0.01 {
		shift(lag)
	}
	return lag
}

// relayoutList lays the list out anew scrolled to y, as revealing its
// row ch asks, from where the heights known put ch: the rows around it
// are built, and the frame shows no gap.
func (e *node) relayoutList(ch *node, y float64) {
	f := e.list
	if f.n == 0 {
		e.st.scrollTo(e.st.scrollX, y)
		return
	}
	req := listRequest{set: true, offset: true, y: y}
	for _, r := range f.rows {
		if r.e == ch {
			req.row, req.hint = r.i, true
		}
	}
	f.s.req = req
	layoutBox(e, e.w, e.h)
}

// remember keeps where the list is for the next frame, and builds another
// for a view that read where it was.
func (s *ListState) remember(e *node, first, last int, atEnd bool, scroll float64) {
	f, n := e.list, e.list.n
	if s.read && (first != s.first || last != s.last || atEnd != s.atEnd) {
		e.c.rt.animating = true
	}
	s.laidOut, s.wroteY, s.req, s.read = true, scroll, listRequest{}, false
	s.first, s.last, s.atEnd = first, last, atEnd
	s.following = s.FollowEnd && atEnd
	s.anchor, s.anchorKey = max(0, first), nil
	if n == 0 {
		s.inset = 0
	} else if s.Key != nil && !s.KeepOffset {
		s.anchorKey = s.Key(first)
	}
	s.selKey, s.pivotKey = nil, nil
	if sel := s.cursor(); sel != nil {
		s.selRow = *sel
		if s.Key != nil && *sel >= 0 && *sel < n {
			s.selKey = s.Key(*sel)
		}
	}
	if s.Key != nil && s.Selection != nil && s.pivot >= 0 && s.pivot < n {
		s.pivotKey = s.Key(s.pivot)
	}
	if s.rows == nil {
		s.rows = map[uint64]listRowID{}
	}
	clear(s.rows)
	for _, r := range f.rows {
		s.rows[r.e.id] = listRowID{r.i, r.key}
	}
}

// pinHeader moves the header of the section of row first, the first in
// view, to the top of the list, below which the next header pushes it,
// and returns it if it moved.
func (e *node) pinHeader(first, hi int, top float64) *node {
	f := e.list
	s := f.s
	hh := s.headerAbove(first, f.n)
	if hh < 0 {
		return nil
	}
	k, ok := f.at[hh]
	if !ok {
		// Far above: build it where its section's rows are.
		f.buildLate(hh)
		k = len(f.rows) - 1
		f.at[hh] = k
		s.heights.set(hh, heightAt(f.rows[k].e, s.width, inf))
		f.rows[k].y = f.rows[f.at[first]].y - (s.heights.top(first, s.gap) - s.heights.top(hh, s.gap))
	}
	size := s.heights.height(hh)
	y := max(f.rows[k].y, top)
	for j := max(hh, first) + 1; j <= hi; j++ {
		if s.Header(j) {
			y = min(y, f.rows[f.at[j]].y-size)
			break
		}
	}
	if y == f.rows[k].y {
		return nil
	}
	f.rows[k].y = y
	return f.rows[k].e
}

// listHeader is the header found last for a row, for the next frame to
// look from there.
type listHeader struct {
	// row is the last header at or above row from, -1 for none, and the
	// rows from stop to from were looked at: none between row and from
	// is a header, and none down to stop either when row is -1.
	row, from, stop int
	ok              bool
}

// headerAbove returns the last header at or above row i, -1 for none.
func (s *ListState) headerAbove(i, n int) int {
	const far = 100_000
	hc := &s.header
	if hc.ok && (hc.row < 0 || hc.row < n && s.Header(hc.row)) {
		switch {
		case i >= hc.from && i-hc.from <= far:
			for j := i; j > hc.from; j-- {
				if s.Header(j) {
					*hc = listHeader{row: j, from: i, stop: j, ok: true}
					return j
				}
			}
			hc.from = i
			return hc.row
		case i >= hc.stop && (hc.row >= 0 || hc.stop == 0):
			return hc.row
		}
	}
	h, stop := -1, max(0, i-far)
	for j := i; j >= stop; j-- {
		if s.Header(j) {
			h, stop = j, j
			break
		}
	}
	*hc = listHeader{row: h, from: i, stop: stop, ok: true}
	return h
}

// listBlock is how many rows' heights a block of listHeights holds.
const listBlock = 64

// listHeights holds the heights measured of a list's rows, in blocks of
// listBlock rows made as rows are measured, so that its memory follows
// the rows measured rather than the rows, and estimates the others as
// their average. The offset of a row sums the totals of the blocks
// before it, summed in order once they change, and its block's heights.
type listHeights struct {
	n int
	// ids are the indexes of the blocks with measured rows, in order.
	ids    []int
	blocks []*heightBlock
	// def is the estimate while no row is measured.
	def float64
	// psum and pcount sum the heights and the counts of measured rows of
	// the blocks before each, up to date unless dirty.
	psum   []float64
	pcount []int
	dirty  bool
}

type heightBlock struct {
	h     [listBlock]float32
	known uint64
	sum   float64
}

// block returns where block b is or would go among the blocks, and
// whether it is there.
func (hs *listHeights) block(b int) (int, bool) {
	return slices.BinarySearch(hs.ids, b)
}

func (hs *listHeights) index() {
	if !hs.dirty && len(hs.psum) == len(hs.blocks)+1 {
		return
	}
	m := len(hs.blocks)
	hs.psum = slices.Grow(hs.psum[:0], m+1)[:m+1]
	hs.pcount = slices.Grow(hs.pcount[:0], m+1)[:m+1]
	hs.psum[0], hs.pcount[0] = 0, 0
	for k, blk := range hs.blocks {
		hs.psum[k+1] = hs.psum[k] + blk.sum
		hs.pcount[k+1] = hs.pcount[k] + bits.OnesCount64(blk.known)
	}
	hs.dirty = false
}

// estimate returns the height of rows not measured.
func (hs *listHeights) estimate() float64 {
	hs.index()
	if k := hs.pcount[len(hs.blocks)]; k > 0 {
		return hs.psum[len(hs.blocks)] / float64(k)
	}
	return hs.def
}

// height returns the height of row i, measured or estimated.
func (hs *listHeights) height(i int) float64 {
	if k, ok := hs.block(i / listBlock); ok {
		if blk := hs.blocks[k]; blk.known&(1<<(i%listBlock)) != 0 {
			return float64(blk.h[i%listBlock])
		}
	}
	return hs.estimate()
}

// top returns how far below the top of row 0 row i starts, with gap
// between rows.
func (hs *listHeights) top(i int, gap float64) float64 {
	i = max(0, min(i, hs.n))
	est := hs.estimate()
	b, j := i/listBlock, i%listBlock
	k, ok := hs.block(b)
	y := hs.psum[k] + float64(b*listBlock-hs.pcount[k])*est
	if ok {
		blk := hs.blocks[k]
		for at := range j {
			if blk.known&(1<<at) != 0 {
				y += float64(blk.h[at])
			} else {
				y += est
			}
		}
	} else {
		y += float64(j) * est
	}
	return y + float64(i)*gap
}

// rowAt returns the row at y below the top of row 0, with its gap.
func (hs *listHeights) rowAt(y, gap float64) int {
	if hs.n == 0 {
		return 0
	}
	lo, hi := 0, (hs.n-1)/listBlock
	for lo < hi {
		if mid := (lo + hi + 1) / 2; hs.top(mid*listBlock, gap) <= y {
			lo = mid
		} else {
			hi = mid - 1
		}
	}
	i := lo * listBlock
	end := min(hs.n, i+listBlock)
	for at := hs.top(i, gap); i < end-1; i++ {
		if at += hs.height(i) + gap; at > y {
			break
		}
	}
	return i
}

// set records that row i measured h.
func (hs *listHeights) set(i int, h float32) {
	if i < 0 || i >= hs.n {
		return
	}
	b, j := i/listBlock, i%listBlock
	k, ok := hs.block(b)
	if !ok {
		hs.ids = slices.Insert(hs.ids, k, b)
		hs.blocks = slices.Insert(hs.blocks, k, new(heightBlock))
	}
	blk := hs.blocks[k]
	bit := uint64(1) << j
	if blk.known&bit != 0 && blk.h[j] == h {
		return
	}
	blk.known |= bit
	blk.h[j] = h
	blk.total()
	hs.dirty = true
}

func (blk *heightBlock) total() {
	blk.sum = 0
	for k := range listBlock {
		if blk.known&(1<<k) != 0 {
			blk.sum += float64(blk.h[k])
		}
	}
}

// resize makes it n rows, forgetting those past them.
func (hs *listHeights) resize(n int) {
	nb := (n + listBlock - 1) / listBlock
	k, _ := hs.block(nb)
	clear(hs.blocks[k:])
	hs.ids, hs.blocks = hs.ids[:k], hs.blocks[:k]
	if k, ok := hs.block(nb - 1); ok && n%listBlock > 0 {
		blk := hs.blocks[k]
		blk.known &= 1<<(n%listBlock) - 1
		blk.total()
	}
	hs.n = n
	hs.dirty = true
}

// shift moves the heights known by d rows, as rows came or went before.
func (hs *listHeights) shift(d int) {
	ids, blocks := hs.ids, hs.blocks
	hs.ids, hs.blocks = nil, nil
	for k, blk := range blocks {
		for at := range listBlock {
			if blk.known&(1<<at) != 0 {
				hs.set(ids[k]*listBlock+at+d, blk.h[at])
			}
		}
	}
	hs.dirty = true
}

// clear forgets every height.
func (hs *listHeights) clear() {
	clear(hs.blocks)
	hs.ids, hs.blocks = hs.ids[:0], hs.blocks[:0]
	hs.dirty = true
}
