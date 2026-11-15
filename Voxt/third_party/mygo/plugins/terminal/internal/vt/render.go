package vt

import "unsafe"

// RenderState is a GhosttyRenderState: what a frame draws of a terminal,
// copied from it by an update, with the rows that changed since the last
// frame marked dirty.
type RenderState struct {
	h     uintptr
	rows  uintptr // the row iterator
	cells uintptr // the cells of the iterator's row
	// raw holds the cells of the row read last.
	raw []Cell
}

// Render state data (GhosttyRenderStateData and its row and cell kinds).
const (
	rsDataCols        = 1
	rsDataRows        = 2
	rsDataDirty       = 3
	rsDataRowIterator = 4
	rsDataCursor      = 18
	rsDataColors      = 19

	rowDataDirty     = 1
	rowDataCells     = 3
	rowDataSelection = 4
	rowDataCellsRaw  = 5
	rowDataID        = 7

	cellDataStyle        = 2
	cellDataGraphemesLen = 3
	cellDataGraphemesBuf = 4
)

// NewRenderState makes an empty render state.
func NewRenderState() (*RenderState, error) {
	r := &RenderState{}
	if err := result(call(fnRenderStateNew, 0, uintptr(unsafe.Pointer(&r.h)))); err != nil {
		return nil, err
	}
	if err := result(call(fnRowIteratorNew, 0, uintptr(unsafe.Pointer(&r.rows)))); err != nil {
		r.Free()
		return nil, err
	}
	if err := result(call(fnRowCellsNew, 0, uintptr(unsafe.Pointer(&r.cells)))); err != nil {
		r.Free()
		return nil, err
	}
	return r, nil
}

// Free frees the render state.
func (r *RenderState) Free() {
	if r.cells != 0 {
		call(fnRowCellsFree, r.cells)
	}
	if r.rows != 0 {
		call(fnRowIteratorFree, r.rows)
	}
	if r.h != 0 {
		call(fnRenderStateFree, r.h)
	}
	*r = RenderState{}
}

// BeginUpdate copies the terminal's state, the only step that needs the
// terminal; EndUpdate finishes the update without it.
func (r *RenderState) BeginUpdate(t *Terminal) error {
	return result(call(fnRenderStateBeginUpdate, r.h, t.h))
}

// EndUpdate finishes the update BeginUpdate started.
func (r *RenderState) EndUpdate() { call(fnRenderStateEndUpdate, r.h) }

// Update updates the render state from the terminal in one step.
func (r *RenderState) Update(t *Terminal) error {
	if err := r.BeginUpdate(t); err != nil {
		return err
	}
	r.EndUpdate()
	return nil
}

// Clean marks everything drawn.
func (r *RenderState) Clean() { call(fnRenderStateClean, r.h) }

func (r *RenderState) get(data int, out unsafe.Pointer) error {
	return result(call(fnRenderStateGet, r.h, uintptr(data), uintptr(out)))
}

// Dirty is how much of a frame changed.
type Dirty int32

const (
	Clean Dirty = iota
	Partial
	Full
)

// Dirty reports how much changed since the last Clean.
func (r *RenderState) Dirty() Dirty {
	var d Dirty
	r.get(rsDataDirty, unsafe.Pointer(&d))
	return d
}

// Size returns the size of the viewport in cells.
func (r *RenderState) Size() (cols, rows int) {
	var c, rr uint16
	r.get(rsDataCols, unsafe.Pointer(&c))
	r.get(rsDataRows, unsafe.Pointer(&rr))
	return int(c), int(rr)
}

// Colors are the colors of a frame: those programs set, else the defaults.
type Colors struct {
	Background, Foreground RGB
	// Cursor is the cursor's color when HasCursor is set; it takes the
	// text's otherwise.
	Cursor    RGB
	HasCursor bool
	Palette   [256]RGB
}

// Colors returns the colors of the frame.
func (r *RenderState) Colors() Colors {
	c := renderColors{size: unsafe.Sizeof(renderColors{})}
	r.get(rsDataColors, unsafe.Pointer(&c))
	return Colors{Background: c.background, Foreground: c.foreground, Cursor: c.cursor, HasCursor: c.cursorHasValue, Palette: c.palette}
}

// Cursor is the cursor of a frame.
type Cursor struct {
	// InView reports that the cursor is in the viewport, at column X of
	// row Y; WideTail that it is on the second half of a wide character.
	InView   bool
	X, Y     int
	WideTail bool
	// Visible reports that programs show the cursor, Blinking that it
	// blinks, Password that the program reads a password.
	Visible, Blinking, Password bool
	Style                       CursorStyle
}

// Cursor returns the cursor of the frame.
func (r *RenderState) Cursor() Cursor {
	c := renderCursor{size: unsafe.Sizeof(renderCursor{})}
	r.get(rsDataCursor, unsafe.Pointer(&c))
	return Cursor{
		InView: c.viewportHasValue, X: int(c.viewportX), Y: int(c.viewportY), WideTail: c.wideTail,
		Visible: c.visible, Blinking: c.blinking, Password: c.passwordInput, Style: CursorStyle(c.visualStyle),
	}
}

// Rows starts going through the rows of the frame, top to bottom, with
// NextRow.
func (r *RenderState) Rows() { r.get(rsDataRowIterator, unsafe.Pointer(&r.rows)) }

// NextRow moves to the next row, and reports whether there is one.
func (r *RenderState) NextRow() bool { return cbool(call(fnRowIteratorNext, r.rows)) }

func (r *RenderState) rowGet(data int, out unsafe.Pointer) bool {
	return ok(call(fnRowGet, r.rows, uintptr(data), uintptr(out)))
}

// RowDirty reports whether the row changed since the last Clean.
func (r *RenderState) RowDirty() bool {
	var v bool
	return r.rowGet(rowDataDirty, unsafe.Pointer(&v)) && v
}

// RowID returns the identity of the row, which it keeps as it scrolls.
func (r *RenderState) RowID() RowID {
	var id RowID
	r.rowGet(rowDataID, unsafe.Pointer(&id))
	return id
}

// RowSelection returns the selected columns of the row, from start to end
// inclusive.
func (r *RenderState) RowSelection() (start, end int, ok bool) {
	s := rowSelection{size: unsafe.Sizeof(rowSelection{})}
	if !r.rowGet(rowDataSelection, unsafe.Pointer(&s)) {
		return 0, 0, false
	}
	return int(s.startX), int(s.endX), true
}

// Wide is how wide a cell's character is.
type Wide uint8

const (
	Narrow Wide = iota
	// WideChar takes this cell and the next, a SpacerTail.
	WideChar
	SpacerTail
	// SpacerHead ends a soft-wrapped row before a wide character that did
	// not fit.
	SpacerHead
)

// Cell is a cell of a row.
type Cell struct {
	// Codepoint is the character, 0 for none; Grapheme reports more
	// codepoints after it (CellGraphemes).
	Codepoint rune
	Grapheme  bool
	// Style identifies the cell's style (CellStyle) among the row's
	// cells: 0 is the default.
	Style     uint16
	Wide      Wide
	Hyperlink bool
	// Background is the color of a cell without text whose background
	// a program filled, when HasBackground; Palette tells that it is
	// the palette's color of index Background.R.
	Background    RGB
	HasBackground bool
	Palette       bool
}

// RowCells returns the cells of the row, valid until the next update.
func (r *RenderState) RowCells() []Cell {
	var v cellsView
	if !r.rowGet(rowDataCellsRaw, unsafe.Pointer(&v)) || v.ptr == 0 {
		return nil
	}
	raw := unsafe.Slice((*uint64)(mem(v.ptr)), v.len)
	r.raw = r.raw[:0]
	for _, c := range raw {
		var out Cell
		switch cell.tag.of(c) {
		case 0, 1: // a codepoint, of a grapheme cluster for 1
			out.Codepoint = rune(cell.codepoint.of(c))
			out.Grapheme = cell.tag.of(c) == 1
		case 2:
			out.HasBackground, out.Palette = true, true
			out.Background.R = uint8(cell.paletteIndex.of(c))
		case 3:
			out.HasBackground = true
			out.Background = RGB{uint8(cell.r.of(c)), uint8(cell.g.of(c)), uint8(cell.b.of(c))}
		}
		out.Style = uint16(cell.styleID.of(c))
		out.Wide = Wide(cell.wide.of(c))
		out.Hyperlink = cell.hyperlink.of(c) != 0
		r.raw = append(r.raw, out)
	}
	// The cells of the row for the style and grapheme lookups.
	r.rowGet(rowDataCells, unsafe.Pointer(&r.cells))
	return r.raw
}

// Color is the color of a style: none (the default), an index of the
// palette, or an RGB color.
type Color struct {
	Kind  uint8 // 0 none, 1 palette, 2 RGB
	Index uint8
	RGB   RGB
}

func (c styleColor) color() Color {
	switch c.tag {
	case colorPalette:
		return Color{Kind: 1, Index: c.value[0]}
	case colorRGB:
		return Color{Kind: 2, RGB: RGB{c.value[0], c.value[1], c.value[2]}}
	}
	return Color{}
}

// Underline styles.
const (
	UnderlineNone = iota
	UnderlineSingle
	UnderlineDouble
	UnderlineCurly
	UnderlineDotted
	UnderlineDashed
)

// Style is how a cell's text shows.
type Style struct {
	FG, BG, UnderlineColor                                                  Color
	Bold, Italic, Faint, Blink, Inverse, Invisible, Strikethrough, Overline bool
	Underline                                                               int
}

// CellStyle returns the style of cell x of the row RowCells read.
func (r *RenderState) CellStyle(x int) Style {
	s := style{size: unsafe.Sizeof(style{})}
	if ok(call(fnRowCellsSelect, r.cells, uintptr(x))) {
		call(fnRowCellsGet, r.cells, cellDataStyle, uintptr(unsafe.Pointer(&s)))
	}
	return Style{
		FG: s.fg.color(), BG: s.bg.color(), UnderlineColor: s.underlineColor.color(),
		Bold: s.bold, Italic: s.italic, Faint: s.faint, Blink: s.blink, Inverse: s.inverse,
		Invisible: s.invisible, Strikethrough: s.strikethrough, Overline: s.overline, Underline: int(s.underline),
	}
}

// CellGraphemes appends the codepoints of cell x of the row RowCells read
// to buf.
func (r *RenderState) CellGraphemes(x int, buf []rune) []rune {
	if !ok(call(fnRowCellsSelect, r.cells, uintptr(x))) {
		return buf
	}
	var n uint32
	call(fnRowCellsGet, r.cells, cellDataGraphemesLen, uintptr(unsafe.Pointer(&n)))
	if n == 0 {
		return buf
	}
	cps := make([]uint32, n)
	call(fnRowCellsGet, r.cells, cellDataGraphemesBuf, uintptr(unsafe.Pointer(&cps[0])))
	for _, c := range cps {
		buf = append(buf, rune(c))
	}
	return buf
}
