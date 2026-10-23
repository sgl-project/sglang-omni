package editor

import (
	"slices"
	"sort"
)

// The display model of a code view, as Zed's display map: every line of
// the text is a row, or several when it wraps, or none when hidden (a
// fold, or unchanged lines a diff collapses); blocks are rows between
// lines that are not text, as the deleted lines of a diff, the spacers
// aligning its sides, or an annotation. Rows are as high as the view's
// line height and blocks as high as they ask, so the top of a line is
// arithmetic on a few sparse sums: the extra rows of wrapped lines, kept
// in a Fenwick tree, and the heights blocks add and hidden lines take
// away, kept sorted with their prefix sums. Lines are found at a height
// by bisection, as everything else is measured from the top of a line.

// block is a row of a view that is not a line of its text.
type block struct {
	line   int  // the line it is attached to
	below  bool // below the line, else above it
	height float32
	kind   blockKind
	// For deleted lines (blockDeleted), the lines of the other text it
	// shows; for spacers and separators, what they stand for.
	from, to int
	// id identifies an annotation; hunk the diff hunk it belongs to.
	id   any
	hunk int
	// sticky keeps the block when its line is hidden, as the separator
	// of the lines a diff collapses does.
	sticky bool
}

type blockKind uint8

const (
	blockSpacer blockKind = iota
	blockDeleted
	blockSeparator
	blockAnnotation
	blockHeader
)

// lineSpan is the lines from to to, inclusive.
type lineSpan struct{ from, to int }

type heightEntry struct {
	line  int
	extra float32 // the height of blocks
}

// heights places lines and blocks vertically.
type heights struct {
	lh      float32
	n       int // lines
	hidden  []lineSpan
	blocks  []block // sorted by line, above before below
	entries []heightEntry
	prefix  []float64 // prefix[i] is the extra of entries[:i]
	// hiddenRows[i] is how many rows the hidden spans before i hide.
	hiddenRows []int
	wraps      wrapRows
	wrapsSeen  uint64 // the version of wraps the spans' rows count
	dirty      bool
	// src is the last slice of blocks given, which the next is compared
	// with by identity; scratch sorts them.
	src     []block
	srcLen  int
	scratch []block
}

// wrapRows keeps how many rows beyond the first each line wraps into.
type wrapRows struct {
	on      bool
	version uint64   // changes with any line's rows
	extra   []uint16 // per line; wrapUnknown when not measured
	tree    []int32  // Fenwick tree of the extra rows
	stale   bool     // the tree needs building
	known   int      // lines measured
}

const wrapUnknown = 0xffff

func (w *wrapRows) reset(n int, on bool) {
	w.on = on
	if !on {
		w.extra, w.tree = nil, nil
		return
	}
	w.extra = slices.Grow(w.extra[:0], n)[:n]
	for i := range w.extra {
		w.extra[i] = wrapUnknown
	}
	w.stale, w.known = true, 0
	w.version++
}

func (w *wrapRows) rows(line int) int {
	if !w.on || line < 0 || line >= len(w.extra) || w.extra[line] == wrapUnknown {
		return 1
	}
	return int(w.extra[line]) + 1
}

func (w *wrapRows) set(line, rows int) {
	if !w.on || line < 0 || line >= len(w.extra) {
		return
	}
	old := w.extra[line]
	v := uint16(min(max(rows-1, 0), wrapUnknown-1))
	if old == v {
		return
	}
	w.extra[line] = v
	w.version++
	if old == wrapUnknown {
		w.known++
		old = 0
	}
	if !w.stale {
		for i := line + 1; i <= len(w.extra); i += i & -i {
			w.tree[i] += int32(v) - int32(old)
		}
	}
}

func (w *wrapRows) build() {
	n := len(w.extra)
	w.tree = slices.Grow(w.tree[:0], n+1)[:n+1]
	clear(w.tree)
	for i, e := range w.extra {
		if e != wrapUnknown {
			w.tree[i+1] += int32(e)
		}
	}
	for i := 1; i <= n; i++ {
		if j := i + i&-i; j <= n {
			w.tree[j] += w.tree[i]
		}
	}
	w.stale = false
}

// before returns the extra rows of the lines before line.
func (w *wrapRows) before(line int) int {
	if !w.on {
		return 0
	}
	if w.stale {
		w.build()
	}
	s := 0
	for i := min(line, len(w.extra)); i > 0; i -= i & -i {
		s += int(w.tree[i])
	}
	return s
}

// splice follows a change of lines: the lines from at to at+removed
// replaced by added ones, which are not measured.
func (w *wrapRows) splice(at, removed, added int) {
	if !w.on {
		return
	}
	at = min(at, len(w.extra))
	end := min(at+removed, len(w.extra))
	for _, e := range w.extra[at:end] {
		if e != wrapUnknown {
			w.known--
		}
	}
	w.extra = slices.Delete(w.extra, at, end)
	ins := make([]uint16, added)
	for i := range ins {
		ins[i] = wrapUnknown
	}
	w.extra = slices.Insert(w.extra, at, ins...)
	w.stale = true
	w.version++
}

func (h *heights) setLines(n int) {
	if h.n != n {
		h.n = n
		h.dirty = true
	}
}

func (h *heights) setHidden(spans []lineSpan) {
	if !slices.Equal(h.hidden, spans) {
		h.hidden = append(h.hidden[:0], spans...)
		h.dirty = true
	}
}

func (h *heights) setBlocks(blocks []block) {
	if len(blocks) == len(h.blocks) && (len(blocks) == 0 || &blocks[0] == &h.src[0]) && h.srcLen == len(blocks) {
		return
	}
	h.src, h.srcLen = blocks, len(blocks)
	sorted := append(h.scratch[:0], blocks...)
	sort.SliceStable(sorted, func(i, j int) bool {
		a, b := &sorted[i], &sorted[j]
		if a.line != b.line {
			return a.line < b.line
		}
		return !a.below && b.below
	})
	h.scratch = sorted
	if !slices.EqualFunc(h.blocks, sorted, func(a, b block) bool { return a == b }) {
		h.blocks = append(h.blocks[:0], sorted...)
		h.dirty = true
	}
}

// isHidden reports whether line is hidden.
func (h *heights) isHidden(line int) bool {
	i := sort.Search(len(h.hidden), func(i int) bool { return h.hidden[i].to >= line })
	return i < len(h.hidden) && h.hidden[i].from <= line
}

// update makes the entries again after the hidden lines or the blocks
// changed.
func (h *heights) update() {
	if h.wraps.on && len(h.hidden) > 0 && h.wrapsSeen != h.wraps.version {
		h.dirty = true
	}
	if !h.dirty {
		return
	}
	h.dirty = false
	h.wrapsSeen = h.wraps.version
	h.entries = h.entries[:0]
	for _, b := range h.blocks {
		if b.line >= h.n || !b.sticky && h.isHidden(b.line) {
			continue
		}
		if n := len(h.entries); n > 0 && h.entries[n-1].line == b.line {
			h.entries[n-1].extra += b.height
			continue
		}
		h.entries = append(h.entries, heightEntry{b.line, b.height})
	}
	h.prefix = slices.Grow(h.prefix[:0], len(h.entries)+1)[:len(h.entries)+1]
	h.prefix[0] = 0
	for i, e := range h.entries {
		h.prefix[i+1] = h.prefix[i] + float64(e.extra)
	}
	h.hiddenRows = slices.Grow(h.hiddenRows[:0], len(h.hidden)+1)[:len(h.hidden)+1]
	h.hiddenRows[0] = 0
	for i, s := range h.hidden {
		h.hiddenRows[i+1] = h.hiddenRows[i] + h.rowsOf(s.from, min(s.to+1, h.n))
	}
}

// rowsOf returns the rows of the lines from to to, exclusive.
func (h *heights) rowsOf(from, to int) int {
	if to <= from {
		return 0
	}
	return to - from + h.wraps.before(to) - h.wraps.before(from)
}

// extraBefore returns the height of the blocks of the lines before line.
func (h *heights) extraBefore(line int) float64 {
	i := sort.Search(len(h.entries), func(i int) bool { return h.entries[i].line >= line })
	return h.prefix[i]
}

// hiddenBefore returns the rows of the hidden lines before line.
func (h *heights) hiddenBefore(line int) int {
	i := sort.Search(len(h.hidden), func(i int) bool { return h.hidden[i].to >= line })
	rows := h.hiddenRows[i]
	if i < len(h.hidden) && h.hidden[i].from < line {
		rows += h.rowsOf(h.hidden[i].from, line)
	}
	return rows
}

// top returns the top of line, above its blocks; n gives the bottom of
// the last.
func (h *heights) top(line int) float64 {
	h.update()
	line = max(0, min(line, h.n))
	return float64(h.lh)*float64(line+h.wraps.before(line)-h.hiddenBefore(line)) + h.extraBefore(line)
}

func (h *heights) total() float64 { return h.top(h.n) }

// lineAt returns the line whose rows or blocks hold height y: the last
// line that is not hidden starting at or above it.
func (h *heights) lineAt(y float64) int {
	h.update()
	lo, hi := 0, max(0, h.n-1)
	for lo < hi {
		m := (lo + hi + 1) / 2
		if h.top(m) <= y {
			lo = m
		} else {
			hi = m - 1
		}
	}
	// A hidden line shares its top with the lines after it, up to the
	// next shown.
	for lo > 0 && h.isHidden(lo) {
		i := sort.Search(len(h.hidden), func(i int) bool { return h.hidden[i].to >= lo })
		if s := h.hidden[i]; s.to+1 < h.n && h.top(s.to+1) <= y {
			lo = s.to + 1
			break
		}
		lo = h.hidden[i].from - 1
	}
	return max(0, lo)
}

// blocksOf returns the blocks of line, those above it and those below,
// leaving out those of a hidden line but sticky ones.
func (h *heights) blocksOf(line int) (above, below []block) {
	i := sort.Search(len(h.blocks), func(i int) bool { return h.blocks[i].line >= line })
	j := i
	for j < len(h.blocks) && h.blocks[j].line == line {
		j++
	}
	bs := h.blocks[i:j]
	hidden := h.isHidden(line)
	k := 0
	for k < len(bs) && !bs[k].below {
		k++
	}
	above, below = bs[:k], bs[k:]
	if hidden {
		above, below = stickyBlocks(above), stickyBlocks(below)
	}
	return above, below
}

func stickyBlocks(bs []block) []block {
	var out []block
	for _, b := range bs {
		if b.sticky {
			out = append(out, b)
		}
	}
	return out
}

// lineGeometry is where a line's parts are: the top of its blocks
// above, of its rows and of its blocks below, and its bottom.
type lineGeometry struct {
	top, rows, below, bottom float64
	hidden                   bool
}

func (h *heights) geometry(line int) lineGeometry {
	g := lineGeometry{top: h.top(line)}
	above, below := h.blocksOf(line)
	g.rows = g.top
	for _, b := range above {
		g.rows += float64(b.height)
	}
	g.hidden = h.isHidden(line)
	g.below = g.rows
	if !g.hidden {
		g.below += float64(h.lh) * float64(h.wraps.rows(line))
	}
	g.bottom = g.below
	for _, b := range below {
		g.bottom += float64(b.height)
	}
	return g
}
