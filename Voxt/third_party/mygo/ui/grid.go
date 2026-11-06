package ui

// Track sizes a column or a row of a grid: Fixed, Fr or FitContent.
type Track struct {
	kind trackKind
	v    float32
}

type trackKind uint8

const (
	trackFit trackKind = iota
	trackFixed
	trackFr
)

// Fixed is a track of v DIPs.
func Fixed(v float32) Track { return Track{trackFixed, v} }

// Fr is a track taking a share f of the room the other tracks leave, as
// CSS's minmax(0, <f>fr): tracks of Fr(1) share it equally, even where
// their content is wider. Where the grid's size is not set, as the height
// of a grid that grows with its rows, they are as large as their content
// asks, in proportion to f.
func Fr(f float32) Track { return Track{trackFr, max(f, 0)} }

// FitContent is a track as wide, or as high, as its content, within the
// room there is, as CSS's auto. Tracks beyond those a grid sets, such as
// the rows it adds for its children, fit their content.
func FitContent() Track { return Track{} }

// gridCell is where an element goes in its grid: lines from 1, 0 for
// automatic placement, and spans, 0 for 1 and negative to the last line.
type gridCell struct {
	col, row         int16
	colSpan, rowSpan int16
}

// Grid creates a grid: its children fill its cells row by row, or go where
// ColumnStart and RowStart put them, spanning ColumnSpan columns and
// RowSpan rows. Columns and ColumnTracks set its columns; it adds rows as
// its children need them.
//
//	ui.Grid(c).Columns(3).Gap(12).Children(func() {
//		for _, p := range photos {
//			ui.Image(c, p).AspectRatio(1).Fit(ui.Cover)
//		}
//	})
func coreGrid(c *context) *node { return coreBox(c).Grid() }

// Grid lays the children out in a grid, as the function Grid does.
func (e *node) Grid() *node {
	if e.row && e.align == Center {
		e.align = alignAuto // as Row set it
	}
	e.grid, e.row = true, false
	return e
}

// Columns gives the grid n columns of equal width, Fr(1) each.
func (e *node) Columns(n int) *node {
	e.cols = e.cols[:0]
	for range n {
		e.cols = append(e.cols, Fr(1))
	}
	return e
}

// Rows gives the grid n rows of equal height, Fr(1) each.
func (e *node) Rows(n int) *node {
	e.rows = e.rows[:0]
	for range n {
		e.rows = append(e.rows, Fr(1))
	}
	return e
}

// ColumnTracks sets the grid's columns, as a sidebar and the rest:
//
//	ui.Grid(c).ColumnTracks(ui.Fixed(220), ui.Fr(1))
func (e *node) ColumnTracks(tracks ...Track) *node {
	e.cols = append(e.cols[:0], tracks...)
	return e
}

// RowTracks sets the grid's first rows; those it adds after fit their
// content.
func (e *node) RowTracks(tracks ...Track) *node {
	e.rows = append(e.rows[:0], tracks...)
	return e
}

// JustifyItems places the children of a grid in their cells horizontally:
// Stretch (the default), Start, Center or End. AlignItems places them
// vertically.
func (e *node) JustifyItems(a Align) *node { e.justifyItems = a; return e }

// JustifySelf places the element in its grid cell horizontally, overriding
// its grid's JustifyItems.
func (e *node) JustifySelf(a Align) *node { e.justifySelf = a; return e }

// ColumnStart puts the element in column n of its grid, counting from 1.
func (e *node) ColumnStart(n int) *node { e.cell.col = int16(max(n, 1)); return e }

// RowStart puts the element in row n of its grid, counting from 1.
func (e *node) RowStart(n int) *node { e.cell.row = int16(max(n, 1)); return e }

// ColumnSpan makes the element span n columns of its grid, or, for a
// negative n, every column from its start to the grid's last, as a header
// across a grid does.
func (e *node) ColumnSpan(n int) *node { e.cell.colSpan = int16(n); return e }

// RowSpan makes the element span n rows of its grid, or, for a negative n,
// every row from its start to the last the grid sets.
func (e *node) RowSpan(n int) *node { e.cell.rowSpan = int16(n); return e }

type gridItem struct {
	e                *node
	col, row         int // from 0
	colSpan, rowSpan int
}

// contrib is the size an item asks of the tracks it spans.
type contrib struct {
	start, span int
	min, max    float32
}

// gridScratch holds the items, cells and tracks of the grids being laid
// out, a nested grid's after its parent's, as flexScratch does.
type gridScratch struct {
	items []gridItem
	occ   []bool
	cs    []contrib
	f     []float32
}

// gridPlace places the in-flow children of a grid in its cells, in the
// scratch, and returns them with the number of columns and rows.
func gridPlace(e *node, s *gridScratch) (items []gridItem, nc, nr int) {
	first := len(s.items)
	nc = max(len(e.cols), 1)
	for c := e.first; c != nil; c = c.next {
		if c.flags&flagAbsolute != 0 {
			continue
		}
		it := gridItem{e: c, col: int(c.cell.col) - 1, row: int(c.cell.row) - 1, colSpan: int(c.cell.colSpan), rowSpan: int(c.cell.rowSpan)}
		switch {
		case it.colSpan < 0:
			it.colSpan = max(len(e.cols)-max(it.col, 0), 1)
		case it.colSpan == 0:
			it.colSpan = 1
		}
		switch {
		case it.rowSpan < 0:
			it.rowSpan = max(len(e.rows)-max(it.row, 0), 1)
		case it.rowSpan == 0:
			it.rowSpan = 1
		}
		nc = max(nc, max(it.col, 0)+it.colSpan)
		s.items = append(s.items, it)
	}
	items = s.items[first:]
	occFirst := len(s.occ)
	occupied := func(r, c int) bool {
		i := occFirst + r*nc + c
		return i < len(s.occ) && s.occ[i]
	}
	fits := func(it *gridItem, r, c int) bool {
		for rr := r; rr < r+it.rowSpan; rr++ {
			for cc := c; cc < c+it.colSpan; cc++ {
				if occupied(rr, cc) {
					return false
				}
			}
		}
		return true
	}
	mark := func(it *gridItem, r, c int) {
		it.row, it.col = r, c
		if need := occFirst + (r+it.rowSpan)*nc; need > len(s.occ) {
			s.occ = append(s.occ, make([]bool, need-len(s.occ))...)
		}
		for rr := r; rr < r+it.rowSpan; rr++ {
			for cc := c; cc < c+it.colSpan; cc++ {
				s.occ[occFirst+rr*nc+cc] = true
			}
		}
		nr = max(nr, r+it.rowSpan)
	}
	// As CSS places them: the items placed in both directions, then those
	// in a given row, then the others in order, with a cursor going row
	// by row.
	for i := range items {
		if it := &items[i]; it.row >= 0 && it.col >= 0 {
			mark(it, it.row, it.col)
		}
	}
	for i := range items {
		it := &items[i]
		if it.row < 0 || it.col >= 0 {
			continue
		}
		c := 0
		for c+it.colSpan < nc && !fits(it, it.row, c) {
			c++
		}
		mark(it, it.row, c)
	}
	cr, cc := 0, 0
	for i := range items {
		it := &items[i]
		if it.row >= 0 {
			continue
		}
		if it.col >= 0 {
			if it.col < cc {
				cr++
			}
			cc = it.col
			for !fits(it, cr, cc) {
				cr++
			}
			mark(it, cr, cc)
			continue
		}
		for {
			for cc+it.colSpan <= nc && !fits(it, cr, cc) {
				cc++
			}
			if cc+it.colSpan <= nc {
				break
			}
			cr, cc = cr+1, 0
		}
		mark(it, cr, cc)
		cc += it.colSpan
	}
	s.occ = s.occ[:occFirst]
	return items, nc, max(nr, len(e.rows))
}

// track returns track i of an axis, past the explicit ones FitContent.
func track(tracks []Track, i int) Track {
	if i < len(tracks) {
		return tracks[i]
	}
	return Track{}
}

// sizeTracks sizes n tracks along an axis, of which tracks are explicit,
// with room avail (inf when unknown) and gap between them, from what the
// items ask: their min-content sizes when minContent. Stretch spreads room
// left over FitContent tracks, when there are no Fr ones. It writes the
// sizes in sizes, using limits as scratch.
func sizeTracks(tracks []Track, n int, avail, gap float32, cs []contrib, minContent, stretch bool, sizes, limits []float32) {
	for i := range n {
		t := track(tracks, i)
		sizes[i], limits[i] = 0, 0
		if t.kind == trackFixed {
			sizes[i], limits[i] = t.v, t.v
		}
	}
	spans := func(c contrib, kind trackKind) int {
		k := 0
		for i := c.start; i < c.start+c.span; i++ {
			if track(tracks, i).kind == kind {
				k++
			}
		}
		return k
	}
	// What items ask of the FitContent tracks they span, those spanning
	// one first.
	for _, c := range cs {
		if c.span == 1 && track(tracks, c.start).kind == trackFit {
			sizes[c.start] = max(sizes[c.start], c.min)
			limits[c.start] = max(limits[c.start], c.max)
		}
	}
	for _, c := range cs {
		if c.span == 1 || spans(c, trackFr) > 0 {
			continue
		}
		fits := spans(c, trackFit)
		if fits == 0 {
			continue
		}
		var sum, lim float32
		for i := c.start; i < c.start+c.span; i++ {
			sum += sizes[i]
			lim += max(limits[i], sizes[i])
		}
		sum += gap * float32(c.span-1)
		lim += gap * float32(c.span-1)
		for i := c.start; i < c.start+c.span; i++ {
			if track(tracks, i).kind != trackFit {
				continue
			}
			if c.min > sum {
				sizes[i] += (c.min - sum) / float32(fits)
			}
			if c.max > lim {
				limits[i] = max(limits[i], sizes[i]) + (c.max-lim)/float32(fits)
			}
		}
	}
	var frSum float32
	for i := range n {
		limits[i] = max(limits[i], sizes[i])
		if t := track(tracks, i); t.kind == trackFr {
			frSum += t.v
		}
	}
	if minContent {
		return
	}
	gaps := gap * float32(max(n-1, 0))
	if !finite(avail) {
		// As large as the content asks: FitContent tracks at their limits,
		// Fr tracks by the largest share any item asks.
		var unit float32
		for _, c := range cs {
			var fr, other float32
			for i := c.start; i < c.start+c.span; i++ {
				if t := track(tracks, i); t.kind == trackFr {
					fr += t.v
				} else {
					other += limits[i]
				}
			}
			if fr > 0 {
				unit = max(unit, (c.max-other-gap*float32(c.span-1))/fr)
			}
		}
		for i := range n {
			if t := track(tracks, i); t.kind == trackFr {
				sizes[i] = unit * t.v
			} else {
				sizes[i] = limits[i]
			}
		}
		return
	}
	// FitContent tracks grow to their limits with the room there is,
	// equally.
	free := avail - gaps
	for i := range n {
		free -= sizes[i]
	}
	for free > 0.01 {
		growing := 0
		for i := range n {
			if track(tracks, i).kind == trackFit && sizes[i] < limits[i] {
				growing++
			}
		}
		if growing == 0 {
			break
		}
		share := free / float32(growing)
		for i := range n {
			if track(tracks, i).kind == trackFit && sizes[i] < limits[i] {
				d := min(share, limits[i]-sizes[i])
				sizes[i] += d
				free -= d
			}
		}
	}
	free = max(free, 0)
	if frSum > 0 {
		unit := free / max(frSum, 1)
		for i := range n {
			if t := track(tracks, i); t.kind == trackFr {
				sizes[i] = unit * t.v
			}
		}
		return
	}
	if stretch && free > 0 {
		fits := 0
		for i := range n {
			if track(tracks, i).kind == trackFit {
				fits++
			}
		}
		for i := range n {
			if fits > 0 && track(tracks, i).kind == trackFit {
				sizes[i] += free / float32(fits)
			}
		}
	}
}

// gridColumns sizes the columns of a grid placed in items, with room avail
// in a content box cw wide (both inf when unknown), in the scratch.
func gridColumns(e *node, s *gridScratch, items []gridItem, nc int, avail, cw float32, minContent bool) []float32 {
	first, cfirst := len(s.f), len(s.cs)
	for range 2 * nc {
		s.f = append(s.f, 0)
	}
	for range items {
		s.cs = append(s.cs, contrib{})
	}
	sizes, limits := s.f[first:first+nc], s.f[first+nc:first+2*nc]
	cs := s.cs[cfirst:]
	for i := range items {
		it := &items[i]
		c := it.e
		cs[i] = contrib{start: it.col, span: it.colSpan}
		if v, ok := c.width.resolve(base(cw)); ok {
			w := c.clampW(v, cw) + c.marginX()
			cs[i].min, cs[i].max = w, w
		} else {
			cs[i].min = intrinsic(c, false) + c.marginX()
			cs[i].max = intrinsic(c, true) + c.marginX()
		}
	}
	sizeTracks(e.cols, nc, avail, e.gapX, cs, minContent, e.justify == alignAuto, sizes, limits)
	s.cs = s.cs[:cfirst]
	return sizes
}

// gridIntrinsic returns the max-content or min-content width of a grid's
// content box.
func gridIntrinsic(e *node, maxContent bool) float32 {
	s := &e.c.rt.grid
	firstItem, firstF := len(s.items), len(s.f)
	defer func() { s.items, s.f = s.items[:firstItem], s.f[:firstF] }()
	items, nc, _ := gridPlace(e, s)
	sizes := gridColumns(e, s, items, nc, inf, inf, !maxContent)
	w := e.gapX * float32(max(nc-1, 0))
	for _, v := range sizes {
		w += v
	}
	return w
}

// gridLayout lays out the in-flow children of a grid in a content box
// cw×ch (inf when unknown) and returns the size its tracks take. With
// commit it gives the children their boxes (relative to e) and lays them
// out in turn.
func gridLayout(e *node, cw, ch float32, commit bool) (usedW, usedH float32) {
	s := &e.c.rt.grid
	firstItem, firstF, firstC := len(s.items), len(s.f), len(s.cs)
	defer func() { s.items, s.f, s.cs = s.items[:firstItem], s.f[:firstF], s.cs[:firstC] }()
	items, nc, nr := gridPlace(e, s)
	cols := gridColumns(e, s, items, nc, cw, cw, false)

	// The rows, from the heights of the items at the widths their columns
	// give them.
	at := len(s.f)
	for range 2*nr + 2*len(items) {
		s.f = append(s.f, 0)
	}
	for range items {
		s.cs = append(s.cs, contrib{})
	}
	rows, limits := s.f[at:at+nr], s.f[at+nr:at+2*nr]
	widths, offs := s.f[at+2*nr:at+2*nr+len(items)], s.f[at+2*nr+len(items):at+2*nr+2*len(items)]
	cs := s.cs[firstC:]
	align := e.align
	if align == alignAuto {
		align = Stretch
	}
	// Right to left, the columns go from the right, and items start at
	// the right of their cells.
	rtl := e.rtl()
	ms, me := 3, 1
	if rtl {
		ms, me = 1, 3
	}
	for i := range items {
		it := &items[i]
		c := it.e
		area := e.gapX * float32(it.colSpan-1)
		for k := it.col; k < it.col+it.colSpan; k++ {
			area += cols[k]
		}
		w, off := gridPlaceIn(c, area, c.width, c.marginX(), ms, me, gridJustify(e, c), cw, func(avail float32) float32 { return fitWidth(c, avail, cw) }, c.clampW)
		widths[i], offs[i] = w, off
		h := heightAt(c, w, ch) + c.marginY()
		cs[i] = contrib{start: it.row, span: it.rowSpan, min: h, max: h}
	}
	sizeTracks(e.rows, nr, ch, e.gapY, cs, false, e.alignContent == alignAuto, rows, limits)

	usedW = e.gapX * float32(max(nc-1, 0))
	for _, v := range cols {
		usedW += v
	}
	usedH = e.gapY * float32(max(nr-1, 0))
	for _, v := range rows {
		usedH += v
	}
	if !commit {
		return usedW, usedH
	}

	// Place the tracks as Justify and AlignContent say, and the items in
	// their cells.
	x0, betweenX := float32(0), float32(0)
	if finite(cw) && e.justify != alignAuto {
		x0, betweenX = justifyOffsets(e.justify, max(cw-usedW, 0), nc)
	}
	y0, betweenY := float32(0), float32(0)
	if finite(ch) && e.alignContent != alignAuto {
		y0, betweenY = justifyOffsets(e.alignContent, max(ch-usedH, 0), nr)
	}
	// Where each track starts.
	at = len(s.f)
	for range nc + nr {
		s.f = append(s.f, 0)
	}
	colAt, rowAt := s.f[at:at+nc], s.f[at+nc:at+nc+nr]
	for k, p := 0, x0; k < nc; k++ {
		colAt[k], p = p, p+cols[k]+e.gapX+betweenX
	}
	for k, p := 0, y0; k < nr; k++ {
		rowAt[k], p = p, p+rows[k]+e.gapY+betweenY
	}
	for i := range items {
		it := &items[i]
		c := it.e
		x := colAt[it.col] + offs[i]
		ah := (e.gapY + betweenY) * float32(it.rowSpan-1)
		for k := it.row; k < it.row+it.rowSpan; k++ {
			ah += rows[k]
		}
		w := widths[i]
		h, offY := gridPlaceIn(c, ah, c.height, c.marginY(), 0, 2, selfAlign(c, align), ch, func(float32) float32 {
			if c.aspect > 0 {
				return c.clampH(w/c.aspect, ch)
			}
			return heightAt(c, w, ch)
		}, c.clampH)
		y := rowAt[it.row] + offY
		if rtl {
			// x is from the right of the content, which is as wide as the
			// columns where its width is not known.
			right := cw
			if !finite(right) {
				right = usedW
			}
			x = right - x - w
		}
		c.x, c.y = e.contentX()+x, e.contentY()+y
		layoutBox(c, w, h)
		dx, dy := c.relative(cw, ch)
		c.x += dx
		c.y += dy
	}
	return usedW, usedH
}

// gridJustify returns how an item goes in its cell horizontally.
func gridJustify(e, c *node) Align {
	if c.justifySelf != alignAuto {
		return c.justifySelf
	}
	if e.justifyItems != alignAuto {
		return e.justifyItems
	}
	return Stretch
}

// gridPlaceIn sizes an item along one axis of its grid area, area DIPs
// long, and returns its size and where it starts in the area, margin
// included: its own size, or the area less its margins when it
// stretches, placed as a says or as its automatic margins (indexes ms and
// me of its margins) take the room left.
func gridPlaceIn(c *node, area float32, size length, margins float32, ms, me int, a Align, cb float32, natural func(avail float32) float32, clamp func(v, cb float32) float32) (float32, float32) {
	autoS, autoE := isAuto(c.margin[ms]), isAuto(c.margin[me])
	var v float32
	if l, ok := size.resolve(base(cb)); ok {
		v = clamp(l, cb)
	} else if a == Stretch && !autoS && !autoE && c.kind != kindIcon && (c.aspect == 0 || ms%2 == 1) {
		v = clamp(area-margins, cb)
	} else {
		v = natural(area - margins)
	}
	free := area - v - margins
	off := c.m(ms)
	switch {
	case autoS && autoE:
		off += max(free, 0) / 2
	case autoS:
		off += max(free, 0)
	case autoE:
	case a == Center:
		off += free / 2
	case a == End:
		off += free
	}
	return v, off
}
