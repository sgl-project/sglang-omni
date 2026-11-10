package editor

import (
	"math"
	"slices"
)

// diffPrepared is what the blocks and hidden lines were made of.
type diffPrepared struct {
	version uint64
	expandV int
	layout  DiffLayout
	context int
	anns    uint64 // the annotations' sides, lines and heights, hashed
	wrapsA  uint64
	wrapsB  uint64
	lh      float32
	wrap    bool
	expAll  bool
	inline  DiffInline
	nOld    int
	nNew    int
}

// diffGap is a run of unchanged lines between hunks, from the old line
// a and the new line b, n lines; hidden says which of them collapse.
type diffGap struct {
	a, b, n      int
	hideA, hideB int // where the hidden lines start in the old and new text
	hidden       int
	before       bool // before the first hunk
	after        bool // after the last
}

const diffSeparatorHeight = 32

// prepare makes the hidden lines and the blocks of each side, again only
// when what they come of changed.
func (d *DiffState) prepare() {
	d.model.update()
	unified := d.Layout == DiffUnified
	edA, edB := &d.eds[0], &d.eds[1]
	viewA := &edA.view
	if unified {
		viewA = &d.oldView
	}
	viewB := &edB.view
	key := diffPrepared{version: d.model.version, expandV: d.expandV, layout: d.Layout, context: d.context(), anns: annotationsKey(d.Annotations),
		wrapsA: viewA.heights.wraps.version, wrapsB: viewB.heights.wraps.version, lh: viewB.lineHeight, wrap: d.Wrap,
		expAll: d.ExpandUnchanged, inline: d.Inline, nOld: viewA.text.LineCount(), nNew: viewB.text.LineCount()}
	if d.prepared == key {
		return
	}
	// Measure the wraps of the lines of the hunks, which spacers align.
	if d.Wrap {
		for _, h := range d.model.hunks {
			for i := h.A0; i < h.A1; i++ {
				viewA.wrapLineNow(i)
			}
			for i := h.B0; i < h.B1; i++ {
				viewB.wrapLineNow(i)
			}
		}
		key.wrapsA, key.wrapsB = viewA.heights.wraps.version, viewB.heights.wraps.version
	}
	d.prepared = key
	gaps := d.gaps(viewA.text.LineCount(), viewB.text.LineCount())
	d.gapList = gaps
	lh := viewB.lineHeight
	var hideA, hideB []lineSpan
	var blkA, blkB []block
	// Blocks below a line that go after its annotations, as the separator
	// of the lines after the last change.
	var lateA, lateB []block
	for gi, g := range gaps {
		if g.hidden <= 0 {
			continue
		}
		hideA = append(hideA, lineSpan{g.hideA, g.hideA + g.hidden - 1})
		hideB = append(hideB, lineSpan{g.hideB, g.hideB + g.hidden - 1})
		sep := func(n, from, total int) block {
			b := block{kind: blockSeparator, height: diffSeparatorHeight, from: from, to: from + g.hidden, hunk: gi, id: g.a}
			if from+g.hidden < total {
				b.line = from + g.hidden
			} else {
				b.line, b.below = max(0, from-1), true
			}
			return b
		}
		for _, side := range []struct {
			blks, late *[]block
			from, n    int
		}{{&blkA, &lateA, g.hideA, viewA.text.LineCount()}, {&blkB, &lateB, g.hideB, viewB.text.LineCount()}} {
			b := sep(gi, side.from, side.n)
			if b.below {
				*side.late = append(*side.late, b)
			} else {
				*side.blks = append(*side.blks, b)
			}
		}
	}
	// A patch's hunks under their headers: above their first line, or,
	// for a hunk with no lines left on a side, below its last line, after
	// what the hunk before puts there (late).
	if pt := d.partial; pt != nil {
		header := func(blks, late *[]block, start, n, k int, h string) {
			b := block{kind: blockHeader, height: diffSeparatorHeight, line: start, hunk: k, id: h}
			if start >= n {
				b.line, b.below = max(0, n-1), true
				*late = append(*late, b)
				return
			}
			*blks = append(*blks, b)
		}
		for k, h := range pt.headers {
			header(&blkA, &lateA, pt.oldStarts[k], viewA.text.LineCount(), k, h)
			header(&blkB, &lateB, pt.newStarts[k], viewB.text.LineCount(), k, h)
		}
	}
	anns := [2]map[int]float32{{}, {}}
	for _, a := range d.Annotations {
		anns[a.Side][a.Line] += a.Height
	}
	rowsOf := func(v *view, line int) float32 { return float32(v.heights.wraps.rows(line)) * lh }
	if unified {
		for hi, h := range d.model.hunks {
			if h.A1 > h.A0 {
				var height float32
				for i := h.A0; i < h.A1; i++ {
					height += rowsOf(viewA, i) + anns[DiffOld][i]
				}
				b := block{kind: blockDeleted, height: height, from: h.A0, to: h.A1, hunk: hi}
				if h.B0 < viewB.text.LineCount() {
					b.line = h.B0
				} else {
					b.line, b.below = viewB.text.LineCount()-1, true
				}
				blkB = append(blkB, b)
			}
		}
		// The old text's annotations on unchanged lines go under the same
		// lines of the new text, as only those show.
		for line, height := range anns[DiffOld] {
			if nl, ok := d.model.newLineOf(line); ok {
				anns[DiffNew][nl] += height
			}
		}
		for line, height := range anns[DiffNew] {
			blkB = append(blkB, block{kind: blockAnnotation, line: line, below: true, height: height, id: line})
		}
		d.setSide(1, hideB, sticky(append(blkB, lateB...)))
		return
	}
	// Side by side: walk the rows of each hunk, the shorter side's lines
	// pushed down by the offset pairing them best, each row as high as
	// the higher side's line and annotations there, fillers where a side
	// has no line.
	type side struct {
		v    *view
		blks *[]block
		ann  map[int]float32
		n    int
	}
	sides := [2]side{{viewA, &blkA, anns[DiffOld], viewA.text.LineCount()}, {viewB, &blkB, anns[DiffNew], viewB.text.LineCount()}}
	emitAnn := func(s side, line int, extra float32) {
		if extra > 0 && line >= 0 {
			*s.blks = append(*s.blks, block{kind: blockSpacer, line: line, below: true, height: extra})
		}
	}
	prevA, prevB := 0, 0
	unchanged := func(a0, b0, n int) {
		// Annotations on unchanged lines leave room on the other side.
		for i := 0; i < n; i++ {
			ha, hb := sides[0].ann[a0+i], sides[1].ann[b0+i]
			if ha == 0 && hb == 0 {
				continue
			}
			hmax := max(ha, hb)
			if ha > 0 {
				*sides[0].blks = append(*sides[0].blks, block{kind: blockAnnotation, line: a0 + i, below: true, height: ha, id: a0 + i})
			}
			if hb > 0 {
				*sides[1].blks = append(*sides[1].blks, block{kind: blockAnnotation, line: b0 + i, below: true, height: hb, id: b0 + i})
			}
			emitAnn(sides[0], a0+i, hmax-ha)
			emitAnn(sides[1], b0+i, hmax-hb)
		}
	}
	for hi, h := range d.model.hunks {
		unchanged(prevA, prevB, h.A0-prevA)
		prevA, prevB = h.A1, h.B1
		nA, nB := h.A1-h.A0, h.B1-h.B0
		off := 0
		if nA > 0 && nB > 0 && nA != nB {
			off = d.model.wordsOf(hi, d.Inline).offset
		}
		rows := max(nA, nB)
		lineOf := func(s int, r int) int {
			n0, n := h.A0, nA
			if s == 1 {
				n0, n = h.B0, nB
			}
			k := r
			if n < rows {
				k = r - off
			}
			if k < 0 || k >= n {
				return -1
			}
			return n0 + k
		}
		var filler [2]float32
		var lastLine [2]int
		lastLine[0], lastLine[1] = -1, -1
		for r := 0; r < rows; r++ {
			var hs [2]float32
			var ls [2]int
			for s := 0; s < 2; s++ {
				ls[s] = lineOf(s, r)
				if ls[s] >= 0 {
					hs[s] = rowsOf(sides[s].v, ls[s]) + sides[s].ann[ls[s]]
				}
			}
			hmax := max(hs[0], hs[1])
			for s := 0; s < 2; s++ {
				sd := sides[s]
				if ls[s] < 0 {
					filler[s] += hmax
					continue
				}
				if filler[s] > 0 {
					*sd.blks = append(*sd.blks, block{kind: blockSpacer, line: ls[s], height: filler[s], from: 1, hunk: hi})
					filler[s] = 0
				}
				if a := sd.ann[ls[s]]; a > 0 {
					*sd.blks = append(*sd.blks, block{kind: blockAnnotation, line: ls[s], below: true, height: a, id: ls[s]})
				}
				emitAnn(sd, ls[s], hmax-hs[s])
				lastLine[s] = ls[s]
			}
		}
		for s := 0; s < 2; s++ {
			if filler[s] <= 0 {
				continue
			}
			sd := sides[s]
			b := block{kind: blockSpacer, height: filler[s], from: 1, hunk: hi}
			pos := h.A0
			if s == 1 {
				pos = h.B0
			}
			switch {
			case lastLine[s] >= 0:
				b.line, b.below = lastLine[s], true
			case pos < sd.n && !(s == 0 && nA == 0 && pos == sd.n-1 && sides[0].v.text.Len() == 0):
				b.line = pos
			default:
				b.line, b.below = max(0, sd.n-1), true
			}
			*sd.blks = append(*sd.blks, b)
		}
	}
	unchanged(prevA, prevB, min(sides[0].n-prevA, sides[1].n-prevB))
	d.setSide(0, hideA, sticky(append(blkA, lateA...)))
	d.setSide(1, hideB, sticky(append(blkB, lateB...)))
}

// sticky keeps the blocks that stand for lines, as separators, deleted
// lines and fillers, where their line is hidden, as with no context
// lines; annotations go with the line they are under.
func sticky(blks []block) []block {
	for i := range blks {
		if blks[i].kind != blockAnnotation {
			blks[i].sticky = true
		}
	}
	return blks
}

// setSide gives side i its hidden lines and blocks, keeping those it has
// when they are the same, as typing within a line leaves them: the
// heights take them without work, and the view, whose separators and
// buttons stand on them, is not built again (blocksV).
func (d *DiffState) setSide(i int, hide []lineSpan, blks []block) {
	s := &d.sides[i]
	if slices.Equal(s.hide, hide) && slices.Equal(s.blks, blks) {
		return
	}
	s.hide, s.blks = hide, blks
	d.blocksV++
}

// wrapLineNow measures the wraps of a line, if not yet.
func (v *view) wrapLineNow(i int) {
	w := &v.heights.wraps
	if !w.on || i < 0 || i >= len(w.extra) || w.extra[i] != wrapUnknown {
		return
	}
	v.line(i)
}

// gaps returns the runs of unchanged lines and which of them collapse.
func (d *DiffState) gaps(nOld, nNew int) []diffGap {
	hunks := d.model.hunks
	if len(hunks) == 0 || d.partial != nil {
		return nil
	}
	ctx := d.context()
	var out []diffGap
	pa, pb := 0, 0
	for i := 0; i <= len(hunks); i++ {
		na, nb := nOld, nNew
		if i < len(hunks) {
			na, nb = hunks[i].A0, hunks[i].B0
		}
		n := nb - pb
		if na-pa != n {
			n = min(n, na-pa)
		}
		g := diffGap{a: pa, b: pb, n: n, before: i == 0, after: i == len(hunks)}
		if !d.ExpandUnchanged && n > 0 {
			top, bottom := ctx, ctx
			if g.before {
				top = 0
			}
			if g.after {
				bottom = 0
			}
			// And the lines expanded at either edge.
			if top < n {
				if end := d.revealedAt(pa + top); end >= 0 {
					top = min(n, end+1-pa)
				}
			}
			if bottom < n {
				if first := d.revealedBefore(pa + n - 1 - bottom); first >= 0 {
					bottom = min(n, pa+n-first)
				}
			}
			hidden := n - top - bottom
			// One line alone shows rather than collapsing.
			if hidden > 1 {
				g.hidden = hidden
				g.hideA, g.hideB = pa+top, pb+top
			}
		}
		out = append(out, g)
		if i < len(hunks) {
			pa, pb = hunks[i].A1, hunks[i].B1
		}
	}
	return out
}

func (d *DiffState) annotationHeight(side DiffSide, line int) float32 {
	var h float32
	for _, a := range d.Annotations {
		if a.Side == side && a.Line == line {
			h += a.Height
		}
	}
	return h
}

// annotationsKey hashes what of the annotations places the blocks: their
// sides, lines and heights (FNV-1a).
func annotationsKey(anns []DiffAnnotation) uint64 {
	h := uint64(14695981039346656037)
	mix := func(v uint64) {
		for range 8 {
			h ^= v & 0xff
			h *= 1099511628211
			v >>= 8
		}
	}
	mix(uint64(len(anns)))
	for _, a := range anns {
		mix(uint64(a.Side))
		mix(uint64(a.Line))
		mix(uint64(math.Float32bits(a.Height)))
	}
	return h
}
