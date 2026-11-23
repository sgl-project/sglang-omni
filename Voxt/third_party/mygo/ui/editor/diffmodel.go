package editor

import (
	"slices"
	"sort"
	"strings"

	"github.com/egoist/mygo/internal/diff"
	"github.com/egoist/mygo/internal/rope"
)

// diffModel keeps the hunks between the lines of two buffers, old and
// new, as either changes. An edit of the new text compares again only
// the lines around it, from the unchanged lines before the hunks it
// touches to those after, as each keystroke would otherwise compare the
// whole texts; a change of the old text compares them anew.
type diffModel struct {
	old, new         *Buffer
	oldText, newText rope.Text
	oldVer, newVer   uint64
	hunks            []diff.Hunk
	// version counts the changes of hunks, for the caches of what they
	// show.
	version uint64
	words   map[wordKey]*hunkWords
}

// diffPad is how many unchanged lines around an edit are compared
// again with it.
const diffPad = 3

// diffMaxRegion is how many lines an edit compares again at most
// before comparing the whole texts.
const diffMaxRegion = 20000

func (d *diffModel) set(old, new *Buffer) {
	if d.old != old || d.new != new {
		d.old, d.new = old, new
		d.oldVer, d.newVer = ^uint64(0), ^uint64(0)
	}
}

// update compares the texts again where they changed, and reports
// whether the hunks changed.
func (d *diffModel) update() bool {
	if d.old == nil || d.new == nil {
		changed := d.hunks != nil
		d.hunks = nil
		if changed {
			d.version++
			d.words = nil
		}
		return changed
	}
	oldText, oldVer := d.old.snapshot()
	newText, newVer := d.new.snapshot()
	if oldVer == d.oldVer && newVer == d.newVer && d.old != d.new {
		return false
	}
	if d.old == d.new {
		changed := d.hunks != nil
		d.hunks, d.oldVer, d.newVer = nil, oldVer, newVer
		d.oldText, d.newText = oldText, newText
		if changed {
			d.version++
			d.words = nil
		}
		return changed
	}
	if oldVer != d.oldVer || newText.Len() == 0 || d.newText.Len() == 0 {
		d.full(oldText, newText, oldVer, newVer)
		return true
	}
	changes, ver, ok := d.new.changesSince(d.newVer)
	if !ok || ver != newVer {
		d.full(oldText, newText, oldVer, newVer)
		return true
	}
	// The lines the edits touched, in the new text, and the hunks moved
	// with the lines after them. Hunks next to an edit are touched too, as
	// lines it inserts where lines were deleted may be those lines again;
	// they are compared again with the edit.
	dirtyFrom, dirtyTo := -1, -1
	grow := func(a, b int) {
		if dirtyFrom < 0 {
			dirtyFrom, dirtyTo = a, b
			return
		}
		dirtyFrom, dirtyTo = min(dirtyFrom, a), max(dirtyTo, b)
	}
	touched := make([]bool, len(d.hunks))
	for _, cs := range changes {
		for _, c := range cs {
			// The lines the change replaced, where the changes before it
			// in its version put them, as the hunks and dirty lines are.
			lo, hi := c.newLine, c.newLine+c.oldEndLine-c.oldLine
			delta := (c.newEndLine - c.newLine) - (hi - lo)
			// The dirty lines after the change move with it.
			if dirtyFrom >= 0 {
				if dirtyFrom > hi {
					dirtyFrom += delta
				}
				if dirtyTo > hi {
					dirtyTo += delta
				} else if dirtyTo >= lo {
					dirtyTo = max(dirtyTo+delta, c.newEndLine)
				}
			}
			grow(c.newLine, c.newEndLine)
			for i := range d.hunks {
				h := &d.hunks[i]
				switch {
				case h.B1 < lo && h.B0 < lo:
				case h.B0 > hi+1:
					h.B0 += delta
					h.B1 += delta
				default:
					touched[i] = true
					h.B0 = min(h.B0, c.newLine)
					h.B1 = max(h.B1+delta, h.B0)
				}
			}
		}
	}
	d.newText, d.newVer = newText, newVer
	if dirtyFrom < 0 {
		return false
	}
	d.redo(dirtyFrom, dirtyTo, touched)
	return true
}

// full compares the whole texts.
func (d *diffModel) full(oldText, newText rope.Text, oldVer, newVer uint64) {
	d.oldText, d.newText, d.oldVer, d.newVer = oldText, newText, oldVer, newVer
	d.hunks = diff.Lines(ropeLines(oldText, 0, oldText.LineCount()), ropeLines(newText, 0, newText.LineCount()))
	d.version++
	d.words = nil
}

// ropeLines returns lines from to to of r: none of an empty text, which
// a diff takes as no lines rather than one empty line, as git does.
func ropeLines(r rope.Text, from, to int) []string {
	if from >= to || r.Len() == 0 {
		return nil
	}
	s := r.Slice(r.LineStart(from), r.LineEnd(to-1))
	return strings.Split(s, "\n")
}

// redo compares again the new lines from to to, inclusive, with a few
// unchanged lines around them and the hunks they touch: the range grows
// until no hunk crosses its edges, so that the lines before and after it
// pair up in both texts.
func (d *diffModel) redo(from, to int, touched []bool) {
	nNew := d.newText.LineCount()
	nOld := d.oldText.LineCount()
	newFrom := max(0, from-diffPad)
	newTo := min(nNew, to+1+diffPad)
	for i, h := range d.hunks {
		if touched[i] {
			newFrom = max(0, min(newFrom, h.B0-1))
			newTo = min(nNew, max(newTo, h.B1+1))
		}
	}
	for changed := true; changed; {
		changed = false
		for i, h := range d.hunks {
			if touched[i] {
				continue
			}
			if h.B0 < newFrom && h.B1 > newFrom {
				newFrom, changed = h.B0, true
			}
			if h.B0 < newTo && h.B1 > newTo {
				newTo, changed = h.B1, true
			}
		}
	}
	if newTo-newFrom > diffMaxRegion {
		d.full(d.oldText, d.newText, d.oldVer, d.newVer)
		return
	}
	// The hunks before and after the range stay; those in it go.
	var before, after []diff.Hunk
	beforeDelta, afterDelta := 0, 0
	for i, h := range d.hunks {
		if touched[i] {
			continue
		}
		switch {
		case h.B1 <= newFrom && (h.B0 < newFrom || h.B0 == h.B1):
			before = append(before, h)
			beforeDelta += (h.A1 - h.A0) - (h.B1 - h.B0)
		case h.B0 >= newTo && (h.B1 > newTo || h.B0 == h.B1):
			after = append(after, h)
			afterDelta += (h.A1 - h.A0) - (h.B1 - h.B0)
		}
	}
	oldFrom := newFrom + beforeDelta
	oldTo := nOld - (nNew - newTo) - afterDelta
	if oldFrom < 0 || oldTo < oldFrom || oldTo > nOld {
		d.full(d.oldText, d.newText, d.oldVer, d.newVer)
		return
	}
	a := ropeLines(d.oldText, oldFrom, oldTo)
	b := ropeLines(d.newText, newFrom, newTo)
	mid := diff.Lines(a, b)
	for i := range mid {
		h := &mid[i]
		h.A0, h.A1, h.B0, h.B1 = h.A0+oldFrom, h.A1+oldFrom, h.B0+newFrom, h.B1+newFrom
	}
	// Lines inserted that could show in more than one place, as a blank
	// line among others, show where the edit was: the line Enter opened
	// is the one added.
	for i := range mid {
		h := &mid[i]
		if h.A0 != h.A1 {
			continue
		}
		lo, hi := h.B0, h.B0
		limLo, limHi := newFrom, newTo-(h.B1-h.B0)
		if i > 0 {
			limLo = max(limLo, mid[i-1].B1+1)
		}
		if i+1 < len(mid) {
			limHi = min(limHi, mid[i+1].B0-1-(h.B1-h.B0))
		}
		for lo > limLo && d.newText.Line(lo-1) == d.newText.Line(lo-1+h.B1-h.B0) {
			lo--
		}
		for hi < limHi && d.newText.Line(hi) == d.newText.Line(hi+h.B1-h.B0) {
			hi++
		}
		best, bestOver := h.B0, overlap(h.B0, h.B1, from, to+1)
		for p := lo; p <= hi; p++ {
			if o := overlap(p, p+h.B1-h.B0, from, to+1); o > bestOver {
				best, bestOver = p, o
			}
		}
		shift := best - h.B0
		h.A0, h.A1, h.B0, h.B1 = h.A0+shift, h.A1+shift, h.B0+shift, h.B1+shift
	}
	hunks := append(append(append([]diff.Hunk{}, before...), mid...), after...)
	d.hunks = mergeHunks(hunks)
	d.version++
	d.words = nil
}

func overlap(a0, a1, b0, b1 int) int { return max(0, min(a1, b1)-max(a0, b0)) }

// mergeHunks joins hunks that touch.
func mergeHunks(hs []diff.Hunk) []diff.Hunk {
	sort.SliceStable(hs, func(i, j int) bool { return hs[i].B0 < hs[j].B0 || hs[i].B0 == hs[j].B0 && hs[i].A0 < hs[j].A0 })
	out := hs[:0]
	for _, h := range hs {
		if h.A0 == h.A1 && h.B0 == h.B1 {
			continue
		}
		if n := len(out); n > 0 && out[n-1].A1 == h.A0 && out[n-1].B1 == h.B0 {
			out[n-1].A1, out[n-1].B1 = h.A1, h.B1
			continue
		}
		out = append(out, h)
	}
	return out
}

// hunkAtNew returns the index of the hunk holding or ending at new line n,
// or -1.
func (d *diffModel) hunkAtNew(n int) int {
	i := sort.Search(len(d.hunks), func(i int) bool { return d.hunks[i].B1 > n || d.hunks[i].B0 >= n })
	if i < len(d.hunks) {
		h := d.hunks[i]
		if h.B0 <= n && n < h.B1 || h.B0 == h.B1 && h.B0 == n {
			return i
		}
	}
	if k := d.pastEnd(); k >= 0 && n == d.newText.LineCount()-1 {
		return k
	}
	return -1
}

// hunkAt returns the index of the hunk holding or ending at line n of
// the old text, or of the new, or -1: hunkAtNew for the new text.
func (d *diffModel) hunkAt(n int, old bool) int {
	if !old {
		return d.hunkAtNew(n)
	}
	i := sort.Search(len(d.hunks), func(i int) bool { return d.hunks[i].A1 > n || d.hunks[i].A0 >= n })
	if i < len(d.hunks) {
		h := d.hunks[i]
		if h.A0 <= n && n < h.A1 || h.A0 == h.A1 && h.A0 == n {
			return i
		}
	}
	// Lines added after the old text's last line go with it.
	if k := len(d.hunks) - 1; k >= 0 && n == d.oldText.LineCount()-1 {
		if h := d.hunks[k]; h.A0 == h.A1 && h.A0 >= d.oldText.LineCount() {
			return k
		}
	}
	return -1
}

// pastEnd returns the hunk deleting lines after the new text's last line,
// as when a text without a final newline loses its last lines, which goes
// with the last line, as no caret nor gutter row is past it; or -1.
func (d *diffModel) pastEnd() int {
	k := len(d.hunks) - 1
	if k < 0 {
		return -1
	}
	if h := d.hunks[k]; h.B0 == h.B1 && h.B0 >= d.newText.LineCount() {
		return k
	}
	return -1
}

// stats returns how many lines the hunks add and delete.
func (d *diffModel) stats() (added, deleted int) {
	for _, h := range d.hunks {
		added += h.B1 - h.B0
		deleted += h.A1 - h.A0
	}
	return added, deleted
}

// wordKey identifies a hunk's word diff by what it compares.
type wordKey struct {
	a0, a1, b0, b1 int
	mode           DiffInline
}

// hunkWords is the inline diff of a hunk: which line of each side pairs
// with which, and the ranges changed in each line.
type hunkWords struct {
	// pairOld[i] is the new line old line A0+i pairs with, or -1;
	// offset is how many rows the shorter side is pushed down to pair
	// its lines with the most similar ones.
	offset   int
	oldWords map[int][]diffRange // by old line
	newWords map[int][]diffRange // by new line
}

// wordsOf returns the inline diff of hunk i.
func (d *diffModel) wordsOf(i int, mode DiffInline) *hunkWords {
	h := d.hunks[i]
	key := wordKey{h.A0, h.A1, h.B0, h.B1, mode}
	if w, ok := d.words[key]; ok {
		return w
	}
	if d.words == nil {
		d.words = map[wordKey]*hunkWords{}
	}
	w := &hunkWords{oldWords: map[int][]diffRange{}, newWords: map[int][]diffRange{}}
	d.words[key] = w
	if h.A0 == h.A1 || h.B0 == h.B1 || mode == DiffInlineNone {
		return w
	}
	a := ropeLines(d.oldText, h.A0, h.A1)
	b := ropeLines(d.newText, h.B0, h.B1)
	w.offset = alignBySimilarity(a, b)
	for i := range a {
		j := i - w.offset
		if len(a) < len(b) {
			j = i + w.offset
		}
		if j < 0 || j >= len(b) {
			continue
		}
		if len(a[i]) > 1000 || len(b[j]) > 1000 {
			continue
		}
		ra, rb := inlineDiff(a[i], b[j], mode)
		if len(ra) > 0 {
			w.oldWords[h.A0+i] = ra
		}
		if len(rb) > 0 {
			w.newWords[h.B0+j] = rb
		}
	}
	return w
}

// alignBySimilarity returns how far to push the lines of the shorter side
// of a hunk down so that they pair with the most similar lines of the
// longer, as Pierre's diffs realign change blocks: by the common prefix
// and suffix of the lines without their whitespace, over their longer
// length, taking an offset only when it pairs better by half a line.
func alignBySimilarity(a, b []string) int {
	short, long := a, b
	if len(a) > len(b) {
		short, long = b, a
	}
	if len(short) == len(long) || len(short) == 0 || len(short)*(len(long)-len(short)+1) > 4096 {
		return 0
	}
	score := func(off int) float64 {
		s := 0.0
		for i, x := range short {
			s += similarity(x, long[i+off])
		}
		return s
	}
	best, bestScore := 0, score(0)
	for off := 1; off <= len(long)-len(short); off++ {
		if s := score(off); s > bestScore+0.5*float64(len(short)) {
			best, bestScore = off, s
		}
	}
	return best
}

func similarity(a, b string) float64 {
	a, b = strings.Join(strings.Fields(a), ""), strings.Join(strings.Fields(b), "")
	if a == b {
		return 1
	}
	n := max(len(a), len(b))
	if n == 0 {
		return 1
	}
	p := 0
	for p < len(a) && p < len(b) && a[p] == b[p] {
		p++
	}
	s := 0
	for s < len(a)-p && s < len(b)-p && a[len(a)-1-s] == b[len(b)-1-s] {
		s++
	}
	return float64(p+s) / float64(n)
}

// inlineDiff returns the ranges of a and b that changed, as mode says.
func inlineDiff(a, b string, mode DiffInline) (ra, rb []diffRange) {
	var xa, xb []diff.Range
	switch mode {
	case DiffInlineChar:
		xa, xb, _ = diff.Chars(a, b)
	default:
		xa, xb, _ = diff.Words(a, b)
	}
	conv := func(rs []diff.Range, s string) []diffRange {
		var out []diffRange
		for _, r := range rs {
			out = append(out, diffRange{r.Start, r.End})
		}
		if mode == DiffInlineWordAlt {
			out = joinWordAlt(out, s)
		}
		if mode == DiffInlineWordLine && len(out) > 0 {
			out = []diffRange{{out[0].start, out[len(out)-1].end}}
		}
		return out
	}
	return conv(xa, a), conv(xb, b)
}

// joinWordAlt joins ranges one character apart, as Pierre's word-alt
// does: "a b c" changed reads as one highlight rather than three.
func joinWordAlt(rs []diffRange, s string) []diffRange {
	if len(rs) < 2 {
		return rs
	}
	out := rs[:1]
	for _, r := range rs[1:] {
		last := &out[len(out)-1]
		gap := s[last.end:r.start]
		if len([]rune(gap)) <= 1 {
			last.end = r.end
			continue
		}
		out = append(out, r)
	}
	return out
}

// revert puts the old lines of hunk i back in the new buffer.
func (d *diffModel) revert(i int) {
	if i >= 0 && i < len(d.hunks) {
		d.new.Edit(hunkEdit(d.hunks[i], d.oldText, d.newText))
	}
}

// hunkEdit returns the edit of the new text that puts the old lines of
// h back.
func hunkEdit(h diff.Hunk, old, nw rope.Text) Edit {
	text := ""
	if h.A1 > h.A0 {
		text = old.Slice(old.LineStart(h.A0), old.LineEnd(h.A1-1))
	}
	n := nw.LineCount()
	switch {
	case h.B1 > h.B0 && h.A1 > h.A0:
		return Edit{Start: nw.LineStart(h.B0), End: nw.LineEnd(h.B1 - 1), Text: text}
	case h.B1 > h.B0:
		// Lines added go with a newline: theirs, or the one before them
		// at the end.
		switch {
		case h.B1 < n:
			return Edit{Start: nw.LineStart(h.B0), End: nw.LineStart(h.B1)}
		case h.B0 > 0:
			return Edit{Start: nw.LineEnd(h.B0 - 1), End: nw.Len()}
		}
		return Edit{Start: 0, End: nw.Len()}
	case h.B0 < n:
		at := nw.LineStart(h.B0)
		return Edit{Start: at, End: at, Text: text + "\n"}
	}
	return Edit{Start: nw.Len(), End: nw.Len(), Text: "\n" + text}
}

// accept makes the new lines of hunk i the old ones, so the hunk goes.
func (d *diffModel) accept(i int) {
	if i < 0 || i >= len(d.hunks) {
		return
	}
	// The same edit, the other way: the old buffer takes the new lines.
	h := d.hunks[i]
	d.old.Edit(hunkEdit(diff.Hunk{A0: h.B0, A1: h.B1, B0: h.A0, B1: h.A1}, d.newText, d.oldText))
}

// hunksTouching returns the hunks whose new lines meet from to to.
func (d *diffModel) hunksTouching(from, to int) []int {
	var out []int
	for i, h := range d.hunks {
		if h.B0 <= to && (h.B1 > from || h.B0 == h.B1 && h.B0 >= from) {
			out = append(out, i)
		}
	}
	if k := d.pastEnd(); k >= 0 && to >= d.newText.LineCount()-1 && !slices.Contains(out, k) {
		out = append(out, k)
	}
	return out
}

// newLineOf returns the new line an unchanged old line is, and false for
// a line a hunk changes.
func (d *diffModel) newLineOf(old int) (int, bool) {
	delta := 0
	for _, h := range d.hunks {
		if old < h.A0 {
			break
		}
		if old < h.A1 {
			return 0, false
		}
		delta = h.B1 - h.A1
	}
	return old + delta, true
}
