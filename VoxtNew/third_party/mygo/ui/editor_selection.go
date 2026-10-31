package ui

import (
	"slices"
	"strings"

	"github.com/egoist/mygo/internal/text"
)

// editorSelection stores only offsets and visual edges, independent of the
// document buffer, layout cache and widget painting. A visual gesture may
// select several logical ranges in mixed-direction text.
type editorSelection struct {
	caret, anchor                 int
	caretAffinity, anchorAffinity text.Affinity
	visualSelection               bool
	selected                      []text.Range
}

func (s editorSelection) clone() editorSelection { s.selected = slices.Clone(s.selected); return s }

func (ed *editor) selectedRanges() []text.Range {
	if ed.visualSelection {
		return ed.selected
	}
	a, z := ed.selection()
	if a == z {
		return nil
	}
	return []text.Range{{Start: a, End: z}}
}

func (ed *editor) selectedText() string {
	var out strings.Builder
	for _, r := range ed.selectedRanges() {
		out.WriteString(ed.buf.slice(r.Start, r.End))
	}
	return out.String()
}

func (ed *editor) movePosition(p text.CaretPosition, extend bool) {
	if !extend {
		ed.move(p.Index, false)
		ed.caretAffinity, ed.anchorAffinity = p.Affinity, p.Affinity
		return
	}
	ed.caret, ed.caretAffinity = p.Index, p.Affinity
	ed.coalesce = false
	ed.visualSelection = true
	ed.selected = ed.visualRanges()
	if ed.area != nil {
		ed.area.reveal = true
	}
}

func (ed *editor) visualRanges() []text.Range {
	a := text.CaretPosition{Index: ed.anchor, Affinity: ed.anchorAffinity}
	z := text.CaretPosition{Index: ed.caret, Affinity: ed.caretAffinity}
	if ed.area == nil {
		if ed.layout == nil {
			return nil
		}
		return ed.layout.SelectionRanges(a, z)
	}
	pa, pz := ed.buf.para(a.Index), ed.buf.para(z.Index)
	if pa > pz {
		pa, pz, a, z = pz, pa, z, a
	}
	var out []text.Range
	for p := pa; p <= pz; p++ {
		l := ed.area.paraLayout(ed, p)
		start := ed.buf.start(p)
		from, to := text.CaretPosition{}, text.CaretPosition{Index: len(l.Runes), Affinity: text.Upstream}
		if p == pa {
			from = text.CaretPosition{Index: a.Index - start, Affinity: a.Affinity}
		}
		if p == pz {
			to = text.CaretPosition{Index: z.Index - start, Affinity: z.Affinity}
		}
		for _, r := range l.SelectionRanges(from, to) {
			out = append(out, text.Range{Start: start + r.Start, End: start + r.End})
		}
		if p < pz {
			end := ed.buf.end(p)
			out = append(out, text.Range{Start: end, End: end + 1})
		}
	}
	slices.SortFunc(out, func(a, b text.Range) int { return a.Start - b.Start })
	merged := out[:0]
	for _, r := range out {
		if n := len(merged); n > 0 && merged[n-1].End >= r.Start {
			merged[n-1].End = max(merged[n-1].End, r.End)
		} else {
			merged = append(merged, r)
		}
	}
	return merged
}

func (ed *editor) visualNext(direction int) text.CaretPosition {
	p := text.CaretPosition{Index: ed.caret, Affinity: ed.caretAffinity}
	if a := ed.area; a != nil {
		para := ed.buf.para(ed.caret)
		l := a.paraLayout(ed, para)
		local := text.CaretPosition{Index: ed.caret - ed.buf.start(para), Affinity: p.Affinity}
		next := l.MoveCaret(local, direction)
		if next != local {
			next.Index += ed.buf.start(para)
			return next
		}
		step := direction
		if len(l.Lines) > 0 && l.Lines[0].RTL {
			step = -step
		}
		para += step
		if para < 0 || para >= len(ed.buf.paras) {
			return p
		}
		if step > 0 {
			return text.CaretPosition{Index: ed.buf.start(para)}
		}
		return text.CaretPosition{Index: ed.buf.end(para), Affinity: text.Upstream}
	}
	if ed.layout != nil {
		return ed.layout.MoveCaret(p, direction)
	}
	if direction < 0 {
		p.Index = ed.graphemes.prev(&ed.buf, p.Index)
	} else {
		p.Index = ed.graphemes.next(&ed.buf, p.Index)
	}
	return p
}

// caretParagraphRTL reports whether the paragraph of the caret reads
// right to left.
func (ed *editor) caretParagraphRTL() bool {
	l := ed.layout
	if a := ed.area; a != nil {
		l = a.paraLayout(ed, ed.buf.para(ed.caret))
	}
	return l != nil && len(l.Lines) > 0 && l.Lines[0].RTL
}

func (ed *editor) hitPosition(x, y float32) text.CaretPosition {
	if a := ed.area; a != nil {
		return a.positionAt(ed, x-ed.originX, float64(y-ed.originY)+a.scroll)
	}
	if ed.layout == nil {
		return text.CaretPosition{}
	}
	p := ed.layout.PositionAt(x-ed.originX+ed.scrollX, y-ed.originY)
	p.Index = ed.textIndex(p.Index)
	return p
}

func (ed *editor) collapseVisual(direction int) text.CaretPosition {
	a, z := text.CaretPosition{Index: ed.anchor, Affinity: ed.anchorAffinity}, text.CaretPosition{Index: ed.caret, Affinity: ed.caretAffinity}
	if ed.area != nil && ed.buf.para(a.Index) != ed.buf.para(z.Index) {
		if (a.Index < z.Index) == (direction < 0) {
			return a
		}
		return z
	}
	l := ed.layout
	if ed.area != nil {
		l = ed.area.paraLayout(ed, ed.buf.para(ed.caret))
	}
	if l == nil {
		return z
	}
	start := 0
	if ed.area != nil {
		start = ed.buf.start(ed.buf.para(ed.caret))
	}
	ax, _, _ := l.CaretAt(text.CaretPosition{Index: a.Index - start, Affinity: a.Affinity})
	zx, _, _ := l.CaretAt(text.CaretPosition{Index: z.Index - start, Affinity: z.Affinity})
	if (ax < zx) == (direction < 0) {
		return a
	}
	return z
}
