package ui

import (
	"slices"
	"strings"
	"sync"
	"unicode/utf16"

	"github.com/egoist/mygo/internal/text"
)

// TextLayout is retained shaped text for a custom text control. It owns font
// and geometry results, not editable document state. Cache layouts of the
// paragraphs or lines you display, and replace only those affected by an edit.
// Its geometry methods are safe from any goroutine. Drawing uses a Painter
// during the view's frame. Offsets are UTF-16, matching TextInputClient.
type TextLayout struct {
	mu         sync.Mutex
	l          *text.Layout
	units      []int // rune offset -> UTF-16 offset
	boundaries []int // grapheme boundaries in runes
	spans      []Span
}

// TextAffinity chooses a caret's preceding or following logical edge. Bidi
// boundaries and soft wraps can give the same index two visual positions.
type TextAffinity = text.Affinity

const (
	TextDownstream = text.Downstream
	TextUpstream   = text.Upstream
)

// TextCaretPosition retains a UTF-16 index and its visual affinity.
type TextCaretPosition struct {
	Index    int
	Affinity TextAffinity
}

func (t *TextLayout) position(p TextCaretPosition) text.CaretPosition {
	return text.CaretPosition{Index: t.runeAt(p.Index, false), Affinity: p.Affinity}
}

// CaretAt returns the rectangle of the requested visual edge.
func (t *TextLayout) CaretAt(p TextCaretPosition) Rect {
	t.mu.Lock()
	defer t.mu.Unlock()
	x, y, h := t.l.CaretAt(t.position(p))
	return Rect{x, y, 1, h}
}

// PositionAt retains affinity while hit-testing whole graphemes.
func (t *TextLayout) PositionAt(point Point) TextCaretPosition {
	t.mu.Lock()
	defer t.mu.Unlock()
	p := t.l.PositionAt(point.X, point.Y)
	return TextCaretPosition{Index: t.units[p.Index], Affinity: p.Affinity}
}

// MoveCaret moves one visual grapheme left (negative) or right (positive).
func (t *TextLayout) MoveCaret(p TextCaretPosition, direction int) TextCaretPosition {
	t.mu.Lock()
	defer t.mu.Unlock()
	next := t.l.MoveCaret(t.position(p), direction)
	return TextCaretPosition{Index: t.units[next.Index], Affinity: next.Affinity}
}

// SelectionRanges maps a visual gesture to logical UTF-16 ranges, preserving
// discontiguous selections through mixed-direction text. Supply these ranges
// to SelectionRects and to your buffer's copy/replacement operations.
func (t *TextLayout) SelectionRanges(anchor, caret TextCaretPosition) []TextInputRange {
	t.mu.Lock()
	defer t.mu.Unlock()
	var out []TextInputRange
	for _, r := range t.l.SelectionRanges(t.position(anchor), t.position(caret)) {
		out = append(out, TextInputRange{Start: t.units[r.Start], End: t.units[r.End]})
	}
	return out
}

// ShapeText shapes text with the system's fonts, fallback, bidi and line
// breaking, wrapping at width DIPs (zero only breaks at newlines). It does not
// retain a copy of an editor document or install keyboard/editing behavior.
func ShapeText(s string, font Font, width float32) *TextLayout { return shapeText(s, font, width, nil) }

// ShapeRichText shapes the existing display-only Span styles into retained
// geometry, so an application-owned rich editor can render its own runs.
func ShapeRichText(spans []Span, font Font, width float32) *TextLayout {
	var b strings.Builder
	for _, sp := range spans {
		b.WriteString(sp.Text)
	}
	return shapeText(b.String(), font, width, slices.Clone(spans))
}

// TextShape is how ShapeRichTextWith lays text out.
type TextShape struct {
	// Width wraps lines at it, in DIPs; zero only breaks at newlines.
	Width float32
	// Align aligns the lines within Width: Start (the default, the right
	// in right-to-left paragraphs, and of a right-to-left Direction),
	// Center or End.
	Align Align
	// Direction is that of the element the text shows in
	// (Context.Direction): right to left, lines start at the right, as
	// those of Text, and paragraphs without a strongly directional
	// character, as an empty one, read right to left. The others read in
	// the direction of their first strong character either way.
	Direction Direction
}

// ShapeRichTextWith shapes spans as ShapeRichText does, laid out as shape
// says: a rich text editor aligns its paragraphs.
func ShapeRichTextWith(spans []Span, font Font, shape TextShape) *TextLayout {
	var b strings.Builder
	for _, sp := range spans {
		b.WriteString(sp.Text)
	}
	return shapeTextAligned(b.String(), font, shape, slices.Clone(spans))
}

func shapeText(s string, font Font, width float32, spans []Span) *TextLayout {
	return shapeTextAligned(s, font, TextShape{Width: width}, spans)
}

func shapeTextAligned(s string, font Font, shape TextShape, spans []Span) *TextLayout {
	params := text.Params{Text: s, Style: font.style(), Width: shape.Width, KeepSpaces: true, Spans: encodeSpans(spans), RTL: shape.Direction == RightToLeft}
	switch {
	case shape.Align == Center:
		params.Align = text.Center
	case shape.Direction == RightToLeft && shape.Align == End:
		params.Align = text.Left
	case shape.Direction == RightToLeft:
		params.Align = text.Right
	case shape.Align == End:
		params.Align = text.End
	}
	l := text.Shared().Shape(params)
	t := &TextLayout{l: l, units: make([]int, len(l.Runes)+1), spans: spans}
	for i, r := range l.Runes {
		t.units[i+1] = t.units[i] + utf16.RuneLen(r)
	}
	var boundaries text.Boundaries
	boundaries.Reset(l.Runes)
	// Boundary iteration is per shaped paragraph, never per document edit.
	t.boundaries = boundaries.GraphemeOffsets()
	return t
}

// TextObject is an inline object of a TextLayout (Span.ObjectWidth): the
// UTF-16 offset of its object replacement character, and its box,
// standing on its line's baseline.
type TextObject struct {
	Index int
	Rect  Rect
}

// Objects returns the inline objects of the layout, in the order of the
// text, where a rich text editor draws its attachments.
func (t *TextLayout) Objects() []TextObject {
	t.mu.Lock()
	defer t.mu.Unlock()
	var out []TextObject
	var spans []text.Span
	for _, line := range t.l.Lines {
		for _, g := range line.Glyphs {
			if !g.Object {
				continue
			}
			if spans == nil {
				spans = text.DecodeSpans(t.l.Params.Spans)
			}
			h := float32(0)
			end := 0
			for _, sp := range spans {
				end = sp.End
				if g.Cluster < end {
					h = sp.ObjectHeight
					break
				}
			}
			out = append(out, TextObject{Index: t.units[g.Cluster], Rect: Rect{X: g.X, Y: line.Baseline - h, W: g.Advance, H: h}})
		}
	}
	return out
}

// Size is the layout's extent in DIPs.
func (t *TextLayout) Size() (float32, float32) {
	t.mu.Lock()
	defer t.mu.Unlock()
	return t.l.Width, t.l.Height
}

func (t *TextLayout) runeAt(index int, forward bool) int {
	i, found := slices.BinarySearch(t.units, max(0, min(index, t.units[len(t.units)-1])))
	if !found && !forward {
		i--
	}
	b, found := slices.BinarySearch(t.boundaries, i)
	if !found {
		if forward {
			i = t.boundaries[b]
		} else {
			i = t.boundaries[b-1]
		}
	}
	return i
}

// Caret gives the caret rectangle at index, snapped to a whole grapheme.
func (t *TextLayout) Caret(index int) Rect {
	t.mu.Lock()
	defer t.mu.Unlock()
	x, y, h := t.l.Caret(t.runeAt(index, false))
	return Rect{x, y, 1, h}
}

// PreviousBoundary and NextBoundary find neighboring whole-grapheme offsets
// for logical deletion and navigation. Visual selection policy remains with
// the application, which can keep multiple ranges and caret affinity.
func (t *TextLayout) PreviousBoundary(index int) int {
	t.mu.Lock()
	defer t.mu.Unlock()
	i := t.runeAt(index, false)
	b, _ := slices.BinarySearch(t.boundaries, i)
	return t.units[t.boundaries[max(0, b-1)]]
}
func (t *TextLayout) NextBoundary(index int) int {
	t.mu.Lock()
	defer t.mu.Unlock()
	i := t.runeAt(index, false)
	b, _ := slices.BinarySearch(t.boundaries, i)
	return t.units[t.boundaries[min(len(t.boundaries)-1, b+1)]]
}

// IndexAt finds the nearest whole-grapheme caret in visual layout order.
func (t *TextLayout) IndexAt(point Point) int { return t.PositionAt(point).Index }

// SelectionRects returns the visual rectangles of one or more logical
// ranges, preserving gaps in bidi selections. An editor supplies its own
// range set and selection policy; the layout does not reduce it to one range.
func (t *TextLayout) SelectionRects(ranges ...TextInputRange) []Rect {
	t.mu.Lock()
	defer t.mu.Unlock()
	var out []Rect
	for _, r := range ranges {
		r = platformTextRange(r)
		for _, b := range t.l.SelectionVisual(t.runeAt(r.Start, false), t.runeAt(r.End, true), false) {
			out = append(out, Rect{b.X, b.Y, b.W, b.H})
		}
	}
	return out
}

func platformTextRange(r TextInputRange) TextInputRange {
	if r.Start > r.End {
		r.Start, r.End = r.End, r.Start
	}
	return r
}

// TextLayout paints a retained layout at (x,y) in DIPs, with color as the
// default foreground. Span colors and decorations paint as RichText does.
func (p *Painter) TextLayout(layout *TextLayout, x, y float32, color Color) {
	if layout == nil {
		return
	}
	layout.mu.Lock()
	defer layout.mu.Unlock()
	p.textLayout(layout.l, x, y, color, textStyle{size: layout.l.Params.Style.FontSize()}, newSpanPaint(layout.spans))
}
