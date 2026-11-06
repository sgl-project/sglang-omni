package ui

import (
	"strings"
	"unicode/utf8"

	"github.com/egoist/mygo/internal/platform"
)

// Inline elements are the text elements built inside another text: they
// continue its paragraph, which lays out its own text and theirs as one,
// with the styles they set as spans over its own. They keep their state
// and their interaction (clicks, hovering, the focus, tooltips, context
// menus, assistive technology), with the boxes of their words, in the
// paragraph's layout, as their area.

// isInline reports whether the element is inside a text.
func (e *node) isInline() bool { return e.parent != nil && e.parent.kind == kindText }

// inlineText makes the text of a paragraph its own followed by that of the
// elements inside it, once they are built, and gives each the range of its
// text in it.
func (e *node) inlineText() {
	if !e.paragraph {
		e.paragraph, e.ownText = true, e.text
	}
	var b strings.Builder
	b.WriteString(e.ownText)
	n := utf8.RuneCountInString(e.ownText)
	for ch := e.first; ch != nil; ch = ch.next {
		m := utf8.RuneCountInString(ch.text)
		ch.runes = [2]int{n, n + m}
		n += m
		b.WriteString(ch.text)
	}
	e.text = b.String()
}

// inlineSpans returns the spans of a paragraph: its own, then those of the
// elements inside it, in the styles they set over it.
func (e *node) inlineSpans() []Span {
	// The spans go into a buffer the element's state keeps, as the frame
	// is done with them once it painted.
	sc := e.st.spanCache()
	out := append(sc.inline[:0], e.spans...)
	if len(e.spans) == 0 && e.ownText != "" {
		out = append(out, Span{Text: e.ownText})
	}
	for ch := e.first; ch != nil; ch = ch.next {
		spans := ch.spans
		switch {
		case ch.first != nil:
			spans = ch.inlineSpans()
		case spans == nil:
			spans = []Span{{Text: ch.text}}
		}
		base := ch.spanStyle()
		for _, s := range spans {
			out = append(out, over(s, base))
		}
	}
	sc.inline = out
	return out
}

// spanStyle returns the text style the element sets, as a span.
func (e *node) spanStyle() Span {
	t := &e.ts
	var s Span
	if t.set&setFamily != 0 {
		s.Font = t.family
	}
	if t.set&setSize != 0 {
		s.Size = t.size
	}
	if t.set&setWeight != 0 {
		s.Weight = t.weight
	}
	if t.set&setItalic != 0 {
		s.Italic = t.italic
	}
	if t.set&setColor != 0 {
		s.Color = t.color
	}
	if t.set&setUnderline != 0 {
		s.Underline, s.WavyUnderline = t.underline && !t.wavy, t.underline && t.wavy
	}
	if t.set&setStrike != 0 {
		s.Strikethrough = t.strike
	}
	if t.set&setSpacing != 0 {
		s.LetterSpacing = t.spacing
	}
	if t.set&setFeatures != 0 {
		s.Features = t.features
	}
	if t.set&setLang != 0 {
		s.Lang = t.lang
	}
	if t.set&setDecoColor != 0 {
		s.DecorationColor = t.decoColor
	}
	if t.set&setDecoThick != 0 {
		s.DecorationThickness = t.decoThick
	}
	switch {
	case t.set&setBackground != 0:
		s.Background = t.background
	case e.fill == fillColor && e.bg.A > 0:
		// The background of an inline element is that of its text.
		s.Background = e.bg
	}
	// What dims the element dims its text.
	alpha := float32(1)
	if e.flags&flagDisabled != 0 {
		alpha = 0.5
	}
	if e.opacitySet {
		alpha *= e.opacity
	}
	if alpha < 1 {
		s.Color = e.resolvedText().color.Alpha(alpha)
	}
	return s
}

// over returns span s over base: what s leaves zero takes base's style.
func over(s, base Span) Span {
	if s.Font == "" {
		s.Font = base.Font
	}
	if s.Size == 0 {
		s.Size = base.Size
	}
	if s.Weight == 0 {
		s.Weight = base.Weight
	}
	if s.Color.A == 0 {
		s.Color = base.Color
	}
	if s.DecorationColor.A == 0 {
		s.DecorationColor = base.DecorationColor
	}
	if s.DecorationThickness == 0 {
		s.DecorationThickness = base.DecorationThickness
	}
	if s.Background.A == 0 {
		s.Background = base.Background
	}
	if s.LetterSpacing == 0 {
		s.LetterSpacing = base.LetterSpacing
	}
	if s.Features == "" {
		s.Features = base.Features
	}
	if s.Lang == "" {
		s.Lang = base.Lang
	}
	s.Italic = s.Italic || base.Italic
	s.Underline = s.Underline || base.Underline
	s.WavyUnderline = s.WavyUnderline || base.WavyUnderline
	s.Strikethrough = s.Strikethrough || base.Strikethrough
	return s
}

// placeInline gives the elements inside e, which starts at rune offset of
// paragraph para's text, and those inside them, the boxes of their text in
// the paragraph's layout, and as their box the box around those.
func placeInline(para, e *node, offset int) {
	ox, oy := para.x+para.contentX(), para.y+para.contentY()
	for ch := e.first; ch != nil; ch = ch.next {
		a, b := offset+ch.runes[0], offset+ch.runes[1]
		ch.frags = ch.frags[:0]
		var box Rect
		if para.tl != nil {
			rt := para.c.rt
			rt.inlineRects = para.tl.AppendSelection(rt.inlineRects[:0], a, b, false)
			for i, r := range rt.inlineRects {
				f := Rect{ox + r.X, oy + r.Y, r.W, r.H}
				ch.frags = append(ch.frags, f)
				if i == 0 {
					box = f
				} else {
					box = union(box, f)
				}
			}
		}
		ch.x, ch.y, ch.w, ch.h = box.X, box.Y, box.W, box.H
		placeInline(para, ch, a)
	}
}

func union(a, b Rect) Rect {
	x0, y0 := min(a.X, b.X), min(a.Y, b.Y)
	x1, y1 := max(a.X+a.W, b.X+b.W), max(a.Y+a.H, b.Y+b.H)
	return Rect{x0, y0, x1 - x0, y1 - y0}
}

// paintInline draws the focus ring of an inline element around its words,
// and those of the elements inside it: the paragraph draws their text.
func (p *Painter) paintInline(e *node) {
	if e.ringShown() {
		for _, r := range e.frags {
			p.FocusRing(r, [4]float32{2, 2, 2, 2})
		}
	}
	for ch := e.first; ch != nil; ch = ch.next {
		if ch.flags&flagInvisible == 0 {
			p.element(ch)
		}
	}
}

// ringShown reports whether the element has the keyboard focus from the
// keyboard and MyGo rings it.
func (e *node) ringShown() bool {
	rt := e.c.rt
	return e.flags&(flagFocusable|flagOwnRing) == flagFocusable && rt.focused == e.id && rt.focusVisible && rt.windowFocus()
}

// accessInline describes the elements inside text e that are more than
// text to assistive technology, such as links, as nodes inside the node
// parent: the text itself already reads theirs.
func (rt *engine) accessInline(t *platform.AccessTree, e *node, parent int) {
	for ch := e.first; ch != nil; ch = ch.next {
		if ch.flags&flagInvisible != 0 || len(ch.frags) == 0 {
			continue
		}
		role, ok := ch.accessRole()
		if !ok || role == platform.RoleText && ch.flags&(flagClickable|flagFocusable) == 0 {
			rt.accessInline(t, ch, parent)
			continue
		}
		n := platform.AccessNode{
			ID: ch.id, Parent: parent, Role: role, Label: ch.label,
			Bounds: accessBounds(ch),
		}
		rt.accessDetails(ch, &n)
		t.Nodes = append(t.Nodes, n)
	}
}
