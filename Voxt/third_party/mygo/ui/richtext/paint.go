package richtext

import (
	"strconv"
	"time"

	"github.com/egoist/mygo/ui"
)

// paint lays out the paragraphs in view of the box r, and paints them:
// their decorations, the selection, their text, the markers of lists, the
// composition, the caret and the scroll bar.
func (v *view) paint(p *ui.Painter, r ui.Rect) {
	v.rect = r
	w, h := max(r.W-2*v.padX, 1), max(r.H-2*v.padY, 1)
	v.layout(w, h)
	ox, oy := r.X+v.padX, r.Y+v.padY
	look := &v.look
	if v.shown.Len() == 0 && v.compose == "" && v.opts.Placeholder != "" {
		pl := ui.ShapeRichTextWith([]ui.Span{{Text: v.opts.Placeholder}}, v.paraFont(v.para(0).style), ui.TextShape{Width: w})
		p.TextLayout(pl, ox, oy, look.Muted)
	}
	a, z := v.selection()
	focused := v.focused
	for i := v.first; i <= v.last && i < len(v.paras); i++ {
		pv := v.paraLayout(i)
		st := v.para(i).style
		top := oy + float32(v.hs.top(i)-v.scroll)
		px, py := v.origin(i)
		x, y := ox+px, oy+float32(py-v.scroll)
		v.paintBlock(p, i, st, ox, top, pv.h, w)
		ps := v.shown.paraStart(i)
		pe := ps + v.para(i).n
		if a != z && a <= pe && z >= ps && (focused || v.opts.ReadOnly) {
			from, to := v.local(i, max(a, ps)), v.local(i, min(z, pe))
			color := look.Selection
			for _, s := range pv.layout.SelectionRects(ui.TextInputRange{Start: pv.units[from], End: pv.units[to]}) {
				p.Fill(ui.Rect{X: x + s.X, Y: y + s.Y, W: s.W, H: s.H}, color, 0)
			}
			if z > pe && i < len(v.paras)-1 {
				// The newline selected, as a space after the line.
				c := pv.layout.CaretAt(ui.TextCaretPosition{Index: pv.units[len(pv.units)-1], Affinity: ui.TextUpstream})
				p.Fill(ui.Rect{X: x + c.X, Y: y + c.Y, W: look.size * 0.35, H: c.H}, color, 0)
			}
		}
		p.TextLayout(pv.layout, x, y, look.Text)
		v.paintAttachments(p, i, pv, x, y)
		v.paintMarker(p, i, st, pv, x, y)
		if v.compose != "" && v.shown.paraAt(v.caret) == i {
			c := v.local(i, v.caret)
			n := len([]rune(v.compose))
			for _, s := range pv.layout.SelectionRects(ui.TextInputRange{Start: pv.units[c], End: pv.units[min(c+n, len(pv.units)-1)]}) {
				p.Fill(ui.Rect{X: x + s.X, Y: y + s.Y + s.H - 2, W: s.W, H: 1}, look.Text, 0)
			}
		}
	}
	if focused && !v.opts.ReadOnly && a == z {
		const blink = 530 * time.Millisecond
		if phase := p.Now().Sub(v.blinkStart); (phase/blink)%2 == 0 || phase < 0 {
			cx, cy, ch := v.caretAt(v.caret, v.composeCaret, v.affinity)
			scale := p.Scale()
			p.Fill(ui.Rect{X: ox + cx, Y: oy + float32(cy-v.scroll), W: max(1/scale, 1), H: ch}, look.Caret, 0)
		}
	}
	if total := v.hs.top(len(v.paras)); total > float64(h) {
		top, size := v.thumbBox()
		p.Fill(ui.Rect{X: r.X + r.W - 8, Y: r.Y + top, W: 5, H: size}, look.Text.Alpha(0.3), 2.5)
	}
}

// paintBlock paints what a paragraph of style st shows behind its text:
// the background of a block of code, the bar of a quotation.
func (v *view) paintBlock(p *ui.Painter, i int, st ParagraphStyle, x, top, height, width float32) {
	look := &v.look
	indent := float32(st.Indent) * look.indent
	if st.CodeBlock {
		// One block for the paragraphs of code next to each other, rounded
		// at its ends.
		radius := look.size * look.CodeBlockRadius
		box := ui.Rect{X: x + indent, Y: top, W: width - indent, H: height}
		first := i == 0 || !v.para(i-1).style.CodeBlock
		last := i+1 >= len(v.paras) || !v.para(i+1).style.CodeBlock
		fill := box
		if !first {
			fill.Y, fill.H = fill.Y-2*radius, fill.H+2*radius
		}
		if !last {
			fill.H += 2 * radius
		}
		p.Clip(box, 0, func() { p.Fill(fill, look.CodeBlockBackground, radius) })
	}
	if st.Quote {
		w := look.QuoteBarWidth
		p.Fill(ui.Rect{X: x + indent, Y: top, W: w, H: height}, look.QuoteBar, w/2)
	}
}

// paintMarker paints the marker of an item of a list: its bullet, its
// number, or its check box.
func (v *view) paintMarker(p *ui.Painter, i int, st ParagraphStyle, pv *paraView, x, y float32) {
	if st.List == NoList {
		return
	}
	look := &v.look
	line := pv.layout.CaretAt(ui.TextCaretPosition{})
	font := v.paraFont(st)
	if st.List == Tasks {
		box := font.Size * 0.95
		bx := x - look.indent + (look.indent-box)/2 - look.indent*0.1
		by := y + line.Y + (line.H-box)/2
		if st.Checked {
			p.Fill(ui.Rect{X: bx, Y: by, W: box, H: box}, look.Checkbox, box*0.25)
			var path ui.Path
			path.MoveTo(bx+box*0.25, by+box*0.52).LineTo(bx+box*0.43, by+box*0.7).LineTo(bx+box*0.76, by+box*0.32)
			p.StrokePath(&path, max(1.5, box*0.12), look.Checkmark)
		} else {
			p.Stroke(ui.Rect{X: bx, Y: by, W: box, H: box}, look.Muted, box*0.25, 1.25)
		}
		return
	}
	mark := v.marker(i, st)
	if pv.markerLayout == nil || pv.marker != mark {
		pv.marker, pv.markerLayout = mark, ui.ShapeText(mark, font, 0)
	}
	ml := pv.markerLayout
	mw, _ := ml.Size()
	mx := x - mw - look.indent*0.25
	if st.List == Bullets {
		mx = x - look.indent*0.5 - mw/2
	}
	p.TextLayout(ml, mx, y+line.Y, look.Marker)
}

// listNumber returns the number of the item of a numbered list that
// paragraph p is: one more than the item of the same level before it, the
// deeper ones between them skipped, from 1 for the first.
func (v *view) listNumber(p int) int {
	if !v.numbersOf.Same(v.shown) {
		clear(v.numbers)
		v.numbersOf = v.shown
	}
	if v.numbers == nil {
		v.numbers = map[int]int{}
	}
	if n, ok := v.numbers[p]; ok {
		return n
	}
	st := v.para(p).style
	n := 1
	for q := p - 1; q >= 0; q-- {
		qs := v.para(q).style
		if qs.List == NoList || qs.Indent < st.Indent {
			break
		}
		if qs.Indent > st.Indent {
			continue
		}
		if qs.List == Numbers {
			n = v.listNumber(q) + 1
		}
		break
	}
	v.numbers[p] = n
	return n
}

// marker returns what marks item p of a list: a bullet for its level, or
// its number, in digits, letters or Roman numerals by level.
func (v *view) marker(p int, st ParagraphStyle) string {
	switch st.List {
	case Bullets:
		return []string{"•", "◦", "▪"}[st.Indent%3]
	case Numbers:
		n := v.listNumber(p)
		switch st.Indent % 3 {
		case 1:
			return alphaNumber(n) + "."
		case 2:
			return romanNumber(n) + "."
		}
		return strconv.Itoa(n) + "."
	}
	return ""
}

// alphaNumber returns n as letters: a, b, … z, aa.
func alphaNumber(n int) string {
	var b []byte
	for ; n > 0; n = (n - 1) / 26 {
		b = append([]byte{byte('a' + (n-1)%26)}, b...)
	}
	return string(b)
}

// romanNumber returns n in lowercase Roman numerals.
func romanNumber(n int) string {
	if n <= 0 || n >= 4000 {
		return strconv.Itoa(n)
	}
	vals := []int{1000, 900, 500, 400, 100, 90, 50, 40, 10, 9, 5, 4, 1}
	syms := []string{"m", "cm", "d", "cd", "c", "xc", "l", "xl", "x", "ix", "v", "iv", "i"}
	var b []byte
	for i, val := range vals {
		for n >= val {
			b = append(b, syms[i]...)
			n -= val
		}
	}
	return string(b)
}

// paintAttachments draws the attachments of paragraph p in the boxes its
// layout, at (x, y), keeps for them: an image, the app's drawing, or a
// box with the attachment's name.
func (v *view) paintAttachments(p *ui.Painter, i int, pv *paraView, x, y float32) {
	objects := pv.layout.Objects()
	if len(objects) == 0 {
		return
	}
	ps := v.shown.paraStart(i)
	for _, o := range objects {
		l := pv.runeOfUnit(o.Index)
		at := v.shown.AttributesAt(v.global(i, l)).Attachment
		if v.compose != "" && v.shown.paraAt(v.caret) == i && l >= v.caret-ps && l < v.caret-ps+len([]rune(v.compose)) {
			continue
		}
		r := ui.Rect{X: x + o.Rect.X, Y: y + o.Rect.Y, W: o.Rect.W, H: o.Rect.H}
		switch {
		case at == nil:
		case at.Draw != nil:
			at.Draw(p, r)
		case at.Image != nil:
			p.Image(at.Image, r, ui.Contain)
		default:
			// Not loaded: a box with its name.
			p.Fill(r, v.look.CodeBlockBackground, 4)
			p.Stroke(r, v.look.border, 4, 1)
			if at.Name != "" {
				l := ui.ShapeText(at.Name, ui.Font{Size: v.look.size * 0.85}, max(r.W-8, 1))
				_, lh := l.Size()
				p.Clip(r, 4, func() { p.TextLayout(l, r.X+4, r.Y+(r.H-lh)/2, v.look.Muted) })
			}
		}
	}
}
