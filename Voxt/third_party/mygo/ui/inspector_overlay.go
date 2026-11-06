//go:build !mygo_noinspector

package ui

import "fmt"

// Chrome's colors for the boxes of an element highlighted over the page.
var (
	inspHiContent = RGBA(111, 168, 220, 0.66)
	inspHiPadding = RGBA(147, 196, 125, 0.55)
	inspHiBorder  = RGBA(255, 229, 153, 0.66)
	inspHiMargin  = RGBA(246, 178, 107, 0.66)
)

// paintHighlight highlights over the content the element under the
// pointer, in the tree or picking, else the element chosen while the tree
// has the focus, as Chrome does: its content in blue, padding in green,
// border in yellow and margins in orange, with a tooltip telling what it
// is.
func (in *inspector) paintHighlight(rt *engine, p *Painter, h float32) {
	e := in.hoverElem
	if e == nil && in.listID != 0 && rt.focused == in.listID {
		e = in.selElem
	}
	if e == nil || e.leaving != 0 {
		return
	}
	savedClip, savedOpacity := p.clip, p.opacity
	p.pushClip(Rect{0, 0, in.appW, h}, [4]float32{})
	p.opacity = 1
	defer func() {
		p.popClip()
		p.clip, p.opacity = savedClip, savedOpacity
	}()
	border := Rect{e.x, e.y, e.w, e.h}
	margin := Rect{border.X - e.m(3), border.Y - e.m(0), border.W + e.marginX(), border.H + e.marginY()}
	padding := Rect{border.X + e.border[3], border.Y + e.border[0], border.W - e.border[1] - e.border[3], border.H - e.border[0] - e.border[2]}
	content := e.contentBox()
	savedTransform := p.transform
	p.transform = e.world
	ring(p, margin, border, inspHiMargin)
	ring(p, border, padding, inspHiBorder)
	ring(p, padding, content, inspHiPadding)
	p.Fill(content, inspHiContent, 0)
	p.transform = savedTransform
	in.paintTooltip(rt, p, e, e.visualRect(margin), h)
}

// ring fills the room between outer and inner, which it holds.
func ring(p *Painter, outer, inner Rect, c Color) {
	if inner.W <= 0 || inner.H <= 0 {
		p.Fill(outer, c, 0)
		return
	}
	p.Fill(Rect{outer.X, outer.Y, outer.W, inner.Y - outer.Y}, c, 0)
	p.Fill(Rect{outer.X, inner.Y + inner.H, outer.W, outer.Y + outer.H - inner.Y - inner.H}, c, 0)
	p.Fill(Rect{outer.X, inner.Y, inner.X - outer.X, inner.H}, c, 0)
	p.Fill(Rect{inner.X + inner.W, inner.Y, outer.X + outer.W - inner.X - inner.W, inner.H}, c, 0)
}

// tipRow is a row of the tooltip: a label and a value, after a swatch of
// a color when swatch is set.
type tipRow struct {
	label, value string
	color        Color
	swatch       bool
}

// paintTooltip paints the tooltip of e, whose margin box is box, above
// it, else below, as Chrome's: what it is and its size, its colors, font
// and spacing, and what assistive technology sees.
func (in *inspector) paintTooltip(rt *engine, p *Painter, e *node, box Rect, h float32) {
	const (
		pad  = 8
		size = 11
		gap  = 4
	)
	ink, muted := Hex("#202124"), Hex("#5f6368")
	ts := e.resolvedText()
	var rows []tipRow
	if e.kind == kindText || e.kind == kindIcon {
		rows = append(rows, tipRow{label: "Color", value: colorText(ts.color), color: ts.color, swatch: true})
		family := ts.family
		if family == "" {
			family = "system-ui"
		}
		rows = append(rows, tipRow{label: "Font", value: pxText(ts.size) + " " + family})
	}
	if e.bg.A > 0 && e.fill == fillColor {
		rows = append(rows, tipRow{label: "Background", value: colorText(e.bg), color: e.bg, swatch: true})
	}
	if v, ok := edgesCSS(e.pad); ok {
		rows = append(rows, tipRow{label: "Padding", value: v})
	}
	if v, ok := edgesCSS(e.margin); ok {
		rows = append(rows, tipRow{label: "Margin", value: v})
	}
	role, _ := accessibleRole(e)
	name := e.label
	if name == "" && e.kind == kindText {
		name = clip(e.text, 40)
	}
	focusable := e.flags&flagFocusable != 0 && !e.IsDisabled()
	acc := []tipRow{{label: "Name", value: name}, {label: "Role", value: role}, {label: "Keyboard-focusable", value: "✓"}}
	if !focusable {
		acc[2].value = "⊘"
	}

	// Measure, then paint: the header, the rows, a line, the accessibility.
	span := func(s string, c Color, weight int) Span {
		return Span{Text: s, Color: c, Size: size, Weight: weight, Font: in.theme.Font}
	}
	// The tooltip is light over light and dark content alike, as Chrome's.
	title := []Span{span(elementName(e), Hex("#881280"), 600)}
	if k, ok := in.keys[e.id]; ok {
		title = append(title, span("#"+clip(fmt.Sprint(k), 24), Hex("#1a1aa6"), 400))
	}
	dims := span(num(e.w)+" × "+num(e.h), muted, 400)
	tw, th := p.MeasureText(0, title...)
	dw, _ := p.MeasureText(0, dims)
	labelW := float32(0)
	for _, r := range append(rows, acc...) {
		w, _ := p.MeasureText(0, span(r.label, muted, 400))
		labelW = max(labelW, w)
	}
	valueW := float32(0)
	for _, r := range append(rows, acc...) {
		w, _ := p.MeasureText(0, span(r.value, ink, 400))
		if r.swatch {
			w += 14
		}
		valueW = max(valueW, w)
	}
	lineH := th + gap
	width := max(tw+16+dw, labelW+16+valueW, 160) + 2*pad
	height := pad + th + gap + float32(len(rows))*lineH + 9 + th + gap + float32(len(acc))*lineH + pad - gap
	width = min(width, max(in.appW-8, 100))

	// Above the element, else below it, else inside its top.
	const arrow = 7
	x := max(4, min(box.X, in.appW-width-4))
	y := box.Y - height - arrow
	below := y < 4
	if below {
		y = box.Y + box.H + arrow
		if y+height > h-4 {
			y, below = max(4, box.Y+4), true
		}
	}
	bg := Hex("#ffffff")
	r := Rect{x, y, width, height}
	p.Shadow(r, 4, 0, 2, 8, 0, RGBA(0, 0, 0, 0.3))
	p.Fill(r, bg, 4)
	p.Stroke(r, Hex("#d0d0d0"), 4, 1)
	// The arrow, pointing at the element.
	ax := max(x+10, min(box.X+12, x+width-10))
	var path Path
	if below {
		path.MoveTo(ax-arrow, y+0.5)
		path.LineTo(ax, y-arrow+0.5)
		path.LineTo(ax+arrow, y+0.5)
	} else {
		path.MoveTo(ax-arrow, y+height-0.5)
		path.LineTo(ax, y+height+arrow-0.5)
		path.LineTo(ax+arrow, y+height-0.5)
	}
	path.Close()
	p.FillPath(&path, bg)

	cx, cy := x+pad, y+pad
	p.RichText(cx, cy, 0, title...)
	p.RichText(x+width-pad-dw, cy, 0, dims)
	cy += th + gap
	row := func(t tipRow, valueColor Color) {
		p.RichText(cx, cy, 0, span(t.label, muted, 400))
		vx := x + width - pad - valueW
		if t.swatch {
			p.Fill(Rect{vx, cy + (th-10)/2, 10, 10}, t.color, 0)
			p.Stroke(Rect{vx, cy + (th-10)/2, 10, 10}, Hex("#c0c0c0"), 0, 1)
			vx += 14
		}
		p.RichText(vx, cy, 0, span(t.value, valueColor, 400))
		cy += lineH
	}
	for _, t := range rows {
		row(t, ink)
	}
	p.Fill(Rect{x + pad, cy + 3, width - 2*pad, 1}, Hex("#e0e0e0"), 0)
	cy += 9
	p.RichText(cx, cy, 0, span("ACCESSIBILITY", muted, 600))
	cy += th + gap
	for i, t := range acc {
		c := ink
		if i == 2 {
			c = Hex("#188038")
			if !focusable {
				c = muted
			}
		}
		row(t, c)
	}
}
