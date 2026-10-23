// Package ink is the overlay's ink check: what a picture of the overlay's
// window shows, measured from its pixels, against the face's layout
// (10-15: the pill's icon side looked wider than its waveform's, which no
// frame check saw).
// The same check reads the Tester's pictures on Linux (overlay_test.go)
// and the Mac's captures of the real window (overlay_check.go).
//
// The face is where the window is opaque (the shadow is faint); its ink
// is what is bright on it: the icon, the bars, the text (the hairline
// and the dark fill are not). Distances are in DIPs.
package ink

import (
	"fmt"
	"image"
	"math"

	"github.com/sgl-project/sglang-omni/VoxtNew/internal/overlay/layout"
)

// Want is what the face should show.
type Want struct {
	W, H int // the face's size
	// row: the pill's row is there (the icon, then the waveform or the
	// loader); even: its ink is as far from the face's left edge as from
	// its right (the waveform's: not the loader's, nor a notice's).
	Row, Even bool
	// text: a line of text under the row; scrolled: it was wider than
	// the line, so it ends at the right padding and is cut at the left.
	Text, Scrolled bool
	// card: the answer card: its ink starts at its left padding and
	// ends above its bottom one, at most gap DIPs above it (no empty
	// space under a short text; a long one scrolls to the padding).
	Card bool
	Gap  float64
}

// Ink is what the picture shows, in DIPs from the face's top left.
type Ink struct {
	faceW, faceH float64
	// The row's ink: the icon's (the first stroke) and the rest's, the
	// left and right edges and the middle of each, top to bottom.
	iconL, iconMid float64
	restR, restMid float64
	// The text line's ink, left and right edges (0, 0: none).
	textL, textR float64
	// All the face's ink: its left edge and its bottom.
	inkL, inkB float64
	// The answer card's header: its ink's top and bottom (the first band
	// of ink rows), and the rule under it (a row lighter than the fill
	// across the card; 0: none found).
	headT, headB, rule float64
}

func (k Ink) String() string {
	s := fmt.Sprintf("face %.1f×%.1f; header %.1f–%.1f, rule %.1f; ink: left %.1f, %.1f above the bottom; row: left %.1f, right %.1f, icon middle %.1f, waveform middle %.1f",
		k.faceW, k.faceH, k.headT, k.headB, k.rule, k.inkL, k.faceH-k.inkB, k.iconL, k.faceW-k.restR, k.iconMid, k.restMid)
	if k.textR > 0 {
		s += fmt.Sprintf("; text: left %.1f, right %.1f, %.1f wide", k.textL, k.faceW-k.textR, k.textR-k.textL)
	}
	return s
}

// Tolerances: of the layout's own sizes, and of the ink's: a glyph's ink
// is a little narrower than its advance, an edge's antialiasing.
const (
	ovInkTol     = 1.0
	ovInkSizeTol = 0.6
	ovInkTextTol = 2.0
)

// measureOverlayInk measures img, a picture of the overlay's window at
// scale pixels a DIP, the face's row padY DIPs from its top.
func measureOverlayInk(img image.Image, scale float64, padY float64, row, text bool) (Ink, error) {
	b := img.Bounds()
	alpha := func(x, y int) uint32 { _, _, _, a := img.At(x, y).RGBA(); return a >> 8 }
	bright := func(x, y int) bool {
		r, g, bl, a := img.At(x, y).RGBA()
		if a>>8 < 128 {
			return false
		}
		// Unpremultiplied, the least channel: white ink, not the fill.
		m := min(r, g, bl) * 255 / max(a, 1)
		return m > 150
	}
	x0, x1, y0, y1 := b.Max.X, b.Min.X-1, b.Max.Y, b.Min.Y-1
	for y := b.Min.Y; y < b.Max.Y; y++ {
		for x := b.Min.X; x < b.Max.X; x++ {
			if alpha(x, y) > 128 {
				x0, x1, y0, y1 = min(x0, x), max(x1, x), min(y0, y), max(y1, y)
			}
		}
	}
	if x1 < x0 {
		return Ink{}, fmt.Errorf("no face")
	}
	var k Ink
	k.faceW, k.faceH = float64(x1+1-x0)/scale, float64(y1+1-y0)/scale
	inset := int(math.Ceil(2 * scale)) // past the hairline
	il, ib := math.MaxInt, -1
	for y := y0 + inset; y <= y1-inset; y++ {
		for x := x0 + inset; x <= x1-inset; x++ {
			if bright(x, y) {
				il, ib = min(il, x), max(ib, y)
			}
		}
	}
	if ib >= 0 {
		k.inkL, k.inkB = float64(il-x0)/scale, float64(ib+1-y0)/scale
	}
	k.headT, k.headB, k.rule = measureCardHead(img, scale, x0, y0, x1, y1, bright)
	if !row {
		return k, nil
	}
	// The row: its band, the columns that have ink in it, split at the
	// first gap wider than 6 DIPs (the icon's slot, the gap, the bars'
	// air: 12 or more; a glyph's own gaps are narrower).
	rowTop, rowBottom := y0+int(math.Round(padY*scale)), y0+int(math.Round((padY+layout.RowH)*scale))
	if !text {
		rowBottom = y1 + 1 - inset
	}
	var cols []int
	for x := x0 + inset; x <= x1-inset; x++ {
		for y := max(rowTop, y0+inset); y < rowBottom; y++ {
			if bright(x, y) {
				cols = append(cols, x)
				break
			}
		}
	}
	if len(cols) == 0 {
		return k, fmt.Errorf("no ink in the row")
	}
	split := len(cols)
	for i := 1; i < len(cols); i++ {
		if float64(cols[i]-cols[i-1]) > 6*scale {
			split = i
			break
		}
	}
	if split == len(cols) {
		return k, fmt.Errorf("the row's ink is one stroke")
	}
	mid := func(xa, xb int) float64 {
		ya, yb := math.MaxInt, -1
		for x := xa; x <= xb; x++ {
			for y := max(rowTop, y0+inset); y < rowBottom; y++ {
				if bright(x, y) {
					ya, yb = min(ya, y), max(yb, y)
				}
			}
		}
		return (float64(ya+yb+1)/2 - float64(y0)) / scale
	}
	k.iconL = float64(cols[0]-x0) / scale
	k.iconMid = mid(cols[0], cols[split-1])
	k.restR = float64(cols[len(cols)-1]+1-x0) / scale
	k.restMid = mid(cols[split], cols[len(cols)-1])
	if text {
		lineTop := rowBottom + int(math.Round(2*scale))
		tl, tr := math.MaxInt, -1
		for y := lineTop; y <= y1-inset; y++ {
			for x := x0 + inset; x <= x1-inset; x++ {
				if bright(x, y) {
					tl, tr = min(tl, x), max(tr, x)
				}
			}
		}
		if tr < 0 {
			return k, fmt.Errorf("no ink in the text's line")
		}
		k.textL, k.textR = float64(tl-x0)/scale, float64(tr+1-x0)/scale
	}
	return k, nil
}

// measureCardHead finds, in the face x0,y0–x1,y1 of img, its first band
// of ink rows (the answer card's header) and, under it, the rule: the
// row whose fill is lightest against the rows 2 DIPs over and under it,
// across the middle of the face. In DIPs from the face's top.
func measureCardHead(img image.Image, scale float64, x0, y0, x1, y1 int, bright func(x, y int) bool) (top, bottom, rule float64) {
	inset := int(math.Ceil(3 * scale))
	ht, hb := -1, -1
	for y := y0 + inset; y <= y1-inset; y++ {
		ink := false
		for x := x0 + inset; x <= x1-inset && !ink; x++ {
			ink = bright(x, y)
		}
		switch {
		case ink && ht < 0:
			ht = y
		case !ink && ht >= 0:
			hb = y
		}
		if hb >= 0 {
			break
		}
	}
	if ht < 0 || hb < 0 {
		return 0, 0, 0
	}
	// Each row's mean least channel over the face's middle.
	mean := func(y int) float64 {
		var sum float64
		n := 0
		for x := x0 + int(30*scale); x <= x1-int(30*scale); x++ {
			r, g, b, a := img.At(x, y).RGBA()
			if a == 0 {
				continue
			}
			sum += float64(min(r, g, b)) / float64(a) * 255
			n++
		}
		return sum / float64(max(n, 1))
	}
	best, by := 0.0, -1
	d := int(math.Round(2 * scale))
	for y := hb; y <= min(hb+int(24*scale), y1-d); y++ {
		if c := mean(y) - (mean(y-d)+mean(y+d))/2; c > best {
			best, by = c, y
		}
	}
	top, bottom = float64(ht-y0)/scale, float64(hb-y0)/scale
	if by >= 0 && best > 3 {
		rule = float64(by-y0) / scale
	}
	return top, bottom, rule
}

// Check measures img and says what is wrong with it: the
// face's size, the row's sides, its middle, the text's sides.
func Check(img image.Image, scale float64, want Want) (Ink, []string) {
	padY := float64(layout.PillPadY)
	if want.Text {
		padY = layout.LivePadY
	}
	k, err := measureOverlayInk(img, scale, padY, want.Row, want.Text)
	if err != nil {
		return k, []string{err.Error()}
	}
	var bad []string
	failf := func(format string, args ...any) { bad = append(bad, fmt.Sprintf(format, args...)) }
	if math.Abs(k.faceW-float64(want.W)) > ovInkSizeTol || math.Abs(k.faceH-float64(want.H)) > ovInkSizeTol {
		failf("the face is %.1f×%.1f, not %d×%d", k.faceW, k.faceH, want.W, want.H)
	}
	if want.Card {
		if math.Abs(k.faceW-layout.AnswerW) > ovInkSizeTol {
			failf("the card is %.1f wide, not %d", k.faceW, layout.AnswerW)
		}
		// The header between the card's top and the rule: as much room
		// over its ink as under it (10-15).
		switch {
		case k.rule == 0:
			failf("no rule under the header")
		case math.Abs(k.rule-layout.CardRuleY) > ovInkTol:
			failf("the rule is %.1f from the top, not %d", k.rule, layout.CardRuleY)
		case math.Abs((k.headT+k.headB)/2-k.rule/2) > 1.5:
			failf("the header's ink is %.1f under the top and %.1f over the rule", k.headT, k.rule-k.headB)
		}
		// The icon's glyph sits a little inside its slot.
		if k.inkL < layout.CardPadX-ovInkTol || k.inkL > layout.CardPadX+4 {
			failf("the card's ink starts %.1f from its left edge, not %d", k.inkL, layout.CardPadX)
		}
		under := k.faceH - k.inkB
		if under < layout.CardPadY-ovInkTextTol || under > layout.CardPadY+want.Gap {
			failf("the card's ink ends %.1f above its bottom edge, not %d to %.0f", under, layout.CardPadY, layout.CardPadY+want.Gap)
		}
	}
	if !want.Row {
		return k, bad
	}
	left, right := k.iconL, k.faceW-k.restR
	if want.Even && math.Abs(left-right) > ovInkTol {
		failf("the row's ink is %.1f from the left edge, %.1f from the right", left, right)
	}
	if d := k.iconMid - k.restMid; math.Abs(d) > ovInkTol {
		failf("the icon's middle is %.1f, the waveform's %.1f", k.iconMid, k.restMid)
	}
	if rowMid := padY + layout.RowH/2; math.Abs(k.restMid-rowMid) > ovInkTol {
		failf("the row's middle is %.1f, not %.1f", k.restMid, rowMid)
	}
	if !want.Text {
		return k, bad
	}
	tl, tr := k.textL, k.faceW-k.textR
	switch {
	case want.Scrolled:
		// The newest words end at the right padding; the oldest go under
		// the left one, fading.
		if math.Abs(tr-layout.LivePadX) > ovInkTextTol || tl < layout.LivePadX-ovInkTol {
			failf("the scrolled text is %.1f from the left edge, %.1f from the right (%d)", tl, tr, layout.LivePadX)
		}
		if math.Abs(k.faceW-layout.LiveMaxW) > ovInkSizeTol {
			failf("the face with long text is %.1f wide, not %d", k.faceW, layout.LiveMaxW)
		}
	default:
		if math.Abs(tl-tr) > ovInkTol {
			failf("the text is %.1f from the left edge, %.1f from the right", tl, tr)
		}
		// As wide as the text and the padding (its ink a little narrower
		// than its advance), or else as the row and the padding (the
		// icon's ink a little inside its slot, the bars' in theirs).
		byText := k.faceW-(k.textR-k.textL+2*layout.LivePadX) <= 2*ovInkTextTol
		byRow := left <= layout.LivePadX+4 && right <= layout.LivePadX+4
		if !byText && !byRow {
			failf("the face is %.1f wide, wider than its text (%.1f) and its row", k.faceW, k.textR-k.textL)
		}
		if tl < layout.LivePadX-ovInkTol {
			failf("the text is cut: %.1f from the left edge", tl)
		}
	}
	return k, bad
}
