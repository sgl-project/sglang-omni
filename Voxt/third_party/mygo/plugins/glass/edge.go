package glass

import "github.com/egoist/mygo/ui"

// ScrollEdge is the scroll edge effect of macOS 26 and later, a
// ui.Material: what macOS draws over content scrolling under a bar of
// controls, so that they stand out from it, as SwiftUI's safeAreaBar with
// scrollEdgeEffectStyle does. Fill an element over the content's edge
// with it, under the bar, as long as the bar and 10 DIPs more when soft,
// as long as the bar when hard:
//
//	ui.Box(c).Absolute().Top(0).Left(0).Right(0).Height(74).PassThrough().Material(glass.ScrollEdge{})
//
// Its looks are macOS 27's, measured from SwiftUI's: the soft edge fades
// the content into its background, with no blur; the hard edge frosts it.
type ScrollEdge struct {
	// Hard makes the edge hard: what is under it blurred by about 6 DIPs, its
	// colors saturated, under 82% of the background, evenly, with a
	// hairline along its other edge. Soft, it fades the background over
	// the content, from 85% at the bar's edge to nothing at the other.
	Hard bool
	// Bottom puts the bar at the element's bottom, the effect rising from
	// it, rather than at its top.
	Bottom bool
	// Background is what the content scrolls over, which the effect lays
	// over it; unset, the theme's Background.
	Background ui.Color
}

// PaintMaterial paints the scroll edge over box, rounded by radii.
func (e ScrollEdge) PaintMaterial(p *ui.Painter, box ui.Rect, radii [4]float32) {
	if box.W <= 0 || box.H <= 0 {
		return
	}
	bg := e.Background
	if bg.A == 0 {
		bg = p.Theme().Background
	}
	if !e.Hard {
		// The background, its alpha 0.85 at the bar's edge, falling
		// linearly to 0 at the other, as AppKit's soft pocket replays the
		// window's background under a gradient mask, in sRGB.
		g := ui.LinearGradient{From: bg.Alpha(softEdge), To: bg.Alpha(0), Angle: 180}
		if e.Bottom {
			g.Angle = 0
		}
		fillGradient(p, box, radii, g)
		return
	}
	// Under a hard edge, the content blurred, saturated by 1.25
	// and lightened by 0.03, as AppKit's HardPocketContentBlur, under the
	// background at 82.45%, desaturated to 0.947 and lightened by 0.03, as
	// its HardPocketBackgroundReplay, evenly.
	s := p.Scale()
	b := hardEdgeBlur * s
	c := toned([3]float32{float32(bg.R) / 255, float32(bg.G) / 255, float32(bg.B) / 255}, 0.947, 0.03)
	p2, p3 := blurTone{saturation: 0.25, offset: 0.03, mix: 0.8245, color: c}.params()
	p.Effect(blurEffect, box, radii, b, [5][4]float32{{}, {b, b, -1, 0}, p2, p3})
	// The hairline along the other edge, outside it, half a DIP: black at
	// 10% in light mode, white at 7% in dark mode.
	line := ui.RGBA(0, 0, 0, 0.1)
	if p.Theme().Dark {
		line = ui.RGBA(255, 255, 255, 0.07)
	}
	h := max(0.5, 1/s)
	line = line.Alpha(0.5 / h)
	y := box.Y + box.H
	if e.Bottom {
		y = box.Y - h
	}
	p.Fill(ui.Rect{X: box.X, Y: y, W: box.W, H: h}, line, 0)
}

// String describes the scroll edge, as the inspector shows it.
func (e ScrollEdge) String() string {
	s := "scroll-edge soft"
	if e.Hard {
		s = "scroll-edge hard"
	}
	if e.Bottom {
		s += " bottom"
	}
	return s
}

// softEdge is the alpha of the background at a soft scroll edge's bar,
// AppKit's; hardEdgeBlur is how much a hard edge blurs, in DIPs: AppKit's
// Gaussian of radius 6 spreads an edge 30 pixels from 10% to 90% at a
// scale of 2, as the renderers' does at 6.4.
const (
	softEdge     = 0.85
	hardEdgeBlur = 6.4
)

// toned returns c saturated by saturation around its luminance, with
// offset added, clamped, as AppKit's color matrices tone.
func toned(c [3]float32, saturation, offset float32) [3]float32 {
	l := 0.2126*c[0] + 0.7152*c[1] + 0.0722*c[2]
	for i := range c {
		c[i] = min(max(l+(c[i]-l)*saturation+offset, 0), 1)
	}
	return c
}

// fillGradient fills box, rounded by radii, with g: as one rounded
// rectangle when its corners are alike, else as a path.
func fillGradient(p *ui.Painter, box ui.Rect, radii [4]float32, g ui.LinearGradient) {
	if radii[0] == radii[1] && radii[1] == radii[2] && radii[2] == radii[3] {
		p.FillGradient(box, g, radii[0])
		return
	}
	var path ui.Path
	x0, y0, x1, y1 := box.X, box.Y, box.X+box.W, box.Y+box.H
	// A cubic Bézier's control points for a quarter circle are k of the
	// radius from its ends.
	const k = 0.5523
	tl, tr, br, bl := radii[0], radii[1], radii[2], radii[3]
	path.MoveTo(x0+tl, y0).LineTo(x1-tr, y0)
	path.CubeTo(x1-tr+k*tr, y0, x1, y0+tr-k*tr, x1, y0+tr).LineTo(x1, y1-br)
	path.CubeTo(x1, y1-br+k*br, x1-br+k*br, y1, x1-br, y1).LineTo(x0+bl, y1)
	path.CubeTo(x0+bl-k*bl, y1, x0, y1-bl+k*bl, x0, y1-bl).LineTo(x0, y0+tl)
	path.CubeTo(x0, y0+tl-k*tl, x0+tl-k*tl, y0, x0+tl, y0).Close()
	p.FillPathGradient(&path, g)
}
