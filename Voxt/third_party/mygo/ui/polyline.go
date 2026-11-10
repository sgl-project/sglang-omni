package ui

import (
	"math"

	"github.com/egoist/mygo/internal/scene"
)

// StrokePolyline draws a line through points in DIPs, width DIPs wide,
// with round joins and caps, as StrokePath draws a path of straight lines:
// a NaN or infinite point breaks the line there, and a point alone draws
// nothing. It suits lines that change every frame, as live charts: on a
// GPU, an opaque line draws each segment as a rounded rectangle instead of
// rasterizing a mask of the whole line, so that its edges can look a
// little bolder where segments overlap. Translucent lines draw as
// StrokePath's, which keeps overlaps from showing darker.
func (p *Painter) StrokePolyline(points [][2]float32, width float32, c Color) {
	finite := func(v float32) bool { return !math.IsNaN(float64(v)) && !math.IsInf(float64(v), 0) }
	if width <= 0 || !finite(width) || c.A == 0 {
		return
	}
	f := &p.rt.paths.flat
	f.pts, f.starts, f.closed = f.pts[:0], f.starts[:0], f.closed[:0]
	f.trimmed, f.kept = false, 0
	from := 0
	flush := func() {
		if len(f.pts)-from > 1 {
			f.starts = append(f.starts, from)
			f.closed = append(f.closed, false)
		} else {
			f.pts = f.pts[:from]
		}
		from = len(f.pts)
	}
	for _, pt := range points {
		pt = [2]float32{pt[0] * p.scale, pt[1] * p.scale}
		if !finite(pt[0]) || !finite(pt[1]) {
			flush()
			continue
		}
		f.pts = append(f.pts, pt)
	}
	flush()
	w := width * p.scale
	if !finite(w) {
		return
	}
	h, ok := p.rt.host.(*windowHost)
	if !ok || !h.onGPU || h.degraded || c.A != 255 || p.opacity != 1 {
		p.drawPath(f, w, c, nil)
		return
	}
	hw := w / 2
	wide := p.wide(c, Color{}, Color{})
	for i := range f.polys() {
		pts := f.poly(i)
		for j := 1; j < len(pts); j++ {
			a, b := pts[j-1], pts[j]
			dx, dy := b[0]-a[0], b[1]-a[1]
			length := float32(math.Hypot(float64(dx), float64(dy)))
			if !finite(length) {
				continue
			}
			m := scene.Translation(a[0], a[1])
			if length > 0 {
				dx, dy = dx/length, dy/length
				m = scene.Matrix(dx, dy, -dy, dx, a[0], a[1])
			}
			p.add(scene.Op{Kind: scene.OpFill, Rect: scene.Rect{X: -hw, Y: -hw, W: length + w, H: w}, Radii: scene.Uniform(hw), Color: c.scene(), Wide: wide, Opacity: p.opacity, Transform: m})
		}
	}
}
