// Package raster draws scenes in memory: the software renderer behind
// headless rendering, window captures and windows without a GPU renderer.
// Coverage comes from signed distances to rounded rectangles, so edges,
// corners and clips are anti-aliased the way the GPU shaders do it.
package raster

import (
	"image"
	"math"
	"runtime"
	"slices"

	"github.com/egoist/mygo/internal/scene"
)

// Image is a premultiplied BGRA bitmap, the layout Windows DIBs, cairo
// and Core Graphics take.
type Image struct {
	W, H, Stride int
	Pix          []byte
}

// NewImage returns a w×h image.
func NewImage(w, h int) *Image {
	return &Image{W: w, H: h, Stride: 4 * w, Pix: make([]byte, 4*w*h)}
}

// Resize makes the image w×h, reusing its memory when it can.
func (m *Image) Resize(w, h int) {
	m.W, m.H, m.Stride = w, h, 4*w
	if cap(m.Pix) >= 4*w*h {
		m.Pix = m.Pix[:4*w*h]
	} else {
		m.Pix = make([]byte, 4*w*h)
	}
}

// RGBA returns the pixels as premultiplied RGBA, as image.RGBA holds them.
func (m *Image) RGBA() []byte {
	out := make([]byte, 4*m.W*m.H)
	for y := 0; y < m.H; y++ {
		src := m.Pix[y*m.Stride:]
		dst := out[y*4*m.W:]
		for x := 0; x < m.W; x++ {
			s, d := src[4*x:4*x+4], dst[4*x:4*x+4]
			d[0], d[1], d[2], d[3] = s[2], s[1], s[0], s[3]
		}
	}
	return out
}

type clip struct {
	shape
	transform, inverse scene.Affine
	valid              bool
	round              bool
	// spans is where in spanBuf the clip, with continuous corners, keeps
	// the spans clipSolid found for the rows of the area, lo and hi a row
	// (noSpan until found), or -1.
	spans int
}

const noSpan = math.MinInt32

type renderer struct {
	dst   *Image
	s     *scene.Scene
	clips []clip
	// area is the part of dst to draw.
	area image.Rectangle
	// bounds is the intersection of the clip rectangles, in whole pixels.
	x0, y0, x1, y1 int
	// profile is scratch space for shadows.
	profile []float32
	// spanBuf holds the clips' spans (clip.spans).
	spanBuf []int32
	// blurred holds, by op, the images blurred of the ops that blur theirs.
	blurred []*blurredImage
}

// Render draws s into dst, which must be s.Width×s.Height.
func Render(dst *Image, s *scene.Scene) {
	drawCanvases(s)
	var r Renderer
	r.d.draw(dst, s, image.Rect(0, 0, dst.W, dst.H), r.opBounds(s, nil))
}

// A large area is drawn on several cores at once, in bands of bandRows
// rows that each takes in turn, as rows differ in how much they draw:
// each pixel is what drawing the area whole gives, as no pixel depends on
// another. An area of fewer than workArea pixels a core is not worth
// waking cores for, and maxWorkers is about where memory bandwidth and
// the lower clock of all cores stop paying for more.
//
// An effect costs much more a pixel and is often short, as a bar: the
// operations from one are drawn in bands of effectRows rows, so that more
// cores share it.
const (
	bandRows   = 64
	effectRows = 16
	workArea   = 128 << 10
	maxWorkers = 8
)

// draw draws the pixels of s within area into dst, with the renderers of
// d.rs, made as needed, leaving out the operations whose bounds (opBounds)
// miss the band drawn.
//
// Each effect is readied (EffectPixels.Begin) before its pixels, and one
// reading its backdrop reads pixels other bands draw: the operations up to
// each effect in the area are drawn first, then its backdrop is read
// (d.bd), then the effect and the operations up to the next. The area
// holds the backdrops of the effects in it (Renderer.diff).
func (d *drawer) draw(dst *Image, s *scene.Scene, area image.Rectangle, bounds []image.Rectangle) {
	area = area.Intersect(image.Rect(0, 0, dst.W, dst.H))
	d.blurred = d.blurImages(s, area, bounds)
	defer func() { d.blurred = nil }()
	// px draws operation from, an effect over the backdrop b.
	from, px, b := 0, scene.EffectPixels(nil), (*scene.BackdropImage)(nil)
	for i := range s.Ops {
		op := &s.Ops[i]
		if op.Kind != scene.OpEffect || int(op.Start) >= len(s.Effects) || !bounds[i].Overlaps(area) {
			continue
		}
		fx := &s.Effects[op.Start]
		next := d.pixels(fx.Effect)
		if next == nil {
			continue
		}
		d.drawOps(dst, s, area, bounds, from, i, px, b)
		b = nil
		if fx.Effect.Backdrop {
			d.bd.read(dst, scene.BackdropOf(op.Transform.Bounds(op.Rect), fx.Blur, dst.W, dst.H))
			d.bd.img.Transform = op.Transform
			b = &d.bd.img
		}
		rect := op.Rect
		if fx.Masked() {
			// Its shader's positions start from the box it fills.
			rect.X, rect.Y = fx.Origin[0], fx.Origin[1]
		}
		next.Begin(fx, rect, scene.FitRadii(op.Rect, op.Radii))
		from, px = i, next
	}
	d.drawOps(dst, s, area, bounds, from, len(s.Ops), px, b)
}

// drawOps draws operations from to to of s within area, as draw does,
// over the pixels the operations before them drew, or over the scene's
// clear color from the first; px draws operation from when it is an
// effect, over the backdrop b.
func (d *drawer) drawOps(dst *Image, s *scene.Scene, area image.Rectangle, bounds []image.Rectangle, from, to int, px scene.EffectPixels, b *scene.BackdropImage) {
	if from >= to && from > 0 {
		return
	}
	rows := bandRows
	if px != nil {
		rows = effectRows
	}
	bands := (area.Dy() + rows - 1) / rows
	n := max(min(runtime.GOMAXPROCS(0), area.Dx()*area.Dy()/workArea, bands, maxWorkers), 1)
	for len(d.rs) < n {
		d.rs = append(d.rs, renderer{})
	}
	for i := range d.rs {
		d.rs[i].blurred = d.blurred
	}
	if n == 1 {
		d.rs[0].render(dst, s, area, bounds, from, to, px, b)
		return
	}
	d.job = bandJob{dst, s, area, bounds, from, to, rows, px, b}
	if d.bands.do == nil {
		d.bands.do = d.band
	}
	d.bands.run(n, bands)
	d.job = bandJob{}
}

// bandJob is what the bands of drawOps draw: operations from to to of s
// within area, in bands of rows rows.
type bandJob struct {
	dst      *Image
	s        *scene.Scene
	area     image.Rectangle
	bounds   []image.Rectangle
	from, to int
	rows     int
	px       scene.EffectPixels
	b        *scene.BackdropImage
}

// band draws band part of d.job with the renderer of member.
func (d *drawer) band(member, part int) {
	j := &d.job
	band := j.area
	band.Min.Y = j.area.Min.Y + part*j.rows
	band.Max.Y = min(band.Min.Y+j.rows, j.area.Max.Y)
	d.rs[member].render(j.dst, j.s, band, j.bounds, j.from, j.to, j.px, j.b)
}

// render draws the pixels of s within area, operations from to to but
// those whose bounds miss it, over the clear color from the first; those
// before only clip. px draws operation from when it is an effect, over
// the backdrop b.
func (r *renderer) render(dst *Image, s *scene.Scene, area image.Rectangle, bounds []image.Rectangle, from, to int, px scene.EffectPixels, b *scene.BackdropImage) {
	r.dst, r.s, r.clips, r.spanBuf = dst, s, r.clips[:0], r.spanBuf[:0]
	r.area = area.Intersect(image.Rect(0, 0, dst.W, dst.H))
	if r.area.Empty() {
		return
	}
	if from == 0 {
		r.clear(s.Clear)
	}
	r.updateBounds()
	for i := range s.Ops[:to] {
		op := &s.Ops[i]
		if op.Kind != scene.OpPushClip && op.Kind != scene.OpPopClip && (i < from || !bounds[i].Overlaps(r.area)) {
			continue
		}
		switch op.Kind {
		case scene.OpFill:
			if op.Transform.Set {
				r.transformed(op, nil, nil)
			} else {
				r.fill(op)
			}
		case scene.OpShadow:
			if op.Transform.Set {
				r.transformed(op, nil, nil)
			} else if op.Inset {
				r.insetShadow(op)
			} else {
				r.shadow(op)
			}
		case scene.OpGlyphs:
			transformed := op.Transform.Set
			for _, g := range s.Glyphs[op.Start:op.End] {
				transformed = transformed || g.Transform.Set
			}
			if transformed {
				r.transformedGlyphs(op)
			} else {
				r.glyphs(op)
			}
		case scene.OpImage:
			if r.blurred != nil && r.blurred[i] != nil {
				// The image blurred, in its place.
				b := *op
				b.Image, b.Src, b.Blur = r.blurred[i].img, r.blurred[i].src, 0
				op = &b
			}
			if op.Transform.Set {
				r.transformed(op, nil, nil)
			} else {
				r.image(op)
			}
		case scene.OpHole:
			if op.Transform.Set {
				r.transformed(op, nil, nil)
			} else {
				r.hole(op)
			}
		case scene.OpEffect:
			if i == from && px != nil {
				if fx := &s.Effects[op.Start]; fx.Masked() {
					r.maskedEffect(op, fx, px)
				} else if op.Transform.Set {
					r.transformed(op, px, b)
				} else {
					r.effect(op, px, b)
				}
			}
		case scene.OpPushClip:
			radii := scene.Corners(op.Rect, op.Radii, op.Continuous)
			c := clip{shape: newShape(op.Rect, radii), round: hasRadii(radii), spans: -1}
			c.transform = op.Transform
			c.inverse, c.valid = op.Transform.Inverse()
			if c.continuous && !c.transform.Set {
				// Continuous corners take long to find the spans of, which
				// every operation within the clip needs.
				c.spans = len(r.spanBuf)
				n := 2 * r.area.Dy()
				r.spanBuf = slices.Grow(r.spanBuf, n)[:c.spans+n]
				for i := c.spans; i < len(r.spanBuf); i++ {
					r.spanBuf[i] = noSpan
				}
			}
			r.clips = append(r.clips, c)
			r.updateBounds()
		case scene.OpPopClip:
			if len(r.clips) > 0 {
				if c := r.clips[len(r.clips)-1]; c.spans >= 0 {
					r.spanBuf = r.spanBuf[:c.spans]
				}
				r.clips = r.clips[:len(r.clips)-1]
				r.updateBounds()
			}
		}
	}
}

func (r *renderer) clear(c scene.Color) {
	p := c.Premul(1)
	px := [4]byte{to8(p[2]), to8(p[1]), to8(p[0]), to8(p[3])}
	d, a := r.dst, r.area
	row := d.Pix[a.Min.Y*d.Stride+4*a.Min.X : a.Min.Y*d.Stride+4*a.Max.X]
	for x := 0; x < a.Dx(); x++ {
		copy(row[4*x:], px[:])
	}
	for y := a.Min.Y + 1; y < a.Max.Y; y++ {
		copy(d.Pix[y*d.Stride+4*a.Min.X:], row)
	}
}

func (r *renderer) updateBounds() {
	r.x0, r.y0, r.x1, r.y1 = r.area.Min.X, r.area.Min.Y, r.area.Max.X, r.area.Max.Y
	for _, c := range r.clips {
		b := c.transform.Bounds(c.r)
		if !c.valid {
			r.x1, r.y1 = r.x0, r.y0
			return
		}
		r.x0 = max(r.x0, int(math.Floor(float64(b.X))))
		r.y0 = max(r.y0, int(math.Floor(float64(b.Y))))
		r.x1 = min(r.x1, int(math.Ceil(float64(b.X+b.W))))
		r.y1 = min(r.y1, int(math.Ceil(float64(b.Y+b.H))))
	}
}

// pixelBounds returns the pixels a rectangle touches within the clip.
func (r *renderer) pixelBounds(rc scene.Rect) (x0, y0, x1, y1 int) {
	x0 = max(r.x0, int(math.Floor(float64(rc.X))))
	y0 = max(r.y0, int(math.Floor(float64(rc.Y))))
	x1 = min(r.x1, int(math.Ceil(float64(rc.X+rc.W))))
	y1 = min(r.y1, int(math.Ceil(float64(rc.Y+rc.H))))
	return
}

// clipCoverage returns how much of pixel (x, y) the clips let through,
// given the clip rectangles' pixel bounds already apply.
func (r *renderer) clipCoverage(x, y int) float32 {
	cov := float32(1)
	for i := range r.clips {
		c := &r.clips[i]
		px, py := float32(x)+0.5, float32(y)+0.5
		if c.transform.Set {
			if !c.valid {
				return 0
			}
			cov *= transformedCoverage(&c.shape, c.transform, c.inverse, px, py)
		} else if c.round {
			cov *= coverage(&c.shape, px, py)
		} else {
			// Partial pixels at fractional clip edges.
			cov *= clamp01(min(px-c.r.X, c.r.X+c.r.W-px)+0.5) * clamp01(min(py-c.r.Y, c.r.Y+c.r.H-py)+0.5)
		}
		if cov == 0 {
			return 0
		}
	}
	return cov
}

// clipSolid returns the pixels of row y that the clips let through
// entirely, so that clipCoverage need not run for them.
func (r *renderer) clipSolid(y int) (lo, hi int) {
	lo, hi = r.x0, r.x1
	for i := range r.clips {
		c := &r.clips[i]
		if c.transform.Set {
			return lo, lo
		}
		var l, h int
		if k := c.spans + 2*(y-r.area.Min.Y); c.spans >= 0 && y >= r.area.Min.Y && y < r.area.Max.Y {
			if r.spanBuf[k] == noSpan {
				l, h = solidSpan(&c.shape, float32(y), float32(y+1))
				r.spanBuf[k], r.spanBuf[k+1] = int32(l), int32(h)
			} else {
				l, h = int(r.spanBuf[k]), int(r.spanBuf[k+1])
			}
		} else {
			l, h = solidSpan(&c.shape, float32(y), float32(y+1))
		}
		lo, hi = max(lo, l), min(hi, h)
	}
	return lo, hi
}

func hasRadii(radii [4]float32) bool {
	return radii[0] != 0 || radii[1] != 0 || radii[2] != 0 || radii[3] != 0
}

// shape is a rounded rectangle, with radii as scene.Corners gives them, as
// the renderer draws it, with its continuous corners, which every pixel
// near an edge needs, worked out once.
type shape struct {
	r scene.Rect
	// radii are positive, continuous or not.
	radii      [4]float32
	continuous bool
	corners    [4]contCorner
}

func newShape(rc scene.Rect, radii [4]float32) shape {
	s := shape{r: rc, radii: radii}
	if radii[0] >= 0 && radii[1] >= 0 && radii[2] >= 0 && radii[3] >= 0 {
		return s
	}
	s.continuous = true
	for i := range s.radii {
		s.radii[i] = abs(radii[i])
	}
	for i, rad := range s.radii {
		// The corners sharing its horizontal and its vertical edge.
		h, v := [4]int{1, 0, 3, 2}[i], [4]int{3, 2, 1, 0}[i]
		s.corners[i] = newContCorner(rad, s.radii[h], s.radii[v], rc.W, rc.H)
	}
	return s
}

// coverage returns how much of the pixel centered at (px, py) the rounded
// rectangle covers, from the signed distance to its edge.
func coverage(s *shape, px, py float32) float32 {
	if s.continuous {
		return contCoverage(s, px, py)
	}
	rc, radii := s.r, &s.radii
	hx, hy := rc.W/2, rc.H/2
	qx, qy := px-rc.X-hx, py-rc.Y-hy
	var rad float32
	switch {
	case qx < 0 && qy < 0:
		rad = radii[0]
	case qx >= 0 && qy < 0:
		rad = radii[1]
	case qx >= 0:
		rad = radii[2]
	default:
		rad = radii[3]
	}
	if rad <= 0 {
		return areaCoverage(rc, px, py)
	}
	ax, ay := abs(qx)-hx+rad, abs(qy)-hy+rad
	var d float32
	if ax > 0 && ay > 0 {
		d = float32(math.Sqrt(float64(ax*ax+ay*ay))) - rad
	} else {
		d = max(ax, ay) - rad
	}
	return clamp01(0.5 - d)
}

// areaCoverage returns the area of the pixel centered at (px, py) inside
// the rectangle, near square corners: exact for lines thinner than a pixel
// too.
func areaCoverage(rc scene.Rect, px, py float32) float32 {
	cx := min(rc.X+rc.W, px+0.5) - max(rc.X, px-0.5)
	cy := min(rc.Y+rc.H, py+0.5) - max(rc.Y, py-0.5)
	return clamp01(cx) * clamp01(cy)
}

// contCoverage is coverage with continuous corners: away from their
// curves, the area of the pixel inside the rectangle; near them, the
// shape's value over the sum of its derivatives, as Core Animation
// antialiases (corner.go).
func contCoverage(s *shape, px, py float32) float32 {
	d, curved := s.contDist(px, py)
	if !curved {
		return areaCoverage(s.r, px, py)
	}
	if abs(d) > 2 {
		// Wholly in or out, however steep the value is.
		if d < 0 {
			return 1
		}
		return 0
	}
	const h = 1.0 / 16
	dx, _ := s.contDist(px+h, py)
	dy, _ := s.contDist(px, py+h)
	return clamp01(0.5 - d*h/max(abs(dx-d)+abs(dy-d), 1e-6))
}

// contDist returns the value of a shape with continuous corners at (px,
// py), the largest of its corners' and the rectangle's signed distance,
// and whether a corner's curve reaches it.
func (s *shape) contDist(px, py float32) (float32, bool) {
	rc := s.r
	left, right := px-rc.X, rc.X+rc.W-px
	top, bottom := py-rc.Y, rc.Y+rc.H-py
	d := max(-left, -right, -top, -bottom)
	curved := false
	for i, uv := range [4][2]float32{{left, top}, {right, top}, {right, bottom}, {left, bottom}} {
		c := &s.corners[i]
		if c.r <= 0 || uv[0] >= c.e || uv[1] >= c.e {
			continue
		}
		d, curved = max(d, c.dist(uv[0], uv[1])), true
	}
	return d, curved
}

// inset returns how far corner i of the rounded rectangle narrows a row v
// pixels from its horizontal edge, at the least.
func inset(s *shape, i int, v float32) float32 {
	if s.continuous {
		_, bound := s.corners[i].inset(v)
		return bound
	}
	rad := s.radii[i]
	switch {
	case rad <= 0 || v >= rad:
		return 0
	case v <= 0:
		return rad
	}
	dy := rad - v
	return rad - float32(math.Sqrt(float64(rad*rad-dy*dy)))
}

// solidSpan returns the pixels of the row between y0 and y1 that the
// rounded rectangle covers entirely; lo >= hi when there are none.
func solidSpan(s *shape, y0, y1 float32) (lo, hi int) {
	rc := s.r
	if y0 < rc.Y || y1 > rc.Y+rc.H {
		return 0, 0
	}
	// The corners narrow the row the most at its edge nearer to theirs.
	top, bottom := y0-rc.Y, rc.Y+rc.H-y1
	left := rc.X + max(inset(s, 0, top), inset(s, 3, bottom))
	right := rc.X + rc.W - max(inset(s, 1, top), inset(s, 2, bottom))
	return int(math.Ceil(float64(left))), int(math.Floor(float64(right)))
}

func abs(v float32) float32 {
	if v < 0 {
		return -v
	}
	return v
}

func clamp01(v float32) float32 {
	if v < 0 {
		return 0
	}
	if v > 1 {
		return 1
	}
	return v
}

func to8(v float32) byte {
	if v <= 0 {
		return 0
	}
	if v >= 1 {
		return 255
	}
	return byte(v*255 + 0.5)
}

// blend composites a premultiplied color with coverage cov over the pixel.
func blend(p []byte, c [4]float32, cov float32) {
	a := c[3] * cov
	if a <= 0 {
		return
	}
	inv := 1 - a
	p[0] = to8(c[2]*cov + float32(p[0])/255*inv)
	p[1] = to8(c[1]*cov + float32(p[1])/255*inv)
	p[2] = to8(c[0]*cov + float32(p[2])/255*inv)
	p[3] = to8(a + float32(p[3])/255*inv)
}

// blendSubpixel composites a straight color c over a pixel with the
// coverage a of each subpixel, times w, each channel by its own: as
// renderers blend subpixel glyphs with a second color for the source's
// alpha.
func blendSubpixel(p []byte, c, a [3]float32, w float32) {
	wr, wg, wb := a[0]*w, a[1]*w, a[2]*w
	wa := (wr + wg + wb) / 3
	p[0] = to8(c[2]*wb + float32(p[0])/255*(1-wb))
	p[1] = to8(c[1]*wg + float32(p[1])/255*(1-wg))
	p[2] = to8(c[0]*wr + float32(p[2])/255*(1-wr))
	p[3] = to8(wa + float32(p[3])/255*(1-wa))
}

// unpremul returns the straight color of a premultiplied one.
func unpremul(c [4]float32) [3]float32 {
	if c[3] <= 0 {
		return [3]float32{}
	}
	return [3]float32{c[0] / c[3], c[1] / c[3], c[2] / c[3]}
}

// blendSpan composites a premultiplied color with coverage cov over a run
// of pixels.
func blendSpan(row []byte, x0, x1 int, c [4]float32, cov float32) {
	if x1 <= x0 {
		return
	}
	a := c[3] * cov
	if a <= 0 {
		return
	}
	px := [4]byte{to8(c[2] * cov), to8(c[1] * cov), to8(c[0] * cov), to8(a)}
	run := row[4*x0 : 4*x1]
	if px[3] == 255 {
		// Opaque: copy the first pixel, then ever longer runs of them.
		copy(run, px[:])
		for n := 4; n < len(run); n *= 2 {
			copy(run[n:], run[:n])
		}
		return
	}
	inv := 255 - uint32(px[3])
	for i := 0; i < len(run); i += 4 {
		p := run[i : i+4 : i+4]
		p[0] = over(px[0], p[0], inv)
		p[1] = over(px[1], p[1], inv)
		p[2] = over(px[2], p[2], inv)
		p[3] = over(px[3], p[3], inv)
	}
}

// over returns src + dst×inv/255, rounded, for 8-bit premultiplied
// channels.
func over(src, dst byte, inv uint32) byte {
	t := uint32(dst)*inv + 128
	return byte(min(uint32(src)+(t+t>>8)>>8, 255))
}

// painter computes the fill of an op at pixel centers: its color, or its
// gradient or stripes, premultiplied and times the op's opacity.
type painter struct {
	paint      scene.Paint
	c1, c2     [4]float32 // premultiplied
	lab1, lab2 [3]float32 // premultiplied Oklab
	g          [4]float32
	origin     [2]float32
	stops      []paintStop
}

// paintStop is a stop of a gradient as the painter mixes it: its color
// premultiplied and times the opacity, the same in Oklab, and its place.
type paintStop struct {
	c   [4]float32
	lab [3]float32
	at  float32
}

func newPainter(op *scene.Op, opacity float32, stops []scene.Stop) painter {
	p := painter{paint: op.Paint, c1: op.Color.Premul(opacity), c2: op.Color2.Premul(opacity), g: op.Gradient, origin: [2]float32{op.Rect.X, op.Rect.Y}}
	if p.paint == scene.PaintOklab {
		p.lab1, p.lab2 = oklab(op.Color), oklab(op.Color2)
		for i := range 3 {
			p.lab1[i] *= p.c1[3]
			p.lab2[i] *= p.c2[3]
		}
	}
	if p.paint.HasStops() {
		for _, st := range opStops(op, stops) {
			c := st.Color.Premul(opacity)
			lab := oklab(st.Color)
			p.stops = append(p.stops, paintStop{c: c, lab: [3]float32{lab[0] * c[3], lab[1] * c[3], lab[2] * c[3]}, at: st.At})
		}
	}
	return p
}

// opStops returns the stops of op's gradient in stops, at most
// scene.MaxStops of them.
func opStops(op *scene.Op, stops []scene.Stop) []scene.Stop {
	a, b := int(op.Stops[0]), int(op.Stops[1])
	if a < 0 || b > len(stops) || a >= b {
		return nil
	}
	return stops[a:min(b, a+scene.MaxStops)]
}

// borderPainter returns the painters of op's inside and of its border,
// that of its border nil when BorderColor paints it: with BorderPaint, its
// gradient paints the border, and plain Color the inside.
func borderPainter(op *scene.Op, opacity float32, stops []scene.Stop) (fill painter, border *painter) {
	if !op.BorderPaint {
		return newPainter(op, opacity, stops), nil
	}
	b := newPainter(op, opacity, stops)
	return painter{paint: scene.PaintSolid, c1: op.Color.Premul(opacity)}, &b
}

// solid reports whether the fill is one color.
func (p *painter) solid() bool { return p.paint == scene.PaintSolid }

// visible reports whether the fill shows anywhere.
func (p *painter) visible() bool {
	if p.paint.HasStops() {
		for _, s := range p.stops {
			if s.c[3] > 0 {
				return true
			}
		}
		return false
	}
	return p.c1[3] > 0 || (!p.solid() && p.c2[3] > 0)
}

// at returns the fill at a pixel center.
func (p *painter) at(px, py float32) [4]float32 {
	switch p.paint {
	case scene.PaintSolid:
		return p.c1
	case scene.PaintStripes:
		s := (px-p.origin[0])*p.g[0] + (py-p.origin[1])*p.g[1]
		period := p.g[3]
		phase := s - period*float32(math.Floor(float64(s/period)))
		cov := clamp01(0.5 - min(max(-phase, phase-p.g[2]), period-phase))
		var c [4]float32
		for i := range c {
			c[i] = p.c1[i]*cov + p.c2[i]*(1-cov)
		}
		return c
	case scene.PaintLinearStops, scene.PaintLinearStopsOklab:
		g := p.g
		dx, dy := g[2]-g[0], g[3]-g[1]
		t := ((px-g[0])*dx + (py-g[1])*dy) / max(dx*dx+dy*dy, 0.0001)
		return p.stopsAt(t)
	case scene.PaintConic, scene.PaintConicOklab:
		t := conicT(px-p.g[0], py-p.g[1], p.g[2])
		return p.stopsAt(t)
	}
	g := p.g
	dx, dy := g[2]-g[0], g[3]-g[1]
	t := float32(0)
	if l := dx*dx + dy*dy; l > 0 {
		t = clamp01(((px-g[0])*dx + (py-g[1])*dy) / max(l, 0.0001))
	}
	a := p.c1[3]*(1-t) + p.c2[3]*t
	if p.paint == scene.PaintLinear {
		return [4]float32{p.c1[0]*(1-t) + p.c2[0]*t, p.c1[1]*(1-t) + p.c2[1]*t, p.c1[2]*(1-t) + p.c2[2]*t, a}
	}
	if a <= 0 {
		return [4]float32{}
	}
	var lab [3]float32
	for i := range lab {
		lab[i] = (p.lab1[i]*(1-t) + p.lab2[i]*t) / a
	}
	rgb := fromOklab(lab)
	return [4]float32{clamp01(rgb[0]) * a, clamp01(rgb[1]) * a, clamp01(rgb[2]) * a, a}
}

// conicT returns where (dx, dy) from a conic gradient's center lies
// around it, from 0 at angle (radians clockwise from upwards) to 1 a
// turn later, as the shaders compute it.
func conicT(dx, dy, angle float32) float32 {
	t := (float32(math.Atan2(float64(dx), float64(-dy))) - angle) * (1 / (2 * math.Pi))
	return t - float32(math.Floor(float64(t)))
}

// stopsAt returns the gradient's color at t: the first stop's before it,
// the last's after it, and between two stops their colors mixed, in sRGB
// or in Oklab, premultiplied, as the shaders' stopsAt.
func (p *painter) stopsAt(t float32) [4]float32 {
	s := p.stops
	n := len(s)
	if n == 0 {
		return [4]float32{}
	}
	if t <= s[0].at {
		return s[0].c
	}
	if t >= s[n-1].at {
		return s[n-1].c
	}
	i := 1
	for i < n-1 && t >= s[i].at {
		i++
	}
	a, b := &s[i-1], &s[i]
	k := clamp01((t - a.at) / max(b.at-a.at, 1e-6))
	alpha := a.c[3]*(1-k) + b.c[3]*k
	if p.paint == scene.PaintLinearStops || p.paint == scene.PaintConic {
		return [4]float32{a.c[0]*(1-k) + b.c[0]*k, a.c[1]*(1-k) + b.c[1]*k, a.c[2]*(1-k) + b.c[2]*k, alpha}
	}
	if alpha <= 0 {
		return [4]float32{}
	}
	var lab [3]float32
	for i := range lab {
		lab[i] = (a.lab[i]*(1-k) + b.lab[i]*k) / alpha
	}
	rgb := fromOklab(lab)
	return [4]float32{clamp01(rgb[0]) * alpha, clamp01(rgb[1]) * alpha, clamp01(rgb[2]) * alpha, alpha}
}

func toLinear(c float32) float32 {
	if c <= 0.04045 {
		return c / 12.92
	}
	return float32(math.Pow(float64((c+0.055)/1.055), 2.4))
}

func toSRGB(c float32) float32 {
	if c <= 0.0031308 {
		return c * 12.92
	}
	return 1.055*float32(math.Pow(float64(max(c, 0)), 1/2.4)) - 0.055
}

// oklab converts an sRGB color to Oklab, as the shaders do.
func oklab(c scene.Color) [3]float32 { return scene.Oklab(c) }

// fromOklab converts an Oklab color to sRGB components, unclamped.
func fromOklab(lab [3]float32) [3]float32 {
	l := lab[0] + 0.3963377774*lab[1] + 0.2158037573*lab[2]
	m := lab[0] - 0.1055613458*lab[1] - 0.0638541728*lab[2]
	s := lab[0] - 0.0894841775*lab[1] - 1.2914855480*lab[2]
	l, m, s = l*l*l, m*m*m, s*s*s
	return [3]float32{
		toSRGB(4.0767416621*l - 3.3077115913*m + 0.2309699292*s),
		toSRGB(-1.2684380046*l + 2.6097574011*m - 0.3413193965*s),
		toSRGB(-0.0041960863*l - 0.7034186147*m + 1.7076147010*s),
	}
}

// dash returns how much of a dashed border of widths w (top, right,
// bottom, left) around r shows at a pixel center, as the shaders compute
// it: each side, which the pixel belongs to when it is nearest that side's
// edge in widths of its border, has an odd number of dashes and gaps of
// equal length, about three widths, starting and ending with a dash.
func dash(r scene.Rect, w [4]float32, px, py float32) float32 {
	qx, qy := px-r.X, py-r.Y
	d := func(dist, width float32) float32 {
		if width > 0 {
			return dist / width
		}
		return 1e9
	}
	dt, dr, db, dl := d(qy, w[0]), d(r.W-qx, w[1]), d(r.H-qy, w[2]), d(qx, w[3])
	var s, length, bw float32
	switch {
	case dt <= dr && dt <= db && dt <= dl:
		s, length, bw = qx, r.W, w[0]
	case dr <= db && dr <= dl:
		s, length, bw = qy, r.H, w[1]
	case db <= dl:
		s, length, bw = r.W-qx, r.W, w[2]
	default:
		s, length, bw = r.H-qy, r.H, w[3]
	}
	n := max(1, float32(math.Floor(float64((length/(3*bw)+1)*0.5+0.5))))
	seg := length / (2*n - 1)
	k := float32(math.Floor(float64(s / seg)))
	f := s - k*seg
	edge := min(f, seg-f)
	if k-2*float32(math.Floor(float64(k*0.5))) < 0.5 {
		edge = -edge
	}
	return clamp01(0.5 - edge)
}

func (r *renderer) fill(op *scene.Op) {
	if op.Rect.Empty() {
		return
	}
	opacity := op.Opacity
	if opacity == 0 {
		opacity = 1
	}
	radii := scene.Corners(op.Rect, op.Radii, op.Continuous)
	outer := newShape(op.Rect, radii)
	pt, bp := borderPainter(op, opacity, r.s.Stops)
	border := op.BorderColor.Premul(opacity)
	bw := op.Border
	if op.BorderColor.A == 0 {
		bw = [4]float32{}
	}
	hasBorder := scene.HasBorder(bw)
	inner := outer
	if hasBorder {
		inner = newShape(scene.InnerRadii(op.Rect, radii, bw))
	}
	hasFill, solid := pt.visible(), pt.solid()
	x0, y0, x1, y1 := r.pixelBounds(op.Rect)
	for y := y0; y < y1; y++ {
		row := r.dst.Pix[y*r.dst.Stride:]
		py := float32(y) + 0.5
		cl, ch := r.clipSolid(y)
		ol, oh := solidSpan(&outer, float32(y), float32(y+1))
		il, ih := ol, oh
		if hasBorder {
			il, ih = 0, 0
			if !inner.r.Empty() {
				il, ih = solidSpan(&inner, float32(y), float32(y+1))
			}
		}
		// The middle run, inside the clips, the shape and its border, is
		// plain fill.
		sl, sh := max(il, cl, x0), min(ih, ch, x1)
		if sl < sh && hasFill && solid {
			blendSpan(row, sl, sh, pt.c1, 1)
		}
		for x := x0; x < x1; x++ {
			if x >= sl && x < sh {
				if !hasFill || solid {
					x = sh - 1 // past the run, which is done
					continue
				}
				blend(row[4*x:4*x+4], pt.at(float32(x)+0.5, py), 1)
				continue
			}
			px := float32(x) + 0.5
			clipCov := float32(1)
			if x < cl || x >= ch {
				clipCov = r.clipCoverage(x, y)
				if clipCov == 0 {
					continue
				}
			}
			oc := float32(1)
			if x < ol || x >= oh {
				oc = coverage(&outer, px, py)
				if oc == 0 {
					continue
				}
			}
			p := row[4*x : 4*x+4]
			if hasFill {
				blend(p, pt.at(px, py), oc*clipCov)
			}
			if hasBorder {
				ic := float32(0)
				if !inner.r.Empty() {
					ic = coverage(&inner, px, py)
				}
				bc := oc - ic
				if bc > 0 && op.Dashed {
					bc *= dash(op.Rect, bw, px, py)
				}
				if bc > 0 {
					blend(p, borderAt(bp, border, op.BorderColor, px, py), bc*clipCov)
				}
			}
		}
	}
}

// borderAt returns the color of a border at a pixel center: border, or,
// painted with a gradient (bp), the gradient's color there times
// BorderColor, straight.
func borderAt(bp *painter, border [4]float32, c scene.Color, px, py float32) [4]float32 {
	if bp == nil {
		return border
	}
	g := bp.at(px, py)
	k := [4]float32{float32(c.R) / 255, float32(c.G) / 255, float32(c.B) / 255, float32(c.A) / 255}
	return [4]float32{g[0] * k[0] * k[3], g[1] * k[1] * k[3], g[2] * k[2] * k[3], g[3] * k[3]}
}

// hole makes a rounded rectangle transparent within the clips: each pixel
// keeps what its coverage leaves of it, as renderers blend with a
// destination factor of one minus the coverage and none of the source.
func (r *renderer) hole(op *scene.Op) {
	if op.Rect.Empty() {
		return
	}
	opacity := op.Opacity
	if opacity == 0 {
		opacity = 1
	}
	shape := newShape(op.Rect, scene.Corners(op.Rect, op.Radii, op.Continuous))
	x0, y0, x1, y1 := r.pixelBounds(op.Rect)
	for y := y0; y < y1; y++ {
		row := r.dst.Pix[y*r.dst.Stride:]
		py := float32(y) + 0.5
		cl, ch := r.clipSolid(y)
		ol, oh := solidSpan(&shape, float32(y), float32(y+1))
		sl, sh := max(ol, cl, x0), min(oh, ch, x1)
		if sl < sh && opacity >= 1 {
			clear(row[4*sl : 4*sh])
		} else if sl < sh {
			for x := sl; x < sh; x++ {
				erase(row[4*x:4*x+4], opacity)
			}
		}
		for x := x0; x < x1; x++ {
			if x >= sl && x < sh {
				x = sh - 1 // past the run, which is done
				continue
			}
			cov := r.clipCoverage(x, y)
			if cov == 0 {
				continue
			}
			if x < ol || x >= oh {
				cov *= coverage(&shape, float32(x)+0.5, py)
			}
			if cov > 0 {
				erase(row[4*x:4*x+4], cov*opacity)
			}
		}
	}
}

// erase takes the coverage cov of a premultiplied pixel away.
func erase(p []byte, cov float32) {
	inv := 1 - cov
	for i := range p[:4] {
		p[i] = to8(float32(p[i]) / 255 * inv)
	}
}

// shadow draws a Gaussian-blurred rounded rectangle, integrating the blur
// along y numerically and along x exactly (Evan Wallace's method), outside
// the box casting it, if any.
func (r *renderer) shadow(op *scene.Op) {
	sigma := op.Blur / 2
	cast := !op.Cast.Empty()
	if sigma < 0.5 && !cast {
		f := *op
		f.Kind, f.Border, f.Paint = scene.OpFill, [4]float32{}, scene.PaintSolid
		r.fill(&f)
		return
	}
	opacity := op.Opacity
	if opacity == 0 {
		opacity = 1
	}
	c := op.Color.Premul(opacity)
	radii := scene.Corners(op.Rect, op.Radii, op.Continuous)
	shadowBox := newShape(op.Rect, radii)
	caster := newShape(op.Cast, scene.Corners(op.Cast, op.CastRadii, op.Continuous))
	// outside returns how much of pixel (x, y) the box casting the shadow
	// leaves to it.
	outside := func(x, y int) float32 {
		return 1 - coverage(&caster, float32(x)+0.5, float32(y)+0.5)
	}
	if sigma < 0.5 {
		// The box itself, outside the box casting it.
		x0, y0, x1, y1 := r.pixelBounds(op.Rect)
		for y := y0; y < y1; y++ {
			row := r.dst.Pix[y*r.dst.Stride:]
			for x := x0; x < x1; x++ {
				v := coverage(&shadowBox, float32(x)+0.5, float32(y)+0.5) * outside(x, y)
				if v > 0 {
					blend(row[4*x:4*x+4], c, v*r.clipCoverage(x, y))
				}
			}
		}
		return
	}
	corner := max(abs(radii[0]), abs(radii[1]), abs(radii[2]), abs(radii[3]))
	// Continuous corners narrow the box over the end of their curves.
	continuous := op.Continuous && corner > 0
	var cc contCorner
	if continuous {
		cc = newContCorner(corner, corner, corner, op.Rect.W, op.Rect.H)
	}
	ext := 3 * sigma
	box := scene.Rect{X: op.Rect.X - ext, Y: op.Rect.Y - ext, W: op.Rect.W + 2*ext, H: op.Rect.H + 2*ext}
	x0, y0, x1, y1 := r.pixelBounds(box)
	if x0 >= x1 || y0 >= y1 {
		return
	}
	cx, cy := op.Rect.X+op.Rect.W/2, op.Rect.Y+op.Rect.H/2
	hx, hy := op.Rect.W/2, op.Rect.H/2
	k := float32(math.Sqrt(0.5)) / sigma
	// The shadow is the box blurred along y, at four samples, of a box
	// blurred exactly along x whose width the corners narrow. Where no
	// corner narrows it, a row is one horizontal profile scaled, and in the
	// middle of a row, far from the narrowed edges, the profile is 1.
	if cap(r.profile) < x1-x0 {
		r.profile = make([]float32, x1-x0)
	}
	profile := r.profile[:x1-x0]
	for x := x0; x < x1; x++ {
		px := float32(x) + 0.5 - cx
		profile[x-x0] = 0.5 * (erf((px+hx)*k) - erf((px-hx)*k))
	}
	far := 2.6 / k // where erf passes 0.9997
	// kx0..kx1 and ky0..ky1 are the pixels the box casting the shadow
	// touches.
	var kx0, ky0, kx1, ky1 int
	if cast {
		kx0, ky0 = int(math.Floor(float64(op.Cast.X))), int(math.Floor(float64(op.Cast.Y)))
		kx1, ky1 = int(math.Ceil(float64(op.Cast.X+op.Cast.W))), int(math.Ceil(float64(op.Cast.Y+op.Cast.H)))
	}
	for y := y0; y < y1; y++ {
		py := float32(y) + 0.5 - cy
		low, high := py-hy, py+hy
		start := min(max(-ext, low), high)
		end := min(max(ext, low), high)
		step := (end - start) / 4
		var weight, half [4]float32
		var sum float32
		narrowest := hx
		yy := start + step*0.5
		for i := range 4 {
			weight[i] = gaussian(yy, sigma) * step
			sum += weight[i]
			half[i] = hx
			if v := hy - abs(py-yy); continuous {
				if v < cc.e {
					at, _ := cc.inset(v)
					half[i] = hx - at
					narrowest = min(narrowest, half[i])
				}
			} else if delta := min(hy-corner-abs(py-yy), 0); delta < 0 {
				half[i] = hx - corner + float32(math.Sqrt(float64(max(0, corner*corner-delta*delta))))
				narrowest = min(narrowest, half[i])
			}
			yy += step
		}
		if sum <= 0.002 {
			continue
		}
		straight := narrowest == hx
		row := r.dst.Pix[y*r.dst.Stride:]
		cl, ch := r.clipSolid(y)
		ml := max(int(math.Ceil(float64(cx-narrowest+far-0.5))), x0, cl)
		mh := min(int(math.Floor(float64(cx+narrowest-far-0.5)))+1, x1, ch)
		// On the row, the box casting the shadow touches kl..kh, which the
		// middle leaves out, and hides it in sl..sh.
		var kl, kh, sl, sh int
		if cast && y >= ky0 && y < ky1 {
			kl, kh = kx0, kx1
			sl, sh = solidSpan(&caster, float32(y), float32(y+1))
		}
		if kl < kh {
			blendSpan(row, ml, min(mh, kl), c, sum)
			blendSpan(row, max(ml, kh), mh, c, sum)
		} else {
			blendSpan(row, ml, mh, c, sum)
		}
		for x := x0; x < x1; x++ {
			if x >= ml && x < mh && (x < kl || x >= kh) {
				x = mh - 1
				if x < kl {
					x = min(mh, kl) - 1
				}
				continue
			}
			if x >= sl && x < sh {
				x = sh - 1
				continue
			}
			var v float32
			if straight {
				v = profile[x-x0] * sum
			} else {
				px := float32(x) + 0.5 - cx
				for i := range 4 {
					v += weight[i] * 0.5 * (erf((px+half[i])*k) - erf((px-half[i])*k))
				}
			}
			if x >= kl && x < kh {
				v *= outside(x, y)
			}
			if v <= 0.002 {
				continue
			}
			if x < cl || x >= ch {
				v *= r.clipCoverage(x, y)
			}
			blend(row[4*x:4*x+4], c, v)
		}
	}
}

// insetShadow draws an inner shadow: inside the box Cast, what the hole
// Rect, blurred as shadow blurs a box, leaves uncovered.
func (r *renderer) insetShadow(op *scene.Op) {
	x0, y0, x1, y1 := r.pixelBounds(op.Cast)
	if x0 >= x1 || y0 >= y1 {
		return
	}
	opacity := op.Opacity
	if opacity == 0 {
		opacity = 1
	}
	c := op.Color.Premul(opacity)
	box := newShape(op.Cast, scene.Corners(op.Cast, op.CastRadii, op.Continuous))
	radii := scene.Corners(op.Rect, op.Radii, op.Continuous)
	hole := newShape(op.Rect, radii)
	empty := op.Rect.Empty()
	sigma := op.Blur / 2
	if sigma < 0.5 || empty {
		for y := y0; y < y1; y++ {
			row := r.dst.Pix[y*r.dst.Stride:]
			// Nothing shows where the hole covers the row's pixels
			// entirely, as it does most of a large box.
			var sl, sh int
			if !empty {
				sl, sh = solidSpan(&hole, float32(y), float32(y+1))
			}
			for x := x0; x < x1; x++ {
				if x >= sl && x < sh {
					x = sh - 1
					continue
				}
				px, py := float32(x)+0.5, float32(y)+0.5
				v := coverage(&box, px, py)
				if !empty {
					v *= 1 - coverage(&hole, px, py)
				}
				if v > 0 {
					blend(row[4*x:4*x+4], c, v*r.clipCoverage(x, y))
				}
			}
		}
		return
	}
	// The hole blurred as shadow blurs a box, four samples along y and
	// exactly along x.
	corner := max(abs(radii[0]), abs(radii[1]), abs(radii[2]), abs(radii[3]))
	continuous := op.Continuous && corner > 0
	var cc contCorner
	if continuous {
		cc = newContCorner(corner, corner, corner, op.Rect.W, op.Rect.H)
	}
	ext := 3 * sigma
	cx, cy := op.Rect.X+op.Rect.W/2, op.Rect.Y+op.Rect.H/2
	hx, hy := op.Rect.W/2, op.Rect.H/2
	k := float32(math.Sqrt(0.5)) / sigma
	far := 2.6 / k // where erf passes 0.9997
	for y := y0; y < y1; y++ {
		py := float32(y) + 0.5 - cy
		low, high := py-hy, py+hy
		start := min(max(-ext, low), high)
		end := min(max(ext, low), high)
		step := (end - start) / 4
		var weight, half [4]float32
		var sum float32
		narrowest := hx
		yy := start + step*0.5
		for i := range 4 {
			weight[i] = gaussian(yy, sigma) * step
			sum += weight[i]
			half[i] = hx
			if v := hy - abs(py-yy); continuous {
				if v < cc.e {
					at, _ := cc.inset(v)
					half[i] = hx - at
					narrowest = min(narrowest, half[i])
				}
			} else if delta := min(hy-corner-abs(py-yy), 0); delta < 0 {
				half[i] = hx - corner + float32(math.Sqrt(float64(max(0, corner*corner-delta*delta))))
				narrowest = min(narrowest, half[i])
			}
			yy += step
		}
		row := r.dst.Pix[y*r.dst.Stride:]
		cl, ch := r.clipSolid(y)
		// In the middle of the row, far from the hole's edges, the hole
		// covers sum of each pixel, and nothing shows where that is all.
		ml, mh := x1, x1
		if 1-sum <= 0.002 {
			ml = max(int(math.Ceil(float64(cx-narrowest+far-0.5))), x0)
			mh = min(int(math.Floor(float64(cx+narrowest-far-0.5)))+1, x1)
		}
		for x := x0; x < x1; x++ {
			if x >= ml && x < mh {
				x = mh - 1
				continue
			}
			px := float32(x) + 0.5
			var covered float32
			for i := range 4 {
				covered += weight[i] * 0.5 * (erf((px-cx+half[i])*k) - erf((px-cx-half[i])*k))
			}
			v := (1 - covered) * coverage(&box, px, float32(y)+0.5)
			if v <= 0.002 {
				continue
			}
			if x < cl || x >= ch {
				v *= r.clipCoverage(x, y)
			}
			blend(row[4*x:4*x+4], c, v)
		}
	}
}

func gaussian(x, sigma float32) float32 {
	return float32(math.Exp(float64(-(x*x)/(2*sigma*sigma)))) / (float32(math.Sqrt(2*math.Pi)) * sigma)
}

// erf approximates the error function within 5e-4, as the GPU renderers
// do (Abramowitz and Stegun 7.1.27).
func erf(x float32) float32 {
	a := abs(x)
	t := 1 + (0.278393+(0.230389+0.078108*(a*a))*a)*a
	t *= t
	e := 1 - 1/(t*t)
	if x < 0 {
		return -e
	}
	return e
}

func (r *renderer) glyphs(op *scene.Op) {
	var pt *painter
	if op.Paint == scene.PaintLinear || op.Paint == scene.PaintOklab || op.Paint.HasStops() {
		opacity := op.Opacity
		if opacity == 0 {
			opacity = 1
		}
		p := newPainter(op, opacity, r.s.Stops)
		pt = &p
	}
	text := r.s.Text
	for _, g := range r.s.Glyphs[op.Start:op.End] {
		atlas := r.s.MaskAtlas
		if g.Colored || g.Subpixel {
			atlas = r.s.ColorAtlas
		}
		if atlas == nil {
			continue
		}
		gx, gy := int(math.Round(float64(g.X))), int(math.Round(float64(g.Y)))
		x0, y0 := max(gx, r.x0), max(gy, r.y0)
		x1, y1 := min(gx+int(g.UW), r.x1), min(gy+int(g.VH), r.y1)
		if x0 >= x1 || y0 >= y1 {
			continue
		}
		tint := g.Color.Premul(1)
		alpha := float32(g.Color.A) / 255
		contrast, boost := text.Contrast, float32(0)
		if g.Subpixel {
			contrast = text.SubpixelContrast
		}
		if g.Thin {
			boost = scene.ThinBoost
		}
		correct := contrast != 0 || boost != 0 || text.GammaRatios != [4]float32{}
		if !g.Colored && !g.Subpixel && pt == nil && !correct && g.Color.A == 255 {
			r.maskOpaque(atlas, g, gx, gy, image.Rect(x0, y0, x1, y1))
			continue
		}
		for y := y0; y < y1; y++ {
			row := r.dst.Pix[y*r.dst.Stride:]
			cl, ch := r.clipSolid(y)
			ay := int(g.V) + y - gy
			for x := x0; x < x1; x++ {
				ax := int(g.U) + x - gx
				cov := float32(1)
				if x < cl || x >= ch {
					if cov = r.clipCoverage(x, y); cov == 0 {
						continue
					}
				}
				p := row[4*x : 4*x+4]
				switch {
				case g.Colored:
					s := atlas.Pix[(ay*atlas.W+ax)*4:]
					c := [4]float32{float32(s[0]) / 255 * alpha, float32(s[1]) / 255 * alpha, float32(s[2]) / 255 * alpha, float32(s[3]) / 255 * alpha}
					blend(p, c, cov)
				case g.Subpixel:
					s := atlas.Pix[(ay*atlas.W+ax)*4:]
					if s[0]|s[1]|s[2] == 0 {
						continue
					}
					if pt != nil {
						tint = pt.at(float32(x)+0.5, float32(y)+0.5)
					}
					c := unpremul(tint)
					a := [3]float32{float32(s[0]) / 255, float32(s[1]) / 255, float32(s[2]) / 255}
					if correct {
						a = scene.SubpixelCoverage(a, c, contrast, boost, text.GammaRatios)
					}
					blendSubpixel(p, c, a, tint[3]*cov)
				default:
					m := atlas.Pix[ay*atlas.W+ax]
					if m == 0 {
						continue
					}
					if pt != nil {
						tint = pt.at(float32(x)+0.5, float32(y)+0.5)
					}
					a := float32(m) / 255
					if correct {
						a = scene.TextCoverage(a, unpremul(tint), contrast, boost, text.GammaRatios)
					}
					blend(p, tint, cov*a)
				}
			}
		}
	}
}

// maskOpaque draws ordinary opaque text with 8-bit coverage. Integer
// blending avoids converting every destination channel to and from
// floats; rounded clip edges still use the general coverage calculation.
func (r *renderer) maskOpaque(a *scene.Atlas, g scene.Glyph, gx, gy int, box image.Rectangle) {
	c := g.Color
	for y := box.Min.Y; y < box.Max.Y; y++ {
		cl, ch := r.clipSolid(y)
		at := (int(g.V)+y-gy)*a.W + int(g.U) + box.Min.X - gx
		masks := a.Pix[at : at+box.Dx()]
		row := r.dst.Pix[y*r.dst.Stride+4*box.Min.X:][:4*box.Dx()]
		for i, m := range masks {
			if m == 0 {
				continue
			}
			p := row[4*i : 4*i+4]
			if x := box.Min.X + i; x < cl || x >= ch {
				blend(p, c.Premul(1), r.clipCoverage(x, y)*(float32(m)/255))
				continue
			}
			if m == 255 {
				p[0], p[1], p[2], p[3] = c.B, c.G, c.R, 255
				continue
			}
			weight := uint32(m)
			p[0] = mixMask(c.B, p[0], weight)
			p[1] = mixMask(c.G, p[1], weight)
			p[2] = mixMask(c.R, p[2], weight)
			p[3] = over(m, p[3], 255-weight)
		}
	}
}

func mixMask(src, dst byte, coverage uint32) byte {
	v := uint32(src)*coverage + uint32(dst)*(255-coverage) + 128
	return byte((v + v>>8) >> 8)
}

func (r *renderer) image(op *scene.Op) {
	img := op.Image
	if img == nil || img.W == 0 || img.H == 0 || op.Rect.Empty() || unflipped(op.Src).Empty() {
		return
	}
	opacity := op.Opacity
	if opacity == 0 {
		opacity = 1
	}
	radii := scene.Corners(op.Rect, op.Radii, op.Continuous)
	box := newShape(op.Rect, radii)
	round := hasRadii(radii)
	sx, sy := op.Src.W/op.Rect.W, op.Src.H/op.Rect.H
	x0, y0, x1, y1 := r.pixelBounds(op.Rect)
	for y := y0; y < y1; y++ {
		row := r.dst.Pix[y*r.dst.Stride:]
		py := float32(y) + 0.5
		cl, ch := r.clipSolid(y)
		ol, oh := solidSpan(&box, float32(y), float32(y+1))
		for x := x0; x < x1; x++ {
			px := float32(x) + 0.5
			cov := opacity
			if x < ol || x >= oh || !round {
				cov *= coverage(&box, px, py)
			}
			if x < cl || x >= ch {
				cov *= r.clipCoverage(x, y)
			}
			if cov <= 0 {
				continue
			}
			c := sampleOp(op, op.Src.X+(px-op.Rect.X)*sx, op.Src.Y+(py-op.Rect.Y)*sy, abs32(sx), abs32(sy))
			blend(row[4*x:4*x+4], c, cov)
		}
	}
}

// sampleOp reads op's image at (u, v), where a pixel spans du by dv of
// its pixels, as op samples, grays and tints it.
func sampleOp(op *scene.Op, u, v, du, dv float32) [4]float32 {
	if op.Pixelated {
		u, v = pixelated(u, du), pixelated(v, dv)
	}
	c := sample(op.Image, u, v, unflipped(op.Src))
	if op.Grayscale {
		l := 0.2126*c[0] + 0.7152*c[1] + 0.0722*c[2]
		c[0], c[1], c[2] = l, l, l
	}
	if t := op.Color; t != (scene.Color{}) {
		a := float32(t.A) / 255
		c[0] *= float32(t.R) / 255 * a
		c[1] *= float32(t.G) / 255 * a
		c[2] *= float32(t.B) / 255 * a
		c[3] *= a
	}
	return c
}

// unflipped returns the rectangle a source rectangle covers, which has a
// negative width or height along an axis it flips.
func unflipped(r scene.Rect) scene.Rect {
	if r.W < 0 {
		r.X, r.W = r.X+r.W, -r.W
	}
	if r.H < 0 {
		r.Y, r.H = r.Y+r.H, -r.H
	}
	return r
}

// pixelated moves t, a coordinate in an image's pixels, so that sampling
// it bilinearly draws each pixel as a square of its color, blended with
// the next only across the pixel of the frame, d of the image's pixels
// wide, that their edge crosses, as the GPU renderers do (Ebitengine's
// FilterPixelated).
func pixelated(t, d float32) float32 {
	c := t - 0.5
	i := float32(math.Floor(float64(c)))
	f := (c-i-0.5)/min(max(d, 1e-5), 1) + 0.5
	return i + min(max(f, 0), 1) + 0.5
}

func abs32(v float32) float32 { return float32(math.Abs(float64(v))) }

// sample reads a premultiplied pixel at (u, v) with bilinear filtering,
// clamped to src.
func sample(img *scene.Image, u, v float32, src scene.Rect) [4]float32 {
	u, v = u-0.5, v-0.5
	x0, y0 := int(math.Floor(float64(u))), int(math.Floor(float64(v)))
	tx, ty := u-float32(x0), v-float32(y0)
	minX, minY := int(src.X), int(src.Y)
	maxX, maxY := int(math.Ceil(float64(src.X+src.W)))-1, int(math.Ceil(float64(src.Y+src.H)))-1
	maxX, maxY = min(maxX, img.W-1), min(maxY, img.H-1)
	at := func(x, y int) []byte {
		x, y = max(minX, min(x, maxX)), max(minY, min(y, maxY))
		return img.Pix[(y*img.W+x)*4:]
	}
	p00, p10, p01, p11 := at(x0, y0), at(x0+1, y0), at(x0, y0+1), at(x0+1, y0+1)
	var c [4]float32
	for i := 0; i < 4; i++ {
		top := float32(p00[i])*(1-tx) + float32(p10[i])*tx
		bot := float32(p01[i])*(1-tx) + float32(p11[i])*tx
		c[i] = (top*(1-ty) + bot*ty) / 255
	}
	return c
}
