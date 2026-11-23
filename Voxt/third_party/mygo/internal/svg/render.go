package svg

import (
	"image"
	"math"

	"github.com/egoist/mygo/internal/vec"
)

// renderer draws nodes into layers of premultiplied RGBA pixels, w×h.
type renderer struct {
	w, h    int
	current rgba
	z       vec.Rasterizer
	// cov is the coverage of what is being drawn, a rectangle of the
	// canvas.
	cov                   []byte
	flat, dashed, outline polys
	stroker               stroker
	free                  []*layer
}

// layer is a canvas: rows of 4×w bytes, with what was drawn into it
// within dirty.
type layer struct {
	pix   []byte
	dirty image.Rectangle
}

func (r *renderer) layer() *layer {
	if n := len(r.free); n > 0 {
		l := r.free[n-1]
		r.free = r.free[:n-1]
		return l
	}
	return &layer{pix: make([]byte, r.w*r.h*4)}
}

// release clears a layer for reuse.
func (r *renderer) release(l *layer) {
	d := l.dirty
	for y := d.Min.Y; y < d.Max.Y; y++ {
		clear(l.pix[(y*r.w+d.Min.X)*4 : (y*r.w+d.Max.X)*4])
	}
	l.dirty = image.Rectangle{}
	r.free = append(r.free, l)
}

// node draws n, whose parent's user space maps to pixels by ctm, into dst
// with an opacity.
func (r *renderer) node(n *node, ctm matrix, dst *layer, alpha float64) {
	a := alpha * n.opacity
	if a <= 0 {
		return
	}
	m := ctm.mul(n.m)
	if n.clip == nil && n.mask == nil && (n.opacity >= 1 || n.single) {
		r.content(n, m, dst, a)
		return
	}
	// Group opacity, clipping and masking apply to the content as a whole.
	l := r.layer()
	r.content(n, m, l, 1)
	if n.clip != nil {
		r.clip(l, n.clip, m, n.bbox, n.hasBox)
	}
	if n.mask != nil {
		r.mask(l, n.mask, m, n.bbox, n.hasBox)
	}
	r.composite(dst, l, a)
	r.release(l)
}

func (r *renderer) content(n *node, m matrix, dst *layer, alpha float64) {
	s := n.shape
	if s == nil {
		for _, k := range n.kids {
			r.node(k, m, dst, alpha)
		}
		return
	}
	if s.strokeFirst {
		r.stroke(s, m, dst, alpha, n.bbox)
		r.fill(s, m, dst, alpha, n.bbox)
	} else {
		r.fill(s, m, dst, alpha, n.bbox)
		r.stroke(s, m, dst, alpha, n.bbox)
	}
}

func (r *renderer) fill(s *shape, m matrix, dst *layer, alpha float64, bbox box) {
	if s.fill.kind == paintNone {
		return
	}
	r.flat.reset()
	s.path.flatten(m, true, &r.flat)
	if rect, ok := r.cover(&r.flat, s.evenOdd); ok {
		r.paint(dst, rect, s.fill, alpha*s.fillOpacity, m, bbox)
	}
}

func (r *renderer) stroke(s *shape, m matrix, dst *layer, alpha float64, bbox box) {
	if s.stroke.kind == paintNone || s.width <= 0 {
		return
	}
	scale := m.stretch()
	if !(scale > 0) {
		return
	}
	r.flat.reset()
	s.path.flatten(m, false, &r.flat)
	lines := &r.flat
	if len(s.dashes) > 0 {
		r.dashed.reset()
		dash(&r.flat, s.dashes, s.dashOffset, &r.dashed, 1<<16)
		lines = &r.dashed
	}
	r.outline.reset()
	st := &r.stroker
	st.hw, st.join, st.cap, st.miter, st.tol, st.out = s.width/2, s.join, s.cap, s.miter, 0.1/scale, &r.outline
	st.stroke(lines)
	r.outline.transform(m)
	if rect, ok := r.cover(&r.outline, false); ok {
		r.paint(dst, rect, s.stroke, alpha*s.strokeOpacity, m, bbox)
	}
}

// cover rasterizes polygons in pixels into r.cov, returning the rectangle
// of the canvas it covers.
func (r *renderer) cover(ps *polys, evenOdd bool) (image.Rectangle, bool) {
	if ps.len() == 0 {
		return image.Rectangle{}, false
	}
	// Shapes far beyond the canvas, or not finite, draw nothing.
	const far = 1 << 24
	x0, y0 := math.Inf(1), math.Inf(1)
	x1, y1 := math.Inf(-1), math.Inf(-1)
	for _, p := range ps.pts {
		if !(p.x > -far && p.x < far && p.y > -far && p.y < far) {
			return image.Rectangle{}, false
		}
		if p.x < x0 {
			x0 = p.x
		}
		if p.x > x1 {
			x1 = p.x
		}
		if p.y < y0 {
			y0 = p.y
		}
		if p.y > y1 {
			y1 = p.y
		}
	}
	rect := image.Rect(int(math.Floor(x0)), int(math.Floor(y0)), int(math.Ceil(x1))+1, int(math.Ceil(y1))+1).
		Intersect(image.Rect(0, 0, r.w, r.h))
	if rect.Empty() {
		return image.Rectangle{}, false
	}
	w, h := rect.Dx(), rect.Dy()
	r.z.Reset(w, h)
	ox, oy := float64(rect.Min.X), float64(rect.Min.Y)
	for i := range ps.len() {
		pts := ps.poly(i)
		r.z.MoveTo(float32(pts[0].x-ox), float32(pts[0].y-oy))
		for _, p := range pts[1:] {
			r.z.LineTo(float32(p.x-ox), float32(p.y-oy))
		}
		r.z.ClosePath()
	}
	if cap(r.cov) < w*h {
		r.cov = make([]byte, w*h)
	}
	r.cov = r.cov[:w*h]
	if evenOdd {
		r.z.MaskEvenOdd(r.cov, w)
	} else {
		r.z.Mask(r.cov, w)
	}
	return rect, true
}

// div255 divides by 255, rounding, for x up to 255×255.
func div255(x uint32) uint32 { return (x + 128 + (x+128)>>8) >> 8 }

// over composites a premultiplied color over the pixel d.
func over(d []byte, r, g, b, a uint32) {
	ia := 255 - a
	d[0] = uint8(r + div255(uint32(d[0])*ia))
	d[1] = uint8(g + div255(uint32(d[1])*ia))
	d[2] = uint8(b + div255(uint32(d[2])*ia))
	d[3] = uint8(a + div255(uint32(d[3])*ia))
}

// premul returns c with an opacity, premultiplied, from 0 to 255.
func premul(c rgba, alpha float64) [4]uint32 {
	a := math.Max(0, math.Min(1, c.a*alpha))
	q := func(v float64) uint32 { return uint32(math.Max(0, math.Min(1, v))*a*255 + 0.5) }
	return [4]uint32{q(c.r), q(c.g), q(c.b), uint32(a*255 + 0.5)}
}

// paint paints the coverage in r.cov, of rect, with a paint and an
// opacity. m maps the user space of the shape it covers to pixels.
func (r *renderer) paint(dst *layer, rect image.Rectangle, p paint, alpha float64, m matrix, bbox box) {
	w := rect.Dx()
	switch p.kind {
	case paintColor, paintCurrent:
		col := p.color
		if p.kind == paintCurrent {
			col = r.current
		}
		c := premul(col, alpha)
		if c[3] == 0 {
			return
		}
		for y := rect.Min.Y; y < rect.Max.Y; y++ {
			cov := r.cov[(y-rect.Min.Y)*w : (y-rect.Min.Y+1)*w]
			row := dst.pix[(y*r.w+rect.Min.X)*4:]
			for x, v := range cov {
				if v == 0 {
					continue
				}
				if v == 255 && c[3] == 255 {
					d := row[x*4 : x*4+4]
					d[0], d[1], d[2], d[3] = uint8(c[0]), uint8(c[1]), uint8(c[2]), 255
					continue
				}
				k := uint32(v)
				over(row[x*4:x*4+4], div255(c[0]*k), div255(c[1]*k), div255(c[2]*k), div255(c[3]*k))
			}
		}
	case paintGradient:
		g := p.grad
		gm := m
		if g.bbox {
			if bbox.w == 0 || bbox.h == 0 {
				return // the gradient has no box to span
			}
			gm = gm.mul(boxMatrix(bbox))
		}
		inv, ok := gm.mul(g.m).invert()
		if !ok {
			return
		}
		var lut [256][4]uint32
		g.lut(r.current, alpha, &lut)
		for y := rect.Min.Y; y < rect.Max.Y; y++ {
			cov := r.cov[(y-rect.Min.Y)*w : (y-rect.Min.Y+1)*w]
			row := dst.pix[(y*r.w+rect.Min.X)*4:]
			for x, v := range cov {
				if v == 0 {
					continue
				}
				t, ok := g.at(inv.apply(point{float64(rect.Min.X+x) + 0.5, float64(y) + 0.5}))
				if !ok {
					continue
				}
				c := &lut[int(t*255+0.5)]
				k := uint32(v)
				over(row[x*4:x*4+4], div255(c[0]*k), div255(c[1]*k), div255(c[2]*k), div255(c[3]*k))
			}
		}
	default:
		return
	}
	dst.dirty = dst.dirty.Union(rect)
}

// at returns where a point in the gradient's space falls along it, spread
// into [0, 1]; ok is false where a radial gradient does not reach.
func (g *gradient) at(p point) (float64, bool) {
	var t float64
	if !g.radial {
		dx, dy := g.x2-g.x1, g.y2-g.y1
		d := dx*dx + dy*dy
		if d == 0 {
			return 1, true // the last stop's color
		}
		t = ((p.x-g.x1)*dx + (p.y-g.y1)*dy) / d
	} else {
		if g.r <= 0 {
			return 1, true
		}
		// The largest t for which p is on the circle around
		// f + t(c - f) of radius fr + t(r - fr).
		cdx, cdy := g.cx-g.fx, g.cy-g.fy
		pdx, pdy := p.x-g.fx, p.y-g.fy
		dr := g.r - g.fr
		a := cdx*cdx + cdy*cdy - dr*dr
		b := pdx*cdx + pdy*cdy + g.fr*dr
		c := pdx*pdx + pdy*pdy - g.fr*g.fr
		if math.Abs(a) < 1e-12 {
			if b == 0 {
				return 0, false
			}
			t = c / (2 * b)
		} else {
			disc := b*b - a*c
			if disc < 0 {
				return 0, false
			}
			s := math.Sqrt(disc)
			t0, t1 := (b+s)/a, (b-s)/a
			t = math.Max(t0, t1)
			if g.fr+t*dr < 0 {
				t = math.Min(t0, t1)
				if g.fr+t*dr < 0 {
					return 0, false
				}
			}
		}
	}
	switch g.spread {
	case spreadReflect:
		t = math.Mod(math.Abs(t), 2)
		if t > 1 {
			t = 2 - t
		}
	case spreadRepeat:
		t -= math.Floor(t)
	default:
		t = math.Max(0, math.Min(1, t))
	}
	if !(t >= 0 && t <= 1) {
		return 0, false
	}
	return t, true
}

// lut fills out with the gradient's colors at 256 positions, with an
// opacity, premultiplied.
func (g *gradient) lut(current rgba, alpha float64, out *[256][4]uint32) {
	stops := g.stops
	color := func(s *stop) rgba {
		c := s.color
		if s.current {
			c = current
		}
		c.a *= s.opacity
		return c
	}
	j := 0
	for i := range out {
		t := float64(i) / 255
		for j+1 < len(stops) && stops[j+1].offset <= t {
			j++
		}
		var c rgba
		switch {
		case t <= stops[0].offset:
			c = color(&stops[0])
		case j == len(stops)-1:
			c = color(&stops[j])
		default:
			a, b := color(&stops[j]), color(&stops[j+1])
			f := (t - stops[j].offset) / (stops[j+1].offset - stops[j].offset)
			c = rgba{a.r + (b.r-a.r)*f, a.g + (b.g-a.g)*f, a.b + (b.b-a.b)*f, a.a + (b.a-a.a)*f}
		}
		out[i] = premul(c, alpha)
	}
}

// composite draws src over dst with an opacity.
func (r *renderer) composite(dst, src *layer, alpha float64) {
	d := src.dirty
	if d.Empty() {
		return
	}
	k := uint32(math.Max(0, math.Min(1, alpha))*255 + 0.5)
	for y := d.Min.Y; y < d.Max.Y; y++ {
		i0, i1 := (y*r.w+d.Min.X)*4, (y*r.w+d.Max.X)*4
		s, t := src.pix[i0:i1], dst.pix[i0:i1]
		for i := 0; i < len(s); i += 4 {
			if s[i+3] == 0 {
				continue
			}
			if k == 255 {
				over(t[i:i+4], uint32(s[i]), uint32(s[i+1]), uint32(s[i+2]), uint32(s[i+3]))
				continue
			}
			over(t[i:i+4], div255(uint32(s[i])*k), div255(uint32(s[i+1])*k), div255(uint32(s[i+2])*k), div255(uint32(s[i+3])*k))
		}
	}
	dst.dirty = dst.dirty.Union(d)
}

// scale multiplies the pixels of l within its dirty rectangle by the
// values of mask, one byte per pixel of that rectangle.
func (r *renderer) scale(l *layer, mask []byte) {
	d := l.dirty
	w := d.Dx()
	for y := d.Min.Y; y < d.Max.Y; y++ {
		row := l.pix[(y*r.w+d.Min.X)*4 : (y*r.w+d.Max.X)*4]
		m := mask[(y-d.Min.Y)*w : (y-d.Min.Y+1)*w]
		for x, v := range m {
			p := row[x*4 : x*4+4]
			switch v {
			case 255:
			case 0:
				p[0], p[1], p[2], p[3] = 0, 0, 0, 0
			default:
				k := uint32(v)
				p[0] = uint8(div255(uint32(p[0]) * k))
				p[1] = uint8(div255(uint32(p[1]) * k))
				p[2] = uint8(div255(uint32(p[2]) * k))
				p[3] = uint8(div255(uint32(p[3]) * k))
			}
		}
	}
}

// clip clips what a layer holds to a clip path, applied in the user space
// m of an element with a bounding box.
func (r *renderer) clip(l *layer, cd *clipDef, m matrix, bbox box, hasBox bool) {
	if l.dirty.Empty() {
		return
	}
	r.scale(l, r.clipMask(cd, m, bbox, hasBox, l.dirty))
}

// clipMask returns the coverage of a clip path over a rectangle.
func (r *renderer) clipMask(cd *clipDef, m matrix, bbox box, hasBox bool, rect image.Rectangle) []byte {
	w := rect.Dx()
	mask := make([]byte, w*rect.Dy())
	cm := m.mul(cd.m)
	if cd.bbox {
		if !hasBox || bbox.w == 0 || bbox.h == 0 {
			return mask // nothing to clip to
		}
		cm = cm.mul(boxMatrix(bbox))
	}
	for _, k := range cd.kids {
		r.clipNode(k, cm, rect, mask)
	}
	if cd.clip != nil {
		inner := r.clipMask(cd.clip, m, bbox, hasBox, rect)
		for i, v := range inner {
			mask[i] = uint8(div255(uint32(mask[i]) * uint32(v)))
		}
	}
	return mask
}

// clipNode adds the coverage of a clip path's node to mask, a rectangle.
func (r *renderer) clipNode(n *node, m matrix, rect image.Rectangle, mask []byte) {
	m = m.mul(n.m)
	if n.shape == nil {
		for _, k := range n.kids {
			r.clipNode(k, m, rect, mask)
		}
		return
	}
	r.flat.reset()
	n.shape.path.flatten(m, true, &r.flat)
	cr, ok := r.cover(&r.flat, n.shape.evenOdd)
	if !ok {
		return
	}
	in := cr.Intersect(rect)
	cw, w := cr.Dx(), rect.Dx()
	for y := in.Min.Y; y < in.Max.Y; y++ {
		src := r.cov[(y-cr.Min.Y)*cw+in.Min.X-cr.Min.X:]
		dst := mask[(y-rect.Min.Y)*w+in.Min.X-rect.Min.X:]
		for x := range in.Dx() {
			a, b := uint32(dst[x]), uint32(src[x])
			dst[x] = uint8(a + b - div255(a*b))
		}
	}
}

// mask masks what a layer holds with a mask, applied in the user space m
// of an element with a bounding box.
func (r *renderer) mask(l *layer, md *maskDef, m matrix, bbox box, hasBox bool) {
	d := l.dirty
	if d.Empty() {
		return
	}
	region := md.region
	cm := m
	if md.units || md.content {
		if !hasBox || bbox.w == 0 || bbox.h == 0 {
			r.scale(l, make([]byte, d.Dx()*d.Dy()))
			return
		}
		if md.units {
			region = box{bbox.x + region.x*bbox.w, bbox.y + region.y*bbox.h, region.w * bbox.w, region.h * bbox.h}
		}
		if md.content {
			cm = m.mul(boxMatrix(bbox))
		}
	}
	ml := r.layer()
	for _, k := range md.kids {
		r.node(k, cm, ml, 1)
	}
	values := make([]byte, d.Dx()*d.Dy())
	w := d.Dx()
	for y := d.Min.Y; y < d.Max.Y; y++ {
		row := ml.pix[(y*r.w+d.Min.X)*4 : (y*r.w+d.Max.X)*4]
		for x := range w {
			p := row[x*4 : x*4+4]
			v := uint32(p[3])
			if !md.alpha {
				// Luminance, of premultiplied colors.
				v = (54*uint32(p[0]) + 183*uint32(p[1]) + 19*uint32(p[2]) + 128) >> 8
			}
			values[(y-d.Min.Y)*w+x] = uint8(min(v, 255))
		}
	}
	r.release(ml)
	// The mask's content shows within its region only.
	var rp path
	rp.rect(region.x, region.y, region.w, region.h, 0, 0)
	r.flat.reset()
	rp.flatten(m, true, &r.flat)
	cr, ok := r.cover(&r.flat, false)
	for y := d.Min.Y; y < d.Max.Y; y++ {
		for x := d.Min.X; x < d.Max.X; x++ {
			i := (y-d.Min.Y)*w + x - d.Min.X
			if !ok || !(image.Point{x, y}.In(cr)) {
				values[i] = 0
				continue
			}
			values[i] = uint8(div255(uint32(values[i]) * uint32(r.cov[(y-cr.Min.Y)*cr.Dx()+x-cr.Min.X])))
		}
	}
	r.scale(l, values)
}
