package raster

import (
	"github.com/egoist/mygo/internal/scene"
	"math"
)

// Contains tests the interior of a rounded clip, using the same continuous
// corner geometry as rendering. Hit testing uses the shape's edge, without
// the pixel coverage used to smooth that edge on screen.
func Contains(r scene.Rect, radii [4]float32, continuous bool, x, y float32) bool {
	if !r.Contains(x, y) {
		return false
	}
	s := newShape(r, scene.Corners(r, radii, continuous))
	if s.continuous {
		d, _ := s.contDist(x, y)
		return d <= 0
	}
	return scene.SDRoundRect(r, s.radii, x, y) <= 0
}

// transformedCoverage integrates a shape that m transforms (inv its
// inverse) over a device pixel. Interior pixels take one distance test;
// edges use a grid in device space. What a projective map puts behind the
// viewer covers nothing.
func transformedCoverage(s *shape, m, inv scene.Affine, x, y float32) float32 {
	if !inv.Set {
		return coverage(s, x, y)
	}
	lx, ly := inv.Point(x, y)
	if m.Perspective() && m.W(lx, ly) <= 0 {
		return 0
	}
	d := scene.SDRoundRect(s.r, s.radii, lx, ly)
	if s.continuous {
		d, _ = s.contDist(lx, ly)
	}
	rx, ry := inv.VectorAt(x, y, 0.5, 0.5)
	sx, sy := inv.VectorAt(x, y, 0.5, -0.5)
	reach := max(abs(rx)+abs(ry), abs(sx)+abs(sy)) * 2
	if d < -reach {
		return 1
	}
	if d > reach {
		return 0
	}
	const samples = 4
	n := 0
	for j := range samples {
		for i := range samples {
			px, py := inv.Point(x+(float32(i)+0.5)/samples-0.5, y+(float32(j)+0.5)/samples-0.5)
			v := scene.SDRoundRect(s.r, s.radii, px, py)
			if s.continuous {
				v, _ = s.contDist(px, py)
			}
			if v <= 0 {
				n++
			}
		}
	}
	return float32(n) / (samples * samples)
}

// transformed inverse maps pixels into the operation's coordinates.
// Identity operations retain the existing span renderer.
func (r *renderer) transformed(op *scene.Op, px scene.EffectPixels, b *scene.BackdropImage) {
	inv, ok := op.Transform.Inverse()
	inset := op.Kind == scene.OpShadow && op.Inset
	if !ok || !inset && op.Rect.Empty() || inset && op.Cast.Empty() {
		return
	}
	radii := scene.Corners(op.Rect, op.Radii, op.Continuous)
	outer := newShape(op.Rect, radii)
	inner := outer
	bw := op.Border
	if op.BorderColor.A == 0 {
		bw = [4]float32{}
	}
	hasBorder := scene.HasBorder(bw)
	if hasBorder {
		inner = newShape(scene.InnerRadii(op.Rect, radii, bw))
	}
	pt, bp := borderPainter(op, opacityOf(op.Opacity), r.s.Stops)
	border := op.BorderColor.Premul(opacityOf(op.Opacity))
	box := op.Rect
	if inset {
		box = op.Cast
	} else if op.Kind == scene.OpShadow {
		e := 1.5*op.Blur + 1
		box = scene.Rect{X: box.X - e, Y: box.Y - e, W: box.W + 2*e, H: box.H + 2*e}
	}
	box = op.Transform.Bounds(box)
	x0, y0, x1, y1 := r.pixelBounds(box)
	for y := y0; y < y1; y++ {
		for x := x0; x < x1; x++ {
			clip := r.clipCoverage(x, y)
			if clip <= 0 {
				continue
			}
			wx, wy := float32(x)+0.5, float32(y)+0.5
			lx, ly := inv.Point(wx, wy)
			if op.Transform.Perspective() && op.Transform.W(lx, ly) <= 0 {
				continue // behind the viewer
			}
			p := r.dst.Pix[y*r.dst.Stride+4*x:][:4]
			if inset {
				// Inside the box, what the hole, blurred, leaves.
				caster := newShape(op.Cast, scene.Corners(op.Cast, op.CastRadii, op.Continuous))
				cov := transformedCoverage(&caster, op.Transform, inv, wx, wy)
				if !op.Rect.Empty() {
					hole := transformedShadowAt(op, &outer, lx, ly)
					if op.Blur < 1 {
						hole = transformedCoverage(&outer, op.Transform, inv, wx, wy)
					}
					cov *= 1 - hole
				}
				blend(p, op.Color.Premul(opacityOf(op.Opacity)), cov*clip)
				continue
			}
			if op.Kind == scene.OpShadow {
				cov := transformedShadowAt(op, &outer, lx, ly)
				if op.Blur < 1 {
					cov = transformedCoverage(&outer, op.Transform, inv, wx, wy)
				}
				if !op.Cast.Empty() {
					caster := newShape(op.Cast, scene.Corners(op.Cast, op.CastRadii, op.Continuous))
					cov *= 1 - transformedCoverage(&caster, op.Transform, inv, wx, wy)
				}
				blend(p, op.Color.Premul(opacityOf(op.Opacity)), cov*clip)
				continue
			}
			cov := transformedCoverage(&outer, op.Transform, inv, wx, wy)
			if cov <= 0 {
				continue
			}
			switch op.Kind {
			case scene.OpFill:
				if pt.visible() {
					blend(p, pt.at(lx, ly), cov*clip)
				}
				if hasBorder {
					ic := float32(0)
					if !inner.r.Empty() {
						ic = transformedCoverage(&inner, op.Transform, inv, wx, wy)
					}
					bc := max(cov-ic, 0)
					if op.Dashed {
						bc *= dash(op.Rect, bw, lx, ly)
					}
					blend(p, borderAt(bp, border, op.BorderColor, lx, ly), bc*clip)
				}
			case scene.OpImage:
				if op.Image == nil || op.Image.W == 0 || op.Image.H == 0 || unflipped(op.Src).Empty() {
					continue
				}
				// How far a pixel of the frame reaches in the image's,
				// as the GPU's fwidth.
				sx, sy := op.Src.W/op.Rect.W, op.Src.H/op.Rect.H
				ax, ay := inv.VectorAt(wx, wy, 1, 0)
				bx, by := inv.VectorAt(wx, wy, 0, 1)
				c := sampleOp(op, op.Src.X+(lx-op.Rect.X)*sx, op.Src.Y+(ly-op.Rect.Y)*sy,
					(abs32(ax)+abs32(bx))*abs32(sx), (abs32(ay)+abs32(by))*abs32(sy))
				blend(p, c, cov*clip*opacityOf(op.Opacity))
			case scene.OpEffect:
				if px != nil {
					blend(p, px.Color(lx, ly, b), cov*clip*opacityOf(op.Opacity))
				}
			case scene.OpHole:
				erase(p, cov*clip*opacityOf(op.Opacity))
			}
		}
	}
}

func opacityOf(v float32) float32 {
	if v == 0 {
		return 1
	}
	return v
}

func transformedShadowAt(op *scene.Op, box *shape, x, y float32) float32 {
	sigma := op.Blur / 2
	if sigma < 0.5 {
		return coverage(box, x, y)
	}
	hx, hy := op.Rect.W/2, op.Rect.H/2
	x, y = x-op.Rect.X-hx, y-op.Rect.Y-hy
	ext := 3 * sigma
	start, end := min(max(-ext, y-hy), y+hy), min(max(ext, y-hy), y+hy)
	step := (end - start) / 4
	k := float32(math.Sqrt(0.5)) / sigma
	corner := max(box.radii[0], box.radii[1], box.radii[2], box.radii[3])
	cc := newContCorner(corner, corner, corner, op.Rect.W, op.Rect.H)
	var v float32
	for i := range 4 {
		yy := start + (float32(i)+0.5)*step
		half := hx
		if box.continuous {
			at, _ := cc.inset(hy - abs(y-yy))
			half -= at
		} else if delta := min(hy-corner-abs(y-yy), 0); delta < 0 {
			half = hx - corner + float32(math.Sqrt(float64(max(0, corner*corner-delta*delta))))
		}
		v += gaussian(yy, sigma) * step * 0.5 * (erf((x+half)*k) - erf((x-half)*k))
	}
	return v
}

func (r *renderer) transformedGlyphs(op *scene.Op) {
	var pt *painter
	if op.Paint == scene.PaintLinear || op.Paint == scene.PaintOklab || op.Paint.HasStops() {
		p := newPainter(op, opacityOf(op.Opacity), r.s.Stops)
		pt = &p
	}
	for _, g := range r.s.Glyphs[op.Start:op.End] {
		m := op.Transform.Mul(g.Transform)
		inv, ok := m.Inverse()
		if !ok {
			continue
		}
		atlas := r.s.MaskAtlas
		if g.Colored || g.Subpixel {
			atlas = r.s.ColorAtlas
		}
		if atlas == nil {
			continue
		}
		box := scene.Rect{X: float32(math.Round(float64(g.X))), Y: float32(math.Round(float64(g.Y))), W: float32(g.UW), H: float32(g.VH)}
		shape := newShape(box, [4]float32{})
		x0, y0, x1, y1 := r.pixelBounds(m.Bounds(box))
		for y := y0; y < y1; y++ {
			for x := x0; x < x1; x++ {
				wx, wy := float32(x)+0.5, float32(y)+0.5
				cov := r.clipCoverage(x, y) * transformedCoverage(&shape, m, inv, wx, wy)
				if cov <= 0 {
					continue
				}
				lx, ly := inv.Point(wx, wy)
				a := atlasSample(atlas, float32(g.U)+lx-box.X, float32(g.V)+ly-box.Y, int(g.U), int(g.V), int(g.UW), int(g.VH))
				tint := g.Color.Premul(1)
				if pt != nil {
					tint = pt.at(lx, ly)
				}
				p := r.dst.Pix[y*r.dst.Stride+4*x:][:4]
				if g.Colored {
					for i := range a {
						a[i] *= float32(g.Color.A) / 255
					}
					blend(p, a, cov)
					continue
				}
				contrast, boost := r.s.Text.Contrast, float32(0)
				if g.Thin {
					boost = scene.ThinBoost
				}
				if g.Subpixel {
					a[0] = (a[0] + a[1] + a[2]) / 3
					contrast = r.s.Text.SubpixelContrast
				}
				coverage := scene.TextCoverage(a[0], unpremul(tint), contrast, boost, r.s.Text.GammaRatios)
				blend(p, tint, cov*coverage)
			}
		}
	}
}

func atlasSample(a *scene.Atlas, u, v float32, ax, ay, w, h int) [4]float32 {
	u, v = u-0.5, v-0.5
	x, y := int(math.Floor(float64(u))), int(math.Floor(float64(v)))
	tx, ty := u-float32(x), v-float32(y)
	var c [4]float32
	for j := range 2 {
		for i := range 2 {
			sx, sy := min(max(x+i, ax), ax+w-1), min(max(y+j, ay), ay+h-1)
			weight := (1 - tx) * (1 - ty)
			if i == 1 {
				weight = tx * (1 - ty)
			}
			if j == 1 {
				weight = (1 - tx) * ty
				if i == 1 {
					weight = tx * ty
				}
			}
			for k := range a.BPP {
				c[k] += float32(a.Pix[(sy*a.W+sx)*a.BPP+k]) / 255 * weight
			}
		}
	}
	return c
}
