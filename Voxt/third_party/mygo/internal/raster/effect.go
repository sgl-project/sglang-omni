package raster

import (
	"math"
	"runtime"

	"github.com/egoist/mygo/internal/scene"
)

// drawer is what draw keeps from scene to scene: the renderers of the
// bands and the team drawing them, the backdrop of an effect, and the
// effects' CPU twins.
type drawer struct {
	rs    []renderer
	bands team
	job   bandJob
	bd    backdrop
	fx    map[*scene.Effect]scene.EffectPixels
	// blurred holds, by op, the images of the scene drawn that blur theirs
	// (blurImages).
	blurred []*blurredImage
}

// pixels returns what draws e on the CPU, made the first time, or nil for
// an effect without.
func (d *drawer) pixels(e *scene.Effect) scene.EffectPixels {
	if e == nil || e.Pixels == nil {
		return nil
	}
	px := d.fx[e]
	if px == nil {
		if d.fx == nil {
			d.fx = map[*scene.Effect]scene.EffectPixels{}
		}
		px = e.Pixels()
		d.fx[e] = px
	}
	return px
}

// backdrop computes the backdrop of an effect, as the GPU renderers do in
// textures, rounded to 8 bits at each step.
type backdrop struct {
	img scene.BackdropImage
	// tmp holds the rows blurred, before the columns are.
	tmp     []float32
	weights []float32

	// lines does the passes of read on several cores, each line of pass
	// over dst (line).
	lines team
	dst   *Image
	pass  int
}

// The passes of backdrop.read.
const (
	averagePass = iota
	rowsPass
	columnsPass
)

// texelWork is how many texels times taps are worth waking other cores
// for.
const texelWork = 64 << 10

// read computes the backdrop b of dst.
func (bd *backdrop) read(dst *Image, b scene.Backdrop) {
	img := &bd.img
	img.Backdrop = b
	img.W, img.H = b.Size()
	n := 4 * img.W * img.H
	if cap(img.Pix) < n {
		img.Pix, bd.tmp = make([]float32, n), make([]float32, n)
	}
	img.Pix, bd.tmp = img.Pix[:n], bd.tmp[:n]
	if n == 0 {
		return
	}
	bd.dst = dst
	k := b.Down
	bd.run(averagePass, img.H, img.W*k*k)
	if b.Radius == 0 {
		return
	}
	bd.weights = bd.weights[:0]
	for i := 0; i <= b.Radius; i++ {
		bd.weights = append(bd.weights, scene.BlurWeight(i, b.Sigma))
	}
	taps := 2*b.Radius + 1
	bd.run(rowsPass, img.H, img.W*taps)
	bd.run(columnsPass, img.W, img.H*taps)
}

// run does lines 0 to n-1 of pass, each work texels times taps, on
// several cores when they are worth it.
func (bd *backdrop) run(pass, n, work int) {
	bd.pass = pass
	workers := max(min(runtime.GOMAXPROCS(0), n*work/texelWork, maxWorkers), 1)
	if workers == 1 {
		for i := range n {
			bd.line(0, i)
		}
		return
	}
	if bd.lines.do == nil {
		bd.lines.do = bd.line
	}
	bd.lines.run(workers, n)
}

// line does line i of the pass.
func (bd *backdrop) line(_, i int) {
	img := &bd.img
	switch bd.pass {
	case averagePass:
		bd.average(bd.dst, i)
	case rowsPass:
		bd.blur(bd.tmp, img.Pix, 4, 4*img.W, img.W, i)
	default:
		bd.blur(img.Pix, bd.tmp, 4*img.W, 4, img.H, i)
	}
}

// average computes row j of the texels: the average of each square of
// the area, those along its far edges repeating its last pixels, as a
// texture's edges do.
func (bd *backdrop) average(dst *Image, j int) {
	img := &bd.img
	k := img.Down
	inv := 1 / float32(k*k)
	for i := range img.W {
		var c [4]float32
		for dy := range k {
			y := min(img.Area.Min.Y+j*k+dy, img.Area.Max.Y-1)
			row := dst.Pix[y*dst.Stride:]
			for dx := range k {
				x := min(img.Area.Min.X+i*k+dx, img.Area.Max.X-1)
				p := row[4*x : 4*x+4]
				// BGRA to RGBA.
				c[0] += float32(p[2]) / 255
				c[1] += float32(p[1]) / 255
				c[2] += float32(p[0]) / 255
				c[3] += float32(p[3]) / 255
			}
		}
		o := img.Pix[4*(j*img.W+i):][:4]
		for ch := range o {
			o[ch] = texel(c[ch] * inv)
		}
	}
}

// texel returns v as an 8-bit channel of a texture holds it.
func texel(v float32) float32 { return float32(to8(v)) / 255 }

// blur blurs line of src into dst: n texels step apart, from line×next,
// clamping at its ends.
func (bd *backdrop) blur(dst, src []float32, step, next, n, line int) {
	rad := bd.img.Radius
	w := bd.weights
	// Every texel takes every weight, those past the ends on the last
	// texel.
	sum := w[0]
	for o := 1; o <= rad; o++ {
		sum += 2 * w[o]
	}
	base := line * next
	for i := range n {
		p := src[base+i*step:][:4]
		c0, c1, c2, c3 := p[0]*w[0], p[1]*w[0], p[2]*w[0], p[3]*w[0]
		if i >= rad && i+rad < n {
			for o := 1; o <= rad; o++ {
				a, b := src[base+(i-o)*step:][:4], src[base+(i+o)*step:][:4]
				wo := w[o]
				c0 += (a[0] + b[0]) * wo
				c1 += (a[1] + b[1]) * wo
				c2 += (a[2] + b[2]) * wo
				c3 += (a[3] + b[3]) * wo
			}
		} else {
			for o := 1; o <= rad; o++ {
				a, b := src[base+max(i-o, 0)*step:][:4], src[base+min(i+o, n-1)*step:][:4]
				wo := w[o]
				c0 += (a[0] + b[0]) * wo
				c1 += (a[1] + b[1]) * wo
				c2 += (a[2] + b[2]) * wo
				c3 += (a[3] + b[3]) * wo
			}
		}
		d := dst[base+i*step:][:4]
		d[0], d[1], d[2], d[3] = texel(c0/sum), texel(c1/sum), texel(c2/sum), texel(c3/sum)
	}
}

// effect draws an effect with px, which Begin readied for op, over its
// backdrop b (nil for an effect reading none), within its shape, whose
// corners are continuous when the op's are, while the effect works with
// circular ones.
func (r *renderer) effect(op *scene.Op, px scene.EffectPixels, b *scene.BackdropImage) {
	if op.Rect.Empty() {
		return
	}
	opacity := op.Opacity
	if opacity == 0 {
		opacity = 1
	}
	radii := scene.Corners(op.Rect, op.Radii, op.Continuous)
	box := newShape(op.Rect, radii)
	round := hasRadii(radii)
	x0, y0, x1, y1 := r.pixelBounds(op.Rect)
	for y := y0; y < y1; y++ {
		row := r.dst.Pix[y*r.dst.Stride:]
		py := float32(y) + 0.5
		cl, ch := r.clipSolid(y)
		ol, oh := solidSpan(&box, float32(y), float32(y+1))
		for x := x0; x < x1; x++ {
			px0 := float32(x) + 0.5
			cov := opacity
			if x < ol || x >= oh || !round {
				cov *= coverage(&box, px0, py)
			}
			if x < cl || x >= ch {
				cov *= r.clipCoverage(x, y)
			}
			if cov <= 0 {
				continue
			}
			c := px.Color(px0, py, b)
			p := row[4*x : 4*x+4]
			if cov >= 1 && c[3] >= 1 {
				// Opaque.
				p[0], p[1], p[2], p[3] = to8(c[2]), to8(c[1]), to8(c[0]), 255
				continue
			}
			blend(p, c, cov)
		}
	}
}

// maskedEffect draws a shader's effect filling mask glyphs (fx.Glyphs) with
// px, as glyphs draws mask glyphs: each pixel the shader's color times the
// glyph's coverage, corrected for that color.
func (r *renderer) maskedEffect(op *scene.Op, fx *scene.EffectOp, px scene.EffectPixels) {
	atlas := r.s.MaskAtlas
	if atlas == nil {
		return
	}
	opacity := opacityOf(op.Opacity)
	text := r.s.Text
	for _, g := range r.s.Glyphs[fx.Glyphs[0]:fx.Glyphs[1]] {
		if g.Colored || g.Subpixel {
			continue
		}
		boost := float32(0)
		if g.Thin {
			boost = scene.ThinBoost
		}
		box := scene.Rect{X: float32(math.Round(float64(g.X))), Y: float32(math.Round(float64(g.Y))), W: float32(g.UW), H: float32(g.VH)}
		m := op.Transform.Mul(g.Transform)
		if !m.Set {
			gx, gy := int(box.X), int(box.Y)
			x0, y0 := max(gx, r.x0), max(gy, r.y0)
			x1, y1 := min(gx+int(g.UW), r.x1), min(gy+int(g.VH), r.y1)
			for y := y0; y < y1; y++ {
				row := r.dst.Pix[y*r.dst.Stride:]
				cl, ch := r.clipSolid(y)
				ay := int(g.V) + y - gy
				for x := x0; x < x1; x++ {
					mk := atlas.Pix[ay*atlas.W+int(g.U)+x-gx]
					if mk == 0 {
						continue
					}
					cov := opacity
					if x < cl || x >= ch {
						if cov *= r.clipCoverage(x, y); cov == 0 {
							continue
						}
					}
					c := px.Color(float32(x)+0.5, float32(y)+0.5, nil)
					a := scene.TextCoverage(float32(mk)/255, unpremul(c), text.Contrast, boost, text.GammaRatios)
					blend(row[4*x:4*x+4], c, cov*a)
				}
			}
			continue
		}
		inv, ok := m.Inverse()
		if !ok {
			continue
		}
		shape := newShape(box, [4]float32{})
		x0, y0, x1, y1 := r.pixelBounds(m.Bounds(box))
		for y := y0; y < y1; y++ {
			for x := x0; x < x1; x++ {
				wx, wy := float32(x)+0.5, float32(y)+0.5
				cov := r.clipCoverage(x, y) * transformedCoverage(&shape, m, inv, wx, wy) * opacity
				if cov <= 0 {
					continue
				}
				lx, ly := inv.Point(wx, wy)
				s := atlasSample(atlas, float32(g.U)+lx-box.X, float32(g.V)+ly-box.Y, int(g.U), int(g.V), int(g.UW), int(g.VH))
				c := px.Color(lx, ly, nil)
				a := scene.TextCoverage(s[0], unpremul(c), text.Contrast, boost, text.GammaRatios)
				blend(r.dst.Pix[y*r.dst.Stride+4*x:][:4], c, cov*a)
			}
		}
	}
}
