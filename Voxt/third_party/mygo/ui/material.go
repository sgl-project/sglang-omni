package ui

import (
	"math"
	"slices"

	"github.com/egoist/mygo/internal/scene"
)

// Material is what Element.Material fills an element with in place of a
// background: a look a package paints, as the glass plugin's Liquid Glass
// (github.com/egoist/mygo/plugins/glass).
type Material interface {
	// PaintMaterial paints the material over box, rounded by radii, in
	// DIPs, under the element's border and children.
	PaintMaterial(p *Painter, box Rect, radii [4]float32)
}

// MaterialBuilder is a Material that follows the element it fills, as
// its state: Element.Material calls BuildMaterial as the element is
// built, and paints the material it returns.
type nodeMaterialBuilder interface {
	Material
	BuildMaterial(e *node) Material
}

// Material fills the element with m in place of a background, shaped by
// its Radius:
//
//	ui.Row(c).Padding(8, 16).Radius(22).Material(glass.Glass{})
func (e *node) Material(m Material) *node {
	if b, ok := m.(nodeMaterialBuilder); ok {
		m = b.BuildMaterial(e)
	}
	e.material, e.fill = m, fillMaterial
	return e
}

// Theme returns the theme of the view being painted.
func (p *Painter) Theme() *Theme { return p.rt.c.theme }

// EffectColor returns c as a parameter of an effect: straight RGBA from 0
// to 1, its sRGB value, and wide, its color outside the sRGB gamut in
// extended sRGB, which the effect's Metal shader takes where e.wide tells
// that the target keeps it (wide is the sRGB value for a color inside the
// gamut). Like a color an element paints, it has the frame drawn in a wide
// gamut where the window shows one.
func (p *Painter) EffectColor(c Color) (srgb, wide [4]float32) {
	srgb = [4]float32{float32(c.R) / 255, float32(c.G) / 255, float32(c.B) / 255, float32(c.A) / 255}
	wide = srgb
	if a, ok := c.wideRGBA(); ok {
		wide = a
		p.addWide(scene.WideColors{Color: a, Set: scene.WideColor})
	}
	return srgb, wide
}

// Effect paints the rectangle r rounded by radii, in DIPs, with an effect:
// shaders the renderers draw it with (see scene.Effect), which only this
// module's packages define, as the official plugins do. params are the
// effect's own, in device pixels (Scale), and blur the standard deviation
// of the blur of its backdrop, in device pixels, for an effect reading one.
func (p *Painter) Effect(fx *scene.Effect, r Rect, radii [4]float32, blur float32, params [5][4]float32) {
	if fx == nil || r.W <= 0 || r.H <= 0 {
		return
	}
	p.add(scene.Op{Kind: scene.OpEffect, Rect: p.snap(r), Radii: p.radii(radii), Continuous: continuousCorners,
		Start: int32(len(p.s.Effects)), Opacity: p.opacity})
	p.s.Effects = append(p.s.Effects, scene.EffectOp{Effect: fx, Blur: blur, Params: params})
}

// ShaderInput is what the effect of a shader of package ui/shader reads:
// the words of its uniforms' lanes, the words a wide gamut reads in their
// place (nil when the same), from EffectWords, and up to four images,
// bitmaps or canvases' (nil for none).
type ShaderInput struct {
	Uniforms, WideUniforms []uint32
	Images                 [4]*Bitmap
	// Blur, for a shader reading its backdrop, is the standard deviation
	// of the backdrop's blur, in pixels.
	Blur float32
}

// EffectBackdrop is where the backdrop of an effect lies (Painter.
// EffectBackdrop), as the renderers capture it: ToFrame maps a point in
// DIPs from a corner to the frame's pixels (x' = ToFrame[0]·(x, y, 1), y'
// = ToFrame[1]·(x, y, 1)); the texels start at Origin, in the frame's
// pixels, each Down pixels across, W×H of them.
type EffectBackdrop struct {
	ToFrame [2][3]float32
	Origin  [2]float32
	Down    float32
	W, H    int
}

// EffectBackdrop returns where the backdrop of an effect over r, in DIPs,
// blurred by blur pixels, lies, its points from inset, in pixels from the
// corner of r on the pixel grid: what the renderers read for it, the
// same for every renderer.
func (p *Painter) EffectBackdrop(r Rect, blur float32, inset Point) EffectBackdrop {
	rect := p.snap(r)
	m := p.transform.Pixels(p.scale)
	bk := scene.BackdropOf(m.Bounds(rect), blur, p.s.Width, p.s.Height)
	w, h := bk.Size()
	s := p.scale
	// Corner, then the op's local pixels, then the frame's.
	cx, cy := rect.X+inset.X, rect.Y+inset.Y
	a, b, c, d, x, y := float32(1), float32(0), float32(0), float32(1), float32(0), float32(0)
	if m.Set {
		a, b, c, d, x, y = m.A, m.B, m.C, m.D, m.X, m.Y
	}
	return EffectBackdrop{
		ToFrame: [2][3]float32{{a * s, c * s, a*cx + c*cy + x}, {b * s, d * s, b*cx + d*cy + y}},
		Origin:  [2]float32{float32(bk.Area.Min.X), float32(bk.Area.Min.Y)},
		Down:    float32(bk.Down),
		W:       w, H: h,
	}
}

// op returns the effect op of fx reading in.
func (in *ShaderInput) op(fx *scene.Effect) scene.EffectOp {
	op := scene.EffectOp{Effect: fx, Uniforms: in.Uniforms, WideUniforms: in.WideUniforms, Blur: in.Blur}
	for i, b := range in.Images {
		if b != nil {
			op.Images[i] = b.img
		}
	}
	return op
}

// EffectWords returns n words, zero, for the uniforms of an effect this
// frame paints (ShaderInput), which last until the frame is drawn.
func (p *Painter) EffectWords(n int) []uint32 {
	at := len(p.s.Words)
	p.s.Words = slices.Grow(p.s.Words, n)[:at+n]
	w := p.s.Words[at : at+n : at+n]
	clear(w)
	return w
}

// ShaderEffect paints the rectangle r rounded by radius, in DIPs, with fx,
// the effect of a shader of package ui/shader (scene.Effect.Shader),
// reading in. Only this module's packages make such effects.
func (p *Painter) ShaderEffect(fx *scene.Effect, r Rect, radius float32, in ShaderInput) {
	if fx == nil || r.W <= 0 || r.H <= 0 {
		return
	}
	p.add(scene.Op{Kind: scene.OpEffect, Rect: p.snap(r), Radii: p.radii([4]float32{radius, radius, radius, radius}), Continuous: continuousCorners,
		Start: int32(len(p.s.Effects)), Opacity: p.opacity})
	p.s.Effects = append(p.s.Effects, in.op(fx))
}

// ShaderFill paints the text, paths and icons that fn paints with fx, the
// effect of a shader of package ui/shader, in place of their colors: each
// pixel the shader's color times the shape's coverage. box, in DIPs, is
// the box the shader fills, whose top left corner its positions start
// from. Glyphs of color, as emoji, keep their colors, and what fn paints
// besides, as rectangles and images, paints as it would. fx reads in.
// Only this module's packages make such effects.
func (p *Painter) ShaderFill(fx *scene.Effect, box Rect, in ShaderInput, fn func()) {
	if fx == nil || fn == nil {
		return
	}
	// Grayscale masks, which a shader's color can fill: no subpixel ones.
	savedOpaque := p.opaque
	p.opaque = false
	start := len(p.s.Ops)
	fn()
	p.opaque = savedOpaque
	origin := p.snap(box)
	at := p.transform.Pixels(p.scale)
	// The ops fn added, each of glyphs becoming the effect's, followed by
	// one of its glyphs of color, if any.
	tail := append([]scene.Op(nil), p.s.Ops[start:]...)
	p.s.Ops = p.s.Ops[:start]
	for _, op := range tail {
		if op.Kind != scene.OpGlyphs || op.End <= op.Start {
			p.s.Ops = append(p.s.Ops, op)
			continue
		}
		masks := int32(len(p.s.Glyphs))
		var bounds scene.Rect
		for _, g := range p.s.Glyphs[op.Start:op.End] {
			if g.Colored {
				continue
			}
			p.s.Glyphs = append(p.s.Glyphs, g)
			gb := scene.Rect{X: float32(math.Floor(float64(g.X))), Y: float32(math.Floor(float64(g.Y))), W: g.W + 1, H: g.H + 1}
			if masks == int32(len(p.s.Glyphs))-1 {
				bounds = gb
			} else {
				x0, y0 := min(bounds.X, gb.X), min(bounds.Y, gb.Y)
				x1, y1 := max(bounds.X+bounds.W, gb.X+gb.W), max(bounds.Y+bounds.H, gb.Y+gb.H)
				bounds = scene.Rect{X: x0, Y: y0, W: x1 - x0, H: y1 - y0}
			}
		}
		end := int32(len(p.s.Glyphs))
		if end == masks {
			p.s.Ops = append(p.s.Ops, op) // color only
			continue
		}
		// The box's corner, where the op's geometry lies.
		ox, oy := origin.X, origin.Y
		if op.Transform != at {
			if inv, ok := op.Transform.Inverse(); ok {
				ox, oy = inv.Point(at.Point(ox, oy))
			}
		}
		p.s.Ops = append(p.s.Ops, scene.Op{Kind: scene.OpEffect, Rect: bounds, Transform: op.Transform,
			Start: int32(len(p.s.Effects)), Opacity: p.opacity})
		fop := in.op(fx)
		fop.Glyphs, fop.Origin = [2]int32{masks, end}, [2]float32{ox, oy}
		p.s.Effects = append(p.s.Effects, fop)
		if end-masks < op.End-op.Start {
			c := op
			c.Start = int32(len(p.s.Glyphs))
			for _, g := range p.s.Glyphs[op.Start:op.End] {
				if g.Colored {
					p.s.Glyphs = append(p.s.Glyphs, g)
				}
			}
			c.End = int32(len(p.s.Glyphs))
			p.s.Ops = append(p.s.Ops, c)
		}
	}
}
