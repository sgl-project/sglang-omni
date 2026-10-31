package gputest

import (
	"math"

	"github.com/egoist/mygo/internal/scene"
)

// CullCases are scenes of overlapping fills and what they hide: text,
// images, shadows and fills under opaque panels, with square, round and
// continuous corners; panels that do not hide what is under them
// (translucent, transformed, at half opacity, gradients, clipped to less,
// rounded clips, an effect reading what is under them); layers as large as
// the frame; and an occluder in the clear color.
func CullCases() []CullCase {
	white := scene.Color{R: 255, G: 255, B: 255, A: 255}
	paper := scene.Color{R: 246, G: 247, B: 249, A: 255}
	red := scene.Color{R: 220, G: 40, B: 40, A: 255}
	blue := scene.Color{R: 37, G: 99, B: 235, A: 255}
	ink := scene.Color{R: 20, G: 24, B: 32, A: 255}
	r4 := func(r float32) [4]float32 { return [4]float32{r, r, r, r} }

	// base returns a scene with a mask atlas holding a disc, an image, and
	// the helpers adding ops.
	type builder struct {
		s    *scene.Scene
		u, v uint16
		img  *scene.Image
	}
	base := func(clear scene.Color) *builder {
		s := &scene.Scene{Width: 400, Height: 300, Clear: clear,
			Text: scene.TextParams{GammaRatios: scene.GammaRatios(1.8), Contrast: 1, SubpixelContrast: 0.5}}
		s.MaskAtlas = scene.NewAtlas(1, 64, 64)
		disc := make([]byte, 16*16)
		for y := range 16 {
			for x := range 16 {
				dx, dy := float64(x)-7.5, float64(y)-7.5
				disc[y*16+x] = byte(255 * clamp(7.5-math.Sqrt(dx*dx+dy*dy)))
			}
		}
		x, y, _ := s.MaskAtlas.Alloc(16, 16)
		s.MaskAtlas.Put(x, y, 16, 16, disc, 16)
		pix := make([]byte, 8*8*4)
		for i := 0; i < len(pix); i += 4 {
			pix[i], pix[i+1], pix[i+2], pix[i+3] = byte(i), byte(255-i), 160, 255
		}
		return &builder{s: s, u: uint16(x), v: uint16(y), img: scene.NewImageRGBA(8, 8, pix)}
	}
	add := func(b *builder, op scene.Op) { b.s.Ops = append(b.s.Ops, op) }
	fill := func(b *builder, r scene.Rect, radius float32, c scene.Color) {
		add(b, scene.Op{Kind: scene.OpFill, Rect: r, Radii: r4(radius), Color: c})
	}
	text := func(b *builder, x, y float32, n int) {
		start := int32(len(b.s.Glyphs))
		for i := range n {
			b.s.Glyphs = append(b.s.Glyphs, scene.Glyph{X: x + float32(i*14), Y: y, W: 16, H: 16, U: b.u, V: b.v, UW: 16, VH: 16, Color: ink})
		}
		add(b, scene.Op{Kind: scene.OpGlyphs, Start: start, End: int32(len(b.s.Glyphs))})
	}
	image := func(b *builder, r scene.Rect) {
		add(b, scene.Op{Kind: scene.OpImage, Rect: r, Image: b.img, Src: scene.Rect{W: 8, H: 8}})
	}
	rect := func(x, y, w, h float32) scene.Rect { return scene.Rect{X: x, Y: y, W: w, H: h} }
	var out []CullCase
	cs := func(name string, b *builder, culled int) { out = append(out, CullCase{name, b.s, culled}) }

	// Stacked panels: text, an image, a shadow and a fill under an opaque
	// panel with square corners, which lies on the page, as large as the
	// frame. 4 ops are hidden: the glyphs, the image, the shadow and the fill.
	b := base(scene.Color{A: 255})
	fill(b, rect(0, 0, 400, 300), 0, paper)
	text(b, 60, 60, 5)
	image(b, rect(100, 100, 40, 40))
	add(b, scene.Op{Kind: scene.OpShadow, Rect: rect(150, 120, 60, 40), Radii: r4(8), Color: scene.Color{A: 90}, Blur: 12})
	fill(b, rect(40.5, 40.25, 200, 150), 0, red)
	fill(b, rect(20, 20, 300, 240), 0, white)
	fill(b, rect(180, 100, 160, 120), 0, blue)
	text(b, 200, 130, 4)
	cs("stacked panels", b, 4)

	// Rounded and continuous corners: a card hides what lies inside its
	// curves, the glyphs; the small fills near its corners stay, and so does
	// the large fill under it.
	for _, cont := range []bool{false, true} {
		b = base(paper)
		fill(b, rect(10, 10, 380, 280), 0, blue)
		fill(b, rect(60, 60, 20, 20), 0, red)
		fill(b, rect(320, 220, 30, 30), 0, red)
		text(b, 140, 130, 3)
		add(b, scene.Op{Kind: scene.OpFill, Rect: rect(50, 50, 300, 200), Radii: r4(40), Continuous: cont, Color: white})
		name := "rounded card"
		if cont {
			name = "continuous card"
		}
		cs(name, b, 1)
	}

	// A translucent panel, one at half opacity, a gradient and a
	// transformed one hide nothing.
	b = base(paper)
	text(b, 60, 60, 4)
	add(b, scene.Op{Kind: scene.OpFill, Rect: rect(20, 20, 200, 100), Color: scene.Color{R: 255, G: 255, B: 255, A: 200}})
	text(b, 60, 140, 4)
	add(b, scene.Op{Kind: scene.OpFill, Rect: rect(20, 120, 200, 100), Color: white, Opacity: 0.5})
	text(b, 240, 60, 4)
	add(b, scene.Op{Kind: scene.OpFill, Rect: rect(230, 20, 150, 100), Color: red, Color2: blue, Paint: scene.PaintLinear, Gradient: [4]float32{230, 0, 380, 0}})
	text(b, 240, 160, 4)
	add(b, scene.Op{Kind: scene.OpFill, Rect: rect(230, 130, 150, 100), Color: white, Transform: scene.Translation(2, 3)})
	cs("panels that hide nothing", b, 0)

	// Transformed content under a panel is hidden all the same, by its
	// bounds, and the panel's own transform is what stops it from hiding.
	b = base(paper)
	start := int32(len(b.s.Glyphs))
	b.s.Glyphs = append(b.s.Glyphs, scene.Glyph{X: 60, Y: 60, W: 16, H: 16, U: b.u, V: b.v, UW: 16, VH: 16, Color: ink})
	add(b, scene.Op{Kind: scene.OpGlyphs, Start: start, End: int32(len(b.s.Glyphs)), Transform: scene.Scaling(1.5, 1.2)})
	add(b, scene.Op{Kind: scene.OpFill, Rect: rect(100, 100, 40, 40), Radii: r4(6), Color: red, Transform: scene.Scaling(1.1, 1).Mul(scene.Translation(3, 2))})
	fill(b, rect(20, 20, 300, 240), 0, white)
	cs("transformed content", b, 2)

	// A wide translucent border is not the fill's color: glyphs under it stay.
	b = base(paper)
	text(b, 25, 25, 4)
	add(b, scene.Op{Kind: scene.OpFill, Rect: rect(20, 20, 200, 100), Color: white, Border: scene.Uniform(30), BorderColor: scene.Color{R: 255, A: 100}})
	cs("wide translucent border", b, 0)

	// A clip: an opaque layer hides what lies in the clip, drawn before or
	// after the clip, and in the layer; the page, which the clip does not
	// reach, and text on it stay.
	b = base(paper)
	fill(b, rect(0, 0, 400, 300), 0, blue)
	text(b, 40, 40, 3)
	add(b, scene.Op{Kind: scene.OpPushClip, Rect: rect(30.5, 30, 200, 100)})
	text(b, 60, 60, 3)
	fill(b, rect(100, 90, 50, 80), 0, red)
	fill(b, rect(0, 0, 400, 300), 0, white)
	add(b, scene.Op{Kind: scene.OpPopClip})
	text(b, 260, 200, 3)
	cs("clipped occluder", b, 3)

	// A clip with a fractional edge antialiases it, and shows what is under.
	b = base(paper)
	text(b, 60, 100, 3)
	add(b, scene.Op{Kind: scene.OpPushClip, Rect: rect(30, 30, 200, 70.5)})
	fill(b, rect(0, 0, 400, 300), 0, white)
	add(b, scene.Op{Kind: scene.OpPopClip})
	cs("fractional clip edge", b, 0)

	b = base(paper)
	add(b, scene.Op{Kind: scene.OpPushClip, Rect: rect(30, 30, 200, 100), Radii: r4(20)})
	text(b, 60, 60, 3)
	fill(b, rect(0, 0, 400, 300), 0, white) // under a rounded clip: antialiased edge, hides nothing
	add(b, scene.Op{Kind: scene.OpPopClip})
	cs("rounded clip", b, 0)

	// A panel and what only partly lies under it.
	b = base(paper)
	text(b, 25, 60, 6) // reaches out of the panel at 20..100? starts at 25, ends ~109
	fill(b, rect(20, 20, 85, 200), 0, white)
	cs("glyphs partly exposed", b, 0)

	// An effect reading what is under it, reaching out of the panel on top.
	b = base(paper)
	fill(b, rect(60, 60, 60, 60), 0, red)
	fx := scene.EffectOp{Effect: LensEffect, Blur: 4}
	fx.Params[0] = [4]float32{2}
	fx.Params[1] = [4]float32{0.5, 0.5, 1, 0.3}
	add(b, scene.Op{Kind: scene.OpEffect, Rect: rect(40, 40, 120, 120), Radii: r4(20), Start: 0})
	b.s.Effects = append(b.s.Effects, fx)
	fill(b, rect(30, 30, 100, 100), 0, white)
	cs("effect between", b, 0)

	// An occluder in the clear color hides what is under it all the same.
	b = base(paper)
	text(b, 60, 60, 4)
	image(b, rect(100, 100, 40, 40))
	fill(b, rect(30, 30, 200, 150), 10, paper)
	cs("occluder in the clear color", b, 2)

	// Layers as large as the frame; hole and shadows under them.
	b = base(scene.Color{A: 255})
	fill(b, rect(0, 0, 400, 300), 0, blue)
	add(b, scene.Op{Kind: scene.OpShadow, Rect: rect(100, 100, 100, 100), Radii: r4(10), Color: scene.Color{A: 120}, Blur: 20})
	add(b, scene.Op{Kind: scene.OpHole, Rect: rect(100, 100, 100, 100)})
	fill(b, rect(-10, -10, 420, 320), 0, white)
	fill(b, rect(0, 0, 400, 300), 0, paper)
	cs("frame sized layers", b, 4)

	// A hole after an opaque fill, and an opaque fill after a hole.
	b = base(paper)
	fill(b, rect(20, 20, 200, 200), 0, blue)
	add(b, scene.Op{Kind: scene.OpHole, Rect: rect(110, 110, 20, 20), Radii: r4(4)})
	fill(b, rect(100, 100, 100, 100), 0, red)
	add(b, scene.Op{Kind: scene.OpHole, Rect: rect(120, 120, 20, 20)})
	fill(b, rect(110, 110, 30, 30), 0, white)
	cs("holes", b, 1)

	// Nested panels, each lying on a larger one drawn later: all but the
	// last are hidden.
	b = base(paper)
	for i := 18; i >= 0; i-- {
		fill(b, rect(float32(i*8), float32(i*8), 400-float32(i*16), 300-float32(i*16)), 0, scene.Color{R: byte(40 + i*10), G: 100, B: 200, A: 255})
	}
	cs("nested panels", b, 18)
	return out
}

// FillScene returns a w×h scene of plain fills, the kind most of a window
// is made of: layers as large as the frame, panels with square and round
// corners at fractional places, rows in a clip that holds some whole and
// cuts others, fills of every opacity, a gradient, a border and a transformed
// fill among them, over the clear color. Sizes at or above 2048 pixels take
// the interpolated coordinates across binades of float32.
func FillScene(w, h int) *scene.Scene {
	s := &scene.Scene{Width: w, Height: h, Clear: scene.Color{R: 12, G: 10, B: 9, A: 255}}
	fw, fh := float32(w), float32(h)
	r4 := func(r float32) [4]float32 { return [4]float32{r, r, r, r} }
	add := func(op scene.Op) { s.Ops = append(s.Ops, op) }
	shade := func(i int) scene.Color {
		return scene.Color{R: byte(20 + i*37%200), G: byte(10 + i*53%220), B: byte(40 + i*71%200), A: 255}
	}
	add(scene.Op{Kind: scene.OpFill, Rect: scene.Rect{W: fw, H: fh}, Color: shade(1)})
	add(scene.Op{Kind: scene.OpFill, Rect: scene.Rect{X: fw / 6, Y: 16.25, W: fw * 2 / 3, H: fh - 40.5}, Radii: r4(24), Color: shade(2)})
	add(scene.Op{Kind: scene.OpFill, Rect: scene.Rect{X: fw/6 + 2, Y: 154, W: fw*2/3 - 4, H: fh - 190}, Color: shade(3)})
	// Rows in a clip: some inside it, one cut at each end.
	cx, cy, cw, ch := fw/6+2, float32(154), fw*2/3-4, fh-190
	add(scene.Op{Kind: scene.OpPushClip, Rect: scene.Rect{X: cx + 10.5, Y: cy + 20.25, W: cw - 21, H: ch - 40}})
	for i := range 40 {
		y := cy + 10 + float32(i)*(ch-20)/40
		add(scene.Op{Kind: scene.OpFill, Rect: scene.Rect{X: cx + 5.5, Y: y, W: cw - 11, H: (ch-20)/40 - 1.5}, Color: shade(i + 4)})
		add(scene.Op{Kind: scene.OpFill, Rect: scene.Rect{X: cx + 30, Y: y + 2, W: cw / 3, H: 6.5}, Radii: r4(3), Color: shade(i + 9), Opacity: 0.5 + float32(i%3)/6})
	}
	add(scene.Op{Kind: scene.OpPopClip})
	// A rounded clip, and an op touching its edge.
	add(scene.Op{Kind: scene.OpPushClip, Rect: scene.Rect{X: cx + 40, Y: cy + 40, W: 300, H: 200}, Radii: r4(20)})
	add(scene.Op{Kind: scene.OpFill, Rect: scene.Rect{X: cx + 30, Y: cy + 100, W: 400, H: 60}, Color: shade(50)})
	add(scene.Op{Kind: scene.OpPopClip})
	add(scene.Op{Kind: scene.OpFill, Rect: scene.Rect{X: cx + 400, Y: cy + 30, W: 200, H: 80}, Color: shade(60), Color2: shade(61), Paint: scene.PaintLinear, Gradient: [4]float32{cx + 400, 0, cx + 600, 0}})
	add(scene.Op{Kind: scene.OpFill, Rect: scene.Rect{X: cx + 400, Y: cy + 130, W: 200, H: 80}, Radii: r4(12), Color: shade(62), Border: scene.Uniform(2), BorderColor: shade(63)})
	add(scene.Op{Kind: scene.OpFill, Rect: scene.Rect{X: cx + 400, Y: cy + 230, W: 200, H: 80}, Color: shade(64), Transform: scene.Scaling(1.01, 1)})
	// The edges of the frame: fills reaching and passing it.
	add(scene.Op{Kind: scene.OpFill, Rect: scene.Rect{X: -5, Y: fh - 30, W: fw + 10, H: 30}, Color: shade(70)})
	add(scene.Op{Kind: scene.OpFill, Rect: scene.Rect{X: fw - 20, Y: 0, W: 20, H: fh}, Color: shade(71), Opacity: 0.4})
	add(scene.Op{Kind: scene.OpFill, Rect: scene.Rect{X: 0, Y: 0, W: fw, H: 1.5}, Color: shade(72)})
	return s
}
