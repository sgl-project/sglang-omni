package scene

import (
	"image"
	"math"
)

// Effect is a kind of drawing that a package outside the renderers
// defines, as the official plugins do (plugins/glass): a fragment shader
// for each GPU renderer, drawn with a pipeline of its own, and its twin
// for the CPU renderer, which must compute the same colors. An effect
// paints the shape of its OpEffect (Rect rounded by Radii) from its
// parameters, and from its backdrop when it reads one: what the
// operations before it painted around it, blurred (see Backdrop).
//
// Each shader is the source of a function that the renderer puts between
// a head and a tail of its own, which declare what it reads and call it
// (EffectSource in packages metal, d3d11 and gl):
//
//	Metal: float4 effect(float2 p, Effect e, texture2d<float> backdrop)
//	HLSL:  float4 effect(float2 p, Effect e)
//	GLSL:  vec4 effect(vec2 p, Effect e)
//
// It returns the premultiplied color at the pixel center p, in device
// pixels, of the shape e.rect rounded by e.radii, from the parameters
// e.p0 to e.p4 (EffectOp.Params); the renderer multiplies it by how much
// of the pixel the shape and the clip cover, and by the opacity. The
// shape's corners may be continuous (Op.Continuous), which the renderer
// covers as such, while e.radii, positive, has the effect take them as
// circular. In Metal, e.wide tells that the target keeps colors outside
// the sRGB gamut, in extended sRGB (Scene.Wide), which the effect may then
// take from a parameter holding such a color (ui.Painter.EffectColor).
// sampleBackdrop(e, q) (with the backdrop texture first in Metal) returns
// the backdrop at q, premultiplied, as BackdropImage.Sample does.
type Effect struct {
	// Name names the effect in errors.
	Name string
	// Backdrop tells that the effect reads its backdrop.
	Backdrop bool
	// Metal and HLSL are the effect's shader in Metal Shading Language
	// and HLSL, with the code compiled ahead of time from it; GLSL is its
	// shader in GLSL 3.30 and GLSL ES 3.00, which drivers compile.
	Metal, HLSL EffectCode
	GLSL        string
	// Pixels makes what draws the effect on the CPU, one for each
	// renderer.
	Pixels func() EffectPixels
	// Shader is set for an effect that a shader of package ui/shader
	// draws: its code is the shader's (the Fragment of package shaderir's
	// backends, and a function calling it), which reads the words of
	// EffectOp.Uniforms in a buffer laid out as Shader says, and the images
	// of EffectOp.Images. Metal's is a whole library, with a tail of its
	// own (ShaderSource in package metal); the others' go between the
	// effects' head and tail.
	Shader *ShaderLayout
}

// ShaderLayout is where a shader's uniforms lie in the buffer its code
// reads: the offset in bytes of each word of EffectOp.Uniforms, and the
// buffer's size in bytes, a multiple of 16.
type ShaderLayout struct {
	Lanes []int32
	Size  int
}

// Pack writes words into buf, grown to the layout's size, as the layout
// lays them out, and returns it. A lane takes 4 bytes, but for a bool of
// Metal, which takes 1 and may lie in the last bytes: the bytes that fit
// are written, little end first, and lanes in ascending order write over
// what the one before spilled.
func (l *ShaderLayout) Pack(buf []byte, words []uint32) []byte {
	if cap(buf) < l.Size {
		buf = make([]byte, l.Size)
	}
	buf = buf[:l.Size]
	clear(buf)
	for i, at := range l.Lanes {
		if i >= len(words) {
			break
		}
		w := words[i]
		for k := 0; k < 4 && int(at)+k < len(buf); k++ {
			buf[int(at)+k] = byte(w >> (8 * k))
		}
	}
	return buf
}

// EffectCode is an effect's shader in a language: its source, and the
// code compiled ahead of time from what the renderer makes of it
// (EffectSource), with the SHA-256 of that (gpu.SourceSum). Renderers
// compile the source when the sum no longer matches, as when the
// renderer's head or tail changed since.
type EffectCode struct {
	Source   string
	Compiled []byte
	Sum      string
}

// EffectOp is the effect of an OpEffect, with its parameters.
type EffectOp struct {
	Effect *Effect
	// Blur is the standard deviation of the blur of the backdrop, in
	// pixels, for an effect that reads its backdrop.
	Blur float32
	// Params are the effect's own, which its shaders read as e.p0 to e.p4.
	Params [5][4]float32
	// Uniforms and Images are what a shader's effect reads
	// (Effect.Shader): the words of its uniforms' lanes, and up to four
	// images, bitmaps or canvases, nil for none.
	Uniforms []uint32
	Images   [4]*Image
	// WideUniforms, unless nil, are the words of Uniforms that renderers
	// drawing a wide gamut (Scene.Wide) read in its place: colors outside
	// the sRGB gamut in extended sRGB, where Uniforms holds the nearest
	// sRGB ones.
	WideUniforms []uint32
	// Glyphs, unless empty, are the mask glyphs Scene.Glyphs[Glyphs[0]:
	// Glyphs[1]] that a shader's effect paints instead of the rounded
	// rectangle: the shapes of text, paths and icons, which it fills, each
	// pixel's color times the glyph's coverage, corrected as text's is
	// (TextCoverage) for the color the shader returns. Rect then holds
	// them, and Origin is the point its shader's positions start from, the
	// top left corner of the box it fills.
	Glyphs [2]int32
	Origin [2]float32
}

// Masked reports whether op paints mask glyphs (Glyphs).
func (op *EffectOp) Masked() bool { return op.Glyphs[1] > op.Glyphs[0] }

// Same reports whether op draws what o does: the same effect with the same
// parameters, the same uniforms and no images, whose pixels may change.
// Glyphs are compared by what they point at, which the caller does.
func (op *EffectOp) Same(o *EffectOp) bool {
	if op.Effect != o.Effect || op.Blur != o.Blur || op.Params != o.Params || len(op.Uniforms) != len(o.Uniforms) ||
		op.Masked() != o.Masked() || op.Glyphs[1]-op.Glyphs[0] != o.Glyphs[1]-o.Glyphs[0] || op.Origin != o.Origin {
		return false
	}
	for i := range op.Images {
		if op.Images[i] != nil || o.Images[i] != nil {
			return false
		}
	}
	for i := range op.Uniforms {
		if op.Uniforms[i] != o.Uniforms[i] {
			return false
		}
	}
	if len(op.WideUniforms) != len(o.WideUniforms) {
		return false
	}
	for i := range op.WideUniforms {
		if op.WideUniforms[i] != o.WideUniforms[i] {
			return false
		}
	}
	return true
}

// EffectPixels draws an effect on the CPU, for one renderer.
type EffectPixels interface {
	// Begin readies drawing op, the shape rect rounded by radii (fitted),
	// before its pixels, which several cores may then ask for at once.
	Begin(op *EffectOp, rect Rect, radii [4]float32)
	// Color returns the premultiplied color of the effect at the pixel
	// center (x, y), as its shaders' effect does, over the backdrop b (nil
	// for an effect reading none).
	Color(x, y float32, b *BackdropImage) [4]float32
}

// Backdrop is what an effect reads of what is behind it: the pixels of
// Area, averaged over squares of Down pixels a side into a smaller image
// (Size), which a Gaussian of standard deviation Sigma (BlurWeight) blurs,
// out to Radius texels each way, first along rows, then along columns.
// Renderers keep each step in 8 bits a channel, rounded to the nearest,
// as textures hold them.
type Backdrop struct {
	Area   image.Rectangle
	Down   int
	Sigma  float32
	Radius int
}

// BackdropOf returns the backdrop of a shape r blurred by blur, in a frame
// of w×h pixels: what the blur reaches around it.
//
// Large blurs are computed at a fraction of the size, as browsers do,
// a power of two up to 8 that leaves the blur at least 2 texels: the
// average of each square adds a variance of Down²/12 pixels², which the
// blur leaves out.
func BackdropOf(r Rect, blur float32, w, h int) Backdrop {
	blur = max(blur, 0)
	down := 1
	for down < 8 && blur/float32(2*down) >= 2 {
		down *= 2
	}
	sigma := float32(math.Sqrt(math.Max(float64(blur*blur)-float64(down*down)/12, 0))) / float32(down)
	if blur == 0 {
		sigma = 0
	}
	radius := int(math.Ceil(float64(3 * sigma)))
	reach := float32(radius*down + down)
	area := image.Rect(int(math.Floor(float64(r.X-reach))), int(math.Floor(float64(r.Y-reach))),
		int(math.Ceil(float64(r.X+r.W+reach))), int(math.Ceil(float64(r.Y+r.H+reach))))
	return Backdrop{Area: area.Intersect(image.Rect(0, 0, w, h)), Down: down, Sigma: sigma, Radius: radius}
}

// ImageBlur returns how an image's area is blurred by blur, the standard
// deviation in its pixels (Op.Blur): averaged over squares of Down pixels
// a side and blurred as a backdrop is (BackdropOf), within the area, which
// it reads as transparent beyond.
func ImageBlur(area image.Rectangle, blur float32) Backdrop {
	b := BackdropOf(Rect{}, blur, 0, 0)
	b.Area = area
	return b
}

// ImageSrc returns where src, a rectangle of an image's pixels within the
// area b blurs, lies in the blurred texels.
func (b Backdrop) ImageSrc(src Rect) Rect {
	k := float32(b.Down)
	return Rect{X: (src.X - float32(b.Area.Min.X)) / k, Y: (src.Y - float32(b.Area.Min.Y)) / k, W: src.W / k, H: src.H / k}
}

// ImageArea returns the pixels of an image of w×h that src, a rectangle
// of them, touches, which a blur of it reads.
func ImageArea(src Rect, w, h int) image.Rectangle {
	if src.W < 0 {
		src.X, src.W = src.X+src.W, -src.W
	}
	if src.H < 0 {
		src.Y, src.H = src.Y+src.H, -src.H
	}
	return image.Rect(int(math.Floor(float64(src.X))), int(math.Floor(float64(src.Y))),
		int(math.Ceil(float64(src.X+src.W))), int(math.Ceil(float64(src.Y+src.H)))).Intersect(image.Rect(0, 0, w, h))
}

// Size returns the size of the averaged image, in texels.
func (b Backdrop) Size() (w, h int) {
	return (b.Area.Dx() + b.Down - 1) / b.Down, (b.Area.Dy() + b.Down - 1) / b.Down
}

// BlurWeight returns the weight of the texel i texels away in a blur of
// standard deviation sigma, before the weights are divided by their sum.
func BlurWeight(i int, sigma float32) float32 {
	if sigma <= 0 {
		return 1
	}
	x := float32(i)
	return float32(math.Exp(float64(-(x * x) / (2 * sigma * sigma))))
}

// BackdropImage is a backdrop as the CPU renderer computes it: W×H texels
// of premultiplied RGBA, each channel a byte over 255, as a texture
// reads.
type BackdropImage struct {
	Backdrop
	// Transform maps local sampling coordinates to the frame.
	Transform Affine
	W, H      int
	Pix       []float32
}

// Sample returns the backdrop at (x, y), in the frame's pixels,
// premultiplied RGB, filtered bilinearly from its texels and clamped to
// its edges, as the shaders' sampleBackdrop.
func (b *BackdropImage) Sample(x, y float32) [3]float32 {
	x, y = b.Transform.Point(x, y)
	k := float32(b.Down)
	u := (x-float32(b.Area.Min.X))/k - 0.5
	v := (y-float32(b.Area.Min.Y))/k - 0.5
	fu, fv := float32(math.Floor(float64(u))), float32(math.Floor(float64(v)))
	tx, ty := u-fu, v-fv
	x0, y0 := min(max(int(fu), 0), b.W-1), min(max(int(fv), 0), b.H-1)
	x1, y1 := min(max(int(fu)+1, 0), b.W-1), min(max(int(fv)+1, 0), b.H-1)
	p00, p10 := b.Pix[4*(y0*b.W+x0):][:3], b.Pix[4*(y0*b.W+x1):][:3]
	p01, p11 := b.Pix[4*(y1*b.W+x0):][:3], b.Pix[4*(y1*b.W+x1):][:3]
	var c [3]float32
	for i := range c {
		top := p00[i]*(1-tx) + p10[i]*tx
		bot := p01[i]*(1-tx) + p11[i]*tx
		c[i] = top*(1-ty) + bot*ty
	}
	return c
}

// SDRoundRect returns the signed distance from (px, py) to the edge of
// the rounded rectangle r with radii (fitted), negative inside, as the
// shaders' sdRoundRect.
func SDRoundRect(r Rect, radii [4]float32, px, py float32) float32 {
	hx, hy := r.W/2, r.H/2
	qx, qy := px-r.X-hx, py-r.Y-hy
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
	ax, ay := absf(qx)-hx+rad, absf(qy)-hy+rad
	mx, my := max(ax, 0), max(ay, 0)
	return float32(math.Sqrt(float64(mx*mx+my*my))) + min(max(ax, ay), 0) - rad
}

func absf(v float32) float32 {
	if v < 0 {
		return -v
	}
	return v
}
