// Package scene is the display list the ui package paints a frame into and
// the renderers draw: rounded rectangles with borders and gradients, box
// shadows, glyphs from a shared atlas, images, effects and clips, in paint
// order.
// Geometry is in device pixels with the origin at the top-left corner.
package scene

import (
	"image"
	"image/draw"
	"sync/atomic"
)

// Color is a straight (not premultiplied) sRGB color.
type Color struct{ R, G, B, A uint8 }

// Premul returns the color premultiplied by its alpha, with every component
// between 0 and 1, times opacity.
func (c Color) Premul(opacity float32) [4]float32 {
	a := float32(c.A) / 255 * opacity
	return [4]float32{float32(c.R) / 255 * a, float32(c.G) / 255 * a, float32(c.B) / 255 * a, a}
}

// WideColors are the colors outside the sRGB gamut of an op or a glyph
// (see Op.Wide): straight RGBA with sRGB-encoded components beyond 0 to 1
// (extended sRGB), for those of Color, Color2 and BorderColor a Set bit
// names, of which a glyph has only Color, times its opacity.
type WideColors struct {
	Color, Color2, Border [4]float32
	Set                   WideSet
}

// WideSet says which colors of WideColors replace those of the op.
type WideSet uint8

const (
	WideColor WideSet = 1 << iota
	WideColor2
	WideBorder
)

// Rect is a rectangle in device pixels.
type Rect struct{ X, Y, W, H float32 }

// Empty reports whether the rectangle has no area.
func (r Rect) Empty() bool { return r.W <= 0 || r.H <= 0 }

// Intersect returns the intersection of r and o.
func (r Rect) Intersect(o Rect) Rect {
	x0, y0 := max(r.X, o.X), max(r.Y, o.Y)
	x1, y1 := min(r.X+r.W, o.X+o.W), min(r.Y+r.H, o.Y+o.H)
	if x1 <= x0 || y1 <= y0 {
		return Rect{X: x0, Y: y0}
	}
	return Rect{x0, y0, x1 - x0, y1 - y0}
}

// Contains reports whether the point is inside r.
func (r Rect) Contains(x, y float32) bool {
	return x >= r.X && y >= r.Y && x < r.X+r.W && y < r.Y+r.H
}

// Kind is the kind of an Op.
type Kind uint8

const (
	// OpFill paints a rounded rectangle with Color, or the Paint of Color
	// and Color2, and, where Border is positive, a border of BorderColor
	// inside its edges, dashed or not.
	OpFill Kind = iota
	// OpShadow paints the blurred shadow of a rounded rectangle: Rect and
	// Radii are the shadow's box, already offset and spread, Blur its blur
	// radius and Color its color. Unless Cast is empty, the shadow shows
	// only outside it, the box casting the shadow, as CSS's box-shadow
	// does. An Inset shadow shows only inside Cast instead, where Rect,
	// the hole it leaves, blurred, does not cover it, as CSS's box-shadow:
	// inset does; Rect may then be empty, which leaves no hole.
	OpShadow
	// OpGlyphs paints Scene.Glyphs[Start:End]. With a gradient Paint, its
	// mask glyphs take the gradient from Color to Color2, times Opacity,
	// instead of their own colors.
	OpGlyphs
	// OpImage paints Src of Image into Rect, clipped to Radii, with
	// Opacity, in shades of gray if Grayscale, tinted by Color unless it
	// is zero (its pixels times its straight color and alpha). A negative
	// Src.W or Src.H flips the image along that axis, Src.X or Src.Y then
	// being its right or bottom edge; nothing outside Src shows.
	OpImage
	// OpPushClip clips the following ops to Rect with Radii, intersected
	// with the clip in effect, until the matching OpPopClip.
	OpPushClip
	// OpPopClip restores the clip in effect before the matching OpPushClip.
	OpPopClip
	// OpEffect paints the rounded rectangle Rect with Radii with an effect,
	// Scene.Effects[Start]: shaders of a package outside the renderers (see
	// Effect), with Opacity; or, for a shader's effect, mask glyphs, which
	// Rect then holds (EffectOp.Glyphs).
	OpEffect
	// OpHole makes the rounded rectangle Rect with Radii transparent,
	// within the clip: what the ops before it painted there is gone, and
	// what the window shows under the frame, as a web view, shows through.
	// The ops after it paint over it as over any transparent pixels. Below
	// full Opacity, what the ops before it painted fades by Opacity
	// instead, so what shows through blends over it.
	OpHole
)

// Op is one drawing operation. Which fields matter depends on Kind.
type Op struct {
	Kind Kind
	Rect Rect
	// Transform maps this operation's geometry to frame pixels. Clips
	// retain the transform with which they were pushed.
	Transform Affine
	// Radii are the corner radii: top-left, top-right, bottom-right and
	// bottom-left.
	Radii [4]float32
	// Continuous curves the rounded corners of Rect, and of Cast, the way
	// Apple draws them (Core Animation's continuous corner curve, SwiftUI's
	// rounded rectangles): the curve starts further from the corner and
	// bends gradually, instead of as a quarter circle. Corners that are
	// round in both directions, as a circle's, stay circular. Only the
	// renderers of macOS draw them, the CPU's and Metal's.
	Continuous bool

	Color Color
	// Paint, unless PaintSolid, fills with Color and Color2 as Gradient
	// says.
	Paint  Paint
	Color2 Color
	// Wide, unless 0, is 1 + the index in Scene.Wide of the op's colors
	// outside the sRGB gamut, which the renderers drawing a wide gamut
	// draw in place of Color, Color2 and BorderColor; those hold the
	// nearest sRGB colors, which every other renderer draws. It fits in
	// the room the fields around it leave.
	Wide     uint16
	Gradient [4]float32

	// Border holds the widths of the border on the top, right, bottom and
	// left; Dashed dashes it.
	Border      [4]float32
	BorderColor Color
	Dashed      bool
	// BorderPaint paints an OpFill's border with Paint, its gradient of
	// stops (Paint.HasStops), times BorderColor (opaque white for the
	// gradient as it is), and its inside with plain Color.
	BorderPaint bool

	// Inset makes an OpShadow an inner shadow.
	Inset bool
	// Blur is an OpShadow's blur, and an OpImage's: the standard deviation
	// of a Gaussian blur of the image's Src, in its pixels, before it is
	// drawn, transparent beyond Src, as CSS's filter: blur() is. Large
	// blurs are computed at a fraction of the size, as backdrops' are
	// (BackdropOf).
	Blur float32
	// Cast is the box casting an OpShadow, with CastRadii.
	Cast      Rect
	CastRadii [4]float32

	Start, End int32
	// Stops are a gradient's colors with stops (PaintLinearStops and the
	// others), Scene.Stops[Stops[0]:Stops[1]].
	Stops [2]int32

	Image     *Image
	Src       Rect
	Grayscale bool
	// Pixelated samples an OpImage's nearest pixel, as pixel art draws,
	// where it would interpolate between pixels.
	Pixelated bool
	// Opacity multiplies the op's colors; 0 means 1.
	Opacity float32
}

// Paint is what fills an OpFill besides plain Color, or the masks of an
// OpGlyphs (gradients only).
type Paint uint8

const (
	PaintSolid Paint = iota
	// PaintLinear is a linear gradient from Color at (Gradient[0],
	// Gradient[1]) to Color2 at (Gradient[2], Gradient[3]), mixed in sRGB;
	// PaintOklab mixes them in Oklab. Both mix premultiplied colors, as
	// CSS does.
	PaintLinear
	PaintOklab
	// PaintStripes draws stripes of Color over Color2: Gradient holds the
	// unit vector across the stripes, their width and their period, from
	// the top-left corner of Rect.
	PaintStripes
	// PaintLinearStops is a linear gradient of the op's Stops, from
	// (Gradient[0], Gradient[1]) at 0 to (Gradient[2], Gradient[3]) at 1,
	// mixed in sRGB; PaintLinearStopsOklab mixes them in Oklab.
	PaintLinearStops
	PaintLinearStopsOklab
	// PaintConic is a conic gradient of the op's Stops around
	// (Gradient[0], Gradient[1]), from 0 at the angle Gradient[2], in
	// radians clockwise from upwards, to 1 a turn later, mixed in sRGB, as
	// CSS's conic-gradient() is; PaintConicOklab mixes them in Oklab.
	PaintConic
	PaintConicOklab
)

// Stop is a color of a gradient with stops (Op.Stops), at a place along
// it from 0 to 1. A gradient shows its first color before its first stop
// and its last after its last, and mixes premultiplied colors between, as
// CSS does; stops in order, at most MaxStops of them.
type Stop struct {
	Color Color
	At    float32
}

// MaxStops is how many stops a gradient has at most, which the shaders
// loop through.
const MaxStops = 32

// HasStops reports whether paint takes its colors from the op's Stops.
func (p Paint) HasStops() bool { return p >= PaintLinearStops && p <= PaintConicOklab }

// Uniform returns the widths of a border as wide on every side.
func Uniform(w float32) [4]float32 { return [4]float32{w, w, w, w} }

// HasBorder reports whether any of the widths is positive.
func HasBorder(w [4]float32) bool { return w[0] > 0 || w[1] > 0 || w[2] > 0 || w[3] > 0 }

// InnerRadii returns the radii of the inner edge of a border of widths w
// (top, right, bottom, left) inside a rounded rectangle r with radii,
// which FitRadii or Corners already fitted: each corner's less the wider
// of its two sides, fitted to the inner rectangle, negative as radii are
// for continuous corners.
func InnerRadii(r Rect, radii, w [4]float32) (Rect, [4]float32) {
	inner := Rect{X: r.X + w[3], Y: r.Y + w[0], W: r.W - w[1] - w[3], H: r.H - w[0] - w[2]}
	continuous := radii[0] < 0 || radii[1] < 0 || radii[2] < 0 || radii[3] < 0
	var out [4]float32
	for i, side := range [4][2]int{{0, 3}, {0, 1}, {2, 1}, {2, 3}} {
		out[i] = max(abs(radii[i])-max(w[side[0]], w[side[1]]), 0)
	}
	return inner, Corners(inner, out, continuous)
}

// Glyph is a glyph mask or color glyph copied from an atlas into a frame.
type Glyph struct {
	// Transform is applied before its Op's transform, for inline text.
	Transform Affine
	// X, Y, W and H place the glyph's bitmap, in device pixels.
	X, Y, W, H float32
	// U, V, UW and VH are the bitmap's rectangle in its atlas.
	U, V, UW, VH uint16
	// Color tints mask glyphs; color glyphs take its alpha only.
	Color Color
	// Wide, unless 0, is 1 + the index in Scene.Wide of the glyph's
	// color outside the sRGB gamut, as Op.Wide.
	Wide uint16
	// Colored glyphs come from Scene.ColorAtlas, the others from
	// Scene.MaskAtlas.
	Colored bool
	// Subpixel glyphs come from Scene.ColorAtlas too, but hold the
	// coverage of the red, green and blue subpixels of each pixel
	// (ClearType), each blending its channel of Color.
	Subpixel bool
	// Thin glyphs, of fonts too thin for antialiasing such as Courier New,
	// take half more contrast whatever their color (see TextCoverage), as
	// Direct2D gives them.
	Thin bool
}

// TextParams corrects the coverage of mask and subpixel glyphs for the
// gamma and contrast text is blended with, as Direct2D does; the zero
// value leaves it as it is.
type TextParams struct {
	// GammaRatios are Direct2D's alpha correction of a gamma (see
	// GammaRatios).
	GammaRatios [4]float32
	// Contrast enhances the contrast of mask glyphs, and SubpixelContrast
	// that of subpixel glyphs, the more the darker their color: DirectWrite's
	// grayscale enhanced contrast and enhanced contrast.
	Contrast, SubpixelContrast float32
}

// GammaRatios returns Direct2D's alpha correction ratios for text blended
// at a gamma between 1 and 2.2, as Windows Terminal computes them.
func GammaRatios(gamma float32) [4]float32 {
	// Microsoft's ratios, for gamma 1.0, 1.1, ... 2.2, divided by 4.
	table := [13][4]float32{
		{0, 0, 0, 0},
		{0.0166, -0.0807, 0.2227, -0.0751},
		{0.0350, -0.1760, 0.4325, -0.1370},
		{0.0543, -0.2821, 0.6302, -0.1876},
		{0.0739, -0.3963, 0.8167, -0.2287},
		{0.0933, -0.5161, 0.9926, -0.2616},
		{0.1121, -0.6395, 1.1588, -0.2877},
		{0.1300, -0.7649, 1.3159, -0.3080},
		{0.1469, -0.8911, 1.4644, -0.3234},
		{0.1627, -1.0170, 1.6051, -0.3347},
		{0.1773, -1.1420, 1.7385, -0.3426},
		{0.1908, -1.2652, 1.8650, -0.3476},
		{0.2031, -1.3864, 1.9851, -0.3501},
	}
	i := min(max(int(gamma*10+0.5), 10), 22) - 10
	const norm13 = float32(float64(0x10000) / (255 * 255) * 4)
	const norm24 = float32(float64(0x100) / 255 * 4)
	r := table[i]
	return [4]float32{norm13 * r[0] / 4, norm24 * r[1] / 4, norm13 * r[2] / 4, norm24 * r[3] / 4}
}

// ThinBoost is the contrast Direct2D adds for thin glyphs.
const ThinBoost = 0.5

// TextCoverage corrects the coverage a of a glyph of straight color c, as
// Direct2D blends text: it enhances the contrast by contrast, the more the
// darker c is, plus boost, and corrects for the gamma of the ratios g.
func TextCoverage(a float32, c [3]float32, contrast, boost float32, g [4]float32) float32 {
	k := contrast*min(max(3-4*(0.30*c[0]+0.59*c[1]+0.11*c[2]), 0), 1) + boost
	a = a * (k + 1) / (a*k + 1)
	f := 0.25*c[0] + 0.5*c[1] + 0.25*c[2]
	return min(max(a+a*(1-a)*((g[0]*f+g[1])*a+(g[2]*f+g[3])), 0), 1)
}

// SubpixelCoverage corrects the coverage of each subpixel of a subpixel
// glyph of straight color c, as Direct2D blends ClearType text.
func SubpixelCoverage(a, c [3]float32, contrast, boost float32, g [4]float32) [3]float32 {
	k := contrast*min(max(3-4*(0.30*c[0]+0.59*c[1]+0.11*c[2]), 0), 1) + boost
	for i := range a {
		v := a[i] * (k + 1) / (a[i]*k + 1)
		a[i] = min(max(v+v*(1-v)*((g[0]*c[i]+g[1])*v+(g[2]*c[i]+g[3])), 0), 1)
	}
	return a
}

// Scene is a frame's display list.
type Scene struct {
	// Width and Height of the frame in device pixels; Scale is how many
	// device pixels a DIP is (0 for 1).
	Width, Height int
	Scale         float32
	// Clear is the color the frame starts with.
	Clear  Color
	Ops    []Op
	Glyphs []Glyph
	// Stops holds the stops of gradients (Op.Stops).
	Stops []Stop
	// Effects holds the effects of OpEffect operations.
	Effects []EffectOp
	// Words holds the words of the effects' uniforms (EffectOp.Uniforms
	// and WideUniforms), which the frame's painter allocates from
	// (Painter.EffectWords), kept from frame to frame.
	Words []uint32
	// Wide holds the colors outside the sRGB gamut of ops and glyphs (see
	// Op.Wide), and those of the parameters of effects, which no op points
	// at. Without any, the scene draws the same on every renderer.
	Wide []WideColors
	// Text corrects the coverage of mask and subpixel glyphs.
	Text TextParams
	// MaskAtlas holds coverage masks (one byte per pixel), ColorAtlas
	// premultiplied RGBA bitmaps such as emoji, and the coverage of
	// subpixel glyphs.
	MaskAtlas, ColorAtlas *Atlas
	// Canvases are the canvases (NewCanvas) drawn anew for this frame,
	// which renderers draw before it, whether it shows them or not, while
	// the atlases their glyphs point at hold them.
	Canvases []*Image
}

// EachCanvas calls fn with the canvases the scene draws anew or shows,
// each once, those a canvas shows before it.
func (s *Scene) EachCanvas(fn func(*Image)) {
	seen := map[*Image]bool{}
	var visit func(c *Image)
	visit = func(c *Image) {
		if c == nil || c.Scene == nil || seen[c] {
			return
		}
		seen[c] = true
		c.Scene.eachShown(visit)
		fn(c)
	}
	for _, c := range s.Canvases {
		visit(c)
	}
	s.eachShown(visit)
}

// eachShown calls fn with the canvases the scene's images show, and its
// shaders read.
func (s *Scene) eachShown(fn func(*Image)) {
	for i := range s.Ops {
		if op := &s.Ops[i]; op.Kind == OpImage && op.Image != nil && op.Image.Scene != nil {
			fn(op.Image)
		}
	}
	for i := range s.Effects {
		for _, img := range s.Effects[i].Images {
			if img != nil && img.Scene != nil {
				fn(img)
			}
		}
	}
}

// Reset empties the scene for another frame, keeping its storage.
func (s *Scene) Reset(width, height int, color Color) {
	s.Width, s.Height, s.Clear = width, height, color
	// Ops and effects can keep images and effect resources alive beyond
	// the end of the shorter list the next frame paints.
	clear(s.Ops)
	s.Ops = s.Ops[:0]
	s.Glyphs = s.Glyphs[:0]
	s.Stops = s.Stops[:0]
	clear(s.Effects)
	s.Effects = s.Effects[:0]
	s.Wide = s.Wide[:0]
	s.Words = s.Words[:0]
	clear(s.Canvases)
	s.Canvases = s.Canvases[:0]
}

var lastImageID atomic.Uint64

// Image is a bitmap a scene draws. Renderers keep a texture per image and
// upload it again when its version changes.
type Image struct {
	id      uint64
	version uint64
	// W and H are the size in pixels; Pix holds premultiplied RGBA rows of
	// 4*W bytes.
	W, H int
	Pix  []byte
	// Scene, in a canvas (NewCanvas), is what the image shows: renderers
	// draw it into the image's texture whenever the version changes,
	// rather than upload Pix, which the CPU renderer draws it into, and
	// PixVersion is the version Pix holds.
	Scene      *Scene
	PixVersion uint64
}

// NewCanvas returns a w×h image that renderers draw from a scene of its
// own, cleared to transparent: render to texture.
func NewCanvas(w, h int) *Image {
	return &Image{id: lastImageID.Add(1), version: 1, W: w, H: h, Scene: &Scene{Width: w, Height: h, Scale: 1}}
}

// NewImage converts img to a premultiplied RGBA bitmap.
func NewImage(img image.Image) *Image {
	b := img.Bounds()
	rgba, ok := img.(*image.RGBA)
	if !ok || rgba.Stride != 4*b.Dx() || b.Min != (image.Point{}) {
		rgba = image.NewRGBA(image.Rect(0, 0, b.Dx(), b.Dy()))
		draw.Draw(rgba, rgba.Bounds(), img, b.Min, draw.Src)
	}
	return &Image{id: lastImageID.Add(1), version: 1, W: b.Dx(), H: b.Dy(), Pix: rgba.Pix}
}

// NewImageRGBA wraps premultiplied RGBA pixels of a w×h bitmap.
func NewImageRGBA(w, h int, pix []byte) *Image {
	return &Image{id: lastImageID.Add(1), version: 1, W: w, H: h, Pix: pix}
}

// ID identifies the image for texture caches.
func (m *Image) ID() uint64 { return m.id }

// Version changes whenever the pixels change.
func (m *Image) Version() uint64 { return m.version }

// Changed records that Pix was modified.
func (m *Image) Changed() { m.version++ }

// Resize keeps the image's texture-cache identity while changing its size.
// Renderers replace the backing texture on a size change, so an animated
// clip does not leave a new texture in the cache on every frame.
func (m *Image) Resize(w, h int) {
	m.W, m.H = w, h
	if cap(m.Pix) >= 4*w*h {
		m.Pix = m.Pix[:4*w*h]
	} else {
		m.Pix = make([]byte, 4*w*h)
	}
	m.Changed()
}

// FitRadii scales corner radii down so that adjacent ones fit along each
// side of r, as CSS does; renderers apply it before drawing.
func FitRadii(r Rect, radii [4]float32) [4]float32 {
	f := float32(1)
	if s := radii[0] + radii[1]; s > r.W && s > 0 {
		f = min(f, r.W/s)
	}
	if s := radii[3] + radii[2]; s > r.W && s > 0 {
		f = min(f, r.W/s)
	}
	if s := radii[0] + radii[3]; s > r.H && s > 0 {
		f = min(f, r.H/s)
	}
	if s := radii[1] + radii[2]; s > r.H && s > 0 {
		f = min(f, r.H/s)
	}
	for i := range radii {
		radii[i] = max(radii[i]*f, 0)
	}
	return radii
}

// Corners returns radii as renderers take them: fitted to r (FitRadii),
// and negative for continuous corners.
func Corners(r Rect, radii [4]float32, continuous bool) [4]float32 {
	radii = FitRadii(r, radii)
	if continuous {
		for i := range radii {
			radii[i] = -radii[i]
		}
	}
	return radii
}

func abs(v float32) float32 {
	if v < 0 {
		return -v
	}
	return v
}
