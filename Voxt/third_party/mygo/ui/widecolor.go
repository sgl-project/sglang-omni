package ui

import (
	"math"

	"github.com/egoist/mygo/internal/gamut"
	"github.com/egoist/mygo/internal/scene"
)

// Oklch returns the color of CSS's oklch(l c h): lightness l from 0 to 1,
// chroma c (about 0 to 0.4) and hue h in degrees. Use Alpha for oklch(l c h
// / a).
//
// A color Display P3 cannot show loses chroma, keeping its lightness and
// hue, as CSS does; a negative chroma is 0. A color outside the sRGB gamut
// is shown as it is by a window that draws a wide gamut, and as the
// nearest sRGB color by every other (see [Color]).
func Oklch(l, c, h float32) Color {
	r, g, b := gamut.Map(gamut.DisplayP3, float64(l), float64(c), float64(h))
	r, g, b = gamut.LinearP3ToSRGB(r, g, b)
	return fromWide(float32(gamut.Encode(r)), float32(gamut.Encode(g)), float32(gamut.Encode(b)))
}

// wideRGB is the red, green and blue of a color outside the sRGB gamut,
// sRGB-encoded and beyond 0 to 1 (extended sRGB), in fixed point: wideOne
// is 1. They reach ±4, beyond what any display shows, in steps finer than
// those of the float16 targets that show them. The zero value is none,
// which no such color is, as some of its components are outside 0 to 1.
type wideRGB [3]int16

const wideOne = 1 << 13

func (w wideRGB) ok() bool { return w != wideRGB{} }

func (w wideRGB) rgb() (r, g, b float32) {
	return float32(w[0]) / wideOne, float32(w[1]) / wideOne, float32(w[2]) / wideOne
}

func toWide(v float32) int16 {
	return int16(max(-math.MaxInt16, min(math.Round(float64(v)*wideOne), math.MaxInt16)))
}

// fromWide returns the opaque color of the extended sRGB components r, g
// and b: R, G and B are them when they are inside the sRGB gamut, and the
// nearest sRGB color, found as CSS Color 4 does, with the wide color kept,
// when they are not. It only depends on the components as wideRGB holds
// them, so that a color mixed with itself is itself.
func fromWide(r, g, b float32) Color {
	const eps = 1.0 / 512
	num := func(v float32) bool { return !math.IsNaN(float64(v)) && !math.IsInf(float64(v), 0) }
	if !num(r) || !num(g) || !num(b) {
		return Color{A: 255}
	}
	w := wideRGB{toWide(r), toWide(g), toWide(b)}
	r, g, b = w.rgb()
	in := func(v float32) bool { return v >= -eps && v <= 1+eps }
	if in(r) && in(g) && in(b) {
		byteOf := func(v float32) uint8 { return uint8(max(0, min(v, 1))*255 + 0.5) }
		return Color{R: byteOf(r), G: byteOf(g), B: byteOf(b), A: 255}
	}
	l, c, h := gamut.Oklch(gamut.Oklab(gamut.Decode(float64(r)), gamut.Decode(float64(g)), gamut.Decode(float64(b))))
	lr, lg, lb := gamut.Map(gamut.SRGB, l, c, h)
	byteOf := func(v float64) uint8 { return uint8(gamut.Encode(v)*255 + 0.5) }
	return Color{R: byteOf(lr), G: byteOf(lg), B: byteOf(lb), A: 255, wide: w}
}

// SRGB returns c without its wide color: the nearest sRGB color, which a
// window that cannot draw a wide gamut shows.
func (c Color) SRGB() Color {
	c.wide = wideRGB{}
	return c
}

// extended returns the red, green and blue of c, sRGB-encoded: from 0 to 1
// for a color inside the sRGB gamut, and beyond for one outside it
// (extended sRGB), as a target drawing a wide gamut takes them.
func (c Color) extended() (r, g, b float32) {
	if c.wide.ok() {
		return c.wide.rgb()
	}
	return float32(c.R) / 255, float32(c.G) / 255, float32(c.B) / 255
}

// wideRGBA returns the wide color of c, with its alpha, if it has one and
// shows.
func (c Color) wideRGBA() ([4]float32, bool) {
	if !c.wide.ok() || c.A == 0 {
		return [4]float32{}, false
	}
	r, g, b := c.wide.rgb()
	return [4]float32{r, g, b, float32(c.A) / 255}, true
}

// mixWide returns the opaque color t of the way from c to o, one of which
// is wide, mixed as Mix does, with R, G and B those of the color mixed.
func mixWide(c, o Color, t float32) Color {
	cr, cg, cb := c.extended()
	or, og, ob := o.extended()
	m := func(a, b float32) float32 { return a*(1-t) + b*t }
	return fromWide(m(cr, or), m(cg, og), m(cb, ob))
}

// wideSet returns the wide colors of an op with the colors c, c2 and
// border, those that have one.
func wideSet(c, c2, border Color) (w scene.WideColors) {
	if a, ok := c.wideRGBA(); ok {
		w.Color, w.Set = a, w.Set|scene.WideColor
	}
	if a, ok := c2.wideRGBA(); ok {
		w.Color2, w.Set = a, w.Set|scene.WideColor2
	}
	if a, ok := border.wideRGBA(); ok {
		w.Border, w.Set = a, w.Set|scene.WideBorder
	}
	return w
}

// wide returns the Wide of an op with the colors c, c2 and border, the
// zero Color for those it does not draw (see scene.Op.Wide).
func (p *Painter) wide(c, c2, border Color) uint16 {
	if !c.wide.ok() && !c2.wide.ok() && !border.wide.ok() {
		return 0
	}
	w := wideSet(c, c2, border)
	if w.Set == 0 {
		return 0
	}
	return p.addWide(w)
}

// glyphWide returns the Wide of a glyph of color c, its opacity included.
func (p *Painter) glyphWide(c Color) uint16 {
	if !c.wide.ok() || c.A == 0 {
		return 0
	}
	return p.addWide(wideSet(c, Color{}, Color{}))
}

// addWide adds w to the wide colors of the scene, unless the last ones
// added are w, and returns their Wide. Past what Wide counts, which no
// scene should reach, the colors draw as their nearest sRGB ones.
func (p *Painter) addWide(w scene.WideColors) uint16 {
	n := len(p.s.Wide)
	if n > 0 && p.s.Wide[n-1] == w {
		return uint16(n)
	}
	if n >= math.MaxUint16 {
		return 0
	}
	p.s.Wide = append(p.s.Wide, w)
	return uint16(n + 1)
}
