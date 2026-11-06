package ui

import (
	"fmt"
	"strings"

	"github.com/egoist/mygo/internal/scene"
)

// Color is a color with straight (not premultiplied) alpha. R, G, B and A are
// its sRGB value. A Color from [Oklch] may lie outside the sRGB gamut and
// then also holds the wide color, which a window that draws a wide gamut
// (on a Mac whose screen shows one) shows in place of R, G and B; every
// other one shows them, the nearest sRGB color, found as CSS Color 4 does.
// Set R, G or B of such a color and the wide color is stale: make a new one
// instead.
type Color struct {
	R, G, B, A uint8
	wide       wideRGB
}

// Transparent is the color of nothing.
var Transparent = Color{}

// RGB returns an opaque color.
func RGB(r, g, b uint8) Color { return Color{R: r, G: g, B: b, A: 255} }

// RGBA returns a color with alpha between 0 and 1.
func RGBA(r, g, b uint8, alpha float32) Color { return Color{R: r, G: g, B: b, A: alphaByte(alpha)} }

// Hex parses "#rgb", "#rgba", "#rrggbb" or "#rrggbbaa". It panics on other
// input, which is a mistake in the program.
func Hex(s string) Color {
	c, err := parseHex(s)
	if err != nil {
		panic(err)
	}
	return c
}

func parseHex(s string) (Color, error) {
	h := strings.TrimPrefix(strings.TrimSpace(s), "#")
	var v [8]uint8
	for i := 0; i < len(h) && i < len(v); i++ {
		d, ok := hexDigit(h[i])
		if !ok {
			return Color{}, fmt.Errorf("ui: invalid color %q", s)
		}
		v[i] = d
	}
	switch len(h) {
	case 3, 4:
		// #rgb and #rgba double each digit.
		a := uint8(15)
		if len(h) == 4 {
			a = v[3]
		}
		return Color{R: v[0] * 17, G: v[1] * 17, B: v[2] * 17, A: a * 17}, nil
	case 6, 8:
		a := uint8(255)
		if len(h) == 8 {
			a = v[6]<<4 | v[7]
		}
		return Color{R: v[0]<<4 | v[1], G: v[2]<<4 | v[3], B: v[4]<<4 | v[5], A: a}, nil
	}
	return Color{}, fmt.Errorf("ui: invalid color %q", s)
}

// hexDigit returns the value of a hexadecimal digit.
func hexDigit(c byte) (uint8, bool) {
	switch {
	case '0' <= c && c <= '9':
		return c - '0', true
	case 'a' <= c && c <= 'f':
		return c - 'a' + 10, true
	case 'A' <= c && c <= 'F':
		return c - 'A' + 10, true
	}
	return 0, false
}

func alphaByte(a float32) uint8 {
	if a <= 0 {
		return 0
	}
	if a >= 1 {
		return 255
	}
	return uint8(a*255 + 0.5)
}

// Alpha returns the color with its alpha multiplied by a.
func (c Color) Alpha(a float32) Color {
	c.A = alphaByte(float32(c.A) / 255 * a)
	return c
}

// gray returns the color in shades of gray, of the same luminance.
func (c Color) gray() Color {
	l := uint8(0.2126*float32(c.R) + 0.7152*float32(c.G) + 0.0722*float32(c.B) + 0.5)
	return Color{R: l, G: l, B: l, A: c.A}
}

// Mix returns the color t of the way from c to o.
func (c Color) Mix(o Color, t float32) Color {
	t = max(0, min(t, 1))
	m := func(a, b uint8) uint8 { return uint8(float32(a)*(1-t) + float32(b)*t + 0.5) }
	if c.wide.ok() || o.wide.ok() {
		// The wide colors mix, so that the mix draws the same on every
		// renderer when it is inside the sRGB gamut, and wide only when
		// it is not.
		r := mixWide(c, o, t)
		r.A = m(c.A, o.A)
		return r
	}
	return Color{R: m(c.R, o.R), G: m(c.G, o.G), B: m(c.B, o.B), A: m(c.A, o.A)}
}

// Over returns c composited over the opaque color base.
func (c Color) Over(base Color) Color {
	a := float32(c.A) / 255
	o := c.Mix(base, 1-a)
	o.A = base.A
	return o
}

func (c Color) scene() scene.Color { return scene.Color{R: c.R, G: c.G, B: c.B, A: c.A} }
