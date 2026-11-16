package glass

import (
	"math"

	"github.com/egoist/mygo/internal/scene"
)

// pixels draws the glass on the CPU, as its shaders do on the GPU.
type pixels struct {
	m          material
	rect       scene.Rect
	radii      [4]float32
	lx, ly     float32
	toneTable  [toneSteps + 1]float32
	toneLinear bool
}

// toneSteps is how many lightnesses the CPU keeps the tone curve at,
// between which it is a line, within 3e-5 of the curve: its power makes
// it slow on the CPU.
const toneSteps = 1024

// Begin reads the material of op and readies its tone curve.
func (px *pixels) Begin(op *scene.EffectOp, rect scene.Rect, radii [4]float32) {
	p := &op.Params
	px.m = material{
		blur:  op.Blur,
		bezel: p[0][0], refraction: p[0][1], rim: p[0][2], rimWidth: p[0][3],
		tint: p[1],
		low:  p[2][0], high: p[2][1], curve: p[2][2], saturation: p[2][3],
	}
	px.lx, px.ly = p[3][0], p[3][1]
	px.rect, px.radii = rect, radii
	px.toneLinear = px.m.curve == 1
	if !px.toneLinear {
		for i := range px.toneTable {
			px.toneTable[i] = 1 - pow(1-float32(i)/toneSteps, px.m.curve)
		}
	}
}

// tone is the material's tone from the table.
func (px *pixels) tone(c [3]float32) [3]float32 {
	m := &px.m
	l := 0.2126*c[0] + 0.7152*c[1] + 0.0722*c[2]
	var v float32
	if px.toneLinear {
		v = min(max(l, 0), 1)
	} else {
		x := min(max(l, 0), 1) * toneSteps
		i := min(int(x), toneSteps-1)
		f := x - float32(i)
		v = px.toneTable[i] + (px.toneTable[i+1]-px.toneTable[i])*f
	}
	return toneAt(c, l, m.low+(m.high-m.low)*v, m)
}

// Color returns the color of the glass at the pixel center (x, y), as the
// shaders' effect does.
func (px *pixels) Color(x, y float32, b *scene.BackdropImage) [4]float32 {
	m := &px.m
	t := max(-scene.SDRoundRect(px.rect, px.radii, x, y), 0)
	var nx, ny float32
	if t < m.bezel || t < 3*m.rimWidth {
		nx, ny = normal(x, y, px.rect, px.radii)
	}
	sx, sy := x, y
	if t < m.bezel {
		d := m.refraction * lens(t, m.bezel)
		sx, sy = x-nx*d, y-ny*d
	}
	c := px.tone(b.Sample(sx, sy))
	for i := range c {
		c[i] += (m.tint[i] - c[i]) * m.tint[3]
	}
	if m.rim > 0 && t < 3*m.rimWidth {
		q := t / m.rimWidth
		a := m.rim * float32(math.Exp(float64(-q*q)))
		l := abs(nx*px.lx + ny*px.ly)
		for i := range c {
			c[i] += (l - c[i]) * a
		}
	}
	return [4]float32{c[0], c[1], c[2], 1}
}
