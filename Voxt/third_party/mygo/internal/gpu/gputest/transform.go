package gputest

import (
	"math"

	"github.com/egoist/mygo/internal/scene"
)

// TransformedScene exercises all primitives and effects under affine
// composition, with nested transformed clips and an inline glyph transform.
func TransformedScene() *scene.Scene {
	s := Scene()
	s.Clear = scene.Color{R: 100, G: 120, B: 160, A: 80}
	m := scene.Translation(25, 20).Mul(scene.Rotation(4)).Mul(scene.Scaling(0.8, 0.9))
	for i := range s.Ops {
		s.Ops[i].Transform = m
	}
	if len(s.Glyphs) > 0 {
		s.Glyphs[0].Transform = scene.Rotation(15)
	}
	return s
}

// PerspectiveScene exercises all primitives and effects seen in
// perspective: the scene turned 20 degrees about its vertical center line,
// seen from 900 pixels, as RotationY and Perspective make it, with a
// skewed glyph and a nested clip turned the other way.
func PerspectiveScene() *scene.Scene {
	s := Scene()
	s.Clear = scene.Color{R: 100, G: 120, B: 160, A: 80}
	turn := func(degrees, d, cx, cy float32) scene.Affine {
		sin, cos := math.Sincos(float64(degrees) * math.Pi / 180)
		m := scene.Projective(float32(cos), 0, 0, 1, 0, 0, float32(sin)/d, 0)
		return scene.Translation(cx, cy).Mul(m).Mul(scene.Translation(-cx, -cy))
	}
	m := turn(20, 900, float32(s.Width)/2, float32(s.Height)/2)
	for i := range s.Ops {
		s.Ops[i].Transform = m
	}
	if len(s.Glyphs) > 0 {
		s.Glyphs[0].Transform = scene.Matrix(1, 0, 0.3, 1, 0, 0)
	}
	return s
}

// AxisClipScene is content under a hard clip that a pose scales and moves
// (a scroll view in a panel that is entering): the clip edges fall in the middle
// of pixels (the CPU renderer samples a pixel on a 4x4 grid, so its
// coverage is exact there).
func AxisClipScene() *scene.Scene {
	s := &scene.Scene{Width: 200, Height: 160, Clear: scene.Color{R: 246, G: 247, B: 249, A: 255}}
	m := scene.Translation(7.5, 5.5).Mul(scene.Scaling(.5, .5))
	red, blue := scene.Color{R: 220, G: 40, B: 40, A: 255}, scene.Color{R: 40, G: 60, B: 220, A: 255}
	s.Ops = []scene.Op{
		{Kind: scene.OpFill, Rect: scene.Rect{X: 5, Y: 5, W: 120, H: 100}, Radii: [4]float32{14, 14, 14, 14}, Color: blue, Transform: m},
		{Kind: scene.OpPushClip, Rect: scene.Rect{X: 10, Y: 12, W: 80, H: 60}, Transform: m},
		{Kind: scene.OpFill, Rect: scene.Rect{W: 200, H: 160}, Color: red, Transform: m},
		{Kind: scene.OpFill, Rect: scene.Rect{X: 40, Y: 30, W: 90, H: 70}, Radii: [4]float32{20, 20, 20, 20}, Color: blue, Transform: m},
		{Kind: scene.OpPopClip, Transform: m},
	}
	return s
}
