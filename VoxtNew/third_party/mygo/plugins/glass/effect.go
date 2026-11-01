package glass

import (
	_ "embed"
	"math"

	"github.com/egoist/mygo/internal/scene"
)

// The shaders of the glass and of the blur, one for each renderer, which
// go generate compiles ahead of time where it can (shaders_darwin.go,
// shaders_windows.go).
var (
	//go:embed glass.metal
	metalSource string
	//go:embed glass.hlsl
	hlslSource string
	//go:embed glass.glsl
	glslSource string
	//go:embed blur.metal
	blurMetalSource string
	//go:embed blur.hlsl
	blurHLSLSource string
	//go:embed blur.glsl
	blurGLSLSource string
)

// Effect is the glass's effect, which the renderers draw (see
// scene.Effect): what is behind the pane, blurred, sampled within the
// bezel where the curved surface refracts it (lens), mapped from its
// lightness (tone), tinted, and lit along the rim.
var Effect = &scene.Effect{
	Name:     "glass",
	Backdrop: true,
	Metal:    scene.EffectCode{Source: metalSource, Compiled: metalLibrary, Sum: metalSum},
	HLSL:     scene.EffectCode{Source: hlslSource, Compiled: hlslBytecode, Sum: hlslSum},
	GLSL:     glslSource,
	Pixels:   func() scene.EffectPixels { return &pixels{} },
}

// blurEffect is a level of a blur (blurLevel), which the renderers draw:
// what is behind it, blurred, by how much the level shows at each pixel,
// along the mask.
var blurEffect = &scene.Effect{
	Name:     "blur",
	Backdrop: true,
	Metal:    scene.EffectCode{Source: blurMetalSource, Compiled: blurMetalLibrary, Sum: blurMetalSum},
	HLSL:     scene.EffectCode{Source: blurHLSLSource, Compiled: blurBytecode, Sum: blurHLSLSum},
	GLSL:     blurGLSLSource,
	Pixels:   func() scene.EffectPixels { return &blurPixels{} },
}

// material is a pane of glass, in device pixels: the effect's parameters.
//
//   - blur is the standard deviation of the blur of what shows through;
//   - bezel is how far from the edge the surface curves, and refraction
//     how far, at most, the bezel bends what shows through toward the
//     middle (lens), which mirrors what is just inside the bezel into it;
//   - low and high are the lightness that black and white, blurred, show
//     at, along a curve bent by curve (1 for a straight line), and
//     saturation multiplies how far colors are from their lightness;
//   - tint colors the glass, straight RGBA, by its alpha, and wideTint
//     is it in extended sRGB, for a target that keeps colors outside the
//     sRGB gamut (ui.Painter.EffectColor);
//   - rim is how much the rim is lit, within about rimWidth of the edge:
//     the more the edge faces the light coming from the angle light
//     (radians, clockwise from the right, the opposite side alike), the
//     lighter, and the less, the darker.
type material struct {
	blur                         float32
	bezel, refraction            float32
	low, high, curve, saturation float32
	tint, wideTint               [4]float32
	rim, rimWidth, light         float32
}

// params returns the effect's parameters, as its shaders read them: p0
// the bezel, the refraction, the rim and its width; p1 the tint; p2 the
// low, high, curve and saturation; p3 the direction of the light; p4 the
// wide tint, which only the Metal shader reads.
func (m *material) params() [5][4]float32 {
	return [5][4]float32{
		{m.bezel, m.refraction, m.rim, m.rimWidth},
		m.tint,
		{m.low, m.high, m.curve, m.saturation},
		{float32(math.Cos(float64(m.light))), float32(math.Sin(float64(m.light)))},
		m.wideTint,
	}
}

// lens returns how far, as a fraction of the refraction, the glass bends
// what shows through it t pixels inside its edge, within a bezel of width
// bezel: the surface's height there is Apple's squircle profile,
// ⁴√(1-(1-x)⁴) for x = t/bezel, and light coming straight down bends as it
// enters glass of refractive index 1.5 (Snell's law), by the tangent of
// the angle it turns, which is √5/2 at most, where the surface is upright.
func lens(t, bezel float32) float32 {
	if bezel <= 0 || t >= bezel {
		return 0
	}
	u := 1 - max(t, 0)/bezel
	v := 1 - u*u*u*u
	// The surface's slope is b/a: the sine and cosine of its angle are
	// b and a over their hypotenuse.
	a := sqrt(v * sqrt(v))
	b := u * u * u
	n := sqrt(a*a + b*b)
	s, c := b/n, a/n
	st := s / 1.5
	ct := sqrt(1 - st*st)
	return (s*ct - c*st) / (c*ct + s*st) / 1.1180340
}

// normal returns the direction away from the middle of a rounded rectangle
// at p, the direction of its edge's normal there, smoothed: as of the
// rectangle with corners half again as round, so that it turns gradually
// around a corner. It is 0 on the lines through the middle, where two
// sides are as near.
func normal(px, py float32, r scene.Rect, radii [4]float32) (gx, gy float32) {
	hx, hy := r.W/2, r.H/2
	cx, cy := px-r.X-hx, py-r.Y-hy
	var rad float32
	switch {
	case cx < 0 && cy < 0:
		rad = radii[0]
	case cx >= 0 && cy < 0:
		rad = radii[1]
	case cx >= 0:
		rad = radii[2]
	default:
		rad = radii[3]
	}
	rad = min(rad*1.5, hx, hy)
	qx, qy := abs(cx)-(hx-rad), abs(cy)-(hy-rad)
	if qx > 0 || qy > 0 {
		mx, my := max(qx, 0), max(qy, 0)
		n := sqrt(mx*mx + my*my)
		return sign(cx) * mx / n, sign(cy) * my / n
	}
	if qx > qy {
		return sign(cx), 0
	}
	return 0, sign(cy)
}

// tone maps the premultiplied color c behind the glass to the glass's own,
// by its lightness, the luminance of its sRGB components, along the
// material's curve, and keeping its colors' distance from it.
func tone(c [3]float32, m *material) [3]float32 {
	l := 0.2126*c[0] + 0.7152*c[1] + 0.0722*c[2]
	return toneAt(c, l, m.low+(m.high-m.low)*(1-pow(max(1-l, 0), m.curve)), m)
}

func toneAt(c [3]float32, l, t float32, m *material) [3]float32 {
	k := (m.high - m.low) * m.saturation
	for i := range c {
		c[i] = min(max(t+(c[i]-l)*k, 0), 1)
	}
	return c
}

func sqrt(v float32) float32   { return float32(math.Sqrt(float64(v))) }
func pow(x, y float32) float32 { return float32(math.Pow(float64(x), float64(y))) }

func abs(v float32) float32 {
	if v < 0 {
		return -v
	}
	return v
}

func sign(v float32) float32 {
	switch {
	case v > 0:
		return 1
	case v < 0:
		return -1
	}
	return 0
}
