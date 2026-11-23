package gputest

import "github.com/egoist/mygo/internal/scene"

// Effects for the renderers' tests, as plugins define theirs (see
// scene.Effect). LensEffect shows its backdrop, sampled further right
// within 8 pixels of its edge, by p0.x pixels a pixel, mixed with the
// straight color p1 by its alpha; TintEffect paints p1 alone, reading no
// backdrop.
var (
	LensEffect = &scene.Effect{
		Name:     "lens",
		Backdrop: true,
		Metal: scene.EffectCode{Source: `
float4 effect(float2 p, Effect e, texture2d<float> backdrop) {
	float d = max(8.0f + sdRoundRect(p, e.rect, e.radii), 0.0f) * e.p0.x;
	float3 c = sampleBackdrop(backdrop, e, p + float2(d, 0.0f));
	c += (e.p1.rgb - c) * e.p1.a;
	return float4(c, 1.0f);
}`},
		HLSL: scene.EffectCode{Source: `
float4 effect(float2 p, Effect e) {
	float d = max(8 + sdRoundRect(p, e.rect, e.radii), 0) * e.p0.x;
	float3 c = sampleBackdrop(e, p + float2(d, 0));
	c += (e.p1.rgb - c) * e.p1.a;
	return float4(c, 1);
}`},
		GLSL: `
vec4 effect(vec2 p, Effect e) {
	float d = max(8.0 + sdRoundRect(p, e.rect, e.radii), 0.0) * e.p0.x;
	vec3 c = sampleBackdrop(e, p + vec2(d, 0.0));
	c += (e.p1.rgb - c) * e.p1.a;
	return vec4(c, 1.0);
}`,
		Pixels: func() scene.EffectPixels { return &testPixels{} },
	}
	TintEffect = &scene.Effect{
		Name: "tint",
		Metal: scene.EffectCode{Source: `
float4 effect(float2 p, Effect e, texture2d<float> backdrop) {
	return float4(e.p1.rgb * e.p1.a, e.p1.a);
}`},
		HLSL: scene.EffectCode{Source: `
float4 effect(float2 p, Effect e) {
	return float4(e.p1.rgb * e.p1.a, e.p1.a);
}`},
		GLSL: `
vec4 effect(vec2 p, Effect e) {
	return vec4(e.p1.rgb * e.p1.a, e.p1.a);
}`,
		Pixels: func() scene.EffectPixels { return &testPixels{} },
	}
)

// testPixels draws LensEffect and TintEffect on the CPU.
type testPixels struct {
	op    *scene.EffectOp
	rect  scene.Rect
	radii [4]float32
}

func (t *testPixels) Begin(op *scene.EffectOp, rect scene.Rect, radii [4]float32) {
	t.op, t.rect, t.radii = op, rect, radii
}

func (t *testPixels) Color(x, y float32, b *scene.BackdropImage) [4]float32 {
	tint := t.op.Params[1]
	if b == nil {
		return [4]float32{tint[0] * tint[3], tint[1] * tint[3], tint[2] * tint[3], tint[3]}
	}
	d := max(8+scene.SDRoundRect(t.rect, t.radii, x, y), 0) * t.op.Params[0][0]
	c := b.Sample(x+d, y)
	for i := range c {
		c[i] += (tint[i] - c[i]) * tint[3]
	}
	return [4]float32{c[0], c[1], c[2], 1}
}
