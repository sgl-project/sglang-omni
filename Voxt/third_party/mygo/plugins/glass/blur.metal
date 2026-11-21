// A level of the blur's effect in Metal Shading Language (see
// scene.Effect and blurLevel), as blur.go draws it on the CPU. p0 is the
// mask's line, from its start to its end in pixels; p1 the blur at its
// start and at its end, and the level's band, from p1.z to p1.w; p2 and p3
// its tone (blurTone): how much more saturated, the offset and how much of
// the color p3 over it.

// blurAt returns the backdrop's texel p, with its alpha, clamped to its
// edges.
float4 blurAt(texture2d<float> t, int2 p, int2 size) { return t.read(uint2(clamp(p, int2(0), size - 1))); }

// blurSample returns the backdrop at q with its alpha, as sampleBackdrop
// does its color (sampleRGBA).
float4 blurSample(texture2d<float> t, Effect e, float2 q) {
	q = framePoint(e, q);
	float2 u = (q - e.area.xy) / e.down - 0.5f;
	float2 f = floor(u);
	float2 w = u - f;
	int2 p = int2(f);
	int2 size = int2(e.area.zw);
	float4 top = blurAt(t, p, size) * (1.0f - w.x) + blurAt(t, p + int2(1, 0), size) * w.x;
	float4 bot = blurAt(t, p + int2(0, 1), size) * (1.0f - w.x) + blurAt(t, p + int2(1, 1), size) * w.x;
	return top * (1.0f - w.y) + bot * w.y;
}

float4 effect(float2 p, Effect e, texture2d<float> backdrop) {
	float2 d = e.p0.zw - e.p0.xy;
	float l = d.x * d.x + d.y * d.y;
	float t = 0.0f;
	if (l > 0.0f) {
		t = clamp(((p.x - e.p0.x) * d.x + (p.y - e.p0.y) * d.y) / max(l, 0.0001f), 0.0f, 1.0f);
	}
	float blur = e.p1.x * (1.0f - t) + e.p1.y * t;
	float w = clamp((blur - e.p1.z) / (e.p1.w - e.p1.z), 0.0f, 1.0f);
	if (w <= 0.0f) {
		return float4(0.0f);
	}
	float4 c = blurSample(backdrop, e, p);
	float lum = 0.2126f * c.r + 0.7152f * c.g + 0.0722f * c.b;
	c.rgb = clamp(lum + (c.rgb - lum) * (1.0f + e.p2.x) + e.p2.y * c.a, 0.0f, c.a);
	c += (float4(e.p3.rgb, 1.0f) - c) * e.p2.z;
	return c * w;
}
