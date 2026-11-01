// A level of the blur's effect in HLSL (see scene.Effect), as blur.metal
// is in Metal Shading Language and blur.go draws it on the CPU.

float4 blurAt(int2 p, int2 size) { return backdropTex.Load(int3(clamp(p, int2(0, 0), size - 1), 0)); }

float4 blurSample(Effect e, float2 q) {
	q = framePoint(e, q);
	float2 u = (q - e.area.xy) / e.down - 0.5;
	float2 f = floor(u);
	float2 w = u - f;
	int2 p = int2(f);
	int2 size = int2(e.area.zw);
	float4 top = blurAt(p, size) * (1 - w.x) + blurAt(p + int2(1, 0), size) * w.x;
	float4 bot = blurAt(p + int2(0, 1), size) * (1 - w.x) + blurAt(p + int2(1, 1), size) * w.x;
	return top * (1 - w.y) + bot * w.y;
}

float4 effect(float2 p, Effect e) {
	float2 d = e.p0.zw - e.p0.xy;
	float l = d.x * d.x + d.y * d.y;
	float t = 0;
	if (l > 0) {
		t = saturate(((p.x - e.p0.x) * d.x + (p.y - e.p0.y) * d.y) / max(l, 0.0001));
	}
	float blur = e.p1.x * (1 - t) + e.p1.y * t;
	float w = saturate((blur - e.p1.z) / (e.p1.w - e.p1.z));
	if (w <= 0) {
		return float4(0, 0, 0, 0);
	}
	float4 c = blurSample(e, p);
	float lum = 0.2126 * c.r + 0.7152 * c.g + 0.0722 * c.b;
	c.rgb = clamp(lum + (c.rgb - lum) * (1 + e.p2.x) + e.p2.y * c.a, 0, c.a);
	c += (float4(e.p3.rgb, 1) - c) * e.p2.z;
	return c * w;
}
