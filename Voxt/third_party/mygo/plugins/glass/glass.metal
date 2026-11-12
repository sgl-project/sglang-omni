// The glass's effect in Metal Shading Language (see scene.Effect), as
// pixels.go draws it on the CPU. The parameters are those of material:
// p0 the bezel, the refraction, the rim and its width; p1 the tint; p2 the
// low, high, curve and saturation; p3 the light's direction; p4 the tint in
// extended sRGB, for a target that keeps colors outside the sRGB gamut.

// glassLens returns how far, as a fraction of the refraction, the glass
// bends what shows through it t pixels inside its edge (lens).
float glassLens(float t, float bezel) {
	if (bezel <= 0.0f || t >= bezel) {
		return 0.0f;
	}
	float u = 1.0f - max(t, 0.0f) / bezel;
	float v = 1.0f - u * u * u * u;
	float a = sqrt(v * sqrt(v));
	float b = u * u * u;
	float n = sqrt(a * a + b * b);
	float s = b / n;
	float c = a / n;
	float st = s / 1.5f;
	float ct = sqrt(1.0f - st * st);
	return (s * ct - c * st) / (c * ct + s * st) / 1.1180340f;
}

// glassNormal returns the direction of the edge's normal at p, smoothed
// (normal).
float2 glassNormal(float2 p, float4 rect, float4 radii) {
	float2 h = rect.zw * 0.5f;
	float2 q = p - rect.xy - h;
	float r = q.x < 0.0f ? (q.y < 0.0f ? radii.x : radii.w) : (q.y < 0.0f ? radii.y : radii.z);
	r = min(r * 1.5f, min(h.x, h.y));
	float2 a = abs(q) - (h - r);
	if (a.x > 0.0f || a.y > 0.0f) {
		float2 m = max(a, float2(0.0f));
		return sign(q) * m / sqrt(m.x * m.x + m.y * m.y);
	}
	return a.x > a.y ? float2(sign(q.x), 0.0f) : float2(0.0f, sign(q.y));
}

// glassTone maps the color behind the glass to the glass's (tone).
float3 glassTone(float3 c, float4 tone) {
	float l = 0.2126f * c.r + 0.7152f * c.g + 0.0722f * c.b;
	float t = tone.x + (tone.y - tone.x) * (1.0f - pow(max(1.0f - l, 0.0f), tone.z));
	float k = (tone.y - tone.x) * tone.w;
	return clamp(t + (c - l) * k, 0.0f, 1.0f);
}

float4 effect(float2 p, Effect e, texture2d<float> backdrop) {
	float t = max(-sdRoundRect(p, e.rect, e.radii), 0.0f);
	float bezel = e.p0.x;
	float rimWidth = e.p0.w;
	float2 n = float2(0.0f);
	if (t < bezel || t < 3.0f * rimWidth) {
		n = glassNormal(p, e.rect, e.radii);
	}
	float2 q = p;
	if (t < bezel) {
		q -= n * (e.p0.y * glassLens(t, bezel));
	}
	float3 c = glassTone(sampleBackdrop(backdrop, e, q), e.p2);
	float4 tint = e.wide ? e.p4 : e.p1;
	c += (tint.rgb - c) * tint.a;
	if (e.p0.z > 0.0f && t < 3.0f * rimWidth) {
		float r = t / rimWidth;
		float a = e.p0.z * exp(-r * r);
		float l = abs(n.x * e.p3.x + n.y * e.p3.y);
		c += (l - c) * a;
	}
	return float4(c, 1.0f);
}
