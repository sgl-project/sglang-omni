// The head and the tail of an effect's pixel shader (scene.Effect): after
// shader.hlsl with EFFECT defined, the head declares what an effect reads,
// the effect's source follows, then the tail, after the line "// effect",
// draws its instances with it (effectps): the rounded rectangle, or, with
// 1 in params.y, a mask glyph a shader's effect fills.

Texture2D backdropTex : register(t3);

// What an effect reads of its instance.
struct Effect {
	float4 rect;  // x, y, width, height in pixels
	float4 radii; // top-left, top-right, bottom-right, bottom-left, circular
	float4 p0, p1, p2, p3, p4;
	float4 area; // where the backdrop's area starts in the frame, and its size in texels
	float down;	 // the size of the squares the backdrop averages
	float4 transform0, transform1, transform2;
};

float2 framePoint(Effect e, float2 q) {
	float3 h = float3(dot(e.transform0.xyz, float3(q, 1.0)), dot(e.transform1.xyz, float3(q, 1.0)),
					  dot(float3(e.transform2.xy, 1.0), float3(q, 1.0)));
	return h.xy / h.z;
}

float3 backdropAt(int2 p, int2 size) { return backdropTex.Load(int3(clamp(p, int2(0, 0), size - 1), 0)).rgb; }

// sampleBackdrop returns the backdrop at q, in the frame's pixels,
// premultiplied, filtered bilinearly from its texels as
// scene.BackdropImage.Sample does.
float3 sampleBackdrop(Effect e, float2 q) {
	q = framePoint(e, q);
	float2 u = (q - e.area.xy) / e.down - 0.5;
	float2 f = floor(u);
	float2 w = u - f;
	int2 p = int2(f);
	int2 size = int2(e.area.zw);
	float3 top = backdropAt(p, size) * (1 - w.x) + backdropAt(p + int2(1, 0), size) * w.x;
	float3 bot = backdropAt(p + int2(0, 1), size) * (1 - w.x) + backdropAt(p + int2(1, 1), size) * w.x;
	return top * (1 - w.y) + bot * w.y;
}

// effect

PSOut effectps(VSOut i) {
	Frame fr = frameOf(i);
	Effect e;
	e.rect = i.rect;
	e.radii = abs(i.radii);
	e.p0 = i.inner;
	e.p1 = i.color;
	e.p2 = i.color2;
	e.p3 = i.border;
	e.p4 = i.grad;
	e.area = i.widths;
	e.down = i.params.z;
	e.transform0 = i.transform0;
	e.transform1 = i.transform1;
	e.transform2 = i.transform2;
	float4 res;
	if (i.params.y > 0.5) {
		// A mask glyph a shader fills (scene.EffectOp.Glyphs), as ps
		// draws one, its paint the shader's color.
		float4 c = effect(i.pt.xy, e);
		float glyphCov = i.transform0.w > 0.5 ? localCoverage(i.pt.xy, i.rect, float4(0, 0, 0, 0), 1, fr) : 1;
		res = c * (textCoverage(sampleLocal(maskTex, i.pt.zw, i.widths, i.transform0.w).r, unpremul(c), i.inner.x,
								i.inner.y, i.radii) *
				   glyphCov * clipCoverage(i) * i.params.w);
	} else {
		res = effect(i.pt.xy, e) *
			  (localCoverage(i.pt.xy, i.rect, i.radii, i.transform0.w, fr) * clipCoverage(i) * i.params.w);
	}
	PSOut o;
	o.color = res;
	o.alpha = res.aaaa;
	return o;
}
