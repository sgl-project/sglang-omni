// The tail of a shader's library (package ui/shader), after the head of
// effect.metal and the shader's code, which ends with shaderMain: Metal
// has no global variables, so a shader's uniforms (buffer 2) and images
// (textures 5 to 8) come to the fragment function, which passes them on.
// Instances with 1 in params.y are mask glyphs it fills, read from the
// mask atlas (texture 0).

fragment PSOut shader_ps(VSOut v [[stage_in]], const device Inst *insts [[buffer(0)]],
						 constant float4 &globals [[buffer(1)]], constant Uniforms &uniforms [[buffer(2)]],
						 texture2d<float> T0 [[texture(5)]], texture2d<float> T1 [[texture(6)]],
						 texture2d<float> T2 [[texture(7)]], texture2d<float> T3 [[texture(8)]],
						 texture2d<float> clipTex [[texture(4)]], texture2d<float> maskTex [[texture(0)]],
						 sampler samp [[sampler(0)]]) {
	Inst i = insts[v.inst];
	Frame fr = frameOf(v, i);
	if (i.params.y > 0.5f) {
		// A mask glyph the shader fills (scene.EffectOp.Glyphs), as ps
		// draws one, its paint the shader's color, its positions from
		// the box's corner in i.grad.
		float4 c = shaderMain(uniforms, T0, T1, T2, T3, v.p, float4(i.grad.xy, i.rect.zw));
		float glyphCov = i.transform0.w > 0.5f ? localCoverage(v.p, i.rect, float4(0.0f), 1.0f, fr) : 1.0f;
		float4 res = c *
					 (textCoverage(sampleLocal(maskTex, samp, v.tex, i.uv, i.transform0.w).r, unpremul(c), i.inner.x,
								   i.inner.y, i.radii) *
					  glyphCov * clipCoverage(v, i, clipTex, samp) * i.params.w);
		return PSOut{res, res.aaaa};
	}
	float4 res = shaderMain(uniforms, T0, T1, T2, T3, v.p, i.rect) *
				 (localCoverage(v.p, i.rect, i.radii, i.transform0.w, fr) * clipCoverage(v, i, clipTex, samp) * i.params.w);
	return PSOut{res, res.aaaa};
}
