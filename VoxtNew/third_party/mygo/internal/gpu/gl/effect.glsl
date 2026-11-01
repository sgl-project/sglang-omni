// The head and the tail of an effect's fragment shader (scene.Effect):
// after shader.glsl with EFFECT defined, the head declares what an effect
// reads, the effect's source follows, then the tail, after the line
// "// effect", draws its instances with it.

uniform sampler2D uBackdrop;

// What an effect reads of its instance.
struct Effect {
	vec4 rect;	// x, y, width, height in pixels
	vec4 radii; // top-left, top-right, bottom-right, bottom-left, circular
	vec4 p0, p1, p2, p3, p4;
	vec4 area;	// where the backdrop's area starts in the frame, and its size in texels
	float down; // the size of the squares the backdrop averages
	vec4 transform0, transform1, transform2;
};

vec2 framePoint(Effect e, vec2 q) {
	vec3 h = vec3(dot(e.transform0.xyz, vec3(q, 1.0)), dot(e.transform1.xyz, vec3(q, 1.0)),
				  dot(vec3(e.transform2.xy, 1.0), vec3(q, 1.0)));
	return h.xy / h.z;
}

vec3 backdropAt(ivec2 p, ivec2 size) { return texelFetch(uBackdrop, clamp(p, ivec2(0), size - 1), 0).rgb; }

// sampleBackdrop returns the backdrop at q, in the frame's pixels,
// premultiplied, filtered bilinearly from its texels as
// scene.BackdropImage.Sample does.
vec3 sampleBackdrop(Effect e, vec2 q) {
	q = framePoint(e, q);
	vec2 u = (q - e.area.xy) / e.down - 0.5;
	vec2 f = floor(u);
	vec2 w = u - f;
	ivec2 p = ivec2(f);
	ivec2 size = ivec2(e.area.zw);
	vec3 top = backdropAt(p, size) * (1.0 - w.x) + backdropAt(p + ivec2(1, 0), size) * w.x;
	vec3 bot = backdropAt(p + ivec2(0, 1), size) * (1.0 - w.x) + backdropAt(p + ivec2(1, 1), size) * w.x;
	return top * (1.0 - w.y) + bot * w.y;
}

// effect

void main() {
	Frame fr = frameOf();
	vec2 p = vPoint.xy;
	Effect e = Effect(vRect, abs(vRadii), vInner, vColor, vColor2, vBorder, vGrad, vWidths, vParams.z,
					  vTransform0, vTransform1, vTransform2);
	if (vParams.y > 0.5) {
		// A mask glyph a shader fills (scene.EffectOp.Glyphs), as the
		// renderer's own shader draws one, its paint the shader's color.
		vec4 c = effect(p, e);
		float glyphCov = vTransform0.w > 0.5 ? localCoverage(p, vRect, vec4(0.0), 1.0, fr) : 1.0;
		fragColor = c * (textCoverage(sampleLocal(uMask, vPoint.zw, vWidths, vTransform0.w).r, unpremul(c), vInner.x,
									  vInner.y, vRadii) *
						 glyphCov * clipCoverage() * vParams.w);
	} else {
		fragColor = effect(p, e) * (localCoverage(p, vRect, vRadii, vTransform0.w, fr) * clipCoverage() * vParams.w);
	}
#ifdef DUAL
	fragAlpha = fragColor.aaaa;
#endif
}
