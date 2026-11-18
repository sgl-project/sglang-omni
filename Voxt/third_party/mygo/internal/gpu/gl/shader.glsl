// The one shader of the OpenGL renderer, as shader.metal is the Metal one
// and shader.hlsl the Direct3D one: every scene op is an instanced quad,
// and the fragment shader computes the coverage of rounded rectangles,
// borders, gradients, stripes and shadows from signed distances, as the
// CPU renderer (internal/raster) does. Colors are straight (not premultiplied)
// and blending happens in sRGB space, as in browsers. The renderer
// compiles it when it starts, as GLSL 3.30 or GLSL ES 3.00, with VERTEX or
// FRAGMENT defined, and the passes computing the backdrops of effects with
// PASS_VERTEX and DOWN or BLUR. The fragment shaders of effects
// (scene.Effect) take the start of its fragment part, with EFFECT
// defined, before effect.glsl.

#ifdef VERTEX

// One instance per op, as internal/gpu builds them.
layout(location = 0) in vec4 aRect;      // x, y, width, height in pixels
layout(location = 1) in vec4 aRadii;     // top-left, top-right, bottom-right, bottom-left
layout(location = 2) in vec4 aInner;     // radii of the border's inner edge, of the box casting a shadow, or of an inner shadow's hole
layout(location = 3) in vec4 aColor;
layout(location = 4) in vec4 aColor2;    // gradient end
layout(location = 5) in vec4 aBorder;    // border color
layout(location = 6) in vec4 aGrad;      // gradient start and end points, or stripes
layout(location = 7) in vec4 aUV;        // texture rectangle, normalized, border widths, the box casting a shadow, or an inner shadow's hole
layout(location = 8) in vec4 aClip;      // the innermost clip rectangle
layout(location = 9) in vec4 aClipRadii;
layout(location = 10) in vec4 aParams; // kind, dashed or grayscale, sigma or paint, opacity

layout(location = 11) in vec4 aTransform0;
layout(location = 12) in vec4 aTransform1;
layout(location = 13) in vec4 aTransform2; // the projective row's g and h

uniform vec2 uSize;

// The frame point times its projective divisor, and the divisor: they
// interpolate in perspective to what divides into the frame point.
out vec3 vWorld;
flat out vec4 vTransform0;
flat out vec4 vTransform1;
flat out vec4 vTransform2;
out vec4 vPoint; // the position in pixels, and the texture coordinates
flat out vec4 vRect;
flat out vec4 vRadii;
flat out vec4 vInner;
flat out vec4 vColor;
flat out vec4 vColor2;
flat out vec4 vBorder;
flat out vec4 vGrad;
flat out vec4 vWidths;
flat out vec4 vClip;
flat out vec4 vClipRadii;
flat out vec4 vParams;

// frameH returns the frame point of local p, in homogeneous coordinates.
vec3 frameH(vec2 p) {
	return vec3(dot(aTransform0.xyz, vec3(p, 1.0)), dot(aTransform1.xyz, vec3(p, 1.0)),
				dot(vec3(aTransform2.xy, 1.0), vec3(p, 1.0)));
}

void main() {
	vec2 corner = vec2(float(gl_VertexID & 1), float(gl_VertexID >> 1));
	vec4 r = aRect;
	float kind = aParams.x;
	if (kind < 0.5 || kind > 5.5) { // fills and effects
		r = vec4(r.xy - 1.0, r.zw + 2.0);
	} else if (kind < 1.5) {
		float e = 3.0 * aParams.z + 1.0;
		r = vec4(r.xy - e, r.zw + 2.0 * e);
	}
	vec3 h = frameH(r.xy + corner * r.zw);
	if (aTransform0.w > 0.5 && h.z > 0.0) {
		// A pixel's reach at this corner, by the map's Jacobian there.
		vec2 s = h.xy / h.z;
		float a = (aTransform0.x - s.x * aTransform2.x) / h.z, c = (aTransform0.y - s.x * aTransform2.y) / h.z;
		float b = (aTransform1.x - s.y * aTransform2.x) / h.z, d = (aTransform1.y - s.y * aTransform2.y) / h.z;
		vec2 e = vec2(abs(d) + abs(c), abs(b) + abs(a)) / abs(a * d - b * c);
		r = vec4(r.xy - e, r.zw + 2.0 * e);
	}
	vec2 p = r.xy + corner * r.zw;
	h = frameH(p);
	gl_Position = vec4(h.xy / uSize * vec2(2.0, -2.0) + h.z * vec2(-1.0, 1.0), 0.0, h.z);
	vPoint = vec4(p, aUV.xy + (p - aRect.xy) / aRect.zw * (aUV.zw - aUV.xy));
	vWorld = h;
	vTransform0 = aTransform0;
	vTransform1 = aTransform1;
	vTransform2 = aTransform2;
	vRect = aRect;
	vRadii = aRadii;
	vInner = aInner;
	vColor = aColor;
	vColor2 = aColor2;
	vBorder = aBorder;
	vGrad = aGrad;
	vWidths = aUV;
	vClip = aClip;
	vClipRadii = aClipRadii;
	vParams = aParams;
}

#endif

#if defined(FRAGMENT) || defined(EFFECT)

in vec3 vWorld;
flat in vec4 vTransform0;
flat in vec4 vTransform1;
flat in vec4 vTransform2;
in vec4 vPoint;
flat in vec4 vRect;
flat in vec4 vRadii;
flat in vec4 vInner;
flat in vec4 vColor;
flat in vec4 vColor2;
flat in vec4 vBorder;
flat in vec4 vGrad;
flat in vec4 vWidths;
flat in vec4 vClip;
flat in vec4 vClipRadii;
flat in vec4 vParams;

uniform sampler2D uMask;
uniform sampler2D uColor;
uniform sampler2D uImage;
uniform sampler2D uClipMask;
// The stops of the frame's gradients (gpu.Builder.Stops): two texels each,
// the color, straight sRGB, then its place and its color in Oklab.
uniform highp sampler2D uStops;
uniform vec2 uSize; // the frame's size in pixels

// The color and, for dual-source blending, the source's alpha of each
// channel. Without it, subpixel glyphs take the mean of their subpixels.
#ifdef DUAL
layout(location = 0, index = 0) out vec4 fragColor;
layout(location = 0, index = 1) out vec4 fragAlpha;
#else
out vec4 fragColor;
#endif

float sdRoundRect(vec2 p, vec4 rect, vec4 radii) {
	vec2 h = rect.zw * 0.5;
	vec2 q = p - rect.xy - h;
	float r = q.x < 0.0 ? (q.y < 0.0 ? radii.x : radii.w) : (q.y < 0.0 ? radii.y : radii.z);
	vec2 a = abs(q) - h + r;
	return length(max(a, vec2(0.0))) + min(max(a.x, a.y), 0.0) - r;
}

float coverage(float d) { return clamp(0.5 - d, 0.0, 1.0); }

// rectCoverage returns how much of the pixel at p a rounded rectangle
// covers: by the distance to its edge near rounded corners, and exactly,
// the area of the pixel inside it, near square ones.
float rectCoverage(vec2 p, vec4 rect, vec4 radii) {
	vec2 q = p - rect.xy - rect.zw * 0.5;
	float r = q.x < 0.0 ? (q.y < 0.0 ? radii.x : radii.w) : (q.y < 0.0 ? radii.y : radii.z);
	if (r > 0.0) {
		return coverage(sdRoundRect(p, rect, radii));
	}
	vec2 c = clamp(min(rect.xy + rect.zw, p + 0.5) - max(rect.xy, p - 0.5), 0.0, 1.0);
	return c.x * c.y;
}

// Frame is where a fragment is in the frame, s, and, when its instance's
// map is projective (persp), the columns of the map back, up to a factor.
struct Frame {
	vec2 s;
	vec3 c0, c1, c2;
	bool persp;
};
Frame frameOf() {
	Frame f;
	f.persp = any(notEqual(vTransform2.xy, vec2(0.0)));
	// In perspective, the pixel's center, as interpolating drifts: the
	// frame's y goes down, the window's up.
	f.s = f.persp ? vec2(gl_FragCoord.x, uSize.y - gl_FragCoord.y) : vWorld.xy / vWorld.z;
	vec3 r0 = vTransform0.xyz, r1 = vTransform1.xyz, r2 = vec3(vTransform2.xy, 1.0);
	f.c0 = cross(r1, r2);
	f.c1 = cross(r2, r0);
	f.c2 = cross(r0, r1);
	return f;
}
// localOf returns the point of the instance that shows at the frame's s.
vec2 localOf(Frame f, vec2 s) {
	vec3 q = f.c0 * s.x + f.c1 * s.y + f.c2;
	return q.xy / q.z;
}
// perspectiveCoverage is localCoverage in perspective: each sample where
// the CPU renderer takes it.
float perspectiveCoverage(vec4 rect, vec4 radii, Frame fr) {
	vec2 p = localOf(fr, fr.s);
	vec2 a = localOf(fr, fr.s + vec2(0.5, 0.5)) - p, b = localOf(fr, fr.s + vec2(0.5, -0.5)) - p;
	float reach = max(abs(a.x) + abs(a.y), abs(b.x) + abs(b.y)) * 2.0;
	float d = sdRoundRect(p, rect, radii);
	if (d < -reach)
		return 1.0;
	if (d > reach)
		return 0.0;
	float n = 0.0;
	for (int y = 0; y < 4; y++)
		for (int x = 0; x < 4; x++) {
			vec2 q = localOf(fr, fr.s + vec2((float(x) + 0.5) / 4.0 - 0.5, (float(y) + 0.5) / 4.0 - 0.5));
			n += sdRoundRect(q, rect, radii) <= 0.0 ? 1.0 : 0.0;
		}
	return n / 16.0;
}
float localCoverage(vec2 p, vec4 rect, vec4 radii, float affine, Frame fr) {
	if (affine < 0.5)
		return rectCoverage(p, rect, radii);
	vec2 dx = dFdx(p), dy = dFdy(p), a = (dx + dy) * 0.5, b = (dx - dy) * 0.5;
	if (fr.persp)
		return perspectiveCoverage(rect, radii, fr);
	float reach = max(abs(a.x) + abs(a.y), abs(b.x) + abs(b.y)) * 2.0;
	float d = sdRoundRect(p, rect, radii);
	if (d < -reach)
		return 1.0;
	if (d > reach)
		return 0.0;
	float n = 0.0;
	for (int y = 0; y < 4; y++)
		for (int x = 0; x < 4; x++) {
			vec2 q = p + dx * ((float(x) + 0.5) / 4.0 - 0.5) + dy * ((float(y) + 0.5) / 4.0 - 0.5);
			n += sdRoundRect(q, rect, radii) <= 0.0 ? 1.0 : 0.0;
		}
	return n / 16.0;
}
float clipCoverage() {
	Frame fr = frameOf();
	vec2 world = fr.s;
	if (vTransform1.w < 0.5)
		return rectCoverage(world, vClip, vClipRadii);
	if (vTransform1.w > 1.5)
		return localCoverage(vPoint.xy, vClip, vec4(0.0), vTransform0.w, fr) *
			   texture(uClipMask, (world - vClipRadii.xy) / vClipRadii.zw).a;
	return texture(uClipMask, (world - vClip.xy) / vClip.zw).a;
}
vec4 sampleLocal(sampler2D t, vec2 uv, vec4 src, float affine) {
	if (affine > 0.5) {
		vec2 halfPixel = 0.5 / vec2(textureSize(t, 0));
		uv = clamp(uv, src.xy + halfPixel, src.zw - halfPixel);
	}
	return texture(t, uv);
}

// sampleImage samples an image held inside src, which is reversed along
// an axis it flips, so that a part of a sheet shows none of its
// neighbors; where pixelated is set, as pixel art draws, each texel a
// square of its color whose edges blend with the next only across the
// pixel they cross (Ebitengine's FilterPixelated): nearest where it shows
// at whole multiples of its size, and texels of even widths with smooth
// edges at any other. dtex is the fwidth of uv, taken where every
// fragment of the quad takes it.
vec4 sampleImage(sampler2D t, vec2 uv, vec4 src, float pixelated, vec2 dtex) {
	vec2 size = vec2(textureSize(t, 0));
	vec2 lo = min(src.xy, src.zw), hi = max(src.xy, src.zw);
	if (pixelated > 0.5) {
		vec2 tc = uv * size - 0.5;
		vec2 i = floor(tc);
		vec2 f = clamp((tc - i - 0.5) / clamp(dtex * size, 1e-5, 1.0) + 0.5, 0.0, 1.0);
		uv = (i + f + 0.5) / size;
	}
	return texture(t, clamp(uv, lo + 0.5 / size, hi - 0.5 / size));
}

// tint multiplies premultiplied c by straight color t.
vec4 tint(vec4 c, vec4 t) { return c * vec4(t.rgb * t.a, t.a); }

vec4 premul(vec4 c) { return vec4(c.rgb * c.a, c.a); }

vec3 unpremul(vec4 c) { return c.a > 0.0 ? c.rgb / c.a : vec3(0.0); }

// textCoverage corrects the coverage a of a glyph of straight color c as
// Direct2D blends text (scene.TextCoverage): it enhances the contrast,
// the more the darker c is, and corrects for gamma with the ratios g.
float textCoverage(float a, vec3 c, float contrast, float boost, vec4 g) {
	float k = contrast * clamp(3.0 - 4.0 * dot(c, vec3(0.30, 0.59, 0.11)), 0.0, 1.0) + boost;
	a = a * (k + 1.0) / (a * k + 1.0);
	float f = dot(c, vec3(0.25, 0.5, 0.25));
	return clamp(a + a * (1.0 - a) * ((g.x * f + g.y) * a + (g.z * f + g.w)), 0.0, 1.0);
}

#endif

// What follows is the renderer's own; effects (effect.glsl) take what is
// above.
#ifdef FRAGMENT

// subpixelCoverage does the same for each subpixel of a subpixel glyph.
vec3 subpixelCoverage(vec3 a, vec3 c, float contrast, float boost, vec4 g) {
	float k = contrast * clamp(3.0 - 4.0 * dot(c, vec3(0.30, 0.59, 0.11)), 0.0, 1.0) + boost;
	a = a * (k + 1.0) / (a * k + 1.0);
	return clamp(a + a * (1.0 - a) * ((g.x * c + g.y) * a + (g.z * c + g.w)), vec3(0.0), vec3(1.0));
}

// erf2 approximates the error function (Abramowitz and Stegun 7.1.27).
vec2 erf2(vec2 x) {
	vec2 s = sign(x);
	vec2 a = abs(x);
	x = 1.0 + (0.278393 + (0.230389 + 0.078108 * (a * a)) * a) * a;
	x *= x;
	return s - s / (x * x);
}

float gaussian(float x, float sigma) {
	return exp(-(x * x) / (2.0 * sigma * sigma)) / (2.50662827463 * sigma);
}

// The blurred rounded box of Evan Wallace: exact along x, four samples
// along y.
float shadowX(float x, float y, float sigma, float corner, vec2 h) {
	float delta = min(h.y - corner - abs(y), 0.0);
	float curved = h.x - corner + sqrt(max(0.0, corner * corner - delta * delta));
	vec2 integral = 0.5 + 0.5 * erf2((x + vec2(-curved, curved)) * (sqrt(0.5) / sigma));
	return integral.y - integral.x;
}

float boxShadow(vec2 p, vec4 rect, float sigma, float corner) {
	vec2 h = rect.zw * 0.5;
	p -= rect.xy + h;
	float low = p.y - h.y;
	float high = p.y + h.y;
	float y0 = clamp(-3.0 * sigma, low, high);
	float y1 = clamp(3.0 * sigma, low, high);
	float dy = (y1 - y0) / 4.0;
	float y = y0 + dy * 0.5;
	float v = 0.0;
	for (int k = 0; k < 4; k++) {
		v += shadowX(p.x, p.y - y, sigma, corner, h) * gaussian(y, sigma) * dy;
		y += dy;
	}
	return v;
}

vec3 toLinear(vec3 c) {
	return mix(pow((c + 0.055) / 1.055, vec3(2.4)), c / 12.92, lessThanEqual(c, vec3(0.04045)));
}

vec3 toSRGB(vec3 c) {
	return mix(1.055 * pow(max(c, vec3(0.0)), vec3(1.0 / 2.4)) - 0.055, c * 12.92,
			   lessThanEqual(c, vec3(0.0031308)));
}

vec3 cbrt3(vec3 v) { return sign(v) * pow(abs(v), vec3(1.0 / 3.0)); }

vec3 oklab(vec3 srgb) {
	vec3 c = toLinear(srgb);
	vec3 lms = cbrt3(vec3(0.4122214708 * c.r + 0.5363325363 * c.g + 0.0514459929 * c.b,
						  0.2119034982 * c.r + 0.6806995451 * c.g + 0.1073969566 * c.b,
						  0.0883024619 * c.r + 0.2817188376 * c.g + 0.6299787005 * c.b));
	return vec3(0.2104542553 * lms.x + 0.7936177850 * lms.y - 0.0040720468 * lms.z,
				1.9779984951 * lms.x - 2.4285922050 * lms.y + 0.4505937099 * lms.z,
				0.0259040371 * lms.x + 0.7827717662 * lms.y - 0.8086757660 * lms.z);
}

vec3 fromOklab(vec3 lab) {
	vec3 lms = vec3(lab.x + 0.3963377774 * lab.y + 0.2158037573 * lab.z,
					lab.x - 0.1055613458 * lab.y - 0.0638541728 * lab.z,
					lab.x - 0.0894841775 * lab.y - 1.2914855480 * lab.z);
	lms = lms * lms * lms;
	return toSRGB(vec3(4.0767416621 * lms.x - 3.3077115913 * lms.y + 0.2309699292 * lms.z,
					   -1.2684380046 * lms.x + 2.6097574011 * lms.y - 0.3413193965 * lms.z,
					   -0.0041960863 * lms.x - 0.7034186147 * lms.y + 1.7076147010 * lms.z));
}

// stop returns texel k of the stops.
vec4 stop(int k) {
	return texelFetch(uStops, ivec2(k % 1024, k / 1024), 0);
}

// stopsAt returns the color at t of the gradient whose stops uStops holds
// from first, n of them (gpu.Builder.Stops): the first stop's before it,
// the last's after it, and between two the colors of both mixed,
// premultiplied, in sRGB, or in Oklab, as internal/raster's stopsAt.
vec4 stopsAt(vec4 c2, float t, bool lab) {
	int first = int(c2.x);
	int n = int(c2.y);
	if (n <= 0) {
		return vec4(0.0);
	}
	vec4 a = stop(2 * first);
	vec4 ap = stop(2 * first + 1);
	if (t <= ap.x) {
		return premul(a);
	}
	vec4 last = stop(2 * (first + n - 1));
	vec4 lastp = stop(2 * (first + n - 1) + 1);
	if (t >= lastp.x) {
		return premul(last);
	}
	vec4 b = last;
	vec4 bp = lastp;
	for (int i = 1; i < n - 1; i++) {
		vec4 c = stop(2 * (first + i));
		vec4 cp = stop(2 * (first + i) + 1);
		if (t < cp.x) {
			b = c;
			bp = cp;
			break;
		}
		a = c;
		ap = cp;
	}
	float f = clamp((t - ap.x) / max(bp.x - ap.x, 1e-6), 0.0, 1.0);
	float alpha = mix(a.a, b.a, f);
	if (!lab) {
		return vec4(mix(a.rgb * a.a, b.rgb * b.a, f), alpha);
	}
	if (alpha <= 0.0) {
		return vec4(0.0);
	}
	return vec4(clamp(fromOklab(mix(ap.yzw * a.a, bp.yzw * b.a, f) / alpha), 0.0, 1.0) * alpha, alpha);
}

// paint returns the premultiplied color at p of plain color, a gradient
// mixed in sRGB (1) or Oklab (2), stripes (3), or a gradient of stops,
// linear (4, 5) or conic (6, 7), mixed in sRGB or in Oklab, as
// scene.Paint says.
vec4 paint(vec2 p, float mode, vec4 rect, vec4 c1, vec4 c2, vec4 g) {
	if (mode < 0.5) {
		return premul(c1);
	}
	if (mode > 3.5) {
		float t;
		if (mode < 5.5) {
			vec2 d = g.zw - g.xy;
			t = dot(p - g.xy, d) / max(dot(d, d), 0.0001);
		} else {
			vec2 d = p - g.xy;
			t = (atan(d.x, -d.y) - g.z) * 0.15915494309189535;
			t -= floor(t);
		}
		return stopsAt(c2, t, mode > 4.5 && mode < 5.5 || mode > 6.5);
	}
	if (mode < 2.5) {
		vec2 d = g.zw - g.xy;
		float t = clamp(dot(p - g.xy, d) / max(dot(d, d), 0.0001), 0.0, 1.0);
		float a = mix(c1.a, c2.a, t);
		if (mode < 1.5) {
			return vec4(mix(c1.rgb * c1.a, c2.rgb * c2.a, t), a);
		}
		vec3 lab = mix(oklab(c1.rgb) * c1.a, oklab(c2.rgb) * c2.a, t);
		return a > 0.0 ? vec4(clamp(fromOklab(lab / a), 0.0, 1.0) * a, a) : vec4(0.0);
	}
	float s = dot(p - rect.xy, g.xy);
	float phase = s - g.w * floor(s / g.w);
	float cov = coverage(min(max(-phase, phase - g.z), g.w - phase));
	return premul(c1) * cov + premul(c2) * (1.0 - cov);
}

// dash returns how much of a dashed border shows at p: each side, which
// the pixel belongs to when it is nearest that side's edge in widths of
// its border, has an odd number of dashes and gaps of equal length, about
// three widths, starting and ending with a dash.
float dash(vec2 p, vec4 rect, vec4 w) {
	vec2 q = p - rect.xy;
	float dt = w.x > 0.0 ? q.y / w.x : 1e9;
	float dr = w.y > 0.0 ? (rect.z - q.x) / w.y : 1e9;
	float db = w.z > 0.0 ? (rect.w - q.y) / w.z : 1e9;
	float dl = w.w > 0.0 ? q.x / w.w : 1e9;
	float s;
	float len;
	float bw;
	if (dt <= dr && dt <= db && dt <= dl) {
		s = q.x;
		len = rect.z;
		bw = w.x;
	} else if (dr <= db && dr <= dl) {
		s = q.y;
		len = rect.w;
		bw = w.y;
	} else if (db <= dl) {
		s = rect.z - q.x;
		len = rect.z;
		bw = w.z;
	} else {
		s = rect.w - q.y;
		len = rect.w;
		bw = w.w;
	}
	// n - 1 is how many periods of a dash and a gap, six widths, fit in
	// len: GPUs divide less exactly than CPUs, and may count one too few
	// or too many where len is a whole number of periods, as sides often
	// are, so the count is checked by multiplying back.
	float n = max(1.0, floor((len / (3.0 * bw) + 1.0) * 0.5 + 0.5));
	float period = 6.0 * bw;
	n += n * period <= len ? 1.0 : 0.0;
	n -= (n - 1.0) * period > len ? 1.0 : 0.0;
	float seg = len / (2.0 * n - 1.0);
	float k = floor(s / seg);
	float f = s - k * seg;
	float edge = min(f, seg - f);
	return coverage(k - 2.0 * floor(k * 0.5) < 0.5 ? -edge : edge);
}

void main() {
	Frame fr = frameOf();
	vec2 p = vPoint.xy;
	vec2 tex = vPoint.zw;
	vec2 dtex = fwidth(tex);
	float kind = vParams.x;
	vec4 res;
	float glyphCov = vTransform0.w > 0.5 ? localCoverage(p, vRect, vec4(0.0), 1.0, fr) : 1.0;
	float clip = clipCoverage();
	// The dispatch below is a chain of ifs with early returns, not one else-if chain:
	// an else-if chain here fails to link in ANGLE's D3D11 backend on Windows (an
	// empty info log, and every WebGL context in the page is lost; Firefox's headless
	// build exits at the link). See https://issues.chromium.org/issues/573007971 and
	// https://bugzilla.mozilla.org/show_bug.cgi?id=2081240. The shape is load-bearing.
	if (kind < 0.5) {
		float outer = localCoverage(p, vRect, vRadii, vTransform0.w, fr);
		// vParams.y: 1 dashes the border, 2 paints it with the gradient (the
		// inside with plain color), 3 both.
		bool borderPaint = vParams.y > 1.5;
		vec4 pc = paint(p, vParams.z, vRect, vColor, vColor2, vGrad);
		res = (borderPaint ? premul(vColor) : pc) * outer;
		vec4 bw = vWidths; // top, right, bottom, left
		if (any(greaterThan(bw, vec4(0.0)))) {
			vec4 ir = vec4(vRect.xy + bw.wx, vRect.zw - bw.yz - bw.wx);
			float innerCov = (ir.z > 0.0 && ir.w > 0.0) ? localCoverage(p, ir, vInner, vTransform0.w, fr) : 0.0;
			float bc = clamp(outer - innerCov, 0.0, 1.0);
			float dashed = vParams.y - (borderPaint ? 2.0 : 0.0);
			if (dashed > 0.5) {
				bc *= dash(p, vRect, bw);
			}
			vec4 b = (borderPaint ? pc * vec4(vBorder.rgb * vBorder.a, vBorder.a) : premul(vBorder)) * bc;
			res = b + res * (1.0 - b.a);
		}
		fragColor = res * clip * vParams.w;
#ifdef DUAL
		fragAlpha = fragColor.aaaa;
#endif
		return;
	}
	if (kind < 1.5 && vParams.y > 0.5) {
		// An inner shadow: inside the box, what the hole, blurred, leaves.
		float sigma = vParams.z;
		float hole = 0.0;
		if (vWidths.z > 0.0 && vWidths.w > 0.0) {
			float corner = max(max(vInner.x, vInner.y), max(vInner.z, vInner.w));
			hole = sigma > 0.0 ? boxShadow(p, vWidths, sigma, corner) : localCoverage(p, vWidths, vInner, vTransform0.w, fr);
		}
		res = premul(vColor) * (localCoverage(p, vRect, vRadii, vTransform0.w, fr) * (1.0 - hole));
		fragColor = res * clip * vParams.w;
#ifdef DUAL
		fragAlpha = fragColor.aaaa;
#endif
		return;
	}
	if (kind < 1.5) {
		float sigma = vParams.z;
		float corner = max(max(vRadii.x, vRadii.y), max(vRadii.z, vRadii.w));
		float s =
			sigma > 0.0 ? boxShadow(p, vRect, sigma, corner) : localCoverage(p, vRect, vRadii, vTransform0.w, fr);
		if (vWidths.z > 0.0 && vWidths.w > 0.0) {
			s *= 1.0 - localCoverage(p, vWidths, vInner, vTransform0.w, fr); // outside the box casting it
		}
		res = premul(vColor) * s;
		fragColor = res * clip * vParams.w;
#ifdef DUAL
		fragAlpha = fragColor.aaaa;
#endif
		return;
	}
	if (kind < 2.5) {
		vec4 c = paint(p, vParams.z, vRect, vColor, vColor2, vGrad);
		res = c *
			  textCoverage(sampleLocal(uMask, tex, vWidths, vTransform0.w).r, unpremul(c), vInner.x, vInner.y,
						   vRadii) *
			  glyphCov;
		fragColor = res * clip * vParams.w;
#ifdef DUAL
		fragAlpha = fragColor.aaaa;
#endif
		return;
	}
	if (kind < 3.5) {
		res = sampleLocal(uColor, tex, vWidths, vTransform0.w) * vColor.a * glyphCov;
		fragColor = res * clip * vParams.w;
#ifdef DUAL
		fragAlpha = fragColor.aaaa;
#endif
		return;
	}
	if (kind < 4.5) {
		res = sampleImage(uImage, tex, vWidths, vParams.z, dtex);
		if (vParams.y > 0.5) {
			res.rgb = vec3(dot(res.rgb, vec3(0.2126, 0.7152, 0.0722)));
		}
		res = tint(res, vColor) * localCoverage(p, vRect, vRadii, vTransform0.w, fr);
		fragColor = res * clip * vParams.w;
#ifdef DUAL
		fragAlpha = fragColor.aaaa;
#endif
		return;
	}
	{
		vec4 c = paint(p, vParams.z, vRect, vColor, vColor2, vGrad);
		vec3 straight = unpremul(c);
		if (vTransform0.w > 0.5) {
			vec3 rgb = sampleLocal(uColor, tex, vWidths, 1.0).rgb;
			res = c * textCoverage((rgb.r + rgb.g + rgb.b) / 3.0, straight, vInner.x, vInner.y, vRadii) *
				  glyphCov;
			fragColor = res * clip * vParams.w;
#ifdef DUAL
			fragAlpha = fragColor.aaaa;
#endif
			return;
		}
		vec3 a = subpixelCoverage(sampleLocal(uColor, tex, vWidths, vTransform0.w).rgb, straight, vInner.x,
								  vInner.y, vRadii);
		vec3 w = a * c.a * clip * vParams.w;
		float wa = (w.r + w.g + w.b) / 3.0;
#ifdef DUAL
		fragColor = vec4(straight * w, wa);
		fragAlpha = vec4(w, wa);
#else
		fragColor = vec4(straight * wa, wa);
#endif
		return;
	}
}

#endif

// The passes computing the backdrop of an effect draw a quad over their
// target, scissored to the texels they compute, and address texels by
// their index in memory, so that the rows of their textures go down as
// the frame's rows do.
#ifdef PASS_VERTEX

void main() {
	vec2 corner = vec2(float(gl_VertexID & 1), float(gl_VertexID >> 1));
	gl_Position = vec4(corner * 2.0 - 1.0, 0.0, 1.0);
}

#endif

#ifdef DOWN

// down averages the square of the area that texel stands for, repeating
// the area's last pixels past its far edges, or, for an image blurred
// (uMode & 1), reading them as transparent. The area of the frame (from
// uOrigin to uLimit) is copied to uSrc from the bottom row up: the row y
// of the frame is uShift.y - y there, and the column x is x - uShift.x;
// an image whose rows go down (uMode & 2) has its row y at y - uShift.y.
uniform sampler2D uSrc;
uniform ivec2 uOrigin;
uniform ivec2 uLimit;
uniform ivec2 uShift;
uniform int uDown;
uniform int uMode;

out vec4 passColor;

void main() {
	ivec2 o = uOrigin + ivec2(gl_FragCoord.xy) * uDown;
	vec4 c = vec4(0.0);
	for (int y = 0; y < uDown; y++) {
		for (int x = 0; x < uDown; x++) {
			ivec2 at = o + ivec2(x, y);
			if ((uMode & 1) != 0 && (at.x > uLimit.x || at.y > uLimit.y)) {
				continue;
			}
			at = min(at, uLimit);
			int row = (uMode & 2) != 0 ? at.y - uShift.y : uShift.y - at.y;
			c += texelFetch(uSrc, ivec2(at.x - uShift.x, row), 0);
		}
	}
	passColor = c * (1.0 / float(uDown * uDown));
}

#endif

#ifdef BLUR

// blur blurs along a row or a column, clamping at its ends, or, for an
// image blurred (uMode & 1), transparent beyond them.
uniform sampler2D uSrc;
uniform ivec2 uLimit;
uniform ivec2 uDir;
uniform int uRadius;
uniform float uSigma;
uniform int uMode;

out vec4 passColor;

void main() {
	ivec2 at = ivec2(gl_FragCoord.xy);
	vec4 c = vec4(0.0);
	float sum = 0.0;
	for (int o = -uRadius; o <= uRadius; o++) {
		float x = float(o);
		float w = uSigma > 0.0 ? exp(-(x * x) / (2.0 * uSigma * uSigma)) : 1.0;
		ivec2 q = at + uDir * o;
		sum += w;
		if ((uMode & 1) != 0 && (any(lessThan(q, ivec2(0))) || any(greaterThan(q, uLimit)))) {
			continue;
		}
		c += texelFetch(uSrc, clamp(q, ivec2(0), uLimit), 0) * w;
	}
	passColor = c / sum;
}

#endif
