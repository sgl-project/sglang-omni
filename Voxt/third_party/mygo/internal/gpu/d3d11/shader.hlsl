// The one shader of the Direct3D 11 renderer: every scene op is an
// instanced quad, and the pixel shader computes the coverage of rounded
// rectangles, borders, gradients, stripes and shadows from signed
// distances, as the software renderer (internal/raster) does, with the
// passes computing the backdrops of effects (passvs, downps and blurps).
// The pixel shaders of effects (scene.Effect) take its start, with EFFECT
// defined, before effect.hlsl. Colors are straight (not
// premultiplied) and blending happens in sRGB space, as in browsers, with a
// second color for the source's alpha of each channel, which subpixel
// glyphs need.
//
// `go generate` compiles it to DXBC (shaders.go) on Windows.

cbuffer Globals : register(b0) {
	float2 viewport;
	float2 pad;
};

struct Inst {
	float4 rect : RECT;         // x, y, width, height in pixels
	float4 radii : RADII;       // top-left, top-right, bottom-right, bottom-left
	float4 inner : INNER;       // radii of the border's inner edge, of the box casting a shadow, or of an inner shadow's hole
	float4 color : COLOR0;
	float4 color2 : COLOR1;     // gradient end
	float4 border : COLOR2;     // border color
	float4 grad : GRAD;         // gradient start and end points, or stripes
	float4 uv : UV;             // texture rectangle, normalized, border widths, the box casting a shadow, or an inner shadow's hole
	float4 clip : CLIP;         // the innermost clip rectangle
	float4 clipRadii : CLIPR;
	float4 transform0 : TRANSFORM0;
	float4 transform1 : TRANSFORM1;
	float4 transform2 : TRANSFORM2; // the projective row's g and h
	float4 params : PARAMS; // kind, dashed or grayscale, sigma or paint, opacity
};

struct VSOut {
	float4 pos : SV_Position;
	float4 pt : PIXEL; // the position in pixels, and the texture coordinates
	nointerpolation float4 rect : RECT;
	nointerpolation float4 radii : RADII;
	nointerpolation float4 inner : INNER;
	nointerpolation float4 color : COLOR0;
	nointerpolation float4 color2 : COLOR1;
	nointerpolation float4 border : COLOR2;
	nointerpolation float4 grad : GRAD;
	nointerpolation float4 widths : WIDTHS;
	nointerpolation float4 clip : CLIP;
	nointerpolation float4 clipRadii : CLIPR;
	nointerpolation float4 params : PARAMS;
	nointerpolation float4 transform0 : TRANSFORM0;
	nointerpolation float4 transform1 : TRANSFORM1;
	nointerpolation float4 transform2 : TRANSFORM2;
};

// frameH returns the frame point of local p, in homogeneous coordinates.
float3 frameH(Inst i, float2 p) {
	return float3(dot(i.transform0.xyz, float3(p, 1)), dot(i.transform1.xyz, float3(p, 1)),
				  dot(float3(i.transform2.xy, 1), float3(p, 1)));
}

VSOut vs(uint vid : SV_VertexID, Inst i) {
	float2 corner = float2(vid & 1, vid >> 1);
	float4 r = i.rect;
	float kind = i.params.x;
	if (kind < 0.5 || kind > 5.5) { // fills and effects
		r = float4(r.xy - 1, r.zw + 2);
	} else if (kind < 1.5) {
		float e = 3 * i.params.z + 1;
		r = float4(r.xy - e, r.zw + 2 * e);
	}
	float3 h = frameH(i, r.xy + corner * r.zw);
	if (i.transform0.w > 0.5 && h.z > 0) {
		// A pixel's reach at this corner, by the map's Jacobian there.
		float2 s = h.xy / h.z;
		float a = (i.transform0.x - s.x * i.transform2.x) / h.z, c = (i.transform0.y - s.x * i.transform2.y) / h.z;
		float b = (i.transform1.x - s.y * i.transform2.x) / h.z, d = (i.transform1.y - s.y * i.transform2.y) / h.z;
		float2 e = float2(abs(d) + abs(c), abs(b) + abs(a)) / abs(a * d - b * c);
		r = float4(r.xy - e, r.zw + 2 * e);
	}
	float2 p = r.xy + corner * r.zw;
	h = frameH(i, p);
	VSOut o;
	o.pos = float4(h.xy / viewport * float2(2, -2) + h.z * float2(-1, 1), 0, h.z);
	o.pt.xy = p;
	o.pt.zw = i.uv.xy + (p - i.rect.xy) / i.rect.zw * (i.uv.zw - i.uv.xy);
	o.rect = i.rect;
	o.radii = i.radii;
	o.inner = i.inner;
	o.color = i.color;
	o.color2 = i.color2;
	o.border = i.border;
	o.grad = i.grad;
	o.widths = i.uv;
	o.clip = i.clip;
	o.clipRadii = i.clipRadii;
	o.params = i.params;
	o.transform0 = i.transform0;
	o.transform1 = i.transform1;
	o.transform2 = i.transform2;
	return o;
}

// The color and, for dual-source blending, the source's alpha of each
// channel.
struct PSOut {
	float4 color : SV_Target0;
	float4 alpha : SV_Target1;
};

Texture2D maskTex : register(t0);
Texture2D colorTex : register(t1);
Texture2D imageTex : register(t2);
Texture2D clipTex : register(t5);
SamplerState samp : register(s0);

float sdRoundRect(float2 p, float4 rect, float4 radii) {
	float2 h = rect.zw * 0.5;
	float2 q = p - rect.xy - h;
	float r = q.x < 0 ? (q.y < 0 ? radii.x : radii.w) : (q.y < 0 ? radii.y : radii.z);
	float2 a = abs(q) - h + r;
	return length(max(a, 0)) + min(max(a.x, a.y), 0) - r;
}

float coverage(float d) { return saturate(0.5 - d); }

// rectCoverage returns how much of the pixel at p a rounded rectangle
// covers: by the distance to its edge near rounded corners, and exactly,
// the area of the pixel inside it, near square ones.
float rectCoverage(float2 p, float4 rect, float4 radii) {
	float2 q = p - rect.xy - rect.zw * 0.5;
	float r = q.x < 0 ? (q.y < 0 ? radii.x : radii.w) : (q.y < 0 ? radii.y : radii.z);
	if (r > 0) {
		return coverage(sdRoundRect(p, rect, radii));
	}
	float2 c = saturate(min(rect.xy + rect.zw, p + 0.5) - max(rect.xy, p - 0.5));
	return c.x * c.y;
}

// Frame is where a fragment is in the frame, s, and, when its instance's
// map is projective (persp), the columns of the map back, up to a factor.
struct Frame {
	float2 s;
	float3 c0, c1, c2;
	bool persp;
};
Frame frameOf(VSOut v) {
	Frame f;
	f.persp = any(v.transform2.xy != 0);
	// In perspective, the pixel's center; else the frame point of the
	// local one, which the affine map moves linearly: Direct3D 10's shaders
	// pass on 16 values at most.
	f.s = f.persp ? v.pos.xy : float2(dot(v.transform0.xyz, float3(v.pt.xy, 1)), dot(v.transform1.xyz, float3(v.pt.xy, 1)));
	float3 r0 = v.transform0.xyz, r1 = v.transform1.xyz, r2 = float3(v.transform2.xy, 1);
	f.c0 = cross(r1, r2);
	f.c1 = cross(r2, r0);
	f.c2 = cross(r0, r1);
	return f;
}
// localOf returns the point of the instance that shows at the frame's s.
float2 localOf(Frame f, float2 s) {
	float3 q = f.c0 * s.x + f.c1 * s.y + f.c2;
	return q.xy / q.z;
}
// perspectiveCoverage is localCoverage in perspective: each sample where
// the CPU renderer takes it.
float perspectiveCoverage(float4 rect, float4 radii, Frame fr) {
	float2 p = localOf(fr, fr.s);
	float2 a = localOf(fr, fr.s + float2(0.5, 0.5)) - p, b = localOf(fr, fr.s + float2(0.5, -0.5)) - p;
	float reach = max(abs(a.x) + abs(a.y), abs(b.x) + abs(b.y)) * 2;
	float d = sdRoundRect(p, rect, radii);
	if (d < -reach)
		return 1;
	if (d > reach)
		return 0;
	float n = 0;
	[loop] for (int y = 0; y < 4; y++)[loop] for (int x = 0; x < 4; x++) {
		float2 q = localOf(fr, fr.s + float2((float(x) + 0.5) / 4 - 0.5, (float(y) + 0.5) / 4 - 0.5));
		n += sdRoundRect(q, rect, radii) <= 0 ? 1 : 0;
	}
	return n / 16;
}
float localCoverage(float2 p, float4 rect, float4 radii, float affine, Frame fr) {
	if (affine < 0.5)
		return rectCoverage(p, rect, radii);
	float2 dx = ddx(p), dy = ddy(p), a = (dx + dy) * 0.5, b = (dx - dy) * 0.5;
	if (fr.persp)
		return perspectiveCoverage(rect, radii, fr);
	float reach = max(abs(a.x) + abs(a.y), abs(b.x) + abs(b.y)) * 2;
	float d = sdRoundRect(p, rect, radii);
	if (d < -reach)
		return 1;
	if (d > reach)
		return 0;
	float n = 0;
	[loop] for (int y = 0; y < 4; y++)[loop] for (int x = 0; x < 4; x++) {
		float2 q = p + dx * ((float(x) + 0.5) / 4 - 0.5) + dy * ((float(y) + 0.5) / 4 - 0.5);
		n += sdRoundRect(q, rect, radii) <= 0 ? 1 : 0;
	}
	return n / 16;
}
float clipCoverage(VSOut i) {
	Frame fr = frameOf(i);
	float2 world = fr.s;
	if (i.transform1.w < 0.5)
		return rectCoverage(world, i.clip, i.clipRadii);
	if (i.transform1.w > 1.5)
		return localCoverage(i.pt.xy, i.clip, float4(0, 0, 0, 0), i.transform0.w, fr) *
			   clipTex.Sample(samp, (world - i.clipRadii.xy) / i.clipRadii.zw).a;
	return clipTex.Sample(samp, (world - i.clip.xy) / i.clip.zw).a;
}
float4 sampleLocal(Texture2D t, float2 uv, float4 src, float affine) {
	if (affine > 0.5) {
		uint w, h;
		t.GetDimensions(w, h);
		float2 halfPixel = 0.5 / float2(w, h);
		uv = clamp(uv, src.xy + halfPixel, src.zw - halfPixel);
	}
	return t.Sample(samp, uv);
}

// sampleImage samples an image held inside src, which is reversed along
// an axis it flips, so that a part of a sheet shows none of its
// neighbors; where pixelated is set, as pixel art draws, each texel a
// square of its color whose edges blend with the next only across the
// pixel they cross (Ebitengine's FilterPixelated): nearest where it shows
// at whole multiples of its size, and texels of even widths with smooth
// edges at any other. dtex is the fwidth of uv, taken where every
// fragment of the quad takes it.
float4 sampleImage(Texture2D t, float2 uv, float4 src, float pixelated, float2 dtex) {
	uint w, h;
	t.GetDimensions(w, h);
	float2 size = float2(w, h);
	float2 lo = min(src.xy, src.zw), hi = max(src.xy, src.zw);
	if (pixelated > 0.5) {
		float2 tc = uv * size - 0.5;
		float2 i = floor(tc);
		float2 f = saturate((tc - i - 0.5) / clamp(dtex * size, 1e-5, 1.0) + 0.5);
		uv = (i + f + 0.5) / size;
	}
	return t.Sample(samp, clamp(uv, lo + 0.5 / size, hi - 0.5 / size));
}

// tint multiplies premultiplied c by straight color t.
float4 tint(float4 c, float4 t) { return c * float4(t.rgb * t.a, t.a); }

float4 premul(float4 c) { return float4(c.rgb * c.a, c.a); }

float3 unpremul(float4 c) { return c.a > 0 ? c.rgb / c.a : float3(0, 0, 0); }

// textCoverage corrects the coverage a of a glyph of straight color c as
// Direct2D blends text (scene.TextCoverage): it enhances the contrast,
// the more the darker c is, and corrects for gamma with the ratios g.
float textCoverage(float a, float3 c, float contrast, float boost, float4 g) {
	float k = contrast * saturate(3 - 4 * dot(c, float3(0.30, 0.59, 0.11))) + boost;
	a = a * (k + 1) / (a * k + 1);
	float f = dot(c, float3(0.25, 0.5, 0.25));
	return saturate(a + a * (1 - a) * ((g.x * f + g.y) * a + (g.z * f + g.w)));
}

// What follows is the renderer's own; effects (effect.hlsl) take what is
// above.
#ifndef EFFECT

// subpixelCoverage does the same for each subpixel of a subpixel glyph.
float3 subpixelCoverage(float3 a, float3 c, float contrast, float boost, float4 g) {
	float k = contrast * saturate(3 - 4 * dot(c, float3(0.30, 0.59, 0.11))) + boost;
	a = a * (k + 1) / (a * k + 1);
	return saturate(a + a * (1 - a) * ((g.x * c + g.y) * a + (g.z * c + g.w)));
}

float2 erf2(float2 x) {
	float2 s = sign(x), a = abs(x);
	x = 1 + (0.278393 + (0.230389 + 0.078108 * (a * a)) * a) * a;
	x *= x;
	return s - s / (x * x);
}

float gaussian(float x, float sigma) { return exp(-(x * x) / (2 * sigma * sigma)) / (2.50662827463 * sigma); }

// The blurred rounded box of Evan Wallace: exact along x, four samples
// along y.
float shadowX(float x, float y, float sigma, float corner, float2 h) {
	float delta = min(h.y - corner - abs(y), 0);
	float curved = h.x - corner + sqrt(max(0, corner * corner - delta * delta));
	float2 integral = 0.5 + 0.5 * erf2((x + float2(-curved, curved)) * (sqrt(0.5) / sigma));
	return integral.y - integral.x;
}

float boxShadow(float2 p, float4 rect, float sigma, float corner) {
	float2 h = rect.zw * 0.5;
	p -= rect.xy + h;
	float low = p.y - h.y, high = p.y + h.y;
	float start = clamp(-3 * sigma, low, high);
	float end = clamp(3 * sigma, low, high);
	float step = (end - start) / 4;
	float y = start + step * 0.5;
	float v = 0;
	for (int k = 0; k < 4; k++) {
		v += shadowX(p.x, p.y - y, sigma, corner, h) * gaussian(y, sigma) * step;
		y += step;
	}
	return v;
}

float3 toLinear(float3 c) { return c <= 0.04045 ? c / 12.92 : pow((c + 0.055) / 1.055, 2.4); }

float3 toSRGB(float3 c) { return c <= 0.0031308 ? c * 12.92 : 1.055 * pow(max(c, 0), 1.0 / 2.4) - 0.055; }

float3 cbrt3(float3 v) { return sign(v) * pow(abs(v), 1.0 / 3.0); }

float3 oklab(float3 srgb) {
	float3 c = toLinear(srgb);
	float3 lms = cbrt3(float3(0.4122214708 * c.r + 0.5363325363 * c.g + 0.0514459929 * c.b,
							  0.2119034982 * c.r + 0.6806995451 * c.g + 0.1073969566 * c.b,
							  0.0883024619 * c.r + 0.2817188376 * c.g + 0.6299787005 * c.b));
	return float3(0.2104542553 * lms.x + 0.7936177850 * lms.y - 0.0040720468 * lms.z,
				  1.9779984951 * lms.x - 2.4285922050 * lms.y + 0.4505937099 * lms.z,
				  0.0259040371 * lms.x + 0.7827717662 * lms.y - 0.8086757660 * lms.z);
}

float3 fromOklab(float3 lab) {
	float3 lms = float3(lab.x + 0.3963377774 * lab.y + 0.2158037573 * lab.z,
						lab.x - 0.1055613458 * lab.y - 0.0638541728 * lab.z,
						lab.x - 0.0894841775 * lab.y - 1.2914855480 * lab.z);
	lms = lms * lms * lms;
	return toSRGB(float3(4.0767416621 * lms.x - 3.3077115913 * lms.y + 0.2309699292 * lms.z,
						 -1.2684380046 * lms.x + 2.6097574011 * lms.y - 0.3413193965 * lms.z,
						 -0.0041960863 * lms.x - 0.7034186147 * lms.y + 1.7076147010 * lms.z));
}

// The stops of the frame's gradients (gpu.Builder.Stops): two texels each,
// the color, straight sRGB, then its place and its color in Oklab.
Texture2D stopsTex : register(t10);

// stop returns texel k of the stops.
float4 stop(int k) {
	return stopsTex.Load(int3(k % 1024, k / 1024, 0));
}

// stopsAt returns the color at t of the gradient whose stops stopsTex
// holds from first, n of them (gpu.Builder.Stops): the first stop's
// before it, the last's after it, and between two the colors of both
// mixed, premultiplied, in sRGB, or in Oklab, as internal/raster's stopsAt.
float4 stopsAt(float4 c2, float t, bool lab) {
	int first = int(c2.x);
	int n = int(c2.y);
	if (n <= 0) {
		return float4(0, 0, 0, 0);
	}
	float4 a = stop(2 * first);
	float4 ap = stop(2 * first + 1);
	if (t <= ap.x) {
		return premul(a);
	}
	float4 last = stop(2 * (first + n - 1));
	float4 lastp = stop(2 * (first + n - 1) + 1);
	if (t >= lastp.x) {
		return premul(last);
	}
	float4 b = last;
	float4 bp = lastp;
	for (int i = 1; i < n - 1; i++) {
		float4 c = stop(2 * (first + i));
		float4 cp = stop(2 * (first + i) + 1);
		if (t < cp.x) {
			b = c;
			bp = cp;
			break;
		}
		a = c;
		ap = cp;
	}
	float f = saturate((t - ap.x) / max(bp.x - ap.x, 1e-6));
	float alpha = lerp(a.a, b.a, f);
	if (!lab) {
		return float4(lerp(a.rgb * a.a, b.rgb * b.a, f), alpha);
	}
	if (alpha <= 0) {
		return float4(0, 0, 0, 0);
	}
	return float4(saturate(fromOklab(lerp(ap.yzw * a.a, bp.yzw * b.a, f) / alpha)) * alpha, alpha);
}

// paint returns the premultiplied color at p of plain color, a gradient
// mixed in sRGB (1) or Oklab (2), stripes (3), or a gradient of stops,
// linear (4, 5) or conic (6, 7), mixed in sRGB or in Oklab, as
// scene.Paint says.
float4 paint(float2 p, float mode, float4 rect, float4 c1, float4 c2, float4 g) {
	if (mode < 0.5) {
		return premul(c1);
	}
	if (mode > 3.5) {
		float t;
		if (mode < 5.5) {
			float2 d = g.zw - g.xy;
			t = dot(p - g.xy, d) / max(dot(d, d), 0.0001);
		} else {
			float2 d = p - g.xy;
			t = (atan2(d.x, -d.y) - g.z) * 0.15915494309189535;
			t -= floor(t);
		}
		return stopsAt(c2, t, mode > 4.5 && mode < 5.5 || mode > 6.5);
	}
	if (mode < 2.5) {
		float2 d = g.zw - g.xy;
		float t = saturate(dot(p - g.xy, d) / max(dot(d, d), 0.0001));
		float a = lerp(c1.a, c2.a, t);
		if (mode < 1.5) {
			return float4(lerp(c1.rgb * c1.a, c2.rgb * c2.a, t), a);
		}
		float3 lab = lerp(oklab(c1.rgb) * c1.a, oklab(c2.rgb) * c2.a, t);
		return a > 0 ? float4(saturate(fromOklab(lab / a)) * a, a) : float4(0, 0, 0, 0);
	}
	float s = dot(p - rect.xy, g.xy);
	float phase = s - g.w * floor(s / g.w);
	float cov = coverage(min(max(-phase, phase - g.z), g.w - phase));
	return premul(c1) * cov + premul(c2) * (1 - cov);
}

// dash returns how much of a dashed border shows at p: each side, which
// the pixel belongs to when it is nearest that side's edge in widths of
// its border, has an odd number of dashes and gaps of equal length, about
// three widths, starting and ending with a dash.
float dash(float2 p, float4 rect, float4 w) {
	float2 q = p - rect.xy;
	float dt = w.x > 0 ? q.y / w.x : 1e9;
	float dr = w.y > 0 ? (rect.z - q.x) / w.y : 1e9;
	float db = w.z > 0 ? (rect.w - q.y) / w.z : 1e9;
	float dl = w.w > 0 ? q.x / w.w : 1e9;
	float s, len, bw;
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
	float n = max(1, floor((len / (3 * bw) + 1) * 0.5 + 0.5));
	float period = 6 * bw;
	n += n * period <= len ? 1 : 0;
	n -= (n - 1) * period > len ? 1 : 0;
	float seg = len / (2 * n - 1);
	float k = floor(s / seg);
	float f = s - k * seg;
	float edge = min(f, seg - f);
	return coverage(k - 2 * floor(k * 0.5) < 0.5 ? -edge : edge);
}

PSOut ps(VSOut i) {
	Frame fr = frameOf(i);
	float2 dtex = fwidth(i.pt.zw);
	float kind = i.params.x;
	float4 res;
	float glyphCov = i.transform0.w > 0.5 ? localCoverage(i.pt.xy, i.rect, float4(0, 0, 0, 0), 1, fr) : 1;
	if (kind < 0.5) {
		float outer = localCoverage(i.pt.xy, i.rect, i.radii, i.transform0.w, fr);
		// params.y: 1 dashes the border, 2 paints it with the gradient (the
		// inside with plain color), 3 both.
		bool borderPaint = i.params.y > 1.5;
		float4 pc = paint(i.pt.xy, i.params.z, i.rect, i.color, i.color2, i.grad);
		res = (borderPaint ? premul(i.color) : pc) * outer;
		float4 bw = i.widths; // top, right, bottom, left
		if (any(bw > 0)) {
			float4 ir = float4(i.rect.xy + bw.wx, i.rect.zw - bw.yz - bw.wx);
			float innerCov = (ir.z > 0 && ir.w > 0) ? localCoverage(i.pt.xy, ir, i.inner, i.transform0.w, fr) : 0;
			float bc = saturate(outer - innerCov);
			float dashed = i.params.y - (borderPaint ? 2 : 0);
			if (dashed > 0.5) {
				bc *= dash(i.pt.xy, i.rect, bw);
			}
			float4 b = (borderPaint ? pc * float4(i.border.rgb * i.border.a, i.border.a) : premul(i.border)) * bc;
			res = b + res * (1 - b.a);
		}
	} else if (kind < 1.5 && i.params.y > 0.5) {
		// An inner shadow: inside the box, what the hole, blurred, leaves.
		float sigma = i.params.z;
		float hole = 0;
		if (i.widths.z > 0 && i.widths.w > 0) {
			float corner = max(max(i.inner.x, i.inner.y), max(i.inner.z, i.inner.w));
			hole = sigma > 0 ? boxShadow(i.pt.xy, i.widths, sigma, corner)
						  : localCoverage(i.pt.xy, i.widths, i.inner, i.transform0.w, fr);
		}
		res = premul(i.color) * (localCoverage(i.pt.xy, i.rect, i.radii, i.transform0.w, fr) * (1 - hole));
	} else if (kind < 1.5) {
		float sigma = i.params.z;
		float corner = max(max(i.radii.x, i.radii.y), max(i.radii.z, i.radii.w));
		float s = sigma > 0 ? boxShadow(i.pt.xy, i.rect, sigma, corner)
							: localCoverage(i.pt.xy, i.rect, i.radii, i.transform0.w, fr);
		if (i.widths.z > 0 && i.widths.w > 0) {
			s *= 1 - localCoverage(i.pt.xy, i.widths, i.inner, i.transform0.w, fr); // outside the box casting it
		}
		res = premul(i.color) * s;
	} else if (kind < 2.5) {
		float4 c = paint(i.pt.xy, i.params.z, i.rect, i.color, i.color2, i.grad);
		res = c *
			  textCoverage(sampleLocal(maskTex, i.pt.zw, i.widths, i.transform0.w).r, unpremul(c), i.inner.x,
						   i.inner.y, i.radii) *
			  glyphCov;
	} else if (kind < 3.5) {
		res = sampleLocal(colorTex, i.pt.zw, i.widths, i.transform0.w) * i.color.a * glyphCov;
	} else if (kind < 4.5) {
		res = sampleImage(imageTex, i.pt.zw, i.widths, i.params.z, dtex);
		if (i.params.y > 0.5) {
			res.rgb = dot(res.rgb, float3(0.2126, 0.7152, 0.0722));
		}
		res = tint(res, i.color) * localCoverage(i.pt.xy, i.rect, i.radii, i.transform0.w, fr);
	} else {
		float4 c = paint(i.pt.xy, i.params.z, i.rect, i.color, i.color2, i.grad);
		float3 straight = unpremul(c);
		if (i.transform0.w > 0.5) {
			float3 rgb = sampleLocal(colorTex, i.pt.zw, i.widths, 1).rgb;
			res = c * textCoverage((rgb.r + rgb.g + rgb.b) / 3, straight, i.inner.x, i.inner.y, i.radii) *
				  glyphCov;
			PSOut gray;
			gray.color = res * clipCoverage(i) * i.params.w;
			gray.alpha = gray.color.aaaa;
			return gray;
		}
		float3 a = subpixelCoverage(sampleLocal(colorTex, i.pt.zw, i.widths, i.transform0.w).rgb, straight,
									i.inner.x, i.inner.y, i.radii);
		float3 w = a * c.a * clipCoverage(i) * i.params.w;
		float wa = (w.r + w.g + w.b) / 3;
		PSOut so;
		so.color = float4(straight * w, wa);
		so.alpha = float4(w, wa);
		return so;
	}
	float clip = clipCoverage(i);
	PSOut o;
	o.color = res * (clip * i.params.w);
	o.alpha = o.color.aaaa;
	return o;
}

// The passes computing the backdrop of an effect draw a quad over their
// target, scissored to the texels they compute.
cbuffer PassConstants : register(b1) {
	int2 passOrigin; // down: the first pixel of the area read, in the frame
	int2 passLimit;	 // the last texel that may be read: in the frame for down
	int2 passShift;	 // down: subtracted from frame pixels to read the source
	int2 passDir;	 // blur: along rows (1, 0) or columns (0, 1)
	int passDown;	 // down: the size of the squares averaged
	int passRadius;	 // blur: how far it reaches each way
	float passSigma; // blur: its standard deviation, 0 for none
	int passMode;	 // 1: transparent beyond the area or the texels (gpu.PassClear)
};

Texture2D passSrc : register(t4);

float4 passvs(uint vid : SV_VertexID) : SV_Position {
	float2 corner = float2(vid & 1, vid >> 1);
	return float4(corner * 2 - 1, 0, 1);
}

// down averages the square of the area that texel stands for, repeating
// the area's last pixels past its far edges, or, for an image blurred
// (mode 1), reading them as transparent.
float4 downps(float4 pos : SV_Position) : SV_Target {
	int2 o = passOrigin + int2(pos.xy) * passDown;
	float4 c = float4(0, 0, 0, 0);
	for (int y = 0; y < passDown; y++) {
		for (int x = 0; x < passDown; x++) {
			int2 at = o + int2(x, y);
			if ((passMode & 1) != 0 && (at.x > passLimit.x || at.y > passLimit.y)) {
				continue;
			}
			c += passSrc.Load(int3(min(at, passLimit) - passShift, 0));
		}
	}
	return c * (1.0 / float(passDown * passDown));
}

// blur blurs along a row or a column, clamping at its ends, or, for an
// image blurred (mode 1), transparent beyond them.
float4 blurps(float4 pos : SV_Position) : SV_Target {
	int2 at = int2(pos.xy);
	float4 c = float4(0, 0, 0, 0);
	float sum = 0;
	for (int o = -passRadius; o <= passRadius; o++) {
		float x = float(o);
		float w = passSigma > 0 ? exp(-(x * x) / (2 * passSigma * passSigma)) : 1;
		int2 q = at + passDir * o;
		sum += w;
		if ((passMode & 1) != 0 && (any(q < int2(0, 0)) || any(q > passLimit))) {
			continue;
		}
		c += passSrc.Load(int3(clamp(q, int2(0, 0), passLimit), 0)) * w;
	}
	return c / sum;
}

#endif
