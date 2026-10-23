// Package gpu turns scenes into what the GPU renderers draw: one instanced
// quad per operation, for a shader that computes rounded rectangles,
// borders, gradients, stripes, shadows and atlas coverage itself, in
// batches that share a scissor rectangle and an image. Every renderer
// (d3d11, metal, gl) draws the same instances, the way internal/raster
// draws scenes on the CPU.
//
// An effect (scene.Effect) draws with a pipeline of its own, made from
// its shader, in a batch of its own. One reading its backdrop has its
// batch start after a Backdrop step, which renderers take between
// batches: they read the backdrop's area of what they drew so far,
// average it over squares (the pass "down") into a texture, blur that
// along rows into another and along columns back (the pass "blur"),
// keeping 8 bits a channel, and bind it for the batch (the shaders'
// backdrop texture), as internal/raster computes it.
//
// The shaders write a second color for blending, the source's alpha for
// each channel: the alpha of the color for everything but subpixel
// glyphs, whose subpixels cover each channel by its own. Renderers blend
// with dual-source blending: the destination times one minus the second
// color, plus the first.
//
// A hole (scene.OpHole) is an opaque fill in a batch of its own, which
// renderers blend with none of the source: the destination times one
// minus the second color, its coverage, which leaves it transparent.
package gpu

import (
	"fmt"
	"math"
	"slices"
	"strings"
	"unsafe"

	"github.com/egoist/mygo/internal/scene"
)

// GoBytes returns the Go declaration of a variable name holding b, as
// code compiled ahead of time is generated.
func GoBytes(name string, b []byte) string {
	var sb strings.Builder
	fmt.Fprintf(&sb, "var %s = []byte{", name)
	for i, c := range b {
		if i%16 == 0 {
			sb.WriteString("\n\t")
		}
		fmt.Fprintf(&sb, "0x%02x, ", c)
	}
	sb.WriteString("\n}\n")
	return sb.String()
}

// Instance is the data of one quad, in device pixels, as the shaders read
// it: thirteen float4s.
type Instance struct {
	// Rect is x, y, width, height; Radii the corners' radii (top-left,
	// top-right, bottom-right, bottom-left), negative for continuous
	// corners (scene.Corners), which only Metal's shader draws, or a
	// glyph's gamma ratios
	// (scene.TextParams); Inner those of a border's inner edge, or of the
	// box casting a shadow, or a glyph's contrast and thin boost in Inner[0]
	// and Inner[1].
	Rect, Radii, Inner [4]float32
	// Color and Color2 (a gradient's end) and Border are straight RGBA;
	// Color tints an image (white for none).
	Color, Color2, Border [4]float32
	// Grad holds a gradient's start and end points, or the stripes' unit
	// vector across them, width and period; UV the texture rectangle,
	// normalized, a fill's border widths (top, right, bottom, left), or
	// the box casting a shadow, which shows only outside it (none when
	// empty).
	//
	// An effect has its parameters (scene.EffectOp.Params) in Inner,
	// Color, Color2, Border and Grad, and in UV where its backdrop's area
	// starts in the frame and its size in texels.
	Grad, UV [4]float32
	// Clip and ClipRadii are the innermost clip, which the shader cuts.
	Clip, ClipRadii [4]float32
	// Params is the kind (0 fill, 1 shadow, 2 mask glyph, 3 color glyph, 4
	// image, 5 subpixel glyph, 6 effect); 1 for a dashed border, a
	// grayscale image or an inner shadow, whose box is in Rect and Radii
	// and the hole it leaves in UV and Inner (none when empty); the
	// shadow's sigma (0 for none), the paint (scene.Paint), 1 for an image
	// sampled at its nearest texel, or the size of the squares an effect's
	// backdrop averages; and the opacity.
	Params [4]float32
	// Transform rows map local pixels to the frame. Transform0.w enables
	// transformed antialiasing, Transform1.w selects how the clip applies: 0
	// the clip rectangle cut by the shader, 1 a cached clip mask, 2 a hard
	// clip over a masked one, and 3 none, as the builder found the clip's
	// rectangle to hold the whole quad with room for the antialiasing (see
	// markFill): the shaders then skip computing it.
	// Transform2 holds the projective row's g and h, the frame point of
	// local (x, y) being Transform0's and Transform1's dot products with
	// (x, y, 1) over g*x + h*y + 1: shaders interpolate in perspective.
	Transform0, Transform1, Transform2 [4]float32
}

// BackdropImage is the image of a shader of package ui/shader that holds
// its backdrop, when it reads it (internal/shader.BackdropImage):
// renderers bind their backdrop's texture in its place.
const BackdropImage = 2

// InstanceSize is the size of an Instance in bytes.
const InstanceSize = int(unsafe.Sizeof(Instance{}))

// Scissor is a scissor rectangle in device pixels, the layout of a
// Direct3D RECT.
type Scissor struct{ Left, Top, Right, Bottom int32 }

// Empty reports whether the rectangle has no pixels.
func (s Scissor) Empty() bool { return s.Right <= s.Left || s.Bottom <= s.Top }

// Batch is a run of instances that draw with the same scissor rectangle
// and image.
type Batch struct {
	Start, Count int
	Scissor      Scissor
	// Image is the renderer's texture of the batch's image, or 0.
	Image uintptr
	// ClipMask is a cached nested clip texture, sampled in frame coordinates.
	ClipMask uintptr
	// Effect is the effect of the batch's one instance, drawn with a
	// pipeline of its own, and Backdrop, unless 0, 1 + the index in
	// Builder.Backdrops of its backdrop, which the renderer reads before
	// drawing it.
	Effect   *scene.Effect
	Backdrop int
	// Shader is the effect's op, for a shader's (Effect.Shader), and
	// Textures the renderer's textures of its images, 0 for none.
	Shader   *scene.EffectOp
	Textures [4]uintptr
	// Uniforms are the words of the shader's uniforms the batch draws
	// with: the op's, or its wide ones for a wide gamut (Wide).
	Uniforms []uint32
	// Hole tells that the batch's instances make their shapes transparent
	// (scene.OpHole): renderers blend them with a source factor of zero.
	Hole bool
	// Opaque tells that every fragment the batch's instances draw within
	// its scissor rectangle is opaque, with coverage exactly 1 (the cores of
	// opaque fills, see addOpaque): renderers may draw it without blending,
	// which writes the same bytes, and which lets a tile-based GPU
	// discard whatever it drew under them before shading it.
	Opaque bool
	// Class is the paths of the shader that the batch's instances need
	// (see Class), and Continuous whether any of them has continuous
	// corners, in its shape or in the clip the shader cuts; renderers that do
	// not specialize their shader ignore them.
	Class      Class
	Continuous bool
}

// Class says which paths of the fragment shader a batch's instances take.
// Renderers specialize their shader for it (Metal's function constants, a
// #define of GLSL, a macro of HLSL): the general shader holds the code of
// every path, and registers for the heaviest, so the plain fills most of a
// window is made of ran at a fraction of the speed it has without the rest.
// A batch holds instances of one class.
type Class uint8

const (
	// ClassGeneral is the whole shader; transformed instances, effects and
	// anything else use it.
	ClassGeneral Class = iota
	// ClassPlain is fills that are solid, at opacity 1, without a border, not
	// transformed.
	ClassPlain
	// ClassFill is the other fills (gradients, stripes, borders, dashes), not
	// transformed.
	ClassFill
	// ClassShadow is shadows, ClassGlyph mask glyphs, ClassSubpixel
	// subpixel glyphs and ClassImage images and color glyphs, not transformed.
	ClassShadow
	ClassGlyph
	ClassSubpixel
	ClassImage
	// NumClasses is how many classes there are.
	NumClasses
)

// classOf returns the class of an instance prepared for its clip, and whether
// it has continuous corners (negative radii) that the shader draws, which
// the general class always has the code for.
func classOf(in *Instance) (Class, bool) {
	if in.Transform0[3] > 0.5 {
		return ClassGeneral, false
	}
	neg := func(r [4]float32) bool { return r[0] < 0 || r[1] < 0 || r[2] < 0 || r[3] < 0 }
	// The shader cuts the clip itself unless a mask or the builder does.
	continuous := in.Transform1[3] == 0 && neg(in.ClipRadii)
	switch in.Params[0] {
	case 0:
		continuous = continuous || neg(in.Radii) || neg(in.Inner)
		// Plain fills take the shader's own path (see its kPlain), for which
		// the opacity is 1: at another the compiler fuses the product of
		// the specialized function differently, and a result at a rounding
		// tie differs by a level.
		if in.Params[2] < 0.5 && in.Params[3] == 1 && !(in.UV[0] > 0 || in.UV[1] > 0 || in.UV[2] > 0 || in.UV[3] > 0) {
			return ClassPlain, continuous
		}
		return ClassFill, continuous
	case 1:
		return ClassShadow, continuous || neg(in.Radii) || neg(in.Inner)
	case 2:
		return ClassGlyph, continuous
	case 3:
		return ClassImage, continuous
	case 4:
		return ClassImage, continuous || neg(in.Radii)
	case 5:
		return ClassSubpixel, continuous
	}
	return ClassGeneral, false
}

// Builder builds the instances and batches of scenes, reusing its memory
// from frame to frame.
type Builder struct {
	Instances []Instance
	Batches   []Batch
	Backdrops []scene.Backdrop
	// Stops holds the stops of the gradients the instances draw with
	// (scene.Stop), two RGBA texels of 32-bit floats each, StopsWidth to
	// a row: the color, straight sRGB, then its place and its color in
	// Oklab. An instance painting with stops has the first's index and
	// how many in Color2.
	Stops []float32
	// BlurredImage, unless nil, returns the texture of an image's area
	// blurred as bk says (scene.ImageBlur), bk.Size() texels, rows going
	// down, which the renderer computes before the frame's pass: the
	// image's texture averaged over squares, blurred along rows and along
	// columns (ImageDownPass, ImageBlurPass). Without it, images draw
	// unblurred.
	BlurredImage func(img *scene.Image, bk scene.Backdrop) uintptr
	// Wide draws the colors of scenes outside the sRGB gamut
	// (Scene.Wide), for a target that keeps them, as Metal's float16
	// drawables do; without it, their nearest sRGB colors.
	Wide bool
	// NoShadowSplit draws every shadow as one instance, for comparison tests.
	NoShadowSplit bool
	// NoRingSplit draws every border-only fill as one instance (see
	// addFill), for tests that compare the two.
	NoRingSplit bool
	// NoClasses puts every batch in ClassGeneral, for tests and
	// measurements that compare the specialized shaders with the general one.
	NoClasses bool
	// Classes separates batches for renderers with specialized fragment shaders.
	Classes bool
	// GeneralClasses has a bit (1 << Class) for each class that draws with the
	// general shader, for tests that find which specialized one differs.
	GeneralClasses uint32
	// Opaque lets large opaque fills be drawn as an opaque core with blended
	// edges around it (see addOpaque), which a renderer sets that gains from
	// drawing the cores without blending (Metal's tile-based GPUs: hidden
	// surfaces are discarded): the others would only draw more calls for the
	// same pixels. Tests and measurements switch it to compare the two.
	Opaque bool
	// NoCull draws every op, even those that later opaque fills cover whole
	// (see planCull), for tests that compare the two.
	NoCull bool
	// GeneralFill makes the shaders take their general path for every fill,
	// and NoClipSkip makes them compute the clip's coverage where it is 1
	// (see markFill), for tests and measurements that compare them.
	GeneralFill, NoClipSkip bool
	// ClipSkip enables the clip-coverage flag understood by the Metal shader.
	ClipSkip bool
	// Culled is how many ops the last Build left out as covered.
	Culled, OpaqueCores int
	opaque              bool
	culled              []bool
	plan                []cullInfo
	planStack           []clip
	split               bool
	splitScissor        Scissor
	stack               []clip
	// encloses tells, for each op of the scene, that it pushes a clip with
	// a transformed op or glyph inside it.
	encloses     []bool
	pushes       []int
	clipShapes   []scene.ClipShape
	maskSlots    []clipMaskSlot
	maskUsed     int
	frame        uint64
	imageTexture func(*scene.Image) uintptr
	preparedMask uintptr
}

// Pass is what a pass computing a backdrop reads, the layout of the
// shaders' Pass: down reads the area from Origin to Limit, frame pixels,
// at their place less Shift in its source, and averages squares of Down
// pixels a side; blur reads texels up to Limit, Radius of them each way
// along Dir, weighted by a Gaussian of standard deviation Sigma.
type Pass struct {
	Origin, Limit, Shift, Dir [2]int32
	Down, Radius              int32
	Sigma                     float32
	// Mode is PassClear for passes reading an image to blur (scene.Op's
	// Blur), and PassUnflipped for those of OpenGL reading an image whose
	// rows go down.
	Mode int32
}

// Modes of passes (Pass.Mode).
const (
	// PassClear reads what lies beyond the area (down) or the texels
	// (blur) as transparent, where it would repeat the last ones.
	PassClear = 1 << iota
	// PassUnflipped tells OpenGL's pass down that its source's rows go
	// down, as an image uploaded: its row y is y less Shift.y.
	PassUnflipped
)

// ImageDownPass returns the pass averaging the squares of an image's
// area that bk blurs (scene.ImageBlur), reading its texture's texels
// where they are, transparent beyond the area.
func ImageDownPass(bk scene.Backdrop) Pass {
	p := DownPass(bk, [2]int32{})
	p.Mode = PassClear
	return p
}

// ImageBlurPass returns the pass blurring the texels of an image's blur
// along rows (dir 1, 0) or columns (0, 1), transparent beyond them.
func ImageBlurPass(bk scene.Backdrop, dir [2]int32) Pass {
	p := BlurPass(bk, dir)
	p.Mode = PassClear
	return p
}

// StopsWidth is how many texels a row of the texture of a frame's stops
// holds (Builder.Stops).
const StopsWidth = 1024

// DownPass returns the pass averaging the squares of bk's area, whose
// frame pixel (x, y) is at (x, y) less shift in the texture it reads.
func DownPass(bk scene.Backdrop, shift [2]int32) Pass {
	return Pass{
		Origin: [2]int32{int32(bk.Area.Min.X), int32(bk.Area.Min.Y)},
		Limit:  [2]int32{int32(bk.Area.Max.X - 1), int32(bk.Area.Max.Y - 1)},
		Shift:  shift,
		Down:   int32(bk.Down),
	}
}

// BlurPass returns the pass blurring bk's texels along rows (dir 1, 0)
// or columns (0, 1).
func BlurPass(bk scene.Backdrop, dir [2]int32) Pass {
	w, h := bk.Size()
	return Pass{Limit: [2]int32{int32(w - 1), int32(h - 1)}, Dir: dir, Radius: int32(bk.Radius), Sigma: bk.Sigma}
}

// BackdropSize returns how large the textures of the backdrops of the
// last scene built must be, in texels: as the largest.
func (b *Builder) BackdropSize() (w, h int) {
	for _, bk := range b.Backdrops {
		bw, bh := bk.Size()
		w, h = max(w, bw), max(h, bh)
	}
	return w, h
}

type clip struct {
	rect    scene.Rect
	radii   [4]float32
	bounds  scene.Rect
	scissor Scissor
	shape   scene.ClipShape
	// moved tells that this clip or one it is nested in is transformed,
	// which the scissor rectangles don't follow.
	moved       bool
	masked      bool
	maskSlot    int
	maskTexture uintptr
}

// pushed returns the clip that op (an OpPushClip) makes within top, without
// a mask slot or texture.
func (b *Builder) pushed(top *clip, op *scene.Op, encloses bool) clip {
	bounds := top.bounds.Intersect(op.Transform.Bounds(op.Rect))
	// The innermost clip shapes its edges in the shader, the others cut
	// with the scissor rectangle.
	c := clip{rect: op.Rect, radii: scene.Corners(op.Rect, op.Radii, op.Continuous), bounds: bounds,
		shape: scene.ClipShape{Rect: op.Rect, Radii: op.Radii, Transform: op.Transform, Continuous: op.Continuous}, maskSlot: -1}
	c.moved = top.moved || op.Transform.Set
	c.masked = c.moved || encloses && (top.masked || c.radii != [4]float32{})
	if bounds.W > 0 && bounds.H > 0 {
		c.scissor = Scissor{int32(math.Floor(float64(bounds.X))), int32(math.Floor(float64(bounds.Y))),
			int32(math.Ceil(float64(bounds.X + bounds.W))), int32(math.Ceil(float64(bounds.Y + bounds.H)))}
	}
	return c
}

// Build turns a scene directly into GPU instances, retaining local geometry
// and composing transforms in the vertex shader. Nested affine clips use
// cached masks; stationary clipping never rasterizes again as content moves.
// A clip is masked only when it is transformed, nested in a transformed
// clip, or rounded (or masked) with transformed content inside: the others
// cut in the shader as in a scene without transforms.
func (b *Builder) Build(s *scene.Scene, image func(*scene.Image) uintptr) *scene.Scene {
	b.markEnclosing(s)
	b.imageTexture = image
	b.maskUsed = 0
	b.frame++
	b.clipShapes = b.clipShapes[:0]
	b.Instances = b.Instances[:0]
	b.Batches = b.Batches[:0]
	b.Backdrops = b.Backdrops[:0]
	b.Stops = b.Stops[:0]
	b.OpaqueCores = 0
	everything := scene.Rect{X: -1e6, Y: -1e6, W: 2e6, H: 2e6}
	b.stack = append(b.stack[:0], clip{rect: everything, bounds: scene.Rect{W: float32(s.Width), H: float32(s.Height)},
		scissor: Scissor{0, 0, int32(s.Width), int32(s.Height)}})
	b.planCull(s)
	for i := range s.Ops {
		op := &s.Ops[i]
		if b.culled[i] {
			continue
		}
		switch op.Kind {
		case scene.OpPushClip:
			c := b.pushed(&b.stack[len(b.stack)-1], op, b.encloses[i])
			if c.masked {
				c.maskSlot = b.maskUsed
				b.maskUsed++
			}
			b.clipShapes = append(b.clipShapes, c.shape)
			b.stack = append(b.stack, c)
		case scene.OpPopClip:
			if len(b.stack) > 1 {
				b.stack = b.stack[:len(b.stack)-1]
				b.clipShapes = b.clipShapes[:len(b.clipShapes)-1]
			}
		case scene.OpFill:
			if op.Rect.Empty() {
				continue
			}
			bw := op.Border
			if op.BorderColor.A == 0 {
				bw = [4]float32{}
			}
			radii := scene.Corners(op.Rect, op.Radii, op.Continuous)
			var inner [4]float32
			if scene.HasBorder(bw) {
				_, inner = scene.InnerRadii(op.Rect, radii, bw)
			}
			dashed := float32(0)
			if op.Dashed {
				dashed = 1
			}
			if op.BorderPaint {
				// Its gradient paints the border.
				dashed += 2
			}
			w := b.wideOf(s, op.Wide)
			c2 := wideColor(w, scene.WideColor2, op.Color2)
			if op.Paint.HasStops() {
				c2 = b.addStops(s, op)
			}
			b.addFill(s, Instance{
				Rect: rect(op.Rect), Radii: radii, Inner: inner,
				Color: wideColor(w, scene.WideColor, op.Color), Color2: c2, Border: wideColor(w, scene.WideBorder, op.BorderColor), Grad: op.Gradient,
				UV:     bw,
				Params: [4]float32{0, dashed, float32(op.Paint), opacity(op.Opacity)},
			}, op)
		case scene.OpShadow:
			if op.Inset {
				// The box in Rect, the hole in UV and Inner, which an
				// inner shadow's 1 in Params[1] tells.
				if op.Cast.Empty() {
					continue
				}
				sigma := op.Blur / 2
				if sigma < 0.5 {
					sigma = 0
				}
				in := Instance{
					Rect: rect(op.Cast), Radii: scene.Corners(op.Cast, op.CastRadii, op.Continuous),
					Color: wideColor(b.wideOf(s, op.Wide), scene.WideColor, op.Color), Params: [4]float32{1, 1, sigma, opacity(op.Opacity)},
				}
				if !op.Rect.Empty() {
					in.UV, in.Inner = rect(op.Rect), scene.Corners(op.Rect, op.Radii, op.Continuous)
				}
				b.add(in, 0, op.Transform)
				continue
			}
			if op.Rect.Empty() {
				continue
			}
			sigma := op.Blur / 2
			if sigma < 0.5 {
				sigma = 0
			}
			in := Instance{
				Rect: rect(op.Rect), Radii: scene.Corners(op.Rect, op.Radii, op.Continuous),
				Color: wideColor(b.wideOf(s, op.Wide), scene.WideColor, op.Color), Params: [4]float32{1, 0, sigma, opacity(op.Opacity)},
			}
			if !op.Cast.Empty() {
				in.UV, in.Inner = rect(op.Cast), scene.Corners(op.Cast, op.CastRadii, op.Continuous)
			}
			b.addShadow(in, op, sigma)
		case scene.OpGlyphs:
			grad := op.Paint == scene.PaintLinear || op.Paint == scene.PaintOklab || op.Paint.HasStops()
			var c1, c2 [4]float32
			if grad {
				w := b.wideOf(s, op.Wide)
				c1, c2 = wideColor(w, scene.WideColor, op.Color), wideColor(w, scene.WideColor2, op.Color2)
				if op.Paint.HasStops() {
					c2 = b.addStops(s, op)
				}
			}
			for _, g := range s.Glyphs[op.Start:op.End] {
				a, kind, contrast := s.MaskAtlas, float32(2), s.Text.Contrast
				switch {
				case g.Colored:
					a, kind = s.ColorAtlas, 3
				case g.Subpixel:
					a, kind, contrast = s.ColorAtlas, 5, s.Text.SubpixelContrast
				}
				if a == nil {
					continue
				}
				boost := float32(0)
				if g.Thin {
					boost = scene.ThinBoost
				}
				aw, ah := float32(a.W), float32(a.H)
				in := Instance{
					Rect:   [4]float32{float32(math.Round(float64(g.X))), float32(math.Round(float64(g.Y))), float32(g.UW), float32(g.VH)},
					Radii:  s.Text.GammaRatios,
					Inner:  [4]float32{contrast, boost},
					UV:     [4]float32{float32(g.U) / aw, float32(g.V) / ah, float32(g.U+g.UW) / aw, float32(g.V+g.VH) / ah},
					Color:  straight(g.Color),
					Params: [4]float32{kind, 0, 0, 1},
				}
				if w := b.wideOf(s, g.Wide); w != nil {
					in.Color = w.Color
				}
				if grad && !g.Colored {
					in.Color, in.Color2, in.Grad = c1, c2, op.Gradient
					in.Params[2], in.Params[3] = float32(op.Paint), opacity(op.Opacity)
				}
				b.add(in, 0, op.Transform.Mul(g.Transform))
			}
		case scene.OpEffect:
			if !op.Rect.Empty() && int(op.Start) < len(s.Effects) && s.Effects[op.Start].Effect != nil {
				b.addEffect(s, op)
			}
		case scene.OpHole:
			if !op.Rect.Empty() {
				b.addHole(op)
			}
		case scene.OpImage:
			img := op.Image
			if img == nil || op.Rect.Empty() || img.W == 0 || img.H == 0 {
				continue
			}
			src := op.Src
			iw, ih := float32(img.W), float32(img.H)
			var tex uintptr
			if op.Blur > 0 && b.BlurredImage != nil {
				// The image's area blurred, which the op samples.
				bk := scene.ImageBlur(scene.ImageArea(op.Src, img.W, img.H), op.Blur)
				w, h := bk.Size()
				if w <= 0 || h <= 0 {
					continue
				}
				tex = b.BlurredImage(img, bk)
				src, iw, ih = bk.ImageSrc(op.Src), float32(w), float32(h)
			} else {
				tex = image(img)
			}
			if tex == 0 {
				continue
			}
			gray, nearest := float32(0), float32(0)
			if op.Grayscale {
				gray = 1
			}
			if op.Pixelated {
				nearest = 1
			}
			tint := [4]float32{1, 1, 1, 1}
			if op.Color != (scene.Color{}) {
				tint = straight(op.Color)
			}
			b.add(Instance{
				Rect:   rect(op.Rect),
				Radii:  scene.Corners(op.Rect, op.Radii, op.Continuous),
				Color:  tint,
				UV:     [4]float32{src.X / iw, src.Y / ih, (src.X + src.W) / iw, (src.Y + src.H) / ih},
				Params: [4]float32{4, gray, nearest, opacity(op.Opacity)},
			}, tex, op.Transform)
		}
	}

	b.dropIdleMasks(idleMaskBytes)
	b.imageTexture = nil
	return s
}

// addStops adds the stops of op's gradient to b.Stops, at most
// scene.MaxStops of them, and returns the instance's Color2 pointing at
// them: the first's index and how many.
func (b *Builder) addStops(s *scene.Scene, op *scene.Op) [4]float32 {
	a, e := int(op.Stops[0]), int(op.Stops[1])
	if a < 0 || e > len(s.Stops) || a >= e {
		return [4]float32{}
	}
	e = min(e, a+scene.MaxStops)
	first := len(b.Stops) / 8
	for _, st := range s.Stops[a:e] {
		lab := scene.Oklab(st.Color)
		c := st.Color
		b.Stops = append(b.Stops, float32(c.R)/255, float32(c.G)/255, float32(c.B)/255, float32(c.A)/255,
			st.At, lab[0], lab[1], lab[2])
	}
	return [4]float32{float32(first), float32(e - a), 0, 0}
}

// StopsSize returns how large the texture of the frame's stops must be,
// in texels, for Stops: StopsWidth across, or 0 by 0 for none.
func (b *Builder) StopsSize() (w, h int) {
	n := len(b.Stops) / 4
	if n == 0 {
		return 0, 0
	}
	return StopsWidth, (n + StopsWidth - 1) / StopsWidth
}

// markEnclosing sets b.encloses for the ops of s.
func (b *Builder) markEnclosing(s *scene.Scene) {
	b.encloses = slices.Grow(b.encloses[:0], len(s.Ops))[:len(s.Ops)]
	clear(b.encloses)
	b.pushes = b.pushes[:0]
	for i := range s.Ops {
		op := &s.Ops[i]
		if op.Kind == scene.OpPopClip {
			if n := len(b.pushes); n > 0 {
				b.pushes = b.pushes[:n-1]
			}
			continue
		}
		transformed := op.Transform.Set
		if op.Kind == scene.OpGlyphs {
			for _, g := range s.Glyphs[op.Start:op.End] {
				transformed = transformed || g.Transform.Set
			}
		}
		// Every clip open around it encloses it, from the innermost out to
		// the first already marked, whose own are.
		for j := len(b.pushes) - 1; transformed && j >= 0 && !b.encloses[b.pushes[j]]; j-- {
			b.encloses[b.pushes[j]] = true
		}
		if op.Kind == scene.OpPushClip {
			b.pushes = append(b.pushes, i)
		}
	}
}

// Release frees the retained clipping resources and instance buffers.
func (b *Builder) Release() { *b = Builder{} }

// add appends an instance within the current clip, starting a batch when
// the scissor rectangle or the image changes, or after an effect's.
func (b *Builder) add(in Instance, image uintptr, m scene.Affine) {
	cur := &b.stack[len(b.stack)-1]
	if !b.prepare(&in, m, cur) {
		return
	}
	if in.Params[0] == 0 {
		b.markFill(&in, m, cur)
	}
	class, continuous := b.classOf(&in)
	n := len(b.Batches)
	sc := b.batchScissor(cur)
	same := n > 0 && b.Batches[n-1].Scissor == sc && b.Batches[n-1].Effect == nil && !b.Batches[n-1].Hole &&
		b.Batches[n-1].Opaque == b.opaque && b.Batches[n-1].ClipMask == b.preparedMask && b.Batches[n-1].Class == class &&
		!(image != 0 && b.Batches[n-1].Image != 0 && b.Batches[n-1].Image != image)
	if same {
		// Instances with and without continuous corners share a batch, with the
		// code of those that need more.
		b.Batches[n-1].Continuous = b.Batches[n-1].Continuous || continuous
	} else {
		b.Batches = append(b.Batches, Batch{Start: len(b.Instances), Scissor: sc, Image: image, ClipMask: b.preparedMask, Opaque: b.opaque, Class: class, Continuous: continuous})
		n++
	}
	if image != 0 {
		b.Batches[n-1].Image = image
	}
	b.Instances = append(b.Instances, in)
	b.Batches[n-1].Count++
}

// plainFill is the shaders' second path for fills, for most of the ones a
// window draws: solid, not transformed and without a border. They then
// compute no gradient, border, dash or transform, which the general path
// does for any kind of fill, and for none the same value (the pixels are
// identical). Instances tell by their fields, as the shaders test them (an
// unset transform, solid paint, no border widths).
//
// markFill sets, for fills not transformed within a clip that is a plain
// rectangle holding the whole quad (with room for the antialiasing around
// it), Transform1.w to 3: the clip then covers every fragment by 1, and the
// shaders skip computing it. With GeneralFill it makes a plain fill take the
// general path, by giving it a border width of a hair, which with a
// transparent border color leaves the pixels as they were.
func (b *Builder) markFill(in *Instance, m scene.Affine, cur *clip) {
	if m.Set {
		return
	}
	if b.GeneralFill && in.Params[2] == float32(scene.PaintSolid) && in.UV == [4]float32{} {
		in.UV = [4]float32{1e-6, 0, 0, 0}
	}
	if !b.ClipSkip || b.NoClipSkip || in.Transform1[3] != 0 || cur.masked || cur.radii != [4]float32{} {
		return
	}
	c, r := cur.rect, scene.Rect{X: in.Rect[0], Y: in.Rect[1], W: in.Rect[2], H: in.Rect[3]}
	const room = 3 // the quad's pixel of antialiasing, and a pixel's reach
	if c.X <= r.X-room && c.Y <= r.Y-room && c.X+c.W >= r.X+r.W+room && c.Y+c.H >= r.Y+r.H+room {
		in.Transform1[3] = 3
	}
}

// shadowSplitMin is the area, in pixels, below which a shadow is not split
// around its caster: a draw call costs more than shading so few fragments.
const shadowSplitMin = 4096

// shadowCore returns the pixels that the box casting a shadow certainly
// covers whole: its rectangle less the corner curves and the antialiased
// edge. A shadow shows only outside its caster, so the shaders write nothing
// there (the coverage of the caster is 1, and the shadow is multiplied by
// what it leaves).
func shadowCore(cast scene.Rect, radii [4]float32, continuous bool) Scissor {
	corner := float32(0)
	for _, r := range radii {
		corner = max(corner, r, -r)
	}
	if continuous {
		corner *= 1.7 // where the curves end, see ContCorner
	}
	m := corner + 2
	return Scissor{
		Left:   int32(math.Ceil(float64(cast.X + m))),
		Top:    int32(math.Ceil(float64(cast.Y + m))),
		Right:  int32(math.Floor(float64(cast.X + cast.W - m))),
		Bottom: int32(math.Floor(float64(cast.Y + cast.H - m))),
	}
}

func (s Scissor) intersect(o Scissor) Scissor {
	return Scissor{max(s.Left, o.Left), max(s.Top, o.Top), min(s.Right, o.Right), min(s.Bottom, o.Bottom)}
}

func (s Scissor) area() int64 {
	if s.Empty() {
		return 0
	}
	return int64(s.Right-s.Left) * int64(s.Bottom-s.Top)
}

// addShadow appends the instance of a shadow. A large shadow of a box
// (Cast) is drawn as up to four instances around the core of that box,
// each with a scissor rectangle of its own, so that the renderers shade no
// fragment of the core, where the shadow is multiplied by zero and the
// pixels are the same either way. A shadow of a box that fills the window
// would otherwise shade, with its blur, all of it, once for each layer of
// an elevation.
func (b *Builder) addShadow(in Instance, op *scene.Op, sigma float32) {
	cur := &b.stack[len(b.stack)-1]
	if b.NoShadowSplit || op.Cast.Empty() || op.Transform.Set || cur.scissor.Empty() {
		b.add(in, 0, op.Transform)
		return
	}
	core := shadowCore(op.Cast, scene.Corners(op.Cast, op.CastRadii, op.Continuous), op.Continuous)
	reach := 3 * sigma // what the vertex shader grows the quad by; shadowSupport bounds it
	quad := Scissor{
		Left: int32(math.Floor(float64(op.Rect.X - reach))), Top: int32(math.Floor(float64(op.Rect.Y - reach))),
		Right: int32(math.Ceil(float64(op.Rect.X + op.Rect.W + reach))), Bottom: int32(math.Ceil(float64(op.Rect.Y + op.Rect.H + reach))),
	}.intersect(cur.scissor)
	if !worthSplitting(core, quad) {
		b.add(in, 0, op.Transform)
		return
	}
	b.addAround(in, op.Transform, core)
}

// addAround appends in as up to four instances around core, each with a
// scissor rectangle of its own: they tile the current clip's scissor
// rectangle less core, whatever the renderer's quad is (core must lie inside
// the quad). The pixels are the same as drawing it whole, for an instance
// that leaves the destination as it is within core.
func (b *Builder) addAround(in Instance, m scene.Affine, core Scissor) {
	cur := &b.stack[len(b.stack)-1]
	whole := cur.scissor
	hole := core.intersect(whole)
	strips := [4]Scissor{
		{whole.Left, whole.Top, whole.Right, hole.Top},
		{whole.Left, hole.Bottom, whole.Right, whole.Bottom},
		{whole.Left, hole.Top, hole.Left, hole.Bottom},
		{hole.Right, hole.Top, whole.Right, hole.Bottom},
	}
	defer b.scissorBatches(Scissor{}, false)
	for _, st := range strips {
		if st.Empty() {
			continue
		}
		b.scissorBatches(st, true)
		b.add(in, 0, m)
	}
}

// scissorBatches makes the batches that add makes from now on scissored to sc,
// within the clip's scissor rectangle, or, when on is false, to the clip's.
// The clip's own scissor rectangle stays as it is: for a masked clip it is also
// the area of the clip's mask and the mapping the shader samples it by.
func (b *Builder) scissorBatches(sc Scissor, on bool) { b.split, b.splitScissor = on, sc }

// batchScissor returns the scissor rectangle of the batch that add starts or
// extends within the clip cur.
func (b *Builder) batchScissor(cur *clip) Scissor {
	if b.split {
		return b.splitScissor
	}
	return cur.scissor
}

// opaqueMin is the area, in pixels, below which a fill is not drawn as an
// opaque core with edges around it: a draw call costs more than the
// overdraw it saves.
const opaqueMin = 16384

// addOpaque appends the instance of an opaque fill as one instance in a
// batch of opaque cores, scissored to core (what occluderOf finds the fill to
// paint with its own opaque color, coverage 1 and alpha 1: blending leaves
// nothing of the destination there, so renderers write the shader's color as
// it is), and as up to four blended pieces around it, for the antialiased
// edge and the corners. The pixels are the same as drawing it whole.
func (b *Builder) addOpaque(in Instance, m scene.Affine, core Scissor) {
	b.scissorBatches(core, true)
	b.opaque = true
	b.add(in, 0, m)
	b.scissorBatches(Scissor{}, false)
	b.opaque = false
	b.OpaqueCores++
	b.addAround(in, m, core)
}

// worthSplitting reports whether drawing around core, a draw call for each
// side, beats shading it: core within the quad is large, and a fair part of
// the quad.
func worthSplitting(core, quad Scissor) bool {
	hole := core.intersect(quad)
	a := hole.area()
	return a >= shadowSplitMin && 4*a >= quad.area()
}

// ringCore returns the pixels that a fill of the rectangle r with corner
// radii (negative when continuous, as in Instance) and border widths bw
// (top, right, bottom, left) certainly covers whole, and whose inner edge
// does too: the outer shape's coverage and the inner one's are both 1, so
// the border's coverage, their difference, is 0 there.
func ringCore(r scene.Rect, radii, bw [4]float32, continuous bool) Scissor {
	inner, innerRadii := scene.InnerRadii(r, radii, bw)
	if inner.W <= 0 || inner.H <= 0 {
		return Scissor{}
	}
	return shadowCore(r, radii, continuous).intersect(shadowCore(inner, innerRadii, continuous))
}

// addFill appends the instance of a fill. A fill that draws only its border
// (a transparent solid color, a border of some width: focus rings, outlines
// and card borders around window-sized panels) is drawn as up to four
// instances around the core inside its border, as addShadow does, so that
// the renderers shade no fragment where the shader writes exactly nothing: the
// outer and inner coverages are both 1 there, the border's is their
// difference, and the color is transparent. Without it a 2 pixel ring around
// a 5 Mpx panel shaded the whole panel, twice over, through the general
// path. The pixels are identical.
//
// An opaque fill with a large core is drawn as that core, in a batch
// renderers draw without blending, and blended pieces around it (addOpaque).
func (b *Builder) addFill(s *scene.Scene, in Instance, op *scene.Op) {
	cur := &b.stack[len(b.stack)-1]
	if b.Opaque && in.Color[3] == 1 && !op.Rect.Empty() && !cur.scissor.Empty() {
		if core := b.occluderOf(s, op, cur); !core.Empty() {
			quad := Scissor{
				Left: int32(math.Floor(float64(op.Rect.X - 1))), Top: int32(math.Floor(float64(op.Rect.Y - 1))),
				Right: int32(math.Ceil(float64(op.Rect.X + op.Rect.W + 1))), Bottom: int32(math.Ceil(float64(op.Rect.Y + op.Rect.H + 1))),
			}.intersect(cur.scissor)
			if a := core.area(); a >= opaqueMin && 2*a >= quad.area() {
				b.addOpaque(in, op.Transform, core)
				return
			}
		}
	}
	if b.NoRingSplit || op.Transform.Set || op.Paint != scene.PaintSolid || in.Color[3] != 0 || !scene.HasBorder(in.UV) ||
		op.Rect.Empty() || cur.scissor.Empty() {
		b.add(in, 0, op.Transform)
		return
	}
	// A fill's quad reaches a pixel out of its rectangle (see the vertex shader).
	quad := Scissor{
		Left: int32(math.Floor(float64(op.Rect.X - 1))), Top: int32(math.Floor(float64(op.Rect.Y - 1))),
		Right: int32(math.Ceil(float64(op.Rect.X + op.Rect.W + 1))), Bottom: int32(math.Ceil(float64(op.Rect.Y + op.Rect.H + 1))),
	}.intersect(cur.scissor)
	core := ringCore(op.Rect, in.Radii, in.UV, op.Continuous)
	if !worthSplitting(core, quad) {
		b.add(in, 0, op.Transform)
		return
	}
	b.addAround(in, op.Transform, core)
}

// addEffect appends the instance of an effect within the current clip, in
// a batch of its own, which reads its backdrop first when it reads one.
func (b *Builder) addEffect(s *scene.Scene, op *scene.Op) {
	cur := &b.stack[len(b.stack)-1]
	if cur.scissor.Empty() {
		return
	}
	fx := &s.Effects[op.Start]
	if fx.Masked() {
		b.addMaskedEffect(s, op, fx, cur)
		return
	}
	p := &fx.Params
	in := Instance{
		Rect: rect(op.Rect), Radii: scene.Corners(op.Rect, op.Radii, op.Continuous),
		Inner: p[0], Color: p[1], Color2: p[2], Border: p[3], Grad: p[4],
		Clip: rect(cur.rect), ClipRadii: cur.radii,
		Params: [4]float32{6, 0, 1, opacity(op.Opacity)},
	}
	if !b.prepare(&in, op.Transform, cur) {
		return
	}
	batch := Batch{Start: len(b.Instances), Count: 1, Scissor: cur.scissor, Effect: fx.Effect, ClipMask: b.preparedMask}
	if fx.Effect.Shader != nil {
		batch.Shader, batch.Uniforms = fx, b.uniformsOf(fx)
		for i, img := range fx.Images {
			if img != nil && img.W > 0 && img.H > 0 && b.imageTexture != nil {
				batch.Textures[i] = b.imageTexture(img)
			}
		}
	}
	if fx.Effect.Backdrop {
		bk := scene.BackdropOf(op.Transform.Bounds(op.Rect), fx.Blur, s.Width, s.Height)
		if bk.Area.Empty() {
			return
		}
		b.Backdrops = append(b.Backdrops, bk)
		batch.Backdrop = len(b.Backdrops)
		tw, th := bk.Size()
		in.UV = [4]float32{float32(bk.Area.Min.X), float32(bk.Area.Min.Y), float32(tw), float32(th)}
		in.Params[2] = float32(bk.Down)
	}
	b.Batches = append(b.Batches, batch)
	b.Instances = append(b.Instances, in)
}

// addMaskedEffect appends the instances of a shader's effect filling mask
// glyphs (scene.EffectOp.Glyphs), in a batch of their own: each glyph's,
// as OpGlyphs has it, with 1 in Params[1], and the point the shader's
// positions start from in Grad.
func (b *Builder) addMaskedEffect(s *scene.Scene, op *scene.Op, fx *scene.EffectOp, cur *clip) {
	a := s.MaskAtlas
	if a == nil || fx.Effect.Shader == nil || fx.Effect.Backdrop {
		return
	}
	batch := Batch{Start: len(b.Instances), Scissor: cur.scissor, Effect: fx.Effect, Shader: fx, Uniforms: b.uniformsOf(fx)}
	for i, img := range fx.Images {
		if img != nil && img.W > 0 && img.H > 0 && b.imageTexture != nil {
			batch.Textures[i] = b.imageTexture(img)
		}
	}
	aw, ah := float32(a.W), float32(a.H)
	first := true
	for _, g := range s.Glyphs[fx.Glyphs[0]:fx.Glyphs[1]] {
		if g.Colored || g.Subpixel {
			continue
		}
		boost := float32(0)
		if g.Thin {
			boost = scene.ThinBoost
		}
		in := Instance{
			Rect:   [4]float32{float32(math.Round(float64(g.X))), float32(math.Round(float64(g.Y))), float32(g.UW), float32(g.VH)},
			Radii:  s.Text.GammaRatios,
			Inner:  [4]float32{s.Text.Contrast, boost},
			UV:     [4]float32{float32(g.U) / aw, float32(g.V) / ah, float32(g.U+g.UW) / aw, float32(g.V+g.VH) / ah},
			Grad:   [4]float32{fx.Origin[0], fx.Origin[1], 1, 0},
			Params: [4]float32{6, 1, 0, opacity(op.Opacity)},
		}
		if !b.prepare(&in, op.Transform.Mul(g.Transform), cur) {
			continue
		}
		if first {
			batch.ClipMask, first = b.preparedMask, false
		}
		b.Instances = append(b.Instances, in)
		batch.Count++
	}
	if batch.Count > 0 {
		b.Batches = append(b.Batches, batch)
	}
}

// uniformsOf returns the words of fx's uniforms to draw with: its wide
// ones, for a wide gamut, when it has them.
func (b *Builder) uniformsOf(fx *scene.EffectOp) []uint32 {
	if b.Wide && fx.WideUniforms != nil {
		return fx.WideUniforms
	}
	return fx.Uniforms
}

// addHole appends a hole within the current clip, in a batch of holes:
// an opaque fill, whose coverage the renderers take away.
func (b *Builder) addHole(op *scene.Op) {
	cur := &b.stack[len(b.stack)-1]
	in := Instance{
		Rect: rect(op.Rect), Radii: scene.Corners(op.Rect, op.Radii, op.Continuous),
		Color:  [4]float32{0, 0, 0, 1},
		Params: [4]float32{0, 0, float32(scene.PaintSolid), opacity(op.Opacity)},
	}
	if !b.prepare(&in, op.Transform, cur) {
		return
	}
	class, continuous := b.classOf(&in)
	n := len(b.Batches)
	if n > 0 && b.Batches[n-1].Hole && b.Batches[n-1].Scissor == cur.scissor && b.Batches[n-1].ClipMask == b.preparedMask {
		if b.Batches[n-1].Class != class {
			b.Batches[n-1].Class = ClassGeneral // holes are few
		}
		b.Batches[n-1].Continuous = b.Batches[n-1].Continuous || continuous
	} else {
		b.Batches = append(b.Batches, Batch{Start: len(b.Instances), Scissor: cur.scissor, Hole: true, ClipMask: b.preparedMask, Class: class, Continuous: continuous})
	}
	b.Instances = append(b.Instances, in)
	b.Batches[len(b.Batches)-1].Count++
}

// classOf is classOf, or the general class under NoClasses.
func (b *Builder) classOf(in *Instance) (Class, bool) {
	if !b.Classes || b.NoClasses {
		return ClassGeneral, false
	}
	c, cont := classOf(in)
	if b.GeneralClasses&(1<<c) != 0 {
		return ClassGeneral, false
	}
	return c, cont
}

func rect(r scene.Rect) [4]float32 { return [4]float32{r.X, r.Y, r.W, r.H} }

func straight(c scene.Color) [4]float32 {
	return [4]float32{float32(c.R) / 255, float32(c.G) / 255, float32(c.B) / 255, float32(c.A) / 255}
}

// wideOf returns the colors outside the sRGB gamut of an op or a glyph of
// s whose Wide is i, or nil when it has none or b does not draw them.
func (b *Builder) wideOf(s *scene.Scene, i uint16) *scene.WideColors {
	if !b.Wide || i == 0 || int(i) > len(s.Wide) {
		return nil
	}
	return &s.Wide[i-1]
}

// wideColor returns the color of w that bit names, or c when w has none.
func wideColor(w *scene.WideColors, bit scene.WideSet, c scene.Color) [4]float32 {
	if w == nil || w.Set&bit == 0 {
		return straight(c)
	}
	switch bit {
	case scene.WideColor:
		return w.Color
	case scene.WideColor2:
		return w.Color2
	}
	return w.Border
}

func opacity(o float32) float32 {
	if o == 0 {
		return 1
	}
	return o
}
