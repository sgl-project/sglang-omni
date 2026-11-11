package gpu

import (
	"math"

	"github.com/egoist/mygo/internal/scene"
)

// Occlusion culling. A later opaque fill replaces what the ops before it
// drew inside the part of its shape where its coverage is exactly 1, and
// blending it leaves nothing of the destination (the destination times
// one minus the source's alpha, plus the source), so an op whose pixels all
// lie in that part never shows. Windows stack such fills (a page, a panel
// on it, a card on that), and a renderer shading every layer of a window
// spends its fill rate on pixels that are overwritten: 1.9 ms of GPU time
// for each full-window layer on an M1 Pro at 3456×2000.
//
// planCull finds those ops in a pass over the scene backwards, which keeps
// the few largest occluders seen so far, and Build leaves the ops out. The
// result is identical to drawing them; the rules that make it so are in
// occluderOf and visibleOf.

// cullOccluders is how many occluders the backward pass keeps. The largest
// cover the most; a bound keeps the pass linear in the ops.
const cullOccluders = 8

// cullMargin returns how far, in pixels, the antialiased edge of an op with
// the transform m reaches out of its rectangle: a pixel whose center is
// within half a pixel of the edge is covered in part, and the shader draws
// one more around a quad it maps. A transformed quad is grown by the map's
// reach in the shader (see its vertex stage), a pixel or two.
func cullMargin(m scene.Affine) float32 {
	if m.Set {
		return 3
	}
	return 1
}

type cullInfo struct {
	// vis holds what the op can change: its bounds within its clip's
	// scissor rectangle. Empty for ops that cannot be culled.
	vis Scissor
	// occ is the part of the frame the op, as an occluder, certainly
	// covers whole: empty unless it qualifies.
	occ Scissor
	// barrier tells that the op reads the frame drawn so far (an effect
	// with a backdrop), so what comes after it cannot hide what came
	// before.
	barrier bool
}

// shadowSupport returns the rectangle, in the shadow's own units, outside
// which the shadow op draws nothing: its box (the caster's, for an inner
// shadow), grown by the blur. The vertex shaders grow the quad of a shadow by
// 3 sigma + 1 = 1.5 blur + 1 before mapping it, and the blur is evaluated in
// those units, so a transform scales the reach with the rest of the shape:
// the margin is added here, before the map, not in device pixels after it.
// 2 blur + 1 has slack over the shader's growth. addShadow's reach (3 sigma)
// is the same bound, as the shader has it.
func shadowSupport(op *scene.Op) scene.Rect {
	r := op.Rect
	if op.Inset {
		r = op.Cast
	}
	// Negative blur is a hard shadow, as Build clamps its sigma to zero.
	g := 2*max(op.Blur, 0) + 1
	return scene.Rect{X: r.X - g, Y: r.Y - g, W: r.W + 2*g, H: r.H + 2*g}
}

// planCull sets b.culled to the ops that later opaque fills cover whole.
func (b *Builder) planCull(s *scene.Scene) {
	n := len(s.Ops)
	if cap(b.culled) < n {
		b.culled = make([]bool, n)
	}
	b.culled = b.culled[:n]
	clear(b.culled)
	b.Culled = 0
	if b.NoCull || n < 2 {
		return
	}
	// Sprite and glyph-only scenes have nothing that can hide an earlier op.
	// Avoid their bounds pass and per-op planning storage entirely. This is
	// deliberately conservative: clipping and wide alpha are checked below.
	candidate := false
	for i := range s.Ops {
		op := &s.Ops[i]
		if op.Kind == scene.OpFill && op.Paint == scene.PaintSolid && op.Color.A == 255 &&
			!op.Transform.Set && (op.Opacity == 0 || op.Opacity == 1) {
			candidate = true
			break
		}
	}
	if !candidate {
		return
	}
	if cap(b.plan) < n {
		b.plan = make([]cullInfo, n)
	}
	plan := b.plan[:n]
	clear(plan)

	// Forwards: what each op can change, and what it certainly covers,
	// with the clips as Build keeps them.
	root := clip{rect: scene.Rect{X: -1e6, Y: -1e6, W: 2e6, H: 2e6}, bounds: scene.Rect{W: float32(s.Width), H: float32(s.Height)},
		scissor: Scissor{0, 0, int32(s.Width), int32(s.Height)}, maskSlot: -1}
	stack := append(b.planStack[:0], root)
	haveOccluder := false
	for i := range s.Ops {
		op := &s.Ops[i]
		cur := &stack[len(stack)-1]
		switch op.Kind {
		case scene.OpPushClip:
			stack = append(stack, b.pushed(cur, op, b.encloses[i]))
		case scene.OpPopClip:
			if len(stack) > 1 {
				stack = stack[:len(stack)-1]
			}
		case scene.OpEffect:
			if int(op.Start) < len(s.Effects) && s.Effects[op.Start].Effect != nil && s.Effects[op.Start].Effect.Backdrop {
				plan[i].barrier = true
			}
		case scene.OpFill:
			if op.Rect.Empty() {
				continue
			}
			plan[i].vis = visibleOf(cur.scissor, op.Transform.Bounds(op.Rect), cullMargin(op.Transform))
			plan[i].occ = b.occluderOf(s, op, cur)
			haveOccluder = haveOccluder || !plan[i].occ.Empty()
		case scene.OpShadow:
			if op.Rect.Empty() && !(op.Inset && !op.Cast.Empty()) {
				continue
			}
			plan[i].vis = visibleOf(cur.scissor, op.Transform.Bounds(shadowSupport(op)), cullMargin(op.Transform))
		case scene.OpHole, scene.OpImage:
			if !op.Rect.Empty() {
				plan[i].vis = visibleOf(cur.scissor, op.Transform.Bounds(op.Rect), cullMargin(op.Transform))
			}
		case scene.OpGlyphs:
			// The run's quads are measured in the backward pass, when
			// an occluder can hide them.
			plan[i].vis = cur.scissor
		}
	}
	b.planStack = stack[:0]
	if !haveOccluder {
		return
	}

	// Backwards: an op inside an occluder after it is left out.
	var occ [cullOccluders]Scissor
	var area [cullOccluders]int64
	count := 0
	for i := n - 1; i >= 0; i-- {
		op := &s.Ops[i]
		pi := &plan[i]
		if pi.barrier {
			count = 0
			continue
		}
		if count > 0 {
			vis := pi.vis
			if op.Kind == scene.OpGlyphs {
				vis = glyphsVisible(s, op, pi.vis)
			}
			if !vis.Empty() {
				for k := 0; k < count; k++ {
					if occ[k].contains(vis) {
						b.culled[i] = true
						b.Culled++
						break
					}
				}
			}
		}
		if pi.occ.Empty() {
			continue
		}
		a := pi.occ.area()
		if count < cullOccluders {
			occ[count], area[count] = pi.occ, a
			count++
			continue
		}
		small := 0
		for k := 1; k < count; k++ {
			if area[k] < area[small] {
				small = k
			}
		}
		if a > area[small] {
			occ[small], area[small] = pi.occ, a
		}
	}
}

func (s Scissor) contains(o Scissor) bool {
	return s.Left <= o.Left && s.Top <= o.Top && s.Right >= o.Right && s.Bottom >= o.Bottom
}

// visibleOf returns the pixels that something within bounds, grown by margin
// for its antialiased edge, can change within the scissor rectangle sc.
func visibleOf(sc Scissor, bounds scene.Rect, margin float32) Scissor {
	if sc.Empty() {
		return Scissor{}
	}
	x0 := math.Max(math.Floor(float64(bounds.X-margin)), float64(sc.Left))
	y0 := math.Max(math.Floor(float64(bounds.Y-margin)), float64(sc.Top))
	x1 := math.Min(math.Ceil(float64(bounds.X+bounds.W+margin)), float64(sc.Right))
	y1 := math.Min(math.Ceil(float64(bounds.Y+bounds.H+margin)), float64(sc.Bottom))
	if !(x1 > x0 && y1 > y0) { // also NaN
		return Scissor{}
	}
	return Scissor{int32(x0), int32(y0), int32(x1), int32(y1)}
}

// glyphsVisible returns what a run of glyphs, an OpGlyphs, can change: the
// union of its glyphs' quads within the scissor rectangle sc of its clip.
func glyphsVisible(s *scene.Scene, op *scene.Op, sc Scissor) Scissor {
	if sc.Empty() || op.Start < 0 || int(op.End) > len(s.Glyphs) || op.Start >= op.End {
		return Scissor{}
	}
	var u scene.Rect
	first := true
	margin := float32(0)
	for _, g := range s.Glyphs[op.Start:op.End] {
		m := op.Transform.Mul(g.Transform)
		margin = max(margin, cullMargin(m))
		r := m.Bounds(scene.Rect{X: float32(math.Round(float64(g.X))), Y: float32(math.Round(float64(g.Y))), W: float32(g.UW), H: float32(g.VH)})
		if first {
			u, first = r, false
			continue
		}
		x0, y0 := min(u.X, r.X), min(u.Y, r.Y)
		x1, y1 := max(u.X+u.W, r.X+r.W), max(u.Y+u.H, r.Y+r.H)
		u = scene.Rect{X: x0, Y: y0, W: x1 - x0, H: y1 - y0}
	}
	if first {
		return Scissor{}
	}
	return visibleOf(sc, u, margin)
}

// occluderOf returns the pixels that the fill op, within cur, certainly
// paints with its own opaque color, or the empty rectangle. That takes a
// solid paint of alpha 1 and opacity 1, untransformed (so that the
// coverage is the plain rectangle's), under a clip that is a plain
// rectangle too (no mask, no rounded corners: their antialiased edge
// covers by less than 1). It excludes the antialiased edge and the curves
// of the corners, as a shadow's caster does (shadowCore), and a border's
// width, so the part left is all the fill's color; except for the sides
// of a rectangle that lie outside the scissor rectangle, where no edge
// shows.
func (b *Builder) occluderOf(s *scene.Scene, op *scene.Op, cur *clip) Scissor {
	if op.Transform.Set || op.Paint != scene.PaintSolid || op.Color.A != 255 || (op.Opacity != 0 && op.Opacity != 1) ||
		cur.masked || cur.radii != [4]float32{} || cur.scissor.Empty() {
		return Scissor{}
	}
	if w := b.wideOf(s, op.Wide); w != nil && w.Set&scene.WideColor != 0 && w.Color[3] != 1 {
		return Scissor{}
	}
	radii := scene.Corners(op.Rect, op.Radii, op.Continuous)
	core := shadowCore(op.Rect, radii, op.Continuous)
	bw := [4]float32{}
	if op.BorderColor.A != 0 {
		bw = op.Border
	}
	if scene.HasBorder(bw) {
		w := int32(math.Ceil(float64(max(bw[0], bw[1], bw[2], bw[3]))))
		core = Scissor{core.Left + w, core.Top + w, core.Right - w, core.Bottom - w}
	}
	if radii == [4]float32{} {
		// A rectangle with square corners whose side lies at or beyond the
		// scissor rectangle (a layer as large as the window) has no edge
		// to antialias there: each pixel of the scissor rectangle is inside
		// it, and covered by 1 exactly.
		r, sc := op.Rect, cur.scissor
		if r.X <= float32(sc.Left) && bw[3] == 0 {
			core.Left = sc.Left
		}
		if r.Y <= float32(sc.Top) && bw[0] == 0 {
			core.Top = sc.Top
		}
		if r.X+r.W >= float32(sc.Right) && bw[1] == 0 {
			core.Right = sc.Right
		}
		if r.Y+r.H >= float32(sc.Bottom) && bw[2] == 0 {
			core.Bottom = sc.Bottom
		}
	}
	// The clip's rectangle covers whole the pixels inside it.
	inner := Scissor{
		int32(math.Ceil(float64(cur.rect.X))), int32(math.Ceil(float64(cur.rect.Y))),
		int32(math.Floor(float64(cur.rect.X + cur.rect.W))), int32(math.Floor(float64(cur.rect.Y + cur.rect.H))),
	}
	return core.intersect(cur.scissor).intersect(inner)
}
