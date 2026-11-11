package raster

import (
	"image"
	"math"
	"runtime"

	"github.com/egoist/mygo/internal/scene"
)

// Renderer draws the successive scenes of a window into Image, redrawing
// only where a scene differs from the one before: a blinking caret, a
// ticking clock or a button under the pointer redraws only itself.
//
// It compares the operations of the two scenes from both ends, so a change
// in the middle of the display list redraws the boxes of the operations
// that changed, clipped as they draw. Glyphs whose pixels changed in an
// atlas, and images whose pixels changed, count as changed.
type Renderer struct {
	// Image holds the last scene drawn, in mem.
	Image Image
	mem   *pixels

	// d draws the bands of large damage.
	d drawer
	// valid tells that the last scene, of w×h pixels, is remembered to
	// compare the next with.
	valid   bool
	w, h    int
	clear   scene.Color
	ops     []scene.Op
	glyphs  []scene.Glyph
	stops   []scene.Stop
	effects []scene.EffectOp
	words   []uint32
	// bounds is where each operation of the last scene drew; versions has
	// the versions of its images.
	bounds, next []image.Rectangle
	versions     []uint64
	mask, color  atlasMark
	damage       []image.Rectangle
	clips        []image.Rectangle
}

// atlasMark is the state of an atlas a scene drew from.
type atlasMark struct {
	atlas        *scene.Atlas
	gen, version uint64
}

// changes returns the rectangles of a that changed since the mark, or
// false when everything may have.
func (m *atlasMark) changes(a *scene.Atlas) ([]image.Rectangle, bool) {
	if a == nil || m.atlas == nil {
		return nil, a == m.atlas
	}
	if a != m.atlas {
		return nil, false
	}
	rects, full := a.Changes(m.gen, m.version)
	return rects, !full
}

func (m *atlasMark) set(a *scene.Atlas) {
	m.atlas = a
	if a != nil {
		m.gen, m.version = a.Generation(), a.Version()
	}
}

// Render draws s, and returns the rectangles of Image it changed. s may
// change once Render returns.
func (r *Renderer) Render(s *scene.Scene) []image.Rectangle {
	drawCanvases(s)
	if r.whole(s) || r.Image.W != s.Width || r.Image.H != s.Height {
		// A new size, or Image was released (Release).
		r.resize(s.Width, s.Height)
		r.damage = append(r.damage[:0], image.Rect(0, 0, s.Width, s.Height))
	}
	for _, d := range r.damage {
		r.d.draw(&r.Image, s, d, r.next)
	}
	r.remember(s)
	return r.damage
}

// pixels is the memory of a Renderer's image (allocPixels).
type pixels struct {
	b       []byte
	mapped  bool
	cleanup runtime.Cleanup
}

// resize makes Image w×h, reusing its memory when it can.
func (r *Renderer) resize(w, h int) {
	n := 4 * w * h
	if r.mem == nil || len(r.mem.b) < n {
		r.mem.free()
		r.mem = allocPixels(n)
	}
	r.Image = Image{W: w, H: h, Stride: 4 * w, Pix: r.mem.b[:n]}
}

// Release frees the image and what Render remembers: the next Render
// draws everything. Image must not be used meanwhile.
func (r *Renderer) Release() {
	r.mem.free()
	*r = Renderer{}
}

func area(rects []image.Rectangle) int {
	n := 0
	for _, d := range rects {
		n += d.Dx() * d.Dy()
	}
	return n
}

// whole sets the damage of s against the last scene, and reports whether
// s must be drawn whole: there is none, or s changes too much of it.
func (r *Renderer) whole(s *scene.Scene) bool {
	r.next = r.opBounds(s, r.next[:0])
	return !r.valid || s.Width != r.w || s.Height != r.h || s.Clear != r.clear || !r.diff(s)
}

// Invalidate makes the next Render draw everything.
func (r *Renderer) Invalidate() { r.valid = false }

// diff sets the damage between the last scene and s, and reports false
// when s must be drawn whole.
func (r *Renderer) diff(s *scene.Scene) bool {
	maskRects, ok := r.mask.changes(s.MaskAtlas)
	if !ok {
		return false
	}
	colorRects, ok := r.color.changes(s.ColorAtlas)
	if !ok {
		return false
	}
	same := func(i, j int) bool { return r.same(i, s, j, maskRects, colorRects) }
	n, m := len(r.ops), len(s.Ops)
	pre := 0
	for pre < n && pre < m && same(pre, pre) {
		pre++
	}
	suf := 0
	for suf < n-pre && suf < m-pre && same(n-1-suf, m-1-suf) {
		suf++
	}
	r.damage = r.damage[:0]
	if n == m {
		for k := pre; k < n-suf; k++ {
			if !same(k, k) {
				r.damage = addRect(r.damage, r.bounds[k])
				r.damage = addRect(r.damage, r.next[k])
			}
		}
	} else {
		for i := pre; i < n-suf; i++ {
			r.damage = addRect(r.damage, r.bounds[i])
		}
		for j := pre; j < m-suf; j++ {
			r.damage = addRect(r.damage, r.next[j])
		}
	}
	r.addBackdrops(s)
	// Past half the window, drawing it whole costs about the same.
	return area(r.damage)*2 < s.Width*s.Height
}

// addBackdrops adds to the damage the effects of s that read their
// backdrops it meets, with those backdrops: draw reads a backdrop after
// drawing the operations before its effect within the area drawn, which
// must hold all of it, and then no other area may draw the effect. So the
// damage becomes rectangles apart from each other.
func (r *Renderer) addBackdrops(s *scene.Scene) {
	if len(s.Effects) == 0 || len(r.damage) == 0 {
		return
	}
	for changed := true; changed; {
		changed = false
		for i := range s.Ops {
			op := &s.Ops[i]
			if op.Kind != scene.OpEffect || int(op.Start) >= len(s.Effects) || r.next[i].Empty() {
				continue
			}
			fx := &s.Effects[op.Start]
			if fx.Effect == nil || !fx.Effect.Backdrop {
				continue
			}
			need := scene.BackdropOf(op.Transform.Bounds(op.Rect), fx.Blur, s.Width, s.Height).Area.Union(r.next[i])
			for _, d := range r.damage {
				if d.Overlaps(need) && !need.In(d) {
					r.damage = addRect(r.damage, need)
					changed = true
					break
				}
			}
		}
		// Rectangles that overlap become one.
		for i := 0; i < len(r.damage); i++ {
			for j := i + 1; j < len(r.damage); j++ {
				if r.damage[i].Overlaps(r.damage[j]) {
					r.damage[i] = r.damage[i].Union(r.damage[j])
					r.damage = append(r.damage[:j], r.damage[j+1:]...)
					changed = true
					j = i
				}
			}
		}
	}
}

// same reports whether operation i of the last scene draws what operation
// j of s does.
func (r *Renderer) same(i int, s *scene.Scene, j int, maskRects, colorRects []image.Rectangle) bool {
	a, b := r.ops[i], s.Ops[j]
	// Glyphs and effects are compared by what Start and End point at.
	sa, sb := a.Start, b.Start
	ga, gb := r.glyphs[:0], s.Glyphs[:0]
	if a.Kind == scene.OpGlyphs && b.Kind == scene.OpGlyphs {
		ga, gb = r.glyphs[a.Start:a.End], s.Glyphs[b.Start:b.End]
	} else if a.Kind == scene.OpEffect && b.Kind == scene.OpEffect && int(sa) < len(r.effects) && int(sb) < len(s.Effects) {
		// A shader's effect filling mask glyphs.
		x, y := r.effects[sa].Glyphs, s.Effects[sb].Glyphs
		ga, gb = r.glyphs[x[0]:x[1]], s.Glyphs[y[0]:y[1]]
	}
	a.Start, a.End, b.Start, b.End = 0, 0, 0, 0
	// Gradients' stops are compared by what Stops point at.
	if !sameStops(r.stops, a.Stops, s.Stops, b.Stops) {
		return false
	}
	a.Stops, b.Stops = [2]int32{}, [2]int32{}
	// The CPU draws the sRGB colors, not those outside its gamut, which
	// Wide points at in each scene's own table.
	a.Wide, b.Wide = 0, 0
	if a != b {
		return false
	}
	if !sameGlyphs(ga, gb, maskRects, colorRects) {
		return false
	}
	switch b.Kind {
	case scene.OpImage:
		if b.Image != nil && b.Image.Version() != r.versions[i] {
			return false
		}
	case scene.OpEffect:
		if int(sa) >= len(r.effects) || int(sb) >= len(s.Effects) || !r.effects[sa].Same(&s.Effects[sb]) {
			return false
		}
	}
	return true
}

// sameStops reports whether stops sa[a[0]:a[1]] of the last scene are
// those sb[b[0]:b[1]] of the new one.
func sameStops(sa []scene.Stop, a [2]int32, sb []scene.Stop, b [2]int32) bool {
	if a[1]-a[0] != b[1]-b[0] {
		return false
	}
	if a[1] <= a[0] {
		return true
	}
	if int(a[1]) > len(sa) || int(b[1]) > len(sb) || a[0] < 0 || b[0] < 0 {
		return false
	}
	for i := range a[1] - a[0] {
		if sa[a[0]+i] != sb[b[0]+i] {
			return false
		}
	}
	return true
}

// sameGlyphs reports whether glyphs ga of the last scene draw what gb do,
// none of them in an area of the atlases that changed.
func sameGlyphs(ga, gb []scene.Glyph, maskRects, colorRects []image.Rectangle) bool {
	if len(ga) != len(gb) {
		return false
	}
	for k := range gb {
		g := &gb[k]
		if ga[k] != *g {
			x := ga[k]
			x.Wide = g.Wide
			if x != *g {
				return false
			}
		}
		rects := maskRects
		if g.Colored {
			rects = colorRects
		}
		at := image.Rect(int(g.U), int(g.V), int(g.U)+int(g.UW), int(g.V)+int(g.VH))
		for _, c := range rects {
			if c.Overlaps(at) {
				return false
			}
		}
	}
	return true
}

// addRect adds a rectangle to damage, merged with one it touches.
func addRect(damage []image.Rectangle, b image.Rectangle) []image.Rectangle {
	if b.Empty() {
		return damage
	}
	for i, d := range damage {
		if d.Inset(-8).Overlaps(b) {
			damage[i] = d.Union(b)
			return damage
		}
	}
	damage = append(damage, b)
	if len(damage) > 8 {
		u := damage[0]
		for _, d := range damage[1:] {
			u = u.Union(d)
		}
		damage = append(damage[:0], u)
	}
	return damage
}

// opBounds appends where each operation of s draws, within the clips
// around it.
func (r *Renderer) opBounds(s *scene.Scene, out []image.Rectangle) []image.Rectangle {
	clip := image.Rect(0, 0, s.Width, s.Height)
	r.clips = r.clips[:0]
	for i := range s.Ops {
		op := &s.Ops[i]
		var b image.Rectangle
		switch op.Kind {
		case scene.OpFill, scene.OpImage, scene.OpEffect, scene.OpHole:
			b = outset(op.Transform.Bounds(op.Rect), 1)
		case scene.OpShadow:
			if op.Inset {
				b = outset(op.Transform.Bounds(op.Cast), 1)
			} else {
				e := 1.5*op.Blur + 1
				box := scene.Rect{X: op.Rect.X - e, Y: op.Rect.Y - e, W: op.Rect.W + 2*e, H: op.Rect.H + 2*e}
				b = outset(op.Transform.Bounds(box), 1)
			}
		case scene.OpGlyphs:
			for _, g := range s.Glyphs[op.Start:op.End] {
				b = b.Union(outset(op.Transform.Mul(g.Transform).Bounds(scene.Rect{X: g.X, Y: g.Y, W: g.W, H: g.H}), 1))
			}
		case scene.OpPushClip:
			// A clip that changes changes everything it cuts.
			b = outset(op.Transform.Bounds(op.Rect), 1)
			r.clips = append(r.clips, clip)
			clip = clip.Intersect(b)
		case scene.OpPopClip:
			if len(r.clips) > 0 {
				clip = r.clips[len(r.clips)-1]
				r.clips = r.clips[:len(r.clips)-1]
			}
		}
		out = append(out, b.Intersect(clip))
	}
	return out
}

func outset(rc scene.Rect, d float32) image.Rectangle {
	return image.Rect(int(math.Floor(float64(rc.X-d))), int(math.Floor(float64(rc.Y-d))),
		int(math.Ceil(float64(rc.X+rc.W+d))), int(math.Ceil(float64(rc.Y+rc.H+d))))
}

// remember keeps what Render needs of s to compare the next scene with.
func (r *Renderer) remember(s *scene.Scene) {
	r.valid = true
	r.w, r.h = s.Width, s.Height
	r.clear = s.Clear
	if len(r.ops) > len(s.Ops) {
		clear(r.ops[len(s.Ops):])
	}
	r.ops = append(r.ops[:0], s.Ops...)
	r.glyphs = append(r.glyphs[:0], s.Glyphs...)
	r.stops = append(r.stops[:0], s.Stops...)
	if len(r.effects) > len(s.Effects) {
		clear(r.effects[len(s.Effects):])
	}
	r.effects = append(r.effects[:0], s.Effects...)
	// The uniforms' words, which the scene reuses next frame (Words), kept
	// apart.
	r.words = r.words[:0]
	for i := range r.effects {
		r.words = append(r.words, r.effects[i].Uniforms...)
		r.words = append(r.words, r.effects[i].WideUniforms...)
	}
	at := 0
	for i := range r.effects {
		fx := &r.effects[i]
		n, w := len(fx.Uniforms), len(fx.WideUniforms)
		fx.Uniforms = r.words[at : at+n : at+n]
		if fx.WideUniforms != nil {
			fx.WideUniforms = r.words[at+n : at+n+w : at+n+w]
		}
		at += n + w
	}
	r.bounds, r.next = r.next, r.bounds
	r.versions = r.versions[:0]
	for i := range s.Ops {
		v := uint64(0)
		if img := s.Ops[i].Image; img != nil {
			v = img.Version()
		}
		r.versions = append(r.versions, v)
	}
	r.mask.set(s.MaskAtlas)
	r.color.set(s.ColorAtlas)
}
