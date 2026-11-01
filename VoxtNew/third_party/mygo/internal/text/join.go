package text

import (
	"image"
	"math"
	"strconv"
	"strings"
)

// JoinsGlyphs reports whether glyphs of a run whose edges share pixels
// look as the system's text stack draws them only joined (GlyphRun), with
// their coverage added before blending (cairo, GTK), rather than blended
// each in turn as renderers do (Core Graphics, Direct2D).
func (s *System) JoinsGlyphs() bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.engine().join()
}

// Shares reports whether glyphs a and b, their bitmaps' top-left corners
// at (ax, ay) and (bx, by) device pixels, both cover a pixel.
func (s *System) Shares(a GlyphImage, ax, ay float32, b GlyphImage, bx, by float32) bool {
	if !a.OK || !b.OK || a.Colored || b.Colored || a.Subpixel != b.Subpixel {
		return false
	}
	ra := image.Rect(int(ax), int(ay), int(ax)+int(a.W), int(ay)+int(a.H))
	rb := image.Rect(int(bx), int(by), int(bx)+int(b.W), int(by)+int(b.H))
	r := ra.Intersect(rb)
	if r.Empty() {
		return false
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	at := s.MaskAtlas
	if a.Subpixel {
		at = s.ColorAtlas
	}
	covers := func(g GlyphImage, x, y int) bool {
		i := (int(g.Y)+y)*at.W + int(g.X) + x
		if at.BPP == 1 {
			return at.Pix[i] != 0
		}
		p := at.Pix[4*i:]
		return p[0]|p[1]|p[2] != 0
	}
	for y := r.Min.Y; y < r.Max.Y; y++ {
		for x := r.Min.X; x < r.Max.X; x++ {
			if covers(a, x-ra.Min.X, y-ra.Min.Y) && covers(b, x-rb.Min.X, y-rb.Min.Y) {
				return true
			}
		}
	}
	return false
}

// GlyphRun returns glyphs ids of a font, their pens at xs device pixels
// from the left, joined as the system's text stack draws a run whose
// antialiased edges share pixels (JoinsGlyphs): their coverage added and
// clamped. The glyphs are for text of a shade, on an opaque background or
// not, as for Glyph. Left is relative to the pixel the first pen is in.
func (s *System) GlyphRun(f *Font, ids []uint32, xs []float32, scale float32, shade Shade, opaque bool) GlyphImage {
	if len(ids) == 0 || len(ids) != len(xs) {
		return GlyphImage{}
	}
	// The glyphs, as placed alone, which the key names.
	gs := make([]GlyphImage, len(ids))
	for i, id := range ids {
		gs[i] = s.Glyph(f, id, scale, xs[i], shade, opaque)
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	shade = f.shadeFor(shade)
	subpixel := opaque && s.subpixel
	origin := math.Floor(float64(xs[0]))
	var key strings.Builder
	key.WriteString(strconv.FormatUint(uint64(uint32(scale*256+0.5)), 10))
	key.WriteByte(byte('0' + shade))
	if subpixel {
		key.WriteByte('s')
	}
	for i, id := range ids {
		q, _, carry := s.place(f, scale, subpixel, xs[i])
		key.WriteByte(' ')
		key.WriteString(strconv.FormatUint(uint64(id), 10))
		key.WriteByte('@')
		key.WriteString(strconv.Itoa(int(math.Floor(float64(xs[i])) - origin + float64(carry))))
		key.WriteByte('+')
		key.WriteString(strconv.Itoa(q))
	}
	rk := runKey{f, key.String()}
	if e, ok := s.runs[rk]; ok {
		e.used = s.frame
		return e.GlyphImage
	}
	b := s.addGlyphs(gs, xs, origin)
	if b.w <= 0 || b.h <= 0 || b.w > 4096 || b.h > 2048 {
		return GlyphImage{}
	}
	inColor := b.subpixel
	x, y, ok := s.alloc(inColor, b.w, b.h)
	if !ok {
		return GlyphImage{}
	}
	if inColor {
		s.ColorAtlas.Put(x, y, b.w, b.h, b.pix, 4*b.w)
	} else {
		s.MaskAtlas.Put(x, y, b.w, b.h, b.pix, b.w)
	}
	g := GlyphImage{OK: true, Subpixel: b.subpixel, Thin: f.thin,
		X: uint16(x), Y: uint16(y), W: uint16(b.w), H: uint16(b.h), Left: float32(b.left), Top: float32(b.top)}
	s.runs[rk] = &atlasEntry{g, s.frame}
	return g
}

type runKey struct {
	font *Font
	key  string
}

// addGlyphs adds the coverage of glyphs placed alone, their pens at xs,
// clamped, as cairo composites the glyphs of a run, into a bitmap placed
// from the pixel origin.
func (s *System) addGlyphs(gs []GlyphImage, xs []float32, origin float64) bitmap {
	var box image.Rectangle
	rects := make([]image.Rectangle, len(gs))
	for i, g := range gs {
		if !g.OK {
			continue
		}
		x := int(math.Floor(float64(xs[i]))-origin) + int(g.Left)
		rects[i] = image.Rect(x, int(g.Top), x+int(g.W), int(g.Top)+int(g.H))
		box = box.Union(rects[i])
	}
	if box.Empty() {
		return bitmap{}
	}
	subpixel := gs[0].Subpixel
	at, bpp := s.MaskAtlas, 1
	if subpixel {
		at, bpp = s.ColorAtlas, 4
	}
	w, h := box.Dx(), box.Dy()
	sum := make([]uint16, w*h*bpp)
	for i, g := range gs {
		if !g.OK {
			continue
		}
		r := rects[i]
		for y := range int(g.H) {
			src := at.Pix[((int(g.Y)+y)*at.W+int(g.X))*bpp:]
			dst := sum[((r.Min.Y-box.Min.Y+y)*w+r.Min.X-box.Min.X)*bpp:]
			for k := range int(g.W) * bpp {
				dst[k] += uint16(src[k])
			}
		}
	}
	pix := make([]byte, len(sum))
	for i, v := range sum {
		pix[i] = uint8(min(v, 255))
	}
	return bitmap{left: box.Min.X, top: box.Min.Y, w: w, h: h, pix: pix, subpixel: subpixel}
}
