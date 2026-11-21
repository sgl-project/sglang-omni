package text

import (
	"image"
	"math"

	"github.com/egoist/mygo/internal/scene"
)

// A Shade is how light the color of text is, from 0 (dark) to Shades-1
// (light). Core Text's font smoothing, which AppKit draws text with,
// emboldens glyphs more the lighter their color is, in four steps of its
// relative luminance: the glyphs of its fonts are rasterized for each.
type Shade uint8

// Shades is the number of shades of text.
const Shades = 4

// Thick is a shade beyond the others, for text drawn thicker: with Core
// Text's font smoothing at its strongest, whatever the text's color and
// the user's setting, as Ghostty's font-thicken draws it. Engines that do
// not thicken glyphs draw them as for other text.
const Thick Shade = Shades

// Flat is a shade beyond the others, for text drawn without font
// smoothing, whatever its color: thinner, as browsers draw text styled
// -webkit-font-smoothing: antialiased on macOS. Engines that do not
// smooth fonts draw it as other text.
const Flat Shade = Shades + 1

// shadeFor returns the shade that glyphs of f for text of shade are drawn
// for.
func (f *Font) shadeFor(shade Shade) Shade {
	switch {
	case shade == Thick && f.thickens:
		return Thick
	case !f.shaded:
		return 0
	case shade == Flat:
		return Flat
	}
	return min(shade, Shades-1)
}

// ShadeOf returns the shade of text of an sRGB color: its relative
// luminance rounded to a quarter, as Core Text rounds it, and at most 3/4.
func ShadeOf(r, g, b uint8) Shade {
	y := 0.2126*linear[r] + 0.7152*linear[g] + 0.0722*linear[b]
	return Shade(min(y*Shades+0.5, Shades-1))
}

// linear maps sRGB values to linear light.
var linear = func() (t [256]float32) {
	for i := range t {
		v := float64(i) / 255
		if v <= 0.04045 {
			v /= 12.92
		} else {
			v = math.Pow((v+0.055)/1.055, 2.4)
		}
		t[i] = float32(v)
	}
	return t
}()

type glyphKey struct {
	font     *Font
	id       uint32
	scale    uint32 // pixels per DIP, in 1/256
	subX     uint8  // the position within a pixel, of placement.n
	shade    Shade
	subpixel bool
}

// placement is how the system's text stack places glyphs of a font at a
// scale horizontally: at n positions within a pixel, the nearest to their
// pen if round, else the one left of it.
type placement struct {
	n     int
	round bool
}

type placeKey struct {
	font     *Font
	scale    uint32
	subpixel bool
}

// GlyphImage is a rasterized glyph in an atlas.
type GlyphImage struct {
	// OK is false for glyphs without pixels, such as spaces.
	OK bool
	// Colored glyphs are in the color atlas, the others in the mask atlas
	// but Subpixel ones, in the color atlas too, which hold the coverage of
	// each pixel's red, green and blue subpixels.
	Colored, Subpixel bool
	// Thin glyphs are of a font too thin for antialiasing, which renderers
	// give more contrast (scene.Glyph.Thin).
	Thin bool
	// X, Y, W and H are the bitmap's rectangle in its atlas.
	X, Y, W, H uint16
	// Left and Top place the bitmap relative to the glyph's origin on the
	// baseline, in pixels: Top is usually negative.
	Left, Top float32
}

// inColor reports whether the glyph is in the color atlas.
func (g GlyphImage) inColor() bool { return g.Colored || g.Subpixel }

// atlasEntry is a cached glyph or mask.
type atlasEntry struct {
	GlyphImage
	used uint64 // the last frame that drew it
}

// BeginFrame starts painting a frame: the masks drawn for the previous
// frame alone are forgotten.
func (s *System) BeginFrame() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.MaskAtlas.ResetTransient()
	clear(s.transient)
	s.full, s.want = [2]bool{}, [2]int{}
	s.wantSize = [2]image.Point{}
	s.rooms = 0
}

// Full reports whether an atlas filled up during the frame being painted
// and left out glyphs or masks it draws. MakeRoom then makes room for them.
func (s *System) Full() bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.full[0] || s.full[1]
}

// MakeRoom makes room in the atlases that filled up: it keeps what the
// frame being painted drew, growing the atlas when that and what the frame
// still needs take much of it, or when room made for the frame before did
// not suffice, and forgets everything else. Everything kept moves, so the
// frame must be painted again.
func (s *System) MakeRoom() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.rooms++
	for i, color := range [2]bool{false, true} {
		if s.full[i] {
			s.makeRoom(color, s.want[i], s.wantSize[i], s.rooms > 1)
		}
	}
	s.full, s.want = [2]bool{}, [2]int{}
	s.wantSize = [2]image.Point{}
}

func (s *System) makeRoom(color bool, want int, size image.Point, grow bool) {
	a := s.MaskAtlas
	if color {
		a = s.ColorAtlas
	}
	var keep []*atlasEntry
	need := want
	if !color {
		for _, g := range s.transient {
			need += (int(g.W) + 1) * (int(g.H) + 1)
		}
		clear(s.transient)
		a.ResetTransient()
		for k, e := range s.masks {
			if e.used != s.frame {
				delete(s.masks, k)
				continue
			}
			keep = append(keep, e)
		}
	}
	for k, e := range s.glyphs {
		if !e.OK || e.inColor() != color {
			continue // blank glyphs take no room
		}
		if e.used != s.frame {
			delete(s.glyphs, k)
			continue
		}
		keep = append(keep, e)
	}
	for k, e := range s.runs {
		if e.inColor() != color {
			continue
		}
		if e.used != s.frame {
			delete(s.runs, k)
			continue
		}
		keep = append(keep, e)
	}
	rects := make([]image.Rectangle, len(keep))
	for i, e := range keep {
		rects[i] = image.Rect(int(e.X), int(e.Y), int(e.X)+int(e.W), int(e.Y)+int(e.H))
		need += (int(e.W) + 1) * (int(e.H) + 1)
	}
	// Shelves waste room: keep the atlas at most half full.
	if grow {
		a.Grow()
	}
	for (need*2 > a.W*a.H || size.X > a.W || size.Y > a.H) && a.Grow() {
	}
	pos, ok := a.Repack(rects)
	if !ok {
		a.Reset()
		for k, e := range s.glyphs {
			if e.OK && e.inColor() == color {
				delete(s.glyphs, k)
			}
		}
		for k, e := range s.runs {
			if e.inColor() == color {
				delete(s.runs, k)
			}
		}
		if !color {
			clear(s.masks)
		}
		return
	}
	for i, e := range keep {
		e.X, e.Y = uint16(pos[i].X), uint16(pos[i].Y)
	}
}

// Glyph rasterizes glyph id of a font at scale pixels per DIP whose pen
// is x device pixels from the left, for text of a shade, placed as the
// system's text stack places glyphs: at one of the positions within a
// pixel it draws them at. Left is relative to the pixel x is in. Glyphs on
// opaque backgrounds may be subpixel ones, when the system's settings
// (TextParams) ask for subpixel antialiasing.
func (s *System) Glyph(f *Font, id uint32, scale, x float32, shade Shade, opaque bool) GlyphImage {
	s.mu.Lock()
	defer s.mu.Unlock()
	if f == nil {
		return GlyphImage{}
	}
	shade = f.shadeFor(shade)
	subpixel := opaque && s.subpixel
	q, n, carry := s.place(f, scale, subpixel, x)
	key := glyphKey{font: f, id: id, scale: uint32(scale*256 + 0.5), subX: uint8(q), shade: shade, subpixel: subpixel}
	e, ok := s.glyphs[key]
	if !ok {
		failed := s.failed
		g := s.rasterize(f, id, scale, float32(q)/float32(n), key.shade, key.subpixel)
		e = &atlasEntry{g, s.frame}
		if s.failed == failed { // not left out of a full atlas
			s.glyphs[key] = e
		}
	}
	e.used = s.frame
	g := e.GlyphImage
	g.Left += carry
	return g
}

// place returns where the system's text stack draws a glyph of f whose pen
// is at x: at position q of the n within a pixel, carry pixels right of
// the pixel x is in.
func (s *System) place(f *Font, scale float32, subpixel bool, x float32) (q, n int, carry float32) {
	pk := placeKey{f, uint32(scale*256 + 0.5), subpixel}
	pl, ok := s.places[pk]
	if !ok {
		pl.n, pl.round = s.engine().positions(f, scale, subpixel)
		pl.n = max(pl.n, 1)
		s.places[pk] = pl
	}
	frac := float64(x) - math.Floor(float64(x))
	if pl.round {
		if q = int(math.Floor(frac*float64(pl.n) + 0.5)); q == pl.n {
			return 0, pl.n, 1
		}
		return q, pl.n, 0
	}
	return min(int(frac*float64(pl.n)), pl.n-1), pl.n, 0
}

// Baseline returns where the system's text stack draws a baseline that is
// y device pixels from the top: on a whole pixel.
func (s *System) Baseline(y float32) float32 {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.engine().baseline(y)
}

func (s *System) rasterize(f *Font, id uint32, scale, dx float32, shade Shade, subpixel bool) GlyphImage {
	b := s.engine().glyph(f, id, scale, dx, shade, subpixel)
	if b.w <= 0 || b.h <= 0 || b.w > 2048 || b.h > 2048 {
		return GlyphImage{}
	}
	inColor := b.color || b.subpixel
	x, y, ok := s.alloc(inColor, b.w, b.h)
	if !ok {
		return GlyphImage{}
	}
	if inColor {
		s.ColorAtlas.Put(x, y, b.w, b.h, b.pix, 4*b.w)
	} else {
		s.MaskAtlas.Put(x, y, b.w, b.h, b.pix, b.w)
	}
	return GlyphImage{OK: true, Colored: b.color, Subpixel: b.subpixel, Thin: f.thin && !b.color,
		X: uint16(x), Y: uint16(y), W: uint16(b.w), H: uint16(b.h), Left: float32(b.left), Top: float32(b.top)}
}

// TextParams returns how renderers correct the coverage of glyphs, as
// the system's settings say, for the frame being painted, and has glyphs
// on opaque backgrounds take subpixel antialiasing if they ask for it.
func (s *System) TextParams() scene.TextParams {
	s.mu.Lock()
	defer s.mu.Unlock()
	p, subpixel := s.engine().textParams()
	s.subpixel = subpixel
	return p
}

// alloc finds lasting room in an atlas, or records that it is full.
func (s *System) alloc(color bool, w, h int) (int, int, bool) {
	a := s.MaskAtlas
	if color {
		a = s.ColorAtlas
	}
	// The first bitmap can size an empty atlas without invalidating any
	// glyphs already painted. In particular, the color atlas starts with
	// one transparent texel and needs no repaint to show its first emoji.
	if a.Version() == 0 {
		for (w+1 > a.W || h+1 > a.H) && a.Grow() {
		}
	}
	if x, y, ok := a.Alloc(w, h); ok {
		return x, y, true
	}
	s.leftOut(color, w, h)
	return 0, 0, false
}

func (s *System) leftOut(color bool, w, h int) {
	i := 0
	if color {
		i = 1
	}
	s.full[i] = true
	s.want[i] += (w + 1) * (h + 1)
	s.wantSize[i].X = max(s.wantSize[i].X, w+1)
	s.wantSize[i].Y = max(s.wantSize[i].Y, h+1)
	s.failed++
}

// Mask returns a coverage mask cached under key in the mask atlas,
// drawing it when it is not: draw returns its size and one byte per pixel.
// Icons and other vector shapes use it.
//
// A mask a frame draws for the first time may be one frame of an
// animation, never drawn again: it lasts for that frame only, in the
// atlas's transient room, and is cached once a later frame draws it too.
func (s *System) Mask(key uint64, draw func() (w, h int, pix []byte)) GlyphImage {
	s.mu.Lock()
	defer s.mu.Unlock()
	if e, ok := s.masks[key]; ok {
		e.used = s.frame
		return e.GlyphImage
	}
	if g, ok := s.transient[key]; ok {
		return g
	}
	last, seen := s.recent[key]
	lasting := seen && last != s.frame
	w, h, pix := draw()
	if w <= 0 || h <= 0 {
		return GlyphImage{}
	}
	var x, y int
	var ok bool
	if lasting {
		x, y, ok = s.MaskAtlas.Alloc(w, h)
	} else {
		x, y, ok = s.MaskAtlas.AllocTransient(w, h)
	}
	if !ok {
		s.leftOut(false, w, h)
		return GlyphImage{}
	}
	s.MaskAtlas.Put(x, y, w, h, pix, w)
	g := GlyphImage{OK: true, X: uint16(x), Y: uint16(y), W: uint16(w), H: uint16(h)}
	if lasting {
		delete(s.recent, key)
		s.masks[key] = &atlasEntry{g, s.frame}
	} else {
		s.transient[key] = g
		s.recent[key] = s.frame
	}
	return g
}
