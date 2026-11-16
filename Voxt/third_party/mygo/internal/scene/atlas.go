package scene

import (
	"cmp"
	"image"
	"slices"
)

// MaxAtlasSize is the largest width and height an Atlas grows to.
const MaxAtlasSize = 4096

// Atlas packs small bitmaps, such as rasterized glyphs, into one larger
// bitmap that renderers keep as a texture. It records which rectangles
// changed so that renderers upload only those. Main thread only.
//
// Lasting bitmaps fill shelves down from the top. Bitmaps a single frame
// draws, such as the masks of animated shapes, fill shelves up from the
// bottom, and ResetTransient frees them all at once, so that they never
// crowd out the lasting ones.
type Atlas struct {
	// BPP is the number of bytes per pixel: 1 for coverage masks, 4 for
	// premultiplied RGBA.
	BPP int
	// W and H are the size in pixels; Pix holds rows of W*BPP bytes.
	W, H int
	Pix  []byte

	gen     uint64
	version uint64
	log     []atlasChange
	// Lasting shelves span rows [0, bottom), transient ones [top, H).
	shelves, transient []shelf
	bottom, top        int
}

type atlasChange struct {
	version uint64
	r       image.Rectangle
}

type shelf struct{ y, h, x int }

// atlasLog is how many changes an atlas remembers; a renderer further
// behind uploads the whole atlas.
const atlasLog = 512

// NewAtlas returns an empty w×h atlas of bpp bytes per pixel.
func NewAtlas(bpp, w, h int) *Atlas {
	return &Atlas{BPP: bpp, W: w, H: h, Pix: make([]byte, w*h*bpp), gen: 1, top: h}
}

// Generation changes when the atlas is resized, reset or repacked, which
// makes renderers create a new texture and upload all of it.
func (a *Atlas) Generation() uint64 { return a.gen }

// Version increases with every change.
func (a *Atlas) Version() uint64 { return a.version }

// Changes returns the rectangles changed since version, or full when the
// renderer, which last uploaded version of generation gen, must upload
// everything.
func (a *Atlas) Changes(gen, version uint64) (rects []image.Rectangle, full bool) {
	if gen != a.gen {
		return nil, true
	}
	if version == a.version {
		return nil, false
	}
	if len(a.log) == 0 || a.log[0].version > version+1 {
		return nil, true
	}
	var union image.Rectangle
	for _, c := range a.log {
		if c.version > version {
			rects = append(rects, c.r)
			union = union.Union(c.r)
		}
	}
	// Many small uploads cost more than one larger one.
	if len(rects) > 32 {
		return []image.Rectangle{union}, false
	}
	return rects, false
}

// Alloc reserves a lasting w×h rectangle, keeping a pixel of padding to its
// right and below it so that filtering never reads a neighbor. It reports
// false when the atlas is full; the caller then repacks, grows or resets
// it.
func (a *Atlas) Alloc(w, h int) (x, y int, ok bool) {
	return a.alloc(&a.shelves, false, w, h)
}

// AllocTransient is Alloc for a rectangle needed until ResetTransient.
func (a *Atlas) AllocTransient(w, h int) (x, y int, ok bool) {
	return a.alloc(&a.transient, true, w, h)
}

func (a *Atlas) alloc(shelves *[]shelf, up bool, w, h int) (x, y int, ok bool) {
	pw, ph := w+1, h+1
	if pw > a.W || ph > a.H {
		return 0, 0, false
	}
	best := -1
	for i, s := range *shelves {
		if s.h >= ph && s.h <= ph+ph/4+2 && a.W-s.x >= pw && (best < 0 || s.h < (*shelves)[best].h) {
			best = i
		}
	}
	if best < 0 {
		if a.bottom+ph > a.top {
			return 0, 0, false
		}
		// A shelf a little taller than its first bitmap takes the next
		// ones, which are often a little taller (larger text, shapes).
		step := max(4, ph/8)
		sh := min((ph+step-1)/step*step, a.top-a.bottom)
		y := a.bottom
		if up {
			a.top -= sh
			y = a.top
		} else {
			a.bottom += sh
		}
		*shelves = append(*shelves, shelf{y: y, h: sh})
		best = len(*shelves) - 1
	}
	s := &(*shelves)[best]
	x, y = s.x, s.y
	s.x += pw
	return x, y, true
}

// ResetTransient frees the rectangles AllocTransient reserved.
func (a *Atlas) ResetTransient() {
	a.transient = a.transient[:0]
	a.top = a.H
}

// Put copies a w×h bitmap with rows of stride bytes to (x, y), and clears
// the padding after it, which may hold an earlier bitmap's pixels.
func (a *Atlas) Put(x, y, w, h int, src []byte, stride int) {
	bpp := a.BPP
	row := w * bpp
	for j := 0; j < h; j++ {
		i := ((y+j)*a.W + x) * bpp
		copy(a.Pix[i:][:row], src[j*stride:][:row])
		if x+w < a.W {
			clear(a.Pix[i+row:][:bpp])
		}
	}
	r := image.Rect(x, y, min(x+w+1, a.W), min(y+h+1, a.H))
	if y+h < a.H {
		i := ((y+h)*a.W + x) * bpp
		clear(a.Pix[i:][:r.Dx()*bpp])
	}
	a.version++
	a.log = append(a.log, atlasChange{a.version, r})
	if len(a.log) > atlasLog {
		a.log = append(a.log[:0], a.log[len(a.log)-atlasLog/2:]...)
	}
}

// Grow doubles the atlas up to MaxAtlasSize, keeping its lasting bitmaps
// where they are; it frees the transient ones. It reports false when the
// atlas is already at its largest.
func (a *Atlas) Grow() bool {
	if a.W >= MaxAtlasSize && a.H >= MaxAtlasSize {
		return false
	}
	w, h := min(a.W*2, MaxAtlasSize), min(a.H*2, MaxAtlasSize)
	pix := make([]byte, w*h*a.BPP)
	for j := 0; j < a.bottom; j++ {
		copy(pix[j*w*a.BPP:], a.Pix[j*a.W*a.BPP:(j+1)*a.W*a.BPP])
	}
	a.W, a.H, a.Pix = w, h, pix
	a.ResetTransient()
	a.changedAll()
	return true
}

// Repack frees the atlas and allocates rects, lasting rectangles of it,
// again, tallest first, moving their pixels along. It returns where each
// went, or false, changing nothing, when they do not fit.
func (a *Atlas) Repack(rects []image.Rectangle) ([]image.Point, bool) {
	order := make([]int, len(rects))
	for i := range order {
		order[i] = i
	}
	slices.SortFunc(order, func(i, j int) int {
		return cmp.Or(cmp.Compare(rects[j].Dy(), rects[i].Dy()), cmp.Compare(rects[j].Dx(), rects[i].Dx()))
	})
	plan := Atlas{W: a.W, H: a.H, top: a.H}
	pos := make([]image.Point, len(rects))
	for _, i := range order {
		x, y, ok := plan.Alloc(rects[i].Dx(), rects[i].Dy())
		if !ok {
			return nil, false
		}
		pos[i] = image.Pt(x, y)
	}
	saved := make([][]byte, len(rects))
	for i, r := range rects {
		row := r.Dx() * a.BPP
		b := make([]byte, row*r.Dy())
		for j := 0; j < r.Dy(); j++ {
			copy(b[j*row:][:row], a.Pix[((r.Min.Y+j)*a.W+r.Min.X)*a.BPP:][:row])
		}
		saved[i] = b
	}
	clear(a.Pix)
	for i, r := range rects {
		row := r.Dx() * a.BPP
		for j := 0; j < r.Dy(); j++ {
			copy(a.Pix[((pos[i].Y+j)*a.W+pos[i].X)*a.BPP:][:row], saved[i][j*row:][:row])
		}
	}
	a.shelves, a.bottom = plan.shelves, plan.bottom
	a.ResetTransient()
	a.changedAll()
	return pos, true
}

// Reset empties the atlas: every allocation becomes invalid.
func (a *Atlas) Reset() {
	clear(a.Pix)
	a.shelves = a.shelves[:0]
	a.bottom = 0
	a.ResetTransient()
	a.changedAll()
}

// changedAll makes renderers upload the whole atlas.
func (a *Atlas) changedAll() {
	a.gen++
	a.version++
	a.log = a.log[:0]
}
