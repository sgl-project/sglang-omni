package ui

import (
	"encoding/binary"
	"hash/maphash"
	"image"
	"image/color"
	"math"
	"sync/atomic"

	"github.com/egoist/mygo/internal/scene"
	"github.com/egoist/mygo/internal/svg"
)

// SVG is a vector image: an icon, a logo, an illustration. It stays sharp
// at any size. Parse it once, not in the view, and show it with Icon, in
// the color of the text, or with Image, in its own colors:
//
//	//go:embed icons/save.svg
//	var saveSVG []byte
//
//	var save = ui.MustParseSVG(saveSVG)
//
//	ui.Icon(c, save)
//
// MyGo draws the paths and basic shapes of SVG, with groups, use and
// symbol elements, transforms, colors, gradients, strokes and their
// dashes, opacity, clip paths, masks and style sheets of simple
// selectors. It leaves out text, embedded images, patterns, markers and
// filters: convert text to paths before.
type SVG struct {
	doc  *svg.Doc
	id   uint64
	w, h float32
}

var lastSVG atomic.Uint64

// ParseSVG parses an SVG document.
func ParseSVG(data []byte) (*SVG, error) {
	d, err := svg.Parse(data)
	if err != nil {
		return nil, err
	}
	w, h := d.Size()
	return &SVG{doc: d, id: lastSVG.Add(1), w: float32(w), h: float32(h)}, nil
}

// MustParseSVG is ParseSVG for SVGs that are part of the program, such as
// embedded files: it panics when one is in error.
func MustParseSVG(data []byte) *SVG {
	s, err := ParseSVG(data)
	if err != nil {
		panic("ui: " + err.Error())
	}
	return s
}

// Size returns the size the SVG says it has, its width and height or
// those of its viewBox, which Image shows as DIPs.
func (s *SVG) Size() (w, h float32) { return s.w, s.h }

func (s *SVG) imageSize() (float32, float32) {
	if s == nil {
		return 0, 0
	}
	return s.w, s.h
}

// Icon creates an element showing an SVG as an icon: its shapes in the
// color of the text, which it takes from its ancestors as text does
// (TextColor), whatever colors the SVG has. It is as high as the font size
// (FontSize), keeping the SVG's aspect ratio, unless given a size, and
// does not stretch across a column. Icons are decorations that assistive
// technology does not see, unless Label names them.
//
//	ui.Row(c).Gap(6).Children(func() {
//		ui.Icon(c, save)
//		ui.Text(c, "Save")
//	})
func coreIcon(c *context, s *SVG) *node {
	e := c.newElement(kindIcon)
	e.svg = s
	if s != nil && s.h > 0 {
		e.aspect = s.w / s.h
	}
	return e
}

// Icon draws the shapes of an SVG fitted in r, in color c.
func (p *Painter) Icon(s *SVG, r Rect, c Color) { p.drawIcon(s, r, c, 0) }

// Rotate rotates an element clockwise around its transform origin, which
// changes no layout sizes. An Icon keeps turning its SVG artwork around its
// center, as a spinner does:
//
//	spin := ui.Icon(c, loader)
//	spin.Rotate(spin.Loop("spin", time.Second, ui.Linear) * 360)
//
// Use Transform(Rotation(degrees)) to rotate an icon's entire border box.
func (e *node) Rotate(degrees float32) *node {
	finiteTransform(degrees)
	if e.kind == kindIcon {
		e.rotate = degrees
	} else {
		e.transform = e.transform.Then(Rotation(degrees))
	}
	return e
}

// Grayscale draws the element's image, or icon, in shades of gray.
func (e *node) Grayscale() *node { e.gray = true; return e }

// svgs holds what the engine reuses to draw SVGs: the job of drawing an
// icon's mask, the pixels it draws them into, and the pictures of SVGs
// drawn in their own colors.
type svgs struct {
	job      iconJob
	canvas   *image.RGBA
	mask     []byte
	pictures map[pictureKey]*picture
}

// iconJob is the mask of an icon drawIcon asks the text system for, which
// draws it when it is not cached: the SVG fitted in w×h pixels, turned by
// rotate degrees in a mask cw×ch.
type iconJob struct {
	s      *SVG
	w, h   int
	cw, ch int
	rotate float32
	draw   func() (w, h int, pix []byte)
}

// pictureKey identifies the picture of an SVG drawn w×h pixels, in its own
// colors with color for currentColor, stretched or not.
type pictureKey struct {
	s       *SVG
	w, h    int
	color   Color
	stretch bool
}

type picture struct {
	img  *scene.Image
	used uint64 // the last frame that drew it
}

var svgSeed = maphash.MakeSeed()

// canvasFor returns a cleared w×h canvas, reusing the engine's memory.
func (rt *engine) canvasFor(w, h int) *image.RGBA {
	c := rt.svgs.canvas
	if c == nil || cap(c.Pix) < w*h*4 {
		c = image.NewRGBA(image.Rect(0, 0, w, h))
		rt.svgs.canvas = c
		return c
	}
	c.Pix, c.Stride, c.Rect = c.Pix[:w*h*4], w*4, image.Rect(0, 0, w, h)
	clear(c.Pix)
	return c
}

// drawIcon paints an SVG's shapes fitted in r, in c, turned by rotate
// degrees around its center, from a mask the text system caches in its
// atlas: once there, the GPU draws the icon in any color without drawing
// its shapes again.
func (p *Painter) drawIcon(s *SVG, r Rect, c Color, rotate float32) {
	if s == nil || c.A == 0 || !p.visible(r, r.W+r.H) {
		return
	}
	d := p.snap(r)
	w, h := int(d.W), int(d.H)
	if w <= 0 || h <= 0 || w > 4096 || h > 4096 {
		return
	}
	// Turned by a sixteenth of a degree at the finest, in a mask as large
	// as it then covers, around the same center.
	rotate = float32(math.Mod(float64(rotate), 360))
	if rotate < 0 {
		rotate += 360
	}
	rotate = float32(math.Round(float64(rotate)*16) / 16)
	if rotate == 360 {
		rotate = 0
	}
	cw, ch := w, h
	if rotate != 0 {
		sin, cos := math.Sincos(float64(rotate) * math.Pi / 180)
		sin, cos = math.Abs(sin), math.Abs(cos)
		cw = int(math.Ceil(float64(w)*cos+float64(h)*sin)) + 2
		ch = int(math.Ceil(float64(w)*sin+float64(h)*cos)) + 2
		// Of the same parity, so that the center stays on the grid.
		cw += (cw - w) & 1
		ch += (ch - h) & 1
	}
	rt := p.rt
	j := &rt.svgs.job
	if j.draw == nil {
		j.draw = rt.rasterizeIcon
	}
	var buf [21]byte
	buf[0] = 's' // apart from the masks of paths
	binary.LittleEndian.PutUint64(buf[1:], s.id)
	binary.LittleEndian.PutUint32(buf[9:], uint32(w))
	binary.LittleEndian.PutUint32(buf[13:], uint32(h))
	binary.LittleEndian.PutUint32(buf[17:], math.Float32bits(rotate))
	j.s, j.w, j.h, j.cw, j.ch, j.rotate = s, w, h, cw, ch, rotate
	gi := rt.text.Mask(maphash.Bytes(svgSeed, buf[:]), j.draw)
	j.s = nil
	if !gi.OK {
		return
	}
	x, y := d.X-float32((cw-w)/2), d.Y-float32((ch-h)/2)
	start := int32(len(p.s.Glyphs))
	c = c.Alpha(p.opacity)
	p.s.Glyphs = append(p.s.Glyphs, scene.Glyph{X: x, Y: y, W: float32(gi.W), H: float32(gi.H), U: gi.X, V: gi.Y, UW: gi.W, VH: gi.H, Color: c.scene(), Wide: p.glyphWide(c)})
	p.add(scene.Op{Kind: scene.OpGlyphs, Start: start, End: start + 1})
}

// rasterizeIcon draws the mask of the icon job: the coverage of its SVG,
// whatever its colors.
func (rt *engine) rasterizeIcon() (int, int, []byte) {
	j := &rt.svgs.job
	img := rt.canvasFor(j.cw, j.ch)
	if j.rotate != 0 {
		j.s.doc.DrawRotated(img, color.NRGBA{A: 255}, float64(j.w), float64(j.h), float64(j.rotate))
	} else {
		j.s.doc.Draw(img, color.NRGBA{A: 255}, false)
	}
	n := j.cw * j.ch
	if cap(rt.svgs.mask) < n {
		rt.svgs.mask = make([]byte, n)
	}
	mask := rt.svgs.mask[:n]
	for i := range mask {
		mask[i] = img.Pix[i*4+3]
	}
	return j.cw, j.ch, mask
}

// drawSVG paints an SVG in its own colors, with current for currentColor,
// scaled into box as fit says, clipped to radius and in gray if gray.
func (p *Painter) drawSVG(s *SVG, box Rect, fit Fit, radius [4]float32, current Color, gray bool) {
	if s == nil || s.w <= 0 || s.h <= 0 || !p.visible(box, 0) {
		return
	}
	dst := box
	if fit != FillBox {
		scale := float32(1)
		switch fit {
		case Contain:
			scale = min(box.W/s.w, box.H/s.h)
		case Cover:
			scale = max(box.W/s.w, box.H/s.h)
		case ScaleDown:
			scale = min(box.W/s.w, box.H/s.h, 1)
		}
		dst.W, dst.H = s.w*scale, s.h*scale
		dst.X += (box.W - dst.W) / 2
		dst.Y += (box.H - dst.H) / 2
	}
	d := p.snap(dst)
	w, h := int(d.W), int(d.H)
	if w <= 0 || h <= 0 || w > 4096 || h > 4096 {
		return
	}
	if !s.doc.UsesCurrentColor() {
		current = Color{}
	}
	// Pictures are bitmaps in sRGB: current draws there as its nearest
	// sRGB color.
	img := p.rt.picture(pictureKey{s: s, w: w, h: h, color: current.SRGB(), stretch: fit == FillBox})
	op := scene.Op{Kind: scene.OpImage, Rect: d, Radii: p.radii(radius), Continuous: continuousCorners, Image: img, Src: scene.Rect{W: float32(w), H: float32(h)}, Opacity: p.opacity, Grayscale: gray}
	if fit == Cover || fit == NaturalSize {
		// Only what falls in the box shows.
		b := p.snap(box).Intersect(d)
		op.Rect, op.Src = b, scene.Rect{X: b.X - d.X, Y: b.Y - d.Y, W: b.W, H: b.H}
	}
	p.add(op)
}

// picture returns the picture of an SVG, drawing it unless a recent frame
// did.
func (rt *engine) picture(k pictureKey) *scene.Image {
	pics := rt.svgs.pictures
	if pics == nil {
		pics = map[pictureKey]*picture{}
		rt.svgs.pictures = pics
	}
	if pic, ok := pics[k]; ok {
		pic.used = rt.frame
		return pic.img
	}
	// A picture of the SVG at this size in another color, as its color
	// changes, is drawn again in place: the GPU updates its texture
	// instead of making another one.
	var img *scene.Image
	for old, pic := range pics {
		if old.s == k.s && old.w == k.w && old.h == k.h && old.stretch == k.stretch && pic.used != rt.frame {
			delete(pics, old)
			img = pic.img
			clear(img.Pix)
			break
		}
	}
	if img == nil {
		img = scene.NewImageRGBA(k.w, k.h, make([]byte, k.w*k.h*4))
	}
	c := k.color
	k.s.doc.Draw(&image.RGBA{Pix: img.Pix, Stride: k.w * 4, Rect: image.Rect(0, 0, k.w, k.h)}, color.NRGBA{c.R, c.G, c.B, c.A}, k.stretch)
	img.Changed()
	pics[k] = &picture{img: img, used: rt.frame}
	return img
}

// prunePictures forgets the pictures no frame drew for a while.
func (rt *engine) prunePictures() {
	for k, pic := range rt.svgs.pictures {
		if rt.frame-pic.used > 30 {
			delete(rt.svgs.pictures, k)
		}
	}
}
