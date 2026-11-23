// Package svg parses SVG documents and draws them into pixels, for the
// icons and pictures of package ui, in Go alone.
//
// It draws paths and the basic shapes, groups, nested svg elements, use
// and symbol elements, transforms, fills and strokes (with joins, caps
// and dashes) of colors, currentColor and linear and radial gradients,
// opacity, clip paths and masks, and style sheets of simple selectors
// (type, class and id). Text, embedded images, patterns, markers and
// filters are not drawn.
package svg

import (
	"image"
	"image/color"
	"sync"
)

// renderers keeps renderers and their memory between drawings.
var renderers = sync.Pool{New: func() any { return new(renderer) }}

// Doc is a parsed SVG document. Draw may draw it from any goroutine.
type Doc struct {
	root *node
	view box
	par  aspect
	w, h float64
	// current is true when something is painted with currentColor.
	current bool
}

// Parse parses an SVG document.
func Parse(data []byte) (*Doc, error) {
	root, err := parseXML(data)
	if err != nil {
		return nil, err
	}
	return newCompiler(root).doc(), nil
}

// Size returns the document's size in pixels: its width and height, or
// those of its viewBox.
func (d *Doc) Size() (w, h float64) { return d.w, d.h }

// UsesCurrentColor reports whether the color Draw takes for currentColor
// changes what it draws.
func (d *Doc) UsesCurrentColor() bool { return d.current }

// Draw draws the document over the pixels of dst, fitting its viewBox in
// them as its preserveAspectRatio says, or stretched to them with
// stretch. currentColor is current where the document does not set the
// color property.
func (d *Doc) Draw(dst *image.RGBA, current color.NRGBA, stretch bool) {
	b := dst.Bounds()
	a := d.par
	if stretch {
		a = aspect{none: true}
	}
	d.draw(dst, current, viewBoxTransform(d.view, a, float64(b.Dx()), float64(b.Dy())))
}

// DrawRotated draws the document as Draw does in a box w×h pixels at the
// center of dst, turned by degrees clockwise around it.
func (d *Doc) DrawRotated(dst *image.RGBA, current color.NRGBA, w, h, degrees float64) {
	b := dst.Bounds()
	m := translate(float64(b.Dx())/2, float64(b.Dy())/2).mul(rotation(degrees)).mul(translate(-w/2, -h/2))
	d.draw(dst, current, m.mul(viewBoxTransform(d.view, d.par, w, h)))
}

func (d *Doc) draw(dst *image.RGBA, current color.NRGBA, m matrix) {
	b := dst.Bounds()
	w, h := b.Dx(), b.Dy()
	if w <= 0 || h <= 0 || d.root == nil {
		return
	}
	r := renderers.Get().(*renderer)
	defer renderers.Put(r)
	if r.w != w || r.h != h {
		r.w, r.h, r.free = w, h, nil
	}
	r.current = rgba{float64(current.R) / 255, float64(current.G) / 255, float64(current.B) / 255, float64(current.A) / 255}
	l := r.layer()
	defer r.release(l)
	r.node(d.root, m, l, 1)
	dr := l.dirty
	for y := dr.Min.Y; y < dr.Max.Y; y++ {
		src := l.pix[(y*w+dr.Min.X)*4 : (y*w+dr.Max.X)*4]
		out := dst.Pix[dst.PixOffset(b.Min.X+dr.Min.X, b.Min.Y+y):]
		for i := 0; i < len(src); i += 4 {
			if a := src[i+3]; a == 255 {
				copy(out[i:i+4], src[i:i+4])
			} else if a > 0 {
				over(out[i:i+4], uint32(src[i]), uint32(src[i+1]), uint32(src[i+2]), uint32(a))
			}
		}
	}
}
