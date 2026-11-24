package ui

import (
	"bytes"
	"encoding/binary"
	"image"
	_ "image/gif"  // DecodeBitmap
	_ "image/jpeg" // DecodeBitmap
	_ "image/png"  // DecodeBitmap
	"math"

	_ "golang.org/x/image/bmp"  // DecodeBitmap
	_ "golang.org/x/image/webp" // DecodeBitmap

	"github.com/egoist/mygo/internal/scene"
)

// Bitmap is an image to show with Image. Create it once: converting an
// image is not free.
type Bitmap struct {
	img  *scene.Image
	w, h int
	// levels are the bitmap halved again and again, made as it shows
	// smaller, for drawing it smoothly: levels[k] is 2^(k+1) times
	// smaller.
	levels []*scene.Image
	// fresh counts the levels made from the current picture; Update
	// keeps the others to redo in place, so their textures stay too.
	fresh int
	// pixelated draws each pixel as a square of its color.
	pixelated bool
}

// Pixelated draws the bitmap's pixels as squares of their colors where it
// shows larger than its size, as pixel art draws, instead of blending
// neighbors smoothly. At whole multiples of its size, each pixel is a
// square of whole pixels of the screen; at others, squares of even sizes
// whose edges blend across one pixel of the screen, so that none is wider
// than the next and they glide when they move, as Ebitengine's
// FilterPixelated. Where it shows smaller, it is as smooth as any. It
// returns the bitmap.
func (b *Bitmap) Pixelated() *Bitmap {
	b.pixelated = true
	return b
}

// NewBitmap converts img.
func NewBitmap(img image.Image) *Bitmap {
	s := scene.NewImage(img)
	return &Bitmap{img: s, w: s.W, h: s.H}
}

// DecodeBitmap decodes a PNG, JPEG, GIF (its first frame), WebP or BMP
// image. A JPEG photo shows upright, as its camera's EXIF orientation
// says.
func DecodeBitmap(data []byte) (*Bitmap, error) {
	img, format, err := image.Decode(bytes.NewReader(data))
	if err != nil {
		return nil, err
	}
	b := NewBitmap(img)
	if format == "jpeg" {
		if o := exifOrientation(data); o > 1 {
			b.img = orient(b.img, o)
			b.w, b.h = b.img.W, b.img.H
		}
	}
	return b, nil
}

// Update shows img in place of the bitmap's picture, for one that changes
// as video does: the GPU keeps the bitmap's texture and uploads the new
// pixels, where a new Bitmap a frame would make a texture each, and keep
// each for a while. An *image.RGBA of rows 4×width bytes long from the
// origin, as image.NewRGBA makes, is taken as it is, without a copy:
// don't change its pixels until the next Update. Call it on the main
// thread, in the view or through Window.Update, which redraws the window
// to show it.
func (b *Bitmap) Update(img image.Image) {
	if b.img.Scene != nil {
		return // a canvas's: Canvas.Draw draws it
	}
	s := scene.NewImage(img)
	if s.W == b.img.W && s.H == b.img.H {
		b.img.Pix = s.Pix
		b.img.Changed()
	} else {
		b.img = s
		b.w, b.h = s.W, s.H
		b.levels = nil
	}
	b.fresh = 0
}

// Size returns the bitmap's size in pixels, which Image shows as DIPs.
func (b *Bitmap) Size() (w, h int) { return b.w, b.h }

func (b *Bitmap) imageSize() (float32, float32) {
	if b == nil {
		return 0, 0
	}
	return float32(b.w), float32(b.h)
}

// smaller returns the smallest of the bitmap and its levels that is no
// smaller than shrink times smaller, for drawing it at that size: a GPU
// sampling a bitmap far larger than it shows sees only some of its
// pixels, which shimmer, where a level averages them all.
func (b *Bitmap) smaller(shrink float32) *scene.Image {
	if b.img.Scene != nil {
		return b.img // a canvas, whose pixels the GPU keeps
	}
	if shrink < 2 {
		return b.img
	}
	k := int(math.Log2(float64(shrink))) // halvings that stay as large
	img := b.img
	for i := 0; i < k; i++ {
		switch {
		case i == len(b.levels):
			if img.W < 2 && img.H < 2 {
				return img
			}
			b.levels = append(b.levels, halve(img))
		case i >= b.fresh: // made from an earlier picture
			halveInto(b.levels[i], img)
			b.levels[i].Changed()
		}
		b.fresh = max(b.fresh, i+1)
		img = b.levels[i]
	}
	return img
}

// halve returns img half as large, each pixel the average of the 2×2 it
// covers, of premultiplied colors; an odd last row or column counts twice.
func halve(img *scene.Image) *scene.Image {
	w, h := max(1, (img.W+1)/2), max(1, (img.H+1)/2)
	out := scene.NewImageRGBA(w, h, make([]byte, 4*w*h))
	halveInto(out, img)
	return out
}

// halveInto is halve into dst, an image of halve's size.
func halveInto(dst, img *scene.Image) {
	w, h := dst.W, dst.H
	out := dst.Pix
	src, stride := img.Pix, 4*img.W
	for y := range h {
		y0, y1 := 2*y, min(2*y+1, img.H-1)
		r0, r1 := src[y0*stride:], src[y1*stride:]
		o := out[4*w*y:]
		for x := range w {
			x0, x1 := 4*2*x, 4*min(2*x+1, img.W-1)
			for c := range 4 {
				o[4*x+c] = uint8((uint32(r0[x0+c]) + uint32(r0[x1+c]) + uint32(r1[x0+c]) + uint32(r1[x1+c]) + 2) / 4)
			}
		}
	}
}

// exifOrientation returns the orientation of a JPEG's EXIF data, 1 to 8,
// or 0 without one.
func exifOrientation(jpg []byte) int {
	if len(jpg) < 4 || jpg[0] != 0xff || jpg[1] != 0xd8 {
		return 0
	}
	for i := 2; i+4 <= len(jpg); {
		if jpg[i] != 0xff {
			return 0
		}
		marker, size := jpg[i+1], int(binary.BigEndian.Uint16(jpg[i+2:]))
		if marker == 0xda || size < 2 || i+2+size > len(jpg) { // the image data starts
			return 0
		}
		seg := jpg[i+4 : i+2+size]
		if marker == 0xe1 && len(seg) > 14 && string(seg[:6]) == "Exif\x00\x00" {
			return tiffOrientation(seg[6:])
		}
		i += 2 + size
	}
	return 0
}

// tiffOrientation reads the Orientation tag of the first IFD of TIFF data.
func tiffOrientation(t []byte) int {
	var bo binary.ByteOrder
	switch string(t[:2]) {
	case "II":
		bo = binary.LittleEndian
	case "MM":
		bo = binary.BigEndian
	default:
		return 0
	}
	ifd := int(bo.Uint32(t[4:]))
	if ifd < 8 || ifd+2 > len(t) {
		return 0
	}
	n := int(bo.Uint16(t[ifd:]))
	for e := ifd + 2; e+12 <= len(t) && n > 0; e, n = e+12, n-1 {
		if bo.Uint16(t[e:]) == 0x0112 && bo.Uint16(t[e+2:]) == 3 { // Orientation, a SHORT
			if o := int(bo.Uint16(t[e+8:])); o >= 1 && o <= 8 {
				return o
			}
			return 0
		}
	}
	return 0
}

// orient turns and flips img as EXIF orientation o says it is to show.
func orient(img *scene.Image, o int) *scene.Image {
	w, h := img.W, img.H
	dw, dh := w, h
	if o >= 5 {
		dw, dh = h, w
	}
	out := make([]byte, 4*dw*dh)
	for y := range dh {
		for x := range dw {
			var sx, sy int
			switch o {
			case 2:
				sx, sy = w-1-x, y
			case 3:
				sx, sy = w-1-x, h-1-y
			case 4:
				sx, sy = x, h-1-y
			case 5:
				sx, sy = y, x
			case 6:
				sx, sy = y, h-1-x
			case 7:
				sx, sy = w-1-y, h-1-x
			case 8:
				sx, sy = w-1-y, x
			default:
				sx, sy = x, y
			}
			copy(out[4*(y*dw+x):4*(y*dw+x)+4], img.Pix[4*(sy*w+sx):])
		}
	}
	return scene.NewImageRGBA(dw, dh, out)
}
