package main

import (
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"fmt"
	"image"
	"image/color"
	"image/png"
	"sync"
)

// pngToICNS renders the sizes macOS expects into an .icns file with PNG
// payloads. It leaves out icp4 and icp5, the 16 and 32 pixel entries:
// IconServices reads their payloads as the old ARGB format even when they
// hold a PNG, and Finder, Spotlight and Get Info draw the small icon as
// noise. macOS scales those sizes down from ic11 and ic12 instead.
func pngToICNS(src []byte) ([]byte, error) {
	img, err := png.Decode(bytes.NewReader(src))
	if err != nil {
		return nil, err
	}
	entries := []struct {
		kind string
		size int
	}{
		{"ic11", 32}, {"ic12", 64}, {"ic07", 128}, {"ic13", 256}, {"ic08", 256},
		{"ic14", 512}, {"ic09", 512}, {"ic10", 1024},
	}
	var body bytes.Buffer
	for _, e := range entries {
		var p bytes.Buffer
		if err := png.Encode(&p, resize(img, e.size)); err != nil {
			return nil, err
		}
		body.WriteString(e.kind)
		_ = binary.Write(&body, binary.BigEndian, uint32(p.Len()+8))
		body.Write(p.Bytes())
	}
	var out bytes.Buffer
	out.WriteString("icns")
	_ = binary.Write(&out, binary.BigEndian, uint32(body.Len()+8))
	out.Write(body.Bytes())
	return out.Bytes(), nil
}

// windowsIconSizes are the sizes of Windows icons, in pixels.
var windowsIconSizes = []int{16, 24, 32, 48, 64, 128, 256}

// renderedIcon holds the images iconImages rendered last, and the hash of
// their source: mygo dev renders the icon of every Windows build.
var renderedIcon struct {
	sync.Mutex
	sum    [32]byte
	images [][]byte
}

// iconImages renders src at each of the windowsIconSizes as PNG.
func iconImages(src []byte) ([][]byte, error) {
	sum := sha256.Sum256(src)
	renderedIcon.Lock()
	defer renderedIcon.Unlock()
	if renderedIcon.images != nil && renderedIcon.sum == sum {
		return renderedIcon.images, nil
	}
	img, err := png.Decode(bytes.NewReader(src))
	if err != nil {
		return nil, err
	}
	images := make([][]byte, len(windowsIconSizes))
	for i, s := range windowsIconSizes {
		var p bytes.Buffer
		if err := png.Encode(&p, resize(img, s)); err != nil {
			return nil, err
		}
		images[i] = p.Bytes()
	}
	renderedIcon.sum, renderedIcon.images = sum, images
	return images, nil
}

// pngToICO renders a Windows .ico with PNG payloads.
func pngToICO(src []byte) ([]byte, error) {
	images, err := iconImages(src)
	if err != nil {
		return nil, err
	}
	sizes := windowsIconSizes
	var out bytes.Buffer
	_ = binary.Write(&out, binary.LittleEndian, [3]uint16{0, 1, uint16(len(sizes))})
	offset := 6 + 16*len(sizes)
	for i, s := range sizes {
		dim := byte(s)
		if s >= 256 {
			dim = 0
		}
		out.Write([]byte{dim, dim, 0, 0})
		_ = binary.Write(&out, binary.LittleEndian, [2]uint16{1, 32})
		_ = binary.Write(&out, binary.LittleEndian, [2]uint32{uint32(len(images[i])), uint32(offset)})
		offset += len(images[i])
	}
	for _, p := range images {
		out.Write(p)
	}
	return out.Bytes(), nil
}

// resize scales img to a size x size square: area averaging when shrinking,
// bilinear sampling when enlarging.
func resize(img image.Image, size int) *image.NRGBA {
	b := img.Bounds()
	w, h := b.Dx(), b.Dy()
	dst := image.NewNRGBA(image.Rect(0, 0, size, size))
	if w == 0 || h == 0 {
		return dst
	}
	sx, sy := float64(w)/float64(size), float64(h)/float64(size)
	at := func(x, y int) (float64, float64, float64, float64) {
		c := color.NRGBAModel.Convert(img.At(b.Min.X+min(max(x, 0), w-1), b.Min.Y+min(max(y, 0), h-1))).(color.NRGBA)
		a := float64(c.A)
		// Premultiply so transparent pixels do not darken edges.
		return float64(c.R) * a, float64(c.G) * a, float64(c.B) * a, a
	}
	for y := range size {
		for x := range size {
			var r, g, bl, a, n float64
			if sx >= 1 {
				x0, x1 := int(float64(x)*sx), max(int(float64(x+1)*sx), int(float64(x)*sx)+1)
				y0, y1 := int(float64(y)*sy), max(int(float64(y+1)*sy), int(float64(y)*sy)+1)
				for yy := y0; yy < y1; yy++ {
					for xx := x0; xx < x1; xx++ {
						cr, cg, cb, ca := at(xx, yy)
						r, g, bl, a, n = r+cr, g+cg, bl+cb, a+ca, n+1
					}
				}
			} else {
				fx, fy := (float64(x)+0.5)*sx-0.5, (float64(y)+0.5)*sy-0.5
				x0, y0 := int(fx), int(fy)
				dx, dy := fx-float64(x0), fy-float64(y0)
				for _, s := range [4]struct {
					x, y int
					wt   float64
				}{{x0, y0, (1 - dx) * (1 - dy)}, {x0 + 1, y0, dx * (1 - dy)}, {x0, y0 + 1, (1 - dx) * dy}, {x0 + 1, y0 + 1, dx * dy}} {
					cr, cg, cb, ca := at(s.x, s.y)
					r, g, bl, a, n = r+cr*s.wt, g+cg*s.wt, bl+cb*s.wt, a+ca*s.wt, n+s.wt
				}
			}
			if a == 0 {
				continue
			}
			dst.SetNRGBA(x, y, color.NRGBA{R: uint8(r/a + 0.5), G: uint8(g/a + 0.5), B: uint8(bl/a + 0.5), A: uint8(a/n + 0.5)})
		}
	}
	return dst
}

func validateIcon(src []byte) error {
	cfg, err := png.DecodeConfig(bytes.NewReader(src))
	if err != nil {
		return fmt.Errorf("icon must be a PNG: %w", err)
	}
	if cfg.Width != cfg.Height {
		return fmt.Errorf("icon must be square, got %dx%d", cfg.Width, cfg.Height)
	}
	return nil
}
