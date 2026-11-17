// Package imgconv turns a pasted or dropped picture (PNG or JPEG) into
// the JPEG a language model request carries: the longer side bounded, as
// the helper's pictures are (docs/helper-protocol.md).
package imgconv

import (
	"bytes"
	"image"
	_ "image/gif" // decode only
	"image/jpeg"
	_ "image/png" // decode only
	"os"

	"golang.org/x/image/draw"
)

// MaxSide is a picture's longest side, as the helper's.
const MaxSide = 1600

// quality is the JPEG's, as the helper's.
const quality = 70

// JPEG converts picture data (any format image.Decode reads) to a JPEG
// whose longer side is at most MaxSide. A JPEG already within the bound
// goes through unchanged.
func JPEG(data []byte) ([]byte, error) {
	if bytes.HasPrefix(data, []byte{0xff, 0xd8, 0xff}) && withinBound(data) {
		return data, nil
	}
	img, _, err := image.Decode(bytes.NewReader(data))
	if err != nil {
		return nil, err
	}
	return encode(img)
}

// JPEGFile is JPEG for the picture at path (a drop from Finder).
func JPEGFile(path string) ([]byte, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	return JPEG(data)
}

// withinBound reports whether the JPEG in data fits MaxSide already.
func withinBound(data []byte) bool {
	cfg, _, err := image.DecodeConfig(bytes.NewReader(data))
	return err == nil && max(cfg.Width, cfg.Height) <= MaxSide
}

// encode scales img to MaxSide and encodes it as a JPEG.
func encode(img image.Image) ([]byte, error) {
	b := img.Bounds()
	w, h := b.Dx(), b.Dy()
	if long := max(w, h); long > MaxSide {
		nw, nh := w*MaxSide/long, h*MaxSide/long
		dst := image.NewRGBA(image.Rect(0, 0, nw, nh))
		draw.CatmullRom.Scale(dst, dst.Bounds(), img, b, draw.Over, nil)
		img = dst
	}
	var out bytes.Buffer
	if err := jpeg.Encode(&out, img, &jpeg.Options{Quality: quality}); err != nil {
		return nil, err
	}
	return out.Bytes(), nil
}
