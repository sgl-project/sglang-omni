//go:build windows

package main

import (
	"bytes"
	"encoding/binary"
	"image"
	"image/jpeg"
	"image/png"
	"os"
	"path/filepath"
	"testing"
)

func testDIB(img image.Image) []byte {
	bounds := img.Bounds()
	w, h := bounds.Dx(), bounds.Dy()
	header := make([]byte, 40)
	binary.LittleEndian.PutUint32(header[0:4], 40)
	binary.LittleEndian.PutUint32(header[4:8], uint32(w))
	binary.LittleEndian.PutUint32(header[8:12], uint32(h))
	binary.LittleEndian.PutUint16(header[12:14], 1)
	binary.LittleEndian.PutUint16(header[14:16], 32)
	pixels := make([]byte, w*h*4)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			r, g, b, a := img.At(x, y).RGBA()
			i := (y*w + x) * 4
			pixels[i], pixels[i+1], pixels[i+2], pixels[i+3] = byte(b>>8), byte(g>>8), byte(r>>8), byte(a>>8)
		}
	}
	return append(header, pixels...)
}

func TestDecodeDIB(t *testing.T) {
	src := image.NewRGBA(image.Rect(0, 0, 4, 2))
	img, err := decodeDIB(testDIB(src))
	if err != nil {
		t.Fatal(err)
	}
	if img.Bounds().Dx() != 4 || img.Bounds().Dy() != 2 {
		t.Fatalf("bounds %v", img.Bounds())
	}
	if _, err := decodeDIB([]byte{1, 2, 3}); err == nil {
		t.Fatal("short DIB decoded")
	}
}

func TestEncodeJPEGDownsizes(t *testing.T) {
	src := image.NewRGBA(image.Rect(0, 0, 3200, 1600))
	data, err := encodeJPEG(src, 1600)
	if err != nil {
		t.Fatal(err)
	}
	img, err := jpeg.Decode(bytes.NewReader(data))
	if err != nil {
		t.Fatal(err)
	}
	if img.Bounds().Dx() != 1600 || img.Bounds().Dy() != 800 {
		t.Fatalf("bounds %v", img.Bounds())
	}
	data, err = encodeJPEG(src, 0)
	if err != nil {
		t.Fatal(err)
	}
	img, _ = jpeg.Decode(bytes.NewReader(data))
	if img.Bounds().Dx() != 3200 {
		t.Fatalf("no-resize bounds %v", img.Bounds())
	}
}

func TestDecodeImageFile(t *testing.T) {
	dir := t.TempDir()
	pngPath := filepath.Join(dir, "pic.png")
	f, _ := os.Create(pngPath)
	png.Encode(f, image.NewRGBA(image.Rect(0, 0, 2, 2)))
	f.Close()
	if img := decodeImageFile(pngPath); img == nil {
		t.Error("a PNG file was not read")
	}
	txtPath := filepath.Join(dir, "notes.txt")
	os.WriteFile(txtPath, []byte("hi"), 0o644)
	if img := decodeImageFile(txtPath); img != nil {
		t.Error("a text file decoded as an image")
	}
}
