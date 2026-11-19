package imgconv

import (
	"bytes"
	"encoding/binary"
	"hash/crc32"
	"image"
	"image/jpeg"
	"image/png"
	"os"
	"path/filepath"
	"testing"
)

// pngData is a PNG of w × h.
func pngData(t *testing.T, w, h int) []byte {
	t.Helper()
	img := image.NewRGBA(image.Rect(0, 0, w, h))
	var out bytes.Buffer
	if err := png.Encode(&out, img); err != nil {
		t.Fatal(err)
	}
	return out.Bytes()
}

func size(t *testing.T, data []byte) (int, int) {
	t.Helper()
	cfg, format, err := image.DecodeConfig(bytes.NewReader(data))
	if err != nil {
		t.Fatal(err)
	}
	if format != "jpeg" {
		t.Fatalf("format %s", format)
	}
	return cfg.Width, cfg.Height
}

// TestJPEG: a PNG becomes a JPEG, bounded on the longer side; a small
// JPEG goes through unchanged; junk is an error.
func TestJPEG(t *testing.T) {
	small := pngData(t, 100, 50)
	j, err := JPEG(small)
	if err != nil {
		t.Fatal(err)
	}
	if w, h := size(t, j); w != 100 || h != 50 {
		t.Errorf("%dx%d", w, h)
	}

	big := pngData(t, 3200, 1600)
	j, err = JPEG(big)
	if err != nil {
		t.Fatal(err)
	}
	if w, h := size(t, j); w != MaxSide || h != MaxSide/2 {
		t.Errorf("%dx%d", w, h)
	}

	var orig bytes.Buffer
	if err := jpeg.Encode(&orig, image.NewRGBA(image.Rect(0, 0, 10, 10)), nil); err != nil {
		t.Fatal(err)
	}
	j, err = JPEG(orig.Bytes())
	if err != nil {
		t.Fatal(err)
	}
	if !bytes.Equal(j, orig.Bytes()) {
		t.Errorf("a small JPEG changed (%d → %d bytes)", orig.Len(), len(j))
	}

	if _, err = JPEG([]byte("not a picture")); err == nil {
		t.Error("junk decoded")
	}
}

func TestJPEGRejectsOversize(t *testing.T) {
	padded := append(pngData(t, 1, 1), make([]byte, 32<<20)...)
	if _, err := JPEG(padded); err == nil || err.Error() != "image is too large" {
		t.Fatalf("oversize data: err=%v", err)
	}
	data := pngData(t, 1, 1)
	binary.BigEndian.PutUint32(data[16:20], 40_000_001)
	binary.BigEndian.PutUint32(data[20:24], 1)
	binary.BigEndian.PutUint32(data[29:33], crc32.ChecksumIEEE(data[12:29]))
	if _, err := JPEG(data); err == nil || err.Error() != "image is too large" {
		t.Fatalf("over 40 megapixels: err=%v", err)
	}
	path := filepath.Join(t.TempDir(), "large.png")
	if err := os.WriteFile(path, make([]byte, (32<<20)+1), 0o644); err != nil {
		t.Fatal(err)
	}
	if _, err := JPEGFile(path); err == nil || err.Error() != "image is too large" {
		t.Fatalf("oversize file: err=%v", err)
	}
	if _, err := JPEGFile("/dev/zero"); err == nil || err.Error() != "image is too large" {
		t.Fatalf("unbounded file: err=%v", err)
	}
}

// TestJPEGFile: a dropped picture file converts; a missing file is an
// error.
func TestJPEGFile(t *testing.T) {
	path := filepath.Join(t.TempDir(), "pic.png")
	if err := os.WriteFile(path, pngData(t, 40, 40), 0o644); err != nil {
		t.Fatal(err)
	}
	j, err := JPEGFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if w, h := size(t, j); w != 40 || h != 40 {
		t.Errorf("%dx%d", w, h)
	}
	if _, err := JPEGFile(path + ".gone"); err == nil {
		t.Error("a missing file converted")
	}
}
