package main

import (
	"bytes"
	"image"
	"image/color"
	"image/png"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// writePNG writes a 4×4 gray picture whose pixel (1,1) has value v.
func writePNG(t *testing.T, dir, name string, v uint8, compress png.CompressionLevel) {
	t.Helper()
	img := image.NewRGBA(image.Rect(0, 0, 4, 4))
	for i := range img.Pix {
		img.Pix[i] = 200
	}
	img.Set(1, 1, color.RGBA{v, v, v, 255})
	var b bytes.Buffer
	if err := (&png.Encoder{CompressionLevel: compress}).Encode(&b, img); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(dir, name), b.Bytes(), 0o644); err != nil {
		t.Fatal(err)
	}
}

func TestCompare(t *testing.T) {
	before, after := t.TempDir(), t.TempDir()
	writePNG(t, before, "same.png", 10, png.DefaultCompression)
	writePNG(t, after, "same.png", 10, png.DefaultCompression)
	writePNG(t, before, "encoded.png", 10, png.DefaultCompression)
	writePNG(t, after, "encoded.png", 10, png.NoCompression)
	writePNG(t, before, "moved.png", 10, png.DefaultCompression)
	writePNG(t, after, "moved.png", 90, png.DefaultCompression)
	writePNG(t, before, "canvas-a.png", 10, png.DefaultCompression)
	writePNG(t, after, "canvas-a.png", 11, png.DefaultCompression)
	writePNG(t, before, "canvas-b.png", 10, png.DefaultCompression)
	writePNG(t, after, "canvas-b.png", 20, png.DefaultCompression)
	writePNG(t, before, "gone.png", 10, png.DefaultCompression)
	os.WriteFile(filepath.Join(before, "SKILL.md"), []byte("a"), 0o644)
	os.WriteFile(filepath.Join(after, "SKILL.md"), []byte("b"), 0o644)

	vol := filepath.Join(t.TempDir(), "volatile.txt")
	os.WriteFile(vol, []byte("# comment\n\ncanvas-*.png  2  glyph rounding\n"), 0o644)
	vs, err := readVolatile(vol)
	if err != nil {
		t.Fatal(err)
	}
	out := t.TempDir()
	r, err := compare([]string{before, after}, options{out: out, volatile: vs})
	if err != nil {
		t.Fatal(err)
	}
	if r.files != 7 || r.identical != 2 || r.tolerated != 1 || len(r.fails) != 4 {
		t.Fatalf("%s\n%s", r.summary(), strings.Join(r.fails, "\n"))
	}
	for i, want := range []string{"SKILL.md: differs", "canvas-b.png: 1 pixels differ (by up to 10), box (1,1)-(2,2)",
		"gone.png: missing in " + after, "moved.png: 1 pixels differ (by up to 80)"} {
		if !strings.HasPrefix(r.fails[i], want) {
			t.Errorf("fail %d = %q, want %q", i, r.fails[i], want)
		}
	}
	if _, err := os.Stat(filepath.Join(out, "moved.png")); err != nil {
		t.Errorf("no diff picture: %v", err)
	}
}

func TestCompareNoiseMask(t *testing.T) {
	a, b, c := t.TempDir(), t.TempDir(), t.TempDir()
	writePNG(t, a, "spin.png", 10, png.DefaultCompression)
	writePNG(t, b, "spin.png", 50, png.DefaultCompression)
	writePNG(t, c, "spin.png", 90, png.DefaultCompression)
	r, err := compare([]string{a, b, c}, options{grow: 0})
	if err != nil {
		t.Fatal(err)
	}
	if r.masked != 1 || len(r.fails) != 0 {
		t.Fatalf("%s %q", r.summary(), r.fails)
	}
}

func TestReadVolatileErrors(t *testing.T) {
	for _, line := range []string{"x.png 2", "x.png two why", "x.png 256 why", "[ 1 why"} {
		f := filepath.Join(t.TempDir(), "v.txt")
		os.WriteFile(f, []byte(line+"\n"), 0o644)
		if _, err := readVolatile(f); err == nil {
			t.Errorf("%q: no error", line)
		}
	}
}
