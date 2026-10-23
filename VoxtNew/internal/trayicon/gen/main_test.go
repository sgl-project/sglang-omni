package main

import (
	"bytes"
	"image"
	"image/png"
	"math"
	"testing"

	"github.com/sgl-project/sglang-omni/VoxtNew/internal/trayicon"
)

// The embedded icons are what gen makes now: run go generate
// ./internal/trayicon after changing it.
func TestIconsUpToDate(t *testing.T) {
	src, err := load("../voxt.png")
	if err != nil {
		t.Fatal(err)
	}
	for _, rec := range []bool{false, true} {
		want := render(src, rec)
		got, err := png.Decode(bytes.NewReader(trayicon.Icon(rec)))
		if err != nil {
			t.Fatal(err)
		}
		if got.Bounds() != want.Bounds() {
			t.Fatalf("recording %v: bounds %v, want %v", rec, got.Bounds(), want.Bounds())
		}
		for y := 0; y < size; y++ {
			for x := 0; x < size; x++ {
				r, g, b, a := got.At(x, y).RGBA()
				if a>>8 != uint32(want.NRGBAAt(x, y).A) || (a > 0 && (r|g|b) != 0) {
					t.Fatalf("recording %v: pixel %d,%d differs; run go generate ./internal/trayicon", rec, x, y)
				}
			}
		}
	}
}

// The dot sits in the corner the microphone leaves empty: at rest that
// corner is clear, recording it is solid.
func TestRecordingDot(t *testing.T) {
	src, err := load("../voxt.png")
	if err != nil {
		t.Fatal(err)
	}
	idle, rec := render(src, false), render(src, true)
	at := image.Pt(int(math.Floor(dotX)), int(math.Floor(dotY)))
	if a := idle.NRGBAAt(at.X, at.Y).A; a != 0 {
		t.Errorf("at rest the corner has alpha %d", a)
	}
	if a := rec.NRGBAAt(at.X, at.Y).A; a < 250 {
		t.Errorf("recording the dot has alpha %d", a)
	}
}
