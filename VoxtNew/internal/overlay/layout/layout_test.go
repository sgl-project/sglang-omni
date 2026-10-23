package layout

import (
	"math"
	"slices"
	"testing"

	"github.com/egoist/mygo"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/session"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/settings"
)

// TestWaveBars: the bars follow a word up quickly and down slowly, and
// a quiet room stays flat.
func TestWaveBars(t *testing.T) {
	quiet := make([]float64, session.Bars)
	for i := range quiet {
		quiet[i] = session.Level(-55)
	}
	for i, v := range WaveBars(quiet) {
		if v != 0 {
			t.Fatalf("quiet bar %d = %v", i, v)
		}
	}
	word := slices.Clone(quiet)
	word[8] = session.Level(-15) // one loud level, then quiet again
	b := WaveBars(word)
	if b[7] != 0 || b[8] < 0.7 || !(b[9] < b[8] && b[9] > 0.4*b[8]) || !(b[15] > 0 && b[15] < b[9]) {
		t.Errorf("word %v", b)
	}
	if got := WaveBars(nil); got[0] != 0 || got[15] != 0 {
		t.Errorf("none %v", got)
	}
	long := append(slices.Clone(quiet), word...)
	if WaveBars(long) != WaveBars(word) {
		t.Error("more levels than bars: the newest count")
	}
	for _, p := range []float64{0, 0.3, 0.6, 0.9, 1.15} {
		sc, al := LoaderDot(p)
		if sc < 0.6 || sc > 1.3 || al < 0.15 || al > 1 {
			t.Errorf("loader at %v: %v %v", p, sc, al)
		}
	}
}

func TestOverlayPlace(t *testing.T) {
	wa := mygo.Rectangle{X: 0, Y: 25, Width: 1440, Height: 875}
	x, y := Place(wa, settings.Overlay{Position: "bottom", Margin: 96}, 272, 84)
	if x != (1440-272)/2 || y != 25+875-84-(96-Pad) {
		t.Errorf("bottom %d,%d", x, y)
	}
	_, y = Place(wa, settings.Overlay{Position: "top", Margin: 40}, 272, 84)
	if y != 25+40-Pad {
		t.Errorf("top %d", y)
	}
}

func TestMicLoader(t *testing.T) {
	if d, a := MicRing(0, true); d != MicRingMin || a != 0.6 {
		t.Errorf("ring at 0: %v %v", d, a)
	}
	if d, a := MicRing(1.199, true); d < MicRingMax-0.1 || a > 0.01 {
		t.Errorf("ring at the end: %v %v", d, a)
	}
	if d, a := MicRing(5, false); d != 22 || a != 0.35 {
		t.Errorf("still ring %v %v", d, a)
	}
	if a, s := MicPulse(0, true); a != 0.55 || s != 0.94 {
		t.Errorf("pulse at 0: %v %v", a, s)
	}
	if a, s := MicPulse(1.25, true); math.Abs(a-0.98) > 1e-9 || math.Abs(s-1.04) > 1e-9 {
		t.Errorf("pulse at its top: %v %v", a, s)
	}
}
