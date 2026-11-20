package layout

import (
	"math"
	"slices"
	"testing"

	"github.com/egoist/mygo"
	"github.com/sgl-project/sglang-omni/Voxt/internal/session"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
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

// TestBubbleTextWidth: a Rewrite input fits its widest line, including its
// heading and selection, up to the conversation bubble's usual maximum.
func TestBubbleTextWidth(t *testing.T) {
	const maxWidth = BubbleW - 2*BubblePadX
	for _, tc := range []struct {
		name               string
		text, title, quote float32
		want               float32
	}{
		{name: "English word", text: 42, title: 24, want: 42},
		{name: "Chinese word and heading", text: 13, title: 22, want: 22},
		{name: "one line", text: 184, title: 24, want: 184},
		{name: "long paragraph", text: 900, title: 24, want: maxWidth},
		{name: "selection is widest", text: 42, title: 24, quote: 120, want: 120},
	} {
		t.Run(tc.name, func(t *testing.T) {
			if got := BubbleTextWidth(tc.text, tc.title, tc.quote, maxWidth); got != tc.want {
				t.Errorf("BubbleTextWidth(%v, %v, %v, %v) = %v, want %v", tc.text, tc.title, tc.quote, maxWidth, got, tc.want)
			}
		})
	}
}

// TestCardFillW: Fill uses the room beyond Large, keeps Large when it fits,
// and clamps only when Large itself cannot fit. Fixed sizes keep their widths.
func TestCardFillW(t *testing.T) {
	for _, test := range []struct {
		width  int
		margin float64
		want   int
	}{
		{width: 2560, margin: 96, want: 2560 - 2*(96+Pad)},
		{width: 900, margin: 96, want: 760},
		{width: 900, margin: 400, want: 760},
		{width: 800, margin: 96, want: 760},
		{width: 520, margin: 96, want: 484},
	} {
		area := mygo.Rectangle{X: 0, Y: 25, Width: test.width, Height: 600}
		if got := CardFillW(area, test.margin); got != test.want {
			t.Errorf("CardFillW(%d, %.0f) = %d, want %d", test.width, test.margin, got, test.want)
		}
	}
	area := mygo.Rectangle{X: 0, Y: 25, Width: 2560, Height: 1415}
	for step, want := range []int{AnswerW, 600, 760, CardFillW(area, 96)} {
		if got := CardSizes[step].Width(area, 96); got != want {
			t.Errorf("CardSizes[%d].Width = %d, want %d", step, got, want)
		}
	}
}

// TestCardSizeButtonInsideCorner: at any corner radius the settings
// allow (0 to 40), the size button's outer corner stays inside the
// rounded corner: its distance from the corner's centre is at most the
// radius.
func TestCardSizeButtonInsideCorner(t *testing.T) {
	for r := float64(CardSizeBtnInset); r <= 40; r++ {
		if d := (r - CardSizeBtnInset) * math.Sqrt2; d > r {
			t.Errorf("radius %.0f: the button's corner %.2f from the corner's centre, past the rounded corner", r, d)
		}
	}
}
