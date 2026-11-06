package text

import (
	"errors"
	"math"

	"github.com/egoist/mygo/internal/scene"
)

// stubEngine lays out text without fonts where the system has no text
// engine MyGo can use: every rune takes half an em and draws nothing.
type stubEngine struct {
	shapeScratch
	fonts map[float32]*Font
}

func (e *stubEngine) font(style Style) *Font {
	size := style.FontSize()
	if f, ok := e.fonts[size]; ok {
		return f
	}
	if e.fonts == nil {
		e.fonts = map[float32]*Font{}
	}
	f := &Font{Size: size, Ascent: size * 0.8, Descent: size * 0.2}
	e.fonts[size] = f
	return f
}

func (e *stubEngine) shape(text []rune, style Style, spans []Span, width float32, rtl, wholeWords bool) []shapedLine {
	f := e.font(style)
	run := shapedRun{font: f, end: len(text)}
	for i := range text {
		run.glyphs = append(run.glyphs, Glyph{Font: f, X: float32(i) * f.Size / 2, Advance: f.Size / 2, Cluster: i, RTL: rtl})
	}
	return []shapedLine{{end: len(text), runs: []shapedRun{run}}}
}

func (e *stubEngine) glyph(f *Font, id uint32, scale, dx float32, _ Shade, _ bool) bitmap {
	return bitmap{}
}

func (e *stubEngine) textParams() (scene.TextParams, bool) { return scene.TextParams{}, false }

func (e *stubEngine) positions(*Font, float32, bool) (int, bool) { return 1, true }

func (e *stubEngine) decorate(decoRange) []Stroke { return nil }

func (e *stubEngine) join() bool { return false }

func (e *stubEngine) baseline(y float32) float32 { return float32(math.Round(float64(y))) }

func (e *stubEngine) appFamilies() []FontFamily { return nil }

func (e *stubEngine) systemFamilies() func() []FontFamily { return nil }

func (e *stubEngine) register(data []byte, family string) error {
	return errors.New("mygo: this system has no text engine to add fonts to")
}
