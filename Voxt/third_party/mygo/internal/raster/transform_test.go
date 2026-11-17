package raster

import (
	"image"
	"testing"

	"github.com/egoist/mygo/internal/scene"
)

func TestAtlasSampleIgnoresAtlasOrigin(t *testing.T) {
	atlas := scene.NewAtlas(1, 4096, 16)
	for _, origin := range []int{7, 1022, 2046, 4094} {
		for y := 0; y < 2; y++ {
			for x := 0; x < 2; x++ {
				atlas.Pix[(9+y)*atlas.W+origin+x] = byte(31 + 67*y + 113*x)
			}
		}
	}
	for _, origin := range []int{1022, 2046, 4094} {
		for numerator := 1; numerator < 1000; numerator++ {
			fraction := float32(numerator) / 1000
			near := atlasSample(atlas, fraction, 9.27, 7, 9, 2, 2)
			far := atlasSample(atlas, fraction, 9.27, origin, 9, 2, 2)
			if far != near {
				t.Fatalf("origin %d, fraction %.5f: near %v, far %v", origin, fraction, near, far)
			}
		}
	}
}

type whiteEffect struct{}

func (whiteEffect) Begin(*scene.EffectOp, scene.Rect, [4]float32) {}

func (whiteEffect) Color(float32, float32, *scene.BackdropImage) [4]float32 {
	return [4]float32{1, 1, 1, 1}
}

func TestGlyphRenderingIgnoresAtlasOrigin(t *testing.T) {
	for _, masked := range []bool{false, true} {
		name := "transformed glyphs"
		if masked {
			name = "masked effect"
		}
		t.Run(name, func(t *testing.T) {
			fractions := []float32{0.0003052, 0.009, 0.1556702, 0.4129944, 0.4902649, 0.5003052, 0.501, 0.999}
			for value := 0; value < 256; value++ {
				for _, fraction := range fractions {
					for _, origin := range []int{1022, 2046, 4094} {
						near := renderGlyphAtOrigin(t, 7, fraction, masked, byte(value))
						far := renderGlyphAtOrigin(t, origin, fraction, masked, byte(value))
						if string(far) != string(near) {
							t.Fatalf("origin %d, fraction %.7f, value %d renders differently", origin, fraction, value)
						}
					}
				}
			}
		})
	}
}

func renderGlyphAtOrigin(t *testing.T, origin int, fraction float32, masked bool, value byte) []byte {
	t.Helper()
	atlas := scene.NewAtlas(1, 4096, 16)
	atlas.Pix[9*atlas.W+origin] = value
	atlas.Pix[9*atlas.W+origin+1] = 0
	atlas.Pix[10*atlas.W+origin] = 0
	atlas.Pix[10*atlas.W+origin+1] = value
	frame := &scene.Scene{
		Width: 4, Height: 16,
		MaskAtlas: atlas,
		Glyphs: []scene.Glyph{{
			Transform: scene.Translation(fraction, 0),
			W:         2, H: 2, U: uint16(origin), V: 9, UW: 2, VH: 2,
			Color: scene.Color{R: 255, G: 255, B: 255, A: 255},
		}},
	}
	op := &scene.Op{Kind: scene.OpGlyphs, Start: 0, End: 1}
	renderer := &renderer{dst: NewImage(4, 16), s: frame, area: image.Rect(0, 0, 4, 16), x0: 0, y0: 0, x1: 4, y1: 16}
	if masked {
		effect := &scene.EffectOp{Glyphs: [2]int32{0, 1}}
		renderer.maskedEffect(op, effect, whiteEffect{})
	} else {
		renderer.transformedGlyphs(op)
	}
	return renderer.dst.Pix
}
