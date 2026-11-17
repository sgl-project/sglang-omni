package raster

import (
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
