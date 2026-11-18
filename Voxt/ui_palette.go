package main

import (
	"math"

	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/settings"
)

type paletteColors struct{ accent, background ui.Color }

// The default keeps Voxt's neutral surfaces and original teal accents.
// Only an explicitly selected custom color tints the window and cards.
var defaultLightTokens = makeTokens(false, paletteColors{accent: ui.Hex(settings.DefaultPalette)})
var defaultDarkTokens = makeTokens(true, paletteColors{accent: ui.Hex("#2bb3a3")})

func paletteTokens(hex string, dark bool) *tokens {
	if hex == "" || hex == settings.DefaultPalette {
		if dark {
			return defaultDarkTokens
		}
		return defaultLightTokens
	}
	return derivePaletteTokens(hex, dark)
}

// strongerTint increases the colour in a fill around its luma.
func strongerTint(fill ui.Color) ui.Color {
	luma := 0.2126*float32(fill.R) + 0.7152*float32(fill.G) + 0.0722*float32(fill.B)
	channel := func(value uint8) uint8 {
		return uint8(min(255, max(0, luma+1.4*(float32(value)-luma)+0.5)))
	}
	return ui.RGB(channel(fill.R), channel(fill.G), channel(fill.B))
}

// readableTint keeps as much colour as the body text, captions and links
// can all support at 4.5:1. The neutral surface is the readable endpoint.
func readableTint(fill, base ui.Color, inks ...ui.Color) ui.Color {
	readable := func(surface ui.Color) bool {
		for _, ink := range inks {
			if contrastRatio(ink, surface) < 4.5 {
				return false
			}
		}
		return true
	}
	if readable(fill) {
		return fill
	}
	low, high := 0, 255
	for high-low > 1 {
		mid := (low + high) / 2
		candidate := fill.Mix(base, float32(mid)/255)
		if readable(candidate) {
			high = mid
		} else {
			low = mid
		}
	}
	return fill.Mix(base, float32(high)/255)
}

var linearSRGB = func() [256]float64 {
	var linear [256]float64
	for i := range linear {
		x := float64(i) / 255
		if x <= 0.04045 {
			linear[i] = x / 12.92
		} else {
			linear[i] = math.Pow((x+0.055)/1.055, 2.4)
		}
	}
	return linear
}()

func contrastRatio(a, b ui.Color) float64 {
	luminance := func(c ui.Color) float64 {
		return 0.2126*linearSRGB[c.R] + 0.7152*linearSRGB[c.G] + 0.0722*linearSRGB[c.B]
	}
	x, y := luminance(a), luminance(b)
	return (max(x, y) + 0.05) / (min(x, y) + 0.05)
}

func derivePaletteTokens(hex string, dark bool) *tokens {
	lightness, chroma, hue := ui.Hex(hex).Oklch()
	accentLightness := 0.48 + (lightness-0.55)*0.10
	surfaceLightness := float32(0.986)
	surfaceChroma := min(0.025, chroma*0.32)
	if dark {
		accentLightness = max(0.73, 0.76+(lightness-0.55)*0.12)
		surfaceLightness = 0.228
		surfaceChroma = min(0.037, chroma*0.46)
	}
	return makeTokens(dark, paletteColors{
		accent:     ui.Oklch(accentLightness, min(0.14, chroma*0.85), hue).SRGB(),
		background: ui.Oklch(surfaceLightness, surfaceChroma, hue).SRGB(),
	})
}
