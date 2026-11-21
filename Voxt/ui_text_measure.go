package main

import (
	"math"
	"strings"
	"unicode"
)

// guessText estimates k's size: its width, and its lines of 1.3 times its size.
func guessText(k ovSized) (float32, float32) {
	width := float32(textWidth(k.text, float64(k.size)))
	lineHeight := float32(math.Ceil(float64(k.size) * 1.3))
	if k.wrap <= 0 {
		return width, lineHeight
	}
	lines := 0
	for _, paragraph := range strings.Split(k.text, "\n") {
		paragraphWidth := float32(textWidth(paragraph, float64(k.size)))
		lines += max(1, int(math.Ceil(float64(paragraphWidth/k.wrap))))
	}
	return min(width, k.wrap), float32(lines) * lineHeight
}

// textWidth estimates text width at size DIPs for overlay layout.
func textWidth(text string, size float64) int {
	width := 0.0
	for _, r := range text {
		switch {
		case wideRune(r):
			width += size
		case r == ' ' || r == 'i' || r == 'l' || r == '.' || r == ',' || r == ':':
			width += 0.32 * size
		default:
			width += 0.58 * size
		}
	}
	return int(math.Ceil(width))
}

func wideRune(r rune) bool {
	return unicode.Is(unicode.Han, r) || unicode.In(r, unicode.Hiragana, unicode.Katakana, unicode.Hangul) ||
		r >= 0xFF00 && r <= 0xFFEF || r >= 0x3000 && r <= 0x303F
}
