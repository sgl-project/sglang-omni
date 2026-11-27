package main

import (
	"hash/fnv"
	"math"
	"strconv"
	"strings"

	"github.com/egoist/mygo/ui"
)

// Speaker colors encode identity, independently of a translated or corrected
// display name. Transcript text stays neutral; labels remain readable without
// color; larger meetings extend the categorical palette with extra hues.
func speakerColor(id string, dark bool) ui.Color {
	light := []string{"#2459a6", "#7842a0", "#166951", "#9a4318", "#9a305e", "#436516", "#145e7a", "#6551a1"}
	night := []string{"#92baff", "#d5a6f5", "#7dd4b3", "#f5b58a", "#f3a0c3", "#bddc87", "#85cce7", "#b9b0ff"}
	index := 0
	if id != "me" {
		if id == "" {
			if dark {
				return ui.Hex("#b8b8b8")
			}
			return ui.Hex("#626262")
		}
		if n, err := strconv.Atoi(strings.TrimLeft(id, "sp")); err == nil && n > 0 {
			if n >= len(light) {
				return speakerHueColor(float64(n)*137.507764, dark)
			}
			index = n
		} else {
			h := fnv.New32a()
			_, _ = h.Write([]byte(id))
			index = int(h.Sum32() % uint32(len(light)))
		}
	}
	if dark {
		return ui.Hex(night[index])
	}
	return ui.Hex(light[index])
}

// Extend the categorical palette without cycling back onto an existing slot.
func speakerHueColor(degrees float64, dark bool) ui.Color {
	h := math.Mod(degrees, 360) / 60
	x := 1 - math.Abs(math.Mod(h, 2)-1)
	var r, g, b float64
	switch int(h) {
	case 0:
		r, g = 1, x
	case 1:
		r, g = x, 1
	case 2:
		g, b = 1, x
	case 3:
		g, b = x, 1
	case 4:
		r, b = x, 1
	default:
		r, b = 1, x
	}
	base, span := 0.10, 0.55
	target, surface := ui.Hex("#000000"), defaultLightTokens.Bg
	if dark {
		base, span = 0.45, 0.50
		target, surface = ui.Hex("#ffffff"), defaultDarkTokens.Bg
	}
	color := ui.RGB(uint8((base+span*r)*255), uint8((base+span*g)*255), uint8((base+span*b)*255))
	for range 16 {
		if contrastRatio(color, surface) >= 4.5 {
			break
		}
		color = color.Mix(target, 0.1)
	}
	return color
}
