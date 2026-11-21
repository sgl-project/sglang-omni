// Package gputest has what the tests of the GPU renderers share: a scene
// that draws every kind of operation, and a comparison of what a renderer
// drew with what the CPU renderer draws, which the renderers must match.
package gputest

import (
	"fmt"
	"image"
	"image/png"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/egoist/mygo/internal/raster"
	"github.com/egoist/mygo/internal/scene"
)

// Scene returns a 320×600 scene with fills, borders of every width and
// dashed, gradients mixed in sRGB and Oklab, stripes, shadows, blurred or
// not, and cut by the boxes casting them, inner shadows, a hole, nested
// rounded clips, glyphs from both atlases and subpixel ones, plain and in
// gradients, with Direct2D's gamma and contrast, images, in color and in
// gray, and effects, reading their backdrop blurred at each size or not,
// over each other, clipped and at the frame's edge, or reading none.
func Scene() *scene.Scene {
	s := &scene.Scene{Width: 320, Height: 740, Clear: scene.Color{R: 246, G: 247, B: 249, A: 255},
		Text: scene.TextParams{GammaRatios: scene.GammaRatios(1.8), Contrast: 1, SubpixelContrast: 0.5}}
	mask := scene.NewAtlas(1, 64, 64)
	color := scene.NewAtlas(4, 64, 32)
	s.MaskAtlas, s.ColorAtlas = mask, color
	// A disc and a ring as glyph masks, and a color glyph.
	disc := make([]byte, 16*16)
	ring := make([]byte, 12*12)
	for y := range 16 {
		for x := range 16 {
			dx, dy := float64(x)-7.5, float64(y)-7.5
			disc[y*16+x] = byte(255 * clamp(7.5-math.Sqrt(dx*dx+dy*dy)))
			if x < 12 && y < 12 {
				ex, ey := float64(x)-5.5, float64(y)-5.5
				d := math.Sqrt(ex*ex + ey*ey)
				ring[y*12+x] = byte(255 * clamp(1.5-math.Abs(d-4)))
			}
		}
	}
	dx, dy, _ := mask.Alloc(16, 16)
	mask.Put(dx, dy, 16, 16, disc, 16)
	rx, ry, _ := mask.Alloc(12, 12)
	mask.Put(rx, ry, 12, 12, ring, 12)
	emoji := make([]byte, 10*10*4)
	for i := 0; i < len(emoji); i += 4 {
		k := i / 4
		// Premultiplied: half transparent in the corner rows.
		a := byte(255)
		if k < 10 || k >= 90 {
			a = 128
		}
		emoji[i], emoji[i+1], emoji[i+2], emoji[i+3] = byte(uint16(240)*uint16(a)/255), byte(uint16(160)*uint16(a)/255), byte(uint16(20)*uint16(a)/255), a
	}
	cx, cy, _ := color.Alloc(10, 10)
	color.Put(cx, cy, 10, 10, emoji, 40)
	// A subpixel disc: its red subpixels a third of a pixel left of its
	// green ones, its blue ones a third right.
	lcd := make([]byte, 16*16*4)
	for y := range 16 {
		for x := range 16 {
			for c := range 3 {
				dx, dy := float64(x)-7.5+float64(c-1)/3, float64(y)-7.5
				lcd[(y*16+x)*4+c] = byte(255 * clamp(6.5-math.Sqrt(dx*dx+dy*dy)))
			}
		}
	}
	lx, ly, _ := color.Alloc(16, 16)
	color.Put(lx, ly, 16, 16, lcd, 64)
	pix := make([]byte, 8*8*4)
	for y := range 8 {
		for x := range 8 {
			i := (y*8 + x) * 4
			pix[i], pix[i+1], pix[i+2], pix[i+3] = byte(x*32), byte(y*32), 160, 255
		}
	}
	img := scene.NewImageRGBA(8, 8, pix)

	red := scene.Color{R: 220, G: 40, B: 40, A: 255}
	blue := scene.Color{R: 37, G: 99, B: 235, A: 255}
	ink := scene.Color{R: 20, G: 24, B: 32, A: 255}
	r4 := func(r float32) [4]float32 { return [4]float32{r, r, r, r} }
	add := func(op scene.Op) { s.Ops = append(s.Ops, op) }
	add(scene.Op{Kind: scene.OpShadow, Rect: scene.Rect{X: 20, Y: 24, W: 120, H: 80}, Radii: r4(12), Color: scene.Color{A: 90}, Blur: 16})
	add(scene.Op{Kind: scene.OpFill, Rect: scene.Rect{X: 20, Y: 20, W: 120, H: 80}, Radii: r4(12), Color: scene.Color{R: 255, G: 255, B: 255, A: 255},
		Border: scene.Uniform(1), BorderColor: scene.Color{R: 200, G: 204, B: 210, A: 255}})
	add(scene.Op{Kind: scene.OpFill, Rect: scene.Rect{X: 160.5, Y: 20.25, W: 140, H: 30}, Radii: [4]float32{15, 4, 15, 4},
		Color: red, Color2: blue, Paint: scene.PaintLinear, Gradient: [4]float32{160, 20, 300, 50}})
	add(scene.Op{Kind: scene.OpFill, Rect: scene.Rect{X: 170, Y: 60, W: 60, H: 40}, Color: scene.Color{R: 37, G: 99, B: 235, A: 128}, Border: scene.Uniform(3), BorderColor: red, Radii: r4(8)})
	add(scene.Op{Kind: scene.OpFill, Rect: scene.Rect{X: 240, Y: 60, W: 40, H: 40}, Radii: r4(20), Color: blue, Opacity: 0.5})
	add(scene.Op{Kind: scene.OpPushClip, Rect: scene.Rect{X: 20, Y: 120, W: 200, H: 100}, Radii: r4(16)})
	add(scene.Op{Kind: scene.OpFill, Rect: scene.Rect{X: 0, Y: 110, W: 320, H: 130}, Color: scene.Color{R: 230, G: 236, B: 250, A: 255}})
	add(scene.Op{Kind: scene.OpPushClip, Rect: scene.Rect{X: 100, Y: 130, W: 200, H: 70}})
	add(scene.Op{Kind: scene.OpShadow, Rect: scene.Rect{X: 120, Y: 150, W: 60, H: 30}, Radii: r4(6), Color: scene.Color{B: 80, A: 140}, Blur: 10})
	add(scene.Op{Kind: scene.OpFill, Rect: scene.Rect{X: 90, Y: 140, W: 50, H: 50}, Color: red, Radii: r4(4)})
	add(scene.Op{Kind: scene.OpPopClip})
	start := int32(len(s.Glyphs))
	for i := range 6 {
		s.Glyphs = append(s.Glyphs, scene.Glyph{X: float32(30 + i*20), Y: 200, W: 16, H: 16, U: uint16(dx), V: uint16(dy), UW: 16, VH: 16, Color: ink},
			scene.Glyph{X: float32(32 + i*20), Y: 182, W: 12, H: 12, U: uint16(rx), V: uint16(ry), UW: 12, VH: 12, Color: blue})
	}
	s.Glyphs = append(s.Glyphs, scene.Glyph{X: 160, Y: 180, W: 10, H: 10, U: uint16(cx), V: uint16(cy), UW: 10, VH: 10, Color: scene.Color{A: 255}, Colored: true})
	add(scene.Op{Kind: scene.OpGlyphs, Start: start, End: int32(len(s.Glyphs))})
	add(scene.Op{Kind: scene.OpPopClip})
	add(scene.Op{Kind: scene.OpImage, Rect: scene.Rect{X: 240, Y: 120, W: 8, H: 8}, Image: img, Src: scene.Rect{W: 8, H: 8}})
	add(scene.Op{Kind: scene.OpImage, Rect: scene.Rect{X: 240, Y: 140, W: 64, H: 64}, Radii: r4(10), Image: img, Src: scene.Rect{W: 8, H: 8}})
	// Pixel art: the whole image 2.5 times as large, each pixel a square of
	// its own color with smooth edges, and a part of it four times, with
	// none.
	add(scene.Op{Kind: scene.OpImage, Rect: scene.Rect{X: 256, Y: 118, W: 20, H: 20}, Image: img, Src: scene.Rect{W: 8, H: 8}, Pixelated: true})
	add(scene.Op{Kind: scene.OpImage, Rect: scene.Rect{X: 284, Y: 120, W: 16, H: 16}, Image: img, Src: scene.Rect{X: 2, Y: 3, W: 4, H: 4}, Pixelated: true})
	// Sprites: a part flipped across both axes and tinted, pixelated, and
	// one flipped across one, smooth, showing none of the pixels around.
	add(scene.Op{Kind: scene.OpImage, Rect: scene.Rect{X: 284, Y: 60, W: 16, H: 16}, Image: img, Src: scene.Rect{X: 6, Y: 7, W: -4, H: -4}, Pixelated: true,
		Color: scene.Color{R: 255, G: 200, B: 80, A: 220}})
	add(scene.Op{Kind: scene.OpImage, Rect: scene.Rect{X: 284, Y: 80, W: 20, H: 20}, Image: img, Src: scene.Rect{X: 7, Y: 1, W: -5, H: 5}})

	// Borders of different widths, dashed ones, Oklab gradients, stripes,
	// glyphs in gradients and a gray image.
	yellow := scene.Color{R: 250, G: 204, B: 21, A: 255}
	add(scene.Op{Kind: scene.OpFill, Rect: scene.Rect{X: 20, Y: 240, W: 60, H: 30}, Radii: r4(10), Color: scene.Color{R: 255, G: 255, B: 255, A: 255},
		Border: [4]float32{1, 4, 2, 0}, BorderColor: ink})
	add(scene.Op{Kind: scene.OpFill, Rect: scene.Rect{X: 90, Y: 240, W: 60, H: 30}, Radii: r4(8), Border: scene.Uniform(2), BorderColor: blue, Dashed: true})
	add(scene.Op{Kind: scene.OpFill, Rect: scene.Rect{X: 160, Y: 240, W: 70, H: 30}, Color: yellow, Border: [4]float32{0, 0, 3, 1}, BorderColor: red, Dashed: true})
	add(scene.Op{Kind: scene.OpFill, Rect: scene.Rect{X: 240, Y: 240, W: 64, H: 30}, Radii: r4(4), Color: blue, Color2: yellow, Paint: scene.PaintOklab, Gradient: [4]float32{240, 0, 304, 0}})
	add(scene.Op{Kind: scene.OpFill, Rect: scene.Rect{X: 20, Y: 280, W: 80, H: 30}, Color: red, Color2: scene.Color{}, Paint: scene.PaintLinear, Gradient: [4]float32{20, 0, 100, 0}})
	add(scene.Op{Kind: scene.OpFill, Rect: scene.Rect{X: 110, Y: 280, W: 80, H: 30}, Radii: r4(6), Color: scene.Color{R: 37, G: 99, B: 235, A: 200}, Color2: yellow,
		Paint: scene.PaintStripes, Gradient: [4]float32{0.7071, -0.7071, 3, 8}, Border: scene.Uniform(1), BorderColor: ink})
	start = int32(len(s.Glyphs))
	for i := range 3 {
		s.Glyphs = append(s.Glyphs, scene.Glyph{X: float32(200 + i*18), Y: 288, W: 16, H: 16, U: uint16(dx), V: uint16(dy), UW: 16, VH: 16, Color: ink})
	}
	add(scene.Op{Kind: scene.OpGlyphs, Start: start, End: int32(len(s.Glyphs)), Paint: scene.PaintOklab, Color: red, Color2: blue, Gradient: [4]float32{200, 288, 252, 304}})
	add(scene.Op{Kind: scene.OpImage, Rect: scene.Rect{X: 270, Y: 280, W: 32, H: 32}, Radii: r4(6), Image: img, Src: scene.Rect{W: 8, H: 8}, Grayscale: true})

	// Shadows that show only outside the boxes casting them: under a
	// translucent card, without blur, and alone.
	add(scene.Op{Kind: scene.OpFill, Rect: scene.Rect{X: 0, Y: 320, W: 320, H: 60}, Color: yellow})
	card := scene.Rect{X: 20, Y: 330, W: 90, H: 36}
	add(scene.Op{Kind: scene.OpShadow, Rect: scene.Rect{X: 20, Y: 334, W: 90, H: 36}, Radii: r4(10), Color: scene.Color{A: 120}, Blur: 12, Cast: card, CastRadii: r4(10)})
	add(scene.Op{Kind: scene.OpFill, Rect: card, Radii: r4(10), Color: scene.Color{R: 255, G: 255, B: 255, A: 140}})
	card = scene.Rect{X: 130.5, Y: 330.25, W: 70, H: 30}
	add(scene.Op{Kind: scene.OpShadow, Rect: scene.Rect{X: 133.5, Y: 333.25, W: 70, H: 30}, Radii: r4(8), Color: ink, Cast: card, CastRadii: r4(8)})
	add(scene.Op{Kind: scene.OpFill, Rect: card, Radii: r4(8), Color: scene.Color{R: 37, G: 99, B: 235, A: 90}})
	add(scene.Op{Kind: scene.OpShadow, Rect: scene.Rect{X: 218, Y: 334, W: 84, H: 32}, Radii: r4(16), Color: red, Blur: 6,
		Cast: scene.Rect{X: 220, Y: 334, W: 80, H: 28}, CastRadii: r4(14)})
	// A hole through them, rounded and cut by a rounded clip, under a
	// translucent fill and its shadow, as a web view under a popover.
	add(scene.Op{Kind: scene.OpPushClip, Rect: scene.Rect{X: 210, Y: 322, W: 104, H: 56}, Radii: r4(12)})
	add(scene.Op{Kind: scene.OpHole, Rect: scene.Rect{X: 246.5, Y: 326.25, W: 80, H: 40}, Radii: r4(10)})
	add(scene.Op{Kind: scene.OpShadow, Rect: scene.Rect{X: 270, Y: 344, W: 34, H: 30}, Radii: r4(6), Color: scene.Color{A: 100}, Blur: 8})
	add(scene.Op{Kind: scene.OpFill, Rect: scene.Rect{X: 270, Y: 340, W: 34, H: 30}, Radii: r4(6), Color: scene.Color{R: 37, G: 99, B: 235, A: 140}})
	add(scene.Op{Kind: scene.OpPopClip})
	// A hole at half opacity, as a web view in a page fading in.
	add(scene.Op{Kind: scene.OpHole, Rect: scene.Rect{X: 112.5, Y: 354.5, W: 60, H: 20}, Radii: r4(6), Opacity: 0.5})

	// Subpixel glyphs, dark on white and light on dark, half transparent,
	// in a gradient, and clipped; thin mask glyphs, dark and light.
	add(scene.Op{Kind: scene.OpFill, Rect: scene.Rect{X: 0, Y: 380, W: 160, H: 30}, Color: scene.Color{R: 255, G: 255, B: 255, A: 255}})
	add(scene.Op{Kind: scene.OpFill, Rect: scene.Rect{X: 160, Y: 380, W: 160, H: 30}, Color: scene.Color{R: 24, G: 24, B: 27, A: 255}})
	sub := func(x float32, c scene.Color) scene.Glyph {
		return scene.Glyph{X: x, Y: 387, W: 16, H: 16, U: uint16(lx), V: uint16(ly), UW: 16, VH: 16, Color: c, Subpixel: true}
	}
	start = int32(len(s.Glyphs))
	s.Glyphs = append(s.Glyphs, sub(8, ink), sub(28, blue), sub(48, scene.Color{R: 20, G: 24, B: 32, A: 128}),
		sub(168, scene.Color{R: 255, G: 255, B: 255, A: 255}), sub(188, yellow), sub(208, scene.Color{R: 113, G: 113, B: 122, A: 255}),
		scene.Glyph{X: 68, Y: 387, W: 16, H: 16, U: uint16(dx), V: uint16(dy), UW: 16, VH: 16, Color: ink, Thin: true},
		scene.Glyph{X: 228, Y: 387, W: 16, H: 16, U: uint16(dx), V: uint16(dy), UW: 16, VH: 16, Color: scene.Color{R: 244, G: 244, B: 245, A: 255}, Thin: true})
	add(scene.Op{Kind: scene.OpGlyphs, Start: start, End: int32(len(s.Glyphs))})
	start = int32(len(s.Glyphs))
	s.Glyphs = append(s.Glyphs, sub(92, ink), sub(110, ink), sub(128, ink))
	add(scene.Op{Kind: scene.OpGlyphs, Start: start, End: int32(len(s.Glyphs)), Paint: scene.PaintLinear, Color: red, Color2: blue, Gradient: [4]float32{92, 387, 144, 403}})
	add(scene.Op{Kind: scene.OpPushClip, Rect: scene.Rect{X: 250, Y: 384, W: 40, H: 22}, Radii: r4(8)})
	start = int32(len(s.Glyphs))
	s.Glyphs = append(s.Glyphs, sub(248, scene.Color{R: 255, G: 255, B: 255, A: 255}), sub(266, scene.Color{R: 255, G: 255, B: 255, A: 255}), sub(284, red))
	add(scene.Op{Kind: scene.OpGlyphs, Start: start, End: int32(len(s.Glyphs))})
	add(scene.Op{Kind: scene.OpPopClip})

	// Panes of glass over stripes, a gradient and glyphs.
	for i := range 8 {
		c := []scene.Color{red, blue, yellow, ink}[i%4]
		add(scene.Op{Kind: scene.OpFill, Rect: scene.Rect{X: float32(i * 40), Y: 410, W: 20, H: 130}, Color: c})
	}
	add(scene.Op{Kind: scene.OpFill, Rect: scene.Rect{X: 0, Y: 470, W: 320, H: 20}, Color: red, Color2: blue, Paint: scene.PaintOklab, Gradient: [4]float32{0, 0, 320, 0}})
	start = int32(len(s.Glyphs))
	for i := range 14 {
		s.Glyphs = append(s.Glyphs, scene.Glyph{X: float32(6 + i*22), Y: 516, W: 16, H: 16, U: uint16(dx), V: uint16(dy), UW: 16, VH: 16, Color: ink})
	}
	add(scene.Op{Kind: scene.OpGlyphs, Start: start, End: int32(len(s.Glyphs))})
	// Effects over them: reading their backdrop blurred at a quarter of
	// the size, over each other at the same size, unblurred, clipped and
	// half transparent at full size, past the frame's edge at half the
	// size; and one reading none.
	effect := func(e *scene.Effect, r scene.Rect, radii [4]float32, blur, shift float32, tint scene.Color, opacity float32) {
		fx := scene.EffectOp{Effect: e, Blur: blur}
		fx.Params[0] = [4]float32{shift}
		fx.Params[1] = [4]float32{float32(tint.R) / 255, float32(tint.G) / 255, float32(tint.B) / 255, float32(tint.A) / 255}
		add(scene.Op{Kind: scene.OpEffect, Rect: r, Radii: radii, Start: int32(len(s.Effects)), Opacity: opacity})
		s.Effects = append(s.Effects, fx)
	}
	white := scene.Color{R: 255, G: 255, B: 255, A: 90}
	add(scene.Op{Kind: scene.OpShadow, Rect: scene.Rect{X: 12, Y: 424, W: 140, H: 44}, Radii: r4(22), Color: scene.Color{A: 40}, Blur: 16,
		Cast: scene.Rect{X: 12, Y: 420, W: 140, H: 44}, CastRadii: r4(22)})
	effect(LensEffect, scene.Rect{X: 12, Y: 420, W: 140, H: 44}, r4(22), 8, 2, white, 0)
	effect(LensEffect, scene.Rect{X: 120, Y: 446, W: 60, H: 60}, r4(30), 12, 3, white, 0)
	effect(LensEffect, scene.Rect{X: 190.5, Y: 418.25, W: 110, H: 50}, r4(16), 0, 1.5, scene.Color{R: 37, G: 99, B: 235, A: 120}, 0)
	add(scene.Op{Kind: scene.OpPushClip, Rect: scene.Rect{X: 30, Y: 480, W: 200, H: 50}, Radii: r4(12)})
	effect(LensEffect, scene.Rect{X: 40, Y: 476, W: 240, H: 44}, [4]float32{22, 6, 22, 6}, 3, 1, scene.Color{A: 60}, 0.7)
	add(scene.Op{Kind: scene.OpPopClip})
	effect(LensEffect, scene.Rect{X: 284, Y: 494, W: 50, H: 50}, r4(14), 6, 2, white, 0)
	effect(TintEffect, scene.Rect{X: 250, Y: 420, W: 30, H: 30}, r4(8), 0, 0, scene.Color{R: 220, G: 40, B: 40, A: 160}, 0)

	// Inner shadows: a hairline blurred along the edge of a dark card,
	// offset and spread, without blur, and with no hole left.
	inset := func(box scene.Rect, radius float32, hole scene.Rect, holeRadius float32, c scene.Color, blur float32) {
		add(scene.Op{Kind: scene.OpFill, Rect: box, Radii: r4(radius), Color: scene.Color{R: 39, G: 39, B: 42, A: 255}})
		add(scene.Op{Kind: scene.OpShadow, Inset: true, Rect: hole, Radii: r4(holeRadius), Color: c, Blur: blur, Cast: box, CastRadii: r4(radius)})
	}
	inset(scene.Rect{X: 10, Y: 548, W: 90, H: 44}, 10, scene.Rect{X: 10, Y: 548, W: 90, H: 44}, 10, scene.Color{R: 255, G: 255, B: 255, A: 120}, 2)
	inset(scene.Rect{X: 110.5, Y: 548.25, W: 90, H: 44}, 14, scene.Rect{X: 117.5, Y: 556.25, W: 82, H: 36}, 10, yellow, 8)
	inset(scene.Rect{X: 210, Y: 548, W: 50, H: 44}, 12, scene.Rect{X: 216, Y: 555, W: 38, H: 30}, 6, red, 0)
	inset(scene.Rect{X: 270, Y: 548, W: 40, H: 44}, 20, scene.Rect{}, 0, blue, 6)

	// Gradients of stops: conic, in sRGB with a hard stop and in Oklab,
	// linear, half transparent and to transparent, painting borders, plain
	// and dashed, and glyphs; and images blurred, at full size and a
	// quarter of it.
	stops := func(st ...scene.Stop) [2]int32 {
		a := int32(len(s.Stops))
		s.Stops = append(s.Stops, st...)
		return [2]int32{a, int32(len(s.Stops))}
	}
	green := scene.Color{R: 34, G: 197, B: 94, A: 255}
	add(scene.Op{Kind: scene.OpFill, Rect: scene.Rect{X: 10, Y: 605, W: 60, H: 60}, Radii: r4(30), Paint: scene.PaintConic,
		Gradient: [4]float32{40, 635, 0.4, 0}, Stops: stops(scene.Stop{Color: red, At: 0}, scene.Stop{Color: yellow, At: 0.3}, scene.Stop{Color: blue, At: 0.3}, scene.Stop{Color: green, At: 0.7}, scene.Stop{Color: red, At: 1})})
	add(scene.Op{Kind: scene.OpFill, Rect: scene.Rect{X: 80.5, Y: 605.25, W: 60, H: 60}, Radii: r4(8), Paint: scene.PaintConicOklab,
		Gradient: [4]float32{100, 625, -1, 0}, Stops: stops(scene.Stop{Color: blue, At: 0.1}, scene.Stop{Color: yellow, At: 0.5}, scene.Stop{Color: red, At: 0.9})})
	add(scene.Op{Kind: scene.OpFill, Rect: scene.Rect{X: 150, Y: 605, W: 70, H: 25}, Radii: r4(6), Paint: scene.PaintLinearStops,
		Gradient: [4]float32{150, 605, 220, 630}, Stops: stops(scene.Stop{Color: red, At: 0.1}, scene.Stop{Color: scene.Color{R: 37, G: 99, B: 235, A: 120}, At: 0.5}, scene.Stop{Color: yellow, At: 0.9})})
	add(scene.Op{Kind: scene.OpFill, Rect: scene.Rect{X: 150, Y: 637, W: 70, H: 25}, Paint: scene.PaintLinearStopsOklab,
		Gradient: [4]float32{150, 0, 220, 0}, Stops: stops(scene.Stop{Color: blue, At: 0}, scene.Stop{Color: green, At: 0.6}, scene.Stop{Color: scene.Color{}, At: 1})})
	add(scene.Op{Kind: scene.OpFill, Rect: scene.Rect{X: 230.5, Y: 605.25, W: 80, H: 60}, Radii: r4(14), Color: scene.Color{R: 255, G: 255, B: 255, A: 120},
		Border: scene.Uniform(6), BorderColor: scene.Color{R: 255, G: 255, B: 255, A: 255}, BorderPaint: true, Paint: scene.PaintConicOklab,
		Gradient: [4]float32{270, 635, 0, 0}, Stops: stops(scene.Stop{Color: red, At: 0}, scene.Stop{Color: blue, At: 0.5}, scene.Stop{Color: red, At: 1})})
	add(scene.Op{Kind: scene.OpFill, Rect: scene.Rect{X: 230, Y: 670, W: 80, H: 30}, Radii: r4(6), Border: scene.Uniform(3),
		BorderColor: scene.Color{R: 255, G: 255, B: 255, A: 200}, BorderPaint: true, Dashed: true, Paint: scene.PaintLinearStops,
		Gradient: [4]float32{230, 0, 310, 0}, Stops: stops(scene.Stop{Color: green, At: 0}, scene.Stop{Color: red, At: 1})})
	start = int32(len(s.Glyphs))
	for i := range 3 {
		s.Glyphs = append(s.Glyphs, scene.Glyph{X: float32(10 + i*18), Y: 676, W: 16, H: 16, U: uint16(dx), V: uint16(dy), UW: 16, VH: 16, Color: ink})
	}
	add(scene.Op{Kind: scene.OpGlyphs, Start: start, End: int32(len(s.Glyphs)), Paint: scene.PaintConic,
		Gradient: [4]float32{36, 684, 0, 0}, Stops: stops(scene.Stop{Color: red, At: 0}, scene.Stop{Color: blue, At: 0.5}, scene.Stop{Color: yellow, At: 1})})
	add(scene.Op{Kind: scene.OpImage, Rect: scene.Rect{X: 70, Y: 672, W: 48, H: 48}, Image: img, Src: scene.Rect{W: 8, H: 8}, Blur: 1.2})
	big := make([]byte, 64*64*4)
	for y := range 64 {
		for x := range 64 {
			i := (y*64 + x) * 4
			if x >= 16 && x < 48 && y >= 20 && y < 44 {
				big[i], big[i+1], big[i+2], big[i+3] = 220, 40, 40, 255
			} else if (x/8+y/8)%2 == 0 && x > 4 && x < 60 && y > 4 && y < 60 {
				big[i], big[i+1], big[i+2], big[i+3] = 18, 50, 117, 128
			}
		}
	}
	add(scene.Op{Kind: scene.OpImage, Rect: scene.Rect{X: 130, Y: 672, W: 64, H: 64}, Radii: r4(8), Image: scene.NewImageRGBA(64, 64, big), Src: scene.Rect{W: 64, H: 64}, Blur: 12})
	return s
}

// Compare compares pixels a renderer drew, premultiplied BGRA rows of
// stride bytes, with what the CPU renderer draws of s. Antialiased edges
// and interpolated images may differ a little; anything else fails t. With
// MYGO_GPU_PNG naming a directory, it saves both images there.
func Compare(t testing.TB, name string, got []byte, stride int, s *scene.Scene) {
	t.Helper()
	want := raster.NewImage(s.Width, s.Height)
	raster.Render(want, s)
	worst, off := 0, 0
	worstAt := image.Point{}
	for y := range s.Height {
		for x := range s.Width {
			g := got[y*stride+4*x:][:4]
			w := want.Pix[y*want.Stride+4*x:][:4]
			d := 0
			for c := range 4 {
				d = max(d, absInt(int(g[c])-int(w[c])))
			}
			if d > worst {
				worst, worstAt = d, image.Pt(x, y)
			}
			if d > 2 {
				off++
			}
		}
	}
	if dir := os.Getenv("MYGO_GPU_PNG"); dir != "" {
		g := &raster.Image{W: s.Width, H: s.Height, Stride: stride, Pix: got}
		save(t, filepath.Join(dir, name+".png"), g)
		save(t, filepath.Join(dir, name+"-cpu.png"), want)
	}
	// One pixel in two hundred may differ by more than 2 of 255, none by
	// more than 24.
	if worst > 24 || off*200 > s.Width*s.Height {
		t.Errorf("%s: %d pixels differ from the CPU renderer's by more than 2, the most by %d at %v", name, off, worst, worstAt)
	}
}

func save(t testing.TB, path string, m *raster.Image) {
	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	img := &image.RGBA{Pix: m.RGBA(), Stride: 4 * m.W, Rect: image.Rect(0, 0, m.W, m.H)}
	if err := png.Encode(f, img); err != nil {
		t.Fatal(fmt.Errorf("%s: %w", path, err))
	}
}

func clamp(v float64) float64 { return max(0, min(1, v)) }

func absInt(v int) int {
	if v < 0 {
		return -v
	}
	return v
}

// ContinuousScene returns a 320×70 scene with continuous corners, which
// only the renderers of macOS draw (scene.Op.Continuous): a card with a
// border, its shadow and an inner one, a pill, a circle, which stays one,
// sides too short for their corners' curves, one with an inner shadow
// without blur, a corner reaching past the middle of its sides, a clip and
// an image, a shadow without blur, cut by the box casting it, and an effect
// reading its backdrop.
func ContinuousScene() *scene.Scene {
	s := &scene.Scene{Width: 320, Height: 70, Clear: scene.Color{R: 236, G: 238, B: 242, A: 255}}
	pix := make([]byte, 8*8*4)
	for y := range 8 {
		for x := range 8 {
			i := (y*8 + x) * 4
			pix[i], pix[i+1], pix[i+2], pix[i+3] = byte(x*32), byte(y*32), 160, 255
		}
	}
	img := scene.NewImageRGBA(8, 8, pix)
	red := scene.Color{R: 220, G: 40, B: 40, A: 255}
	blue := scene.Color{R: 37, G: 99, B: 235, A: 255}
	ink := scene.Color{R: 20, G: 24, B: 32, A: 255}
	yellow := scene.Color{R: 250, G: 204, B: 21, A: 255}
	r4 := func(r float32) [4]float32 { return [4]float32{r, r, r, r} }
	add := func(op scene.Op) {
		op.Continuous = true
		s.Ops = append(s.Ops, op)
	}
	card := scene.Rect{X: 10, Y: 8, W: 70, H: 44}
	add(scene.Op{Kind: scene.OpShadow, Rect: scene.Rect{X: 10, Y: 12, W: 70, H: 44}, Radii: r4(14), Color: scene.Color{A: 110}, Blur: 10,
		Cast: card, CastRadii: r4(14)})
	add(scene.Op{Kind: scene.OpFill, Rect: card, Radii: r4(14), Color: scene.Color{R: 255, G: 255, B: 255, A: 255},
		Border: scene.Uniform(2), BorderColor: blue})
	pad, padRadii := scene.InnerRadii(card, r4(14), scene.Uniform(2))
	add(scene.Op{Kind: scene.OpShadow, Inset: true, Rect: scene.Rect{X: pad.X + 3, Y: pad.Y + 5, W: pad.W - 2, H: pad.H - 2}, Radii: [4]float32{11, 11, 11, 11},
		Color: scene.Color{R: 220, G: 40, B: 40, A: 160}, Blur: 8, Cast: pad, CastRadii: padRadii})
	add(scene.Op{Kind: scene.OpFill, Rect: scene.Rect{X: 88.5, Y: 10.25, W: 70, H: 22}, Radii: r4(999), Color: red, Color2: blue,
		Paint: scene.PaintLinear, Gradient: [4]float32{88, 10, 158, 32}})
	add(scene.Op{Kind: scene.OpFill, Rect: scene.Rect{X: 88, Y: 38, W: 24, H: 24}, Radii: r4(12), Color: ink})
	add(scene.Op{Kind: scene.OpFill, Rect: scene.Rect{X: 118, Y: 38, W: 40, H: 24}, Radii: r4(10), Color: blue, Border: scene.Uniform(1), BorderColor: ink})
	add(scene.Op{Kind: scene.OpShadow, Inset: true, Rect: scene.Rect{X: 120, Y: 41, W: 36, H: 20}, Radii: r4(8), Color: yellow,
		Cast: scene.Rect{X: 119, Y: 39, W: 38, H: 22}, CastRadii: r4(9)})
	add(scene.Op{Kind: scene.OpFill, Rect: scene.Rect{X: 166.5, Y: 8.25, W: 40, H: 54}, Radii: [4]float32{26, 6, 0, 0}, Color: red})
	add(scene.Op{Kind: scene.OpPushClip, Rect: scene.Rect{X: 214, Y: 6, W: 50, H: 58}, Radii: r4(16)})
	add(scene.Op{Kind: scene.OpFill, Rect: scene.Rect{X: 204, Y: 0, W: 70, H: 70}, Color: yellow})
	add(scene.Op{Kind: scene.OpImage, Rect: scene.Rect{X: 222, Y: 14, W: 34, H: 34}, Radii: r4(9), Image: img, Src: scene.Rect{W: 8, H: 8}})
	add(scene.Op{Kind: scene.OpPopClip})
	card = scene.Rect{X: 272.5, Y: 10.25, W: 40, H: 40}
	add(scene.Op{Kind: scene.OpShadow, Rect: scene.Rect{X: 275.5, Y: 14.25, W: 40, H: 40}, Radii: r4(12), Color: ink, Cast: card, CastRadii: r4(12)})
	// An effect reading its backdrop, over the pill and the card.
	fx := scene.EffectOp{Effect: LensEffect, Blur: 4}
	fx.Params[0] = [4]float32{2}
	fx.Params[1] = [4]float32{1, 1, 1, 0.35}
	add(scene.Op{Kind: scene.OpEffect, Rect: scene.Rect{X: 60, Y: 18, W: 70, H: 30}, Radii: r4(15), Start: int32(len(s.Effects))})
	s.Effects = append(s.Effects, fx)
	return s
}

// ShadowCase is a panel with a stack of shadows cast by it, as an elevated
// surface has: its corner radius, the blur, spread and offset of each layer.
type ShadowCase struct {
	Radius     float32
	Continuous bool
	// Blur, Spread and DY describe each layer.
	Blur, Spread, DY []float32
}

// ShadowPanelScene returns a 480×360 scene with a panel of 340×220 that casts
// the shadows of c, each shown only outside the panel (Cast), as kit surfaces
// draw them, and a fill over the panel.
func ShadowPanelScene(c ShadowCase) *scene.Scene {
	s := &scene.Scene{Width: 480, Height: 360, Clear: scene.Color{R: 240, G: 241, B: 244, A: 255}}
	panel := scene.Rect{X: 70, Y: 60, W: 340, H: 220}
	radii := [4]float32{c.Radius, c.Radius, c.Radius, c.Radius}
	for i := range c.Blur {
		sp := c.Spread[i]
		box := scene.Rect{X: panel.X - sp, Y: panel.Y - sp + c.DY[i], W: panel.W + 2*sp, H: panel.H + 2*sp}
		s.Ops = append(s.Ops, scene.Op{Kind: scene.OpShadow, Rect: box, Radii: radii, Continuous: c.Continuous,
			Color: scene.Color{A: 60}, Blur: c.Blur[i], Cast: panel, CastRadii: radii})
	}
	s.Ops = append(s.Ops, scene.Op{Kind: scene.OpFill, Rect: panel, Radii: radii, Continuous: c.Continuous,
		Color: scene.Color{R: 255, G: 255, B: 255, A: 255}})
	return s
}

// ShadowCases are panels of several radii, blurs, spreads and offsets,
// among them an offset larger than the blur and a shadow that does not reach
// out of the panel.
func ShadowCases() []ShadowCase {
	return []ShadowCase{
		{Radius: 0, Blur: []float32{2, 24}, Spread: []float32{0, 0}, DY: []float32{1, 12}},
		{Radius: 12, Blur: []float32{2, 6, 12, 24}, Spread: []float32{0, -1, -3, -6}, DY: []float32{1, 3, 6, 12}},
		{Radius: 40, Blur: []float32{16}, Spread: []float32{0}, DY: []float32{4}},
		{Radius: 16, Continuous: true, Blur: []float32{12, 30}, Spread: []float32{2, 0}, DY: []float32{3, 8}},
		{Radius: 8, Blur: []float32{4}, Spread: []float32{-2}, DY: []float32{40}},
		{Radius: 60, Blur: []float32{8, 0}, Spread: []float32{0, 5}, DY: []float32{0, 0}},
		{Radius: 30, Blur: []float32{60}, Spread: []float32{-40}, DY: []float32{-10}},
	}
}

// RingCase is a scene of border-only fills (a transparent solid color and a
// border), over content that shows when one of them wrongly changes a pixel.
type RingCase struct {
	Name  string
	Scene *scene.Scene
	// Split is whether the builder draws the first ring around its core.
	Split bool
}

// RingCases are border-only fills at the size of panels (the shaders shade
// their cores for nothing unless the builder draws around them): square,
// round and continuous corners, uneven widths and radii, a dashed border,
// fractional places, borders under clips, translucent and opaque border
// colors, at half opacity, one with a border wider than half the panel,
// and fills that are not split: a gradient, a translucent color, a
// transformed fill, a small one and a thin one.
func RingCases() []RingCase {
	blue := scene.Color{R: 37, G: 99, B: 235, A: 255}
	red := scene.Color{R: 220, G: 40, B: 40, A: 255}
	paper := scene.Color{R: 246, G: 247, B: 249, A: 255}
	r4 := func(r float32) [4]float32 { return [4]float32{r, r, r, r} }
	// content returns a 480×360 scene of fills, a shadow and a glyph run,
	// which the ring is drawn over, and the clip to push, if any.
	content := func() *scene.Scene {
		s := &scene.Scene{Width: 480, Height: 360, Clear: scene.Color{R: 30, G: 34, B: 40, A: 255},
			Text: scene.TextParams{GammaRatios: scene.GammaRatios(1.8), Contrast: 1, SubpixelContrast: 0.5}}
		s.MaskAtlas, s.Text, s.Glyphs = glyphContent()
		s.Ops = append(s.Ops,
			scene.Op{Kind: scene.OpFill, Rect: scene.Rect{X: 10, Y: 10, W: 460, H: 340}, Radii: r4(20), Color: paper},
			scene.Op{Kind: scene.OpFill, Rect: scene.Rect{X: 100, Y: 80, W: 200, H: 120}, Color: blue, Color2: red, Paint: scene.PaintLinear, Gradient: [4]float32{100, 0, 300, 0}},
			scene.Op{Kind: scene.OpShadow, Rect: scene.Rect{X: 160, Y: 160, W: 160, H: 100}, Radii: r4(12), Color: scene.Color{A: 120}, Blur: 24})
		s.Ops = append(s.Ops, scene.Op{Kind: scene.OpGlyphs, Start: 0, End: int32(len(s.Glyphs))})
		return s
	}
	var out []RingCase
	add := func(name string, split bool, clip *scene.Op, ops ...scene.Op) {
		s := content()
		if clip != nil {
			s.Ops = append(s.Ops, *clip)
		}
		s.Ops = append(s.Ops, ops...)
		if clip != nil {
			s.Ops = append(s.Ops, scene.Op{Kind: scene.OpPopClip})
		}
		out = append(out, RingCase{name, s, split})
	}
	ring := func(r scene.Rect, radii [4]float32, w [4]float32, c scene.Color) scene.Op {
		return scene.Op{Kind: scene.OpFill, Rect: r, Radii: radii, Border: w, BorderColor: c}
	}
	panel := scene.Rect{X: 20, Y: 20, W: 440, H: 320}
	half := scene.Color{R: 255, G: 255, B: 255, A: 90}
	add("square 1px", true, nil, ring(panel, [4]float32{}, scene.Uniform(1), half))
	add("square 2px opaque", true, nil, ring(panel, [4]float32{}, scene.Uniform(2), red))
	add("rounded", true, nil, ring(panel, r4(16), scene.Uniform(2), half))
	add("continuous", true, nil, func() scene.Op {
		op := ring(panel, r4(24), scene.Uniform(2), half)
		op.Continuous = true
		return op
	}())
	add("uneven widths and radii", true, nil, ring(panel, [4]float32{30, 0, 8, 20}, [4]float32{1, 3, 2, 5}, half))
	add("uneven continuous", true, nil, func() scene.Op {
		op := ring(panel, [4]float32{30, 6, 8, 20}, [4]float32{3, 1, 4, 2}, half)
		op.Continuous = true
		return op
	}())
	add("fractional place", true, nil, ring(scene.Rect{X: 20.5, Y: 20.25, W: 439.3, H: 320.6}, r4(12), scene.Uniform(1.5), half))
	add("dashed", true, nil, func() scene.Op {
		op := ring(panel, r4(10), scene.Uniform(2), blue)
		op.Dashed = true
		return op
	}())
	add("half opacity", true, nil, func() scene.Op {
		op := ring(panel, r4(10), scene.Uniform(3), blue)
		op.Opacity = 0.5
		return op
	}())
	add("wide border", true, nil, ring(panel, r4(40), scene.Uniform(60), half))
	add("clipped", true, &scene.Op{Kind: scene.OpPushClip, Rect: scene.Rect{X: 30, Y: 30, W: 300, H: 280}}, ring(panel, r4(16), scene.Uniform(2), half))
	add("fractional clip", true, &scene.Op{Kind: scene.OpPushClip, Rect: scene.Rect{X: 30.5, Y: 30.25, W: 300.4, H: 280.7}}, ring(panel, r4(16), scene.Uniform(2), half))
	add("rounded clip", true, &scene.Op{Kind: scene.OpPushClip, Rect: scene.Rect{X: 30, Y: 30, W: 300, H: 280}, Radii: r4(30)}, ring(panel, r4(16), scene.Uniform(2), half))
	add("frame edges", true, nil, ring(scene.Rect{X: -10, Y: -10, W: 500, H: 380}, r4(12), scene.Uniform(2), half))
	add("two rings", true, nil, ring(panel, r4(16), scene.Uniform(2), half), ring(scene.Rect{X: 30, Y: 30, W: 420, H: 300}, r4(8), scene.Uniform(1), red))
	add("a fill with a border", false, nil, func() scene.Op {
		op := ring(panel, r4(16), scene.Uniform(2), half)
		op.Color = scene.Color{R: 255, G: 255, B: 255, A: 1}
		return op
	}())
	add("gradient", false, nil, func() scene.Op {
		op := ring(panel, r4(16), scene.Uniform(2), half)
		op.Paint, op.Color, op.Color2, op.Gradient = scene.PaintLinear, scene.Color{A: 0}, scene.Color{R: 255, A: 200}, [4]float32{20, 0, 460, 0}
		return op
	}())
	add("transformed", false, nil, func() scene.Op {
		op := ring(panel, r4(16), scene.Uniform(2), half)
		op.Transform = scene.Scaling(0.98, 0.98)
		return op
	}())
	add("small", false, nil, ring(scene.Rect{X: 100, Y: 100, W: 40, H: 30}, r4(6), scene.Uniform(1), half))
	add("thin", false, nil, ring(scene.Rect{X: 20, Y: 20, W: 440, H: 6}, r4(2), scene.Uniform(1), half))
	add("border fills the panel", false, nil, ring(scene.Rect{X: 20, Y: 20, W: 100, H: 100}, [4]float32{}, scene.Uniform(60), half))
	return out
}

// OpaqueScene returns a w×h scene (at least 600×500) of opaque fills large
// enough for the builder to draw as cores without blending, over content that
// shows if one is wrong: translucent shadows, glyphs and half transparent
// fills under them, and fills with square, round and continuous corners, a
// translucent border, fractional places, edges at the frame's, rectangular
// clips with whole and fractional edges, and some that are not cores (half
// opacity, a gradient, a rounded clip, a transformed fill).
func OpaqueScene(w, h int) *scene.Scene {
	s := &scene.Scene{Width: w, Height: h, Clear: scene.Color{R: 30, G: 34, B: 40, A: 255}}
	s.MaskAtlas, s.Text, s.Glyphs = glyphContent()
	fw, fh := float32(w), float32(h)
	r4 := func(r float32) [4]float32 { return [4]float32{r, r, r, r} }
	add := func(op scene.Op) { s.Ops = append(s.Ops, op) }
	color := func(i int) scene.Color {
		return scene.Color{R: byte(20 + i*37%200), G: byte(10 + i*53%220), B: byte(40 + i*71%200), A: 255}
	}
	under := func(x, y float32) {
		add(scene.Op{Kind: scene.OpShadow, Rect: scene.Rect{X: x, Y: y, W: 160, H: 100}, Radii: r4(12), Color: scene.Color{A: 140}, Blur: 20})
		add(scene.Op{Kind: scene.OpFill, Rect: scene.Rect{X: x + 20, Y: y + 20, W: 200, H: 80}, Color: scene.Color{R: 255, G: 90, B: 20, A: 160}})
		add(scene.Op{Kind: scene.OpGlyphs, Start: 0, End: int32(len(s.Glyphs))})
	}
	under(fw/6, 80)
	add(scene.Op{Kind: scene.OpFill, Rect: scene.Rect{W: fw, H: fh}, Color: color(1)})
	under(fw/6, 100)
	add(scene.Op{Kind: scene.OpFill, Rect: scene.Rect{X: 40, Y: 30.25, W: fw - 80.5, H: fh - 60}, Radii: r4(24), Continuous: true, Color: color(2),
		Border: scene.Uniform(2), BorderColor: scene.Color{R: 255, G: 255, B: 255, A: 70}})
	under(fw/3, 200)
	add(scene.Op{Kind: scene.OpFill, Rect: scene.Rect{X: 60.5, Y: 70, W: fw - 140, H: fh - 140.5}, Color: color(3)})
	add(scene.Op{Kind: scene.OpFill, Rect: scene.Rect{X: 100, Y: 90, W: fw/2 + 0.5, H: fh/2 - 3.25}, Radii: r4(12), Color: color(4)})
	add(scene.Op{Kind: scene.OpFill, Rect: scene.Rect{X: fw / 3, Y: 120, W: fw / 2, H: fh / 2}, Radii: [4]float32{40, 0, 16, 5}, Color: color(5)})
	add(scene.Op{Kind: scene.OpFill, Rect: scene.Rect{X: -5, Y: fh - 130, W: fw + 10, H: 130}, Color: color(6)})
	add(scene.Op{Kind: scene.OpFill, Rect: scene.Rect{X: 0, Y: 0, W: 150, H: fh}, Color: color(7)})
	add(scene.Op{Kind: scene.OpPushClip, Rect: scene.Rect{X: 200, Y: 150, W: fw - 300, H: fh - 250}})
	under(260, 200)
	add(scene.Op{Kind: scene.OpFill, Rect: scene.Rect{X: 190, Y: 140, W: fw - 280, H: fh - 230}, Color: color(8)})
	add(scene.Op{Kind: scene.OpPopClip})
	add(scene.Op{Kind: scene.OpPushClip, Rect: scene.Rect{X: 210.5, Y: 160.25, W: fw - 320.5, H: fh - 270.25}})
	add(scene.Op{Kind: scene.OpFill, Rect: scene.Rect{X: 190, Y: 140, W: fw - 280, H: fh - 230}, Color: color(9)})
	add(scene.Op{Kind: scene.OpPopClip})
	add(scene.Op{Kind: scene.OpPushClip, Rect: scene.Rect{X: 220, Y: 170, W: fw - 340, H: fh - 290}, Radii: r4(30)})
	add(scene.Op{Kind: scene.OpFill, Rect: scene.Rect{X: 190, Y: 140, W: fw - 280, H: fh - 230}, Color: color(10)})
	add(scene.Op{Kind: scene.OpPopClip})
	add(scene.Op{Kind: scene.OpFill, Rect: scene.Rect{X: 250, Y: 200, W: 300, H: 200}, Color: color(11), Opacity: 0.5})
	add(scene.Op{Kind: scene.OpFill, Rect: scene.Rect{X: 250, Y: 200, W: 300, H: 200}, Color: color(12), Color2: color(13), Paint: scene.PaintLinear, Gradient: [4]float32{250, 0, 550, 0}})
	add(scene.Op{Kind: scene.OpFill, Rect: scene.Rect{X: 260, Y: 210, W: 280, H: 180}, Color: color(14), Transform: scene.Scaling(1.01, 1)})
	add(scene.Op{Kind: scene.OpFill, Rect: scene.Rect{X: 300, Y: 250, W: 120, H: 90}, Color: color(15)}) // small: not a core
	return s
}

// CullCase is a scene of ops, some of them under later opaque fills, and how
// many of them a renderer may leave out, which must change no pixel. The
// identity tests use it as a named scene.
type CullCase struct {
	Name  string
	Scene *scene.Scene
	// Culled is how many ops the shared builder leaves out.
	Culled int
}

// glyphContent returns a mask atlas holding a disc, the text parameters, and a run
// of twelve glyphs of it, for scenes that draw text.
func glyphContent() (*scene.Atlas, scene.TextParams, []scene.Glyph) {
	atlas := scene.NewAtlas(1, 64, 64)
	disc := make([]byte, 16*16)
	for y := range 16 {
		for x := range 16 {
			dx, dy := float64(x)-7.5, float64(y)-7.5
			disc[y*16+x] = byte(255 * clamp(7.5-math.Sqrt(dx*dx+dy*dy)))
		}
	}
	x, y, _ := atlas.Alloc(16, 16)
	atlas.Put(x, y, 16, 16, disc, 16)
	var glyphs []scene.Glyph
	for i := range 12 {
		glyphs = append(glyphs, scene.Glyph{X: 40 + float32(i*16), Y: 120, W: 16, H: 16, U: uint16(x), V: uint16(y), UW: 16, VH: 16, Color: scene.Color{A: 255}})
	}
	return atlas, scene.TextParams{GammaRatios: scene.GammaRatios(1.8), Contrast: 1, SubpixelContrast: 0.5}, glyphs
}

// Transformed returns a copy of s with a one pixel transformed fill drawn
// first, which makes the builder give rounded and rotated clips masks (a scene
// with any transformed op has them), so that every case also runs with masked
// clips.
func Transformed(s *scene.Scene) *scene.Scene {
	c := *s
	c.Ops = append([]scene.Op{{Kind: scene.OpFill, Rect: scene.Rect{X: 1, Y: 1, W: 3, H: 3}, Color: scene.Color{R: 200, A: 255}, Transform: scene.Scaling(1.01, 1)}}, s.Ops...)
	return &c
}

// MaskedCase is a named scene whose clips are masked.
type MaskedCase struct {
	Name  string
	Scene *scene.Scene
}

// MaskedCases are scenes with rounded, continuous and rotated clips (which
// are masks in a scene with a transformed op) holding what the builder
// draws in pieces or as cores: a border-only ring, a shadow cast by a box, large
// opaque fills, plain fills, glyphs and holes, with content after them in the
// same clip that reaches the clip's corners.
func MaskedCases() []MaskedCase {
	r4 := func(r float32) [4]float32 { return [4]float32{r, r, r, r} }
	build := func(name string, clip scene.Op, inner func(add func(scene.Op))) MaskedCase {
		s := &scene.Scene{Width: 600, Height: 500, Clear: scene.Color{R: 240, G: 240, B: 240, A: 255}}
		s.MaskAtlas, s.Text, s.Glyphs = glyphContent()
		s.Ops = append(s.Ops, scene.Op{Kind: scene.OpFill, Rect: scene.Rect{X: 5, Y: 5, W: 10, H: 10}, Color: scene.Color{R: 200, A: 255}, Transform: scene.Scaling(1.01, 1)})
		s.Ops = append(s.Ops, scene.Op{Kind: scene.OpFill, Rect: scene.Rect{W: 600, H: 500}, Color: scene.Color{R: 90, G: 120, B: 200, A: 255}})
		clip.Kind = scene.OpPushClip
		s.Ops = append(s.Ops, clip)
		inner(func(op scene.Op) { s.Ops = append(s.Ops, op) })
		s.Ops = append(s.Ops, scene.Op{Kind: scene.OpFill, Rect: scene.Rect{X: 0, Y: 0, W: 600, H: 120}, Color: scene.Color{G: 160, A: 200}},
			scene.Op{Kind: scene.OpFill, Rect: scene.Rect{X: 0, Y: 400, W: 600, H: 100}, Color: scene.Color{R: 160, A: 255}},
			scene.Op{Kind: scene.OpGlyphs, Start: 0, End: int32(len(s.Glyphs))},
			scene.Op{Kind: scene.OpPopClip})
		return MaskedCase{name, s}
	}
	ring := func(add func(scene.Op)) {
		add(scene.Op{Kind: scene.OpFill, Rect: scene.Rect{X: 30, Y: 20, W: 540, H: 460}, Radii: r4(20), Border: scene.Uniform(3), BorderColor: scene.Color{B: 200, A: 255}})
	}
	shadow := func(add func(scene.Op)) {
		add(scene.Op{Kind: scene.OpShadow, Rect: scene.Rect{X: 60, Y: 60, W: 400, H: 300}, Radii: r4(12), Color: scene.Color{A: 120}, Blur: 20,
			Cast: scene.Rect{X: 60, Y: 60, W: 400, H: 300}, CastRadii: r4(12)})
	}
	opaque := func(add func(scene.Op)) {
		add(scene.Op{Kind: scene.OpHole, Rect: scene.Rect{X: 200, Y: 200, W: 40, H: 40}})
		add(scene.Op{Kind: scene.OpFill, Rect: scene.Rect{X: 20, Y: 20, W: 560, H: 460}, Color: scene.Color{R: 250, G: 250, B: 250, A: 255}})
		add(scene.Op{Kind: scene.OpFill, Rect: scene.Rect{X: 50, Y: 50, W: 300, H: 300}, Radii: r4(16), Color: scene.Color{R: 30, G: 200, B: 90, A: 255}})
		add(scene.Op{Kind: scene.OpHole, Rect: scene.Rect{X: 100, Y: 100, W: 40, H: 40}})
		add(scene.Op{Kind: scene.OpFill, Rect: scene.Rect{X: 80, Y: 80, W: 200, H: 200}, Color: scene.Color{R: 10, B: 90, A: 255}})
	}
	all := func(add func(scene.Op)) { shadow(add); ring(add); opaque(add) }
	rounded := scene.Op{Rect: scene.Rect{X: 40, Y: 30, W: 520, H: 440}, Radii: r4(40)}
	continuous := scene.Op{Rect: scene.Rect{X: 40, Y: 30, W: 520, H: 440}, Radii: r4(40), Continuous: true}
	rotated := scene.Op{Rect: scene.Rect{X: 40, Y: 30, W: 520, H: 440}, Transform: scene.Rotation(3)}
	return []MaskedCase{
		build("masked ring", rounded, ring), build("masked shadow", rounded, shadow), build("masked opaque", rounded, opaque),
		build("masked continuous", continuous, all), build("masked rotated", rotated, all), build("masked everything", rounded, all),
	}
}

// WideScene returns a 600×500 scene with colors outside the sRGB gamut
// (Scene.Wide, Op.Wide, Glyph.Wide) in large opaque fills, a ring, a gradient
// glyph run inside a continuous rounded clip, and plain fills, which the
// float16 targets of Metal keep and the others draw as their nearest sRGB
// colors.
func WideScene() *scene.Scene {
	r4 := func(r float32) [4]float32 { return [4]float32{r, r, r, r} }
	s := &scene.Scene{Width: 600, Height: 500, Clear: scene.Color{R: 30, G: 34, B: 40, A: 255}}
	s.MaskAtlas, s.Text, s.Glyphs = glyphContent()
	s.Wide = []scene.WideColors{
		{Color: [4]float32{-0.25, 1.1, -0.1, 1}, Set: scene.WideColor},
		{Color: [4]float32{1.15, -0.2, -0.1, 1}, Color2: [4]float32{-0.2, 0.3, 1.2, 1}, Set: scene.WideColor | scene.WideColor2},
		{Border: [4]float32{1.2, 0.2, -0.1, 0.5}, Set: scene.WideBorder},
		{Color: [4]float32{0.2, 1.2, 0.3, 0.5}, Set: scene.WideColor},
	}
	green, red := scene.Color{G: 255, A: 255}, scene.Color{R: 255, A: 255}
	s.Glyphs = append([]scene.Glyph(nil), s.Glyphs...)
	for i := range s.Glyphs {
		s.Glyphs[i].Wide = 1
	}
	s.Ops = append(s.Ops,
		scene.Op{Kind: scene.OpFill, Rect: scene.Rect{W: 600, H: 500}, Color: green, Wide: 1},
		scene.Op{Kind: scene.OpFill, Rect: scene.Rect{X: 30, Y: 30, W: 540, H: 440}, Radii: r4(24), Continuous: true, Color: green, Wide: 1,
			Border: scene.Uniform(3), BorderColor: scene.Color{R: 255, A: 255}},
		scene.Op{Kind: scene.OpFill, Rect: scene.Rect{X: 20, Y: 20, W: 560, H: 460}, Radii: r4(16), Border: scene.Uniform(2), BorderColor: scene.Color{B: 255, A: 200}, Wide: 3},
		scene.Op{Kind: scene.OpFill, Rect: scene.Rect{X: 100, Y: 100, W: 300, H: 200}, Paint: scene.PaintOklab, Gradient: [4]float32{100, 0, 400, 0}, Color: green, Color2: red, Wide: 2},
		scene.Op{Kind: scene.OpFill, Rect: scene.Rect{X: 150, Y: 150, W: 300, H: 250}, Color: scene.Color{R: 60, G: 60, B: 255, A: 128}, Wide: 4},
		scene.Op{Kind: scene.OpPushClip, Rect: scene.Rect{X: 60, Y: 60, W: 400, H: 300}, Radii: r4(30), Continuous: true},
		scene.Op{Kind: scene.OpGlyphs, Start: 0, End: int32(len(s.Glyphs)), Paint: scene.PaintOklab, Gradient: [4]float32{40, 0, 240, 0}, Color: green, Color2: red, Wide: 2},
		scene.Op{Kind: scene.OpFill, Rect: scene.Rect{X: 70, Y: 200, W: 380, H: 100}, Color: red, Wide: 1},
		scene.Op{Kind: scene.OpPopClip})
	return s
}

// WithoutNewPaints returns s without what shaders older than gradients of
// stops draw differently: ops painting with stops (scene.Stop), borders
// painted with a gradient (Op.BorderPaint) and images blurred, for tests
// that compare a renderer with an earlier copy of its shader.
func WithoutNewPaints(s *scene.Scene) *scene.Scene {
	out := *s
	out.Ops = nil
	for _, op := range s.Ops {
		if op.Paint.HasStops() || op.BorderPaint || op.Kind == scene.OpImage && op.Blur > 0 {
			continue
		}
		out.Ops = append(out.Ops, op)
	}
	return &out
}
