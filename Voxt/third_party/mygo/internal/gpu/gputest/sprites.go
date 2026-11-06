package gputest

import "github.com/egoist/mygo/internal/scene"

// SpriteScene returns a w×h frame of a 2D game: layers of tiles of 16
// pixels drawn scale times as large, pixelated, from one sheet, covering
// the frame, and sprites of 16×32 over them, all ops of one image. With
// scale 3 at 1920×1080, three layers and 300 sprites, it draws 1,872
// sprites, about what a top-down farming game draws.
func SpriteScene(w, h int, scale float32, layers, sprites int) *scene.Scene {
	s := &scene.Scene{Width: w, Height: h, Clear: scene.Color{R: 40, G: 44, B: 52, A: 255}}
	// A sheet of 8×8 tiles of 16 pixels, each of its own colors.
	const tiles, tile = 8, 16
	pix := make([]byte, tiles*tile*tiles*tile*4)
	for y := range tiles * tile {
		for x := range tiles * tile {
			i := (y*tiles*tile + x) * 4
			tx, ty := x/tile, y/tile
			pix[i] = byte(tx*32 + x%tile*2)
			pix[i+1] = byte(ty*32 + y%tile*2)
			pix[i+2] = byte((tx + ty) * 16)
			pix[i+3] = 255
		}
	}
	sheet := scene.NewImageRGBA(tiles*tile, tiles*tile, pix)
	size := tile * scale
	cols, rows := int(float32(w)/size)+1, int(float32(h)/size)+1
	for l := range layers {
		for r := range rows {
			for c := range cols {
				k := (r*cols + c + l*7) % (tiles * tiles)
				src := scene.Rect{X: float32(k%tiles) * tile, Y: float32(k/tiles) * tile, W: tile, H: tile}
				if l > 0 && (r+c+l)%3 != 0 {
					continue // upper layers cover a third
				}
				s.Ops = append(s.Ops, scene.Op{Kind: scene.OpImage, Rect: scene.Rect{X: float32(c) * size, Y: float32(r) * size, W: size, H: size},
					Image: sheet, Src: src, Pixelated: true})
			}
		}
	}
	for i := range sprites {
		x := float32(i*97%max(w-int(size), 1)) + 0.5
		y := float32(i*61%max(h-int(2*size), 1)) + 0.25
		src := scene.Rect{X: float32(i%tiles) * tile, Y: 0, W: tile, H: 2 * tile}
		if i%2 == 1 {
			src.X, src.W = src.X+src.W, -src.W // facing left
		}
		s.Ops = append(s.Ops, scene.Op{Kind: scene.OpImage, Rect: scene.Rect{X: x, Y: y, W: size, H: 2 * size}, Image: sheet, Src: src, Pixelated: true})
	}
	return s
}
