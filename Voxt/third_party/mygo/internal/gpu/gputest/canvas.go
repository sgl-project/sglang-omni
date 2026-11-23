package gputest

import "github.com/egoist/mygo/internal/scene"

// CanvasScene returns a 120×80 scene showing canvases (scene.NewCanvas): a
// 16×12 one drawn with fills, a rounded one, an image and a canvas of its
// own, shown 4 times as large pixelated, then smooth and flipped; and a
// canvas shown only within another, with nothing new to draw the second
// time the renderer draws the scene; and a canvas drawing a conic gradient,
// shown blurred, as a view blurred is (ui.Element.Blur).
func CanvasScene() *scene.Scene {
	s := &scene.Scene{Width: 120, Height: 80, Clear: scene.Color{R: 246, G: 247, B: 249, A: 255}}
	pix := make([]byte, 4*4*4)
	for i := range 16 {
		pix[4*i], pix[4*i+1], pix[4*i+2], pix[4*i+3] = byte(i*16), byte(255-i*16), 120, 255
	}
	img := scene.NewImageRGBA(4, 4, pix)

	inner := scene.NewCanvas(4, 4)
	inner.Scene.Ops = []scene.Op{{Kind: scene.OpFill, Rect: scene.Rect{W: 4, H: 4}, Color: scene.Color{R: 250, G: 204, B: 21, A: 255}},
		{Kind: scene.OpFill, Rect: scene.Rect{X: 1, Y: 1, W: 2, H: 2}, Color: scene.Color{R: 20, G: 24, B: 32, A: 255}}}

	c := scene.NewCanvas(16, 12)
	c.Scene.Ops = []scene.Op{
		{Kind: scene.OpFill, Rect: scene.Rect{W: 16, H: 12}, Color: scene.Color{R: 37, G: 99, B: 235, A: 255}},
		{Kind: scene.OpFill, Rect: scene.Rect{X: 1, Y: 1, W: 6, H: 6}, Radii: [4]float32{3, 3, 3, 3}, Color: scene.Color{R: 220, G: 40, B: 40, A: 200}},
		{Kind: scene.OpImage, Rect: scene.Rect{X: 9, Y: 1, W: 4, H: 4}, Image: img, Src: scene.Rect{W: 4, H: 4}},
		{Kind: scene.OpImage, Rect: scene.Rect{X: 2, Y: 7, W: 4, H: 4}, Image: inner, Src: scene.Rect{W: 4, H: 4}},
		{Kind: scene.OpFill, Rect: scene.Rect{X: 8, Y: 8, W: 8, H: 4}, Color: scene.Color{R: 255, G: 255, B: 255, A: 128}},
	}
	soft := scene.NewCanvas(40, 34)
	soft.Scene.Stops = []scene.Stop{{Color: scene.Color{R: 220, G: 40, B: 40, A: 255}, At: 0}, {Color: scene.Color{R: 37, G: 99, B: 235, A: 255}, At: 0.6},
		{Color: scene.Color{R: 220, G: 40, B: 40, A: 255}, At: 1}}
	soft.Scene.Ops = []scene.Op{
		{Kind: scene.OpFill, Rect: scene.Rect{X: 6, Y: 6, W: 28, H: 22}, Radii: [4]float32{6, 6, 6, 6}, Paint: scene.PaintConic,
			Gradient: [4]float32{20, 17, 0.3, 0}, Stops: [2]int32{0, 3}},
		{Kind: scene.OpFill, Rect: scene.Rect{X: 14, Y: 2, W: 4, H: 30}, Color: scene.Color{R: 20, G: 24, B: 32, A: 160}},
	}
	s.Canvases = []*scene.Image{c, soft}
	s.Ops = []scene.Op{
		{Kind: scene.OpImage, Rect: scene.Rect{X: 4, Y: 4, W: 64, H: 48}, Image: c, Src: scene.Rect{W: 16, H: 12}, Pixelated: true},
		{Kind: scene.OpImage, Rect: scene.Rect{X: 72, Y: 4, W: 44, H: 33}, Image: c, Src: scene.Rect{X: 16, W: -16, H: 12}},
		{Kind: scene.OpImage, Rect: scene.Rect{X: 74, Y: 42, W: 40, H: 34}, Image: soft, Src: scene.Rect{W: 40, H: 34}, Blur: 2.5},
		{Kind: scene.OpImage, Rect: scene.Rect{X: 4, Y: 56, W: 40, H: 20}, Image: soft, Src: scene.Rect{X: 0, Y: 34, W: 40, H: -34}, Blur: 5},
	}
	return s
}
