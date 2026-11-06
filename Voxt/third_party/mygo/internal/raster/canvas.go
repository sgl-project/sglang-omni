package raster

import (
	"image"

	"github.com/egoist/mygo/internal/scene"
)

// drawCanvases draws into their pixels the canvases s draws anew or shows,
// those changed since, each before the canvases and the scene that show
// it.
func drawCanvases(s *scene.Scene) {
	s.EachCanvas(func(c *scene.Image) {
		if c.PixVersion == c.Version() && len(c.Pix) == 4*c.W*c.H {
			return
		}
		cs := c.Scene
		cs.Width, cs.Height = c.W, c.H
		img := NewImage(c.W, c.H)
		var r Renderer
		r.d.draw(img, cs, image.Rect(0, 0, c.W, c.H), r.opBounds(cs, nil))
		if len(c.Pix) != 4*c.W*c.H {
			c.Pix = make([]byte, 4*c.W*c.H)
		}
		// From BGRA to the RGBA of images.
		for i := 0; i+3 < len(img.Pix); i += 4 {
			c.Pix[i], c.Pix[i+1], c.Pix[i+2], c.Pix[i+3] = img.Pix[i+2], img.Pix[i+1], img.Pix[i], img.Pix[i+3]
		}
		c.PixVersion = c.Version()
	})
}
