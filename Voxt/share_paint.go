package main

// The share cards' painted pictures, made in code the way the mockup's
// CSS draws them: a background (a 120° gradient through three colors,
// under a glow of the first color at the top left and of the last at the
// bottom right) and the hat studio's palette strip, whose cells run into
// each other. Each is drawn once per size and kept.

import (
	"image"
	"image/color"
	"math"
	"sync"

	"github.com/egoist/mygo/ui"
)

var (
	sharePaintMu    sync.Mutex
	shareBgPictures = map[[3]int]*ui.Bitmap{}
	sharePalettePic *ui.Bitmap
)

// shareBackgroundBitmap is background index painted w×h pixels.
func shareBackgroundBitmap(index, w, h int) *ui.Bitmap {
	key := [3]int{index, w, h}
	sharePaintMu.Lock()
	defer sharePaintMu.Unlock()
	if bm, ok := shareBgPictures[key]; ok {
		return bm
	}
	bg := shareBackgrounds[index]
	a, b, c := shareRGB(bg.A), shareRGB(bg.B), shareRGB(bg.C)
	img := image.NewRGBA(image.Rect(0, 0, w, h))
	// CSS's gradient line for 120°: through the middle, long enough that
	// the corners take the end colors.
	ang := 120 * math.Pi / 180
	dx, dy := math.Sin(ang), -math.Cos(ang)
	length := math.Abs(float64(w)*dx) + math.Abs(float64(h)*dy)
	glow := func(px, py float64, col [3]float64, stop float64, x, y float64) ([3]float64, float64) {
		cx, cy := px*float64(w), py*float64(h)
		r := math.Max(math.Max(math.Hypot(cx, cy), math.Hypot(float64(w)-cx, cy)), math.Max(math.Hypot(cx, float64(h)-cy), math.Hypot(float64(w)-cx, float64(h)-cy)))
		t := math.Hypot(x-cx, y-cy) / r
		return col, math.Max(0, 1-t/stop)
	}
	for y := range h {
		for x := range w {
			fx, fy := float64(x)+.5, float64(y)+.5
			t := ((fx-float64(w)/2)*dx+(fy-float64(h)/2)*dy)/length + .5
			var col [3]float64
			if t < .55 {
				col = shareLerp(a, b, math.Max(0, t)/.55)
			} else {
				col = shareLerp(b, c, math.Min(1, (t-.55)/.45))
			}
			// The bottom right glow, then the top left one over it.
			for _, g := range [][4]float64{{.92, .95, 2, .55}, {.15, .25, 0, .6}} {
				src := a
				if g[2] == 2 {
					src = c
				}
				gc, alpha := glow(g[0], g[1], src, g[3], fx, fy)
				col = shareLerp(col, gc, alpha)
			}
			img.SetRGBA(x, y, color.RGBA{uint8(col[0] + .5), uint8(col[1] + .5), uint8(col[2] + .5), 255})
		}
	}
	bm := ui.NewBitmap(img)
	shareBgPictures[key] = bm
	return bm
}

// sharePaletteBitmap is the palette strip at 2×: its cells, then a blur
// of 2.2 DIPs, the edge cells carried past the strip so its rim stays
// solid.
func sharePaletteBitmap() *ui.Bitmap {
	sharePaintMu.Lock()
	defer sharePaintMu.Unlock()
	if sharePalettePic != nil {
		return sharePalettePic
	}
	const scale, pad = 2, 12
	cw, ch := studioCellW*scale, studioCellH*scale
	w, h := 10*cw, 3*ch
	W, H := w+2*pad, h+2*pad
	src := make([][3]float64, W*H)
	for y := range H {
		row := min(max((y-pad)/ch, 0), 2)
		for x := range W {
			col := min(max((x-pad)/cw, 0), 9)
			src[y*W+x] = shareRGB(shareStudioPalette[row][col])
		}
	}
	blurred := shareBlur(src, W, H, 2.2*scale)
	img := image.NewRGBA(image.Rect(0, 0, w, h))
	for y := range h {
		for x := range w {
			p := blurred[(y+pad)*W+x+pad]
			img.SetRGBA(x, y, color.RGBA{uint8(p[0] + .5), uint8(p[1] + .5), uint8(p[2] + .5), 255})
		}
	}
	sharePalettePic = ui.NewBitmap(img)
	return sharePalettePic
}

// shareBlur is a separable Gaussian blur of sigma pixels, the edges
// clamped.
func shareBlur(src [][3]float64, w, h int, sigma float64) [][3]float64 {
	radius := int(math.Ceil(sigma * 3))
	kernel := make([]float64, 2*radius+1)
	var sum float64
	for i := range kernel {
		d := float64(i - radius)
		kernel[i] = math.Exp(-d * d / (2 * sigma * sigma))
		sum += kernel[i]
	}
	for i := range kernel {
		kernel[i] /= sum
	}
	pass := func(in [][3]float64, horizontal bool) [][3]float64 {
		out := make([][3]float64, len(in))
		for y := range h {
			for x := range w {
				var acc [3]float64
				for k, kv := range kernel {
					sx, sy := x, y
					if horizontal {
						sx = min(max(x+k-radius, 0), w-1)
					} else {
						sy = min(max(y+k-radius, 0), h-1)
					}
					p := in[sy*w+sx]
					acc[0] += p[0] * kv
					acc[1] += p[1] * kv
					acc[2] += p[2] * kv
				}
				out[y*w+x] = acc
			}
		}
		return out
	}
	return pass(pass(src, true), false)
}

func shareRGB(hex string) [3]float64 {
	c := ui.Hex(hex)
	return [3]float64{float64(c.R), float64(c.G), float64(c.B)}
}

func shareLerp(p, q [3]float64, t float64) [3]float64 {
	return [3]float64{p[0] + (q[0]-p[0])*t, p[1] + (q[1]-p[1])*t, p[2] + (q[2]-p[2])*t}
}

// renderTransparent draws view w×h DIPs at scale with a transparent
// background. An offscreen render paints the window's background first,
// so this renders the view over the light window and over the dark one,
// and recovers each pixel's alpha and color from the two: a pixel the
// view leaves alone differs between them as the backgrounds do, one it
// covers not at all. The view must not change with the appearance.
func renderTransparent(view func(c *ui.Context), w, h int, scale float32) image.Image {
	frame := func(dark bool, v func(c *ui.Context)) *image.RGBA {
		tt := ui.NewTester(v, w, h)
		tt.SetScale(scale)
		tt.SetDark(dark)
		tt.Frame()
		return cloneRGBA(tt.Image())
	}
	light, dark := frame(false, view), frame(true, view)
	bgLight, bgDark := frame(false, func(*ui.Context) {}).RGBAAt(0, 0), frame(true, func(*ui.Context) {}).RGBAAt(0, 0)
	spread := float64(int(bgLight.R)+int(bgLight.G)+int(bgLight.B)) - float64(int(bgDark.R)+int(bgDark.G)+int(bgDark.B))
	out := image.NewNRGBA(light.Bounds())
	for y := light.Bounds().Min.Y; y < light.Bounds().Max.Y; y++ {
		for x := light.Bounds().Min.X; x < light.Bounds().Max.X; x++ {
			l, d := light.RGBAAt(x, y), dark.RGBAAt(x, y)
			diff := float64(int(l.R)+int(l.G)+int(l.B)) - float64(int(d.R)+int(d.G)+int(d.B))
			alpha := 1.0
			if spread != 0 {
				alpha = math.Min(1, math.Max(0, 1-diff/spread))
			}
			if alpha < 1.0/255 {
				continue
			}
			un := func(o, bg uint8) uint8 {
				v := (float64(o) - (1-alpha)*float64(bg)) / alpha
				return uint8(math.Min(255, math.Max(0, v)+.5))
			}
			out.SetNRGBA(x, y, color.NRGBA{un(d.R, bgDark.R), un(d.G, bgDark.G), un(d.B, bgDark.B), uint8(alpha*255 + .5)})
		}
	}
	return out
}

func cloneRGBA(img *image.RGBA) *image.RGBA {
	out := image.NewRGBA(img.Bounds())
	copy(out.Pix, img.Pix)
	return out
}
