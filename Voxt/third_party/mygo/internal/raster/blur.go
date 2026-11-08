package raster

import (
	"image"
	"runtime"
	"sync"

	"github.com/egoist/mygo/internal/scene"
)

// blurredImage is an image blurred (Op.Blur), as an op draws it: its
// texels, and where the op's Src lies in them.
type blurredImage struct {
	img *scene.Image
	src scene.Rect
}

// blurImages returns, for each OpImage of s that blurs its image and
// whose bounds meet area, the image blurred, as the GPU renderers compute
// it; nil when none does. Those missing the area are not drawn.
func (d *drawer) blurImages(s *scene.Scene, area image.Rectangle, bounds []image.Rectangle) []*blurredImage {
	var out []*blurredImage
	for i := range s.Ops {
		op := &s.Ops[i]
		if op.Kind != scene.OpImage || op.Blur <= 0 || op.Image == nil || op.Image.W == 0 || op.Image.H == 0 || !bounds[i].Overlaps(area) {
			continue
		}
		if out == nil {
			out = make([]*blurredImage, len(s.Ops))
		}
		out[i] = blurImage(op.Image, op.Src, op.Blur)
	}
	return out
}

// blurImage returns the pixels of img within src blurred by blur, the
// standard deviation of a Gaussian in its pixels, transparent beyond src:
// averaged over squares, then blurred along rows and along columns, each
// step kept in 8 bits a channel, as a texture holds it, and as the GPU
// renderers' passes "down" and "blur" compute it.
func blurImage(img *scene.Image, src scene.Rect, blur float32) *blurredImage {
	b := scene.ImageBlur(scene.ImageArea(src, img.W, img.H), blur)
	w, h := b.Size()
	out := &blurredImage{img: &scene.Image{W: w, H: h}, src: b.ImageSrc(src)}
	if w <= 0 || h <= 0 || len(img.Pix) < 4*img.W*img.H {
		out.img.W, out.img.H = 0, 0
		return out
	}
	tex := make([]float32, 4*w*h)
	k := b.Down
	inv := 1 / float32(k*k)
	for j := range h {
		for i := range w {
			var c [4]float32
			for dy := range k {
				y := b.Area.Min.Y + j*k + dy
				if y >= b.Area.Max.Y {
					continue // transparent beyond the area
				}
				for dx := range k {
					x := b.Area.Min.X + i*k + dx
					if x >= b.Area.Max.X {
						continue
					}
					p := img.Pix[4*(y*img.W+x):][:4]
					for ch := range c {
						c[ch] += float32(p[ch]) / 255
					}
				}
			}
			o := tex[4*(j*w+i):][:4]
			for ch := range o {
				o[ch] = texel(c[ch] * inv)
			}
		}
	}
	if b.Radius > 0 {
		weights := make([]float32, b.Radius+1)
		sum := float32(0)
		for o := range weights {
			weights[o] = scene.BlurWeight(o, b.Sigma)
			if o == 0 {
				sum += weights[o]
			} else {
				sum += 2 * weights[o]
			}
		}
		tmp := make([]float32, len(tex))
		blurLines(tmp, tex, weights, sum, 4, 4*w, w, h)
		blurLines(tex, tmp, weights, sum, 4*w, 4, h, w)
	}
	out.img.Pix = make([]byte, len(tex))
	for i, v := range tex {
		out.img.Pix[i] = to8(v)
	}
	return out
}

// blurLines blurs lines of n texels step apart in src into dst, lines
// next apart, those beyond a line's ends transparent: every texel takes
// every weight, divided by their sum. Lines go to several cores when they
// are worth it.
func blurLines(dst, src, weights []float32, sum float32, step, next, n, lines int) {
	rad := len(weights) - 1
	workers := max(min(runtime.GOMAXPROCS(0), lines*n*(2*rad+1)/texelWork, maxWorkers), 1)
	if workers == 1 {
		for line := range lines {
			blurLine(dst, src, weights, sum, step, line*next, n)
		}
		return
	}
	var wg sync.WaitGroup
	per := (lines + workers - 1) / workers
	for w := range workers {
		lo, hi := w*per, min((w+1)*per, lines)
		wg.Add(1)
		go func() {
			defer wg.Done()
			for line := lo; line < hi; line++ {
				blurLine(dst, src, weights, sum, step, line*next, n)
			}
		}()
	}
	wg.Wait()
}

// blurLine blurs the line of n texels step apart from base.
func blurLine(dst, src, weights []float32, sum float32, step, base, n int) {
	rad := len(weights) - 1
	for i := range n {
		p := src[base+i*step:][:4]
		w0 := weights[0]
		c0, c1, c2, c3 := p[0]*w0, p[1]*w0, p[2]*w0, p[3]*w0
		if i >= rad && i+rad < n {
			// Far from both ends: every tap is a texel.
			for o := 1; o <= rad; o++ {
				a, b := src[base+(i-o)*step:][:4], src[base+(i+o)*step:][:4]
				wo := weights[o]
				c0 += (a[0] + b[0]) * wo
				c1 += (a[1] + b[1]) * wo
				c2 += (a[2] + b[2]) * wo
				c3 += (a[3] + b[3]) * wo
			}
		} else {
			for o := 1; o <= rad; o++ {
				wo := weights[o]
				if j := i - o; j >= 0 {
					a := src[base+j*step:][:4]
					c0, c1, c2, c3 = c0+a[0]*wo, c1+a[1]*wo, c2+a[2]*wo, c3+a[3]*wo
				}
				if j := i + o; j < n {
					b := src[base+j*step:][:4]
					c0, c1, c2, c3 = c0+b[0]*wo, c1+b[1]*wo, c2+b[2]*wo, c3+b[3]*wo
				}
			}
		}
		o := dst[base+i*step:][:4]
		o[0], o[1], o[2], o[3] = texel(c0/sum), texel(c1/sum), texel(c2/sum), texel(c3/sum)
	}
}
