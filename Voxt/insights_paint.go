package main

// The insights cards' pictures (insights_themes.go), painted in code as
// the share card's backgrounds are (share_paint.go), at 2×, once each.
//
// A paper theme's background is a sheet of paper in the theme's color:
// the fibres of a real sheet (assets/share/paper-grain.jpg, each pixel
// how much lighter or darker than the sheet it is) laid over the color,
// lit a little from the top left.
//
// The marks behind the cards are rings as of a sound from a corner, a
// waveform's bars, hills and a sun; on paper they are layers cut from
// the same sheet, each a little lighter than the one behind and casting
// a soft shadow on it. They are too large for a path, whose mask would
// not fit beside the text's glyphs.

import (
	"bytes"
	_ "embed"
	"image"
	"image/color"
	_ "image/jpeg" // the paper's grain
	"math"
	"sync"

	"github.com/egoist/mygo/ui"
)

//go:embed assets/share/paper-grain.jpg
var insightsPaperGrainJPEG []byte

// insightsPaperGrain is the sheet's fibres, 1200×800: 128 where the
// paper is as light as the sheet, more where it is lighter.
var insightsPaperGrain = sync.OnceValue(func() *image.Gray {
	img, _, err := image.Decode(bytes.NewReader(insightsPaperGrainJPEG))
	if err != nil {
		return nil
	}
	if gray, ok := img.(*image.Gray); ok {
		return gray
	}
	bounds := img.Bounds()
	gray := image.NewGray(bounds)
	for y := bounds.Min.Y; y < bounds.Max.Y; y++ {
		for x := bounds.Min.X; x < bounds.Max.X; x++ {
			gray.Set(x, y, img.At(x, y))
		}
	}
	return gray
})

// insightsPaperFibre is how strongly the fibres show, of how they do on
// the sheet they were taken from.
const insightsPaperFibre = .75

var insightsPaperPictures = map[string]*ui.Bitmap{}

// insightsPaperBitmap is a sheet of paper in color hex, the card's size
// at 2×.
func insightsPaperBitmap(hex string) *ui.Bitmap {
	sharePaintMu.Lock()
	defer sharePaintMu.Unlock()
	if bm, ok := insightsPaperPictures[hex]; ok {
		return bm
	}
	base := shareRGB(hex)
	grain := insightsPaperGrain()
	w, h := insightsCardW*insightsMarkScale, insightsCardH*insightsMarkScale
	img := image.NewRGBA(image.Rect(0, 0, w, h))
	for y := range h {
		for x := range w {
			fibre := 1.0
			if grain != nil {
				b := grain.Bounds()
				fibre = 1 + insightsPaperFibre*(float64(grain.GrayAt(b.Min.X+x%b.Dx(), b.Min.Y+y%b.Dy()).Y)/128-1)
			}
			// Lit from the top left: a little lighter there, darker at
			// the far corner.
			lit := 1.05 - .12*(float64(x)/float64(w)+float64(y)/float64(h))/2
			var px [3]uint8
			for i, channel := range base {
				// The fibres show a little less on a dark sheet's weak
				// channels than a plain multiply would leave them.
				v := channel*lit*fibre + 14*(fibre-1)
				px[i] = uint8(math.Min(255, math.Max(0, v)) + .5)
			}
			img.SetRGBA(x, y, color.RGBA{px[0], px[1], px[2], 255})
		}
	}
	bm := ui.NewBitmap(img)
	insightsPaperPictures[hex] = bm
	return bm
}

const insightsMarkScale = 2

var insightsMarkPictures = map[insightsMarkKey]*ui.Bitmap{}

// insightsMarkKey is a painted mark: its card, and whether it is cut
// from paper.
type insightsMarkKey struct {
	kind string
	cut  bool
}

// insightsWaveBars are the waveform's bars: their heights, of the tallest.
var insightsWaveBars = [...]float64{.22, .4, .3, .62, .46, .86, .58, 1, .7, .9, .5, .74, .38, .56, .28, .42, .2}

// insightsLayer is a shape of a mark: how far a point (in the card's
// DIPs) is outside it, negative inside, and how much lighter than the
// background it is (0–1).
type insightsLayer struct {
	outside func(x, y float64) float64
	light   float64
}

// insightsMarkLayers are card kind's shapes, the farthest first.
func insightsMarkLayers(kind string) (layers []insightsLayer) {
	rings := func(cx, cy float64) {
		for i := range 4 {
			r := float64(110 + i*78)
			layers = append(layers, insightsLayer{func(x, y float64) float64 { return math.Abs(math.Hypot(x-cx, y-cy)-r) - 17 }, .10})
		}
	}
	switch kind {
	case insightsCardCharacters:
		// Hills that rise to the right, the nearest lowest.
		for k := range 4 {
			f := float64(k)
			layers = append(layers, insightsLayer{func(x, y float64) float64 {
				t := x / insightsCardW
				edge := 322 + 25*f - (158-24*f)*t*t + 9*math.Sin(x/72+1.3*f)
				slope := -(158-24*f)*2*t/insightsCardW + 9/72.0*math.Cos(x/72+1.3*f)
				return (edge - y) / math.Hypot(1, slope)
			}, .05 + .045*f})
		}
	case insightsCardStreak:
		// A sun going up behind the week: discs, the nearest smallest.
		for k, r := range []float64{236, 176, 118, 62} {
			layers = append(layers, insightsLayer{func(x, y float64) float64 { return math.Hypot(x-500, y-268) - r }, .05 + .05*float64(k)})
		}
	case insightsCardSpeed:
		rings(insightsCardW, 0)
	case insightsCardSaved:
		const barW, gap, tall, mid = 18.0, 14.0, 250.0, 96.0
		for i, level := range insightsWaveBars {
			cx, half := 69+float64(i)*(barW+gap), tall*level/2
			layers = append(layers, insightsLayer{func(x, y float64) float64 {
				// A capsule: the distance to its centre line, less its radius.
				return math.Hypot(x-cx, math.Max(math.Abs(y-mid)-(half-barW/2), 0)) - barW/2
			}, .10})
		}
	case insightsCardDays:
		rings(0, insightsCardH)
	default:
		rings(insightsCardW, insightsCardH)
	}
	return layers
}

// insightsMarkBitmap is the mark behind card kind: white where it
// lightens the background and, when it is cut from paper, black where a
// layer's shadow falls on the layers behind it.
func insightsMarkBitmap(kind string, cut bool) *ui.Bitmap {
	sharePaintMu.Lock()
	defer sharePaintMu.Unlock()
	key := insightsMarkKey{kind, cut}
	if bm, ok := insightsMarkPictures[key]; ok {
		return bm
	}
	layers := insightsMarkLayers(kind)
	// A cut layer's shadow: down and to the right, fading over a few DIPs.
	const shadowX, shadowY, shadowSoft, shadowDark = 3.0, 5.0, 7.0, .22
	w, h := insightsCardW*insightsMarkScale, insightsCardH*insightsMarkScale
	img := image.NewNRGBA(image.Rect(0, 0, w, h))
	for py := range h {
		for px := range w {
			x, y := (float64(px)+.5)/insightsMarkScale, (float64(py)+.5)/insightsMarkScale
			light, dark := 0.0, 0.0
			for _, layer := range layers {
				d := layer.outside(x, y)
				cover := math.Min(1, math.Max(0, .5-d*insightsMarkScale))
				if !cut {
					light = math.Max(light, layer.light*cover)
					continue
				}
				if far := layer.outside(x-shadowX, y-shadowY); far < 4*shadowSoft {
					dark += (1 - dark) * shadowDark * math.Exp(-math.Max(far, 0)/shadowSoft)
				}
				// The layer hides what is behind it, shadows too; its edge
				// catches a little light.
				rim := 0.0
				if d < 0 {
					rim = .05 * math.Exp(d/1.5)
				}
				light = light*(1-cover) + (layer.light+rim)*cover
				dark *= 1 - cover
			}
			// White at light over black at dark, as one color.
			alpha := light + dark*(1-light)
			if alpha <= 0 {
				continue
			}
			grey := uint8(255*light/alpha + .5)
			img.SetNRGBA(px, py, color.NRGBA{grey, grey, grey, uint8(alpha*255 + .5)})
		}
	}
	bm := ui.NewBitmap(img)
	insightsMarkPictures[key] = bm
	return bm
}
