// Command gen makes the menu bar icons of package trayicon from Voxt's
// microphone (voxt.png): menubar.png and menubar-recording.png, 36×36
// template images. With -preview DIR it also writes how they look on a
// light and a dark menu bar, at 1× and 2×, magnified.
package main

import (
	"bytes"
	"flag"
	"fmt"
	"image"
	"image/color"
	"image/png"
	"log"
	"math"
	"os"
	"path/filepath"

	"golang.org/x/image/draw"
)

// size is the icons' side in pixels: 18 pt at 2×.
const size = 36

// The recording dot, in icon pixels (of 36): its centre and radius, and
// the gap cut around it. The top right corner is the one the microphone
// leaves empty (its stand's arms take the bottom ones).
const (
	dotX, dotY = 30.5, 5.5
	dotR       = 4.5
	dotGap     = 1.75
)

func main() {
	dir := flag.String("dir", ".", "the trayicon package directory")
	preview := flag.String("preview", "", "also write previews into this directory")
	flag.Parse()
	src, err := load(filepath.Join(*dir, "voxt.png"))
	if err != nil {
		log.Fatal(err)
	}
	icons := map[string]*image.NRGBA{
		"menubar.png":           render(src, false),
		"menubar-recording.png": render(src, true),
	}
	for name, img := range icons {
		if err := os.WriteFile(filepath.Join(*dir, name), encode(img), 0o644); err != nil {
			log.Fatal(err)
		}
	}
	if *preview != "" {
		if err := os.MkdirAll(*preview, 0o755); err != nil {
			log.Fatal(err)
		}
		for _, scale := range []int{1, 2} {
			img := previewSheet(icons["menubar.png"], icons["menubar-recording.png"], scale)
			name := filepath.Join(*preview, fmt.Sprintf("menubar-%dx.png", scale))
			if err := os.WriteFile(name, encode(img), 0o644); err != nil {
				log.Fatal(err)
			}
		}
	}
}

func load(path string) (image.Image, error) {
	b, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	return png.Decode(bytes.NewReader(b))
}

func encode(img image.Image) []byte {
	var b bytes.Buffer
	e := png.Encoder{CompressionLevel: png.BestCompression}
	if err := e.Encode(&b, img); err != nil {
		panic(err)
	}
	return b.Bytes()
}

// render scales Voxt's microphone to 36×36 (Catmull-Rom; the whole
// canvas, so its margins stay Voxt's), adds the recording dot if asked
// (drawn at the source's resolution, so it is scaled like the rest), and
// keeps only the alpha: a template image is black and transparent.
func render(src image.Image, recording bool) *image.NRGBA {
	b := src.Bounds()
	mask := image.NewAlpha(b)
	for y := b.Min.Y; y < b.Max.Y; y++ {
		for x := b.Min.X; x < b.Max.X; x++ {
			_, _, _, a := src.At(x, y).RGBA()
			mask.SetAlpha(x, y, color.Alpha{uint8(a >> 8)})
		}
	}
	if recording {
		k := float64(b.Dx()) / size // source pixels per icon pixel
		cx, cy := float64(b.Min.X)+dotX*k, float64(b.Min.Y)+dotY*k
		r, gap := dotR*k, (dotR+dotGap)*k
		for y := b.Min.Y; y < b.Max.Y; y++ {
			for x := b.Min.X; x < b.Max.X; x++ {
				d := math.Hypot(float64(x)+0.5-cx, float64(y)+0.5-cy)
				a := float64(mask.AlphaAt(x, y).A) / 255
				a *= clamp01(d - gap + 0.5)  // the gap clears the microphone
				a = max(a, clamp01(r-d+0.5)) // the dot
				mask.SetAlpha(x, y, color.Alpha{uint8(math.Round(a * 255))})
			}
		}
	}
	small := image.NewAlpha(image.Rect(0, 0, size, size))
	draw.CatmullRom.Scale(small, small.Bounds(), mask, b, draw.Src, nil)
	out := image.NewNRGBA(small.Bounds())
	for y := 0; y < size; y++ {
		for x := 0; x < size; x++ {
			out.SetNRGBA(x, y, color.NRGBA{0, 0, 0, small.AlphaAt(x, y).A})
		}
	}
	return out
}

func clamp01(v float64) float64 { return min(max(v, 0), 1) }

// The menu bars of the previews: their background and the colour macOS
// tints a template icon with (approximate).
var bars = []struct{ bg, ink color.NRGBA }{
	{color.NRGBA{0xEC, 0xEC, 0xEC, 0xFF}, color.NRGBA{0x00, 0x00, 0x00, 0xD9}}, // light
	{color.NRGBA{0x24, 0x24, 0x26, 0xFF}, color.NRGBA{0xFF, 0xFF, 0xFF, 0xF2}}, // dark
}

// previewSheet shows the icons as a menu bar of the given scale shows
// them (24 pt tall, the icon 18 pt): light on top, dark below; at rest on
// the left, recording on the right; magnified 8× (1×) or 4× (2×) with
// square pixels so each device pixel can be seen.
func previewSheet(idle, rec *image.NRGBA, scale int) *image.NRGBA {
	const barPt, iconPt, padPt = 24, 18, 10
	cell := (padPt*2 + iconPt) * scale
	device := image.NewNRGBA(image.Rect(0, 0, 2*cell, len(bars)*barPt*scale))
	for row, bar := range bars {
		y0 := row * barPt * scale
		draw.Draw(device, image.Rect(0, y0, 2*cell, y0+barPt*scale), image.NewUniform(bar.bg), image.Point{}, draw.Src)
		for col, icon := range []*image.NRGBA{idle, rec} {
			// AppKit draws the 36 px image into 18 pt: as is at 2×,
			// scaled down at 1×.
			px := iconPt * scale
			alpha := image.NewAlpha(image.Rect(0, 0, px, px))
			draw.CatmullRom.Scale(alpha, alpha.Bounds(), icon, icon.Bounds(), draw.Src, nil)
			at := image.Pt(col*cell+padPt*scale, y0+(barPt-iconPt)/2*scale)
			draw.DrawMask(device, alpha.Bounds().Add(at), image.NewUniform(bar.ink), image.Point{}, alpha, image.Point{}, draw.Over)
		}
	}
	zoom := 8 / scale
	big := image.NewNRGBA(image.Rect(0, 0, device.Bounds().Dx()*zoom, device.Bounds().Dy()*zoom))
	draw.NearestNeighbor.Scale(big, big.Bounds(), device, device.Bounds(), draw.Src, nil)
	return big
}
