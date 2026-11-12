// Package glass draws Liquid Glass in native UI, as macOS 26 and later
// draw it (AppKit's NSGlassEffectView), on every platform: what the
// elements under it painted shows through, blurred, bent near its edges as
// through the rim of a lens, and lit along its hairline rim.
// MyGo's renderers draw it with shaders of this package, on the GPU or the
// CPU, so it looks the same everywhere.
//
//	ui.Row(c).Padding(8, 16).Radius(22).Material(glass.Glass{})
//
// ScrollEdge is macOS's scroll edge effect, which content scrolling under
// a bar fades or frosts under, and Blur blurs what is under an element
// without the glass, evenly or fading along a gradient.
package glass

//go:generate go run ./internal/gen

import (
	"fmt"
	"math"
	"time"

	"github.com/egoist/mygo/ui"
)

// Glass is Liquid Glass, a ui.Material: Element.Material fills an element
// with it, in place of a background, shaped by its Radius. What was
// painted under the element shows through it, as content that scrolls
// under a bar placed over it with Absolute, and other glass.
type Glass struct {
	// Style is the glass's material.
	Style Style
	// Tint colors the glass, by its alpha, as for a prominent button: an
	// opaque tint lets a little of the glass show, as AppKit's does.
	Tint ui.Color
	// Interactive makes the glass grow a little while the element is
	// pressed, as AppKit's interactive glass does.
	Interactive bool
}

// Style is the material of Glass.
type Style uint8

const (
	// Regular frosts what shows through and lightens it, or darkens it in
	// dark mode, so that what is on the glass reads over anything: for
	// bars, buttons and panels. Larger panes are frostier.
	Regular Style = iota
	// Clear barely blurs nor tones what shows through, for glass over
	// photos and video, where what is on it brings its own contrast.
	Clear
)

// pressKey is the key of an interactive glass's animation.
type pressKey struct{}

// pressed is an interactive glass as its element is built: how far it
// has grown, from 0 to 1.
type pressed struct {
	Glass
	grow float32
}

// BuildMaterial follows the press of an interactive glass's element: the
// glass grows as it is pressed and shrinks back as it is let go, within
// 150 ms, as AppKit's does.
func (g Glass) BuildMaterial(e ui.Element) ui.Material {
	if !g.Interactive {
		return g
	}
	target := float32(0)
	if e.Pressed() {
		target = 1
	}
	return pressed{g, e.Animate(pressKey{}, target, 150*time.Millisecond)}
}

// PaintMaterial paints the glass over box, rounded by radii.
func (g Glass) PaintMaterial(p *ui.Painter, box ui.Rect, radii [4]float32) {
	paint(p, box, radii, g)
}

// PaintMaterial paints the glass grown by how far it is pressed: as
// macOS 27's, measured from NSGlassEffectView, by about 1.1 DIPs left and
// right and 0.45 DIPs above and below, whatever its size. AppKit stretches
// the content too, a pixel or so, which the children here are not.
func (g pressed) PaintMaterial(p *ui.Painter, box ui.Rect, radii [4]float32) {
	dx, dy := 1.1*g.grow, 0.45*g.grow
	box = ui.Rect{X: box.X - dx, Y: box.Y - dy, W: box.W + 2*dx, H: box.H + 2*dy}
	for i := range radii {
		if radii[i] > 0 {
			radii[i] += dy
		}
	}
	paint(p, box, radii, g.Glass)
}

// String describes the glass, as the inspector shows it.
func (g Glass) String() string {
	s := "glass"
	if g.Style == Clear {
		s += " clear"
	}
	if g.Tint.A > 0 {
		s += fmt.Sprintf(" tint(#%02x%02x%02x%02x)", g.Tint.R, g.Tint.G, g.Tint.B, g.Tint.A)
	}
	if g.Interactive {
		s += " interactive"
	}
	return s
}

// Paint paints a pane of glass shaped as r rounded by radius, in a
// drawing (Element.Draw), over what the painter painted before it.
func Paint(p *ui.Painter, r ui.Rect, radius float32, g Glass) {
	paint(p, r, [4]float32{radius, radius, radius, radius}, g)
}

// paint paints a pane of glass over box, rounded by radii.
//
// Its material follows macOS 27's, measured from NSGlassEffectView: the
// regular one maps black to 54% and white to 100% in light mode, and to
// 15% and 51% in dark mode, the clear one adds an eighth; larger panes
// blur more, up to 10 DIPs; the bezel curves the last 36 DIPs of the
// pane, at most half of it, and bends what shows through by up to 1.6
// times that, mirroring what is inside it, as AppKit's do; light comes
// from above and below, along a rim a pixel wide, and the pane casts no
// shadow, as NSGlassEffectView's and the glass bezel's do not.
func paint(p *ui.Painter, box ui.Rect, radii [4]float32, g Glass) {
	if box.W <= 0 || box.H <= 0 {
		return
	}
	s := p.Scale()
	m := min(box.W, box.H)
	bezel := min(36, m/2)
	mat := material{
		bezel:      bezel * s,
		refraction: 1.6 * bezel * s,
		rim:        0.35,
		rimWidth:   1,
		light:      math.Pi / 2,
	}
	switch {
	case g.Style == Clear:
		mat.blur = s
		mat.low, mat.high, mat.curve, mat.saturation = 0.125, 1.082, 1, 1
	case p.Theme().Dark:
		mat.blur = min(max(m*0.035, 1), 10) * s
		mat.low, mat.high, mat.curve, mat.saturation = 0.15, 0.51, 2, 2.2
	default:
		mat.blur = min(max(m*0.035, 1), 10) * s
		mat.low, mat.high, mat.curve, mat.saturation = 0.541, 1, 1.2, 1
	}
	if g.Tint.A > 0 {
		mat.tint, mat.wideTint = p.EffectColor(g.Tint)
		mat.tint[3] *= 0.88
		mat.wideTint[3] *= 0.88
	}
	p.Effect(Effect, box, radii, mat.blur, mat.params())
}
