package ui

import (
	"fmt"
	"math"
	"time"
)

// hsva is a color as its hue, from 0 to 360, saturation, value and alpha,
// from 0 to 1, which a color picker keeps: a color of red, green and blue
// loses the hue of its grays.
type hsva struct{ h, s, v, a float64 }

// toHSVA returns the hue, saturation, value and alpha of c.
func toHSVA(c Color) hsva {
	r, g, b := float64(c.R)/255, float64(c.G)/255, float64(c.B)/255
	hi, lo := math.Max(r, math.Max(g, b)), math.Min(r, math.Min(g, b))
	d := hi - lo
	var h float64
	switch {
	case d == 0:
	case hi == r:
		h = 60 * math.Mod((g-b)/d, 6)
	case hi == g:
		h = 60 * ((b-r)/d + 2)
	default:
		h = 60 * ((r-g)/d + 4)
	}
	if h < 0 {
		h += 360
	}
	s := 0.0
	if hi > 0 {
		s = d / hi
	}
	return hsva{h, s, hi, float64(c.A) / 255}
}

// color returns the color of x.
func (x hsva) color() Color {
	c := x.v * x.s
	h := math.Mod(x.h, 360) / 60
	y := c * (1 - math.Abs(math.Mod(h, 2)-1))
	var r, g, b float64
	switch {
	case h < 1:
		r, g = c, y
	case h < 2:
		r, g = y, c
	case h < 3:
		g, b = c, y
	case h < 4:
		g, b = y, c
	case h < 5:
		r, b = y, c
	default:
		r, b = c, y
	}
	m := x.v - c
	return Color{R: byte8(r + m), G: byte8(g + m), B: byte8(b + m), A: byte8(x.a)}
}

func byte8(v float64) uint8 { return uint8(math.Round(math.Max(0, math.Min(1, v)) * 255)) }

// hexOf returns c as #rrggbb, or #rrggbbaa when it is not opaque.
func hexOf(c Color) string {
	if c.A == 255 {
		return fmt.Sprintf("#%02x%02x%02x", c.R, c.G, c.B)
	}
	return fmt.Sprintf("#%02x%02x%02x%02x", c.R, c.G, c.B, c.A)
}

// pickerState is the state of a color picker: the color as it keeps it,
// the color it last set or saw, and the text of its hex field.
type pickerState struct {
	hsv  hsva
	last Color
	hex  string
}

// swatches are the colors a picker offers, by name: the case of the
// message colorName of MyGo's strings.
var swatches = []struct {
	name  string
	color Color
}{
	{"red", Hex("#ef4444")}, {"orange", Hex("#f97316")}, {"yellow", Hex("#eab308")}, {"green", Hex("#22c55e")},
	{"teal", Hex("#14b8a6")}, {"blue", Hex("#3b82f6")}, {"indigo", Hex("#6366f1")}, {"purple", Hex("#a855f7")},
	{"pink", Hex("#ec4899")}, {"brown", Hex("#92400e")}, {"gray", Hex("#71717a")}, {"black", Hex("#000000")},
}

// ColorPicker creates a picker of *color, as AppKit's color panel: a
// square choosing the saturation, across, and the brightness, up, of the
// hue its slider chooses below it, a slider of opacity, the color in hex,
// which the user edits, and swatches of common colors. The arrows move
// across the square while it has the focus, Up and Down changing the
// brightness and Left and Right the saturation. Changed reports a new
// color.
func coreColorPicker(c *context, color *Color) *node {
	t := c.theme
	panel := coreColumn(c).Gap(t.Space(2.5)).Width(t.Space(60)).Shrink(0)
	panel.widget = "ColorPicker"
	st := coreLocal(panel, "picker", func() pickerState {
		return pickerState{hsv: toHSVA(*color), last: *color, hex: hexOf(*color)}
	})
	if *color != st.last {
		// The app set another color.
		st.hsv, st.last, st.hex = toHSVA(*color), *color, hexOf(*color)
	}
	set := func(x hsva) {
		x.s, x.v, x.a = math.Max(0, math.Min(1, x.s)), math.Max(0, math.Min(1, x.v)), math.Max(0, math.Min(1, x.a))
		st.hsv = x
		if next := x.color(); next != *color {
			*color = next
			panel.st.markChanged()
			c.rt.consumed = true
		}
		st.last, st.hex = *color, hexOf(*color)
	}
	x := st.hsv
	panel.Children(func() {
		// The square of saturation and brightness.
		sq := coreBox(c).Height(t.Space(40)).Radius(t.Radius).Focusable().FocusRing(false).Role(RoleSlider).Label(c.texts().Saturation())
		sq.flags |= flagDraggable | flagHover | flagOwnRing
		sq.Cursor(CursorCrosshair)
		sq.afterInput(func() {
			if s := sq.st; s.pressed && s.w > 0 && s.h > 0 {
				px, py := s.local(c.rt.pointerX, c.rt.pointerY)
				set(hsva{x.h, float64(px / s.w), 1 - float64(py/s.h), x.a})
			}
			switch {
			case sq.Shortcut(0, KeyRight):
				set(hsva{x.h, x.s + 0.01, x.v, x.a})
			case sq.Shortcut(0, KeyLeft):
				set(hsva{x.h, x.s - 0.01, x.v, x.a})
			case sq.Shortcut(0, KeyUp):
				set(hsva{x.h, x.s, x.v + 0.01, x.a})
			case sq.Shortcut(0, KeyDown):
				set(hsva{x.h, x.s, x.v - 0.01, x.a})
			}

		})
		x = st.hsv
		sq.hasRange, sq.accRange, sq.accStep = true, [3]float64{0, 100, math.Round(x.s * 100)}, 1
		sat, bri := math.Round(x.s*100), math.Round(x.v*100)
		sq.accValue = c.formatted(fmtKey{c.rt.lang, fmtColorValue, int64(sat)*1000 + int64(bri)}, func() string {
			return c.texts().ColorValue(sat/100, bri/100)
		})
		hue := hsva{x.h, 1, 1, 1}.color()
		sq.Draw(func(p *Painter, r Rect) {
			p.FillGradient(r, LinearGradient{From: RGB(255, 255, 255), To: hue, Angle: 90}, t.Radius)
			p.FillGradient(r, LinearGradient{From: RGBA(0, 0, 0, 0), To: RGB(0, 0, 0), Angle: 180}, t.Radius)
			// The color chosen, ringed.
			cx, cy := r.X+r.W*float32(x.s), r.Y+r.H*float32(1-x.v)
			d := t.Space(3.5)
			p.Stroke(Rect{cx - d/2, cy - d/2, d, d}, RGB(255, 255, 255), d/2, 2)
			p.Stroke(Rect{cx - d/2 - 1, cy - d/2 - 1, d + 2, d + 2}, RGBA(0, 0, 0, 0.3), d/2+1, 1)
			if sq.FocusVisible() {
				p.FocusRing(r, [4]float32{t.Radius, t.Radius, t.Radius, t.Radius})
			}
		})
		// The hue, and the opacity.
		h, a := x.h, x.a
		hs := channelSlider(c, &h, 360, "Hue", func(p *Painter, r Rect) {
			hues := []Color{RGB(255, 0, 0), RGB(255, 255, 0), RGB(0, 255, 0), RGB(0, 255, 255), RGB(0, 0, 255), RGB(255, 0, 255), RGB(255, 0, 0)}
			w := r.W / 6
			for i := range 6 {
				p.FillGradient(Rect{r.X + w*float32(i), r.Y, w + 0.5, r.H}, LinearGradient{From: hues[i], To: hues[i+1], Angle: 90}, 0)
			}
		})
		hs.afterInput(func() {
			if hs.Changed() {
				set(hsva{h, x.s, x.v, x.a})
			}

		})
		opaque := hsva{x.h, x.s, x.v, 1}.color()
		as := channelSlider(c, &a, 1, "Opacity", func(p *Painter, r Rect) {
			checkers(p, r, t.Space(1.5))
			p.FillGradient(r, LinearGradient{From: opaque.Alpha(0), To: opaque, Angle: 90}, 0)
		})
		as.afterInput(func() {
			if as.Changed() {
				set(hsva{x.h, x.s, x.v, a})
			}

		})
		// The color, and its hex.
		coreRow(c).Gap(t.Space(2)).AlignItems(Center).Children(func() {
			now := *color
			coreBox(c).Size(t.Space(7), t.Space(7)).Radius(t.Radius).Border(1, t.Border).Clip().Shrink(0).Draw(func(p *Painter, r Rect) {
				checkers(p, r, t.Space(1.5))
				p.Fill(r, now, 0)
			})
			in := coreTextInput(c, &st.hex).Label(c.texts().Hex()).Grow(1).FontFeatures("tnum")
			in.afterInput(func() {
				if v, err := parseHex(st.hex); err == nil && in.Changed() {
					hex := st.hex
					set(toHSVA(v))
					st.hex = hex // as typed
				}
				if !in.Focused() && st.hex != hexOf(*color) {
					st.hex = hexOf(*color)
				}
			})

		})
		// Swatches.
		coreGrid(c).ColumnTracks(Fr(1), Fr(1), Fr(1), Fr(1), Fr(1), Fr(1)).Gap(t.Space(1.5)).Children(func() {
			for i, sw := range swatches {
				name := c.formatted(fmtKey{c.rt.lang, fmtColorName, int64(i)}, func() string { return c.texts().ColorName(sw.name) })
				b := coreButtonBase(c).Height(t.Space(5)).Radius(t.Radius).Background(sw.color).Label(name).Tooltip(name)
				if *color == sw.color {
					b.Border(2, t.Text)
				}
				b.afterInput(func() {
					if b.Clicked() {
						set(toHSVA(sw.color))
					}
				})

			}
		})
	})
	return panel
}

// channelSlider creates a slider of a picker's channel from 0 to max,
// whose track track paints.
func channelSlider(c *context, v *float64, max float64, name string, track func(p *Painter, r Rect)) *node {
	t := c.theme
	kw := t.Space(3)
	s := sliderBase(c, v, 0, max, 0).Height(t.Space(4)).PaddingX(kw / 2).FocusRing(false).Label(name)
	frac := fraction(*v, 0, max)
	held := s.Animate("held", b2f(s.st.pressed), 150*time.Millisecond)
	s.Draw(func(p *Painter, r Rect) {
		bar := Rect{r.X, r.Y + t.Space(0.5), r.W, r.H - t.Space(1)}
		p.Clip(bar, bar.H/2, func() { track(p, bar) })
		x := r.X + kw/2 + (r.W-kw)*frac
		k := Rect{x - kw/2, r.Y, kw, r.H}
		g := 1 + 0.2*held
		k = Rect{x - k.W*g/2, r.Y + r.H/2 - k.H*g/2, k.W * g, k.H * g}
		p.Shadow(k, k.W/2, 0, 1, 3, 0, RGBA(0, 0, 0, 0.3))
		p.Fill(k, RGB(255, 255, 255), k.W/2)
		if s.FocusVisible() {
			p.FocusRing(k, [4]float32{k.W / 2, k.W / 2, k.W / 2, k.W / 2})
		}
	})
	return s
}

// checkers paints r with the gray squares that show transparency.
func checkers(p *Painter, r Rect, size float32) {
	p.Fill(r, RGB(255, 255, 255), 0)
	for y, row := r.Y, 0; y < r.Y+r.H; y, row = y+size, row+1 {
		for x, col := r.X, 0; x < r.X+r.W; x, col = x+size, col+1 {
			if (row+col)%2 == 0 {
				p.Fill(Rect{x, y, min(size, r.X+r.W-x), min(size, r.Y+r.H-y)}, RGB(204, 204, 204), 0)
			}
		}
	}
}

// ColorWell creates a swatch of *color that opens a ColorPicker below it,
// as AppKit's color well: a click, Enter or Space opens it, and Escape or
// a click outside closes it. Changed reports a new color. Assistive
// technology sees a color well whose value is the color in hex; name it
// with Label.
func coreColorWell(c *context, color *Color) *node {
	t := c.theme
	b := coreButtonBase(c).Padding(t.Space(1)).Radius(t.Radius).Background(t.Surface).Border(1, t.Border).Shrink(0)
	b.widget, b.role, b.accValue = "ColorWell", RoleColorWell, hexOf(*color)
	// Read as its value where buttons have none, as on Linux.
	b.description = hexOf(*color)
	open := coreLocal(b, "open", func() bool { return false })
	b.afterInput(func() {
		if b.Clicked() {
			*open = !*open
		}

	})
	b.expanded = *open
	now := *color
	b.Children(func() {
		coreBox(c).Size(t.Space(9), t.Space(5)).Radius(t.Space(1)).Clip().Shrink(0).Role(RoleNone).Draw(func(p *Painter, r Rect) {
			checkers(p, r, t.Space(1.25))
			p.Fill(r, now, 0)
		})
	})
	corePopoverBase(c, b, open, func(panel *node) {
		stylePanel(c, panel)
		panel.Padding(t.Space(3))
		picker := coreColorPicker(c, color)
		picker.afterInput(func() {
			if picker.Changed() {
				b.st.markChanged()
			}
		})
	})
	return b
}
