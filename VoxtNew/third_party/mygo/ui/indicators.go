package ui

import (
	"math"
	"time"
	"unicode"
)

// Spinner creates an indicator of work of unknown length, as AppKit's
// spinning progress indicator: spokes turning, as high as the font size,
// in the theme's muted text color, or the color TextColor gives it, as one
// on a chosen row of a list needs. Assistive technology sees a progress
// indicator of unknown length; name it with Label.
func coreSpinner(c *context) *node {
	t := c.theme
	size := t.FontSize * 1.25
	e := coreBox(c).Size(size, size).Shrink(0).Role(RoleProgress).TextColor(t.TextMuted)
	e.widget = "Spinner"
	e.hasRange, e.accRange = true, [3]float64{0, 1, -1}
	e.Draw(func(p *Painter, r Rect) {
		// A spoke a twelfth of a turn on every twelfth of 0.9 s: it is
		// painted again for the next, while it shows, without building the
		// view.
		const spokes, turn = 12, 900
		ms := int(p.Now().UnixMilli() % turn)
		lead := ms * spokes / turn
		p.After(time.Duration((lead+1)*turn/spokes-ms) * time.Millisecond)
		cx, cy := r.X+r.W/2, r.Y+r.H/2
		in, out, w := r.W*0.22, r.W*0.46, r.W*0.09
		for i := range spokes {
			at := rotate(cx, cy, float32(i)*360/spokes)
			// Fading behind the spoke leading.
			age := (lead - i + spokes) % spokes
			alpha := 1 - float32(age)/spokes*0.85
			var path Path
			path.MoveTo(at(0, -in)).LineTo(at(0, -out))
			p.StrokePath(&path, w, e.ts.color.Alpha(alpha))
		}
	})
	return e
}

// MeterLevels are the levels of a Meter from which its value shows it is
// worse: the theme's Warning color from Warning, its Danger color from
// Critical. A Critical below Warning makes low values the bad ones, as a
// battery's. Their zero value shows the theme's Success color throughout.
type MeterLevels struct {
	Warning, Critical float64
}

// Meter creates a bar showing value between lo and hi, as AppKit's level
// indicator and the web's meter: the theme's Success color, or Warning and
// Danger from the levels given, nil for none. Assistive technology sees a
// level indicator of that value; name it with Label.
//
//	ui.Meter(c, app.disk.Used, 0, app.disk.Size, &ui.MeterLevels{Warning: 0.8 * size, Critical: 0.95 * size}).Label("Disk")
func coreMeter(c *context, value, lo, hi float64, levels *MeterLevels) *node {
	t := c.theme
	rad := t.Space(1)
	e := coreBox(c).Height(t.Space(2)).MinWidth(t.Space(20)).Radius(rad).Background(t.Border).Clip().Role(RoleMeter)
	e.widget = "Meter"
	e.hasRange, e.accRange = true, [3]float64{lo, hi, value}
	color := t.Success
	if l := levels; l != nil && l.Warning != l.Critical {
		low := l.Critical < l.Warning
		switch {
		case !low && value >= l.Critical, low && value <= l.Critical:
			color = t.Danger
		case !low && value >= l.Warning, low && value <= l.Warning:
			color = t.Warning
		}
	}
	frac := fraction(value, lo, hi)
	rtl := e.rtl()
	e.Draw(func(p *Painter, r Rect) {
		// From its start: the right right to left.
		x := r.X
		if rtl {
			x = r.X + r.W*(1-frac)
		}
		p.Fill(Rect{x, r.Y, r.W * frac, r.H}, color, rad)
	})
	return e
}

// Rating creates a row of max stars, as AppKit's rating level indicator,
// of which *value are filled: a click on a star sets it, a click on the
// star set clears it, and the arrows, Home and End change it while the
// rating has the focus. Changed reports a new value. Assistive technology
// sees a slider from 0 to max; name it with Label.
func coreRating(c *context, value *int, max int) *node {
	t := c.theme
	e := coreRow(c).Gap(t.Space(0.5)).Focusable().Shrink(0).Role(RoleSlider)
	e.widget = "Rating"
	e.flags |= flagHover
	set := func(v int) {
		v = min(max, v)
		if v < 0 {
			v = 0
		}
		if v != *value {
			*value = v
			e.st.markChanged()
			c.rt.consumed = true
		}
	}
	e.afterInput(func() {
		// The stars go from the right right to left, as the arrows do.
		back, forth := e.arrows()
		switch {
		case e.Shortcut(0, forth), e.Shortcut(0, KeyUp):
			set(*value + 1)
		case e.Shortcut(0, back), e.Shortcut(0, KeyDown):
			set(*value - 1)
		case e.Shortcut(0, KeyHome):
			set(0)
		case e.Shortcut(0, KeyEnd):
			set(max)
		}
	})
	e.hasRange, e.accRange, e.accStep = true, [3]float64{0, float64(max), float64(*value)}, 1
	// The stars the pointer would set, shown as it rests on them.
	hover := -1
	e.Children(func() {
		for i := range max {
			star := coreBox(c).Size(t.FontSize*1.25, t.FontSize*1.25).Shrink(0).Role(RoleNone)
			star.flags |= flagClickable | flagHover
			star.afterInput(func() {
				if star.Clicked() {
					if *value == i+1 {
						set(0)
					} else {
						set(i + 1)
					}
					e.Focus()
				}
			})
			if star.Hovered() {
				hover = i
			}
		}
	})
	k := 0
	for ch := e.first; ch != nil; ch = ch.next {
		i := k
		k++
		ch.Draw(func(p *Painter, r Rect) {
			filled := i < *value
			color := t.Warning
			if hover >= 0 {
				filled, color = i <= hover, t.Warning.Alpha(0.6)
			}
			path := starPath(r)
			if filled {
				p.FillPath(path, color)
			} else {
				p.StrokePath(path, 1.2, t.TextMuted)
			}
		})
	}
	e.DrawOver(func(p *Painter, r Rect) {
		// Around the stars, which a row stretched leaves at its start.
		if first, last := e.first, e.last; e.FocusVisible() && first != nil {
			stars := union(Rect{first.x, first.y, first.w, first.h}, Rect{last.x, last.y, last.w, last.h})
			p.FocusRing(stars, [4]float32{t.Radius, t.Radius, t.Radius, t.Radius})
		}
	})
	e.flags |= flagOwnRing
	return e
}

// starPath returns a five-pointed star in r.
func starPath(r Rect) *Path {
	var path Path
	cx, cy := r.X+r.W/2, r.Y+r.H/2+r.H*0.04
	outer, inner := r.W*0.48, r.W*0.2
	for i := range 10 {
		rad := outer
		if i%2 == 1 {
			rad = inner
		}
		a := float64(i)*math.Pi/5 - math.Pi/2
		x, y := cx+rad*float32(math.Cos(a)), cy+rad*float32(math.Sin(a))
		if i == 0 {
			path.MoveTo(x, y)
		} else {
			path.LineTo(x, y)
		}
	}
	path.Close()
	return &path
}

// Stepper creates a pair of arrows changing *value by step between lo and
// hi, as AppKit's stepper beside a field: a click on an arrow steps up or
// down, as do Up and Down while the stepper has the focus, and holding an
// arrow keeps stepping. Changed reports a new value. Assistive technology
// sees a spin button; name it with Label.
func coreStepper(c *context, value *float64, lo, hi, step float64) *node {
	t := c.theme
	e := coreColumn(c).Focusable().Shrink(0).Width(t.Space(5)).Radius(t.Radius).Border(1, t.Border).Background(t.Surface).Clip().Role(RoleStepper)
	e.widget = "Stepper"
	e.flags |= flagOwnRing
	set := func(v float64) {
		v = snap(math.Max(lo, math.Min(hi, v)), lo, hi, step)
		if v != *value {
			*value = v
			e.st.markChanged()
			c.rt.consumed = true
		}
	}
	e.afterInput(func() {
		back, forth := e.arrows()
		switch {
		case e.Shortcut(0, KeyUp), e.Shortcut(0, forth):
			set(*value + step)
		case e.Shortcut(0, KeyDown), e.Shortcut(0, back):
			set(*value - step)
		case e.Shortcut(0, KeyHome):
			set(lo)
		case e.Shortcut(0, KeyEnd):
			set(hi)
		}

	})
	e.hasRange, e.accRange, e.accStep = true, [3]float64{lo, hi, *value}, step
	e.Children(func() {
		for _, up := range []bool{true, false} {
			arrow := coreBox(c).Height(t.Space(3.5)).Role(RoleNone)
			arrow.flags |= flagClickable | flagHover
			can := up && *value < hi || !up && *value > lo
			// A press steps at once, and again as it is held, faster
			// after a while, as AppKit's.
			arrow.afterInput(func() {
				if arrow.Pressed() && can {
					s := arrow.st
					held := c.now.Sub(s.holdStart)
					if s.holdStart.IsZero() || !s.holding {
						s.holdStart, s.holding, s.holdSteps = c.now, true, 0
						held = 0
					}
					due := 1
					if held > 400*time.Millisecond {
						due += int((held - 400*time.Millisecond) / (80 * time.Millisecond))
					}
					for s.holdSteps < due {
						s.holdSteps++
						if up {
							set(*value + step)
						} else {
							set(*value - step)
						}
					}
					e.Focus()
					c.After(80 * time.Millisecond)
				} else {
					arrow.st.holding = false
				}

			})
			arrow.styleFn = func(a *node) {
				if a.Pressed() {
					a.bg = t.SurfacePressed
				} else if a.Hovered() {
					a.bg = t.SurfaceHover
				}
			}
			arrow.Draw(func(p *Painter, r Rect) {
				cx, cy, d := r.X+r.W/2, r.Y+r.H/2, r.H*0.22
				var path Path
				if up {
					path.MoveTo(cx-d, cy+d/2).LineTo(cx, cy-d/2).LineTo(cx+d, cy+d/2)
				} else {
					path.MoveTo(cx-d, cy-d/2).LineTo(cx, cy+d/2).LineTo(cx+d, cy-d/2)
				}
				color := t.Text
				if !can {
					color = t.TextMuted.Alpha(0.5)
				}
				p.StrokePath(&path, 1.5, color)
			})
		}
	})
	e.DrawOver(func(p *Painter, r Rect) {
		if e.FocusVisible() {
			p.FocusRing(r, e.radius)
		}
	})
	return e
}

// RangeSlider creates a slider of two knobs setting *low and *high between
// lo and hi, as a price range: each knob takes the focus, which Tab moves
// between, and the arrows, Home and End move it while it has the focus;
// a press on the track moves the knob nearest to it. A step above 0 snaps
// the values to lo and multiples of step from it, with tick marks. Changed
// reports a new value. Assistive technology sees two sliders, named by
// the RangeSlider's Label and "minimum" and "maximum".
func coreRangeSlider(c *context, low, high *float64, lo, hi, step float64) *node {
	t := c.theme
	kw := t.Space(5)
	e := coreBox(c).Height(t.Space(5)).MinWidth(t.Space(25)).PaddingX(kw / 2).Shrink(0).Role(RoleGroup)
	e.widget = "RangeSlider"
	// Pressed, not clicked: no button to assistive technology.
	e.flags |= flagDraggable | flagHover
	ticks := tickCount(lo, hi, step)
	if ticks > 0 {
		e.Height(t.Space(7))
	}
	*low, *high = math.Max(lo, math.Min(*low, *high)), math.Min(hi, math.Max(*low, *high))
	set := func(v *float64, to float64, from, until float64) {
		to = snap(math.Max(from, math.Min(until, to)), lo, hi, step)
		if to != *v {
			*v = to
			e.st.markChanged()
			c.rt.consumed = true
		}
	}
	keyStep := step
	if keyStep <= 0 {
		keyStep = (hi - lo) / 100
	}
	// Which knob the pointer drags: the one nearest to where it pressed.
	dragging := coreLocal(e, "knob", func() int { return -1 })
	st := e.st
	// Right to left, lo is at the right, as a Slider's.
	rtl := e.rtl()
	at := func() float64 {
		x, _ := st.local(c.rt.pointerX, c.rt.pointerY)
		f := (x + st.x - st.cx) / max(st.cw, 1)
		if rtl {
			f = 1 - f
		}
		return lo + float64(max(0, min(1, f)))*(hi-lo)
	}
	if !st.pressed {
		*dragging = -1
	}
	var knobs [2]*node
	e.Children(func() {
		for k, v := range []*float64{low, high} {
			from, to := lo, *high
			if k == 1 {
				from, to = *low, hi
			}
			frac := fraction(*v, lo, hi)
			knob := coreBox(c).Absolute().Top(0).Bottom(0).Width(kw).Focusable().FocusRing(false).Role(RoleSlider)
			knob.StartPercent(frac * 100).MarginStart(-frac * kw)
			knob.flags |= flagDraggable | flagHover
			knob.hasRange, knob.accRange, knob.accStep = true, [3]float64{lo, hi, *v}, keyStep
			// Named after the slider, as "Price minimum".
			knob.label, knob.nameFrom, knob.nameJoin = []string{"minimum", "maximum"}[k], e, true
			knob.afterInput(func() {
				back, forth := knob.arrows()
				switch {
				case knob.Shortcut(0, back), knob.Shortcut(0, KeyDown):
					set(v, *v-keyStep, from, to)
				case knob.Shortcut(0, forth), knob.Shortcut(0, KeyUp):
					set(v, *v+keyStep, from, to)
				case knob.Shortcut(0, KeyHome):
					set(v, from, from, to)
				case knob.Shortcut(0, KeyEnd):
					set(v, to, from, to)
				}

			})
			if knob.st.pressed {
				*dragging = k
			}
			knobs[k] = knob
		}
	})
	e.afterInput(func() {
		if (st.pressed || knobs[0].st.pressed || knobs[1].st.pressed) && !e.disabled() {
			x := at()
			if *dragging < 0 {
				// The nearest knob, the high one when they meet past it.
				*dragging = 0
				if math.Abs(x-*high) < math.Abs(x-*low) || x > *high {
					*dragging = 1
				}
			}
			if *dragging == 0 {
				set(low, x, lo, *high)
			} else {
				set(high, x, *low, hi)
			}
			knobs[*dragging].Focus()
		}

	})
	held := e.Animate("held", b2f(*dragging >= 0), 150*time.Millisecond)
	which := *dragging
	e.Draw(func(p *Painter, r Rect) {
		h := t.Space(1.5)
		cy := r.Y + t.Space(2.5)
		paintTicks(p, t, r, kw, ticks, cy+t.Space(2)+t.Space(1))
		f0, f1 := fraction(*low, lo, hi), fraction(*high, lo, hi)
		if rtl {
			f0, f1 = 1-f0, 1-f1
		}
		x0 := r.X + kw/2 + (r.W-kw)*f0
		x1 := r.X + kw/2 + (r.W-kw)*f1
		p.Fill(Rect{r.X, cy - h/2, r.W, h}, t.Border, h/2)
		p.Fill(Rect{min(x0, x1), cy - h/2, max(x0, x1) - min(x0, x1), h}, t.Accent, h/2)
		for k, x := range []float32{x0, x1} {
			g := float32(0)
			if k == which {
				g = held
			}
			paintKnob(p, t, knobRect(t, x, cy, g), g, knobs[k].FocusVisible())
		}
	})
	return e
}

// Avatar creates a picture of a person or a thing named name: the image,
// cropped to a circle, or the initials of name on a color it picks from
// it, as high as twice the font size. Assistive technology sees an image
// named name.
func coreAvatar(c *context, name string, image *Bitmap) *node {
	t := c.theme
	size := t.FontSize * 2.25
	e := coreBox(c).Size(size, size).Radius(size / 2).Clip().Shrink(0).Center().Role(RoleImage).Label(name)
	e.widget = "Avatar"
	if image != nil {
		e.Children(func() { coreImage(c, image).Fit(Cover).Size(size, size) })
		return e
	}
	// The hue of the name's FNV-1a hash.
	h := uint32(2166136261)
	for i := 0; i < len(name); i++ {
		h = (h ^ uint32(name[i])) * 16777619
	}
	hue := float64(h % 360)
	e.Background(hslColor(hue, 0.45, 0.55)).TextColor(RGB(255, 255, 255))
	// The initials of the last frame's name, unless it changed.
	in := coreLocal(e, "initials", func() [2]string { return [2]string{} })
	if in[0] != name || in[1] == "" && name != "" {
		in[0], in[1] = name, initials(name)
	}
	e.Children(func() {
		coreText(c, in[1]).FontWeight(600).FontSize(size * 0.4).SingleLine()
	})
	return e
}

// initials returns the first letters of the first and last words of name.
func initials(name string) string {
	var first, last rune
	inWord := false
	for _, r := range name {
		letter := unicode.IsLetter(r) || unicode.IsDigit(r)
		if letter && !inWord {
			if first == 0 {
				first = unicode.ToUpper(r)
			} else {
				last = unicode.ToUpper(r)
			}
		}
		inWord = letter
	}
	switch {
	case first == 0:
		return ""
	case last == 0:
		return string(first)
	}
	return string([]rune{first, last})
}

// hslColor returns the color of hue h in degrees, saturation s and
// lightness l.
func hslColor(h, s, l float64) Color {
	c := (1 - math.Abs(2*l-1)) * s
	x := c * (1 - math.Abs(math.Mod(h/60, 2)-1))
	m := l - c/2
	var r, g, b float64
	switch {
	case h < 60:
		r, g = c, x
	case h < 120:
		r, g = x, c
	case h < 180:
		g, b = c, x
	case h < 240:
		g, b = x, c
	case h < 300:
		r, b = x, c
	default:
		r, b = c, x
	}
	return RGB(uint8((r+m)*255), uint8((g+m)*255), uint8((b+m)*255))
}
