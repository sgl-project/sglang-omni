package ui

import (
	"fmt"
	"math"
	"strconv"
	"strings"
	"time"
)

// Box creates a container that lays its children out in a column.
func coreBox(c *context) *node { return c.newElement(kindBox) }

// Column creates a container that lays its children out from top to
// bottom, stretched to its width.
func coreColumn(c *context) *node { return c.newElement(kindBox) }

// Row creates a container that lays its children out from left to right,
// centered vertically.
func coreRow(c *context) *node { return c.newElement(kindBox).Row() }

// Text creates a text, which wraps at the width it gets.
func coreText(c *context, s string) *node {
	e := c.newElement(kindText)
	e.text = s
	return e
}

// Textf creates a text formatted with fmt.Sprintf.
func coreTextf(c *context, format string, args ...any) *node {
	return coreText(c, fmt.Sprintf(format, args...))
}

// Spacer creates an empty element that takes the free space of its row or
// column, pushing its siblings apart.
func coreSpacer(c *context) *node { return coreBox(c).Grow(1) }

// Divider creates a thin line across its row or column.
func coreDivider(c *context) *node {
	row := c.parent.row
	e := coreBox(c).Background(c.theme.Border).Shrink(0).AlignSelf(Stretch)
	if row {
		return e.Width(1)
	}
	return e.Height(1)
}

// Animate returns a value that moves to target over d, easing out, and
// keeps frames coming while it moves; key tells apart the animations of
// the element. The value starts at the first target. When the desktop asks
// for less motion (Preferences.ReduceMotion), it goes to target at once.
// AnimateTo takes springs and delays too.
func (e *node) Animate(key any, target float32, d time.Duration) float32 {
	return e.AnimateWith(key, target, d, EaseOut)
}

// AnimateWith returns a value that moves to target over d as Animate does,
// along ease.
func (e *node) AnimateWith(key any, target float32, d time.Duration, ease Easing) float32 {
	return e.AnimateTo(key, target, Animation{Duration: max(d, time.Millisecond), Ease: ease})
}

// Loop returns the progress of an animation that starts over every
// period, from 0 to 1 along ease, and keeps frames coming while the
// element is built; key tells apart the animations of the element. A
// spinner turns with Loop("spin", time.Second, ui.Linear) * 360, and a
// placeholder pulses with an opacity of 0.5 + 0.5*Loop("pulse", d,
// ui.Bounce(ui.EaseInOut)).
func (e *node) Loop(key any, period time.Duration, ease Easing) float32 {
	st := e.st
	if st.anims == nil {
		st.anims = map[any]*anim{}
	}
	a := st.anims[key]
	now := e.c.now
	if a == nil {
		a = &anim{start: now}
		st.anims[key] = a
	}
	e.c.AnimationFrame()
	p := max(period, time.Millisecond)
	t := float32(now.Sub(a.start)%p) / float32(p)
	return ease(t)
}

// Easing maps the time an animation has run, from 0 to 1 of its duration,
// to how far its value has gone, as CSS's timing functions do. Linear,
// EaseIn, EaseOut and EaseInOut are easings, and Bounce makes more.
type Easing func(t float32) float32

// Linear moves at one speed.
func Linear(t float32) float32 { return t }

// EaseIn starts slowly (a cubic).
func EaseIn(t float32) float32 { return t * t * t }

// EaseOut ends slowly (a cubic), as Animate moves.
func EaseOut(t float32) float32 {
	u := 1 - t
	return 1 - u*u*u
}

// EaseInOut starts and ends slowly (a cubic).
func EaseInOut(t float32) float32 {
	if t < 0.5 {
		return 4 * t * t * t
	}
	u := 2 - 2*t
	return 1 - u*u*u/2
}

// Bounce returns an easing that goes along ease and comes back, in the
// same time, as a pulse does.
func Bounce(ease Easing) Easing {
	return func(t float32) float32 {
		if t < 0.5 {
			return ease(2 * t)
		}
		return ease(2 - 2*t)
	}
}

// anim is a Loop's: when it started.
type anim struct{ start time.Time }

func b2f(b bool) float32 {
	if b {
		return 1
	}
	return 0
}

// Button creates a button showing label. Ask Clicked whether it was
// clicked; give it other content with Children and an empty label.
func coreButton(c *context, label string) *node { return button(c, label, false) }

// PrimaryButton creates a button in the accent color, for the main action.
func corePrimaryButton(c *context, label string) *node { return button(c, label, true) }

func button(c *context, label string, primary bool) *node {
	b := coreButtonBase(c)
	styleButton(c, b, primary)
	if label != "" {
		b.Children(func() { coreText(c, label).SingleLine() })
	}
	return b
}

// styleButton gives a button the theme's look, in the accent color when
// primary: without a face of its own until hovered in a toolbar, and as a
// segment in a ToggleGroup or Segmented.
func styleButton(c *context, b *node, primary bool) {
	t := c.theme
	b.Padding(t.Space(1.5), t.Space(3.5)).Gap(t.Space(1.5)).Radius(t.Radius)
	base, hover, pressed, fg, border := t.Surface, t.SurfaceHover, t.SurfacePressed, t.Text, t.Border
	switch {
	case primary:
		base, hover, pressed, fg, border = t.Accent, t.AccentHover, t.AccentPressed, t.AccentText, Color{}
	case c.buttons == toolbarButtons:
		b.Padding(t.Space(1.5), t.Space(2.5))
		base, border = Color{}, Color{}
	case c.buttons == segmentButtons:
		b.Padding(t.Space(1), t.Space(3)).Radius(max(t.Radius-t.Space(0.5), 0))
		base, border = Color{}, Color{}
		b.segment = true
	}
	b.Background(base).TextColor(fg)
	if border.A > 0 {
		b.Border(1, border)
	}
	b.styleFn = func(b *node) {
		if b.bg != base || b.IsDisabled() {
			return
		}
		if b.Pressed() {
			b.bg = pressed
		} else if b.Hovered() {
			b.bg = hover
		}
	}
}

// Link creates a text that opens url in the browser when clicked, or
// Enter while it has the focus. In a page of a Router, a path without a
// scheme, as "/notes/42" or "edit", goes there in the router (Push).
// Inside a RichText it is a link within the paragraph; give it an empty
// label and Children to style parts of its text.
func coreLink(c *context, label, url string) *node {
	t := c.theme
	e := coreText(c, label).TextColor(t.Accent).Cursor(CursorPointer).Focusable()
	e.widget, e.role = "Link", RoleLink
	router := c.router
	e.afterInput(func() {
		if e.Clicked() && url != "" {
			if r := router; r != nil && isPath(url) {
				r.Push(url)
			} else {
				c.rt.host.openURL(url, nil)
			}
		}
		if e.Hovered() {
			e.Underline()
		}
	})
	return e
}

func checkPath(r Rect) *Path {
	var p Path
	return p.MoveTo(r.X+r.W*0.22, r.Y+r.H*0.52).LineTo(r.X+r.W*0.42, r.Y+r.H*0.71).LineTo(r.X+r.W*0.78, r.Y+r.H*0.31)
}

// Checkbox creates a check box toggling *checked, with a label.
func coreCheckbox(c *context, checked *bool, label string) *node {
	t := c.theme
	row := coreCheckboxBase(c, checked).Gap(t.Space(2)).FocusRing(false)
	on := *checked
	row.Children(func() {
		box := coreBox(c).Size(t.Space(4), t.Space(4)).Radius(t.Space(1)).Shrink(0)
		if on {
			box.Background(t.Accent)
		} else {
			box.Background(t.Background).Border(1, t.Border.Mix(t.Text, 0.25))
		}
		box.DrawOver(func(p *Painter, r Rect) {
			if on {
				p.StrokePath(checkPath(r), t.Space(0.5), t.AccentText)
			}
			if row.FocusVisible() {
				p.FocusRing(r, box.radius)
			}
		})
		box.styleFn = func(box *node) {
			if !on && row.Hovered() {
				box.borderC = t.Accent
			}
		}
		if label != "" {
			coreText(c, label)
		}
	})
	return row
}

// Radio creates a radio button that selects value into *selected, with a
// label.
func coreRadio[T comparable](c *context, selected *T, value T, label string) *node {
	t := c.theme
	row := coreRadioBase(c, selected, value).Gap(t.Space(2)).FocusRing(false)
	on := *selected == value
	row.Children(func() {
		dot := coreBox(c).Size(t.Space(4), t.Space(4)).Radius(t.Space(2)).Shrink(0)
		if on {
			dot.Background(t.Accent)
		} else {
			dot.Background(t.Background).Border(1, t.Border.Mix(t.Text, 0.25))
		}
		dot.DrawOver(func(p *Painter, r Rect) {
			if on {
				d := t.Space(1.5)
				p.Fill(Rect{r.X + (r.W-d)/2, r.Y + (r.H-d)/2, d, d}, t.AccentText, d/2)
			}
			if row.FocusVisible() {
				p.FocusRing(r, dot.radius)
			}
		})
		dot.styleFn = func(dot *node) {
			if !on && row.Hovered() {
				dot.borderC = t.Accent
			}
		}
		if label != "" {
			coreText(c, label)
		}
	})
	return row
}

// Switch creates a switch toggling *on.
func coreSwitch(c *context, on *bool) *node {
	t := c.theme
	sw := coreSwitchBase(c, on).Size(t.Space(9), t.Space(5)).Radius(t.Space(2.5))
	pos := sw.Animate("knob", b2f(*on), 140*time.Millisecond)
	off := t.Border.Mix(t.Text, 0.15)
	sw.Background(off.Mix(t.Accent, pos))
	rtl := sw.rtl()
	sw.Draw(func(p *Painter, r Rect) {
		in := t.Space(0.5)
		d := r.H - 2*in
		// On at the end: the left right to left, as UISwitch mirrors.
		at := pos
		if rtl {
			at = 1 - pos
		}
		knob := Rect{r.X + in + at*(r.W-r.H), r.Y + in, d, d}
		p.Shadow(knob, d/2, 0, 1, 3, 0, RGBA(0, 0, 0, 0.25))
		p.Fill(knob, RGB(255, 255, 255), d/2)
	})
	return sw
}

// Slider creates a slider setting *value between lo and hi.
func coreSlider(c *context, value *float64, lo, hi float64) *node {
	return slider(c, value, lo, hi, 0)
}

// StepSlider creates a slider setting *value to lo or a multiple of step
// from it, up to hi, with a tick mark at each, as AppKit's sliders with
// tick marks: dragging snaps to them, and the arrows step by step.
func coreStepSlider(c *context, value *float64, lo, hi, step float64) *node {
	return slider(c, value, lo, hi, step)
}

// slider creates a Slider, or with step a StepSlider.
func slider(c *context, value *float64, lo, hi, step float64) *node {
	t := c.theme
	// The knob, a capsule like AppKit's, moves across the content box,
	// half of it inside the padding on each side, so it stays on the
	// track, which spans the slider. Held, it grows into a lens of glass
	// that the track shows through, as AppKit's does.
	kw, kh := t.Space(5), t.Space(4)
	s := sliderBase(c, value, lo, hi, step).Height(t.Space(5)).MinWidth(t.Space(20)).PaddingX(kw / 2).FocusRing(false)
	ticks := tickCount(lo, hi, step)
	if ticks > 0 {
		s.Height(t.Space(7))
	}
	frac := fraction(*value, lo, hi)
	// Right to left, lo is at the right (sliderInput), and the track
	// fills from there.
	rtl := s.rtl()
	if rtl {
		frac = 1 - frac
	}
	held := s.Animate("held", b2f(s.st.pressed), 150*time.Millisecond)
	s.Draw(func(p *Painter, r Rect) {
		// As thick as Progress; above the ticks.
		h := t.Space(1.5)
		cy := r.Y + t.Space(2.5)
		track := Rect{r.X, cy - h/2, r.W, h}
		paintTicks(p, t, r, kw, ticks, cy+kh/2+t.Space(1))
		x := r.X + kw/2 + (r.W-kw)*frac
		k := knobRect(t, x, cy, held)
		if held > 0 {
			// The lens lightens what is behind it, under the track.
			p.Fill(k, RGBA(255, 255, 255, 0.1*held), k.H/2)
		}
		p.Fill(track, t.Border, h/2)
		if rtl {
			p.Fill(Rect{x, track.Y, track.X + track.W - x, h}, t.Accent, h/2)
		} else {
			p.Fill(Rect{track.X, track.Y, x - track.X, h}, t.Accent, h/2)
		}
		paintKnob(p, t, k, held, s.FocusVisible())
	})
	return s
}

// fraction returns where v is from lo to hi, from 0 to 1.
func fraction(v, lo, hi float64) float32 {
	if hi <= lo {
		return 0
	}
	return float32(max(0, min(1, (v-lo)/(hi-lo))))
}

// snap returns v rounded to lo or a multiple of step from it, within hi;
// v itself for no step.
func snap(v, lo, hi, step float64) float64 {
	if step <= 0 {
		return v
	}
	v = lo + math.Round((v-lo)/step)*step
	if v > hi {
		v -= step
	}
	// Away from the rounding of floating point, as 0.1 steps add up: to
	// the decimals of the step and of lo, as browsers round the values of
	// their number inputs.
	d := max(decimals(step), decimals(lo))
	if d >= 15 {
		return v
	}
	r, err := strconv.ParseFloat(strconv.FormatFloat(v, 'f', d, 64), 64)
	if err != nil {
		return v
	}
	return r
}

// decimals returns how many digits a number has after the decimal point,
// written as briefly as it reads back the same.
func decimals(v float64) int {
	s := strconv.FormatFloat(math.Abs(v), 'e', -1, 64)
	mant, exp, _ := strings.Cut(s, "e")
	e, _ := strconv.Atoi(exp)
	digits := 0
	if _, frac, ok := strings.Cut(mant, "."); ok {
		digits = len(frac)
	}
	return max(digits-e, 0)
}

// tickCount returns how many tick marks a slider of step shows, 0 for none
// or too many to tell apart.
func tickCount(lo, hi, step float64) int {
	if step <= 0 || hi <= lo {
		return 0
	}
	n := int(math.Floor((hi-lo)/step+1e-9)) + 1
	if n > 61 {
		return 0
	}
	return n
}

// paintTicks paints n tick marks across a slider's track, at y.
func paintTicks(p *Painter, t *Theme, r Rect, kw float32, n int, y float32) {
	for i := range n {
		x := r.X + kw/2 + (r.W-kw)*float32(i)/float32(max(n-1, 1))
		p.Fill(Rect{x - 0.5, y, 1, t.Space(1.5)}, t.TextMuted.Alpha(0.6), 0)
	}
}

// knobRect returns the box of a slider's knob centered at (x, y), held
// from 0 to 1, when it grows into a lens.
func knobRect(t *Theme, x, y, held float32) Rect {
	kw, kh := t.Space(5), t.Space(4)
	g := 1 + 0.35*held
	return Rect{x - kw*g/2, y - kh*g/2, kw * g, kh * g}
}

// paintKnob paints a slider's knob k, held from 0 to 1, over its track:
// a capsule of white, which grows into a lens of glass as it is held, as
// AppKit's does.
func paintKnob(p *Painter, t *Theme, k Rect, held float32, focus bool) {
	face, rim, drop := RGB(255, 255, 255), RGBA(0, 0, 0, 0.22), RGBA(0, 0, 0, 0.14)
	if t.Dark {
		face, rim, drop = RGB(224, 225, 225), RGBA(0, 0, 0, 0.7), RGBA(0, 0, 0, 0.3)
	}
	rad := k.H / 2
	if held > 0 {
		p.Shadow(k, rad, 0, 8, 14, -3, drop.Alpha(held))
	}
	if held < 1 {
		// A tight shadow edges the knob, a soft one lifts it.
		p.Shadow(k, rad, 0, 0.5, 1, 0, RGBA(0, 0, 0, 0.08*(1-held)))
		p.Shadow(k, rad, 0, 1.5, 7, 0, RGBA(0, 0, 0, 0.1*(1-held)))
		p.Fill(k, face.Alpha(1-held), rad)
	}
	if held > 0 {
		// The lens's rim, lit on the inside.
		p.Stroke(k, rim.Alpha(held), rad, 1)
		p.Stroke(Rect{k.X + 1, k.Y + 1, k.W - 2, k.H - 2}, RGBA(255, 255, 255, 0.25*held), rad-1, 1)
	}
	if focus {
		p.FocusRing(k, [4]float32{rad, rad, rad, rad})
	}
}

// Progress creates a progress bar filled to value between 0 and 1; a
// negative value shows activity of unknown length. It fills from its
// start: the right right to left (Direction). Reverse fills it from the
// other side.
func coreProgress(c *context, value float64) *node {
	t := c.theme
	rad := t.Space(0.75)
	e := coreBox(c).Height(t.Space(1.5)).Radius(rad).Background(t.Border).Clip()
	e.role, e.hasRange, e.accRange = RoleProgress, true, [3]float64{0, 1, value}
	e.Draw(func(p *Painter, r Rect) {
		fromRight := e.reverse != e.rtl()
		if value >= 0 {
			w := r.W * float32(math.Min(value, 1))
			x := r.X
			if fromRight {
				x = r.X + r.W - w
			}
			p.Fill(Rect{x, r.Y, w, r.H}, t.Accent, rad)
			return
		}
		// It moves while it shows, painted again without building the
		// view.
		p.AnimationFrame()
		phase := float32(p.Now().UnixMilli()%1400) / 1400
		w := r.W * 0.3
		x := r.X - w + (r.W+w)*phase
		if fromRight {
			x = r.X + r.W - (r.W+w)*phase
		}
		p.Clip(r, rad, func() { p.Fill(Rect{x, r.Y, w, r.H}, t.Accent, rad) })
	})
	return e
}

// Scroll creates a container that scrolls its children vertically. Give
// it a size, or Grow it within its parent.
func coreScroll(c *context) *node {
	e := coreBox(c)
	e.flags |= flagScrollY | flagHover
	return e
}

// ScrollHorizontal creates a row that scrolls its children horizontally.
func coreScrollHorizontal(c *context) *node {
	e := coreRow(c)
	e.flags |= flagScrollX | flagHover
	return e
}

// ScrollBoth creates a container that scrolls its children both ways, as
// a canvas, a wide table or code does. Give it a size, or Grow it within
// its parent.
func coreScrollBoth(c *context) *node {
	e := coreBox(c)
	e.flags |= flagScrollX | flagScrollY | flagHover
	return e
}

// ImageSource is what Image shows: a *Bitmap, or an *SVG in its own
// colors.
type ImageSource interface {
	imageSize() (w, h float32)
}

// Image creates an element showing a bitmap, or an SVG in its own colors
// (with the text color for its currentColor), by default at its size as
// DIPs, scaled to fit when given another size.
func coreImage(c *context, src ImageSource) *node {
	e := c.newElement(kindImage)
	switch s := src.(type) {
	case *Bitmap:
		e.image = s
	case *SVG:
		e.svg = s
	}
	if src != nil {
		if w, h := src.imageSize(); h > 0 {
			e.aspect = w / h
		}
	}
	return e
}

// intrinsicSize returns the size of an image's picture, or of an icon: as
// high as the font size.
func (e *node) intrinsicSize() (w, h float32) {
	switch e.kind {
	case kindImage:
		if e.image != nil {
			return e.image.imageSize()
		}
		return e.svg.imageSize()
	case kindIcon:
		em := e.resolvedText().size
		if s := e.svg; s != nil && s.h > 0 {
			return em * s.w / s.h, em
		}
		return em, em
	}
	return 0, 0
}

// Fit sets how an Image fills its box.
func (e *node) Fit(f Fit) *node { e.fit = f; return e }

// Overlay builds fn's elements above the rest of the window. Place them
// with Absolute, Left and Top, in DIPs relative to the window, or beside
// another element with AttachTo. Each, as it goes with the focus in it,
// gives the focus back to the element that had it as it came.
func coreOverlay(c *context, fn func()) {
	saved := c.parent
	o := c.overlayRoot()
	c.parent = o
	last := o.last
	// What fn builds reads in the direction of where it was asked for.
	dir := o.dir
	o.dir = saved.dir &^ dirSet
	fn()
	o.dir = dir
	c.parent = saved
	first := o.first
	if last != nil {
		first = last.next
	}
	if c.inert {
		return
	}
	for e := first; e != nil; e = e.next {
		c.rt.openOverlay(e)
	}
}

// Modal shows a dialog built by fn over a dimmed window while *open is
// true; clicking outside it or pressing Escape sets *open to false, which
// Dismissed reports.
func coreModal(c *context, open *bool, fn func()) *node {
	if !*open {
		return nil
	}
	t := c.theme
	return coreDialogBase(c, open, func(back, panel *node) {
		back.Background(RGBA(0, 0, 0, 0.4))
		panel.Padding(t.Space(5)).Gap(t.Space(3)).Radius(t.Space(2.5)).Background(t.Background).MaxWidth(c.w - t.Space(10)).MaxHeight(c.h - t.Space(10))
		panel.Shadow(0, 10, 30, 0, RGBA(0, 0, 0, 0.3))
		fn()
	})
}

// Popover shows fn's elements in a panel below anchor while *open is
// true; clicking outside it or pressing Escape sets *open to false, which
// Dismissed reports.
func corePopover(c *context, anchor *node, open *bool, fn func()) *node {
	if !*open {
		return nil
	}
	return corePopoverBase(c, anchor, open, func(panel *node) {
		panel.MinWidth(anchor.Bounds().W)
		stylePanel(c, panel)
		fn()
	})
}

// stylePanel gives the panel of a popup the theme's look.
func stylePanel(c *context, panel *node) {
	t := c.theme
	panel.Margin(t.Space(1), 0, 0, 0).Padding(t.Space(1)).Radius(t.Radius+2).Background(t.Background).Border(1, t.Border)
	panel.Shadow(0, 6, 20, 0, RGBA(0, 0, 0, 0.18))
}

// Select creates a drop-down choosing one of options into *selected.
func coreSelect(c *context, selected *string, options []string) *node {
	t := c.theme
	sel := coreSelectBase(c, selected)
	b := sel.Trigger
	styleButton(c, b, false)
	b.Justify(SpaceBetween).MinWidth(t.Space(35))
	b.Children(func() {
		coreText(c, *selected).SingleLine()
		chevron(c)
	})
	sel.Popup(func(panel *node) {
		stylePanel(c, panel)
		// A long list scrolls in the room the window has below the
		// trigger (at most 20 rows' worth), instead of running off the
		// window; the highlighted option scrolls into view (Item).
		r := b.Bounds()
		room := c.h - (r.Y + r.H) - t.Space(4)
		maxH := max(t.Space(30), min(t.Space(80), room))
		list := coreScroll(c).MaxHeight(maxH)
		list.Children(func() {
			coreColumn(c).Children(func() {
				for _, opt := range options {
					item := sel.Item(opt).Padding(t.Space(1.5), t.Space(2.5)).Radius(t.Radius)
					switch {
					case item.Highlighted():
						item.Background(t.Accent).TextColor(t.AccentText)
					case opt == *selected:
						item.Background(t.Surface)
					}
					item.Children(func() { coreText(c, opt).SingleLine() })
				}
			})
		})
	})
	return b
}

// chevron draws the arrow of a button opening something below it.
func chevron(c *context) {
	t := c.theme
	coreBox(c).Size(t.Space(2.5), t.Space(2.5)).Shrink(0).Draw(func(p *Painter, r Rect) {
		var path Path
		path.MoveTo(r.X+r.W*0.1, r.Y+r.H*0.3).LineTo(r.X+r.W*0.5, r.Y+r.H*0.7).LineTo(r.X+r.W*0.9, r.Y+r.H*0.3)
		p.StrokePath(&path, 1.5, t.TextMuted)
	})
}

// MenuButton creates a button showing label and an arrow, which opens a
// menu below it that build fills with items, as the pointer goes down on
// it or for Enter, Space or Down while it has the focus (Element.Menu):
//
//	ui.MenuButton(c, "Sort by", func(m *ui.Menu) {
//		for _, by := range []string{"Name", "Date", "Size"} {
//			if m.Item(by).Checked(app.sort == by).Chosen() {
//				app.sort = by
//			}
//		}
//	})
func coreMenuButton(c *context, label string, build func(m *Menu)) *node {
	b := button(c, "", false)
	b.Children(func() {
		if label != "" {
			coreText(c, label).SingleLine()
		}
		chevron(c)
	})
	return b.Menu(build)
}
