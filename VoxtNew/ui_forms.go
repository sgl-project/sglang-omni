package main

// The kit's forms and dialogs (ui_kit.go): segmented controls, the number
// input, popovers with a footer, and the dialog frame.

import (
	"math"
	"strconv"
	"strings"

	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/i18n"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/icons"
)

// segmentedFit is ui.Segmented as wide as its segments, for a form's
// column: there a plain Segmented stretches across, its segments keeping
// their width and leaving the rest of the track empty.
func segmentedFit(c *ui.Context, selected *int, labels ...string) ui.Element {
	return ui.Segmented(c, selected, labels...).AlignSelf(ui.Start)
}

// numOpts configure numberInput (set before it is built: Disabled also
// disables its steps). Width is the whole control's.
type numOpts struct {
	Label, Tooltip string
	Width          float32 // of the whole control; 0 for the default
	Disabled       bool
	Grow           bool // take the width the row leaves
}

// numberInput is a field for *value in [lo, hi] by step, with a − and +
// as one stepper beside it, the field's height and in Solar's icons (the
// kit's replacement for ui.NumberInput, whose small text buttons do not
// line up with its field). Up and Down step while it has the focus. It
// reports whether typing, the arrows or a step changed the value.
func numberInput(c *ui.Context, value *float64, lo, hi, step float64, o numOpts) bool {
	t := c.Theme()
	k := tok(t)
	decimals := 0
	if s := strconv.FormatFloat(step, 'f', -1, 64); strings.Contains(s, ".") {
		decimals = len(s) - strings.IndexByte(s, '.') - 1
	}
	format := func(v float64) string { return strconv.FormatFloat(v, 'f', decimals, 64) }
	changed := false
	set := func(v float64) {
		v = math.Max(lo, math.Min(hi, v))
		p := math.Pow(10, float64(decimals))
		v = math.Round(v*p) / p
		if v != *value {
			*value, changed = v, true
		}
	}
	row := ui.Row(c).Gap(px(t, 4)).AlignItems(ui.Stretch).Shrink(0)
	text := ui.Local(row, "text", func() string { return format(*value) })
	row.Children(func() {
		// Width is the whole control's, as ui.NumberInput's was: the field
		// takes what the stepper leaves; it grows when the control does.
		in := ui.TextInput(c.Key("field"), text).Label(o.Label).Disabled(o.Disabled).Grow(1).Shrink(1).MinWidth(px(t, 56))
		if o.Width > 0 {
			in.Width(max(o.Width-px(t, 4+28*2+3), px(t, 56)))
		} else {
			in.Width(px(t, 80))
		}
		if in.Changed() {
			if v, err := strconv.ParseFloat(strings.TrimSpace(*text), 64); err == nil && v >= lo && v <= hi {
				set(v)
			}
		}
		if in.Shortcut(0, ui.KeyUp) {
			set(*value + step)
			*text = format(*value)
		}
		if in.Shortcut(0, ui.KeyDown) {
			set(*value - step)
			*text = format(*value)
		}
		if !in.Focused() {
			*text = format(*value) // what the app set, or what was typed, in full
		}
		ui.Row(c.Key("steps")).AlignItems(ui.Stretch).Radius(px(t, radiusControl)).Border(1, k.Line).Shrink(0).Children(func() {
			step1 := func(key string, ic icons.Icon, label string, delta float64, off bool) {
				b := ui.ButtonBase(c.Key(key)).Width(px(t, 28)).Center().Label(label).Tooltip(label).Disabled(o.Disabled || off)
				switch {
				case b.Pressed():
					b.Background(k.Pressed)
				case b.Hovered():
					b.Background(k.Hover)
				}
				b.Children(func() {
					col := k.Text2
					if o.Disabled || off {
						col = k.Text3
					}
					ui.Icon(c, ic.Linear).Size(px(t, 14), px(t, 14)).TextColor(col)
				})
				if b.Clicked() {
					set(*value + delta)
					*text = format(*value)
				}
			}
			step1("minus", icons.Minus, i18n.T("Decrease"), -step, *value <= lo)
			ui.Box(c).Width(1).Background(k.Line)
			step1("plus", icons.Add, i18n.T("Increase"), step, *value >= hi)
		})
	})
	if o.Tooltip != "" {
		row.Tooltip(o.Tooltip)
	}
	if o.Grow {
		row.Grow(1).MinWidth(0)
	}
	return changed
}

// popoverWithFooter is a popover's content: body in a scroll as tall as
// the window allows, and footer (Reset…) pinned under it, past a hairline,
// so it shows however long the body is.
func popoverWithFooter(c *ui.Context, key string, footer, body func()) {
	t := c.Theme()
	_, h := c.Size()
	maxH := max(px(t, 180), min(px(t, 420), h*0.4)) // room above or below the button it opens from
	ui.Column(c.Key(key + "-pop-frame")).Children(func() {
		ui.Scroll(c.Key(key + "-pop-scroll")).MaxHeight(maxH).Children(body)
		ui.Divider(c)
		ui.Column(c.Key(key + "-pop-footer")).Padding(px(t, 10)).AlignItems(ui.Stretch).Children(footer)
	})
}

// segmentedSome is ui.Segmented (as wide as its segments, as
// segmentedFit) whose segments may be off: off[i] true greys segment i
// out, and tips[i] ("" for none) says why, as its tooltip. Its Changed
// reports a new choice.
func segmentedSome(c *ui.Context, selected *int, labels []string, off []bool, tips []string) ui.Element {
	t := c.Theme()
	parts := ui.SegmentedBase(c, selected, len(labels))
	tr := parts.Track.AlignSelf(ui.Start).Shrink(0).AlignItems(ui.Stretch).Padding(t.Space(0.5)).Gap(t.Space(0.5)).
		Radius(t.Radius).Background(t.Surface).Border(1, t.Border)
	tr.Children(func() {
		for i, label := range labels {
			s := parts.Segment(i).Padding(t.Space(1), t.Space(3)).Radius(max(t.Radius-t.Space(0.5), 0)).Center().Label(label)
			dis := i < len(off) && off[i]
			if dis {
				s.Disabled(true)
			}
			if i < len(tips) && tips[i] != "" {
				s.Tooltip(tips[i])
			}
			switch {
			case i == *selected:
				face := t.Background
				if t.Dark {
					face = t.SurfacePressed
				}
				s.Background(face).Shadow(0, 1, 2, 0, ui.RGBA(0, 0, 0, 0.12))
			case dis:
			case s.Pressed():
				s.Background(t.SurfacePressed)
			case s.Hovered():
				s.Background(t.SurfaceHover)
			}
			s.Children(func() {
				tx := ui.Text(c, label).SingleLine()
				if dis {
					tx.TextColor(tok(t).Text3)
				}
			})
		}
	})
	return tr
}

// dialogFrame is a dialog in the look of the Settings dialog: a panel
// over the dimmed page, at most width wide and as tall as its content up
// to the window's height less margins, with a title row (title, a
// caption under it, "" for none, and ×), body in a scroll, and footer
// (the dialog's actions: one primaryButton at most) pinned under a
// hairline. Escape, a click beside it or × close it (*open false).
func dialogFrame(c *ui.Context, key string, open *bool, title, sub string, width float32, body, footer func()) ui.Element {
	t := c.Theme()
	k := tok(t)
	w, h := c.Size()
	margin := px(t, sp6)
	return ui.DialogBase(c.Key(key), open, func(backdrop, panel ui.Element) {
		backdrop.Background(ui.RGBA(0, 0, 0, 0.32))
		panel.Width(min(w-2*margin, width)).MaxHeight(h-2*margin).Radius(px(t, radiusCard)).Background(k.Bg).Clip().
			Shadow(0, 16, 48, 0, ui.RGBA(0, 0, 0, 0.28)).Label(title)
		ui.Column(c).Grow(1).MinHeight(0).Children(func() {
			ui.Row(c).Shrink(0).AlignItems(ui.Center).Padding(px(t, sp4), px(t, sp3), px(t, sp2), px(t, sp6)).Gap(px(t, sp2)).Children(func() {
				ui.Column(c).Grow(1).MinWidth(0).Gap(px(t, 2)).Children(func() {
					sectionHeading(c, title)
					if sub != "" {
						truncText(c, sub).FontSize(px(t, fsCaption)).TextColor(k.Text2)
					}
				})
				if iconButton(c.Key(key+"-close"), icons.Close, i18n.T("Close")).Clicked() {
					*open = false
				}
			})
			ui.Scroll(c.Key(key + "-body")).Grow(1).Shrink(1).MinHeight(0).Children(func() {
				ui.Column(c).Padding(px(t, sp2), px(t, sp6), px(t, sp4)).Gap(px(t, sp2)).Children(body)
			})
			ui.Box(c).Height(1).Background(k.Line).Shrink(0)
			ui.Row(c.Key(key+"-footer")).Shrink(0).AlignItems(ui.Center).Padding(px(t, sp3), px(t, sp6)).Gap(px(t, sp2)).Children(footer)
		})
	})
}
