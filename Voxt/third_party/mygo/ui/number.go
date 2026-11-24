package ui

import (
	"math"
	"strconv"
	"strings"
)

// NumberInput creates a text input editing *value as a number between lo
// and hi, which the arrows Up and Down, and the buttons beside it, step
// by step. What is typed applies as soon as it is a number in range, and
// shows rounded to the decimals of step once the input loses the focus.
// Changed reports a new value.
//
//	ui.NumberInput(c, &app.copies, 1, 99, 1)
func coreNumberInput(c *context, value *float64, lo, hi, step float64) *node {
	t := c.theme
	row := coreRow(c).Gap(t.Space(1)).Shrink(0).AlignItems(Center)
	row.widget = "NumberInput"
	decimals := 0
	if s := strconv.FormatFloat(step, 'f', -1, 64); strings.Contains(s, ".") {
		decimals = len(s) - strings.IndexByte(s, '.') - 1
	}
	format := func(v float64) string { return strconv.FormatFloat(v, 'f', decimals, 64) }
	text := coreLocal(row, "text", func() string { return format(*value) })
	set := func(v float64) {
		v = math.Max(lo, math.Min(hi, v))
		if decimals >= 0 {
			p := math.Pow(10, float64(decimals))
			v = math.Round(v*p) / p
		}
		if v != *value {
			*value = v
			row.st.markChanged()
			c.rt.consumed = true
		}
	}
	row.Children(func() {
		in := coreTextInput(c, text).Width(t.Space(20))
		in.widget = "NumberInput"
		in.afterInput(func() {
			if in.Changed() {
				if v, err := strconv.ParseFloat(strings.TrimSpace(*text), 64); err == nil && v >= lo && v <= hi {
					set(v)
				}
			}
			if in.Shortcut(0, KeyUp) {
				set(*value + step)
				*text = format(*value)
			}
			if in.Shortcut(0, KeyDown) {
				set(*value - step)
				*text = format(*value)
			}
			if !in.Focused() {
				// What the app set, or what was typed, shown in full.
				*text = format(*value)
			}
		})
		in.hasRange, in.accRange, in.accStep = true, [3]float64{lo, hi, *value}, step
		for _, b := range []struct {
			label string
			delta float64
		}{{"−", -step}, {"+", step}} {
			btn := coreButton(c, b.label).Padding(t.Space(1), t.Space(2)).Label(map[bool]string{true: "Increase", false: "Decrease"}[b.delta > 0])
			btn.Disabled(b.delta < 0 && *value <= lo || b.delta > 0 && *value >= hi)
			btn.afterInput(func() {
				if btn.Clicked() {
					set(*value + b.delta)
					*text = format(*value)
				}
			})
			btn.TextColor(t.Text)
		}
	})
	return row
}
