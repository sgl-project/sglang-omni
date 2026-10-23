package main

import (
	"testing"

	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/VoxtNew/internal/icons"
)

// TestButtonHeights: the kit's text buttons — ink, secondary, ghost, with
// or without an icon, record — stand the same height, so a row of them
// lines up (an empty state's "Load sample" beside "Add the first line").
func TestButtonHeights(t *testing.T) {
	var els []ui.Element
	var hs []float32
	names := []string{"ui.Button", "primary", "primaryIcon", "secondaryIcon", "ghost", "ghostIcon", "dropButton", "record"}
	view := func(c *ui.Context) {
		th := *c.Theme()
		styleTheme(&th)
		c.SetTheme(&th)
		els = els[:0]
		ui.Row(c).Gap(8).AlignItems(ui.Start).Children(func() {
			els = append(els,
				ui.Button(c.Key("a"), "Secondary"),
				primaryButton(c.Key("b"), "Primary"),
				primaryIconButton(c.Key("c"), icons.Play, "Primary icon"),
				secondaryIconButton(c.Key("d"), icons.Refresh, "Secondary icon"),
				ghostButton(c.Key("e"), "Ghost"),
				ghostIconButton(c.Key("f"), icons.Refresh, "Ghost icon"),
				dropButton(c.Key("g"), icons.Refresh, "Drop"),
				recordButton(c.Key("h"), "Record"),
			)
		})
		hs = hs[:0]
		for _, e := range els {
			hs = append(hs, e.Bounds().H) // the last frame's layout
		}
	}
	for _, scale := range []float32{1, 1.25} {
		tt := ui.NewTester(view, 1200, 200)
		tt.SetScale(scale)
		tt.Frame()
		tt.Frame()
		tt.Frame()
		want := hs[0]
		for i, h := range hs {
			if h != want {
				t.Errorf("scale %v: %s is %.1f high, ui.Button %.1f", scale, names[i], h, want)
			}
		}
	}
}

// TestNumberInput: the stepper stands as tall as its field, steps by step
// within the bounds, and typing a number sets it.
func TestNumberInput(t *testing.T) {
	v := 1000.0
	changes := 0
	var stepsH float32
	view := func(c *ui.Context) {
		th := *c.Theme()
		styleTheme(&th)
		c.SetTheme(&th)
		ui.Column(c).Padding(20).AlignItems(ui.Start).Children(func() {
			if numberInput(c.Key("keep"), &v, 0, 1010, 10, numOpts{Label: "Keep at most", Width: 180}) {
				changes++
			}
		})
	}
	tt := ui.NewTester(view, 400, 200)
	tt.Frame()
	tt.Frame()
	if err := tt.Click("Increase"); err != nil {
		t.Fatal(err)
	}
	tt.Frame()
	if v != 1010 || changes != 1 {
		t.Errorf("after +: %v (%d changes)", v, changes)
	}
	tt.Click("Increase") // at the top: nothing
	tt.Frame()
	if v != 1010 {
		t.Errorf("past the top: %v", v)
	}
	tt.Click("Decrease")
	tt.Frame()
	if v != 1000 {
		t.Errorf("after −: %v", v)
	}
	field, _ := tt.Find("Keep at most")
	minus, _ := tt.Find("Decrease")
	stepsH = minus.H
	if field.H == 0 || stepsH < field.H-2 {
		t.Errorf("stepper %.0f high beside a %.0f field", stepsH, field.H)
	}
}
