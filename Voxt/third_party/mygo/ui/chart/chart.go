// Package chart builds charts in native UI without a look of their own,
// as headless libraries do on the web: scales that map data onto a plot
// and pick round ticks, a plot with axes around it, marks that are
// elements you style as any other, paths of lines, areas and pie slices
// for Draw functions, stacking, and a selection that the pointer and the
// keyboard move.
//
//	x := chart.NewBand(months...).Padding(0.3, 0.15)
//	y := chart.NewLinear(0, 120).Nice(4)
//	chart.Plot(c, chart.Axes{Left: y.Ticks(4), Bottom: x.Ticks()}, func(area ui.Element) {
//		for i, v := range sales {
//			chart.Rect(c, x.Band(i), y.Span(0, v)).Background(t.Accent).Radius(4)
//		}
//	}).Height(200).Gap(6)
//
// Positions along a plot are fractions of it, from 0 to 1: along x from its
// left edge, and along y from its bottom up, the way values grow. Marks
// place themselves in percentages of the plot, so they need no layout to
// be built, follow the plot as it resizes, and move smoothly with a
// Transition as the data changes; Draw functions get the plot's box, and
// X and Y turn fractions into points in it.
//
// See docs/ui/chart.md.
package chart

import (
	"slices"

	"github.com/egoist/mygo/ui"
)

// Side is a side of a plot, where an axis goes.
type Side int

const (
	Left Side = iota
	Right
	Top
	Bottom
)

// Axes are the ticks along the sides of a plot, whose labels go beside it:
// a scale's Ticks, or ticks of your own.
type Axes struct {
	Left, Right, Top, Bottom []Tick
	// Label builds the label of a tick on a side, centered on the tick;
	// nil shows its Label as text. Labels sit in boxes of no width or
	// height, so keep their text on one line with NoWrap.
	Label func(side Side, t Tick)
}

// Plot builds a plot with the axes around it: a grid whose middle cell is
// the plot's area, which build gets to style, as with a background or a
// Draw function painting grid lines, and to add marks to. Give the plot a
// height, and Gap to keep the labels away from the area:
//
//	chart.Plot(c, chart.Axes{Left: y.Ticks(5)}, func(area ui.Element) {
//		area.Draw(func(p *ui.Painter, r ui.Rect) {
//			for _, tick := range y.Ticks(5) {
//				p.Line(r.X, chart.Y(r, tick.At), r.X+r.W, chart.Y(r, tick.At), 1, t.Border)
//			}
//		})
//	}).Height(160).Gap(8)
//
// Name the plot for assistive technology with Label, and describe what it
// shows with Description.
func Plot(c *ui.Context, axes Axes, build func(area ui.Element)) ui.Element {
	label := axes.Label
	if label == nil {
		label = func(_ Side, t Tick) { ui.Text(c, t.Label).NoWrap() }
	}
	cols := []ui.Track{ui.Fr(1)}
	rows := []ui.Track{ui.Fr(1)}
	col, row := 1, 1
	if len(axes.Left) > 0 {
		cols = append([]ui.Track{ui.FitContent()}, cols...)
		col = 2
	}
	if len(axes.Right) > 0 {
		cols = append(cols, ui.FitContent())
	}
	if len(axes.Top) > 0 {
		rows = append([]ui.Track{ui.FitContent()}, rows...)
		row = 2
	}
	if len(axes.Bottom) > 0 {
		rows = append(rows, ui.FitContent())
	}
	return ui.Grid(c).ColumnTracks(cols...).RowTracks(rows...).Children(func() {
		area := ui.Box(c).ColumnStart(col).RowStart(row)
		area.Children(func() { build(area) })
		if len(axes.Left) > 0 {
			axis(c, Left, axes.Left, label).ColumnStart(1).RowStart(row)
		}
		if len(axes.Right) > 0 {
			axis(c, Right, axes.Right, label).ColumnStart(col + 1).RowStart(row)
		}
		if len(axes.Top) > 0 {
			axis(c, Top, axes.Top, label).ColumnStart(col).RowStart(1)
		}
		if len(axes.Bottom) > 0 {
			axis(c, Bottom, axes.Bottom, label).ColumnStart(col).RowStart(row + 1)
		}
	})
}

// axis lays out the labels of ticks along a side of the plot: each in a
// box of no width (or height) at its tick, which centers it there, with
// spaces between them that grow by the fractions of the plot between the
// ticks. The labels stay in the layout, so the axis is as wide (or as
// high) as they need.
func axis(c *ui.Context, side Side, ticks []Tick, label func(Side, Tick)) ui.Element {
	vertical := side == Left || side == Right
	// Where each tick is from the axis's start: the left, or the top.
	pos := func(t Tick) float32 {
		if vertical {
			return 1 - t.At
		}
		return t.At
	}
	sorted := slices.Clone(ticks)
	slices.SortStableFunc(sorted, func(a, b Tick) int {
		switch pa, pb := pos(a), pos(b); {
		case pa < pb:
			return -1
		case pa > pb:
			return 1
		}
		return 0
	})
	const eps = 1e-4
	var e ui.Element
	if vertical {
		e = ui.Column(c)
	} else {
		e = ui.Row(c)
	}
	return e.Children(func() {
		prev := float32(0)
		for _, t := range sorted {
			p := pos(t)
			if p < -eps || p > 1+eps {
				continue
			}
			p = max(prev, min(1, p))
			ui.Box(c).Basis(0).Grow(p - prev)
			var slot ui.Element
			switch side {
			case Left:
				slot = ui.Row(c).Height(0).AlignItems(ui.Center).Justify(ui.End)
			case Right:
				slot = ui.Row(c).Height(0).AlignItems(ui.Center)
			case Top:
				slot = ui.Column(c).Width(0).AlignItems(ui.Center).Justify(ui.End)
			default:
				slot = ui.Column(c).Width(0).AlignItems(ui.Center)
			}
			slot.Children(func() { label(side, t) })
			prev = p
		}
		ui.Box(c).Basis(0).Grow(1 - prev)
	})
}

// percent turns a fraction into a percentage.
func percent(f float32) float32 { return 100 * f }

// Rect places a mark over the stretch x along the plot and y up it, as a
// bar, a band of a range or a cell of a heat map: an element in the plot's
// area, which you style as any other, and name for assistive technology.
// Build it in the function of Plot.
//
//	chart.Rect(c, x.Band(i), y.Span(0, v)).Background(t.Accent).Radius(4).Tooltip(label)
func Rect(c *ui.Context, x, y Span) ui.Element {
	x0, x1 := min(x.Lo, x.Hi), max(x.Lo, x.Hi)
	y0, y1 := min(y.Lo, y.Hi), max(y.Lo, y.Hi)
	return ui.Box(c).Absolute().
		LeftPercent(percent(x0)).WidthPercent(percent(x1 - x0)).
		TopPercent(percent(1 - y1)).HeightPercent(percent(y1 - y0))
}

// Point places a mark of w by h DIPs centered on the point at x along the
// plot and y up it, as a dot of a scatter plot or the point a selection is
// on.
//
//	chart.Point(c, x.At(v.Weight), y.At(v.Height), 8, 8).Radius(4).Background(t.Accent)
func Point(c *ui.Context, x, y, w, h float32) ui.Element {
	return ui.Box(c).Absolute().LeftPercent(percent(x)).TopPercent(percent(1-y)).
		Size(w, h).Margin(-h/2, 0, 0, -w/2)
}

// RuleX places a vertical line of width DIPs across the plot at x along
// it, as a threshold or the selection of a line chart; RuleY a horizontal
// one at y up it. Color them with Background.
func RuleX(c *ui.Context, x, width float32) ui.Element {
	return ui.Box(c).Absolute().LeftPercent(percent(x)).Top(0).Bottom(0).
		Width(width).Margin(0, 0, 0, -width/2)
}

// RuleY places a horizontal line of height DIPs across the plot at y up
// it.
func RuleY(c *ui.Context, y, height float32) ui.Element {
	return ui.Box(c).Absolute().TopPercent(percent(1-y)).Left(0).Right(0).
		Height(height).Margin(-height/2, 0, 0, 0)
}

// Anchor builds what build adds at the point at x along the plot and y up
// it, of any size: as a label over a bar, or a tooltip by the selection.
// h and v place it: Center centers it on the point, Start puts its left
// (h) or top (v) edge there, and End its right or bottom edge, so that
// End, End shows it above and to the left of the point.
//
//	chart.Anchor(c, x.At(i), y.At(v), ui.Center, ui.End, func() {
//		ui.Text(c, label).NoWrap().Margin(0, 0, 4, 0)
//	})
func Anchor(c *ui.Context, x, y float32, h, v ui.Align, build func()) ui.Element {
	return ui.Row(c).Absolute().LeftPercent(percent(x)).TopPercent(percent(1-y)).
		Size(0, 0).AlignItems(v).Children(func() {
		ui.Column(c).Width(0).AlignItems(h).Children(build)
	})
}

// selection is the state Select keeps in the area.
type selection struct {
	hovering bool
	x        float32 // where the pointer was
	last     int     // the choice in the last frame
}

// Select makes the plot's area choose one of the points at xs, fractions
// along it in ascending order: *index is the one nearest the pointer while
// it is over the area, and -1, none, once it leaves. The area takes the
// keyboard focus, as one stop of Tab, where the left and right arrows,
// Home and End move the choice, and Escape clears it. Changed reports a
// new choice.
//
// describe tells assistive technology about a point, "March: 42 sales",
// which it reads as the area's value, and as the arrows come to the point.
// Show the choice with marks, as a rule, a point and a label by it:
//
//	if chart.Select(c, area, &app.selected, xs, describe) { ... }
//	if i := app.selected; i >= 0 {
//		chart.RuleX(c, xs[i], 1).Background(t.Border)
//		chart.Point(c, xs[i], ys[i], 8, 8).Radius(4).Background(t.Accent)
//	}
func Select(c *ui.Context, area ui.Element, index *int, xs []float32, describe func(i int) string) bool {
	area.Focusable()
	st := ui.Local(area, "chart.Select", func() selection { return selection{last: -1} })
	old := *index
	n := len(xs)
	if *index >= n {
		*index = -1
	}
	announce := false
	move := func(i int) {
		*index = max(0, min(n-1, i))
		announce = true
	}
	switch {
	case n == 0:
	case area.Shortcut(0, ui.KeyRight):
		if *index < 0 {
			move(0)
		} else {
			move(*index + 1)
		}
	case area.Shortcut(0, ui.KeyLeft):
		if *index < 0 {
			move(n - 1)
		} else {
			move(*index - 1)
		}
	case area.Shortcut(0, ui.KeyHome):
		move(0)
	case area.Shortcut(0, ui.KeyEnd):
		move(n - 1)
	case *index >= 0 && area.Shortcut(0, ui.KeyEscape):
		*index = -1
	}
	// The pointer chooses as it comes and moves, not while it rests where
	// it was as the keys move the choice.
	if x, _, over := area.PointerPosition(); over {
		if w := area.Bounds().W; w > 0 && n > 0 && !announce && (!st.hovering || x != st.x) {
			*index = Nearest(xs, x/w)
		}
		st.hovering, st.x = true, x
	} else if st.hovering {
		st.hovering = false
		*index = -1
	}
	if *index >= 0 && describe != nil {
		desc := describe(*index)
		area.Value(desc)
		if announce && *index != st.last {
			c.Announce(desc)
		}
	}
	st.last = *index
	return *index != old
}
