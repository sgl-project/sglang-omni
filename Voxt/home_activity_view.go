package main

// Home's activity calendar: the days Voxt gave a result, as squares by
// week (a column a week, Sunday on top) over half a year, shaded by how
// many results each day had. The arrows page through earlier halves.

import (
	"strconv"
	"time"

	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/icons"
)

const (
	activityCell    = 10 // the legend's squares
	activityCellMax = 12 // the calendar's, at most
	activityGap     = 4  // a third of a square: each day reads apart
	activityRadius  = 3
	// activityEmpty is a day without results: the accent, faint, so the
	// calendar keeps to the theme's color rather than a gray of its own.
	activityEmpty = 0.1
	activityDayW  = 22 // the weekdays' column, as wide as an arrow
	activityRowH  = 22 // the legend's row, as tall as the months' (an arrow)
	// A week's days light up in activityLight, each week activityStagger
	// after the one before it: all of them within half a second.
	activityLight   = 180 * time.Millisecond
	activityStagger = 12 * time.Millisecond
)

func homeActivitySection(c *ui.Context, state *homeViewState, actions homeActions) {
	t := c.Theme()
	act := state.Activity
	if act == nil {
		return // the log is on its way
	}
	sectionTitle(c, i18n.T("Activity"))
	if !state.KeepHistory {
		caption(c, i18n.T("History is off, so nothing is counted."))
		return
	}
	fillCard(c.Key("home-activity")).Children(func() {
		// The numbers sit on the calendar's middle line: the legend over the
		// squares is as tall as the months under them.
		ui.Row(c.Key("activity-body")).Gap(px(t, sp6)).AlignItems(ui.Center).Children(func() {
			ui.Column(c.Key("activity-calendar")).Gap(px(t, activityGap)).Grow(1).Basis(0).MinWidth(0).Children(func() {
				homeActivityCalendar(c, act, actions)
			})
			homeActivityNumbers(c, act.Stats)
		})
	})
}

// homeActivityCalendar is the calendar under its legend.
func homeActivityCalendar(c *ui.Context, act *homeActivityViewState, actions homeActions) {
	t := c.Theme()
	k := tok(t)
	// Compact, as legends are: the squares almost touch, the words close by.
	// Lifted a little off the squares; the translation leaves the layout,
	// and so the middle line, as it is.
	ui.Row(c.Key("activity-legend")).Height(px(t, activityRowH)).Gap(px(t, 6)).AlignItems(ui.Center).Translate(0, -px(t, 3)).Children(func() {
		ui.Text(c, i18n.T("Less often")).FontSize(px(t, fsMicro)).TextColor(k.Text2).SingleLine()
		ui.Row(c.Key("legend-squares")).Gap(px(t, 3)).Children(func() {
			for level := 1; level <= act.Levels; level++ {
				activitySquare(c.Key("legend-"+strconv.Itoa(level)), level, false, 1).Size(px(t, activityCell), px(t, activityCell))
			}
		})
		ui.Text(c, i18n.T("More often")).FontSize(px(t, fsMicro)).TextColor(k.Text2).SingleLine()
	})
	homeActivityGrid(c, act, actions)
}

// homeActivityGridW is the widest the calendar grows: its weekdays'
// column and its weeks at activityCellMax.
func homeActivityGridW(c *ui.Context, act *homeActivityViewState) float32 {
	t := c.Theme()
	return px(t, activityDayW) + float32(len(act.Weeks))*(px(t, activityCellMax)+px(t, activityGap))
}

// homeActivityNumbers is the four usage numbers beside the calendar,
// after a hairline.
func homeActivityNumbers(c *ui.Context, stats []homeActivityStatViewState) {
	t := c.Theme()
	k := tok(t)
	ui.Box(c.Key("numbers-line")).Width(1).AlignSelf(ui.Stretch).Background(k.Line).Shrink(0)
	ui.Column(c.Key("activity-numbers")).Gap(px(t, sp4)).Justify(ui.Center).Padding(0, px(t, sp2)).Shrink(0).Children(func() {
		for index, stat := range stats {
			ui.Row(c.Key(index)).Gap(px(t, 8)).AlignItems(ui.End).Label(stat.Value + " " + stat.Label).Children(func() {
				ui.Text(c, stat.Value).FontSize(px(t, fsTitle)).FontWeight(600).TextColor(k.Text).FontFeatures("tnum").SingleLine()
				ui.Text(c, stat.Label).FontSize(px(t, fsCaption)).TextColor(k.Text2).SingleLine().Margin(0, 0, px(t, 3), 0)
			})
		}
	})
}

// homeActivityGrid is a row a weekday (its name, then that day of each
// week), and the months under the weeks they start in. The squares share
// the card's width, up to activityCellMax each. When it appears, the days
// with results light up a week after another, from left to right.
func homeActivityGrid(c *ui.Context, act *homeActivityViewState, actions homeActions) {
	t := c.Theme()
	k := tok(t)
	gap := px(t, activityGap)
	grid := ui.Column(c.Key("activity-grid")).Gap(gap).MaxWidth(homeActivityGridW(c, act))
	tl := grid.Timeline("light", ui.TimelineOptions{})
	lit := make([]float32, len(act.Weeks))
	for week := range lit {
		lit[week] = tl.Tween(ui.Tween{From: 0, To: 1, Duration: activityLight, Ease: ui.EaseOut}, ui.AtTime(time.Duration(week)*activityStagger))
	}
	grid.Children(func() {
		for weekday, name := range act.Weekdays {
			ui.Row(c.Key("weekday-" + strconv.Itoa(weekday))).Gap(gap).AlignItems(ui.Center).Children(func() {
				text := ""
				if weekday%2 == 1 { // every other day, as calendars do
					text = name
				}
				ui.Text(c, text).FontSize(px(t, fsMicro)).TextColor(k.Text3).SingleLine().Width(px(t, activityDayW)).Shrink(0)
				for week, days := range act.Weeks {
					day := days[weekday]
					square := activitySquare(c.Key(strconv.Itoa(week)), day.Level, day.Future, lit[week]).Grow(1).Basis(0).AspectRatio(1)
					if !day.Future {
						square.Label(day.Date + " " + day.Count)
						activityTip(c, square, day)
					}
				}
			})
		}
		// The months, each under the week its first day is in, between the
		// arrows that page the calendar.
		ui.Row(c.Key("activity-months")).Gap(gap).AlignItems(ui.Center).Children(func() {
			if activityArrow(c.Key("activity-back"), icons.AltArrowLeft, i18n.T("Earlier weeks"), act.CanBack) {
				actions.activityBack()
			}
			labels := map[int]string{}
			last := -3
			for _, month := range act.Months {
				if month.Week-last >= 3 { // not over the last label
					labels[month.Week], last = month.Label, month.Week
				}
			}
			for week := range act.Weeks {
				ui.Box(c.Key(strconv.Itoa(week))).Grow(1).Basis(0).Children(func() {
					if label := labels[week]; label != "" {
						ui.Text(c, label).FontSize(px(t, fsMicro)).TextColor(k.Text3).SingleLine().Width(px(t, 40)).Absolute().Left(0).Top(px(t, -7))
					}
				})
			}
			if activityArrow(c.Key("activity-forward"), icons.AltArrowRight, i18n.T("Later weeks"), act.CanForward) {
				actions.activityForward()
			}
		})
	})
}

// activitySquare is a day's square at level (0: none), sized by its
// caller; a future day's is an outline. A day with results shows its
// color lit fraction (0 to 1) over an empty square's.
func activitySquare(c *ui.Context, level int, future bool, lit float32) ui.Element {
	t := c.Theme()
	k := tok(t)
	square := ui.Box(c).Radius(px(t, activityRadius)).Shrink(0)
	switch {
	case future:
		square.Border(1, k.Accent.Alpha(2*activityEmpty))
	case level == 0:
		square.Background(k.Accent.Alpha(activityEmpty))
	default:
		square.Background(k.Accent.Alpha(activityEmpty)).Children(func() {
			ui.Box(c).Fill().Radius(px(t, activityRadius)).Background(k.Accent.Alpha(0.25 + 0.25*float32(level-1))).Opacity(lit)
		})
	}
	return square
}

// activityTip is a square's tooltip, in the theme's tooltip colors: its
// day, softer, over its results.
func activityTip(c *ui.Context, square ui.Element, day homeActivityDayViewState) {
	t := c.Theme()
	ui.TooltipBase(c, square, func(tip ui.Element) {
		tip.Padding(px(t, 6), px(t, 10)).Radius(px(t, 6)).Background(t.Inverse).Gap(px(t, 2)).
			Shadow(0, 2, 8, 0, ui.RGBA(0, 0, 0, 0.2))
		tip.AttachTo(square, ui.AnchorTop, ui.AnchorBottom).Margin(0, 0, px(t, 6), 0)
		tip.Children(func() {
			ui.Text(c, day.Date).FontSize(px(t, fsCaption)).TextColor(t.InverseText.Alpha(0.6)).SingleLine()
			ui.Text(c, day.Count).FontSize(px(t, fsCaption)).TextColor(t.InverseText).SingleLine()
		})
	})
}

// activityArrow is a small icon button that pages the calendar. With no
// page that way it leaves its room empty, so the months stay in place.
func activityArrow(c *ui.Context, ic icons.Icon, label string, enabled bool) bool {
	t := c.Theme()
	k := tok(t)
	if !enabled {
		ui.Box(c).Size(px(t, 22), px(t, 22)).Shrink(0)
		return false
	}
	return ghostBase(c).Size(px(t, 22), px(t, 22)).Shrink(0).Center().Label(label).Tooltip(label).Children(func() {
		ui.Icon(c, ic.Linear).Size(px(t, 14), px(t, 14)).TextColor(k.Text2)
	}).Clicked()
}
