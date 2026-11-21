package main

// Home's activity calendar: the days Voxt gave a result, as squares by
// week (a column a week, Sunday on top) over half a year, shaded by how
// many results each day had. The arrows page through earlier halves.

import (
	"strconv"

	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/icons"
)

const (
	activityCell    = 12 // the legend's squares
	activityCellMax = 20 // the calendar's, at most
	activityGap     = 3
	activityDayW    = 16 // the weekdays' column
)

func homeActivitySection(c *ui.Context, state *homeViewState, actions homeActions) {
	t := c.Theme()
	k := tok(t)
	act := state.Activity
	if !state.KeepHistory || act == nil {
		return // Today already says the history is off; the log is on its way
	}
	sectionTitle(c, i18n.T("Activity"))
	fillCard(c.Key("home-activity")).Gap(px(t, sp3)).Children(func() {
		homeActivityGrid(c, act)
		ui.Row(c.Key("activity-foot")).Gap(px(t, sp2)).AlignItems(ui.Center).Children(func() {
			if activityArrow(c.Key("activity-back"), icons.AltArrowLeft, i18n.T("Earlier weeks"), act.CanBack) {
				actions.activityBack()
			}
			if activityArrow(c.Key("activity-forward"), icons.AltArrowRight, i18n.T("Later weeks"), act.CanForward) {
				actions.activityForward()
			}
			ui.Spacer(c)
			ui.Text(c, i18n.T("Less often")).FontSize(px(t, fsMicro)).TextColor(k.Text2).SingleLine()
			for level := 1; level <= act.Levels; level++ {
				activitySquare(c.Key("legend-"+strconv.Itoa(level)), level, false).Size(px(t, activityCell), px(t, activityCell))
			}
			ui.Text(c, i18n.T("More often")).FontSize(px(t, fsMicro)).TextColor(k.Text2).SingleLine()
		})
	})
}

// homeActivityGrid is a row a weekday (its name, then that day of each
// week), and the months under the weeks they start in. The squares share
// the card's width, up to activityCellMax each.
func homeActivityGrid(c *ui.Context, act *homeActivityViewState) {
	t := c.Theme()
	k := tok(t)
	gap := px(t, activityGap)
	weeks := float32(len(act.Weeks))
	ui.Column(c.Key("activity-grid")).Gap(gap).MaxWidth(px(t, activityDayW) + weeks*(px(t, activityCellMax)+gap)).Children(func() {
		for weekday, name := range act.Weekdays {
			ui.Row(c.Key("weekday-" + strconv.Itoa(weekday))).Gap(gap).AlignItems(ui.Center).Children(func() {
				text := ""
				if weekday%2 == 1 { // every other day, as calendars do
					text = name
				}
				ui.Text(c, text).FontSize(px(t, 9)).TextColor(k.Text3).SingleLine().Width(px(t, activityDayW)).Shrink(0)
				for week, days := range act.Weeks {
					day := days[weekday]
					square := activitySquare(c.Key(strconv.Itoa(week)), day.Level, day.Future).Grow(1).Basis(0).AspectRatio(1)
					if !day.Future {
						square.Label(day.Tip).Tooltip(day.Tip)
					}
				}
			})
		}
		// The months, each over the week its first day is in.
		ui.Row(c.Key("activity-months")).Gap(gap).Height(px(t, 14)).Children(func() {
			ui.Box(c).Width(px(t, activityDayW)).Shrink(0)
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
						ui.Text(c, label).FontSize(px(t, 9)).TextColor(k.Text3).SingleLine().Width(px(t, 40)).Absolute().Left(0).Top(0)
					}
				})
			}
		})
	})
}

// activitySquare is a day's square at level (0: none), sized by its
// caller; a future day's is an outline.
func activitySquare(c *ui.Context, level int, future bool) ui.Element {
	t := c.Theme()
	k := tok(t)
	square := ui.Box(c).Radius(px(t, 3)).Shrink(0)
	switch {
	case future:
		square.Border(1, k.Line)
	case level == 0:
		square.Background(k.Hover)
	default:
		square.Background(k.Accent.Alpha(0.25 + 0.25*float32(level-1)))
	}
	return square
}

// activityArrow is a small icon button that pages the calendar.
func activityArrow(c *ui.Context, ic icons.Icon, label string, enabled bool) bool {
	t := c.Theme()
	k := tok(t)
	b := ghostBase(c).Size(px(t, 22), px(t, 22)).Center().Label(label).Tooltip(label).Disabled(!enabled)
	color := k.Text2
	if !enabled {
		color = k.Text3.Alpha(0.5)
	}
	return b.Children(func() {
		ui.Icon(c, ic.Linear).Size(px(t, 14), px(t, 14)).TextColor(color)
	}).Clicked() && enabled
}
