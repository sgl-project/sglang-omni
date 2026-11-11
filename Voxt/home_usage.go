package main

// Home's usage: the finished results of the last weeks as a square a
// day, shaded by its characters, and the numbers over it (today, 30
// days, the speaking rate, the recognition time).

import (
	"context"
	"strings"
	"time"

	"github.com/egoist/mygo/ui"
	"github.com/sgl-project/sglang-omni/Voxt/internal/i18n"
	"github.com/sgl-project/sglang-omni/Voxt/internal/store"
)

// usageWeeks is how many weeks Home's usage reads, and shows at most.
const usageWeeks = 26

// usageStats are the finished results of the last usageWeeks weeks.
type usageStats struct {
	charsByDay map[string]int // keyed by the local date, 2006-01-02
	audioMS    int64          // of the results that say how long they recorded
	audioChars int            // their characters
	asrMS      int64          // of the results that say how long recognition took
	asrCount   int
	lastModel  string // the newest result's speech model id
}

const dayKey = "2006-01-02"

// readUsage reads the finished results of the last usageWeeks weeks.
func readUsage(ctx context.Context, h store.History, now time.Time) (usageStats, error) {
	y, m, d := now.Date()
	since := time.Date(y, m, d, 0, 0, 0, 0, now.Location()).AddDate(0, 0, -7*usageWeeks)
	usage := usageStats{charsByDay: map[string]int{}}
	const page = 500
	for off := 0; ; off += page {
		entries, err := h.List(ctx, store.HistoryQuery{Limit: page, Offset: off})
		if err != nil {
			return usage, err
		}
		for _, e := range entries {
			if e.CreatedAt.Before(since) {
				return usage, nil
			}
			if e.Status != store.StatusDone || strings.TrimSpace(e.Result) == "" {
				continue
			}
			chars := countChars(e.Result)
			usage.charsByDay[e.CreatedAt.In(now.Location()).Format(dayKey)] += chars
			if e.AudioMS > 0 {
				usage.audioMS += e.AudioMS
				usage.audioChars += chars
			}
			if e.ASRMS > 0 {
				usage.asrMS += e.ASRMS
				usage.asrCount++
			}
			if usage.lastModel == "" {
				usage.lastModel = e.ASRModel
			}
		}
		if len(entries) < page {
			return usage, nil
		}
	}
}

// homeUsage is Home's numbers and, under them, the usage card: a square
// a day, a column a week, sized so the page fills the window.
func (a *app) homeUsage(c *ui.Context) {
	t := c.Theme()
	k := tok(t)
	usage := a.hm.usage
	now := time.Now()
	y, m, d := now.Date()
	today := time.Date(y, m, d, 0, 0, 0, 0, now.Location())
	tile := func(key, value, label string) {
		fillCard(c.Key(key)).Gap(px(t, 2)).Grow(1).Basis(0).MinWidth(0).Padding(px(t, sp4)).Children(func() {
			ui.Text(c, value).FontSize(px(t, 28)).FontWeight(600).SingleLine().Ellipsis("…").MinWidth(0)
			ui.Text(c, label).FontSize(px(t, fsCaption)).TextColor(k.Text2).SingleLine()
		})
	}
	month := 0
	for ago := range 30 {
		month += usage.charsByDay[today.AddDate(0, 0, -ago).Format(dayKey)]
	}
	rate, latency := "–", "–"
	if usage.audioMS > 0 {
		rate = itoa(int(float64(usage.audioChars) / (float64(usage.audioMS) / 60000)))
	}
	if usage.asrCount > 0 {
		latency = i18n.T("%.1f s", float64(usage.asrMS)/float64(usage.asrCount)/1000)
	}
	ui.Row(c.Key("home-usage")).Gap(px(t, sp3)).Margin(px(t, sp4), 0, 0, 0).Children(func() {
		tile("usage-today", groupDigits(usage.charsByDay[today.Format(dayKey)]), i18n.T("Today"))
		tile("usage-month", groupDigits(month), i18n.T("30 days"))
		tile("usage-rate", rate, i18n.T("Per minute"))
		tile("usage-latency", latency, i18n.T("Latency"))
	})

	// The squares take the height the window leaves under the tiles, and
	// as many weeks as then fit the card's width.
	const gap = 4
	_, windowHeight := c.Size()
	gridWidth := min(a.pageWidth(c), px(t, pageMaxW)) - px(t, 2*sp6+2*sp4)
	gridHeight := windowHeight - titleBarHeight - px(t, 340)
	cell := min(max((gridHeight-6*px(t, gap))/7, px(t, 10)), px(t, 22))
	weeks := min(usageWeeks, max(4, int((gridWidth+px(t, gap))/(cell+px(t, gap)))))
	peak := 1
	for _, chars := range usage.charsByDay {
		peak = max(peak, chars)
	}
	level := func(chars int) ui.Color { // four steps of ink, by the share of the busiest day
		if chars == 0 {
			return k.Hover
		}
		return k.Text.Alpha(0.2 + 0.2*float32(min(3, 4*chars/(peak+1))))
	}
	first := today.AddDate(0, 0, -int(today.Weekday())-7*(weeks-1))
	shown := today
	card(c.Key("home-heatmap")).Gap(px(t, sp3)).Children(func() {
		sectionTitleText(c, i18n.T("Daily usage"))
		ui.Row(c).Gap(px(t, gap)).Justify(ui.SpaceBetween).Children(func() {
			for week := range weeks {
				ui.Column(c).Gap(px(t, gap)).Children(func() {
					for weekday := range 7 {
						day := first.AddDate(0, 0, 7*week+weekday)
						box := ui.Box(c).Size(cell, cell).Radius(px(t, 4)).Shrink(0)
						if day.After(today) {
							continue
						}
						box.Background(level(usage.charsByDay[day.Format(dayKey)]))
						if box.Hovered() {
							shown = day
							box.Border(1, k.Text)
						}
					}
				})
			}
		})
		// The day under the pointer (today otherwise), and the legend.
		ui.Row(c).Gap(px(t, sp2)).AlignItems(ui.Center).Children(func() {
			ui.Text(c, shown.Format(dayKey)).FontSize(px(t, fsCaption)).TextColor(k.Text2).SingleLine()
			ui.Text(c, groupDigits(usage.charsByDay[shown.Format(dayKey)])+" "+i18n.T("characters")).FontSize(px(t, fsCaption)).FontWeight(600).SingleLine()
			ui.Spacer(c)
			for _, chars := range []int{0, 1, peak / 2, 3 * peak / 4, peak} {
				ui.Box(c).Size(px(t, 10), px(t, 10)).Radius(px(t, 2)).Background(level(chars)).Shrink(0)
			}
		})
	})
}
